#!/usr/bin/env python3
"""SEDA：共享编码器的双注意力（参数匹配版）

问题：之前的混合模型 2.76M 参数，官方只有 0.93M（3 倍），这个对比不公平。

本设计：**不新增特征提取**。块分支已经算出每块 S 个 token 的特征
    k = kv_proj([proj_f(力) ‖ proj_t(力矩)])   -> (B, H, S, L, d)
token 路径直接在这**同一份特征**上再做一次块内注意力，以残差形式叠加：
    h = h + σ(g)·TokenAttend(h, k)
只多两个线性层（tok_q, tok_out，约 33k 参数），总参数量 ≈ 961k，
与官方 928k 相差 3.5%，属同一预算。

这样比较才干净：**同等参数下，多一条注意力路径是否更好**。

用法: python seda.py <mode> <seed>
    mode: seda  | base（= 官方结构，同代码路径，作为对照）
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/h2official")
import ab_arch as AB  # noqa: E402
from transformer import TimeEmb, TransformerBlock  # noqa: E402

DEV = AB.DEV
H, S = AB.H, AB.S
DM, NB, LV, NH, DROP = 128, 4, 5, 4, 0.5
OUT_DIR = "/root/autodl-tmp/h2official/seeds_seda"
os.makedirs(OUT_DIR, exist_ok=True)


class SEDA(nn.Module):
    def __init__(self, mode="seda", d=DM, nl=NB, nh=NH, drop=DROP):
        super().__init__()
        self.mode = mode
        self.d = d
        self.nh = nh
        self.dh = d // nh
        # ---- 与官方一致的编码器 ----
        self.q_proj = nn.Linear(3, d)
        self.kv_proj_f = nn.Linear(6, d)
        self.kv_proj_t = nn.Linear(6, d)
        self.kv_proj = nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.LayerNorm(2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.temb = TimeEmb(d)
        self.t_mlp = nn.Sequential(nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([TransformerBlock(d, nh, mlp_ratio=4, drop=drop)
                                     for _ in range(nl)])
        self.pre_out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, 3)

        # ---- 新增：块内 token 注意力（只两个线性层 + 一个逐维门控）----
        if mode == "seda":
            self.tok_q = nn.Linear(d, d)
            self.tok_out = nn.Linear(d, d)
            self.tok_gate = nn.Parameter(torch.zeros(3))

    def _token_attend(self, h, k, v):
        """在共享特征上做块内 token 注意力，按小波级熵加权融合。

        h: (B,H,d) 查询; k,v: (B,H,S,L,d)
        """
        Bq, Hq, Sq, Lv, d = k.shape
        nh, dh = self.nh, self.dh
        q = self.tok_q(h).view(Bq, Hq, nh, dh)
        ctxs, ents = [], []
        for lv in range(Lv):
            kk = k[..., lv, :].view(Bq, Hq, Sq, nh, dh).transpose(2, 3)   # (B,H,nh,S,dh)
            vv = v[..., lv, :].view(Bq, Hq, Sq, nh, dh).transpose(2, 3)
            sc = torch.einsum("bhnd,bhnsd->bhns", q, kk) * (dh ** -0.5)
            att = torch.softmax(sc, dim=-1)                                # (B,H,nh,S)
            ent = -(att * (att + 1e-8).log()).sum(-1).mean(-1)             # (B,H) 级内熵
            ctx = torch.einsum("bhns,bhnsd->bhnd", att, vv).reshape(Bq, Hq, d)
            ctxs.append(ctx)
            ents.append(ent)
        C = torch.stack(ctxs, 0)            # (L,B,H,d)
        E = torch.stack(ents, 0)            # (L,B,H)
        w = torch.softmax(-E, dim=0)        # 熵低者权重大
        return self.tok_out((w[..., None] * C).sum(0))

    def forward(self, noisy, t, apx_f, apx_t):
        af = torch.stack(torch.split(apx_f, S, dim=1), dim=1)     # (B,H,S,L,6)
        at = torch.stack(torch.split(apx_t, S, dim=1), dim=1)
        q = self.q_proj(noisy)
        f_k = self.kv_proj_f(af)
        v_k = self.kv_proj_t(at)
        k = self.kv_proj(torch.cat([f_k, v_k], dim=-1))
        v = k
        h = q + self.t_mlp(self.temb(t)).unsqueeze(1)

        for blk in self.blocks:
            h = blk(h, k, v)
            if self.mode == "seda":
                g = torch.sigmoid(self.tok_gate).view(1, 1, -1)    # (1,1,3)
                # 残差叠加：门控按输出维度作用在**投影前的 d 维**上是不可行的，
                # 因此这里用标量化的平均门控缩放整条 token 路径，逐维门控在最终输出上施加
                h = h + g.mean() * self._token_attend(h, k, v)

        out = self.out_proj(self.pre_out_norm(h))
        if self.mode == "seda":
            # 逐维门控：两路输出按维度加权（token 路径的贡献已含在 h 里，
            # 这里再给一个逐维的温度，等价于让网络自己决定每个输出维度
            # 对 token 信息的敏感度）
            out = out * (1.0 + 0.5 * torch.sigmoid(self.tok_gate).view(1, 1, -1))
        return out


def get_data():
    F_, Tau, Y = AB.gen_data(AB.N_TRAIN + AB.N_TEST, seed=AB.SEED)
    F_, Tau, Y = (torch.tensor(a) for a in (F_, Tau, Y))
    tr, te = slice(0, AB.N_TRAIN), slice(AB.N_TRAIN, AB.N_TRAIN + AB.N_TEST)
    f_pair = torch.cat([F_[..., :3], Tau[..., :3]], -1)
    t_pair = torch.cat([F_[..., 3:], Tau[..., 3:]], -1)
    f_off = AB.swt_official(f_pair, 5)
    t_off = AB.swt_official(t_pair, 5)

    def mk(sl):
        return {"y": Y[sl].to(DEV), "f": f_off[sl].to(DEV), "t": t_off[sl].to(DEV)}

    return mk(tr), mk(te)


def train(model, dat):
    opt = torch.optim.Adam(model.parameters(), lr=AB.LR)
    model.train()
    n = dat["y"].shape[0]
    for _ in range(AB.STEPS):
        idx = torch.randint(0, n, (AB.BATCH,), device=DEV)
        y0 = dat["y"][idx]
        t = torch.randint(0, AB.T_DIFF, (AB.BATCH,), device=DEV)
        noise = torch.randn_like(y0)
        yt = AB.q_sample(y0, t, noise)
        pred = model(yt, t, dat["f"][idx], dat["t"][idx])
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def ev(model, dat, n_seed=8, t_eval=20):
    model.eval()
    Bq = dat["y"].shape[0]
    ab = AB.ABAR[t_eval]
    outs = []
    for _ in range(n_seed):
        noise = torch.randn_like(dat["y"])
        yt = ab.sqrt() * dat["y"] + (1 - ab).sqrt() * noise
        t = torch.full((Bq,), t_eval, dtype=torch.long, device=DEV)
        eps = model(yt, t, dat["f"], dat["t"])
        outs.append((yt - (1 - ab).sqrt() * eps) / ab.sqrt())
    return torch.stack(outs, 0).mean(0)


def main():
    mode, seed = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(seed); np.random.seed(seed)
    dtr, dte = get_data()
    model = SEDA(mode=mode).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    train(model, dtr)
    st = AB.r2_corr(ev(model, dte), dte["y"])
    rec = {"mode": mode, "seed": seed, "params": npar,
           "r2": [float(r) for r, _ in st], "corr": [float(c) for _, c in st],
           "avg_corr": float(np.mean([c for _, c in st]))}
    if mode == "seda":
        rec["gate"] = torch.sigmoid(model.tok_gate).detach().cpu().numpy().tolist()
    with open(f"{OUT_DIR}/{mode}_{seed}.json", "w") as f:
        json.dump(rec, f)
    print(f"  DONE {mode} seed={seed} params={npar:,} R²={['%.3f' % v for v in rec['r2']]}")


if __name__ == "__main__":
    main()
