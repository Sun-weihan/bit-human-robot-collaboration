#!/usr/bin/env python3
"""ST-MoE：双流专用专家 + 逐维输入条件化路由

动机（来自我们自己的实验）：
  * 静态门控学不动（一直卡在 0.5），因为它没有信息决定该选谁；
    文献里有效的是 **MoE 路由**——门控是输入的函数。
  * vx 主要靠**稳态力**（持续推才走，R²≈0.63）；
    ωz 主要靠**力矩瞬态**（转向是突变，R²≈0.93）；
    vy 本身没信号（标签 std 0.034）。
    三个维度该走不同的专家。

架构：
  稳态流  ：db4 近似系数（5 级）        -> 块注意力(d=64, 2 头)
  瞬态流  ：小波散射特征（4 个尺度）     -> 块注意力(d=64, 2 头)
  路由    ：由输入的谱通量/各频带能量 -> MLP -> 每个输出维度两个专家的权重
  输出    ：按维加权后进入扩散去噪头

参数控制：两流各 64 宽 ≈ 单一 128 宽，总参数量与基线同量级。

用法: python st_moe.py <mode> <seed>
    mode: mono  —— 单一 128 宽基线（对照）
          stmoe —— 双流 + 路由（本方案）
"""
import collections
import json
import os
import sys

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/h2official")
import ab_arch as AB  # noqa: E402
from transformer import TimeEmb, TransformerBlock  # noqa: E402

DEV = AB.DEV
H, S = AB.H, AB.S
LEVELS = 5
J_SCALES = 4
OUT_DIR = "/root/autodl-tmp/h2official/seeds_stmoe"
os.makedirs(OUT_DIR, exist_ok=True)
DATA = "/root/autodl-tmp/h2data/real_dataset.npz"
STEPS = 6000
BATCH = 64


# ---------------- 前端 ----------------
def swt_approx(x, levels=LEVELS):
    arr = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    N, Tn, Dd = arr.shape
    target = 2 ** int(np.ceil(np.log2(Tn)))
    pad = target - Tn
    out = np.zeros((N, Tn, levels, Dd), dtype=np.float32)
    for n in range(N):
        blk = np.concatenate([np.zeros((pad, Dd), dtype=arr.dtype), arr[n]], 0)
        for c in range(Dd):
            co = pywt.swt(blk[:, c], "db4", level=levels, trim_approx=False)
            ca = [cA for cA, cD in reversed(co)]
            out[n, :, :, c] = np.stack(ca, -1)[pad:]
    return torch.from_numpy(out)


def _bandpass(x, scale):
    """简化的带通：x 减去 scale 尺度上的滑动平均，得到该尺度的细节带。"""
    k = max(3, int(2 ** scale))
    k = k + (k % 2 == 0)
    pad = k // 2
    xp = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
    ker = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / k
    ker = ker.repeat(x.shape[-1], 1, 1)
    sm = F.conv1d(xp, ker, groups=x.shape[-1]).transpose(1, 2)[:, :x.shape[1], :]
    return x - sm


def scattering_features(x, scales=J_SCALES):
    """简化小波散射：各尺度带通 -> 取模 -> 低通平均（保留时间分辨）。

    x: (B,T,D) -> (B,T,D*scales)
    与正式散射变换的区别：只做一阶散射、低通用滑动平均代替，
    但保留了"多尺度包络 + 对形变稳定"的核心性质，且零可学习参数。
    """
    feats = []
    for j in range(scales):
        band = _bandpass(x, j)
        env = band.abs()
        # 低通：短窗平均，保留时间结构
        k = 5
        pad = k // 2
        ep = F.pad(env.transpose(1, 2), (pad, pad), mode="replicate")
        ker = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / k
        ker = ker.repeat(x.shape[-1], 1, 1)
        lp = F.conv1d(ep, ker, groups=x.shape[-1]).transpose(1, 2)[:, :x.shape[1], :]
        feats.append(lp)
    return torch.cat(feats, dim=-1)


def input_stats(x):
    """路由输入：谱通量 + 各频带能量 + 变化率（都是低维、可解释的量）。"""
    d = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(1, 2))       # 平均变化率
    nrg = x.abs().mean(dim=(1, 2))                          # 平均能量
    bands = []
    for j in range(J_SCALES):
        b = _bandpass(x, j).abs().mean(dim=(1, 2))
        bands.append(b)
    pk = x.abs().amax(dim=(1, 2))
    return torch.stack([d, nrg, pk] + bands, dim=-1)         # (B, 3+J)


# ---------------- 模型 ----------------
class StreamEncoder(nn.Module):
    def __init__(self, in_ch, d, nl, nh, drop, nlevels):
        super().__init__()
        self.nlevels = nlevels
        self.proj = nn.Linear(in_ch * nlevels, d)
        self.temb = TimeEmb(d)
        self.t_mlp = nn.Sequential(nn.GELU(), nn.Linear(d, d))
        self.q_proj = nn.Linear(3, d)
        self.kv = nn.Sequential(nn.Linear(d, 2 * d), nn.LayerNorm(2 * d),
                                nn.GELU(), nn.Linear(2 * d, d))
        self.blocks = nn.ModuleList([TransformerBlock(d, nh, mlp_ratio=4, drop=drop)
                                     for _ in range(nl)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 3)

    def forward(self, noisy, t, lv):
        """lv: (B,H,S,L,ch) 按级堆叠，或 (B,H,S,ch) 已拼接"""
        if lv.dim() == 5:
            B, Hh, Sq, L, ch = lv.shape
            feat = lv.reshape(B, Hh, Sq, L * ch)
        else:
            feat = lv
        k = self.kv(self.proj(feat))                         # (B,H,S,d)
        # 复用块注意力：把 (B,H,S,d) 视作 (B,H,S,1,d)
        k5 = k.unsqueeze(3)
        v5 = k5
        q = self.q_proj(noisy)
        h = q + self.t_mlp(self.temb(t)).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, k5, v5)
        self.last_h = self.norm(h)          # 供辅助任务读取
        return self.out(self.last_h)


class STMoe(nn.Module):
    def __init__(self, mode="stmoe", d=128, nl=4, nh=4, drop=0.5):
        super().__init__()
        self.mode = mode
        if mode == "mono":
            self.mono = StreamEncoder(6, d, nl, nh, drop, LEVELS)
        elif mode == "mono64":
            # 消融：半宽单流，用于隔离"参数减半"本身的影响
            self.mono = StreamEncoder(6, d // 2, nl, nh, drop, LEVELS)
        else:
            self.sus = StreamEncoder(6, d // 2, nl, nh, drop, LEVELS)
            # 散射特征已把各尺度拼进通道维，所以级数按 1 算
            self.tra = StreamEncoder(6 * J_SCALES, d // 2, nl, nh, drop, 1)
            if mode == "stmoe":
                self.router = nn.Sequential(
                    nn.Linear(3 + J_SCALES, 32), nn.GELU(), nn.Linear(32, 2 * 3))

    def forward(self, noisy, t, apx_f, apx_t, scat_f, scat_t, stats):
        if self.mode in ("mono", "mono64"):
            return self.mono(noisy, t, apx_f)
        o_s = self.sus(noisy, t, apx_f)                      # (B,H,3)
        o_t = self.tra(noisy, t, scat_f)                     # (B,H,3)
        if self.mode == "fixed":
            # 消融：双流固定 50/50，无路由参数
            return 0.5 * (o_s + o_t)
        # 路由输出 2 专家 × 3 输出维度 = 6 个数，按 (B,3,2) 解读
        w = torch.softmax(self.router(stats).view(-1, 3, 2), dim=-1)   # (B,3,2)
        w = w.unsqueeze(1).expand(-1, o_s.shape[1], -1, -1)            # (B,H,3,2)
        out = w[..., 0] * o_s + w[..., 1] * o_t                        # 逐维加权
        return out


# ---------------- 数据 ----------------
def direction_of(name):
    return str(name).split("__")[-1].split("_")[0]


def load_split():
    d = np.load(DATA)
    f = torch.tensor(d["f"]); t = torch.tensor(d["t"]); y = torch.tensor(d["y"])
    sizes = d["traj_sizes"]; names = d["traj_names"]
    groups = collections.defaultdict(list)
    for i, n in enumerate(names):
        groups[direction_of(n)].append(i)
    tr_t, te_t = [], []
    for g, ids in sorted(groups.items()):
        ids = sorted(ids)
        k = max(1, int(round(len(ids) * 0.15)))
        te_t += ids[-k:]; tr_t += ids[:-k]

    def sidx(tset):
        out, off = [], 0
        for i, s in enumerate(sizes):
            if i in tset:
                out.extend(range(off, off + int(s)))
            off += int(s)
        return np.array(out)

    a, b = sidx(set(tr_t)), sidx(set(te_t))
    print(f"  轨迹 {len(tr_t)}/{len(te_t)}；样本 {len(a)}/{len(b)}")
    print("  前端：SWT 近似 + 散射特征…", flush=True)
    A_f, A_t = swt_approx(f), swt_approx(t)
    S_f, S_t = scattering_features(f), scattering_features(t)
    ST = input_stats(f)

    def split(x, ii):
        return x[ii].to(DEV)

    def mk(ii):
        # 每个 horizon 块内再切成 S 个 token
        def blk(x):
            Bq = x.shape[0]
            return x.reshape(Bq, H, S, *x.shape[2:])
        return {"y": y[ii].to(DEV),
                "apx_f": blk(A_f[ii]).to(DEV), "scat_f": blk(S_f[ii]).to(DEV),
                "scat_t": blk(S_t[ii]).to(DEV), "stats": ST[ii].to(DEV)}

    return mk(a), mk(b)


def train(model, dat):
    opt = torch.optim.Adam(model.parameters(), lr=AB.LR)
    model.train()
    n = dat["y"].shape[0]
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), device=DEV)
        y0 = dat["y"][idx]
        tt = torch.randint(0, AB.T_DIFF, (BATCH,), device=DEV)
        noise = torch.randn_like(y0)
        yt = AB.q_sample(y0, tt, noise)
        pred = model(yt, tt, dat["apx_f"][idx], None, dat["scat_f"][idx],
                     dat["scat_t"][idx], dat["stats"][idx])
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0:
            print(f"    step {step} loss {loss.item():.4f}", flush=True)


@torch.no_grad()
def ev(model, dat, n_seed=8, t_eval=20, bs=256):
    model.eval()
    ab = AB.ABAR[t_eval]
    outs = []
    for _ in range(n_seed):
        acc = []
        for i in range(0, dat["y"].shape[0], bs):
            y0 = dat["y"][i:i + bs]
            noise = torch.randn_like(y0)
            yt = ab.sqrt() * y0 + (1 - ab).sqrt() * noise
            tt = torch.full((len(y0),), t_eval, dtype=torch.long, device=DEV)
            eps = model(yt, tt, dat["apx_f"][i:i + bs], None, dat["scat_f"][i:i + bs],
                        dat["scat_t"][i:i + bs], dat["stats"][i:i + bs])
            acc.append((yt - (1 - ab).sqrt() * eps) / ab.sqrt())
        outs.append(torch.cat(acc, 0))
    return torch.stack(outs, 0).mean(0)


def metrics(pred, true):
    out = []
    for d in range(true.shape[-1]):
        p, q = pred[..., d].reshape(-1), true[..., d].reshape(-1)
        if q.std() < 1e-6:
            out.append((float("nan"), float("nan"))); continue
        r2 = 1 - ((q - p) ** 2).sum() / (((q - q.mean()) ** 2).sum() + 1e-9)
        out.append((float(r2), float(np.corrcoef(p.cpu().numpy(), q.cpu().numpy())[0, 1])))
    return out


def main():
    mode, seed = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(seed); np.random.seed(seed)
    dtr, dte = load_split()
    model = STMoe(mode=mode).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  [{mode}] 参数量 {npar:,}", flush=True)
    train(model, dtr)
    m = metrics(ev(model, dte), dte["y"])
    rec = {"mode": mode, "seed": seed, "params": npar,
           "r2": [float(x[0]) for x in m], "corr": [float(x[1]) for x in m],
           "avg_corr": float(np.mean([x[1] for x in m]))}
    with open(f"{OUT_DIR}/{mode}_{seed}.json", "w") as fh:
        json.dump(rec, fh)
    print(f"  DONE {mode} seed={seed} R²=" +
          " ".join(f"{['vx','vy','wz'][i]}={rec['r2'][i]:+.3f}" for i in range(3)))


if __name__ == "__main__":
    main()
