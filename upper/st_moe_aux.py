#!/usr/bin/env python3
"""ST-MoE + 辅助力预测（自监督辅助任务，文献支撑）。

动机：自监督辅助任务能提升主回归任务（Self-Correction Transformer, IET ITS 2025）。
这里不需要额外标注——数据里已有"目标窗口内的平均力"，
让编码器在预测速度的同时**也预测接下来这段窗口的力**，
逼它学习力的动态，而不是只拟合速度。

    L = MSE(ε_pred, ε) + λ·MSE(aux_pred, f_next),  λ = 0.1

与 stmoe 的差别：只多一个 Linear(64,6) 辅助头，参数量 +390。

用法: python st_moe_aux.py <mode> <seed>
    mode: stmoe_aux  —— 加辅助任务
          stmoe      —— 对照（同代码路径，辅助权重置 0）
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
import st_moe as SM  # noqa: E402

DEV = AB.DEV
OUT_DIR = "/root/autodl-tmp/h2official/seeds_aux"
os.makedirs(OUT_DIR, exist_ok=True)
DATA = "/root/autodl-tmp/h2data/real_dataset.npz"
STEPS = 6000
BATCH = 64
AUX_W = 0.1


class AuxModel(nn.Module):
    def __init__(self, use_aux=True):
        super().__init__()
        self.core = SM.STMoe(mode="stmoe")
        self.use_aux = use_aux
        self.aux = nn.Linear(64, 6)          # 稳态流隐状态 -> 6 轴力

    def forward(self, noisy, t, apx_f, scat_f, stats, f_next=None):
        o_s = self.core.sus(noisy, t, apx_f)
        h_s = self.core.sus.last_h                        # (B,H,64)
        o_t = self.core.tra(noisy, t, scat_f)
        w = torch.softmax(self.core.router(stats).view(-1, 3, 2), dim=-1)
        w = w.unsqueeze(1).expand(-1, o_s.shape[1], -1, -1)
        out = w[..., 0] * o_s + w[..., 1] * o_t
        aux_pred = self.aux(h_s.mean(1))                  # (B,6)
        return out, aux_pred


def load_split():
    d = np.load(DATA)
    f = torch.tensor(d["f"]); t = torch.tensor(d["t"]); y = torch.tensor(d["y"])
    fn = torch.tensor(d["f_next"])
    sizes = d["traj_sizes"]; names = d["traj_names"]
    groups = {}
    for i, n in enumerate(names):
        k = str(n).split("__")[-1].split("_")[0]
        groups.setdefault(k, []).append(i)
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
    print(f"  轨迹 {len(tr_t)}/{len(te_t)}；样本 {len(a)}/{len(b)}", flush=True)
    print("  前端：SWT 近似 + 散射特征…", flush=True)
    A_f = SM.swt_approx(f)
    S_f = SM.scattering_features(f)
    ST = SM.input_stats(f)

    def mk(ii):
        def blk(x):
            return x.reshape(x.shape[0], SM.H, SM.S, *x.shape[2:])
        return {"y": y[ii].to(DEV), "apx_f": blk(A_f[ii]).to(DEV),
                "scat_f": blk(S_f[ii]).to(DEV), "stats": ST[ii].to(DEV),
                "fn": fn[ii].to(DEV)}

    return mk(a), mk(b)


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
    use_aux = mode == "stmoe_aux"
    torch.manual_seed(seed); np.random.seed(seed)
    dtr, dte = load_split()
    model = AuxModel(use_aux).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  [{mode}] 参数量 {npar:,}  辅助权重 {AUX_W if use_aux else 0}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=AB.LR)
    n = dtr["y"].shape[0]
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), device=DEV)
        y0 = dtr["y"][idx]
        tt = torch.randint(0, AB.T_DIFF, (BATCH,), device=DEV)
        noise = torch.randn_like(y0)
        yt = AB.q_sample(y0, tt, noise)
        out, aux = model(yt, tt, dtr["apx_f"][idx], dtr["scat_f"][idx],
                         dtr["stats"][idx], dtr["fn"][idx])
        loss = F.mse_loss(out, noise)
        if use_aux:
            loss = loss + AUX_W * F.mse_loss(aux, dtr["fn"][idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    ab = AB.ABAR[20]
    outs = []
    with torch.no_grad():
        for _ in range(8):
            acc = []
            for i in range(0, dte["y"].shape[0], 256):
                y0 = dte["y"][i:i + 256]
                noise = torch.randn_like(y0)
                yt = ab.sqrt() * y0 + (1 - ab).sqrt() * noise
                tt = torch.full((len(y0),), 20, dtype=torch.long, device=DEV)
                o, _ = model(yt, tt, dte["apx_f"][i:i + 256], dte["scat_f"][i:i + 256],
                             dte["stats"][i:i + 256])
                acc.append((yt - (1 - ab).sqrt() * o) / ab.sqrt())
            outs.append(torch.cat(acc, 0))
    pred = torch.stack(outs, 0).mean(0)
    m = metrics(pred, dte["y"])
    rec = {"mode": mode, "seed": seed, "params": npar,
           "r2": [float(x[0]) for x in m], "corr": [float(x[1]) for x in m]}
    with open(f"{OUT_DIR}/{mode}_{seed}.json", "w") as fh:
        json.dump(rec, fh)
    print(f"  DONE {mode} seed={seed} R²=" +
          " ".join(f"{['vx','vy','wz'][i]}={rec['r2'][i]:+.3f}" for i in range(3)))


if __name__ == "__main__":
    main()
