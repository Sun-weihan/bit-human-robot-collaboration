#!/usr/bin/env python3
"""容量与训练量扫描。

前几轮发现：
  * 双流等权融合（2×64，455k）vx 0.706，显著优于单流 898k（0.535）
  * 逐级标准化有害（瞬态特征被放大成噪声）
  * 辅助力预测有害

剩下两个最基础的问题没查：
  A) 训练量是否足够？当前 6000 步 × batch 64 ≈ 38 万样本，数据有 3 万 ——  约 13 个 epoch
  B) 容量是否不足？双流各 64 宽可能偏小

用法: python st_moe_scale.py <mode> <seed>
    mode: fixed        —— 现状（2×64，6000 步）
          fixed_long   —— 2×64，12000 步
          fixed_wide   —— 2×128，6000 步
"""
import collections
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
OUT_DIR = "/root/autodl-tmp/h2official/seeds_scale"
os.makedirs(OUT_DIR, exist_ok=True)
DATA = "/root/autodl-tmp/h2data/real_dataset.npz"
BATCH = 64
CFG = {
    "fixed":      dict(width=64, steps=6000),
    "fixed_long": dict(width=64, steps=12000),
    "fixed_wide": dict(width=128, steps=6000),
}


class FixedWidth(nn.Module):
    """双流等权融合，宽度可配置。"""

    def __init__(self, width):
        super().__init__()
        self.sus = SM.StreamEncoder(6, width, 4, max(2, width // 32), 0.5, SM.LEVELS)
        self.tra = SM.StreamEncoder(6 * SM.J_SCALES, width, 4, max(2, width // 32), 0.5, 1)

    def forward(self, noisy, t, apx_f, scat_f):
        return 0.5 * (self.sus(noisy, t, apx_f) + self.tra(noisy, t, scat_f))


def build_split():
    d = np.load(DATA)
    f = torch.tensor(d["f"]); y = torch.tensor(d["y"])
    sizes = d["traj_sizes"]; names = d["traj_names"]
    groups = collections.defaultdict(list)
    for i, n in enumerate(names):
        groups[str(n).split("__")[-1].split("_")[0]].append(i)
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
    A = SM.swt_approx(f)
    S = SM.scattering_features(f)

    def mk(ii):
        def blk(x):
            return x.reshape(x.shape[0], SM.H, SM.S, *x.shape[2:])
        return {"y": y[ii].to(DEV), "apx_f": blk(A[ii]).to(DEV), "scat_f": blk(S[ii]).to(DEV)}

    return mk(a), mk(b)


def main():
    mode, seed = sys.argv[1], int(sys.argv[2])
    cfg = CFG[mode]
    torch.manual_seed(seed); np.random.seed(seed)
    dtr, dte = build_split()
    model = FixedWidth(cfg["width"]).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  [{mode}] 宽 {cfg['width']}×2  步数 {cfg['steps']}  参数量 {npar:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=AB.LR)
    n = dtr["y"].shape[0]
    model.train()
    for step in range(cfg["steps"]):
        idx = torch.randint(0, n, (BATCH,), device=DEV)
        y0 = dtr["y"][idx]
        tt = torch.randint(0, AB.T_DIFF, (BATCH,), device=DEV)
        noise = torch.randn_like(y0)
        yt = AB.q_sample(y0, tt, noise)
        loss = F.mse_loss(model(yt, tt, dtr["apx_f"][idx], dtr["scat_f"][idx]), noise)
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
                o = model(yt, tt, dte["apx_f"][i:i + 256], dte["scat_f"][i:i + 256])
                acc.append((yt - (1 - ab).sqrt() * o) / ab.sqrt())
            outs.append(torch.cat(acc, 0))
    pred = torch.stack(outs, 0).mean(0)
    m = []
    for d_ in range(3):
        p, q = pred[..., d_].reshape(-1), dte["y"][..., d_].reshape(-1)
        m.append(float(1 - ((q - p) ** 2).sum() / (((q - q.mean()) ** 2).sum() + 1e-9)))
    rec = {"mode": mode, "seed": seed, "params": npar, "r2": m}
    with open(f"{OUT_DIR}/{mode}_{seed}.json", "w") as fh:
        json.dump(rec, fh)
    print(f"  DONE {mode} seed={seed} R²=vx={m[0]:+.3f} vy={m[1]:+.3f} wz={m[2]:+.3f}")


if __name__ == "__main__":
    main()
