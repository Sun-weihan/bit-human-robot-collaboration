#!/usr/bin/env python3
"""双流多尺度模型 + 逐级输入标准化。

发现：整条管线从未对特征做标准化——模型直接吃原始牛顿值和原始小波系数。
小波各级的系数量级差异很大（细节级能量远小于近似级），
输入投影层容易被大尺度通道主导。

做法：用**训练集**统计量对每个（级, 通道）做 z-score 标准化，
再喂给模型。测试集沿用训练集的均值方差（不泄漏）。

用法: python st_moe_norm.py <mode> <seed>
    mode: fixed      —— 原始输入（对照）
          fixed_norm —— 逐级标准化
"""
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/h2official")
import ab_arch as AB  # noqa: E402
import st_moe as SM  # noqa: E402

DEV = AB.DEV
OUT_DIR = "/root/autodl-tmp/h2official/seeds_norm"
os.makedirs(OUT_DIR, exist_ok=True)
DATA = "/root/autodl-tmp/h2data/real_dataset.npz"
STEPS = 6000
BATCH = 64


def build_split():
    d = np.load(DATA)
    f = torch.tensor(d["f"]); t = torch.tensor(d["t"]); y = torch.tensor(d["y"])
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
    print("  前端计算…", flush=True)
    A = SM.swt_approx(f)                      # (N,T,L,6)
    S = SM.scattering_features(f)             # (N,T,6*J)
    return A, S, y, a, b


def normalize(A, S, a, b):
    """用训练集统计量逐（级,通道）标准化。"""
    muA = A[a].mean(dim=(0, 1), keepdim=True)         # (1,1,L,6)
    sdA = A[a].std(dim=(0, 1), keepdim=True) + 1e-6
    muS = S[a].mean(dim=(0, 1), keepdim=True)         # (1,1,6J)
    sdS = S[a].std(dim=(0, 1), keepdim=True) + 1e-6
    print(f"  近似级标准差: " + " ".join(f"{v:.2f}" for v in sdA[0, 0, :, 0].tolist()))
    print(f"  散射通道标准差(前6): " + " ".join(f"{v:.2f}" for v in sdS[0, 0, :6].tolist()))
    return (A - muA) / sdA, (S - muS) / sdS


def main():
    mode, seed = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(seed); np.random.seed(seed)
    A, S, y, a, b = build_split()
    if mode == "fixed_norm":
        A, S = normalize(A, S, a, b)

    def mk(ii):
        def blk(x):
            return x.reshape(x.shape[0], SM.H, SM.S, *x.shape[2:])
        return {"y": y[ii].to(DEV), "apx_f": blk(A[ii]).to(DEV),
                "scat_f": blk(S[ii]).to(DEV)}

    dtr, dte = mk(a), mk(b)
    model = SM.STMoe(mode="fixed").to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  [{mode}] 参数量 {npar:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=AB.LR)
    n = dtr["y"].shape[0]
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), device=DEV)
        y0 = dtr["y"][idx]
        tt = torch.randint(0, AB.T_DIFF, (BATCH,), device=DEV)
        noise = torch.randn_like(y0)
        yt = AB.q_sample(y0, tt, noise)
        out = model(yt, tt, dtr["apx_f"][idx], None, dtr["scat_f"][idx],
                    None, None)
        loss = F.mse_loss(out, noise)
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
                o = model(yt, tt, dte["apx_f"][i:i + 256], None,
                          dte["scat_f"][i:i + 256], None, None)
                acc.append((yt - (1 - ab).sqrt() * o) / ab.sqrt())
            outs.append(torch.cat(acc, 0))
    pred = torch.stack(outs, 0).mean(0)
    m = []
    for d_ in range(3):
        p, q = pred[..., d_].reshape(-1), dte["y"][..., d_].reshape(-1)
        r2 = 1 - ((q - p) ** 2).sum() / (((q - q.mean()) ** 2).sum() + 1e-9)
        m.append(float(r2))
    rec = {"mode": mode, "seed": seed, "params": npar, "r2": m}
    with open(f"{OUT_DIR}/{mode}_{seed}.json", "w") as fh:
        json.dump(rec, fh)
    print(f"  DONE {mode} seed={seed} R²=vx={m[0]:+.3f} vy={m[1]:+.3f} wz={m[2]:+.3f}")


if __name__ == "__main__":
    main()
