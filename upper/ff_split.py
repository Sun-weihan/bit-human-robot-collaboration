#!/usr/bin/env python3
"""纯前馈的维度分流架构（可闭环部署）。

出发点：
    abs 模型（只吃力）是唯一在闭环里稳定的——因为它**没有反馈回路**。
    而闭环不稳定全部来自"吃自身速度历史"（split 失控 156%，DAgger 欠调 49%）。

    所以正确方向不是去修反馈，而是**在纯前馈前提下把架构做好**。

    "维度特异分流"这个想法本来就不需要速度输入：
        平移支路：低通力（多尺度池化）  → (vx, vy)
        转向支路：力的时间导数          → wz
    依据来自实测：wz 用导数 0.46，用力本身只有 0.22；vx 用低通力最好。

对照：
    abs       : 力的原始窗口 → 单个 MLP → 3 维        （当前唯一可部署的）
    ff_concat : [低通力, 导数] 拼接 → 单个 MLP → 3 维  （同信息，不分流）
    ff_split  : 分流（平移←低通 / 转向←导数）          ← 本方案

用法: python ff_split.py <mode> <seed>
"""
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "/root/autodl-tmp/h2data/real_dataset.npz"
OUT_DIR = os.environ.get("FF_DIR", "/root/autodl-tmp/h2official/seeds_ff")
os.makedirs(OUT_DIR, exist_ok=True)
EPOCHS = 120
BS = 256
POOLS = (4, 16, 64)


def lowpass_feat(w):
    out = []
    for p in POOLS:
        n = len(w) // p
        out.append(w[:n * p].reshape(n, p, -1).mean(1).reshape(-1))
    return np.concatenate(out)


def deriv_feat(w):
    return np.diff(w, axis=0).reshape(-1)


def split_trajs(names, sizes):
    g = collections.defaultdict(list)
    for i, n in enumerate(names):
        g[str(n).split("__")[-1].split("_")[0]].append(i)
    tr, te = [], []
    for k, ids in sorted(g.items()):
        ids = sorted(ids); kk = max(1, int(round(len(ids) * 0.15)))
        te += ids[-kk:]; tr += ids[:-kk]
    return set(tr), set(te)


def build(mode):
    d = np.load(DATA)
    f, y, sizes, names = d["f"], d["y"], d["traj_sizes"], d["traj_names"]
    tr_t, _ = split_trajs(names, sizes)
    Y = np.asarray(y, dtype=np.float32)
    Xtr, Xte, Ytr, Yte = [], [], [], []
    off = 0
    for i, s in enumerate(sizes):
        s = int(s); is_tr = i in tr_t
        for j in range(s):
            gi = off + j
            w = np.asarray(f[gi], dtype=np.float32)
            lp = lowpass_feat(w)
            dv = deriv_feat(w)
            if mode == "abs":
                x = w.reshape(-1)
            elif mode == "ff_concat":
                x = np.concatenate([lp, dv])
            else:                     # ff_split
                x = (lp, dv)
            tgt = Y[gi, 0, :]
            (Xtr if is_tr else Xte).append(x)
            (Ytr if is_tr else Yte).append(tgt)
        off += s

    if mode == "ff_split":
        def pack(X):
            return (np.stack([a for a, b in X]).astype(np.float32),
                    np.stack([b for a, b in X]).astype(np.float32))
        return pack(Xtr), np.stack(Ytr).astype(np.float32), \
               pack(Xte), np.stack(Yte).astype(np.float32)
    return (np.stack(Xtr).astype(np.float32), np.stack(Ytr).astype(np.float32),
            np.stack(Xte).astype(np.float32), np.stack(Yte).astype(np.float32))


def mlp(d_in, d_out, widths=(512, 512)):
    L, d = [], d_in
    for w in widths:
        L += [nn.Linear(d, w), nn.LayerNorm(w), nn.GELU(), nn.Dropout(0.2)]
        d = w
    L += [nn.Linear(d, d_out)]
    return nn.Sequential(*L)


class Flat(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = mlp(d_in, 3)

    def forward(self, x):
        return self.net(x)


class FFSplit(nn.Module):
    """平移支路吃低通，转向支路吃导数。纯前馈，无速度输入。"""

    def __init__(self, d_lp, d_dv):
        super().__init__()
        self.trans = mlp(d_lp, 2, widths=(512, 256))
        self.rot = mlp(d_dv, 1, widths=(512, 256))

    def forward(self, lp, dv):
        return torch.cat([self.trans(lp), self.rot(dv)], -1)


def r2(pred, true):
    out = []
    for k in range(3):
        p, q = pred[:, k], true[:, k]
        out.append(float(1 - ((q - p) ** 2).sum() / (((q - q.mean()) ** 2).sum() + 1e-9)))
    return out


def main():
    mode, seed = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Xte, Yte = build(mode)
    split = mode == "ff_split"

    if split:
        stats = [(Xtr[k].mean(0), Xtr[k].std(0) + 1e-6) for k in range(2)]
        A = tuple(torch.tensor((Xtr[k] - stats[k][0]) / stats[k][1], device=DEV) for k in range(2))
        B = tuple(torch.tensor((Xte[k] - stats[k][0]) / stats[k][1], device=DEV) for k in range(2))
        model = FFSplit(Xtr[0].shape[1], Xtr[1].shape[1]).to(DEV)
        def fwd(x):
            return model(*x)
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        stats = (mu, sd)
        A = torch.tensor((Xtr - mu) / sd, device=DEV)
        B = torch.tensor((Xte - mu) / sd, device=DEV)
        model = Flat(A.shape[1]).to(DEV)
        def fwd(x):
            return model(x)

    Yt = torch.tensor(Ytr, device=DEV)
    npar = sum(p.numel() for p in model.parameters())
    n = len(Yt); nv = max(1, n // 10)
    perm = np.random.default_rng(seed).permutation(n)
    vi, ti = perm[:nv], perm[nv:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best, bs = 1e9, None
    for ep in range(EPOCHS):
        model.train()
        idx = torch.tensor(ti[torch.randperm(len(ti))], device=DEV)
        for k in range(0, len(idx), BS):
            j = idx[k:k + BS]
            xin = tuple(t[j] for t in A) if split else A[j]
            loss = torch.nn.functional.mse_loss(fwd(xin), Yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            xv = tuple(t[vi] for t in A) if split else A[vi]
            vl = torch.nn.functional.mse_loss(fwd(xv), Yt[vi]).item()
        if vl < best:
            best = vl; bs = {k: t.clone() for k, t in model.state_dict().items()}
    if bs:
        model.load_state_dict(bs)

    model.eval()
    with torch.no_grad():
        pred = torch.cat([fwd(tuple(t[i:i+512] for t in B) if split else B[i:i+512])
                          for i in range(0, len(Yte), 512)], 0).cpu().numpy()
    m = r2(pred, Yte)
    rec = {"mode": mode, "seed": seed, "params": npar, "r2": m}
    json.dump(rec, open(f"{OUT_DIR}/{mode}_{seed}.json", "w"))
    torch.save({"state": model.state_dict(), "mode": mode, "seed": seed,
                "params": npar, "r2": m,
                "stats": [(s[0].tolist(), s[1].tolist()) for s in stats]
                if split else (stats[0].tolist(), stats[1].tolist()),
                "pools": POOLS}, f"{OUT_DIR}/{mode}_{seed}.pt")
    print(f"  DONE {mode:<10} seed={seed} R²=vx={m[0]:+.3f} vy={m[1]:+.3f} wz={m[2]:+.3f}",
          flush=True)


if __name__ == "__main__":
    main()
