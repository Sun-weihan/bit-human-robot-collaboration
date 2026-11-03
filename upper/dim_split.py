#!/usr/bin/env python3
"""维度特异的分流通路（最后一个架构尝试）。

依据（实测）：
    vx: 低通力最好（高 Δv 帧 R²=0.395）
    vy: 力+导数 0.263
    wz: **导数** 0.460，而力本身只有 0.219
    → 平移靠低频、转向靠高频。分流依据来自测量，不是拍脑袋。

三种结构对照：
  concat        : [力, 导数, 速度历史] 全部拼接 → 一个 MLP → 3 维（基线）
  split_correct : 平移支路(低通力) → vx,vy ；转向支路(导数) → wz      ← 本方案
  split_wrong   : 平移支路(导数) → vx,vy ；转向支路(低通力) → wz      ← 对照
                  （若 split_wrong 不比 concat 好，说明增益来自"正确的分配"，
                    而不是"分成两路"这个动作本身）

用法: python dim_split.py <mode> <seed>
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
OUT_DIR = os.environ.get("SPL_DIR", "/root/autodl-tmp/h2official/seeds_split")
os.makedirs(OUT_DIR, exist_ok=True)
CTX = 3
EPOCHS = 120
BS = 256
POOLS = (4, 16, 64)          # 低通的多尺度池化


def lowpass_feat(w):
    """多尺度平均池化：只保留低频成分"""
    out = []
    for p in POOLS:
        n = len(w) // p
        out.append(w[:n * p].reshape(n, p, -1).mean(1).reshape(-1))
    return np.concatenate(out)


def deriv_feat(w):
    """一阶差分（高频）"""
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
            vh = Y[[max(gi - k, off) for k in range(CTX, 0, -1)], 0, :].reshape(-1)
            if mode == "concat":
                x = np.concatenate([lp, dv, vh])
            elif mode == "split_correct":
                x = (lp, dv, vh)          # 元组：交给模型按支路拆
            else:                          # split_wrong
                x = (lp, dv, vh)
            tgt = Y[gi, 0, :]
            if is_tr:
                Xtr.append(x); Ytr.append(tgt)
            else:
                Xte.append(x); Yte.append(tgt)
        off += s

    def pack(X, mode):
        if mode == "concat":
            return np.stack(X).astype(np.float32)
        lp = np.stack([a for a, b, c in X]).astype(np.float32)
        dv = np.stack([b for a, b, c in X]).astype(np.float32)
        vh = np.stack([c for a, b, c in X]).astype(np.float32)
        return (lp, dv, vh)

    return pack(Xtr, mode), np.stack(Ytr).astype(np.float32), \
           pack(Xte, mode), np.stack(Yte).astype(np.float32)


def mlp(d_in, d_out, widths=(512, 512)):
    L, d = [], d_in
    for w in widths:
        L += [nn.Linear(d, w), nn.LayerNorm(w), nn.GELU(), nn.Dropout(0.2)]
        d = w
    L += [nn.Linear(d, d_out)]
    return nn.Sequential(*L)


class Concat(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = mlp(d_in, 3)

    def forward(self, lp, dv, vh):
        return self.net(torch.cat([lp, dv, vh], -1))


class DimSplit(nn.Module):
    """平移支路吃低通、转向支路吃导数。"""

    def __init__(self, d_lp, d_dv, d_vh, correct=True):
        super().__init__()
        self.correct = correct
        self.lp_proj = nn.Linear(d_lp, 256)
        self.dv_proj = nn.Linear(d_dv, 256)
        self.vh_proj = nn.Linear(d_vh, 32)
        # 平移支路：低通 + 速度历史
        self.trans = mlp(256 + 32, 2)
        # 转向支路：导数 + 速度历史
        self.rot = mlp(256 + 32, 1)
        self.act = nn.GELU()

    def forward(self, lp, dv, vh):
        a = self.act(self.lp_proj(lp))
        b = self.act(self.dv_proj(dv))
        c = self.act(self.vh_proj(vh))
        if self.correct:
            v_xy = self.trans(torch.cat([a, c], -1))
            w_z = self.rot(torch.cat([b, c], -1))
        else:
            v_xy = self.trans(torch.cat([b, c], -1))
            w_z = self.rot(torch.cat([a, c], -1))
        return torch.cat([v_xy, w_z], -1)


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

    if mode == "concat":
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        A = torch.tensor((Xtr - mu) / sd, device=DEV)
        B = torch.tensor((Xte - mu) / sd, device=DEV)
        model = Concat(A.shape[1]).to(DEV)
        def fwd(X):
            return model(X, X, X)
    else:
        parts = []
        for k in range(3):
            mu = Xtr[k].mean(0); sd = Xtr[k].std(0) + 1e-6
            parts.append((mu, sd))
        A = tuple(torch.tensor((Xtr[k] - parts[k][0]) / parts[k][1], device=DEV) for k in range(3))
        B = tuple(torch.tensor((Xte[k] - parts[k][0]) / parts[k][1], device=DEV) for k in range(3))
        model = DimSplit(Xtr[0].shape[1], Xtr[1].shape[1], Xtr[2].shape[1],
                         correct=(mode == "split_correct")).to(DEV)
        def fwd(X):
            return model(*X)

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
            loss = torch.nn.functional.mse_loss(fwd((A[0][j], A[1][j], A[2][j])
                                                   if isinstance(A, tuple) else A[j]), Yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            xs = tuple(t[vi] for t in A) if isinstance(A, tuple) else A[vi]
            vl = torch.nn.functional.mse_loss(fwd(xs), Yt[vi]).item()
        if vl < best:
            best = vl; bs = {k: t.clone() for k, t in model.state_dict().items()}
    if bs:
        model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([fwd(tuple(t[i:i+512] for t in B) if isinstance(B, tuple) else B[i:i+512])
                          for i in range(0, len(Yte), 512)], 0).cpu().numpy()
    m = r2(pred, Yte)
    rec = {"mode": mode, "seed": seed, "params": npar, "r2": m}
    json.dump(rec, open(f"{OUT_DIR}/{mode}_{seed}.json", "w"))
    torch.save({"state": model.state_dict(), **rec}, f"{OUT_DIR}/{mode}_{seed}.pt")
    print(f"  DONE {mode:<14} seed={seed} R²=vx={m[0]:+.3f} vy={m[1]:+.3f} wz={m[2]:+.3f}",
          flush=True)


if __name__ == "__main__":
    main()
