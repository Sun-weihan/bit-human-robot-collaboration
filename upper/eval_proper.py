#!/usr/bin/env python3
"""用**完整 DDIM 采样**评测上层，修掉两处评测错误。

错误 1（训练脚本 ev()）：用真值构造初始潜变量
    yt = sqrt(abar[t]) * y0 + sqrt(1-abar[t]) * noise
  t=20 时 yt 里 94.3% 是真值 —— 等于把答案告诉模型。

错误 2（我早先的 DualStreamUpper.sample）：只做一步去噪
    yt = sqrt(1-abar[20]) * noise   （方差也不对）
  不可能得到有意义的输出。

正确做法：从 t = T-1 的纯噪声出发，按 DDIM(eta=0) 迭代到 t = 0。
    x0 = (y_t - sqrt(1-ab_t) * eps) / sqrt(ab_t)
    y_{t-1} = sqrt(ab_{t-1}) * x0 + sqrt(1-ab_{t-1}) * eps
"""
import collections
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/h2official")
import ab_arch as AB  # noqa: E402
import st_moe as SM  # noqa: E402

DEV = AB.DEV


@torch.no_grad()
def ddim(model, apx_f, scat_f, n_avg=4, K=50):
    """apx_f: (B,H,S,L,6), scat_f: (B,H,S,6J) -> (B,H,3)"""
    B, H = apx_f.shape[0], apx_f.shape[1]
    outs = []
    for _ in range(n_avg):
        y = torch.randn(B, H, 3, device=DEV)
        for t in reversed(range(0, AB.T_DIFF, max(1, AB.T_DIFF // K))):
            ab = AB.ABAR[t]
            tt = torch.full((B,), t, dtype=torch.long, device=DEV)
            eps = model(y, tt, apx_f, None, scat_f, None, None)
            x0 = (y - (1 - ab).sqrt() * eps) / ab.sqrt()
            if t > 0:
                ab_prev = AB.ABAR[max(t - 1, 0)]
                y = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            else:
                y = x0
        outs.append(y)
    return torch.stack(outs, 0).mean(0)


def r2(pred, true):
    out = []
    for k in range(true.shape[-1]):
        p, q = pred[..., k].reshape(-1), true[..., k].reshape(-1)
        if q.std() < 1e-6:
            out.append(float("nan")); continue
        out.append(float(1 - ((q - p) ** 2).sum() / (((q - q.mean()) ** 2).sum() + 1e-9)))
    return out


def build(ckpt):
    ck = torch.load(ckpt, map_location=DEV, weights_only=False)
    model = SM.STMoe(mode=ck.get("mode", "fixed")).to(DEV)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def load_test(limit=None):
    d = np.load("/root/autodl-tmp/h2data/real_dataset.npz")
    f, y, sizes, names = d["f"], d["y"], d["traj_sizes"], d["traj_names"]
    g = collections.defaultdict(list)
    for i, n in enumerate(names):
        g[str(n).split("__")[-1].split("_")[0]].append(i)
    tr, te = [], []
    for k, ids in sorted(g.items()):
        ids = sorted(ids); kk = max(1, int(round(len(ids) * 0.15)))
        te += ids[-kk:]; tr += ids[:-kk]
    off, te_idx = 0, []
    for i, s in enumerate(sizes):
        if i in set(te):
            te_idx.extend(range(off, off + int(s)))
        off += int(s)
    te_idx = np.array(te_idx)
    if limit:
        te_idx = te_idx[:limit]
    ft = torch.tensor(f[te_idx])
    A = SM.swt_approx(ft).to(DEV)
    S = SM.scattering_features(ft).to(DEV)
    A = A.reshape(1, -1, SM.S, *A.shape[2:])[0] if False else A.reshape(len(te_idx), SM.H, SM.S, *A.shape[2:])
    S = S.reshape(len(te_idx), SM.H, SM.S, S.shape[-1])
    Y = torch.tensor(y[te_idx]).to(DEV)
    return A, S, Y, len(te_idx)


def main():
    ckpts = sys.argv[1:]
    A, S, Y, n = load_test(limit=800)
    print(f"  测试样本 {n}（分层划分，与训练评测同一批）")
    for ck in ckpts:
        model, meta = build(ck)
        pred = ddim(model, A, S)
        m = r2(pred, Y)
        print(f"  {meta.get('mode'):<7} seed={meta.get('seed')} "
              f"正确采样 R² = [{m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}]  "
              f"（训练脚本报的是 {[round(x,3) for x in meta.get('r2',[])]}）")


if __name__ == "__main__":
    main()
