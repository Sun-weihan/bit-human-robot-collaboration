#!/usr/bin/env python3
"""双流编码器 + 直接回归头（不做扩散）。

为什么要有这一版：
  实测发现扩散版本只在"输入里已含 94% 真值"时表现好（t=20 时 R²=0.76），
  一旦从纯噪声采样，输出尺度就错 4-6 倍（vx 标准差 0.351 vs 真值 0.084），
  R² 变成 -3.8。说明模型没学到条件分布，之前的 R² 是"oracle 初始化"的假象。

  对于被 R² 评价的 3 维回归任务，扩散本来就不是必需的；
  论文用它是为了多模态（一个力可能对应多种意图），而不是精度。
  所以这里保留同一套双流编码器，把扩散头换成直接回归头，给出**诚实**的 R²。

用法: python st_moe_reg.py <mode> <seed>    mode: mono | fixed
"""
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/h2official")
import st_moe as SM  # noqa: E402

DEV = SM.DEV
OUT_DIR = os.environ.get("REG_OUT", "/root/autodl-tmp/h2official/seeds_reg")
os.makedirs(OUT_DIR, exist_ok=True)
STEPS = 6000
BATCH = 64


class RegModel(torch.nn.Module):
    """复用 STMoe 的两个编码器，改为直接输出 (B,H,3)。"""

    def __init__(self, mode="fixed"):
        super().__init__()
        self.mode = mode
        if mode == "mono":
            self.mono = SM.StreamEncoder(6, 128, 4, 4, 0.5, SM.LEVELS)
        else:
            self.sus = SM.StreamEncoder(6, 64, 4, 2, 0.5, SM.LEVELS)
            self.tra = SM.StreamEncoder(6 * SM.J_SCALES, 64, 4, 2, 0.5, 1)

    def forward(self, apx_f, scat_f):
        # 用零噪声、零时间步走到编码器（编码器内部的 q_proj 输入 3 维任务向量）
        B, H = apx_f.shape[0], apx_f.shape[1]
        dummy = torch.zeros(B, H, 3, device=apx_f.device)
        t0 = torch.zeros(B, dtype=torch.long, device=apx_f.device)
        if self.mode == "mono":
            return self.mono(dummy, t0, apx_f)
        return 0.5 * (self.sus(dummy, t0, apx_f) + self.tra(dummy, t0, scat_f))


def load_split():
    d = np.load("/root/autodl-tmp/h2data/real_dataset.npz")
    f = torch.tensor(d["f"]); y = torch.tensor(d["y"])
    sizes = d["traj_sizes"]; names = d["traj_names"]
    g = collections.defaultdict(list)
    for i, n in enumerate(names):
        g[str(n).split("__")[-1].split("_")[0]].append(i)
    tr_t, te_t = [], []
    for k, ids in sorted(g.items()):
        ids = sorted(ids); kk = max(1, int(round(len(ids) * 0.15)))
        te_t += ids[-kk:]; tr_t += ids[:-kk]

    def sidx(ts):
        out, off = [], 0
        for i, s in enumerate(sizes):
            if i in ts: out.extend(range(off, off + int(s)))
            off += int(s)
        return np.array(out)

    a, b = sidx(set(tr_t)), sidx(set(te_t))
    print(f"  轨迹 {len(tr_t)}/{len(te_t)}；样本 {len(a)}/{len(b)}", flush=True)
    A = SM.swt_approx(f); S = SM.scattering_features(f)

    def mk(ii):
        Ax = A[ii].reshape(len(ii), SM.H, SM.S, *A.shape[2:]).to(DEV)
        Sx = S[ii].reshape(len(ii), SM.H, SM.S, S.shape[-1]).to(DEV)
        return {"y": y[ii].to(DEV), "apx": Ax, "scat": Sx}

    return mk(a), mk(b)


def metrics(pred, true):
    out = []
    for k in range(3):
        p, q = pred[..., k].reshape(-1), true[..., k].reshape(-1)
        out.append(float(1 - ((q - p) ** 2).sum() / (((q - q.mean()) ** 2).sum() + 1e-9)))
    return out


def main():
    mode, seed = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(seed); np.random.seed(seed)
    dtr, dte = load_split()
    model = RegModel(mode).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  [{mode}] 直接回归，参数量 {npar:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = dtr["y"].shape[0]
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), device=DEV)
        pred = model(dtr["apx"][idx], dtr["scat"][idx])
        loss = F.mse_loss(pred, dtr["y"][idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0:
            print(f"    step {step} mse {loss.item():.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(dte["apx"][i:i+256], dte["scat"][i:i+256])
                          for i in range(0, len(dte["y"]), 256)], 0)
    m = metrics(pred, dte["y"])
    rec = {"mode": mode, "seed": seed, "params": npar, "r2": m}
    with open(f"{OUT_DIR}/{mode}_{seed}.json", "w") as fh:
        json.dump(rec, fh)
    torch.save({"state": model.state_dict(), "mode": mode, "seed": seed,
                "params": npar, "r2": m}, f"{OUT_DIR}/{mode}_{seed}.pt")
    print(f"  DONE {mode} seed={seed} 直接回归 R²=vx={m[0]:+.3f} vy={m[1]:+.3f} wz={m[2]:+.3f}")


if __name__ == "__main__":
    main()
