#!/usr/bin/env python3
"""上层改为「力 + 上一帧速度 → 速度增量」的残差模型。

动机（实测依据）：
  速度标签极其平滑——用"上一帧速度"直接当预测，R² 就有
  vx 0.971 / vy 0.825 / wz 0.623（自相关 0.986/0.912/0.811）。
  所以「力 → 绝对速度」这个提法本身很弱：绝对速度主要由自身历史决定。

  正确提法（也是导纳控制的标准形式）：力 → 速度增量 Δv，再叠加到当前速度：
      v̂[t] = v[t-1] + Δv̂
  机器人在闭环里本来就知道自己的速度，用它作为输入完全合法。

三种模式（消融设计）：
  abs     —— 力 → 绝对速度（对照，当前做法）
  delta   —— 力 + 上一帧速度 → Δv，再叠加        ← 本方案
  delta_nv—— 力 → Δv，但不给上一帧速度（消融：
             用于区分"增量形式"和"知道当前速度"各自的贡献）

用法: python st_moe_delta.py <mode> <seed>
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
import st_moe as SM  # noqa: E402

DEV = SM.DEV
OUT_DIR = os.environ.get("DLT_OUT", "/root/autodl-tmp/h2official/seeds_delta")
os.makedirs(OUT_DIR, exist_ok=True)
STEPS = 6000
BATCH = 64


class DeltaNet(nn.Module):
    """双流编码器（各输出 3 维）+ 可选的速度条件 + 残差头。"""

    def __init__(self, mode="delta"):
        super().__init__()
        self.mode = mode
        self.sus = SM.StreamEncoder(6, 64, 4, 2, 0.5, SM.LEVELS)
        self.tra = SM.StreamEncoder(6 * SM.J_SCALES, 64, 4, 2, 0.5, 1)
        in_dim = 6 + (3 if mode == "delta" else 0)      # 两路各 3 维 + 速度条件
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, apx_f, scat_f, v_prev):
        B, H = apx_f.shape[0], apx_f.shape[1]
        z = torch.zeros(B, H, 3, device=apx_f.device)
        t0 = torch.zeros(B, dtype=torch.long, device=apx_f.device)
        o_s = self.sus(z, t0, apx_f)                    # (B,H,3)
        o_t = self.tra(z, t0, scat_f)                   # (B,H,3)
        feat = torch.cat([o_s, o_t], -1)                # (B,H,6)
        if self.mode == "delta":
            feat = torch.cat([feat, v_prev.unsqueeze(1).expand(-1, H, -1)], -1)
        return self.head(feat)                          # (B,H,3)


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

    # 全局样本 -> 轨迹内位置；取同轨迹上一帧速度作为 v_prev
    pos = np.zeros(len(y), dtype=int)
    off = 0
    for s in sizes:
        s = int(s); pos[off:off + s] = np.arange(s); off += s
    Y = np.asarray(y)
    prev = np.zeros((len(y), 3), dtype=np.float32)
    for i in range(len(y)):
        prev[i] = Y[i - 1, 0, :] if pos[i] > 0 else Y[i, 0, :]
    v_prev = torch.tensor(prev)

    def sidx(ts):
        out, o2 = [], 0
        for i, s in enumerate(sizes):
            if i in ts: out.extend(range(o2, o2 + int(s)))
            o2 += int(s)
        return np.array(out)

    a, b = sidx(set(tr_t)), sidx(set(te_t))
    print(f"  轨迹 {len(tr_t)}/{len(te_t)}；样本 {len(a)}/{len(b)}", flush=True)
    A = SM.swt_approx(f); S = SM.scattering_features(f)

    def mk(ii):
        Ax = A[ii].reshape(len(ii), SM.H, SM.S, *A.shape[2:]).to(DEV)
        Sx = S[ii].reshape(len(ii), SM.H, SM.S, S.shape[-1]).to(DEV)
        return {"y": y[ii].to(DEV), "apx": Ax, "scat": Sx, "vp": v_prev[ii].to(DEV)}

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
    model = DeltaNet(mode).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  [{mode}] 参数量 {npar:,}", flush=True)

    use_delta = mode.startswith("delta")

    def target_of(yy, vp):
        """Δv 目标：每个 horizon 帧相对 v_prev 的增量"""
        if not use_delta:
            return yy
        return yy - vp.unsqueeze(1)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = dtr["y"].shape[0]
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), device=DEV)
        o = model(dtr["apx"][idx], dtr["scat"][idx], dtr["vp"][idx])
        loss = F.mse_loss(o, target_of(dtr["y"][idx], dtr["vp"][idx]))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0:
            print(f"    step {step} mse {loss.item():.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        o = torch.cat([model(dte["apx"][i:i+256], dte["scat"][i:i+256], dte["vp"][i:i+256])
                       for i in range(0, len(dte["y"]), 256)], 0)
    pred = dte["vp"].unsqueeze(1) + o if use_delta else o
    m = metrics(pred, dte["y"])
    rec = {"mode": mode, "seed": seed, "params": npar, "r2": m}
    with open(f"{OUT_DIR}/{mode}_{seed}.json", "w") as fh:
        json.dump(rec, fh)
    torch.save({"state": model.state_dict(), "mode": mode, "seed": seed,
                "params": npar, "r2": m}, f"{OUT_DIR}/{mode}_{seed}.pt")
    print(f"  DONE {mode} seed={seed} R²=vx={m[0]:+.3f} vy={m[1]:+.3f} wz={m[2]:+.3f}")


if __name__ == "__main__":
    main()
