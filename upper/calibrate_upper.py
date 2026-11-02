#!/usr/bin/env python3
"""上层输出的增益/偏置校准（抵消 MSE 训练的"回归到均值"）。

问题：
  用 MSE 训练的回归器会把极端值往均值压缩——实测 abs 模型预测的 vx 均值 0.28，
  真人 0.336，系统性低估约 17%。这直接吃掉端到端复现率。

做法：
  在**训练集**上拟合仿射校准  y ≈ κ·p + b （逐维最小二乘），
  然后在**测试集**上验证 R² 是否提升，与未校准对照。

纪律：
  κ、b 只在训练集拟合，测试集仅用于报告，不参与任何选择。
"""
import collections
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/h2official")
import st_moe as SM  # noqa: E402
import st_moe_reg as R  # noqa: E402

DEV = SM.DEV
OUT = "/root/autodl-tmp/h2official/calib.json"


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
        out, o = [], 0
        for i, s in enumerate(sizes):
            if i in ts: out.extend(range(o, o + int(s)))
            o += int(s)
        return np.array(out)

    a, b = sidx(set(tr_t)), sidx(set(te_t))
    A = SM.swt_approx(f); S = SM.scattering_features(f)

    def mk(ii):
        Ax = A[ii].reshape(len(ii), SM.H, SM.S, *A.shape[2:]).to(DEV)
        Sx = S[ii].reshape(len(ii), SM.H, SM.S, S.shape[-1]).to(DEV)
        return Ax, Sx, y[ii].to(DEV)

    return mk(a), mk(b)


@torch.no_grad()
def predict(model, A, S, bs=256):
    return torch.cat([model(A[i:i+bs], S[i:i+bs]) for i in range(0, len(A), bs)], 0)


def r2(pred, true):
    out = []
    for k in range(3):
        p, q = pred[..., k].reshape(-1), true[..., k].reshape(-1)
        out.append(float(1 - ((q - p) ** 2).sum() / (((q - q.mean()) ** 2).sum() + 1e-9)))
    return out


def main():
    ck = torch.load("/root/autodl-tmp/h2official/seeds_reg/fixed_1.pt",
                    map_location=DEV, weights_only=False)
    model = R.RegModel(ck["mode"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()

    (Atr, Str, Ytr), (Ate, Ste, Yte) = load_split()
    print(f"  训练 {len(Atr)}  测试 {len(Ate)}")

    Ptr = predict(model, Atr, Str)
    Pte = predict(model, Ate, Ste)

    # 只用训练集拟合逐维仿射  y = k*p + b
    ptr = Ptr.reshape(-1, 3).cpu().numpy()
    ytr = Ytr.reshape(-1, 3).cpu().numpy()
    K, B = [], []
    for k in range(3):
        X = np.stack([ptr[:, k], np.ones(len(ptr))], 1)
        sol, *_ = np.linalg.lstsq(X, ytr[:, k], rcond=None)
        K.append(float(sol[0])); B.append(float(sol[1]))

    print("\n  === 训练集拟合的校准参数 ===")
    for k, nm in enumerate(["vx", "vy", "wz"]):
        print(f"    {nm}: κ={K[k]:.4f}  b={B[k]:+.4f}")

    # 测试集对照
    Kt = torch.tensor(K, device=DEV).view(1, 1, 3)
    Bt = torch.tensor(B, device=DEV).view(1, 1, 3)
    Pte_cal = Pte * Kt + Bt

    r_raw = r2(Pte, Yte)
    r_cal = r2(Pte_cal, Yte)
    print("\n  === 测试集 R² ===")
    print(f"    {'维度':<5}{'未校准':>12}{'校准后':>12}{'变化':>10}")
    for k, nm in enumerate(["vx", "vy", "wz"]):
        print(f"    {nm:<5}{r_raw[k]:>12.4f}{r_cal[k]:>12.4f}{r_cal[k]-r_raw[k]:>+10.4f}")

    # 均值偏差
    m_raw = Pte.reshape(-1, 3).mean(0).cpu().numpy()
    m_cal = Pte_cal.reshape(-1, 3).mean(0).cpu().numpy()
    m_true = Yte.reshape(-1, 3).mean(0).cpu().numpy()
    print("\n  === 均值（低估问题）===")
    for k, nm in enumerate(["vx", "vy", "wz"]):
        print(f"    {nm}: 未校准 {m_raw[k]:+.4f}  校准后 {m_cal[k]:+.4f}  真值 {m_true[k]:+.4f}")

    json.dump({"kappa": K, "bias": B, "r2_raw": r_raw, "r2_cal": r_cal},
              open(OUT, "w"), indent=1)
    print(f"\n  已保存 {OUT}")


if __name__ == "__main__":
    main()
