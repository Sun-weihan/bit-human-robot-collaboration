#!/usr/bin/env python3
"""诊断：真实数据上力到底能不能解释速度？

如果连"力窗口的简单统计量 -> 速度"的线性回归都学不出东西，
那问题不在模型架构，而在于标签（WHAM 位姿算出的速度）或时间对齐。

做法：
  1. 从 198 帧力窗口提取简单统计量（均值/标准差/首尾差/线性趋势）
  2. 岭回归预测速度，按轨迹划分
  3. 报告 R² 与相关系数

对照：同一流程在合成数据上跑（已知合成数据是可学的）。
"""
import numpy as np
import torch

H, S, T = 6, 33, 198


def feats(f):
    """f: (N,T,6) -> (N, 6*5) 简单统计量"""
    mu = f.mean(1)                       # 均值
    sd = f.std(1)                        # 标准差
    d = f[:, -1, :] - f[:, 0, :]         # 首尾差
    mx = f.max(1)
    mn = f.min(1)
    return np.concatenate([mu, sd, d, mx, mn], 1)


def ridge_fit(Xtr, Ytr, lam=1.0):
    X = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    A = X.T @ X + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, X.T @ Ytr)
    return W


def ridge_pred(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def report(name, Yte, P):
    out = []
    for d in range(Yte.shape[1]):
        y, p = Yte[:, d], P[:, d]
        ss = ((y - p) ** 2).sum()
        st = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss / (st + 1e-9)
        c = np.corrcoef(p, y)[0, 1] if p.std() > 1e-9 and y.std() > 1e-9 else 0.0
        out.append((r2, c))
    print(f"  {name}")
    for d, nm in enumerate(("vx", "vy", "vz")):
        print(f"    {nm}: R²={out[d][0]:+.3f}  corr={out[d][1]:+.3f}")


def main():
    # ---- 真实数据 ----
    d = np.load("/root/autodl-tmp/h2data/real_dataset.npz")
    f, sizes = d["f"], d["traj_sizes"]
    y = d["y"][:, 0, :]      # 预测 horizon 第一步（N,3）
    n_tr = int(len(sizes) * 0.85)
    idx = int(sizes[:n_tr].sum())
    X = feats(f)
    print("=" * 66)
    print(f"  真实数据：{len(y)} 样本，训练 {idx} / 测试 {len(y)-idx}（按轨迹划分）")
    print(f"  力特征维度 {X.shape[1]}，标签均值 {y.mean(0).round(3)}，"
          f"标准差 {y.std(0).round(3)}")
    W = ridge_fit(X[:idx], y[:idx])
    report("岭回归（简单力统计量 -> 速度）", y[idx:], ridge_pred(W, X[idx:]))

    # 常量预测（猜均值）作为下限
    Pc = np.tile(y[:idx].mean(0), (len(y) - idx, 1))
    report("常量预测（猜训练集均值，下限）", y[idx:], Pc)

    # ---- 合成数据对照 ----
    print("\n" + "=" * 66)
    import sys
    sys.path.insert(0, "/root/autodl-tmp/h2official")
    import ab_arch as AB
    Fs, Tau, Ys = AB.gen_data(1500, seed=0)
    Fs, Ys = Fs.numpy(), Ys.numpy()[:, 0, :]
    Xs = feats(Fs)
    n = int(len(Ys) * 0.85)
    Ws = ridge_fit(Xs[:n], Ys[:n])
    print(f"  合成数据：{len(Ys)} 样本（同流程对照）")
    report("岭回归（合成数据）", Ys[n:], ridge_pred(Ws, Xs[n:]))


if __name__ == "__main__":
    main()
