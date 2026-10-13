"""第 4 周 · 题 4.3：LQR 稳定倒立摆

任务
----
(a) 用 scipy.linalg.solve_continuous_are 解 Riccati 方程，实现 u = -Kx，
    仿真从 theta = 0.2 rad 恢复直立。
(b) 标量情形对照：对标量系统 x_dot = a*x + b*u，验证
        P 满足  2aP - P^2 b^2 / r + q = 0
        K = P b / r
    数值解必须吻合到 1e-10。
(c) 扫描 R，观察 R->0（激进）与 R->inf（保守）时闭环极点的移动趋势。

运行
----
    python cartpole_lqr.py

依赖
----
    numpy, scipy, matplotlib（本机通用 venv 已装）
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

# 复用题 4.1 的模型
from cartpole_model import M, m, l, g, build_AB

# --------------------------------------------------------------------------
# LQR 核心
# --------------------------------------------------------------------------
def solve_lqr(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """求解连续时间 LQR。

    求解代数 Riccati 方程（ARE）:
        A^T P + P A - P B R^-1 B^T P + Q = 0
    返回反馈增益 K = R^-1 B^T P 与 P 本身。

    Returns
    -------
    K : (1, 4) ndarray
    P : (4, 4) ndarray
    """
    P = sla.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


def verify_are(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray) -> float:
    """把 P 代回 ARE，返回残差的无穷范数。应当接近 0。"""
    residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    return float(np.max(np.abs(residual)))


# --------------------------------------------------------------------------
# 题 4.3(b)：标量情形解析对照
# --------------------------------------------------------------------------
def scalar_lqr_check(a: float = -1.0, b: float = 2.0, q: float = 1.0, r: float = 0.5) -> None:
    """标量系统 x_dot = a*x + b*u 的 LQR。

    解析解：2aP - P^2 b^2 / r + q = 0
            => P = (a*r + sqrt(a^2 r^2 + q*b^2*r)) / b^2   （取正根，要求 P > 0）
            K = P*b/r
    """
    print("\n=== 题 4.3(b) 标量 LQR 解析 vs 数值 ===")
    print(f"  a={a}, b={b}, q={q}, r={r}")

    # 解析解
    disc = a * a * r * r + q * b * b * r
    P_analytic = (a * r + np.sqrt(disc)) / (b * b)
    K_analytic = P_analytic * b / r

    # 数值解
    A = np.array([[a]])
    B = np.array([[b]])
    Q = np.array([[q]])
    R = np.array([[r]])
    K_num, P_num = solve_lqr(A, B, Q, R)

    err_P = abs(P_analytic - P_num[0, 0])
    err_K = abs(K_analytic - K_num[0, 0])

    print(f"  P: 解析 = {P_analytic:.15f}")
    print(f"    数值 = {P_num[0, 0]:.15f}   误差 = {err_P:.3e}")
    print(f"  K: 解析 = {K_analytic:.15f}")
    print(f"    数值 = {K_num[0, 0]:.15f}   误差 = {err_K:.3e}")

    # 验证 P 确实满足标量 Riccati 方程
    riccati_val = 2 * a * P_analytic - P_analytic**2 * b**2 / r + q
    print(f"  代入 2aP - P^2 b^2/r + q = {riccati_val:.3e}  （应≈0）")

    if max(err_P, err_K) < 1e-10:
        print("  ✓ 通过：吻合到 1e-10")
    else:
        print("  ✗ 不通过")


# --------------------------------------------------------------------------
# 仿真
# --------------------------------------------------------------------------
def simulate(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    x0: np.ndarray,
    dt: float = 0.002,
    T: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用欧拉法仿真闭环 x_dot = (A - B K) x。

    返回值
    ------
    t   : (N,)     时间序列
    X   : (N, 4)   状态轨迹
    U   : (N,)     控制量序列
    """
    n = int(T / dt)
    t = np.arange(n) * dt
    X = np.zeros((n, 4))
    U = np.zeros(n)

    x = x0.astype(float).copy()
    for k in range(n):
        u = float(-K @ x)
        U[k] = u
        X[k] = x
        x = x + dt * (A @ x + B.flatten() * u)

    return t, X, U


# --------------------------------------------------------------------------
# 题 4.3(c)：扫描 R
# --------------------------------------------------------------------------
def scan_R(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R_values: list[float]) -> None:
    print("\n=== 题 4.3(c) 扫描 R：闭环极点移动 ===")
    print(f"  {'R':>10}  {'||K||':>12}   闭环极点（实部，排序后）")
    for r_val in R_values:
        R = np.diag([r_val])
        K, _ = solve_lqr(A, B, Q, R)
        cl = A - B @ K
        poles = np.sort_complex(np.linalg.eigvals(cl))
        pole_str = "  ".join(f"{p.real:+.3f}" for p in poles)
        print(f"  {r_val:>10.4g}  {np.linalg.norm(K):>12.4f}   [{pole_str}]")
    print("\n  观察要点：")
    print("    R -> 0   : 控制代价便宜 -> 增益变大 -> 极点实部更负（响应更快、更激进）")
    print("    R -> inf : 控制代价昂贵 -> 增益变小 -> 极点趋近开环（响应慢、更保守）")


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_results(t: np.ndarray, X: np.ndarray, U: np.ndarray) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[跳过绘图] 未安装 matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    labels = [r"$p$ [m]", r"$\theta$ [rad]", r"$\dot p$ [m/s]", r"$\dot\theta$ [rad/s]"]
    for idx, ax in enumerate(axes.flat):
        ax.plot(t, X[:, idx], lw=1.6)
        ax.set_ylabel(labels[idx])
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("time [s]")

    fig.suptitle("LQR 稳定倒立摆（初始 $\\theta_0 = 0.2$ rad）", fontsize=13)
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(t, U, lw=1.6, color="crimson")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("control force $u$ [N]")
    ax2.set_title("控制输入")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()

    out = "week04_lqr_result.png"
    fig.savefig(out, dpi=120)
    print(f"\n  图已保存: {out}")
    plt.show()


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 64)
    print("第 4 周 · 题 4.3  LQR 稳定倒立摆")
    print("=" * 64)

    # ---- 标量解析对照（不依赖题 4.1）----
    scalar_lqr_check()

    # ---- 完整系统 ----
    try:
        A, B = build_AB()
    except NotImplementedError as exc:
        print(f"\n[跳过倒立摆部分] {exc}")
        print("先完成 cartpole_model.py 的 build_AB()。")
        return 1

    Q = np.diag([1.0, 10.0, 1.0, 1.0])   # 题面要求
    R = np.diag([0.1])                    # 题面要求

    K, P = solve_lqr(A, B, Q, R)
    print("\n=== 题 4.3(a) LQR 增益 ===")
    print(f"  K = {K.flatten()}")
    print(f"  P 的对角线 = {np.diag(P)}")
    res = verify_are(A, B, Q, R, P)
    print(f"  ARE 残差 = {res:.3e}  " + ("✓" if res < 1e-8 else "✗"))

    cl_poles = np.linalg.eigvals(A - B @ K)
    print(f"  闭环极点 = {np.sort_complex(cl_poles)}")
    print("  " + ("✓ 全部实部为负，闭环稳定" if np.all(cl_poles.real < 0) else "✗ 存在不稳定极点"))

    # 仿真
    x0 = np.array([0.0, 0.2, 0.0, 0.0])   # theta = 0.2 rad
    t, X, U = simulate(A, B, K, x0)

    print("\n=== 仿真结果 ===")
    print(f"  初始 theta = {np.degrees(x0[1]):.2f} deg")
    print(f"  末态 theta = {np.degrees(X[-1, 1]):.4f} deg")
    print(f"  末态 p     = {X[-1, 0]:.4f} m")
    print(f"  末态速度   = {np.max(np.abs(X[-1, 2:])):.3e}")
    print(f"  最大控制力 = {np.max(np.abs(U)):.4f} N")

    settled = np.abs(X[-1, 2:]).max() < 1e-3 and abs(X[-1, 1]) < 1e-3
    print("  " + ("✓ 已恢复直立并静止" if settled else "✗ 未收敛"))

    # 扫描 R
    scan_R(A, B, Q, [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])

    # 绘图
    plot_results(t, X, U)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
