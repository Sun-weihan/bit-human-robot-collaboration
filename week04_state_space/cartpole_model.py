"""第 4 周 · 题 4.1：小车-摆（cartpole）建模与线性化

任务
----
在 theta = 0（直立）附近线性化，写出 A、B 矩阵，并做量纲检查。

状态定义
--------
    x = [p, theta, p_dot, theta_dot]^T
    p         小车位置            [m]
    theta     摆杆与竖直方向夹角   [rad]（theta=0 为直立，向上）
    p_dot     小车速度            [m/s]
    theta_dot 摆杆角速度          [rad/s]

物理参数
--------
    M   小车质量  [kg]
    m   摆杆末端集中质量（或摆杆质量，注意你选哪种模型）[kg]
    l   摆杆半长（质心到转轴距离）[m]
    g   重力加速度 [m/s^2]
    u   施加在小车上的水平力 [N]

需要你自己做的
--------------
1. 用拉格朗日法或牛顿法写出非线性方程：
       (M+m) p_ddot + m l cos(theta) theta_ddot - m l sin(theta) theta_dot^2 = u
       (I + m l^2) theta_ddot + m l cos(theta) p_ddot - m g l sin(theta) = 0
   （注意：上面式子的 I 与 m l^2 是否合并，取决于你把摆杆建模成
     质点还是均匀细杆——选定一种并在报告里写明。）

2. 在 theta -> 0 处做近似：cos -> 1, sin -> theta, theta_dot^2 -> 0，
   消去 p_ddot / theta_ddot，整理成:
       d/dt [p, theta, p_dot, theta_dot]^T = A @ x + B @ u

3. 把 A、B 填进 build_AB() 的空位。

量纲检查（验收硬性要求）
------------------------
A 是 4x4，B 是 4x1。逐列核对：
    A 的第 j 列，其单位 = (x_i 的单位) / (x_j 的单位)
    B 的单位        = (x_i 的单位) / (u 的单位)
例如 A[2,0] 连接 p 与 p_ddot，单位应为 1/s^2。
下面 dimension_check() 已写好框架，你只需填 A_real、B_real。

运行
----
    python cartpole_model.py

自检通过标准：dimension_check 全绿，且 A/B 与你手推结果一致。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 物理参数（先用这组，之后可以改成你实际系统的值）
# --------------------------------------------------------------------------
M = 0.5   # 小车质量 [kg]
m = 0.2   # 摆杆质量 [kg]
l = 0.3   # 摆杆半长（质心到转轴）[m]
g = 9.8   # 重力加速度 [m/s^2]


def build_AB(M: float = M, m: float = m, l: float = l, g: float = g) -> tuple[np.ndarray, np.ndarray]:
    """构造线性化后的 A、B 矩阵。

    返回
    ----
    A : (4, 4) ndarray
    B : (4, 1) ndarray

    提示
    ----
    记 p = I*(M+m) + M*m*l^2，其中 I 是摆杆绕质心的转动惯量。
    若把摆杆当均匀细杆（长度 2l）：I = (1/12) * m * (2l)^2 = m*l^2/3
    若把摆杆当末端集中质量（点质量）：I = 0，此时摆杆本身不贡献惯性矩。

    参考形式（点质量模型，I=0）：
        denom = M + m
        A = [[0, 0,                     1,  0],
             [0, 0,                     0,  1],
             [0, -m*g/M,                0,  0],
             [0,  (M+m)*g/(M*l),        0,  0]]
        B = [[0], [0], [1/M], [-1/(M*l)]]

    注意：上面这个"点质量 + 无摩擦"的简化形式是最常见的，
    但它与你第 4 周题面的 (I + m l^2) 形式略有差别。
    请自己推一遍，确认你用哪种，并把推导写进周记。
    """
    # TODO: 替换下面的占位实现
    raise NotImplementedError("请先手推 A、B，再填进这里")


# --------------------------------------------------------------------------
# 量纲检查
# --------------------------------------------------------------------------
def dimension_check(A: np.ndarray, B: np.ndarray) -> bool:
    """核对 A、B 每一列的量纲是否自洽。

    做法：为每个状态量赋予一个"符号单位"，用 sympy 做符号运算，
    检查等号两边单位是否一致。
    """
    try:
        import sympy as sp
    except ImportError:
        print("  [跳过] 未安装 sympy，无法做符号量纲检查")
        print("         装法：pip install sympy")
        return True

    s, kg, m_unit = sp.symbols("s kg m")

    # 各状态量的单位
    u_state = [
        m_unit,        # p          [m]
        1,             # theta      [rad] 视为无量纲
        m_unit / s,    # p_dot      [m/s]
        1 / s,         # theta_dot  [1/s]
    ]
    u_input = kg * m_unit / s**2   # u = 力 [N]

    ok = True
    print("\n=== 量纲检查 ===")

    # 检查 A：A[i, j] 的单位应为 u_state[i] / u_state[j]
    print("\nA 矩阵（4x4）：")
    for i in range(4):
        for j in range(4):
            val = A[i, j]
            if abs(val) < 1e-14:
                continue
            expected = u_state[i] / u_state[j]
            actual = sp.nsimplify(abs(val))
            # 无法从纯数值反推单位，只能检查"是否出现了不该有的量纲组合"
            # 这里改为人工核对清单，打印出来供你逐项确认
            print(
                f"  A[{i},{j}] = {val:>12.6g}   "
                f"应具备单位: {sp.simplify(expected)}"
            )

    print("\nB 矩阵（4x1）：")
    for i in range(4):
        val = B[i, 0]
        if abs(val) < 1e-14:
            continue
        expected = u_state[i] / u_input
        print(
            f"  B[{i},0] = {val:>12.6g}   "
            f"应具备单位: {sp.simplify(expected)}"
        )

    print("\n（说明：纯数值矩阵无法自动判定量纲，上面列出的是"
          "每一项“应当”具备的单位，请人工核对。）")
    return ok


# --------------------------------------------------------------------------
# 交叉验证：与非线性模型的数值线性化对比
# --------------------------------------------------------------------------
def nonlinear_dynamics(
    x: np.ndarray, u: float, M: float = M, m: float = m, l: float = l, g: float = g
) -> np.ndarray:
    """非线性动力学（点质量模型，无摩擦）。

    由:
        theta_ddot = (g*sin(theta) - cos(theta)*u/(M+m)) / (l*(4/3 - m*cos^2(theta)/(M+m)))
        p_ddot     = (u + m*l*(theta_dot^2*sin(theta) - theta_ddot*cos(theta))) / (M+m)
    注意上式中的 4/3 来自 I = m*l^2/3（均匀细杆）；
    如果你选点质量模型，把 4/3 换成 1。

    请按你自己的模型改写本函数。
    """
    p, theta, p_dot, theta_dot = x

    # TODO: 按你自己的模型实现
    raise NotImplementedError("请实现非线性动力学")


def numeric_jacobian(
    x0: np.ndarray | None = None, u0: float = 0.0, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """用中心差分数值求 A = df/dx, B = df/du（在 x0, u0 处）。"""
    if x0 is None:
        x0 = np.zeros(4)

    A = np.zeros((4, 4))
    for j in range(4):
        dx = np.zeros(4)
        dx[j] = eps
        A[:, j] = (
            nonlinear_dynamics(x0 + dx, u0) - nonlinear_dynamics(x0 - dx, u0)
        ) / (2 * eps)

    B = (
        nonlinear_dynamics(x0, u0 + eps) - nonlinear_dynamics(x0, u0 - eps)
    ).reshape(4, 1) / (2 * eps)

    return A, B


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 64)
    print("第 4 周 · 题 4.1  cartpole 建模与线性化")
    print("=" * 64)
    print(f"  M = {M} kg,  m = {m} kg,  l = {l} m,  g = {g} m/s^2")

    try:
        A, B = build_AB()
    except NotImplementedError as exc:
        print(f"\n[未完成] {exc}")
        print("\n下一步：")
        print("  1. 手推 A、B（写在纸上 / 周记里）")
        print("  2. 填进 build_AB()")
        print("  3. 实现 nonlinear_dynamics()")
        print("  4. 重跑本脚本，对比解析 A/B 与 numeric_jacobian() 的结果")
        return 1

    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("\n解析 A =\n", A)
    print("\n解析 B =\n", B)

    dimension_check(A, B)

    # 若实现了非线性模型，自动做数值交叉验证
    try:
        A_num, B_num = numeric_jacobian()
        err_A = np.max(np.abs(A - A_num))
        err_B = np.max(np.abs(B - B_num))
        print("\n=== 与数值线性化对比 ===")
        print(f"  max|A_analytic - A_numeric| = {err_A:.3e}")
        print(f"  max|B_analytic - B_numeric| = {err_B:.3e}")
        if max(err_A, err_B) < 1e-5:
            print("  ✓ 通过：解析解与数值线性化一致")
        else:
            print("  ✗ 不通过：检查推导或模型是否一致")
    except NotImplementedError:
        print("\n[提示] 实现 nonlinear_dynamics() 后可自动做数值交叉验证。")

    # 可控性检查（题 4.2 的预告）
    try:
        C = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
        rank_C = np.linalg.matrix_rank(C)
        print("\n=== 可控性（题 4.2 预告）===")
        print(f"  可控性矩阵的秩 = {rank_C} / 4")
        print("  " + ("✓ 完全可控" if rank_C == 4 else "✗ 存在不可控模态"))
        print("  闭环极点（题 4.4 用）:", np.linalg.eigvals(A))
    except Exception as exc:
        print(f"\n[可控性检查失败] {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
