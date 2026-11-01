#!/usr/bin/env python3
"""从紧凑格式构建真实数据训练集（服务器端执行）。

输入（每条轨迹一个目录）：
    pose.npz      pose_world(N,72) trans_world(N,3) frame_ids(N,)
    forces_mini40.csv / forces_mini45.csv    timestamp,fx,fy,fz,tx,ty,tz
    img_ts.txt    图像文件名（含时间戳），每行一个

输出：real_dataset.npz  f(T,6) t(T,6) y(H,3) w(H)

与官方 dataset.py 的对应关系：
    力：1 kHz 六轴，按图像时间戳对齐后插值到 T=S*H 个点
    标签：本体系线速度 + 偏航角速度
    样本：起点 t0 的 198 个力采样 -> 之后 H=6 帧速度

与合成数据版的两处差异（都是为了真实数据的噪声）：
  1. 偏航角速度用**速度航向角**的变化率，并对平滑后截断到 ±2 rad/s
     （WHAM 逐帧朝向差分噪声达 ±46 rad/s，不可用）
  2. 增加质量过滤：剔除 |v| > 1.5 m/s 或单步跳变过大的片段，
     这些是 WHAM 估计失败而非真实运动
"""
import glob
import os
import sys

import numpy as np

H = 6
S = 33
T = H * S
VMAX = 1.5          # 物理过滤：人搬箱子不会超过 1.5 m/s
WMAX = 2.0          # 偏航角速度上限 rad/s


def aa_to_R(aa):
    th = np.linalg.norm(aa, axis=1, keepdims=True)
    th_safe = np.where(th < 1e-9, 1.0, th)
    k = aa / th_safe
    K = np.zeros((len(aa), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = np.eye(3)[None]
    R = I + np.sin(th)[:, :, None] * K + (1 - np.cos(th))[:, :, None] * (K @ K)
    small = th[:, 0] < 1e-9
    if small.any():
        R[small] = np.eye(3)[None]
    return R


def smooth(x, w=5):
    if w <= 1 or len(x) < w:
        return x
    k = np.ones(w) / w
    if x.ndim == 1:
        pad = np.pad(x, (w // 2, w // 2), mode="edge")
        return np.convolve(pad, k, mode="valid")[:len(x)]
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        pad = np.pad(x[:, c], (w // 2, w // 2), mode="edge")
        out[:, c] = np.convolve(pad, k, mode="valid")[:len(x)]
    return out


def build_one(d, stats):
    ts_file = os.path.join(d, "img_ts.txt")
    if not os.path.exists(ts_file):
        return None
    names = [x.strip() for x in open(ts_file) if x.strip()]
    if len(names) < 20:
        return None
    t_img = np.array([float(os.path.splitext(n)[0]) for n in names])
    dt = float(np.median(np.diff(t_img)))
    if not (0.01 < dt < 0.2):
        return None

    pz = np.load(os.path.join(d, "pose.npz"))
    pose_world = pz["pose_world"].astype(np.float64)
    trans_world = pz["trans_world"].astype(np.float64)
    n = min(len(pose_world), len(t_img))

    R = aa_to_R(pose_world[:n, :3])
    pos_s = smooth(trans_world[:n], 5)
    v_global = np.diff(pos_s, axis=0) / dt
    vel_local = np.einsum("nji,nj->ni", R[1:], v_global)

    spd = np.linalg.norm(v_global[:, :2], axis=1)
    moving = spd > 0.15
    heading = np.unwrap(np.arctan2(v_global[:, 1], v_global[:, 0]))
    heading_s = smooth(heading, 15)
    omega = np.clip(np.diff(heading_s) / dt, -WMAX, WMAX)
    omega = np.where(moving[:-1] & moving[1:], omega, 0.0)

    # ---- 力 ----
    def load_force(p):
        a = np.loadtxt(p, delimiter=",", skiprows=1)
        return a[:, 0], a[:, 1:7]

    p40 = os.path.join(d, "forces_mini40.csv")
    p45 = os.path.join(d, "forces_mini45.csv")
    if not (os.path.exists(p40) and os.path.exists(p45)):
        return None
    t40, f40 = load_force(p40)
    t45, f45 = load_force(p45)
    if len(t40) < 10 or len(t45) < 10:
        return None

    def nearest(ts, f, tq):
        idx = np.searchsorted(ts, tq).clip(0, len(ts) - 1)
        lo = np.clip(idx - 1, 0, len(ts) - 1)
        pick = np.where(np.abs(ts[lo] - tq) < np.abs(ts[idx] - tq), lo, idx)
        return f[pick]

    f40a = nearest(t40, f40, t_img[:n])
    f45a = nearest(t45, f45, t_img[:n])
    force_pair = np.concatenate([f40a[:, :3], f45a[:, :3]], 1)
    tau_pair = np.concatenate([f40a[:, 3:], f45a[:, 3:]], 1)

    # ---- 切样本 + 质量过滤 ----
    n_valid = len(vel_local) - H
    F, TAU, Y, W, FN = [], [], [], [], []
    dropped = 0
    for t0 in range(max(n_valid, 0)):
        y = vel_local[t0:t0 + H]
        w = omega[t0:t0 + H]
        # 过滤：速度超出物理范围、或窗口内出现异常跳变
        if np.abs(y[:, :2]).max() > VMAX or np.abs(w).max() > WMAX:
            dropped += 1
            continue
        # 因果对齐：力窗口取 t0 **之前**的 H 帧（约 0.2 s），
        # 目标是 t0 **之后**的 H 帧。之前写成 [t0, t0+H] 与目标重叠，
        # 等于让模型看着正在发生的力去猜正在发生的事，是标签泄漏。
        seg = np.linspace(t_img[max(t0 - H, 0)], t_img[t0], T)
        fi = np.searchsorted(t_img[:n], seg).clip(0, n - 1)
        F.append(force_pair[fi].astype(np.float32))
        TAU.append(tau_pair[fi].astype(np.float32))
        # 论文的指令是**平面**的 (vx, vy, ωz)：vz 是上下颠簸，
        # 实测力对它的解释力为 R²=-7.07（纯噪声），放进标签只会污染指标
        Y.append(np.stack([y[:, 0], y[:, 1], w], axis=1).astype(np.float32))
        W.append(w.astype(np.float32))
        # 辅助任务标签：目标窗口内的平均力（自监督"下一步力预测"，不需要额外标注）
        nxt = force_pair[t0:t0 + H].mean(0) if t0 + H <= n else force_pair[t0:n].mean(0)
        FN.append(nxt.astype(np.float32))

    if not F:
        return None
    stats["dropped"] += dropped
    if dropped > 0.5 * max(n_valid, 1):
        stats["bad"] += 1
    return (np.stack(F), np.stack(TAU), np.stack(Y), np.stack(W), np.stack(FN))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/h2data/real"
    dirs = sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])
    print(f"  轨迹目录 {len(dirs)} 个")
    acc = {"f": [], "t": [], "y": [], "w": [], "fn": []}
    sizes = []
    tnames = []
    stats = {"dropped": 0, "bad": 0, "ok": 0, "skip": 0}
    for i, d in enumerate(dirs, 1):
        try:
            r = build_one(d, stats)
        except Exception as e:
            stats["skip"] += 1
            continue
        if r is None:
            stats["skip"] += 1
            continue
        F, TAU, Y, W, FN = r
        acc["f"].append(F); acc["t"].append(TAU)
        acc["y"].append(Y); acc["w"].append(W); acc["fn"].append(FN)
        sizes.append(len(F))
        tnames.append(os.path.basename(d))
        stats["ok"] += 1
        if i % 20 == 0:
            print(f"  [{i}/{len(dirs)}] 有效 {stats['ok']}，过滤样本 {stats['dropped']}")

    if not acc["f"]:
        print("  没有可用轨迹")
        return
    f = np.concatenate(acc["f"]); t = np.concatenate(acc["t"])
    y = np.concatenate(acc["y"]); w = np.concatenate(acc["w"])
    fn = np.concatenate(acc["fn"])
    out = "/root/autodl-tmp/h2data/real_dataset.npz"
    np.savez_compressed(out, f=f, t=t, y=y, w=w, f_next=fn, traj_sizes=np.array(sizes), traj_names=np.array(tnames))
    print(f"\n  已保存 {out}")
    print(f"  轨迹: 有效 {stats['ok']} / 跳过 {stats['skip']} / 高丢弃 {stats['bad']}")
    print(f"  样本 {len(f)}，过滤掉 {stats['dropped']} 个异常窗口")
    print(f"  vx[{y[...,0].min():+.2f},{y[...,0].max():+.2f}] "
          f"vy[{y[...,1].min():+.2f},{y[...,1].max():+.2f}] "
          f"vz[{y[...,2].min():+.2f},{y[...,2].max():+.2f}]")
    print(f"  omega[{w.min():+.2f},{w.max():+.2f}]  |f|max={np.abs(f).max():.1f} N")


if __name__ == "__main__":
    main()
