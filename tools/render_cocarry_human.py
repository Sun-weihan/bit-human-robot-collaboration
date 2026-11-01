#!/usr/bin/env python3
"""渲染「人机共抬箱子」：画出人、机器人、以及人施加的力。

场景对应论文的设定：人握箱子一侧、机器人握另一侧，
人的发力通过箱子传到机器人腕部（这是上层扩散模型的输入）。

画面元素：
  * 蓝色示意人 —— 站在箱子另一侧，双臂伸向箱子
  * 橙色箱子   —— 夹在人与机器人之间
  * 红色箭头   —— 箱子/人施加在机器人每只手上的向下负载
  * 青色箭头   —— 人当前施加的水平推力（来自闭环记录里的 ft 字段）
  * 角标文字   —— 实时数值
"""
import sys

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TRAJ = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/mj3/demo_human_traj.npz"
POL = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/mj3/policy_carry8.npz"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/root/autodl-tmp/mj3/cocarry_human.mp4"
FPS = 30
W, H = 640, 480
FOVY = 45.0
FONT = "/root/autodl-tmp/mj3/fonts/msyh.ttc"
XML = "/root/autodl-tmp/mj3/tiangong3_cocarry.xml"
LOAD_PER_HAND = 10.0
HUMAN_STANDOFF = 0.53      # 人相对箱心距离 = 臂长 0.42 + 箱半深 0.11，手正好落在箱子远侧面

tj = np.load(TRAJ)
pol = np.load(POL)
root, jpos = tj["root"], tj["qpos"]
dt = float(tj["step_dt"])
m2i = pol["mujoco_to_isaac_idx"].astype(int)
HAS_FT = "ft" in tj.files
ft_arr = tj["ft"] if HAS_FT else np.zeros((len(root), 6))

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
adr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "carry_box_joint")
bq = m.jnt_qposadr[bj]
hj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "human_joint")
hq = m.jnt_qposadr[hj]
bl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_roll_l_link")
br = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_roll_r_link")

f_big = ImageFont.truetype(FONT, 19)
f_mid = ImageFont.truetype(FONT, 15)
f_sm = ImageFont.truetype(FONT, 13)

ren = mujoco.Renderer(m, height=H, width=W)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation = 2.9, -9


def eye_of(c):
    el, az = np.radians(c.elevation), np.radians(c.azimuth)
    return np.array(c.lookat) + c.distance * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])


def project(p, c):
    eye, look = eye_of(c), np.array(c.lookat)
    fwd = look - eye
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)
    v = p - eye
    zc = float(np.dot(v, fwd))
    if zc <= 1e-6:
        return None
    f = (H / 2.0) / np.tan(np.radians(FOVY) / 2.0)
    return (W / 2.0 + f * float(np.dot(v, right)) / zc,
            H / 2.0 - f * float(np.dot(v, up)) / zc)


def arrow(dr, p0, p1, color, width=4):
    dr.line([p0, p1], fill=color, width=width)
    ang = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    L = 11
    for s in (+1, -1):
        dr.line([p1, (p1[0] - L * np.cos(ang - s * 0.4),
                      p1[1] - L * np.sin(ang - s * 0.4))], fill=color, width=width)


frames = []
SCALE = 0.055
n_out = int(len(root) * dt * FPS)

for k in range(n_out):
    idx = min(int(k / FPS / dt), len(root) - 1)
    d.qpos[:] = 0.0
    d.qpos[0:3] = root[idx, 0:3]
    d.qpos[3:7] = root[idx, 3:7]
    d.qpos[adr] = jpos[idx][m2i]
    mujoco.mj_forward(m, d)

    qw = d.qpos[3:7].copy()
    w, x, y, z = qw
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    pl, pr = d.xpos[bl].copy(), d.xpos[br].copy()
    Rl, Rr = d.xmat[bl].reshape(3, 3), d.xmat[br].reshape(3, 3)
    y_ax = pl - pr
    y_ax /= np.linalg.norm(y_ax) + 1e-9
    z_ax = Rl[:, 2] + Rr[:, 2]
    z_ax /= np.linalg.norm(z_ax) + 1e-9
    x_ax = np.cross(y_ax, z_ax)
    x_ax /= np.linalg.norm(x_ax) + 1e-9
    z_ax = np.cross(x_ax, y_ax)
    R = np.column_stack([x_ax, y_ax, z_ax])
    qb = np.zeros(4)
    mujoco.mju_mat2Quat(qb, R.flatten())
    center = 0.5 * (pl + pr)
    d.qpos[bq:bq + 3] = center
    d.qpos[bq + 3:bq + 7] = qb

    # ---- 人站在箱子对面，双臂伸向箱子 ----
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    human_p = center + fwd * HUMAN_STANDOFF
    human_p[2] = center[2] - 0.28      # 让局部 0.28 高处的双手落在箱心高度
    d.qpos[hq:hq + 3] = human_p
    # 人的朝向与机器人相同：其局部 -x 指向机器人，双臂（朝 -x）才能伸到箱子上
    d.qpos[hq + 3:hq + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)

    cam.azimuth = 115 - np.degrees(yaw)
    cam.lookat[:] = [root[idx, 0], root[idx, 1], 0.95]
    ren.update_scene(d, cam)
    img = Image.fromarray(ren.render())
    dr = ImageDraw.Draw(img)

    # 负载箭头（机器人双手）
    for p in (pl, pr):
        p0 = project(p, cam)
        p1 = project(p + np.array([0.0, 0.0, -LOAD_PER_HAND * SCALE]), cam)
        if p0 and p1:
            arrow(dr, p0, p1, (255, 90, 70), 4)
            dr.ellipse([p0[0] - 4, p0[1] - 4, p0[0] + 4, p0[1] + 4], fill=(255, 200, 80))

    # 人施加的力
    fx, fy, fz = (float(ft_arr[idx][0]), float(ft_arr[idx][1]), float(ft_arr[idx][2]))
    mag_h = float(np.hypot(fx, fy))
    if mag_h > 0.5:
        # 注：录制时 ft 的 x 分量存的是力的模长（norm），方向信息已丢失。
        # 物理上人是"领导者"，沿前进方向牵引箱子，所以箭头画在 +fwd 方向
        # （即从箱子指向站在前方的人），长度按模长缩放。
        f_world = -fwd * mag_h * SCALE   # 指向站在前方的人（领导者牵引方向）
        p0 = project(center, cam)
        p1 = project(center + f_world, cam)
        if p0 and p1:
            arrow(dr, p0, p1, (110, 205, 255), 5)
            dr.text((p1[0] + 7, p1[1] - 7), f"人的推力 {mag_h:.0f} N",
                    font=f_sm, fill=(160, 220, 255))

    # 左上角说明
    dr.rectangle([8, 8, 322, 96], fill=(20, 24, 30))
    dr.text((16, 13), "人机共抬 · 力示意", font=f_big, fill=(255, 235, 200))
    dr.text((16, 38), "人推箱子 -> 力经箱子传到机器人双手", font=f_mid, fill=(215, 225, 240))
    dr.text((16, 58), "青箭头 = 人施加的推力", font=f_sm, fill=(150, 215, 255))
    dr.text((16, 76), f"红箭头 = 手受负载（各 {LOAD_PER_HAND:.0f} N）", font=f_sm,
            fill=(255, 160, 140))

    # 右下角实时数值
    dr.rectangle([W - 168, H - 62, W - 8, H - 8], fill=(20, 24, 30))
    dr.text((W - 158, H - 56), f"人的推力 {mag_h:5.1f} N", font=f_sm, fill=(160, 220, 255))
    dr.text((W - 158, H - 36), f"负载合计 {2*LOAD_PER_HAND:.0f} N", font=f_sm,
            fill=(255, 160, 140))

    frames.append(np.array(img))

ren.close()
iio.mimwrite(OUT, frames, fps=FPS, quality=8)
print(f"  已写出 {OUT}  {len(frames)} 帧")
