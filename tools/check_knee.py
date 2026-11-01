#!/usr/bin/env python3
"""量膝部（和大腿）的横向间距——这才是"双腿交叉穿模"的真实指标。

之前一直用"两脚横向距离"和"脚是否越过中线"判断，
但走路时摆动脚本来就会经过支撑脚的横向位置，这是正常步态，
所以那个指标长期误报（一直是 3-4%）。

真正会碰擦的是膝部：knee_pitch 连杆在两侧大腿末端，
站距收窄时这里最先接触。
"""
import sys

import mujoco
import numpy as np

TRAJ = sys.argv[1]
POL = sys.argv[2]
XML = sys.argv[3] if len(sys.argv) > 3 else "/root/autodl-tmp/mj3/tiangong3_carry.xml"

tj = np.load(TRAJ)
pol = np.load(POL)
root, jpos = tj["root"], tj["qpos"]
m2i = pol["mujoco_to_isaac_idx"].astype(int)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
adr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])

# 找膝关节与大腿连杆
def bid(*names):
    for n in names:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        if i >= 0:
            return i
    return -1

kl = bid("knee_pitch_l_link", "knee_l_link")
kr = bid("knee_pitch_r_link", "knee_r_link")
hl = bid("hip_roll_l_link", "hip_l_link")
hr = bid("hip_roll_r_link", "hip_r_link")
print(f"  膝 body id: 左 {kl} 右 {kr}； 髋 body id: 左 {hl} 右 {hr}")

y_knee, y_hip, dist_knee = [], [], []
for k in range(len(root)):
    d.qpos[:] = 0.0
    d.qpos[0:3] = root[k, 0:3]
    d.qpos[3:7] = root[k, 3:7]
    d.qpos[adr] = jpos[k][m2i]
    mujoco.mj_forward(m, d)
    qw = d.qpos[3:7].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, qw)
    R = R.reshape(3, 3)
    if kl >= 0 and kr >= 0:
        rel = R.T @ (d.xpos[kl] - d.xpos[kr])
        y_knee.append(rel[1])
        dist_knee.append(np.linalg.norm(d.xpos[kl] - d.xpos[kr]))
    if hl >= 0 and hr >= 0:
        rel2 = R.T @ (d.xpos[hl] - d.xpos[hr])
        y_hip.append(rel2[1])

y_knee = np.array(y_knee)
dist_knee = np.array(dist_knee)
print(f"\n  膝部横向间距: 均值 {y_knee.mean()*100:+.1f} cm  "
      f"范围 [{y_knee.min()*100:+.1f}, {y_knee.max()*100:+.1f}] cm")
print(f"  膝部最小间距: {dist_knee.min()*100:.1f} cm   "
      f"（< 8 cm 视为可能接触）")
neg = (y_knee < 0).mean() * 100
print(f"  膝部左右顺序交叉帧占比: {neg:.1f}%")
if len(y_hip):
    y_hip = np.array(y_hip)
    print(f"  髋部横向间距: 均值 {y_hip.mean()*100:+.1f} cm  "
          f"最小 {y_hip.min()*100:+.1f} cm")
print(f"\n  >>> {'膝部存在交叉' if neg > 2 else '膝部无交叉'}")
