# 额外奖励项：让手臂按步态相位自然摆动，并压住肩部外张
#
# 背景：3.0 用纯 PPO 训练（无 AMP 示范数据），策略会把双臂当配重乱摆，
#       实测 shoulder_roll 摆到 1.56 rad（外张 89°）。2 Lite 因为 AMP 学的是
#       人类示范动作，所以没这个问题。
#
# 做法：不引入 AMP，直接用奖励约束手臂运动学：
#   1. arm_swing_periodic : 肩 pitch 跟随步态相位做正弦摆动，
#                           且与同侧腿反相（人走路就是这样）
#   2. arm_cross_body     : 肩 roll/yaw 不能外张，保持贴近身体

from __future__ import annotations

import math

import torch

from legged_lab.mdp.rewards import gait_clock


def arm_swing_periodic(
    env,
    amp_sh: float = 0.175,
    amp_el: float = 0.35,
    sh_center: float = 0.05,
    el_center: float = -0.45,
    std: float = 0.30,
):
    """手臂摆动：肩 pitch + 肘 pitch 同时按步态相位正弦摆动，与同侧腿反相。

    参数按 2 Lite 的实测步态标定（见 docs/天工3.0_步调优记录.md）：
        2 Lite 肩 pitch 峰峰 20.6°  -> 振幅 0.175 rad
        2 Lite 肘 pitch 峰峰 36-45° -> 振幅 0.35 rad
    关键点：**摆臂主要来自肘关节屈伸**，只摆肩会显得像两根棍子（僵硬）。
    """
    robot = env.scene["robot"]
    phase = env.gait_phase                      # (N, 2)：[左腿, 右腿]

    def idx(key):
        ids, _ = robot.find_joints([key], preserve_order=True)
        return ids[0]

    q_sh_l = robot.data.joint_pos[:, idx("shoulder_pitch_l_joint")]
    q_sh_r = robot.data.joint_pos[:, idx("shoulder_pitch_r_joint")]
    q_el_l = robot.data.joint_pos[:, idx("elbow_pitch_l_joint")]
    q_el_r = robot.data.joint_pos[:, idx("elbow_pitch_r_joint")]

    # 左臂跟右腿相位，右臂跟左腿相位（人走路就是反相）
    s_l = torch.sin(2.0 * math.pi * phase[:, 1])
    s_r = torch.sin(2.0 * math.pi * phase[:, 0])

    t_sh_l = sh_center - amp_sh * s_l
    t_sh_r = sh_center - amp_sh * s_r
    t_el_l = el_center - amp_el * s_l
    t_el_r = el_center - amp_el * s_r

    err = ((q_sh_l - t_sh_l) ** 2 + (q_sh_r - t_sh_r) ** 2
           + (q_el_l - t_el_l) ** 2 + (q_el_r - t_el_r) ** 2)
    return torch.exp(-err / (2.0 * std ** 2))

def arm_cross_body(
    env,
    asset_cfg=None,
):
    """肩 roll/yaw 偏离默认值就惩罚，防止手臂像鸟翅膀一样外张。"""
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    q = robot.data.joint_pos[:, ids]
    q0 = robot.data.default_joint_pos[:, ids]
    return torch.sum(torch.abs(q - q0), dim=1)


def track_vy_exp(
    env,
    std: float = 0.20,
):
    """显式约束侧向速度跟踪。

    实测：指令 vy=0，但机器人每秒侧移 0.11-0.12 m，走不直。
    原有的 track_lin_vel_xy_exp 把 vx/vy 合在一起用 exp 奖励（std=0.5），
    对 0.1 m/s 这种小误差几乎没梯度，所以单独给 vy 一个更陡的指数项。
    """
    robot = env.scene["robot"]
    vy = robot.data.root_lin_vel_b[:, 1]
    cmd_vy = env.command_generator.command[:, 1]
    return torch.exp(-torch.square(vy - cmd_vy) / (std ** 2))

def gait_foot_clearance(
    env,
    height_target: float = 0.12,
    delta_t: float = 0.02,
):
    """摆动相奖励脚离地高度。

    实测问题：天工3.0 抬脚行程只有 6.9-9.6 cm，而 2 Lite 是 17-19 cm，
    等于在"拖脚走"而不是迈步，看起来就不流畅。
    现有三条步态奖励只惩罚「摆动相踩地」「支撑相脚滑」，
    没有任何一条鼓励「摆动相把脚抬起来」，所以策略学会了贴地滑行。
    """
    robot = env.scene["robot"]
    feet = env.feet_body_ids                      # [左, 右]
    foot_z = robot.data.body_pos_w[:, feet, 2]    # 世界系 z
    h = torch.clamp(foot_z / height_target, 0.0, 1.0)

    left_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    return left_mask * h[:, 0] + right_mask * h[:, 1]

def stride_length(env, v_target: float = 0.85, delta_t: float = 0.02):
    """摆动相奖励脚的水平速度 -> 步幅更大。

    实测：天工3.0 步频 1.25 Hz 与 2 Lite 一致，但速度只有 0.38 m/s（2 Lite 0.54），
    说明差距在"步幅"而不是"步频"。固定步频下，摆动相脚的水平速度直接决定步幅。
    """
    robot = env.scene["robot"]
    feet = env.feet_body_ids
    v = robot.data.body_lin_vel_w[:, feet, :2]          # (N, 2, 2) 水平速度
    spd = torch.norm(v, dim=-1)                          # (N, 2)
    s = torch.clamp(spd / v_target, 0.0, 1.0)

    left_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    return left_mask * s[:, 0] + right_mask * s[:, 1]

def carry_pose(
    env,
    target=None,
    std: float = 0.35,
):
    """持箱姿态：双臂前伸、肘部弯曲，使双手位于身体前方箱子的两侧。

    之前策略沿用了 2 Lite 的摆臂步态（手臂自然垂在身侧），
    但协同搬运时必须改为"抬着箱子"的姿态，否则箱子会穿过身体、
    且腕部负载的力臂为零（负载感不到）。

    目标姿态（rad）：
        shoulder_pitch  +0.55   上臂前抬约 32°
        shoulder_roll   ±0.18   略微外展，避开躯干
        elbow_pitch     -1.05   肘弯 60°
        wrist_*          0      手腕中立（自然托住箱子）
    """
    robot = env.scene["robot"]
    if target is None:
        # 由 solve_carry_pose.py 用正运动学反解得到（误差 0.0 cm）：
        # 手落在躯干前方 0.35 m、左右 ±0.22 m（间距 0.44 m）、肩下 0.05 m。
        # 注意 shoulder_pitch 是**负**的（约 -11°）：肘部弯 54° 后前臂朝前，
        # 上臂反而略向后。之前按 "+0.85 前抬" 设置，手会跑到身体后方。
        target = {
            "shoulder_pitch_l_joint": -0.196,
            "shoulder_pitch_r_joint": -0.193,
            "shoulder_roll_l_joint": 0.020,
            "shoulder_roll_r_joint": -0.001,
            "shoulder_yaw_l_joint": -0.143,
            "shoulder_yaw_r_joint": 0.107,
            "elbow_pitch_l_joint": -0.946,
            "elbow_pitch_r_joint": -0.946,
            "wrist_pitch_.*_joint": 0.0,
            "wrist_roll_.*_joint": 0.0,
        }
    err = 0.0
    n = 0
    for key, val in target.items():
        ids, _ = robot.find_joints([key], preserve_order=True)
        if len(ids) == 0:
            continue
        q = robot.data.joint_pos[:, ids]
        err = err + torch.square(q - val).sum(-1)
        n += len(ids)
    return torch.exp(-err / (2.0 * std ** 2 * max(n, 1)))

def carry_pose_l2(
    env,
    target=None,
    roll_weight: float = 3.0,
):
    """持箱姿态的 L2 惩罚（对偏离给线性梯度）。

    carry_pose 用指数形式，偏离很远时梯度趋于 0：
    实测肩 roll 卡在 25°（目标 1°）不动，因为 exp 奖励对它已经没有梯度。
    这里对肩 roll 单独给平方惩罚，保证远处也有明确的修正方向。
    """
    robot = env.scene["robot"]
    if target is None:
        target = {"shoulder_roll_l_joint": 0.02, "shoulder_roll_r_joint": -0.02}
    pen = 0.0
    for key, val in target.items():
        ids, _ = robot.find_joints([key], preserve_order=True)
        if len(ids) == 0:
            continue
        q = robot.data.joint_pos[:, ids]
        pen = pen + roll_weight * torch.square(q - val).sum(-1)
    return -pen

def box_stability(
    env,
    w_vertical: float = 2.0,
    w_relative: float = 1.5,
    w_tilt: float = 0.5,
):
    """箱子稳定性：搬东西时箱子应该走直线、保持水平。

    人搬箱子时手臂像悬挂系统——躯干上下起伏由腿和腰吸收，
    手臂反向补偿，使箱子在世界系里几乎走直线。

    实测当前策略：箱子竖直晃动 19.4 cm 峰峰，且
    「手臂自身抖动 / 躯干抖动」= 1.58，说明手臂在**放大**躯干起伏，
    正好和人相反。所以这里直接罚箱子（双手中点）的运动：
        w_vertical * |v_z|²        箱子上下晃
        w_relative * |v_rel|²      手臂相对躯干的运动（手臂自身抖）
        w_tilt     * |ω_hand|²     箱子翻转
    """
    robot = env.scene["robot"]
    ids, _ = robot.find_bodies(
        ["wrist_roll_l_link", "wrist_roll_r_link"], preserve_order=True
    )
    p = robot.data.body_pos_w[:, ids, :]            # (N,2,3)
    v = robot.data.body_lin_vel_w[:, ids, :]
    w = robot.data.body_ang_vel_w[:, ids, :]
    mid_v = v.mean(1)
    rel_v = mid_v - robot.data.root_lin_vel_w
    rel_p = p.mean(1) - robot.data.root_pos_w
    cost = (w_vertical * torch.square(mid_v[:, 2])
            + w_relative * torch.square(rel_v).sum(-1)
            + w_tilt * torch.square(w.mean(1)).sum(-1)
            + 0.5 * w_relative * torch.square(rel_p[:, 2]))
    # 返回正的代价值，配合负权重
    return cost


def arm_vel_l2(env, keys=None):
    """手臂关节速度惩罚：抑制抖动。

    实测肘关节速度峰值 6.01 rad/s、手臂加速度 RMS 20.8 rad/s²，
    远超正常搬运动作。直接罚关节角速度。
    """
    robot = env.scene["robot"]
    if keys is None:
        keys = ["shoulder_.*_joint", "elbow_.*_joint", "wrist_.*_joint"]
    ids, _ = robot.find_joints(keys, preserve_order=True)
    if len(ids) == 0:
        return torch.zeros(env.num_envs, device=robot.device)
    # 返回正的代价值，配合负权重使用（返回负值 + 负权重会变成奖励抖动）
    return torch.square(robot.data.joint_vel[:, ids]).sum(-1)