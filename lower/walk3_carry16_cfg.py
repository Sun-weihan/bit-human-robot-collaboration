# 天工3.0 AMP 步态训练 —— 用 AMPPPO + 自建示范数据
#
# 为什么：3.0 没有官方 AMP 示范数据，纯 PPO 训出的手臂乱摆（肩 roll 摆到 1.5 rad）。
# 2 Lite 的自然步态正是 AMP 学人类示范得来的。这里给 3.0 合成一份人类风格步态当示范。
#
# 关键改动：
#   1. AMPLoader 的维度常量 20/20/12 -> 26/26/12（3.0 的臂 7+7、腿 6+6，腰部不入 AMP）
#   2. 指向自己生成的示范数据 motion_amp_expert3/walk3.txt
#   3. runner/algorithm 换回 AmpOnPolicyRunner + AMPPPO
#   4. 手臂偏差惩罚恢复原值（-0.2），交给 AMP 来塑形，而不是硬压

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import CuboidCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.sim import MassPropertiesCfg
from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.mdp.carry_rewards import (arm_swing_periodic, track_vy_exp, gait_foot_clearance, stride_length, carry_pose, carry_pose_l2, box_stability, arm_vel_l2, feet_no_cross)

from legged_lab.envs.tienkung.walk3_cfg import (
    TienKung3WalkAgentCfg,
    TienKung3WalkFlatEnvCfg,
)

# ---- 1. 改 AMPLoader 维度（必须在任何 AMPLoader 实例化之前执行）----
from rsl_rl.utils.motion_loader import AMPLoader

AMP_JOINT = 20            # 右臂4 + 左臂4 + 右腿6 + 左腿6（与环境 find_joints 一致）
AMP_EE = 12               # 4 个末端点 × xyz

AMPLoader.JOINT_POS_SIZE = AMP_JOINT
AMPLoader.JOINT_VEL_SIZE = AMP_JOINT
AMPLoader.END_EFFECTOR_POS_SIZE = AMP_EE
AMPLoader.JOINT_POSE_START_IDX = 0
AMPLoader.JOINT_POSE_END_IDX = AMP_JOINT
AMPLoader.JOINT_VEL_START_IDX = AMP_JOINT
AMPLoader.JOINT_VEL_END_IDX = 2 * AMP_JOINT
AMPLoader.END_POS_START_IDX = 2 * AMP_JOINT
AMPLoader.END_POS_END_IDX = 2 * AMP_JOINT + AMP_EE    # = 52

MOTION_FILE = "legged_lab/envs/tienkung/datasets/motion_amp_expert3/walk3.txt"
NUM_DOF_3 = 29


# ===== 搬运箱：作为 SceneCfg 的声明字段 =====
from legged_lab.envs.base.base_config import BaseSceneCfg


@configclass
class CarrySceneCfg(BaseSceneCfg):
    """在基础场景上增加一个搬运箱刚体。

    箱子是场景里的真实刚体（可被渲染、可查询位姿），
    但设为 kinematic + 关闭重力：由脚本每步同步到双手中点。
    机器人承受的负载仍以腕部外力形式施加（论文 Eq.16）。
    """

    carry_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/carry_box",
        spawn=CuboidCfg(
            size=(0.22, 0.50, 0.20),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=MassPropertiesCfg(mass=2.0),
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.55, 0.28),
                roughness=0.55,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.85)),
    )


@configclass
class TienKung3Carry16EnvCfg(TienKung3WalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # AMP 用自己生成的 3.0 示范数据（64 列），显示也用同一份避免维度错配
        self.amp_motion_files_display = [MOTION_FILE]

        # ---- 锁腰：腰部关节动作置零，保持默认姿态 ----
        # 实测 waist_roll 摆到 50°（±25°），是"上体左右晃"的直接原因；
        # 软惩罚（-1.0）压不住，改用硬约束。真实行走腰部基本锁死。
        # 腰部逐步放开：0=锁死（学步态阶段）-> 0.3=小幅活动（共搬微调）-> 1=完全放开
        # 上一版放开到 0.3，waist_roll 立刻从 10° 飙到 47°（上一轮刚修掉的问题复发），
# 且腰部带着上半身摆会让手臂看起来"左右晃"。学步态阶段必须锁住。
        # 放开要留到共搬微调阶段（腰部那时才需要承重），且按 0.1 -> 0.2 -> 0.3 逐步来。
        self.robot.waist_action_scale = 0.0

        # 肩 roll 硬钳位：软惩罚反复失效（-1.0→88°、-3.0→38°、-3.8→105°），
        # 直接限制相对默认姿态的偏移为 ±0.35 rad（±20°，与 2 Lite 的 17-24° 对齐）
        # 2 Lite 肩 roll 上限是 24°，钳到 ±20° 偏紧、牺牲了速度；放宽到 ±25°
        self.robot.shoulder_roll_limit = 0.44

        # 手腕硬钳位：软惩罚 -3.0 只把 56.6° 压到 40.8°，不够。
        # 真人走路手腕只动几度，直接钳到 ±0.15 rad（约 ±8.6°）
        self.robot.wrist_limit = 0.15

        # 把场景换成带箱子的子类：先构造新实例，再把父类已配好的字段搬过来
        _old = self.scene
        _new = CarrySceneCfg()
        for _k, _v in _old.__dict__.items():
            setattr(_new, _k, _v)
        self.scene = _new

        # ---- 搬运时双臂基本静止 ----
        # 实测手臂仍在摆（肩 pitch 22°、肘 35° 峰峰），箱子跟着晃，不像在搬东西。
        # 真实搬运时手臂相对躯干固定，靠身体带着箱子走。
        # 用硬钳位把 8 个手臂关节锁在反解出的持箱姿态 ±0.12 rad（约 ±7°）
        self.robot.arm_pose_clamp = 0.04

        # ---- 侧移时禁止双腿交叉（穿模）----
        # feet_y_distance 只管"两脚间距"，两脚同时跨过中线时仍满足，
        # 于是侧移时双腿交叉、互相穿模。这里直接惩罚"左脚跑到右脚右边"。
        self.reward.feet_no_cross = RewTerm(
            func=feet_no_cross,
            weight=-60.0,      # -3.0 实测无效（占 0.07%）；-25 仍剩 3.4% 交叉；再加一档
            params={"margin": 0.02},
        )
        self.robot.arm_pose_target = {
            "shoulder_pitch_l_joint": -0.196,
            "shoulder_pitch_r_joint": -0.193,
            "shoulder_roll_l_joint": 0.020,
            "shoulder_roll_r_joint": -0.001,
            "shoulder_yaw_l_joint": -0.143,
            "shoulder_yaw_r_joint": 0.107,
            "elbow_pitch_l_joint": -0.946,
            "elbow_pitch_r_joint": -0.946,
        }


        # ================= H2-COMPACT 下层：负载自适应 =================
        # 论文关键增量：在两侧 wrist 施加 z 向随机外力 f ~ U[-3,+15] N
        # （+15 N 约等于单侧承担 1.5 kg 的等效向下力）
        # 载荷以「箱子重心」为作用点：采样箱子质量与重心偏移，
        # 再按杠杆关系分配到两只手，合力恒等于箱子重力、作用点即重心。
        # （论文 Eq.16 是对两腕独立采样 U[-3,15] N；这里给出等价的、语义更明确的模型）
        self.domain_rand.events.wrist_payload = EventTerm(
            func=mdp.apply_box_payload_force,
            mode="reset",
            params={
                "mass_range": (0.0, 3.0),        # 论文：payload 0-3 kg
                "com_offset_range": (-0.06, 0.06),
                "hand_separation": 0.45,
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["wrist_roll_l_link", "wrist_roll_r_link"]
                ),
            },
        )

        # 地面摩擦随机化（论文 [0.1, 1.25]，这里取较温和的 [0.4, 1.0] 以保证能训起来）
        self.domain_rand.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.1, 1.25),   # 论文 Table III
                "dynamic_friction_range": (0.1, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            },
        )

        # 附加质量 [-1.0, +3.0] kg（论文值）
        self.domain_rand.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "mass_distribution_params": (-1.0, 3.0),
                "operation": "add",
            },
        )

        # 每 10-15 秒推一次（论文 5 s 太频繁，先放宽保证收敛）
        self.domain_rand.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 5.0),      # 论文 Table III: Push interval 5 s
            params={
                "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},   # 论文: Max push velocity 1.5 m/s
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ---- 步态规律性与直线性强化 ----
        # 实测问题：步频变异系数 24%（不流畅）、横向 vy=-0.116 m/s（不走直）
        self.reward.gait_feet_frc_perio.weight = 2.2
        self.reward.gait_feet_spd_perio.weight = 2.2
        self.reward.gait_feet_frc_support_perio.weight = 1.4
        self.reward.feet_slide.weight = -0.5          # 减少脚底打滑
        # 站距奖励：原实现的 y_vel_flag 阈值是 0.1，**侧移时该奖励被完全关闭**，
        # 导致侧移时没有任何东西维持站距 -> 双腿收拢 -> 膝部碰擦（实测 9.5cm，膝部视觉重叠）。
        # 这里把阈值放到 10（始终生效），并把目标站距从 0.299 加大到 0.34。
        self.reward.feet_y_distance.weight = -8.0
        self.reward.feet_y_distance.params = {"vel_threshold": 10.0, "target": 0.34}
        self.reward.ang_vel_xy_l2.weight = -0.10       # 抑制 roll/pitch 角速度
        # 平滑性：2 Lite 高频能量 0.6%，我们 1.6%，加大关节加速度惩罚
        self.reward.dof_acc_l2.weight = -2.5e-6

        # ---- 显式驱动摆臂幅度：肩 pitch 跟随步态相位正弦摆动 ----
        # AMP 示范幅度加大到 51° 但策略只跟到 37°/20°，说明 AMP 权重不足以塑形。
        # 这里用显式奖励直接给正弦目标（与同侧腿反相，符合人走路）。
        # ---- 持箱姿态：双臂前伸抬箱，替代自然垂臂摆 ----
        # 之前沿用摆臂步态，手臂垂在身侧 -> 箱子会穿过身体、腕部力臂为零
        # ---- 抱箱稳定性：箱子走直线、手臂不抖 ----
        # 实测箱子竖直晃动 19.4 cm、手臂抖动是躯干的 1.58 倍（人应该相反）
        self.reward.box_stability = RewTerm(
            func=box_stability,
            weight=-0.5,
            params={"w_vertical": 2.0, "w_relative": 1.5, "w_tilt": 0.5},
        )
        self.reward.arm_vel_l2 = RewTerm(
            func=arm_vel_l2,
            weight=-0.02,
        )

        # 肩 roll 单独用 L2：指数奖励在 25° 处已无梯度，加平方项保证收敛
        self.reward.carry_roll_l2 = RewTerm(
            func=carry_pose_l2,
            weight=1.0,
            params={"roll_weight": 3.0},
        )

        self.reward.carry_pose = RewTerm(
            func=carry_pose,
            weight=4.0,        # 实测 1.2 只能达到目标的 30-40%，必须加重
            params={"std": 0.35},
        )

        # ---- 步幅：步频已与 2 Lite 一致(1.25Hz)，差距在步幅 ----
        self.reward.stride = RewTerm(
            func=stride_length,
            weight=0.6,
            params={"v_target": 0.85},
        )

        # ---- 抬脚高度：实测 3.0 只有 6.9-9.6 cm，2 Lite 是 17-19 cm（拖脚走）----
        self.reward.foot_clearance = RewTerm(
            func=gait_foot_clearance,
            weight=0.8,
            params={"height_target": 0.15},
        )

        # 持箱时不做前后摆臂：摆臂奖励与 carry_pose 直接冲突，置零
        self.reward.arm_swing = RewTerm(
            func=arm_swing_periodic,
            weight=0.0,
            params={"amp_sh": 0.28, "amp_el": 0.30, "std": 0.35},   # 按真人：肩 32°、肘 31°
        )

        # ---- 显式抑制侧向漂移 ----
        # 实测 vy = -0.106 m/s（指令 0），总的 xy 跟踪奖励梯度太弱，单独加一项
        self.reward.track_vy = RewTerm(
            func=track_vy_exp,
            weight=1.0,
            params={"std": 0.20},
        )

        # ---- 禁止手臂左右张，只留前后摆 ----
        # 实测肩 roll 摆到 100°（右）/42°（左），yaw 27°，而示范数据里 roll/yaw 恒为 0。
        # AMP 权重压低后约束不住，所以用显式惩罚禁掉 roll/yaw，
        # 前后摆（shoulder_pitch）留给 AMP 塑形。
        # 按 2 Lite 实测标定：它的肩 roll 本身就摆 17-24°，所以不能压到 9°
        # 惩罚降到 -1.0，只约束明显外张，不压死正常摆动
        # 侧向自由度守恒：腰锁死后右臂接管侧向平衡（roll 88°）；惩罚 -4.5 又压到 8°。
        # 2 Lite 无腰部关节、肩 roll 摆 17-24°，故目标 ~20°，按实测插值取 -3.8。
        # 强惩罚只留给肩 roll/yaw —— 这是"左右张"的来源，必须硬压
        self.reward.joint_deviation_arms.weight = -2.5
        self.reward.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["shoulder_roll_.*_joint"],   # 肩 yaw 放开：真人摆臂里它有 23-32°，压死会显得僵
        )

        # 手部（肘偏/腕 pitch/腕 roll）改为温和惩罚，允许自然抖动
        # 用户反馈"手部关节僵硬"：原来它们和肩回到同一个 -3.8 惩罚里被锁死了
        # 肘偏允许自然活动
        self.reward.joint_deviation_hands = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["elbow_yaw_.*_joint"])},
        )

        # 手腕单独收紧：实测腕 pitch 摆幅 56.6°，真人走路手腕只动几度，
        # 摆太大显得"手在甩"。注意：只改这一项，不动 dof_acc_l2
        #（上一版同时把 dof_acc_l2 放大 4 倍，导致回合从 990 崩到 105 步）
        self.reward.joint_deviation_wrist = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-3.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["wrist_pitch_.*_joint", "wrist_roll_.*_joint"],
                )
            },
        )

        # ---- 抑制躯干左右晃：腰部此前没有任何偏差惩罚 ----
        # 实测 waist_roll 峰峰 50°，策略把它当免费平衡关节用；
        # 正常人行走腰部几乎锁死（几度以内）。
        self.reward.joint_deviation_waist = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                )
            },
        )
        # 髋 roll 也收一收（实测 18-27°，正常约 5°）
        self.reward.joint_deviation_hip.weight = -0.5


@configclass
class TienKung3Carry16AgentCfg(TienKung3WalkAgentCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "walk3_carry16"
        self.max_iterations = 30000

        # ---- 换回 AMP ----
        self.runner_class_name = "AmpOnPolicyRunner"
        self.algorithm.class_name = "AMPPPO"
        self.amp_motion_files = [MOTION_FILE]
        self.amp_num_preload_transitions = 200000
        self.amp_reward_coef = 0.20
        self.amp_task_reward_lerp = 0.80
        self.amp_discr_hidden_dims = [1024, 512, 256]
        self.min_normalized_std = [0.05] * NUM_DOF_3

        # ---- 从 walk3_amp 的 checkpoint warm start（保留已学到的步态风格）----
        self.resume = True
        self.load_run = "resume_seed"
        self.load_checkpoint = "model_.*.pt"
        self.max_iterations = 30000
