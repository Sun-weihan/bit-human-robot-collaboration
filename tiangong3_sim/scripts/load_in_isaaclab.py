"""在 Isaac Lab 中加载天工 3.0：URDF -> USD 转换 + 场景 + 站立测试。

运行（服务器上，务必先激活 conda 环境再走 isaaclab.sh）：

    conda activate env_isaaclab
    cd /root/autodl-tmp/IsaacLab
    ./isaaclab.sh -p <包目录>/scripts/load_in_isaaclab.py --headless --convert-only
    ./isaaclab.sh -p <包目录>/scripts/load_in_isaaclab.py --headless --steps 500

两个必须遵守的顺序约束（都实际踩过，不是风格问题）
--------------------------------------------------
1. **必须先 `conda activate env_isaaclab`。**
   IsaacLab 2.3.2 的 `isaaclab.sh` 用 `extract_python_exe()` 从 `$CONDA_PREFIX`
   取 python；环境没激活时它会回退到 `_isaac_sim/python.sh` 的 kit python。

2. **必须先启动 `AppLauncher`，再导入 `isaaclab.sim.converters`。**
   USD 的 Python 绑定 `pxr` 来自 Isaac Sim 运行时。若在 App 启动前导入转换器，
   会在 `isaaclab/utils/mesh.py` 的 `from pxr import Usd, UsdGeom` 处报
   `ModuleNotFoundError: No module named 'pxr'`。
   官方 `scripts/tools/convert_urdf.py` 也是这个顺序（第 75 行起 App，第 87 行起导入）。

关于"站立测试"的合理预期
------------------------
`fix_base=False` 的浮动基座人形机器人在固定 PD 增益下**本来就会下蹲/失稳**，
这不是 URDF 的问题。所以本脚本把判定拆成两层：
  * **资产完整性**（刚体数 / 关节数 / 总质量）—— 这才是"URDF 导入是否正确"的判据；
  * **沉降表现** —— 只如实汇报高度变化，不伪装成通过/失败。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# 服务器若 LANG=C / POSIX，stdout 默认 ASCII，中文 print 会崩
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------------------
# 路径与参数
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = PROJECT_ROOT / "urdf" / "tiangong3_isaaclab.urdf"
USD_DIR = PROJECT_ROOT / "usd"
USD_FILE_NAME = "tiangong3_isaaclab.usd"

ROOT_LINK = "pelvis"

# 与 URDF 实测一致（39 links / 38 joints，其中 31 revolute + 7 fixed）
EXPECTED_BODIES = 39
EXPECTED_JOINTS = 31
EXPECTED_MASS_KG = 65.82

STIFFNESS = 200.0
DAMPING = 20.0
SIM_DT = 0.005


def parse_args() -> argparse.Namespace:
    # 只导入 AppLauncher：它是安全的，不触碰 pxr
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description="加载天工 3.0 到 Isaac Lab")
    p.add_argument("--convert-only", action="store_true", help="只做 URDF->USD 转换后退出")
    p.add_argument("--force-convert", action="store_true", help="即使 USD 已存在也重新转换")
    p.add_argument("--steps", type=int, default=500, help="沉降测试的仿真步数")
    p.add_argument("--print-joints", action="store_true", help="打印刚体/关节名单与限位后退出")
    # 由 AppLauncher 提供 --headless / --device / --enable_cameras 等
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# URDF -> USD（只能在 AppLauncher 启动后调用）
# ---------------------------------------------------------------------------
def convert_urdf_to_usd() -> Path:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.utils.assets import check_file_path

    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"找不到 URDF: {URDF_PATH}\n"
            f"urdf/ 与 meshes_convex/ 必须保持同级（URDF 内用 ../meshes_convex/ 引用网格）。"
        )

    urdf_abs = str(URDF_PATH.resolve())
    # check_file_path 语义：1=本地存在, 0=不存在, 2=在 Nucleus 上
    if not check_file_path(urdf_abs):
        raise ValueError(f"URDF 路径无效: {urdf_abs}")

    USD_DIR.mkdir(parents=True, exist_ok=True)

    # 字段名依据本机实测的 IsaacLab 2.3.2 源码 urdf_converter_cfg.py：
    #   usd_file_name / collider_type / make_instanceable 均**存在**（未废弃）
    #   输出路径 = os.path.join(usd_dir, usd_file_name)，即平铺在 usd/ 下
    cfg = UrdfConverterCfg(
        asset_path=urdf_abs,
        usd_dir=str(USD_DIR.resolve()),
        usd_file_name=USD_FILE_NAME,
        # 人形机器人：基座必须浮动，否则会被钉在空中
        fix_base=False,
        # 必须 False：URDF 里有 7 个 fixed 关节，合并会改变 link 结构
        merge_fixed_joints=False,
        force_usd_conversion=True,
        # 关节驱动：位置控制 + PD 增益
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=STIFFNESS,
                damping=DAMPING,
            ),
        ),
    )

    print("[1/4] 转换 URDF -> USD")
    print(f"      输入: {urdf_abs}")
    converter = UrdfConverter(cfg)
    print(f"      输出: {converter.usd_path}")
    return Path(converter.usd_path)


# ---------------------------------------------------------------------------
# 场景 + 信息 + 沉降测试
# ---------------------------------------------------------------------------
def run_simulation(usd_path: Path, steps: int, print_joints: bool) -> None:
    # 下面这些导入同样必须在 App 启动之后
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass

    print("[2/4] 构建场景")

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        """一个最小的天工 3.0 场景。

        注意：InteractiveScene 只接受 AssetBaseCfg 子类。GroundPlaneCfg / DomeLightCfg
        是 **Spawner**，直接当场景项写会报
        `ValueError: Unknown asset config type for ground`。
        必须用 AssetBaseCfg(prim_path=..., spawn=...) 包一层
        （与官方 scripts/tutorials/02_scene/create_scene.py 第 55 行一致）。
        """

        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
        )

        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),  # 略微抬高，自然落到地面
                joint_pos={".*": 0.0},
            ),
            # 用 IsaacLab 的显式执行器模型，而不是依赖 USD 自带的 drive
            actuators={
                "all_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=STIFFNESS,
                    damping=DAMPING,
                ),
            },
        )

    sim = SimulationContext(SimulationCfg(dt=SIM_DT))
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 1.0))

    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()

    robot: Articulation = scene["robot"]

    # ---- 资产信息：这才是 URDF 导入是否正确的判据 ----
    n_bodies = len(robot.body_names)
    n_joints = len(robot.joint_names)
    mass = robot.root_physx_view.get_masses().sum().item()

    print("\n[3/4] 资产完整性核对")
    print(f"      根 link（URDF 里的 base_link）: {ROOT_LINK}")
    print(f"      刚体数 bodies : {n_bodies:>3}  (URDF 期望 {EXPECTED_BODIES})"
          f"  {'OK' if n_bodies == EXPECTED_BODIES else 'MISMATCH'}")
    print(f"      关节数 joints : {n_joints:>3}  (URDF 期望 {EXPECTED_JOINTS})"
          f"  {'OK' if n_joints == EXPECTED_JOINTS else 'MISMATCH'}")
    print(f"      总质量        : {mass:>7.2f} kg  (URDF 期望 {EXPECTED_MASS_KG})"
          f"  {'OK' if abs(mass - EXPECTED_MASS_KG) < 0.5 else 'MISMATCH'}")

    print("\n      刚体列表:")
    for i, name in enumerate(robot.body_names):
        print(f"        [{i:>2}] {name}")

    print("\n      关节列表（含限位）:")
    limits = robot.root_physx_view.get_dof_limits()[0]
    for i, name in enumerate(robot.joint_names):
        lo, hi = limits[i][0].item(), limits[i][1].item()
        print(f"        [{i:>2}] {name:<28} [{lo:+.3f}, {hi:+.3f}] rad")

    if print_joints:
        print("\n--print-joints 指定，退出。")
        return

    # ---- 沉降测试：如实汇报，不粉饰 ----
    print(f"\n[4/4] 重力沉降测试（{steps} 步 @ dt={SIM_DT}s）")
    print("      说明：浮动基座人形机器人在固定 PD 增益下本来就会下蹲/失稳，")
    print("            这一步只用来确认仿真能稳定跑下去，不作为 URDF 对错的判据。")

    target = robot.data.default_joint_pos.clone()
    robot.set_joint_position_target(target)

    heights: list[float] = []
    for i in range(steps):
        robot.write_data_to_sim()
        sim.step()
        scene.update(SIM_DT)

        pos = robot.data.root_pos_w[0]
        h = pos[2].item()
        heights.append(h)

        if i % max(1, steps // 5) == 0:
            dev = robot.data.joint_pos[0].abs().max().item()
            print(f"      step {i:>4}: 根高度 {h:6.3f} m | 关节最大偏离 {dev:.4f} rad")

    h0, h1 = heights[0], heights[-1]
    print(f"\n      起始根高度 : {h0:.3f} m")
    print(f"      结束根高度 : {h1:.3f} m")
    print(f"      最低根高度 : {min(heights):.3f} m")
    print(f"      仿真是否跑完 : 是（{steps} 步无异常退出）")

    print("\n" + "=" * 62)
    ok_asset = (n_bodies == EXPECTED_BODIES and n_joints == EXPECTED_JOINTS
                and abs(mass - EXPECTED_MASS_KG) < 0.5)
    if ok_asset:
        print(" 结论：URDF -> USD 导入正确（39 刚体 / 31 关节 / 65.82 kg 全部对上）")
    else:
        print(" 结论：资产规模与 URDF 不符，请核对 merge_fixed_joints 与网格路径")
    print("=" * 62)
    print("\n下一步：真实训练请用 manager-based 环境注册任务，")
    print("        或接入 Open-X-Humanoid/TienKung-Lab 的走跑任务。")


def main() -> int:
    args = parse_args()

    # ---- 关键：先起 App，再导入任何触碰 pxr 的东西 ----
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        usd_path = USD_DIR / USD_FILE_NAME
        if args.force_convert or not usd_path.exists():
            usd_path = convert_urdf_to_usd()
        else:
            print(f"[1/4] USD 已存在，跳过转换: {usd_path}", flush=True)

        if args.convert_only:
            print(f"\n--convert-only 指定，退出。USD: {usd_path}", flush=True)
        else:
            run_simulation(usd_path, args.steps, args.print_joints)
    except BaseException:
        # 必须在这里打印 traceback。
        # Isaac Sim 的 simulation_app.close() 会直接结束进程；如果只靠 finally 里 close，
        # 异常信息连同非零退出码会被一起吞掉，表现为"exit 0 但没有结果"。
        traceback.print_exc()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
