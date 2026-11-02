#!/usr/bin/env python3
"""用**真实人类的力轨迹**驱动闭环，测端到端意图实现能力。

为什么要这一版：
  原闭环里的"人"是合成的（恒定 fz=-15 N、零力矩），与真实数据分布不同。
  用它评测"真实数据训练的上层"是不公平的——实测双流层会输出反向指令（vx=-0.107）。
  只有用真实力轨迹驱动，上层与下层的对比才有意义。

做法：
  1. 从真实数据集取一条轨迹，按 50 Hz 机器人控制周期映射到 30 Hz 数据索引
  2. 每个控制步：喂该索引的 198 帧真实力窗口给上层 -> 得到速度指令 cmd
     同时把该窗口最后一帧的六轴力施加到机器人腕部（作为"人施的力"）
  3. 记录机器人实际速度，与真人速度标签比对

指标：
  cmd_err  上层预测 vs 真人速度的 RMSE      —— 上层的准确度
  vel_err  机器人实际 vs 真人速度的 RMSE    —— 端到端意图实现能力（关键指标）
  disp     机器人位移 / 真人对应的位移
"""
import sys
import traceback
from collections import deque

import numpy as np

H, S = 6, 33
DATASET = "/root/autodl-tmp/h2data/real_dataset.npz"


def main():
    policy_npz = sys.argv[1]
    upper_kind = sys.argv[2] if len(sys.argv) > 2 else "reg_fixed"
    traj_id = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    out_json = sys.argv[4] if len(sys.argv) > 4 else "/root/autodl-tmp/mj3/replay.json"
    cfg_name = sys.argv[5] if len(sys.argv) > 5 else "walk3_carry24"

    import json
    import torch

    sys.path.insert(0, "/root/autodl-tmp/mj3")
    sys.path.insert(0, "/root/autodl-tmp/h2official")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import importlib
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    def load_cfg(name):
        mod = importlib.import_module(f"legged_lab.envs.tienkung.{name}_cfg")
        for a in dir(mod):
            if a.endswith("EnvCfg") and not a.startswith("_"):
                return getattr(mod, a)
        raise SystemExit(f"找不到 {name}")

    # ---- 上层 ----
    if upper_kind.startswith("reg"):
        import st_moe_reg as R
        ck = torch.load(f"/root/autodl-tmp/h2official/seeds_reg/{upper_kind}_1.pt",
                        map_location=DEV, weights_only=False)
        umodel = R.RegModel(ck["mode"]).to(DEV)
        umodel.load_state_dict(ck["state"]); umodel.eval()
        print(f"  上层：直接回归 {ck['mode']}  R²={[round(x,3) for x in ck['r2']]}")
    elif upper_kind.startswith("diff"):
        from ds_upper import DualStreamUpper
        umodel = DualStreamUpper(f"/root/autodl-tmp/h2official/seeds_closed/{upper_kind[5:]}_1.pt")
        print(f"  上层：扩散 {upper_kind[5:]}")
    else:
        raise SystemExit("upper_kind 需以 reg 或 diff 开头")

    def infer(win198):
        """win198: (198,6) -> (3,)"""
        if upper_kind.startswith("reg"):
            import st_moe as SM
            x = torch.as_tensor(win198, dtype=torch.float32, device=DEV).reshape(1, H * S, 6)
            A = SM.swt_approx(x).reshape(1, H, S, *SM.swt_approx(x).shape[2:]).to(DEV)
            Sc = SM.scattering_features(x).reshape(1, H, S, 24).to(DEV)
            with torch.no_grad():
                return umodel(A, Sc)[0, 0].cpu().numpy()
        return umodel.sample(win198)

    # ---- 数据 ----
    d = np.load(DATASET)
    fa, ya, sizes = d["f"], d["y"], d["traj_sizes"]
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    if traj_id < 0:
        # 自动选平均 |vx| 最大的轨迹
        best, bi = -1, 0
        for i in range(len(sizes)):
            v = np.abs(np.asarray(ya[offs[i]:offs[i+1], 0, 0])).mean()
            if v > best: best, bi = v, i
        traj_id = bi
    i0, i1 = offs[traj_id], offs[traj_id + 1]
    F = fa[i0:i1]; Y = ya[i0:i1]
    print(f"  轨迹 {traj_id}  样本 {i1-i0}  平均|vx| {np.abs(Y[:,0,0]).mean():.3f} m/s")

    # ---- 环境 ----
    cfg = load_cfg(cfg_name)()
    try:
        cfg.domain_rand.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    except Exception:
        pass
    for ev in ("wrist_payload", "add_base_mass", "push_robot", "physics_material"):
        try:
            setattr(cfg.domain_rand.events, ev, None)
        except Exception:
            pass
    cfg.scene.num_envs = 1
    cfg.scene.terrain_type = "plane"
    cfg.scene.terrain_generator = None
    cfg.scene.seed = 42
    env = TienKungEnv(cfg, headless=True)
    obs, _ = env.get_observations()
    step_dt = env.step_dt
    bids, _ = env.robot.find_bodies(["wrist_roll_l_link", "wrist_roll_r_link"],
                                    preserve_order=True)

    p = np.load(policy_npz)
    W0, b0, W2, b2, W4, b4, W6, b6 = (p[k] for k in
                                      ("W0", "b0", "W2", "b2", "W4", "b4", "W6", "b6"))
    elu = lambda x: np.where(x > 0, x, np.expm1(np.clip(x, -30.0, 0.0)))

    def policy(o):
        h = elu(o @ W0.T + b0); h = elu(h @ W2.T + b2); h = elu(h @ W4.T + b4)
        return h @ W6.T + b6

    n = len(F)
    T = int(n / 30.0 / step_dt)
    rec_cmd, rec_vel, rec_hum, rec_cmdh = [], [], [], []
    for i in range(T):
        t = min(int(i * step_dt * 30.0), n - 1)
        win = np.asarray(F[t], dtype=np.float32)          # (198,6)
        human_v = np.asarray(Y[t, 0, :], dtype=np.float32)  # (3,)
        v = np.clip(infer(win), [-0.6, -0.3, -0.5], [0.6, 0.3, 0.5])
        try:
            env.command_generator.command[:, 0] = float(v[0])
            env.command_generator.command[:, 1] = float(v[1])
            if env.command_generator.command.shape[1] > 3:
                env.command_generator.command[:, 3] = 0.0
        except Exception:
            pass
        a = np.clip(policy(obs.cpu().numpy() if torch.is_tensor(obs) else obs), -100, 100)
        obs, _, _, _ = env.step(torch.as_tensor(a, dtype=torch.float32, device=env.device))

        rv = env.robot.data.root_lin_vel_w[0].cpu().numpy()
        # 把真实力施加到腕部（取窗口最后一帧）
        try:
            f6 = np.asarray(F[t][-1], dtype=np.float32)
            force = torch.zeros((env.num_envs, len(bids), 3), device=env.device)
            force[..., 0] = float(f6[0]); force[..., 1] = float(f6[1]); force[..., 2] = float(f6[2])
            env.robot.set_external_force_and_torque(force, torch.zeros_like(force), body_ids=bids)
            env.robot.write_data_to_sim()
        except Exception:
            pass

        rec_cmd.append(v[:2]); rec_vel.append(np.array([rv[0], rv[1]]))
        rec_hum.append(human_v[:2]); rec_cmdh.append(human_v[:2])

    rec_cmd = np.array(rec_cmd); rec_vel = np.array(rec_vel); rec_hum = np.array(rec_hum)
    res = {
        "upper": upper_kind, "traj": int(traj_id), "steps": int(T),
        "cmd_err": float(np.sqrt(((rec_cmd - rec_hum) ** 2).sum(1).mean())),
        "vel_err": float(np.sqrt(((rec_vel - rec_hum) ** 2).sum(1).mean())),
        "robot_disp": float(np.linalg.norm(rec_vel.sum(0) * step_dt)),
        "human_disp": float(np.linalg.norm(rec_hum.sum(0) * step_dt)),
        "cmd_mean": rec_cmd.mean(0).tolist(), "human_mean": rec_hum.mean(0).tolist(),
        "vel_mean": rec_vel.mean(0).tolist(),
    }
    json.dump(res, open(out_json, "w"))
    print("=" * 56)
    print(f"  上层 {upper_kind}  轨迹 {traj_id}  步数 {T}")
    print(f"    cmd_err  上层预测 vs 真人  = {res['cmd_err']:.4f} m/s")
    print(f"    vel_err  机器人实际 vs 真人 = {res['vel_err']:.4f} m/s")
    print(f"    机器人位移 {res['robot_disp']:.3f} m / 真人 {res['human_disp']:.3f} m")
    print(f"    cmd 均值 {np.round(res['cmd_mean'],3).tolist()}  "
          f"真人 {np.round(res['human_mean'],3).tolist()}")
    print("=" * 56)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
