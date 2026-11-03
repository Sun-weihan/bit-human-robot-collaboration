#!/usr/bin/env python3
"""上下层闭环演示：真实人类力 → 上层意图推断 → 下层步态 → 机器人随人走。

与 bench_replay 的区别：
  bench_replay 只记录 cmd/vel/hum 用于统计；
  本脚本额外记录 root / qpos / ft（真实人力），可直接喂给 render_cocarry_human.py 出视频。

用法:
  python demo_closed_loop.py <lower_policy.npz> <cfg_name> <traj_id> <duration_s> <out.npz> [upper]
"""
import sys
import traceback

import numpy as np

H, S = 6, 33


def main():
    pol_path = sys.argv[1]
    cfg_name = sys.argv[2]
    traj_id = int(sys.argv[3])
    dur = float(sys.argv[4])
    out_path = sys.argv[5]
    upper_kind = sys.argv[6] if len(sys.argv) > 6 else "ff"

    import torch
    sys.path.insert(0, "/root/autodl-tmp/mj3")
    sys.path.insert(0, "/root/autodl-tmp/h2official")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    import importlib
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 上层：纯前馈（低通 + 导数），闭环稳定 ----
    import ff_split as FF
    ck = torch.load("/root/autodl-tmp/h2official/seeds_ff/ff_concat_1.pt",
                    map_location=DEV, weights_only=False)
    up = FF.Flat(len(ck["stats"][0])).to(DEV)
    up.load_state_dict(ck["state"]); up.eval()
    MU = np.array(ck["stats"][0], dtype=np.float32)
    SD = np.array(ck["stats"][1], dtype=np.float32)
    print(f"  上层 ff_concat R²={[round(x,3) for x in ck['r2']]}")

    def infer(win198):
        w = np.asarray(win198, dtype=np.float32)
        x = np.concatenate([FF.lowpass_feat(w), FF.deriv_feat(w)])
        xn = (x - MU) / SD
        with torch.no_grad():
            return up(torch.tensor(xn[None], device=DEV))[0].cpu().numpy()

    # ---- 数据 ----
    d = np.load("/root/autodl-tmp/h2data/real_dataset.npz")
    fa, ya, sizes = d["f"], d["y"], d["traj_sizes"]
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    i0, i1 = offs[traj_id], offs[traj_id + 1]
    F, Y = fa[i0:i1], ya[i0:i1]
    print(f"  轨迹 {traj_id}: {len(F)} 样本")

    # ---- 环境 ----
    def load_cfg(name):
        mod = importlib.import_module(f"legged_lab.envs.tienkung.{name}_cfg")
        for a in dir(mod):
            if a.endswith("EnvCfg") and not a.startswith("_"):
                return getattr(mod, a)
        raise SystemExit(f"找不到 {name}")

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
    env = TienKungEnv(cfg, headless=True)
    step_dt = env.step_dt
    bids, _ = env.robot.find_bodies(["wrist_roll_l_link", "wrist_roll_r_link"], preserve_order=True)

    p = np.load(pol_path)
    W0, b0, W2, b2, W4, b4, W6, b6 = (p[k] for k in ("W0","b0","W2","b2","W4","b4","W6","b6"))
    elu = lambda x: np.where(x > 0, x, np.expm1(np.clip(x, -30.0, 0.0)))

    def policy(o):
        h = elu(o @ W0.T + b0); h = elu(h @ W2.T + b2); h = elu(h @ W4.T + b4)
        return h @ W6.T + b6

    n_steps = int(dur / step_dt)
    env.reset(torch.arange(env.num_envs, device=env.device))
    obs = env.get_observations()[0]

    roots, qposes, cmds, vels, hums, fts = [], [], [], [], [], []
    for i in range(n_steps):
        t = min(int(i * step_dt * 30.0), len(F) - 1)
        win = np.asarray(F[t], dtype=np.float32)
        human_v = np.asarray(Y[t, 0, :], dtype=np.float32)
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

        rp = env.robot.data.root_pos_w[0].cpu().numpy()
        rq = env.robot.data.root_quat_w[0].cpu().numpy()
        rv = env.robot.data.root_lin_vel_w[0].cpu().numpy()
        jp = env.robot.data.joint_pos[0].cpu().numpy()
        # 真实人力施加到腕部
        f6 = np.asarray(F[t][-1], dtype=np.float32)
        try:
            force = torch.zeros((env.num_envs, len(bids), 3), device=env.device)
            force[..., 0] = float(f6[0]); force[..., 1] = float(f6[1]); force[..., 2] = float(f6[2])
            env.robot.set_external_force_and_torque(force, torch.zeros_like(force), body_ids=bids)
            env.robot.write_data_to_sim()
        except Exception:
            pass
        q = rq; w_, x_, y_, z_ = q
        yaw = np.arctan2(2*(w_*z_ + x_*y_), 1 - 2*(y_*y_ + z_*z_))
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        vb = np.array([c*rv[0] - s_*rv[1], s_*rv[0] + c*rv[1]], dtype=np.float32)
        roots.append(np.concatenate([rp, rq]))
        qposes.append(jp)
        cmds.append(v.copy())
        vels.append(vb)
        hums.append(human_v.copy())
        fts.append(f6.copy())
        if i % 50 == 0:
            print(f"    {i}/{n_steps}  cmd={v[0]:+.2f}  vx={vb[0]:+.2f}  人={human_v[0]:+.2f}", flush=True)

    roots = np.array(roots); qposes = np.array(qposes)
    np.savez_compressed(out_path,
                        root=roots, qpos=qposes, step_dt=np.float32(step_dt),
                        joint_names=np.array(env.robot.data.joint_names),
                        cmd=np.array(cmds), vel=np.array(vels), hum=np.array(hums),
                        ft=np.array(fts))
    print(f"  已保存 {out_path}  {len(roots)} 步")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
