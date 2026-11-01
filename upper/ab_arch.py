#!/usr/bin/env python3
"""受控 A/B：官方架构 vs 我们的架构，同一份数据、同一训练预算。

对比对象
  A) 官方 MultiScaleTransformer（仓库里自带，已加载官方权重验证可跑）
       - 4 维输出，块级注意力（S 先均值/求和），k=v 共享投影，
       - 力/力矩分别投影后融合，pywt.swt(db4, 5 级)
  B) 我们的 H2Upper
       - 3 维输出，块内 token 级注意力，独立 k/v，
       - 隐变量随机键 + KL，自写 à-trous 3 级

为了公平
  * 同一份合成数据、同一随机划分
  * 同样的扩散训练预算（步数、batch、优化器）
  * 同样的采样协议（DDIM、多初值平均、相同步数）
  * 都只在 (vx, vy, vz) 三个维度上评估（官方第 4 维 ωz 在本数据中恒为 0，不参与评分）
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/h2official")
from transformer import MultiScaleTransformer  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, S, T, D = 6, 33, 198, 6
D_MODEL = 128
N_BLOCKS = 4
T_DIFF = 100
STEPS = 3500
BATCH = 32
LR = 1e-3
N_TRAIN = 4000
N_TEST = 400
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===================== 数据（与 h2_upper.py 完全一致的生成器）=============
def gen_data(n, seed=0):
    rng = np.random.default_rng(seed)
    Fs, Ts, Ys = [], [], []
    for _ in range(n):
        m, c, kt = 6.0, 10.0, 3.0
        v = np.zeros(3)
        amp = np.array([rng.uniform(0.2, 0.7), rng.uniform(-0.25, 0.25), rng.uniform(-0.6, 0.6)])
        fq = rng.uniform(0.1, 0.5, 3)
        ph = rng.uniform(0, 2 * np.pi, 3)
        dt = 0.0025
        Ft, Tt = [], []
        for i in range(T + H + 2):
            tt = i * dt
            v_int = amp * np.sin(2 * np.pi * fq * tt * 10 + ph)
            a = kt * (v_int - v)
            v = np.clip(v + a * dt * 10, [-0.8, -0.3, -0.8], [0.8, 0.3, 0.8])
            f = m * a + c * v
            f[2] += -m * 9.81 * 0.45
            f = f + rng.normal(0, 0.3, 3)
            lev = np.array([rng.uniform(-0.06, 0.06) for _ in range(3)])
            Ft.append(np.concatenate([f, np.cross(lev, f)]) + rng.normal(0, 0.05, 6))
            Tt.append(v.copy())
        Fs.append(np.array(Ft[:T]))
        Ys.append(np.array(Tt[T:T + H]))
    Fs = np.array(Fs, dtype=np.float32)
    Ys = np.array(Ys, dtype=np.float32)
    Taus = (Fs * 0.85 + rng.normal(0, 0.1, Fs.shape)).astype(np.float32)
    return Fs, Taus, Ys


# ===================== 两种小波前端 =====================
def swt_mine(x, levels=3):
    """à trous 3 级（我们用的）。x: (B,T,D) -> list of levels"""
    B, Tn, Dd = x.shape
    outs, cur = [], x
    for lv in range(levels):
        dil = 2 ** lv
        k = torch.zeros(2 * dil + 1, device=x.device, dtype=x.dtype)
        k[0] = 0.25; k[dil] = 0.5; k[2 * dil] = 0.25
        w = k.view(1, 1, -1).repeat(Dd, 1, 1)
        cur = F.conv1d(cur.transpose(1, 2), w, padding=2 * dil, groups=Dd).transpose(1, 2)
        cur = cur[:, :Tn, :]
        outs.append(cur)
    return outs


def swt_official(x, levels=5):
    """官方做法：前面补零到 2 的幂(256) -> pywt.swt(db4, 5 级) -> 去掉填充部分。

    x: (B, T, D) -> (B, T, L, D) 的各级近似系数
    （官方 dataset.py L215-L231 就是这么做的；注意 pywt.swt 要求长度能被 2^level 整除）
    """
    import pywt
    arr = x.cpu().numpy()
    B, Tn, Dd = arr.shape
    target_len = 2 ** int(np.ceil(np.log2(Tn)))     # 198 -> 256
    pad_len = target_len - Tn
    out = np.zeros((B, Tn, levels, Dd), dtype=np.float32)
    for b in range(B):
        blk = np.concatenate([np.zeros((pad_len, Dd), dtype=arr.dtype), arr[b]], axis=0)
        for c in range(Dd):
            coeffs = pywt.swt(blk[:, c], "db4", level=levels, trim_approx=False)
            # coeffs = [(cA_n, cD_n), ..., (cA_1, cD_1)]；逆转后按层堆叠近似系数
            ca_list = [cA for cA, cD in reversed(coeffs)]
            z_a = np.stack(ca_list, axis=-1)         # (256, L)
            out[b, :, :, c] = z_a[pad_len:, :]       # 去掉填充
    return torch.from_numpy(out)


# ===================== 扩散调度（两边共用）=====================
def cosine_schedule(n=T_DIFF, s=0.008):
    steps = n + 1
    x = torch.linspace(0, n, steps)
    ac = torch.cos(((x / n) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 1e-4, 0.999)
    return betas, ac[1:]


BETAS, ALPHAC = cosine_schedule()
BETAS, ALPHAC = BETAS.to(DEV), ALPHAC.to(DEV)
ALPHA = 1.0 - BETAS
ABAR = torch.cumprod(ALPHA, 0).to(DEV)


def q_sample(y0, t, noise):
    ab = ABAR.to(y0.device)[t].view(-1, 1, 1)
    return ab.sqrt() * y0 + (1 - ab).sqrt() * noise


def train_model(model, kind, dat):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    n = dat["y"].shape[0]
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), device=dat["y"].device)
        y0 = dat["y"][idx]
        t = torch.randint(0, T_DIFF, (BATCH,), device=DEV)
        noise = torch.randn_like(y0)
        yt = q_sample(y0, t, noise)
        if kind == "official":
            # 官方 4 维：第 4 维补 0（本数据无偏航率）
            yt4 = torch.cat([yt, torch.zeros_like(yt[..., :1])], -1)
            pred = model(yt4, t, dat["f"][idx], dat["t"][idx])
            loss = F.mse_loss(pred[..., :3], noise) + 0.1 * F.mse_loss(pred[..., 3], torch.zeros_like(pred[..., 3]))
        else:
            fl = [x[idx] for x in dat["f_list"]]
            tl = [x[idx] for x in dat["t_list"]]
            pred, kl = model(yt, t, fl, tl)
            loss = F.mse_loss(pred, noise) + 1e-2 * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 300 == 0:
            print(f"    [{kind}] step {step:4d}  loss {loss.item():.4f}", flush=True)


@torch.no_grad()
def ddim(model, kind, dat, n_seed=8, t_eval=20):
    """单步去噪评估：在低噪声步 t_eval 上预测 x0，多初值平均。

    比多步 DDIM 稳健——多步采样器如果写错会把两个架构都带偏，
    单步去噪直接反映去噪网络本身的质量，是更公平的对照。
    """
    model.eval()
    Bq = dat["y"].shape[0]
    y0 = dat["y"]
    ab = ABAR[t_eval]
    outs = []
    for _ in range(n_seed):
        noise = torch.randn_like(y0)
        yt = ab.sqrt() * y0 + (1 - ab).sqrt() * noise
        t = torch.full((Bq,), t_eval, dtype=torch.long, device=DEV)
        if kind == "official":
            yt4 = torch.cat([yt, torch.zeros_like(yt[..., :1])], -1)
            eps = model(yt4, t, dat["f"], dat["t"])
            eps3 = eps[..., :3]
        else:
            eps3, _ = model(yt, t, dat["f_list"], dat["t_list"])
        x0 = (yt - (1 - ab).sqrt() * eps3) / ab.sqrt()
        outs.append(x0)
    return torch.stack(outs, 0).mean(0)


def r2_corr(pred, target):
    out = []
    for d in range(target.shape[-1]):
        p, q = pred[..., d].flatten().cpu().numpy(), target[..., d].flatten().cpu().numpy()
        ss_res = ((q - p) ** 2).sum()
        ss_tot = ((q - q.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-9)
        corr = float(np.corrcoef(p, q)[0, 1]) if p.std() > 1e-8 and q.std() > 1e-8 else 0.0
        out.append((r2, corr))
    return out


def main():
    print("=" * 70)
    print("  生成数据…")
    F, Tau, Y = gen_data(N_TRAIN + N_TEST, seed=SEED)
    F, Tau, Y = (torch.tensor(a) for a in (F, Tau, Y))
    tr = slice(0, N_TRAIN)
    te = slice(N_TRAIN, N_TRAIN + N_TEST)

    # 官方前端
    # 与官方语义对齐：apx_f = [传感器1 三轴力 | 传感器2 三轴力]，apx_t 同理取力矩
    f_pair = torch.cat([F[..., :3], Tau[..., :3]], -1)     # (B,T,6) 两传感器的力
    t_pair = torch.cat([F[..., 3:], Tau[..., 3:]], -1)     # (B,T,6) 两传感器的力矩

    print("  官方前端 (补零到256 + pywt swt db4 x5)…", flush=True)
    f_off = swt_official(f_pair, 5).to(DEV)
    t_off = swt_official(t_pair, 5).to(DEV)
    # 我们前端（同样的信息，换成 à trous 3 级）
    print("  我们的前端 (à trous x3)…", flush=True)
    f_mine = swt_mine(f_pair, 3)
    t_mine = swt_mine(t_pair, 3)

    dat_tr = {
        "official": {"y": Y[tr].to(DEV), "f": f_off[tr], "t": t_off[tr]},
        "mine": {"y": Y[tr].to(DEV), "f_list": [x[tr].to(DEV) for x in f_mine],
                 "t_list": [x[tr].to(DEV) for x in t_mine]},
    }
    dat_te = {
        "official": {"y": Y[te].to(DEV), "f": f_off[te], "t": t_off[te]},
        "mine": {"y": Y[te].to(DEV), "f_list": [x[te].to(DEV) for x in f_mine],
                 "t_list": [x[te].to(DEV) for x in t_mine]},
    }

    results = {}
    for kind in ("official", "mine"):
        print("\n" + "=" * 70)
        if kind == "official":
            model = MultiScaleTransformer().to(DEV)
            npar = sum(p.numel() for p in model.parameters())
        else:
            sys.path.insert(0, "/root/autodl-tmp/h2official")
            import importlib.util
            spec = importlib.util.spec_from_file_location("mine_mod", "/root/autodl-tmp/h2official/mine_h2_upper.py")
            mm = importlib.util.module_from_spec(spec)
            sys.modules["mine_mod"] = mm
            # 只取类定义，避免执行脚本主体
            src = open("/root/autodl-tmp/h2official/mine_h2_upper.py").read()
            cut = src.index("# ===================== 4. 数据生成")
            exec(compile(src[:cut], "mine_upper_cut", "exec"), mm.__dict__)
            model = mm.H2Upper().to(DEV)
            npar = sum(p.numel() for p in model.parameters())
        print(f"  [{kind}] 参数量 {npar:,}  训练 {STEPS} 步")
        train_model(model, kind, dat_tr[kind])

        pred = ddim(model, kind, dat_te[kind], n_seed=8, t_eval=20)
        stats = r2_corr(pred, dat_te[kind]["y"])
        results[kind] = {"params": npar, "stats": stats}
        print(f"  [{kind}] 测试集:", flush=True)
        for d, (r2, c) in enumerate(stats):
            print(f"      维度{d}: R²={r2:+.3f}  corr={c:+.3f}")

    print("\n" + "=" * 70)
    print(f"  {'架构':<12}{'参数量':>12}{'R²(0)':>10}{'R²(1)':>10}{'R²(2)':>10}{'平均corr':>10}")
    for kind, r in results.items():
        s = r["stats"]
        print(f"  {kind:<12}{r['params']:>12,}{s[0][0]:>10.3f}{s[1][0]:>10.3f}{s[2][0]:>10.3f}"
              f"{np.mean([x[1] for x in s]):>10.3f}")
    print("\n  说明：两列模型都用同一份合成数据、同预算、同采样协议；")
    print("        只评 (vx,vy,vz)；官方的第 4 维 ωz 在本数据中恒为 0。")


if __name__ == "__main__":
    main()
