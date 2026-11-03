#!/usr/bin/env python3
"""维度分流模型的闭环推理模块。

模型结构（split_correct）：
    平移支路：低通力 + 速度历史 → (vx, vy)
    转向支路：力的导数 + 速度历史 → wz

部署要点：
  * 速度历史用**机器人自己的**（体坐标系），不是真人的——这是唯一可得的
  * 归一化统计量从训练集重算（与训练时一致），不保存也能复现
"""
import collections
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/h2official")
import dim_split as DS  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "/root/autodl-tmp/h2data/real_dataset.npz"


class SplitUpper:
    def __init__(self, ckpt):
        ck = torch.load(ckpt, map_location=DEV, weights_only=False)
        self.model = DS.DimSplit(0, 0, 0, correct=(ck["mode"] == "split_correct"))
        # 先建好再载入（DimSplit 的输入维度要从数据推）
        self._stats = None
        self.ckpt = ckpt
        self.meta = {k: ck.get(k) for k in ("mode", "seed", "params", "r2")}

    def _build(self):
        """从训练集推出输入维度与归一化统计量"""
        d = np.load(DATA)
        f, y, sizes, names = d["f"], d["y"], d["traj_sizes"], d["traj_names"]
        tr_t, _ = DS.split_trajs(names, sizes)
        Y = np.asarray(y, dtype=np.float32)
        LP, DV, VH = [], [], []
        off = 0
        n = 0
        for i, s in enumerate(sizes):
            s = int(s)
            if i not in tr_t:
                off += s; continue
            for j in range(s):
                gi = off + j
                w = np.asarray(f[gi], dtype=np.float32)
                LP.append(DS.lowpass_feat(w))
                DV.append(DS.deriv_feat(w))
                VH.append(Y[[max(gi - k, off) for k in range(DS.CTX, 0, -1)], 0, :].reshape(-1))
                n += 1
                if n >= 4000:      # 4000 个样本足够估计统计量
                    break
            off += s
            if n >= 4000:
                break
        LP = np.stack(LP); DV = np.stack(DV); VH = np.stack(VH)
        self._stats = [(LP.mean(0), LP.std(0) + 1e-6),
                       (DV.mean(0), DV.std(0) + 1e-6),
                       (VH.mean(0), VH.std(0) + 1e-6)]
        ck = torch.load(self.ckpt, map_location=DEV, weights_only=False)
        self.model = DS.DimSplit(LP.shape[1], DV.shape[1], VH.shape[1],
                                 correct=(ck["mode"] == "split_correct")).to(DEV)
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    @torch.no_grad()
    def sample(self, win198, vhist):
        """win198: (198,6); vhist: (CTX*3,) 机器人近 3 帧体坐标系速度"""
        if self._stats is None:
            self._build()
        w = np.asarray(win198, dtype=np.float32)
        lp = (DS.lowpass_feat(w) - self._stats[0][0]) / self._stats[0][1]
        dv = (DS.deriv_feat(w) - self._stats[1][0]) / self._stats[1][1]
        vh = (np.asarray(vhist, dtype=np.float32) - self._stats[2][0]) / self._stats[2][1]
        out = self.model(torch.tensor(lp[None], device=DEV),
                         torch.tensor(dv[None], device=DEV),
                         torch.tensor(vh[None], device=DEV))
        return out[0].cpu().numpy()


if __name__ == "__main__":
    up = SplitUpper(sys.argv[1] if len(sys.argv) > 1 else
                    "/root/autodl-tmp/h2official/seeds_split/split_correct_1.pt")
    d = np.load(DATA)
    f, y = d["f"], d["y"]
    Y = np.asarray(y, dtype=np.float32)
    for i in range(3):
        vh = Y[[max(i - k, 0) for k in range(DS.CTX, 0, -1)], 0, :].reshape(-1)
        p = up.sample(f[i], vh)
        print(f"  样本 {i}: 预测 {np.round(p,3).tolist()}  真值 {np.round(Y[i,0,:],3).tolist()}")
