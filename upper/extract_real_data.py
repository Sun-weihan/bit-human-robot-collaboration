#!/usr/bin/env python3
"""从官方数据集 zip 中提取训练所需的紧凑数据（本地执行）。

为什么要在本地剥离：WHAM 的 pkl 里 `verts`（778×6890×3 顶点）占了约 90% 体积，
而训练只需要 pose_world / trans_world / frame_ids。剥离后每条轨迹从 62 MB
降到约 0.25 MB，108 条总共 ~30 MB，上传很快。

输出目录结构（与本仓库 build_real_data2.py 兼容）：
    <out>/<轨迹名>/pose.npz          # pose_world, trans_world, frame_ids
    <out>/<轨迹名>/forces_mini40.csv # timestamp,fx,fy,fz,tx,ty,tz
    <out>/<轨迹名>/forces_mini45.csv
    <out>/<轨迹名>/img_ts.txt        # 图像文件名（含时间戳），每行一个
"""
import io
import os
import sys
import zipfile

import joblib
import numpy as np

ZIP = r"G:\数据集\3d_printing-20261031T071152Z-1-001等5项文件\extracted_data.zip"
OUT = r"E:\ROBOT C\carry\real_data_compact"

# 优先用这个 pkl（已验证 778 帧与图像数一致）；缺失时退回 output/ 下的
PREFERRED = "video_without_objects/wham_output.pkl"
FALLBACK = "video_without_objects/output/wham_output.pkl"


def main():
    os.makedirs(OUT, exist_ok=True)
    zf = zipfile.ZipFile(ZIP)
    names = zf.namelist()

    # 按轨迹分组（顶层两个目录 ex_data / ex_data.bak）
    trajs = {}
    for n in names:
        parts = n.split("/")
        if len(parts) < 4:
            continue
        # 实际结构: extracted_data/<ex_data|ex_data.bak>/<轨迹名>/...
        traj = f"{parts[0]}/{parts[1]}/{parts[2]}"
        rel = "/".join(parts[3:])
        trajs.setdefault(traj, {})[rel] = n

    print(f"  轨迹数: {len(trajs)}")
    ok = skip = 0
    for i, (traj, files) in enumerate(sorted(trajs.items()), 1):
        outdir = os.path.join(OUT, traj.replace("/", "__"))
        os.makedirs(outdir, exist_ok=True)
        pose_p = os.path.join(outdir, "pose.npz")

        # ---- 位姿 ----
        if not os.path.exists(pose_p):
            key = PREFERRED if PREFERRED in files else (FALLBACK if FALLBACK in files else None)
            if key is None:
                print(f"  [{i}/{len(trajs)}] {traj}: 无 wham pkl，跳过")
                skip += 1
                continue
            with zf.open(files[key]) as f:
                d = joblib.load(io.BytesIO(f.read()))
            # 不同轨迹的 pkl 结构可能不同：遍历找到含 pose_world 的那个 track
            trk = None
            if isinstance(d, dict):
                for kk in d:
                    vv = d[kk]
                    if isinstance(vv, dict) and "pose_world" in vv:
                        trk = vv
                        break
                if trk is None:
                    trk = d[list(d.keys())[0]]
            else:
                trk = d
            if not (isinstance(trk, dict) and "pose_world" in trk):
                print(f"  [{i}/{len(trajs)}] {traj}: pkl 结构未知，跳过")
                skip += 1
                continue
            np.savez_compressed(
                pose_p,
                pose_world=np.asarray(trk["pose_world"], dtype=np.float32),
                trans_world=np.asarray(trk["trans_world"], dtype=np.float32),
                frame_ids=np.asarray(trk["frame_ids"], dtype=np.int32),
            )

        # ---- 力 ----
        for sensor, outn in (("ati_mini40", "forces_mini40.csv"),
                             ("ati_mini45", "forces_mini45.csv")):
            dst = os.path.join(outdir, outn)
            if os.path.exists(dst):
                continue
            key = f"forces/{sensor}/{sensor}.txt"
            if key in files:
                with zf.open(files[key]) as f:
                    raw = f.read()
                with open(dst, "wb") as g:
                    g.write(raw)

        # ---- 图像时间戳 ----
        ts_p = os.path.join(outdir, "img_ts.txt")
        if not os.path.exists(ts_p):
            imgs = sorted(os.path.basename(v) for k, v in files.items()
                          if k.startswith("images/") and k.endswith(".jpg"))
            with open(ts_p, "w") as g:
                g.write("\n".join(imgs))

        ok += 1
        if i % 10 == 0 or i == len(trajs):
            print(f"  [{i}/{len(trajs)}] 完成 {ok}，跳过 {skip}")

    zf.close()
    # 统计体积
    total = 0
    for root, _, fs in os.walk(OUT):
        for f in fs:
            total += os.path.getsize(os.path.join(root, f))
    print(f"\n  输出: {OUT}")
    print(f"  轨迹 {ok} 条（跳过 {skip}），总大小 {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
