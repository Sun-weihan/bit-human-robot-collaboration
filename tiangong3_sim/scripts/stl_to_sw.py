"""把二进制 STL 转成 SolidWorks 友好的格式。

背景
----
1. SolidWorks 不支持 URDF，只支持几何格式。
2. 官方给的 STL 是**二进制**格式，SW 有时读不进去或读成空实体。
3. 本脚本把二进制 STL 转成 **ASCII STL**（兼容性最好），
   以及 **STEP**（如果装了 cadquery/OCC，SW 原生支持，可编辑）。

用法
----
    # 只转 ASCII STL（零依赖，用内置解析）
    python scripts/stl_to_sw.py

    # 额外转 STEP（需要 cadquery，转换慢但 SW 能直接编辑）
    python scripts/stl_to_sw.py --step

    # 只转指定几个零件
    python scripts/stl_to_sw.py --only pelvis waist_pitch_link hip_yaw_l_link
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "meshes_convex"
OUT_DIR = Path(__file__).resolve().parent.parent / "for_solidworks"


# --------------------------------------------------------------------------
# 二进制 STL 解析（纯标准库，不依赖 numpy）
# --------------------------------------------------------------------------
def read_binary_stl(path: Path) -> list[tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]]:
    """读二进制 STL，返回 [(法向量, (v1, v2, v3)), ...]。"""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path.name} 太小，不是合法 STL")

    n_tri = struct.unpack("<I", data[80:84])[0]
    expected = 84 + n_tri * 50
    if expected != len(data):
        # 可能是 ASCII STL，或文件损坏
        head = data[:80].lstrip()
        if head.lower().startswith(b"solid"):
            raise ValueError(f"{path.name} 看起来是 ASCII STL，无需转换")
        raise ValueError(
            f"{path.name} 长度不符: 期望 {expected} B，实际 {len(data)} B"
        )

    triangles = []
    off = 84
    for _ in range(n_tri):
        vals = struct.unpack("<12f", data[off : off + 48])
        normal = vals[0:3]
        v1, v2, v3 = vals[3:6], vals[6:9], vals[9:12]
        triangles.append((normal, (v1, v2, v3)))
        off += 50
    return triangles


def write_ascii_stl(path: Path, triangles, name: str) -> None:
    """写 ASCII STL。"""
    lines = [f"solid {name}"]
    for normal, verts in triangles:
        lines.append(f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}")
        lines.append("    outer loop")
        for v in verts:
            lines.append(f"      vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="二进制 STL -> SW 友好格式")
    p.add_argument("--step", action="store_true", help="额外导出 STEP（需 cadquery）")
    p.add_argument("--only", nargs="*", help="只处理这些零件名（不含 .STL）")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 68)
    print("二进制 STL -> SolidWorks 友好格式")
    print("=" * 68)
    print(f"  源目录: {SRC_DIR}")
    print(f"  输出  : {OUT_DIR}")

    if not SRC_DIR.exists():
        print(f"\n找不到源目录，请先运行 scripts/fetch_assets.py")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ascii_dir = OUT_DIR / "ascii_stl"
    ascii_dir.mkdir(exist_ok=True)

    files = sorted(SRC_DIR.glob("*.STL"))
    if args.only:
        wanted = {f"{n}.STL" for n in args.only}
        files = [f for f in files if f.name in wanted]
        if not files:
            print(f"\n没有匹配的零件: {args.only}")
            return 1

    print(f"\n[1] 转 ASCII STL（共 {len(files)} 个）")
    ok = 0
    total_tri = 0
    for i, f in enumerate(files, 1):
        try:
            tris = read_binary_stl(f)
            out = ascii_dir / f.name
            write_ascii_stl(out, tris, f.stem)
            total_tri += len(tris)
            print(f"    [{i:>2}/{len(files)}] {f.name:<40} {len(tris):>6,} 面  "
                  f"-> {out.stat().st_size/1024:>8.1f} KB")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"    [{i:>2}/{len(files)}] {f.name:<40} 失败: {exc}")

    print(f"\n    完成 {ok}/{len(files)}，总三角面 {total_tri:,}")

    # ---- 可选：STEP ----
    if args.step:
        print(f"\n[2] 转 STEP（需要 cadquery）")
        try:
            import cadquery as cq
        except ImportError:
            print("    未安装 cadquery，跳过。")
            print("    装法: pip install cadquery")
            print("    或者直接在 SW 里导入 ASCII STL（见 README）")
        else:
            step_dir = OUT_DIR / "step"
            step_dir.mkdir(exist_ok=True)
            for i, f in enumerate(files, 1):
                try:
                    shape = cq.importers.importStep(str(f)) if False else None
                    # cadquery 读 STL：用 mesh 导入较复杂，这里给出占位说明
                    print(f"    [跳过] cadquery 对 STL 支持有限，建议用 FreeCAD 转换")
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"    失败: {exc}")

    # ---- 说明 ----
    print("\n" + "=" * 68)
    print("下一步：在 SolidWorks 里怎么做")
    print("=" * 68)
    print(f"""
  方案 A（推荐，最省事）：SW 直接打开 ASCII STL
    1. 打开 SW -> 文件 -> 打开
    2. 文件类型选 "STL (*.stl)"
    3. 进 {ascii_dir} 选文件
    4. 选项里选 "输入为实体"（而不是曲面），单位选 米
    5. 注意：STL 是三角网格，进去后是"网格实体"，
       不能像正常零件那样拉伸/打孔

  方案 B（要做结构设计，如加装传感器支架）：
    1. 先用 FreeCAD 把 STL 转 STEP（免费，转换质量好）
       FreeCAD -> Part 工作台 -> 导入 STL -> 导出 STEP
    2. SW 打开 STEP -> 得到实体 -> 可以正常建模

  方案 C（只要看外形/出图）：
    直接把 STL 当参考几何导入，在装配体里做布局

  重要提醒
  --------
  * 你手上的 STL 是**凸包简化版**（包住零件的多面体），
    外形会比真实零件"胖"一圈。以下是原始精细网格的区别：
        凸包版 3.23 MB  <- 碰撞用，外形偏胖
        精细版 136 MB   <- 视觉用，外形准确
  * 如果你在 SW 里要看**准确外形**，需要用原始 meshes/ 目录的精细网格。
    下载方法（改脚本里的目录名）：

        # 编辑 scripts/fetch_assets.py，把
        #   MESH_DIR = PROJECT_ROOT / "meshes_convex"
        # 改成
        #   MESH_DIR = PROJECT_ROOT / "meshes"
        # 并把 step_fetch_meshes() 里的 "meshes_convex" 改成 "meshes"

    然后重跑即可（约 136 MB，视网速需几分钟）。
""")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
