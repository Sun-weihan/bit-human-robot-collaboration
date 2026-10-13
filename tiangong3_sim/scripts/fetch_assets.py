"""从天工 3.0 URDF 仓库下载 URDF + 凸包网格。

为什么写这个脚本
----------------
1. raw.githubusercontent.com 在本机网络下不通，只能走 GitHub API 的 base64 接口。
2. 仓库里的 URDF 用 `package://tiangong3_0_V2_urdf/meshes/` 引用网格，
   但实际目录名是 `tiangong3_urdf/`，且凸包在 `meshes_convex/` —— 必须重写路径。
3. 原始网格 136 MB、凸包仅 3.2 MB。Isaac Lab 的 GPU 并行仿真只需要凸包（碰撞体）。

本脚本做三件事
--------------
A. 下载 urdf/tiangong3.urdf
B. 下载 meshes_convex/ 下全部 39 个 STL
C. 生成一份「路径已修正」的 URDF：tiangong3_isaaclab.urdf
   - package://tiangong3_0_V2_urdf/meshes/  ->  ../meshes_convex/
   - 这样 URDF 与网格的相对位置就自洽了，Isaac Lab 直接能读

用法
----
    python scripts/fetch_assets.py

依赖：仅标准库（urllib）。
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
REPO = "Open-X-Humanoid/TienKung_URDF"
API = f"https://api.github.com/repos/{REPO}/contents"
SRC_DIR = "tiangong3_urdf"          # 仓库里的实际目录名
URDF_SRC = f"{SRC_DIR}/urdf/tiangong3.urdf"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = PROJECT_ROOT / "urdf"
MESH_DIR = PROJECT_ROOT / "meshes_convex"

# URDF 里写的包名（与实际目录名不符，必须改）
URDF_PKG_OLD = "package://tiangong3_0_V2_urdf/meshes/"
URDF_PKG_NEW = "../meshes_convex/"

# 代理（本机 7890 时通时不通，脚本会先试代理再试直连）
PROXY = "http://127.0.0.1:7890"

TIMEOUT = 60


def _request(url: str, use_proxy: bool) -> bytes:
    """发一个 GET 请求，可选代理。"""
    if use_proxy:
        handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
        opener = urllib.request.build_opener(handler)
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    req = urllib.request.Request(url, headers={"User-Agent": "tienkung-fetch/1.0"})
    with opener.open(req, timeout=TIMEOUT) as resp:
        return resp.read()


def get(url: str, retries: int = 3) -> bytes:
    """带重试与代理回退的下载。"""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        for use_proxy in (True, False):
            try:
                data = _request(url, use_proxy)
                via = "proxy" if use_proxy else "direct"
                print(f"      [{via}] OK", end="")
                return data
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if attempt < retries:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"下载失败: {url}\n  最后一次错误: {last_err}")


def fetch_api_json(path: str) -> list | dict:
    raw = get(f"{API}/{path}")
    return json.loads(raw.decode("utf-8"))


def fetch_file_content(path: str) -> bytes:
    """通过 API 取文件内容（base64 解码）。"""
    meta = fetch_api_json(path)
    if isinstance(meta, dict) and "content" in meta:
        return base64.b64decode(meta["content"])
    raise RuntimeError(f"无法获取 {path}: {str(meta)[:150]}")


# ---------------------------------------------------------------------------
# 步骤 A：URDF
# ---------------------------------------------------------------------------
def step_fetch_urdf() -> str:
    print("\n[A] 下载 URDF ...")
    URDF_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_file_content(URDF_SRC)
    out = URDF_DIR / "tiangong3.urdf"
    out.write_bytes(data)
    text = data.decode("utf-8", errors="replace")
    print(f"\n      已保存 {out}  ({len(data)} B, {len(text.splitlines())} 行)")
    return text


# ---------------------------------------------------------------------------
# 步骤 B：凸包网格
# ---------------------------------------------------------------------------
def step_fetch_meshes() -> int:
    print("\n[B] 下载凸包网格 (meshes_convex) ...")
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    items = fetch_api_json(f"{SRC_DIR}/meshes_convex")
    if not isinstance(items, list):
        raise RuntimeError(f"目录列举失败: {str(items)[:150]}")

    stl_items = [i for i in items if i["name"].lower().endswith(".stl")]
    print(f"      共 {len(stl_items)} 个 STL，开始下载")

    ok = 0
    for idx, item in enumerate(stl_items, 1):
        name = item["name"]
        dest = MESH_DIR / name
        if dest.exists() and dest.stat().st_size == item["size"]:
            print(f"      [{idx:>2}/{len(stl_items)}] {name:<42} 已存在，跳过")
            ok += 1
            continue

        print(f"      [{idx:>2}/{len(stl_items)}] {name:<42} {item['size']/1024:>7.1f} KB", end=" ")
        try:
            blob = fetch_file_content(f"{SRC_DIR}/meshes_convex/{name}")
            dest.write_bytes(blob)
            print(" -> 完成")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f" -> 失败: {exc}")

    return ok


# ---------------------------------------------------------------------------
# 步骤 C：路径修正
# ---------------------------------------------------------------------------
def step_patch_urdf(urdf_text: str) -> Path:
    print("\n[C] 修正 URDF 中的网格路径 ...")
    print(f"      旧路径: {URDF_PKG_OLD}<文件名>.STL")
    print(f"      新路径: {URDF_PKG_NEW}<文件名>.STL")

    n = urdf_text.count(URDF_PKG_OLD)
    patched = urdf_text.replace(URDF_PKG_OLD, URDF_PKG_NEW)

    if n == 0:
        print("      [警告] 未找到待替换的旧路径，配置可能已过期")

    out = URDF_DIR / "tiangong3_isaaclab.urdf"
    out.write_text(patched, encoding="utf-8")
    print(f"      已替换 {n} 处，保存为 {out}")

    # 校验：新路径引用的文件是否都存在
    missing = []
    for m in re.finditer(r'filename="\.\./meshes_convex/([^"]+)"', patched):
        f = MESH_DIR / m.group(1)
        if not f.exists():
            missing.append(m.group(1))

    if missing:
        print(f"      [警告] 缺少 {len(missing)} 个网格文件: {missing[:5]}")
    else:
        print("      ✓ 所有引用的网格文件均在位")

    return out


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def report(urdf_path: Path) -> None:
    text = urdf_path.read_text(encoding="utf-8")
    print("\n" + "=" * 68)
    print("汇总")
    print("=" * 68)
    print(f"  仓库          : https://github.com/{REPO}")
    print(f"  URDF          : {urdf_path.relative_to(PROJECT_ROOT)}")
    print(f"  网格目录      : {MESH_DIR.relative_to(PROJECT_ROOT)}/")

    links = len(re.findall(r"<link\s+name=", text))
    joints = len(re.findall(r"<joint\s+name=", text))
    revolute = len(re.findall(r'type="revolute"', text))
    fixed = len(re.findall(r'type="fixed"', text))
    inertials = len(re.findall(r"<inertial>", text))

    print(f"\n  结构统计:")
    print(f"    link      : {links}")
    print(f"    joint     : {joints}  (revolute {revolute}, fixed {fixed})")
    print(f"    inertial  : {inertials}   <- 有惯性才能做动力学仿真")

    if inertials < links:
        print(f"    [警告] {links - inertials} 个 link 缺少 <inertial>，Isaac Lab 可能报错")

    total = sum(f.stat().st_size for f in MESH_DIR.glob("*.stl"))
    print(f"\n  凸包网格总量  : {total/1024/1024:.2f} MB ({len(list(MESH_DIR.glob('*.stl')))} 个)")
    print(f"  （原始精细网格 136 MB，凸包仅此，提速约 42 倍）")

    print("\n  下一步：在 Isaac Lab 中加载")
    print(f"    USD: {urdf_path.name}")
    print("    见 README_天工3_IsaacLab.md")


def main() -> int:
    print("=" * 68)
    print("天工 3.0 URDF 资材获取（Isaac Lab 版）")
    print("=" * 68)

    urdf_text = step_fetch_urdf()
    n_ok = step_fetch_meshes()
    urdf_path = step_patch_urdf(urdf_text)
    report(urdf_path)

    total = len(list(MESH_DIR.glob("*.stl")))
    if n_ok < total:
        print(f"\n[注意] 有 {total - n_ok} 个网格未下载成功，重跑本脚本即可续传。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
