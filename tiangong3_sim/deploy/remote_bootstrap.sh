#!/usr/bin/env bash
# =============================================================================
# 天工 3.0 -> IsaacLab 2.3.2 服务器端引导脚本
#
# 用法：bash remote_bootstrap.sh [包目录] [模式]
#   模式： report | convert | test | all(默认)
#
# 为什么这样写（针对本实例实测，不是通用模板）：
#   * IsaacLab 2.3.2 的 isaaclab.sh 通过 extract_python_exe() 从 $CONDA_PREFIX
#     取 python。所以**必须先 conda activate**，否则会回退到 _isaac_sim/python.sh
#     的 kit python，行为与预期不同。
#   * SOP 规定：统一走 ./isaaclab.sh -p <脚本>；不要跑 isaaclab -i（会联网重装 ~10GB）。
#   * 首次启动要 1~2 分钟初始化 PhysX，属正常，不是卡死。
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODE="${2:-all}"

CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
ISAACLAB_PATH="${ISAACLAB_PATH:-/root/autodl-tmp/IsaacLab}"
PY_SCRIPT="$PKG_DIR/scripts/load_in_isaaclab.py"

fail() { echo ""; echo "!! 失败: $*" >&2; exit 1; }

echo "=============================================================="
echo " 天工 3.0 -> IsaacLab  [$MODE]"
echo "=============================================================="
echo "  包目录  : $PKG_DIR"
echo "  模式    : $MODE"

[ -f "$PY_SCRIPT" ] || fail "找不到 $PY_SCRIPT"
[ -d "$PKG_DIR/urdf" ] || fail "找不到 $PKG_DIR/urdf"
[ -d "$PKG_DIR/meshes_convex" ] || fail "找不到 $PKG_DIR/meshes_convex（URDF 用 ../meshes_convex/ 相对引用，不能拆散）"

# ---------- 环境体检 ----------
if [ "$MODE" = "report" ] || [ "$MODE" = "all" ]; then
  echo ""
  echo "---------- 环境体检 ----------"
  echo "hostname : $(hostname)"
  echo "os       : $( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -s)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
               --format=csv,noheader | sed 's/^/gpu      : /'
  else
    echo "gpu      : !! 没有 nvidia-smi"
  fi
  df -h / "$HOME" /root/autodl-tmp 2>/dev/null | sed 's/^/disk     : /'
fi

# ---------- conda 激活（必须在调用 isaaclab.sh 之前） ----------
CONDA_SH=""
for CSH in /root/miniconda3/etc/profile.d/conda.sh \
           /opt/conda/etc/profile.d/conda.sh \
           "$HOME/miniconda3/etc/profile.d/conda.sh"; do
  [ -f "$CSH" ] && { CONDA_SH="$CSH"; break; }
done
[ -n "$CONDA_SH" ] || fail "没找到 conda.sh，无法激活 $CONDA_ENV"
# shellcheck disable=SC1090
. "$CONDA_SH"

# 坑：conda 的 activate 脚本会引用未绑定变量，在 set -u 下会直接中止当前 shell。
# 之前这里还加了 2>/dev/null，把 "unbound variable" 吞掉，表现为"无声退出"。
# 所以：临时关掉 nounset，并且**不要**重定向 stderr。
set +u
conda activate "$CONDA_ENV"
RC=$?
set -u
if [ $RC -ne 0 ]; then
  echo "conda env list:" >&2
  conda env list >&2
  fail "conda activate $CONDA_ENV 失败（退出码 $RC）"
fi
echo ""
echo "conda    : $CONDA_ENV  ->  $(command -v python)"
echo "python   : $(python -V 2>&1)"

[ -f "$ISAACLAB_PATH/isaaclab.sh" ] || fail "找不到 $ISAACLAB_PATH/isaaclab.sh"
echo "isaaclab : $ISAACLAB_PATH  (VERSION=$(cat "$ISAACLAB_PATH/VERSION" 2>/dev/null || echo '?'))"
echo "isaacsim : $(cat /root/autodl-tmp/isaacsim/VERSION 2>/dev/null || echo '?')"

cd "$ISAACLAB_PATH" || fail "cd $ISAACLAB_PATH 失败"
export PYTHONIOENCODING=utf-8
# 坑：Isaac Sim 的 simulation_app.close() 会直接结束进程，跳过 Python 的 stdout
# 缓冲区刷新。输出重定向到管道/文件时是块缓冲，末尾的 print 会被整段吞掉
# （表现为"脚本 exit 0，但关键结果不见了"）。所以必须关掉缓冲。
export PYTHONUNBUFFERED=1

# ---------- convert ----------
if [ "$MODE" = "convert" ] || [ "$MODE" = "all" ]; then
  echo ""
  echo "---------- URDF -> USD 转换 ----------"
  echo "（首次约 1~3 分钟：要编译 PhysX / 材质缓存）"
  echo ""
  ./isaaclab.sh -p "$PY_SCRIPT" --headless --convert-only
  RC=$?
  [ $RC -ne 0 ] && fail "--convert-only 退出码 $RC"
  echo ""
  echo "生成的 USD："
  find "$PKG_DIR/usd" -type f \( -name '*.usd' -o -name '*.usda' \) 2>/dev/null | sed 's/^/  /'
fi

# ---------- test ----------
if [ "$MODE" = "test" ] || [ "$MODE" = "all" ]; then
  echo ""
  echo "---------- 载入 + 站立测试 ----------"
  ./isaaclab.sh -p "$PY_SCRIPT" --headless --steps 500
  RC=$?
  [ $RC -ne 0 ] && fail "站立测试退出码 $RC"
fi

echo ""
echo "=============================================================="
echo " 完成 [$MODE]"
echo "=============================================================="
