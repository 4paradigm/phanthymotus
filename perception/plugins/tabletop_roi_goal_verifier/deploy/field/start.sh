#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -f "$ROOT/runtime.json" ]] || { echo "缺少 runtime.json，请先安装" >&2; exit 1; }
[[ -f /opt/ros/humble/setup.bash ]] || { echo "未找到 /opt/ros/humble/setup.bash；未启动任何服务" >&2; exit 1; }
set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_LOG_DIR="$ROOT/logs/ros" ROS_HOME="$ROOT/.ros" PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$ROOT/.cache" TMPDIR="$ROOT/.tmp"
export PYTHONPATH="$ROOT/deps:$ROOT/src/perception${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$ROOT/bin/runner.py" start --root "$ROOT" --port "${ROI_PORT:-18084}"
