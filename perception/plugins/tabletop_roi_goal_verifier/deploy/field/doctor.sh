#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi
export ROS_LOG_DIR="$ROOT/logs/ros" ROS_HOME="$ROOT/.ros" PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$ROOT/.cache" TMPDIR="$ROOT/.tmp"
export PYTHONPATH="$ROOT/deps:$ROOT/src/perception${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$ROOT/dds-local.xml" ]] || { echo "缺少经过校验的官方 dds-local.xml" >&2; exit 1; }
export FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/dds-local.xml"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset FASTDDS_BUILTIN_TRANSPORTS
RUNTIME_PYTHON=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["python"])' "$ROOT/runtime.json")
URL=$(python3 -c 'import json,sys,pathlib;p=pathlib.Path(sys.argv[1]);print(json.loads(p.read_text())["url"] if p.exists() else "http://127.0.0.1:18084")' "$ROOT/service.json")
exec "$RUNTIME_PYTHON" "$ROOT/bin/doctor.py" --config "$ROOT/config.json" --url "$URL" "$@"
