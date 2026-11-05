#!/usr/bin/env bash
# docker exec does not inherit the ROS environment sourced by Core's CMD.
set -e
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash
FLOW_TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLOW_PYTHON='/work/.venv/bin/python'
if [ ! -x "$FLOW_PYTHON" ]; then
    FLOW_PYTHON="$(command -v python3)"
fi
exec "$FLOW_PYTHON" -B "$FLOW_TOOLS_DIR/run_flow.py" "$@"
