#!/usr/bin/env bash
# Canonical docker run for TTS leaderboard on Jetson (GPU required).
#
# judgeflow reference service should call this script (or equivalent flags)
# instead of plain: docker run --privileged --network=host ...
#
# Usage:
#   ./deploy/judgeflow_tts_run.sh <image> <container_name> <mcp_port> <ws_port>
#
# Example (instance 0):
#   ./deploy/judgeflow_tts_run.sh phanthymotus-perception-tts:3794763 \
#     phanthymotus-perception-tts-0 15720 15721
set -euo pipefail

IMAGE="${1:?image required}"
NAME="${2:?container name required}"
MCP_PORT="${3:?MCP_PORT required}"
WS_PORT="${4:?WS_PORT required}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

RUN_ARGS=(
    docker run -d
    --name "${NAME}"
    --runtime nvidia
    --network host
    --privileged
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e MCP_PORT="${MCP_PORT}"
    -e WS_PORT="${WS_PORT}"
)

# Optional host model cache (if mounted on eval Jetson)
if [ -d /models ]; then
    RUN_ARGS+=(-v /models:/models)
fi

RUN_ARGS+=(
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
    -e FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-DEFAULT}"
)

RUN_ARGS+=("${IMAGE}")

echo "[judgeflow_tts_run] ${RUN_ARGS[*]}"
"${RUN_ARGS[@]}"
