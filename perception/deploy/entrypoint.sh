#!/bin/bash
# Jetson TTS perception entrypoint — verify CUDA before loading sherpa GPU provider.
# Note: do not use "set -u" here; ROS setup.bash references optional vars (e.g. COLCON_TRACE).
set -eo pipefail

log() { echo "[entrypoint] $*" >&2; }

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# ROS DDS: do not set ROS_DOMAIN_ID in the image — judgeflow injects it at
# docker run (must match robot-tts-evaluation). Log runtime values for debug.
log "starting (LD_PRELOAD=${LD_PRELOAD})"
log "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>} FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-<unset>}"

if [ "${TTS_REQUIRE_CUDA:-1}" = "1" ]; then
    log "checking CUDA via torch..."
    if ! cuda_name="$(python3 - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(1)
print(torch.cuda.get_device_name(0))
PY
    )"; then
        log "FATAL: hw_provider=cuda but CUDA is not available."
        log "judgeflow must start the container with GPU runtime, for example:"
        log "  docker run --runtime nvidia \\"
        log "    -e NVIDIA_VISIBLE_DEVICES=all \\"
        log "    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\"
        log "    ... <image>"
        log "See deploy/judgeflow_tts_run.sh in this repository."
        exit 125
    fi
    log "CUDA ok: ${cuda_name}"
fi

log "sourcing /opt/ros/humble/install/setup.bash"
source /opt/ros/humble/install/setup.bash
log "sourcing /ros_ws/install/setup.bash"
source /ros_ws/install/setup.bash

log "launching /work/main.py"
exec python3 /work/main.py
