#!/usr/bin/env bash
# Local disposable integration only: no network, host devices or credentials.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'Usage: test_tianyi_teleop_isolated.sh DRIVER_SOURCE_ROOT CPU_ACTUCORE_IMAGE' >&2
  exit 2
fi
core_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
driver_root="$(cd "$1" && pwd)"
test -f "$driver_root/x-humanoid/tianyi2.0/teleop_executor.py"
docker run --rm --network none --read-only --cpus 4 --memory 2g \
  --tmpfs /tmp:rw,nosuid,size=256m \
  -v "$core_root:/candidate/core:ro" -v "$driver_root:/candidate/driver:ro" \
  -v "$core_root/agent-core/deploy/dds-local.xml:/opt/phanthy-motus/dds-local.xml:ro" \
  -v "$core_root/agent-core/deploy/dds-local.xml:/deploy/dds-local.xml:ro" \
  -e PYTHONDONTWRITEBYTECODE=1 -e ROS_LOG_DIR=/tmp/ros \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e TIANYI_DRIVER_SOURCE=/candidate/driver/x-humanoid/tianyi2.0 \
  -e ACTUCORE_PLUGIN_ROOT=/work/plugins \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/opt/phanthy-motus/dds-local.xml \
  --entrypoint bash "$2" \
  -c 'source /opt/ros/humble/setup.bash && exec python3 /candidate/core/actucore/tests/tianyi_ros_e2e.py'
