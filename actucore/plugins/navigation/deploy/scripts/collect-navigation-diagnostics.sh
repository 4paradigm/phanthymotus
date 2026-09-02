#!/usr/bin/env bash
set -euo pipefail

DURATION_SEC="${DURATION_SEC:-300}"
ACTUCORE_CONTAINER="${ACTUCORE_CONTAINER:-embodied-actucore-test}"
OUTPUT_DIR="${OUTPUT_DIR:-navigation-diagnostics-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! "$DURATION_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR=DURATION_SEC must be a positive integer" >&2
  exit 2
fi
if [[ ! "$ACTUCORE_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "ERROR=ACTUCORE_CONTAINER is invalid" >&2
  exit 2
fi
if ! docker inspect "$ACTUCORE_CONTAINER" >/dev/null 2>&1; then
  echo "ERROR=container not found: $ACTUCORE_CONTAINER" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
STATUS_FILE="$OUTPUT_DIR/pipeline-status.jsonl"
RESOURCE_FILE="$OUTPUT_DIR/resources.jsonl"

docker exec \
  -e DIAGNOSTICS_DURATION_SEC="$DURATION_SEC" \
  "$ACTUCORE_CONTAINER" bash -lc '
    set +u
    if [ -f /opt/ros/humble/install/setup.bash ]; then
      source /opt/ros/humble/install/setup.bash
    else
      source /opt/ros/humble/setup.bash
    fi
    [ ! -f /opt/fast_livo_ws/install/setup.bash ] || source /opt/fast_livo_ws/install/setup.bash
    [ ! -f /ros_ws/install/setup.bash ] || source /ros_ws/install/setup.bash
    set -u
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
    export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-DEFAULT}"
    timeout "$DIAGNOSTICS_DURATION_SEC" ros2 topic echo --full-length --field data \
      /ubuntu/navigation/fast_livo2/status std_msgs/msg/String
  ' |
  python3 -u -c '
import ast, json, sys, time
for line in sys.stdin:
    value = line.strip()
    if not value or value == "---":
        continue
    if value.startswith("data:"):
        value = value[5:].strip()
    try:
        if value[:1] in {"\"", "\047"}:
            value = ast.literal_eval(value)
        payload = json.loads(value)
    except (SyntaxError, ValueError, TypeError):
        continue
    print(json.dumps({"kind": "pipeline_status", "observed_unix_ns": time.time_ns(), "payload": payload}, separators=(",", ":")), flush=True)
' >"$STATUS_FILE" &
STATUS_PID=$!

cleanup() {
  kill "$STATUS_PID" 2>/dev/null || true
  wait "$STATUS_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + DURATION_SEC))
sample=0
while (( SECONDS < deadline )); do
  observed_unix_ns="$(( $(date +%s) * 1000000000 ))"
  stats="$(docker stats --no-stream --format '{{json .}}' "$ACTUCORE_CONTAINER")"
  printf '{"kind":"docker_stats","observed_unix_ns":%s,"payload":%s}\n' \
    "$observed_unix_ns" "$stats" >>"$RESOURCE_FILE"

  if command -v tegrastats >/dev/null 2>&1; then
    tegra="$(timeout 2 tegrastats --interval 1000 --count 1 2>/dev/null || true)"
    NAV_DIAG_PAYLOAD="$tegra" NAV_DIAG_STAMP="$observed_unix_ns" python3 -c '
import json, os
print(json.dumps({"kind": "tegrastats", "observed_unix_ns": int(os.environ["NAV_DIAG_STAMP"]), "payload": os.environ["NAV_DIAG_PAYLOAD"]}, separators=(",", ":")))
' >>"$RESOURCE_FILE"
  fi

  if (( sample % 5 == 0 )); then
    top="$(docker top "$ACTUCORE_CONTAINER" -eo pid,ppid,pcpu,pmem,rss,comm,args 2>/dev/null || true)"
    NAV_DIAG_PAYLOAD="$top" NAV_DIAG_STAMP="$observed_unix_ns" python3 -c '
import json, os
print(json.dumps({"kind": "docker_top", "observed_unix_ns": int(os.environ["NAV_DIAG_STAMP"]), "payload": os.environ["NAV_DIAG_PAYLOAD"]}, separators=(",", ":")))
' >>"$RESOURCE_FILE"
  fi
  sample=$((sample + 1))
  sleep 1
done

cleanup
trap - EXIT INT TERM
echo "NAVIGATION_DIAGNOSTICS_DIR=$OUTPUT_DIR"
echo "PIPELINE_STATUS=$STATUS_FILE"
echo "RESOURCES=$RESOURCE_FILE"
