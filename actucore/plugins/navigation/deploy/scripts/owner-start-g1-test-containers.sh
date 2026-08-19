#!/usr/bin/env bash
set -Eeuo pipefail

STAGE="${STAGE:-preflight}"
ACTUCORE_IMAGE="${ACTUCORE_IMAGE:-}"
ACTUCORE_CONTAINER="${ACTUCORE_CONTAINER:-embodied-actucore-test}"
AGENT_CORE_CONTAINER="${AGENT_CORE_CONTAINER:-phanthy-motus-agent-core-1}"
DRIVER_CONTAINER="${DRIVER_CONTAINER:-embodied-unitree-g1}"
MODELS_DIR="${MODELS_DIR:-/opt/embodied/models}"
MAP_DIR="${MAP_DIR:-/opt/phanthy-motus/data/fast_livo2/maps}"
RECORDING_DIR="${RECORDING_DIR:-/opt/phanthy-motus/data/fast_livo2/recordings}"
MCP_URL="${MCP_URL:-http://127.0.0.1:15730/mcp}"
START_TIMEOUT_SEC="${START_TIMEOUT_SEC:-120}"
OWNER_LABEL="com.phanthymotus.test-owner"
OWNER_VALUE="navigation-card"
LEGACY_OWNER_VALUE="nav2-card"

die() { printf 'ERROR=%s\n' "$*" >&2; exit 1; }
exists() { docker container inspect "$1" >/dev/null 2>&1; }
running() {
  [[ "$(docker container inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == true ]]
}
require_running() { exists "$1" && running "$1" || die "required container is not running: $1"; }
require_port_free() {
  [[ -z "$(ss -H -ltn "sport = :$1" 2>/dev/null || true)" ]] || die "TCP port $1 is already in use"
}

probe() {
  local response
  if ! response="$(
    curl --fail --silent --max-time 5 \
      --header 'Content-Type: application/json' \
      --data '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
      "${MCP_URL}" 2>/dev/null
  )"; then
    return 1
  fi
  python3 -c '
import json, sys
tools = {item.get("name") for item in json.load(sys.stdin).get("result", {}).get("tools", [])}
raise SystemExit(0 if "ControlledSemanticSpatial" in tools else 1)
' <<<"${response}" >/dev/null 2>&1
}

status() {
  if ! exists "${ACTUCORE_CONTAINER}"; then
    printf '%s|absent\n' "${ACTUCORE_CONTAINER}"
    return
  fi
  docker container inspect --format \
    '{{.Name}}|image={{.Config.Image}}|status={{.State.Status}}|running={{.State.Running}}|owner={{ index .Config.Labels "com.phanthymotus.test-owner" }}' \
    "${ACTUCORE_CONTAINER}" | sed 's#^/##'
}

preflight() {
  command -v docker >/dev/null || die "required command not found: docker"
  command -v curl >/dev/null || die "required command not found: curl"
  command -v python3 >/dev/null || die "required command not found: python3"
  command -v ss >/dev/null || die "required command not found: ss"
  [[ "$(uname -m)" == "aarch64" ]] || die "this script only runs on G1 aarch64"
  [[ -n "${ACTUCORE_IMAGE}" ]] || die "set ACTUCORE_IMAGE to the exact built image"
  docker image inspect "${ACTUCORE_IMAGE}" >/dev/null 2>&1 || die "image not found: ${ACTUCORE_IMAGE}"
  [[ "$(docker image inspect --format '{{.Architecture}}' "${ACTUCORE_IMAGE}")" == arm64 ]] || \
    die "ActuCore image must be arm64"
  require_running "${AGENT_CORE_CONTAINER}"
  require_running "${DRIVER_CONTAINER}"
  exists "${ACTUCORE_CONTAINER}" && die "${ACTUCORE_CONTAINER} already exists; use STAGE=stop"
  require_port_free 15730
  [[ -d "${MODELS_DIR}" ]] || die "models directory does not exist: ${MODELS_DIR}"
  [[ -d "${MAP_DIR}" && -w "${MAP_DIR}" ]] || die "map directory is missing or not writable: ${MAP_DIR}"
  [[ -d "${RECORDING_DIR}" && -w "${RECORDING_DIR}" ]] || die "recording directory is missing or not writable: ${RECORDING_DIR}"
  printf 'G1_NAVIGATION_TEST_PREFLIGHT=PASS\nACTUCORE_IMAGE=%s\n' "${ACTUCORE_IMAGE}"
}

start() {
  preflight
  docker run --detach \
    --name "${ACTUCORE_CONTAINER}" \
    --label "${OWNER_LABEL}=${OWNER_VALUE}" \
    --network host --ipc host --pid host --privileged \
    --volume /dev:/dev \
    --volume "${MODELS_DIR}:/models" \
    --volume "${MAP_DIR}:/opt/fast_livo_ws/src/fast_livo/Log/pcd:rw" \
    --volume "${RECORDING_DIR}:/opt/phanthy-motus/data/fast_livo2/recordings:rw" \
    --env ROS_DOMAIN_ID=42 \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    --env PYTHONUNBUFFERED=1 \
    --restart no \
    "${ACTUCORE_IMAGE}" >/dev/null
  for ((elapsed=0; elapsed<START_TIMEOUT_SEC; elapsed+=2)); do
    if running "${ACTUCORE_CONTAINER}" && probe; then
      status
      printf 'G1_NAVIGATION_TEST_CONTAINER_START=PASS\n'
      return
    fi
    sleep 2
  done
  docker logs --tail 120 "${ACTUCORE_CONTAINER}" >&2 || true
  die "Navigation tool was not ready within ${START_TIMEOUT_SEC}s"
}

stop() {
  if ! exists "${ACTUCORE_CONTAINER}"; then
    printf '%s=absent\n' "${ACTUCORE_CONTAINER}"
    return
  fi
  owner="$(docker container inspect --format "{{ index .Config.Labels \"${OWNER_LABEL}\" }}" "${ACTUCORE_CONTAINER}")"
  case "${owner}" in
    "${OWNER_VALUE}") ;;
    "${LEGACY_OWNER_VALUE}")
      printf 'G1_NAVIGATION_TEST_OWNER_MIGRATION=%s->%s\n' \
        "${LEGACY_OWNER_VALUE}" "${OWNER_VALUE}"
      ;;
    *) die "refusing to remove container owned by ${owner:-unknown}" ;;
  esac
  docker rm --force "${ACTUCORE_CONTAINER}" >/dev/null
  printf 'G1_NAVIGATION_TEST_CONTAINER_STOP=PASS\n'
}

case "${STAGE}" in
  preflight) preflight ;;
  start) start ;;
  status) status; running "${ACTUCORE_CONTAINER}" && probe && printf 'G1_NAVIGATION_TEST_MCP=PASS\n' ;;
  stop) stop ;;
  *) die "unsupported STAGE=${STAGE}; expected preflight, start, status, or stop" ;;
esac
