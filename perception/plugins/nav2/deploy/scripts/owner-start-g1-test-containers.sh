#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_REVISION="unknown"
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
  SOURCE_REVISION="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
fi

STAGE="${STAGE:-preflight}"
PERCEPTION_IMAGE="${PERCEPTION_IMAGE:-}"
NAV2_IMAGE="${NAV2_IMAGE:-}"

PERCEPTION_CONTAINER="${PERCEPTION_CONTAINER:-embodied-perception-test}"
NAV2_CONTAINER="${NAV2_CONTAINER:-embodied-perception-nav2-test}"
AGENT_CORE_CONTAINER="${AGENT_CORE_CONTAINER:-phanthy-motus-agent-core-1}"
DRIVER_CONTAINER="${DRIVER_CONTAINER:-embodied-unitree-g1}"
CORE_ACCESS_TOKEN="${CORE_ACCESS_TOKEN:-}"

AGENT_CORE_URL="${AGENT_CORE_URL:-https://localhost:15678}"
PERCEPTION_MCP_URL="${PERCEPTION_MCP_URL:-http://127.0.0.1:15720/mcp}"
MAP_DIR="${MAP_DIR:-/opt/phanthy-motus/data/nav2/maps}"
MODELS_DIR="${MODELS_DIR:-/opt/embodied/models}"
START_TIMEOUT_SEC="${START_TIMEOUT_SEC:-120}"

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
NAV2_MODE="${NAV2_MODE:-mapping}"
NAV2_MAP_NAME="${NAV2_MAP_NAME:-}"

TEST_OWNER_LABEL="com.phanthymotus.test-owner"
TEST_OWNER_VALUE="nav2-card"

die() {
  printf 'ERROR=%s\n' "$*" >&2
  exit 1
}

note() {
  printf '%s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_owner() {
  [[ "${I_AM_G1_OWNER:-0}" == "1" ]] || die "set I_AM_G1_OWNER=1 for STAGE=${STAGE}"
}

container_exists() {
  docker container inspect "$1" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker container inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

container_owner() {
  docker container inspect \
    --format "{{ index .Config.Labels \"${TEST_OWNER_LABEL}\" }}" \
    "$1" 2>/dev/null || true
}

require_test_owned() {
  local name="$1"
  local owner
  owner="$(container_owner "${name}")"
  [[ "${owner}" == "${TEST_OWNER_VALUE}" ]] || \
    die "refusing to manage ${name}: ${TEST_OWNER_LABEL}=${owner:-missing}"
}

require_running_container() {
  local name="$1"
  container_exists "${name}" || die "required container is absent: ${name}"
  container_running "${name}" || die "required container is not running: ${name}"
}

project_running_state() {
  local response
  local -a auth_args=()
  if [[ -n "${CORE_ACCESS_TOKEN}" ]]; then
    auth_args=(--header "Authorization: Bearer ${CORE_ACCESS_TOKEN}")
  fi
  response="$(curl --insecure --fail --silent --show-error \
    --max-time 5 \
    "${auth_args[@]}" \
    "${AGENT_CORE_URL}/api/config/project-running")" || return 1
  printf '%s' "${response}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
value = payload.get("running")
if value is None and isinstance(payload.get("project"), dict):
    value = payload["project"].get("running")
if value is True:
    print("true")
elif value is False:
    print("false")
else:
    raise SystemExit("response does not contain a boolean running state")
'
}

require_canvas_stopped() {
  local state
  if [[ "${I_CONFIRM_CANVAS_STOPPED:-0}" == "1" ]]; then
    note "G1_NAV2_TEST_CANVAS_STATE=owner-confirmed-stopped"
    return
  fi
  if state="$(project_running_state)"; then
    [[ "${state}" == "false" ]] || \
      die "Canvas project is running; stop it before managing test containers"
    note "G1_NAV2_TEST_CANVAS_STATE=stopped"
    return
  fi
  die "cannot verify Canvas state; stop the project manually, then set I_CONFIRM_CANVAS_STOPPED=1"
}

require_arm64_image() {
  local image="$1"
  local arch
  docker image inspect "${image}" >/dev/null 2>&1 || die "image not found: ${image}"
  arch="$(docker image inspect --format '{{.Architecture}}' "${image}")"
  [[ "${arch}" == "arm64" ]] || die "image ${image} has architecture ${arch}, expected arm64"
}

require_port_free() {
  local port="$1"
  local listeners
  listeners="$(ss -H -ltn "sport = :${port}" 2>/dev/null || true)"
  [[ -z "${listeners}" ]] || die "TCP port ${port} is already in use"
}

probe_nav2_tool() {
  local response
  response="$(curl --fail --silent --show-error \
    --max-time 5 \
    --header 'Content-Type: application/json' \
    --data '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
    "${PERCEPTION_MCP_URL}" 2>/dev/null)" || return 1
  printf '%s' "${response}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
tools = payload.get("result", {}).get("tools", [])
raise SystemExit(0 if any(tool.get("name") == "nav2" for tool in tools) else 1)
' >/dev/null
}

print_container_status() {
  local name="$1"
  if ! container_exists "${name}"; then
    printf '%s|absent\n' "${name}"
    return
  fi
  docker container inspect --format \
    '{{.Name}}|image={{.Config.Image}}|status={{.State.Status}}|running={{.State.Running}}|restart_count={{.RestartCount}}|owner={{ index .Config.Labels "com.phanthymotus.test-owner" }}' \
    "${name}" | sed 's#^/##'
}

preflight() {
  require_command docker
  require_command curl
  require_command python3
  require_command ss
  docker info >/dev/null 2>&1 || die "Docker daemon is not accessible"

  [[ "$(uname -m)" == "aarch64" ]] || die "this script only starts containers on the G1 aarch64 host"
  [[ -n "${PERCEPTION_IMAGE}" ]] || die "set PERCEPTION_IMAGE to the exact built Perception image"
  [[ -n "${NAV2_IMAGE}" ]] || die "set NAV2_IMAGE to the exact built Nav2 companion image"

  require_arm64_image "${PERCEPTION_IMAGE}"
  require_arm64_image "${NAV2_IMAGE}"
  require_running_container "${AGENT_CORE_CONTAINER}"
  require_running_container "${DRIVER_CONTAINER}"
  require_canvas_stopped

  container_exists "${PERCEPTION_CONTAINER}" && \
    die "${PERCEPTION_CONTAINER} already exists; inspect it with STAGE=status or remove it with STAGE=stop"
  container_exists "${NAV2_CONTAINER}" && \
    die "${NAV2_CONTAINER} already exists; inspect it with STAGE=status or remove it with STAGE=stop"

  require_port_free 15720
  require_port_free 15721
  [[ -d "${MAP_DIR}" ]] || \
    die "map directory does not exist: ${MAP_DIR}"
  [[ -w "${MAP_DIR}" ]] || \
    die "map directory is not writable by $(id -un): ${MAP_DIR}"
  [[ -d "${MODELS_DIR}" ]] || \
    die "Perception models directory does not exist: ${MODELS_DIR}"

  note "G1_NAV2_TEST_PREFLIGHT=PASS"
  note "PERCEPTION_IMAGE=${PERCEPTION_IMAGE}"
  note "NAV2_IMAGE=${NAV2_IMAGE}"
  note "NOTE=read-only preflight; no container or robot command was issued"
}

remove_created_container() {
  local name="$1"
  if ! container_exists "${name}"; then
    return
  fi
  if [[ "$(container_owner "${name}")" != "${TEST_OWNER_VALUE}" ]]; then
    note "CLEANUP_SKIPPED=${name}:owner-label-mismatch"
    return
  fi
  if container_running "${name}"; then
    docker stop --time 30 "${name}" >/dev/null 2>&1 || true
  fi
  docker rm "${name}" >/dev/null 2>&1 || true
}

show_failure_logs() {
  local name
  for name in "${PERCEPTION_CONTAINER}" "${NAV2_CONTAINER}"; do
    if container_exists "${name}"; then
      note "${name}_LOG_TAIL_BEGIN"
      docker logs --tail 80 "${name}" 2>&1 || true
      note "${name}_LOG_TAIL_END"
    fi
  done
}

wait_until_ready() {
  local started_at now
  started_at="$(date +%s)"
  while true; do
    container_running "${NAV2_CONTAINER}" || return 1
    if container_running "${PERCEPTION_CONTAINER}" && probe_nav2_tool; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - started_at >= START_TIMEOUT_SEC )); then
      return 1
    fi
    sleep 2
  done
}

start_test_containers() {
  local map_gid
  require_owner
  preflight
  map_gid="$(stat -c '%g' "${MAP_DIR}")"

  note "Starting ${NAV2_CONTAINER}"
  if ! docker run --detach \
    --name "${NAV2_CONTAINER}" \
    --label "${TEST_OWNER_LABEL}=${TEST_OWNER_VALUE}" \
    --label "com.phanthymotus.source-revision=${SOURCE_REVISION}" \
    --network host \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --tmpfs /root/.ros:rw,nosuid,nodev,noexec,size=64m \
    --volume "${MAP_DIR}:/maps" \
    --group-add "${map_gid}" \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    --env "NAV2_MODE=${NAV2_MODE}" \
    --env "NAV2_MAP_NAME=${NAV2_MAP_NAME}" \
    --env NAV2_LIDAR_X=-0.00368 \
    --env NAV2_LIDAR_Y=0.00003 \
    --env NAV2_LIDAR_Z=0.46018 \
    --env NAV2_LIDAR_ROLL=0.0 \
    --env NAV2_LIDAR_PITCH=0.04014257279586953 \
    --env NAV2_LIDAR_YAW=0.0 \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --log-driver local \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    --restart no \
    "${NAV2_IMAGE}" >/dev/null; then
    die "failed to start ${NAV2_CONTAINER}"
  fi

  note "Starting ${PERCEPTION_CONTAINER} with the full Perception configuration"
  if ! docker run --detach \
    --name "${PERCEPTION_CONTAINER}" \
    --label "${TEST_OWNER_LABEL}=${TEST_OWNER_VALUE}" \
    --label "com.phanthymotus.source-revision=${SOURCE_REVISION}" \
    --network host \
    --ipc host \
    --pid host \
    --privileged \
    --volume /dev:/dev \
    --volume "${MODELS_DIR}:/models" \
    --env "AGENT_CORE_URL=${AGENT_CORE_URL}" \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    --env PYTHONUNBUFFERED=1 \
    --log-driver local \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    --restart no \
    "${PERCEPTION_IMAGE}" >/dev/null; then
    remove_created_container "${NAV2_CONTAINER}"
    die "failed to start ${PERCEPTION_CONTAINER}; companion was removed"
  fi

  if ! wait_until_ready; then
    show_failure_logs
    remove_created_container "${PERCEPTION_CONTAINER}"
    remove_created_container "${NAV2_CONTAINER}"
    die "test stack did not expose the nav2 MCP tool within ${START_TIMEOUT_SEC}s; created containers were removed"
  fi

  print_container_status "${PERCEPTION_CONTAINER}"
  print_container_status "${NAV2_CONTAINER}"
  note "G1_NAV2_TEST_CONTAINERS_START=PASS"
  note "NOTE=full Perception is running with restart=no; Canvas remains stopped and no navigation action was issued"
}

status_test_containers() {
  require_command docker
  require_command curl
  require_command python3
  docker info >/dev/null 2>&1 || die "Docker daemon is not accessible"

  print_container_status "${PERCEPTION_CONTAINER}"
  print_container_status "${NAV2_CONTAINER}"
  if container_running "${PERCEPTION_CONTAINER}"; then
    if probe_nav2_tool; then
      note "G1_NAV2_TEST_MCP=PASS"
    else
      die "${PERCEPTION_CONTAINER} is running but the nav2 MCP tool is not ready"
    fi
  fi
}

stop_one_test_container() {
  local name="$1"
  if ! container_exists "${name}"; then
    note "${name}=absent"
    return
  fi
  require_test_owned "${name}"
  if container_running "${name}"; then
    docker stop --time 30 "${name}" >/dev/null
  fi
  docker rm "${name}" >/dev/null
  note "${name}=removed"
}

stop_test_containers() {
  require_owner
  require_command docker
  require_command curl
  require_command python3
  docker info >/dev/null 2>&1 || die "Docker daemon is not accessible"
  require_canvas_stopped

  stop_one_test_container "${PERCEPTION_CONTAINER}"
  stop_one_test_container "${NAV2_CONTAINER}"
  note "G1_NAV2_TEST_CONTAINERS_STOP=PASS"
  note "NOTE=only containers labelled ${TEST_OWNER_LABEL}=${TEST_OWNER_VALUE} were removed"
}

case "${STAGE}" in
  preflight)
    preflight
    ;;
  start)
    start_test_containers
    ;;
  status)
    status_test_containers
    ;;
  stop)
    stop_test_containers
    ;;
  *)
    die "unsupported STAGE=${STAGE}; expected preflight, start, status, or stop"
    ;;
esac
