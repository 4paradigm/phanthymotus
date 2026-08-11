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
FAST_LIVO2_IMAGE="${FAST_LIVO2_IMAGE:-}"
NAV2_IMAGE="${NAV2_IMAGE:-}"

PERCEPTION_CONTAINER="${PERCEPTION_CONTAINER:-embodied-perception-test}"
FAST_LIVO2_CONTAINER="${FAST_LIVO2_CONTAINER:-embodied-perception-fast-livo2-test}"
NAV2_CONTAINER="${NAV2_CONTAINER:-embodied-perception-nav2-test}"
AGENT_CORE_CONTAINER="${AGENT_CORE_CONTAINER:-phanthy-motus-agent-core-1}"
DRIVER_CONTAINER="${DRIVER_CONTAINER:-embodied-unitree-g1}"

AGENT_CORE_URL="${AGENT_CORE_URL:-https://localhost:15678}"
PERCEPTION_MCP_URL="${PERCEPTION_MCP_URL:-http://127.0.0.1:15720/mcp}"
MODELS_DIR="${MODELS_DIR:-/opt/embodied/models}"
FAST_LIVO2_MAP_DIR="${FAST_LIVO2_MAP_DIR:-/opt/phanthy-motus/data/fast_livo2/maps}"
START_TIMEOUT_SEC="${START_TIMEOUT_SEC:-120}"

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
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

probe_navigation_tools() {
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
tools = {tool.get("name") for tool in payload.get("result", {}).get("tools", [])}
raise SystemExit(0 if {"fast_livo2", "nav2"}.issubset(tools) else 1)
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
  [[ -n "${FAST_LIVO2_IMAGE}" ]] || die "set FAST_LIVO2_IMAGE to the exact built FAST-LIVO2 companion image"
  [[ -n "${NAV2_IMAGE}" ]] || die "set NAV2_IMAGE to the exact built Nav2 companion image"

  require_arm64_image "${PERCEPTION_IMAGE}"
  require_arm64_image "${FAST_LIVO2_IMAGE}"
  require_arm64_image "${NAV2_IMAGE}"
  require_running_container "${AGENT_CORE_CONTAINER}"
  require_running_container "${DRIVER_CONTAINER}"

  container_exists "${PERCEPTION_CONTAINER}" && \
    die "${PERCEPTION_CONTAINER} already exists; inspect it with STAGE=status or remove it with STAGE=stop"
  container_exists "${FAST_LIVO2_CONTAINER}" && \
    die "${FAST_LIVO2_CONTAINER} already exists; inspect it with STAGE=status or remove it with STAGE=stop"
  container_exists "${NAV2_CONTAINER}" && \
    die "${NAV2_CONTAINER} already exists; inspect it with STAGE=status or remove it with STAGE=stop"

  require_port_free 15720
  require_port_free 15721
  [[ -d "${MODELS_DIR}" ]] || \
    die "Perception models directory does not exist: ${MODELS_DIR}"
  [[ -d "${FAST_LIVO2_MAP_DIR}" ]] || \
    die "FAST-LIVO2 map directory does not exist: ${FAST_LIVO2_MAP_DIR}"
  [[ -w "${FAST_LIVO2_MAP_DIR}" ]] || \
    die "FAST-LIVO2 map directory is not writable: ${FAST_LIVO2_MAP_DIR}"

  note "G1_NAV2_TEST_PREFLIGHT=PASS"
  note "PERCEPTION_IMAGE=${PERCEPTION_IMAGE}"
  note "FAST_LIVO2_IMAGE=${FAST_LIVO2_IMAGE}"
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
  for name in "${PERCEPTION_CONTAINER}" "${NAV2_CONTAINER}" "${FAST_LIVO2_CONTAINER}"; do
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
    container_running "${FAST_LIVO2_CONTAINER}" || return 1
    container_running "${NAV2_CONTAINER}" || return 1
    if container_running "${PERCEPTION_CONTAINER}" && probe_navigation_tools; then
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
  preflight

  note "Starting ${FAST_LIVO2_CONTAINER}"
  if ! docker run --detach \
    --name "${FAST_LIVO2_CONTAINER}" \
    --label "${TEST_OWNER_LABEL}=${TEST_OWNER_VALUE}" \
    --label "com.phanthymotus.source-revision=${SOURCE_REVISION}" \
    --network host \
    --ipc host \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=512m \
    --tmpfs /root/.ros:rw,nosuid,nodev,noexec,size=64m \
    --volume "${FAST_LIVO2_MAP_DIR}:/opt/fast_livo_ws/src/fast_livo/Log/pcd:rw" \
    --group-add "$(stat -c '%g' "${FAST_LIVO2_MAP_DIR}")" \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --log-driver local \
    --log-opt max-size=20m \
    --log-opt max-file=3 \
    --restart no \
    "${FAST_LIVO2_IMAGE}" >/dev/null; then
    die "failed to start ${FAST_LIVO2_CONTAINER}"
  fi

  note "Starting ${NAV2_CONTAINER}"
  if ! docker run --detach \
    --name "${NAV2_CONTAINER}" \
    --label "${TEST_OWNER_LABEL}=${TEST_OWNER_VALUE}" \
    --label "com.phanthymotus.source-revision=${SOURCE_REVISION}" \
    --network host \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --tmpfs /root/.ros:rw,nosuid,nodev,noexec,size=64m \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --log-driver local \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    --restart no \
    "${NAV2_IMAGE}" >/dev/null; then
    remove_created_container "${FAST_LIVO2_CONTAINER}"
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
    remove_created_container "${FAST_LIVO2_CONTAINER}"
    die "failed to start ${PERCEPTION_CONTAINER}; companions were removed"
  fi

  if ! wait_until_ready; then
    show_failure_logs
    remove_created_container "${PERCEPTION_CONTAINER}"
    remove_created_container "${NAV2_CONTAINER}"
    remove_created_container "${FAST_LIVO2_CONTAINER}"
    die "test stack did not expose the FAST-LIVO2 and Nav2 MCP tools within ${START_TIMEOUT_SEC}s; created containers were removed"
  fi

  print_container_status "${PERCEPTION_CONTAINER}"
  print_container_status "${FAST_LIVO2_CONTAINER}"
  print_container_status "${NAV2_CONTAINER}"
  note "G1_NAV2_TEST_CONTAINERS_START=PASS"
  note "NOTE=full Perception is running with restart=no; Canvas state was not queried or changed and no navigation action was issued"
}

status_test_containers() {
  require_command docker
  require_command curl
  require_command python3
  docker info >/dev/null 2>&1 || die "Docker daemon is not accessible"

  print_container_status "${PERCEPTION_CONTAINER}"
  print_container_status "${FAST_LIVO2_CONTAINER}"
  print_container_status "${NAV2_CONTAINER}"
  if container_running "${PERCEPTION_CONTAINER}"; then
    if probe_navigation_tools; then
      note "G1_NAV2_TEST_MCP=PASS"
    else
      die "${PERCEPTION_CONTAINER} is running but the FAST-LIVO2/Nav2 MCP tools are not ready"
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
  require_command docker
  require_command curl
  require_command python3
  docker info >/dev/null 2>&1 || die "Docker daemon is not accessible"

  stop_one_test_container "${PERCEPTION_CONTAINER}"
  stop_one_test_container "${NAV2_CONTAINER}"
  stop_one_test_container "${FAST_LIVO2_CONTAINER}"
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
