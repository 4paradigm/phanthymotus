#!/usr/bin/env bash
# deploy.sh — lifecycle management for the PR Review Agent.
#
#   ./deploy.sh [up]            Build (if needed) and start
#   ./deploy.sh rebuild         Rebuild the image and recreate the container
#   ./deploy.sh stop            Stop the container, keep it and the data
#   ./deploy.sh start           Start a stopped container (no rebuild)
#   ./deploy.sh restart         Restart the container (no rebuild)
#   ./deploy.sh down            Stop and remove the container, keep the data
#   ./deploy.sh down --purge    Also delete all state in DATA_HOST_DIR
#   ./deploy.sh status          Show container state and agent status
#   ./deploy.sh logs [-n N]     Follow logs (default: last 100 lines)
#
# State (review history, build logs, bare clones) lives on the host at
# DATA_HOST_DIR, default /opt/phanthy-motus/pr-review.
#
# Mirrors default to Tencent Cloud. Override in .env for a host outside the
# Tencent VPC:
#   MIRROR_BASE=docker.io
#   PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple/
#   APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE="docker compose"
SERVICE="pr-review-agent"

# Read PORT/BIND_ADDR from .env rather than hardcoding them, so the URLs this
# script prints and curls always match what compose actually published. A
# hardcoded port here is how the "dashboard unreachable" bug stayed invisible.
_env_val() {
    [ -f .env ] || { echo "$2"; return; }
    local v
    v="$(grep -E "^${1}=" .env | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
    echo "${v:-$2}"
}
PORT="$(_env_val PORT 25000)"
BIND_ADDR="$(_env_val BIND_ADDR 0.0.0.0)"
DATA_HOST_DIR="$(_env_val DATA_HOST_DIR /opt/phanthy-motus/pr-review)"

# The bind-mount target must exist before the container starts, or Docker
# creates it root-owned with no warning and the agent writes into a path nobody
# expects.
ensure_data_dir() {
    if [ ! -d "$DATA_HOST_DIR" ]; then
        info "Creating data directory $DATA_HOST_DIR"
        mkdir -p "$DATA_HOST_DIR"
    fi
}

info() { echo "==> $*"; }
die()  { echo "Error: $*" >&2; exit 1; }

require_env() {
    [ -f .env ] || die "$(printf '.env not found. Create it first:\n  cp .env.example .env\n  $EDITOR .env      # GITHUB_TOKEN, REGISTRY_*, LLM_*')"
}

# Fail on placeholder credentials rather than starting an agent that will
# error on its first API call.
validate_env() {
    local missing=()
    local var value
    for var in GITHUB_TOKEN REGISTRY REGISTRY_USER REGISTRY_PASSWORD; do
        value="$(grep -E "^${var}=" .env | tail -1 | cut -d= -f2- || true)"
        if [ -z "$value" ] || [[ "$value" == your_* ]] || [[ "$value" == ghp_your_* ]]; then
            missing+=("$var")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "Error: these .env values are unset or still placeholders:" >&2
        printf '  - %s\n' "${missing[@]}" >&2
        exit 1
    fi
}

# QEMU registration for ARM64 cross-compilation, pulled from the Tencent
# mirror. Idempotent, and it persists on the host until reboot — the repos'
# build scripts assume it is already there.
setup_qemu() {
    local binfmt="${BINFMT_IMAGE:-mirror.ccs.tencentyun.com/tonistiigi/binfmt}"
    info "Registering QEMU for ARM64 cross-compilation ($binfmt)"
    docker run --rm --privileged "$binfmt" --install arm64 >/dev/null 2>&1 \
        || echo "    warning: QEMU registration failed; ARM64 builds may not work"
}

show_endpoints() {
    # Show the address the dashboard is actually reachable at, which depends on
    # how BIND_ADDR was set — printing a localhost-only URL for an exposed
    # deployment (or vice versa) just sends people down the wrong path.
    local lan_ip
    lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo
    if [ "$BIND_ADDR" = "127.0.0.1" ] || [ "$BIND_ADDR" = "localhost" ]; then
        echo "  Dashboard:  http://localhost:${PORT}/"
        echo "              bound to loopback — from a laptop, first run:"
        echo "                ssh -L ${PORT}:localhost:${PORT} root@${lan_ip:-<this-host>}"
    else
        echo "  Dashboard:  http://${lan_ip:-<this-host>}:${PORT}/"
        echo "              also http://localhost:${PORT}/ on this host"
        echo "              NOTE: no authentication — keep this on a trusted network."
    fi
    echo "  Status:     curl -s http://localhost:${PORT}/api/status | python3 -m json.tool"
    echo "  Jobs:       curl -s http://localhost:${PORT}/api/jobs | python3 -m json.tool"
    echo "  Logs:       $0 logs"
    echo
    echo "Trigger a review by commenting /request_bot_review on a PR."
    echo "Polling picks it up within POLL_INTERVAL_SECONDS (default 30s)."
}

cmd_up() {
    require_env; validate_env; ensure_data_dir; setup_qemu
    info "Building image"
    $COMPOSE build
    info "Starting agent"
    $COMPOSE up -d
    echo; info "PR Review Agent is running."
    show_endpoints
}

cmd_rebuild() {
    require_env; validate_env; ensure_data_dir; setup_qemu
    info "Rebuilding image without cache"
    $COMPOSE build --no-cache
    info "Recreating container"
    $COMPOSE up -d --force-recreate
    echo; info "PR Review Agent rebuilt and running."
    show_endpoints
}

cmd_stop() {
    require_env
    # Compose waits out stop_grace_period, during which the agent tells any
    # in-flight jobs' PRs that they were interrupted.
    info "Stopping agent (in-flight jobs will be notified on their PRs)"
    $COMPOSE stop
    info "Stopped. Container and data are kept — './deploy.sh start' resumes."
}

cmd_start() {
    require_env
    info "Starting existing container"
    $COMPOSE start
    info "Started."
    show_endpoints
}

cmd_restart() {
    require_env
    info "Restarting agent (in-flight jobs will be notified on their PRs)"
    $COMPOSE restart
    info "Restarted."
    show_endpoints
}

cmd_down() {
    require_env
    if [ "${1:-}" = "--purge" ]; then
        # `docker compose down -v` only removes named volumes; the data is a
        # bind mount, so it has to be deleted explicitly or --purge would claim
        # to have purged and silently left everything in place.
        echo "This deletes ALL agent state under:"
        echo "    $DATA_HOST_DIR"
        echo "  - jobs.db          review history"
        echo "  - logs/            full build logs"
        echo "  - *.git            bare clones (re-cloned on next start)"
        echo "  - poller_state.json watermarks; trigger comments older than"
        echo "                     POLL_INITIAL_LOOKBACK_MINUTES will be missed"
        printf "Type 'purge' to confirm: "
        read -r confirm
        [ "$confirm" = "purge" ] || die "Aborted — nothing deleted."
        info "Removing container"
        $COMPOSE down
        if [ -d "$DATA_HOST_DIR" ]; then
            info "Deleting $DATA_HOST_DIR"
            rm -rf -- "${DATA_HOST_DIR:?}"/*
            rm -f -- "${DATA_HOST_DIR:?}"/.[!.]* 2>/dev/null || true
        fi
        info "Removed, including data."
    else
        info "Removing container (data kept in $DATA_HOST_DIR)"
        $COMPOSE down
        info "Removed. Data kept for the next start."
    fi
}

cmd_status() {
    require_env
    info "Container"
    $COMPOSE ps
    echo
    info "Agent status"
    if curl -sf --max-time 5 "http://localhost:${PORT}/api/status" -o /tmp/pr_review_status.json; then
        python3 -m json.tool /tmp/pr_review_status.json 2>/dev/null \
            || cat /tmp/pr_review_status.json
        rm -f /tmp/pr_review_status.json
    else
        echo "  Not responding on localhost:${PORT}."
        echo "  Check '$0 logs' — the container may be starting or unhealthy."
        return 1
    fi
}

cmd_logs() {
    require_env
    local tail_n=100
    if [ "${1:-}" = "-n" ] && [ -n "${2:-}" ]; then
        tail_n="$2"
    fi
    $COMPOSE logs -f --tail "$tail_n"
}

usage() {
    # Print the header comment block, stopping at the first non-comment line so
    # this stays correct as the header is edited.
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
}

case "${1:-up}" in
    up|"")    cmd_up ;;
    rebuild)  cmd_rebuild ;;
    stop)     cmd_stop ;;
    start)    cmd_start ;;
    restart)  cmd_restart ;;
    down)     shift; cmd_down "$@" ;;
    status)   cmd_status ;;
    logs)     shift; cmd_logs "$@" ;;
    -h|--help|help) usage ;;
    *)        echo "Unknown command: $1" >&2; echo >&2; usage >&2; exit 1 ;;
esac
