#!/usr/bin/env bash
# run-pr-image.sh — run a freshly built image on this machine for a quick test.
#
#   ./deploy/run-pr-image.sh <image-ref>     pull, generate compose, start
#   ./deploy/run-pr-image.sh --status        list everything this script started
#   ./deploy/run-pr-image.sh --logs  [name]  follow a container's logs
#   ./deploy/run-pr-image.sh --shell [name]  open a shell inside a container
#   ./deploy/run-pr-image.sh --down  [name]  stop and remove one
#   ./deploy/run-pr-image.sh --down --all    stop and remove all of them
#
# [name] is the service or container name shown by --status (e.g. unitree-g1
# or embodied-unitree-g1). It may be omitted while only one thing is running.
#
# Intended for throwaway testing of an image built from a PR. It leaves the
# host's real deployment at /opt/phanthy-motus/docker-compose.yml untouched:
# the generated compose file lives on its own under PR_IMAGE_DIR, so `--down`
# cleans up completely.
#
# Every driver and the perception bundle ship a compose service fragment at
# /deploy/service.yml inside the image. This script extracts that fragment and
# wraps it in a standalone compose file — the same fragment Agent Core merges
# into the host compose file when you deploy from the web console. Using compose
# rather than a hand-written `docker run` is what keeps the privileged flags,
# host networking and device mounts each service needs exactly as its author
# declared them.
#
# Several images can be under test at once, and they do not disturb each other:
# each gets its own compose project, named after the service in its fragment.
# A machine normally runs several drivers plus perception, so starting one must
# never touch the rest — see the note above start_run().
#
# For a lasting deployment, use the web console instead. This script is the
# quick "does it work" path.
#
# Requires: docker with the compose plugin. Nothing else — no python, no yq.
set -euo pipefail

PR_IMAGE_DIR="${PR_IMAGE_DIR:-/tmp/pr-image}"
SERVICE_YAML_PATH="/deploy/service.yml"

die()  { echo "Error: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# The throwaway container used to read the service fragment out of the image.
# Cleaned up from an EXIT trap rather than a RETURN trap: the extraction exits
# the script when the fragment is missing, and `exit` inside a function does not
# fire RETURN traps — which leaked a `Created` container on that path.
TEMP_CID=""
cleanup_temp_container() {
    [ -n "$TEMP_CID" ] || return 0
    docker rm -f "$TEMP_CID" >/dev/null 2>&1 || true
    TEMP_CID=""
}
trap cleanup_temp_container EXIT

usage() {
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
}

require_docker() {
    command -v docker >/dev/null 2>&1 || die "docker is not installed."
    docker compose version >/dev/null 2>&1 \
        || die "the docker compose plugin is not available."
}

# ── Runs ──────────────────────────────────────────────────────────────────────
#
# One directory per run, named after the service, each holding its own compose
# file and state. IMAGE/SERVICE/CONTAINER/PROJECT come from the state file.
run_dir()     { printf '%s/%s' "$PR_IMAGE_DIR" "$1"; }
compose_file(){ printf '%s/%s/docker-compose.yml' "$PR_IMAGE_DIR" "$1"; }
state_file()  { printf '%s/%s/state' "$PR_IMAGE_DIR" "$1"; }

# Compose project names may only contain [a-z0-9_-] and must start
# alphanumerically; service names may also carry dots and capitals. The prefix
# is `pr-image-` rather than `pr-` so it cannot collide with a project someone
# already runs on the machine — the build host has a `pr-review` one.
project_name() {
    # printf, not echo: `tr -c` would turn echo's trailing newline into a dash.
    printf 'pr-image-%s' "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '-')"
}

list_runs() {
    local d
    for d in "$PR_IMAGE_DIR"/*/; do
        [ -f "${d}state" ] || continue
        basename "$d"
    done
}

load_run() {
    local name="$1"
    # Cleared first: these are globals, and iterating over several runs would
    # otherwise carry a field from the previous one into a state file that
    # happens not to set it.
    unset IMAGE SERVICE CONTAINER PROJECT
    # shellcheck disable=SC1090
    . "$(state_file "$name")"
    COMPOSE_FILE="$(compose_file "$name")"
    [ -f "$COMPOSE_FILE" ] || die "Compose file missing: $COMPOSE_FILE"
    # PROJECT was added when runs were split per service; a state file written
    # before that has none, and its container carries the old shared project.
    PROJECT="${PROJECT:-$(project_name "$name")}"
}

# Resolve the run a command applies to: the one named, or the only one there is.
select_run() {
    local want="${1:-}" runs
    runs="$(list_runs)"
    [ -n "$runs" ] || die "$(printf 'Nothing started yet.\nRun: %s <image-ref>' "$0")"

    if [ -n "$want" ]; then
        if printf '%s\n' "$runs" | grep -qx -- "$want"; then
            load_run "$want"
            return
        fi
        # Also accept the container name: that is what the PR comment and
        # `docker ps` show, and it differs from the service name
        # (embodied-unitree-g1 vs unitree-g1).
        local name
        while read -r name; do
            load_run "$name"
            if [ "$CONTAINER" = "$want" ]; then return; fi
        done <<< "$runs"
        die "$(printf 'No such run: %s\nRunning:\n%s' "$want" \
               "$(printf '%s\n' "$runs" | sed 's/^/  /')")"
    fi

    if [ "$(printf '%s\n' "$runs" | wc -l)" -gt 1 ]; then
        die "$(printf 'Several images are under test — name the one you mean:\n%s' \
               "$(printf '%s\n' "$runs" | sed 's/^/  /')")"
    fi
    load_run "$runs"
}

compose() { docker compose -p "$PROJECT" -f "$COMPOSE_FILE" "$@"; }

# Runs started before this script split them per service all shared one compose
# file at the top of PR_IMAGE_DIR, under the project `pr-image`. Move such a run
# into its own directory, keeping its original project name so compose can still
# find the container it started — otherwise it would be left behind with nothing
# able to stop it.
migrate_legacy_run() {
    local old_compose="${PR_IMAGE_DIR}/docker-compose.yml"
    local old_state="${PR_IMAGE_DIR}/state"
    [ -f "$old_compose" ] && [ -f "$old_state" ] || return 0

    local SERVICE=""
    # shellcheck disable=SC1090
    . "$old_state"
    [ -n "$SERVICE" ] || { rm -f "$old_compose" "$old_state"; return 0; }

    local dir; dir="$(run_dir "$SERVICE")"
    mkdir -p "$dir"
    mv -f "$old_compose" "${dir}/docker-compose.yml"
    mv -f "$old_state"   "${dir}/state"
    printf 'PROJECT=%s\n' "pr-image" >> "${dir}/state"
    rm -f "${PR_IMAGE_DIR}/service.yml"
    info "Adopted an earlier run of ${SERVICE} into its own project directory."
}

# ── Extract the service fragment out of the image ─────────────────────────────
#
# Via `docker create` + `docker cp` rather than `docker run --entrypoint cat`,
# so it works regardless of what the image contains — the same approach Agent
# Core uses (api/drivers.py:_deploy_sync).
extract_service_yaml() {
    local image="$1" dest="$2"
    TEMP_CID="$(docker create "$image" 2>/dev/null)" \
        || die "Could not create a container from $image"

    if ! docker cp "${TEMP_CID}:${SERVICE_YAML_PATH}" "$dest" >/dev/null 2>&1; then
        cat >&2 <<EOF
Error: this image has no ${SERVICE_YAML_PATH}, so it cannot be started here.

  Agent Core (the 'core' image) is the one target without a service fragment:
  it is the agent itself, and it is updated in place from the web console —
  the dashboard pulls the image and hands over to a restart helper. Starting a
  second copy by hand would fight the running one.

  Drivers and the perception bundle do ship one, and work with this script.
EOF
        exit 1
    fi
    cleanup_temp_container
}

service_name_of() {
    local fragment="$1" service
    service="$(grep -m1 -E '^[A-Za-z0-9_.-]+:' "$fragment" | sed 's/:.*//')"
    [ -n "$service" ] || die "Could not find a service name in $SERVICE_YAML_PATH"
    printf '%s' "$service"
}

# ── Build a standalone compose file from the fragment ─────────────────────────
#
# The fragment is a single mapping, `<service-name>: {...}`. Wrapping it needs
# only a `services:` header and two spaces of indent, so this is done with text
# transforms and carries no YAML-parser dependency.
generate_compose() {
    local fragment="$1" image="$2" service="$3" out="$4" project="$5" state="$6"

    {
        echo "# Generated by run-pr-image.sh — safe to delete."
        echo "# Source: ${SERVICE_YAML_PATH} inside ${image}"
        echo "services:"
        # Replace whatever the image declares (normally the __IMAGE__
        # placeholder) with the ref actually being tested, then indent.
        sed -E "s|^([[:space:]]*image:[[:space:]]*).*|\1${image}|" "$fragment" \
            | sed 's/^/  /'
    } > "$out"

    local container
    container="$(sed -nE 's|^[[:space:]]*container_name:[[:space:]]*(.*)$|\1|p' \
                 "$fragment" | head -1)"

    printf 'IMAGE=%s\nSERVICE=%s\nCONTAINER=%s\nPROJECT=%s\n' \
        "$image" "$service" "${container:-$service}" "$project" > "$state"
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_up() {
    local image="$1"
    require_docker
    mkdir -p "$PR_IMAGE_DIR"
    migrate_legacy_run

    info "Pulling $image"
    if ! docker pull "$image"; then
        # A pull failure is not fatal if the image is already here: the registry
        # may be briefly unreachable, or the image may have been built locally
        # and never pushed. Only give up when we have no copy at all.
        if docker image inspect "$image" >/dev/null 2>&1; then
            info "Pull failed, but the image is already present locally — using that."
        else
            die "$(printf 'Pull failed and the image is not present locally.\nCheck the reference, and that this host is logged in to the registry.')"
        fi
    fi

    local fragment="${PR_IMAGE_DIR}/.service.yml"
    info "Reading ${SERVICE_YAML_PATH} from the image"
    extract_service_yaml "$image" "$fragment"

    local service; service="$(service_name_of "$fragment")"
    local dir;     dir="$(run_dir "$service")"
    mkdir -p "$dir"
    mv -f "$fragment" "${dir}/service.yml"

    local project; project="$(project_name "$service")"
    generate_compose "${dir}/service.yml" "$image" "$service" \
                     "$(compose_file "$service")" "$project" "$(state_file "$service")"
    info "Generated $(compose_file "$service") (service: $service, project: $project)"

    load_run "$service"

    # Every image gets its own project, and the project holds exactly this one
    # service — so `up` replaces a previous run of *this* image and nothing else.
    # All runs used to share one project, which meant `--remove-orphans` treated
    # every other driver already under test as an orphan and removed it: testing
    # a second driver silently killed the first. A machine runs several drivers
    # at once, so that has to stay impossible.
    info "Starting"
    compose up -d

    echo
    info "Running. Container: ${CONTAINER}"
    echo "  logs  : $0 --logs ${SERVICE}"
    echo "  shell : $0 --shell ${SERVICE}"
    echo "  stop  : $0 --down ${SERVICE}"
}

cmd_logs()  { migrate_legacy_run; select_run "${1:-}"; compose logs -f; }
cmd_shell() {
    migrate_legacy_run
    select_run "${1:-}"
    # bash is not guaranteed in every image; fall back to sh.
    compose exec "$SERVICE" bash 2>/dev/null || compose exec "$SERVICE" sh
}

down_one() {
    local name="$1"
    load_run "$name"
    info "Stopping and removing ${CONTAINER}"
    compose down
    rm -rf "$(run_dir "$name")"
}

cmd_down() {
    migrate_legacy_run
    if [ "${1:-}" = "--all" ]; then
        local runs; runs="$(list_runs)"
        [ -n "$runs" ] || die "Nothing started yet."
        local name
        while read -r name; do down_one "$name"; done <<< "$runs"
        rmdir "$PR_IMAGE_DIR" 2>/dev/null || true
        info "Removed. Nothing of these tests remains."
        return
    fi
    select_run "${1:-}"
    down_one "$SERVICE"
    rmdir "$PR_IMAGE_DIR" 2>/dev/null || true
    info "Removed. Nothing of this test remains."
}

cmd_status() {
    migrate_legacy_run
    local runs; runs="$(list_runs)"
    [ -n "$runs" ] || die "$(printf 'Nothing started yet.\nRun: %s <image-ref>' "$0")"

    local name
    while read -r name; do
        load_run "$name"
        echo "  image     : $IMAGE"
        echo "  service   : $SERVICE"
        echo "  container : $CONTAINER"
        echo "  project   : $PROJECT"
        echo "  compose   : $COMPOSE_FILE"
        echo
        compose ps
        echo
    done <<< "$runs"
}

case "${1:-}" in
    --logs)          cmd_logs "${2:-}" ;;
    --shell|--exec)  cmd_shell "${2:-}" ;;
    --down|--stop)   cmd_down "${2:-}" ;;
    --status|--ps)   cmd_status ;;
    -h|--help|help|"") usage ;;
    -*)              die "Unknown option: $1  (try --help)" ;;
    *)               cmd_up "$1" ;;
esac
