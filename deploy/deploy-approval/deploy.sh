#!/usr/bin/env bash
# deploy.sh — manage the Deploy Approval Agent container.
#
# Commands:
#   ./deploy.sh up|rebuild|start|stop|restart|down|status|logs
#
# Fixed inputs:
#   machine policy: ./machines.yaml
#   COS secrets: ./secrets.yaml
#
# No purge mode, no runtime API token, no env indirection for file locations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/deploy.sh"
cd "$SCRIPT_DIR"
COMPOSE="docker compose"
MACHINES_FILE="./machines.yaml"
SECRETS_FILE="./secrets.yaml"
TMP_ENV=""

die() { echo "ERROR: $*" >&2; exit 1; }

require_runtime_user() {
    RUNTIME_UID="$(id -u)"
    RUNTIME_GID="$(id -g)"
    [ "$RUNTIME_UID" -gt 0 ] || die "deploy.sh runtime user must be non-root"
    [ "$RUNTIME_GID" -gt 0 ] || die "deploy.sh runtime group must be non-root"
    export RUNTIME_UID RUNTIME_GID
}

dotenv_parse() {
    local file="$1"
    [ -f "$file" ] || { echo "MISSING:$file" >&2; return 2; }
    [ -r "$file" ] || { echo "UNREADABLE:$file" >&2; return 2; }
    python3 - "$file" <<'PY'
import re, sys
path = sys.argv[1]
seen = set()
with open(path, "r", encoding="utf-8") as fh:
    data = fh.read()
if "\0" in data:
    raise SystemExit("NUL byte in dotenv")
for lineno, raw in enumerate(data.splitlines(), 1):
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        raise SystemExit(f"malformed dotenv line {lineno}")
    name, _, value = line.partition("=")
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise SystemExit(f"invalid key at line {lineno}")
    if name in seen:
        raise SystemExit(f"duplicate key {name}")
    seen.add(name)
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    if "\n" in value:
        raise SystemExit(f"unsafe value at line {lineno}")
    print(f"{name}={value}")
PY
}

dotenv_value() {
    local file="$1" key="$2"
    dotenv_parse "$file" | sed -n "s/^${key}=//p" | head -n1
}

dotenv_has_key() {
    local file="$1" key="$2"
    dotenv_parse "$file" | awk -F= -v key="$key" '$1 == key { found = 1 } END { exit(found ? 0 : 1) }'
}

require_env_value() {
    local file="$1" key="$2" label="$3"
    dotenv_has_key "$file" "$key" || die "$label is missing from $file"
    local value
    value="$(dotenv_value "$file" "$key")"
    [ -n "$value" ] || die "$label is empty in $file"
    printf '%s\n' "$value"
}

require_github_app_inputs() {
    [ -n "${GITHUB_APP_ID:-}" ] || die "GITHUB_APP_ID is required"
    [ -n "${GITHUB_INSTALLATION_ID:-}" ] || die "GITHUB_INSTALLATION_ID is required"
    case "$GITHUB_APP_ID" in ''|*[!0-9]*) die "GITHUB_APP_ID must be positive decimal";; esac
    case "$GITHUB_INSTALLATION_ID" in ''|*[!0-9]*) die "GITHUB_INSTALLATION_ID must be positive decimal";; esac
    [ "$GITHUB_APP_ID" -gt 0 ] || die "GITHUB_APP_ID must be positive decimal"
    [ "$GITHUB_INSTALLATION_ID" -gt 0 ] || die "GITHUB_INSTALLATION_ID must be positive decimal"
    [ -n "${GITHUB_APP_PRIVATE_KEY_FILE:-}" ] || die "GITHUB_APP_PRIVATE_KEY_FILE is required"
    python3 - "$GITHUB_APP_PRIVATE_KEY_FILE" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_absolute():
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE must be an absolute path")
try:
    info = path.lstat()
except OSError:
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE does not exist")
if stat.S_ISLNK(info.st_mode):
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE must not be a symlink")
if not stat.S_ISREG(info.st_mode):
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE must be a regular file")
if info.st_uid != os.getuid():
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE must be owned by invoking user")
if os.getuid() == 0:
    raise SystemExit("deploy.sh must run as a non-root user")
if stat.S_IMODE(info.st_mode) & 0o077:
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE must not grant group/world access")
if not os.access(path, os.R_OK):
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE is not readable")
try:
    text = path.read_text(encoding="ascii")
except (OSError, UnicodeDecodeError):
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE must be PEM text")
if "PRIVATE KEY" not in text or "-----BEGIN" not in text:
    raise SystemExit("GITHUB_APP_PRIVATE_KEY_FILE must contain a PEM private key")
print("GITHUB_APP_PRIVATE_KEY_OK")
PY
}



require_private_local_inputs() {
    python3 - "$MACHINES_FILE" "$SECRETS_FILE" <<'PY'
import os
import stat
import sys
from pathlib import Path

machines_file = Path(sys.argv[1])
secrets_file = Path(sys.argv[2])
expected_uid = os.getuid()


def _check_owned_private(path: Path, *, label: str, directory: bool) -> None:
    try:
        st = path.lstat()
    except OSError:
        raise SystemExit(f"{label} is missing")
    if stat.S_ISLNK(st.st_mode):
        raise SystemExit(f"{label} must not be a symlink")
    if directory:
        if not stat.S_ISDIR(st.st_mode):
            raise SystemExit(f"{label} must be a directory")
        if not (st.st_mode & stat.S_IXUSR):
            raise SystemExit(f"{label} must be searchable by owner")
    elif not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"{label} must be a regular file")
    if st.st_uid != expected_uid:
        raise SystemExit(f"{label} must be owned by the invoking user")
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise SystemExit(f"{label} must not grant group/world permissions")


_check_owned_private(machines_file, label="machines.yaml", directory=False)
_check_owned_private(secrets_file, label="secrets.yaml", directory=False)
print("LOCAL_INPUT_PERMISSIONS_OK")
PY
}

require_machine_policy() {
    [ -f "$MACHINES_FILE" ] && [ ! -L "$MACHINES_FILE" ] || die "Machine policy file must be a regular file: $MACHINES_FILE"
    [ -r "$MACHINES_FILE" ] || die "Machine policy file not readable: $MACHINES_FILE"
    python3 - "$MACHINES_FILE" <<'PY'
import sys, yaml
from pathlib import Path
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh)
if not isinstance(data, dict) or data.get("version") != 1:
    raise SystemExit("machine policy must be a version 1 mapping")
machines = data.get("machines")
if not isinstance(machines, dict) or not machines:
    raise SystemExit("machines must be a non-empty mapping")
seen_node_ids = set()
for alias, machine in machines.items():
    if not isinstance(alias, str) or not alias.strip():
        raise SystemExit("machine aliases must be non-empty strings")
    if not isinstance(machine, dict):
        raise SystemExit(f"machine {alias!r} must be a mapping")
    node_id = machine.get("node_id")
    if not isinstance(node_id, str) or not node_id.strip():
        raise SystemExit(f"machine {alias!r} must define a non-empty node_id")
    if node_id in seen_node_ids:
        raise SystemExit(f"duplicate node_id {node_id!r}")
    seen_node_ids.add(node_id)
    owners = machine.get("owners")
    if not isinstance(owners, list) or not owners:
        raise SystemExit(f"machine {alias!r} must define a non-empty owners list")
    for owner in owners:
        if not isinstance(owner, str) or not owner.strip():
            raise SystemExit(f"machine {alias!r} has an invalid owner entry")
    node_host = machine.get("node_host")
    if not isinstance(node_host, str) or not node_host.strip():
        raise SystemExit(f"machine {alias!r} must define a non-empty node_host")
print("MACHINE_POLICY_OK")
PY
}

require_secrets() {
    [ -f "$SECRETS_FILE" ] && [ ! -L "$SECRETS_FILE" ] || die "COS secrets file must be a regular file: $SECRETS_FILE"
    [ -r "$SECRETS_FILE" ] || die "COS secrets file not readable: $SECRETS_FILE"
    python3 - "$SECRETS_FILE" <<'PY'
import sys, yaml
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh)
if not isinstance(data, dict):
    raise SystemExit("secrets file must be a mapping")
if data.get("version") != 1:
    raise SystemExit("secrets file must be version 1")
cos = data.get("cos", {})
if cos is None:
    cos = {}
if not isinstance(cos, dict):
    raise SystemExit("cos section must be a mapping")
for key in ("region", "bucket", "secret_id", "secret_key"):
    value = cos.get(key, "")
    if value is not None and not isinstance(value, str):
        raise SystemExit(f"cos.{key} must be a string")
tokens = data.get("agent_core_tokens")
if not isinstance(tokens, dict) or not tokens:
    raise SystemExit("agent_core_tokens must be a non-empty mapping")
for alias, token in tokens.items():
    if not isinstance(alias, str) or not alias.strip():
        raise SystemExit("agent_core_tokens keys must be non-empty strings")
    if not isinstance(token, str) or not token.strip():
        raise SystemExit(f"agent_core_tokens[{alias!r}] must be a non-empty string")
rct = data.get("review_comment_trust")
if rct is None:
    raise SystemExit("review_comment_trust must be present in secrets.yaml")
if not isinstance(rct, dict):
    raise SystemExit("review_comment_trust must be a mapping")
aid = rct.get("author_id", "")
if aid is None or not isinstance(aid, str):
    raise SystemExit("review_comment_trust.author_id must be a string")
al = rct.get("author_login", "")
if al is None or not isinstance(al, str):
    raise SystemExit("review_comment_trust.author_login must be a string")
if aid != "7950763":
    raise SystemExit("review_comment_trust.author_id must be \"7950763\"")
if al != "kentcyq":
    raise SystemExit("review_comment_trust.author_login must be \"kentcyq\"")
print("SECRETS_OK")
PY
}

require_runtime_inputs() {
    require_runtime_user
    require_github_app_inputs
    require_private_local_inputs
    require_machine_policy
    require_secrets
}

build_env_file() {
    local tmp
    tmp="$(mktemp)"
    chmod 600 "$tmp"
    {        printf 'DEPLOY_APPROVAL_RUNTIME_UID=%s\n' "$RUNTIME_UID"
        printf 'DEPLOY_APPROVAL_RUNTIME_GID=%s\n' "$RUNTIME_GID"
        for key in GITHUB_REPOS POLL_ENABLED POLL_INTERVAL_SECONDS WEBHOOK_ENABLED GITHUB_WEBHOOK_SECRET REGISTRY REGISTRY_USER REGISTRY_PASSWORD; do
            value="${!key:-}"
            if [ -n "$value" ]; then
                printf '%s=%s\n' "$key" "$value"
            fi
        done
    } > "$tmp"
    echo "$tmp"
}

cmd_up() {
    require_runtime_inputs
    TMP_ENV="$(build_env_file)"
    trap 'rm -f "$TMP_ENV"' EXIT
    $COMPOSE --env-file "$TMP_ENV" build
    $COMPOSE --env-file "$TMP_ENV" up -d
    echo "Deploy Approval Agent is running on http://127.0.0.1:25001"
}

cmd_rebuild() {
    require_runtime_inputs
    TMP_ENV="$(build_env_file)"
    trap 'rm -f "$TMP_ENV"' EXIT
    $COMPOSE --env-file "$TMP_ENV" build --no-cache
    $COMPOSE --env-file "$TMP_ENV" up -d --force-recreate
}

cmd_stop() { $COMPOSE stop; }
cmd_start() {
    require_runtime_inputs
    TMP_ENV="$(build_env_file)"
    trap 'rm -f "$TMP_ENV"' EXIT
    $COMPOSE --env-file "$TMP_ENV" up -d --no-build
}
cmd_restart() {
    require_runtime_inputs
    TMP_ENV="$(build_env_file)"
    trap 'rm -f "$TMP_ENV"' EXIT
    $COMPOSE --env-file "$TMP_ENV" up -d --no-build --force-recreate
}
cmd_down() { $COMPOSE down; }

cmd_status() {
    curl -sf --max-time 5 http://127.0.0.1:25001/healthz >/dev/null \
        && echo "deploy-approval: healthy" || echo "deploy-approval: not responding"
    $COMPOSE ps
}

cmd_logs() {
    $COMPOSE logs -f --tail "${1:-100}"
}

usage() { sed -n '1,22p' "$SCRIPT_PATH"; }

case "${1:-up}" in
    up|"") cmd_up ;;
    rebuild) cmd_rebuild ;;
    stop) cmd_stop ;;
    start) cmd_start ;;
    restart) cmd_restart ;;
    down) cmd_down ;;
    status) cmd_status ;;
    logs) shift; cmd_logs "$@" ;;
    -h|--help|help) usage ;;
    *) echo "Unknown command: $1" >&2; usage >&2; exit 1 ;;
esac
