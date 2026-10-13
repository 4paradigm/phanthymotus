#!/usr/bin/env bash
# deploy.sh — manage the Deploy Approval Agent container.
#
# Commands:
#   ./deploy.sh up|rebuild|start|stop|restart|down|status|logs|preflight
#
# Fixed persistent inputs (survive release directory changes):
#   machine policy: ${DEPLOY_APPROVAL_CONFIG_DIR}/machines.yaml
#   COS secrets:    ${DEPLOY_APPROVAL_CONFIG_DIR}/secrets.yaml
# DEPLOY_APPROVAL_CONFIG_DIR must be set before running this script.
#
# No purge mode, no runtime API token, no env indirection for file locations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/deploy.sh"
COMPOSE=(docker compose -f "$SCRIPT_DIR/docker-compose.yml")

# ── Portable configuration directory ────────────────────────────────
# DEPLOY_APPROVAL_CONFIG_DIR must be set to an absolute directory
# containing machines.yaml and secrets.yaml.  This makes the deploy
# script portable across development, test, and production machines
# without hardcoding any server-specific path.
PERSISTENT_DIR="${DEPLOY_APPROVAL_CONFIG_DIR:?DEPLOY_APPROVAL_CONFIG_DIR is required — set it to the absolute directory containing machines.yaml and secrets.yaml}"
# Reject relative paths — DEPLOY_APPROVAL_CONFIG_DIR must be absolute
die() { echo "ERROR: $*" >&2; exit 1; }
case "$PERSISTENT_DIR" in /*) ;; *) die "DEPLOY_APPROVAL_CONFIG_DIR must be an absolute path, got: $PERSISTENT_DIR" ;; esac
MACHINES_FILE="${PERSISTENT_DIR}/machines.yaml"
SECRETS_FILE="${PERSISTENT_DIR}/secrets.yaml"

TMP_ENV=""

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
    python3 - "$MACHINES_FILE" "$SECRETS_FILE" "$PERSISTENT_DIR" <<'PY'
import os
import re
import stat
import sys
from pathlib import Path

machines_file = Path(sys.argv[1])
secrets_file = Path(sys.argv[2])
persistent_dir = Path(sys.argv[3])
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


# --- machines.yaml ---
_check_owned_private(machines_file, label="machines.yaml", directory=False)

# --- secrets.yaml ---
_check_owned_private(secrets_file, label="secrets.yaml", directory=False)

print("LOCAL_INPUT_PERMISSIONS_OK")
PY
}

require_machine_policy() {
    [ -f "$MACHINES_FILE" ] && [ ! -L "$MACHINES_FILE" ] || die "Machine policy file must be a regular file: $MACHINES_FILE"
    [ -r "$MACHINES_FILE" ] || die "Machine policy file not readable: $MACHINES_FILE"
    python3 - "$MACHINES_FILE" <<'PY'
import sys, yaml, re
from pathlib import Path

path = Path(sys.argv[1])
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
    # ── Reject unreplaced placeholder values ──────────────────────────
    # Common placeholder patterns that indicate the YAML was copied from
    # an example but never filled in with real values.
    placeholder_re = re.compile(r'<[^>]+>', re.IGNORECASE)
    for field_name in ("node_id", "node_host"):
        val = machine.get(field_name, "")
        if isinstance(val, str) and placeholder_re.search(val):
            raise SystemExit(
                f"machine {alias!r} field '{field_name}' contains an "
                f"unreplaced placeholder: {val!r}"
            )
    for platform in machine.get("platforms", []):
        if isinstance(platform, str) and placeholder_re.search(platform):
            raise SystemExit(
                f"machine {alias!r} has an unreplaced platform placeholder: "
                f"{platform!r}"
            )
    for variant in machine.get("variants", []):
        if isinstance(variant, str) and placeholder_re.search(variant):
            raise SystemExit(
                f"machine {alias!r} has an unreplaced variant placeholder: "
                f"{variant!r}"
            )
    for target in machine.get("targets", []):
        if isinstance(target, str) and placeholder_re.search(target):
            raise SystemExit(
                f"machine {alias!r} has an unreplaced target placeholder: "
                f"{target!r}"
            )
print("MACHINE_POLICY_OK")
PY
}

require_secrets() {
    [ -f "$SECRETS_FILE" ] && [ ! -L "$SECRETS_FILE" ] || die "COS secrets file must be a regular file: $SECRETS_FILE"
    [ -r "$SECRETS_FILE" ] || die "COS secrets file not readable: $SECRETS_FILE"
    python3 - "$SECRETS_FILE" <<'PY'
import sys, yaml, re
from pathlib import Path

path = Path(sys.argv[1])
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

# ── Region validation ───────────────────────────────────────────────
region = cos.get("region", "")
if not isinstance(region, str) or not region.strip():
    raise SystemExit("cos.region must be a non-empty string")
# Tencent COS regions match patterns like "ap-beijing", "ap-guangzhou", etc.
if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]*", region.strip()):
    raise SystemExit(f"cos.region has invalid format: {region!r}")

# ── Bucket validation ───────────────────────────────────────────────
bucket = cos.get("bucket", "")
if not isinstance(bucket, str) or not bucket.strip():
    raise SystemExit("cos.bucket must be a non-empty string")
# Buckets are alphanumeric with hyphens, typically 10-23 digits for Tencent
if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", bucket.strip()):
    raise SystemExit(f"cos.bucket has invalid format: {bucket!r}")

# ── Secret ID / Secret Key — reject obvious placeholders ─────────────
placeholder_re = re.compile(r'<[^>]+>', re.IGNORECASE)
for key in ("secret_id", "secret_key"):
    value = cos.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"cos.{key} must be a non-empty string")
    if placeholder_re.search(value):
        raise SystemExit(
            f"cos.{key} contains an unreplaced placeholder"
        )

# ── Agent Core tokens ───────────────────────────────────────────────
tokens = data.get("agent_core_tokens")
if not isinstance(tokens, dict) or not tokens:
    raise SystemExit("agent_core_tokens must be a non-empty mapping")
for alias, token in tokens.items():
    if not isinstance(alias, str) or not alias.strip():
        raise SystemExit("agent_core_tokens keys must be non-empty strings")
    if not isinstance(token, str) or not token.strip():
        raise SystemExit(f"agent_core_tokens[{alias!r}] must be a non-empty string")
    if placeholder_re.search(token):
        raise SystemExit(
            f"agent_core_tokens[{alias!r}] contains an unreplaced placeholder"
        )

# ── Review comment trust ────────────────────────────────────────────
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
        for key in GITHUB_REPOS POLL_ENABLED POLL_INTERVAL_SECONDS WEBHOOK_ENABLED GITHUB_WEBHOOK_SECRET; do
            value="${!key:-}"
            if [ -n "$value" ]; then
                printf '%s=%s\n' "$key" "$value"
            fi
        done
    } > "$tmp"
    echo "$tmp"
}

cmd_preflight() {
    # Read-only preflight: validate persistent configuration directory,
    # machine policy, COS secrets, and private key — without building
    # or starting any container.
    # DEPLOY_APPROVAL_CONFIG_DIR must be set; no fallback to hardcoded path.
    local PREFLIGHT_DIR="${DEPLOY_APPROVAL_CONFIG_DIR:?DEPLOY_APPROVAL_CONFIG_DIR is required — set it to the absolute directory containing machines.yaml and secrets.yaml}"
    # Reject relative paths at preflight boundary too
    case "$PREFLIGHT_DIR" in /*) ;; *) echo "FAIL: PREFLIGHT_DIR must be an absolute path"; exit 1 ;; esac
    local PREFLIGHT_MACHINES="${PREFLIGHT_DIR}/machines.yaml"
    local PREFLIGHT_SECRETS="${PREFLIGHT_DIR}/secrets.yaml"

    echo "=== Deploy Approval Preflight ==="
    echo "Persistent directory : $PREFLIGHT_DIR"
    echo "Machines file        : $PREFLIGHT_MACHINES"
    echo "Secrets file         : $PREFLIGHT_SECRETS"
    echo ""

    # 1. Persistent directory check
    if [ ! -d "$PREFLIGHT_DIR" ]; then
        echo "FAIL: Persistent directory does not exist: $PREFLIGHT_DIR"
        exit 1
    fi
    if [ -L "$PREFLIGHT_DIR" ]; then
        echo "FAIL: Persistent directory must not be a symlink: $PREFLIGHT_DIR"
        exit 1
    fi
    echo "OK: Persistent directory exists and is not a symlink"

    # 2. machines.yaml check (matches Compose mount path)
    if [ ! -f "$PREFLIGHT_MACHINES" ] || [ -L "$PREFLIGHT_MACHINES" ]; then
        echo "FAIL: Machine policy file must be a regular file (not missing/symlink): $PREFLIGHT_MACHINES"
        exit 1
    fi
    if [ ! -r "$PREFLIGHT_MACHINES" ]; then
        echo "FAIL: Machine policy file not readable: $PREFLIGHT_MACHINES"
        exit 1
    fi
    # Check path matches Compose mount target
    local expected_mount="/run/deploy-approval/machines.yaml"
    echo "OK: Machine policy file exists at expected path (mounted to $expected_mount)"

    # 3. secrets.yaml check (matches Compose mount path)
    if [ ! -f "$PREFLIGHT_SECRETS" ] || [ -L "$PREFLIGHT_SECRETS" ]; then
        echo "FAIL: COS secrets file must be a regular file (not missing/symlink): $PREFLIGHT_SECRETS"
        exit 1
    fi
    if [ ! -r "$PREFLIGHT_SECRETS" ]; then
        echo "FAIL: COS secrets file not readable: $PREFLIGHT_SECRETS"
        exit 1
    fi
    local expected_secrets_mount="/run/deploy-approval/secrets.yaml"
    echo "OK: COS secrets file exists at expected path (mounted to $expected_secrets_mount)"

    # 4. Permission checks (same as require_private_local_inputs)
    SECRETS_FILE="$PREFLIGHT_SECRETS" MACHINES_FILE="$PREFLIGHT_MACHINES" PERSISTENT_DIR="$PREFLIGHT_DIR" require_private_local_inputs
    echo "OK: Private input permissions verified"

    # 5. Machine policy content validation
    require_machine_policy
    echo "OK: Machine policy content valid"

    # 6. Secrets content validation
    require_secrets
    echo "OK: COS secrets content valid"

    # 7. GitHub App key validation — REQUIRED
    require_github_app_inputs
    echo "OK: GitHub App private key valid"

    echo ""
    echo "=== Preflight PASSED — configuration is ready ==="
}

cmd_up() {
    cmd_preflight
    require_runtime_inputs
    TMP_ENV="$(build_env_file)"
    trap 'rm -f "$TMP_ENV"' EXIT
    "${COMPOSE[@]}" --env-file "$TMP_ENV" build
    "${COMPOSE[@]}" --env-file "$TMP_ENV" up -d
    echo "Deploy Approval Agent is running on http://127.0.0.1:25001"
}

cmd_rebuild() {
    cmd_preflight
    require_runtime_inputs
    TMP_ENV="$(build_env_file)"
    trap 'rm -f "$TMP_ENV"' EXIT
    "${COMPOSE[@]}" --env-file "$TMP_ENV" build --no-cache
    "${COMPOSE[@]}" --env-file "$TMP_ENV" up -d --force-recreate
}

cmd_stop() { "${COMPOSE[@]}" stop; }
cmd_start() {
    cmd_preflight
    require_runtime_inputs
    TMP_ENV="$(build_env_file)"
    trap 'rm -f "$TMP_ENV"' EXIT
    "${COMPOSE[@]}" --env-file "$TMP_ENV" up -d --no-build
}
cmd_restart() {
    cmd_preflight
    require_runtime_inputs
    TMP_ENV="$(build_env_file)"
    trap 'rm -f "$TMP_ENV"' EXIT
    "${COMPOSE[@]}" --env-file "$TMP_ENV" up -d --no-build --force-recreate
}
cmd_down() { "${COMPOSE[@]}" down; }

cmd_status() {
    curl -sf --max-time 5 http://127.0.0.1:25001/healthz >/dev/null \
        && echo "deploy-approval: healthy" || echo "deploy-approval: not responding"
    "${COMPOSE[@]}" ps
}

cmd_logs() {
    "${COMPOSE[@]}" logs -f --tail "${1:-100}"
}

usage() { sed -n '1,22p' "$SCRIPT_PATH"; }

case "${1:-up}" in
    up|"") cmd_up ;;
    preflight) cmd_preflight ;;
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
