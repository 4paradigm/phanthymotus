#!/bin/sh
# entrypoint.sh — extract compose from new image, replace image ref, restart service
# Aligned with install.sh behavior: full compose extraction + sed replacement + clean restart
# Env: COMPOSE_DIR, SERVICE, NEW_IMAGE, CONTAINER_NAME (optional)
set -e

: "${COMPOSE_DIR:?COMPOSE_DIR required}"
: "${SERVICE:?SERVICE required}"
: "${NEW_IMAGE:?NEW_IMAGE required}"

CONTAINER_NAME="${CONTAINER_NAME:-phanthy-motus-${SERVICE}-1}"

echo "[restart] service=${SERVICE} image=${NEW_IMAGE}"
echo "[restart] compose_dir=${COMPOSE_DIR}"
echo "[restart] container=${CONTAINER_NAME}"

COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"

# ── Step 1: Extract compose from new image (single source of truth) ──────────
echo "[restart] extracting compose from new image..."
CID=$(docker create "${NEW_IMAGE}")
docker cp "${CID}:/deploy/docker-compose.yml" /tmp/new-compose.yml 2>/dev/null || true
docker rm "${CID}" >/dev/null

if [ ! -f /tmp/new-compose.yml ]; then
    echo "[restart] ERROR: /deploy/docker-compose.yml not found in image"
    exit 1
fi

# ── Step 2: Update compose file ──────────────────────────────────────────────
# Merge the target service's new definition into the existing compose, preserving
# every other service. Held under the same lock agent-core's driver deploy takes
# (agent-core/src/api/drivers.py :: _compose_lock) — both write this one
# host-mounted file, and without the lock two overlapping open(...,'w') writers
# leave a truncated document that no parser accepts.
#
# The old code had an `else` branch that shutil.copy2'd the image's template over
# the host compose whenever it saw no other services — including when it read a
# momentarily-truncated file. That silently deleted every other service. There is
# no wholesale-overwrite path any more: a compose we cannot parse aborts the
# upgrade instead of being replaced.

LOCK_FILE="${COMPOSE_DIR}/.compose.lock"
exec 9>"${LOCK_FILE}"
if ! flock -w 120 9; then
    echo "[restart] ERROR: timed out waiting for ${LOCK_FILE}; a driver deploy is in progress"
    exit 1
fi

python3 - "${COMPOSE_FILE}" /tmp/new-compose.yml "${NEW_IMAGE}" "${SERVICE}" <<'PY'
import os, sys, yaml

compose_path, new_path, new_image, service = sys.argv[1:5]


def die(msg):
    sys.stderr.write(f'[restart] ERROR: {msg}\n')
    sys.exit(1)


with open(new_path) as f:
    new = yaml.safe_load(f) or {}
new_services = new.get('services') or {}
if service not in new_services:
    die(f'{new_path} has no service {service!r}')

try:
    with open(compose_path) as f:
        raw = f.read()
except FileNotFoundError:
    # Genuine fresh install — nothing to preserve.
    raw, existing = '', {}
else:
    # Existing-but-empty means truncated or damaged, not fresh.
    if not raw.strip():
        die(f'{compose_path} exists but is empty — refusing to overwrite')
    try:
        existing = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        die(f'{compose_path} is not valid YAML, refusing to overwrite: {e}')
    if not isinstance(existing, dict):
        die(f'{compose_path} is not a mapping, refusing to overwrite')

existing_services = existing.setdefault('services', {})
if not isinstance(existing_services, dict):
    die(f'{compose_path}: services is not a mapping, refusing to overwrite')

preserved = set(existing_services)
new_services[service]['image'] = new_image
existing_services[service] = new_services[service]

text = yaml.dump(existing, default_flow_style=False, allow_unicode=True, sort_keys=False)

# Confirm the result parses and kept every service we started with.
try:
    check = yaml.safe_load(text) or {}
except yaml.YAMLError as e:
    die(f'refusing to write unparseable compose: {e}')
missing = preserved - set(check.get('services') or {})
if missing:
    die(f'refusing to write: would drop service(s) {sorted(missing)}')

if raw:
    with open(compose_path + '.bak', 'w') as f:
        f.write(raw)

tmp = compose_path + '.tmp'
with open(tmp, 'w') as f:
    f.write(text)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, compose_path)
PY

flock -u 9

echo "[restart] compose file updated"

# ── Step 3: Stop and remove old container (clean slate, same as install.sh) ──
echo "[restart] stopping old container..."
cd "${COMPOSE_DIR}"
docker compose stop "${SERVICE}" 2>/dev/null || true
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ── Step 4: Start service ────────────────────────────────────────────────────
echo "[restart] starting service..."
docker compose up -d "${SERVICE}"

echo "[restart] done. ${SERVICE} is up."
