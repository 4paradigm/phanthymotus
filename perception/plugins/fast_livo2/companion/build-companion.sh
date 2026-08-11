#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${SCRIPT_DIR}/source-lock.env"

[[ -f "${LOCK_FILE}" ]] || {
  printf 'ERROR=source lock does not exist: %s\n' "${LOCK_FILE}" >&2
  exit 1
}

requested_image="${FAST_LIVO2_IMAGE:-}"
set -a
# shellcheck disable=SC1090
source "${LOCK_FILE}"
set +a
if [[ -n "${requested_image}" ]]; then
  FAST_LIVO2_IMAGE="${requested_image}"
  export FAST_LIVO2_IMAGE
fi

[[ -n "${FAST_LIVO2_BASE_IMAGE:-}" ]] || {
  printf 'ERROR=FAST_LIVO2_BASE_IMAGE is missing from source-lock.env\n' >&2
  exit 1
}
[[ -n "${FAST_LIVO2_BASE_IMAGE_ID:-}" ]] || {
  printf 'ERROR=FAST_LIVO2_BASE_IMAGE_ID is missing from source-lock.env\n' >&2
  exit 1
}

actual_id="$(docker image inspect \
  --format '{{.Id}}' "${FAST_LIVO2_BASE_IMAGE}" 2>/dev/null || true)"
[[ -n "${actual_id}" ]] || {
  printf 'ERROR=locked FAST-LIVO2 base image is absent: %s\n' \
    "${FAST_LIVO2_BASE_IMAGE}" >&2
  exit 1
}
[[ "${actual_id}" == "${FAST_LIVO2_BASE_IMAGE_ID}" ]] || {
  printf 'ERROR=FAST-LIVO2 base image ID mismatch: expected=%s actual=%s\n' \
    "${FAST_LIVO2_BASE_IMAGE_ID}" "${actual_id}" >&2
  exit 1
}

cd "${SCRIPT_DIR}"
docker compose --env-file "${LOCK_FILE}" build fast_livo2
