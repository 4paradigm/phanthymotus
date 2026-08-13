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

for required in \
  FAST_LIVO2_BASE_IMAGE \
  FAST_LIVO2_COMMIT \
  FAST_LIVO2_RUNTIME_PATCH_SHA256 \
  FAST_LIVO2_PCD_SAVE_PATCH_SHA256
do
  [[ -n "${!required:-}" ]] || {
    printf 'ERROR=%s is missing from source-lock.env\n' "${required}" >&2
    exit 1
  }
done

metadata="$(docker image inspect --format \
  '{{.Architecture}}|{{ index .Config.Labels "org.opencontainers.image.revision" }}|{{ index .Config.Labels "org.opencontainers.image.fast-livo2-runtime-patch" }}|{{ index .Config.Labels "org.opencontainers.image.fast-livo2-pcd-save-patch" }}' \
  "${FAST_LIVO2_BASE_IMAGE}" 2>/dev/null || true)"
[[ -n "${metadata}" ]] || {
  printf 'ERROR=locked FAST-LIVO2 base image is absent: %s\n' \
    "${FAST_LIVO2_BASE_IMAGE}" >&2
  exit 1
}

IFS='|' read -r actual_arch actual_revision actual_runtime_patch actual_pcd_patch \
  <<<"${metadata}"
[[ "${actual_arch}" == "arm64" ]] || {
  printf 'ERROR=FAST-LIVO2 base image architecture mismatch: expected=arm64 actual=%s\n' \
    "${actual_arch}" >&2
  exit 1
}
[[ "${actual_revision}" == "${FAST_LIVO2_COMMIT}" ]] || {
  printf 'ERROR=FAST-LIVO2 revision mismatch: expected=%s actual=%s\n' \
    "${FAST_LIVO2_COMMIT}" "${actual_revision}" >&2
  exit 1
}
[[ "${actual_runtime_patch}" == "${FAST_LIVO2_RUNTIME_PATCH_SHA256}" ]] || {
  printf 'ERROR=FAST-LIVO2 runtime patch mismatch: expected=%s actual=%s\n' \
    "${FAST_LIVO2_RUNTIME_PATCH_SHA256}" "${actual_runtime_patch}" >&2
  exit 1
}
[[ "${actual_pcd_patch}" == "${FAST_LIVO2_PCD_SAVE_PATCH_SHA256}" ]] || {
  printf 'ERROR=FAST-LIVO2 PCD patch mismatch: expected=%s actual=%s\n' \
    "${FAST_LIVO2_PCD_SAVE_PATCH_SHA256}" "${actual_pcd_patch}" >&2
  exit 1
}

cd "${SCRIPT_DIR}"
docker compose --env-file "${LOCK_FILE}" build fast_livo2
