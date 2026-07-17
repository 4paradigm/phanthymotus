#!/usr/bin/env bash
# build_inspection.sh — build the Inspection Stack image and optionally register it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/build_common.sh"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

eval "$(parse_mirror_arg "$@")"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: deploy/build_inspection.sh [--mirror tencent|tuna|none]"
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "Unknown option: $1"
    exit 1
fi

RESOURCE_CENTER_URL="${RESOURCE_CENTER_URL:-https://motus.phanthy.com}"
PUSH_ENABLED=true
if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[info] Registry not configured — building locally only (no push)."
    PUSH_ENABLED=false
    REGISTRY="${REGISTRY:-local}"
    IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-phanthy-motus}"
fi

DATE="$(date +%y%m%d)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"
TAG="release.${DATE}.${COMMIT}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/inspection:${TAG}"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

select_mirror
do_build "${REPO_ROOT}/inspection/Dockerfile" "${REPO_ROOT}/inspection" "${FULL_IMAGE}"

if ${PUSH_ENABLED}; then
    do_push "${FULL_IMAGE}"
fi

echo "Done: ${FULL_IMAGE}"

if ${PUSH_ENABLED} && [ -n "${RESOURCE_CENTER_API_KEY:-}" ]; then
    HTTP_STATUS=$(curl -s -o /tmp/rc_register_inspection_resp.json -w "%{http_code}" \
        -X POST "${RESOURCE_CENTER_URL}/api/admin/register" \
        -H "Content-Type: application/json" \
        -H "x-api-key: ${RESOURCE_CENTER_API_KEY}" \
        -d "{\"imageRef\":\"${FULL_IMAGE}\",\"registryImage\":\"inspection\",\"tag\":\"${TAG}\",\"category\":\"inspection\",\"name\":\"Inspection Stack\"}")
    if [ "${HTTP_STATUS}" != "200" ] && [ "${HTTP_STATUS}" != "201" ]; then
        echo "Warning: registration failed (HTTP ${HTTP_STATUS}): $(cat /tmp/rc_register_inspection_resp.json)"
    fi
fi
