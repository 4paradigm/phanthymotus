#!/usr/bin/env bash
# build_core.sh — 构建 agent-core（大脑层）镜像并推送
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/build_common.sh"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

eval "$(parse_mirror_arg "$@")"

: "${REGISTRY:?REGISTRY not set. Copy deploy/.env.example to deploy/.env and fill in values.}"
: "${REGISTRY_USER:?REGISTRY_USER not set}"
: "${REGISTRY_PASSWORD:?REGISTRY_PASSWORD not set}"
: "${IMAGE_NAMESPACE:?IMAGE_NAMESPACE not set}"

RESOURCE_CENTER_URL="${RESOURCE_CENTER_URL:-https://motus.phanthy.com}"

DATE="$(date +%y%m%d)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"
TAG="release.${DATE}.${COMMIT}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/core:${TAG}"

echo "============================================"
echo "Building agent-core image"
echo "Image : ${FULL_IMAGE}"
echo "Arch  : ${ARCH} (native=${IS_ARM64})"
echo "============================================"

echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin

select_mirror

do_build "${REPO_ROOT}/src/agent_core/Dockerfile" \
         "${REPO_ROOT}/src/agent_core" \
         "${FULL_IMAGE}" \
         "IMAGE_TAG=${TAG}"

do_push "${FULL_IMAGE}"

echo ""
echo "Done. Image pushed: ${FULL_IMAGE}"

# ── 注册到 resource-center（可选）────────────────────────────────────────────
if [ -n "${RESOURCE_CENTER_API_KEY:-}" ]; then
    SYNC_CONFIRM="y"
    if [ -t 0 ] || [ -e /dev/tty ]; then
        printf "Sync to resource-center (%s)? [Y/n]: " "${RESOURCE_CENTER_URL}" >/dev/tty
        read -r SYNC_CONFIRM </dev/tty || SYNC_CONFIRM="y"
    fi
    if [[ ! "${SYNC_CONFIRM}" =~ ^[Nn] ]]; then
        echo "Registering image to resource-center (${RESOURCE_CENTER_URL})..."
        HTTP_STATUS=$(curl -s -o /tmp/rc_register_resp.json -w "%{http_code}" \
            -X POST "${RESOURCE_CENTER_URL}/api/admin/register" \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${RESOURCE_CENTER_API_KEY}" \
            -d "{
                \"imageRef\": \"${FULL_IMAGE}\",
                \"registryImage\": \"core\",
                \"tag\": \"${TAG}\",
                \"category\": \"core\",
                \"name\": \"Agent Core\"
            }")

        if [ "${HTTP_STATUS}" = "200" ] || [ "${HTTP_STATUS}" = "201" ]; then
            echo "Registered: $(cat /tmp/rc_register_resp.json)"
        else
            echo "Warning: registration failed (HTTP ${HTTP_STATUS}): $(cat /tmp/rc_register_resp.json)"
        fi
    else
        echo "跳过同步。"
    fi
fi
