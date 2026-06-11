#!/usr/bin/env bash
# build_perception.sh — 构建 perception-stack（感知层）镜像并推送
#
# Usage:
#   ./build_perception.sh                           # CPU 版（默认），交互选源
#   ./build_perception.sh --variant jetson          # Jetson GPU 版
#   ./build_perception.sh --variant jetson --mirror tuna
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/build_common.sh"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

eval "$(parse_mirror_arg "$@")"

# ── 解析参数 ─────────────────────────────────────────────────────────
VARIANT="cpu"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

: "${REGISTRY:?REGISTRY not set. Copy deploy/.env.example to deploy/.env and fill in values.}"
: "${REGISTRY_USER:?REGISTRY_USER not set}"
: "${REGISTRY_PASSWORD:?REGISTRY_PASSWORD not set}"
: "${IMAGE_NAMESPACE:?IMAGE_NAMESPACE not set}"

RESOURCE_CENTER_URL="${RESOURCE_CENTER_URL:-https://motus.phanthy.com}"

DATE="$(date +%y%m%d)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"

# ── 根据 variant 选择 Dockerfile、context、tag ────────────────────────
case "${VARIANT}" in
    cpu)
        DOCKERFILE="${REPO_ROOT}/src/perception_stack/Dockerfile"
        BUILD_CONTEXT="${REPO_ROOT}/src/perception_stack"
        TAG="release.${DATE}.${COMMIT}"
        ;;
    jetson)
        DOCKERFILE="${REPO_ROOT}/src/perception_stack/Dockerfile.jetson"
        BUILD_CONTEXT="${REPO_ROOT}"  # 需要访问 deploy/ros-base/audio_msgs/
        TAG="release.${DATE}.${COMMIT}-jetson"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}  (supported: cpu, jetson)"
        exit 1
        ;;
esac

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/perception:${TAG}"

echo "============================================"
echo "Building perception-stack image"
echo "Variant: ${VARIANT}"
echo "Image  : ${FULL_IMAGE}"
echo "Arch   : ${ARCH} (native=${IS_ARM64})"
echo "============================================"

echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin

select_mirror

do_build "${DOCKERFILE}" "${BUILD_CONTEXT}" "${FULL_IMAGE}"

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
                \"registryImage\": \"perception\",
                \"tag\": \"${TAG}\",
                \"category\": \"perception\",
                \"name\": \"Perception Stack\"
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
