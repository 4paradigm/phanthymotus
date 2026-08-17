#!/usr/bin/env bash
# build_perception.sh — 构建 perception-stack（感知层）镜像并推送
#
# Usage:
#   ./build_perception.sh                                       # 单容器导航版（默认），交互选源
#   ./build_perception.sh --mirror tuna                         # 单容器导航版（默认），清华源
#   ./build_perception.sh --variant cpu                         # 旧 CPU 版
#   ./build_perception.sh --variant jetson                      # Jetson GPU 版, JetPack 5.11
#   ./build_perception.sh --variant jetson --jp-version 6.1     # Jetson GPU 版，JetPack 6.1
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
VARIANT="navigation"
JP_VERSION="5.11"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --jp-version) JP_VERSION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RESOURCE_CENTER_URL="${RESOURCE_CENTER_URL:-https://motus.phanthy.com}"

# If registry not configured, build locally only
PUSH_ENABLED=true
if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[info] Registry not configured — building locally only (no push)."
    PUSH_ENABLED=false
    REGISTRY="${REGISTRY:-local}"
    IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-phanthy-motus}"
fi

DATE="$(date +%y%m%d)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"

# ── 根据 variant 选择 Dockerfile、context、tag ────────────────────────
case "${VARIANT}" in
    cpu)
        DOCKERFILE="${REPO_ROOT}/perception/Dockerfile"
        BUILD_CONTEXT="${REPO_ROOT}/perception"
        TAG="release.${DATE}.${COMMIT}"
        ;;
    jetson)
        DOCKERFILE="${REPO_ROOT}/perception/Dockerfile.jetson"
        BUILD_CONTEXT="${REPO_ROOT}"
        TAG="release.${DATE}.${COMMIT}-jetson-jp${JP_VERSION}"
        ;;
    navigation)
        DOCKERFILE="${REPO_ROOT}/perception/Dockerfile.navigation"
        BUILD_CONTEXT="${REPO_ROOT}"
        TAG="release.${DATE}.${COMMIT}-navigation"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}  (supported: cpu, jetson, navigation)"
        exit 1
        ;;
esac

RESOURCE_CENTER_NAME="Perception Stack"
RESOURCE_CENTER_DESCRIPTION="语音感知套件 — ASR 语音识别 + TTS 语音合成 + VAD 静音检测 + 唤醒词检测"
if [ "${VARIANT}" = "navigation" ]; then
    RESOURCE_CENTER_NAME="Perception Stack with Navigation"
    RESOURCE_CENTER_DESCRIPTION="单容器 Navigation 卡片 — FAST-LIVO2 建图定位 + Nav2 规划控制 + 视觉语义航点"
fi

BUILD_ARGS=()
# ── 根据 jp_version 选择 base image  ────────────────────────
if [ "${VARIANT}" = "jetson" ]; then
    case "${JP_VERSION}" in
        5.11) BUILD_ARGS+=("JP_VERSION=511") ;;
        6.1) BUILD_ARGS+=("JP_VERSION=61") ;;
        *)
            echo "Unknown JetPack version: ${JP_VERSION} (support: 5.11, 6.1)"
            exit 1
            ;;
    esac
fi

if [ "${VARIANT}" = "navigation" ]; then
    source "${REPO_ROOT}/perception/plugins/navigation/runtime/fast_livo2-source-lock.env"
    source "${REPO_ROOT}/perception/plugins/navigation/runtime/nav2-source-lock.env"
    BUILD_ARGS+=(
        "ROS_BASE_IMAGE=${ROS_BASE_IMAGE}"
        "GIT_MIRROR_PREFIX=${GIT_MIRROR_PREFIX}"
        "FAST_LIVO2_REPO=${FAST_LIVO2_REPO}"
        "FAST_LIVO2_COMMIT=${FAST_LIVO2_COMMIT}"
        "FAST_LIVO2_RUNTIME_PATCH_SHA256=${FAST_LIVO2_RUNTIME_PATCH_SHA256}"
        "FAST_LIVO2_PCD_SAVE_PATCH_SHA256=${FAST_LIVO2_PCD_SAVE_PATCH_SHA256}"
        "RPG_VIKIT_REPO=${RPG_VIKIT_REPO}"
        "RPG_VIKIT_COMMIT=${RPG_VIKIT_COMMIT}"
        "LIVOX_ROS_DRIVER2_REPO=${LIVOX_ROS_DRIVER2_REPO}"
        "LIVOX_ROS_DRIVER2_COMMIT=${LIVOX_ROS_DRIVER2_COMMIT}"
        "LIVOX_SDK2_REPO=${LIVOX_SDK2_REPO}"
        "LIVOX_SDK2_COMMIT=${LIVOX_SDK2_COMMIT}"
        "SOPHUS_REPO=${SOPHUS_REPO}"
        "SOPHUS_COMMIT=${SOPHUS_COMMIT}"
        "PYTHON_COLCON_VERSION=${PYTHON_COLCON_VERSION}"
        "PYTHON_PYTEST_VERSION=${PYTHON_PYTEST_VERSION}"
        "ROS_NAV2_BRINGUP_VERSION_AMD64=${ROS_NAV2_BRINGUP_VERSION_AMD64}"
        "ROS_NAV2_BRINGUP_VERSION_ARM64=${ROS_NAV2_BRINGUP_VERSION_ARM64}"
        "ROS_NAVIGATION2_VERSION_AMD64=${ROS_NAVIGATION2_VERSION_AMD64}"
        "ROS_NAVIGATION2_VERSION_ARM64=${ROS_NAVIGATION2_VERSION_ARM64}"
        "ROS_RMW_FASTRTPS_CPP_VERSION_AMD64=${ROS_RMW_FASTRTPS_CPP_VERSION_AMD64}"
        "ROS_RMW_FASTRTPS_CPP_VERSION_ARM64=${ROS_RMW_FASTRTPS_CPP_VERSION_ARM64}"
    )
fi

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/perception:${TAG}"

echo "============================================"
echo "Building perception-stack image"
echo "Variant: ${VARIANT}"
[ "${VARIANT}" != "jetson" ] || echo "PyTorch for JetPack: JP${JP_VERSION}"
echo "Image  : ${FULL_IMAGE}"
echo "Arch   : ${ARCH} (native=${IS_ARM64})"
echo "Push   : ${PUSH_ENABLED}"
echo "============================================"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

select_mirror

do_build "${DOCKERFILE}" "${BUILD_CONTEXT}" "${FULL_IMAGE}" "${BUILD_ARGS[@]}"

if ${PUSH_ENABLED}; then
    do_push "${FULL_IMAGE}"
    echo ""
    echo "Done. Image pushed: ${FULL_IMAGE}"
else
    echo ""
    echo "Done. Image built locally: ${FULL_IMAGE}"
fi

# ── 注册到 resource-center（可选）────────────────────────────────────────────
if ${PUSH_ENABLED} && [ -n "${RESOURCE_CENTER_API_KEY:-}" ]; then
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
                \"name\": \"${RESOURCE_CENTER_NAME}\",
                \"description\": \"${RESOURCE_CENTER_DESCRIPTION}\"
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
