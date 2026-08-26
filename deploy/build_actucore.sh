#!/usr/bin/env bash
# build_actucore.sh — 构建 actucore（执行模型层）镜像并推送
#
# 只有 Jetson 版：执行模型多数要 GPU（VLA / 抓取策略 / locomotion），
# 没有 CPU 变体。navigation 卡片本身不用 GPU，但和它们共用这一个镜像。
#
# 镜像里会从锁定源码编译 FAST-LIVO2 与 Nav2（base 是 Focal，ros-humble-*
# 只有 Jammy 的 Debian 包），首次构建约 1-3 小时，之后走 layer 缓存。
#
# Usage:
#   ./build_actucore.sh                          # JetPack 5.11（默认），交互选源
#   ./build_actucore.sh --jp-version 6.1         # JetPack 6.1
#   ./build_actucore.sh --mirror tuna
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
JP_VERSION="5.11"
while [[ $# -gt 0 ]]; do
    case "$1" in
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

# ── Jetson-only：执行模型都要 GPU，没有 CPU 变体 ──────────────────────
DOCKERFILE="${REPO_ROOT}/actucore/Dockerfile.jetson"
BUILD_CONTEXT="${REPO_ROOT}"
TAG="release.${DATE}.${COMMIT}-jetson-jp${JP_VERSION}"

BUILD_ARGS=()
# ── 根据 jp_version 选择 base image  ────────────────────────
# 表在 build_common.sh 的 jetpack_vars 里，build_perception.sh 共用同一份。
jetpack_vars "${JP_VERSION}" || exit 1
BUILD_ARGS+=("JP_VERSION=${JP_ARG}")

# Dockerfile.jetson 基于 L4T base image —— 只有 arm64
CPU_ARCH="arm64"

# ── navigation 卡片的源码锁 ──────────────────────────────────────────
# FAST-LIVO2 与 Nav2 都在本镜像里从锁定源码编译（base 是 Focal，没有
# ros-humble-* 的 Debian 包可用），所以每个 revision 都要作为 build arg
# 传进去，Dockerfile 里再校验成完整 SHA。
NAV_RUNTIME_DIR="${REPO_ROOT}/actucore/plugins/navigation/runtime"
source "${NAV_RUNTIME_DIR}/fast_livo2-source-lock.env"
source "${NAV_RUNTIME_DIR}/nav2-source-lock.env"
BUILD_ARGS+=(
    "GIT_MIRROR_PREFIX=${GIT_MIRROR_PREFIX}"
    "FAST_LIVO2_REPO=${FAST_LIVO2_REPO}"
    "FAST_LIVO2_COMMIT=${FAST_LIVO2_COMMIT}"
    "FAST_LIVO2_RUNTIME_PATCH_SHA256=${FAST_LIVO2_RUNTIME_PATCH_SHA256}"
    "FAST_LIVO2_PCD_SAVE_PATCH_SHA256=${FAST_LIVO2_PCD_SAVE_PATCH_SHA256}"
    "FAST_LIVO2_PCD_FLUSH_PATCH_SHA256=${FAST_LIVO2_PCD_FLUSH_PATCH_SHA256}"
    "RPG_VIKIT_REPO=${RPG_VIKIT_REPO}"
    "RPG_VIKIT_COMMIT=${RPG_VIKIT_COMMIT}"
    "SOPHUS_REPO=${SOPHUS_REPO}"
    "SOPHUS_COMMIT=${SOPHUS_COMMIT}"
    "NAVIGATION2_REPO=${NAVIGATION2_REPO}"
    "NAVIGATION2_COMMIT=${NAVIGATION2_COMMIT}"
    "BEHAVIORTREE_CPP_REPO=${BEHAVIORTREE_CPP_REPO}"
    "BEHAVIORTREE_CPP_COMMIT=${BEHAVIORTREE_CPP_COMMIT}"
    "ANGLES_REPO=${ANGLES_REPO}"
    "ANGLES_COMMIT=${ANGLES_COMMIT}"
    "BOND_CORE_REPO=${BOND_CORE_REPO}"
    "BOND_CORE_COMMIT=${BOND_CORE_COMMIT}"
    "DIAGNOSTICS_REPO=${DIAGNOSTICS_REPO}"
    "DIAGNOSTICS_COMMIT=${DIAGNOSTICS_COMMIT}"
    "NAVIGATION_MSGS_REPO=${NAVIGATION_MSGS_REPO}"
    "NAVIGATION_MSGS_COMMIT=${NAVIGATION_MSGS_COMMIT}"
    "LASER_GEOMETRY_REPO=${LASER_GEOMETRY_REPO}"
    "LASER_GEOMETRY_COMMIT=${LASER_GEOMETRY_COMMIT}"
)

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/actucore:${TAG}"

echo "============================================"
echo "Building actucore image (Jetson only)"
echo "PyTorch for JetPack: JP${JP_VERSION}"
echo "Image  : ${FULL_IMAGE}"
echo "Arch   : ${ARCH} (native=${IS_ARM64})"
echo "Runs on: ${ACC_ARCH} / ${CPU_ARCH}"
echo "Push   : ${PUSH_ENABLED}"
echo "============================================"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

select_mirror

BUILD_STARTED_AT="$(date +%s)"
do_build "${DOCKERFILE}" "${BUILD_CONTEXT}" "${FULL_IMAGE}" "${BUILD_ARGS[@]}"
echo "ACTUCORE_BUILD_DURATION_SEC=$(( $(date +%s) - BUILD_STARTED_AT ))"

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
    # Ask only if there is a terminal to ask on; otherwise sync (the key being
    # set is the opt-in). Test by opening /dev/tty, not with `[ -e ]`: the device
    # node exists in any container, but opening it without a controlling
    # terminal fails with ENXIO — which under `set -e` aborted the whole script
    # here, reporting a successful build as failed.
    SYNC_CONFIRM="y"
    if { : >/dev/tty; } 2>/dev/null; then
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
                \"registryImage\": \"actucore\",
                \"tag\": \"${TAG}\",
                \"category\": \"actucore\",
                \"acc_arch\": \"${ACC_ARCH}\",
                \"cpu_arch\": \"${CPU_ARCH}\",
                \"name\": \"ActuCore\",
                \"port\": 15730,
                \"description\": \"执行模型层 — VLA 策略 / 导航 / 抓取 / locomotion / 全身控制，以 processor 卡片接入\"
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
