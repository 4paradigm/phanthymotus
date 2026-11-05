#!/usr/bin/env bash
# build_actucore.sh — 构建 actucore（执行模型层）镜像并推送
#
# 只有 Jetson 版：执行模型多数要 GPU（VLA / 抓取策略 / locomotion），
# 没有 CPU 变体。navigation 卡片本身不用 GPU，但和它们共用这一个镜像。
#
# FAST-LIVO2 与 Nav2 在 ActuCore base 中从锁定源码编译；JP5.11 日常
# 镜像直接继承版本化基座并编译仓库自有 ROS 包。
#
# Usage:
#   ./build_actucore.sh                          # JetPack 5.11（默认），交互选源
#   ./build_actucore.sh --jp-version 6.1         # VLA / navi，保留上游本地推理
#   ./build_actucore.sh --mirror tuna
#   ./prepare_navigation_base.sh --mirror tuna  # 依赖变化时构建 navigation base
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/build_common.sh"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

# Compatibility for existing callers; the separate entry owns base builds.
for arg in "$@"; do
    if [ "${arg}" = "--base" ]; then
        args=()
        for item in "$@"; do [ "${item}" = "--base" ] || args+=("${item}"); done
        exec bash "${SCRIPT_DIR}/prepare_navigation_base.sh" "${args[@]}"
    fi
done

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
BUILD_CONTEXT="${REPO_ROOT}"
BUILD_JOBS="${BUILD_JOBS:-4}"
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR=BUILD_JOBS must be a positive integer" >&2
    exit 2
fi
BUILD_ARGS=("BUILD_JOBS=${BUILD_JOBS}")
# ── 根据 jp_version 选择 base image  ────────────────────────
# 表在 build_common.sh 的 jetpack_vars 里，build_perception.sh 共用同一份。
jetpack_vars "${JP_VERSION}" || exit 1

# Dockerfile.jetson 基于 L4T base image —— 只有 arm64
CPU_ARCH="arm64"
BASE_REPOSITORY="${BASE_REGISTRY:-bj-warehouse.tencentcloudcr.com}/${BASE_NAMESPACE:-phanthy-motus}"
DOCKERFILE="${REPO_ROOT}/actucore/Dockerfile.jetson"
IMAGE_NAME="actucore"
TAG="release.${DATE}.${COMMIT}-jetson-jp${JP_VERSION}"
CARDS_JSON='[{"name":"controlled_semantic_spatial","type":"processor"},{"name":"vla","type":"processor"},{"name":"navi","type":"processor"}]'
if [ "${JP_VERSION}" = "6.1" ]; then
    if [ -n "${ACTUCORE_NAVIGATION_BASE_IMAGE:-}" ]; then
        echo "ERROR=JP6.1 navigation runtime is not supported; build VLA without a navigation override" >&2
        exit 2
    fi
    BASE_IMAGE="${BASE_REPOSITORY}/jetson-base-actucore:jp${JP_ARG}-torch"
    CARDS_JSON='[{"name":"vla","type":"processor"},{"name":"navi","type":"processor"}]'
    echo "[info] JP6.1: VLA providers enabled; navigation is unavailable on this image."
else
    BASE_IMAGE="${ACTUCORE_NAVIGATION_BASE_IMAGE:-${BASE_REPOSITORY}/actucore-navigation-base:jp511-v1}"
    BASE_TAG="${BASE_IMAGE##*/}"
    if [[ "${BASE_TAG}" != *:* || "${BASE_TAG}" == *:latest || "${BASE_TAG}" == *: || "${BASE_IMAGE}" == *@* ]]; then
        echo "ERROR=ACTUCORE_NAVIGATION_BASE_IMAGE must use a versioned, non-latest tag" >&2
        exit 2
    fi
fi
BUILD_ARGS+=("JP_VERSION=${JP_ARG}" "BASE_IMAGE=${BASE_IMAGE}")

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}:${TAG}"

echo "============================================"
echo "Building ${IMAGE_NAME} image (Jetson only)"
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
                \"description\": \"执行模型层（导航支持 JP5.11；VLA 按 JetPack 提供模型后端）\",
                \"cards\": ${CARDS_JSON}
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
