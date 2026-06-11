#!/usr/bin/env bash
# build_ros_base.sh — 构建 ROS2 + audio_msgs 基础镜像并推送
#
# Usage:
#   ./build_ros_base.sh                # 交互选源
#   ./build_ros_base.sh --mirror tuna  # 指定源
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/build_common.sh"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

eval "$(parse_mirror_arg "$@")"

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
TAG="release.${DATE}.${COMMIT}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/ros-base:${TAG}"
LATEST_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/ros-base:latest"

echo "============================================"
echo "Building ros-base image"
echo "Image : ${FULL_IMAGE}"
echo "Latest: ${LATEST_IMAGE}"
echo "Arch  : ${ARCH} (native=${IS_ARM64})"
echo "Push  : ${PUSH_ENABLED}"
echo "============================================"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

select_mirror

do_build "${SCRIPT_DIR}/ros-base/Dockerfile" \
         "${SCRIPT_DIR}/ros-base" \
         "${FULL_IMAGE}"

# Tag latest
docker tag "${FULL_IMAGE}" "${LATEST_IMAGE}"

if ${PUSH_ENABLED}; then
    do_push "${FULL_IMAGE}"
    do_push "${LATEST_IMAGE}"
    echo ""
    echo "Done. Images pushed:"
    echo "  ${FULL_IMAGE}"
    echo "  ${LATEST_IMAGE}"
else
    echo ""
    echo "Done. Images built locally:"
    echo "  ${FULL_IMAGE}"
    echo "  ${LATEST_IMAGE}"
fi
