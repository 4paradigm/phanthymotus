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

: "${REGISTRY:?REGISTRY not set. Copy deploy/.env.example to deploy/.env and fill in values.}"
: "${REGISTRY_USER:?REGISTRY_USER not set}"
: "${REGISTRY_PASSWORD:?REGISTRY_PASSWORD not set}"
: "${IMAGE_NAMESPACE:?IMAGE_NAMESPACE not set}"

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
echo "============================================"

echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin

select_mirror

do_build "${SCRIPT_DIR}/ros-base/Dockerfile" \
         "${SCRIPT_DIR}/ros-base" \
         "${FULL_IMAGE}"

# Tag latest
docker tag "${FULL_IMAGE}" "${LATEST_IMAGE}"

do_push "${FULL_IMAGE}"
do_push "${LATEST_IMAGE}"

echo ""
echo "Done. Images pushed:"
echo "  ${FULL_IMAGE}"
echo "  ${LATEST_IMAGE}"
