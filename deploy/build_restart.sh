#!/usr/bin/env bash
# build_restart.sh — 构建 restart helper 镜像并推送到腾讯云
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

: "${REGISTRY:?REGISTRY not set. Copy deploy/.env.example to deploy/.env and fill in values.}"
: "${REGISTRY_USER:?REGISTRY_USER not set}"
: "${REGISTRY_PASSWORD:?REGISTRY_PASSWORD not set}"
: "${IMAGE_NAMESPACE:?IMAGE_NAMESPACE not set}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/restart:latest"

echo "Building restart helper: ${FULL_IMAGE}"

echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin

docker buildx build \
    --platform linux/arm64 \
    --file "${SCRIPT_DIR}/restart/Dockerfile" \
    --tag "${FULL_IMAGE}" \
    --push \
    "${SCRIPT_DIR}/restart"

echo "Done: ${FULL_IMAGE}"
