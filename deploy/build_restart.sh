#!/usr/bin/env bash
# build_restart.sh — 构建 restart helper 镜像并推送
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

# If registry not configured, build locally only
PUSH_ENABLED=true
if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[info] Registry not configured — building locally only (no push)."
    PUSH_ENABLED=false
    REGISTRY="${REGISTRY:-local}"
    IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-phanthy-motus}"
fi

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/restart:latest"

echo "Building restart helper: ${FULL_IMAGE}"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

docker buildx build \
    --platform linux/arm64 \
    --file "${SCRIPT_DIR}/restart/Dockerfile" \
    --tag "${FULL_IMAGE}" \
    $(${PUSH_ENABLED} && echo "--push" || echo "--load") \
    "${SCRIPT_DIR}/restart"

echo "Done: ${FULL_IMAGE}"
