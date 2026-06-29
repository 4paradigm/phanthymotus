#!/usr/bin/env bash
# prepare_jp_v511.sh — 构建含 GPU PyTorch 的 Jetson base 镜像并推送到 TCR
#
# 需要在能访问 developer.download.nvidia.com 的环境执行（海外或代理）
# 产出：jetson-base:jp511-torch 镜像，包含 JetPack 5.1.1 + PyTorch GPU
#
# Usage:
#   ./prepare_jp_v511.sh [--mirror tuna|tencent|none]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[error] Registry not configured. This script requires a registry to push images."
    echo "        Copy deploy/.env.example to deploy/.env and fill in values."
    exit 1
fi

BASE_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:humble-desktop-l4t-r35.3.1"
TARGET="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:jp511-torch"

# JetPack 5.1.1 PyTorch wheel (NVIDIA official)
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"

echo "============================================"
echo "Building Jetson PyTorch base image"
echo "Base:   ${BASE_IMAGE}"
echo "Target: ${TARGET}"
echo "Torch:  ${TORCH_URL}"
echo "============================================"

# 生成临时 Dockerfile
TMPFILE="$(mktemp)"
cat > "${TMPFILE}" <<DOCKERFILE
FROM dustynv/l4t-pytorch:r35.3.1 AS pytorch-donor
FROM ${BASE_IMAGE}
RUN rm -f /etc/apt/sources.list.d/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends libopenblas-base libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir ${TORCH_URL}
# Copy pre-compiled torchvision (with CUDA NMS ops) from dustynv image
COPY --from=pytorch-donor /usr/local/lib/python3.8/dist-packages/torchvision /usr/local/lib/python3.8/dist-packages/torchvision
COPY --from=pytorch-donor /usr/local/lib/python3.8/dist-packages/torchvision-*.dist-info /usr/local/lib/python3.8/dist-packages/
DOCKERFILE

echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin

docker build -f "${TMPFILE}" -t "${TARGET}" .
rm -f "${TMPFILE}"

echo "Pushing → ${TARGET}"
docker push "${TARGET}"

echo ""
echo "Done. Image available at:"
echo "  ${TARGET}"
echo ""
echo "Update Dockerfile.jetson BASE_IMAGE to:"
echo "  ARG BASE_IMAGE=${TARGET}"
