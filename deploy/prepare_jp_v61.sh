#!/usr/bin/env bash
# prepare_jp_v61.sh — 构建含 GPU PyTorch 的 Jetson base 镜像并推送到 TCR
#
# 需要在能访问 developer.download.nvidia.com 的环境执行（海外或代理）
# 产出：jetson-base:jp61-torch 镜像，包含 JetPack 6.1 + PyTorch GPU
#
# Usage:
#   ./prepare_jp_v61.sh [--mirror tuna|tencent|none]
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

BASE_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:humble-desktop-l4t-r36.4.0"
TARGET="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:jp61-torch"

# JetPack 6.1 PyTorch wheel (NVIDIA official)
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

echo "============================================"
echo "Building Jetson PyTorch base image"
echo "Base:   ${BASE_IMAGE}"
echo "Target: ${TARGET}"
echo "Torch:  ${TORCH_URL}"
echo "============================================"

# 生成临时 Dockerfile
TMPFILE="$(mktemp)"
cat > "${TMPFILE}" <<DOCKERFILE
FROM dustynv/l4t-pytorch:r36.4.0 AS pytorch-donor
FROM ${BASE_IMAGE}
RUN rm -f /etc/apt/sources.list.d/* && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated libopenblas-base && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN apt-get remove -y --purge python3-sympy && pip3 install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/ --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/ ${TORCH_URL}
RUN wget -q -O /tmp/libcusparse_lt.tar.xz 'https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz' && \
    tar xJf /tmp/libcusparse_lt.tar.xz -C /tmp/ && \
    cp -r /tmp/libcusparse_lt-*/lib/libcusparseLt.so* /usr/local/lib/ && ldconfig && \
    rm -rf /tmp/libcusparse_lt.tar.xz /tmp/libcusparse_lt-*
# Copy pre-compiled torchvision (with CUDA NMS ops) from dustynv image
COPY --from=pytorch-donor /usr/local/lib/python3.10/dist-packages/torchvision /usr/local/lib/python3.10/dist-packages/torchvision
COPY --from=pytorch-donor /usr/local/lib/python3.10/dist-packages/torchvision-0.19.0a0+48b1edf.dist-info/ /usr/local/lib/python3.10/dist-packages/torchvision-0.19.0a0+48b1edf.dist-info
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
