#!/usr/bin/env bash
# prepare_llm_base.sh — 编译 llama.cpp (CUDA) 并推送为 Jetson base 镜像
#
# 产出：jetson-base:jp511-llama 镜像（含 llama-server 二进制）
#
# Usage:
#   ./prepare_llm_base.sh
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

BASE_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:jp511-torch"
TARGET="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:jp511-llama"
LLAMA_CPP_URL="https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/llama.cpp-master.zip"

echo "============================================"
echo "Building Jetson llama.cpp base image"
echo "Base:    ${BASE_IMAGE}"
echo "Target:  ${TARGET}"
echo "Source:  ${LLAMA_CPP_URL}"
echo "============================================"

# 检查本地是否已有该镜像
if docker image inspect "${TARGET}" >/dev/null 2>&1; then
    echo ""
    echo "Image already exists locally: ${TARGET}"
    printf "Skip build and go straight to push? [Y/n]: "
    read -r SKIP </dev/tty || SKIP="y"
    if [[ ! "${SKIP}" =~ ^[Nn] ]]; then
        echo "Skipping build, proceeding to push..."
        echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
        echo "Pushing → ${TARGET}"
        docker push "${TARGET}"
        echo ""
        echo "Done. Image available at:"
        echo "  ${TARGET}"
        exit 0
    fi
    echo "Rebuilding..."
fi

# 生成临时 Dockerfile
TMPFILE="$(mktemp)"
cat > "${TMPFILE}" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN rm -f /etc/apt/sources.list.d/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake build-essential libcurl4-openssl-dev unzip curl && \
    rm -rf /var/lib/apt/lists/*

ARG LLAMA_CPP_URL
RUN curl -fSL -o /tmp/llama.cpp.zip ${LLAMA_CPP_URL} && \
    unzip -q /tmp/llama.cpp.zip -d /tmp && \
    mv /tmp/llama.cpp-master /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="72;87" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined -L/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--allow-shlib-undefined -L/usr/local/cuda/lib64/stubs" && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/llama-server /usr/local/bin/ && \
    cp build/bin/llama-cli /usr/local/bin/ && \
    cp build/bin/*.so /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/llama.cpp /tmp/llama.cpp.zip

# 清理编译工具（保留 build-essential，下游 colcon build 需要）
RUN apt-get purge -y unzip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
DOCKERFILE

echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin

docker build -f "${TMPFILE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "LLAMA_CPP_URL=${LLAMA_CPP_URL}" \
    -t "${TARGET}" \
    .

rm -f "${TMPFILE}"

echo "Pushing → ${TARGET}"
docker push "${TARGET}"

echo ""
echo "Done. Image available at:"
echo "  ${TARGET}"
echo ""
echo "Next steps:"
echo "  1. Build perception image: ./build_perception.sh --variant jetson"
echo "  2. Place model in deploy/core/models/ (auto-downloaded from COS on first start)"
echo "  3. Start: docker-compose --profile perception up -d"
