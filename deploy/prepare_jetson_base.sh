#!/usr/bin/env bash
# prepare_jetson_base.sh — 中转 dustynv/ros 镜像到腾讯云仓库
# 通过 DaoCloud 加速拉取 Docker Hub 镜像，retag 后推送到私有仓库
#
# Usage:
#   ./prepare_jetson_base.sh
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

SOURCE="docker.m.daocloud.io/dustynv/ros:humble-desktop-l4t-r35.3.1"
TARGET="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:humble-desktop-l4t-r35.3.1"

echo "============================================"
echo "Pulling Jetson base from DaoCloud mirror"
echo "Source: ${SOURCE}"
echo "Target: ${TARGET}"
echo "============================================"

docker pull "${SOURCE}"

echo "Tagging → ${TARGET}"
docker tag "${SOURCE}" "${TARGET}"

echo "Pushing to registry..."
echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
docker push "${TARGET}"

echo ""
echo "Done. Jetson base image available at:"
echo "  ${TARGET}"
echo ""
echo "You can now remove the DaoCloud cached image:"
echo "  docker rmi ${SOURCE}"
