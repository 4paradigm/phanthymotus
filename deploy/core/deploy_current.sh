#!/usr/bin/env bash
# deploy_current.sh — 构建当前代码、推送到 TCR、拉取并重启本地 agent-core
# 部署目录与 install.sh 一致：/opt/phanthy-motus
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/.." && pwd)"
INSTALL_DIR="/opt/phanthy-motus"

# 从上级 deploy/.env 读取 REGISTRY/IMAGE_NAMESPACE（build_core.sh 也读这个）
if [ -f "${DEPLOY_DIR}/.env" ]; then
    source "${DEPLOY_DIR}/.env"
fi

: "${REGISTRY:?REGISTRY not set. Ensure deploy/.env exists.}"
: "${IMAGE_NAMESPACE:?IMAGE_NAMESPACE not set.}"

# 1. 构建并推送（复用现有 build_core.sh）
echo "=== Building & pushing agent-core ==="
bash "${DEPLOY_DIR}/build_core.sh"

# 2. 计算 TAG（跟 build_core.sh 相同逻辑）
DATE="$(date +%y%m%d)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"
TAG="release.${DATE}.${COMMIT}"
CORE_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/core:${TAG}"

# 3. 创建部署目录及持久化文件
mkdir -p "${INSTALL_DIR}/data/world"
touch "${INSTALL_DIR}/data/data.db"
[ -f "${INSTALL_DIR}/data/prompt_memory.md" ] || echo "" > "${INSTALL_DIR}/data/prompt_memory.md"

# 4. 写入 docker-compose.yml（与 install.sh 保持一致）
cat > "${INSTALL_DIR}/docker-compose.yml" <<COMPOSE
services:
  agent-core:
    image: ${CORE_IMAGE}
    platform: linux/arm64
    network_mode: host
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./data/data.db:/work/resource/data.db
      - ./data/prompt_memory.md:/work/resource/memory/prompt_memory.md
      - ./data/world:/work/resource/world
    environment:
      - DB_PATH=/work/resource/data.db
      - ROS_DOMAIN_ID=42
      - RMW_IMPLEMENTATION=rmw_fastrtps_cpp
      - FASTDDS_BUILTIN_TRANSPORTS=UDPv4
    restart: unless-stopped
COMPOSE

cat > "${INSTALL_DIR}/.env" <<ENV
# 由 deploy_current.sh 生成 — $(date '+%Y-%m-%d %H:%M:%S')
CORE_IMAGE=${CORE_IMAGE}
DB_PATH=/work/resource/data.db
ENV

# 5. 拉取新镜像
echo ""
echo "=== Pulling agent-core (${TAG}) ==="
docker pull --platform linux/arm64 "${CORE_IMAGE}"

# 6. 停止旧容器并启动
cd "${INSTALL_DIR}"
docker compose down --remove-orphans 2>/dev/null || true
docker rm -f phanthy-motus-agent-core-1 2>/dev/null || true
docker compose up -d

echo ""
echo "Agent Core running at http://localhost:15678"
echo "安装目录: ${INSTALL_DIR}"
echo "Logs: docker compose -f ${INSTALL_DIR}/docker-compose.yml logs -f"
