#!/usr/bin/env bash
# build_core.sh — 构建 agent-core（大脑层）镜像并推送
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/build_common.sh"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

eval "$(parse_mirror_arg "$@")"

RESOURCE_CENTER_URL="${RESOURCE_CENTER_URL:-https://motus.phanthy.com}"

DATE="$(date +%y%m%d)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"
TAG="release.${DATE}.${COMMIT}"

# If registry not configured, build locally only
PUSH_ENABLED=true
if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[info] Registry not configured — building locally only (no push)."
    PUSH_ENABLED=false
    REGISTRY="${REGISTRY:-local}"
    IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-phanthy-motus}"
fi

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/core:${TAG}"

echo "============================================"
echo "Building agent-core image"
echo "Image : ${FULL_IMAGE}"
echo "Arch  : ${ARCH} (native=${IS_ARM64})"
echo "Push  : ${PUSH_ENABLED}"
echo "============================================"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

select_mirror

do_build "${REPO_ROOT}/agent-core/Dockerfile" \
         "${REPO_ROOT}" \
         "${FULL_IMAGE}" \
         "IMAGE_TAG=${TAG}"

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
        # acc_arch=agnostic 是硬要求：core 不依赖 CUDA，且必须在所有主机可见 —— 一旦被架构
        # 过滤掉，agent-core 的 api/system.py:_check_update_sync 会默默报「已是最新」。
        #
        # cards 与 src/start.py:_register_core_mcp() 里注册的两个内建 internal MCP 设备
        # （AgentCore 的 decision_core，Channel 的 channel_request/channel_reply）手动保持
        # 一致 —— 这些卡片不是走真实 MCP HTTP 进程暴露的，是 agent-core 启动时自己注册的，
        # 新增/改名时记得同步这里。decision_core 的 type 是 controller（既不是可以绕过
        # barrier 的 sensor/resource，也不是普通 actuator/processor，见
        # src/event/llm.py 的 _needs_barrier）。
        HTTP_STATUS=$(curl -s -o /tmp/rc_register_resp.json -w "%{http_code}" \
            -X POST "${RESOURCE_CENTER_URL}/api/admin/register" \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${RESOURCE_CENTER_API_KEY}" \
            -d "{
                \"imageRef\": \"${FULL_IMAGE}\",
                \"registryImage\": \"core\",
                \"tag\": \"${TAG}\",
                \"category\": \"core\",
                \"acc_arch\": \"agnostic\",
                \"cpu_arch\": \"arm64\",
                \"name\": \"Agent Core\",
                \"cards\": [
                    {\"name\": \"decision_core\", \"type\": \"controller\"},
                    {\"name\": \"remote_mic\", \"type\": \"sensor\"},
                    {\"name\": \"remote_message\", \"type\": \"sensor\"},
                    {\"name\": \"remote_audio\", \"type\": \"sensor\"},
                    {\"name\": \"remote_image\", \"type\": \"sensor\"},
                    {\"name\": \"channel_request\", \"type\": \"sensor\"},
                    {\"name\": \"channel_reply\", \"type\": \"actuator\"}
                ]
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
