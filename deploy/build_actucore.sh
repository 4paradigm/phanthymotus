#!/usr/bin/env bash
# build_actucore.sh — 构建 actucore（执行模型层）镜像并推送
#
# 本入口构建普通 Jetson bundle；CPU 隔离验证使用 build_tianyi_actucore.sh。
#
# Usage:
#   ./build_actucore.sh                          # JetPack 5.11（默认，与 build_perception.sh 一致）
#   ./build_actucore.sh --jp-version 6.1         # JetPack 6.1
#   ./build_actucore.sh --mirror tuna
#
# 默认值与 build_perception.sh 保持一致（5.11）。两个脚本并排放着，不带参数跑
# 却落到不同的 JetPack 线上，是那种要等到部署时才发现的意外。
#
# 两条线用同一份 Dockerfile，只有 base 不同 —— 应用层是逐字节一样的：
#
#   5.11  jetson-base           只有远端 provider，薄镜像（默认）
#   6.1   jetson-base-actucore  本地推理（lerobot + CUDA torch 2.9），~18.6 GB
#
# 这个差别不是取舍，是事实：jp5.11 的 CUDA 是 11.4，而 lerobot 要 torch >= 2.2.1，
# 没有任何 torch >= 2.2 支持 CUDA 11.4（官方矩阵最低 11.8）。所以那条线上
# **不可能**有本地推理，跑远端 provider 才是它的形态。详见
# deploy/prepare_actucore_base.sh 的说明。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/build_common.sh"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

eval "$(parse_mirror_arg "$@")"

# ── 解析参数 ─────────────────────────────────────────────────────────
JP_VERSION="5.11"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jp-version) JP_VERSION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done



RESOURCE_CENTER_URL="${RESOURCE_CENTER_URL:-https://motus.phanthy.com}"

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

# ── 普通 Jetson bundle ───────────────────────────────────────────────
DOCKERFILE="${REPO_ROOT}/actucore/Dockerfile.jetson"
BUILD_CONTEXT="${REPO_ROOT}"
TAG="release.${DATE}.${COMMIT}-jetson-jp${JP_VERSION}"

# 数组，不是字符串。do_build 把每个额外参数当一个独立的 --build-arg，所以把
# 多个 KEY=VALUE 拼进一个字符串传过去会变成一个畸形参数：第二个之后的全部
# 被当成第一个的值，静默不生效。build_perception.sh 至今只传一个参数，所以
# 那个写法在它那里一直没露馅 —— 这里传两个，第一次构建 jp5.11 就拿到了
# jp6.1 的 base（Python 3.10、带 lerobot 的 18.6 GB 镜像），构建本身还成功了。
BUILD_ARGS=()
# ── 根据 jp_version 选择 base image  ────────────────────────
# 表在 build_common.sh 的 jetpack_vars 里，build_perception.sh 共用同一份。
jetpack_vars "${JP_VERSION}" || exit 1
BUILD_ARGS+=("JP_VERSION=${JP_ARG}")

# 同一份 Dockerfile，两个 base。只有 6.1 那条线有本地推理所需的 torch/lerobot。
#
# base 的来源和推送目标是两件事，不能共用 REGISTRY：上面那段在没配凭据时会把
# REGISTRY 设成 "local"（表示"只构建不推送"），而 base 无论如何都要从真实仓库
# 拉 —— TCR 对 phanthy-motus 允许匿名拉取，所以没凭据的机器照样构建得了。
BASE_REGISTRY="${BASE_REGISTRY:-bj-warehouse.tencentcloudcr.com}"
BASE_NAMESPACE="${BASE_NAMESPACE:-phanthy-motus}"
if [ "${JP_VERSION}" = "6.1" ]; then
    BASE_IMAGE="${BASE_REGISTRY}/${BASE_NAMESPACE}/jetson-base-actucore:jp${JP_ARG}-torch"
else
    BASE_IMAGE="${BASE_REGISTRY}/${BASE_NAMESPACE}/jetson-base:jp${JP_ARG}-torch"
    echo ""
    echo "[note] JetPack ${JP_VERSION}：只构建远端 provider 可用的薄镜像。"
    echo "       本机推理（provider: smolvla）在这条线上装不了 —— CUDA 11.4 撑不住"
    echo "       lerobot 要求的 torch >= 2.2.1。卡片会在启动时说明，不会静默失败。"
    echo ""
fi
BUILD_ARGS+=("BASE_IMAGE=${BASE_IMAGE}")

# Dockerfile.jetson 基于 L4T base image —— 只有 arm64
CPU_ARCH="arm64"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/actucore:${TAG}"

echo "============================================"
echo "Building actucore image (Jetson only)"
echo "PyTorch for JetPack: JP${JP_VERSION}"
echo "Image  : ${FULL_IMAGE}"
echo "Arch   : ${ARCH} (native=${IS_ARM64})"
echo "Runs on: ${ACC_ARCH} / ${CPU_ARCH}"
echo "Push   : ${PUSH_ENABLED}"
echo "============================================"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

select_mirror

do_build "${DOCKERFILE}" "${BUILD_CONTEXT}" "${FULL_IMAGE}" "${BUILD_ARGS[@]}"

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
        # cards 目前为空：actucore/plugins/ 还没有任何已注册的卡片（见
        # actucore/main.py 的卡片注册区注释和 actucore/README.md）。第一个卡片落地时
        # 把它加进这个数组，不要漏掉。
        HTTP_STATUS=$(curl -s -o /tmp/rc_register_resp.json -w "%{http_code}" \
            -X POST "${RESOURCE_CENTER_URL}/api/admin/register" \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${RESOURCE_CENTER_API_KEY}" \
            -d "{
                \"imageRef\": \"${FULL_IMAGE}\",
                \"registryImage\": \"actucore\",
                \"tag\": \"${TAG}\",
                \"category\": \"actucore\",
                \"acc_arch\": \"${ACC_ARCH}\",
                \"cpu_arch\": \"${CPU_ARCH}\",
                \"name\": \"ActuCore\",
                \"port\": 15730,
                \"description\": \"执行模型层 — VLA 策略 / 导航 / 抓取 / locomotion / 全身控制，以 processor 卡片接入\",
                \"cards\": []
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
