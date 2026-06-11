#!/usr/bin/env bash
# build_common.sh — 构建脚本公共函数（平台检测 + 镜像源选择 + docker build 封装）
#
# 使用方式：在各 build 脚本中 source 本文件
#   source "${SCRIPT_DIR}/build_common.sh"

# ── 平台检测 ──────────────────────────────────────────────────────────
ARCH="$(uname -m)"
IS_ARM64=false
[[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]] && IS_ARM64=true

# ── 镜像源选择 ────────────────────────────────────────────────────────
# 通过 --mirror 参数或 MIRROR 环境变量或交互式选择
select_mirror() {
    local mirror="${MIRROR:-}"

    if [ -z "${mirror}" ]; then
        echo ""
        echo "Select mirror / 选择镜像源:"
        echo "  1) tencent  — 腾讯云（VPC 内网）"
        echo "  2) tuna     — 清华 TUNA（公网）"
        echo "  3) none     — 官方源（海外 / 裸连）"
        printf "Choice [1/2/3] (default: 2): "
        read -r choice </dev/tty || choice=""
        case "${choice}" in
            1) mirror="tencent" ;;
            3) mirror="none" ;;
            *) mirror="tuna" ;;
        esac
    fi

    case "${mirror}" in
        tencent)
            PYPI_MIRROR="https://mirrors.tencentyun.com/pypi/simple/"
            APT_MIRROR="mirrors.tencentyun.com"
            ROS_MIRROR="http://packages.ros.org/ros2/ubuntu"
            BINFMT_IMAGE="mirror.ccs.tencentyun.com/tonistiigi/binfmt"
            ;;
        tuna)
            PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple/"
            APT_MIRROR="mirrors.tuna.tsinghua.edu.cn"
            ROS_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu"
            BINFMT_IMAGE="docker.io/tonistiigi/binfmt"
            ;;
        none|*)
            PYPI_MIRROR="https://pypi.org/simple/"
            APT_MIRROR=""
            ROS_MIRROR="http://packages.ros.org/ros2/ubuntu"
            BINFMT_IMAGE="docker.io/tonistiigi/binfmt"
            ;;
    esac

    export MIRROR="${mirror}" PYPI_MIRROR APT_MIRROR ROS_MIRROR BINFMT_IMAGE
    echo "Mirror: ${mirror} | PyPI: ${PYPI_MIRROR}"
    echo ""
}

# ── Docker build 封装 ─────────────────────────────────────────────────
# do_build <dockerfile> <context> <full_image> [extra_build_arg...]
#
# ARM64 原生: docker build（无 buildx，无 binfmt）
# x86_64:     docker buildx build --platform linux/arm64（交叉编译）
#
# 使用 PUSH_ENABLED 环境变量控制是否推送（默认 true）
do_build() {
    local dockerfile="$1"; shift
    local context="$1"; shift
    local full_image="$1"; shift

    local build_args=(
        --build-arg "PYPI_MIRROR=${PYPI_MIRROR}"
        --build-arg "APT_MIRROR=${APT_MIRROR:-}"
        --build-arg "ROS_MIRROR=${ROS_MIRROR}"
    )
    while [[ $# -gt 0 ]]; do
        build_args+=(--build-arg "$1"); shift
    done

    if ${IS_ARM64}; then
        echo "[native ARM64] docker build"
        docker build \
            --file "${dockerfile}" \
            "${build_args[@]}" \
            --tag "${full_image}" \
            "${context}"
    else
        echo "[cross-compile x86→ARM64] docker buildx build"
        docker run --privileged --rm "${BINFMT_IMAGE}" --install arm64
        local output_flag="--push"
        if [ "${PUSH_ENABLED:-true}" = "false" ]; then
            output_flag="--output=type=docker"
        fi
        docker buildx build \
            --builder default \
            --platform linux/arm64 \
            --file "${dockerfile}" \
            "${build_args[@]}" \
            --tag "${full_image}" \
            ${output_flag} \
            "${context}"
    fi
}

# ── Push（原生构建后手动 push；buildx --push 已含） ────────────────────
do_push() {
    local full_image="$1"
    if ${IS_ARM64}; then
        docker push "${full_image}"
    fi
}

# ── 解析 --mirror 参数（从 $@ 中提取并移除） ──────────────────────────
# 用法: eval "$(parse_mirror_arg "$@")"
# 设置 MIRROR 变量并输出剩余参数赋值
parse_mirror_arg() {
    local remaining=()
    local _mirror=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mirror) _mirror="$2"; shift 2 ;;
            *) remaining+=("$1"); shift ;;
        esac
    done
    # 输出赋值语句供 eval 执行
    if [ -n "${_mirror}" ]; then
        printf 'MIRROR=%q; ' "${_mirror}"
    fi
    if [ ${#remaining[@]} -gt 0 ]; then
        printf 'set -- '
        printf '%q ' "${remaining[@]}"
    else
        printf 'set --'
    fi
    echo ""
}
