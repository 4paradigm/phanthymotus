#!/usr/bin/env bash
# prepare_actucore_base.sh — build jetson-base-actucore and push it to TCR
#
# ActuCore's execution models need a newer torch than the shared jetson-base
# carries, so they get their own base image rather than moving the floor under
# perception and the fourteen drivers.
#
# Why a whole image and not a RUN layer in actucore's Dockerfile: the six fixes
# below are all about *replacing* things the shared base installed, and doing
# that per-application build would repeat a ~4 GB download on every build and
# leave the removals interleaved with application layers where they are easy to
# reorder by accident.
#
# What this fixes, all measured on Orin 6 (JetPack 6.1, L4T R36.4.3):
#
#  1. torchvision 0.19 in the shared base is below lerobot's >=0.21 floor, and
#     pinning it makes pip give up with ResolutionImpossible. Upgrading torch is
#     the only way out.
#  2. NVIDIA's torch for JP6.1 is built USE_DISTRIBUTED=0 — `torch.distributed`
#     has no `device_mesh`, which modern diffusers and accelerate dereference at
#     import. jetson-ai-lab's 2.9.1 has it (verified by the presence of the
#     `_c10d_init` symbol, against NVIDIA's build as a known negative).
#  3. torch 2.9 needs libcudss, absent from the shared base.
#  4. The apt scipy is built against numpy 1 and breaks once anything pulls
#     numpy 2; same story for the apt Pillow, which predates
#     `PIL.Image.Resampling`. Purged so the pip versions are the only ones.
#  5. NVIDIA's opencv-contrib-python is numpy-1 only, and it shares the `cv2/`
#     package directory with the PyPI wheel — `pip uninstall` leaves the .so
#     behind, so the directory has to be removed by hand. See the CUDA note at
#     the bottom.
#  6. The shared base's pip is configured to reach pypi.jetson-ai-lab.io, which
#     is unreachable from inside a container here, so every pip call passes an
#     explicit index.
#
# The wheels are served from COS rather than jetson-ai-lab: that index runs at
# ~12 KB/s from both the rigs and the VPC build host, which turns a 218 MB
# download into hours of a build that merely looks hung.
#
# Usage:
#   ./prepare_actucore_base.sh [--jp-version 6.1] [--no-push]
#
# Build it on an arm64 Jetson (native, ~15 min). The VPC host cross-compiles
# through qemu and takes far longer for an image this size.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JP_VERSION="6.1"
PUSH=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jp-version) JP_VERSION="$2"; shift 2 ;;
        --no-push)    PUSH=0; shift ;;
        *) echo "[error] unknown argument: $1"; exit 1 ;;
    esac
done

# JetPack 5.11 has no version of this image, and the reason is worth stating in
# full because the obvious summary ("just upgrade Python") is wrong and would
# send somebody down a long road to a wall. Three doors, measured on Orin 5:
#
#  1. **Python.** That line is Ubuntu 20.04, system Python 3.8.10, and lerobot
#     requires >= 3.10. ROS2 Humble is built from source into
#     /opt/ros/humble/install/lib/python3.8, so moving Python means rebuilding
#     ROS2 as well — and every compiled wheel in the image with it.
#
#  2. **No CUDA torch for cp310.** NVIDIA publishes JetPack 5 wheels only as
#     cp38 (checked v51, v511, v512), and pypi.jetson-ai-lab.io has no jp5 index
#     at all — its root lists jp6/{cu126,cu128,cu129} and nothing else. So the
#     wheel would have to be built from source.
#
#  3. **And a source build still cannot work.** JetPack 5.11 is CUDA 11.4.
#     lerobot needs torch >= 2.2.1, and PyTorch's own support matrix floors
#     torch 2.2 at CUDA 11.8. No amount of building produces a torch that is
#     both new enough for lerobot and able to use this CUDA.
#
# Door 3 is closed, so doors 1 and 2 do not matter. The jp5.11 line runs actucore
# with remote providers only, on the plain jetson-base — see build_actucore.sh,
# which builds exactly that from the same Dockerfile.
if [ "${JP_VERSION}" != "6.1" ]; then
    echo "[error] JetPack ${JP_VERSION} 没有这个 base，也不可能有。"
    echo "        jp5.11 是 CUDA 11.4，而 lerobot 要 torch >= 2.2.1，"
    echo "        没有任何 torch >= 2.2 支持 CUDA 11.4（官方矩阵最低 11.8）。"
    echo "        那条线用 ./build_actucore.sh --jp-version 5.11 构建薄镜像，"
    echo "        只跑远端 provider。"
    exit 1
fi

JP_TAG="61"
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

REGISTRY="${REGISTRY:-bj-warehouse.tencentcloudcr.com}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-phanthy-motus}"
BASE_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:jp${JP_TAG}-torch"
TARGET="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base-actucore:jp${JP_TAG}-torch"

PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple/}"
COS_BASE="${COS_BASE:-https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/jetson/jp61}"
TORCH_WHL="torch-2.9.1-cp310-cp310-linux_aarch64.whl"
VISION_WHL="torchvision-0.24.1-cp310-cp310-linux_aarch64.whl"

echo "============================================"
echo "Building ActuCore base image"
echo "Base:   ${BASE_IMAGE}"
echo "Target: ${TARGET}"
echo "Wheels: ${COS_BASE}"
echo "============================================"

TMPFILE="$(mktemp)"
cat > "${TMPFILE}" <<DOCKERFILE
FROM ${BASE_IMAGE}
ARG PYPI_MIRROR=${PYPI_MIRROR}
ARG COS_BASE=${COS_BASE}

# Debian-packaged copies of libraries the Python stack is about to replace.
# Both are built against numpy 1 and win on sys.path, so leaving them means an
# ImportError deep inside transformers that names neither of them.
RUN apt-get remove -y --purge python3-scipy python3-pil python3-pil.imagetk || true

# libcudss, which torch 2.9 links against and the shared base does not carry.
RUN pip3 install --no-cache-dir -i \${PYPI_MIRROR} nvidia-cudss-cu12 && \\
    echo /usr/local/lib/python3.10/dist-packages/nvidia/cu12/lib \\
        > /etc/ld.so.conf.d/nvidia-cu12.conf && \\
    ldconfig

# CUDA torch/torchvision with distributed support. From COS: the upstream index
# serves this machine at ~12 KB/s.
RUN pip3 install --no-cache-dir -i \${PYPI_MIRROR} \\
        \${COS_BASE}/${TORCH_WHL} \\
        \${COS_BASE}/${VISION_WHL}

# NVIDIA's opencv is numpy-1 only. pip uninstall leaves its .so in the shared
# cv2/ package directory, so the directory goes too — otherwise the old binary
# shadows whatever is installed next and fails with
# "numpy.core.multiarray failed to import".
RUN pip3 uninstall -y opencv-contrib-python opencv-python opencv-python-headless || true && \\
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2

RUN pip3 install --no-cache-dir -i \${PYPI_MIRROR} \\
        'lerobot[smolvla]' scipy Pillow opencv-python-headless

# Fail the build here rather than on a robot. Each assertion is a bug that was
# actually hit while working this out.
#
# One line, not a heredoc. Docker hands a RUN to the shell as a single command,
# so \`python3 - <<'EOF'\` has no following lines to read: bash warns about a
# here-document delimited by end-of-file, python reads empty stdin, and the step
# exits 0 having verified nothing. The first build of this image did exactly
# that and "passed" — precisely what this check exists to prevent.
RUN python3 -c "import torch, torchvision, numpy, scipy, transformers, PIL, cv2; \\
import torch.distributed as dist; \\
assert torch.__version__.startswith('2.9'), torch.__version__; \\
assert torchvision.__version__.startswith('0.24'), torchvision.__version__; \\
assert dist.is_available(), 'USE_DISTRIBUTED=0 — diffusers and accelerate will not import'; \\
assert hasattr(dist, 'device_mesh'), 'no device_mesh'; \\
exec('from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy'); \\
print('actucore base OK:', 'torch', torch.__version__, '| tv', torchvision.__version__, \\
      '| np', numpy.__version__, '| scipy', scipy.__version__, \\
      '| tf', transformers.__version__, '| PIL', PIL.__version__, '| cv2', cv2.__version__)"
DOCKERFILE

echo "[build] this takes ~15 min natively on a Jetson"
docker build -f "${TMPFILE}" -t "${TARGET}" "${SCRIPT_DIR}/.."
rm -f "${TMPFILE}"

if [ "${PUSH}" = "1" ]; then
    if [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ]; then
        echo "[warn] no registry credentials in deploy/.env — built locally, not pushed"
        echo "       image: ${TARGET}"
        exit 0
    fi
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
    docker push "${TARGET}"
    echo "Pushed → ${TARGET}"
fi

cat <<'NOTE'

Done.

CUDA OpenCV is NOT in this image, and that is a known limitation rather than an
oversight. Three candidates were tried on Orin 6 and none works here:

  - NVIDIA's opencv-contrib-python 4.10  — CUDA, correct for Jetson, numpy 1 only
  - PyPI opencv                          — numpy 2, no CUDA          (what is installed)
  - jetson-ai-lab's CUDA contrib 4.12    — built for discrete GPUs; wants
                                           libnvcuvid.so.1 and libnvidia-encode.so.1,
                                           neither of which exists on this Jetson

A card that genuinely needs GPU OpenCV will have to build OpenCV for Jetson
against numpy 2. Do not paper over it by symlinking the Jetson V4L2 shims to
those names: the import then succeeds and the call into a mismatched API fails
later, somewhere else.
NOTE
