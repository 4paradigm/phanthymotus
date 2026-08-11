#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

cd "${REPO_ROOT}"

[[ "$(uname -m)" == "aarch64" ]] || {
  printf 'ERROR=this script must run on the G1 aarch64 host\n' >&2
  exit 1
}

BUILD_DATE="$(date +%y%m%d)"
COMMIT="$(git rev-parse --short=7 HEAD)"

export PERCEPTION_IMAGE="local/phanthy-motus/perception:release.${BUILD_DATE}.${COMMIT}-jetson"
export FAST_LIVO2_IMAGE="phanthy-fast-livo2:nav2-card-${COMMIT}"
export NAV2_IMAGE="phanthy-nav2:nav2-card-${COMMIT}"
export ROS_BASE_IMAGE="${ROS_BASE_IMAGE:-bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base@sha256:82d45949e7c3fd85e6baf4a2b24b384a3ec020a5e237c5f801bc2f2269ca649f}"

if [[ ! -d /opt/phanthy-motus/data/fast_livo2/maps ]]; then
  sudo install -d -m 0775 -o "$(id -u)" -g "$(id -g)" \
    /opt/phanthy-motus/data/fast_livo2/maps
fi

./deploy/build_perception.sh --variant jetson --mirror tuna

DOCKER_BUILDKIT=0 \
FAST_LIVO2_IMAGE="${FAST_LIVO2_IMAGE}" \
  bash perception/plugins/fast_livo2/companion/build-companion.sh

(
  cd perception/plugins/nav2/companion
  DOCKER_BUILDKIT=0 \
  NAV2_IMAGE="${NAV2_IMAGE}" \
  ROS_BASE_IMAGE="${ROS_BASE_IMAGE}" \
    docker compose --env-file source-lock.env build nav2
)

STAGE=preflight \
  bash perception/plugins/nav2/deploy/scripts/owner-start-g1-test-containers.sh
STAGE=start \
  bash perception/plugins/nav2/deploy/scripts/owner-start-g1-test-containers.sh
