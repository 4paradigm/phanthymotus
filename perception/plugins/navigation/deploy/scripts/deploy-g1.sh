#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
RUNTIME_SCRIPT="${SCRIPT_DIR}/owner-start-g1-test-containers.sh"
PERCEPTION_CONTAINER="${PERCEPTION_CONTAINER:-embodied-perception-test}"

cd "${REPO_ROOT}"

[[ "$(uname -m)" == "aarch64" ]] || {
  printf 'ERROR=this script must run on the G1 aarch64 host\n' >&2
  exit 1
}

BUILD_DATE="$(date +%y%m%d)"
COMMIT="$(git rev-parse --short=7 HEAD)"
export PERCEPTION_IMAGE="local/phanthy-motus/perception:release.${BUILD_DATE}.${COMMIT}-navigation"

./deploy/build_perception.sh --mirror tuna

if docker container inspect "${PERCEPTION_CONTAINER}" >/dev/null 2>&1; then
  STAGE=stop bash "${RUNTIME_SCRIPT}"
fi

STAGE=preflight bash "${RUNTIME_SCRIPT}"
STAGE=start bash "${RUNTIME_SCRIPT}"
