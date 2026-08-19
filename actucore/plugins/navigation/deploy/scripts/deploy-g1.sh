#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
RUNTIME_SCRIPT="${SCRIPT_DIR}/owner-start-g1-test-containers.sh"
ACTUCORE_CONTAINER="${ACTUCORE_CONTAINER:-embodied-actucore-test}"

cd "${REPO_ROOT}"

[[ "$(uname -m)" == "aarch64" ]] || {
  printf 'ERROR=this script must run on the G1 aarch64 host\n' >&2
  exit 1
}

BUILD_DATE="$(date +%y%m%d)"
COMMIT="$(git rev-parse --short=7 HEAD)"
JP_VERSION="${JP_VERSION:-5.11}"
export ACTUCORE_IMAGE="local/phanthy-motus/actucore:release.${BUILD_DATE}.${COMMIT}-jetson-jp${JP_VERSION}"

./deploy/build_actucore.sh --mirror tuna --jp-version "${JP_VERSION}"

if docker container inspect "${ACTUCORE_CONTAINER}" >/dev/null 2>&1; then
  STAGE=stop bash "${RUNTIME_SCRIPT}"
fi

STAGE=preflight bash "${RUNTIME_SCRIPT}"
STAGE=start bash "${RUNTIME_SCRIPT}"
