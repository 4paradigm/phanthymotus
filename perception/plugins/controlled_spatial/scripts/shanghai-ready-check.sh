#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly CARD_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly PERCEPTION_REPO="$(cd -- "${CARD_DIR}/../../.." && pwd -P)"

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  echo "Runs the fixed read-only preflight for Shanghai G1: g1-sh-wifi / ubuntu / eth0" >&2
  exit 2
fi

readonly COMMON_GIT_DIR="$(git -C "${PERCEPTION_REPO}" rev-parse --path-format=absolute --git-common-dir)"
readonly PRIMARY_PERCEPTION_REPO="$(cd -- "${COMMON_GIT_DIR}/.." && pwd -P)"
readonly PHANTHY_MOTUS_ROOT="$(cd -- "${PRIMARY_PERCEPTION_REPO}/.." && pwd -P)"
readonly DRIVER_REPO="${PHANTHYMOTUS_DRIVER_REPO:-${PHANTHY_MOTUS_ROOT}/phanthymotus-driver}"
readonly ACCEPT_SCRIPT="${DRIVER_REPO}/unitree/g1/traditional_navigation/accept-g1-navigation-shadow.sh"

if [[ ! -x "${ACCEPT_SCRIPT}" ]]; then
  echo "missing executable driver acceptance helper: ${ACCEPT_SCRIPT}" >&2
  echo "Set PHANTHYMOTUS_DRIVER_REPO to the phanthymotus-driver checkout if it is not a sibling repo." >&2
  exit 1
fi

exec "${ACCEPT_SCRIPT}" preflight g1-sh-wifi ubuntu eth0
