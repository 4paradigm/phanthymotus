#!/usr/bin/env bash
set -euo pipefail

readonly TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly CARD_DIR="$(cd -- "${TEST_DIR}/.." && pwd -P)"
readonly READY_SCRIPT="${CARD_DIR}/scripts/shanghai-ready-check.sh"
readonly PERCEPTION_REPO="$(cd -- "${CARD_DIR}/../../.." && pwd -P)"
readonly COMMON_GIT_DIR="$(git -C "${PERCEPTION_REPO}" rev-parse --path-format=absolute --git-common-dir)"
readonly PRIMARY_PERCEPTION_REPO="$(cd -- "${COMMON_GIT_DIR}/.." && pwd -P)"
readonly PHANTHY_MOTUS_ROOT="$(cd -- "${PRIMARY_PERCEPTION_REPO}/.." && pwd -P)"
readonly DRIVER_REPO="${PHANTHY_MOTUS_ROOT}/phanthymotus-driver"
readonly FAKE_COMMAND="${DRIVER_REPO}/unitree/g1/traditional_navigation/tests/fake-navigation-command.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

expect_status() {
  expected="$1"
  shift
  set +e
  "$@" >/dev/null 2>&1
  actual="$?"
  set -e
  test "${actual}" -eq "${expected}" || fail "expected status ${expected}, got ${actual}: $*"
}

bash -n "${READY_SCRIPT}"
test -x "${FAKE_COMMAND}" || fail "missing driver fake command: ${FAKE_COMMAND}"
expect_status 2 "${READY_SCRIPT}" unexpected-argument

readonly TEST_ROOT="$(mktemp -d /private/tmp/controlled-spatial-shells.XXXXXX)"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT
mkdir -p "${TEST_ROOT}/bin"
ln -s "${FAKE_COMMAND}" "${TEST_ROOT}/bin/ssh"
readonly FAKE_PATH="${TEST_ROOT}/bin:${PATH}"
readonly READY_LOG="${TEST_ROOT}/ready.log"

(
  cd /private/tmp
  PATH="${FAKE_PATH}" G1_FAKE_LOG="${READY_LOG}" \
    "${READY_SCRIPT}" >/dev/null
)

grep -Fq 'g1-sh-wifi' "${READY_LOG}"
grep -Fq "ip -br link show 'eth0'" "${READY_LOG}"
if grep -Eq 'mkdir -p|docker (compose|build|load).*(up|down|build|load)' "${READY_LOG}"; then
  fail "Shanghai ready check emitted a robot write command"
fi

echo "controlled_spatial_shell_tests=PASS"
