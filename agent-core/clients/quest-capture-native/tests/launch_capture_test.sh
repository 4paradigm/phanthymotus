#!/bin/sh
set -eu

if [ "${MOTUS_FAKE_ADB:-}" = 1 ]; then
  : "${MOTUS_FAKE_ADB_CAPTURE:?}"
  umask 077
  : > "$MOTUS_FAKE_ADB_CAPTURE"
  for argument in "$@"; do
    printf '%s\n' "$argument" >> "$MOTUS_FAKE_ADB_CAPTURE"
  done
  exit 0
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
launch_script="$project_dir/scripts/launch_capture.sh"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/motus-launch-capture.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM

fail() {
  echo "launch_capture_test: $*" >&2
  exit 1
}

assert_line() {
  capture=$1
  line_number=$2
  expected=$3
  actual=$(sed -n "${line_number}p" "$capture")
  [ "$actual" = "$expected" ] || \
    fail "line $line_number: expected '$expected', got '$actual'"
}

run_fake() {
  capture=$1
  shift
  MOTUS_FAKE_ADB=1 \
  MOTUS_FAKE_ADB_CAPTURE="$capture" \
  ADB="$0" \
  "$launch_script" "$@"
}

meta_resume="$test_dir/meta-resume.args"
ADB_SERIAL='quest-serial' run_fake "$meta_resume" --platform meta --resume
assert_line "$meta_resume" 1 -s
assert_line "$meta_resume" 2 quest-serial
assert_line "$meta_resume" 3 shell
assert_line "$meta_resume" 6 -S
assert_line "$meta_resume" 7 -n
assert_line "$meta_resume" 8 \
  com.phanthymotus.questcapture/android.app.NativeActivity

pico_resume="$test_dir/pico-resume.args"
ADB_SERIAL= run_fake "$pico_resume" --resume --platform=pico
assert_line "$pico_resume" 1 shell
assert_line "$pico_resume" 4 -S
assert_line "$pico_resume" 5 -n
assert_line "$pico_resume" 6 \
  com.phanthymotus.picocapture/android.app.NativeActivity

pico_pair="$test_dir/pico-pair.args"
pico_output="$test_dir/pico-pair.output"
MOTUS_FAKE_ADB=1 \
MOTUS_FAKE_ADB_CAPTURE="$pico_pair" \
ADB="$0" \
ADB_SERIAL='pico-serial' \
CORE_WSS_URL='wss://core.example/ws/teleop-capture' \
PAIRING_ID='pairing-id' \
PAIRING_CODE='one-time-secret' \
CA_CERT_BASE64='VEVTVA==' \
  "$launch_script" --platform pico > "$pico_output" 2>&1
[ ! -s "$pico_output" ] || fail "launch output must not disclose bootstrap values"
assert_line "$pico_pair" 1 -s
assert_line "$pico_pair" 2 pico-serial
assert_line "$pico_pair" 7 -n
assert_line "$pico_pair" 8 \
  com.phanthymotus.picocapture/android.app.NativeActivity
grep -Fx -- one-time-secret "$pico_pair" >/dev/null || \
  fail "pairing code was not passed as an Activity extra"
grep -Fx -- wss://core.example/ws/teleop-capture "$pico_pair" >/dev/null || \
  fail "Core URL was not passed as an Activity extra"

meta_pair="$test_dir/meta-pair.args"
MOTUS_FAKE_ADB=1 \
MOTUS_FAKE_ADB_CAPTURE="$meta_pair" \
ADB="$0" \
CORE_WSS_URL='wss://core.example/ws/teleop-capture' \
PAIRING_ID='pairing-id' \
PAIRING_CODE='meta-secret' \
CA_CERT_BASE64='VEVTVA==' \
  "$launch_script" --platform=meta
assert_line "$meta_pair" 6 \
  com.phanthymotus.questcapture/android.app.NativeActivity

for invalid in \
  '' \
  '--platform unknown' \
  '--platform meta extra' \
  '--platform meta --platform pico'
do
  set +e
  # Intentional word splitting exercises the CLI parser with each invalid form.
  ADB="$0" "$launch_script" $invalid > "$test_dir/invalid.output" 2>&1
  status=$?
  set -e
  [ "$status" -eq 2 ] || fail "invalid arguments '$invalid' returned $status"
done

echo "launch_capture_test: Meta/PICO CLI contract passed"
