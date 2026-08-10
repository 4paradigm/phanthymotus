#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
build_dir=$(mktemp -d /tmp/motus-quest-capture-host.XXXXXX)
trap 'rm -r "$build_dir"' EXIT HUP INT TERM

"$project_dir/tests/launch_capture_test.sh"

for headset in meta pico; do
  case "$headset" in
    meta)
      headset_definition=-DMOTUS_CAPTURE_HEADSET_META=1
      test_definition=-DTEST_EXPECT_META=1
      ;;
    pico)
      headset_definition=-DMOTUS_CAPTURE_HEADSET_PICO=1
      test_definition=-DTEST_EXPECT_PICO=1
      ;;
  esac
  "${CXX:-clang++}" \
    -std=c++20 \
    -Wall \
    -Wextra \
    -Werror \
    -pedantic \
    "$headset_definition" \
    "$test_definition" \
    -I"$project_dir/app/src/main/cpp" \
    "$project_dir/app/src/main/cpp/runtime_profile.cpp" \
    "$project_dir/tests/runtime_profile_test.cpp" \
    -o "$build_dir/runtime_profile_test_$headset"

  "$build_dir/runtime_profile_test_$headset"
done

"${CXX:-clang++}" \
  -std=c++20 \
  -Wall \
  -Wextra \
  -Werror \
  -pedantic \
  -I"$project_dir/include" \
  "$project_dir/src/frame_v1.cpp" \
  "$project_dir/tests/frame_v1_test.cpp" \
  -o "$build_dir/frame_v1_test"

"$build_dir/frame_v1_test"

"${CXX:-clang++}" \
  -std=c++20 \
  -Wall \
  -Wextra \
  -Werror \
  -pedantic \
  -I"$project_dir/include" \
  "$project_dir/src/frame_v1.cpp" \
  "$project_dir/src/capture_session.cpp" \
  "$project_dir/tests/capture_session_test.cpp" \
  -o "$build_dir/capture_session_test"

"$build_dir/capture_session_test"

"${CXX:-clang++}" \
  -std=c++20 \
  -Wall \
  -Wextra \
  -Werror \
  -pedantic \
  -I"$project_dir/include" \
  "$project_dir/src/enrollment.cpp" \
  "$project_dir/tests/enrollment_test.cpp" \
  -o "$build_dir/enrollment_test"

"$build_dir/enrollment_test"

if [ -n "${NLOHMANN_JSON_INCLUDE:-}" ]; then
  "${CXX:-clang++}" \
    -std=c++20 \
    -Wall \
    -Wextra \
    -Werror \
    -pedantic \
    -I"$project_dir/include" \
    -I"$NLOHMANN_JSON_INCLUDE" \
    "$project_dir/src/frame_v1.cpp" \
    "$project_dir/src/capture_session.cpp" \
    "$project_dir/src/capture_wire.cpp" \
    "$project_dir/tests/capture_wire_test.cpp" \
    -o "$build_dir/capture_wire_test"

  "$build_dir/capture_wire_test"
fi

"${CXX:-clang++}" \
  -std=c++20 \
  -Wall \
  -Wextra \
  -Werror \
  -pedantic \
  -I"$project_dir/include" \
  "$project_dir/src/frame_v1.cpp" \
  "$project_dir/tests/emit_frame_fixture.cpp" \
  -o "$build_dir/emit_frame_fixture"

"$build_dir/emit_frame_fixture" > "$build_dir/frame_v1.json"
python3 "$project_dir/tests/verify_frame_fixture.py" "$build_dir/frame_v1.json"

if [ -n "${PHANTHYMOTUS_DRIVER_G1_ROOT:-}" ]; then
  python3 "$project_dir/tests/verify_frame_fixture.py" \
    "$build_dir/frame_v1.json" \
    --driver-root "$PHANTHYMOTUS_DRIVER_G1_ROOT"
fi
