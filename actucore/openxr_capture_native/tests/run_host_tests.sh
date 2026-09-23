#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/motus-openxr-capture-host.XXXXXX")
trap 'rm -r "$build_dir"' EXIT HUP INT TERM

manifest="$project_dir/app/src/main/AndroidManifest.xml"
grep -F 'android:exported="true"' "$manifest" >/dev/null
grep -F 'android:permission="android.permission.DUMP"' "$manifest" >/dev/null

# The build must execute Gradle only from a fresh extraction of a private,
# verified archive, never by reopening or executing a shared cache path.
build_script="$project_dir/scripts/build_android.sh"
grep -F 'verified_gradle_root=$(mktemp -d "${TMPDIR:-/tmp}/motus-gradle-' \
  "$build_script" >/dev/null
grep -F 'cp "$archive" "$verified_archive"' "$build_script" >/dev/null
grep -F 'actual=$(shasum -a 256 "$verified_archive"' "$build_script" >/dev/null
grep -F 'unzip -q "$verified_archive" -d "$verified_gradle_root"' \
  "$build_script" >/dev/null
verified_line=$(grep -n -F 'actual=$(shasum -a 256 "$verified_archive"' \
  "$build_script" | cut -d: -f1)
unzip_line=$(grep -n -F 'unzip -q "$verified_archive"' "$build_script" | cut -d: -f1)
if [ -z "$verified_line" ] || [ -z "$unzip_line" ] || [ "$verified_line" -ge "$unzip_line" ]; then
  echo "build_android.sh must verify the private archive before extracting it" >&2
  exit 1
fi
grep -F 'verified_gradle="$verified_gradle_root/gradle-$gradle_version/bin/gradle"' \
  "$build_script" >/dev/null
if grep -F 'gradle_home="$tool_cache/gradle-$gradle_version"' "$build_script" >/dev/null; then
  echo "build_android.sh must not execute a predictable expanded Gradle cache" >&2
  exit 1
fi

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
  "${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror -pedantic \
    -I"$project_dir/app/src/main/cpp" -I"$NLOHMANN_JSON_INCLUDE" \
    "$project_dir/tests/operator_state_test.cpp" -o "$build_dir/operator_state_test"
  "$build_dir/operator_state_test"
  "${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror -pedantic \
    -I"$project_dir/app/src/main/cpp" -I"$NLOHMANN_JSON_INCLUDE" \
    "$project_dir/tests/ik_visualization_test.cpp" -o "$build_dir/ik_visualization_test"
  "$build_dir/ik_visualization_test"
  if [ -n "${MOTUS_IK_REPLAY:-}" ]; then
    "${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror -pedantic \
      -I"$project_dir/app/src/main/cpp" -I"$NLOHMANN_JSON_INCLUDE" \
      "$project_dir/tests/ik_visualization_replay.cpp" -o "$build_dir/ik_visualization_replay"
    "$build_dir/ik_visualization_replay" "$MOTUS_IK_REPLAY"
  fi
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

  # Feed real Python producer output to the native parser. Hand-written G1
  # fixtures alone missed the Cartesian capability added for motion_control.
  "${PYTHON:-python3}" "$project_dir/tests/emit_assignment_fixture.py" \
    > "$build_dir/assignments.jsonl"
  "$build_dir/capture_wire_test" "$build_dir/assignments.jsonl"
else
  echo "capture_wire_test: SKIP (set NLOHMANN_JSON_INCLUDE)" >&2
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

python3 "$project_dir/tests/verify_frame_fixture.py" \
  "$build_dir/frame_v1.json" --actucore-plugins "$project_dir/../plugins"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror -pedantic \
  -I"$project_dir/include" "$project_dir/tests/operator_panel_test.cpp" \
  -o "$build_dir/operator_panel_test"
"$build_dir/operator_panel_test"
