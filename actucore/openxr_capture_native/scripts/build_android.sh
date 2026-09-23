#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage: build_android.sh [--platform meta|pico|all] [--build-type debug|release]

Builds both Android OpenXR APKs by default. Select one platform to shorten an
incremental build.
EOF
}

platform=all
build_type=debug
platform_seen=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --build-type)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      build_type=$2
      shift 2
      ;;
    --platform)
      if [ "$platform_seen" = true ] || [ "$#" -lt 2 ]; then
        usage
        exit 2
      fi
      platform=$2
      platform_seen=true
      shift 2
      ;;
    --platform=*)
      if [ "$platform_seen" = true ]; then
        usage
        exit 2
      fi
      platform=${1#--platform=}
      platform_seen=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$platform" in
  meta|pico|all) ;;
  *)
    echo "Unsupported platform: $platform (expected meta, pico, or all)" >&2
    exit 2
    ;;
esac

case "$build_type" in
  debug) variant=Debug ;;
  release)
    variant=Release
    for variable in MOTUS_APK_KEYSTORE MOTUS_APK_STORE_PASSWORD MOTUS_APK_KEY_ALIAS MOTUS_APK_KEY_PASSWORD; do
      # Only print the variable name, never its signing secret.
      value=$(printenv "$variable" || true)
      [ -n "$value" ] || { echo "Release signing variable required: $variable" >&2; exit 2; }
    done
    [ -f "$MOTUS_APK_KEYSTORE" ] || { echo "Release keystore is missing" >&2; exit 2; }
    ;;
  *) echo "Unsupported build type" >&2; exit 2 ;;
esac

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
android_sdk_root=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}
if [ -z "$android_sdk_root" ]; then
  echo "ANDROID_SDK_ROOT is required" >&2
  exit 2
fi
# Gradle rejects inherited SDK locations that disagree with the selected SDK.
export ANDROID_HOME="$android_sdk_root"
export ANDROID_SDK_ROOT="$android_sdk_root"
if [ -z "${JAVA_HOME:-}" ] || [ ! -x "$JAVA_HOME/bin/java" ]; then
  echo "JAVA_HOME must point to JDK 17" >&2
  exit 2
fi

for required in \
  "$android_sdk_root/platforms/android-35/android.jar" \
  "$android_sdk_root/build-tools/35.0.1/aapt2" \
  "$android_sdk_root/ndk/27.0.12077973/build/cmake/android.toolchain.cmake" \
  "$android_sdk_root/cmake/3.22.1/bin/cmake"
do
  if [ ! -e "$required" ]; then
    echo "Android build dependency missing: $required" >&2
    exit 2
  fi
done

gradle_version=8.9
gradle_sha256=d725d707bfabd4dfdc958c624003b3c80accc03f7037b5122c4b1d0ef15cecab
tool_cache=${MOTUS_ANDROID_TOOL_CACHE:-${TMPDIR:-/tmp}/motus-android-tools}
archive="$tool_cache/gradle-$gradle_version-bin.zip"

# The reusable cache is only an untrusted byte source.  Copy or download the
# archive into a private build-scoped directory, then checksum and extract that
# same private file.  A concurrent cache replacement can therefore only cause
# a checksum failure; it can never change the program we execute after verify.
verified_gradle_root=$(mktemp -d "${TMPDIR:-/tmp}/motus-gradle-$gradle_version.verified.XXXXXX")
chmod 700 "$verified_gradle_root"
trap 'rm -rf "$verified_gradle_root"' EXIT HUP INT TERM
verified_archive="$verified_gradle_root/gradle-$gradle_version-bin.zip"

if [ -f "$archive" ]; then
  cp "$archive" "$verified_archive"
else
  curl --fail --location --silent --show-error \
    "https://services.gradle.org/distributions/gradle-$gradle_version-bin.zip" \
    --output "$verified_archive"
fi
actual=$(shasum -a 256 "$verified_archive" | awk '{print $1}')
if [ "$actual" != "$gradle_sha256" ]; then
  echo "Gradle checksum mismatch: $actual" >&2
  exit 3
fi

unzip -q "$verified_archive" -d "$verified_gradle_root"
verified_gradle="$verified_gradle_root/gradle-$gradle_version/bin/gradle"
if [ ! -x "$verified_gradle" ]; then
  echo "Verified Gradle archive did not contain the expected executable" >&2
  exit 3
fi

case "$platform" in
  meta)
    "$verified_gradle" \
      --no-daemon \
      --project-dir "$project_dir" \
      ":app:assembleMeta${variant}"
    ;;
  pico)
    "$verified_gradle" \
      --no-daemon \
      --project-dir "$project_dir" \
      ":app:assemblePico${variant}"
    ;;
  all)
    "$verified_gradle" \
      --no-daemon \
      --project-dir "$project_dir" \
      ":app:assembleMeta${variant}" \
      ":app:assemblePico${variant}"
    ;;
esac
