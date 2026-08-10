#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage: build_android.sh [--platform meta|pico|all]

Builds both Android OpenXR APKs by default. Select one platform to shorten an
incremental build.
EOF
}

platform=all
platform_seen=false
while [ "$#" -gt 0 ]; do
  case "$1" in
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

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
android_sdk_root=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}
if [ -z "$android_sdk_root" ]; then
  echo "ANDROID_SDK_ROOT is required" >&2
  exit 2
fi
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
gradle_home="$tool_cache/gradle-$gradle_version"
if [ ! -x "$gradle_home/bin/gradle" ]; then
  mkdir -p "$tool_cache"
  archive="$tool_cache/gradle-$gradle_version-bin.zip"
  curl --fail --location --silent --show-error \
    "https://services.gradle.org/distributions/gradle-$gradle_version-bin.zip" \
    --output "$archive"
  actual=$(shasum -a 256 "$archive" | awk '{print $1}')
  if [ "$actual" != "$gradle_sha256" ]; then
    echo "Gradle checksum mismatch: $actual" >&2
    exit 3
  fi
  unzip -q -o "$archive" -d "$tool_cache"
fi

case "$platform" in
  meta)
    exec "$gradle_home/bin/gradle" \
      --no-daemon \
      --project-dir "$project_dir" \
      :app:assembleMetaDebug
    ;;
  pico)
    exec "$gradle_home/bin/gradle" \
      --no-daemon \
      --project-dir "$project_dir" \
      :app:assemblePicoDebug
    ;;
  all)
    exec "$gradle_home/bin/gradle" \
      --no-daemon \
      --project-dir "$project_dir" \
      :app:assembleMetaDebug \
      :app:assemblePicoDebug
    ;;
esac
