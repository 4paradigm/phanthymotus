#!/bin/sh
set -eu

usage() {
  cat >&2 <<'EOF'
Usage: launch_capture.sh --platform meta|pico [--resume]

Set ADB_SERIAL when more than one Android device is connected. First pairing
also requires CORE_WSS_URL, PAIRING_ID, and CA_CERT_BASE64 or CA_CERT_FILE.
EOF
}

platform=
resume=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --platform)
      if [ -n "$platform" ] || [ "$#" -lt 2 ]; then
        usage
        exit 2
      fi
      platform=$2
      shift 2
      ;;
    --platform=*)
      if [ -n "$platform" ]; then
        usage
        exit 2
      fi
      platform=${1#--platform=}
      shift
      ;;
    --resume)
      if [ "$resume" = true ]; then
        usage
        exit 2
      fi
      resume=true
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
  meta)
    package_activity=com.phanthymotus.questcapture/android.app.NativeActivity
    ;;
  pico)
    package_activity=com.phanthymotus.picocapture/android.app.NativeActivity
    ;;
  '')
    echo "--platform meta|pico is required" >&2
    usage
    exit 2
    ;;
  *)
    echo "Unsupported platform: $platform (expected meta or pico)" >&2
    exit 2
    ;;
esac

adb_bin=${ADB:-}
if [ -z "$adb_bin" ]; then
  adb_bin=$(command -v adb || true)
elif [ "${adb_bin#*/}" = "$adb_bin" ]; then
  adb_bin=$(command -v "$adb_bin" || true)
fi
if [ -z "$adb_bin" ] && [ -n "${ANDROID_SDK_ROOT:-}" ]; then
  adb_bin="$ANDROID_SDK_ROOT/platform-tools/adb"
fi
if [ -z "$adb_bin" ] || [ ! -x "$adb_bin" ]; then
  echo "adb is required; set ADB or ANDROID_SDK_ROOT" >&2
  exit 2
fi

set -- "$adb_bin"
if [ -n "${ADB_SERIAL:-}" ]; then
  set -- "$@" -s "$ADB_SERIAL"
fi

if [ "$resume" = true ]; then
  exec "$@" shell am start -S -n "$package_activity"
fi

if [ -z "${CORE_WSS_URL:-}" ] || [ -z "${PAIRING_ID:-}" ] || \
   { [ -z "${CA_CERT_BASE64:-}" ] && [ -z "${CA_CERT_FILE:-}" ]; }; then
  echo "CORE_WSS_URL, PAIRING_ID and CA_CERT_BASE64 (or CA_CERT_FILE) are required for first pairing" >&2
  exit 2
fi
if [ -n "${CA_CERT_FILE:-}" ] && [ ! -f "$CA_CERT_FILE" ]; then
  echo "CA_CERT_FILE must be a readable PEM file" >&2
  exit 2
fi

if [ -z "${PAIRING_CODE:-}" ]; then
  if [ ! -t 0 ]; then
    echo "PAIRING_CODE is required when stdin is not a terminal" >&2
    exit 2
  fi
  printf "One-time pairing code: " >&2
  restore_echo() {
    stty echo 2>/dev/null || true
  }
  trap restore_echo EXIT HUP INT TERM
  stty -echo
  IFS= read -r PAIRING_CODE
  stty echo
  trap - EXIT HUP INT TERM
  printf "\n" >&2
fi
if [ -n "${CA_CERT_BASE64:-}" ]; then
  ca_base64=$CA_CERT_BASE64
else
  ca_base64=$(base64 < "$CA_CERT_FILE" | tr -d '\r\n')
fi
case "$ca_base64" in
  ''|*[!A-Za-z0-9+/=]*)
    echo "CA certificate base64 is invalid" >&2
    exit 2
    ;;
esac
if [ $((${#ca_base64} % 4)) -ne 0 ] || [ ${#ca_base64} -gt 49152 ]; then
  echo "CA certificate base64 has an invalid size" >&2
  exit 2
fi

# The pairing code is a 60-second, one-use secret. It is supplied only as an
# Activity extra and is never placed in the WSS URL or Android log output.
exec "$@" shell am start -S \
  -n "$package_activity" \
  --es core_wss_url "$CORE_WSS_URL" \
  --es pairing_id "$PAIRING_ID" \
  --es pairing_code "$PAIRING_CODE" \
  --es ca_certificate_base64 "$ca_base64"
