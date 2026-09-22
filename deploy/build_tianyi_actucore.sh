#!/usr/bin/env bash
# Build only. No push, image transfer, Compose changes or service startup.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: build_tianyi_actucore.sh LOCAL_IMAGE_TAG' >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build --platform linux/arm64 --progress plain \
  -f "$root/actucore/Dockerfile.cpu" -t "$1" "$root"
