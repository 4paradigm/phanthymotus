#!/usr/bin/env bash
# Build only. No push, image transfer, Compose changes or service startup.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# == 2 && "$2" != --trajectory ) ]]; then
  echo 'Usage: build_tianyi_actucore.sh LOCAL_IMAGE_TAG [--trajectory]' >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trajectory=false
if [[ ${2:-} == --trajectory ]]; then trajectory=true; fi
docker build --platform linux/arm64 --progress plain \
  --build-arg "INSTALL_TRAJECTORY=$trajectory" \
  -f "$root/actucore/Dockerfile.cpu" -t "$1" "$root"
