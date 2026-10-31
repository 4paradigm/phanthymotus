#!/bin/bash
# Keep the VPC build host's disk from filling up with BuildKit cache.
#
# Why this exists. On 2026-10-31 the build host (172.18.66.241) had 25 GB left
# on a 985 GB disk: 686 GB of it was build cache, nearly all reclaimable, and
# nothing ever removed it. A ModelHub build needs the 11 GB (compressed) vLLM
# base unpacked on top of that, so it could not run at all — and a full disk
# fails every other build on the machine too, in ways that look like network
# or registry trouble.
#
# What it removes, in order, and nothing else:
#   1. build cache not used for KEEP_HOURS (default 30 days);
#   2. if free space is still under MIN_FREE_GB, build cache not used for
#      SHORT_KEEP_HOURS (default 7 days);
#   3. dangling images (untagged — nothing can refer to them by name).
# Tagged images, containers and volumes are never touched. Cache a running
# build is using is never pruned (BuildKit skips in-use records), and the
# 16 GB perception layers stay warm as long as somebody built them this week.
#
# Runs from phanthy-build-cache-gc.timer; safe to run by hand:
#   sudo /opt/phanthy-motus/build-cache-gc.sh
set -uo pipefail

KEEP_HOURS="${KEEP_HOURS:-720}"
SHORT_KEEP_HOURS="${SHORT_KEEP_HOURS:-168}"
MIN_FREE_GB="${MIN_FREE_GB:-150}"

# A prune of a few hundred GB takes tens of minutes; never stack two.
exec 9>/run/phanthy-build-cache-gc.lock
if ! flock -n 9; then
    echo "another cache GC is still running — skipping"
    exit 0
fi

ROOT_DIR="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
free_gb() { df -BG --output=avail "${ROOT_DIR}" | tail -1 | tr -dc '0-9'; }

before="$(free_gb)"
echo "free ${before} GB on ${ROOT_DIR}; pruning build cache unused for ${KEEP_HOURS}h"
docker builder prune -f --filter "until=${KEEP_HOURS}h" | tail -1

if [ "$(free_gb)" -lt "${MIN_FREE_GB}" ]; then
    echo "still under ${MIN_FREE_GB} GB free; pruning build cache unused for ${SHORT_KEEP_HOURS}h"
    docker builder prune -f --filter "until=${SHORT_KEEP_HOURS}h" | tail -1
fi

docker image prune -f | tail -1

after="$(free_gb)"
echo "free ${before} GB -> ${after} GB"
if [ "${after}" -lt "${MIN_FREE_GB}" ]; then
    # Loud, not fatal: what is left is cache from this week or tagged images,
    # which is a decision for a person, not for a timer.
    echo "WARNING: only ${after} GB free after GC — look at 'docker system df'"
fi
