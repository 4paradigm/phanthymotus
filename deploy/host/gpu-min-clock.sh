#!/bin/bash
# Raise the GPU's *minimum* clock on a Jetson.
#
# Why this exists. On jp6.1 the pose/vop/depth/hand engines run far slower than
# the hardware can: the same TensorRT plan measures 4.67 ms under trtexec and
# 21.7 ms through our runtime on the same Orin. Sampling
# /sys/class/devfreq/*.gpu/cur_freq during each shows why — trtexec ramps the
# GPU 306 -> 1173 MHz, our path never leaves 306. Every inference is followed
# by a blocking synchronise and some host-side work, so the governor sees a
# mostly-idle GPU and keeps it at the floor; the low clock then makes each
# kernel slower. A DVFS feedback trap.
#
# This is a host setting and deliberately NOT something a container does to
# itself. It needs root on the host, it affects every GPU consumer on the
# machine (perception's vop/depth/pose/hand and actucore's VLA alike), and it
# costs power — so it is a provisioning decision somebody makes once and can
# see, not a side effect of starting a card.
#
# Measured on Orin 6 (Orin NX, jp6.1), idle board power against one hand
# inference through the production runtime:
#
#     min_freq    idle power   inference    gain per watt
#      306 MHz      5.68 W      17.98 ms    (the default)
#      612 MHz      6.94 W      11.69 ms    5.0 ms/W
#      816 MHz      7.88 W      10.83 ms    0.9 ms/W
#     1020 MHz      8.19 W       8.69 ms    6.6 ms/W   <- default here
#     1173 MHz     10.23 W       7.88 ms    0.4 ms/W   <- poor value, skip
#
# 1020 is the knee: the last step to the top costs 2 W for 0.8 ms. On a battery
# robot +2.5 W is a few percent of the whole machine, so pick deliberately.
#
# **jp5.11 does not need this.** Orin 5 measures 7.6 ms whether an inference is
# issued alone or a hundred back to back, matching its own trtexec — the trap
# does not occur there. This script checks before it changes anything.
#
#   sudo ./gpu-min-clock.sh            # apply the default (1020 MHz)
#   sudo ./gpu-min-clock.sh 612        # a cheaper point
#   sudo ./gpu-min-clock.sh --show     # report, change nothing
#   sudo ./gpu-min-clock.sh --reset    # back to the hardware floor
#
# The proper long-term fix is in the runtime, not here: keeping the GPU busy
# (CUDA graphs, batching, not synchronising every call) makes the governor ramp
# on its own and costs nothing while idle. Until then this buys most of it.
set -euo pipefail

DEV=$(ls -d /sys/class/devfreq/*gpu* 2>/dev/null | head -1 || true)
if [ -z "$DEV" ]; then
  echo "no GPU devfreq node found — is this a Jetson?" >&2
  exit 1
fi

MIN=$(cat "$DEV/min_freq"); MAX=$(cat "$DEV/max_freq"); CUR=$(cat "$DEV/cur_freq")
FLOOR=$(tr ' ' '\n' < "$DEV/available_frequencies" | sort -n | head -1)

show() {
  printf 'device   %s\n' "$DEV"
  printf 'min      %s MHz\n' "$((MIN / 1000000))"
  printf 'max      %s MHz\n' "$((MAX / 1000000))"
  printf 'current  %s MHz\n' "$((CUR / 1000000))"
}

case "${1:-}" in
  --show) show; exit 0 ;;
  --reset) TARGET=$FLOOR ;;
  "") TARGET=1020000000 ;;
  *) TARGET=$(( ${1%%[^0-9]*} ))
     [ "$TARGET" -lt 10000 ] && TARGET=$((TARGET * 1000000)) ;;
esac

if [ "$TARGET" -gt "$MAX" ]; then
  echo "requested $((TARGET / 1000000)) MHz is above this GPU's max $((MAX / 1000000)) MHz" >&2
  exit 1
fi

# Only worth doing where the trap actually occurs. Keyed off the L4T release
# rather than guessing from the board name: R36 is JetPack 6.x, R35 is 5.x.
REL=$(sed -nE 's/^# R([0-9]+) .*/\1/p' /etc/nv_tegra_release 2>/dev/null | head -1)
if [ "${FORCE:-0}" != "1" ] && [ "${REL:-0}" -lt 36 ] && [ "$TARGET" != "$FLOOR" ]; then
  echo "L4T R${REL:-?} (JetPack 5.x) does not show the idle-downclock problem —"
  echo "measured 7.6 ms per inference whether issued alone or back to back."
  echo "Re-run with FORCE=1 if you have measured otherwise on this machine."
  exit 0
fi

echo "$TARGET" > "$DEV/min_freq"
sleep 1
MIN=$(cat "$DEV/min_freq"); CUR=$(cat "$DEV/cur_freq")
show
