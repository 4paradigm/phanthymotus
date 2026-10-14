#!/usr/bin/env python3
"""
plugins/echo_cancellation.py — keep the robot's own voice out of its own microphone.

The robot's TTS reaches its microphone acoustically, and nothing in perception
removed it until this card existed. The consequences were spread across three
plugins and none of them looked like an echo problem: ASR transcribes the robot's
own sentence and hands it to the LLM as if a person had said it; speaker
recognition enrols that voice as a person (`plugins/speaker.py` says so in a
comment); and nobody can interrupt the robot mid-sentence, because the ASR it
would have to reach is busy listening to the robot.

This is a card in the audio path, not a switch inside ASR, for one reason that
decides it: `soundevent` subscribes to the microphone topic too, and speaker
recognition runs off the segment ASR cut. Cleaning the signal inside ASR cleans
it for ASR alone. Wired as `mic -> echo_cancellation -> {asr, soundevent}` it is
cleaned once, for everything downstream, and the cleaned audio is a topic a human
can listen to on the dashboard.

Two modes in this version:

  off                 pass the microphone through untouched (for A/B)
  mute_while_talking  publish silence while the robot's voice could be in the mic

`mute_while_talking` is what this card ships with, and the honest description of
it is half-duplex: it cannot be interrupted, because while the robot talks the
microphone genuinely carries nothing. It is also the only mode that works on a
robot whose speech never passes through Agent Core at all (vendor TTS over a ROS
service — Tianyi), and it removes 100% of the echo rather than 20 dB of it.
Cancellation, which keeps the mic open and allows barge-in, needs an adaptive
filter and is the next step; the delay measurement it will depend on is already
here (see _DelayTracker) precisely so that its one unanswerable parameter can be
read off a running robot rather than guessed.

### The reference signal, and why it is not a canvas connection

The card needs to know what the robot is saying. That is the TTS card's audio
output — the same PCM the speaker driver plays.

It is a config field and not a second in-port because the canvas cannot express
it: connections are matched by `format` alone (agent-core/web/js/canvas.js), a
card-level drop picks the first in-port of that format, and what reaches the
plugin is a role-less `input_topics` list ordered by click order
(web/js/topic-derive.js). Two `audio/pcm-16k` in-ports would be a coin flip.
More to the point, the reference is not input data; it is "what another card on
this machine is publishing right now", and there is almost always exactly one
right answer — so the default (`auto`) resolves it by asking the TTS plugin in
this same process, the way ASR already takes a handle to the speaker-recognition
plugin (main.py). Nobody has to fill anything in.

The cost of `auto` is that it can resolve to nothing — a robot with vendor TTS,
or a TTS card in another container — and silently degrade this card to a
passthrough. So `info()` always reports which topic it resolved to and whether
any reference audio has actually arrived, and the card's own description says so
when it has not.

### Why the mute has a tail, and why the tail is `auto` by default

Publishing the last frame of an utterance is not the end of the sound. The
speaker path buffers: the G1 driver prefills 9600 bytes (~300 ms), merges into
300 ms blocks and runs up to 240 ms ahead of the MCU's queue
(phanthymotus-driver unitree/g1/device.py), and the measured gap between "TTS
finished publishing" and "the robot stopped being audible" is 0.7-1.2 s on that
robot. A mute that ends when the reference stream ends therefore transcribes the
last few words of the robot's own sentence — which is the exact failure this card
exists to remove.

That number is a property of the robot and its speaker, nobody knows it in
advance, and it drifts with bus load. So the default is `auto`: the tail follows
the delay the card measures for itself, and the explicit values stay available
and authoritative for anyone who wants to pin it.
"""

from __future__ import annotations

import array
import copy
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)


# ── audio contract (perception/README.md § Audio Requirements) ───────────────

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
AUDIO_FORMAT = "audio/pcm-16k"
# Same alias set the other audio plugins accept: some drivers still publish the
# old name and refusing them here would silence a robot that works today.
_AUDIO_FORMATS = frozenset((AUDIO_FORMAT, "pcm_16k_16bit_mono"))

# The 8-byte end-of-utterance marker plugins/tts.py publishes on its audio
# topic. It is 4 samples of square wave, not speech: counted as "the utterance
# ended", never as reference audio, and never fed to the delay tracker.
AUDIO_EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'

OUTPUT_SUFFIX = "/clean"

_AUDIO_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)


# ── config vocabulary ────────────────────────────────────────────────────────

# No mode field, and that is a decision rather than an omission. This card
# cancels; it does not offer muting as an alternative. Half-duplex muting cannot
# do the job the card exists for — being interruptible — so offering it as a
# choice would be offering a setting that fails the requirement, and asking an
# operator to understand the trade in order to pick the only workable side of
# it. What remains of muting is two bounded internal behaviours, neither of
# them a setting:
#
#   * the first seconds of the robot's first utterance, before the delay has
#     been measured. Subtracting an unaligned reference adds noise instead of
#     removing echo, and passing the audio through means ASR transcribes the
#     robot — so the microphone is held silent until there is something to align
#     to. Roughly 3 s of the first sentence, once per session.
#   * nothing else. When the coupling disappears mid-session the card passes
#     audio through rather than muting: no measurable echo means there is
#     nothing to remove, and a permanent mute would be exactly the unacceptable
#     behaviour this paragraph exists to rule out.

REF_AUTO = "auto"
DEFAULT_ROBOT_VOICE_TOPIC = REF_AUTO

MUTE_TAIL_AUTO = "auto"
# Explicit choices, in the units the card reports back in `info()` so the two
# can be compared by eye. Not a free-form number field: the answer is between
# 0.3 and 1.5 s on every robot measured, and a text box invites 5000.
MUTE_TAIL_CHOICES: Dict[str, Optional[float]] = {
    MUTE_TAIL_AUTO: None,
    "300ms": 0.3,
    "800ms": 0.8,
    "1500ms": 1.5,
}
# Also fixed. `auto` follows the delay the card measures for itself, which is
# the only answer nobody has to look up — and since muting is now a bounded
# warmup rather than a mode, there is nothing left for an operator to tune here.
DEFAULT_MUTE_TAIL = MUTE_TAIL_AUTO
# What `auto` uses before it has measured anything. 0.8 s is the middle of the
# 0.7-1.2 s measured on G1 — deliberately not the smallest of them, because
# erring short leaks the end of the robot's sentence into ASR while erring long
# only delays when the mic reopens.
AUTO_TAIL_FALLBACK_S = 0.8
# Added on top of the measured delay in `auto`, for the acoustic path and the
# jitter in the measurement itself.
AUTO_TAIL_MARGIN_S = 0.2


# ── delay measurement ────────────────────────────────────────────────────────

# 10 ms envelope hop: what is being measured is a 300-1200 ms bulk transport
# delay, and an envelope carries that as well as the waveform does at 1/160th of
# the arithmetic.
ENVELOPE_HOP_SAMPLES = 160
ENVELOPE_HOP_S = ENVELOPE_HOP_SAMPLES / SAMPLE_RATE
ENVELOPE_HISTORY_S = 6.0
# Search range. Covers G1's measured 0.7-1.2 s with room above it; a longer
# range costs nothing here (it is 120 lags of a 300-point dot product) but a
# shorter one would silently report the wrong peak on a slow speaker.
DELAY_SEARCH_MAX_S = 1.5
DELAY_WINDOW_S = 3.0
DELAY_ESTIMATE_INTERVAL_S = 1.0
# Below this much reference audio in the window there is nothing to correlate
# against and the peak would be noise wearing a confidence interval.
DELAY_MIN_REF_ACTIVE_S = 0.5
# An accepted peak has to be both strong in absolute terms and clearly above the
# rest of the lag profile. Without the second test a flat profile still has a
# maximum, and reporting its position is how a card ends up stating a delay of
# 0 ms with no echo in the room at all.
DELAY_MIN_PEAK = 0.30
DELAY_MIN_PROMINENCE = 2.0
DELAY_HISTORY = 20
# How many seconds of the robot actually talking may pass with no accepted
# measurement before the one on hand is treated as stale. Counted in far-end
# activity rather than wall clock: a measurement taken an hour ago is still the
# right answer if the robot has been silent since. Crossing it means the
# microphone has stopped hearing the speaker — somebody turned the volume down,
# or unplugged it — and cancellation has nothing to align to.
DELAY_STALE_AFTER_FAR_ACTIVE_S = 15.0

# RMS below this (on int16 scale) is silence rather than speech, for deciding
# whether the robot is talking and whether a window is worth correlating.
SILENCE_RMS = 60.0


def _chunk_timestamp(message: Any) -> float:
    """Publisher's stamp, falling back to now.

    Same rule as plugins/asr.py and plugins/soundevent.py: a stamp below 1e9 is
    an unset header rather than 1970, and a producer that leaves it unset is
    common enough that refusing its audio would be a regression.
    """
    try:
        stamp = message.header.stamp
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError):
        return time.time()
    if not math.isfinite(timestamp) or timestamp < 1e9:
        return time.time()
    return timestamp


def _envelope(pcm: bytes) -> np.ndarray:
    """Per-hop RMS of a PCM chunk, int16 scale, dropping the partial tail hop."""
    samples = np.frombuffer(pcm, dtype='<i2')
    hops = samples.size // ENVELOPE_HOP_SAMPLES
    if hops == 0:
        return np.empty(0, dtype=np.float32)
    block = samples[:hops * ENVELOPE_HOP_SAMPLES].reshape(hops, ENVELOPE_HOP_SAMPLES)
    return np.sqrt(np.mean(np.square(block.astype(np.float32)), axis=1))


# A chunk boundary is not an envelope boundary: the documented minimum chunk is
# 512 samples (perception/README.md) and the hop is 160, so every chunk leaves a
# 32-sample remainder. Dropping it loses 6% of the signal at that size — and far
# more at the 170-sample chunks a 48 kHz USB mic produces before the driver
# buffers them. The remainder is carried into the next chunk instead, and only
# discarded when the stream is not continuous there.
MAX_ENVELOPE_JOIN_GAP_S = 0.05
# How far behind the write cursor a chunk's stamp may be and still be treated as
# a continuation. plugins/tts.py publishes the first PREBUF_FRAMES=5 frames of
# every utterance in one burst — `emit()` backdates the pacing schedule so they
# are all already due — and `publish()` stamps each one with now(), so 500 ms of
# audio arrives carrying near-identical stamps. Positioning that by stamp piles
# five chunks on top of each other at the start of every utterance, which is
# exactly where the echo starts, and leaves the series non-monotonic in time —
# `np.interp` then returns nonsense with no warning at all.
MAX_ENVELOPE_BACKWARD_S = 1.0


class _ContiguousStream:
    """Timeline bookkeeping for one stream of nominally contiguous audio.

    The rule, which both the envelope and the reference ring follow: **audio is
    contiguous, and the stamp is only used to detect that it is not.** A chunk
    whose stamp continues the write cursor (or sits behind it, as a burst does)
    extends the cursor by its own duration; only a stamp well ahead of the
    cursor is treated as a gap and resynchronised.

    `position(stamp, samples)` returns the time of the chunk's first sample and
    whether it was a resynchronisation.
    """

    def __init__(self) -> None:
        self._next_t: Optional[float] = None
        self._last_t = 0.0

    @property
    def next_t(self) -> Optional[float]:
        return self._next_t

    def reset(self) -> None:
        self._next_t = None

    def position(self, stamp: float) -> Tuple[float, bool]:
        if self._next_t is None:
            start = stamp
            resynced = True
        else:
            drift = stamp - self._next_t
            if drift > MAX_ENVELOPE_JOIN_GAP_S or drift < -MAX_ENVELOPE_BACKWARD_S:
                # Never backwards past what has already been written: a clock
                # that jumped must not make the series non-monotonic, because
                # every consumer of it interpolates.
                start = max(stamp, self._last_t)
                resynced = True
            else:
                start = self._next_t
                resynced = False
        self._last_t = max(self._last_t, start)
        return start, resynced

    def advance(self, to_t: float) -> None:
        self._next_t = to_t
        self._last_t = max(self._last_t, to_t)


class _EnvelopeStream:
    """One stream's 100 Hz RMS envelope, with the sub-hop remainder carried over.

    Separate from the tracker because the carry has a failure mode of its own:
    joining a chunk onto a remainder from before a gap would date the joined
    audio to the wrong time, which is indistinguishable downstream from a change
    in the echo delay — the one thing this whole file exists to measure.
    """

    def __init__(self) -> None:
        self.points: Deque[Tuple[float, float]] = deque()
        self._timeline = _ContiguousStream()
        self._carry = b''

    def add(self, pcm: bytes, stamp: float, now: float) -> None:
        start_t, resynced = self._timeline.position(stamp)
        if resynced:
            self._carry = b''
        elif self._carry:
            pcm = self._carry + pcm
            self._carry = b''

        samples = len(pcm) // BYTES_PER_SAMPLE
        hops = samples // ENVELOPE_HOP_SAMPLES
        if hops:
            consumed = hops * ENVELOPE_HOP_SAMPLES * BYTES_PER_SAMPLE
            envelope = _envelope(pcm[:consumed])
            for index, value in enumerate(envelope):
                self.points.append((start_t + index * ENVELOPE_HOP_S, float(value)))
            self._carry = pcm[consumed:]
        else:
            self._carry = pcm
        self._timeline.advance(start_t + hops * ENVELOPE_HOP_S)

        horizon = now - ENVELOPE_HISTORY_S
        while self.points and self.points[0][0] < horizon:
            self.points.popleft()

    def __len__(self) -> int:
        return len(self.points)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        count = len(self.points)
        times = np.fromiter((t for t, _ in self.points), dtype=np.float64, count=count)
        values = np.fromiter((v for _, v in self.points), dtype=np.float64, count=count)
        return times, values


class _DelayTracker:
    """How long after the robot sends audio its own voice reaches the mic.

    Normalised cross-correlation of the two envelopes over a lag range. Kept
    separate from the node so it can be tested against synthetic signals with no
    ROS, no threads and no audio hardware, which is the only way the numbers in
    its thresholds can be defended.

    Thread-safety is the caller's: the node feeds it from the two subscription
    callbacks, which `ThreadingHTTPServer` and the executor run concurrently.
    """

    def __init__(self) -> None:
        self._mic = _EnvelopeStream()
        self._ref = _EnvelopeStream()
        self._estimates: Deque[float] = deque(maxlen=DELAY_HISTORY)
        self._last_estimate_at = 0.0
        self._attempts = 0
        self._rejected = 0
        self._far_active_since_accept = 0.0
        # The median is cached because `measured` is read once per microphone
        # chunk on the cancellation path, and profiling on Orin 5 put
        # `np.median` at a third of that path's cost.
        self._median: Optional[float] = None

    # ── feeding ──────────────────────────────────────────────────────────

    def add_mic(self, pcm: bytes, start_t: float, now: float) -> None:
        self._mic.add(pcm, start_t, now)

    def add_ref(self, pcm: bytes, start_t: float, now: float) -> None:
        self._ref.add(pcm, start_t, now)

    def note_far_active(self, seconds: float) -> None:
        """Audible reference audio, for judging whether a measurement is stale."""
        self._far_active_since_accept += seconds

    def fresh(self) -> bool:
        """True when the measurement still describes the current acoustic path."""
        return (bool(self._estimates)
                and self._far_active_since_accept < DELAY_STALE_AFTER_FAR_ACTIVE_S)

    # ── estimating ───────────────────────────────────────────────────────

    def due(self, now: float) -> bool:
        return (now - self._last_estimate_at) >= DELAY_ESTIMATE_INTERVAL_S

    def estimate(self, now: float) -> Optional[float]:
        """Try to measure the delay, in seconds. None when the data cannot say.

        Returning None is the common case and not a failure: it means no
        reference audio was playing, or the correlation had no clear peak. A
        measurement is only offered when it is defensible.
        """
        self._last_estimate_at = now
        mic_t, mic_v = self._mic.arrays()
        ref_t, ref_v = self._ref.arrays()
        if mic_t.size < 32 or ref_t.size < 10:
            return None

        # The window ends where the microphone data ends, not at `now`. This
        # runs from the microphone callback, so the newest audio on hand is
        # always a chunk or two old — anchoring the window to the wall clock
        # made the coverage check fail every single time, which looks exactly
        # like "the correlation never finds anything".
        window_end = min(now, float(mic_t[-1]) + ENVELOPE_HOP_S)
        window_start = window_end - DELAY_WINDOW_S
        if float(mic_t[0]) > window_start:
            return None            # the mic does not cover the whole window yet

        self._attempts += 1
        points = int(round(DELAY_WINDOW_S / ENVELOPE_HOP_S))
        grid = window_start + np.arange(points) * ENVELOPE_HOP_S
        mic = np.interp(grid, mic_t, mic_v)

        # mic[t] can only carry the echo of ref[t - lag], so the reference grid
        # reaches DELAY_SEARCH_MAX_S further back. Outside the data it is zero,
        # not the nearest sample: np.interp clamps by default, which would
        # extend the last loud value of a finished utterance into a flat tail
        # and then count that tail as the robot still talking.
        lags = int(round(DELAY_SEARCH_MAX_S / ENVELOPE_HOP_S))
        ref_grid = (grid[0] - DELAY_SEARCH_MAX_S
                    + np.arange(points + lags) * ENVELOPE_HOP_S)
        ref_full = np.interp(ref_grid, ref_t, ref_v, left=0.0, right=0.0)

        active = float(np.count_nonzero(ref_full[lags:] > SILENCE_RMS)) * ENVELOPE_HOP_S
        if active < DELAY_MIN_REF_ACTIVE_S:
            return None

        mic_z = mic - mic.mean()
        mic_norm = float(np.linalg.norm(mic_z))
        if mic_norm <= 0.0:
            return None
        mic_z /= mic_norm

        profile = np.empty(lags, dtype=np.float64)
        for lag in range(lags):
            # ref_full is oldest-first and starts DELAY_SEARCH_MAX_S before the
            # mic window, so lag 0 sits at the far end of the search range.
            offset = lags - lag
            window = ref_full[offset:offset + points]
            window = window - window.mean()
            norm = float(np.linalg.norm(window))
            profile[lag] = 0.0 if norm <= 0.0 else float(np.dot(mic_z, window / norm))

        peak_index = int(np.argmax(profile))
        peak = float(profile[peak_index])
        typical = float(np.median(np.abs(profile)))
        if peak < DELAY_MIN_PEAK or peak < DELAY_MIN_PROMINENCE * max(typical, 1e-6):
            self._rejected += 1
            return None
        if peak_index >= lags - 1:
            # A peak pinned to the far end of the search range is the signature
            # of a delay *beyond* it, not of one exactly at it: measured with a
            # 2 s echo and a 1.5 s range, this bin is what comes back. Reporting
            # it would hand the gate a number that is wrong in the one direction
            # that matters, so it is refused and the range is the thing to widen.
            self._rejected += 1
            return None

        delay = peak_index * ENVELOPE_HOP_S
        self._accept(delay)
        return delay

    def _accept(self, delay: float) -> None:
        """Record an accepted measurement and refresh the cached median."""
        self._estimates.append(delay)
        self._far_active_since_accept = 0.0
        self._median = float(np.median(
            np.fromiter(self._estimates, dtype=np.float64, count=len(self._estimates))))

    # ── reporting ────────────────────────────────────────────────────────

    @property
    def measured(self) -> Optional[float]:
        """Median of the accepted estimates, in seconds. None before the first.

        Median rather than latest: a single estimate is one 3 s window, and the
        gate should not follow its noise.
        """
        return self._median

    @property
    def spread(self) -> Optional[float]:
        """p90-p10 of the accepted estimates, in seconds — the 'it moves' number."""
        if len(self._estimates) < 4:
            return None
        values = np.fromiter(self._estimates, dtype=np.float64)
        return float(np.percentile(values, 90) - np.percentile(values, 10))

    def report(self) -> Dict[str, Any]:
        measured = self.measured
        spread = self.spread
        return {
            "measured_echo_delay_ms": None if measured is None else round(measured * 1000),
            "echo_delay_spread_ms": None if spread is None else round(spread * 1000),
            "echo_delay_samples": len(self._estimates),
            "echo_delay_attempts": self._attempts,
            "echo_delay_rejected": self._rejected,
            "echo_delay_fresh": self.fresh(),
            "robot_talk_since_measurement_s": round(self._far_active_since_accept, 1),
        }


# ── cancellation ─────────────────────────────────────────────────────────────

# 16 ms blocks, 512-point FFT, 50% overlap — the overlap-save framing every
# frequency-domain adaptive filter uses. 16 ms is short enough that the extra
# latency this adds to the microphone path is below one chunk, and long enough
# that the FFT cost per second of audio stays in the single-digit milliseconds.
AEC_BLOCK = 256
AEC_FFT = 2 * AEC_BLOCK
AEC_BINS = AEC_FFT // 2 + 1

# How much echo *spread* the filter covers — reverberation and alignment error,
# **not** the transport delay. 128 ms, measured: against a 60 ms synthetic room,
# 64 ms of tail gives 23 dB, 128 ms gives 20, 256 ms gives 18 and 512 ms gives
# 14 — more partitions mean more gradient noise, so a tail far longer than the
# path actively costs ERLE. 128 ms leaves ~96 ms for the room and the alignment
# error after the 32 ms backoff below. The delay is measured and compensated by shifting
# the reference (see _RefBuffer), which is what keeps this short: covering G1's
# 450-1200 ms of speaker buffering with filter taps instead would need 5-75x the
# arithmetic to model a path that is mostly a constant offset.
AEC_TAIL_S = 0.128
# The reference is aligned to (measured delay - this), so the real echo lands
# inside the tail with slack on both sides of the estimate.
AEC_ALIGN_BACKOFF_S = 0.032

# NLMS step size, normalised by the reference power across partitions. Measured
# on synthetic echo paths (12 s, three seeds, 60 ms room): 0.2 reaches 20-23 dB
# ERLE and 0.3 reaches 22-25, but 0.4 collapses to ~0 dB and 0.5 diverges. The
# edge is between 0.3 and 0.4, so this sits a factor of two below it rather than
# taking the extra 2 dB — a diverged filter is worse than no filter, and the
# robots this runs on have speakers nothing has characterised yet.
AEC_MU = 0.2
# Bleeds the filter towards zero, bounding what a long run of bad updates can
# accumulate. 0.9995 per 16 ms block is a ~32 s time constant: slow enough not
# to fight convergence, fast enough that a diverged filter recovers.
AEC_LEAK = 0.9995
AEC_EPS = 1e-6

# Residual suppression. A broadband soft gate on the residual, ramped across the
# block so it cannot click — deliberately not a per-bin spectral suppressor,
# which needs its own windowed overlap-add framing; that is the next step here
# and would buy better double-talk quality rather than more echo removal.
# Residual suppression. A broadband soft gate on the residual, ramped across the
# block so it cannot click — deliberately not a per-bin spectral suppressor,
# which needs its own windowed overlap-add framing; that is the next step here,
# and it would buy better double-talk quality rather than more echo removal.
#
# The three values are the measured trade (12 s, 60 ms room, a person talking
# over the robot at a quarter of its level), ERLE against how much of the
# person's voice survives:
#
#   0.0  18.1 dB / 18.1 dB      0.5  22.9 dB / 12.0 dB      1.0  27.9 dB / 6.5 dB
#
# It stops at 1.0 because ERLE saturates there: 1.5 and 2.5 also give 27.9 dB
# and cost another 2-3 dB of the person's voice, so offering them would only let
# somebody choose a strictly worse point.
SUPPRESSION_STRENGTHS = {
    "gentle":     0.0,   # linear filter only — best double-talk, least echo removed
    "normal":     0.5,
    "aggressive": 1.0,
}
# Fixed, not offered on the card. `normal` keeps an interrupting voice 12 dB
# above the residual — enough for ASR — while removing 5 dB more of the robot
# than `gentle` does, and that 5 dB is the margin against the robot's own speech
# reaching ASR. Still overridable from config.yaml or an `action=config` call,
# which is how we would try `gentle` on a real robot without a rebuild; what it
# is not is a question put to an operator.
#
# The number that could overturn this is not in hand: whether the residual's
# *absolute* level still trips the VAD needs a real microphone, a real speaker
# and ASR in the loop.
DEFAULT_REMOVAL_STRENGTH = "normal"
# The floor the soft gate may attenuate to. Not zero: a residual gated to
# silence makes the robot's own voice disappear *and* takes the room with it, so
# a listener cannot tell a working canceller from a dead microphone.
SUPPRESSION_FLOOR = 0.05

# Divergence guard: zero the filter when its output is measurably *worse* than
# its input, over a window of far-end-active blocks.
#
# The threshold is -3 dB and not the +1 dB it started as, and the difference was
# only visible on a rig. A filter converged to 20 dB that is then talked over
# looks identical, *in powers alone*, to a filter that has learned nothing: in
# both cases the residual is dominated by the person. With +1 dB, Orin 5 zeroed
# a converged filter three times during one 4 s double-talk stretch and put
# "this microphone barely hears the speaker" on the card — reintroducing from
# inside the guard the exact failure the per-block skip exists to prevent.
#
# Powers cannot separate those two cases. Correlation can, and the delay tracker
# already measures it, so *that* decides whether there is coupling worth
# cancelling (`_DelayTracker.fresh`, and the node's cancel path). What is left
# here is the unambiguous half: an output worse than its input is never worth
# keeping.
AEC_GUARD_BLOCKS = 64            # ~1 s of far-end audio
AEC_GUARD_MIN_DB = -3.0
AEC_GUARD_FREEZE_BLOCKS = 300    # ~5 s before trying again
# ...doubling each consecutive time the answer is still "no", up to ~40 s. A
# robot with no usable coupling should spend less and less time re-learning
# that: each attempt costs one guard window of junk on the output.
AEC_GUARD_FREEZE_MAX_FACTOR = 8

# ── the shadow filter ────────────────────────────────────────────────────────
#
# A second filter, adapting with none of the double-talk protection, whose
# output is never published. Every window the two are compared on the error they
# would have produced, and the shadow replaces the foreground when it is clearly
# better. This is the standard two-filter arrangement, and it is here because
# the protection the foreground needs makes it unable to recover from anything:
#
#   the skip guard refuses an update whenever the error exceeds the microphone,
#   which is what double-talk looks like — and also what a *changed echo path*
#   looks like. So when the robot turned its head mid-session the filter was
#   locked to the old path forever: measured at -2.9 dB (worse than passing the
#   microphone through, because a stale filter adds a wrong estimate) and never
#   recovering, with 355 consecutive updates refused. The same run with the
#   guards disabled recovered to 17.4 dB within six seconds.
#
# Double-talk and a changed path cannot be separated by powers, which is the
# third time that has been true in this file. The shadow separates them by
# *outcome* instead: during double-talk it adapts to a person and gets worse, so
# it is not promoted; after a path change it learns the new path and gets
# better, so it is. Measured: path change -2.9 → 16.6 dB with one promotion;
# steady and double-talk unchanged, with none.
#
# It costs one extra filter's arithmetic — about 3.8 ms per second of audio on
# Orin 5, doubling this card's cost and still leaving it cheap.
AEC_SHADOW_WINDOW = 64        # far-end-active blocks per comparison, ~1 s
AEC_SHADOW_MARGIN_DB = 2.0    # how much better the shadow must be to be promoted

# ERLE is reported over roughly the last 1.5 s of robot speech (16 ms blocks, and
# only blocks where it *is* speaking — pauses freeze the figure rather than
# diluting it), once there has been half a second of it to average.
#
# 1.5 s and not 5 s: on Orin 5 a 5 s window still read 3.8 dB at the end of a
# session whose last settled phase removed 17 dB, because the robot had only
# talked for about one time constant in total and the average was still mostly
# the convergence period. A card that says 3.8 while removing 17 is not a
# diagnostic, it is a distraction.
ERLE_ALPHA = 1.0 / 94
ERLE_MIN_BLOCKS = 32


class _RefBuffer:
    """The reference signal on a time axis, so a block can fetch its own echo.

    Samples rather than chunks: the canceller needs an arbitrary window starting
    at an arbitrary time, and a list of chunks cannot answer that without
    stitching on every call. Follows the same contiguity rule as
    `_EnvelopeStream` — stamps detect gaps, they do not position every chunk —
    because the reference arrives in bursts whose stamps are nearly identical.
    """

    def __init__(self, seconds: float) -> None:
        self._capacity = max(1, int(seconds * SAMPLE_RATE))
        self._samples = np.zeros(0, dtype=np.int16)
        self._start_t: Optional[float] = None
        self._timeline = _ContiguousStream()

    def add(self, pcm: bytes, stamp: float) -> None:
        block = np.frombuffer(pcm, dtype='<i2')
        if block.size == 0:
            return
        start, _ = self._timeline.position(stamp)
        if self._start_t is None:
            self._start_t = start
            self._samples = block.copy()
        else:
            index = int(round((start - self._start_t) * SAMPLE_RATE))
            end = self._samples.size
            if index > end:
                # A hole means the speaker played nothing, so it is silence —
                # not a seam to be closed by sliding the next audio earlier.
                if index - end > self._capacity:
                    self._start_t, self._samples = start, block.copy()
                else:
                    self._samples = np.concatenate(
                        (self._samples, np.zeros(index - end, dtype=np.int16), block))
            elif index == end:
                self._samples = np.concatenate((self._samples, block))
            elif index >= 0:
                self._samples = np.concatenate((self._samples[:index], block))
            else:
                self._start_t, self._samples = start, block.copy()
        self._timeline.advance(start + block.size / SAMPLE_RATE)

        if self._samples.size > self._capacity:
            drop = self._samples.size - self._capacity
            self._samples = self._samples[drop:]
            self._start_t += drop / SAMPLE_RATE

    def fetch(self, start_t: float, count: int) -> np.ndarray:
        """`count` samples from `start_t`, zero-filled where there is nothing."""
        out = np.zeros(count, dtype=np.float32)
        if self._start_t is None or count <= 0:
            return out
        index = int(round((start_t - self._start_t) * SAMPLE_RATE))
        lo, hi = max(index, 0), min(index + count, self._samples.size)
        if hi > lo:
            out[lo - index:hi - index] = self._samples[lo:hi]
        return out

    def covers(self, start_t: float, count: int) -> bool:
        if self._start_t is None:
            return False
        index = int(round((start_t - self._start_t) * SAMPLE_RATE))
        return index >= 0 and index + count <= self._samples.size


class _EchoCanceller:
    """Partitioned-block frequency-domain NLMS, plus a soft gate on the residual.

    Written in numpy rather than bound to speexdsp's MDF on purpose. The binding
    on PyPI is an sdist needing swig and `libspeexdsp-dev` in a 16 GB image built
    for two JetPack lines and cp38, it exposes `process(near, far)` and nothing
    else — no delay handling, no access to the preprocessor's residual stage —
    and it would have to be hand-fed lock-step frames anyway. This is ~40 lines
    of arithmetic, runs on both lines with the numpy already in the image, can be
    tested against synthetic echo paths offline, and can be hot-copied to a rig.

    What it does not do is the hard part of a production AEC: nonlinear echo (a
    speaker driven into its limiter), and sample clock drift between the
    microphone and the speaker. Both need a machine with a speaker and a
    microphone in one room before there is any point writing code for them.
    """

    def __init__(self, tail_s: float = AEC_TAIL_S, mu: float = AEC_MU,
                 strength: str = DEFAULT_REMOVAL_STRENGTH) -> None:
        self.partitions = max(1, int(math.ceil(tail_s * SAMPLE_RATE / AEC_BLOCK)))
        self._mu = mu
        self._beta = SUPPRESSION_STRENGTHS.get(strength,
                                               SUPPRESSION_STRENGTHS[DEFAULT_REMOVAL_STRENGTH])
        self._weights = np.zeros((self.partitions, AEC_BINS), dtype=np.complex128)
        self._shadow = np.zeros((self.partitions, AEC_BINS), dtype=np.complex128)
        self._spectra = np.zeros((self.partitions, AEC_BINS), dtype=np.complex128)
        self._ref_prev = np.zeros(AEC_BLOCK, dtype=np.float32)
        self._power = np.full(AEC_BINS, AEC_EPS)
        self._gain = 1.0          # carried across blocks so the ramp is smooth
        self._blocks = 0
        self._adapted = 0
        self._skipped = 0
        # usefulness guard
        self._window_mic = 0.0
        self._window_err = 0.0
        self._window_blocks = 0
        self._frozen_for = 0
        self._benefit_db: Optional[float] = None
        self._resets = 0
        self._diverged_streak = 0
        # shadow comparison
        self._fg_error = 0.0
        self._sh_error = 0.0
        self._mic_power_sum = 0.0
        self._compare_blocks = 0
        self._promotions = 0
        # ERLE accounting: an exponentially-weighted average over blocks where
        # the reference was actually playing, not a lifetime one. A lifetime
        # average is the wrong statistic for something an operator reads off a
        # card — it never recovers from the seconds spent converging, and it
        # cannot show that the filter has *stopped* working. Measured on Orin 5:
        # a session whose settled phases removed 11.7 and 17.0 dB reported 3.8
        # dB as a lifetime figure.
        self._mic_energy = 0.0
        self._out_energy = 0.0
        self._erle_blocks = 0

    def set_strength(self, strength: str) -> None:
        """Change the suppressor without discarding a converged filter."""
        if strength in SUPPRESSION_STRENGTHS:
            self._beta = SUPPRESSION_STRENGTHS[strength]

    # ── one 16 ms block ──────────────────────────────────────────────────

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        window = np.concatenate((self._ref_prev, ref))
        self._ref_prev = ref
        spectrum = np.fft.rfft(window)
        self._spectra = np.roll(self._spectra, 1, axis=0)
        self._spectra[0] = spectrum

        estimate = np.fft.irfft((self._weights * self._spectra).sum(axis=0))[AEC_BLOCK:]
        error = mic - estimate

        mic_power = float(np.mean(np.square(mic)))
        ref_power = float(np.mean(np.square(ref)))
        error_power = float(np.mean(np.square(error)))
        far_active = ref_power > SILENCE_RMS ** 2

        # Echo-dominant: the far end is playing and what is left after
        # subtraction is well below the microphone, i.e. this block is not
        # mostly somebody talking. Both the full-rate update and the ERLE figure
        # are restricted to these blocks.
        echo_dominant = far_active and error_power <= 0.5 * mic_power
        # The reference power estimate is updated on every far-active block,
        # not only on blocks the foreground is allowed to learn from: it
        # describes the *reference*, which double-talk does not touch, and both
        # filters normalise by it.
        spectrum_error = None
        if far_active:
            self._power = (0.9 * self._power
                           + 0.1 * np.abs(self._spectra).__pow__(2).sum(axis=0))
            spectrum_error = np.fft.rfft(
                np.concatenate((np.zeros(AEC_BLOCK, dtype=np.float32), error)))

        if far_active and self._frozen_for <= 0:
            # Only adapt while the far end is playing: with no reference there
            # is no information in the gradient, and updating anyway is how a
            # converged filter is slowly destroyed by near-end speech.
            if error_power > mic_power:
                # Worse than doing nothing — double-talk, or divergence. Skip
                # rather than scale: this is the guard that keeps a bad second
                # from undoing a good minute.
                self._skipped += 1
            else:
                scale = 1.0
                if error_power > 0.5 * mic_power:
                    # Near-end speech probably present: slow down rather than
                    # stop, so a permanently noisy room still converges.
                    scale = 0.25
                step = (self._mu * scale) / (self._power + AEC_EPS)
                self._weights *= AEC_LEAK
                self._weights += step * np.conj(self._spectra) * spectrum_error
                self._adapted += 1

        if far_active:
            self._run_shadow(mic, mic_power, error_power, spectrum_error)

        out = self._suppress(error, estimate, far_active)
        self._blocks += 1
        if echo_dominant:
            # ERLE is measured on echo-dominant blocks only. Averaging
            # double-talk blocks into it would be the same mistake the
            # divergence guard made: while a person is talking the residual is
            # dominated by their voice, so ERLE there is near 0 dB and says
            # nothing about echo removal. On Orin 5 the last 0.4 s of a robot
            # utterance overlapping the next person's turn was enough to report
            # 4.3 dB for a filter that was removing 17.
            out_power = float(np.mean(np.square(out)))
            if self._erle_blocks == 0:
                self._mic_energy, self._out_energy = mic_power, out_power
            else:
                self._mic_energy += ERLE_ALPHA * (mic_power - self._mic_energy)
                self._out_energy += ERLE_ALPHA * (out_power - self._out_energy)
            self._erle_blocks += 1
        if far_active:
            # Divergence, in contrast, is judged over *every* far-active block —
            # harm has to be visible wherever it happens — and on the linear
            # error rather than the suppressed output, so a diverging filter
            # cannot hide behind the soft gate.
            self._check_divergence(mic_power, error_power)
        if self._frozen_for > 0:
            self._frozen_for -= 1
        return out

    def _run_shadow(self, mic: np.ndarray, mic_power: float, error_power: float,
                    spectrum_error: np.ndarray) -> None:
        """Adapt the unprotected filter, and promote it when it is clearly better."""
        shadow_estimate = np.fft.irfft(
            (self._shadow * self._spectra).sum(axis=0))[AEC_BLOCK:]
        shadow_error = mic - shadow_estimate
        shadow_spectrum = np.fft.rfft(
            np.concatenate((np.zeros(AEC_BLOCK, dtype=np.float32), shadow_error)))
        step = self._mu / (self._power + AEC_EPS)
        self._shadow *= AEC_LEAK
        self._shadow += step * np.conj(self._spectra) * shadow_spectrum

        self._fg_error += error_power
        self._sh_error += float(np.mean(np.square(shadow_error)))
        self._mic_power_sum += mic_power
        self._compare_blocks += 1
        if self._compare_blocks < AEC_SHADOW_WINDOW:
            return

        margin = 10.0 ** (-AEC_SHADOW_MARGIN_DB / 10.0)
        if self._sh_error < self._fg_error * margin:
            # Clearly better over a whole window — not a lucky block. Promoting
            # on anything less would let double-talk noise win a coin toss.
            self._weights = self._shadow.copy()
            self._promotions += 1
            self._frozen_for = 0
            self._diverged_streak = 0
        elif self._sh_error > self._mic_power_sum:
            # The shadow is worse than not filtering at all. Nothing it has
            # learned is worth keeping, and starting it over is what lets it
            # catch a path change that happened while it was lost.
            self._shadow[:] = 0.0
        self._fg_error = self._sh_error = self._mic_power_sum = 0.0
        self._compare_blocks = 0

    def _check_divergence(self, mic_power: float, error_power: float) -> None:
        self._window_mic += mic_power
        self._window_err += error_power
        self._window_blocks += 1
        if self._window_blocks < AEC_GUARD_BLOCKS:
            return
        benefit = 10.0 * math.log10((self._window_mic + AEC_EPS)
                                    / (self._window_err + AEC_EPS))
        self._window_mic = self._window_err = 0.0
        self._window_blocks = 0
        self._benefit_db = round(benefit, 1)
        if benefit >= AEC_GUARD_MIN_DB:
            self._diverged_streak = 0
            return
        # Diverging. Zeroing the weights is what makes this safe: with no
        # weights the estimate is zero, so the error is the microphone and the
        # suppressor's ratio is zero — the card becomes a passthrough rather
        # than a noise source.
        self._resets += 1
        self._diverged_streak += 1
        factor = min(AEC_GUARD_FREEZE_MAX_FACTOR, 2 ** (self._diverged_streak - 1))
        self._frozen_for = AEC_GUARD_FREEZE_BLOCKS * factor
        self._weights[:] = 0.0
        self._gain = 1.0

    def _suppress(self, error: np.ndarray, estimate: np.ndarray,
                  far_active: bool) -> np.ndarray:
        """Soft-gate the residual by how much echo was there to begin with.

        The gain is ramped across the block from the previous block's value, so
        a decision that changes every 16 ms cannot be heard as a click.
        """
        target = 1.0
        if far_active and self._beta > 0.0:
            residual = float(np.mean(np.square(error)))
            echo = float(np.mean(np.square(estimate)))
            if residual > AEC_EPS:
                # How much of what is left looks like echo rather than speech.
                ratio = math.sqrt(echo / (residual + AEC_EPS))
                target = max(SUPPRESSION_FLOOR, 1.0 - self._beta * min(ratio, 1.0))
        ramp = np.linspace(self._gain, target, error.size, dtype=np.float32)
        self._gain = target
        return error * ramp

    # ── reporting ────────────────────────────────────────────────────────

    @property
    def erle_db(self) -> Optional[float]:
        """Echo return loss enhancement over the last few seconds of robot speech."""
        if self._mic_energy <= 0.0 or self._erle_blocks < ERLE_MIN_BLOCKS:
            return None
        return round(10.0 * math.log10((self._mic_energy + AEC_EPS)
                                       / (self._out_energy + AEC_EPS)), 1)

    def report(self) -> Dict[str, Any]:
        return {
            "echo_removed_db": self.erle_db,
            "filter_tail_ms": round(self.partitions * AEC_BLOCK / SAMPLE_RATE * 1000),
            "aec_blocks": self._blocks,
            "aec_blocks_adapted": self._adapted,
            "aec_updates_skipped": self._skipped,
            # False means the filter was measured to be doing nothing and has
            # been zeroed — on this robot the microphone does not hear the
            # speaker enough to cancel, and `mute_while_talking` is the mode
            # that works. None means it has not been judged yet.
            "filter_benefit_db": self._benefit_db,
            "aec_filter_resets": self._resets,
            "aec_shadow_promotions": self._promotions,
        }


# ── MCP tool declaration ─────────────────────────────────────────────────────

# Field names and titles are English and the choices are enumerated, because the
# people who fill this in are operators wiring a canvas, not DSP engineers.
# Every field has to be answerable from something they can hear; anything that
# can only be answered with a measurement is reported by `info()` instead, and
# the one field that genuinely needs a number (`mute_tail`) defaults to `auto`
# so that it follows that measurement until somebody overrides it.
#
# Deliberately *not* offered: frame size, filter length, adaptation step size,
# double-talk thresholds. Those either derive from the fields below or stay
# internal constants. A card with sixteen knobs is a card nobody dares touch.
TOOLS = [
    {
        "name": "echo_cancellation",
        "type": "processor",
        "multiInstance": True,
        "description": (
            "回声消除 — 把机器人自己说话的声音从麦克风里拿掉，否则 ASR 会把机器人"
            "自己说的话当成别人说的。接在麦克风卡和 ASR 卡之间"
            "（Echo cancellation: keeps the robot's own voice out of its microphone）"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform",
                },
                "input_topic": {
                    "type": "string",
                    "description": "Microphone AudioChunk topic (required for start)",
                },
                # `instance_id` is deliberately NOT declared, though every call
                # carries one. The canvas renders each inputSchema property it
                # does not recognise as a manual input box on the card — it only
                # hides `input_topic` (by name pattern) and an `action` whose
                # values are all system actions — so declaring it put an
                # "INSTANCE_ID" text field and an 执行 button on the face of the
                # card, asking an operator to type a canvas id that the canvas
                # fills in itself (canvas.js, "Inject instance_id from card
                # identity"). No other plugin here declares it either.
            },
            "required": ["action"],
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "robot_voice_topic": {
                    "type": "string",
                    "title": "Where the robot's own voice comes from",
                    "default": DEFAULT_ROBOT_VOICE_TOPIC,
                    "scope": "instance",
                    # `oneOf` is filled in per request by get_tools() with the
                    # topics the TTS card is actually publishing right now, so
                    # this is a list of real topics rather than a string to
                    # type. A topic set in config.yaml is still honoured even if
                    # it is not in the list — that is the escape hatch for a TTS
                    # running in another container.
                    "oneOf": [
                        {"const": REF_AUTO,
                         "title": "Automatic — follow the TTS card on this robot"},
                    ],
                    "description": (
                        "The audio the TTS card publishes — the same sound the "
                        "speaker plays. 'Automatic' is right unless this robot's "
                        "speech never passes through Agent Core (vendor TTS), in "
                        "which case this card cannot tell when it is talking."
                    ),
                },
            },
        },
        "topic_in": [{"format": AUDIO_FORMAT, "desc": "microphone audio"}],
        "topic_out": [{"format": AUDIO_FORMAT,
                       "desc": "microphone audio, robot's own voice removed"}],
    }
]


# ── ROS2 node ────────────────────────────────────────────────────────────────

class _EchoCancellationNode(Node):
    """One microphone in, the same microphone cleaned out.

    Processing happens in the microphone callback rather than on a worker
    thread, and that is a deliberate choice for this version: the work is a
    comparison and a memset, the delay estimate is a 300x150 dot product once a
    second, and a queue plus a thread would add latency to the audio path and a
    second lifecycle to get wrong. It stops being the right choice as soon as an
    adaptive filter lands here — revisit it then, with the CPU figure this card
    reports.

    Lifecycle follows the five rules in perception/README.md § Plugin
    Concurrency: the plugin registers this node before starting it, `stop`
    signals cancellation before taking the lock, and the subscriptions outlive
    `stop()` so the executor's wait set is never torn out from under it.
    """

    def __init__(self, input_topic: str, ref_topic: str = '', *,
                 node_suffix: str = '', mute_tail: str = DEFAULT_MUTE_TAIL,
                 removal_strength: str = DEFAULT_REMOVAL_STRENGTH) -> None:
        super().__init__(f"echo_cancellation_{node_suffix}" if node_suffix
                         else "echo_cancellation")
        self._input_topic = input_topic or ''
        self._output_topic = f"{self._input_topic}{OUTPUT_SUFFIX}" if self._input_topic else ''
        # A reference topic equal to the microphone topic would make the card
        # correlate the mic against itself and mute permanently at delay 0.
        self._ref_topic = ref_topic if ref_topic and ref_topic != self._input_topic else ''
        if ref_topic and not self._ref_topic:
            log.warning("[echo_cancellation] ignoring robot_voice_topic %r: it is the "
                        "microphone topic", ref_topic)

        self._mute_tail = mute_tail if mute_tail in MUTE_TAIL_CHOICES else DEFAULT_MUTE_TAIL
        self._removal_strength = (removal_strength if removal_strength in SUPPRESSION_STRENGTHS
                                  else DEFAULT_REMOVAL_STRENGTH)

        self._node_lock = threading.RLock()
        self._retired = False
        self.state = "idle"

        self._gate_lock = threading.Lock()
        self._mute_until = 0.0          # mic-clock time the gate may reopen
        self._last_ref_audio_t = 0.0    # publisher stamp of the last ref chunk
        self._ref_seen = False
        self._last_eof_t = 0.0

        self._tracker = _DelayTracker()
        self._tracker_lock = threading.Lock()

        # Cancellation state. The reference ring is kept whatever the mode is,
        # so switching into remove_echo does not start from an empty buffer.
        self._aec_lock = threading.Lock()
        self._aec = _EchoCanceller(strength=self._removal_strength)
        self._ref_buffer = _RefBuffer(DELAY_SEARCH_MAX_S + 1.0)
        self._mic_timeline = _ContiguousStream()
        self._pending = b''        # mic samples not yet a whole block
        self._pending_t = 0.0      # time of the first pending sample
        self._out = b''            # cancelled samples not yet published
        self._out_t = 0.0

        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "mic_chunks": 0, "mic_bytes": 0, "published": 0, "muted_chunks": 0,
            "muted_bytes": 0, "ref_chunks": 0, "ref_eof": 0, "mic_gaps": 0,
            "format_violations": 0, "misaligned_chunks": 0, "publish_errors": 0,
            "cancel_waiting_for_delay": 0, "cancel_resyncs": 0,
            "cancel_coupling_lost": 0,
        }
        self._last_mic_t = 0.0
        self._process_s = 0.0   # processing time, for the CPU figure
        self._audio_s = 0.0     # audio time it was spent on

        self._pub = None
        self._sub = None
        self._ref_sub = None

        log.info("[echo_cancellation] node created: mic=%s -> %s, robot_voice=%s, "
                 "strength=%s",
                 self._input_topic or "(none)", self._output_topic or "(none)",
                 self._ref_topic or "(none)", self._removal_strength)

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> Dict[str, Any]:
        with self._node_lock:
            if self._retired:
                # A concurrent stop destroyed this node between the plugin
                # registering it and this call.
                return self.status()
            if self.state == "running":
                return self.status()
            if not self._input_topic:
                # Idle, **not** error. A card on the canvas with nothing wired
                # into it yet is an ordinary state here — tts, ocr and barcode
                # all support a topic-less card — and reporting `error` has a
                # consequence far beyond this card: agent-core's start-project
                # rolls the *whole project* back when any card returns it
                # (`api/config.py`, "有 card 失败 → 全部回滚"). Measured on Orin 5:
                # a second, unwired echo_cancellation card made every other card
                # — OCR, face, TTS, ASR, and the wired copy of this one — start
                # correctly and then stop one second later, with the reason
                # visible only in the start-project progress events.
                return {"state": "idle",
                        "message": "No microphone connected — draw a line from "
                                   "the mic card to this one"}
            from audio_msgs.msg import AudioChunk
            if self._pub is None:
                self._pub = self.create_publisher(AudioChunk, self._output_topic, _AUDIO_QOS)
            if self._sub is None:
                self._sub = self.create_subscription(
                    AudioChunk, self._input_topic, self._mic_cb, _AUDIO_QOS)
            if self._ref_topic and self._ref_sub is None:
                self._ref_sub = self.create_subscription(
                    AudioChunk, self._ref_topic, self._ref_cb, _AUDIO_QOS)
            self.state = "running"
            log.info("[echo_cancellation] started: %s -> %s",
                     self._input_topic, self._output_topic)
            return self.status()

    def request_retire(self) -> None:
        """Non-blocking cancellation, callable before any lock is taken."""
        self._retired = True

    def retire(self) -> Dict[str, Any]:
        self.request_retire()
        return self.stop()

    def stop(self) -> Dict[str, Any]:
        # The subscriptions are left in place for dispose_node() to release with
        # the node: destroying them here can race the executor's wait set and
        # raise InvalidHandle, which has taken the whole ROS layer down before
        # (perception/README.md).
        with self._node_lock:
            self.state = "idle"
        with self._gate_lock:
            self._mute_until = 0.0
        log.info("[echo_cancellation] stopped: %s", self._input_topic)
        return {"state": "idle"}

    def apply_config(self, mute_tail: Optional[str] = None,
                     removal_strength: Optional[str] = None) -> None:
        """Change the live knobs. Neither needs the node rebuilt.

        `robot_voice_topic` is not here on purpose — a subscription cannot be
        recreated on a node the executor is spinning (it wedges: the callback
        never fires again while the graph and the executor both look healthy),
        so the plugin disposes the node and builds a new one instead.
        """
        with self._node_lock:
            if mute_tail in MUTE_TAIL_CHOICES:
                self._mute_tail = mute_tail
            if removal_strength in SUPPRESSION_STRENGTHS:
                self._removal_strength = removal_strength
                with self._aec_lock:
                    # Live, without discarding the converged filter: the
                    # suppressor sits after it and shares no state with it.
                    self._aec.set_strength(removal_strength)

    # ── the gate ─────────────────────────────────────────────────────────

    def _tail_s(self) -> float:
        explicit = MUTE_TAIL_CHOICES.get(self._mute_tail)
        if explicit is not None:
            return explicit
        with self._tracker_lock:
            measured = self._tracker.measured
        if measured is None:
            return AUTO_TAIL_FALLBACK_S
        return measured + AUTO_TAIL_MARGIN_S

    def _ref_cb(self, message: Any) -> None:
        if self.state != "running" or self._retired:
            return
        try:
            pcm = bytes(message.data)
        except (TypeError, ValueError):
            return
        now = time.time()
        timestamp = _chunk_timestamp(message)

        if pcm == AUDIO_EOF_MAGIC:
            # End of the utterance's *publishing*, not of the sound: the speaker
            # path still holds several hundred ms. Recorded for info(), and
            # deliberately not used to open the gate.
            with self._gate_lock:
                self._last_eof_t = timestamp
            with self._stats_lock:
                self._stats["ref_eof"] += 1
            return

        if not pcm or len(pcm) % BYTES_PER_SAMPLE:
            return

        with self._stats_lock:
            self._stats["ref_chunks"] += 1

        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        envelope = _envelope(pcm)
        talking = bool(envelope.size) and float(envelope.max()) > SILENCE_RMS

        with self._tracker_lock:
            self._tracker.add_ref(pcm, timestamp, now)
            if talking:
                self._tracker.note_far_active(duration)
        with self._aec_lock:
            self._ref_buffer.add(pcm, timestamp)

        with self._gate_lock:
            self._ref_seen = True
            if talking:
                self._last_ref_audio_t = timestamp
                # Extend rather than overwrite: reference chunks arrive in a
                # burst of five at the start of an utterance (TTS pre-buffer),
                # so an overwrite would shorten the window the burst just set.
                self._mute_until = max(self._mute_until,
                                       timestamp + duration + self._tail_s())

    # ── the audio path ───────────────────────────────────────────────────

    def _mic_cb(self, message: Any) -> None:
        if self.state != "running" or self._retired:
            return
        started = time.perf_counter()

        fmt = getattr(message, 'format', '') or ''
        if fmt not in _AUDIO_FORMATS:
            with self._stats_lock:
                self._stats["format_violations"] += 1
        try:
            pcm = bytes(message.data)
        except (TypeError, ValueError):
            return
        if not pcm:
            return
        if len(pcm) % BYTES_PER_SAMPLE:
            # Truncate rather than drop, as ASR does: an odd byte count breaks
            # int16 framing, and dropping would change the behaviour of drivers
            # that work today.
            pcm = pcm[:len(pcm) // BYTES_PER_SAMPLE * BYTES_PER_SAMPLE]
            with self._stats_lock:
                self._stats["misaligned_chunks"] += 1
            if not pcm:
                return

        now = time.time()
        timestamp = _chunk_timestamp(message)
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

        # A gap in the microphone stream is not the same thing as a change in
        # delay, and conflating them is how a correlator ends up chasing a
        # phantom: count it so a busy bus is visible in info().
        with self._stats_lock:
            self._stats["mic_chunks"] += 1
            self._stats["mic_bytes"] += len(pcm)
            if self._last_mic_t and timestamp - self._last_mic_t > 1.5 * duration:
                self._stats["mic_gaps"] += 1
            self._last_mic_t = timestamp

        # Fed from the raw microphone, before anything is subtracted from it or
        # silenced in it.
        with self._tracker_lock:
            self._tracker.add_mic(pcm, timestamp, now)
            if self._tracker.due(now):
                self._tracker.estimate(now)

        self._cancel(pcm, message, timestamp)

        with self._stats_lock:
            self._process_s += time.perf_counter() - started
            self._audio_s += duration

    def _cancel(self, pcm: bytes, source: Any, timestamp: float) -> None:
        """Subtract the robot's own voice, in 16 ms blocks.

        Two things make this different from the gate path. The reference is
        fetched by *time* — aligned using the measured delay, so the filter only
        models the echo's spread and not the speaker's several hundred ms of
        buffering — and the output is reframed, because 16 ms blocks do not
        divide an arbitrary chunk size. Reframing means the audio published with
        a given chunk is up to one block older than the chunk that triggered it,
        so it carries **its own** stamp rather than the input's. ASR reads that
        stamp as the capture time; mislabelling it by 16 ms would be a small lie
        told on every chunk.
        """
        with self._tracker_lock:
            measured = self._tracker.measured
            fresh = self._tracker.fresh()
        if measured is None or not fresh:
            # Nothing to align to, and the two reasons get different treatment.
            #
            # `measured is None` is the warmup — the first seconds of the
            # robot's first utterance. Here the microphone is held silent while
            # the robot is audible: subtracting an unaligned reference would add
            # noise, and passing it through means ASR transcribes the robot,
            # which is the bug this card exists to remove. Bounded, once per
            # session, and the card says what it is doing.
            #
            # `not fresh` is 15 s of the robot talking with no measurable echo
            # since: the microphone has stopped hearing the speaker. There is
            # nothing to remove, so the audio passes through. Muting here would
            # be a permanent half-duplex microphone, which is the one behaviour
            # this card must not have.
            muted = False
            if measured is None:
                with self._gate_lock:
                    muted = timestamp < self._mute_until
            with self._stats_lock:
                self._stats["cancel_waiting_for_delay"] += 1
                if measured is not None:
                    self._stats["cancel_coupling_lost"] += 1
            self._publish(b'\x00' * len(pcm) if muted else pcm, source, muted)
            return

        align = max(0.0, measured - AEC_ALIGN_BACKOFF_S)
        start_t, resynced = self._mic_timeline.position(timestamp)
        with self._aec_lock:
            if resynced or not self._pending:
                if resynced and self._pending:
                    with self._stats_lock:
                        self._stats["cancel_resyncs"] += 1
                self._pending = pcm
                self._pending_t = start_t
            else:
                self._pending += pcm
            self._mic_timeline.advance(start_t + len(pcm)
                                       / (SAMPLE_RATE * BYTES_PER_SAMPLE))

            block_bytes = AEC_BLOCK * BYTES_PER_SAMPLE
            produced = bytearray()
            produced_t = self._pending_t
            while len(self._pending) >= block_bytes:
                block = np.frombuffer(self._pending[:block_bytes],
                                      dtype='<i2').astype(np.float32)
                reference = self._ref_buffer.fetch(self._pending_t - align, AEC_BLOCK)
                cleaned = self._aec.process(block, reference)
                produced += np.clip(cleaned, -32768, 32767).astype('<i2').tobytes()
                self._pending = self._pending[block_bytes:]
                self._pending_t += AEC_BLOCK / SAMPLE_RATE

            if produced:
                if not self._out:
                    self._out_t = produced_t
                self._out += bytes(produced)

            if len(self._out) < len(pcm):
                return                      # not a whole chunk's worth yet
            chunk, self._out = self._out[:len(pcm)], self._out[len(pcm):]
            stamp = self._out_t
            self._out_t += len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

        self._publish(chunk, source, False, stamp=stamp)

    def _publish(self, pcm: bytes, source: Any, muted: bool,
                 stamp: Optional[float] = None) -> None:
        publisher = self._pub
        if publisher is None:
            return
        from audio_msgs.msg import AudioChunk
        out = AudioChunk()
        # The *input's* stamp, not now(): downstream reads it as the capture
        # time (ASR's audio_start_ts comes from it), so re-stamping would charge
        # this card's own latency to the speaker's timeline.
        if stamp is None:
            out.header.stamp = source.header.stamp
        else:
            # Reframed audio describes its own capture time, not the time of the
            # chunk that happened to trigger it.
            out.header.stamp.sec = int(stamp)
            out.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
        out.format = AUDIO_FORMAT
        # `array.array('B')` and not `list(pcm)`, which is what the rest of the
        # project writes. The generated `data` setter validates a list
        # element-by-element — `all(isinstance(v, int) and 0 <= v < 256 ...)`,
        # 1024 iterations per chunk — and takes a buffer fast path for an array
        # of matching typecode. Measured on Orin 5: 0.226 ms vs 0.021 ms per
        # chunk, which was the single largest cost in this callback. The same
        # saving is available to plugins/tts.py and the mic drivers.
        out.data = array.array('B', pcm)
        try:
            publisher.publish(out)
        except Exception as error:  # noqa: BLE001 - a dead publisher must not kill the callback
            with self._stats_lock:
                self._stats["publish_errors"] += 1
            log.warning("[echo_cancellation] publish failed on %s: %s",
                        self._output_topic, error)
            return
        with self._stats_lock:
            self._stats["published"] += 1
            if muted:
                self._stats["muted_chunks"] += 1
                self._stats["muted_bytes"] += len(pcm)

    # ── reporting ────────────────────────────────────────────────────────

    def topics(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if not self._input_topic:
            return [], []
        return ([{"topic": self._input_topic, "format": AUDIO_FORMAT,
                  "desc": "microphone"}],
                [{"topic": self._output_topic, "format": AUDIO_FORMAT,
                  "desc": "microphone, robot's own voice removed"}])

    def diagnostics(self) -> Dict[str, Any]:
        """The numbers an operator needs to tell "working" from "wired wrong"."""
        now = time.time()
        with self._stats_lock:
            stats = dict(self._stats)
            process_s, audio_s = self._process_s, self._audio_s
        with self._gate_lock:
            ref_seen = self._ref_seen
            last_ref = self._last_ref_audio_t
            muting = now < self._mute_until
        with self._tracker_lock:
            report = self._tracker.report()
            measured = self._tracker.measured
            fresh = self._tracker.fresh()

        published = stats["published"] or 1
        tail = self._tail_s()
        with self._aec_lock:
            aec_report = self._aec.report()
        result: Dict[str, Any] = {
            "robot_voice_topic": self._ref_topic or None,
            "robot_voice_seen": ref_seen,
            "last_robot_voice_s_ago": (None if not last_ref
                                       else round(now - last_ref, 2)),
            "muting_now": muting,
            "mute_tail": self._mute_tail,
            "mute_tail_effective_ms": round(tail * 1000),
            "muted_fraction": round(stats["muted_chunks"] / published, 4),
            "mic_chunks": stats["mic_chunks"],
            "published_chunks": stats["published"],
            "mic_gaps": stats["mic_gaps"],
            "format_violations": stats["format_violations"],
            "misaligned_chunks": stats["misaligned_chunks"],
            "publish_errors": stats["publish_errors"],
            "ref_chunks": stats["ref_chunks"],
            "ref_utterances_ended": stats["ref_eof"],
            # Wall time spent inside the microphone callback per second of
            # audio, in ms. **Not** CPU time, and the difference is not small:
            # nine other plugins' threads contend for the GIL in this process,
            # so this inflates when the box is busy. Measured on Orin 5 with a
            # reference playing: 29.7 ms/s live against 3.6 ms/s for the same
            # callback profiled in isolation. Read it as a headroom indicator
            # and a regression tripwire, not as a CPU share — and profile in
            # isolation before concluding anything about cost.
            "callback_wall_ms_per_audio_s": (None if audio_s <= 0
                                             else round(process_s / audio_s * 1000, 2)),
            **report,
            **aec_report,
            "removal_strength": self._removal_strength,
            "waiting_for_delay": (measured is None),
            "coupling_lost": (measured is not None) and not fresh,
            "cancel_chunks_without_a_delay": stats["cancel_waiting_for_delay"],
            "cancel_resyncs": stats["cancel_resyncs"],
            "cancel_chunks_after_coupling_lost": stats["cancel_coupling_lost"],
        }
        # The one comparison nobody can do in their head, spelled out: an
        # explicit tail shorter than the delay the card measured means the end
        # of the robot's own sentence reaches ASR.
        if measured is not None and MUTE_TAIL_CHOICES.get(self._mute_tail) is not None:
            result["mute_tail_shorter_than_measured_delay"] = tail < measured
        return result

    def status(self) -> Dict[str, Any]:
        topic_in, topic_out = self.topics()
        return {
            "state": self.state,
            "topic_in": topic_in,
            "topic_out": topic_out,
            "diagnostics": self.diagnostics(),
        }


# ── Plugin ───────────────────────────────────────────────────────────────────

_DEFAULT_INSTANCE = "_default"


class EchoCancellationPlugin:
    """Perception facade for the echo cancellation card.

    Holds no model and loads nothing, so unlike the vision cards it has no
    `loading` state and the constructor cannot fail on a missing file. What it
    does hold is the resolution of the reference topic, which is the only part
    of this card an operator can get wrong.

    `tts` is the TTSPlugin living in this same process, handed in by main.py the
    way ASR is handed the speaker-recognition plugin. It is optional: without it
    `auto` resolves to nothing and the card says so.
    """

    PREFIX = "echo_cancellation"
    _DESC = "Keeps the robot's own voice out of its microphone"

    def __init__(self, plugin_cfg: Mapping[str, Any], executor: Any,
                 tts: Any = None) -> None:
        self._plugin_cfg: Dict[str, Any] = dict(plugin_cfg or {})
        self._executor = executor
        self._tts = tts
        self._nodes: Dict[str, _EchoCancellationNode] = {}
        self._instance_configs: Dict[str, Dict[str, Any]] = {}
        self._state_lock = threading.RLock()
        log.info("[echo_cancellation] plugin init: robot_voice_topic=%s, "
                 "strength=%s, mute_tail=%s, tts_handle=%s",
                 self._plugin_cfg.get("robot_voice_topic", DEFAULT_ROBOT_VOICE_TOPIC),
                 self._plugin_cfg.get("removal_strength", DEFAULT_REMOVAL_STRENGTH),
                 self._plugin_cfg.get("mute_tail", DEFAULT_MUTE_TAIL),
                 "yes" if tts is not None else "no")

    # ── the reference topic ──────────────────────────────────────────────

    def _tts_audio_topics(self) -> List[str]:
        """Audio topics the TTS card is publishing, asked of the plugin itself.

        Goes through the same `info` action the dashboard uses rather than
        reaching into TTSPlugin's internals: that plugin is a facade over a
        swappable engine, and its node dict lives inside whichever engine is
        currently loaded. Never raises — a TTS in a bad state must not stop this
        card from answering `tools/list`.
        """
        if self._tts is None:
            return []
        try:
            info = self._tts.dispatch("tts", {"action": "info"}) or {}
            topics = [str(entry.get("topic") or "")
                      for entry in (info.get("topic_out") or [])
                      if (entry.get("format") or "") in _AUDIO_FORMATS]
        except Exception as error:  # noqa: BLE001
            log.debug("[echo_cancellation] could not ask TTS for its topics: %s", error)
            return []
        seen: List[str] = []
        for topic in topics:
            if topic and topic not in seen:
                seen.append(topic)
        return seen

    def _resolve_ref(self, cfg: Mapping[str, Any]) -> Tuple[str, str]:
        """(topic, how) for the configured reference setting.

        `how` is reported verbatim in `info()`, because "automatic found the TTS
        card" and "automatic found nothing" are the same empty string downstream
        and only the second one is a misconfiguration.
        """
        configured = str(cfg.get("robot_voice_topic") or DEFAULT_ROBOT_VOICE_TOPIC).strip()
        if configured and configured != REF_AUTO:
            return configured, "configured"
        topics = self._tts_audio_topics()
        if not topics:
            return "", "auto:none"
        if len(topics) > 1:
            # Several TTS cards: picking one silently would be a coin flip in a
            # place nobody would look. Take the first and say which.
            log.info("[echo_cancellation] auto found %d TTS audio topics, using %s",
                     len(topics), topics[0])
        return topics[0], "auto"

    def get_tools(self) -> List[Dict[str, Any]]:
        """TOOLS with the reference-topic dropdown filled from live TTS topics.

        Rebuilt per call because `main.py get_all_tools` does not cache, so the
        dashboard sees the topics that exist at the moment it asks. A deep copy:
        mutating the module-level TOOLS would accumulate stale topics for the
        life of the process.
        """
        tools = copy.deepcopy(TOOLS)
        options = tools[0]["configSchema"]["properties"]["robot_voice_topic"]["oneOf"]
        configured = str(self._plugin_cfg.get("robot_voice_topic") or "").strip()
        known = {REF_AUTO}
        for topic in self._tts_audio_topics():
            if topic not in known:
                options.append({"const": topic, "title": topic})
                known.add(topic)
        if configured and configured not in known:
            # A topic set in config.yaml or by an earlier `config` call stays
            # selectable even when its publisher is currently down; otherwise
            # the form would silently reset it to `auto`.
            options.append({"const": configured, "title": f"{configured} (configured)"})
        return tools

    # ── MCP dispatch ─────────────────────────────────────────────────────

    def dispatch(self, name: str, args: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        action = str(args.get("action", name))
        instance_id = str(args.get("instance_id", "") or "")
        input_topic = str(args.get("input_topic", "") or "")

        if action == "info":
            return self._do_info(instance_id, input_topic)
        if action == "start":
            return self._do_start(instance_id, input_topic)
        if action == "stop":
            return self._do_stop(instance_id)
        if action == "config":
            return self._do_config(instance_id, args)
        return None

    # ── config ───────────────────────────────────────────────────────────

    def _merged_cfg(self, node_key: str = "") -> Dict[str, Any]:
        with self._state_lock:
            return {**self._plugin_cfg, **self._instance_configs.get(node_key, {})}

    def _do_config(self, instance_id: str, args: Mapping[str, Any]) -> Dict[str, Any]:
        cfg = {k: v for k, v in args.items()
               if k not in ("action", "instance_id", "input_topic")
               and v is not None and v != ""}

        with self._state_lock:
            if instance_id:
                previous = self._instance_configs.get(instance_id, {})
                self._instance_configs[instance_id] = {**previous, **cfg}
                keys = [instance_id]
            else:
                self._plugin_cfg = {**self._plugin_cfg, **cfg}
                keys = list(self._nodes)
            targets = [(key, self._nodes.get(key)) for key in keys]

        rebuild: List[Tuple[str, str]] = []
        for key, node in targets:
            if node is None:
                continue
            merged = self._merged_cfg(key)
            node.apply_config(mute_tail=merged.get("mute_tail"),
                              removal_strength=merged.get("removal_strength"))
            wanted, _ = self._resolve_ref(merged)
            if wanted != node._ref_topic:
                # A subscription cannot be recreated on a spinning node — it
                # wedges silently (perception/README.md § Plugin Concurrency),
                # so the node is replaced rather than re-subscribed.
                rebuild.append((key, node._input_topic))

        for key, input_topic in rebuild:
            log.info("[echo_cancellation] robot_voice_topic changed, rebuilding %s", key)
            self._do_stop(key)
            self._do_start(key, input_topic)

        result = {"status": "configured"}
        if instance_id:
            result["instance_id"] = instance_id
        return result

    # ── start / stop ─────────────────────────────────────────────────────

    def _create_node(self, node_key: str, input_topic: str) -> _EchoCancellationNode:
        cfg = self._merged_cfg(node_key)
        ref_topic, _ = self._resolve_ref(cfg)
        return _EchoCancellationNode(
            input_topic,
            ref_topic,
            node_suffix=node_key.replace('/', '_').replace('-', '_').strip('_'),
            mute_tail=str(cfg.get("mute_tail") or DEFAULT_MUTE_TAIL),
            removal_strength=str(cfg.get("removal_strength")
                                 or DEFAULT_REMOVAL_STRENGTH),
        )

    def _dispose(self, node_key: str, node: _EchoCancellationNode) -> None:
        try:
            node.retire()
        finally:
            dispose_node(self._executor, node, label=f"echo_cancellation/{node_key}")
        log.info("[echo_cancellation] node disposed: %s", node_key)

    def _do_start(self, instance_id: str, input_topic: str) -> Dict[str, Any]:
        node_key = instance_id or input_topic or _DEFAULT_INSTANCE

        retired = None
        with self._state_lock:
            existing = self._nodes.get(node_key)
            if existing is not None and existing._input_topic != input_topic:
                # Topics are fixed at construction; rebind the key.
                retired = self._nodes.pop(node_key)
        if retired is not None:
            self._dispose(node_key, retired)

        with self._state_lock:
            existing = self._nodes.get(node_key)
        if existing is not None:
            return existing.start()        # idempotent re-start

        node = self._create_node(node_key, input_topic)
        # Registered under the lock *before* being started, so a concurrent stop
        # can always find it. A started-but-unregistered node keeps a live
        # subscription and a publisher nothing can reach for the life of the
        # process.
        with self._state_lock:
            current = self._nodes.get(node_key)
            registered = False
            if current is None:
                try:
                    self._executor.add_node(node)
                except Exception as error:  # noqa: BLE001
                    log.error("[echo_cancellation] failed to register instance %r: %s",
                              node_key, error)
                else:
                    self._nodes[node_key] = node
                    registered = True
        if not registered:
            try:
                node.destroy_node()   # never started, never registered
            except Exception:  # noqa: BLE001
                pass
            if current is not None:
                return current.start()
            return {"state": "idle", "input": input_topic}
        try:
            result = node.start()
        except Exception:
            with self._state_lock:
                if self._nodes.get(node_key) is node:
                    del self._nodes[node_key]
            self._dispose(node_key, node)
            raise
        with self._state_lock:
            still_ours = self._nodes.get(node_key) is node
        if not still_ours:
            # A concurrent stop retired it while start() ran; its dispose did
            # the teardown — make sure this start left nothing running.
            node.stop()
            return {"state": "idle", "input": input_topic}
        return result

    def _do_stop(self, instance_id: str) -> Dict[str, Any]:
        to_dispose: List[Tuple[str, _EchoCancellationNode]] = []
        with self._state_lock:
            if instance_id:
                node = self._nodes.pop(instance_id, None)
                if node is not None:
                    to_dispose.append((instance_id, node))
            else:
                to_dispose.extend(self._nodes.items())
                self._nodes = {}
        for node_key, node in to_dispose:
            self._dispose(node_key, node)
        return {"state": "idle"}

    # ── info ─────────────────────────────────────────────────────────────

    def _desc(self, node: Optional[_EchoCancellationNode], how: str) -> str:
        """The card's one-line face. Carries the misconfiguration, when there is one.

        `auto` resolving to nothing is the failure this card can suffer silently:
        it keeps publishing, the audio is simply untouched. Saying so here puts
        it on the card in the dashboard rather than only in a diagnostics field
        nobody opens.
        """
        if node is None:
            return self._DESC
        if not node._input_topic:
            return "Not connected to a microphone — draw a line from the mic card"
        if not node._ref_topic:
            if how == "auto:none":
                return ("No TTS audio to listen for — this robot's speech does not "
                        "pass through Agent Core, so the mic is passed through "
                        "unchanged")
            return "No robot voice topic set — microphone passed through unchanged"
        diagnostics = node.diagnostics()
        if not diagnostics.get("robot_voice_seen"):
            return (f"Waiting for audio on {node._ref_topic} — nothing heard "
                    f"yet, so nothing is being removed")
        if diagnostics.get("coupling_lost"):
            return ("This microphone has stopped hearing the speaker, so there "
                    "is nothing to remove — passing audio through. Check the "
                    "speaker volume")
        if diagnostics.get("waiting_for_delay"):
            return ("Measuring how late the robot's voice comes back — the mic "
                    "is held silent for the first few seconds of its speech")
        removed = diagnostics.get("echo_removed_db")
        if removed is not None:
            return f"Removing {removed} dB of the robot's own voice"
        return self._DESC

    def _do_info(self, instance_id: str, input_topic: str) -> Dict[str, Any]:
        base = {"name": "EchoCancellation", "manufacture": "Embodied",
                "model": "aec"}
        with self._state_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            keys = list(self._nodes)
            nodes = [self._nodes[key] for key in keys]

        if instance_id:
            if node is not None:
                _, how = self._resolve_ref(self._merged_cfg(instance_id))
                return {**base, **node.status(), "desc": self._desc(node, how)}
            topic_in = ([{"topic": input_topic, "format": AUDIO_FORMAT, "desc": ""}]
                        if input_topic else [])
            topic_out = ([{"topic": f"{input_topic}{OUTPUT_SUFFIX}",
                           "format": AUDIO_FORMAT, "desc": ""}]
                         if input_topic else [])
            return {**base, "state": "idle", "desc": self._DESC,
                    "topic_in": topic_in, "topic_out": topic_out}

        topics_in: List[Dict[str, str]] = []
        topics_out: List[Dict[str, str]] = []
        for item in nodes:
            item_in, item_out = item.topics()
            topics_in.extend(item_in)
            topics_out.extend(item_out)
        if not nodes and input_topic:
            topics_in = [{"topic": input_topic, "format": AUDIO_FORMAT, "desc": ""}]
            topics_out = [{"topic": f"{input_topic}{OUTPUT_SUFFIX}",
                           "format": AUDIO_FORMAT, "desc": ""}]

        state = "running" if any(item.state == "running" for item in nodes) else "idle"
        if any(item.state == "error" for item in nodes):
            state = "error"
        result: Dict[str, Any] = {
            **base,
            "state": state,
            "desc": self._DESC,
            "topic_in": topics_in,
            "topic_out": topics_out,
        }
        if nodes:
            result["instances"] = {key: {"state": self._nodes[key].state}
                                   for key in keys if key in self._nodes}
            # Report the instance that is actually doing the work, not whichever
            # happens to be first in the dict. A canvas routinely holds a second
            # card nobody has wired up yet, and on Orin 5 the summary was coming
            # from that one: it truthfully said "no microphone, no reference
            # heard" while the wired card beside it was running — which reads as
            # the working card being broken.
            index = next((i for i, item in enumerate(nodes)
                          if item.state == "running" and item._input_topic),
                         0)
            chosen, chosen_key = nodes[index], keys[index]
            _, how = self._resolve_ref(self._merged_cfg(chosen_key))
            result["desc"] = self._desc(chosen, how)
            result["diagnostics"] = chosen.diagnostics()
            # Which one these numbers describe. Without it a summary that picks
            # a different instance than last time is indistinguishable from the
            # same instance changing its mind.
            result["diagnostics"]["instance_id"] = chosen_key
        return result


__all__ = ["EchoCancellationPlugin", "TOOLS", "AUDIO_FORMAT", "OUTPUT_SUFFIX"]
