#!/usr/bin/env python3
"""
plugins/kokoro_worker.py — run Kokoro's Japanese graph in a process of its own.

Perception is one process holding two ONNX Runtimes: sherpa-onnx bundles its own, and
the standalone `onnxruntime` wheel serves face recognition and this module. They cannot
both build a CUDA session on the Kokoro graph, and the reason is in the dynamic linker
rather than in either library:

`libonnxruntime_providers_shared.so` is an 8 KB library exporting exactly
`Provider_GetHost` and `Provider_SetHost` — a process-global slot holding **one** pointer
to **one** runtime's `ProviderHost`. It carries a SONAME, and ld.so deduplicates a dlopen
by matching the requested basename against already-loaded objects, so the second runtime's
copy is never mapped: both get the first one. Each runtime writes its own host into that
slot as it loads a provider, last writer wins, and the next session built runs against the
other runtime's framework objects:

    Error mapping output names: Could not find OrtValue with name '/Squeeze_2_output_0'

Measured on Orin 6 in both orders, with `Provider_GetHost()` observed changing value and
only ever one bridge mapped. On jp5.11 the same collision is a **SIGSEGV that kills the
whole perception process**, taking ASR, VOP, OCR and face with it.

Renaming the bridge and the CUDA provider in a private ORT build does fix it — verified —
but it forks ONNX Runtime's ABI and needs a second build per JetPack line. A process
boundary gets the same result with none of that, and buys the GPU as well.

**This module no longer owns a process.** It owns the *policy* for Japanese — which
device is safe, whether the duration came out right, and when to give the memory back —
while `plugins/ort_worker.py` owns the one child that holds every standalone-ORT session
in this process, face's included. Two children would mean two CUDA contexts; the point of
moving out of the parent was to keep the count where it already was, not to raise it.

So encoding, the style-table lookup and chunking all run in the parent, as they always
did, and only `session.run()` crosses: tokens and a 1 kB style row in, the waveform out.

Measured on Orin 6, one 9.3 s Japanese utterance, 121 tokens:

    in-process, cpu (what this replaces)   RTF 0.525
    worker, cuda                           RTF 0.063 warm, 0.861 on the first call

    memory, whole-box MemAvailable
      in-process cpu session                -820 MB
      worker holding a cuda session        -1192 MB      net cost ~+372 MB

The first CUDA call costs 8.03 s of lazy kernel loading, so the child is warmed during
startup and kept alive; a per-utterance child would be far worse than the CPU path. It is
reaped after `idle_timeout_s` without work, which also fixes a leak this module inherits:
`KokoroDirect` was assigned once and never released, so a card that spoke Japanese once
kept the session for the adapter's whole life even after switching back to English.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
import uuid

import numpy as np

log = logging.getLogger(__name__)

# Generous enough for a cold CUDA session (2.5 s build + 8.0 s first inference measured
# on Orin 6) on a box that is also serving ASR and video.
START_TIMEOUT_S = 45.0
# One utterance is 0.59 s warm. This is a "the child is wedged" bound, not a budget.
CALL_TIMEOUT_S = 30.0
# Long enough that a bilingual tour switching languages does not pay the ~10.5 s cold
# path repeatedly; short enough that a card that has finished with Japanese gives the
# memory back. Card stop and engine switch close it immediately regardless.
IDLE_TIMEOUT_S = 120.0

# The warmup probe doubles as a correctness gate on the chosen device, because one
# JetPack line executes this graph wrong on CUDA.
#
# The graph is stochastic — 4 RandomNormalLike and 7 RandomUniformLike nodes — so the
# *waveform* differs run to run and between providers by design, and "cpu output must
# equal cuda output" is not a valid check. The **duration** is not stochastic: measured
# 8 runs per provider, CPU gives one distinct length with stdev exactly 0, and gives the
# same length on both JetPack lines. That makes duration a portable reference.
#
#   probe "konnichiwa", 10 tokens, speaker 0, speed 1.0:
#     jp6.1   cpu  47400, 47400, 47400      cuda 47400, 47400, 47400   <- correct
#     jp5.11  cpu  47400, 47400, 47400      cuda 16800, 21000, 21000   <- 35-44%
#
#   the full sentence, 8 runs each:
#     jp6.1   cpu 8.35s stdev 0.000   cuda 8.35s stdev 0.000
#     jp5.11  cpu 8.35s stdev 0.000   cuda 6.05s stdev 0.053   <- 27.5% short
#
# jp5.11's ONNX Runtime is 1.15.1 and its CUDA execution of the duration path is simply
# wrong; 27% short is audibly rushed speech. Gating on the version number would be the
# wrong fix — it would not catch the next line with the same defect — so gate on the
# measurement instead. The probe already runs as warmup, so this costs nothing.
PROBE_TEXT = "konnichiwa"
PROBE_SAMPLES = 47400
# Wide enough that a legitimately different build is not rejected, far tighter than the
# 56% error it has to catch.
PROBE_TOLERANCE = 0.10

# Two guards are needed, and the order matters: the duration check above runs *after* the
# session is built, so it cannot prevent the allocation that builds it from taking the
# box down. Measured on jp5.11, asking for a CUDA child:
#
#   5237 MB available -> sherpa's own Kokoro GPU adapter takes 3.2 GB -> 2048 MB left
#   -> the CUDA child's allocation -> whole-box OOM, SIGKILL, before the probe ran
#
# (`dmesg`: "Out of memory: Killed process ... (python3) anon-rss:2420028kB".) The same
# adapter costs ~950 MB on jp6.1, where a CUDA child fits in the 1585 MB it needs. So the
# headroom figure is the child's measured cost plus a margin, checked before spawning.
#
# On jp5.11 this lands on cpu, which is also where the duration check would have put it.
# Two independent reasons, one outcome — and the memory one has to be first because it is
# the one that can kill the process.
CUDA_HEADROOM_MB = 2500


def _mem_available_mb() -> int:
    """Whole-box MemAvailable. -1 if unreadable, which must not block the GPU."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


class KokoroWorkerProxy:
    """`KokoroDirect`'s surface, with its ONNX session in the shared ORT worker.

    Constructed by `plugins/tts.py:_direct()`; its interface is deliberately unchanged
    from when this class owned a child process, so nothing in tts.py had to move.

    Falls back to an in-process CPU `KokoroDirect` if the worker cannot serve it. That
    fallback is what shipped before any of this existed, so a worker failure costs speed
    and nothing else — Japanese must not break because a subprocess did.
    """

    def __init__(self, model_dir: str, weights: str, device: str = "gpu",
                 num_threads: int = 0, idle_timeout_s: float = IDLE_TIMEOUT_S):
        self._model_dir = model_dir
        self._weights = weights
        self._device = device
        self._num_threads = num_threads
        self._idle_timeout_s = float(idle_timeout_s)

        self._lock = threading.RLock()
        self._direct = None
        self._device_used = device
        self._last_used = time.monotonic()
        self._closed = False
        self._fallback = None
        self._fallback_warned = False

        self._build()

        self._reaper = threading.Thread(target=self._reap_loop, daemon=True,
                                        name="kokoro-session-reaper")
        self._reaper.start()
        atexit.register(self.close)

    # ── properties mirroring KokoroDirect ────────────────────────────────────

    @property
    def num_speakers(self) -> int:
        with self._lock:
            return (self._direct or self._use_fallback()).num_speakers

    @property
    def sample_rate(self) -> int:
        with self._lock:
            return (self._direct or self._use_fallback()).sample_rate

    @property
    def providers(self) -> list:
        """What the session actually got. `['CPUExecutionProvider']` after asking for
        gpu means a guard declined it — worth seeing in `info` rather than assuming."""
        with self._lock:
            if self._direct is None:
                return []
            return list(self._direct._session.get_providers())

    @property
    def device_used(self) -> str:
        return self._device_used

    # ── build, with both guards ──────────────────────────────────────────────

    def _build(self) -> None:
        """Load the session in the shared worker, on a device that is actually safe.

        Two guards, and the order matters. The duration check runs *after* the session
        is built, so it cannot prevent the allocation that builds it from taking the box
        down — the headroom check has to come first.
        """
        from plugins.kokoro_direct import KokoroDirect

        device = self._device
        if device in ("cuda", "gpu"):
            headroom = _mem_available_mb()
            if 0 <= headroom < CUDA_HEADROOM_MB:
                log.warning("[tts] kokoro: %d MB available, under the %d MB a CUDA "
                            "session needs — using cpu (RTF ~0.52 rather than ~0.07). "
                            "Asking anyway OOM-killed a jp5.11 rig.",
                            headroom, CUDA_HEADROOM_MB)
                device = "cpu"

        started = time.monotonic()
        for candidate, why in ((device, "requested"), ("cpu", "fallback")):
            if why == "fallback" and candidate == device:
                break                       # already tried it, and it failed the check
            direct = KokoroDirect(self._model_dir, self._weights,
                                  num_threads=self._num_threads, provider=candidate,
                                  session_key="tts.kokoro.ja")
            # The warmup was always needed — the first CUDA call costs 8.03 s of lazy
            # kernel loading against 0.59 s warm — so checking its output length is
            # free. A wrong length means this device computes the duration path
            # incorrectly, and a fast wrong answer is worse than a slow right one.
            n = len(direct.synthesize(PROBE_TEXT, speaker_id=0, speed=1.0))
            off = abs(n - PROBE_SAMPLES) / PROBE_SAMPLES
            if off <= PROBE_TOLERANCE:
                self._direct = direct
                self._device_used = candidate
                log.info("[tts] kokoro session ready in %.1fs on %s%s: providers=%s",
                         time.monotonic() - started, candidate,
                         "" if candidate == self._device
                         else f" (asked for {self._device})",
                         direct._session.get_providers())
                return
            log.error("[tts] %s produced %d samples for the probe, expected ~%d "
                      "(%.0f%% off) — this device computes the duration path wrongly; "
                      "not using it", candidate, n, PROBE_SAMPLES, off * 100)
            direct.close()
        raise RuntimeError(
            f"neither {self._device} nor cpu produced a plausible probe duration; the "
            f"model or the phoneme table does not match this code")

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Unload the session, giving its weights and GPU pool back.

        Unloads rather than killing the child: face's sessions live in the same process
        and must survive Japanese being done with. Called on card stop, engine switch,
        idle expiry and at exit; safe to call more than once.
        """
        with self._lock:
            self._closed = True
            direct, self._direct = self._direct, None
            self._fallback = None
        if direct is not None:
            try:
                direct.close()
            except Exception as exc:                              # noqa: BLE001
                log.warning("[tts] unloading the kokoro session failed: %s", exc)

    def _reap_loop(self) -> None:
        """Give the memory back when Japanese stops being used.

        Not on `set_language`, deliberately: the cold path is ~15 s measured end to end
        and a tour that alternates languages would pay it on every switch. Idle time is
        the signal that actually means "done with Japanese".
        """
        while True:
            time.sleep(5.0)
            direct = None
            with self._lock:
                if self._closed:
                    return
                idle = time.monotonic() - self._last_used
                if self._direct is not None and idle > self._idle_timeout_s:
                    log.info("[tts] kokoro session idle %.0fs, unloading", idle)
                    direct, self._direct = self._direct, None
            # Unload outside the lock: it is a round trip to the child, and holding the
            # lock across it would stall an utterance that arrived meanwhile.
            if direct is not None:
                try:
                    direct.close()
                except Exception as exc:                          # noqa: BLE001
                    log.warning("[tts] idle unload failed: %s", exc)

    # ── the one call ─────────────────────────────────────────────────────────

    def synthesize(self, phonemes: str, speaker_id: int = 0, speed: float = 1.0):
        with self._lock:
            self._last_used = time.monotonic()
            if self._closed:
                return self._use_fallback().synthesize(phonemes, speaker_id, speed)
            if self._direct is None:
                try:
                    self._build()
                except Exception as exc:                          # noqa: BLE001
                    log.warning("[tts] kokoro session could not be rebuilt (%s); "
                                "Japanese falls back to the in-process CPU session",
                                exc)
                    return self._use_fallback().synthesize(phonemes, speaker_id, speed)
            try:
                return self._direct.synthesize(phonemes, speaker_id=speaker_id,
                                               speed=speed)
            except Exception as exc:                              # noqa: BLE001
                log.warning("[tts] kokoro synthesis in the worker failed (%s); "
                            "falling back to the in-process CPU session", exc)
                self._direct = None
                return self._use_fallback().synthesize(phonemes, speaker_id, speed)

    def _use_fallback(self):
        """The in-process CPU session — what shipped before any of this existed."""
        if self._fallback is None:
            from plugins.kokoro_direct import KokoroDirect
            self._fallback = KokoroDirect(self._model_dir, self._weights,
                                          num_threads=self._num_threads,
                                          in_process=True)
        if not self._fallback_warned:
            self._fallback_warned = True
            log.warning("[tts] Japanese is using the in-process CPU session "
                        "(RTF ~0.52 against ~0.07 in the worker)")
        return self._fallback
