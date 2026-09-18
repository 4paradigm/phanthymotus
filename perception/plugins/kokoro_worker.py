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
# Idle unloading is **off**, and that is a correction rather than a default.
#
# It was added to "give the memory back when Japanese stops being used", and then
# measured: unloading moved whole-box MemAvailable by **+0 MB**. The CUDA pool is
# returned on process exit, not on session destruction, so the only thing it achieved
# was making the next utterance pay the cold path again — observed on Orin 6 as
# "kokoro session ready in 7.4s" after a 120 s gap, which is what a tour with pauses in
# it feels as the TTS being slow.
#
# Card stop and engine switch still release it: those are the points where the answer
# to "is this needed again" is actually known.
#
# Set a positive number to re-enable it, but know what it does and does not buy.
IDLE_TIMEOUT_S = 0.0

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
# **Confirmed by ear**: jp5.11's CUDA renders this graph as garbled speech, while
# jp6.1's matches its own CPU exactly. The cause is unknown, and five candidates have
# been ruled out by measurement — the two-runtime collision, the model file, TF32, the
# ONNX Runtime version, and the CUDA provider binary. The record is in
# perception/docs/jp511-cuda-kokoro.md.
#
# Which is why the gate is a **measurement** and not a version check. A version check
# would have been wrong twice: written against 1.15.1 it lets the official 1.16.0
# through, and 1.16.0 produces the same garbled audio. The probe already runs as
# warmup, so it costs nothing.
PROBE_TEXT = "konnichiwa"
PROBE_SAMPLES = 47400
# Wide enough that a legitimately different build is not rejected, far tighter than the
# 56% error it has to catch.
PROBE_TOLERANCE = 0.10

# Two guards are needed, and the order matters: the duration check above runs *after*
# the session is built, so it cannot prevent the allocation that builds it from taking
# the box down. Measured on jp5.11, asking for a CUDA session with no headroom check:
#
#   5237 MB available -> sherpa's own Kokoro GPU adapter takes 3.2 GB -> 2048 MB left
#   -> the CUDA allocation -> whole-box OOM, SIGKILL, before the probe ran
#
# (`dmesg`: "Out of memory: Killed process ... (python3) anon-rss:2420028kB".)
#
# The figure depends on whether the ORT worker already holds a CUDA session, because
# the first one in that process pays for the context and the rest do not. Measured on
# Orin 6:
#
#   Kokoro alone in the child, bringing its own context       1585 MB
#   Kokoro beside the face service, context already there      967 MB
#
# A single conservative number would refuse the second case on the first case's
# evidence — which it did: with 2158 MB free on an otherwise idle box, a 967 MB
# allocation was declined because the threshold was 2500.
CUDA_HEADROOM_FRESH_MB = 2500
CUDA_HEADROOM_SHARED_MB = 1400


def _cuda_verdict_path(model_dir: str) -> str:
    """Where this machine records that CUDA failed the duration check.

    Persistent, not in-memory, because the process that learns it may not be the one
    that needs it: a crash restarts perception and an in-memory verdict is lost, so the
    next start tries CUDA again — which is exactly what happened on Orin 5, where the
    retry contributed to a whole-box OOM.

    Keyed by the ONNX Runtime version, because the defect is in that build. A new
    runtime gets a fresh evaluation rather than inheriting a verdict about a different
    one.
    """
    return os.path.join(model_dir, ".cuda-duration-verdict")


def _cuda_known_bad(model_dir: str, ort_version: str) -> bool:
    try:
        with open(_cuda_verdict_path(model_dir), encoding="utf-8") as handle:
            return handle.read().strip() == ort_version
    except OSError:
        return False


def _remember_cuda_is_bad(model_dir: str, ort_version: str) -> None:
    """Record the verdict so the attempt is made once per machine, not once per start.

    The attempt is not free, which is why this exists. A CUDA session that then gets
    rejected still leaves its pool behind — measured at ~967 MB that `unload` does not
    return — so retrying it on every start is a permanent cost for a known answer, and
    on a full box it is what tips the machine into the OOM killer.
    """
    try:
        with open(_cuda_verdict_path(model_dir), "w", encoding="utf-8") as handle:
            handle.write(ort_version)
    except OSError as exc:
        log.warning("[tts] could not record the CUDA verdict (%s); it will be "
                    "re-evaluated on the next start", exc)


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


class DeviceUnavailable(RuntimeError):
    """The configured device cannot be used, and substituting one silently is worse.

    Distinct from every other failure on purpose. A worker that dies, a queue that
    times out, a child that will not spawn — those are infrastructure, and falling back
    to an in-process CPU session keeps Japanese working at a cost of speed, which is
    the right trade. **This** is the operator having asked for a device that will not
    do the job, and quietly giving them a different one produces the worst state there
    is: a card configured for gpu, running at RTF 0.52, with the reason in a log line
    nobody reads.

    So this one propagates: the card goes `state: error` carrying the message, and the
    operator decides.
    """


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
        """Load the session on the device asked for, or fail saying why.

        **No silent substitution.** An earlier version quietly used the CPU when the
        GPU was unavailable, which produced the worst possible state: a card configured
        for `gpu`, running at RTF 0.52 instead of 0.07, with the reason in a log line
        nobody reads. The card now goes `state: error` and names which of the two
        problems it hit, so the operator can set `japanese_worker_device: cpu` knowing
        what they are accepting.

        The two are independent and either one alone is disqualifying:

        - **not enough memory.** A CUDA session needs ~967 MB beside an existing
          context and ~1585 MB bringing its own. Asking anyway OOM-killed a jp5.11 rig.
        - **wrong durations.** On jp5.11 the CUDA path does not agree with its own
          CPU: the long sentence came back 22% short and the *short* one 54% long, and
          three renders of one input gave 1.4 s, 1.9 s and 2.95 s. Confirmed by ear as
          garbled. Not a version problem — NVIDIA's official 1.16.0 behaves the same as
          the 1.15.1 that ships — and nothing to do with memory.
        """
        from plugins.kokoro_direct import KokoroDirect

        want_cuda = self._device in ("cuda", "gpu")
        ort_version = "unknown"
        try:
            import onnxruntime as _ort
            ort_version = _ort.__version__
        except Exception:                                         # noqa: BLE001
            pass

        if want_cuda:
            # Known-bad first: the attempt is not free. A CUDA session that then gets
            # rejected still leaves its pool behind — measured at 3.8 GB on Orin 5 that
            # unloading does not return — so a machine that has already answered this
            # question must not pay again on every restart.
            if _cuda_known_bad(self._model_dir, ort_version):
                raise DeviceUnavailable(
                    f"this machine's onnxruntime {ort_version} computes Kokoro's "
                    f"duration path wrongly on CUDA — measured once and recorded in "
                    f"{_cuda_verdict_path(self._model_dir)}. Japanese would be 27-51% "
                    f"too short, which is audibly rushed speech. Not a memory problem. "
                    f"Set japanese_worker_device: cpu (RTF ~0.52, still real time)."
                )

            from plugins import ort_worker
            shared = False
            try:
                shared = ort_worker.get_worker().has_cuda_session()
            except Exception as exc:                              # noqa: BLE001
                log.debug("[tts] could not ask the worker about CUDA (%s); assuming a "
                          "context has to be paid for", exc)
            needed = CUDA_HEADROOM_SHARED_MB if shared else CUDA_HEADROOM_FRESH_MB
            headroom = _mem_available_mb()
            if 0 <= headroom < needed:
                raise DeviceUnavailable(
                    f"not enough memory for a CUDA session: {headroom} MB available, "
                    f"{needed} MB needed{' beside the existing context' if shared else ' including a context of its own'}. "
                    f"This is a **memory** problem, not a model problem — the GPU "
                    f"itself is fine. Free memory on the box, or set "
                    f"japanese_worker_device: cpu (RTF ~0.52, still real time). "
                    f"Asking anyway OOM-killed a jp5.11 rig mid-utterance."
                )

        started = time.monotonic()
        direct = KokoroDirect(self._model_dir, self._weights,
                              num_threads=self._num_threads, provider=self._device,
                              session_key="tts.kokoro.ja")
        # The warmup was always needed — the first CUDA call costs 8 s of lazy kernel
        # loading against 0.6 s warm — so checking its output length is free.
        n = len(direct.synthesize(PROBE_TEXT, speaker_id=0, speed=1.0))
        off = abs(n - PROBE_SAMPLES) / PROBE_SAMPLES
        if off > PROBE_TOLERANCE:
            direct.close()
            if want_cuda:
                _remember_cuda_is_bad(self._model_dir, ort_version)
            raise DeviceUnavailable(
                f"{self._device} produced {n} samples for the probe, expected "
                f"~{PROBE_SAMPLES} ({off * 100:.0f}% off): this device computes "
                f"Kokoro's duration path wrongly, which comes out as audibly rushed "
                f"speech. Not a memory problem. "
                + ("Set japanese_worker_device: cpu (RTF ~0.52, still real time)."
                   if want_cuda else
                   "The model or the phoneme table does not match this code.")
            )

        self._direct = direct
        self._device_used = self._device
        log.info("[tts] kokoro session ready in %.1fs on %s: providers=%s",
                 time.monotonic() - started, self._device,
                 direct._session.get_providers())

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
        if self._idle_timeout_s <= 0:
            return                      # off; see IDLE_TIMEOUT_S for why that is default
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
        pieces = list(self.synthesize_stream(phonemes, speaker_id, speed))
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    def synthesize_stream(self, phonemes: str, speaker_id: int = 0, speed: float = 1.0,
                          max_chunk_tokens: int | None = None):
        """Like `synthesize`, but yields each chunk as `KokoroDirect` computes it.

        `max_chunk_tokens` is passed straight through to `KokoroDirect` — see its
        docstring. Applies equally to the worker and the in-process fallback, so
        switching devices mid-utterance (the failure path below) does not change
        how finely the rest of the utterance is chunked.

        Falls back to the in-process CPU session on the same conditions
        `synthesize` used to, but only up to the first chunk: once a chunk from
        the worker has already reached the caller, restarting on the fallback
        would either replay it (an audible repeat) or skip it (a dropped chunk).
        A failure after that point is left to surface like any other
        mid-utterance synthesis failure — the caller already plays whatever
        arrived before the error.
        """
        with self._lock:
            self._last_used = time.monotonic()
            if self._closed:
                yield from self._use_fallback().synthesize_stream(
                    phonemes, speaker_id, speed, max_chunk_tokens)
                return
            if self._direct is None:
                try:
                    self._build()
                except DeviceUnavailable:
                    # Not ours to paper over — the operator asked for a device that
                    # cannot do the job, and the card should say so.
                    raise
                except Exception as exc:                          # noqa: BLE001
                    log.warning("[tts] kokoro session could not be rebuilt (%s); "
                                "Japanese falls back to the in-process CPU session",
                                exc)
                    yield from self._use_fallback().synthesize_stream(
                        phonemes, speaker_id, speed, max_chunk_tokens)
                    return
            started = False
            try:
                for chunk in self._direct.synthesize_stream(
                        phonemes, speaker_id=speaker_id, speed=speed,
                        max_chunk_tokens=max_chunk_tokens):
                    started = True
                    yield chunk
            except Exception as exc:                              # noqa: BLE001
                if started:
                    raise
                log.warning("[tts] kokoro synthesis in the worker failed (%s); "
                            "falling back to the in-process CPU session", exc)
                self._direct = None
                yield from self._use_fallback().synthesize_stream(
                    phonemes, speaker_id, speed, max_chunk_tokens)

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
