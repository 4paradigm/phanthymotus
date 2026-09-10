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
    """`KokoroDirect`'s surface, executed in a spawned child.

    Only `synthesize` is forwarded — the phoneme string goes in (~121 characters) and
    float32 samples come back (~91 KB). Resampling, framing, pacing and EOF all stay in
    the parent, so the boundary sits where the data is smallest and `plugins/tts.py`
    does not change.

    Falls back to an in-process CPU `KokoroDirect` if the child cannot be started or
    dies. That fallback is exactly what shipped before this module existed, so a worker
    failure costs speed and nothing else — Japanese must not break because a subprocess
    did.
    """

    def __init__(self, model_dir: str, weights: str, device: str = "gpu",
                 num_threads: int = 0, idle_timeout_s: float = IDLE_TIMEOUT_S):
        self._model_dir = model_dir
        self._weights = weights
        self._device = device
        self._num_threads = num_threads
        self._idle_timeout_s = float(idle_timeout_s)

        self._lock = threading.RLock()
        self._ctx = None
        self._proc = None
        self._cmd_q = None
        self._res_q = None
        self._last_used = time.monotonic()
        self._closed = False
        self._fallback = None
        self._fallback_warned = False

        # Filled by the child's handshake; the properties need them before any call.
        self._n_speakers = 0
        self._sample_rate = 0
        self._providers = []
        self._device_used = device

        self._start()

        self._reaper = threading.Thread(target=self._reap_loop, daemon=True,
                                        name="kokoro-worker-reaper")
        self._reaper.start()
        atexit.register(self.close)

    # ── properties mirroring KokoroDirect ────────────────────────────────────

    @property
    def num_speakers(self) -> int:
        if self._n_speakers:
            return self._n_speakers
        return self._use_fallback().num_speakers

    @property
    def sample_rate(self) -> int:
        if self._sample_rate:
            return self._sample_rate
        return self._use_fallback().sample_rate

    @property
    def providers(self) -> list:
        """What the child actually got. `['CPUExecutionProvider']` means the GPU was
        asked for and refused — worth seeing in `info` rather than assuming."""
        return list(self._providers)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _start(self) -> None:
        """Spawn the child and wait for it to report a warmed session.

        `spawn`, never the platform default: on Linux that is `fork`, which copies the
        address space including already-dlopened libraries. A forked child would
        inherit sherpa's ONNX Runtime and the isolation this module exists for would be
        worthless.
        """
        import multiprocessing as mp

        device = self._device
        if device in ("cuda", "gpu"):
            headroom = _mem_available_mb()
            if 0 <= headroom < CUDA_HEADROOM_MB:
                log.warning("[tts] kokoro worker: %d MB available, under the %d MB a "
                            "CUDA child needs — using cpu (RTF ~0.52 rather than "
                            "~0.07). Asking anyway OOM-killed a jp5.11 rig.",
                            headroom, CUDA_HEADROOM_MB)
                device = "cpu"

        self._ctx = mp.get_context("spawn")
        self._cmd_q = self._ctx.Queue()
        self._res_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_kokoro_worker,
            args=(self._model_dir, self._weights, device, self._num_threads,
                  self._cmd_q, self._res_q, log.getEffectiveLevel()),
            daemon=False,
            name="kokoro_ja_worker",
        )
        started = time.monotonic()
        self._proc.start()

        try:
            kind, payload = self._res_q.get(timeout=START_TIMEOUT_S)
        except queue.Empty:
            self._terminate()
            raise RuntimeError(
                f"Kokoro worker did not report ready within {START_TIMEOUT_S:.0f}s")
        if kind != "ready":
            self._terminate()
            raise RuntimeError(f"Kokoro worker failed to start: {payload}")

        self._n_speakers = int(payload["num_speakers"])
        self._sample_rate = int(payload["sample_rate"])
        self._providers = list(payload["providers"])
        self._device_used = payload.get("device_used", self._device)
        log.info("[tts] kokoro worker ready in %.1fs: pid=%s device=%s%s providers=%s, "
                 "%d speakers, warmup %.2fs",
                 time.monotonic() - started, self._proc.pid, self._device_used,
                 "" if self._device_used == self._device
                 else f" (asked for {self._device})",
                 self._providers, self._n_speakers, payload["warmup_s"])

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _terminate(self) -> None:
        proc, self._proc = self._proc, None
        self._cmd_q = self._res_q = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        except Exception:                                        # noqa: BLE001
            pass

    def close(self) -> None:
        """Kill the child and release its CUDA context.

        A process exit is the only thing that returns a CUDA context at all — an
        in-process session never does. Called on card stop, engine switch, idle
        expiry and at exit; safe to call more than once.
        """
        with self._lock:
            if self._closed and self._proc is None:
                return
            self._closed = True
            if self._proc is not None:
                log.info("[tts] kokoro worker closing (pid=%s)", self._proc.pid)
            self._terminate()
            self._fallback = None

    def _reap_loop(self) -> None:
        """Give the memory back when Japanese stops being used.

        Not on `set_language`, deliberately: the cold path is ~10.5 s and a tour that
        alternates languages would pay it on every switch. Idle time is the signal that
        actually means "done with Japanese".
        """
        while True:
            time.sleep(5.0)
            with self._lock:
                if self._closed:
                    return
                idle = time.monotonic() - self._last_used
                if self._alive() and idle > self._idle_timeout_s:
                    log.info("[tts] kokoro worker idle %.0fs, reaping (pid=%s)",
                             idle, self._proc.pid)
                    self._terminate()

    # ── the one forwarded call ───────────────────────────────────────────────

    def synthesize(self, phonemes: str, speaker_id: int = 0, speed: float = 1.0):
        with self._lock:
            self._last_used = time.monotonic()
            if self._closed:
                return self._use_fallback().synthesize(phonemes, speaker_id, speed)
            if not self._alive():
                try:
                    self._start()
                except Exception as exc:                          # noqa: BLE001
                    log.warning("[tts] kokoro worker could not be restarted (%s); "
                                "Japanese falls back to the in-process CPU session",
                                exc)
                    return self._use_fallback().synthesize(phonemes, speaker_id, speed)

            req_id = uuid.uuid4().hex[:8]
            try:
                self._cmd_q.put(("synthesize", req_id, phonemes, speaker_id, speed))
                kind, payload = self._recv(req_id)
            except Exception as exc:                              # noqa: BLE001
                log.warning("[tts] kokoro worker call failed (%s); falling back to "
                            "the in-process CPU session", exc)
                self._terminate()
                return self._use_fallback().synthesize(phonemes, speaker_id, speed)

            if kind == "error":
                # The child is fine, the request was not — surface it rather than
                # silently re-synthesizing somewhere else and hiding a real bug.
                raise RuntimeError(f"kokoro worker: {payload}")
            return payload

    def _recv(self, want_id: str):
        """Read until this request's answer, tolerating a late reply to an earlier one.

        Calls are serialised by `self._lock`, so out-of-order traffic means a previous
        call timed out and its result arrived afterwards. Dropping it is right; blocking
        on it would deadlock the next utterance.
        """
        deadline = time.monotonic() + CALL_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no reply within {CALL_TIMEOUT_S:.0f}s (child alive="
                    f"{self._alive()})")
            kind, req_id, payload = self._res_q.get(timeout=remaining)
            if req_id == want_id:
                return kind, payload
            log.debug("[tts] kokoro worker: dropping stale reply %s", req_id)

    def _use_fallback(self):
        """The in-process CPU session — what shipped before this module existed."""
        if self._fallback is None:
            from plugins.kokoro_direct import KokoroDirect
            self._fallback = KokoroDirect(self._model_dir, self._weights,
                                          num_threads=self._num_threads)
        if not self._fallback_warned:
            self._fallback_warned = True
            log.warning("[tts] Japanese is using the in-process CPU session "
                        "(RTF ~0.52 against ~0.06 on the worker)")
        return self._fallback


def _build_checked(KokoroDirect, model_dir, weights, device, num_threads, wlog):
    """Build on `device`, warm it up, and check the warmup's duration is right.

    The warmup was always needed — the first CUDA call costs 8.03 s of lazy kernel
    loading against 0.59 s warm — so measuring its output length is free. If the length
    is wrong the device is executing the duration path incorrectly (jp5.11's ORT 1.15.1
    returns 35-44% of it), and CPU is the only honest answer: 27% short is audibly
    rushed speech, and a fast wrong answer is worse than a slow right one.

    Returns `(runtime, device_actually_used)`.
    """
    for candidate, why in ((device, "requested"), ("cpu", "fallback")):
        if why == "fallback" and candidate == device:
            break                       # already tried, and it failed the check
        direct = KokoroDirect(model_dir, weights, num_threads=num_threads,
                              provider=candidate)
        n = len(direct.synthesize(PROBE_TEXT, speaker_id=0, speed=1.0))
        off = abs(n - PROBE_SAMPLES) / PROBE_SAMPLES
        if off <= PROBE_TOLERANCE:
            if why == "fallback":
                wlog.warning("running on cpu after %r failed the duration check", device)
            return direct, candidate
        wlog.error("%s produced %d samples for the probe, expected ~%d (%.0f%% off) — "
                   "this device computes the duration path wrongly; not using it",
                   candidate, n, PROBE_SAMPLES, off * 100)
        del direct
    raise RuntimeError(
        f"neither {device} nor cpu produced a plausible probe duration; the model or "
        f"the phoneme table does not match this code")


def _kokoro_worker(model_dir: str, weights: str, device: str, num_threads: int,
                   cmd_q, res_q, log_level: int) -> None:
    """Child entry point. Owns one `KokoroDirect` and nothing else.

    Deliberately imports neither `rclpy` nor `sherpa_onnx`: this process exists to be
    the only ONNX Runtime in its address space, and the parent keeps the ROS node, the
    publisher and the pacing.
    """
    # A spawned child gets a fresh interpreter and does not inherit the parent's
    # sys.stdout, so the atomic writer has to be reinstalled — without it a subprocess
    # writing on the parent's fd 1 tears Docker log records.
    try:
        from utils import logsafe
        logsafe.install(check_fd=False)
    except Exception:                                             # noqa: BLE001
        pass

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
        datefmt='%H:%M:%S')
    wlog = logging.getLogger("tts.kokoro_worker")

    try:
        from plugins.kokoro_direct import KokoroDirect
        started = time.monotonic()
        direct, device_used = _build_checked(KokoroDirect, model_dir, weights, device,
                                             num_threads, wlog)
        res_q.put(("ready", {
            "num_speakers": direct.num_speakers,
            "sample_rate": direct.sample_rate,
            "providers": list(direct._session.get_providers()),
            "device_used": device_used,
            "warmup_s": round(time.monotonic() - started, 2),
        }))
    except Exception as exc:                                      # noqa: BLE001
        res_q.put(("failed", f"{type(exc).__name__}: {exc}"))
        return

    while True:
        try:
            command = cmd_q.get()
        except (EOFError, OSError):
            return
        if command is None or command[0] == "stop":
            return
        if command[0] != "synthesize":
            wlog.warning("unknown command %r", command[0])
            continue
        _, req_id, phonemes, speaker_id, speed = command
        try:
            samples = direct.synthesize(phonemes, speaker_id=speaker_id, speed=speed)
            res_q.put(("ok", req_id, np.asarray(samples, dtype=np.float32)))
        except Exception as exc:                                  # noqa: BLE001
            res_q.put(("error", req_id, f"{type(exc).__name__}: {exc}"))
