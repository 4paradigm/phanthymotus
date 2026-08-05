"""Instance-local acoustic echo cancellation with timestamped TTS references.

The timing model follows the proven voice-service path: TTS PCM is sliced into
fixed frames carrying their expected playback wall-clock timestamp. Microphone
frames consume the matching reference at ``capture_time - delay``. Once aligned,
the stream is consumed in order so bursty ROS delivery does not destroy AEC.
"""

from __future__ import annotations

import collections
import ctypes
import ctypes.util
import math
import threading
from typing import Callable, Deque, Optional, Tuple

import numpy as np


SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_SECONDS = FRAME_MS / 1000.0
LOCK_TOLERANCE_SECONDS = 0.15
DRIFT_TOLERANCE_SECONDS = 0.30
ENV_HISTORY = 1200


class AECBackendError(RuntimeError):
    pass


class _LiveKitBackend:
    name = "livekit-apm"

    def __init__(self, sample_rate: int, frame_samples: int):
        try:
            from livekit.rtc.apm import (  # type: ignore
                AudioFrame,
                AudioProcessingModule,
            )
        except Exception as exc:
            raise AECBackendError(f"LiveKit APM unavailable: {exc}") from exc
        self._frame_class = AudioFrame
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=False,
        )

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        reference_frame = self._frame_class(
            reference.tobytes(), self._sample_rate, 1, self._frame_samples
        )
        self._apm.process_reverse_stream(reference_frame)
        mic_frame = self._frame_class(
            mic.tobytes(), self._sample_rate, 1, self._frame_samples
        )
        self._apm.process_stream(mic_frame)
        return np.frombuffer(mic_frame.data, dtype=np.int16).copy()

    def close(self) -> None:
        return None


class _SpeexDSPBackend:
    name = "speexdsp"
    _SET_SAMPLING_RATE = 24

    def __init__(self, sample_rate: int, frame_samples: int, filter_length_ms: int):
        candidates = []
        discovered = ctypes.util.find_library("speexdsp")
        if discovered:
            candidates.append(discovered)
        candidates.extend(("libspeexdsp.so.1", "libspeexdsp.so"))

        library = None
        errors = []
        for candidate in candidates:
            try:
                library = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        if library is None:
            raise AECBackendError(
                "SpeexDSP unavailable; install libspeexdsp1 (" + "; ".join(errors) + ")"
            )

        sample_ptr = ctypes.POINTER(ctypes.c_int16)
        library.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        library.speex_echo_state_init.restype = ctypes.c_void_p
        library.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        library.speex_echo_state_destroy.restype = None
        library.speex_echo_cancellation.argtypes = [
            ctypes.c_void_p,
            sample_ptr,
            sample_ptr,
            sample_ptr,
        ]
        library.speex_echo_cancellation.restype = None
        library.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        library.speex_echo_ctl.restype = ctypes.c_int

        filter_samples = max(frame_samples, sample_rate * filter_length_ms // 1000)
        state = library.speex_echo_state_init(frame_samples, filter_samples)
        if not state:
            raise AECBackendError("speex_echo_state_init returned NULL")
        rate = ctypes.c_int(sample_rate)
        if library.speex_echo_ctl(
            state, self._SET_SAMPLING_RATE, ctypes.byref(rate)
        ) != 0:
            library.speex_echo_state_destroy(state)
            raise AECBackendError("speex_echo_ctl failed to set sampling rate")

        self._library = library
        self._state = state

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        mic = np.ascontiguousarray(mic, dtype=np.int16)
        reference = np.ascontiguousarray(reference, dtype=np.int16)
        output = np.empty_like(mic)
        sample_ptr = ctypes.POINTER(ctypes.c_int16)
        self._library.speex_echo_cancellation(
            self._state,
            mic.ctypes.data_as(sample_ptr),
            reference.ctypes.data_as(sample_ptr),
            output.ctypes.data_as(sample_ptr),
        )
        return output

    def close(self) -> None:
        if getattr(self, "_state", None):
            self._library.speex_echo_state_destroy(self._state)
            self._state = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_backend(
    backend: str,
    sample_rate: int,
    frame_samples: int,
    filter_length_ms: int,
):
    backend = (backend or "auto").strip().lower()
    factories = {
        "livekit": lambda: _LiveKitBackend(sample_rate, frame_samples),
        "speexdsp": lambda: _SpeexDSPBackend(
            sample_rate, frame_samples, filter_length_ms
        ),
    }
    if backend != "auto":
        if backend not in factories:
            raise AECBackendError(
                f"unsupported AEC backend {backend!r}; expected auto/livekit/speexdsp"
            )
        return factories[backend]()

    failures = []
    for name in ("livekit", "speexdsp"):
        try:
            return factories[name]()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    raise AECBackendError("no AEC backend available (" + "; ".join(failures) + ")")


class AECProcessor:
    """Thread-safe reference alignment plus one instance-local AEC backend."""

    def __init__(
        self,
        *,
        delay_ms: int = 100,
        backend: str = "auto",
        filter_length_ms: int = 200,
        backend_factory: Optional[Callable[[], object]] = None,
    ):
        if not 0 <= int(delay_ms) <= 2500:
            raise ValueError("delay_ms must be between 0 and 2500")
        if not 50 <= int(filter_length_ms) <= 1000:
            raise ValueError("filter_length_ms must be between 50 and 1000")
        self.delay_ms = int(delay_ms)
        self.delay_seconds = self.delay_ms / 1000.0
        self.filter_length_ms = int(filter_length_ms)
        self._backend = (
            backend_factory()
            if backend_factory is not None
            else build_backend(
                backend,
                SAMPLE_RATE,
                FRAME_SAMPLES,
                self.filter_length_ms,
            )
        )
        self.backend_name = getattr(self._backend, "name", backend)
        self._lock = threading.Lock()
        self._reference_queue: Deque[Tuple[float, np.ndarray]] = collections.deque(
            maxlen=4000
        )
        self._reference_env: Deque[Tuple[float, float]] = collections.deque(
            maxlen=ENV_HISTORY
        )
        self._mic_env: Deque[Tuple[float, float, float, bool]] = collections.deque(
            maxlen=ENV_HISTORY
        )
        self._mic_carry = bytearray()
        self._mic_cursor_ts: Optional[float] = None
        self._locked = False
        self._frames_processed = 0
        self._frames_with_reference = 0
        self._aligned_frames = 0
        self._silence_frames = 0
        self._dropped_expired = 0
        self._reference_overruns = 0
        self._relocks = 0
        self._mic_relocks = 0
        self._process_errors = 0

    def push_reference(self, pcm_bytes: bytes, play_start_ts: float) -> None:
        if not pcm_bytes:
            return
        if len(pcm_bytes) % 2:
            raise ValueError("TTS reference PCM must contain complete int16 samples")
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        chunks = []
        for offset in range(0, len(samples), FRAME_SAMPLES):
            chunk = samples[offset : offset + FRAME_SAMPLES]
            if len(chunk) < FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
            chunk = np.ascontiguousarray(chunk, dtype=np.int16)
            ts = float(play_start_ts) + (offset / SAMPLE_RATE)
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            chunks.append((ts, chunk, rms))

        with self._lock:
            for ts, chunk, rms in chunks:
                if len(self._reference_queue) == self._reference_queue.maxlen:
                    self._reference_overruns += 1
                self._reference_queue.append((ts, chunk))
                self._reference_env.append((ts, rms))

    def _take_reference_locked(self, target_ts: float) -> Optional[np.ndarray]:
        queue = self._reference_queue
        if not queue:
            self._locked = False
            return None
        if self._locked:
            if abs(queue[0][0] - target_ts) > DRIFT_TOLERANCE_SECONDS:
                self._locked = False
                self._relocks += 1
            else:
                return queue.popleft()[1]
        while queue and queue[0][0] < target_ts - LOCK_TOLERANCE_SECONDS:
            queue.popleft()
            self._dropped_expired += 1
        if queue and queue[0][0] <= target_ts + LOCK_TOLERANCE_SECONDS:
            self._locked = True
            return queue.popleft()[1]
        return None

    def process_pcm(self, pcm_bytes: bytes, capture_ts: float) -> bytes:
        if not pcm_bytes:
            return b""
        if len(pcm_bytes) % 2:
            raise ValueError("microphone PCM must contain complete int16 samples")

        packet_duration = len(pcm_bytes) / (SAMPLE_RATE * 2)
        if self._mic_cursor_ts is None:
            self._mic_cursor_ts = float(capture_ts) - packet_duration
        self._mic_carry.extend(pcm_bytes)
        output = bytearray()

        while len(self._mic_carry) >= FRAME_BYTES:
            raw = bytes(self._mic_carry[:FRAME_BYTES])
            del self._mic_carry[:FRAME_BYTES]
            mic = np.frombuffer(raw, dtype=np.int16).copy()
            mic_start_ts = self._mic_cursor_ts
            self._mic_cursor_ts += FRAME_SECONDS
            target_reference_ts = mic_start_ts - self.delay_seconds
            with self._lock:
                reference = self._take_reference_locked(target_reference_ts)
            had_reference = reference is not None
            if reference is None:
                reference = np.zeros(FRAME_SAMPLES, dtype=np.int16)
                self._silence_frames += 1
            else:
                self._aligned_frames += 1

            try:
                clean = self._backend.process(mic, reference)
            except Exception:
                self._process_errors += 1
                raise
            self._frames_processed += 1
            if had_reference:
                self._frames_with_reference += 1
            in_rms = float(np.sqrt(np.mean(mic.astype(np.float32) ** 2)))
            out_rms = float(np.sqrt(np.mean(clean.astype(np.float32) ** 2)))
            with self._lock:
                self._mic_env.append(
                    (mic_start_ts, in_rms, out_rms, had_reference)
                )
            output.extend(np.ascontiguousarray(clean, dtype=np.int16).tobytes())

        expected_end = self._mic_cursor_ts + len(self._mic_carry) / (SAMPLE_RATE * 2)
        if abs(expected_end - float(capture_ts)) > DRIFT_TOLERANCE_SECONDS:
            self._mic_cursor_ts = float(capture_ts) - len(self._mic_carry) / (
                SAMPLE_RATE * 2
            )
            self._mic_relocks += 1
        return bytes(output)

    def stats(self) -> dict:
        with self._lock:
            queue_size = len(self._reference_queue)
            locked = self._locked
            mic_env = list(self._mic_env)
        total = self._aligned_frames + self._silence_frames
        align_rate = self._aligned_frames / total if total else 0.0
        return {
            "backend": self.backend_name,
            "delay_ms": self.delay_ms,
            "filter_length_ms": self.filter_length_ms,
            "locked": locked,
            "frames_processed": self._frames_processed,
            "frames_with_reference": self._frames_with_reference,
            "aligned_frames": self._aligned_frames,
            "silence_frames": self._silence_frames,
            "align_rate": round(align_rate, 3),
            "ref_queue_size": queue_size,
            "dropped_expired": self._dropped_expired,
            "reference_overruns": self._reference_overruns,
            "relocks": self._relocks,
            "mic_relocks": self._mic_relocks,
            "process_errors": self._process_errors,
            "erle_db": self._erle_db(mic_env),
        }

    @staticmethod
    def _erle_db(mic_env) -> Optional[float]:
        input_energy = 0.0
        output_energy = 0.0
        for _ts, input_rms, output_rms, had_reference in mic_env:
            if had_reference:
                input_energy += input_rms * input_rms
                output_energy += output_rms * output_rms
        if input_energy <= 0 or output_energy <= 0:
            return None
        return round(10.0 * math.log10(input_energy / output_energy), 1)

    def calibrate(self, min_lag_s: float = -0.5, max_lag_s: float = 2.5) -> dict:
        with self._lock:
            mic = list(self._mic_env)
            reference = list(self._reference_env)
        if len(mic) < 200 or len(reference) < 100:
            return {
                "ok": False,
                "reason": (
                    f"history not enough (mic={len(mic)}, ref={len(reference)}); "
                    "play several seconds of TTS before calibration"
                ),
            }
        start = min(mic[0][0], reference[0][0])
        end = max(mic[-1][0], reference[-1][0])
        size = int(round((end - start) / FRAME_SECONDS)) + 2
        if size > 4000:
            return {"ok": False, "reason": "time span too large"}
        # NaN marks time buckets not covered by that stream. Normalizing a
        # zero-padded union window depresses peak correlation solely because
        # the two streams are shifted; calibrate each lag on its true overlap.
        mic_env = np.full(size, np.nan, dtype=np.float64)
        ref_env = np.full(size, np.nan, dtype=np.float64)
        for ts, input_rms, _output_rms, _had_reference in mic:
            mic_env[int(round((ts - start) / FRAME_SECONDS))] = input_rms
        for ts, rms in reference:
            ref_env[int(round((ts - start) / FRAME_SECONDS))] = rms
        low = int(min_lag_s / FRAME_SECONDS)
        high = int(max_lag_s / FRAME_SECONDS)
        best_lag = None
        best_correlation = -1.0
        for lag in range(low, high + 1):
            if lag >= 0:
                mic_slice = mic_env[lag:]
                ref_slice = ref_env[: size - lag]
            else:
                mic_slice = mic_env[: size + lag]
                ref_slice = ref_env[-lag:]
            valid = np.isfinite(mic_slice) & np.isfinite(ref_slice)
            if int(valid.sum()) < 100:
                continue
            mic_overlap = mic_slice[valid]
            ref_overlap = ref_slice[valid]
            mic_centered = mic_overlap - mic_overlap.mean()
            ref_centered = ref_overlap - ref_overlap.mean()
            denominator = float(
                np.sqrt(np.sum(mic_centered ** 2) * np.sum(ref_centered ** 2))
            )
            if denominator <= 0:
                continue
            correlation = float(np.dot(mic_centered, ref_centered) / denominator)
            if correlation > best_correlation:
                best_correlation = correlation
                best_lag = lag
        if best_lag is None:
            return {"ok": False, "reason": "no non-flat overlapping envelopes"}
        return {
            "ok": True,
            "d_real_ms": int(best_lag * FRAME_SECONDS * 1000),
            "peak_corr": round(best_correlation, 3),
            "current_delay_ms": self.delay_ms,
            "mic_env_points": len(mic),
            "ref_env_points": len(reference),
        }

    def close(self) -> None:
        self._backend.close()
        with self._lock:
            self._reference_queue.clear()
            self._reference_env.clear()
            self._mic_env.clear()
            self._mic_carry.clear()
