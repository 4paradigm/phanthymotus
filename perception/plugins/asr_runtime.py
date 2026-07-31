"""Shared PCM conversion and live VAD sessions for ASR entry points."""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, Optional


SAMPLE_RATE = 16000
_SUPPORTED_VAD_BACKENDS = {"energy", "sherpa_onnx", "webrtc", "firered"}

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised in minimal runtime images
    _np = None


def pcm16_to_float_samples(pcm: bytes):
    """Convert little-endian PCM16 bytes to normalized float samples."""
    pcm = pcm[: len(pcm) // 2 * 2]
    if not pcm:
        return []
    if _np is not None:
        return _np.frombuffer(pcm, dtype="<i2").astype(_np.float32) / 32768.0
    sample_count = len(pcm) // 2
    return [
        sample / 32768.0
        for sample in struct.unpack(f"<{sample_count}h", pcm)
    ]


def float_samples_to_pcm16(samples: Iterable[float]) -> bytes:
    """Convert normalized float samples to clipped little-endian PCM16."""
    if _np is not None:
        values = _np.asarray(samples, dtype=_np.float32)
        return (
            _np.clip(values * 32768.0, -32768, 32767)
            .astype("<i2")
            .tobytes()
        )
    integers = [
        int(max(-32768, min(32767, sample * 32768.0))) for sample in samples
    ]
    return struct.pack(f"<{len(integers)}h", *integers)


def normalize_vad_backend(backend: str | None) -> str:
    """Return a canonical VAD backend name or reject unsupported config."""
    name = (backend or "sherpa_onnx").strip().lower()
    aliases = {
        "sherpa": "sherpa_onnx",
        "silero": "sherpa_onnx",
        "silero_onnx": "sherpa_onnx",
        "webrtcvad": "webrtc",
        "fireredvad": "firered",
        "fire_red": "firered",
    }
    name = aliases.get(name, name)
    if name not in _SUPPORTED_VAD_BACKENDS:
        expected = ", ".join(sorted(_SUPPORTED_VAD_BACKENDS | {"silero"}))
        raise ValueError(
            f"Unsupported VAD backend: {backend}. Expected one of: {expected}"
        )
    return name


def resolve_vad_settings(cfg: dict) -> dict:
    """Resolve nested config-file and flat MCP VAD settings consistently."""
    vad_cfg = cfg.get("vad") if isinstance(cfg.get("vad"), dict) else {}

    def _value(flat_key: str, nested_key: str, default):
        flat_value = cfg.get(flat_key)
        if flat_value is not None and flat_value != "":
            return flat_value
        return vad_cfg.get(nested_key, default)

    return {
        "backend": normalize_vad_backend(
            _value("vad_backend", "model", "sherpa_onnx")
        ),
        "threshold": float(_value("vad_threshold", "threshold", 0.5)),
        "silence_ms": int(_value("vad_silence_ms", "silence_ms", 400)),
        "pre_roll_ms": int(_value("vad_pre_roll_ms", "pre_roll_ms", 500)),
        "model_dir": str(
            _value("vad_model_dir", "model_dir", "/models/sherpa-onnx/vad")
        ),
    }


class _FrameVadSession:
    """Collect fixed-size VAD decisions into timestamped utterances."""

    def __init__(
        self,
        is_speech: Callable[[bytes], bool],
        silence_ms: int,
        pre_roll_ms: int,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 30,
    ):
        self._is_speech = is_speech
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._frame_bytes = sample_rate * frame_ms // 1000 * 2
        self._frame_seconds = frame_ms / 1000.0
        self._silence_limit = max(1, math.ceil(silence_ms / frame_ms))
        pre_roll_frames = max(0, math.ceil(pre_roll_ms / frame_ms))
        self._pre_roll: Deque[tuple[bytes, float, float]] = deque(
            maxlen=pre_roll_frames
        )
        self._completed: Deque[tuple[bytes, float, float]] = deque()
        self.reset()

    def reset(self) -> None:
        self._pending_chunks: Deque[tuple[bytes, float]] = deque()
        self._pending_samples = 0
        self._in_speech = False
        self._speech_frames: list[tuple[bytes, float, float]] = []
        self._silence_frames = 0
        self._pre_roll.clear()
        self._completed.clear()

    def process_chunk(
        self, chunk: bytes, now_ts: float
    ) -> Optional[tuple[bytes, float, float]]:
        if not chunk:
            return self._completed.popleft() if self._completed else None
        chunk = chunk[: len(chunk) // 2 * 2]
        self._pending_chunks.append((chunk, now_ts))
        self._pending_samples += len(chunk) // 2

        frame_samples = self._frame_bytes // 2
        while self._pending_samples >= frame_samples:
            frame, frame_start_ts, frame_end_ts = self._take_pending(frame_samples)
            result = self._consume_frame(frame, frame_start_ts, frame_end_ts)
            if result is not None:
                self._completed.append(result)

        return self._completed.popleft() if self._completed else None

    def _take_pending(self, sample_count: int) -> tuple[bytes, float, float]:
        parts = []
        start_ts = None
        end_ts = None
        remaining = sample_count
        while remaining:
            pcm, chunk_ts = self._pending_chunks.popleft()
            available = len(pcm) // 2
            taken = min(remaining, available)
            parts.append(pcm[: taken * 2])
            if start_ts is None:
                start_ts = chunk_ts
            end_ts = chunk_ts + taken / self._sample_rate
            if taken < available:
                self._pending_chunks.appendleft((pcm[taken * 2 :], end_ts))
            remaining -= taken
            self._pending_samples -= taken
        return b"".join(parts), start_ts, end_ts

    def _consume_frame(
        self, frame: bytes, frame_start_ts: float, frame_end_ts: float
    ) -> Optional[tuple[bytes, float, float]]:
        speech = self._is_speech(frame)
        if not self._in_speech:
            if not speech:
                self._pre_roll.append((frame, frame_start_ts, frame_end_ts))
                return None
            self._speech_frames = list(self._pre_roll)
            self._pre_roll.clear()
            self._speech_frames.append((frame, frame_start_ts, frame_end_ts))
            self._in_speech = True
            self._silence_frames = 0
            return None

        self._speech_frames.append((frame, frame_start_ts, frame_end_ts))
        if speech:
            self._silence_frames = 0
            return None

        self._silence_frames += 1
        if self._silence_frames < self._silence_limit:
            return None
        return self._finish_utterance()

    def _finish_utterance(self) -> tuple[bytes, float, float]:
        frames = self._speech_frames
        utterance = b"".join(frame for frame, _, _ in frames)
        start_ts = frames[0][1]
        end_ts = frames[-1][2]
        self._speech_frames = []
        self._in_speech = False
        self._silence_frames = 0
        return utterance, start_ts, end_ts

    def flush(self) -> bytes:
        if self._speech_frames and self._pending_samples:
            pending = self._take_pending(self._pending_samples)
            self._speech_frames.append(pending)
        if not self._speech_frames:
            return b""
        utterance, _, _ = self._finish_utterance()
        return utterance


@dataclass(frozen=True)
class _HistoryChunk:
    start_sample: int
    pcm: bytes
    timestamp: float

    @property
    def sample_count(self) -> int:
        return len(self.pcm) // 2


class _TimedPcmHistory:
    """A bounded PCM history addressable by absolute sample offset."""

    def __init__(self, max_samples: int, sample_rate: int = SAMPLE_RATE):
        self._max_samples = max_samples
        self._sample_rate = sample_rate
        self._chunks: Deque[_HistoryChunk] = deque()
        self.total_samples = 0

    def clear(self) -> None:
        self._chunks.clear()
        self.total_samples = 0

    def append(self, pcm: bytes, timestamp: float) -> None:
        pcm = pcm[: len(pcm) // 2 * 2]
        if not pcm:
            return
        chunk = _HistoryChunk(self.total_samples, pcm, timestamp)
        self._chunks.append(chunk)
        self.total_samples += chunk.sample_count
        cutoff = self.total_samples - self._max_samples
        while self._chunks:
            first = self._chunks[0]
            if first.start_sample + first.sample_count >= cutoff:
                break
            self._chunks.popleft()

    def slice(
        self, start_sample: int, end_sample: int
    ) -> tuple[bytes, Optional[float], Optional[float]]:
        parts = []
        first_timestamp = None
        last_timestamp = None
        for chunk in self._chunks:
            chunk_end = chunk.start_sample + chunk.sample_count
            overlap_start = max(start_sample, chunk.start_sample)
            overlap_end = min(end_sample, chunk_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - chunk.start_sample
            local_end = overlap_end - chunk.start_sample
            parts.append(chunk.pcm[local_start * 2 : local_end * 2])
            if first_timestamp is None:
                first_timestamp = chunk.timestamp + local_start / self._sample_rate
            last_timestamp = chunk.timestamp + local_end / self._sample_rate
        return b"".join(parts), first_timestamp, last_timestamp


class _SherpaVadSession:
    """sherpa-onnx Silero VAD with timestamped pre-roll reconstruction."""

    def __init__(
        self,
        threshold: float,
        silence_ms: int,
        pre_roll_ms: int,
        model_dir: str,
        sample_rate: int = SAMPLE_RATE,
    ):
        import os

        import sherpa_onnx

        from utils.model_downloader import ensure_model

        ensure_model("vad", model_dir)
        model_path = os.path.join(model_dir, "silero_vad.onnx")
        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=model_path,
                threshold=threshold,
                min_silence_duration=silence_ms / 1000.0,
                min_speech_duration=0.1,
                window_size=512,
                max_speech_duration=30,
            ),
            sample_rate=sample_rate,
            num_threads=1,
            provider="cpu",
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=30
        )
        self._sample_rate = sample_rate
        self._silence_samples = int(sample_rate * silence_ms / 1000)
        self._pre_roll_samples = int(sample_rate * pre_roll_ms / 1000)
        self._history = _TimedPcmHistory(sample_rate * 31, sample_rate)
        self._completed: Deque[tuple[bytes, float, float]] = deque()

    def reset(self) -> None:
        if hasattr(self._vad, "reset"):
            self._vad.reset()
        self._history.clear()
        self._completed.clear()

    def process_chunk(
        self, chunk: bytes, now_ts: float
    ) -> Optional[tuple[bytes, float, float]]:
        self._history.append(chunk, now_ts)
        samples = pcm16_to_float_samples(chunk)
        if len(samples):
            self._vad.accept_waveform(samples)
            self._drain(now_ts)
        return self._completed.popleft() if self._completed else None

    def _drain(self, fallback_ts: float) -> None:
        while not self._vad.empty():
            segment = self._vad.front
            segment_samples = segment.samples
            segment_start = getattr(segment, "start", None)
            if segment_start is None:
                segment_start = max(
                    0,
                    self._history.total_samples
                    - len(segment_samples)
                    - self._silence_samples,
                )
            segment_start = int(segment_start)
            # Extend segment end by tail_pad_ms to capture weak trailing
            # syllables that the VAD misclassifies as silence (e.g., case 7
            # "举双手" tail 144ms classified as silence → truncated → hotwords
            # can't recover).  [[arm-int8-drift]]
            tail_pad_samples = int(self._sample_rate * 0.3)  # 300ms
            segment_end_orig = segment_start + len(segment_samples)
            segment_end = min(segment_end_orig + tail_pad_samples,
                              self._history.total_samples)
            pre_start = max(0, segment_start - self._pre_roll_samples)
            pre_pcm, start_ts, _ = self._history.slice(pre_start, segment_start)
            segment_pcm, segment_ts, end_ts = self._history.slice(
                segment_start, segment_end
            )
            if len(segment_pcm) < len(segment_samples) * 2:
                segment_pcm = float_samples_to_pcm16(segment_samples)
            utterance = pre_pcm + segment_pcm
            if start_ts is None:
                start_ts = segment_ts
            if start_ts is None:
                start_ts = fallback_ts - len(utterance) / 2 / self._sample_rate
            if end_ts is None:
                end_ts = start_ts + len(utterance) / 2 / self._sample_rate
            self._completed.append((utterance, start_ts, end_ts))
            self._vad.pop()

    def flush(self) -> bytes:
        if hasattr(self._vad, "flush"):
            self._vad.flush()
            self._drain(0.0)
        if not self._completed:
            return b""
        return self._completed.popleft()[0]


class _FireRedVadSession:
    """FireRedVAD (DFSMN ONNX) session: buffer audio, detect after trailing silence.

    FireRedVAD is non-streaming, so we buffer all PCM and run one detect
    after seeing >= silence_ms of actual silence (zero-valued PCM). This
    matches the eval pipeline's "send audio → send 1500ms silence → flush"
    flow without the segment-boundary drift of incremental re-detection.

    Each returned segment gets pre_roll + 300ms tail pad to protect weak
    trailing syllables (see [[asr-miss-root-cause]]).
    """

    _TAIL_PAD_S = 0.3
    _SILENCE_TO_DETECT_S = 1.0  # run one detect after >=1s of trailing silence
    _STARVE_TRIGGER_S = 1.5     # fallback: stream quiet this long → detect anyway
    _MAX_BUFFER_S = 15.0        # hard cap: buffer this long → force detect
    _WHOLE_BUFFER_MIN_S = 0.5   # min duration to fallback to whole-buffer output

    def __init__(
        self,
        threshold: float,
        silence_ms: int,
        pre_roll_ms: int,
        model_dir: str,
        sample_rate: int = SAMPLE_RATE,
    ):
        from plugins.firered_vad import FireRedVadOnnx

        self._detector = FireRedVadOnnx(
            model_dir,
            speech_threshold=threshold,
            min_silence_frame=max(1, round(silence_ms / 10)),
        )
        self._sample_rate = sample_rate
        self._pre_roll_s = pre_roll_ms / 1000.0
        self._pcm = bytearray()
        self._detected = False
        self._silence_samples = 0
        self._last_chunk_ts = 0.0
        self._last_speech_ts = 0.0   # only non-zero chunks; trickle silence can't reset this
        self._completed: Deque[tuple[bytes, float, float]] = deque()

    def reset(self) -> None:
        self._pcm.clear()
        self._detected = False
        self._silence_samples = 0
        self._last_chunk_ts = 0.0
        self._last_speech_ts = 0.0
        self._completed.clear()

    def _total_s(self) -> float:
        return len(self._pcm) / 2 / self._sample_rate

    def _slice(self, start_s: float, end_s: float) -> bytes:
        start_b = max(0, int(start_s * self._sample_rate) * 2)
        end_b = min(len(self._pcm), int(end_s * self._sample_rate) * 2)
        return bytes(self._pcm[start_b:end_b])

    def _run_detect(self) -> None:
        if self._detected or _np is None:
            return
        total_s = self._total_s()
        samples = _np.frombuffer(bytes(self._pcm), dtype="<i2").astype(_np.float32)
        if samples.shape[0] < int(0.05 * self._sample_rate):
            return
        try:
            segs = self._detector.detect(samples)
        except Exception as e:
            logger_fr = logging.getLogger("asr_runtime.firered")
            logger_fr.warning(f"[firered-vad] detect failed: {e}")
            return
        self._detected = True
        if not segs:
            return
        parts = []
        for start_s, end_s in segs:
            start = max(0.0, start_s - self._pre_roll_s)
            end = min(end_s + self._TAIL_PAD_S, total_s)
            parts.append(self._slice(start, end))
        utterance = b"".join(parts)
        # timestamps are approximate: span covers all audio up to total_s
        self._completed.append((
            utterance,
            -total_s,  # start_ts relative (not used by caller)
            0.0,
        ))

    def process_chunk(
        self, chunk: bytes, now_ts: float
    ) -> Optional[tuple[bytes, float, float]]:
        if self._detected:
            return self._completed.popleft() if self._completed else None
        silence_thresh_samples = int(self._SILENCE_TO_DETECT_S * self._sample_rate)
        if chunk:
            self._pcm += chunk
            self._last_chunk_ts = now_ts
            # is this chunk all-zero? (PCM16 silence)
            if chunk == b"\x00" * len(chunk):
                self._silence_samples += len(chunk) // 2
            else:
                self._silence_samples = 0
                self._last_speech_ts = now_ts
            if self._silence_samples >= silence_thresh_samples:
                self._run_detect()
        return self._completed.popleft() if self._completed else None

    def notify_idle(
        self, now_ts: float
    ) -> Optional[tuple[bytes, float, float]]:
        """Starvation fallback: run detect when the stream went quiet
        without a clean run of zero chunks, OR the buffer has accumulated
        too much un-examined audio.

        On the judge under 10-instance load, UDP reordering can:
          (a) insert a speech packet into the silence tail → reset the
              zero counter → detect never fires
          (b) deliver trickle packets indefinitely → wall-clock
              starvation (1.5s of silence) never triggers
          (c) deliver only silence packets → _last_chunk_ts keeps
              advancing while _last_speech_ts is frozen

        OR conditions handle all three:
          1. Wall-clock silence ≥ 1.5s (existing, uses _last_speech_ts
             so trickle silence can't defeat it)
          2. Buffer duration ≥ _MAX_BUFFER_S (hard cap)"""
        if self._detected or not self._pcm:
            return None
        total_s = self._total_s()
        starve = (
            self._last_speech_ts > 0
            and now_ts - self._last_speech_ts >= self._STARVE_TRIGGER_S
        )
        buf_full = total_s >= self._MAX_BUFFER_S
        if not (starve or buf_full):
            return None
        if buf_full:
            logger_fb = logging.getLogger("asr_runtime.firered")
            logger_fb.warning(
                "[firered-vad] buffer cap triggered %.1fs (last spk %.1fs ago)",
                total_s, now_ts - self._last_speech_ts if self._last_speech_ts else -1,
            )
        self._run_detect()
        return self._completed.popleft() if self._completed else None

    def flush(self) -> bytes:
        if not self._detected:
            self._run_detect()
        if not self._completed:
            return b""
        return self._completed.popleft()[0]

    def force_flush(self) -> bytes:
        """Last-resort: re-detect the FULL buffer and emit something.

        Called by stop() before pausing. Bypasses the _detected guard
        (which an earlier silence trigger may have set) and re-runs
        detect on everything accumulated.  If FireRedVAD still finds
        no segments but the buffer has meaningful audio, fall back to
        sending the whole buffer — even garbled output beats empty
        (CER=1.0).  A 5% zero-sample threshold guards against
        emitting pure-silence buffers."""
        if _np is None:
            return b""
        self._detected = False  # re-enable for one last pass
        self._run_detect()
        if self._completed:
            return self._completed.popleft()[0]
        # No segments even on re-detect — whole-buffer fallback
        total_s = self._total_s()
        if total_s < self._WHOLE_BUFFER_MIN_S:
            return b""
        samples = _np.frombuffer(bytes(self._pcm), dtype="<i2").astype(_np.float32)
        nonzero = int(_np.sum(_np.abs(samples) > 20.0))  # 20 ~= -64dBFS
        if nonzero <= int(samples.shape[0] * 0.05):
            return b""
        logger_fb = logging.getLogger("asr_runtime.firered")
        logger_fb.info(
            "[firered-vad] force_flush: no segments, falling back to "
            "whole buffer %.2fs (%d/%d nonzero samples)",
            total_s, nonzero, samples.shape[0],
        )
        return bytes(self._pcm)


class VadSession:
    """Backend-selecting VAD session shared by ROS and WebSocket ASR."""

    def __init__(
        self,
        backend: str = "sherpa_onnx",
        threshold: float = 0.5,
        silence_ms: int = 400,
        pre_roll_ms: int = 500,
        model_dir: str = "/models/sherpa-onnx/vad",
    ):
        self.backend = normalize_vad_backend(backend)
        if self.backend == "sherpa_onnx":
            self._impl = _SherpaVadSession(
                threshold, silence_ms, pre_roll_ms, model_dir
            )
        elif self.backend == "firered":
            self._impl = _FireRedVadSession(
                threshold, silence_ms, pre_roll_ms, model_dir
            )
        else:
            if self.backend == "webrtc":
                import webrtcvad

                vad = webrtcvad.Vad()
                vad.set_mode(max(0, min(3, round(threshold * 3))))
                is_speech = lambda frame: vad.is_speech(frame, SAMPLE_RATE)
            else:
                speech_threshold = threshold * 0.1

                def is_speech(frame: bytes) -> bool:
                    samples = pcm16_to_float_samples(frame)
                    if not len(samples):
                        return False
                    if _np is not None:
                        rms = float(_np.sqrt(_np.mean(samples * samples)))
                    else:
                        rms = math.sqrt(
                            sum(sample * sample for sample in samples) / len(samples)
                        )
                    return rms >= speech_threshold

            self._impl = _FrameVadSession(
                is_speech,
                silence_ms=silence_ms,
                pre_roll_ms=pre_roll_ms,
            )

    def init(self) -> None:
        self._impl.reset()

    def process_chunk(
        self, chunk: bytes, now_ts: float
    ) -> Optional[tuple[bytes, float, float]]:
        return self._impl.process_chunk(chunk, now_ts)

    def notify_idle(
        self, now_ts: float
    ) -> Optional[tuple[bytes, float, float]]:
        """Forward starvation ticks to backends that support them."""
        fn = getattr(self._impl, "notify_idle", None)
        if fn is None:
            return None
        return fn(now_ts)

    def force_flush(self) -> bytes:
        """Flush even when the normal endpoint hasn't fired (corner cases)."""
        fn = getattr(self._impl, "force_flush", None)
        if fn is None:
            return self._impl.flush()
        return fn()

    def flush(self) -> bytes:
        return self._impl.flush()
