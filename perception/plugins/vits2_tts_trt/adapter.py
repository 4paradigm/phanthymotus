"""In-process adapter for the Jetson VITS2 TensorRT runtime."""

from __future__ import annotations

import logging
import os
import threading
from copy import deepcopy
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .runtime.backends.trt_numpy_tts_engine import (
    SynthesisFrameLimitError, TensorRTNumpyTTSEngine,
)


log = logging.getLogger(__name__)


def _env_int(name: str, default: int, low: int, high: int, even: bool = False) -> int:
    """Read a bounded int from the environment, falling back on bad input.

    Same reasoning as plugin._env_int: this module is imported while the plugin
    list is built, so a bad tuning variable must not abort the whole perception
    process.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("[vits2_tts_trt] %s=%r is not an integer; using %d", name, raw, default)
        return default
    if not low <= value <= high or (even and value % 2):
        log.warning("[vits2_tts_trt] %s=%d is unusable; using %d", name, value, default)
        return default
    return value


SAMPLE_RATE = 16000
# PCM frame size on the wire. Must stay even (16-bit samples) and match what
# the Speaker side expects; 3200 bytes is 100 ms at 16 kHz mono.
CHUNK_BYTES = _env_int("MIX_VITS_CHUNK_BYTES", 3200, 2, 1 << 20, even=True)
PCM_FRAME_MS = CHUNK_BYTES * 1000.0 / (SAMPLE_RATE * 2)
MAX_CHUNK_TOKENS = _env_int("MIX_VITS_MAX_TEXT_TOKENS", 256, 1, 4096)
CHUNK_PAUSE_MS = _env_int("MIX_VITS_CHUNK_PAUSE_MS", 0, 0, 1000)
WARMUP_CASES = (
    "你好，语音服务已经准备好了。",
    "今天可以使用平板电脑查看消息。",
    "The device is ready for a short test."
)


def _quiet_samples(samples, reverse=False):
    """Count only verified low-energy 5-ms windows, retaining a 10-ms guard."""
    view = samples[::-1] if reverse else samples
    window = SAMPLE_RATE // 200
    quiet = 0
    for index in range(0, len(view) - window + 1, window):
        values = view[index:index + window].astype(np.float32)
        if np.max(np.abs(values)) > 64 or np.sqrt(np.mean(values * values)) > 15:
            break
        quiet = index + window
    return max(0, quiet - SAMPLE_RATE // 100)


def _join_pcm(left, right):
    """Join mono PCM with a two-millisecond overlap, without changing speed."""
    if not len(left):
        return right
    if not len(right):
        return left
    overlap = SAMPLE_RATE // 500
    if len(left) <= overlap or len(right) <= overlap:
        raise ValueError("TTS segment is too short to join")
    ramp = np.linspace(0, 1, overlap)
    mixed = np.rint(left[-overlap:].astype(np.float32) * (1 - ramp)
                    + right[:overlap].astype(np.float32) * ramp).astype(np.int16)
    return np.concatenate((left[:-overlap], mixed, right[overlap:]))


class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)

    def warmup(self) -> int:
        return 0

    def set_speed(self, speed: float) -> None:
        del speed


class Vits2TensorRTAdapter(TTSAdapter):
    def __init__(
        self,
        speed: float = 1.0,
        *,
        engine=None,
        model_config: str = "",
        engine_dir: str = "",
        max_chunk_tokens: int | None = None,
    ):
        # Both paths come from the installed release (see
        # utils.model_downloader.ensure_vits2_model), never from an env default:
        # engines are per JetPack family, so a hardcoded fallback would happily
        # hand TensorRT 10 a plan built by TensorRT 8.
        if engine is None and not (model_config and engine_dir):
            raise ValueError("model_config and engine_dir are required")
        self._lock = threading.Lock()
        self.set_speed(speed)
        self._engine = engine or TensorRTNumpyTTSEngine(model_config, engine_dir)
        self.max_chunk_tokens = int(
            MAX_CHUNK_TOKENS if max_chunk_tokens is None else max_chunk_tokens
        )
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        engine_limit = int(getattr(self._engine, "max_text_tokens", self.max_chunk_tokens))
        if self.max_chunk_tokens > engine_limit:
            raise ValueError(
                f"adapter text limit {self.max_chunk_tokens} exceeds engine limit "
                f"{engine_limit}"
            )

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if speed <= 0 or speed > 4:
            raise ValueError("TTS speed must be greater than zero and at most four")
        with self._lock:
            self._speed = speed

    def _token_count(self, text: str) -> int:
        return len(self._engine._get_text_ids(text, normalized=True)[0])

    def iter_text_chunks(self, text: str, reading_language=None):
        """Yield the exact normalized text chunks and IDs used in production."""
        text = text.strip()
        if not text:
            raise ValueError("TTS text must not be empty")
        from plugins.vits2_tts_trt.frontend.cleaner import normalize_text_mix
        from plugins.vits2_tts_trt.frontend.chunking import (
            iter_text_chunks as _frontend_iter_text_chunks,
        )

        normalized = (normalize_text_mix(text, reading_language=reading_language)
                      if reading_language else normalize_text_mix(text))
        for chunk in _frontend_iter_text_chunks(normalized, self._token_count, self.max_chunk_tokens):
            text_ids = self._engine._get_text_ids(chunk, normalized=True)
            yield chunk, text_ids

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def _isolated_pcm(self, chunk, text_ids, span, parts):
        """Synthesize a verified term while retaining its original left context.

        Token durations are estimates, not forced alignment. If their layout
        cannot be validated, return the unchanged whole-chunk audio.
        """
        start, end, blank = span
        scale = 1.0 / self._speed
        pcm, durations = self._engine.synthesize_with_durations(
            chunk, text_ids=text_ids, length_scale=scale,
        )
        samples = np.frombuffer(pcm, dtype='<i2')
        frame_count = sum(durations)
        if len(durations) != len(text_ids[0]) or frame_count <= 0:
            return pcm
        hop = round(len(samples) / frame_count)
        if not 32 <= hop <= 4096 or abs(len(samples) - hop * frame_count) > 8:
            return pcm
        prefix_text, suffix_text = chunk[:start], chunk[end:]
        if suffix_text.strip() and not any(c.isalnum() for c in suffix_text):
            # A question/exclamation needs the original sentence prosody.
            if any(c in suffix_text for c in '?!！？'):
                return pcm
            suffix_text = ''
        if prefix_text.strip():
            lo = sum(durations[:blank]) * hop
            hi = lo + durations[blank] * hop
            window = SAMPLE_RATE // 400
            if lo < 0 or hi > len(samples) or hi - lo <= 2 * window:
                return pcm
            cut = min(range(lo + window, hi - window),
                      key=lambda i: float(np.mean(samples[i-window:i+window].astype(np.float32) ** 2)))
            radius = SAMPLE_RATE // 1000
            crossings = [i for i in range(max(lo, cut-radius), min(hi-1, cut+radius))
                         if int(samples[i]) * int(samples[i+1]) <= 0]
            if crossings:
                cut = min(crossings, key=lambda i: abs(i-cut))
            prefix = samples[:cut]
            if len(prefix) <= SAMPLE_RATE // 500:
                return pcm
        else:
            prefix = np.empty(0, dtype=np.int16)

        # A short whole word can expand into many letter-name tokens. Check
        # each independent text against the engine profile before inference;
        # if splitting would exceed it, retain the original complete chunk.
        try:
            part_ids = [self._engine._get_text_ids(part, normalized=True) for part in parts]
            suffix_ids = (self._engine._get_text_ids(suffix_text, normalized=True)
                          if suffix_text.strip() else None)
        except (AssertionError, ValueError):
            return pcm
        engine_limit = int(getattr(self._engine, 'max_text_tokens', self.max_chunk_tokens))
        if (any(not ids[0] or len(ids[0]) > engine_limit for ids in part_ids)
                or (suffix_ids is not None
                    and (not suffix_ids[0] or len(suffix_ids[0]) > engine_limit))):
            return pcm

        # Isolated word/letter parts use the same request seed as an isolated
        # utterance. Restore the full-chunk RNG afterwards so later ordinary
        # chunks retain their original random stream.
        state = deepcopy(self._engine._rng.bit_generator.state)
        try:
            segments = []
            for part, ids in zip(parts, part_ids):
                self._engine.reset_random_state()
                segment = self._engine.synthesize(part, text_ids=ids, length_scale=scale)
                segments.append(np.frombuffer(segment, dtype='<i2'))
            suffix = np.empty(0, dtype=np.int16)
            if suffix_ids is not None:
                self._engine.reset_random_state()
                suffix_pcm = self._engine.synthesize(suffix_text, text_ids=suffix_ids, length_scale=scale)
                suffix = np.frombuffer(suffix_pcm, dtype='<i2')
        except SynthesisFrameLimitError:
            # The full chunk already succeeded; if an isolated part expands
            # beyond the frame profile, preserve that complete original audio.
            return pcm
        finally:
            self._engine._rng.bit_generator.state = state
        if not segments or any(len(segment) <= SAMPLE_RATE // 250 for segment in segments):
            return pcm
        output = prefix
        for index, segment in enumerate(segments):
            head = _quiet_samples(segment) if len(output) else 0
            tail = _quiet_samples(segment, reverse=True) if index + 1 < len(segments) or len(suffix) else 0
            segment = segment[head:len(segment)-tail if tail else None]
            if len(segment) <= SAMPLE_RATE // 250:
                return pcm
            if len(output) and len(output) <= SAMPLE_RATE // 500:
                return pcm
            output = _join_pcm(output, segment)
        if len(suffix):
            suffix = suffix[_quiet_samples(suffix):]
            if len(suffix) <= SAMPLE_RATE // 250:
                return pcm
            output = _join_pcm(output, suffix)
        return output.astype('<i2', copy=False).tobytes()

    def synthesize_stream(self, text: str):
        pause_samples = SAMPLE_RATE * CHUNK_PAUSE_MS // 1000
        silence = b"\x00\x00" * pause_samples
        with self._lock:
            # One deterministic latent stream per complete request. Resetting for
            # every text chunk would repeat the same noise prefix at boundaries.
            reset_random_state = getattr(self._engine, "reset_random_state", None)
            if reset_random_state is not None:
                reset_random_state()
            prepared = self.iter_text_chunks(text)
            isolation = None
            if (hasattr(self._engine, 'synthesize_with_durations')
                    and hasattr(self._engine, '_rng') and reset_random_state is not None):
                from .frontend.chunking import find_isolated_plan, map_isolated_term
                from .frontend.external_lexicon import isolated_synthesis_entries

                entries = isolated_synthesis_entries()
                plan = find_isolated_plan(text, entries) if entries else None
                if plan:
                    prepared = list(prepared)
                    term, mode, parts = plan
                    matches = [(index, span) for index, (chunk, ids) in enumerate(prepared)
                               if (span := map_isolated_term(
                                   chunk, ids, term, mode, getattr(self._engine, 'add_blank', False)))]
                    if len(matches) == 1:
                        isolation = (matches[0][0], matches[0][1], parts)
            for chunk_index, (chunk, text_ids) in enumerate(prepared):
                token_count = len(text_ids[0])
                log.info(
                    "text redacted: chars=%d chunk=%d tokens=%d",
                    len(text.strip()),
                    chunk_index,
                    token_count,
                )
                if chunk_index and silence:
                    yield silence
                if isolation and chunk_index == isolation[0]:
                    pcm = self._isolated_pcm(chunk, text_ids, isolation[1], isolation[2])
                else:
                    pcm = self._engine.synthesize(
                        chunk, text_ids=text_ids, length_scale=1.0 / self._speed,
                    )
                for offset in range(0, len(pcm), CHUNK_BYTES):
                    yield pcm[offset : offset + CHUNK_BYTES]

    def warmup(self) -> int:
        warmup_bytes = 0
        for text in WARMUP_CASES:
            case_bytes = sum(len(pcm) for pcm in self.synthesize_stream(text))
            if not case_bytes:
                raise RuntimeError("TensorRT warmup produced no audio")
            warmup_bytes += case_bytes
        return warmup_bytes


def _trt_adapter(cfg: dict, *, progress_cb=None, stage_cb=None) -> Vits2TensorRTAdapter:
    root = Path(cfg.get("model_dir", "/models/vits2"))
    _configure_frontend_paths(root, progress_cb=progress_cb, stage_cb=stage_cb)
    if stage_cb is not None:
        stage_cb("initialize")
    return Vits2TensorRTAdapter(
        speed=float(cfg.get("speed", 1.0)),
        model_config=str(root / "config.json"),
        engine_dir=cfg["engine_dir"],
        max_chunk_tokens=int(cfg.get("max_chunk_tokens", MAX_CHUNK_TOKENS)),
    )


def _configure_frontend_paths(root: Path, *, progress_cb=None, stage_cb=None) -> None:
    """Point inference at the release-controlled frontend data."""
    from .frontend.release_paths import configure_release_paths
    from .lexicon_resources import ensure_external_lexicon

    ensure_external_lexicon(root, progress_cb=progress_cb, stage_cb=stage_cb)
    configure_release_paths(root)


def build_adapter(cfg: dict, *, progress_cb=None, stage_cb=None) -> TTSAdapter:
    speaker_id = int(cfg.get("speaker_id", 0))
    if speaker_id != 0:
        raise ValueError("The VITS2 model supports only speaker_id=0")
    backend = str(cfg.get("backend", "trt")).lower()
    if backend != "trt":
        raise ValueError("The JP6 VITS2 plugin supports backend=trt only")
    return _trt_adapter(cfg, progress_cb=progress_cb, stage_cb=stage_cb)
