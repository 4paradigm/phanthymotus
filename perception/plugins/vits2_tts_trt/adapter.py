"""In-process adapter for the JetPack 6 VITS2 TensorRT runtime."""

from __future__ import annotations

import logging
import os
import re
import threading
from abc import ABC, abstractmethod

import jieba

from .runtime.backends.trt_numpy_tts_engine import TensorRTNumpyTTSEngine


SAMPLE_RATE = 16000
CHUNK_BYTES = int(os.getenv("MIX_VITS_CHUNK_BYTES", "6400"))
MAX_CHUNK_TOKENS = int(os.getenv("MIX_VITS_MAX_TEXT_TOKENS", "64"))
CHUNK_PAUSE_MS = int(os.getenv("MIX_VITS_CHUNK_PAUSE_MS", "0"))
MODEL_CONFIG = os.getenv("MIX_VITS_CONFIG_PATH", "/models/vits2-mix/config.json")
ENGINE_DIR = os.getenv("MIX_VITS_TRT_ENGINE_DIR", "/models/vits2-mix/engines")
WARMUP_CASES = (
    "你好。",
    "晚上我用FaceTime和家人视频聊天。",
    "周末我拍了一张selfie发给朋友。",
    "Lucy今天去公园散步并喝coffee，David开会前仔细检查PPT。",
)
log = logging.getLogger(__name__)

if CHUNK_BYTES <= 0 or CHUNK_BYTES % 2:
    raise ValueError("MIX_VITS_CHUNK_BYTES must be a positive even number")
if not 0 <= CHUNK_PAUSE_MS <= 1000:
    raise ValueError("MIX_VITS_CHUNK_PAUSE_MS must be between 0 and 1000")

_ZH_NUMBER_UNIT_RE = re.compile(
    r"[零〇一二两三四五六七八九十百千万亿点]+"
    r"(?:K\s*B|M\s*B|G\s*B|T\s*B|P\s*B)(?:每[A-Za-z])?",
    re.IGNORECASE,
)
_SPLIT_PUNCTUATION = "，,、。．.｡！!？?；;：:…⋯—–~～\n\r"
_CLOSING_PUNCTUATION = "”’\"'」』》〉】〕〗〙〛）)]｝}"
_PUNCTUATION_UNIT_RE = re.compile(
    rf".*?[{re.escape(_SPLIT_PUNCTUATION)}]+"
    rf"[{re.escape(_CLOSING_PUNCTUATION)}]*|.+$",
    flags=re.DOTALL,
)


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


def _language_kind(char: str) -> str | None:
    if "\u4e00" <= char <= "\u9fff":
        return "ZH"
    if char.isascii() and char.isalnum():
        return "EN"
    return None


def _split_positions(text: str) -> tuple[list[int], list[int], list[int]]:
    protected = [match.span() for match in _ZH_NUMBER_UNIT_RE.finditer(text)]

    def is_safe(index: int) -> bool:
        return not any(start < index < end for start, end in protected)

    boundaries = []
    previous_kind = None
    for index, char in enumerate(text):
        kind = _language_kind(char)
        if kind is None:
            continue
        if previous_kind is not None and kind != previous_kind:
            if is_safe(index):
                boundaries.append(index)
        previous_kind = kind
    usable = [index for index in boundaries if 1 < index < len(text) - 1]
    word_boundaries = []
    offset = 0
    for word in jieba.cut(text):
        offset += len(word)
        if 1 < offset < len(text) - 1 and is_safe(offset):
            word_boundaries.append(offset)
    fallback = [index for index in range(1, len(text)) if is_safe(index)]
    if not fallback:
        raise ValueError("Unable to split protected number-unit expression")
    return usable, word_boundaries, fallback


class Vits2TensorRTAdapter(TTSAdapter):
    def __init__(self, speed: float = 1.0):
        self._lock = threading.Lock()
        self.set_speed(speed)
        self._engine = TensorRTNumpyTTSEngine(MODEL_CONFIG, ENGINE_DIR)

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if speed <= 0 or speed > 4:
            raise ValueError("TTS speed must be greater than zero and at most four")
        with self._lock:
            self._speed = speed

    def _iter_unit_chunks(self, text: str):
        text_ids = self._engine._get_text_ids(text, normalized=True)
        if len(text_ids[0]) <= MAX_CHUNK_TOKENS:
            yield text, text_ids
            return
        if len(text) <= 1:
            raise ValueError("Unable to split text within TensorRT profile")
        language_boundaries, word_boundaries, fallback = _split_positions(text)

        def longest_fitting(positions):
            low, high = 0, len(positions) - 1
            best = None
            while low <= high:
                middle = (low + high) // 2
                position = positions[middle]
                prefix_ids = self._engine._get_text_ids(
                    text[:position], normalized=True
                )
                if len(prefix_ids[0]) <= MAX_CHUNK_TOKENS:
                    best = position
                    low = middle + 1
                else:
                    high = middle - 1
            return best

        split_at = longest_fitting(language_boundaries)
        if split_at is None:
            split_at = longest_fitting(word_boundaries)
        if split_at is None:
            log.warning(
                "no punctuation, language, or word boundary fits TensorRT profile; "
                "using forced split: chars=%d",
                len(text),
            )
            split_at = longest_fitting(fallback)
        if split_at is None:
            raise ValueError("Unable to split text within TensorRT profile")
        left, right = text[:split_at], text[split_at:]
        if not left.strip() or not right.strip():
            raise ValueError("Unable to split text within TensorRT profile")
        yield from self._iter_unit_chunks(left)
        yield from self._iter_unit_chunks(right)

    def _iter_text_chunks(self, text: str):
        # Make every natural punctuation unit fit first. Greedy packing below
        # only joins already-valid units and never re-splits a merged unit.
        units = _PUNCTUATION_UNIT_RE.findall(text)
        chunks = []
        for unit in units:
            if not unit.strip():
                continue
            chunks.extend(self._iter_unit_chunks(unit))

        pending_text = ""
        pending_ids = None
        for chunk, text_ids in chunks:
            if not pending_text:
                pending_text, pending_ids = chunk, text_ids
                continue

            combined = pending_text + chunk
            combined_ids = self._engine._get_text_ids(combined, normalized=True)
            if len(combined_ids[0]) <= MAX_CHUNK_TOKENS:
                pending_text, pending_ids = combined, combined_ids
                continue

            yield pending_text, pending_ids
            pending_text, pending_ids = chunk, text_ids

        if pending_text:
            yield pending_text, pending_ids

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        text = text.strip()
        if not text:
            raise ValueError("TTS text must not be empty")
        from .runtime.frontend.cleaner import normalize_text_mix

        text = normalize_text_mix(text)
        pause_samples = SAMPLE_RATE * CHUNK_PAUSE_MS // 1000
        silence = b"\x00\x00" * pause_samples
        with self._lock:
            for chunk_index, (chunk, text_ids) in enumerate(
                self._iter_text_chunks(text)
            ):
                token_count = len(text_ids[0])
                log.info(
                    "text redacted: chars=%d chunk=%d tokens=%d",
                    len(text),
                    chunk_index,
                    token_count,
                )
                if chunk_index and silence:
                    yield silence
                pcm = self._engine.synthesize(
                    chunk,
                    text_ids=text_ids,
                    length_scale=1.0 / self._speed,
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


def build_adapter(cfg: dict) -> TTSAdapter:
    speaker_id = int(cfg.get("speaker_id", 0))
    if speaker_id != 0:
        raise ValueError("The VITS2 model supports only speaker_id=0")
    return Vits2TensorRTAdapter(speed=float(cfg.get("speed", 1.0)))
