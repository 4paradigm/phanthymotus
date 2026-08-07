"""Adapter between the shared ROS2 TTS plugin and the VITS2 CPU engine."""

from __future__ import annotations

import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from .onnx_cpu_engine import OnnxCpuEngine


SAMPLE_RATE = 16000
CHUNK_BYTES = 3200
log = logging.getLogger(__name__)


class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)

    def warmup(self) -> int:
        return 0


class Vits2OnnxCpuAdapter(TTSAdapter):
    def __init__(
        self,
        model_dir: str,
        speed: float = 1.0,
        num_threads: int = 1,
        max_chunk_tokens: int = 64,
    ):
        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero")

        root = Path(model_dir)
        package_dir = Path(__file__).resolve().parent
        os.environ["NLTK_DATA"] = str(root / "nltk_data")
        os.environ["EN_TN_CACHE_DIR"] = str(root / "tn_cache")
        os.environ["TN_CACHE_DIR"] = str(root / "tn_cache")
        os.environ["VITS2_FRONTEND_DATA_DIR"] = str(root / "frontend_data")

        config_path = package_dir / "config.json"
        self._engine = OnnxCpuEngine(
            config_path=config_path,
            model_dir=root / "onnx",
            num_threads=num_threads,
        )
        if self._engine.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"VITS2 sample rate must be {SAMPLE_RATE}, got {self._engine.sample_rate}"
            )
        self._max_chunk_tokens = max(
            16, min(int(max_chunk_tokens), self._engine.max_text_tokens)
        )
        self._length_scale = 1.0 / speed
        self._lock = threading.Lock()

    def warmup(self) -> int:
        with self._lock:
            pcm = self._engine.synthesize(
                "你好。", length_scale=self._length_scale
            )
        if not pcm:
            raise RuntimeError("VITS2 warmup produced no audio")
        return len(pcm)

    def _split_text(self, text: str) -> list[str]:
        limit = self._max_chunk_tokens
        units = re.findall(r".*?[。！？!?；;\n]+|.+$", text, flags=re.DOTALL)
        chunks = []
        current = ""

        for unit in units:
            candidate = current + unit
            if self._engine.text_token_count(candidate) <= limit:
                current = candidate
                continue
            if current.strip():
                chunks.append(current.strip())
                current = ""

            remainder = unit.strip()
            while remainder:
                if self._engine.text_token_count(remainder) <= limit:
                    current = remainder
                    break
                low, high = 1, len(remainder)
                while low < high:
                    middle = (low + high + 1) // 2
                    if self._engine.text_token_count(remainder[:middle]) <= limit:
                        low = middle
                    else:
                        high = middle - 1
                if low < 1:
                    raise ValueError("Unable to split text within the token limit")
                chunks.append(remainder[:low].strip())
                remainder = remainder[low:].strip()

        if current.strip():
            chunks.append(current.strip())
        token_counts = [self._engine.text_token_count(chunk) for chunk in chunks]
        log.info(
            "[vits2_tts] text redacted: tokens=%d chunks=%d chunk_tokens=%s",
            sum(token_counts),
            len(chunks),
            token_counts,
        )
        return chunks

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        with self._lock:
            chunks = self._split_text(text)
            silence = b"\x00\x00" * (SAMPLE_RATE // 10)
            audio = [
                self._engine.synthesize(chunk, length_scale=self._length_scale)
                for chunk in chunks
            ]
            return silence.join(audio)

    def synthesize_stream(self, text: str):
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        with self._lock:
            chunks = self._split_text(text)
            silence = b"\x00\x00" * (SAMPLE_RATE // 10)
            for chunk_index, chunk in enumerate(chunks):
                if chunk_index:
                    yield silence
                pcm = self._engine.synthesize(
                    chunk, length_scale=self._length_scale
                )
                for offset in range(0, len(pcm), CHUNK_BYTES):
                    yield pcm[offset:offset + CHUNK_BYTES]


def build_adapter(cfg: dict) -> TTSAdapter:
    speaker_id = int(cfg.get("speaker_id", 0))
    if speaker_id != 0:
        raise ValueError("The VITS2 model supports only speaker_id=0")
    return Vits2OnnxCpuAdapter(
        model_dir=cfg.get("vits2_model_dir", "/models/vits2-mix"),
        speed=float(cfg.get("speed", 1.0)),
        num_threads=max(1, int(cfg.get("vits2_num_threads", 1))),
        max_chunk_tokens=int(cfg.get("vits2_max_chunk_tokens", 64)),
    )
