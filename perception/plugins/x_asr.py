"""Offline X-ASR transducer adapter for the product ASR plugin."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
import tempfile
import threading
import wave
from pathlib import Path


SAMPLE_RATE = 16000
TAIL_PADDING_SECONDS = 0.5
MAX_ACTIVE_PATHS = 10
HOTWORDS_SCORE = 2.5

log = logging.getLogger(__name__)


def _prepare_hotwords_file(
    source: Path,
    score: float = HOTWORDS_SCORE,
    output_dir: Path = Path("/tmp/asr_x_asr_hotwords"),
) -> Path:
    """Convert one phrase per line to sherpa's character-separated BPE form."""
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(
        source_bytes + b"\0" + str(float(score)).encode("ascii")
    ).hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"hotwords-char-bpe-{digest}.txt"
    if output.is_file():
        return output

    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in source_bytes.decode("utf-8").splitlines():
        phrase = raw_line.strip()
        if not phrase or phrase.startswith("#"):
            continue
        compact = "".join(phrase.split())
        if not compact or compact in seen:
            continue
        seen.add(compact)
        lines.append(f"{' '.join(compact)} :{float(score)}\n")

    if not lines:
        raise ValueError(f"No usable hotwords in {source}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{output.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


class XASRAdapter:
    """Decode complete utterances with X-ASR and the packaged hotword list."""

    def __init__(
        self,
        model_dir: str,
        hw_provider: str = "cpu",
        num_threads: int = 2,
    ):
        root = Path(model_dir)
        encoder = root / "encoder-epoch-99-avg-1.int8.onnx"
        decoder = root / "decoder-epoch-99-avg-1.onnx"
        joiner = root / "joiner-epoch-99-avg-1.int8.onnx"
        tokens = root / "tokens.txt"
        bpe_model = root / "bpe.model"
        bpe_vocab = root / "bpe.vocab"
        hotwords = root / "hotwords.txt"
        required = (
            encoder,
            decoder,
            joiner,
            tokens,
            bpe_model,
            bpe_vocab,
            hotwords,
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"X-ASR model bundle is incomplete at {root}: {', '.join(missing)}"
            )

        import sherpa_onnx

        encoded_hotwords = _prepare_hotwords_file(hotwords)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=int(num_threads),
            provider=hw_provider,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=MAX_ACTIVE_PATHS,
            hotwords_file=str(encoded_hotwords),
            hotwords_score=HOTWORDS_SCORE,
            modeling_unit="bpe",
            bpe_vocab=str(bpe_vocab),
        )
        self._decode_lock = threading.Lock()
        log.info(
            "[asr] X-ASR adapter loaded: encoder=%s, provider=%s, "
            "max_active_paths=%d, hotwords_score=%.1f",
            encoder,
            hw_provider,
            MAX_ACTIVE_PATHS,
            HOTWORDS_SCORE,
        )

    def transcribe(self, wav_bytes: bytes, language: str) -> str:
        del language
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("X-ASR expects mono 16-bit PCM WAV")
            sample_rate = wav_file.getframerate()
            pcm = wav_file.readframes(wav_file.getnframes())

        sample_count = len(pcm) // 2
        samples = [
            sample / 32768.0
            for sample in struct.unpack(f"<{sample_count}h", pcm)
        ]
        samples.extend([0.0] * int(sample_rate * TAIL_PADDING_SECONDS))

        with self._decode_lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self._recognizer.decode_streams([stream])
            result = stream.result
        return str(getattr(result, "text", result or "")).strip()
