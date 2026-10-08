"""Offline X-ASR transducer adapter for the product ASR plugin."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
import tempfile
import threading
import unicodedata
import wave
from pathlib import Path


SAMPLE_RATE = 16000
TAIL_PADDING_SECONDS = 0.3
MAX_ACTIVE_PATHS = 3
HOTWORDS_SCORE = 2.5

# The native decoder reads this process-wide option during Decode, not construction.
_EARLY_HOTWORD_ENV = "SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE"
_DECODE_LOCK = threading.Lock()

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


def _boost_hotwords(encoded: Path, boosts: dict) -> Path:
    """Apply per-token scores to packaged CJK phrases without changing the source."""
    scores = {"".join(str(key).split()): float(value) for key, value in boosts.items()}
    source = encoded.read_bytes()
    digest = hashlib.sha256(source + repr(sorted(scores.items())).encode()).hexdigest()[:16]
    directory = Path('/tmp/asr_x_asr_hotwords')
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f'hotwords-boosted-{digest}.txt'
    if output.is_file():
        return output
    found, lines = set(), []
    for line in source.decode('utf-8').splitlines():
        phrase, separator, _ = line.rpartition(' :')
        compact = ''.join(phrase.split())
        if separator and compact in scores:
            line = f'{phrase} :{scores[compact]}'
            found.add(compact)
        lines.append(line+'\n')
    missing = set(scores)-found
    if missing:
        raise ValueError('Boosted CJK phrases are not packaged hotwords: '+', '.join(sorted(missing)))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.writelines(lines)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _custom_hotwords_file(encoded: Path, words: str) -> Path:
    """Merge user phrases; the recognizer applies its BPE vocabulary to these units."""
    additions = {}
    for phrase in words.splitlines():
        units, run = [], []
        for char in phrase.strip():
            cjk = '\u3400' <= char <= '\u4dbf' or '\u4e00' <= char <= '\u9fff'
            if cjk or char.isspace() or unicodedata.category(char).startswith(('P', 'S')):
                if run:
                    units.append(''.join(run))
                    run = []
                if cjk:
                    units.append(char)
            else:
                run.append(char)
        if run:
            units.append(''.join(run))
        if units:
            additions[' '.join(units)] = 4.0
    if not additions:
        return encoded
    lines = {}
    for line in encoded.read_text(encoding='utf-8').splitlines():
        phrase, sep, score = line.rpartition(' :')
        if sep:
            lines[phrase] = float(score)
    lines.update(additions)
    # A task-local generated file; never change the packaged vocabulary.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt',
                                     prefix='asr-hotwords-', delete=False) as stream:
        stream.writelines(f'{phrase} :{score}\n' for phrase, score in lines.items())
    return Path(stream.name)


class XASRAdapter:
    """Decode complete utterances with X-ASR and the packaged hotword list."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        num_threads: int = 2,
        max_active_paths: int = MAX_ACTIVE_PATHS,
        tail_padding_seconds: float = TAIL_PADDING_SECONDS,
        prefix_lm_path: str = "",
        prefix_lm_scale: float = 0.0,
        entity_boost: dict = None,
        custom_hotwords: str = "",
    ):
        from utils.onnx_provider import provider_for_device

        root = Path(model_dir)
        dtype = "" if device == "gpu" else ".int8"
        encoder = root / f"encoder-epoch-99-avg-1{dtype}.onnx"
        decoder = root / "decoder-epoch-99-avg-1.onnx"
        joiner = root / f"joiner-epoch-99-avg-1{dtype}.onnx"
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

        self._max_active_paths = int(max_active_paths)
        self._tail_padding_seconds = float(tail_padding_seconds)
        scale = float(prefix_lm_scale)
        if self._max_active_paths < 1 or self._tail_padding_seconds < 0 or not 0 <= scale <= 1:
            raise ValueError("Invalid X-ASR beam, tail padding or prefix LM scale")
        lm_options = {}
        if scale:
            if getattr(sherpa_onnx, "XASR_PREFIX_LM_VERSION", 0) != 1:
                raise RuntimeError("X-ASR prefix LM requires a prefix-enabled sherpa-onnx runtime")
            if not Path(prefix_lm_path).is_file():
                raise FileNotFoundError("X-ASR prefix LM model is missing")
            lm_options = {"lm": prefix_lm_path, "lm_scale": scale}
        provider = provider_for_device(device,
                                       (str(encoder), str(decoder), str(joiner)))
        encoded_hotwords = root / "hotwords.bpe.txt"
        if not encoded_hotwords.is_file():
            encoded_hotwords = _prepare_hotwords_file(hotwords)
        self._early_hotword_min_score = None
        if entity_boost:
            if not scale:
                raise ValueError('Early entity boosting requires prefix LM scoring')
            if getattr(sherpa_onnx, 'XASR_EARLY_HOTWORD_VERSION', 0) != 1:
                raise RuntimeError('Early entity boosting requires the full entity-enabled runtime')
            early_min = min(float(value) for value in entity_boost.values()) - 0.1
            if early_min <= HOTWORDS_SCORE:
                raise ValueError(f'Entity per-token scores must exceed {HOTWORDS_SCORE + 0.1}')
            encoded_hotwords = _boost_hotwords(encoded_hotwords, entity_boost)
            self._early_hotword_min_score = f'{early_min:g}'
        custom_file = _custom_hotwords_file(encoded_hotwords, custom_hotwords)
        if custom_file != encoded_hotwords:
            if not scale or getattr(sherpa_onnx, "XASR_EARLY_HOTWORD_VERSION", 0) != 1:
                custom_file.unlink()
                raise RuntimeError("自定义热词需要支持上下文辅助和热词增强的运行库")
            self._early_hotword_min_score = "3.9"
        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                num_threads=int(num_threads),
                provider=provider,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="modified_beam_search",
                max_active_paths=self._max_active_paths,
                hotwords_file=str(custom_file),
                hotwords_score=HOTWORDS_SCORE,
                modeling_unit="bpe",
                bpe_vocab=str(bpe_vocab),
                **lm_options,
            )
        finally:
            if custom_file != encoded_hotwords:
                custom_file.unlink(missing_ok=True)
        log.info(
            "[asr] X-ASR adapter loaded: encoder=%s, device=%s, provider=%s, "
            "max_active_paths=%d, hotwords_score=%.1f, tail_padding=%.2fs, prefix_lm_scale=%.3f",
            encoder,
            device,
            provider,
            self._max_active_paths,
            HOTWORDS_SCORE,
            self._tail_padding_seconds,
            scale,
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
        samples.extend([0.0] * int(sample_rate * self._tail_padding_seconds))

        # Serialize all X-ASR adapters (plugin and WebSocket), not just this instance.
        with _DECODE_LOCK:
            previous = os.environ.get(_EARLY_HOTWORD_ENV)
            try:
                if self._early_hotword_min_score is None:
                    os.environ.pop(_EARLY_HOTWORD_ENV, None)
                else:
                    os.environ[_EARLY_HOTWORD_ENV] = self._early_hotword_min_score
                stream = self._recognizer.create_stream()
                stream.accept_waveform(sample_rate, samples)
                self._recognizer.decode_streams([stream])
                result = stream.result
            finally:
                if previous is None:
                    os.environ.pop(_EARLY_HOTWORD_ENV, None)
                else:
                    os.environ[_EARLY_HOTWORD_ENV] = previous
        return str(getattr(result, "text", result or "")).strip()
