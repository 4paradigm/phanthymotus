"""Official Paraformer adapter for the default ASR plugin."""

from __future__ import annotations

from __future__ import annotations

import io
import json
import logging
import threading
import wave
from pathlib import Path

from plugins.asr_runtime import pcm16_to_float_samples


SAMPLE_RATE = 16000
log = logging.getLogger(__name__)

# Product vocabulary missing from the downloaded 234-term robot-action list.
# Keep this in code so every image gets the same additions without mutating or
# checking model artifacts into Git.
_DOMAIN_HOTWORDS = (
    "进入零力矩模式",
    "进入阻尼模式",
    "飞吻",
    "来个飞吻",
    "大疆",
    "大疆创新",
    "仙元路",
    "大疆天空之城",
    "高举你的双手",
    "双手打叉",
)


def _resolve_model_file(model_root: Path, configured_model: str) -> Path:
    if configured_model:
        configured_path = Path(configured_model)
        candidate = (
            configured_path
            if configured_path.is_absolute()
            else model_root / configured_path
        )
        if candidate.is_file():
            return candidate.resolve()

    preferred = model_root / "model.int8.onnx"
    if preferred.is_file():
        return preferred.resolve()

    onnx_files = sorted(model_root.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX model found in {model_root}")
    return onnx_files[0].resolve()


def _find_transducer_files(model_root: Path) -> tuple[str, str, str] | None:
    """检测 transducer 三件套（encoder/decoder/joiner），存在则返回路径，优先 int8。"""

    def _pick(prefix: str) -> str:
        files = sorted(model_root.glob(f"{prefix}*.onnx"))
        if not files:
            return ""
        int8 = [f for f in files if "int8" in f.name]
        return str((int8[0] if int8 else files[0]).resolve())

    encoder = _pick("encoder")
    decoder = _pick("decoder")
    joiner = _pick("joiner")
    if encoder and decoder and joiner:
        return encoder, decoder, joiner
    return None


def _prepare_hotwords_file(
    hotwords_path: Path, tmp_dir: Path
) -> tuple[Path, str]:
    """把热词原文转成 sherpa bpe 模式可编码的逐字空格格式。

    x-asr 的 tokens.txt 4000 个中文 token 全是 ▁X（▁+单字），没有裸字符。
    bpe.model 整词编码会产出 tokens.txt 里没有的多字 piece（如 "尼模式"），
    sherpa 直接丢弃。逐字喂入让 bpe 把每个字编码成 ▁X，230/234 可编码。

    输入原文（hotwords.txt，每行一个短语）：
        阻尼模式
        举双手
    输出（char-separated + :score）：
        阻 尼 模 式 :2.0
        举 双 手 :2.0

    返回 (转换后文件路径, modeling_unit)。
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "hotwords_char_bpe.txt"
    with hotwords_path.open("r", encoding="utf-8") as src:
        phrases = [line.strip() for line in src]

    seen: set[str] = set()
    with out_path.open("w", encoding="utf-8") as dst:
        for phrase in (*phrases, *_DOMAIN_HOTWORDS):
            if not phrase or phrase.startswith("#"):
                continue
            compact = phrase.replace(" ", "")
            if compact in seen:
                continue
            seen.add(compact)
            dst.write(" ".join(compact) + " :2.0\n")
    return out_path, "bpe"


def _create_sherpa_recognizer(
    model_path: str,
    config: dict,
    sample_rate: int = SAMPLE_RATE,
    bits_per_sample: int = 16,
):
    del bits_per_sample
    import sherpa_onnx

    root = Path(model_path)
    model_root = root if root.is_dir() else root.parent

    configured_tokens = Path(config.get("tokens", "tokens.txt"))
    tokens = (
        configured_tokens
        if configured_tokens.is_absolute()
        else model_root / configured_tokens
    ).resolve()
    if not tokens.is_file():
        raise FileNotFoundError(f"Token file not found: {tokens}")

    recognizer_config = config.get("recognizerConfig", {})
    common_kwargs = {
        "tokens": str(tokens),
        "num_threads": int(config.get("numThreads", 1)),
        "sample_rate": int(sample_rate),
        "feature_dim": int(config.get("featureConfig", {}).get("featureDim", 80)),
        "decoding_method": recognizer_config.get(
            "decodingMethod", "greedy_search"
        ),
        "debug": bool(config.get("debug", False)),
        "provider": config.get("provider", "cpu"),
    }

    # Transducer 模型（encoder/decoder/joiner 三件套，如 x-asr-punct）优先
    transducer_files = _find_transducer_files(model_root)
    if transducer_files:
        encoder, decoder, joiner = transducer_files
        method = common_kwargs["decoding_method"]
        log.info(
            f"[asr-offline] transducer model detected: {model_root} "
            f"(decoding={method})"
        )
        transducer_kwargs = dict(common_kwargs)
        # modified_beam_search 需显式传 max_active_paths（默认 4），
        # 允许 config 覆盖以启用更宽的 beam 配置。
        if method != "greedy_search":
            max_active = int(recognizer_config.get("maxActivePaths", 4))
            transducer_kwargs["max_active_paths"] = max_active

        # hotwords 偏置：仅在 modified_beam_search 下生效（greedy 不走 context
        # graph）。x-asr 的 tokens.txt 全是 ▁X 单字 token，整词 BPE 编码会失败，
        # 所以把 hotwordsFile 原文转成逐字空格格式让 bpe 逐字编码。
        hotwords_file = recognizer_config.get("hotwordsFile")
        if hotwords_file and method != "greedy_search":
            hw_path = Path(hotwords_file)
            if not hw_path.is_absolute():
                hw_path = model_root / hw_path
            hw_path = hw_path.resolve()
            if hw_path.is_file():
                tmp_dir = Path("/tmp/asr_hotwords") if Path("/tmp").is_dir() else model_root.parent
                encoded_path, modeling_unit = _prepare_hotwords_file(hw_path, tmp_dir)
                transducer_kwargs["hotwords_file"] = str(encoded_path)
                transducer_kwargs["hotwords_score"] = float(
                    recognizer_config.get("hotwordsScore", 2.0)
                )
                transducer_kwargs["modeling_unit"] = modeling_unit
                bpe_vocab = recognizer_config.get("bpeVocab")
                if bpe_vocab:
                    bpe_path = Path(bpe_vocab)
                    if not bpe_path.is_absolute():
                        bpe_path = model_root / bpe_path
                    transducer_kwargs["bpe_vocab"] = str(bpe_path.resolve())
                log.info(
                    f"[asr-offline] hotwords enabled: {hw_path} "
                    f"-> {encoded_path} (score={transducer_kwargs['hotwords_score']})"
                )
            else:
                log.warning(f"[asr-offline] hotwordsFile not found: {hw_path}, skip")

        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            **transducer_kwargs,
        )

    # Paraformer 单模型文件
    configured_model = config.get("model") or config.get("model_path") or ""
    model_file = _resolve_model_file(model_root, configured_model or str(root))
    kwargs = {
        **common_kwargs,
        "paraformer": str(model_file),
        "rule_fsts": recognizer_config.get("ruleFsts", ""),
        "rule_fars": recognizer_config.get("ruleFars", ""),
    }
    try:
        return sherpa_onnx.OfflineRecognizer.from_paraformer(**kwargs)
    except TypeError:
        kwargs.pop("rule_fsts")
        kwargs.pop("rule_fars")
        return sherpa_onnx.OfflineRecognizer.from_paraformer(**kwargs)


class OfflineASRAdapter:
    """Decode complete WAV utterances with a cached offline recognizer."""

    _instances: dict[tuple[str, int, str], "OfflineASRAdapter"] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str,
        config: dict | None = None,
        num_threads: int = 1,
        provider: str = "cpu",
    ):
        model_root = Path(model_path)
        loaded_config: dict = {}
        config_path = model_root / "config.json"
        if config is not None:
            loaded_config = dict(config)
        elif config_path.is_file():
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw_config.get("sherpa"), dict):
                loaded_config = dict(raw_config["sherpa"])
                loaded_config.setdefault("model", raw_config.get("model_path", ""))
            else:
                loaded_config = dict(raw_config)

        loaded_config["numThreads"] = int(num_threads)
        loaded_config["provider"] = provider
        self._recognizer = _create_sherpa_recognizer(
            model_path, loaded_config, sample_rate=SAMPLE_RATE
        )
        self._decode_lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        model_path: str,
        config: dict | None = None,
        num_threads: int = 1,
        provider: str = "cpu",
    ) -> "OfflineASRAdapter":
        key = (str(Path(model_path).resolve()), int(num_threads), provider)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(
                    model_path,
                    config=config,
                    num_threads=num_threads,
                    provider=provider,
                )
            return cls._instances[key]

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._instances.clear()

    def transcribe(self, wav_bytes: bytes, language: str = "zh-CN") -> str:
        del language
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("Offline ASR expects mono 16-bit PCM WAV")
            sample_rate = wav_file.getframerate()
            raw_pcm = wav_file.readframes(wav_file.getnframes())

        samples = list(pcm16_to_float_samples(raw_pcm))
        # 0.5s tail padding: critical for transducer to decode final tokens
        # (without this, short utterance ends can regress to wrong language/output).
        # param_experiment.py verified: mbs(5)+hotwords ARM 1-CER 0.8388 with pad,
        # vs 0.7130/degraded-to-English without.  [[arm-int8-drift]]
        samples.extend([0.0] * int(sample_rate * 0.5))
        with self._decode_lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self._recognizer.decode_stream(stream)

            result = getattr(stream, "result", None)
            if result is None and hasattr(self._recognizer, "get_result"):
                result = self._recognizer.get_result(stream)
            text = getattr(result, "text", result or "")
        return str(text).strip()
