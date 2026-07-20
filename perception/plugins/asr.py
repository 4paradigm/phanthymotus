#!/usr/bin/env python3
"""
plugins/asr.py — ASRPlugin: sherpa-onnx VAD + KWS + ASR pipeline.

Pipeline: Audio → ONNX VAD → KWS (wake word gate) → ASR transcription
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import struct
import threading
import time
import wave
from abc import ABC, abstractmethod
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

log = logging.getLogger(__name__)

SAMPLE_RATE    = 16000
SPEECH_THRESH  = 0.5
SILENCE_THRESH = 0.35
SILENCE_FRAMES = 16

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)


# ── IPA phoneme matching for asr_kws mode ────────────────────────────────────

def _text_to_ipa(text: str) -> list:
    """Convert text to IPA phoneme sequence using phonemizer (espeak-ng backend).
    Returns a list of IPA phoneme strings (one per word/character).
    """
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    # Separate Chinese and non-Chinese segments
    segments = []
    current = ''
    current_is_cjk = None
    for char in text:
        is_cjk = '\u4e00' <= char <= '\u9fff'
        if current_is_cjk is None:
            current_is_cjk = is_cjk
        if is_cjk != current_is_cjk:
            if current.strip():
                segments.append((current.strip(), current_is_cjk))
            current = ''
            current_is_cjk = is_cjk
        current += char
    if current.strip():
        segments.append((current.strip(), current_is_cjk))

    ipa_seq = []
    sep = Separator(phone=' ', word='  ', syllable='')
    for seg_text, is_cjk in segments:
        lang = 'cmn' if is_cjk else 'en-us'
        try:
            ipa = phonemize(seg_text, language=lang, backend='espeak',
                           separator=sep, strip=True,
                           with_stress=False, tie=False,
                           language_switch='remove-flags')
            # Remove tone numbers and diacritics for fuzzy matching
            import re
            ipa = re.sub(r'[0-9˥˦˧˨˩¹²³⁴⁵]', '', ipa)
            phones = [p for p in ipa.split() if p]
            ipa_seq.extend(phones)
        except Exception:
            # Fallback: use characters as-is
            ipa_seq.extend(list(seg_text))
    return ipa_seq


def _phoneme_edit_distance(seq1: list, seq2: list) -> float:
    """Normalized edit distance with phoneme similarity (0=match, 1=different)."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = _phoneme_sub_cost(seq1[i - 1], seq2[j - 1])
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / max(m, n)


# Similar phoneme groups — substitution cost 0.3 instead of 1.0
_SIMILAR_GROUPS = [
    {'t', 'd'},       # alveolar stops
    {'p', 'b'},       # bilabial stops
    {'k', 'g'},       # velar stops
    {'f', 'v'},       # labiodental fricatives
    {'s', 'z'},       # alveolar fricatives
    {'s.', 'z.'},     # retroflex fricatives
    {'ɕ', 'ʃ', 'ʂ'}, # postalveolar/retroflex sibilants
    {'tsh', 'dz'},    # affricates
    {'n', 'ŋ'},       # nasals
    {'l', 'r', 'ɹ'},  # liquids
    {'t', 'tsh'},     # stop ~ affricate
    {'f', 't'},       # common confusion in noisy env
    {'x', 'h'},       # velar/glottal fricatives
    {'ɑu', 'au', 'ɑo', 'ao'},  # diphthong variants
    {'ou', 'uo'},     # vowel variants
    {'i', 'i.'},      # apical vowel variant
]


def _phoneme_sub_cost(a: str, b: str) -> float:
    """Substitution cost: 0 if same, 0.3 if similar, 1.0 otherwise."""
    if a == b:
        return 0
    for group in _SIMILAR_GROUPS:
        if a in group and b in group:
            return 0.3
    return 1.0


def _find_keyword_in_ipa(text_ipa: list, keyword_ipa: list, threshold: float):
    """Sliding window search for keyword in text IPA. Returns (matched, end_position)."""
    kw_len = len(keyword_ipa)
    if kw_len == 0 or len(text_ipa) < kw_len:
        return False, -1

    best_dist = float('inf')
    best_end = -1
    for i in range(len(text_ipa) - kw_len + 1):
        window = text_ipa[i:i + kw_len]
        dist = _phoneme_edit_distance(window, keyword_ipa)
        if dist < best_dist:
            best_dist = dist
            best_end = i + kw_len
    return best_dist <= threshold, best_end


def _extract_after_keyword(text: str, keyword_text: str, end_pos: int) -> str:
    """Extract text after the matched keyword.
    Uses keyword text length to determine how many characters to skip,
    then handles the case where ASR text has slightly different char count.
    """
    # Count phoneme-producing characters in original text up to end_pos
    # Simpler approach: use the keyword character length as skip count
    kw_chars = len([c for c in keyword_text if '\u4e00' <= c <= '\u9fff' or c.isalpha()])

    # Skip that many phoneme-producing characters in text
    skipped = 0
    cut_idx = 0
    for i, char in enumerate(text):
        if '\u4e00' <= char <= '\u9fff' or char.isalpha():
            skipped += 1
        if skipped >= kw_chars:
            cut_idx = i + 1
            break

    if cut_idx == 0:
        return ''
    remaining = text[cut_idx:]
    remaining = remaining.lstrip('，。！？、；：,.!?;: ')
    return remaining

_ASR_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "asr",
        "type": "processor",
        "multiInstance": True,
        "description": "ASR — start/stop speech recognition or get status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 audio topic (e.g. /hostname/mic/audio, required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "asr_model":     {"type": "string", "enum": ["paraformer-zh-en", "paraformer-offline", "zipformer-en", "sensevoice-small"], "description": "ASR model (paraformer-zh-en = bilingual streaming, paraformer-offline = bilingual offline, zipformer-en = English streaming, sensevoice-small = multilingual offline)", "default": "sensevoice-small", "scope": "shared"},
                "trigger_mode":  {"type": "string", "enum": ["vad", "kws", "asr_kws"], "description": "Trigger mode (vad = always listen, kws = KWS model, asr_kws = ASR + phoneme matching)", "default": "kws", "scope": "shared"},
                "kws_model":     {"type": "string", "enum": ["zh", "en", "zh-en"], "description": "KWS 模型 (zh=纯中文, en=纯英文, zh-en=双语)", "default": "zh", "scope": "shared", "x-show-when": {"trigger_mode": "kws"}},
                "kws_keywords":  {"type": "string", "description": "Wake word (zh: 'f àn sh ì x iǎo g ǒu @范式小狗', en: '▁FA N C Y ▁RO B O T @FANCY_ROBOT')", "scope": "shared", "x-show-when": {"trigger_mode": "kws"}},
                "asr_kws_keyword": {"type": "string", "description": "唤醒词文本（如'范式小狗'、'hello robot'）", "scope": "shared", "x-show-when": {"trigger_mode": "asr_kws"}},
                "asr_kws_threshold": {"type": "number", "description": "音素匹配阈值（0-1，越小越严格，推荐0.3）", "default": 0.3, "scope": "shared", "x-show-when": {"trigger_mode": "asr_kws"}},
                "vad_threshold": {"type": "number", "description": "VAD speech threshold (0-1, higher = stricter)", "default": 0.5, "scope": "shared"},
                "vad_silence_ms":{"type": "integer", "description": "Silence duration (ms) before sentence end", "default": 400, "scope": "shared"},
                "save_vad_segments": {"type": "boolean", "description": "Save VAD segments as WAV to /opt/embodied/models/vad_segments/", "default": False, "scope": "shared"},
                "max_saved_segments": {"type": "integer", "description": "Max saved VAD segments (oldest deleted when exceeded)", "default": 1000, "scope": "shared"},
            },
            "required": []
        },
        "topic_in":  [{"format": "audio/pcm-16k", "desc": "mic audio input"}],
        "topic_out": [{"format": "data/json",     "desc": "ASR result event"}],
    }
]


# ── WAV helper ────────────────────────────────────────────────────────────────

def _pcm16_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate); w.writeframes(pcm)
    return buf.getvalue()



# ── ASR Adapters ──────────────────────────────────────────────────────────────

class ASRAdapter(ABC):
    @abstractmethod
    def transcribe(self, wav_bytes: bytes, language: str) -> str: ...


class SherpaOnnxASRAdapter(ASRAdapter):
    """On-device streaming ASR using sherpa-onnx paraformer (no network required)."""

    def __init__(self, model_dir: str, hw_provider: str = "cuda", num_threads: int = 2):
        from utils.model_downloader import ensure_model
        ensure_model("asr", model_dir)

        import sherpa_onnx
        # Streaming paraformer uses encoder + decoder (not a single model file)
        encoder_path = os.path.join(model_dir, "encoder.int8.onnx")
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join(model_dir, "encoder.onnx")
        decoder_path = os.path.join(model_dir, "decoder.int8.onnx")
        if not os.path.exists(decoder_path):
            decoder_path = os.path.join(model_dir, "decoder.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=hw_provider,
            sample_rate=SAMPLE_RATE,
            decoding_method="greedy_search",
        )
        log.info(f"[asr] sherpa-onnx paraformer adapter loaded: encoder={encoder_path}, provider={hw_provider}")

    def transcribe(self, wav_bytes: bytes, language: str) -> str:
        import io as _io, wave as _wave
        with _wave.open(_io.BytesIO(wav_bytes)) as wf:
            pcm = wf.readframes(wf.getnframes())
        n = len(pcm) // 2
        samples = struct.unpack(f'<{n}h', pcm)
        float_samples = [s / 32768.0 for s in samples]
        # Pad 500ms silence at the end to avoid last-token truncation
        float_samples += [0.0] * int(SAMPLE_RATE * 0.5)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, float_samples)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_streams([stream])
        result = self._recognizer.get_result(stream)
        # result may be a string directly or an object with .text
        text = result.text if hasattr(result, 'text') else str(result)
        return text.strip()


class SherpaOnnxZipformerAdapter(ASRAdapter):
    """On-device streaming ASR using sherpa-onnx zipformer transducer (English)."""

    def __init__(self, model_dir: str, hw_provider: str = "cuda", num_threads: int = 2):
        from utils.model_downloader import ensure_model
        ensure_model("asr_en", model_dir)

        import sherpa_onnx
        import glob as _glob

        # Find encoder/decoder/joiner (prefer int8 + chunk-16)
        def _find(prefix, prefer_int8=True):
            pattern = os.path.join(model_dir, f"{prefix}-*.onnx")
            files = _glob.glob(pattern)
            if not files:
                return ""
            chunk16 = [f for f in files if "chunk-16" in f]
            cands = chunk16 if chunk16 else files
            if prefer_int8:
                int8f = [f for f in cands if "int8" in f]
                if int8f:
                    return int8f[0]
            else:
                fp32f = [f for f in cands if "int8" not in f]
                if fp32f:
                    return fp32f[0]
            return cands[0]

        encoder_path = _find("encoder", prefer_int8=True)
        decoder_path = _find("decoder", prefer_int8=False)
        joiner_path = _find("joiner", prefer_int8=True)
        tokens_path = os.path.join(model_dir, "tokens.txt")

        if not all([encoder_path, decoder_path, joiner_path]):
            raise RuntimeError(f"[asr] zipformer model files not found in {model_dir}")

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=hw_provider,
            sample_rate=SAMPLE_RATE,
            decoding_method="greedy_search",
        )
        log.info(f"[asr] sherpa-onnx zipformer adapter loaded: encoder={encoder_path}, provider={hw_provider}")

    def transcribe(self, wav_bytes: bytes, language: str) -> str:
        import io as _io, wave as _wave
        with _wave.open(_io.BytesIO(wav_bytes)) as wf:
            pcm = wf.readframes(wf.getnframes())
        n = len(pcm) // 2
        samples = struct.unpack(f'<{n}h', pcm)
        float_samples = [s / 32768.0 for s in samples]
        float_samples += [0.0] * int(SAMPLE_RATE * 0.5)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, float_samples)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_streams([stream])
        result = self._recognizer.get_result(stream)
        text = result.text if hasattr(result, 'text') else str(result)
        return text.strip()


class SherpaOnnxSenseVoiceAdapter(ASRAdapter):
    """Offline non-autoregressive ASR using SenseVoice-Small (zh/en/ja/ko/cantonese).

    Extremely fast inference (10s audio in ~70ms). Best for Chinese-English
    code-switching scenarios. Uses sherpa_onnx.OfflineRecognizer.
    """

    def __init__(self, model_dir: str, hw_provider: str = "cuda", num_threads: int = 2):
        from utils.model_downloader import ensure_model
        ensure_model("asr_sensevoice", model_dir)

        import sherpa_onnx
        model_path = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=hw_provider,
            use_itn=True,
            language="auto",
        )
        log.info(f"[asr] sherpa-onnx sensevoice adapter loaded: model={model_path}, provider={hw_provider}")

    def transcribe(self, wav_bytes: bytes, language: str) -> str:
        import io as _io, wave as _wave
        with _wave.open(_io.BytesIO(wav_bytes)) as wf:
            pcm = wf.readframes(wf.getnframes())
        n = len(pcm) // 2
        samples = struct.unpack(f'<{n}h', pcm)
        float_samples = [s / 32768.0 for s in samples]
        # Pad 500ms silence at the end to avoid last-token truncation
        float_samples += [0.0] * int(SAMPLE_RATE * 0.5)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, float_samples)
        self._recognizer.decode_streams([stream])
        text = stream.result.text
        return text.strip()


class SherpaOnnxOfflineParaformerAdapter(ASRAdapter):
    """Offline non-streaming Paraformer (zh+en, small).

    Better accuracy than streaming version — no tail truncation.
    Uses sherpa_onnx.OfflineRecognizer.from_paraformer.
    """

    def __init__(self, model_dir: str, hw_provider: str = "cuda", num_threads: int = 2):
        from utils.model_downloader import ensure_model
        ensure_model("asr_paraformer_offline", model_dir)

        import sherpa_onnx
        model_path = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=hw_provider,
            sample_rate=SAMPLE_RATE,
            decoding_method="greedy_search",
        )
        log.info(f"[asr] sherpa-onnx offline paraformer adapter loaded: model={model_path}, provider={hw_provider}")

    def transcribe(self, wav_bytes: bytes, language: str) -> str:
        import io as _io, wave as _wave
        with _wave.open(_io.BytesIO(wav_bytes)) as wf:
            pcm = wf.readframes(wf.getnframes())
        n = len(pcm) // 2
        samples = struct.unpack(f'<{n}h', pcm)
        float_samples = [s / 32768.0 for s in samples]

        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, float_samples)
        self._recognizer.decode_streams([stream])
        text = stream.result.text
        return text.strip()


# ASR model registry
ASR_MODELS = {
    "paraformer-zh-en": {
        "label": "Paraformer Bilingual (zh+en, streaming)",
        "adapter": SherpaOnnxASRAdapter,
        "default_model_dir": "/models/sherpa-onnx/asr",
    },
    "paraformer-offline": {
        "label": "Paraformer Offline (zh+en, small)",
        "adapter": SherpaOnnxOfflineParaformerAdapter,
        "default_model_dir": "/models/sherpa-onnx/asr-paraformer-offline",
    },
    "zipformer-en": {
        "label": "Zipformer English (streaming)",
        "adapter": SherpaOnnxZipformerAdapter,
        "default_model_dir": "/models/sherpa-onnx/asr-en",
    },
    "sensevoice-small": {
        "label": "SenseVoice Small (zh+en+ja+ko+yue)",
        "adapter": SherpaOnnxSenseVoiceAdapter,
        "default_model_dir": "/models/sherpa-onnx/sensevoice",
    },
}


def _build_asr_adapter(cfg: dict) -> Optional[ASRAdapter]:
    model_name = cfg.get('asr_model', 'paraformer-zh-en')
    model_info = ASR_MODELS.get(model_name)
    if not model_info:
        log.warning(f"[asr] unknown model '{model_name}', falling back to paraformer-zh-en")
        model_info = ASR_MODELS["paraformer-zh-en"]
        model_name = "paraformer-zh-en"

    model_dir = cfg.get('model_dir', model_info["default_model_dir"])
    # If model_dir points to another model's default, use correct default
    other_defaults = [v["default_model_dir"] for k, v in ASR_MODELS.items() if k != model_name]
    if model_dir in other_defaults:
        model_dir = model_info["default_model_dir"]

    hw_provider = cfg.get('hw_provider', 'cpu')
    num_threads = int(cfg.get('num_threads', 2))
    return model_info["adapter"](model_dir, hw_provider, num_threads)


# ── VAD Worker Process ────────────────────────────────────────────────────────

def _vad_worker(pcm_q: multiprocessing.Queue, result_q: multiprocessing.Queue,
                stop_evt: multiprocessing.Event,
                backend: str, threshold: float, silence_ms: int,
                kws_cfg: dict = None,
                save_vad_segments: bool = False, max_saved_segments: int = 1000):
    """Runs in a child process — sherpa-onnx ONNX VAD + optional KWS gate.

    Pipeline: Audio → VAD → (KWS gate) → utterance output
    - If kws_cfg is provided and enabled, only output utterances after keyword detected
    - Otherwise (kws disabled), output all utterances (backward compat)
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    _log = logging.getLogger("asr.vad_worker")

    # ── Initialize VAD ──
    import sherpa_onnx
    from utils.model_downloader import ensure_model

    vad_model_dir = '/models/sherpa-onnx/vad'
    ensure_model("vad", vad_model_dir)
    vad_model_path = os.path.join(vad_model_dir, "silero_vad.onnx")

    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=vad_model_path,
            threshold=threshold,
            min_silence_duration=silence_ms / 1000.0,
            min_speech_duration=0.1,
            window_size=512,
            max_speech_duration=30,
        ),
        sample_rate=SAMPLE_RATE,
        num_threads=1,
        provider="cpu",
    )
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
    _log.info(f"[vad-worker] sherpa-onnx VAD initialized (threshold={threshold}, silence_ms={silence_ms})")

    # ── Initialize KWS (optional) ──
    kws_spotter = None
    kws_stream = None
    kws_enabled = (kws_cfg.get('trigger_mode', 'kws') == 'kws') if kws_cfg else False
    if kws_enabled:
        # Select KWS model based on kws_model config
        kws_model_variant = kws_cfg.get('kws_model', 'zh')
        if kws_model_variant == 'zh':
            kws_model_dir = '/models/sherpa-onnx/kws_zh'
            ensure_model("kws_zh", kws_model_dir)
        elif kws_model_variant == 'en':
            kws_model_dir = '/models/sherpa-onnx/kws_en'
            ensure_model("kws_en", kws_model_dir)
        else:  # zh-en
            kws_model_dir = kws_cfg.get('model_dir', '/models/sherpa-onnx/kws')
            ensure_model("kws", kws_model_dir)
        keywords = kws_cfg.get('keywords', [])
        if keywords:
            import glob as _glob
            # Find model files (prefer int8 + chunk-8)
            def _find(prefix, prefer_int8=True):
                pattern = os.path.join(kws_model_dir, f"{prefix}-*.onnx")
                files = _glob.glob(pattern)
                if not files:
                    return ""
                chunk8 = [f for f in files if "chunk-8" in f]
                cands = chunk8 if chunk8 else files
                if prefer_int8:
                    int8f = [f for f in cands if "int8" in f]
                    if int8f: return int8f[0]
                else:
                    fp32f = [f for f in cands if "int8" not in f]
                    if fp32f: return fp32f[0]
                return cands[0]

            encoder = _find("encoder", prefer_int8=True)
            decoder = _find("decoder", prefer_int8=False)
            joiner = _find("joiner", prefer_int8=True)
            tokens = os.path.join(kws_model_dir, "tokens.txt")

            if encoder and decoder and joiner and os.path.exists(tokens):
                # Write keywords file
                kws_keywords_file = os.path.join(kws_model_dir, "keywords.txt")
                with open(kws_keywords_file, 'w', encoding='utf-8') as f:
                    for kw in keywords:
                        f.write(f"{kw}\n")

                kws_spotter = sherpa_onnx.KeywordSpotter(
                    tokens=tokens,
                    encoder=encoder,
                    decoder=decoder,
                    joiner=joiner,
                    keywords_file=kws_keywords_file,
                    num_threads=1,
                    provider="cpu",
                    keywords_score=1.5,
                    keywords_threshold=0.1,
                )
                kws_stream = kws_spotter.create_stream()
                _log.info(f"[vad-worker] KWS initialized, keywords={keywords}")
            else:
                _log.warning(f"[vad-worker] KWS model files not found in {kws_model_dir}, disabling KWS")
                kws_enabled = False
        else:
            _log.info("[vad-worker] KWS enabled but no keywords configured, disabling")
            kws_enabled = False

    # ── State machine ──
    # States: 'waiting_wake' (KWS mode) or 'listening' (direct mode / post-wake)
    state = 'waiting_wake' if kws_enabled else 'listening'
    speech_buf = b''
    start_ts = None
    end_ts = None
    kws_cooldown_until = 0.0

    _log.info(f"[vad-worker] process started (pid={os.getpid()}, backend=sherpa_onnx, kws={kws_enabled})")
    audio_count = 0

    # VAD segment saving
    _VAD_SEG_DIR = '/models/vad_segments'
    _seg_count = [0]

    def _save_segment(float_samples_list, count_ref):
        """Save float samples as WAV."""
        try:
            import wave as _wave, time as _time2
            os.makedirs(_VAD_SEG_DIR, exist_ok=True)
            seg_pcm = _struct.pack(f'<{len(float_samples_list)}h',
                                   *[int(max(-32768, min(32767, s * 32768))) for s in float_samples_list])
            fname = os.path.join(_VAD_SEG_DIR, f"seg_{int(_time2.time()*1000)}.wav")
            with _wave.open(fname, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(seg_pcm)
            # Enforce max segments limit
            if count_ref[0] >= max_saved_segments:
                files = sorted(os.listdir(_VAD_SEG_DIR))
                for old in files[:len(files) - max_saved_segments + 1]:
                    os.remove(os.path.join(_VAD_SEG_DIR, old))
        except Exception:
            pass

    def _save_segment_pcm(pcm_bytes, count_ref):
        """Save raw PCM bytes as WAV."""
        try:
            import wave as _wave, time as _time2
            os.makedirs(_VAD_SEG_DIR, exist_ok=True)
            fname = os.path.join(_VAD_SEG_DIR, f"seg_{int(_time2.time()*1000)}.wav")
            with _wave.open(fname, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm_bytes)
            if count_ref[0] >= max_saved_segments:
                files = sorted(os.listdir(_VAD_SEG_DIR))
                for old in files[:len(files) - max_saved_segments + 1]:
                    os.remove(os.path.join(_VAD_SEG_DIR, old))
        except Exception:
            pass

    while not stop_evt.is_set():
        try:
            pcm, ts = pcm_q.get(timeout=1)
        except Exception:
            continue

        audio_count += 1
        if audio_count == 1:
            _log.info(f"[vad-worker] first audio chunk received! len={len(pcm)}")

        # Convert PCM bytes to float samples
        n = len(pcm) // 2
        if n < 160:
            continue
        import struct as _struct
        samples = _struct.unpack(f'<{n}h', pcm)
        float_samples = [s / 32768.0 for s in samples]

        # Feed VAD
        vad.accept_waveform(float_samples)

        if state == 'waiting_wake':
            # Feed KWS with AGC-normalized audio for better detection at distance
            if kws_spotter:
                kws_stream.accept_waveform(SAMPLE_RATE, float_samples)
                while kws_spotter.is_ready(kws_stream):
                    kws_spotter.decode_stream(kws_stream)
                result = kws_spotter.get_result(kws_stream)
                kw = result.keyword if hasattr(result, 'keyword') else str(result)
                if kw and kw.strip():
                    now = time.time()
                    if now >= kws_cooldown_until:
                        kws_cooldown_until = now + 2.0
                        _log.info(f"[vad-worker] WAKE WORD detected: {kw.strip()}")
                        # Transition to listening — start recording immediately
                        state = 'listening'
                        speech_buf = pcm  # include current frame (user may already be speaking)
                        start_ts = ts
                        end_ts = ts
                        # Reset KWS stream for next wake
                        kws_stream = kws_spotter.create_stream()
            # Drain any completed VAD segments (discard in wake-wait mode)
            while not vad.empty():
                seg = vad.front
                if save_vad_segments:
                    _save_segment(seg.samples, _seg_count)
                    _seg_count[0] += 1
                vad.pop()

        elif state == 'listening':
            # Collect completed VAD segments
            while not vad.empty():
                seg = vad.front
                seg_pcm = _struct.pack(f'<{len(seg.samples)}h',
                                       *[int(max(-32768, min(32767, s * 32768))) for s in seg.samples])
                if not start_ts:
                    start_ts = ts
                speech_buf += seg_pcm
                end_ts = ts
                vad.pop()

                # Output the segment as an utterance
                if len(speech_buf) > SAMPLE_RATE:  # >500ms
                    _log.info(f"[vad-worker] utterance complete, len={len(speech_buf)} bytes")
                    if save_vad_segments:
                        _save_segment_pcm(speech_buf, _seg_count)
                        _seg_count[0] += 1
                    result_q.put((speech_buf, start_ts or ts, end_ts or ts))
                    speech_buf = b''
                    start_ts = None
                    end_ts = None
                    # Return to waiting for wake word (if KWS enabled)
                    if kws_enabled:
                        state = 'waiting_wake'

    _log.info("[vad-worker] process exiting")


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class _ASRNode(Node):
    def __init__(self, input_topic: str, adapter: Optional[ASRAdapter], language: str,
                 vad_backend: str = 'sherpa_onnx', vad_threshold: float = SPEECH_THRESH, vad_silence_ms: int = 400,
                 kws_cfg: dict = None, node_suffix: str = '',
                 save_vad_segments: bool = False, max_saved_segments: int = 1000):
        node_name = f"asr_{node_suffix}" if node_suffix else "asr"
        super().__init__(node_name)
        self._input_topic  = input_topic
        self._output_topic = f"{input_topic}/asr"
        self._adapter  = adapter
        self._language = language
        self.state     = "idle"
        self._sub      = None
        self._pub      = self.create_publisher(String, self._output_topic, _ASR_PUB_QOS)
        # VAD runs in a separate process to avoid GIL contention
        self._vad_backend = vad_backend
        self._vad_threshold = vad_threshold
        self._vad_silence_ms = vad_silence_ms
        self._kws_cfg = kws_cfg or {}
        self._save_vad_segments = save_vad_segments
        self._max_saved_segments = max_saved_segments
        self._pcm_queue: Optional[multiprocessing.Queue] = None
        self._utterance_queue: Optional[multiprocessing.Queue] = None
        self._vad_stop: Optional[multiprocessing.Event] = None
        self._vad_proc: Optional[multiprocessing.Process] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._first_chunk_event = threading.Event()

    def start(self) -> dict:
        if self.state == "running":
            return self._status_dict()
        if not self._adapter:
            return {"state": "error", "message": "ASR adapter not configured"}
        try:
            return self._start_inner()
        except Exception as e:
            log.error(f"[asr] start failed: {e}", exc_info=True)
            self.state = "error"
            return {"state": "error", "message": str(e)}

    def _start_inner(self) -> dict:
        from audio_msgs.msg import AudioChunk
        log.info(f"[asr] subscribing to topic={self._input_topic}, publishing to={self._output_topic}")
        self._first_chunk_event = threading.Event()
        self._sub = self.create_subscription(AudioChunk, self._input_topic, self._audio_cb, _LOW_LAT_QOS)
        self._stop_event.clear()
        # Start VAD in a child process
        self._pcm_queue = multiprocessing.Queue(maxsize=1000)
        self._utterance_queue = multiprocessing.Queue(maxsize=100)
        self._vad_stop = multiprocessing.Event()
        self._vad_proc = multiprocessing.Process(
            target=_vad_worker,
            args=(self._pcm_queue, self._utterance_queue, self._vad_stop,
                  self._vad_backend, self._vad_threshold, self._vad_silence_ms,
                  self._kws_cfg, self._save_vad_segments, self._max_saved_segments),
            daemon=True, name="vad_worker",
        )
        self._vad_proc.start()
        log.info(f"[asr] VAD worker process started (pid={self._vad_proc.pid})")
        # Transcription worker thread (reads from utterance_queue)
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "starting"
        self._worker_ready = threading.Event()
        log.info("[asr] waiting for first audio chunk...")
        # Block until first audio chunk arrives or stop() cancels
        self._first_chunk_event.wait()
        if self._stop_event.is_set():
            self.state = "idle"
            return {"state": "idle"}
        # Wait for worker to finish initialization (IPA precompute etc.)
        self._worker_ready.wait()
        if self.state == "error":
            return {"state": "error", "message": "ASR worker failed to initialize (check logs)"}
        self.state = "running"
        log.info("[asr] started, receiving audio data")
        return self._status_dict()

    def stop(self) -> dict:
        # Stop subscription first to prevent new audio_cb calls
        if self._sub:
            self.destroy_subscription(self._sub); self._sub = None
        self._stop_event.set()
        # Unblock start() if it's waiting
        if hasattr(self, '_first_chunk_event'):
            self._first_chunk_event.set()
        if hasattr(self, '_worker_ready'):
            self._worker_ready.set()
        if self._vad_stop:
            self._vad_stop.set()
        # Cancel feeder threads immediately — avoids BrokenPipeError spam
        for q in (self._pcm_queue, self._utterance_queue):
            if q:
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:
                    pass
        if self._vad_proc and self._vad_proc.is_alive():
            self._vad_proc.join(timeout=5)
            if self._vad_proc.is_alive():
                self._vad_proc.terminate()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def _audio_cb(self, msg):
        if self._stop_event.is_set():
            return
        # Signal first chunk arrival to unblock start()
        if not self._first_chunk_event.is_set():
            self._first_chunk_event.set()
        # Detect dead VAD subprocess to avoid BrokenPipeError in queue feeder
        if self._vad_proc and not self._vad_proc.is_alive():
            log.warning(f"[asr] VAD worker died (exitcode={self._vad_proc.exitcode}), stopping ASR")
            self._stop_event.set()
            # Clean up queues to suppress feeder thread errors
            for q in (self._pcm_queue, self._utterance_queue):
                if q:
                    try:
                        q.cancel_join_thread()
                        q.close()
                    except Exception:
                        pass
            self.state = "error"
            return
        pcm = bytes(msg.data)
        ts  = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            self._pcm_queue.put_nowait((pcm, ts))
        except Exception:
            pass  # drop if severely behind

    def _worker(self):
        try:
            self._worker_inner()
        except Exception as e:
            log.error(f"[asr] worker fatal error: {e}", exc_info=True)
            self.state = "error"
            if hasattr(self, '_worker_ready'):
                self._worker_ready.set()

    def _worker_inner(self):
        # Pre-compute keyword IPA if in asr_kws mode
        trigger_mode = self._kws_cfg.get('trigger_mode', 'kws')
        keyword_ipa = None
        asr_kws_threshold = float(self._kws_cfg.get('asr_kws_threshold', 0.3))
        if trigger_mode == 'asr_kws':
            kw_text = self._kws_cfg.get('asr_kws_keyword', '')
            if kw_text:
                keyword_ipa = _text_to_ipa(kw_text)
                log.info(f"[asr] asr_kws mode: keyword='{kw_text}' ipa={keyword_ipa} threshold={asr_kws_threshold}")
            else:
                log.warning("[asr] asr_kws mode but no keyword configured, falling back to vad mode")
                trigger_mode = 'vad'

        # Signal that worker init is complete
        if hasattr(self, '_worker_ready'):
            self._worker_ready.set()

        while not self._stop_event.is_set():
            try:
                utterance, start_ts, end_ts = self._utterance_queue.get(timeout=1)
            except Exception:
                continue
            try:
                wav   = _pcm16_to_wav(utterance)
                text  = self._adapter.transcribe(wav, self._language)
                if not text.strip(): continue

                # ASR-based keyword spotting
                if trigger_mode == 'asr_kws' and keyword_ipa:
                    text_ipa = _text_to_ipa(text)
                    matched, end_pos = _find_keyword_in_ipa(text_ipa, keyword_ipa, asr_kws_threshold)
                    log.info(f"[asr] asr_kws: text='{text}' ipa={text_ipa} dist_matched={matched} end={end_pos}")
                    if not matched:
                        continue
                    # Extract text after keyword
                    remaining = _extract_after_keyword(text, kw_text, end_pos)
                    log.info(f"[asr] asr_kws TRIGGERED: '{text}' → '{remaining}'")
                    if not remaining.strip():
                        continue
                    text = remaining

                result = {"text": text, "audio_start_ts": start_ts,
                          "audio_end_ts": end_ts, "asr_complete_ts": time.time()}
                msg = String(); msg.data = json.dumps(result, ensure_ascii=False)
                self._pub.publish(msg)
                log.info(f"[asr] {text!r}")
            except Exception as e:
                log.error(f"[asr] transcribe error: {e}", exc_info=True)

    def _status_dict(self) -> dict:
        return {
            "state":     self.state,
            "topic_in":  [{"topic": self._input_topic,  "format": "audio/pcm-16k", "desc": ""}],
            "topic_out": [{"topic": self._output_topic, "format": "data/json",     "desc": "ASR result"}],
        }


# ── Plugin ────────────────────────────────────────────────────────────────────

class ASRPlugin:
    PREFIX = "asr"

    def __init__(self, plugin_cfg: dict, executor):
        self._language     = plugin_cfg.get('language', 'zh-CN')
        self._asr_model    = plugin_cfg.get('asr_model', 'paraformer-zh-en')
        self._plugin_cfg   = plugin_cfg
        self._loading      = False
        self._load_error   = None
        self._adapter      = _build_asr_adapter(plugin_cfg)
        vad_cfg            = plugin_cfg.get('vad', {})
        self._vad_backend  = vad_cfg.get('model', 'sherpa_onnx') or 'sherpa_onnx'
        self._vad_threshold = float(vad_cfg.get('threshold', SPEECH_THRESH))
        self._vad_silence_ms = int(vad_cfg.get('silence_ms', 400))
        self._kws_cfg      = plugin_cfg.get('kws', {})
        self._save_vad_segments = False
        self._max_saved_segments = 1000
        self._nodes: dict[str, _ASRNode] = {}           # key = instance_id
        self._executor = executor
        log.info(f"[asr] plugin init: model={self._asr_model}, vad={self._vad_backend}, threshold={self._vad_threshold}, "
                 f"silence_ms={self._vad_silence_ms}, kws_enabled={self._kws_cfg.get('enabled', False)}")

    def get_tools(self) -> list:
        return TOOLS

    def _load_model_async(self, model_name: str):
        """Download and load ASR model in a background thread."""
        import threading
        def _do_load():
            try:
                log.info(f"[asr] downloading/loading model '{model_name}'...")
                self._plugin_cfg['asr_model'] = model_name
                adapter = _build_asr_adapter(self._plugin_cfg)
                self._adapter = adapter
                self._loading = False
                self._load_error = None
                log.info(f"[asr] model '{model_name}' ready")
            except Exception as e:
                log.error(f"[asr] failed to load model '{model_name}': {e}", exc_info=True)
                self._loading = False
                self._load_error = str(e)

        self._loading = True
        self._load_error = None
        threading.Thread(target=_do_load, daemon=True, name="asr_model_loader").start()

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "asr" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            # Report loading/error state at plugin level
            if self._loading:
                return {
                    "name": "ASR", "manufacture": "Embodied", "model": self._asr_model,
                    "state": "loading",
                    "desc": f"Downloading model '{self._asr_model}'...",
                }
            if self._load_error:
                return {
                    "name": "ASR", "manufacture": "Embodied", "model": self._asr_model,
                    "state": "error",
                    "desc": f"Model load failed: {self._load_error}",
                }
            input_topic = args.get("input_topic", "")
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                return {
                    "name": "ASR", "manufacture": "Embodied", "model": "asr",
                    "state": node.state,
                    "topic_in":  [{"topic": node._input_topic,  "format": "audio/pcm-16k", "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "data/json",     "desc": ""}],
                    "desc": "ASR service — converts audio/pcm-16k to text",
                }
            if instance_id:
                # Instance requested but not running — return inferred topics for this instance only.
                # Do NOT fall through to aggregate path (which would mix in other instances' topics).
                inferred_out = f"{input_topic}/asr" if input_topic else ""
                return {
                    "name": "ASR", "manufacture": "Embodied", "model": "asr",
                    "state": "idle",
                    "topic_in":  [{"topic": input_topic,   "format": "audio/pcm-16k", "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out,  "format": "data/json",     "desc": ""}] if inferred_out else [],
                    "desc": "ASR service — converts audio/pcm-16k to text",
                }
            # Aggregate info for all instances (no instance_id = ping/overview only)
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "audio/pcm-16k", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/asr" if input_topic else ""
                topics_in = [{"topic": input_topic, "format": "audio/pcm-16k", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "data/json", "desc": ""}]
                state = "idle"
            return {
                "name": "ASR", "manufacture": "Embodied", "model": "asr",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "ASR service — converts audio/pcm-16k to text",
            }

        elif action == "start":
            if self._loading:
                # Wait for model to finish loading
                import time as _time
                while self._loading:
                    _time.sleep(0.5)
            if self._load_error:
                return {"state": "error", "message": f"Model failed to load: {self._load_error}"}
            if not self._adapter:
                return {"state": "error", "message": "ASR model not loaded"}
            input_topic = args.get("input_topic")
            # Also accept input_topics list (sent by canvas when multiple connections exist)
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                node = _ASRNode(input_topic, self._adapter, self._language,
                                self._vad_backend, self._vad_threshold, self._vad_silence_ms,
                                kws_cfg=self._kws_cfg,
                                node_suffix=node_key.replace('/', '_').replace('-', '_'),
                                save_vad_segments=self._save_vad_segments,
                                max_saved_segments=self._max_saved_segments)
                self._executor.add_node(node)
                self._nodes[node_key] = node
            else:
                # Sync latest config into existing node before restart
                node = self._nodes[node_key]
                node._adapter = self._adapter
                node._language = self._language
                node._vad_backend = self._vad_backend
                node._vad_threshold = self._vad_threshold
                node._vad_silence_ms = self._vad_silence_ms
                node._kws_cfg = self._kws_cfg
                node._save_vad_segments = self._save_vad_segments
                node._max_saved_segments = self._max_saved_segments
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                # Stop all instances (backward compat / project stop)
                results = []
                for key in list(self._nodes.keys()):
                    node = self._nodes[key]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[key]
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            # Shared config update
            self._language = cfg.get('language', self._language)
            if 'vad_threshold' in cfg:
                self._vad_threshold = float(cfg['vad_threshold'])
            if 'vad_silence_ms' in cfg:
                self._vad_silence_ms = int(cfg['vad_silence_ms'])
            if 'trigger_mode' in cfg:
                self._kws_cfg['trigger_mode'] = cfg['trigger_mode']
            if 'kws_model' in cfg:
                self._kws_cfg['kws_model'] = cfg['kws_model']
            if 'kws_keywords' in cfg:
                self._kws_cfg['keywords'] = [cfg['kws_keywords']]
            if 'asr_kws_keyword' in cfg:
                self._kws_cfg['asr_kws_keyword'] = cfg['asr_kws_keyword']
            if 'asr_kws_threshold' in cfg:
                self._kws_cfg['asr_kws_threshold'] = float(cfg['asr_kws_threshold'])
            if 'save_vad_segments' in cfg:
                self._save_vad_segments = bool(cfg['save_vad_segments'])
            if 'max_saved_segments' in cfg:
                self._max_saved_segments = int(cfg['max_saved_segments'])
            # ASR model switch — load in background if changed
            if 'asr_model' in cfg and cfg['asr_model'] != self._asr_model:
                # Stop all running nodes first
                for key in list(self._nodes.keys()):
                    node = self._nodes.pop(key, None)
                    if node:
                        node.stop()
                        self._executor.remove_node(node)
                self._asr_model = cfg['asr_model']
                self._load_model_async(self._asr_model)
                return {"status": "loading", "asr_model": self._asr_model,
                        "message": f"Switching to model '{self._asr_model}', downloading..."}
            # Stop all nodes (they'll use new config on next start)
            # Only stop VAD internals — keep node in executor to avoid DDS re-discovery delay
            for key in list(self._nodes.keys()):
                node = self._nodes[key]
                node.stop()
            return {"status": "configured", "asr_model": self._asr_model}

        return None


# ── VAD test helper (called by /vad/test HTTP endpoint) ───────────────────────

def _vad_segment_sync(audio_bytes: bytes, model: str = 'silero',
                      threshold: float = 0.5, silence_ms: int = 800) -> list:
    """Run VAD on raw WAV bytes, return list of {start, end, wav} dicts."""
    import io, wave, struct, base64 as _b64, collections as _col

    SAMPLE_RATE = 16000
    USE_WEBRTC  = (model == 'webrtc')
    CHUNK_SAMPLES = 480 if USE_WEBRTC else 512
    CHUNK_BYTES   = CHUNK_SAMPLES * 2
    SILENCE_FRAMES = max(1, int(silence_ms / (1000 * CHUNK_SAMPLES / SAMPLE_RATE)))

    # Convert to WAV if needed via ffmpeg, then decode
    import subprocess as _sp
    try:
        with wave.open(io.BytesIO(audio_bytes)):
            pass  # already valid WAV
    except Exception:
        try:
            r = _sp.run(
                ['ffmpeg', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 'wav', 'pipe:1'],
                input=audio_bytes, capture_output=True, timeout=15,
            )
            if r.returncode == 0:
                audio_bytes = r.stdout
        except FileNotFoundError:
            pass  # no ffmpeg, try parsing as-is

    try:
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            orig_rate = wf.getframerate()
            orig_ch   = wf.getnchannels()
            orig_sw   = wf.getsampwidth()
            pcm_raw   = wf.readframes(wf.getnframes())
    except Exception:
        raise ValueError('无法解析音频文件，请上传 WAV 格式（或安装 ffmpeg 支持其他格式）')

    n_samples = len(pcm_raw) // orig_sw
    if orig_sw == 2:
        samples = list(struct.unpack(f'<{n_samples}h', pcm_raw))
    elif orig_sw == 1:
        samples = [(b - 128) * 256 for b in pcm_raw]
    else:
        raise ValueError(f'不支持的采样位深: {orig_sw * 8}bit')

    if orig_ch > 1:
        samples = samples[::orig_ch]

    if orig_rate != SAMPLE_RATE:
        ratio   = SAMPLE_RATE / orig_rate
        new_len = int(len(samples) * ratio)
        resampled = []
        for i in range(new_len):
            pos = i / ratio
            lo  = int(pos)
            hi  = min(lo + 1, len(samples) - 1)
            resampled.append(int(samples[lo] + (samples[hi] - samples[lo]) * (pos - lo)))
        samples = resampled

    pcm16 = struct.pack(f'<{len(samples)}h', *samples)

    # Load VAD engine
    if USE_WEBRTC:
        import webrtcvad
        vad_engine = webrtcvad.Vad()
        vad_engine.set_mode(min(3, int(threshold * 4)))
        def is_speech(chunk):
            try: return vad_engine.is_speech(chunk, SAMPLE_RATE)
            except Exception: return False
    else:
        import torch
        silero = _get_silero_model()
        def is_speech(chunk):
            n = len(chunk) // 2
            t = torch.tensor(struct.unpack(f'<{n}h', chunk), dtype=torch.float32, device=_get_torch_device()) / 32768.0
            return silero(t, SAMPLE_RATE).item() >= threshold

    preroll: _col.deque = _col.deque(maxlen=8)
    state = 'idle'
    speech_buf = []
    silence_count = 0
    start_s = end_s = 0.0
    segments = []
    chunk_dur = CHUNK_BYTES / 2 / SAMPLE_RATE

    def _flush_segment():
        utterance = b''.join(speech_buf)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(utterance)
        segments.append({'start': round(start_s, 3), 'end': round(end_s, 3),
                         'wav': _b64.b64encode(buf.getvalue()).decode()})

    for i in range(0, len(pcm16), CHUNK_BYTES):
        chunk = pcm16[i:i + CHUNK_BYTES]
        if len(chunk) < CHUNK_BYTES:
            break
        ts = i / 2 / SAMPLE_RATE

        if state == 'idle':
            preroll.append(chunk)

        if is_speech(chunk):
            if state == 'idle':
                pr = list(preroll)
                speech_buf = pr[:-1]
                start_s = ts - chunk_dur * (len(pr) - 1)
                preroll.clear()
            state = 'speaking'
            silence_count = 0
            speech_buf.append(chunk)
            end_s = ts
        elif state == 'speaking':
            speech_buf.append(chunk)
            silence_count += 1
            end_s = ts
            if silence_count >= SILENCE_FRAMES:
                _flush_segment()
                speech_buf = []; silence_count = 0
                state = 'idle'; start_s = end_s = 0.0

    if state == 'speaking' and speech_buf:
        _flush_segment()

    return segments
