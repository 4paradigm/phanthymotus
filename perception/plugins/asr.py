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

from plugins.vad_preroll import PcmHistory

log = logging.getLogger(__name__)

SAMPLE_RATE    = 16000
SPEECH_THRESH  = 0.5
SILENCE_THRESH = 0.35
SILENCE_FRAMES = 16

# Documented AudioChunk contract — see perception/README.md ("Audio Requirements
# for ASR"): 16 kHz mono PCM_S16_LE, at least 512 samples per chunk.
AUDIO_FORMAT       = "audio/pcm-16k"
# agent-core's remote-control mic bridge publishes this spelling instead; accept
# it rather than warning on every chunk of a path that already works.
_AUDIO_FORMAT_ALIASES = frozenset({AUDIO_FORMAT, "pcm_16k_16bit_mono"})
MIN_CHUNK_BYTES    = 1024  # 512 samples — one Silero VAD window

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)


# ── IPA phoneme matching for asr_kws mode ────────────────────────────────────

import re as _re

# ── Persistent EspeakBackend instances (avoid 70ms+ re-init per call) ─────────
_ESPEAK_BACKENDS = {}  # lang -> EspeakBackend instance
_ESPEAK_SEP = None


def _get_espeak_backend(lang):
    global _ESPEAK_SEP
    if _ESPEAK_SEP is None:
        from phonemizer.separator import Separator
        _ESPEAK_SEP = Separator(phone=' ', word='  ', syllable='')
    if lang not in _ESPEAK_BACKENDS:
        from phonemizer.backend import EspeakBackend
        _ESPEAK_BACKENDS[lang] = EspeakBackend(lang, with_stress=False)
    return _ESPEAK_BACKENDS[lang]


def _phonemize_safe(text: str, lang: str) -> str:
    """Phonemize with persistent backend; auto-rebuild on failure."""
    backend = _get_espeak_backend(lang)
    try:
        return backend.phonemize([text], separator=_ESPEAK_SEP, strip=True)[0]
    except Exception:
        # espeak-ng may have crashed — rebuild backend and retry once
        _ESPEAK_BACKENDS.pop(lang, None)
        backend = _get_espeak_backend(lang)
        return backend.phonemize([text], separator=_ESPEAK_SEP, strip=True)[0]


def _text_to_ipa(text: str) -> list:
    """Convert text to IPA phoneme sequence using persistent espeak-ng backend.
    Returns a list of IPA phoneme strings (one per word/character).
    """
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
    for seg_text, is_cjk in segments:
        lang = 'cmn' if is_cjk else 'en-us'
        try:
            ipa = _phonemize_safe(seg_text, lang)
            # Remove tone numbers and diacritics for fuzzy matching
            ipa = _re.sub(r'[0-9˥˦˧˨˩¹²³⁴⁵]', '', ipa)
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
    {'t', 'd'},           # alveolar stops
    {'p', 'b'},           # bilabial stops
    {'k', 'g'},           # velar stops
    {'f', 'v'},           # labiodental fricatives
    {'s', 'z'},           # alveolar fricatives
    {'s.', 'z.'},         # retroflex fricatives
    {'ɕ', 'ʃ', 'ʂ'},     # postalveolar/retroflex sibilants
    {'tsh', 'dz'},        # affricates
    {'n', 'ŋ'},           # nasals
    {'l', 'r', 'ɹ'},      # liquids
    {'t', 'tsh'},         # stop ~ affricate
    {'f', 't'},           # common confusion in noisy env
    {'x', 'h'},           # velar/glottal fricatives
    {'ɑu', 'au', 'ɑo', 'ao'},  # diphthong variants
    {'ou', 'uo'},         # vowel variants
    {'i', 'i.'},          # apical vowel variant
    # ── Chinese ASR common confusions ──
    {'a', 'ɑ'},           # open vowels (same sound, different notation)
    {'an', 'ɑn'},         # front nasal variants
    {'f', 'kh'},          # 范/康 confusion in noisy env
    {'f', 'x'},           # 范/欢 confusion
    {'ts.', 'tɕh'},       # retroflex/palatal affricate confusion
    {'ɑ', 'ɑu'},          # vowel truncation
    {'ai', 'a'},          # diphthong simplification
    {'aiɜ', 'ai', 'a'},   # diphthong variants
    {'iɜ', 'i'},          # rhotacized vowel
    {'əɜ', 'ə', 'e'},     # schwa variants
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
    """Extract text after the matched keyword using IPA end_pos to locate cut point.

    end_pos is the IPA phoneme index where the keyword match ends.
    We map this back to the original text by counting phoneme-producing
    characters and their IPA token counts per segment.
    """
    # Build segments of phoneme-producing characters
    segments = []
    current = ''
    current_is_cjk = None
    for char in text:
        is_cjk = '\u4e00' <= char <= '\u9fff'
        is_alpha = char.isalpha()
        if not is_cjk and not is_alpha:
            continue
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

    # Count IPA tokens per segment to find the text position for end_pos
    ipa_idx = 0
    phoneme_char_pos = 0

    for seg_text, is_cjk in segments:
        lang = 'cmn' if is_cjk else 'en-us'
        try:
            ipa = _phonemize_safe(seg_text, lang)
            ipa = _re.sub(r'[0-9˥˦˧˨˩¹²³⁴⁵]', '', ipa)
            phones = [p for p in ipa.split() if p]
        except Exception:
            phones = list(seg_text)

        seg_ipa_count = len(phones)
        if ipa_idx + seg_ipa_count >= end_pos:
            offset_in_seg = end_pos - ipa_idx
            chars_in_seg = len(seg_text)
            if seg_ipa_count > 0:
                cut_chars = round(offset_in_seg * chars_in_seg / seg_ipa_count)
            else:
                cut_chars = chars_in_seg
            cut_chars = min(cut_chars, chars_in_seg)

            found = 0
            for i, c in enumerate(text):
                if '\u4e00' <= c <= '\u9fff' or c.isalpha():
                    found += 1
                if found >= phoneme_char_pos + cut_chars:
                    cut_idx = i + 1
                    remaining = text[cut_idx:]
                    remaining = remaining.lstrip('，。！？、；：,.!?;: ')
                    return remaining
            return ''
        ipa_idx += seg_ipa_count
        phoneme_char_pos += len(seg_text)

    return ''

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
                "asr_model":     {"type": "string", "enum": ["x-asr-zh-en", "paraformer-zh-en", "paraformer-offline", "zipformer-en", "sensevoice-small"], "description": "ASR model (x-asr-zh-en = bilingual offline transducer with hotwords, paraformer-zh-en = bilingual streaming, paraformer-offline = bilingual offline, zipformer-en = English streaming, sensevoice-small = multilingual offline)", "default": "x-asr-zh-en", "scope": "shared"},
                "trigger_mode":  {"type": "string", "enum": ["vad", "kws", "asr_kws"], "description": "Trigger mode (vad = always listen, kws = KWS model, asr_kws = ASR + phoneme matching)", "default": "kws", "scope": "shared"},
                "kws_model":     {"type": "string", "enum": ["zh", "en", "zh-en"], "description": "KWS 模型 (zh=纯中文, en=纯英文, zh-en=双语)", "default": "zh", "scope": "shared", "x-show-when": {"trigger_mode": "kws"}},
                "kws_keywords":  {"type": "string", "description": "Wake word (zh: 'f àn sh ì x iǎo g ǒu @范式小狗', en: '▁FA N C Y ▁RO B O T @FANCY_ROBOT')", "scope": "shared", "x-show-when": {"trigger_mode": "kws"}},
                "asr_kws_keyword": {"type": "string", "description": "唤醒词文本（如'范式小狗'、'hello robot'）", "scope": "shared", "x-show-when": {"trigger_mode": "asr_kws"}},
                "asr_kws_threshold": {"type": "number", "description": "音素匹配阈值（0-1，越小越严格，推荐0.3）", "default": 0.3, "scope": "shared", "x-show-when": {"trigger_mode": "asr_kws"}},
                "vad_threshold": {"type": "number", "description": "VAD speech threshold (0-1, higher = stricter)", "default": 0.5, "scope": "shared"},
                "vad_silence_ms":{"type": "integer", "description": "Silence duration (ms) before sentence end", "default": 400, "scope": "shared"},
                "vad_pre_roll_ms":{"type": "integer", "description": "Audio retained before detected speech (ms)", "default": 500, "scope": "shared"},
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


class SherpaOnnxXASRAdapter(ASRAdapter):
    """Offline X-ASR transducer with general robot-domain hotword biasing."""

    def __init__(self, model_dir: str, hw_provider: str = "cpu", num_threads: int = 2):
        from utils.model_downloader import ensure_model
        from plugins.x_asr import XASRAdapter

        ensure_model("asr_x_asr", model_dir)
        self._delegate = XASRAdapter(model_dir, hw_provider, num_threads)

    def transcribe(self, wav_bytes: bytes, language: str) -> str:
        return self._delegate.transcribe(wav_bytes, language)


# ASR model registry
ASR_MODELS = {
    "x-asr-zh-en": {
        "label": "X-ASR Bilingual (zh+en, offline transducer)",
        "adapter": SherpaOnnxXASRAdapter,
        "default_model_dir": "/models/sherpa-onnx/x-asr-zh-en",
    },
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
                save_vad_segments: bool = False, max_saved_segments: int = 1000,
                pre_roll_ms: int = 500):
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
    pre_roll_samples = max(0, int(SAMPLE_RATE * pre_roll_ms / 1000))
    silence_samples = max(0, int(SAMPLE_RATE * silence_ms / 1000))
    pcm_history = PcmHistory(SAMPLE_RATE * 31)
    _log.info(
        f"[vad-worker] sherpa-onnx VAD initialized (threshold={threshold}, "
        f"silence_ms={silence_ms}, pre_roll_ms={pre_roll_ms})"
    )

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
    _kws_triggered = False
    speech_buf = b''
    start_ts = None
    end_ts = None
    kws_cooldown_until = 0.0
    _was_speaking = False  # Track speech onset for hook notification

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
        pcm_history.append(pcm[:n * 2])
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
                        _kws_triggered = True
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
            # Detect speech onset → notify main thread for on_hearing hook
            _is_speaking = vad.is_speech_detected()
            if _is_speaking and not _was_speaking:
                result_q.put(("speech_start", ts, ts, False))
            _was_speaking = _is_speaking

            # Collect completed VAD segments.
            # The VAD only reports a segment after min_silence_duration of
            # trailing silence has elapsed, and that silence is not part of
            # seg.samples — so the current chunk timestamp sits one silence
            # window past the real end of speech. Rewind it, the same way
            # PcmHistory.pre_roll() rewinds by silence_samples to locate the
            # segment start in the sample domain.
            seg_end_ts = ts - silence_samples / SAMPLE_RATE
            while not vad.empty():
                seg = vad.front
                seg_pcm = _struct.pack(f'<{len(seg.samples)}h',
                                       *[int(max(-32768, min(32767, s * 32768))) for s in seg.samples])
                if start_ts is None:
                    pre_pcm = pcm_history.pre_roll(
                        getattr(seg, 'start', None),
                        len(seg.samples),
                        pre_roll_samples,
                        silence_samples,
                    )
                    start_ts = seg_end_ts - (len(pre_pcm) + len(seg_pcm)) / 2 / SAMPLE_RATE
                    speech_buf = pre_pcm + seg_pcm
                else:
                    pre_pcm = b''
                    speech_buf += seg_pcm
                end_ts = seg_end_ts
                vad.pop()

                # Output the segment as an utterance
                if len(speech_buf) > SAMPLE_RATE:  # >500ms
                    _log.info(
                        f"[vad-worker] utterance complete, len={len(speech_buf)} bytes, "
                        f"pre_roll_bytes={len(pre_pcm)}"
                    )
                    if save_vad_segments:
                        _save_segment_pcm(speech_buf, _seg_count)
                        _seg_count[0] += 1
                    result_q.put((speech_buf,
                                  seg_end_ts if start_ts is None else start_ts,
                                  seg_end_ts if end_ts is None else end_ts,
                                  _kws_triggered))
                    _kws_triggered = False
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
                 save_vad_segments: bool = False, max_saved_segments: int = 1000,
                 vad_pre_roll_ms: int = 500):
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
        self._vad_pre_roll_ms = vad_pre_roll_ms
        self._kws_cfg = kws_cfg or {}
        self._save_vad_segments = save_vad_segments
        self._max_saved_segments = max_saved_segments
        self._pcm_queue: Optional[multiprocessing.Queue] = None
        self._utterance_queue: Optional[multiprocessing.Queue] = None
        self._vad_stop: Optional[multiprocessing.Event] = None
        self._vad_proc: Optional[multiprocessing.Process] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._audio_contract_warns: dict = {}

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
                  self._kws_cfg, self._save_vad_segments, self._max_saved_segments,
                  self._vad_pre_roll_ms),
            daemon=True, name="vad_worker",
        )
        self._vad_proc.start()
        log.info(f"[asr] VAD worker process started (pid={self._vad_proc.pid})")
        # Transcription worker thread (reads from utterance_queue)
        self.state = "running"
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        log.info("[asr] started, waiting for audio data")
        return self._status_dict()

    def stop(self) -> dict:
        # Stop subscription first to prevent new audio_cb calls
        if self._sub:
            try:
                self.destroy_subscription(self._sub)
            except Exception:
                pass  # may already be invalid
            self._sub = None
        self._stop_event.set()
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
                self._vad_proc.join(timeout=1)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def _audio_cb(self, msg):
        if self._stop_event.is_set():
            return
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
        pcm = self._check_audio_contract(getattr(msg, 'format', '') or '', bytes(msg.data))
        if not pcm:
            return
        ts  = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if ts < 1e9:  # header.stamp not set by publisher
            ts = time.time()
        try:
            self._pcm_queue.put_nowait((pcm, ts))
        except Exception:
            pass  # drop if severely behind

    def _warn_audio_contract(self, key: str, detail: str) -> None:
        """Warn on the first violation of a kind, then every 500th one.

        A misbehaving mic driver publishes ~30 chunks/s, so an unthrottled
        warning would bury the log.
        """
        n = self._audio_contract_warns.get(key, 0) + 1
        self._audio_contract_warns[key] = n
        if n == 1 or n % 500 == 0:
            log.warning(f"[asr] audio contract violated on {self._input_topic}: {detail} "
                        f"(occurrence {n}) — see perception/README.md")

    def _check_audio_contract(self, fmt: str, pcm: bytes) -> bytes:
        """Validate the documented AudioChunk contract, return usable PCM.

        Only the 16-bit alignment is corrected: a chunk with an odd byte count
        makes the VAD worker's struct.unpack() raise, killing the subprocess and
        taking ASR to the error state. Format and chunk size are reported but
        left alone — undersized chunks are the most common cause of "ASR hears
        audio but never emits text", and dropping them here would change the
        behaviour of producers that work today.
        """
        if fmt not in _AUDIO_FORMAT_ALIASES:
            self._warn_audio_contract(
                'format', f"format={fmt!r}, expected {AUDIO_FORMAT!r}")

        if len(pcm) % 2:
            self._warn_audio_contract(
                'align', f"{len(pcm)} bytes is not 16-bit aligned, truncating")
            pcm = pcm[:len(pcm) // 2 * 2]

        if len(pcm) and len(pcm) < MIN_CHUNK_BYTES:
            self._warn_audio_contract(
                'size', f"{len(pcm)}-byte chunk is below the {MIN_CHUNK_BYTES}-byte "
                        f"minimum, VAD may never emit a segment")

        return pcm

    def _worker(self):
        try:
            self._worker_inner()
        except Exception as e:
            log.error(f"[asr] worker fatal error: {e}", exc_info=True)
            self.state = "error"

    def _worker_inner(self):
        # Pre-compute keyword IPA if in asr_kws mode
        _kws_triggered = False  # track whether current utterance was KWS-triggered
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

        while not self._stop_event.is_set():
            try:
                item = self._utterance_queue.get(timeout=1)
                # Handle speech onset signal from VAD worker
                if len(item) >= 2 and item[0] == "speech_start":
                    try:
                        import urllib.request as _urllib_req
                        import json as _json_hook
                        _hook_req = _urllib_req.Request(
                            "https://localhost:15678/api/hooks/fire",
                            data=_json_hook.dumps({"hook": "on_hearing"}).encode(),
                            headers={"Content-Type": "application/json"},
                            method="POST"
                        )
                        import ssl as _ssl
                        _ctx = _ssl.create_default_context()
                        _ctx.check_hostname = False
                        _ctx.verify_mode = _ssl.CERT_NONE
                        _urllib_req.urlopen(_hook_req, timeout=2, context=_ctx)
                    except Exception as _he:
                        log.debug(f"[asr] fire on_hearing failed: {_he}")
                    continue
                if len(item) == 4:
                    utterance, start_ts, end_ts, _kws_from_vad = item
                else:
                    utterance, start_ts, end_ts = item[:3]
                    _kws_from_vad = False
            except Exception:
                continue
            try:
                wav   = _pcm16_to_wav(utterance)

                # 性能 span 记录
                _spans = []
                _t0 = time.time()
                text  = self._adapter.transcribe(wav, self._language)
                _spans.append({"span": "asr_transcribe", "component": "perception",
                               "start_ts": _t0, "end_ts": time.time(),
                               "meta": {"audio_ms": int(len(utterance) / 32)}})
                if not text.strip(): continue

                # ASR-based keyword spotting
                if trigger_mode == 'asr_kws' and keyword_ipa:
                    _t1 = time.time()
                    text_ipa = _text_to_ipa(text)
                    _spans.append({"span": "kws_phonemize", "component": "perception",
                                   "start_ts": _t1, "end_ts": time.time(),
                                   "meta": {"text": text[:20]}})

                    _t2 = time.time()
                    matched, end_pos = _find_keyword_in_ipa(text_ipa, keyword_ipa, asr_kws_threshold)
                    _spans.append({"span": "kws_match", "component": "perception",
                                   "start_ts": _t2, "end_ts": time.time(),
                                   "meta": {"matched": matched}})

                    log.info(f"[asr] asr_kws: text='{text}' ipa={text_ipa} dist_matched={matched} end={end_pos}")
                    if not matched:
                        continue
                    # Extract text after keyword
                    remaining = _extract_after_keyword(text, kw_text, end_pos)
                    log.info(f"[asr] asr_kws TRIGGERED: '{text}' → '{remaining}'")
                    _kws_triggered = True
                    if not remaining.strip():
                        continue
                    text = remaining

                _kws_was_triggered = _kws_triggered or _kws_from_vad
                _kws_triggered = False
                self._kws_triggered = False
                result = {"text": text, "audio_start_ts": start_ts,
                          "audio_end_ts": end_ts, "asr_complete_ts": time.time(),
                          "audio_duration_ms": int(len(utterance) / 32),
                          "text_length": len(text),
                          "priority": 1,
                          "kws_triggered": _kws_was_triggered,
                          "spans": _spans}
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
        self._plugin_cfg   = dict(plugin_cfg)
        self._load_error   = None
        self._adapter      = None
        self._adapter_lock = threading.Lock()
        vad_cfg            = plugin_cfg.get('vad', {})
        self._vad_backend  = vad_cfg.get('model', 'sherpa_onnx') or 'sherpa_onnx'
        self._vad_threshold = float(vad_cfg.get('threshold', SPEECH_THRESH))
        self._vad_silence_ms = int(vad_cfg.get('silence_ms', 400))
        self._vad_pre_roll_ms = int(vad_cfg.get('pre_roll_ms', 500))
        self._kws_cfg      = plugin_cfg.get('kws', {})
        self._save_vad_segments = False
        self._max_saved_segments = 1000
        self._nodes: dict[str, _ASRNode] = {}           # key = instance_id
        self._executor = executor
        log.info(f"[asr] plugin init: model={self._asr_model}, vad={self._vad_backend}, threshold={self._vad_threshold}, "
                 f"silence_ms={self._vad_silence_ms}, pre_roll_ms={self._vad_pre_roll_ms}, "
                 f"kws_enabled={self._kws_cfg.get('enabled', False)}, adapter=lazy")

    def get_tools(self) -> list:
        return TOOLS

    def _ensure_adapter(self) -> Optional[ASRAdapter]:
        """Load the configured ASR runtime once, on the first valid start."""
        if self._adapter is not None:
            return self._adapter

        with self._adapter_lock:
            if self._adapter is not None:
                return self._adapter
            try:
                log.info(f"[asr] loading model '{self._asr_model}' on first start...")
                adapter = _build_asr_adapter(self._plugin_cfg)
            except Exception as e:
                self._load_error = str(e)
                log.error(
                    f"[asr] failed to load model '{self._asr_model}': {e}",
                    exc_info=True,
                )
                raise

            self._adapter = adapter
            self._load_error = None
            log.info(f"[asr] model '{self._asr_model}' ready")
            return adapter

    def _dispose_node(self, key: str) -> dict:
        """Stop and fully destroy one ROS node so its endpoints do not leak."""
        node = self._nodes.pop(key, None)
        if node is None:
            return {"state": "idle"}

        try:
            result = node.stop()
        finally:
            try:
                self._executor.remove_node(node)
            except Exception as error:
                log.warning(f"[asr] failed to remove ROS node '{key}': {error}")
            try:
                node.destroy_node()
            except Exception as error:
                log.warning(f"[asr] failed to destroy ROS node '{key}': {error}")
        return result

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "asr" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
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
            input_topic = args.get("input_topic")
            # Also accept input_topics list (sent by canvas when multiple connections exist)
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            try:
                adapter = self._ensure_adapter()
            except Exception as e:
                return {"state": "error", "message": f"Model failed to load: {e}"}
            if not adapter:
                return {"state": "error", "message": "ASR model not loaded"}
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                node = _ASRNode(input_topic, adapter, self._language,
                                self._vad_backend, self._vad_threshold, self._vad_silence_ms,
                                kws_cfg=self._kws_cfg,
                                node_suffix=node_key.replace('/', '_').replace('-', '_'),
                                save_vad_segments=self._save_vad_segments,
                                max_saved_segments=self._max_saved_segments,
                                vad_pre_roll_ms=self._vad_pre_roll_ms)
                try:
                    self._executor.add_node(node)
                except Exception:
                    node.destroy_node()
                    raise
                self._nodes[node_key] = node
            else:
                # Sync latest config into existing node before restart
                node = self._nodes[node_key]
                node._adapter = adapter
                node._language = self._language
                node._vad_backend = self._vad_backend
                node._vad_threshold = self._vad_threshold
                node._vad_silence_ms = self._vad_silence_ms
                node._vad_pre_roll_ms = self._vad_pre_roll_ms
                node._kws_cfg = self._kws_cfg
                node._save_vad_segments = self._save_vad_segments
                node._max_saved_segments = self._max_saved_segments
            result = self._nodes[node_key].start()
            if result.get("state") == "error":
                self._dispose_node(node_key)
            return result

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                return self._dispose_node(instance_id)
            elif not instance_id and self._nodes:
                # Stop all instances (backward compat / project stop)
                results = []
                for key in list(self._nodes.keys()):
                    self._dispose_node(key)
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
            if 'vad_pre_roll_ms' in cfg:
                self._vad_pre_roll_ms = int(cfg['vad_pre_roll_ms'])
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
            adapter_keys = ('asr_model', 'model_dir', 'hw_provider', 'num_threads')
            adapter_changed = any(
                key in cfg and cfg[key] != self._plugin_cfg.get(key)
                for key in adapter_keys
            )
            if adapter_changed:
                # Stop all running nodes first
                for key in list(self._nodes.keys()):
                    self._dispose_node(key)
                for key in adapter_keys:
                    if key in cfg:
                        self._plugin_cfg[key] = cfg[key]
                self._asr_model = self._plugin_cfg.get('asr_model', self._asr_model)
                with self._adapter_lock:
                    self._adapter = None
                    self._load_error = None
                return {
                    "status": "configured",
                    "asr_model": self._asr_model,
                    "message": "Model will load on next start",
                }
            # Hot-reload: stop running nodes, apply new config, restart automatically
            was_running = [(key, node) for key, node in self._nodes.items() if node.state == "running"]
            for key, node in was_running:
                node.stop()
            # Sync updated config into nodes and restart
            for key, node in was_running:
                node._adapter = self._adapter
                node._language = self._language
                node._vad_backend = self._vad_backend
                node._vad_threshold = self._vad_threshold
                node._vad_silence_ms = self._vad_silence_ms
                node._vad_pre_roll_ms = self._vad_pre_roll_ms
                node._kws_cfg = self._kws_cfg
                node._save_vad_segments = self._save_vad_segments
                node._max_saved_segments = self._max_saved_segments
                node.start()
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
