#!/usr/bin/env python3
"""
plugins/tts.py — the TTS tool contract and its ONNX Runtime engines.

Engines are named `<model>-<languages>`, the same shape `asr_model` uses:
`vits2-zh-en` (TensorRT, implemented in plugins/vits2_tts_trt), `matcha-zh-en`
(Matcha-icefall), `mms-th` (MMS VITS) and `kokoro-multi` (Kokoro-82M v1.0). The
last three all run on sherpa-onnx, which is why naming any of them after the
framework did not work.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

from utils.resample import downsample_24k_to_16k
from plugins.zh_text_norm import ZhTextNormalizer

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200  # 100ms @ 16kHz 16-bit mono
PCM_FRAME_S = CHUNK_BYTES / (SAMPLE_RATE * 2)  # 0.1s of audio per frame

# Kokoro's native rate. The only engine here that is not SAMPLE_RATE: its adapter
# resamples 24000 -> 16000 internally (utils/resample.py) so that the topic stays
# audio/pcm-16k and nothing downstream has to learn a second rate.
KOKORO_SAMPLE_RATE = 24000

# Frames held back before pacing starts, then published in one burst, so the
# consumer begins with a real cushion. 5 frames = 500ms, matching the
# vits2_tts_trt engine's MIX_VITS_PREBUFFER_FRAMES default. This — not the pacing
# interval — is where the downstream margin comes from.
PREBUF_FRAMES = 5
# Pace at exactly the audio each frame carries. Anything shorter over-delivers
# forever, and over-delivery has no safe landing on a live consumer: it either
# buffers without bound or has to discard audio. Briefly setting this to 0.07s
# (matching what the vits2 engine then did) accrued 30ms of surplus per frame
# until the browser player hit its lead cap, rewound its own schedule into audio
# it had already queued, and played back overlapped and 1.43x too fast.
FRAME_INTERVAL_S = PCM_FRAME_S
if FRAME_INTERVAL_S > PCM_FRAME_S:
    log.warning(
        "[tts] FRAME_INTERVAL_S=%.3fs is slower than the %.3fs of audio each "
        "frame carries; downstream will underrun on every utterance",
        FRAME_INTERVAL_S, PCM_FRAME_S,
    )
# Depth of the synthesis→publish handoff queue, in frames (20s of audio).
SYNTH_QUEUE_FRAMES = 200

# Sentinel closing the synthesis→publish queue. A dedicated object rather than
# None so a genuinely empty frame could never be mistaken for end-of-stream.
_SYNTH_DONE = object()

# EOF magic: 8 bytes (4 samples [1, -1, 1, -1])，标记 utterance 结束
# 正常 chunk 始终 3200 bytes，8 bytes 短 chunk 不会被误判
# 即使被不识别 EOF 的旧 Speaker 播放，也只是 0.25ms 极微弱交流声
AUDIO_EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
)


def _agent_core_url() -> str:
    import os
    return os.environ.get("AGENT_CORE_URL", "https://localhost:15678")


def _unverified_ssl_context():
    """Agent Core serves HTTPS with a self-signed certificate."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _post_json(path: str, payload: dict, timeout: float) -> None:
    import urllib.request
    request = urllib.request.Request(
        f"{_agent_core_url()}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(request, timeout=timeout, context=_unverified_ssl_context())


def fire_hook(name: str) -> None:
    """Fire an agent-core hook without blocking the caller.

    on_speaking used to be a synchronous urlopen(timeout=2) inside the publish
    loop, fired after the pacing clock had already been latched — so the request
    latency was charged to the utterance's pacing budget and every frame after
    the prebuffer went out late. Hooks are advisory (LED state); they must never
    sit between two audio frames.
    """
    def _send():
        try:
            _post_json("/api/hooks/fire", {"hook": name}, timeout=2)
        except Exception as exc:
            log.debug("[tts] hook %s failed: %s", name, exc)

    threading.Thread(target=_send, name=f"tts-hook-{name}", daemon=True).start()


def _complete_action(action_id: str, text: str, frames_sent: int,
                     interrupted: bool) -> None:
    """Notify Agent Core that a speak action has terminated.

    Module level, not a node method: an utterance can also die before the worker
    ever sees it (interrupt/stop drains the queue), and the ACP barrier in
    agent-core waits out its full timeout for every action it registered but
    never heard back about.
    """
    if not action_id:
        return
    try:
        _post_json("/api/acp/complete", {
            "action_id": action_id,
            "status": "cancelled" if interrupted else "completed",
            "result": {"text": text[:100], "frames": frames_sent},
        }, timeout=3)
        log.info("[tts] ACP complete: %s (%s)", action_id,
                 "cancelled" if interrupted else "completed")
    except Exception as exc:
        log.warning("[tts] ACP callback failed: %s", exc)


TOOLS = [
    {
        "name": "tts",
        "type": "processor",
        "multiInstance": True,
        "description": "TTS — start/stop speech synthesis, speak text, or get status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "speak", "info", "config", "interrupt"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 topic for text input (data/json, required for action=start)"
                },
                "text": {
                    "type": "string",
                    "description": "Text to synthesize (required for action=speak)"
                },
            },
            "required": ["action"],
            "x-completion": {
                "actions": ["speak"],
                "timeout": 60
            },
            # The speaker is a single exclusive channel: two `speak` calls must
            # serialise, but speaking while the base drives or an arm moves is fine
            # and used to be blocked by the old global ACP barrier.
            "x-resource": "mouth",
            "x-hooks": {
                "on_interrupt_speak": {"action": "interrupt"},
                "on_notify": {"action": "speak"},
            }
        },
        "configSchema": {
            "type": "object",
            "properties": {
                # Engine belongs here, not only in config.yaml: the dashboard
                # builds the config form from configSchema, so an engine that
                # exists solely as a baked YAML key cannot be seen or switched
                # without rebuilding the image. Mirrors asr_model in asr.py.
                "tts_engine": {"type": "string",
                               "enum": ["vits2-zh-en", "matcha-zh-en", "mms-th",
                                        "kokoro-multi"],
                               "description": "TTS engine, named <model>-<languages> to match "
                                              "asr_model (vits2-zh-en = VITS2 on TensorRT, "
                                              "matcha-zh-en = Matcha-icefall, "
                                              "mms-th = MMS Thai, "
                                              "kokoro-multi = Kokoro-82M v1.0)",
                               "default": "vits2-zh-en", "scope": "shared"},
                # matcha-zh-en, mms-th and kokoro-multi only — vits2-zh-en is a
                # TensorRT engine and never touches ONNX Runtime, so this field does
                # nothing for it. Matcha's weights are fp32, so both devices load the
                # same files and only the provider changes; measured 4.3x faster on
                # gpu. kokoro-multi is the first engine where the two devices load
                # *different weight files* — fp32 for gpu, int8 for cpu, because
                # provider_for_device refuses int8 on CUDA — so changing this one
                # downloads a different archive, not just a different provider.
                "device":      {"type": "string", "enum": ["cpu", "gpu"],
                                "description": "Inference device for the ONNX Runtime engines "
                                               "(gpu needs the CUDA sherpa-onnx wheel; ~4.3x faster). "
                                               "kokoro-multi defaults to gpu and loads fp32 there, "
                                               "int8 on cpu",
                                "default": "cpu", "scope": "shared",
                                "x-show-when": {"tts_engine": ["matcha-zh-en", "mms-th",
                                                               "kokoro-multi"]}},
                # Kokoro takes `lang` per generate() call, so this is a scale on the
                # resident model like `speed` — not a session key.
                #
                # Every language Kokoro v1.0 has voices for. The value selects the
                # espeak voice for the **non-Chinese** runs of the text; Chinese is
                # phonemised from lexicon-zh.txt on every setting, so a mixed zh/en
                # sentence code-switches on its own. `zh` maps to an English voice for
                # exactly that reason — see KokoroTTSAdapter.LANGUAGE_VOICES, which
                # also records that these espeak names are measured, not guessed
                # (`en-gb` produces no audio and `fr-FR` silently truncates).
                #
                # Only en-us/en-gb/zh have been listened to; sherpa-onnx documents
                # English and Chinese as the supported pair for this model, so the
                # rest are offered as best-effort. Kept in one enum rather than split
                # into "supported" and "experimental" fields because the dashboard
                # renders one dropdown and the caveat belongs in the docs.
                "tts_language": {
                    "type": "string",
                    "enum": ["en-us", "en-gb", "zh", "ja", "es", "fr", "it",
                             "pt-br", "hi"],
                    "description": "Pronunciation language for kokoro-multi. Chinese is "
                                   "spoken correctly on any setting; this picks the espeak "
                                   "voice for the non-Chinese parts of the text",
                    "default": "en-us", "scope": "shared",
                    "x-show-when": {"tts_engine": ["kokoro-multi"]}},
                # Thai has no spaces between words, and MMS was trained on text
                # that spaced only phrase boundaries. Segmenting every word may
                # help prosody or hurt it — off until it has been A/B'd on device.
                "thai_phrase_spacing": {
                    "type": "boolean",
                    "description": "Insert word boundaries before synthesis (Thai only; "
                                   "experimental — may change prosody either way)",
                    "default": False, "scope": "shared",
                    "x-show-when": {"tts_engine": "mms-th"}},
                "speaker_id": {"type": "integer", "description": "Speaker ID (vits2-zh-en and mms-th support 0 only; for kokoro-multi it is an index within the selected language — 0 is that language's first voice, and the voice moves with tts_language)", "default": 0, "scope": "shared"},
                "speed":      {"type": "number", "description": "Speech speed (1.0 = normal)", "default": 1.0, "scope": "shared"},
            },
            "required": []
        },
        "topic_in":  [{"format": "data/json",     "desc": "text to synthesize"}],
        "topic_out": [{"format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
    }
]


# ── TTS Adapter ──────────────────────────────────────────────────────────────

class TTSAdapter(ABC):
    # Text _TTSNode.start() synthesizes to prove the model works before declaring
    # `running`. Per-adapter, because a probe is only valid if the adapter's own
    # frontend keeps it: "." is punctuation, and the Thai frontend normalises
    # punctuation to a space and then strips it, so the probe reached the model as
    # an empty string and every Thai start failed with "TTS dry-run produced no
    # audio". Keep it to one syllable — it is synthesized on every start.
    dry_run_text = "."

    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    def synthesize_stream(self, text: str):
        """Yield raw PCM bytes as they arrive. Default: collect all."""
        yield self.synthesize(text)

    def warmup(self) -> int:
        """Pay the first-inference cost at load; return the bytes produced.

        Two separate one-off costs, both measured on Orin5 with the Thai model:

        - **The model.** First utterance after the session is built takes 2361 ms
          to its first frame; the second takes 342 ms — 1695 ms of lazy CUDA
          kernels and memory pool.
        - **The frontend.** `ThaiFrontend.normalize()` costs **3183 ms** on its
          first call and 0.2 ms after, because pythainlp loads its corpora and the
          newmm dictionary lazily. That is the larger of the two and is pure CPU.

        Synthesizing `dry_run_text` covers both, because it goes through the
        adapter's own frontend. Deliberately one short syllable, not a coverage
        pass: engine construction happens inside the config path that
        ENGINE_SWITCH_WAIT_S bounds, so a long warmup would push a switch into
        answering `loading`.

        `plugins.tts.warmup` in config.yaml has existed all along and, until this,
        did nothing for either sherpa-onnx engine — only the VITS2 plugin honoured
        it.
        """
        return sum(len(chunk) for chunk in self.synthesize_stream(self.dry_run_text))

    def set_speed(self, speed: float) -> None:
        """Change speed on the resident model, without rebuilding it.

        Mirrors Vits2TensorRTAdapter.set_speed. sherpa-onnx takes `speed` on every
        `generate()` call, so nothing has to be reloaded — which is the whole point:
        `config` used to rebuild the session for any change at all.
        """
        del speed

    def set_language(self, language: str) -> None:
        """Change the pronunciation language on the resident model, if it has one.

        A no-op by default, so `_config` can call it unconditionally: only Kokoro
        takes a language, and only because sherpa-onnx reads `lang` out of
        GenerationConfig.extra per call. An engine whose language is baked into the
        checkpoint (Matcha, MMS Thai, VITS2) has nothing to change, and its
        configSchema field is hidden by `x-show-when` anyway.
        """
        del language


class MatchaTTSAdapter(TTSAdapter):
    """On-device TTS using sherpa-onnx Matcha (flow-matching, fast non-autoregressive)."""

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0,
                 device: str = "cpu"):
        import os
        from utils.model_downloader import ensure_model
        from utils.onnx_provider import provider_for_device
        ensure_model("tts", model_dir)
        ensure_model("tts_vocoder", model_dir)

        import sherpa_onnx
        # Matcha model files
        acoustic_model = os.path.join(model_dir, "model-steps-3.onnx")
        vocoder = os.path.join(model_dir, "vocos-16khz-univ.onnx")
        lexicon_path = os.path.join(model_dir, "lexicon.txt")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        data_dir = os.path.join(model_dir, "espeak-ng-data")
        if not os.path.isdir(data_dir):
            data_dir = ""
        # Both weights are fp32, so there is only one file set and device just
        # picks the provider — measured 4.3x faster on gpu at num_threads=2.
        provider = provider_for_device(device, (acoustic_model, vocoder))

        # The ZH number/date/phone FSTs are applied by us, in Python, not handed to
        # sherpa-onnx as rule_fsts. sherpa runs rule_fsts over the *whole* text
        # before its frontend decides what is Chinese, so passing them here rewrote
        # every digit into Chinese characters and the frontend then read them in
        # Chinese — "We have 25 exhibits" came out as "We have 二十五 exhibits".
        # plugins/zh_text_norm.py applies them only to the Chinese runs.
        self._zh_norm = ZhTextNormalizer(model_dir)
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=acoustic_model,
                    vocoder=vocoder,
                    lexicon=lexicon_path if os.path.exists(lexicon_path) else "",
                    tokens=tokens_path,
                    data_dir=data_dir,
                    length_scale=1.0 / speed if speed else 1.0,
                ),
                num_threads=2,
                provider=provider,
            ),
            rule_fsts="",
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._sid = speaker_id
        self._speed = speed
        log.info(f"[tts] sherpa-onnx Matcha loaded: model_dir={model_dir}, "
                 f"speaker_id={speaker_id}, speed={speed}, "
                 f"device={device}, provider={provider}, "
                 f"zh_text_norm={'on' if self._zh_norm.available else 'off'}")

    def synthesize(self, text: str) -> bytes:
        return b''.join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        import struct
        # Normalise before the engine sees the text, and only the Chinese parts of
        # it — see the rule_fsts comment in __init__.
        text = self._zh_norm.normalize(text)
        audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        float_samples = audio.samples
        # Matcha + vocos-16khz outputs 16kHz directly, no resampling needed
        pcm = struct.pack(f'<{len(float_samples)}h',
                         *[int(max(-32768, min(32767, s * 32767))) for s in float_samples])
        for i in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[i:i + CHUNK_BYTES]

    def set_speed(self, speed: float) -> None:
        # sherpa-onnx applies `speed` per generate() call (it overrides the
        # session's length_scale when speed != 1), so there is nothing to reload.
        self._speed = speed


class MmsThaiTTSAdapter(TTSAdapter):
    """On-device Thai TTS using sherpa-onnx VITS (MMS Thai, character-level).

    A separate adapter rather than a flag on the Matcha one: the two share no
    model file. Matcha is an acoustic model plus a vocos vocoder and reads ZH rule
    FSTs; MMS VITS is end-to-end, has no vocoder, and has no rule FSTs at all
    because sherpa-onnx ships none for Thai. Everything Thai-specific therefore
    happens in Python, in plugins/thai_frontend.py, before the text gets here.

    The frontend is not optional. The tokenizer is character-level over 71
    characters and silently drops the rest, so unnormalised text loses its digits,
    its Latin words, and — because ``ำ`` is not in the table — the vowel of a
    large fraction of ordinary Thai words.
    """

    # A single Thai consonant, not the "." the other adapters use: the frontend
    # normalises punctuation to a space and strips it, so "." reached the model as
    # an empty string and every start failed with "dry-run produced no audio".
    # Measured 0.384 s of audio and 108 ms to synthesize — the cheapest probe that
    # still proves the model runs.
    dry_run_text = "ก"

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0,
                 device: str = "cpu", phrase_spacing: bool = False):
        import os
        from utils.model_downloader import ensure_thai_tts_model
        from utils.onnx_provider import provider_for_device

        if speaker_id != 0:
            # MMS Thai is single-speaker. Accepting a stray id would silently
            # synthesize speaker 0 anyway and make the card look configurable.
            raise ValueError(
                f"the Thai VITS model has one speaker; speaker_id must be 0, got {speaker_id}"
            )

        # Refuse to start without the text frontend's dependencies. Every function
        # in thai_frontend degrades to a warning when an import fails, which is
        # right for a single missing transliterator but wrong as a whole: without
        # pythainlp no number is converted, and the digits 3 and 5-9 are not in the
        # token table, so they are dropped from the audio. The card would come up
        # `running` and mispronounce every utterance carrying a number. Better a
        # visible `state: error` on this one card than a robot that sounds fine and
        # says the wrong thing.
        try:
            import pythainlp  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "the Thai TTS engine needs pythainlp (perception/plugins/"
                "requirements.thai.txt); without it numbers are dropped from the "
                f"audio rather than spoken: {error}"
            ) from error

        model_dir = ensure_thai_tts_model(model_dir)
        model_path = os.path.join(model_dir, "model.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        for path in (model_path, tokens_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Thai TTS model is incomplete: {path} is missing")
        _validate_thai_manifest(model_dir)

        import sherpa_onnx

        provider = provider_for_device(device, (model_path,))
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model_path,
                    tokens=tokens_path,
                    # Character-level: no lexicon, and no espeak-ng data dir.
                    # Passing either makes sherpa-onnx take a phoneme path this
                    # model was not trained for.
                    lexicon="",
                    data_dir="",
                    length_scale=1.0 / speed if speed else 1.0,
                ),
                num_threads=2,
                provider=provider,
            ),
            # No rule_fsts: sherpa-onnx has no Thai number/date FSTs. That work is
            # thai_frontend.normalize()'s, and it has to happen anyway because the
            # digits 3 and 5-9 are not in the token table at all.
            rule_fsts="",
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)

        model_rate = int(getattr(self._tts, "sample_rate", 0) or 0)
        if model_rate and model_rate != SAMPLE_RATE:
            # The ROS topic is audio/pcm-16k and the pacing constants are derived
            # from that rate, so a 22.05 kHz voice would not merely need resampling
            # — it would play back at the wrong pitch *and* drift against the
            # 100 ms frame clock. Refuse instead. (FEMALEV2 is 22.05 kHz; MALE-
            # NARRATOR is 16 kHz, which is why that one is the packaged voice.)
            raise RuntimeError(
                f"Thai TTS model is {model_rate} Hz but the audio pipeline is "
                f"{SAMPLE_RATE} Hz; a resampler would have to be added first"
            )

        from plugins.thai_frontend import ThaiFrontend

        self._frontend = ThaiFrontend(
            vocab=_read_token_chars(tokens_path),
            phrase_spacing=phrase_spacing,
        )
        self._sid = speaker_id
        self._speed = speed
        log.info(f"[tts] sherpa-onnx Thai VITS loaded: model_dir={model_dir}, "
                 f"speed={speed}, device={device}, provider={provider}, "
                 f"sample_rate={model_rate or SAMPLE_RATE}")

        # Fail here, not on the first start. _TTSNode.start() refuses to declare
        # `running` unless the probe produces audio, and a probe the frontend
        # normalises away can never do that — which is how "." made every Thai
        # start report "dry-run produced no audio" while the model itself was
        # fine. Checking at construction turns a future frontend change that
        # swallows this probe into a load error naming the cause.
        if not self._frontend.normalize(self.dry_run_text):
            raise RuntimeError(
                f"the Thai frontend normalises the dry-run probe "
                f"{self.dry_run_text!r} to nothing, so no start could ever succeed"
            )

    def synthesize(self, text: str) -> bytes:
        return b''.join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        import struct

        normalized = self._frontend.normalize(text)
        if not normalized:
            log.warning("[tts] nothing speakable left in %r after Thai normalisation", text)
            return
        for chunk in self._frontend.iter_chunks(normalized):
            audio = self._tts.generate(chunk, sid=self._sid, speed=self._speed)
            float_samples = audio.samples
            pcm = struct.pack(f'<{len(float_samples)}h',
                              *[int(max(-32768, min(32767, s * 32767))) for s in float_samples])
            for i in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[i:i + CHUNK_BYTES]

    def set_speed(self, speed: float) -> None:
        # sherpa-onnx applies `speed` per generate() call (it overrides the
        # session's length_scale when speed != 1), so there is nothing to reload.
        self._speed = speed


def _validate_thai_manifest(model_dir: str) -> None:
    """Check the release's manifest before sherpa-onnx gets a chance to exit(-1).

    sherpa-onnx picks its text frontend from the ONNX `frontend` metadata string,
    and only the exact value "characters" selects the character-level path this
    model needs. Any other value reaches a branch that logs "Not a model using
    characters as modeling unit" and calls `SHERPA_ONNX_EXIT(-1)` — a **process
    exit**, so main.py's try/except around the TTS plugin cannot turn it into a
    card in `state: error`; it takes ASR, VOP and OCR down with it.

    The manifest that tools/export_mms_thai_onnx.py writes alongside the model
    records what it set, so a mismatched or hand-assembled release fails here as
    an ordinary exception instead. Mirrors _validate_manifest in
    plugins/vits2_tts_trt/runtime/backends/trt_numpy_tts_engine.py.
    """
    import os

    manifest_path = os.path.join(model_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} is missing; this release was not produced by "
            "tools/export_mms_thai_onnx.py and cannot be checked before load"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    frontend = manifest.get("frontend")
    if frontend != "characters":
        raise RuntimeError(
            f"Thai TTS release declares frontend={frontend!r}; sherpa-onnx needs "
            "'characters' and hard-exits the process on anything else"
        )
    rate = int(manifest.get("sample_rate") or 0)
    if rate != SAMPLE_RATE:
        raise RuntimeError(
            f"Thai TTS release is {rate} Hz but the audio pipeline is "
            f"{SAMPLE_RATE} Hz; a resampler would have to be added first"
        )
    speakers = int(manifest.get("n_speakers") or 1)
    if speakers != 1:
        raise RuntimeError(
            f"Thai TTS release declares {speakers} speakers; the adapter only "
            "supports a single-speaker model"
        )


def _read_token_chars(tokens_path: str) -> set:
    """Read the model's own token table so the frontend checks against it.

    The frontend has a hardcoded copy for unit tests, but the guarantee "the
    output only contains characters this model can say" is only true if it is
    checked against the model that is actually loaded.
    """
    chars = set()
    with open(tokens_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            # "<token> <id>", and the token may itself be a space — so split off
            # the id from the right rather than splitting the line.
            token, _, _ = line.rpartition(" ")
            if token:
                chars.add(token)
    return chars


class KokoroTTSAdapter(TTSAdapter):
    """On-device TTS using sherpa-onnx Kokoro (Kokoro-82M v1.0, 24 kHz).

    A third adapter rather than a flag on either existing one, for three reasons
    that are all structural:

    - It is configured through `OfflineTtsKokoroModelConfig`, which shares no field
      with the Matcha or VITS ones: the voice is a style vector read out of a
      separate `voices.bin`, not a speaker embedding inside the graph.
    - Its text frontend is sherpa-onnx's own `KokoroMultiLangLexicon`, which splits
      the text on Chinese vs non-Chinese character runs and phonemises each
      separately. So zh/en code-switching inside one sentence works without any
      Python-side detection — unlike mms-th, which needs plugins/thai_frontend.py.
    - **It is the only engine whose sample rate is not the pipeline's.** Kokoro is
      24 kHz; everything downstream is 16 kHz and has no way to be told otherwise
      (AudioChunk carries the rate inside its `format` string, and the publisher's
      pacing derives from SAMPLE_RATE). The conversion therefore happens here, in
      utils/resample.py, and what leaves this adapter is 16 kHz like every other
      engine's output. Nothing outside this class knows Kokoro is 24 kHz.

    Language is a per-utterance argument, not a session key: sherpa-onnx reads it
    from `GenerationConfig.extra["lang"]` on every call. So `set_language` is a
    field assignment, the same shape as `set_speed`, and switching language costs
    nothing. Verified on the shipped wheels for both Python ABIs (cp38/jp5.11 and
    cp310/jp6.1): `extra` accepts a plain dict and round-trips.
    """

    # Not the inherited ".". Kokoro's frontend has an explicit punctuation branch
    # that turns "." into a three-token near-silent sequence, and _TTSNode.start()
    # refuses to declare `running` unless the probe produced audio — the same trap
    # that made every Thai start fail. "ok" is one syllable, is phonemised by
    # espeak on every language setting (routing is by character class, so an
    # English probe exercises the espeak path even when lang=zh), and is audible.
    dry_run_text = "ok"

    # Every language Kokoro v1.0 has voices for. The value on the left is what the
    # dashboard and config.yaml use; the value on the right is the espeak-ng voice
    # name that phonemises the non-Chinese runs of the text.
    #
    # The right-hand names are **measured, not guessed** — each was checked on
    # Orin 6 against this release's espeak-ng-data with a sentence in that language
    # containing digits. Guessing is not safe here, in two distinct ways:
    #
    #   en-gb  -> "Failed to set eSpeak-ng voice", no samples at all. The data ships
    #             en-GB-x-rp, en-GB-scotland, en-GB-x-gbclan, en-GB-x-gbcwmd and
    #             en-029, but no plain en-GB. (Matching is case-insensitive, which
    #             is why en-us resolves to en-US and made en-gb look plausible.)
    #   fr-FR  -> *worse*: it "works" and silently truncates. 18280 samples against
    #             63277 for `fr` on the same sentence — most of the utterance simply
    #             missing, with nothing raised and nothing logged.
    #
    # `zh` maps to an English voice on purpose, and is not redundant with `en-us`.
    # Chinese is phonemised from lexicon-zh.txt on every setting — that branch of
    # sherpa-onnx's frontend never consults `lang` — so this field only decides how
    # the *embedded Latin* in a Chinese sentence is read, and English is the right
    # answer for that. Digits are already converted to Chinese characters upstream
    # by the ZH rule FSTs. The label exists because an operator setting up a Chinese
    # robot looks for it, and its absence reads as "Chinese is unsupported".
    #
    # CAVEAT, and it is a real one: sherpa-onnx's own documentation says of this
    # model "it is a multi-lingual model, but we only add English and Chinese
    # support for it". Kokoro was trained with misaki G2P, and for the languages
    # below other than English the espeak phoneme set is not guaranteed to be the
    # one the acoustic model learned. All nine produce audio of a plausible length;
    # en-us, en-gb and zh are the ones that have been listened to. Treat the rest as
    # best-effort until someone who reads the language has heard them.
    LANGUAGE_VOICES = {
        "en-us": "en-us",       # ids 0-19   af_*/am_*
        "en-gb": "en-gb-x-rp",  # ids 20-27  bf_*/bm_*
        "zh": "en-us",          # ids 45-52  zf_*/zm_*  (see above)
        # NOT "ja", and this is the one entry that must not be "corrected".
        #
        # Kokoro's 114-token table is the misaki phoneme inventory — it has
        # `ʣ ʥ ʦ ʨ ᵝ`, so the model was trained to speak Japanese — but it does not
        # contain `ʑ`, which is exactly what espeak-ja emits for じ, along with the
        # combining diacritics U+0308 and U+031E. sherpa phonemises with espeak and
        # looks the result up in that table, silently discarding whatever is missing.
        # Measured on Orin 6: 12 phonemes dropped from one sentence, and the audible
        # holes made it unintelligible.
        #
        # plugins/ja_text_norm.py therefore emits **romaji**, and this picks the
        # espeak voice by its phoneme inventory rather than by its language. Measured
        # on the same sentence, dropped-phoneme count and duration:
        #
        #     kana    + ja      12 dropped   4.81 s
        #     romaji  + ja       0 dropped  14.78 s   (spelled out letter by letter)
        #     romaji  + it       0 dropped   4.60 s
        #     romaji  + es       0 dropped   4.83 s
        #     romaji  + en-us    0 dropped   5.19 s
        #
        # Italian of the three: five pure vowels like Japanese, and native geminate
        # consonants for っ (`nikki`), which Spanish lacks and English would reduce.
        "ja": "it",             # ids 37-41  jf_*/jm_*
        "es": "es",             # ids 28-29, 53  ef_*/em_*
        "fr": "fr",             # id  30     ff_siwis
        "it": "it",             # ids 35-36  if_*/im_*
        "pt-br": "pt-BR",       # ids 42-44  pf_*/pm_*
        "hi": "hi",             # ids 31-34  hf_*/hm_*
    }
    LANGUAGES = tuple(LANGUAGE_VOICES)
    DEFAULT_LANGUAGE = "en-us"

    # Spellings that are not the canonical label but are what someone would write.
    # `cn` is accepted here even though the repo's naming rule rejects it as an
    # engine-name component — refusing an operator's config is not the place to make
    # that point, and the alternative is a Chinese robot silently speaking English.
    LANGUAGE_ALIASES = {
        "en": "en-us", "en-au": "en-us", "en-029": "en-gb", "en-gb-x-rp": "en-gb",
        "zh-cn": "zh", "cmn": "zh", "cn": "zh", "zh-hans": "zh",
        "pt": "pt-br", "pt-pt": "pt-br",
        "es-419": "es", "es-es": "es",
        "fr-fr": "fr", "fr-ca": "fr",
        "jp": "ja", "ja-jp": "ja",
        "hi-in": "hi", "it-it": "it",
    }

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0,
                 device: str = "gpu", language: str = DEFAULT_LANGUAGE,
                 japanese_worker: bool = True, japanese_worker_device: str = "gpu"):
        import os
        from utils.model_downloader import ensure_kokoro_model
        from utils.onnx_provider import normalize_device, pick_weights, provider_for_device

        device = normalize_device(device)
        language = self._normalize_language(language)

        model_dir = ensure_kokoro_model(model_dir, device)
        # gpu directories hold fp32, cpu directories hold int8 — pick_weights'
        # documented behaviour of falling back to the *last* candidate means a
        # missing file produces an error naming the one this device wanted.
        candidates = (("model.onnx", "model.int8.onnx") if device == "gpu"
                      else ("model.int8.onnx", "model.onnx"))
        model_path = pick_weights(model_dir, *candidates)
        voices_path = os.path.join(model_dir, "voices.bin")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        data_dir = os.path.join(model_dir, "espeak-ng-data")
        lexicon_path = os.path.join(model_dir, "lexicon-zh.txt")

        manifest = _validate_kokoro_manifest(model_dir)

        for path in (model_path, voices_path, tokens_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Kokoro release is incomplete: {path} is missing")
        # The four files sherpa-onnx's own Validate() checks for inside data_dir.
        # Checking them here turns a missing espeak tree into an exception naming
        # the file, instead of a Validate() failure during OfflineTts construction.
        missing = [name for name in ("phontab", "phonindex", "phondata", "intonations")
                   if not os.path.exists(os.path.join(data_dir, name))]
        if missing:
            raise FileNotFoundError(
                f"{data_dir} is not an espeak-ng data directory (missing "
                f"{', '.join(missing)}); sherpa-onnx requires all four"
            )

        # Only lexicon-zh.txt. The two English lexicons upstream ships can never be
        # read: for non-Chinese runs sherpa-onnx short-circuits to espeak whenever
        # `lang` is non-empty, and `lang` falls back to the model's own
        # meta_data.voice ("en-us") when unset, so it never is. The Chinese branch
        # does not consult `lang`, which is why this one is load-bearing.
        lexicon = lexicon_path if os.path.exists(lexicon_path) else ""

        # THE process-exit guard. For a version >= 2 Kokoro model with *both*
        # `lexicon` and `lang` empty, sherpa-onnx's InitFrontend logs and calls
        # SHERPA_ONNX_EXIT(-1) — a process exit, not an exception, so main.py's
        # try/except around the TTS plugin cannot turn it into a card in
        # `state: error`; it takes ASR, VOP and OCR down with it. `language` is
        # normalised to a non-empty value above, so this can only fire if someone
        # later makes it optional.
        if not lexicon and not language:
            raise RuntimeError(
                "Kokoro v1.0 needs a lexicon or a language; with neither, "
                "sherpa-onnx exits the whole process instead of raising"
            )

        n_speakers = int(manifest.get("n_speakers") or 0)
        if n_speakers and not 0 <= int(speaker_id) < n_speakers:
            raise ValueError(
                f"speaker_id must be 0..{n_speakers - 1} for kokoro-multi, "
                f"got {speaker_id}"
            )

        import sherpa_onnx

        provider = provider_for_device(device, (model_path,))

        # The ZH number/date/phone FSTs are applied by us, not by sherpa-onnx. Given
        # to it as rule_fsts they run over the *whole* text before the frontend
        # splits it, so the digits become Chinese characters and are then read in
        # Chinese whatever `lang` says — "opened in 2026" became "opened in
        # 二千零二十六" in English, Spanish, Japanese, all of them.
        # plugins/zh_text_norm.py applies them only to the Chinese runs.
        self._zh_norm = ZhTextNormalizer(model_dir)
        # Japanese needs the opposite treatment to Chinese: not "normalise the
        # numbers", but "get every kanji out of the text". sherpa-onnx routes
        # [一-鿿] to its Chinese branch with no language check, and Japanese kanji
        # live in that range, so without this they are pronounced in Mandarin.
        # Built lazily on the first switch to `ja` — Janome loads a ~180 MB
        # dictionary, and a Chinese or English deployment must not pay for it.
        self._ja_frontend = None
        # Japanese does not go through sherpa at all — see _synthesize_japanese —
        # so the direct runtime needs to know which weights and provider to reuse.
        self._direct_runtime = None
        self._model_dir = model_dir
        self._weights_name = os.path.basename(model_path)
        self._provider = provider
        self._japanese_worker = bool(japanese_worker)
        self._japanese_worker_device = japanese_worker_device

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=model_path,
                    voices=voices_path,
                    tokens=tokens_path,
                    lexicon=lexicon,
                    data_dir=data_dir,
                    # Ignored since sherpa-onnx v1.12.15 (it logs "you don't need to
                    # provide dict_dir" if given), so the release ships no dict/.
                    dict_dir="",
                    length_scale=1.0 / speed if speed else 1.0,
                    # Also set at construction, not only per call: it is what makes
                    # the InitFrontend check above pass, and it is the fallback if a
                    # generate() ever arrives without extra["lang"]. The espeak
                    # voice name, not the configured label — see LANGUAGE_VOICES.
                    lang=self.LANGUAGE_VOICES[language],
                ),
                num_threads=2,
                provider=provider,
            ),
            rule_fsts="",
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)

        model_rate = int(getattr(self._tts, "sample_rate", 0) or 0)
        if model_rate and model_rate != KOKORO_SAMPLE_RATE:
            # utils/resample.py's filter is designed for exactly 24000 -> 16000
            # (2:3). Another rate would need its own filter, and using this one
            # would resample by the wrong ratio — audio at the wrong pitch that
            # also drifts against the 100 ms frame clock.
            raise RuntimeError(
                f"Kokoro model is {model_rate} Hz but the adapter's resampler is "
                f"built for {KOKORO_SAMPLE_RATE} Hz -> {SAMPLE_RATE} Hz"
            )

        self._speed = speed
        self._language = language
        self._manifest = manifest
        self._lock = threading.Lock()
        # `voice_index` is per language; `_sid` is the global index the model wants.
        self._voice_index = int(speaker_id)
        self._sid = self._resolve_sid(self._voice_index, language, strict=True)

        # Re-check against the loaded graph, not just the manifest: the manifest is
        # our own file and could have been written for a different voices.bin.
        live_speakers = int(getattr(self._tts, "num_speakers", 0) or 0)
        if live_speakers and not 0 <= self._sid < live_speakers:
            raise ValueError(
                f"resolved speaker index {self._sid} is outside this Kokoro "
                f"release's 0..{live_speakers - 1}; the manifest does not match "
                "voices.bin"
            )

        log.info(f"[tts] sherpa-onnx Kokoro loaded: model_dir={model_dir}, "
                 f"weights={os.path.basename(model_path)}, "
                 f"speaker_id={self._voice_index} ({self.voice_name}, global "
                 f"{self._sid}), language={language} (espeak {self._voice}), "
                 f"speed={speed}, device={device}, provider={provider}, "
                 f"model_rate={model_rate or KOKORO_SAMPLE_RATE} -> {SAMPLE_RATE}, "
                 f"lexicon={'zh' if lexicon else 'none'}, "
                 f"zh_text_norm={'on' if self._zh_norm.available else 'off'}")

        if language == "ja":
            # Build the Japanese frontend before the probe, so a missing janome is a
            # load error naming the build flag rather than a card that comes up
            # `running` and then speaks Mandarin.
            self._ja()

        # Prove the espeak voice resolves before anything asks this adapter to
        # speak. A bad voice name is close to invisible at runtime: sherpa-onnx
        # writes "Failed to set eSpeak-ng voice" to stderr, `generate` returns no
        # samples, and the utterance is silently skipped — which is how `en-gb`
        # (a voice espeak-ng does not have; it is en-GB-x-rp) looked like a model
        # problem. `dry_run_text` is Latin, so it exercises exactly this path.
        #
        # Same reasoning as the Thai adapter checking its probe survives the
        # frontend, and it also pays the first-inference cost, so the warmup a
        # moment later is cheap.
        if not any(self.synthesize_stream(self.dry_run_text)):
            raise RuntimeError(
                f"Kokoro produced no audio for the probe {self.dry_run_text!r} with "
                f"language={self._language} (espeak voice {self._voice!r}); the "
                "voice name is probably not one this espeak-ng-data provides"
            )

    @classmethod
    def _normalize_language(cls, value) -> str:
        """Coerce a configured language to one this release can actually phonemize.

        Falls back rather than raising, the way provider_for_device does: a stale
        value in a baked config.yaml should degrade to a working voice, not stop the
        card from loading. The dashboard's enum is the place that constrains it.

        Aliases exist for the spellings someone would reasonably write — `en` for
        `en-us`, `pt` for `pt-br`, `cmn` for `zh` — because the alternative is a
        silent fall back to English on a Portuguese robot.
        """
        raw = str(value or "").strip().lower().replace("_", "-")
        if raw in cls.LANGUAGE_VOICES:
            return raw
        alias = cls.LANGUAGE_ALIASES.get(raw)
        if alias:
            return alias
        if raw:
            log.warning("[tts] unknown kokoro language %r, using %s (supported: %s)",
                        value, cls.DEFAULT_LANGUAGE, ", ".join(cls.LANGUAGES))
        return cls.DEFAULT_LANGUAGE

    @property
    def _voice(self) -> str:
        """The espeak-ng voice name for the configured language."""
        return self.LANGUAGE_VOICES[self._language]

    def _voice_ids(self, language: str) -> list:
        """The model's global speaker ids for `language`, in order."""
        return list(self._manifest.get("languages", {}).get(language) or [])

    def _resolve_sid(self, voice_index: int, language: str, strict: bool) -> int:
        """Map a per-language voice index to the model's global speaker id.

        `speaker_id` is an index *within the selected language* — 0 is the first
        voice of that language, whatever the model happens to number it. The global
        ids are not learnable: Japanese starts at 37, Spanish is 28, 29 and 53.
        Exposing them made it easy to pick an American voice, switch the language to
        Japanese, and be left wondering why the Japanese sounded wrong.

        Because the index is language-relative, an incoherent voice/language pair is
        no longer *representable* — which is why this adapter has no mismatch
        warning. An earlier version had one; designing the mistake out beats warning
        about it.

        `strict` separates the two callers. At construction an out-of-range value is
        a configuration error and must be reported. On `set_language` it must not be:
        language is a per-utterance setting that changes freely, the languages have
        different voice counts (French has exactly one), and a language switch that
        could fail would make the dropdown a trap.
        """
        ids = self._voice_ids(language)
        if not ids:
            raise RuntimeError(
                f"the Kokoro release manifest lists no voices for {language!r}; "
                f"it has {sorted(self._manifest.get('languages', {}))}"
            )
        if 0 <= voice_index < len(ids):
            return ids[voice_index]
        if strict:
            raise ValueError(
                f"speaker_id must be 0..{len(ids) - 1} for language {language!r} "
                f"({len(ids)} voices); got {voice_index}. speaker_id is an index "
                "within the selected language, not a global speaker number"
            )
        log.warning(
            "[tts] kokoro speaker_id=%d is out of range for %s (%d voices); "
            "using 0 (%s)", voice_index, language, len(ids),
            self._manifest.get("id2speaker", {}).get(str(ids[0]), "?"))
        return ids[0]

    @property
    def voice_name(self) -> str:
        """The model's own name for the resident voice, e.g. `af_heart`."""
        return self._manifest.get("id2speaker", {}).get(str(self._sid), "?")

    def _ja(self):
        """The Japanese frontend, built on first use and then kept.

        Lazy because Janome loads a ~180 MB dictionary and only `ja` needs it; a
        Chinese or English deployment must not pay for it at every start. Built
        eagerly at construction and in `set_language` when `ja` is selected, so a
        missing janome surfaces as a load error naming the build flag rather than as
        a failed utterance halfway through a tour.
        """
        if self._ja_frontend is None:
            from plugins.ja_text_norm import JapaneseFrontend
            self._ja_frontend = JapaneseFrontend()
        return self._ja_frontend

    def _direct(self):
        """The phoneme-driven runtime, built on the first Japanese utterance.

        A second session on the same weights, so it is lazy and only Japanese pays for
        it. Everything else keeps using sherpa, which is correct for those languages
        and better tested. A card configured for any of the other eight languages
        never builds this at all.

        **It runs in a separate process.** A CUDA session on this graph cannot coexist
        with sherpa's in one process — the two ONNX Runtimes share a single provider
        bridge holding one `ProviderHost` pointer, so the second session built runs
        against the wrong runtime's objects; on jp5.11 that is a SIGSEGV that kills all
        of perception. `plugins/kokoro_worker.py` has the full mechanism and the
        measurements. The boundary also buys the GPU: RTF 0.063 against 0.525
        in-process on CPU, for ~372 MB.

        Falling back to the in-process CPU session is deliberate and safe — that is
        exactly what shipped before the worker existed.
        """
        if self._direct_runtime is None:
            if self._japanese_worker:
                from plugins.kokoro_worker import DeviceUnavailable, KokoroWorkerProxy
                try:
                    self._direct_runtime = KokoroWorkerProxy(
                        self._model_dir, self._weights_name,
                        device=self._japanese_worker_device)
                    return self._direct_runtime
                except DeviceUnavailable:
                    # The configured device cannot do the job. Substituting the CPU
                    # here would hide it behind a card that still says `gpu` — the
                    # state that made Japanese "mysteriously slow" and took a
                    # measurement to explain. Let it surface.
                    raise
                except Exception as exc:                          # noqa: BLE001
                    log.warning("[tts] kokoro worker unavailable (%s); Japanese uses "
                                "the in-process CPU session", exc)
            from plugins.kokoro_direct import KokoroDirect
            self._direct_runtime = KokoroDirect(
                self._model_dir, self._weights_name)
        return self._direct_runtime

    def close(self) -> None:
        """Release the Japanese worker, if there is one.

        Called when the card stops or the engine is switched. Without it a card that
        spoke Japanese once holds the session for the adapter's whole life — true of
        the in-process runtime too, and a leak this fixes for the worker case, because
        a process exit is the only thing that returns a CUDA context.
        """
        runtime, self._direct_runtime = self._direct_runtime, None
        closer = getattr(runtime, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception as exc:                              # noqa: BLE001
                log.warning("[tts] closing the kokoro worker failed: %s", exc)

    def synthesize(self, text: str) -> bytes:
        return b''.join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        import sherpa_onnx

        # Normalise Chinese numbers here, not via sherpa's rule_fsts — see the
        # comment in __init__. Non-Chinese text comes back untouched, so espeak
        # reads "2026" in whatever language it is phonemising.
        text = self._zh_norm.normalize(text)

        if self._language == "ja":
            # Japanese leaves sherpa entirely. Its frontend phonemises with espeak,
            # whose alphabet Kokoro's misaki-derived vocabulary cannot represent, and
            # drops the misses — 12 phonemes from one sentence, audible as holes.
            # plugins/ja_phonemes produces the phonemes the model was actually
            # trained on, and plugins/kokoro_direct feeds them to the graph, because
            # sherpa has no phoneme input path.
            yield from self._synthesize_japanese(text)
            return

        # One generate() at a time. sherpa-onnx's OfflineTts is not documented as
        # thread-safe, and dispatch() runs on a ThreadingHTTPServer thread per
        # tools/call, so a config-driven speak can overlap a topic-driven one.
        with self._lock:
            config = sherpa_onnx.GenerationConfig()
            config.sid = self._sid
            config.speed = self._speed
            # Per-utterance, which is the whole reason language is not a session
            # key. Empty would fall back to meta_data.voice, but we always set it.
            config.extra = {"lang": self._voice}
            try:
                audio = self._tts.generate(text, config)
            except Exception as error:
                # Re-raise with the input attached. ONNX Runtime's own message names
                # a graph node ("SequenceInsert", "Loop") and nothing about what was
                # being said, so a failure in the field arrives as a stack trace with
                # no way to reproduce it. One robot hit
                # "SequenceInsert ... tensor to be added has a different data type"
                # on a sentence that synthesizes fine on both Orins, and the log gave
                # no voice, language or text to work from.
                raise RuntimeError(
                    f"kokoro generate failed (language={self._language}, espeak="
                    f"{self._voice}, speaker={self._voice_index}/{self.voice_name}, "
                    f"global_sid={self._sid}, {len(text)} chars): {error}\n"
                    f"  text: {text!r}"
                ) from error

        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            log.warning("[tts] kokoro produced no audio for %r", text)
            return
        # Resample the *whole* utterance, then frame it. Doing it per 3200-byte
        # frame would restart the filter every 100 ms and inject a transient each
        # time — see utils/resample.resample_poly.
        pcm = downsample_24k_to_16k(samples)
        for i in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[i:i + CHUNK_BYTES]

    def _synthesize_japanese(self, text: str):
        """Kanji -> kana -> misaki phonemes -> the ONNX graph, bypassing sherpa."""
        from plugins import ja_phonemes

        kana = self._ja().to_kana_only(text)
        phonemes = ja_phonemes.kana_to_phonemes(kana)
        if not phonemes.strip():
            log.warning("[tts] nothing speakable left in %r after Japanese "
                        "normalisation", text)
            return

        with self._lock:
            samples = self._direct().synthesize(
                phonemes, speaker_id=self._sid, speed=self._speed)

        if samples.size == 0:
            log.warning("[tts] kokoro_direct produced no audio for %r (%r)",
                        text, phonemes)
            return
        pcm = downsample_24k_to_16k(samples)
        for i in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[i:i + CHUNK_BYTES]

    def set_speed(self, speed: float) -> None:
        # Applied per generate() call, so nothing reloads — same as the other two.
        self._speed = speed

    def set_language(self, language: str) -> None:
        """Change the pronunciation language on the resident model.

        The point of the whole design: `lang` rides in GenerationConfig.extra, so
        this costs a field assignment rather than the ~2.7 s session rebuild a
        session key would. Which is why `tts_language` is deliberately absent from
        _session_keys.

        The voice moves with the language, because `speaker_id` is an index within
        it — switching to `ja` gives voice 0 of Japanese, not whatever global id the
        English voice 0 happened to occupy. Out of range clamps rather than raising:
        French has one voice, and a free per-utterance setting must not be able to
        fail. `sid` is a per-generate() argument, so none of this reloads anything.
        """
        resolved = self._normalize_language(language)
        if resolved == self._language:
            return
        self._language = resolved
        self._sid = self._resolve_sid(self._voice_index, resolved, strict=False)
        if resolved == "ja":
            # Build now, not on the first utterance: a missing janome should stop
            # the config call with a clear error, not a speak.
            self._ja()
        log.info("[tts] kokoro language -> %s (espeak %s), voice -> %s (global %d)",
                 resolved, self._voice, self.voice_name, self._sid)


def _validate_kokoro_manifest(model_dir: str) -> dict:
    """Check the release before sherpa-onnx gets a chance to exit(-1) or mis-tune.

    Mirrors _validate_thai_manifest above and _validate_manifest in
    plugins/vits2_tts_trt/runtime/backends/trt_numpy_tts_engine.py: the release
    carries a manifest written by the repack script from the ONNX graph's own
    metadata, so a mismatched or hand-assembled directory fails here as an ordinary
    exception rather than during model load.

    Two of these checks cannot be made later:

    - `model_version` must be 2. Only Kokoro >= 1.0 honours `lang`; on a v0.19
      graph sherpa-onnx takes the PiperPhonemizeLexicon path and the language
      selector would silently do nothing.
    - `sample_rate` must be 24000, because utils/resample.py's filter is designed
      for that ratio and nothing downstream can be told about another one.

    Returns the parsed manifest, which the adapter uses for its speaker-name and
    language-coherence checks.
    """
    import os

    manifest_path = os.path.join(model_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} is missing; this release was not produced by "
            "tools/repack_kokoro_v1_0.py and cannot be checked before load"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    version = int(manifest.get("model_version") or 0)
    if version != 2:
        raise RuntimeError(
            f"Kokoro release declares model_version={version}; the language "
            "selector needs 2 (Kokoro >= 1.0), and v0.19 ignores `lang` entirely"
        )
    rate = int(manifest.get("sample_rate") or 0)
    if rate != KOKORO_SAMPLE_RATE:
        raise RuntimeError(
            f"Kokoro release is {rate} Hz but the adapter resamples "
            f"{KOKORO_SAMPLE_RATE} Hz -> {SAMPLE_RATE} Hz; another rate needs its "
            "own filter in utils/resample.py first"
        )
    if not manifest.get("id2speaker"):
        raise RuntimeError(
            "Kokoro release manifest has no id2speaker map; the adapter needs it "
            "to check that the voice matches the language"
        )
    return manifest


def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    import os
    from utils.onnx_provider import normalize_device
    engine = str(cfg.get('engine', '')).lower()
    default_dir = ENGINE_MODEL_DIRS.get(engine, '/models/sherpa-onnx/tts')
    model_dir = cfg.get('model_dir', default_dir)
    speaker_id = int(cfg.get('speaker_id', 0))
    speed = float(cfg.get('speed', 1.0))
    # An engine may want somewhere other than cpu when nothing is configured;
    # normalize_device turns an absent value into 'cpu', so the default has to be
    # applied before it, not after.
    device = normalize_device(
        cfg.get('device') or ENGINE_DEVICE_DEFAULTS.get(engine),
        cfg.get('hw_provider'),
    )
    if engine == 'mms-th':
        return MmsThaiTTSAdapter(
            model_dir, speaker_id, speed, device,
            phrase_spacing=bool(cfg.get('thai_phrase_spacing', False)),
        )
    if engine == 'kokoro-multi':
        return KokoroTTSAdapter(
            model_dir, speaker_id, speed, device,
            # `tts_language` is the configSchema name the dashboard sends;
            # `language` is the short form config.yaml uses. Both accepted, the
            # same way the facade takes `tts_engine` or `engine`.
            language=(cfg.get('tts_language') or cfg.get('language')
                      or KokoroTTSAdapter.DEFAULT_LANGUAGE),
            # Japanese runs in its own process — the only way a CUDA session on this
            # graph can coexist with sherpa's. Not exposed in configSchema: it is a
            # correctness constraint, not an operator preference. config.yaml can
            # turn it off to fall back to the in-process CPU session.
            japanese_worker=bool(cfg.get('japanese_worker', True)),
            japanese_worker_device=str(cfg.get('japanese_worker_device') or 'gpu'),
        )
    return MatchaTTSAdapter(model_dir, speaker_id, speed, device)


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class _TTSNode(Node):
    def __init__(self, input_topic: Optional[str], adapter: Optional[TTSAdapter], node_suffix: str = ''):
        node_name = f"tts_{node_suffix}" if node_suffix else "tts"
        super().__init__(node_name)
        self._input_topic  = input_topic or ''
        self._output_topic = f"{input_topic}/tts" if input_topic else '/perception/tts'
        self._adapter      = adapter
        self.state         = "idle"
        self._text_queue   = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        self._interrupt_flag = threading.Event()  # 打断标志：设置后立即停止当前 utterance
        # 每次 interrupt 递增的「代」。入队时给 utterance 打上当时的代号，worker
        # 就能区分「打断之前入队」（丢弃）和「打断之后入队」（正常播）。单靠
        # _interrupt_flag 做不到：它是粘性的，空闲时收到的打断没有任何 utterance
        # 循环去消费它，于是残留下来把**下一句**吞掉。
        self._interrupt_lock = threading.Lock()
        self._interrupt_gen = 0
        from audio_msgs.msg import AudioChunk
        self._pub = self.create_publisher(AudioChunk, self._output_topic, _LOW_LAT_QOS)
        self._perf_pub = self.create_publisher(String, '/perception/perf_spans', _LOW_LAT_QOS)
        if input_topic:
            self._sub = self.create_subscription(String, self._input_topic, self._text_cb, _LOW_LAT_QOS)
        else:
            self._sub = None
        log.info(f"[tts] node created: subscribing={self._input_topic or '(none)'}, publishing={self._output_topic}")

    def start(self) -> dict:
        while not self._text_queue.empty():
            try: self._text_queue.get_nowait()
            except Exception: break
        if self.state == "running":
            return self._status_dict()
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        # Dry-run: verify model can synthesize before declaring running. The probe
        # comes from the adapter, not a literal here — see TTSAdapter.dry_run_text.
        probe = getattr(self._adapter, "dry_run_text", ".")
        try:
            test_chunks = list(self._adapter.synthesize_stream(probe))
            if not test_chunks:
                return {"state": "error",
                        "message": f"TTS dry-run produced no audio for {probe!r}"}
        except Exception as e:
            return {"state": "error", "message": f"TTS dry-run failed: {e}"}
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self._status_dict()

    def stop(self) -> dict:
        self._stop_event.set()
        self._complete_discarded_actions()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def _complete_discarded_actions(self) -> int:
        """Cancel queued ACP actions that will never reach the worker."""
        discarded = []
        while True:
            try:
                item = self._text_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                text = str(item[0])
                action_id = item[2] if len(item) >= 3 else ''
            else:
                text, action_id = str(item), ''
            discarded.append((text, action_id))
        for text, action_id in discarded:
            _complete_action(action_id, text, 0, interrupted=True)
        return len(discarded)

    def interrupt(self) -> dict:
        """立即中止当前播放：清空队列 + 设置 interrupt flag 让 worker 停止当前 utterance。

        空闲时调用、或连续调用两次都是安全的 —— agent-core 的 barge-in 兜底打断和
        LLM 自己显式调的 `tts(action=interrupt)` 经常在几秒内先后到达。
        """
        # 清空待播放队列。丢掉的 item 必须逐个回 ACP cancelled，否则 agent-core
        # 的 barrier 会为每个注册过的 action 干等到超时。
        cleared = self._complete_discarded_actions()
        with self._interrupt_lock:
            self._interrupt_gen += 1
            # 递增和置位放在同一个临界区：否则一个在新代号下入队的 utterance 可能
            # 先被 worker 装载（清掉 flag），再被这里的 set() 误杀。
            self._interrupt_flag.set()
        log.info(f"[tts] interrupted: cleared {cleared} queued item(s)")
        return {"status": "interrupted", "cleared": cleared}

    def _current_gen(self) -> int:
        with self._interrupt_lock:
            return self._interrupt_gen

    def enqueue(self, text: str, trace_id: str = '', action_id: str = ''):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        # One queue item = one utterance = one EOF = one ACP action. The 280-char
        # split happens inside the worker instead: splitting here put each
        # segment on the queue as its own utterance, so a long text emitted an
        # EOF and reset the pacing clock every 280 characters, and the gap
        # between segments was a full synthesis with no audio flowing at all.
        self._text_queue.put((text, trace_id, action_id, self._current_gen()))

    @staticmethod
    def _split_text(text: str, max_chars: int = 280) -> list:
        """按标点分段，每段不超过 max_chars 字。"""
        import re as _re
        sentences = _re.split(r'(?<=[。！？；\n])', text)
        segments = []
        current = ""
        for sent in sentences:
            if not sent:
                continue
            if len(current) + len(sent) > max_chars and current:
                segments.append(current)
                current = sent
            else:
                current += sent
        if current:
            segments.append(current)
        return segments if segments else [text]

    def _text_cb(self, msg: String):
        if self.state != "running": return
        try:
            text = json.loads(msg.data).get("text","")
        except Exception:
            text = msg.data.strip()
        if text:
            log.info(f"[tts] received text from topic: {text[:50]}...")
            self._text_queue.put((text, '', '', self._current_gen()))

    def _worker(self):
        from audio_msgs.msg import AudioChunk
        import time as _time

        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            # Unpack queue item: (text, trace_id, action_id, gen) or legacy
            # formats. A missing gen means "cannot be stale", so such an item is
            # always played rather than silently dropped.
            _gen = None
            if isinstance(item, tuple):
                if len(item) >= 4:
                    text, _trace_id, _action_id, _gen = item[:4]
                elif len(item) == 3:
                    text, _trace_id, _action_id = item
                elif len(item) == 2:
                    text, _trace_id = item
                    _action_id = ''
                else:
                    text, _trace_id, _action_id = str(item[0]), '', ''
            else:
                text, _trace_id, _action_id = item, '', ''
            # 只丢弃早于最后一次打断的 utterance。空闲时收到的打断不能碰下一句 ——
            # 那正是以前把一整句回复吞掉的原因。
            with self._interrupt_lock:
                _stale = _gen is not None and _gen < self._interrupt_gen
                if not _stale:
                    # 为本句装载：上一次打断留下的 flag 到此为止，只有从现在起
                    # 到达的打断才能取消它。
                    self._interrupt_flag.clear()
            if _stale:
                self._publish_eof()
                _complete_action(_action_id, text, 0, interrupted=True)
                continue
            synth_thread = None
            # Set on every exit path. The stop/interrupt flags are not enough to
            # release the synth thread: if the consumer dies on an exception,
            # nothing is cancelled and nothing is draining, so a blocking put
            # would wedge that thread forever. Bound before the try so the
            # finally can always reach it.
            utterance_abort = threading.Event()
            try:
                t_start = _time.monotonic()
                t_start_wall = _time.time()  # wall-clock for perf span
                t0_wall = None  # wall-clock when playback starts (prebuf complete)
                total = 0
                buf = b''
                t0 = None  # monotonic start of the pacing schedule
                frames_sent = 0
                prebuf = []   # pre-buffer queue

                def publish(frame: bytes) -> None:
                    nonlocal frames_sent
                    msg = AudioChunk()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.format = "audio/pcm-16k"
                    msg.data = list(frame)
                    self._pub.publish(msg)
                    frames_sent += 1

                def emit(frame: bytes) -> None:
                    """Pace and publish one frame; latches the clock on the first."""
                    nonlocal t0, t0_wall
                    if t0 is None:
                        # Backdate by the frames already in the prebuffer so all
                        # of them are due in the past and go out in one burst —
                        # the consumer then starts holding PREBUF_FRAMES of audio.
                        now = _time.monotonic()
                        t0 = now - max(0, len(prebuf) - 1) * FRAME_INTERVAL_S
                        t0_wall = _time.time()
                        fire_hook("on_speaking")
                    target = t0 + frames_sent * FRAME_INTERVAL_S
                    now = _time.monotonic()
                    if now < target:
                        _time.sleep(target - now)
                    # No rebase when behind: publishing immediately is the
                    # catch-up, since the schedule is already ahead of realtime.
                    publish(frame)

                def flush_prebuf() -> None:
                    while prebuf:
                        emit(prebuf[0])
                        prebuf.pop(0)

                # 分段：超过 280 字按标点切分，避免超长合成导致延迟或失败。分段是
                # 合成的实现细节 —— pacing/prebuffer/EOF 都跨段延续，下游看到的
                # 仍然是一句完整的话。
                segments = self._split_text(text, max_chars=280)
                if len(segments) > 1:
                    log.info(f"[tts] split {len(text)} chars into {len(segments)} segments")

                # Synthesis runs on its own thread. This adapter's
                # synthesize_stream calls generate() once per segment and blocks
                # until the whole segment's audio exists, so doing it on the
                # publishing thread meant no frame at all went out for the
                # duration of every segment after the first — a guaranteed gap
                # every 280 characters, as long as the synthesis took. Here
                # segment N+1 is synthesized while segment N is still playing.
                frame_queue: queue.Queue = queue.Queue(maxsize=SYNTH_QUEUE_FRAMES)
                synth_state: dict = {"total": 0}

                def enqueue_frame(frame) -> bool:
                    """Blocking put that still honours interrupt/stop/abort."""
                    while True:
                        if (self._stop_event.is_set() or self._interrupt_flag.is_set()
                                or utterance_abort.is_set()):
                            return False
                        try:
                            frame_queue.put(frame, timeout=0.1)
                            return True
                        except queue.Full:
                            continue

                def synthesize_into_queue() -> None:
                    pending = b''
                    try:
                        for seg in segments:
                            for raw_chunk in self._adapter.synthesize_stream(seg):
                                pending += raw_chunk
                                synth_state["total"] += len(raw_chunk)
                                while len(pending) >= CHUNK_BYTES:
                                    frame, pending = pending[:CHUNK_BYTES], pending[CHUNK_BYTES:]
                                    if not enqueue_frame(frame):
                                        return
                        if pending:
                            enqueue_frame(pending)
                    except BaseException as exc:  # surfaced on the worker thread
                        synth_state["error"] = exc
                    finally:
                        # Unblock the consumer on every exit path.
                        enqueue_frame(_SYNTH_DONE)

                synth_thread = threading.Thread(
                    target=synthesize_into_queue, name="tts-synth", daemon=True)
                synth_thread.start()

                interrupted = False
                while True:
                    if self._stop_event.is_set() or self._interrupt_flag.is_set():
                        interrupted = True
                        break
                    try:
                        frame = frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if frame is _SYNTH_DONE:
                        break
                    if len(frame) < CHUNK_BYTES:
                        # Trailing partial frame — paced, then done.
                        buf = frame
                        break
                    if t0 is None:
                        prebuf.append(frame)
                        if len(prebuf) >= PREBUF_FRAMES:
                            flush_prebuf()
                        continue
                    emit(frame)

                cancelled = self._stop_event.is_set() or self._interrupt_flag.is_set()
                synth_thread.join(timeout=5)
                if "error" in synth_state and not cancelled:
                    raise synth_state["error"]
                total = synth_state["total"]  # read after join, not concurrently
                # Flush any remaining pre-buffer (utterances < PREBUF_FRAMES)
                if prebuf and not cancelled:
                    flush_prebuf()

                # flush remainder
                if buf and not cancelled:
                    if t0 is not None:
                        target = t0 + frames_sent * FRAME_INTERVAL_S
                        now = _time.monotonic()
                        if now < target:
                            _time.sleep(target - now)
                    publish(buf)

                # Capture before clearing: reading the flag after the clear below
                # made this always False, so an interrupted utterance reported ACP
                # "completed" instead of "cancelled".
                was_interrupted = self._interrupt_flag.is_set()
                # 这里刻意不再 clear()。flag 的装载/解除只在出队时、_interrupt_lock
                # 下进行；在这条路径上也清一次会和并发的 interrupt() 抢跑，把信号
                # 丢给下一句。
                if was_interrupted:
                    log.info(f"[tts] utterance interrupted after {frames_sent} frames")
                else:
                    log.info(f"[tts] spoke {len(text)} chars → {total} bytes ({frames_sent} frames) in {_time.monotonic() - t_start:.2f}s")

                # 发布 EOF 标记：告知下游 Speaker 当前 utterance 已结束
                self._publish_eof()
                # 上报 TTS perf spans（生成 + 播放）
                try:
                    import json as _json
                    t_end_wall = _time.time()
                    spans = []
                    _span_base = {"type": "perf_span", "component": "perception"}
                    if _trace_id:
                        _span_base["trace_id"] = _trace_id
                    if t0_wall:
                        spans.append({**_span_base, "span": "tts_generate",
                                      "start_ts": t_start_wall, "end_ts": t0_wall,
                                      "meta": {"chars": len(text)}})
                        spans.append({**_span_base, "span": "tts_playback",
                                      "start_ts": t0_wall, "end_ts": t_end_wall,
                                      "meta": {"frames": frames_sent}})
                    else:
                        # 没有 prebuf（极短文本），合并为一个 span
                        spans.append({**_span_base, "span": "tts_generate",
                                      "start_ts": t_start_wall, "end_ts": t_end_wall,
                                      "meta": {"chars": len(text), "frames": frames_sent}})
                    for sp in spans:
                        perf_msg = String()
                        perf_msg.data = _json.dumps(sp)
                        self._perf_pub.publish(perf_msg)
                except Exception:
                    pass

                # ACP: 推送动作完成回调到 Agent Core
                # Also fire on_idle hook (LED off immediately after playback)
                if _action_id:
                    fire_hook("on_idle")
                    _complete_action(_action_id, text, frames_sent, was_interrupted)
            except Exception as e:
                log.error(f"[tts] synthesis error: {e}", exc_info=True)
                _complete_action(_action_id, text, 0, interrupted=True)
                self._publish_eof()
            finally:
                # Release the synth thread on every path, including the one where
                # the consumer above died: it can otherwise sit on a blocking put
                # forever, holding a reference to the adapter.
                utterance_abort.set()
                if synth_thread is not None and synth_thread.is_alive():
                    synth_thread.join(timeout=5)
                    if synth_thread.is_alive():
                        log.error("[tts] synth thread did not exit")

    def _status_dict(self) -> dict:
        return {
            "state":     self.state,
            "topic_in":  [{"topic": self._input_topic,  "format": "data/json",     "desc": "text to synthesize"}],
            "topic_out": [{"topic": self._output_topic, "format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
        }

    def _publish_eof(self):
        """发布 EOF magic chunk，标记当前 utterance 结束。"""
        from audio_msgs.msg import AudioChunk
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "audio/pcm-16k"
        msg.data = list(AUDIO_EOF_MAGIC)
        self._pub.publish(msg)


# ── Plugin ────────────────────────────────────────────────────────────────────

# Every shared field on the tool, derived from the schema rather than restated.
# TTSPlugin._config has to carry these into a newly built engine, and the list was
# previously a hardcoded ("speaker_id", "speed", "device") that missed
# thai_phrase_spacing — so a switch built the engine without it, and the next
# config saw a change and rebuilt the session that had just been built (2.7 s).
# Deriving it means a new configSchema field cannot be forgotten here again.
SHARED_CONFIG_KEYS = tuple(
    key for key, spec in TOOLS[0]["configSchema"]["properties"].items()
    if spec.get("scope") == "shared" and key != "tts_engine"
)


def _session_keys(cfg: dict) -> dict:
    """The subset of config the loaded sherpa-onnx session is built from.

    Normalised, because the comparison in `config` is only meaningful if both
    sides went through the same conversion: the dashboard sends `speed: 1` where
    config.yaml has `1.0`, and `device` may arrive as any string. `speed` is
    deliberately absent — it is applied per generate() call, so changing it must
    not reload anything.

    Used at construction as well as on config: without it the first config after
    an engine switch always looked like a change (the new plugin's _cfg had the
    raw config.yaml values, or none at all for a dashboard-only field like
    thai_phrase_spacing) and rebuilt a session that had just been built. Measured:
    that one extra rebuild cost 2.7 s.
    """
    out: dict = {}
    if 'speaker_id' in cfg and cfg['speaker_id'] is not None:
        out['speaker_id'] = int(cfg['speaker_id'])
    if cfg.get('device'):
        out['device'] = str(cfg['device'])
    if cfg.get('model_dir'):
        out['model_dir'] = str(cfg['model_dir'])
    if 'thai_phrase_spacing' in cfg:
        out['thai_phrase_spacing'] = bool(cfg['thai_phrase_spacing'])
    return out


class SherpaOnnxTTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg      = plugin_cfg
        # Normalise the session keys up front so the first `config` that repeats
        # them compares equal instead of rebuilding what was just built.
        self._cfg.update(_session_keys(plugin_cfg))
        self._loading  = False
        self._load_error = None
        try:
            self._adapter  = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] failed to load model: {e}", exc_info=True)
            self._adapter = None
            self._load_error = str(e)
        else:
            # Pay the first-inference cost here rather than on whoever speaks
            # first — 1695 ms on Orin5's gpu for the Thai model. `warmup` is a
            # config.yaml key that both sherpa-onnx engines silently ignored until
            # now; only the VITS2 plugin honoured it.
            #
            # A failed warmup is logged, not fatal: the model loaded, and
            # _TTSNode.start()'s dry run is the gate that decides whether it can
            # actually speak. Refusing here would turn a slow first utterance into
            # a dead card.
            if plugin_cfg.get("warmup", True):
                try:
                    started = time.monotonic()
                    warmed = self._adapter.warmup()
                    log.info("[tts] warmup: %d bytes in %.2fs", warmed,
                             time.monotonic() - started)
                    if not warmed:
                        log.warning("[tts] warmup produced no audio")
                except Exception:
                    log.warning("[tts] warmup failed; the first utterance will "
                                "pay the cold-start cost", exc_info=True)
        self._nodes: dict[str, _TTSNode] = {}
        # main.py serves MCP over ThreadingHTTPServer, so start/stop/speak/config
        # can run concurrently. Every read-modify-write of _nodes must hold this:
        # otherwise two threads both pass a "key not in _nodes" check, both build
        # a node, and the dict keeps only the last — leaving the other running but
        # unreachable, with a duplicate publisher on the same topic that nothing
        # can stop. See perception/README.md § Plugin Concurrency.
        # RLock: dispatch paths nest (start → _dispose_node).
        self._nodes_lock = threading.RLock()
        self._executor = executor
        log.info(f"[tts] plugin init: sherpa-onnx VITS, "
                 f"speaker_id={plugin_cfg.get('speaker_id', 0)}, speed={plugin_cfg.get('speed', 1.0)}")

    def _dispose_node(self, node: _TTSNode, key: str = "") -> dict:
        """Stop a node and release its ROS endpoints. Caller holds _nodes_lock.

        destroy_node() matters: without it the publisher and the ROS node name
        outlive the node object, so a later start on the same key collides with a
        still-registered ghost.
        """
        result = {"state": "idle"}
        try:
            result = node.stop()
        except Exception:
            log.error(f"[tts] node.stop() failed while disposing '{key}'", exc_info=True)
        try:
            self._executor.remove_node(node)
        except Exception as error:
            log.warning(f"[tts] failed to remove ROS node '{key}': {error}")
        try:
            node.destroy_node()
        except Exception as error:
            log.warning(f"[tts] failed to destroy ROS node '{key}': {error}")
        return result

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._loading:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "loading",
                    "desc": "Downloading TTS model...",
                }
            if self._load_error:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "error",
                    "desc": f"Model load failed: {self._load_error}",
                }
            input_topic = args.get("input_topic", "")
            # Snapshot under the lock: info is a heartbeat probe and iterating the
            # live dict can raise "dictionary changed size" mid-start.
            with self._nodes_lock:
                node = self._nodes.get(instance_id) if instance_id else None
                nodes_snapshot = list(self._nodes.values())
            if instance_id and node is not None:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": node.state,
                    "topic_in":  [{"topic": node._input_topic,  "format": "data/json",     "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "audio/pcm-16k", "desc": ""}],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            if instance_id:
                # Instance requested but not running — return inferred topics for this instance only.
                inferred_out = f"{input_topic}/tts" if input_topic else "/perception/tts"
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "idle",
                    "topic_in":  [{"topic": input_topic,  "format": "data/json",     "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            # Aggregate info (no instance_id = ping/overview only)
            if nodes_snapshot:
                topics_in = [{"topic": n._input_topic, "format": "data/json", "desc": ""} for n in nodes_snapshot]
                topics_out = [{"topic": n._output_topic, "format": "audio/pcm-16k", "desc": ""} for n in nodes_snapshot]
                states = list(set(n.state for n in nodes_snapshot))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/tts" if input_topic else "/perception/tts"
                topics_in = [{"topic": input_topic, "format": "data/json", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}]
                state = "idle"
            return {
                "name": "TTS", "manufacture": "Embodied", "model": "tts",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "TTS service — converts text to audio/pcm-16k",
            }

        elif action == "start":
            if self._loading:
                return {"state": "loading", "message": "TTS model is being downloaded, please wait..."}
            if self._load_error:
                return {"state": "error", "message": f"TTS model failed to load: {self._load_error}"}
            if not self._adapter:
                return {"state": "error", "message": "TTS model not loaded"}
            input_topic = args.get("input_topic") or ''
            node_key = instance_id or input_topic or '_default'
            with self._nodes_lock:
                # Clean up _default node if it would conflict with this instance
                if '_default' in self._nodes and node_key != '_default':
                    default_node = self._nodes['_default']
                    if default_node._input_topic == input_topic or default_node._output_topic == (f"{input_topic}/tts" if input_topic else '/perception/tts'):
                        del self._nodes['_default']
                        self._dispose_node(default_node, '_default')
                node = self._nodes.get(node_key)
                if node is None:
                    node = _TTSNode(input_topic or None, self._adapter,
                                    node_suffix=node_key.replace('/', '_').replace('-', '_'))
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                elif input_topic and node._input_topic != input_topic:
                    # Input topic changed for existing instance — recreate
                    del self._nodes[node_key]
                    self._dispose_node(node, node_key)
                    node = _TTSNode(input_topic, self._adapter,
                                    node_suffix=node_key.replace('/', '_').replace('-', '_'))
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                return node.start()

        elif action == "stop":
            with self._nodes_lock:
                if instance_id:
                    node = self._nodes.pop(instance_id, None)
                    if node is None:
                        return {"state": "idle"}
                    return self._dispose_node(node, instance_id)
                for key in list(self._nodes.keys()):
                    self._dispose_node(self._nodes.pop(key), key)
                return {"state": "idle"}

        elif action == "speak":
            if self._loading:
                return {"state": "loading", "message": "TTS model is being downloaded, please wait..."}
            if self._load_error or not self._adapter:
                return {"state": "error", "message": f"TTS model not available: {self._load_error or 'not loaded'}"}
            text = args.get("text", "")
            if not text:
                raise ValueError("text is required")
            # Find any existing running node to reuse
            with self._nodes_lock:
                node = None
                for n in self._nodes.values():
                    if n.state == "running":
                        node = n
                        break
                if node is None:
                    # No running node — use instance key or fallback
                    node_key = instance_id or '_default'
                    node = self._nodes.get(node_key)
                    if node is None:
                        input_topic = args.get("input_topic") or None
                        # No per-instance adapter: `config` writes into self._cfg
                        # globally (it strips instance_id), so there has never
                        # been anywhere to read a per-instance config from. This
                        # used to consult a self._instance_configs that is never
                        # assigned anywhere, i.e. `speak` with an instance_id and
                        # no running node raised AttributeError instead of
                        # synthesizing.
                        node = _TTSNode(input_topic, self._adapter,
                                        node_suffix=node_key.replace('/', '_').replace('-', '_'))
                        self._executor.add_node(node)
                        self._nodes[node_key] = node
                    if node.state != "running":
                        node.start()
            # ACP: 生成 action_id
            import uuid as _uuid
            action_id = f"speak-{_uuid.uuid4().hex[:8]}"
            node.enqueue(text, trace_id=args.get('_trace_id', ''), action_id=action_id)
            return {"status": "queued", "action_id": action_id, "text": text}

        elif action == "config":
            # No `and v` filter here. It used to drop every falsy value, which made
            # `thai_phrase_spacing: False` and `speaker_id: 0` unsendable — you
            # could turn phrase spacing on and never off. _session_keys() decides
            # per key whether a value counts as present, so an empty `device` is
            # still ignored while `False` is honoured.
            cfg = {k: v for k, v in args.items()
                   if k not in ('action', 'instance_id')}
            # Rebuild ONLY when something the loaded session is built from actually
            # changed. This used to rebuild unconditionally, and an identical no-op
            # config measured **2.6-2.8 s** on Orin5 for the Thai model — a full
            # ONNX session teardown and reload, plus a re-verification of the model
            # archive — and then disposed every node, forcing the card to start
            # again. The dashboard re-applies a card's config around a speak, so
            # that cost was paid on *every* utterance: "every speak takes 5 s".
            #
            # `speed` is deliberately not in this set. sherpa-onnx takes it on each
            # generate() call, so it is a scale on the resident model — the same
            # reason vits2's _config only calls set_speed (see
            # plugins/vits2_tts_trt/plugin.py: "avoids tearing down a resident
            # model for a slider change"). vits2 already behaved this way, which is
            # why only the two sherpa-onnx engines were slow.
            #
            # `tts_language` is not in it either, for exactly the same reason:
            # Kokoro reads `lang` out of GenerationConfig.extra on every call, so a
            # language change is a field assignment. Putting it in _session_keys
            # would cost a 310 MB fp32 session rebuild per dropdown change.
            #
            # Values are normalised before comparing, or `speed: 1` from the
            # dashboard would differ from a stored `1.0` and defeat the check.
            incoming = _session_keys(cfg)
            speed = float(cfg['speed']) if 'speed' in cfg else None
            language = str(cfg['tts_language']) if cfg.get('tts_language') else None

            needs_rebuild = any(self._cfg.get(key) != value
                                for key, value in incoming.items())
            self._cfg.update(incoming)
            # Both of these have to be written back explicitly, because they are
            # absent from `incoming` by design and _build_tts_adapter reads them out
            # of self._cfg. Without this, a rebuild triggered by some *other* key
            # would construct the adapter with the default language rather than the
            # one the operator selected.
            if speed is not None:
                self._cfg['speed'] = speed
            if language is not None:
                self._cfg['tts_language'] = language

            if needs_rebuild or self._adapter is None:
                changed = sorted(incoming) if needs_rebuild else ['(no model loaded)']
                log.info("[tts] rebuilding the adapter: %s changed", ", ".join(changed))
                self._adapter = _build_tts_adapter(self._cfg)
                self._load_error = None
                # Nodes hold the old adapter, so they have to go — but only when
                # there really is a new adapter for them to pick up.
                with self._nodes_lock:
                    for key in list(self._nodes.keys()):
                        self._dispose_node(self._nodes.pop(key), key)
            else:
                if speed is not None:
                    self._adapter.set_speed(speed)
                if language is not None:
                    # A no-op on every engine but Kokoro; TTSAdapter defines it so
                    # this does not need to know which one is resident.
                    self._adapter.set_language(language)
            return {"status": "configured", "rebuilt": bool(needs_rebuild)}

        elif action == "interrupt":
            # 立即中止所有 TTS 播放（清空队列 + 停止当前 utterance）
            total_cleared = 0
            interrupted_count = 0
            with self._nodes_lock:
                if instance_id:
                    targets = [self._nodes[instance_id]] if instance_id in self._nodes else []
                else:
                    targets = [n for n in self._nodes.values() if n.state == "running"]
            for node in targets:
                result = node.interrupt()
                total_cleared += result.get('cleared', 0)
                interrupted_count += 1
            return {"status": "interrupted", "nodes": interrupted_count, "cleared": total_cleared}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        return self._adapter.synthesize(text)


DEFAULT_TTS_ENGINE = "vits2-zh-en"
# `<model>-<languages>`, the shape `asr_model` in plugins/asr.py already uses
# (x-asr-zh-en, paraformer-zh-en, zipformer-en). Two reasons to match it rather
# than invent a second convention: the dashboard renders the raw enum string, so
# this is what an operator reads in the dropdown right next to the ASR one; and
# naming an engine after its runtime said nothing about what you would hear —
# `matcha` and `mms-th` both run on sherpa-onnx.
#
# Language codes, not country codes: `zh`, not `cn`.
TTS_ENGINES = ("vits2-zh-en", "matcha-zh-en", "mms-th", "kokoro-multi")
# Older spellings, still accepted. `vits2_trt` and `sherpa_onnx` are not
# decoration: both are already persisted in ConfigDB rows and in config.yaml on
# every deployed robot, and _select_engine raises on an unknown engine — so
# dropping them would turn every existing TTS card into "Unsupported TTS engine"
# on the next restart. The bare `vits2`/`matcha`/`mms_thai` forms existed only on
# this branch before the languages were added, and cost one line each to keep.
ENGINE_ALIASES = {
    "vits2-trt": "vits2-zh-en",
    "vits2": "vits2-zh-en",
    "sherpa-onnx": "matcha-zh-en",
    "matcha": "matcha-zh-en",
    "mms-thai": "mms-th",
    "mms-tts-thai": "mms-th",
}
# Where each engine keeps its own model files. Used for any engine other than
# the one config.yaml was written for; see TTSPlugin._model_dir_for.
ENGINE_MODEL_DIRS = {
    "vits2-zh-en": "/models/vits2",
    # Kept at the old path: it is already populated on deployed robots and
    # renaming it would force every one of them to re-download the Matcha pair.
    "matcha-zh-en": "/models/sherpa-onnx/tts",
    # Its own directory, not a sibling file in the Matcha one: both engines call
    # their weights by different names but share nothing, and pointing them at one
    # directory is the mistake ENGINE_MODEL_DIRS exists to prevent.
    "mms-th": "/models/mms-th",
    # Its own tree, and the archives land in `<dir>/gpu` and `<dir>/cpu` beneath
    # it: the two devices load different weight files (fp32 vs int8), so they are
    # separate downloads that must not overwrite each other.
    "kokoro-multi": "/models/kokoro-multi",
}
# Where an engine wants to run when nothing has been configured. Only Kokoro
# differs: it is a 310 MB fp32 graph on gpu and the CUDA provider is worth having,
# whereas Matcha and MMS are small enough that cpu is a reasonable default.
#
# Known wart: the dashboard renders the configSchema's `device.default` ("cpu"),
# so the form and the effective default disagree until someone touches the field.
# JSON Schema cannot express a per-engine default, and a second engine-specific
# device field would be worse — the value is stated in the field description
# instead.
ENGINE_DEVICE_DEFAULTS = {
    "kokoro-multi": "gpu",
}
# How long an `action=config` engine switch waits for the new engine before
# answering `loading`. Sized so the bounded part of a build finishes inside it
# (constructing the sherpa session measured ~2 s on cpu, ~5 s on gpu) while a cold
# model download does not — that one genuinely has to go async. Also under the 30 s
# the LLM path allows a tools/call (agent-core/src/mcp_client.py), in case a config
# ever arrives from there rather than from the dashboard.
ENGINE_SWITCH_WAIT_S = 20


class TTSPlugin:
    """The single public TTS plugin, delegating to a config-selected engine.

    A facade rather than a `__new__` switch: the engine is a configSchema field,
    so it can change at runtime (`action=config`, `tts_engine=...`) and not only
    at process start. Switching disposes the previous engine's nodes and builds
    the new one on a background thread, because sherpa-onnx downloads its Matcha
    model in its constructor and that is open-ended.

    `action=config` then waits up to ENGINE_SWITCH_WAIT_S for that build and only
    answers `loading` if it is still going, so the bounded part — constructing the
    session, ~2 s on cpu and ~5 s on gpu — is not something callers have to poll
    for. It can afford to wait: the dashboard's start-project path
    (agent-core/src/api/mcp_manage.py mcp_call_tool) sets no client timeout at all,
    and its loading watcher polls for up to 900 s. The 60 s figure that used to be
    cited here belongs to the *LLM* tool path (agent-core/src/mcp_client.py), which
    does not send config.
    """

    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor
        self._lock = threading.Lock()
        self._impl = None
        self._impl_engine = ""
        self._building = ""          # engine name while a build is in flight
        self._build_error = None
        # `start` calls that arrived while a build was in flight, keyed by
        # instance_id, replayed against the new engine once it is resident. Agent
        # Core's contract for a start that answers `state: loading` is that the
        # tool will reach `running` on its own — it polls `info` and reports
        # "启动已取消" if it ever sees `idle` (see agent-core api/config.py
        # _settle_loading_item). Dropping the start satisfied the letter of
        # "never block" and broke that contract: switching engine and starting in
        # the same batch, which is exactly what the dashboard does, left the card
        # cancelled even though the engine had loaded fine.
        self._pending_starts: dict[str, dict] = {}
        engine = self._select_engine(self._cfg.get("engine")
                                     or self._cfg.get("tts_engine"))
        self._engine = engine
        # model_dir is per engine: sherpa-onnx wants its Matcha/vocoder pair,
        # VITS2 wants its TensorRT release. config.yaml carries one model_dir,
        # written for the engine it also declares — handing that same path to the
        # other engine made sherpa download its models into /models/vits2 and
        # then try to load them from there. So the configured path applies only
        # to the configured engine; every other engine gets its own default.
        self._configured_engine = engine
        self._configured_model_dir = self._cfg.get("model_dir") or ""
        # Built inline at startup so info/start are immediately truthful, and so
        # a misconfigured engine shows up in the boot log rather than on the
        # first utterance. Runtime switches take the background path below.
        try:
            self._impl = self._build(engine)
            self._impl_engine = engine
        except Exception as error:  # noqa: BLE001 - surfaced via info/start
            log.error("[tts] failed to build engine %r: %s", engine, error, exc_info=True)
            self._build_error = str(error)

    # ── engine plumbing ─────────────────────────────────────────────────

    @staticmethod
    def _select_engine(value) -> str:
        engine = str(value or DEFAULT_TTS_ENGINE).strip().lower()
        # Underscores fold to hyphens before the alias lookup, so both the stored
        # `vits2_trt` and a hand-typed `vits2_zh_en` land on the same key and the
        # alias table only has to spell each old name once.
        engine = engine.replace("_", "-")
        # Resolve before validating, so a stored `sherpa_onnx` keeps working and
        # everything downstream — _build, ENGINE_MODEL_DIRS, the `engine` field in
        # info — sees only the current name.
        engine = ENGINE_ALIASES.get(engine, engine)
        if engine not in TTS_ENGINES:
            raise ValueError(f"Unsupported TTS engine: {engine}")
        return engine

    def _model_dir_for(self, engine: str) -> str:
        if engine == self._configured_engine and self._configured_model_dir:
            return self._configured_model_dir
        return ENGINE_MODEL_DIRS[engine]

    def _build(self, engine: str):
        cfg = dict(self._cfg)
        cfg["engine"] = engine
        cfg["model_dir"] = self._model_dir_for(engine)
        impl = (self._build_vits2(cfg) if engine == "vits2-zh-en"
                else SherpaOnnxTTSPlugin(cfg, self._executor))
        # An implementation may swallow its own model-load failure and come back
        # as an object that reports error through info (sherpa does exactly
        # that). Installing it would make the facade claim ready and let a start
        # or a speak "succeed" against a model that never loaded, so ask it.
        state = impl.dispatch("tts", {"action": "info"}) or {}
        if state.get("state") == "error":
            raise RuntimeError(
                state.get("error") or state.get("desc")
                or f"engine {engine} reported an error after construction"
            )
        return impl

    def _build_vits2(self, cfg: dict):
        from plugins.vits2_tts import Vits2TTSPlugin

        return Vits2TTSPlugin(cfg, self._executor)

    def _build_async(self, engine: str) -> None:
        """Build an engine off the request thread; the old one is already gone."""
        def _run():
            try:
                impl = self._build(engine)
            except Exception as error:  # noqa: BLE001
                log.error("[tts] failed to build engine %r: %s", engine, error,
                          exc_info=True)
                with self._lock:
                    if self._building == engine:
                        self._building = ""
                        self._build_error = str(error)
                return
            stale = None
            replay = []
            with self._lock:
                if self._building != engine:
                    stale = impl          # another switch superseded this one
                else:
                    self._impl = impl
                    self._impl_engine = engine
                    self._building = ""
                    self._build_error = None
                    replay = list(self._pending_starts.values())
                    self._pending_starts.clear()
            if stale is not None:
                _dispose_impl(stale)
                return

            # Honour the starts that arrived mid-build. Without this the engine
            # comes up idle, and Agent Core — which polls `info` after a start
            # answered `loading` — reports the card as "启动已取消".
            for start_args in replay:
                instance = (start_args.get("instance_id")
                            or start_args.get("input_topic") or "?")
                try:
                    log.info("[tts] replaying start deferred during the %s build: %s",
                             engine, instance)
                    result = impl.dispatch("tts", start_args) or {}
                except Exception:
                    log.error("[tts] deferred start failed after the %s build",
                              engine, exc_info=True)
                    continue
                # dispatch REPORTS failure, it does not raise it — start() returns
                # {"state": "error", ...} for a failed dry-run — so the except above
                # never fired and a replayed start that failed was completely
                # silent. That is what made a broken Thai start look like a
                # successful one: the only trace was the frontend's own warning,
                # and the card kept whatever state Agent Core inferred from
                # polling. Inspect the result.
                if result.get("state") == "error":
                    log.error("[tts] deferred start for %s failed after the %s "
                              "build: %s", instance, engine,
                              result.get("message") or result.get("desc") or result)

        threading.Thread(target=_run, name=f"tts-engine-{engine}", daemon=True).start()

    def get_tools(self) -> list:
        return TOOLS

    # ── dispatch ────────────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        if action == "config":
            return self._config(name, args)

        with self._lock:
            impl = self._impl
            building = self._building
            error = self._build_error
            engine = self._engine
        if impl is not None:
            result = impl.dispatch(name, args)
            if isinstance(result, dict) and action == "info":
                # Assignment, not setdefault: the facade is the only thing that
                # knows which engine is live, and an implementation that reports
                # its own name overrode it. vits2_tts_trt hardcoded
                # `"engine": "vits2_trt"`, so after the rename the card displayed
                # a value that is not even in the configSchema enum — while the
                # sherpa engines, which report no engine of their own, showed the
                # right one. That asymmetry is what made it look like the default
                # had not been renamed.
                result["engine"] = engine
            return result

        # No engine resident: only happens while a switch is building, or after
        # a build failure. Never block the caller waiting for it.
        if action == "info":
            state = "loading" if building else "error"
            result = {
                "name": "TTS", "manufacture": "Embodied", "model": engine,
                "engine": engine, "state": state,
                "desc": (f"Switching to the {building} engine..." if building
                         else f"Engine {engine} failed to load: {error}"),
            }
            if state == "error" and error:
                result["error"] = error
            return result
        if building:
            if action == "start":
                # Defer, do not drop. The dashboard sends config (which triggers
                # the switch) and start back to back, so this is the normal path,
                # not a rare race — and _build_async replays it.
                with self._lock:
                    if self._building:
                        key = args.get("instance_id") or args.get("input_topic") or ""
                        self._pending_starts[key] = dict(args)
                        deferred = True
                    else:
                        deferred = False   # build finished while we waited on the lock
                if not deferred:
                    return self.dispatch(name, args)
            elif action == "stop":
                # Cancel a deferred start rather than letting it revive the node
                # after the operator asked for it to stop.
                with self._lock:
                    key = args.get("instance_id") or args.get("input_topic") or ""
                    if key in self._pending_starts:
                        self._pending_starts.pop(key, None)
                    elif not key:
                        self._pending_starts.clear()
                return {"state": "idle"}
            return {"state": "loading",
                    "message": f"TTS engine {building} is initializing, retry shortly"}
        return {"state": "error", "message": f"TTS engine {engine} failed: {error}"}

    def _await_build(self, engine: str, timeout_s: float) -> tuple[bool, str | None]:
        """Wait for an in-flight build. Returns (ready, error).

        `ready` false with no error means it is still going, which is the caller's
        cue to answer `loading` and let the deferred-start machinery finish the job.
        Another switch superseding this one also reads as not-ready: this build's
        result is about to be discarded, so claiming success would be wrong.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._engine != engine:
                    return False, None          # superseded by a newer switch
                if self._impl is not None and self._impl_engine == engine:
                    return True, None
                if self._build_error:
                    return False, self._build_error
                if not self._building:
                    # Neither resident nor building nor failed: nothing to wait on.
                    return False, None
            if time.monotonic() >= deadline:
                return False, None
            time.sleep(0.2)

    def _config(self, name: str, args: dict) -> dict:
        """Apply shared config, switching engines when tts_engine changes."""
        requested = args.get("tts_engine") or args.get("engine")
        forwarded = {k: v for k, v in args.items()
                     if k not in ("tts_engine", "engine")}
        for key in SHARED_CONFIG_KEYS:
            if key in args:
                self._cfg[key] = args[key]

        if requested:
            engine = self._select_engine(requested)
            with self._lock:
                switching = engine != self._impl_engine or self._impl is None
                if switching:
                    outgoing, self._impl = self._impl, None
                    self._impl_engine = ""
                    self._engine = engine
                    self._cfg["engine"] = engine
                    self._building = engine
                    self._build_error = None
            if switching:
                # Stop the old engine's nodes before the new one publishes on
                # the same topics — two live TTS publishers on one topic is the
                # duplicate-audio failure mode in README § Plugin Concurrency.
                if outgoing is not None:
                    _dispose_impl(outgoing)
                self._build_async(engine)
                log.info("[tts] switching engine to %s", engine)
                # Wait for it, up to a bound. Only part of a build is open-ended
                # (downloading a model); constructing the session afterwards took
                # ~2 s on cpu and ~5 s on gpu, and answering `loading` for that is
                # what made the dashboard send a start the engine could not honour.
                # A time bound rather than "does it need to download" because the
                # download is not the only slow phase — the gpu paraformer encoder
                # is 636 MB and reading it cold takes seconds on its own.
                ready, error = self._await_build(engine, ENGINE_SWITCH_WAIT_S)
                if error:
                    return {"status": "error", "engine": engine,
                            "state": "error", "message": error}
                if ready:
                    log.info("[tts] engine %s ready", engine)
                    return {"status": "configured", "engine": engine}
                return {"status": "configured", "state": "loading",
                        "engine": engine,
                        "message": f"loading the {engine} engine"}

        with self._lock:
            impl = self._impl
            engine = self._engine
        if impl is None:
            return {"status": "configured", "state": "loading", "engine": engine}
        result = impl.dispatch(name, {**forwarded, "action": "config"})
        if isinstance(result, dict):
            result.setdefault("engine", engine)
        return result

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        with self._lock:
            impl = self._impl
            engine = self._engine
            error = self._build_error
        if impl is None:
            raise RuntimeError(f"TTS engine {engine} not ready: {error or 'loading'}")
        return impl.synthesize_raw(text)


def _dispose_impl(impl) -> None:
    """Stop every node an engine implementation owns before dropping it."""
    try:
        impl.dispatch("tts", {"action": "stop"})
    except Exception:
        log.error("[tts] failed to stop the outgoing engine", exc_info=True)
    # Kokoro's Japanese path may own a worker process. Dropping the impl does not
    # reap it — a child is not garbage — and on `device: gpu` it is holding a CUDA
    # context the incoming engine is about to want.
    adapter = getattr(impl, "_adapter", None)
    closer = getattr(adapter, "close", None)
    if closer is not None:
        try:
            closer()
        except Exception:
            log.error("[tts] failed to close the outgoing adapter", exc_info=True)
