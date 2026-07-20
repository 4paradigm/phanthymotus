#!/usr/bin/env python3
"""
plugins/tts.py — TTSPlugin: sherpa-onnx VITS TTS.

On-device text-to-speech using sherpa-onnx MeloTTS (Chinese + English).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200  # 100ms @ 16kHz 16-bit mono
MAX_SEGMENT_CHARS = 60
# Local synthesis buffer. 600 frames is about 60 seconds / 1.9 MB of PCM.
# It lets the producer synthesize the next sentence while the current one plays.
SYNTH_QUEUE_FRAMES = 600


def _maybe_set_cpu_affinity() -> None:
    """Optional CPU pinning for Jetson benchmarks (TTS_CPU_AFFINITY=0,1,2,3)."""
    import os

    if not hasattr(os, "sched_setaffinity"):
        return
    spec = os.environ.get("TTS_CPU_AFFINITY", "").strip()
    if not spec:
        return
    cores = {int(x.strip()) for x in spec.split(",") if x.strip()}
    if cores:
        os.sched_setaffinity(0, cores)
        log.info(f"[tts] CPU affinity set to {sorted(cores)}")


def _process_rss_mb() -> float:
    """Current process RSS in MB (for Jetson memory benchmarking)."""
    import os

    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


_maybe_set_cpu_affinity()

_STRONG_SENTENCE_END = frozenset("。！？!?；;")
_WEAK_SENTENCE_END = frozenset("，,、：:")
_CLOSING_PUNCTUATION = frozenset("”’\"'》〉】〕）)]}」』")

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
)

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
                    "enum": ["start", "stop", "speak", "info", "config"],
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
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "speaker_id": {"type": "integer", "description": "Speaker ID", "default": 0, "scope": "shared"},
                "speed":      {"type": "number", "description": "Speech speed (1.0 = normal)", "default": 1.0, "scope": "shared"},
            },
            "required": []
        },
        "topic_in":  [{"format": "data/json",     "desc": "text to synthesize"}],
        "topic_out": [{"format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
    }
]


# ── TTS Adapter ──────────────────────────────────────────────────────────────


def _is_cjk(char: str) -> bool:
    """Return True for common CJK code-point ranges."""
    if not char:
        return False
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
    """Split an unusually long sentence at weak punctuation or whitespace."""
    if max_chars <= 0 or len(segment) <= max_chars:
        return [segment]

    parts: list[str] = []
    remaining = segment
    min_cut = max(1, max_chars // 2)

    while len(remaining) > max_chars:
        cut = -1

        # Prefer a comma/colon-like boundary near the maximum length.
        for index in range(max_chars - 1, min_cut - 1, -1):
            if remaining[index] in _WEAK_SENTENCE_END:
                cut = index + 1
                break

        # For English text, prefer a whitespace boundary rather than
        # splitting through the middle of a word.
        if cut < 0:
            space_index = remaining.rfind(" ", min_cut, max_chars + 1)
            if space_index >= 0:
                cut = space_index + 1

        if cut < 0:
            cut = max_chars

        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def _split_text_for_tts(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into TTS-friendly sentences while retaining punctuation.

    Primary boundaries are Chinese/English sentence-ending punctuation and
    newlines. English full stops are kept inside decimal numbers. A very long
    sentence is split again at comma/colon-like punctuation or whitespace.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    segments: list[str] = []
    current: list[str] = []
    text_len = len(normalized)
    index = 0

    while index < text_len:
        char = normalized[index]
        current.append(char)
        is_boundary = char == "\n" or char in _STRONG_SENTENCE_END

        if char == ".":
            previous = normalized[index - 1] if index > 0 else ""
            following = normalized[index + 1] if index + 1 < text_len else ""
            is_decimal = previous.isdigit() and following.isdigit()
            # Avoid splitting 3.14, but support both "Hello. Next" and
            # mixed text such as "Hello.下一句".
            is_boundary = not is_decimal and (
                not following
                or following.isspace()
                or following in _CLOSING_PUNCTUATION
                or _is_cjk(following)
            )

        if is_boundary:
            # Keep closing quotes/brackets with the sentence-ending mark.
            next_index = index + 1
            while (
                next_index < text_len
                and normalized[next_index] in _CLOSING_PUNCTUATION
            ):
                current.append(normalized[next_index])
                next_index += 1
            index = next_index - 1

            sentence = "".join(current).strip()
            if sentence:
                segments.extend(_split_long_segment(sentence, max_chars))
            current = []

        index += 1

    tail = "".join(current).strip()
    if tail:
        segments.extend(_split_long_segment(tail, max_chars))

    return segments

class TTSAdapter(ABC):
    @abstractmethod
    def _synthesize_segment(self, text: str) -> bytes: ...

    def split_text(self, text: str) -> list[str]:
        max_chars = getattr(self, "max_segment_chars", MAX_SEGMENT_CHARS)
        return _split_text_for_tts(text, max_chars)

    def synthesize(self, text: str) -> bytes:
        """Synthesize all segments and return one concatenated PCM stream."""
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        """Yield concatenated PCM chunks, synthesized one sentence at a time."""
        yield from self.synthesize_segments_stream(self.split_text(text))

    def synthesize_segments_stream(self, segments: list[str]):
        """Synthesize pre-split segments and yield one continuous PCM stream."""
        buffer = b""
        for segment in segments:
            buffer += self._synthesize_segment(segment)
            while len(buffer) >= CHUNK_BYTES:
                yield buffer[:CHUNK_BYTES]
                buffer = buffer[CHUNK_BYTES:]
        if buffer:
            yield buffer


def _resample_to_16k(samples, src_rate: int):
    """Resample float PCM to 16 kHz for audio/pcm-16k output."""
    if src_rate == SAMPLE_RATE:
        return samples
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    g = gcd(src_rate, SAMPLE_RATE)
    return resample_poly(np.asarray(samples, dtype=np.float32), SAMPLE_RATE // g, src_rate // g)


def _float_samples_to_pcm16(samples) -> bytes:
    import struct

    return struct.pack(
        f"<{len(samples)}h",
        *[int(max(-32768, min(32767, s * 32767))) for s in samples],
    )


def _resolve_vits_model_path(model_dir: str) -> str:
    """Prefer model.int8.onnx when present (44.1k quantized package)."""
    import os

    for name in ("model.int8.onnx", "model.onnx"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(model_dir, "model.onnx")


class SherpaOnnxVitsTTSAdapter(TTSAdapter):
    """On-device TTS using sherpa-onnx VITS (e.g. vits-melo-tts-zh_en-8k)."""

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0,
                 model_name: str = "tts_melo_8k", hw_provider: str = "cpu",
                 num_threads: int = 4):
        import os
        from utils.model_downloader import ensure_model

        ensure_model(model_name, model_dir)

        import sherpa_onnx

        mem_before = _process_rss_mb()
        model_path = _resolve_vits_model_path(model_dir)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0
        tokens_path = os.path.join(model_dir, "tokens.txt")
        espeak_data_dir = os.path.join(model_dir, "espeak-ng-data")
        lexicon_path = os.path.join(model_dir, "lexicon.txt")
        dict_dir = os.path.join(model_dir, "dict")

        use_espeak = os.path.isdir(espeak_data_dir)

        rule_fsts = []
        for name in ("date.fst", "number.fst", "phone.fst"):
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                rule_fsts.append(p)

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model_path,
                    tokens=tokens_path,
                    lexicon="" if use_espeak else (lexicon_path if os.path.exists(lexicon_path) else ""),
                    dict_dir="" if use_espeak else (dict_dir if os.path.isdir(dict_dir) else ""),
                    data_dir=espeak_data_dir if use_espeak else "",
                    length_scale=1.0 / speed if speed else 1.0,
                ),
                num_threads=num_threads,
                provider=hw_provider,
            ),
            rule_fsts=",".join(rule_fsts) if rule_fsts else "",
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._sid = speaker_id
        self._speed = speed
        self._model_sr = self._tts.sample_rate
        self.max_segment_chars = MAX_SEGMENT_CHARS
        mode = "espeak" if use_espeak else "lexicon"
        mem_after = _process_rss_mb()
        log.info(
            f"[tts] sherpa-onnx VITS loaded: model_dir={model_dir}, mode={mode}, "
            f"model={os.path.basename(model_path)}, "
            f"sample_rate={self._model_sr}, model_size_mb={model_size_mb:.1f}, "
            f"speaker_id={speaker_id}, speed={speed}, "
            f"provider={hw_provider}, num_threads={num_threads}, "
            f"memory_mb={mem_before:.1f}->{mem_after:.1f}"
        )

    def _synthesize_segment(self, text: str) -> bytes:
        audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        samples = _resample_to_16k(audio.samples, self._model_sr)
        return _float_samples_to_pcm16(samples)


class SherpaOnnxTTSAdapter(TTSAdapter):
    """On-device TTS using sherpa-onnx Matcha (flow-matching, fast non-autoregressive)."""

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0):
        import os
        from utils.model_downloader import ensure_model
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

        # Gather rule FSTs
        rule_fsts = []
        for name in ("date-zh.fst", "number-zh.fst", "phone-zh.fst"):
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                rule_fsts.append(p)

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
                provider="cpu",
            ),
            rule_fsts=",".join(rule_fsts) if rule_fsts else "",
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._sid = speaker_id
        self._speed = speed
        log.info(f"[tts] sherpa-onnx Matcha loaded: model_dir={model_dir}, "
                 f"speaker_id={speaker_id}, speed={speed}")

    def _synthesize_segment(self, text: str) -> bytes:
        audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        samples = _resample_to_16k(audio.samples, self._tts.sample_rate)
        return _float_samples_to_pcm16(samples)




def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    model_dir = cfg.get("model_dir", "/models/sherpa-onnx/tts")
    speaker_id = int(cfg.get("speaker_id", 0))
    speed = float(cfg.get("speed", 1.0))
    backend = cfg.get("backend", "vits")
    if backend == "vits":
        adapter = SherpaOnnxVitsTTSAdapter(
            model_dir,
            speaker_id,
            speed,
            model_name=cfg.get("model_name", "tts_melo_8k"),
            hw_provider=cfg.get("hw_provider", "cpu"),
            num_threads=int(cfg.get("num_threads", 2)),
        )
    else:
        adapter = SherpaOnnxTTSAdapter(model_dir, speaker_id, speed)
    adapter.max_segment_chars = int(cfg.get("max_segment_chars", MAX_SEGMENT_CHARS))
    return adapter


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class _TTSNode(Node):
    def __init__(
        self,
        input_topic: Optional[str],
        adapter: Optional[TTSAdapter],
        node_suffix: str = '',
        realtime_pacing: bool = False,
    ):
        node_name = f"tts_{node_suffix}" if node_suffix else "tts"
        super().__init__(node_name)
        self._input_topic  = input_topic or ''
        self._output_topic = f"{input_topic}/tts" if input_topic else '/perception/tts'
        self._adapter      = adapter
        self._realtime_pacing = realtime_pacing
        self.state         = "idle"
        self._text_queue   = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        from audio_msgs.msg import AudioChunk
        self._pub = self.create_publisher(AudioChunk, self._output_topic, _LOW_LAT_QOS)
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
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self._status_dict()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def enqueue(self, text: str):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        self._text_queue.put(text)

    def _publish_frame(self, frame: bytes, frames_sent: int, t0: Optional[float], frame_duration: float):
        from audio_msgs.msg import AudioChunk
        import time as _time

        if self._realtime_pacing and t0 is not None:
            target = t0 + frames_sent * frame_duration
            now = _time.monotonic()
            if now < target:
                _time.sleep(target - now)
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "audio/pcm-16k"
        msg.data = list(frame)
        self._pub.publish(msg)
        return frames_sent + 1

    def _text_cb(self, msg: String):
        if self.state != "running": return
        try:
            text = json.loads(msg.data).get("text","")
        except Exception:
            text = msg.data.strip()
        if text:
            log.info(f"[tts] received text from topic: {text[:50]}...")
            self._text_queue.put(text)

    def _worker(self):
        from audio_msgs.msg import AudioChunk
        import time as _time

        # Real-time pacing: publish frames at playback rate to avoid bursts/gaps
        FRAME_DURATION = CHUNK_BYTES / (SAMPLE_RATE * 2)  # 0.1s per 3200-byte frame
        PREBUF_FRAMES  = 1  # 1 frame (~100ms); was 3 (~300ms) for lower judged TTFT

        while not self._stop_event.is_set():
            try:
                text = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                import time as _time
                t_start = _time.monotonic()
                total = 0
                buf   = b''
                t0    = None  # wall-clock start of playback
                frames_sent = 0
                prebuf = []   # pre-buffer queue
                first_audio_latency = None
                segments = self._adapter.split_text(text)
                if not segments:
                    continue

                log.info(
                    f"[tts] split {len(text)} chars into {len(segments)} segment(s): "
                    f"{[len(segment) for segment in segments]}"
                )

                # Decouple offline sentence synthesis from real-time publishing.
                # The producer can generate the next sentence while audio from
                # the current sentence is being paced to the ROS2 topic.
                audio_queue = queue.Queue(maxsize=SYNTH_QUEUE_FRAMES)
                stream_end = object()
                producer_error = []
                synth_elapsed = [0.0]

                def _queue_put(item) -> bool:
                    while not self._stop_event.is_set():
                        try:
                            audio_queue.put(item, timeout=0.1)
                            return True
                        except queue.Full:
                            continue
                    return False

                def _produce_audio():
                    synth_t0 = _time.monotonic()
                    try:
                        for chunk in self._adapter.synthesize_segments_stream(segments):
                            if self._stop_event.is_set() or not _queue_put(chunk):
                                break
                    except Exception as exc:
                        producer_error.append(exc)
                    finally:
                        synth_elapsed[0] = _time.monotonic() - synth_t0
                        _queue_put(stream_end)

                producer_thread = threading.Thread(target=_produce_audio, daemon=True)
                producer_thread.start()

                while not self._stop_event.is_set():
                    try:
                        raw_chunk = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        if not producer_thread.is_alive() and audio_queue.empty():
                            break
                        continue

                    if raw_chunk is stream_end:
                        break
                    if first_audio_latency is None:
                        first_audio_latency = _time.monotonic() - t_start

                    if self._stop_event.is_set():
                        break
                    buf  += raw_chunk
                    total += len(raw_chunk)
                    # split into CHUNK_BYTES frames
                    while len(buf) >= CHUNK_BYTES:
                        frame = buf[:CHUNK_BYTES]
                        buf   = buf[CHUNK_BYTES:]

                        # Pre-buffer phase: accumulate a few frames before pacing
                        if t0 is None:
                            prebuf.append(frame)
                            if len(prebuf) >= PREBUF_FRAMES:
                                t0 = _time.monotonic() if self._realtime_pacing else 0.0
                                for pf in prebuf:
                                    frames_sent = self._publish_frame(
                                        pf, frames_sent, t0, FRAME_DURATION
                                    )
                                prebuf = []
                            continue

                        frames_sent = self._publish_frame(
                            frame, frames_sent, t0, FRAME_DURATION
                        )

                # Flush any remaining pre-buffer (short utterances < PREBUF_FRAMES)
                if prebuf and not self._stop_event.is_set():
                    if t0 is None:
                        t0 = _time.monotonic() if self._realtime_pacing else 0.0
                    for pf in prebuf:
                        frames_sent = self._publish_frame(
                            pf, frames_sent, t0, FRAME_DURATION
                        )

                # flush remainder
                if buf and not self._stop_event.is_set():
                    if self._realtime_pacing and t0 is not None:
                        target = t0 + frames_sent * FRAME_DURATION
                        now = _time.monotonic()
                        if now < target:
                            _time.sleep(target - now)
                    from audio_msgs.msg import AudioChunk
                    msg = AudioChunk()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.format = "audio/pcm-16k"
                    msg.data = list(buf)
                    self._pub.publish(msg)
                    frames_sent += 1

                producer_thread.join(timeout=0.2)
                if producer_error:
                    raise producer_error[0]

                elapsed = _time.monotonic() - t_start
                audio_duration = total / (SAMPLE_RATE * 2) if total else 0.0
                synth_rtf = (synth_elapsed[0] / audio_duration) if audio_duration > 0 else 0.0
                e2e_rtf = (elapsed / audio_duration) if audio_duration > 0 else 0.0
                mem_mb = _process_rss_mb()
                first_audio_text = (
                    f", TTFT={first_audio_latency:.2f}s"
                    if first_audio_latency is not None
                    else ""
                )
                log.info(
                    f"[tts] spoke {len(text)} chars in {len(segments)} segment(s) "
                    f"→ {total} bytes ({frames_sent} frames) in {elapsed:.2f}s"
                    f"{first_audio_text}, "
                    f"audio={audio_duration:.2f}s, synth_RTF={synth_rtf:.2f}, "
                    f"e2e_RTF={e2e_rtf:.2f}, memory_mb={mem_mb:.1f}"
                )
            except Exception as e:
                log.error(f"[tts] synthesis error: {e}", exc_info=True)

    def _status_dict(self) -> dict:
        return {
            "state":     self.state,
            "topic_in":  [{"topic": self._input_topic,  "format": "data/json",     "desc": "text to synthesize"}],
            "topic_out": [{"topic": self._output_topic, "format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
        }


def _warmup_tts_adapter(adapter: TTSAdapter, text: str = "。") -> None:
    """Run one silent synthesis to warm up ORT/CUDA before the first speak request."""
    import time as _time

    log.info(f"[tts] warmup starting: text={text!r}")
    t0 = _time.monotonic()
    pcm = adapter._synthesize_segment(text)
    elapsed = _time.monotonic() - t0
    log.info(f"[tts] warmup done in {elapsed:.2f}s ({len(pcm)} bytes)")


def _run_tts_warmup(adapter: TTSAdapter, plugin_cfg: dict) -> None:
    """Warm up ORT/CUDA before the first speak request."""
    if not plugin_cfg.get("warmup", True):
        return
    text = plugin_cfg.get(
        "warmup_text",
        "你好，欢迎使用语音合成服务，这是一段预热测试文本。",
    )
    try:
        _warmup_tts_adapter(adapter, text)
    except Exception as e:
        log.warning(f"[tts] warmup failed (non-fatal): {e}", exc_info=True)


def _start_warmup_background(adapter: TTSAdapter, plugin_cfg: dict) -> None:
    """Optional async warmup (warmup_async=true)."""
    def _run() -> None:
        _run_tts_warmup(adapter, plugin_cfg)

    threading.Thread(target=_run, daemon=True, name="tts-warmup").start()


# ── Plugin ────────────────────────────────────────────────────────────────────

class TTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg      = plugin_cfg
        self._loading  = False
        self._load_error = None
        self._realtime_pacing = bool(plugin_cfg.get("realtime_pacing", False))
        try:
            self._adapter  = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] failed to load model: {e}", exc_info=True)
            self._adapter = None
            self._load_error = str(e)
        if self._adapter:
            if plugin_cfg.get("warmup_async", False):
                _start_warmup_background(self._adapter, plugin_cfg)
            else:
                _run_tts_warmup(self._adapter, plugin_cfg)
        self._nodes: dict[str, _TTSNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._executor = executor
        log.info(
            f"[tts] plugin init: sherpa-onnx VITS, "
            f"speaker_id={plugin_cfg.get('speaker_id', 0)}, "
            f"speed={plugin_cfg.get('speed', 1.0)}, "
            f"max_segment_chars={plugin_cfg.get('max_segment_chars', MAX_SEGMENT_CHARS)}, "
            f"realtime_pacing={self._realtime_pacing}"
        )

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
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
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
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "audio/pcm-16k", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
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
            # Clean up _default node if it would conflict with this instance
            if '_default' in self._nodes and node_key != '_default':
                default_node = self._nodes['_default']
                if default_node._input_topic == input_topic or default_node._output_topic == (f"{input_topic}/tts" if input_topic else '/perception/tts'):
                    default_node.stop()
                    self._executor.remove_node(default_node)
                    del self._nodes['_default']
            if node_key not in self._nodes:
                node = _TTSNode(
                    input_topic or None,
                    self._adapter,
                    node_suffix=node_key.replace('/', '_').replace('-', '_'),
                    realtime_pacing=self._realtime_pacing,
                )
                self._executor.add_node(node)
                self._nodes[node_key] = node
            elif input_topic and self._nodes[node_key]._input_topic != input_topic:
                # Input topic changed for existing instance — recreate
                old_node = self._nodes[node_key]
                old_node.stop()
                self._executor.remove_node(old_node)
                node = _TTSNode(
                    input_topic,
                    self._adapter,
                    node_suffix=node_key.replace('/', '_').replace('-', '_'),
                    realtime_pacing=self._realtime_pacing,
                )
                self._executor.add_node(node)
                self._nodes[node_key] = node
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                for key in list(self._nodes.keys()):
                    self._nodes[key].stop()
                    self._executor.remove_node(self._nodes[key])
                    del self._nodes[key]
                return {"state": "idle"}
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
            node = None
            for n in self._nodes.values():
                if n.state == "running":
                    node = n
                    break
            if node is None:
                # No running node — use instance key or fallback
                node_key = instance_id or '_default'
                if node_key not in self._nodes:
                    input_topic = args.get("input_topic") or None
                    adapter = self._adapter
                    if instance_id and instance_id in self._instance_configs:
                        inst_adapter = _build_tts_adapter(self._instance_configs[instance_id])
                        if inst_adapter:
                            adapter = inst_adapter
                    node = _TTSNode(
                        input_topic,
                        adapter,
                        node_suffix=node_key.replace('/', '_').replace('-', '_'),
                        realtime_pacing=self._realtime_pacing,
                    )
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                else:
                    node = self._nodes[node_key]
                if node.state != "running":
                    node.start()
            node.enqueue(text)
            return {"status": "queued", "text": text}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v}
            # Update config and rebuild adapter
            if 'speaker_id' in cfg:
                self._cfg['speaker_id'] = int(cfg['speaker_id'])
            if 'speed' in cfg:
                self._cfg['speed'] = float(cfg['speed'])
            self._adapter = _build_tts_adapter(self._cfg)
            # Stop all nodes (they'll use new adapter on next start)
            for key in list(self._nodes.keys()):
                self._nodes[key].stop()
                self._executor.remove_node(self._nodes[key])
                del self._nodes[key]
            return {"status": "configured"}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        return self._adapter.synthesize(text)