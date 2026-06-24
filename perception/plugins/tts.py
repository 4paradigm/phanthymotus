#!/usr/bin/env python3
"""
plugins/tts.py — TTSPlugin: TTS 感知封装。

从 perception/tts/main.py 提取，作为 PerceptionBundle 的插件。
核心逻辑（TTSAdapter、TTSNode）不变，去掉独立 main/MCP server。
"""

from __future__ import annotations

import base64
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
                "api_key": {"type": "string", "description": "API Key", "format": "password", "scope": "shared"},
                "url":     {"type": "string", "description": "自定义 URL (可选)", "scope": "shared"},
                "model":   {"type": "string", "description": "模型名称", "scope": "instance"},
                "voice":   {"type": "string", "description": "音色名称", "scope": "instance"},
            },
            "required": ["api_key"]
        },
        "topic_in":  [{"format": "data/json",     "desc": "text to synthesize"}],
        "topic_out": [{"format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
    }
]


# ── TTS Adapters ──────────────────────────────────────────────────────────────

_DASHSCOPE_WS_URL      = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'
_DASHSCOPE_REALTIME_URL = 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime'

class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    def synthesize_stream(self, text: str):
        """Yield raw PCM bytes as they arrive. Default: collect all."""
        yield self.synthesize(text)


class AliyunDashScopeTTSAdapter(TTSAdapter):
    """Aliyun DashScope TTS via dashscope SDK.

    - cosyvoice-* models  → tts_v2.SpeechSynthesizer (WS streaming)
    - qwen*-tts-* models  → qwen_tts_realtime.QwenTtsRealtime (realtime WS)
    """

    def __init__(self, api_key: str, model: str, voice: str, url: str = ''):
        self.api_key = api_key
        self.model   = model or 'qwen3-tts-flash-realtime'
        self.voice   = voice or 'Ethan'
        self.url     = url

    def _is_qwen(self) -> bool:
        return 'qwen' in self.model.lower()

    def synthesize(self, text: str) -> bytes:
        return b''.join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        if self._is_qwen():
            yield from self._stream_qwen(text)
        else:
            yield from self._stream_cosyvoice(text)

    def _stream_cosyvoice(self, text: str):
        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
        import queue as _queue

        pcm_q = _queue.Queue()
        _DONE = object()
        error = [None]

        class _Cb(ResultCallback):
            def on_data(self_cb, data: bytes) -> None:
                pcm_q.put(data)
            def on_complete(self_cb) -> None:
                pcm_q.put(_DONE)
            def on_error(self_cb, message) -> None:
                error[0] = message
                pcm_q.put(_DONE)
            def on_close(self_cb) -> None:
                pass

        dashscope.api_key = self.api_key
        if self.url:
            dashscope.base_websocket_api_url = self.url

        synth = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            callback=_Cb(),
        )
        synth.call(text)  # async_call=True with callback, returns immediately

        while True:
            item = pcm_q.get(timeout=120)
            if item is _DONE:
                break
            yield item

        if error[0]:
            raise RuntimeError(f'DashScope CosyVoice TTS error: {error[0]}')

    def _stream_qwen(self, text: str):
        import base64 as _b64, dashscope
        from dashscope.audio.qwen_tts_realtime import (
            QwenTtsRealtime, QwenTtsRealtimeCallback,
        )
        import queue as _queue
        pcm_q = _queue.Queue()
        _DONE  = object()
        error  = [None]

        # use realtime endpoint; ignore any inference-URL override (CosyVoice URL)
        if self.url and 'realtime' in self.url:
            url = self.url
        else:
            url = _DASHSCOPE_REALTIME_URL
            if self.url:
                log.warning(f'[tts/qwen] ignoring non-realtime url={self.url!r}, using default')

        class _Cb(QwenTtsRealtimeCallback):
            def on_open(self_cb): pass
            def on_close(self_cb, code, msg):
                pcm_q.put(_DONE)
            def on_event(self_cb, response):
                t = response.get('type', '') if isinstance(response, dict) else ''
                if t == 'response.audio.delta':
                    raw = response.get('delta', '')
                    data = _b64.b64decode(raw) if raw else b''
                    if data:
                        pcm_q.put(data)
                elif t == 'session.finished':
                    pcm_q.put(_DONE)
                elif t == 'error':
                    error[0] = repr(response)
                    pcm_q.put(_DONE)

        dashscope.api_key = self.api_key
        cb  = _Cb()
        cli = QwenTtsRealtime(model=self.model, callback=cb, url=url)
        cli.connect()
        cli.update_session(voice=self.voice, mode='server_commit')
        cli.append_text(text)
        cli.finish()

        total = 0
        while True:
            item = pcm_q.get(timeout=60)
            if item is _DONE:
                break
            total += len(item)
            # 24kHz s16le → 16kHz s16le: 每3个样本取2个（简单线性插值）
            yield _resample_24k_to_16k(item)

        log.info(f'[tts] qwen spoke {len(text)} chars → {total} bytes (streamed)')

        if error[0]:
            raise RuntimeError(f'DashScope Qwen TTS error: {error[0]}')


def _resample_24k_to_16k(data: bytes) -> bytes:
    """Resample PCM s16le from 24kHz to 16kHz using linear interpolation.

    Ratio: 16000/24000 = 2/3, so for every 3 input samples we produce 2 output samples.
    Uses array module for ~10x speedup over struct.unpack loop.
    """
    import array
    n_samples = len(data) // 2
    if n_samples == 0:
        return b''
    samples = array.array('h')
    samples.frombytes(data)
    n_out = int(n_samples * 2 / 3)
    out = array.array('h', bytes(n_out * 2))
    for i in range(n_out):
        pos = i * 1.5
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < n_samples:
            s = samples[idx] + (samples[idx + 1] - samples[idx]) * frac
        else:
            s = samples[idx] if idx < n_samples else 0
        out[i] = int(max(-32768, min(32767, s)))
    return out.tobytes()


def _decode_to_pcm16k(encoded: bytes, input_fmt: str = None, input_rate: int = None) -> bytes:
    """Convert audio to PCM 16 kHz mono s16le via ffmpeg.

    If input_fmt/input_rate are given, treat input as raw PCM with those params.
    Otherwise auto-detect (mp3/wav/opus/…).
    """
    import subprocess
    if input_fmt and input_rate:
        cmd = ['ffmpeg', '-y',
               '-f', input_fmt, '-ar', str(input_rate), '-ac', '1', '-i', 'pipe:0',
               '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1']
    else:
        cmd = ['ffmpeg', '-y', '-i', 'pipe:0',
               '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1']
    result = subprocess.run(cmd, input=encoded, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg decode failed: {result.stderr.decode(errors="ignore")[-300:]}')
    return result.stdout


def _build_tts_adapter(cfg: dict) -> Optional[TTSAdapter]:
    api_key = cfg.get('api_key', '')
    if not api_key:
        return None
    return AliyunDashScopeTTSAdapter(
        api_key=api_key,
        model=cfg.get('model', ''),
        voice=cfg.get('voice', ''),
        url=cfg.get('url', ''),
    )


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
        PREBUF_FRAMES  = 3  # buffer 3 frames (~300ms) before starting real-time pacing

        while not self._stop_event.is_set():
            try:
                text = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                total = 0
                buf   = b''
                t0    = None  # wall-clock start of playback
                frames_sent = 0
                prebuf = []   # pre-buffer queue

                for raw_chunk in self._adapter.synthesize_stream(text):
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
                                # Flush pre-buffer and start real-time clock
                                t0 = _time.monotonic()
                                for pf in prebuf:
                                    msg = AudioChunk()
                                    msg.header.stamp = self.get_clock().now().to_msg()
                                    msg.format = "audio/pcm-16k"
                                    msg.data   = list(pf)
                                    self._pub.publish(msg)
                                    frames_sent += 1
                                prebuf = []
                            continue

                        # Real-time pacing
                        target = t0 + frames_sent * FRAME_DURATION
                        now = _time.monotonic()
                        if now < target:
                            _time.sleep(target - now)
                        msg = AudioChunk()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.format = "audio/pcm-16k"
                        msg.data   = list(frame)
                        self._pub.publish(msg)
                        frames_sent += 1

                # Flush any remaining pre-buffer (short utterances < PREBUF_FRAMES)
                if prebuf and not self._stop_event.is_set():
                    t0 = _time.monotonic()
                    for pf in prebuf:
                        msg = AudioChunk()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.format = "audio/pcm-16k"
                        msg.data   = list(pf)
                        self._pub.publish(msg)
                        frames_sent += 1

                # flush remainder
                if buf and not self._stop_event.is_set():
                    if t0 is not None:
                        target = t0 + frames_sent * FRAME_DURATION
                        now = _time.monotonic()
                        if now < target:
                            _time.sleep(target - now)
                    msg = AudioChunk()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.format = "audio/pcm-16k"
                    msg.data   = list(buf)
                    self._pub.publish(msg)
                log.info(f"[tts] spoke {len(text)} chars → {total} bytes ({frames_sent} frames, streaming)")
            except Exception as e:
                log.error(f"[tts] synthesis error: {e}", exc_info=True)

    def _status_dict(self) -> dict:
        return {
            "state":     self.state,
            "topic_in":  [{"topic": self._input_topic,  "format": "data/json",     "desc": "text to synthesize"}],
            "topic_out": [{"topic": self._output_topic, "format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
        }


# ── Plugin ────────────────────────────────────────────────────────────────────

class TTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._adapter  = _build_tts_adapter(plugin_cfg)
        self._nodes: dict[str, _TTSNode] = {}           # key = instance_id
        self._instance_configs: dict[str, dict] = {}    # key = instance_id → per-instance config
        self._executor = executor
        log.info(f"[tts] plugin init: provider={plugin_cfg.get('provider')}, "
                 f"model={plugin_cfg.get('model','') or '(default)'}, "
                 f"api_key={'set' if plugin_cfg.get('api_key') else 'MISSING'}")
        if not self._adapter:
            log.warning("[tts] adapter not configured (missing API key)")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
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
                inferred_out = f"{input_topic}/tts" if input_topic else ""
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "idle",
                    "topic_in":  [{"topic": input_topic,  "format": "data/json",     "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}] if inferred_out else [],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            # Aggregate info (no instance_id = ping/overview only)
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "audio/pcm-16k", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/tts" if input_topic else ""
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
            input_topic = args.get("input_topic") or ''
            node_key = instance_id or input_topic or '_default'
            if node_key not in self._nodes:
                adapter = self._adapter
                if instance_id and instance_id in self._instance_configs:
                    inst_adapter = _build_tts_adapter(self._instance_configs[instance_id])
                    if inst_adapter:
                        adapter = inst_adapter
                node = _TTSNode(input_topic or None, adapter,
                                node_suffix=node_key.replace('/', '_').replace('-', '_'))
                self._executor.add_node(node)
                self._nodes[node_key] = node
            elif input_topic and self._nodes[node_key]._input_topic != input_topic:
                # Input topic changed for existing instance — recreate
                old_node = self._nodes[node_key]
                old_node.stop()
                self._executor.remove_node(old_node)
                adapter = self._adapter
                if instance_id and instance_id in self._instance_configs:
                    inst_adapter = _build_tts_adapter(self._instance_configs[instance_id])
                    if inst_adapter:
                        adapter = inst_adapter
                node = _TTSNode(input_topic, adapter,
                                node_suffix=node_key.replace('/', '_').replace('-', '_'))
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
            text = args.get("text", "")
            if not text:
                raise ValueError("text is required")
            node_key = instance_id or '_default'
            if node_key not in self._nodes:
                adapter = self._adapter
                if instance_id and instance_id in self._instance_configs:
                    inst_adapter = _build_tts_adapter(self._instance_configs[instance_id])
                    if inst_adapter:
                        adapter = inst_adapter
                node = _TTSNode(None, adapter,
                                node_suffix=node_key.replace('/', '_').replace('-', '_'))
                self._executor.add_node(node)
                self._nodes[node_key] = node
            node = self._nodes[node_key]
            if node.state != "running":
                node.start()
            node.enqueue(text)
            return {"status": "queued", "text": text}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v}
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    self._nodes[instance_id].stop()
                    self._executor.remove_node(self._nodes[instance_id])
                    del self._nodes[instance_id]
                return {"status": "configured", "instance_id": instance_id}
            else:
                self._adapter = _build_tts_adapter(cfg)
                for key in list(self._nodes.keys()):
                    self._nodes[key].stop()
                    self._executor.remove_node(self._nodes[key])
                    del self._nodes[key]
                return {"status": "configured", "adapter_ok": self._adapter is not None}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        return self._adapter.synthesize(text)
