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
            "x-hooks": {
                "on_interrupt_speak": {"action": "interrupt"},
            }
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

class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    def synthesize_stream(self, text: str):
        """Yield raw PCM bytes as they arrive. Default: collect all."""
        yield self.synthesize(text)


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

    def synthesize(self, text: str) -> bytes:
        return b''.join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        import struct
        audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        float_samples = audio.samples
        # Matcha + vocos-16khz outputs 16kHz directly, no resampling needed
        pcm = struct.pack(f'<{len(float_samples)}h',
                         *[int(max(-32768, min(32767, s * 32767))) for s in float_samples])
        for i in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[i:i + CHUNK_BYTES]




def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    import os
    model_dir = cfg.get('model_dir', '/models/sherpa-onnx/tts')
    speaker_id = int(cfg.get('speaker_id', 0))
    speed = float(cfg.get('speed', 1.0))
    return SherpaOnnxTTSAdapter(model_dir, speaker_id, speed)


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
        # Dry-run: verify model can synthesize before declaring running
        try:
            test_chunks = list(self._adapter.synthesize_stream("."))
            if not test_chunks:
                return {"state": "error", "message": "TTS dry-run produced no audio"}
        except Exception as e:
            return {"state": "error", "message": f"TTS dry-run failed: {e}"}
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

    def interrupt(self) -> dict:
        """立即中止当前播放：清空队列 + 设置 interrupt flag 让 worker 停止当前 utterance。"""
        # 清空待播放队列
        cleared = 0
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        # 设置 interrupt flag（worker 在每个 frame 前检查）
        self._interrupt_flag.set()
        log.info(f"[tts] interrupted: cleared {cleared} queued item(s)")
        return {"status": "interrupted", "cleared": cleared}

    def enqueue(self, text: str, trace_id: str = '', action_id: str = ''):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        # 分段：超过 280 字按标点切分，避免超长合成导致延迟或失败
        segments = self._split_text(text, max_chars=280)
        if len(segments) <= 1:
            self._text_queue.put((text, trace_id, action_id))
        else:
            # 只有最后一段带 action_id（触发 ACP callback）
            for i, seg in enumerate(segments):
                is_last = (i == len(segments) - 1)
                self._text_queue.put((seg, trace_id, action_id if is_last else ''))
            log.info(f"[tts] split {len(text)} chars into {len(segments)} segments")

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
            self._text_queue.put((text, ''))

    def _worker(self):
        from audio_msgs.msg import AudioChunk
        import time as _time

        # Real-time pacing: publish frames at playback rate to avoid bursts/gaps
        FRAME_DURATION = CHUNK_BYTES / (SAMPLE_RATE * 2)  # 0.1s per 3200-byte frame
        PREBUF_FRAMES  = 3  # buffer 3 frames (~300ms) before starting real-time pacing

        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            # Unpack queue item: (text, trace_id, action_id) or legacy formats
            if isinstance(item, tuple):
                if len(item) == 3:
                    text, _trace_id, _action_id = item
                elif len(item) == 2:
                    text, _trace_id = item
                    _action_id = ''
                else:
                    text, _trace_id, _action_id = str(item[0]), '', ''
            else:
                text, _trace_id, _action_id = item, '', ''
            try:
                import time as _time
                t_start = _time.monotonic()
                t_start_wall = _time.time()  # wall-clock for perf span
                t0_wall = None  # wall-clock when playback starts (prebuf complete)
                total = 0
                buf   = b''
                t0    = None  # wall-clock start of playback
                frames_sent = 0
                prebuf = []   # pre-buffer queue

                for raw_chunk in self._adapter.synthesize_stream(text):
                    if self._stop_event.is_set() or self._interrupt_flag.is_set():
                        break
                    buf  += raw_chunk
                    total += len(raw_chunk)
                    # split into CHUNK_BYTES frames
                    while len(buf) >= CHUNK_BYTES:
                        frame = buf[:CHUNK_BYTES]
                        buf   = buf[CHUNK_BYTES:]

                        # Check interrupt before publishing each frame
                        if self._interrupt_flag.is_set():
                            break

                        # Pre-buffer phase: accumulate a few frames before pacing
                        if t0 is None:
                            prebuf.append(frame)
                            if len(prebuf) >= PREBUF_FRAMES:
                                # Flush pre-buffer and start real-time clock
                                t0 = _time.monotonic()
                                t0_wall = _time.time()
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
                if prebuf and not self._stop_event.is_set() and not self._interrupt_flag.is_set():
                    t0 = _time.monotonic()
                    for pf in prebuf:
                        msg = AudioChunk()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.format = "audio/pcm-16k"
                        msg.data   = list(pf)
                        self._pub.publish(msg)
                        frames_sent += 1

                # flush remainder
                if buf and not self._stop_event.is_set() and not self._interrupt_flag.is_set():
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

                # Clear interrupt flag after utterance is done (interrupted or complete)
                if self._interrupt_flag.is_set():
                    self._interrupt_flag.clear()
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
                if _action_id:
                    try:
                        import urllib.request as _urllib
                        import ssl as _ssl
                        import os as _os
                        _agent_core_url = _os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
                        _ctx = _ssl.create_default_context()
                        _ctx.check_hostname = False
                        _ctx.verify_mode = _ssl.CERT_NONE
                        was_interrupted = self._interrupt_flag.is_set()
                        _payload = json.dumps({
                            "action_id": _action_id,
                            "status": "cancelled" if was_interrupted else "completed",
                            "result": {"text": text[:100], "frames": frames_sent},
                        }).encode()
                        _req = _urllib.Request(
                            f"{_agent_core_url}/api/acp/complete",
                            data=_payload,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        _urllib.urlopen(_req, timeout=3, context=_ctx)
                        log.info(f"[tts] ACP complete: {_action_id} ({'cancelled' if was_interrupted else 'completed'})")
                    except Exception as e:
                        log.warning(f"[tts] ACP callback failed: {e}")
            except Exception as e:
                log.error(f"[tts] synthesis error: {e}", exc_info=True)

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

class TTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg      = plugin_cfg
        self._loading  = False
        self._load_error = None
        try:
            self._adapter  = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] failed to load model: {e}", exc_info=True)
            self._adapter = None
            self._load_error = str(e)
        self._nodes: dict[str, _TTSNode] = {}
        self._executor = executor
        log.info(f"[tts] plugin init: sherpa-onnx VITS, "
                 f"speaker_id={plugin_cfg.get('speaker_id', 0)}, speed={plugin_cfg.get('speed', 1.0)}")

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
                node = _TTSNode(input_topic or None, self._adapter,
                                node_suffix=node_key.replace('/', '_').replace('-', '_'))
                self._executor.add_node(node)
                self._nodes[node_key] = node
            elif input_topic and self._nodes[node_key]._input_topic != input_topic:
                # Input topic changed for existing instance — recreate
                old_node = self._nodes[node_key]
                old_node.stop()
                self._executor.remove_node(old_node)
                node = _TTSNode(input_topic, self._adapter,
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
                    node = _TTSNode(input_topic, adapter,
                                    node_suffix=node_key.replace('/', '_').replace('-', '_'))
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                else:
                    node = self._nodes[node_key]
                if node.state != "running":
                    node.start()
            # ACP: 生成 action_id
            import uuid as _uuid
            action_id = f"speak-{_uuid.uuid4().hex[:8]}"
            node.enqueue(text, trace_id=args.get('_trace_id', ''), action_id=action_id)
            return {"status": "queued", "action_id": action_id, "text": text}

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

        elif action == "interrupt":
            # 立即中止所有 TTS 播放（清空队列 + 停止当前 utterance）
            total_cleared = 0
            interrupted_count = 0
            if instance_id and instance_id in self._nodes:
                result = self._nodes[instance_id].interrupt()
                total_cleared += result.get('cleared', 0)
                interrupted_count += 1
            elif not instance_id:
                for node in self._nodes.values():
                    if node.state == "running":
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
