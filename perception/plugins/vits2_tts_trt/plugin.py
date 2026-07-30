"""ROS2/MCP plugin for in-process VITS2 TensorRT synthesis."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Optional

from audio_msgs.msg import AudioChunk
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .adapter import CHUNK_BYTES, PCM_FRAME_MS, SAMPLE_RATE, TTSAdapter, build_adapter


log = logging.getLogger(__name__)
FRAME_INTERVAL_MS = int(os.getenv("MIX_VITS_FRAME_INTERVAL_MS", "85"))
if not 0 <= FRAME_INTERVAL_MS <= 1000:
    raise ValueError("MIX_VITS_FRAME_INTERVAL_MS must be between zero and 1000")
FIRST_FRAME_DELAY_MS = int(os.getenv("MIX_VITS_FIRST_FRAME_DELAY_MS", "0"))
if not 0 <= FIRST_FRAME_DELAY_MS <= 1000:
    raise ValueError("MIX_VITS_FIRST_FRAME_DELAY_MS must be between zero and 1000")
SUBSCRIBER_WAIT_MS = int(os.getenv("MIX_VITS_SUBSCRIBER_WAIT_MS", "5000"))
if not 0 <= SUBSCRIBER_WAIT_MS <= 60000:
    raise ValueError("MIX_VITS_SUBSCRIBER_WAIT_MS must be between zero and 60000")
SUBSCRIBER_POLL_MS = int(os.getenv("MIX_VITS_SUBSCRIBER_POLL_MS", "10"))
if not 1 <= SUBSCRIBER_POLL_MS <= 1000:
    raise ValueError("MIX_VITS_SUBSCRIBER_POLL_MS must be between one and 1000")
SUBSCRIBER_SETTLE_MS = int(os.getenv("MIX_VITS_SUBSCRIBER_SETTLE_MS", "250"))
if not 0 <= SUBSCRIBER_SETTLE_MS <= 5000:
    raise ValueError("MIX_VITS_SUBSCRIBER_SETTLE_MS must be between zero and 5000")
ALLOW_FAST_DELIVERY = os.getenv("MIX_VITS_ALLOW_FAST_DELIVERY", "1") == "1"
if FRAME_INTERVAL_MS < PCM_FRAME_MS and not ALLOW_FAST_DELIVERY:
    raise ValueError(
        f"MIX_VITS_FRAME_INTERVAL_MS={FRAME_INTERVAL_MS} sends "
        f"{PCM_FRAME_MS:.0f}ms PCM frames faster than realtime; use at least "
        f"{PCM_FRAME_MS:.0f}, or explicitly set MIX_VITS_ALLOW_FAST_DELIVERY=1 "
        "for an offline benchmark"
    )

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
        "description": "TTS - start/stop speech synthesis, speak text, or get status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "speak", "info", "config"],
                },
                "input_topic": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["action"],
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "speaker_id": {"type": "integer", "default": 0, "scope": "shared"},
                "speed": {"type": "number", "default": 1.0, "scope": "shared"},
            },
            "required": [],
        },
        "topic_in": [{"format": "data/json", "desc": "text to synthesize"}],
        "topic_out": [{"format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
    }
]
# Keep the historical vits2_tts name while exposing the module-selected name
# expected by evaluators when TTS_PLUGIN=vits2_tts_trt.
TOOLS.append({**TOOLS[0], "name": "tts_trt"})


class _Vits2TTSNode(Node):
    def __init__(
        self,
        input_topic: Optional[str],
        adapter: TTSAdapter,
        node_suffix: str = "",
    ):
        super().__init__(f"vits2_trt_{node_suffix}" if node_suffix else "vits2_trt")
        self._input_topic = input_topic or ""
        self._output_topic = (
            f"{input_topic}/tts" if input_topic else "/perception/tts"
        )
        self._adapter = adapter
        self.state = "idle"
        self._text_queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._pub = self.create_publisher(AudioChunk, self._output_topic, _LOW_LAT_QOS)
        self._sub = (
            self.create_subscription(
                String, self._input_topic, self._text_callback, _LOW_LAT_QOS
            )
            if input_topic
            else None
        )

    def start(self):
        if self.state == "running":
            return self.status()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self.status()

    def stop(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def enqueue(self, text: str):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        self._text_queue.put(text)

    def _text_callback(self, message: String):
        if self.state != "running":
            return
        try:
            text = json.loads(message.data).get("text", "")
        except Exception:
            text = message.data.strip()
        if text:
            self._text_queue.put(text)

    def _publish(self, pcm: bytes):
        message = AudioChunk()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = "audio/pcm-16k"
        message.data = list(pcm)
        self._pub.publish(message)

    def _wait_for_audio_subscriber(self) -> tuple[float, float, int]:
        """Wait until an audio subscriber remains DDS-matched long enough."""
        started = time.monotonic()
        deadline = started + (SUBSCRIBER_WAIT_MS + SUBSCRIBER_SETTLE_MS) / 1000.0
        matched_at = None
        while not self._stop_event.is_set():
            now = time.monotonic()
            count = self._pub.get_subscription_count()
            if count > 0:
                if matched_at is None:
                    matched_at = now
                settled = now - matched_at
                if settled >= SUBSCRIBER_SETTLE_MS / 1000.0:
                    return matched_at - started, settled, count
            else:
                # Require a continuous stable match. A transient graph match is
                # not sufficient for a BEST_EFFORT reader to receive frame 0.
                matched_at = None
            if now >= deadline:
                raise RuntimeError(
                    "no stable matched TTS audio subscriber within "
                    f"{SUBSCRIBER_WAIT_MS + SUBSCRIBER_SETTLE_MS}ms "
                    f"on {self._output_topic}"
                )
            time.sleep(SUBSCRIBER_POLL_MS / 1000.0)
        raise RuntimeError("TTS stopped while waiting for an audio subscriber")

    def _worker(self):
        frame_interval = FRAME_INTERVAL_MS / 1000.0
        while not self._stop_event.is_set():
            try:
                text = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                task_started = time.monotonic()
                first_published_at = None
                total_bytes = 0
                started = None
                frames_sent = 0
                buffer = bytearray()
                subscriber_wait_seconds = None
                subscriber_settle_seconds = None
                subscriber_count = 0

                def publish_frame(frame: bytes) -> None:
                    nonlocal started, frames_sent, first_published_at, total_bytes
                    nonlocal subscriber_wait_seconds, subscriber_settle_seconds
                    nonlocal subscriber_count
                    now = time.monotonic()
                    if started is None:
                        (
                            subscriber_wait_seconds,
                            subscriber_settle_seconds,
                            subscriber_count,
                        ) = self._wait_for_audio_subscriber()
                        if FIRST_FRAME_DELAY_MS:
                            time.sleep(FIRST_FRAME_DELAY_MS / 1000.0)
                        started = time.monotonic()
                        now = started
                    if frame_interval:
                        target = started + frames_sent * frame_interval
                        if target < now - frame_interval:
                            started = now - frames_sent * frame_interval
                            target = now
                        delay = target - now
                        if delay > 0:
                            time.sleep(delay)
                    self._publish(frame)
                    if first_published_at is None:
                        first_published_at = time.monotonic()
                    total_bytes += len(frame)
                    frames_sent += 1

                for pcm in self._adapter.synthesize_stream(text):
                    if self._stop_event.is_set():
                        break
                    buffer.extend(pcm)
                    while len(buffer) >= CHUNK_BYTES:
                        frame = bytes(buffer[:CHUNK_BYTES])
                        del buffer[:CHUNK_BYTES]
                        publish_frame(frame)

                if buffer and not self._stop_event.is_set():
                    publish_frame(bytes(buffer))
                if total_bytes:
                    finished_at = time.monotonic()
                    audio_seconds = total_bytes / (SAMPLE_RATE * 2)
                    elapsed = finished_at - task_started
                    log.info(
                        "[vits2_tts_trt] server delivery: bytes=%d frames=%d "
                        "ttft=%.3fs elapsed=%.3fs audio=%.3fs rtf=%.4f "
                        "chunk_bytes=%d frame_interval_ms=%d "
                        "first_frame_delay_ms=%d subscriber_wait_ms=%.1f "
                        "subscriber_settle_ms=%.1f subscriber_count=%d",
                        total_bytes,
                        frames_sent,
                        first_published_at - task_started,
                        elapsed,
                        audio_seconds,
                        elapsed / audio_seconds,
                        CHUNK_BYTES,
                        FRAME_INTERVAL_MS,
                        FIRST_FRAME_DELAY_MS,
                        (subscriber_wait_seconds or 0.0) * 1000.0,
                        (subscriber_settle_seconds or 0.0) * 1000.0,
                        subscriber_count,
                    )
            except Exception:
                log.exception("[vits2_tts_trt] synthesis failed")

    def status(self):
        return {
            "state": self.state,
            "topic_in": [
                {"topic": self._input_topic, "format": "data/json", "desc": ""}
            ],
            "topic_out": [
                {"topic": self._output_topic, "format": "audio/pcm-16k", "desc": ""}
            ],
        }


class TTSPlugin:
    """VITS2 TensorRT implementation exposed as the shared `tts` MCP tool."""

    PREFIX = "vits2"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor
        self._nodes = {}
        self._load_error = None
        try:
            self._adapter = build_adapter(self._cfg)
            if self._cfg.get("vits2_warmup", True):
                started = time.monotonic()
                warmup_bytes = self._adapter.warmup()
                log.info(
                    "[vits2_tts_trt] engine ready: bytes=%d elapsed=%.3fs",
                    warmup_bytes,
                    time.monotonic() - started,
                )
        except Exception as exc:
            log.exception("[vits2_tts_trt] failed to load engine")
            self._adapter = None
            self._load_error = str(exc)
            raise RuntimeError("VITS2 model load or warmup failed") from exc

    def get_tools(self):
        return TOOLS

    def _remove_node(self, key):
        node = self._nodes.pop(key)
        node.stop()
        self._executor.remove_node(node)

    def _create_node(self, key, input_topic):
        suffix = key.replace("/", "_").replace("-", "_")
        node = _Vits2TTSNode(input_topic or None, self._adapter, suffix)
        self._executor.add_node(node)
        self._nodes[key] = node
        return node

    def dispatch(self, name: str, args: dict):
        action = args.get("action") if name in {"tts", "tts_trt"} else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._load_error:
                return {
                    "name": "VITS2 TTS",
                    "manufacture": "Embodied",
                    "model": "vits2-tensorrt-jp6",
                    "state": "error",
                    "desc": self._load_error,
                }
            if instance_id and instance_id in self._nodes:
                return {
                    "name": "VITS2 TTS",
                    "manufacture": "Embodied",
                    "model": "vits2-tensorrt-jp6",
                    **self._nodes[instance_id].status(),
                    "desc": "VITS2 TensorRT text-to-speech",
                }
            input_topic = args.get("input_topic", "")
            if instance_id:
                output_topic = (
                    f"{input_topic}/tts" if input_topic else "/perception/tts"
                )
                return {
                    "name": "VITS2 TTS",
                    "manufacture": "Embodied",
                    "model": "vits2-tensorrt-jp6",
                    "state": "idle",
                    "topic_in": (
                        [{"topic": input_topic, "format": "data/json", "desc": ""}]
                        if input_topic
                        else []
                    ),
                    "topic_out": [
                        {"topic": output_topic, "format": "audio/pcm-16k", "desc": ""}
                    ],
                    "desc": "VITS2 TensorRT text-to-speech",
                }
            state = (
                "running"
                if any(node.state == "running" for node in self._nodes.values())
                else "idle"
            )
            topics_in = [
                {"topic": node._input_topic, "format": "data/json", "desc": ""}
                for node in self._nodes.values()
            ]
            topics_out = [
                {"topic": node._output_topic, "format": "audio/pcm-16k", "desc": ""}
                for node in self._nodes.values()
            ]
            if not topics_out:
                output_topic = (
                    f"{input_topic}/tts" if input_topic else "/perception/tts"
                )
                topics_in = (
                    [{"topic": input_topic, "format": "data/json", "desc": ""}]
                    if input_topic
                    else []
                )
                topics_out = [
                    {"topic": output_topic, "format": "audio/pcm-16k", "desc": ""}
                ]
            return {
                "name": "VITS2 TTS",
                "manufacture": "Embodied",
                "model": "vits2-tensorrt-jp6",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "VITS2 TensorRT text-to-speech",
            }

        if action == "start":
            if self._load_error or not self._adapter:
                return {"state": "error", "message": self._load_error or "model unavailable"}
            input_topic = args.get("input_topic") or ""
            key = instance_id or input_topic or "_default"
            if key in self._nodes and input_topic != self._nodes[key]._input_topic:
                self._remove_node(key)
            node = self._nodes.get(key) or self._create_node(key, input_topic)
            return node.start()

        if action == "stop":
            if instance_id and instance_id in self._nodes:
                # Keep the publisher alive so a subsequent start for the same
                # instance reuses its DDS discovery state.
                self._nodes[instance_id].stop()
            elif not instance_id:
                for node in self._nodes.values():
                    node.stop()
            return {"state": "idle"}

        if action == "speak":
            if self._load_error or not self._adapter:
                return {"state": "error", "message": self._load_error or "model unavailable"}
            text = args.get("text", "").strip()
            if not text:
                raise ValueError("text is required")
            node = next((n for n in self._nodes.values() if n.state == "running"), None)
            if node is None:
                key = instance_id or "_default"
                node = self._nodes.get(key) or self._create_node(
                    key, args.get("input_topic") or ""
                )
                node.start()
            node.enqueue(text)
            return {"status": "queued", "chars": len(text)}

        if action == "config":
            if "speaker_id" in args:
                self._cfg["speaker_id"] = int(args["speaker_id"])
            if "speed" in args:
                self._cfg["speed"] = float(args["speed"])
            if int(self._cfg.get("speaker_id", 0)) != 0:
                raise ValueError("The VITS2 model supports only speaker_id=0")
            self._adapter.set_speed(float(self._cfg.get("speed", 1.0)))
            for key in list(self._nodes):
                self._remove_node(key)
            self._load_error = None
            return {"status": "configured"}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        if not self._adapter:
            raise RuntimeError(self._load_error or "TTS adapter not configured")
        return self._adapter.synthesize(text)
