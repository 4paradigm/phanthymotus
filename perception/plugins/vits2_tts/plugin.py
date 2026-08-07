"""Independent ROS2/MCP plugin for VITS2 ONNX CPU synthesis."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Optional

from audio_msgs.msg import AudioChunk
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .adapter import CHUNK_BYTES, SAMPLE_RATE, TTSAdapter, build_adapter


log = logging.getLogger(__name__)

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


class _Vits2TTSNode(Node):
    def __init__(
        self,
        input_topic: Optional[str],
        adapter: TTSAdapter,
        node_suffix: str = "",
    ):
        super().__init__(f"vits2_tts_{node_suffix}" if node_suffix else "vits2_tts")
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

    def _worker(self):
        frame_duration = CHUNK_BYTES / (SAMPLE_RATE * 2)
        while not self._stop_event.is_set():
            try:
                text = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                started = None
                for frame_index, pcm in enumerate(self._adapter.synthesize_stream(text)):
                    if self._stop_event.is_set():
                        break
                    now = time.monotonic()
                    if started is None:
                        started = now
                    target = started + frame_index * frame_duration
                    # Rebase after an ONNX segment takes longer than one frame;
                    # otherwise all buffered frames would be published at once.
                    if target < now - frame_duration:
                        started = now - frame_index * frame_duration
                        target = now
                    delay = target - now
                    if delay > 0:
                        time.sleep(delay)
                    self._publish(pcm)
            except Exception:
                log.exception("[vits2_tts] synthesis failed")

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
    """VITS2 implementation exposed as the `vits2_tts` MCP tool."""

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
                    "[vits2_tts] warmup completed: bytes=%d elapsed=%.3fs",
                    warmup_bytes,
                    time.monotonic() - started,
                )
        except Exception as exc:
            log.exception("[vits2_tts] failed to load model")
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
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._load_error:
                return {
                    "name": "VITS2 TTS",
                    "manufacture": "Embodied",
                    "model": "vits2-onnx-cpu",
                    "state": "error",
                    "desc": self._load_error,
                }
            if instance_id and instance_id in self._nodes:
                return {
                    "name": "VITS2 TTS",
                    "manufacture": "Embodied",
                    "model": "vits2-onnx-cpu",
                    **self._nodes[instance_id].status(),
                    "desc": "VITS2 ONNX CPU text-to-speech",
                }
            input_topic = args.get("input_topic", "")
            if instance_id:
                output_topic = (
                    f"{input_topic}/tts" if input_topic else "/perception/tts"
                )
                return {
                    "name": "VITS2 TTS",
                    "manufacture": "Embodied",
                    "model": "vits2-onnx-cpu",
                    "state": "idle",
                    "topic_in": (
                        [{"topic": input_topic, "format": "data/json", "desc": ""}]
                        if input_topic
                        else []
                    ),
                    "topic_out": [
                        {"topic": output_topic, "format": "audio/pcm-16k", "desc": ""}
                    ],
                    "desc": "VITS2 ONNX CPU text-to-speech",
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
                "model": "vits2-onnx-cpu",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "VITS2 ONNX CPU text-to-speech",
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
                self._remove_node(instance_id)
            elif not instance_id:
                for key in list(self._nodes):
                    self._remove_node(key)
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
            return {"status": "queued", "text": text}

        if action == "config":
            if "speaker_id" in args:
                self._cfg["speaker_id"] = int(args["speaker_id"])
            if "speed" in args:
                self._cfg["speed"] = float(args["speed"])
            adapter = build_adapter(self._cfg)
            for key in list(self._nodes):
                self._remove_node(key)
            self._adapter = adapter
            self._load_error = None
            return {"status": "configured"}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        if not self._adapter:
            raise RuntimeError(self._load_error or "TTS adapter not configured")
        return self._adapter.synthesize(text)
