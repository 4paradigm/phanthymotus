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

from .adapter import (
    CHUNK_BYTES,
    PCM_FRAME_MS,
    SAMPLE_RATE,
    TTSAdapter,
    Vits2TensorRTAdapter,
    build_adapter,
)


log = logging.getLogger(__name__)


# End-of-utterance marker used by the public TTS audio protocol.
AUDIO_EOF_MAGIC = b"\x01\x00\xff\xff\x01\x00\xff\xff"


def _ensure_release(model_dir: str) -> None:
    """Install and verify the immutable TensorRT runtime release."""
    from utils.model_downloader import ensure_model

    ensure_model("vits2", model_dir)
FRAME_INTERVAL_MS = int(os.getenv("MIX_VITS_FRAME_INTERVAL_MS", "70"))
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
SUBSCRIBER_SETTLE_MS = int(os.getenv("MIX_VITS_SUBSCRIBER_SETTLE_MS", "500"))
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
                    "enum": [
                        "start",
                        "stop",
                        "speak",
                        "info",
                        "config",
                        "interrupt",
                    ],
                },
                "input_topic": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["action"],
            "x-completion": {"actions": ["speak"], "timeout": 60},
            "x-hooks": {"on_interrupt_speak": {"action": "interrupt"}},
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
        self._interrupt_event = threading.Event()
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
        self._interrupt_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self.status()

    def stop(self):
        self.request_stop()
        self._complete_discarded_actions()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def request_stop(self):
        """Request worker shutdown without waiting for thread termination."""
        self._stop_event.set()

    def interrupt(self) -> dict:
        """Cancel the active utterance and discard queued utterances."""
        self._interrupt_event.set()
        cleared = self._complete_discarded_actions()
        return {"status": "interrupted", "cleared": cleared}

    def _complete_discarded_actions(self) -> int:
        """Cancel queued MCP actions that will not reach the worker."""
        discarded = []
        while True:
            try:
                item = self._text_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                text, action_id = item
            else:
                text, action_id = str(item), ""
            discarded.append((text, action_id))
        for text, action_id in discarded:
            self._complete_action(action_id, text, 0, interrupted=True)
        return len(discarded)

    def enqueue(self, text: str, action_id: str = ""):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        self._text_queue.put((text, action_id))

    def _text_callback(self, message: String):
        if self.state != "running":
            return
        try:
            text = json.loads(message.data).get("text", "")
        except Exception:
            text = message.data.strip()
        if text:
            self._text_queue.put((text, ""))

    def _publish(self, pcm: bytes):
        message = AudioChunk()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = "audio/pcm-16k"
        message.data = list(pcm)
        self._pub.publish(message)

    def _publish_eof(self):
        """Publish the protocol end-of-utterance marker."""
        self._publish(AUDIO_EOF_MAGIC)

    @staticmethod
    def _complete_action(
        action_id: str, text: str, frames_sent: int, interrupted: bool
    ) -> None:
        """Notify Agent Core that an MCP speak action has terminated."""
        if not action_id:
            return
        try:
            import urllib.request
            from urllib.parse import urlparse

            agent_core_url = os.getenv("AGENT_CORE_URL", "https://localhost:15678")
            if urlparse(agent_core_url).scheme != "https":
                raise ValueError("AGENT_CORE_URL must use HTTPS")
            payload = json.dumps(
                {
                    "action_id": action_id,
                    "status": "cancelled" if interrupted else "completed",
                    "result": {"text": text[:100], "frames": frames_sent},
                }
            ).encode()
            request = urllib.request.Request(
                f"{agent_core_url}/api/acp/complete",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(request, timeout=3)
        except Exception as exc:
            log.warning("[vits2_tts_trt] ACP completion callback failed: %s", exc)

    def _utterance_cancelled(self) -> bool:
        return self._stop_event.is_set() or self._interrupt_event.is_set()

    def _wait_for_audio_subscriber(
        self, cancel_event: Optional[threading.Event] = None
    ) -> tuple[float, float, int]:
        """Wait until an audio subscriber remains DDS-matched long enough."""
        started = time.monotonic()
        deadline = started + (SUBSCRIBER_WAIT_MS + SUBSCRIBER_SETTLE_MS) / 1000.0
        matched_at = None
        while not self._utterance_cancelled() and not (
            cancel_event and cancel_event.is_set()
        ):
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
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("subscriber wait cancelled")
        if self._interrupt_event.is_set():
            raise RuntimeError("TTS interrupted while waiting for an audio subscriber")
        raise RuntimeError("TTS stopped while waiting for an audio subscriber")

    def _worker(self):
        frame_interval = FRAME_INTERVAL_MS / 1000.0
        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            if isinstance(item, tuple):
                text, action_id = item
            else:
                text, action_id = str(item), ""
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()
                self._publish_eof()
                self._complete_action(action_id, text, 0, interrupted=True)
                continue
            subscriber_gate_cancel = threading.Event()
            subscriber_gate_done = threading.Event()
            subscriber_gate_result = {}

            def wait_for_subscriber() -> None:
                try:
                    subscriber_gate_result["value"] = (
                        self._wait_for_audio_subscriber(subscriber_gate_cancel)
                    )
                except BaseException as exc:
                    subscriber_gate_result["error"] = exc
                finally:
                    subscriber_gate_done.set()

            subscriber_gate_thread = threading.Thread(
                target=wait_for_subscriber,
                name="vits2-trt-subscriber-gate",
                daemon=True,
            )
            # DDS discovery/settling runs in parallel with frontend + first
            # TensorRT synthesis so the stronger BEST_EFFORT guard does not
            # become pure TTFT overhead.
            subscriber_gate_thread.start()
            eof_published = False
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

                def publish_frame(frame: bytes) -> bool:
                    nonlocal started, frames_sent, first_published_at, total_bytes
                    nonlocal subscriber_wait_seconds, subscriber_settle_seconds
                    nonlocal subscriber_count
                    if self._utterance_cancelled():
                        return False
                    now = time.monotonic()
                    if started is None:
                        while not subscriber_gate_done.wait(timeout=0.05):
                            if self._utterance_cancelled():
                                return False
                        if "error" in subscriber_gate_result:
                            if self._utterance_cancelled():
                                return False
                            raise subscriber_gate_result["error"]
                        (
                            subscriber_wait_seconds,
                            subscriber_settle_seconds,
                            subscriber_count,
                        ) = subscriber_gate_result["value"]
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
                    if self._utterance_cancelled():
                        return False
                    self._publish(frame)
                    if first_published_at is None:
                        first_published_at = time.monotonic()
                    total_bytes += len(frame)
                    frames_sent += 1
                    return True

                interrupted = False
                for pcm in self._adapter.synthesize_stream(text):
                    if self._utterance_cancelled():
                        interrupted = True
                        break
                    buffer.extend(pcm)
                    while len(buffer) >= CHUNK_BYTES:
                        frame = bytes(buffer[:CHUNK_BYTES])
                        del buffer[:CHUNK_BYTES]
                        if not publish_frame(frame):
                            interrupted = True
                            break
                    if interrupted:
                        break

                if buffer and not self._utterance_cancelled():
                    if not publish_frame(bytes(buffer)):
                        interrupted = True
                interrupted = interrupted or self._interrupt_event.is_set()
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
                self._publish_eof()
                eof_published = True
                self._complete_action(action_id, text, frames_sent, interrupted)
                if interrupted:
                    log.info(
                        "[vits2_tts_trt] utterance interrupted after %d frames",
                        frames_sent,
                    )
                self._interrupt_event.clear()
            except Exception:
                log.exception("[vits2_tts_trt] synthesis failed")
                self._complete_action(action_id, text, 0, interrupted=True)
            finally:
                if not eof_published:
                    self._publish_eof()
                subscriber_gate_cancel.set()
                subscriber_gate_thread.join(timeout=0.1)

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
    """VITS2 TensorRT implementation exposed as an optional MCP tool."""

    PREFIX = "vits2"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor
        self._nodes = {}
        # Serializes instance lifecycle state across concurrent MCP requests.
        self._nodes_lock = threading.RLock()
        self._starting = {}
        self._adapter = None
        self._model_name = "vits2"
        self._load_error = None
        self._load_lock = threading.Lock()
        backend = str(self._cfg.get("backend", "trt")).lower()
        if backend != "trt":
            raise ValueError("The JP6 VITS2 plugin supports backend=trt only")
        if int(self._cfg.get("speaker_id", 0)) != 0:
            raise ValueError("The VITS2 model supports only speaker_id=0")

    def _ensure_adapter(self):
        if self._adapter is not None:
            return self._adapter
        with self._load_lock:
            if self._adapter is not None:
                return self._adapter
            model_dir = self._cfg.get("model_dir", "/models/vits2")
            try:
                _ensure_release(model_dir)
                adapter = build_adapter(self._cfg)
                if self._cfg.get("warmup", True):
                    started = time.monotonic()
                    warmup_bytes = adapter.warmup()
                    log.info(
                        "[vits2_tts_trt] engine ready: bytes=%d elapsed=%.3fs",
                        warmup_bytes,
                        time.monotonic() - started,
                    )
                self._adapter = adapter
                if not isinstance(adapter, Vits2TensorRTAdapter):
                    raise RuntimeError("Unexpected non-TensorRT VITS2 adapter")
                self._model_name = "vits2-tensorrt-jp6"
                self._load_error = None
            except Exception as exc:
                self._load_error = str(exc)
                log.exception("[vits2_tts_trt] failed to load engine")
                raise RuntimeError("VITS2 model load or warmup failed") from exc
            return self._adapter

    def get_tools(self):
        return TOOLS

    def _dispose_node(self, node, key):
        """Release a node after its caller has removed it from _nodes."""
        try:
            node.request_stop()
            node.stop()
        except Exception:
            log.exception("[vits2_tts_trt] node stop failed: %s", key)
        try:
            self._executor.remove_node(node)
        except Exception:
            log.exception("[vits2_tts_trt] node removal failed: %s", key)
        try:
            node.destroy_node()
        except Exception:
            log.exception("[vits2_tts_trt] node destroy failed: %s", key)

    def _create_node(self, key, input_topic, adapter):
        suffix = key.replace("/", "_").replace("-", "_")
        node = _Vits2TTSNode(input_topic or None, adapter, suffix)
        self._executor.add_node(node)
        self._nodes[key] = node
        return node

    def _reserve_start(self, key):
        """Reserve an instance while model initialization executes outside the lock."""
        with self._nodes_lock:
            if key in self._starting:
                return None
            marker = threading.Event()
            self._starting[key] = marker
            return marker

    def _finish_start(self, key, marker):
        with self._nodes_lock:
            if self._starting.get(key) is marker:
                del self._starting[key]

    def dispatch(self, name: str, args: dict):
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._load_error:
                return {
                    "name": "VITS2 TTS",
                    "manufacture": "Embodied",
                    "model": self._model_name,
                    "state": "error",
                    "desc": self._load_error,
                }
            with self._nodes_lock:
                node = self._nodes.get(instance_id) if instance_id else None
                nodes = list(self._nodes.values())
                loading = bool(self._starting)
            if instance_id and node is not None:
                return {
                    "name": "VITS2 TTS",
                    "manufacture": "Embodied",
                    "model": self._model_name,
                    **node.status(),
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
                    "model": self._model_name,
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
                if any(node.state == "running" for node in nodes)
                else "loading" if loading else "idle"
            )
            topics_in = [
                {"topic": node._input_topic, "format": "data/json", "desc": ""}
                for node in nodes
            ]
            topics_out = [
                {"topic": node._output_topic, "format": "audio/pcm-16k", "desc": ""}
                for node in nodes
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
                "model": self._model_name,
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "VITS2 TensorRT text-to-speech",
            }

        if action == "start":
            input_topic = args.get("input_topic") or ""
            key = instance_id or input_topic or "_default"
            marker = self._reserve_start(key)
            if marker is None:
                return {"state": "loading", "message": "TTS start is already in progress"}
            try:
                adapter = self._ensure_adapter()
            except RuntimeError:
                self._finish_start(key, marker)
                return {"state": "error", "message": self._load_error}
            old_node = None
            try:
                with self._nodes_lock:
                    if marker.is_set():
                        return {"state": "idle"}
                    node = self._nodes.get(key)
                    if node is not None and input_topic != node._input_topic:
                        old_node = self._nodes.pop(key)
                        node = None
                    if node is not None:
                        return node.start()
                if old_node is not None:
                    self._dispose_node(old_node, key)
                with self._nodes_lock:
                    if marker.is_set():
                        return {"state": "idle"}
                    node = self._create_node(key, input_topic, adapter)
                    return node.start()
            finally:
                self._finish_start(key, marker)

        if action == "stop":
            with self._nodes_lock:
                keys = [instance_id] if instance_id else list(self._nodes)
                markers = [self._starting[key] for key in keys if key in self._starting]
                if not instance_id:
                    markers = list(self._starting.values())
                nodes = [(key, self._nodes.pop(key)) for key in keys if key in self._nodes]
            for marker in markers:
                marker.set()
            for key, node in nodes:
                self._dispose_node(node, key)
            return {"state": "idle"}

        if action == "speak":
            text = args.get("text", "").strip()
            if not text:
                raise ValueError("text is required")
            import uuid

            action_id = f"speak-{uuid.uuid4().hex[:8]}"
            with self._nodes_lock:
                node = next((n for n in self._nodes.values() if n.state == "running"), None)
            if node is not None:
                node.enqueue(text, action_id=action_id)
                return {"status": "queued", "action_id": action_id, "text": text}
            key = instance_id or "_default"
            marker = self._reserve_start(key)
            if marker is None:
                return {"status": "loading", "message": "TTS start is already in progress"}
            try:
                adapter = self._ensure_adapter()
            except RuntimeError:
                self._finish_start(key, marker)
                return {"state": "error", "message": self._load_error}
            try:
                with self._nodes_lock:
                    if marker.is_set():
                        return {"state": "idle"}
                    node = next(
                        (n for n in self._nodes.values() if n.state == "running"),
                        None,
                    )
                    if node is None:
                        node = self._create_node(
                            key, args.get("input_topic") or "", adapter
                        )
                        node.start()
            finally:
                self._finish_start(key, marker)
            node.enqueue(text, action_id=action_id)
            return {"status": "queued", "action_id": action_id, "text": text}

        if action == "config":
            if "speaker_id" in args:
                self._cfg["speaker_id"] = int(args["speaker_id"])
            if "speed" in args:
                self._cfg["speed"] = float(args["speed"])
            if int(self._cfg.get("speaker_id", 0)) != 0:
                raise ValueError("The VITS2 model supports only speaker_id=0")
            with self._nodes_lock:
                markers = list(self._starting.values())
                nodes = list(self._nodes.items())
                self._nodes.clear()
            for marker in markers:
                marker.set()
            if self._adapter is not None:
                self._adapter.set_speed(float(self._cfg.get("speed", 1.0)))
            for key, node in nodes:
                self._dispose_node(node, key)
            self._load_error = None
            return {"status": "configured"}

        if action == "interrupt":
            with self._nodes_lock:
                if instance_id:
                    targets = [self._nodes[instance_id]] if instance_id in self._nodes else []
                else:
                    targets = [node for node in self._nodes.values() if node.state == "running"]
            cleared = 0
            for node in targets:
                result = node.interrupt()
                cleared += result.get("cleared", 0)
            return {"status": "interrupted", "nodes": len(targets), "cleared": cleared}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        return self._ensure_adapter().synthesize(text)
