#!/usr/bin/env python3
"""
Non-speech sound event detection for Perception.
"""


from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from utils.ros_lifecycle import dispose_node


log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 15_600
HOP_SAMPLES = 7_680
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
HOP_SECONDS = HOP_SAMPLES / SAMPLE_RATE
WINDOW_BYTES = WINDOW_SAMPLES * 2
HOP_BYTES = HOP_SAMPLES * 2

AUDIO_FORMAT = "audio/pcm-16k"
_AUDIO_FORMATS = frozenset((AUDIO_FORMAT, "pcm_16k_16bit_mono"))
THRESHOLD = 0.75
NUM_THREADS = 1
QUEUE_DEPTH = 32
INPUT_TIMEOUT_SECONDS = 2.0
PARENT_SUPPRESSION_RATIO = 0.70

_AUDIO_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)

_RESULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


TOOLS = [
    {
        "name": "soundevent",
        "type": "processor",
        "multiInstance": True,
        "description": "Sound Event Detection — detect non-speech sounds from audio",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform",
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 AudioChunk topic (required for start)",
                },
            },
            "required": ["action"],
        },
        "configSchema": {"type": "object", "properties": {}},
        "topic_in": [{"format": AUDIO_FORMAT, "desc": "16 kHz mono PCM16 audio"}],
        "topic_out": [{"format": "data/json", "desc": "detected sound event"}],
    }
]


_SPEECH_MARKERS = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "babbling",
    "whispering",
)

_NON_EVENT_LABELS = frozenset(
    (
        "Silence",
        "Inside, small room",
        "Inside, large room or hall",
        "Outside, urban or manmade",
        "Outside, rural or natural",
        "Environmental noise",
    )
)

# YAMNet scores ontology parents together with useful leaf labels. Suppress a
# generic parent when a sufficiently strong child describes the same sound.
_PARENTS = {
    "Dog": ("Canidae, dogs, wolves", "Domestic animals, pets", "Animal"),
    "Bark": ("Dog", "Canidae, dogs, wolves", "Domestic animals, pets", "Animal"),
    "Bow-wow": ("Bark", "Dog", "Canidae, dogs, wolves", "Domestic animals, pets", "Animal"),
    "Yip": ("Bark", "Dog", "Canidae, dogs, wolves", "Domestic animals, pets", "Animal"),
    "Howl": ("Dog", "Canidae, dogs, wolves", "Domestic animals, pets", "Animal"),
    "Growling": ("Dog", "Canidae, dogs, wolves", "Domestic animals, pets", "Animal"),
    "Whimper (dog)": (
        "Whimper",
        "Dog",
        "Canidae, dogs, wolves",
        "Domestic animals, pets",
        "Animal",
    ),
    "Bird": ("Wild animals", "Animal"),
    "Bird vocalization, bird call, bird song": ("Bird", "Wild animals", "Animal"),
    "Chirp, tweet": ("Bird vocalization, bird call, bird song", "Bird", "Wild animals", "Animal"),
    "Pigeon, dove": ("Bird", "Wild animals", "Animal"),
    "Coo": (
        "Bird vocalization, bird call, bird song",
        "Pigeon, dove",
        "Bird",
        "Wild animals",
        "Animal",
    ),
    "Bird flight, flapping wings": ("Bird", "Wild animals", "Animal"),
    "Glass breaking": ("Glass",),
    "Shatter": ("Glass",),
    "Fire alarm": ("Alarm",),
    "Smoke detector, smoke alarm": ("Alarm",),
    "Alarm clock": ("Alarm",),
}


def _safe_suffix(value: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_")
    return (suffix or "default")[:80]


def _is_event_label(label: str) -> bool:
    lowered = label.casefold()
    return label not in _NON_EVENT_LABELS and not any(
        marker in lowered for marker in _SPEECH_MARKERS
    )


def labels_from_model(model_path: Path) -> List[str]:
    """Read the label list embedded in Google's metadata-bearing TFLite file."""
    try:
        with zipfile.ZipFile(str(model_path)) as archive:
            with archive.open("yamnet_label_list.txt") as label_file:
                labels = [
                    line.decode("utf-8").strip()
                    for line in label_file
                    if line.strip()
                ]
    except (KeyError, zipfile.BadZipFile) as error:
        raise RuntimeError("YAMNet model has no embedded label list") from error
    if len(labels) != 521:
        raise RuntimeError("unexpected YAMNet label count: %d" % len(labels))
    return labels


class YamNetLite:
    """Locked wrapper around the TFLite interpreter shared by all instances."""

    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError("YAMNet model not found: %s" % path)
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as error:
            raise RuntimeError("tflite-runtime is not installed") from error

        self.labels = labels_from_model(path)
        self._interpreter = Interpreter(model_path=str(path), num_threads=NUM_THREADS)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self._lock = threading.Lock()

    def predict(self, waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.shape != (WINDOW_SAMPLES,):
            raise ValueError("YAMNet input must contain %d samples" % WINDOW_SAMPLES)
        with self._lock:
            self._interpreter.set_tensor(
                self._input["index"], np.ascontiguousarray(waveform)
            )
            self._interpreter.invoke()
            scores = self._interpreter.get_tensor(self._output["index"])
        result = np.asarray(scores, dtype=np.float32).reshape(-1)
        if result.shape != (len(self.labels),):
            raise RuntimeError("unexpected YAMNet output shape: %r" % (result.shape,))
        return result


def _build_model() -> YamNetLite:
    from utils.model_downloader import ensure_soundevent_model

    return YamNetLite(ensure_soundevent_model())


class _Detector:
    """Return at most one useful non-speech event from a score vector."""

    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)
        self._allowed = np.asarray(
            [_is_event_label(label) for label in self.labels], dtype=bool
        )
        self._index = {label: index for index, label in enumerate(self.labels)}

    def detect(self, scores: np.ndarray) -> List[Dict[str, Any]]:
        row = np.asarray(scores, dtype=np.float32).reshape(-1)
        if row.shape != (len(self.labels),):
            raise ValueError("score shape does not match label count")
        row = np.clip(
            np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )

        suppressed = set()
        for child, parents in _PARENTS.items():
            child_index = self._index.get(child)
            if child_index is None:
                continue
            child_score = float(row[child_index])
            for parent in parents:
                parent_index = self._index.get(parent)
                if (
                    parent_index is not None
                    and child_score >= float(row[parent_index]) * PARENT_SUPPRESSION_RATIO
                ):
                    suppressed.add(parent_index)

        candidates = np.flatnonzero(self._allowed & (row >= THRESHOLD))
        candidates = [int(index) for index in candidates if int(index) not in suppressed]
        if not candidates:
            return []
        index = max(candidates, key=lambda item: (float(row[item]), -item))
        return [{"name": self.labels[index], "confidence": round(float(row[index]), 6)}]


class _SoundEventNode(Node):
    """Subscribe to one audio topic and run inference outside the ROS callback."""

    def __init__(self, input_topic: str, model: YamNetLite, node_suffix: str):
        super().__init__("soundevent_%s" % _safe_suffix(node_suffix))
        self.input_topic = input_topic
        self.output_topic = "%s/soundevent" % input_topic
        self.state = "idle"

        self._model = model
        self._detector = _Detector(model.labels)
        self._publisher = self.create_publisher(String, self.output_topic, _RESULT_QOS)
        self._subscription = None
        self._queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self._thread = None  # type: Optional[threading.Thread]
        self._stop_event = threading.Event()
        self._retired = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._stream_lock = threading.Lock()
        self._stream_generation = 0
        self._stats_lock = threading.Lock()
        self._statistics = {
            "chunks_received": 0,
            "windows_processed": 0,
            "dropped_chunks": 0,
            "inference_errors": 0,
        }

    def start(self) -> Dict[str, Any]:
        with self._lifecycle_lock:
            if self._retired.is_set():
                return {"state": "idle"}
            if self.state == "running":
                return self.status()

            from audio_msgs.msg import AudioChunk

            with self._stream_lock:
                self._stop_event.clear()
                self._stream_generation += 1
                self._queue = queue.Queue(maxsize=QUEUE_DEPTH)
            self._subscription = self.create_subscription(
                AudioChunk, self.input_topic, self._audio_callback, _AUDIO_QOS
            )
            self._thread = threading.Thread(
                target=self._worker,
                name="soundevent_%s" % _safe_suffix(self.input_topic),
                daemon=True,
            )
            self.state = "running"
            self._thread.start()
            log.info("[soundevent] started: %s -> %s", self.input_topic, self.output_topic)
            return self.status()

    def request_retire(self) -> None:
        self._retired.set()
        self._stop_event.set()

    def retire(self) -> Dict[str, Any]:
        self.request_retire()
        return self.stop()

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        with self._lifecycle_lock:
            if self.state == "idle" and self._thread is None:
                return {"state": "idle"}
            with self._stream_lock:
                self._stop_event.set()
                self._stream_generation += 1
            if self._subscription is not None:
                self.destroy_subscription(self._subscription)
                self._subscription = None
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(None)

            thread = self._thread
            if thread is not None:
                thread.join(timeout=3.0)
            if thread is not None and thread.is_alive():
                self.state = "error"
                return {"state": "error", "message": "SoundEvent worker did not stop"}

            self._thread = None
            self.state = "idle"
            log.info("[soundevent] stopped: %s", self.input_topic)
            return {"state": "idle"}

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._statistics[name] += amount

    @staticmethod
    def _timestamp(message: Any) -> float:
        try:
            stamp = message.header.stamp
            timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except (AttributeError, TypeError, ValueError):
            timestamp = 0.0
        return timestamp if math.isfinite(timestamp) and timestamp > 0.0 else time.time()

    def _audio_callback(self, message: Any) -> None:
        if self.state != "running" or self._stop_event.is_set():
            return
        self._increment("chunks_received")
        if getattr(message, "format", "") not in _AUDIO_FORMATS:
            return
        try:
            pcm = bytes(message.data)
        except (TypeError, ValueError):
            return
        if not pcm or len(pcm) % 2:
            return

        dropped = 0
        with self._stream_lock:
            if self.state != "running" or self._stop_event.is_set():
                return
            packet = (self._stream_generation, pcm, self._timestamp(message))
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                while True:
                    try:
                        self._queue.get_nowait()
                        dropped += 1
                    except queue.Empty:
                        break
                self._stream_generation += 1
                packet = (self._stream_generation, pcm, packet[2])
                self._queue.put_nowait(packet)
        if dropped:
            self._increment("dropped_chunks", dropped)

    def _publish(
        self, events: Sequence[Mapping[str, Any]], timestamp: float, generation: int
    ) -> bool:
        with self._stream_lock:
            if self._stop_event.is_set() or generation != self._stream_generation:
                return False
            payload = {"timestamp": round(float(timestamp), 6), "events": list(events)}
            message = String()
            message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self._publisher.publish(message)
            return True

    def _worker(self) -> None:
        buffer = bytearray()
        buffer_start = None  # type: Optional[float]
        buffer_generation = None  # type: Optional[int]
        last_input = None  # type: Optional[float]
        try:
            while not self._stop_event.is_set():
                try:
                    packet = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if (
                        last_input is not None
                        and time.monotonic() - last_input >= INPUT_TIMEOUT_SECONDS
                    ):
                        buffer.clear()
                        buffer_start = None
                        last_input = None
                    continue
                if packet is None:
                    break

                generation, pcm, timestamp = packet
                with self._stream_lock:
                    current = generation == self._stream_generation
                if not current:
                    buffer.clear()
                    buffer_start = None
                    buffer_generation = None
                    last_input = None
                    continue
                if generation != buffer_generation:
                    buffer.clear()
                    buffer_start = None
                    last_input = None
                    buffer_generation = generation
                if buffer_start is None:
                    buffer_start = timestamp
                buffer.extend(pcm)
                last_input = time.monotonic()

                while len(buffer) >= WINDOW_BYTES and not self._stop_event.is_set():
                    waveform = np.frombuffer(
                        bytes(buffer[:WINDOW_BYTES]), dtype="<i2"
                    ).astype(np.float32)
                    waveform *= 1.0 / 32768.0
                    window_end = float(buffer_start) + WINDOW_SECONDS
                    try:
                        events = self._detector.detect(self._model.predict(waveform))
                    except Exception:
                        self._increment("inference_errors")
                        raise
                    if events:
                        current = self._publish(events, window_end, generation)
                    else:
                        with self._stream_lock:
                            current = (
                                not self._stop_event.is_set()
                                and generation == self._stream_generation
                            )
                    if not current:
                        buffer.clear()
                        buffer_start = None
                        buffer_generation = None
                        last_input = None
                        break
                    self._increment("windows_processed")
                    del buffer[:HOP_BYTES]
                    buffer_start += HOP_SECONDS
        except Exception:
            self.state = "error"
            log.exception("[soundevent] worker failed")

    def status(self) -> Dict[str, Any]:
        with self._stats_lock:
            statistics = dict(self._statistics)
        return {
            "state": self.state,
            "topic_in": [
                {"topic": self.input_topic, "format": AUDIO_FORMAT, "desc": ""}
            ],
            "topic_out": [
                {"topic": self.output_topic, "format": "data/json", "desc": ""}
            ],
            "statistics": statistics,
        }


class SoundEventPlugin:
    """Perception facade around one shared YAMNet model."""

    PREFIX = "soundevent"
    _DESC = "On-device non-speech sound event detection with YAMNet"

    def __init__(self, plugin_cfg: Mapping[str, Any], executor: Any):
        self._executor = executor
        self._nodes = {}  # type: Dict[str, _SoundEventNode]
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._model = None  # type: Optional[YamNetLite]
        self._model_error = None  # type: Optional[str]
        try:
            self._model = _build_model()
        except Exception as error:
            self._model_error = str(error)
            log.exception("[soundevent] model load failed")

    def get_tools(self) -> List[Dict[str, Any]]:
        return TOOLS

    @staticmethod
    def _topics(input_topic: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if not input_topic:
            return [], []
        return (
            [{"topic": input_topic, "format": AUDIO_FORMAT, "desc": ""}],
            [
                {
                    "topic": "%s/soundevent" % input_topic,
                    "format": "data/json",
                    "desc": "",
                }
            ],
        )

    def _info(self, key: str, input_topic: str) -> Dict[str, Any]:
        with self._lock:
            if key:
                nodes = [self._nodes[key]] if key in self._nodes else []
            else:
                nodes = list(self._nodes.values())
            model_error = self._model_error

        if len(nodes) == 1:
            result = nodes[0].status()
        elif nodes:
            statuses = [node.status() for node in nodes]
            states = {status["state"] for status in statuses}
            if "error" in states:
                state = "error"
            elif "running" in states:
                state = "running"
            else:
                state = "idle"
            result = {
                "state": state,
                "topic_in": [item for status in statuses for item in status["topic_in"]],
                "topic_out": [item for status in statuses for item in status["topic_out"]],
            }
        else:
            topic_in, topic_out = self._topics(input_topic)
            result = {
                "state": "error" if model_error else "idle",
                "topic_in": topic_in,
                "topic_out": topic_out,
            }

        result.update(
            {
                "name": "SoundEvent",
                "manufacture": "Embodied",
                "model": "yamnet",
                "desc": model_error or self._DESC,
            }
        )
        return result

    def _dispose(self, key: str, node: _SoundEventNode) -> Dict[str, Any]:
        try:
            return node.retire()
        finally:
            dispose_node(self._executor, node, label="soundevent/%s" % key)

    def _start(self, key: str, input_topic: str) -> Dict[str, Any]:
        if not input_topic:
            raise ValueError("input_topic is required for start")
        if self._model is None:
            return {"state": "error", "message": self._model_error or "model unavailable"}

        with self._lifecycle_lock:
            with self._lock:
                node = self._nodes.get(key)
                reuse = (
                    node is not None
                    and node.input_topic == input_topic
                    and node.state != "error"
                )
                previous = None if reuse else self._nodes.pop(key, None)
                if previous is not None:
                    previous.request_retire()
            if previous is not None:
                result = self._dispose(key, previous)
                if result.get("state") != "idle":
                    return result

            if not reuse:
                node = _SoundEventNode(input_topic, self._model, node_suffix=key)
                with self._lock:
                    try:
                        self._executor.add_node(node)
                    except Exception:
                        node.destroy_node()
                        raise
                    self._nodes[key] = node

            try:
                return node.start()
            except Exception:
                with self._lock:
                    if self._nodes.get(key) is node:
                        del self._nodes[key]
                node.request_retire()
                try:
                    self._dispose(key, node)
                except Exception:
                    log.exception("[soundevent] failed to clean up start error")
                raise

    def _stop(self, instance_id: str) -> Dict[str, Any]:
        # Signal before waiting for another lifecycle transition. A stop can
        # therefore cancel a node that has been registered but not started yet.
        with self._lock:
            if instance_id:
                node = self._nodes.get(instance_id)
                pending = [node] if node is not None else []
            else:
                pending = list(self._nodes.values())
            for node in pending:
                node.request_retire()

        with self._lifecycle_lock:
            with self._lock:
                if instance_id:
                    node = self._nodes.pop(instance_id, None)
                    items = [(instance_id, node)] if node is not None else []
                else:
                    items = list(self._nodes.items())
                    self._nodes.clear()
                for _, node in items:
                    node.request_retire()

            failure = None
            for key, node in items:
                try:
                    result = self._dispose(key, node)
                except Exception as error:
                    log.exception("[soundevent] failed to dispose %s", key)
                    result = {"state": "error", "message": str(error)}
                if result.get("state") != "idle" and failure is None:
                    failure = result
            return failure or {"state": "idle"}

    def dispatch(self, name: str, args: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        action = str(args.get("action", name))
        instance_id = str(args.get("instance_id", "") or "")
        input_topic = str(args.get("input_topic", "") or "")
        key = instance_id or input_topic

        if action == "info":
            return self._info(key, input_topic)
        if action == "start":
            return self._start(key, input_topic)
        if action == "stop":
            return self._stop(instance_id)
        if action == "config":
            return {"status": "configured"}
        return None


__all__ = ["SoundEventPlugin", "TOOLS"]
