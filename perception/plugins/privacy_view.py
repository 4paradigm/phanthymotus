"""Multi-instance privacy processor. A failed frame has no image output.

One node and one latest-frame worker per instance. Model loading and inference
run off the DDS callback; the only model handle is a lease on the ORT child.
The publication gate linearizes stop/config with image publication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import asdict

from plugins.face_proxy import FaceServiceProxy
from plugins.face_runtime import DEFAULT_FACE_MODEL, DEFAULT_FACE_MODEL_DIR
from plugins.privacy_runtime import (
    MAX_JPEG_BYTES,
    PrivacyConfig,
    PrivacyError,
    decode_jpeg,
    encode_jpeg,
    redact,
    validate_reply,
)
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from utils.latest_frame import LatestFrame
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node

_CONFIG = {
    "mode": {
        "type": "string",
        "enum": ["opaque", "pixelate", "gaussian"],
        "title": "遮挡模式",
        "description": "opaque 完全遮挡；马赛克和模糊的隐私强度较低",
    },
    "detect_fps": {
        "type": "number",
        "minimum": 0.1,
        "maximum": 60,
        "title": "处理帧率上限",
    },
    "confidence_threshold": {
        "type": "number",
        "minimum": 0.1,
        "maximum": 0.9,
        "title": "检测置信度",
        "description": "调高会增加漏检风险",
    },
    "bbox_padding": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "title": "每侧人脸框扩张比例",
    },
    "pixelate_strength": {
        "type": "integer",
        "minimum": 4,
        "maximum": 128,
        "title": "马赛克降采样倍数",
    },
    "gaussian_strength": {
        "type": "integer",
        "minimum": 3,
        "maximum": 301,
        "title": "高斯核尺寸（奇数）",
    },
    "inference_timeout_s": {
        "type": "number",
        "minimum": 0.05,
        "maximum": 10,
        "title": "推理等待上限（秒）",
    },
    "max_frame_age_s": {
        "type": "number",
        "minimum": 0.1,
        "maximum": 10,
        "title": "源帧最大年龄（秒）",
    },
}
for _key, _value in asdict(PrivacyConfig()).items():
    _CONFIG[_key].update(default=_value, scope="instance")

TOOLS = [
    {
        "name": "privacy_view",
        "type": "processor",
        "multiInstance": True,
        "description": (
            "实时人脸隐私遮挡。复用 SCRFD，输出 JPEG 和 JSON 状态。"
            "故障不发布图像。零检测不保证无人脸；模型可能漏检。"
        ),
        "inputSchema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 JPEG camera topic",
                },
            },
            "x-action-params": {
                "start": {"params": ["input_topic"], "description": "开始处理摄像头"},
                "stop": {"params": [], "description": "取消并释放本实例"},
                "info": {
                    "params": ["input_topic"],
                    "description": "状态、端口、配置及指标",
                },
                "config": {"params": [], "description": "校验配置并丢弃旧代次任务"},
            },
        },
        "configSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": _CONFIG,
        },
        "topic_in": [{"format": "image/jpeg", "desc": "Camera JPEG"}],
        "topic_out": [
            {"format": "image/jpeg", "desc": "Privacy JPEG"},
            {"format": "data/json", "desc": "Privacy status"},
        ],
    }
]


def _topic(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*", value
    ):
        raise PrivacyError("input_topic must be an absolute ROS2 topic")
    if len(value) > 200:
        raise PrivacyError("input_topic is too long")
    return value


def output_topics(topic: str, key: str):
    # Including the instance identity also isolates two cards on one camera.
    suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
    root = topic + "/privacy_view/instance_" + suffix
    return [
        {"topic": root + "/image", "format": "image/jpeg"},
        {"topic": root + "/status", "format": "data/json"},
    ]


def _stamp_ns(header):
    try:
        sec, nano = header.stamp.sec, header.stamp.nanosec
        if (
            type(sec) is not int
            or type(nano) is not int
            or sec < 0
            or not 0 <= nano < 10**9
        ):
            raise ValueError
        value = sec * 10**9 + nano
        if (
            value <= 0
            or not isinstance(header.frame_id, str)
            or len(header.frame_id) > 256
        ):
            raise ValueError
        return value
    except (AttributeError, ValueError, TypeError) as exc:
        raise PrivacyError("invalid_source_header") from exc


class _PrivacyNode(Node):
    def __init__(self, key, topic, cfg, factory, finished):
        self.key, self.input_topic = key, topic
        self.topics = output_topics(topic, key)
        super().__init__("privacy_view_" + uuid.uuid4().hex[:16])
        self._gate = threading.RLock()
        self._stop = threading.Event()
        self.done = threading.Event()
        self._queue = LatestFrame()
        self._cfg, self._version = cfg, 0
        self._generation = uuid.uuid4().hex
        self._factory, self._finished = factory, finished
        self._last_stamp = 0
        self._seq = 0
        self._last_received = None
        self._started = time.monotonic()
        self._state, self._error = "loading", None
        self._model_ready = False
        self._load_progress = "Preparing SCRFD in the ORT worker"
        self._last_output = None
        self._last_output_stamp = None
        self._metrics = {
            "received": 0,
            "processed": 0,
            "published": 0,
            "errors": 0,
            "rejected": 0,
            "discarded_generation": 0,
        }
        self._latency = {}
        self._thread = None
        self._disposing = False
        try:
            self._pub = self.create_publisher(
                CompressedImage, self.topics[0]["topic"], CAMERA_QOS
            )
            self._status_pub = self.create_publisher(
                String, self.topics[1]["topic"], CAMERA_QOS
            )
            self._sub = self.create_subscription(
                CompressedImage, topic, self._receive, CAMERA_QOS
            )
            self._timer = self.create_timer(0.5, self._tick)
        except Exception:
            self.destroy_node()
            raise

    def start(self):
        with self._gate:
            if self._stop.is_set() or self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name=self.get_name(), daemon=True
            )
            self._thread.start()

    def request_stop(self):
        self._stop.set()  # cancel before waiting for any lock
        self._queue.close()
        with self._gate:
            self._version += 1
            self._state, self._error = "stopping", "stopped"

    def finish(self):
        with self._gate:
            if self._disposing:
                return
            self._disposing = True
        try:
            self._finished(self)
        finally:
            self.done.set()

    def apply_config(self, cfg):
        with self._gate:
            if self._stop.is_set():
                return self.snapshot()
            if self._cfg != cfg:
                self._cfg = cfg
                self._version += 1
                # queued frames carry their configuration generation too
                self._state, self._error = (
                    "loading" if not self._model_ready else "running",
                    "awaiting_new_frame",
                )
            return self.snapshot()

    def snapshot(self):
        with self._gate:
            now = time.monotonic()
            age = now - (
                self._last_received
                if self._last_received is not None
                else self._started
            )
            stale = age > self._cfg.max_frame_age_s
            source_age = (
                (time.time_ns() - self._last_stamp) / 10**9
                if self._last_stamp
                else None
            )
            output_age = (
                (time.time_ns() - self._last_output_stamp) / 10**9
                if self._last_output_stamp is not None
                else None
            )
            state, error = self._state, self._error
            if self._stop.is_set():
                state, error = "stopping", "stopped"
            elif stale and self._model_ready:
                state, error = "error", "camera_timeout"
            return {
                "state": state,
                "error": error,
                "instance_id": self.key,
                "generation": self._generation,
                "processing_generation": self._version,
                "config": asdict(self._cfg),
                "model_ready": self._model_ready,
                "desc": self._load_progress
                if not self._model_ready
                else "SCRFD privacy processing",
                "receive_age_s": age,
                "source_age_s": source_age,
                "output_age_s": output_age,
                "fresh": output_age is not None
                and -0.1 <= output_age <= self._cfg.max_frame_age_s
                and not stale
                and error is None,
                "metrics": {**self._metrics, "queue_overwrites": self._queue.dropped},
                "latency_ms": dict(self._latency),
                "topic_in": [{"topic": self.input_topic, "format": "image/jpeg"}],
                "topic_out": copy.deepcopy(self.topics),
                "privacy_guaranteed": False,
                "detection_scope": "all post-NMS SCRFD detections; missed faces remain possible",
            }

    def _status(self, frame=None, **values):
        if self._stop.is_set():
            return
        payload = {
            "schema": "motus.privacy_view/1",
            **self.snapshot(),
            "image_published": False,
            **values,
        }
        if frame is not None:
            payload["source"] = {
                "stamp_ns": frame[2],
                "frame_id": frame[1].frame_id,
                "frame_id_available": bool(frame[1].frame_id),
                "sequence": frame[3],
                "frame_token": frame[5],
            }
        msg = String()
        msg.data = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        self._status_pub.publish(msg)

    def _tick(self):
        with self._gate:
            self._status()

    def _receive(self, message):
        with self._gate:
            if self._stop.is_set():
                return
            self._metrics["received"] += 1
            try:
                stamp = _stamp_ns(message.header)
                now_ns = time.time_ns()
                if (
                    stamp > now_ns + 100_000_000
                    or now_ns - stamp > self._cfg.max_frame_age_s * 10**9
                ):
                    raise PrivacyError("untrusted_source_timestamp")
                if stamp <= self._last_stamp:
                    raise PrivacyError("nonincreasing_source_timestamp")
                if not isinstance(
                    message.format, str
                ) or message.format.lower() not in (
                    "jpeg",
                    "jpg",
                    "image/jpeg",
                    "bgr8; jpeg compressed bgr8",
                    "rgb8; jpeg compressed bgr8",
                ):
                    raise PrivacyError("unsupported_image_format")
                if not 4 <= len(message.data) <= MAX_JPEG_BYTES:
                    raise PrivacyError("invalid_jpeg_size")
                self._last_stamp = stamp
                self._seq += 1
                self._last_received = time.monotonic()
                token = f"{self._generation}:{self._version}:{self._seq}"
                frame = (
                    bytes(message.data),
                    copy.deepcopy(message.header),
                    stamp,
                    self._seq,
                    self._last_received,
                    token,
                    self._version,
                )
                self._queue.push(frame)
            except (PrivacyError, AttributeError, TypeError, ValueError):
                self._metrics["rejected"] += 1
                self._version += 1
                self._queue.clear()
                self._state, self._error = "error", "invalid_source_frame"

    def _current(self, frame):
        return (
            not self._stop.is_set()
            and frame[6] == self._version
            and time.monotonic() - frame[4] <= self._cfg.max_frame_age_s
            and -100_000_000
            <= time.time_ns() - frame[2]
            <= self._cfg.max_frame_age_s * 10**9
        )

    def _loading_status(self, text):
        with self._gate:
            if not self._stop.is_set():
                self._load_progress = text
                self._status()

    def _run(self):
        detector = None
        try:
            if self._stop.is_set():
                return
            detector = self._factory(self._loading_status)
            with self._gate:
                if self._stop.is_set():
                    return
                self._model_ready = True
                self._state, self._error = "running", "awaiting_new_frame"
            next_frame = 0.0
            while not self._stop.is_set():
                if self._stop.wait(max(0, next_frame - time.monotonic())):
                    break
                frame = self._queue.pop(timeout=0.1)
                if frame is None:
                    continue
                with self._gate:
                    cfg = self._cfg
                    if not self._current(frame):
                        self._metrics["discarded_generation"] += 1
                        continue
                next_frame = time.monotonic() + 1.0 / cfg.detect_fps
                self._process(frame, cfg, detector)
        except Exception:  # noqa: BLE001 - fail closed on model or worker failure
            with self._gate:
                if not self._stop.is_set():
                    self._state, self._error = "error", "model_load_failed"
                    self._metrics["errors"] += 1
                    self._status()
                # Retain the error node for info/stop; no repeated model loads.
            self._stop.wait()
        finally:
            try:
                if detector is not None:
                    detector.close()
            finally:
                self._queue.close()
                self.finish()

    def _process(self, frame, cfg, detector):
        started = time.monotonic()
        try:
            image = decode_jpeg(frame[0])
            decoded = time.monotonic()
            reply = detector.detect(
                frame[0],
                frame[5],
                cfg.confidence_threshold,
                timeout=cfg.inference_timeout_s,
            )
            detected = time.monotonic()
            regions = validate_reply(reply, frame[0], frame[5], image.shape, cfg)
            output = redact(image, regions, cfg)
            redacted = time.monotonic()
            encoded = encode_jpeg(output)
            ended = time.monotonic()
            with self._gate:
                self._metrics["processed"] += 1
                if not self._current(frame):
                    self._metrics["discarded_generation"] += 1
                    if frame[6] == self._version and not self._stop.is_set():
                        self._state, self._error = "error", "frame_expired"
                        self._status(frame)
                    return
                if detected - decoded > cfg.inference_timeout_s:
                    raise PrivacyError("inference_timeout")
                msg = CompressedImage()
                msg.header = frame[1]
                msg.format, msg.data = "jpeg", encoded
                self._pub.publish(msg)
                self._metrics["published"] += 1
                self._state, self._error = "running", None
                self._last_output = ended
                self._last_output_stamp = frame[2]
                self._latency = {
                    "decode": (decoded - started) * 1000,
                    "detect": (detected - decoded) * 1000,
                    "redact": (redacted - detected) * 1000,
                    "encode": (ended - redacted) * 1000,
                    "processing": (ended - started) * 1000,
                    "receive_to_publish": (ended - frame[4]) * 1000,
                }
                self._status(
                    frame,
                    image_published=True,
                    faces_detected=len(regions),
                    detection_complete=True,
                    mode=cfg.mode,
                )
        except Exception as exc:  # noqa: BLE001 - no codec/model error may pass pixels
            with self._gate:
                if self._stop.is_set() or frame[6] != self._version:
                    return
                self._metrics["errors"] += 1
                self._state = "error"
                self._error = (
                    str(exc) if isinstance(exc, PrivacyError) else "processing_failed"
                )
                self._status(frame, detection_complete=False)


class PrivacyViewPlugin:
    PREFIX = "privacy_view"

    def __init__(self, cfg, executor):
        allowed = set(_CONFIG) | {
            "enabled",
            "model",
            "model_dir",
            "device",
            "num_threads",
        }
        if set(cfg) - allowed:
            raise PrivacyError("unknown privacy configuration field")
        threads = cfg.get("num_threads", 2)
        if type(threads) is not int or not 1 <= threads <= 16:
            raise PrivacyError("num_threads must be an integer in [1, 16]")
        self._executor = executor
        self._lock = threading.RLock()
        self._nodes = {}
        self._configs = {}
        self._closed = False
        self._defaults = PrivacyConfig().updated(
            {k: v for k, v in cfg.items() if k in _CONFIG}
        )
        self._model_options = {
            k: cfg[k]
            for k in ("model", "model_dir", "device", "num_threads")
            if k in cfg
        }
        self._model_options.setdefault("model", DEFAULT_FACE_MODEL)
        self._model_options.setdefault("model_dir", DEFAULT_FACE_MODEL_DIR)
        self._model_options.setdefault("device", "cpu")

    def get_tools(self):
        return copy.deepcopy(TOOLS)

    def _make_detector(self, on_status=None):
        return FaceServiceProxy(**self._model_options, on_status=on_status)

    def _finished(self, node):
        dispose_node(self._executor, node, label="privacy_view")
        with self._lock:
            if self._nodes.get(node.key) is node:
                del self._nodes[node.key]

    def dispatch(self, name, args):
        action = args.get("action", name)
        key = args.get("instance_id") or "_default"
        if not isinstance(key, str) or len(key) > 256:
            return {"state": "error", "error": "invalid_instance_id"}
        try:
            if action == "info":
                with self._lock:
                    node = self._nodes.get(key)
                    cfg = self._configs.get(key, self._defaults)
                    nodes = list(self._nodes.values())
                if node is not None:
                    return node.snapshot()
                topic = args.get("input_topic")
                return {
                    "state": "idle",
                    "config": asdict(cfg),
                    "topic_in": [{"topic": _topic(topic), "format": "image/jpeg"}]
                    if topic
                    else [],
                    "topic_out": output_topics(_topic(topic), key) if topic else [],
                    "instances": {n.key: n.snapshot() for n in nodes},
                }
            if action == "start":
                topic = args.get("input_topic")
                topics = args.get("input_topics")
                if topics is not None:
                    if not isinstance(topics, list) or len(topics) != 1:
                        raise PrivacyError("exactly one input camera is required")
                    if topic and topic != topics[0]:
                        raise PrivacyError("conflicting input topics")
                    topic = topics[0]
                topic = _topic(topic)
                with self._lock:
                    if self._closed:
                        raise PrivacyError("plugin is shut down")
                    node = self._nodes.get(key)
                    if node is not None and node.input_topic != topic:
                        raise PrivacyError(
                            "stop the instance before changing its camera"
                        )
                    if node is None:
                        if len(self._nodes) >= 32:
                            raise PrivacyError("instance capacity reached (32)")
                        node = _PrivacyNode(
                            key,
                            topic,
                            self._configs.get(key, self._defaults),
                            self._make_detector,
                            self._finished,
                        )
                        try:
                            self._executor.add_node(node)
                        except Exception:
                            node.destroy_node()
                            raise
                        self._nodes[key] = node
                node.start()
                # Stop may have won between registering the node and start().
                if node._stop.is_set() and node._thread is None:
                    node.finish()
                return node.snapshot()
            if action == "stop":
                with self._lock:
                    nodes = (
                        ([self._nodes[key]] if key in self._nodes else [])
                        if args.get("instance_id")
                        else list(self._nodes.values())
                    )
                    for node in nodes:
                        node.request_stop()
                    if args.get("instance_id"):
                        self._configs.pop(key, None)
                    else:
                        self._configs.clear()
                for node in nodes:
                    if node._thread is not None:
                        node._thread.join(timeout=0.25)
                return {
                    "state": "stopping"
                    if any(not n.done.is_set() for n in nodes)
                    else "idle",
                    "image_publication_disabled": True,
                }
            if action == "config":
                values = args.get(
                    "config",
                    {
                        k: v
                        for k, v in args.items()
                        if k not in ("action", "instance_id")
                    },
                )
                with self._lock:
                    if self._closed:
                        raise PrivacyError("plugin is shut down")
                    current = (
                        self._configs.get(key, self._defaults)
                        if args.get("instance_id")
                        else self._defaults
                    )
                    cfg = current.updated(values)
                    if args.get("instance_id"):
                        if key not in self._configs and len(self._configs) >= 32:
                            raise PrivacyError("configuration capacity reached (32)")
                        self._configs[key] = cfg
                        nodes = [self._nodes[key]] if key in self._nodes else []
                    else:
                        self._defaults = cfg
                        nodes = [
                            n for k, n in self._nodes.items() if k not in self._configs
                        ]
                    for node in nodes:
                        node.apply_config(cfg)
                return {"status": "configured", "config": asdict(cfg)}
        except (PrivacyError, TypeError, ValueError) as exc:
            return {"state": "error", "error": str(exc)}
        return None

    def shutdown(self):
        with self._lock:
            self._closed = True
            nodes = list(self._nodes.values())
        self.dispatch(self.PREFIX, {"action": "stop"})
        # A downloading model remains tracked and cancelled until it returns;
        # it can never publish, and finally releases its lease and node.
        for node in nodes:
            if node._thread is not None:
                node._thread.join(timeout=1.0)
