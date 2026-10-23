"""Adapt an existing hardware depth stream to visual_depth's consumer contract.

The Tianyi input is Orbbec camera_depth, NOT an RGB image or a laser scan.
No model, scale fit, image crop, hole filling, or RGB registration is performed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
import zlib

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.camera_info import for_topic, inherit
from plugins.visual_depth import sample_region, summarize_depth
from utils.ros_lifecycle import dispose_node

WIDTH, HEIGHT = 640, 480
RAW_SIZE = WIDTH * HEIGHT * 2
FORMAT = "image/depth-zlib"
ROS_FORMAT = "16UC1; compressedDepth zlib"
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1,
                 durability=DurabilityPolicy.VOLATILE)
DEFAULTS = {"fps": 8.0, "max_frame_age_ms": 500.0,
            "rgb_alignment_verified": False}
REGIONS = ("center", "left", "right", "full")


def output_topics_for(topic):
    # Keep a card-specific namespace, AND the suffixes older navi understands.
    # Writing back to the driver's topic or RGB visual_depth would collide.
    root = topic.rstrip("/") + "/lidar_depth"
    return root + "/visual_depth", root + "/visual_depth_summary"


def decode_hardware_depth(payload, fmt):
    """Bounded decode of the fixed uint16-mm contract; 0 stays unknown."""
    if fmt != ROS_FORMAT:
        raise ValueError(f"expected {ROS_FORMAT!r}, got {fmt!r}; JPEG/PNG is not supported")
    if not payload or len(payload) > RAW_SIZE + 4096:
        raise ValueError("invalid compressed depth payload size")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, RAW_SIZE + 1)
    except zlib.error as error:
        raise ValueError(f"invalid zlib depth: {error}") from error
    if (len(raw) != RAW_SIZE or not decoder.eof or decoder.unconsumed_tail
            or decoder.unused_data):
        raise ValueError("expected one complete 640x480 little-endian uint16 depth frame")
    mm = np.frombuffer(raw, dtype="<u2").reshape(HEIGHT, WIDTH)
    metres = mm.astype(np.float32) / 1000.0
    metres[mm == 0] = np.nan
    return metres


def _settings(values):
    cfg = {**DEFAULTS, **{k: v for k, v in values.items() if k in DEFAULTS}}
    for name, low, high in (("fps", 1, 30), ("max_frame_age_ms", 50, 5000)):
        if isinstance(cfg[name], bool):
            raise ValueError(f"{name} must be numeric, not boolean")
        value = float(cfg[name])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        cfg[name] = value
    if not isinstance(cfg["rgb_alignment_verified"], bool):
        raise ValueError("rgb_alignment_verified must be a boolean")
    return cfg


def _declared_format(bindings, topic):
    if isinstance(bindings, dict):
        item = bindings.get(topic)
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(item.get("format") or "")
        bindings = bindings.get("bindings") or bindings.get("inputs") or []
    for item in bindings or []:
        if isinstance(item, dict) and item.get("topic") == topic:
            return str(item.get("format") or "")
    return ""


def _select_input(args):
    topics = list(dict.fromkeys(args.get("input_topics") or []))
    topic = args.get("input_topic")
    if topic:
        topics = [topic]
    elif len(topics) > 1:
        topics = [t for t in topics
                  if _declared_format(args.get("input_bindings"), t) == FORMAT]
    if len(topics) != 1:
        raise ValueError("请只连接一路 camera_depth 的 image/depth-zlib 硬件深度输出")
    topic = topics[0]
    if not isinstance(topic, str) or not topic.startswith("/") or topic.endswith("/"):
        raise ValueError("input_topic must be an absolute ROS topic without a trailing slash")
    if any(c.isspace() for c in topic) or any(c in topic for c in "~{}"):
        raise ValueError("invalid input_topic")
    fmt = _declared_format(args.get("input_bindings"), topic)
    if fmt and fmt != FORMAT:
        raise ValueError(f"需要 {FORMAT}，收到 {fmt}；不要连接 RGB、点云或激光扫描")
    if topic.endswith(("/visual_depth", "/visual_depth_summary")):
        raise ValueError("需要原始硬件深度，不接受本卡或 visual_depth 的输出回接")
    return topic


class _HardwareDepthNode(Node):
    def __init__(self, key, topic, cfg, declaration):
        suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        super().__init__("lidar_depth_" + suffix)
        self.topic, self.cfg = topic, cfg
        self.depth_topic, self.summary_topic = output_topics_for(topic)
        self.declaration = declaration
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._active = False
        self._pending = None
        self._worker = None
        self._latest = None
        self._latest_at = None
        self._error = ""
        self.counts = {k: 0 for k in ("received", "published", "rejected", "dropped")}
        try:
            self._depth_pub = self.create_publisher(CompressedImage, self.depth_topic, QOS)
            self._summary_pub = self.create_publisher(String, self.summary_topic, QOS)
            self.create_subscription(CompressedImage, topic, self._on_frame, QOS)
        except Exception:
            self.destroy_node()
            raise

    def start(self):
        with self._lock:
            # A retired node is never resurrected by a delayed concurrent start.
            if self._stop.is_set():
                return {"state": "idle"}
            if not self._active:
                self._active = True
                self._worker = threading.Thread(target=self._run, daemon=True,
                                                name=self.get_name())
                self._worker.start()
        return self.info()

    def request_stop(self):
        self._stop.set()
        self._wake.set()

    def stop(self):
        self.request_stop()
        with self._lock:
            self._active = False
            self._pending = None
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2)
        return {"state": "idle"}

    def _on_frame(self, msg):
        with self._lock:
            if not self._active or self._stop.is_set():
                return
            self.counts["received"] += 1
            if len(msg.data) > RAW_SIZE + 4096:
                self.counts["rejected"] += 1
                self._error = "compressed hardware depth exceeds payload limit"
                return
            if self._pending is not None:
                self.counts["dropped"] += 1
            self._pending = (bytes(msg.data), msg.format, copy.deepcopy(msg.header),
                             time.monotonic(), time.time())
            self._wake.set()

    def _run(self):
        next_at = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=0.1)
            if self._stop.wait(max(0.0, next_at - time.monotonic())):
                break
            with self._lock:
                pending, self._pending = self._pending, None
                self._wake.clear()
            if pending is None:
                continue
            next_at = time.monotonic() + 1.0 / self.cfg["fps"]
            try:
                self._process(pending)
            except Exception as error:
                with self._lock:
                    self.counts["rejected"] += 1
                    self._error = str(error)

    def _process(self, pending):
        payload, fmt, header, received_at, received_wall = pending
        limit = self.cfg["max_frame_age_ms"] / 1000
        stamp = getattr(header, "stamp", None)
        sec = getattr(stamp, "sec", 0)
        nanosec = getattr(stamp, "nanosec", 0)
        capture = float(sec) + float(nanosec) / 1e9
        # Zero/uptime stamps cannot be compared to Unix time. Preserve them,
        # report receipt-based timing, and never manufacture a capture stamp.
        unix_stamp = sec >= 1_000_000_000
        age = received_wall - capture if unix_stamp else 0.0
        elapsed = time.monotonic() - received_at
        if elapsed > limit or age + elapsed > limit or age < -1.0:
            with self._lock:
                self.counts["dropped"] += 1
                self._error = "stale or future hardware depth frame"
            return
        metres = decode_hardware_depth(payload, fmt)
        summary = summarize_depth(metres)
        summary.update({"source": "hardware_depth", "input_topic": self.topic,
                        "timestamp": received_wall,
                        "stamp_source": "source" if unix_stamp else "received",
                        "source_stamp": {"sec": sec, "nanosec": nanosec},
                        "frame_id": getattr(header, "frame_id", ""),
                        "rgb_alignment_verified": self.cfg["rgb_alignment_verified"]})
        out = CompressedImage()
        out.header, out.format, out.data = header, ROS_FORMAT, payload
        summary_msg = String()
        summary_msg.data = json.dumps(summary, allow_nan=False)
        with self._lock:
            if not self._active or self._stop.is_set():
                return
            elapsed = time.monotonic() - received_at
            if elapsed > limit or age + elapsed > limit:
                self.counts["dropped"] += 1
                self._error = "hardware depth expired during processing"
                return
            self._depth_pub.publish(out)
            self._summary_pub.publish(summary_msg)
            self._latest = (metres, summary)
            self._latest_at = received_at - max(0.0, age)
            self.counts["published"] += 1
            self._error = ""

    def info(self):
        with self._lock:
            age = ((time.monotonic() - self._latest_at) * 1000
                   if self._latest_at is not None else None)
            fresh = age is not None and age <= self.cfg["max_frame_age_ms"]
            available = self._active and fresh and self._latest[1]["valid_fraction"] > 0
            result = {"state": "running" if self._active else "idle",
                      "input": self.topic, "topic_out": self.ports(),
                      "available": bool(available),
                      "data_state": "ready" if available else (
                          "waiting" if age is None else "stale" if not fresh else "invalid"),
                      "frame_age_ms": round(age, 1) if age is not None else None,
                      "counts": dict(self.counts), "error": self._error,
                      "config": dict(self.cfg), "calibration": "hardware:unchanged"}
            camera = []
            for port in self.ports():
                camera.extend(inherit(self.declaration, topic=port["topic"],
                                      fmt=port["format"], stage="perception/lidar_depth"))
            if camera:
                result["camera_info"] = camera
            result["alignment_note"] = (
                "RGB/depth alignment verified by operator; this card does not register images"
                if self.cfg["rgb_alignment_verified"] else
                "RGB/depth alignment unverified: check centre and off-centre targets before navigation")
            return result

    def ports(self):
        return [{"topic": self.depth_topic, "format": FORMAT},
                {"topic": self.summary_topic, "format": "data/json"}]

    def read(self, region):
        with self._lock:
            info = self.info()
            if not info["available"]:
                return {"ok": False, "reason": info["data_state"], "error": self._error}
            metres, summary = self._latest
            return {"ok": True, "region": region, "unit": "m",
                    **sample_region(metres, region), "summary": copy.deepcopy(summary)}


class LidarDepthPerceptionPlugin:
    PREFIX = "lidar_depth"

    def __init__(self, cfg, executor):
        self._cfg = _settings(cfg)
        self._executor = executor
        self._lock = threading.RLock()
        self._nodes = {}
        self._retiring = set()
        self._configs = {}
        self._closing = False

    def get_tools(self):
        properties = {
            "fps": {"type": "number", "title": "Output FPS", "default": 8,
                    "minimum": 1, "maximum": 30, "scope": "instance"},
            "max_frame_age_ms": {"type": "number", "title": "Maximum frame age (ms)",
                                 "default": 500, "minimum": 50, "maximum": 5000,
                                 "scope": "instance"},
            "rgb_alignment_verified": {"type": "boolean", "title": "RGB alignment verified",
                                       "default": False, "scope": "instance",
                                       "description": "Operator attestation only; does not align depth to RGB."}}
        return [{"name": self.PREFIX, "type": "processor", "multiInstance": True,
                 "description": "硬件深度感知：接收 camera_depth 的米制深度，输出与 visual_depth 一致的深度图和左中右摘要。天轶使用 Orbbec 深度相机；不运行单目模型，不接收激光扫描，不做 RGB 配准或尺度标定。",
                 "topic_in": [{"format": FORMAT, "desc": "相机硬件深度"}],
                 "topic_out": [{"format": FORMAT, "desc": "硬件深度图"},
                               {"format": "data/json", "desc": "左中右距离摘要"}],
                 "inputSchema": {"type": "object", "required": ["action"],
                                 "properties": {
                                     "action": {"type": "string", "enum": ["start", "stop", "info", "read", "config"]},
                                     "input_topic": {"type": "string", "description": "camera_depth 的 image/depth-zlib 话题"},
                                     "region": {"type": "string", "enum": list(REGIONS), "default": "center"},
                                     **copy.deepcopy(properties)},
                                 "x-action-params": {
                                     "start": {"params": ["input_topic"], "description": "接收硬件深度"},
                                     "stop": {"params": [], "description": "停止当前实例"},
                                     "info": {"params": [], "description": "查看输入、输出、帧龄、计数和对齐状态"},
                                     "read": {"params": ["region"], "description": "读取指定区域的新鲜硬件深度（米）"},
                                     "config": {"params": list(properties), "description": "配置当前实例；运行中修改后需重新 start"}}},
                 "configSchema": {"type": "object", "properties": properties}}]

    def _retire(self, key):
        with self._lock:
            if key in self._retiring:
                return
            node = self._nodes.get(key)
            if node is not None:
                self._retiring.add(key)
                node.request_stop()
        if node is not None:
            try:
                node.stop()
                dispose_node(self._executor, node, label="lidar_depth/" + key)
            finally:
                with self._lock:
                    self._nodes.pop(key, None)
                    self._retiring.discard(key)

    def dispatch(self, name, args):
        action = args.get("action", name)
        key = args.get("instance_id", "")
        try:
            if action == "start":
                topic = _select_input(args)
                key = key or topic
                declaration = for_topic(args.get("camera_info"), topic)
                if declaration and (declaration.get("width"), declaration.get("height")) != (WIDTH, HEIGHT):
                    raise ValueError("camera_info must describe the input's actual 640x480 depth geometry")
                with self._lock:
                    if self._closing:
                        raise ValueError("lidar_depth is shutting down")
                    node = self._nodes.get(key)
                    if key in self._retiring:
                        raise ValueError("previous instance is stopping; retry start after stop completes")
                    if node is not None and node.topic != topic:
                        raise ValueError("stop this instance before changing its input topic")
                    if node is None:
                        if any(n.topic == topic for n in self._nodes.values()):
                            raise ValueError("this hardware input already has a lidar_depth instance; reuse it")
                        cfg = _settings({**self._cfg, **self._configs.get(key, {})})
                        node = _HardwareDepthNode(key, topic, cfg, declaration)
                        try:
                            self._executor.add_node(node)
                        except Exception:
                            node.destroy_node()
                            raise
                        self._nodes[key] = node
                result = node.start()
                return {**result, "instance_id": key}
            if action == "stop":
                with self._lock:
                    keys = [key] if key else list(self._nodes)
                for item in keys:
                    self._retire(item)
                return {"state": "idle", "stopped_instances": keys}
            if action == "config":
                if any(k in args for k in ("cal_a", "cal_b", "calibration_preset", "depth_scale")):
                    raise ValueError("hardware depth is passed through unchanged; monocular calibration is not supported")
                supplied = {k: v for k, v in args.items() if k in DEFAULTS}
                with self._lock:
                    previous = self._configs.get(key, self._cfg) if key else self._cfg
                    cfg = _settings({**previous, **supplied})
                    if not key and self._nodes:
                        raise ValueError("pass instance_id when configuring a running card")
                    if key:
                        self._configs[key] = cfg
                        running = key in self._nodes
                    else:
                        self._cfg = cfg
                        running = False
                if running:
                    self._retire(key)
                return {"status": "configured", "config": cfg, "restart_required": running,
                        "persist": "Runtime only; save the canvas configuration or startup YAML to retain changes"}
            if action in ("info", "read"):
                with self._lock:
                    nodes = dict(self._nodes)
                if action == "read":
                    node = nodes.get(key) if key else (next(iter(nodes.values())) if len(nodes) == 1 else None)
                    if node is None:
                        raise ValueError("start one instance or pass its instance_id")
                    region = args.get("region") or "center"
                    if region not in REGIONS:
                        raise ValueError("unknown region")
                    return node.read(region)
                if key:
                    node = nodes.get(key)
                    return node.info() if node else {"state": "idle", "instance_id": key}
                instances = {k: n.info() for k, n in nodes.items()}
                return {"name": "LidarDepthPerception", "model": "hardware_depth_adapter",
                        "state": "running" if nodes else "idle", "scale": "metric", "unit": "m",
                        "instances": instances,
                        "topic_out": [port for n in nodes.values() for port in n.ports()],
                        "camera_info": [entry for item in instances.values()
                                        for entry in item.get("camera_info", [])]}
        except (ValueError, TypeError, zlib.error) as error:
            return {"state": "error", "message": str(error)}
        return None

    def shutdown(self):
        with self._lock:
            self._closing = True
            keys = list(self._nodes)
        for key in keys:
            self._retire(key)
