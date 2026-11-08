"""Shared RGB rectification processor: camera -> undistort -> all consumers."""
from __future__ import annotations

import copy
import hashlib
import logging
import queue
import re
import threading
import time

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from plugins.camera_info import for_topic
from plugins.undistortion_runtime import Rectifier, resolve, validate
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)

TOOLS = [{
    "name": "undistort", "type": "processor", "multiInstance": True,
    "description": "相机去畸变：输出同一路校正 RGB，供 visual_depth、VOP 和导航共同使用。参数默认自动选择：先用上游相机声明的 K/D，再按相机 id 查内置标定表；都没有就报错，不猜内参。输出自动裁掉无效边缘。",
    "inputSchema": {
        "type": "object", "required": ["action"],
        "properties": {
            "action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
            "input_topic": {"type": "string", "description": "原始相机 JPEG 话题"},
        },
        "x-action-params": {
            "start": {"params": ["input_topic"], "description": "开始发布校正后的 RGB"},
            "stop": {"params": [], "description": "停止本实例"},
            "info": {"params": ["input_topic"], "description": "查看输出话题、相机参数和处理状态"},
            "config": {"params": [], "description": "停止实例后修改参数，修改后重新校准下游深度尺度"},
        },
    },
    "configSchema": {
        "type": "object", "properties": {
            "calibration": {
                "type": "object", "title": "Camera calibration", "default": {}, "scope": "instance",
                "description": "留空 {} = 自动：先用上游相机声明的 K/D，再按相机 id 查内置标定表。只有要覆盖时才手填准确的 id、width、height、K、D、distortion_model，必须匹配实际输入流。",
                "properties": {
                    "id": {"type": "string"},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1},
                    "K": {"type": "array", "items": {"type": "number"}},
                    "D": {"type": "array", "items": {"type": "number"}},
                    "distortion_model": {"type": "string", "enum": ["plumb_bob", "rational_polynomial", "equidistant"]},
                },
            },
            "fps": {"type": "integer", "minimum": 1, "maximum": 30, "default": 15,
                    "scope": "instance", "title": "Output FPS"},
            "jpeg_quality": {"type": "integer", "minimum": 1, "maximum": 100, "default": 90,
                             "scope": "instance", "title": "JPEG quality"},
        },
    },
    "topic_in": [{"format": "image/jpeg", "desc": "原始相机 RGB"}],
    "topic_out": [{"format": "image/jpeg", "desc": "校正 RGB：分给深度、VOP 和导航"}],
}]


def output_topic(topic, key):
    ident = re.sub(r"[^a-zA-Z0-9_]", "_", key)[:48]
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    return topic.rstrip("/") + f"/undistorted/instance_{ident}_{digest}/rgb"


def _options(config):
    cfg = {"calibration": {}, "fps": 15, "jpeg_quality": 90, **config}
    validate(cfg["calibration"])
    for key, maximum in (("fps", 30), ("jpeg_quality", 100)):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= maximum:
            raise ValueError(f"{key} must be an integer in 1..{maximum}")
    return cfg


class _UndistortNode(Node):
    def __init__(self, key, topic, rectifier, cfg):
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        super().__init__(f"undistort_{digest}")
        self.topic = topic
        self.output = output_topic(topic, key)
        self.rectifier = rectifier
        self.cfg = cfg
        self.frames = 0
        self.error = ""
        self.last_frame_at = None
        self._last_received = -float("inf")
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = None
        self.publisher = self.create_publisher(CompressedImage, self.output, CAMERA_QOS)
        self.subscription = self.create_subscription(CompressedImage, topic, self._image_cb, CAMERA_QOS)

    def start(self):
        self._worker = threading.Thread(target=self._run, daemon=True, name=self.get_name())
        self._worker.start()

    def _image_cb(self, message):
        if self._stop.is_set():
            return
        now = time.monotonic()
        if now - self._last_received < 1 / self.cfg["fps"]:
            return
        self._last_received = now
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                pass

    def _run(self):
        import cv2
        while not self._stop.is_set():
            try:
                message = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                frame = cv2.imdecode(np.frombuffer(bytes(message.data), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Cannot decode input JPEG")
                frame = self.rectifier.apply(frame)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["jpeg_quality"]])
                if not ok:
                    raise ValueError("Cannot encode rectified JPEG")
                result = CompressedImage()
                result.header = copy.deepcopy(message.header)
                result.format = "jpeg"
                result.data = encoded.tobytes()
                if self._stop.is_set():
                    break
                self.publisher.publish(result)
                self.frames += 1
                self.last_frame_at = time.monotonic()
                self.error = ""
            except Exception as exc:
                if self.error != str(exc):
                    log.warning("undistort %s: %s", self.topic, exc)
                self.error = str(exc)

    def stop(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=5)
            if self._worker.is_alive():
                raise RuntimeError("Undistortion worker is still stopping; retry stop")


class UndistortPlugin:
    PREFIX = "undistort"

    def __init__(self, plugin_cfg, executor):
        self._defaults = _options({k: v for k, v in plugin_cfg.items()
                                  if k in ("fps", "jpeg_quality")})
        self._executor = executor
        self._configs = {}
        self._nodes = {}
        self._declarations = {}
        self._bindings = {}
        self._sources = {}
        self._lock = threading.RLock()
        self._closing = False

    def get_tools(self):
        return copy.deepcopy(TOOLS)

    def _info(self, key, topic=""):
        node = self._nodes.get(key)
        topic = node.topic if node else topic or self._bindings.get(key, "")
        camera = copy.deepcopy(self._declarations.get(key))
        result = {"state": "running" if node else "idle",
                  "topic_in": [{"topic": topic, "format": "image/jpeg"}] if topic else [],
                  "topic_out": [{"topic": output_topic(topic, key), "format": "image/jpeg"}] if topic else []}
        if camera:
            result["camera_info"] = [camera]
        if key in self._sources:
            # manual / upstream / table:<id> — which kind of number is
            # rectifying this picture. A wrong table row and a correct typed-in
            # one look identical from downstream; this is what separates them.
            result["calibration_source"] = self._sources[key]
        if node:
            result.update(frame_count=node.frames, error=node.error,
                          last_frame_age_s=(time.monotonic() - node.last_frame_at
                                            if node.last_frame_at is not None else None))
        return result

    def _retire(self, key):
        node = self._nodes.get(key)
        if node:
            node.stop()
            dispose_node(self._executor, node)
            self._nodes.pop(key, None)

    def dispatch(self, name, args):
        action = args.get("action", name)
        topics = args.get("input_topics") or []
        topic = args.get("input_topic") or (topics[0] if len(topics) == 1 else "")
        key = args.get("instance_id") or topic or "_default"
        with self._lock:
            try:
                if action == "config":
                    updates = {k: args[k] for k in ("calibration", "fps", "jpeg_quality") if k in args}
                    cfg = _options({**self._defaults, **self._configs.get(key, {}), **updates})
                    if key in self._nodes and cfg != self._nodes[key].cfg:
                        raise ValueError("Stop this instance before changing rectification configuration")
                    self._configs[key] = copy.deepcopy(cfg)
                    if key not in self._nodes:
                        self._declarations.pop(key, None)
                        self._sources.pop(key, None)
                    return {"status": "configured", "instance_id": key, "config": cfg}
                if action == "info":
                    if key == "_default" and len(self._nodes) == 1:
                        key = next(iter(self._nodes))
                    return self._info(key, topic)
                if action == "stop":
                    keys = list(self._nodes) if key == "_default" else [key]
                    for item in keys:
                        self._retire(item)
                    return {"state": "idle"}
                if action != "start":
                    return None
                if self._closing:
                    raise ValueError("Undistortion plugin is shutting down")
                if not topic or not topic.startswith("/"):
                    raise ValueError("Connect exactly one absolute camera input_topic")
                if topics and len(set(topics + [topic])) > 1:
                    raise ValueError("Connect exactly one camera per undistort instance")
                if key in self._nodes:
                    if self._nodes[key].topic != topic:
                        raise ValueError("Stop this instance before changing its input camera")
                    return self._info(key)
                cfg = copy.deepcopy(self._configs.get(key, self._defaults))
                upstream = for_topic(args.get("camera_info"), topic)
                if "perception/undistort" in (upstream.get("pipeline") or []):
                    raise ValueError("Input is already rectified; connect its RGB directly downstream")
                cal, source = resolve(cfg["calibration"], upstream)
                rectifier = Rectifier(cal)
                camera = rectifier.camera_info(upstream)
                output = output_topic(topic, key)
                camera.update(topic=output, format="image/jpeg")
                node = _UndistortNode(key, topic, rectifier, cfg)
                try:
                    self._executor.add_node(node)
                    node.start()
                except Exception:
                    node.stop()
                    dispose_node(self._executor, node)
                    raise
                self._nodes[key] = node
                self._declarations[key] = camera
                self._bindings[key] = topic
                self._sources[key] = source
                return self._info(key)
            except (ValueError, RuntimeError) as exc:
                return {"state": "error", "status": "error", "message": str(exc)}

    def shutdown(self):
        with self._lock:
            self._closing = True
            for key in list(self._nodes):
                self._retire(key)
