"""Perception MCP processor and per-camera ROS2 lifecycle."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node

from .config import CONFIG_SCHEMA, MosaicConfig, MosaicError
from .runtime import MAX_JPEG_BYTES, MosaicEngine
from .stream import InputFrame, MosaicStream

PREFIX = "yunet_lk_mosaic"
SCHEMA = "motus.yunet_lk_mosaic/1"
MAX_INSTANCES = 4
NS = 1_000_000_000
LOG = logging.getLogger(__name__)
TOPIC_NAME = re.compile(r"/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*")
JPEG_FORMAT = re.compile(r"(?:[^;]+;\s*)?(?:jpeg|jpg)(?:\s+compressed(?:\s+\w+)?)?", re.I)
JPEG_PORT = {"format": "image/jpeg"}
TOOLS = [{
    "name": PREFIX,
    "type": "processor",
    "multiInstance": True,
    "description": "YuNet + LK 实时人脸局部马赛克。仅供视觉效果，不能保证匿名或隐私保护。",
    "inputSchema": {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
            "input_topic": {"type": "string", "description": "ROS2 摄像头 JPEG 话题"},
            "input_topics": {"type": "array", "items": {"type": "string"},
                             "minItems": 1, "maxItems": 1},
            "instance_id": {"type": "string"},
            "config": CONFIG_SCHEMA,
        },
        "x-action-params": {
            "start": {"params": ["input_topic"], "description": "启动马赛克"},
            "stop": {"params": [], "description": "停止马赛克"},
            "info": {"params": ["input_topic"], "description": "查看状态和统计"},
            "config": {"params": [], "description": "配置参数"},
        },
    },
    "configSchema": CONFIG_SCHEMA,
    "topic_in": [{**JPEG_PORT, "desc": "Camera JPEG"}],
    "topic_out": [
        {**JPEG_PORT, "desc": "Face mosaic JPEG"},
        {"format": "data/json", "desc": "Mosaic state and counters"},
    ],
}]


def _suffix(key):
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _input_topic(args, required=False):
    topic = args.get("input_topic")
    topics = args.get("input_topics")
    if topics == [] and not required:
        topics = None
    if topics is not None:
        if not isinstance(topics, list) or len(topics) != 1:
            raise MosaicError("exactly_one_camera_required")
        if topic and topic != topics[0]:
            raise MosaicError("conflicting_input_topics")
        topic = topics[0]
    if topic is None and not required:
        return None
    if not isinstance(topic, str) or len(topic) > 160 or not TOPIC_NAME.fullmatch(topic):
        raise MosaicError("invalid_input_topic")
    return topic


def _ports(topic, key):
    if topic is None:
        return {"topic_in": [], "topic_out": []}
    prefix = f"{topic}/{PREFIX}/instance_{_suffix(key)}"
    return {
        "topic_in": [{"topic": topic, **JPEG_PORT}],
        "topic_out": [
            {"topic": prefix + "/image", **JPEG_PORT},
            {"topic": prefix + "/status", "format": "data/json"},
        ],
    }


class MosaicNode(Node):
    def __init__(self, key, topic, config, path, finished):
        super().__init__(f"{PREFIX}_{_suffix(key)}")
        self.key, self.topic, self.config = key, topic, config
        self._path, self._finished = path, finished
        self._gate = threading.RLock()
        self._state, self._reason = "loading", "verifying_model"
        self._progress = None
        self._stream = None
        self._stamp = 0
        self._last_output_stamp = 0
        self._last_output_at = None
        self._received = self._invalid = self._stale = self._out_of_order = 0
        self._errors = 0
        self._failed_after_stamp = 0
        topics = _ports(topic, key)["topic_out"]
        try:
            self._image_pub = self.create_publisher(CompressedImage, topics[0]["topic"], CAMERA_QOS)
            self._status_pub = self.create_publisher(String, topics[1]["topic"], CAMERA_QOS)
            self.create_subscription(CompressedImage, topic, self._receive, CAMERA_QOS)
            self.create_timer(.5, self._publish_status)
        except Exception:
            self.destroy_node()
            raise
        self.thread = threading.Thread(target=self._run, name=self.get_name(), daemon=True)

    def snapshot(self):
        with self._gate:
            age = (time.monotonic() - self._last_output_at
                   if self._last_output_at is not None else None)
            metrics = self._stream.snapshot() if self._stream is not None else {}
            return {
                "schema": SCHEMA, "instance_id": self.key,
                "privacy_guaranteed": False,
                "state": self._state, "reason": self._reason,
                "model_fetch": self._progress,
                "input_topic": self.topic, **_ports(self.topic, self.key),
                "config": asdict(self.config),
                "last_output_stamp_ns": self._last_output_stamp,
                "output_age_s": age, "fresh": age is not None and age < .6,
                "metrics": {**metrics, "received": self._received, "invalid_input": self._invalid,
                            "stale_input_at_receipt": self._stale,
                            "out_of_order_input": self._out_of_order,
                            "errors": self._errors, "target_fps": self.config.target_fps},
            }

    def _receive(self, msg):
        with self._gate:
            if self._state != "running":
                return
            stamp = msg.header.stamp.sec * NS + msg.header.stamp.nanosec
            if (stamp <= 0 or not 0 <= msg.header.stamp.nanosec < NS
                    or not 4 <= len(msg.data) <= MAX_JPEG_BYTES
                    or not JPEG_FORMAT.fullmatch(msg.format.strip())):
                self._invalid += 1
                return
            received = time.monotonic()
            source_age = (self.get_clock().now().nanoseconds - stamp) / NS
            if source_age > .5:
                self._stale += 1
                return
            if source_age < -.1:
                self._invalid += 1
                return
            if stamp <= self._stamp:
                self._out_of_order += 1
                return
            self._stamp = stamp
            self._received += 1
            self._stream.inbox.push(InputFrame(
                bytes(msg.data), stamp, received - max(0., source_age), received, msg.header,
            ))

    def _publish_image(self, frame, encoded, rectangles):
        with self._gate:
            if self._state != "running":
                return False
            msg = CompressedImage()
            msg.header, msg.format, msg.data = frame.header, "jpeg", encoded
            self._image_pub.publish(msg)
            self._last_output_stamp = frame.stamp
            self._last_output_at = time.monotonic()
            # A buffered success may predate a newer processing error.
            if frame.stamp > self._failed_after_stamp:
                self._reason = "face_roi_mosaic" if rectangles else "no_face"
            return True

    def _publish_status(self):
        with self._gate:
            if self._state != "stopping":
                msg = String()
                msg.data = json.dumps(self.snapshot(), ensure_ascii=False)
                self._status_pub.publish(msg)

    def _progress_update(self, percent, mb_done, mb_total):
        with self._gate:
            if self._state == "loading":
                self._reason = "downloading_model"
                self._progress = {
                    "percent": percent, "bytes_done": round(mb_done * 1024 ** 2),
                    "bytes_total": round(mb_total * 1024 ** 2),
                }

    def _processing_error(self, exc, frame):
        with self._gate:
            if self._state == "running":
                self._errors += 1
                self._reason = f"processing_failed:{type(exc).__name__}"
                self._failed_after_stamp = max(self._failed_after_stamp, frame.stamp)

    def _run(self):
        engine = None
        try:
            engine = MosaicEngine(self._path, self.config, progress=self._progress_update)
            with self._gate:
                if self._state == "stopping":
                    return
            engine.detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            with self._gate:
                if self._state == "stopping":
                    return
                self._stream = MosaicStream(engine, self.config, self._publish_image, self._processing_error)
                self._state, self._reason = "running", "waiting_for_camera"
            self._stream.run()
        except Exception as exc:
            with self._gate:
                if self._state != "stopping":
                    self._state, self._reason = "error", f"startup_failed:{exc}"
                    self._errors += 1
            LOG.exception("[%s] instance worker failed", PREFIX)
            self._publish_status()
        finally:
            try:
                if engine is not None:
                    engine.close()
            finally:
                self._finished(self)

    def stop(self):
        with self._gate:
            self._state, self._reason = "stopping", "stopped"
            if self._stream is not None:
                self._stream.stop()


class YuNetLkMosaicPlugin:
    PREFIX = PREFIX

    def __init__(self, cfg, executor):
        if not isinstance(cfg, dict):
            raise MosaicError("plugin_config_must_be_object")
        self.defaults = MosaicConfig().updated({
            key: val for key, val in cfg.items() if key not in ("enabled", "model_path")
        })
        model_path = cfg.get("model_path")
        if model_path is not None and not isinstance(model_path, str):
            raise MosaicError("invalid_config:model_path")
        self.model_path = model_path or os.environ.get("YUNET_LK_MOSAIC_MODEL") or None
        self.executor = executor
        self._lock = threading.RLock()
        self._nodes, self._configs = {}, {}
        self._errors = OrderedDict()
        self._closed = False

    def get_tools(self):
        return copy.deepcopy(TOOLS)

    def _finished(self, node):
        dispose_node(self.executor, node, label=PREFIX)
        with self._lock:
            snapshot = node.snapshot()
            if snapshot["state"] == "error":
                self._errors[node.key] = snapshot
                if len(self._errors) > MAX_INSTANCES:
                    self._errors.popitem(last=False)
            self._nodes.pop(node.key)

    def _start(self, key, cfg, topic):
        node = self._nodes.get(key)
        if node is not None:
            if node.topic != topic:
                raise MosaicError("stop_before_change_camera")
            return node.snapshot()
        if not cfg.consent_acknowledged:
            raise MosaicError("consent_acknowledgment_required")
        if len(self._nodes) >= MAX_INSTANCES:
            raise MosaicError("max_instances_exceeded")
        node = MosaicNode(key, topic, cfg, self.model_path, self._finished)
        self._nodes[key] = node
        try:
            if self.executor.add_node(node) is False:
                raise MosaicError("executor_rejected_node")
            node.thread.start()
        except Exception:
            self._nodes.pop(key)
            dispose_node(self.executor, node, label=PREFIX)
            raise
        self._errors.pop(key, None)
        return node.snapshot()

    def dispatch(self, name, args):
        if not isinstance(args, dict):
            return {"state": "error", "error": "invalid_args"}
        action = args.get("action", name)
        instance_id = args.get("instance_id")
        key = "_default" if instance_id is None else instance_id
        if not isinstance(key, str) or not 1 <= len(key) <= 256:
            return {"state": "error", "error": "invalid_instance_id"}
        try:
            if action == "stop":
                return self._stop(instance_id)
            with self._lock:
                cfg = self._configs.get(key, self.defaults)
                if action == "info":
                    node = self._nodes.get(key)
                    if node is not None:
                        return node.snapshot()
                    if key in self._errors:
                        return copy.deepcopy(self._errors[key])
                    return {
                        "state": "idle", "instance_id": key, "privacy_guaranteed": False,
                        "config": asdict(cfg), **_ports(_input_topic(args), key),
                    }
                if self._closed:
                    raise MosaicError("plugin_closed")
                if action == "config":
                    if key in self._nodes:
                        raise MosaicError("stop_before_config")
                    values = args.get("config", {
                        k: v for k, v in args.items() if k not in ("action", "instance_id")
                    })
                    updated = cfg.updated(values)
                    self._configs[key] = updated
                    self._errors.pop(key, None)
                    return {"state": "configured", "config": asdict(updated)}
                if action == "start":
                    return self._start(key, cfg, _input_topic(args, required=True))
                raise MosaicError("unknown_action")
        except (ValueError, TypeError) as exc:
            return {"state": "error", "error": str(exc), "privacy_guaranteed": False}

    def _stop(self, key=None):
        with self._lock:
            if key is None:
                nodes = list(self._nodes.values())
                self._configs.clear()
                self._errors.clear()
            else:
                nodes = [self._nodes[key]] if key in self._nodes else []
                self._configs.pop(key, None)
                self._errors.pop(key, None)
            for node in nodes:
                node.stop()
        for node in nodes:
            node.thread.join(timeout=2)
        return {"state": "stopping" if any(node.thread.is_alive() for node in nodes) else "idle",
                "image_publication_disabled": True}

    def shutdown(self):
        with self._lock:
            self._closed = True
        self._stop()
