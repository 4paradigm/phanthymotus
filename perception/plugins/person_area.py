#!/usr/bin/env python3
"""Publish the largest detected person's image area from a JPEG camera topic."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.latest_frame import LatestFrame
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)

_RESULT_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1,
                         durability=DurabilityPolicy.VOLATILE)
_MODEL_URL = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/yolov8s-worldv2.pt"

TOOLS = [{
    "name": "area", "type": "processor", "multiInstance": True,
    "description": "Detect people in a camera feed and publish the largest person's image area.",
    "inputSchema": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
        "input_topic": {"type": "string", "description": "JPEG camera topic required for start."},
        "instance_id": {"type": "string"},
    }, "required": ["action"], "additionalProperties": False},
    "configSchema": {"type": "object", "properties": {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5,
                       "description": "Minimum person detection confidence.", "scope": "instance"},
        "fps": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5,
                "description": "Maximum inference frames per second.", "scope": "instance"},
    }},
    "topic_in": [{"format": "image/jpeg", "desc": "camera image input"}],
    "topic_out": [{"format": "data/json", "desc": "largest detected person's image area"}],
}]


def _output_topic(input_topic: str) -> str:
    return f"{input_topic}/person_area"


class _YoloPersonDetector:
    """A person-only detector; it does not alter or share VOP's output contract."""

    def __init__(self, model_name: str):
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO

        self._cv2, self._np = cv2, np
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(self._resolve_model_path(model_name))
        self._model.set_classes(["person"])
        if self._device != "cpu":
            self._model.model.to(self._device)

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        filename = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        if os.path.isfile(filename):
            return filename
        model_dir = os.environ.get("YOLO_MODEL_DIR", "/models")
        cached = os.path.join(model_dir, os.path.basename(filename))
        if os.path.isfile(cached):
            return cached
        if os.path.basename(filename) != "yolov8s-worldv2.pt":
            return filename
        os.makedirs(model_dir, exist_ok=True)
        log.info("[person_area] downloading model to %s", cached)
        urllib.request.urlretrieve(_MODEL_URL, cached)
        return cached

    def detect(self, jpeg: bytes, confidence: float) -> tuple[int, int, list[dict]]:
        frame = self._cv2.imdecode(self._np.frombuffer(jpeg, self._np.uint8), self._cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("unable to decode JPEG frame")
        height, width = frame.shape[:2]
        people = []
        for box in self._model(frame, conf=confidence, verbose=False)[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2 = max(0, min(width, round(x1))), max(0, min(width, round(x2)))
            y1, y2 = max(0, min(height, round(y1))), max(0, min(height, round(y2)))
            area_px = max(0, x2 - x1) * max(0, y2 - y1)
            if area_px:
                people.append({"bbox_xyxy": [x1, y1, x2, y2], "area_px": area_px,
                               "area_ratio": round(area_px / float(width * height), 6),
                               "confidence": round(float(box.conf[0]), 3)})
        return width, height, people


def _build_detector(cfg: dict) -> _YoloPersonDetector:
    return _YoloPersonDetector(str(cfg.get("model", "yolov8s-worldv2")))


class _PersonAreaNode(Node):
    def __init__(self, input_topic: str, detector, confidence: float, fps: float, suffix: str):
        super().__init__(f"person_area_{suffix}")
        self._input_topic, self._output_topic = input_topic, _output_topic(input_topic)
        self._detector, self._confidence = detector, confidence
        self._interval = 1.0 / max(fps, 0.1)
        self._pub = self.create_publisher(String, self._output_topic, _RESULT_QOS)
        self._sub = None
        self._frames: LatestFrame[bytes] = LatestFrame()
        self._stop_event = threading.Event()
        self._worker = None
        self._last_inference_at = 0.0
        self.state = "idle"

    def status(self) -> dict:
        return {"state": self.state, "input": self._input_topic, "output": self._output_topic}

    def start(self) -> dict:
        if self.state == "running":
            return self.status()
        self._frames, self._stop_event = LatestFrame(), threading.Event()
        if self._sub is None:
            self._sub = self.create_subscription(CompressedImage, self._input_topic, self._on_image, CAMERA_QOS)
        self.state = "running"
        self._worker = threading.Thread(target=self._run, daemon=True, name=f"person_area_{self._input_topic}")
        self._worker.start()
        return self.status()

    def stop(self) -> dict:
        self.state = "idle"
        self._stop_event.set()
        self._frames.close()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        return {"state": "idle"}

    def _on_image(self, msg: CompressedImage) -> None:
        if self.state == "running" and not self._stop_event.is_set():
            self._frames.push(bytes(msg.data))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            jpeg = self._frames.pop(timeout=0.5)
            if jpeg is None:
                continue
            wait = self._interval - (time.monotonic() - self._last_inference_at)
            if wait > 0 and self._stop_event.wait(wait):
                return
            try:
                width, height, people = self._detector.detect(jpeg, self._confidence)
                payload = {"timestamp": time.time(), "image_width": width, "image_height": height,
                           "person_count": len(people),
                           "largest_person": max(people, key=lambda item: item["area_px"], default=None)}
                message = String()
                message.data = json.dumps(payload, ensure_ascii=False)
                self._pub.publish(message)
                self._last_inference_at = time.monotonic()
            except Exception as exc:
                log.warning("[person_area] inference failed: %s", exc)


class PersonAreaPlugin:
    PREFIX = "person"

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        del namespace
        self._cfg, self._executor = dict(plugin_cfg), executor
        self._detector = None
        self._loading, self._load_error = False, None
        self._nodes: dict[str, _PersonAreaNode] = {}
        self._pending: dict[str, tuple[str, dict]] = {}
        self._lock = threading.RLock()

    def get_tools(self) -> list:
        return TOOLS

    def _load_and_start(self) -> None:
        try:
            detector = _build_detector(self._cfg)
            with self._lock:
                self._detector, pending = detector, list(self._pending.items())
                self._pending.clear()
            for key, (topic, cfg) in pending:
                self._start_node(key, topic, cfg)
        except Exception as exc:
            with self._lock:
                self._load_error, self._pending = str(exc), {}
            log.exception("[person_area] model load failed")
        finally:
            with self._lock:
                self._loading = False

    def _start_node(self, key: str, input_topic: str, instance_cfg: dict) -> None:
        with self._lock:
            old = self._nodes.pop(key, None)
        if old:
            old.stop()
            dispose_node(self._executor, old)
        confidence = float(instance_cfg.get("confidence", self._cfg.get("confidence", 0.5)))
        fps = float(instance_cfg.get("fps", self._cfg.get("fps", 5)))
        suffix = key.replace("/", "_").replace("-", "_").strip("_") or "default"
        node = _PersonAreaNode(input_topic, self._detector, confidence, fps, suffix)
        self._executor.add_node(node)
        node.start()
        with self._lock:
            self._nodes[key] = node

    def dispatch(self, name: str, args: dict) -> dict | None:
        action, instance_id = args.get("action", name), str(args.get("instance_id", ""))
        if action == "info":
            with self._lock:
                input_topic = self._nodes[instance_id]._input_topic if instance_id in self._nodes else str(args.get("input_topic", ""))
                state = "loading" if self._loading else "error" if self._load_error else "running" if self._nodes else "idle"
                return {"state": state, "model": self._cfg.get("model", "yolov8s-worldv2"), "error": self._load_error,
                        "instances": {key: node.status() for key, node in self._nodes.items()},
                        "topic_in": [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else [],
                        "topic_out": [{"topic": _output_topic(input_topic), "format": "data/json"}] if input_topic else []}
        if action == "start":
            input_topic = str(args.get("input_topic", ""))
            if not input_topic:
                raise ValueError("input_topic is required")
            key, instance_cfg = instance_id or input_topic, {k: args[k] for k in ("confidence", "fps") if k in args}
            with self._lock:
                if self._detector is None:
                    if self._load_error:
                        return {"state": "error", "message": self._load_error}
                    self._pending[key] = (input_topic, instance_cfg)
                    if not self._loading:
                        self._loading = True
                        threading.Thread(target=self._load_and_start, daemon=True, name="person_area_model_load").start()
                    return {"state": "loading", "input": input_topic, "output": _output_topic(input_topic)}
            self._start_node(key, input_topic, instance_cfg)
            return self._nodes[key].status()
        if action == "stop":
            with self._lock:
                keys = [instance_id] if instance_id else list(self._nodes)
                nodes = [self._nodes.pop(key) for key in keys if key in self._nodes]
                if instance_id:
                    self._pending.pop(instance_id, None)
                else:
                    self._pending.clear()
            for node in nodes:
                node.stop()
                dispose_node(self._executor, node)
            return {"state": "idle"}
        if action == "config":
            for key in ("confidence", "fps"):
                if key in args:
                    self._cfg[key] = args[key]
            return {"state": "configured", "confidence": self._cfg.get("confidence", 0.5), "fps": self._cfg.get("fps", 5)}
        return None
