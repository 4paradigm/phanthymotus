"""Opt-in robot model recognition using VOP boxes and a local CLIP encoder."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.image_input import BadInput, load_image_bytes
from plugins.robot_model_catalog import CANDIDATES, decide_model
from plugins.robot_model_proxy import RobotModelProxy
from utils.latest_frame import LatestFrame
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node


ROBOT_CLASSES = frozenset(("robot", "quadruped robot", "humanoid robot", "drone"))
TOOLS = [{
    "name": "robot_model", "type": "processor", "multiInstance": True,
    "description": "Identify visible robot models locally from a photo or camera frame",
    "inputSchema": {
        "type": "object", "required": ["action"],
        "properties": {
            "action": {"type": "string", "enum": ["start", "stop", "info", "recognize_by_photo", "recognize_by_url"]},
            "input_topic": {"type": "string", "description": "Camera image/jpeg topic"},
            "image_path": {"type": "string", "format": "file", "accept": "image/*", "uploadTo": "mcp"},
            "url": {"type": "string", "description": "http(s) image URL"},
        },
        "x-action-params": {
            "start": {"params": ["input_topic"]},
            "stop": {"params": []},
            "info": {"params": []},
            "recognize_by_photo": {"params": ["image_path"]},
            "recognize_by_url": {"params": ["url"]},
        },
    },
    "topic_in": [{"format": "image/jpeg", "desc": "camera frame"}],
    "topic_out": [{"format": "data/json", "desc": "robot model results"}],
}]


_RESULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class _RobotModelNode(Node):
    def __init__(self, topic: str, recognize, fps: float):
        super().__init__("robot_model_" + str(abs(hash(topic)) % 100000000))
        self.input_topic = topic
        self.output_topic = topic + "/robot_models"
        self._recognize = recognize
        self._interval = 1.0 / fps if fps > 0 else 0.0
        self._frames = LatestFrame()
        self._lock = threading.Lock()
        self._stopped = False
        self._pub = self.create_publisher(String, self.output_topic, _RESULT_QOS)
        self._sub = self.create_subscription(
            CompressedImage, topic, self._image_cb, CAMERA_QOS)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _image_cb(self, message):
        self._frames.push(bytes(message.data))

    def _worker(self):
        import cv2
        last = -float("inf")
        while True:
            data = self._frames.pop(timeout=0.2)
            with self._lock:
                if self._stopped:
                    return
            if data is None:
                continue
            now = time.monotonic()
            if now - last < self._interval:
                continue
            last = now
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            result = self._recognize(frame)
            with self._lock:
                if not self._stopped:
                    message = String()
                    message.data = json.dumps(result)
                    self._pub.publish(message)

    def stop(self):
        with self._lock:
            self._stopped = True
            self._frames.close()
        self.destroy_subscription(self._sub)
        self._thread.join(timeout=3)


class RobotModelPlugin:
    def __init__(self, plugin_cfg: dict, detector, executor):
        self._cfg = dict(plugin_cfg)
        self._detector = detector
        self._executor = executor
        self._proxy = None
        self._proxy_lock = threading.Lock()
        self._nodes = {}
        self._nodes_lock = threading.Lock()

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "robot_model" else name
        if action == "info":
            with self._nodes_lock:
                instances = {key: {"state": "running", "input_topic": node.input_topic,
                                   "output_topic": node.output_topic}
                             for key, node in self._nodes.items()}
            return {"name": "Robot Model", "model": "robot_model",
                    "state": "running" if instances else
                             ("idle" if self._detector else "unavailable"),
                    "model_loaded": self._proxy is not None,
                    "calibrated": self._cfg.get("min_score") is not None and
                                  self._cfg.get("min_margin") is not None,
                    "detail": "VOP detector is required" if self._detector is None else
                              "Local robot model recognition is available",
                    "instances": instances,
                    "topic_in": [{"topic": node["input_topic"], "format": "image/jpeg"}
                                 for node in instances.values()],
                    "topic_out": [{"topic": node["output_topic"], "format": "data/json"}
                                  for node in instances.values()]}
        if action == "start":
            if self._detector is None:
                return self._unavailable("VOP detector is disabled")
            topic = str(args.get("input_topic") or "")
            if not topic:
                return {"state": "idle", "detail": "Photo recognition is available on demand"}
            key = str(args.get("instance_id") or topic)
            with self._nodes_lock:
                old = self._nodes.pop(key, None)
                if old is not None:
                    old.stop()
                    dispose_node(self._executor, old, label=f"robot_model/{key}")
                node = _RobotModelNode(topic, self.recognize_frame,
                                       float(self._cfg.get("fps", 1)))
                self._executor.add_node(node)
                self._nodes[key] = node
            return {"state": "running", "input": topic,
                    "output": node.output_topic}
        if action == "stop":
            key = str(args.get("instance_id") or args.get("input_topic") or "")
            with self._nodes_lock:
                keys = [key] if key else list(self._nodes)
                nodes = [(item, self._nodes.pop(item)) for item in keys
                         if item in self._nodes]
            for item, node in nodes:
                node.stop()
                dispose_node(self._executor, node, label=f"robot_model/{item}")
            return {"state": "idle"}
        if action in ("recognize_by_photo", "recognize_by_url"):
            return self._recognize_image(args)
        return None

    @staticmethod
    def _unavailable(detail: str) -> dict:
        return {"ok": False, "reason": "engine_unavailable", "detail": detail}

    def _get_proxy(self):
        with self._proxy_lock:
            if self._proxy is None:
                self._proxy = RobotModelProxy(
                    str(self._cfg.get("model_dir", "/models/robot_model")))
            return self._proxy

    def recognize_frame(self, frame: np.ndarray) -> dict:
        if self._detector is None:
            return self._unavailable("VOP detector is disabled")
        try:
            detections = self._detector.detect_frame(
                frame, float(self._cfg.get("detector_confidence", 0.3)))
        except Exception as error:
            return self._unavailable(f"VOP detection failed: {error}")

        height, width = frame.shape[:2]
        objects = []
        for detected in detections:
            if detected.get("name") not in ROBOT_CLASSES:
                continue
            x1, y1, x2, y2 = detected["bbox"]
            x1, x2 = max(0, int(x1)), min(width, int(x2))
            y1, y2 = max(0, int(y1)), min(height, int(y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            import cv2
            encoded, data = cv2.imencode(".jpg", frame[y1:y2, x1:x2])
            if not encoded:
                continue
            try:
                proxy = self._get_proxy()
                embedding = np.asarray(proxy.embed(data.tobytes()), dtype=np.float32)
                scores = {model_id: float(score) for (model_id, _), score in
                          zip(CANDIDATES, proxy.vectors @ embedding)}
                decision = decide_model(scores, self._cfg.get("min_score"),
                                        self._cfg.get("min_margin"))
            except Exception as error:
                return self._unavailable(f"Robot model encoder unavailable: {error}")
            objects.append({
                "bbox": [x1, y1, x2, y2],
                "detector_class": detected["name"],
                "detector_score": detected["confidence"],
                **decision,
            })
        return {"ok": True, "objects": objects}

    def _recognize_image(self, args: dict) -> dict:
        if self._detector is None:
            return self._unavailable("VOP detector is disabled")
        try:
            data, source = load_image_bytes(
                args, self._cfg, url_action="recognize_by_url")
        except BadInput as error:
            return error.as_result()
        import cv2
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return BadInput("could not decode that file as an image", source).as_result()
        return {"source": source, **self.recognize_frame(frame)}
