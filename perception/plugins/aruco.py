"""ArUco Processor: compressed camera frames in, metric poses and JPEG out."""

import hashlib
import json
import logging
import math
import threading
import time

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.aruco_runtime import Calibration, DEFAULT_DICTIONARY, DICTIONARIES, MarkerDetector
from plugins.camera_info import for_topic
from utils.latest_frame import LatestFrame
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node


log = logging.getLogger(__name__)
MAX_JPEG_BYTES = 8 * 1024 * 1024
DEFAULTS = {"dictionary": DEFAULT_DICTIONARY, "marker_size_m": 0.1, "rate_hz": 5.0}
CONFIG = {
    "dictionary": {"type": "string", "title": "ArUco dictionary", "enum": list(DICTIONARIES),
                   "default": DEFAULT_DICTIONARY, "scope": "instance"},
    "marker_size_m": {"type": "number", "title": "Marker side (m)", "exclusiveMinimum": 0,
                      "default": DEFAULTS["marker_size_m"], "scope": "instance",
                      "description": "Measure one side of the outer black square, excluding the white margin."},
    "rate_hz": {"type": "number", "title": "Maximum processing rate (Hz)", "minimum": 0.1,
                "maximum": 30, "default": DEFAULTS["rate_hz"], "scope": "instance"},
    "calibration": {"type": "object", "title": "Camera calibration override", "scope": "instance",
                    "description": "Optional: width, height, row-major K, distortion_model and D for the actual JPEG. "
                                   "Enter {} to clear an override and use the wired camera's calibration."},
}
TOOLS = [{
    "name": "aruco",
    "type": "processor",
    "multiInstance": True,
    "description": "Estimate calibrated ArUco poses relative to the camera and display XYZ axes in Monitor. "
                   "ArUco 定位：输出相机光学坐标系下的标记位姿和叠加坐标轴的画面。",
    "inputSchema": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
                       "input_topic": {"type": "string", "description": "Camera JPEG topic"}},
        "required": ["action"],
        "x-action-params": {"start": {"params": ["input_topic"]}, "stop": {"params": []},
                            "info": {"params": []}, "config": {"params": []}},
    },
    "configSchema": {"type": "object", "properties": CONFIG},
    "topic_in": [{"format": "image/jpeg", "desc": "Calibrated camera JPEG"}],
    "topic_out": [{"format": "data/json", "desc": "Marker poses in the camera optical frame (m)"},
                  {"format": "image/jpeg", "desc": "Camera image with marker IDs and XYZ axes"}],
}]


def _validate(options):
    if options["dictionary"] not in DICTIONARIES:
        raise ValueError("Select a supported ArUco dictionary.")
    for key in ("marker_size_m", "rate_hz"):
        value = options[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number.")
    if not 0.1 <= options["rate_hz"] <= 30:
        raise ValueError("rate_hz must be between 0.1 and 30.")
    override = options.get("calibration")
    if override is not None and override != {}:
        Calibration.from_declaration(override)


class _ArucoNode(Node):
    def __init__(self, topic, key, options, declaration, detector):
        self.calibration = Calibration.from_declaration(declaration)
        suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        super().__init__(f"aruco_{suffix}")
        self.input_topic = topic
        self.options = options
        self.camera_id = declaration.get("id") or ""
        self.detector = detector
        self.frames = LatestFrame()
        self._closed = threading.Event()
        self._publish_lock = threading.Lock()
        self.last_error = ""
        base = f"{topic}/aruco/i_{suffix}"
        self.outputs = [{"topic": f"{base}/poses", **TOOLS[0]["topic_out"][0]},
                        {"topic": f"{base}/overlay", **TOOLS[0]["topic_out"][1]}]
        try:
            # Keep outputs reliable for Monitor and other DDS subscribers.
            self._poses = self.create_publisher(String, self.outputs[0]["topic"], 1)
            self._overlay = self.create_publisher(CompressedImage, self.outputs[1]["topic"], 1)
            self.create_subscription(CompressedImage, topic, self.frames.push, CAMERA_QOS)
        except Exception:
            self.destroy_node()
            raise
        self.worker = threading.Thread(target=self._run, name=f"aruco_{suffix}", daemon=True)

    def close(self):
        with self._publish_lock:
            self._closed.set()
        self.frames.close()
        if self.worker.ident is not None:
            self.worker.join(timeout=2)

    def _run(self):
        interval = 1.0 / self.options["rate_hz"]
        while not self._closed.is_set():
            frame = self.frames.pop(timeout=1.0)
            if frame is None:
                continue
            start = time.monotonic()
            try:
                self._process(frame)
            except Exception:
                log.exception("ArUco could not process a camera frame")
            # Pop after waiting so the next iteration sees the newest frame.
            self._closed.wait(max(0, interval - (time.monotonic() - start)))

    def _process(self, frame):
        cv = self.detector.cv
        stamp = frame.header.stamp
        result = {"camera_id": self.camera_id, "frame_id": frame.header.frame_id,
                  "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
                  "coordinate_frame": "camera_optical", "dictionary": self.options["dictionary"],
                  "marker_size_m": self.options["marker_size_m"], "markers": []}
        image, overlay = None, None
        try:
            if not 0 < len(frame.data) <= MAX_JPEG_BYTES:
                raise ValueError("Camera JPEG is empty or exceeds the size limit.")
            image = cv.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv.IMREAD_COLOR | cv.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                raise ValueError("Could not decode camera JPEG.")
            result["markers"] = self.detector.estimate(image, self.calibration, self.options["marker_size_m"])
        except (ValueError, cv.error) as exc:
            result["error"] = str(exc)
        if image is not None:
            ok, encoded = cv.imencode(".jpg", image)
            if ok:
                overlay = CompressedImage()
                overlay.header = frame.header
                overlay.format = "jpeg"
                overlay.data = encoded.tobytes()
        message = String()
        message.data = json.dumps(result, allow_nan=False)
        with self._publish_lock:
            if self._closed.is_set():
                return
            self.last_error = result.get("error", "")
            self._poses.publish(message)
            if overlay is not None:
                self._overlay.publish(overlay)


class ArucoPlugin:
    PREFIX = "aruco"

    def __init__(self, cfg, executor):
        self._executor = executor
        self._defaults = {**DEFAULTS, **{key: cfg[key] for key in CONFIG if key in cfg}}
        self._overrides = {}
        self._nodes = {}
        self._lock = threading.Lock()
        self._closed = False

    def get_tools(self):
        return TOOLS

    def dispatch(self, name, args):
        args = args or {}
        action = args.get("action", name)
        instance = args.get("instance_id") or ""
        topic = args.get("input_topic") or ""
        with self._lock:
            if action == "start":
                return self._start(instance, topic, args.get("camera_info"))
            if action == "info":
                node = self._nodes.get(instance or topic)
                if not (instance or topic) and len(self._nodes) == 1:
                    node = next(iter(self._nodes.values()))
                return self._info(node)
            if action == "stop":
                self._stop(instance or topic)
                return {"state": "idle"}
            if action == "config":
                return self._config(instance, args)
        return None

    def _info(self, node):
        if node is not None:
            return {"state": "running", "topic_in": [{"topic": node.input_topic, "format": "image/jpeg"}],
                    "topic_out": node.outputs, "last_frame_error": node.last_error}
        try:
            MarkerDetector(DEFAULT_DICTIONARY)
        except RuntimeError as exc:
            return {"state": "error", "error": str(exc)}
        return {"state": "idle", "topic_out": [], "instances": list(self._nodes)}

    def _config(self, instance, args):
        changes = {key: args[key] for key in CONFIG if key in args}
        options = {**self._defaults, **self._overrides.get(instance, {}), **changes}
        try:
            _validate(options)
        except (ValueError, OverflowError) as exc:
            # The MCP handler turns this into a JSON-RPC error. Core aborts its
            # pre-start config call on that error instead of using old settings.
            raise ValueError(str(exc)) from exc
        if instance:
            self._overrides[instance] = {**self._overrides.get(instance, {}), **changes}
        else:
            self._defaults.update(changes)
        return {"state": "configured", "message": "Settings take effect on the next start."}

    def _start(self, instance, topic, camera_info):
        if self._closed or self._executor is None:
            return {"state": "error", "error": "ArUco ROS executor is unavailable."}
        if not isinstance(topic, str) or not topic.startswith("/") or topic.endswith("/"):
            return {"state": "error", "error": "Connect a camera or provide an absolute camera JPEG topic."}
        key = instance or topic
        previous = self._nodes.get(key)
        if previous is not None and previous.input_topic == topic:
            return self._info(previous)
        options = {**self._defaults, **self._overrides.get(instance, {})}
        node = None
        try:
            _validate(options)
            declarations = camera_info if isinstance(camera_info, (dict, list, tuple)) else None
            upstream = for_topic(declarations, topic)
            declaration = options.get("calibration") or upstream
            if not declaration:
                raise ValueError("No calibration for this JPEG topic. Configure camera calibration or a card override.")
            if options.get("calibration") and upstream.get("id"):
                declaration = {"id": upstream["id"], **declaration}
            detector = MarkerDetector(options["dictionary"])
            node = _ArucoNode(topic, key, options, declaration, detector)
            if self._executor.add_node(node) is False:
                raise RuntimeError("Could not register the ArUco node in the ROS executor.")
            node.worker.start()
        except Exception as exc:
            if node is not None:
                self._dispose(node)
            return {"state": "error", "error": str(exc)}
        if previous is not None:
            self._dispose(previous)
        self._nodes[key] = node
        return self._info(node)

    def _dispose(self, node):
        node.close()
        dispose_node(self._executor, node, label=node.input_topic)

    def _stop(self, key=""):
        for existing in ([key] if key else list(self._nodes)):
            node = self._nodes.pop(existing, None)
            if node is not None:
                self._dispose(node)

    def shutdown(self):
        with self._lock:
            self._closed = True
            self._stop()
