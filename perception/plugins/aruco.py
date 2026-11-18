"""Locate ArUco markers in a calibrated camera stream with an optional overlay."""

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
from plugins.camera_info import SCHEMA, for_topic
from utils.latest_frame import LatestFrame
from utils.log_sampling import SampledLogGate, escape_log_text
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node


log = logging.getLogger(__name__)
MAX_JPEG_BYTES = 8 * 1024 * 1024
DEFAULTS = {"dictionary": DEFAULT_DICTIONARY, "marker_size_m": 0.1, "rate_hz": 5.0, "publish_overlay": True}
CONFIG = {
    "dictionary": {"type": "string", "title": "ArUco dictionary", "enum": list(DICTIONARIES),
                   "default": DEFAULT_DICTIONARY, "scope": "instance",
                   "description": "Match the printed marker's dictionary. IDs belong to that dictionary, whose "
                                  "patterns are designed to reduce ID mix-ups. For new markers, use DICT_4X4_50 for "
                                  "up to 50 IDs when tags look small in the image, DICT_5X5_50 for 50 IDs with more "
                                  "separation between patterns, DICT_6X6_250 for up to 250 IDs, or DICT_7X7_50 for up to "
                                  "50 highly distinct patterns when the camera can resolve the finer cells."},
    "marker_size_m": {"type": "number", "title": "Marker side (m)", "exclusiveMinimum": 0,
                      "default": DEFAULTS["marker_size_m"], "scope": "instance",
                      "description": "Default 0.10 m (10 cm). If your marker differs, enter its measured outer "
                                     "black side in meters. Exclude the white margin. Use one size per instance."},
    "rate_hz": {"type": "number", "title": "Maximum processing rate (Hz)", "minimum": 0.1,
                "maximum": 30, "default": DEFAULTS["rate_hz"], "scope": "instance"},
    "publish_overlay": {"type": "boolean", "title": "Publish Monitor overlay", "scope": "instance",
                        "default": DEFAULTS["publish_overlay"],
                        "description": "Turn off to publish only pose JSON and skip drawing and JPEG encoding."},
}
TOOLS = [{
    "name": "aruco",
    "type": "processor",
    "multiInstance": True,
    "description": "Locate multiple ArUco markers relative to the wired camera. The default black side is 10 cm "
                   "(0.10 m). If yours differs, set Marker side (m) to the measured value. "
                   "同帧定位同一词典内的多个标记。默认黑框外边长为 10 cm（0.10 m），实际不同时请在配置中填写实测边长（米）。",
    "inputSchema": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
                       "input_topic": {"type": "string", "description": "Camera JPEG topic"}},
        "required": ["action"],
        # Core injects camera_info independently of the canvas's editable fields.
        "x-action-params": {
            "start": {"params": ["input_topic"],
                      "description": "Requires the wired camera's calibrated motus.camera/1 declaration. "
                                     "Direct callers can supply camera_info keyed by topic or as a list."},
            "stop": {"params": []}, "info": {"params": []}, "config": {"params": []},
        },
    },
    "configSchema": {"type": "object", "properties": CONFIG},
    "topic_in": [{"format": "image/jpeg", "desc": "Camera JPEG"}],
    "topic_out": [{"format": "data/json", "desc": "Marker poses in the camera optical frame (m)"},
                  {"format": "image/jpeg", "desc": "Optional Monitor image with marker IDs and XYZ axes"}],
}]


def _validate_options(options):
    if options["dictionary"] not in DICTIONARIES:
        raise ValueError("Select a supported ArUco dictionary.")
    for key in ("marker_size_m", "rate_hz"):
        value = options[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number.")
    if not 0.1 <= options["rate_hz"] <= 30:
        raise ValueError("rate_hz must be between 0.1 and 30.")
    if type(options["publish_overlay"]) is not bool:
        raise ValueError("publish_overlay must be a boolean.")


def _wired_calibration(camera_info, topic):
    declarations = camera_info if isinstance(camera_info, (dict, list, tuple)) else None
    declaration = for_topic(declarations, topic)
    message = ("左侧相机卡片未提供可用内参，请反馈给相机卡片开发者修复。 "
               f"Report missing or invalid intrinsics for {topic} to the connected camera card's developer.")
    if (declaration.get("schema") != SCHEMA or declaration.get("topic") != topic
            or declaration.get("format") not in (None, "", "image/jpeg") or declaration.get("K") is None):
        raise ValueError(message)
    try:
        calibration = Calibration.from_declaration(declaration)
    except ValueError as exc:
        raise ValueError(f"{message} {exc}") from exc
    camera_id = declaration.get("id")
    return calibration, camera_id if isinstance(camera_id, str) else ""


class _ArucoNode(Node):
    def __init__(self, topic, key, options, calibration, detector, camera_id):
        self.calibration = calibration
        suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        super().__init__(f"aruco_{suffix}")
        self.input_topic = topic
        self.options = options
        self.camera_id = camera_id
        self.detector = detector
        self.frames = LatestFrame()
        self._closed = threading.Event()
        self._publish_lock = threading.Lock()
        self._error_log = SampledLogGate()
        self.last_error = ""
        self.worker_error = ""
        base = f"{topic}/aruco/i_{suffix}"
        # Keep port 1 in place so a saved overlay connection cannot resolve to the JSON port.
        overlay_topic = f"{base}/overlay" if options["publish_overlay"] else ""
        self.outputs = [{"topic": f"{base}/poses", **TOOLS[0]["topic_out"][0]},
                        {"topic": overlay_topic, **TOOLS[0]["topic_out"][1]}]
        try:
            # Keep outputs reliable for Monitor and other DDS subscribers.
            self._poses = self.create_publisher(String, self.outputs[0]["topic"], 1)
            self._overlay = self.create_publisher(CompressedImage, overlay_topic, 1) if overlay_topic else None
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
            except Exception as exc:
                self._worker_error(frame, exc)
            else:
                self._error_log.check("processed")
            # Pop after waiting so the next iteration sees the newest frame.
            self._closed.wait(max(0, interval - (time.monotonic() - start)))

    def _frame_result(self, frame):
        stamp = frame.header.stamp
        return {"camera_id": self.camera_id, "frame_id": frame.header.frame_id,
                "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec}, "pose_status": "calibrated",
                "coordinate_frame": "camera_optical", "dictionary": self.options["dictionary"],
                "marker_size_m": self.options["marker_size_m"], "markers": []}

    def _worker_error(self, frame, error):
        detail = escape_log_text(error)
        message = f"ArUco could not process camera frame: {detail}"
        with self._publish_lock:
            if self._closed.is_set():
                return
            self.last_error = self.worker_error = message
        should_log, transition, _ = self._error_log.check("failed")
        if should_log:
            log.error("ArUco could not process camera frame: %s", detail, exc_info=transition)
        try:
            result = self._frame_result(frame)
            result["error"] = message
            self._publish(result, worker_error=message)
        except Exception as publish_error:
            if should_log:
                log.warning("Could not publish ArUco frame error: %s", escape_log_text(publish_error))

    def _process(self, frame):
        cv = self.detector.cv
        result = self._frame_result(frame)
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
        if image is not None and self._overlay is not None:
            ok, encoded = cv.imencode(".jpg", image)
            if not ok:
                raise RuntimeError("Could not encode the ArUco overlay JPEG.")
            overlay = CompressedImage()
            overlay.header = frame.header
            overlay.format = "jpeg"
            overlay.data = encoded.tobytes()
        self._publish(result, overlay)

    def _publish(self, result, overlay=None, *, worker_error=""):
        message = String()
        message.data = json.dumps(result, allow_nan=False)
        with self._publish_lock:
            if self._closed.is_set():
                return
            self._poses.publish(message)
            if overlay is not None:
                self._overlay.publish(overlay)
            self.last_error = result.get("error", "")
            self.worker_error = worker_error


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
            info = {"state": "running", "topic_in": [{"topic": node.input_topic, "format": "image/jpeg"}],
                    "topic_out": node.outputs, "last_frame_error": node.last_error, "pose_status": "calibrated",
                    "publish_overlay": node.options["publish_overlay"]}
            if not node.worker.is_alive():
                info.update(state="error", error="ArUco frame worker stopped. Restart the card.")
            elif node.worker_error:
                info.update(state="error", error=node.worker_error)
            return info
        try:
            MarkerDetector(self._defaults["dictionary"], draw_overlay=self._defaults["publish_overlay"])
        except RuntimeError as exc:
            return {"state": "error", "error": str(exc)}
        return {"state": "idle", "topic_out": [], "instances": list(self._nodes)}

    def _config(self, instance, args):
        changes = {key: args[key] for key in CONFIG if key in args}
        options = {**self._defaults, **self._overrides.get(instance, {}), **changes}
        try:
            _validate_options(options)
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
        options = {**self._defaults, **self._overrides.get(instance, {})}
        try:
            calibration, camera_id = _wired_calibration(camera_info, topic)
        except ValueError:
            # A disconnected or changed camera invalidates this instance's old metric stream.
            self._stop(key)
            raise
        node = None
        try:
            _validate_options(options)
            if (previous is not None and previous.input_topic == topic and previous.worker.is_alive()
                    and previous.camera_id == camera_id and previous.calibration.matches(calibration)
                    and previous.options == options):
                return self._info(previous)
            detector = MarkerDetector(options["dictionary"], draw_overlay=options["publish_overlay"])
            # The replacement uses the same ROS node name and output topics.
            if previous is not None:
                self._stop(key)
            node = _ArucoNode(topic, key, options, calibration, detector, camera_id)
            if self._executor.add_node(node) is False:
                raise RuntimeError("Could not register the ArUco node in the ROS executor.")
            node.worker.start()
        except Exception as exc:
            if node is not None:
                self._dispose(node)
            return {"state": "error", "error": str(exc)}
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
