"""Locate ArUco markers in a calibrated camera stream with an optional overlay."""

import hashlib
import itertools
import json
import logging
import math
import threading
import time
from io import BytesIO

import numpy as np
from PIL import Image
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
MAX_FRAME_PIXELS = 36_000_000  # Covers DCI 8K and bounds the full-resolution BGR allocation.
_NODE_SEQUENCE = itertools.count()
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
    camera_id = declaration.get("id")
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError(f"{message} Missing camera ID.")
    try:
        calibration = Calibration.from_declaration(declaration)
    except ValueError as exc:
        raise ValueError(f"{message} {exc}") from exc
    return calibration, camera_id


class _ArucoNode(Node):
    def __init__(self, topic, key, options, calibration, detector, camera_id):
        self.calibration = calibration
        suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        # A retiring incarnation may still own its ROS name during a concurrent start.
        super().__init__(f"aruco_{suffix}_{next(_NODE_SEQUENCE)}")
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

    def start_worker(self):
        with self._publish_lock:
            if not self._closed.is_set():
                self.worker.start()

    def request_stop(self):
        self._closed.set()
        self.frames.close()

    def close(self):
        with self._publish_lock:
            self.request_stop()
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
            try:
                with Image.open(BytesIO(frame.data), formats=("JPEG",)) as header:
                    width, height = header.size
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise ValueError("Could not decode camera JPEG header.") from exc
            if width * height > MAX_FRAME_PIXELS:
                raise ValueError("Camera JPEG exceeds the decoded pixel limit.")
            if (width, height) != (self.calibration.width, self.calibration.height):
                raise ValueError("Camera JPEG dimensions differ from camera calibration.")
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
        self._pending_starts = {}
        self._lock = threading.Lock()
        self._closed = False

    def get_tools(self):
        return TOOLS

    def dispatch(self, name, args):
        args = args or {}
        action = args.get("action", name)
        instance = args.get("instance_id") or ""
        topic = args.get("input_topic") or ""
        if action == "start":
            return self._start(instance, topic, args.get("camera_info"))
        if action == "info":
            return self._info(instance or topic)
        if action == "stop":
            self._stop(instance or topic)
            return {"state": "idle"}
        if action == "config":
            with self._lock:
                return self._config(instance, args)
        return None

    @staticmethod
    def _node_info(node):
        info = {"state": "running", "topic_in": [{"topic": node.input_topic, "format": "image/jpeg"}],
                "topic_out": node.outputs, "last_frame_error": node.last_error, "pose_status": "calibrated",
                "publish_overlay": node.options["publish_overlay"]}
        if not node.worker.is_alive():
            info.update(state="error", error="ArUco frame worker stopped. Restart the card.")
        elif node.worker_error:
            info.update(state="error", error=node.worker_error)
        return info

    def _info(self, key):
        with self._lock:
            instances = sorted(self._nodes.keys() | self._pending_starts.keys())
            if not key and len(instances) == 1:
                key = instances[0]
            if key in self._pending_starts:
                return {"state": "starting", "topic_out": [], "instances": instances}
            if key in self._nodes:
                return self._node_info(self._nodes[key])
            options = {**self._defaults, **self._overrides.get(key, {})}
        try:
            MarkerDetector(options["dictionary"], draw_overlay=options["publish_overlay"])
        except RuntimeError as exc:
            return {"state": "error", "error": str(exc)}
        return {"state": "idle", "topic_out": [], "instances": instances}

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
        if not isinstance(topic, str) or not topic.startswith("/") or topic.endswith("/"):
            return {"state": "error", "error": "Connect a camera or provide an absolute camera JPEG topic."}
        key = instance or topic
        calibration_error = None
        with self._lock:
            if self._closed or self._executor is None:
                return {"state": "error", "error": "ArUco ROS executor is unavailable."}
            previous = self._nodes.get(key)
            options = {**self._defaults, **self._overrides.get(instance, {})}
            try:
                _validate_options(options)
            except (ValueError, OverflowError) as exc:
                return {"state": "error", "error": str(exc)}
            try:
                calibration, camera_id = _wired_calibration(camera_info, topic)
            except ValueError as exc:
                # A failed rewire to another topic cannot invalidate the still-calibrated source.
                if previous is not None and previous.input_topic != topic:
                    raise
                # For the active topic, stop emitting measurements as soon as it loses its calibration.
                calibration_error = exc
                self._pending_starts.pop(key, None)
            else:
                if (previous is not None and previous.input_topic == topic and previous.worker.is_alive()
                        and previous.camera_id == camera_id and previous.calibration.matches(calibration)
                        and previous.options == options):
                    return self._node_info(previous)
                ticket = object()
                self._pending_starts[key] = ticket
            self._nodes.pop(key, None)
            if previous is not None:
                previous.request_stop()

        if calibration_error is not None:
            if previous is not None:
                self._dispose(previous)
            raise calibration_error
        node = None
        registered = False
        try:
            if previous is not None:
                self._dispose(previous)
            detector = MarkerDetector(options["dictionary"], draw_overlay=options["publish_overlay"])
            with self._lock:
                if self._pending_starts.get(key) is not ticket:
                    return {"state": "idle"}
                node = _ArucoNode(topic, key, options, calibration, detector, camera_id)
                if self._executor.add_node(node) is False:
                    raise RuntimeError("Could not register the ArUco node in the ROS executor.")
                self._nodes[key] = node
                registered = True
            node.start_worker()
            with self._lock:
                return self._node_info(node) if self._nodes.get(key) is node else {"state": "idle"}
        except Exception as exc:
            # After registration, a concurrent stop or replacement owns cleanup if it took the node.
            with self._lock:
                if registered and self._nodes.get(key) is node:
                    self._nodes.pop(key)
                elif registered:
                    node = None
            if node is not None:
                self._dispose(node)
            return {"state": "error", "error": str(exc)}
        finally:
            with self._lock:
                if self._pending_starts.get(key) is ticket:
                    del self._pending_starts[key]

    def _dispose(self, node):
        node.close()
        dispose_node(self._executor, node, label=node.input_topic)

    def _stop(self, key=""):
        with self._lock:
            if key:
                self._pending_starts.pop(key, None)
                nodes = [self._nodes.pop(key)] if key in self._nodes else []
            else:
                self._pending_starts.clear()
                nodes = list(self._nodes.values())
                self._nodes.clear()
            for node in nodes:
                node.request_stop()
        for node in nodes:
            self._dispose(node)

    def shutdown(self):
        with self._lock:
            self._closed = True
        self._stop()
