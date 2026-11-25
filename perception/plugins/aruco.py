"""ArUco markers in a camera stream.

Marker IDs and pixel centres work with any camera. A metric pose needs a focal
length, which comes from the camera card's `motus.camera/1` declaration: its K
(`pose_status: calibrated`), or K derived from its field of view
(`approximate`). A camera that declares neither still gets IDs
(`uncalibrated`) — G1, Go2 and the browser camera declare nothing, and
refusing to start there would make the card unusable on half the fleet.
"""

import collections
import itertools
import json
import logging
import math
import threading
import time

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.aruco_runtime import DEFAULT_DICTIONARY, DICTIONARIES, MarkerDetector, intrinsics, unavailable_reason
from plugins.camera_info import for_topic
from utils.latest_frame import LatestFrame
from utils.log_sampling import SampledLogGate, escape_log_text
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node


log = logging.getLogger(__name__)
DEFAULTS = {"dictionary": DEFAULT_DICTIONARY, "marker_size_m": 0.1, "rate_hz": 5.0, "publish_overlay": True}
TOOLS = [{
    "name": "aruco",
    "type": "processor",
    "multiInstance": True,
    "description": "ArUco 标记定位：输出画面里每个标记的 ID，以及相对相机的位置（米）",
    "inputSchema": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
                       "input_topic": {"type": "string", "description": "Camera JPEG topic"}},
        "required": ["action"],
    },
    "configSchema": {"type": "object", "properties": {
        "dictionary": {"type": "string", "title": "ArUco dictionary", "enum": list(DICTIONARIES),
                       "default": DEFAULT_DICTIONARY, "scope": "instance",
                       "description": "打印标记时用的词典。4X4 格子粗、远处小标记也能认；6X6_250 可编 250 个 ID"},
        "marker_size_m": {"type": "number", "title": "Marker side (m)", "exclusiveMinimum": 0,
                          "default": DEFAULTS["marker_size_m"], "scope": "instance",
                          "description": "黑框外边长（米），不含白边。填错会按比例放大或缩小距离"},
        "rate_hz": {"type": "number", "title": "Rate (Hz)", "minimum": 0.1, "maximum": 30,
                    "default": DEFAULTS["rate_hz"], "scope": "instance"},
        "publish_overlay": {"type": "boolean", "title": "Publish overlay", "default": True, "scope": "instance",
                            "description": "发布画了标记和坐标轴的图像；关掉只发 JSON"},
    }},
    "topic_in": [{"format": "image/jpeg", "desc": "Camera JPEG"}],
    "topic_out": [{"format": "data/json", "desc": "Marker IDs and poses in the camera optical frame"},
                  {"format": "image/jpeg", "desc": "Overlay with marker IDs and axes"}],
}]


def output_topics(input_topic, publish_overlay):
    # Both ports stay listed even when idle or with the overlay off, so a saved
    # connection to port 1 never shifts onto the JSON port.
    return [{**TOOLS[0]["topic_out"][0], "topic": f"{input_topic}/aruco" if input_topic else ""},
            {**TOOLS[0]["topic_out"][1],
             "topic": f"{input_topic}/aruco/overlay" if input_topic and publish_overlay else ""}]


def validate(options):
    if options["dictionary"] not in DICTIONARIES:
        raise ValueError(f"dictionary must be one of {', '.join(DICTIONARIES)}")
    for key in ("marker_size_m", "rate_hz"):
        options[key] = float(options[key])
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"{key} must be a positive number")
    if not 0.1 <= options["rate_hz"] <= 30:
        raise ValueError("rate_hz must be between 0.1 and 30")
    options["publish_overlay"] = bool(options["publish_overlay"])
    return options


class _ArucoNode(Node):
    def __init__(self, name, topic, options, declaration):
        super().__init__(name)
        self.input_topic = topic
        self.options = options
        self.declaration = declaration
        self.outputs = output_topics(topic, options["publish_overlay"])
        self.detector = MarkerDetector(options["dictionary"])
        self.pose_status = ""
        self.last_error = ""
        self.frames = LatestFrame()
        self._intrinsics = {}
        self._had_markers = False
        self._stop = threading.Event()
        self._log_gate = SampledLogGate()
        self._poses = self.create_publisher(String, self.outputs[0]["topic"], 10)
        overlay = self.outputs[1]["topic"]
        self._overlay = self.create_publisher(CompressedImage, overlay, 1) if overlay else None
        self.create_subscription(CompressedImage, topic, self.frames.push, CAMERA_QOS)
        self.worker = threading.Thread(target=self._run, name=name, daemon=True)

    def request_stop(self):
        self._stop.set()
        self.frames.close()

    def join(self):
        if self.worker.ident is not None:
            self.worker.join(timeout=2)

    def _run(self):
        interval = 1.0 / self.options["rate_hz"]
        while not self._stop.is_set():
            frame = self.frames.pop(timeout=1.0)
            if frame is None:
                continue
            started = time.monotonic()
            try:
                self._process(frame)
                self.last_error = ""
            except Exception as exc:
                self.last_error = escape_log_text(exc)
                should_log, transition, _ = self._log_gate.check("failed")
                if should_log:
                    log.warning("[aruco] %s: %s", self.input_topic, self.last_error, exc_info=transition)
            self._stop.wait(max(0.0, interval - (time.monotonic() - started)))

    def _process(self, frame):
        cv = self.detector.cv
        image = cv.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not decode the camera JPEG")
        height, width = image.shape[:2]
        if (width, height) not in self._intrinsics:
            self._intrinsics[(width, height)] = intrinsics(self.declaration, width, height)
        matrix, distortion, self.pose_status = self._intrinsics[(width, height)]
        markers = self.detector.detect(image, self.options["marker_size_m"], matrix, distortion,
                                       draw=self._overlay is not None)
        # Publish while markers are in view, plus one empty message when they
        # leave, rather than an empty array at every frame.
        if markers or self._had_markers:
            stamp = frame.header.stamp
            message = String()
            message.data = json.dumps({
                "camera_id": (self.declaration or {}).get("id", ""),
                "frame_id": frame.header.frame_id,
                "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
                "coordinate_frame": "camera_optical", "pose_status": self.pose_status,
                "dictionary": self.options["dictionary"], "marker_size_m": self.options["marker_size_m"],
                "markers": markers})
            self._poses.publish(message)
        self._had_markers = bool(markers)
        if self._overlay is not None:
            ok, encoded = cv.imencode(".jpg", image)
            if ok:
                overlay = CompressedImage()
                overlay.header = frame.header
                overlay.format = "jpeg"
                overlay.data = encoded.tobytes()
                self._overlay.publish(overlay)


class ArucoPlugin:
    PREFIX = "aruco"

    def __init__(self, cfg, executor):
        self._executor = executor
        self._defaults = validate({**DEFAULTS, **{k: cfg[k] for k in DEFAULTS if k in cfg}})
        self._instance_cfg = {}
        self._nodes = {}
        # Bumped by stop, so a start that was building its node meanwhile
        # discards it instead of registering a node nothing will stop.
        self._stops = collections.Counter()
        self._names = itertools.count()
        self._lock = threading.Lock()

    def get_tools(self):
        return TOOLS

    def dispatch(self, name, args):
        args = args or {}
        action = args.get("action", name)
        instance = args.get("instance_id") or ""
        topic = args.get("input_topic") or ""
        if action == "start":
            return self._start(instance or topic, instance, topic, args.get("camera_info"))
        if action == "stop":
            self._stop(instance or topic)
            return {"state": "idle"}
        if action == "info":
            return self._info(instance or topic, instance, topic)
        if action == "config":
            return self._config(instance, args)
        return None

    def _options(self, instance):
        return {**self._defaults, **self._instance_cfg.get(instance, {})}

    @staticmethod
    def _node_info(node):
        info = {"state": "running", "topic_in": [{"topic": node.input_topic, "format": "image/jpeg"}],
                "topic_out": node.outputs, "pose_status": node.pose_status, "last_error": node.last_error}
        if not node.worker.is_alive():
            info.update(state="error", error="ArUco worker stopped; restart the card")
        return info

    def _info(self, key, instance, topic):
        with self._lock:
            node = self._nodes.get(key)
        if node is not None:
            return self._node_info(node)
        options = self._options(instance)
        info = {"state": "idle", "topic_out": output_topics(topic, options["publish_overlay"])}
        reason = unavailable_reason()
        if reason:
            info.update(state="error", error=reason)
        return info

    def _config(self, instance, args):
        changes = {k: args[k] for k in DEFAULTS if args.get(k) not in (None, "")}
        options = validate({**self._options(instance), **changes})
        with self._lock:
            if instance:
                self._instance_cfg[instance] = {**self._instance_cfg.get(instance, {}),
                                                **{k: options[k] for k in changes}}
            else:
                self._defaults = options
        return {"state": "configured", "message": "Takes effect on the next start"}

    def _start(self, key, instance, topic, camera_info):
        reason = unavailable_reason()
        if reason:
            return {"state": "error", "error": reason}
        if not topic.startswith("/"):
            return {"state": "error", "error": "Connect a camera to the ArUco card"}
        options = self._options(instance)
        declaration = for_topic(camera_info, topic) or None
        with self._lock:
            current = self._nodes.get(key)
            if (current is not None and current.worker.is_alive() and current.input_topic == topic
                    and current.options == options and current.declaration == declaration):
                return self._node_info(current)
            previous = self._nodes.pop(key, None)
            stops = self._stops[key], self._stops[""]
        if previous is not None:
            self._dispose(previous)
        try:
            node = _ArucoNode(f"aruco_{next(self._names)}", topic, options, declaration)
        except Exception as exc:
            return {"state": "error", "error": escape_log_text(exc)}
        with self._lock:
            registered = (self._stops[key], self._stops[""]) == stops and key not in self._nodes
            if registered:
                self._executor.add_node(node)
                self._nodes[key] = node
        if not registered:
            node.destroy_node()
            return {"state": "idle"}
        node.worker.start()
        return self._node_info(node)

    def _dispose(self, node):
        node.request_stop()
        node.join()
        dispose_node(self._executor, node, label=f"aruco {node.input_topic}")

    def _stop(self, key=""):
        with self._lock:
            self._stops[key] += 1  # "" stops every instance
            keys = [key] if key else list(self._nodes)
            nodes = [self._nodes.pop(k) for k in keys if k in self._nodes]
        for node in nodes:
            self._dispose(node)

    def shutdown(self):
        self._stop()
