"""`height` card — measure a person's or a floor-standing object's height.

One card, read-only: it subscribes to the metric depth map a camera card
publishes (and, optionally, to one detection stream that says *which* thing to
measure), and answers a tool call with a number in metres plus the evidence it
was derived from. It never publishes control, never declares `x-resource`, and
never blocks on another card's action.

What it does **not** do: it does not invent a height when the picture does not
support one. No depth, no intrinsics, no floor and no mount calibration all
return a `reason` instead of a plausible-looking number — the failure mode this
project keeps paying for is a confident wrong number, not a missing one.

Two ways to get "above the floor":

* **floor** — fit the floor plane in the same frame; the target's distance to
  that plane is its height. Needs the floor in view, and automatically absorbs
  the robot's own posture (a stand whose pitch moved 0.5 degrees is measured
  from the floor in *this* frame, not from a stored angle).
* **mount** — the camera's calibrated height and mounting pitch. Needs no floor
  in view, but is only as good as that one-time calibration.

`auto` tries the floor first and falls back to the mount. The reply says which
one was used, because the two have very different error sources.

The card is disabled in `config.yaml` by default: it is a new capability and an
existing deployment must not start measuring anything until an operator wires a
depth card to it and enables it.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time

import numpy as np

from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins import height_math as geometry
from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)

# Sensor streams are BEST_EFFORT on this platform end to end. A RELIABLE
# subscriber against a BEST_EFFORT publisher never receives anything, with no
# error anywhere — see the note in actucore/plugins/navi/plugin.py.
_SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1,
                         durability=DurabilityPolicy.VOLATILE)
_RESULT_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1,
                         durability=DurabilityPolicy.VOLATILE)

# How long a measurement may take before it answers with what it has. The
# driver publishes at 30 Hz, but the observed delivery rate on Bumi has been as
# low as ~1.8 Hz, so five frames can be three seconds. Startup and the frame
# window are bounded on purpose: a tool call that hangs is worse than a tool
# call that says "unstable".
_MEASURE_TIMEOUT_S = 3.0
_MAX_FRAME_SPREAD_M = 0.03
# Floor fits outside this band are not a floor in front of a standing robot.
_MIN_PLANE_HEIGHT_M = 0.20
_MAX_PLANE_HEIGHT_M = 3.0

# Defaults are also the `configSchema` defaults and the `config.yaml` values.
# They have to agree: agent-core sends every schema field (default included) on
# every `config` call, so a schema default that disagrees with the file
# silently wins. The navi card has the long version of this note.
_CONFIG_DEFAULTS = {
    "camera_height_m": 0.0,      # 0 = not calibrated
    "camera_pitch_deg": 0.0,     # positive = optical axis below horizontal
    "camera_roll_deg": 0.0,
    "fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0,   # 0 = unset
    "frames": 5,
    "max_range_m": 4.5,
    "allow_fov_intrinsics": False,
    # Card-only settings: deliberately not in configSchema, so a schema default
    # cannot silently override them (same reasoning as actucore's navi card).
    "max_detection_age_s": 0.8,
    "min_valid_ratio": 0.5,
    "calibration_frames": 12,
}

_TOOL = {
    "name": "height",
    "type": "processor",
    "multiInstance": False,
    "description": (
        "Measure the height of one person or one floor-standing object from "
        "the metric depth camera. Requires the object to be still and fully "
        "in the picture (a person typically 2-3.5 m away). Returns the height "
        "in metres with an uncertainty and the mode used, or a reason it "
        "could not measure — it never guesses."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "measure", "calibrate", "info",
                         "config"],
            },
            "target": {
                "type": "string",
                "description": (
                    "What to measure: 'person', 'object', 'auto' (default), or "
                    "a class name from the vop detections (e.g. 'box'). A "
                    "person is located from the pose card's boxes, an object "
                    "from the vop card's boxes."
                ),
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional normalised [x1,y1,x2,y2] in 0..1. For testing or "
                    "when an external caller already knows the target box; "
                    "normally the detection input supplies it."
                ),
            },
        },
        "required": ["action"],
        "x-action-params": {
            "measure": {
                "params": ["target", "bbox"],
                "description": (
                    "Measure one target's height. The target must be still and "
                    "its top (person: head, object: top face) must be inside "
                    "the picture with a few pixels of margin. Returns "
                    "height_m / uncertainty_m / mode / frames_used, or an "
                    "error with a reason (no_camera_info, "
                    "no_ground_reference, target_clipped, too_far, "
                    "ambiguous_target...)."
                ),
            },
            "calibrate": {
                "params": [],
                "description": (
                    "One-time mount calibration from the floor in view: "
                    "returns camera_height_m / camera_pitch_deg / "
                    "camera_roll_deg and a suggested config block. Save those "
                    "values in the card's configuration; the camera is fixed, "
                    "so this is not needed per measurement."
                ),
            },
        },
    },
    "configSchema": {
        "type": "object",
        "properties": {
            "camera_height_m": {
                "type": "number", "default": 0.0, "scope": "shared",
                "minimum": 0.0, "maximum": 3.0,
                "title": "Camera height above floor (m)",
                "description": (
                    "0 = not calibrated (floor mode only). Set this from the "
                    "`calibrate` action or a tape measure; the camera is "
                    "fixed, so it is measured once per robot."
                ),
            },
            "camera_pitch_deg": {
                "type": "number", "default": 0.0, "scope": "shared",
                "minimum": -45.0, "maximum": 45.0,
                "title": "Camera pitch (deg, positive = looking down)",
                "description": "Mounting pitch from the `calibrate` action; "
                               "used by mount mode when the floor is not in "
                               "view.",
            },
            "camera_roll_deg": {
                "type": "number", "default": 0.0, "scope": "shared",
                "minimum": -45.0, "maximum": 45.0,
                "title": "Camera roll (deg)",
                "description": "Mounting roll from the `calibrate` action.",
            },
            "frames": {
                "type": "integer", "default": 5, "scope": "shared",
                "minimum": 2, "maximum": 30,
                "title": "Frames averaged per measurement",
                "description": "The target must hold still for this many "
                               "frames; the spread across them must stay "
                               "under 3 cm.",
            },
            "max_range_m": {
                "type": "number", "default": 4.5, "scope": "shared",
                "minimum": 0.5, "maximum": 10.0,
                "title": "Maximum depth range (m)",
                "description": "Depth beyond this is ignored for the floor fit "
                               "and the target; the D435i is most accurate "
                               "inside about 3 m.",
            },
            "allow_fov_intrinsics": {
                "type": "boolean", "default": False, "scope": "shared",
                "advanced": True,
                "title": "Allow FOV-derived intrinsics",
                "description": (
                    "Off by default. When on, a camera declaration with only "
                    "half-FOV angles (no K) is accepted by assuming a centred "
                    "principal point; that is a guess and inflates the error."
                ),
            },
            "fx": {"type": "number", "default": 0.0, "scope": "shared",
                   "advanced": True, "minimum": 0.0,
                   "title": "Manual fx (px), 0 = use the camera's K"},
            "fy": {"type": "number", "default": 0.0, "scope": "shared",
                   "advanced": True, "minimum": 0.0,
                   "title": "Manual fy (px), 0 = use the camera's K"},
            "cx": {"type": "number", "default": 0.0, "scope": "shared",
                   "advanced": True, "minimum": 0.0,
                   "title": "Manual cx (px), 0 = use the camera's K"},
            "cy": {"type": "number", "default": 0.0, "scope": "shared",
                   "advanced": True, "minimum": 0.0,
                   "title": "Manual cy (px), 0 = use the camera's K"},
        },
        "required": [],
    },
    # Declared rather than discovered: a card that names no port until it runs
    # leaves the canvas with "missing topic" whenever it fails to start.
    "topic_in": [
        {"format": "image/depth-zlib",
         "desc": "Metric depth, 640x480 uint16 mm, aligned to the colour grid"},
        {"format": "data/json",
         "desc": "Optional detection: pose /poses (people) or vop /objects"},
    ],
    "topic_out": [
        {"format": "data/json",
         "desc": "Last measurement or calibration result (dashboard / bus)"},
    ],
}

TOOLS = [_TOOL]


# ── ROS node ─────────────────────────────────────────────────────────────────

class _HeightNode(Node):
    """Owns the two subscriptions and the frame-window condition variable.

    Deliberately does not touch the scene until the card is asked to measure:
    the callbacks only keep the newest depth payload and the newest detection,
    so the card costs one 614 KB buffer and one dict while idle. No worker
    thread — a measurement is a bounded loop over the next N frames, and a
    thread would only add a second way to be wrong about freshness.
    """

    def __init__(self, depth_topic: str, detection_topic: str,
                 result_topic: str):
        super().__init__("perception_height")
        self._depth_topic = depth_topic
        self._detection_topic = detection_topic
        self._condition = threading.Condition()
        self._depth_seq = 0
        self._depth_bytes = None
        self._depth_arrival = 0.0
        self._detection = None
        self._detection_arrival = 0.0
        self._depth_sub = None
        self._detection_sub = None
        self._result_pub = (
            self.create_publisher(String, result_topic, _RESULT_QOS)
            if result_topic else None)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> dict:
        if self._depth_sub is None:
            self._depth_sub = self.create_subscription(
                CompressedImage, self._depth_topic, self._on_depth,
                _SENSOR_QOS)
        if self._detection_topic and self._detection_sub is None:
            self._detection_sub = self.create_subscription(
                String, self._detection_topic, self._on_detection, _SENSOR_QOS)
        return {"state": "running"}

    def stop(self) -> dict:
        # Subscriptions are intentionally not destroyed here: utils.
        # ros_lifecycle.dispose_node takes the node off the executor *before*
        # destroy_node(). Destroying a subscription whose handle is still in
        # the executor's wait list raises InvalidHandle and kills the whole
        # perception spin thread (see the note in plugins/visual_depth.py).
        self._depth_sub = None
        self._detection_sub = None
        return {"state": "idle"}

    # -- callbacks ---------------------------------------------------------

    def _on_depth(self, message) -> None:
        with self._condition:
            self._depth_bytes = bytes(message.data)
            self._depth_arrival = time.monotonic()
            self._depth_seq += 1
            self._condition.notify_all()

    def _on_detection(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self._condition:
            self._detection = payload
            self._detection_arrival = time.monotonic()

    # -- frame window ------------------------------------------------------

    def depth_sequence(self) -> int:
        with self._condition:
            return self._depth_seq

    def wait_for_depth(self, after_seq: int, timeout_s: float):
        """The next depth payload after `after_seq`, or `(None, seq)`.

        Waiting for a *newer* sequence rather than taking the latest frame is
        the difference between "five frames" and "the same frame five times":
        a static person must still be seen across time for the spread to mean
        anything.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while self._depth_seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, self._depth_seq
                self._condition.wait(remaining)
            return (self._depth_bytes, self._depth_arrival), self._depth_seq

    def latest_detection(self):
        with self._condition:
            return self._detection, self._detection_arrival

    def publish_result(self, result: dict) -> None:
        if self._result_pub is None:
            return
        message = String()
        message.data = json.dumps(result, ensure_ascii=False)
        self._result_pub.publish(message)


# ── card ────────────────────────────────────────────────────────────────────

class HeightPlugin:
    PREFIX = "height"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._cfg = dict(_CONFIG_DEFAULTS)
        self._merge_config(plugin_cfg or {})
        self._instances: dict = {}
        self._lock = threading.RLock()

    # -- tool contract -----------------------------------------------------

    def get_tools(self) -> list:
        return list(TOOLS)

    def dispatch(self, name: str, args: dict):
        args = args or {}
        action = args.get("action") or name
        if action == "start":
            return self._start(args)
        if action == "stop":
            return self._stop(args)
        if action == "measure":
            return self._measure(args)
        if action == "calibrate":
            return self._calibrate(args)
        if action == "info":
            return self._info(args)
        if action == "config":
            return self._config(args)
        return None

    # -- configuration -----------------------------------------------------

    def _merge_config(self, incoming: dict) -> None:
        for key, default in _CONFIG_DEFAULTS.items():
            if key not in incoming:
                continue
            value = incoming[key]
            try:
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int):
                    value = int(value)
                else:
                    value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number")
            self._cfg[key] = value
        self._validate_config()

    def _validate_config(self) -> None:
        if not 0.0 <= self._cfg["camera_height_m"] <= 3.0:
            raise ValueError("camera_height_m must be in [0, 3]")
        for key in ("camera_pitch_deg", "camera_roll_deg"):
            if not -45.0 <= self._cfg[key] <= 45.0:
                raise ValueError(f"{key} must be in [-45, 45]")
        for key in ("fx", "fy", "cx", "cy"):
            if self._cfg[key] < 0.0:
                raise ValueError(f"{key} must be >= 0")
        if not 2 <= self._cfg["frames"] <= 30:
            raise ValueError("frames must be in [2, 30] (a single frame has "
                             "no spread to check)")
        if not 0.5 <= self._cfg["max_range_m"] <= 10.0:
            raise ValueError("max_range_m must be in [0.5, 10]")
        if not 0.05 <= self._cfg["max_detection_age_s"] <= 5.0:
            raise ValueError("max_detection_age_s must be in [0.05, 5]")
        if not 0.0 <= self._cfg["min_valid_ratio"] <= 1.0:
            raise ValueError("min_valid_ratio must be in [0, 1]")
        if not 2 <= self._cfg["calibration_frames"] <= 60:
            raise ValueError("calibration_frames must be in [2, 60]")

    def _public_config(self) -> dict:
        return {key: self._cfg[key] for key in _CONFIG_DEFAULTS}

    def _config(self, args: dict):
        body = {key: value for key, value in args.items()
                if key in _CONFIG_DEFAULTS}
        if not body:
            return {"status": "configured", "config": self._public_config()}
        try:
            self._merge_config(body)
        except ValueError as error:
            return {"status": "error", "message": str(error)}
        return {"status": "configured", "config": self._public_config()}

    # -- lifecycle ---------------------------------------------------------

    def _start(self, args: dict):
        instance_id = str(args.get("instance_id") or "")
        topics = _input_topics(args)
        depth_topic = next((t for t in topics if _is_depth_topic(t)), "")
        detection_topic = next((t for t in topics if _is_detection_topic(t)), "")
        if not depth_topic:
            return _error(
                "no_depth_input",
                "the height card needs a depth topic. Wire the camera card's "
                "`image/depth-zlib` output into this card, then start the "
                "project again.")

        declarations = args.get("camera_info") or {}
        declaration = geometry.declaration_for(declarations, depth_topic)
        try:
            intrinsics = geometry.resolve_intrinsics(declaration, self._cfg)
        except geometry.GeometryError as error:
            return _error(error.reason, error.message)

        detection_size = (geometry.DEPTH_WIDTH, geometry.DEPTH_HEIGHT)
        if detection_topic:
            detection_decl = geometry.declaration_for(declarations,
                                                      detection_topic)
            if detection_decl.get("width") and detection_decl.get("height"):
                detection_size = (int(detection_decl["width"]),
                                  int(detection_decl["height"]))

        degraded = list(intrinsics.get("notes") or [])
        if self._cfg["camera_height_m"] <= 0.0:
            degraded.append(
                "no mount calibration: floor mode only; run `calibrate` or set "
                "camera_height_m")
        if not detection_topic:
            degraded.append(
                "no detection input: measure requires an explicit normalised "
                "bbox")

        self._retire(instance_id)
        result_topic = geometry.output_topic_for(depth_topic)
        node = _HeightNode(depth_topic, detection_topic, result_topic)
        self._executor.add_node(node)
        node.start()
        instance = {
            "instance_id": instance_id,
            "node": node,
            "depth_topic": depth_topic,
            "detection_topic": detection_topic,
            "detection_size": detection_size,
            "result_topic": result_topic,
            "intrinsics": intrinsics,
            "degraded": degraded,
            "last_result": None,
            "last_calibration": None,
        }
        with self._lock:
            self._instances[instance_id] = instance
        log.info("[height] started: depth=%s detection=%s intrinsics=%s",
                 depth_topic, detection_topic or "(none)",
                 intrinsics["source"])
        return {
            "state": "running",
            "instance_id": instance_id,
            "inputs": {"depth": depth_topic,
                       "detection": detection_topic or None},
            "intrinsics": {key: intrinsics[key]
                           for key in ("fx", "fy", "cx", "cy",
                                       "width", "height", "source")},
            "calibration": self._calibration(),
            "degraded": degraded,
        }

    def _stop(self, args: dict):
        instance_id = str(args.get("instance_id") or "")
        if instance_id:
            self._retire(instance_id)
        else:
            with self._lock:
                ids = list(self._instances)
            for key in ids:
                self._retire(key)
        return {"state": "idle"}

    def _retire(self, instance_id: str) -> None:
        with self._lock:
            instance = self._instances.pop(instance_id, None)
        if instance is None:
            return
        node = instance["node"]
        try:
            node.stop()
        except Exception:                                     # noqa: BLE001
            log.exception("[height] node stop failed")
        dispose_node(self._executor, node, label=f"height/{instance_id or 'default'}")
        log.info("[height] stopped: %s", instance_id or "(default)")

    # -- measurement -------------------------------------------------------

    def _measure(self, args: dict):
        instance = self._instance(args)
        if instance is None:
            return _error("not_running", "the height card is not running; "
                                         "start it from the canvas first")
        try:
            target = self._resolve_target(instance, args)
        except geometry.GeometryError as error:
            return _error(error.reason, error.message)

        node = instance["node"]
        intrinsics = instance["intrinsics"]
        samples = []
        last_failure = None
        baseline = node.depth_sequence()
        deadline = time.monotonic() + _MEASURE_TIMEOUT_S
        for _ in range(int(self._cfg["frames"])):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            frame, sequence = node.wait_for_depth(baseline,
                                                  min(1.5, remaining))
            if frame is None:
                break
            baseline = sequence
            try:
                samples.append(self._sample(instance, frame, target))
            except geometry.GeometryError as error:
                last_failure = error
        if len(samples) < 2:
            return _error(
                last_failure.reason if last_failure else "insufficient_frames",
                last_failure.message if last_failure else
                f"only {len(samples)} usable frame(s); the target must hold "
                f"still while the depth frames arrive")

        _, spread = geometry.median_and_spread(
            [sample["height_m"] for sample in samples])
        if spread > _MAX_FRAME_SPREAD_M:
            return _error(
                "unstable",
                f"the height moved {spread * 100:.1f} cm across "
                f"{len(samples)} frames; ask the target to stand still (and, "
                f"for a person, to stop shifting weight) and measure again")
        valid_ratio = float(np.median([sample["valid_ratio"]
                                       for sample in samples]))
        if valid_ratio < float(self._cfg["min_valid_ratio"]):
            return _error(
                "no_depth_at_target",
                f"only {valid_ratio * 100:.0f}% of the sampled top columns "
                f"had a depth reading")
        height_m, _ = geometry.median_and_spread(
            [sample["height_m"] for sample in samples])
        if not geometry.plausible_height(height_m):
            return _error(
                "implausible_height",
                f"measured {height_m:.2f} m, outside the "
                f"{geometry.MIN_HEIGHT_M:.2f}-{geometry.MAX_HEIGHT_M:.2f} m "
                f"this card will report")
        modes = sorted({sample["mode"] for sample in samples})
        result = {
            "state": "ok",
            "height_m": round(height_m, 3),
            "uncertainty_m": round(max(0.02, spread / 2.0), 3),
            "height_definition": ("floor_to_top_of_head"
                                  if target["kind"] == "person"
                                  else "floor_to_top_of_object"),
            "target": {
                "kind": target["kind"],
                "label": target["label"],
                "track_id": target.get("track_id"),
                "bbox_px": [round(float(v), 1) for v in target["bbox_px"]],
                "candidate_count": target.get("candidate_count", 1),
            },
            "mode": modes[0] if len(modes) == 1 else "mixed",
            "camera_height_m": _reported_camera_height(
                instance, self._cfg["camera_height_m"]),
            "frames_used": len(samples),
            "spread_m": round(spread, 3),
            "valid_ratio": round(valid_ratio, 3),
            "distance_m": round(float(np.median([sample["z"] for sample in samples])), 3),
            "degraded": list(instance["degraded"]),
            "measured_at": _timestamp(),
        }
        instance["last_result"] = result
        node.publish_result(result)
        return result

    def _sample(self, instance: dict, frame, target: dict) -> dict:
        payload, _ = frame
        depth = geometry.decode_depth(payload)
        intrinsics = instance["intrinsics"]
        bbox = target["bbox_px"]
        top = geometry.top_point(depth, bbox, intrinsics)
        plane = None
        mode = ""
        try:
            plane = geometry.floor_plane_from_depth(
                depth, intrinsics, max_range_m=self._cfg["max_range_m"],
                exclude_bbox=bbox)
            usable = (
                _MIN_PLANE_HEIGHT_M <= plane["height_m"] <= _MAX_PLANE_HEIGHT_M
                and plane["residual_m"] <= geometry.MAX_PLANE_RESIDUAL_M
                and plane["inlier_ratio"] >= geometry.MIN_PLANE_INLIERS)
            if usable:
                height_m = geometry.height_above_plane(top["point"], plane)
                mode = "floor"
            else:
                raise geometry.GeometryError(
                    "no_ground_reference", "the fitted plane is not a floor")
        except geometry.GeometryError:
            if self._cfg["camera_height_m"] > 0.0:
                height_m = geometry.height_from_mount(
                    top["point"], self._cfg["camera_height_m"],
                    math.radians(self._cfg["camera_pitch_deg"]),
                    math.radians(self._cfg["camera_roll_deg"]))
                mode = "mount"
            else:
                raise geometry.GeometryError(
                    "no_ground_reference",
                    "no usable floor in view and no mount calibration; set "
                    "camera_height_m/camera_pitch_deg once (or run `calibrate` "
                    "somewhere the floor is visible)")
        if not geometry.plausible_height(height_m):
            raise geometry.GeometryError(
                "implausible_height",
                f"frame measured {height_m:.2f} m, outside the "
                f"{geometry.MIN_HEIGHT_M:.2f}-{geometry.MAX_HEIGHT_M:.2f} m "
                f"band")
        return {
            "height_m": height_m,
            "mode": mode,
            "z": top["z"],
            "valid_ratio": top["valid_ratio"],
            "plane": plane,
        }

    def _resolve_target(self, instance: dict, args: dict) -> dict:
        bbox = args.get("bbox")
        if bbox is not None:
            try:
                values = [float(v) for v in bbox]
            except (TypeError, ValueError):
                raise geometry.GeometryError(
                    "bad_bbox", "bbox must be four numbers in 0..1")
            if len(values) != 4 or any(not 0.0 <= v <= 1.0 for v in values):
                raise geometry.GeometryError(
                    "bad_bbox", "bbox must be four numbers in 0..1")
            width, height = geometry.DEPTH_WIDTH, geometry.DEPTH_HEIGHT
            return {
                "bbox_px": geometry.clamp_bbox(
                    [values[0] * width, values[1] * height,
                     values[2] * width, values[3] * height], width, height),
                "kind": "manual",
                "label": "manual",
                "track_id": None,
                "candidate_count": 1,
            }

        if not instance["detection_topic"]:
            raise geometry.GeometryError(
                "no_detection",
                "no detection input and no explicit bbox. Wire the pose card "
                "(people) or the vop card (objects) into this card's "
                "data/json input.")
        node = instance["node"]
        payload, arrival = node.latest_detection()
        if payload is None:
            raise geometry.GeometryError(
                "no_detection",
                "the detection input has not published anything yet")
        if time.monotonic() - arrival > float(self._cfg["max_detection_age_s"]):
            raise geometry.GeometryError(
                "stale_detection",
                f"the newest detection is {time.monotonic() - arrival:.1f} s "
                f"old; the detector may have stopped")
        source = "pose" if instance["detection_topic"].endswith("/poses") else "vop"
        wanted = str(args.get("target") or "auto").strip()
        if source == "pose" and wanted.lower() not in ("", "auto", "person"):
            wanted = "auto"
        return geometry.pick_detection(
            payload, source, wanted, instance["detection_size"],
            (geometry.DEPTH_WIDTH, geometry.DEPTH_HEIGHT))

    # -- calibration -------------------------------------------------------

    def _calibrate(self, args: dict):
        instance = self._instance(args)
        if instance is None:
            return _error("not_running", "the height card is not running; "
                                         "start it from the canvas first")
        node = instance["node"]
        intrinsics = instance["intrinsics"]
        wanted = int(self._cfg["calibration_frames"])
        planes = []
        baseline = node.depth_sequence()
        deadline = time.monotonic() + 6.0
        while len(planes) < wanted and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            frame, sequence = node.wait_for_depth(baseline,
                                                  min(1.5, remaining))
            if frame is None:
                break
            baseline = sequence
            try:
                depth = geometry.decode_depth(frame[0])
                plane = geometry.floor_plane_from_depth(
                    depth, intrinsics, max_range_m=self._cfg["max_range_m"])
            except geometry.GeometryError:
                continue
            if (_MIN_PLANE_HEIGHT_M <= plane["height_m"] <= _MAX_PLANE_HEIGHT_M
                    and plane["residual_m"] <= geometry.MAX_PLANE_RESIDUAL_M):
                planes.append(plane)
        if len(planes) < 3:
            return _error(
                "no_ground_reference",
                "not enough frames with a visible floor. Put the robot where "
                "the floor in front of it is in view (about 2.5-4 m, no table "
                "or cable in the way), then run `calibrate` again.")
        height_m = float(np.median([plane["height_m"] for plane in planes]))
        pitch = float(np.median([plane["pitch_rad"] for plane in planes]))
        roll = float(np.median([plane["roll_rad"] for plane in planes]))
        residual = float(np.median([plane["residual_m"] for plane in planes]))
        result = {
            "state": "ok",
            "camera_height_m": round(height_m, 4),
            "camera_pitch_deg": round(math.degrees(pitch), 2),
            "camera_roll_deg": round(math.degrees(roll), 2),
            "residual_m": round(residual, 4),
            "frames": len(planes),
            "mode": "floor-fit",
            "suggested_config": {
                "camera_height_m": round(height_m, 4),
                "camera_pitch_deg": round(math.degrees(pitch), 2),
                "camera_roll_deg": round(math.degrees(roll), 2),
            },
            "note": ("Save suggested_config in this card's configuration; the "
                     "camera is fixed, so this is a one-time calibration."),
            "measured_at": _timestamp(),
        }
        instance["last_calibration"] = result
        instance["last_result"] = result
        node.publish_result(result)
        return result

    # -- info --------------------------------------------------------------

    def _info(self, args: dict):
        instance = self._instance(args)
        base = {
            "name": "HeightMeasurement",
            "manufacture": "Embodied",
            "config": self._public_config(),
        }
        if instance is None:
            return {**base, "state": "idle", "degraded": ["not_running"]}
        topics_in = [{"topic": instance["depth_topic"],
                      "format": "image/depth-zlib"}]
        if instance["detection_topic"]:
            topics_in.append({"topic": instance["detection_topic"],
                              "format": "data/json"})
        intrinsics = instance["intrinsics"]
        return {
            **base,
            "state": "running",
            "instance_id": instance["instance_id"],
            "inputs": {"depth": instance["depth_topic"],
                       "detection": instance["detection_topic"] or None},
            "topic_in": topics_in,
            "topic_out": [{"topic": instance["result_topic"],
                           "format": "data/json"}],
            "intrinsics": {key: intrinsics[key]
                           for key in ("fx", "fy", "cx", "cy", "width",
                                       "height", "source")},
            "calibration": self._calibration(),
            "last_result": instance.get("last_result"),
            "last_calibration": instance.get("last_calibration"),
            "degraded": list(instance["degraded"]),
        }

    # -- helpers -----------------------------------------------------------

    def _instance(self, args: dict):
        instance_id = str(args.get("instance_id") or "")
        with self._lock:
            if instance_id and instance_id in self._instances:
                return self._instances[instance_id]
            if not instance_id:
                return next(iter(self._instances.values()), None)
        return None

    def _calibration(self) -> dict:
        return {
            "camera_height_m": self._cfg["camera_height_m"],
            "camera_pitch_deg": self._cfg["camera_pitch_deg"],
            "camera_roll_deg": self._cfg["camera_roll_deg"],
            "source": "configured" if self._cfg["camera_height_m"] > 0
                      else "uncalibrated",
        }


# ── module helpers ──────────────────────────────────────────────────────────

def _input_topics(args: dict) -> list:
    """Union of `input_topics` and `input_topic`, first occurrence wins.

    agent-core sends the plural only when more than one input is wired, and
    sends the singular as the first of them; cards that read one or the other
    silently bind nothing on the other path (see navi's note).
    """
    topics: list = []
    for topic in args.get("input_topics") or []:
        if isinstance(topic, str) and topic and topic not in topics:
            topics.append(topic)
    single = args.get("input_topic") or ""
    if isinstance(single, str) and single and single not in topics:
        topics.insert(0, single)
    return topics


def _is_depth_topic(topic: str) -> bool:
    return any(topic.endswith(suffix)
               for suffix in ("/camera/depth", "/visual_depth", "/depth"))


def _is_detection_topic(topic: str) -> bool:
    return topic.endswith("/objects") or topic.endswith("/poses")


def _reported_camera_height(instance: dict, configured: float):
    calibration = instance.get("last_calibration") or {}
    if "camera_height_m" in calibration:
        return round(float(calibration["camera_height_m"]), 3)
    return round(float(configured), 3) if configured > 0 else None


def _error(reason: str, message: str) -> dict:
    return {"state": "error", "reason": reason, "message": message}


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")
