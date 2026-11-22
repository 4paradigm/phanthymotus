"""Read-only dual RGB/depth and VOP consistency processor."""
from collections import OrderedDict
import json
import math
import threading
import time
import zlib

import numpy as np

from plugins.cross_view_geometry import calibration, compare
from utils.frame_source import source_header
from utils.ros_lifecycle import dispose_node

NAME = "cross_view_object_consistency"
FORMATS = {"rgb": "image/jpeg", "objects": "data/json", "depth": "image/depth-zlib"}


def decode_depth(data):
    size = 640 * 480 * 2
    if len(data) > size + 4096:
        raise ValueError("oversized_depth")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(bytes(data), size + 1)
    if len(raw) != size or not decoder.eof or decoder.unused_data:
        raise ValueError("invalid_depth")
    result = np.frombuffer(raw, dtype="<u2").reshape(480, 640).astype(np.float32)
    result[result == 0] = np.nan
    return result / 1000


class Synchronizer:
    def __init__(self, config):
        self.session = config["session_id"]
        self.ids = tuple(config["camera_ids"])
        if (not isinstance(self.session, str) or not self.session or "@" in self.session
                or len(self.ids) != 2 or len(set(self.ids)) != 2
                or any(not isinstance(i, str) or not i or "@" in i for i in self.ids)):
            raise ValueError("distinct camera_ids and capture session_id required")
        self.capacity = config.get("buffer_size", 8)
        self.skew = float(config.get("max_time_offset_ms", 40)) / 1000
        self.age = float(config.get("max_age_ms", 1000)) / 1000
        self.tolerance = float(config.get("tolerance_m", 0.06))
        self.gate = float(config.get("match_gate_m", 0.25))
        self.ambiguity = float(config.get("ambiguity_m", 0.03))
        self.minimum = float(config.get("min_confidence", 0.35))
        if (type(self.capacity) is not int or not 1 <= self.capacity <= 32
                or not all(math.isfinite(x) for x in (self.skew, self.age, self.tolerance,
                                                     self.gate, self.ambiguity, self.minimum))
                or not 0 < self.skew < self.age <= 10
                or not 0 < self.tolerance < self.gate <= 5
                or not 0 <= self.ambiguity < self.gate or not 0 <= self.minimum <= 1):
            raise ValueError("invalid_buffer_or_thresholds")
        self.start_ns = time.time_ns()
        self.buffers = [{role: OrderedDict() for role in FORMATS} for _ in self.ids]
        self.latest = [{role: 0 for role in FORMATS} for _ in self.ids]
        self.received = [{role: 0.0 for role in FORMATS} for _ in self.ids]
        self.rejected = [None, None]
        self._comparison = None
        self.cameras = None
        self.calibration_reason = "missing_calibration"
        if "calibration" in config:
            try:
                records = [config["calibration"][i] for i in self.ids]
                if records[0]["world_frame"] != records[1]["world_frame"]:
                    raise ValueError("different_world_frames")
                self.cameras = [calibration(r, i) for r, i in zip(records, self.ids)]
            except (TypeError, ValueError, KeyError, OverflowError):
                self.calibration_reason = "invalid_calibration"

    def ingest(self, side, role, source, data, now):
        self.received[side][role] = now
        if not isinstance(source, dict):
            self.rejected[side] = "missing_capture_provenance"
            return
        stamp = source.get("stamp_ns")
        if source.get("frame_id") != f"{self.ids[side]}@{self.session}":
            self.rejected[side] = "unverified_camera_or_session"
            return
        if (type(stamp) is not int or not self.start_ns <= stamp < 2**63
                or not 0 <= now - stamp / 1e9 <= self.age):
            self.rejected[side] = "source_timestamp_outside_live_window"
            return
        if stamp <= self.latest[side][role]:
            return
        self.rejected[side] = None
        self.latest[side][role] = stamp
        buffer = self.buffers[side][role]
        buffer[stamp] = data
        if len(buffer) > self.capacity:
            buffer.popitem(last=False)

    def result(self, now):
        result = dict(state="insufficient_evidence", camera_a_id=self.ids[0],
                      camera_b_id=self.ids[1], timestamp=now, matched_targets=[],
                      unmatched_targets=[], position_residual_m=None,
                      time_offset_ms=None, confidence=0.0, reason="awaiting_synchronized_frames")
        complete = []
        for side, streams in enumerate(self.buffers):
            for buffer in streams.values():
                while buffer and now - next(iter(buffer)) / 1e9 > self.age:
                    buffer.popitem(last=False)
            complete.append(sorted(set(streams["rgb"]) & set(streams["objects"])))
            for role in ("rgb", "objects"):
                last = self.received[side][role]
                if now - (last or self.start_ns / 1e9) > self.age:
                    result.update(state="stale", reason=f"camera_{'ab'[side]}_{role}_stream_stale")
                    return result
        candidates = [(a, b) for a in complete[0] for b in complete[1]
                      if abs(a-b) <= self.skew * 1e9]
        if not candidates:
            if complete[0] and complete[1]:
                result["reason"] = "cross_view_time_offset_exceeded"
                result["time_offset_ms"] = min(abs(a-b) for a in complete[0] for b in complete[1]) / 1e6
            elif any(self.rejected):
                result["reason"] = next(r for r in self.rejected if r)
            return result
        # Prefer the freshest common time and then the smallest skew.
        geometric = [pair for pair in candidates
                     if all(stamp in self.buffers[side]["depth"] for side, stamp in enumerate(pair))]
        a, b = max(geometric or candidates, key=lambda p: (min(p), -abs(p[0]-p[1])))
        result.update(timestamp=max(a, b) / 1e9, time_offset_ms=abs(a-b) / 1e6)
        if self.cameras is None:
            result["reason"] = self.calibration_reason
            return result
        objects, depths = [], []
        for side, stamp in enumerate((a, b)):
            streams = self.buffers[side]
            payload = streams["objects"][stamp]
            if payload.get("source_size") != list(self.cameras[side][2:4]):
                result["reason"] = f"camera_{'ab'[side]}_calibration_size_mismatch"
                return result
            if stamp not in streams["depth"]:
                result["reason"] = f"camera_{'ab'[side]}_aligned_depth_unavailable"
                return result
            objects.append(payload["objects"])
            depths.append(streams["depth"][stamp])
        if self._comparison is None or self._comparison[0] != (a, b):
            self._comparison = ((a, b), compare(
                objects, depths, self.cameras, self.tolerance, self.gate,
                self.ambiguity, self.minimum))
        result.update(self._comparison[1])
        return result


class CrossViewObjectConsistencyPlugin:
    PREFIX = NAME

    def __init__(self, config, executor):
        self._rig_json = config.get("rig_json", "")
        self._executor = executor
        self._topic = config.get("topic", "/perception/cross_view_object_consistency/state")
        self._lock = threading.RLock()
        self._lifecycle = threading.Lock()
        self._node = None
        self._sync = None
        self._bindings = {}

    def _ports(self):
        inputs = []
        for side in "ab":
            for role, fmt in FORMATS.items():
                port = {"format": fmt, "desc": f"Camera {side.upper()} {role}"}
                topic = self._bindings.get((side, role))
                if topic:
                    port["topic"] = topic
                inputs.append(port)
        return {
            "topic_in": inputs,
            "topic_out": [{"topic": self._topic, "format": "data/json", "desc": "跨视角一致性"}],
        }

    def get_tools(self):
        return [{
            "name": NAME, "type": "processor", "multiInstance": False,
            "description": "双视角目标时空一致性诊断",
            "inputSchema": {"type": "object", "required": ["action"], "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
            }, "x-action-params": {"info": {"params": [], "description": "读取当前一致性诊断"}}},
            "configSchema": {"type": "object", "required": ["rig_json"], "properties": {
                "rig_json": {"type": "string", "scope": "instance", "default": "",
                             "description": "双视角相机 ID、采集会话、ROS topic 和标定 JSON"},
            }},
            **self._ports(),
        }]

    def start(self):
        pass  # Wait for canvas wiring.

    def stop(self):
        with self._lifecycle:
            with self._lock:
                node, self._node = self._node, None
                self._sync = None
                self._bindings = {}
            if node is not None:
                dispose_node(self._executor, node)

    def dispatch(self, name, args):
        if name != NAME:
            return None
        args = args or {}
        action = args.get("action", "info")
        if action == "stop":
            self.stop()
        elif action == "config":
            with self._lifecycle:
                try:
                    rig_json = args.get("rig_json", self._rig_json)
                    candidate = Synchronizer(json.loads(rig_json))
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    return {"state": "error", "adapter_ok": False, "error": str(error)}
                self._rig_json = rig_json
                return {"state": "configured", "calibrated": candidate.cameras is not None}
        elif action == "start":
            with self._lifecycle:
                if self._node is None:
                    try:
                        self._open(args)
                    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError, OverflowError) as error:
                        return {"state": "error", "error": str(error)}
        elif action != "info":
            return {"state": "error", "error": "unsupported action"}
        with self._lock:
            output = {"state": "running" if self._node is not None else "idle",
                      **self._ports()}
            if self._sync is not None:
                output["diagnostic"] = self._sync.result(time.time())
            return output

    def _open(self, args):
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        config = json.loads((args.get("instance_config") or {}).get("rig_json")
                            or self._rig_json or "{}")
        sync = Synchronizer(config)
        connected = set(args.get("input_topics") or [])
        if args.get("input_topic"):
            connected.add(args["input_topic"])
        bindings = [(side, role, topic) for side, key in enumerate("ab")
                    for role, topic in config["topics"][key].items()]
        required = {(side, role) for side in range(2) for role in ("rgb", "objects")}
        topics = {topic for _, _, topic in bindings}
        if (not required.issubset({(s, r) for s, r, _ in bindings})
                or any(role not in FORMATS for _, role, _ in bindings)
                or len(topics) != len(bindings) or topics != connected or self._topic in topics):
            raise ValueError("rig topics must match connected canvas topics exactly")
        node = Node(NAME)
        try:
            publisher = node.create_publisher(String, self._topic, 10)
            for side, role, topic in bindings:
                kind = String if role == "objects" else CompressedImage
                node.create_subscription(
                    kind, topic, lambda msg, s=side, r=role: self._receive(sync, s, r, msg),
                    qos_profile_sensor_data,
                )
            node.create_timer(0.2, lambda: self._publish(sync, publisher, String))
            self._executor.add_node(node)
        except Exception:
            dispose_node(self._executor, node)
            raise
        with self._lock:
            self._bindings = {("ab"[side], role): topic for side, role, topic in bindings}
            self._node, self._sync = node, sync

    def _receive(self, sync, side, role, msg):
        with self._lock:
            if self._sync is not sync:
                return
            now = time.time()
            try:
                if role == "objects":
                    if len(msg.data) > 262144:
                        return
                    payload = json.loads(msg.data)
                    source = payload.get("source")
                    objects = payload["objects"]
                    if (not isinstance(objects, list) or len(objects) > 64
                            or any(not isinstance(obj, dict) for obj in objects)):
                        return
                else:
                    source = source_header(msg.header)
                    payload = None
                    if role == "rgb" and (not msg.data or "jpeg" not in msg.format.lower()):
                        return
                    if role == "depth":
                        if msg.format != "16UC1; compressedDepth zlib":
                            return
                        payload = decode_depth(msg.data)
                sync.ingest(side, role, source, payload, now)
            except (AttributeError, TypeError, ValueError, KeyError, zlib.error):
                return

    def _publish(self, sync, publisher, message_type):
        with self._lock:
            if self._sync is sync:
                message = message_type()
                message.data = json.dumps(sync.result(time.time()), allow_nan=False)
                publisher.publish(message)
