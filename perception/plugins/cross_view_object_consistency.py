"""Read-only dual RGB/depth and VOP consistency processor."""
from collections import OrderedDict, deque
import json
import math
import re
import threading
import time
import zlib

import numpy as np

from plugins.cross_view_geometry import calibration, compare, epipolar_candidates
from utils.frame_source import capture_identity, source_header
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
        self.session = config.get("session_id")
        self.device_ids = config.get("device_ids")
        self.live_capture = self.device_ids is not None
        if self.live_capture and (not isinstance(self.device_ids, list)
                or len(self.device_ids) != 2
                or any(not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{32}", d)
                       for d in self.device_ids)
                or self.device_ids[0] == self.device_ids[1] or self.session is not None):
            raise ValueError("two distinct device_ids required in capture-provenance mode")
        self.ids = tuple(config["camera_ids"])
        if ((not self.live_capture and
             (not isinstance(self.session, str) or not self.session or "@" in self.session))
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
        self.time_error_ms = config.get("time_error_ms")
        if self.time_error_ms is not None and (type(self.time_error_ms) not in (int, float)
                or not 0 <= self.time_error_ms < self.skew * 1000):
            raise ValueError("invalid measured time_error_ms")
        self.start_ns = time.time_ns()
        self.cutover_ns = 0
        self.identities = [None, None]
        self.rgb_high = [0, 0]
        self.retired = [deque(maxlen=64), deque(maxlen=64)]
        self.latest_sequence = [{role: 0 for role in FORMATS} for _ in self.ids]
        self.headers = [{role: OrderedDict() for role in FORMATS} for _ in self.ids]
        self.recovering = False
        topics = config.get("topics")
        self.objects_connected = ([True, True] if topics is None
                                  else ["objects" in topics.get(side, {}) for side in "ab"])
        self.calibration_version = None
        self.buffers = [{role: OrderedDict() for role in FORMATS} for _ in self.ids]
        self.latest = [{role: 0 for role in FORMATS} for _ in self.ids]
        self.received = [{role: 0.0 for role in FORMATS} for _ in self.ids]
        self.rejected = [None, None]
        self._comparison = None
        self.cameras = None
        self.depth_registered = False
        self.epipolar_px = float(config.get("epipolar_tolerance_px", 5))
        if not 0 < self.epipolar_px <= 100:
            raise ValueError("invalid epipolar_tolerance_px")
        self.calibration_reason = "missing_calibration"
        if "calibration" in config:
            try:
                records = [config["calibration"][i] for i in self.ids]
                if records[0]["world_frame"] != records[1]["world_frame"]:
                    raise ValueError("different_world_frames")
                if self.live_capture:
                    versions = [r["calibration_version"] for r in records]
                    if (not isinstance(versions[0], str) or not versions[0]
                            or versions[0] != versions[1]
                            or [r["device_id"] for r in records] != self.device_ids):
                        raise ValueError("calibration_source_or_version_mismatch")
                self.cameras = [calibration(r, i, require_registered=False)
                                for r, i in zip(records, self.ids)]
                self.depth_registered = all(r.get("depth_registered") is True for r in records)
                if self.live_capture:
                    self.calibration_version = versions[0]
            except (TypeError, ValueError, KeyError, OverflowError):
                self.calibration_reason = "invalid_calibration"

    def _invalidate(self, cutoff):
        self.cutover_ns = max(self.cutover_ns, cutoff)
        for buffers in (*self.buffers, *self.headers):
            for buffer in buffers.values():
                buffer.clear()
        self._comparison = None

    def _accept_capture(self, side, role, source, stamp):
        identity = capture_identity(source.get("frame_id"))
        if identity is None or not identity["frame_valid"]:
            self.rejected[side] = "invalid_or_offline_capture_frame"
            if role == "rgb":
                self._invalidate(stamp if type(stamp) is int and 0 < stamp < 2**63
                                 else max(self.rgb_high))
            return False
        if (identity["camera_id"] != self.ids[side]
                or identity["optical_frame_id"] != f"{self.ids[side]}_color_optical_frame"
                or identity["device_id"] != self.device_ids[side]):
            self.rejected[side] = "capture_camera_or_device_mismatch"
            if role == "rgb":
                self._invalidate(max(self.rgb_high) + 1)
            return False
        session, clock = identity["capture_session_id"], identity["timestamp_source"]
        other = self.identities[1 - side]
        if other is not None and session == other[0]:
            self.rejected[side] = "capture_session_reused_between_cameras"
            return False
        if (session, clock) != self.identities[side]:
            if (role != "rgb" or stamp <= self.rgb_high[side] or session in self.retired[side]
                    or (self.identities[side] is not None and session == self.identities[side][0])):
                return False
            if self.identities[side] is not None:
                self.retired[side].append(self.identities[side][0])
                self._invalidate(stamp)
            self.identities[side] = (session, clock)
            self.latest_sequence[side] = {stream: 0 for stream in FORMATS}
        sequence = identity["frame_sequence"]
        if sequence <= self.latest_sequence[side][role]:
            return False
        if role == "rgb":
            self.rgb_high[side] = stamp
        self.latest_sequence[side][role] = sequence
        return True

    def ingest(self, side, role, source, data, now):
        if not self.live_capture:
            self.received[side][role] = now
        if not isinstance(source, dict):
            self.rejected[side] = "missing_capture_provenance"
            return
        stamp = source.get("stamp_ns")
        if self.live_capture and source.get("frame_valid") is False:
            self._accept_capture(side, role, source, stamp)
            return
        if (type(stamp) is not int or not max(self.start_ns, self.cutover_ns) <= stamp < 2**63
                or not 0 <= now - stamp / 1e9 <= self.age):
            self.rejected[side] = "source_timestamp_outside_live_window"
            return
        if not self.live_capture and source.get("frame_id") != f"{self.ids[side]}@{self.session}":
            self.rejected[side] = "unverified_camera_or_session"
            return
        if stamp <= self.latest[side][role]:
            return
        if self.live_capture and not self._accept_capture(side, role, source, stamp):
            return
        last = self.received[side][role]
        if self.live_capture and role in ("rgb", "objects") and last and now - last > self.age and not self.recovering:
            self._invalidate(stamp)
            self.recovering = True
        self.received[side][role] = now
        self.rejected[side] = None
        self.latest[side][role] = stamp
        buffer = self.buffers[side][role]
        headers = self.headers[side][role]
        buffer[stamp] = data
        headers[stamp] = source.get("frame_id")
        if len(buffer) > self.capacity:
            buffer.popitem(last=False)
            headers.popitem(last=False)

    def result(self, now):
        result = dict(state="insufficient_evidence", camera_a_id=self.ids[0],
                      camera_b_id=self.ids[1], timestamp=now, matched_targets=[],
                      unmatched_targets=[], position_residual_m=None,
                      time_offset_ms=None, confidence=0.0, reason="awaiting_synchronized_frames",
                      capability_level=0 if self.cameras is None else 1,
                      calibration_version=self.calibration_version, sync_status="waiting")
        if self.live_capture:
            result["capture_sessions"] = [item[0] if item else None for item in self.identities]
            result["timestamp_sources"] = [item[1] if item else None for item in self.identities]
            result["time_error_ms"] = self.time_error_ms
        complete = []
        object_problem = None
        for side, streams in enumerate(self.buffers):
            headers = self.headers[side]
            for role, buffer in streams.items():
                while buffer and now - next(iter(buffer)) / 1e9 > self.age:
                    stamp, _ = buffer.popitem(last=False)
                    headers[role].pop(stamp, None)
            complete.append(sorted(stamp for stamp in streams["rgb"].keys() & streams["objects"].keys()
                                   if headers["rgb"][stamp] == headers["objects"][stamp]))
            last = self.received[side]["rgb"]
            if now - (last or self.start_ns / 1e9) > self.age:
                if self.live_capture and not self.recovering:
                    self._invalidate(int(now * 1e9))
                    self.recovering = True
                result.update(state="stale", sync_status="stale",
                              reason=f"camera_{'ab'[side]}_rgb_stream_stale")
                return result
            last_objects = self.received[side]["objects"]
            if not self.objects_connected[side]:
                object_problem = object_problem or f"camera_{'ab'[side]}_objects_not_connected"
            elif now - (last_objects or self.start_ns / 1e9) > self.age:
                object_problem = object_problem or f"camera_{'ab'[side]}_objects_stream_stale"
        self.recovering = False
        candidates = [(a, b) for a in complete[0] for b in complete[1]
                      if abs(a-b) <= self.skew * 1e9]
        if not candidates:
            raw = [list(streams["rgb"]) for streams in self.buffers]
            if raw[0] and raw[1]:
                nearest = min(((a, b) for a in raw[0] for b in raw[1]),
                              key=lambda pair: (abs(pair[0]-pair[1]), -min(pair)))
                self._record_pair(result, nearest)
                if result["time_offset_ms"] > self.skew * 1000:
                    result.update(reason="cross_view_time_offset_exceeded", sync_status="offset_exceeded")
                elif object_problem:
                    result["reason"] = object_problem
            elif any(self.rejected):
                result["reason"] = next(r for r in self.rejected if r)
            return result
        # Prefer the freshest common time and then the smallest skew.
        geometric = [pair for pair in candidates
                     if all(stamp in self.buffers[side]["depth"]
                            and self.headers[side]["depth"][stamp] == self.headers[side]["rgb"][stamp]
                            for side, stamp in enumerate(pair))]
        a, b = max(geometric or candidates, key=lambda p: (min(p), -abs(p[0]-p[1])))
        bounded = self._record_pair(result, (a, b))
        if self.cameras is None:
            result["reason"] = self.calibration_reason
            return result
        if self.live_capture and not bounded:
            result["reason"] = "capture_clock_error_unverified_or_exceeds_limit"
            return result
        objects, depths = [], []
        depth_problem = None if self.depth_registered else "registered_depth_unverified"
        for side, stamp in enumerate((a, b)):
            streams = self.buffers[side]
            payload = streams["objects"][stamp]
            if payload.get("source_size") != list(self.cameras[side][2:4]):
                result["reason"] = f"camera_{'ab'[side]}_calibration_size_mismatch"
                return result
            objects.append(payload["objects"])
            headers = self.headers[side]
            if stamp not in streams["depth"] or headers["depth"][stamp] != headers["rgb"][stamp]:
                depth_problem = depth_problem or f"camera_{'ab'[side]}_aligned_depth_unavailable"
            else:
                depth = streams["depth"][stamp]
                if not np.any(np.isfinite(depth) & (depth > 0)):
                    depth_problem = depth_problem or f"camera_{'ab'[side]}_depth_invalid"
                depths.append(depth)
        if depth_problem:
            result.update(reason=depth_problem,
                          epipolar_candidate_count=epipolar_candidates(
                              objects, self.cameras, self.epipolar_px))
            return result
        if self._comparison is None or self._comparison[0] != (a, b):
            self._comparison = ((a, b), compare(
                objects, depths, self.cameras, self.tolerance, self.gate,
                self.ambiguity, self.minimum))
        result.update(self._comparison[1], capability_level=2)
        return result

    def _record_pair(self, result, pair):
        a, b = pair
        result.update(timestamp=max(a, b) / 1e9, time_offset_ms=abs(a-b) / 1e6)
        if self.live_capture:
            result["capture_frames"] = [
                {"capture_timestamp_ns": stamp,
                 **capture_identity(self.headers[side]["rgb"][stamp])}
                for side, stamp in enumerate((a, b))]
        bounded = (self.time_error_ms is not None
                   and self.time_error_ms + result["time_offset_ms"] <= self.skew * 1000)
        result["sync_status"] = ("legacy_session" if not self.live_capture
                                 else "bounded" if bounded else "unqualified")
        return bounded


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
                if self._node is not None and rig_json != self._rig_json:
                    return {"state": "error", "adapter_ok": False, "error": "stop card before changing rig"}
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
        required = {(side, "rgb") for side in range(2)}
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
