"""Fuse #407 semantic masks with registered depth without owning a camera."""
import hashlib
import json
import math
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass

import numpy as np

from plugins.camera_info import SCHEMA, for_topic
from plugins.surface_geometry import (
    MAX_PACKET, MAX_PIXELS, decode_depth, decode_mask, estimate, unavailable,
)

NAME = "semantic_surface_fusion"
FORMATS = {"mask": "image/png", "classes": "data/json", "depth": "image/depth-zlib"}
NSEC = 1_000_000_000


@dataclass(frozen=True)
class Settings:
    sync_ms: float = 25
    stale_ms: float = 1500
    timeout_ms: float = 200
    depth_scale: float = 0.001
    max_points: int = 0

    def __post_init__(self):
        limits = {"sync_ms": (0, 100), "stale_ms": (250, 5000),
                  "timeout_ms": (10, 1000), "depth_scale": (1e-6, 0.01),
                  "max_points": (0, 20000)}
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if type(value) not in (float, int) or not low <= value <= high:
                raise ValueError(f"{key} must be in [{low}, {high}]")
        if type(self.max_points) is not int:
            raise ValueError("max_points must be an integer")


def stamp(sec, nanosec):
    if type(sec) is not int or type(nanosec) is not int or sec < 0 or not 0 <= nanosec < NSEC:
        raise ValueError("invalid_capture_stamp")
    return sec * NSEC + nanosec


def inputs(args):
    topics = args.get("input_topics") or [args.get("input_topic", "")]
    bound = {}
    for topic in topics:
        if not topic:
            continue
        role = topic.rsplit("/", 1)[-1]
        if role not in FORMATS or role in bound:
            raise ValueError("Connect one segmentation mask, its classes, and registered depth")
        bound[role] = topic
    if set(bound) != set(FORMATS) or bound["mask"].rsplit("/", 1)[0] != bound["classes"].rsplit("/", 1)[0]:
        raise ValueError("Connect mask and classes from the same segmentation instance, and depth")
    return bound


def calibration(declarations, topics):
    mask = for_topic(declarations, topics["mask"])
    depth = for_topic(declarations, topics["depth"])
    if not mask.get("id") or mask.get("id") != depth.get("id"):
        raise ValueError("camera_identity_missing_or_mismatched")
    if any(d.get("schema") != SCHEMA for d in (mask, depth)):
        raise ValueError("missing_camera_declaration")
    vendor = depth.get("vendor")
    if not isinstance(vendor, dict) or vendor.get("depth_aligned_to") != "color":
        raise ValueError("depth_registration_unconfirmed")
    width, height = depth.get("width"), depth.get("height")
    if (type(width) is not int or type(height) is not int or min(width, height) <= 0
            or width * height > MAX_PIXELS or (width, height) != (mask.get("width"), mask.get("height"))):
        raise ValueError("image_size_mismatch")
    if any(d.get("distortion_model") != "pinhole"
           or any(v != 0 for v in (d.get("D") or [])) for d in (mask, depth)):
        raise ValueError("rectified_pinhole_required")
    k = depth.get("K") or []
    if (len(k) != 9 or not all(type(v) in (int, float) and math.isfinite(v) for v in k)
            or not (1 <= k[0] <= 1e6 and 1 <= k[4] <= 1e6)
            or not (0 <= k[2] < width and 0 <= k[5] < height)
            or [k[i] for i in (1, 3, 6, 7, 8)] != [0, 0, 0, 0, 1]
            or k != mask.get("K")):
        raise ValueError("missing_or_incompatible_intrinsics")
    return width, height, k, depth["id"]


class Frames:
    """Small per-stream rings. Called under the runtime lock."""
    def __init__(self, settings):
        self.settings = settings
        self.buffers = {role: deque(maxlen=32 if role == "depth" else 8) for role in FORMATS}
        self.depth_bytes = 0
        self.watermark = 0

    def add(self, role, capture, frame, message):
        queue = self.buffers[role]
        if role == "depth":
            size = len(message.data) if hasattr(message, "data") else 0
            while queue and (len(queue) == queue.maxlen or self.depth_bytes + size > 16 * 1024**2):
                self.depth_bytes -= queue.popleft()[3]
            self.depth_bytes += size
            queue.append((capture, frame, message, size))
        else:
            queue.append((capture, frame, message))

    def take(self, now):
        max_age = self.settings.stale_ms * 1_000_000
        tolerance = self.settings.sync_ms * 1_000_000
        masks, classes, depths = (self.buffers[key] for key in FORMATS)
        for capture, frame, mask in sorted(masks, key=lambda entry: entry[0], reverse=True):
            if capture <= self.watermark or not 0 <= now - capture <= max_age:
                continue
            report = next((msg for t, f, msg in classes if (t, f) == (capture, frame)), None)
            candidates = [(abs(t - capture), msg) for t, f, msg, _ in depths
                          if f == frame and 0 <= now - t <= max_age and abs(t - capture) <= tolerance]
            if report is not None and candidates:
                self.watermark = capture
                return capture, frame, mask, report, min(candidates, key=lambda item: item[0])[1]
        return None


class Runtime:
    def __init__(self, executor, key, topics, outputs, settings, declarations):
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage, PointCloud2
        from std_msgs.msg import String
        from utils.qos import CAMERA_QOS

        self.executor = executor
        self.settings = settings
        self.topics = topics
        self.frames = Frames(settings)
        self.cv = threading.Condition()
        self.closed = False
        self.job = None
        self.result = None
        self.last_success = 0
        self.cloud_header = None
        self.last_receive = {}
        self.started = time.monotonic()
        self.failure = ""
        try:
            self.calibration = calibration(declarations, topics)
        except (ValueError, TypeError) as error:
            self.calibration = None
            self.failure = str(error)
        self.node = Node(key)
        self.worker = threading.Thread(target=self.run, name=key, daemon=True)
        try:
            self.publisher = self.node.create_publisher(String, outputs[0]["topic"], 2)
            self.cloud = (self.node.create_publisher(PointCloud2, outputs[1]["topic"], 1)
                          if settings.max_points else None)
            for role, topic in topics.items():
                cls = String if role == "classes" else CompressedImage
                self.node.create_subscription(cls, topic,
                                              lambda msg, role=role: self.receive(role, msg), CAMERA_QOS)
            self.node.create_timer(0.05, self.tick)
            executor.add_node(self.node)
            self.worker.start()
        except Exception:
            executor.remove_node(self.node)
            self.node.destroy_node()
            raise

    def receive(self, role, msg):
        try:
            if role == "classes":
                if len(msg.data) > 65536:
                    return
                msg = json.loads(msg.data)
                capture = stamp(**msg["stamp"])
                frame = msg["frame_id"]
            else:
                if len(msg.data) > MAX_PACKET:
                    return
                capture = stamp(msg.header.stamp.sec, msg.header.stamp.nanosec)
                frame = msg.header.frame_id
            if not capture or not isinstance(frame, str):
                return
        except (ValueError, KeyError, TypeError):
            return
        with self.cv:
            if not self.closed:
                self.last_receive[role] = time.monotonic()
                self.frames.add(role, capture, frame, msg)
                self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.closed or self.job is not None)
                if self.closed:
                    return
                job = self.job
            started, pair = job
            try:
                payload, cloud = self.fuse(pair)
            except (ValueError, TypeError, zlib.error, OverflowError, RuntimeError) as error:
                payload, cloud = unavailable(str(error)), None
            except Exception:
                self.node.get_logger().error("semantic surface fusion failed")
                payload, cloud = unavailable("processing_error"), None
            with self.cv:
                if self.closed:
                    return
                if (time.monotonic() - started) * 1000 >= self.settings.timeout_ms:
                    payload, cloud = unavailable("processing_timeout"), None
                self.result = pair[0], payload, cloud
                self.job = None

    def fuse(self, pair):
        deadline = time.monotonic() + self.settings.timeout_ms / 1000
        capture, frame, mask_msg, report, depth_msg = pair
        width, height, k, camera_id = self.calibration
        if (report.get("width"), report.get("height")) != (width, height):
            raise ValueError("image_size_mismatch")
        if mask_msg.format != "png" or depth_msg.format != "16UC1; compressedDepth zlib":
            raise ValueError("unsupported_image_encoding")
        mask = decode_mask(mask_msg.data, width, height)
        depth = decode_depth(depth_msg.data, width, height, self.settings.depth_scale)
        # Bumi's registered RGB/depth topics both omit frame_id. The declarations
        # identify the shared color optical coordinate system in that case.
        if not frame:
            color_topic, separator, _ = self.topics["mask"].rpartition("/segment/")
            depth_root = self.topics["depth"].rsplit("/", 1)[0]
            if not separator or color_topic != depth_root + "/color":
                raise ValueError("headerless_camera_source_unconfirmed")
            frame = depth_root.strip("/").replace("/", "_") + "_color_optical_frame"
        metadata = dict(camera_id=camera_id, frame_id=frame,
                        timestamp=dict(sec=capture // NSEC, nanosec=capture % NSEC))
        classes = report.get("classes")
        if not isinstance(classes, list) or not 1 <= len(classes) <= 256:
            raise ValueError("invalid_class_table")
        histogram = np.bincount(mask.ravel(), minlength=256)
        observations, clouds = [], []
        labels = set()
        for label in classes:
            if not isinstance(label, dict):
                raise ValueError("invalid_class_table")
            ident, name = label.get("id"), label.get("name")
            if (type(ident) is not int or not 0 <= ident <= 255 or ident in labels
                    or not isinstance(name, str) or not name or label.get("pixels") != histogram[ident]):
                raise ValueError("invalid_class_table")
            labels.add(ident)
        if 0 not in labels or set(np.flatnonzero(histogram)) - labels:
            raise ValueError("unmapped_mask_labels")
        foreground = [label for label in classes if label["id"]]
        budget = self.settings.max_points
        remaining = sum(bool(histogram[label["id"]]) for label in foreground)
        for label in foreground:
            if time.monotonic() >= deadline:
                return unavailable("processing_timeout"), None
            if histogram[label["id"]]:
                limit = min(budget, max(1, budget // remaining))
                observation, cloud = estimate(mask, depth, label["id"], k, limit)
                remaining -= 1
            else:
                observation, cloud = unavailable("empty_mask"), None
            observations.append(dict(observation, **metadata, class_name=label["name"]))
            if cloud is not None:
                clouds.append(cloud)
                budget -= len(cloud)
        if not observations:
            result = unavailable("no_foreground_classes")
            result.update(metadata)
        else:
            rank = {"insufficient_evidence": 0, "degraded": 1, "valid": 2}
            result = dict(max(observations, key=lambda obs: rank[obs["state"]]))
        result["observations"] = observations
        return result, np.concatenate(clouds) if clouds else None

    def tick(self):
        from std_msgs.msg import String

        with self.cv:
            if self.closed:
                return
            now = self.node.get_clock().now().nanoseconds
            elapsed = time.monotonic()
            max_age = self.settings.stale_ms / 1000
            result, self.result = self.result, None
            if self.failure:
                payload = unavailable(self.failure)
            elif self.job and (elapsed - self.job[0]) * 1000 >= self.settings.timeout_ms:
                payload = unavailable("processing_timeout")
            elif result and 0 <= now - result[0] <= max_age * NSEC:
                _, payload, cloud = result
                self.last_success = result[0]
                if self.cloud is not None and payload.get("frame_id"):
                    self.publish_cloud(payload, cloud)
            elif any(elapsed - self.last_receive.get(role, self.started) >= max_age for role in FORMATS):
                missing = [role for role in FORMATS
                           if elapsed - self.last_receive.get(role, self.started) >= max_age]
                payload = unavailable("stream_stale:" + ",".join(missing), "stale")
            elif elapsed - self.started >= max_age and not 0 <= now - self.last_success <= max_age * NSEC:
                payload = unavailable("no_recent_synchronized_frames", "stale")
            elif not self.last_success:
                payload = unavailable("waiting_for_synchronized_frames")
            else:
                payload = None

            if payload is not None:
                if self.cloud_header and not payload.get("frame_id"):
                    self.publish_cloud(self.cloud_header, None)
                msg = String()
                msg.data = json.dumps({"observations": [], **payload}, allow_nan=False)
                self.publisher.publish(msg)
            if self.calibration and self.job is None:
                pair = self.frames.take(now)
                if pair:
                    self.job = elapsed, pair
                    self.cv.notify()

    def publish_cloud(self, payload, cloud):
        from array import array
        from sensor_msgs.msg import PointCloud2, PointField

        msg = PointCloud2()
        msg.header.frame_id = payload["frame_id"]
        msg.header.stamp.sec = payload["timestamp"]["sec"]
        msg.header.stamp.nanosec = payload["timestamp"]["nanosec"]
        msg.height = 1
        msg.width = len(cloud) if cloud is not None else 0
        msg.fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                      for i, name in enumerate("xyz")]
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.is_dense = True
        if cloud is not None:
            msg.data = array("B", cloud.tobytes())
        self.cloud.publish(msg)
        self.cloud_header = payload if cloud is not None else None

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify()
        self.executor.remove_node(self.node)
        self.worker.join()
        self.node.destroy_node()


class SemanticSurfaceFusionPlugin:
    PREFIX = NAME

    def __init__(self, cfg, executor):
        self.defaults = {key: cfg[key] for key in Settings.__dataclass_fields__ if key in cfg}
        Settings(**self.defaults)
        self.executor = executor
        self.instances = {}
        self.pending = {}
        self.lock = threading.RLock()

    def get_tools(self):
        properties = {
            "action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
            "instance_id": {"type": "string"},
            "input_topics": {"type": "array", "items": {"type": "string"}},
            "camera_info": {"type": "object"},
        }
        for name in Settings.__dataclass_fields__:
            default = self.defaults.get(name, getattr(Settings(), name))
            properties[name] = {"type": "integer" if name == "max_points" else "number",
                                "default": default}
        return [dict(name=NAME, type="processor", multiInstance=True,
                     description="Fuse semantic masks and registered depth into visible optical-frame surfaces.",
                     inputSchema=dict(type="object", required=["action"], properties=properties),
                     configSchema=dict(type="object", properties={
                         name: {**properties[name], "scope": "instance"}
                         for name in Settings.__dataclass_fields__}),
                     topic_in=self.ports(), topic_out=self.outputs("", False))]

    @staticmethod
    def ports(topics=None):
        return [dict(topic=(topics or {}).get(role, ""), format=fmt, desc=role)
                for role, fmt in FORMATS.items()]

    @staticmethod
    def outputs(root, cloud):
        return [dict(topic=root + "/state" if root else "", format="data/json",
                     desc="Visible semantic surface observations"),
                dict(topic=root + "/points" if root and cloud else "", format="sensor/pointcloud2",
                     desc="Optional bounded visible surface points")]

    def dispatch(self, name, args):
        if name != NAME:
            return None
        instance = args.get("instance_id") or "default"
        if not isinstance(instance, str):
            raise ValueError("instance_id must be a string")
        key = NAME + "_" + hashlib.sha256(instance.encode()).hexdigest()[:16]
        action = args.get("action", "info")
        overrides = {k: args[k] for k in Settings.__dataclass_fields__ if k in args}
        with self.lock:
            current = self.instances.get(key)
            if action == "config":
                if not args.get("instance_id"):
                    settings = Settings(**{**self.defaults, **overrides})
                    self.defaults.update(overrides)
                else:
                    settings = Settings(**{**self.defaults, **self.pending.get(key, {}), **overrides})
                    self.pending.setdefault(key, {}).update(overrides)
                return dict(state="configured", settings=vars(settings))
            if action == "start":
                from rclpy.validate_full_topic_name import validate_full_topic_name

                topics = inputs(args)
                for topic in topics.values():
                    validate_full_topic_name(topic)
                previous = vars(current[0][1]) if current else self.defaults
                settings = Settings(**{**previous, **self.pending.get(key, {}), **overrides})
                declarations = args.get("camera_info") or {}
                signature = topics, settings, declarations
                if current is None or current[0] != signature:
                    self.remove(key)
                    outputs = self.outputs("/perception/" + key, settings.max_points)
                    self.instances[key] = signature, Runtime(
                        self.executor, key, topics, outputs, settings, declarations)
                self.pending.pop(key, None)
            elif action == "stop":
                self.pending.pop(key, None)
                self.remove(key)
            elif action != "info":
                raise ValueError("Unknown semantic surface fusion action")
            entry = self.instances.get(key)
            if entry:
                (topics, settings, _), runtime = entry
                return dict(state="running", message=runtime.failure, settings=vars(settings),
                            topic_in=self.ports(topics),
                            topic_out=self.outputs("/perception/" + key, settings.max_points))
            return dict(state="idle", topic_in=self.ports(), topic_out=self.outputs("", False))

    def remove(self, key):
        entry = self.instances.pop(key, None)
        if entry:
            entry[1].close()

    def shutdown(self):
        with self.lock:
            self.pending.clear()
            for key in list(self.instances):
                self.remove(key)
