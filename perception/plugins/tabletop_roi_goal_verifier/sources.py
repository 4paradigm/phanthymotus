"""Image-only ROS2 adapter. Header capture time is never replaced by receipt time."""

import base64
import hashlib
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from uuid import uuid4

import cv2
import numpy as np

from .contracts import OBSERVATION, need, number


class Sources:
    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        self.condition = threading.Condition()
        self.frames = {}
        self.previews = OrderedDict()
        self.epoch = uuid4().hex
        self.sequence = {}
        self.nodes = []
        self.closed = False
        self.cancelled = threading.Event()
        self.source_epochs = {key: uuid4().hex for key in config["sources"]}
        self.errors = {}
        self.ros = None
        self.thread = None
        self.declared_sizes = {}

    def reject(self, source_id, exc):
        with self.condition:
            if not self.closed:
                self.errors[source_id] = str(exc)
                self.frames.pop(source_id, None)
                self.source_epochs[source_id] = uuid4().hex
                self.condition.notify_all()

    def image(self, source_id, msg):
        """Read a ROS image. No camera device or robot object is constructed."""
        binding = self.config["sources"][source_id]
        try:
            need(
                binding.get("stamp_semantics") == "capture"
                and binding.get("capture_time_record"),
                "CAPTURE_TIME_UNPROVEN",
            )
            sec, nano = msg.header.stamp.sec, msg.header.stamp.nanosec
            need(
                type(sec) is int
                and type(nano) is int
                and sec >= 0
                and 0 <= nano < 10**9,
                "INVALID_IMAGE_STAMP",
            )
            stamp = sec + nano / 1e9
            need(0 <= time.time() - stamp <= self.engine.max_age, "STALE_OR_FUTURE")
            need(msg.header.frame_id == binding["frame_id"], "CAMERA_FRAME_MISMATCH")
            if binding["message_type"] == "compressed":
                need(
                    len(msg.data) <= 8 * 1024 * 1024 and "jpeg" in msg.format.lower(),
                    "INVALID_JPEG",
                )
                pixels = cv2.imdecode(
                    np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR
                )
                need(isinstance(pixels, np.ndarray), "INVALID_JPEG")
            else:
                need(
                    msg.encoding in ("bgr8", "rgb8")
                    and msg.width > 0
                    and msg.height > 0
                    and msg.width <= 8192
                    and msg.height <= 8192
                    and msg.step >= msg.width * 3
                    and len(msg.data) == msg.height * msg.step <= 32 * 1024 * 1024,
                    "INVALID_RAW_IMAGE",
                )
                pixels = (
                    np.frombuffer(bytes(msg.data), np.uint8)
                    .reshape(msg.height, msg.step)[:, : msg.width * 3]
                    .reshape(msg.height, msg.width, 3)
                    .copy()
                )
                if msg.encoding == "rgb8":
                    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            self.feed_image(
                source_id, pixels, stamp, provenance=binding.get("provenance", "real")
            )
        except Exception as exc:  # noqa: BLE001 — return transport/source diagnostics
            self.reject(source_id, exc)

    def feed_image(self, source_id, pixels, captured_at, *, provenance):
        """Also supports labelled recorded-data tests; no reception-time retiming."""
        with self.condition:
            need(not self.closed and not self.cancelled.is_set(), "SOURCE_CLOSED")
            b = self.config["sources"][source_id]
            need(provenance == b.get("provenance", "real"), "PROVENANCE_MISMATCH")
            need(
                provenance == "real" or b.get("allow_test_data") is True,
                "TEST_DATA_REFUSED",
            )
            need(
                b.get("view_mode") == "fixed" and b.get("fixed_view_record"),
                "VIEW_EVIDENCE_REQUIRED",
                "原始图像缺少逐帧位姿证据：须核对固定拍摄几何，移动相机请提供逐帧注册观测",
            )
            need(
                isinstance(pixels, np.ndarray)
                and pixels.dtype == np.uint8
                and pixels.ndim == 3
                and pixels.shape[2] == 3,
                "INVALID_IMAGE",
            )
            need(
                1 <= pixels.shape[0] <= 8192
                and 1 <= pixels.shape[1] <= 8192
                and pixels.nbytes <= 32 * 1024 * 1024,
                "INVALID_IMAGE_SIZE",
            )
            if source_id in self.declared_sizes:
                need(
                    self.declared_sizes[source_id]
                    == [pixels.shape[1], pixels.shape[0]],
                    "CAMERA_DECLARED_SIZE_MISMATCH",
                )
            need(
                0 <= time.time() - number(captured_at) <= self.engine.max_age,
                "STALE_OR_FUTURE",
            )
            previous = self.frames.get(source_id)
            need(
                previous is None or captured_at > previous[1]["captured_at"],
                "IMAGE_REPLAY",
            )
            if (
                previous
                and captured_at - previous[1]["captured_at"] > self.engine.max_age
            ):
                self.source_epochs[source_id] = uuid4().hex
            sequence = self.sequence.get(source_id, 0) + 1
            self.sequence[source_id] = sequence
            snapshot = pixels.copy()
            meta = {
                "schema": OBSERVATION,
                "observation_id": uuid4().hex,
                "source_id": source_id,
                "source_epoch": self.source_epochs[source_id],
                "sequence": sequence,
                "camera_id": b["camera_id"],
                "frame_id": b["frame_id"],
                "view_id": b["view_id"],
                "image_size": [pixels.shape[1], pixels.shape[0]],
                "distortion": b["distortion"],
                "captured_at": captured_at,
                "timestamp_basis": "capture",
                "provenance": provenance,
                "image_sha256": hashlib.sha256(snapshot.tobytes()).hexdigest(),
            }
            if b.get("calibration_id"):
                meta["calibration_id"] = b["calibration_id"]
            self.frames[source_id] = (snapshot, meta)
            self.errors.pop(source_id, None)
            self.condition.notify_all()

    def preview(self, source_id, observation_id):
        with self.condition:
            return deepcopy(self.previews.get((source_id, observation_id)))

    def read_image(self, source_id, *, after=None, timeout=3):
        """Read a new camera frame directly, without invoking either detector."""
        from .roi import FRAME_SCHEMA

        deadline = time.monotonic() + timeout
        with self.condition:
            need(source_id in self.config["sources"], "UNKNOWN_SOURCE")
            need(
                self.config["sources"][source_id].get("kind")
                in ("ros_image", "buffered_image"),
                "ROI_IMAGE_SOURCE_REQUIRED",
            )
            while True:
                need(not self.closed and not self.cancelled.is_set(), "SOURCE_CLOSED")
                need(
                    not self.errors.get(source_id),
                    "SOURCE_INPUT_ERROR",
                    self.errors.get(source_id),
                )
                frame = self.frames.get(source_id)
                if frame and (after is None or frame[1]["captured_at"] > after):
                    break
                remaining = deadline - time.monotonic()
                need(
                    remaining > 0,
                    "SOURCE_TIMEOUT",
                    "等待新的相机帧超时；不能复用旧帧验证",
                )
                self.condition.wait(min(remaining, 0.1))
            image, meta = frame[0].copy(), deepcopy(frame[1])
            need(
                0 <= time.time() - meta["captured_at"] <= self.engine.max_age,
                "STALE_OR_FUTURE",
            )
            meta["schema"] = FRAME_SCHEMA
            scale = min(1.0, 640 / image.shape[1], 480 / image.shape[0])
            thumbnail = cv2.resize(
                image,
                (
                    max(1, round(image.shape[1] * scale)),
                    max(1, round(image.shape[0] * scale)),
                ),
            )
            ok, jpeg = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 75])
            need(ok, "IMAGE_ENCODING_FAILED")
            self.previews[(source_id, meta["observation_id"])] = {
                "observation": deepcopy(meta),
                "jpeg_base64": base64.b64encode(jpeg).decode(),
                "historical_frame": True,
            }
            while len(self.previews) > 64:
                self.previews.popitem(last=False)
            return {"frame": meta, "pixels": image}

    def start_ros(self, executor=None):
        bindings = [
            (k, v)
            for k, v in self.config["sources"].items()
            if v.get("kind", "").startswith("ros_")
        ]
        if not bindings:
            return
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, Image

        if executor is None:
            from rclpy.context import Context
            from rclpy.executors import MultiThreadedExecutor

            domains = {b["domain_id"] for _, b in bindings}
            need(len(domains) == 1, "ONE_VISUAL_DOMAIN_PER_SERVICE")
            context = Context()
            # Record ownership before init/executor construction so partial
            # startup failures can release an already initialized context.
            self.ros = (context, None)
            rclpy.init(context=context, domain_id=next(iter(domains)))
            executor = MultiThreadedExecutor(context=context)
            self.ros = (context, executor)
        for key, b in bindings:
            need(
                type(b["domain_id"]) is int and b["domain_id"] == 42,
                "ROS_DOMAIN_42_REQUIRED",
            )
            context = executor.context
            domain = getattr(context, "get_domain_id", None)
            need(
                callable(domain) and domain() == b["domain_id"],
                "ROS_CONTEXT_DOMAIN_MISMATCH",
            )
            node = Node(
                "tabletop_roi_" + uuid4().hex[:12],
                context=context,
                enable_rosout=False,
                start_parameter_services=False,
            )
            self.nodes.append((node, executor))
            typ = CompressedImage if b["message_type"] == "compressed" else Image
            node.create_subscription(
                typ,
                b["topic"],
                lambda msg, k=key: self.image(k, msg),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            )
            executor.add_node(node)
        if self.ros:

            def spin():
                try:
                    executor.spin()
                except Exception as exc:  # noqa: BLE001 — wake waiters on ROS failure
                    for key, _ in bindings:
                        self.reject(key, exc)

            self.thread = threading.Thread(target=spin, daemon=True)
            self.thread.start()

    def interrupt(self):
        """Wake acquisition waiters without waiting for the plugin lifecycle lock."""
        self.cancelled.set()
        with self.condition:
            self.condition.notify_all()

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
            self.cancelled.set()
            self.epoch = uuid4().hex
            self.frames.clear()
            self.previews.clear()
            self.condition.notify_all()
        cleanup_errors = []

        def attempt(operation):
            try:
                operation()
            except Exception as exc:  # noqa: BLE001 — finish all owned cleanup
                cleanup_errors.append(str(exc))

        if self.ros and self.ros[1]:
            attempt(lambda: self.ros[1].shutdown(timeout_sec=2))
            if self.thread:
                attempt(lambda: self.thread.join(2))
        for node, executor in self.nodes:
            attempt(lambda node=node, executor=executor: executor.remove_node(node))
            attempt(node.destroy_node)
        self.nodes = []
        if self.ros:
            context, _ = self.ros
            self.ros = None
            attempt(lambda: context.shutdown() if context.ok() else None)
        if cleanup_errors:
            raise RuntimeError("ROS_CLEANUP_FAILED: " + "; ".join(cleanup_errors))
