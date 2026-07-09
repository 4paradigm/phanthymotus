#!/usr/bin/env python3
"""
plugins/stereo.py — StereoPointCloudPlugin: stereo image pair to colored point cloud.

Subscribes to a rectified left/right image/jpeg topic pair, computes disparity
(OpenCV SGBM by default), reprojects to 3D via cv2.reprojectImageTo3D and
publishes sensor/pointcloud frames in the same passthrough binary format as
the lidar drivers: [uint32 point_step][uint32 total_points][raw point bytes].
Supports multi-instance (one instance per stereo pair).

Matching backends are pluggable via StereoMatcher — future deep-learning
matchers (RAFT-Stereo, IGEV, ...) plug in as new subclasses selected by the
`matcher` config key.
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
from abc import ABC, abstractmethod
from array import array
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt8MultiArray

log = logging.getLogger(__name__)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

# Published point layout: x, y, z float32 + packed rgb uint32 (0x00RRGGBB)
_POINT_STEP = 16

TOOLS = [
    {
        "name": "stereo",
        "type": "processor",
        "multiInstance": True,
        "description": "Stereo Point Cloud — convert a rectified stereo image pair to a colored 3D point cloud",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Left and right ROS2 image topics, in order [left, right] (required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "fps":        {"type": "integer", "description": "Max stereo matching frames per second", "default": 5, "scope": "instance"},
                "sync_ms":    {"type": "number", "description": "Max left/right timestamp difference (ms) to accept a pair", "default": 50, "scope": "instance"},
                "max_points": {"type": "integer", "description": "Max points per published cloud (randomly downsampled above)", "default": 40000, "scope": "instance"},
                "fx":         {"type": "number", "description": "Focal length in pixels from stereo calibration (0 = synthetic, 0.8×width)", "default": 0, "scope": "instance"},
                "baseline":   {"type": "number", "description": "Stereo baseline in meters from calibration (0 = arbitrary scale)", "default": 0, "scope": "instance"},
                "cx":         {"type": "number", "description": "Principal point x in pixels (0 = image center)", "default": 0, "scope": "instance"},
                "cy":         {"type": "number", "description": "Principal point y in pixels (0 = image center)", "default": 0, "scope": "instance"},
                "max_depth":  {"type": "number", "description": "Discard points beyond this depth in meters (0 = no limit)", "default": 0, "scope": "instance"},
            },
        },
        "topic_in": [
            {"format": "image/jpeg", "desc": "left camera image (rectified)"},
            {"format": "image/jpeg", "desc": "right camera image (rectified)"},
        ],
        "topic_out": [{"format": "sensor/pointcloud", "desc": "colored 3D point cloud"}],
    }
]


def _import_cv2():
    """Import cv2, fixing the broken system cv2 on Jetson (circular import
    in mat_wrapper) by loading the .so directly — same workaround as vop.py."""
    try:
        import cv2
        _ = cv2.IMREAD_COLOR  # test if cv2 is functional
        return cv2
    except (ImportError, AttributeError):
        import glob as _glob
        import importlib.util
        import sys as _sys
        _so_candidates = _glob.glob("/usr/lib/python*/dist-packages/cv2/python-*/cv2.cpython-*.so")
        if _so_candidates:
            _spec = importlib.util.spec_from_file_location("cv2", _so_candidates[0])
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _sys.modules["cv2"] = _mod
        import cv2
        return cv2


# ── Stereo matchers ───────────────────────────────────────────────────────────

class StereoMatcher(ABC):
    """Disparity backend. compute() takes a rectified BGR image pair and
    returns a float32 disparity map in pixels, with invalid pixels set to 0."""

    @abstractmethod
    def compute(self, imgL: np.ndarray, imgR: np.ndarray) -> np.ndarray: ...


class SGBMMatcher(StereoMatcher):
    """OpenCV semi-global block matching (CPU, no model weights needed)."""

    def __init__(self, sgbm_cfg: dict):
        cv2 = _import_cv2()
        self._cv2 = cv2
        self._min_disp = int(sgbm_cfg.get("min_disparity", 16))
        num_disp = int(sgbm_cfg.get("num_disparities", 144))
        num_disp = max(16, (num_disp // 16) * 16)  # must be a multiple of 16
        block_size = int(sgbm_cfg.get("block_size", 5))
        self._stereo = cv2.StereoSGBM_create(
            minDisparity=self._min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,       # smoothness penalty (small disparity jumps)
            P2=32 * 3 * block_size ** 2,      # smoothness penalty (large disparity jumps)
            disp12MaxDiff=1,                  # left-right consistency check threshold
            uniquenessRatio=10,               # best match must beat 2nd best by this %
            speckleWindowSize=100,            # remove small noisy connected regions
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, imgL: np.ndarray, imgR: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        # compute() returns fixed-point disparity in 1/16 pixel units
        disparity = self._stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        disparity[disparity < self._min_disp] = 0.0  # mark invalid pixels
        return disparity


def _build_stereo_matcher(name: str, plugin_cfg: dict) -> StereoMatcher:
    if name == "sgbm":
        return SGBMMatcher(plugin_cfg.get("sgbm") or {})
    raise ValueError(f"unknown stereo matcher: {name}")


# ── Reprojection ──────────────────────────────────────────────────────────────

def _build_q_matrix(w: int, h: int, calib: dict) -> np.ndarray:
    """Build the disparity-to-depth matrix Q.

    With real calibration (fx + baseline from cv2.stereoRectify) the cloud is
    metric. Without it, fx falls back to 0.8×width and baseline to 1.0 —
    shape is correct but the absolute scale is arbitrary.
    Y axis is flipped so "up" in the cloud matches "up" in the image.
    """
    fx = float(calib.get("fx") or 0) or 0.8 * w
    cx = float(calib.get("cx") or 0) or 0.5 * w
    cy = float(calib.get("cy") or 0) or 0.5 * h
    baseline = float(calib.get("baseline") or 0) or 1.0
    return np.float32([
        [1, 0, 0, -cx],
        [0, -1, 0, cy],
        [0, 0, 0, fx],
        [0, 0, 1.0 / baseline, 0],
    ])


# ── ROS2 Node (one per instance/stereo pair) ──────────────────────────────────

class _StereoNode(Node):
    """Per-pair stereo matching node: left+right images in, point cloud out."""

    def __init__(self, left_topic: str, right_topic: str, matcher: StereoMatcher,
                 calib: dict, fps: float, sync_ms: float, max_points: int,
                 max_depth: float, node_suffix: str):
        super().__init__(f"stereo_{node_suffix}")
        self._left_topic = left_topic
        self._right_topic = right_topic
        self._output_topic = f"{left_topic}/cloud"
        self._matcher = matcher
        self._calib = calib
        self._fps = fps
        self._frame_interval = 1.0 / max(fps, 0.1)
        self._sync_s = max(sync_ms, 0.0) / 1000.0
        self._max_points = max_points
        self._max_depth = max_depth

        self._pub = self.create_publisher(UInt8MultiArray, self._output_topic, _PUB_QOS)
        self._subs: list = []
        self._latest: list = [None, None]  # per side: (stamp, jpeg bytes)
        self._pair_lock = threading.Lock()
        self._pair_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_pair_time = 0.0
        self._q_matrix: Optional[np.ndarray] = None  # built lazily from first frame size
        self._cloud_count = 0

    def start(self) -> dict:
        if self._subs:
            return {"state": "running", "input": [self._left_topic, self._right_topic],
                    "output": self._output_topic}
        self._stop_event.clear()
        self._subs = [
            self.create_subscription(
                CompressedImage, self._left_topic,
                lambda msg: self._image_cb(msg, 0), _LOW_LAT_QOS),
            self.create_subscription(
                CompressedImage, self._right_topic,
                lambda msg: self._image_cb(msg, 1), _LOW_LAT_QOS),
        ]
        self._worker = threading.Thread(target=self._matching_worker, daemon=True,
                                        name=f"stereo_worker_{self._left_topic}")
        self._worker.start()
        log.info(f"[stereo] started: [{self._left_topic}, {self._right_topic}] → {self._output_topic}")
        return {"state": "running", "input": [self._left_topic, self._right_topic],
                "output": self._output_topic,
                "topic_out": [{"topic": self._output_topic, "format": "sensor/pointcloud"}]}

    def stop(self) -> dict:
        for sub in self._subs:
            self.destroy_subscription(sub)
        self._subs = []
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        log.info(f"[stereo] stopped: {self._left_topic}")
        return {"state": "idle", "input": [self._left_topic, self._right_topic]}

    def _image_cb(self, msg: CompressedImage, side: int):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0:
            stamp = time.time()  # driver does not fill header.stamp — use arrival time
        with self._pair_lock:
            self._latest[side] = (stamp, bytes(msg.data))
            left, right = self._latest
            if left is None or right is None:
                return
            if abs(left[0] - right[0]) > self._sync_s:
                return
            now = time.monotonic()
            if now - self._last_pair_time < self._frame_interval:
                return
            self._last_pair_time = now
            pair = (left[1], right[1])
            self._latest = [None, None]  # consume the pair
        # Drop old pair if queue full (no backpressure)
        try:
            self._pair_queue.put_nowait(pair)
        except queue.Full:
            try:
                self._pair_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pair_queue.put_nowait(pair)
            except queue.Full:
                pass

    def _matching_worker(self):
        cv2 = _import_cv2()
        while not self._stop_event.is_set():
            try:
                jpeg_left, jpeg_right = self._pair_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                imgL = cv2.imdecode(np.frombuffer(jpeg_left, np.uint8), cv2.IMREAD_COLOR)
                imgR = cv2.imdecode(np.frombuffer(jpeg_right, np.uint8), cv2.IMREAD_COLOR)
                if imgL is None or imgR is None:
                    continue
                if imgL.shape != imgR.shape:
                    log.warning(f"[stereo] left/right size mismatch: {imgL.shape} vs {imgR.shape}, skipped")
                    continue
                disparity = self._matcher.compute(imgL, imgR)
                payload = self._disparity_to_cloud(cv2, disparity, imgL)
                if payload is not None:
                    self._publish_cloud(payload)
            except Exception as e:
                log.error(f"[stereo] matching error: {e}", exc_info=True)

    def _disparity_to_cloud(self, cv2, disparity: np.ndarray, imgL: np.ndarray) -> Optional[bytes]:
        """Reproject disparity to 3D and pack the passthrough binary payload."""
        h, w = disparity.shape
        if self._q_matrix is None:
            self._q_matrix = _build_q_matrix(w, h, self._calib)
        points_3d = cv2.reprojectImageTo3D(disparity, self._q_matrix)  # (h, w, 3)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)                 # (h, w, 3)

        mask = disparity > 0
        mask &= np.isfinite(points_3d).all(axis=2)
        if self._max_depth > 0:
            mask &= points_3d[:, :, 2] < self._max_depth

        pts = points_3d[mask]
        cols = colors[mask]
        n = len(pts)
        if n == 0:
            return None
        if n > self._max_points:
            idx = np.random.choice(n, self._max_points, replace=False)
            pts, cols = pts[idx], cols[idx]
            n = self._max_points

        # [x f32][y f32][z f32][rgb u32 packed 0x00RRGGBB] per point
        rgb = ((cols[:, 0].astype(np.uint32) << 16)
               | (cols[:, 1].astype(np.uint32) << 8)
               | cols[:, 2].astype(np.uint32))
        buf = np.empty((n, 4), dtype=np.float32)
        buf[:, :3] = pts
        buf[:, 3] = rgb.view(np.float32)
        return struct.pack("<II", _POINT_STEP, n) + buf.tobytes()

    def _publish_cloud(self, payload: bytes):
        self._cloud_count += 1
        msg = UInt8MultiArray()
        msg.data = array("B", payload)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class StereoPointCloudPlugin:
    PREFIX = "stereo"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = plugin_cfg
        self._executor = executor
        self._matcher_name = plugin_cfg.get("matcher", "sgbm")
        self._fps = int(plugin_cfg.get("fps", 5))
        self._sync_ms = float(plugin_cfg.get("sync_ms", 50))
        self._max_points = int(plugin_cfg.get("max_points", 40000))
        self._max_depth = float(plugin_cfg.get("max_depth", 0))
        self._calib_keys = ("fx", "baseline", "cx", "cy")
        self._nodes: dict[str, _StereoNode] = {}
        self._instance_configs: dict[str, dict] = {}  # per-instance config overrides

    def _resolve_input_topics(self, args: dict) -> list[str]:
        """Collect input topics from args: input_topics list or single input_topic."""
        topics = list(args.get("input_topics") or [])
        single = args.get("input_topic")
        if single and single not in topics:
            topics.insert(0, single)
        return topics

    def _start_node(self, node_key: str, left_topic: str, right_topic: str):
        """Create and start a StereoNode for the given topic pair."""
        icfg = self._instance_configs.get(node_key, {})
        matcher = _build_stereo_matcher(self._matcher_name, self._cfg)
        calib = {k: icfg.get(k, self._cfg.get(k, 0)) for k in self._calib_keys}
        fps = int(icfg.get("fps", self._fps))
        sync_ms = float(icfg.get("sync_ms", self._sync_ms))
        max_points = int(icfg.get("max_points", self._max_points))
        max_depth = float(icfg.get("max_depth", self._max_depth))
        suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
        node = _StereoNode(left_topic, right_topic, matcher, calib, fps, sync_ms,
                           max_points, max_depth, node_suffix=suffix)
        self._executor.add_node(node)
        self._nodes[node_key] = node

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            instances = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": [node._left_topic, node._right_topic],
                    "output": node._output_topic,
                    "fps": node._fps,
                    "cloud_count": node._cloud_count,
                }
            # Determine topic info: from running instance, args, or empty
            topics = self._resolve_input_topics(args)
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                topics = [node._left_topic, node._right_topic]
            elif not topics and self._nodes:
                first_node = next(iter(self._nodes.values()))
                topics = [first_node._left_topic, first_node._right_topic]
            topics_in = [{"topic": t, "format": "image/jpeg"} for t in topics]
            topics_out = [{"topic": f"{topics[0]}/cloud", "format": "sensor/pointcloud"}] if topics else []
            state = "running" if instances else "idle"
            return {
                "name": "StereoPointCloud", "manufacture": "Embodied", "model": self._matcher_name,
                "state": state,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Stereo matching point cloud (first connected topic = left camera)",
            }

        elif action == "start":
            topics = self._resolve_input_topics(args)
            if len(topics) < 2:
                raise ValueError("stereo requires two input topics: connect both left and right image topics (first = left)")
            left_topic, right_topic = topics[0], topics[1]
            node_key = instance_id or left_topic
            if node_key not in self._nodes:
                self._start_node(node_key, left_topic, right_topic)
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                results = []
                for key in list(self._nodes.keys()):
                    node = self._nodes[key]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[key]
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            if instance_id:
                self._instance_configs[instance_id] = cfg
                # If instance is running, stop it — re-created with new config on next start
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[instance_id]
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                # Update global defaults
                if "fps" in cfg:
                    self._fps = int(cfg["fps"])
                if "sync_ms" in cfg:
                    self._sync_ms = float(cfg["sync_ms"])
                if "max_points" in cfg:
                    self._max_points = int(cfg["max_points"])
                if "max_depth" in cfg:
                    self._max_depth = float(cfg["max_depth"])
                for k in self._calib_keys:
                    if k in cfg:
                        self._cfg[k] = float(cfg[k])
                return {"status": "configured", "config": cfg}

        return None
