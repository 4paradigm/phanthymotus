"""
plugins/htmsg/odometry_process.py — KISS-ICP based LiDAR odometry subprocess.

Follows the SmartMotionProxy/Process pattern from safety_harness.py:
- OdometryProxy: main process lightweight proxy, receives poses via mp.Queue
- _run_odometry_process: subprocess entry, subscribes DDS LiDAR, runs KISS-ICP
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import struct
import threading
import time
from typing import Callable, Optional

import numpy as np

from .types import Pose6D

log = logging.getLogger(__name__)


class OdometryProxy:
    """Main-process proxy that receives odometry from KISS-ICP subprocess."""

    def __init__(self, namespace: str, config: dict):
        self._namespace = namespace
        self._config = config
        self._callback: Optional[Callable] = None
        self._proc: Optional[mp.Process] = None
        self._pose_queue: Optional[mp.Queue] = None
        self._cmd_queue: Optional[mp.Queue] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._running = False

    def set_callback(self, cb: Callable[[Pose6D, float, Optional[np.ndarray]], None]):
        """Set callback: (pose, timestamp, cloud_xyz) called on each new pose."""
        self._callback = cb

    def start(self):
        """Start the odometry subprocess."""
        if self._running:
            return

        ctx = mp.get_context("spawn")
        self._pose_queue = ctx.Queue(maxsize=200)
        self._cmd_queue = ctx.Queue()

        self._proc = ctx.Process(
            target=_run_odometry_process,
            args=(self._namespace, self._config, self._pose_queue, self._cmd_queue),
            name="htmsg_odometry",
            daemon=True,
        )
        self._proc.start()
        log.info(f"[htmsg:odom] subprocess started → pid={self._proc.pid}")

        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume_poses, daemon=True, name="htmsg_odom_consumer"
        )
        self._consumer_thread.start()

    def stop(self):
        """Stop the subprocess."""
        self._running = False
        if self._cmd_queue:
            try:
                self._cmd_queue.put({"cmd": "shutdown"})
            except Exception:
                pass
        if self._proc and self._proc.is_alive():
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
        self._proc = None
        log.info("[htmsg:odom] subprocess stopped")

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _consume_poses(self):
        """Background thread: reads poses from subprocess queue and invokes callback."""
        while self._running:
            try:
                item = self._pose_queue.get(timeout=1.0)
            except Exception:
                continue

            if item is None:
                break

            pose_data = item["pose"]
            timestamp = item["timestamp"]
            cloud_xyz = item.get("cloud_xyz")  # Optional Nx3 numpy array

            pose = Pose6D(
                x=pose_data[0], y=pose_data[1], z=pose_data[2],
                qw=pose_data[3], qx=pose_data[4], qy=pose_data[5], qz=pose_data[6],
            )

            if self._callback:
                try:
                    self._callback(pose, timestamp, cloud_xyz)
                except Exception as e:
                    log.warning(f"[htmsg:odom] callback error: {e}")


def _run_odometry_process(namespace: str, config: dict,
                          pose_queue: mp.Queue, cmd_queue: mp.Queue):
    """Subprocess entry: subscribes to ROS2 LiDAR topic, runs KISS-ICP, outputs poses.

    This runs in its own process with its own GIL — no contention with perception main.
    Uses ROS2 subscription to /{namespace}/lidar/cloud (passthrough from driver).
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"[htmsg:odom:pid={os.getpid()}] starting KISS-ICP odometry subprocess")

    # ── Try to import KISS-ICP ──
    try:
        from kiss_icp.pipeline import OdometryPipeline
        from kiss_icp.config import KISSConfig
        HAS_KISS_ICP = True
    except ImportError:
        HAS_KISS_ICP = False
        print("[htmsg:odom] kiss-icp not installed, using simple ICP fallback")

    # ── Initialize ROS2 ──
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import UInt8MultiArray

    rclpy.init()

    _QOS = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        durability=DurabilityPolicy.VOLATILE,
    )

    # ── State ──
    running = True
    pose_matrix = np.eye(4, dtype=np.float64)  # cumulative T_world_body
    frame_count = 0

    # KISS-ICP pipeline (if available)
    kiss_pipeline = None
    if HAS_KISS_ICP:
        try:
            kiss_cfg = KISSConfig()
            kiss_cfg.data.max_range = 100.0
            kiss_cfg.data.min_range = 0.5
            kiss_cfg.data.deskew = False
            # kiss-icp >=1.0 removed the standalone OdometryPipeline constructor;
            # use KissICP core directly instead.
            from kiss_icp.kiss_icp import KissICP
            kiss_pipeline = KissICP(config=kiss_cfg)
            print("[htmsg:odom] KISS-ICP pipeline initialized (KissICP core)")
        except Exception as _e:
            print(f"[htmsg:odom] KISS-ICP init failed ({_e}), using simple ICP fallback")
            kiss_pipeline = None

    def _matrix_to_pose(T: np.ndarray) -> tuple:
        """Extract translation + quaternion from 4x4 homogeneous matrix."""
        t = T[:3, 3]
        R = T[:3, :3]
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return (float(t[0]), float(t[1]), float(t[2]),
                float(qw), float(qx), float(qy), float(qz))

    def _parse_ros2_cloud(data: bytes) -> Optional[np.ndarray]:
        """Parse ROS2 UInt8MultiArray from LidarPlugin passthrough.

        Format: [uint32 point_step][uint32 total_points][raw PointCloud2 bytes]
        PointCloud2 fields assumed: x(offset 0), y(offset 4), z(offset 8) as float32.
        """
        if len(data) < 8:
            return None

        point_step = struct.unpack_from('<I', data, 0)[0]
        total_points = struct.unpack_from('<I', data, 4)[0]
        raw_data = data[8:]

        if total_points == 0 or point_step == 0:
            return None

        # Limit to 100k points for performance
        num_points = min(total_points, 100000)
        available = len(raw_data) // point_step
        num_points = min(num_points, available)

        if num_points < 10:
            return None

        # Vectorized parsing — assume x at offset 0, y at 4, z at 8 (standard Livox layout)
        raw = np.frombuffer(raw_data, dtype=np.uint8, count=num_points * point_step)
        raw = raw.reshape(num_points, point_step)

        x = raw[:, 0:4].view(np.float32).ravel()
        y = raw[:, 4:8].view(np.float32).ravel()
        z = raw[:, 8:12].view(np.float32).ravel()

        # Filter invalid
        valid = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
            (np.abs(x) < 100) & (np.abs(y) < 100) & (np.abs(z) < 50)
        )
        x, y, z = x[valid], y[valid], z[valid]

        if len(x) < 10:
            return None

        # Filter by range (min 0.5m, max 100m)
        r2 = x * x + y * y + z * z
        range_valid = (r2 > 0.25) & (r2 < 10000)
        x, y, z = x[range_valid], y[range_valid], z[range_valid]

        if len(x) < 10:
            return None

        return np.column_stack([x, y, z]).astype(np.float64)

    class _OdomNode(Node):
        def __init__(self):
            super().__init__("htmsg_odom")
            cloud_topic = f"/{namespace}/lidar/cloud"
            self._sub = self.create_subscription(
                UInt8MultiArray, cloud_topic, self._on_cloud, _QOS
            )
            print(f"[htmsg:odom] subscribed to ROS2 topic: {cloud_topic}")

        def _on_cloud(self, msg):
            nonlocal pose_matrix, frame_count, running

            if not running:
                return

            data = msg.data if isinstance(msg.data, (bytes, bytearray)) else bytes(msg.data)
            points = _parse_ros2_cloud(data)
            if points is None:
                return

            now = time.time()
            frame_count += 1

            # Run KISS-ICP
            if kiss_pipeline is not None:
                try:
                    kiss_pipeline.register_frame(points, timestamps=None)
                    pose_matrix = kiss_pipeline.poses[-1] if kiss_pipeline.poses else np.eye(4)
                except Exception as e:
                    if frame_count <= 3:
                        print(f"[htmsg:odom] KISS-ICP error (frame {frame_count}): {e}")
                    return
            else:
                return

            pose_tuple = _matrix_to_pose(pose_matrix)

            # Subsample cloud for keyframe storage (max 10k points)
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                cloud_for_kf = points[indices].astype(np.float32)
            else:
                cloud_for_kf = points.astype(np.float32)

            try:
                pose_queue.put_nowait({
                    "pose": pose_tuple,
                    "timestamp": now,
                    "cloud_xyz": cloud_for_kf,
                })
            except Exception:
                pass  # queue full, drop frame

    # ── Create node and spin ──
    node = _OdomNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # ── Main loop: spin ROS2 + check for shutdown ──
    while running:
        executor.spin_once(timeout_sec=0.05)
        try:
            cmd = cmd_queue.get_nowait()
            if cmd and cmd.get("cmd") == "shutdown":
                running = False
                break
        except Exception:
            pass

    node.destroy_node()
    rclpy.shutdown()
    print(f"[htmsg:odom:pid={os.getpid()}] shutting down (processed {frame_count} frames)")
