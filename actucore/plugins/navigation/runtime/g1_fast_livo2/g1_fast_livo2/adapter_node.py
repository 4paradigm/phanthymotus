"""Normalize FAST-LIVO2 raw outputs into the canonical Nav2 contract."""

from __future__ import annotations

from collections import deque
from itertools import chain
import json
import math
from pathlib import Path
import re
import threading
import time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from .frame_adapter_core import (
    InvalidFastLivo2Frame,
    OdomHealthMonitor,
    Pose3,
    Quaternion,
    RelocalizationRejected,
    TemporalOccupancyMap,
    VoxelMap,
    bracketed_stamped_pose,
    canonical_base_pose,
    compose_pose,
    estimate_planar_relocalization,
    encode_map_view_points,
    normalize_obstacle_height_range,
    obstacle_height_ranges_match,
    quaternion_from_rpy,
    read_pcd_xyz,
    source_age_is_valid,
    write_pcd_xyz_atomic,
    yaw_from_quaternion,
)
from .vectorized_cloud import (
    absolute_point_time_span_ms,
    decode_xyz_array,
    map_view_with_pose,
    transform_xyz_array,
    xyz_array_bytes,
)


_MAP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MAP_VIEW_MAX_POINTS = 40_000
_MAP_VIEW_POSE_REFRESH_HZ = 1.0
_RELOCALIZATION_MIN_MATCH_RATIO = 0.35
_RELOCALIZATION_HISTORY_SEC = 2.0
_RELOCALIZATION_MAX_FRAMES = 20
_RELOCALIZATION_MAX_POINTS_PER_FRAME = 2_000


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class FastLivo2Adapter(Node):
    def __init__(self) -> None:
        super().__init__("g1_fast_livo2_adapter")
        self.declare_parameter("raw_odom_topic", "/ubuntu/navigation/fast_livo2/raw/odom")
        self.declare_parameter("raw_cloud_topic", "/ubuntu/navigation/fast_livo2/raw/cloud_registered")
        self.declare_parameter("lidar_topic", "/ubuntu/navigation/lidar")
        self.declare_parameter("imu_topic", "/ubuntu/navigation/imu")
        self.declare_parameter("odom_topic", "/ubuntu/navigation/odom")
        self.declare_parameter("cloud_topic", "/ubuntu/navigation/cloud_registered")
        self.declare_parameter("obstacle_map_topic", "/ubuntu/navigation/obstacle_map")
        self.declare_parameter("static_map_topic", "/ubuntu/navigation/static_map")
        self.declare_parameter("map_view_topic", "/ubuntu/navigation/fast_livo2/map_view")
        self.declare_parameter("diagnostics_topic", "/ubuntu/navigation/fast_livo2/diagnostics")
        self.declare_parameter("reset_topic", "/ubuntu/navigation/fast_livo2/reset_map")
        self.declare_parameter("map_control_topic", "/ubuntu/navigation/fast_livo2/map_control")
        self.declare_parameter("map_control_status_topic", "/ubuntu/navigation/fast_livo2/map_control_status")
        self.declare_parameter("map_root", "/opt/fast_livo_ws/src/fast_livo/Log/pcd")
        self.declare_parameter("map_load_max_points", 200000)
        self.declare_parameter("static_map_load_max_points", 200000)
        self.declare_parameter("live_cloud_max_bytes", 67108864)
        self.declare_parameter("source_max_age_sec", 0.5)
        self.declare_parameter("source_age_tolerance_sec", 0.05)
        self.declare_parameter("map_voxel_size_m", 0.10)
        self.declare_parameter("static_angular_bin_deg", 1.0)
        self.declare_parameter("static_grid_margin_m", 6.0)
        self.declare_parameter("static_pose_match_tolerance_sec", 0.05)
        self.declare_parameter("obstacle_min_height_m", -0.30)
        self.declare_parameter("obstacle_max_height_m", 0.30)

        self._source_max_age = float(self.get_parameter("source_max_age_sec").value)
        self._source_age_tolerance = float(
            self.get_parameter("source_age_tolerance_sec").value
        )
        if not 0 <= self._source_age_tolerance <= 0.1:
            raise ValueError("source_age_tolerance_sec must be within [0, 0.1]")
        self._map_root = Path(str(self.get_parameter("map_root").value)).resolve()
        self._map_load_max_points = int(
            self.get_parameter("map_load_max_points").value
        )
        self._static_map_load_max_points = int(
            self.get_parameter("static_map_load_max_points").value
        )
        self._live_cloud_max_bytes = int(
            self.get_parameter("live_cloud_max_bytes").value
        )
        if self._map_load_max_points < 40 or self._static_map_load_max_points < 40:
            raise ValueError("map load point limits must be at least 40")
        if self._live_cloud_max_bytes < 1:
            raise ValueError("live_cloud_max_bytes must be positive")
        self._base_to_sensor: Pose3 | None = None
        self._sensor_frame: str | None = None
        self._lidar_frame: str | None = None
        self._imu_frame: str | None = None
        self._last_lidar_source_stamp_ns: int | None = None
        self._last_imu_source_stamp_ns: int | None = None
        self._point_time_ready = False
        self._imu_time_ready = False
        self._point_time_span_ms: float | None = None
        self._base_to_sensor_tf_ready = False
        self._sensor_tf_error: str | None = None
        self._odom_health = OdomHealthMonitor()
        map_voxel_size = float(self.get_parameter("map_voxel_size_m").value)
        self._map_view_voxel_size = max(map_voxel_size, 0.20)
        self._map_view_context = VoxelMap(self._map_view_voxel_size)
        self._static_map = TemporalOccupancyMap(
            map_voxel_size,
            angular_bin_deg=float(
                self.get_parameter("static_angular_bin_deg").value
            ),
            grid_margin_m=float(
                self.get_parameter("static_grid_margin_m").value
            ),
            max_evidence_points=self._static_map_load_max_points,
        )
        self._static_pose_match_tolerance = float(
            self.get_parameter("static_pose_match_tolerance_sec").value
        )
        if not 0 < self._static_pose_match_tolerance <= 0.2:
            raise ValueError(
                "static_pose_match_tolerance_sec must be within (0, 0.2]"
            )
        self._static_pose_match_tolerance_ns = int(
            self._static_pose_match_tolerance * 1_000_000_000
        )
        self._obstacle_min_height = float(self.get_parameter("obstacle_min_height_m").value)
        self._obstacle_max_height = float(self.get_parameter("obstacle_max_height_m").value)
        if not math.isfinite(self._obstacle_min_height) or not math.isfinite(self._obstacle_max_height):
            raise ValueError("obstacle projection heights must be finite")
        if self._obstacle_min_height >= self._obstacle_max_height:
            raise ValueError("obstacle_min_height_m must be less than obstacle_max_height_m")
        self._lock = threading.Lock()
        self._rejection_counts: dict[str, int] = {}
        self._static_lock = threading.Lock()
        self._mapping_work_condition = threading.Condition()
        self._mapping_work = None
        self._mapping_generation = 0
        self._mapping_work_generation = 0
        self._mapping_work_latest_monotonic: float | None = None
        self._mapping_work_dropped = 0
        self._mapping_worker_stop = False
        self._latest_pose: Pose3 | None = None
        self._pose_history = deque(maxlen=128)
        self._latest_session_pose: Pose3 | None = None
        self._latest_session_points = ()
        self._relocalization_cloud_history = deque(
            maxlen=_RELOCALIZATION_MAX_FRAMES
        )
        self._relocalization_preview_pose: Pose3 | None = None
        self._relocalization_preview_points = ()
        self._latest_mapped_points = ()
        self._reference_points: tuple[tuple[float, float, float], ...] = ()
        self._map_from_session: Pose3 | None = None
        self._mode = "idle"
        self._last_match: dict | None = None
        self._last_odom_monotonic: float | None = None
        self._last_cloud_monotonic: float | None = None
        self._last_navigation_cloud_monotonic: float | None = None
        self._last_odom_source_age: float | None = None
        self._last_cloud_source_age: float | None = None
        self._invalid_odom = 0
        self._invalid_cloud = 0
        self._unmatched_navigation_cloud = 0
        self._unmatched_static_cloud = 0
        self._last_cloud_pose_skew_sec: float | None = None
        self._pending_cloud = None
        self._static_map_error: str | None = None
        self._static_save_result: dict | None = None
        self._session_name: str | None = None
        self._map_view_cache: bytes | None = None
        self._map_view_cache_monotonic: float | None = None
        self._latency_ms: dict[str, float] = {}
        self._latency_max_ms: dict[str, float] = {}
        self._static_map_load_time = self.get_clock().now().to_msg()
        self._callbacks = ReentrantCallbackGroup()
        self._display_callbacks = MutuallyExclusiveCallbackGroup()

        self._odom_pub = self.create_publisher(Odometry, str(self.get_parameter("odom_topic").value), qos_profile_sensor_data)
        self._cloud_pub = self.create_publisher(PointCloud2, str(self.get_parameter("cloud_topic").value), qos_profile_sensor_data)
        self._obstacle_map_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("obstacle_map_topic").value),
            qos_profile_sensor_data,
        )
        static_map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._static_map_pub = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter("static_map_topic").value),
            static_map_qos,
        )
        map_view_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._map_view_pub = self.create_publisher(
            UInt8MultiArray,
            str(self.get_parameter("map_view_topic").value),
            map_view_qos,
        )
        self._diagnostics_pub = self.create_publisher(String, str(self.get_parameter("diagnostics_topic").value), 10)
        self._map_control_status_pub = self.create_publisher(
            String, str(self.get_parameter("map_control_status_topic").value), 10
        )
        self._tf = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
            spin_thread=False,
        )
        latest_sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("lidar_topic").value),
            self._on_lidar_contract,
            latest_sensor_qos,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("imu_topic").value),
            self._on_imu_contract,
            latest_sensor_qos,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("raw_odom_topic").value),
            self._on_odom,
            latest_sensor_qos,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("raw_cloud_topic").value),
            self._on_cloud,
            latest_sensor_qos,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("reset_topic").value),
            self._on_reset,
            10,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("map_control_topic").value),
            self._on_map_control,
            10,
            callback_group=self._callbacks,
        )
        self.create_timer(
            1.0,
            self._publish_periodic,
            callback_group=self._display_callbacks,
        )
        self.create_timer(
            1.0 / _MAP_VIEW_POSE_REFRESH_HZ,
            self._publish_map_view,
            callback_group=self._display_callbacks,
        )
        self._mapping_worker = threading.Thread(
            target=self._mapping_worker_main,
            name="g1-fast-livo2-static-map",
            daemon=True,
        )
        self._mapping_worker.start()

    def destroy_node(self):
        with self._mapping_work_condition:
            self._mapping_worker_stop = True
            self._mapping_work = None
            self._mapping_work_condition.notify_all()
        if self._mapping_worker.is_alive():
            self._mapping_worker.join(timeout=2.0)
        return super().destroy_node()

    def _invalidate_mapping_work_locked(self) -> None:
        self._mapping_generation += 1
        with self._mapping_work_condition:
            self._mapping_work = None
            self._mapping_work_generation = self._mapping_generation
            self._mapping_work_latest_monotonic = None

    def _invalidate_map_view_cache_locked(self) -> None:
        self._map_view_cache = None
        self._map_view_cache_monotonic = None

    def _record_latency_locked(self, name: str, elapsed_sec: float) -> None:
        elapsed_ms = max(0.0, float(elapsed_sec) * 1000.0)
        if not hasattr(self, "_latency_ms"):
            self._latency_ms = {}
            self._latency_max_ms = {}
        self._latency_ms[name] = elapsed_ms
        self._latency_max_ms[name] = max(
            elapsed_ms,
            self._latency_max_ms.get(name, 0.0),
        )

    def _queue_mapping_scan(self, work: dict) -> None:
        with self._mapping_work_condition:
            generation = int(work["generation"])
            receive_monotonic = float(work["receive_monotonic"])
            if generation < self._mapping_work_generation or (
                generation == self._mapping_work_generation
                and self._mapping_work_latest_monotonic is not None
                and receive_monotonic <= self._mapping_work_latest_monotonic
            ):
                self._mapping_work_dropped += 1
                return
            if self._mapping_work is not None:
                self._mapping_work_dropped += 1
            self._mapping_work_generation = generation
            self._mapping_work_latest_monotonic = receive_monotonic
            self._mapping_work = work
            self._mapping_work_condition.notify()

    def _mapping_worker_main(self) -> None:
        while True:
            with self._mapping_work_condition:
                while self._mapping_work is None and not self._mapping_worker_stop:
                    self._mapping_work_condition.wait()
                if self._mapping_worker_stop:
                    return
                work = self._mapping_work
                self._mapping_work = None
            if not isinstance(work, dict):
                continue
            with self._lock:
                generation = self._mapping_generation
                if (
                    work.get("generation") != generation
                    or self._mode != "mapping"
                    or self._map_from_session != work.get("map_from_session")
                ):
                    continue
                map_view_context = self._map_view_context
            error = None
            try:
                with self._static_lock:
                    map_view_context.add(work["out_of_band_points"])
                    self._static_map.observe_scan(
                        sensor_origin=work["sensor_origin"],
                        points=work["mapped_points"],
                        now_monotonic=work["receive_monotonic"],
                        obstacle_min_height_m=work["obstacle_min_height"],
                        obstacle_max_height_m=work["obstacle_max_height"],
                    )
            except ValueError as exc:
                error = str(exc)
            if error is not None:
                with self._lock:
                    if (
                        self._mapping_generation == generation
                        and self._mode == "mapping"
                    ):
                        self._static_map_error = error
                        self._mode = "mapping_error"
                self.get_logger().error(
                    f"stopping static-map accumulation: {error}"
                )

    def _source_age(self, stamp) -> float:
        source = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9 - source

    def _warn_rejected(self, stream: str, detail) -> None:
        with self._lock:
            counts = getattr(self, "_rejection_counts", None)
            if counts is None:
                counts = self._rejection_counts = {}
            count = counts.get(stream, 0) + 1
            counts[stream] = count
        if count == 1 or count % 100 == 0:
            bounded = str(detail).replace("\r", "\\r").replace("\n", "\\n")[:200]
            self.get_logger().warning(
                f"rejecting {stream} count={count}: {bounded}"
            )

    def _mark_valid(self, stream: str) -> None:
        with self._lock:
            counts = getattr(self, "_rejection_counts", None)
            rejected = counts.pop(stream, 0) if counts else 0
        if rejected:
            self.get_logger().info(
                f"{stream} recovered after {rejected} rejected samples"
            )

    def _on_lidar_contract(self, message: PointCloud2) -> None:
        try:
            frame = message.header.frame_id.strip()
            if not frame:
                raise InvalidFastLivo2Frame("lidar frame_id is empty")
            source_stamp_ns = _stamp_ns(message.header.stamp)
            with self._lock:
                previous = self._last_lidar_source_stamp_ns
            if previous is not None and source_stamp_ns <= previous:
                raise InvalidFastLivo2Frame("lidar source stamp did not advance")
            span_ms = absolute_point_time_span_ms(
                fields=message.fields,
                data=bytes(message.data),
                point_step=int(message.point_step),
                row_step=int(message.row_step),
                width=int(message.width),
                height=int(message.height),
                is_bigendian=bool(message.is_bigendian),
                header_stamp_ns=source_stamp_ns,
            )
        except (InvalidFastLivo2Frame, TypeError, ValueError) as exc:
            with self._lock:
                self._point_time_ready = False
                self._point_time_span_ms = None
            self._warn_rejected("navigation lidar contract", exc)
            return
        self._mark_valid("navigation lidar contract")
        with self._lock:
            if frame != self._lidar_frame:
                self._base_to_sensor = None
                self._base_to_sensor_tf_ready = False
            self._lidar_frame = frame
            self._last_lidar_source_stamp_ns = source_stamp_ns
            self._point_time_ready = True
            self._point_time_span_ms = span_ms
        self._refresh_sensor_contract()

    def _on_imu_contract(self, message: Imu) -> None:
        try:
            frame = message.header.frame_id.strip()
            if not frame:
                raise InvalidFastLivo2Frame("imu frame_id is empty")
            source_stamp_ns = _stamp_ns(message.header.stamp)
            if source_stamp_ns <= 0:
                raise InvalidFastLivo2Frame("imu source stamp must be positive")
            with self._lock:
                previous = self._last_imu_source_stamp_ns
            if previous is not None and source_stamp_ns <= previous:
                raise InvalidFastLivo2Frame("imu source stamp did not advance")
        except (InvalidFastLivo2Frame, TypeError, ValueError) as exc:
            with self._lock:
                self._imu_time_ready = False
            self._warn_rejected("navigation imu contract", exc)
            return
        self._mark_valid("navigation imu contract")
        with self._lock:
            if frame != self._imu_frame:
                self._base_to_sensor = None
                self._base_to_sensor_tf_ready = False
            self._imu_frame = frame
            self._last_imu_source_stamp_ns = source_stamp_ns
            self._imu_time_ready = True
        self._refresh_sensor_contract()

    def _refresh_sensor_contract(self) -> None:
        with self._lock:
            lidar_frame = self._lidar_frame
            imu_frame = self._imu_frame
            lidar_stamp = self._last_lidar_source_stamp_ns
            imu_stamp = self._last_imu_source_stamp_ns
            if (
                not lidar_frame
                or lidar_frame != imu_frame
                or not self._point_time_ready
                or not self._imu_time_ready
                or lidar_stamp is None
                or imu_stamp is None
                or abs(lidar_stamp - imu_stamp) > 200_000_000
            ):
                self._sensor_frame = None
                self._base_to_sensor = None
                self._base_to_sensor_tf_ready = False
                return
            sensor_frame = lidar_frame
            if (
                self._sensor_frame == sensor_frame
                and self._base_to_sensor_tf_ready
                and self._base_to_sensor is not None
            ):
                return
        try:
            transform = self._tf_buffer.lookup_transform(
                "base_link",
                sensor_frame,
                Time(),
            ).transform
            base_to_sensor = Pose3(
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
                Quaternion(
                    float(transform.rotation.x),
                    float(transform.rotation.y),
                    float(transform.rotation.z),
                    float(transform.rotation.w),
                ),
            )
            # Validate both translation and quaternion before caching the TF.
            canonical_base_pose(base_to_sensor, base_to_sensor)
        except (TransformException, InvalidFastLivo2Frame, TypeError, ValueError) as exc:
            with self._lock:
                self._sensor_frame = sensor_frame
                self._base_to_sensor = None
                self._base_to_sensor_tf_ready = False
                self._sensor_tf_error = str(exc)
            return
        with self._lock:
            if self._lidar_frame == self._imu_frame == sensor_frame:
                self._sensor_frame = sensor_frame
                self._base_to_sensor = base_to_sensor
                self._base_to_sensor_tf_ready = True
                self._sensor_tf_error = None

    def _sensor_contract_ready_locked(self) -> bool:
        return (
            self._sensor_frame is not None
            and self._point_time_ready
            and self._imu_time_ready
            and self._base_to_sensor_tf_ready
            and self._base_to_sensor is not None
        )

    def _readiness_blockers_locked(self) -> list[str]:
        blockers = []
        if not self._lidar_frame or self._lidar_frame != self._imu_frame:
            blockers.append("sensor_frame_mismatch")
        if (
            not self._point_time_ready
            or not self._imu_time_ready
            or self._last_lidar_source_stamp_ns is None
            or self._last_imu_source_stamp_ns is None
            or abs(
                self._last_lidar_source_stamp_ns
                - self._last_imu_source_stamp_ns
            )
            > 200_000_000
        ):
            blockers.append("point_time_invalid")
        if (
            self._lidar_frame
            and self._lidar_frame == self._imu_frame
            and not self._base_to_sensor_tf_ready
        ):
            blockers.append("sensor_tf_unavailable")
        if self._odom_health.reason == "raw_odom_discontinuity":
            blockers.append("raw_odom_discontinuity")
        return blockers

    def _on_odom(self, message: Odometry) -> None:
        try:
            if message.header.frame_id.strip() != "camera_init" or message.child_frame_id.strip() != "aft_mapped":
                raise InvalidFastLivo2Frame("raw odom must be camera_init -> aft_mapped")
            source_age = self._source_age(message.header.stamp)
            if not source_age_is_valid(
                source_age,
                max_age_sec=self._source_max_age,
                tolerance_sec=self._source_age_tolerance,
            ):
                raise InvalidFastLivo2Frame(f"raw odom source age {source_age:.3f}s is invalid")
            pose = message.pose.pose
            raw_sensor_pose = Pose3(
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                Quaternion(
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ),
            )
            source_stamp_ns = _stamp_ns(message.header.stamp)
            with self._lock:
                if not self._sensor_contract_ready_locked():
                    raise InvalidFastLivo2Frame(
                        "navigation sensor contract is not ready"
                    )
                if not self._odom_health.observe(source_stamp_ns, raw_sensor_pose):
                    raise InvalidFastLivo2Frame(
                        self._odom_health.detail or "raw odom is unhealthy"
                    )
                base_to_sensor = self._base_to_sensor
            session_pose = canonical_base_pose(
                raw_sensor_pose,
                base_to_sensor,
            )
        except (InvalidFastLivo2Frame, ValueError, TypeError) as exc:
            self._invalid_odom += 1
            self._warn_rejected("FAST-LIVO2 odom", exc)
            return
        self._mark_valid("FAST-LIVO2 odom")

        with self._lock:
            self._latest_session_pose = session_pose
            self._last_odom_monotonic = time.monotonic()
            self._last_odom_source_age = source_age
            map_from_session = self._map_from_session
        if map_from_session is None:
            return
        canonical = compose_pose(map_from_session, session_pose)
        with self._lock:
            if self._map_from_session != map_from_session:
                return
            self._latest_pose = canonical
            self._pose_history.append((source_stamp_ns, canonical))

        output = Odometry()
        output.header.stamp = message.header.stamp
        output.header.frame_id = "map"
        output.child_frame_id = "base_link"
        output.pose.pose.position.x = canonical.x
        output.pose.pose.position.y = canonical.y
        output.pose.pose.position.z = canonical.z
        output.pose.pose.orientation.x = canonical.q.x
        output.pose.pose.orientation.y = canonical.q.y
        output.pose.pose.orientation.z = canonical.q.z
        output.pose.pose.orientation.w = canonical.q.w
        output.pose.covariance = message.pose.covariance
        output.twist = message.twist
        self._odom_pub.publish(output)

        transform = TransformStamped()
        transform.header = output.header
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = canonical.x
        transform.transform.translation.y = canonical.y
        transform.transform.translation.z = canonical.z
        transform.transform.rotation = output.pose.pose.orientation
        self._tf.sendTransform(transform)
        self._drain_pending_cloud()

    def _on_cloud(self, message: PointCloud2) -> None:
        receive_monotonic = time.monotonic()
        try:
            with self._lock:
                if (
                    not self._sensor_contract_ready_locked()
                    or not self._odom_health.ready
                ):
                    raise InvalidFastLivo2Frame(
                        "navigation sensor or odom contract is not ready"
                    )
            if message.header.frame_id.strip() != "camera_init":
                raise InvalidFastLivo2Frame("raw registered cloud must use camera_init")
            source_age = self._source_age(message.header.stamp)
            if not source_age_is_valid(
                source_age,
                max_age_sec=self._source_max_age,
                tolerance_sec=self._source_age_tolerance,
            ):
                raise InvalidFastLivo2Frame(f"raw cloud source age {source_age:.3f}s is invalid")
            width = int(message.width)
            height = int(message.height)
            point_step = int(message.point_step)
            if point_step <= 0 or width < 0 or height < 0:
                raise InvalidFastLivo2Frame("invalid PointCloud2 dimensions")
            if width * height > self._map_load_max_points:
                raise InvalidFastLivo2Frame(
                    "PointCloud2 exceeds "
                    f"{self._map_load_max_points} point safety limit"
                )
            if len(message.data) > self._live_cloud_max_bytes:
                raise InvalidFastLivo2Frame(
                    "PointCloud2 exceeds "
                    f"{self._live_cloud_max_bytes} byte safety limit"
                )
            points = decode_xyz_array(
                fields=message.fields,
                data=bytes(message.data),
                point_step=point_step,
                row_step=int(getattr(message, "row_step", width * point_step)),
                width=width,
                height=height,
                is_bigendian=bool(message.is_bigendian),
                max_points=self._map_load_max_points,
                max_data_bytes=self._live_cloud_max_bytes,
            )
        except (InvalidFastLivo2Frame, ValueError, TypeError) as exc:
            self._invalid_cloud += 1
            self._warn_rejected("FAST-LIVO2 cloud", exc)
            return
        self._mark_valid("FAST-LIVO2 cloud")

        decode_end_monotonic = time.monotonic()
        sample = (
            message,
            points,
            receive_monotonic,
            source_age,
            decode_end_monotonic,
        )
        with self._lock:
            self._latest_session_points = sample[1]
            self._last_cloud_monotonic = receive_monotonic
            self._last_cloud_source_age = source_age
            if self._mode in {"awaiting_relocalization", "relocalized"}:
                stride = max(
                    1,
                    math.ceil(
                        len(points) / _RELOCALIZATION_MAX_POINTS_PER_FRAME
                    ),
                )
                self._relocalization_cloud_history.append(
                    (
                        receive_monotonic,
                        points[
                            ::stride
                        ][:_RELOCALIZATION_MAX_POINTS_PER_FRAME].copy(),
                    )
                )
                cutoff = receive_monotonic - _RELOCALIZATION_HISTORY_SEC
                while (
                    self._relocalization_cloud_history
                    and self._relocalization_cloud_history[0][0] < cutoff
                ):
                    self._relocalization_cloud_history.popleft()
            self._record_latency_locked(
                "cloud_decode",
                decode_end_monotonic - receive_monotonic,
            )
            if self._map_from_session is None:
                return
            if self._pending_cloud is not None:
                self._unmatched_navigation_cloud += 1
                if self._mode == "mapping":
                    self._unmatched_static_cloud += 1
            self._pending_cloud = sample
        self._drain_pending_cloud()

    def _drain_pending_cloud(self) -> None:
        with self._lock:
            sample = self._pending_cloud
            if sample is None or self._map_from_session is None:
                return
            message, points, receive_monotonic, _source_age = sample[:4]
            decode_end_monotonic = (
                sample[4] if len(sample) > 4 else receive_monotonic
            )
            source_stamp_ns = _stamp_ns(message.header.stamp)
            pose_history = tuple(self._pose_history)
            if not pose_history or pose_history[-1][0] < source_stamp_ns:
                return
            self._pending_cloud = None
            map_from_session = self._map_from_session
            mode = self._mode
            obstacle_min_height = self._obstacle_min_height
            obstacle_max_height = self._obstacle_max_height
        matched_pose = bracketed_stamped_pose(
            pose_history,
            source_stamp_ns,
            tolerance_ns=self._static_pose_match_tolerance_ns,
        )
        if matched_pose is None:
            with self._lock:
                self._unmatched_navigation_cloud += 1
                if mode == "mapping":
                    self._unmatched_static_cloud += 1
            return
        pose_skew_sec = min(
            abs(int(stamp_ns) - source_stamp_ns)
            for stamp_ns, _pose in pose_history
        ) / 1_000_000_000.0
        transform_start_monotonic = time.monotonic()
        try:
            mapped_points = transform_xyz_array(map_from_session, points)
            navigation_points = mapped_points[
                (mapped_points[:, 2] >= obstacle_min_height)
                & (mapped_points[:, 2] <= obstacle_max_height)
            ]
            out_of_band_points = mapped_points[
                (mapped_points[:, 2] < obstacle_min_height)
                | (mapped_points[:, 2] > obstacle_max_height)
            ]
            transform_end_monotonic = time.monotonic()
            navigation_cloud = self._xyz_cloud(
                navigation_points,
                message.header.stamp,
            )
        except (InvalidFastLivo2Frame, OverflowError) as exc:
            self._invalid_cloud += 1
            self._warn_rejected("transformed FAST-LIVO2 cloud", exc)
            return
        self._mark_valid("transformed FAST-LIVO2 cloud")
        with self._lock:
            if self._map_from_session != map_from_session:
                return
            self._last_cloud_pose_skew_sec = pose_skew_sec
            self._latest_mapped_points = mapped_points
            generation = self._mapping_generation
        self._cloud_pub.publish(navigation_cloud)
        publish_end_monotonic = time.monotonic()
        with self._lock:
            self._last_navigation_cloud_monotonic = publish_end_monotonic
            self._record_latency_locked(
                "cloud_pose_wait",
                transform_start_monotonic - decode_end_monotonic,
            )
            self._record_latency_locked(
                "cloud_transform_filter",
                transform_end_monotonic - transform_start_monotonic,
            )
            self._record_latency_locked(
                "cloud_pack_publish",
                publish_end_monotonic - transform_end_monotonic,
            )
            self._record_latency_locked(
                "cloud_end_to_end",
                publish_end_monotonic - receive_monotonic,
            )
        if mode == "mapping":
            with self._lock:
                base_to_sensor = self._base_to_sensor
            if base_to_sensor is None:
                return
            sensor_pose = compose_pose(matched_pose, base_to_sensor)
            self._queue_mapping_scan(
                {
                    "generation": generation,
                    "map_from_session": map_from_session,
                    "sensor_origin": (
                        sensor_pose.x,
                        sensor_pose.y,
                        sensor_pose.z,
                    ),
                    "mapped_points": mapped_points,
                    "out_of_band_points": out_of_band_points,
                    "receive_monotonic": receive_monotonic,
                    "obstacle_min_height": obstacle_min_height,
                    "obstacle_max_height": obstacle_max_height,
                }
            )

    def _on_reset(self, message: String) -> None:
        with self._lock:
            self._odom_health.reset()
            self._invalidate_mapping_work_locked()
            with self._static_lock:
                cleared = self._static_map.cleared_snapshot()
                self._static_map.clear()
                retired_map_view = self._map_view_context
                self._map_view_context = VoxelMap(self._map_view_voxel_size)
            self._static_map_load_time = self.get_clock().now().to_msg()
            self._session_name = message.data.strip() or None
            self._mode = "mapping"
            self._map_from_session = Pose3(
                0.0, 0.0, 0.0, quaternion_from_rpy(0.0, 0.0, 0.0)
            )
            self._latest_pose = None
            self._pose_history.clear()
            self._latest_session_pose = None
            self._latest_session_points = ()
            self._relocalization_cloud_history.clear()
            self._relocalization_preview_pose = None
            self._relocalization_preview_points = ()
            self._latest_mapped_points = ()
            self._reference_points = ()
            self._last_match = None
            self._pending_cloud = None
            self._last_cloud_pose_skew_sec = None
            self._last_navigation_cloud_monotonic = None
            self._unmatched_navigation_cloud = 0
            self._unmatched_static_cloud = 0
            self._static_map_error = None
            self._static_save_result = None
            self._invalidate_map_view_cache_locked()
        self._static_map_pub.publish(self._occupancy_grid(cleared))
        _ = retired_map_view

    def _on_map_control(self, message: String) -> None:
        request_id = ""
        action = ""
        post_response = None
        try:
            request = json.loads(message.data)
            if not isinstance(request, dict):
                raise InvalidFastLivo2Frame("map control must be an object")
            request_id = str(request.get("request_id", ""))
            action = str(request.get("action", ""))
            args = request.get("args") or {}
            if not isinstance(args, dict):
                raise InvalidFastLivo2Frame("map control args must be an object")
            operation_deadline = request.get("operation_deadline_monotonic")
            if (
                isinstance(operation_deadline, bool)
                or not isinstance(operation_deadline, (int, float))
                or not math.isfinite(float(operation_deadline))
            ):
                raise InvalidFastLivo2Frame(
                    "map control operation deadline must be finite"
                )
            args = dict(args)
            args["_operation_deadline_monotonic"] = float(operation_deadline)
            self._require_map_control_deadline(args, stage="dispatch")
            if action == "load_map":
                result = self._load_saved_map(args)
            elif action == "validate_map":
                result = self._load_saved_map(args, validate_only=True)
            elif action == "save_static_map":
                result = self._save_static_map(args)
            elif action == "relocalize":
                result = self._relocalize(args)
            elif action == "unload_map":
                result = self._unload_saved_map(args)
            elif action == "configure_obstacle_filter":
                result = self._configure_obstacle_filter(args)
            else:
                result = {
                    "status": "error",
                    "error_code": "unsupported_action",
                    "error": f"unsupported map control action {action}",
                }
            post_response = result.pop("_post_response", None)
        except TimeoutError as exc:
            result = {
                "status": "error",
                "error_code": "map_control_timeout",
                "error": str(exc),
                "retryable": True,
            }
        except OSError as exc:
            result = {
                "status": "error",
                "error_code": "map_control_io_failed",
                "error": str(exc),
                "retryable": True,
            }
        except RelocalizationRejected as exc:
            candidate = exc.result.map_base_pose
            result = {
                "status": "error",
                "error_code": "map_control_failed",
                "error": str(exc),
                "retryable": True,
                "reject_reason": exc.reason,
                "match_ratio": exc.result.match_ratio,
                "matched_points": exc.result.matched_points,
                "evaluated_points": exc.result.evaluated_points,
                "required_match_ratio": _RELOCALIZATION_MIN_MATCH_RATIO,
                "preview_available": True,
                "candidate_pose": {
                    "x": candidate.x,
                    "y": candidate.y,
                    "z": candidate.z,
                    "yaw": yaw_from_quaternion(candidate.q),
                },
            }
        except (InvalidFastLivo2Frame, TypeError, ValueError) as exc:
            result = {
                "status": "error",
                "error_code": "map_control_failed",
                "error": str(exc),
                "retryable": False,
            }
        response = String()
        response.data = json.dumps(
            {"event": "response", "request_id": request_id, "action": action, **result},
            separators=(",", ":"),
        )
        self._map_control_status_pub.publish(response)
        if post_response is not None:
            try:
                post_response()
            except Exception as exc:
                self.get_logger().error(
                    f"map control {action} post-response publication failed: {exc}"
                )

    @staticmethod
    def _require_map_control_deadline(args: dict, *, stage: str) -> None:
        deadline = args.get("_operation_deadline_monotonic")
        if deadline is None or time.monotonic() >= float(deadline):
            raise TimeoutError(
                f"map control operation deadline exceeded before {stage}"
            )

    def _configure_obstacle_filter(self, args: dict) -> dict:
        self._require_map_control_deadline(args, stage="obstacle filter update")
        minimum = float(args.get("min_height_m"))
        maximum = float(args.get("max_height_m"))
        if not all(math.isfinite(value) for value in (minimum, maximum)):
            raise InvalidFastLivo2Frame("obstacle height limits must be finite")
        if not -3.0 <= minimum < maximum <= 3.0:
            raise InvalidFastLivo2Frame(
                "obstacle heights must satisfy -3.0 <= min < max <= 3.0"
            )
        with self._lock:
            if self._mode != "idle":
                raise InvalidFastLivo2Frame(
                    "obstacle height limits can change only while navigation "
                    "mapping/localization is idle"
                )
            self._obstacle_min_height = minimum
            self._obstacle_max_height = maximum
        return {
            "status": "configured",
            "obstacle_height_range_m": [minimum, maximum],
        }

    def _save_static_map(self, args: dict) -> dict:
        self._require_map_control_deadline(args, stage="static map save")
        map_name = str(args.get("map_name", "")).strip()
        if not _MAP_NAME_RE.fullmatch(map_name):
            raise InvalidFastLivo2Frame(
                "map_name must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
            )
        with self._lock:
            if self._session_name != map_name:
                raise InvalidFastLivo2Frame("requested mapping session is not active")
            if self._static_map_error is not None:
                raise InvalidFastLivo2Frame(self._static_map_error)
            if (
                self._static_save_result is not None
                and self._static_save_result.get("map_name") == map_name
            ):
                return dict(self._static_save_result)
            if self._mode not in {"mapping", "finalizing"}:
                raise InvalidFastLivo2Frame("requested mapping session is not active")
            minimum = self._obstacle_min_height
            maximum = self._obstacle_max_height
            with self._static_lock:
                points = self._static_map.confirmed_points
            if len(points) < 40:
                raise InvalidFastLivo2Frame(
                    "confirmed static map has too few points to persist"
                )
            if len(points) > self._static_map_load_max_points:
                raise InvalidFastLivo2Frame(
                    "confirmed static map exceeds "
                    f"{self._static_map_load_max_points} point safety limit"
                )
            self._require_map_control_deadline(
                args,
                stage="static map finalization",
            )
            self._mode = "finalizing"
            self._invalidate_mapping_work_locked()
            self._pose_history.clear()
            self._pending_cloud = None
        static_root = (self._map_root / "static").resolve()
        filename = f"{map_name}-{time.time_ns()}.static.pcd"
        destination = (static_root / filename).resolve()
        if destination.parent != static_root:
            raise InvalidFastLivo2Frame("static map path escaped map root")
        count = write_pcd_xyz_atomic(destination, points)
        try:
            self._require_map_control_deadline(
                args,
                stage="static map receipt",
            )
        except TimeoutError:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        result = {
            "status": "static_map_saved",
            "map_name": map_name,
            "static_map_pcd": str(destination),
            "static_map_file": filename,
            "static_point_count": count,
            "obstacle_height_range_m": [minimum, maximum],
            "temporal_filter": (
                "spatial_shard_motion_gate_with_multi_frame_hits_and_"
                "free_ray_clearing_v3"
            ),
        }
        with self._lock:
            if self._session_name != map_name or self._mode != "finalizing":
                raise InvalidFastLivo2Frame(
                    "mapping session changed while saving the static map"
                )
            self._static_save_result = dict(result)
        return result

    def _load_saved_map(self, args: dict, *, validate_only: bool = False) -> dict:
        self._require_map_control_deadline(args, stage="map load")
        map_name = str(args.get("map_name", "")).strip()
        files = args.get("pcd_files") or []
        if not map_name or not isinstance(files, list) or not files:
            raise InvalidFastLivo2Frame("map_name and pcd_files are required")
        if len(files) > 64:
            raise InvalidFastLivo2Frame("pcd_files must contain at most 64 paths")
        loaded = VoxelMap(float(self.get_parameter("map_voxel_size_m").value))
        remaining_samples = self._map_load_max_points
        for index, path in enumerate(files):
            if not isinstance(path, str):
                raise InvalidFastLivo2Frame("pcd_files must contain paths")
            resolved = Path(path).resolve()
            if resolved.parent != self._map_root or not resolved.is_file():
                raise InvalidFastLivo2Frame("PCD path must be an existing map-root file")
            remaining_files = len(files) - index
            per_file_limit = max(1, remaining_samples // remaining_files)
            sampled = read_pcd_xyz(
                resolved,
                max_points=per_file_limit,
                deadline_monotonic=args["_operation_deadline_monotonic"],
            )
            remaining_samples -= len(sampled)
            loaded.add(sampled)
            self._require_map_control_deadline(
                args,
                stage="raw map validation",
            )
        if loaded.point_count < 40:
            raise InvalidFastLivo2Frame("saved map has too few finite points")
        static_file = args.get("static_map_pcd")
        static_source = "legacy_raw"
        static_loaded = loaded
        with self._lock:
            active_height_range = (
                self._obstacle_min_height,
                self._obstacle_max_height,
            )
        if static_file is not None:
            if not isinstance(static_file, str):
                raise InvalidFastLivo2Frame("static_map_pcd must be a path")
            resolved_static = Path(static_file).resolve()
            static_root = (self._map_root / "static").resolve()
            if resolved_static.parent != static_root or not resolved_static.is_file():
                raise InvalidFastLivo2Frame(
                    "static PCD path must be an existing map-root/static file"
                )
            try:
                saved_height_range = normalize_obstacle_height_range(
                    args.get("obstacle_height_range_m"),
                    field_name="saved static map obstacle_height_range_m",
                )
            except ValueError as exc:
                raise InvalidFastLivo2Frame(str(exc)) from exc
            if not obstacle_height_ranges_match(
                saved_height_range,
                active_height_range,
            ):
                raise InvalidFastLivo2Frame(
                    "saved static map obstacle height range does not match "
                    "the active card configuration; restore the saved range "
                    "or rebuild the map"
                )
            static_loaded = VoxelMap(
                float(self.get_parameter("map_voxel_size_m").value)
            )
            static_loaded.add(
                read_pcd_xyz(
                    resolved_static,
                    max_declared_points=self._static_map_load_max_points,
                    deadline_monotonic=args["_operation_deadline_monotonic"],
                )
            )
            self._require_map_control_deadline(
                args,
                stage="static map validation",
            )
            if static_loaded.point_count < 40:
                raise InvalidFastLivo2Frame(
                    "confirmed static map has too few finite points"
                )
            static_source = "confirmed_static_pcd"
        map_view_context = VoxelMap(self._map_view_voxel_size)
        map_view_context.add(
            point
            for point in loaded.points
            if not active_height_range[0] <= point[2] <= active_height_range[1]
        )
        with self._static_lock:
            prepared = self._static_map.prepare_confirmed(static_loaded.points)
            snapshot = self._static_map.prepared_occupancy_snapshot(
                prepared,
                center_x=static_loaded.points[0][0],
                center_y=static_loaded.points[0][1],
                min_z=active_height_range[0],
                max_z=active_height_range[1],
            )
        self._require_map_control_deadline(args, stage="map activation")
        if validate_only:
            return {
                "status": "map_validated",
                "map_name": map_name,
                "map_point_count": loaded.point_count,
                "static_map_point_count": static_loaded.point_count,
                "static_map_source": static_source,
                "map_view_context_point_count": map_view_context.point_count,
                "obstacle_height_range_m": list(active_height_range),
            }
        with self._static_lock:
            cleared = self._static_map.cleared_snapshot()
        self._require_map_control_deadline(args, stage="map activation commit")
        with self._lock:
            if (
                self._obstacle_min_height,
                self._obstacle_max_height,
            ) != active_height_range:
                raise InvalidFastLivo2Frame(
                    "obstacle height range changed while loading the map"
                )
            self._require_map_control_deadline(
                args,
                stage="map activation commit",
            )
            self._invalidate_mapping_work_locked()
            with self._static_lock:
                retired_static = self._static_map.apply_prepared_confirmed(prepared)
                retired_adapter = (
                    self._reference_points,
                    self._pose_history,
                    self._latest_session_points,
                    self._relocalization_cloud_history,
                    self._relocalization_preview_points,
                    self._latest_mapped_points,
                    self._map_view_context,
                )
            self._static_map_load_time = self.get_clock().now().to_msg()
            self._reference_points = static_loaded.points
            self._odom_health.reset(require_near_origin=False)
            self._session_name = map_name
            self._mode = "awaiting_relocalization"
            self._map_from_session = None
            self._latest_pose = None
            self._pose_history = deque(maxlen=128)
            self._latest_session_pose = None
            self._latest_session_points = ()
            self._relocalization_cloud_history = deque(
                maxlen=_RELOCALIZATION_MAX_FRAMES
            )
            self._relocalization_preview_pose = None
            self._relocalization_preview_points = ()
            self._latest_mapped_points = ()
            self._map_view_context = map_view_context
            self._last_odom_monotonic = None
            self._last_cloud_monotonic = None
            self._last_navigation_cloud_monotonic = None
            self._pending_cloud = None
            self._last_cloud_pose_skew_sec = None
            self._last_match = None
            self._static_map_error = None
            self._static_save_result = None
            self._invalidate_map_view_cache_locked()
        def publish_loaded_map() -> None:
            self._static_map_pub.publish(self._occupancy_grid(cleared))
            self._static_map_pub.publish(self._occupancy_grid(snapshot))
            # Keep detached large containers alive until after the response and
            # latched-map publication, outside the deadline-protected commit.
            _ = retired_static, retired_adapter

        return {
            "status": "map_loaded",
            "map_name": map_name,
            "map_point_count": loaded.point_count,
            "static_map_point_count": static_loaded.point_count,
            "static_map_source": static_source,
            "map_view_context_point_count": map_view_context.point_count,
            "obstacle_height_range_m": list(active_height_range),
            "localization_state": "awaiting_relocalization",
            "_post_response": publish_loaded_map,
        }

    def _relocalize(self, args: dict) -> dict:
        self._require_map_control_deadline(args, stage="relocalization")
        with self._lock:
            if self._mode not in {"awaiting_relocalization", "relocalized"}:
                raise InvalidFastLivo2Frame("no saved map is loaded")
            session_pose = self._latest_session_pose
            now = time.monotonic()
            while (
                self._relocalization_cloud_history
                and self._relocalization_cloud_history[0][0]
                < now - _RELOCALIZATION_HISTORY_SEC
            ):
                self._relocalization_cloud_history.popleft()
            session_frames = tuple(
                points for _stamp, points in self._relocalization_cloud_history
            )
            odom_age = None if self._last_odom_monotonic is None else now - self._last_odom_monotonic
            cloud_age = None if self._last_cloud_monotonic is None else now - self._last_cloud_monotonic
            reference = self._reference_points
            map_name = self._session_name
        if session_pose is None or not session_frames:
            raise InvalidFastLivo2Frame("FAST-LIVO2 odom and registered cloud are not ready")
        if odom_age is None or cloud_age is None or max(odom_age, cloud_age) > self._source_max_age:
            raise InvalidFastLivo2Frame("FAST-LIVO2 odom or registered cloud is stale")
        session_points = np.concatenate(
            tuple(
                np.asarray(points, dtype=np.float64).reshape((-1, 3))
                for points in session_frames
            ),
            axis=0,
        )
        if len(session_points) == 0:
            raise InvalidFastLivo2Frame("FAST-LIVO2 registered cloud is empty")
        initial = Pose3(
            float(args["initial_x"]),
            float(args["initial_y"]),
            float(args.get("initial_z", 0.0)),
            quaternion_from_rpy(0.0, 0.0, float(args["initial_yaw"])),
        )
        try:
            result = estimate_planar_relocalization(
                reference_points=reference,
                session_points=session_points,
                session_base_pose=session_pose,
                initial_map_base_pose=initial,
                search_xy_m=float(args.get("search_xy_m", 1.0)),
                search_yaw_rad=float(args.get("search_yaw_rad", 0.35)),
                min_z=self._obstacle_min_height,
                max_z=self._obstacle_max_height,
                min_match_ratio=_RELOCALIZATION_MIN_MATCH_RATIO,
            )
        except RelocalizationRejected as exc:
            self._require_map_control_deadline(
                args,
                stage="relocalization preview",
            )
            preview_points = transform_xyz_array(
                exc.result.map_from_session,
                session_points,
            )
            with self._lock:
                self._relocalization_preview_pose = exc.result.map_base_pose
                self._relocalization_preview_points = preview_points
                self._last_match = {
                    "accepted": False,
                    "reject_reason": exc.reason,
                    "match_ratio": exc.result.match_ratio,
                    "matched_points": exc.result.matched_points,
                    "evaluated_points": exc.result.evaluated_points,
                    "required_match_ratio": _RELOCALIZATION_MIN_MATCH_RATIO,
                    "input_frame_count": len(session_frames),
                    "input_point_count": len(session_points),
                }
                self._invalidate_map_view_cache_locked()
            raise
        match = {
            "accepted": True,
            "match_ratio": result.match_ratio,
            "matched_points": result.matched_points,
            "evaluated_points": result.evaluated_points,
            "required_match_ratio": _RELOCALIZATION_MIN_MATCH_RATIO,
            "input_frame_count": len(session_frames),
            "input_point_count": len(session_points),
        }
        self._require_map_control_deadline(
            args,
            stage="relocalization update",
        )
        with self._lock:
            self._map_from_session = result.map_from_session
            self._odom_health.reset(require_near_origin=False)
            self._latest_pose = result.map_base_pose
            self._relocalization_preview_pose = None
            self._relocalization_preview_points = ()
            self._latest_mapped_points = ()
            self._pending_cloud = None
            self._last_cloud_pose_skew_sec = None
            self._last_navigation_cloud_monotonic = None
            self._invalidate_map_view_cache_locked()
            self._mode = "relocalized"
            self._last_match = match
        return {
            "status": "relocalized",
            "map_name": map_name,
            "pose": {
                "x": result.map_base_pose.x,
                "y": result.map_base_pose.y,
                "z": result.map_base_pose.z,
                "yaw": yaw_from_quaternion(result.map_base_pose.q),
            },
            **match,
            "continuous_global_correction": False,
        }

    def _unload_saved_map(self, args: dict) -> dict:
        self._require_map_control_deadline(args, stage="map unload")
        with self._lock:
            map_name = self._session_name
        with self._static_lock:
            cleared = self._static_map.cleared_snapshot()
        self._require_map_control_deadline(args, stage="map unload commit")
        with self._lock:
            if self._session_name != map_name:
                raise InvalidFastLivo2Frame(
                    "loaded map changed while unloading"
                )
            self._require_map_control_deadline(args, stage="map unload commit")
            self._invalidate_mapping_work_locked()
            with self._static_lock:
                retired_static = self._static_map.retire_and_clear()
                retired_adapter = (
                    self._reference_points,
                    self._pose_history,
                    self._latest_session_points,
                    self._relocalization_cloud_history,
                    self._relocalization_preview_points,
                    self._latest_mapped_points,
                    self._map_view_context,
                )
            self._static_map_load_time = self.get_clock().now().to_msg()
            self._session_name = None
            self._odom_health.reset(require_near_origin=False)
            self._mode = "idle"
            self._map_from_session = None
            self._latest_pose = None
            self._pose_history = deque(maxlen=128)
            self._latest_session_pose = None
            self._latest_session_points = ()
            self._relocalization_cloud_history = deque(
                maxlen=_RELOCALIZATION_MAX_FRAMES
            )
            self._relocalization_preview_pose = None
            self._relocalization_preview_points = ()
            self._latest_mapped_points = ()
            self._map_view_context = VoxelMap(self._map_view_voxel_size)
            self._reference_points = ()
            self._last_odom_monotonic = None
            self._last_cloud_monotonic = None
            self._last_navigation_cloud_monotonic = None
            self._pending_cloud = None
            self._last_cloud_pose_skew_sec = None
            self._last_match = None
            self._static_map_error = None
            self._static_save_result = None
            self._invalidate_map_view_cache_locked()
        def publish_cleared_map() -> None:
            self._static_map_pub.publish(self._occupancy_grid(cleared))
            _ = retired_static, retired_adapter

        return {
            "status": "unloaded",
            "map_name": map_name,
            "_post_response": publish_cleared_map,
        }

    def _obstacle_cloud(self, points) -> PointCloud2:
        output = PointCloud2()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "map"
        output.height = 1
        output.width = len(points)
        output.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        output.is_bigendian = False
        output.point_step = 12
        output.row_step = output.point_step * output.width
        output.data = xyz_array_bytes(points)
        output.is_dense = True
        return output

    def _xyz_cloud(self, points, stamp) -> PointCloud2:
        output = self._obstacle_cloud(points)
        output.header.stamp = stamp
        return output

    def _publish_map_view(self) -> None:
        started = time.monotonic()
        with self._lock:
            cached = self._map_view_cache
            pose = self._latest_pose or self._relocalization_preview_pose
        if cached is None or pose is None:
            return
        try:
            data = map_view_with_pose(cached, pose)
        except InvalidFastLivo2Frame as exc:
            self._warn_rejected("cached map view", exc)
            return
        self._mark_valid("cached map view")
        frame = UInt8MultiArray()
        frame.data = data
        self._map_view_pub.publish(frame)
        with self._lock:
            self._record_latency_locked(
                "map_view_pose_publish",
                time.monotonic() - started,
            )

    def _occupancy_grid(self, snapshot) -> OccupancyGrid:
        output = OccupancyGrid()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "map"
        output.info.map_load_time = self._static_map_load_time
        output.info.resolution = snapshot.resolution
        output.info.width = snapshot.width
        output.info.height = snapshot.height
        output.info.origin.position.x = snapshot.origin_x
        output.info.origin.position.y = snapshot.origin_y
        output.info.origin.orientation.w = 1.0
        output.data = list(snapshot.data)
        return output

    def _publish_periodic(self) -> None:
        now = time.monotonic()
        static_grid = None
        encoded_map_view = None
        map_view_encode_sec = None
        with self._lock:
            confirmed_pose = self._latest_pose
            pose = confirmed_pose or self._relocalization_preview_pose
            odom_age = None if self._last_odom_monotonic is None else now - self._last_odom_monotonic
            cloud_age = None if self._last_cloud_monotonic is None else now - self._last_cloud_monotonic
            navigation_cloud_age = (
                None
                if self._last_navigation_cloud_monotonic is None
                else now - self._last_navigation_cloud_monotonic
            )
            obstacle_min_height = self._obstacle_min_height
            obstacle_max_height = self._obstacle_max_height
            live_points = ()
            if cloud_age is not None and cloud_age <= self._source_max_age:
                live_points = self._latest_mapped_points
            if confirmed_pose is None and self._relocalization_preview_pose is not None:
                live_points = self._relocalization_preview_points
            map_view_context = self._map_view_context
            state = {
                "session_name": self._session_name,
                "mode": self._mode,
                "map_alignment_confirmed": self._map_from_session is not None,
                "last_match": self._last_match,
                "odom_source_age": self._last_odom_source_age,
                "cloud_source_age": self._last_cloud_source_age,
                "reference_map_point_count": len(self._reference_points),
                "relocalization_history_frame_count": len(
                    self._relocalization_cloud_history
                ),
                "relocalization_history_point_count": sum(
                    len(points)
                    for _stamp, points in self._relocalization_cloud_history
                ),
                "relocalization_preview_available": (
                    self._relocalization_preview_pose is not None
                ),
                "cloud_pose_skew_sec": self._last_cloud_pose_skew_sec,
                "pending_navigation_cloud": self._pending_cloud is not None,
                "unmatched_navigation_cloud": self._unmatched_navigation_cloud,
                "unmatched_static_cloud": self._unmatched_static_cloud,
                "static_map_error": self._static_map_error,
                "invalid_odom": self._invalid_odom,
                "invalid_cloud": self._invalid_cloud,
                "latency_ms": dict(self._latency_ms),
                "latency_max_ms": dict(self._latency_max_ms),
                "map_view_cache_monotonic": self._map_view_cache_monotonic,
                "sensor_frame": self._sensor_frame,
                "sensor_contract_ready": self._sensor_contract_ready_locked(),
                "base_to_sensor_tf_ready": self._base_to_sensor_tf_ready,
                "sensor_tf_error": self._sensor_tf_error,
                "point_time_span_ms": self._point_time_span_ms,
                "odom_health": self._odom_health.diagnostics(),
                "readiness_blockers": self._readiness_blockers_locked(),
            }

        with self._static_lock:
            obstacle_points = self._static_map.project_xy(
                min_z=obstacle_min_height,
                max_z=obstacle_max_height,
            )
            live_out_of_band = tuple(
                point
                for point in live_points
                if not obstacle_min_height <= point[2] <= obstacle_max_height
            )
            if confirmed_pose is not None:
                snapshot = self._static_map.occupancy_snapshot(
                    center_x=confirmed_pose.x,
                    center_y=confirmed_pose.y,
                    min_z=obstacle_min_height,
                    max_z=obstacle_max_height,
                )
                static_grid = self._occupancy_grid(snapshot)
            if pose is not None:
                map_view_encode_started = time.monotonic()
                encoded_map_view = encode_map_view_points(
                    chain(
                        self._static_map.map_view_points,
                        map_view_context.points,
                        live_points,
                    ),
                    pose,
                    obstacle_min_height_m=obstacle_min_height,
                    obstacle_max_height_m=obstacle_max_height,
                    max_points=_MAP_VIEW_MAX_POINTS,
                )
                map_view_encode_sec = time.monotonic() - map_view_encode_started
            static_diagnostics = {
                "map_point_count": self._static_map.point_count,
                "static_free_cell_count": self._static_map.free_cell_count,
                "map_view_context_point_count": map_view_context.point_count,
            }
        if encoded_map_view is not None:
            with self._lock:
                self._map_view_cache = encoded_map_view
                self._map_view_cache_monotonic = time.monotonic()
                self._record_latency_locked(
                    "map_view_encode",
                    map_view_encode_sec,
                )
                state["map_view_cache_monotonic"] = self._map_view_cache_monotonic
                state["latency_ms"] = dict(self._latency_ms)
                state["latency_max_ms"] = dict(self._latency_max_ms)
        with self._mapping_work_condition:
            mapping_work_dropped = self._mapping_work_dropped
        ready = (
            state["mode"] in {"mapping", "relocalized"}
            and state["map_alignment_confirmed"]
            and odom_age is not None
            and cloud_age is not None
            and navigation_cloud_age is not None
            and odom_age <= self._source_max_age
            and cloud_age <= self._source_max_age
            and navigation_cloud_age <= self._source_max_age
            and not state["readiness_blockers"]
        )
        payload = {
            "schema": "phanthy.navigation.fast_livo2_diagnostics.v1",
            "ready": ready,
            "session_name": state["session_name"],
            "localization_state": state["mode"],
            "map_alignment_confirmed": state["map_alignment_confirmed"],
            "last_match": state["last_match"],
            "odom_receive_age_sec": odom_age,
            "cloud_receive_age_sec": cloud_age,
            "navigation_cloud_receive_age_sec": navigation_cloud_age,
            "odom_source_age_sec": state["odom_source_age"],
            "cloud_source_age_sec": state["cloud_source_age"],
            "reference_map_point_count": state["reference_map_point_count"],
            "relocalization_history_sec": _RELOCALIZATION_HISTORY_SEC,
            "relocalization_history_frame_count": state[
                "relocalization_history_frame_count"
            ],
            "relocalization_history_point_count": state[
                "relocalization_history_point_count"
            ],
            "relocalization_preview_available": state[
                "relocalization_preview_available"
            ],
            **static_diagnostics,
            "static_pose_match_tolerance_sec": self._static_pose_match_tolerance,
            "cloud_pose_skew_sec": state["cloud_pose_skew_sec"],
            "pending_navigation_cloud": state["pending_navigation_cloud"],
            "unmatched_navigation_cloud": state["unmatched_navigation_cloud"],
            "unmatched_static_cloud": state["unmatched_static_cloud"],
            "mapping_work_dropped": mapping_work_dropped,
            "static_map_error": state["static_map_error"],
            "map_view_max_point_count": _MAP_VIEW_MAX_POINTS,
            "map_view_point_refresh_hz": 1.0,
            "map_view_pose_refresh_hz": _MAP_VIEW_POSE_REFRESH_HZ,
            "map_view_cache_age_sec": (
                None
                if state["map_view_cache_monotonic"] is None
                else max(0.0, now - state["map_view_cache_monotonic"])
            ),
            "latency_ms": state["latency_ms"],
            "latency_max_ms": state["latency_max_ms"],
            "map_view_live_out_of_band_point_count": len(live_out_of_band),
            "obstacle_point_count": len(obstacle_points),
            "obstacle_height_range_m": [obstacle_min_height, obstacle_max_height],
            "invalid_odom": state["invalid_odom"],
            "invalid_cloud": state["invalid_cloud"],
            "raw_odom_frame": "camera_init -> aft_mapped",
            "canonical_odom_frame": "map -> base_link",
            "canonical_cloud_frame": "map",
            "obstacle_map_frame": "map",
            "static_map_frame": "map",
            "sensor_frame": state["sensor_frame"],
            "sensor_contract_ready": state["sensor_contract_ready"],
            "base_to_sensor_tf_ready": state["base_to_sensor_tf_ready"],
            "sensor_tf_error": state["sensor_tf_error"],
            "point_time_span_ms": state["point_time_span_ms"],
            "odom_health": state["odom_health"],
            "readiness_blockers": state["readiness_blockers"],
        }
        if static_grid is not None:
            self._static_map_pub.publish(static_grid)
        self._obstacle_map_pub.publish(self._obstacle_cloud(obstacle_points))
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._diagnostics_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FastLivo2Adapter()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
