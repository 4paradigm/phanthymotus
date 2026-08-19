"""Normalize FAST-LIVO2 raw outputs into the canonical Nav2 contract."""

from __future__ import annotations

from collections import deque
from itertools import chain
import json
import math
from pathlib import Path
import re
import struct
import threading
import time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray
from tf2_ros import TransformBroadcaster

from .frame_adapter_core import (
    InvalidFastLivo2Frame,
    Pose3,
    Quaternion,
    TemporalOccupancyMap,
    VoxelMap,
    bracketed_stamped_pose,
    canonical_base_pose,
    compose_pose,
    estimate_planar_relocalization,
    encode_map_view_points,
    iter_xyz_points,
    normalize_obstacle_height_range,
    obstacle_height_ranges_match,
    quaternion_from_rpy,
    read_pcd_xyz,
    source_age_is_valid,
    transform_points,
    write_pcd_xyz_atomic,
    yaw_from_quaternion,
)


_MAP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MAP_VIEW_MAX_POINTS = 80_000


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class FastLivo2Adapter(Node):
    def __init__(self) -> None:
        super().__init__("g1_fast_livo2_adapter")
        self.declare_parameter("raw_odom_topic", "/ubuntu/navigation/fast_livo2/raw/odom")
        self.declare_parameter("raw_cloud_topic", "/ubuntu/navigation/fast_livo2/raw/cloud_registered")
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
        self.declare_parameter("static_confirmation_frames", 8)
        self.declare_parameter("static_candidate_ttl_sec", 1.0)
        self.declare_parameter("static_clear_miss_frames", 3)
        self.declare_parameter("static_angular_bin_deg", 1.0)
        self.declare_parameter("static_grid_margin_m", 6.0)
        self.declare_parameter("static_pose_match_tolerance_sec", 0.05)
        self.declare_parameter("static_component_motion_window_sec", 0.40)
        self.declare_parameter("static_component_history_sec", 0.80)
        self.declare_parameter("static_component_motion_distance_m", 0.03)
        self.declare_parameter("static_component_motion_speed_mps", 0.03)
        self.declare_parameter("static_component_stationary_sec", 1.50)
        self.declare_parameter("static_component_max_span_m", 1.00)
        self.declare_parameter("static_component_match_distance_m", 0.60)
        self.declare_parameter("static_dynamic_filter_enabled", False)
        self.declare_parameter("obstacle_min_height_m", -0.30)
        self.declare_parameter("obstacle_max_height_m", 0.30)
        self.declare_parameter("base_to_sensor_x", -0.00368)
        self.declare_parameter("base_to_sensor_y", 0.00003)
        self.declare_parameter("base_to_sensor_z", 0.46018)
        self.declare_parameter("base_to_sensor_roll", 0.0)
        self.declare_parameter("base_to_sensor_pitch", 0.04014257279586953)
        self.declare_parameter("base_to_sensor_yaw", 0.0)

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
        self._base_to_sensor = Pose3(
            float(self.get_parameter("base_to_sensor_x").value),
            float(self.get_parameter("base_to_sensor_y").value),
            float(self.get_parameter("base_to_sensor_z").value),
            quaternion_from_rpy(
                float(self.get_parameter("base_to_sensor_roll").value),
                float(self.get_parameter("base_to_sensor_pitch").value),
                float(self.get_parameter("base_to_sensor_yaw").value),
            ),
        )
        map_voxel_size = float(self.get_parameter("map_voxel_size_m").value)
        self._map_view_voxel_size = max(map_voxel_size, 0.20)
        self._map_view_context = VoxelMap(self._map_view_voxel_size)
        self._static_map = TemporalOccupancyMap(
            map_voxel_size,
            confirmation_frames=int(
                self.get_parameter("static_confirmation_frames").value
            ),
            candidate_ttl_sec=float(
                self.get_parameter("static_candidate_ttl_sec").value
            ),
            clear_miss_frames=int(
                self.get_parameter("static_clear_miss_frames").value
            ),
            angular_bin_deg=float(
                self.get_parameter("static_angular_bin_deg").value
            ),
            grid_margin_m=float(
                self.get_parameter("static_grid_margin_m").value
            ),
            component_motion_window_sec=float(
                self.get_parameter("static_component_motion_window_sec").value
            ),
            component_history_sec=float(
                self.get_parameter("static_component_history_sec").value
            ),
            component_motion_distance_m=float(
                self.get_parameter("static_component_motion_distance_m").value
            ),
            component_motion_speed_mps=float(
                self.get_parameter("static_component_motion_speed_mps").value
            ),
            component_stationary_sec=float(
                self.get_parameter("static_component_stationary_sec").value
            ),
            component_max_span_m=float(
                self.get_parameter("static_component_max_span_m").value
            ),
            component_match_distance_m=float(
                self.get_parameter("static_component_match_distance_m").value
            ),
            max_evidence_points=self._static_map_load_max_points,
            dynamic_filter_enabled=bool(
                self.get_parameter("static_dynamic_filter_enabled").value
            ),
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
        self._latest_pose: Pose3 | None = None
        self._pose_history = deque(maxlen=128)
        self._latest_session_pose: Pose3 | None = None
        self._latest_session_points: tuple[tuple[float, float, float], ...] = ()
        self._latest_mapped_points: tuple[tuple[float, float, float], ...] = ()
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
        self._static_map_load_time = self.get_clock().now().to_msg()

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
        self._map_view_pub = self.create_publisher(UInt8MultiArray, str(self.get_parameter("map_view_topic").value), qos_profile_sensor_data)
        self._diagnostics_pub = self.create_publisher(String, str(self.get_parameter("diagnostics_topic").value), 10)
        self._map_control_status_pub = self.create_publisher(
            String, str(self.get_parameter("map_control_status_topic").value), 10
        )
        self._tf = TransformBroadcaster(self)
        latest_sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("raw_odom_topic").value),
            self._on_odom,
            latest_sensor_qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("raw_cloud_topic").value),
            self._on_cloud,
            latest_sensor_qos,
        )
        self.create_subscription(String, str(self.get_parameter("reset_topic").value), self._on_reset, 10)
        self.create_subscription(
            String,
            str(self.get_parameter("map_control_topic").value),
            self._on_map_control,
            10,
        )
        self.create_timer(1.0, self._publish_periodic)

    def _source_age(self, stamp) -> float:
        source = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9 - source

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
            session_pose = canonical_base_pose(
                Pose3(
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    Quaternion(
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    ),
                ),
                self._base_to_sensor,
            )
        except (InvalidFastLivo2Frame, ValueError, TypeError) as exc:
            self._invalid_odom += 1
            self.get_logger().warning(f"rejecting FAST-LIVO2 odom: {exc}")
            return

        with self._lock:
            self._latest_session_pose = session_pose
            self._last_odom_monotonic = time.monotonic()
            self._last_odom_source_age = source_age
            map_from_session = self._map_from_session
        if map_from_session is None:
            return
        canonical = compose_pose(map_from_session, session_pose)
        source_stamp_ns = _stamp_ns(message.header.stamp)
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
            points = list(
                iter_xyz_points(
                    fields=message.fields,
                    data=bytes(message.data),
                    point_step=point_step,
                    width=width,
                    height=height,
                    is_bigendian=bool(message.is_bigendian),
                    max_points=self._map_load_max_points,
                    max_data_bytes=self._live_cloud_max_bytes,
                )
            )
        except (InvalidFastLivo2Frame, ValueError, TypeError) as exc:
            self._invalid_cloud += 1
            self.get_logger().warning(f"rejecting FAST-LIVO2 cloud: {exc}")
            return

        sample = (message, tuple(points), receive_monotonic, source_age)
        with self._lock:
            self._latest_session_points = sample[1]
            self._last_cloud_monotonic = receive_monotonic
            self._last_cloud_source_age = source_age
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
            message, points, receive_monotonic, _source_age = sample
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
        try:
            mapped_points = tuple(transform_points(map_from_session, points))
            navigation_points = tuple(
                point
                for point in mapped_points
                if obstacle_min_height <= point[2] <= obstacle_max_height
            )
            navigation_cloud = self._xyz_cloud(
                navigation_points,
                message.header.stamp,
            )
        except (InvalidFastLivo2Frame, OverflowError, struct.error) as exc:
            self._invalid_cloud += 1
            self.get_logger().warning(
                f"rejecting transformed FAST-LIVO2 cloud: {exc}"
            )
            return
        with self._lock:
            if self._map_from_session != map_from_session:
                return
            self._last_cloud_pose_skew_sec = pose_skew_sec
            self._latest_mapped_points = mapped_points
            if self._mode == "mapping":
                self._map_view_context.add(
                    point
                    for point in mapped_points
                    if not obstacle_min_height <= point[2] <= obstacle_max_height
                )
        self._cloud_pub.publish(navigation_cloud)
        with self._lock:
            self._last_navigation_cloud_monotonic = time.monotonic()
        if mode == "mapping":
            with self._lock:
                if self._mode != "mapping" or self._map_from_session != map_from_session:
                    return
                try:
                    sensor_pose = compose_pose(matched_pose, self._base_to_sensor)
                    self._static_map.observe_scan(
                        sensor_origin=(sensor_pose.x, sensor_pose.y, sensor_pose.z),
                        points=mapped_points,
                        now_monotonic=receive_monotonic,
                        obstacle_min_height_m=obstacle_min_height,
                        obstacle_max_height_m=obstacle_max_height,
                    )
                except ValueError as exc:
                    self._static_map_error = str(exc)
                    self._mode = "mapping_error"
                    self.get_logger().error(
                        f"stopping static-map accumulation: {exc}"
                    )

    def _on_reset(self, message: String) -> None:
        with self._lock:
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
            points = self._static_map.confirmed_points
            minimum = self._obstacle_min_height
            maximum = self._obstacle_max_height
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
        with self._lock:
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
            retired_static = self._static_map.apply_prepared_confirmed(prepared)
            retired_adapter = (
                self._reference_points,
                self._pose_history,
                self._latest_session_points,
                self._latest_mapped_points,
                self._map_view_context,
            )
            self._static_map_load_time = self.get_clock().now().to_msg()
            self._reference_points = loaded.points
            self._session_name = map_name
            self._mode = "awaiting_relocalization"
            self._map_from_session = None
            self._latest_pose = None
            self._pose_history = deque(maxlen=128)
            self._latest_session_pose = None
            self._latest_session_points = ()
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
            session_points = self._latest_session_points
            odom_age = None if self._last_odom_monotonic is None else time.monotonic() - self._last_odom_monotonic
            cloud_age = None if self._last_cloud_monotonic is None else time.monotonic() - self._last_cloud_monotonic
            reference = self._reference_points
            map_name = self._session_name
        if session_pose is None or not session_points:
            raise InvalidFastLivo2Frame("FAST-LIVO2 odom and registered cloud are not ready")
        if odom_age is None or cloud_age is None or max(odom_age, cloud_age) > self._source_max_age:
            raise InvalidFastLivo2Frame("FAST-LIVO2 odom or registered cloud is stale")
        initial = Pose3(
            float(args["initial_x"]),
            float(args["initial_y"]),
            float(args.get("initial_z", 0.0)),
            quaternion_from_rpy(0.0, 0.0, float(args["initial_yaw"])),
        )
        result = estimate_planar_relocalization(
            reference_points=reference,
            session_points=session_points,
            session_base_pose=session_pose,
            initial_map_base_pose=initial,
            search_xy_m=float(args.get("search_xy_m", 1.0)),
            search_yaw_rad=float(args.get("search_yaw_rad", 0.35)),
            min_z=self._obstacle_min_height,
            max_z=self._obstacle_max_height,
        )
        match = {
            "match_ratio": result.match_ratio,
            "matched_points": result.matched_points,
            "evaluated_points": result.evaluated_points,
        }
        self._require_map_control_deadline(
            args,
            stage="relocalization update",
        )
        with self._lock:
            self._map_from_session = result.map_from_session
            self._latest_pose = result.map_base_pose
            self._pending_cloud = None
            self._last_cloud_pose_skew_sec = None
            self._last_navigation_cloud_monotonic = None
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
            cleared = self._static_map.cleared_snapshot()
        self._require_map_control_deadline(args, stage="map unload commit")
        with self._lock:
            if self._session_name != map_name:
                raise InvalidFastLivo2Frame(
                    "loaded map changed while unloading"
                )
            self._require_map_control_deadline(args, stage="map unload commit")
            retired_static = self._static_map.retire_and_clear()
            retired_adapter = (
                self._reference_points,
                self._pose_history,
                self._latest_session_points,
                self._latest_mapped_points,
                self._map_view_context,
            )
            self._static_map_load_time = self.get_clock().now().to_msg()
            self._session_name = None
            self._mode = "idle"
            self._map_from_session = None
            self._latest_pose = None
            self._pose_history = deque(maxlen=128)
            self._latest_session_pose = None
            self._latest_session_points = ()
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
        def publish_cleared_map() -> None:
            self._static_map_pub.publish(self._occupancy_grid(cleared))
            _ = retired_static, retired_adapter

        return {
            "status": "unloaded",
            "map_name": map_name,
            "_post_response": publish_cleared_map,
        }

    def _obstacle_cloud(self, points: tuple[tuple[float, float, float], ...]) -> PointCloud2:
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
        output.data = b"".join(struct.pack("<fff", *point) for point in points)
        output.is_dense = True
        return output

    def _xyz_cloud(self, points, stamp) -> PointCloud2:
        output = self._obstacle_cloud(tuple(points))
        output.header.stamp = stamp
        return output

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
        with self._lock:
            pose = self._latest_pose
            odom_age = None if self._last_odom_monotonic is None else now - self._last_odom_monotonic
            cloud_age = None if self._last_cloud_monotonic is None else now - self._last_cloud_monotonic
            navigation_cloud_age = (
                None
                if self._last_navigation_cloud_monotonic is None
                else now - self._last_navigation_cloud_monotonic
            )
            self._static_map.expire(now_monotonic=now)
            obstacle_points = self._static_map.project_xy(
                min_z=self._obstacle_min_height,
                max_z=self._obstacle_max_height,
            )
            candidate_points = self._static_map.candidate_points
            live_points = ()
            live_out_of_band = ()
            if cloud_age is not None and cloud_age <= self._source_max_age:
                live_points = self._latest_mapped_points
                live_out_of_band = tuple(
                    point
                    for point in live_points
                    if not self._obstacle_min_height
                    <= point[2]
                    <= self._obstacle_max_height
                )
            if pose is not None:
                snapshot = self._static_map.occupancy_snapshot(
                    center_x=pose.x,
                    center_y=pose.y,
                    min_z=self._obstacle_min_height,
                    max_z=self._obstacle_max_height,
                )
                static_grid = self._occupancy_grid(snapshot)
            if pose is not None:
                frame = UInt8MultiArray()
                frame.data = encode_map_view_points(
                    chain(
                        self._static_map.map_view_points,
                        self._map_view_context.points,
                        candidate_points,
                        live_points,
                    ),
                    pose,
                    obstacle_min_height_m=self._obstacle_min_height,
                    obstacle_max_height_m=self._obstacle_max_height,
                    max_points=_MAP_VIEW_MAX_POINTS,
                )
                self._map_view_pub.publish(frame)
            ready = (
                self._mode in {"mapping", "relocalized"}
                and self._map_from_session is not None
                and odom_age is not None
                and cloud_age is not None
                and navigation_cloud_age is not None
                and odom_age <= self._source_max_age
                and cloud_age <= self._source_max_age
                and navigation_cloud_age <= self._source_max_age
            )
            payload = {
                "schema": "phanthy.navigation.fast_livo2_diagnostics.v1",
                "ready": ready,
                "session_name": self._session_name,
                "localization_state": self._mode,
                "map_alignment_confirmed": self._map_from_session is not None,
                "last_match": self._last_match,
                "odom_receive_age_sec": odom_age,
                "cloud_receive_age_sec": cloud_age,
                "navigation_cloud_receive_age_sec": navigation_cloud_age,
                "odom_source_age_sec": self._last_odom_source_age,
                "cloud_source_age_sec": self._last_cloud_source_age,
                "map_point_count": self._static_map.point_count,
                "reference_map_point_count": len(self._reference_points),
                "static_candidate_point_count": self._static_map.candidate_count,
                "static_free_cell_count": self._static_map.free_cell_count,
                "static_dynamic_track_count": self._static_map.dynamic_track_count,
                "static_dynamic_filter_enabled": (
                    self._static_map.dynamic_filter_enabled
                ),
                "static_quarantined_point_count": (
                    self._static_map.quarantined_point_count
                ),
                "static_pose_match_tolerance_sec": self._static_pose_match_tolerance,
                "cloud_pose_skew_sec": self._last_cloud_pose_skew_sec,
                "pending_navigation_cloud": self._pending_cloud is not None,
                "unmatched_navigation_cloud": self._unmatched_navigation_cloud,
                "unmatched_static_cloud": self._unmatched_static_cloud,
                "static_map_error": self._static_map_error,
                "map_view_context_point_count": self._map_view_context.point_count,
                "map_view_max_point_count": _MAP_VIEW_MAX_POINTS,
                "map_view_live_out_of_band_point_count": len(live_out_of_band),
                "obstacle_point_count": len(obstacle_points),
                "obstacle_height_range_m": [self._obstacle_min_height, self._obstacle_max_height],
                "invalid_odom": self._invalid_odom,
                "invalid_cloud": self._invalid_cloud,
                "raw_odom_frame": "camera_init -> aft_mapped",
                "canonical_odom_frame": "map -> base_link",
                "canonical_cloud_frame": "map",
                "obstacle_map_frame": "map",
                "static_map_frame": "map",
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
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
