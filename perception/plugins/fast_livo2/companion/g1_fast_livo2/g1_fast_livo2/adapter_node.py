"""Normalize FAST-LIVO2 raw outputs into the canonical Nav2 contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import threading
import time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray
from tf2_ros import TransformBroadcaster

from .frame_adapter_core import (
    InvalidFastLivo2Frame,
    Pose3,
    Quaternion,
    VoxelMap,
    canonical_base_pose,
    compose_pose,
    estimate_planar_relocalization,
    iter_xyz_points,
    quaternion_from_rpy,
    read_pcd_xyz,
    transform_points,
    yaw_from_quaternion,
)


class FastLivo2Adapter(Node):
    def __init__(self) -> None:
        super().__init__("g1_fast_livo2_adapter")
        self.declare_parameter("raw_odom_topic", "/ubuntu/navigation/fast_livo2/raw/odom")
        self.declare_parameter("raw_cloud_topic", "/ubuntu/navigation/fast_livo2/raw/cloud_registered")
        self.declare_parameter("odom_topic", "/ubuntu/navigation/odom")
        self.declare_parameter("cloud_topic", "/ubuntu/navigation/cloud_registered")
        self.declare_parameter("obstacle_map_topic", "/ubuntu/navigation/obstacle_map")
        self.declare_parameter("map_view_topic", "/ubuntu/navigation/fast_livo2/map_view")
        self.declare_parameter("diagnostics_topic", "/ubuntu/navigation/fast_livo2/diagnostics")
        self.declare_parameter("reset_topic", "/ubuntu/navigation/fast_livo2/reset_map")
        self.declare_parameter("map_control_topic", "/ubuntu/navigation/fast_livo2/map_control")
        self.declare_parameter("map_control_status_topic", "/ubuntu/navigation/fast_livo2/map_control_status")
        self.declare_parameter("map_root", "/opt/fast_livo_ws/src/fast_livo/Log/pcd")
        self.declare_parameter("source_max_age_sec", 0.5)
        self.declare_parameter("map_voxel_size_m", 0.10)
        self.declare_parameter("obstacle_min_height_m", -1.15)
        self.declare_parameter("obstacle_max_height_m", 0.80)
        self.declare_parameter("base_to_sensor_x", -0.00368)
        self.declare_parameter("base_to_sensor_y", 0.00003)
        self.declare_parameter("base_to_sensor_z", 0.46018)
        self.declare_parameter("base_to_sensor_roll", 0.0)
        self.declare_parameter("base_to_sensor_pitch", 0.04014257279586953)
        self.declare_parameter("base_to_sensor_yaw", 0.0)

        self._source_max_age = float(self.get_parameter("source_max_age_sec").value)
        self._map_root = Path(str(self.get_parameter("map_root").value)).resolve()
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
        self._map = VoxelMap(float(self.get_parameter("map_voxel_size_m").value))
        self._obstacle_min_height = float(self.get_parameter("obstacle_min_height_m").value)
        self._obstacle_max_height = float(self.get_parameter("obstacle_max_height_m").value)
        if not math.isfinite(self._obstacle_min_height) or not math.isfinite(self._obstacle_max_height):
            raise ValueError("obstacle projection heights must be finite")
        if self._obstacle_min_height >= self._obstacle_max_height:
            raise ValueError("obstacle_min_height_m must be less than obstacle_max_height_m")
        self._lock = threading.Lock()
        self._latest_pose: Pose3 | None = None
        self._latest_session_pose: Pose3 | None = None
        self._latest_session_points: tuple[tuple[float, float, float], ...] = ()
        self._reference_points: tuple[tuple[float, float, float], ...] = ()
        self._map_from_session: Pose3 | None = None
        self._mode = "idle"
        self._last_match: dict | None = None
        self._last_odom_monotonic: float | None = None
        self._last_cloud_monotonic: float | None = None
        self._last_odom_source_age: float | None = None
        self._last_cloud_source_age: float | None = None
        self._invalid_odom = 0
        self._invalid_cloud = 0
        self._session_name: str | None = None

        self._odom_pub = self.create_publisher(Odometry, str(self.get_parameter("odom_topic").value), qos_profile_sensor_data)
        self._cloud_pub = self.create_publisher(PointCloud2, str(self.get_parameter("cloud_topic").value), qos_profile_sensor_data)
        self._obstacle_map_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("obstacle_map_topic").value),
            qos_profile_sensor_data,
        )
        self._map_view_pub = self.create_publisher(UInt8MultiArray, str(self.get_parameter("map_view_topic").value), qos_profile_sensor_data)
        self._diagnostics_pub = self.create_publisher(String, str(self.get_parameter("diagnostics_topic").value), 10)
        self._map_control_status_pub = self.create_publisher(
            String, str(self.get_parameter("map_control_status_topic").value), 10
        )
        self._tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, str(self.get_parameter("raw_odom_topic").value), self._on_odom, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, str(self.get_parameter("raw_cloud_topic").value), self._on_cloud, qos_profile_sensor_data)
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
            if source_age < -0.1 or source_age > self._source_max_age:
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
        with self._lock:
            self._latest_pose = canonical

    def _on_cloud(self, message: PointCloud2) -> None:
        try:
            if message.header.frame_id.strip() != "camera_init":
                raise InvalidFastLivo2Frame("raw registered cloud must use camera_init")
            source_age = self._source_age(message.header.stamp)
            if source_age < -0.1 or source_age > self._source_max_age:
                raise InvalidFastLivo2Frame(f"raw cloud source age {source_age:.3f}s is invalid")
            points = list(
                iter_xyz_points(
                    fields=message.fields,
                    data=bytes(message.data),
                    point_step=int(message.point_step),
                    width=int(message.width),
                    height=int(message.height),
                    is_bigendian=bool(message.is_bigendian),
                )
            )
        except (InvalidFastLivo2Frame, ValueError, TypeError) as exc:
            self._invalid_cloud += 1
            self.get_logger().warning(f"rejecting FAST-LIVO2 cloud: {exc}")
            return

        with self._lock:
            self._latest_session_points = tuple(points)
            self._last_cloud_monotonic = time.monotonic()
            self._last_cloud_source_age = source_age
            map_from_session = self._map_from_session
        if map_from_session is None:
            return
        mapped_points = tuple(transform_points(map_from_session, points))
        self._cloud_pub.publish(self._xyz_cloud(mapped_points, message.header.stamp))
        with self._lock:
            self._map.add(mapped_points)

    def _on_reset(self, message: String) -> None:
        with self._lock:
            self._map.clear()
            self._session_name = message.data.strip() or None
            self._mode = "mapping"
            self._map_from_session = Pose3(
                0.0, 0.0, 0.0, quaternion_from_rpy(0.0, 0.0, 0.0)
            )
            self._latest_pose = None
            self._latest_session_pose = None
            self._latest_session_points = ()
            self._reference_points = ()
            self._last_match = None

    def _on_map_control(self, message: String) -> None:
        request_id = ""
        action = ""
        try:
            request = json.loads(message.data)
            if not isinstance(request, dict):
                raise InvalidFastLivo2Frame("map control must be an object")
            request_id = str(request.get("request_id", ""))
            action = str(request.get("action", ""))
            args = request.get("args") or {}
            if not isinstance(args, dict):
                raise InvalidFastLivo2Frame("map control args must be an object")
            if action == "load_map":
                result = self._load_saved_map(args)
            elif action == "relocalize":
                result = self._relocalize(args)
            elif action == "unload_map":
                result = self._unload_saved_map()
            else:
                result = {
                    "status": "error",
                    "error_code": "unsupported_action",
                    "error": f"unsupported map control action {action}",
                }
        except (InvalidFastLivo2Frame, OSError, TypeError, ValueError) as exc:
            result = {
                "status": "error",
                "error_code": "map_control_failed",
                "error": str(exc),
            }
        response = String()
        response.data = json.dumps(
            {"event": "response", "request_id": request_id, "action": action, **result},
            separators=(",", ":"),
        )
        self._map_control_status_pub.publish(response)

    def _load_saved_map(self, args: dict) -> dict:
        map_name = str(args.get("map_name", "")).strip()
        files = args.get("pcd_files") or []
        if not map_name or not isinstance(files, list) or not files:
            raise InvalidFastLivo2Frame("map_name and pcd_files are required")
        loaded = VoxelMap(float(self.get_parameter("map_voxel_size_m").value))
        for path in files:
            if not isinstance(path, str):
                raise InvalidFastLivo2Frame("pcd_files must contain paths")
            resolved = Path(path).resolve()
            if resolved.parent != self._map_root or not resolved.is_file():
                raise InvalidFastLivo2Frame("PCD path must be an existing map-root file")
            loaded.add(read_pcd_xyz(resolved))
        if loaded.point_count < 40:
            raise InvalidFastLivo2Frame("saved map has too few finite points")
        with self._lock:
            self._map = loaded
            self._reference_points = loaded.points
            self._session_name = map_name
            self._mode = "awaiting_relocalization"
            self._map_from_session = None
            self._latest_pose = None
            self._latest_session_pose = None
            self._latest_session_points = ()
            self._last_odom_monotonic = None
            self._last_cloud_monotonic = None
            self._last_match = None
        return {
            "status": "map_loaded",
            "map_name": map_name,
            "map_point_count": loaded.point_count,
            "localization_state": "awaiting_relocalization",
        }

    def _relocalize(self, args: dict) -> dict:
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
        with self._lock:
            self._map_from_session = result.map_from_session
            self._latest_pose = result.map_base_pose
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

    def _unload_saved_map(self) -> dict:
        with self._lock:
            map_name = self._session_name
            self._map.clear()
            self._session_name = None
            self._mode = "idle"
            self._map_from_session = None
            self._latest_pose = None
            self._latest_session_pose = None
            self._latest_session_points = ()
            self._reference_points = ()
            self._last_odom_monotonic = None
            self._last_cloud_monotonic = None
            self._last_match = None
        return {"status": "unloaded", "map_name": map_name}

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

    def _publish_periodic(self) -> None:
        now = time.monotonic()
        with self._lock:
            pose = self._latest_pose
            obstacle_points = self._map.project_xy(
                min_z=self._obstacle_min_height,
                max_z=self._obstacle_max_height,
            )
            if pose is not None and self._map.point_count:
                frame = UInt8MultiArray()
                frame.data = self._map.encode(pose)
                self._map_view_pub.publish(frame)
            odom_age = None if self._last_odom_monotonic is None else now - self._last_odom_monotonic
            cloud_age = None if self._last_cloud_monotonic is None else now - self._last_cloud_monotonic
            ready = (
                self._mode in {"mapping", "relocalized"}
                and self._map_from_session is not None
                and odom_age is not None
                and cloud_age is not None
                and odom_age <= self._source_max_age
                and cloud_age <= self._source_max_age
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
                "odom_source_age_sec": self._last_odom_source_age,
                "cloud_source_age_sec": self._last_cloud_source_age,
                "map_point_count": self._map.point_count,
                "obstacle_point_count": len(obstacle_points),
                "obstacle_height_range_m": [self._obstacle_min_height, self._obstacle_max_height],
                "invalid_odom": self._invalid_odom,
                "invalid_cloud": self._invalid_cloud,
                "raw_odom_frame": "camera_init -> aft_mapped",
                "canonical_odom_frame": "map -> base_link",
                "canonical_cloud_frame": "map",
                "obstacle_map_frame": "map",
            }
        if obstacle_points:
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
