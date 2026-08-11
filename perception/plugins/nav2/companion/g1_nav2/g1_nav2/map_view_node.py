"""Publish the live Nav2 occupancy map in Canvas' existing mapping format."""

from __future__ import annotations

import math

from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from std_msgs.msg import UInt8MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

from .map_view_core import (
    CANVAS_MAPPING_MAX_POINTS,
    CanvasMapSnapshot,
    InvalidMapView,
    build_occupancy_snapshot,
    encode_canvas_mapping_frame,
)


_MAP_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class CanvasMapView(Node):
    def __init__(self) -> None:
        super().__init__("g1_nav2_canvas_map_view")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter(
            "output_topic", "/ubuntu/navigation/nav2/map_view"
        )
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("occupancy_threshold", 65)
        self.declare_parameter("max_points", CANVAS_MAPPING_MAX_POINTS)
        self.declare_parameter("publish_rate_hz", 1.0)

        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._occupancy_threshold = int(
            self.get_parameter("occupancy_threshold").value
        )
        self._max_points = int(self.get_parameter("max_points").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if not math.isfinite(publish_rate_hz) or publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be finite and positive")

        self._snapshot: CanvasMapSnapshot | None = None
        self._invalid_maps = 0
        self._tf_failures = 0
        self._publisher = self.create_publisher(
            UInt8MultiArray,
            str(self.get_parameter("output_topic").value),
            qos_profile_sensor_data,
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("map_topic").value),
            self._on_map,
            _MAP_QOS,
        )
        self.create_timer(1.0 / publish_rate_hz, self._publish_view)

    def _on_map(self, message: OccupancyGrid) -> None:
        source_frame = message.header.frame_id.strip()
        if source_frame != self._map_frame:
            self._report_invalid_map(
                f"map frame must be {self._map_frame!r}, got {source_frame!r}"
            )
            return

        origin = message.info.origin
        try:
            snapshot = build_occupancy_snapshot(
                width=int(message.info.width),
                height=int(message.info.height),
                resolution=float(message.info.resolution),
                origin_x=float(origin.position.x),
                origin_y=float(origin.position.y),
                origin_yaw=_yaw_from_quaternion(
                    float(origin.orientation.x),
                    float(origin.orientation.y),
                    float(origin.orientation.z),
                    float(origin.orientation.w),
                ),
                data=message.data,
                occupancy_threshold=self._occupancy_threshold,
                max_points=self._max_points,
            )
        except (InvalidMapView, TypeError, ValueError) as exc:
            self._report_invalid_map(str(exc))
            return

        self._snapshot = snapshot
        self._invalid_maps = 0

    def _report_invalid_map(self, error: str) -> None:
        self._snapshot = None
        self._invalid_maps += 1
        if self._invalid_maps <= 3 or self._invalid_maps % 100 == 0:
            self.get_logger().warning(f"invalid Canvas map view input: {error}")

    def _publish_view(self) -> None:
        if self._snapshot is None:
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._base_frame,
                Time(),
            )
        except TransformException as exc:
            self._tf_failures += 1
            if self._tf_failures <= 3 or self._tf_failures % 100 == 0:
                self.get_logger().warning(
                    f"Canvas map view waiting for map -> base_link TF: {exc}"
                )
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        try:
            payload = encode_canvas_mapping_frame(
                self._snapshot,
                robot_x=float(translation.x),
                robot_y=float(translation.y),
                robot_yaw=_yaw_from_quaternion(
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                ),
            )
        except InvalidMapView as exc:
            self.get_logger().error(f"cannot encode Canvas map view: {exc}")
            return

        self._tf_failures = 0
        message = UInt8MultiArray()
        message.data = payload
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CanvasMapView()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
