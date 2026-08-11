"""ROS bridge from the G1 navigation PCV2 stream to PointCloud2."""

from __future__ import annotations

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import rclpy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import UInt8MultiArray

from .canvas_pointcloud_core import InvalidCanvasPointCloud, decode_canvas_pointcloud


class CanvasPointCloudBridge(Node):
    def __init__(self) -> None:
        super().__init__("g1_canvas_pointcloud_bridge")
        self.declare_parameter("input_topic", "/ubuntu/navigation/lidar")
        self.declare_parameter(
            "output_topic", "/ubuntu/navigation/nav2/cloud"
        )
        self.declare_parameter("legacy_frame_id", "")
        self.declare_parameter("source_timeout", 0.5)
        self.declare_parameter("source_future_tolerance", 0.1)
        source_timeout = float(self.get_parameter("source_timeout").value)
        future_tolerance = float(
            self.get_parameter("source_future_tolerance").value
        )
        if source_timeout <= 0.0 or future_tolerance < 0.0:
            raise ValueError(
                "source_timeout must be positive and source_future_tolerance "
                "must be non-negative"
            )
        self._max_source_age_ns = int(source_timeout * 1_000_000_000)
        self._max_future_skew_ns = int(future_tolerance * 1_000_000_000)
        self._last_driver_source_stamp_ns: int | None = None
        self._invalid = 0
        self._publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_topic").value),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            UInt8MultiArray,
            str(self.get_parameter("input_topic").value),
            self._on_cloud,
            qos_profile_sensor_data,
        )

    def _on_cloud(self, message: UInt8MultiArray) -> None:
        try:
            cloud = decode_canvas_pointcloud(
                message.data,
                receive_stamp_ns=self.get_clock().now().nanoseconds,
                legacy_frame_id=str(
                    self.get_parameter("legacy_frame_id").value
                ),
                max_source_age_ns=self._max_source_age_ns,
                max_future_skew_ns=self._max_future_skew_ns,
            )
            if (
                cloud.timestamp_source == "driver"
                and self._last_driver_source_stamp_ns is not None
                and cloud.source_stamp_ns < self._last_driver_source_stamp_ns
            ):
                raise InvalidCanvasPointCloud(
                    "source_stamp_ns moved backwards"
                )
        except InvalidCanvasPointCloud as exc:
            self._invalid += 1
            if self._invalid <= 3 or self._invalid % 100 == 0:
                self.get_logger().warning(f"invalid canvas point cloud: {exc}")
            return

        if cloud.timestamp_source == "driver":
            self._last_driver_source_stamp_ns = cloud.source_stamp_ns

        output = PointCloud2()
        output.header.stamp.sec = cloud.source_stamp_ns // 1_000_000_000
        output.header.stamp.nanosec = cloud.source_stamp_ns % 1_000_000_000
        output.header.frame_id = cloud.frame_id
        output.height = 1
        output.width = cloud.point_count
        output.fields = [
            PointField(
                name=field.name,
                offset=field.offset,
                datatype=field.datatype,
                count=field.count,
            )
            for field in cloud.fields
        ]
        output.is_bigendian = False
        output.point_step = cloud.point_step
        output.row_step = cloud.point_step * cloud.point_count
        output.data = cloud.data
        output.is_dense = False
        self._publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CanvasPointCloudBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
