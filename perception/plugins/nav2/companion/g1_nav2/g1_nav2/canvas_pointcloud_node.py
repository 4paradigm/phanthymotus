"""ROS bridge from the released G1 canvas cloud to PointCloud2."""

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
        self.declare_parameter("input_topic", "/ubuntu/lidar/cloud")
        self.declare_parameter(
            "output_topic", "/ubuntu/navigation/nav2/cloud"
        )
        self.declare_parameter("legacy_frame_id", "")
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
            )
        except InvalidCanvasPointCloud as exc:
            self._invalid += 1
            if self._invalid <= 3 or self._invalid % 100 == 0:
                self.get_logger().warning(f"invalid canvas point cloud: {exc}")
            return

        output = PointCloud2()
        output.header.stamp.sec = cloud.source_stamp_ns // 1_000_000_000
        output.header.stamp.nanosec = cloud.source_stamp_ns % 1_000_000_000
        output.header.frame_id = cloud.frame_id
        output.height = 1
        output.width = cloud.point_count
        output.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
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
