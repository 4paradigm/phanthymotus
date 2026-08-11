"""Bridge the timestamped G1 Driver cloud envelope to Nav2 PointCloud2."""

from __future__ import annotations

import json

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
import rclpy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray

from .canvas_pointcloud_core import (
    CanvasPointCloud,
    InvalidCanvasPointCloud,
    decode_canvas_pointcloud,
)


_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class CanvasPointCloudBridge(Node):
    def __init__(self) -> None:
        super().__init__("g1_canvas_pointcloud_bridge")
        self.declare_parameter("input_topic", "/ubuntu/lidar/cloud")
        self.declare_parameter(
            "output_topic", "/ubuntu/navigation/nav2/cloud"
        )
        self.declare_parameter(
            "status_topic", "/ubuntu/navigation/nav2/lidar_status"
        )
        self.declare_parameter("output_frame_id", "livox_frame")
        self.declare_parameter("max_source_age", 0.5)
        self.declare_parameter("source_future_tolerance", 0.1)
        max_source_age = float(self.get_parameter("max_source_age").value)
        future_tolerance = float(
            self.get_parameter("source_future_tolerance").value
        )
        if max_source_age <= 0.0 or future_tolerance < 0.0:
            raise ValueError(
                "max_source_age must be positive and "
                "source_future_tolerance must be non-negative"
            )
        self._output_frame_id = str(
            self.get_parameter("output_frame_id").value
        )
        self._max_source_age_ns = int(max_source_age * 1_000_000_000)
        self._max_future_skew_ns = int(future_tolerance * 1_000_000_000)
        self._last_source_stamp_ns: int | None = None
        self._invalid = 0
        self._received = 0
        self._published = 0
        self._publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_topic").value),
            _SENSOR_QOS,
        )
        self._status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            _SENSOR_QOS,
        )
        self.create_subscription(
            UInt8MultiArray,
            str(self.get_parameter("input_topic").value),
            self._on_cloud,
            _SENSOR_QOS,
        )

    def _on_cloud(self, message: UInt8MultiArray) -> None:
        self._received += 1
        receive_stamp_ns = self.get_clock().now().nanoseconds
        cloud: CanvasPointCloud | None = None
        try:
            cloud = decode_canvas_pointcloud(
                message.data,
                receive_stamp_ns=receive_stamp_ns,
                output_frame_id=self._output_frame_id,
                max_source_age_ns=self._max_source_age_ns,
                max_future_skew_ns=self._max_future_skew_ns,
            )
            if (
                self._last_source_stamp_ns is not None
                and cloud.source_stamp_ns <= self._last_source_stamp_ns
            ):
                raise InvalidCanvasPointCloud(
                    "Driver receive timestamp did not advance"
                )
        except (InvalidCanvasPointCloud, TypeError, ValueError) as exc:
            self._invalid += 1
            if self._invalid <= 3 or self._invalid % 100 == 0:
                self.get_logger().warning(f"invalid canvas point cloud: {exc}")
            self._publish_status(
                state="invalid",
                receive_stamp_ns=receive_stamp_ns,
                cloud=cloud,
                error=str(exc),
            )
            return

        output = PointCloud2()
        output.header.stamp.sec, output.header.stamp.nanosec = divmod(
            cloud.source_stamp_ns, 1_000_000_000
        )
        output.header.frame_id = cloud.frame_id
        output.height = cloud.height
        output.width = cloud.width
        output.fields = [
            PointField(
                name=field.name,
                offset=field.offset,
                datatype=field.datatype,
                count=field.count,
            )
            for field in cloud.fields
        ]
        output.is_bigendian = cloud.is_bigendian
        output.point_step = cloud.point_step
        output.row_step = cloud.row_step
        output.data = cloud.data
        output.is_dense = cloud.is_dense
        self._publisher.publish(output)
        self._last_source_stamp_ns = cloud.source_stamp_ns
        self._published += 1
        self._publish_status(
            state="ready",
            receive_stamp_ns=receive_stamp_ns,
            cloud=cloud,
        )

    def _publish_status(
        self,
        *,
        state: str,
        receive_stamp_ns: int,
        cloud: CanvasPointCloud | None,
        error: str | None = None,
    ) -> None:
        source_stamp_ns = cloud.source_stamp_ns if cloud is not None else None
        payload = {
            "schema": "phanthy.navigation.lidar_adapter.v2",
            "state": state,
            "source_schema": cloud.source_schema if cloud is not None else None,
            "timestamp_source": (
                cloud.timestamp_source if cloud is not None else None
            ),
            "frame_source": cloud.frame_source if cloud is not None else None,
            "bridge_receive_stamp_ns": receive_stamp_ns,
            "source_stamp_ns": source_stamp_ns,
            "source_age_ms": (
                round((receive_stamp_ns - source_stamp_ns) / 1_000_000, 3)
                if source_stamp_ns is not None
                else None
            ),
            "driver_receive_monotonic_ns": (
                cloud.driver_receive_monotonic_ns if cloud is not None else None
            ),
            "output_frame_id": cloud.frame_id if cloud is not None else None,
            "raw_lidar_header": (
                {
                    "stamp_ns": cloud.raw_lidar_stamp_ns,
                    "stamp_valid": cloud.raw_lidar_stamp_valid,
                    "frame_id": cloud.raw_lidar_frame_id,
                }
                if cloud is not None
                else None
            ),
            "point_data_transform": (
                cloud.point_data_transform if cloud is not None else None
            ),
            "point_count": cloud.point_count if cloud is not None else None,
            "metadata_footer": "PCLMETA2",
            "received": self._received,
            "published": self._published,
            "invalid": self._invalid,
            "error": error,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._status_publisher.publish(message)


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
