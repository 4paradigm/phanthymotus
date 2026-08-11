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
    LidarClockNormalizer,
    decode_canvas_pointcloud,
)
from .timestamp_contract import (
    InvalidSourceTimestamp,
    validate_source_timestamp_ns,
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
        self.declare_parameter("timestamp_mode", "auto")
        self.declare_parameter("clock_warmup_samples", 8)
        self.declare_parameter("clock_window_samples", 200)
        self.declare_parameter("already_aligned_tolerance", 2.0)
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
        self._input_topic = str(self.get_parameter("input_topic").value)
        self._max_source_age_ns = int(max_source_age * 1_000_000_000)
        self._max_future_skew_ns = int(future_tolerance * 1_000_000_000)
        # rclpy.node.Node owns ``self._clock``; do not shadow it or
        # ``get_clock()`` will return the LiDAR normalizer instead of ROS time.
        self._lidar_clock = LidarClockNormalizer(
            mode=str(self.get_parameter("timestamp_mode").value),
            warmup_samples=int(
                self.get_parameter("clock_warmup_samples").value
            ),
            window_samples=int(
                self.get_parameter("clock_window_samples").value
            ),
            aligned_tolerance_ns=int(
                float(
                    self.get_parameter("already_aligned_tolerance").value
                )
                * 1_000_000_000
            ),
        )
        self._last_source_stamp_ns: int | None = None
        self._source_topology_error: str | None = None
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
            self._input_topic,
            self._on_cloud,
            _SENSOR_QOS,
        )

    def _on_cloud(self, message: UInt8MultiArray) -> None:
        self._received += 1
        receive_stamp_ns = self.get_clock().now().nanoseconds
        cloud: CanvasPointCloud | None = None
        normalized_stamp_ns: int | None = None
        try:
            input_publishers = self.count_publishers(self._input_topic)
            if input_publishers != 1:
                topology_error = (
                    f"expected exactly one publisher on {self._input_topic}; "
                    f"found {input_publishers}"
                )
                if self._source_topology_error is None:
                    self._lidar_clock.reset()
                    self._last_source_stamp_ns = None
                self._source_topology_error = topology_error
                raise InvalidCanvasPointCloud(topology_error)
            self._source_topology_error = None
            cloud = decode_canvas_pointcloud(
                message.data,
                receive_stamp_ns=receive_stamp_ns,
                output_frame_id=self._output_frame_id,
                max_source_age_ns=self._max_source_age_ns,
                max_future_skew_ns=self._max_future_skew_ns,
            )
            if not cloud.raw_lidar_stamp_valid or cloud.raw_lidar_stamp_ns is None:
                raise InvalidCanvasPointCloud(
                    "raw LiDAR header timestamp is required for Nav2"
                )
            normalized_stamp_ns = self._lidar_clock.normalize(
                raw_stamp_ns=cloud.raw_lidar_stamp_ns,
                driver_receive_stamp_ns=cloud.source_stamp_ns,
                scan_end_offset_ns=cloud.scan_end_offset_ns,
            )
            if normalized_stamp_ns is None:
                self._publish_status(
                    state="warming_up",
                    receive_stamp_ns=receive_stamp_ns,
                    normalized_stamp_ns=None,
                    cloud=cloud,
                )
                return
            validate_source_timestamp_ns(
                normalized_stamp_ns,
                receive_stamp_ns,
                max_source_age_ns=self._max_source_age_ns,
                max_future_skew_ns=self._max_future_skew_ns,
            )
            if (
                self._last_source_stamp_ns is not None
                and normalized_stamp_ns <= self._last_source_stamp_ns
            ):
                raise InvalidCanvasPointCloud(
                    "normalized LiDAR scan-start timestamp did not advance"
                )
        except (
            InvalidCanvasPointCloud,
            InvalidSourceTimestamp,
            TypeError,
            ValueError,
        ) as exc:
            self._invalid += 1
            if self._invalid <= 3 or self._invalid % 100 == 0:
                self.get_logger().warning(f"invalid canvas point cloud: {exc}")
            self._publish_status(
                state="invalid",
                receive_stamp_ns=receive_stamp_ns,
                normalized_stamp_ns=normalized_stamp_ns,
                cloud=cloud,
                error=str(exc),
            )
            return

        assert cloud is not None and normalized_stamp_ns is not None
        output = PointCloud2()
        output.header.stamp.sec, output.header.stamp.nanosec = divmod(
            normalized_stamp_ns, 1_000_000_000
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
        self._last_source_stamp_ns = normalized_stamp_ns
        self._published += 1
        self._publish_status(
            state="ready",
            receive_stamp_ns=receive_stamp_ns,
            normalized_stamp_ns=normalized_stamp_ns,
            cloud=cloud,
        )

    def _publish_status(
        self,
        *,
        state: str,
        receive_stamp_ns: int,
        normalized_stamp_ns: int | None,
        cloud: CanvasPointCloud | None,
        error: str | None = None,
    ) -> None:
        clock = self._lidar_clock.snapshot()
        driver_receive_stamp_ns = (
            cloud.source_stamp_ns if cloud is not None else None
        )
        payload = {
            "schema": "phanthy.navigation.lidar_adapter.v2",
            "state": state,
            "source_schema": cloud.source_schema if cloud is not None else None,
            "timestamp_source": (
                "raw_lidar_header_normalized_to_driver_receive"
                if cloud is not None
                else None
            ),
            "metadata_timestamp_source": (
                cloud.timestamp_source if cloud is not None else None
            ),
            "frame_source": cloud.frame_source if cloud is not None else None,
            "bridge_receive_stamp_ns": receive_stamp_ns,
            "source_stamp_ns": normalized_stamp_ns,
            "source_age_ms": (
                round(
                    (receive_stamp_ns - normalized_stamp_ns) / 1_000_000,
                    3,
                )
                if normalized_stamp_ns is not None
                else None
            ),
            "driver_receive_stamp_ns": driver_receive_stamp_ns,
            "driver_receive_age_ms": (
                round(
                    (receive_stamp_ns - driver_receive_stamp_ns) / 1_000_000,
                    3,
                )
                if driver_receive_stamp_ns is not None
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
            "scan_end_offset_ms": (
                round(cloud.scan_end_offset_ns / 1_000_000, 3)
                if cloud is not None
                else None
            ),
            "clock": {
                "ready": clock.ready,
                "mode": clock.mode,
                "samples": clock.samples,
                "offset_ns": clock.offset_ns,
                "residual_ms": (
                    round(clock.residual_ns / 1_000_000, 3)
                    if clock.residual_ns is not None
                    else None
                ),
                "resets": clock.resets,
            },
            "point_count": cloud.point_count if cloud is not None else None,
            "metadata_footer": "PCLMETA2",
            "input_topic": self._input_topic,
            "input_publishers": self.count_publishers(self._input_topic),
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
