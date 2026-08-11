"""Normalize the robot's native PointCloud2 clock and pass it to Nav2."""

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
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from .canvas_pointcloud_core import (
    InvalidCanvasPointCloud,
    LidarClockNormalizer,
    point_time_max_offset_ns,
    validate_standard_pointcloud,
)
from .timestamp_contract import InvalidSourceTimestamp, validate_source_timestamp_ns


_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class CanvasPointCloudBridge(Node):
    def __init__(self) -> None:
        super().__init__("g1_canvas_pointcloud_bridge")
        self.declare_parameter("input_topic", "/utlidar/cloud")
        self.declare_parameter(
            "output_topic", "/ubuntu/navigation/nav2/cloud"
        )
        self.declare_parameter(
            "status_topic", "/ubuntu/navigation/nav2/lidar_status"
        )
        self.declare_parameter("timestamp_mode", "auto")
        self.declare_parameter("clock_warmup_samples", 8)
        self.declare_parameter("clock_window_samples", 200)
        self.declare_parameter("already_aligned_tolerance", 2.0)
        self.declare_parameter("max_normalized_source_age", 2.0)
        self.declare_parameter("source_future_tolerance", 0.1)
        max_source_age = float(
            self.get_parameter("max_normalized_source_age").value
        )
        future_tolerance = float(
            self.get_parameter("source_future_tolerance").value
        )
        if max_source_age <= 0.0 or future_tolerance < 0.0:
            raise ValueError(
                "max_normalized_source_age must be positive and "
                "source_future_tolerance must be non-negative"
            )
        self._max_source_age_ns = int(max_source_age * 1_000_000_000)
        self._max_future_skew_ns = int(future_tolerance * 1_000_000_000)
        self._clock = LidarClockNormalizer(
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
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._on_cloud,
            _SENSOR_QOS,
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        self._received += 1
        receive_stamp_ns = self.get_clock().now().nanoseconds
        raw_stamp_ns: int | None = None
        scan_end_offset_ns: int | None = None
        try:
            raw_stamp_ns = validate_standard_pointcloud(
                stamp_sec=message.header.stamp.sec,
                stamp_nanosec=message.header.stamp.nanosec,
                receive_stamp_ns=None,
                frame_id=message.header.frame_id,
                height=message.height,
                width=message.width,
                point_step=message.point_step,
                row_step=message.row_step,
                data_length=len(message.data),
                field_names=(field.name for field in message.fields),
                max_source_age_ns=self._max_source_age_ns,
                max_future_skew_ns=self._max_future_skew_ns,
            )
            scan_end_offset_ns = point_time_max_offset_ns(
                data=message.data,
                fields=message.fields,
                height=message.height,
                width=message.width,
                point_step=message.point_step,
                row_step=message.row_step,
                is_bigendian=message.is_bigendian,
            )
            source_stamp_ns = self._clock.normalize(
                raw_stamp_ns=raw_stamp_ns,
                receive_stamp_ns=receive_stamp_ns,
                scan_end_offset_ns=scan_end_offset_ns,
            )
            if source_stamp_ns is None:
                self._publish_status(
                    state="warming_up",
                    receive_stamp_ns=receive_stamp_ns,
                    raw_stamp_ns=raw_stamp_ns,
                    normalized_stamp_ns=None,
                    scan_end_offset_ns=scan_end_offset_ns,
                )
                return
            validate_source_timestamp_ns(
                source_stamp_ns,
                receive_stamp_ns,
                max_source_age_ns=self._max_source_age_ns,
                max_future_skew_ns=self._max_future_skew_ns,
            )
            if (
                self._last_source_stamp_ns is not None
                and source_stamp_ns <= self._last_source_stamp_ns
            ):
                raise InvalidCanvasPointCloud(
                    "normalized PointCloud2 header stamp did not advance"
                )
        except (InvalidCanvasPointCloud, InvalidSourceTimestamp, ValueError) as exc:
            self._invalid += 1
            if self._invalid <= 3 or self._invalid % 100 == 0:
                self.get_logger().warning(f"invalid canvas point cloud: {exc}")
            self._publish_status(
                state="invalid",
                receive_stamp_ns=receive_stamp_ns,
                raw_stamp_ns=raw_stamp_ns,
                normalized_stamp_ns=None,
                scan_end_offset_ns=scan_end_offset_ns,
                error=str(exc),
            )
            return

        self._last_source_stamp_ns = source_stamp_ns
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(
            source_stamp_ns, 1_000_000_000
        )
        self._publisher.publish(message)
        self._published += 1
        self._publish_status(
            state="ready",
            receive_stamp_ns=receive_stamp_ns,
            raw_stamp_ns=raw_stamp_ns,
            normalized_stamp_ns=source_stamp_ns,
            scan_end_offset_ns=scan_end_offset_ns,
        )

    def _publish_status(
        self,
        *,
        state: str,
        receive_stamp_ns: int,
        raw_stamp_ns: int | None,
        normalized_stamp_ns: int | None,
        scan_end_offset_ns: int | None,
        error: str | None = None,
    ) -> None:
        clock = self._clock.snapshot()
        payload = {
            "schema": "phanthy.navigation.lidar_clock.v1",
            "state": state,
            "timestamp_source": "native_header_normalized",
            "frame_source": "native_header",
            "receive_stamp_ns": receive_stamp_ns,
            "raw_stamp_ns": raw_stamp_ns,
            "normalized_stamp_ns": normalized_stamp_ns,
            "normalized_source_age_ms": (
                round((receive_stamp_ns - normalized_stamp_ns) / 1_000_000, 3)
                if normalized_stamp_ns is not None
                else None
            ),
            "scan_end_offset_ms": (
                round(scan_end_offset_ns / 1_000_000, 3)
                if scan_end_offset_ns is not None
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
