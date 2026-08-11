"""ROS 2 bridge from the G1 native locomotion JSON topic to Nav2 odometry."""

from __future__ import annotations

import json
import time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from .loco_odom_core import InvalidLocoState, OriginNormalizer


POSE_COVARIANCE = [
    0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.05, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 999.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 999.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 999.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.10,
]

TWIST_COVARIANCE = [
    0.10, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.10, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 999.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 999.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 999.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.20,
]


class G1LocoOdomBridge(Node):
    def __init__(self) -> None:
        super().__init__("g1_loco_odom_bridge")
        self.declare_parameter("input_topic", "/ubuntu/loco/state")
        self.declare_parameter("odom_topic", "/ubuntu/navigation/nav2/odom")
        self.declare_parameter(
            "status_topic", "/ubuntu/navigation/nav2/odom_status"
        )
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("reset_origin", True)
        self.declare_parameter("velocity_frame", "body")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("source_timeout", 0.5)
        self.declare_parameter("source_future_tolerance", 0.1)

        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._source_timeout = float(self.get_parameter("source_timeout").value)
        source_future_tolerance = float(
            self.get_parameter("source_future_tolerance").value
        )
        if self._source_timeout <= 0.0 or source_future_tolerance < 0.0:
            raise ValueError(
                "source_timeout must be positive and source_future_tolerance "
                "must be non-negative"
            )
        self._normalizer = OriginNormalizer(
            reset_origin=bool(self.get_parameter("reset_origin").value),
            velocity_frame=str(self.get_parameter("velocity_frame").value),
            max_source_age_ns=int(self._source_timeout * 1_000_000_000),
            max_future_skew_ns=int(source_future_tolerance * 1_000_000_000),
        )
        self._received = 0
        self._invalid = 0
        self._last_receive_monotonic: float | None = None
        self._last_source_stamp_ns: int | None = None
        self._last_timestamp_source: str | None = None
        self._last_frame_source: str | None = None
        self._last_source_schema: str | None = None

        self._odom_publisher = self.create_publisher(
            Odometry, str(self.get_parameter("odom_topic").value), 20
        )
        self._status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            String,
            str(self.get_parameter("input_topic").value),
            self._on_state,
            qos_profile_sensor_data,
        )
        # The navigation gate rejects status receipts older than source_timeout
        # (0.5 s by default), so publish health at the native 10 Hz cadence.
        self.create_timer(0.1, self._publish_status)

    def _on_state(self, message: String) -> None:
        try:
            odom = self._normalizer.convert(
                message.data,
                receive_stamp_ns=self.get_clock().now().nanoseconds,
            )
        except (InvalidLocoState, TypeError, ValueError) as exc:
            self._invalid += 1
            if self._invalid <= 3 or self._invalid % 100 == 0:
                self.get_logger().warning(f"invalid loco state: {exc}")
            return

        self._received += 1
        self._last_receive_monotonic = time.monotonic()
        self._last_source_stamp_ns = odom.source_stamp_ns
        self._last_timestamp_source = odom.timestamp_source
        self._last_frame_source = odom.frame_source
        self._last_source_schema = odom.source_schema
        half_yaw = odom.yaw / 2.0
        import math

        quaternion_z = math.sin(half_yaw)
        quaternion_w = math.cos(half_yaw)

        output = Odometry()
        output.header.stamp.sec = odom.source_stamp_ns // 1_000_000_000
        output.header.stamp.nanosec = odom.source_stamp_ns % 1_000_000_000
        output.header.frame_id = self._odom_frame
        output.child_frame_id = self._base_frame
        output.pose.pose.position.x = odom.x
        output.pose.pose.position.y = odom.y
        output.pose.pose.orientation.z = quaternion_z
        output.pose.pose.orientation.w = quaternion_w
        output.pose.covariance = POSE_COVARIANCE
        output.twist.twist.linear.x = odom.vx
        output.twist.twist.linear.y = odom.vy
        output.twist.twist.angular.z = odom.wz
        output.twist.covariance = TWIST_COVARIANCE
        self._odom_publisher.publish(output)

        if self._publish_tf:
            transform = TransformStamped()
            transform.header.stamp = output.header.stamp
            transform.header.frame_id = self._odom_frame
            transform.child_frame_id = self._base_frame
            transform.transform.translation.x = odom.x
            transform.transform.translation.y = odom.y
            transform.transform.rotation.z = quaternion_z
            transform.transform.rotation.w = quaternion_w
            self._tf_broadcaster.sendTransform(transform)

    def _publish_status(self) -> None:
        receive_age = None
        if self._last_receive_monotonic is not None:
            receive_age = max(0.0, time.monotonic() - self._last_receive_monotonic)
        source_age = None
        if (
            self._last_source_stamp_ns is not None
            and self._last_timestamp_source == "driver"
        ):
            source_age = (
                self.get_clock().now().nanoseconds - self._last_source_stamp_ns
            ) / 1_000_000_000.0
        timestamp_fresh = (
            self._last_timestamp_source == "adapter_receive"
            or (
                self._last_timestamp_source == "driver"
                and source_age is not None
                and -0.1 <= source_age <= self._source_timeout
            )
        )
        state = (
            "ready"
            if receive_age is not None
            and receive_age <= self._source_timeout
            and timestamp_fresh
            else "waiting_for_native_odom"
        )
        message = String()
        message.data = json.dumps(
            {
                "state": state,
                "received": self._received,
                "invalid": self._invalid,
                "receive_age_sec": round(receive_age, 3)
                if receive_age is not None
                else None,
                "source_age_sec": round(source_age, 3)
                if source_age is not None
                else None,
                "source_stamp_ns": self._last_source_stamp_ns,
                "timestamp_source": self._last_timestamp_source,
                "frame_source": self._last_frame_source,
                "source_schema": self._last_source_schema,
                "odom_frame": self._odom_frame,
                "base_frame": self._base_frame,
            }
        )
        self._status_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = G1LocoOdomBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
