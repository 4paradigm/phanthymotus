#!/usr/bin/env python3
"""Deterministic ROS fixture and full-output recorder for Nav2 stage six."""

from __future__ import annotations

import json
import math
import os
import signal
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import String, UInt8MultiArray


def _cloud_points(radius: float = 2.5, points: int = 720) -> tuple[bytes, bytes]:
    xyz_points = bytearray()
    xyzirt_points = bytearray()
    for index in range(points):
        angle = 2.0 * math.pi * index / points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        # Static lidar extrinsics lift the cloud by 0.46018 m. Keeping points
        # around base_link z=0 places the deterministic ring in the laser slice.
        z = -0.46018
        xyz_points.extend(struct.pack("<fff", x, y, z))
        time_offset_ns = 100_000_000.0 * index / points
        xyzirt_points.extend(
            struct.pack("<ffffHf", x, y, z, 1.0, index % 4, time_offset_ns)
        )
    return bytes(xyz_points), bytes(xyzirt_points)


def _v2_cloud(source_stamp_ns: int, points: int, data: bytes) -> bytes:
    frame = b"livox_frame"
    return (
        struct.pack(
            "<4sHHIIqH",
            b"PCV2",
            2,
            0x0001,
            22,
            points,
            source_stamp_ns,
            len(frame),
        )
        + frame
        + data
    )


class StageSixFixture(Node):
    def __init__(self) -> None:
        super().__init__("nav2_stage6_fixture")
        self._inputs_active = True
        self._input_stopped_at: float | None = None
        self._proposals = 0
        self._post_stale_nonzero = 0
        self._statuses = 0
        self._sensor_schema = os.getenv(
            "NAV2_FIXTURE_SENSOR_SCHEMA", "legacy"
        ).strip()
        if self._sensor_schema not in {"legacy", "v2"}:
            raise ValueError("NAV2_FIXTURE_SENSOR_SCHEMA must be legacy or v2")
        self._point_count = 720
        xyz_points, xyzirt_points = _cloud_points(points=self._point_count)
        self._legacy_cloud = (
            struct.pack("<II", 12, self._point_count) + xyz_points
        )
        self._v2_cloud_data = xyzirt_points
        signal.signal(signal.SIGUSR1, self._on_stop_signal)
        signal.signal(signal.SIGUSR2, self._on_resume_signal)

        self._loco_pub = self.create_publisher(
            String, "/ubuntu/loco/state", qos_profile_sensor_data
        )
        self._cloud_pub = self.create_publisher(
            UInt8MultiArray, "/ubuntu/lidar/cloud", qos_profile_sensor_data
        )
        self.create_subscription(
            String,
            "/nav2_stage6/fixture_control",
            self._on_control,
            10,
        )
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            "/ubuntu/navigation/nav2/velocity_proposal",
            self._on_proposal,
            reliable,
        )
        self.create_subscription(
            String,
            "/ubuntu/navigation/nav2/status",
            self._on_status,
            status_qos,
        )
        self.create_timer(0.1, self._publish_inputs)
        self.create_timer(5.0, self._publish_summary)

    def _publish_inputs(self) -> None:
        if not self._inputs_active:
            return
        source_stamp_ns = self.get_clock().now().nanoseconds
        state_payload = {
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "imu": {"rpy": [0.0, 0.0, 0.0]},
            "yaw_speed": 0.0,
        }
        if self._sensor_schema == "v2":
            state_payload.update(
                {
                    "schema": "phanthy.g1.loco_state.v2",
                    "schema_version": 2,
                    "source_stamp_ns": source_stamp_ns,
                    "frame_id": "odom_source",
                }
            )
        state = String()
        state.data = json.dumps(state_payload, separators=(",", ":"))
        cloud = UInt8MultiArray()
        cloud_payload = (
            _v2_cloud(
                source_stamp_ns,
                self._point_count,
                self._v2_cloud_data,
            )
            if self._sensor_schema == "v2"
            else self._legacy_cloud
        )
        cloud.data = list(cloud_payload)
        self._loco_pub.publish(state)
        self._cloud_pub.publish(cloud)

    def _on_control(self, message: String) -> None:
        command = message.data.strip()
        if command == "stop_inputs":
            self._set_inputs(False, command)
        elif command == "resume_inputs":
            self._set_inputs(True, command)
        else:
            return

    def _on_stop_signal(self, _signum: int, _frame: object) -> None:
        self._set_inputs(False, "stop_inputs_signal")

    def _on_resume_signal(self, _signum: int, _frame: object) -> None:
        self._set_inputs(True, "resume_inputs_signal")

    def _set_inputs(self, active: bool, command: str) -> None:
        self._inputs_active = active
        self._input_stopped_at = None if active else time.monotonic()
        self._emit("fixture_control", {"command": command})

    def _on_proposal(self, message: String) -> None:
        self._proposals += 1
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"malformed": message.data}
        velocity = payload.get("velocity") if isinstance(payload, dict) else None
        nonzero = isinstance(velocity, dict) and any(
            abs(float(velocity.get(axis, 0.0))) > 1e-9
            for axis in ("x", "y", "yaw")
        )
        post_stale = (
            self._input_stopped_at is not None
            and time.monotonic() - self._input_stopped_at > 0.55
        )
        if nonzero and post_stale:
            self._post_stale_nonzero += 1
        self._emit(
            "proposal",
            {
                "inputs_active": self._inputs_active,
                "post_stale": post_stale,
                "nonzero": nonzero,
                "payload": payload,
            },
        )

    def _on_status(self, message: String) -> None:
        self._statuses += 1
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"malformed": message.data}
        self._emit("status", {"payload": payload})

    def _publish_summary(self) -> None:
        self._emit(
            "summary",
            {
                "inputs_active": self._inputs_active,
                "proposal_count": self._proposals,
                "status_count": self._statuses,
                "post_stale_nonzero": self._post_stale_nonzero,
            },
        )

    @staticmethod
    def _emit(event: str, payload: dict) -> None:
        print(
            json.dumps(
                {
                    "event": event,
                    "observed_at_unix_ms": int(time.time() * 1000),
                    **payload,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )


def main() -> None:
    rclpy.init()
    node = StageSixFixture()
    try:
        rclpy.spin(node)
    finally:
        node._publish_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
