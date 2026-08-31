from __future__ import annotations

import json
import math
import struct
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

import ros2_bridge  # noqa: E402


def _vector(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _orientation(yaw: float):
    return SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )


def _header(frame_id="map", sec=12, nanosec=34):
    return SimpleNamespace(
        frame_id=frame_id,
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
    )


class Ros2BridgeNavigationEncodingTest(unittest.TestCase):
    def test_pointcloud_is_encoded_for_the_existing_renderer(self) -> None:
        fields = [
            SimpleNamespace(name=name, offset=offset, datatype=7, count=1)
            for name, offset in (("x", 0), ("y", 4), ("z", 8))
        ]
        first_row = struct.pack("<ffffff", 1, 2, 3, 4, 5, 6)
        second_row = struct.pack("<ffffff", 7, 8, 9, 10, 11, 12)
        message = SimpleNamespace(
            point_step=12,
            row_step=28,
            width=2,
            height=2,
            is_bigendian=False,
            fields=fields,
            data=first_row + b"pad!" + second_row,
        )

        payload = ros2_bridge._encode_message(message, "sensor/pointcloud")

        self.assertEqual(struct.unpack_from("<II", payload), (12, 4))
        self.assertEqual(payload[8:], first_row + second_row)

    def test_odometry_is_encoded_as_navigation_json(self) -> None:
        message = SimpleNamespace(
            header=_header(),
            child_frame_id="base_link",
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=_vector(1.25, -0.5, 0.7),
                    orientation=_orientation(0.4),
                )
            ),
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=_vector(0.3, 0.0, 0.0),
                    angular=_vector(0.0, 0.0, -0.2),
                )
            ),
        )

        payload = json.loads(
            ros2_bridge._encode_message(message, "sensor/odometry")
        )

        self.assertEqual(payload["schema"], "phanthy.sensor.odometry.v1")
        self.assertEqual(payload["frame_id"], "map")
        self.assertEqual(payload["child_frame_id"], "base_link")
        self.assertEqual(payload["stamp_ns"], 12_000_000_034)
        self.assertAlmostEqual(payload["position"]["x"], 1.25)
        self.assertAlmostEqual(payload["yaw"], 0.4)
        self.assertAlmostEqual(payload["linear_velocity"]["x"], 0.3)
        self.assertAlmostEqual(payload["angular_velocity"]["z"], -0.2)

    def test_path_is_encoded_with_ordered_poses(self) -> None:
        message = SimpleNamespace(
            header=_header(sec=20, nanosec=50),
            poses=[
                SimpleNamespace(
                    pose=SimpleNamespace(
                        position=_vector(0.0, 0.0, 0.0),
                        orientation=_orientation(0.0),
                    )
                ),
                SimpleNamespace(
                    pose=SimpleNamespace(
                        position=_vector(1.0, 2.0, 0.0),
                        orientation=_orientation(1.2),
                    )
                ),
            ],
        )

        payload = json.loads(ros2_bridge._encode_message(message, "sensor/path"))

        self.assertEqual(payload["schema"], "phanthy.navigation.path.v1")
        self.assertEqual(payload["frame_id"], "map")
        self.assertEqual(len(payload["poses"]), 2)
        self.assertEqual(payload["poses"][1]["x"], 1.0)
        self.assertEqual(payload["poses"][1]["y"], 2.0)
        self.assertAlmostEqual(payload["poses"][1]["yaw"], 1.2)

    def test_occupancy_grid_uses_the_costmap_json_contract(self) -> None:
        message = SimpleNamespace(
            header=_header(),
            info=SimpleNamespace(
                resolution=0.05,
                width=2,
                height=1,
                origin=SimpleNamespace(
                    position=_vector(-1.0, -2.0, 0.0),
                    orientation=_orientation(0.2),
                ),
            ),
            data=[-1, 100],
        )

        payload = json.loads(
            ros2_bridge._encode_message(message, "sensor/occupancy-grid")
        )

        self.assertEqual(payload["schema"], "phanthy.navigation.costmap.v1")
        self.assertEqual(payload["data"], [-1, 100])

    def test_navigation_formats_resolve_to_their_declared_ros_types(self) -> None:
        sensor_msgs = ModuleType("sensor_msgs")
        sensor_msgs_msg = ModuleType("sensor_msgs.msg")
        pointcloud_type = type("PointCloud2", (), {})
        sensor_msgs_msg.PointCloud2 = pointcloud_type
        sensor_msgs.msg = sensor_msgs_msg
        nav_msgs = ModuleType("nav_msgs")
        nav_msgs_msg = ModuleType("nav_msgs.msg")
        occupancy_type = type("OccupancyGrid", (), {})
        nav_msgs_msg.OccupancyGrid = occupancy_type
        nav_msgs.msg = nav_msgs_msg

        with patch.dict(sys.modules, {
            "sensor_msgs": sensor_msgs,
            "sensor_msgs.msg": sensor_msgs_msg,
            "nav_msgs": nav_msgs,
            "nav_msgs.msg": nav_msgs_msg,
        }):
            self.assertIs(
                ros2_bridge._resolve_msg_type("sensor/pointcloud"),
                pointcloud_type,
            )
            self.assertIs(
                ros2_bridge._resolve_msg_type("sensor/occupancy-grid"),
                occupancy_type,
            )

    def test_existing_byte_array_and_text_messages_are_unchanged(self) -> None:
        self.assertEqual(
            ros2_bridge._encode_message(
                SimpleNamespace(data=[1, 2, 255]), "sensor/mapping"
            ),
            bytes([1, 2, 255]),
        )
        self.assertEqual(
            ros2_bridge._encode_message(
                SimpleNamespace(data='{"ok":true}'), "data/json"
            ),
            b'{"ok":true}',
        )


if __name__ == "__main__":
    unittest.main()
