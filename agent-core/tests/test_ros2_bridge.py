from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


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
