from __future__ import annotations

import math
import struct
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "fast_livo2"
    / "companion"
    / "g1_fast_livo2"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from g1_fast_livo2.frame_adapter_core import (  # noqa: E402
    InvalidFastLivo2Frame,
    Pose3,
    Quaternion,
    VoxelMap,
    canonical_base_pose,
    iter_xyz_points,
    quaternion_from_rpy,
    yaw_from_quaternion,
)


class FastLivo2FrameAdapterTest(unittest.TestCase):
    def test_sensor_pose_is_converted_to_base_pose(self) -> None:
        base_to_sensor = Pose3(0.0, 0.0, 0.46, Quaternion(0.0, 0.0, 0.0, 1.0))
        map_to_sensor = Pose3(1.0, 2.0, 0.46, Quaternion(0.0, 0.0, 0.0, 1.0))

        base = canonical_base_pose(map_to_sensor, base_to_sensor)

        self.assertAlmostEqual(base.x, 1.0)
        self.assertAlmostEqual(base.y, 2.0)
        self.assertAlmostEqual(base.z, 0.0)

    def test_sensor_extrinsic_is_rotated_in_map_frame(self) -> None:
        yaw = math.pi / 2
        map_to_sensor = Pose3(2.0, 3.0, 0.0, quaternion_from_rpy(0.0, 0.0, yaw))
        base_to_sensor = Pose3(1.0, 0.0, 0.0, Quaternion(0.0, 0.0, 0.0, 1.0))

        base = canonical_base_pose(map_to_sensor, base_to_sensor)

        self.assertAlmostEqual(base.x, 2.0, places=6)
        self.assertAlmostEqual(base.y, 2.0, places=6)
        self.assertAlmostEqual(yaw_from_quaternion(base.q), yaw, places=6)

    def test_pointcloud_layout_validation_and_voxel_encoding(self) -> None:
        fields = [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
        ]
        data = struct.pack("<ffffff", 1.01, 2.01, 0.0, 1.04, 2.04, 0.0)
        points = list(
            iter_xyz_points(
                fields=fields,
                data=data,
                point_step=12,
                width=2,
                height=1,
                is_bigendian=False,
            )
        )
        self.assertEqual(len(points), 2)

        voxel_map = VoxelMap(0.10)
        voxel_map.add(points)
        self.assertEqual(voxel_map.point_count, 1)
        frame = voxel_map.encode(
            Pose3(0.5, -0.25, 0.0, quaternion_from_rpy(0.0, 0.0, 1.0))
        )
        robot_x, robot_y, robot_yaw, flags, point_count = struct.unpack_from(
            "<fffBI", frame
        )
        self.assertAlmostEqual(robot_x, 0.5)
        self.assertAlmostEqual(robot_y, -0.25)
        self.assertAlmostEqual(robot_yaw, 1.0)
        self.assertEqual(flags, 0x03)
        self.assertEqual(point_count, 1)

        with self.assertRaises(InvalidFastLivo2Frame):
            list(
                iter_xyz_points(
                    fields=fields,
                    data=data[:11],
                    point_step=12,
                    width=2,
                    height=1,
                    is_bigendian=False,
                )
            )
        with self.assertRaises(InvalidFastLivo2Frame):
            list(
                iter_xyz_points(
                    fields=fields,
                    data=data,
                    point_step=12,
                    width=2,
                    height=1,
                    is_bigendian=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
