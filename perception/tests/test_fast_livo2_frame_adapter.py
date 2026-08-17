from __future__ import annotations

import io
import math
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "navigation"
    / "runtime"
    / "g1_fast_livo2"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from g1_fast_livo2.frame_adapter_core import (  # noqa: E402
    InvalidFastLivo2Frame,
    Pose3,
    Quaternion,
    VoxelMap,
    canonical_base_pose,
    estimate_planar_relocalization,
    inverse_pose,
    iter_xyz_points,
    quaternion_from_rpy,
    read_pcd_xyz,
    source_age_is_valid,
    transform_points,
    yaw_from_quaternion,
)
from g1_fast_livo2.runtime_core import controlled_stop_succeeded  # noqa: E402


class FastLivo2FrameAdapterTest(unittest.TestCase):
    def test_controlled_stop_and_source_age_jitter_are_bounded(self) -> None:
        self.assertTrue(controlled_stop_succeeded(0))
        self.assertTrue(controlled_stop_succeeded(-2))
        self.assertTrue(controlled_stop_succeeded(-6))
        self.assertFalse(controlled_stop_succeeded(1))
        self.assertFalse(controlled_stop_succeeded(-9))

        self.assertTrue(
            source_age_is_valid(
                0.51,
                max_age_sec=0.5,
                tolerance_sec=0.05,
            )
        )
        self.assertFalse(
            source_age_is_valid(
                0.551,
                max_age_sec=0.5,
                tolerance_sec=0.05,
            )
        )

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

        full_map = VoxelMap(0.10)
        full_map.add((index * 0.20, 0.0, 0.0) for index in range(80_001))
        self.assertEqual(full_map.point_count, 80_001)

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

    def test_obstacle_projection_excludes_floor_and_ceiling(self) -> None:
        voxel_map = VoxelMap(0.10)
        voxel_map.add(
            [
                (1.01, 2.01, -1.30),
                (1.01, 2.01, -0.50),
                (1.04, 2.04, 0.40),
                (2.01, 3.01, 1.70),
                (3.01, 4.01, 0.20),
                (4.01, 5.01, -1.20),
            ]
        )

        projected = voxel_map.project_xy(min_z=-1.25, max_z=0.30)

        self.assertEqual(len(projected), 3)
        self.assertEqual(
            [(round(x, 2), round(y, 2), z) for x, y, z in projected],
            [(1.05, 2.05, 0.0), (3.05, 4.05, 0.0), (4.05, 5.05, 0.0)],
        )
        with self.assertRaises(ValueError):
            voxel_map.project_xy(min_z=1.0, max_z=1.0)

    def test_ascii_and_binary_pcd_are_read_fail_closed(self) -> None:
        header = (
            "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\n"
            "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
            "WIDTH 2\nHEIGHT 1\nPOINTS 2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            ascii_path = Path(directory) / "ascii.pcd"
            ascii_path.write_bytes(
                (header + "DATA ascii\n1 2 0 3\n4 5 0.5 6\n").encode("ascii")
            )
            self.assertEqual(read_pcd_xyz(ascii_path), ((1.0, 2.0, 0.0), (4.0, 5.0, 0.5)))

            binary_path = Path(directory) / "binary.pcd"
            binary_path.write_bytes(
                (header + "DATA binary\n").encode("ascii")
                + struct.pack("<ffffffff", 1, 2, 0, 3, 4, 5, 0.5, 6)
            )
            self.assertEqual(read_pcd_xyz(binary_path), ((1.0, 2.0, 0.0), (4.0, 5.0, 0.5)))

            compressed = Path(directory) / "compressed.pcd"
            compressed.write_text(header + "DATA binary_compressed\n", encoding="ascii")
            with self.assertRaisesRegex(InvalidFastLivo2Frame, "unsupported"):
                read_pcd_xyz(compressed)

            sampled_header = header.replace("WIDTH 2", "WIDTH 10").replace(
                "POINTS 2", "POINTS 10"
            )
            sampled = Path(directory) / "sampled.pcd"
            sampled.write_bytes(
                (sampled_header + "DATA binary\n").encode("ascii")
                + b"".join(
                    struct.pack("<ffff", float(index), 0.0, 0.0, 1.0)
                    for index in range(10)
                )
            )
            self.assertEqual(
                read_pcd_xyz(sampled, max_points=3),
                ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (8.0, 0.0, 0.0)),
            )

    def test_binary_pcd_reader_never_requests_the_full_payload(self) -> None:
        class TrackingStream(io.BytesIO):
            unbounded_read = False

            def read(self, size=-1):
                if size < 0:
                    self.unbounded_read = True
                return super().read(size)

        header = (
            "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            "WIDTH 4\nHEIGHT 1\nPOINTS 4\nDATA binary\n"
        ).encode("ascii")
        content = header + b"".join(
            struct.pack("<fff", float(index), 0.0, 0.0) for index in range(4)
        )
        stream = TrackingStream(content)
        with mock.patch.object(Path, "open", return_value=stream), mock.patch.object(
            Path,
            "stat",
            return_value=SimpleNamespace(st_size=len(content)),
        ):
            points = read_pcd_xyz("bounded.pcd", max_points=2)

        self.assertEqual(points, ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
        self.assertFalse(stream.unbounded_read)

    def test_bounded_planar_relocalization_recovers_known_pose(self) -> None:
        reference = []
        for index in range(60):
            reference.append((index * 0.10, 0.0, 0.0))
            reference.append((0.0, index * 0.10, 0.0))
        expected = Pose3(2.0, -1.0, 0.0, quaternion_from_rpy(0.0, 0.0, 0.30))
        session_points = tuple(transform_points(inverse_pose(expected), reference))
        result = estimate_planar_relocalization(
            reference_points=reference,
            session_points=session_points,
            session_base_pose=Pose3(0.0, 0.0, 0.0, Quaternion(0.0, 0.0, 0.0, 1.0)),
            initial_map_base_pose=Pose3(
                2.20, -1.15, 0.0, quaternion_from_rpy(0.0, 0.0, 0.40)
            ),
            search_xy_m=0.5,
            search_yaw_rad=0.25,
            min_z=-0.5,
            max_z=0.5,
            min_match_ratio=0.5,
        )
        self.assertAlmostEqual(result.map_base_pose.x, expected.x, delta=0.15)
        self.assertAlmostEqual(result.map_base_pose.y, expected.y, delta=0.15)
        self.assertAlmostEqual(
            yaw_from_quaternion(result.map_base_pose.q),
            yaw_from_quaternion(expected.q),
            delta=0.08,
        )
        self.assertGreaterEqual(result.match_ratio, 0.5)

        with self.assertRaisesRegex(InvalidFastLivo2Frame, "too few"):
            estimate_planar_relocalization(
                reference_points=reference[:5],
                session_points=session_points,
                session_base_pose=Pose3(
                    0.0, 0.0, 0.0, Quaternion(0.0, 0.0, 0.0, 1.0)
                ),
                initial_map_base_pose=expected,
                search_xy_m=0.5,
                search_yaw_rad=0.25,
                min_z=-0.5,
                max_z=0.5,
            )

    def test_relocalization_ties_prefer_operator_guess_and_stay_bounded(self) -> None:
        session = []
        for index in range(40):
            session.append((index * 0.20, 0.0, 0.0))
            session.append((0.0, index * 0.20, 0.0))
        reference = session + [(x + 2.0, y, z) for x, y, z in session]
        initial = Pose3(2.0, 0.0, 0.0, quaternion_from_rpy(0.0, 0.0, 0.0))

        result = estimate_planar_relocalization(
            reference_points=reference,
            session_points=session,
            session_base_pose=Pose3(
                0.0, 0.0, 0.0, Quaternion(0.0, 0.0, 0.0, 1.0)
            ),
            initial_map_base_pose=initial,
            search_xy_m=2.0,
            search_yaw_rad=0.10,
            min_z=-0.5,
            max_z=0.5,
            min_match_ratio=0.5,
        )

        self.assertAlmostEqual(result.map_base_pose.x, initial.x, places=6)
        self.assertLessEqual(abs(result.map_base_pose.y - initial.y), 2.0)
        self.assertAlmostEqual(
            yaw_from_quaternion(result.map_base_pose.q),
            yaw_from_quaternion(initial.q),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
