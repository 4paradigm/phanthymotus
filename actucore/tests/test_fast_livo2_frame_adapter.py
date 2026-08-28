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
    FastLivo2PersistenceError,
    InvalidFastLivo2Frame,
    OdomHealthMonitor,
    Pose3,
    Quaternion,
    RelocalizationRejected,
    TemporalOccupancyMap,
    VoxelMap,
    bracketed_stamped_pose,
    canonical_base_pose,
    compose_pose,
    encode_map_view_points,
    estimate_planar_relocalization,
    inverse_pose,
    iter_xyz_points,
    nearest_stamped_pose,
    normalize_obstacle_height_range,
    obstacle_height_ranges_match,
    quaternion_from_rpy,
    read_pcd_xyz,
    source_age_is_valid,
    transform_points,
    write_pcd_xyz_atomic,
    yaw_from_quaternion,
)
from g1_fast_livo2.runtime_core import controlled_stop_succeeded  # noqa: E402


class FastLivo2FrameAdapterTest(unittest.TestCase):
    def test_raw_odom_discontinuity_recovers_after_three_healthy_samples(self) -> None:
        monitor = OdomHealthMonitor()
        identity = Quaternion(0.0, 0.0, 0.0, 1.0)

        self.assertTrue(monitor.observe(1_000_000_000, Pose3(0.0, 0.0, 0.0, identity)))
        self.assertFalse(
            monitor.observe(1_100_000_000, Pose3(1000.0, 0.0, 0.0, identity))
        )
        self.assertEqual(monitor.reason, "raw_odom_discontinuity")
        self.assertFalse(monitor.observe(1_100_000_000, Pose3(0.1, 0.0, 0.0, identity)))
        self.assertFalse(monitor.observe(1_200_000_000, Pose3(0.2, 0.0, 0.0, identity)))
        self.assertTrue(monitor.observe(1_300_000_000, Pose3(0.3, 0.0, 0.0, identity)))
        self.assertEqual(monitor.state, "ready")

    def test_raw_odom_initial_pose_and_rotation_rate_fail_closed(self) -> None:
        identity = Quaternion(0.0, 0.0, 0.0, 1.0)
        monitor = OdomHealthMonitor()
        self.assertFalse(monitor.observe(1_000_000_000, Pose3(2.1, 0.0, 0.0, identity)))

        monitor.reset()
        self.assertTrue(monitor.observe(1_000_000_000, Pose3(0.0, 0.0, 0.0, identity)))
        self.assertFalse(
            monitor.observe(
                1_010_000_000,
                Pose3(0.0, 0.0, 0.0, quaternion_from_rpy(0.0, 0.0, math.pi)),
            )
        )

        monitor.reset()
        self.assertFalse(
            monitor.observe(
                1_000_000_000,
                Pose3(0.0, 0.0, 0.0, Quaternion(math.nan, 0.0, 0.0, 1.0)),
            )
        )

    def test_persisted_obstacle_height_range_is_strictly_validated(self) -> None:
        self.assertEqual(
            normalize_obstacle_height_range([-0.30, 0.30]),
            (-0.30, 0.30),
        )
        self.assertTrue(
            obstacle_height_ranges_match(
                (-0.30, 0.30),
                (-0.3000001, 0.3000001),
            )
        )
        self.assertFalse(
            obstacle_height_ranges_match((-0.30, 0.30), (-0.20, 0.30))
        )
        for invalid in (None, [-0.30], [True, 0.30], ["-0.30", 0.30], [0.3, -0.3]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_obstacle_height_range(invalid)

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

    def test_extreme_finite_coordinates_fail_closed(self) -> None:
        identity = Quaternion(0.0, 0.0, 0.0, 1.0)
        huge = 1e300
        float64_fields = [
            {"name": "x", "offset": 0, "datatype": 8, "count": 1},
            {"name": "y", "offset": 8, "datatype": 8, "count": 1},
            {"name": "z", "offset": 16, "datatype": 8, "count": 1},
        ]

        with self.assertRaisesRegex(InvalidFastLivo2Frame, "float32 range"):
            list(
                iter_xyz_points(
                    fields=float64_fields,
                    data=struct.pack("<ddd", huge, 0.0, 0.0),
                    point_step=24,
                    width=1,
                    height=1,
                    is_bigendian=False,
                )
            )
        with self.assertRaises(InvalidFastLivo2Frame):
            VoxelMap(0.10).add(((huge, 0.0, 0.0),))
        with self.assertRaises(InvalidFastLivo2Frame):
            TemporalOccupancyMap(0.10).observe_scan(
                sensor_origin=(0.0, 0.0, 0.0),
                points=((huge, 0.0, 0.0),),
                now_monotonic=0.0,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        with self.assertRaises(InvalidFastLivo2Frame):
            tuple(
                transform_points(
                    Pose3(0.0, 0.0, 0.0, identity),
                    ((huge, 0.0, 0.0),),
                )
            )
        with self.assertRaisesRegex(InvalidFastLivo2Frame, "composed pose"):
            compose_pose(
                Pose3(3e38, 0.0, 0.0, identity),
                Pose3(3e38, 0.0, 0.0, identity),
            )

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
            Pose3(0.5, -0.25, 0.0, quaternion_from_rpy(0.0, 0.0, 1.0)),
            obstacle_min_height_m=-0.8,
            obstacle_max_height_m=0.4,
        )
        robot_x, robot_y, robot_yaw, flags, point_count = struct.unpack_from(
            "<fffBI", frame
        )
        self.assertAlmostEqual(robot_x, 0.5)
        self.assertAlmostEqual(robot_y, -0.25)
        self.assertAlmostEqual(robot_yaw, 1.0)
        self.assertEqual(flags, 0x03)
        self.assertEqual(point_count, 1)
        metadata_offset = 17 + point_count * 12
        magic, minimum, maximum = struct.unpack_from("<8sff", frame, metadata_offset)
        self.assertEqual(magic, b"MVFILT2\0")
        self.assertAlmostEqual(minimum, -0.8)
        self.assertAlmostEqual(maximum, 0.4)
        self.assertEqual(len(frame), metadata_offset + struct.calcsize("<8sff"))

        full_map = VoxelMap(0.10)
        full_map.add((index * 0.20, 0.0, 0.0) for index in range(80_001))
        self.assertEqual(full_map.point_count, 80_001)

        bounded_map = VoxelMap(0.10)
        bounded_map.add(
            [(index * 0.20, 0.0, -1.20) for index in range(100)]
            + [(index * 0.20, 1.0, 0.0) for index in range(100)]
            + [(index * 0.20, 2.0, 1.20) for index in range(100)]
        )
        bounded_frame = bounded_map.encode(
            Pose3(0.0, 0.0, 0.0, quaternion_from_rpy(0.0, 0.0, 0.0)),
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
            max_points=20,
        )
        bounded_count = struct.unpack_from("<I", bounded_frame, 13)[0]
        bounded_points = [
            struct.unpack_from("<fff", bounded_frame, 17 + index * 12)
            for index in range(bounded_count)
        ]
        self.assertEqual(bounded_count, 20)
        self.assertEqual(sum(point[2] < -0.30 for point in bounded_points), 7)
        self.assertEqual(
            sum(-0.30 <= point[2] <= 0.30 for point in bounded_points),
            11,
        )
        self.assertEqual(sum(point[2] > 0.30 for point in bounded_points), 2)
        with self.assertRaises(ValueError):
            bounded_map.encode(
                Pose3(0.0, 0.0, 0.0, quaternion_from_rpy(0.0, 0.0, 0.0)),
                max_points=0,
            )

        direct_frame = encode_map_view_points(
            (
                point
                for point in (
                    [(index * 0.20, 0.0, -1.20) for index in range(100)]
                    + [(index * 0.20, 1.0, 0.0) for index in range(100)]
                    + [(index * 0.20, 2.0, 1.20) for index in range(100)]
                )
            ),
            Pose3(0.0, 0.0, 0.0, quaternion_from_rpy(0.0, 0.0, 0.0)),
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
            max_points=20,
        )
        direct_count = struct.unpack_from("<I", direct_frame, 13)[0]
        direct_points = [
            struct.unpack_from("<fff", direct_frame, 17 + index * 12)
            for index in range(direct_count)
        ]
        self.assertEqual(direct_count, 20)
        self.assertEqual(sum(point[2] < -0.30 for point in direct_points), 7)
        self.assertEqual(
            sum(-0.30 <= point[2] <= 0.30 for point in direct_points),
            11,
        )
        self.assertEqual(sum(point[2] > 0.30 for point in direct_points), 2)

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
        with self.assertRaisesRegex(InvalidFastLivo2Frame, "point safety limit"):
            list(
                iter_xyz_points(
                    fields=fields,
                    data=data,
                    point_step=12,
                    width=2,
                    height=1,
                    is_bigendian=False,
                    max_points=1,
                )
            )
        with self.assertRaisesRegex(InvalidFastLivo2Frame, "byte safety limit"):
            list(
                iter_xyz_points(
                    fields=fields,
                    data=data,
                    point_step=12,
                    width=2,
                    height=1,
                    is_bigendian=False,
                    max_data_bytes=len(data) - 1,
                )
            )

    def test_obstacle_projection_excludes_floor_and_ceiling(self) -> None:
        voxel_map = VoxelMap(0.10)
        voxel_map.add(
            [
                (1.01, 2.01, -1.30),
                (1.01, 2.01, -0.50),
                (1.01, 2.01, -0.35),
                (1.01, 2.01, -0.30),
                (1.04, 2.04, 0.40),
                (2.01, 3.01, 1.70),
                (3.01, 4.01, 0.20),
                (4.01, 5.01, -1.20),
            ]
        )

        projected = voxel_map.project_xy(min_z=-0.30, max_z=0.30)

        self.assertEqual(len(projected), 2)
        self.assertEqual(
            [(round(x, 2), round(y, 2), z) for x, y, z in projected],
            [(1.05, 2.05, 0.0), (3.05, 4.05, 0.0)],
        )
        with self.assertRaises(ValueError):
            voxel_map.project_xy(min_z=1.0, max_z=1.0)

    def test_direct_accumulation_keeps_first_observation_without_motion_gate(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        first = (1.05, 0.05, 0.0)
        static_map.observe_scan(
            sensor_origin=(0.05, 0.05, 0.0),
            points=[first],
            now_monotonic=0.0,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )

        self.assertEqual(static_map.confirmed_points, (first,))

        static_map.observe_scan(
            sensor_origin=(0.05, 0.05, 0.0),
            points=[(2.05, 0.05, 0.0)],
            now_monotonic=0.1,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )
        self.assertIn(first, static_map.confirmed_points)
        self.assertEqual(static_map.point_count, 2)

    def test_out_of_navigation_height_points_never_enter_static_map(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        for frame in range(3):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(1.05, 0.05, 1.0)],
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertEqual(static_map.point_count, 0)

    def test_cloud_uses_only_a_source_timestamp_matched_pose(self) -> None:
        first = Pose3(1.0, 0.0, 0.0, Quaternion(0.0, 0.0, 0.0, 1.0))
        second = Pose3(2.0, 0.0, 0.0, Quaternion(0.0, 0.0, 0.0, 1.0))
        history = ((1_000_000_000, first), (1_100_000_000, second))

        self.assertEqual(
            nearest_stamped_pose(
                history,
                1_045_000_000,
                tolerance_ns=50_000_000,
            ),
            first,
        )
        self.assertIsNone(
            nearest_stamped_pose(
                history,
                1_049_999_999,
                tolerance_ns=40_000_000,
            )
        )

        self.assertIsNone(
            bracketed_stamped_pose(
                history[:1],
                1_020_000_000,
                tolerance_ns=50_000_000,
            )
        )
        self.assertEqual(
            bracketed_stamped_pose(
                history,
                1_020_000_000,
                tolerance_ns=50_000_000,
            ),
            first,
        )
        self.assertIsNone(
            bracketed_stamped_pose(
                history,
                1_045_000_000,
                tolerance_ns=40_000_000,
            )
        )

    def test_large_motion_keeps_sparse_map_and_bounded_grid_window(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            grid_margin_m=0.10,
            max_grid_dimension_cells=20,
            max_grid_cells=400,
        )
        static_map.observe_scan(
            sensor_origin=(0.05, 0.05, 0.0),
            points=[(50.0, 50.0, 0.0), (1.05, 0.05, 0.0)],
            now_monotonic=0.0,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )
        self.assertEqual(static_map.point_count, 1)

        static_map.observe_scan(
            sensor_origin=(2.05, 0.05, 0.0),
            points=[(3.05, 0.05, 0.0)],
            now_monotonic=0.1,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )
        self.assertEqual(static_map.point_count, 2)
        snapshot = static_map.occupancy_snapshot(
            center_x=625.0,
            center_y=-49.0,
            min_z=-0.30,
            max_z=0.30,
        )
        self.assertEqual((snapshot.width, snapshot.height), (3, 3))
        self.assertEqual(len(snapshot.data), 9)

    def test_large_loaded_map_uses_bounded_grid_window(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            grid_margin_m=0.10,
            max_grid_dimension_cells=20,
            max_grid_cells=400,
        )
        large_map = ((0.55, 0.05, 0.0), (500.05, 500.05, 0.0))
        static_map.load_confirmed(large_map)
        self.assertEqual(set(static_map.confirmed_points), set(large_map))
        self.assertEqual(static_map.validate_confirmed(large_map), 2)
        snapshot = static_map.occupancy_snapshot(
            center_x=0.55,
            center_y=0.05,
            min_z=-0.30,
            max_z=0.30,
        )
        self.assertEqual((snapshot.width, snapshot.height), (3, 3))
        self.assertEqual(snapshot.occupied_cells, 1)

        cleared = static_map.cleared_snapshot()
        self.assertEqual(cleared.occupied_cells, 0)
        self.assertEqual(cleared.free_cells, cleared.width * cleared.height)
        self.assertEqual(set(cleared.data), {0})

    def test_static_evidence_limit_rejects_whole_frame_atomically(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            max_evidence_points=40,
        )
        baseline = [
            (0.5 + index * 0.20, 0.0, 0.0)
            for index in range(39)
        ]
        static_map.observe_scan(
            sensor_origin=(0.05, 0.05, 0.0),
            points=baseline,
            now_monotonic=0.0,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )
        before = dict(static_map._evidence)

        with self.assertRaisesRegex(ValueError, "evidence exceeds 40"):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(1.0, 1.0, 0.0), (1.0, 2.0, 0.0)],
                now_monotonic=0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertEqual(static_map._evidence, before)
        self.assertEqual(static_map._last_observation_monotonic, 0.0)

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

            oversized_count = Path(directory) / "oversized-count.pcd"
            oversized_count.write_text(
                "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                "WIDTH 200001\nHEIGHT 1\nPOINTS 200001\nDATA ascii\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                InvalidFastLivo2Frame,
                "declared point count exceeds 200000",
            ):
                read_pcd_xyz(
                    oversized_count,
                    max_declared_points=200000,
                )

            with self.assertRaisesRegex(
                InvalidFastLivo2Frame,
                "byte safety limit",
            ):
                read_pcd_xyz(ascii_path, max_file_bytes=1)

            extra_row = Path(directory) / "extra-row.pcd"
            extra_row.write_text(
                header.replace("WIDTH 2", "WIDTH 1")
                .replace("POINTS 2", "POINTS 1")
                + "DATA ascii\n1 2 0 3\n4 5 0.5 6\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                InvalidFastLivo2Frame,
                "more rows than declared",
            ):
                read_pcd_xyz(extra_row)

            missing_row = Path(directory) / "missing-row.pcd"
            missing_row.write_text(
                header + "DATA ascii\n1 2 0 3\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                InvalidFastLivo2Frame,
                "row count does not match",
            ):
                read_pcd_xyz(missing_row)

    def test_pcd_reader_bounds_text_records_and_checks_deadline(self) -> None:
        class TrackingStream(io.BytesIO):
            readline_sizes: list[int] = []

            def readline(self, size=-1):
                self.readline_sizes.append(size)
                return super().readline(size)

        oversized_header = b"X" * 65_537
        header_stream = TrackingStream(oversized_header)
        with mock.patch.object(
            Path,
            "open",
            return_value=header_stream,
        ), mock.patch.object(
            Path,
            "stat",
            return_value=SimpleNamespace(st_size=len(oversized_header)),
        ):
            with self.assertRaisesRegex(InvalidFastLivo2Frame, "header exceeds"):
                read_pcd_xyz("oversized-header.pcd")
        self.assertTrue(header_stream.readline_sizes)
        self.assertLessEqual(max(header_stream.readline_sizes), 65_537)
        self.assertNotIn(-1, header_stream.readline_sizes)

        header = (
            "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            "WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extra_token = root / "extra-token.pcd"
            extra_token.write_text(header + "1 2 3 4\n", encoding="ascii")
            with self.assertRaisesRegex(InvalidFastLivo2Frame, "token count"):
                read_pcd_xyz(extra_token)

            oversized_record = root / "oversized-record.pcd"
            oversized_record.write_bytes(
                header.encode("ascii") + b"1 2 3 " + b"0" * 65_537 + b"\n"
            )
            with self.assertRaisesRegex(InvalidFastLivo2Frame, "record exceeds"):
                read_pcd_xyz(oversized_record)

            deadline = root / "deadline.pcd"
            deadline.write_text(header + "1 2 3\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "deadline_monotonic"):
                read_pcd_xyz(deadline, deadline_monotonic=math.nan)
            with mock.patch(
                "g1_fast_livo2.frame_adapter_core.time.monotonic",
                side_effect=[0.0] * 9 + [2.0],
            ):
                with self.assertRaisesRegex(TimeoutError, "ASCII parsing"):
                    read_pcd_xyz(deadline, deadline_monotonic=1.0)

    def test_pcd_reader_rejects_extreme_finite_float64_point(self) -> None:
        header = (
            "FIELDS x y z\nSIZE 8 8 8\nTYPE F F F\nCOUNT 1 1 1\n"
            "WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "huge.pcd"
            path.write_text(header + "1e300 0 0\n", encoding="ascii")
            with self.assertRaisesRegex(InvalidFastLivo2Frame, "float32 range"):
                read_pcd_xyz(path)

    def test_static_pcd_writer_is_atomic_and_round_trips(self) -> None:
        points = ((1.0, 2.0, 0.0), (3.0, 4.0, 0.5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static" / "office.pcd"

            count = write_pcd_xyz_atomic(path, points)

            self.assertEqual(count, 2)
            self.assertEqual(read_pcd_xyz(path), points)
            self.assertFalse(path.with_name("office.pcd.tmp").exists())

    def test_static_pcd_writer_preserves_retryable_io_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static" / "office.pcd"
            with mock.patch.object(
                Path,
                "open",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(
                    FastLivo2PersistenceError,
                    "disk full",
                ):
                    write_pcd_xyz_atomic(path, ((1.0, 2.0, 0.0),))

            self.assertFalse(path.exists())
            self.assertFalse(path.with_name("office.pcd.tmp").exists())

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

    def test_binary_pcd_reader_rejects_extra_records(self) -> None:
        header = (
            "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            "WIDTH 40\nHEIGHT 1\nPOINTS 40\nDATA binary\n"
        ).encode("ascii")
        payload = b"".join(
            struct.pack("<fff", float(index), 0.0, 0.0)
            for index in range(41)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extra.pcd"
            path.write_bytes(header + payload)

            with self.assertRaisesRegex(
                InvalidFastLivo2Frame,
                "more records than declared",
            ):
                read_pcd_xyz(path, max_declared_points=200_000)

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

    def test_planar_relocalization_does_not_depend_on_initial_height(self) -> None:
        reference = [
            (index * 0.10, 0.0, 0.0)
            for index in range(60)
        ] + [
            (0.0, index * 0.10, 0.0)
            for index in range(60)
        ]
        expected = Pose3(
            2.0,
            -1.0,
            0.0,
            quaternion_from_rpy(0.0, 0.0, 0.30),
        )
        session_points = tuple(transform_points(inverse_pose(expected), reference))

        result = estimate_planar_relocalization(
            reference_points=reference,
            session_points=session_points,
            session_base_pose=Pose3(
                0.0,
                0.0,
                -3.0,
                Quaternion(0.0, 0.0, 0.0, 1.0),
            ),
            initial_map_base_pose=Pose3(
                2.1,
                -1.1,
                5.0,
                quaternion_from_rpy(0.0, 0.0, 0.35),
            ),
            search_xy_m=0.5,
            search_yaw_rad=0.25,
            min_z=-0.5,
            max_z=0.5,
            min_match_ratio=0.5,
        )

        self.assertAlmostEqual(result.map_base_pose.x, expected.x, delta=0.15)
        self.assertAlmostEqual(result.map_base_pose.y, expected.y, delta=0.15)
        self.assertEqual(result.map_base_pose.z, 5.0)

    def test_relocalization_rejection_exposes_best_candidate(self) -> None:
        points = [
            (index * 0.10, 0.0, 0.0)
            for index in range(60)
        ] + [
            (0.0, index * 0.10, 0.0)
            for index in range(60)
        ]

        with self.assertRaises(RelocalizationRejected) as caught:
            estimate_planar_relocalization(
                reference_points=points,
                session_points=points,
                session_base_pose=Pose3(
                    0.0,
                    0.0,
                    0.0,
                    Quaternion(0.0, 0.0, 0.0, 1.0),
                ),
                initial_map_base_pose=Pose3(
                    5.0,
                    5.0,
                    0.0,
                    Quaternion(0.0, 0.0, 0.0, 1.0),
                ),
                search_xy_m=0.5,
                search_yaw_rad=0.2,
                min_z=-0.5,
                max_z=0.5,
            )

        self.assertEqual(caught.exception.reason, "match_ratio_below_threshold")
        self.assertLess(caught.exception.result.match_ratio, 0.35)
        self.assertAlmostEqual(caught.exception.result.map_base_pose.x, 5.0)

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

    def test_relocalization_rejects_best_candidate_on_search_boundary(self) -> None:
        session = []
        for index in range(40):
            session.append((index * 0.15, 0.0, 0.0))
            session.append((0.0, index * 0.15, 0.0))
        reference = tuple((x + 0.5, y, z) for x, y, z in session)

        with self.assertRaisesRegex(InvalidFastLivo2Frame, "search boundary"):
            estimate_planar_relocalization(
                reference_points=reference,
                session_points=session,
                session_base_pose=Pose3(
                    0.0,
                    0.0,
                    0.0,
                    Quaternion(0.0, 0.0, 0.0, 1.0),
                ),
                initial_map_base_pose=Pose3(
                    0.0,
                    0.0,
                    0.0,
                    Quaternion(0.0, 0.0, 0.0, 1.0),
                ),
                search_xy_m=0.5,
                search_yaw_rad=0.2,
                min_z=-0.5,
                max_z=0.5,
                match_voxel_m=0.1,
                min_match_ratio=0.5,
            )


if __name__ == "__main__":
    unittest.main()
