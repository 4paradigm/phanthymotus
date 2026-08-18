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
    Pose3,
    Quaternion,
    TemporalOccupancyMap,
    VoxelMap,
    canonical_base_pose,
    compose_pose,
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

    def test_single_frame_person_never_enters_static_and_expires(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            confirmation_frames=3,
            candidate_ttl_sec=1.0,
            clear_miss_frames=2,
        )
        static_map.observe_scan(
            sensor_origin=(0.05, 0.05, 0.0),
            points=[(1.05, 0.05, 0.0)],
            now_monotonic=0.0,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )

        self.assertEqual(static_map.point_count, 0)
        self.assertEqual(static_map.candidate_count, 1)
        self.assertEqual(static_map.candidate_points, ((1.05, 0.05, 0.0),))
        static_map.expire(now_monotonic=1.01)
        self.assertEqual(static_map.candidate_count, 0)
        self.assertEqual(static_map.candidate_points, ())
        self.assertEqual(static_map.confirmed_points, ())

    def test_out_of_navigation_height_points_never_enter_static_map(self) -> None:
        static_map = TemporalOccupancyMap(0.10, confirmation_frames=3)
        for frame in range(3):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(1.05, 0.05, 1.0)],
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertEqual(static_map.point_count, 0)
        self.assertEqual(static_map.candidate_count, 0)

    def test_repeated_wall_frames_promote_one_static_voxel(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            confirmation_frames=3,
            candidate_ttl_sec=1.0,
            clear_miss_frames=2,
            component_motion_window_sec=0.1,
            component_history_sec=0.2,
            component_motion_distance_m=0.1,
            component_motion_speed_mps=0.5,
            component_stationary_sec=0.3,
        )
        for frame, x in enumerate((2.01, 2.04, 2.08, 2.08, 2.08)):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(x, 0.05, 0.0)] * 10,
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
            self.assertEqual(static_map.point_count, 1 if frame == 4 else 0)

        static_map.expire(now_monotonic=10.0)
        self.assertEqual(static_map.point_count, 1)
        self.assertEqual(len(static_map.confirmed_points), 1)

    def test_nonconsecutive_hits_do_not_promote_static_voxel(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            confirmation_frames=3,
            candidate_ttl_sec=1.0,
            clear_miss_frames=2,
        )
        point = (2.05, 0.05, 0.0)
        for frame, points in enumerate(([point], [], [point], [point])):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=points,
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertEqual(static_map.point_count, 0)
        self.assertEqual(static_map.candidate_count, 1)

    def test_free_ray_clears_static_only_after_bounded_misses(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            confirmation_frames=3,
            candidate_ttl_sec=1.0,
            clear_miss_frames=2,
            angular_bin_deg=1.0,
            component_motion_window_sec=0.1,
            component_history_sec=0.2,
            component_motion_distance_m=0.1,
            component_motion_speed_mps=0.5,
            component_stationary_sec=0.3,
        )
        origin = (0.05, 0.05, 0.0)
        person = (1.05, 0.05, 0.0)
        wall = (2.05, 0.05, 0.0)
        for frame in range(5):
            static_map.observe_scan(
                sensor_origin=origin,
                points=[person],
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        self.assertEqual(static_map.point_count, 1)

        static_map.observe_scan(
            sensor_origin=origin,
            points=[wall],
            now_monotonic=0.6,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )
        self.assertEqual(static_map.point_count, 1)
        static_map.observe_scan(
            sensor_origin=origin,
            points=[wall],
            now_monotonic=0.7,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )
        self.assertEqual(static_map.point_count, 0)

        snapshot = static_map.occupancy_snapshot(
            center_x=0.05,
            center_y=0.05,
            min_z=-0.30,
            max_z=0.30,
        )
        person_ix = math.floor(person[0] / snapshot.resolution)
        person_iy = math.floor(person[1] / snapshot.resolution)
        origin_ix = math.floor(snapshot.origin_x / snapshot.resolution)
        origin_iy = math.floor(snapshot.origin_y / snapshot.resolution)
        offset = (person_iy - origin_iy) * snapshot.width + person_ix - origin_ix
        self.assertEqual(snapshot.data[offset], 0)

    def test_loaded_static_map_is_seeded_and_rejects_live_mutation(self) -> None:
        static_map = TemporalOccupancyMap(0.10, confirmation_frames=3)
        saved = ((1.05, 0.05, 0.0), (2.05, 0.05, 0.0))

        static_map.load_confirmed(saved)

        self.assertEqual(static_map.point_count, 2)
        self.assertEqual(static_map.candidate_count, 0)
        self.assertEqual(set(static_map.confirmed_points), set(saved))
        with self.assertRaisesRegex(ValueError, "immutable"):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(3.05, 0.05, 0.0)],
                now_monotonic=0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        self.assertEqual(set(static_map.confirmed_points), set(saved))

    def test_default_requires_hits_and_component_observation_window(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        point = (1.05, 0.05, 0.0)
        for frame in range(8):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[point],
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        self.assertEqual(static_map.point_count, 0)

        static_map.observe_scan(
            sensor_origin=(0.05, 0.05, 0.0),
            points=[point],
            now_monotonic=0.8,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )
        self.assertEqual(static_map.point_count, 0)

        for frame in range(9, 54):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[point],
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        self.assertEqual(static_map.point_count, 1)

    def test_dense_moving_components_never_enter_static_map(self) -> None:
        for speed_mps in (0.03, 0.035, 0.05, 0.10, 0.20, 0.40, 0.80):
            with self.subTest(speed_mps=speed_mps):
                static_map = TemporalOccupancyMap(0.10)
                saw_dynamic = False
                for frame in range(100):
                    offset = speed_mps * frame * 0.1
                    person = [
                        (2.0 + x_offset, -0.20 + y_offset + offset, 0.0)
                        for x_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                        for y_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                    ]
                    static_map.observe_scan(
                        sensor_origin=(0.05, 0.05, 0.0),
                        points=person,
                        now_monotonic=frame * 0.1,
                        obstacle_min_height_m=-0.30,
                        obstacle_max_height_m=0.30,
                    )
                    self.assertEqual(static_map.point_count, 0)
                    saw_dynamic = saw_dynamic or (
                        static_map.dynamic_track_count > 0
                    )

                self.assertTrue(saw_dynamic)

    def test_sparse_moving_components_never_enter_static_map(self) -> None:
        for point_count in (1, 2, 3, 4):
            with self.subTest(point_count=point_count):
                static_map = TemporalOccupancyMap(0.10)
                saw_dynamic = False
                for frame in range(100):
                    offset = 0.05 * frame * 0.1
                    points = [
                        (2.0 + index * 0.08, -0.5 + offset, 0.0)
                        for index in range(point_count)
                    ]
                    static_map.observe_scan(
                        sensor_origin=(0.05, 0.05, 0.0),
                        points=points,
                        now_monotonic=frame * 0.1,
                        obstacle_min_height_m=-0.30,
                        obstacle_max_height_m=0.30,
                    )
                    self.assertEqual(static_map.point_count, 0)
                    saw_dynamic = saw_dynamic or (
                        static_map.dynamic_track_count > 0
                    )

                self.assertTrue(saw_dynamic)

    def test_diagonal_minimum_speed_never_enters_static_map(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        axis_speed = 0.03 / math.sqrt(2.0)
        saw_dynamic = False
        for frame in range(80):
            elapsed = frame * 0.1
            point = (
                2.001 + axis_speed * elapsed,
                2.001 + axis_speed * elapsed,
                0.0,
            )
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[point],
                now_monotonic=elapsed,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
            self.assertEqual(static_map.point_count, 0)
            saw_dynamic = saw_dynamic or static_map.dynamic_track_count > 0

        self.assertTrue(saw_dynamic)

    def test_component_history_has_a_hard_sample_limit(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        frequency_hz = 200
        for frame in range(frequency_hz * 6 + 1):
            elapsed = frame / frequency_hz
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(2.0, -0.5, 0.0)],
                now_monotonic=elapsed,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertTrue(static_map._component_tracks)
        self.assertTrue(
            all(
                len(track.samples) <= static_map._component_track_sample_limit
                for track in static_map._component_tracks.values()
            )
        )
        track = next(iter(static_map._component_tracks.values()))
        self.assertGreaterEqual(
            track.samples[-1].stamp - track.samples[0].stamp,
            static_map._component_observation_window - 0.10,
        )
        self.assertLessEqual(
            static_map.component_history_unit_count,
            static_map.component_history_unit_limit,
        )

    def test_dense_static_history_uses_bounded_packed_cell_summaries(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        points = [
            (-3.95 + column * 0.10, -3.95 + row * 0.10, 0.0)
            for column in range(80)
            for row in range(80)
        ]
        for frame in range(106):
            static_map.observe_scan(
                sensor_origin=(0.0, 0.0, 0.0),
                points=points,
                now_monotonic=frame / 20.0,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertLessEqual(
            static_map.component_history_unit_count,
            700_000,
        )
        self.assertLessEqual(
            static_map.component_history_unit_count,
            static_map.component_history_unit_limit,
        )
        self.assertTrue(
            all(
                isinstance(sample.payload, bytes)
                for track in static_map._component_tracks.values()
                for sample in track.samples
            )
        )

    def test_component_history_pressure_fails_closed_without_track_growth(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            max_evidence_points=1_000,
            max_component_history_units=128,
        )
        isolated = [
            (1.0 + (index % 10) * 0.30, -1.5 + (index // 10) * 0.30, 0.0)
            for index in range(80)
        ]
        for frame in range(120):
            static_map.observe_scan(
                sensor_origin=(0.0, 0.0, 0.0),
                points=isolated,
                now_monotonic=frame / 20.0,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertLessEqual(static_map.component_history_unit_count, 128)
        self.assertLessEqual(len(static_map._component_tracks), 1)
        self.assertEqual(static_map.point_count, 0)

    def test_component_history_configuration_rejects_unbounded_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation window exceeds"):
            TemporalOccupancyMap(0.10, component_motion_speed_mps=1e-9)
        with self.assertRaisesRegex(ValueError, "history_units must be within"):
            TemporalOccupancyMap(
                0.10,
                max_component_history_units=1_000_001,
            )

    def test_high_rate_moving_components_remain_dynamic_and_non_static(self) -> None:
        for frequency_hz in (20, 200, 300):
            for speed_mps in (0.03, 0.035):
                with self.subTest(frequency_hz=frequency_hz, speed_mps=speed_mps):
                    static_map = TemporalOccupancyMap(0.10)
                    saw_dynamic = False
                    for frame in range(frequency_hz * 6 + 1):
                        elapsed = frame / frequency_hz
                        offset = speed_mps * elapsed
                        person = [
                            (2.0 + x_offset, -0.20 + y_offset + offset, 0.0)
                            for x_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                            for y_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                        ]
                        static_map.observe_scan(
                            sensor_origin=(0.05, 0.05, 0.0),
                            points=person,
                            now_monotonic=elapsed,
                            obstacle_min_height_m=-0.30,
                            obstacle_max_height_m=0.30,
                        )
                        self.assertEqual(static_map.point_count, 0)
                        saw_dynamic = saw_dynamic or (
                            static_map.dynamic_track_count > 0
                        )
                    self.assertTrue(saw_dynamic)

    def test_high_rate_person_connected_to_wall_does_not_erase_or_ghost(self) -> None:
        frequency_hz = 200
        wall = [(2.0, -1.2 + index * 0.08, 0.0) for index in range(31)]
        baseline = TemporalOccupancyMap(0.10)
        for frame in range(frequency_hz * 6 + 1):
            baseline.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=wall,
                now_monotonic=frame / frequency_hz,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        baseline_wall = {point[:2] for point in baseline.confirmed_points}

        static_map = TemporalOccupancyMap(0.10)
        for frame in range(frequency_hz * 6 + 1):
            elapsed = frame / frequency_hz
            offset = 0.035 * elapsed
            person = [
                (2.0 + x_offset, -0.20 + y_offset + offset, 0.0)
                for x_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                for y_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
            ]
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[*wall, *person],
                now_monotonic=elapsed,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        confirmed_wall = {
            point[:2] for point in static_map.confirmed_points if point[0] <= 2.05
        }
        self.assertEqual(confirmed_wall, baseline_wall)
        self.assertFalse(
            [point for point in static_map.confirmed_points if point[0] > 2.15]
        )

    def test_equal_timestamp_replaces_sample_without_history_growth(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        for offset in range(100):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(2.0, -0.5 + offset * 0.0001, 0.0)],
                now_monotonic=1.0,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        track = next(iter(static_map._component_tracks.values()))
        self.assertEqual(len(track.samples), 1)
        with self.assertRaisesRegex(ValueError, "must not move backwards"):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(2.0, -0.5, 0.0)],
                now_monotonic=0.9,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

    def test_confirmed_compact_object_is_revoked_when_it_starts_moving(self) -> None:
        static_map = TemporalOccupancyMap(0.10)

        def person(offset: float):
            return [
                (2.0 + x_offset, -0.20 + y_offset + offset, 0.0)
                for x_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                for y_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
            ]

        for frame in range(60):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=person(0.0),
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        self.assertGreater(static_map.point_count, 0)

        for frame in range(60, 100):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=person(0.20 * (frame - 59) * 0.1),
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertEqual(static_map.point_count, 0)
        self.assertGreaterEqual(static_map.dynamic_track_count, 1)

    def test_static_wall_confirms_while_separate_person_keeps_moving(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        wall = [
            (5.0, 2.0 + index * 0.08, 0.0)
            for index in range(26)
        ]
        for frame in range(100):
            offset = 0.20 * frame * 0.1
            person = [
                (2.0 + x_offset, -2.0 + y_offset + offset, 0.0)
                for x_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                for y_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
            ]
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[*wall, *person],
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertGreater(static_map.point_count, 0)
        self.assertTrue(
            all(point[0] > 4.5 for point in static_map.confirmed_points)
        )
        self.assertGreaterEqual(static_map.dynamic_track_count, 1)

    def test_person_connected_to_wall_is_excluded_without_erasing_wall(self) -> None:
        wall = [(2.0, -1.2 + index * 0.08, 0.0) for index in range(31)]
        baseline = TemporalOccupancyMap(0.10)
        for frame in range(100):
            baseline.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=wall,
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        baseline_wall = {
            point[:2] for point in baseline.confirmed_points
        }

        for speed_mps in (0.03, 0.035, 0.05, 0.10, 0.20, 0.40, 0.80):
            with self.subTest(speed_mps=speed_mps):
                static_map = TemporalOccupancyMap(0.10)
                for frame in range(100):
                    offset = speed_mps * frame * 0.1
                    person = [
                        (2.0 + x_offset, -0.20 + y_offset + offset, 0.0)
                        for x_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                        for y_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                    ]
                    static_map.observe_scan(
                        sensor_origin=(0.05, 0.05, 0.0),
                        points=[*wall, *person],
                        now_monotonic=frame * 0.1,
                        obstacle_min_height_m=-0.30,
                        obstacle_max_height_m=0.30,
                    )

                confirmed_wall = {
                    point[:2]
                    for point in static_map.confirmed_points
                    if point[0] <= 2.05
                }
                self.assertEqual(confirmed_wall, baseline_wall)
                self.assertFalse(
                    [
                        point
                        for point in static_map.confirmed_points
                        if point[0] > 2.15
                    ]
                )

    def test_compact_static_component_confirms_after_motion_observation(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        pillar = [
            (2.0 + x_offset, -0.16 + y_offset, 0.0)
            for x_offset in (0.0, 0.08, 0.16, 0.24, 0.32)
            for y_offset in (0.0, 0.08, 0.16, 0.24, 0.32)
        ]
        for frame in range(60):
            jitter = 0.004 if frame % 2 else -0.004
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=[(x + jitter, y - jitter, z) for x, y, z in pillar],
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertGreater(static_map.point_count, 0)
        self.assertEqual(static_map.dynamic_track_count, 0)

    def test_moving_component_can_become_static_after_stationary_cooldown(self) -> None:
        static_map = TemporalOccupancyMap(0.10)

        def person(offset: float):
            return [
                (2.0 + x_offset, -0.20 + y_offset + offset, 0.0)
                for x_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
                for y_offset in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
            ]

        for frame in range(40):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=person(0.20 * frame * 0.1),
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        self.assertEqual(static_map.point_count, 0)

        stopped_offset = 0.20 * 39 * 0.1
        for frame in range(40, 180):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=person(stopped_offset),
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertGreater(static_map.point_count, 0)
        self.assertEqual(static_map.dynamic_track_count, 0)

    def test_long_wall_visibility_changes_do_not_trigger_component_gate(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        for frame in range(60):
            start = 0 if frame % 2 == 0 else 4
            wall = [
                (2.0, -1.0 + index * 0.08, 0.0)
                for index in range(start, start + 20)
            ]
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=wall,
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertGreater(static_map.point_count, 0)
        self.assertEqual(static_map.dynamic_track_count, 0)

    def test_short_wall_visibility_change_does_not_purge_shared_cells(self) -> None:
        static_map = TemporalOccupancyMap(0.10)
        wall = [(2.0, -0.40 + index * 0.10, 0.0) for index in range(9)]
        for frame in range(60):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=wall,
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        confirmed_before = set(static_map.confirmed_points)

        for frame in range(60, 66):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=wall[:5],
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertEqual(static_map.dynamic_track_count, 0)
        self.assertTrue(
            set(static_map.confirmed_points).intersection(confirmed_before)
        )

    def test_same_wall_voxels_with_shifted_samples_are_not_motion(self) -> None:
        static_map = TemporalOccupancyMap(0.10)

        def wall(offset: float):
            return [
                (2.01, -0.49 + index * 0.10 + offset, 0.0)
                for index in range(8)
            ]

        for frame in range(60):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=wall(0.0),
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )
        confirmed_before = static_map.point_count
        for frame in range(60, 70):
            static_map.observe_scan(
                sensor_origin=(0.05, 0.05, 0.0),
                points=wall(0.06),
                now_monotonic=frame * 0.1,
                obstacle_min_height_m=-0.30,
                obstacle_max_height_m=0.30,
            )

        self.assertEqual(confirmed_before, 8)
        self.assertEqual(static_map.point_count, 8)
        self.assertEqual(static_map.dynamic_track_count, 0)

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

    def test_large_motion_keeps_sparse_map_and_bounded_grid_window(self) -> None:
        static_map = TemporalOccupancyMap(
            0.10,
            confirmation_frames=3,
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
        self.assertEqual(static_map.candidate_count, 1)

        static_map.observe_scan(
            sensor_origin=(2.05, 0.05, 0.0),
            points=[(3.05, 0.05, 0.0)],
            now_monotonic=0.1,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
        )
        self.assertEqual(static_map.candidate_count, 2)
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
            confirmation_frames=3,
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
            confirmation_frames=3,
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
