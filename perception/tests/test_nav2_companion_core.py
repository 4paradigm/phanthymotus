from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "nav2"
    / "companion"
    / "g1_nav2"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from g1_nav2.canvas_pointcloud_core import (  # noqa: E402
    InvalidCanvasPointCloud,
    LidarClockNormalizer,
    point_time_max_offset_ns,
    validate_standard_pointcloud,
)
from g1_nav2.execution_protocol import (  # noqa: E402
    ProtocolError,
    Velocity,
    VelocityProposal,
    build_velocity_proposal,
)
from g1_nav2.loco_odom_core import (  # noqa: E402
    InvalidLocoState,
    OriginNormalizer,
)
from g1_nav2.map_view_core import (  # noqa: E402
    InvalidMapView,
    build_occupancy_snapshot,
    encode_canvas_mapping_frame,
)
from g1_nav2.map_store import MapStore, MapStoreError  # noqa: E402
from g1_nav2.readiness import (  # noqa: E402
    evaluate_readiness,
    navigation_motion_blocker,
)
from g1_nav2.runtime_process import build_launch_command  # noqa: E402


class Nav2CompanionCoreTest(unittest.TestCase):
    def test_canvas_map_view_encodes_occupied_cells_and_robot_pose(self) -> None:
        snapshot = build_occupancy_snapshot(
            width=3,
            height=2,
            resolution=0.5,
            origin_x=1.0,
            origin_y=-1.0,
            origin_yaw=0.0,
            data=[0, 65, -1, 100, 10, 90],
            occupancy_threshold=65,
            max_points=2,
        )
        self.assertEqual(snapshot.occupied_cell_count, 3)
        self.assertEqual(snapshot.point_count, 2)

        payload = encode_canvas_mapping_frame(
            snapshot,
            robot_x=1.25,
            robot_y=-0.75,
            robot_yaw=0.5,
        )
        robot_x, robot_y, robot_yaw, flags, point_count = struct.unpack_from(
            "<fffBI", payload, 0
        )
        self.assertAlmostEqual(robot_x, 1.25)
        self.assertAlmostEqual(robot_y, -0.75)
        self.assertAlmostEqual(robot_yaw, 0.5)
        self.assertEqual(flags, 0x03)
        self.assertEqual(point_count, 2)
        first_point = struct.unpack_from("<fff", payload, 17)
        self.assertEqual(first_point, (1.75, -0.75, 0.0))

        with self.assertRaises(InvalidMapView):
            build_occupancy_snapshot(
                width=2,
                height=2,
                resolution=0.5,
                origin_x=0.0,
                origin_y=0.0,
                origin_yaw=0.0,
                data=[0, 100],
            )

    def test_native_pointcloud2_preserves_header_contract(self) -> None:
        stamp_ns = validate_standard_pointcloud(
            stamp_sec=10,
            stamp_nanosec=20,
            receive_stamp_ns=None,
            frame_id="utlidar_lidar",
            height=1,
            width=2,
            point_step=22,
            row_step=44,
            data_length=44,
            field_names=("x", "y", "z", "intensity", "ring", "time"),
        )
        self.assertEqual(stamp_ns, 10_000_000_020)

        with self.assertRaisesRegex(InvalidCanvasPointCloud, "x/y/z"):
            validate_standard_pointcloud(
                stamp_sec=10,
                stamp_nanosec=0,
                receive_stamp_ns=None,
                frame_id="utlidar_lidar",
                height=1,
                width=2,
                point_step=22,
                row_step=44,
                data_length=44,
                field_names=("x", "y", "time"),
            )

    def test_native_lidar_clock_is_mapped_to_ros_system_time(self) -> None:
        normalizer = LidarClockNormalizer(
            warmup_samples=3,
            window_samples=5,
        )
        raw = 1_700_000_000_000_000_000
        clock_offset = 16_857_528_000_000_000
        scan_span = 99_840_000

        self.assertIsNone(
            normalizer.normalize(
                raw_stamp_ns=raw,
                receive_stamp_ns=raw + clock_offset + scan_span + 8_000_000,
                scan_end_offset_ns=scan_span,
            )
        )
        self.assertIsNone(
            normalizer.normalize(
                raw_stamp_ns=raw + 100_000_000,
                receive_stamp_ns=(
                    raw + 100_000_000 + clock_offset + scan_span + 2_000_000
                ),
                scan_end_offset_ns=scan_span,
            )
        )
        normalized = normalizer.normalize(
            raw_stamp_ns=raw + 200_000_000,
            receive_stamp_ns=(
                raw + 200_000_000 + clock_offset + scan_span + 5_000_000
            ),
            scan_end_offset_ns=scan_span,
        )

        self.assertEqual(
            normalized,
            raw + 200_000_000 + clock_offset + 2_000_000,
        )
        self.assertEqual(normalizer.snapshot().mode, "normalize")

        late = normalizer.normalize(
            raw_stamp_ns=raw + 300_000_000,
            receive_stamp_ns=(
                raw + 300_000_000 + clock_offset + scan_span + 400_000_000
            ),
            scan_end_offset_ns=scan_span,
        )
        self.assertEqual(
            late,
            raw + 300_000_000 + clock_offset + 2_000_000,
        )
        self.assertEqual(normalizer.snapshot().resets, 0)

    def test_aligned_lidar_stamp_is_passed_through(self) -> None:
        normalizer = LidarClockNormalizer(warmup_samples=3)
        normalized = normalizer.normalize(
            raw_stamp_ns=10_000_000_000,
            receive_stamp_ns=10_120_000_000,
            scan_end_offset_ns=100_000_000,
        )

        self.assertEqual(normalized, 10_000_000_000)
        self.assertEqual(normalizer.snapshot().mode, "passthrough")

    def test_native_point_time_field_defines_scan_end(self) -> None:
        data = bytearray(44)
        struct.pack_into("<f", data, 18, 5_000.0)
        struct.pack_into("<f", data, 40, 99_840_000.0)
        fields = [
            type("Field", (), {"name": "time", "offset": 18, "datatype": 7, "count": 1})()
        ]

        self.assertEqual(
            point_time_max_offset_ns(
                data=data,
                fields=fields,
                height=1,
                width=2,
                point_step=22,
                row_step=44,
                is_bigendian=False,
            ),
            99_840_000,
        )

    def test_canvas_map_view_is_installed_and_launched(self) -> None:
        setup = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
        launch = (
            PACKAGE_ROOT / "launch" / "g1_nav2.launch.py"
        ).read_text(encoding="utf-8")
        node = (PACKAGE_ROOT / "g1_nav2" / "map_view_node.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("canvas_map_view = g1_nav2.map_view_node:main", setup)
        self.assertIn('executable="canvas_map_view"', launch)
        self.assertIn(
            'default_value="/ubuntu/navigation/nav2/map_view"', launch
        )
        invalid_map_handler = node.split(
            "def _report_invalid_map", 1
        )[1].split("def _publish_view", 1)[0]
        self.assertIn("self._snapshot = None", invalid_map_handler)

    def test_velocity_proposal_contract_and_terminal_zero(self) -> None:
        payload = build_velocity_proposal(
            nav_id="nav-001",
            sequence=1,
            ttl_ms=250,
            navigation_status="navigating",
            velocity=Velocity(x=0.15, y=0.0, yaw=0.35),
            issued_at_unix_ms=1,
        )
        parsed = VelocityProposal.from_payload(payload)
        self.assertEqual(parsed.nav_id, "nav-001")

        reverse = build_velocity_proposal(
            nav_id="nav-001",
            sequence=2,
            ttl_ms=250,
            navigation_status="navigating",
            velocity=Velocity(x=-0.15, y=0.0, yaw=0.0),
            issued_at_unix_ms=2,
        )
        self.assertEqual(
            VelocityProposal.from_payload(reverse).velocity.x, -0.15
        )
        with self.assertRaises(ProtocolError):
            build_velocity_proposal(
                nav_id="nav-001",
                sequence=3,
                ttl_ms=250,
                navigation_status="navigating",
                velocity=Velocity(x=-0.151, y=0.0, yaw=0.0),
                issued_at_unix_ms=3,
            )

        with self.assertRaises(ProtocolError):
            build_velocity_proposal(
                nav_id="nav-001",
                sequence=2,
                ttl_ms=250,
                navigation_status="navigating",
                velocity=Velocity(x=0.10, y=0.01, yaw=0.0),
                issued_at_unix_ms=2,
            )

        payload["nav_status"] = "arrived"
        with self.assertRaises(ProtocolError):
            VelocityProposal.from_payload(payload)
        with self.assertRaises(ProtocolError):
            build_velocity_proposal(
                nav_id="nav-001",
                sequence=2,
                ttl_ms=251,
                navigation_status="navigating",
                velocity=Velocity.zero(),
                issued_at_unix_ms=2,
            )

    def test_controller_faces_path_and_disables_lateral_motion(self) -> None:
        params_path = PACKAGE_ROOT / "config" / "nav2_params.yaml"
        params = params_path.read_text(encoding="utf-8")
        follow_path = params.split("    FollowPath:\n", 1)[1].split(
            "\n\nlocal_costmap:", 1
        )[0]
        smoother = params.split("velocity_smoother:\n", 1)[1]

        expected_follow_path = (
            "plugin: nav2_rotation_shim_controller::RotationShimController",
            "primary_controller: dwb_core::DWBLocalPlanner",
            "min_vel_y: 0.0",
            "max_vel_y: 0.0",
            "vy_samples: 1",
            "rotate_to_goal_heading: false",
        )
        for expected in expected_follow_path:
            self.assertIn(expected, follow_path)
        self.assertIn("max_velocity: [0.15, 0.0, 0.25]", smoother)
        self.assertIn("min_velocity: [-0.15, 0.0, -0.25]", smoother)

    def test_speed_parameter_reaches_controller_and_backup_is_usable(self) -> None:
        params = (PACKAGE_ROOT / "config" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
        setup = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
        launch = (PACKAGE_ROOT / "launch" / "g1_nav2.launch.py").read_text(
            encoding="utf-8"
        )
        command = (
            PACKAGE_ROOT / "g1_nav2" / "navigation_command_node.py"
        ).read_text(encoding="utf-8")
        tree = ET.parse(
            PACKAGE_ROOT
            / "behavior_trees"
            / "navigate_to_pose_w_replanning_and_recovery.xml"
        )

        backup = tree.find(".//BackUp")
        self.assertIsNotNone(backup)
        self.assertEqual(backup.attrib["backup_speed"], "0.15")
        self.assertIn(
            "speed_limit_topic: /ubuntu/navigation/nav2/speed_limit", params
        )
        self.assertIn('glob("behavior_trees/*.xml")', setup)
        self.assertIn('"behavior_tree_path": os.path.join(', launch)
        self.assertIn("from nav2_msgs.msg import SpeedLimit", command)
        self.assertIn("self._apply_controller_speed_limit(speed_limit)", command)
        self.assertIn("wait_for_all_acked", command)
        self.assertIn("goal.behavior_tree = self._behavior_tree_path", command)

    def test_legacy_loco_adapter_labels_receive_time(self) -> None:
        normalizer = OriginNormalizer()
        first = normalizer.convert(
            {
                "position": [2.0, 3.0],
                "velocity": [0.1, 0.0],
                "imu": {"rpy": [0.0, 0.0, 0.2]},
                "yaw_speed": 0.1,
            },
            receive_stamp_ns=1_000_000_000,
        )
        self.assertAlmostEqual(first.x, 0.0)
        self.assertAlmostEqual(first.y, 0.0)
        self.assertEqual(first.timestamp_source, "adapter_receive")

    def test_loco_v2_preserves_driver_receive_time(self) -> None:
        source_stamp_ns = 1_000_000_000
        normalizer = OriginNormalizer()
        odom = normalizer.convert(
            {
                "schema": "phanthy.g1.loco_state.v2",
                "source_stamp_ns": source_stamp_ns,
                "timestamp_source": "driver_receive",
                "frame_id": "odom_source",
                "position": [2.0, 3.0],
                "velocity": [0.1, 0.0],
                "imu": {"rpy": [0.0, 0.0, 0.2]},
                "yaw_speed": 0.1,
            },
            receive_stamp_ns=1_100_000_000,
        )
        self.assertEqual(odom.timestamp_source, "driver_receive")
        self.assertEqual(odom.source_stamp_ns, source_stamp_ns)

    def test_loco_v2_rejects_bad_clock_and_schema(self) -> None:
        stale_normalizer = OriginNormalizer()
        stale_state = {
            "schema_version": 2,
            "source_stamp_ns": 1_000_000_000,
            "timestamp_source": "driver_receive",
            "frame_id": "odom_source",
            "position": [2.0, 3.0],
            "velocity": [0.1, 0.0],
            "imu": {"rpy": [0.0, 0.0, 0.2]},
            "yaw_speed": 0.1,
        }
        with self.assertRaisesRegex(InvalidLocoState, "stale"):
            stale_normalizer.convert(
                stale_state,
                receive_stamp_ns=1_600_000_001,
            )
        self.assertFalse(stale_normalizer.initialized)

        conflict = dict(stale_state)
        conflict["schema"] = "unitree.g1.loco_state.legacy"
        with self.assertRaisesRegex(InvalidLocoState, "conflicts"):
            OriginNormalizer().convert(
                conflict,
                receive_stamp_ns=1_100_000_000,
            )

        ambiguous_time = dict(stale_state)
        ambiguous_time.pop("timestamp_source")
        ambiguous_time["source_stamp_ns"] = 1_000_000_000
        with self.assertRaisesRegex(InvalidLocoState, "driver_receive"):
            OriginNormalizer().convert(
                ambiguous_time,
                receive_stamp_ns=1_100_000_000,
            )

    def test_v2_loco_state_rejects_source_time_regression(self) -> None:
        normalizer = OriginNormalizer()
        state = {
            "schema": "phanthy.g1.loco_state.v2",
            "source_stamp_ns": 1_000_000_000,
            "timestamp_source": "driver_receive",
            "frame_id": "odom_source",
            "position": [2.0, 3.0],
            "velocity": [0.1, 0.0],
            "imu": {"rpy": [0.0, 0.0, 0.2]},
            "yaw_speed": 0.1,
        }
        normalizer.convert(state, receive_stamp_ns=1_100_000_000)
        state["source_stamp_ns"] = 900_000_000
        with self.assertRaisesRegex(InvalidLocoState, "moved backwards"):
            normalizer.convert(state, receive_stamp_ns=1_150_000_000)

    def test_readiness_is_fail_closed_for_stale_inputs(self) -> None:
        ready = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_status={
                "state": "ready",
                "timestamp_source": "driver",
                "source_age_sec": 0.2,
            },
            odom_status_received_at=9.8,
            scan_received_at=9.8,
            scan_source_age_sec=0.2,
            sensor_stamp_skew_sec=0.05,
            max_sensor_stamp_skew_sec=0.2,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            map_ready=True,
            map_to_base_ready=True,
        )
        self.assertTrue(ready["navigation_ready"])
        self.assertIsNone(navigation_motion_blocker(ready))

        stale = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_status={"state": "ready", "timestamp_source": "adapter_receive"},
            odom_status_received_at=9.0,
            scan_received_at=None,
            scan_source_age_sec=None,
            sensor_stamp_skew_sec=None,
            max_sensor_stamp_skew_sec=0.2,
            lifecycle_states={"planner_server": 2},
            action_server_ready=False,
            map_ready=False,
            map_to_base_ready=False,
        )
        self.assertFalse(stale["n3_ready"])
        self.assertFalse(stale["navigation_ready"])
        self.assertIn("odom_status_stale", stale["readiness_blockers"])
        self.assertIn("scan_stale", stale["readiness_blockers"])
        blocker = navigation_motion_blocker(stale)
        self.assertIsNotNone(blocker)
        self.assertIn("odom_status_stale", blocker)
        self.assertIn("scan_stale", blocker)

        legacy_async = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_status={"state": "ready", "timestamp_source": "adapter_receive"},
            odom_status_received_at=9.9,
            scan_received_at=9.9,
            scan_source_age_sec=0.1,
            sensor_stamp_skew_sec=0.45,
            max_sensor_stamp_skew_sec=0.2,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            map_ready=True,
            map_to_base_ready=True,
        )
        self.assertNotIn("sensor_stamp_skew", legacy_async["readiness_blockers"])

        driver_receive_async = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_status={
                "state": "ready",
                "timestamp_source": "driver_receive",
                "source_age_sec": 0.1,
            },
            odom_status_received_at=9.9,
            scan_received_at=9.9,
            scan_source_age_sec=0.1,
            sensor_stamp_skew_sec=0.45,
            max_sensor_stamp_skew_sec=0.2,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            map_ready=True,
            map_to_base_ready=True,
        )
        self.assertNotIn(
            "sensor_stamp_skew",
            driver_receive_async["readiness_blockers"],
        )

        skewed = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_status={
                "state": "ready",
                "timestamp_source": "driver",
                "source_age_sec": 0.1,
            },
            odom_status_received_at=9.9,
            scan_received_at=9.9,
            scan_source_age_sec=0.1,
            sensor_stamp_skew_sec=0.201,
            max_sensor_stamp_skew_sec=0.2,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            map_ready=True,
            map_to_base_ready=True,
        )
        self.assertIn("sensor_stamp_skew", skewed["readiness_blockers"])

    def test_map_store_finalizes_atomically_and_rejects_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = MapStore(temporary)
            with self.assertRaises(MapStoreError):
                store.begin_mapping("../escape")

            session = store.begin_mapping("room-a")
            (session.path / "map.yaml").write_text("image: map.pgm\n", encoding="utf-8")
            (session.path / "map.pgm").write_bytes(b"P2\n1 1\n255\n0\n")
            (session.path / "map.posegraph").write_bytes(b"posegraph")
            (session.path / "map.data").write_bytes(b"data")
            summary = store.finalize_mapping(session)
            self.assertEqual(summary["status"], "ready")
            self.assertEqual(store.list_maps()[0]["map_name"], "room-a")

            directory = store.directory_for_map("room-a")
            tag = store.put_tag(
                directory, "room-a", "door", "exit", {"x": 1, "y": 2, "yaw": 0}
            )
            self.assertEqual(tag["name"], "door")
            self.assertEqual(len(store.list_tags(directory, "room-a")), 1)
            store.remove_tag(directory, "room-a", "door")
            self.assertEqual(store.list_tags(directory, "room-a"), [])

    def test_runtime_launch_requires_extrinsics_and_saved_map(self) -> None:
        env = {
            "NAV2_LIDAR_X": "0",
            "NAV2_LIDAR_Y": "0",
            "NAV2_LIDAR_Z": "0.46",
            "NAV2_LIDAR_ROLL": "0",
            "NAV2_LIDAR_PITCH": "0.04",
            "NAV2_LIDAR_YAW": "0",
        }
        mapping = build_launch_command(mode="mapping", environ=env)
        self.assertIn("mode:=mapping", mapping)
        self.assertIn("publish_lidar_static_tf:=true", mapping)
        env["NAV2_PUBLISH_LIDAR_STATIC_TF"] = "false"
        self.assertIn(
            "publish_lidar_static_tf:=false",
            build_launch_command(mode="mapping", environ=env),
        )
        env["NAV2_PUBLISH_LIDAR_STATIC_TF"] = "true"
        with tempfile.TemporaryDirectory() as temporary:
            map_dir = Path(temporary) / "room-a"
            map_dir.mkdir()
            (map_dir / "map.yaml").write_text("image: map.pgm\n", encoding="utf-8")
            localized = build_launch_command(
                mode="localization",
                map_name="room-a",
                maps_root=temporary,
                environ=env,
            )
            self.assertIn("map_name:=room-a", localized)
        with self.assertRaises(ValueError):
            build_launch_command(mode="mapping", environ={})

    def test_runtime_mode_switch_action_reuses_supervisor(self) -> None:
        command = (
            PACKAGE_ROOT / "g1_nav2" / "navigation_command_node.py"
        ).read_text(encoding="utf-8")
        supervisor = (
            PACKAGE_ROOT / "g1_nav2" / "runtime_supervisor.py"
        ).read_text(encoding="utf-8")

        self.assertIn('if action == "switch_runtime_mode":', command)
        self.assertIn("def _switch_runtime_mode", command)
        self.assertIn('"stop_mapping is required before switching', command)
        self.assertIn('target_mode not in {"mapping", "localization"}', command)
        self.assertIn('target_mode not in VALID_MODES', supervisor)

    def test_numeric_map_name_is_forced_to_string_in_launch(self) -> None:
        launch = (
            PACKAGE_ROOT / "launch" / "g1_nav2.launch.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '"startup_map_name": ParameterValue(map_name, value_type=str)',
            launch,
        )


if __name__ == "__main__":
    unittest.main()
