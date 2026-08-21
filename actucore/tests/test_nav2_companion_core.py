from __future__ import annotations

import math
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "navigation"
    / "runtime"
    / "g1_nav2"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from g1_nav2.costmap_validation import (  # noqa: E402
    CostmapError,
    CostmapSnapshot,
    GoalCellRejected,
    goal_cell_receipt,
    validated_goal_cell_receipt,
)
from g1_nav2.execution_protocol import (  # noqa: E402
    MotionLimits,
    Pose2D,
    ProtocolError,
    Velocity,
    VelocityProposal,
    apply_g1_motion_floor,
    apply_g1_motion_limits,
    build_velocity_proposal,
    limit_forward_velocity,
    proposal_context_is_current,
    proposal_context_is_publishable,
    shape_terminal_approach,
)
from g1_nav2.readiness import (  # noqa: E402
    evaluate_readiness,
    navigation_motion_blocker,
)


class Nav2CompanionCoreTest(unittest.TestCase):
    def test_velocity_proposal_contract_and_terminal_zero(self) -> None:
        payload = build_velocity_proposal(
            nav_id="nav-001",
            sequence=1,
            ttl_ms=250,
            navigation_status="navigating",
            velocity=Velocity(x=1.0, y=0.0, yaw=2.0),
            issued_at_unix_ms=1,
        )
        self.assertEqual(VelocityProposal.from_payload(payload).nav_id, "nav-001")

        with self.assertRaises(ProtocolError):
            build_velocity_proposal(
                nav_id="nav-001",
                sequence=2,
                ttl_ms=250,
                navigation_status="navigating",
                velocity=Velocity(x=0.10, y=1.01, yaw=0.0),
                issued_at_unix_ms=2,
            )
        payload["nav_status"] = "arrived"
        with self.assertRaises(ProtocolError):
            VelocityProposal.from_payload(payload)

    def test_requested_speed_is_enforced_on_forward_proposals(self) -> None:
        limited = limit_forward_velocity(
            Velocity(x=0.50, y=0.0, yaw=0.2),
            max_forward_mps=0.10,
        )
        self.assertEqual(limited, Velocity(x=0.10, y=0.0, yaw=0.2))
        reverse = limit_forward_velocity(
            Velocity(x=-1.0, y=0.0, yaw=0.0),
            max_forward_mps=0.10,
        )
        self.assertEqual(reverse, Velocity.zero())

    def test_g1_motion_policy_is_axis_exclusive_and_clears_dead_zones(self) -> None:
        self.assertEqual(apply_g1_motion_floor(Velocity.zero()), Velocity.zero())
        self.assertEqual(
            apply_g1_motion_floor(Velocity(x=0.01, y=0.0, yaw=0.0)),
            Velocity(x=0.30, y=0.0, yaw=0.0),
        )
        self.assertEqual(
            apply_g1_motion_floor(Velocity(x=0.0, y=0.0, yaw=-0.02)),
            Velocity(x=0.0, y=0.0, yaw=-1.00),
        )
        self.assertEqual(
            apply_g1_motion_floor(Velocity(x=0.01, y=0.0, yaw=-0.19)),
            Velocity(x=0.30, y=0.0, yaw=0.0),
        )
        self.assertEqual(
            apply_g1_motion_floor(Velocity(x=-0.05, y=0.0, yaw=0.20)),
            Velocity(x=0.0, y=0.0, yaw=1.00),
        )
        self.assertEqual(
            apply_g1_motion_floor(Velocity(x=0.50, y=0.0, yaw=-2.0)),
            Velocity(x=0.0, y=0.0, yaw=-2.0),
        )

    def test_terminal_approach_suppresses_floor_amplification_without_a_lock(self) -> None:
        target = Pose2D(x=1.0, y=2.0, yaw=-math.pi + 0.05)
        raw = Velocity(x=0.04, y=0.0, yaw=0.08)

        approaching, phase = shape_terminal_approach(
            raw,
            current_pose=Pose2D(x=0.70, y=2.0, yaw=0.0),
            target_pose=target,
        )
        self.assertEqual((approaching, phase), (raw, "approach"))

        rotating, phase = shape_terminal_approach(
            raw,
            current_pose=Pose2D(x=0.90, y=2.0, yaw=0.0),
            target_pose=target,
        )
        self.assertEqual((rotating, phase), (Velocity(yaw=0.08), "rotate"))

        reached, phase = shape_terminal_approach(
            raw,
            current_pose=Pose2D(x=0.90, y=2.0, yaw=math.pi - 0.05),
            target_pose=target,
        )
        self.assertEqual((reached, phase), (Velocity.zero(), "reached"))

        latched, phase = shape_terminal_approach(
            raw,
            current_pose=Pose2D(x=0.70, y=2.0, yaw=0.0),
            target_pose=target,
            position_reached=True,
        )
        self.assertEqual((latched, phase), (Velocity(yaw=0.08), "rotate"))

        next_goal, phase = shape_terminal_approach(
            raw,
            current_pose=Pose2D(x=0.90, y=2.0, yaw=math.pi - 0.05),
            target_pose=Pose2D(x=2.0, y=2.0, yaw=0.0),
        )
        self.assertEqual((next_goal, phase), (raw, "approach"))

    def test_stale_async_proposal_context_is_rejected(self) -> None:
        active = {"nav_id": "nav-1", "attempt": 2, "status": "navigating"}
        self.assertTrue(
            proposal_context_is_current(
                active, nav_id="nav-1", attempt=2, status="navigating"
            )
        )
        for nav_id, attempt, status in (
            ("nav-old", 2, "navigating"),
            ("nav-1", 1, "navigating"),
            ("nav-1", 2, "arrived"),
        ):
            with self.subTest(nav_id=nav_id, attempt=attempt, status=status):
                self.assertFalse(
                    proposal_context_is_current(
                        active, nav_id=nav_id, attempt=attempt, status=status
                    )
                )

    def test_terminal_context_cannot_keep_periodic_proposals_alive(self) -> None:
        navigating = {"nav_id": "nav-1", "attempt": 2, "status": "navigating"}
        self.assertTrue(
            proposal_context_is_publishable(
                navigating, nav_id="nav-1", attempt=2, status="navigating"
            )
        )

        for status in (
            "arrived",
            "cancelled",
            "stopped",
            "error",
            "aborted",
            "rejected",
            "paused",
        ):
            with self.subTest(status=status):
                active = {"nav_id": "nav-1", "attempt": 2, "status": status}
                self.assertFalse(
                    proposal_context_is_publishable(
                        active, nav_id="nav-1", attempt=2, status=status
                    )
                )

    def test_card_motion_limits_apply_axis_floors_caps_and_disable_lateral(self) -> None:
        limits = MotionLimits(
            min_x_mps=0.40,
            max_x_mps=0.80,
            min_y_mps=0.10,
            max_y_mps=0.30,
            min_yaw_rps=1.20,
            max_yaw_rps=1.60,
        )
        self.assertEqual(
            apply_g1_motion_limits(
                Velocity(x=0.05), limits=limits, max_forward_mps=0.50
            ),
            Velocity(x=0.40),
        )
        self.assertEqual(
            apply_g1_motion_limits(
                Velocity(x=0.90), limits=limits, max_forward_mps=0.50
            ),
            Velocity(x=0.50),
        )
        self.assertEqual(
            apply_g1_motion_limits(
                Velocity(x=0.05),
                limits=MotionLimits(min_x_mps=0.80, max_x_mps=1.0),
                max_forward_mps=0.50,
            ),
            Velocity(x=0.50),
        )
        self.assertEqual(
            apply_g1_motion_limits(
                Velocity(x=-0.90), limits=limits, max_forward_mps=0.50
            ),
            Velocity.zero(),
        )
        self.assertEqual(
            apply_g1_motion_limits(
                Velocity(y=-0.02), limits=limits, max_forward_mps=0.50
            ),
            Velocity(y=-0.10),
        )
        self.assertEqual(
            apply_g1_motion_limits(
                Velocity(y=0.8), limits=limits, max_forward_mps=0.50
            ),
            Velocity(y=0.30),
        )
        self.assertEqual(
            apply_g1_motion_limits(
                Velocity(y=0.20, yaw=-0.30),
                limits=limits,
                max_forward_mps=0.50,
            ),
            Velocity(y=0.0, yaw=-1.20),
        )
        self.assertEqual(
            apply_g1_motion_limits(
                Velocity(y=0.20),
                limits=MotionLimits(max_y_mps=0.0),
                max_forward_mps=0.50,
            ),
            Velocity.zero(),
        )

    def test_motion_limit_payload_is_complete_and_validated(self) -> None:
        limits = MotionLimits.from_payload(
            {
                "min_x_mps": 0.4,
                "max_x_mps": 0.8,
                "min_y_mps": 0.1,
                "max_y_mps": 0.3,
                "min_yaw_rps": 1.2,
                "max_yaw_rps": 1.8,
            }
        )
        self.assertEqual(limits.max_y_mps, 0.3)
        with self.assertRaises(ProtocolError):
            MotionLimits.from_payload(
                {
                    **limits.as_dict(),
                    "min_yaw_rps": 1.9,
                    "max_yaw_rps": 1.8,
                }
            )

    def test_fast_livo2_readiness_is_fail_closed(self) -> None:
        ready = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_received_at=9.8,
            odom_source_age_sec=0.2,
            odom_frame_ready=True,
            obstacle_received_at=9.8,
            obstacle_source_age_sec=0.2,
            obstacle_frame_ready=True,
            source_transform_ready=True,
            source_stamp_skew_sec=0.05,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            global_to_base_ready=True,
        )
        self.assertTrue(ready["navigation_ready"])
        self.assertIsNone(navigation_motion_blocker(ready))

        stale = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_received_at=9.0,
            odom_source_age_sec=1.0,
            odom_frame_ready=False,
            obstacle_received_at=None,
            obstacle_source_age_sec=None,
            obstacle_frame_ready=False,
            source_transform_ready=False,
            source_stamp_skew_sec=None,
            lifecycle_states={"planner_server": 2},
            action_server_ready=False,
            global_to_base_ready=False,
        )
        self.assertFalse(stale["navigation_ready"])
        self.assertIn("fast_livo2_odom_stale", stale["navigation_blockers"])
        self.assertIn("registered_cloud_stale", stale["navigation_blockers"])
        self.assertIn("map_to_base_unavailable", stale["navigation_blockers"])
        self.assertIn("fast_livo2_odom_frame_invalid", stale["navigation_blockers"])
        self.assertIn("registered_cloud_frame_invalid", stale["navigation_blockers"])
        self.assertIn("fast_livo2_odom_stale", navigation_motion_blocker(stale))

        skewed = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_received_at=9.9,
            odom_source_age_sec=0.1,
            odom_frame_ready=True,
            obstacle_received_at=9.9,
            obstacle_source_age_sec=0.1,
            obstacle_frame_ready=True,
            source_transform_ready=True,
            source_stamp_skew_sec=0.6,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            global_to_base_ready=True,
        )
        self.assertTrue(skewed["navigation_ready"])
        self.assertEqual(skewed["fast_livo2_source_stamp_skew_sec"], 0.6)

        unpaired = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            odom_received_at=9.9,
            odom_source_age_sec=0.1,
            odom_frame_ready=True,
            obstacle_received_at=9.9,
            obstacle_source_age_sec=0.1,
            obstacle_frame_ready=True,
            source_transform_ready=False,
            source_stamp_skew_sec=0.01,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            global_to_base_ready=True,
        )
        self.assertIn(
            "registered_cloud_transform_unavailable",
            unpaired["navigation_blockers"],
        )

        boundary_jitter = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            source_max_age_sec=1.0,
            odom_received_at=9.9,
            odom_source_age_sec=0.71,
            odom_frame_ready=True,
            obstacle_received_at=9.9,
            obstacle_source_age_sec=0.71,
            obstacle_frame_ready=True,
            source_transform_ready=True,
            source_stamp_skew_sec=0.01,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            global_to_base_ready=True,
        )
        self.assertTrue(boundary_jitter["navigation_ready"])

        too_old = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.5,
            source_max_age_sec=1.0,
            odom_received_at=9.9,
            odom_source_age_sec=1.001,
            odom_frame_ready=True,
            obstacle_received_at=9.9,
            obstacle_source_age_sec=1.001,
            obstacle_frame_ready=True,
            source_transform_ready=True,
            source_stamp_skew_sec=0.01,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            global_to_base_ready=True,
        )
        self.assertIn("odom_source_stamp_stale", too_old["navigation_blockers"])
        self.assertEqual(boundary_jitter["sensor_receive_max_age_sec"], 0.5)
        self.assertEqual(boundary_jitter["sensor_source_max_age_sec"], 1.0)

        scheduler_jitter = evaluate_readiness(
            now_monotonic=10.0,
            max_age_sec=0.8,
            source_max_age_sec=1.0,
            odom_received_at=9.37,
            odom_source_age_sec=0.63,
            odom_frame_ready=True,
            obstacle_received_at=9.37,
            obstacle_source_age_sec=0.63,
            obstacle_frame_ready=True,
            source_transform_ready=True,
            source_stamp_skew_sec=0.01,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            global_to_base_ready=True,
        )
        self.assertTrue(scheduler_jitter["navigation_ready"])

    def test_goal_cell_is_checked_before_nav2_action_dispatch(self) -> None:
        snapshot = CostmapSnapshot.from_values(
            frame_id="map",
            stamp_ns=123,
            resolution=0.5,
            width=4,
            height=3,
            origin_x=-1.0,
            origin_y=-0.5,
            origin_yaw=0.0,
            data=[
                0,
                0,
                0,
                0,
                0,
                20,
                99,
                100,
                -1,
                0,
                0,
                0,
            ],
            received_monotonic=10.0,
        )

        free = goal_cell_receipt(
            snapshot,
            x=-0.75,
            y=-0.25,
            expected_frame="map",
            max_receive_age_sec=2.0,
            now_monotonic=10.5,
        )
        self.assertEqual(free["cost"], 0)
        self.assertFalse(free["collision"])

        inscribed = goal_cell_receipt(
            snapshot,
            x=0.25,
            y=0.25,
            expected_frame="map",
            max_receive_age_sec=2.0,
            now_monotonic=10.5,
        )
        self.assertEqual(inscribed["cost"], 99)
        self.assertTrue(inscribed["collision"])
        with self.assertRaises(GoalCellRejected) as rejected:
            validated_goal_cell_receipt(
                snapshot,
                x=0.25,
                y=0.25,
                expected_frame="map",
                max_receive_age_sec=2.0,
                now_monotonic=10.5,
            )
        self.assertEqual(rejected.exception.code, "goal_in_collision")

        with self.assertRaises(GoalCellRejected) as rejected:
            validated_goal_cell_receipt(
                snapshot,
                x=-0.75,
                y=0.75,
                expected_frame="map",
                max_receive_age_sec=2.0,
                now_monotonic=10.5,
            )
        self.assertEqual(rejected.exception.code, "goal_cost_unknown")

        diagnostics = snapshot.diagnostics(now_monotonic=10.5)
        self.assertEqual(diagnostics["inflated_cells"], 1)
        self.assertEqual(diagnostics["inscribed_cells"], 1)
        self.assertEqual(diagnostics["lethal_cells"], 1)
        self.assertEqual(diagnostics["unknown_cells"], 1)
        with self.assertRaisesRegex(CostmapError, "outside"):
            goal_cell_receipt(
                snapshot,
                x=2.0,
                y=0.0,
                expected_frame="map",
                max_receive_age_sec=2.0,
                now_monotonic=10.5,
            )
        with self.assertRaises(GoalCellRejected) as rejected:
            validated_goal_cell_receipt(
                snapshot,
                x=2.0,
                y=0.0,
                expected_frame="map",
                max_receive_age_sec=2.0,
                now_monotonic=10.5,
            )
        self.assertEqual(rejected.exception.code, "goal_outside_costmap")
        with self.assertRaisesRegex(CostmapError, "receive age"):
            goal_cell_receipt(
                snapshot,
                x=0.0,
                y=0.0,
                expected_frame="map",
                max_receive_age_sec=2.0,
                now_monotonic=12.1,
            )

    def test_runtime_is_planner_controller_only(self) -> None:
        setup = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
        launch = (PACKAGE_ROOT / "launch" / "g1_nav2.launch.py").read_text(
            encoding="utf-8"
        )
        package = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")
        dockerfile = (
            Path(__file__).resolve().parents[1] / "Dockerfile.jetson"
        ).read_text(encoding="utf-8")
        service = (
            Path(__file__).resolve().parents[1] / "deploy" / "service.yml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "planner_command_bridge = g1_nav2.planner_command_node:main", setup
        )
        for removed in (
            "runtime_supervisor",
            "canvas_pointcloud_bridge",
            "canvas_map_view",
            "loco_odom_bridge",
        ):
            self.assertNotIn(f'"{removed} =', setup)
        self.assertIn('executable="planner_command_bridge"', launch)
        self.assertIn("GroupAction(", launch)
        self.assertIn("scoped=True", launch)
        self.assertIn('default_value="/ubuntu/navigation/odom"', launch)
        self.assertIn(
            'default_value="/ubuntu/navigation/cloud_registered"', launch
        )
        self.assertNotIn("slam_toolbox", package)
        self.assertNotIn("pointcloud_to_laserscan", package)
        self.assertNotIn("slam_toolbox", dockerfile)
        self.assertNotIn("pointcloud_to_laserscan", dockerfile)
        self.assertNotIn("NAV2_MODE", service)
        self.assertNotIn("container_name: embodied-perception-nav2", service)

    def test_costmaps_combine_confirmed_static_map_and_live_clearing(self) -> None:
        params = (PACKAGE_ROOT / "config" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("amcl:", params)
        self.assertNotIn("map_server:", params)
        self.assertNotIn("data_type: LaserScan", params)
        self.assertEqual(params.count("topic: /ubuntu/navigation/cloud_registered"), 2)
        self.assertNotIn("topic: /ubuntu/navigation/obstacle_map", params)
        self.assertEqual(params.count("map_topic: /ubuntu/navigation/static_map"), 1)
        self.assertEqual(params.count("data_type: PointCloud2"), 2)
        self.assertIn("min_obstacle_height: -3.0", params)
        self.assertIn("max_obstacle_height: 3.0", params)
        self.assertIn("adapter already applies the card-configured Z band", params)
        self.assertIn("global_frame: map", params)
        self.assertIn("rolling_window: true", params)
        local = params.split("local_costmap:\n", 1)[1].split(
            "\nglobal_costmap:", 1
        )[0]
        global_map = params.split("global_costmap:\n", 1)[1].split(
            "\nplanner_server:", 1
        )[0]
        self.assertIn("sensor_frame: base_link", local)
        self.assertIn("clearing: true", local)
        self.assertIn("raytrace_max_range: 8.5", local)
        self.assertIn("plugins: [static_layer, obstacle_layer, inflation_layer]", global_map)
        self.assertIn("plugin: nav2_costmap_2d::StaticLayer", global_map)
        self.assertIn("map_subscribe_transient_local: true", global_map)
        self.assertIn("subscribe_to_updates: false", global_map)
        self.assertIn("sensor_frame: base_link", global_map)
        self.assertIn("clearing: true", global_map)
        self.assertIn("marking: true", global_map)
        self.assertIn("raytrace_max_range: 8.5", global_map)
        self.assertNotIn("clearing: false", global_map)
        self.assertIn("inflation_radius: 0.55", local)
        self.assertIn("inflation_radius: 0.55", global_map)
        footprint_radius = math.hypot(0.32, 0.28)
        self.assertAlmostEqual(footprint_radius, 0.4252058, places=6)
        self.assertGreater(0.55 - footprint_radius, 0.12)
        self.assertLess(0.55 - footprint_radius, 0.13)

    def test_controller_faces_path_and_disables_lateral_motion(self) -> None:
        params = (PACKAGE_ROOT / "config" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
        follow_path = params.split("    FollowPath:\n", 1)[1].split(
            "\n\nlocal_costmap:", 1
        )[0]
        smoother = params.split("velocity_smoother:\n", 1)[1]
        for expected in (
            "plugin: nav2_rotation_shim_controller::RotationShimController",
            "primary_controller: dwb_core::DWBLocalPlanner",
            "min_vel_y: 0.0",
            "max_vel_y: 0.0",
            "min_speed_xy: 0.30",
            "vy_samples: 1",
        ):
            self.assertIn(expected, follow_path)
        self.assertIn("rotate_to_heading_angular_vel: 1.00", follow_path)
        self.assertIn("min_speed_theta: 1.00", follow_path)
        self.assertIn("max_vel_theta: 2.00", follow_path)
        self.assertIn("max_velocity: [1.0, 0.0, 2.0]", smoother)
        self.assertIn("min_velocity: [0.0, 0.0, -2.0]", smoother)
        self.assertIn("odom_topic: /ubuntu/navigation/odom", smoother)

    def test_speed_limit_and_behavior_tree_reach_planner_bridge(self) -> None:
        command = (
            PACKAGE_ROOT / "g1_nav2" / "planner_command_node.py"
        ).read_text(encoding="utf-8")
        launch = (PACKAGE_ROOT / "launch" / "g1_nav2.launch.py").read_text(
            encoding="utf-8"
        )
        params = (PACKAGE_ROOT / "config" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
        tree = ET.parse(
            PACKAGE_ROOT
            / "behavior_trees"
            / "navigate_to_pose_w_replanning_and_recovery.xml"
        )
        through_tree = ET.parse(
            PACKAGE_ROOT
            / "behavior_trees"
            / "navigate_through_poses_w_replanning_and_recovery.xml"
        )
        self.assertIsNone(tree.find(".//BackUp"))
        self.assertIsNone(through_tree.find(".//BackUp"))
        self.assertIsNotNone(through_tree.find(".//ComputePathThroughPoses"))
        self.assertIn(
            "behavior_plugins: [spin, drive_on_heading, wait]",
            params,
        )
        self.assertIn(
            "default_nav_to_pose_bt_xml: "
            "/ros_ws/install/g1_nav2/share/g1_nav2/behavior_trees/"
            "navigate_to_pose_w_replanning_and_recovery.xml",
            params,
        )
        self.assertIn(
            "default_nav_through_poses_bt_xml: "
            "/ros_ws/install/g1_nav2/share/g1_nav2/behavior_trees/"
            "navigate_through_poses_w_replanning_and_recovery.xml",
            params,
        )
        self.assertIn("bt_navigator_navigate_through_poses_rclcpp_node", params)
        self.assertIn("from nav2_msgs.msg import SpeedLimit", command)
        self.assertIn("self._publish_controller_speed_limit(speed_limit)", command)
        self.assertIn("MotionLimits.from_payload", command)
        self.assertIn("apply_g1_motion_limits", command)
        self.assertIn("shape_terminal_approach", command)
        self.assertIn('"terminal_phase": "approach"', command)
        self.assertIn(
            'position_reached=terminal_phase in {"rotate", "reached"}', command
        )
        self.assertIn('"terminal_xy_tolerance_m": 0.18', launch)
        self.assertIn('"terminal_yaw_tolerance_rad": 0.45', launch)
        self.assertIn('"sensor_max_age_sec": 0.8', launch)
        self.assertIn('"sensor_source_max_age_sec": 1.0', launch)
        self.assertIn(
            "odom_source_age = self._source_age(odom_source_stamp_ns)",
            command,
        )
        self.assertIn(
            "obstacle_source_age = self._source_age(obstacle_source_stamp_ns)",
            command,
        )
        self.assertNotIn("_last_odom_source_age_sec", command)
        self.assertNotIn("_last_obstacle_source_age_sec", command)
        self.assertIn('"goal_xy_tolerance_m": 0.20', launch)
        self.assertIn('"goal_yaw_tolerance_rad": 0.50', launch)
        self.assertIn("xy_goal_tolerance: 0.20", params)
        self.assertIn("yaw_goal_tolerance: 0.50", params)
        self.assertIn("controller_frequency: 5.0", params)
        self.assertIn("bt_loop_duration: 50", params)
        self.assertIn("smoothing_frequency: 5.0", params)
        self.assertIn('self.declare_parameter("proposal_frequency_hz", 5.0)', command)
        self.assertIn("depth=1", command)
        self.assertIn("proposal_context_is_publishable", command)
        self.assertIn('payload.get("velocity_limits")', command)
        self.assertIn("goal.pose.header.frame_id = self._global_frame", command)
        send_goal = command.split("    def _send_active_goal", 1)[1].split(
            "    def _publish_controller_speed_limit", 1
        )[0]
        self.assertLess(
            send_goal.index("self._validate_goal_cell(target)"),
            send_goal.index("self._action_client.wait_for_server"),
        )
        self.assertIn('"goal_costmap_max_age_sec": 2.0', launch)
        self.assertNotIn("source_age_tolerance_sec", launch)


if __name__ == "__main__":
    unittest.main()
