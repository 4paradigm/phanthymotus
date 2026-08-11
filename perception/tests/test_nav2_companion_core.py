from __future__ import annotations

import sys
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

from g1_nav2.execution_protocol import (  # noqa: E402
    ProtocolError,
    Velocity,
    VelocityProposal,
    build_velocity_proposal,
    limit_forward_velocity,
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
            velocity=Velocity(x=1.0, y=0.0, yaw=0.35),
            issued_at_unix_ms=1,
        )
        self.assertEqual(VelocityProposal.from_payload(payload).nav_id, "nav-001")

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
        self.assertEqual(reverse.x, -1.0)

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
            source_stamp_skew_sec=0.6,
            lifecycle_states={"planner_server": 3, "bt_navigator": 3},
            action_server_ready=True,
            global_to_base_ready=True,
        )
        self.assertFalse(skewed["navigation_ready"])
        self.assertIn(
            "fast_livo2_source_stamp_skew", skewed["navigation_blockers"]
        )

    def test_runtime_is_planner_controller_only(self) -> None:
        setup = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
        launch = (PACKAGE_ROOT / "launch" / "g1_nav2.launch.py").read_text(
            encoding="utf-8"
        )
        package = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")
        companion = PACKAGE_ROOT.parent
        dockerfile = (companion / "Dockerfile").read_text(encoding="utf-8")
        compose = (companion / "compose.yml").read_text(encoding="utf-8")

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
        self.assertNotIn("/maps", compose)
        self.assertNotIn("NAV2_MODE", compose)

    def test_costmaps_consume_fast_livo2_registered_cloud(self) -> None:
        params = (PACKAGE_ROOT / "config" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("amcl:", params)
        self.assertNotIn("map_server:", params)
        self.assertNotIn("static_layer", params)
        self.assertNotIn("data_type: LaserScan", params)
        self.assertEqual(
            params.count("topic: /ubuntu/navigation/cloud_registered"), 2
        )
        self.assertEqual(params.count("data_type: PointCloud2"), 2)
        self.assertIn("global_frame: map", params)
        self.assertIn("rolling_window: true", params)

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
            "min_speed_xy: 0.10",
            "vy_samples: 1",
        ):
            self.assertIn(expected, follow_path)
        self.assertIn("max_velocity: [1.0, 0.0, 0.25]", smoother)
        self.assertIn("odom_topic: /ubuntu/navigation/odom", smoother)

    def test_speed_limit_and_behavior_tree_reach_planner_bridge(self) -> None:
        command = (
            PACKAGE_ROOT / "g1_nav2" / "planner_command_node.py"
        ).read_text(encoding="utf-8")
        tree = ET.parse(
            PACKAGE_ROOT
            / "behavior_trees"
            / "navigate_to_pose_w_replanning_and_recovery.xml"
        )
        backup = tree.find(".//BackUp")
        self.assertIsNotNone(backup)
        self.assertEqual(backup.attrib["backup_speed"], "0.15")
        self.assertIn("from nav2_msgs.msg import SpeedLimit", command)
        self.assertIn("self._publish_controller_speed_limit(speed_limit)", command)
        self.assertIn("limit_forward_velocity", command)
        self.assertIn("goal.pose.header.frame_id = self._global_frame", command)


if __name__ == "__main__":
    unittest.main()
