from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))

from plugins.navigation.planning.contract import (  # noqa: E402
    NAV2_ACTIONS,
    NAV2_FULL_CONFIG_SCHEMA,
    NAV2_LIFECYCLE_ACTIONS,
    nav2_tool_definition,
)


class Nav2ContractTest(unittest.TestCase):
    def test_public_identity_lifecycle_and_topics(self) -> None:
        tool = nav2_tool_definition("ubuntu")

        self.assertEqual(tool["name"], "nav2")
        self.assertEqual(tool["displayName"], "Nav2")
        self.assertEqual(tool["type"], "processor")
        self.assertFalse(tool["multiInstance"])
        actions = tool["inputSchema"]["properties"]["action"]["enum"]
        self.assertEqual(actions[:4], list(NAV2_LIFECYCLE_ACTIONS))
        self.assertEqual(actions[4:], list(NAV2_ACTIONS))
        self.assertEqual(
            actions[4:],
            [
                "navigate_to_pose",
                "wait_navigation_done",
                "pause_nav",
                "resume_nav",
                "stop_nav",
            ],
        )

        inputs = {item["port"]: item for item in tool["topic_in"]}
        self.assertEqual(
            set(inputs),
            {"livo_odom", "registered_cloud", "static_map", "goal_pose"},
        )
        self.assertEqual(inputs["livo_odom"]["topic"], "/ubuntu/navigation/odom")
        self.assertEqual(
            inputs["registered_cloud"]["topic"],
            "/ubuntu/navigation/cloud_registered",
        )
        self.assertEqual(
            inputs["registered_cloud"]["ros_type"],
            "sensor_msgs/msg/PointCloud2",
        )
        self.assertEqual(
            inputs["registered_cloud"]["qos"],
            "BEST_EFFORT + KEEP_LAST(depth=1) + VOLATILE",
        )
        self.assertEqual(inputs["registered_cloud"]["frame_id"], "map")
        self.assertEqual(inputs["static_map"]["topic"], "/ubuntu/navigation/static_map")
        self.assertEqual(inputs["static_map"]["ros_type"], "nav_msgs/msg/OccupancyGrid")
        self.assertEqual(
            inputs["static_map"]["qos"],
            "RELIABLE + KEEP_LAST(depth=1) + TRANSIENT_LOCAL",
        )
        self.assertEqual(inputs["static_map"]["frame_id"], "map")
        self.assertFalse(inputs["goal_pose"]["required"])

        outputs = {item["port"]: item for item in tool["topic_out"]}
        self.assertEqual(set(outputs), {"velocity_proposal", "plan", "costmap"})
        self.assertEqual(tool["topic_out"][0]["port"], "velocity_proposal")
        proposal = outputs["velocity_proposal"]
        self.assertEqual(proposal["schema"], "phanthy.navigation.velocity_proposal.v1")
        self.assertEqual(proposal["max_age_ms"], 250)
        self.assertEqual(proposal["rate_hz"], 5)
        self.assertEqual(
            proposal["qos"],
            "RELIABLE + KEEP_LAST(depth=1) + VOLATILE",
        )
        plan = outputs["plan"]
        self.assertEqual(plan["topic"], "/plan")
        self.assertEqual(plan["format"], "sensor/path")
        self.assertEqual(plan["ros_type"], "nav_msgs/msg/Path")
        self.assertEqual(plan["schema"], "phanthy.navigation.path.v1")
        costmap = outputs["costmap"]
        self.assertEqual(costmap["topic"], "/global_costmap/costmap")
        self.assertEqual(costmap["format"], "sensor/costmap")
        self.assertEqual(costmap["ros_type"], "nav_msgs/msg/OccupancyGrid")
        self.assertEqual(costmap["schema"], "phanthy.navigation.costmap.v1")
        self.assertTrue(costmap["default_preview"])

        self.assertNotIn("x-execution-control", tool)

        companion_root = (
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "runtime"
            / "g1_nav2"
        )
        launch = (companion_root / "launch" / "g1_nav2.launch.py").read_text(
            encoding="utf-8"
        )
        planner_bridge = (
            companion_root / "g1_nav2" / "planner_command_node.py"
        ).read_text(encoding="utf-8")
        package_xml = (companion_root / "package.xml").read_text(
            encoding="utf-8"
        )
        runtime_dir = companion_root.parent
        dockerfile = (ACTUCORE_ROOT / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )
        service = (ACTUCORE_ROOT / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        source_lock = (runtime_dir / "nav2-source.lock").read_text(
            encoding="utf-8"
        )
        goal_schema = json.loads(
            (runtime_dir / "protocol" / "goal-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        proposal_schema = json.loads(
            (
                runtime_dir
                / "protocol"
                / "velocity-proposal-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn('default_value="/ubuntu/navigation/odom"', launch)
        self.assertIn('default_value="/ubuntu/navigation/cloud_registered"', launch)
        self.assertIn("from nav_msgs.msg import Odometry", planner_bridge)
        self.assertIn("from sensor_msgs.msg import PointCloud2", planner_bridge)
        self.assertNotIn("slam_toolbox", package_xml)
        self.assertNotIn("python3-numpy", dockerfile)
        self.assertNotIn("PYTHON_NUMPY_VERSION", service)
        self.assertNotIn("PYTHON_NUMPY_VERSION", source_lock)
        self.assertEqual(goal_schema["properties"]["speed"]["maximum"], 1.0)
        self.assertEqual(goal_schema["properties"]["speed"]["minimum"], 0.30)
        self.assertEqual(goal_schema["properties"]["speed"]["default"], 0.5)
        self.assertEqual(
            proposal_schema["properties"]["velocity"]["properties"]["x"],
            {"type": "number", "minimum": -1.0, "maximum": 1.0},
        )
        self.assertEqual(
            proposal_schema["properties"]["velocity"]["properties"]["y"],
            {"type": "number", "minimum": -1.0, "maximum": 1.0},
        )

        params = yaml.safe_load(
            (companion_root / "config" / "nav2_params.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            params["global_costmap"]["global_costmap"]["ros__parameters"]
            ["track_unknown_space"]
        )
        self.assertFalse(
            params["planner_server"]["ros__parameters"]["GridBased"]
            ["allow_unknown"]
        )

    def test_config_and_speed_bounds_are_fail_closed(self) -> None:
        tool = nav2_tool_definition("ubuntu")
        properties = tool["inputSchema"]["properties"]
        speed = properties["speed"]
        self.assertEqual(speed["minimum"], 0.30)
        self.assertEqual(speed["maximum"], 1.0)
        self.assertEqual(speed["default"], 0.5)
        self.assertNotIn("mode", properties)

        full_config = NAV2_FULL_CONFIG_SCHEMA["properties"]
        self.assertEqual(full_config["min_x_mps"]["default"], 0.30)
        self.assertEqual(full_config["max_x_mps"]["default"], 1.0)
        self.assertEqual(full_config["min_y_mps"]["default"], 0.0)
        self.assertEqual(full_config["max_y_mps"]["default"], 0.0)
        self.assertEqual(full_config["min_yaw_rps"]["default"], 0.3)
        self.assertEqual(full_config["max_yaw_rps"]["default"], 0.3)
        for field in (
            "min_x_mps",
            "max_x_mps",
            "min_y_mps",
            "max_y_mps",
            "min_yaw_rps",
            "max_yaw_rps",
        ):
            self.assertNotIn("const", full_config[field])

        action_params = tool["inputSchema"]["x-action-params"]
        self.assertNotIn("mode", action_params["navigate_to_pose"]["params"])
        topic_action = tool["x-topic-actions"][0]
        self.assertNotIn("mode", topic_action["allowed_fields"])
        self.assertEqual(topic_action["required_fields"], ["x", "y", "yaw"])

        config = tool["configSchema"]
        self.assertFalse(config["additionalProperties"])
        self.assertEqual(
            set(config["properties"]),
            {
                "backend",
                "request_timeout_sec",
                "discovery_timeout_sec",
                "min_x_mps",
                "max_x_mps",
                "min_y_mps",
                "max_y_mps",
                "min_yaw_rps",
                "max_yaw_rps",
            },
        )

    def test_tool_definition_is_deep_copied(self) -> None:
        first = nav2_tool_definition("ubuntu")
        first["topic_in"][0]["topic"] = "/mutated"
        second = nav2_tool_definition("ubuntu")
        self.assertEqual(
            second["topic_in"][0]["topic"], "/ubuntu/navigation/odom"
        )

    def test_g1_test_container_script_has_no_canvas_auth_gate(self) -> None:
        script = (
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "deploy"
            / "scripts"
            / "owner-start-g1-test-containers.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("CORE_ACCESS_TOKEN", script)
        self.assertNotIn("/api/config/project-running", script)
        self.assertNotIn("require_canvas_stopped", script)
        self.assertIn('OWNER_LABEL="com.phanthymotus.test-owner"', script)
        # ActuCore 只占一个 MCP 端口；15721 是 perception 的 WebSocket ASR，
        # 卡片迁走后不该再出现在这里。
        self.assertIn("require_port_free 15730", script)
        self.assertNotIn("require_port_free 15720", script)
        self.assertNotIn("require_port_free 15721", script)
        self.assertNotIn("FAST_LIVO2_CONTAINER", script)
        self.assertNotIn("NAV2_CONTAINER", script)


if __name__ == "__main__":
    unittest.main()
