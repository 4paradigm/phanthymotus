from __future__ import annotations

import sys
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.nav2.contract import (  # noqa: E402
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
        self.assertIn("switch_runtime_mode", actions)

        inputs = {item["port"]: item for item in tool["topic_in"]}
        self.assertEqual(set(inputs), {"loco_state", "lidar_cloud", "goal_pose"})
        self.assertEqual(inputs["loco_state"]["topic"], "/ubuntu/loco/state")
        self.assertEqual(
            inputs["lidar_cloud"]["topic"], "/ubuntu/lidar/cloud"
        )
        self.assertEqual(
            inputs["lidar_cloud"]["schema"],
            "phanthy.g1.lidar_cloud.v2",
        )
        self.assertEqual(
            inputs["lidar_cloud"]["ros_type"],
            "std_msgs/msg/UInt8MultiArray",
        )
        self.assertEqual(
            inputs["lidar_cloud"]["qos"],
            "BEST_EFFORT + KEEP_LAST(depth=1) + VOLATILE",
        )
        self.assertNotIn("compatible_schemas", inputs["lidar_cloud"])
        self.assertIn("BEST_EFFORT", inputs["lidar_cloud"]["qos"])
        self.assertFalse(inputs["goal_pose"]["required"])

        outputs = {item["port"]: item for item in tool["topic_out"]}
        self.assertEqual(set(outputs), {"velocity_proposal", "map_view"})
        self.assertEqual(tool["topic_out"][0]["port"], "velocity_proposal")
        self.assertEqual(tool["topic_out"][1]["port"], "map_view")
        proposal = outputs["velocity_proposal"]
        self.assertEqual(proposal["schema"], "phanthy.navigation.velocity_proposal.v1")
        self.assertEqual(proposal["max_age_ms"], 250)
        map_view = outputs["map_view"]
        self.assertEqual(map_view["topic"], "/ubuntu/navigation/nav2/map_view")
        self.assertEqual(map_view["format"], "sensor/mapping")
        self.assertEqual(map_view["ros_type"], "std_msgs/msg/UInt8MultiArray")
        self.assertTrue(map_view["default_preview"])

        companion_root = (
            PERCEPTION_ROOT
            / "plugins"
            / "nav2"
            / "companion"
            / "g1_nav2"
        )
        launch = (companion_root / "launch" / "g1_nav2.launch.py").read_text(
            encoding="utf-8"
        )
        bridge = (
            companion_root / "g1_nav2" / "canvas_pointcloud_node.py"
        ).read_text(encoding="utf-8")
        odom_bridge = (
            companion_root / "g1_nav2" / "loco_odom_node.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'default_value="/ubuntu/lidar/cloud"', launch
        )
        self.assertIn(
            'self.declare_parameter("input_topic", "/ubuntu/lidar/cloud")',
            bridge,
        )
        self.assertIn('default_value="livox_frame"', launch)
        self.assertIn("UInt8MultiArray", bridge)
        self.assertIn("decode_canvas_pointcloud", bridge)
        self.assertIn('"metadata_footer": "PCLMETA2"', bridge)
        self.assertIn("LidarClockNormalizer", bridge)
        self.assertIn("self._lidar_clock = LidarClockNormalizer", bridge)
        self.assertNotIn("self._clock = LidarClockNormalizer", bridge)
        self.assertIn('"timestamp_mode": "auto"', launch)
        self.assertIn("expected exactly one publisher", bridge)
        self.assertIn("expected exactly one publisher", odom_bridge)
        self.assertIn('"queue_size": 1', launch)

        canvas = (
            PERCEPTION_ROOT.parent / "agent-core" / "web" / "js" / "canvas.js"
        ).read_text(encoding="utf-8")
        self.assertIn("t.default_preview === true", canvas)

    def test_config_and_speed_bounds_are_fail_closed(self) -> None:
        tool = nav2_tool_definition("ubuntu")
        properties = tool["inputSchema"]["properties"]
        speed = properties["speed"]
        self.assertEqual(speed["minimum"], 0.10)
        self.assertEqual(speed["maximum"], 0.15)
        self.assertEqual(speed["default"], 0.15)
        self.assertNotIn("mode", properties)
        self.assertEqual(
            properties["runtime_mode"]["enum"], ["mapping", "localization"]
        )

        full_config = NAV2_FULL_CONFIG_SCHEMA["properties"]
        self.assertEqual(full_config["max_lateral_mps"]["const"], 0.0)
        self.assertEqual(full_config["max_lateral_mps"]["default"], 0.0)
        self.assertEqual(full_config["max_reverse_mps"]["const"], 0.15)
        self.assertEqual(full_config["max_reverse_mps"]["default"], 0.15)

        action_params = tool["inputSchema"]["x-action-params"]
        self.assertEqual(
            action_params["switch_runtime_mode"]["params"],
            ["runtime_mode", "map_name"],
        )
        self.assertNotIn("mode", action_params["navigate_to_tag"]["params"])
        self.assertNotIn("mode", action_params["navigate_to_pose"]["params"])
        topic_action = tool["x-topic-actions"][0]
        self.assertNotIn("mode", topic_action["allowed_fields"])

        config = tool["configSchema"]
        self.assertFalse(config["additionalProperties"])
        self.assertEqual(
            set(config["properties"]),
            {
                "backend",
                "request_timeout_sec",
                "runtime_switch_timeout_sec",
                "discovery_timeout_sec",
            },
        )

    def test_tool_definition_is_deep_copied(self) -> None:
        first = nav2_tool_definition("ubuntu")
        first["topic_in"][0]["topic"] = "/mutated"
        second = nav2_tool_definition("ubuntu")
        self.assertEqual(second["topic_in"][0]["topic"], "/ubuntu/loco/state")


if __name__ == "__main__":
    unittest.main()
