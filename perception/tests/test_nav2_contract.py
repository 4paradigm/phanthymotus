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

        inputs = {item["port"]: item for item in tool["topic_in"]}
        self.assertEqual(set(inputs), {"loco_state", "lidar_cloud", "goal_pose"})
        self.assertEqual(inputs["loco_state"]["topic"], "/ubuntu/loco/state")
        self.assertEqual(inputs["lidar_cloud"]["topic"], "/ubuntu/lidar/cloud")
        self.assertIn("BEST_EFFORT", inputs["lidar_cloud"]["qos"])
        self.assertFalse(inputs["goal_pose"]["required"])

        self.assertEqual(len(tool["topic_out"]), 1)
        output = tool["topic_out"][0]
        self.assertEqual(output["port"], "velocity_proposal")
        self.assertEqual(output["schema"], "phanthy.navigation.velocity_proposal.v1")
        self.assertEqual(output["max_age_ms"], 250)

    def test_config_and_speed_bounds_are_fail_closed(self) -> None:
        tool = nav2_tool_definition("ubuntu")
        properties = tool["inputSchema"]["properties"]
        speed = properties["speed"]
        self.assertEqual(speed["minimum"], 0.05)
        self.assertEqual(speed["maximum"], 0.15)
        self.assertEqual(speed["default"], 0.15)
        self.assertNotIn("mode", properties)

        full_config = NAV2_FULL_CONFIG_SCHEMA["properties"]
        self.assertEqual(full_config["max_lateral_mps"]["const"], 0.0)
        self.assertEqual(full_config["max_lateral_mps"]["default"], 0.0)

        action_params = tool["inputSchema"]["x-action-params"]
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
