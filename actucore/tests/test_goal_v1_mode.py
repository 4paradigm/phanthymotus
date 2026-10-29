"""Goal v1 mode agreement across protocol and navigation tool contracts."""

import json
import unittest
from pathlib import Path

import jsonschema

from plugins.navigation.contract import navigation_tool_definition
from plugins.navigation.planning.contract import NAV2_ACTION_PARAMS, nav2_tool_definition
from plugins.navigation.planning.core import NavigationBackendError, _normalize


class GoalV1ModeTest(unittest.TestCase):
    def test_protocol_and_mcp_accept_only_zero_mode(self):
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "plugins/navigation/runtime/protocol/goal-v1.schema.json"
        )
        goal_schema = json.loads(protocol_path.read_text(encoding="utf-8"))
        planning = nav2_tool_definition("ubuntu")["inputSchema"]
        public = navigation_tool_definition("ubuntu")["inputSchema"]
        expected = goal_schema["properties"]["mode"]
        for schema in (planning, public):
            mode = schema["properties"]["mode"]
            self.assertEqual({k: mode[k] for k in expected}, expected)
            self.assertIn("mode", schema["x-action-params"]["navigate_to_pose"]["params"])
            self.assertIn("mode", schema["x-topic-actions"][0]["allowed_fields"])
            self.assertNotIn("mode", schema["x-topic-actions"][0]["required_fields"])
        self.assertIn("mode", NAV2_ACTION_PARAMS["navigate_to_pose"]["params"])
        self.assertFalse(public["additionalProperties"])

        goal = {"schema": "phanthy.navigation.goal.v1", "goal_id": "g-1",
                "x": 1, "y": 2, "yaw": 0}
        call = {"action": "navigate_to_pose", "x": 1, "y": 2, "yaw": 0}
        for schema, base in ((goal_schema, goal), (public, call)):
            validator = jsonschema.Draft202012Validator(schema)
            for value in (0, 0.0, -0.0):
                with self.subTest(schema=schema.get("title", "MCP"), value=value):
                    self.assertTrue(validator.is_valid({**base, "mode": value}))
            self.assertTrue(validator.is_valid(base), "mode remains optional")
            for value in (False, True, None, "0", 1, -1, 0.1, [], {}):
                with self.subTest(schema=schema.get("title", "MCP"), value=value):
                    self.assertFalse(validator.is_valid({**base, "mode": value}))

    def test_planner_integer_and_legacy_rejections(self):
        base = {"x": 1, "y": 2, "yaw": 0}
        self.assertEqual(_normalize("navigate_to_pose", base)["mode"], 0)
        self.assertEqual(_normalize("navigate_to_pose", {**base, "mode": 0})["mode"], 0)
        for value in (False, True, None, "0", 1, -1, 0.1, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(NavigationBackendError):
                    _normalize("navigate_to_pose", {**base, "mode": value})

    def test_planner_accepts_json_schema_integer_float_zero(self):
        base = {"x": 1, "y": 2, "yaw": 0}
        for value in (0.0, -0.0):
            with self.subTest(value=value):
                normalized = _normalize("navigate_to_pose", {**base, "mode": value})
                self.assertEqual(normalized["mode"], 0)
                self.assertIs(type(normalized["mode"]), int)


if __name__ == "__main__":
    unittest.main()
