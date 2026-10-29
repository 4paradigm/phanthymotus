"""Goal v1 mode agreement across protocol and navigation tool contracts."""

import json
import unittest
from pathlib import Path

from plugins.navigation.contract import navigation_tool_definition
from plugins.navigation.planning.contract import NAV2_ACTION_PARAMS, nav2_tool_definition
from plugins.navigation.planning.core import NavigationBackendError, _normalize


class GoalV1ModeTest(unittest.TestCase):
    def test_protocol_and_mcp_define_optional_zero_mode(self):
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "plugins/navigation/runtime/protocol/goal-v1.schema.json"
        )
        goal_schema = json.loads(protocol_path.read_text(encoding="utf-8"))
        planning = nav2_tool_definition("ubuntu")["inputSchema"]
        public = navigation_tool_definition("ubuntu")["inputSchema"]
        # Assert the actual schema contract rather than reimplementing a JSON
        # Schema validator; the planner's value handling is tested below.
        expected = {"type": "integer", "const": 0, "default": 0}
        for schema in (goal_schema, planning, public):
            mode = schema["properties"]["mode"]
            self.assertEqual({k: v for k, v in mode.items() if k != "description"}, expected)
            self.assertIs(type(mode["const"]), int)
            self.assertIs(type(mode["default"]), int)
            self.assertNotIn("mode", schema.get("required", []))
        for schema in (planning, public):
            self.assertIn("mode", schema["x-action-params"]["navigate_to_pose"]["params"])
            self.assertIn("mode", schema["x-topic-actions"][0]["allowed_fields"])
            self.assertNotIn("mode", schema["x-topic-actions"][0]["required_fields"])
        self.assertIn("mode", NAV2_ACTION_PARAMS["navigate_to_pose"]["params"])
        self.assertFalse(public["additionalProperties"])

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
