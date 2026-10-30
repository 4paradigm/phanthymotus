"""Mode agreement across MCP navigation tool contracts."""

import unittest

from plugins.controlled_semantic_spatial.contract import controlled_semantic_spatial_tool_definition
from plugins.controlled_semantic_spatial.planning.contract import NAV2_ACTION_PARAMS, nav2_tool_definition
from plugins.controlled_semantic_spatial.planning.core import NavigationBackendError, _normalize


class GoalV1ModeTest(unittest.TestCase):
    def test_mcp_defines_optional_zero_mode(self):
        planning = nav2_tool_definition("ubuntu")["inputSchema"]
        public = controlled_semantic_spatial_tool_definition("ubuntu")["inputSchema"]
        # Assert the actual schema contract rather than reimplementing a JSON
        # Schema validator; the planner's value handling is tested below.
        expected = {"type": "integer", "const": 0, "default": 0}
        for schema in (planning, public):
            mode = schema["properties"]["mode"]
            self.assertEqual({k: v for k, v in mode.items() if k != "description"}, expected)
            self.assertIs(type(mode["const"]), int)
            self.assertIs(type(mode["default"]), int)
            self.assertNotIn("mode", schema.get("required", []))
        for schema in (planning, public):
            self.assertIn("mode", schema["x-action-params"]["navigate_to_pose"]["params"])
            self.assertEqual(schema.get("x-topic-actions", []), [])
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
