from __future__ import annotations

import json
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
sys.path.insert(1, str(CORE_ROOT.parent / "actucore"))

config = types.ModuleType("config")
config.main = {}

# The host-side unit-test environment intentionally does not install Agent
# Core's HTTP runtime dependencies.  topic_actions only needs the registry.
mcp_client = types.ModuleType("mcp_client")
mcp_client.registry = {}

jsonschema = types.ModuleType("jsonschema")


class ValidationError(Exception):
    pass


def validate(instance, schema):
    for field in schema.get("required", []):
        if field not in instance:
            raise ValidationError(f"{field} is required")
    for field, value in instance.items():
        field_schema = schema.get("properties", {}).get(field, {})
        if field_schema.get("type") == "number" and isinstance(value, bool):
            raise ValidationError(f"{field} is not a number")
        if "minimum" in field_schema and value < field_schema["minimum"]:
            raise ValidationError(f"{field} is below minimum")
        if "maximum" in field_schema and value > field_schema["maximum"]:
            raise ValidationError(f"{field} is above maximum")


jsonschema.ValidationError = ValidationError
jsonschema.validate = validate

module_name = "topic_actions_under_test"
spec = importlib.util.spec_from_file_location(
    module_name, CORE_ROOT / "src" / "topic_actions.py"
)
assert spec and spec.loader
topic_actions = importlib.util.module_from_spec(spec)
with patch.dict(
    sys.modules,
    {
        "config": config,
        "mcp_client": mcp_client,
        "jsonschema": jsonschema,
        module_name: topic_actions,
    },
):
    spec.loader.exec_module(topic_actions)


def _tool_definition() -> dict:
    return {
        "name": "nav2",
        "topic_in": [
            {"port": "odom", "topic": "/odom", "format": "sensor/odometry"},
            {
                "port": "goal_pose",
                "topic": "/ubuntu/navigation/goal_pose",
                "format": "data/json",
            },
        ],
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate_to_pose"],
                },
                "instance_id": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "yaw": {"type": "number"},
                "speed": {"type": "number", "minimum": 0.3, "maximum": 1.0},
            },
            "required": ["action"],
            "x-topic-actions": [
                {
                    "port": "goal_pose",
                    "action": "navigate_to_pose",
                    "schema": "phanthy.navigation.goal.v1",
                    "id_field": "goal_id",
                    "id_argument": "_control_nav_id",
                    "allowed_fields": ["x", "y", "yaw", "speed"],
                    "required_fields": ["x", "y", "yaw"],
                }
            ],
        },
    }


def _layout() -> dict:
    return {
        "cards": [
            {
                "id": "goal-source",
                "mcpId": "goal-source",
                "toolName": "goal-source",
                "topicOut": [
                    {
                        "port": "goal_pose",
                        "topic": "/ubuntu/navigation/goal_pose",
                        "format": "data/json",
                    }
                ],
            },
            {
                "id": "nav2-card",
                "mcpId": "perception",
                "toolName": "nav2",
            },
        ],
        "connections": [
            {
                "fromCardId": "goal-source",
                "fromPortIdx": "0",
                "fromTopic": "/ubuntu/navigation/goal_pose",
                "toCardId": "nav2-card",
                "toPortIdx": "1",
                "format": "data/json",
            }
        ],
    }


class TopicActionsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.original_registry = dict(mcp_client.registry)
        config.main = {"core": {"project_running": True}, "services": {"mcp": []}}
        mcp_client.registry.clear()
        mcp_client.registry["perception"] = {
            "tool_definitions": [_tool_definition()]
        }
        self.manager = topic_actions.TopicActionManager()

    async def asyncTearDown(self) -> None:
        await self.manager.stop()
        mcp_client.registry.clear()
        mcp_client.registry.update(self.original_registry)

    def test_routes_require_a_real_canvas_connection(self) -> None:
        self.assertEqual(topic_actions.build_routes({"cards": [], "connections": []}), [])

        layout = _layout()
        layout["connections"] = []
        self.assertEqual(topic_actions.build_routes(layout), [])

    def test_top_level_topic_action_extension_is_not_a_valid_route(self) -> None:
        tool = _tool_definition()
        tool["x-topic-actions"] = tool["inputSchema"].pop("x-topic-actions")
        mcp_client.registry["perception"] = {"tool_definitions": [tool]}

        self.assertEqual(topic_actions.build_routes(_layout()), [])

    def test_route_resolves_declared_port_and_topic(self) -> None:
        routes = topic_actions.build_routes(_layout())

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].port, "goal_pose")
        self.assertEqual(routes[0].topic, "/ubuntu/navigation/goal_pose")
        self.assertEqual(routes[0].action, "navigate_to_pose")
        self.assertEqual(routes[0].required_fields, ("x", "y", "yaw"))
        self.assertEqual(routes[0].id_argument, "_control_nav_id")

    def test_live_source_topic_wins_over_stale_connection_cache(self) -> None:
        layout = _layout()
        layout["cards"][0]["topicOut"][0]["topic"] = "/live/goal_pose"

        routes = topic_actions.build_routes(layout)

        self.assertEqual(routes[0].topic, "/live/goal_pose")

    def test_hot_path_logs_are_sampled_and_escaped(self) -> None:
        runtime = topic_actions.RouteRuntime(
            route=topic_actions.build_routes(_layout())[0]
        )
        with patch("builtins.print") as log:
            for _ in range(101):
                topic_actions._log_hot_path(runtime, "rejected", "bad\n\x1b[31m")

        lines = [call.args[0] for call in log.call_args_list]
        self.assertEqual(len(lines), 2)
        self.assertIn("count=1", lines[0])
        self.assertIn("count=100", lines[1])
        self.assertIn(r"bad\n\x1b[31m", lines[0])
        self.assertNotIn("\n", lines[0])
        self.assertNotIn("\x1b", lines[0])

    def test_unified_navigation_contract_builds_goal_route(self) -> None:
        from plugins.navigation.contract import navigation_tool_definition

        tool = navigation_tool_definition("ubuntu")
        mcp_client.registry["perception"] = {"tool_definitions": [tool]}
        goal_index = next(
            index
            for index, item in enumerate(tool["topic_in"])
            if item["port"] == "goal_pose"
        )
        layout = _layout()
        layout["cards"][1]["toolName"] = tool["name"]
        layout["connections"][0]["toPortIdx"] = str(goal_index)

        routes = topic_actions.build_routes(layout)

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].action, "navigate_to_pose")
        self.assertEqual(routes[0].schema, "phanthy.navigation.goal.v1")
        self.assertEqual(routes[0].id_argument, "_control_nav_id")

    async def test_invalid_live_route_change_keeps_current_subscription(self) -> None:
        with patch.object(topic_actions.ros2_bridge, "subscribe"), patch.object(
            topic_actions.ros2_bridge, "unsubscribe"
        ) as unsubscribe:
            await self.manager.start(_layout())
            active_keys = set(self.manager._runtimes)
            invalid = _layout()
            invalid["connections"][0]["format"] = "raw"

            with self.assertRaisesRegex(ValueError, "requires data/json"):
                await self.manager.start(invalid)

            self.assertEqual(set(self.manager._runtimes), active_keys)
            unsubscribe.assert_not_called()

    async def test_valid_goal_dispatches_once_and_stop_unsubscribes(self) -> None:
        callbacks = {}
        dispatched = []

        def subscribe(key, topic, fmt, loop, callback):
            callbacks[key] = callback

        async def call_tool(mcp_id, request):
            dispatched.append((mcp_id, request.tool, request.arguments))
            return {"code": 200, "data": []}

        class Request:
            def __init__(self, tool, arguments):
                self.tool = tool
                self.arguments = arguments

        fake_api = types.ModuleType("api.mcp_manage")
        fake_api.MCPCallRequest = Request
        fake_api.mcp_call_tool = call_tool

        with patch.object(topic_actions.ros2_bridge, "subscribe", side_effect=subscribe), patch.object(
            topic_actions.ros2_bridge, "unsubscribe"
        ) as unsubscribe, patch.dict(sys.modules, {"api.mcp_manage": fake_api}), patch(
            "builtins.print"
        ) as log:
            await self.manager.start(_layout())
            callback = next(iter(callbacks.values()))
            payload = {
                "schema": "phanthy.navigation.goal.v1",
                "goal_id": "goal-001",
                "x": 1.2,
                "y": -0.8,
                "yaw": 0.1,
                "speed": 0.5,
            }
            await callback(json.dumps(payload).encode(), "data/json")
            await callback(json.dumps(payload).encode(), "data/json")

            self.assertEqual(
                dispatched,
                [
                    (
                        "perception",
                        "nav2",
                        {
                            "action": "navigate_to_pose",
                            "instance_id": "nav2-card",
                            "x": 1.2,
                            "y": -0.8,
                            "yaw": 0.1,
                            "speed": 0.5,
                            "_control_nav_id": "goal-001",
                        },
                    )
                ],
            )
            stats = next(iter(self.manager.snapshot().values()))
            self.assertEqual(stats["received"], 2)
            self.assertEqual(stats["dispatched"], 1)
            self.assertEqual(stats["duplicates"], 1)
            self.assertNotIn(
                "goal-001", "\n".join(call.args[0] for call in log.call_args_list)
            )

            await self.manager.stop()
            unsubscribe.assert_called_once_with("__topic_action__#nav2-card#goal_pose")

    async def test_invalid_messages_never_dispatch(self) -> None:
        callbacks = {}

        def subscribe(key, topic, fmt, loop, callback):
            callbacks[key] = callback

        with patch.object(topic_actions.ros2_bridge, "subscribe", side_effect=subscribe), patch.object(
            topic_actions.ros2_bridge, "unsubscribe"
        ):
            await self.manager.start(_layout())
            callback = next(iter(callbacks.values()))
            invalid_payloads = [
                {"schema": "wrong", "goal_id": "a", "x": 1, "y": 2, "yaw": 0},
                {"schema": "phanthy.navigation.goal.v1", "goal_id": "b", "x": 1, "yaw": 0},
                {
                    "schema": "phanthy.navigation.goal.v1",
                    "goal_id": "c",
                    "x": 1,
                    "y": 2,
                    "yaw": 0,
                    "mode": 0,
                },
                {
                    "schema": "phanthy.navigation.goal.v1",
                    "goal_id": "d",
                    "x": 1,
                    "y": 2,
                    "yaw": 0,
                    "speed": 2.0,
                },
            ]
            for payload in invalid_payloads:
                await callback(json.dumps(payload).encode(), "data/json")

            stats = next(iter(self.manager.snapshot().values()))
            self.assertEqual(stats["received"], 4)
            self.assertEqual(stats["invalid"], 4)
            self.assertEqual(stats["dispatched"], 0)

    async def test_failed_delivery_can_retry_the_same_goal_id(self) -> None:
        callbacks = {}
        attempts = []

        def subscribe(key, topic, fmt, loop, callback):
            callbacks[key] = callback

        async def call_tool(mcp_id, request):
            attempts.append(request.arguments["x"])
            if len(attempts) == 1:
                return {
                    "code": 200,
                    "data": [{
                        "type": "text",
                        "text": json.dumps({
                            "ok": False,
                            "status": "error",
                            "error": "temporary domain failure",
                        }),
                    }],
                }
            return {"code": 200, "data": []}

        class Request:
            def __init__(self, tool, arguments):
                self.tool = tool
                self.arguments = arguments

        fake_api = types.ModuleType("api.mcp_manage")
        fake_api.MCPCallRequest = Request
        fake_api.mcp_call_tool = call_tool
        payload = {
            "schema": "phanthy.navigation.goal.v1",
            "goal_id": "retry-001",
            "x": 1.0,
            "y": 2.0,
            "yaw": 0.0,
        }

        with patch.object(
            topic_actions.ros2_bridge, "subscribe", side_effect=subscribe
        ), patch.object(
            topic_actions.ros2_bridge, "unsubscribe"
        ), patch.dict(
            sys.modules, {"api.mcp_manage": fake_api}
        ), patch("builtins.print"):
            await self.manager.start(_layout())
            callback = next(iter(callbacks.values()))
            encoded = json.dumps(payload).encode()
            await callback(encoded, "data/json")
            await callback(encoded, "data/json")
            await callback(encoded, "data/json")

        stats = next(iter(self.manager.snapshot().values()))
        self.assertEqual(attempts, [1.0, 1.0])
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["dispatched"], 1)
        self.assertEqual(stats["duplicates"], 1)


if __name__ == "__main__":
    unittest.main()
