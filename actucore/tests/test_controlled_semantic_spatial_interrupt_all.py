"""Hook metadata and cancellation dispatch tests using fake backends, not ROS."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.controlled_semantic_spatial.contract import controlled_semantic_spatial_tool_definition  # noqa: E402
from plugins.controlled_semantic_spatial.planning.contract import nav2_tool_definition  # noqa: E402
from plugins.controlled_semantic_spatial.planning.plugin import Nav2Plugin  # noqa: E402
from plugins.controlled_semantic_spatial.plugin import ControlledSemanticSpatialPlugin  # noqa: E402


class FakeBackend:
    def __init__(self, *, confirm=True):
        self.confirm = confirm
        self.calls = []
        self.callback = None

    def set_terminal_callback(self, callback):
        self.callback = callback

    def info(self):
        return {"state": "ready", "bridge_subscribers": 1, "navigation_ready": True}

    def execute(self, action, args, *, nav_id):
        self.calls.append((action, nav_id))
        if action == "navigate_to_pose":
            return {"status": "navigating"}
        if action == "stop_nav":
            if not self.confirm:
                return {"status": "error", "error_code": "stop_unconfirmed"}
            self.callback({"nav_id": nav_id, "status": "cancelled"})
            return {"status": "stopped", "terminal_confirmed": True}
        raise AssertionError(action)

    def stop(self):
        pass


def _goal(nav_id="goal-1"):
    return {"action": "navigate_to_pose", "x": 1, "y": 2, "yaw": 0,
            "_control_nav_id": nav_id}


def _started_planner(backend, completions):
    planner = Nav2Plugin({}, None, backend=backend)
    planner.set_completion_callback(completions.append)
    bindings = [{"port": item["port"], "topic": item["topic"]}
                for item in planner.get_tools()[0]["topic_in"]
                if item.get("required", True)]
    assert planner.dispatch("nav2", {"action": "start", "input_bindings": bindings})["state"] == "ready"
    return planner


def test_both_cards_expose_same_standard_hook_and_keep_navigation_hook():
    planning = nav2_tool_definition("ubuntu")["inputSchema"]["x-hooks"]
    unified = controlled_semantic_spatial_tool_definition("ubuntu")["inputSchema"]["x-hooks"]
    assert planning["on_interrupt_all"] == {"action": "stop_nav"}
    assert planning["on_interrupt_navigation"] == {"action": "stop_nav"}
    assert unified == planning


def test_planning_hook_uses_existing_cancel_path_and_emits_terminal_once():
    backend, completions = FakeBackend(), []
    planner = _started_planner(backend, completions)
    assert planner.dispatch("nav2", _goal())["status"] == "navigating"
    hook = planner.get_tools()[0]["inputSchema"]["x-hooks"]["on_interrupt_all"]
    stopped = planner.dispatch("nav2", dict(hook))
    assert stopped["status"] == "stopped"
    assert backend.calls == [("navigate_to_pose", "goal-1"), ("stop_nav", "goal-1")]
    assert [event["action_id"] for event in completions] == ["goal-1"]
    assert completions[0]["status"] == "cancelled"
    backend.callback({"nav_id": "goal-1", "status": "cancelled"})
    assert planner.dispatch("nav2", dict(hook))["already_idle"] is True
    assert len(completions) == 1


def test_unconfirmed_interrupt_retains_goal_and_does_not_complete():
    backend, completions = FakeBackend(confirm=False), []
    planner = _started_planner(backend, completions)
    assert planner.dispatch("nav2", _goal())["status"] == "navigating"
    hook = planner.get_tools()[0]["inputSchema"]["x-hooks"]["on_interrupt_all"]
    result = planner.dispatch("nav2", dict(hook))
    assert result["error_code"] == "navigation_stop_pending"
    assert completions == []
    assert planner.dispatch("nav2", _goal("goal-2"))["error_code"] == "navigation_active"


def test_unified_dispatch_routes_hook_through_generation_fenced_planner():
    class Runtime:
        def info(self):
            return {"running": True}

    class Planner:
        def __init__(self):
            self.calls = []

        def dispatch(self, name, args):
            self.calls.append((name, dict(args)))
            return {"status": "stopped", "terminal_confirmed": True}

    planner = Planner()
    plugin = ControlledSemanticSpatialPlugin({}, "ubuntu", None, runtime=Runtime(),
                              planning_plugin=planner)
    plugin._started = True
    plugin._generation = object()
    hook = plugin.get_tools()[0]["inputSchema"]["x-hooks"]["on_interrupt_all"]
    assert plugin.dispatch(plugin.PREFIX, dict(hook))["status"] == "stopped"
    assert planner.calls == [("nav2", {"action": "stop_nav"})]
    plugin._stop_requested.set()
    assert plugin.dispatch(plugin.PREFIX, dict(hook))["error_code"] == "navigation_stopping"
    assert len(planner.calls) == 1
