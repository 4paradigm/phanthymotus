"""Canvas teardown must not release an unconfirmed active navigation."""
from __future__ import annotations

import pytest

from tests.test_nav2_plugin import Nav2Plugin, ReadyBackend, _bindings


class StopResultBackend(ReadyBackend):
    def __init__(self, stop_result):
        super().__init__()
        self.stop_result = stop_result

    def execute(self, action, args, *, nav_id):
        if action == "stop_nav":
            self.execute_calls.append((action, dict(args), nav_id))
            return dict(self.stop_result)
        return super().execute(action, args, nav_id=nav_id)


def _started_plugin(stop_result):
    backend = StopResultBackend(stop_result)
    plugin = Nav2Plugin({}, None, backend=backend)
    assert plugin.dispatch("nav2", {"action": "start", "input_bindings": _bindings(plugin)})["state"] == "ready"
    assert plugin.dispatch("nav2", {"action": "navigate_to_pose", "x": 0, "y": 0, "yaw": 0})["status"] == "navigating"
    return plugin, backend


@pytest.mark.parametrize("stop_result", [
    {},
    {"status": "ok"},
    {"status": "cancel_unconfirmed"},
    {"status": "stopped"},
    {"status": "stopped", "terminal_confirmed": False},
    {"status": "stopped", "terminal_confirmed": 1},
    {"status": "stopped", "already_terminal": "arrived"},
])
def test_unconfirmed_stop_keeps_backend_and_canvas(stop_result):
    plugin, backend = _started_plugin(stop_result)
    result = plugin.dispatch("nav2", {"action": "stop"})
    assert result["error_code"] == "navigation_stop_pending"
    assert backend.stop_calls == 0
    assert plugin._canvas_started
    assert plugin._core is not None


@pytest.mark.parametrize("stop_result", [
    {"status": "stopped", "terminal_confirmed": True},
    {"status": "arrived", "already_terminal": "arrived"},
    {"status": "stopped", "already_terminal": "stopped"},
])
def test_confirmed_terminal_allows_canvas_teardown(stop_result):
    plugin, backend = _started_plugin(stop_result)
    result = plugin.dispatch("nav2", {"action": "stop"})
    assert result["state"] == "idle"
    assert backend.stop_calls == 1
    assert not plugin._canvas_started
