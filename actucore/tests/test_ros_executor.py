"""Exercise the ActuCore ROS startup path when a queued timer is destroyed."""

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("returns_ready", [False, True])
def test_destroyed_queued_timer_does_not_interrupt_other_callbacks(monkeypatch, returns_ready):
    class InvalidHandle(Exception):
        pass

    dispatched = []

    class Executor:
        def _take_timer(self, timer):
            if timer == "destroyed":
                raise InvalidHandle("cannot use Destroyable because destruction was requested")
            return True if returns_ready else None

        async def _execute_timer(self, timer, ready):
            if not returns_ready or ready:
                dispatched.append(timer)

        async def _dispatch(self):
            for timer in ("before", "destroyed", "after"):
                ready = self._take_timer(timer)
                await self._execute_timer(timer, ready)

        def spin(self):
            asyncio.run(self._dispatch())

        def shutdown(self):
            dispatched.append("shutdown")

    executors = ModuleType("rclpy.executors")
    executors.MultiThreadedExecutor = Executor
    monkeypatch.setitem(sys.modules, "rclpy.executors", executors)
    rclpy = sys.modules["rclpy"]
    monkeypatch.setattr(rclpy, "executors", executors, raising=False)
    monkeypatch.setattr(rclpy, "init", lambda: None, raising=False)
    monkeypatch.setattr(rclpy, "shutdown", lambda: None, raising=False)
    exceptions = ModuleType("rclpy.exceptions")
    exceptions.InvalidHandle = InvalidHandle
    monkeypatch.setitem(sys.modules, "rclpy.exceptions", exceptions)

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "perception" / "utils"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import main

    monkeypatch.setattr(main, "_bundle", None)
    monkeypatch.setattr(main, "_load_config", dict)
    monkeypatch.setattr(main, "ActuCoreBundle", lambda cfg, executor: None)
    monkeypatch.setattr(main, "_start_registration", lambda *args: None)
    monkeypatch.setattr(main, "make_handler", lambda: None)
    monkeypatch.setattr(main, "ThreadingHTTPServer", lambda *args: SimpleNamespace(serve_forever=lambda: None))
    monkeypatch.setattr(main, "signal", SimpleNamespace(SIGTERM=15, SIGINT=2, signal=lambda *args: None))
    monkeypatch.setattr(main, "threading", SimpleNamespace(
        Thread=lambda target, **kwargs: SimpleNamespace(start=target, join=lambda **kwargs: None,
                                                        is_alive=lambda: False)
    ))

    main.main()

    assert dispatched == ["before", "after", "shutdown"]


def test_navi_detaches_from_executor_before_destroying_timers():
    from plugins.navi.plugin import NaviPlugin

    calls = []
    node = SimpleNamespace(
        destroy_timer=lambda timer: calls.append(timer),
        destroy_node=lambda: calls.append("node"),
    )
    executor = SimpleNamespace(remove_node=lambda node: calls.append("detach"))
    card = NaviPlugin(executor=executor, plugin_cfg={})
    card._running = True
    card._node = node
    card._timer = "planning timer"
    card._command_timer = "publication timer"
    card._depth_bind_timer = "depth bind retry"

    assert card._stop() == {"state": "stopped"}
    assert calls == ["detach", "depth bind retry", "planning timer", "publication timer", "node"]
