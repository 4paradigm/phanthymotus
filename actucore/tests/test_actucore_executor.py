from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))
sys.path.insert(0, str(ACTUCORE_ROOT.parent / "perception" / "utils"))


class InvalidHandle(Exception):
    pass


def _load_main_module():
    import logsafe

    rclpy = types.ModuleType("rclpy")
    executors = types.ModuleType("rclpy.executors")
    bindings = types.ModuleType("rclpy._rclpy_pybind11")
    bindings.InvalidHandle = InvalidHandle
    rclpy.executors = executors
    rclpy.ok = lambda: True
    spec = importlib.util.spec_from_file_location(
        "actucore_main_executor_test", ACTUCORE_ROOT / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(logsafe, "install"), mock.patch.dict(
        sys.modules,
        {
            "rclpy": rclpy,
            "rclpy.executors": executors,
            "rclpy._rclpy_pybind11": bindings,
        },
    ):
        spec.loader.exec_module(module)
    return module


class SpinExecutorTest(unittest.TestCase):
    def test_invalid_handle_during_node_teardown_does_not_kill_executor(self):
        module = _load_main_module()

        class Executor:
            calls = 0

            def spin(self):
                self.calls += 1
                if self.calls == 1:
                    raise InvalidHandle("node was destroyed")

        executor = Executor()
        module._spin_executor(executor)

        self.assertEqual(executor.calls, 2)

    def test_invalid_handle_during_shutdown_is_not_retried(self):
        module = _load_main_module()
        module.rclpy.ok = lambda: False

        class Executor:
            calls = 0

            def spin(self):
                self.calls += 1
                raise InvalidHandle("context is shutting down")

        executor = Executor()
        module._spin_executor(executor)

        self.assertEqual(executor.calls, 1)

    def test_main_stops_cards_before_joining_spin_and_destroying_context(self):
        module = _load_main_module()
        for confirmed in (True, False):
            with self.subTest(confirmed=confirmed):
                events = []
                executor = mock.Mock()
                executor.shutdown.side_effect = lambda: events.append("executor")
                bundle = mock.Mock()
                bundle.stop.side_effect = lambda: events.append("cards") or confirmed
                thread = mock.Mock()
                thread.join.side_effect = lambda **kw: events.append("join")
                thread.is_alive.return_value = False
                with mock.patch.object(module, "_load_config", return_value={}), \
                     mock.patch.object(module.rclpy, "init", create=True), \
                     mock.patch.object(module.rclpy, "shutdown", create=True,
                                       side_effect=lambda: events.append("context")), \
                     mock.patch.object(module.rclpy.executors, "MultiThreadedExecutor",
                                       create=True, return_value=executor), \
                     mock.patch.object(module, "ActuCoreBundle", return_value=bundle), \
                     mock.patch.object(module.threading, "Thread", return_value=thread) as start, \
                     mock.patch.object(module, "_start_registration"), \
                     mock.patch.object(module, "ThreadingHTTPServer"), \
                     mock.patch.object(module.signal, "signal"):
                    module.main()
                self.assertEqual(events, ["cards", "executor", "join", "context"])
                self.assertIs(start.call_args.kwargs["target"], module._spin_executor)
                thread.join.assert_called_once_with(timeout=5.0)

    def test_bundle_shutdown_retries_a_retryable_card_stop(self):
        module = _load_main_module()

        class Plugin:
            PREFIX = "navigation"

            def __init__(self):
                self.calls = 0

            def stop(self):
                self.calls += 1
                if self.calls < 3:
                    return {"state": "error", "retryable": True}
                return {"state": "idle"}

        plugin = Plugin()
        bundle = module.ActuCoreBundle.__new__(module.ActuCoreBundle)
        bundle._plugins = [plugin]

        with mock.patch.object(module.time, "sleep") as sleep:
            confirmed = bundle.stop()

        self.assertTrue(confirmed)
        self.assertEqual(plugin.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_bundle_shutdown_reports_an_unconfirmed_card_stop(self):
        module = _load_main_module()

        class Plugin:
            PREFIX = "navigation"

            def __init__(self):
                self.calls = 0

            def stop(self):
                self.calls += 1
                return {"state": "error", "retryable": True}

        plugin = Plugin()
        bundle = module.ActuCoreBundle.__new__(module.ActuCoreBundle)
        bundle._plugins = [plugin]

        with mock.patch.object(module.time, "sleep") as sleep:
            confirmed = bundle.stop()

        self.assertFalse(confirmed)
        self.assertEqual(plugin.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_bundle_shutdown_rejects_missing_or_nonterminal_status(self):
        module = _load_main_module()

        for result in (None, {}, {"state": "stopping"}, {"status": "finalizing"}):
            with self.subTest(result=result):
                plugin = mock.Mock(PREFIX="navigation")
                plugin.stop.return_value = result
                bundle = module.ActuCoreBundle.__new__(module.ActuCoreBundle)
                bundle._plugins = [plugin]

                self.assertFalse(bundle.stop())


if __name__ == "__main__":
    unittest.main()
