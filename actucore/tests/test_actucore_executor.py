from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))


class InvalidHandle(Exception):
    pass


def _load_main_module():
    import utils.logsafe as logsafe

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
            bundle.stop()

        self.assertEqual(plugin.calls, 3)
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
