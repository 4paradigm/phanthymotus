import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]


def _load_main_module():
    rclpy = types.ModuleType("rclpy")
    rclpy.executors = types.ModuleType("rclpy.executors")
    yaml = types.ModuleType("yaml")
    yaml.safe_load = mock.Mock()
    with mock.patch.dict(
        sys.modules,
        {"rclpy": rclpy, "rclpy.executors": rclpy.executors, "yaml": yaml},
    ):
        spec = importlib.util.spec_from_file_location(
            "perception_main_lifecycle", PERCEPTION_ROOT / "main.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class MainLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _load_main_module()

    def test_spin_failure_is_recorded_and_stops_http_server(self):
        failure = RuntimeError("spin failed")
        executor = mock.Mock()
        executor.spin.side_effect = failure
        server = mock.Mock()
        errors = []

        with self.assertLogs("perception_main_lifecycle", level="ERROR"):
            self.main._spin_executor(executor, server, errors)

        self.assertEqual(errors, [failure])
        server.shutdown.assert_called_once_with()

    def test_bundle_runs_optional_shutdown_phases(self):
        lifecycle_plugin = mock.Mock()
        legacy_plugin = types.SimpleNamespace(PREFIX="legacy")
        bundle = object.__new__(self.main.PerceptionBundle)
        bundle._plugins = [lifecycle_plugin, legacy_plugin]

        bundle.prepare_shutdown()
        bundle.destroy_nodes()

        lifecycle_plugin.prepare_shutdown.assert_called_once_with()
        lifecycle_plugin.destroy_nodes.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
