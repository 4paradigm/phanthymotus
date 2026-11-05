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


class BundleConfigurationTest(unittest.TestCase):
    def test_shipped_config_loads_enabled_card(self):
        module = _load_main_module()
        with mock.patch.dict("os.environ", {}, clear=True):
            cfg = module._load_config()
        card_cfg = cfg["plugins"]["controlled_semantic_spatial"]
        self.assertTrue(card_cfg["enabled"])
        self.assertNotIn("navigation", cfg["plugins"])
        # Isolate this card while using the actual shipped configuration.
        cfg["plugins"] = {"controlled_semantic_spatial": card_cfg}
        executor = object()
        with mock.patch("plugins.controlled_semantic_spatial.ControlledSemanticSpatialPlugin") as plugin:
            bundle = module.ActuCoreBundle(cfg, executor)
        plugin.assert_called_once_with(
            card_cfg, "ubuntu", executor, completion_callback=module.sse_push
        )
        self.assertEqual(bundle._plugins, [plugin.return_value])

    def test_disabled_or_missing_card_is_not_loaded(self):
        module = _load_main_module()
        for plugins in (
            {"controlled_semantic_spatial": {"enabled": False}},
            {},
        ):
            with self.subTest(plugins=plugins), mock.patch(
                "plugins.controlled_semantic_spatial.ControlledSemanticSpatialPlugin"
            ) as plugin:
                bundle = module.ActuCoreBundle({"plugins": plugins}, object())
                self.assertEqual(bundle._plugins, [])
                plugin.assert_not_called()

    def test_legacy_key_is_rejected(self):
        module = _load_main_module()
        with mock.patch("plugins.controlled_semantic_spatial.ControlledSemanticSpatialPlugin") as plugin:
            with self.assertRaisesRegex(ValueError, r"plugins\.navigation was renamed to plugins\.controlled_semantic_spatial"):
                module.ActuCoreBundle({"plugins": {"navigation": {"enabled": True}}}, None)
            plugin.assert_not_called()

    def test_invalid_card_config_is_rejected(self):
        module = _load_main_module()
        for value in (None, [], "enabled"):
            with self.subTest(value=value), mock.patch("plugins.controlled_semantic_spatial.ControlledSemanticSpatialPlugin") as plugin:
                with self.assertRaisesRegex(
                    ValueError, r"plugins\.controlled_semantic_spatial must be an object"
                ):
                    module.ActuCoreBundle({"plugins": {"controlled_semantic_spatial": value}}, None)
                plugin.assert_not_called()

    def test_namespace_validation_and_normalization(self):
        module = _load_main_module()
        key = "controlled_semantic_spatial"
        for namespace in (None, 42, ["ubuntu"]):
            with self.subTest(namespace=namespace), mock.patch(
                "plugins.controlled_semantic_spatial.ControlledSemanticSpatialPlugin"
            ) as plugin:
                with self.assertRaisesRegex(
                    ValueError, r"plugins\.controlled_semantic_spatial\.namespace must be a string"
                ):
                    module.ActuCoreBundle({"plugins": {key: {"enabled": True, "namespace": namespace}}}, None)
                plugin.assert_not_called()
        for namespace in ("", "other", "/other/"):
            with self.subTest(namespace=namespace), mock.patch(
                "plugins.controlled_semantic_spatial.ControlledSemanticSpatialPlugin"
            ) as plugin:
                with self.assertRaisesRegex(ValueError, "requires namespace=ubuntu"):
                    module.ActuCoreBundle({"plugins": {key: {"enabled": True, "namespace": namespace}}}, None)
                plugin.assert_not_called()
        for settings in ({"enabled": True}, {"enabled": True, "namespace": " /ubuntu/ "}):
            with self.subTest(settings=settings), mock.patch(
                "plugins.controlled_semantic_spatial.ControlledSemanticSpatialPlugin"
            ) as plugin:
                module.ActuCoreBundle({"plugins": {key: settings}}, None)
                plugin.assert_called_once_with(settings, "ubuntu", None, completion_callback=module.sse_push)


class BundleDispatchTest(unittest.TestCase):
    def test_longest_prefix_and_self_named_tools(self):
        module = _load_main_module()
        short = mock.Mock(PREFIX="controlled")
        spatial = mock.Mock(PREFIX="controlled_semantic_spatial")
        short.dispatch.return_value = {"plugin": "short"}
        spatial.dispatch.return_value = {"plugin": "spatial"}
        bundle = module.ActuCoreBundle.__new__(module.ActuCoreBundle)
        bundle._plugins = [short, spatial]
        args = {"action": "info"}

        self.assertEqual(bundle.dispatch("controlled_semantic_spatial", args), {"plugin": "spatial"})
        spatial.dispatch.assert_called_once_with("controlled_semantic_spatial", args)
        self.assertEqual(bundle.dispatch("controlled_semantic_spatial_get_map", args), {"plugin": "spatial"})
        spatial.dispatch.assert_called_with("get_map", args)
        short.dispatch.assert_not_called()
        self.assertEqual(bundle.dispatch("controlled_other", args), {"plugin": "short"})
        short.dispatch.assert_called_once_with("other", args)
        self.assertIsNone(bundle.dispatch("controlledsemantic", args))
        self.assertIsNone(bundle.dispatch("unknown_tool", args))

    def test_legacy_prefixes_route_unchanged(self):
        module = _load_main_module()
        vla = mock.Mock(PREFIX="vla")
        navi = mock.Mock(PREFIX="navi")
        bundle = module.ActuCoreBundle.__new__(module.ActuCoreBundle)
        bundle._plugins = [vla, navi]
        args = {"action": "info"}

        for plugin, full_name, expected in (
            (vla, "vla_info", "info"),
            (vla, "vla", "vla"),
            (navi, "navi_navigate_to", "navigate_to"),
            (navi, "navi", "navi"),
        ):
            with self.subTest(full_name=full_name):
                bundle.dispatch(full_name, args)
                plugin.dispatch.assert_called_with(expected, args)


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
            PREFIX = "controlled_semantic_spatial"

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
            PREFIX = "controlled_semantic_spatial"

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
                plugin = mock.Mock(PREFIX="controlled_semantic_spatial")
                plugin.stop.return_value = result
                bundle = module.ActuCoreBundle.__new__(module.ActuCoreBundle)
                bundle._plugins = [plugin]

                self.assertFalse(bundle.stop())


if __name__ == "__main__":
    unittest.main()
