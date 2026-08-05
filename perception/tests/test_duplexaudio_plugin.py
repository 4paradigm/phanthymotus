from __future__ import annotations

import pathlib
import sys
import types
import unittest
from unittest import mock


PERCEPTION_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.duplexaudio.plugin import DuplexAudioPlugin  # noqa: E402


class _Executor:
    def __init__(self):
        self.added = []
        self.removed = []

    def add_node(self, node):
        self.added.append(node)

    def remove_node(self, node):
        self.removed.append(node)


class _NodeBase:
    def __init__(self):
        self.state = "idle"
        self.destroyed = False

    def start(self):
        self.state = "running"
        return {"state": "running"}

    def stop(self):
        self.state = "idle"
        return {"state": "idle"}

    def destroy_node(self):
        self.destroyed = True


class _TTSNode(_NodeBase):
    def __init__(self, input_topic, adapter, **kwargs):
        super().__init__()
        self.input_topic = input_topic
        self.adapter = adapter
        self.output_topic = kwargs["output_topic"]
        self.frame_observer = kwargs["frame_observer"]
        self.queued = []

    def enqueue(self, text, trace_id=""):
        self.queued.append((text, trace_id))


class _Bridge(_NodeBase):
    def __init__(self, input_topic, clean_topic, instance_id, aec, failure_policy):
        super().__init__()
        self.state = "running"
        self.input_topic = input_topic
        self.clean_topic = clean_topic
        self.instance_id = instance_id
        self.aec = aec
        self.failure_policy = failure_policy

    def stats(self):
        return {
            "enabled": self.aec is not None,
            "bridge_state": self.state,
            "failure_policy": self.failure_policy,
        }


class _AEC:
    name = "fake-aec"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.references = []

    def push_reference(self, pcm, play_start_ts):
        self.references.append((pcm, play_start_ts))

    def calibrate(self):
        return {"ok": True, "d_real_ms": self.kwargs["delay_ms"]}

    def close(self):
        self.closed = True


def _fake_modules(aec_class=_AEC):
    tts_module = types.ModuleType("plugins.tts")
    tts_module._TTSNode = _TTSNode
    tts_module._build_tts_adapter = lambda cfg: {"kind": "tts", "cfg": cfg}
    node_module = types.ModuleType("plugins.duplexaudio.node")
    node_module.DuplexAudioNode = _Bridge
    import plugins.duplexaudio.aec as aec_module

    return {
        "plugins.tts": tts_module,
        "plugins.duplexaudio.node": node_module,
    }, mock.patch.object(aec_module, "AECProcessor", aec_class)


class DuplexAudioPluginTests(unittest.TestCase):
    def test_start_speak_stats_and_stop_form_one_instance_lifecycle(self):
        executor = _Executor()
        plugin = DuplexAudioPlugin({"enabled": True}, executor)
        modules, aec_patch = _fake_modules()
        with mock.patch.dict(sys.modules, modules), aec_patch:
            started = plugin.dispatch(
                "duplexaudio",
                {
                    "action": "start",
                    "instance_id": "card-1",
                    "input_topic": "/robot/ext_mic/audio",
                },
            )
            spoken = plugin.dispatch(
                "duplexaudio",
                {
                    "action": "speak",
                    "instance_id": "card-1",
                    "text": "你好",
                    "_trace_id": "trace-1",
                },
            )
            stats = plugin.dispatch(
                "duplexaudio", {"action": "aec_stats", "instance_id": "card-1"}
            )
            stopped = plugin.dispatch(
                "duplexaudio", {"action": "stop", "instance_id": "card-1"}
            )

        self.assertEqual(started["state"], "running")
        self.assertEqual(
            [item["topic"] for item in started["topic_out"]],
            [
                "/robot/ext_mic/audio/duplexaudio/clean",
                "/robot/ext_mic/audio/duplexaudio/tts",
            ],
        )
        self.assertEqual(spoken["status"], "queued")
        tts_node = executor.added[1]
        self.assertEqual(tts_node.queued, [("你好", "trace-1")])
        self.assertIsNotNone(tts_node.frame_observer)
        self.assertTrue(stats["enabled"])
        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(len(executor.added), 2)
        self.assertEqual(len(executor.removed), 2)
        self.assertTrue(all(node.destroyed for node in executor.added))

    def test_effective_config_has_no_asr_model(self):
        plugin = DuplexAudioPlugin({"enabled": True}, _Executor())

        config = plugin._effective_config("card-1")

        self.assertNotIn("asr", config)
        self.assertIn("tts", config)
        self.assertIn("aec", config)

    def test_missing_aec_backend_fails_closed_before_tts_model_load(self):
        class BrokenAEC:
            def __init__(self, **kwargs):
                raise RuntimeError("no backend")

        executor = _Executor()
        plugin = DuplexAudioPlugin({"enabled": True}, executor)
        modules, aec_patch = _fake_modules(BrokenAEC)
        with mock.patch.dict(sys.modules, modules), aec_patch:
            result = plugin.dispatch(
                "duplexaudio",
                {
                    "action": "start",
                    "instance_id": "card-1",
                    "input_topic": "/robot/ext_mic/audio",
                },
            )
        self.assertEqual(result["state"], "error")
        self.assertEqual(result["error"], "aec_unavailable")
        self.assertEqual(executor.added, [])

    def test_passthrough_policy_keeps_error_visible(self):
        class BrokenAEC:
            def __init__(self, **kwargs):
                raise RuntimeError("no backend")

        executor = _Executor()
        plugin = DuplexAudioPlugin({"enabled": True}, executor)
        modules, aec_patch = _fake_modules(BrokenAEC)
        with mock.patch.dict(sys.modules, modules), aec_patch:
            plugin.dispatch(
                "duplexaudio",
                {
                    "action": "config",
                    "instance_id": "card-1",
                    "aec_failure_policy": "passthrough",
                },
            )
            result = plugin.dispatch(
                "duplexaudio",
                {
                    "action": "start",
                    "instance_id": "card-1",
                    "input_topic": "/robot/ext_mic/audio",
                },
            )
            stats = plugin.dispatch(
                "duplexaudio", {"action": "aec_stats", "instance_id": "card-1"}
            )
        self.assertEqual(result["state"], "running")
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["init_error"], "no backend")
        self.assertEqual(stats["failure_policy"], "passthrough")

    def test_instance_aec_config_stops_only_target_instance(self):
        executor = _Executor()
        plugin = DuplexAudioPlugin({"enabled": True}, executor)
        modules, aec_patch = _fake_modules()
        with mock.patch.dict(sys.modules, modules), aec_patch:
            for instance_id in ("one", "two"):
                result = plugin.dispatch(
                    "duplexaudio",
                    {
                        "action": "start",
                        "instance_id": instance_id,
                        "input_topic": f"/{instance_id}/mic",
                    },
                )
                self.assertEqual(result["state"], "running")
            configured = plugin.dispatch(
                "duplexaudio",
                {
                    "action": "config",
                    "instance_id": "one",
                    "aec_delay_ms": 350,
                },
            )
            one = plugin.dispatch(
                "duplexaudio", {"action": "info", "instance_id": "one"}
            )
            two = plugin.dispatch(
                "duplexaudio", {"action": "info", "instance_id": "two"}
            )
        self.assertEqual(configured["stopped_instances"], ["one"])
        self.assertEqual(one["state"], "idle")
        self.assertEqual(two["state"], "running")

    def test_duplicate_microphone_topic_is_rejected(self):
        executor = _Executor()
        plugin = DuplexAudioPlugin({"enabled": True}, executor)
        modules, aec_patch = _fake_modules()
        with mock.patch.dict(sys.modules, modules), aec_patch:
            first = plugin.dispatch(
                "duplexaudio",
                {
                    "action": "start",
                    "instance_id": "one",
                    "input_topic": "/shared/mic/",
                },
            )
            second = plugin.dispatch(
                "duplexaudio",
                {
                    "action": "start",
                    "instance_id": "two",
                    "input_topic": "/shared/mic",
                },
            )

        self.assertEqual(first["state"], "running")
        self.assertEqual(second["state"], "error")
        self.assertEqual(second["error"], "input_topic_in_use")
        self.assertEqual(len(executor.added), 2)

    def test_sanitized_instance_ids_keep_distinct_internal_topics(self):
        executor = _Executor()
        plugin = DuplexAudioPlugin({"enabled": True}, executor)
        modules, aec_patch = _fake_modules()
        with mock.patch.dict(sys.modules, modules), aec_patch:
            for instance_id, topic in (("card-a", "/mic/a"), ("card_a", "/mic/b")):
                result = plugin.dispatch(
                    "duplexaudio",
                    {
                        "action": "start",
                        "instance_id": instance_id,
                        "input_topic": topic,
                    },
                )
                self.assertEqual(result["state"], "running")

        bridges = executor.added[::2]
        self.assertEqual(len(bridges), 2)
        self.assertNotEqual(bridges[0].clean_topic, bridges[1].clean_topic)
        self.assertNotEqual(bridges[0].instance_id, bridges[1].instance_id)


if __name__ == "__main__":
    unittest.main()
