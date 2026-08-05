from __future__ import annotations

import importlib
import pathlib
import sys
import types
import unittest


PERCEPTION_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Node:
    def __init__(self, name):
        self.name = name
        self.publishers = []

    def create_publisher(self, _message_type, _topic, _qos):
        publisher = _Publisher()
        self.publishers.append(publisher)
        return publisher

    def create_subscription(self, *_args, **_kwargs):
        return object()

    def get_clock(self):
        return types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(
                to_msg=lambda: types.SimpleNamespace(sec=1, nanosec=0)
            )
        )


class _QoSProfile:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Policy:
    BEST_EFFORT = "best_effort"
    KEEP_LAST = "keep_last"
    VOLATILE = "volatile"


class _String:
    def __init__(self):
        self.data = ""


class _AudioChunk:
    def __init__(self):
        self.header = types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=0, nanosec=0)
        )
        self.format = ""
        self.data = []


def _install_ros_stubs() -> None:
    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_node.Node = _Node
    rclpy_qos.QoSProfile = _QoSProfile
    rclpy_qos.ReliabilityPolicy = _Policy
    rclpy_qos.HistoryPolicy = _Policy
    rclpy_qos.DurabilityPolicy = _Policy
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = _String
    audio_msgs = types.ModuleType("audio_msgs")
    audio_msgs_msg = types.ModuleType("audio_msgs.msg")
    audio_msgs_msg.AudioChunk = _AudioChunk
    sys.modules.update(
        {
            "rclpy": rclpy,
            "rclpy.node": rclpy_node,
            "rclpy.qos": rclpy_qos,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs_msg,
            "audio_msgs": audio_msgs,
            "audio_msgs.msg": audio_msgs_msg,
        }
    )


_install_ros_stubs()
tts = importlib.import_module("plugins.tts")
duplex_node = importlib.import_module("plugins.duplexaudio.node")


class _FakeAEC:
    backend_name = "fake-aec"

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def process_pcm(self, pcm: bytes, capture_ts: float) -> bytes:
        self.calls.append((pcm, capture_ts))
        if self.fail:
            raise RuntimeError("backend failed")
        return pcm

    def stats(self) -> dict:
        return {"backend": self.backend_name, "calls": len(self.calls)}


def _audio_message(pcm: bytes, fmt: str = "audio/pcm-16k") -> _AudioChunk:
    message = _AudioChunk()
    message.header.stamp.sec = 1000
    message.format = fmt
    message.data = list(pcm)
    return message


class ExistingNodeHookTests(unittest.TestCase):
    def test_tts_output_topic_default_remains_backward_compatible(self):
        default = tts._TTSNode("/text", object())
        no_input = tts._TTSNode(None, object())
        custom = tts._TTSNode(
            None, object(), output_topic="/duplexaudio/tts"
        )

        self.assertEqual(default._output_topic, "/text/tts")
        self.assertEqual(no_input._output_topic, "/perception/tts")
        self.assertEqual(custom._output_topic, "/duplexaudio/tts")

    def test_tts_frame_observer_receives_exact_published_pcm_and_timestamp(self):
        observed = []
        node = tts._TTSNode(
            None,
            object(),
            output_topic="/duplexaudio/tts",
            frame_observer=lambda pcm, ts: observed.append((pcm, ts)),
        )
        frame = b"\x01\x00\x02\x00"

        node._publish_frame(frame, 1234.5)

        self.assertEqual(observed, [(frame, 1234.5)])
        self.assertEqual(len(node._pub.messages), 1)
        published = node._pub.messages[0]
        self.assertEqual(bytes(published.data), frame)
        self.assertEqual(published.format, "audio/pcm-16k")


class DuplexAudioBridgeTests(unittest.TestCase):
    def test_clean_pcm_is_rechunked_without_padding_or_loss(self):
        aec = _FakeAEC()
        node = duplex_node.DuplexAudioNode(
            "/mic", "/clean", "card", aec
        )

        node._on_audio(_audio_message(b"\x01\x00" * 300))
        self.assertEqual(node.stats()["pending_output_bytes"], 600)
        self.assertEqual(node._publisher.messages, [])

        node._on_audio(_audio_message(b"\x02\x00" * 300))

        self.assertEqual(len(node._publisher.messages), 1)
        published = node._publisher.messages[0]
        self.assertEqual(len(published.data), 1024)
        self.assertEqual(node.stats()["pending_output_bytes"], 176)
        self.assertEqual(len(aec.calls), 2)

    def test_unsupported_format_stops_bridge_without_output(self):
        node = duplex_node.DuplexAudioNode(
            "/mic", "/clean", "card", _FakeAEC()
        )

        node._on_audio(_audio_message(b"\x00\x00" * 512, "audio/wav"))

        self.assertEqual(node.state, "error")
        self.assertIn("unsupported audio format", node.last_error)
        self.assertEqual(node._publisher.messages, [])

    def test_processing_failure_is_fail_closed_by_default(self):
        node = duplex_node.DuplexAudioNode(
            "/mic", "/clean", "card", _FakeAEC(fail=True)
        )

        node._on_audio(_audio_message(b"\x00\x00" * 512))

        self.assertEqual(node.state, "error")
        self.assertIn("AEC processing failed", node.last_error)
        self.assertEqual(node._publisher.messages, [])

    def test_processing_failure_passthrough_remains_visible(self):
        raw = b"\x01\x00" * 512
        node = duplex_node.DuplexAudioNode(
            "/mic",
            "/clean",
            "card",
            _FakeAEC(fail=True),
            failure_policy="passthrough",
        )

        node._on_audio(_audio_message(raw))

        self.assertEqual(node.state, "running")
        self.assertIn("AEC processing failed", node.last_error)
        self.assertEqual(len(node._publisher.messages), 1)
        self.assertEqual(bytes(node._publisher.messages[0].data), raw)


if __name__ == "__main__":
    unittest.main()
