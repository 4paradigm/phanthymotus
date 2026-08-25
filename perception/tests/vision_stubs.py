"""
tests/vision_stubs.py — Shared fakes for vision-plugin unit tests.

Importing this module installs fake rclpy / sensor_msgs / std_msgs modules
into sys.modules (so plugin modules import cleanly off-robot) and puts the
perception root on sys.path. test_ocr_plugin.py and test_obstacle_plugin.py
both import from here; conftest.py imports it first so the stubs are in place
before any plugin module is imported.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))


# ── fake ROS ─────────────────────────────────────────────────────────────────

class _FakePublisher:
    def __init__(self, topic):
        self.topic = topic
        self.messages = []
        # TTS gates each utterance on a matched reader; tests that do not care
        # about the gate get a reader by default.
        self.subscription_count = 1

    def publish(self, msg):
        self.messages.append(msg.data)

    def get_subscription_count(self):
        return self.subscription_count


class _FakeSubscription:
    def __init__(self, topic, callback, qos):
        self.topic = topic
        self.callback = callback
        self.qos = qos


class _FakeNode:
    instances: list = []

    def __init__(self, name):
        self._name = name
        self.publishers = []
        self.subscriptions = []
        self.destroyed = False
        _FakeNode.instances.append(self)

    def get_name(self):
        return self._name

    def create_publisher(self, msg_type, topic, qos):
        publisher = _FakePublisher(topic)
        self.publishers.append(publisher)
        return publisher

    def create_subscription(self, msg_type, topic, callback, qos):
        subscription = _FakeSubscription(topic, callback, qos)
        self.subscriptions.append(subscription)
        return subscription

    def destroy_node(self):
        self.destroyed = True

    def get_clock(self):
        return _FakeClock()


class _FakeClock:
    def now(self):
        return self

    def to_msg(self):
        return 0


class _FakeExecutor:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        self.nodes.remove(node)


class _FakeString:
    def __init__(self):
        self.data = ""


class _FakeCompressedImage:
    def __init__(self, data: bytes, fmt="jpeg"):
        self.data = data
        self.format = fmt


class _FakeAudioChunk:
    """Stands in for audio_msgs.msg.AudioChunk (built from a ROS .msg on device)."""

    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None)
        self.format = ""
        self.data = []


def _install_fake_ros():
    rclpy = types.ModuleType("rclpy")
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _FakeNode
    qos_mod = types.ModuleType("rclpy.qos")

    class QoSProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __eq__(self, other):
            return isinstance(other, QoSProfile) and self.__dict__ == other.__dict__

    qos_mod.QoSProfile = QoSProfile
    qos_mod.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT="best_effort", RELIABLE="reliable")
    qos_mod.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last", KEEP_ALL="keep_all")
    qos_mod.DurabilityPolicy = types.SimpleNamespace(VOLATILE="volatile", TRANSIENT_LOCAL="tl")
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.CompressedImage = _FakeCompressedImage
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = _FakeString
    audio_msgs = types.ModuleType("audio_msgs")
    audio_msgs_msg = types.ModuleType("audio_msgs.msg")
    audio_msgs_msg.AudioChunk = _FakeAudioChunk
    for name, module in {
        "rclpy": rclpy, "rclpy.node": node_mod, "rclpy.qos": qos_mod,
        "sensor_msgs": sensor_msgs, "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg,
        "audio_msgs": audio_msgs, "audio_msgs.msg": audio_msgs_msg,
    }.items():
        sys.modules[name] = module


_install_fake_ros()


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()
