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

    def destroy_subscription(self, subscription):
        # Plugins drop the subscription on stop() and recreate it on the next
        # start(); the tests assert on what is left registered here.
        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)

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
    # Defaults so a plugin that *publishes* one can construct it the way ROS
    # does — `CompressedImage()` then assign — as well as tests that build an
    # incoming frame in one call.
    def __init__(self, data: bytes = b"", fmt="jpeg"):
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
    # main.py does `import rclpy.executors`, which needs rclpy to be a package
    # with that submodule attached — a bare ModuleType is not, and the import
    # fails with "'rclpy' is not a package".
    executors_mod = types.ModuleType("rclpy.executors")
    executors_mod.MultiThreadedExecutor = _FakeExecutor
    executors_mod.SingleThreadedExecutor = _FakeExecutor
    rclpy.executors = executors_mod
    rclpy.node = node_mod
    rclpy.init = lambda *args, **kwargs: None
    rclpy.shutdown = lambda *args, **kwargs: None
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
        "rclpy.executors": executors_mod,
        "sensor_msgs": sensor_msgs, "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg,
        "audio_msgs": audio_msgs, "audio_msgs.msg": audio_msgs_msg,
    }.items():
        sys.modules[name] = module


def _install_fake_cv2():
    """Minimal cv2 for host-side tests.

    The vision plugins do `import cv2` inside their worker threads, so without
    this every worker dies on its first frame and the failure surfaces only as
    a PytestUnhandledThreadExceptionWarning — easy to scroll past while
    believing the pipeline was exercised. Only the handful of calls those
    workers make are implemented; anything else raises rather than quietly
    returning something plausible.

    Skipped when the real cv2 is importable, so a machine that has it tests
    against the real thing.
    """
    try:
        import cv2  # noqa: F401
        return
    except ImportError:
        pass

    import numpy as _np

    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.INTER_NEAREST = 0
    cv2.INTER_LINEAR = 1

    def imdecode(buf, flags):
        # Tests hand in a raw "WxH" marker rather than a real JPEG; anything
        # unparseable decodes to None, which is what a corrupt frame does.
        try:
            width, height = (int(v) for v in bytes(buf).decode().split("x"))
        except Exception:
            return None
        return _np.zeros((height, width, 3), dtype=_np.uint8)

    def resize(src, dsize, interpolation=0):
        width, height = dsize
        rows = (_np.arange(height) * src.shape[0] // height).clip(0, src.shape[0] - 1)
        cols = (_np.arange(width) * src.shape[1] // width).clip(0, src.shape[1] - 1)
        return src[rows][:, cols]

    def cvtColor(src, code):
        if code != cv2.COLOR_BGR2GRAY:
            raise NotImplementedError(f"fake cv2: cvtColor code {code}")
        return src.mean(axis=2).astype(_np.uint8)

    def connectedComponents(image, connectivity=8):
        """Two-pass labelling with union-find. 4-connectivity only.

        Small and slow, which is fine: the frames in these tests are tens of
        pixels across. It exists so `lens_barrel_mask` runs for real here —
        stubbing it out to "no mask" would make every test of the masking path
        pass by not testing it.
        """
        rows, cols = image.shape
        labels = _np.zeros((rows, cols), dtype=_np.int32)
        parent = [0]

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for r in range(rows):
            for c in range(cols):
                if not image[r, c]:
                    continue
                up = labels[r - 1, c] if r else 0
                left = labels[r, c - 1] if c else 0
                if up and left:
                    labels[r, c] = min(up, left)
                    union(up, left)
                elif up or left:
                    labels[r, c] = up or left
                else:
                    parent.append(len(parent))
                    labels[r, c] = len(parent) - 1

        remap, nxt = {0: 0}, 1
        for r in range(rows):
            for c in range(cols):
                if not labels[r, c]:
                    continue
                root = find(labels[r, c])
                if root not in remap:
                    remap[root] = nxt
                    nxt += 1
                labels[r, c] = remap[root]
        return nxt, labels

    cv2.COLOR_BGR2GRAY = 6
    cv2.imdecode = imdecode
    cv2.resize = resize
    cv2.cvtColor = cvtColor
    cv2.connectedComponents = connectedComponents
    sys.modules["cv2"] = cv2


_install_fake_ros()
_install_fake_cv2()


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()
