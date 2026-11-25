import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.autonomous_task.cameras import CameraInputs, camera_ports, resolve_inputs


HEAD = "/limx_tron2/camera"
LEFT = "/limx_tron2/camera_left"
RIGHT = "/limx_tron2/camera_right"


def test_ports_and_binding_are_independent_of_order_and_missing_wires():
    state = "/test/state/eef"
    cfg = {"head_topic": HEAD, "left_wrist_topic": LEFT}
    assert resolve_inputs({"input_topic": state}, cfg) == (state, {})
    wired, bound = resolve_inputs({"input_topics": [LEFT, state, HEAD], "input_topic": LEFT}, cfg)
    assert wired == state and bound == {"head": HEAD, "left_wrist": LEFT}
    assert [p.get("topic") for p in camera_ports(bound)] == [HEAD, LEFT, None]
    assert all(p["format"] == "image/jpeg" for p in camera_ports())


def test_generic_topics_require_a_declared_or_configured_mount():
    args = {"input_topics": ["/a", "/robot/measure", "/b"],
            "camera_info": {"/a": {"id": "fixture/right_wrist", "format": "image/jpeg"}}}
    assert resolve_inputs(args, {"head_topic": "/b", "state_topic": "/retired/observations"}) == (
        "/robot/measure", {"head": "/b", "right_wrist": "/a"})

    with pytest.raises(ValueError) as ambiguous:
        resolve_inputs(args, {"state_topic": "/robot/measure"})
    assert "state_topic" not in str(ambiguous.value)
    assert "head_topic" in str(ambiguous.value)

    identified = {**args, "input_topics": ["/a", "/robot/state/eef", "/b"]}
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_inputs(identified, {"head_topic": "/a"})
    with pytest.raises(ValueError, match="Unknown camera role for /b"):
        resolve_inputs(identified, {})


def test_state_connection_must_be_present_and_unique():
    camera = {"input_topic": "/image", "camera_info": {"/image": {"id": "head_camera"}}}
    with pytest.raises(ValueError, match="state/eef"):
        resolve_inputs(camera, {})
    with pytest.raises(ValueError, match="state/eef"):
        resolve_inputs({"input_topics": ["/a/state/eef", "/b/state/eef"]}, {})


class Node:
    def __init__(self):
        self.callbacks = {}
        self.clock_read = threading.Event()

    def create_subscription(self, cls, topic, callback, qos):
        assert qos.depth == 1 and qos.reliability == "best_effort"
        self.callbacks[topic] = callback

    def get_clock(self):
        return self

    def now(self):
        value = time.time_ns()
        self.clock_read.set()
        return SimpleNamespace(nanoseconds=value)

    def send(self, topic, payload, *, stamp=None, frame_id=""):
        sec, nanos = divmod(stamp if stamp is not None else time.time_ns(), 1_000_000_000)
        header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanos), frame_id=frame_id)
        self.callbacks[topic](SimpleNamespace(header=header, data=payload))


def test_each_step_requires_new_genuine_images_from_all_connected_views():
    node = Node()
    cameras = CameraInputs(node, {"head": HEAD, "left_wrist": LEFT})
    # An idle camera callback does not even inspect the message payload.
    node.callbacks[HEAD](object())
    prior_stamp = time.time_ns()
    head_frame = b"\xff\xd8head\xff\xd9"
    wrist_frame = b"\xff\xd8wrist\xff\xd9"
    cancel = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(cameras.snapshot, cancel, 1)
        assert node.clock_read.wait(1)
        node.send(LEFT, wrist_frame)
        node.send(HEAD, head_frame, stamp=prior_stamp)
        assert not future.done()
        node.send(HEAD, head_frame)
        assert future.result(1) == {"head": head_frame, "left_wrist": wrist_frame}
        node.clock_read.clear()
        future = pool.submit(cameras.snapshot, cancel, 0.2)
        assert node.clock_read.wait(1)
        node.send(LEFT, wrist_frame, frame_id="tron2_right_wrist_camera")
        node.send(HEAD, head_frame)
        with pytest.raises(TimeoutError, match="left_wrist .*frame_id identifies right_wrist"):
            future.result(1)


def test_observation_errors_name_the_failed_input_and_stop_wakes_waiters():
    node = Node()
    cameras = CameraInputs(node, {"left_wrist": LEFT})
    cancel = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(cameras.snapshot, cancel, 0.15)
        assert node.clock_read.wait(1)
        node.send(LEFT, b"not a JPEG")
        with pytest.raises(TimeoutError, match="expected a JPEG"):
            future.result(1)
        node.clock_read.clear()
        future = pool.submit(cameras.snapshot, cancel)
        assert node.clock_read.wait(1)
        cameras.close()
        with pytest.raises(InterruptedError):
            future.result(1)
    with pytest.raises(RuntimeError, match="No RGB"):
        CameraInputs(Node(), {}).snapshot(cancel)
