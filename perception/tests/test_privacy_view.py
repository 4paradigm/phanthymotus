"""MCP/lifecycle tests with ROS transport doubles; real pixel processing."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from plugins import privacy_view as pv
from plugins.privacy_runtime import decode_jpeg, encode_jpeg
from vision_stubs import _FakeCompressedImage, _FakeExecutor


def wait_for(test, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if test():
            return
        time.sleep(0.005)
    assert test(), "condition timed out"


def message(stamp=None):
    image = np.full((64, 96, 3), 220, np.uint8)
    msg = _FakeCompressedImage(encode_jpeg(image))
    stamp = stamp or time.time_ns()
    msg.header.stamp = SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9)
    msg.header.frame_id = "synthetic_camera"
    return msg


class Detector:
    def __init__(self):
        self.closed = False
        self.calls = 0
        self.failure = None
        self.entered = threading.Event()
        self.release = None
        self.mutate = lambda x: x

    def detect(self, data, token, threshold, timeout):
        self.calls += 1
        self.entered.set()
        if self.release:
            assert self.release.wait(3)
        if self.failure:
            raise self.failure
        h, w = decode_jpeg(data).shape[:2]
        return self.mutate(
            {
                "ok": True,
                "complete": True,
                "truncated": False,
                "frame_token": token,
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": w,
                "height": h,
                "faces": [{"bbox": [20, 20, 50, 50], "det_score": 0.9}],
            }
        )

    def close(self):
        self.closed = True


@pytest.fixture
def setup(monkeypatch):
    detectors = []

    def factory(**kwargs):
        d = Detector()
        detectors.append(d)
        return d

    monkeypatch.setattr(pv, "FaceServiceProxy", factory)
    executor = _FakeExecutor()
    plugin = pv.PrivacyViewPlugin({}, executor)
    yield plugin, executor, detectors
    plugin.shutdown()
    wait_for(lambda: not executor.nodes)
    assert all(d.closed for d in detectors)


def call(plugin, action, key="a", **args):
    return plugin.dispatch(
        "privacy_view", {"action": action, "instance_id": key, **args}
    )


def start(setup, key="a", topic="/camera/rgb"):
    plugin, _executor, detectors = setup
    call(plugin, "start", key, input_topic=topic)
    node = plugin._nodes[key]
    wait_for(lambda: node._model_ready)
    return node, detectors[-1]


@pytest.mark.parametrize("frame_id", ["synthetic_camera", ""])
def test_real_pixels_header_and_status(setup, frame_id):
    node, _detector = start(setup)
    msg = message()
    msg.header.frame_id = frame_id
    original = msg.data
    node._receive(msg)
    wait_for(lambda: len(node._pub.messages) == 1)
    out = node._pub.raw_messages[0]
    assert out.data != original
    assert msg.data == original
    assert out.header.stamp.sec == msg.header.stamp.sec
    assert out.header.stamp.nanosec == msg.header.stamp.nanosec
    assert out.header.frame_id == msg.header.frame_id
    assert np.max(decode_jpeg(out.data)[25:45, 25:45]) < 5
    status = json.loads(node._status_pub.messages[-1])
    assert status["image_published"] is True
    assert status["faces_detected"] == 1
    assert status["privacy_guaranteed"] is False


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("detector failed"), TimeoutError(), RuntimeError("worker restarted")],
)
def test_inference_fault_has_no_image(setup, failure):
    node, detector = start(setup)
    detector.failure = failure
    node._receive(message())
    wait_for(lambda: node._metrics["errors"] > 0)
    assert node._pub.messages == []
    assert json.loads(node._status_pub.messages[-1])["image_published"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_token", "old"),
        ("width", 1),
        ("sha256", "wrong"),
        ("complete", False),
        ("truncated", True),
        ("faces", [{"bbox": [0, 0, float("nan"), 5], "det_score": 0.9}]),
        ("faces", None),
    ],
)
def test_untrusted_detection_has_no_image(setup, field, value):
    node, detector = start(setup)
    detector.mutate = lambda result: {**result, field: value}
    node._receive(message())
    wait_for(lambda: node._metrics["errors"] > 0)
    assert node._pub.messages == []


@pytest.mark.parametrize("kind", ["decode", "encode"])
def test_codec_fault_has_no_image(setup, monkeypatch, kind):
    node, _detector = start(setup)
    msg = message()

    def fail(*args):
        raise RuntimeError("codec failure")

    monkeypatch.setattr(pv, "decode_jpeg" if kind == "decode" else "encode_jpeg", fail)
    node._receive(msg)
    wait_for(lambda: node._metrics["errors"] > 0)
    assert node._pub.messages == []


@pytest.mark.parametrize(
    "kind", ["zero", "future", "old", "nanosec", "frame_id", "jpeg"]
)
def test_source_rejection(setup, kind):
    node, _detector = start(setup)
    msg = message()
    if kind == "zero":
        msg.header.stamp.sec = msg.header.stamp.nanosec = 0
    if kind == "future":
        msg.header.stamp.sec += 100
    if kind == "old":
        msg.header.stamp.sec -= 100
    if kind == "nanosec":
        msg.header.stamp.nanosec = 10**9
    if kind == "frame_id":
        msg.header.frame_id = None
    if kind == "jpeg":
        msg.data = b"invalid JPEG"
    node._receive(msg)
    wait_for(lambda: node._metrics["rejected"] > 0 or node._metrics["errors"] > 0)
    assert node._pub.messages == []


def test_zero_faces_is_success_with_risk_stated(setup):
    node, detector = start(setup)
    detector.mutate = lambda r: {**r, "faces": []}
    node._receive(message())
    wait_for(lambda: len(node._pub.messages) == 1)
    status = json.loads(node._status_pub.messages[-1])
    assert status["faces_detected"] == 0 and status["detection_complete"]
    assert not status["privacy_guaranteed"]


def test_duplicate_and_backwards_frames(setup):
    node, _detector = start(setup)
    msg = message()
    node._receive(msg)
    wait_for(lambda: len(node._pub.messages) == 1)
    node._receive(msg)
    node._receive(message(time.time_ns() - 10**9))
    assert node._metrics["rejected"] == 2
    assert len(node._pub.messages) == 1


@pytest.mark.parametrize("action", ["stop", "config", "timeout"])
def test_delayed_result_cannot_cross_boundary(setup, action):
    plugin, _executor, _detectors = setup
    node, detector = start(setup)
    detector.release = threading.Event()
    if action == "timeout":
        call(plugin, "config", inference_timeout_s=0.05)
    node._receive(message())
    assert detector.entered.wait(1)
    if action == "stop":
        result = call(plugin, "stop")
        assert result["image_publication_disabled"]
    elif action == "config":
        call(plugin, "config", mode="pixelate")
    else:
        time.sleep(0.06)
    detector.release.set()
    wait_for(
        lambda: (
            node.done.is_set()
            or node._metrics["processed"] > 0
            or node._metrics["errors"] > 0
        )
    )
    assert node._pub.messages == []


def test_stale_result_and_stream_loss(setup):
    plugin, _executor, _detectors = setup
    node, detector = start(setup)
    call(plugin, "config", max_frame_age_s=0.1)
    detector.release = threading.Event()
    node._receive(message())
    assert detector.entered.wait(1)
    time.sleep(0.12)
    detector.release.set()
    wait_for(lambda: node._metrics["discarded_generation"] > 0)
    assert node.snapshot()["error"] == "camera_timeout"
    assert node._pub.messages == []


def test_multicamera_isolation_and_queue_bound(setup):
    plugin, _executor, _detectors = setup
    a, da = start(setup, "a", "/cam_a")
    b, _db = start(setup, "b", "/cam_b")
    call(plugin, "config", "a", mode="pixelate")
    assert a._cfg.mode == "pixelate" and b._cfg.mode == "opaque"
    assert a.topics != b.topics
    da.release = threading.Event()
    a._receive(message())
    assert da.entered.wait(1)
    for _ in range(100):
        a._receive(message())
    assert a._queue.dropped >= 99
    b._receive(message())
    wait_for(lambda: len(b._pub.messages) == 1)
    da.release.set()
    call(plugin, "stop", "a")
    b._receive(message())
    wait_for(lambda: len(b._pub.messages) == 2)


def test_concurrent_repeated_start_stop_and_cleanup(setup):
    plugin, executor, detectors = setup
    for _ in range(8):
        with ThreadPoolExecutor(8) as pool:
            list(
                pool.map(
                    lambda _: call(plugin, "start", input_topic="/camera/rgb"),
                    range(16),
                )
            )
        assert len(executor.nodes) == 1
        node = plugin._nodes["a"]
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda _: call(plugin, "stop"), range(16)))
        wait_for(lambda: not executor.nodes)
        assert node.destroyed
        if node._thread:
            node._thread.join(1)
            assert not node._thread.is_alive()
    assert all(d.closed for d in detectors)


def test_stop_during_model_load_stays_cancelled(setup, monkeypatch):
    plugin, _executor, _detectors = setup
    entered, release = threading.Event(), threading.Event()
    detector = Detector()

    def factory(on_status=None):
        entered.set()
        assert release.wait(3)
        return detector

    monkeypatch.setattr(plugin, "_make_detector", factory)
    call(plugin, "start", input_topic="/camera")
    node = plugin._nodes["a"]
    assert entered.wait(1)
    node._receive(message())
    assert call(plugin, "stop")["state"] == "stopping"
    call(plugin, "config", mode="pixelate")
    assert call(plugin, "info")["state"] == "stopping"
    assert call(plugin, "start", input_topic="/camera")["state"] == "stopping"
    release.set()
    wait_for(lambda: node.done.is_set())
    assert detector.closed and node.destroyed and not node._pub.messages


def test_load_failure_and_stop(setup, monkeypatch):
    plugin, _executor, _detectors = setup

    def fail(on_status=None):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(plugin, "_make_detector", fail)
    call(plugin, "start", input_topic="/camera")
    node = plugin._nodes["a"]
    wait_for(lambda: node._metrics["errors"] > 0)
    node._receive(message())
    assert node.snapshot()["error"] == "model_load_failed"
    assert node._pub.messages == []
    call(plugin, "stop")
    wait_for(lambda: node.destroyed)


def test_config_validation_and_replay(setup):
    plugin, _executor, _detectors = setup
    node, _ = start(setup)
    version = node._version
    assert call(plugin, "config", mode="opaque")["status"] == "configured"
    assert node._version == version
    assert call(plugin, "config", gaussian_strength=4)["state"] == "error"
    assert node._version == version
    with ThreadPoolExecutor(8) as pool:
        list(
            pool.map(
                lambda i: call(
                    plugin, "config", mode="pixelate" if i % 2 else "opaque"
                ),
                range(30),
            )
        )
    assert node._cfg.mode in ("opaque", "pixelate")


def test_stop_wins_before_many_concurrent_starts_create_worker(setup, monkeypatch):
    plugin, executor, _detectors = setup
    barrier, release = threading.Barrier(11), threading.Event()
    original = pv._PrivacyNode.start

    def delayed_start(node):
        barrier.wait(3)
        assert release.wait(3)
        original(node)

    monkeypatch.setattr(pv._PrivacyNode, "start", delayed_start)
    disposals = []
    original_remove = executor.remove_node

    def remove(node):
        disposals.append(node)
        original_remove(node)

    monkeypatch.setattr(executor, "remove_node", remove)
    with ThreadPoolExecutor(12) as pool:
        futures = [
            pool.submit(call, plugin, "start", input_topic="/camera") for _ in range(10)
        ]
        barrier.wait(3)
        node = plugin._nodes["a"]
        call(plugin, "stop")
        release.set()
        for future in futures:
            future.result(3)
    assert node.destroyed and node.done.is_set()
    assert len(disposals) == 1 and not executor.nodes


def test_multiple_instances_repeated_parallel_teardown(setup):
    plugin, executor, _detectors = setup
    for _ in range(5):
        with ThreadPoolExecutor(8) as pool:
            list(
                pool.map(
                    lambda i: call(plugin, "start", str(i), input_topic=f"/camera_{i}"),
                    range(4),
                )
            )
        nodes = list(plugin._nodes.values())
        assert len(nodes) == 4 and len(executor.nodes) == 4
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda i: call(plugin, "stop", str(i)), range(4)))
        wait_for(lambda: not executor.nodes)
        for node in nodes:
            assert node.destroyed
            if node._thread is not None:
                node._thread.join(1)
                assert not node._thread.is_alive()


def test_bundle_registration_and_dispatch(setup):
    import main

    bundle = main.PerceptionBundle(
        {
            "plugins": {
                "privacy_view": {"enabled": True},
                "echo_cancellation": {"enabled": False},
                "audio_mixer": {"enabled": False},
            }
        },
        _FakeExecutor(),
    )
    try:
        tools = bundle.get_all_tools()
        assert len(tools) == 1 and tools[0]["name"] == "privacy_view"
        assert tools[0]["type"] == "processor" and tools[0]["multiInstance"]
        assert [p["format"] for p in tools[0]["topic_out"]] == [
            "image/jpeg",
            "data/json",
        ]
        assert bundle.dispatch("privacy_view_info", {})["state"] == "idle"
        assert (
            bundle.dispatch("privacy_view", {"action": "config", "mode": "bad"})[
                "state"
            ]
            == "error"
        )
    finally:
        bundle.shutdown()


@pytest.mark.parametrize(
    "cfg",
    [{"typo": 1}, {"num_threads": True}, {"num_threads": 0}, {"num_threads": 10000}],
)
def test_invalid_deployment_config_rejected(cfg):
    with pytest.raises(pv.PrivacyError):
        pv.PrivacyViewPlugin(cfg, _FakeExecutor())
