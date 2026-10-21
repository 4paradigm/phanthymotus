"""Hardware depth adapter contract, lifecycle, freshness and old-navi compatibility."""
import importlib.util
import json
from pathlib import Path
import threading
import time
import types
import zlib

import numpy as np
import pytest

from vision_stubs import _FakeCompressedImage, _FakeExecutor, _wait_until
from plugins import lidar_depth as ld

TOPIC = "/nvidia_desktop/camera/head/depth"
KEY = "card-hardware-depth"


def message(mm=1500, values=None):
    pixels = np.full((480, 640), mm, dtype="<u2") if values is None else values
    msg = _FakeCompressedImage(zlib.compress(pixels.tobytes(), 1), ld.ROS_FORMAT)
    msg.header.stamp = types.SimpleNamespace(sec=0, nanosec=0)
    msg.header.frame_id = "orbbec_depth_optical_frame"
    return msg


@pytest.fixture
def plugin():
    executor = _FakeExecutor()
    adapter = ld.LidarDepthPerceptionPlugin({}, executor)
    yield adapter, executor
    adapter.shutdown()


def start(adapter, **kwargs):
    return adapter.dispatch("lidar_depth", {"action": "start", "instance_id": KEY,
                                           "input_topic": TOPIC, **kwargs})


def feed(adapter, msg=None):
    node = adapter._nodes[KEY]
    node.subscriptions[0].callback(msg or message())
    return node


def test_passthrough_units_header_summary_and_navigation_decoder(plugin):
    adapter, executor = plugin
    result = start(adapter)
    assert result["state"] == "running"
    assert result["available"] is False
    assert result["data_state"] == "waiting"
    pixels = np.full((480, 640), 1500, dtype="<u2")
    pixels[:, :213] = 1000
    pixels[:, 427:] = 2000
    pixels[0, :4] = [0, 1, 1234, 65535]
    msg = message(values=pixels)
    node = feed(adapter, msg)
    _wait_until(lambda: len(node._summary_pub.messages) == 1)
    output = node._depth_pub.raw_messages[0]
    assert output.data == msg.data
    assert output.header.stamp.sec == 0
    assert output.header.frame_id == msg.header.frame_id
    assert output.format == ld.ROS_FORMAT
    np.testing.assert_array_equal(
        np.frombuffer(zlib.decompress(output.data), dtype="<u2").reshape(480, 640), pixels)
    summary = json.loads(node._summary_pub.messages[0])
    assert summary["nearest_by_region"] == {"left": 1.0, "center": 1.5, "right": 2.0}
    assert summary["source"] == "hardware_depth"
    assert summary["unit"] == "m" and summary["scale"] == "metric"
    assert summary["rgb_alignment_verified"] is False
    assert "unverified" in node.info()["alignment_note"]
    reading = adapter.dispatch("lidar_depth", {"action": "read", "instance_id": KEY})
    assert reading["ok"] and reading["distance_m"] == 1.5
    path = Path(__file__).parents[2] / "actucore/plugins/navi/depth.py"
    spec = importlib.util.spec_from_file_location("navi_hardware_depth_consumer_test", path)
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    decoded = consumer.decode(output.data)
    assert np.isnan(decoded[0, 0])
    assert decoded[240, 320] == 1.5
    assert len(executor.nodes) == 1


def test_output_topics_match_old_navi_and_do_not_collide_with_sources(plugin):
    adapter, _ = plugin
    result = start(adapter)
    depth, summary = [p["topic"] for p in result["topic_out"]]
    assert depth == TOPIC + "/lidar_depth/visual_depth"
    assert summary == TOPIC + "/lidar_depth/visual_depth_summary"
    assert depth != TOPIC
    assert depth != "/nvidia_desktop/camera/head/visual_depth"
    # These are the original role suffixes, before hardware-depth support.
    assert depth.endswith("/visual_depth")
    assert summary.endswith("/visual_depth_summary")
    tool, = adapter.get_tools()
    assert tool["name"] == "lidar_depth" and tool["multiInstance"]
    assert tool["topic_in"][0]["format"] == ld.FORMAT
    assert "calibrate" not in tool["inputSchema"]["properties"]["action"]["enum"]


@pytest.mark.parametrize("payload,fmt", [
    (b"jpeg", "jpeg"), (b"png", "16UC1; compressedDepth png"),
    (b"not zlib", ld.ROS_FORMAT), (zlib.compress(b"short"), ld.ROS_FORMAT),
    (zlib.compress(b"x" * (ld.RAW_SIZE + 100)), ld.ROS_FORMAT),
    (message().data + b"trailing", ld.ROS_FORMAT),
    (message().data[:-2], ld.ROS_FORMAT),
    (b"x" * (ld.RAW_SIZE + 4097), ld.ROS_FORMAT),
])
def test_rejects_bad_frames_without_publishing(plugin, payload, fmt):
    adapter, _ = plugin
    start(adapter)
    node = feed(adapter, _FakeCompressedImage(payload, fmt))
    _wait_until(lambda: node.counts["rejected"] == 1)
    assert not node._depth_pub.messages
    assert not node._summary_pub.messages
    assert node.info()["error"]


def test_all_invalid_is_unknown_and_new_valid_frame_recovers(plugin):
    adapter, _ = plugin
    start(adapter)
    node = feed(adapter, message(0))
    _wait_until(lambda: node.counts["published"] == 1)
    summary = json.loads(node._summary_pub.messages[0])
    assert summary["range"] is None
    assert all(v is None for v in summary["nearest_by_region"].values())
    assert not node.info()["available"]
    assert node.info()["data_state"] == "invalid"
    feed(adapter, message())
    _wait_until(lambda: node.counts["published"] == 2)
    assert node.info()["available"]


def test_latest_frame_only_and_no_periodic_replay(plugin):
    adapter, _ = plugin
    start(adapter)
    node = feed(adapter, message(1000))
    _wait_until(lambda: node.counts["published"] == 1)
    feed(adapter, message(2000))
    feed(adapter, message(3000))
    _wait_until(lambda: node.counts["published"] == 2)
    assert node.counts["dropped"] >= 1
    assert adapter.dispatch("lidar_depth", {"action": "read"})["distance_m"] == 3.0
    time.sleep(0.2)
    assert node.counts["published"] == 2


def test_stale_capture_queue_age_and_future_stamp_are_rejected(plugin, monkeypatch):
    adapter, _ = plugin
    start(adapter)
    node = adapter._nodes[KEY]
    clock = types.SimpleNamespace(now=100.0)
    monkeypatch.setattr(ld, "time", types.SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: 1_800_000_000.0))
    msg = message()
    pending = (msg.data, msg.format, msg.header, 99.0, 1_800_000_000.0)
    node._process(pending)
    assert node.counts["dropped"] == 1
    msg.header.stamp.sec = 1_799_999_999
    node._process((msg.data, msg.format, msg.header, 100, 1_800_000_000.0))
    assert node.counts["dropped"] == 2
    msg.header.stamp.sec = 1_800_000_002
    node._process((msg.data, msg.format, msg.header, 100, 1_800_000_000.0))
    assert node.counts["dropped"] == 3
    assert not node._depth_pub.messages


def test_read_expires_and_capture_time_counts_toward_age(plugin, monkeypatch):
    adapter, _ = plugin
    start(adapter)
    node = adapter._nodes[KEY]
    clock = types.SimpleNamespace(now=100.0)
    monkeypatch.setattr(ld, "time", types.SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: 1_800_000_000.0))
    msg = message()
    msg.header.stamp.sec, msg.header.stamp.nanosec = 1_799_999_999, 600_000_000
    node._process((msg.data, msg.format, msg.header, 100, 1_800_000_000.0))
    assert node.read("center")["ok"]
    clock.now += 0.2
    assert node.info()["data_state"] == "stale"
    assert node.read("center")["reason"] == "stale"


def test_disposal_restart_and_per_instance_live_configuration(plugin):
    adapter, executor = plugin
    start(adapter)
    old = adapter._nodes[KEY]
    result = adapter.dispatch("lidar_depth", {"action": "config", "instance_id": KEY,
                                             "fps": 3, "rgb_alignment_verified": True})
    assert result["restart_required"]
    assert not executor.nodes and old.destroyed and not old._worker.is_alive()
    assert start(adapter)["state"] == "running"
    new = adapter._nodes[KEY]
    assert new is not old
    assert new.cfg["fps"] == 3 and new.cfg["rgb_alignment_verified"]
    assert len(executor.nodes) == 1
    assert adapter.dispatch("lidar_depth", {"action": "stop", "instance_id": KEY})["state"] == "idle"
    assert not executor.nodes


def test_concurrent_starts_do_not_duplicate_and_stop_cancels_pending_start(plugin, monkeypatch):
    adapter, executor = plugin
    threads = [threading.Thread(target=lambda: start(adapter)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert len(executor.nodes) == len(adapter._nodes) == 1
    node = adapter._nodes[KEY]
    assert len(node.subscriptions) == 1
    adapter.dispatch("lidar_depth", {"action": "stop"})
    entered, release = threading.Event(), threading.Event()
    original = ld._HardwareDepthNode.start

    def delayed_start(self):
        entered.set()
        release.wait(2)
        return original(self)

    monkeypatch.setattr(ld._HardwareDepthNode, "start", delayed_start)
    result = []
    thread = threading.Thread(target=lambda: result.append(start(adapter)))
    thread.start()
    assert entered.wait(1)
    adapter.dispatch("lidar_depth", {"action": "stop", "instance_id": KEY})
    release.set()
    thread.join(2)
    assert result[0]["state"] == "idle"
    assert not executor.nodes and not adapter._nodes


def test_camera_metadata_is_inherited_without_inventing_rgb_identity(plugin):
    adapter, _ = plugin
    declaration = {"schema": "motus.camera/1", "id": "orbbec/depth",
                   "width": 640, "height": 480, "K": [400, 0, 320, 0, 400, 240, 0, 0, 1],
                   "source": "vendor", "pipeline": ["driver/camera_depth"]}
    result = start(adapter, camera_info={TOPIC: declaration})
    entry = result["camera_info"][0]
    assert entry["id"] == "orbbec/depth"
    assert entry["K"] == declaration["K"]
    assert entry["pipeline"][-1] == "perception/lidar_depth"
    assert entry["topic"] == result["topic_out"][0]["topic"]


@pytest.mark.parametrize("args", [
    {"input_topic": "/cam", "input_bindings": {"/cam": "image/jpeg"}},
    {"input_topic": "/cam/visual_depth"},
    {"input_topic": "relative"}, {"input_topic": "/cam bad/depth"},
    {"input_topics": ["/cam/a", "/cam/b"]},
])
def test_bad_or_ambiguous_input_is_rejected(plugin, args):
    adapter, executor = plugin
    result = adapter.dispatch("lidar_depth", {"action": "start", **args})
    assert result["state"] == "error"
    assert not executor.nodes


def test_explicit_format_can_select_depth_among_camera_inputs(plugin):
    adapter, _ = plugin
    result = adapter.dispatch("lidar_depth", {
        "action": "start", "input_topics": ["/cam/rgb", TOPIC],
        "input_bindings": [{"topic": "/cam/rgb", "format": "image/jpeg"},
                           {"topic": TOPIC, "format": ld.FORMAT}]})
    assert result["state"] == "running" and result["input"] == TOPIC


@pytest.mark.parametrize("cfg", [{"fps": 0}, {"fps": float("nan")},
                                 {"fps": True}, {"max_frame_age_ms": 0},
                                 {"rgb_alignment_verified": "false"}, {"cal_b": -1}])
def test_invalid_settings_are_not_silently_accepted(plugin, cfg):
    adapter, _ = plugin
    assert adapter.dispatch("lidar_depth", {"action": "config", **cfg})["state"] == "error"


def test_duplicate_input_and_live_global_config_are_refused(plugin):
    adapter, _ = plugin
    start(adapter)
    assert start(adapter, instance_id="second")["state"] == "error"
    assert adapter.dispatch("lidar_depth", {"action": "config", "fps": 5})["state"] == "error"
    assert adapter.dispatch("lidar_depth", {"action": "calibrate"}) is None


def test_bundle_registers_dispatches_and_can_disable_the_card():
    import main
    cfg = {"plugins": {"echo_cancellation": {"enabled": False},
                       "audio_mixer": {"enabled": False},
                       "lidar_depth": {"enabled": True}}}
    bundle = main.PerceptionBundle(cfg, _FakeExecutor())
    try:
        assert bundle.owns("lidar_depth")
        assert bundle.dispatch("lidar_depth", {"action": "info"})["state"] == "idle"
        assert "lidar_depth" in [t["name"] for t in bundle.get_all_tools()]
    finally:
        bundle.shutdown()
    cfg["plugins"]["lidar_depth"]["enabled"] = False
    bundle = main.PerceptionBundle(cfg, _FakeExecutor())
    assert not bundle.owns("lidar_depth")
