"""Depth-only pointcloud lifecycle, projection, calibration and metadata tests."""

from __future__ import annotations

import json
import struct
import types
import zlib

import numpy as np
import pytest

from vision_stubs import (
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeImage,
    _FakeNode,
    _wait_until,
)

import plugins.pointcloud as pc  # noqa: E402

def _plugin(cfg=None):
    executor = _FakeExecutor()
    plugin = pc.PointCloudPerceptionPlugin(cfg or {}, "testns", executor)
    return plugin, executor


def _cloud_pub(node):
    return next(p for p in node.publishers if p.topic.endswith("/pointcloud"))


def _summary_pub(node):
    return next(p for p in node.publishers if p.topic.endswith("/pointcloud_summary"))


def _decode_cloud(raw: bytes) -> np.ndarray:
    magic, n = struct.unpack("<II", raw[:8])
    assert magic == 12
    return np.frombuffer(raw[8:], dtype="<f4").reshape(n, 3)


def test_pointcloud_no_longer_advertises_or_executes_calibration_action():
    tool = pc.TOOLS[0]
    assert 'calibrate' not in tool['inputSchema']['properties']['action']['enum']
    assert 'calibrate' not in tool['inputSchema'].get('x-action-params', {})
    plugin, _ = _plugin()
    assert plugin.dispatch('pointcloud', {'action': 'calibrate'}) is None


@pytest.mark.parametrize(('format_name', 'expected'), [
    ('image/depth-z16', '_depth_image_cb'),
    ('image/depth-zlib', '_depth_comp_cb'),
])
def test_declared_input_format_uses_only_its_ros_message_type(format_name, expected):
    plugin, executor = _plugin()
    result = plugin.dispatch('pointcloud', {
        'action': 'start', 'input_topic': '/camera/depth',
        'camera_info': {'/camera/depth': {'format': format_name}},
    })
    assert result['state'] == 'running'
    assert [sub.callback.__name__ for sub in executor.nodes[0].subscriptions] == [expected]
    plugin.dispatch('pointcloud', {'action': 'stop'})


def test_float_depth_frames_are_rejected():
    """点云卡片不接收 32FC1 米制浮点深度：驱动侧只有 uint16 毫米深度，
    浮点帧按不可用拒掉，绝不拿它当米制值反投影。"""
    msg = _FakeImage(data=np.array([[1.0, 2.5], [3.0, 4.0]], dtype="<f4").tobytes(),
                     encoding="32FC1", width=2, height=2, step=8)
    assert pc._decode_depth_message(msg) == (None, "32FC1")


@pytest.mark.parametrize("endian,big", [("<", 0), (">", 1)])
def test_uint16_depth_keeps_metres_with_row_padding(endian, big):
    rows = [np.array([1000, 2500, 9999], dtype=endian + "u2").tobytes() + b"pp",
            np.array([3000, 4000, 9999], dtype=endian + "u2").tobytes() + b"pp"]
    msg = _FakeImage(data=b"".join(rows), encoding="16UC1", width=3,
                     height=2, step=8, is_bigendian=big)
    result, encoding = pc._decode_depth_message(msg)
    assert encoding == "16UC1"
    np.testing.assert_allclose(result, [[1.0, 2.5, 9.999], [3.0, 4.0, 9.999]])


def test_metric_cloud_preserves_header_and_optical_coordinates():
    plugin, executor = _plugin({"metric_cloud": True})
    camera = metric_z16_declaration()
    plugin.dispatch("pointcloud", {"action": "start", "mode": "depth", "input_topic": "/depth",
                                   "camera_info": {"/depth": camera}})
    node = executor.nodes[0]
    msg = _FakeImage(data=np.array([2000, 2000], dtype="<u2").tobytes(),
                     encoding="16UC1", width=2, height=1, step=4)
    msg.header.frame_id = "rgb_optical"
    msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
    node._emit_depth(msg)
    pub = next(p for p in node.publishers if p.topic == "/depth/pointcloud_metric")
    cloud = pub.raw_messages[0]
    assert cloud.header.frame_id == "rgb_optical"
    assert (cloud.header.stamp.sec, cloud.header.stamp.nanosec) == (123, 456)
    assert cloud.point_step == 12 and cloud.row_step == 24
    np.testing.assert_allclose(np.frombuffer(cloud.data, dtype="<f4").reshape(-1, 3),
                               [[0, 0, 2], [1, 0, 2]])
    assert _cloud_pub(node).messages  # existing visualization stays available
    info = plugin.dispatch("pointcloud", {"action": "info"})
    assert {"topic": "/depth/pointcloud_metric", "format": "sensor/pointcloud2"} in info["topic_out"]
    assert {"topic": "/depth", "format": "image/depth-z16"} in info["topic_in"]
    projection = next(c for c in info["camera_info"] if c["topic"] == pub.topic)
    assert projection["id"] == camera["id"]
    assert projection["K"] == [2., 0., 0., 0., 2., 0., 0., 0., 1.]
    node.stop()


def test_metric_cloud_refuses_unknown_frame_or_inferred_calibration():
    plugin, executor = _plugin({"metric_cloud": True})
    plugin.dispatch("pointcloud", {"action": "start", "mode": "depth", "input_topic": "/depth", "camera_info": [metric_z16_declaration(1)]})
    node = executor.nodes[0]
    msg = _FakeImage(data=np.array([2000], dtype="<u2").tobytes(),
                     encoding="16UC1", width=1, height=1, step=2)
    msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
    node._emit_depth(msg)
    pub = next(p for p in node.publishers if p.topic == "/depth/pointcloud_metric")
    assert not pub.messages
    msg.header.frame_id = "optical"
    msg.header.stamp = types.SimpleNamespace(sec=0, nanosec=0)
    node._emit_depth(msg)
    assert not pub.messages
    msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
    msg.width = 2
    msg.step = 4
    msg.data = np.array([2000, 2000], dtype="<u2").tobytes()
    node._emit_depth(msg)
    assert not pub.messages
    node.stop()


def test_metric_info_explains_missing_frame_and_rejected_float_depth():
    plugin, executor = _plugin({'metric_cloud': True})
    plugin.dispatch('pointcloud', {'action': 'start', 'mode': 'depth', 'input_topic': '/depth', 'camera_info': [metric_z16_declaration(1)]})
    node = executor.nodes[0]
    msg = _FakeImage(data=np.array([2000], dtype='<u2').tobytes(), encoding='16UC1',
                     width=1, height=1, step=2)
    msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
    node._emit_depth(msg)
    info = plugin.dispatch('pointcloud', {'action': 'info'})['instances']['/depth']['metric_cloud']
    assert info['published'] == 0 and info['last_skip_reason'] == 'missing_frame'
    assert info['skipped']['missing_frame'] == 1
    # 32FC1 不再被解码，帧在 _decode_depth_message 就被拒（invalid_depth）。
    bad = _FakeImage(data=np.array([2.], dtype='<f4').tobytes(), encoding='32FC1',
                     width=1, height=1, step=4)
    node._emit_depth(bad)
    info = plugin.dispatch('pointcloud', {'action': 'info'})['instances']['/depth']['metric_cloud']
    assert info['last_skip_reason'] == 'invalid_depth'
    node.stop()


def test_metric_start_rejects_explicit_non_metric_upstream_format():
    plugin, executor = _plugin({'metric_cloud': True})
    declaration = {**_UPSTREAM_DECL, 'topic': '/depth', 'format': 'image/depth-z16'}
    result = plugin.dispatch('pointcloud', {'action': 'start', 'mode': 'depth',
                                             'input_topic': '/depth', 'camera_info': [declaration]})
    assert result['state'] == 'error' and not executor.nodes


def test_metric_cloud_does_not_start_with_inferred_intrinsics():
    plugin, executor = _plugin({"metric_cloud": True})
    result = plugin.dispatch("pointcloud", {"action": "start", "mode": "depth", "input_topic": "/depth"})
    assert result["state"] == "error"
    assert not executor.nodes


def test_global_config_can_enable_metric_cloud():
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud", {"action": "config", "metric_cloud": True})
    result = plugin.dispatch("pointcloud", {"action": "start", "mode": "depth", "input_topic": "/depth", "camera_info": [metric_z16_declaration(1)]})
    assert result["state"] == "running"
    assert executor.nodes[0]._metric_cloud_enabled
    executor.nodes[0].stop()


def test_canvas_metric_config_starts_depth_without_mode_argument():
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud", {"action": "config", "instance_id": "card-1",
                                   "metric_cloud": True})
    result = plugin.dispatch("pointcloud", {"action": "start", "instance_id": "card-1",
                                            "input_topic": "/depth", "camera_info": [metric_z16_declaration(1)]})
    assert result["state"] == "running"
    node = executor.nodes[0]
    assert node._mode == "depth"
    assert [s.callback.__name__ for s in node.subscriptions] == ["_depth_image_cb"]
    node.stop()


def test_metric_cloud_accepts_calibrated_depth_declaration():
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud", {"action": "config", "instance_id": "card-1", "metric_cloud": True})
    declaration = {"schema": "motus.camera/1", "topic": "/depth", "format": "image/depth-z16",
                   "id": "optical", "source": "derived-from-K", "distortion_model": "pinhole",
                   "width": 2, "height": 1, "K": [2., 0., 0., 0., 2., 0., 0., 0., 1.],
                   "vendor": {"depth_unit": "mm", "depth_source": "measured", "timestamp_semantics": "source_capture_mapped"}}
    result = plugin.dispatch("pointcloud", {"action": "start", "instance_id": "card-1",
                                            "input_topic": "/depth", "camera_info": [declaration]})
    assert result["state"] == "running"
    assert executor.nodes[0]._calibration == {"fx": 2., "fy": 2., "cx": 0., "cy": 0.,
                                               "width": 2, "height": 1}
    executor.nodes[0].stop()


@pytest.mark.parametrize('source,model', [('manual', 'pinhole'), ('derived-from-K', 'unknown')])
def test_metric_cloud_rejects_unverified_or_distorted_declaration(source, model):
    plugin, executor = _plugin({"metric_cloud": True})
    declaration = {"schema": "motus.camera/1", "topic": "/depth", "format": "image/depth-z16",
                   "id": "optical", "source": source, "distortion_model": model,
                   "width": 2, "height": 1, "K": [2., 0., 0., 0., 2., 0., 0., 0., 1.]}
    result = plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/depth",
                                            "camera_info": [declaration]})
    assert result["state"] == "error"
    assert not executor.nodes


def test_metric_cloud_rejects_declaration_without_capture_time_semantics():
    plugin, executor = _plugin({"metric_cloud": True})
    declaration = {"schema": "motus.camera/1", "topic": "/depth", "format": "image/depth-z16",
                   "id": "optical", "source": "derived-from-K", "distortion_model": "pinhole",
                   "width": 2, "height": 1, "K": [2., 0., 0., 0., 2., 0., 0., 0., 1.]}
    result = plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/depth",
                                            "camera_info": [declaration]})
    assert result["state"] == "error"
    assert not executor.nodes


def test_metric_cloud_rejects_intrinsics_with_unhandled_skew():
    plugin, executor = _plugin({"metric_cloud": True})
    declaration = {"schema": "motus.camera/1", "topic": "/depth", "format": "image/depth-z16",
                   "id": "optical", "source": "derived-from-K", "distortion_model": "pinhole",
                   "width": 2, "height": 1, "K": [2., .2, 0., 0., 2., 0., 0., 0., 1.],
                   "vendor": {"depth_unit": "mm", "depth_source": "measured", "timestamp_semantics": "source_capture_mapped"}}
    result = plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/depth",
                                            "camera_info": [declaration]})
    assert result["state"] == "error"
    assert not executor.nodes


def test_metric_depth_subscribes_only_to_image_type():
    plugin, executor = _plugin({"metric_cloud": True})
    plugin.dispatch("pointcloud", {"action": "start", "mode": "depth", "input_topic": "/depth", "camera_info": [metric_z16_declaration(1)]})
    node = executor.nodes[0]
    assert [s.callback.__name__ for s in node.subscriptions] == ["_depth_image_cb"]
    node.stop()


def test_metric_cloud_keeps_source_header_through_worker_queue():
    plugin, executor = _plugin({"metric_cloud": True, "fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "mode": "depth", "input_topic": "/depth", "camera_info": [metric_z16_declaration(1)]})
    node = executor.nodes[0]
    msg = _FakeImage(data=np.array([2000], dtype="<u2").tobytes(),
                     encoding="16UC1", width=1, height=1, step=2)
    msg.header.frame_id = "optical"
    msg.header.stamp = types.SimpleNamespace(sec=456, nanosec=789)
    node._depth_image_cb(msg)
    pub = next(p for p in node.publishers if p.topic == "/depth/pointcloud_metric")
    assert _wait_until(lambda: bool(pub.raw_messages))
    assert pub.raw_messages[0].header.stamp.sec == 456
    assert pub.raw_messages[0].header.stamp.nanosec == 789
    assert pub.raw_messages[0].header.frame_id == "optical"
    node.stop()


def _wait_cloud_and_summary(node):
    """等一对输出。_publish 先发云再发摘要，只等云就断言摘要会撞上
    中间态（review 指出的竞态），两个都等到才算一帧完整落地。"""
    cloud_pub = _cloud_pub(node)
    summary_pub = _summary_pub(node)
    assert _wait_until(
        lambda: bool(cloud_pub.messages) and bool(summary_pub.messages))
    return cloud_pub.messages[0], json.loads(summary_pub.messages[0])


# ── review issue 2：auto 按消息类型分派，不嗅探话题名 ─────────────────────────

@pytest.mark.parametrize(('publisher_type', 'callback'), [
    ('sensor_msgs/msg/Image', '_depth_image_cb'),
    ('sensor_msgs/msg/CompressedImage', '_depth_comp_cb'),
])
def test_undeclared_auto_waits_for_late_publisher_and_binds_one_type(publisher_type, callback):
    plugin, executor = _plugin()
    result = plugin.dispatch('pointcloud', {'action': 'start', 'input_topic': '/cam/image_raw'})
    assert result['state'] == 'running'
    node = executor.nodes[0]
    assert node.subscriptions == []
    timer = node._input_discovery_timer
    assert timer is not None
    node._topic_types['/cam/image_raw'] = [publisher_type]
    timer.callback()
    assert [sub.callback.__name__ for sub in node.subscriptions] == [callback]
    expected_format = 'image/depth-z16' if publisher_type.endswith('/Image') else 'image/depth-zlib'
    assert plugin.dispatch('info', {})['topic_in'] == [{'topic': '/cam/image_raw', 'format': expected_format}]
    assert node._input_discovery_timer is None
    assert timer.cancelled
    assert node._worker is not None and node._worker.is_alive()
    plugin.shutdown()


def test_undeclared_auto_rejects_conflicting_publisher_types(monkeypatch):
    plugin, executor = _plugin()
    original = pc._PointCloudNode._input_ros_type

    def both_types(self):
        self._topic_types['/cam/image_raw'] = [
            'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage']
        return original(self)

    monkeypatch.setattr(pc._PointCloudNode, '_input_ros_type', both_types)
    result = plugin.dispatch('pointcloud', {'action': 'start', 'input_topic': '/cam/image_raw'})
    assert result['state'] == 'error'
    assert 'conflicting ROS publisher types' in result['message']
    assert not executor.nodes
    plugin.shutdown()


def test_auto_mode_depth_image_on_rgb_named_topic_publishes():
    """z16 Image 发在 '/camera/image_raw'（不含 'depth'）也必须出云。

    旧实现按 '"depth" in input_topic' 判成 mono，只订 CompressedImage，
    Image 帧静默丢弃、一帧云都不出——review issue 2 的原始复现。
    """
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/camera/image_raw",
                                   "camera_info": {"/camera/image_raw": {"format": "image/depth-z16"}}})
    node = executor.nodes[0]

    depth_mm = np.full((4, 4), 1500, dtype="<u2")
    msg = _FakeImage(data=depth_mm.tobytes(), encoding="16UC1",
                     width=4, height=4, step=8)
    image_cb = next(s.callback for s in node.subscriptions
                    if s.callback.__name__ == "_depth_image_cb")
    image_cb(msg)

    _, summary = _wait_cloud_and_summary(node)
    assert summary["mode"] == "depth"
    assert summary["nearest"] == pytest.approx(1.5, abs=1e-3)


def test_auto_mode_depth_zlib_compressed_image_publishes():
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/whatever",
                                   "camera_info": {"/cam/whatever": {"format": "image/depth-zlib"}}})
    node = executor.nodes[0]

    raw = np.full((480, 640), 2500, dtype="<u2").tobytes()
    msg = _FakeCompressedImage(zlib.compress(raw), fmt="depth-zlib; compressedDepth")
    comp_cb = next(s.callback for s in node.subscriptions
                   if s.callback.__name__ == "_depth_comp_cb")
    comp_cb(msg)

    _, summary = _wait_cloud_and_summary(node)
    assert summary["mode"] == "depth"
    assert summary["nearest"] == pytest.approx(2.5, abs=1e-3)


def test_auto_mode_recognizes_visual_depth_producer_format():
    """review 第 9 轮：栈内生产者 visual_depth 发的格式串是
    "16UC1; compressedDepth zlib"（visual_depth.py _publish），不含小写
    "depth" —— 旧实现按 `"depth" in fmt` 匹配把它判成 JPEG：触发 TRT
    引擎下载、JPEG 解码失败、一帧云都不出。小写归一化后必须出云。
    """
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/depth_raw",
                                   "camera_info": {"/cam/depth_raw": {"format": "image/depth-zlib"}}})
    node = executor.nodes[0]

    raw = np.full((480, 640), 3000, dtype="<u2").tobytes()
    msg = _FakeCompressedImage(zlib.compress(raw), fmt="16UC1; compressedDepth zlib")
    comp_cb = next(s.callback for s in node.subscriptions
                   if s.callback.__name__ == "_depth_comp_cb")
    comp_cb(msg)

    _, summary = _wait_cloud_and_summary(node)
    assert summary["mode"] == "depth"
    assert summary["nearest"] == pytest.approx(3.0, abs=1e-3)




@pytest.mark.parametrize(('format_name', 'expected'), [
    ('image/depth-z16', '_depth_image_cb'),
    ('image/depth-zlib', '_depth_comp_cb'),
])
def test_explicit_depth_mode_subscribes_only_to_declared_type(format_name, expected):
    plugin, executor = _plugin()
    result = plugin.dispatch('pointcloud', {'action': 'start',
        'input_topic': '/cam/depth', 'mode': 'depth',
        'camera_info': {'/cam/depth': {'format': format_name}}})
    assert result['state'] == 'running'
    assert [sub.callback.__name__ for sub in executor.nodes[0].subscriptions] == [expected]
    plugin.shutdown()


# ── 深度链路本身（A 路）──────────────────────────────────────────────────────

def test_depth_z16_image_publishes_cloud():
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]

    depth_mm = np.full((4, 4), 2000, dtype="<u2")
    msg = _FakeImage(data=depth_mm.tobytes(), encoding="16UC1",
                     width=4, height=4, step=8)
    node._depth_image_cb(msg)

    raw_cloud, summary = _wait_cloud_and_summary(node)
    cloud = _decode_cloud(raw_cloud)
    assert cloud.shape[0] > 0
    assert summary["nearest"] == pytest.approx(2.0, abs=1e-3)


def test_depth_zlib_compressed_publishes_cloud():
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]

    raw = np.full((480, 640), 3000, dtype="<u2").tobytes()
    node._depth_comp_cb(_FakeCompressedImage(zlib.compress(raw), fmt="depth-zlib"))

    _, summary = _wait_cloud_and_summary(node)
    assert summary["nearest"] == pytest.approx(3.0, abs=1e-3)


def test_depth_image_with_wrong_encoding_is_ignored():
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]

    msg = _FakeImage(data=b"\x00" * 32, encoding="rgb8",
                     width=4, height=4, step=12)
    node._depth_image_cb(msg)
    time_wait = _wait_until(lambda: False, timeout=0.2)
    assert _cloud_pub(node).messages == []


def test_truncated_depth_image_is_ignored():
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]

    msg = _FakeImage(data=b"\x00" * 4, encoding="16UC1",
                     width=4, height=4, step=8)
    node._depth_image_cb(msg)
    _wait_until(lambda: False, timeout=0.2)
    assert _cloud_pub(node).messages == []


def test_depth_respects_min_max_range():
    plugin, executor = _plugin(cfg={"fps": 1000, "min_depth_m": 1.0, "max_depth_m": 3.0})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]

    depth_mm = np.array([[500, 2000, 3500]], dtype="<u2")  # 0.5 / 2.0 / 3.5 m
    msg = _FakeImage(data=depth_mm.tobytes(), encoding="16UC1",
                     width=3, height=1, step=6)
    node._depth_image_cb(msg)

    _, summary = _wait_cloud_and_summary(node)
    assert summary["nearest"] == pytest.approx(2.0, abs=1e-3)
    # 1x3 图 stride=1，只有中间那点落在 [1, 3] m
    assert summary["points"] == 1


def test_max_points_cap_is_honoured():
    plugin, executor = _plugin(cfg={"fps": 1000, "max_points": 100})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]

    depth_mm = np.full((60, 60), 2000, dtype="<u2")  # 3600 点 > 100
    msg = _FakeImage(data=depth_mm.tobytes(), encoding="16UC1",
                     width=60, height=60, step=120)
    node._depth_image_cb(msg)

    _, summary = _wait_cloud_and_summary(node)
    # ceil(sqrt(3600/100)) = 6 → (60/6)^2 = 100 点
    assert summary["points"] == 100


# ── review 建议：mode / input_topics 数量校验 ─────────────────────────────────

def test_start_rejects_unknown_mode():
    """mode 拼错（如 detph）以前静默当 mono 处理 —— z16 帧永远收不到。"""
    plugin, executor = _plugin()
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topic": "/cam/depth", "mode": "detph"})
    assert reply["state"] == "error"
    assert "mode" in reply["message"]
    assert executor.nodes == []


def test_start_rejects_more_than_two_input_topics():
    """第三路以前被静默忽略 —— 双目只认前两路，宁拒不错。"""
    plugin, executor = _plugin()
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/a", "/b", "/c"]})
    assert reply["state"] == "error"
    assert "input_topics" in reply["message"]
    assert executor.nodes == []


@pytest.mark.parametrize("mode", ["depth", "auto"])
def test_start_rejects_explicit_mode_without_any_input(mode):
    """Saved depth/auto calls without an input must not report running."""
    plugin, executor = _plugin()
    reply = plugin.dispatch("pointcloud", {"action": "start", "mode": mode})
    assert reply["state"] == "error"
    assert "input_topic" in reply["message"]
    assert executor.nodes == []


@pytest.mark.parametrize('index,bad', [(0, 0), (0, float('nan')), (0, 'abc'), (4, float('inf')), (2, float('nan')), (5, float('inf')), (8, 0)])
def test_metric_start_rejects_invalid_driver_intrinsics(index, bad):
    decl = metric_z16_declaration()
    decl['K'][index] = bad
    plugin, executor = _plugin({'metric_cloud': True})
    assert plugin.dispatch('start', {'input_topic': '/depth', 'camera_info': [decl]})['state'] == 'error'
    assert not executor.nodes


# ── review 第 17 轮：实例 config 校验 / shutdown / 数值范围 ─────────────────


@pytest.mark.parametrize("cfg,needle", [
    ({"min_depth_m": 5.0, "max_depth_m": 3.0}, "深度范围"),   # min > max
    ({"min_depth_m": 11.0}, "深度范围"),                       # 11 > 默认 max 10
    ({"max_depth_m": 0.05}, "深度范围"),                       # 0.05 < 默认 min 0.1
    ({"min_depth_m": -1.0}, "深度范围"),                       # 负距离
    ({"fps": 0}, "fps"),                                       # 0 fps → 静默钳 0.1
    ({"fps": -3}, "fps"),
    ({"fps": "abc"}, "数值异常"),
])
def test_config_rejects_invalid_ranges(cfg, needle):
    """review 第 17 轮 Issue 3：min>=max / 非正 fps 以前照收 configured——
    反投影的 z 窗口为空，卡片 running 却一帧云不出；fps<=0 被静默
    钳到 0.1 fps。实例与全局同一条检查。"""
    plugin, _ = _plugin()
    reply = plugin.dispatch("pointcloud", {"action": "config", **cfg})
    assert reply["state"] == "error", f"{cfg} 应被拒收"
    assert needle in reply["message"]
    # 未落库
    assert plugin._fps == 2
    assert plugin._min_depth_m == 0.1
    assert plugin._max_depth_m == 10.0

    # 实例分支同一条检查
    plugin2, executor2 = _plugin()
    reply = plugin2.dispatch("pointcloud", {
        "action": "config", "instance_id": "/cam/x", **cfg})
    assert reply["state"] == "error"
    assert executor2.nodes == []


def test_config_accepts_valid_ranges_and_stepwise_updates():
    """预检不误伤：合法范围照常落库；只改一侧、与现值组合仍合法也放行
    （允许分两步把 (0.1, 10) 改成 (5, 20)）。"""
    plugin, _ = _plugin()
    reply = plugin.dispatch("pointcloud", {
        "action": "config", "min_depth_m": 5.0})  # 与默认 max=10 组合合法
    assert reply["status"] == "configured"
    assert plugin._min_depth_m == 5.0

    reply = plugin.dispatch("pointcloud", {
        "action": "config", "max_depth_m": 20.0, "fps": 5})
    assert reply["status"] == "configured"
    assert plugin._max_depth_m == 20.0
    assert plugin._fps == 5


def test_plugin_shutdown_retires_all_depth_nodes():
    """Shutdown retires all depth workers/subscriptions and remains idempotent."""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/a", "mode": "depth"})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/b", "mode": "depth"})
    assert len(executor.nodes) == 2

    plugin.shutdown()

    assert plugin._nodes == {}
    assert executor.nodes == []
    # 幂等：再调一次不炸
    plugin.shutdown()


# ── review 第 18 轮：实例 config 合并语义 / 有界解压 ─────────────────────────

def test_instance_config_patches_instead_of_replacing():
    plugin, executor = _plugin()
    plugin.dispatch('config', {'instance_id': 'card', 'min_depth_m': 2.})
    result = plugin.dispatch('config', {'instance_id': 'card', 'fps': 5})
    assert result['config'] == {'min_depth_m': 2., 'fps': 5}
    plugin.dispatch('start', {'instance_id': 'card', 'input_topic': '/depth', 'mode': 'depth'})
    node = executor.nodes[0]
    assert node._min_depth_m == 2. and node._fps == 5
    plugin.shutdown()


def test_depth_zlib_decompress_is_bounded():
    """review 第 18 轮 Issue 2：zlib.decompress 以前无上限——几百 KB 的
    压缩包能声明几十 MB 的解压产物，在回调里把 perception 进程 OOM。
    有界解压：正常 614400 B 帧放行，超 1 MiB 上限（解压炸弹）拒收，
    截断流照旧报错。"""
    import zlib
    from plugins.pointcloud import zlib_decompress

    # 正常帧：640x480 uint16 = 614400 B
    good = zlib.compress(b"\x27\x10" * 307200, 1)
    assert len(zlib_decompress(good)) == 614400

    # 解压炸弹：40 MB 的零压到几 KB，必须在 1 MiB 处被截停
    bomb = zlib.compress(b"\x00" * (40 << 20), 9)
    assert len(bomb) < 100_000  # 压缩端确实很小
    with pytest.raises(ValueError, match="1 MiB"):
        zlib_decompress(bomb)

    # 略超上限：1.25 MB —— 同样拒收（unconsumed_tail 非空）
    over = zlib.compress(b"\x01" * (1 << 20) + b"\x02" * (1 << 18), 1)
    with pytest.raises(ValueError, match="1 MiB"):
        zlib_decompress(over)

    # 截断流：解不出来，照旧抛 zlib 的错误（外部有 fail_count 兜底）
    with pytest.raises(Exception):
        zlib_decompress(good[:len(good) // 2])


def test_depth_zlib_oversized_payload_does_not_reach_worker():
    """超大压缩帧在解码处被拒（返回 None），不会以任何形式进 worker ——
    回调侧的防线：坏帧只计数告警，进程不倒。"""
    import zlib
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]

    bomb = zlib.compress(b"\x00" * (40 << 20), 9)
    msg = _FakeCompressedImage(bomb, fmt="16UC1; compressedDepth zlib")
    # _depth_comp_cb 只节流入队，解码发生在 worker —— 直接调解码入口
    from plugins.pointcloud import _decode_depth_message
    depth, enc = _decode_depth_message(msg)
    assert depth is None, "超限解压必须被拒，不得产出深度图"


# ── review 第 6 轮：每帧日志节流，坏流不淹容器日志 ───────────────────────────

def test_zlib_decode_failure_logs_once_then_samples(caplog):
    """畸形 depth-zlib 帧以前每回调一条 warning —— 2 fps 的坏流一天
    十几万行。现在第 1 条记录、之后每第 100 条采样，好帧复位计数。"""
    import logging as _logging

    pc._zlib_fail_count = 0
    bad = _FakeCompressedImage(b"not-zlib-at-all", fmt="depth-zlib")
    good = _FakeCompressedImage(
        zlib.compress(np.full((480, 640), 1500, dtype="<u2").tobytes()),
        fmt="depth-zlib")

    with caplog.at_level(_logging.WARNING, logger="plugins.pointcloud"):
        for _ in range(199):
            depth, _ = pc._decode_depth_message(bad)
            assert depth is None
        # 第 1 条与第 100 条；2..99、101..199 静默
        warns = [r for r in caplog.records if "depth-zlib" in r.message]
        assert len(warns) == 2
        assert "第 1 帧" in warns[0].message
        assert "第 100 帧" in warns[1].message
        # 好帧复位：之后的第一条坏帧是新一轮的第 1 条
        depth, _ = pc._decode_depth_message(good)
        assert depth is not None
        depth, _ = pc._decode_depth_message(bad)
        warns = [r for r in caplog.records if "depth-zlib" in r.message]
        assert len(warns) == 3
        assert "第 1 帧" in warns[2].message
    pc._zlib_fail_count = 0


def test_worker_error_logs_full_trace_once_then_samples(caplog):
    """worker 异常以前每帧一条完整 traceback。现在首条带 traceback、
    之后每第 100 条采样且有界，处理成功即复位。"""
    import logging as _logging

    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]

    def _boom(self_inner, msg):
        raise RuntimeError("kaboom " + "x" * 500)

    original = pc._PointCloudNode._emit_depth
    pc._PointCloudNode._emit_depth = _boom
    try:
        with caplog.at_level(_logging.ERROR, logger="plugins.pointcloud"):
            for _ in range(150):
                node._frame_queue.put(("depth_z16", "fake"))
                assert _wait_until(lambda: node._frame_queue.empty(), timeout=2.0)
            errors = [r for r in caplog.records if "worker error" in r.message]
            assert len(errors) == 2
            assert errors[0].exc_info is not None  # 首条带 traceback
            assert "worker error" in errors[0].message  # 首条无计数标记，是状态转换
            assert len(errors[0].message) > 300  # 首条带完整错误详情
            assert "第 100 帧" in errors[1].message
            assert len(errors[1].message) < 300  # 采样条只有有界摘要
            # 好帧复位：之后第一条 worker 错误是新一轮的第 1 条（带 traceback）
            pc._PointCloudNode._emit_depth = original
            good = _FakeImage(data=np.full((4, 4), 2000, dtype="<u2").tobytes(),
                              encoding="16UC1", width=4, height=4, step=8)
            node._frame_queue.put(("depth_z16", good))
            assert _wait_until(lambda: bool(_cloud_pub(node).messages), timeout=2.0)
            # worker 先 publish 后才复位计数，主线程必须等 reset 可见，
            # 否则微秒级的竞争窗口会让这条好帧断言失败。
            assert _wait_until(lambda: node._worker_error_count == 0, timeout=2.0)
            pc._PointCloudNode._emit_depth = _boom
            node._frame_queue.put(("depth_z16", "fake"))
            assert _wait_until(lambda: node._worker_error_count == 1, timeout=2.0)
            errors = [r for r in caplog.records
                      if "worker error" in r.message and r.exc_info is not None]
            assert len(errors) == 2  # 新一轮首条又是完整 traceback
    finally:
        pc._PointCloudNode._emit_depth = original
        node.request_stop()


# ── review issue 3：camera_info 声明沿链路传递 ────────────────────────────────

_UPSTREAM_DECL = {
    "schema": "motus.camera/1", "topic": "/cam/rgb", "format": "image/depth-z16",
    "id": "unitree/r1/camera_main", "width": 1280, "height": 720,
    "distortion_model": "unknown", "D": None,
    "K": [700.0, 0.0, 640.0, 0.0, 700.0, 360.0, 0.0, 0.0, 1.0],
    "half_fov_rad": 0.888, "half_fov_v_rad": None, "source": "measured",
    "measured_on": "r1_sz", "pipeline": ["unitree/r1/camera_main"], "vendor": {},
}


def test_start_records_upstream_declaration_and_info_carries_it():
    """review Medium：以前 start 不收 camera_info、info() 也不传 ——
    上游声明了真实内参，反投影却按 90° hfov 硬算，下游拿不到任何
    motus.camera/1 声明。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start",
                                   "input_topic": "/cam/rgb",
                                   "camera_info": [_UPSTREAM_DECL]})
    info = plugin.dispatch("pointcloud", {"action": "info",
                                          "input_topic": "/cam/rgb"})
    entries = info["camera_info"]
    assert [e["topic"] for e in entries] == [
        "/cam/rgb/pointcloud", "/cam/rgb/pointcloud_summary"]
    assert all(e["schema"] == "motus.camera/1" for e in entries)
    assert all(e["pipeline"][-1] == "perception/pointcloud" for e in entries)
    # 点云不裁剪不缩放（抽稀只丢点不丢视场）：K 与 half_fov_rad 原样继承
    assert all(e["K"] == _UPSTREAM_DECL["K"] for e in entries)
    assert all(e["half_fov_rad"] == 0.888 for e in entries)
    assert all(e["width"] == 1280 and e["height"] == 720 for e in entries)
    assert "camera_info_note" not in info


def test_info_without_upstream_declaration_says_whose_problem_it_is():
    """"上游没声明"和"本卡弄丢了"在下游看来一样，只有前者该找相机卡片
    的人 —— note 必须区分。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pointcloud", {"action": "info",
                                          "input_topic": "/cam/rgb"})
    assert "camera_info" not in info
    assert "上游相机没有声明" in info["camera_info_note"]


def test_retire_node_drops_the_upstream_declaration():
    """重新接线的卡片拿着已不喂它的相机的光学参数回答 info()，比不答
    还糟 —— 声明随节点走。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start",
                                   "input_topic": "/cam/rgb",
                                   "camera_info": [_UPSTREAM_DECL]})
    assert "/cam/rgb" in plugin._upstream_camera
    plugin.dispatch("pointcloud", {"action": "stop"})
    assert "/cam/rgb" not in plugin._upstream_camera


def test_restart_without_declaration_replaces_the_stale_one():
    """重启不再带声明时必须替换旧值，而不是留着过期镜头。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start",
                                   "input_topic": "/cam/rgb",
                                   "camera_info": [_UPSTREAM_DECL]})
    plugin.dispatch("pointcloud", {"action": "stop"})
    plugin.dispatch("pointcloud", {"action": "start",
                                   "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pointcloud", {"action": "info",
                                          "input_topic": "/cam/rgb"})
    assert "camera_info" not in info
    assert "上游相机没有声明" in info["camera_info_note"]


def test_pinhole_for_uses_upstream_k_rescaled_across_resolutions():
    """A/B 模式反投影：上游声明了 K 就用它（按分辨率缩放），而不是按
    90° hfov 硬算横向坐标。无上游无标定时才落 90° 兜底。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start",
                                   "input_topic": "/cam/rgb",
                                   "camera_info": [_UPSTREAM_DECL]})
    node = executor.nodes[0]
    # 同分辨率：原样 K
    fx, fy, cx, cy = node._pinhole_for(1280, 720)
    assert fx == pytest.approx(700.0) and fy == pytest.approx(700.0)
    assert cx == pytest.approx(640.0) and cy == pytest.approx(360.0)
    # 半分辨率：K 按像素缩一半
    fx, fy, cx, cy = node._pinhole_for(640, 360)
    assert fx == pytest.approx(350.0) and fy == pytest.approx(350.0)
    assert cx == pytest.approx(320.0) and cy == pytest.approx(180.0)


def test_display_fallback_uses_declared_fov_without_claiming_metric_calibration():
    declaration = {**_UPSTREAM_DECL, 'K': None, 'half_fov_rad': np.pi / 6,
                   'half_fov_v_rad': None}
    plugin, executor = _plugin()
    result = plugin.dispatch('pointcloud', {'action': 'start', 'mode': 'depth',
        'input_topic': '/cam/rgb', 'camera_info': [declaration]})
    assert result['state'] == 'running'
    node = executor.nodes[0]
    fx, fy, _, _ = node._pinhole_for(640, 480)
    assert fx == pytest.approx(320 / np.tan(np.pi / 6))
    assert fy == pytest.approx(fx)
    assert plugin.dispatch('pointcloud', {'action': 'info'})['instances']['/cam/rgb']['calibration_source'] == 'declared-fov-approximate'
    node.stop()
    metric, _ = _plugin({'metric_cloud': True})
    assert metric.dispatch('pointcloud', {'action': 'start', 'mode': 'depth',
        'input_topic': '/cam/rgb', 'camera_info': [declaration]})['state'] == 'error'


def test_invalid_upstream_k_does_not_override_usable_fov():
    declaration = {**_UPSTREAM_DECL, 'K': [float('nan')] * 9,
                   'half_fov_rad': np.pi / 6, 'half_fov_v_rad': None}
    plugin, executor = _plugin()
    plugin.dispatch('pointcloud', {'action': 'start', 'mode': 'depth',
        'input_topic': '/cam/rgb', 'camera_info': [declaration]})
    node = executor.nodes[0]
    assert node._pinhole_for(640, 480)[0] == pytest.approx(320 / np.tan(np.pi / 6))
    node.stop()


def test_publish_after_stop_is_suppressed():
    node = pc._PointCloudNode('/cam', None, fps=2, max_points=100,
                              min_depth_m=.1, max_depth_m=10., node_suffix='test')
    node.request_stop()
    node._publish(np.array([[0., 0., 1.]]))
    assert not _cloud_pub(node).messages and not _summary_pub(node).messages


def test_running_instance_rejects_input_rebinding():
    plugin, executor = _plugin()
    result = plugin.dispatch('pointcloud', {'action': 'start', 'instance_id': 'card',
                                            'mode': 'depth', 'input_topic': '/first'})
    assert result['state'] == 'running'
    result = plugin.dispatch('pointcloud', {'action': 'start', 'instance_id': 'card',
                                            'mode': 'depth', 'input_topic': '/second'})
    assert result['state'] == 'error'
    assert executor.nodes[0]._input_topic == '/first'
    plugin.dispatch('pointcloud', {'action': 'stop', 'instance_id': 'card'})


def test_pinhole_for_falls_back_to_90deg_without_anything():
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    fx, fy, cx, cy = node._pinhole_for(640, 360)
    assert fx == pytest.approx(320.0)  # 90° hfov → w/2
    assert cx == pytest.approx(320.0) and cy == pytest.approx(180.0)


# ── info / config / 生命周期 ─────────────────────────────────────────────────

def test_info_of_idle_plugin_is_idle():
    plugin, _ = _plugin()
    info = plugin.dispatch("pointcloud", {"action": "info"})
    assert info["state"] == "idle"
    assert info["topic_in"] == []


def test_info_reports_depth_format_for_depth_mode():
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    info = plugin.dispatch("pointcloud", {"action": "info"})
    assert info["topic_in"][0]["format"] == "image/depth-z16"
    assert info["instances"]["/cam/depth"]["mode"] == "depth"


def test_stop_destroys_the_node():
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]
    plugin.dispatch("pointcloud", {"action": "stop"})
    assert executor.nodes == []
    assert node.destroyed is True


def test_config_updates_global_defaults():
    plugin, _ = _plugin()
    plugin.dispatch("pointcloud",
                    {"action": "config", "fps": 5, "max_points": 500, "max_depth_m": 4.0})
    assert plugin._fps == 5
    assert plugin._max_points == 500
    assert plugin._max_depth_m == 4.0


@pytest.mark.parametrize('initial', [False, True])
def test_repeated_effective_config_keeps_running_instance(initial):
    plugin, executor = _plugin()
    cfg = {'instance_id': 'card', 'fps': 2, 'max_points': 20000,
           'min_depth_m': .1, 'max_depth_m': 10., 'metric_cloud': False}
    if initial:
        plugin.dispatch('config', cfg)
    plugin.dispatch('start', {'instance_id': 'card', 'input_topic': '/depth', 'mode': 'depth'})
    node = executor.nodes[0]
    result = plugin.dispatch('config', cfg)
    assert result['status'] == 'configured'
    assert executor.nodes == [node]
    assert plugin._nodes['card'] is node
    assert not node.destroyed


def test_invalid_merged_config_keeps_running_instance_and_previous_settings():
    plugin, executor = _plugin()
    plugin.dispatch('config', {'instance_id': 'card', 'min_depth_m': 3., 'max_depth_m': 5.})
    plugin.dispatch('start', {'instance_id': 'card', 'input_topic': '/depth', 'mode': 'depth'})
    node = executor.nodes[0]
    result = plugin.dispatch('config', {'instance_id': 'card', 'max_depth_m': 2.})
    assert result['state'] == 'error'
    assert executor.nodes == [node]
    assert plugin._instance_configs['card'] == {'min_depth_m': 3., 'max_depth_m': 5.}


def test_equivalent_clamped_config_preserves_node_but_actual_change_retires_it():
    plugin, executor = _plugin({'max_points': 100})
    plugin.dispatch('start', {'instance_id': 'card', 'input_topic': '/depth', 'mode': 'depth'})
    node = executor.nodes[0]
    plugin.dispatch('config', {'instance_id': 'card', 'max_points': 50})
    assert executor.nodes == [node]
    plugin.dispatch('config', {'instance_id': 'card', 'max_points': 200})
    assert not executor.nodes and node.destroyed
    plugin.dispatch('start', {'instance_id': 'card', 'input_topic': '/depth', 'mode': 'depth'})
    assert executor.nodes[0]._max_points == 200
    plugin.shutdown()


def test_output_topics_follow_input():
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]
    assert node._cloud_topic == "/cam/depth/pointcloud"
    assert node._summary_topic == "/cam/depth/pointcloud_summary"


def test_multi_instance_info_never_reports_default_topics():
    """review 第 12 轮：多实例时 info() 不再把默认话题 /perception/...
    塞进 topic_out —— 每个节点只发自己的 {input}/... 对，Agent Core 按
    topic_out 接线，默认话题会把第二张卡之后的下游引向不存在的话题。"""
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/a",
                                    "mode": "depth"})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/b",
                                    "mode": "depth"})
    assert len(executor.nodes) == 2

    # 不指定实例：全部实例的成对输出，且不含任何默认话题
    info = plugin.dispatch("pointcloud", {"action": "info"})
    out = [t["topic"] for t in info["topic_out"]]
    assert "/cam/a/pointcloud" in out
    assert "/cam/a/pointcloud_summary" in out
    assert "/cam/b/pointcloud" in out
    assert "/cam/b/pointcloud_summary" in out
    assert "/perception/pointcloud" not in out
    assert "/perception/pointcloud_summary" not in out

    # 指定实例：只有该实例的一对，没有杂音
    info_a = plugin.dispatch("pointcloud",
                             {"action": "info", "instance_id": "/cam/a"})
    assert [t["topic"] for t in info_a["topic_out"]] == [
        "/cam/a/pointcloud", "/cam/a/pointcloud_summary"]


def metric_zlib_declaration():
    return {'schema': 'motus.camera/1', 'topic': '/depth', 'format': 'image/depth-zlib',
            'id': 'optical', 'source': 'derived-from-K', 'distortion_model': 'pinhole',
            'width': 640, 'height': 480, 'K': [2., 0., 0., 0., 2., 0., 0., 0., 1.],
            'vendor': {'depth_unit': 'mm', 'depth_source': 'measured',
                       'timestamp_semantics': 'source_capture_mapped'}}


def metric_z16_declaration(width=2, height=1):
    return {**metric_zlib_declaration(), 'format': 'image/depth-z16',
            'width': width, 'height': height}


def test_card_exposes_existing_depth_formats_without_manual_calibration():
    tool = pc.TOOLS[0]
    assert {p['format'] for p in tool['topic_in']} == {
        'image/depth-z16', 'image/depth-zlib'}
    for schema in ('inputSchema', 'configSchema'):
        assert not {'calibration', 'cal_a', 'cal_b'} & tool[schema]['properties'].keys()
    for action in tool['inputSchema']['x-action-params'].values():
        assert 'calibration' not in action['params']


def test_measured_z16_uses_driver_parameters_despite_saved_manual_calibration():
    old = {'calibration': {'fx': 99., 'fy': 99., 'cx': 1., 'cy': 1.},
           'cal_a': 2., 'cal_b': 3.}
    plugin, executor = _plugin({'metric_cloud': True, **old})
    plugin.dispatch('config', {'instance_id': 'card', **old})
    result = plugin.dispatch('start', {'instance_id': 'card', 'input_topic': '/depth',
        'camera_info': [metric_z16_declaration()], **old})
    assert result['state'] == 'running'
    node = executor.nodes[0]
    try:
        msg = _FakeImage(data=np.array([2000, 2000], dtype='<u2').tobytes(),
                         encoding='16UC1', width=2, height=1, step=4)
        msg.header.frame_id = 'optical'
        msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
        node._emit_depth(msg)
        pub = next(p for p in node.publishers if p.topic == '/depth/pointcloud_metric')
        np.testing.assert_allclose(np.frombuffer(pub.raw_messages[0].data, dtype='<f4').reshape(-1, 3),
                                   [[0., 0., 2.], [1., 0., 2.]])
        assert not {'calibration', 'cal_a', 'cal_b'} & plugin._instance_configs['card'].keys()
    finally:
        plugin.shutdown()


def test_saved_manual_calibration_cannot_enable_metric_output_without_driver_parameters():
    plugin, executor = _plugin({'metric_cloud': True, 'calibration': {
        'fx': 2., 'fy': 2., 'cx': 0., 'cy': 0., 'width': 1, 'height': 1}})
    assert plugin.dispatch('start', {'input_topic': '/depth'})['state'] == 'error'
    assert not executor.nodes


@pytest.mark.parametrize('old', [
    {'calibration': '/missing/camera.json'}, {'calibration': {'fx': 0}},
    {'cal_a': 2., 'cal_b': 3.},
])
def test_saved_manual_settings_do_not_retire_or_override_display_instance(old):
    plugin, executor = _plugin(old)
    plugin.dispatch('start', {'instance_id': 'card', 'input_topic': '/depth', 'mode': 'depth',
                             'camera_info': [metric_z16_declaration()], **old})
    node = executor.nodes[0]
    try:
        plugin.dispatch('config', {'instance_id': 'card', **old})
        assert executor.nodes == [node]
        assert node._pinhole_for(2, 1) == (2., 2., 0., 0.)
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('field,bad', [('depth_source', 'estimated'), ('depth_source', None),
    ('depth_unit', 'm'), ('timestamp_semantics', 'publication')])
def test_metric_z16_requires_measured_source_units_and_capture_time(field, bad):
    declaration = metric_z16_declaration()
    declaration['vendor'][field] = bad
    plugin, executor = _plugin({'metric_cloud': True})
    assert plugin.dispatch('start', {'input_topic': '/depth', 'camera_info': [declaration]})['state'] == 'error'
    assert not executor.nodes


@pytest.mark.parametrize('field,bad', [('D', [.1, 0, 0, 0]), ('width', 0), ('height', None)])
def test_metric_rejects_distortion_and_missing_image_dimensions(field, bad):
    declaration = metric_z16_declaration()
    declaration[field] = bad
    plugin, executor = _plugin({'metric_cloud': True})
    assert plugin.dispatch('start', {'input_topic': '/depth', 'camera_info': [declaration]})['state'] == 'error'
    assert not executor.nodes


@pytest.mark.parametrize('endian,big', [('<', 0), ('>', 1)])
@pytest.mark.parametrize('encoding', ['16UC1', 'mono16'])
def test_metric_z16_uses_row_stride_endian_and_millimetre_scale(endian, big, encoding):
    plugin, executor = _plugin({'metric_cloud': True})
    plugin.dispatch('start', {'input_topic': '/depth', 'camera_info': [metric_z16_declaration(2, 2)]})
    node = executor.nodes[0]
    msg = _FakeImage(data=np.array([[2000, 2000, 9999], [2000, 2000, 9999]], dtype=endian+'u2').tobytes(),
                     encoding=encoding, width=2, height=2, step=6, is_bigendian=big)
    msg.header.frame_id = 'optical'
    msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
    try:
        node._emit_depth(msg)
        pub = next(p for p in node.publishers if p.topic.endswith('/pointcloud_metric'))
        np.testing.assert_allclose(np.frombuffer(pub.raw_messages[0].data, dtype='<f4').reshape(-1, 3),
                                   [[0., 0., 2.], [1., 0., 2.], [0., 1., 2.], [1., 1., 2.]])
    finally:
        plugin.shutdown()


def test_metric_zlib_uses_supported_transport_and_preserves_geometry_and_header():
    plugin, executor = _plugin({'metric_cloud': True})
    result = plugin.dispatch('start', {'input_topic': '/depth',
                                       'camera_info': [metric_zlib_declaration()]})
    assert result['state'] == 'running'
    node = executor.nodes[0]
    assert [s.callback.__name__ for s in node.subscriptions] == ['_depth_comp_cb']
    pixels = np.zeros((480, 640), dtype='<u2')
    pixels[0, 0] = pixels[0, 4] = 2000
    msg = _FakeCompressedImage(zlib.compress(pixels.tobytes()), fmt='16UC1; compressedDepth zlib')
    msg.header.frame_id = 'optical'
    msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
    node._depth_comp_cb(msg)
    pub = next(p for p in node.publishers if p.topic == '/depth/pointcloud_metric')
    _wait_until(lambda: bool(pub.raw_messages))
    cloud = pub.raw_messages[0]
    np.testing.assert_allclose(np.frombuffer(cloud.data, dtype='<f4').reshape(-1, 3),
                               [[0., 0., 2.], [4., 0., 2.]])
    assert cloud.header.frame_id == 'optical'
    assert (cloud.header.stamp.sec, cloud.header.stamp.nanosec) == (123, 456)
    info = plugin.dispatch('info', {'instance_id': '/depth'})
    assert info['topic_in'] == [{'topic': '/depth', 'format': 'image/depth-zlib'}]
    node.stop()


@pytest.mark.parametrize('field,bad', [('depth_unit', 'm'), ('depth_source', 'estimated'),
                                      ('timestamp_semantics', 'publication')])
def test_metric_zlib_rejects_unproven_source_even_with_explicit_calibration(field, bad):
    declaration = metric_zlib_declaration()
    declaration['vendor'][field] = bad
    plugin, executor = _plugin({'metric_cloud': True, 'calibration': {
        'fx': 2., 'fy': 2., 'cx': 0., 'cy': 0., 'width': 640, 'height': 480}})
    result = plugin.dispatch('start', {'input_topic': '/depth', 'camera_info': [declaration]})
    assert result['state'] == 'error'
    assert not executor.nodes


@pytest.mark.parametrize('fault,reason', [
    ('missing_frame', 'missing_frame'), ('zero_stamp', 'invalid_capture_stamp'),
    ('short', 'invalid_depth'), ('extra', 'invalid_depth'),
])
def test_metric_zlib_rejects_bad_frames_and_recovers(fault, reason):
    plugin, executor = _plugin({'metric_cloud': True})
    assert plugin.dispatch('start', {'input_topic': '/depth',
        'camera_info': [metric_zlib_declaration()]})['state'] == 'running'
    node = executor.nodes[0]
    pixels = np.full((480, 640), 2000, dtype='<u2').tobytes()
    data = pixels[:-2] if fault == 'short' else pixels + b'xx' if fault == 'extra' else pixels
    msg = _FakeCompressedImage(zlib.compress(data), fmt='16UC1; compressedDepth zlib')
    msg.header.frame_id = '' if fault == 'missing_frame' else 'optical'
    msg.header.stamp = types.SimpleNamespace(sec=0 if fault == 'zero_stamp' else 123, nanosec=456)
    pub = next(p for p in node.publishers if p.topic == '/depth/pointcloud_metric')
    try:
        node._emit_depth(msg)
        assert not pub.raw_messages
        assert node._metric_status()['skipped'][reason] == 1
        msg.data = zlib.compress(pixels)
        msg.header.frame_id = 'optical'
        msg.header.stamp.sec = 124
        node._emit_depth(msg)
        assert len(pub.raw_messages) == 1
        assert pub.raw_messages[0].header.stamp.sec == 124
    finally:
        node.stop()
