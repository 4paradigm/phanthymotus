"""
tests/test_pointcloud_plugin.py — pointcloud 插件层测试（host-side, no GPU）。

三条 load-bearing 回归，对应 review 的三个 issue：

    1. raw 档标定不再从 P1[0,3] 推基线（恒 0 → Q 除零，worker 每帧报错
       却不发布）——用假 cv2 验证 stereoRectify 之后 Tx 来自 T；
    2. auto 模式按消息真实类型分派，不再嗅探话题名——z16 深度 Image 发在
       不含 "depth" 的话题上也必须出云；
    3. calibrate 的 name 不允许路径穿越。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import struct
import threading
import zlib

import numpy as np
import pytest

from vision_stubs import (
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeImage,
    _FakeNode,
    _wait_until,
    frame_bytes,
)

import plugins.pointcloud as pc  # noqa: E402

RAW_BLOB = {
    "K1": [[300, 0, 160], [0, 300, 120], [0, 0, 1]],
    "D1": [0, 0, 0, 0],
    "K2": [[300, 0, 160], [0, 300, 120], [0, 0, 1]],
    "D2": [0, 0, 0, 0],
    "T": [[-0.05], [0], [0]],
}
RECTIFIED_BLOB = {"fx": 300.0, "cx": 160.0, "cy": 120.0, "Tx": 0.05}


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


def _wait_cloud_and_summary(node):
    """等一对输出。_publish 先发云再发摘要，只等云就断言摘要会撞上
    中间态（review 指出的竞态），两个都等到才算一帧完整落地。"""
    cloud_pub = _cloud_pub(node)
    summary_pub = _summary_pub(node)
    assert _wait_until(
        lambda: bool(cloud_pub.messages) and bool(summary_pub.messages))
    return cloud_pub.messages[0], json.loads(summary_pub.messages[0])


# ── review issue 2：auto 按消息类型分派，不嗅探话题名 ─────────────────────────

def test_auto_mode_subscribes_to_both_image_and_compressed():
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/image_raw"})
    node = executor.nodes[0]
    kinds = [type(s.msg_type).__name__ if hasattr(s, "msg_type") else None
             for s in node.subscriptions]
    # fake subscription 没存 msg_type，按回调区分即可
    cbs = {s.callback.__name__ for s in node.subscriptions}
    assert cbs == {"_auto_image_cb", "_auto_comp_cb"}
    assert len(node.subscriptions) == 2


def test_auto_mode_depth_image_on_rgb_named_topic_publishes():
    """z16 Image 发在 '/camera/image_raw'（不含 'depth'）也必须出云。

    旧实现按 '"depth" in input_topic' 判成 mono，只订 CompressedImage，
    Image 帧静默丢弃、一帧云都不出——review issue 2 的原始复现。
    """
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/camera/image_raw"})
    node = executor.nodes[0]

    depth_mm = np.full((4, 4), 1500, dtype="<u2")
    msg = _FakeImage(data=depth_mm.tobytes(), encoding="16UC1",
                     width=4, height=4, step=8)
    image_cb = next(s.callback for s in node.subscriptions
                    if s.callback.__name__ == "_auto_image_cb")
    image_cb(msg)

    _, summary = _wait_cloud_and_summary(node)
    assert summary["mode"] == "depth"
    assert summary["nearest"] == pytest.approx(1.5, abs=1e-3)


def test_auto_mode_depth_zlib_compressed_image_publishes():
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/whatever"})
    node = executor.nodes[0]

    raw = np.full((480, 640), 2500, dtype="<u2").tobytes()
    msg = _FakeCompressedImage(zlib.compress(raw), fmt="depth-zlib; compressedDepth")
    comp_cb = next(s.callback for s in node.subscriptions
                   if s.callback.__name__ == "_auto_comp_cb")
    comp_cb(msg)

    _, summary = _wait_cloud_and_summary(node)
    assert summary["mode"] == "depth"
    assert summary["nearest"] == pytest.approx(2.5, abs=1e-3)


def test_auto_mode_jpeg_frame_routes_to_mono():
    class _FakeModel:
        def __init__(self):
            self.calls = 0

        @property
        def input_size(self):
            return (4, 4)

        def infer(self, frame):
            self.calls += 1
            from plugins.vision_runtime import LetterboxMeta
            return [np.full(frame.shape[:2], 2.0, dtype=np.float32)], \
                LetterboxMeta(1.0, 0, 0, frame.shape[1], frame.shape[0])

    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    model = _FakeModel()
    plugin._model = model  # start 已过，经 _plugin_model 钩子被 worker 取到

    comp_cb = next(s.callback for s in node.subscriptions
                   if s.callback.__name__ == "_auto_comp_cb")
    comp_cb(_FakeCompressedImage(frame_bytes(8, 8), fmt="jpeg"))

    pub = _cloud_pub(node)
    assert _wait_until(lambda: bool(pub.messages) and model.calls >= 1)
    summary = json.loads(_summary_pub(node).messages[0])
    assert summary["mode"] == "mono"


def test_auto_mode_cold_start_jpeg_triggers_lazy_load_and_publishes():
    """review issue 1 的回归：auto 节点冷启动、引擎未就绪时，首帧 jpeg
    必须触发 worker 里的懒加载并在加载完成后出云 —— 旧实现回调先改
    _mode，worker 的懒加载分支永远轮不到，帧被静默丢。"""
    class _FakeModel:
        @property
        def input_size(self):
            return (4, 4)

        def infer(self, frame):
            from plugins.vision_runtime import LetterboxMeta
            return [np.full(frame.shape[:2], 2.0, dtype=np.float32)], \
                LetterboxMeta(1.0, 0, 0, frame.shape[1], frame.shape[0])

    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]

    load_started = threading.Event()
    release_load = threading.Event()

    def _blocking_loader():
        load_started.set()
        release_load.wait(timeout=5.0)
        plugin._model = _FakeModel()   # 加载"完成"：引擎就位

    node._ensure_mono_model = _blocking_loader

    comp_cb = next(s.callback for s in node.subscriptions
                   if s.callback.__name__ == "_auto_comp_cb")
    comp_cb(_FakeCompressedImage(frame_bytes(8, 8), fmt="jpeg"))

    assert load_started.wait(timeout=3.0), "worker never triggered lazy load"
    # 加载期间 info 报 loading（review issue 2）—— 但这里的加载没走
    # _load_model_sync，状态位由下面的显式 mono 测试覆盖。
    assert _cloud_pub(node).messages == []  # 引擎没就绪，不该有云
    release_load.set()

    _, summary = _wait_cloud_and_summary(node)
    assert summary["mode"] == "mono"


def test_info_reports_loading_while_explicit_mono_engine_loads():
    """review issue 2 的回归：显式 mono 后台加载期间 info 必须报 loading
    （带下载进度），而不是 idle。用假 _ensure_model 卡住加载窗口。"""
    plugin, executor = _plugin()

    load_started = threading.Event()
    release_load = threading.Event()

    def _fake_ensure_model():
        load_started.set()
        release_load.wait(timeout=5.0)
        plugin._model = object()

    plugin._ensure_model = _fake_ensure_model
    reply = plugin.dispatch("pointcloud",
                            {"action": "start", "input_topic": "/cam/rgb", "mode": "mono"})
    assert reply["state"] == "loading"

    assert load_started.wait(timeout=3.0)
    info = plugin.dispatch("pointcloud", {"action": "info"})
    assert info["state"] == "loading"

    release_load.set()
    assert _wait_until(
        lambda: plugin.dispatch("pointcloud", {"action": "info"})["state"] == "running")
    # 节点在引擎就绪后才创建
    assert len(executor.nodes) == 1


def test_info_reports_error_when_engine_load_fails():
    plugin, _ = _plugin()

    def _failing_ensure_model():
        raise RuntimeError("download failed: no route to host")

    plugin._ensure_model = _failing_ensure_model
    reply = plugin.dispatch("pointcloud",
                            {"action": "start", "input_topic": "/cam/rgb", "mode": "mono"})
    assert reply["state"] == "loading"

    assert _wait_until(
        lambda: plugin.dispatch("pointcloud", {"action": "info"})["state"] == "error")
    info = plugin.dispatch("pointcloud", {"action": "info"})
    assert "download failed" in info["desc"]
    # 失败后再次 start 直接报错，而不是再挂一次加载
    reply = plugin.dispatch("pointcloud",
                            {"action": "start", "input_topic": "/cam/rgb", "mode": "mono"})
    assert reply["state"] == "error"


def test_rectify_pair_rejects_zero_baseline():
    """坏标定（基线 0）在 _rectify_pair 就报 ValueError，而不是 worker
    每帧撞 Q 除零（review 建议）。"""
    fake = _install_fake_stereo_rectify()
    try:
        blob = json.loads(json.dumps(RAW_BLOB))
        blob["T"] = [[0.0], [0], [0]]
        blob["Tx"] = 0.0
        left = np.zeros((6, 8, 3), dtype=np.uint8)
        right = np.zeros((6, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="基线"):
            pc._rectify_pair(left, right, blob)
    finally:
        fake.restore()


def test_real_path_mono_load_does_not_deadlock(monkeypatch, caplog):
    """review 第 6 轮指控：_load_model_sync 持非重入锁调 _ensure_model
    → 重入死锁，真冷启动永远 loading。缩进上其实没有（state-guard 的
    with 块在调 _ensure_model 前已退出），但旧测试全用假 _ensure_model
    绕过了真实路径 —— 重构一旦真把调用挪进锁里，只有这条能抓住：
    它会卡在 loading 直到 _wait_until 超时。下载器与会话都换成桩，
    锁路径保持原样。"""
    import logging as _logging

    import utils.model_downloader as downloader
    import plugins.vision_runtime as runtime

    class _FakeSession:
        def __init__(self, engine_path, **kw):
            self.path = engine_path

        @property
        def input_size(self):
            return (4, 4)

    monkeypatch.setattr(downloader, "ensure_depth_model",
                        lambda dir_, **kw: {"yolo26n-depth.engine": "/models/depth/fake.engine"})
    monkeypatch.setattr(runtime, "VisionEngineSession", _FakeSession)

    plugin, executor = _plugin()
    with caplog.at_level(_logging.WARNING, logger="plugins.pointcloud"):
        reply = plugin.dispatch("pointcloud",
                                {"action": "start", "input_topic": "/cam/rgb", "mode": "mono"})
        assert reply["state"] == "loading"
        assert _wait_until(
            lambda: plugin.dispatch("pointcloud", {"action": "info"})["state"] == "running")
        assert len(executor.nodes) == 1
    info = plugin.dispatch("pointcloud", {"action": "info"})
    assert info["instances"]["/cam/rgb"]["mode"] == "mono"
    assert isinstance(plugin._model, _FakeSession)


def test_raw_blob_forwards_calibrated_R_to_stereoRectify():
    """review 第 6 轮 issue 2：raw 档 stereoRectify 以前恒传 np.eye(3)，
    标定出的 R（相机间相对旋转）被丢掉 —— 有转角的双目校正映射错，
    SGBM 极线不水平，视差全是噪声。blob 带 R 时必须原样转发。"""
    fake = _install_fake_stereo_rectify()
    seen = {}
    real_rectify = fake._cv2().stereoRectify

    def capturing_rectify(K1, D1, K2, D2, size, R, T, flags=0, alpha=0.0):
        seen["R"] = np.asarray(R)
        return real_rectify(K1, D1, K2, D2, size, R, T, flags=flags, alpha=alpha)

    fake._cv2().stereoRectify = capturing_rectify
    try:
        blob = json.loads(json.dumps(RAW_BLOB))
        theta = np.deg2rad(30.0)
        blob["R"] = [[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]]
        left = np.zeros((6, 8, 3), dtype=np.uint8)
        right = np.zeros((6, 8, 3), dtype=np.uint8)
        pc._rectify_pair(left, right, blob)
        assert "R" in seen, "stereoRectify 没被调用"
        np.testing.assert_allclose(seen["R"], np.asarray(blob["R"]), atol=1e-12)
    finally:
        fake.restore()


def test_legacy_raw_blob_without_R_falls_back_to_identity():
    """老 blob 没有 R 字段：退单位阵（= 旧行为，两相机按共线假设），
    不拒收。"""
    fake = _install_fake_stereo_rectify()
    seen = {}
    real_rectify = fake._cv2().stereoRectify

    def capturing_rectify(K1, D1, K2, D2, size, R, T, flags=0, alpha=0.0):
        seen["R"] = np.asarray(R)
        return real_rectify(K1, D1, K2, D2, size, R, T, flags=flags, alpha=alpha)

    fake._cv2().stereoRectify = capturing_rectify
    try:
        blob = json.loads(json.dumps(RAW_BLOB))
        assert "R" not in blob
        left = np.zeros((6, 8, 3), dtype=np.uint8)
        right = np.zeros((6, 8, 3), dtype=np.uint8)
        fx, cx, cy, Tx, _L, _R = pc._rectify_pair(left, right, blob)
        assert Tx == pytest.approx(-0.05)
        np.testing.assert_allclose(seen["R"], np.eye(3), atol=1e-12)
    finally:
        fake.restore()


def test_explicit_depth_mode_still_subscribes_image_and_compressed():
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]
    cbs = {s.callback.__name__ for s in node.subscriptions}
    assert cbs == {"_depth_image_cb", "_depth_comp_cb"}


def test_explicit_mono_mode_subscribes_compressed_only():
    plugin, executor = _plugin()
    plugin._model = object()  # mono 启动前必须就绪引擎，测试里直接注入
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/rgb", "mode": "mono"})
    node = executor.nodes[0]
    cbs = {s.callback.__name__ for s in node.subscriptions}
    assert cbs == {"_jpeg_cb"}


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


# ── stereo 生命周期与 info 契约 ───────────────────────────────────────────────

def test_stereo_start_without_calibration_runs_capture_only():
    """review 指出的死锁回归：calibrate 要运行中的 stereo 节点，stereo 又
    要求先有标定 → 新卡片永远进不了第一次标定。无标定 stereo 现在以
    capture-only 启动（订阅/配对/存帧，不出云），calibrate 可用。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    assert reply["state"] == "running"
    node = executor.nodes[0]
    assert node._mode == "stereo"

    # 左右帧照常配对存帧（calibrate 的输入），但 unknown 档不出云
    node._left_cb(_FakeCompressedImage(frame_bytes(8, 6), fmt="jpeg"))
    node._right_cb(_FakeCompressedImage(frame_bytes(8, 6), fmt="jpeg"))
    assert _wait_until(lambda: node._last_stereo_pair is not None)
    _wait_until(lambda: False, timeout=0.2)
    assert _cloud_pub(node).messages == []

    # calibrate 不再报 "no stereo instance"，而是走到棋盘检测。
    # fake cv2 没实现棋盘 API → cv2_unavailable；真 cv2 下 8×6 帧放不下
    # 9×6 棋盘，尺寸预检 → no_checkerboard（此前直接 cv2.error 炸掉
    # 整个请求 —— review 指出、镜像里复现的崩溃）。两者都说明已越过
    # "没有实例"这一步。
    reply = plugin.dispatch("pointcloud", {"action": "calibrate", "name": "first"})
    assert reply["ok"] is False
    assert reply["reason"] in ("no_checkerboard", "cv2_unavailable")


def test_stereo_start_requires_both_topics():
    plugin, executor = _plugin(cfg={"calibration": json.dumps(RAW_BLOB)})
    # 不带 mode：单路 auto 合法（首帧判深度/单目），stereo 才要求两路
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "mode": "stereo", "input_topics": ["/cam/left"]})
    assert reply["state"] == "error"


def test_stereo_info_reports_both_input_topics():
    """agent-core 的 _dropped_inputs 在 wanted≥2 时按 info() 判定：
    双目必须报出两路，否则卡片以'忽略了一路输入'失败。"""
    plugin, executor = _plugin(cfg={"calibration": json.dumps(RAW_BLOB)})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    info = plugin.dispatch("pointcloud", {"action": "info"})
    topics = [t["topic"] for t in info["topic_in"]]
    assert topics == ["/cam/left", "/cam/right"]


def test_stereo_frame_pairing_requires_sync_window():
    plugin, executor = _plugin(cfg={"calibration": json.dumps(RECTIFIED_BLOB),
                                    "fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]

    node._left_cb(_FakeCompressedImage(frame_bytes(8, 6), fmt="jpeg"))
    with node._left_lock:
        node._left_latest = (0.0, np.zeros((6, 8, 3), dtype=np.uint8))
    node._right_cb(_FakeCompressedImage(frame_bytes(8, 6), fmt="jpeg"))
    _wait_until(lambda: False, timeout=0.2)
    assert node._stereo_queue.empty()


def test_stereo_pairing_publishes_with_rectified_blob():
    """rectified 档端到端：左右同帧 → 配对 → worker → 发布。

    fake cv2 没实现 SGBM/reprojectImageTo3D，这里在实例上替换
    _emit_stereo，只验证链路（回调→配对→worker→发布）与协议格式。
    """
    plugin, executor = _plugin(cfg={"calibration": json.dumps(RECTIFIED_BLOB),
                                    "fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]

    def _fake_emit_stereo(self_inner, left, right):
        xyz = np.array([[0.0, 0.0, 2.5]], dtype=np.float32)
        self_inner._publish(xyz)

    original = type(node)._emit_stereo
    type(node)._emit_stereo = _fake_emit_stereo
    try:
        node._left_cb(_FakeCompressedImage(frame_bytes(8, 6), fmt="jpeg"))
        node._right_cb(_FakeCompressedImage(frame_bytes(8, 6), fmt="jpeg"))
        pub = _cloud_pub(node)
        assert _wait_until(lambda: bool(pub.messages))
        cloud = _decode_cloud(pub.messages[0])
        assert cloud.shape == (1, 3)
        summary = json.loads(_summary_pub(node).messages[0])
        assert summary["mode"] == "stereo"
    finally:
        type(node)._emit_stereo = original


# ── review issue 1：raw 档基线来自 T，不再除零 ────────────────────────────────

def test_rectify_pair_raw_tier_takes_baseline_from_T():
    """raw 档的 Tx 必须来自外参 T（stereoRectify 把平移放 P2[0,3]，
    P1[0,3] 恒 0 —— 旧实现 -1/0 直接 ZeroDivisionError，worker 每帧
    报错但永不发布）。主机无真 cv2，用假 stereoRectify 复现符号。"""
    fake = _install_fake_stereo_rectify()
    try:
        left = np.zeros((6, 8, 3), dtype=np.uint8)
        right = np.zeros((6, 8, 3), dtype=np.uint8)
        fx, cx, cy, Tx, L, R = pc._rectify_pair(left, right, RAW_BLOB)
        # 基线长度 = |T[0]| = 0.05，符号归一到负
        assert Tx == pytest.approx(-0.05)
        assert fx == pytest.approx(300.0)
        assert cx == pytest.approx(160.0) and cy == pytest.approx(120.0)
        # make_Q 里 -1/Tx 不再除零：Q[3,2] > 0
        Q = pc.Q_from_baseline(fx, cx, cy, Tx)
        assert Q[3, 2] == pytest.approx(20.0)  # 1/0.05
    finally:
        fake.restore()


def test_rectify_pair_raw_tier_maps_cached_per_size():
    """review 指出：docstring 说缓存 rectify 映射、代码却每帧重算
    stereoRectify + initUndistortRectifyMap。缓存以帧尺寸为键：同尺寸
    第二帧不再碰 stereoRectify，换尺寸重算一次。"""
    fake = _install_fake_stereo_rectify()
    calls = {"rectify": 0}

    real_rectify = fake._cv2().stereoRectify

    def counting_rectify(*args, **kwargs):
        calls["rectify"] += 1
        return real_rectify(*args, **kwargs)

    fake._cv2().stereoRectify = counting_rectify
    try:
        left = np.zeros((6, 8, 3), dtype=np.uint8)
        right = np.zeros((6, 8, 3), dtype=np.uint8)
        cache = {}
        pc._rectify_pair(left, right, RAW_BLOB, cache)
        pc._rectify_pair(left, right, RAW_BLOB, cache)
        assert calls["rectify"] == 1  # 同尺寸：命中缓存
        bigger = np.zeros((12, 16, 3), dtype=np.uint8)
        pc._rectify_pair(bigger, bigger.copy(), RAW_BLOB, cache)
        assert calls["rectify"] == 2  # 换尺寸：重算
    finally:
        fake.restore()


def test_rectify_pair_rectified_tier_uses_blob_tx():
    fake = _install_fake_stereo_rectify()
    try:
        left = np.zeros((6, 8, 3), dtype=np.uint8)
        right = np.zeros((6, 8, 3), dtype=np.uint8)
        fx, cx, cy, Tx, L, R = pc._rectify_pair(left, right, RECTIFIED_BLOB)
        assert Tx == pytest.approx(-0.05)  # abs → 负号归一
    finally:
        fake.restore()


def test_rectify_pair_rectified_tier_rejects_zero_tx():
    """review 指出：rectified 档 Tx=0 以前直接 -abs(0) 放行，
    Q_from_baseline 的 -1/Tx 每帧除零。worker 侧必须拦下。"""
    fake = _install_fake_stereo_rectify()
    try:
        left = np.zeros((6, 8, 3), dtype=np.uint8)
        right = np.zeros((6, 8, 3), dtype=np.uint8)
        blob = {"fx": 300.0, "cx": 160.0, "cy": 120.0, "Tx": 0}
        with pytest.raises(ValueError, match="Tx"):
            pc._rectify_pair(left, right, blob)
    finally:
        fake.restore()


class _FakeStereoRectify:
    """把 cv2.stereoRectify / initUndistortRectifyMap / remap / cvtColor
    打上最小假实现，使 raw 档路径能离线走到 Q 构造。"""

    def __init__(self):
        import sys
        self._sys = sys
        try:
            import cv2
            self._real = cv2
        except ImportError:
            self._real = None
        self._stubs = {}

    def _cv2(self):
        return self._sys.modules.get("cv2")

    def install(self):
        cv2 = self._cv2()
        assert cv2 is not None, "vision_stubs must be imported first"
        self._saved = {name: getattr(cv2, name, None) for name in (
            "stereoRectify", "initUndistortRectifyMap", "remap",
            "cvtColor", "COLOR_BGR2GRAY", "INTER_LINEAR",
            "CALIB_ZERO_DISPARITY", "CV_32FC1")}
        cv2.COLOR_BGR2GRAY = 6
        cv2.CALIB_ZERO_DISPARITY = 1 << 17
        cv2.CV_32FC1 = 5
        cv2.INTER_LINEAR = 1

        def stereoRectify(K1, D1, K2, D2, size, R, T, flags=0, alpha=0.0):
            # 与 OpenCV 相同的关键行为：以左目为参考系，平移落在 P2[0,3]，
            # P1[0,3] == 0 —— 这是旧 bug 的前提。
            w, h = size
            fx = float(K1[0, 0]); cx = float(K1[0, 2]); cy = float(K1[1, 2])
            P1 = np.eye(4, dtype=np.float64) * 0
            P1[0, 0] = fx; P1[1, 1] = fx; P1[2, 2] = 1
            P1[0, 2] = cx; P1[1, 2] = cy
            P2 = P1.copy()
            P2[0, 3] = -float(np.asarray(T).ravel()[0]) * fx
            Q = np.zeros((4, 4))
            return np.eye(3), np.eye(3), P1, P2, Q, None, None

        def initUndistortRectifyMap(K, D, R, P, size, m1type):
            return np.zeros(size, dtype=np.float32), np.zeros(size, dtype=np.float32)

        def remap(src, m1, m2, interp):
            return src

        def cvtColor(src, code):
            return src

        cv2.stereoRectify = stereoRectify
        cv2.initUndistortRectifyMap = initUndistortRectifyMap
        cv2.remap = remap
        cv2.cvtColor = cvtColor
        return self

    def restore(self):
        cv2 = self._cv2()
        for name, value in self._saved.items():
            if value is not None:
                setattr(cv2, name, value)
            else:
                if hasattr(cv2, name) and name not in ("COLOR_BGR2GRAY",):
                    delattr(cv2, name)


def _install_fake_stereo_rectify():
    return _FakeStereoRectify().install()


# ── review issue 3：calibrate 文件名校验 ──────────────────────────────────────

def test_calibrate_rejects_a_traversal_name():
    """'../../x' 之类 name 会写出 /models/stereo_calib 之外 —— 必须拒绝。"""
    plugin, executor = _plugin(cfg={"calibration": json.dumps(RAW_BLOB),
                                    "fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]

    depth = np.zeros((6, 8, 3), dtype=np.uint8)
    node._last_stereo_pair = (depth, depth.copy())

    # 找不到棋盘格也无所谓：name 校验在棋盘检测之前，坏名先被拒。
    reply = plugin.dispatch("pointcloud", {
        "action": "calibrate", "name": "../../evil"})
    assert reply["ok"] is False
    assert reply["reason"] == "bad_name"


def test_calibrate_rejects_path_like_and_dotted_names():
    plugin, executor = _plugin(cfg={"calibration": json.dumps(RAW_BLOB),
                                    "fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]
    depth = np.zeros((6, 8, 3), dtype=np.uint8)
    node._last_stereo_pair = (depth, depth.copy())

    for bad in ("a/b", "a\\b", "..", ".", "x;rm", "pct%"):
        reply = plugin.dispatch("pointcloud", {"action": "calibrate", "name": bad})
        assert reply["ok"] is False, bad
        assert reply["reason"] == "bad_name", bad


def test_calibrate_without_stereo_instance_says_so():
    plugin, executor = _plugin()
    reply = plugin.dispatch("pointcloud", {"action": "calibrate"})
    assert reply["ok"] is False


def test_calibrate_rejects_frame_too_small_for_board():
    """review 指出、镜像里复现的崩溃：8×6 帧配 9×6 棋盘时
    findChessboardCorners 的 adaptiveThreshold 断言失败抛 cv2.error，
    dispatch 只接 ValueError → MCP 请求直接异常。尺寸预检 + cv2.error
    捕获后都按 no_checkerboard 契约回复。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]

    small = np.zeros((6, 8, 3), dtype=np.uint8)  # 8 列 < board_w=9
    node._last_stereo_pair = (small, small.copy())
    reply = plugin.dispatch("pointcloud", {"action": "calibrate", "name": "t"})
    if reply["reason"] == "cv2_unavailable":
        pytest.skip("fake cv2 has no chessboard API — size precheck untested here")
    assert reply["ok"] is False
    assert reply["reason"] == "no_checkerboard"


def test_calibrate_rejects_cv2_error_as_no_checkerboard(monkeypatch):
    """真 cv2 下图能容纳棋盘仍可能断言失败（纯色/过小）—— cv2.error
    必须落回 no_checkerboard 契约，不能让请求抛异常。用假检测触发。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]

    big = np.zeros((60, 80, 3), dtype=np.uint8)  # 放得下 9×6，过尺寸预检
    node._last_stereo_pair = (big, big.copy())

    import cv2

    class _Cv2Error(Exception):
        pass

    def _boom(gray, pattern, flags):
        raise _Cv2Error("adaptiveThreshold assertion failed")

    monkeypatch.setattr(cv2, "findChessboardCorners", _boom, raising=False)
    # fake cv2 没有 cv2.error 名字 —— 插件里 except cv2.error 需要它存在
    monkeypatch.setattr(cv2, "error", _Cv2Error, raising=False)
    # flags 常量同样缺（在 try 之外取值，缺了会先撞 AttributeError）
    monkeypatch.setattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 1, raising=False)
    monkeypatch.setattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 2, raising=False)
    reply = plugin.dispatch("pointcloud", {"action": "calibrate", "name": "t"})
    assert reply["ok"] is False
    assert reply["reason"] == "no_checkerboard"


def test_calibrate_same_pair_is_not_a_new_pose():
    """review 指出的重复采样：连打 15 次 calibrate 不挪棋盘，会采到 15 份
    相同观测，stereoCalibrate 拟合出貌似成功的退化标定。同一对帧（seq
    未变）必须被拒（same_pose），新帧才算新姿态。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]

    # 假棋盘检测：图够大就"找到"角点，坐标随灰度图尺寸变（模拟不同姿态）。
    import cv2

    def _fake_find(gray, pattern, flags):
        pts = np.array([[[(i % gray.shape[1]), (j % gray.shape[0])]]
                        for i in range(pattern[0]) for j in range(pattern[1])],
                       dtype=np.float32)
        return True, pts

    def _fake_subpix(gray, corners, win, zone, crit):
        return corners

    monkey = pytest.MonkeyPatch()
    monkey.setattr(cv2, "findChessboardCorners", _fake_find, raising=False)
    monkey.setattr(cv2, "cornerSubPix", _fake_subpix, raising=False)
    # fake cv2 缺棋盘检测用的常量与 cv2.error —— 一并补上
    monkey.setattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 1, raising=False)
    monkey.setattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 2, raising=False)
    monkey.setattr(cv2, "error", Exception, raising=False)
    monkey.setattr(cv2, "TERM_CRITERIA_EPS", 1, raising=False)
    monkey.setattr(cv2, "TERM_CRITERIA_MAX_ITER", 2, raising=False)
    try:
        node._last_stereo_pair = (np.zeros((60, 80, 3), dtype=np.uint8),) * 2
        node._stereo_pair_seq = 1
        first = plugin.dispatch("pointcloud", {
            "action": "calibrate", "pairs": 3, "name": "t"})
        # 采集未满 3 对会停在 collecting（不碰 stereoCalibrate）。
        assert first["ok"] is True
        assert first["state"] == "collecting"
        assert first["pairs_used"] == 1

        # 同一对帧（seq 没变）：拒绝，不计数
        again = plugin.dispatch("pointcloud", {
            "action": "calibrate", "pairs": 3, "name": "t"})
        assert again["ok"] is False
        assert again["reason"] == "same_pose"
        assert again["pairs_used"] == 1

        # 新帧（seq 变了）：重新计数，pairs_used 前进
        node._stereo_pair_seq = 2
        fresh = plugin.dispatch("pointcloud", {
            "action": "calibrate", "pairs": 3, "name": "t"})
        assert fresh["ok"] is True
        assert fresh["pairs_used"] == 2
    finally:
        monkey.undo()


def _calib_monkeypatch():
    """calibrate 走通全程需要的假 cv2 棋盘 API：能"找到"角点（坐标随
    灰度图尺寸变，模拟不同姿态）+ 假 stereoCalibrate 返回 9 元组。"""
    import cv2

    def _fake_find(gray, pattern, flags):
        pts = np.array([[(i % gray.shape[1]), (j % gray.shape[0])]
                        for i in range(pattern[0]) for j in range(pattern[1])],
                       dtype=np.float32).reshape(-1, 1, 2)
        return True, pts

    def _fake_subpix(gray, corners, win, zone, crit):
        return corners

    def _fake_stereo_calibrate(obj, l, r, K1, D1, K2, D2, size, criteria=None):
        # 与真 stereoCalibrate 同构的 9 元组返回；K/D 就用初值。
        return (0.5,
                np.asarray(K1, dtype=np.float64).copy(),
                np.asarray(D1, dtype=np.float64).copy(),
                np.asarray(K2, dtype=np.float64).copy(),
                np.asarray(D2, dtype=np.float64).copy(),
                np.eye(3, dtype=np.float64),
                np.array([[-0.05], [0.0], [0.0]], dtype=np.float64),
                np.eye(3, dtype=np.float64),
                np.eye(3, dtype=np.float64))

    monkey = pytest.MonkeyPatch()
    monkey.setattr(cv2, "findChessboardCorners", _fake_find, raising=False)
    monkey.setattr(cv2, "cornerSubPix", _fake_subpix, raising=False)
    monkey.setattr(cv2, "stereoCalibrate", _fake_stereo_calibrate, raising=False)
    monkey.setattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 1, raising=False)
    monkey.setattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 2, raising=False)
    monkey.setattr(cv2, "error", Exception, raising=False)
    monkey.setattr(cv2, "TERM_CRITERIA_EPS", 1, raising=False)
    monkey.setattr(cv2, "TERM_CRITERIA_MAX_ITER", 2, raising=False)
    return monkey


def _collect_calib_pairs(plugin, node, target):
    """推进采集到 target 对（每对前自增 seq 模拟新帧）。"""
    reply = None
    for seq in range(1, target + 1):
        node._stereo_pair_seq = seq
        reply = plugin.dispatch("pointcloud", {
            "action": "calibrate", "pairs": target, "name": "t"})
    return reply


def test_calibrate_done_keeps_raw_blob_by_default():
    """review High：旧的"rectify 后量 Δy 分档"是同义反复 —— stereoRectify
    本来就把两图行对齐，任何标定 rectify 后 Δy ≈ 0，必然落 rectified 档、
    把刚算出的 raw K1/K2/D1/D2/R/T 丢掉 → 运行时跳过去畸变，SGBM 吃
    畸变图。默认必须保留 raw blob。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]
    node._last_stereo_pair = (np.zeros((60, 80, 3), dtype=np.uint8),) * 2

    monkey = _calib_monkeypatch()
    try:
        reply = _collect_calib_pairs(plugin, node, 3)
        assert reply["ok"] is True
        assert reply["state"] == "done"
        assert reply["tier"] == "raw"
        blob = reply["calibration"]
        for key in ("K1", "D1", "K2", "D2", "R", "T"):
            assert key in blob, key
        assert "epipolar_dy_px" not in reply
        assert "epipolar_dy_px" not in blob
    finally:
        monkey.undo()


def test_calibrate_pre_rectified_flag_lands_rectified_shortcut():
    """只有调用方显式 pre_rectified=true（上游保证两路已立体校正）才落
    rectified 快捷档 —— blob 只剩 Q 反投影四要素。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    node = executor.nodes[0]
    node._last_stereo_pair = (np.zeros((60, 80, 3), dtype=np.uint8),) * 2

    monkey = _calib_monkeypatch()
    try:
        node._stereo_pair_seq = 1
        reply = plugin.dispatch("pointcloud", {
            "action": "calibrate", "pairs": 1, "name": "t",
            "pre_rectified": True})
        assert reply["ok"] is True
        assert reply["tier"] == "rectified"
        assert set(reply["calibration"]) == {"fx", "cx", "cy", "Tx",
                                             "rms", "pairs_used"}
        assert reply["calibration"]["Tx"] == pytest.approx(0.05)
    finally:
        monkey.undo()


# ── review issue 2：rectified 档 Tx=0 启动即拒 ────────────────────────────────

def test_start_rejects_rectified_calibration_with_zero_tx():
    """review Medium：Tx=0 的 rectified 标定以前 start 照收，worker 里
    Q_from_baseline 的 -1/Tx 每帧除零。必须在 start 就报配置错误。"""
    plugin, executor = _plugin(cfg={"calibration": json.dumps(
        {"fx": 300.0, "cx": 160.0, "cy": 120.0, "Tx": 0})})
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    assert reply["state"] == "error"
    assert "Tx" in reply["message"]
    assert executor.nodes == []


def test_start_rejects_rectified_calibration_with_nonfinite_tx():
    plugin, executor = _plugin(cfg={"calibration": json.dumps(
        {"fx": 300.0, "cx": 160.0, "cy": 120.0, "Tx": "abc"})})
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    assert reply["state"] == "error"
    assert "Tx" in reply["message"]


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


# ── review 第 6 轮：start 预检 raw blob，坏配置当场报错 ─────────────────────

def test_start_rejects_raw_blob_with_missing_K2():
    """K2 缺了以前 start 照收（只查档位），首帧才在 worker 里炸 ——
    卡片显示 running 却一云不出。"""
    blob = json.loads(json.dumps(RAW_BLOB))
    del blob["K2"]
    plugin, executor = _plugin(cfg={"calibration": json.dumps(blob)})
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    assert reply["state"] == "error"
    assert "raw" in reply["message"]
    assert executor.nodes == []


def test_start_rejects_raw_blob_with_malformed_K1():
    """K1 不是 9 元素（3x3 reshape 失败）同理当场报错。"""
    blob = json.loads(json.dumps(RAW_BLOB))
    blob["K1"] = [300, 0, 160, 0, 300, 120]  # 6 元素：reshape(3,3) 必炸
    plugin, executor = _plugin(cfg={"calibration": json.dumps(blob)})
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    assert reply["state"] == "error"
    assert "K1/K2" in reply["message"]
    assert executor.nodes == []


def test_start_rejects_raw_blob_with_zero_baseline():
    """T 全零且没有 Tx：Q 反投影必然除零，start 拒收（与 rectified 档
    的 Tx 校验同一契约）。"""
    blob = json.loads(json.dumps(RAW_BLOB))
    blob["T"] = [[0.0], [0], [0]]
    plugin, executor = _plugin(cfg={"calibration": json.dumps(blob)})
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    assert reply["state"] == "error"
    assert "基线" in reply["message"]
    assert executor.nodes == []


def test_start_accepts_a_good_raw_blob():
    """预检不误伤：合法 raw blob（含 R）照常启动 —— R 本身有 30° 转角
    也照样收，因为转角正是要交给 stereoRectify 处理的东西。"""
    blob = json.loads(json.dumps(RAW_BLOB))
    theta = np.deg2rad(30.0)
    blob["R"] = [[np.cos(theta), -np.sin(theta), 0.0],
                 [np.sin(theta), np.cos(theta), 0.0],
                 [0.0, 0.0, 1.0]]
    plugin, executor = _plugin(cfg={"calibration": json.dumps(blob),
                                    "fps": 1000})
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    assert reply["state"] == "running"
    assert len(executor.nodes) == 1


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
            assert node._worker_error_count == 0
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
    "schema": "motus.camera/1", "topic": "/cam/rgb", "format": "image/jpeg",
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


def test_loading_info_still_carries_the_declaration():
    """loading 早退分支不能把声明丢在地上（visual_depth/vop 的教训）。"""
    plugin, executor = _plugin(cfg={"fps": 1000})
    plugin.dispatch("pointcloud", {"action": "start",
                                   "input_topic": "/cam/rgb",
                                   "camera_info": [_UPSTREAM_DECL]})
    plugin._model_loading = True  # 模拟 mono 引擎加载窗口
    info = plugin.dispatch("pointcloud", {"action": "info",
                                          "input_topic": "/cam/rgb"})
    assert info["state"] == "loading"
    assert [e["topic"] for e in info["camera_info"]] == [
        "/cam/rgb/pointcloud", "/cam/rgb/pointcloud_summary"]


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


def test_output_topics_follow_input():
    plugin, executor = _plugin()
    plugin.dispatch("pointcloud",
                    {"action": "start", "input_topic": "/cam/depth", "mode": "depth"})
    node = executor.nodes[0]
    assert node._cloud_topic == "/cam/depth/pointcloud"
    assert node._summary_topic == "/cam/depth/pointcloud_summary"
