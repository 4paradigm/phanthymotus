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

def test_stereo_start_requires_calibration():
    plugin, executor = _plugin()
    reply = plugin.dispatch("pointcloud", {
        "action": "start", "input_topics": ["/cam/left", "/cam/right"]})
    assert reply["state"] == "error"


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
        fx, cx, cy, Tx, L, R, map1 = pc._rectify_pair(left, right, RAW_BLOB)
        # 基线长度 = |T[0]| = 0.05，符号归一到负
        assert Tx == pytest.approx(-0.05)
        assert fx == pytest.approx(300.0)
        assert cx == pytest.approx(160.0) and cy == pytest.approx(120.0)
        # make_Q 里 -1/Tx 不再除零：Q[3,2] > 0
        Q = pc.Q_from_baseline(fx, cx, cy, Tx)
        assert Q[3, 2] == pytest.approx(20.0)  # 1/0.05
    finally:
        fake.restore()


def test_rectify_pair_rectified_tier_uses_blob_tx():
    fake = _install_fake_stereo_rectify()
    try:
        left = np.zeros((6, 8, 3), dtype=np.uint8)
        right = np.zeros((6, 8, 3), dtype=np.uint8)
        fx, cx, cy, Tx, L, R, map1 = pc._rectify_pair(left, right, RECTIFIED_BLOB)
        assert Tx == pytest.approx(-0.05)  # abs → 负号归一
        assert map1 is None
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
