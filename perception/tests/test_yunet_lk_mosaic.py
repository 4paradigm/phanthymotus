import hashlib
import os
import threading
import time

import cv2
import numpy as np
import pytest

from plugins.yunet_lk_mosaic.config import CONFIG_SCHEMA, MosaicConfig, MosaicError
from plugins.yunet_lk_mosaic.model import MODEL_FILE, resolve_model
from plugins.yunet_lk_mosaic.runtime import MosaicEngine, _iou
from plugins.yunet_lk_mosaic.stream import InputFrame, MosaicStream


pytestmark = pytest.mark.skipif(
    not hasattr(cv2, "calcOpticalFlowPyrLK"), reason="requires real OpenCV LK and JPEG",
)


FACE = (80, 65, 155, 142)
OTHER = (200, 65, 275, 142)


def box_at(box, dx):
    x1, y1, x2, y2 = box
    return x1 + dx, y1, x2 + dx, y2


def rows(*boxes, score=.83):
    out = np.zeros((len(boxes), 15), dtype=np.float32)
    for row, (x1, y1, x2, y2) in zip(out, boxes):
        row[:4] = [x1, y1, x2 - x1, y2 - y1]
        row[14] = score
    return out


class Detector:
    def __init__(self, answers):
        self.answers = iter(answers)

    def detect(self, _):
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def close(self):
        pass


class ScriptedFlow:
    def __init__(self, dx=3., error=0., inconsistent=False):
        self.dx, self.error, self.inconsistent = dx, error, inconsistent

    def __getattr__(self, key):
        return getattr(cv2, key)

    def calcOpticalFlowPyrLK(self, prev, gray, points, *args, **kwargs):
        moved = points.copy()
        moved[:, 0, 0] += self.dx
        if self.inconsistent:
            moved[:, 0, 1] += np.linspace(-20, 20, len(points))
        status = np.ones((len(points), 1), np.uint8)
        error = np.full((len(points), 1), self.error, np.float32)
        return moved, status, error


def encode(image, quality=90):
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return jpeg.tobytes()


@pytest.fixture
def picture():
    # Deterministic corners within the scripted face rectangles.
    rng = np.random.default_rng(20261122)
    return rng.integers(0, 256, (180, 320, 3), np.uint8)


@pytest.fixture
def jpeg(picture):
    return encode(picture)


def make_engine(answers, cv=None, **values):
    cfg = MosaicConfig().updated({"output_width": 320, **values})
    return MosaicEngine(None, cfg, cv=cv, detector=Detector(answers))


def analyze(engine, jpeg, frame):
    return engine.analyze(jpeg, frame, frame / 30)


@pytest.mark.parametrize("fps", [15, 20, 24, 30])
@pytest.mark.parametrize("delay", [0, 200, 300, 500])
def test_supported_configs(fps, delay):
    assert MosaicConfig().updated({"target_fps": fps, "publish_delay_ms": delay}).target_fps == fps


@pytest.mark.parametrize("bad", [
    {"target_fps": 29}, {"publish_delay_ms": 100}, {"max_buffer_frames": 0},
    {"detect_every_frames": 6}, {"detect_every_frames": True}, {"unknown": 1},
    {"miss_compensation": 1}, {"confidence_threshold": float("nan")},
    {"stable_expansion": .8}, {"unstable_expansion": 2.0},
])
def test_invalid_configs_raise(bad):
    with pytest.raises(MosaicError):
        MosaicConfig().updated(bad)


def test_config_schema_matches_runtime():
    for field, meta in CONFIG_SCHEMA["properties"].items():
        assert meta["default"] == getattr(MosaicConfig(), field)


def test_no_face_and_local_pixels(jpeg):
    engine = make_engine([None, rows(FACE)], detect_every_frames=1)
    raw = engine.decode(jpeg)
    empty = analyze(engine, jpeg, 1)
    unchanged, tiles = engine.render(empty)
    assert tiles == []
    assert unchanged == encode(raw, quality=72)
    face = analyze(engine, jpeg, 2)
    altered, tiles = engine.render(face)
    assert altered != unchanged
    assert tiles == [(53, 38, 181, 168)]
    local = np.zeros(raw.shape[:2], bool)
    x1, y1, x2, y2 = tiles[0]
    local[y1:y2, x1:x2] = True
    assert np.array_equal(raw[~local], face.image[~local])
    assert np.any(raw[local] != face.image[local])


def test_model_failure_distinct_from_empty(jpeg):
    engine = make_engine([RuntimeError("inference broke"), None])
    emitted, errors = [], []

    def on_output(frame, _jpeg, rectangles):
        emitted.append((frame.stamp, rectangles))
        stream.stop()

    def on_error(exc, frame):
        errors.append((str(exc), frame.stamp))
        now = time.monotonic()
        stream.inbox.push(InputFrame(jpeg, 2, now, now))

    stream = MosaicStream(engine, engine.cfg, on_output, on_error)
    now = time.monotonic()
    stream.inbox.push(InputFrame(jpeg, 1, now, now))
    worker = threading.Thread(target=stream.run)
    worker.start()
    worker.join(timeout=2)
    stream.stop()
    worker.join(timeout=2)
    assert errors == [("inference broke", 1)]
    assert emitted == [(2, [])]
    assert not stream.pending
    assert stream.snapshot()["processed"] == 1
    assert stream.snapshot()["processing_errors"] == 1


def test_slow_frame_expires_and_latest_input_recovers(jpeg):
    engine = make_engine([None, None])
    first = engine.analyze
    entered = threading.Event()
    finish_first = threading.Event()

    def slow_once(data, stamp, at):
        if stamp == 1:
            entered.set()
            assert finish_first.wait(2)
        return first(data, stamp, at)

    engine.analyze = slow_once
    outputs = []

    def on_output(frame, *_):
        outputs.append(frame.stamp)
        stream.stop()

    stream = MosaicStream(engine, engine.cfg, on_output, lambda exc, _: pytest.fail(str(exc)))
    now = time.monotonic()
    stream.inbox.push(InputFrame(jpeg, 1, now - .45, now))
    thread = threading.Thread(target=stream.run)
    thread.start()
    try:
        assert entered.wait(1)
        for stamp in range(2, 12):
            now = time.monotonic()
            stream.inbox.push(InputFrame(jpeg, stamp, now, now))
        time.sleep(.07)
        finish_first.set()
        thread.join(timeout=2)
        assert outputs == [11]
        assert stream.snapshot()["input_overwritten"] == 9
        assert stream.snapshot()["expired_output"] == 1
    finally:
        finish_first.set()
        stream.stop()
        thread.join(timeout=2)


@pytest.mark.parametrize("enabled", [True, False])
def test_miss_on_detection_frame_and_recovery(jpeg, enabled):
    engine = make_engine([rows(FACE), None, rows(box_at(FACE, 6))],
                         cv=ScriptedFlow(), detect_every_frames=1, miss_compensation=enabled)
    first = analyze(engine, jpeg, 1)
    missed = analyze(engine, jpeg, 2)
    recovered = analyze(engine, jpeg, 3)
    assert bool(missed.masks) is enabled
    if enabled:
        key = next(iter(first.masks))
        assert missed.masks[key].box == box_at(FACE, 3)
        assert missed.masks[key].expanded
        assert recovered.masks[key].box == box_at(FACE, 6)
        assert not recovered.masks[key].expanded
        a = engine.render(first)[1][0]
        b = engine.render(missed)[1][0]
        assert (b[2] - b[0]) > (a[2] - a[0])
    assert engine.detection_calls == 3


def test_real_lk_moves_each_face_independently():
    rng = np.random.default_rng(17)
    width, height = FACE[2] - FACE[0], FACE[3] - FACE[1]
    patches = [rng.integers(0, 256, (height, width, 3), np.uint8) for _ in range(2)]

    def pattern(dx):
        img = np.full((180, 320, 3), 128, np.uint8)
        for position, patch in zip((box_at(FACE, dx), box_at(OTHER, -dx)), patches):
            x1, y1, x2, y2 = position
            img[y1:y2, x1:x2] = patch
        return encode(img)

    engine = make_engine([rows(FACE, OTHER)])
    analyze(engine, pattern(0), 1)
    moved = analyze(engine, pattern(4), 2)
    assert engine.lk_calls == 1
    assert [_iou(mask.box, expected) > .97 for mask, expected in
            zip(moved.masks.values(), (box_at(FACE, 4), box_at(OTHER, -4)))] == [True, True]


@pytest.mark.parametrize("failure", [ScriptedFlow(error=50), ScriptedFlow(inconsistent=True),
                                    ScriptedFlow(dx=42)])
def test_quality_failure_detects_then_only_holds_once(jpeg, failure):
    engine = make_engine([rows(FACE), None, None, rows(FACE)],
                         cv=failure, adaptive_detection=True)
    analyze(engine, jpeg, 1)
    held = analyze(engine, jpeg, 2)
    assert len(held.masks) == 1 and next(iter(held.masks.values())).expanded
    assert engine.detection_calls == 2
    assert not analyze(engine, jpeg, 3).masks
    assert engine.detection_calls == 3
    assert not analyze(engine, jpeg, 4).masks
    assert analyze(engine, jpeg, 5).masks


def test_adaptive_motion_and_idle_cost(jpeg):
    fixed = make_engine([rows(FACE)] * 10, cv=ScriptedFlow(14), detect_every_frames=4)
    motion = make_engine([rows(FACE)] * 10, cv=ScriptedFlow(14), adaptive_detection=True)
    idle = make_engine([rows(FACE)] * 10, cv=ScriptedFlow(0), adaptive_detection=True)
    for n in range(1, 11):
        for engine in (fixed, motion, idle):
            analyze(engine, jpeg, n)
    assert [e.detection_calls for e in (fixed, motion, idle)] == [3, 10, 2]


@pytest.mark.parametrize("interval", range(1, 6))
def test_fixed_interval_and_entrance(jpeg, interval):
    engine = make_engine([None, rows(FACE)] + [None] * 14, detect_every_frames=interval)
    for frame in range(1, interval + 1):
        assert not analyze(engine, jpeg, frame).masks
    assert analyze(engine, jpeg, interval + 1).masks
    assert engine.detection_calls == 2
    # A gap beyond the coast limit invalidates the old track.
    assert not engine.analyze(jpeg, 1000, 10.).masks


def test_coast_expires_through_repeated_misses(jpeg):
    engine = make_engine([rows(FACE)] + [None] * 22, cv=ScriptedFlow(0),
                         detect_every_frames=1)
    for n in range(1, 18):
        assert analyze(engine, jpeg, n).masks
    assert not analyze(engine, jpeg, 18).masks


def test_yunet_threshold_selection_and_bad_output(jpeg):
    for threshold, expected in ((.1, True), (.22, False), (.9, False)):
        engine = make_engine([rows(FACE, score=.2)], confidence_threshold=threshold)
        assert bool(analyze(engine, jpeg, 1).masks) is expected
    engine = make_engine([np.full((1, 15), float("nan"))])
    with pytest.raises(MosaicError, match="invalid_yunet_output"):
        analyze(engine, jpeg, 1)


def test_over_capacity_remains_local(jpeg):
    engine = make_engine([rows(FACE, OTHER)], max_faces=1)
    result = analyze(engine, jpeg, 1)
    assert engine.over_capacity == 1
    assert engine.render(result)[1] == [(53, 38, 181, 168)]


@pytest.mark.parametrize("delay", [200, 300, 500])
def test_delayed_frames_are_corrected_and_published_in_order(jpeg, delay):
    engine = make_engine([rows(FACE), None, rows(box_at(FACE, 16))],
                         cv=ScriptedFlow(), detect_every_frames=2, publish_delay_ms=delay)
    emitted = []
    stream = MosaicStream(engine, engine.cfg, lambda f, _, tiles: emitted.append((f.stamp, tiles)),
                          lambda exc, _: pytest.fail(str(exc)))
    now = time.monotonic()
    for n in range(1, 6):
        stream._accept(InputFrame(jpeg, n, now, now))
    stream._flush()
    assert not emitted
    assert stream.snapshot()["corrected_frames"] == 3
    # Synthetic expected trajectory: 4px/frame. The fake LK deliberately
    # underestimates it by 1px/frame and the end detection supplies the residual.
    first_key = next(iter(stream.pending[0].analysis.masks))
    for index in range(5):
        item = stream.pending[index]
        mask = item.analysis.masks[first_key]
        assert mask.box == box_at(FACE, index * 4)
        assert not mask.expanded
        assert item.frame.jpeg == b""
        item.due = now - .01
    stream._flush()
    assert [stamp for stamp, _ in emitted] == [1, 2, 3, 4, 5]
    assert stream.snapshot()["published_corrected_frames"] == 3


@pytest.mark.parametrize("second", [rows(box_at(FACE, 120)), rows(FACE, box_at(FACE, 6))])
def test_no_interpolation_for_unknown_or_ambiguous_match(jpeg, second):
    engine = make_engine([rows(FACE), second], cv=ScriptedFlow(0),
                         detect_every_frames=2, publish_delay_ms=300)
    analyze(engine, jpeg, 1)
    analyze(engine, jpeg, 2)
    assert not analyze(engine, jpeg, 3).corrections


def test_capacity_expired_and_stop(jpeg):
    engine = make_engine([None] * 5, publish_delay_ms=200, max_buffer_frames=2)
    published = []
    stream = MosaicStream(engine, engine.cfg, lambda *args: published.append(args), None)
    now = time.monotonic()
    stream._accept(InputFrame(jpeg, 1, now - 1., now))
    for n in range(2, 6):
        stream._accept(InputFrame(jpeg, n, now, now))
    assert [item.frame.stamp for item in stream.pending] == [2, 3]
    stream.pending[0].frame = InputFrame(b"", 2, now - 1., now)
    stream._flush()
    assert not published
    stream.stop()
    stream.run()
    info = stream.snapshot()
    assert info["expired_input"] == info["expired_output"] == info["discarded_on_stop"] == 1
    assert info["dropped_capacity"] == 2
    assert info["peak_buffer_frames"] == 2
    assert info["buffer_frames"] == info["buffer_bytes"] == 0


def test_byte_capacity(jpeg, monkeypatch):
    from plugins.yunet_lk_mosaic import stream as module

    monkeypatch.setattr(module, "MAX_BUFFER_BYTES", 180 * 320 * 3 * 2 - 1)
    engine = make_engine([None], publish_delay_ms=500)
    stream = MosaicStream(engine, engine.cfg, None, None)
    now = time.monotonic()
    for n in range(1, 3):
        stream._accept(InputFrame(jpeg, n, now, now))
    assert stream.snapshot()["peak_buffer_frames"] == 1
    assert stream.snapshot()["dropped_capacity"] == 1


def test_local_model_checksum_and_default_downloader(tmp_path, monkeypatch):
    from plugins.yunet_lk_mosaic import model

    wrong = tmp_path / "yunet.onnx"
    wrong.write_bytes(b"wrong")
    with pytest.raises(MosaicError, match="model_verification_failed: size mismatch"):
        resolve_model(str(wrong))
    wrong.write_bytes(b"x" * model.MODEL_METADATA["size"])
    with pytest.raises(MosaicError, match="model_verification_failed: SHA256 mismatch"):
        resolve_model(str(wrong))

    def fake_download(name, directory, sources, files, progress_cb):
        assert name == "yunet_lk_mosaic"
        assert directory.endswith("/yunet_lk_mosaic")
        assert sources == "https://example.org/models"
        assert files[MODEL_FILE]["size"] == 232589
        return {MODEL_FILE: str(wrong)}

    monkeypatch.setenv("YUNET_LK_MOSAIC_MODEL_BASE_URL", "https://example.org/models")
    monkeypatch.setattr(model, "ensure_verified_bundle", fake_download)
    assert resolve_model(None) == str(wrong)


@pytest.fixture
def mosaic_tool(monkeypatch):
    from plugins.yunet_lk_mosaic.plugin import MosaicNode, PREFIX, YuNetLkMosaicPlugin
    from vision_stubs import _FakeExecutor

    monkeypatch.setattr(MosaicNode, "create_timer", lambda *args: None, raising=False)
    monkeypatch.delenv("YUNET_LK_MOSAIC_MODEL", raising=False)
    executor = _FakeExecutor()
    plugin = YuNetLkMosaicPlugin({}, executor)
    try:
        def call(action, **params):
            return plugin.dispatch(PREFIX, {"action": action, **params})

        assert call("config", consent_acknowledged=True)["state"] == "configured"
        yield call
    finally:
        plugin.shutdown()


@pytest.fixture
def model_load_probe(monkeypatch):
    from plugins.yunet_lk_mosaic import runtime

    warming_up = threading.Event()
    ready = threading.Event()
    paths = []

    class DetectorProbe:
        def __init__(self, cv, path, config):
            paths.append(path)

        def detect(self, image):
            warming_up.set()
            assert ready.wait(5)

        def close(self):
            pass

    monkeypatch.setattr(runtime, "YuNetDetector", DetectorProbe)
    try:
        yield warming_up, ready, paths
    finally:
        ready.set()


def test_cold_download_reports_shared_status_in_info(mosaic_tool, model_load_probe, monkeypatch):
    from plugins.yunet_lk_mosaic import model

    began, allow_progress, reported, allow_completion = (threading.Event() for _ in range(4))
    warming_up, ready, paths = model_load_probe

    def download(name, directory, sources, files, progress_cb):
        began.set()
        assert allow_progress.wait(5)
        progress_cb(40, 2, 5)
        reported.set()
        assert allow_completion.wait(5)
        return {MODEL_FILE: "/verified/yunet.onnx"}

    monkeypatch.setattr(model, "ensure_verified_bundle", download)
    try:
        assert mosaic_tool("start", input_topic="/camera/test")["state"] == "loading"
        assert began.wait(5)
        info = mosaic_tool("info")
        assert info["state"] == "loading"
        assert info["message"] == "正在准备模型 'YuNet' …"

        allow_progress.set()
        assert reported.wait(5)
        info = mosaic_tool("info")
        assert info["state"] == "loading"
        assert info["reason"] == "downloading_model"
        assert info["message"] == "正在下载模型 'YuNet' … 40% (2/5 MB)"
        assert "model_fetch" not in info

        allow_completion.set()
        assert warming_up.wait(5)
        info = mosaic_tool("info")
        assert info["state"] == "loading"
        assert info["message"] == "正在加载模型 'YuNet' …"
        assert paths == ["/verified/yunet.onnx"]
    finally:
        allow_progress.set()
        allow_completion.set()
        ready.set()
    assert mosaic_tool("stop")["state"] == "idle"


def test_verified_cache_loads_without_downloading(mosaic_tool, model_load_probe, tmp_path, monkeypatch):
    from plugins.yunet_lk_mosaic import model
    from utils import model_downloader

    contents = b"cached test model"
    cached = tmp_path / MODEL_FILE
    cached.write_bytes(contents)
    monkeypatch.setattr(model, "require_models_subpath", lambda _: str(tmp_path))
    monkeypatch.setattr(model, "MODEL_METADATA", {
        "size": len(contents), "sha256": hashlib.sha256(contents).hexdigest(),
    })

    def unexpected_download(*args, **kwargs):
        pytest.fail("a verified cached model must not be downloaded")

    monkeypatch.setattr(model_downloader, "urlopen", unexpected_download)
    warming_up, ready, paths = model_load_probe
    try:
        assert mosaic_tool("start", input_topic="/camera/test")["state"] == "loading"
        assert warming_up.wait(5)
        info = mosaic_tool("info")
        assert info["state"] == "loading"
        assert info["message"] == "正在加载模型 'YuNet' …"
        assert paths == [str(cached)]

        ready.set()
        deadline = time.monotonic() + 5
        while info["state"] == "loading" and time.monotonic() < deadline:
            time.sleep(.01)
            info = mosaic_tool("info")
        assert info["state"] == "running"
        assert info["message"] is None
    finally:
        ready.set()
    assert mosaic_tool("stop")["state"] == "idle"


@pytest.mark.parametrize("cause", [OSError("network offline"), ValueError("SHA256 mismatch")])
def test_model_download_failure_stays_visible_in_info(mosaic_tool, monkeypatch, cause):
    from plugins.yunet_lk_mosaic import model
    from vision_stubs import _FakeNode

    def failed_download(*args, progress_cb, **kwargs):
        progress_cb(100, .22, .22)
        raise RuntimeError("all model mirrors failed") from cause

    monkeypatch.setattr(model, "ensure_verified_bundle", failed_download)
    mosaic_tool("start", input_topic="/camera/test")
    node = _FakeNode.instances[-1]
    node.thread.join(timeout=5)
    assert not node.thread.is_alive() and node.destroyed
    info = mosaic_tool("info")
    assert info["state"] == "error"
    assert info["reason"].startswith("startup_failed:model_download_failed:")
    assert str(cause) in info["message"]
    assert "正在下载" not in info["message"]


def test_default_bundle_discovers_mosaic_without_starting(monkeypatch):
    import main
    from plugins.yunet_lk_mosaic import runtime
    from plugins.yunet_lk_mosaic.plugin import PREFIX
    from vision_stubs import _FakeExecutor, _FakeNode

    def unexpected_model(*args, **kwargs):
        pytest.fail("registering a tool must not prepare a model")

    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.setattr(runtime, "resolve_model", unexpected_model)
    cfg = main._load_config()
    assert cfg["mcp_port"] == 15720
    assert cfg["plugins"][PREFIX]["enabled"] is True
    assert cfg["plugins"][PREFIX]["consent_acknowledged"] is False
    # Keep a real neighbouring processor in this off-device bundle.
    plugins = {"echo_cancellation": cfg["plugins"]["echo_cancellation"],
               "audio_mixer": {"enabled": False}}
    executor = _FakeExecutor()
    existing = main.PerceptionBundle({"plugins": plugins}, executor)
    bundle = main.PerceptionBundle(
        {"plugins": {**plugins, PREFIX: cfg["plugins"][PREFIX]}}, executor,
    )
    try:
        tools = bundle.get_all_tools()
        assert [t for t in tools if t["name"] != PREFIX] == existing.get_all_tools()
        assert [t["type"] for t in tools if t["name"] == PREFIX] == ["processor"]
        assert bundle.dispatch(PREFIX, {"action": "info"})["state"] == "idle"
        assert bundle.dispatch(PREFIX, {
            "action": "start", "input_topic": "/camera/test",
        })["error"] == "consent_acknowledgment_required"
        assert not executor.nodes and not _FakeNode.instances
    finally:
        bundle.shutdown()
        existing.shutdown()


@pytest.mark.parametrize("fallback", [False, True])
def test_real_model_known_face_and_blank(fallback, monkeypatch):
    path = os.getenv("TEST_YUNET_MODEL")
    if not path:
        pytest.skip("set TEST_YUNET_MODEL to the pinned YuNet model")
    face_path = os.getenv("TEST_YUNET_FACE_IMAGE")
    if face_path:
        image = cv2.imread(face_path)
        assert image is not None
    else:
        data = pytest.importorskip("skimage.data")
        image = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    worker = None
    if fallback:
        pytest.importorskip("onnxruntime")
        from plugins import ort_worker

        worker = ort_worker.OrtWorker()
        monkeypatch.setattr(ort_worker, "_worker", worker)
    cv = ScriptedFlow()
    if fallback:
        cv.FaceDetectorYN = None
    cfg = MosaicConfig().updated({"detect_every_frames": 1})
    engine = MosaicEngine(path, cfg, cv=cv)
    try:
        face = analyze(engine, encode(image), 1)
        assert len(face.masks) == 1
        assert engine.detector.backend == ("onnxruntime-cpu" if fallback else "opencv")
        engine.reset()
        blank = analyze(engine, encode(np.zeros((320, 640, 3), np.uint8)), 2)
        assert not blank.masks
        if worker is not None:
            child = worker._proc
            child.terminate()
            child.join(timeout=2)
            assert not child.is_alive()
            engine.reset()
            assert len(analyze(engine, encode(image), 3).masks) == 1
            assert worker.alive and worker._proc is not child
    finally:
        engine.close()
        if worker is not None:
            worker.close()


def test_intermediate_overlap_vetoes_correction_even_after_faces_separate(jpeg):
    # Both faces approach, overlap heavily in frame two and then separate.
    # Their end positions alone would permit a match.
    left, right = FACE, box_at(FACE, 80)

    class CrossingFlow(ScriptedFlow):
        def __init__(self):
            super().__init__()
            self.sign = 1
            self.first_count = 0

        def calcOpticalFlowPyrLK(self, prev, gray, points, *args, **kwargs):
            moved, status, error = super().calcOpticalFlowPyrLK(prev, gray, points, *args, **kwargs)
            delta = np.where(np.arange(len(points)) < self.first_count, 29, -29) * self.sign
            moved[:, 0, 0] = points[:, 0, 0] + delta
            return moved, status, error

    flow = CrossingFlow()
    engine = make_engine([rows(left, right), rows(left, right)], cv=flow,
                         detect_every_frames=3, publish_delay_ms=300)
    first = analyze(engine, jpeg, 1)
    keys = tuple(first.masks)
    flow.first_count = len(engine.tracks[keys[0]].points)
    analyze(engine, jpeg, 2)
    assert all(not track.association_clear for track in engine.tracks.values())
    flow.sign = -1
    flow.first_count = len(engine.tracks[keys[0]].points)
    recovered = analyze(engine, jpeg, 3)
    assert tuple(recovered.masks) == keys
    assert not recovered.corrections


def test_small_buffer_keeps_due_frame_and_accepts_new_input(jpeg, monkeypatch):
    from plugins.yunet_lk_mosaic import stream as module

    engine = make_engine([None] * 6, publish_delay_ms=500, max_buffer_frames=1)
    emitted = []
    stream = MosaicStream(engine, engine.cfg, lambda f, *_: emitted.append(f.stamp), None)
    now = time.monotonic()
    # Calls are synchronous. Simulate active arrivals and exact deadlines.
    clock = [now]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    for index in range(21):
        clock[0] = now + index * .06
        stream._accept(InputFrame(jpeg, index + 1, clock[0], clock[0]))
    assert emitted == [1, 10]
    assert stream.snapshot()["dropped_capacity"] == 18
    assert stream.snapshot()["buffer_frames"] == 1
