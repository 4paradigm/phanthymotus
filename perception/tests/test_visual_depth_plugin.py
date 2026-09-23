"""
tests/test_visual_depth_plugin.py — depth encoding, region summary, and lifecycle
(host-side, no GPU).

The encoding tests are the load-bearing ones: agent-core's depth renderer
hardcodes a 640x480 canvas and silently drops any frame shorter than that, so
a regression in the resample path shows up as a blank dashboard panel with
nothing in any log.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import threading
import time
import zlib

import numpy as np
import pytest

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _wait_until,
)

import plugins.visual_depth as depth_plugin  # noqa: E402
from plugins.vision_runtime import LetterboxMeta  # noqa: E402

W, H = depth_plugin.DEPTH_WIDTH, depth_plugin.DEPTH_HEIGHT


class _FakeModel:
    """Stands in for VisionEngineSession: infer(frame) -> (outputs, meta)."""

    def __init__(self, depth=None, shape=(H, W)):
        self.calls = 0
        self._depth = depth if depth is not None else np.full(shape, 2.0, dtype=np.float32)

    @property
    def input_size(self):
        return (self._depth.shape[1], self._depth.shape[0])

    def infer(self, frame):
        self.calls += 1
        # Identity letterbox sized to the engine output, so decode_depth's crop
        # is a no-op and the tests assert on the plugin's own behaviour.
        meta = LetterboxMeta(1.0, 0, 0, self._depth.shape[1], self._depth.shape[0])
        return [self._depth], meta


def _plugin(cfg=None, model=None):
    executor = _FakeExecutor()
    plugin = depth_plugin.VideoDepthPerceptionPlugin(cfg or {}, "testns", executor)
    if model is not None:
        plugin._model = model
    return plugin, executor


# ── encoding: the renderer contract ──────────────────────────────────────────

def test_encoded_depth_decompresses_to_exactly_640x480_uint16():
    depth = np.full((H, W), 1.234, dtype=np.float32)
    raw = zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=20.0))
    values = np.frombuffer(raw, dtype="<u2")
    # Shorter than this and DepthZlibRenderer returns early, rendering nothing.
    assert values.size == W * H
    assert values[0] == 1234          # metres → millimetres


def test_wrong_shape_is_refused_rather_than_published():
    """Publishing a mis-sized map produces a blank panel and no log line."""
    with pytest.raises(ValueError, match="640x480"):
        depth_plugin.encode_depth(np.zeros((480, 512), dtype=np.float32), max_depth_m=20.0)


def test_out_of_range_becomes_invalid_not_wrapped():
    """A 70 m reading must not come out as an obstacle at arm's length."""
    depth = np.full((H, W), 70.0, dtype=np.float32)
    values = np.frombuffer(zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=20.0)), dtype="<u2")
    assert set(values.tolist()) == {0}


def test_sub_millimetre_and_nonfinite_become_invalid():
    depth = np.zeros((H, W), dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = np.inf
    depth[0, 2] = 0.0001          # rounds below 1 mm
    depth[0, 3] = 5.0
    values = np.frombuffer(zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=20.0)), dtype="<u2")
    assert values[0] == 0 and values[1] == 0 and values[2] == 0
    assert values[3] == 5000


def test_max_depth_ceiling_is_honoured():
    depth = np.full((H, W), 15.0, dtype=np.float32)
    kept = np.frombuffer(zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=20.0)), dtype="<u2")
    dropped = np.frombuffer(zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=10.0)), dtype="<u2")
    assert kept[0] == 15000
    assert dropped[0] == 0


# ── summary ──────────────────────────────────────────────────────────────────

def test_summary_reports_nearest_per_region():
    depth = np.full((H, W), 9.0, dtype=np.float32)
    depth[:, : W // 3] = 1.0            # something close on the left
    summary = depth_plugin.summarize_depth(depth, "metric")
    assert summary["nearest_by_region"]["left"] == pytest.approx(1.0)
    assert summary["nearest_by_region"]["center"] == pytest.approx(9.0)
    assert summary["nearest_by_region"]["right"] == pytest.approx(9.0)


def test_summary_ignores_a_few_outlier_pixels():
    """The 5th percentile is the point: edge artefacts must not invent an obstacle."""
    depth = np.full((H, W), 5.0, dtype=np.float32)
    depth[0, :10] = 0.01               # a handful of edge artefacts
    summary = depth_plugin.summarize_depth(depth, "metric")
    assert summary["nearest_by_region"]["left"] == pytest.approx(5.0)


def test_summary_marks_uncalibrated_output_as_relative():
    """An agent reading relative numbers as metres is the failure to prevent."""
    summary = depth_plugin.summarize_depth(np.full((H, W), 2.0, dtype=np.float32), "relative")
    assert summary["scale"] == "relative"
    assert summary["unit"] == "relative"


def test_summary_handles_an_all_invalid_map():
    summary = depth_plugin.summarize_depth(np.zeros((H, W), dtype=np.float32), "metric")
    assert summary["nearest_by_region"] == {"left": None, "center": None, "right": None}
    assert summary["range"] is None
    assert summary["valid_fraction"] == 0.0


def test_valid_fraction_reflects_partial_coverage():
    depth = np.zeros((H, W), dtype=np.float32)
    depth[: H // 2] = 3.0
    assert depth_plugin.summarize_depth(depth, "metric")["valid_fraction"] == pytest.approx(0.5)


# ── pipeline ─────────────────────────────────────────────────────────────────

def _feed(node, marker=b"640x480"):
    node._image_cb(_FakeCompressedImage(marker))


def test_both_topics_are_published_for_one_frame():
    plugin, executor = _plugin(model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node)

    depth_pub = next(p for p in node.publishers if p.topic.endswith("/visual_depth"))
    summary_pub = next(p for p in node.publishers if p.topic.endswith("/visual_depth_summary"))
    assert _wait_until(lambda: depth_pub.messages and summary_pub.messages)

    values = np.frombuffer(zlib.decompress(depth_pub.messages[0]), dtype="<u2")
    assert values.size == W * H
    assert json.loads(summary_pub.messages[0])["scale"] == "metric"


def test_model_output_is_resampled_to_the_renderer_size():
    """The model runs at its own resolution; the renderer only accepts 640x480."""
    plugin, executor = _plugin(model=_FakeModel(depth=np.full((768, 768), 3.0, dtype=np.float32)))
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node)

    depth_pub = next(p for p in node.publishers if p.topic.endswith("/visual_depth"))
    assert _wait_until(lambda: bool(depth_pub.messages))
    values = np.frombuffer(zlib.decompress(depth_pub.messages[0]), dtype="<u2")
    assert values.size == W * H
    assert values[0] == 3000


def test_depth_scale_is_applied():
    plugin, executor = _plugin(cfg={"depth_scale": 2.0, "calibrated": True},
                               model=_FakeModel(depth=np.full((H, W), 1.0, dtype=np.float32)))
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node)

    depth_pub = next(p for p in node.publishers if p.topic.endswith("/visual_depth"))
    assert _wait_until(lambda: bool(depth_pub.messages))
    values = np.frombuffer(zlib.decompress(depth_pub.messages[0]), dtype="<u2")
    assert values[0] == 2000          # 1.0 * 2.0 m → 2000 mm


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_info_reports_metres_and_says_whose_calibration():
    """The engine bakes ultralytics' metric fit in, so the output IS metres —
    what info has to convey is that the fit is generic, not this camera's."""
    plugin, _ = _plugin()
    info = plugin.dispatch("visual_depth", {"action": "info"})
    assert info["scale"] == "metric"
    assert info["unit"] == "m"
    assert info["calibration"] == "model-default"
    assert "warning" not in info
    # No prose. `calibration` carries the fact in one token; the explanation
    # lives in the tool description, which is read once rather than re-sent
    # with every answer.
    assert "note" not in info


def test_info_reports_a_site_calibration_once_one_is_set():
    plugin, _ = _plugin(cfg={"cal_a": 0.97, "cal_b": 0.12})
    info = plugin.dispatch("visual_depth", {"action": "info"})
    # The label gained a suffix: a fit typed in by hand, one taken from a preset
    # table and one fitted live against a tape measure are three different
    # levels of evidence, and only the last was measured on *this* camera.
    assert info["calibration"].startswith("site")
    assert info["calibration"] == "site:manual"
    assert "note" not in info


def test_info_drops_the_warning_once_calibrated():
    plugin, _ = _plugin(cfg={"calibrated": True})
    info = plugin.dispatch("visual_depth", {"action": "info"})
    assert info["scale"] == "metric"
    assert "warning" not in info


def test_info_advertises_both_output_formats():
    plugin, executor = _plugin(model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("visual_depth", {"action": "info"})
    formats = {t["format"] for t in info["topic_out"]}
    assert formats == {"image/depth-zlib", "data/json"}


def test_start_then_stop_destroys_the_node():
    plugin, executor = _plugin(model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    plugin.dispatch("visual_depth", {"action": "stop", "instance_id": "/cam/rgb"})
    assert executor.nodes == []
    assert node.destroyed is True


def test_concurrent_starts_create_exactly_one_node():
    plugin, executor = _plugin(model=_FakeModel())
    barrier = threading.Barrier(8)

    def _start():
        barrier.wait()
        plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})

    threads = [threading.Thread(target=_start) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(executor.nodes) == 1


def test_config_updates_global_defaults():
    plugin, _ = _plugin()
    plugin.dispatch("visual_depth", {"action": "config", "fps": 7, "cal_b": 0.25, "max_depth_m": 5.0})
    assert plugin._fps == 7
    assert plugin._cal_b == 0.25
    assert plugin._max_depth_m == 5.0


def test_a_frame_that_fails_to_decode_is_skipped_not_fatal():
    # fps high enough that the rate limiter does not swallow the second frame —
    # at the default 2 fps the good frame lands inside the 500 ms window and is
    # dropped, which would make this pass or fail for the wrong reason.
    plugin, executor = _plugin(cfg={"fps": 1000}, model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node, b"not-a-frame")
    time.sleep(0.01)          # clear the 1 ms rate-limit window between frames
    _feed(node, b"640x480")

    depth_pub = next(p for p in node.publishers if p.topic.endswith("/visual_depth"))
    assert _wait_until(lambda: bool(depth_pub.messages))


# ── measurements and the spoken description ──────────────────────────────────

def _graded_depth(left=1.0, center=2.0, right=3.0):
    """A depth map whose three vertical thirds have known, distinct values."""
    depth = np.zeros((H, W), dtype=np.float32)
    # Same edges summarize_depth uses; W is not divisible by 3, so a fixture
    # that splits on W//3 disagrees with the code by one column and the test
    # then fails on an off-by-one of its own making.
    edges = [round(i * W / 3) for i in range(4)]
    for i, value in enumerate((left, center, right)):
        depth[:, edges[i]:edges[i + 1]] = value
    return depth


def test_measure_depth_adds_averages_and_the_closest_direction():
    stats = depth_plugin.measure_depth(_graded_depth(), "metric")
    assert stats["nearest"] == pytest.approx(1.0)
    assert stats["farthest"] == pytest.approx(3.0)
    assert stats["average"] == pytest.approx(2.0, abs=0.01)
    assert stats["closest_region"] == "left"
    assert stats["farthest_region"] == "right"
    assert stats["average_by_region"]["center"] == pytest.approx(2.0)


def test_measure_depth_ignores_invalid_pixels_in_the_average():
    depth = np.full((H, W), 4.0, dtype=np.float32)
    depth[: H // 2] = 0.0          # 0 means "no reading", not "at the lens"
    stats = depth_plugin.measure_depth(depth, "metric")
    assert stats["average"] == pytest.approx(4.0)
    assert stats["valid_fraction"] == pytest.approx(0.5)


# ── one-shot recognition ─────────────────────────────────────────────────────

def _photo_plugin(tmp_path, depth=None, **cfg):
    base = {"image_roots": [str(tmp_path)], "max_image_bytes": 1 << 20}
    base.update(cfg)
    return _plugin(cfg=base, model=_FakeModel(depth=depth if depth is not None else _graded_depth()))


def _write_frame(tmp_path, name="scene.jpg", marker=b"200x100"):
    path = tmp_path / name
    path.write_bytes(marker)
    return str(path)


def test_recognize_by_photo_answers_without_any_instance(tmp_path):
    """The point of the action: answer about a picture with no camera running."""
    plugin, executor = _photo_plugin(tmp_path)
    result = plugin.dispatch("visual_depth", {
        "action": "recognize_by_photo", "image_path": _write_frame(tmp_path)})

    assert executor.nodes == []          # nothing was started
    assert result["ok"] is True
    assert result["scale"] == "metric"
    assert result["image_size"] == [200, 100]
    assert result["closest_region"] == "left"
    assert "latency_ms" in result
    assert "published_to" not in result  # no instance to echo onto


def test_recognize_by_photo_answers_in_metres(tmp_path):
    plugin, _ = _photo_plugin(tmp_path)
    result = plugin.dispatch("visual_depth", {
        "action": "recognize_by_photo", "image_path": _write_frame(tmp_path)})
    assert result["scale"] == "metric"
    assert result["unit"] == "m"
    assert result["calibration"] == "model-default"
    assert result["nearest"] == pytest.approx(1.0)
    assert result["farthest"] == pytest.approx(3.0)


def test_recognize_by_photo_refuses_a_path_outside_the_roots(tmp_path):
    plugin, _ = _photo_plugin(tmp_path)
    result = plugin.dispatch("visual_depth", {
        "action": "recognize_by_photo", "image_path": "/etc/passwd"})
    assert result["ok"] is False
    assert result["reason"] == "bad_input"
    # Must name this plugin's own action, not some other card's.
    assert "recognize_by_url" in result["detail"]


def test_recognize_by_photo_on_an_undecodable_file(tmp_path):
    plugin, _ = _photo_plugin(tmp_path)
    path = _write_frame(tmp_path, "junk.jpg", b"this is not an image")
    result = plugin.dispatch("visual_depth", {
        "action": "recognize_by_photo", "image_path": path})
    assert result["ok"] is False
    assert "decode" in result["detail"]


def test_recognize_by_url_shares_the_photo_path(tmp_path, monkeypatch):
    import plugins.image_input as image_input
    monkeypatch.setattr(image_input, "fetch_url", lambda url, max_bytes: b"200x100")
    plugin, _ = _photo_plugin(tmp_path)
    result = plugin.dispatch("visual_depth", {
        "action": "recognize_by_url", "url": "https://example.com/scene.jpg"})
    assert result["ok"] is True
    assert result["source"] == "https://example.com/scene.jpg"


def test_one_shot_echoes_onto_a_running_instance(tmp_path):
    """A topic-less card wired into the canvas has to show data flowing."""
    plugin, executor = _photo_plugin(tmp_path)
    plugin.dispatch("visual_depth", {"action": "start"})
    node = executor.nodes[0]

    result = plugin.dispatch("visual_depth", {
        "action": "recognize_by_photo", "image_path": _write_frame(tmp_path)})
    # Asserted on the bus, not on a field in the reply: where the echo went is
    # not something the caller asked about, and the reply is re-read by the
    # model on every turn.
    assert "published_to" not in result

    depth_pub = next(p for p in node.publishers if p.topic == depth_plugin.DEFAULT_DEPTH_TOPIC)
    summary_pub = next(p for p in node.publishers if p.topic == depth_plugin.DEFAULT_SUMMARY_TOPIC)
    # Echoed onto the bus at the renderer's size, whatever the model's was.
    assert np.frombuffer(zlib.decompress(depth_pub.messages[0]), dtype="<u2").size == W * H
    assert json.loads(summary_pub.messages[0])["closest_region"] == "left"


# ── topic-less (on-demand) mode ──────────────────────────────────────────────

def test_start_without_a_topic_subscribes_to_nothing_but_still_publishes():
    plugin, executor = _plugin(model=_FakeModel())
    state = plugin.dispatch("visual_depth", {"action": "start"})
    assert state["mode"] == "on_demand"
    assert state["depth_topic"] == depth_plugin.DEFAULT_DEPTH_TOPIC

    node = executor.nodes[0]
    assert node.subscriptions == []      # no camera, so nothing to subscribe to
    assert {p.topic for p in node.publishers} == {depth_plugin.DEFAULT_DEPTH_TOPIC,
                                                 depth_plugin.DEFAULT_SUMMARY_TOPIC}


def test_info_reports_the_topics_of_a_topic_less_instance():
    """Without this the running on-demand card looks unwired on the canvas."""
    plugin, _ = _plugin(model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start"})
    info = plugin.dispatch("visual_depth", {"action": "info"})
    assert [t["topic"] for t in info["topic_out"]] == [depth_plugin.DEFAULT_DEPTH_TOPIC,
                                                      depth_plugin.DEFAULT_SUMMARY_TOPIC]
    assert info["topic_in"] == []


# ── site calibration ─────────────────────────────────────────────────────────

def test_identity_calibration_returns_the_array_untouched():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    assert depth_plugin.apply_site_calibration(depth, 1.0, 0.0) is depth


def test_site_calibration_is_log_affine_not_a_multiplier():
    """`exp(a·log d + b)` == d**a * e**b — the shape ultralytics fits, so a
    model.calibrate() result transfers here unchanged. The old linear
    depth_scale was this only when a == 1."""
    depth = np.array([[1.0, 4.0]], dtype=np.float32)
    out = depth_plugin.apply_site_calibration(depth, 0.5, np.log(3.0))
    assert out[0, 0] == pytest.approx(3.0)      # 1**0.5 * 3
    assert out[0, 1] == pytest.approx(6.0)      # 4**0.5 * 3


def test_site_calibration_keeps_invalid_pixels_invalid():
    depth = np.array([[0.0, 2.0]], dtype=np.float32)
    out = depth_plugin.apply_site_calibration(depth, 0.5, 0.0)
    assert out[0, 0] == 0.0                     # 0 means "no reading", stays so
    assert np.isfinite(out).all()


def test_legacy_depth_scale_maps_onto_cal_b():
    """A card configured with the old linear knob keeps its behaviour instead
    of silently reverting to identity."""
    cal_a, cal_b = depth_plugin._calibration_from_cfg({"depth_scale": 2.0})
    assert cal_a == 1.0
    assert cal_b == pytest.approx(np.log(2.0))
    depth = np.array([[3.0]], dtype=np.float32)
    assert depth_plugin.apply_site_calibration(depth, cal_a, cal_b)[0, 0] == pytest.approx(6.0)


def test_an_explicit_cal_b_wins_over_legacy_depth_scale():
    _, cal_b = depth_plugin._calibration_from_cfg({"depth_scale": 2.0, "cal_b": 0.0})
    assert cal_b == 0.0


def test_calibration_reaches_the_published_depth_map():
    plugin, executor = _plugin(cfg={"cal_a": 1.0, "cal_b": float(np.log(2.0)), "fps": 1000},
                               model=_FakeModel(depth=np.full((H, W), 1.5, dtype=np.float32)))
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node)
    depth_pub = next(p for p in node.publishers if p.topic.endswith("/visual_depth"))
    assert _wait_until(lambda: bool(depth_pub.messages))
    values = np.frombuffer(zlib.decompress(depth_pub.messages[0]), dtype="<u2")
    assert values[0] == 3000          # 1.5 m * e^ln2 → 3.0 m → 3000 mm


# ── calibrate action ─────────────────────────────────────────────────────────

def test_fit_cal_b_is_the_geometric_mean_ratio():
    """A 2x-over and a 2x-under reading must cancel, which only happens in log
    space — an arithmetic mean of ratios would not."""
    assert depth_plugin.fit_cal_b([4.0, 1.0], [2.0, 2.0]) == pytest.approx(0.0)
    assert depth_plugin.fit_cal_b([4.0], [2.0]) == pytest.approx(np.log(0.5))


def test_fit_cal_b_rejects_unusable_pairs():
    with pytest.raises(ValueError):
        depth_plugin.fit_cal_b([0.0], [2.0])


def test_sample_region_depth_uses_the_median_of_the_centre_box():
    depth = np.full((H, W), 9.0, dtype=np.float32)
    cx, cy = W // 2, H // 2
    depth[cy - 40:cy + 40, cx - 50:cx + 50] = 2.0      # the target
    assert depth_plugin.sample_region_depth(depth, "center") == pytest.approx(2.0)
    # `full` sees mostly background, so it must NOT report the target.
    assert depth_plugin.sample_region_depth(depth, "full") == pytest.approx(9.0)


def test_sample_region_depth_ignores_invalid_pixels():
    depth = np.zeros((H, W), dtype=np.float32)
    cx, cy = W // 2, H // 2
    depth[cy - 40:cy + 40, cx - 50:cx + 50] = 3.0
    assert depth_plugin.sample_region_depth(depth, "center") == pytest.approx(3.0)


def test_sample_region_depth_rejects_an_unknown_region():
    with pytest.raises(ValueError):
        depth_plugin.sample_region_depth(np.ones((H, W), dtype=np.float32), "behind")


def _calibratable(tmp_path, predicted=4.0):
    """A plugin whose engine reports a constant `predicted` metres everywhere."""
    return _photo_plugin(tmp_path, depth=np.full((H, W), predicted, dtype=np.float32))


def test_calibrate_from_a_photo_fits_and_applies_immediately(tmp_path):
    plugin, _ = _calibratable(tmp_path, predicted=4.0)
    result = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": _write_frame(tmp_path)})

    assert result["ok"] is True
    assert result["cal_a"] == 1.0
    assert result["cal_b"] == pytest.approx(np.log(0.5), abs=1e-4)
    assert result["samples"] == 1
    assert result["sample"]["predicted_m"] == pytest.approx(4.0)
    assert result["residuals"][0]["corrected_m"] == pytest.approx(2.0, abs=0.01)
    assert result["calibration"].startswith("site:calibrate")
    # In-memory only — saying so is the difference between a calibration that
    # survives a restart and one that quietly does not.
    assert "cal_a / cal_b" in result["persist"]


def test_calibrate_answers_in_the_new_scale_afterwards(tmp_path):
    plugin, _ = _calibratable(tmp_path, predicted=4.0)
    plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": _write_frame(tmp_path)})
    after = plugin.dispatch("visual_depth", {
        "action": "recognize_by_photo", "image_path": _write_frame(tmp_path)})
    assert after["nearest"] == pytest.approx(2.0, abs=0.01)


def test_repeated_calibration_refits_rather_than_compounding(tmp_path):
    """Fitting against already-corrected depth would converge on the first
    guess; the node keeps the raw map precisely so it does not."""
    plugin, _ = _calibratable(tmp_path, predicted=4.0)
    photo = _write_frame(tmp_path)
    first = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": photo})
    second = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": photo})
    assert second["samples"] == 2
    assert second["cal_b"] == pytest.approx(first["cal_b"], abs=1e-6)


def test_calibrate_reset_returns_to_the_engine_default(tmp_path):
    plugin, _ = _calibratable(tmp_path, predicted=4.0)
    plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": _write_frame(tmp_path)})
    result = plugin.dispatch("visual_depth", {"action": "calibrate", "reset": True})
    assert result["samples"] == 0
    assert (result["cal_a"], result["cal_b"]) == (1.0, 0.0)
    assert result["calibration"] == "model-default"


def test_calibrate_requires_a_distance(tmp_path):
    plugin, _ = _calibratable(tmp_path)
    result = plugin.dispatch("visual_depth", {
        "action": "calibrate", "image_path": _write_frame(tmp_path)})
    assert result["ok"] is False
    assert "distance_m is required" in result["detail"]


def test_calibrate_rejects_a_nonsense_distance(tmp_path):
    plugin, _ = _calibratable(tmp_path)
    result = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": -1, "image_path": _write_frame(tmp_path)})
    assert result["ok"] is False


def test_calibrate_without_a_frame_or_a_photo_says_so():
    plugin, _ = _plugin(model=_FakeModel())
    result = plugin.dispatch("visual_depth", {"action": "calibrate", "distance_m": 2.0})
    assert result["ok"] is False
    assert "start this card on a camera" in result["detail"]


def test_calibrate_uses_the_live_frame_of_a_running_instance():
    plugin, executor = _plugin(cfg={"fps": 1000},
                               model=_FakeModel(depth=np.full((H, W), 4.0, dtype=np.float32)))
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node)
    depth_pub = next(p for p in node.publishers if p.topic.endswith("/visual_depth"))
    assert _wait_until(lambda: bool(depth_pub.messages))

    result = plugin.dispatch("visual_depth", {"action": "calibrate", "distance_m": 2.0})
    assert result["ok"] is True
    assert result["cal_b"] == pytest.approx(np.log(0.5), abs=1e-4)
    # Applied to the running node too, without a restart.
    assert node._cal_b == pytest.approx(np.log(0.5), abs=1e-4)


def test_calibration_advice_escalates_with_the_evidence(tmp_path):
    plugin, _ = _calibratable(tmp_path, predicted=4.0)
    photo = _write_frame(tmp_path)
    one = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": photo})
    assert "只固定了整体比例" in one["message"]

    two = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.1, "image_path": photo})
    assert "距离都差不多" in two["message"]     # no spread — not yet informative

    three = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 8.0, "image_path": photo})
    assert "error_pct" in three["message"]


# ── flat-plane procedure ─────────────────────────────────────────────────────

def _plane(distance=2.0):
    """What the camera sees facing a wall square-on: one distance everywhere."""
    return np.full((H, W), distance, dtype=np.float32)


def _corridor():
    """Not a plane: depth ramps across the frame."""
    return np.tile(np.linspace(1.0, 8.0, W, dtype=np.float32), (H, 1))


def test_flatness_is_near_zero_for_a_plane_and_large_for_a_corridor():
    assert depth_plugin.sample_region(_plane(), "full")["flatness"] == pytest.approx(0.0)
    assert depth_plugin.sample_region(_corridor(), "full")["flatness"] > 0.5


def test_flatness_is_scale_free():
    """A wall at 8 m is as flat as a wall at 1 m — the check must not drift
    with distance or it would only ever fire at range."""
    near = depth_plugin.sample_region(_plane(1.0), "full")["flatness"]
    far = depth_plugin.sample_region(_plane(8.0), "full")["flatness"]
    assert near == pytest.approx(far)


def test_calibrating_against_a_plane_raises_no_warning(tmp_path):
    plugin, _ = _photo_plugin(tmp_path, depth=_plane(4.0))
    result = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": _write_frame(tmp_path)})
    assert result["flatness"] == pytest.approx(0.0)
    assert "warnings" not in result


def test_calibrating_against_something_that_is_not_a_plane_warns(tmp_path):
    """One distance cannot stand for the region unless the region is at one
    distance — which is the entire reason the operator is asked for a wall."""
    plugin, _ = _photo_plugin(tmp_path, depth=_corridor())
    result = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "region": "full",
        "image_path": _write_frame(tmp_path)})
    assert result["ok"] is True          # still fits; the operator decides
    assert any("不是一个平面" in w for w in result["warnings"])
    assert any("reset_calibration" in w for w in result["warnings"])


def test_a_disagreeing_sample_is_named_not_silently_averaged(tmp_path):
    """A typo (2 for 20) would otherwise just drag the mean."""
    plugin, _ = _photo_plugin(tmp_path, depth=_plane(4.0))
    photo = _write_frame(tmp_path)
    plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": photo})
    result = plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 20.0, "image_path": photo})
    assert result["samples"] == 2
    assert any("对不上" in w for w in result["warnings"])


def test_three_distances_over_a_real_spread_fit_cleanly(tmp_path):
    """The 1 m / 2 m / 3 m procedure on a camera whose error IS a constant
    factor: every residual should come out small."""
    plugin, _ = _photo_plugin(tmp_path)
    photo = _write_frame(tmp_path)
    result = None
    for truth, predicted in ((1.0, 2.0), (2.0, 4.0), (3.0, 6.0)):
        plugin._model = _FakeModel(depth=_plane(predicted))
        result = plugin.dispatch("visual_depth", {
            "action": "calibrate", "distance_m": truth, "image_path": photo})
    assert result["samples"] == 3
    assert result["cal_b"] == pytest.approx(np.log(0.5), abs=1e-4)
    assert result["max_error_pct"] < 1.0
    assert "warnings" not in result
    assert "error_pct" in result["message"]


def test_reset_calibration_is_its_own_action(tmp_path):
    plugin, _ = _photo_plugin(tmp_path, depth=_plane(4.0))
    plugin.dispatch("visual_depth", {
        "action": "calibrate", "distance_m": 2.0, "image_path": _write_frame(tmp_path)})
    result = plugin.dispatch("visual_depth", {"action": "reset_calibration"})
    assert result["samples"] == 0
    assert (result["cal_a"], result["cal_b"]) == (1.0, 0.0)
    assert result["calibration"] == "model-default"


def test_the_procedure_is_in_every_calibration_reply(tmp_path):
    """Whoever is holding the tape measure should not have to find the docs."""
    plugin, _ = _photo_plugin(tmp_path, depth=_plane(4.0))
    result = plugin.dispatch("visual_depth", {"action": "reset_calibration"})
    assert "平整的墙" in result["procedure"]
    missing = plugin.dispatch("visual_depth", {"action": "calibrate"})
    assert missing["ok"] is False
    assert "平整的墙" in missing["detail"]


def test_a_one_shot_answer_carries_no_boilerplate(tmp_path):
    """Every field here is re-read by the model on every turn, so the reply
    holds answers only — no standing prose, no plumbing detail."""
    plugin, _ = _photo_plugin(tmp_path)
    result = plugin.dispatch("visual_depth", {
        "action": "recognize_by_photo", "image_path": _write_frame(tmp_path)})
    assert "note" not in result           # said the same paragraph every call
    assert "published_to" not in result   # which topic it echoed to is plumbing
    assert "warning" not in result
    # No prose summary either: it only restated the numbers below it, which a
    # model reading them can phrase itself.
    assert "description" not in result
    # What does survive: the measurements, and one token of provenance.
    assert result["calibration"] == "model-default"
    assert result["nearest"] and result["farthest"] and result["average"]




# ── one sample is one frame, and that was the whole problem ──────────────────

def test_a_sample_is_the_median_of_many_frames_not_one():
    """Measured on r1_sz: robot stationary, same wall, fourteen consecutive
    frames spanned 1.68-2.09 m — 22% peak to peak. Two `calibrate` calls at the
    same 1.6 m gave 3.82 and 5.00, 31% apart. Four such single-frame samples
    were then fitted with two parameters, and the residuals argued convincingly
    for a log-slope that was entirely noise."""
    frames = [np.full((48, 64), value, dtype=np.float32)
              for value in (1.68, 1.81, 1.83, 1.92, 2.09)]
    out = depth_plugin.sample_region_over_frames(frames, "center")
    assert out["distance_m"] == pytest.approx(1.83, abs=0.01), "median of the per-frame medians"
    assert out["frames"] == 5


def test_one_bad_frame_does_not_move_the_answer():
    """The median of the per-frame medians, not the median of everything
    pooled: pooling lets one frame move the answer by its share of the pixels."""
    frames = [np.full((48, 64), 2.0, dtype=np.float32) for _ in range(6)]
    frames.append(np.full((48, 64), 40.0, dtype=np.float32))
    assert depth_plugin.sample_region_over_frames(frames, "center")["distance_m"] == pytest.approx(2.0)


def test_the_scatter_is_reported_because_it_is_the_error_bar():
    """Without it the next person reads structure into the residuals and adds a
    parameter to explain it — which is exactly what happened."""
    steady = [np.full((48, 64), 2.0, dtype=np.float32) for _ in range(5)]
    assert depth_plugin.sample_region_over_frames(steady, "center")["scatter"] == pytest.approx(0.0)

    wobbly = [np.full((48, 64), v, dtype=np.float32) for v in (1.7, 1.8, 2.1)]
    out = depth_plugin.sample_region_over_frames(wobbly, "center")
    assert out["scatter"] == pytest.approx((2.1 - 1.7) / 1.8, abs=0.01)
    assert out["scatter"] > depth_plugin._SCATTER_LIMIT, "22% on r1_sz has to trip the warning"


def test_a_frame_with_nothing_valid_is_skipped_rather_than_fatal():
    frames = [np.full((48, 64), np.nan, dtype=np.float32),
              np.full((48, 64), 2.0, dtype=np.float32)]
    out = depth_plugin.sample_region_over_frames(frames, "center")
    assert out["frames"] == 1 and out["distance_m"] == pytest.approx(2.0)


def test_no_usable_frame_is_an_error_not_a_zero():
    """Zero metres is the most alarming thing a depth consumer can be told."""
    with pytest.raises(ValueError):
        depth_plugin.sample_region_over_frames(
            [np.full((48, 64), np.nan, dtype=np.float32)], "center")


# ── presets: a number that looks like a fact needs a provenance ──────────────

def test_a_preset_supplies_a_fit_when_nothing_was_typed():
    cfg = {"calibration_preset": "Unitree R1"}
    a, b = depth_plugin._calibration_from_cfg(cfg)
    assert (a, b) == (1.0, depth_plugin.CALIBRATION_PRESETS["Unitree R1"]["cal_b"])


def test_the_selector_decides_when_there_is_one():
    """The dropdown says what is in effect instead of leaving it to be
    inferred. A preset selected while numbers sit in the fields means the
    preset."""
    cfg = {"calibration_preset": "Unitree R1", "cal_a": 1.0, "cal_b": -0.5}
    assert depth_plugin._calibration_from_cfg(cfg)[1] != -0.5
    cfg["calibration_preset"] = depth_plugin.CAL_MANUAL
    assert depth_plugin._calibration_from_cfg(cfg) == (1.0, -0.5)


def test_a_file_config_without_a_selector_still_applies_its_numbers():
    """`perception/config.yaml` has no dropdown, and writing `cal_b: -0.9`
    there has always meant "apply this". Requiring the selector would silently
    stop that working — which is how a file config becomes decoration."""
    assert depth_plugin._calibration_from_cfg({"cal_a": 1.0, "cal_b": -0.9}) == (1.0, -0.9)
    assert depth_plugin.calibration_origin({"cal_b": -0.9}) == "manual"


def test_numbers_the_selector_is_ignoring_are_said_out_loud():
    """Dropping them quietly is the failure this card keeps running into."""
    assert depth_plugin.calibration_conflict(
        {"calibration_preset": depth_plugin.CAL_NONE, "cal_b": -0.9})
    assert not depth_plugin.calibration_conflict(
        {"calibration_preset": depth_plugin.CAL_MANUAL, "cal_b": -0.9})
    assert not depth_plugin.calibration_conflict({"cal_b": -0.9})


def test_an_untouched_pair_does_not_shadow_the_preset():
    cfg = {"calibration_preset": "Unitree R1", "cal_a": 1.0, "cal_b": 0.0}
    assert depth_plugin._calibration_from_cfg(cfg)[1] != 0.0


def test_the_dropdown_has_no_blank_row():
    """An option you cannot see is an option you cannot choose deliberately —
    the same shape as every other silent default this card has been bitten by.
    "" is still accepted from an existing config, but it is not offered."""
    enum = depth_plugin.TOOLS[0]["configSchema"]["properties"]["calibration_preset"]["enum"]
    assert "" not in enum
    assert enum[0] == depth_plugin.CAL_NONE
    assert depth_plugin.CAL_MANUAL in enum
    assert depth_plugin._calibration_from_cfg({"calibration_preset": ""}) == (1.0, 0.0)


def test_an_unknown_preset_falls_back_rather_than_guessing():
    assert depth_plugin._calibration_from_cfg({"calibration_preset": "nope"}) == (1.0, 0.0)


def test_every_preset_records_where_it_came_from():
    """A preset chosen for the wrong camera fails exactly the way the engine's
    own default failed on r1_sz — silently, by a factor of three. Whoever picks
    one has to be able to see what it was measured on."""
    for name, preset in depth_plugin.CALIBRATION_PRESETS.items():
        for field in ("camera", "measured_on", "samples", "range_m", "note"):
            assert preset.get(field), f"{name} 缺少出处字段 {field}"


def test_the_origin_is_reported_and_distinguishes_the_four_cases():
    assert depth_plugin.calibration_origin({}) == "model-default"
    assert depth_plugin.calibration_origin(
        {"calibration_preset": "Unitree R1"}) == "preset:Unitree R1"
    assert depth_plugin.calibration_origin(
        {"calibration_preset": depth_plugin.CAL_MANUAL, "cal_b": -0.9}) == "manual"
    assert depth_plugin.calibration_origin({"depth_scale": 0.4}) == "legacy-depth_scale"


def test_a_preset_names_the_camera_even_though_it_is_labelled_by_robot():
    """Named for the robot because that is what the person choosing one knows,
    but it is a property of the lens — and a variant shipping a different
    camera must not quietly inherit this entry. The `camera` field is where
    that check can be made; r1_sz's lens is ~102 degrees across, which is why
    the engine's general-purpose fit was out by 3.2x to begin with."""
    for name, preset in depth_plugin.CALIBRATION_PRESETS.items():
        assert len(str(preset["camera"])) > 8, (
            f"{name} 的 camera 字段太短，说不清它到底标的是哪个镜头")


def test_the_default_is_no_preset_and_no_correction():
    """Nothing preselected: an unconfigured card must behave exactly as it did
    before presets existed, because a preset applied to the wrong camera is the
    failure this whole mechanism exists to make visible."""
    schema = depth_plugin.TOOLS[0]["configSchema"]["properties"]
    assert schema["calibration_preset"]["default"] == depth_plugin.CAL_NONE
    assert schema["cal_a"]["default"] == 1.0 and schema["cal_b"]["default"] == 0.0
    assert depth_plugin._calibration_from_cfg({}) == (1.0, 0.0)
    assert depth_plugin.calibration_origin({}) == "model-default"


def test_every_schema_default_matches_the_file_default():
    """**agent-core sends every schema field on every config call, defaults
    included**, so a key here silently overrides the same key in
    perception/config.yaml. The navi card was bitten by exactly this: the file
    said 0.6, the card reported 0.4, and the only trace was a note that read
    like the file had never been edited.

    A field may live in the schema or in the file; if it lives in both, the two
    defaults have to agree or the file is decoration."""
    import os
    import yaml

    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "config.yaml")) as handle:
        cfg = yaml.safe_load(handle)["plugins"].get("visual_depth") or {}

    for key, spec in depth_plugin.TOOLS[0]["configSchema"]["properties"].items():
        if key in cfg and "default" in spec:
            assert cfg[key] == spec["default"], (
                f"{key}: config.yaml 是 {cfg[key]}，schema 默认是 "
                f"{spec['default']} —— 画布会用后者覆盖前者")
