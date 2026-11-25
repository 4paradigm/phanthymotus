"""Graphics-thread ownership, immutable physics snapshots and bounded work."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import sys
import threading
import time

import pytest

np = pytest.importorskip("numpy")
mujoco = pytest.importorskip("mujoco")
Image = pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_service.contracts import model_fingerprint
from motion_service.rendering import FrameRenderer


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <asset><mesh name="shell" vertex="0 0 0 .2 0 0 0 .2 0 0 0 .2"/></asset>
      <worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1"/>
        <body pos="0 0 .8"><freejoint/>
          <geom type="mesh" mesh="shell" group="2" contype="0" conaffinity="0" rgba=".8 .2 .1 1"/>
          <geom type="sphere" size=".1" mass="1" group="3"/>
        </body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.time = 1.25
    return model, data


def snapshot(data):
    return {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "time": data.time}


@pytest.fixture
def graphics(monkeypatch):
    class Graphics:
        calls = []
        instances = []
        gate = None
        entered = threading.Event()

        def __init__(self, model, *, width, height):
            self.model = model
            self.width, self.height = width, height
            self.instances.append(self)
            self.note("create")

        def note(self, operation):
            self.calls.append((operation, threading.get_ident()))

        def update_scene(self, data, *, camera, scene_option):
            self.note("update")
            self.state = snapshot(data)
            self.options = scene_option.geomgroup.copy()

        def render(self):
            self.note("render")
            self.entered.set()
            if self.gate is not None:
                assert self.gate.wait(5), "test did not release graphics thread"
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            image[:, :, 0] = np.arange(self.width, dtype=np.uint8)
            return image

        def close(self):
            self.note("close")

    monkeypatch.setattr(mujoco, "Renderer", Graphics)
    return Graphics


def test_private_model_and_data_with_one_graphics_thread(scene, graphics):
    model, live = scene
    fingerprint = model_fingerprint(model)
    before = snapshot(live)
    render = FrameRenderer(model)
    try:
        state = snapshot(live)
        state["qpos"][0] = .25
        result = render.render(state, width=320, height=240)
        assert result["mime_type"] == "image/jpeg"
        assert Image.open(io.BytesIO(result["data"])).size == (320, 240)
        instance = graphics.instances[0]
        assert instance.model is not model
        assert instance.state["qpos"][0] == .25
        assert instance.state["time"] == before["time"]
        assert instance.options[0] == instance.options[2] == 1
        assert instance.options[3] == 0
        assert model_fingerprint(model) == fingerprint
        np.testing.assert_array_equal(live.qpos, before["qpos"])
        np.testing.assert_array_equal(live.qvel, before["qvel"])
        assert live.time == before["time"]
    finally:
        render.close()
    render.close()
    assert [name for name, _ in graphics.calls] == ["create", "update", "render", "close"]
    assert len({tid for _, tid in graphics.calls}) == 1
    assert graphics.calls[0][1] != threading.get_ident()
    with pytest.raises(RuntimeError, match="closed"):
        render.render(snapshot(live))


def test_resize_releases_previous_context_on_owner_thread(scene, graphics):
    model, data = scene
    render = FrameRenderer(model)
    try:
        for width, height in [(320, 240), (640, 360), (640, 360)]:
            result = render.render(snapshot(data), width=width, height=height)
            assert Image.open(io.BytesIO(result["data"])).size == (width, height)
    finally:
        render.close()
    assert len(graphics.instances) == 2
    assert [x[0] for x in graphics.calls].count("close") == 2
    assert len({x[1] for x in graphics.calls}) == 1


def test_same_aspect_sizes_and_quality_reuse_one_context(scene, graphics):
    model, data = scene
    render = FrameRenderer(model)
    try:
        for width, height, quality in [(640, 360, 70), (960, 540, 88),
                                       (480, 270, 40), (960, 540, 95)]:
            frame = render.render(snapshot(data), width=width, height=height, jpeg_quality=quality)
            assert (frame["width"], frame["height"]) == (width, height)
            assert Image.open(io.BytesIO(frame["data"])).size == (width, height)
        assert len(graphics.instances) == 1
        assert (graphics.instances[0].width, graphics.instances[0].height) == (960, 540)
        high = render.render(snapshot(data), width=640, height=360, jpeg_quality=95)
        low = render.render(snapshot(data), width=640, height=360, jpeg_quality=40)
        assert len(low["data"]) < len(high["data"])
        assert len(graphics.instances) == 1
    finally:
        render.close()
    assert [x[0] for x in graphics.calls].count("close") == 1


def test_failed_context_resize_can_return_to_the_previous_aspect(scene, graphics, monkeypatch):
    attempts = 0

    def create(model, *, width, height):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("one transient context allocation failure")
        return graphics(model, width=width, height=height)

    monkeypatch.setattr(mujoco, "Renderer", create)
    model, data = scene
    render = FrameRenderer(model)
    try:
        render.render(snapshot(data), width=320, height=240)
        with pytest.raises(RuntimeError, match="transient context"):
            render.render(snapshot(data), width=640, height=360)
        recovered = render.render(snapshot(data), width=320, height=240)
        assert Image.open(io.BytesIO(recovered["data"])).size == (320, 240)
        assert attempts == 3
    finally:
        render.close()


@pytest.mark.parametrize("size", [(160, 120), (320, 240), (640, 360), (960, 540), (959, 539)])
def test_native_viewport_preserves_exact_aspect_and_bounds(scene, graphics, size):
    model, _ = scene
    render = FrameRenderer(model)
    try:
        width, height = size
        native_width, native_height = render._native_size(width, height)
        assert width <= native_width <= 960
        assert height <= native_height <= 640
        assert native_width * height == native_height * width
    finally:
        render.close()


@pytest.mark.parametrize("arguments", [
    {"width": 961}, {"width": True}, {"height": 63}, {"timeout": 0},
    {"camera": {"distance": 0}}, {"camera": {"azimuth": float("nan")}},
    {"camera": {"lookat": [0, 0]}}, {"camera": {"elevation": 90}},
    {"camera": {"unknown": 1}},
    {"jpeg_quality": 39}, {"jpeg_quality": 96}, {"jpeg_quality": True}, {"jpeg_quality": 70.5},
])
def test_invalid_view_is_rejected_without_creating_gl(scene, graphics, arguments):
    model, data = scene
    render = FrameRenderer(model)
    try:
        with pytest.raises(ValueError):
            render.render(snapshot(data), **arguments)
        assert graphics.instances == []
    finally:
        render.close()


@pytest.mark.parametrize("field,value", [("qpos", [0]), ("qvel", [float("nan")] * 6), ("time", float("inf"))])
def test_invalid_snapshot_is_rejected(scene, graphics, field, value):
    model, data = scene
    render = FrameRenderer(model)
    try:
        state = snapshot(data)
        state[field] = value
        with pytest.raises(ValueError):
            render.render(state)
        assert graphics.instances == []
    finally:
        render.close()


def test_bounded_queue_drops_timed_out_frames_and_recovers(scene, graphics):
    model, data = scene
    graphics.gate = threading.Event()
    render = FrameRenderer(model, queue_size=1)
    with ThreadPoolExecutor(max_workers=1) as callers:
        active = callers.submit(render.render, snapshot(data))
        try:
            assert graphics.entered.wait(2)
            with pytest.raises(TimeoutError, match="timed out"):
                render.render(snapshot(data), timeout=.02)
            with pytest.raises(RuntimeError, match="busy"):
                render.render(snapshot(data))
            graphics.gate.set()
            assert active.result(timeout=2)["mime_type"] == "image/jpeg"
            # The cancelled queued frame is discarded by the render worker.
            limit = time.monotonic() + 2
            while not render._queue.empty() and time.monotonic() < limit:
                time.sleep(.005)
            assert render.render(snapshot(data))["mime_type"] == "image/jpeg"
            assert [x[0] for x in graphics.calls].count("render") == 2
        finally:
            graphics.gate.set()
            render.close()


def test_close_rejects_queued_frames_and_finishes_in_flight_frame(scene, graphics):
    model, data = scene
    graphics.gate = threading.Event()
    render = FrameRenderer(model)
    with ThreadPoolExecutor(max_workers=3) as callers:
        active = callers.submit(render.render, snapshot(data))
        assert graphics.entered.wait(2)
        queued = callers.submit(render.render, snapshot(data))
        limit = time.monotonic() + 2
        while render._queue.empty() and time.monotonic() < limit:
            time.sleep(.005)
        closing = callers.submit(render.close)
        try:
            with pytest.raises(RuntimeError, match="closed"):
                queued.result(timeout=2)
        finally:
            graphics.gate.set()
        assert active.result(timeout=2)["mime_type"] == "image/jpeg"
        closing.result(timeout=2)
    assert not render._thread.is_alive()
    assert [x[0] for x in graphics.calls].count("render") == 1
    assert [x[0] for x in graphics.calls].count("close") == 1


def test_close_can_discard_an_already_cancelled_frame(scene, graphics):
    model, data = scene
    graphics.gate = threading.Event()
    render = FrameRenderer(model)
    with ThreadPoolExecutor(max_workers=2) as callers:
        active = callers.submit(render.render, snapshot(data))
        assert graphics.entered.wait(2)
        with pytest.raises(TimeoutError):
            render.render(snapshot(data), timeout=.02)
        closing = callers.submit(render.close)
        graphics.gate.set()
        active.result(timeout=2)
        closing.result(timeout=2)
    assert not render._thread.is_alive()


def test_gl_error_is_explicit_and_never_returns_a_synthetic_image(scene, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("EGL unavailable")

    monkeypatch.setattr(mujoco, "Renderer", unavailable)
    model, data = scene
    render = FrameRenderer(model)
    try:
        with pytest.raises(RuntimeError, match="EGL unavailable"):
            render.render(snapshot(data))
    finally:
        render.close()


@pytest.mark.skipif(os.environ.get("MOTION_RENDER_TEST_GL") != "1", reason="requires an explicitly selected real GL backend")
def test_real_offscreen_jpeg_uses_mesh_state_without_stepping(scene):
    model, data = scene
    before = snapshot(data)
    render = FrameRenderer(model)
    try:
        frame = render.render(before, width=320, height=240)
        image = np.asarray(Image.open(io.BytesIO(frame["data"])))
        assert image.shape == (240, 320, 3)
        assert image.std() > 5
        moved = snapshot(data)
        moved["qpos"][0] += .5
        camera = {"lookat": [0, 0, .8], "distance": 2.5}
        a = render.render(before, camera=camera, width=320, height=240)
        b = render.render(moved, camera=camera, width=320, height=240)
        assert a["data"] != b["data"]
        np.testing.assert_array_equal(data.qpos, before["qpos"])
        assert data.time == before["time"]
    finally:
        render.close()
