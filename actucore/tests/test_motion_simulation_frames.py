"""Frame API consistency and isolation; real graphics have separate tests."""
from __future__ import annotations

import base64
import copy
import json
import sys
import threading
from types import ModuleType

import pytest

pytest.importorskip("mujoco")
from test_motion_simulation import profile, create, call
from motion_service.simulation import SimulationManager


@pytest.fixture
def rendering(monkeypatch):
    class Frames:
        instances = []

        def __init__(self, model):
            self.nq = model.nq
            self.calls = []
            self.qualities = []
            self.closed = False
            self.failure = None
            self.entered = None
            self.release = None
            self.instances.append(self)

        def render(self, frame, camera, width, height, timeout, jpeg_quality=88):
            self.calls.append(copy.deepcopy((frame, camera, width, height)))
            self.qualities.append(jpeg_quality)
            if self.entered is not None:
                self.entered.set()
                assert self.release.wait(3)
            if self.failure is not None:
                raise self.failure
            return {"mime_type": "image/jpeg", "width": width, "height": height,
                    "data": b"frame-transport-fixture"}

        def close(self):
            self.closed = True

    module = ModuleType("motion_service.rendering")
    module.FrameRenderer = Frames
    monkeypatch.setitem(sys.modules, "motion_service.rendering", module)
    return Frames


@pytest.fixture
def session(profile, tmp_path, rendering):
    manager = SimulationManager({"test": profile}, tmp_path)
    sid = create(manager, control_mode="pd_hold")["session"]["id"]
    yield manager, sid
    manager.close()


def test_live_frame_and_metadata_share_one_snapshot_without_stepping(session, rendering):
    manager, sid = session
    before = call(manager, sid, "step", steps=2)
    response = call(manager, sid, "render", width=640, height=360,
                    azimuth=40, elevation=-15, distance=2, lookat=[0., 0., .8])
    after = call(manager, sid, "inspect")
    assert before == after
    assert response["snapshot"] == before["snapshot"]
    assert response["session"] == before["session"]
    assert base64.b64decode(response["image"]["data"]) == b"frame-transport-fixture"
    frame, camera, width, height = rendering.instances[0].calls[-1]
    assert frame["qpos"] == before["snapshot"]["qpos"]
    assert frame["time"] == before["snapshot"]["time"]
    assert camera == {"azimuth": 40., "elevation": -15., "distance": 2., "lookat": [0., 0., .8]}
    assert (width, height) == (640, 360)
    assert rendering.instances[0].qualities == [88]


def test_compact_frame_uses_light_metadata_without_changing_live_state(session, rendering, monkeypatch):
    manager, sid = session
    before = call(manager, sid, "inspect")
    native = manager.sessions[sid]
    with monkeypatch.context() as patch:
        def no_full_observation():
            raise AssertionError("compact metadata must not allocate full body/geom state")
        patch.setattr(native, "_observation", no_full_observation)
        response = call(manager, sid, "render", width=640, height=360, jpeg_quality=70, detail="compact")
    compact = response["snapshot"]
    full = before["snapshot"]
    for key in ["time", "step", "dt", "root_pose", "joint_names", "joint_pos"]:
        assert compact[key] == full[key]
    assert compact["contact_count"] == native.data.ncon
    assert compact["bodies"] == [{"id": b["id"], "name": b["name"]} for b in full["bodies"]]
    assert not {"geoms", "contacts", "qpos", "joint_vel"} & compact.keys()
    assert len(json.dumps(compact)) < len(json.dumps(full)) / 2
    assert call(manager, sid, "inspect") == before
    assert rendering.instances[0].qualities == [70]
    assert rendering.instances[0].calls[0][0]["qpos"] == full["qpos"]


def test_explicit_full_and_default_frame_response_are_compatible(session):
    manager, sid = session
    default = call(manager, sid, "render")
    explicit = call(manager, sid, "render", detail="full", jpeg_quality=88)
    assert default == explicit
    assert "geoms" in default["snapshot"] and "contacts" in default["snapshot"]


def test_compact_inspect_is_read_only_and_mutations_keep_full_metadata(session):
    manager, sid = session
    before = call(manager, sid, "inspect")
    compact = call(manager, sid, "inspect", detail="compact")
    assert compact["snapshot"]["time"] == before["snapshot"]["time"]
    assert compact["snapshot"]["root_pose"] == before["snapshot"]["root_pose"]
    assert compact["snapshot"]["joint_pos"] == before["snapshot"]["joint_pos"]
    assert "geoms" not in compact["snapshot"]
    assert call(manager, sid, "inspect") == before
    assert call(manager, sid, "inspect", detail="full") == before
    stepped = call(manager, sid, "step", detail="compact")
    assert "geoms" in stepped["snapshot"] and "contacts" in stepped["snapshot"]
    call(manager, sid, "close")
    closed = call(manager, sid, "inspect", detail="compact")
    assert closed["session"]["state"] == "closed"
    assert closed["snapshot"]["time"] == stepped["snapshot"]["time"]
    assert "geoms" not in closed["snapshot"]


@pytest.mark.parametrize("detail", ["other", None, {}, True])
def test_invalid_inspect_detail_is_rejected(session, detail):
    manager, sid = session
    with pytest.raises(ValueError, match="detail"):
        call(manager, sid, "inspect", detail=detail)


def test_recorded_frame_render_does_not_replace_live_state(session, rendering):
    manager, sid = session
    call(manager, sid, "record")
    recorded = call(manager, sid, "step")
    live = call(manager, sid, "reset")
    response = call(manager, sid, "render", offset=1)
    assert response["snapshot"]["time"] == pytest.approx(.02)
    assert response["snapshot"]["qpos"] == recorded["snapshot"]["qpos"]
    assert response["session"]["time"] == 0
    assert response["total"] == 2
    assert call(manager, sid, "inspect") == live
    assert rendering.instances[0].calls[-1][0]["time"] == pytest.approx(.02)


def test_compact_replay_metadata_and_image_use_recorded_frame(session, rendering):
    manager, sid = session
    call(manager, sid, "record")
    recorded = call(manager, sid, "step")
    live = call(manager, sid, "reset")
    response = call(manager, sid, "render", offset=1, detail="compact", jpeg_quality=70)
    for key in ["time", "root_pose", "joint_names", "joint_pos"]:
        assert response["snapshot"][key] == recorded["snapshot"][key]
    assert response["snapshot"]["time"] == pytest.approx(.02)
    assert response["session"]["time"] == 0
    assert "geoms" not in response["snapshot"] and "contacts" not in response["snapshot"]
    assert rendering.instances[0].calls[-1][0]["qpos"] == recorded["snapshot"]["qpos"]
    assert rendering.instances[0].calls[-1][1]["lookat"] == recorded["snapshot"]["root_pose"]["pos"]
    assert call(manager, sid, "inspect") == live


@pytest.mark.parametrize("params", [
    {"width": True}, {"width": 961}, {"height": 641},
    {"azimuth": float("nan")}, {"elevation": 90}, {"distance": 0},
    {"lookat": [0, 1]}, {"lookat": [0, 0, float("inf")]}, {"offset": 0},
    {"jpeg_quality": 39}, {"jpeg_quality": 96}, {"jpeg_quality": True}, {"jpeg_quality": 70.5},
    {"detail": "other"}, {"detail": {}},
])
def test_invalid_frame_request_fails_before_graphics_allocation(session, rendering, params):
    manager, sid = session
    with pytest.raises(ValueError):
        call(manager, sid, "render", **params)
    assert not rendering.instances


def test_render_failure_is_explicit_and_does_not_fail_physics(session, rendering):
    manager, sid = session
    call(manager, sid, "render")
    rendering.instances[0].failure = RuntimeError("EGL context unavailable")
    with pytest.raises(RuntimeError, match="EGL"):
        call(manager, sid, "render")
    assert call(manager, sid, "step")["session"]["state"] == "paused"
    call(manager, sid, "close")
    assert rendering.instances[0].closed
    with pytest.raises(ValueError, match="closed"):
        call(manager, sid, "render")


def test_frame_wait_does_not_hold_physics_lock_and_concurrency_is_bounded(session, rendering):
    manager, sid = session
    call(manager, sid, "render")
    renderer = rendering.instances[0]
    renderer.entered, renderer.release = threading.Event(), threading.Event()
    replies, failures = [], []

    def request():
        try:
            replies.append(call(manager, sid, "render"))
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=request)
    thread.start()
    try:
        assert renderer.entered.wait(2)
        with pytest.raises(RuntimeError, match="busy"):
            call(manager, sid, "render")
        assert call(manager, sid, "step")["snapshot"]["time"] == pytest.approx(.02)
    finally:
        renderer.release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert not failures
    assert replies[0]["snapshot"]["time"] == 0
