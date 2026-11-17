"""Named reference poses, immutable assets and bounded render-only resources."""
from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
mujoco = pytest.importorskip("mujoco")
Image = pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_service.contracts import validate_motion, validate_profile
from motion_service.dataset_render import DatasetRenderer
from motion_service.store import AssetStore, atomic_json, file_sha256


SCENE = """<mujoco><compiler angle="radian"/><option timestep=".002"/>
<worldbody><light pos="0 0 4"/><geom type="plane" size="5 5 .1"/>
<body name="terrain" pos="100 80 0"><geom type="sphere" size="20"/></body>
<body name="robot/pelvis" pos="0 0 1"><freejoint name="robot/root"/>
<geom type="box" size=".12 .1 .18" mass="2" rgba=".8 .3 .1 1"/>
<body name="robot/arm" pos="0 0 .2"><joint name="robot/shoulder" axis="0 1 0" range="-2 2"/>
<geom type="capsule" fromto="0 0 0 .3 0 0" size=".04" mass=".3"/>
<body name="robot/hand" pos=".3 0 0"><joint name="robot/elbow" axis="0 1 0" range="-2 2"/>
<geom type="capsule" fromto="0 0 0 .25 0 0" size=".03" mass=".2"/></body></body>
</body></worldbody></mujoco>"""


def bind(tmp_path, source=SCENE, **changes):
    path = tmp_path / "robot.xml"
    path.write_text(source)
    return validate_profile({"id": "test", "label": "Test Robot", "robot_type": "fixture",
        "model_version": "1", "model_path": str(path), "root_body_name": "pelvis",
        # Deliberately opposite to compiled joint order, with mjlab prefixes.
        "joint_names": ["elbow", "shoulder"], "control_hz": 50., "physics_dt": .002,
        "default_joint_pos": [0., 0.], "action_scale": [1., 1.], "kp": [20., 20.], "kd": [1., 1.],
        "joint_limits": {"lower": [-2., -2.], "upper": [2., 2.]}, "quaternion_order": "wxyz", **changes})


def add_dataset(store, name="motion", **changes):
    profile = store.profile("test")
    values = {"joint_names": np.array(profile["joint_names"]),
              "joint_pos": np.array([[.1, -.2], [.7, -.9], [.1, -.2]]),
              "joint_vel": np.array([[1., 2.], [3., 4.], [1., 2.]]),
              "root_pos_w": np.array([[0., 0., 1.], [3., -2., 1.5], [7., 5., 2.]]),
              "root_quat_w": np.array([[1., 0., 0., 0.],
                  [math.sqrt(.5), 0., 0., math.sqrt(.5)], [1., 0., 0., 0.]]),
              "root_lin_vel_w": np.array([[1., 2., 3.]] * 3),
              "root_ang_vel_w": np.array([[1., 0., 0.]] * 3), "fps": np.array(50.), **changes}
    path = store.root / "datasets" / (name + ".npz")
    np.savez(path, **values)
    metadata = {"id": name, "name": name + ".npz", "profile_id": profile["id"],
                "profile_hash": profile["profile_hash"], "sha256": file_sha256(path),
                **validate_motion(path, profile)}
    atomic_json(path.with_suffix(".json"), metadata)
    return path, values


@pytest.fixture
def assets(tmp_path):
    profile = bind(tmp_path)
    store = AssetStore(tmp_path / "store", {profile["id"]: profile})
    path, values = add_dataset(store)
    return store, path, values


@pytest.fixture
def graphics(monkeypatch):
    class Graphics:
        instances = []

        def __init__(self, model, *, width, height):
            self.width, self.height = width, height
            self.closed = False
            self.states = []
            self.instances.append(self)

        def update_scene(self, data, *, camera, scene_option):
            self.states.append({"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "time": data.time,
                                "xpos": data.xpos.copy(), "xquat": data.xquat.copy()})

        def render(self):
            picture = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            picture[:, :, 0] = np.arange(self.width, dtype=np.uint8)
            return picture

        def close(self):
            self.closed = True

    monkeypatch.setattr(mujoco, "Renderer", Graphics)
    return Graphics


def test_named_pose_and_world_velocities_are_mapped_without_physics(assets, graphics, monkeypatch):
    store, path, values = assets
    original = file_sha256(path)
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("dataset preview advanced physics"))
    renderer = DatasetRenderer(store)
    try:
        result = renderer.render("motion", offset=1, width=640, height=360)
        assert result["frame_index"] == 1 and result["frames"] == 3
        assert result["fps"] == 50 and result["time"] == .02 and result["duration_s"] == .06
        assert result["dataset_id"] == "motion" and result["profile_id"] == "test"
        np.testing.assert_allclose(result["root_pose"]["pos"], values["root_pos_w"][1])
        state = graphics.instances[0].states[0]
        # Compiled shoulder precedes elbow; the NPZ deliberately reverses it.
        np.testing.assert_allclose(state["qpos"][7:], [-.9, .7])
        np.testing.assert_allclose(state["qvel"][6:], [4., 3.])
        np.testing.assert_allclose(state["qvel"][:3], [1., 2., 3.])
        np.testing.assert_allclose(state["qvel"][3:6], [0., -1., 0.], atol=1e-14)
        np.testing.assert_allclose(state["qpos"][:7], [*values["root_pos_w"][1], *values["root_quat_w"][1]])
        assert state["time"] == .02
        image = result["image"]
        assert image["mime"] == "image/jpeg"
        assert Image.open(io.BytesIO(base64.b64decode(image["data"]))).size == (640, 360)
        assert file_sha256(path) == original
        renderer.render("motion", offset=0, width=960, height=540)
        assert len(graphics.instances) == 1  # Same ratio reuses its graphics context.
        np.testing.assert_allclose(graphics.instances[0].states[-1]["qpos"][7:], [-.2, .1])
    finally:
        renderer.close()
    assert all(x.closed for x in graphics.instances)


def test_camera_follows_geometry_not_static_terrain_and_accepts_user_view(assets, graphics):
    store, _, values = assets
    renderer = DatasetRenderer(store)
    try:
        first = renderer.render("motion", offset=0)
        translated = renderer.render("motion", offset=2)
        np.testing.assert_allclose(np.array(translated["camera"]["lookat"]) - first["camera"]["lookat"],
                                   values["root_pos_w"][2] - values["root_pos_w"][0])
        assert translated["camera"]["distance"] == pytest.approx(first["camera"]["distance"])
        assert first["camera"]["distance"] < 3
        portrait = renderer.render("motion", width=160, height=640)
        assert portrait["camera"]["distance"] > first["camera"]["distance"]
        custom = renderer.render("motion", azimuth=20, elevation=-30, distance=4, lookat=[1., 2., 3.])
        assert custom["camera"] == {"azimuth": 20., "elevation": -30., "distance": 4., "lookat": [1., 2., 3.]}
    finally:
        renderer.close()


@pytest.mark.parametrize("args", [
    {"offset": -1}, {"offset": 3}, {"offset": True}, {"width": 159}, {"width": 961},
    {"height": 119}, {"height": 641}, {"distance": float("nan")}, {"distance": 0},
    {"elevation": 90}, {"azimuth": True}, {"lookat": [0, 0]}, {"jpeg_quality": 39},
])
def test_bad_frame_and_camera_requests_do_not_create_graphics(assets, graphics, args):
    renderer = DatasetRenderer(assets[0])
    try:
        with pytest.raises(ValueError):
            renderer.render("motion", **args)
        assert not graphics.instances
    finally:
        renderer.close()


def test_dataset_and_model_hashes_remain_enforced_after_caching(assets, graphics):
    store, path, _ = assets
    renderer = DatasetRenderer(store)
    try:
        renderer.render("motion")
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="文件已改变"):
            renderer.render("motion")
        path.write_bytes(original)
        model = Path(store.profile("test")["model_path"])
        model.write_text(model.read_text().replace('size=".12 .1 .18"', 'size=".2 .1 .18"'))
        with pytest.raises(ValueError, match="模型或 MuJoCo"):
            renderer.render("motion")
    finally:
        renderer.close()


def test_profile_binding_and_catalog_metadata_are_checked(assets, graphics):
    store, path, _ = assets
    record = store.get("datasets", "motion")
    atomic_json(path.with_suffix(".json"), {**record, "profile_hash": "different"})
    renderer = DatasetRenderer(store)
    try:
        with pytest.raises(ValueError, match="数据集与"):
            renderer.render("motion")
        atomic_json(path.with_suffix(".json"), {**record, "fps": 25})
        with pytest.raises(ValueError, match="目录记录"):
            renderer.render("motion")
        assert not graphics.instances
        atomic_json(path.with_suffix(".json"), record)
        renderer.render("motion")
        atomic_json(path.with_suffix(".json"), {**record, "fps": 25})
        with pytest.raises(ValueError, match="目录记录"):
            renderer.render("motion")
    finally:
        renderer.close()


@pytest.mark.parametrize("source,changes,match", [
    (SCENE.replace('<freejoint name="robot/root"/>', ''), {}, "自由根"),
    (SCENE.replace('<joint name="robot/elbow" axis="0 1 0" range="-2 2"/>',
                   '<joint name="robot/elbow" type="ball"/>'), {}, "单自由度"),
    (SCENE, {"joint_names": ["elbow", "absent"]}, "不存在或不唯一"),
    (SCENE, {"root_body_name": "arm"}, "自由根刚体"),
    (SCENE.replace('<body name="robot/pelvis"',
                   '<body name="other"><freejoint/><geom size=".1"/></body><body name="robot/pelvis"'), {}, "自由根"),
])
def test_unsupported_joint_layouts_fail_instead_of_copying_wrong_dofs(tmp_path, graphics, source, changes, match):
    profile = bind(tmp_path, source, **changes)
    store = AssetStore(tmp_path / "store", {profile["id"]: profile})
    add_dataset(store)
    renderer = DatasetRenderer(store)
    try:
        with pytest.raises(ValueError, match=match):
            renderer.render("motion")
        assert not graphics.instances
    finally:
        renderer.close()


def test_two_entry_lru_releases_graphics_and_close_is_idempotent(assets, graphics):
    store, _, _ = assets
    add_dataset(store, "second")
    add_dataset(store, "third")
    renderer = DatasetRenderer(store)
    renderer.render("motion")
    renderer.render("second")
    renderer.render("motion")  # Keep the original and evict the second.
    renderer.render("third")
    assert len(renderer._scenes) == 2
    assert [entry.closed for entry in graphics.instances] == [False, True, False]
    renderer.close()
    renderer.close()
    assert all(entry.closed for entry in graphics.instances)
    with pytest.raises(RuntimeError, match="已关闭"):
        renderer.render("motion")


def test_busy_request_does_not_queue_or_change_existing_scene(assets, graphics):
    renderer = DatasetRenderer(assets[0])
    renderer._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="正在渲染"):
            renderer.render("motion")
        assert not graphics.instances and not renderer._scenes
    finally:
        renderer._lock.release()
        renderer.close()


def test_graphics_error_releases_request_lock_for_retry(assets, graphics, monkeypatch):
    renderer = DatasetRenderer(assets[0])
    render = graphics.render
    monkeypatch.setattr(graphics, "render", lambda _: (_ for _ in ()).throw(RuntimeError("graphics failure")))
    try:
        with pytest.raises(RuntimeError, match="graphics failure"):
            renderer.render("motion")
        monkeypatch.setattr(graphics, "render", render)
        assert renderer.render("motion")["available"]
    finally:
        renderer.close()


@pytest.mark.skipif(os.environ.get("MOTION_TEST_GL") != "1", reason="explicit local graphics validation")
def test_real_offscreen_reference_frames(assets, monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("dataset preview advanced physics"))
    renderer = DatasetRenderer(assets[0])
    try:
        first = renderer.render("motion", offset=0, width=320, height=240)
        second = renderer.render("motion", offset=1, width=320, height=240)
        assert first["image"]["data"] != second["image"]["data"]
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(first["image"]["data"]))))
        assert pixels.shape == (240, 320, 3) and pixels.std() > 5
    finally:
        renderer.close()
