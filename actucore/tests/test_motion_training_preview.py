"""Real sampled-state provenance, bounded work and render-only preview lifecycle."""
from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
mujoco = pytest.importorskip("mujoco")
Image = pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_motion_simulation import profile
from motion_service.contracts import validate_profile
from motion_service.store import AssetStore, atomic_json, read_json
from motion_service.training_preview import PreviewRecorder, TrainingPreview, _PreviewCamera
import motion_service.training_preview as preview_module


@pytest.fixture
def setup(profile, tmp_path):
    bound = validate_profile({**profile, "action_scale": [1.]})
    store = AssetStore(tmp_path / "store", {bound["id"]: bound})
    folder = store.root / "jobs" / "job1"
    atomic_json(folder / "request.json", {"kind": "train", "profile": bound})
    model = mujoco.MjModel.from_xml_path(bound["model_path"])
    qpos = np.repeat(model.qpos0[None, :], 32, axis=0)
    qpos[:, 0] = np.arange(32) * .25
    qvel = np.repeat(np.arange(32, dtype=float)[:, None], model.nv, axis=1)
    data = SimpleNamespace(qpos=qpos, qvel=qvel, time=np.arange(32) * .02)
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=SimpleNamespace(data=data)))
    return store, folder, env


@pytest.fixture
def graphics(monkeypatch):
    class Render:
        instances = []

        def __init__(self, model, *, max_width, max_height):
            self.samples = []
            self.cameras = []
            self.closed = False
            self.nq, self.nv = model.nq, model.nv
            self.instances.append(self)

        def render(self, sample, *, camera, width, height, jpeg_quality, timeout):
            self.samples.append(copy.deepcopy(sample))
            self.cameras.append(copy.deepcopy(camera))
            color = (int(sample["qpos"][0] * 20) % 256, int(sample["time"] * 100) % 256, 40)
            picture = Image.new("RGB", (width, height), color)
            out = io.BytesIO()
            picture.save(out, format="JPEG")
            return {"data": out.getvalue()}

        def close(self):
            self.closed = True

    monkeypatch.setattr("motion_service.rendering.FrameRenderer", Render)
    return Render


def test_rank_samples_are_real_distinct_and_throttled(setup, monkeypatch):
    _, folder, env = setup
    before_qpos, before_qvel = env.unwrapped.sim.data.qpos.copy(), env.unwrapped.sim.data.qvel.copy()
    now = [10.]
    monkeypatch.setattr(preview_module.time, "monotonic", lambda: now[0])
    records = [PreviewRecorder(folder, rank, 3, background=False) for rank in range(3)]
    for recorder in records:
        assert recorder.capture(env, 17)
        assert not recorder.capture(env, 18)
    snapshots = [read_json(recorder.path) for recorder in records]
    assert [len(s["samples"]) for s in snapshots] == [3, 3, 2]
    assert sum(len(s["samples"]) for s in snapshots) == 8
    for snapshot in snapshots:
        assert snapshot["num_envs"] == 32
        assert len({sample["env_id"] for sample in snapshot["samples"]}) == len(snapshot["samples"])
        for sample in snapshot["samples"]:
            assert sample["iteration"] == 17
            np.testing.assert_array_equal(sample["qpos"], before_qpos[sample["env_id"]])
            np.testing.assert_array_equal(sample["qvel"], before_qvel[sample["env_id"]])
            assert sample["time"] == env.unwrapped.sim.data.time[sample["env_id"]]
    np.testing.assert_array_equal(env.unwrapped.sim.data.qpos, before_qpos)
    np.testing.assert_array_equal(env.unwrapped.sim.data.qvel, before_qvel)
    now[0] = 11.
    assert records[0].capture(env, 19)
    assert read_json(records[0].path)["samples"][0]["iteration"] == 19
    assert not list(folder.rglob("*.tmp"))


def test_more_ranks_than_preview_slots_do_not_create_extra_samples(setup):
    _, folder, env = setup
    recorder = PreviewRecorder(folder, 8, 10, background=False)
    assert not recorder.capture(env, 1)
    assert not recorder.path.exists()


def test_capture_failure_replaces_old_frame_and_does_not_abort_training(setup, monkeypatch):
    _, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, background=False)
    assert recorder.capture(env, 3)
    monkeypatch.setattr(preview_module.time, "monotonic", lambda: recorder._next_capture + 1)
    env.unwrapped.sim.data.qpos[0, 0] = np.nan
    assert not recorder.capture(env, 4)
    value = read_json(recorder.path)
    assert value["samples"] == []
    assert "nonfinite" in value["error"]
    assert recorder.last_error


def test_empty_preview_creates_no_renderer(setup, graphics):
    store, _, _ = setup
    preview = TrainingPreview(store)
    try:
        result = preview.render("job1")
        assert result["available"] is False
        assert "image" not in result
        assert graphics.instances == []
    finally:
        preview.close()


def test_grid_uses_each_real_rank_state_and_never_steps_physics(setup, graphics, monkeypatch):
    store, folder, env = setup
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("preview advanced physics"))
    for rank in range(2):
        assert PreviewRecorder(folder, rank, 2, background=False).capture(env, 12 + rank)
    before = copy.deepcopy(env.unwrapped.sim.data)
    preview = TrainingPreview(store)
    try:
        result = preview.render("job1")
        assert result["available"] is True and result["stale"] is False
        assert result["total_envs"] == 64 and result["world_size"] == 2
        assert len(result["samples"]) == 8
        assert {s["rank"] for s in result["samples"]} == {0, 1}
        assert len(graphics.instances) == 1
        assert len(graphics.instances[0].samples) == 8
        for meta, state in zip(result["samples"], graphics.instances[0].samples, strict=True):
            assert meta["iteration"] == 12 + meta["rank"]
            assert state["qpos"] == before.qpos[meta["env_id"]].tolist()
        picture = Image.open(io.BytesIO(base64.b64decode(result["image"]["data"])))
        assert picture.size == (768, 432)
        assert np.asarray(picture).std() > 1
        assert "qpos" not in result["samples"][0]
        preview.render("job1")
        assert len(graphics.instances) == 1
        np.testing.assert_array_equal(env.unwrapped.sim.data.qpos, before.qpos)
        np.testing.assert_array_equal(env.unwrapped.sim.data.qvel, before.qvel)
        np.testing.assert_array_equal(env.unwrapped.sim.data.time, before.time)
    finally:
        preview.close()
    assert all(r.closed for r in graphics.instances)


def test_camera_follows_robot_translation_without_fitting_static_terrain(profile, monkeypatch):
    source = Path(profile["model_path"]).read_text().replace(
        '<body name="robot"',
        '<body name="terrain" pos="100 80 0"><geom type="sphere" size="20"/></body>'
        '<body name="robot"')
    model = mujoco.MjModel.from_xml_string(source)
    camera = _PreviewCamera(model)
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("camera advanced physics"))
    sample = {"qpos": model.qpos0.tolist(), "qvel": [0.] * model.nv}
    before = copy.deepcopy(sample)
    first = camera.view(sample, 192, 216)
    shift = np.array([17., -8., 2.])
    sample["qpos"][:3] = (np.array(sample["qpos"][:3]) + shift).tolist()
    moved = copy.deepcopy(sample)
    second = camera.view(sample, 192, 216)
    assert sample == moved
    assert second["distance"] == pytest.approx(first["distance"])
    np.testing.assert_allclose(np.array(second["lookat"]) - first["lookat"], shift)
    assert first["distance"] < 2  # The unrelated 20-metre terrain body is excluded.
    assert np.linalg.norm(np.array(first["lookat"]) - before["qpos"][:3]) < .5
    assert second["elevation"] < -float(model.vis.global_.fovy) / 2


def test_camera_includes_robot_geometry_and_adapts_to_portrait_tiles(profile):
    source = Path(profile["model_path"]).read_text().replace(
        '<geom type="sphere" size=".1" mass="1"/>',
        '<geom type="box" size="1 .15 .1" mass="1"/>')
    model = mujoco.MjModel.from_xml_string(source)
    sample = {"qpos": model.qpos0.tolist(), "qvel": [0.] * model.nv}
    camera = _PreviewCamera(model)
    wide = camera.view(sample, 768, 432)
    portrait = camera.view(sample, 192, 216)
    assert portrait["distance"] > wide["distance"]


def test_missing_or_old_rank_is_explicitly_stale(setup, graphics):
    store, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 2, background=False)
    assert recorder.capture(env, 20)
    preview = TrainingPreview(store)
    try:
        result = preview.render("job1")
        assert result["stale"] and len(result["samples"]) == 4
        assert PreviewRecorder(folder, 1, 2, background=False).capture(env, 20)
        old = read_json(recorder.path)
        old["captured_at"] -= 60
        atomic_json(recorder.path, old)
        assert preview.render("job1")["stale"] is True
    finally:
        preview.close()


def test_bound_model_changes_are_rejected(setup, graphics):
    store, folder, env = setup
    PreviewRecorder(folder, 0, 1, background=False).capture(env, 1)
    path = Path(store.profiles["test"]["model_path"])
    path.write_text(path.read_text().replace('mass="1"', 'mass="2"'))
    preview = TrainingPreview(store)
    try:
        with pytest.raises(ValueError, match="模型"):
            preview.render("job1")
        assert graphics.instances == []
    finally:
        preview.close()


def test_renderer_cache_is_bounded_and_close_is_final(setup, graphics):
    store, folder, env = setup
    PreviewRecorder(folder, 0, 1, background=False).capture(env, 1)
    preview = TrainingPreview(store)
    preview.render("job1", width=768, height=432)
    preview.render("job1", width=640, height=360)
    preview.render("job1", width=960, height=540)
    assert len(graphics.instances) == 3
    assert graphics.instances[0].closed
    assert len(preview._renderers) == 2
    preview.close()
    preview.close()
    assert all(r.closed for r in graphics.instances)
    with pytest.raises(RuntimeError, match="closed"):
        preview.render("job1")


@pytest.mark.parametrize("alter", [
    lambda value: value.update(rank=2),
    lambda value: value["samples"][0].update(qpos=[0.]),
    lambda value: value["samples"][1].update(env_id=value["samples"][0]["env_id"]),
    lambda value: value["samples"][0].update(time=float("nan")),
])
def test_corrupt_snapshot_is_not_rendered_as_success(setup, graphics, alter):
    store, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, background=False)
    recorder.capture(env, 1)
    value = read_json(recorder.path)
    alter(value)
    recorder.path.write_text(json.dumps(value))
    preview = TrainingPreview(store)
    try:
        with pytest.raises(ValueError):
            preview.render("job1")
    finally:
        preview.close()


def test_job_id_cannot_escape_store(setup):
    store, _, _ = setup
    preview = TrainingPreview(store)
    with pytest.raises(ValueError):
        preview.render("../job1")
    preview.close()


def test_background_writer_never_holds_training_and_drops_busy_samples(setup, monkeypatch):
    import threading
    import time

    _, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, sample_hz=4)
    entered, release = threading.Event(), threading.Event()
    original = recorder._publish
    def delayed(packet):
        entered.set()
        assert release.wait(3)
        original(packet)
    monkeypatch.setattr(recorder, "_publish", delayed)
    before = env.unwrapped.sim.data.qpos.copy()
    try:
        began = time.perf_counter()
        assert recorder.capture(env, 1)
        assert time.perf_counter() - began < .2
        assert entered.wait(1)
        env.unwrapped.sim.data.qpos[:] += 10
        recorder._next_capture = 0
        assert not recorder.capture(env, 2)
        assert recorder.stats()["skipped_busy"] == 1
        release.set()
        assert recorder.flush()
        value = read_json(recorder.path)
        np.testing.assert_array_equal(value["samples"][0]["qpos"], before[0])
        assert value["samples"][0]["iteration"] == 1
    finally:
        release.set()
        recorder.close()
    assert not recorder._thread.is_alive()


def test_history_is_bounded_and_replay_uses_recorded_states_without_changing_live(setup, graphics):
    store, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, history_frames=3, background=False)
    for step in range(5):
        recorder._next_capture = 0
        env.unwrapped.sim.data.qpos[:, 0] += .05
        assert recorder.capture(env, step + 1)
    recorder.close()
    assert len(list((recorder.path.parent / "preview-frames").glob("*.json"))) == 3
    history = read_json(recorder.path.parent / "preview-history.json")["frames"]
    assert [f["iteration"] for f in history] == [3, 4, 5]
    original = recorder.path.read_bytes()
    preview = TrainingPreview(store)
    try:
        live = preview.render("job1")
        replay = preview.render("job1", frame=0)
        assert replay["playback"]["frame_count"] == 3
        assert replay["playback"]["mode"] == "replay"
        assert all(s["iteration"] == 3 for s in replay["samples"])
        assert all(s["iteration"] == 5 for s in live["samples"])
        assert replay["frame_id"] != live["frame_id"]
        assert recorder.path.read_bytes() == original
        assert preview.render("job1")["frame_id"] == live["frame_id"]
    finally:
        preview.close()


def test_unchanged_frame_reuses_jpeg_and_large_asset_hash(setup, graphics, monkeypatch):
    store, folder, env = setup
    PreviewRecorder(folder, 0, 1, background=False).capture(env, 1)
    actual = preview_module.file_sha256
    reads = []
    monkeypatch.setattr(preview_module, "file_sha256", lambda path: (reads.append(path), actual(path))[1])
    preview = TrainingPreview(store)
    try:
        first = preview.render("job1")
        second = preview.render("job1")
        assert second["cached"] and not first["cached"]
        assert first["frame_id"] == second["frame_id"]
        assert first["captured_at"] == second["captured_at"]
        assert second["age_s"] >= first["age_s"]
        assert len(reads) == 1
        assert len(graphics.instances[0].samples) == 8
    finally:
        preview.close()


def test_snapshot_change_cannot_hide_behind_same_sequence_cache(setup, graphics):
    store, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, background=False)
    recorder.capture(env, 1)
    preview = TrainingPreview(store)
    try:
        preview.render("job1")
        value = read_json(recorder.path)
        value["samples"][0]["qpos"] = [0.]
        atomic_json(recorder.path, value)
        with pytest.raises(ValueError, match="维度"):
            preview.render("job1")
    finally:
        preview.close()


def test_single_sample_is_one_actual_environment_and_switch_reuses_context(setup, graphics):
    store, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, background=False)
    recorder.capture(env, 1)
    preview = TrainingPreview(store)
    try:
        grid = preview.render("job1")
        single = preview.render("job1", sample=3)
        assert single["sample_count"] == 8 and single["selected_sample"] == 3
        assert len(single["samples"]) == 1
        assert single["samples"][0] == grid["samples"][3]
        assert single["sample_sources"][3]["env_id"] == grid["samples"][3]["env_id"]
        assert single["frame_id"] != grid["frame_id"]
        assert graphics.instances[-1].samples[-1]["qpos"] == env.unwrapped.sim.data.qpos[12].tolist()
        preview.render("job1", sample=7)
        preview.render("job1")
        assert len(graphics.instances) == 2
    finally:
        preview.close()


def test_wide_monitor_keeps_eight_samples_in_four_by_two_grid(setup, graphics):
    store, folder, env = setup
    PreviewRecorder(folder, 0, 1, background=False).capture(env, 1)
    preview = TrainingPreview(store)
    try:
        result = preview.render("job1", width=896, height=320)
        assert len(result["samples"]) == 8
        assert any(key[-2:] == (224, 160) for key in preview._renderers)
        assert result["playback"]["selected_sequence"] == 1
    finally:
        preview.close()


def test_short_monitor_groups_keep_real_sources_and_separate_cached_frames(setup, graphics):
    store, folder, env = setup
    for rank in range(2):
        PreviewRecorder(folder, rank, 2, background=False).capture(env, rank + 10)
    preview = TrainingPreview(store)
    before = env.unwrapped.sim.data.qpos.copy()
    try:
        first = preview.render("job1", width=896, height=240, sample_start=0, sample_limit=4)
        second = preview.render("job1", width=896, height=240, sample_start=4, sample_limit=4)
        assert first["visible_sample_indices"] == [0, 1, 2, 3]
        assert second["visible_sample_indices"] == [4, 5, 6, 7]
        assert (second["grid_columns"], second["grid_rows"]) == (4, 1)
        assert first["sample_count"] == second["sample_count"] == 8
        assert first["frame_id"] != second["frame_id"]
        assert {s["rank"] for s in first["samples"]} == {0}
        assert {s["rank"] for s in second["samples"]} == {1}
        assert first["sample_sources"] == second["sample_sources"]
        repeated = preview.render("job1", width=896, height=240, sample_start=0, sample_limit=4)
        assert repeated["cached"] and repeated["frame_id"] == first["frame_id"]
        assert len(graphics.instances) == 1  # Paging does not allocate more GPU contexts.
        assert len(graphics.instances[0].samples) == 8
        np.testing.assert_array_equal(env.unwrapped.sim.data.qpos, before)
        portrait = preview.render("job1", width=768, height=432, sample_start=0, sample_limit=4)
        assert (portrait["grid_columns"], portrait["grid_rows"]) == (4, 1)
        assert len(portrait["samples"]) == 4
    finally:
        preview.close()


def test_grouped_history_and_partial_group_use_only_available_samples(setup, graphics):
    store, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, background=False)
    recorder.capture(env, 3)
    recorder._next_capture = 0
    recorder.capture(env, 4)
    preview = TrainingPreview(store)
    try:
        result = preview.render("job1", height=160, frame=0, sample_start=6, sample_limit=4)
        assert result["visible_sample_indices"] == [6, 7]
        assert len(result["samples"]) == 2
        assert all(s["iteration"] == 3 for s in result["samples"])
        assert result["playback"]["mode"] == "replay"
        assert result["image"]["height"] == 160
    finally:
        preview.close()


@pytest.mark.parametrize("options", [
    {"sample_start": -1}, {"sample_start": 8}, {"sample_start": True},
    {"sample_limit": 0}, {"sample_limit": 9}, {"sample_limit": 1.5},
    {"sample": 1, "sample_start": 0}, {"sample": 1, "sample_limit": 4},
])
def test_preview_group_bounds_rejected_before_rendering(setup, graphics, options):
    store, _, _ = setup
    preview = TrainingPreview(store)
    try:
        with pytest.raises(ValueError):
            preview.render("job1", **options)
        assert not graphics.instances
    finally:
        preview.close()


def test_finished_legacy_job_has_last_frame_but_no_invented_history(setup, graphics):
    store, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, history_frames=0, background=False)
    recorder.capture(env, 24)
    preview = TrainingPreview(store)
    try:
        result = preview.render("job1")
        assert result["available"] and result["playback"]["frame_count"] == 0
        with pytest.raises(ValueError, match="不存在"):
            preview.render("job1", frame=0)
    finally:
        preview.close()


def test_invalid_history_entries_are_rejected_as_validation_errors(setup, graphics):
    store, folder, env = setup
    recorder = PreviewRecorder(folder, 0, 1, background=False)
    recorder.capture(env, 1)
    path = recorder.path.parent / "preview-history.json"
    value = read_json(path)
    value["frames"] = ["corrupt"]
    atomic_json(path, value)
    preview = TrainingPreview(store)
    try:
        with pytest.raises(ValueError, match="索引元素"):
            preview.render("job1")
    finally:
        preview.close()


@pytest.mark.parametrize("rate", [0, 9, float("nan"), True])
def test_preview_rate_is_bounded(setup, rate):
    _, folder, _ = setup
    with pytest.raises(ValueError):
        PreviewRecorder(folder, 0, 1, sample_hz=rate)
