"""Asset validation, real HTTP boundary and durable subprocess job lifecycle."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

pytest.importorskip("mujoco")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_service.contracts import validate_profile, validate_motion
from motion_service.store import AssetStore, atomic_json, read_json
from motion_service.jobs import JobManager
from motion_service.server import MotionService, make_handler
from motion_service.worker import compare_onnx_actions
from motion_service.resources import ResourceManager


@pytest.fixture
def resources():
    return ResourceManager(gpu_query=lambda: [
        {"id": 2, "uuid": "GPU-fixture-2", "name": "test GPU", "memory_total_mib": 81920,
         "memory_used_mib": 0, "utilization_percent": 0}],
        runtime_query=lambda: {"torch": True, "supermimic": True, "mjlab": True, "cuda": True, "nccl": True},
        allowed=[2], cache_seconds=0, background_probe=False)


@pytest.mark.parametrize("exported", [np.array([[np.nan, 0.]]), np.array([[np.inf, 0.]]),
                                    np.zeros((2, 1)), np.array([[.01, 0.]])])
def test_invalid_onnx_outputs_cannot_be_reported_as_matching(exported):
    with pytest.raises(ValueError):
        compare_onnx_actions(exported, np.zeros((1, 2)))


def test_finite_onnx_difference_is_measured():
    assert compare_onnx_actions(np.array([[.00001, 0.]]), np.zeros((1, 2))) == pytest.approx(.00001)


@pytest.fixture
def profile(tmp_path):
    model = tmp_path / "model.xml"
    model.write_text('<mujoco><option timestep="0.002"/><worldbody><body name="pelvis" pos="0 0 1">'
                     '<joint name="j" type="hinge" range="-1 1"/><geom type="sphere" size=".1" mass="1"/>'
                     '</body></worldbody><actuator><motor joint="j"/></actuator></mujoco>')
    return validate_profile({"id": "test-g1", "label": "test", "robot_type": "unitree_g1",
                             "model_version": "test", "task_id": "SuperMimic-GenCoord-G1",
                             "model_path": str(model), "joint_names": ["j"], "control_hz": 50,
                             "physics_dt": .002, "default_joint_pos": [0], "action_scale": [.2],
                             "kp": [10], "kd": [1], "joint_limits": {"lower": [-1], "upper": [1]},
                             "quaternion_order": "wxyz", "history_length": 10})


def motion_bytes(**updates):
    fields = {"joint_names": np.array(["j"]), "joint_pos": np.zeros((4, 1)),
              "joint_vel": np.zeros((4, 1)), "root_pos_w": np.zeros((4, 3)),
              "root_quat_w": np.tile([1., 0, 0, 0], (4, 1)), "root_lin_vel_w": np.zeros((4, 3)),
              "root_ang_vel_w": np.zeros((4, 3)), "fps": np.array(50)}
    fields.update(updates)
    stream = io.BytesIO()
    np.savez_compressed(stream, **fields)
    return stream.getvalue()


def upload(store, data=None):
    blob = data or motion_bytes()
    id = store.upload_begin({"profile_id": "test-g1", "name": "clip.npz", "size": len(blob)})["upload_id"]
    store.upload_chunk({"upload_id": id, "offset": 0, "data": base64.b64encode(blob).decode()})
    return store.upload_finish({"upload_id": id})["dataset"]


@pytest.fixture
def store(tmp_path, profile):
    return AssetStore(tmp_path / "data", {profile["id"]: profile})


def test_upload_roundtrip_is_idempotent_and_hides_server_paths(store):
    data = motion_bytes()
    id = store.upload_begin({"profile_id": "test-g1", "name": "clip.npz", "size": len(data)})["upload_id"]
    chunk = {"upload_id": id, "offset": 0, "data": base64.b64encode(data[:100]).decode()}
    assert store.upload_chunk(chunk)["offset"] == 100
    assert store.upload_chunk(chunk)["offset"] == 100
    with pytest.raises(ValueError, match="完整"):
        store.upload_finish({"upload_id": id})
    store.upload_chunk({"upload_id": id, "offset": 100, "data": base64.b64encode(data[100:]).decode()})
    result = store.upload_finish({"upload_id": id})
    assert store.upload_finish({"upload_id": id}) == result
    assert result["dataset"]["frames"] == 4
    assert "model_path" not in store.catalog()["profiles"][0]
    assert len(store.preview({"dataset_id": id, "limit": 2})["frames"]) == 2


@pytest.mark.parametrize("updates,match", [
    ({"joint_names": np.array(["wrong"])}, "关节"),
    ({"joint_pos": np.array([[np.nan]] * 4)}, "数值"),
    ({"joint_pos": np.array([[3.]] * 4)}, "限位"),
    ({"root_quat_w": np.zeros((4, 4))}, "四元数"),
    ({"fps": np.array(30)}, "采样率"),
    ({"joint_names": np.array(["j"], dtype=object)}, "Object arrays"),
])
def test_incompatible_motion_is_never_catalogued(store, updates, match):
    with pytest.raises(ValueError, match=match):
        upload(store, motion_bytes(**updates))
    assert store.catalog()["datasets"] == []


def test_profile_digest_prevents_reinterpreting_existing_data(store):
    d = upload(store)
    store.profiles["test-g1"]["profile_hash"] = "changed"
    with pytest.raises(ValueError, match="不匹配"):
        store.dataset(d["id"])


def test_upload_rejects_paths_offsets_and_quota(store):
    with pytest.raises(ValueError):
        store.get("datasets", "../secret")
    with pytest.raises(ValueError):
        store.upload_begin({"profile_id": "test-g1", "name": "big.npz", "size": 1 << 30})
    id = store.upload_begin({"profile_id": "test-g1", "name": "x.npz", "size": 5})["upload_id"]
    with pytest.raises(ValueError, match="偏移"):
        store.upload_chunk({"upload_id": id, "offset": 2, "data": "YQ=="})


def test_profiles_require_precise_physics_and_joint_contract(profile):
    for update in ({"joint_names": ["j", "j"]}, {"physics_dt": .003},
                   {"kp": [-1]}, {"quaternion_order": "xyzw"}, {"control_hz": True}):
        with pytest.raises(ValueError):
            validate_profile({**profile, **update})


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    # A subprocess fixture exercises the queue boundary, not a production provider.
    (tmp_path / "test_motion_worker_fixture.py").write_text('''
import json,pathlib,sys,time,os
p=pathlib.Path.cwd(); request=json.loads(pathlib.Path(sys.argv[1]).read_text())
mode=os.environ.get("TEST_MOTION_JOB_MODE", "success")
if mode=="sleep": time.sleep(120)
if mode=="fail": raise RuntimeError("test worker failed")
if mode=="missing": sys.exit(0)
for name in ["actor.pt", "world_model.pt"]: (p/name).write_bytes(b"test fixture only")
(p/"run.json").write_text("{}")
(p/"result.json").write_text(json.dumps({"actor":"actor.pt","world_model":"world_model.pt","onnx":None}))
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return "test_motion_worker_fixture"


def wait_for(manager, id, states):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        job = manager.inspect(id)["job"]
        if job["state"] in states:
            return job
        time.sleep(.02)
    pytest.fail(f"job did not reach {states}: {job}")


def submit(manager, store):
    d = upload(store)
    return manager.submit("train", {"profile_id": "test-g1", "dataset_id": d["id"],
                                    "num_envs": 1, "max_iterations": 2})["job"]["id"]


def test_job_success_requires_real_artifact_contract(store, fake_worker, resources):
    m = JobManager(store, worker_module=fake_worker, resources=resources)
    try:
        id = submit(m, store)
        job = wait_for(m, id, {"succeeded", "failed"})
        assert job["state"] == "succeeded"
        assert store.get("models", id)["qualification"] == "unverified"
        assert m.inspect(id)["artifacts"] == ["actor.pt", "world_model.pt"]
    finally:
        m.close()


@pytest.mark.parametrize("mode", ["fail", "missing"])
def test_failed_or_empty_worker_never_publishes_a_model(store, fake_worker, monkeypatch, mode, resources):
    monkeypatch.setenv("TEST_MOTION_JOB_MODE", mode)
    m = JobManager(store, worker_module=fake_worker, resources=resources)
    try:
        id = submit(m, store)
        assert wait_for(m, id, {"failed"})["state"] == "failed"
        assert store.list("models") == []
    finally:
        m.close()


def test_running_job_can_be_cancelled_without_publishing(store, fake_worker, monkeypatch, resources):
    monkeypatch.setenv("TEST_MOTION_JOB_MODE", "sleep")
    m = JobManager(store, worker_module=fake_worker, resources=resources)
    try:
        id = submit(m, store)
        wait_for(m, id, {"running"})
        assert m.cancel(id)["job"]["state"] == "cancelling"
        assert wait_for(m, id, {"cancelled"})["state"] == "cancelled"
        assert not m.processes and not store.list("models")
    finally:
        m.close()


def test_restart_preserves_history_but_does_not_resume_training(store, fake_worker, resources):
    atomic_json(store.root / "jobs" / "old" / "job.json", {"id": "old", "state": "running", "created_at": 1})
    m = JobManager(store, worker_module=fake_worker, resources=resources)
    try:
        assert m.inspect("old")["job"]["state"] == "interrupted"
    finally:
        m.close()


def test_replaced_motion_is_rejected_before_training_or_preview(store):
    d = upload(store)
    path = store.root / "datasets" / (d["id"] + ".npz")
    path.write_bytes(motion_bytes(joint_pos=np.full((4, 1), .1)))
    with pytest.raises(ValueError, match="改变或损坏"):
        store.dataset(d["id"])
    with pytest.raises(ValueError, match="改变或损坏"):
        store.preview({"dataset_id": d["id"]})


def test_replaced_model_is_rejected_before_evaluation_or_export(store, fake_worker, resources):
    m = JobManager(store, worker_module=fake_worker, resources=resources)
    try:
        id = submit(m, store)
        wait_for(m, id, {"succeeded"})
        model, folder = store.verified_model(id)
        (folder / model["actor"]).write_bytes(b"replaced checkpoint")
        with pytest.raises(ValueError, match="改变或损坏"):
            store.verified_model(id)
        with pytest.raises(ValueError, match="改变或损坏"):
            m.submit("evaluate", {"model_id": id, "profile_id": model["profile_id"],
                                  "dataset_id": model["dataset_id"]})
        with pytest.raises(ValueError, match="改变或损坏"):
            store.export(id)
    finally:
        m.close()


def test_real_http_auth_upload_and_simulation_boundary(tmp_path, profile, resources):
    service = MotionService(tmp_path / "http", [profile], resources=resources)
    http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "test-credential-123456"))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{http.server_port}"
    from plugins.motion.client import MotionClient
    client = MotionClient(endpoint, "test-credential-123456")
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(endpoint + "/health")
        assert e.value.code == 401
        assert client.health()["protocol"] == "motus.motion/1"
        assert client.call("catalog")["profiles"][0]["id"] == "test-g1"
        plan = client.call("plan", gpu_mode="auto", num_envs=64, preset="small")
        assert plan["ready"] and plan["gpu_ids"] == [2] and plan["world_size"] == 1
        assert not client.call("jobs")["jobs"]
        created = client.call("simulation_create", profile_id="test-g1")
        id = created["session"]["id"]
        stepped = client.call("simulation_step", session_id=id)
        assert stepped["snapshot"]["time"] > 0
        client.call("simulation_close", session_id=id)
    finally:
        http.shutdown()
        http.server_close()
        service.close()
        thread.join()
