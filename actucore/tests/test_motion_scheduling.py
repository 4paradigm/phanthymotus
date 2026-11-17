"""Auto allocation through durable queuing, real subprocesses and HTTP."""
from __future__ import annotations

import copy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.request

import pytest

from motion_service.jobs import JobManager
from motion_service.resources import ResourceUnavailable
from motion_service.server import MotionService, make_handler
from motion_service.store import read_json
from test_motion_service import profile, store, upload, wait_for


def plan(gpu=2, total=64):
    return {"gpu_mode": "auto", "gpu_ids": [gpu], "gpu_uuids": [f"GPU-fixture-{gpu}"],
            "world_size": 1, "num_envs": total, "envs_per_rank": total,
            "selection_reason": f"selected GPU {gpu}", "policy": {"target_envs_per_gpu": 1024}}


class Plans:
    def __init__(self, auto=None, manual=None):
        self.auto = list(auto or [plan()])
        self.manual = list(manual or [plan()])
        self.calls = []
        self.lock = threading.Lock()

    def _next(self, mode, args):
        with self.lock:
            self.calls.append((mode, args))
            sequence = self.auto if mode == "auto" else self.manual
            response = sequence.pop(0) if len(sequence) > 1 else sequence[0]
        if callable(response):
            response = response()
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)

    def auto_plan(self, num_envs, preset="small", *, refresh=True):
        return self._next("auto", (num_envs, preset, refresh))

    def plan(self, gpu_ids, num_envs, *, refresh=True):
        return self._next("manual", (gpu_ids, num_envs, refresh))


@pytest.fixture
def worker(tmp_path, monkeypatch):
    (tmp_path / "schedule_worker_fixture.py").write_text('''
import json,os,pathlib,sys
p=pathlib.Path.cwd()
request=json.loads(pathlib.Path(sys.argv[1]).read_text())
(p/'observed.json').write_text(json.dumps({'request':request,'visible':os.environ['CUDA_VISIBLE_DEVICES']}))
for name in ('actor.pt','world_model.pt'): (p/name).write_bytes(b'test fixture only')
(p/'run.json').write_text('{}')
(p/'result.json').write_text(json.dumps({'actor':'actor.pt','world_model':'world_model.pt','onnx':None}))
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return "schedule_worker_fixture"


def arguments(store, **overrides):
    data = upload(store)
    return {"profile_id": "test-g1", "dataset_id": data["id"], "num_envs": 64,
            "max_iterations": 2, "preset": "small", **overrides}


def eventually(check, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    pytest.fail("condition did not become true")


def test_auto_reselects_at_launch_and_worker_receives_actual_uuid(store, worker):
    resources = Plans(auto=[plan(0), plan(2)])
    manager = JobManager(store, worker_module=worker, resources=resources)
    try:
        submitted = manager.submit("train", arguments(store))["job"]
        assert submitted["config"]["gpu_mode"] == "auto"
        assert submitted["config"]["gpu_ids"] == [0] and submitted["allocation_pending"]
        job = wait_for(manager, submitted["id"], {"succeeded", "failed"})
        assert job["state"] == "succeeded"
        assert job["config"]["gpu_ids"] == [2] and not job["allocation_pending"]
        assert job["config"]["selection_reason"] == "selected GPU 2"
        observed = read_json(manager._path(job["id"]) / "observed.json")
        assert observed["visible"] == "GPU-fixture-2"
        assert observed["request"]["gpu_ids"] == [2]
        assert observed["request"]["gpu_mode"] == "auto"
        assert [mode for mode, _ in resources.calls] == ["auto", "auto"]
    finally:
        manager.close()


def test_auto_submission_and_launch_can_wait_then_allocate(store, worker):
    resources = Plans(auto=[ResourceUnavailable("显存暂时不足"), ResourceUnavailable("等待资源释放"), plan(4)])
    manager = JobManager(store, worker_module=worker, resources=resources, resource_retry_seconds=.03)
    try:
        submitted = manager.submit("train", arguments(store))["job"]
        assert submitted["state"] == "queued" and submitted["waiting_for_resources"]
        assert submitted["config"]["gpu_ids"] == [] and submitted["config"]["world_size"] is None
        assert manager._with_progress(submitted)["state"] == "queued"
        job = wait_for(manager, submitted["id"], {"succeeded", "failed"})
        assert job["state"] == "succeeded" and job["config"]["gpu_ids"] == [4]
        assert not job["waiting_for_resources"] and "next_retry_at" not in job
        assert "message" not in job and len(resources.calls) == 3
    finally:
        manager.close()


def test_cancel_deferred_auto_does_not_wait_for_retry_or_start_worker(store, worker):
    resources = Plans(auto=[ResourceUnavailable("暂时无可用显存")])
    manager = JobManager(store, worker_module=worker, resources=resources, resource_retry_seconds=60)
    try:
        job = manager.submit("train", arguments(store))["job"]
        eventually(lambda: "next_retry_at" in manager.inspect(job["id"])["job"])
        start = time.monotonic()
        cancelled = manager.cancel(job["id"])["job"]
        assert time.monotonic() - start < .5
        assert cancelled["state"] == "cancelled" and not cancelled["waiting_for_resources"]
        assert not cancelled["allocation_pending"] and "next_retry_at" not in cancelled
        assert not manager.processes and store.list("models") == []
        assert not (manager._path(job["id"]) / "observed.json").exists()
    finally:
        start = time.monotonic()
        manager.close()
        assert time.monotonic() - start < 1 and not manager.thread.is_alive()


def test_close_cancels_waiting_jobs_and_wakes_long_retry_timer(store, worker):
    manager = JobManager(store, worker_module=worker,
                         resources=Plans(auto=[ResourceUnavailable("等待")]), resource_retry_seconds=60)
    job = manager.submit("train", arguments(store))["job"]
    eventually(lambda: "next_retry_at" in manager.inspect(job["id"])["job"])
    start = time.monotonic()
    manager.close()
    assert time.monotonic() - start < 1
    assert not manager.thread.is_alive() and not manager.processes
    assert manager.inspect(job["id"])["job"]["state"] == "cancelled"


def test_cancel_during_launch_probe_wins_before_spawn_without_locking_user_out(store, worker):
    entered, release = threading.Event(), threading.Event()
    def probe():
        entered.set()
        assert release.wait(3)
        return plan(4)
    manager = JobManager(store, worker_module=worker, resources=Plans(auto=[plan(0), probe]))
    try:
        job = manager.submit("train", arguments(store))["job"]
        assert entered.wait(2)
        start = time.monotonic()
        assert manager.cancel(job["id"])["job"]["state"] == "cancelled"
        assert time.monotonic() - start < .5
        release.set()
    finally:
        release.set()
        manager.close()
    assert not (manager._path(job["id"]) / "observed.json").exists()
    assert store.list("models") == []


def test_waiting_auto_job_does_not_block_later_runnable_manual_job(store, worker):
    resources = Plans(auto=[ResourceUnavailable("自动保留显存不足")])
    manager = JobManager(store, worker_module=worker, resources=resources, resource_retry_seconds=60)
    try:
        waiting = manager.submit("train", arguments(store))["job"]
        eventually(lambda: "next_retry_at" in manager.inspect(waiting["id"])["job"])
        manual = manager.submit("train", arguments(store, gpu_ids=[2]))["job"]
        finished = wait_for(manager, manual["id"], {"succeeded", "failed"})
        assert finished["state"] == "succeeded" and finished["config"]["gpu_mode"] == "manual"
        assert manager.inspect(waiting["id"])["job"]["state"] == "queued"
        assert [mode for mode, _ in resources.calls].count("manual") == 2
    finally:
        manager.close()


@pytest.mark.parametrize("failure", [ResourceUnavailable("指定 GPU 暂时繁忙"),
                                     {**plan(2), "gpu_uuids": ["GPU-replaced"]}])
def test_manual_launch_revalidation_never_changes_gpu_or_starts_auto_wait(store, worker, failure):
    resources = Plans(manual=[plan(2), failure])
    manager = JobManager(store, worker_module=worker, resources=resources)
    try:
        job = manager.submit("train", arguments(store, gpu_mode="manual", gpu_ids=[2]))["job"]
        failed = wait_for(manager, job["id"], {"failed"})
        assert failed["config"]["gpu_ids"] == [2]
        assert not failed["waiting_for_resources"] and "next_retry_at" not in failed
        assert not (manager._path(job["id"]) / "observed.json").exists()
        assert [mode for mode, _ in resources.calls] == ["manual", "manual"]
    finally:
        manager.close()


def test_invalid_auto_plan_is_hard_error_at_submit_or_launch(store, worker):
    resources = Plans(auto=[ValueError("任务超过服务器总容量")])
    manager = JobManager(store, worker_module=worker, resources=resources)
    try:
        with pytest.raises(ValueError, match="总容量"):
            manager.submit("train", arguments(store))
        assert manager.list() == []
        resources.auto = [ResourceUnavailable("等待"), ValueError("策略配置已失效")]
        job = manager.submit("train", arguments(store))["job"]
        failed = wait_for(manager, job["id"], {"failed"})
        assert "策略配置" in failed["message"]
        assert not failed["waiting_for_resources"] and not failed["allocation_pending"]
        assert not manager._waiting
    finally:
        manager.close()


@pytest.mark.parametrize("fields", [
    {"gpu_mode": "auto", "gpu_ids": [2]}, {"gpu_mode": "auto", "gpu_ids": []},
    {"gpu_mode": "auto", "gpu_ids": ""}, {"gpu_mode": "manual"},
    {"gpu_mode": "bogus"}, {"gpu_mode": None}, {"gpu_mode": []},
])
def test_ambiguous_or_invalid_mode_does_not_enqueue(store, worker, fields):
    resources = Plans()
    manager = JobManager(store, worker_module=worker, resources=resources)
    try:
        with pytest.raises(ValueError):
            manager.submit("train", arguments(store, **fields))
        assert not manager.list() and not resources.calls
    finally:
        manager.close()


def test_http_auto_request_waits_and_can_be_cancelled_without_gpu(tmp_path, profile):
    service = MotionService(tmp_path / "http-data", [profile],
                            resources=Plans(auto=[ResourceUnavailable("训练环境正在检测")]),
                            simulation_factory=lambda *a, **kw: SimpleNamespace(close=lambda: None))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "scheduling-fixture-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def call(action, **args):
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/call",
            data=json.dumps({"action": action, **args}).encode(),
            headers={"Authorization": "Bearer scheduling-fixture-token", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 200
            return json.load(response)
    try:
        waiting = call("train", **arguments(service.store, gpu_mode="auto"))["job"]
        assert waiting["state"] == "queued" and waiting["waiting_for_resources"]
        assert call("jobs")["jobs"][0]["id"] == waiting["id"]
        cancelled = call("cancel", job_id=waiting["id"])["job"]
        assert cancelled["state"] == "cancelled"
        assert not service.jobs.processes and service.store.list("models") == []
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        service.close()
