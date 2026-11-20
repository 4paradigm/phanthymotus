"""Physical GPU selection, shared admission, launcher and progress contracts."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import threading
import time

import pytest

from motion_service.resources import ResourceManager, ResourceUnavailable
from motion_service.jobs import JobManager
from motion_service.store import atomic_json, read_json
from test_motion_service import profile, store, fake_worker, upload, wait_for


@pytest.fixture
def inventory():
    return [{"id": i, "uuid": f"GPU-test-{i}", "name": "NVIDIA A100",
             "memory_total_mib": 81920, "memory_used_mib": 337 if i == 2 else 8600,
             "utilization_percent": 0} for i in range(8)]


def resources(rows, **kwargs):
    return ResourceManager(gpu_query=lambda: copy.deepcopy(rows), allowed=list(range(8)), cache_seconds=0, background_probe=False,
                           runtime_query=lambda: {"torch": True, "supermimic": True, "mjlab": True, "cuda": True,
                                                  "nccl": True}, **kwargs)


def test_all_eight_cards_can_be_shared_with_actual_memory_and_load(inventory):
    manager = resources(inventory)
    view = manager.inventory()
    assert len(view["gpus"]) == 8 and all(g["selectable"] for g in view["gpus"])
    assert view["training"]["default_gpu_ids"] == [2]
    assert view["gpus"][0]["shared"] and not view["gpus"][2]["shared"]
    plan = manager.plan([0, 2], 64)
    assert plan == {"gpu_ids": [0, 2], "gpu_uuids": ["GPU-test-0", "GPU-test-2"],
                    "world_size": 2, "num_envs": 64, "envs_per_rank": 32}
    assert manager.plan(list(range(8)), 64)["envs_per_rank"] == 8


def test_auto_chooses_both_count_and_low_load_devices_from_workload(inventory):
    inventory[2]["utilization_percent"] = 70
    inventory[3]["memory_used_mib"] = 100
    manager = resources(inventory)
    small = manager.auto_plan(64)
    assert small["gpu_ids"] == [3] and small["envs_per_rank"] == 64
    larger = manager.auto_plan(2048)
    assert larger["world_size"] == 2 and larger["envs_per_rank"] == 1024
    assert 2 not in larger["gpu_ids"]
    assert manager.auto_plan(2048, "standard")["world_size"] == 4


def test_auto_plan_does_not_round_robot_count_or_need_manual_gpu_count(inventory):
    result = resources(inventory).auto_plan(4097)
    assert result["num_envs"] == 4097
    assert result["world_size"] * result["envs_per_rank"] == 4097
    assert result["gpu_ids"]


def test_auto_reselects_using_current_resource_conditions(inventory):
    manager = resources(inventory)
    assert manager.auto_plan(64)["gpu_ids"] == [2]
    inventory[2]["memory_used_mib"] = 80000
    inventory[0]["utilization_percent"] = 100
    assert manager.auto_plan(64)["gpu_ids"] == [1]


def test_auto_temporarily_busy_resources_are_queueable_and_recover(inventory):
    manager = resources(inventory)
    for row in inventory:
        row["utilization_percent"] = 100
    with pytest.raises(ResourceUnavailable):
        manager.auto_plan(64)
    preview = manager.training_plan({"num_envs": 64})
    assert not preview["ready"] and preview["queueable"]
    assert preview["gpu_ids"] == [] and preview["world_size"] is None
    inventory[6]["utilization_percent"] = 0
    preview = manager.training_plan({"num_envs": 64})
    assert preview["ready"] and preview["gpu_ids"] == [6]
    assert "gpu_uuids" not in preview


def test_auto_uses_feasible_count_when_desired_count_cannot_get_headroom(inventory):
    for row in inventory:
        row["memory_used_mib"] = 80000
    inventory[2]["memory_used_mib"] = 1000
    result = resources(inventory).auto_plan(2048)
    assert result["gpu_ids"] == [2] and result["envs_per_rank"] == 2048
    assert result["policy"]["min_free_mib_per_gpu"] == 16384


def test_auto_distinguishes_impossible_policy_from_temporary_memory_pressure(inventory):
    for row in inventory:
        row["memory_total_mib"] = 16000
        row["memory_used_mib"] = 1000
    manager = resources(inventory)
    with pytest.raises(ValueError, match="显存保留策略") as caught:
        manager.auto_plan(64, "standard")
    assert not isinstance(caught.value, ResourceUnavailable)
    for row in inventory:
        row["memory_total_mib"] = 81920
        row["memory_used_mib"] = 80000
    with pytest.raises(ResourceUnavailable):
        manager.auto_plan(64)


def test_auto_respects_allocation_nccl_and_server_policy(inventory):
    manager = resources(inventory, auto_targets={"small": 32, "standard": 16})
    manager._allowed = [0, 2]
    result = manager.auto_plan(64)
    assert set(result["gpu_ids"]) == {0, 2} and result["world_size"] == 2
    manager._runtime_query = lambda: {"torch": True, "supermimic": True, "mjlab": True, "cuda": True, "nccl": False}
    manager._runtime = None
    assert manager.auto_plan(64)["world_size"] == 1


@pytest.mark.parametrize("args", [{"gpu_mode": "bogus"}, {"gpu_mode": "auto", "gpu_ids": [2]},
                                  {"gpu_mode": "auto", "gpu_ids": []}, {"gpu_mode": "manual"},
                                  {"num_envs": True}, {"num_envs": 0}, {"num_envs": 8193},
                                  {"preset": "other"}])
def test_auto_invalid_input_is_not_queueable(inventory, args):
    with pytest.raises(ValueError) as caught:
        resources(inventory).training_plan(args)
    assert not isinstance(caught.value, ResourceUnavailable)


def test_live_memory_loss_is_rechecked_before_admission(inventory):
    manager = resources(inventory)
    assert manager.plan([0, 2], 64)["world_size"] == 2
    inventory[0]["memory_used_mib"] = 80000
    with pytest.raises(ValueError, match="剩余显存不足"):
        manager.plan([0, 2], 64)


@pytest.mark.parametrize("ids,total", [([], 64), ([0, 0], 64), ([True], 64), ([0, 8], 64),
                                       ([0, 2], 63), ([0, 2], 1), ([0], True), ([0], 8193),
                                       ("0,2", 64), ([0, 1, 2, 3, 4, 5, 6, 7, 8], 72)])
def test_invalid_resource_requests_never_start_a_worker(inventory, ids, total):
    with pytest.raises(ValueError):
        resources(inventory).plan(ids, total)


def test_gpu_allocation_and_uuid_visibility_are_respected(inventory, monkeypatch):
    manager = resources(inventory)
    manager._allowed = None
    monkeypatch.delenv("MOTION_GPU_IDS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-2,GPU-test-0")
    view = manager.inventory()
    assert {g["id"] for g in view["gpus"] if g["selectable"]} == {0, 2}
    monkeypatch.setenv("MOTION_GPU_IDS", "1,3")
    assert {g["id"] for g in manager.inventory()["gpus"] if g["selectable"]} == {1, 3}


def test_missing_nccl_does_not_advertise_multi_gpu(inventory):
    manager = resources(inventory)
    manager._runtime_query = lambda: {"torch": True, "supermimic": True, "mjlab": True, "cuda": True, "nccl": False}
    assert manager.plan([2], 64)["world_size"] == 1
    with pytest.raises(ValueError, match="NCCL"):
        manager.plan([0, 2], 64)


def test_missing_driver_reports_unavailable_instead_of_fake_gpu():
    def missing():
        raise FileNotFoundError("nvidia-smi")
    manager = ResourceManager(gpu_query=missing, runtime_query=lambda: {}, background_probe=False)
    result = manager.inventory()
    assert result["gpus"] == [] and not result["training"]["available"]
    with pytest.raises(ValueError, match="NVIDIA"):
        manager.plan([0], 64)


def test_slow_runtime_probe_returns_gpu_inventory_immediately_and_then_ready(inventory):
    release = threading.Event()
    entered = threading.Event()
    calls = []
    def probe():
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return {"torch": True, "supermimic": True, "mjlab": True, "cuda": True, "nccl": True}
    manager = ResourceManager(gpu_query=lambda: inventory, runtime_query=probe, allowed=list(range(8)))
    try:
        started = time.monotonic()
        result = manager.inventory()
        assert time.monotonic() - started < .5
        assert entered.wait(1)
        assert len(result["gpus"]) == 8 and result["training"]["checking"]
        assert not result["training"]["available"]
        manager.inventory(refresh=True)
        assert calls == [1]
        release.set()
        deadline = time.monotonic() + 2
        while not manager.inventory()["training"]["available"] and time.monotonic() < deadline:
            time.sleep(.01)
        assert manager.inventory()["training"]["available"]
        assert manager.plan([0, 2], 64)["envs_per_rank"] == 32
    finally:
        release.set()


def test_launcher_uses_selected_world_size_and_single_rank_stays_direct(store, inventory):
    manager = JobManager(store, resources=resources(inventory))
    try:
        folder = store.root / "jobs" / "example"
        command = manager._command({"kind": "train", "world_size": 2}, folder)
        assert "torch.distributed.run" in command and "--nproc-per-node=2" in command
        assert "--max-restarts=0" in command and "--standalone" in command
        assert command[-1] == str(folder / "request.json")
        assert "torch.distributed.run" not in manager._command({"kind": "train", "world_size": 1}, folder)
        assert "torch.distributed.run" not in manager._command({"kind": "evaluate", "world_size": 1}, folder)
    finally:
        manager.close()


def test_job_passes_physical_uuids_and_no_api_secret_to_real_subprocess(store, inventory, tmp_path, monkeypatch):
    # Verify the actual child environment, including non-contiguous GPU IDs.
    (tmp_path / "environment_worker_fixture.py").write_text('''
import os,json,pathlib,sys
p=pathlib.Path.cwd()
(p/'observed.json').write_text(json.dumps({k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','MOTION_JOB_DIR','MOTION_API_TOKEN']}))
raise RuntimeError('fixture stops before training')
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("MOTION_API_TOKEN", "private-fixture-token")
    manager = JobManager(store, worker_module="environment_worker_fixture", resources=resources(inventory))
    try:
        data = upload(store)
        job = manager.submit("train", {"profile_id": "test-g1", "dataset_id": data["id"],
                                       "gpu_ids": [2], "num_envs": 64, "max_iterations": 10})["job"]
        wait_for(manager, job["id"], {"failed"})
        folder = manager._path(job["id"])
        observed = read_json(folder / "observed.json")
        assert observed["CUDA_VISIBLE_DEVICES"] == "GPU-test-2"
        assert observed["MOTION_JOB_DIR"] == str(folder.resolve())
        assert observed["MOTION_API_TOKEN"] is None
        assert manager.inspect(job["id"])["job"]["config"]["envs_per_rank"] == 64
    finally:
        manager.close()


def test_progress_is_derived_from_slowest_actual_rank_and_waits_for_all(store, inventory):
    manager = JobManager(store, resources=resources(inventory))
    try:
        job = {"id": "progress", "state": "running", "config": {"world_size": 2, "max_iterations": 100}}
        def write(rank, iteration):
            atomic_json(manager._path(job["id"]) / "ranks" / f"rank-{rank}" / "progress.json",
                        {"rank": rank, "iteration": iteration, "elapsed_s": 20,
                         "phase": "training", "world_model_updates": 4, "policy_updates": 4})
        write(0, 20)
        partial = manager._with_progress(job)["progress"]
        assert partial["iteration"] == 0 and partial["ranks_ready"] == 1 and "eta_s" not in partial
        write(1, 18)
        complete = manager._with_progress(job)["progress"]
        assert complete["iteration"] == 18 and complete["ranks_ready"] == 2
        assert complete["eta_s"] == pytest.approx(20 * 82 / 18)
    finally:
        manager.close()


def test_distributed_success_requires_actual_rank_sync_evidence(store, inventory):
    manager = JobManager(store, resources=resources(inventory))
    try:
        folder = manager._path("invalid-result")
        atomic_json(folder / "result.json", {})
        with pytest.raises(ValueError, match="同步验证"):
            manager._collect({"kind": "train", "config": {"world_size": 2}}, folder)
    finally:
        manager.close()
