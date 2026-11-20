"""Topology contract plus a real two-process gradient/normalization check."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from motion_service.distributed import RankProgress, Synchronizer, Topology, run_loop, split_batches


def request(**changes):
    return {"gpu_ids": [0, 2], "world_size": 2, "num_envs": 64, "envs_per_rank": 32, **changes}


def topology(rank=0):
    return Topology.from_request(request(), {"WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)})


def test_total_envs_and_physical_to_logical_device_mapping():
    rank = topology(1)
    assert rank.gpu_ids[rank.rank] == 2
    assert rank.device == "cuda:1"
    assert rank.envs_per_rank == 32
    assert not rank.primary
    cfg = SimpleNamespace(wm=SimpleNamespace(batch_size=64), policy=SimpleNamespace(batch_size=4096))
    assert split_batches(cfg, rank) == {"wm": {"requested_total": 64, "effective_total": 64, "per_rank": 32},
                                       "policy": {"requested_total": 4096, "effective_total": 4096, "per_rank": 2048}}
    assert cfg.wm.batch_size == 32


@pytest.mark.parametrize("change", [
    {"gpu_ids": []}, {"gpu_ids": None}, {"gpu_ids": 2}, {"gpu_ids": [0, 0]}, {"gpu_ids": [False, 2]},
    {"gpu_ids": [-1, 2]}, {"gpu_ids": "0,2"}, {"world_size": True},
    {"world_size": 3}, {"num_envs": 63}, {"num_envs": 0}, {"num_envs": True},
    {"envs_per_rank": 64},
])
def test_invalid_request_rejected(change):
    with pytest.raises(ValueError):
        Topology.from_request(request(**change), {"WORLD_SIZE": "2"})


@pytest.mark.parametrize("environ", [{}, {"WORLD_SIZE": "bad"},
    {"WORLD_SIZE": "2", "RANK": "2", "LOCAL_RANK": "2"},
    {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "0"}])
def test_launcher_must_match_request(environ):
    with pytest.raises(ValueError):
        Topology.from_request(request(), environ)


def test_old_single_gpu_request_stays_compatible():
    rank = Topology.from_request({"num_envs": 64}, {})
    assert rank.world_size == 1 and rank.envs_per_rank == 64 and rank.primary


@pytest.mark.parametrize("size", range(1, 9))
def test_global_batch_rounds_up_for_any_gpu_count(size):
    rank = Topology(tuple(range(size)), size, 0, 0, 32 * size, 32)
    cfg = SimpleNamespace(wm=SimpleNamespace(batch_size=64), policy=SimpleNamespace(batch_size=4096))
    batches = split_batches(cfg, rank)
    for value in batches.values():
        assert value["effective_total"] == size * value["per_rank"]
        assert value["requested_total"] <= value["effective_total"] < value["requested_total"] + size


def test_rank_progress_is_atomic_throttled_and_terminal(tmp_path):
    progress = RankProgress(topology(1), 100, tmp_path)
    updates = {"world_model": 7, "policy": 6}
    progress.iteration = 9
    progress.write("policy", updates, force=True)
    before = progress.path.read_text()
    progress.write("training", updates)
    assert progress.path.read_text() == before
    progress.write("failed", updates, force=True, error="rank lost")
    record = json.loads(progress.path.read_text())
    assert record["rank"] == 1 and record["iteration"] == 9
    assert record["policy_updates"] == 6 and record["error"] == "rank lost"
    assert record["pid"] == os.getpid()
    assert not list(progress.path.parent.glob("*.tmp"))


def test_import_failure_keeps_rank_pid_and_terminal_progress(monkeypatch, tmp_path):
    from motion_service.worker import train
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    with pytest.raises(ImportError):
        train({"num_envs": 64, "max_iterations": 100, "preset": "small"})
    value = json.loads((tmp_path / "ranks/rank-0/progress.json").read_text())
    assert value["phase"] == "failed" and value["iteration"] == 0
    assert value["pid"] == os.getpid() and "torch" in value["error"]


def _optimize_step(*args, **kwargs):
    return "unchanged global"


class LearnerFixture:
    def _train_batch(self, item):
        return _optimize_step(item)


def test_learner_adaptation_does_not_patch_modules_or_other_instances():
    sync = object.__new__(Synchronizer)
    sync.method_hashes = {}
    sync.optimize = lambda label, value: (label, value)
    first, second = LearnerFixture(), LearnerFixture()
    sync.bind_method(first, "_train_batch", "actor")
    assert first._train_batch(12) == ("actor", 12)
    assert second._train_batch(12) == "unchanged global"
    assert _optimize_step() == "unchanged global"
    assert len(sync.method_hashes["actor._train_batch"]) == 64


def test_single_rank_unknown_optimizer_count_is_not_reported_as_zero(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sync = object.__new__(Synchronizer)
    sync.topology = Topology.from_request({"num_envs": 64}, {})
    sync.initialized = False
    sync.optimizer_steps = {"world_model": 0, "actor": 0, "critic": 0}
    measured = {"world_model": 186, "policy": 186}
    # A parameterless module isolates metadata ownership from tensor hashing.
    evidence = sync.verify({"module": SimpleNamespace(state_dict=lambda: {})}, measured)
    assert evidence[0]["optimizer_steps"] is None
    assert evidence[0]["updates"] == measured
    saved = json.loads((tmp_path / "ranks/rank-0/result.json").read_text())
    assert saved["optimizer_steps"] is None and saved["updates"] == measured


def test_secondary_rank_does_not_accumulate_episode_tensors_or_write_artifacts(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "supermimic.runners.supermimic_runner",
                        SimpleNamespace(Ratio=lambda ratio: lambda step: 1))
    extras = []
    def collect():
        extras.append(object())
        assert len(extras) <= 2
        return 32
    def forbidden(*args, **kwargs):
        pytest.fail("secondary rank wrote logs or artifacts")
    runner = SimpleNamespace(env=SimpleNamespace(num_envs=32, reset=lambda: None),
        _seed_sim_history=lambda: None, step=collect,
        cfg=SimpleNamespace(max_iterations=15, log_interval=2, save_interval=1, replay_ratio=16,
                            wm=SimpleNamespace(num_batches=1, batch_size=32, horizon=8)),
        replay_buffer=SimpleNamespace(num_transitions=0),
        logger=SimpleNamespace(ep_extras=extras, log=forbidden),
        _create_log_dir=forbidden, _init_wandb=forbidden)
    rank = topology(1)
    progress = RankProgress(rank, 15, tmp_path)
    run_loop(runner, rank, progress, {"world_model": 0, "policy": 0}, forbidden)
    assert progress.iteration == 15
    assert len(extras) <= 2


def _cpu_rank(rank, folder):
    """Top-level spawn target; intentionally different data on both ranks."""
    import torch
    from rsl_rl.modules import EmpiricalNormalization
    os.chdir(folder)
    torch.set_num_threads(1)
    sync = Synchronizer(topology(rank), backend="gloo", timeout_s=30,
                        init_method="file://" + str(Path(folder) / "rendezvous"))
    try:
        modules = {name: torch.nn.Linear(1, 1, bias=False) for name in ("world_model", "actor", "critic")}
        for module in modules.values():
            module.weight.data.fill_(1 + rank)
        normalizer = EmpiricalNormalization(1)
        modules["normalization"] = normalizer
        sync.broadcast_modules(modules.values())
        for name in ("world_model", "actor", "critic"):
            module = modules[name]
            optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
            # Mean(10,-2)=4, then clip to1. Clipping locally before averaging
            # would yield zero, so this distinguishes the required ordering.
            loss = module(torch.tensor([[10.0 if rank == 0 else -2.0]])).sum()
            sync.optimize(name, [optimizer], loss, module.parameters(), max_norm=1.0)
            assert abs(module.weight.item()) < 1e-5
        # Upstream global statistics, including between-rank variance correction.
        normalizer.update(torch.tensor([[0.0], [2.0]]) + 4 * rank)
        torch.testing.assert_close(normalizer.mean, torch.tensor([3.0]))
        torch.testing.assert_close(normalizer._var, torch.tensor([[5.0]]))
        assert normalizer.count.item() == 4
        # Differently unused parameters must still join a fixed collective order.
        a = torch.nn.Parameter(torch.tensor([1.0]))
        b = torch.nn.Parameter(torch.tensor([1.0]))
        a.grad = torch.tensor([2.0]) if rank == 0 else None
        b.grad = None
        sync.reduce_gradients([a, b])
        torch.testing.assert_close(a.grad, torch.tensor([1.0]))
        assert b.grad is None
        ranks = sync.verify(modules, {"world_model": 1, "policy": 1})
        assert len(ranks) == 2
        assert ranks[0]["state_hashes"] == ranks[1]["state_hashes"]
        if rank == 1:
            modules["critic"].weight.data.add_(0.25)
        try:
            sync.verify(modules, {"world_model": 1, "policy": 1})
        except RuntimeError as exc:
            assert "不一致" in str(exc)
        else:
            raise AssertionError("divergent rank must never be reported synchronized")
    finally:
        sync.close()


def test_two_real_cpu_ranks_sync_gradients_before_clipping_and_global_normalization(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    torch.multiprocessing.spawn(_cpu_rank, args=(str(tmp_path),), nprocs=2, join=True)
    results = [json.loads((tmp_path / "ranks" / f"rank-{rank}" / "result.json").read_text()) for rank in range(2)]
    assert results[0]["state_hashes"] == results[1]["state_hashes"]
    assert results[0]["optimizer_steps"] == {"world_model": 1, "actor": 1, "critic": 1}
