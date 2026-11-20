"""Single-host synchronous SuperMimic training, without changing upstream files.

The adapter binds an isolated copy of the upstream learner methods to each
learner. Only their optimization boundary changes: backward, mean gradients
across ranks, then the original clipping and optimizers. No torch class,
optimizer, or upstream module global is patched. CUDA graph replay is disabled
because every real update must execute its collectives.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import inspect
import os
from pathlib import Path
import time
import types

from .store import atomic_json


@dataclass(frozen=True)
class Topology:
    gpu_ids: tuple[int, ...]
    world_size: int
    rank: int
    local_rank: int
    total_envs: int
    envs_per_rank: int

    @property
    def primary(self):
        return self.rank == 0

    @property
    def device(self):
        return f"cuda:{self.local_rank}"

    @classmethod
    def from_request(cls, request, environ=None):
        environ = os.environ if environ is None else environ
        ids = request.get("gpu_ids", [0])
        total = request["num_envs"]
        if (not isinstance(ids, list) or not ids or len(ids) > 8
                or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids)):
            raise ValueError("gpu_ids 必须是互不重复的物理 GPU 编号")
        size = request.get("world_size", len(ids))
        if type(size) is not int or size != len(ids):
            raise ValueError("world_size 必须与所选 GPU 数量一致")
        if type(total) is not int or total < size or total % size:
            raise ValueError("总并行环境数必须能被 GPU 数量整除")
        try:
            actual = int(environ.get("WORLD_SIZE", "1"))
            rank = int(environ.get("RANK", "0"))
            local = int(environ.get("LOCAL_RANK", "0"))
        except ValueError as exc:
            raise ValueError("分布式进程环境变量非法") from exc
        if actual != size or not 0 <= rank < size or local != rank:
            raise ValueError("请求与 torchrun 单机进程拓扑不一致")
        per_rank = total // size
        if request.get("envs_per_rank", per_rank) != per_rank:
            raise ValueError("每卡环境数与总环境数不一致")
        return cls(tuple(ids), size, rank, local, total, per_rank)

    def metadata(self):
        return {"gpu_ids": list(self.gpu_ids), "world_size": self.world_size,
                "num_envs": self.total_envs, "envs_per_rank": self.envs_per_rank,
                "mode": "synchronous_allreduce" if self.world_size > 1 else "single_gpu"}


def split_batches(cfg, topology):
    """Equal local batches; round global sizes up by at most world_size - 1."""
    result = {}
    for name in ("wm", "policy"):
        item = getattr(cfg, name)
        total = item.batch_size
        if type(total) is not int or total < 1:
            raise ValueError(f"{name} batch_size 必须是正整数")
        item.batch_size = (total + topology.world_size - 1) // topology.world_size
        result[name] = {"requested_total": total, "effective_total": item.batch_size * topology.world_size,
                        "per_rank": item.batch_size}
    return result


class RankProgress:
    def __init__(self, topology, iterations, root="ranks"):
        self.rank = topology.rank
        self.path = Path(root) / f"rank-{self.rank}" / "progress.json"
        self.total = iterations
        self.start = time.monotonic()
        self.last = -1.0
        self.iteration = 0

    def write(self, phase, updates, *, force=False, error=None):
        now = time.monotonic()
        if not force and now - self.last < 1:
            return
        value = {"rank": self.rank, "iteration": self.iteration,
                 "pid": os.getpid(), "launcher_pid": os.getppid(),
                 "total_iterations": self.total, "elapsed_s": round(now - self.start, 3),
                 "phase": phase, "world_model_updates": updates["world_model"],
                 "policy_updates": updates["policy"], "updated_at": time.time()}
        if error is not None:
            value["error"] = str(error)[:2000]
        atomic_json(self.path, value)
        self.last = now


class Synchronizer:
    def __init__(self, topology, *, backend="nccl", timeout_s=180, init_method=None):
        import torch
        self.topology = topology
        self.torch = torch
        self.dist = torch.distributed
        self.optimizer_steps = {"world_model": 0, "actor": 0, "critic": 0}
        self.method_hashes = {}
        self.initialized = False
        if topology.world_size > 1:
            if backend == "nccl":
                torch.cuda.set_device(topology.local_rank)
            self.dist.init_process_group(backend=backend, rank=topology.rank,
                                         world_size=topology.world_size,
                                         timeout=timedelta(seconds=timeout_s), init_method=init_method)
            self.initialized = True

    def close(self):
        # Never barrier on an exception: torchrun terminates surviving siblings.
        if self.initialized:
            self.dist.destroy_process_group()
            self.initialized = False

    def broadcast_modules(self, modules):
        if not self.initialized:
            return
        with self.torch.no_grad():
            seen = set()
            for module in modules:
                for value in (*module.parameters(), *module.buffers()):
                    if id(value) not in seen:
                        self.dist.broadcast(value, src=0)
                        seen.add(id(value))

    def reduce_gradients(self, params):
        """Fixed ordered buckets, including unused parameters, before clipping."""
        torch = self.torch
        params = list(params)
        if not self.initialized or not params:
            return
        present = torch.tensor([p.grad is not None for p in params],
                               device=params[0].device, dtype=torch.int32)
        self.dist.all_reduce(present)
        # A parameter unused on every rank stays None (Adam/Muon must not decay
        # its state). A locally unused one still joins the global mean gradient.
        used = present.tolist()
        groups = {}
        for count, parameter in zip(used, params, strict=True):
            if count:
                groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)
        for group in groups.values():
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                              for p in group])
            self.dist.all_reduce(flat)
            flat.div_(self.topology.world_size)
            offset = 0
            for parameter in group:
                n = parameter.numel()
                gradient = flat[offset:offset + n].view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = gradient.clone()
                else:
                    parameter.grad.copy_(gradient)
                offset += n

    def optimize(self, label, optimizers, loss, params=None, max_norm=None, clip_value=0.0):
        torch = self.torch
        params = list(params) if params is not None else [
            p for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.reduce_gradients(params)
        grad_norm = None
        if max_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
            if clip_value > 0:
                torch.nn.utils.clip_grad_value_(params, clip_value)
        for optimizer in optimizers:
            optimizer.step()
        self.optimizer_steps[label] += 1
        return grad_norm

    def bind_method(self, learner, name, label):
        """Retain the installed upstream algorithm with an instance-only seam."""
        original = getattr(learner, name).__func__
        if original.__closure__ or "_optimize_step" not in original.__code__.co_names:
            raise RuntimeError(f"不兼容的 SuperMimic learner 方法：{name}")
        scope = dict(original.__globals__)
        scope["_optimize_step"] = lambda *args, **kwargs: self.optimize(label, *args, **kwargs)
        adapted = types.FunctionType(original.__code__, scope, original.__name__, original.__defaults__)
        adapted.__kwdefaults__ = original.__kwdefaults__
        setattr(learner, name, types.MethodType(adapted, learner))
        self.method_hashes[f"{label}.{name}"] = hashlib.sha256(
            inspect.getsource(original).encode()).hexdigest()

    def prepare(self, runner):
        from rsl_rl.modules import EmpiricalNormalization
        source = inspect.getsource(EmpiricalNormalization.update)
        if self.initialized and ("torch.distributed.is_initialized()" not in source
                                 or "torch.distributed.all_reduce" not in source):
            raise RuntimeError("当前 rsl_rl 归一化实现不支持全局分布式统计")
        wm, policy = runner.world_model_learner, runner.policy_learner
        modules = {"world_model": runner._wm.dynamics, "actor": policy.actor}
        if getattr(policy, "critic", None) is not None:
            modules["critic"] = policy.critic
        if self.initialized:
            self.broadcast_modules([*modules.values(), runner._wm.representation, runner._wm.sim_context])
            self.bind_method(wm, "_update_from", "world_model")
            self.bind_method(policy, "_train_batch", "actor")
            if "critic" in modules:
                self.bind_method(policy, "_fit_critic", "critic")
            # Explicit instance-only bypass. TORCHDYNAMO_DISABLE does not turn
            # off SuperMimic's independent CapturedUpdate CUDA graph wrapper.
            wm._captured_update = wm._update_from
            policy._captured_update = policy._update_from
        return modules

    def verify(self, modules, updates):
        hashes = {}
        for name, module in modules.items():
            digest = hashlib.sha256()
            for key, value in sorted(module.state_dict().items()):
                if not self.torch.isfinite(value).all():
                    raise RuntimeError(f"训练参数或归一化状态非有限值：{name}.{key}")
                digest.update(key.encode())
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(
                    self.torch.uint8).numpy().tobytes())
            hashes[name] = digest.hexdigest()
        evidence = {"rank": self.topology.rank, "pid": os.getpid(),
                    "gpu_id": self.topology.gpu_ids[self.topology.rank], "updates": dict(updates),
                    # Native single-GPU CUDA graph steps are not instrumented.
                    # Unknown must not be reported as zero successful steps.
                    "optimizer_steps": dict(self.optimizer_steps) if self.initialized else None,
                    "state_hashes": hashes}
        all_ranks = [evidence]
        if self.initialized:
            all_ranks = [None] * self.topology.world_size
            self.dist.all_gather_object(all_ranks, evidence)
            for other in all_ranks:
                if any(other[key] != evidence[key] for key in ("updates", "optimizer_steps", "state_hashes")):
                    raise RuntimeError("不同 GPU 的更新次数、模型参数或归一化统计不一致")
        atomic_json(Path("ranks") / f"rank-{self.topology.rank}" / "result.json", evidence)
        return all_ranks


def run_loop(runner, topology, progress, updates, checkpoint, preview=None):
    """Upstream collection/Ratio/update order, with rank-zero output ownership."""
    from supermimic.runners.supermimic_runner import Ratio
    log_dir = None
    if topology.primary:
        log_dir = runner._create_log_dir()
        runner._init_wandb(log_dir)
    runner.env.reset()
    runner._seed_sim_history()
    cfg = runner.cfg
    should_train = Ratio(runner.env.num_envs / cfg.wm.num_batches / cfg.wm.batch_size
                         / cfg.wm.horizon * cfg.replay_ratio)
    collected, log_count = 0, 0
    start = time.monotonic()
    wm_time, policy_time, losses = 0.0, 0.0, {}
    for step in range(cfg.max_iterations):
        progress.write("collecting", updates)
        collected += runner.step() * topology.world_size
        if preview is not None:
            preview.capture(runner.env, step + 1)
        for _ in range(should_train(step)):
            if runner.replay_buffer.num_transitions < cfg.wm.horizon:
                continue
            # Materialize on every rank; a rank with invalid replay fails its
            # process and torchrun tears down peers, rather than silently skip.
            data = [runner.replay_buffer.sample(cfg.wm.horizon, cfg.wm.batch_size)
                    for _ in range(cfg.wm.num_batches)]
            progress.write("world_model", updates)
            began = time.monotonic()
            losses.update(runner.world_model_learner.update(data))
            updates["world_model"] += 1
            wm_time += time.monotonic() - began
            progress.write("policy", updates)
            began = time.monotonic()
            losses.update(runner.policy_learner.update())
            updates["policy"] += 1
            policy_time += time.monotonic() - began
        progress.iteration = step + 1
        progress.write("training", updates)
        if step % cfg.log_interval == 0:
            if topology.primary:
                runner.logger.log(it=log_count, total_it=cfg.max_iterations // cfg.log_interval,
                                  collect_time=time.monotonic() - start - wm_time - policy_time,
                                  wm_learn_time=wm_time, policy_learn_time=policy_time,
                                  loss_dict=losses, num_collected_steps=collected)
            else:
                # process_env_step runs on every rank and queues episode
                # tensors. Only rank zero logs them; bound the others' lifetime.
                runner.logger.ep_extras.clear()
            log_count += 1
            start, wm_time, policy_time, collected = time.monotonic(), 0.0, 0.0, 0
        if topology.primary and step % cfg.save_interval == 0:
            checkpoint(log_dir)
