"""Durable single-node training queue with one or several GPUs per job."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid

from .contracts import identifier, integer
from .store import atomic_json, read_json, file_sha256
from .resources import ResourceManager, ResourceUnavailable

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class JobManager:
    def __init__(self, store, python=None, worker_module="motion_service.worker", resources=None,
                 resource_retry_seconds=3.0):
        self.store = store
        self.python = python or sys.executable
        self.worker_module = worker_module
        self.resources = resources or ResourceManager(self.python)
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        self._waiting = {}
        self._wake = threading.Event()
        self.resource_retry_seconds = resource_retry_seconds
        self.processes = {}
        self.closing = False
        for p in (store.root / "jobs").glob("*/job.json"):
            job = read_json(p)
            if job["state"] not in TERMINAL:
                job.update(state="interrupted", finished_at=time.time(),
                           message="服务已重启；此任务需要重新发起")
                atomic_json(p, job)
        self.thread = threading.Thread(target=self._loop, daemon=True, name="motion-jobs")
        self.thread.start()

    def _path(self, id):
        return self.store.root / "jobs" / identifier(id)

    def _get(self, id):
        return read_json(self._path(id) / "job.json")

    def list(self):
        return [self._with_progress(job) for job in
                sorted(self._records(), key=lambda j: j["created_at"], reverse=True)[:100]]

    def _with_progress(self, job):
        job = dict(job)
        config = job.get("config") or {}
        world_size = config.get("world_size", 1) or 0
        ranks = []
        for rank in range(world_size):
            path = self._path(job["id"]) / "ranks" / f"rank-{rank}" / "progress.json"
            try:
                value = read_json(path)
                if not isinstance(value, dict) or value.get("rank", rank) != rank:
                    continue
                ranks.append({**value, "rank": rank})
            except (OSError, ValueError, TypeError):
                continue
        if ranks:
            total = config.get("max_iterations", 0)
            iteration = min(max(0, int(r.get("iteration", 0))) for r in ranks) if len(ranks) == world_size else 0
            elapsed = max(max(0, float(r.get("elapsed_s", 0))) for r in ranks)
            progress = {"iteration": min(iteration, total), "total_iterations": total,
                        "elapsed_s": elapsed, "phase": min(ranks, key=lambda r: r.get("iteration", 0)).get("phase", "initializing"),
                        "ranks_ready": len(ranks), "world_size": world_size,
                        "world_model_updates": min(int(r.get("world_model_updates", 0)) for r in ranks),
                        "policy_updates": min(int(r.get("policy_updates", 0)) for r in ranks)}
            if 0 < iteration < total and elapsed > 0 and len(ranks) == world_size:
                progress["eta_s"] = elapsed * (total - iteration) / iteration
            job.update(progress=progress, ranks=ranks)
        return job

    def _records(self):
        return [read_json(p) for p in (self.store.root / "jobs").glob("*/job.json")]

    def inspect(self, id):
        job = self._with_progress(self._get(id))
        path = self._path(id) / "worker.log"
        log = ""
        if path.exists():
            with path.open("rb") as f:
                f.seek(max(0, path.stat().st_size - 24000))
                log = f.read(24000).decode("utf-8", "replace")
        result = {"job": job, "log": log, "artifacts": job.get("artifacts", [])}
        report = self._path(id) / "evaluation.json"
        if report.exists() and job["state"] in TERMINAL:
            result["evaluation"] = read_json(report)
            rollout = self._path(id) / "rollout.json"
            if rollout.exists():
                result["rollout"] = read_json(rollout)
        return result

    def submit(self, kind, args):
        profile = self.store.profile(args.get("profile_id"))
        dataset, path = self.store.dataset(args.get("dataset_id"), profile["id"])
        if not profile.get("task_id"):
            raise ValueError("此机器人配置没有已适配的 SuperMimic 训练任务")
        request = {"kind": kind, "profile": profile, "dataset": dataset, "motion_file": str(path)}
        waiting_reason = None
        if kind == "train":
            preset = args.get("preset", "small")
            if preset not in {"small", "standard"}:
                raise ValueError("不支持的训练预设")
            request.update(num_envs=integer(args.get("num_envs", 64), 1, 8192, "并行环境数"),
                           max_iterations=integer(args.get("max_iterations", 1000), 1, 1_000_000, "迭代次数"),
                           preset=preset)
            mode = args.get("gpu_mode", "manual" if args.get("gpu_ids") is not None else "auto")
            if mode not in ("auto", "manual"):
                raise ValueError("GPU 调度方式必须是 auto 或 manual")
            if mode == "auto":
                if args.get("gpu_ids") is not None:
                    raise ValueError("自动调度时不能同时指定 GPU 编号")
                try:
                    request.update(self.resources.auto_plan(request["num_envs"], preset=preset))
                except ResourceUnavailable as exc:
                    request.update(gpu_ids=[], gpu_uuids=[], world_size=None, envs_per_rank=None)
                    waiting_reason = str(exc)[:1000]
            else:
                if args.get("gpu_ids") is None:
                    raise ValueError("手动调度时请选择 GPU")
                request.update(self.resources.plan(args["gpu_ids"], request["num_envs"]))
            request["gpu_mode"] = mode
        elif kind == "evaluate":
            model, _ = self.store.verified_model(args.get("model_id"))
            if model["profile_hash"] != profile["profile_hash"]:
                raise ValueError("评测模型与机器人配置不匹配")
            request.update(model=model, model_dir=str(self._path(model["job_id"])),
                           max_steps=integer(args.get("max_steps", min(dataset["frames"], 10000)),
                                             1, 10000, "评测步数"))
            request.update(self.resources.plan(args.get("gpu_ids"), 1))
        else:
            raise ValueError("未知作业类型")
        with self.lock:
            if self.closing:
                raise ValueError("服务正在停止")
            if sum(j["state"] not in TERMINAL for j in self._records()) >= 16:
                raise ValueError("作业队列已满")
            id = uuid.uuid4().hex
            job = {"id": id, "kind": kind, "state": "queued", "profile_id": profile["id"],
                   "dataset_id": dataset["id"], "profile_hash": profile["profile_hash"],
                   "created_at": time.time()}
            job["config"] = self._config(request)
            if request.get("gpu_mode") == "auto":
                job["allocation_pending"] = True
            if waiting_reason is not None:
                job.update(waiting_for_resources=True, message=waiting_reason)
            if kind == "evaluate":
                job["model_id"] = request["model"]["id"]
            atomic_json(self._path(id) / "request.json", request)
            atomic_json(self._path(id) / "job.json", job)
            self.queue.put(id)
            self._wake.set()
            return {"job": job}

    @staticmethod
    def _config(request):
        return {key: request[key] for key in (
            "gpu_mode", "gpu_ids", "world_size", "num_envs", "envs_per_rank",
            "max_iterations", "preset", "selection_reason", "policy") if key in request}

    def cancel(self, id):
        with self.lock:
            job = self._get(id)
            if job["state"] in TERMINAL:
                return {"job": job}
            process = self.processes.get(id)
            job["state"] = "cancelling" if process else "cancelled"
            if not process:
                job["finished_at"] = time.time()
                job["message"] = "任务已取消"
            job.pop("next_retry_at", None)
            job["allocation_pending"] = False
            job["waiting_for_resources"] = False
            self._waiting.pop(id, None)
            atomic_json(self._path(id) / "job.json", job)
            if process:
                self._terminate_workers(process, self._path(id), signal.SIGTERM)
            self._wake.set()
            return {"job": job}

    @staticmethod
    def _signal(process, sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            else:
                process.terminate() if sig == signal.SIGTERM else process.kill()
        except ProcessLookupError:
            pass

    @staticmethod
    def _rank_pids(folder):
        """Linux torchrun children can own separate process groups.

        Match only the exact private job-directory marker inherited by this
        launcher. Do not rely on stale PID files or terminate other GPU users.
        """
        proc = Path("/proc")
        if not proc.is_dir():
            return []
        marker = b"MOTION_JOB_DIR=" + os.fsencode(str(folder.resolve()))
        result = []
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                if marker not in (entry / "environ").read_bytes().split(b"\0"):
                    continue
                if (entry / "stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    continue
                result.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue
        return result

    def _terminate_workers(self, process, folder, sig):
        self._signal(process, sig)
        for pid in self._rank_pids(folder):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def _finish_children(self, process, folder):
        remaining = self._rank_pids(folder)
        if not remaining:
            return
        self._terminate_workers(process, folder, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while self._rank_pids(folder) and time.monotonic() < deadline:
            time.sleep(.1)
        if self._rank_pids(folder):
            self._terminate_workers(process, folder, signal.SIGKILL)
            deadline = time.monotonic() + 2
            while self._rank_pids(folder) and time.monotonic() < deadline:
                time.sleep(.1)
        if self._rank_pids(folder):
            raise RuntimeError("训练进程尚未完全退出，请检查服务器任务日志")

    def _command(self, request, folder):
        if request["kind"] == "train" and request.get("world_size", 1) > 1:
            return [self.python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc-per-node={request['world_size']}", "--max-restarts=0", "--monitor-interval=1",
                    "--log-dir", str(folder / "launcher"), "--redirects=3", "--tee=3",
                    "--module", self.worker_module, str(folder / "request.json")]
        return [self.python, "-m", self.worker_module, str(folder / "request.json")]

    def _loop(self):
        while True:
            # Deferred auto jobs do not hold the queue head or a manager lock.
            # Submitting, cancelling or closing wakes the scheduler immediately.
            self._wake.clear()
            with self.lock:
                if self.closing:
                    return
                now = time.monotonic()
                for waiting_id, deadline in list(self._waiting.items()):
                    if deadline <= now:
                        self.queue.put(waiting_id)
                        del self._waiting[waiting_id]
                delay = min((max(0, deadline - now) for deadline in self._waiting.values()), default=None)
            try:
                id = self.queue.get_nowait()
            except queue.Empty:
                self._wake.wait(delay)
                continue
            try:
                retry = self._run(id)
                if retry:
                    with self.lock:
                        if not self.closing and self._get(id)["state"] == "queued":
                            self._waiting[id] = time.monotonic() + self.resource_retry_seconds
            except Exception as exc:
                with self.lock:
                    job = self._get(id)
                    if job["state"] not in TERMINAL:
                        job.update(state="failed", message=str(exc)[:1000], finished_at=time.time(),
                                   allocation_pending=False, waiting_for_resources=False)
                        job.pop("next_retry_at", None)
                        atomic_json(self._path(id) / "job.json", job)
            finally:
                self.queue.task_done()

    def _run(self, id):
        folder = self._path(id)
        with self.lock:
            job = self._get(id)
            if self.closing or job["state"] != "queued":
                return
            request = read_json(folder / "request.json")
        auto = request.get("gpu_mode") == "auto"
        try:
            # Resource probes can be slow. Cancellation must remain available
            # while the fresh launch-time allocation is being computed.
            plan = (self.resources.auto_plan(request["num_envs"], preset=request["preset"])
                    if auto else self.resources.plan(request["gpu_ids"], request["num_envs"]))
        except ResourceUnavailable as exc:
            if not auto:
                raise
            with self.lock:
                job = self._get(id)
                if self.closing or job["state"] != "queued":
                    return
                request.update(gpu_ids=[], gpu_uuids=[], world_size=None, envs_per_rank=None)
                request.pop("selection_reason", None)
                request.pop("policy", None)
                job.update(config=self._config(request), allocation_pending=True,
                           waiting_for_resources=True, message=str(exc)[:1000],
                           next_retry_at=time.time() + self.resource_retry_seconds)
                atomic_json(folder / "request.json", request)
                atomic_json(folder / "job.json", job)
            return True
        if not auto and plan["gpu_uuids"] != request["gpu_uuids"]:
            raise ValueError("GPU 编号映射已改变，请刷新资源后重新提交任务")
        with (folder / "worker.log").open("ab", buffering=0) as log:
            with self.lock:
                job = self._get(id)
                if self.closing or job["state"] != "queued":
                    return
                if auto:
                    request.update(plan, gpu_mode="auto")
                    atomic_json(folder / "request.json", request)
                job.update(config=self._config(request), allocation_pending=False,
                           waiting_for_resources=False)
                job.pop("next_retry_at", None)
                job.pop("message", None)
                atomic_json(folder / "job.json", job)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
                env["PYTHONUNBUFFERED"] = "1"
                env["MOTION_PARENT_PID"] = str(os.getpid())
                env["MOTION_JOB_DIR"] = str(folder.resolve())
                env["CUDA_VISIBLE_DEVICES"] = ",".join(request["gpu_uuids"])
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["OMP_NUM_THREADS"] = "1"
                env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
                env.pop("MOTION_API_TOKEN", None)
                # The worker owns local checkpoints. W&B is configured only on the server.
                env.setdefault("WANDB_MODE", "offline")
                process = subprocess.Popen(self._command(request, folder),
                                           cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                self.processes[id] = process
                job.update(state="running", started_at=time.time())
                atomic_json(folder / "job.json", job)
            cancelling_at = None
            while process.poll() is None:
                with self.lock:
                    cancelling = self._get(id)["state"] == "cancelling"
                if cancelling:
                    cancelling_at = cancelling_at or time.monotonic()
                    if time.monotonic() - cancelling_at > 5:
                        self._terminate_workers(process, folder, signal.SIGKILL)
                time.sleep(0.05)
            try:
                self._finish_children(process, folder)
            except Exception:
                with self.lock:
                    self.processes.pop(id, None)
                raise
            with self.lock:
                self.processes.pop(id, None)
                job = self._get(id)
                if job["state"] == "cancelling":
                    job.update(state="cancelled", message="任务已取消；仅保留已写入的产物")
                elif process.returncode != 0:
                    job.update(state="failed", message=self._failure_message(job, folder, process.returncode))
                else:
                    self._collect(job, folder)
                    job["state"] = "succeeded"
                job.update(exit_code=process.returncode, finished_at=time.time())
                atomic_json(folder / "job.json", job)

    def _failure_message(self, job, folder, code):
        ranks = self._with_progress(job).get("ranks", [])
        errors = [str(r.get("error", "")) for r in ranks if r.get("error")]
        if not errors:
            try:
                with (folder / "worker.log").open("rb") as stream:
                    stream.seek(max(0, (folder / "worker.log").stat().st_size - 12000))
                    tail = stream.read(12000).decode("utf-8", "replace")
                errors = [tail] if "out of memory" in tail.lower() else []
            except OSError:
                pass
        message = errors[0] if errors else f"训练进程退出（代码 {code}），请查看任务日志"
        if "out of memory" in message.lower():
            return "GPU 显存不足：请减少机器人实例总数、选择小规模预设或增加 GPU 数量"
        return message[-1000:]

    def _collect(self, job, folder):
        result = read_json(folder / "result.json")
        if job["kind"] == "train":
            world_size = job.get("config", {}).get("world_size", 1)
            if world_size > 1:
                distributed = result.get("distributed") or {}
                ranks = distributed.get("ranks") or []
                if (distributed.get("world_size") != world_size or distributed.get("synchronized") is not True
                        or len(ranks) != world_size
                        or {rank.get("rank") for rank in ranks} != set(range(world_size))):
                    raise ValueError("多 GPU 训练缺少完整的同步验证结果")
                reference = ranks[0].get("state_hashes")
                for rank in ranks:
                    if (not reference or rank.get("state_hashes") != reference
                            or not all(rank.get("updates", {}).get(k, 0) > 0 for k in ("world_model", "policy"))
                            or rank.get("gpu_id") != job["config"]["gpu_ids"][rank["rank"]]
                            or read_json(folder / "ranks" / f"rank-{rank['rank']}" / "result.json") != rank):
                        raise ValueError("多 GPU 更新、参数一致性或进程产物校验失败")
            # Only worker-created local results can become loadable models.
            for key in ("actor", "world_model", "onnx"):
                if result.get(key):
                    path = (folder / result[key]).resolve(strict=True)
                    if not path.is_relative_to(folder.resolve()) or not path.is_file():
                        raise ValueError("训练产物路径无效")
            if not result.get("actor") or not result.get("world_model"):
                raise ValueError("训练结束但没有完整模型产物")
            artifacts = [result[key] for key in ("actor", "world_model", "onnx") if result.get(key)] + ["run.json"]
            hashes = {relative: file_sha256(folder / relative) for relative in artifacts}
            model = {"id": job["id"], "job_id": job["id"], "name": f"SuperMimic {job['id'][:8]}",
                     "profile_id": job["profile_id"], "profile_hash": job["profile_hash"],
                     "dataset_id": job["dataset_id"], "created_at": time.time(),
                     "qualification": "unverified", **result, "artifact_sha256": hashes}
            atomic_json(self.store.root / "models" / (model["id"] + ".json"), model)
            job["model_id"] = model["id"]
            job["artifacts"] = [v for k, v in result.items() if k in {"actor", "world_model", "onnx"} and v]
        else:
            evaluation = read_json(folder / "evaluation.json")
            model = self.store.get("models", job["model_id"])
            model.update(evaluation_job_id=job["id"], qualification="evaluated",
                         evaluation_summary=evaluation,
                         evaluation_artifact_sha256={name: file_sha256(folder / name)
                                                     for name in ("evaluation.json", "rollout.json")})
            atomic_json(self.store.root / "models" / (model["id"] + ".json"), model)
            job["artifacts"] = ["evaluation.json", "rollout.json"]

    def close(self):
        with self.lock:
            self.closing = True
            for job in self._records():
                if job["state"] not in TERMINAL:
                    self.cancel(job["id"])
            self._waiting.clear()
            self._wake.set()
        self.thread.join(timeout=10)
