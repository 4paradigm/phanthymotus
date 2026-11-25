"""GPU inventory and admission for one server; no CUDA context in the API process."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import subprocess
import sys
import threading
import time


class ResourceUnavailable(ValueError):
    """A valid automatic request can run after resources become available."""


def _gpu_query():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 6:
            raise ValueError("GPU inventory returned invalid columns")
        index, uuid, name, total, used, utilization = [v.strip() for v in row]
        rows.append({"id": int(index), "uuid": uuid, "name": name,
                     "memory_total_mib": int(float(total)), "memory_used_mib": int(float(used)),
                     "utilization_percent": int(float(utilization))})
    return rows


def _runtime_query(python):
    # Probe the selected worker interpreter, which can differ from the API's.
    code = """import importlib.util,json
names=('torch','supermimic','mjlab')
r={n:importlib.util.find_spec(n) is not None for n in names}
r['nccl']=False
r['cuda']=False
if r['torch']:
 import torch
 import torch.distributed as d
 r['nccl']=d.is_available() and d.is_nccl_available()
 r['cuda']=torch.cuda.is_available()
print(json.dumps(r))
"""
    env = {**os.environ, "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60, check=True, env=env)
    return json.loads(result.stdout.strip().splitlines()[-1])


class ResourceManager:
    """Report actual memory/load; occupied GPUs may be shared within their allocation.

    MOTION_GPU_IDS optionally narrows the server allocation. CUDA visibility is
    respected when no explicit allocation is supplied. Worker processes use UUIDs
    so CUDA's logical device order cannot select a different physical card.
    """

    def __init__(self, python=None, *, gpu_query=None, runtime_query=None, allowed=None,
                 min_free_mib=4096, cache_seconds=2, background_probe=True,
                 auto_targets=None, auto_free_mib=None, auto_max_utilization=None):
        self.python = python or sys.executable
        self._query = gpu_query or _gpu_query
        self._runtime_query = runtime_query or (lambda: _runtime_query(self.python))
        self._allowed = allowed
        self.min_free_mib = min_free_mib
        self.cache_seconds = cache_seconds
        self._lock = threading.RLock()
        self._sample = None
        self._sample_at = 0
        self._runtime = None
        self._runtime_at = 0
        self._runtime_loading = False
        self._background_probe = background_probe
        self.auto_targets = auto_targets or {
            preset: int(os.environ.get(f"MOTION_AUTO_ENVS_PER_GPU_{preset.upper()}", default))
            for preset, default in (("small", 1024), ("standard", 512))}
        self.auto_free_mib = auto_free_mib or {
            preset: int(os.environ.get(f"MOTION_AUTO_FREE_MIB_{preset.upper()}", default))
            for preset, default in (("small", 8192), ("standard", 32768))}
        self.auto_max_utilization = (int(os.environ.get("MOTION_AUTO_MAX_UTILIZATION", 90))
                                     if auto_max_utilization is None else auto_max_utilization)
        if (any(type(table.get(p)) is not int or table[p] < 1
                for table in (self.auto_targets, self.auto_free_mib) for p in ("small", "standard"))
                or type(self.auto_max_utilization) is not int
                or not 0 <= self.auto_max_utilization <= 100):
            raise ValueError("自动 GPU 调度策略配置无效")

    def _refresh_runtime(self):
        try:
            result = self._runtime_query()
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            result = {}
        with self._lock:
            self._runtime = result
            self._runtime_at = time.monotonic()
            self._runtime_loading = False
            self._sample_at = 0

    def _ensure_runtime(self, now):
        retry_after = 300 if self._runtime else 10
        if not self._runtime_loading and (self._runtime is None or now - self._runtime_at > retry_after):
            self._runtime_loading = True
            if self._background_probe:
                threading.Thread(target=self._refresh_runtime, name="motion-runtime-probe", daemon=True).start()
            else:
                self._refresh_runtime()

    def _allowed_ids(self, rows):
        raw = self._allowed
        if raw is None:
            raw = os.environ.get("MOTION_GPU_IDS")
            if raw is None:
                raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw is None or raw == "all":
            return {row["id"] for row in rows}
        values = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
        allowed = set()
        for value in values:
            value = str(value).strip()
            for row in rows:
                if value in {str(row["id"]), row["uuid"]}:
                    allowed.add(row["id"])
        return allowed

    def inventory(self, *, refresh=False):
        with self._lock:
            now = time.monotonic()
            if not refresh and self._sample and now - self._sample_at < self.cache_seconds:
                return self._sample
            error = ""
            try:
                rows = self._query()
                if (len({r["id"] for r in rows}) != len(rows)
                        or len({r["uuid"] for r in rows}) != len(rows)):
                    raise ValueError("duplicate GPU identity")
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
                rows = []
                error = "无法读取 NVIDIA GPU，请检查服务器驱动与容器显卡分配"
            self._ensure_runtime(now)
            runtime = self._runtime or {}
            checking = self._runtime_loading and not runtime
            runtime_ready = all(runtime.get(n) for n in ("torch", "supermimic", "mjlab", "cuda"))
            allowed = self._allowed_ids(rows)
            gpus = []
            for row in rows:
                free = max(0, row["memory_total_mib"] - row["memory_used_mib"])
                reason = ("未分配给训练服务" if row["id"] not in allowed else
                          "正在检测训练运行环境" if checking else
                          "训练运行环境不可用" if not runtime_ready else
                          f"剩余显存不足 {self.min_free_mib // 1024} GiB" if free < self.min_free_mib else "")
                gpus.append({**row, "memory_free_mib": free, "selectable": not reason,
                             "reason": reason, "shared": row["memory_used_mib"] > 1024})
            selectable = sorted((g for g in gpus if g["selectable"]),
                                key=lambda g: (g["utilization_percent"], g["memory_used_mib"], g["id"]))
            reason = error or ("正在检测训练运行环境，请稍后刷新" if checking else
                               "训练 Python 的 CUDA、PyTorch、SuperMimic 或 mjlab 不可用" if not runtime_ready else
                               "没有显存余量足够的已分配 GPU" if not selectable else "")
            self._sample = {"sampled_at": time.time(), "gpus": gpus,
                            "training": {"available": bool(selectable),
                                         "runtime_ready": runtime_ready,
                                         "inventory_available": not error,
                                         "multi_gpu_supported": runtime_ready and bool(runtime.get("nccl")),
                                         "checking": checking,
                                         "max_gpus": min(8, len(allowed)), "max_num_envs": 8192,
                                         "max_iterations": 1_000_000, "reason": reason,
                                         "default_gpu_ids": [selectable[0]["id"]] if selectable else []}}
            self._sample_at = now
            return self._sample

    def auto_plan(self, num_envs, preset="small", *, refresh=True):
        """Choose count and physical devices using an explicit admission policy.

        These are configurable headroom and parallelism targets, not measured
        peak-memory estimates or a promise that more GPUs will run faster.
        """
        if type(num_envs) is not int or not 1 <= num_envs <= 8192:
            raise ValueError("机器人实例总数须为 1–8192 的整数")
        if not isinstance(preset, str) or preset not in {"small", "standard"}:
            raise ValueError("不支持的训练预设")
        inventory = self.inventory(refresh=refresh)
        training = inventory["training"]
        if training["checking"]:
            raise ResourceUnavailable("等待训练运行环境检测完成")
        if not training["inventory_available"] or not training["runtime_ready"]:
            raise ValueError(training["reason"] or "训练运行环境不可用")
        allowed = self._allowed_ids(inventory["gpus"])
        rows = [gpu for gpu in inventory["gpus"] if gpu["id"] in allowed]
        if not rows:
            raise ValueError("训练服务尚未分配 GPU")
        target = self.auto_targets[preset]
        headroom = max(self.min_free_mib, self.auto_free_mib[preset])
        limit = min(8, len(rows)) if training["multi_gpu_supported"] else 1
        desired = max(1, min(limit, math.ceil(num_envs / target)))
        counts = sorted((count for count in range(1, limit + 1) if num_envs % count == 0),
                        key=lambda count: (abs(count - desired), count))
        possible = False
        for count in counts:
            each = num_envs // count
            # Exceeding the per-GPU target needs proportionately more free
            # memory under this policy; never silently increase the env count.
            required = math.ceil(headroom * max(1, each / target))
            if sum(gpu["memory_total_mib"] >= required for gpu in rows) < count:
                continue
            possible = True
            candidates = sorted((gpu for gpu in rows
                                 if gpu["memory_free_mib"] >= required
                                 and gpu["utilization_percent"] <= self.auto_max_utilization),
                                key=lambda gpu: (gpu["utilization_percent"],
                                                 -gpu["memory_free_mib"], gpu["id"]))
            if len(candidates) < count:
                continue
            chosen = candidates[:count]
            return {"gpu_mode": "auto", "gpu_ids": [gpu["id"] for gpu in chosen],
                    "gpu_uuids": [gpu["uuid"] for gpu in chosen], "world_size": count,
                    "num_envs": num_envs, "envs_per_rank": each,
                    "selection_reason": f"自动分配 {count} 张 GPU，每卡 {each} 个机器人实例",
                    "policy": {"name": "headroom-v1", "target_envs_per_gpu": target,
                               "min_free_mib_per_gpu": required,
                               "max_utilization_percent": self.auto_max_utilization,
                               "sampled_at": inventory["sampled_at"]}}
        if not possible:
            raise ValueError("当前硬件不满足自动调度的显存保留策略，请减少机器人实例总数或使用手动配置")
        raise ResourceUnavailable("等待 GPU 资源：可用显存或负载暂不满足自动调度条件")

    def training_plan(self, args):
        """Read-only UI preview; reserving devices is the job launcher's job."""
        ids = args.get("gpu_ids")
        mode = args.get("gpu_mode", "manual" if ids is not None else "auto")
        if not isinstance(mode, str) or mode not in {"auto", "manual"}:
            raise ValueError("GPU 调度模式须为 auto 或 manual")
        envs, preset = args.get("num_envs", 64), args.get("preset", "small")
        if not isinstance(preset, str) or preset not in {"small", "standard"}:
            raise ValueError("不支持的训练预设")
        if mode == "auto" and ids is not None:
            raise ValueError("自动调度不接受指定 GPU，请切换为手动配置")
        if mode == "manual" and ids is None:
            raise ValueError("请选择 GPU")
        try:
            plan = self.auto_plan(envs, preset) if mode == "auto" else self.plan(ids, envs)
        except ResourceUnavailable as exc:
            return {"gpu_mode": mode, "ready": False, "queueable": True, "reason": str(exc),
                    "gpu_ids": [], "world_size": None, "num_envs": envs, "envs_per_rank": None}
        return {key: value for key, value in {
            **plan, "gpu_mode": mode, "ready": True, "queueable": True,
            "reason": plan.get("selection_reason", "")}.items() if key != "gpu_uuids"}

    def plan(self, gpu_ids, num_envs, *, refresh=True):
        inventory = self.inventory(refresh=refresh)
        training = inventory["training"]
        if not training["available"]:
            raise ValueError(training["reason"])
        if gpu_ids is None:
            gpu_ids = training["default_gpu_ids"]
        if (not isinstance(gpu_ids, list) or not 1 <= len(gpu_ids) <= 8
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in gpu_ids)
                or len(set(gpu_ids)) != len(gpu_ids)):
            raise ValueError("请选择 1–8 张不同的 GPU")
        if len(gpu_ids) > 1 and not training["multi_gpu_supported"]:
            raise ValueError("当前训练环境未提供 NCCL 多 GPU 通信支持")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or not 1 <= num_envs <= 8192:
            raise ValueError("机器人实例总数须为 1–8192 的整数")
        if num_envs % len(gpu_ids):
            raise ValueError("机器人实例总数须能被 GPU 数量整除")
        by_id = {g["id"]: g for g in inventory["gpus"]}
        for i in gpu_ids:
            if i not in by_id or not by_id[i]["selectable"]:
                raise ValueError(f"GPU {i} 不可用：{by_id.get(i, {}).get('reason', '服务器未发现此显卡')}")
        return {"gpu_ids": list(gpu_ids), "gpu_uuids": [by_id[i]["uuid"] for i in gpu_ids],
                "world_size": len(gpu_ids), "num_envs": num_envs,
                "envs_per_rank": num_envs // len(gpu_ids)}
