"""Durable host tasks. Browser disconnects and manager restarts do not kill workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time

from . import PROTOCOL

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
RECIPE = "supermimic-g1-v1"
SAFE_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}\Z")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + secrets.token_hex(4))
    with temp.open("x") as stream:
        os.chmod(temp, 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def secret(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("凭证文件必须仅服务账户可读写（0600）")
    value = path.read_text().strip()
    if len(value) < 32:
        raise ValueError("服务凭证至少需要 32 个字符")
    return value


def worker_alive(job):
    pid = job.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return b"motion_environment.worker" in cmd and job["id"].encode() in cmd
    except OSError:
        return False


class EnvironmentManager:
    def __init__(self, root, config, *, launcher=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.config_path = Path(config).resolve(strict=True)
        self.config = read_json(self.config_path)
        self.targets = {p["id"]: p for p in self.config["targets"]}
        if not self.targets or any(not SAFE_ID.fullmatch(key) for key in self.targets):
            raise ValueError("目标配置标识无效")
        self.lock = threading.Lock()
        self.launcher = launcher or self._launch
        self.children = []

    def health(self):
        return {"protocol": PROTOCOL, "state": "ready", "targets": len(self.targets)}

    def _folder(self, id):
        if not isinstance(id, str) or not re.fullmatch(r"[a-f0-9]{32}", id):
            raise ValueError("环境任务标识无效")
        folder = self.root / "jobs" / id
        if not (folder / "job.json").is_file():
            raise ValueError("环境任务不存在")
        return folder

    def _job(self, id):
        folder = self._folder(id)
        job = read_json(folder / "job.json")
        # A durable worker survives a manager restart; only stale ownership is interrupted.
        if (job["state"] not in TERMINAL and time.time() - job["updated_at"] > 10
                and not worker_alive(job)):
            # The worker may have committed its terminal result between the first
            # read and the PID check. Never overwrite that result with stale state.
            job = read_json(folder / "job.json")
            if job["state"] in TERMINAL:
                return job
            job.update(state="interrupted", error="安装进程已中断，可重试；已有文件和容器将检查后复用",
                       updated_at=time.time())
            atomic_json(folder / "job.json", job)
        return job

    def _launch(self, job):
        folder = self._folder(job["id"])
        with (folder / "worker.log").open("ab") as stream:
            process = subprocess.Popen(
                [sys.executable, "-m", "motion_environment.worker", "--root", str(self.root),
                 "--config", str(self.config_path), "--job-id", job["id"]],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
        self.children = [p for p in self.children if p.poll() is None] + [process]

    def submit(self, action, args):
        target_id = args.get("target_id")
        if target_id not in self.targets:
            raise ValueError("请选择已配置的目标服务器")
        mode = args.get("mode", "reuse") if action == "install" else "check"
        if mode not in {"reuse", "isolated", "check"} or (action == "install" and mode == "check"):
            raise ValueError("安装方式必须为 reuse 或 isolated")
        if args.get("recipe_id", RECIPE) != RECIPE:
            raise ValueError("不支持此环境版本")
        if mode == "reuse" and not self.targets[target_id].get("existing"):
            raise ValueError("此服务器没有配置可复用的 Motion 服务")
        with self.lock:
            for path in (self.root / "jobs").glob("*/job.json"):
                prior = self._job(path.parent.name)
                if prior["target_id"] == target_id and prior["state"] not in TERMINAL:
                    if prior["mode"] == mode:
                        return {"job": prior, "reused": True}
                    raise ValueError("此服务器已有环境任务运行，请等待或取消后重试")
            now = time.time()
            job = {"id": secrets.token_hex(16), "target_id": target_id, "recipe_id": RECIPE,
                   "mode": mode, "action": action, "state": "queued", "stage": "queued",
                   "created_at": now, "updated_at": now, "pid": None, "checks": [],
                   "error": None}
            atomic_json(self.root / "jobs" / job["id"] / "job.json", job)
            try:
                self.launcher(job)
            except Exception:
                job.update(state="failed", error="无法启动环境任务进程", updated_at=time.time())
                atomic_json(self.root / "jobs" / job["id"] / "job.json", job)
                raise
            return {"job": job}

    def status(self, args):
        if args.get("job_id") is None:
            target = args.get("target_id")
            jobs = [self._job(p.parent.name) for p in (self.root / "jobs").glob("*/job.json")]
            return {"jobs": sorted((j for j in jobs if target is None or j["target_id"] == target),
                                   key=lambda j: j["created_at"], reverse=True)[:50]}
        job = self._job(args["job_id"])
        cursor = args.get("cursor", 0)
        if type(cursor) is not int or cursor < 0:
            raise ValueError("日志游标无效")
        path = self._folder(job["id"]) / "events.jsonl"
        events = []
        if path.exists():
            # Byte cursor avoids rereading a multi-GB dependency build log on each poll.
            size = path.stat().st_size
            if cursor > size:
                raise ValueError("日志游标超出范围")
            with path.open("rb") as stream:
                stream.seek(cursor)
                for _ in range(200):
                    before = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        stream.seek(before)
                        break
                    events.append(json.loads(line))
                cursor = stream.tell()
        return {"job": job, "events": events, "next_cursor": cursor}

    def cancel(self, args):
        job = self._job(args.get("job_id"))
        if job["state"] not in TERMINAL:
            (self._folder(job["id"]) / "cancel").touch(mode=0o600)
        return {"job": job, "cancel_requested": job["state"] not in TERMINAL}

    def connection(self, args):
        job = self._job(args.get("job_id"))
        if job["state"] != "succeeded" or job["action"] != "install":
            raise ValueError("环境安装验证尚未成功")
        private = read_json(self._folder(job["id"]) / "connection.json")
        return {"endpoint": private["endpoint"], "api_key": secret(private["token_file"]),
                "target_id": job["target_id"], "job_id": job["id"]}

    def call(self, args):
        action = args.get("action")
        if action == "targets":
            history = self.status({})["jobs"]
            return {"targets": [{"id": p["id"], "label": p.get("label", p["id"]),
                                  "host": p.get("host"), "gpu_ids": p.get("gpu_ids", []),
                                  "existing_service": {"endpoint": p["existing"]["endpoint"]}
                                  if p.get("existing") else None,
                                  "recipes": [{"id": RECIPE, "label": "SuperMimic · G1 · MuJoCo"}],
                                  "latest_job": next((j for j in history if j["target_id"] == p["id"]), None),
                                  "capabilities": {"check": True, "isolated": True,
                                                   "reuse": bool(p.get("existing"))}}
                                 for p in self.targets.values()]}
        if action in {"check", "install"}:
            return self.submit(action, args)
        if action == "status":
            return self.status(args)
        if action == "cancel":
            return self.cancel(args)
        if action == "connection":
            return self.connection(args)
        raise ValueError("不支持的环境操作")

    def close(self):
        # Workers deliberately outlive this HTTP process. They own their cancellation markers.
        for child in self.children:
            child.poll()
