"""Reviewed local Docker operations. No user commands, images, paths or mounts."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import urllib.parse
import zipfile
from email.parser import Parser

from .manager import RECIPE, atomic_json, read_json, secret

OWNER = "io.phanthymotus.motion-environment"

PROBE_CLEANUP = r"""
import os,signal,sys,time
from pathlib import Path
job=sys.argv[1]
path=Path('/tmp/phanthymotus-environment-probe-'+job+'.pid')
if path.exists():
    pid=int(path.read_text())
    proc=Path('/proc')/str(pid)/'cmdline'
    try: cmd=proc.read_bytes().split(b'\0')
    except FileNotFoundError: cmd=[]
    if pid>1 and job.encode() in cmd and any(b'probe' in x and x.endswith(b'.py') for x in cmd):
        try:
            grouped=os.getpgid(pid)==pid
            kill=os.killpg if grouped else os.kill
            kill(pid,signal.SIGTERM)
            for _ in range(50):
                if not proc.exists(): break
                time.sleep(.1)
            else: kill(pid,signal.SIGKILL)
        except ProcessLookupError: pass
    path.unlink(missing_ok=True)
Path('/tmp/phanthymotus-environment-probe-'+job+'.npz').unlink(missing_ok=True)
"""


class Cancelled(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Motion 地址发生重定向，请配置最终私网地址；认证信息未转发")


class Task:
    def __init__(self, root, job, target, config):
        self.root = Path(root)
        self.job = job
        self.folder = self.root / "jobs" / job["id"]
        self.target = target
        self.config = config
        self.redactions = []
        self.stage("starting", "任务进程已启动", state="running", pid=os.getpid())

    def cancelled(self):
        if (self.folder / "cancel").exists():
            raise Cancelled("任务已取消；已下载镜像及持久数据保留")

    def log(self, message):
        value = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(message))
        for item in self.redactions:
            value = value.replace(item, "[redacted]")
        value = re.sub(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[redacted]@", value)
        value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9_.\-]+", r"\1[redacted]", value)
        event = {"time": time.time(), "stage": self.job["stage"], "message": value[:4096]}
        with (self.folder / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def stage(self, stage, message=None, **changes):
        self.job.update(stage=stage, updated_at=time.time(), **changes)
        atomic_json(self.folder / "job.json", self.job)
        if message:
            self.log(message)

    def run(self, argv, *, timeout=120, capture=False, env=None):
        self.cancelled()
        started = time.monotonic()
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True, env=env)
        self.job["command_pid"] = process.pid
        atomic_json(self.folder / "job.json", self.job)
        output = bytearray()
        pending = bytearray()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                self.cancelled()
                if time.monotonic() - started > timeout:
                    raise TimeoutError("此步骤超时，可检查日志后重试")
                for key, _ in selector.select(.2):
                    part = os.read(key.fileobj.fileno(), 4096)
                    if not part:
                        selector.unregister(key.fileobj)
                        continue
                    if capture:
                        output.extend(part)
                        if len(output) > 4 * 1024 * 1024:
                            raise RuntimeError("命令输出超过限制")
                    pending.extend(part.replace(b"\r", b"\n"))
                    while b"\n" in pending or len(pending) > 4096:
                        if b"\n" in pending:
                            line, _, pending = pending.partition(b"\n")
                        else:
                            line, pending = pending[:4096], pending[4096:]
                        self.log(line.decode(errors="replace"))
            if pending:
                self.log(pending.decode(errors="replace"))
            code = process.wait(timeout=5)
            if code:
                raise RuntimeError(f"步骤执行失败（退出码 {code}），请查看本步骤日志")
            return output.decode(errors="replace").strip()
        finally:
            selector.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            process.stdout.close()
            self.job.pop("command_pid", None)
            atomic_json(self.folder / "job.json", self.job)

    def check(self):
        self.stage("checking", "检查 Docker、GPU 和磁盘，不修改主机驱动及现有容器")
        checks = []
        for name, command in (
            ("docker", ["docker", "version", "--format", "{{.Server.Version}}"]),
            ("platform", ["docker", "info", "--format", "{{.OSType}} {{.Architecture}}"]),
            ("gpu_runtime", ["docker", "info", "--format", "{{json .Runtimes}}"]),
            ("gpus", ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]),
        ):
            try:
                detail = self.run(command, capture=True, timeout=30)
                if name == "platform" and detail not in {"linux x86_64", "linux amd64"}:
                    raise RuntimeError("当前环境安装包仅支持 Linux x86_64")
                if name == "gpu_runtime" and "nvidia" not in detail:
                    raise RuntimeError("Docker 未配置 NVIDIA runtime；需要管理员准备主机环境")
                if name == "gpus" and not detail:
                    raise RuntimeError("没有检测到 NVIDIA GPU")
                checks.append({"name": name, "state": "passed", "detail": detail})
            except Cancelled:
                raise
            except (OSError, RuntimeError, TimeoutError) as exc:
                checks.append({"name": name, "state": "failed", "detail": str(exc)})
                self.stage("checking", checks=checks)
                raise
            self.stage("checking", checks=checks)
        free = shutil.disk_usage(self.root).free
        checks.append({"name": "workspace_disk", "state": "passed" if free >= 30 * 1024**3 else "warning",
                       "detail": f"环境任务目录剩余 {free / 1024**3:.1f} GiB；Docker 数据盘空间由实际构建检查"})
        existing = self.target.get("existing")
        if existing:
            try:
                health = self.http(existing, "health")
                checks.append({"name": "existing_service", "state": "passed", "detail": health})
            except Exception as exc:
                checks.append({"name": "existing_service", "state": "warning", "detail": str(exc)})
        self.stage("checking", checks=checks)

    def http(self, connection, action, **args):
        token = secret(connection["token_file"])
        self.redactions.append(token)
        body = json.dumps({"action": action, **args}).encode()
        request = urllib.request.Request(connection["endpoint"].rstrip("/") + "/v1/call", body,
                                         {"Content-Type": "application/json", "Authorization": "Bearer " + token})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            with opener.open(request, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read(8192)).get("message", "服务请求失败")
            except Exception:
                message = "服务请求失败"
            raise RuntimeError(f"Motion HTTP {exc.code}: {message}") from None

    def verify(self, connection, *, finalize=True):
        self.stage("verifying", "验证服务、工作台控制权、G1 物理步进与渲染")
        self.cancelled()
        health = self.http(connection, "health")
        if health.get("protocol") != "motus.motion/1" or not all(
                health.get("runtime", {}).get(name) is True
                for name in ("mujoco", "supermimic", "mjlab", "onnxruntime")):
            raise RuntimeError("Motion 服务协议或运行依赖不完整")
        catalog = self.http(connection, "catalog")
        profiles = catalog.get("profiles", [])
        profile = next((p for p in profiles if p.get("robot_type") == "unitree_g1"), None)
        if not profile:
            raise RuntimeError("服务缺少已绑定 G1 机器人配置")
        self.cancelled()
        view = "environment-" + self.job["id"]
        session = self.http(connection, "simulation_create", profile_id=profile["id"], control_mode="pd_hold",
                            view_id=view, request_id=self.job["id"])
        sid = session["session"]["id"]
        control = session.get("control") or {}
        owned = {"session_id": sid}
        if type(control.get("epoch")) is int:
            owned.update(view_id=view, control_epoch=control["epoch"])
        try:
            self.cancelled()
            if (control.get("owner_view_id") != view or control.get("is_controller") is not True
                    or type(control.get("epoch")) is not int):
                raise RuntimeError("Motion 服务缺少新版工作台会话控制权接口，请更新完整源码快照")
            observer = self.http(connection, "simulation_inspect", session_id=sid,
                                 view_id=view + "-observer", detail="compact").get("control") or {}
            if (observer.get("is_controller") is not False or observer.get("owner_view_id") != view
                    or observer.get("epoch") != control["epoch"]):
                raise RuntimeError("Motion 服务没有正确隔离工作台只读观察者")
            self.http(connection, "simulation_heartbeat", **owned)
            stepped = self.http(connection, "simulation_step", **owned, steps=1)
            image = self.http(connection, "simulation_render", session_id=sid,
                              view_id=view + "-observer", width=320, height=240,
                              jpeg_quality=70, detail="compact")["image"]
            decoded = base64.b64decode(image["data"], validate=True)
            if image["mime"] != "image/jpeg" or not decoded.startswith(b"\xff\xd8"):
                raise RuntimeError("没有收到真实渲染 JPEG")
            (self.folder / "verification.jpg").write_bytes(decoded)
            self.job["verification"] = {"health": health, "profile_id": profile["id"],
                                        "simulation_time": stepped["snapshot"]["time"],
                                        "view_lease": {"control_epoch": control["epoch"],
                                                       "observer_read_only": True, "heartbeat": True},
                                        "render": {"mime": image["mime"], "bytes": len(decoded),
                                                   "sha256": hashlib.sha256(decoded).hexdigest()}}
        finally:
            self.http(connection, "simulation_close", **owned)
        self.cancelled()
        if finalize:
            atomic_json(self.folder / "connection.json", connection)
            self.stage("complete", "环境验证通过，可连接到工作台", state="succeeded",
                       endpoint=connection["endpoint"], credential_ref=self.job["id"])

    def verify_existing_gpu(self):
        connection = self.target["existing"]
        container = connection.get("container")
        python = connection.get("python")
        if not container or not python:
            raise RuntimeError("管理员尚未配置已有环境的真实 GPU 计算验证入口")
        self.stage("verifying_gpu", "检查已有环境的实际 CUDA 和 Warp 计算，不启动训练")
        destination = "/tmp/phanthymotus-env-probe-" + self.job["id"] + ".py"
        probe = Path(__file__).parent / "recipes" / "runtime_probe.py"
        self.run(["docker", "cp", str(probe), container + ":" + destination])
        try:
            gpu = str(self.target.get("render_gpu", self.target["gpu_ids"][0]))
            self.probe(container, python, destination, gpu)
            self.job["gpu_verification"] = {"cuda_compute": True, "warp_compute": True,
                                            "gpu_id": int(gpu)}
        finally:
            # Fixed owned temporary file, including when cancellation is requested.
            # docker cp creates this fixed temporary file as root; use the same
            # owner to remove it on containers whose default user is unprivileged.
            subprocess.run(["docker", "exec", "--user", "0", container, "rm", "-f", destination],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

    def probe(self, container, python, script, gpu, *, arguments=(), timeout=180):
        try:
            self.run(["docker", "exec", "--env", "CUDA_VISIBLE_DEVICES=" + str(gpu), container,
                      "timeout", "--signal=TERM", "--kill-after=5s", f"{timeout}s", python, script,
                      "--job-id", self.job["id"], *arguments], timeout=timeout + 10)
        finally:
            # Stopping docker exec's client alone does not stop its container process.
            # The fixed probe identifies itself before imports; only that PID/group is killed.
            cleanup = subprocess.run(["docker", "exec", container, python, "-c", PROBE_CLEANUP,
                                      self.job["id"]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     timeout=15)
            if cleanup.returncode:
                self.log(f"GPU 验证进程清理失败；容器内 timeout 仍限制验证进程最长运行 {timeout + 5} 秒")
                raise RuntimeError("GPU 验证进程未确认清理，请检查环境任务日志")

    def verify_gpu_environment(self, container, gpu):
        self.stage("verifying_gpu", "创建官方 G1 单实例并验证真实 GPU 物理，不启动训练")
        destination = "/tmp/phanthymotus-env-full-probe-" + self.job["id"] + ".py"
        source = Path(__file__).parent / "recipes" / "full_runtime_probe.py"
        self.run(["docker", "cp", str(source), container + ":" + destination])
        try:
            self.probe(container, "python", destination, gpu,
                       arguments=("--profiles", "/profiles/profiles.json"), timeout=600)
            self.job["gpu_verification"].update(mjlab_gpu_physics=True, diagnostic_steps=2)
        finally:
            subprocess.run(["docker", "exec", "--user", "0", container, "rm", "-f", destination],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

    def verify_distributed(self, container):
        gpu_ids = self.target["gpu_ids"]
        if len(gpu_ids) < 2:
            self.job["gpu_verification"]["distributed"] = {"state": "not_applicable", "world_size": 1}
            return
        first = self.target.get("render_gpu", gpu_ids[0])
        second = next(gpu for gpu in gpu_ids if gpu > first) if any(gpu > first for gpu in gpu_ids) else next(gpu for gpu in gpu_ids if gpu != first)
        selected = [first, second]
        self.stage("verifying_gpu", "验证两张 GPU 的真实 NCCL 通信，不启动训练")
        destination = "/tmp/phanthymotus-env-distributed-probe-" + self.job["id"] + ".py"
        source = Path(__file__).parent / "recipes" / "distributed_probe.py"
        self.run(["docker", "cp", str(source), container + ":" + destination])
        try:
            self.probe(container, "python", destination, ",".join(map(str, selected)), timeout=120)
            self.job["gpu_verification"]["distributed"] = {"nccl_all_reduce": True,
                                                           "world_size": 2, "gpu_ids": selected,
                                                           "bytes_per_rank": 1048576}
        finally:
            subprocess.run(["docker", "exec", "--user", "0", container, "rm", "-f", destination],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

    def _container(self, name):
        raw = self.run(["docker", "ps", "-a", "--filter", "name=^/" + name + "$",
                        "--format", "{{.ID}}"], capture=True)
        if not raw:
            return None
        template = '{"Config":{"Image":{{json .Config.Image}},"Labels":{{json .Config.Labels}}},"State":{"Running":{{json .State.Running}}},"NetworkSettings":{"Networks":{{json .NetworkSettings.Networks}}}}'
        info = json.loads(self.run(["docker", "inspect", "--format", template, raw], capture=True))
        # Never act on an unowned name collision, even if its image looks compatible.
        if info["Config"].get("Labels", {}).get(OWNER) != self.target["id"]:
            raise RuntimeError("目标容器名称已被其他环境使用，已停止操作")
        return info

    def ensure_volume(self, name):
        names = self.run(["docker", "volume", "ls", "--format", "{{.Name}}"], capture=True).splitlines()
        if name not in names:
            self.run(["docker", "volume", "create", "--label", OWNER + "=" + self.target["id"], name])
        # Recheck after create: Docker silently reuses a pre-existing volume, including races.
        labels = json.loads(self.run(["docker", "volume", "inspect", "--format", "{{json .Labels}}", name],
                                     capture=True))
        if not isinstance(labels, dict) or labels.get(OWNER) != self.target["id"]:
            raise RuntimeError("目标数据卷属于其他环境，已停止操作；原数据未修改")

    def cleanup_started_runtime(self):
        name = self.job.get("container")
        if not name:
            return
        # Cancellation cannot call run(), whose first operation checks the
        # cancellation marker. Inspect only whitelisted fields, never Env.
        template = '{"image":{{json .Config.Image}},"labels":{{json .Config.Labels}},"running":{{json .State.Running}}}'
        info = subprocess.run(["docker", "inspect", "--format", template, name],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=15)
        if info.returncode:
            return
        value = json.loads(info.stdout)
        labels = value.get("labels") or {}
        owned = labels.get(OWNER) == self.target["id"] and value["image"] == self.job.get("image")
        created_here = labels.get(OWNER + ".job") == self.job["id"]
        if not owned or not (created_here or self.job.get("started_container") == name):
            return  # Reusing an already-running service never authorizes stopping it.
        if value["running"]:
            subprocess.run(["docker", "stop", "--time", "5", name], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            self.log("已停止本任务启动的独立容器，镜像及持久数据保留")

    def isolated(self):
        source = Path(self.config["source_dir"]).resolve(strict=True)
        package = Path(self.config["profile_package"]).resolve(strict=True)
        context = self.folder / "build"
        context.mkdir(exist_ok=True)
        self.stage("preparing", "准备固定版本运行环境与 G1 机器人资产包")
        unpack_profile(package, context / "profiles")
        shutil.copytree(source / "motion_service", context / "motion_service", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        recipe_dir = Path(__file__).parent / "recipes"
        for name in ("Dockerfile", "constraints.txt", "runtime_probe.py", "entrypoint.py", "install_cached.py"):
            shutil.copy2(recipe_dir / name, context / name)
        self.prepare_wheels(context / "wheels")
        build_env = dict(os.environ)
        build_options = []
        proxy = self.config.get("download_proxy")
        if proxy:
            if not isinstance(proxy, str) or not proxy.startswith(("http://", "https://")):
                raise ValueError("管理员下载代理配置无效")
            self.redactions.append(proxy)
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                build_env[name] = proxy
                build_options.extend(["--build-arg", name])
            self.stage("preparing_base", "通过已配置代理下载官方 Python 镜像并校验各层摘要，不修改 Docker 设置")
            result = self.run([sys.executable, "-m", "motion_environment.image_fetch", "--cache",
                               str(self.root / "image-cache")], capture=True, timeout=1200, env=build_env)
            base = json.loads(result.splitlines()[-1])
            existing_id = self.run(["docker", "image", "ls", "--no-trunc", "--filter",
                                    "reference=" + base["image"], "--format", "{{.ID}}"], capture=True)
            if existing_id and existing_id != base["image_id"]:
                raise RuntimeError("基础镜像名称被不同镜像占用，未覆盖")
            if not existing_id:
                self.run(["docker", "load", "--input", base["archive"]], timeout=180)
            loaded = self.run(["docker", "image", "inspect", "--format", "{{.Id}}", base["image"]], capture=True)
            if loaded != base["image_id"]:
                raise RuntimeError("Docker 加载后的官方镜像摘要不匹配")
            dockerfile = context / "Dockerfile"
            lines = dockerfile.read_text().splitlines()
            lines[0] = "FROM " + base["image"]
            dockerfile.write_text("\n".join(lines) + "\n")
            self.job["base_image"] = {k: v for k, v in base.items() if k != "archive"}
        digest = hashlib.sha256()
        for path in sorted(context.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(context)).encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        revision = digest.hexdigest()
        image = "local/phanthymotus-motion:" + revision[:16]
        name = "motus-env-" + self.target["id"] + "-g1-" + revision[:8]
        self.job.update(image=image, source_sha256=revision, container=name)
        existing = self._container(name)
        if existing and existing["Config"]["Image"] != image:
            raise RuntimeError("已有独立环境使用其他版本；请由管理员另建版本，当前环境未修改")
        if not existing:
            self.stage("building", "构建独立镜像；日志来自实际下载和安装输出")
            self.run(["docker", "build", "--force-rm", "--label", OWNER + "=" + self.target["id"],
                      *build_options, "-t", image, str(context)], timeout=7200, env=build_env)
        self.cancelled()
        credentials = self.root / "credentials"
        credentials.mkdir(exist_ok=True, mode=0o700)
        token_path = credentials / (name + ".token")
        if not token_path.exists():
            fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(secrets.token_urlsafe(48))
        token = secret(token_path)
        self.redactions.append(token)
        gpu_ids = self.target["gpu_ids"]
        render_gpu = self.target.get("render_gpu", gpu_ids[0])
        if any(type(i) is not int or i < 0 for i in gpu_ids) or render_gpu not in gpu_ids:
            raise ValueError("管理员 GPU 配置无效")
        envfile = self.folder / "runtime.env"
        fd = os.open(envfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(f"MOTION_API_TOKEN={token}\nMOTION_GPU_IDS={','.join(map(str, gpu_ids))}\n"
                         f"CUDA_VISIBLE_DEVICES={render_gpu}\nMUJOCO_EGL_DEVICE_ID={render_gpu}\n"
                         "MUJOCO_GL=egl\nNVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics\n")
        self.stage("starting_service", "启动独立容器，不发布宿主机公网端口")
        if not existing:
            volume = name + "-data"
            self.ensure_volume(volume)
            self.run(["docker", "create", "--name", name, "--label", OWNER + "=" + self.target["id"],
                      "--label", OWNER + ".job=" + self.job["id"],
                      "--gpus", "all", "--init", "--shm-size", "1g", "--restart", "unless-stopped", "--env-file", str(envfile),
                      "--mount", f"type=volume,source={volume},target=/data", image])
            existing = self._container(name)
        if not existing["State"]["Running"]:
            self.job["started_container"] = name
            atomic_json(self.folder / "job.json", self.job)
            self.run(["docker", "start", name])
        info = self._container(name)
        ip = info["NetworkSettings"]["Networks"]["bridge"]["IPAddress"]
        if not ip:
            raise RuntimeError("独立容器没有私网地址")
        connection = {"endpoint": "http://" + ip + ":15732", "token_file": str(token_path)}
        self.stage("verifying_gpu", "验证实际 CUDA 和 Warp 计算，不启动训练")
        self.probe(name, "python", "/app/runtime_probe.py", render_gpu)
        self.job["gpu_verification"] = {"cuda_compute": True, "warp_compute": True,
                                        "gpu_id": render_gpu}
        self.verify_gpu_environment(name, render_gpu)
        self.verify_distributed(name)
        deadline = time.monotonic() + 90
        while True:
            self.cancelled()
            try:
                self.http(connection, "health")
                break
            except (OSError, RuntimeError) as exc:
                if time.monotonic() > deadline:
                    self.run(["docker", "logs", "--tail", "80", name])
                    raise RuntimeError("服务未就绪，请查看容器启动日志") from exc
                time.sleep(.3)
        self.verify(connection)

    def prepare_wheels(self, destination):
        destination.mkdir(exist_ok=True)
        configured = self.config.get("wheel_cache")
        if not configured:
            atomic_json(destination / "manifest.json", {"wheels": []})
            return
        from .wheel_cache import PACKAGES, digest, wheel_rank
        root = Path(configured).resolve(strict=True)
        self.await_wheel_manifest(root)
        manifest = read_json(root / "manifest.json")
        expected = {**PACKAGES, "torch": "2.14.1+cu126", "warp-lang": "1.18.0+cu12"}
        entries = manifest.get("wheels")
        if (not isinstance(entries, list) or len(entries) != len(expected)
                or {entry.get("name") for entry in entries} != set(expected)):
            raise ValueError("原始 wheel 缓存清单不完整")
        for entry in entries:
            self.cancelled()
            name = entry["filename"]
            if (Path(name).name != name or wheel_rank(name) is None
                    or entry["version"] != expected[entry["name"]]
                    or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                    or urllib.parse.urlparse(entry["official_url"]).hostname not in {
                        "files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org", "github.com"}):
                raise ValueError("原始 wheel 缓存来源或版本无效")
            source = root / name
            if source.is_symlink() or source.stat().st_size != entry["size"] or digest(source) != entry["sha256"]:
                raise ValueError("原始 wheel 缓存 SHA 校验失败")
            with zipfile.ZipFile(source) as archive:
                metadata = [p for p in archive.infolist() if p.filename.endswith(".dist-info/METADATA")]
                if len(metadata) != 1 or metadata[0].file_size > 1024 * 1024:
                    raise ValueError("wheel 元信息无效")
                package = Parser().parsestr(archive.read(metadata[0]).decode())
                normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
                if normalize(package["Name"]) != normalize(entry["name"]) or package["Version"] != entry["version"]:
                    raise ValueError("wheel 实际名称或版本与官方清单不匹配")
            try:
                os.link(source, destination / name)
            except OSError:
                shutil.copy2(source, destination / name)
            self.log(f"已校验原始安装包 {entry['name']} {entry['version']}，SHA256 {entry['sha256']}")
        atomic_json(destination / "manifest.json", manifest)

    def await_wheel_manifest(self, root):
        seconds = self.config.get("cache_wait_seconds", 0)
        if type(seconds) is not int or not 0 <= seconds <= 7200:
            raise ValueError("管理员缓存等待时间必须为 0–7200 秒")
        deadline = time.monotonic() + seconds
        last_log = 0.0
        while not (root / "manifest.json").is_file():
            self.cancelled()
            if (root / "transfer-failed.json").is_file():
                raise RuntimeError("原始安装包上传已失败；请恢复上传后重试，详情见管理员传输日志")
            now = time.monotonic()
            if not seconds or now >= deadline:
                raise TimeoutError("原始安装包缓存尚未上传完整；可在上传完成后重试")
            if now >= last_log + 10:
                received, seen = 0, set()
                for path in root.iterdir():
                    if path.is_symlink() or not (path.name.endswith(".whl") or path.name.endswith(".whl.uploading")):
                        continue
                    try:
                        stat = path.stat()
                    except FileNotFoundError:  # An upload may be atomically renamed during this scan.
                        continue
                    identity = stat.st_dev, stat.st_ino
                    if identity not in seen and path.is_file():
                        received += stat.st_size
                        seen.add(identity)
                self.stage("preparing", f"准备安装包：已接收 {received / 1024**2:.1f} MiB；完整校验通过后才开始构建")
                last_log = now
            time.sleep(.5)


def unpack_profile(package, destination):
    destination = Path(destination)
    destination.mkdir(exist_ok=True)
    allowed = {"profiles.json", "scene.mjb", "manifest.json"}
    with tarfile.open(package, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != 3 or {m.name for m in members} != allowed:
            raise ValueError("G1 资产包文件清单无效")
        for member in members:
            if not member.isfile() or member.size > 200 * 1024**2:
                raise ValueError("G1 资产包包含非法文件")
            with archive.extractfile(member) as source, (destination / member.name).open("wb") as target:
                shutil.copyfileobj(source, target)
    manifest = read_json(destination / "manifest.json")
    for name in ("profiles.json", "scene.mjb"):
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError("G1 资产包校验失败")
    profiles = read_json(destination / "profiles.json")
    if (len(profiles) != 1 or profiles[0].get("model_path") != "/profiles/scene.mjb"
            or profiles[0].get("mujoco_version") != "3.16.0"
            or profiles[0].get("robot_type") != "unitree_g1"):
        raise ValueError("G1 资产包与固定环境版本不匹配")


def execute(root, config, job_id):
    folder = Path(root) / "jobs" / job_id
    job = read_json(folder / "job.json")
    targets = {p["id"]: p for p in config["targets"]}
    task = Task(root, job, targets[job["target_id"]], config)
    try:
        task.check()
        task.cancelled()
        if job["action"] == "check":
            if task.target.get("existing"):
                task.verify_existing_gpu()
                task.verify(task.target["existing"], finalize=False)
            task.stage("complete", "服务器检查完成", state="succeeded")
        elif job["mode"] == "reuse":
            task.verify_existing_gpu()
            task.verify(task.target["existing"])
        else:
            task.isolated()
    except Cancelled as exc:
        try:
            task.cleanup_started_runtime()
        except Exception as cleanup_error:
            task.job["cleanup_error"] = str(cleanup_error)
        task.stage(task.job["stage"], str(exc), state="cancelled", error=str(exc))
    except Exception as exc:
        try:
            task.cleanup_started_runtime()
        except Exception as cleanup_error:
            task.job["cleanup_error"] = str(cleanup_error)
        task.stage(task.job["stage"], str(exc), state="failed", error=str(exc))
