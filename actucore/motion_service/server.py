"""Authenticated motion service. No ROS, browser credentials or shell commands."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import threading

from .contracts import PROTOCOL, validate_profile
from .jobs import JobManager
from .resources import ResourceManager
from .store import AssetStore, read_json

log = logging.getLogger(__name__)


class MotionService:
    def __init__(self, root, profiles, python=None, simulation_factory=None, resources=None):
        validated = [validate_profile(p) for p in profiles]
        if len({p["id"] for p in validated}) != len(validated):
            raise ValueError("重复机器人配置标识")
        self.profiles = {p["id"]: p for p in validated}
        self.store = AssetStore(root, self.profiles)
        self.resources = resources or ResourceManager(python)
        self.jobs = JobManager(self.store, python=python, resources=self.resources)
        from .training_preview import TrainingPreview
        self.training_preview = TrainingPreview(self.store)
        from .dataset_render import DatasetRenderer
        self.dataset_renderer = DatasetRenderer(self.store)
        if simulation_factory is None:
            from .simulation import SimulationManager
            simulation_factory = SimulationManager
        self.simulations = simulation_factory(self.profiles, self.store.root / "sessions",
                                             model_resolver=self._controller,
                                             dataset_resolver=lambda id: self.store.dataset(id))

    def _controller(self, model_id, profile):
        model, folder = self.store.verified_model(model_id)
        if model["profile_hash"] != profile["profile_hash"]:
            raise ValueError("策略与仿真机器人配置不匹配")
        if not model.get("onnx"):
            raise ValueError("模型没有可部署的 ONNX 策略")
        from .policy import SuperMimicPolicy
        return SuperMimicPolicy(folder / model["onnx"], profile,
                               self.store.dataset(model["dataset_id"])[1], manifest=model)

    def health(self):
        return {"protocol": PROTOCOL, "state": "ready", "profiles": len(self.profiles),
                "runtime": {name: importlib.util.find_spec(name) is not None
                            for name in ("mujoco", "supermimic", "mjlab", "onnxruntime")},
                "training_engine": "mjlab_mujoco_warp", "interactive_engine": "native_mujoco",
                "resume_training": False}

    def call(self, args):
        action = args.get("action")
        if action == "health":
            return self.health()
        if action == "catalog":
            return self.store.catalog()
        if action == "resources":
            return self.resources.inventory(refresh=args.get("refresh") is True)
        if action == "plan":
            return self.resources.training_plan(args)
        if action == "training_preview":
            return self.training_preview.render(args.get("job_id"),
                                                width=args.get("width", 768), height=args.get("height", 432),
                                                frame=args.get("frame"), sample=args.get("sample"),
                                                sample_start=args.get("sample_start"), sample_limit=args.get("sample_limit"))
        if action == "dataset_preview":
            return self.store.preview(args)
        if action == "dataset_render":
            return self.dataset_renderer.render(args.get("dataset_id"), **{
                key: args[key] for key in ("offset", "width", "height", "azimuth", "elevation", "distance", "lookat", "jpeg_quality")
                if key in args})
        if action in {"upload_begin", "upload_chunk", "upload_finish", "artifact_read"}:
            return getattr(self.store, action)(args)
        if action in {"train", "evaluate"}:
            return self.jobs.submit(action, args)
        if action == "jobs":
            return {"jobs": self.jobs.list()}
        if action == "job":
            return self.jobs.inspect(args.get("job_id"))
        if action == "cancel":
            return self.jobs.cancel(args.get("job_id"))
        if action == "export":
            return self.store.export(args.get("model_id"))
        if isinstance(action, str) and action.startswith("simulation_"):
            return self.simulations.call(action, args)
        raise ValueError("不支持的操作")

    def close(self):
        self.jobs.close()
        self.training_preview.close()
        self.dataset_renderer.close()
        self.simulations.close()


def make_handler(service, token):
    if not isinstance(token, str) or len(token) < 16:
        raise ValueError("MOTION_API_TOKEN 至少需要 16 个字符")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Request bodies and credentials must never enter access logs.

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send_json(self, status, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            value = self.headers.get("Authorization", "")
            if not hmac.compare_digest(value.encode(), ("Bearer " + token).encode()):
                self.send_json(401, {"state": "error", "message": "服务认证失败"})
                return False
            return True

        def do_GET(self):
            if self.authorized():
                self.send_json(200, service.health()) if self.path == "/health" else self.send_error(404)

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/call":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("请求大小无效")
                args = json.loads(self.rfile.read(length))
                if not isinstance(args, dict):
                    raise ValueError("请求必须是对象")
                self.send_json(200, service.call(args))
            except (ValueError, KeyError, FileNotFoundError) as exc:
                message = "资源不存在" if isinstance(exc, (KeyError, FileNotFoundError)) else str(exc)
                self.send_json(400, {"state": "error", "message": message})
            except (ImportError, RuntimeError, TimeoutError) as exc:
                self.send_json(503, {"state": "error", "message": str(exc)})
            except Exception:
                log.exception("motion service operation failed")
                self.send_json(500, {"state": "error", "message": "服务内部错误，请检查服务器日志"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Phanthymotus motion service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15732)
    parser.add_argument("--data-dir", default="./motion-data")
    parser.add_argument("--profiles", required=True, help="管理员配置的机器人配置 JSON 数组")
    parser.add_argument("--worker-python", default=None)
    args = parser.parse_args()
    token = os.environ.get("MOTION_API_TOKEN", "")
    if len(token) < 16:
        parser.error("请设置至少 16 字符的 MOTION_API_TOKEN")
    root = Path(args.data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        service = MotionService(root, read_json(args.profiles), python=args.worker_python)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(service, token))
        server.daemon_threads = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        try:
            server.serve_forever()
        finally:
            server.server_close()
            service.close()


if __name__ == "__main__":
    main()
