"""Whole-body motion card. Jobs live on the service, not in the iframe."""

from __future__ import annotations

from pathlib import Path
import json
import os
import threading
import uuid

from .client import MotionClient, ServiceError
from .session import WorkbenchSession


REMOTE_ACTIONS = {
    "catalog": (), "resources": (), "jobs": (), "job": ("job_id",), "cancel": ("job_id",),
    "train": ("profile_id", "dataset_id", "num_envs", "max_iterations", "preset", "gpu_mode", "gpu_ids"),
    "evaluate": ("model_id", "dataset_id", "profile_id"),
    "upload_begin": ("name", "size", "profile_id"),
    "upload_chunk": ("upload_id", "offset", "data"),
    "upload_finish": ("upload_id",),
    "dataset_preview": ("dataset_id", "limit"),
    "export": ("model_id",),
    "artifact_read": ("package_id", "offset", "limit"),
}
CAMERA_PARAMS = ("width", "height", "azimuth", "elevation", "distance", "lookat", "offset", "jpeg_quality")
SIMULATION_ACTIONS = {
    "simulation_create": ("profile_id", "model_id", "view_id", "request_id"),
    "simulation_list": ("view_id",),
    "simulation_attach": ("session_id", "profile_id", "model_id", "view_id"),
    "simulation_detach": ("session_id", "view_id"),
    "simulation_claim": ("session_id", "view_id", "control_epoch", "takeover"),
    "simulation_heartbeat": ("session_id", "view_id", "control_epoch"),
    "simulation_release": ("session_id", "view_id", "control_epoch"),
    "simulation_run": ("session_id",), "simulation_pause": ("session_id",),
    "simulation_step": ("session_id", "steps"), "simulation_reset": ("session_id",),
    "simulation_close": ("session_id",), "simulation_record": ("session_id",),
    "simulation_record_stop": ("session_id",),
    "simulation_perturb": ("session_id", "body_id", "force", "torque", "duration_s"),
}
for _action, _params in list(SIMULATION_ACTIONS.items()):
    if _action not in {"simulation_create", "simulation_list", "simulation_attach", "simulation_detach"}:
        SIMULATION_ACTIONS[_action] = tuple(dict.fromkeys((*_params, "view_id", "control_epoch")))
ENVIRONMENT_ACTIONS = {
    "env_targets": (), "env_check": ("target_id",),
    "env_install": ("target_id", "recipe_id", "mode"), "env_cancel": ("job_id",),
    "env_connect": ("job_id",),
}
INSPECT_ACTIONS = {"resources": (), "jobs": (), "job": ("job_id",), "training_preview": ("job_id", "frame", "sample", "sample_start", "sample_limit", "width", "height"),
                   "plan": ("num_envs", "preset", "gpu_mode", "gpu_ids"),
                   "dataset_render": ("dataset_id", *CAMERA_PARAMS),
                   "simulation_frame": ("session_id", "detail", "view_id", *CAMERA_PARAMS),
                   "simulation_state": ("session_id", "detail", "view_id"),
                   "simulation_replay": ("session_id", "offset", "limit", "view_id"),
                   "env_status": ("job_id", "cursor", "target_id")}


def connection_schema():
    return {"type": "object", "properties": {
        "endpoint": {"type": "string", "title": "服务器地址", "scope": "shared"},
        "api_key": {"type": "string", "title": "访问凭证", "format": "password",
                    "scope": "shared", "x-sensitive": True},
        "environment_job": {"type": "string", "title": "已接入环境任务", "scope": "shared"},
    }, "required": []}


class MotionPlugin:
    PREFIX = "motion"

    def __init__(self, plugin_cfg: dict, executor=None, namespace=""):
        self._cfg = dict(plugin_cfg or {})
        self._lock = threading.RLock()
        self._state_dir = Path(self._cfg.get("state_dir") or os.environ.get("MOTION_WORKBENCH_STATE_DIR")
                               or "~/.local/state/phanthymotus/motion").expanduser()
        self._binding_path = self._state_dir / "environment.json"
        self._environment_job = ""
        try:
            self._environment = MotionClient(self._cfg.get("environment_endpoint") or os.environ.get("MOTION_ENVIRONMENT_URL", ""),
                                             self._cfg.get("environment_token") or os.environ.get("MOTION_ENVIRONMENT_TOKEN", ""))
        except (ServiceError, ValueError, TypeError):
            self._environment = MotionClient()
        endpoint = self._cfg.get("endpoint") or os.environ.get("MOTION_SERVICE_URL", "")
        key = self._cfg.get("api_key") or os.environ.get("MOTION_API_TOKEN", "")
        try:
            binding = json.loads(self._binding_path.read_text())
            if binding.get("environment_job") and self._environment.configured:
                endpoint, key = binding["endpoint"], binding["api_key"]
                self._environment_job = binding["environment_job"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self._configuration_error = ""
        try:
            self._client = MotionClient(endpoint, key)
        except (ServiceError, ValueError, TypeError) as exc:
            self._client = MotionClient()
            self._configuration_error = str(exc)
        self._cfg.update(endpoint=self._client.endpoint, api_key=key)
        self._session = WorkbenchSession(self._state_dir / "sessions", self._client.endpoint)

    def start(self):
        """Bundle startup never creates a training job or moves a robot."""

    def stop(self):
        """Remote jobs are durable; closing a client does not cancel them."""
        with self._lock:
            self._session.pause_owned(self._client)

    def get_tools(self):
        actions = ["ui", "info", "start", "stop", "config", "check", "inspect",
                   *REMOTE_ACTIONS, *SIMULATION_ACTIONS, *ENVIRONMENT_ACTIONS]
        props = {"action": {"type": "string", "enum": actions}}
        all_params = {**REMOTE_ACTIONS, **SIMULATION_ACTIONS, **ENVIRONMENT_ACTIONS, **INSPECT_ACTIONS}
        for key in {p for params in all_params.values() for p in params}:
            props[key] = {"type": "array" if key in {"gpu_ids", "lookat", "force", "torque"} else
                          "integer" if key in {"size", "offset", "limit", "num_envs", "max_iterations", "width", "height", "jpeg_quality", "steps", "body_id", "cursor", "frame", "sample", "sample_start", "sample_limit", "control_epoch"} else
                          "number" if key in {"azimuth", "elevation", "distance", "duration_s"} else "string"}
        props["takeover"] = {"type": "boolean"}
        for key in ("lookat", "force", "torque"):
            props[key].update(items={"type": "number"}, minItems=3, maxItems=3)
        props["gpu_ids"].update(items={"type": "integer", "minimum": 0}, minItems=1, uniqueItems=True)
        props["gpu_mode"] = {"type": "string", "enum": ["auto", "manual"]}
        props["mode"] = {"type": "string", "enum": [*INSPECT_ACTIONS, "reuse", "isolated"]}
        return [{"name": self.PREFIX, "type": "actuator", "multiInstance": False,
                 "description": "动作学习：动作导入、SuperMimic 训练监看、版本试跑和模型导出。",
                 "inputSchema": {
                     "type": "object", "properties": props, "required": ["action"],
                     "x-workbench": {"protocol": "motus.workbench/1", "kind": "motion-training",
                                     "title": "动作学习", "entry": "ui", "actions": actions,
                                     "config": ["endpoint", "api_key", "environment_job"]},
                     "x-action-params": {
                         "ui": {"params": [], "description": "打开动作学习"},
                         "catalog": {"params": [], "description": "查询机器人配置、动作数据和模型"},
                         "resources": {"params": [], "description": "查询真实 GPU 占用及训练可用资源"},
                         "inspect": {"params": ["mode", *sorted({p for params in INSPECT_ACTIONS.values() for p in params})], "required": ["mode"],
                                     "description": "只读查询任务、环境配置进度、动作预览及仿真画面"},
                         "jobs": {"params": [], "description": "查询训练和评测任务"},
                         "job": {"params": ["job_id"], "required": ["job_id"], "description": "查询任务详情"},
                         "cancel": {"params": ["job_id"], "required": ["job_id"], "description": "取消指定任务"},
                     },
                 }, "configSchema": connection_schema()}]

    def _info(self, view_id=None):
        with self._lock:
            current = self._session.for_view(view_id)
            return {"state": "ready" if self._client.configured else "unconfigured", "busy": False,
                    "config_values": {"endpoint": self._client.endpoint, "environment_job": self._environment_job},
                    "config_secrets": {"api_key": self._client.has_key},
                    "session_id": current.get("id") if current else None, "current_session": current,
                    "environment_available": self._environment.configured,
                    "message": self._configuration_error}

    def _set_connection(self, client, key, job_id=""):
        if client.endpoint != self._client.endpoint:
            self._session.assert_switchable(self._client)
        self._state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if job_id:
            temporary = self._binding_path.with_name("environment." + uuid.uuid4().hex + ".tmp")
            try:
                fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w") as stream:
                    json.dump({"environment_job": job_id, "endpoint": client.endpoint, "api_key": key}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self._binding_path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            self._binding_path.unlink(missing_ok=True)
        if client.endpoint != self._client.endpoint:
            previous = self._session
            self._session = WorkbenchSession(self._state_dir / "sessions", client.endpoint)
            self._session.inherit_disconnected_views(previous)
        self._cfg.update(endpoint=client.endpoint, api_key=key)
        self._client, self._environment_job = client, job_id
        self._configuration_error = ""

    def _connect_environment(self, job_id):
        if not isinstance(job_id, str) or not job_id:
            raise ServiceError("请先检查或安装环境")
        value = self._environment.call("connection", job_id=job_id)
        connection = value.get("connection", value)
        client = MotionClient(connection.get("endpoint", ""), connection.get("api_key", ""))
        if not client.configured:
            raise ServiceError("环境尚未通过检查")
        client.health()
        self._set_connection(client, connection["api_key"], job_id)
        return self._info()

    def dispatch(self, name: str, args: dict):
        args = args or {}
        action = args.get("action") or name
        try:
            if action == "ui":
                return {"protocol": "motus.workbench/1", "html": Path(__file__).with_name("workbench.html").read_text()}
            if action in {"info", "start"}:
                with self._lock:
                    try:
                        self._session.refresh(self._client, args.get("view_id"))
                    except ServiceError as exc:
                        # Environment preparation must remain reachable when a
                        # previous Motion service is unavailable. Keep ownership
                        # until a successful list proves that session is gone.
                        return {**self._info(args.get("view_id")), "session_reachable": False, "message": str(exc)}
                    return {**self._info(args.get("view_id")), "session_reachable": True}
            if action == "stop":
                if args.get("view_id"):
                    if type(args.get("control_epoch")) is int:
                        with self._lock:
                            self._session.call(self._client, "simulation_pause", {
                                key: args[key] for key in ("session_id", "view_id", "control_epoch") if key in args})
                else:
                    self.stop()
                return self._info(args.get("view_id"))
            if action == "config":
                with self._lock:
                    job_id = args.get("environment_job", self._environment_job)
                    if job_id:
                        return self._info() if job_id == self._environment_job else self._connect_environment(job_id)
                    endpoint = args.get("endpoint", self._client.endpoint)
                    key = args.get("api_key", self._cfg.get("api_key", ""))
                    client = MotionClient(endpoint, key)
                    self._set_connection(client, key)
                return self._info()
            with self._lock:
                client = self._client
            if action == "inspect":
                mode = args.get("mode")
                if not isinstance(mode, str) or mode not in INSPECT_ACTIONS:
                    raise ServiceError("不支持的只读查看模式")
                values = {key: args[key] for key in INSPECT_ACTIONS[mode] if key in args}
                if mode == "training_preview":
                    values.setdefault("width", 896)
                    values.setdefault("height", 448 if values.get("sample") is not None else 320)
                    for dimension, lower, upper in (("width", 320, 960), ("height", 160, 640)):
                        value = values[dimension]
                        if type(value) is not int or not lower <= value <= upper:
                            raise ServiceError(f"{dimension} 必须是 {lower}–{upper} 之间的整数")
                if mode == "env_status":
                    return self._environment.call("status", **values)
                if mode.startswith("simulation_"):
                    remote = {"simulation_frame": "simulation_render", "simulation_state": "simulation_inspect",
                              "simulation_replay": "simulation_replay"}[mode]
                    with self._lock:
                        return self._session.call(self._client, remote, values)
                return client.call(mode, **values)
            if action in SIMULATION_ACTIONS:
                with self._lock:
                    return self._session.call(self._client, action, {key: args[key] for key in SIMULATION_ACTIONS[action] if key in args})
            if action in ENVIRONMENT_ACTIONS:
                values = {key: args[key] for key in ENVIRONMENT_ACTIONS[action] if key in args}
                if action == "env_connect":
                    with self._lock:
                        return self._connect_environment(values.get("job_id"))
                return self._environment.call(action.removeprefix("env_"), **values)
            if action == "check":
                return client.health()
            if action in REMOTE_ACTIONS:
                # Routing, tokens, and arbitrary remote actions cannot be supplied by the iframe.
                return client.call(action, **{key: args[key] for key in REMOTE_ACTIONS[action] if key in args})
            return {"state": "error", "message": "不支持的全身动作操作", "error": "不支持的全身动作操作"}
        except (ServiceError, ValueError, TypeError, OSError) as exc:
            return {"state": "error", "message": str(exc), "error": str(exc)}
