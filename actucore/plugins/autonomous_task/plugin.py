"""Motus lifecycle and one-observation/one-action loop for tabletop tasks."""
from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
import uuid
import urllib.request
from collections import OrderedDict
from pathlib import Path

from .cameras import CameraInputs, camera_ports, resolve_cameras
from .model import Model, TransientModelError

log = logging.getLogger(__name__)
_MODES = {"left": ("left",), "right": ("right",), "both": ("left", "right")}
_FIELDS = ("base_url", "key", "model_name", "arm_mode",
           "head_topic", "left_wrist_topic", "right_wrist_topic")
_OUTPUTS = {
    "arm": {"topic": "/actucore/autonomous_task/arm", "format": "data/json",
            "desc": "已执行的末端位姿命令（含手臂、基座系、实参与回执）"},
    "gripper": {"topic": "/actucore/autonomous_task/gripper", "format": "data/json",
                "desc": "已执行的夹爪开口命令（含手臂、毫米与回执）"},
}
_TIME_LIMIT = 3600
_STEP_LIMIT = 100


def _completion(action_id, status, reason):
    url = os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
    event = {"action_id": action_id, "tool": "autonomous_task", "status": status,
             "result": {"status": status, "reason": reason}, "ts": time.time()}
    request = urllib.request.Request(url + "/api/acp/complete", json.dumps(event).encode(),
                                     {"Content-Type": "application/json"})
    try:
        # On-device Agent Core uses its local self-signed certificate.
        with urllib.request.urlopen(request, timeout=5, context=ssl._create_unverified_context()):
            pass
    except OSError as error:
        log.warning("autonomous_task ACP %s: %s", action_id, error)


class AutonomousTaskPlugin:
    # The host joins PREFIX and tool name and routes on the first underscore.
    PREFIX = "autonomous"

    def __init__(self, plugin_cfg, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor
        self._lock = threading.RLock()
        self._node = None
        self._cameras = None
        self._binding = {}
        self._publishers = {}
        self._runs = OrderedDict()
        self._active = None
        self._cancel = None

    def get_tools(self):
        config = {key: {"type": "string", "scope": "instance",
                        "default": self._cfg.get(key, "")} for key in _FIELDS if key != "key"}
        config["key"] = {"type": "string", "format": "password", "scope": "instance",
                         "title": "模型 API Key"}
        config["arm_mode"].update(enum=list(_MODES), default=self._cfg.get("arm_mode", "left"))
        for role in ("head", "left_wrist", "right_wrist"):
            config[role + "_topic"]["description"] = f"自定义相机话题匹配到 {role}（可选）"
        actions = {
            "submit": {"params": ["instruction", "arm_mode", "request_id"],
                       "description": "提交自然语言桌面任务。异步返回 action_id，以 status 查询。"},
            "cancel": {"params": ["action_id"], "description": "停止后续动作，等待已发送的动作回执。"},
            "status": {"params": ["action_id", "include_history"], "description": "查询任务进度、结果与失败原因。"},
        }
        return [{"name": "task", "type": "processor", "multiInstance": False,
                 "description": "自主桌面操作。根据接入的 RGB、夹爪和手臂状态逐步决策与复核。",
                 "topic_in": camera_ports(), "topic_out": list(_OUTPUTS.values()),
                 "configSchema": {"type": "object", "properties": config},
                 "inputSchema": {
                     "type": "object", "required": ["action"],
                     "properties": {
                         "action": {"type": "string", "enum": ["info", "config", "start", "stop", *actions]},
                         "instruction": {"type": "string"},
                         "arm_mode": {"type": "string", "enum": list(_MODES)},
                         "request_id": {"type": "string", "description": "可选，重试时使用同一编号"},
                         "action_id": {"type": "string"}, "include_history": {"type": "boolean"}},
                     "x-action-params": actions, "x-is-dangerous": True,
                     "x-hooks": {"on_interrupt_all": {"action": "cancel"},
                                 "on_interrupt_motion": {"action": "cancel"}},
                     "x-resource": ["arm_l", "arm_r", "hand_l", "hand_r"],
                     "x-completion": {"actions": ["submit"], "timeout": _TIME_LIMIT + 240}}}]

    def dispatch(self, name, args):
        action = args.get("action", name)
        with self._lock:
            try:
                if action == "info":
                    return {"state": "running" if self._node else "idle",
                            "topic_in": camera_ports(self._binding if args.get("instance_id") else None),
                            "topic_out": list(_OUTPUTS.values()), "inputs": dict(self._binding),
                            "configured": all(self._cfg.get(k) for k in ("base_url", "key", "model_name")),
                            "arm_mode": self._cfg.get("arm_mode", "left"), "active_task": self._active}
                if action == "config":
                    if self._active:
                        raise ValueError("任务正在执行，暂不能修改配置")
                    changes = {k: args[k] for k in _FIELDS if k in args}
                    if any(not isinstance(v, str) for v in changes.values()):
                        raise ValueError("配置项必须为字符串")
                    if changes.get("arm_mode", "left") not in _MODES:
                        raise ValueError("arm_mode 需要是 left、right 或 both")
                    if self._node and any(k.endswith("_topic") and v != self._cfg.get(k, "")
                                          for k, v in changes.items()):
                        raise ValueError("修改相机映射前请停止卡片")
                    self._cfg.update(changes)
                    return self.dispatch("info", {})
                if action == "start":
                    return self._start(args)
                if action == "stop":
                    if self._cancel:
                        self._cancel.set()
                    self._close()
                    return {"state": "idle", "pending_task": self._active}
                if action == "submit":
                    return self._submit(args)
                if action in ("cancel", "status"):
                    action_id = args.get("action_id") or self._active or next(reversed(self._runs), None)
                    if action_id is None:
                        return {"state": "idle"}
                    if action_id not in self._runs:
                        raise ValueError("未知 action_id")
                    if action == "cancel" and action_id == self._active:
                        self._cancel.set()
                        self._runs[action_id]["state"] = "cancelling"
                    return self._status(action_id, args.get("include_history") is True)
            except (ValueError, RuntimeError, OSError, TypeError) as error:
                return {"state": "error", "message": str(error)}
        return None

    def _start(self, args):
        from rclpy.node import Node
        from std_msgs.msg import String

        binding = resolve_cameras(args, self._cfg)
        if self._node:
            if binding != self._binding:
                raise ValueError("卡片已运行，请先停止再更换接线")
            return self.dispatch("info", args | {"action": "info"})
        if self._active:
            raise ValueError("仍在等待上次任务结束")
        self._node = Node("autonomous_task")
        try:
            self._cameras = CameraInputs(self._node, binding)
            self._publishers = {k: self._node.create_publisher(String, v["topic"], 10)
                                for k, v in _OUTPUTS.items()}
            self._executor.add_node(self._node)
            self._binding = binding
        except Exception:
            self._close()
            raise
        return self.dispatch("info", {"instance_id": args.get("instance_id")})

    def _close(self):
        if self._cameras:
            self._cameras.close()
        if self._node:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        self._node = self._cameras = None
        self._binding, self._publishers = {}, {}

    def _submit(self, args):
        instruction = args.get("instruction")
        mode = args.get("arm_mode", self._cfg.get("arm_mode", "left"))
        request_id = args.get("request_id")
        if not isinstance(instruction, str) or not instruction.strip() or mode not in _MODES:
            raise ValueError("需要自然语言 instruction 和有效的 arm_mode")
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("request_id 必须为非空字符串")
            for run in self._runs.values():
                if run["request_id"] == request_id:
                    if (instruction, mode) != (run["instruction"], run["arm_mode"]):
                        raise ValueError("此 request_id 已用于其他任务")
                    return self._status(run["action_id"])
        if self._active:
            raise ValueError(f"任务 {self._active} 正在执行")
        if not self._cameras or not self._binding:
            raise ValueError("请启动卡片并接入至少一路 RGB；头部、左腕、右腕均可选")
        model = Model(*(self._cfg.get(k, "") for k in ("base_url", "key", "model_name")))
        action_id = "autonomous_task_" + uuid.uuid4().hex
        run = {"action_id": action_id, "request_id": request_id, "instruction": instruction,
               "arm_mode": mode, "state": "running", "step": 0, "phase": "starting",
               "started_at": time.time(), "history": []}
        if len(self._runs) >= 32:
            self._runs.popitem(last=False)
        self._runs[action_id] = run
        self._active, self._cancel = action_id, threading.Event()
        threading.Thread(target=self._work, args=(run, model, self._cameras, self._cancel),
                         name="autonomous_task", daemon=True).start()
        return self._status(action_id)

    def _status(self, action_id, history=False):
        task = self._runs[action_id]
        out = {k: v for k, v in task.items() if k != "history"}
        if task["history"]:
            out["last"] = task["history"][-1]
        if history:
            out["history"] = list(task["history"])
        return out

    def _work(self, run, model, cameras, cancel):
        action_id = run["action_id"]
        path = None
        status = "failed"
        reason = "任务未完成"
        try:
            from .tron2 import Tron2

            if self._cfg.get("record_dir"):
                path = Path(self._cfg["record_dir"]) / action_id
                path.mkdir(parents=True, mode=0o700)
            status, reason = self._loop(run, model, Tron2(), cameras, cancel, path)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            log.warning("%s step %d: %s", action_id, run["step"], reason)
        with self._lock:
            if cancel.is_set():
                status, reason = "cancelled", f"已停止后续决策。最后反馈：{reason}"
            run.update(state=status, phase="finished", reason=reason, finished_at=time.time())
            self._active = self._cancel = None
        try:
            if path:
                (path / "result.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
        finally:
            _completion(action_id, status, reason)

    def _loop(self, run, model, driver, cameras, cancel, path):
        arms = _MODES[run["arm_mode"]]
        review = False
        inference_failures = 0
        deadline = time.monotonic() + _TIME_LIMIT

        def check():
            if cancel.is_set():
                raise InterruptedError("任务已取消")
            if time.monotonic() >= deadline:
                raise TimeoutError("任务运行超过一小时")

        for step in range(1, _STEP_LIMIT + 1):
            check()
            with self._lock:
                run.update(step=step, phase="observing")
            if review:
                cancel.wait(3)
                check()
            images = cameras.snapshot(cancel)
            state = driver.observe(arms)
            if path:
                for role, jpeg in images.items():
                    (path / f"{step:03d}-{role}.jpg").write_bytes(jpeg)
            check()
            with self._lock:
                run["phase"] = "reviewing" if review else "deciding"
            decide = model.review if review else model.decide
            try:
                name, args = decide(run["instruction"], arms, images, state, run["history"])
            except TransientModelError as error:
                inference_failures += 1
                self._record(run, path, {"step": step, "review": review,
                                         "tool": "model_unavailable", "note": str(error),
                                         "feedback": {"motion_sent": False}}, state)
                if inference_failures >= 3:
                    return "failed", f"连续三次推理没有获得响应：{error}"
                cancel.wait(2 * inference_failures)
                continue
            inference_failures = 0
            record = {"step": step, "review": review, "tool": name, "args": args,
                      "note": model.last_note, "model": model.last_model}
            check()
            if name not in ("observe", "done", "failed"):
                arm, _, tool = name.partition("_")
                if review or arm not in arms or tool not in ("move_to", "move_by", "set_gripper"):
                    raise ValueError(f"当前模式未提供 {name}")
                with self._lock:
                    run["phase"] = "executing"
                try:
                    result = driver.execute(name, args, cancel)
                except Exception as error:
                    record["feedback"] = {"status": "error", "reason": str(error)}
                    self._record(run, path, record, state)
                    raise
                record["feedback"] = result
                if result["motion_sent"] and result["status"] == "completed":
                    self._publish(run["action_id"], step, name, args, result)
            self._record(run, path, record, state)
            check()
            if name == "failed":
                return "failed", args["reason"]
            if name == "done" and review:
                return "completed", args["reason"]
            review = name == "done"
        return "failed", f"已达到 {_STEP_LIMIT} 轮，未能复核任务完成"

    def _record(self, run, path, record, state):
        with self._lock:
            run["history"].append(record)
        if path:
            with (path / "steps.jsonl").open("a") as trace:
                trace.write(json.dumps({**record, "before": state}, ensure_ascii=False) + "\n")

    def _publish(self, action_id, step, name, args, feedback):
        from std_msgs.msg import String
        message = String()
        message.data = json.dumps({"action_id": action_id, "step": step, "tool": name,
                                   "arguments": args, "feedback": feedback, "executed_at": time.time()})
        with self._lock:
            publisher = self._publishers.get("gripper" if name.endswith("_set_gripper") else "arm")
            if publisher:
                publisher.publish(message)
