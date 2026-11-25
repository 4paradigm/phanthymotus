"""Motus lifecycle and one-observation/one-action loop for tabletop tasks."""
from __future__ import annotations

import json
import logging
import os
import re
import ssl
import threading
import time
import uuid
import urllib.request
from collections import OrderedDict
from pathlib import Path

from .cameras import CameraInputs, camera_ports, resolve_inputs
from .eef import EefEndpoint, EefInterface
from .model import InvalidDecisionError, Model, TransientModelError
from .model_log import ModelLog

log = logging.getLogger(__name__)
_MODES = {"left": ("left",), "right": ("right",), "both": ("left", "right")}
_FIELDS = ("base_url", "key", "model_name", "arm_mode",
           "head_topic", "left_wrist_topic", "right_wrist_topic", "frame_description")
_CONTROL_TOPIC = "/actucore/autonomous_task/control/eef"
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
        self._model_log_root = Path(self._cfg.get("model_log_dir") or "/models/autonomous_task/model_logs")
        self._executor = executor
        self._lock = threading.RLock()
        self._node = None
        self._cameras = None
        self._binding = {}
        self._eef = None
        self._interface = None
        self._stopping = False
        self._runs = OrderedDict()
        self._active = None
        self._cancel = None

    def _ports(self, binding=None):
        binding = binding or {}
        state = {"format": "state/eef", "desc": "驱动实测 TCP 位姿和夹爪开口"}
        if "eef_state" in binding:
            state["topic"] = binding["eef_state"]
        return camera_ports(binding) + [state]

    def _output(self):
        return [{"format": "control/eef", "topic": _CONTROL_TOPIC,
                 "desc": "完整的绝对 TCP 位姿与夹爪开口目标"}]

    def get_tools(self):
        config = {name: {"type": "string", "scope": "shared"} for name in _FIELDS}
        for name, field in config.items():
            if name != "key":
                field["default"] = self._cfg.get(name, "")
        config["base_url"]["description"] = "LLM API URL"
        config["key"].update(description="LLM API Key", format="password")
        config["model_name"]["description"] = "LLM 模型名称"
        config["arm_mode"].update(enum=list(_MODES), default=self._cfg.get("arm_mode", "left"))
        for role in ("head", "left_wrist", "right_wrist"):
            config[role + "_topic"]["description"] = f"自定义相机话题匹配到 {role}（可选）"
        config["frame_description"]["description"] = "平台已知的基座轴与相机方向约定（可选）"
        actions = {
            "submit": {"params": ["instruction", "request_id"],
                       "description": "提交自然语言桌面任务。异步返回 action_id，以 status 查询。"},
            "cancel": {"params": ["action_id"], "description": "停止后续决策，将在途目标更新为实测保持位置。"},
            "status": {"params": ["action_id", "include_history"], "description": "查询任务进度、结果与失败原因。"},
        }
        return [{"name": "task", "type": "processor", "multiInstance": False,
                 "description": "自主桌面操作。根据接入的 RGB、夹爪和手臂状态逐步决策与复核。",
                 "topic_in": self._ports(), "topic_out": self._output(),
                 "configSchema": {"type": "object", "properties": config},
                 "inputSchema": {
                     "type": "object", "required": ["action"],
                     "properties": {
                         "action": {"type": "string", "enum": ["info", "config", "start", "stop", *actions]},
                         "instruction": {"type": "string"},
                         "request_id": {"type": "string", "description": "可选，重试时使用同一编号"},
                         "action_id": {"type": "string"}, "include_history": {"type": "boolean"}},
                     "x-action-params": actions, "x-is-dangerous": True,
                     "x-hooks": {"on_interrupt_all": {"action": "cancel"},
                                 "on_interrupt_motion": {"action": "cancel"}},
                     "x-resource": ["arm_l", "arm_r", "gripper_l", "gripper_r"],
                     "x-completion": {"actions": ["submit"], "timeout": _TIME_LIMIT + 240}}}]

    def dispatch(self, name, args):
        action = args.get("action", name)
        with self._lock:
            try:
                if action == "info":
                    fault = self._eef.fault if self._eef else None
                    state = "stopping" if self._stopping else "running" if self._node else "idle"
                    return {"state": "error" if fault else state,
                            "fault": fault,
                            "topic_in": self._ports(self._binding if args.get("instance_id") else None),
                            "topic_out": self._output(), "inputs": dict(self._binding),
                            "configured": all(self._cfg.get(k) for k in ("base_url", "key", "model_name")),
                            "arm_mode": self._cfg.get("arm_mode", "left"), "active_task": self._active}
                if action == "config":
                    if self._active:
                        raise ValueError("任务正在执行，暂不能修改配置")
                    changes = {k: args[k] for k in _FIELDS if k in args}
                    if changes and args.get("instance_id"):
                        raise ValueError("请在 autonomous_task 的全局设置中修改配置")
                    if any(not isinstance(v, str) for v in changes.values()):
                        raise ValueError("配置项必须为字符串")
                    if changes.get("arm_mode", "left") not in _MODES:
                        raise ValueError("arm_mode 需要是 left、right 或 both")
                    if any(k.endswith("_topic") and v and not v.startswith("/") for k, v in changes.items()):
                        raise ValueError("话题名必须为绝对 ROS 话题名")
                    if self._node and any(k.endswith("_topic") and v != self._cfg.get(k, "")
                                          for k, v in changes.items()):
                        raise ValueError("修改接线前请停止卡片")
                    self._cfg.update(changes)
                    return self.dispatch("info", {})
                if action == "start":
                    return self._start(args)
                if action == "stop":
                    if self._cancel:
                        self._stopping = True
                        self._interrupt()
                        return {"state": "stopping", "pending_task": self._active}
                    self._close()
                    return {"state": "idle", "pending_task": self._active}
                if action == "submit":
                    return self._submit(args)
                if action in ("cancel", "status"):
                    action_id = args.get("action_id") or self._active or next(reversed(self._runs), None)
                    if action_id is None:
                        return {"state": "idle"}
                    if action == "cancel" and action_id == self._active:
                        self._interrupt()
                    return self._status(action_id, args.get("include_history") is True)
            except (ValueError, RuntimeError, OSError, TypeError) as error:
                return {"state": "error", "message": str(error)}
        return None

    def _interrupt(self):
        if not self._cancel.is_set() and self._runs[self._active].get("phase") != "finalizing":
            self._cancel.set()
            self._runs[self._active]["state"] = "cancelling"
            self._eef.cancel()

    def _start(self, args):
        from rclpy.node import Node

        if self._stopping:
            raise ValueError("仍在等待上次任务结束")
        state_topic, cameras = resolve_inputs(args, self._cfg)
        control_topic = _CONTROL_TOPIC
        if control_topic in (state_topic, *cameras.values()):
            raise ValueError("控制输出不能与输入共用话题")
        interface = EefInterface(args.get("control_interface"))
        binding = {**cameras, "eef_state": state_topic}
        if self._node:
            if binding != self._binding or interface.descriptor != self._interface.descriptor:
                raise ValueError("卡片已运行，请先停止再更换接线")
            return self.dispatch("info", {**args, "action": "info"})
        self._node = Node("autonomous_task")
        try:
            self._cameras = CameraInputs(self._node, cameras)
            self._eef = EefEndpoint(self._node, state_topic, control_topic, interface)
            self._executor.add_node(self._node)
            self._binding = binding
            self._interface = interface
        except Exception:
            self._close()
            raise
        return self.dispatch("info", {"instance_id": args.get("instance_id")})

    def _close(self):
        if self._cameras:
            self._cameras.close()
        if self._eef:
            self._eef.close()
        if self._node:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        self._node = self._cameras = self._eef = self._interface = None
        self._binding = {}
        self._stopping = False

    def _submit(self, args):
        instruction = args.get("instruction")
        mode = self._cfg.get("arm_mode", "left")
        request_id = args.get("request_id")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("需要非空的自然语言 instruction")
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("request_id 必须为非空字符串")
            for run in self._runs.values():
                if run["request_id"] == request_id:
                    if instruction != run["instruction"] or args.get("arm_mode", run["arm_mode"]) != run["arm_mode"]:
                        raise ValueError("此 request_id 已用于其他任务")
                    return self._status(run["action_id"])
        if mode not in _MODES:
            raise ValueError("请在全局设置中选择有效的 arm_mode：left、right 或 both")
        # Older clients may still echo the configured side. Never switch arms on submit.
        if args.get("arm_mode", mode) != mode:
            raise ValueError(f"全局设置的 arm_mode 为 {mode}，请在 autonomous_task 的设置中切换")
        if self._active:
            raise ValueError(f"任务 {self._active} 正在执行")
        if not self._eef or not any(role in self._binding for role in ("head", "left_wrist", "right_wrist")):
            raise ValueError("请启动卡片并接入至少一路 RGB；头部、左腕、右腕均可选")
        if any(side not in self._interface.tip_frames for side in _MODES[mode]):
            raise ValueError(f"下游控制接口不支持 {mode} 模式")
        model = Model(*(self._cfg.get(k, "") for k in ("base_url", "key", "model_name")))
        self._eef.begin_task()
        action_id = "autonomous_task_" + uuid.uuid4().hex
        run = {"action_id": action_id, "request_id": request_id, "instruction": instruction,
               "arm_mode": mode, "state": "running", "step": 0, "phase": "starting",
               "started_at": time.time(), "history": []}
        model.model_log = ModelLog(self._model_log_root, run)
        run["model_log_path"] = str(model.model_log.path)
        if len(self._runs) >= 32:
            self._runs.popitem(last=False)
        self._runs[action_id] = run
        self._active, self._cancel = action_id, threading.Event()
        try:
            threading.Thread(target=self._work, args=(run, model, self._eef, self._cameras, self._cancel),
                             name="autonomous_task", daemon=True).start()
        except RuntimeError as error:
            self._active = self._cancel = None
            run.update(state="failed", phase="finished", reason=str(error), finished_at=time.time())
            model.model_log.append("task_end", status="failed", reason=str(error))
        return self._status(action_id)

    def _status(self, action_id, history=False):
        task = self._runs.get(action_id)
        if task is None:
            archive = self._cfg.get("record_dir")
            if not archive or not isinstance(action_id, str) or not re.fullmatch(r"autonomous_task_[0-9a-f]{32}", action_id):
                raise ValueError("未知 action_id")
            try:
                task = json.loads((Path(archive) / action_id / "result.json").read_text())
            except FileNotFoundError:
                raise ValueError("未知 action_id") from None
            if task.get("action_id") != action_id or task.get("state") not in ("completed", "failed", "cancelled"):
                raise ValueError("无有效的任务终态记录")
        out = {k: v for k, v in task.items() if k != "history"}
        if task["history"]:
            out["last"] = task["history"][-1]
        if history:
            out["history"] = list(task["history"])
        return out

    def _work(self, run, model, eef, cameras, cancel):
        action_id = run["action_id"]
        path = None
        status = "failed"
        reason = "任务未完成"
        try:
            if self._cfg.get("record_dir"):
                path = Path(self._cfg["record_dir"]) / action_id
                path.mkdir(parents=True, mode=0o700, exist_ok=True)
            status, reason = self._loop(run, model, eef, cameras, cancel, path)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            log.warning("%s step %d: %s", action_id, run["step"], reason)
        with self._lock:
            if eef.fault:
                status, reason = "failed", eef.fault
            elif cancel.is_set():
                status, reason = "cancelled", f"已停止后续决策。最后反馈：{reason}"
            else:
                # Retire this task's last goal before another task can be accepted.
                eef.cancel()
            run["phase"] = "finalizing"
        # Publish the end record before exposing the terminal state to exporters.
        try:
            model.model_log.append("task_end", status=status, reason=reason)
        except (OSError, ValueError) as error:
            log.exception("%s: cannot finalize model log", action_id)
            with self._lock:
                run["model_log_error"] = str(error)
            status, reason = "failed", f"{reason}；模型日志写入失败：{error}"
        with self._lock:
            run.update(state=status, phase="finished", reason=reason, finished_at=time.time())
            self._active = self._cancel = None
            if self._stopping:
                self._close()
        try:
            if path:
                pending = path / "result.tmp"
                pending.write_text(json.dumps(run, ensure_ascii=False, indent=2))
                pending.replace(path / "result.json")
        finally:
            _completion(action_id, status, reason)

    def _loop(self, run, model, driver, cameras, cancel, path):
        arms = _MODES[run["arm_mode"]]
        review = False
        inference_failures = 0
        invalid_decisions = 0
        deadline = time.monotonic() + _TIME_LIMIT

        def check():
            driver.check_fault()
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
            state = driver.observe(arms, cancel)
            if self._cfg.get("frame_description"):
                state["frame_description"] = self._cfg["frame_description"]
            if path:
                for role, jpeg in images.items():
                    (path / f"{step:03d}-{role}.jpg").write_bytes(jpeg)
            check()
            with self._lock:
                run["phase"] = "reviewing" if review else "deciding"
            decide = model.review if review else model.decide
            try:
                name, args = decide(run["instruction"], arms, images, state, run["history"])
            except InvalidDecisionError as error:
                invalid_decisions += 1
                self._record(run, path, {"step": step, "review": review, "tool": "invalid_decision",
                                         "note": error.visible_output, "model": model.last_model,
                                         "feedback": {"motion_sent": False, "reason": str(error)}}, state)
                if invalid_decisions >= 3:
                    return "failed", f"连续三次未得到可执行动作或明确任务状态：{error}"
                continue
            except TransientModelError as error:
                inference_failures += 1
                self._record(run, path, {"step": step, "review": review,
                                         "tool": "model_unavailable", "note": str(error),
                                         "feedback": {"motion_sent": False}}, state)
                if inference_failures >= 3:
                    return "failed", f"连续三次推理没有获得响应：{error}"
                cancel.wait(2 * inference_failures)
                continue
            inference_failures = invalid_decisions = 0
            record = {"step": step, "review": review, "tool": name, "args": args,
                      "note": model.last_note, "model": model.last_model}
            check()
            if name not in ("observe", "done", "failed"):
                arm, _, tool = name.partition("_")
                if review or arm not in arms or tool not in ("move_by", "set_gripper"):
                    raise ValueError(f"当前模式未提供 {name}")
                with self._lock:
                    run["phase"] = "executing"
                try:
                    record["feedback"] = driver.execute(name, args, cancel)
                except Exception as error:
                    record["feedback"] = {"status": "error", "reason": str(error)}
                    self._record(run, path, record, state, model=model, tool_error=error)
                    raise
                self._record(run, path, record, state, model=model)
            else:
                self._record(run, path, record, state)
            check()
            if record.get("feedback", {}).get("status") == "outcome_unknown":
                return "failed", record["feedback"]["reason"]
            if name == "failed":
                return "failed", args["reason"]
            if name == "done" and review:
                return "completed", args["reason"]
            review = name == "done"
        return "failed", f"已达到 {_STEP_LIMIT} 轮，未能复核任务完成"

    def _record(self, run, path, record, state, *, model=None, tool_error=None):
        with self._lock:
            run["history"].append(record)
        model_log = getattr(model, "model_log", None)
        if model_log is not None:
            # Capture the executor's outcome even if there will be no next inference.
            outcome = ({"payload": record["feedback"]} if tool_error is None else
                       {"error": {"type": type(tool_error).__name__, "message": str(tool_error)}})
            try:
                model_log.append("tool_result", step=record["step"], phase="decide",
                                 call_id=getattr(model, "last_call_id", None), name=record["tool"],
                                 arguments=record["args"], **outcome)
            except (OSError, ValueError, TypeError) as error:
                with self._lock:
                    run["model_log_error"] = str(error)
                raise RuntimeError(f"工具 {record['tool']} 已结束，工具结果日志写入失败：{error}") from error
        if path:
            with (path / "steps.jsonl").open("a") as trace:
                trace.write(json.dumps({**record, "before": state}, ensure_ascii=False) + "\n")
