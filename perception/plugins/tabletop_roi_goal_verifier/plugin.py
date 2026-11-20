"""Opt-in Perception processor and standalone Workbench; image comparison only."""

import threading
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator

from .configuration import validate
from .contracts import NAME, RESULT, canonical, error_result, need
from .engine import Engine
from .schema import ACTION_FIELDS, OPTIONAL, OUTPUT_SCHEMA, input_schema
from .sources import Sources


class TabletopROIGoalVerifierPlugin:
    PREFIX = NAME

    def __init__(self, plugin_cfg, executor=None, namespace=""):
        self.config = validate(plugin_cfg)
        self.executor = executor
        self.lock = threading.RLock()
        self.engine = Engine(self.config)
        self.sources = Sources(self.config, self.engine)
        self.validator = Draft202012Validator(input_schema())
        self.started = False
        self._stop_generation = 0
        self._signal_lock = threading.Lock()

    def get_tools(self):
        return [
            {
                "name": NAME,
                "type": "processor",
                "multiInstance": False,
                "description": "操作前后目标 ROI 稳定视觉变化验证，输出 PASS/FAIL/INCONCLUSIVE；无需物体检测，不控制机器人。",
                "inputSchema": input_schema(),
                "outputSchema": OUTPUT_SCHEMA,
                "annotations": {"readOnlyHint": True, "destructiveHint": False},
                "topic_in": [
                    {"format": "image/jpeg", "desc": "固定视角 ROS2 RGB 图像"}
                ],
                "topic_out": [],
            }
        ]

    def start(self):
        """Bundle startup only advertises this Card; subscription is explicit."""

    def _signal_stop(self):
        with self._signal_lock:
            self._stop_generation += 1
            self.sources.interrupt()

    def stop(self):
        self._signal_stop()
        with self.lock:
            try:
                self.sources.close()
            finally:
                self.engine.stop()
                self.started = False

    def dispatch(self, name, args):
        # Signal stop before the lock: an acquisition may be waiting for a frame.
        if name == NAME and isinstance(args, dict) and args.get("action") == "stop":
            self._signal_stop()
        generation = self._stop_generation
        # Serialize receipts and session commits, including concurrent retries.
        with self.lock:
            action = None
            try:
                need(name == NAME, "WRONG_TOOL")
                need(generation == self._stop_generation, "OPERATION_CANCELLED")
                args = deepcopy(args or {})
                need(isinstance(args, dict), "INVALID_ARGUMENTS")
                if "_tool_name" in args:
                    need(args.pop("_tool_name") == NAME, "WRONG_TOOL")
                canonical(args)
                problem = next(self.validator.iter_errors(args), None)
                need(
                    problem is None,
                    "ARGUMENT_SCHEMA",
                    problem.message if problem else "",
                )
                action = args.pop("action")
                need(not set(ACTION_FIELDS[action]) - args.keys(), "MISSING_FIELD")
                allowed = (
                    set(ACTION_FIELDS[action])
                    | set(OPTIONAL.get(action, ()))
                    | {"instance_id", "_trace_id"}
                )
                need(not args.keys() - allowed, "ACTION_ARGUMENT_MISMATCH")
                args.pop("instance_id", None)
                args.pop("_trace_id", None)
                if action == "ui":
                    return {
                        "protocol": "motus.workbench/1",
                        "html": Path(__file__).with_name("workbench.html").read_text(),
                    }
                if action == "info":
                    return dict(
                        state="running" if self.started else "idle",
                        busy=False,
                        **self.engine.inspect(),
                        source_errors=deepcopy(self.sources.errors),
                        motion_capability=False,
                        api_version=RESULT,
                        topic_out=[],
                    )
                if action == "stop":
                    self.stop()
                    return {
                        "state": "stopped",
                        "sessions_retired": True,
                        "motion_executed": False,
                    }
                if action == "start":
                    configured = {
                        b.get("topic")
                        for b in self.config["sources"].values()
                        if b.get("topic")
                    }
                    need(
                        not args.get("input_topics")
                        or set(args["input_topics"]) <= configured,
                        "CANVAS_SOURCE_BINDING_MISMATCH",
                    )
                    if args.get("camera_info"):
                        from ..camera_info import for_topic

                        for binding in self.config["sources"].values():
                            declaration = for_topic(
                                args["camera_info"], binding.get("topic")
                            )
                            if declaration:
                                need(
                                    declaration.get("schema") == "motus.camera/1"
                                    and declaration.get("id") == binding["camera_id"],
                                    "CAMERA_DECLARATION_MISMATCH",
                                )
                                self.sources.declared_sizes[binding["source_id"]] = [
                                    declaration.get("width"),
                                    declaration.get("height"),
                                ]
                    if not self.started:
                        if self.sources.closed or self.sources.cancelled.is_set():
                            self.sources = Sources(self.config, self.engine)
                        self.engine.running = True
                        try:
                            self.sources.start_ros(self.executor)
                            need(generation == self._stop_generation, "START_CANCELLED")
                        except Exception:
                            self.sources.close()
                            self.engine.stop()
                            raise
                        self.started = True
                    return {
                        "state": "running",
                        "topic_out": [],
                        "motion_executed": False,
                    }
                if action == "observe":
                    packet = self.sources.read_image(args["source_id"])
                    return {"state": "observed", "observation": packet["frame"]}
                if action == "get_preview":
                    need(args["source_id"] in self.config["sources"], "UNKNOWN_SOURCE")
                    return {
                        "state": "preview",
                        "preview": self.sources.preview(
                            args["source_id"], args["observation_id"]
                        ),
                    }
                if action not in ("get_result", "export_session"):
                    cached = self.engine.replay(action, args)
                    if cached is not None:
                        return cached
                observed = None
                if action in ("capture_before", "verify_after"):
                    source, after = self.engine.acquisition(
                        args, after=action == "verify_after"
                    )
                    if action == "capture_before":
                        observed = self.sources.read_image(source)
                    else:
                        _, options = self.engine.acquisition_mode(args)
                        observed = []
                        for _ in range(options["stable_frames"]):
                            packet = self.sources.read_image(source, after=after)
                            observed.append(packet)
                            after = packet["frame"]["captured_at"]
                result = self.engine.call(action, args, observed=observed)
                need(generation == self._stop_generation, "OPERATION_CANCELLED")
                return result
            except Exception as exc:  # noqa: BLE001 — return transport/source diagnostics
                if action == "verify_after" and isinstance(args, dict):
                    try:
                        return self.engine.record_failure(args, exc)
                    except Exception:  # noqa: BLE001, S110 — invalid session/conflicting receipt uses original error
                        pass
                result = error_result(exc)
                if isinstance(args, dict):
                    result.update(
                        {k: args[k] for k in ("session_id", "request_id") if k in args}
                    )
                return result
