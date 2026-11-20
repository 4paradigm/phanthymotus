"""Official Perception plugin + native Core workbench; explicitly PiPER only."""

from copy import deepcopy
from pathlib import Path
import threading

from jsonschema import Draft202012Validator

from .contracts import NAME, canonical, error_result, need
from .engine import Engine
from .schema import ACTION_FIELDS, OPTIONAL, OUTPUT_SCHEMA, input_schema
from .sources import Sources
from .configuration import validate


class GoalVerifierPlugin:
    PREFIX = NAME

    def __init__(self, plugin_cfg, executor=None, namespace=""):
        self.config = validate(plugin_cfg)
        self.executor = executor
        self.lock = threading.RLock()
        self.engine = Engine(self.config)
        self.sources = Sources(self.config, self.engine, vop_call=self._vop_call)
        self.image_exchange = None
        self.validator = Draft202012Validator(input_schema())
        self.started = False

    def get_tools(self):
        return [
            dict(
                name=NAME,
                type="processor",
                multiInstance=False,
                description="PiPER 操作结果视觉验证：推动/抓放/人工移动后独立判断物体目标，输出 PASS/FAIL/INCONCLUSIVE；不控制机械臂。",
                inputSchema=input_schema(),
                outputSchema=OUTPUT_SCHEMA,
                annotations=dict(readOnlyHint=True, destructiveHint=False),
                topic_in=[
                    dict(
                        format="data/json",
                        desc="带采集时间和相机身份的规范观测；原VOP发布时间不够",
                    )
                ],
                topic_out=[],
            )
        ]

    def start(self):
        """Bundle startup advertises only; never starts a camera automatically."""

    def stop(self):
        with self.lock:
            self.sources.close()
            self.engine.stop()
            self.started = False
            if self.image_exchange:
                self.image_exchange.clear()

    def _vop_call(self, binding, jpeg):
        from .server import rpc, loads

        need(self.image_exchange is not None, "VOP_IMAGE_EXCHANGE_REQUIRED")
        with self.image_exchange.expose(jpeg) as image_url:
            result = rpc(
                binding["vop_mcp_url"],
                "tools/call",
                dict(
                    name=binding.get("vop_tool", "vop"),
                    arguments=dict(action="recognize_by_url", url=image_url),
                ),
            )
            need(not result.get("isError"), "VOP_FAILED")
            result = loads(result["content"][0]["text"])
            need(
                result.get("ok") is True,
                "VOP_FAILED",
                result.get("detail") or result.get("reason"),
            )
            need(result.get("source") == image_url, "VOP_FRAME_REFERENCE_MISMATCH")
            return result

    def dispatch(self, name, args):
        action = None
        try:
            need(name == NAME, "WRONG_TOOL")
            args = deepcopy(args or {})
            need(isinstance(args, dict), "INVALID_ARGUMENTS")
            if "_tool_name" in args:
                need(args.pop("_tool_name") == NAME, "WRONG_TOOL")
            canonical(args)
            problem = next(self.validator.iter_errors(args), None)
            need(problem is None, "ARGUMENT_SCHEMA", problem.message if problem else "")
            action = args["action"]
            need(action in ACTION_FIELDS, "UNKNOWN_ACTION")
            need(not set(ACTION_FIELDS[action]) - args.keys(), "MISSING_FIELD")
            allowed = (
                set(ACTION_FIELDS[action])
                | set(OPTIONAL.get(action, ()))
                | {"action", "instance_id"}
            )
            need(not args.keys() - allowed, "ACTION_ARGUMENT_MISMATCH")
            args.pop("instance_id", None)
            args.pop("action")
            if action == "ui":
                return dict(
                    protocol="motus.workbench/1",
                    html=Path(__file__).with_name("workbench.html").read_text(),
                )
            if action == "info":
                return dict(
                    state="running" if self.started else "idle",
                    busy=False,
                    **self.engine.inspect(),
                    source_errors=deepcopy(self.sources.errors),
                    robot_model="agilex_piper",
                    motion_capability=False,
                    topic_out=[],
                    api_version="motus.piper-object-goal/1",
                )
            if action == "stop":
                self.stop()
                return dict(
                    state="stopped", sessions_retired=True, motion_executed=False
                )
            if action == "start":
                with self.lock:
                    if self.sources.closed:
                        self.sources = Sources(
                            self.config, self.engine, vop_call=self._vop_call
                        )
                    declared = args.get("input_topics") or []
                    configured = {
                        b.get("topic")
                        for b in self.config["sources"].values()
                        if b.get("topic")
                    }
                    need(
                        not declared or set(declared) <= configured,
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
                        self.engine.running = True
                        try:
                            self.sources.start_ros(self.executor)
                        except Exception:
                            self.sources.close()
                            self.engine.stop()
                            raise
                        self.started = True
                return dict(state="running", topic_out=[], motion_executed=False)
            if action == "submit_observation":
                self.engine.external_allowed(args["observation"])
                o = self.sources.push(args["observation"])
                return dict(
                    state="observed",
                    observation_id=o["observation_id"],
                    provenance=o["provenance"],
                )
            if action == "observe":
                need(
                    args["source_id"] in self.config.get("sources", {}),
                    "UNKNOWN_SOURCE",
                )
                o = self.sources.read(args["source_id"])
                self.engine.ingest(o)
                return dict(state="observed", observation=o)
            if action == "get_preview":
                need(
                    args["source_id"] in self.config.get("sources", {}),
                    "UNKNOWN_SOURCE",
                )
                return dict(
                    state="preview",
                    preview=self.sources.preview(
                        args["source_id"], args["observation_id"]
                    ),
                    message="关联观测的历史缩略图；规范数据来源可能没有图像，不能冒充实时画面",
                )
            if action not in ("get_result", "export_session"):
                cached = self.engine.replay(action, args)
                if cached is not None:
                    return cached
            observed = None
            if action in ("capture_before", "verify_after"):
                source, after = self.engine.acquisition(
                    args, after=action == "verify_after"
                )
                need(
                    not self.sources.errors.get(source),
                    "SOURCE_INPUT_ERROR",
                    self.sources.errors.get(source),
                )
                if "observation" not in args:
                    observed = self.sources.read(source, after=after)
            return self.engine.call(action, args, observed=observed)
        except Exception as exc:
            if action == "verify_after" and isinstance(args, dict):
                try:
                    return self.engine.record_failure(args, exc)
                except Exception:
                    pass
            result = error_result(exc)
            if isinstance(args, dict):
                result.update(
                    {k: args[k] for k in ("session_id", "request_id") if k in args}
                )
            return result
