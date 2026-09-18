"""The VLA card — one card, a pluggable provider, and whatever it is wired to.

Two axes, kept orthogonal on purpose, because changing the model and changing
the robot are unrelated events and a design that couples them needs an edit per
(model × robot) pair:

  **provider** — which model, and where it runs. Discovered from
  `providers/`, never enumerated here (see providers/__init__.py).

  **embodiment** — which robot. Not configured at all: the card drives whatever
  is connected to its output on the canvas. agent-core asks that card for its
  action space and hands it here as `control_interface` on `start`, the same way
  `canvas_binding.py` decides which MCPs the agent may reach. There is no
  `embodiment: "g1_dex1"` string to get wrong, and no URDF to misread as an
  action interface.

The card's own job is small: negotiate once, then pace a chunk out one command
at a time onto a `control/*` topic. Everything that keeps an arm safe lives in
the driver — `ControlSink` checks each command, holds on silence, and stops on
force. That split is deliberate: this process can crash, be OOM-killed, or lose
its network, and none of those must be what stops the robot.

Design: phanthymotus/docs/vla-integration.md
"""

from __future__ import annotations

import json
import logging
import threading
import time

from . import negotiate
from .message import build as build_message
from .providers import discover

log = logging.getLogger(__name__)

DEFAULT_TOPIC = "/actucore/vla/cmd"
# Which topic format to publish. Only the ones a driver can consume today; the
# card refuses anything else rather than publishing into a void.
FORMATS = {"joint_position": "control/joint",
           "joint_velocity": "control/joint-velocity",
           "joint_torque": "control/joint-torque",
           "twist": "control/velocity",
           "eef_pose": "control/waypoint"}


class Observation:
    """一帧观测。字段名就是 provider 协议里的那几个。

    刻意做成普通容器而不是 dataclass：provider 用 getattr 读它，云端那个也用
    同样的字段名往 motus.vla/1 的 payload 里塞，两边都不该依赖这里的类型。
    """

    __slots__ = ("images", "state", "prompt", "t_capture_ms")

    def __init__(self, images=None, state=None, prompt="", t_capture_ms=0):
        self.images = images or {}
        self.state = state
        self.prompt = prompt
        self.t_capture_ms = int(t_capture_ms or 0)


class VLAPlugin:
    PREFIX = "vla"        # no underscore — dispatch routes on partition("_")

    def __init__(self, plugin_cfg: dict, executor, namespace: str = ""):
        self._cfg = dict(plugin_cfg or {})
        self._executor = executor
        self._namespace = (namespace or "").strip("/")
        self._topic = self._cfg.get("topic") or DEFAULT_TOPIC

        # start/stop/config arrive on separate threads (ThreadingHTTPServer).
        # The lock guards bookkeeping only — never a provider construction, an
        # inference, or a node start/stop, or a stop would queue behind the
        # start it is meant to cancel.
        self._lock = threading.RLock()
        self._provider = None
        self._descriptor: dict = {}
        self._capabilities: dict = {}
        self._node = None
        self._publisher = None
        self._timer = None
        self._running = False
        # Provider construction is where a local checkpoint is *downloaded*, and
        # that runs inside `start` — several gigabytes on a cold /models. Without
        # this flag `_state()` answered "idle" for the whole of it, because
        # `_running` is only set afterwards: the card looked stopped while it was
        # in fact doing the longest thing it ever does.
        self._starting = False
        self._load_status = ""
        self._task = ""
        # Paused by `pause` or `interrupt`; both stop emitting, and the chunk
        # state above is what tells them apart. See `_halt`.
        self._paused = False

        # 观测。独立的锁：图像和状态回调来自 ROS 执行器线程，而 _tick 在定时器
        # 线程上读它们；和 _lock 共用会让发布排在一帧图像的解码后面。
        self._obs_lock = threading.RLock()
        self._images: dict = {}
        self._image_ms: dict = {}
        self._proprio = None
        self._state_ms = 0
        self._binding = None
        self._rate_hz = 30.0
        self._ttl_ms = 100
        self._seq = 0
        self._chunk: list = []
        self._chunk_index = 0
        self._chunk_obs_ms = 0
        self._last_error = ""
        self._published = 0

    # ── tools ────────────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        providers = discover()
        available = sorted(providers)
        # Each provider declares what kind of model name it takes, so the form
        # can offer the right control instead of one field pretending to serve
        # all three. Adding a provider is still adding a file.
        def _wanting(kind):
            return sorted(n for n, factory in providers.items()
                          if getattr(factory, "MODEL_NAMES", None) == kind)
        staged_providers = _wanting("staged")
        remote_providers = _wanting("remote")
        local_models = sorted(self._cfg.get("models") or {})
        default_model = self._cfg.get("model_name") or (
            local_models[0] if local_models else "")
        return [{
            "name": "vla",
            "type": "processor",
            "multiInstance": False,
            "description": "语言指令驱动的端到端策略；把动作流发到连上的驱动命令卡片",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["start", "stop", "execute", "interrupt",
                                        "pause", "resume", "info", "config"]},
                    "task": {"type": "string", "description": "自然语言指令"},
                    # Handed over by agent-core from the card wired downstream.
                    # Not operator-editable: it is a reading of the other card,
                    # and a hand-typed copy is a copy that goes stale.
                    "control_interface": {"type": "object"},
                },
                "required": ["action"],
                # Long-running and cancellable, but never "complete": a policy
                # runs until stopped. Declaring x-completion would hold an ACP
                # pending open for the life of the card and block every other
                # actuator behind the barrier.
                # This list is the model's entire reach: agent-core splits a
                # tool into one LLM-callable function per entry (mcp_client.py
                # `_to_openai_schema`), so an action absent here is callable by
                # the canvas and not by the model. `start`/`stop` are absent
                # deliberately — they belong to the project lifecycle, and
                # `start` needs the downstream `control_interface`, which only
                # agent-core can supply. A model that stopped this card could
                # not start it again.
                #
                # The three that are here are the levers over a policy that is
                # already running, and the two halts are genuinely different
                # because this card emits *chunks* — a plan several steps into
                # the future:
                #
                #   pause      stop emitting, keep the chunk. Resume continues
                #              the plan the policy already made.
                #   interrupt  stop emitting and throw the chunk away. Resume
                #              re-infers from the world as it is now.
                #
                # That distinction is the whole reason both exist. "Hold on a
                # second" and "no, stop, that is wrong" are different
                # instructions, and carrying out the second by replaying a plan
                # made before the objection would be the wrong answer.
                "x-action-params": {
                    "execute": {"params": ["task"],
                                "description": "下达自然语言指令并开始执行；"
                                               "已在执行时即为换指令"},
                    "pause": {"params": [],
                              "description": "暂停执行，保留策略已经规划好的动作块；"
                                             "resume 从原计划继续"},
                    "interrupt": {"params": [],
                                  "description": "立即停止并丢弃已规划的动作块；"
                                                 "resume 会基于当前情况重新推理"},
                    "resume": {"params": [], "description": "继续执行"},
                },
                "x-hooks": {"on_interrupt_all": {"action": "interrupt"},
                            "on_interrupt_motion": {"action": "interrupt"}},
                "x-is-dangerous": True,
                "x-resource": self._resources(),
            },
            "configSchema": {
                "type": "object",
                "properties": {
                    # Built from what was discovered, not written down — adding
                    # a provider file is the whole of adding a provider.
                    "provider": {"type": "string", "enum": available,
                                 "default": "mock" if "mock" in available else
                                 (available[0] if available else ""),
                                 "scope": "shared"},
                    # Which checkpoint. For a local provider this selects one
                    # of the staged models under `models:`; for vla_cloud it is
                    # the name the server knows it by. One field either way —
                    # the question "which model" is the same question, and
                    # splitting it would give the same thing two names.
                    #
                    # The published checkpoints keep their upstream names
                    # (`smolvla_base`); a version fine-tuned for a particular
                    # robot gets a name that says so (`smolvla_tianyi`,
                    # `smolvla_q5`), because the action space it fits is the
                    # thing an operator has to get right.
                    # Two fields, not one, because they are two different
                    # controls and the form cannot switch a field between them:
                    # `enum` renders as a <select>, so a single field carrying
                    # the staged names would leave an operator on vla_cloud
                    # unable to type the name their server knows. That is worse
                    # than the cosmetic problem it would have solved.
                    #
                    # No `description`/`title` on either: the form renders
                    # `title || description || key` as the label with no separate
                    # hint element, so prose here replaces the field name rather
                    # than accompanying it. The explanation lives in config.yaml.
                    "model_name": {"type": "string", "default": default_model,
                                   "scope": "shared",
                                   "x-show-when": {"provider": staged_providers},
                                   **({"enum": local_models} if local_models else {})},
                    "cloud_model_name": {"type": "string", "scope": "shared",
                                         "x-show-when": {"provider": remote_providers}},
                    # Only vla_cloud has anywhere to send a request. Hiding
                    # these for a local provider is not cosmetic: a filled-in
                    # endpoint beside `provider: smolvla` reads as configured
                    # and is ignored, which is the kind of thing an operator
                    # spends an afternoon on.
                    "endpoint": {"type": "string", "scope": "shared",
                                 "x-show-when": {"provider": remote_providers}},
                    "api_key": {"type": "string", "scope": "shared",
                                "x-show-when": {"provider": remote_providers}},
                    "timeout_ms": {"type": "number", "default": 500,
                                   "scope": "shared",
                                   "x-show-when": {"provider": remote_providers}},
                    "topic": {"type": "string", "default": DEFAULT_TOPIC,
                              "scope": "instance"},
                    "rate_hz": {"type": "number", "scope": "instance"},
                    "priority": {"type": "integer", "default": 50,
                                 "scope": "instance"},
                },
                "required": [],
            },
            # `topic` is declared, not left to be discovered after start —
            # every other card in the project does this and this one did not.
            #
            # agent-core resolves a consumer's `input_topic` from its source's
            # *running* `info()`, falling back to the connection's persisted
            # `fromTopic` and then to this declaration. A card that names no
            # topic until it starts leaves all three empty, so the moment this
            # card fails to start — for any reason — the card downstream fails
            # too, with "连线缺少 topic"， which reads like a wiring problem on a
            # canvas that is wired correctly. Seen on Tianyi.
            #
            # There is nothing to discover anyway: the topic comes from config
            # and is known here. It also makes the feedback wiring
            # (vla → servo → vla) resolvable, since in a cycle neither card can
            # learn its input from a source that has already started.
            "topic_out": [{"topic": self._topic, "format": self._format(),
                           "desc": "motus.control/1 命令流"}],
            # 观测。连哪几路由 provider 的 capabilities() 说了算 —— mock 声明
            # n_cameras=0/needs_state=false，开环，不连也能跑；smolvla 两样都要，
            # 缺了会在启动时被拒绝，而不是跑起来空转。见 _bind_inputs。
            "topic_in": [
                {"format": "image/jpeg", "desc": "主视角相机"},
                {"format": "state/joint", "desc": "本体状态（接驱动命令卡片的状态输出）"},
            ],
        }]

    def dispatch(self, name: str, args: dict):
        action = args.get("action") or name
        if action == "start":
            return self._start(args)
        if action == "stop":
            return self._stop()
        if action == "execute":
            return self._execute(args)
        if action == "pause":
            return self._halt(drop_chunk=False)
        if action == "interrupt":
            return self._halt(drop_chunk=True)
        if action == "resume":
            return self._resume()
        if action == "info":
            return self._info()
        if action == "config":
            return self._config(args)
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Bundle lifecycle. Deliberately inert — same rule as the driver card.

        A policy must not resume because a container restarted. It starts when
        someone wires it and asks.
        """

    def stop(self):
        self._stop()

    # ── actions ──────────────────────────────────────────────────────────────

    def _start(self, args: dict):
        descriptor = args.get("control_interface") or {}
        if not descriptor:
            return self._error(
                "没有拿到下游动作空间 —— 请把本卡片的输出连到一张驱动命令卡片"
                "（control/* 端口），agent-core 会在启动时把它的 descriptor 传过来")

        provider_name = self._cfg.get("provider") or "mock"
        providers = discover()
        if provider_name not in providers:
            detail = discover.errors.get(provider_name)
            return self._error(
                f"provider {provider_name!r} 不可用"
                + (f"：{detail}" if detail else f"，可用的有 {sorted(providers)}"))

        # Deliberately outside `_lock` — the comment on that lock says it guards
        # bookkeeping only, never a provider construction, or a stop would queue
        # behind the download it is meant to cancel.
        self._starting = True
        self._load_status = ""
        try:
            provider = providers[provider_name](
                descriptor, self._cfg,
                on_status=lambda text: setattr(self, "_load_status", text))
            capabilities = provider.capabilities()
        except Exception as error:      # noqa: BLE001
            return self._error(f"provider {provider_name} 初始化失败：{error}")
        finally:
            # Cleared on the failure path too: one failed start must not leave
            # every later info() claiming to be loading.
            self._starting = False
            self._load_status = ""

        problems = negotiate.check(capabilities, descriptor)
        if problems:
            self._close(provider)
            # Refused rather than started and left to fail per command: at
            # 30 Hz the second outcome is a stopped robot with no reason given.
            return self._error("模型与下游动作空间不匹配：" + "；".join(problems))

        mode = descriptor.get("mode")
        if mode not in FORMATS:
            self._close(provider)
            return self._error(f"下游 mode {mode!r} 还没有对应的 topic 格式")

        rate = negotiate.effective_rate(capabilities, descriptor,
                                        self._cfg.get("rate_hz"))
        with self._lock:
            if self._running:
                return self._error("already running")
            self._provider = provider
            self._descriptor = descriptor
            self._capabilities = capabilities
            self._task = args.get("task") or self._cfg.get("task") or ""
            self._rate_hz = rate
            self._ttl_ms = negotiate.ttl_ms(rate, descriptor)
            self._seq = 0
            self._chunk = []
            self._chunk_index = 0
            self._paused = False
            self._published = 0
            self._last_error = ""
            self._running = True

        # 连上来的观测源。单数形式是 agent-core 对单连接的写法，复数是多连接；
        # 两个都读并取并集，因为它同时发两者（见 agent-core _start_and_resolve
        # 里那段关于只发复数会让卡片"绑定到空"的注释）。
        inputs = list(args.get("input_topics") or [])
        single = args.get("input_topic")
        if single and single not in inputs:
            inputs.insert(0, single)

        try:
            self._open_publisher(mode)
            binding, problem = self._bind_inputs(self._node, inputs, capabilities)
        except Exception as error:      # noqa: BLE001
            with self._lock:
                self._running = False
                self._provider = None
            self._close_node()
            self._close(provider)
            return self._error(f"publisher 启动失败：{error}")

        if problem:
            with self._lock:
                self._running = False
                self._provider = None
            self._close_node()
            self._close(provider)
            # 启动就拒绝，而不是跑起来一条指令都不发：后者的原因只留在
            # info().error 里，而画布上那张卡看着是 running 的。
            return self._error(problem)
        self._binding = binding

        log.info("vla started: provider=%s topic=%s %.1f Hz ttl=%d ms task=%r",
                 provider_name, self._topic, rate, self._ttl_ms, self._task)
        result = {"state": self._state(), "topic": self._topic, "rate_hz": rate,
                  "ttl_ms": self._ttl_ms, "provider": provider_name,
                  "capabilities": capabilities}
        if result["state"] == "loading":
            # agent-core keeps the card visibly starting and polls info() until
            # it settles. Reporting ready here is how an operator ends up with a
            # card that claims ready seconds before it can act.
            result["message"] = "模型加载中，就绪后自动开始发布"
        return result

    def _execute(self, args: dict):
        """下达指令。这是模型对这张卡唯一的输入口。

        换指令**必然丢掉在飞的动作块**。块是为上一条指令算出来的一段未来计划；
        留着它执行，等于拿旧指令的计划去做新指令，而且前几十毫秒的动作会看起来
        完全合理 —— 这正是最难发现的那种错。所以这里走的是 interrupt 那条路，
        不是 pause。

        同时解除暂停：说"去把杯子递给我"的人，意思不会是"记下但先别动"。
        """
        task = (args.get("task") or "").strip()
        if not task:
            return self._error("execute 需要 task —— 一句自然语言指令")
        with self._lock:
            if not self._running:
                return {"state": "idle",
                        "message": "卡片未在运行，请先在画布上启动"}
            self._task = task
            self._chunk = []
            self._chunk_index = 0
            self._paused = False
        log.info("vla execute: task=%r", task)
        return {"state": self._state(), "task": task, "topic": self._topic}

    def _halt(self, *, drop_chunk: bool):
        """`pause` (keep the plan) and `interrupt` (throw it away).

        Both stop emitting immediately. Neither unsubscribes or tears the card
        down: the receiving driver's watchdog sees the silence and holds the
        arm within `watchdog_ms`, which is the correct resting state, and the
        project's view of what is running stays true.

        The difference is what `resume` then does, and it matters because this
        card emits a chunk — a plan made some steps ago. Resuming a *pause*
        replays the rest of that plan, which is right for "hold on". Resuming
        an *interrupt* re-infers, which is right for "no, that is wrong":
        replaying a plan made before the objection is exactly what the
        objection was about.
        """
        with self._lock:
            if not self._running:
                return {"state": "idle", "message": "卡片未在运行"}
            self._paused = True
            if drop_chunk:
                self._chunk = []
                self._chunk_index = 0
        return {"state": "interrupted" if drop_chunk else "paused",
                "topic": self._topic,
                "chunk_pending": 0 if drop_chunk
                                 else max(0, len(self._chunk) - self._chunk_index)}

    def _resume(self):
        with self._lock:
            if not self._running:
                return {"state": "idle", "message": "卡片未在运行"}
            self._paused = False
        return {"state": "running", "topic": self._topic}

    def _stop(self):
        with self._lock:
            node, self._node = self._node, None
            timer, self._timer = self._timer, None
            provider, self._provider = self._provider, None
            was_running, self._running = self._running, False
            self._publisher = None

        if timer is not None and node is not None:
            try:
                node.destroy_timer(timer)
            except Exception:           # noqa: BLE001 — teardown is best effort
                pass
        if node is not None:
            try:
                if self._executor is not None:
                    self._executor.remove_node(node)
            finally:
                # destroy_node, not only remove_node: otherwise the publisher
                # and the ROS node name leak and a restart collides with itself.
                node.destroy_node()
        self._close(provider)
        if was_running:
            log.info("vla stopped (%s)", self._topic)
        # Stopping publishing is not stopping the robot. The driver's watchdog
        # is what brings it to rest, which is the point of it being there.
        return {"state": "idle"}

    def _info(self):
        with self._lock:
            return {
                "state": self._state(),
                "topic": self._topic,
                "format": self._format(),
                "provider": self._cfg.get("provider") or "mock",
                "providers_available": sorted(discover()),
                "providers_unavailable": dict(discover.errors),
                "task": self._task,
                "rate_hz": self._rate_hz,
                "ttl_ms": self._ttl_ms,
                "published": self._published,
                "capabilities": dict(self._capabilities),
                "control_interface": dict(self._descriptor),
                "error": self._last_error,
                # Present only while something is being fetched, so a caller can
                # tell "still loading" from "loading, 40% of 906 MB in".
                **({"message": self._load_status} if self._load_status else {}),
            }

    def _config(self, args: dict):
        for key, value in (args or {}).items():
            if key in ("action", "instance_id"):
                continue
            self._cfg[key] = value
        if "topic" in (args or {}):
            self._topic = args["topic"] or DEFAULT_TOPIC
        return {"state": "running" if self._running else "idle", "config": dict(self._cfg)}

    # ── publishing ───────────────────────────────────────────────────────────

    # ── 观测输入 ─────────────────────────────────────────────────────────────

    def _bind_inputs(self, node, topics, capabilities):
        """把连上来的话题按**消息类型**分派到角色，并对账 provider 的需求。

        agent-core 只传话题名，不传格式，所以角色不能靠名字猜 —— 一个叫
        `/robot/state` 的话题完全可能是别的东西。问 ROS 图它上面发的是什么类型
        是确定的答案，项目里已有先例（t800、lynx_m20 都这么查图）。

        对账放在这里、放在启动时，是因为 provider 早就如实声明了 needs_state /
        n_cameras，而在此之前**全项目没有一个地方读它们**。于是"选了 smolvla 却
        没连相机"会一路跑起来、报 running、一条指令都不发，原因只藏在
        info().error 里。缺什么就在启动时说清楚，是这条对账唯一的意义。
        """
        # ROS 的 import 留到真要建订阅时 —— 角色判断只看类型名的字符串，而
        # "什么都没连所以拒绝"这条路不该需要一个 ROS 环境才能走到。
        by_type = dict(node.get_topic_names_and_types())
        bound, unknown = {"images": {}, "state": None}, []
        for topic in topics:
            types = by_type.get(topic) or []
            if any(name.endswith(("CompressedImage", "Image")) for name in types):
                bound["images"][f"cam{len(bound['images'])}"] = topic
            elif any(name.endswith("String") for name in types):
                bound["state"] = topic
            else:
                unknown.append(f"{topic}({'/'.join(types) or '无发布者'})")

        needs_state = bool(capabilities.get("needs_state"))
        n_cameras = int(capabilities.get("n_cameras") or 0)
        missing = []
        if n_cameras > len(bound["images"]):
            missing.append(f"相机 {len(bound['images'])}/{n_cameras} 路")
        if needs_state and not bound["state"]:
            missing.append("本体状态")
        if missing:
            return None, ("模型需要的观测没有连上：" + "、".join(missing)
                          + "。请在画布上把相机卡片、以及驱动命令卡片的状态输出"
                            "连到本卡片的输入端口。"
                          + (f"（无法识别的输入：{'、'.join(unknown)}）" if unknown else ""))

        # 每种消息类型只在真的要订它时才 import：开环的 provider（mock）不连
        # 任何东西也能跑，不该因为进程里没有 sensor_msgs 就起不来。
        if bound["images"]:
            from sensor_msgs.msg import CompressedImage, Image

            for name, topic in bound["images"].items():
                types = by_type.get(topic) or []
                message_type = CompressedImage if any(
                    n.endswith("CompressedImage") for n in types) else Image
                node.create_subscription(
                    message_type, topic,
                    lambda message, key=name: self._on_image(key, message), 1)
        if bound["state"]:
            from std_msgs.msg import String

            node.create_subscription(
                String, bound["state"], self._on_state, 1)
        return bound, ""

    def _on_image(self, name, message):
        data = bytes(getattr(message, "data", b"") or b"")
        if not data:
            return
        with self._obs_lock:
            self._images[name] = data
            self._image_ms[name] = self._stamp_of(message)

    def _on_state(self, message):
        try:
            payload = json.loads(message.data)
        except Exception:      # noqa: BLE001 — 一帧坏数据不该拖垮流
            return
        values = payload.get("values")
        if not isinstance(values, list):
            return
        with self._obs_lock:
            self._proprio = [float(v) for v in values]
            self._state_ms = int(payload.get("stamp_ms") or 0) or int(time.time() * 1000)

    @staticmethod
    def _stamp_of(message):
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return int(time.time() * 1000)
        return int(stamp.sec * 1000 + stamp.nanosec // 1_000_000)

    def observation(self):
        """当前观测，或 None —— provider 要而没有的东西缺一样就返回 None。

        `t_capture_ms` 取参与这一帧的各路里**最旧**的那个。RTC 的
        inference_delay 是按观测年龄算的，报现在等于宣称所有通道刚刚更新过，
        而实际最旧的那路可能已经很陈旧 —— 那会让补偿按错误的时间做。
        """
        with self._obs_lock:
            images = dict(self._images)
            image_ms = dict(self._image_ms)
            state = list(self._proprio) if self._proprio is not None else None
            state_ms = self._state_ms

        capabilities = self._capabilities or {}
        if int(capabilities.get("n_cameras") or 0) > len(images):
            return None
        if capabilities.get("needs_state") and state is None:
            return None

        stamps = [ms for ms in image_ms.values() if ms]
        if state_ms:
            stamps.append(state_ms)
        return Observation(images=images, state=state, prompt=self._task,
                           t_capture_ms=min(stamps) if stamps else 0)

    def _close_node(self):
        """拆掉 ROS 节点。start 半途失败时必须走这一步 —— 留下一个已注册到
        executor 的节点，下一次 start 会撞上同名节点而失败，而症状（"节点名已
        存在"）和真正的原因（上一次启动没清干净）看不出关系。"""
        with self._lock:
            node, self._node = self._node, None
            self._publisher = self._timer = None
        if node is None:
            return
        try:
            if self._executor is not None:
                self._executor.remove_node(node)
            node.destroy_node()
        except Exception:      # noqa: BLE001
            pass

    def _open_publisher(self, mode: str):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            # One deep: a queued command is a stale command, and the receiver
            # would drop it on ttl anyway.
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        node = Node("actucore_vla")
        publisher = node.create_publisher(String, self._topic, qos)
        timer = node.create_timer(1.0 / self._rate_hz, self._tick)
        if self._executor is not None:
            self._executor.add_node(node)
        with self._lock:
            self._node, self._publisher, self._timer = node, publisher, timer

    def _tick(self):
        publisher = self._publisher
        if publisher is None or not self._running or self._paused:
            # Emitting nothing is how a halt reaches the arm: the driver's
            # watchdog holds within watchdog_ms. Nothing here needs to command
            # a stop, and commanding one would fight the hold.
            return
        provider = self._provider
        # A provider that loads its weights in the background is not an error
        # while it does so — it simply has nothing to say yet. Publishing
        # nothing keeps the receiver's watchdog holding the arm, which is the
        # right state for "no policy is driving".
        if provider is not None and not provider.health():
            return
        # 需要观测却还没有，就什么都不发。下游的 watchdog 会 hold 住手臂，那是
        # "没有策略在驱动"的正确状态；拿上一条指令顶着才是危险的。
        capabilities = self._capabilities or {}
        if (int(capabilities.get("n_cameras") or 0) or capabilities.get("needs_state")) \
                and self.observation() is None:
            return
        try:
            message = self.next_command()
        except Exception as error:      # noqa: BLE001
            with self._lock:
                self._last_error = f"{type(error).__name__}: {error}"
            log.warning("vla inference failed: %s", error)
            # Publishing nothing is the correct failure: the receiver's watchdog
            # brings the arm to rest, whereas a repeated last command would keep
            # driving it on a policy that is no longer running.
            return
        from std_msgs.msg import String
        payload = String()
        payload.data = json.dumps(message, ensure_ascii=False)
        publisher.publish(payload)
        with self._lock:
            self._published += 1

    def next_command(self) -> dict:
        """One command, refilling the chunk when it runs out.

        Public and ROS-free so the whole of the wire format — sequence numbers,
        the two timestamps, chunk indices — is testable without a node. `_tick`
        is only this plus a publish.
        """
        if self._chunk_index >= len(self._chunk):
            # The observation timestamp belongs to the moment the chunk was
            # computed, and every command paced out of it carries that same
            # value while its own stamp advances. That is what makes an ageing
            # chunk visible downstream instead of looking perpetually fresh.
            observation = self.observation()
            # 观测的采集时刻，不是现在 —— 这条 chunk 里每一条指令都带着它，
            # 下游据此判断指令有多陈旧。用 now() 会让一条基于 800ms 前画面算出
            # 的指令看起来永远新鲜。
            self._chunk_obs_ms = (observation.t_capture_ms if observation
                                  and observation.t_capture_ms
                                  else int(time.time() * 1000))
            self._chunk = list(self._provider.infer(observation) or [])
            self._chunk_index = 0
            if not self._chunk:
                raise RuntimeError("provider returned an empty chunk")

        values = self._chunk[self._chunk_index]
        index, size = self._chunk_index, len(self._chunk)
        self._chunk_index += 1
        self._seq += 1
        return build_message(
            seq=self._seq,
            values=values,
            mode=self._descriptor.get("mode", "joint_position"),
            dof=int(self._descriptor.get("dof") or len(values)),
            source=f"mcp__actucore__{self.PREFIX}",
            stamp_ms=int(time.time() * 1000),
            obs_stamp_ms=self._chunk_obs_ms,
            ttl_ms=self._ttl_ms,
            priority=int(self._cfg.get("priority", 50)),
            chunk_index=index,
            chunk_size=size,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _state(self) -> str:
        """idle | loading | paused | running.

        `loading` is a real state, not a nicety: a local provider reads its
        checkpoint's config in milliseconds and its weights in seconds, and
        agent-core has a watcher for exactly this (`api/config.py`
        `_settle_loading_item`). Skipping it would report ready while the card
        still cannot produce a command.
        """
        if self._starting:
            # Downloading or constructing. agent-core's `_settle_loading_item`
            # watcher polls info() until this settles, which is exactly the
            # behaviour a multi-gigabyte fetch wants.
            return "loading"
        if not self._running:
            return "idle"
        provider = self._provider
        if provider is not None and not provider.health():
            return "loading"
        # Its own state rather than "running": an operator reading running
        # beside a motionless arm goes looking for a fault that is not there.
        # `loading` still wins — a paused card whose weights are not in yet is
        # not ready either, and saying "paused" would claim it is.
        if self._paused:
            return "paused"
        return "running"

    def _format(self) -> str:
        return FORMATS.get(self._descriptor.get("mode"), "control/joint")

    def _resources(self) -> list:
        """Physical channels this card occupies, for the ACP barrier.

        Taken from the negotiated descriptor's `groups` once there is one: only
        the downstream driver knows what it actually owns. Tianyi's action space
        is four channels (both arms, both hands); a card that kept claiming a
        single configured `arm` would let something else drive the hands while a
        policy was moving them.

        Before `start` there is no descriptor — the tool list is fetched long
        before anything is wired — so the configured value stands in. agent-core
        re-reads the schema on heartbeat (`api/mcp_manage.py` extracts
        `x-resource` there as well as at registration), so the negotiated set
        replaces it shortly after the card starts.
        """
        groups = self._descriptor.get("groups") or []
        negotiated = []
        for group in groups:
            resource = (group or {}).get("resource")
            if resource and resource not in negotiated:
                negotiated.append(resource)
        if negotiated:
            return negotiated

        configured = self._cfg.get("resource") or "arm"
        return configured if isinstance(configured, list) else [configured]

    def _error(self, message: str) -> dict:
        with self._lock:
            self._last_error = message
        log.warning("vla: %s", message)
        return {"state": "error", "message": message}

    @staticmethod
    def _close(provider):
        if provider is None:
            return
        try:
            provider.close()
        except Exception as error:      # noqa: BLE001
            log.warning("vla provider close failed: %s", error)
