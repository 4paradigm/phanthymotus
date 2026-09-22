"""The navi card — go to a thing you can see.

Three wires in (what is there, how far away it is, and optionally how fast we
are actually going) and one command stream out. The card's own job is small:
negotiate once at start, then run `policy.step` on a timer and publish what it
returns. All of the behaviour is in `policy.py`, all of the geometry in
`depth.py`, and both are ROS-free so they can be tested without a robot.

Built on the same skeleton as `plugins/vla/plugin.py`, including the several
rules in it that were learned the hard way and apply verbatim here:

* the downstream action space arrives as `control_interface` on `start`; with
  no downstream card there is nothing to negotiate against and the card refuses
  rather than publishing into a void;
* `negotiate.check` failing means **refuse to start**, not start and fail once
  per command — at 10 Hz the second outcome is a stopped robot with no reason
  given anywhere an operator looks;
* `input_topic` and `input_topics` are both read and unioned, because
  agent-core sends both and reading only the plural binds the card to nothing;
* inputs are assigned to roles by **what is on the topic**, never by its name;
* `start`/`stop` are absent from `x-action-params`: agent-core turns each entry
  into an LLM-callable function, and this card's `start` needs both an input
  topic and a downstream descriptor that only agent-core can supply. A model
  that stopped it could not start it again.

── one way this card is not like the vla card ───────────────────────────────

`navigate_to` **completes**. A policy runs until stopped, which is why the vla
card cannot declare `x-completion`; arriving somewhere is a finite action with
an answer, so this one does, and pushes the ACP callback when it gets there.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid

from ..control_stream import build as build_message
from ..control_stream import negotiate
from . import depth as depth_mod
from . import odom as odom_mod
from . import policy as policy_mod

log = logging.getLogger(__name__)

DEFAULT_TOPIC = "/actucore/navi/cmd"
CONTROL_MODE = "twist"
ACTION_DIM = 6
# The format the downstream port must speak. Only one, unlike the vla card:
# a body twist is the only thing this card knows how to produce.
TOPIC_FORMAT = "control/velocity"


def _sensor_qos():
    """BEST_EFFORT, to match what every producer this card reads actually uses.

    **A reliable subscriber does not match a best-effort publisher.** ROS2 treats
    that as an incompatible QoS pair and simply never delivers — no error on
    either side, no warning in any log, and `ros2 topic info` still shows one
    publisher and one subscription. The card reports "还没有收到任何检测结果"
    while vop is demonstrably publishing, which reads as a broken camera.

    Passing a bare `1` for the depth argument is what produced that: the integer
    form is shorthand for the *default* profile, and the default is RELIABLE.
    Every producer here is best-effort and deliberately so — perception's vop and
    visual_depth both use `_PUB_QOS`, the R1 driver's loco_state uses
    `_LOW_LAT_QOS`, and all three would rather drop a frame than block on one.

    Depth 2 rather than 10: this card only ever reads the newest sample, and a
    deeper queue just means the frame it eventually processes is older.
    """
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)

    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST,
                      depth=2,
                      durability=DurabilityPolicy.VOLATILE)


class NaviPlugin:
    PREFIX = "navi"          # no underscore — dispatch routes on partition("_")

    def __init__(self, plugin_cfg: dict, executor, namespace: str = ""):
        self._cfg = dict(plugin_cfg or {})
        self._executor = executor
        self._namespace = (namespace or "").strip("/")
        self._topic = self._cfg.get("topic") or DEFAULT_TOPIC

        # Guards bookkeeping only — never a node start/stop, or a stop would
        # queue behind the start it is meant to cancel. Same rule as every
        # other plugin here that holds per-instance state.
        self._lock = threading.RLock()
        self._descriptor: dict = {}
        self._node = None
        self._publisher = None
        self._timer = None
        self._running = False
        self._paused = False
        self._rate_hz = 10.0
        self._ttl_ms = 300
        self._seq = 0
        self._published = 0
        self._last_error = ""
        self._last_decision = None
        self._last_tick = 0.0

        self._config = policy_mod.Config(**{
            k: v for k, v in self._cfg.items()
            if k in policy_mod.Config.__dataclass_fields__
        })
        self._state = policy_mod.State()

        # Observations. A separate lock from `_lock`: these are written by ROS
        # executor threads and read by the timer thread, and sharing would put
        # a publish behind the decode of a depth frame.
        self._obs_lock = threading.RLock()
        self._objects = None
        self._objects_ms = 0
        # 最近 N 帧的检测结果，给 list_visible_objects 求交集用。deque 而不是
        # list：这是热路径上每帧都写的东西，而且必须有界 —— 一个跑了一小时的
        # 卡片不该攒着三万帧检测结果。
        from collections import deque
        self._recent = deque(maxlen=int(self._cfg.get("stable_frames", 10)))
        self._depth_map = None
        self._depth_bands = {}
        self._depth_ms = 0
        self._odom = None
        self._odom_ms = 0
        self._binding = {}
        self._acp_action_id = ""

    # ── tool ─────────────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        return [{
            "name": "navi",
            "type": "processor",
            "multiInstance": False,
            "description": ("视觉导航：看到目标就朝它走过去，按深度减速避障，"
                            "到设定距离停下。接 vop 的检测结果和一路深度图。"),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["start", "stop", "list_visible_objects",
                                        "navigate_to", "stop_navigation",
                                        "pause", "resume", "info", "config"]},
                    "target": {"type": "string",
                               "description": "要走过去的东西。用 list_visible_objects "
                                              "返回的 key（如 chair#azure）最准；"
                                              "也接受纯名字，但同名多个时挑哪一个不保证"},
                    "stop_distance_m": {"type": "number",
                                        "description": "在目标前多远停下，默认 1.2 m"},
                    # Handed over by agent-core from the card wired downstream.
                    # Not operator-editable: it is a reading of another card,
                    # and a hand-typed copy is a copy that goes stale.
                    "control_interface": {"type": "object"},
                },
                "required": ["action"],
                "x-action-params": {
                    "list_visible_objects": {
                        "params": [],
                        "description": "列出当前稳定看到的物体。先调它，再从返回的 "
                                       "key 里挑一个给 navigate_to —— 单帧检测会闪，"
                                       "直接凭印象填名字很可能指向一个下一帧就不在的东西",
                    },
                    "navigate_to": {
                        "params": ["target", "stop_distance_m"],
                        "description": "朝一个看得见的目标走过去；已在导航时即为换目标",
                    },
                    "stop_navigation": {"params": [],
                                        "description": "停止导航，放弃当前目标"},
                    "pause": {"params": [], "description": "暂停，保留目标"},
                    "resume": {"params": [], "description": "继续"},
                },
                # Unlike the vla card, this one finishes: arriving is an answer.
                "x-completion": {"actions": ["navigate_to"], "timeout": 300},
                "x-hooks": {"on_interrupt_all": {"action": "stop_navigation"},
                            "on_interrupt_motion": {"action": "stop_navigation"}},
                "x-is-dangerous": True,
                "x-resource": ["base"],
            },
            "configSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "default": DEFAULT_TOPIC,
                              "scope": "instance"},
                    "rate_hz": {"type": "number", "default": 10,
                                "scope": "instance"},
                    "priority": {"type": "integer", "default": 50,
                                 "scope": "instance"},
                    "stop_distance_m": {"type": "number", "default": 1.2,
                                        "scope": "instance"},
                    "slow_distance_m": {"type": "number", "default": 1.8,
                                        "scope": "instance"},
                    "obstacle_stop_m": {"type": "number", "default": 0.8,
                                        "scope": "instance"},
                    "vx_max": {"type": "number", "default": 0.4,
                               "scope": "instance"},
                    "wz_max": {"type": "number", "default": 0.8,
                               "scope": "instance"},
                    "align_tol": {"type": "number", "default": 0.08,
                                  "scope": "instance"},
                    "min_confidence": {"type": "number", "default": 0.35,
                                       "scope": "instance"},
                    "max_obs_age_ms": {"type": "number", "default": 500,
                                       "scope": "instance"},
                },
                "required": [],
            },
            # Declared rather than discovered after start — a card that names no
            # topic until it runs leaves a downstream card with "连线缺少 topic"
            # the moment this one fails to start, which reads like a wiring
            # problem on a canvas that is wired correctly.
            "topic_out": [{"topic": self._topic, "format": TOPIC_FORMAT,
                           "desc": "motus.control/1 twist 指令流"}],
            "topic_in": [
                {"format": "data/json",
                 "desc": "vop 的检测结果（必需）"},
                {"format": "image/depth-zlib",
                 "desc": "深度图 640x480 uint16 mm（与深度摘要二选一）"},
                {"format": "state/odom",
                 "desc": "motus.odom/1（可选，接上才有卡死检测）"},
            ],
        }]

    def dispatch(self, name: str, args: dict):
        action = args.get("action") or name
        if action == "start":
            return self._start(args)
        if action == "stop":
            return self._stop()
        if action == "list_visible_objects":
            return self._list_visible_objects()
        if action == "navigate_to":
            return self._navigate_to(args)
        if action == "stop_navigation":
            return self._stop_navigation()
        if action == "pause":
            return self._halt(True)
        if action == "resume":
            return self._halt(False)
        if action == "info":
            return self._info()
        if action == "config":
            return self._config_action(args)
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Bundle lifecycle. Deliberately inert — same rule as the driver cards.

        A container restart must not put a robot back on the move towards a
        target somebody asked for an hour ago.
        """

    def stop(self):
        self._stop()

    # ── actions ──────────────────────────────────────────────────────────────

    def _start(self, args: dict):
        descriptor = args.get("control_interface") or {}
        if not descriptor:
            return self._error(
                "没有拿到下游动作空间 —— 请把本卡片的输出连到一张驱动的底盘命令"
                "卡片（control/velocity 端口），agent-core 会在启动时把它的 "
                "descriptor 传过来")

        capabilities = self._capabilities()
        problems = negotiate.check(capabilities, descriptor, label="策略")
        if problems:
            return self._error("导航策略与下游动作空间不匹配："
                               + "；".join(problems))

        mode = descriptor.get("mode")
        if mode != CONTROL_MODE:
            return self._error(
                f"下游 mode 是 {mode!r}，这张卡片只会产生 {CONTROL_MODE!r} —— "
                "它输出的是底盘速度，不是关节角")

        rate = negotiate.effective_rate(capabilities, descriptor,
                                        self._cfg.get("rate_hz"))

        inputs = list(args.get("input_topics") or [])
        single = args.get("input_topic")
        if single and single not in inputs:
            inputs.insert(0, single)

        with self._lock:
            if self._running:
                return self._error("already running")
            self._descriptor = descriptor
            self._rate_hz = rate
            self._ttl_ms = negotiate.ttl_ms(rate, descriptor)
            self._seq = 0
            self._published = 0
            self._paused = False
            self._last_error = ""
            self._state = policy_mod.State()
            self._running = True

        try:
            self._open_publisher()
            binding, problem = self._bind_inputs(self._node, inputs)
        except Exception as error:                            # noqa: BLE001
            self._unwind()
            return self._error(f"启动失败：{error}")

        if problem:
            self._unwind()
            # Refused at start rather than running and publishing nothing: the
            # latter leaves the reason only in info().error while the canvas
            # shows a card that looks perfectly healthy.
            return self._error(problem)

        self._binding = binding
        log.info("navi started: topic=%s %.1f Hz ttl=%d ms inputs=%s",
                 self._topic, rate, self._ttl_ms, binding)
        return {"state": "running", "topic": self._topic, "rate_hz": rate,
                "ttl_ms": self._ttl_ms, "inputs": binding,
                "degraded": self._degradations()}

    def _capabilities(self) -> dict:
        return {"control_mode": CONTROL_MODE,
                "action_dim": ACTION_DIM,
                "control_hz": float(self._cfg.get("rate_hz") or 10.0),
                "chunk_size": 1,
                "model": "visual-servo/1"}

    def _list_visible_objects(self):
        """当前稳定看到的东西，按可靠程度排序。

        取最近 N 帧的**交集**而不是最新一帧：单帧检测会闪 —— 一把确实在那里的
        椅子可能十帧里只出现七帧，而某一帧会凭空多出一个 0.4 置信度的东西。把
        单帧结果交给 LLM 去挑目标，它迟早会挑到一个下一帧就不存在的，然后卡片
        立刻进入搜索模式，表现为机器人朝一个从来没有过的方向转圈。

        每一项都带 `key`，那是 navigate_to 应当收到的东西：名字区分不了两把不同
        颜色的椅子，而 key 可以。
        """
        with self._obs_lock:
            frames = list(self._recent)
            depth = ({"map": self._depth_map, "bands": dict(self._depth_bands)}
                     if self._depth_ms else None)
        if not frames:
            return {"objects": [], "frames": 0,
                    "message": "还没有收到任何检测结果 —— 确认 vop 卡片在运行且相机有画面"}

        need = max(1, int(round(len(frames) * 0.6)))
        items = policy_mod.stable_objects(frames, min_frames=need,
                                          config=self._config)
        out = []
        for item in items:
            distance = policy_mod.target_distance(item["object"], depth,
                                                  self._config)
            out.append({
                "key": item["key"],
                "name": item["name"],
                "color": item["color"],
                "bearing": round(item["bearing"], 3),
                "distance_m": distance,
                "confidence": item["confidence"],
                "stability": f"{item['seen_in_frames']}/{item['of_frames']} 帧",
                "description": policy_mod.describe(item["object"], distance),
            })
        return {"objects": out, "count": len(out),
                "frames": len(frames), "min_frames": need,
                # 降级说明一并带出来：距离是 None 还是个数，取决于接了哪种深度，
                # 而只看列表是看不出差别的。
                "degraded": self._degradations()}

    def _navigate_to(self, args: dict):
        target = (args.get("target") or "").strip()
        if not target:
            return self._error("navigate_to 需要 target —— 一个 vop 认得的名字")
        with self._lock:
            if not self._running:
                return {"state": "idle",
                        "message": "卡片未在运行，请先在画布上启动"}
            if "stop_distance_m" in args and args["stop_distance_m"] is not None:
                self._config.stop_distance_m = float(args["stop_distance_m"])
            # A new target discards everything remembered about the old one —
            # the search direction and the stuck timer both refer to a target
            # that is no longer being chased.
            self._state = policy_mod.State(target=target)
            self._paused = False
            # **The card mints this, agent-core does not supply it.** The ACP
            # contract runs the other way from what it looks like: an async tool
            # returns an `action_id` in its reply, agent-core parses it out and
            # registers the pending action (`mcp_client.py`, "ACP: 异步工具"),
            # and the completion later refers back to it. Reading it out of
            # `args` — which is what this did first — finds nothing, so no
            # pending is ever registered and no completion can be matched.
            self._acp_action_id = f"navi_{uuid.uuid4().hex[:12]}"
        log.info("navi navigate_to: target=%r", target)
        out = {"state": "running", "target": target, "topic": self._topic,
               # agent-core reads this back out to register the pending action.
               # Without it in the reply the action completes instantly as far
               # as the barrier is concerned, and the completion POST below
               # later refers to an id nobody is waiting on.
               "action_id": self._acp_action_id,
               "stop_distance_m": self._config.stop_distance_m,
               "degraded": self._degradations()}

        # 立刻回答「这个目标现在看得见吗」。
        #
        # 不看得见**不拒绝** —— 「转过去找那把椅子」是正当用法，目标本来就可能
        # 在视野外。但也不能只回一个 "running" 就完事：早先正是这样，一个拼错的
        # key 换来一句「已开始」，然后机器人转满一圈、16 秒后才在 info().last 里
        # 留下失败原因，而调用方那时早就不在看了。它在下指令的这一刻就能知道。
        visible = self._visible_keys()
        if visible and not self._matches_anything(target, visible):
            out["warning"] = (f"目标 {target!r} 不在当前可见列表里，将转身搜索；"
                              f"若它其实就在眼前，多半是 key 抄错了")
            out["visible_now"] = visible
        return out

    def _visible_keys(self) -> list:
        with self._obs_lock:
            frames = list(self._recent)
        if not frames:
            return []
        need = max(1, int(round(len(frames) * 0.6)))
        return [o["key"] for o in policy_mod.stable_objects(
            frames, min_frames=need, config=self._config)]

    @staticmethod
    def _matches_anything(target: str, keys) -> bool:
        """和 select_target 同一套宽松度：key 全等，或名字部分匹配。"""
        wanted = target.strip().lower()
        return any(wanted == k.lower() or wanted in k.split("#")[0].lower()
                   for k in keys)

    def _stop_navigation(self):
        with self._lock:
            if not self._running:
                return {"state": "idle", "message": "卡片未在运行"}
            self._state = policy_mod.State()
        return {"state": "running", "message": "已停止导航，卡片仍在运行"}

    def _halt(self, halted: bool):
        with self._lock:
            if not self._running:
                return {"state": "idle", "message": "卡片未在运行"}
            self._paused = bool(halted)
        return {"state": "paused" if halted else "running",
                "target": self._state.target}

    def _stop(self):
        with self._lock:
            if not self._running:
                return {"state": "idle"}
            self._running = False
            self._paused = False
            node, self._node = self._node, None
            timer, self._timer = self._timer, None
            self._publisher = None

        if timer is not None and node is not None:
            try:
                node.destroy_timer(timer)
            except Exception:                                 # noqa: BLE001
                pass
        if node is not None:
            try:
                self._executor.remove_node(node)
                node.destroy_node()
            except Exception:                                 # noqa: BLE001
                pass
        return {"state": "stopped"}

    def _unwind(self):
        with self._lock:
            self._running = False
        self._stop()

    def _config_action(self, args: dict):
        changed = {}
        for key, value in (args or {}).items():
            if key in policy_mod.Config.__dataclass_fields__ and value is not None:
                setattr(self._config, key, float(value))
                changed[key] = float(value)
        return {"status": "configured", "config": changed}

    def _info(self):
        with self._lock:
            decision = self._last_decision
            return {
                "state": ("running" if self._running and not self._paused else
                          "paused" if self._running else "idle"),
                "topic": self._topic,
                "target": self._state.target,
                "rate_hz": self._rate_hz,
                "published": self._published,
                "inputs": dict(self._binding),
                # Which precision the card is actually operating at. A card
                # falling back to the summary and a card working properly look
                # identical from downstream, and the difference is metres.
                "degraded": self._degradations(),
                "last": ({"status": decision.status, "reason": decision.reason,
                          "distance_m": decision.distance_m,
                          "bearing": decision.bearing}
                         if decision else None),
                "error": self._last_error,
            }

    def _degradations(self) -> list:
        out = []
        if not self._binding.get("depth_map"):
            out.append("只接了深度摘要，没有深度图 —— 距离按目标所在的三分之一"
                       "画面估计，精度明显变差")
        if not self._binding.get("odom"):
            out.append("没接 state/odom —— 无卡死保护，撞上东西不会自己停")
        for hint in (self._binding.get("unknown") or []):
            out.append(f"有一路输入没有被使用：{hint}")
        return out

    def _error(self, message: str):
        with self._lock:
            self._last_error = message
        log.warning("navi: %s", message)
        return {"state": "error", "message": message}

    # ── wiring ───────────────────────────────────────────────────────────────

    # perception 自己定义的输出话题后缀（plugins/vop.py::output_topic_for，
    # plugins/visual_depth.py::output_topics_for）。角色由它们决定，而不是由
    # 「这条话题上现在有没有发布者」决定 —— 见 _bind_inputs。
    #
    # 顺序有意义：visual_depth_summary 必须排在 visual_depth 前面，否则带
    # `/visual_depth` 前缀的摘要话题会先被当成深度图。
    _ROLE_SUFFIXES = (
        ("/visual_depth_summary", "depth_summary"),
        ("/visual_depth", "depth_map"),
        ("/objects", "objects"),
        ("/state/odom", "odom"),
    )

    def _bind_inputs(self, node, topics):
        """把连上来的话题分派到角色。

        **上游此刻有没有在发布，不参与这个判断。** 这一条是真机上换来的：
        perception 的卡片启动时会先回一个 `loading`（TensorRT engine 在后台加
        载），agent-core 明确支持这种返回并会轮询到 `settled`，但在那之前 ROS
        图上是空的。早先这里靠查图里的消息类型定角色，于是 visual_depth 还在
        加载时，它的话题被判成「无发布者、认不出」，navi 启动失败，整个项目按
        严格模式回滚 —— 而日志的下一行正是 `visual_depth settled: running`。
        连线完全正确，报错却说没连上。

        所以角色按**话题名**判定：后缀由 perception 自己的 `output_topic_for` /
        `output_topics_for` 决定，是个确定的契约，不依赖时序。消息类型只在名字
        认不出来时作为兜底 —— 那时图里有没有发布者才真的有参考价值。

        还没有数据不是错误：`observation()` 的过期检查已经覆盖它，而那条路径的
        结论是「什么都不发，让下游 watchdog 停住底盘」，正是此时该做的事。
        """
        bound = {"objects": "", "depth_map": "", "depth_summary": "", "odom": ""}
        unknown = []

        for topic in topics:
            role = self._role_of(topic)
            if not role:
                role, hint = self._role_from_graph(node, topic)
                if not role:
                    unknown.append(hint)
                    continue
            # 同一角色接了多路时保留第一条：多接一路深度图是画布上的手误，
            # 静默换成后接的那条只会让「为什么距离不对」更难查。
            if not bound[role]:
                bound[role] = topic

        missing = []
        if not bound["objects"]:
            missing.append("vop 的检测结果")
        if not bound["depth_map"] and not bound["depth_summary"]:
            missing.append("深度（深度图或深度摘要，至少一路）")
        if missing:
            return bound, ("导航需要的观测没有连上：" + "、".join(missing)
                           + "。请在画布上把 vop 卡片和 visual_depth 卡片的输出"
                             "连到本卡片的输入端口。"
                           + (f"（无法识别的输入：{'、'.join(unknown)}）"
                              if unknown else ""))

        # 必需的都在，但还有认不出的连线：不拦启动（该有的观测都有了），却也不
        # 能装作没看见 —— 操作员画那根线是有意图的，而被静默忽略的一路输入在画
        # 布上和正常工作的一路长得一模一样。记进降级说明里。
        bound["unknown"] = unknown

        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        qos = _sensor_qos()
        if bound["depth_map"]:
            node.create_subscription(CompressedImage, bound["depth_map"],
                                     self._on_depth_map, qos)
        for role in ("objects", "depth_summary", "odom"):
            if bound[role]:
                node.create_subscription(
                    String, bound[role],
                    lambda message, r=role: self._on_string(r, message), qos)
        return bound, ""

    def _role_of(self, topic: str) -> str:
        """角色，纯按话题名。认不出返回 ''。"""
        for suffix, role in self._ROLE_SUFFIXES:
            if topic.endswith(suffix):
                return role
        return ""

    def _role_from_graph(self, node, topic: str):
        """兜底：名字认不出时问 ROS 图。返回 (role, hint)。

        到这里才看发布者是合理的 —— 名字已经没能给出答案，图是仅剩的线索。
        """
        types = dict(node.get_topic_names_and_types()).get(topic) or []
        if any(n.endswith(("CompressedImage", "Image")) for n in types):
            return "depth_map", ""
        if any(n.endswith("String") for n in types):
            # String 上三种载荷都可能，名字又认不出，猜错的代价是拿摘要当里程
            # 计、或者反过来 —— 那会让机器人按完全错误的数字行动。不猜。
            return "", f"{topic}（String，但话题名不符合 perception 的命名约定，无法确定用途）"
        return "", f"{topic}（{'/'.join(types) or '暂无发布者，且话题名不符合命名约定'}）"

    def _on_string(self, role, message):
        try:
            payload = json.loads(message.data)
        except Exception:                                     # noqa: BLE001
            return          # one bad frame must not take the stream down
        now = int(time.time() * 1000)
        with self._obs_lock:
            if role == "objects":
                self._objects = payload
                self._objects_ms = now
                self._recent.append(payload)
            elif role == "depth_summary":
                self._depth_bands = depth_mod.bands_from_summary(payload)
                self._depth_ms = now
            elif role == "odom":
                # `axis_of` returns None for an unmeasured axis rather than
                # 0.0 — the stuck detector depends on the difference.
                self._odom = {"vx": odom_mod.axis_of(payload, "vx")}
                self._odom_ms = now

    def _on_depth_map(self, message):
        data = bytes(getattr(message, "data", b"") or b"")
        if not data:
            return
        try:
            depth = depth_mod.decode(data)
        except depth_mod.DepthError as error:
            with self._lock:
                self._last_error = f"深度图解码失败：{error}"
            return
        with self._obs_lock:
            self._depth_map = depth
            self._depth_bands = depth_mod.nearest_by_band(depth)
            self._depth_ms = int(time.time() * 1000)

    def _open_publisher(self):
        from rclpy.node import Node
        from std_msgs.msg import String

        node = Node(f"navi_{abs(hash(self._topic)) % 100000}")
        self._publisher = node.create_publisher(String, self._topic, 10)
        self._timer = node.create_timer(1.0 / max(self._rate_hz, 0.1), self._tick)
        self._executor.add_node(node)
        with self._lock:
            self._node = node

    # ── the loop ─────────────────────────────────────────────────────────────

    def observation(self):
        """The three inputs, with anything stale replaced by None.

        Staleness is checked here rather than in the policy so that "too old"
        and "never arrived" are the same thing to the decision — both mean the
        card cannot see, and both must stop the robot.
        """
        now = int(time.time() * 1000)
        max_age = self._config.max_obs_age_ms
        with self._obs_lock:
            objects = self._objects if now - self._objects_ms <= max_age else None
            fresh_depth = now - self._depth_ms <= max_age
            depth = ({"map": self._depth_map, "bands": dict(self._depth_bands)}
                     if fresh_depth else None)
            odom = self._odom if now - self._odom_ms <= max_age else None
        return objects, depth, odom

    def next_command(self):
        """One tick of the policy, as a message or None.

        Public and ROS-free so the whole decision path — including the sequence
        numbers and the two timestamps — is testable without a node.
        """
        now = time.time()
        dt = (now - self._last_tick) if self._last_tick else 1.0 / self._rate_hz
        self._last_tick = now

        objects, depth, odom = self.observation()
        decision = policy_mod.step(detections=objects, depth=depth, odom=odom,
                                   config=self._config, state=self._state,
                                   dt=dt)
        self._last_decision = decision
        if not decision.publishes:
            return None

        self._seq += 1
        obs_ms = self._objects_ms or int(now * 1000)
        return build_message(
            seq=self._seq,
            values=decision.values,
            mode=CONTROL_MODE,
            dof=ACTION_DIM,
            source=f"mcp__actucore__{self.PREFIX}",
            stamp_ms=int(now * 1000),
            # When the picture this was computed from was taken, not now. A
            # command generated this instant from a 900 ms old frame is the
            # characteristic failure here, and only this number catches it.
            obs_stamp_ms=obs_ms,
            ttl_ms=self._ttl_ms,
            priority=int(self._cfg.get("priority", 50)),
        )

    def _tick(self):
        publisher = self._publisher
        if publisher is None or not self._running or self._paused:
            # Publishing nothing is how a halt reaches the chassis: the
            # driver's watchdog stops it within watchdog_ms. Commanding a stop
            # from here would fight that.
            return
        try:
            message = self.next_command()
        except Exception as error:                            # noqa: BLE001
            with self._lock:
                self._last_error = f"{type(error).__name__}: {error}"
            log.warning("navi tick failed: %s", error)
            return
        if message is None:
            self._maybe_complete()
            return

        from std_msgs.msg import String

        payload = String()
        payload.data = json.dumps(message, ensure_ascii=False)
        publisher.publish(payload)
        with self._lock:
            self._published += 1
        self._maybe_complete()

    def _maybe_complete(self):
        """Report the outcome once, on arrival **or** failure.

        Both are answers the caller is owed. Reporting only success is how an
        action that failed sits open until the ACP timeout, holding the barrier
        and telling nobody why.
        """
        decision = self._last_decision
        if decision is None:
            return
        if decision.status == policy_mod.ARRIVED:
            status = "completed"
        elif decision.status == policy_mod.FAILED:
            status = "failed"
        else:
            return

        with self._lock:
            action_id, self._acp_action_id = self._acp_action_id, ""
        if not action_id:
            return

        result = {"status": decision.status, "reason": decision.reason,
                  "target": self._state.target,
                  "distance_m": decision.distance_m}
        threading.Thread(target=self._post_acp, args=(action_id, status, result),
                         daemon=True, name="navi_acp").start()

    def _post_acp(self, action_id: str, status: str, result: dict):
        """POST the completion to agent-core, off the timer thread.

        On a thread because this runs from the publish tick: a slow or hanging
        HTTP call here would stall the command stream, and a stalled stream is
        a robot whose watchdog stops it — a network hiccup must not become a
        motion fault.
        """
        import json as _json
        import os as _os
        import ssl as _ssl
        import urllib.request as _urllib

        url = _os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
        context = _ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = _ssl.CERT_NONE
        payload = _json.dumps({"action_id": action_id, "status": status,
                               "result": result, "tool": "navi",
                               "ts": time.time()}).encode()
        request = _urllib.Request(f"{url}/api/acp/complete", data=payload,
                                  headers={"Content-Type": "application/json"})
        try:
            _urllib.urlopen(request, timeout=10, context=context).read()
            log.info("navi ACP %s: %s", status, result.get("reason"))
        except Exception as error:                            # noqa: BLE001
            log.warning("navi ACP callback failed: %s", error)
