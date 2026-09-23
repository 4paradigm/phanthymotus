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

# How often the descriptor is echoed back into the stream. See next_command.
_DESCRIPTOR_EVERY = 50

# The six twist axes. **The order is fixed by the protocol** — motus.control/1's
# MODES comment, loco_servo's AXIS_NAMES on the driver side and motus.odom/1's
# AXES are all this same list — so a twist producer knows what its six numbers
# are called without asking anything downstream. That is what lets this card
# name them even with no chassis wired, instead of leaving the renderer to call
# them joint1..joint6.
TWIST_AXES = ["vx", "vy", "vz", "wx", "wy", "wz"]


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
        # A second, human-only port. Derived from the command topic so a card
        # given a custom one keeps the pair together.
        self._view_topic = str(self._cfg.get("view_topic") or
                               (self._topic.rsplit("/", 1)[0] + "/view"))
        self._view_hz = float(self._cfg.get("view_hz", 5.0))

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
        self._view_publisher = None
        self._view_next_at = 0.0
        self._view_failed = False
        # (width, height) of the frame vop's boxes are in, from its camera
        # declaration. (0, 0) until one arrives — and then no box is converted,
        # which is the honest answer rather than a guessed resolution.
        self._objects_frame = (0, 0)
        self._objects_ms = 0
        # The last N detection payloads, for `list_visible_objects` to intersect.
        # A deque rather than a list: this is written every frame on the hot
        # path and has to be bounded — a card left running for an hour should
        # not be holding thirty thousand frames of detections.
        from collections import deque
        self._recent = deque(maxlen=int(self._cfg.get("stable_frames", 10)))
        self._depth_map = None
        self._depth_bands = {}
        self._depth_ms = 0
        self._odom = None
        self._odom_ms = 0
        self._binding = {}
        self._acp_action_id = ""
        self._limit_notes: list = []
        # What `_adopt_camera` had to change or assume. Separate from
        # `_limit_notes` only because they are populated at different points in
        # `_start`; both end up in `degraded`.
        self._camera_notes: list = []

    # ── tool ─────────────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        return [{
            "name": "navi",
            "type": "processor",
            "multiInstance": False,
            "description": ("视觉导航：朝一个目标走过去。目标不在视野里时会**自行"
                            "原地转身搜索**，所以不需要先手动转动机器人。"
                            "路上按深度减速避障，到设定距离停下。"),
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
                        "description": "列出**当前视野里**稳定看到的物体（最近若干帧"
                                       "的交集，单帧检测会闪）。"
                                       "用在**需要甄别或让人挑选**的时候：眼前有好几"
                                       "个同类、要报给用户看、或者指令本身含糊（「过去"
                                       "那边」）。目的明确的寻找和跟随不用先调它，"
                                       "直接 navigate_to 即可。"
                                       "另外它只看得到眼前 —— 返回为空**不代表目标不"
                                       "存在**，navigate_to 会自己转身去找，别拿它当"
                                       "存在性检查。",
                    },
                    "navigate_to": {
                        "params": ["target", "stop_distance_m"],
                        "description":
                            "朝目标走过去。**目标不必现在就看得见** —— 看不见时会"
                            "自行原地转身搜索，转满一圈仍未找到才算失败，所以不要"
                            "为了让它「先看见」而自己去发转向指令。已在导航时再调"
                            "即为换目标（旧目标立即作废）。异步：立即返回 "
                            "action_id，到达或失败时回调通知，失败的回调里带着原因"
                            "（没找到 / 被挡住 / 长时间没有进展）。"
                            "**如果失败原因是「没找到」，先调 list_visible_objects "
                            "看一眼机器人现在到底认出了什么**，再把看到的东西报给"
                            "用户让他确认 —— 检测模型给同一个物体的类别并不稳定"
                            "（同一个灭火器可能这一分钟叫 fire extinguisher、"
                            "下一分钟叫 bottle），所以「没找到」往往是名字对不上，"
                            "而不是东西不在。"
                            "**任务目的明确时直接调这个就行**（「去沙发那边」"
                            "「跟着那个人」），用普通名字即可，不必先 "
                            "list_visible_objects —— 那一步在这里只是多一次往返。",
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
            # **Three knobs, not twenty-two.**
            #
            # Everything else lives in `actucore/config.yaml`, where it has the
            # paragraph of explanation it needs, and stays reachable through the
            # `config` action for anyone tuning. What it must not be is a form
            # field: an operator asked to pick `release_frac` or
            # `bearingless_std_factor` has no way to judge the answer, and the
            # dialog's length hides the three that actually matter.
            #
            # There is a second, sharper reason. **agent-core sends every field
            # in this schema on every config call, defaults included** — so a
            # key here silently overrides the same key in config.yaml. On r1_sz
            # the file said `vx_max: 0.6` and the card reported 0.4, because the
            # schema default won. Every field removed from here is one fewer
            # place for the file to be quietly ignored; every field kept has to
            # carry the same default the file does, which `test_navi_card.py`
            # now checks.
            "configSchema": {
                "type": "object",
                "properties": {
                    "rate_hz": {"type": "number", "default": 10,
                                "description": "指令频率，Hz。会被下游的 max_hz 夹住",
                                "scope": "instance"},
                    "stop_distance_m": {"type": "number", "default": 0.8,
                                        "description": "走到目标前多远算到达（米）",
                                        "scope": "instance"},
                    "obstacle_stop_m": {"type": "number", "default": 0.6,
                                        "description": "正前方障碍近于这个距离就完全"
                                                       "不前进（米）。必须小于 "
                                                       "stop_distance_m，否则机器人会"
                                                       "被它正要走向的目标挡停",
                                        "scope": "instance"},
                },
                "required": [],
            },
            # Declared rather than discovered after start — a card that names no
            # topic until it runs leaves a downstream card with "missing topic"
            # the moment this one fails to start, which reads like a wiring
            # problem on a canvas that is wired correctly.
            "topic_out": [{"topic": self._topic, "format": TOPIC_FORMAT,
                           "desc": "motus.control/1 twist 指令流"},
                          {"topic": self._view_topic, "format": "image/jpeg",
                           "desc": "这张卡片看到的与决定的：深度图、走廊、目标框与"
                                   "距离、三个轴的指令。只给人看"}],
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
        # No downstream is **not** a reason to refuse starting.
        #
        # "Don't wire the chassis yet, I just want to see what it computes" is a
        # legitimate way to bring this card up, and the most common one while
        # tuning it: the commands still go out on the topic, you watch them, and
        # the robot does not move. Refusing here demanded a robot that can move
        # before its decisions could be observed at all.
        #
        # Nothing is relaxed when a downstream *is* present — that is the case
        # where hardware moves, and a mismatched action space has to be refused
        # at start rather than failing one command at a time at 10 Hz.
        descriptor = args.get("control_interface") or {}
        capabilities = self._capabilities()

        if descriptor:
            problems = negotiate.check(capabilities, descriptor, label="策略")
            if problems:
                return self._error("导航策略与下游动作空间不匹配："
                                   + "；".join(problems))
            mode = descriptor.get("mode")
            if mode != CONTROL_MODE:
                return self._error(
                    f"下游 mode 是 {mode!r}，这张卡片只会产生 {CONTROL_MODE!r} —— "
                    "它输出的是底盘速度，不是关节角")

        # Fit the policy's ceilings to the robot before the first tick.
        #
        # Without this the card happily runs a configuration the chassis cannot
        # execute — a `wz_max` under the robot's own deadband means every turn
        # command snaps to all-or-nothing, and nothing anywhere says so. The
        # notes go into `degraded` so the adjustment is visible rather than
        # merely correct.
        self._limit_notes = policy_mod.adopt_limits(self._config, descriptor)

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

        # Camera geometry, now that the inputs are bound and we know which
        # topic carries the depth. This has to come after `_bind_inputs` and
        # after `adopt_limits` — the corridor's half-width is read from the
        # chassis footprint and its *angular* extent from the camera, and both
        # have to be settled before the first tick converts one into the other.
        camera_problem = self._adopt_camera(args.get("camera_info"), binding)
        if camera_problem:
            self._unwind()
            return self._error(camera_problem)

        self._binding = binding
        log.info("navi started: topic=%s %.1f Hz ttl=%d ms inputs=%s",
                 self._topic, rate, self._ttl_ms, binding)
        return {"state": "running", "topic": self._topic, "rate_hz": rate,
                "ttl_ms": self._ttl_ms, "inputs": binding,
                "degraded": self._degradations()}

    def _adopt_camera(self, declarations, binding: dict) -> str:
        """Adopt the depth camera's geometry; return a reason to refuse, or "".

        Two jobs, and only the second can refuse a start.

        **Adopt.** The half field of view belongs to the camera and used to be
        typed into this card's config by hand. On r1_sz it read 0.55 rad against
        a lens measuring 0.888, which made the metric corridor 1.86 m wide —
        wider than any door — so every doorframe counted as dead ahead and the
        robot turned away from openings it fitted through. Missing declarations
        are not fatal: the conservative fallback stands and `degraded` says so.

        **Check both eyes are the same eye.** vop reports a *normalised* lateral
        offset and the depth map is a grid of distances; this card turns both
        into metres with one field of view, which is only correct if both come
        from the same lens. Nothing has ever enforced that — inputs are bound by
        what they carry, deliberately, so wiring camera A's vop to camera B's
        depth has always been available and would produce confidently wrong
        distances. Now that both sides declare an `id`, it is a comparison.
        """
        from .camera import camera_id, for_topic

        depth_topic = binding.get("depth_map") or binding.get("depth_summary")
        depth_decl = for_topic(declarations, depth_topic) if depth_topic else {}
        self._camera_notes = policy_mod._adopt_camera(self._config, depth_decl)

        objects_decl = for_topic(declarations, binding.get("objects"))
        self._objects_frame = (objects_decl.get("width") or 0,
                               objects_decl.get("height") or 0)
        depth_id, objects_id = camera_id(depth_decl), camera_id(objects_decl)
        if depth_id and objects_id and depth_id != objects_id:
            return (f"两路输入来自不同的相机：检测结果来自 {objects_id}，深度图来自 "
                    f"{depth_id}。这张卡片用同一个视场角把两者都换算成米，"
                    f"两颗镜头就会算出看起来合理但错的距离 —— 请把 vop 和 "
                    f"visual_depth 接到同一颗相机上")
        return ""

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
            # Report your own state before pointing at anyone else's. A card
            # that is not running receives nothing, of course — and this used to
            # answer "check that vop is running and the camera has a picture"
            # regardless, sending people upstream to investigate something that
            # was working, when the answer was in this card's own state.
            if not self._running:
                return {"objects": [], "frames": 0,
                        "message": "本卡片未在运行（state=idle），因此没有订阅任何"
                                   "输入。请在画布上启动项目 —— 容器重启后卡片不会"
                                   "自行恢复订阅。"}
            if not self._binding.get("objects"):
                return {"objects": [], "frames": 0,
                        "message": "没有接 vop 的检测结果输入，无从列出物体"}
            return {"objects": [], "frames": 0,
                    "message": f"已订阅 {self._binding['objects']}，但还没收到任何"
                               f"检测结果 —— 确认 vop 卡片在运行且相机有画面"}

        # Same bar as the tracker's: something that would be *chased* after
        # `confirm_hits` of `confirm_window` frames should be *listed* on the
        # same evidence. The two used to disagree by an order of magnitude —
        # ten frames to appear in this list, one frame to start driving a
        # chassis — and the list was the strict one.
        need = self._stability_bar(len(frames))
        items = policy_mod.stable_objects(frames, min_frames=need,
                                          config=self._config)
        out = []
        for item in items:
            distance = policy_mod.target_distance(item["object"], depth,
                                                  self._config)
            out.append({
                "key": item["key"],
                "name": item["name"],
                # Colour as readable text, not raw fields. Under
                # `publish_color: full` vop carries twelve numbers per object;
                # on r1_sz this reply reached 7356 characters, and it goes into
                # LLM context whole on every call. None of those numbers helps
                # the caller, who is choosing a target and needs only the key
                # and a description.
                "color": policy_mod.colour_text(item["object"]),
                "bearing": round(item["bearing"], 3),
                "distance_m": distance,
                "confidence": item["confidence"],
                "stability": f"{item['seen_in_frames']}/{item['of_frames']} 帧",
                "description": policy_mod.describe(item["object"], distance),
            })
        return {"objects": out, "count": len(out),
                "frames": len(frames), "min_frames": need,
                # Carry the degradations here too: whether `distance_m` is a
                # number or None depends on which depth source is wired, and the
                # list alone does not show the difference.
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
            # registers the pending action (`mcp_client.py`, the ACP section),
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

        # Answer "is this target visible right now" immediately.
        #
        # Not visible is **not** a refusal — "turn around and find the chair" is
        # legitimate, and the target may well be out of frame. But replying only
        # "running" is not enough either: that is what this did, so a mistyped
        # key bought a cheerful "started", and the reason surfaced sixteen
        # seconds later in info().last, after a full turn, long after the caller
        # had stopped looking. It can be known at the moment of the call.
        visible = self._visible_keys()
        if visible and not self._matches_anything(target, visible):
            out["warning"] = (f"目标 {target!r} 不在当前可见列表里，将转身搜索；"
                              f"若它其实就在眼前，多半是 key 抄错了")
            out["visible_now"] = visible
        return out

    def _stability_bar(self, frames: int) -> int:
        """How many of `frames` an object must appear in to count as really there."""
        ratio = self._config.confirm_hits / max(1, self._config.confirm_window)
        return max(1, min(frames, int(round(frames * ratio))))

    def _visible_keys(self) -> list:
        with self._obs_lock:
            frames = list(self._recent)
        if not frames:
            return []
        return [o["key"] for o in policy_mod.stable_objects(
            frames, min_frames=self._stability_bar(len(frames)),
            config=self._config)]

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
            field = policy_mod.Config.__dataclass_fields__.get(key)
            if field is None or value is None:
                continue
            # Coerce to the field's own type. Blanket `float()` turned
            # `use_lateral` into 1.0 — truthy, so it worked, which is exactly
            # how a field ends up holding the wrong type for a year.
            cast = (bool if field.type in ("bool", bool)
                    else int if field.type in ("int", int) else float)
            setattr(self._config, key, cast(value))
            changed[key] = getattr(self._config, key)
        return {"status": "configured", "config": changed}

    def _info(self):
        with self._lock:
            decision = self._last_decision
            return {
                "state": ("running" if self._running and not self._paused else
                          "paused" if self._running else "idle"),
                "topic": self._topic,
                # **agent-core registers monitored topics from here, not from
                # the tool schema.** After starting each card, api/config.py
                # takes `topic_out` out of its `info()` reply and calls
                # `register_topic_internal`; inspection subscribes off the back
                # of that, and only then does the canvas data-flow panel have
                # anything to show. Without it the card is entirely healthy, the
                # messages really are on the bus, and the panel is empty for
                # ever — which is exactly how it presented on r1_sz: subscribing
                # to /actucore/navi/cmd with rclpy received commands at 10 Hz
                # while the canvas showed nothing.
                # Both ports. agent-core registers the bus topics off this
                # reply, so a port missing here is a card that publishes into a
                # topic the dashboard never subscribes to — healthy everywhere,
                # invisible in the panel.
                "topic_out": [{"topic": self._topic, "format": TOPIC_FORMAT},
                              {"topic": self._view_topic,
                               "format": "image/jpeg"}],
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
                # **Coasting has to be visible here.** A robot walking towards a
                # prediction and a robot walking towards something it can see
                # produce identical commands, and only this says which is which.
                "track": self._state.tracker.describe(),
                "error": self._last_error,
            }

    def _degradations(self) -> list:
        # No degradations while not running. A degradation means "running, but
        # missing something"; a card that has not started has nothing bound
        # because it has not started, and reporting "only the depth summary is
        # wired" there is inventing a fact — which it did on r1_sz, with
        # `inputs` empty.
        if not self._running:
            return []
        out = []
        if not self._binding.get("depth_map"):
            out.append("只接了深度摘要，没有深度图 —— 距离按目标所在的三分之一"
                       "画面估计，精度明显变差；避障也退回按画面三等分判断，"
                       "那是一个**角度**扇区，近处比机器人还窄（0.8 m 处只覆盖"
                       "±0.16 m），肩膀会擦到判据之外的东西。摘要还无法表达"
                       "「这一段有多少像素是有效的」，所以满是空洞的一段会被"
                       "当成空旷")
        if not self._binding.get("odom"):
            out.append("没接 state/odom —— 无卡死保护，撞上东西不会自己停；"
                       "且目标被遮挡时只能按**指令**（而非实测）推算它去了哪，"
                       "dry_run、姿态被拒、死区归零都会让两者对不上")
        # The chassis is wired but swallowing everything. Without this the card
        # reports a healthy stream of commands, the driver reports APPLIED, and
        # the robot stands still — which is exactly how it presented on r1_sz
        # right after a deploy reset the driver's config to its image defaults.
        if self._descriptor.get("dry_run"):
            out.append("下游底盘是 dry_run —— 指令会被完整接收、检查、计数，"
                       "然后**丢掉**，机器人不会动")
        if self._descriptor.get("rotate_only"):
            out.append("下游底盘是 rotate_only —— vx/vy 会被清零，只执行转向")
        if self._running and not self._descriptor:
            out.append("没有接驱动的底盘命令卡片 —— 指令只发到话题上，"
                       "不会驱动任何硬件（想看它算什么的话，这是对的）")
        out.extend(self._limit_notes)
        out.extend(self._camera_notes)
        for hint in (self._binding.get("unknown") or []):
            out.append(f"有一路输入没有被使用：{hint}")
        return out

    def _error(self, message: str):
        with self._lock:
            self._last_error = message
        log.warning("navi: %s", message)
        return {"state": "error", "message": message}

    # ── wiring ───────────────────────────────────────────────────────────────

    # The output-topic suffixes perception defines for itself
    # (plugins/vop.py::output_topic_for, plugins/visual_depth.py::
    # output_topics_for). Roles come from these, not from whether a topic
    # currently has a publisher — see _bind_inputs.
    #
    # The order matters: visual_depth_summary has to precede visual_depth, or
    # the summary topic is matched as a depth map by its shared prefix.
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
            # First topic of a role wins. Two depth maps wired to one card is
            # a slip on the canvas, and silently switching to whichever came
            # last only makes "why is the distance wrong" harder to answer.
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

        # Everything required is present, but something else is wired and not
        # recognised: do not block the start — the observations are all there —
        # and do not pretend not to have noticed either. The operator drew that
        # line on purpose, and an ignored input looks exactly like a working one
        # on the canvas. It goes into the degradations.
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
            # Three different payloads ride on String and the name says
            # nothing, so a wrong guess means reading a depth summary as
            # odometry or the reverse — and the robot then acts on entirely the
            # wrong numbers. Do not guess.
            return "", f"{topic}（String，但话题名不符合 perception 的命名约定，无法确定用途）"
        return "", f"{topic}（{'/'.join(types) or '暂无发布者，且话题名不符合命名约定'}）"

    def _normalise_boxes(self, payload: dict) -> dict:
        """Give every detection a `bbox_norm`, from the pixel box vop publishes.

        **These two sides never agreed on a key.** vop publishes `bbox` in *its
        camera's* pixels; `policy.target_distance` asks for `bbox_norm` in 0..1.
        Nobody publishes that name, so the box path has never once run: every
        target's distance has come from the centre-patch fallback, and
        `_degradations()` did not catch it because it checks the config switch
        rather than whether a box ever arrived.

        `sample_box`'s own docstring says why the normalised form is the one
        that crosses the boundary: vop's pixels are in its camera's resolution
        and the depth map has been resampled to 640x480, so passing pixels
        across "produces plausible numbers for the wrong part of the image".
        The contract was right; the key name was never wired.

        The frame size comes from the camera declaration on the objects topic
        (`motus.camera/1`). **No declaration, no conversion** — inferring the
        resolution from a box that happens to be large would be exactly the kind
        of plausible guess this card keeps being bitten by.
        """
        objects = (payload or {}).get("objects")
        if not isinstance(objects, list):
            return payload
        width, height = self._objects_frame
        for obj in objects:
            if not isinstance(obj, dict) or obj.get("bbox_norm"):
                continue
            box = obj.get("bbox")
            if not box or len(box) != 4 or not (width and height):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            obj["bbox_norm"] = [x1 / width, y1 / height, x2 / width, y2 / height]
        return payload

    def _on_string(self, role, message):
        try:
            payload = json.loads(message.data)
        except Exception:                                     # noqa: BLE001
            return          # one bad frame must not take the stream down
        now = int(time.time() * 1000)
        with self._obs_lock:
            if role == "objects":
                self._objects = self._normalise_boxes(payload)
                self._objects_ms = now
                self._recent.append(payload)
            elif role == "depth_summary":
                self._depth_bands = depth_mod.bands_from_summary(payload)
                self._depth_ms = now
            elif role == "odom":
                # All three axes the policy uses, not just `vx`. The stuck
                # detector only ever wanted forward speed, but the tracker
                # compensates its prediction for the robot's whole twist —
                # yaw most of all, since on a legged chassis turning is what
                # moves a target across the frame fastest.
                #
                # `axis_of` returns None for an unmeasured axis rather than
                # 0.0, and both consumers depend on the difference: a robot
                # that cannot answer "am I moving" must not look stopped, and
                # one that cannot answer "am I turning" must not have its own
                # rotation assumed to be zero.
                self._odom = {axis: odom_mod.axis_of(payload, axis)
                              for axis in ("vx", "vy", "wz")}
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
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        node = Node(f"navi_{abs(hash(self._topic)) % 100000}")
        self._publisher = node.create_publisher(String, self._topic, 10)
        if self._view_hz > 0:
            self._view_publisher = node.create_publisher(
                CompressedImage, self._view_topic, 2)
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

        # Lift the command out of the robot's deadband, or drop it to zero. The
        # threshold comes from the downstream descriptor — it is a property of
        # the robot, and this policy should not know any robot's numbers.
        #
        # **Which threshold depends on what the command itself asks for.** On a
        # legged chassis the yaw deadband is a property of the gait, not of the
        # axis: R1 needs 1.0 rad/s to start turning from a standstill and 0.05
        # once it is already walking. Applying the standing figure to a command
        # that also translates inflates a small correction twenty-fold — and
        # this line would have done exactly that to every command the policy
        # had just been careful not to quantise.
        values = policy_mod.apply_deadband(decision.values,
                                           self._deadband_for(decision.values))

        self._seq += 1
        obs_ms = self._objects_ms or int(now * 1000)
        message = build_message(
            seq=self._seq,
            values=values,
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

        # Echo the downstream descriptor periodically, so the stream describes
        # itself.
        #
        # The dashboard's control renderer takes `joint_names` and `limits` out
        # of `control_interface`; without them it can only label by index and
        # infer a range from the extremes it has seen. Under `twist` that
        # labelling actively misleads: the one non-zero entry is captioned
        # "joint6" when it is the yaw rate, and the reader concludes the robot
        # is driving six joints.
        #
        # Every N commands rather than every one: the descriptor is a hundred
        # numbers and a constant — it cannot change after negotiation — so
        # carrying it at 10 Hz would multiply the command stream for nothing.
        # The first command always carries it, so a panel opened at any time has
        # names straight away instead of waiting out a cycle.
        if self._seq % _DESCRIPTOR_EVERY == 1:
            # With a downstream, echo its descriptor — real limits are what
            # give the panel's range bars a scale. Without one, still carry the
            # axis names: the order is the protocol's, not anyone's to ask.
            message["control_interface"] = self._descriptor or {
                "control_interface": negotiate.SCHEMA,
                "mode": CONTROL_MODE,
                "dof": ACTION_DIM,
                "joint_names": list(TWIST_AXES),
            }
        return message

    def _deadband_for(self, values) -> list:
        """The floors that apply to *this* command.

        `min_magnitude_moving` is optional, so a chassis that does not declare
        one keeps the standing floors everywhere — the old behaviour, which errs
        towards commanding too much rather than too little.
        """
        limits = (self._descriptor.get("limits") or {}) if self._descriptor else {}
        standing = limits.get("min_magnitude")
        moving = limits.get("min_magnitude_moving")
        if not moving or not (values[0] or values[1]):
            return standing
        return list(moving)

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
        # **Before the early return, not after it.** The tick publishes nothing
        # when the decision is to publish nothing — blind, searching, arrived,
        # refused — and those are exactly the moments somebody wants the picture
        # for. Drawing only while commands flow would make the overlay go dark
        # precisely when the robot stops explaining itself.
        self._publish_view()

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

    def _publish_view(self):
        """Draw the decision, if it is time and if drawing works.

        **Wrapped whole, and rate-limited separately from the command stream.**
        A picture for a human must never be able to stop a robot: an exception
        in a colour map, a missing codec, an image this card has no business
        failing on — none of that may reach the tick that feeds the chassis. So
        a failure here is counted and logged once, and the commands carry on.
        """
        publisher = self._view_publisher
        if publisher is None or self._view_hz <= 0:
            return
        now = time.monotonic()
        if now < self._view_next_at:
            return
        self._view_next_at = now + 1.0 / self._view_hz
        try:
            from sensor_msgs.msg import CompressedImage

            from . import view as view_mod

            frame = view_mod.render(
                depth_m=self._depth_map,
                decision=self._last_decision,
                track=self._state.tracker.describe(),
                box=self._target_box(),
                clearance=self._state.last_clearance,
                coverage=self._state.last_coverage,
                config=self._config)
            message = CompressedImage()
            message.format = "jpeg"
            message.data = view_mod.encode(frame)
            publisher.publish(message)
        except Exception as error:                            # noqa: BLE001
            if not self._view_failed:
                self._view_failed = True
                log.warning("navi view render failed, disabling it: %s", error,
                            exc_info=True)
            self._view_publisher = None

    def _target_box(self):
        """The current detection's box for the thing being chased, or None.

        Associated here rather than carried through the tracker on purpose: the
        tracker works in metres and bearings and has no use for a rectangle, and
        threading one through it to draw a picture would put display concerns
        into the estimator. The cost is that this is a *display-time* guess —
        the nearest detection by bearing, of the right name — and when the
        detector is not seeing the target at all there is simply no box, which
        is itself the thing worth showing.
        """
        payload = self._objects or {}
        target = (self._state.target or "").strip().lower()
        track = self._state.tracker.track
        if not target or track is None:
            return None
        best, best_gap = None, 1e9
        for obj in payload.get("objects") or []:
            name = str(obj.get("name") or "").lower()
            if target not in name and name not in target:
                continue
            box = obj.get("bbox_norm")
            if not box:
                continue
            position = obj.get("position") or [0.0, 0.0]
            bearing = float(position[0]) * self._config.half_fov_rad
            gap = abs(bearing - track.bearing_rad)
            if gap < best_gap:
                best, best_gap = box, gap
        return best

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
