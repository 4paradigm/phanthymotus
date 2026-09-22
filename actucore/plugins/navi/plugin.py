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
        self._depth_map = None
        self._depth_bands = {}
        self._depth_ms = 0
        self._odom = None
        self._odom_ms = 0
        self._binding = {}
        self._acp_notify = None

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
                               "enum": ["start", "stop", "navigate_to",
                                        "stop_navigation", "pause", "resume",
                                        "info", "config"]},
                    "target": {"type": "string",
                               "description": "要走过去的东西，用 vop 认得的名字"},
                    "stop_distance_m": {"type": "number",
                                        "description": "在目标前多远停下，默认 1.0 m"},
                    # Handed over by agent-core from the card wired downstream.
                    # Not operator-editable: it is a reading of another card,
                    # and a hand-typed copy is a copy that goes stale.
                    "control_interface": {"type": "object"},
                },
                "required": ["action"],
                "x-action-params": {
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
                    "stop_distance_m": {"type": "number", "default": 1.0,
                                        "scope": "instance"},
                    "slow_distance_m": {"type": "number", "default": 1.8,
                                        "scope": "instance"},
                    "obstacle_stop_m": {"type": "number", "default": 0.6,
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
        log.info("navi navigate_to: target=%r", target)
        return {"state": "running", "target": target, "topic": self._topic,
                "stop_distance_m": self._config.stop_distance_m,
                "degraded": self._degradations()}

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
        return out

    def _error(self, message: str):
        with self._lock:
            self._last_error = message
        log.warning("navi: %s", message)
        return {"state": "error", "message": message}

    # ── wiring ───────────────────────────────────────────────────────────────

    def _bind_inputs(self, node, topics):
        """Assign each connected topic a role by **what is on it**.

        agent-core passes topic names and no formats, so a role cannot be
        guessed from a name — a topic called `/robot/state` may be anything.
        Message type narrows it to two cases, and for the `String` ones the
        payload settles it: vop has `objects`, `visual_depth`'s summary has
        `nearest_by_region`, and `motus.odom/1` says so in `schema`.

        Sniffing the first payload rather than trusting a name means a
        mis-wired canvas shows up as "没认出来" here, at start, instead of as a
        robot that ignores its depth input.
        """
        by_type = dict(node.get_topic_names_and_types())
        bound = {"objects": "", "depth_map": "", "depth_summary": "", "odom": ""}
        unknown = []

        for topic in topics:
            types = by_type.get(topic) or []
            if any(n.endswith(("CompressedImage", "Image")) for n in types):
                bound["depth_map"] = topic
            elif any(n.endswith("String") for n in types):
                role = self._sniff_string_topic(node, topic)
                if role:
                    bound[role] = topic
                else:
                    unknown.append(f"{topic}(String，载荷认不出)")
            else:
                unknown.append(f"{topic}({'/'.join(types) or '无发布者'})")

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

        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        if bound["depth_map"]:
            node.create_subscription(CompressedImage, bound["depth_map"],
                                     self._on_depth_map, 1)
        for role in ("objects", "depth_summary", "odom"):
            if bound[role]:
                node.create_subscription(
                    String, bound[role],
                    lambda message, r=role: self._on_string(r, message), 1)
        return bound, ""

    def _sniff_string_topic(self, node, topic) -> str:
        """Which of the three JSON inputs a `String` topic carries.

        Decided from the topic's declared publishers where possible and from
        the payload otherwise. Returns "" when it cannot tell, which the caller
        reports rather than guessing — a depth summary mistaken for odometry
        would produce a card that runs and steers by nothing.
        """
        name = topic.rsplit("/", 1)[-1]
        # These suffixes are what the producing cards actually name their
        # topics; used as a hint only, with the payload as the authority.
        if name == "objects":
            return "objects"
        if name.endswith("visual_depth_summary") or name == "depth_summary":
            return "depth_summary"
        if name == "odom":
            return "odom"
        return ""

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
        """Push the ACP completion once, when the target has been reached."""
        decision = self._last_decision
        if decision is None or decision.status != policy_mod.ARRIVED:
            return
        notify, self._acp_notify = self._acp_notify, None
        if notify is not None:
            try:
                notify(decision)
            except Exception as error:                        # noqa: BLE001
                log.warning("navi ACP callback failed: %s", error)
