"""The control law: one observation in, one body twist out.

Every decision this card makes is here, as a pure function over plain data. No
ROS, no publisher, no clock of its own — `plugin.py` is this plus a timer and a
topic. That split is what makes "what does it do when the target disappears
behind a person while an obstacle is on the left" a test rather than an
afternoon on a robot.

── the shape of the behaviour ───────────────────────────────────────────────

Reactive visual servoing, and nothing more. There is no map, no global path,
and no memory of where the target was a minute ago. It will not get around a
U-shaped obstacle, and it is not supposed to: `path_planner.py` in the go2
bundle is the other approach, and conflating the two gives something that is
bad at both.

  1. pick the target out of what vop reports
  2. turn to face it        wz from its horizontal offset
  3. drive towards it       vx from its distance, zero until roughly aligned
  4. slow and steer for obstacles the depth bands report
  5. stop at `stop_distance_m`, and say so

── the safety property that matters more than any of the above ──────────────

**When in doubt, emit nothing.** Stale observation, unreadable depth, target
lost for too long, stuck, arrived — all of them return `values=None`, and the
card then publishes nothing at all. The driver's watchdog brings the chassis to
a stop within `watchdog_ms`.

That is a deliberate choice against the alternative, which is to emit an
explicit zero. Emitting zero would also stop the robot, but it would keep the
watchdog fed, so a policy that silently died would leave a robot that believes
it is being driven. Silence is the one signal a broken upstream cannot fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Status values. `arrived` and `stuck` are terminal for one `navigate_to`;
# the rest are things that happen on the way.
APPROACHING = "approaching"
ALIGNING = "aligning"
AVOIDING = "avoiding"
SEARCHING = "searching"
ARRIVED = "arrived"
STUCK = "stuck"
BLIND = "blind"          # no usable observation — distinct from "nothing seen"
IDLE = "idle"


@dataclass
class Config:
    """Everything an operator can turn. Defaults are for a walking humanoid."""

    stop_distance_m: float = 1.0
    slow_distance_m: float = 1.8
    obstacle_stop_m: float = 0.6
    align_tol: float = 0.08
    k_yaw: float = 1.2
    k_fwd: float = 0.6
    vx_max: float = 0.4
    wz_max: float = 0.8
    search_rate: float = 0.4
    search_timeout_s: float = 8.0
    search_sweep_rad: float = 6.4        # a bit over one full turn
    lost_frames: int = 10
    max_obs_age_ms: int = 500
    stuck_window_s: float = 3.0
    stuck_speed_ratio: float = 0.2
    min_confidence: float = 0.35


@dataclass
class State:
    """What the policy remembers between ticks. Owned by the card, passed in."""

    target: str = ""
    missing_frames: int = 0
    last_seen_side: float = 1.0          # +1 target was right, -1 it was left
    searching_for_s: float = 0.0
    searched_rad: float = 0.0
    commanded_vx: float = 0.0
    moving_for_s: float = 0.0
    arrived: bool = False
    notes: list = field(default_factory=list)


@dataclass
class Decision:
    """One tick's answer. `values` is None when nothing should be published."""

    values: list | None
    status: str
    reason: str = ""
    distance_m: float | None = None
    bearing: float | None = None

    @property
    def publishes(self) -> bool:
        return self.values is not None


def _twist(vx: float, wz: float) -> list:
    """A body twist in `motus.control/1` order.

    `vy` stays 0: the depth map says nothing about what is beside the robot,
    and sidestepping blind is worse than turning. The descriptor keeps the axis
    open so a future policy with a wider sensor can use it.
    """
    return [vx, 0.0, 0.0, 0.0, 0.0, wz]


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def select_target(objects, name: str, config: Config):
    """The detection to chase, or None.

    Exact name first, then substring, so `navigate_to("chair")` finds
    `"office chair"` without `navigate_to("air")` finding it too. Among equals,
    the most confident; among equally confident, the most central — which keeps
    the choice stable frame to frame when two of the same thing are in view.
    """
    wanted = (name or "").strip().lower()
    if not wanted:
        return None

    candidates = [o for o in (objects or [])
                  if float(o.get("confidence") or 0) >= config.min_confidence]
    # `list_visible_objects` 给出的 key（`chair#azure`）优先于名字。它是那份列表
    # 里唯一能区分「两把不同颜色的椅子」的东西，而让 LLM 从列表里挑一个、再把
    # 它退化成名字来匹配，等于把刚做出来的区分又丢掉。
    keyed = [o for o in candidates if object_key(o).lower() == wanted]
    if keyed:
        pool = keyed
    else:
        exact = [o for o in candidates if str(o.get("name", "")).lower() == wanted]
        pool = exact or [o for o in candidates
                         if wanted in str(o.get("name", "")).lower()]
    if not pool:
        return None
    return max(pool, key=lambda o: (round(float(o.get("confidence") or 0), 2),
                                    -abs(_bearing(o))))


def _bearing(obj) -> float:
    position = obj.get("position") or [0.0, 0.0]
    return float(position[0]) if position else 0.0


def target_distance(obj, depth, config: Config) -> float | None:
    """How far the chosen object is, by whatever the depth source allows.

    `bbox` is much better than a centre point — it samples the whole object
    rather than one patch that may land on a gap — so it is used when vop
    publishes it. Both paths return None rather than a guess.
    """
    if depth is None:
        return None
    from . import depth as depth_mod

    if depth.get("map") is not None:
        bbox = obj.get("bbox_norm")
        if bbox:
            return depth_mod.sample_box(depth["map"], bbox)
        return depth_mod.sample_point(depth["map"], obj.get("position") or [0, 0])
    # Summary-only: the best available is the band the target sits in, which is
    # the nearest obstacle there and not the target itself. Usable for "am I
    # about to arrive", misleading for anything finer.
    bands = depth.get("bands") or {}
    return bands.get(_band_of(_bearing(obj)))


def _band_of(bearing: float) -> str:
    if bearing < -1 / 3:
        return "left"
    if bearing > 1 / 3:
        return "right"
    return "center"


def step(*, detections, depth, odom, config: Config, state: State,
         dt: float) -> Decision:
    """One tick.

    `detections` — vop's latest payload, or None if stale/absent.
    `depth`      — `{"map": ndarray|None, "bands": dict}`, or None.
    `odom`       — `{"vx": float|None}` from motus.odom/1, or None if unwired.
                   **`None` and `0.0` are different** and the stuck detector
                   depends on it: a robot with no odometry must not look like a
                   robot that has stopped.
    `dt`         — seconds since the previous tick.
    """
    if state.arrived:
        return Decision(None, ARRIVED, "已到达，等待下一条指令")

    if not state.target:
        return Decision(None, IDLE, "没有导航目标")

    # No usable observation is not the same as "nothing is there". Emitting
    # nothing lets the watchdog stop the robot, which is the right resting
    # state for "this policy cannot see".
    if detections is None or depth is None:
        return Decision(None, BLIND,
                        "观测过期或缺失（vop / 深度），停止发布，交给下游 watchdog")

    bands = depth.get("bands") or {}
    target = select_target(detections.get("objects"), state.target, config)

    if target is None:
        return _search(config, state, dt)

    state.missing_frames = 0
    state.searching_for_s = 0.0
    state.searched_rad = 0.0
    bearing = _bearing(target)
    state.last_seen_side = 1.0 if bearing >= 0 else -1.0
    distance = target_distance(target, depth, config)

    # Turn towards it. Positive bearing means the target is right of centre, and
    # `wz` is positive counter-clockwise, so the sign is inverted here.
    wz = _clamp(-config.k_yaw * bearing, config.wz_max)

    if distance is not None and distance <= config.stop_distance_m \
            and abs(bearing) <= config.align_tol:
        state.arrived = True
        # One explicit zero before going quiet, so the chassis stops on a
        # command rather than on a watchdog timeout — arriving is a success and
        # should not look like a dropped link in the driver's log.
        return Decision(_twist(0.0, 0.0), ARRIVED,
                        f"目标在 {distance:.2f} m，已到达停止距离",
                        distance_m=distance, bearing=bearing)

    if abs(bearing) > config.align_tol:
        # Turn in place first. Driving while badly misaligned traces an arc
        # into whatever is beside the target.
        state.commanded_vx = 0.0
        return Decision(_twist(0.0, wz), ALIGNING,
                        f"目标偏 {bearing:+.2f}，先转正",
                        distance_m=distance, bearing=bearing)

    vx = config.vx_max
    status, reason = APPROACHING, "接近中"
    if distance is not None:
        vx = min(vx, max(0.0, config.k_fwd * (distance - config.stop_distance_m)))
        reason = f"目标 {distance:.2f} m"

    vx, wz, status, reason = _avoid(vx, wz, bands, config, status, reason)

    stuck = _stuck(odom, state, vx, dt, config)
    if stuck:
        return Decision(None, STUCK, stuck, distance_m=distance, bearing=bearing)

    state.commanded_vx = vx
    return Decision(_twist(vx, wz), status, reason,
                    distance_m=distance, bearing=bearing)


def _avoid(vx, wz, bands, config: Config, status, reason):
    """Slow for what is ahead, and steer towards the freer side if it is close.

    Only forward motion is affected. Backing away is not available — the depth
    map covers what the camera sees and nothing behind the robot, so reversing
    is moving blind.
    """
    ahead = bands.get("center")
    if ahead is None:
        return vx, wz, status, reason

    if ahead <= config.obstacle_stop_m:
        left, right = bands.get("left"), bands.get("right")
        # Turn towards whichever side reports more room. If neither reports
        # anything, keep the bearing-derived wz — it is at least aimed at the
        # target, and turning arbitrarily is not an improvement on that.
        if left is not None and right is not None:
            wz = config.wz_max * (1.0 if left > right else -1.0)
        return 0.0, wz, AVOIDING, f"正前方 {ahead:.2f} m，停止前进并转向"

    if ahead < config.slow_distance_m:
        span = max(1e-6, config.slow_distance_m - config.obstacle_stop_m)
        scale = max(0.0, (ahead - config.obstacle_stop_m) / span)
        return vx * scale, wz, AVOIDING, f"{reason}；正前方 {ahead:.2f} m，减速"

    return vx, wz, status, reason


def _search(config: Config, state: State, dt: float) -> Decision:
    """Target not in frame. Turn towards where it last was, then give up.

    Two ways to stop, and the first is the better one. With odometry the sweep
    is measured in **radians actually turned**, so a full circle means a full
    circle. Without it the only available measure is wall-clock, which gives up
    early on a robot that turns slowly and late on one that spins — a timeout
    is the fallback, not the design.
    """
    state.missing_frames += 1
    if state.missing_frames < config.lost_frames:
        # Briefly occluded — a person walking past. Emitting nothing for a few
        # frames lets the watchdog hold rather than starting a search for
        # something that has not actually gone.
        return Decision(None, SEARCHING,
                        f"目标暂时不可见（{state.missing_frames}/{config.lost_frames} 帧）")

    state.searching_for_s += dt
    # Negated, for the same reason the main loop negates its bearing:
    # `last_seen_side` is +1 when the target was to the *right*, and turning
    # right is *clockwise*, which is negative wz. Without the minus sign the
    # robot sweeps away from the last place it saw the thing — which still
    # finds it eventually, by going all the way round, and so reads as "search
    # is just slow" rather than as a wrong sign.
    wz = -config.search_rate * state.last_seen_side
    state.searched_rad += abs(wz) * dt

    if state.searched_rad >= config.search_sweep_rad:
        return Decision(None, SEARCHING,
                        f"转过 {state.searched_rad:.1f} rad 仍未看到目标，停止")
    if state.searching_for_s >= config.search_timeout_s:
        return Decision(None, SEARCHING,
                        f"搜索 {state.searching_for_s:.1f}s 未果，停止")

    return Decision(_twist(0.0, wz), SEARCHING,
                    f"目标丢失，朝最后出现的一侧转找")


def _stuck(odom, state: State, commanded_vx: float, dt: float,
           config: Config) -> str:
    """Commanded to move and measurably not moving — we have hit something.

    Returns a reason, or "" if not stuck (or if it cannot be judged).

    **Unwired odometry, and an axis the robot does not measure, both return ""
    rather than "stuck".** They arrive here as `None`, and the whole reason
    `motus.odom/1` forbids reporting `0.0` for an unmeasured axis is this
    function: a robot that cannot answer "am I moving" would otherwise be
    declared stuck on its first step, every time.

    This matters most exactly where the rest of the policy is weakest —
    monocular depth against a flat wall is unreliable, so "the wall is still
    1.5 m away" can persist while the robot leans into it.
    """
    if commanded_vx <= 0.05:
        state.moving_for_s = 0.0
        return ""
    if odom is None:
        return ""
    measured = odom.get("vx")
    if measured is None:
        return ""

    if abs(measured) >= config.stuck_speed_ratio * commanded_vx:
        state.moving_for_s = 0.0
        return ""

    state.moving_for_s += dt
    if state.moving_for_s >= config.stuck_window_s:
        return (f"指令 {commanded_vx:.2f} m/s 已 {state.moving_for_s:.1f}s，"
                f"实测 {measured:.2f} m/s —— 判定被挡住，停止")
    return ""

# ── 稳定物体列表 ──────────────────────────────────────────────────────────────
#
# 单帧检测是会闪的：一把椅子在连续十帧里可能只出现在七帧，而某一帧会凭空多出
# 一个 0.4 置信度的「backpack」。把单帧结果直接交给 LLM 去挑目标，它会挑到一个
# 下一帧就不存在的东西，然后导航卡片立刻进入搜索模式 —— 表现为机器人朝一个从
# 来没有过的方向转圈。
#
# 所以对外暴露的是**最近若干帧的交集**：只有反复出现的才算数。

# 目标身份的键。名字之外还要带颜色，否则「两把椅子」在列表里是一个条目，而
# navigate_to("chair") 永远指向其中随机的一把。颜色来自 vop 的三元组，它在
# `publish_color: name` 下就是一个可直接比较的字符串。
def object_key(obj) -> str:
    name = str(obj.get("name") or "?")
    colour = str(obj.get("color") or "").strip()
    # 只取色相那一段（三元组的最后一个词）：亮度会随光照在帧间跳动，把它算进
    # 身份里会让同一把椅子在明暗之间变成两个条目。
    hue = colour.split(" ")[-1] if colour else ""
    return f"{name}#{hue}" if hue and hue != "neutral" else name


def describe(obj, distance_m=None) -> str:
    """一句人（和 LLM）能读、也能原样回填给 navigate_to 的描述。"""
    parts = [str(obj.get("name") or "?")]
    colour = str(obj.get("color") or "").strip()
    if colour:
        parts.append(colour)
    bearing = _bearing(obj)
    parts.append("正前方" if abs(bearing) <= 0.15
                 else ("偏左" if bearing < 0 else "偏右")
                 + ("很多" if abs(bearing) > 0.6 else ""))
    if distance_m is not None:
        parts.append(f"{distance_m:.1f}m")
    return " · ".join(parts)


def stable_objects(frames, *, min_frames: int = 6, config: Config = None) -> list:
    """在最近这些帧里反复出现的物体。

    `frames` 是最近 N 帧的 vop 载荷，最新的在最后。`min_frames` 是至少要出现在
    几帧里 —— 不要求全部出现过，因为遮挡和边缘抖动会让一个确实在那里的东西漏掉
    一两帧，而把阈值定成「全部」等于把列表清空。

    返回按「出现帧数、置信度」排序，所以最可靠的排在最前 —— LLM 通常取第一个。
    """
    config = config or Config()
    seen: dict = {}
    for index, frame in enumerate(frames):
        for obj in (frame or {}).get("objects") or []:
            if float(obj.get("confidence") or 0) < config.min_confidence:
                continue
            key = object_key(obj)
            entry = seen.setdefault(key, {"frames": set(), "last": obj,
                                          "confidence": 0.0})
            entry["frames"].add(index)
            # 留最新的一帧作为方位来源：物体和机器人都可能在动，旧帧的方位会把
            # 目标指到它已经不在的地方。
            entry["last"] = obj
            entry["confidence"] = max(entry["confidence"],
                                      float(obj.get("confidence") or 0))

    out = []
    for key, entry in seen.items():
        count = len(entry["frames"])
        if count < min_frames:
            continue
        out.append({
            "key": key,
            "name": entry["last"].get("name"),
            "color": entry["last"].get("color"),
            "position": entry["last"].get("position"),
            "bearing": _bearing(entry["last"]),
            "confidence": round(entry["confidence"], 2),
            "seen_in_frames": count,
            "of_frames": len(frames),
            "object": entry["last"],
        })
    out.sort(key=lambda o: (-o["seen_in_frames"], -o["confidence"]))
    return out
