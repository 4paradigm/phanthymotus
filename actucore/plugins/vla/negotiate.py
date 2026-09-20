"""Reconcile a model's output against the arm that will execute it.

This runs once, at `start`, and it is the difference between a card that
refuses to come up and a card that discovers the mismatch one command at a time
at 30 Hz — by which point the arm has already moved, or the driver is rejecting
everything and the operator is looking at a stopped robot with no reason given.

Kept as plain functions over two dicts so it can be tested without a card, a
provider, a canvas or a robot. The card only reports what these return.

Error messages name the two numbers. "shape mismatch" sends somebody to read
code; "模型输出 32 维，下游只接受 14 维" sends them to the connection they drew.
"""

from __future__ import annotations

SCHEMA = "motus.control/1"


def check(capabilities: dict, descriptor: dict) -> list:
    """Every reason this model cannot drive this arm. Empty means it can.

    All reasons are collected rather than returned one at a time: an operator
    fixing a canvas should see the whole disagreement, not discover a second
    problem after restarting for the first.
    """
    problems = []
    capabilities = capabilities or {}
    descriptor = descriptor or {}

    if descriptor.get("control_interface") != SCHEMA:
        problems.append(
            f"下游卡片没有声明 {SCHEMA}（收到 "
            f"{descriptor.get('control_interface')!r}）—— 它可能不是一张控制卡片"
        )
        return problems              # nothing else is meaningful without this

    # ── 动作空间必须对上，而维度相同**不代表**空间相同 ──────────────────────
    #
    # 这是这个函数里最后补上的一条检查，补它的理由值得写下来：在它之前，比的只有
    # `dof` 和 `control_hz`。于是一个 23 维的**末端位姿**模型（UnifoLM-VLA 的
    # EE_R6_G1：2 × [xyz(3) + R6(6) + 夹爪(1)] + 腰 rpy(3)）连到一张 23 维的
    # `joint_position` 卡片上，两个数字完全吻合，协商通过，然后把位姿当关节角发下去。
    #
    # 它不是个别现象。已经在用的模型里至少三种空间：
    #
    #     SmolVLA / mock           joint_position
    #     π0.5 (DROID)             joint_velocity      ← 归一化关节速度
    #     OpenVLA (bridge)         末端 delta 位姿 + 夹爪
    #     UnifoLM-VLA (G1)         EE_R6_G1，motus.control/1 的 MODES 里**没有**这个
    #
    # **没声明也要拒。** 「不确定就拒绝，不要猜」在这里格外重要：猜错的代价不是报错，
    # 是机械臂走到错误的地方。所以缺字段时报错里直接写出该去哪儿设。
    declared = str(capabilities.get("control_mode") or "").strip()
    mode = str(descriptor.get("mode") or "").strip()
    if not declared:
        problems.append(
            "模型没有声明自己的动作空间（capabilities 里缺 control_mode）—— "
            f"下游是 {mode or '未知'}，而维度对得上并不代表空间对得上。"
            "远端模型在 phanthymotus-cloud 的模型声明里设 CONTROL_MODE，"
            "本机 provider 在 capabilities() 里返回 control_mode"
        )
    elif mode and declared != mode:
        problems.append(
            f"模型输出 {declared!r} 空间的动作，下游接受 {mode!r} —— "
            "两者维度可能相同，但含义不同，发下去就是让机械臂走到错误的地方"
        )

    action_dim = capabilities.get("action_dim")
    dof = descriptor.get("dof")
    if action_dim is not None and dof is not None and int(action_dim) != int(dof):
        problems.append(f"模型输出 {action_dim} 维动作，下游只接受 {dof} 维")

    hz = capabilities.get("control_hz")
    max_hz = ((descriptor.get("rate") or {}).get("max_hz"))
    if hz and max_hz and float(hz) > float(max_hz):
        problems.append(f"模型按 {hz:g} Hz 输出，下游上限 {float(max_hz):g} Hz")

    # A chunk longer than the watchdog window is not an error — the card paces
    # it out one step at a time — but a chunk shorter than one step is: the
    # stream would stall between inferences with nothing to send.
    chunk = capabilities.get("chunk_size")
    if chunk is not None and int(chunk) < 1:
        problems.append(f"chunk_size 为 {chunk}，至少要 1")

    # `groups` is read long after start — the card reports it as `x-resource` on
    # every schema fetch, which is a *different* code path from the one that
    # stored it. So a malformed value costs nothing here and everything there:
    # it is accepted at start, survives `stop` (the descriptor is not cleared),
    # and from then on every `tools/list` raises. agent-core then has no schema
    # for any card in the bundle, and the canvas shows cards with no ports —
    # a symptom that points at the canvas, with the traceback in actucore's log.
    #
    # Only the shape is checked. Whether the groups tile the vector is the
    # driver's own invariant and it already enforces it (`common/control/
    # descriptor.py::_parse_groups`); re-deriving it here would be a second
    # opinion that can disagree.
    groups = descriptor.get("groups")
    if groups is not None:
        if not isinstance(groups, (list, tuple)):
            problems.append(f"descriptor.groups 应当是列表，收到 {type(groups).__name__}")
        else:
            bad = [i for i, entry in enumerate(groups) if not isinstance(entry, dict)]
            if bad:
                problems.append(
                    f"descriptor.groups 的第 {', '.join(map(str, bad))} 项不是对象 —— "
                    f"每一项应形如 {{name, offset, count, resource}}"
                )

    return problems


def effective_rate(capabilities: dict, descriptor: dict, requested_hz=None) -> float:
    """The rate to actually publish at, clamped by everyone who has a say.

    The card's own preference is a preference; the driver's `max_hz` is a limit
    it declared about its own hardware, and the model's `control_hz` is what its
    actions were computed for. Publishing faster than the model's own rate does
    not make the arm smoother, it makes each action mean something slightly
    different from what it meant when it was generated.
    """
    candidates = []
    for value in (requested_hz,
                  (capabilities or {}).get("control_hz"),
                  ((descriptor or {}).get("rate") or {}).get("expected_hz")):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            candidates.append(value)
    rate = min(candidates) if candidates else 30.0

    max_hz = ((descriptor or {}).get("rate") or {}).get("max_hz")
    try:
        max_hz = float(max_hz)
    except (TypeError, ValueError):
        max_hz = 0.0
    if max_hz > 0:
        rate = min(rate, max_hz)
    return rate


def ttl_ms(rate_hz: float, descriptor: dict) -> int:
    """How long a command stays valid.

    Two periods, floored at 50 ms and capped by the driver's watchdog. The
    reasoning is in phanthymotus/docs/vla-integration.md §"频率与 ttl 的由来":
    a generous ttl
    removes the protection while still appearing to provide it, because a stale
    command then executes and the robot resumes from a pause on an old picture.
    Longer than the watchdog is meaningless — the driver has already given up.
    """
    period_ms = 1000.0 / rate_hz if rate_hz > 0 else 33.0
    ttl = max(50.0, period_ms * 2)
    watchdog = ((descriptor or {}).get("rate") or {}).get("watchdog_ms")
    try:
        watchdog = float(watchdog)
    except (TypeError, ValueError):
        watchdog = 0.0
    if watchdog > 0:
        ttl = min(ttl, watchdog)
    return int(ttl)
