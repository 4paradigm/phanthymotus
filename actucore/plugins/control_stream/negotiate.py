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


def check(capabilities: dict, descriptor: dict, label: str = "模型") -> list:
    """Every reason this model cannot drive this arm. Empty means it can.

    All reasons are collected rather than returned one at a time: an operator
    fixing a canvas should see the whole disagreement, not discover a second
    problem after restarting for the first.

    `label` is what the upstream is called in the complaints. It defaults to
    「模型」 because that is what the first caller was; a visual servo is not a
    model, and telling its operator that "模型输出 6 维" sends them looking for a
    checkpoint that does not exist.
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
    # EE_R6_G1）连到一张 23 维的 `joint_position` 卡片上，两个数字完全吻合，协商
    # 通过，然后把位姿当关节角发下去。
    #
    # （这里原本抄着一句「2 × [xyz(3) + R6(6) + 夹爪(1)] + 腰 rpy(3)」。那句话来自
    # 上游的枚举注释，而上游自己的数据管线排的不是这个 —— 两个夹爪都在尾部，且右
    # 在前左在后。真实布局记在 phanthymotus-cloud 的 runtimes/common/normalize.py。
    # 这条检查不依赖那个布局，但抄错的注释会把下一个人送进两只手互换的坑。）
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
            f"{label}没有声明自己的动作空间（capabilities 里缺 control_mode）—— "
            f"下游是 {mode or '未知'}，而维度对得上并不代表空间对得上。"
            "远端模型在 phanthymotus-cloud 的模型声明里设 CONTROL_MODE，"
            "本机 provider 在 capabilities() 里返回 control_mode"
        )
    elif mode and declared != mode:
        problems.append(
            f"{label}输出 {declared!r} 空间的动作，下游接受 {mode!r} —— "
            "两者维度可能相同，但含义不同，发下去就是让机械臂走到错误的地方"
        )
    problems.extend(_group_problems(capabilities, descriptor, label))

    action_dim = capabilities.get("action_dim")
    dof = descriptor.get("dof")
    if action_dim is not None and dof is not None and int(action_dim) != int(dof):
        problems.append(f"{label}输出 {action_dim} 维动作，下游只接受 {dof} 维")

    hz = capabilities.get("control_hz")
    max_hz = ((descriptor.get("rate") or {}).get("max_hz"))
    if hz and max_hz and float(hz) > float(max_hz):
        problems.append(f"{label}按 {hz:g} Hz 输出，下游上限 {float(max_hz):g} Hz")

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


def _group_problems(capabilities: dict, descriptor: dict, label: str = "模型") -> list:
    """逐段比动作空间，当两边都按段声明的时候。

    顶层那一条 `control_mode` vs `descriptor.mode` 比的是整条向量，而规范化之后的
    向量是**混的**：G1 的 19 维是两个末端位姿、两个归一化夹爪、三个腰关节角。这种
    向量上「顶层 mode 相同」几乎什么都没保证 —— 两边都报 `eef_pose`，而一边的第
    8 维是夹爪、另一边是腰，维度和顶层 mode 全都吻合，指令照发。

    **只有两边都声明了段才逐段比。** 一边有一边没有不算错：今天每个模型都是单一
    空间，没有段是常态，那时候顶层那条检查已经是完整的。这里要抓的是**都声明了却
    对不上**，那是真的分歧。

    逐段比三件事：`mode`、`(offset, count)`，以及 **advisory vs optional** ——
    最后一条比的不是形状，而是「这一段会不会被丢掉，以及丢它有没有得到许可」。
    """
    model_groups = capabilities.get("control_groups")
    driver_groups = descriptor.get("groups")
    if not isinstance(model_groups, (list, tuple)) or not model_groups:
        return []
    if not isinstance(driver_groups, (list, tuple)) or not driver_groups:
        return []

    default_mode = str(descriptor.get("mode") or "").strip()
    problems = []
    if len(model_groups) != len(driver_groups):
        problems.append(
            f"{label}把动作分成 {len(model_groups)} 段，下游卡片分成 "
            f"{len(driver_groups)} 段 —— 分段不同就无从逐段核对，"
            f"{label}侧：{_shape(model_groups, '')}；"
            f"下游：{_shape(driver_groups, default_mode)}"
        )
        return problems

    for i, (mine, theirs) in enumerate(zip(model_groups, driver_groups)):
        if not isinstance(mine, dict) or not isinstance(theirs, dict):
            continue                    # 形状问题由下面那条 groups 检查报
        # 段的 mode 留空表示继承顶层，和 `motus.control/1` 的驱动侧同一条规矩。
        mine_mode = str(mine.get("mode") or "").strip()
        theirs_mode = str(theirs.get("mode") or "").strip() or default_mode
        if mine_mode and theirs_mode and mine_mode != theirs_mode:
            problems.append(
                f"第 {i} 段（{label}叫 {mine.get('name')!r}，下游叫 "
                f"{theirs.get('name')!r}）：{label}输出 {mine_mode!r}，"
                f"下游接受 {theirs_mode!r}"
            )
        # **丢维要双方都同意过。** 驱动侧的 `advisory` 是「我收下但不执行」，
        # 生产者侧的 `optional` 是「任务不要求执行」。只有后者存在，前者才被允许
        # —— 这一条就是让「静默丢掉几维」在这套协议里不可能发生的那道门。
        #
        # 两个名字故意不同：语义不对称，同名会让一次复制粘贴把「可以不执行」变成
        # 「已经没执行」。缺省都是 false，所以今天每一对声明都落在「必须执行 /
        # 会执行」上，这条检查对它们是透明的。
        if theirs.get("advisory") and not mine.get("optional"):
            problems.append(
                f"第 {i} 段：下游把 {theirs.get('name')!r} 声明成 advisory"
                f"（收下但不执行），而{label}没有把 {mine.get('name')!r} 声明成 "
                "optional —— 这一段会被丢掉，而"f"{label}""认为它必须被执行。"
                "要么换一台真能执行它的机器人，要么由"f"{label}""声明这一段可以不执行")
        if mine.get("offset") != theirs.get("offset") or \
                mine.get("count") != theirs.get("count"):
            problems.append(
                f"第 {i} 段的位置对不上：{label} "
                f"[{mine.get('offset')}, +{mine.get('count')})，下游 "
                f"[{theirs.get('offset')}, +{theirs.get('count')}) —— "
                "总维度相同而分段错位，是最难从症状看出来的一种：每一段都把邻段的"
                "数字当成自己的"
            )
    return problems


def _shape(groups, default_mode: str) -> str:
    parts = []
    for group in groups:
        if not isinstance(group, dict):
            parts.append("?")
            continue
        mode = str(group.get("mode") or "").strip() or default_mode or "?"
        parts.append(f"{group.get('name')}×{group.get('count')}({mode})")
    return " + ".join(parts)


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
