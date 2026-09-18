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
