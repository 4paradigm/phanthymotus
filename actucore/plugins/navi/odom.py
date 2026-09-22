"""Reading `motus.odom/1`, from the consumer's side.

**This deliberately re-implements the reader rather than importing one.** The
producing half lives in `phanthymotus-driver/common/odom.py`, in a different
repository and a different container image, so importing it is not available —
but even if it were, the project's rule for cross-repo protocols applies:

    the spec is a document, not a shared library; both sides implement against
    it and each run their own contract tests

(CLAUDE.md, on `motus.vla/1`, and the same reasoning that keeps `motus.control/1`
from having a shared schema package.) A shared module couples the two release
cadences to each other's implementation details; the `schema` field is what
decouples them.

So what is here is small on purpose — three functions and one rule.

── the rule ─────────────────────────────────────────────────────────────────

**`None` is not `0.0`.** An axis a robot does not measure arrives as JSON
`null`, and it must stay `None` all the way to the stuck detector. A robot with
no odometry reporting 0.0 m/s is indistinguishable from a robot that has
stopped, and the stuck detector would then fire on every robot without
odometry, on its first step. That is the failure the whole format is shaped
around, and this file is where a consumer would undo it — one `or 0.0` is all
it takes.

Spec: `phanthymotus-driver/README_dev.md` § "Robot Odometry (motus.odom/1)".
"""

from __future__ import annotations

SCHEMA = "motus.odom/1"

# Must match the producer's order, which is `motus.control/1` twist order. The
# contract test in tests/test_navi_odom.py pins it against the descriptor this
# card negotiates with, so a divergence is caught here rather than as a robot
# comparing its forward speed against its yaw rate.
AXES = ("vx", "vy", "vz", "wx", "wy", "wz")

BODY_FRAME = "body"


def axis_of(sample: dict, name: str):
    """One axis out of a sample, or `None` if unmeasured, absent or unusable.

    Returns `None` — never 0.0 — for every kind of absence, including a sample
    in the wrong frame. A world-frame reading is real odometry, but comparing
    it against a body-frame command needs the robot's heading and a rotation;
    treating it as body-frame silently gives a wrong answer whenever the robot
    is not facing along world x.
    """
    if not isinstance(sample, dict):
        return None
    if sample.get("schema") != SCHEMA:
        return None
    if sample.get("frame") != BODY_FRAME:
        return None
    try:
        index = AXES.index(name)
    except ValueError:
        return None
    values = sample.get("twist") or []
    if index >= len(values):
        return None
    value = values[index]
    if value is None or isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def interface_provides(info: dict, name: str) -> bool:
    """Whether a driver declared, at start, that it measures this axis.

    Read once from the state card's `info()`. The per-sample `None`s say "not
    in this frame"; this says "not ever", which is what lets the card tell an
    operator that it has no stuck protection instead of appearing to have one.
    """
    interface = (info or {}).get("odom_interface") or {}
    if interface.get("schema") != SCHEMA:
        return False
    return name in (interface.get("provides") or [])
