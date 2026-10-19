"""Reading `motus.camera/1` — the consumer half of the camera declaration.

The spec is `phanthymotus-driver/README_dev.md` § Camera Parameters. This is a
**separate implementation of it**, not a shared import: the same rule as
`motus.control/1`, where a shared schema package would couple the two
repositories' release cadence and the `schema` field exists precisely to keep
them independent. What is shared is the document.

So this module is deliberately small and forgiving where the driver's is strict.
A driver must not be able to *emit* a declaration that is out of spec — its
`build()` validates and raises, in its own unit test. A consumer facing a
declaration it does not like has a different job: carry on, conservatively, and
say what it could not read. Refusing to navigate because a camera card shipped a
malformed field would turn one card's bug into a robot that will not move.

Why any of this exists: navi's avoidance corridor is metric, so every tick it
converts its half-width back into a range of image columns using the camera's
horizontal half field of view. On r1_sz that number was typed into navi's own
config as 0.55 rad against a lens measuring 0.888, and the corridor came out
1.86 m wide — wider than any door. Every doorframe read as dead ahead and the
robot turned away from openings it fitted through, while the depth map reported
clear. The camera knew, and had no way to say so.
"""
from __future__ import annotations

import math

SCHEMA = "motus.camera/1"


def for_topic(declarations, topic: str) -> dict:
    """The declaration covering `topic`, or `{}`.

    How a consumer joins: it already knows which topic it bound as its depth
    input, so it looks that topic up. Not by position in the list — a card may
    publish several ports and a consumer may have several inputs — and not by
    the upstream card's name, which would undo the reason this card dispatches
    inputs by what they carry.

    `declarations` is what agent-core passes on `start`: a mapping from topic to
    declaration. A list is accepted too, so this also works when handed a
    card's raw `info()` output.
    """
    if isinstance(declarations, dict):
        entry = declarations.get(topic)
        return entry if isinstance(entry, dict) else {}
    for entry in declarations or ():
        if isinstance(entry, dict) and entry.get("topic") == topic:
            return entry
    return {}


def half_fov(declaration: dict) -> tuple:
    """`(half_fov_rad, source)` for the horizontal axis, or `(None, reason)`.

    **`K` wins over the declared angle when both are present.** A calibration
    matrix is solved from many observations; the angle beside it is usually a
    tape measure and some trigonometry. Preferring the coarser number because it
    sits in a handier field would be backwards.

    Anything unusable returns `None` with a reason fit to put in front of an
    operator, rather than a default — telling the two apart is the entire point
    of the format, and a consumer that substitutes a plausible number here
    recreates the bug it was written to prevent.
    """
    if not isinstance(declaration, dict) or not declaration:
        return None, "没有声明"

    schema = declaration.get("schema")
    if schema != SCHEMA:
        return None, f"schema 是 {schema!r}，不是 {SCHEMA!r}"

    K = declaration.get("K")
    width = declaration.get("width")
    if isinstance(K, (list, tuple)) and len(K) == 9 and width:
        try:
            fx, width = float(K[0]), float(width)
        except (TypeError, ValueError):
            fx = 0.0
        if fx > 0 and width > 0:
            return math.atan((width / 2.0) / fx), "derived-from-K"

    angle = declaration.get("half_fov_rad")
    if angle is None:
        return None, "声明里没有视场角"
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        return None, f"half_fov_rad 不是数字：{angle!r}"
    angle = float(angle)
    if not 0.0 < angle < math.pi / 2:
        # The driver's `build()` refuses these, so reaching here means a card
        # that did not go through it. Named rather than clamped: the likeliest
        # cause is the full angle in a half-angle field, and silently halving it
        # would be a guess about somebody else's bug.
        return None, (f"half_fov_rad={angle:.3f} 不在 (0, π/2) 内 —— 这个字段是"
                      f"**半**视场角，填进来的像是全视场角")

    source = declaration.get("source") or "unspecified"
    return angle, str(source)


def camera_id(declaration: dict) -> str:
    """The camera identity, which survives every stage of the chain.

    What makes "these two inputs are looking through the same lens" a checkable
    statement rather than an assumption.
    """
    if not isinstance(declaration, dict):
        return ""
    return str(declaration.get("id") or "")
