#!/usr/bin/env python3
"""
plugins/hand_gesture.py — static hand shapes from 21 keypoints.

Pure numpy. No second model, and that is a decision rather than a shortcut:
a learned gesture classifier has no "this is not a gesture" class, so a hand
doing nothing in particular gets a confident wrong label. `pose_stgcn.py`
records the same lesson at length — fed a real person lying on pavement, its
NTU-60 model returned "play with phone/tablet" at 0.997. Geometry can abstain,
and every threshold here is one somebody can read, check and move.

**Every threshold below was measured, not assumed**, on real photographs read
through the shipped RTMPose engine. Two quantities carry almost all of it:

    reach   how far a fingertip is from the wrist compared with that finger's
            middle joint, in palm lengths — positive means the finger is
            extended, negative means it is folded away

    ti_gap  thumb tip to index tip, in palm lengths — the OK sign is defined
            by those two touching and almost nothing else closes them

Measured per finger (reach), by picture:

    picture        thumb  index middle  ring  pinky   ti_gap
    open palm      +0.66  +0.42  +0.51  +0.50  +0.36   1.10
    OK (a)         +0.31  -0.28  +0.54  +0.54  +0.39   0.09
    OK (b)         +0.65  +0.26  +0.53  +0.48  +0.41   0.28
    loose fist     +0.49  -0.34  -0.34  -0.36  -0.22   0.56
    closed fist    +0.11  -0.27  -0.17  -0.29  -0.18   0.20
    thumbs up      +0.77  -0.59  -0.62  -0.63  -0.52   1.07

The four fingers separate cleanly: extended readings start at +0.26 and folded
ones stop at -0.17. The thresholds here sit at +0.25 and -0.05 with the gap
between them deliberately left as **neither** — a finger in that band is
unknown, and a gesture needing it is not reported. Refusing is the same call
`pose_action.py` makes for an occluded hip.

**The thumb does not work this way and is treated separately.** It reads
positive in every picture above, including both fists: it extends along its
own axis whatever the hand is doing, so "reach" says nothing about it. What
does separate a fist from a thumbs-up is how far the thumb tip sits from the
palm, and that is measured on three pictures only — so the band between them
is wide and anything inside it is reported as nothing at all.

What is deliberately **not** here: `thumbs_down`, `pinch`, `gun`, `rock`,
finger counting, and every two-handed shape. They are listed in
docs/hand-gesture.md and they are not implemented because there is no
measurement behind them yet, and a rule nobody has calibrated is a coin flip
with a name on it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from plugins.hand_runtime import HAND_INDEX, N_HAND_KEYPOINTS

#: Above this a finger counts as extended. Measured extended readings start at
#: +0.26; this leaves almost no margin above, which is deliberate — the cost
#: of calling an extended finger "unknown" is one unreported gesture, and the
#: cost of calling a folded one extended is a wrong one.
FINGER_EXTENDED = 0.25

#: Below this a finger counts as folded. Measured folded readings stop at
#: -0.17, so this has 0.12 of margin.
FINGER_CURLED = -0.05

#: Thumb tip to index tip, in palm lengths, for the OK sign. Measured 0.09 and
#: 0.28 for OK against 0.20-1.10 for everything else — and note the closed
#: fist at 0.20 sits *inside* that range, which is why OK also requires the
#: other three fingers to be extended rather than keying on the gap alone.
OK_TOUCH_GAP = 0.35

#: Thumb tip to palm centre, in palm lengths, separating a fist from a
#: thumbs-up once all four fingers are folded. **Measured on three pictures**
#: — fists at 0.23 and 0.70, a thumbs-up at 1.29 — so the band between 0.75
#: and 1.00 is reported as nothing rather than guessed at. Three samples is
#: not a calibration; it is enough to place a wide refusal band and no more.
THUMB_AWAY = 1.00
THUMB_NEAR = 0.75

#: Joint visibility below which the shape is not judged at all. RTMPose
#: returns a coordinate for every joint whether or not it could see it, and a
#: hidden fingertip placed by the model's prior is exactly the input that
#: makes a rule confident and wrong.
MIN_JOINT_CONF = 0.30

#: Chinese names, for the agent-facing vocabulary.
GESTURE_LABELS_ZH = {
    "open_palm": "张开手掌",
    "fist": "握拳",
    "pointing": "食指指向",
    "victory": "V 字",
    "ok": "OK",
    "thumbs_up": "点赞",
}

#: What `list_actions` offers by default. `thumbs_up` is in the vocabulary but
#: not here: see THUMB_AWAY for why three samples is not enough to turn on.
DEFAULT_HAND_GESTURES = ("open_palm", "fist", "pointing", "victory", "ok")

_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
_FINGERS = ("index", "middle", "ring", "pinky")


class HandShape:
    """One hand's measured geometry, before any naming happens.

    Kept separate from the labelling so that `info` and `evidence` can show
    the numbers a verdict came from. A gesture that reads wrong on a robot is
    almost never fixed by staring at its name.
    """

    __slots__ = ("reach", "ti_gap", "thumb_palm", "scale", "visible")

    def __init__(self, reach: dict, ti_gap: float, thumb_palm: float,
                 scale: float, visible: bool):
        self.reach = reach
        self.ti_gap = ti_gap
        self.thumb_palm = thumb_palm
        #: Palm length in pixels — wrist to middle knuckle. Everything else is
        #: divided by it, which is what makes the thresholds independent of
        #: how far away the person is standing.
        self.scale = scale
        self.visible = visible

    def as_evidence(self) -> dict:
        return {
            "reach": {k: round(v, 2) for k, v in self.reach.items()},
            "ti_gap": round(self.ti_gap, 2),
            "thumb_palm": round(self.thumb_palm, 2),
            "palm_px": round(self.scale, 1),
        }


def measure(keypoints, min_conf: float = MIN_JOINT_CONF) -> Optional[HandShape]:
    """Reduce 21 keypoints to the handful of scale-free numbers rules use.

    Returns None when the hand is too poorly seen to judge — the palm itself
    unmeasurable, or the joints a shape depends on invisible. None means "do
    not ask me", not "no gesture".
    """
    k = np.asarray(keypoints, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] != N_HAND_KEYPOINTS or k.shape[1] < 3:
        return None

    wrist = k[0, :2]
    middle_mcp = k[HAND_INDEX["middle_mcp"], :2]
    scale = float(np.linalg.norm(middle_mcp - wrist))
    if scale < 1e-3:
        return None

    conf = k[:, 2]
    visible = bool(conf[0] >= min_conf and conf[HAND_INDEX["middle_mcp"]] >= min_conf)

    reach = {}
    for name, tip in _TIPS.items():
        pip = _PIPS[name]
        if conf[tip] < min_conf or conf[pip] < min_conf:
            continue
        reach[name] = float(
            np.linalg.norm(k[tip, :2] - wrist) - np.linalg.norm(k[pip, :2] - wrist)
        ) / scale

    ti_gap = float(np.linalg.norm(k[4, :2] - k[8, :2])) / scale
    palm = (k[HAND_INDEX["index_mcp"], :2] + k[HAND_INDEX["pinky_mcp"], :2]) / 2.0
    thumb_palm = float(np.linalg.norm(k[4, :2] - palm)) / scale
    return HandShape(reach, ti_gap, thumb_palm, scale, visible)


def _state(shape: HandShape, finger: str) -> Optional[bool]:
    """True extended, False folded, None unknown.

    The band between the two thresholds is genuinely unknown and says so. A
    rule that treated it as one or the other would be deciding a close call
    silently, every time, in the same direction.
    """
    value = shape.reach.get(finger)
    if value is None:
        return None
    if value >= FINGER_EXTENDED:
        return True
    if value <= FINGER_CURLED:
        return False
    return None


def classify(keypoints, *, allowed=DEFAULT_HAND_GESTURES,
             min_conf: float = MIN_JOINT_CONF) -> dict:
    """Name the shape of one hand, or decline to.

    Returns `{"gesture": name or None, "evidence": {...}, "reason": str}`.
    The evidence rides along either way: "it said open_palm" is not checkable
    on a robot, and neither is "it said nothing".
    """
    shape = measure(keypoints, min_conf=min_conf)
    if shape is None:
        return {"gesture": None, "evidence": {}, "reason": "unreadable"}
    if not shape.visible:
        return {"gesture": None, "evidence": shape.as_evidence(),
                "reason": "palm_occluded"}

    states = {finger: _state(shape, finger) for finger in _FINGERS}
    evidence = {**shape.as_evidence(),
                "fingers": {f: ("ext" if s else "fold") if s is not None else "?"
                            for f, s in states.items()}}
    allowed = set(allowed or ())

    def have(*names, value=True):
        return all(states.get(n) is value for n in names)

    # OK first: it is the only shape defined by two fingertips touching, and
    # the index in that pose reads anywhere from -0.28 to +0.26 across two
    # real pictures — so keying it on the index's state would miss half of
    # them. The other three fingers extended is what keeps a closed fist (gap
    # 0.20) out.
    if ("ok" in allowed and shape.ti_gap <= OK_TOUCH_GAP
            and have("middle", "ring", "pinky")):
        return {"gesture": "ok", "evidence": evidence, "reason": "thumb_index_touch"}

    if "open_palm" in allowed and have("index", "middle", "ring", "pinky"):
        return {"gesture": "open_palm", "evidence": evidence, "reason": "all_extended"}

    if ("victory" in allowed and have("index", "middle")
            and have("ring", "pinky", value=False)):
        return {"gesture": "victory", "evidence": evidence, "reason": "two_up"}

    if ("pointing" in allowed and have("index")
            and have("middle", "ring", "pinky", value=False)):
        return {"gesture": "pointing", "evidence": evidence,
                "reason": "index_only", **_pointing_direction(keypoints)}

    folded = [f for f in _FINGERS if states.get(f) is False]
    extended = [f for f in _FINGERS if states.get(f) is True]
    # **Three folded is enough for a fist, and four is required of nothing
    # else.** In a closed fist the fingers occlude each other by construction,
    # so one of them routinely falls below the visibility threshold — measured
    # on a real photograph, a fist read as three folded fingers and a ring
    # finger the model could not see. Demanding all four would make the one
    # gesture whose self-occlusion is intrinsic the one that is never named.
    # Nothing is extended, so this cannot quietly swallow another shape.
    if len(folded) >= 3 and not extended:
        # All four folded is a fist *or* a thumbs-up, and which one depends on
        # the thumb — measured on three pictures, so the band between them is
        # wide and anything inside it gets no name.
        if "thumbs_up" in allowed and shape.thumb_palm >= THUMB_AWAY:
            return {"gesture": "thumbs_up", "evidence": evidence,
                    "reason": "fingers_folded_thumb_out"}
        if "fist" in allowed and shape.thumb_palm <= THUMB_NEAR:
            return {"gesture": "fist", "evidence": evidence,
                    "reason": "fingers_folded_thumb_in"}
        return {"gesture": None, "evidence": evidence,
                "reason": "folded_thumb_ambiguous"}

    unknown = [f for f, s in states.items() if s is None]
    return {"gesture": None, "evidence": evidence,
            "reason": f"no_match (unclear: {','.join(unknown)})" if unknown
                      else "no_match"}


def _pointing_direction(keypoints) -> dict:
    """Which way the index finger points, as a unit vector and a word.

    Image coordinates, so +x is to the camera's right and +y is down. This is
    the whole content of a pointing gesture — "they pointed" without it is not
    something a robot can act on — and it is read from the finger rather than
    from the forearm, which is what the body channel has to settle for.
    """
    k = np.asarray(keypoints, dtype=np.float32)
    mcp = k[HAND_INDEX["index_mcp"], :2]
    tip = k[HAND_INDEX["index_tip"], :2]
    vector = tip - mcp
    norm = float(np.linalg.norm(vector))
    if norm < 1e-3:
        return {}
    unit = (vector / norm).astype(float)
    dx, dy = float(unit[0]), float(unit[1])
    if abs(dx) >= abs(dy):
        word = "right" if dx > 0 else "left"
    else:
        word = "down" if dy > 0 else "up"
    return {"point_vector": [round(dx, 3), round(dy, 3)], "point_direction": word}


def gesture_catalogue() -> list:
    """Every shape this module can name, for `list_actions`."""
    return [
        {"name": name, "name_zh": GESTURE_LABELS_ZH[name],
         "default": name in DEFAULT_HAND_GESTURES}
        for name in sorted(GESTURE_LABELS_ZH)
    ]
