"""
tests/test_hand_gesture.py — naming the shape of a hand from 21 keypoints.

Entirely numpy, no engine. Hands are built from the *measured* geometry rather
than from imagination: `_hand()` takes a reach per finger and lays the joints
out so that `measure()` reads back those numbers, and the gesture cases use
the values actually observed through the shipped RTMPose engine on real
photographs (the table in plugins/hand_gesture.py). A test built from invented
proportions would pass against invented thresholds and say nothing.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

import vision_stubs  # noqa: F401  (installs the cv2 / ROS stubs)

from plugins.hand_gesture import (  # noqa: E402
    DEFAULT_HAND_GESTURES,
    FINGER_CURLED,
    FINGER_EXTENDED,
    GESTURE_LABELS_ZH,
    OK_TOUCH_GAP,
    THUMB_AWAY,
    THUMB_NEAR,
    classify,
    gesture_catalogue,
    measure,
)
from plugins.hand_runtime import HAND_INDEX, N_HAND_KEYPOINTS  # noqa: E402

PALM = 100.0          # wrist -> middle knuckle, in pixels
WRIST = (300.0, 500.0)

_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_PIP = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def _two_circles(a, ra, b, rb, fallback):
    """A point at distance `ra` from `a` and `rb` from `b`, or `fallback`.

    The thumb has to satisfy both measured quantities at once — its gap to the
    index tip and its distance from the palm — and they are two distance
    constraints on one point. Solving them is what makes the builder able to
    reproduce a real photograph's numbers instead of only one of them.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = float(np.linalg.norm(b - a))
    if d < 1e-9 or d > ra + rb or d < abs(ra - rb):
        return fallback
    x = (ra * ra - rb * rb + d * d) / (2 * d)
    h2 = ra * ra - x * x
    if h2 < 0:
        return fallback
    h = float(np.sqrt(h2))
    base = a + (b - a) * (x / d)
    perp = np.array([-(b - a)[1], (b - a)[0]], dtype=np.float64) / d
    return tuple(base + perp * h)


def _hand(reach=None, ti_gap=1.0, thumb_palm=1.2, conf=0.9, palm=PALM):
    """A 21-joint hand whose *measured* geometry is what you asked for.

    Every finger's middle joint sits exactly one palm length from the wrist
    and its tip `reach` palm lengths beyond that, so
    `(|tip-wrist| - |pip-wrist|) / palm` is the number you passed. An earlier
    version spread the fingers sideways, which made those distances larger
    than the palm and quietly turned every threshold test into a test of the
    builder.
    """
    reach = {**{f: 0.5 for f in _TIP}, **(reach or {})}
    k = np.zeros((N_HAND_KEYPOINTS, 3), dtype=np.float32)
    k[:, 2] = conf
    wx, wy = WRIST
    k[0] = (wx, wy, conf)
    k[HAND_INDEX["middle_mcp"]] = (wx, wy - palm, conf)
    for finger, tip in _TIP.items():
        pip = _PIP[finger]
        k[pip] = (wx, wy - palm, conf)
        k[tip] = (wx, wy - palm * (1.0 + reach[finger]), conf)
    k[HAND_INDEX["index_mcp"]] = (wx - 0.3 * palm, wy - 0.9 * palm, conf)
    k[HAND_INDEX["pinky_mcp"]] = (wx + 0.3 * palm, wy - 0.9 * palm, conf)

    index_tip = tuple(float(v) for v in k[_TIP["index"], :2])
    centre = ((k[HAND_INDEX["index_mcp"], 0] + k[HAND_INDEX["pinky_mcp"], 0]) / 2.0,
              (k[HAND_INDEX["index_mcp"], 1] + k[HAND_INDEX["pinky_mcp"], 1]) / 2.0)
    thumb = _two_circles(index_tip, ti_gap * palm, centre, thumb_palm * palm,
                         fallback=(index_tip[0] + ti_gap * palm, index_tip[1]))
    k[4] = (thumb[0], thumb[1], conf)
    return k


def _reach_of(k, finger):
    return measure(k).reach[finger]


# ── the builder itself ───────────────────────────────────────────────────────

def test_the_builder_produces_the_reach_it_was_asked_for():
    """Otherwise every threshold test below is measuring the builder."""
    k = _hand(reach={"index": 0.42, "pinky": -0.33})
    assert _reach_of(k, "index") == pytest.approx(0.42, abs=0.01)
    assert _reach_of(k, "pinky") == pytest.approx(-0.33, abs=0.01)


def test_measurements_do_not_depend_on_how_far_away_the_hand_is():
    """Everything is divided by the palm, which is what makes a threshold
    usable at 1 m and at 3 m."""
    near = measure(_hand(reach={"index": 0.4}, palm=240.0))
    far = measure(_hand(reach={"index": 0.4}, palm=40.0))
    assert near.reach["index"] == pytest.approx(far.reach["index"], abs=0.01)
    assert near.scale != far.scale


# ── the measured gestures ────────────────────────────────────────────────────
#
# The numbers in each case are the ones read off a real photograph through the
# shipped engine, listed in the module docstring.

def test_an_open_palm_is_named():
    k = _hand(reach={"index": 0.42, "middle": 0.51, "ring": 0.50, "pinky": 0.36},
              ti_gap=1.10)
    assert classify(k)["gesture"] == "open_palm"


def test_a_loose_fist_is_named():
    k = _hand(reach={"index": -0.34, "middle": -0.34, "ring": -0.36, "pinky": -0.22},
              ti_gap=0.56, thumb_palm=0.70)
    result = classify(k, allowed=("fist",))
    assert result["gesture"] == "fist"


def test_a_closed_fist_is_named():
    k = _hand(reach={"index": -0.27, "middle": -0.17, "ring": -0.29, "pinky": -0.18},
              ti_gap=0.20, thumb_palm=0.23)
    assert classify(k, allowed=("fist",))["gesture"] == "fist"


def test_the_two_ok_photographs_both_read_as_ok():
    """Their index fingers measured -0.28 and +0.26 — opposite sides of every
    extended/folded threshold. Keying OK on the index would have named one of
    them and missed the other, which is why it keys on the touch instead."""
    a = _hand(reach={"index": -0.28, "middle": 0.54, "ring": 0.54, "pinky": 0.39},
              ti_gap=0.09)
    b = _hand(reach={"index": 0.26, "middle": 0.53, "ring": 0.48, "pinky": 0.41},
              ti_gap=0.28)
    assert classify(a)["gesture"] == "ok"
    assert classify(b)["gesture"] == "ok"


def test_a_closed_fist_is_not_called_ok_despite_the_close_thumb():
    """Its thumb-index gap measured 0.20, inside the OK band. The three
    extended fingers are what separate them."""
    k = _hand(reach={"index": -0.27, "middle": -0.17, "ring": -0.29, "pinky": -0.18},
              ti_gap=0.20, thumb_palm=0.23)
    assert classify(k)["gesture"] != "ok"


def test_a_v_sign_is_named():
    k = _hand(reach={"index": 0.45, "middle": 0.50, "ring": -0.30, "pinky": -0.25},
              ti_gap=0.8)
    assert classify(k)["gesture"] == "victory"


def test_pointing_is_named_and_carries_a_direction():
    """A pointing event without the direction is not actionable — the
    direction is the entire content of the gesture."""
    k = _hand(reach={"index": 0.45, "middle": -0.30, "ring": -0.30, "pinky": -0.25},
              ti_gap=0.9)
    result = classify(k)
    assert result["gesture"] == "pointing"
    assert result["point_direction"] == "up"       # the builder points fingers up
    assert result["point_vector"][1] < 0


# ── the thumb, and refusing to guess ─────────────────────────────────────────

def test_a_thumbs_up_is_not_reported_by_default():
    """It is in the vocabulary but out of the default set: the thumb was
    separated on three pictures, which places a refusal band and nothing
    more."""
    k = _hand(reach={"index": -0.59, "middle": -0.62, "ring": -0.63, "pinky": -0.52},
              ti_gap=1.07, thumb_palm=1.29)
    assert "thumbs_up" not in DEFAULT_HAND_GESTURES
    assert classify(k)["gesture"] != "thumbs_up"


def test_a_thumbs_up_is_named_when_asked_for():
    k = _hand(reach={"index": -0.59, "middle": -0.62, "ring": -0.63, "pinky": -0.52},
              ti_gap=1.07, thumb_palm=1.29)
    assert classify(k, allowed=("thumbs_up", "fist"))["gesture"] == "thumbs_up"


def test_a_folded_hand_with_an_ambiguous_thumb_gets_no_name():
    """Between the two measured populations. Naming it would be deciding a
    close call silently, every time, in the same direction."""
    k = _hand(reach={"index": -0.4, "middle": -0.4, "ring": -0.4, "pinky": -0.4},
              ti_gap=0.9, thumb_palm=(THUMB_NEAR + THUMB_AWAY) / 2)
    result = classify(k, allowed=("thumbs_up", "fist"))
    assert result["gesture"] is None
    assert result["reason"] == "folded_thumb_ambiguous"


# ── the dead band ────────────────────────────────────────────────────────────

def test_a_finger_between_the_thresholds_is_unknown_not_rounded():
    k = _hand(reach={"index": (FINGER_CURLED + FINGER_EXTENDED) / 2,
                     "middle": 0.5, "ring": 0.5, "pinky": 0.5}, ti_gap=1.0)
    result = classify(k)
    assert result["gesture"] is None
    assert result["evidence"]["fingers"]["index"] == "?"


def test_the_dead_band_sits_between_the_measured_populations():
    """Extended readings started at +0.26 and folded ones stopped at -0.17."""
    assert FINGER_CURLED < 0 < FINGER_EXTENDED
    assert FINGER_EXTENDED <= 0.26
    assert FINGER_CURLED >= -0.17


# ── abstaining ───────────────────────────────────────────────────────────────

def test_an_invisible_fingertip_takes_its_finger_out_of_the_judgement():
    """RTMPose returns a coordinate for every joint whether it saw it or not,
    and a hidden fingertip placed by the model's prior is exactly the input
    that makes a rule confident and wrong."""
    k = _hand(reach={"index": 0.42, "middle": 0.51, "ring": 0.50, "pinky": 0.36})
    k[_TIP["ring"], 2] = 0.05
    result = classify(k)
    assert result["gesture"] is None
    assert result["evidence"]["fingers"]["ring"] == "?"


def test_an_occluded_palm_declines_outright():
    k = _hand(reach={"index": 0.42, "middle": 0.51, "ring": 0.50, "pinky": 0.36})
    k[0, 2] = 0.05
    result = classify(k)
    assert result["gesture"] is None
    assert result["reason"] == "palm_occluded"


def test_a_degenerate_hand_is_unreadable_rather_than_a_division_by_zero():
    k = np.zeros((N_HAND_KEYPOINTS, 3), dtype=np.float32)
    k[:, 2] = 0.9
    assert classify(k)["reason"] == "unreadable"


def test_a_wrong_shaped_array_is_refused():
    assert measure(np.zeros((17, 3), dtype=np.float32)) is None


# ── the vocabulary ───────────────────────────────────────────────────────────

def test_the_whitelist_is_honoured():
    k = _hand(reach={"index": 0.42, "middle": 0.51, "ring": 0.50, "pinky": 0.36},
              ti_gap=1.10)
    assert classify(k, allowed=("fist",))["gesture"] is None


def test_every_default_gesture_has_a_chinese_name():
    for name in DEFAULT_HAND_GESTURES:
        assert GESTURE_LABELS_ZH.get(name)


def test_the_catalogue_marks_which_are_on_by_default():
    catalogue = {entry["name"]: entry["default"] for entry in gesture_catalogue()}
    assert catalogue["open_palm"] is True
    assert catalogue["thumbs_up"] is False


def test_evidence_is_returned_even_when_nothing_is_named():
    """"It said nothing" has to be checkable on a robot too."""
    k = _hand(reach={"index": 0.1, "middle": 0.1, "ring": 0.1, "pinky": 0.1})
    result = classify(k)
    assert result["gesture"] is None
    assert result["evidence"]["reach"]
    assert "palm_px" in result["evidence"]


def test_a_fist_survives_one_finger_the_model_could_not_see():
    """Measured on a real photograph: a closed fist read as three folded
    fingers and a ring finger below the visibility threshold. Fingers occlude
    each other in a fist by construction, so demanding all four would make the
    one gesture whose self-occlusion is intrinsic the one never named."""
    k = _hand(reach={"index": -0.24, "middle": -0.25, "ring": -0.30, "pinky": -0.19},
              ti_gap=0.27, thumb_palm=0.21)
    k[_TIP["ring"], 2] = 0.05
    assert classify(k, allowed=("fist",))["gesture"] == "fist"


def test_the_relaxation_cannot_swallow_another_shape():
    """Three folded is enough only when nothing is extended."""
    k = _hand(reach={"index": 0.45, "middle": -0.30, "ring": -0.30, "pinky": -0.25},
              ti_gap=0.9, thumb_palm=0.3)
    assert classify(k, allowed=("fist",))["gesture"] is None


def test_two_unreadable_fingers_are_still_too_many():
    k = _hand(reach={"index": -0.24, "middle": -0.25, "ring": -0.30, "pinky": -0.19},
              ti_gap=0.27, thumb_palm=0.21)
    k[_TIP["ring"], 2] = 0.05
    k[_TIP["pinky"], 2] = 0.05
    assert classify(k, allowed=("fist",))["gesture"] is None
