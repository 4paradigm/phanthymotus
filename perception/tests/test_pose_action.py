"""
tests/test_pose_action.py — the rules that turn keypoints into an action label.

Entirely synthetic bodies and entirely numpy, which is the design point: the
pose plugin's only hardware-bound call is the engine's `infer()`, so every
decision about what counts as sitting, waving or falling is testable here.

The bodies are built from fractions of a person's height (see `_body`), because
that is what the rules themselves are written in — a test in raw pixels would
pass or fail depending on how far away the imaginary person stood.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import vision_stubs  # noqa: F401  (installs the cv2 / ROS stubs)

from plugins.pose_action import (  # noqa: E402
    RULE_TO_ACTIVITY,
    LabelStabiliser,
    ACTION_LABELS_ZH,
    ACTION_PRIORITY,
    DEFAULT_THRESHOLDS,
    EVENT_ACTIONS,
    PoseActionClassifier,
    PoseFrame,
    PoseTracker,
    action_catalogue,
)
from plugins.vision_runtime import COCO_INDEX, N_KEYPOINTS  # noqa: E402

H = 400.0          # the imaginary person's height in pixels
CX = 320.0         # their horizontal centre
TOP = 100.0        # top of their bounding box


def _kp(**named) -> np.ndarray:
    """COCO-17 array from {joint_name: (x, y)}; everything else invisible."""
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    for name, xy in named.items():
        index = COCO_INDEX[name]
        keypoints[index, 0] = xy[0]
        keypoints[index, 1] = xy[1]
        keypoints[index, 2] = 0.9
    return keypoints


def _head(cx=CX, top=TOP, h=H) -> dict:
    return {
        "nose":      (cx, top + 0.06 * h),
        "left_eye":  (cx - 0.02 * h, top + 0.05 * h),
        "right_eye": (cx + 0.02 * h, top + 0.05 * h),
        "left_ear":  (cx - 0.04 * h, top + 0.06 * h),
        "right_ear": (cx + 0.04 * h, top + 0.06 * h),
    }


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _body(*, cx=CX, top=TOP, h=H, left_ankle_dx=0.0, right_ankle_dx=0.0,
          shoulder_half=0.10) -> dict:
    """A plain standing body. Knees sit midway between hip and ankle, so the
    legs stay straight however the ankles are displaced."""
    hips = {"left_hip": (cx - 0.07 * h, top + 0.52 * h),
            "right_hip": (cx + 0.07 * h, top + 0.52 * h)}
    ankles = {"left_ankle": (cx - 0.07 * h + left_ankle_dx, top + 0.97 * h),
              "right_ankle": (cx + 0.07 * h + right_ankle_dx, top + 0.97 * h)}
    shoulders = {"left_shoulder": (cx - shoulder_half * h, top + 0.18 * h),
                 "right_shoulder": (cx + shoulder_half * h, top + 0.18 * h)}
    arms = {}
    for side in ("left", "right"):
        sign = -1.0 if side == "left" else 1.0
        arms[f"{side}_wrist"] = (cx + sign * 0.12 * h, top + 0.50 * h)
        arms[f"{side}_elbow"] = _mid(shoulders[f"{side}_shoulder"],
                                     arms[f"{side}_wrist"])
    return {
        **_head(cx, top, h), **shoulders, **arms, **hips,
        "left_knee": _mid(hips["left_hip"], ankles["left_ankle"]),
        "right_knee": _mid(hips["right_hip"], ankles["right_ankle"]),
        **ankles,
    }


def _standing_box(cx=CX, top=TOP, h=H, half_w=0.15):
    return (cx - half_w * h, top, cx + half_w * h, top + h)


def _frame(joints: dict, box, t: float = 0.0, min_conf: float = 0.3) -> PoseFrame:
    return PoseFrame(t, box, _kp(**joints), min_conf)


def _sequence(joints_at, box_at, times) -> list:
    return [_frame(joints_at(t), box_at(t), t) for t in times]


# -- reading the two-channel result ------------------------------------------
#
# The card used to flatten posture and activity into one `action` field. These
# helpers express that old view over the new contract, so assertions that only
# care about "what is this person doing, in one word" stay readable. Tests that
# care about the split assert on `posture` and `activity` directly.

ACTIVITY_TO_RULE = {v: k for k, v in RULE_TO_ACTIVITY.items()}


def _action(result):
    """Most specific label: the activity if there is one, else the posture."""
    activity = result.get("activity")
    if activity:
        return ACTIVITY_TO_RULE.get(activity["name"], activity["name"])
    return result.get("posture") or "unknown"


def _labels(result):
    """Everything the result asserts, in the rules' own spellings."""
    out = set()
    if result.get("posture"):
        out.add(result["posture"])
    activity = result.get("activity")
    if activity:
        out.add(ACTIVITY_TO_RULE.get(activity["name"], activity["name"]))
    return out


def _classify(frames, **config) -> dict:
    return PoseActionClassifier(**config).classify(frames)


def _steady(joints: dict, box, *, duration=1.5, fps=10.0) -> list:
    steps = int(round(duration * fps)) + 1
    return [_frame(joints, box, i / fps) for i in range(steps)]


# ── postures ────────────────────────────────────────────────────────────────

def test_a_standing_body_is_standing():
    result = _classify(_steady(_body(), _standing_box()))
    assert _action(result) == "standing"
    assert "standing" in _labels(result)
    assert result["posture_confidence"] >= DEFAULT_THRESHOLDS["min_confidence"]


def test_a_motionless_person_simply_has_no_activity():
    """`still` used to be a label. With posture and activity as separate
    channels it is redundant: the absence of an activity *is* stillness, and a
    field saying "not doing anything" beside an empty activity is noise."""
    result = _classify(_steady(_body(), _standing_box()))
    assert result["posture"] == "standing"
    assert result["activity"] is None


def _sitting_body(cx=CX, top=TOP, h=H) -> dict:
    hips = {"left_hip": (cx - 0.07 * h, top + 0.52 * h),
            "right_hip": (cx + 0.07 * h, top + 0.52 * h)}
    # Knees forward at hip height, shins dropping to the floor: the knee angle
    # is a right angle and the hips and knees are level.
    knees = {"left_knee": (cx - 0.07 * h + 0.20 * h, top + 0.52 * h),
             "right_knee": (cx + 0.07 * h + 0.20 * h, top + 0.52 * h)}
    ankles = {"left_ankle": (knees["left_knee"][0], top + 0.75 * h),
              "right_ankle": (knees["right_knee"][0], top + 0.75 * h)}
    shoulders = {"left_shoulder": (cx - 0.10 * h, top + 0.18 * h),
                 "right_shoulder": (cx + 0.10 * h, top + 0.18 * h)}
    return {**_head(cx, top, h), **shoulders, **hips, **knees, **ankles}


def test_a_seated_body_is_sitting_not_standing():
    box = (CX - 0.22 * H, TOP, CX + 0.30 * H, TOP + 0.80 * H)
    result = _classify(_steady(_sitting_body(), box))
    assert _action(result) == "sitting"
    assert "standing" not in _labels(result)


def _crouching_body(cx=CX, top=TOP, h=H) -> dict:
    hips = {"left_hip": (cx - 0.07 * h, top + 0.60 * h),
            "right_hip": (cx + 0.07 * h, top + 0.60 * h)}
    knees = {"left_knee": (cx - 0.07 * h + 0.12 * h, top + 0.70 * h),
             "right_knee": (cx + 0.07 * h + 0.12 * h, top + 0.70 * h)}
    ankles = {"left_ankle": (cx - 0.07 * h, top + 0.75 * h),
              "right_ankle": (cx + 0.07 * h, top + 0.75 * h)}
    shoulders = {"left_shoulder": (cx - 0.10 * h, top + 0.30 * h),
                 "right_shoulder": (cx + 0.10 * h, top + 0.30 * h)}
    return {**_head(cx, top, h), **shoulders, **hips, **knees, **ankles}


def test_a_crouching_body_is_crouching_not_sitting():
    """Folded knees past the sitting range, hips down near the ankles."""
    box = (CX - 0.20 * H, TOP, CX + 0.22 * H, TOP + 0.78 * H)
    result = _classify(_steady(_crouching_body(), box))
    assert _action(result) == "crouching"
    assert "sitting" not in _labels(result)


def _bending_body(cx=CX, top=TOP, h=H) -> dict:
    hips = {"left_hip": (cx - 0.07 * h, top + 0.52 * h),
            "right_hip": (cx + 0.07 * h, top + 0.52 * h)}
    ankles = {"left_ankle": (cx - 0.07 * h, top + 0.97 * h),
              "right_ankle": (cx + 0.07 * h, top + 0.97 * h)}
    shoulders = {"left_shoulder": (cx + 0.25 * h, top + 0.28 * h),
                 "right_shoulder": (cx + 0.25 * h, top + 0.32 * h)}
    return {
        **_head(cx + 0.30 * h, top + 0.20 * h, h), **shoulders, **hips,
        "left_knee": _mid(hips["left_hip"], ankles["left_ankle"]),
        "right_knee": _mid(hips["right_hip"], ankles["right_ankle"]),
        **ankles,
    }


def test_bending_over_is_not_crouching():
    """Straight legs are the whole difference — both put the torso over the
    floor, only one folds the knees."""
    box = (CX - 0.15 * H, TOP + 0.18 * H, CX + 0.40 * H, TOP + H)
    result = _classify(_steady(_bending_body(), box))
    assert _action(result) == "bending"
    assert "crouching" not in _labels(result)
    assert "lying" not in _labels(result)


def _lying_body(cx=CX, floor=TOP + 0.90 * H, h=H) -> dict:
    hips = {"left_hip": (cx + 0.05 * h, floor + 0.01 * h),
            "right_hip": (cx + 0.05 * h, floor + 0.03 * h)}
    shoulders = {"left_shoulder": (cx - 0.30 * h, floor),
                 "right_shoulder": (cx - 0.30 * h, floor + 0.04 * h)}
    knees = {"left_knee": (cx + 0.25 * h, floor + 0.02 * h),
             "right_knee": (cx + 0.25 * h, floor + 0.04 * h)}
    ankles = {"left_ankle": (cx + 0.42 * h, floor + 0.02 * h),
              "right_ankle": (cx + 0.42 * h, floor + 0.04 * h)}
    return {**_head(cx - 0.40 * h, floor - 0.04 * h, h),
            **shoulders, **hips, **knees, **ankles}


def _lying_box(cx=CX, floor=TOP + 0.90 * H, h=H):
    return (cx - 0.45 * h, floor - 0.06 * h, cx + 0.45 * h, floor + 0.08 * h)


def test_a_horizontal_body_is_lying():
    result = _classify(_steady(_lying_body(), _lying_box(), duration=0.5))
    assert result["posture"] == "lying"


# ── occlusion ───────────────────────────────────────────────────────────────

def test_a_body_with_no_visible_hips_says_upright_not_standing():
    """Sitting at a desk and standing behind it have the same torso axis, so
    neither may be claimed. But the axis itself is measurable, and reporting
    `unknown` threw that away too — which is what the common webcam framing
    (head, shoulders, arms, nothing below the waist) produced on every frame.

    `upright` is the honest middle: this body is vertical, and which of
    standing or sitting it is cannot be told from here.
    """
    upper = {**_head(),
             "left_shoulder": (CX - 0.10 * H, TOP + 0.18 * H),
             "right_shoulder": (CX + 0.10 * H, TOP + 0.18 * H)}
    result = _classify(_steady(upper, _standing_box()))
    assert result["posture"] == "upright"
    assert "standing" not in _labels(result)
    assert "sitting" not in _labels(result)


def test_a_nearly_invisible_body_is_unknown():
    result = _classify(_steady({"nose": (CX, TOP)}, _standing_box()))
    assert _action(result) == "unknown"
    assert result["evidence"]["visible_keypoints"] == 1


def test_an_empty_history_is_unknown_rather_than_an_error():
    assert _action(PoseActionClassifier().classify([])) == "unknown"


# ── arms ────────────────────────────────────────────────────────────────────

def _raised_arm(joints: dict, *, side="left", wrist_dx=0.0, cx=CX, top=TOP,
                h=H) -> dict:
    sign = -1.0 if side == "left" else 1.0
    shoulder = (cx + sign * 0.10 * h, top + 0.18 * h)
    wrist = (cx + sign * 0.10 * h + wrist_dx, top + 0.05 * h)
    return {**joints,
            f"{side}_shoulder": shoulder,
            f"{side}_wrist": wrist,
            f"{side}_elbow": _mid(shoulder, wrist)}


def test_a_held_raised_hand_is_raising_hand():
    frames = _steady(_raised_arm(_body()), _standing_box(), duration=1.0)
    result = _classify(frames)
    assert _action(result) == "raising_hand"
    assert result["evidence"]["side"] == "left"
    assert result["evidence"]["held_s"] >= DEFAULT_THRESHOLDS["raise_hold_s"]


def test_a_glimpsed_raised_hand_is_not_raising_hand():
    """An arm passing through shoulder height on one frame is not a raise."""
    plain, raised = _body(), _raised_arm(_body())
    frames = [_frame(plain, _standing_box(), 0.0),
              _frame(plain, _standing_box(), 0.1),
              _frame(raised, _standing_box(), 0.2)]
    assert "raising_hand" not in _labels(_classify(frames))


def test_a_hand_that_oscillates_while_raised_is_waving():
    """This is the "someone is calling the robot over" case."""
    def joints_at(t):
        return _raised_arm(_body(), wrist_dx=0.15 * H * math.sin(2 * math.pi * 1.5 * t))
    frames = _sequence(joints_at, lambda _t: _standing_box(),
                       [i / 10.0 for i in range(11)])
    result = _classify(frames)
    assert _action(result) == "waving"
    # Only the most specific activity is reported. `raising hand` is implied by
    # a wave and listing both would be noise — the single-activity channel is
    # what replaced the old multi-label list.
    assert result["activity"]["name"] == "hand waving"
    evidence = result["evidence"]
    assert evidence["reversals"] >= DEFAULT_THRESHOLDS["wave_reversals"]
    assert (DEFAULT_THRESHOLDS["wave_freq_min_hz"] <= evidence["frequency_hz"]
            <= DEFAULT_THRESHOLDS["wave_freq_max_hz"])


def test_a_still_raised_hand_is_not_waving():
    frames = _steady(_raised_arm(_body()), _standing_box(), duration=1.0)
    assert "waving" not in _labels(_classify(frames))


def test_an_arm_swinging_below_the_shoulder_is_not_waving():
    """Gating waving on the raise is what keeps a walking arm swing out."""
    def joints_at(t):
        joints = _body()
        shoulder = joints["left_shoulder"]
        wrist = (CX - 0.12 * H + 0.15 * H * math.sin(2 * math.pi * 1.5 * t),
                 TOP + 0.50 * H)
        joints["left_wrist"] = wrist
        joints["left_elbow"] = _mid(shoulder, wrist)
        return joints
    frames = _sequence(joints_at, lambda _t: _standing_box(),
                       [i / 10.0 for i in range(11)])
    actions = _labels(_classify(frames))
    assert "waving" not in actions and "raising_hand" not in actions


def test_a_straight_horizontal_arm_is_pointing_and_reports_a_direction():
    joints = _body()
    shoulder = (CX - 0.10 * H, TOP + 0.18 * H)
    wrist = (CX - 0.40 * H, TOP + 0.19 * H)
    joints.update({"left_shoulder": shoulder, "left_wrist": wrist,
                   "left_elbow": _mid(shoulder, wrist)})
    box = (CX - 0.45 * H, TOP, CX + 0.15 * H, TOP + H)
    result = _classify(_steady(joints, box))
    assert _action(result) == "pointing"
    direction = result["point_direction"]
    assert direction[0] == pytest.approx(-1.0, abs=0.05)
    assert abs(direction[1]) < 0.2
    assert np.hypot(*direction) == pytest.approx(1.0, abs=1e-2)


def test_two_wrists_across_the_midline_at_chest_height_are_arms_crossed():
    joints = _body()
    joints.update({
        "left_wrist": (CX + 0.08 * H, TOP + 0.40 * H),
        "right_wrist": (CX - 0.08 * H, TOP + 0.40 * H),
        "left_elbow": (CX - 0.18 * H, TOP + 0.35 * H),
        "right_elbow": (CX + 0.18 * H, TOP + 0.35 * H),
    })
    result = _classify(_steady(joints, _standing_box(half_w=0.20)))
    assert _action(result) == "arms_crossed"
    assert "pointing" not in _labels(result)


# ── motion ──────────────────────────────────────────────────────────────────

def test_alternating_ankles_under_an_upright_torso_is_walking():
    def joints_at(t):
        swing = 0.12 * H * math.sin(2 * math.pi * 1.0 * t)
        return _body(left_ankle_dx=swing, right_ankle_dx=-swing)
    frames = _sequence(joints_at, lambda _t: _standing_box(half_w=0.20),
                       [i / 10.0 for i in range(16)])
    result = _classify(frames)
    assert "walking" in _labels(result)
    assert _action(result) == "walking"      # above standing: more informative
    cadence = result["evidence"]["cadence_hz"]
    assert (DEFAULT_THRESHOLDS["walk_cadence_min_hz"] <= cadence
            <= DEFAULT_THRESHOLDS["walk_cadence_max_hz"])


def test_a_seated_body_shuffling_its_feet_is_not_walking():
    """Walking is gated on an upright standing torso for exactly this case."""
    def joints_at(t):
        joints = _sitting_body()
        swing = 0.12 * H * math.sin(2 * math.pi * 1.0 * t)
        joints["left_ankle"] = (joints["left_ankle"][0] + swing,
                                joints["left_ankle"][1])
        joints["right_ankle"] = (joints["right_ankle"][0] - swing,
                                 joints["right_ankle"][1])
        return joints
    box = (CX - 0.30 * H, TOP, CX + 0.40 * H, TOP + 0.80 * H)
    frames = _sequence(joints_at, lambda _t: box, [i / 10.0 for i in range(16)])
    assert "walking" not in _labels(_classify(frames))


def test_a_narrowing_shoulder_width_is_turning():
    def joints_at(t):
        half = 0.10 - 0.06 * (t / 1.5)
        return _body(shoulder_half=max(half, 0.02))
    frames = _sequence(joints_at, lambda _t: _standing_box(),
                       [i / 10.0 for i in range(16)])
    result = _classify(frames)
    assert "turning" in _labels(result)


# ── fall ────────────────────────────────────────────────────────────────────

def _fall_sequence(*, sit_before_landing=False, fps=10.0,
                   upright_s=1.0, settle_s=1.2) -> list:
    frames = []
    t = 0.0
    step = 1 / fps
    while t < upright_s:
        frames.append(_frame(_body(), _standing_box(), t))
        t += step
    if sit_before_landing:
        frames.append(_frame(_sitting_body(),
                             (CX - 0.22 * H, TOP, CX + 0.30 * H, TOP + 0.80 * H),
                             t))
        t += step
    landed_at = t
    while t < landed_at + settle_s:
        frames.append(_frame(_lying_body(), _lying_box(), t))
        t += step
    return frames


def test_a_fast_drop_into_a_sustained_horizontal_pose_is_a_fall():
    result = _classify(_fall_sequence())
    assert _action(result) == "fall"
    assert result["activity"]["name"] == "falling down"
    evidence = result["evidence"]
    assert evidence["is_fall"] is True
    assert evidence["drop_ratio"] >= DEFAULT_THRESHOLDS["fall_drop_ratio"]
    assert evidence["drop_ms"] <= DEFAULT_THRESHOLDS["fall_drop_window_s"] * 1000
    assert evidence["settle_ms"] >= DEFAULT_THRESHOLDS["fall_settle_s"] * 1000
    assert evidence["had_sitting_phase"] is False


def test_sitting_down_then_lying_down_is_lying_not_a_fall():
    """Sitting on the way down is the signature of a deliberate descent."""
    result = _classify(_fall_sequence(sit_before_landing=True))
    assert result["posture"] == "lying"
    assert "fall" not in _labels(result)
    assert result["evidence"]["had_sitting_phase"] is True
    assert result["evidence"]["is_fall"] is False


def test_someone_already_lying_down_is_not_a_fall():
    """Lying on the floor and lying on a sofa are the same terminal state —
    without the drop there is no fall to report."""
    result = _classify(_steady(_lying_body(), _lying_box(), duration=2.0))
    assert result["posture"] == "lying"
    assert "fall" not in _labels(result)
    assert result["evidence"]["is_fall"] is False
    assert result["evidence"]["reason"] == "no fast drop into the horizontal pose"


def test_a_fall_is_not_declared_before_the_body_has_settled():
    """Bending down to pick something up is horizontal too, briefly."""
    result = _classify(_fall_sequence(settle_s=0.4))
    assert "fall" not in _labels(result)
    assert result["posture"] == "lying"
    assert "is_fall" not in result["evidence"]


def test_the_fall_drop_is_measured_against_the_standing_height():
    """Normalising by the current box would divide the drop by the post-fall
    height — a person on the floor has a short, wide box."""
    frames = _fall_sequence()
    tall = max(f.height for f in frames)
    short = frames[-1].height
    assert short < tall / 2
    evidence = _classify(frames)["evidence"]
    assert evidence["drop_ratio"] < 1.0        # would exceed 1.0 if divided by `short`


def test_a_raised_fall_threshold_suppresses_the_same_fall():
    """The fall thresholds are instance config, not constants: the same fall
    measures differently depending on where the camera is."""
    frames = _fall_sequence()
    assert _action(_classify(frames)) == "fall"
    strict = _classify(frames, thresholds={"fall_drop_ratio": 0.95})
    assert strict["posture"] == "lying"
    assert strict["evidence"]["is_fall"] is False


# ── classifier plumbing ─────────────────────────────────────────────────────

def test_unknown_threshold_keys_are_ignored():
    classifier = PoseActionClassifier(thresholds={"nonsense": 1, "upright_deg": 10})
    assert "nonsense" not in classifier.thresholds
    assert classifier.thresholds["upright_deg"] == 10


def test_none_valued_thresholds_fall_back_to_the_default():
    """An unset field on a canvas card arrives as None, not as absent."""
    classifier = PoseActionClassifier(thresholds={"upright_deg": None})
    assert classifier.thresholds["upright_deg"] == DEFAULT_THRESHOLDS["upright_deg"]


def test_history_s_covers_the_whole_fall_window():
    """Ask for too little history and fall can never fire, with nothing in any
    log to say why."""
    classifier = PoseActionClassifier(action_window_s=0.5)
    th = classifier.thresholds
    assert classifier.history_s >= th["fall_drop_window_s"] + th["fall_settle_s"]


def test_the_action_window_bounds_what_the_arm_rules_see():
    """A hand raised two seconds ago is not a hand raised now."""
    frames = _steady(_raised_arm(_body()), _standing_box(), duration=1.0)
    frames += [_frame(_body(), _standing_box(), 1.0 + i / 10.0)
               for i in range(1, 12)]
    assert "raising_hand" not in _labels(_classify(frames, action_window_s=0.5))


def test_every_priority_entry_has_a_chinese_label():
    assert set(ACTION_LABELS_ZH) == set(ACTION_PRIORITY)
    assert ACTION_PRIORITY[-1] == "unknown"
    assert ACTION_PRIORITY[0] == "fall"


def test_the_catalogue_marks_fall_as_an_event_not_a_pose():
    catalogue = {entry["action"]: entry for entry in action_catalogue()}
    assert catalogue["fall"]["kind"] == "event"
    assert catalogue["standing"]["kind"] == "pose"
    assert set(EVENT_ACTIONS) == {"fall"}
    assert all(entry["label_zh"] for entry in catalogue.values())


# ── tracking ────────────────────────────────────────────────────────────────

def test_an_overlapping_box_keeps_its_track_id():
    tracker = PoseTracker()
    keypoints = [_kp(**_body())]
    first = tracker.update([_standing_box()], keypoints, 0.0)[0]
    moved = (CX - 0.15 * H + 10, TOP + 5, CX + 0.15 * H + 10, TOP + H + 5)
    second = tracker.update([moved], keypoints, 0.1)[0]
    assert first.id == second.id
    assert len(second.history) == 2


def test_a_disjoint_box_starts_a_new_track():
    """Each detection carries the keypoints that actually lie inside its box.

    This test used to hand two far-apart boxes the *same* absolute keypoint
    array, which is physically impossible — a person at (0,0,50,100) cannot
    have joints at y=260 — and it only passed while association looked at the
    box alone. Now that the visible-joint centroid is a second cue, an
    incoherent input produces an incoherent answer, so the input is fixed.
    """
    tracker = PoseTracker()
    near_box, far_box = (0, 0, 50, 100), (500, 400, 560, 500)
    near = tracker.update(
        [near_box], [_kp(**_body(cx=25, top=0, h=100))], 0.0)[0]
    far = tracker.update(
        [far_box], [_kp(**_body(cx=530, top=400, h=100))], 0.1)[0]
    assert near.id != far.id


def test_two_people_keep_separate_timelines():
    tracker = PoseTracker()
    left, right = (0, 0, 100, 300), (400, 0, 500, 300)
    keypoints = [_kp(**_body()), _kp(**_body(cx=450))]
    a1, b1 = tracker.update([left, right], keypoints, 0.0)
    # Same two people, reported in the opposite order this frame.
    b2, a2 = tracker.update([right, left], list(reversed(keypoints)), 0.1)
    assert a1.id == a2.id and b1.id == b2.id
    assert a1.id != b1.id


def test_a_track_that_vanishes_for_too_long_is_not_reused():
    tracker = PoseTracker(timeout_s=0.5)
    keypoints = [_kp(**_body())]
    first = tracker.update([_standing_box()], keypoints, 0.0)[0]
    later = tracker.update([_standing_box()], keypoints, 2.0)[0]
    assert first.id != later.id
    assert len(tracker.tracks) == 1          # the stale one is gone


def test_history_is_trimmed_to_the_configured_window():
    tracker = PoseTracker(history_s=0.5)
    keypoints = [_kp(**_body())]
    for i in range(30):
        tracks = tracker.update([_standing_box()], keypoints, i / 10.0)
    history = tracks[0].history
    assert history[-1].t - history[0].t <= 0.5
    assert len(history) <= 7


def test_tracking_survives_a_frame_with_nobody_in_it():
    tracker = PoseTracker(timeout_s=1.0)
    keypoints = [_kp(**_body())]
    first = tracker.update([_standing_box()], keypoints, 0.0)[0]
    assert tracker.update([], [], 0.1) == []
    again = tracker.update([_standing_box()], keypoints, 0.2)[0]
    assert first.id == again.id


def test_still_is_not_reported_for_a_body_too_occluded_to_place():
    """A motionless person whose posture is unreadable must come back as
    `unknown`, not as `still` — otherwise "I can't see them" is reported as a
    positive observation about them."""
    upper = {**_head(),
             "left_shoulder": (CX - 0.10 * H, TOP + 0.18 * H),
             "right_shoulder": (CX + 0.10 * H, TOP + 0.18 * H)}
    result = _classify(_steady(upper, _standing_box()))



def test_an_occluded_body_can_still_report_its_arms():
    """The arm rules only need shoulder/elbow/wrist, so a person visible from
    the waist up can still be seen calling the robot over."""
    upper = _raised_arm({**_head(),
                         "left_shoulder": (CX - 0.10 * H, TOP + 0.18 * H),
                         "right_shoulder": (CX + 0.10 * H, TOP + 0.18 * H)})
    result = _classify(_steady(upper, _standing_box(), duration=1.0))
    assert _action(result) == "raising_hand"


# ── single-image classification ─────────────────────────────────────────────

def test_a_single_frame_drops_the_hold_requirement_on_a_raised_hand():
    """The hold only exists to tell a held gesture from an arm swinging
    through shoulder height, and a photo has no "swinging through"."""
    frame = _frame(_raised_arm(_body()), _standing_box(), 0.0)
    result = PoseActionClassifier().classify_frame(frame)
    assert _action(result) == "raising_hand"
    assert result["temporal"] is False


def test_a_single_frame_says_which_actions_it_cannot_answer():
    """Waving, walking and falling are motion; one image carries none of them.

    Saying so beats answering a narrower question than was asked — a caller
    who asked "is this person waving" would otherwise get `raising_hand` back
    as though it settled the matter.
    """
    frame = _frame(_body(), _standing_box(), 0.0)
    result = PoseActionClassifier().classify_frame(frame)
    assert _action(result) == "standing"
    assert "hand waving" in result["unavailable_activities"]
    assert "falling down" in result["unavailable_activities"]



def test_a_single_frame_reports_upright_for_an_occluded_body():
    """One image of a head-and-shoulders view: the axis is measurable even
    though standing and sitting are not separable."""
    upper = {**_head(),
             "left_shoulder": (CX - 0.10 * H, TOP + 0.18 * H),
             "right_shoulder": (CX + 0.10 * H, TOP + 0.18 * H)}
    result = PoseActionClassifier().classify_frame(
        _frame(upper, _standing_box(), 0.0))
    assert result["posture"] == "upright"


# ── track continuity through a fall ─────────────────────────────────────────

def test_a_falling_person_keeps_their_track():
    """The case bounding-box overlap cannot handle, and the one that matters most.

    Measured: a standing box [260,100,380,500] against the same person's lying
    box [140,436,500,492] scores IoU 0.109 — under any usable iou_min. Before
    the centroid cue the track split at the instant of the fall, the new track
    started with an empty history, and _fall_evidence could only ever see the
    horizontal frames and report "no fast drop". Fall detection required
    continuity across precisely the event that destroys box overlap.
    """
    from plugins.pose_action import _iou
    stand, lie = _standing_box(), _lying_box()
    assert _iou(stand, lie) < 0.2, "the premise of this test is that IoU fails"

    tracker = PoseTracker(history_s=3.0)
    ids = []
    for i in range(6):
        ids.append(tracker.update([stand], [_kp(**_body())], i / 10.0)[0].id)
    for i in range(6, 18):
        ids.append(tracker.update([lie], [_kp(**_lying_body())], i / 10.0)[0].id)
    assert len(set(ids)) == 1, f"track split at the fall: {ids}"


@pytest.mark.parametrize("fps", [5, 10, 15])
def test_a_fall_is_detected_through_the_real_tracker(fps):
    """End to end: detections in, one track out, `fall` from its own history.

    The earlier fall tests fed the classifier a hand-built frame list, so they
    passed while the real pipeline could never produce that list.
    """
    tracker = PoseTracker(history_s=3.0)
    classifier = PoseActionClassifier()
    t, step = 0.0, 1.0 / fps
    track = None
    while t < 1.0:
        track = tracker.update([_standing_box()], [_kp(**_body())], t)[0]
        t += step
    while t < 2.4:
        track = tracker.update([_lying_box()], [_kp(**_lying_body())], t)[0]
        t += step
    result = classifier.classify(list(track.history))
    assert _action(result) == "fall", result
    assert result["evidence"]["is_fall"] is True


def test_an_iou_match_still_outranks_a_centroid_match():
    """Overlap is the stronger cue when it is available, so two people standing
    close together must not be swapped by whichever centroid is nearer."""
    tracker = PoseTracker()
    left, right = (0, 0, 100, 300), (90, 0, 190, 300)
    kps = [_kp(**_body(cx=50)), _kp(**_body(cx=140))]
    a1, b1 = tracker.update([left, right], kps, 0.0)
    a2, b2 = tracker.update([left, right], kps, 0.1)
    assert (a1.id, b1.id) == (a2.id, b2.id)
    assert a1.id != b1.id


def test_a_detection_with_no_visible_joints_cannot_be_matched_by_centroid():
    """No joints means no centroid, so the only cue left is overlap — and an
    invented match would hand a learned backend somebody else's history."""
    tracker = PoseTracker()
    blank = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    first = tracker.update([(0, 0, 100, 300)], [_kp(**_body())], 0.0)[0]
    far = tracker.update([(400, 0, 500, 300)], [blank], 0.1)[0]
    assert first.id != far.id


def test_the_centroid_gate_is_a_sanity_bound_not_the_discriminator():
    """Greedy nearest-match does the work; the gate only vetoes absurd jumps."""
    tracker = PoseTracker(centroid_max_travel=1.0)
    near = tracker.update([_standing_box()], [_kp(**_body())], 0.0)[0]
    # Five body heights away, with no overlap: not the same person.
    away = (CX + 5 * H, TOP, CX + 5 * H + 0.3 * H, TOP + H)
    other = tracker.update([away], [_kp(**_body(cx=CX + 5 * H))], 0.1)[0]
    assert near.id != other.id


# ── viewpoint robustness: the bug behind "standing and sitting are both wrong" ──

def _standing_with_foreshortened_legs(k, cx=CX, top=TOP, h=H):
    """A standing person whose legs are vertically compressed by `k`.

    What a low camera does: looking up at someone, the floor-to-hip span
    projects to far fewer pixels than the hip-to-head span. The person is still
    standing — knees straight, torso vertical — only image-space distances
    change. A robot camera at 0.4-1.2 m looking at someone 1-2 m away is well
    into this regime.
    """
    hip_y = top + 0.52 * h
    knee_y = hip_y + 0.225 * h * k
    ankle_y = hip_y + 0.45 * h * k
    joints = dict(_body(cx=cx, top=top, h=h))
    joints.update({
        "left_hip": (cx - 0.07 * h, hip_y), "right_hip": (cx + 0.07 * h, hip_y),
        "left_knee": (cx - 0.07 * h, knee_y), "right_knee": (cx + 0.07 * h, knee_y),
        "left_ankle": (cx - 0.07 * h, ankle_y), "right_ankle": (cx + 0.07 * h, ankle_y),
    })
    return joints, (cx - 0.15 * h, top, cx + 0.15 * h, ankle_y)


@pytest.mark.parametrize("k", [1.0, 0.6, 0.45, 0.25, 0.15])
def test_standing_survives_any_leg_foreshortening(k):
    """The regression this whole rewrite exists for.

    The first version gated `standing` on `hip_knee_dy >= 0.15` — an
    image-space length ratio — on top of the torso-tilt and knee-angle tests.
    Measured, for a person definitely standing:

        k=1.00  hip_knee_dy 0.232  knee 180  torso 0  -> standing
        k=0.45  hip_knee_dy 0.140  knee 180  torso 0  -> UNKNOWN
        k=0.25  hip_knee_dy 0.089  knee 180  torso 0  -> UNKNOWN

    The angles were right the whole way down. The ratio added no information
    and only a viewpoint dependence.
    """
    joints, box = _standing_with_foreshortened_legs(k)
    result = _classify(_steady(joints, box))
    assert _action(result) == "standing", f"k={k}: {result}"


@pytest.mark.parametrize("k", [1.0, 0.6, 0.35, 0.15])
def test_sitting_survives_any_leg_foreshortening(k):
    """And is never mistaken for standing, which is the other half of the bug:
    `sitting` used to key on the opposite side of the same fragile quantity."""
    hip_y = TOP + 0.52 * H
    joints = dict(_body())
    joints.update({
        "left_knee": (CX + 0.13 * H, hip_y), "right_knee": (CX + 0.27 * H, hip_y),
        "left_ankle": (CX + 0.13 * H, hip_y + 0.23 * H * k),
        "right_ankle": (CX + 0.27 * H, hip_y + 0.23 * H * k),
    })
    box = (CX - 0.22 * H, TOP, CX + 0.34 * H, hip_y + 0.23 * H * k + 10)
    result = _classify(_steady(joints, box))
    assert _action(result) == "sitting", f"k={k}: {result}"
    assert "standing" not in _labels(result)


def test_a_cropped_view_with_knees_but_no_feet_still_reads_as_standing():
    """The common robot framing: someone close enough that their feet are out
    of frame. Requiring the whole hip-knee-ankle chain gave them no posture."""
    joints = {k: v for k, v in _body().items()
              if k not in ("left_ankle", "right_ankle")}
    box = (CX - 0.15 * H, TOP, CX + 0.15 * H, TOP + 0.80 * H)
    result = _classify(_steady(joints, box))
    assert _action(result) == "standing"


def test_a_torso_only_view_reports_upright_and_nothing_sharper():
    """With no knee in frame there is no hip angle, so standing and sitting
    stay indistinguishable — but the body is plainly vertical and that is worth
    saying."""
    joints = {k: v for k, v in _body().items()
              if "knee" not in k and "ankle" not in k and "hip" not in k}
    box = (CX - 0.15 * H, TOP, CX + 0.15 * H, TOP + 0.55 * H)
    result = _classify(_steady(joints, box))
    assert result["posture"] == "upright"
    assert "standing" not in _labels(result) and "sitting" not in _labels(result)


def test_a_head_with_no_shoulders_still_reports_nothing():
    """The real limit. Without shoulders there is no body axis at all, and
    `upright` would be a guess rather than a coarser truth."""
    box = (CX - 0.1 * H, TOP, CX + 0.1 * H, TOP + 0.1 * H)
    assert _classify(_steady(_head(), box))["posture"] is None


def test_the_posture_rules_use_no_image_space_length_ratios():
    """A guard on the rule that produced the bug. Every posture threshold must
    be an angle; a `_dy`/ratio threshold creeping back in is the regression."""
    from plugins.pose_action import DEFAULT_THRESHOLDS as TH
    posture_keys = [k for k in TH
                    if k.endswith("_deg") or "dy" in k or k.endswith("_aspect")]
    offenders = [k for k in posture_keys if "dy" in k]
    assert offenders == [], f"image-space length thresholds are back: {offenders}"


def test_a_squat_is_crouching_only_and_not_also_sitting():
    """Both fired at first. Priority hid it, but `actions` still carried it."""
    box = (CX - 0.20 * H, TOP, CX + 0.22 * H, TOP + 0.78 * H)
    result = _classify(_steady(_crouching_body(), box))
    assert _action(result) == "crouching"
    assert "sitting" not in _labels(result)


def test_a_foreshortened_knee_angle_cannot_veto_an_open_hip():
    """2D angles are invariant to scale and rotation — not to foreshortening.

    Measured on a real skeleton from the engine, legs compressed to 0.35 of
    their vertical extent: the hip angle moved 170.3 -> 162.7 deg while the knee
    angle moved 151.1 -> 126.0. Thigh and shin are never exactly collinear, and
    squashing y amplifies whatever lateral offset they have. Requiring a
    straight knee for `standing` therefore repeated, one level down, the mistake
    the `hip_knee_dy` rewrite removed: a fragile measurement vetoing a robust
    one. The knee spans two short noisy segments; the hip spans the torso.
    """
    joints = dict(_body())
    hip_y = TOP + 0.52 * H
    # Knees and ankles pulled towards the hip, with the small lateral offset a
    # real leg has, which is what bends the apparent knee angle.
    for name, frac, dx in (("knee", 0.225, 0.02), ("ankle", 0.45, -0.01)):
        for side in ("left", "right"):
            key = f"{side}_{name}"
            sign = -1.0 if side == "left" else 1.0
            joints[key] = (CX + sign * 0.07 * H + dx * H,
                           hip_y + frac * H * 0.35)
    box = (CX - 0.15 * H, TOP, CX + 0.15 * H, hip_y + 0.45 * H * 0.35 + 10)
    frame = _frame(joints, box)
    assert frame.hip_deg is not None and frame.hip_deg >= 145
    assert frame.knee_deg < 150, "the premise: the knee angle has bent"
    assert _action(_classify(_steady(joints, box))) == "standing"


def test_a_squat_can_still_veto_standing():
    """The knee keeps one job: objecting to a clearly folded leg. Without that
    the loosened rule would call a squat standing."""
    box = (CX - 0.20 * H, TOP, CX + 0.22 * H, TOP + 0.78 * H)
    result = _classify(_steady(_crouching_body(), box))
    assert _action(result) == "crouching"
    assert "standing" not in _labels(result)


def test_a_straight_knee_alone_is_enough_when_the_hip_is_unreadable():
    """Either robust cue suffices; the rule is a disjunction, not a chain."""
    joints = {k: v for k, v in _body().items() if "shoulder" not in k}
    joints.update({"left_shoulder": (CX - 0.10 * H, TOP + 0.18 * H)})
    frame = _frame(joints, _standing_box())
    assert frame.knee_deg >= 150
    assert _action(_classify(_steady(joints, _standing_box()))) == "standing"


# ── a person on the ground is not gesturing ─────────────────────────────────
#
# Reported from a rig: two photographs of someone who had fallen came back as
# `pointing` and as `raising_hand`. Three bugs, all of the same family as the
# `hip_knee_dy` one — a claim about the *body* measured against the *image*.

_FLOOR = TOP + 0.90 * H


def _lying_with_arm(dx, dy):
    """A fallen person whose left arm is displaced (dx, dy) from the shoulder.

    In `_lying_body` the head is at -x and the feet at +x, so the body's own
    "down" points along +x. An arm at -x is stretched past the head; an arm at
    +dy is held perpendicular to the body, i.e. up towards the ceiling.
    """
    joints = dict(_lying_body())
    shoulder = (CX - 0.30 * H, _FLOOR)
    joints.update({
        "left_shoulder": shoulder,
        "left_elbow": (shoulder[0] + dx * 0.5, shoulder[1] + dy * 0.5),
        "left_wrist": (shoulder[0] + dx, shoulder[1] + dy),
    })
    box = (min(CX - 0.55 * H, shoulder[0] + dx - 10), _FLOOR - 0.10 * H,
           CX + 0.45 * H, _FLOOR + 0.10 * H)
    return joints, box


def test_a_fallen_person_with_an_arm_along_their_body_is_not_pointing():
    """The reported failure. `pointing` measured the arm against the image's
    horizon, and a person on the ground is horizontal by construction — so a
    straight arm resting alongside them scored as pointing every time."""
    joints, box = _lying_with_arm(+0.15 * H, +0.02 * H)
    result = _classify(_steady(joints, box, duration=1.5, fps=12))
    assert result["posture"] == "lying"
    assert "pointing" not in _labels(result)
    assert "raising_hand" not in _labels(result)


def test_a_fallen_person_is_reported_as_lying_even_with_the_hips_occluded():
    """The other half of it. `lying` reached the body axis only through
    `torso_deg`, which needs shoulders *and* hips — so on exactly the frames
    where "this person is on the ground" matters most, the posture rules
    returned nothing and the arm rules were left to name the frame."""
    joints, box = _lying_with_arm(+0.15 * H, +0.02 * H)
    joints = {k: v for k, v in joints.items() if "hip" not in k}
    frame = _frame(joints, box)
    assert frame.torso_deg is None, "the premise: no hips, so no torso angle"
    assert frame.body_down is not None, "the head-to-shoulder fallback"
    result = _classify(_steady(joints, box, duration=1.5, fps=12))
    assert result["posture"] == "lying"
    assert "pointing" not in _labels(result)


def test_a_fallen_person_can_still_be_seen_raising_an_arm():
    """Not suppressed, deliberately: someone on the floor waving for help is
    exactly the case this card exists for. The posture wins the primary label,
    the gesture survives in `actions`."""
    joints, box = _lying_with_arm(-0.20 * H, +0.01 * H)   # past the head
    result = _classify(_steady(joints, box, duration=1.5, fps=12))
    assert result["posture"] == "lying"
    assert "raising_hand" in _labels(result)


def test_an_arm_perpendicular_to_a_fallen_body_is_pointing():
    """"Horizontal" now means perpendicular to the body, so an arm held up
    towards the ceiling by someone lying down reads as pointing — which is what
    it is."""
    joints, box = _lying_with_arm(+0.02 * H, -0.18 * H)
    result = _classify(_steady(joints, box, duration=1.5, fps=12))
    assert result["posture"] == "lying"
    assert "pointing" in _labels(result)


def test_arm_thresholds_do_not_scale_with_the_bounding_box():
    """A lying person's box is ~56 px tall where they stood 400, so every
    fraction-of-box-height threshold collapsed to a few pixels and noise walked
    through it. The scale has to be the body's."""
    joints, box = _lying_with_arm(+0.15 * H, +0.02 * H)
    frame = _frame(joints, box)
    assert frame.height < 0.25 * H, "the premise: the box has collapsed"
    assert frame.body_scale > 0.25 * H, "the body scale must not collapse with it"


def test_the_body_frame_reduces_to_the_image_frame_when_standing():
    """The rewrite must not move anything for an upright person."""
    frame = _frame(_body(), _standing_box())
    assert frame.body_down is not None
    assert frame.body_down[1] > 0.99, "down is down for someone standing"


# ── two real photographs of fallen people ───────────────────────────────────
#
# Keypoints as the engine actually produced them on a rig, pinned here. They are
# the only cases in this file that are not synthetic, and between them they mark
# where the geometry works and where it stops.

def _photo_frame(joints_px, box, image_size, t=0.0):
    """A frame from literal engine output. Unlisted joints are invisible."""
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    for name, (x, y, v) in joints_px.items():
        keypoints[COCO_INDEX[name]] = (x, y, v)
    return PoseFrame(t, box, keypoints, 0.3, image_size=image_size)


# Trampoline: on their back, head down, legs in the air. Reported as `pointing`.
_INVERTED = {
    "nose": (140, 211, 0.92),
    "left_shoulder": (107, 218, 1.0), "right_shoulder": (197, 212, 1.0),
    "left_hip": (86, 155, 1.0), "right_hip": (144, 148, 1.0),
    "left_knee": (53, 118, 1.0), "right_knee": (162, 103, 1.0),
    "left_ankle": (27, 41, 1.0), "right_ankle": (131, 15, 0.99),
}
_INVERTED_BOX = (15, 0, 274, 250)

# Elderly person fallen on pavement, camera above and looking along the body.
# Reported as `raising_hand`.
_ALONG_AXIS = {
    "nose": (168, 200, 0.98),
    "left_shoulder": (252, 175, 0.99), "right_shoulder": (187, 243, 1.0),
    "left_hip": (343, 320, 1.0), "right_hip": (291, 352, 1.0),
    "left_knee": (316, 442, 1.0), "right_knee": (232, 438, 1.0),
    "left_ankle": (409, 560, 0.66), "right_ankle": (346, 533, 0.97),
}
_ALONG_AXIS_BOX = (6, 132, 427, 577)


def test_an_inverted_body_is_lying_not_gesturing():
    """Head down, legs in the air. Came back as `pointing`.

    Torso tilt read 30.2 deg — "upright" — because `_tilt_from_vertical_deg`
    takes the absolute value of the vertical component, so an inverted body and
    an upright one are the same number to it. Aspect was 1.04, under the 1.2
    gate. Both halves of the old `lying` test failed on a person who was plainly
    on their back.
    """
    frame = _photo_frame(_INVERTED, _INVERTED_BOX, (300, 400))
    assert frame.extent is not None and frame.extent < -2.0, frame.extent
    frames = [_photo_frame(_INVERTED, _INVERTED_BOX, (300, 400), i / 12)
              for i in range(20)]
    result = PoseActionClassifier().classify(frames)
    assert result["posture"] == "lying", result
    assert "pointing" not in _labels(result)
    assert "standing" not in _labels(result)


def test_a_body_seen_along_its_own_axis_is_not_separable_from_a_standing_one():
    """The limit, pinned so nobody tries to close it with a threshold.

    A person lying on pavement, photographed from above and along their body.
    The projection puts head above hips above feet exactly as it does for
    someone standing: the measured head-to-foot span is **+2.17** body scales
    against **+2.68** for an upright reference — a 19% gap.

    A threshold could separate *these two samples*. What makes it unusable is
    what else lives in the 1.0-2.4 band: a standing person with their feet
    cropped, or knees slightly bent, or a child's proportions. Catching this
    fall means calling those people fallen, and a false "someone has collapsed"
    makes the robot drop what it is doing to ask whether a standing person is
    hurt.

    So this case is left to the cues that do carry the information: the ground
    plane (depth), the camera pose, or the *transition* that put them there —
    which is why `fall` keys on the drop rather than on the terminal pose.
    """
    frame = _photo_frame(_ALONG_AXIS, _ALONG_AXIS_BOX, (427, 583))
    assert frame.extent == pytest.approx(2.17, abs=0.1)
    standing = _frame(_body(), _standing_box())
    assert standing.extent == pytest.approx(2.68, abs=0.1)
    gap = (standing.extent - frame.extent) / standing.extent
    assert gap < 0.20, f"the two are {gap:.0%} apart, not separable in practice"
    # And it is correctly NOT claimed as lying, rather than forced with a
    # threshold that would misfire on ordinary standing people.
    frames = [_photo_frame(_ALONG_AXIS, _ALONG_AXIS_BOX, (427, 583), i / 12)
              for i in range(20)]
    assert PoseActionClassifier().classify(frames)["posture"] != "lying"


def test_the_aspect_ratio_no_longer_vetoes_lying():
    """Both photographs had a *taller-than-wide* box — 1.04 and 0.95 — because a
    fallen body photographed end-on is tall in the image. Aspect only separates
    anything for a camera level with the body and perpendicular to it."""
    inverted = _photo_frame(_INVERTED, _INVERTED_BOX, (300, 400))
    assert inverted.aspect < DEFAULT_THRESHOLDS["lying_aspect"]
    frames = [_photo_frame(_INVERTED, _INVERTED_BOX, (300, 400), i / 12)
              for i in range(20)]
    assert PoseActionClassifier().classify(frames)["posture"] == "lying"


# ── label stability ─────────────────────────────────────────────────────────

def test_a_label_does_not_flip_on_a_threshold_crossing():
    """Every threshold here is a cliff, and at the boundaries the raw answer
    flipped on **every frame**: 9 changes in 9 comparisons with the model's
    score oscillating around 0.40, 6 in 6 with motion oscillating around its
    gate. A label that alternates twelve times a second is worse than one that
    is simply wrong — nothing downstream can act on it, an operator cannot read
    it, and an agent asked to react gets a different world each turn.
    """
    noisy = ["standing", "waving", "standing", "waving", "standing", "waving"]
    stabiliser = LabelStabiliser(hold=3)
    out = [stabiliser.update(label)[0] for label in noisy]
    assert out == ["standing"] * 6
    assert len(set(out)) == 1


def test_a_sustained_change_does_get_through():
    stabiliser = LabelStabiliser(hold=3)
    for label in ["standing"] * 3:
        stabiliser.update(label)
    out = [stabiliser.update("waving")[0] for _ in range(5)]
    assert out == ["standing", "standing", "waving", "waving", "waving"]


def test_the_latency_of_the_hold_is_exactly_hold_frames():
    """The cost, paid on every real change. Measured rather than asserted
    loosely, because it is the half of this trade that hurts: at 12 fps a hold
    of 3 is 167 ms before a genuine new action is reported."""
    for hold in (1, 2, 3, 5):
        stabiliser = LabelStabiliser(hold=hold)
        sequence = ["standing"] * 5 + ["waving"] * 10
        out = [stabiliser.update(label)[0] for label in sequence]
        assert out.index("waving") - 5 == hold - 1


def test_the_first_label_is_adopted_immediately():
    """Nothing to be stable about yet, and making a new person wait would mean
    somebody walking into frame is `unknown` for a quarter of a second."""
    assert LabelStabiliser(hold=5).update("standing") == ("standing", None)


def test_the_challenger_is_reported_while_it_is_being_held():
    """So the raw instantaneous answer stays visible — otherwise hysteresis is
    indistinguishable from the classifier being stuck."""
    stabiliser = LabelStabiliser(hold=3)
    stabiliser.update("standing")
    assert stabiliser.update("waving") == ("standing", "waving")


def test_an_interrupted_challenge_starts_over():
    """Two frames of `waving`, one of something else, two more of `waving` is
    not three consecutive — that is the flapping this exists to absorb."""
    stabiliser = LabelStabiliser(hold=3)
    stabiliser.update("standing")
    stabiliser.update("waving")
    stabiliser.update("waving")
    stabiliser.update("sitting")
    assert stabiliser.update("waving")[0] == "standing"


def test_each_person_is_stabilised_separately():
    """A shared stabiliser would let one person's gesture suppress another's."""
    tracker = PoseTracker(label_hold=3)
    left, right = (0, 0, 100, 300), (400, 0, 500, 300)
    kps = [_kp(**_body(cx=50, top=0, h=300)), _kp(**_body(cx=450, top=0, h=300))]
    a, b = tracker.update([left, right], kps, 0.0)
    assert a.posture_stabiliser is not b.posture_stabiliser
    assert a.activity_stabiliser is not b.activity_stabiliser
    a.posture_stabiliser.update("standing")
    assert b.posture_stabiliser.current is None
