#!/usr/bin/env python3
"""
plugins/pose_action.py — turn a COCO-17 keypoint sequence into an action label.

Pure numpy. No model, no weights, no GPU — which is the point: this layer is
fully testable on a laptop, and the only part of the pose plugin that needs
hardware is `VisionEngineSession.infer()`. Same split as
`actucore/tests/test_smolvla_provider.py` gets from `_build_policy`/`_predict`.

Three things shape everything here.

**Two output fields, not one.** `action` is the single primary label (what goes
into the LLM prompt) and `actions` is every label that holds. "Standing while
waving" is two things that are simultaneously true, and collapsing them into one
string loses whichever matters to the caller.

**Normalised by body height, never by pixels.** Every threshold below is a
fraction of the person's bounding-box height, so a person 3 m away and the same
person 1 m away produce the same numbers. A pixel threshold would be a distance
threshold wearing a disguise.

**Occlusion yields `unknown`, not a guess.** A person behind a desk has no
visible hips or knees, and there is no way to tell sitting from standing from
the torso alone. Inferring one anyway is worse than silence — "I don't know"
and "nothing is wrong" have to be two different answers, the same rule
`visual_depth`'s lens-barrel mask exists for. Arm labels still work in that
case, because they only need shoulder/elbow/wrist.

`fall` is the one label that is not a posture: see `_fall_evidence`.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from plugins.vision_runtime import COCO_INDEX, N_KEYPOINTS

log = logging.getLogger(__name__)

# ── label set ────────────────────────────────────────────────────────────────

# Two vocabularies, because two different questions are being answered and
# neither answer substitutes for the other.
#
# A **posture** is the shape a body is in. It is a state, readable from one
# frame, and a person always has one.
#
# An **activity** is what somebody is doing. It is a process, needs motion to be
# visible at all, and a person may not be doing anything nameable.
#
# They used to be flattened into one `action` field picked by priority across
# both, which meant `standing` and `reading` — different kinds of fact — shared
# a field, and the field's vocabulary was the union of everything. A consumer
# could not rely on it coming from a known set.

#: What the geometry can say about a body's shape. Closed.
#:
#: `upright` is the coarse one: torso vertical, legs not readable. It exists
#: because "I cannot tell standing from sitting" and "I know nothing about this
#: body" were being reported as the same thing, and they are not. The common
#: webcam framing — head, shoulders and arms, nothing below the waist — gave
#: `unknown` on every frame while the body axis was perfectly measurable.
POSTURES = ("lying", "crouching", "sitting", "bending", "standing", "upright")

#: Activities the geometry can read without a model. Open to the model's own
#: vocabulary on top — see plugins/pose_stgcn.py.
GEOMETRY_ACTIVITIES = ("falling down", "hand waving", "point to something",
                       "raising hand", "arms crossed", "walking", "turning")

#: Order within each vocabulary, most informative first. `standing` is last
#: among postures for the reason `walking` beat it before: it says the least.
#: Most specific first: `upright` only wins when nothing sharper holds.
POSTURE_PRIORITY = ("lying", "crouching", "sitting", "bending", "standing",
                    "upright")

#: Within activities: the one a robot acts on first, then gestures aimed at it,
#: then what somebody is doing on their own.
ACTIVITY_PRIORITY = ("falling down", "hand waving", "point to something",
                     "raising hand", "arms crossed", "turning", "walking")

#: Kept for the geometry's internal rule names, which are shorter than the NTU
#: spellings they map onto.
RULE_TO_ACTIVITY = {
    "fall": "falling down",
    "waving": "hand waving",
    "pointing": "point to something",
    "raising_hand": "raising hand",
    "arms_crossed": "arms crossed",
    "walking": "walking",
    "turning": "turning",
}

ACTION_PRIORITY = (
    "fall", "lying", "waving", "raising_hand", "pointing", "arms_crossed",
    "crouching", "sitting", "bending", "walking", "turning", "standing",
    "still", "unknown",
)

ACTIONS = tuple(a for a in ACTION_PRIORITY if a != "unknown")

EVENT_ACTIONS = ("fall",)

#: Activities that cannot exist in a single image, however good it is. Waving is
#: motion that comes back, walking is a cadence, a fall is a transition — one
#: frame carries none of them.
TEMPORAL_ACTIVITIES = ("falling down", "hand waving", "walking", "turning")
TEMPORAL_ACTIONS = TEMPORAL_ACTIVITIES

#: Chinese names for the activities the geometry can read. The model's own
#: classes carry theirs in plugins/pose_stgcn.py NTU60.
ACTIVITY_LABELS_ZH = {
    "falling down": "跌倒",
    "hand waving": "挥手",
    "point to something": "指向某处",
    "raising hand": "举手",
    "arms crossed": "抱臂",
    "walking": "走动",
    "turning": "转身",
}

#: Chinese names for the postures.
POSTURE_LABELS_ZH = {
    "standing": "站立", "sitting": "坐", "crouching": "蹲",
    "bending": "弯腰", "lying": "躺", "upright": "直立(分不清站/坐)",
}

# Kept while callers migrate off the flattened vocabulary.
ACTION_LABELS_ZH = {
    "standing": "站立",
    "sitting": "坐",
    "crouching": "蹲",
    "bending": "弯腰",
    "lying": "躺",
    "raising_hand": "举手",
    "waving": "挥手/招手",
    "pointing": "指向",
    "arms_crossed": "抱臂",
    "walking": "走动",
    "turning": "转身",
    "still": "静止",
    "fall": "跌倒",
    "unknown": "不确定",
}

# Every threshold the rules use, in one place, so the plugin can expose them
# per instance. The fall ones in particular are NOT constants: `drop_ratio`
# measured for the same fall differs with camera height, pitch and focal
# length, so a default that works on one rig is wrong on the next.
DEFAULT_THRESHOLDS = {
    # visibility
    "kpt_confidence": 0.3,
    # posture
    "upright_deg": 25.0,          # torso within this of vertical = upright
    "bend_deg": 35.0,             # torso past this = leaning
    "lying_deg": 60.0,            # torso past this = horizontal
    "lying_aspect": 1.2,          # bbox w/h past this corroborates lying
    # Head-to-foot image-vertical span, in body scales. Standing measures
    # +2.5..+3.0 on real photographs; a body on the floor seen side-on about 0;
    # an inverted one negative. See _head_to_foot_extent for the case this
    # cannot see.
    "upright_extent": 2.0,
    "lying_extent": 1.0,
    "leg_straight_deg": 150.0,    # knee angle past this = straight leg
    "knee_bent_max_deg": 140.0,   # sitting: knee not straighter than this
    "knee_bent_min_deg": 70.0,    # crouching: knee folded past this
    # Hip angle (shoulder-hip-knee). ~90 deg seated, ~175 deg standing. These
    # replaced three image-space `dy` ratios that could not survive a camera
    # below eye level — see _posture_labels.
    "hip_folded_min_deg": 55.0,   # sitting: thigh folded up to at least here
    "hip_folded_max_deg": 135.0,  # sitting: and not beyond here
    "hip_open_min_deg": 145.0,    # standing: hip essentially unfolded
    # arms
    "raise_wrist_above_shoulder": 0.10,   # (y_shoulder - y_wrist) / h
    "elbow_open_deg": 90.0,
    "arm_straight_deg": 150.0,
    "point_horizontal_deg": 30.0,
    "raise_hold_s": 0.3,
    "wave_reversals": 2,
    "wave_amplitude": 0.08,       # wrist x travel / h
    "wave_freq_min_hz": 0.5,
    "wave_freq_max_hz": 4.0,
    # motion
    "still_speed": 0.06,          # body-heights per second
    "walk_amplitude": 0.05,       # ankle separation travel / h
    "walk_cadence_min_hz": 0.7,
    "walk_cadence_max_hz": 3.0,
    "turn_width_change": 0.30,    # shoulder-width change across the window
    # fall — tune these on the rig, with the camera where it will actually be
    "fall_drop_ratio": 0.35,      # hip drop / standing height
    "fall_drop_window_s": 0.6,
    "fall_settle_s": 1.0,
    # output
    "min_confidence": 0.45,       # below this the primary label is `unknown`
}

_K = COCO_INDEX


# ── small geometry helpers ───────────────────────────────────────────────────

def _visible(keypoints: np.ndarray, index: int, min_conf: float) -> bool:
    return bool(keypoints[index, 2] >= min_conf)


def _point(keypoints: np.ndarray, name: str, min_conf: float) -> Optional[np.ndarray]:
    index = _K[name]
    if not _visible(keypoints, index, min_conf):
        return None
    return keypoints[index, :2].astype(np.float32)


def _midpoint(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Midpoint of two joints, or the one that is visible, or None.

    Falling back to a single side is deliberate: a person seen from 45 degrees
    routinely has one hip occluded, and refusing to produce a torso axis for
    that is refusing to classify most real frames.
    """
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def _mid_of(points: list) -> Optional[np.ndarray]:
    """Mean of however many of a joint pair came back visible."""
    if not points:
        return None
    return np.mean(np.stack(points), axis=0).astype(np.float32)


def _angle_deg(a: Optional[np.ndarray], b: Optional[np.ndarray],
               c: Optional[np.ndarray]) -> Optional[float]:
    """Interior angle at `b`, in degrees. 180 = straight."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.dot(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _tilt_from_vertical_deg(vector: Optional[np.ndarray]) -> Optional[float]:
    """Angle between a vector and the image's vertical axis, 0-90.

    `abs` on the dot product so an inverted torso (someone upside down) reads
    as 0 rather than 180: what the postures care about is whether the body is
    aligned with gravity, not which end is up.
    """
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return math.degrees(math.acos(min(1.0, abs(float(vector[1])) / norm)))


def _body_down(frame) -> Optional[np.ndarray]:
    """Unit vector pointing from the head towards the feet, in image space.

    The body's own "down". Every arm rule needs it, because "the wrist is above
    the shoulder" and "the arm is horizontal" are claims about the *body*, not
    about the image — and a person lying on the ground has a body whose down
    points sideways on screen.

    Three sources, strongest first:

    1. shoulder-mid → hip-mid. The torso, longest and best-detected segment.
    2. nose → shoulder-mid. Available whenever the head and shoulders are, which
       is the case the hips-occluded failure lives in: a fallen person seen from
       a robot usually shows head and shoulders and nothing below.
    3. the ears' midpoint → shoulder-mid, for a head seen from behind.

    No PCA fallback. The principal axis of the visible joints is tempting and is
    wrong for the pose that matters: a standing person with both arms out is
    wider than they are tall when only the upper body is in frame, so the major
    axis turns horizontal and the body reads as lying down.
    """
    if frame.shoulder is None:
        return None
    if frame.hip is not None:
        vector = frame.hip - frame.shoulder
    else:
        head = frame.joint("nose")
        if head is None:
            head = _mid_of([p for p in (frame.joint("left_ear"),
                                        frame.joint("right_ear"))
                            if p is not None])
        if head is None:
            return None
        vector = frame.shoulder - head
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return (vector / norm).astype(np.float32)


def _head_to_foot_extent(frame) -> Optional[float]:
    """Image-vertical distance from the head to the feet, in body scales.

    Measured on real photographs through the real engine:

        standing (reference)                     +2.5 to +3.0
        fallen, camera side-on                    about 0
        inverted (head down, legs up)             -2.50
        fallen, camera looking ALONG the body     +2.17   <- indistinguishable

    The first three are what makes this a better `lying` cue than the torso's
    image-space angle: it keys on the fact that a body on the floor has its head
    and its feet at the same *height*, which is a statement about the world and
    survives the camera tilting down. Torso angle and bbox aspect do not — a
    fallen person photographed from above measured 37.4 deg of tilt and an aspect
    of 0.95, failing both gates.

    The fourth line is the limit, and it is a real one: when the camera looks
    down the length of a fallen body, the projection puts the head above the
    feet exactly as it does for someone standing. That information is not in the
    skeleton. No threshold recovers it — it needs the ground plane (depth), the
    camera pose, or the *transition* that got them there.
    """
    nose = frame.joint("nose")
    if nose is None:
        nose = _mid_of([p for p in (frame.joint("left_ear"),
                                    frame.joint("right_ear")) if p is not None])
    feet = _mid_of([p for p in (frame.joint("left_ankle"),
                                frame.joint("right_ankle")) if p is not None])
    if feet is None:
        # Hips are a usable stand-in at roughly a third of the span; a person
        # whose legs are out of frame still has a head and hips.
        feet = frame.hip
        if feet is None or nose is None or frame.body_scale is None:
            return None
        return float(feet[1] - nose[1]) / frame.body_scale * 2.9
    if nose is None or frame.body_scale is None:
        return None
    return float(feet[1] - nose[1]) / frame.body_scale


def _body_scale(frame) -> Optional[float]:
    """Pixel length the arm thresholds are fractions of.

    **Never the bounding box.** A person lying down has a box 56 px tall where
    they stood 400, so every fraction-of-box-height threshold collapses to a few
    pixels and sensor noise walks straight through it — which is half of why a
    fallen person was reported as raising a hand.

    Torso length when the hips are visible. Otherwise the head-to-shoulder span
    scaled up: on the COCO skeleton nose-to-shoulder runs about 0.12 of standing
    height against 0.34 for shoulder-to-hip, so ~2.8x. An estimate, and a stable
    one — it does not change when the person changes posture, which is the whole
    requirement.
    """
    if frame.torso_px is not None:
        return frame.torso_px
    if frame.shoulder is None:
        return None
    head = frame.joint("nose")
    if head is None:
        head = _mid_of([p for p in (frame.joint("left_ear"),
                                    frame.joint("right_ear"))
                        if p is not None])
    if head is None:
        return None
    span = float(np.linalg.norm(frame.shoulder - head))
    return max(span * 2.8, 1.0)


def body_height(box) -> float:
    """The scale every threshold is a fraction of. Floored, never zero."""
    return max(float(box[3]) - float(box[1]), 1.0)


# ── per-frame features ──────────────────────────────────────────────────────

class PoseFrame:
    """Everything a single frame can say about one person.

    Separated from the rules so the rules read as thresholds on named
    quantities, and so a feature can be asserted on directly in a test.
    """

    __slots__ = ("t", "box", "keypoints", "height", "min_conf", "image_size",
                 "shoulder", "hip", "torso_deg", "knee_deg", "hip_deg",
                 "hip_knee_dy", "hip_ankle_dy", "aspect", "torso_px",
                 "body_down", "body_scale", "extent", "has_torso", "has_legs")

    def __init__(self, t: float, box, keypoints: np.ndarray, min_conf: float,
                 image_size=None):
        # Checked here rather than at each use: this class indexes fixed COCO
        # slots by name, so a differently-sized array fails as an IndexError
        # from somewhere deep in the geometry — which reads as a bug in the
        # rules rather than as the wrong input it is.
        shape = getattr(keypoints, "shape", None)
        if shape is None or len(shape) != 2 or shape[0] != N_KEYPOINTS or shape[1] < 3:
            raise ValueError(
                f"keypoints must be ({N_KEYPOINTS}, 3+) — x, y and visibility "
                f"for each COCO-17 joint; got {shape}"
            )
        self.t = float(t)
        # The frame these pixel coordinates are in. The geometry rules never
        # need it — every threshold there is relative to the body — but a
        # learned backend does: PYSKL normalises a skeleton by the *frame*,
        # because where the person is and how large they appear within it is
        # information the network was trained with. Optional, and None is
        # honest; plugins/pose_stgcn.py refuses rather than guessing, since
        # guessing it from the bounding box would be a silent misnormalisation
        # and those produce confident nonsense rather than errors.
        self.image_size = (tuple(int(v) for v in image_size)
                           if image_size and image_size[0] and image_size[1]
                           else None)
        self.box = [float(v) for v in box]
        self.keypoints = keypoints
        self.min_conf = float(min_conf)
        self.height = body_height(box)
        width = max(self.box[2] - self.box[0], 1.0)
        self.aspect = width / self.height

        ls = _point(keypoints, "left_shoulder", min_conf)
        rs = _point(keypoints, "right_shoulder", min_conf)
        lh = _point(keypoints, "left_hip", min_conf)
        rh = _point(keypoints, "right_hip", min_conf)
        self.shoulder = _midpoint(ls, rs)
        self.hip = _midpoint(lh, rh)
        self.has_torso = self.shoulder is not None and self.hip is not None
        # Shoulder-to-hip distance in pixels: a body scale that does not change
        # when the legs leave the frame, unlike the bounding box. Used for the
        # arm and motion thresholds, which are the ones a crop silently
        # rescaled — measured bias was up to 2x at a waist-up crop.
        self.torso_px = (None if not self.has_torso else
                         max(float(np.linalg.norm(self.hip - self.shoulder)), 1.0))
        self.torso_deg = _tilt_from_vertical_deg(
            None if not self.has_torso else self.hip - self.shoulder)

        # Knee angle: the mean over whichever legs are fully visible. A single
        # visible leg is enough — see _midpoint.
        knees, knee_pts, ankle_pts = [], [], []
        for side in ("left", "right"):
            hip_p = _point(keypoints, f"{side}_hip", min_conf)
            knee_p = _point(keypoints, f"{side}_knee", min_conf)
            ankle_p = _point(keypoints, f"{side}_ankle", min_conf)
            angle = _angle_deg(hip_p, knee_p, ankle_p)
            if angle is not None:
                knees.append(angle)
            if knee_p is not None:
                knee_pts.append(knee_p)
            if ankle_p is not None:
                ankle_pts.append(ankle_p)
        self.knee_deg = float(np.mean(knees)) if knees else None

        # Hip angle: shoulder-hip-knee, i.e. how far the thigh is folded up
        # towards the chest. This is what actually separates sitting from
        # standing, and it is an ANGLE — see the class docstring on why the
        # image-space version of that test could not work.
        hips_angles = []
        for side in ("left", "right"):
            hip_p = _point(keypoints, f"{side}_hip", min_conf)
            knee_p = _point(keypoints, f"{side}_knee", min_conf)
            shoulder_p = _point(keypoints, f"{side}_shoulder", min_conf)
            # Explicit None test, never `or`: these are numpy arrays, and
            # `array or fallback` raises on truthiness rather than falling back.
            anchor = shoulder_p if shoulder_p is not None else self.shoulder
            angle = _angle_deg(anchor, hip_p, knee_p)
            if angle is not None:
                hips_angles.append(angle)
        self.hip_deg = float(np.mean(hips_angles)) if hips_angles else None

        # Legs are "readable" as soon as the hip angle or the knee angle is
        # available. Requiring the full hip-knee-ankle chain meant a person
        # whose feet were out of frame — which is most people, seen from a
        # robot — had no posture at all.
        self.has_legs = (self.knee_deg is not None or self.hip_deg is not None)

        knee_mid = _mid_of(knee_pts)
        ankle_mid = _mid_of(ankle_pts)
        self.hip_knee_dy = (
            None if (self.hip is None or knee_mid is None)
            else float(knee_mid[1] - self.hip[1]) / self.height)
        self.hip_ankle_dy = (
            None if (self.hip is None or ankle_mid is None)
            else float(ankle_mid[1] - self.hip[1]) / self.height)

        self.body_down = _body_down(self)
        self.body_scale = _body_scale(self)
        self.extent = _head_to_foot_extent(self)

    def joint(self, name: str) -> Optional[np.ndarray]:
        return _point(self.keypoints, name, self.min_conf)

    def visible_count(self) -> int:
        return int(np.count_nonzero(self.keypoints[:, 2] >= self.min_conf))


# ── window features ─────────────────────────────────────────────────────────

def _confidence(value: float, threshold: float, *, below: bool = True) -> float:
    """Heuristic 0.5-1.0 score for "how far past its threshold is this".

    These are NOT probabilities and must not be read as any. They exist so two
    labels that both hold can be ordered, and so `min_confidence` has something
    to compare against — a rule that only just scrapes past its threshold gets
    ~0.5, one that clears it by a wide margin approaches 1.0.
    """
    if threshold <= 0:
        return 0.5
    margin = (threshold - value) / threshold if below else (value - threshold) / threshold
    return float(0.5 + 0.5 * max(0.0, min(1.0, margin)))


def _joint_speed(frames: list, name: str) -> Optional[float]:
    """Mean speed of one joint across the window, in body-heights per second."""
    samples = [(f.t, f.joint(name), f.height) for f in frames]
    samples = [(t, p, h) for t, p, h in samples if p is not None]
    if len(samples) < 2:
        return None
    total, seconds = 0.0, 0.0
    for (t0, p0, h0), (t1, p1, _h1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        total += float(np.linalg.norm(p1 - p0)) / max(h0, 1.0)
        seconds += dt
    if seconds <= 0:
        return None
    return total / seconds


def _reversals(signal: list) -> int:
    """Direction changes in a 1-D signal.

    This is what tells waving from reaching and walking from standing still:
    both pairs differ by whether the motion comes back, not by how fast it is.
    """
    directions = [b - a for a, b in zip(signal, signal[1:])]
    directions = [d for d in directions if abs(d) > 1e-9]
    return sum(1 for a, b in zip(directions, directions[1:]) if a * b < 0)


def _body_speed(frames: list) -> Optional[float]:
    """Largest per-joint speed over the window, body-heights per second."""
    speeds = [s for s in (_joint_speed(frames, name) for name in COCO_INDEX)
              if s is not None]
    return max(speeds) if speeds else None


# ── posture rules (single frame) ─────────────────────────────────────────────

def _posture_labels(frame: PoseFrame, th: dict) -> list:
    """Postures that hold for this frame, as (label, confidence) pairs.

    **Angles only. No image-space length ratios.** That rule is the result of a
    measured failure, not a preference. The first version gated `standing` on
    `hip_knee_dy >= 0.15` — the vertical gap between hips and knees as a
    fraction of bounding-box height — on top of the torso-tilt and knee-angle
    tests. Simulating a person who is definitely standing, with the legs
    foreshortened as a low camera foreshortens them:

        leg compression  hip_knee_dy  knee_deg  torso_deg   verdict
             1.00            0.232      180.0       0.0     standing
             0.60            0.171      180.0       0.0     standing
             0.45            0.140      180.0       0.0     UNKNOWN
             0.25            0.089      180.0       0.0     UNKNOWN

    `knee_deg=180` and `torso_deg=0` stay exactly right the whole way down —
    upright, legs straight, no ambiguity. The length ratio added no information
    and contributed only a viewpoint dependence that killed the rule. A robot
    camera at 0.4-1.2 m looking up at someone 1-2 m away is well past 0.45.

    Worse, `sitting` keyed on the *other side of the same fragile quantity*
    (`|hip_knee_dy| <= 0.15`), so a standing person seen from a low camera
    either vanished into `unknown` or landed in the sitting band. "Standing and
    sitting are both wrong" was the predictable consequence.

    So: sitting is now a *hip* angle (shoulder-hip-knee, the thigh folded up
    towards the chest) plus a knee angle. Both are angles, both survive
    perspective, and neither needs the feet to be in frame.

    Visibility is graded rather than all-or-nothing: a torso alone supports
    `lying` and `bending`, and legs are "readable" as soon as *either* the hip
    or the knee angle is available. Requiring the whole hip-knee-ankle chain
    meant a person whose feet were out of frame had no posture at all.
    """
    out = []

    # `lying` first, and deliberately NOT gated on the hips being visible.
    #
    # It used to reach the body axis only through `torso_deg`, which needs
    # shoulders *and* hips. A fallen person seen from a robot's low camera
    # usually shows head and shoulders and little below — so on exactly the
    # frames where "this person is on the ground" is the single most important
    # thing to report, the posture rules returned nothing at all and the arm
    # rules were left to name the frame. That is how a person lying on the floor
    # came back as `pointing`.
    #
    # `body_down` falls back to the head-to-shoulder vector, so the axis
    # survives the occlusion that matters.
    axis_deg = _tilt_from_vertical_deg(frame.body_down)
    extent = frame.extent

    # Three independent signs of a body that is not upright, ORed. They were a
    # single conjunction (tilt >= 60 AND aspect >= 1.2) and that pair is only
    # valid for a camera looking at the body side-on, level with it. Two real
    # photographs of fallen people failed both halves: 37.4 deg / 0.95 aspect,
    # and 30.2 deg / 1.04.
    inverted = extent is not None and extent <= -th["upright_extent"] * 0.4
    flattened = extent is not None and abs(extent) < th["lying_extent"]
    side_on = axis_deg is not None and axis_deg >= th["lying_deg"]
    if inverted or flattened or side_on:
        score = max(
            _confidence(axis_deg, th["lying_deg"], below=False) if side_on else 0.0,
            _confidence(abs(extent), th["lying_extent"]) if flattened else 0.0,
            0.8 if inverted else 0.0,
        )
        # Aspect corroborates rather than gates: a wide box makes this more
        # certain, a tall one no longer vetoes it.
        if frame.aspect >= th["lying_aspect"]:
            score = min(1.0, score + 0.1)
        out.append(("lying", score))

    # Legs or hips unreadable — the common webcam framing is head, shoulders and
    # arms with nothing below the waist. The body axis is still perfectly
    # measurable (`body_down` falls back to the head-to-shoulder vector), so say
    # the part that is known. "I cannot tell standing from sitting" and "I know
    # nothing about this body" were being reported as the same `unknown`, and a
    # head-and-shoulders view can certainly tell either of them from lying down.
    if not out and axis_deg is not None and axis_deg <= th["upright_deg"]:
        out.append(("upright", _confidence(axis_deg, th["upright_deg"])))

    if frame.torso_deg is None:
        return out

    upright = frame.torso_deg <= th["upright_deg"]

    # Straight legs separate bending from crouching: both put the torso over
    # the floor, only one folds the knees. Needs a knee angle specifically.
    if (frame.knee_deg is not None
            and th["bend_deg"] <= frame.torso_deg < th["lying_deg"]
            and frame.knee_deg >= th["leg_straight_deg"]):
        out.append(("bending", _confidence(frame.torso_deg, th["bend_deg"],
                                           below=False)))

    if not frame.has_legs:
        return out

    knee = frame.knee_deg
    hip = frame.hip_deg

    # Sitting: thigh folded up towards the chest. The hip angle is the primary
    # test because it is readable without the feet; the knee angle corroborates
    # when it is there, and is required to not be straight.
    if upright and hip is not None and th["hip_folded_min_deg"] <= hip <= th["hip_folded_max_deg"]:
        # The knee must be bent, but not folded shut: past `knee_bent_min_deg`
        # it is a squat, and a squat reporting `sitting` alongside `crouching`
        # is noise — priority would hide it, but `actions` would still carry it.
        knee_ok = knee is None or (th["knee_bent_min_deg"] <= knee
                                   <= th["knee_bent_max_deg"])
        if knee_ok:
            out.append(("sitting", _confidence(abs(hip - 90.0),
                                               th["hip_folded_max_deg"] - 90.0)))

    # Crouching: knees folded shut. Distinguished from sitting by how far —
    # a squat closes the knee past where a chair does.
    if knee is not None and knee < th["knee_bent_min_deg"]:
        out.append(("crouching", _confidence(knee, th["knee_bent_min_deg"])))

    # Standing: upright torso, and the legs not folded. Stated as the absence
    # of folding rather than the presence of a vertical gap, which is the whole
    # point of this rewrite.
    # A body whose feet are above its head is not standing, whatever the torso
    # angle says. The trampoline photograph read 30.2 deg of torso tilt — which
    # is "upright" — because `_tilt_from_vertical_deg` takes the absolute value
    # of the vertical component, so an inverted body and an upright one are the
    # same number to it. That choice is right for `bending` and wrong here.
    if upright and not inverted:
        # The hip angle carries this, and the knee may only veto a *clearly*
        # folded leg. Making a straight knee a requirement repeated the mistake
        # this rewrite was supposed to remove, one level down.
        #
        # 2D joint angles are invariant to scale and rotation but NOT to the
        # anisotropic scaling that foreshortening is. Measured on a real
        # skeleton from the engine, legs compressed to 0.35 of their vertical
        # extent: the hip angle moved 170.3 -> 162.7 deg while the knee angle
        # moved 151.1 -> 126.0, because the thigh and shin are never exactly
        # collinear and squashing y amplifies whatever lateral offset they have.
        # The knee spans two short noisy segments; the hip spans the torso,
        # which is the longest and best-detected part of the body. So the robust
        # measurement decides and the fragile one only objects to a squat.
        hip_open = hip is not None and hip >= th["hip_open_min_deg"]
        knee_unfolded = knee is not None and knee >= th["leg_straight_deg"]
        not_squatting = knee is None or knee >= th["knee_bent_min_deg"]
        if (hip_open or knee_unfolded) and not_squatting:
            out.append(("standing", _confidence(frame.torso_deg, th["upright_deg"])))

    return out


# ── arm rules (frame + short window) ────────────────────────────────────────

def _raised_sides(frame: PoseFrame, th: dict) -> list:
    """Sides whose wrist is above the shoulder, **in the body's own frame**.

    "Above" has to mean "towards the head", not "smaller y". A person lying on
    the ground has a body whose head direction points sideways on screen, so an
    arm resting on the floor beside them is, in image terms, as far "above"
    their shoulder as a raised arm is for someone standing. Measuring against
    `body_down` instead of the image axis is what makes the test mean what its
    name says.

    The scale is the body's, never the bounding box — see `_body_scale`.
    """
    sides = []
    if frame.body_down is None or frame.body_scale is None:
        return sides
    for side in ("left", "right"):
        shoulder = frame.joint(f"{side}_shoulder")
        elbow = frame.joint(f"{side}_elbow")
        wrist = frame.joint(f"{side}_wrist")
        if shoulder is None or wrist is None:
            continue
        # Positive when the wrist lies towards the head along the body axis.
        rise = float(np.dot(shoulder - wrist, frame.body_down)) / frame.body_scale
        if rise < th["raise_wrist_above_shoulder"]:
            continue
        elbow_deg = _angle_deg(shoulder, elbow, wrist)
        # No visible elbow: the wrist being above the shoulder is still the
        # thing being claimed, so this is accepted rather than dropped.
        if elbow_deg is not None and elbow_deg < th["elbow_open_deg"]:
            continue
        sides.append((side, rise))
    return sides


def _arm_labels(frames: list, th: dict) -> list:
    """Arm/interaction labels, as (label, confidence, extra) triples."""
    current = frames[-1]
    out = []
    raised_now = _raised_sides(current, th)

    for side, rise in raised_now:
        # Held, not glimpsed: an arm swinging through shoulder height on one
        # frame is not someone raising their hand.
        held_from = None
        for frame in reversed(frames):
            if side not in [s for s, _ in _raised_sides(frame, th)]:
                break
            held_from = frame.t
        held_s = 0.0 if held_from is None else current.t - held_from
        if held_s < th["raise_hold_s"]:
            continue
        out.append(("raising_hand",
                    _confidence(rise, th["raise_wrist_above_shoulder"], below=False),
                    {"side": side, "held_s": round(held_s, 2)}))

        # Waving is raising_hand plus "it comes back". Gating it on the raise
        # is what keeps an arm swinging while walking from reading as a wave.
        xs, ts = [], []
        for frame in frames:
            wrist = frame.joint(f"{side}_wrist")
            shoulder = frame.joint(f"{side}_shoulder")
            if (wrist is None or shoulder is None or frame.body_down is None
                    or frame.body_scale is None):
                continue
            # Travel *across* the body axis, in body scales. The image's x axis
            # is the wrong one for the same reason it is wrong for `raised`: a
            # person waving while lying down waves across their own body, which
            # is vertical on screen.
            across = np.array([-frame.body_down[1], frame.body_down[0]],
                              dtype=np.float32)
            xs.append(float(np.dot(wrist - shoulder, across)) / frame.body_scale)
            ts.append(frame.t)
        if len(xs) < 3:
            continue
        reversals = _reversals(xs)
        amplitude = max(xs) - min(xs)
        duration = ts[-1] - ts[0]
        cycles = reversals / 2.0
        frequency = cycles / duration if duration > 0 else 0.0
        if (reversals >= th["wave_reversals"]
                and amplitude >= th["wave_amplitude"]
                and th["wave_freq_min_hz"] <= frequency <= th["wave_freq_max_hz"]):
            out.append(("waving",
                        _confidence(amplitude, th["wave_amplitude"], below=False),
                        {"side": side, "reversals": reversals,
                         "amplitude": round(amplitude, 3),
                         "frequency_hz": round(frequency, 2)}))

    for side in ("left", "right"):
        shoulder = current.joint(f"{side}_shoulder")
        elbow = current.joint(f"{side}_elbow")
        wrist = current.joint(f"{side}_wrist")
        if shoulder is None or wrist is None or elbow is None:
            continue
        if (_angle_deg(shoulder, elbow, wrist) or 0.0) < th["arm_straight_deg"]:
            continue
        reach = wrist - shoulder
        norm = float(np.linalg.norm(reach))
        if norm < 1e-6 or current.body_down is None:
            continue
        # "Horizontal" means *perpendicular to the body*, not parallel to the
        # image's horizon. Measured against the image, a person lying on the
        # ground has a horizontal arm by construction — their whole body is
        # horizontal — so every fallen person with a straight arm scored as
        # pointing. Against the body axis, an arm resting alongside the body is
        # parallel to it and scores nothing, while an arm held out to the side
        # scores whatever posture its owner is in.
        along = abs(float(np.dot(reach / norm, current.body_down)))
        from_perpendicular = math.degrees(math.asin(min(1.0, along)))
        if from_perpendicular > th["point_horizontal_deg"]:
            continue
        other = "right" if side == "left" else "left"
        other_speed = _joint_speed(frames, f"{other}_wrist")
        if other_speed is not None and other_speed > th["still_speed"]:
            continue
        out.append(("pointing",
                    _confidence(from_perpendicular, th["point_horizontal_deg"]),
                    # The direction is worth more than the label: "someone is
                    # pointing" is far less actionable than where.
                    {"side": side,
                     "point_direction": [round(float(reach[0] / norm), 3),
                                         round(float(reach[1] / norm), 3)]}))

    crossed = 0
    if (current.shoulder is not None and current.hip is not None
            and current.body_down is not None):
        # Also in the body frame: "across the midline" and "between shoulders
        # and hips" are body-relative statements, and reading them off the image
        # axes makes them mean something else for anyone not standing upright.
        across = np.array([-current.body_down[1], current.body_down[0]],
                          dtype=np.float32)
        centre = (current.shoulder + current.hip) / 2.0
        torso_span = float(np.dot(current.hip - current.shoulder, current.body_down))
        for side in ("left", "right"):
            wrist = current.joint(f"{side}_wrist")
            shoulder = current.joint(f"{side}_shoulder")
            if wrist is None or shoulder is None:
                continue
            own_side = float(np.dot(shoulder - centre, across))
            wrist_side = float(np.dot(wrist - centre, across))
            depth = float(np.dot(wrist - current.shoulder, current.body_down))
            if own_side * wrist_side < 0 and 0.0 <= depth <= torso_span:
                crossed += 1
    if crossed == 2:
        out.append(("arms_crossed", 0.7, {}))

    return out


# ── motion rules (window only) ──────────────────────────────────────────────

def _motion_labels(frames: list, th: dict, postures: list) -> list:
    out = []
    duration = frames[-1].t - frames[0].t
    if duration <= 0 or len(frames) < 3:
        return out

    # `still` needs the same structural basis a posture does. "They are not
    # moving" derived from a nose and one shoulder is a confident claim built
    # on nothing, and it lands on exactly the frames where the honest answer is
    # `unknown` — a person behind a desk would come back as motionless rather
    # than as unreadable, which is the failure this plugin is supposed to avoid.
    # walking and turning carry their own structural requirements (two ankles,
    # two shoulders), so only this one needs the gate.
    # No `still` label. With posture and activity as separate channels the
    # absence of an activity *is* stillness, and a label saying "not doing
    # anything" beside an empty activity field is noise. `still_speed` is still
    # used, by the rules that need a limb to be stationary.

    separations, widths = [], []
    for frame in frames:
        la, ra = frame.joint("left_ankle"), frame.joint("right_ankle")
        if la is not None and ra is not None:
            separations.append(float(la[0] - ra[0]) / frame.height)
        ls, rs = frame.joint("left_shoulder"), frame.joint("right_shoulder")
        if ls is not None and rs is not None:
            widths.append(abs(float(ls[0] - rs[0])) / frame.height)

    # Walking is gated on an upright torso, so a seated person fidgeting their
    # feet is not reported as walking.
    upright = any(label in ("standing", "bending") for label, _ in postures)
    if len(separations) >= 3 and upright:
        reversals = _reversals(separations)
        amplitude = max(separations) - min(separations)
        cadence = (reversals / 2.0) / duration
        if (amplitude >= th["walk_amplitude"]
                and th["walk_cadence_min_hz"] <= cadence <= th["walk_cadence_max_hz"]):
            out.append(("walking",
                        _confidence(amplitude, th["walk_amplitude"], below=False),
                        {"cadence_hz": round(cadence, 2),
                         "amplitude": round(amplitude, 3)}))

    if len(widths) >= 3:
        widest = max(widths)
        change = abs(widths[-1] - widths[0]) / widest if widest > 1e-6 else 0.0
        if change >= th["turn_width_change"]:
            out.append(("turning",
                        _confidence(change, th["turn_width_change"], below=False),
                        {"shoulder_width_change": round(change, 3)}))

    return out


# ── fall: the one label that judges a transition, not a state ───────────────

def _fall_evidence(frames: list, postures: list, th: dict) -> Optional[dict]:
    """Decide whether the person now on the floor *fell* onto it.

    Lying on the floor and lying on a sofa are the same terminal state, so a
    posture test cannot tell them apart and neither can a single frame. What
    distinguishes a fall is the transition into it:

      1. the hips drop more than `fall_drop_ratio` of standing height
      2. in under `fall_drop_window_s`
      3. and the body stays horizontal for `fall_settle_s` afterwards
      4. with no `sitting` phase on the way down — sitting down is slow and
         has an unambiguous knee angle

    Returns the evidence whenever the person is settled and horizontal, with
    `is_fall` saying whether the drop was there too. The rejected case is
    returned rather than swallowed on purpose: "they are lying down and here is
    why we did not call it a fall" is what someone reads when asking why the
    robot said nothing.

    Height comes from the tallest box in the window, not the current one: a
    person on the floor has a short wide box, so normalising by the current
    height would divide the drop by the post-fall height and understate it.
    """
    if not frames or "lying" not in [label for label, _ in postures[-1]]:
        return None

    settle_start = len(frames) - 1
    while settle_start > 0 and "lying" in [label for label, _ in postures[settle_start - 1]]:
        settle_start -= 1
    settle_s = frames[-1].t - frames[settle_start].t
    if settle_s < th["fall_settle_s"]:
        # Horizontal, but not for long enough to tell a fall from bending to
        # pick something up. Deliberately not a fall *yet* rather than a
        # negative — the next frames decide.
        return None

    reference_height = max(frame.height for frame in frames)
    landed = frames[settle_start]
    if landed.hip is None:
        return {"is_fall": False, "reason": "hips not visible at landing",
                "settle_ms": int(settle_s * 1000)}

    best = None
    for index in range(settle_start, -1, -1):
        gap = landed.t - frames[index].t
        if gap > th["fall_drop_window_s"]:
            break
        start = frames[index]
        if start.hip is None:
            continue
        drop = float(landed.hip[1] - start.hip[1]) / reference_height
        if best is None or drop > best[0]:
            best = (drop, gap, index)

    if best is None:
        return {"is_fall": False, "reason": "no hip reading before landing",
                "settle_ms": int(settle_s * 1000)}

    drop_ratio, drop_s, from_index = best
    had_sitting = any(
        "sitting" in [label for label, _ in postures[i]]
        for i in range(from_index, settle_start + 1)
    )
    evidence = {
        "drop_ratio": round(drop_ratio, 3),
        "drop_ms": int(drop_s * 1000),
        "settle_ms": int(settle_s * 1000),
        "had_sitting_phase": had_sitting,
    }
    if drop_ratio < th["fall_drop_ratio"]:
        evidence["is_fall"] = False
        evidence["reason"] = "no fast drop into the horizontal pose"
        return evidence
    if had_sitting:
        evidence["is_fall"] = False
        evidence["reason"] = "passed through sitting on the way down"
        return evidence
    evidence["is_fall"] = True
    evidence["confidence"] = _confidence(drop_ratio, th["fall_drop_ratio"],
                                         below=False)
    return evidence


# ── classifier ──────────────────────────────────────────────────────────────

class PoseActionClassifier:
    """Rules backend: keypoint sequence in, action labels out.

    `action_backend` on the plugin selects this; a learned skeleton-action model
    would replace this class and nothing else, which is why `classify` takes
    plain PoseFrames and returns a plain dict.
    """

    def __init__(self, thresholds: Optional[dict] = None,
                 action_window_s: float = 1.5):
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.thresholds.update({k: v for k, v in (thresholds or {}).items()
                                if k in DEFAULT_THRESHOLDS and v is not None})
        self.action_window_s = float(action_window_s)

    @property
    def last_error(self) -> Optional[str]:
        """Nothing to report: the geometry has no engine that can fail.

        Present so the three backends answer the same questions — the card's
        `info` asks every backend this without knowing which one it holds.
        """
        return None

    @property
    def history_s(self) -> float:
        """How much history the tracker has to keep for these rules to work.

        Fall needs the drop window *plus* the settle window, which is longer
        than the action window — ask for too little and fall can never fire,
        with nothing in any log to say why.
        """
        th = self.thresholds
        return max(self.action_window_s,
                   th["fall_drop_window_s"] + th["fall_settle_s"] + 0.5)

    def classify_frame(self, frame) -> dict:
        """Classify a single image — what one frame can honestly support.

        The hold requirement on a raised hand is dropped here, because it only
        exists to tell a held gesture from an arm passing through shoulder
        height, and a photo has no "passing through". Everything in
        TEMPORAL_ACTIVITIES stays unreachable and is named in the reply, so a
        caller who asked "is she waving" is told the question needs a stream
        rather than being handed `raising hand` as if it answered.
        """
        single = PoseActionClassifier(
            thresholds={**self.thresholds, "raise_hold_s": 0.0},
            action_window_s=self.action_window_s,
        )
        result = single.classify([frame])
        result["temporal"] = False
        result["unavailable_activities"] = list(TEMPORAL_ACTIVITIES)
        return result

    def classify(self, frames: list, want_activity: bool = True) -> dict:
        """Two channels: what shape the body is in, and what it is doing.

        `want_activity` exists so all three backends share one signature — the
        caller throttles the learned one and must not have to know which it
        holds. The geometry's own activities cost 0.68 ms, so it computes them
        either way and the flag changes nothing here.

        Neither is derived from the other and neither is a fallback for the
        other. A person always has a posture; they may well not have an
        activity, and saying so is the honest answer rather than picking the
        least-wrong verb.
        """
        th = self.thresholds
        if not frames:
            return _nothing("no frames")

        current = frames[-1]
        window = [f for f in frames
                  if current.t - f.t <= self.action_window_s] or [current]
        postures = [_posture_labels(f, th) for f in frames]

        scored: dict = {}
        extras: dict = {}

        def _offer(label, confidence, extra=None):
            if label not in scored or confidence > scored[label]:
                scored[label] = float(confidence)
                extras[label] = dict(extra or {})

        for label, confidence in postures[-1]:
            _offer(label, confidence)
        for label, confidence, extra in _arm_labels(window, th):
            _offer(label, confidence, extra)
        for label, confidence, extra in _motion_labels(window, th, postures[-1]):
            _offer(label, confidence, extra)

        fall = _fall_evidence(frames, postures, th)
        if fall is not None:
            if fall.get("is_fall"):
                _offer("fall", fall.get("confidence", 0.5), fall)
            else:
                extras.setdefault("lying", {}).update(fall)

        held = {label: conf for label, conf in scored.items()
                if conf >= th["min_confidence"] or label in EVENT_ACTIONS}

        result: dict = {"posture": None, "posture_confidence": 0.0,
                        "activity": None}

        posture = next((p for p in POSTURE_PRIORITY if p in held), None)
        if posture is not None:
            result["posture"] = posture
            result["posture_confidence"] = round(held[posture], 2)
            result["evidence"] = extras.get(posture, {})
        else:
            result["evidence"] = {
                "reason": ("occluded" if not current.has_torso
                           else "no posture rule matched with enough confidence"),
                "visible_keypoints": current.visible_count(),
                "has_torso": current.has_torso,
                "has_legs": current.has_legs,
            }

        # The geometry's own activities, named the way the model names them so
        # the two sources cannot report one thing under two spellings.
        activities = [(RULE_TO_ACTIVITY[label], held[label], extras.get(label, {}))
                      for label in held if label in RULE_TO_ACTIVITY]
        if activities:
            activities.sort(key=lambda item: ACTIVITY_PRIORITY.index(item[0])
                            if item[0] in ACTIVITY_PRIORITY else 99)
            name, confidence, extra = activities[0]
            result["activity"] = {
                "name": name, "name_zh": ACTIVITY_LABELS_ZH.get(name, name),
                "score": round(confidence, 2), "source": "rules",
            }
            if extra:
                result["evidence"] = {**result["evidence"], **extra}
            direction = extras.get("pointing", {}).get("point_direction")
            if direction is not None:
                result["point_direction"] = direction
        return result

    def _nothing_compat(self):          # pragma: no cover - kept for clarity
        return _nothing("no frames")


def _nothing(reason: str) -> dict:
    return {"posture": None, "posture_confidence": 0.0, "activity": None,
            "evidence": {"reason": reason}}


# ── tracking ────────────────────────────────────────────────────────────────

def _visible_centroid(keypoints: np.ndarray, min_conf: float):
    """Mean of the visible joints, or None.

    Continuous through a fall in a way the bounding box is not: the box flips
    from tall-and-narrow to short-and-wide, but the body's visible joints stay
    in roughly the same place from one frame to the next.
    """
    visible = keypoints[keypoints[:, 2] >= min_conf]
    if len(visible) == 0:
        return None
    return visible[:, :2].mean(axis=0)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PoseTrack:
    """One person's timeline. `history` is what the classifier reads."""

    __slots__ = ("id", "history", "last_seen",
                 "posture_stabiliser", "activity_stabiliser",
                 "activity", "activity_t")

    def __init__(self, track_id: int, label_hold: int = 3):
        self.id = track_id
        self.history: deque = deque()
        self.last_seen = 0.0
        # Per person, because two people in frame change independently and a
        # shared stabiliser would let one person's gesture suppress the other's.
        # Per channel, because a posture settling must not hold back an
        # activity, or vice versa — they change on different timescales.
        self.posture_stabiliser = LabelStabiliser(label_hold)
        self.activity_stabiliser = LabelStabiliser(label_hold)
        # Last activity the learned backend produced, and when. The model is not
        # re-run every frame — its window is seconds long, so consecutive runs
        # share almost all their input. See the throttle in plugins/pose.py.
        self.activity = None
        self.activity_t = -1e9

    @property
    def current(self) -> Optional[PoseFrame]:
        return self.history[-1] if self.history else None


class PoseTracker:
    """Greedy IoU association, so actions have a timeline to be computed over.

    This is NOT re-identification and does not pretend to be: a person who
    leaves the frame and comes back gets a new id. Recognising *who* someone is
    is the `face_recognition` card's job, and conflating the two would promise
    an identity this cannot keep.
    """

    def __init__(self, history_s: float = 3.0, iou_min: float = 0.2,
                 timeout_s: float = 1.0, min_conf: float = 0.3,
                 centroid_max_travel: float = 1.0,
                 label_hold: int = 3):
        self.history_s = float(history_s)
        self.iou_min = float(iou_min)
        self.timeout_s = float(timeout_s)
        self.min_conf = float(min_conf)
        # How far the visible-joint centroid may move between frames, as a
        # fraction of body height, and still count as the same person.
        #
        # This is a sanity bound, NOT the discriminator — greedy nearest-match
        # does the actual work, since each detection and each track is used
        # once and the closest pair is taken first. The bound only has to be
        # loose enough not to veto a real transition, and a fall moves the
        # centroid by about half a body height *by definition*: a standing
        # person's centroid sits ~0.55h above the floor and a fallen one's
        # ~0.05h. Measured on the synthetic fall it is 0.53h in a single frame
        # (the worst case — one frame at 5 fps covering the whole descent; at
        # 15 fps each step is a third of that). A gate of 0.5 vetoed it by
        # 0.03, which is how a threshold set to the magnitude of the thing it
        # must admit behaves.
        #
        # Erring loose is the right direction: merging two people who swapped
        # places costs one wrong action label, while splitting a track makes
        # every temporal label undetectable. This tracker is explicitly not
        # re-identification, so it has no identity to protect.
        self.centroid_max_travel = float(centroid_max_travel)
        self.label_hold = max(1, int(label_hold))
        self._tracks: list = []
        self._next_id = 1

    @property
    def tracks(self) -> list:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks = []

    def _affinity(self, box, keypoints, track) -> Optional[float]:
        """How much this detection looks like the continuation of `track`.

        IoU alone is not enough, and the case it fails is the one that matters
        most. Measured on a synthetic fall: a standing box [260,100,380,500]
        against the same person's lying box [140,436,500,492] scores
        **IoU 0.109**, under any usable iou_min — so the track split at the
        instant of the fall, the new track started with an empty history, and
        `_fall_evidence` could only ever see the horizontal frames and report
        "no fast drop". Fall detection required continuity across precisely the
        event that destroys box overlap.

        So a second, independent cue: the centroid of the visible joints,
        compared against the body's own scale. The box changes shape when
        someone falls; the body does not teleport. Either cue passing is enough,
        because they fail in different situations — IoU covers a person standing
        still whose keypoints are noisy, the centroid covers a person whose box
        geometry changes abruptly.

        A learned skeleton-action backend needs this fix just as much as the
        rules do: it is fed one continuous (T, V, C) sequence per person, so a
        track that splits mid-action hides the action from it too.
        """
        current = track.current
        if current is None:
            return None
        overlap = _iou(box, current.box)
        if overlap >= self.iou_min:
            return 1.0 + overlap          # ranked above any centroid-only match

        here = _visible_centroid(keypoints, self.min_conf)
        there = _visible_centroid(current.keypoints, self.min_conf)
        if here is None or there is None:
            return None
        # Scale is the larger of the two bodies' heights, so the gate means "it
        # did not move more than `centroid_max_travel` of a body height" and
        # carries no pixel constant.
        scale = max(body_height(box), current.height)
        travel = float(np.linalg.norm(here - there)) / max(scale, 1.0)
        if travel > self.centroid_max_travel:
            return None
        return 1.0 - travel / self.centroid_max_travel

    def update(self, boxes, keypoints, now: float, image_size=None) -> list:
        """Associate this frame's detections, returning one track per detection
        in the order the detections came in (so the caller can zip them)."""
        self._expire(now)

        boxes = [list(map(float, box)) for box in boxes]
        arrays = [np.asarray(k, dtype=np.float32) for k in keypoints]
        pairs = []
        for d_index, box in enumerate(boxes):
            for t_index, track in enumerate(self._tracks):
                score = self._affinity(box, arrays[d_index], track)
                if score is not None:
                    pairs.append((score, d_index, t_index))
        pairs.sort(reverse=True)

        taken_d: set = set()
        taken_t: set = set()
        assigned: dict = {}
        for _score, d_index, t_index in pairs:
            if d_index in taken_d or t_index in taken_t:
                continue
            taken_d.add(d_index)
            taken_t.add(t_index)
            assigned[d_index] = self._tracks[t_index]

        out = []
        for d_index, box in enumerate(boxes):
            track = assigned.get(d_index)
            if track is None:
                track = PoseTrack(self._next_id, self.label_hold)
                self._next_id += 1
                self._tracks.append(track)
            frame = PoseFrame(now, box, np.asarray(keypoints[d_index],
                                                   dtype=np.float32),
                              self.min_conf, image_size=image_size)
            track.history.append(frame)
            track.last_seen = now
            self._trim(track, now)
            out.append(track)
        return out

    def _trim(self, track, now: float) -> None:
        while track.history and now - track.history[0].t > self.history_s:
            track.history.popleft()

    def _expire(self, now: float) -> None:
        self._tracks = [t for t in self._tracks
                        if now - t.last_seen <= self.timeout_s]


def action_catalogue() -> list:
    """What `list_actions` answers with."""
    return [
        {
            "action": name,
            "label_zh": ACTION_LABELS_ZH[name],
            "kind": "event" if name in EVENT_ACTIONS else "pose",
        }
        for name in ACTION_PRIORITY
    ]


class LabelStabiliser:
    """Hysteresis on the primary label, per tracked person.

    Every threshold in this file and in the learned backend is a cliff: a score
    hovering at `min_score`, a clip whose motion sits at `MIN_MOTION`, a torso
    at exactly `upright_deg`. Measured at those boundaries the card flipped its
    answer on **every single frame** — 9 changes in 9 comparisons with the
    model's score oscillating around 0.40, and 6 in 6 with the motion oscillating
    around the gate.

    A label that alternates twelve times a second is worse than a label that is
    simply wrong: nothing downstream can act on it, an operator cannot read it,
    and an agent asked to react gets a different world each turn. So a challenger
    has to win `hold` consecutive classifications before it takes over, and the
    previous answer stands until it does.

    The cost is latency — `hold` frames, 250 ms at 12 fps — paid on every real
    change too. That is the trade being made deliberately, and `pending` is
    reported so the raw instantaneous answer stays visible.
    """

    __slots__ = ("hold", "current", "_candidate", "_count")

    def __init__(self, hold: int = 3):
        self.hold = max(1, int(hold))
        self.current: Optional[str] = None
        self._candidate: Optional[str] = None
        self._count = 0

    def update(self, label: str) -> tuple:
        """Feed this frame's label; returns (stable label, pending or None)."""
        if self.current is None:
            self.current = label
            self._candidate, self._count = None, 0
            return self.current, None
        if label == self.current:
            self._candidate, self._count = None, 0
            return self.current, None
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = label, 1
        if self._count >= self.hold:
            self.current = label
            self._candidate, self._count = None, 0
            return self.current, None
        return self.current, self._candidate
