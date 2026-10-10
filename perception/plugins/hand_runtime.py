#!/usr/bin/env python3
"""
plugins/hand_runtime.py — hand ROI geometry and the 21-keypoint decode.

The pose card already reports COCO-17 per person. This module takes those body
keypoints and gets 21 keypoints per hand out of a second engine, so the card can
say what a hand is *shaped* like rather than only where the wrist is.

Everything here is geometry and bookkeeping. The engine itself runs through
`plugins.vision_runtime.VisionEngineSession` and decodes through
`vision_runtime.decode_poses(..., n_kpts=21)` — which needs no change, because
the hand export is the same YOLO26 NMS-free pose head with a different keypoint
count, and `_pose_layout` already keys off the row width.

**Why there is a crop at all, rather than running the engine on the frame.**
Measured on real photographs (docs/hand-gesture.md §3): the model wants the hand
to occupy 120–320 px of its input, best around 31% of the input edge, and it
degrades at both ends — a hand filling 424 px of a 640 input scored 0.069. On a
1920x1080 frame letterboxed whole into the network, a hand is that big only
within about 0.7 m of the camera, which is useless for a robot. Cropping the
hand's neighbourhood out of the *native* frame and scaling that up instead held
the keypoint error to 0.059 at an estimated 3 m, against 0.154 for the whole
frame. So the crop is the normal path, not a far-field special case.

**Why the distance gate is geometric and not the model's confidence.** The
failure mode at long range is not a missing detection, it is a confident wrong
one: at a 100 px hand the model returned confidence 0.92 with a shape error of
0.169 — an average joint off by 17% of the hand's width, about a finger segment,
which makes any extended/curled decision noise. Confidence stayed at 0.9 while
the error tripled, so it cannot be used to decide whether the hand is close
enough. Hand width is roughly 0.45 of forearm length and the forearm is right
there in COCO-17, so the gate is forearm pixels, computed before any inference
and costing nothing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from plugins.vision_runtime import decode_poses

#: The 21 joints, in the order the weights emit them. Read out of the engine's
#: own metadata (`kpt_names`) rather than copied from a paper — same rule as
#: COCO_KEYPOINTS in vision_runtime: the order is part of the weights, and
#: reindexing it silently swaps joints while still drawing a plausible hand.
HAND_KEYPOINTS = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

N_HAND_KEYPOINTS = len(HAND_KEYPOINTS)

HAND_INDEX = {name: i for i, name in enumerate(HAND_KEYPOINTS)}

#: The bones, as index pairs into the 21. Palm edges included, so a hand with
#: curled fingers still reads as a hand rather than as five loose stalks.
HAND_SKELETON = (
    (0, 1), (1, 2), (2, 3), (3, 4),              # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),              # index
    (9, 10), (10, 11), (11, 12),                 # middle
    (13, 14), (14, 15), (15, 16),                # ring
    (0, 17), (17, 18), (18, 19), (19, 20),       # pinky
    (5, 9), (9, 13), (13, 17),                   # knuckles
)

#: Fraction of the network input edge the hand should land on. 0.3125 is
#: 200/640, the measured optimum; the same fraction at 448 is 140 px. Expressed
#: as a fraction rather than as pixels so that changing the engine's input size
#: does not silently move the hand out of the band it was chosen for.
TARGET_HAND_FRACTION = 200.0 / 640.0

#: Hand size as a multiple of forearm length, for turning a COCO-17 forearm
#: into the crop scale.
#:
#: **0.9, not 0.45.** This shipped at 0.45 and that was a reasoning error, not
#: a tuning choice: 0.45 is hand *width*, while what the crop has to contain is
#: what the model's box covers — the hand with its fingers extended, which is
#: hand *length* and runs 0.75–1.0 of the forearm.
#:
#: It was wrong in the direction that hides itself. The crop came out about
#: half the size it should be, so the hand filled 70% of it instead of 31%, and
#: 70% is the oversized end of the measured band where detection collapses.
#: Measured on Orin 5 against a real photograph of a person making a V sign
#: with both hands, sweeping this constant and reading the engine's score:
#:
#:     ratio  left          right         hand as % of crop
#:     0.45   miss          0.71          70%
#:     0.65   0.69          0.75          49–60%
#:     0.90   0.76          0.78          39%
#:     1.20   0.77          0.72          32–35%
#:
#: One hand missed outright and the other scored while sitting right at the
#: cliff — which looks like "the model is unreliable" rather than "the crop is
#: half the size it should be". n is one subject and two hands, so 0.9 is a
#: correction of a clear error rather than a tuned optimum.
HAND_PER_FOREARM = 0.9

#: Crop scales to try, as multiples of the nominal one, until a hand is found.
#:
#: **No single ratio works, and that is not a tuning failure.** Both the
#: forearm and the hand are projections, and they foreshorten independently: an
#: arm raised in the image plane shows its full forearm beside a hand seen
#: end-on, while an arm angled towards the camera shows a short forearm beside
#: a full hand. Measured on two real photographs, sweeping HAND_PER_FOREARM and
#: reading the engine's score for the hand that was hardest to find:
#:
#:     photo                     0.45   0.55   0.65   0.90   1.20
#:     child, V sign, both hands miss   0.30   0.69   0.76   0.77
#:     adult, one hand raised    0.65   miss   miss   miss   miss
#:
#: Opposite answers from the same constant. So the nominal crop is a starting
#: guess and a miss is retried at a tighter one, which costs an extra inference
#: only when the first attempt found nothing — and nothing at all once a hand
#: has been found, because the previous frame's box then sizes the crop from a
#: measurement instead of a guess.
HAND_RETRY_SCALES = (1.0, 0.55)

#: How long a hand keeps being reported after it stops being found.
#:
#: Without this a single miss erases the hand until the next inference, and at
#: the throttled rate that is a quarter of a second of no fingers — which on a
#: live camera reads as the hand flickering in and out. The throttle itself is
#: not the cause and was measured not to be: with a steady image, 129 of 129
#: published frames carried both hands, because the frames between inferences
#: reuse the last result. What the rate does is set how *long* each miss
#: lasts.
#:
#: Two things were conflated and only one of them should expire instantly.
#: The previous frame's **box** must not survive a miss — cropping from it
#: would chase a hand that is no longer there, so the next attempt re-derives
#: the ROI from the arm. The previous frame's **keypoints** may survive a
#: little longer, for the same reason the gesture tracker has a release
#: window: one dropped frame is not the hand going away.
DEFAULT_HAND_HOLD_S = 0.5

#: Score threshold for the hand detector, separate from the person detector's.
#:
#: These were one number, and that was simply the wrong wiring: a hand and a
#: person are different detectors with different score distributions, and
#: using the person threshold for both meant tightening person detection
#: silently dropped hands. Measured on a moving stream, with the hold
#: disabled so the threshold is the only variable:
#:
#:     threshold   hand at the side   hand raised
#:     0.40              35%              100%
#:     0.30              47%              100%
#:     0.25              52%              100%
#:     0.20              52%              100%
#:
#: A raised hand is found at any of them. What the threshold buys is the
#: *relaxed* hand hanging at the side — small, half-turned, against clothing —
#: and it saturates at 0.25, so there is nothing to gain by going lower and a
#: false-positive budget to lose.
DEFAULT_HAND_CONFIDENCE = 0.25

#: Minimum forearm length in native frame pixels before a hand is attempted.
#: The floor that matters is about 45 px of *hand*, which is where the measured
#: curve starts losing detections outright; at HAND_PER_FOREARM that is 50 px
#: of forearm. (This was 90 px while HAND_PER_FOREARM was half its correct
#: value — the two were consistent with each other and both wrong, so the gate
#: moves with the ratio rather than independently of it.) Exposed as config
#: rather than frozen here: the pixel-to-distance mapping depends on the
#: camera's field of view, which this process does not know.
DEFAULT_MIN_FOREARM_PX = 50.0

#: How far past the wrist, along the forearm, the hand's centre is assumed to
#: be — in forearm lengths. A hand hangs off the end of the arm, so centring
#: the crop on the wrist itself puts a third of the crop on the forearm and
#: clips the fingers.
HAND_CENTRE_OFFSET = 0.35

#: Why a hand was not attempted. Reported per instance so that "no hands" is
#: never ambiguous between these three and a broken engine.
SKIP_REASONS = ("wrist_occluded", "elbow_occluded", "too_far", "throttled")


class HandDecodeError(ValueError):
    """Raised when an engine cannot be read as an end-to-end 21-keypoint head."""


class HandRoi:
    """A square crop of the native frame, in native frame pixels."""

    __slots__ = ("cx", "cy", "side", "side_name", "source", "hand_px")

    def __init__(self, cx: float, cy: float, side: float, side_name: str,
                 source: str, hand_px: float):
        self.cx = float(cx)
        self.cy = float(cy)
        self.side = float(side)
        #: "left" or "right", meaning the person's own left or right. Taken
        #: from which wrist the ROI was derived from, never from a handedness
        #: output: the body keypoints already know, and a monocular hand model
        #: guessing it from appearance gets mirrored hands wrong.
        self.side_name = side_name
        #: "wrist" (extrapolated from the arm) or "previous" (last frame's
        #: decoded hand box, which is tighter once there is one).
        self.source = source
        #: The hand size this ROI was sized for, in native pixels. Kept so the
        #: caller can report *why* a hand was skipped or how marginal it was.
        self.hand_px = float(hand_px)

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return (f"HandRoi({self.side_name}, centre=({self.cx:.0f}, {self.cy:.0f}), "
                f"side={self.side:.0f}, hand~{self.hand_px:.0f}px, {self.source})")

    @property
    def box(self) -> tuple:
        half = self.side / 2.0
        return (self.cx - half, self.cy - half, self.cx + half, self.cy + half)


# ── the end2end guard ────────────────────────────────────────────────────────

def expected_row_width(n_kpts: int = N_HAND_KEYPOINTS) -> int:
    """Row width of an end-to-end head: box, score, class, then the triples."""
    return 6 + 3 * n_kpts


def raw_head_width(n_kpts: int = N_HAND_KEYPOINTS) -> int:
    """Channel count of the *raw* head, which is one narrower."""
    return 5 + 3 * n_kpts


def assert_end2end(session, n_kpts: int = N_HAND_KEYPOINTS) -> tuple:
    """Refuse an engine that is not an end-to-end export. Returns its shape.

    This check is not optional and it cannot be done from metadata.

    A raw (non-end2end) YOLO pose head emits `4 + 1 + 3*K` channels — 68 for
    21 keypoints — and `vision_runtime._pose_layout` accepts exactly that width
    as its *offset-5* layout. Every content check passes too: the class score
    and the keypoint visibilities have all been through a sigmoid, so they are
    in [0, 1] as the layout requires. The result is that several thousand
    un-suppressed anchors decode as several thousand hands, with xywh boxes read
    as xyxy, and nothing raises — `vision_runtime`'s own comment for the
    detection path says it: every wrong reading of these numbers still produces
    a plausible-looking result.

    Metadata cannot settle it either. `utils.tensorrt_runtime.read_engine_file`
    only finds an `end2end` flag when the ultralytics exporter wrote the JSON
    header, and an engine built with `trtexec` — which is how these are built,
    since nothing here goes through ultralytics' loader — has no header at all
    and reports `{}`.

    So the shape is the evidence, and it is taken from a real inference rather
    than from a declaration, at load time rather than per frame: a card that
    comes up `running` and then fails on every frame is much harder to diagnose
    than one that refuses to start.
    """
    width, height = session.input_size
    probe = np.zeros((height, width, 3), dtype=np.uint8)
    outputs, _ = session.infer(probe)
    shapes = []
    for array in (outputs if isinstance(outputs, (list, tuple)) else [outputs]):
        array = np.asarray(array)
        shape = tuple(int(v) for v in array.shape)
        shapes.append(shape)
        if expected_row_width(n_kpts) in shape:
            return shape
    raise HandDecodeError(
        f"hand engine outputs {shapes}, none of which has a "
        f"{expected_row_width(n_kpts)}-wide axis. A width of "
        f"{raw_head_width(n_kpts)} means the raw detection head was exported "
        f"instead of the end-to-end one — and that decodes *silently* as a "
        f"valid pose layout, so it must be rejected here. Re-export with "
        f"nms=True (ultralytics >= 8.4.17x needs it explicitly for YOLO26; "
        f"older versions produced end2end without it)."
    )


# ── ROI derivation ───────────────────────────────────────────────────────────

def _joint(keypoints: np.ndarray, index: int, min_conf: float):
    """One visible joint as (x, y), or None."""
    if index >= len(keypoints):
        return None
    x, y, visibility = keypoints[index][:3]
    if float(visibility) < min_conf:
        return None
    return np.array([float(x), float(y)], dtype=np.float32)


def forearm_length(keypoints: np.ndarray, side: str, min_conf: float,
                   coco_index: Optional[dict] = None) -> Optional[float]:
    """Elbow-to-wrist distance in frame pixels, or None if either is occluded.

    This is the scale everything else is derived from, so it returns None
    rather than a guess: a hand attempted at the wrong scale produces keypoints
    that look fine and are wrong, which is the failure this module is built to
    avoid.
    """
    if coco_index is None:
        from plugins.vision_runtime import COCO_INDEX as coco_index
    wrist = _joint(keypoints, coco_index[f"{side}_wrist"], min_conf)
    elbow = _joint(keypoints, coco_index[f"{side}_elbow"], min_conf)
    if wrist is None or elbow is None:
        return None
    length = float(np.linalg.norm(wrist - elbow))
    return length if length > 1e-3 else None


def roi_from_body(keypoints: np.ndarray, side: str, *,
                  min_conf: float = 0.3,
                  min_forearm_px: float = DEFAULT_MIN_FOREARM_PX,
                  target_fraction: float = TARGET_HAND_FRACTION,
                  coco_index: Optional[dict] = None) -> tuple:
    """Derive one hand's crop from the body keypoints.

    Returns `(HandRoi, None)` or `(None, reason)`, where reason is one of
    SKIP_REASONS. Two outcomes rather than an exception because "this hand is
    not attemptable right now" is the normal case, not an error, and the reason
    has to reach `info`.
    """
    if coco_index is None:
        from plugins.vision_runtime import COCO_INDEX as coco_index
    wrist = _joint(keypoints, coco_index[f"{side}_wrist"], min_conf)
    if wrist is None:
        return None, "wrist_occluded"
    elbow = _joint(keypoints, coco_index[f"{side}_elbow"], min_conf)
    if elbow is None:
        # The wrist alone gives a position but no scale, and guessing the scale
        # is exactly how a crop ends up with the hand at 500 px — a size the
        # measured curve says loses 3 of 5 detections.
        return None, "elbow_occluded"

    forearm = float(np.linalg.norm(wrist - elbow))
    if forearm < min_forearm_px:
        return None, "too_far"

    hand_px = forearm * HAND_PER_FOREARM
    # A hand hangs off the end of the forearm, so the crop follows that
    # direction rather than centring on the wrist joint itself.
    direction = wrist - elbow
    norm = float(np.linalg.norm(direction))
    centre = wrist + (direction / norm) * (forearm * HAND_CENTRE_OFFSET) if norm > 1e-6 else wrist
    side_px = hand_px / max(target_fraction, 1e-6)
    return HandRoi(centre[0], centre[1], side_px, side, "wrist", hand_px), None


def roi_from_previous(box, side: str, *,
                      target_fraction: float = TARGET_HAND_FRACTION) -> Optional[HandRoi]:
    """Derive the next crop from the hand box this engine just produced.

    Tighter than the arm extrapolation once there is one, for the reason
    MediaPipe's pipeline uses the same trick: the previous frame's hand box is
    an actual measurement of where the hand is, while the forearm direction is
    an assumption about where it should be. Falls back to the body whenever the
    hand is lost, which is what keeps a dropped frame from stranding the crop.
    """
    if box is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    hand_px = max(x2 - x1, y2 - y1)
    if hand_px <= 1e-3:
        return None
    return HandRoi((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                   hand_px / max(target_fraction, 1e-6), side, "previous", hand_px)


# ── cropping ─────────────────────────────────────────────────────────────────

def crop_roi(frame: np.ndarray, roi: HandRoi, out_size: int) -> np.ndarray:
    """Cut the ROI out of the native frame and scale it to the engine's input.

    The border is extended by reflection rather than filled with the letterbox
    grey. A raised hand is frequently at the edge of frame, and a flat grey
    half-field is a texture the weights never saw during training, whereas
    reflected skin and background is at least in distribution.
    """
    import cv2

    height, width = frame.shape[:2]
    side = max(1, int(round(roi.side)))
    x0 = int(round(roi.cx - side / 2.0))
    y0 = int(round(roi.cy - side / 2.0))

    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - width), max(0, y0 + side - height)
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_REFLECT_101)
        x0 += pad_l
        y0 += pad_t

    window = frame[y0:y0 + side, x0:x0 + side]
    if window.shape[0] != side or window.shape[1] != side:
        # Can only happen if the ROI is so far outside the frame that even the
        # reflected border does not reach it. Refuse rather than resize a
        # wrong-shaped window into something plausible.
        raise HandDecodeError(
            f"ROI {roi!r} does not intersect a {width}x{height} frame even "
            f"after reflection; got {window.shape[:2]}")
    interpolation = cv2.INTER_AREA if side > out_size else cv2.INTER_CUBIC
    return cv2.resize(window, (out_size, out_size), interpolation=interpolation)


def keypoints_to_frame(keypoints: np.ndarray, roi: HandRoi,
                       out_size: int) -> np.ndarray:
    """Map keypoints from crop coordinates back to native frame pixels.

    Deliberately not clipped to the frame, for the reason
    `vision_runtime.undo_letterbox_points` gives: a fingertip genuinely is
    sometimes outside the frame, and pinning it to the border invents a
    position that then reads as a real joint to the gesture rules.
    """
    scale = roi.side / float(max(out_size, 1))
    out = np.array(keypoints, dtype=np.float32, copy=True)
    out[..., 0] = out[..., 0] * scale + (roi.cx - roi.side / 2.0)
    out[..., 1] = out[..., 1] * scale + (roi.cy - roi.side / 2.0)
    return out


def box_to_frame(box, roi: HandRoi, out_size: int) -> list:
    """Map one xyxy box from crop coordinates back to native frame pixels."""
    scale = roi.side / float(max(out_size, 1))
    x0 = roi.cx - roi.side / 2.0
    y0 = roi.cy - roi.side / 2.0
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1 * scale + x0, y1 * scale + y0, x2 * scale + x0, y2 * scale + y0]


# ── merging into the skeleton payload ────────────────────────────────────────

def merged_keypoint_names(body_names) -> list:
    """Body names, then left hand, then right hand — matching `merge_keypoints`.

    The `_hand_` infix is not decoration. COCO-17 already has `left_wrist`, and
    so does the hand set — a plain `left_` prefix produces two joints called
    `left_wrist` in one payload, and a consumer that looks joints up by name
    (which is what `keypoint_names` is published for) silently gets whichever
    comes first.
    """
    return (list(body_names)
            + [f"left_hand_{name}" for name in HAND_KEYPOINTS]
            + [f"right_hand_{name}" for name in HAND_KEYPOINTS])


def merged_skeleton(body_skeleton, n_body: int) -> list:
    """Body bones plus both hands' bones, with the hand indices offset.

    Also joins each wrist to its hand, so the hand does not float detached
    from the arm in the renderer.
    """
    edges = [list(edge) for edge in body_skeleton]
    from plugins.vision_runtime import COCO_INDEX

    for order, side in enumerate(("left", "right")):
        offset = n_body + order * N_HAND_KEYPOINTS
        edges.extend([a + offset, b + offset] for a, b in HAND_SKELETON)
        edges.append([COCO_INDEX[f"{side}_wrist"], offset + HAND_INDEX["wrist"]])
    return edges


def merge_keypoints(body: np.ndarray, left: Optional[np.ndarray],
                    right: Optional[np.ndarray]) -> np.ndarray:
    """Body keypoints with both hands **appended**, never interleaved.

    Appending is what keeps this backwards compatible: a consumer that indexes
    the first 17 entries — including the dashboard renderer's own copy of the
    COCO bone table, and any payload a robot has already been wired against —
    keeps reading exactly what it read before. Interleaving by anatomy would
    look tidier and would silently move every existing index.

    A hand that was not decoded is filled with zero visibility rather than
    omitted, so the array's length does not depend on what was in frame. The
    renderer and the rules both already skip joints below a visibility
    threshold, so "absent" and "invisible" are the same thing downstream,
    whereas a variable-length array is a second shape for everyone to handle.
    """
    body = np.asarray(body, dtype=np.float32).reshape(-1, 3)
    blank = np.zeros((N_HAND_KEYPOINTS, 3), dtype=np.float32)
    parts = [body]
    for hand in (left, right):
        if hand is None:
            parts.append(blank.copy())
        else:
            parts.append(np.asarray(hand, dtype=np.float32).reshape(-1, 3))
    return np.concatenate(parts, axis=0)


# ── the per-node channel ─────────────────────────────────────────────────────

class HandChannel:
    """Runs the hand engine for the people in a frame, under a budget.

    Kept out of `plugins/pose.py` so that everything except `session.infer()`
    is testable without a GPU, and kept out of the node so that the throttle
    and the ROI bookkeeping have one owner.

    **Why it is throttled.** Measured end-to-end per ROI on an idle Orin 5
    (jp5.11) at 448: 11.26 ms, which is roughly what the whole body pose pass
    costs. Two hands every frame at 12 fps would be 22.5 ms on top of the
    41 ms the body channel already spends with three people in frame, leaving
    about 20 ms of the 83 ms frame for vop, depth, ASR and TTS on the same GPU.
    At 4 Hz the same two hands amortise to 7.5 ms per frame. A gesture lasts
    long enough that 4 Hz is not the limiting factor; the GPU is.

    The throttle is per (track, side) and staggered by track id, so two people
    do not both pay on the same frame — that would show up as a periodic
    stutter rather than as a higher average. Same device, and the same reason,
    as the activity backend's `activity_interval_s` in pose.py.
    """

    def __init__(self, session, *, max_rois: int = 2, interval_s: float = 0.25,
                 min_forearm_px: float = DEFAULT_MIN_FOREARM_PX,
                 min_conf: float = 0.3,
                 confidence: float = DEFAULT_HAND_CONFIDENCE,
                 target_fraction: float = TARGET_HAND_FRACTION,
                 retry_scales=HAND_RETRY_SCALES,
                 hold_s: float = DEFAULT_HAND_HOLD_S):
        self._session = session
        self._max_rois = max(1, int(max_rois))
        self._interval_s = max(0.0, float(interval_s))
        self._min_forearm_px = float(min_forearm_px)
        self._min_conf = float(min_conf)
        self._confidence = float(confidence)
        self._target_fraction = float(target_fraction)
        self._retry_scales = tuple(float(v) for v in retry_scales) or (1.0,)
        self._hold_s = max(0.0, float(hold_s))
        #: (track_id, side) -> {"box", "kpts", "t", "seen"}.
        #:
        #: Two timestamps, and they are not the same thing. `t` is the last
        #: *attempt* and drives the throttle; `seen` is the last *success* and
        #: drives the hold. Sharing one made a hand that could not be found
        #: cost an inference on every single frame — the throttle only ever
        #: applied to hands that were already working — while a shared
        #: timestamp updated on a miss would also mean the hold never expired.
        self._state: dict = {}
        #: Why hands were not produced, cumulative. Reported through `info`,
        #: because "no hands" has four innocent explanations and one broken one.
        self.skipped: dict = {reason: 0 for reason in SKIP_REASONS}
        self.ran = 0
        self.last_error: Optional[str] = None

    @property
    def input_size(self) -> int:
        width, _ = self._session.input_size
        return int(width)

    def reset(self) -> None:
        self._state.clear()

    def update(self, persons: list, frame, now: float) -> None:
        """Attach `person["hands"]` to each record, in place.

        Each value is `{"left": (21,3) or None, "right": ...}`. Runs at most
        `max_rois` inferences, spending them on the **largest** hands in frame
        — largest means nearest, and a nearer hand is both more likely to be
        addressing the robot and the only one the model can resolve fingers on.
        """
        from plugins.vision_runtime import COCO_INDEX, decode_poses

        candidates = []
        for person in persons:
            person.setdefault("hands", {"left": None, "right": None})
            keypoints = person.get("keypoints")
            if keypoints is None:
                continue
            track_id = person.get("id")
            for side in ("left", "right"):
                key = (track_id, side)
                previous = self._state.get(key)
                # **The body gate decides whether a hand is attempted at all,
                # and the previous box only decides where to crop.** Consulting
                # the previous box first looks equivalent and is not: once a
                # hand has a box, the crop would keep following it after the
                # wrist went out of view or the person walked out of range, so
                # the distance gate would apply to the first frame only and a
                # hand could be tracked indefinitely from a stale box. The arm
                # is what knows the hand is still there.
                roi, reason = roi_from_body(
                    keypoints, side, min_conf=self._min_conf,
                    min_forearm_px=self._min_forearm_px,
                    target_fraction=self._target_fraction,
                    coco_index=COCO_INDEX)
                if roi is not None:
                    tighter = roi_from_previous(
                        previous.get("box") if previous else None, side,
                        target_fraction=self._target_fraction)
                    if tighter is not None:
                        roi = tighter
                if roi is None:
                    # Out of reach. The hand is held for the grace period and
                    # then forgotten — a wrist that blinks below the
                    # visibility threshold for one frame is not the person
                    # turning away, but one that stays gone is. The whole
                    # entry goes once there is nothing left to hold: with no
                    # arm there is nothing to retry either, so there is no
                    # attempt to throttle.
                    held = self._hold(key, now)
                    if held is None:
                        self._state.pop(key, None)
                    if held is not None:
                        person["hands"][side] = held
                    if reason:
                        self.skipped[reason] += 1
                    continue
                if not self._due(key, now, track_id):
                    self.skipped["throttled"] += 1
                    if previous is not None and previous.get("kpts") is not None:
                        person["hands"][side] = previous["kpts"]
                    continue
                candidates.append((roi.hand_px, person, side, key, roi))

        # Largest first, and only as many as the budget allows. The ones
        # dropped here are counted as throttled rather than silently vanishing.
        candidates.sort(key=lambda item: item[0], reverse=True)
        for hand_px, person, side, key, roi in candidates[self._max_rois:]:
            self.skipped["throttled"] += 1
            previous = self._state.get(key)
            if previous is not None and previous.get("kpts") is not None:
                person["hands"][side] = previous["kpts"]

        for hand_px, person, side, key, roi in candidates[:self._max_rois]:
            found = None
            # Scales are only retried on a miss, and the loop stops at the
            # first hit — a hand that is found at the nominal crop costs
            # exactly one inference, as before.
            for scale in self._retry_scales:
                attempt = roi if scale == 1.0 else HandRoi(
                    roi.cx, roi.cy, roi.side * scale, roi.side_name,
                    roi.source, roi.hand_px * scale)
                try:
                    crop = crop_roi(frame, attempt, self.input_size)
                    outputs, _ = self._session.infer(crop)
                    boxes, scores, keypoints = decode_poses(
                        outputs, _identity_meta(self.input_size),
                        self._confidence, n_kpts=N_HAND_KEYPOINTS)
                except Exception as error:  # noqa: BLE001 — one hand must not kill the frame
                    self.last_error = f"{type(error).__name__}: {error}"
                    break
                self.ran += 1
                if len(scores):
                    found = (attempt, boxes, scores, keypoints)
                    break
            if found is None:
                # Nothing found at any scale. The **box** goes immediately —
                # the next attempt must re-derive the ROI from the arm rather
                # than chase a hand that is not there — but the keypoints are
                # held for the grace period, because one miss is not the hand
                # going away and dropping them instantly is what makes a live
                # hand flicker.
                # Record the attempt so the throttle applies to a hand that
                # cannot be found, not only to one that can. Without this a
                # hand the engine keeps missing is retried every frame — at
                # two scales — which measured as +49 ms per frame on a stream
                # where one of the two hands was marginal.
                entry = self._state.setdefault(key, {})
                entry.pop("box", None)
                entry["t"] = now
                held = self._hold(key, now)
                if held is not None:
                    person["hands"][side] = held
                continue
            roi, boxes, scores, keypoints = found
            best = _nearest_to_centre(boxes, self.input_size)
            mapped = keypoints_to_frame(keypoints[best], roi, self.input_size)
            person["hands"][side] = mapped
            self._state[key] = {
                "box": box_to_frame(boxes[best], roi, self.input_size),
                "kpts": mapped,
                "t": now,        # last attempt  -> throttle
                "seen": now,     # last success  -> hold
            }

    def _hold(self, key, now: float):
        """The last known keypoints for this hand, while they are still fresh.

        Measured against `seen` (the last success), not `t` (the last
        attempt), or recording a failed attempt would keep the hold alive
        forever.

        The entry itself survives an expired hold: it still carries `t`, which
        is what stops an unfindable hand from being retried on every frame.
        Only the stale keypoints are dropped.
        """
        entry = self._state.get(key)
        if entry is None:
            return None
        if self._hold_s <= 0 or (now - float(entry.get("seen", 0.0))) > self._hold_s:
            entry.pop("kpts", None)
            return None
        return entry.get("kpts")

    def _due(self, key, now: float, track_id) -> bool:
        if self._interval_s <= 0:
            return True
        previous = self._state.get(key)
        if previous is None:
            return True
        elapsed = now - float(previous.get("t", 0.0))
        if elapsed >= self._interval_s:
            return True
        # Stagger by track so two people do not both come due on the same
        # frame; that reads as a periodic stutter rather than a higher mean.
        try:
            offset = (int(track_id) % 3) * self._interval_s / 3.0
        except (TypeError, ValueError):
            offset = 0.0
        return elapsed >= (self._interval_s + offset)


def hands_in_frame(session, frame, *, confidence: float = 0.4,
                   scales=(1.0, 0.6, 1.6), max_hands: int = 4) -> list:
    """Find hands in a whole image, with no body to crop from.

    The ROI path needs an elbow and a wrist, and a photograph of a hand alone
    has neither — the body detector reports no arm, the geometric gate refuses,
    and the engine never runs. But this engine is a hand *detector* as well as
    a keypoint model, so when there is no arm to crop from, the frame itself
    is the crop.

    **Photographs only.** On a stream this would run the engine on every frame
    in which nobody's arms are visible, which is exactly the frames where the
    per-frame budget has nothing to spend. There the body gate is the right
    answer; here it is in the way.

    Returns a list of (keypoints_in_frame_pixels, score), largest first. No
    left/right: without an arm there is nothing that knows which hand it is,
    and guessing it from appearance is what the body keypoints exist to avoid.
    """
    import cv2

    height, width = frame.shape[:2]
    net = int(session.input_size[0])
    best: list = []
    for scale in scales:
        if scale == 1.0:
            view, offset_x, offset_y, used = frame, 0.0, 0.0, max(width, height)
        else:
            side = int(round(max(width, height) * scale))
            cx, cy = width / 2.0, height / 2.0
            roi = HandRoi(cx, cy, side, "", "frame", side)
            try:
                view = crop_roi(frame, roi, net)
            except HandDecodeError:
                continue
            offset_x, offset_y, used = cx - side / 2.0, cy - side / 2.0, side
        try:
            outputs, meta = session.infer(view)
            boxes, scores, keypoints = decode_poses(
                outputs, meta, confidence, n_kpts=N_HAND_KEYPOINTS)
        except Exception:  # noqa: BLE001 — a photo must still answer
            continue
        if not len(scores):
            continue
        order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1][:max_hands]
        for index in order:
            points = np.array(keypoints[index], dtype=np.float32, copy=True)
            if scale == 1.0:
                # Nothing to do: `session.infer` letterboxed the frame and
                # `decode_poses` already undid it, so these are frame pixels.
                # Undoing it a second time here inflated every hand by 1/scale
                # and pushed it off the left edge — a 500x917 photo came back
                # with a 770x634 hand starting at x = -159.
                pass
            else:
                factor = used / float(net)
                points[:, 0] = points[:, 0] * factor + offset_x
                points[:, 1] = points[:, 1] * factor + offset_y
            best.append((points, float(scores[index])))
        if best:
            break
    best.sort(key=lambda item: item[1], reverse=True)

    # **Second pass, sized from the measurement rather than from a guess.**
    # The first pass only has the frame to go on, and a photograph of a hand
    # fills most of it — about 90% against the 31% the model wants. It still
    # returns something (0.81 on the picture this was found with), and that
    # something is wrong in a way the score does not show: the skeleton came
    # back about 40% too small, with the wrist in the middle of the palm and
    # the fingertips stopping at the knuckles. Now that a box exists, re-crop
    # so the hand lands where the model is accurate and read it again.
    refined = []
    for points, score in best:
        xs, ys = points[:, 0], points[:, 1]
        span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        if span <= 1.0:
            refined.append((points, score)); continue
        roi = HandRoi(float((xs.min() + xs.max()) / 2.0),
                      float((ys.min() + ys.max()) / 2.0),
                      span / TARGET_HAND_FRACTION, "", "refined", span)
        try:
            crop = crop_roi(frame, roi, net)
            outputs, _ = session.infer(crop)
            boxes2, scores2, kpts2 = decode_poses(
                outputs, _identity_meta(net), confidence,
                n_kpts=N_HAND_KEYPOINTS)
        except Exception:  # noqa: BLE001 — keep the first pass rather than fail
            refined.append((points, score)); continue
        if not len(scores2):
            refined.append((points, score)); continue
        pick = int(np.argmax(scores2))
        refined.append((keypoints_to_frame(kpts2[pick], roi, net),
                        float(scores2[pick])))
    return refined[:max_hands]


def _nearest_to_centre(boxes, net: int) -> int:
    """Which detection in this crop is the hand the crop was cut for.

    **Not the highest-scoring one.** The crop is 3.2x the estimated hand box
    and is centred on where the arm says the hand is, so when a person holds
    their hands near each other — in front of the chest, or one over the
    other — both crops contain both hands. Picking by score then lets the two
    crops pick the *same* hand, and the card reports two skeletons stacked on
    one hand while the other has none.

    The crop's own centre is the prior: we cut it there because that is where
    the hand should be. A detection far from it is somebody else's hand, or
    the other one. This is the same trade the distance gate makes — geometry
    is reliable here and the score is not, and the score cannot even express
    "this is the wrong hand" because it is a perfectly good hand.
    """
    centre = net / 2.0
    best, best_d = 0, None
    for i, box in enumerate(boxes):
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        d = (cx - centre) ** 2 + (cy - centre) ** 2
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def _identity_meta(size: int):
    """A letterbox that did nothing, for a crop already at the engine's size.

    `decode_poses` insists on a meta so it can map boxes back; the crop was
    resized to the input exactly, so the inverse is the identity and the real
    mapping is `keypoints_to_frame`'s job.
    """
    from plugins.vision_runtime import LetterboxMeta

    return LetterboxMeta(1.0, 0, 0, size, size)
