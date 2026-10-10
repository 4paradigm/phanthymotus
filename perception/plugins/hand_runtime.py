#!/usr/bin/env python3
"""
plugins/hand_runtime.py — hand ROI geometry and the 21-keypoint decode.

The pose card already reports COCO-17 per person. This module takes those body
keypoints and gets 21 keypoints per hand out of a second engine, so the card can
say what a hand is *shaped* like rather than only where the wrist is.

Everything here is geometry, normalisation and bookkeeping; the engine itself
runs through `utils.tensorrt_runtime.TensorRTEngine`.

**The keypoint model is RTMPose-m (hand5), not a YOLO pose head.** The card
shipped first with a YOLO26-pose model fine-tuned on hand keypoints, and on a
real camera it was not good enough to build gestures on: an OK sign came back
with the wrist in the middle of the palm, an open hand came back as a clump,
and motion blur collapsed the joints while the detection score stayed at 0.8.
Compared side by side on the same crops, RTMPose was right on every picture the
YOLO model got wrong, and it is also **cheaper** — 3.58 ms against 7.99 ms per
hand on an Orin NX at 1020 MHz, 4.33 vs 8.19 on jp5.11.

Two consequences shape this file. RTMPose is **top-down**: it assumes whatever
is in the box is a hand, so there is no detector and no detection score, and
the box has to be roughly right. And it is **SimCC**: the output is two 1-D
heatmaps per joint rather than rows of detections, so `decode_poses` is not
involved at all.

**Why there is a crop.** Top-down needs one, and the arm supplies it: measured
on a real photograph, RTMPose read the hand correctly from an arm-derived box
anywhere between 0.30 and 0.65 of the box the old model wanted — a 2.2x range,
far more slack than the old model had. It fails only at the very loose crop the
YOLO model needed, which is why an early comparison appeared to show RTMPose
losing: it had been handed the wrong box.

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

#: Fraction of the box edge the hand should occupy.
#:
#: 0.6 for RTMPose, against 0.31 for the YOLO model this replaced — a top-down
#: pose model wants the hand to fill its box, a detector wants room to look
#: around. Measured by sweeping the box on a real photograph and reading the
#: model's own confidence: 0.30-0.65 of the old crop all produced a correct
#: hand, peaking at 0.65 (which is 0.6 expressed this way); the old 3.2x crop
#: failed outright.
TARGET_HAND_FRACTION = 0.6

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

#: A retry at a second crop scale is **not** needed with this model and was
#: removed with the one that needed it. The YOLO head had to be fed a hand at
#: 31% of its input and missed outright outside a narrow band, so a miss was
#: retried tighter; RTMPose read the same hand correctly anywhere between 0.30
#: and 0.65 of that crop. One attempt, one inference.

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


#: ImageNet statistics, in RGB. RTMPose normalises with these rather than
#: scaling to [0, 1] the way the YOLO engines do, and the two are not
#: interchangeable: feeding /255 produces a confident, completely wrong hand.
#: Taken from the pipeline.json shipped inside the model's own ONNX SDK bundle,
#: not from memory.
RTMPOSE_MEAN = (123.675, 116.28, 103.53)
RTMPOSE_STD = (58.395, 57.12, 57.375)


class HandDecodeError(ValueError):
    """Raised when an engine cannot be read as a 21-keypoint SimCC head."""


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


# ── the SimCC head ───────────────────────────────────────────────────────────

def _simcc_outputs(session, outputs) -> tuple:
    """Pick (simcc_x, simcc_y) out of the engine's outputs, **by name**.

    Not by position, and not by content — the two tensors have identical
    shapes (N, 21, 512), so there is nothing in the numbers to tell them
    apart, and this repository has already been bitten by TensorRT listing an
    engine's outputs in a different order on jp6.1 than on jp5.11 (see
    `vision_runtime._find_detection_rows`). Swapping them transposes every
    hand and still draws a perfectly plausible one.
    """
    names = list(session.output_names)
    arrays = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    by_name = {name: np.asarray(a) for name, a in zip(names, arrays)}
    try:
        return by_name["simcc_x"], by_name["simcc_y"]
    except KeyError:
        raise HandDecodeError(
            f"hand engine outputs {names}; expected 'simcc_x' and 'simcc_y'. "
            f"The two tensors are the same shape, so they cannot be told "
            f"apart by content — an engine that does not name them cannot be "
            f"used. Rebuild from the published ONNX, which does."
        ) from None


def assert_simcc(session, n_kpts: int = N_HAND_KEYPOINTS) -> tuple:
    """Refuse an engine that is not a 21-keypoint SimCC head. Returns shapes.

    Checked at load from a real inference rather than from a declaration, and
    at load rather than per frame: a card that comes up `running` and then
    fails on every frame is much harder to diagnose than one that refuses to
    start.
    """
    width, height = session.input_size
    probe = np.zeros((1, 3, height, width), dtype=session.input_dtype)
    outputs = session.infer(probe)
    sx, sy = _simcc_outputs(session, outputs)
    if sx.ndim != 3 or sx.shape[1] != n_kpts or sx.shape != sy.shape:
        raise HandDecodeError(
            f"hand engine produced simcc {sx.shape}/{sy.shape}; expected "
            f"(N, {n_kpts}, bins) for both"
        )
    if sx.shape[2] % width:
        raise HandDecodeError(
            f"simcc_x has {sx.shape[2]} bins for a {width}px input, which is "
            f"not a whole split ratio — this is not the model it claims to be"
        )
    return tuple(sx.shape), tuple(sy.shape)


def decode_simcc(sx: np.ndarray, sy: np.ndarray, net: int) -> np.ndarray:
    """Two 1-D heatmaps per joint -> (N, 21, 3) in network-input pixels.

    The bin count is a whole multiple of the input edge (512 bins for 256 px,
    a split ratio of 2), and it is read off the tensor rather than hardcoded:
    a different RTMPose size would change it and a wrong constant would scale
    every hand by a constant factor while still looking like a hand.

    The confidence per joint is the *smaller* of the two peaks. A joint is
    only as well located as its worse axis, and taking the larger would report
    a joint that is certain in x and a guess in y as certain.
    """
    sx = np.asarray(sx, dtype=np.float32)
    sy = np.asarray(sy, dtype=np.float32)
    ratio = sx.shape[-1] / float(net)
    xs = sx.argmax(-1) / ratio
    ys = sy.argmax(-1) / ratio
    conf = np.minimum(sx.max(-1), sy.max(-1))
    return np.stack([xs, ys, conf], axis=-1).astype(np.float32)


def to_rtmpose_blob(crop: np.ndarray, dtype) -> np.ndarray:
    """BGR HWC uint8 -> NCHW RGB, ImageNet-normalised, in the engine dtype.

    Deliberately not `vision_runtime.to_blob`: that scales to [0, 1], which is
    what the YOLO engines were trained with. RTMPose subtracts a mean and
    divides by a standard deviation, and feeding it the other one yields a
    confident hand in the wrong place — the failure mode this whole module
    exists to avoid.
    """
    rgb = np.asarray(crop)[:, :, ::-1].astype(np.float32)
    rgb -= np.asarray(RTMPOSE_MEAN, dtype=np.float32)
    rgb /= np.asarray(RTMPOSE_STD, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=dtype)


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


class HandEngine:
    """The hand engine, with the small surface the channel needs.

    Deliberately **not** `vision_runtime.VisionEngineSession`. That one
    letterboxes a frame and scales it to [0, 1] on the way in, which is what
    the YOLO engines were trained with; RTMPose wants a square crop resized
    without padding and normalised with ImageNet statistics. Routing this
    through the other wrapper is not a small mismatch — it produces a
    confident hand in the wrong place, with nothing raising.
    """

    def __init__(self, engine_path, *, device_id: int = 0):
        from utils.tensorrt_runtime import TensorRTEngine

        self._engine = TensorRTEngine(engine_path, device_id=device_id)
        shape = self._engine.input_shape or self._engine.optimization_shape
        if shape is None or len(shape) != 4:
            raise HandDecodeError(
                f"hand engine must take one NCHW input; got shape {shape}")
        if int(shape[2]) != int(shape[3]):
            raise HandDecodeError(
                f"hand engine input must be square; got {shape[2]}x{shape[3]} "
                f"— the crop is a square cut out of the frame and a "
                f"non-square input would stretch it")
        self._net = int(shape[3])

    @property
    def input_size(self) -> tuple:
        return (self._net, self._net)

    @property
    def input_dtype(self):
        return self._engine.input_dtype

    @property
    def output_names(self) -> list:
        return list(self._engine.output_names)

    def infer(self, blob):
        return self._engine.infer(blob)

    def close(self) -> None:
        self._engine.close()


# ── the per-node channel ─────────────────────────────────────────────────────

class HandChannel:
    """Runs the hand model for the people in a frame, under a budget.

    Kept out of `plugins/pose.py` so that everything except `session.infer()`
    is testable without a GPU, and kept out of the node so that the throttle
    and the ROI bookkeeping have one owner.

    **Why it is throttled.** One hand costs 3.58 ms on an Orin NX at 1020 MHz
    and 4.33 ms on jp5.11 — cheaper than the model this replaced, but the body
    pass, vop, depth, ASR and TTS share the same GPU, and a gesture lasts long
    enough that a few samples a second is not the limiting factor. The
    throttle is per (track, side) and staggered by track id so two people do
    not both pay on the same frame, which would read as a periodic stutter
    rather than a higher average.

    **There is no detection step.** RTMPose always returns 21 joints for
    whatever is in the box, with a per-joint localisation confidence, so a
    "miss" here means *low confidence*, not *nothing found*. That is the one
    place a score is trusted in this module, and it is a different kind of
    score from a detector's: it is the peak of the joint's own 1-D heatmap,
    and it did move the right way on the box that was wrong (0.40 against
    0.68-0.79 on boxes that worked).
    """

    def __init__(self, session, *, max_rois: int = 2, interval_s: float = 0.25,
                 min_forearm_px: float = DEFAULT_MIN_FOREARM_PX,
                 min_conf: float = 0.3,
                 confidence: float = DEFAULT_HAND_CONFIDENCE,
                 target_fraction: float = TARGET_HAND_FRACTION,
                 hold_s: float = DEFAULT_HAND_HOLD_S):
        self._session = session
        self._max_rois = max(1, int(max_rois))
        self._interval_s = max(0.0, float(interval_s))
        self._min_forearm_px = float(min_forearm_px)
        self._min_conf = float(min_conf)
        self._confidence = float(confidence)
        self._target_fraction = float(target_fraction)
        self._hold_s = max(0.0, float(hold_s))
        #: (track_id, side) -> {"box", "kpts", "t", "seen"}.
        #:
        #: Two timestamps, and they are not the same thing. `t` is the last
        #: *attempt* and drives the throttle; `seen` is the last *success* and
        #: drives the hold. Sharing one made a hand that could not be read
        #: cost an inference on every single frame — the throttle only ever
        #: applied to hands that were already working — while a shared
        #: timestamp updated on a failure would also mean the hold never
        #: expired.
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
        addressing the robot and the one whose fingers can be resolved.
        """
        from plugins.vision_runtime import COCO_INDEX

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
                # hand could be tracked indefinitely from a stale box.
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
                    held = self._hold(key, now)
                    if held is None:
                        self._state.pop(key, None)
                    else:
                        person["hands"][side] = held
                    if reason:
                        self.skipped[reason] += 1
                    continue
                if not self._due(key, now, track_id):
                    self.skipped["throttled"] += 1
                    held = self._hold(key, now)
                    if held is not None:
                        person["hands"][side] = held
                    continue
                candidates.append((roi.hand_px, person, side, key, roi))

        # Largest first, and only as many as the budget allows. The ones
        # dropped here are counted as throttled rather than silently vanishing.
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _px, person, side, key, _roi in candidates[self._max_rois:]:
            self.skipped["throttled"] += 1
            held = self._hold(key, now)
            if held is not None:
                person["hands"][side] = held

        for _px, person, side, key, roi in candidates[:self._max_rois]:
            try:
                crop = crop_roi(frame, roi, self.input_size)
                blob = to_rtmpose_blob(crop, self._session.input_dtype)
                outputs = self._session.infer(blob)
                sx, sy = _simcc_outputs(self._session, outputs)
                decoded = decode_simcc(sx, sy, self.input_size)[0]
            except Exception as error:  # noqa: BLE001 — one hand must not kill the frame
                self.last_error = f"{type(error).__name__}: {error}"
                entry = self._state.setdefault(key, {})
                entry["t"] = now
                continue
            self.ran += 1
            # Record the attempt so the throttle applies to a hand that cannot
            # be read, not only to one that can. Without this a hand whose
            # confidence keeps falling short is retried on every frame.
            entry = self._state.setdefault(key, {})
            entry["t"] = now
            if float(np.mean(decoded[:, 2])) < self._confidence:
                entry.pop("box", None)
                held = self._hold(key, now)
                if held is not None:
                    person["hands"][side] = held
                continue
            mapped = keypoints_to_frame(decoded, roi, self.input_size)
            person["hands"][side] = mapped
            xs, ys = mapped[:, 0], mapped[:, 1]
            entry.update({
                # The hand's own extent is the next crop's measurement, in
                # place of the arm's estimate.
                "box": [float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())],
                "kpts": mapped,
                "seen": now,
            })

    def _hold(self, key, now: float):
        """The last known keypoints for this hand, while they are still fresh.

        Measured against `seen` (the last success), not `t` (the last
        attempt), or recording a failed attempt would keep the hold alive
        forever.

        The entry itself survives an expired hold: it still carries `t`, which
        is what stops an unreadable hand from being retried on every frame.
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


def hands_in_frame(session, frame, *, confidence: float = DEFAULT_HAND_CONFIDENCE,
                   max_hands: int = 1) -> list:
    """Read a hand from a whole image, with no body to crop from.

    The ROI path needs an elbow and a wrist, and a photograph of a hand alone
    has neither — the body detector reports no arm, the geometric gate
    refuses, and nothing runs. But a top-down model's assumption is exactly
    that the box *is* the hand, and for a close-up the frame is that box. No
    detector is involved and none is needed.

    **Photographs only.** On a stream this would run on every frame in which
    nobody's arms are visible, which is exactly the frames where the per-frame
    budget has nothing to spend. There the body gate is the right answer.

    Returns a list of (keypoints_in_frame_pixels, confidence). At most one:
    without a detector there is no way to find a second hand, and pretending
    otherwise would be inventing one. No left/right either — that comes from
    which wrist the crop was derived from, and there is no wrist here.
    """
    import cv2

    net = int(session.input_size[0])
    height, width = frame.shape[:2]
    # A square view of the whole picture, padded by reflection rather than
    # squashed: squashing changes the hand's aspect ratio, and every joint
    # then lands on a hand shape nobody has.
    side = max(width, height)
    roi = HandRoi(width / 2.0, height / 2.0, side, "", "frame", side)
    try:
        crop = crop_roi(frame, roi, net)
        blob = to_rtmpose_blob(crop, session.input_dtype)
        sx, sy = _simcc_outputs(session, session.infer(blob))
        decoded = decode_simcc(sx, sy, net)[0]
    except Exception:  # noqa: BLE001 — a photo must still answer
        return []
    score = float(np.mean(decoded[:, 2]))
    if score < confidence:
        return []
    return [(keypoints_to_frame(decoded, roi, net), score)][:max_hands]
