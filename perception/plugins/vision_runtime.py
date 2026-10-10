#!/usr/bin/env python3
"""
plugins/vision_runtime.py — letterbox preprocessing and output decoding for the
YOLO-family TensorRT engines used by vop (detection), visual_depth (dense depth)
and pose (COCO-17 keypoints).

Everything CUDA/TensorRT-related lives in `utils.tensorrt_runtime.TensorRTEngine`;
this module is the model-specific layer on top, exactly as
`plugins/ocr_runtime.py` is for OCR.

**Why this exists rather than `ultralytics.YOLO("....engine")`.** Measured on an
Orin NX 8GB at 640, batch 1:

    yolov8s-worldv2, PyTorch eager (what this replaced)   35.6 ms
    yoloe-26s-seg,  TensorRT engine *via ultralytics*     37.3 ms
    pure GPU time for the engine (trtexec)                 ~5   ms

Swapping the backend under ultralytics bought nothing: its Python
pre/post-processing costs ~30 ms per frame either way. The speedup only exists
if that path is bypassed, which is what this module does.

The decoding here assumes engines exported with `nms=False`, i.e. YOLO26's
NMS-free end-to-end head, whose output is already final boxes. That is what
`tools/export_vision_engines.py` produces; an engine exported the other way
has a different output layout and is rejected rather than silently misread.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

PAD_VALUE = 114          # ultralytics' letterbox grey; matching it keeps the
                         # input distribution the weights were trained on


class VisionDecodeError(ValueError):
    """Raised when an engine's output does not match any layout we can read."""


# ── preprocessing ────────────────────────────────────────────────────────────

class LetterboxMeta:
    """How a frame was fitted into the network input, so boxes can come back.

    Getting this wrong does not raise anywhere — it just puts every box in
    slightly the wrong place, which is why the inverse lives next to the
    forward transform and both are unit-tested.
    """

    __slots__ = ("scale", "pad_x", "pad_y", "orig_w", "orig_h")

    def __init__(self, scale: float, pad_x: float, pad_y: float, orig_w: int, orig_h: int):
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.orig_w = orig_w
        self.orig_h = orig_h

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return (f"LetterboxMeta(scale={self.scale:.4f}, pad=({self.pad_x}, {self.pad_y}), "
                f"orig={self.orig_w}x{self.orig_h})")


def letterbox(image: np.ndarray, dst_w: int, dst_h: int) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize preserving aspect ratio and pad to (dst_h, dst_w). Returns HWC uint8."""
    import cv2

    orig_h, orig_w = image.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"empty frame: {image.shape}")

    scale = min(dst_w / orig_w, dst_h / orig_h)
    new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((dst_h, dst_w, image.shape[2]), PAD_VALUE, dtype=np.uint8)
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, LetterboxMeta(scale, pad_x, pad_y, orig_w, orig_h)


def to_blob(canvas: np.ndarray, dtype) -> np.ndarray:
    """HWC BGR uint8 → NCHW RGB float in [0,1], contiguous, in the engine dtype."""
    rgb = canvas[:, :, ::-1]
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    blob /= 255.0
    return blob.astype(dtype, copy=False)


def undo_letterbox(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map xyxy boxes from network input space back to original frame pixels."""
    if boxes.size == 0:
        return boxes
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] -= meta.pad_x
    out[:, [1, 3]] -= meta.pad_y
    out /= meta.scale
    # Assigned back explicitly: `np.clip(out[:, [0, 2]], ..., out=out[:, [0, 2]])`
    # writes into the temporary that fancy indexing produces, so the clip is
    # silently lost and boxes keep running off the frame.
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, meta.orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, meta.orig_h)
    return out


# ── detection decoding ───────────────────────────────────────────────────────

def _looks_like_detection_rows(rows: np.ndarray) -> bool:
    """Do these rows actually carry [x1, y1, x2, y2, score, class, ...]?

    Orientation is decided by checking the columns mean what they would have to
    mean, not by matching a shape. An earlier version keyed off "an axis of
    length 6" and rejected the real engine outright: yoloe-26s-seg emits
    (1, 300, 38) — 4 box + score + class + 32 mask coefficients — so no axis is
    6 at all. Shape alone also cannot tell (N, C) from (C, N) once both exceed
    6, whereas scores confined to [0, 1] and integral class ids can.
    """
    if rows.ndim != 2 or rows.shape[1] < 6:
        return False
    if rows.shape[0] == 0:
        # No detections is a valid answer, not an unreadable layout. The width
        # still settles the orientation: an empty (0, 6+) is the rows form,
        # while its transpose is (6+, 0) and fails the width check above.
        return True
    scores = rows[:, 4]
    if not np.all((scores >= -1e-3) & (scores <= 1.0 + 1e-3)):
        return False
    classes = rows[:, 5]
    return bool(np.all(classes >= -1e-3) and np.allclose(classes, np.round(classes), atol=1e-3))


def _as_candidates(outputs) -> list:
    """Normalize one array or a list of engine outputs into a candidate list."""
    if isinstance(outputs, (list, tuple)):
        return [np.asarray(o) for o in outputs]
    return [np.asarray(outputs)]


def _find_detection_rows(outputs) -> Optional[np.ndarray]:
    """Pick the output that reads as detection rows, in either orientation.

    Selected by content, never by index. The same yoloe-26s-seg export lists
    its two tensors as ['output0', 'output1'] under TensorRT 10.3 (jp6.1) and
    ['output1', 'output0'] under TensorRT 8.5 (jp5.11) — so `outputs[0]` is the
    boxes on one JetPack line and the mask prototypes on the other. Indexing
    would have worked on whichever line it was written against and failed on
    the other.
    """
    for array in _as_candidates(outputs):
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            continue
        if _looks_like_detection_rows(array):
            return array
        if _looks_like_detection_rows(array.T):
            return array.T
    return None


def decode_detections(outputs, meta: LetterboxMeta, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode an NMS-free end-to-end head into (boxes_xyxy, scores, class_ids).

    Takes the engine's full output list and finds the right tensor itself.
    Columns beyond the sixth (mask coefficients, for a -seg export) are ignored:
    vop publishes boxes, and carrying prototypes through would cost a matrix
    multiply per frame for something nothing consumes.

    Outputs that satisfy no orientation raise rather than being interpreted,
    because every wrong reading of these numbers still produces
    plausible-looking boxes.
    """
    rows = _find_detection_rows(outputs)
    if rows is None:
        shapes = [tuple(a.shape) for a in _as_candidates(outputs)]
        raise VisionDecodeError(
            f"no engine output {shapes} reads as [x1,y1,x2,y2,score,class,...] "
            "in either orientation — was the engine exported with nms=False?"
        )

    scores = rows[:, 4].astype(np.float32)
    keep = scores >= conf
    if not keep.any():
        empty = np.empty((0, 4), dtype=np.float32)
        return empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    kept = rows[keep]
    return (
        undo_letterbox(kept[:, :4], meta),
        kept[:, 4].astype(np.float32),
        kept[:, 5].astype(np.int32),
    )


# ── pose decoding ────────────────────────────────────────────────────────────
#
# COCO-17, the keypoint set every YOLO pose export uses. Order is part of the
# weights, not a convention we get to pick — reindexing it silently swaps left
# and right, which still produces a plausible-looking skeleton.
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

N_KEYPOINTS = len(COCO_KEYPOINTS)

# Index of each name, so callers say COCO_INDEX["left_wrist"] rather than 9.
COCO_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# The 19 bones drawn between those joints. Shared with the dashboard renderer
# (web/js/renderers/pose2d.js keeps its own copy, since it cannot import this).
COCO_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),      # legs + hips
    (5, 11), (6, 12), (5, 6),                              # torso
    (5, 7), (7, 9), (6, 8), (8, 10),                        # arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # head + neck
)


def _pose_layout(rows: np.ndarray, n_kpts: int) -> Optional[int]:
    """Column offset at which the keypoint triples start, or None.

    Two layouts exist in the wild and they differ by exactly one column:

        [x1, y1, x2, y2, score, class, kx, ky, kv × K]   offset 6
        [x1, y1, x2, y2, score,        kx, ky, kv × K]   offset 5

    Pose is single-class, so the class column carries no information — but
    whether it is *there* decides where the keypoints begin, and reading them
    one column off shifts every joint by half a coordinate. For K=17 the two
    widths are 57 and 56, so the width alone picks the layout unambiguously;
    the content checks below are what reject a tensor that merely happens to
    be that wide.
    """
    if rows.ndim != 2:
        return None
    for offset in (6, 5):
        if rows.shape[1] != offset + 3 * n_kpts:
            continue
        if rows.shape[0] == 0:
            # No detections is a valid answer; the width already settled the
            # layout. Same rule as _looks_like_detection_rows.
            return offset
        scores = rows[:, 4]
        if not np.all((scores >= -1e-3) & (scores <= 1.0 + 1e-3)):
            continue
        if offset == 6:
            classes = rows[:, 5]
            if not (np.all(classes >= -1e-3)
                    and np.allclose(classes, np.round(classes), atol=1e-3)):
                continue
        # Per-keypoint visibility is also a probability. This is the check that
        # actually discriminates: box coordinates and keypoint coordinates are
        # both large positive numbers, so only the [0, 1] columns tell the
        # orientation and the offset apart.
        visibility = rows[:, offset + 2::3]
        if not np.all((visibility >= -1e-3) & (visibility <= 1.0 + 1e-3)):
            continue
        return offset
    return None


def _find_pose_rows(outputs, n_kpts: int) -> tuple[Optional[np.ndarray], int]:
    """Pick the output that reads as pose rows, in either orientation.

    By content, never by index — the same export lists its tensors in opposite
    order under TensorRT 10.4 and 8.5, which is recorded at length on
    _find_detection_rows. Indexing here would have worked on one JetPack line
    and quietly decoded garbage on the other.
    """
    for array in _as_candidates(outputs):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            continue
        for candidate in (array, array.T):
            offset = _pose_layout(candidate, n_kpts)
            if offset is not None:
                return candidate, offset
    return None, 0


def undo_letterbox_points(points: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map xy points from network input space back to original frame pixels.

    Deliberately NOT clipped to the frame, unlike undo_letterbox. A box is a
    region of the image and a region outside it is meaningless, but a wrist
    *is* sometimes outside the frame, and pinning it to the border invents a
    position on the edge — which then reads as a real joint to both the action
    rules and the renderer. Out-of-frame stays out of frame; the caller decides
    what to do with it.
    """
    if points.size == 0:
        return points
    out = points.astype(np.float32, copy=True)
    out[..., 0] -= meta.pad_x
    out[..., 1] -= meta.pad_y
    out /= meta.scale
    return out


def decode_poses(outputs, meta: LetterboxMeta, conf: float,
                 n_kpts: int = N_KEYPOINTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a pose head into (boxes_xyxy, scores, keypoints).

    `keypoints` is (N, n_kpts, 3) — x, y in original frame pixels and the
    per-joint visibility the model emitted, left as it came. Boxes come back
    through undo_letterbox (clipped), keypoints through
    undo_letterbox_points (not clipped; see there).

    decode_detections cannot be reused for this. It settles the orientation by
    requiring column 5 to be integral class ids, and in a pose row column 5 is
    either a class id or a keypoint x depending on the layout — so it would
    accept the 6-column reading of a 57-wide tensor and treat 51 keypoint
    numbers as mask coefficients to ignore.

    An output that satisfies no layout raises rather than being interpreted:
    every wrong reading of these numbers still draws a plausible skeleton.
    """
    rows, offset = _find_pose_rows(outputs, n_kpts)
    if rows is None:
        shapes = [tuple(np.asarray(a).shape) for a in _as_candidates(outputs)]
        raise VisionDecodeError(
            f"no engine output {shapes} reads as "
            f"[x1,y1,x2,y2,score,(class,)kx,ky,kv × {n_kpts}] in either "
            f"orientation — expected width {5 + 3 * n_kpts} or "
            f"{6 + 3 * n_kpts}; was the engine exported from a *-pose model "
            f"with nms=False?"
        )

    scores = rows[:, 4].astype(np.float32)
    keep = scores >= conf
    if not keep.any():
        return (np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty((0, n_kpts, 3), dtype=np.float32))

    kept = rows[keep]
    triples = kept[:, offset:offset + 3 * n_kpts].reshape(-1, n_kpts, 3)
    keypoints = np.empty_like(triples, dtype=np.float32)
    keypoints[:, :, :2] = undo_letterbox_points(triples[:, :, :2], meta)
    keypoints[:, :, 2] = triples[:, :, 2]
    return (
        undo_letterbox(kept[:, :4], meta),
        kept[:, 4].astype(np.float32),
        keypoints,
    )


# ── depth decoding ───────────────────────────────────────────────────────────

def decode_depth(outputs, meta: LetterboxMeta) -> np.ndarray:
    """Decode a dense depth output and crop the letterbox padding back off.

    Like decode_detections, this takes the engine's full output list and picks
    by shape rather than by index — output ordering is not stable across
    TensorRT versions.

    The padded border carries no real measurement; leaving it in would put a
    band of invented depth down the sides of every frame from a camera whose
    aspect ratio is not the network's.
    """
    array = None
    for candidate in _as_candidates(outputs):
        squeezed = np.squeeze(np.asarray(candidate, dtype=np.float32))
        if squeezed.ndim == 2:
            array = squeezed
            break
    if array is None:
        shapes = [tuple(np.asarray(a).shape) for a in _as_candidates(outputs)]
        raise VisionDecodeError(f"no engine output {shapes} is a 2-D depth map")

    inner_w = max(1, round(meta.orig_w * meta.scale))
    inner_h = max(1, round(meta.orig_h * meta.scale))
    pad_x, pad_y = int(meta.pad_x), int(meta.pad_y)
    if pad_y + inner_h <= array.shape[0] and pad_x + inner_w <= array.shape[1]:
        array = array[pad_y:pad_y + inner_h, pad_x:pad_x + inner_w]
    else:
        log.warning("[vision] depth output %s smaller than its letterbox window; "
                    "using it uncropped", array.shape)
    return array


# ── engine session ───────────────────────────────────────────────────────────

class VisionEngineSession:
    """One TensorRT engine plus the letterbox that feeds it.

    Thread safety comes from TensorRTEngine, which serializes `infer`; this
    class adds no mutable state of its own beyond the cached input geometry.
    """

    def __init__(self, engine_path, *, device_id: int = 0):
        from utils.tensorrt_runtime import TensorRTEngine

        self._engine = TensorRTEngine(engine_path, device_id=device_id)
        shape = self._engine.input_shape or self._engine.optimization_shape
        if shape is None or len(shape) != 4:
            raise VisionDecodeError(
                f"vision engine must take one NCHW input; got shape {shape}"
            )
        self._in_h, self._in_w = int(shape[2]), int(shape[3])
        log.info("[vision] engine %s input %dx%d, outputs %s",
                 engine_path, self._in_w, self._in_h, self._engine.output_names)

    @property
    def input_size(self) -> tuple[int, int]:
        return self._in_w, self._in_h

    @property
    def output_names(self) -> list:
        return list(self._engine.output_names)

    @property
    def metadata(self) -> dict:
        """The JSON header ultralytics' exporter prefixes to the engine."""
        return dict(getattr(self._engine, "metadata", None) or {})

    def class_names(self) -> list:
        """Class names as recorded *inside the engine*, or [] if absent.

        This is the authoritative vocabulary: it was written by the same export
        that baked the classes into the weights, so unlike a file shipped
        alongside it, it cannot drift out of order or out of date.
        """
        names = self.metadata.get("names")
        if isinstance(names, dict):
            # ultralytics writes {0: "person", 1: "door", ...}, and a JSON round
            # trip turns those keys into strings.
            try:
                return [names[key] for key in sorted(names, key=lambda k: int(k))]
            except (ValueError, TypeError):
                return []
        if isinstance(names, (list, tuple)):
            return list(names)
        return []

    def infer(self, frame: np.ndarray) -> tuple[list, LetterboxMeta]:
        canvas, meta = letterbox(frame, self._in_w, self._in_h)
        blob = to_blob(canvas, self._engine.input_dtype)
        return self._engine.infer(blob), meta

    def close(self) -> None:
        self._engine.close()
