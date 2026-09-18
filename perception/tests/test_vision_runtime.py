"""
tests/test_vision_runtime.py — letterbox geometry and engine-output decoding.

This is the part of the TensorRT path that fails *silently*: a wrong scale or a
transposed output does not raise, it just puts every box somewhere slightly
wrong, or reads scores as coordinates. Hence the round-trip tests.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

import vision_stubs  # noqa: F401  (installs the cv2 / ROS stubs)

from plugins.vision_runtime import (  # noqa: E402
    PAD_VALUE,
    LetterboxMeta,
    VisionDecodeError,
    VisionEngineSession,
    decode_depth,
    decode_detections,
    letterbox,
    to_blob,
    undo_letterbox,
)


# ── letterbox geometry ───────────────────────────────────────────────────────

def test_letterbox_preserves_aspect_and_centers():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, meta = letterbox(image, 640, 640)
    assert canvas.shape == (640, 640, 3)
    assert meta.scale == pytest.approx(1.0)       # 640/640 vs 640/480 → min is 1.0
    assert meta.pad_x == 0
    assert meta.pad_y == 80                       # (640-480)//2
    assert meta.orig_w == 640 and meta.orig_h == 480


def test_letterbox_pads_with_the_training_grey():
    image = np.zeros((100, 400, 3), dtype=np.uint8)
    canvas, meta = letterbox(image, 640, 640)
    # A row inside the padding band must be entirely PAD_VALUE.
    assert (canvas[0] == PAD_VALUE).all()
    assert meta.pad_y > 0


def test_letterbox_handles_a_portrait_frame():
    image = np.zeros((800, 400, 3), dtype=np.uint8)
    canvas, meta = letterbox(image, 640, 640)
    assert canvas.shape == (640, 640, 3)
    assert meta.scale == pytest.approx(0.8)
    assert meta.pad_x == 160 and meta.pad_y == 0


def test_empty_frame_is_refused():
    with pytest.raises(ValueError):
        letterbox(np.zeros((0, 10, 3), dtype=np.uint8), 640, 640)


# ── the inverse transform ────────────────────────────────────────────────────

def test_box_round_trips_through_the_letterbox():
    """A box in original coords → network coords → back must land where it started."""
    meta = LetterboxMeta(scale=0.5, pad_x=20, pad_y=60, orig_w=1280, orig_h=720)
    original = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)

    forward = original * meta.scale
    forward[:, [0, 2]] += meta.pad_x
    forward[:, [1, 3]] += meta.pad_y

    assert undo_letterbox(forward, meta) == pytest.approx(original, abs=1e-4)


def test_undo_letterbox_clips_to_the_frame():
    meta = LetterboxMeta(scale=1.0, pad_x=0, pad_y=0, orig_w=640, orig_h=480)
    boxes = np.array([[-50.0, -20.0, 900.0, 700.0]], dtype=np.float32)
    assert undo_letterbox(boxes, meta).tolist() == [[0.0, 0.0, 640.0, 480.0]]


def test_undo_letterbox_on_no_boxes():
    meta = LetterboxMeta(1.0, 0, 0, 640, 480)
    assert undo_letterbox(np.empty((0, 4), dtype=np.float32), meta).shape == (0, 4)


# ── blob ─────────────────────────────────────────────────────────────────────

def test_blob_is_nchw_rgb_normalized():
    canvas = np.zeros((4, 4, 3), dtype=np.uint8)
    canvas[..., 0] = 255          # pure blue in BGR
    blob = to_blob(canvas, np.float32)
    assert blob.shape == (1, 3, 4, 4)
    # BGR→RGB means the blue channel must land in index 2, not 0.
    assert blob[0, 2].max() == pytest.approx(1.0)
    assert blob[0, 0].max() == pytest.approx(0.0)


def test_blob_honours_the_engine_dtype():
    canvas = np.zeros((4, 4, 3), dtype=np.uint8)
    assert to_blob(canvas, np.float16).dtype == np.float16


# ── detection decoding ───────────────────────────────────────────────────────

def _rows(*entries):
    return np.array(entries, dtype=np.float32)[None]      # (1, N, 6)


def test_decode_filters_by_confidence():
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    output = _rows(
        [10, 10, 20, 20, 0.9, 0],
        [30, 30, 40, 40, 0.1, 1],
    )
    boxes, scores, classes = decode_detections(output, meta, conf=0.5)
    assert boxes.shape == (1, 4)
    assert scores.tolist() == pytest.approx([0.9])
    assert classes.tolist() == [0]


def test_decode_accepts_the_transposed_layout():
    """(1, 6, N) must decode identically to (1, N, 6)."""
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    rows = _rows([10, 10, 20, 20, 0.9, 3])
    from_rows = decode_detections(rows, meta, conf=0.5)
    from_cols = decode_detections(rows.transpose(0, 2, 1), meta, conf=0.5)
    assert from_rows[0] == pytest.approx(from_cols[0])
    assert from_rows[2].tolist() == from_cols[2].tolist() == [3]


def test_decode_applies_the_inverse_letterbox():
    meta = LetterboxMeta(scale=0.5, pad_x=20, pad_y=60, orig_w=1280, orig_h=720)
    output = _rows([70.0, 160.0, 170.0, 260.0, 0.9, 0])
    boxes, _, _ = decode_detections(output, meta, conf=0.5)
    # (70-20)/0.5 = 100, (160-60)/0.5 = 200, etc.
    assert boxes[0].tolist() == pytest.approx([100.0, 200.0, 300.0, 400.0])


def test_decode_returns_empty_when_nothing_passes():
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    boxes, scores, classes = decode_detections(_rows([1, 1, 2, 2, 0.01, 0]), meta, conf=0.5)
    assert boxes.shape == (0, 4) and scores.size == 0 and classes.size == 0


def test_the_real_seg_layout_decodes():
    """yoloe-26s-seg emits (1, 300, 38): 4 box + score + class + 32 mask coeffs.

    The first decoder keyed off "an axis of length 6" and rejected this
    outright — no axis is 6. Values here are taken from a real engine run.
    """
    meta = LetterboxMeta(scale=0.5926, pad_x=80, pad_y=0, orig_w=810, orig_h=1080)
    row = np.zeros(38, dtype=np.float32)
    row[:6] = [478.0, 233.5, 559.0, 519.0, 0.93, 0.0]
    row[6:] = np.linspace(-3, 3, 32)          # mask coefficients, ignored
    output = np.zeros((1, 300, 38), dtype=np.float32)
    output[0, 0] = row

    boxes, scores, classes = decode_detections(output, meta, conf=0.25)
    assert len(boxes) == 1
    assert classes.tolist() == [0]
    assert scores[0] == pytest.approx(0.93, abs=1e-4)
    # (478-80)/0.5926 ≈ 671.6 — and within the 810-wide original frame.
    assert boxes[0][0] == pytest.approx(671.6, abs=0.5)
    assert boxes[0][2] <= 810.0


def _seg_outputs(order):
    """The two tensors a yoloe-26s-seg engine emits, in the requested order."""
    row = np.zeros(38, dtype=np.float32)
    row[:6] = [478.0, 233.5, 559.0, 519.0, 0.93, 7.0]
    boxes = np.zeros((1, 300, 38), dtype=np.float32)
    boxes[0, 0] = row
    protos = np.random.default_rng(1).normal(0, 1, (1, 32, 160, 160)).astype(np.float32)
    return [boxes, protos] if order == "boxes-first" else [protos, boxes]


@pytest.mark.parametrize("order", ["boxes-first", "protos-first"])
def test_detection_output_is_found_regardless_of_engine_output_order(order):
    """TensorRT 10.3 lists these as [output0, output1]; TensorRT 8.5 reverses it.

    Indexing outputs[0] worked on jp6.1 and decoded mask prototypes as boxes on
    jp5.11 — a real failure caught only by running both lines.
    """
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    boxes, scores, classes = decode_detections(_seg_outputs(order), meta, conf=0.25)
    assert classes.tolist() == [7]
    assert scores[0] == pytest.approx(0.93, abs=1e-4)


@pytest.mark.parametrize("order", ["depth-only", "depth-second"])
def test_depth_output_is_found_regardless_of_order(order):
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    depth = np.full((1, 1, 640, 640), 4.0, dtype=np.float32)
    outputs = [depth] if order == "depth-only" else [
        np.zeros((1, 32, 160, 160), dtype=np.float32), depth
    ]
    assert decode_depth(outputs, meta).shape == (640, 640)


def test_an_unrecognised_layout_raises_instead_of_being_guessed():
    """Every wrong reading of these numbers still looks like plausible boxes."""
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    # The pre-e2e layout: (1, 4+nc, anchors), logits — no column of scores in
    # [0,1] sitting next to a column of integral class ids.
    bad = np.random.default_rng(0).normal(5.0, 20.0, size=(1, 85, 8400)).astype(np.float32)
    with pytest.raises(VisionDecodeError, match="nms=False"):
        decode_detections(bad, meta, conf=0.5)


def test_scores_outside_0_1_are_not_read_as_scores():
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    rows = np.tile(np.array([10, 10, 20, 20, 7.5, 3], dtype=np.float32), (12, 1))[None]
    with pytest.raises(VisionDecodeError):
        decode_detections(rows, meta, conf=0.5)


def test_non_integral_class_column_is_not_read_as_classes():
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    rows = np.tile(np.array([10, 10, 20, 20, 0.9, 3.7], dtype=np.float32), (12, 1))[None]
    with pytest.raises(VisionDecodeError):
        decode_detections(rows, meta, conf=0.5)


def test_a_rank_1_output_raises():
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    with pytest.raises(VisionDecodeError):
        decode_detections(np.zeros((6,), dtype=np.float32), meta, conf=0.5)


# ── depth decoding ───────────────────────────────────────────────────────────

def test_depth_is_cropped_back_to_the_real_image_area():
    """The padded band carries no measurement and must not reach the consumer."""
    meta = LetterboxMeta(scale=1.0, pad_x=0, pad_y=80, orig_w=640, orig_h=480)
    output = np.zeros((640, 640), dtype=np.float32)
    output[80:560, :] = 5.0          # the real image area
    cropped = decode_depth(output, meta)
    assert cropped.shape == (480, 640)
    assert (cropped == 5.0).all()


def test_depth_squeezes_leading_axes():
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    assert decode_depth(np.zeros((1, 1, 640, 640), dtype=np.float32), meta).shape == (640, 640)


def test_depth_with_an_unreadable_shape_raises():
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    with pytest.raises(VisionDecodeError):
        decode_depth(np.zeros((3, 4, 5), dtype=np.float32), meta)


# ── class names out of engine metadata ───────────────────────────────────────

class _MetaOnlySession(VisionEngineSession):
    """Exercises class_names() without a real engine behind it."""

    def __init__(self, metadata):
        self._meta = metadata

    @property
    def metadata(self):
        return self._meta


def test_class_names_come_back_in_index_order():
    """JSON turns ultralytics' int keys into strings; order must survive."""
    session = _MetaOnlySession({"names": {"2": "forklift", "0": "person", "10": "door"}})
    assert session.class_names() == ["person", "forklift", "door"]


def test_class_names_accept_a_plain_list():
    assert _MetaOnlySession({"names": ["a", "b"]}).class_names() == ["a", "b"]


@pytest.mark.parametrize("metadata", [
    {},
    {"names": None},
    {"names": {"a": "person"}},      # non-integer keys: order is unknowable
])
def test_unusable_names_metadata_yields_nothing(metadata):
    """Empty lets the caller fall back to vocab.json rather than guess an order."""
    assert _MetaOnlySession(metadata).class_names() == []


def test_an_empty_detection_output_decodes_to_no_boxes():
    """Finding nothing is an answer, not an unreadable layout.

    The e2e head emits a fixed 300 rows so hardware never produces this, but a
    zero-row output is well-formed and used to raise — which surfaced as a
    photo with no objects in it failing instead of returning an empty list.
    """
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)
    boxes, scores, classes = decode_detections(
        np.zeros((1, 0, 6), dtype=np.float32), meta, conf=0.25)
    assert boxes.shape == (0, 4)
    assert scores.size == 0 and classes.size == 0
