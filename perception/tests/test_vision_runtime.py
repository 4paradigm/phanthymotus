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
    COCO_INDEX,
    COCO_KEYPOINTS,
    COCO_SKELETON,
    N_KEYPOINTS,
    PAD_VALUE,
    LetterboxMeta,
    VisionDecodeError,
    VisionEngineSession,
    decode_depth,
    decode_detections,
    decode_poses,
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


# ── pose decoding ────────────────────────────────────────────────────────────

def _pose_rows(entries, *, with_class: bool, n_kpts: int = 17) -> np.ndarray:
    """Build an engine-shaped pose output from (box, score, keypoints) tuples.

    `keypoints` is a list of (x, y, visibility) in *network input* space, which
    is what an engine emits — the decoder's job is to get them back out.
    """
    offset = 6 if with_class else 5
    rows = np.zeros((len(entries), offset + 3 * n_kpts), dtype=np.float32)
    for i, (box, score, keypoints) in enumerate(entries):
        rows[i, :4] = box
        rows[i, 4] = score
        if with_class:
            rows[i, 5] = 0.0                      # pose is single-class
        rows[i, offset:] = np.asarray(keypoints, dtype=np.float32).reshape(-1)
    return rows


def _uniform_keypoints(x: float, y: float, visibility: float = 0.9,
                       n_kpts: int = 17) -> list:
    return [(x, y, visibility)] * n_kpts


# scale 0.5, pad_y 80: a 1280x960 frame letterboxed into 640x640.
_POSE_META = LetterboxMeta(0.5, 0, 80, 1280, 960)


@pytest.mark.parametrize("with_class", [True, False])
def test_pose_round_trip_maps_box_and_keypoints_back(with_class):
    """Both layouts must yield the same geometry — the class column carries no
    information, it only moves where the keypoints begin."""
    rows = _pose_rows(
        [((50, 100, 250, 300), 0.9, _uniform_keypoints(100, 180))],
        with_class=with_class,
    )
    boxes, scores, keypoints = decode_poses(rows[None], _POSE_META, conf=0.25)

    assert boxes.shape == (1, 4)
    assert boxes[0] == pytest.approx([100.0, 40.0, 500.0, 440.0])
    assert scores == pytest.approx([0.9])
    assert keypoints.shape == (1, 17, 3)
    # (100 - 0) / 0.5 = 200 ; (180 - 80) / 0.5 = 200
    assert keypoints[0, 0, :2] == pytest.approx([200.0, 200.0])
    assert keypoints[0, :, 2] == pytest.approx([0.9] * 17)


def test_pose_decodes_the_transposed_orientation():
    """Output order and orientation are not stable across TensorRT majors."""
    rows = _pose_rows(
        [((50, 100, 250, 300), 0.8, _uniform_keypoints(100, 180))],
        with_class=True,
    )
    straight = decode_poses(rows, _POSE_META, conf=0.25)
    transposed = decode_poses(rows.T, _POSE_META, conf=0.25)
    for a, b in zip(straight, transposed):
        assert a == pytest.approx(b)


def test_pose_picks_its_tensor_by_content_not_by_index():
    """A pose engine emits more than one output; the pose rows may be second."""
    rows = _pose_rows(
        [((50, 100, 250, 300), 0.7, _uniform_keypoints(100, 180))],
        with_class=True,
    )
    decoy = np.full((1, 32, 160, 160), 0.5, dtype=np.float32)   # mask prototypes
    boxes, _, keypoints = decode_poses([decoy, rows], _POSE_META, conf=0.25)
    assert boxes.shape == (1, 4)
    assert keypoints[0, 0, :2] == pytest.approx([200.0, 200.0])


def test_pose_filters_by_score():
    rows = _pose_rows(
        [
            ((50, 100, 250, 300), 0.9, _uniform_keypoints(100, 180)),
            ((60, 110, 260, 310), 0.1, _uniform_keypoints(120, 200)),
        ],
        with_class=True,
    )
    boxes, scores, keypoints = decode_poses(rows, _POSE_META, conf=0.5)
    assert boxes.shape == (1, 4) and keypoints.shape == (1, 17, 3)
    assert scores == pytest.approx([0.9])


def test_pose_with_no_detections_decodes_to_empty_arrays():
    """Finding nobody is an answer, with the keypoint axis still present —
    a consumer that indexes [:, 0] must not get an IndexError on an empty frame."""
    boxes, scores, keypoints = decode_poses(
        np.zeros((1, 0, 57), dtype=np.float32), _POSE_META, conf=0.25)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert keypoints.shape == (0, 17, 3)


def test_pose_below_threshold_keeps_the_keypoint_axis():
    rows = _pose_rows(
        [((50, 100, 250, 300), 0.1, _uniform_keypoints(100, 180))],
        with_class=True,
    )
    _, _, keypoints = decode_poses(rows, _POSE_META, conf=0.5)
    assert keypoints.shape == (0, 17, 3)


def test_pose_keypoints_are_not_clipped_but_boxes_are():
    """A wrist outside the frame is real information; a box outside it is not.

    Pinning an off-frame joint to the border invents a position on the edge,
    which then reads as a real joint to both the action rules and the renderer.
    """
    rows = _pose_rows(
        # Keypoint at network (-50, 0) → original (-100, -160): the frame was
        # padded on y only, so pad_y is what pushes y negative.
        [((-100, -100, 250, 300), 0.9, _uniform_keypoints(-50, 0))],
        with_class=True,
    )
    boxes, _, keypoints = decode_poses(rows, _POSE_META, conf=0.25)
    assert boxes[0, 0] == pytest.approx(0.0)        # clipped into the frame
    assert boxes[0, 1] == pytest.approx(0.0)
    assert keypoints[0, 0, 0] == pytest.approx(-100.0)   # left as measured
    assert keypoints[0, 0, 1] == pytest.approx(-160.0)


def test_a_detection_only_output_is_refused_as_a_pose():
    """vop's 6-wide rows are not a pose; reading them as one must raise."""
    rows = np.zeros((4, 6), dtype=np.float32)
    rows[:, 4] = 0.9
    with pytest.raises(VisionDecodeError, match="pose"):
        decode_poses(rows, _POSE_META, conf=0.25)


def test_an_off_by_one_width_is_refused_rather_than_shifted():
    """Reading the keypoints one column off shifts every joint by half a
    coordinate and still draws a plausible skeleton — so it must not decode."""
    rows = np.zeros((2, 58), dtype=np.float32)
    rows[:, 4] = 0.9
    with pytest.raises(VisionDecodeError):
        decode_poses(rows, _POSE_META, conf=0.25)


def test_visibility_outside_zero_one_is_refused():
    """The visibility columns are the check that actually settles the layout:
    box and keypoint coordinates are both large positive numbers."""
    rows = _pose_rows(
        [((50, 100, 250, 300), 0.9, _uniform_keypoints(100, 180, visibility=7.5))],
        with_class=True,
    )
    with pytest.raises(VisionDecodeError):
        decode_poses(rows, _POSE_META, conf=0.25)


def test_a_dense_depth_output_is_not_mistaken_for_a_pose():
    with pytest.raises(VisionDecodeError):
        decode_poses(np.zeros((1, 1, 480, 640), dtype=np.float32),
                     _POSE_META, conf=0.25)


def test_skeleton_edges_reference_real_keypoints():
    """A bad index here draws a bone to a joint that does not exist, or
    silently to the wrong one."""
    assert len(COCO_KEYPOINTS) == N_KEYPOINTS == 17
    assert len(COCO_SKELETON) == 19
    for a, b in COCO_SKELETON:
        assert 0 <= a < N_KEYPOINTS and 0 <= b < N_KEYPOINTS
        assert a != b
    assert len(set(COCO_SKELETON)) == len(COCO_SKELETON)


def test_keypoint_index_matches_the_name_order():
    """COCO order is part of the weights; reindexing it swaps left and right."""
    assert COCO_INDEX["nose"] == 0
    assert COCO_INDEX["left_shoulder"] == 5 and COCO_INDEX["right_shoulder"] == 6
    assert COCO_INDEX["left_wrist"] == 9 and COCO_INDEX["right_wrist"] == 10
    assert COCO_INDEX["left_hip"] == 11 and COCO_INDEX["right_hip"] == 12
    assert COCO_INDEX["left_ankle"] == 15 and COCO_INDEX["right_ankle"] == 16
    assert all(COCO_INDEX[name] == i for i, name in enumerate(COCO_KEYPOINTS))


def test_the_real_pose_engine_geometry_decodes():
    """The shape the shipped engine actually declares.

    Measured, not assumed: exporting yolo26s-pose inside the jp6.1 perception
    image (TensorRT 10.4) reported

        input  "images"  shape(1, 3, 640, 640)  FLOAT
        output "output0" shape(1, 300, 57)      FLOAT

    57 = 6 + 3x17, so the engine carries the class column and the keypoints
    start at index 6; 300 rows is the NMS-free end-to-end head, the same fixed
    row count yoloe-26s-seg's (1, 300, 38) has. One output tensor, so the
    per-JetPack output *ordering* problem does not arise for pose — but the
    decode still picks by content, because that was also true of vop's engine
    on one of the two lines and not the other.

    Pinning it here means a future re-export that changes the layout fails in a
    test rather than on a robot.
    """
    rows = np.zeros((1, 300, 57), dtype=np.float32)
    rows[0, 0, :4] = (10, 20, 110, 420)
    rows[0, 0, 4] = 0.91
    rows[0, 0, 5] = 0.0                       # the class column
    rows[0, 0, 6:] = np.tile([55.0, 66.0, 0.8], 17)
    meta = LetterboxMeta(1.0, 0, 0, 640, 640)

    boxes, scores, keypoints = decode_poses(rows, meta, conf=0.5)
    assert boxes.shape == (1, 4) and keypoints.shape == (1, 17, 3)
    assert scores[0] == pytest.approx(0.91)
    assert keypoints[0, 0].tolist() == pytest.approx([55.0, 66.0, 0.8])
    # The other 299 rows are all-zero: score 0 is below any usable threshold,
    # so a fixed-row head does not report 299 phantom people.
    assert len(boxes) == 1
