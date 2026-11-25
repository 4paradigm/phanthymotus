"""B6's declared dense layout, class-aware NMS, and legacy compatibility."""

import numpy as np
import pytest

from plugins.vision_runtime import (
    LetterboxMeta,
    VisionDecodeError,
    decode_detections,
)


METADATA = {
    "detection_layout": "yolo_dense_v1",
    "num_classes": 220,
    "nms_iou": 0.7,
    "max_det": 300,
}


def test_dense_class_aware_nms_and_threshold():
    output = np.zeros((1, 224, 8400), np.float32)
    output[0, :4, :4] = np.array([
        [100, 100, 100, 300], [100, 100, 100, 300],
        [80, 80, 80, 20], [80, 80, 80, 20],
    ])
    output[0, 4, :4] = [.9, .8, .2, .34]
    output[0, 5, 2] = .85
    boxes, scores, classes = decode_detections(
        [output], LetterboxMeta(.5, 20, 10, 1280, 720), .35,
        metadata=METADATA,
    )
    assert list(classes) == [0, 1]
    np.testing.assert_allclose(scores, [.9, .85])
    np.testing.assert_allclose(boxes, [[80, 100, 240, 260]] * 2)


def test_dense_nms_runs_before_original_image_clipping():
    output = np.zeros((1, 224, 8400), np.float32)
    output[0, :4, :2] = [[80, 100], [50, 50], [80, 40], [40, 40]]
    output[0, 4, :2] = [.9, .8]
    boxes, scores, _ = decode_detections(
        output, LetterboxMeta(1, 100, 0, 100, 100), .35,
        metadata=METADATA,
    )
    # IoU is .5 before clipping. Clipping first would make these identical
    # and incorrectly discard the second detection.
    assert len(scores) == 2
    np.testing.assert_allclose(boxes, [[0, 30, 20, 70]] * 2)


def test_dense_keeps_one_class_per_anchor_and_limits_sorted_results():
    output = np.zeros((1, 224, 8400), np.float32)
    output[0, :4, :3] = [[100, 200, 300], [100, 100, 100],
                           [20, 20, 20], [20, 20, 20]]
    output[0, 4, :3] = [.8, .7, .8]
    output[0, 5, 0] = .9
    _, scores, classes = decode_detections(
        output, LetterboxMeta(1, 0, 0, 640, 640), .35,
        metadata={**METADATA, "max_det": 2},
    )
    np.testing.assert_allclose(scores, [.9, .8])
    assert list(classes) == [1, 0]


def test_dense_empty_invalid_shape_and_nonfinite():
    output = np.zeros((1, 224, 8400), np.float16)
    meta = LetterboxMeta(1, 0, 0, 640, 640)
    assert decode_detections(output, meta, .35, metadata=METADATA)[0].shape == (0, 4)
    with pytest.raises(VisionDecodeError):
        decode_detections(output[:, :, :10], meta, .35, metadata=METADATA)
    output[0, 0, 0] = np.nan
    with pytest.raises(VisionDecodeError):
        decode_detections(output, meta, .35, metadata=METADATA)


def test_unknown_detection_layout_is_rejected():
    output = np.zeros((1, 224, 8400), np.float32)
    meta = LetterboxMeta(1, 0, 0, 640, 640)
    with pytest.raises(VisionDecodeError, match="Unsupported detection_layout"):
        decode_detections(output, meta, .35,
                          metadata={"detection_layout": "unknown"})


def test_legacy_detection_not_changed():
    output = np.array([[[10, 20, 30, 40, .9, 2],
                        [10, 20, 30, 40, .8, 2]]], np.float32)
    boxes, _, classes = decode_detections(
        output, LetterboxMeta(1, 0, 0, 640, 640), .35,
    )
    assert len(boxes) == 2
    assert list(classes) == [2, 2]
