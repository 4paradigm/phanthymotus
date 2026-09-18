#!/usr/bin/env python3
"""
plugins/scrfd_decode.py — SCRFD's detector head, decoded where it is produced.

## Why this is its own module

The detector returns its head **before thresholding**: nine arrays covering every
anchor of every stride. At 640x640 with strides 8/16/32 and two anchors that is

    (80*80 + 40*40 + 20*20) * 2 = 16 800 candidates

    scores  16 800 x 1  float32 =  67 kB
    bbox    16 800 x 4  float32 = 269 kB
    kps     16 800 x 10 float32 = 672 kB
                                 -------
                                 0.96 MiB per detection

of which a typical frame keeps **nought to eight**. Everything else is discarded by the
score threshold and NMS.

With the session in `plugins/ort_worker.py`'s child, leaving the decode in the parent
means shipping 16 800 candidates across a process boundary in order to throw 99.95% of
them away on the far side — measured at +27 ms per detection, which is most of the cost
of the process boundary. Thresholding belongs on the side that produced the data.

So this module holds the decode, imports **nothing but numpy**, and is used by both
sides: `plugins/face_runtime.py` calls it directly when the session is in-process, and
the worker child imports it by name to post-process before replying. One implementation,
no duplication, and the "these are pure functions" claim is structural rather than a
comment.

Nothing here knows about ONNX Runtime, cv2, ROS or the plugin. It takes the nine arrays
and the letterbox scale, and returns the survivors.
"""

from __future__ import annotations

import numpy as np

# SCRFD-500M-BNKPS: three feature levels, two anchors each, five keypoints. These
# describe the shipped `det_500m.onnx` and are asserted against its output count at
# load time in face_runtime.py — a detector without keypoints cannot be aligned.
FEAT_STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
NUM_KPS = 5


def distance2bbox(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    points = []
    for index in range(0, distances.shape[1], 2):
        points.append(centers[:, 0] + distances[:, index])
        points.append(centers[:, 1] + distances[:, index + 1])
    return np.stack(points, axis=-1)


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list:
    """Plain greedy IoU suppression — no cv2.dnn, no torchvision."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[current] + areas[rest] - inter)
        order = rest[iou <= thresh]
    return keep


def decode(outputs, input_h: int, input_w: int, scale: float,
           det_thresh: float, nms_thresh: float):
    """The nine head arrays -> `(boxes, keypoints, scores)` for the survivors only.

    `boxes` is (N,4) xyxy in **source-image** coordinates (the letterbox scale is
    divided out here), `keypoints` is (N,5,2), `scores` is (N,). N is typically 0-8
    against the 16 800 candidates that went in, which is the entire point.

    Clipping to the frame, sorting by area and the `max_faces` cut stay with the
    caller: they need the source dimensions and the plugin's own ordering, and they are
    free.

    This is a faithful move of the loop that was in `FaceAnalyzer.detect`, including the
    order of operations — NMS runs on the already-rescaled boxes, because it did there
    and the `+1` in its area terms makes it very slightly scale-dependent.
    """
    levels = len(FEAT_STRIDES)
    boxes_all, kps_all, scores_all = [], [], []

    for index, stride in enumerate(FEAT_STRIDES):
        scores = np.asarray(outputs[index]).reshape(-1)
        bbox_preds = np.asarray(outputs[index + levels]).reshape(-1, 4) * stride
        kps_preds = (np.asarray(outputs[index + levels * 2])
                     .reshape(-1, NUM_KPS * 2) * stride)

        grid_h, grid_w = input_h // stride, input_w // stride
        centers = np.stack(
            np.mgrid[:grid_h, :grid_w][::-1], axis=-1
        ).astype(np.float32).reshape(-1, 2) * stride
        if NUM_ANCHORS > 1:
            centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape(-1, 2)

        positive = np.where(scores >= det_thresh)[0]
        if positive.size == 0:
            continue
        boxes_all.append(distance2bbox(centers, bbox_preds)[positive])
        kps_all.append(
            distance2kps(centers, kps_preds)[positive].reshape(-1, NUM_KPS, 2))
        scores_all.append(scores[positive])

    if not scores_all:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros((0, NUM_KPS, 2), dtype=np.float32), \
            np.zeros((0,), dtype=np.float32)

    boxes = np.concatenate(boxes_all) / scale
    keypoints = np.concatenate(kps_all) / scale
    scores = np.concatenate(scores_all)
    keep = nms(boxes, scores, nms_thresh)
    return (boxes[keep].astype(np.float32),
            keypoints[keep].astype(np.float32),
            scores[keep].astype(np.float32))
