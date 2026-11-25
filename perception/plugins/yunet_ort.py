"""CPU fallback for the OpenCV Zoo YuNet 2023mar ONNX graph."""

import numpy as np


_SIDE = 640
_STRIDES = (8, 16, 32)
_HEADS = (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10))
_SCHEMA = {
    f"{head}_{stride}": [1, (_SIDE // stride) ** 2, channels]
    for stride in _STRIDES for head, channels in _HEADS
}


def cpu_session_spec(path):
    return {
        "model_path": path,
        "providers": ["CPUExecutionProvider"],
        "options": {
            "intra_op_num_threads": 2,
            "session_config": {"session.intra_op.allow_spinning": "0"},
        },
    }


def _nms(rows, limit):
    # Match FaceDetectorYN's score ordering and integer rectangle IoU.
    order = np.argsort(-rows[:, 14], kind="stable")[:5000]
    rects = np.trunc(rows[:, :4])
    start = rects[:, :2]
    end = start + rects[:, 2:]
    areas = np.prod(rects[:, 2:], axis=1)
    kept = []
    while order.size and len(kept) < limit:
        best, order = order[0], order[1:]
        kept.append(best)
        overlap = np.maximum(0, np.minimum(end[best], end[order]) -
                             np.maximum(start[best], start[order]))
        intersection = np.prod(overlap, axis=1)
        order = order[intersection <= 0.3 * (areas[best] + areas[order] - intersection)]
    return rows[kept]


def _decode(outputs, width, height, threshold, limit):
    candidates = []
    for stride in _STRIDES:
        cls, obj, bbox, kps = (outputs[f"{head}_{stride}"][0] for head, _ in _HEADS)
        if not all(np.isfinite(head).all() for head in (cls, obj, bbox, kps)):
            raise ValueError("non-finite YuNet model output")
        scores = np.sqrt(np.clip(cls[:, 0], 0, 1) * np.clip(obj[:, 0], 0, 1))
        indices = np.flatnonzero(scores >= threshold)
        y, x = np.divmod(indices, _SIDE // stride)
        inside = (x * stride < width) & (y * stride < height)
        indices = indices[inside]
        grid = np.column_stack((x[inside], y[inside])).astype(np.float32)
        centers = (bbox[indices, :2] + grid) * stride
        with np.errstate(over="ignore", under="ignore"):
            sizes = np.exp(bbox[indices, 2:]) * stride
        points = (kps[indices].reshape(-1, 5, 2) + grid[:, None, :]) * stride
        top_left = centers - sizes / 2
        rows = np.column_stack((top_left, sizes, points.reshape(-1, 10), scores[indices]))
        valid = (np.isfinite(rows).all(axis=1) & (sizes > 0).all(axis=1)
                 & (top_left < (width, height)).all(axis=1)
                 & (centers + sizes / 2 > 0).all(axis=1))
        candidates.append(rows[valid])
    return _nms(np.concatenate(candidates), limit)


class YuNetOrtService:
    """Own one session inside Perception's shared, restartable ORT worker."""

    def __init__(self, session, threshold, limit):
        self.session = session
        self.threshold, self.limit = threshold, limit
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].shape != [1, 3, _SIDE, _SIDE]
                or inputs[0].type != "tensor(float)"
                or {out.name: out.shape for out in outputs} != _SCHEMA
                or any(out.type != "tensor(float)" for out in outputs)):
            raise ValueError("unsupported YuNet 2023mar ONNX model schema")
        self.input_name = inputs[0].name

    def close(self):
        self.session = None

    def describe(self):
        return {"max_side": _SIDE, "providers": self.session.get_providers()}

    def detect(self, image):
        height, width = image.shape[:2]
        if not (0 < height <= _SIDE and 0 < width <= _SIDE):
            raise ValueError("YuNet frame exceeds ONNX input dimensions")
        # FaceDetectorYN uses raw float BGR in NCHW order with zero padding
        # below and to the right. This graph fixes the padded size at 640x640.
        blob = np.zeros((1, 3, _SIDE, _SIDE), dtype=np.float32)
        blob[0, :, :height, :width] = image.transpose(2, 0, 1)
        names = list(_SCHEMA)
        outputs = dict(zip(names, self.session.run(names, {self.input_name: blob})))
        if {name: out.shape for name, out in outputs.items()} != {
            name: tuple(shape) for name, shape in _SCHEMA.items()
        }:
            raise ValueError("unexpected YuNet 2023mar output shape")
        return _decode(outputs, width, height, self.threshold, self.limit)
