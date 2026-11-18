"""Decode the Ultralytics YOLOE end-to-end segmentation head into per-card class rasters."""
from io import BytesIO
import math

import cv2
import numpy as np

from plugins.vision_runtime import VisionDecodeError, undo_letterbox


def decode_camera_jpeg(data):
    from PIL import Image
    if not data or len(data) > 16 * 1024 * 1024:
        raise ValueError("Expected a JPEG under 16 MiB")
    with Image.open(BytesIO(data)) as header:
        if header.format != "JPEG" or header.width * header.height > 3840 * 2160:
            raise ValueError("Expected a camera JPEG of at most 8.3 MP")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("Invalid camera JPEG")
    return image


def _head(outputs):
    # TensorRT 8/10 may swap binding order. The coefficient count is the
    # channel count of the prototype tensor, independent of the export layout.
    arrays = [np.asarray(a) for a in outputs]
    protos = [a[0] for a in arrays if a.ndim == 4 and a.shape[0] == 1]
    if len(protos) != 1:
        raise VisionDecodeError("Expected one segmentation prototype tensor [1,C,H,W]")
    proto = protos[0]
    for a in arrays:
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 2:
            for rows in (a, a.T):
                if rows.shape[1] == 6 + len(proto):
                    return rows, proto.astype(np.float32, copy=False)
    raise VisionDecodeError("Expected segmentation detections [xyxy, score, class, mask coefficients]")


def semantic_maps(outputs, meta, input_size, vocabulary, presets, confidence):
    """One uint8 raster per preset. 0 is background, 1..N indexes its configured class list."""
    rows, proto = _head(outputs)
    width, height = input_size
    maps = {key: np.zeros((meta.orig_h, meta.orig_w), np.uint8) for key in presets}
    targets = {}
    for key, labels in presets.items():
        for index, name in enumerate(labels, 1):
            targets.setdefault(name, []).append((key, index))
    active = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= confidence) & (rows[:, 4] <= 1)]
    boxes = undo_letterbox(active[:, :4], meta)
    content_width, content_height = max(1, round(meta.orig_w * meta.scale)), max(1, round(meta.orig_h * meta.scale))
    if meta.pad_x + content_width > width or meta.pad_y + content_height > height:
        raise VisionDecodeError("Letterbox geometry does not match model input")
    for i in np.argsort(-active[:, 4], kind="stable"):
        row, box = active[i], boxes[i]
        cls = int(row[5])
        if row[5] != cls or not 0 <= cls < len(vocabulary):
            raise VisionDecodeError("Segmentation class ID is outside the engine vocabulary")
        destinations = targets.get(vocabulary[cls])
        if not destinations:
            continue
        x1, y1, x2, y2 = map(math.ceil, box)
        if x1 >= x2 or y1 >= y2:
            continue
        logits = (row[6:].astype(np.float32) @ proto.reshape(len(proto), -1)).reshape(proto.shape[1:])
        net = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR)
        content = net[meta.pad_y:meta.pad_y+content_height, meta.pad_x:meta.pad_x+content_width]
        mask = cv2.resize(content, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)[y1:y2, x1:x2] > 0
        for key, label in destinations:
            region = maps[key][y1:y2, x1:x2]
            region[mask & (region == 0)] = label
    return maps


def overlay(image, labels):
    palette = np.array([[0, 0, 0]] + [[(79*i+41) % 208 + 35, (97*i+99) % 208 + 35,
                                       (53*i+11) % 208 + 35] for i in range(1, 256)], dtype=np.uint8)
    blend = cv2.addWeighted(image, 0.48, palette[labels], 0.52, 0)
    return np.where(labels[:, :, None] != 0, blend, image)
