"""Exact OSF preprocessing, optionally inside the existing isolated ORT worker.

Transmit original uint8 crop pixels, then apply the same float32 resize and
normalization as the frozen pipeline. Resizing uint8 before conversion would
change interpolation/rounding and is deliberately avoided.
"""
import cv2
import numpy as np

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def prepare_crop(bgr):
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 4:
        raise ValueError('invalid OSF crop')
    rgb = bgr[:, :, ::-1].astype(np.float32)
    crop = cv2.resize(rgb, (224, 224)) / 255.
    crop -= MEAN
    crop /= STD
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None])


def prepare_feeds(feeds, input_name):
    """The raw ROI is a transport envelope, never an ONNX model input."""
    if set(feeds) == {'bgr_crop'}:
        return {input_name: prepare_crop(feeds['bgr_crop'])}
    if set(feeds) != {input_name}:
        raise ValueError('unexpected OSF input fields')
    tensor = feeds[input_name]
    if tensor.dtype != np.float32 or tensor.shape != (1, 3, 224, 224):
        raise ValueError('invalid prepared OSF tensor')
    return feeds
