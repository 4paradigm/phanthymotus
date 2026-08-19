"""Shared image and configuration helpers for obstacle inference backends."""

from __future__ import annotations

import os
import time
from typing import Mapping

import numpy as np

from .contracts import ErrorCode, ObstacleDistanceError


SEG_INPUT_SIZE = 640
SEG_CONF_FLOOR = 0.05


def check_deadline(deadline_monotonic: float) -> None:
    if deadline_monotonic > 0 and time.monotonic() >= deadline_monotonic:
        raise ObstacleDistanceError(ErrorCode.TIMEOUT, "model inference timed out")


def decode_image(image_bytes: bytes) -> np.ndarray:
    import cv2

    try:
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
    except Exception:
        image = None
    if image is None or image.size == 0:
        raise ObstacleDistanceError(
            ErrorCode.INVALID_IMAGE,
            "image bytes could not be decoded",
        )
    return image


def letterbox(
    image: np.ndarray,
    size: int,
) -> tuple[np.ndarray, float, int, int]:
    import cv2

    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    unpad_h = int(round(height * ratio))
    unpad_w = int(round(width * ratio))
    dw = (size - unpad_w) // 2
    dh = (size - unpad_h) // 2
    resized = cv2.resize(
        image,
        (unpad_w, unpad_h),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[dh : dh + unpad_h, dw : dw + unpad_w] = resized
    return canvas, ratio, dw, dh


def model_path(config: Mapping, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value or not os.path.isfile(value):
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            f"{key} does not identify a model file",
        )
    return value
