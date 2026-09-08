"""DAv2-Small metric depth on the shared Jetson TensorRT runtime."""

from __future__ import annotations

import logging
import math
import threading
import time
from numbers import Real
from typing import Mapping

import numpy as np

from utils.tensorrt_runtime import TensorRTEngine

from .contracts import ErrorCode, ObstacleDistanceError

log = logging.getLogger(__name__)

_INPUT_HEIGHT = 384
_INPUT_WIDTH = 512


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _check_deadline(deadline: float) -> None:
    if deadline > 0 and time.monotonic() >= deadline:
        raise ObstacleDistanceError(ErrorCode.TIMEOUT, "model inference timed out")


def _prepare_image(image: np.ndarray) -> np.ndarray:
    import cv2

    resized = cv2.resize(image, (_INPUT_WIDTH, _INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1))
    # Channel normalization and patch padding are already part of the engine.
    return np.ascontiguousarray(chw, dtype=np.float32)[None] / 255.0


def _resize_align_corners(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if image.shape == (height, width):
        return image.astype(np.float32, copy=False)
    source_height, source_width = image.shape
    x = np.linspace(0, source_width - 1, width, dtype=np.float32)
    y = np.linspace(0, source_height - 1, height, dtype=np.float32)
    map_x, map_y = np.meshgrid(x, y)
    return cv2.remap(
        image.astype(np.float32, copy=False), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


class MetricDepthBackend:
    """Return the ROI percentile directly in metres, without score calibration."""

    def __init__(self, engine_path: str, config: Mapping[str, object]):
        self._engine_path = engine_path
        self._engine: TensorRTEngine | None = None
        self._engine_init_lock = threading.Lock()
        if not isinstance(config, Mapping):
            raise ValueError("indoor configuration must be a mapping")
        self._roi = config.get("roi", (0, 300, 213, 426))
        reference = config.get("roi_reference_size", (480, 640))
        if (
            not isinstance(reference, (list, tuple)) or len(reference) != 2
            or any(type(v) is not int or v <= 0 for v in reference)
        ):
            raise ValueError("ROI reference size must contain two positive integers")
        self._reference_height, self._reference_width = reference
        if (
            not isinstance(self._roi, (list, tuple)) or len(self._roi) != 4
            or any(type(v) is not int for v in self._roi)
        ):
            raise ValueError("ROI must contain four integer coordinates")
        r0, r1, c0, c1 = self._roi
        if not (0 <= r0 < r1 <= reference[0] and 0 <= c0 < c1 <= reference[1]):
            raise ValueError("ROI is outside the reference image")
        self._percentile = _finite_number(config.get("depth_percentile", 1.0), "depth percentile")
        if not 0 <= self._percentile <= 100:
            raise ValueError("depth percentile must be between 0 and 100")
        self._minimum = _finite_number(config.get("min_output_distance_m", 0.3), "minimum distance")
        self._maximum = _finite_number(config.get("max_output_distance_m", 10.0), "maximum distance")
        if not 0 <= self._minimum < self._maximum:
            raise ValueError("indoor depth output range is invalid")
        self._min_valid_pixels = config.get("min_valid_pixels", 64)
        if type(self._min_valid_pixels) is not int or self._min_valid_pixels <= 0:
            raise ValueError("minimum valid pixels must be a positive integer")

    def _get_engine(self) -> TensorRTEngine:
        if self._engine is None:
            with self._engine_init_lock:
                if self._engine is None:
                    started = time.monotonic()
                    engine = TensorRTEngine(self._engine_path)
                    try:
                        if not engine.is_static or engine.input_shape != (1, 3, _INPUT_HEIGHT, _INPUT_WIDTH):
                            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "indoor engine input shape is incompatible")
                        if len(engine.output_names) != 1:
                            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "indoor engine must have one output")
                    except Exception:
                        engine.close()
                        raise
                    self._engine = engine
                    log.info("[obstacle] indoor depth engine loaded in %.1fms", 1000 * (time.monotonic() - started))
        return self._engine

    def close(self) -> None:
        with self._engine_init_lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()

    def predict_indoor_distance(self, image_bytes: bytes, deadline_monotonic: float) -> float:
        import cv2

        _check_deadline(deadline_monotonic)
        try:
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None or image.size == 0:
            raise ObstacleDistanceError(ErrorCode.INVALID_IMAGE, "image bytes could not be decoded")
        height, width = image.shape[:2]
        outputs = self._get_engine().infer(_prepare_image(image))
        if len(outputs) != 1:
            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "indoor engine must have one output")
        depth = np.asarray(outputs[0]).squeeze()
        if depth.shape != (_INPUT_HEIGHT, _INPUT_WIDTH):
            raise ObstacleDistanceError(ErrorCode.INVALID_DEPTH, "indoor engine output shape is incompatible")
        depth = _resize_align_corners(depth, height, width)
        r0, r1, c0, c1 = self._roi
        r0, r1 = (round(height * r / self._reference_height) for r in (r0, r1))
        c0, c1 = (round(width * c / self._reference_width) for c in (c0, c1))
        values = depth[r0:r1, c0:c1]
        values = values[np.isfinite(values) & (values > 0)]
        if values.size < self._min_valid_pixels:
            raise ObstacleDistanceError(ErrorCode.NO_VALID_DEPTH, "indoor ROI does not contain enough valid depth")
        distance = float(np.clip(np.percentile(values, self._percentile), self._minimum, self._maximum))
        _check_deadline(deadline_monotonic)
        return distance
