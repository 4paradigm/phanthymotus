"""Independent TensorRT backends for ZipDepth indoor decisions and YOLO vehicle depth.

ZipDepth predicts affine-invariant inverse depth, not metric depth. The indoor
branch converts a calibrated ROI statistic to a distance while preserving the
configured decision boundary. Vehicle images use metric YOLO depth and
segmentation.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from numbers import Real
from typing import Mapping

import numpy as np

from .contracts import (
    DepthBackend,
    DepthPrediction,
    ErrorCode,
    InstanceSegmentationBackend,
    ObstacleDistanceError,
    SceneDomain,
)
from .native_tensorrt_backends import (
    NativeTensorRTSegBackend,
    _NativeTensorRTEngine,
    _YOLO_DEPTH_INPUT_SIZE,
    _prepare_yolo_image,
    _resize_align_corners,
    _scale_depth_to_original,
)
from .runtime_utils import check_deadline, decode_image, model_path


_ZIPDEPTH_INPUT_HEIGHT = 384
_ZIPDEPTH_INPUT_WIDTH = 512

log = logging.getLogger(__name__)


def _finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, f"{name} is invalid")
    converted = float(value)
    if (
        not math.isfinite(converted)
        or (minimum is not None and converted < minimum)
        or (maximum is not None and converted > maximum)
    ):
        raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, f"{name} is invalid")
    return converted


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, f"{name} is invalid")
    return value


def _roi(value: object) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "ZipDepth ROI is invalid")
    row_start, row_end, col_start, col_end = value
    if row_start < 0 or col_start < 0 or row_end <= row_start or col_end <= col_start:
        raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "ZipDepth ROI is invalid")
    return row_start, row_end, col_start, col_end


def _prepare_zipdepth_image(image: np.ndarray) -> np.ndarray:
    import cv2

    resized = cv2.resize(
        image,
        (_ZIPDEPTH_INPUT_WIDTH, _ZIPDEPTH_INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = resized[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1))
    return np.ascontiguousarray(chw, dtype=np.float32)[None] / 255.0


class ZipDepthYoloTensorRTDepthBackend:
    """ZipDepth direct indoor classifier plus metric YOLO vehicle depth."""

    def __init__(
        self,
        indoor_engine: str,
        vehicle_engine: str,
        *,
        indoor_config: Mapping[str, object],
        decision_threshold_m: object = 1.0,
    ) -> None:
        # Load each engine on first use so a single-domain deployment does not
        # pay for the other domain's TensorRT engine and CUDA allocations.
        self._indoor_engine_path = indoor_engine
        self._vehicle_engine_path = vehicle_engine
        self._indoor: _NativeTensorRTEngine | None = None
        self._vehicle: _NativeTensorRTEngine | None = None
        self._engine_init_lock = threading.Lock()

        self._roi = _roi(indoor_config.get("roi"))
        reference_size = indoor_config.get("roi_reference_size", (480, 640))
        if (
            not isinstance(reference_size, (list, tuple))
            or len(reference_size) != 2
        ):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "ZipDepth ROI reference size is invalid",
            )
        self._reference_height = _positive_integer(
            reference_size[0], name="ZipDepth ROI reference height"
        )
        self._reference_width = _positive_integer(
            reference_size[1], name="ZipDepth ROI reference width"
        )
        self._percentile = _finite_number(
            indoor_config.get("inverse_depth_percentile", 95.0),
            name="ZipDepth percentile",
            minimum=0.0,
            maximum=100.0,
        )
        self._score_threshold = _finite_number(
            indoor_config.get("score_threshold", -0.06),
            name="ZipDepth score threshold",
        )
        self._distance_scale = _finite_number(
            indoor_config.get(
                "inverse_depth_distance_scale",
                -27.0,
            ),
            name="ZipDepth distance scale",
        )
        self._distance_bias_m = _finite_number(
            indoor_config.get(
                "inverse_depth_distance_bias_m",
                3.7,
            ),
            name="ZipDepth distance bias",
        )
        self._decision_threshold_m = _finite_number(
            decision_threshold_m,
            name="obstacle decision threshold",
            minimum=0.0,
        )
        if self._decision_threshold_m <= 0:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "obstacle decision threshold is invalid",
            )
        self._classification_margin_m = _finite_number(
            indoor_config.get("classification_margin_m", 0.001),
            name="ZipDepth classification margin",
            minimum=0.0,
        )
        if not 0 < self._classification_margin_m < self._decision_threshold_m:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "ZipDepth classification margin is invalid",
            )
        self._min_output_distance_m = _finite_number(
            indoor_config.get("min_output_distance_m", 0.0),
            name="ZipDepth minimum output distance",
            minimum=0.0,
        )
        self._max_output_distance_m = _finite_number(
            indoor_config.get("max_output_distance_m", 10.0),
            name="ZipDepth maximum output distance",
            minimum=0.0,
        )
        if (
            self._min_output_distance_m
            > self._decision_threshold_m - self._classification_margin_m
            or self._max_output_distance_m < self._decision_threshold_m
        ):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "ZipDepth output range is incompatible with the decision threshold",
            )
        self._min_valid_pixels = _positive_integer(
            indoor_config.get("min_valid_pixels", 64),
            name="ZipDepth minimum valid pixels",
        )

    def _scaled_roi(self, height: int, width: int) -> tuple[int, int, int, int]:
        row_start, row_end, col_start, col_end = self._roi
        scaled = (
            round(height * row_start / self._reference_height),
            round(height * row_end / self._reference_height),
            round(width * col_start / self._reference_width),
            round(width * col_end / self._reference_width),
        )
        r0, r1, c0, c1 = scaled
        if (
            r0 < 0
            or c0 < 0
            or r1 > height
            or c1 > width
            or r1 <= r0
            or c1 <= c0
        ):
            raise ObstacleDistanceError(
                ErrorCode.INVALID_DEPTH,
                "scaled ZipDepth ROI is outside the image",
            )
        return scaled

    def _get_indoor_engine(self) -> _NativeTensorRTEngine:
        engine = self._indoor
        if engine is None:
            with self._engine_init_lock:
                if self._indoor is None:
                    started = time.monotonic()
                    indoor = _NativeTensorRTEngine(self._indoor_engine_path)
                    if indoor.input_shape != (
                        1,
                        3,
                        _ZIPDEPTH_INPUT_HEIGHT,
                        _ZIPDEPTH_INPUT_WIDTH,
                    ):
                        raise ObstacleDistanceError(
                            ErrorCode.MODEL_ERROR,
                            "ZipDepth TensorRT engine input shape is incompatible",
                        )
                    if len(indoor.output_names) != 1:
                        raise ObstacleDistanceError(
                            ErrorCode.MODEL_ERROR,
                            "ZipDepth TensorRT engine must have one output",
                        )
                    log.info(
                        "[obstacle] ZipDepth indoor engine loaded in %.1fms",
                        1000.0 * (time.monotonic() - started),
                    )
                    self._indoor = indoor
                engine = self._indoor
        return engine

    def _get_vehicle_engine(self) -> _NativeTensorRTEngine:
        engine = self._vehicle
        if engine is None:
            with self._engine_init_lock:
                if self._vehicle is None:
                    started = time.monotonic()
                    vehicle = _NativeTensorRTEngine(
                        self._vehicle_engine_path,
                        expected_task="depth",
                    )
                    if vehicle.input_shape != (
                        1,
                        3,
                        _YOLO_DEPTH_INPUT_SIZE,
                        _YOLO_DEPTH_INPUT_SIZE,
                    ):
                        raise ObstacleDistanceError(
                            ErrorCode.MODEL_ERROR,
                            "YOLO depth TensorRT engine input shape is incompatible",
                        )
                    log.info(
                        "[obstacle] YOLO depth engine loaded in %.1fms",
                        1000.0 * (time.monotonic() - started),
                    )
                    self._vehicle = vehicle
                engine = self._vehicle
        return engine

    def close(self) -> None:
        """Release both depth engines and their CUDA buffers."""
        with self._engine_init_lock:
            engines = [self._indoor, self._vehicle]
            self._indoor = None
            self._vehicle = None
        for engine in engines:
            if engine is not None:
                engine.close()

    def predict_indoor_distance(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> float:
        check_deadline(deadline_monotonic)
        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        outputs = self._get_indoor_engine().infer(
            _prepare_zipdepth_image(image)
        )
        if len(outputs) != 1:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "ZipDepth TensorRT engine must have one output",
            )
        inverse_depth = np.asarray(outputs[0]).squeeze()
        if inverse_depth.ndim != 2:
            raise ObstacleDistanceError(
                ErrorCode.INVALID_DEPTH,
                "ZipDepth output must be a two-dimensional map",
            )
        inverse_depth = _resize_align_corners(inverse_depth, height, width)
        r0, r1, c0, c1 = self._scaled_roi(height, width)
        values = inverse_depth[r0:r1, c0:c1]
        values = values[np.isfinite(values)]
        if values.size < self._min_valid_pixels:
            raise ObstacleDistanceError(
                ErrorCode.NO_VALID_DEPTH,
                "ZipDepth ROI does not contain enough valid values",
            )
        inverse_depth_percentile = float(
            np.percentile(values, self._percentile)
        )
        score = -inverse_depth_percentile
        distance_m = (
            self._distance_scale * inverse_depth_percentile
            + self._distance_bias_m
        )
        distance_m = float(
            np.clip(
                distance_m,
                self._min_output_distance_m,
                self._max_output_distance_m,
            )
        )
        if score < self._score_threshold:
            distance_m = min(
                distance_m,
                self._decision_threshold_m - self._classification_margin_m,
            )
        else:
            distance_m = max(distance_m, self._decision_threshold_m)
        check_deadline(deadline_monotonic)
        return distance_m

    def predict_depth(
        self,
        image_bytes: bytes,
        domain: SceneDomain,
        deadline_monotonic: float,
    ) -> DepthPrediction:
        if domain is not SceneDomain.VEHICLE:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "ZipDepth indoor inference requires predict_indoor_distance",
            )
        check_deadline(deadline_monotonic)
        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        tensor, _, _, _ = _prepare_yolo_image(image, _YOLO_DEPTH_INPUT_SIZE)
        outputs = self._get_vehicle_engine().infer(tensor)
        if len(outputs) != 1:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "YOLO depth TensorRT engine must have one output",
            )
        depth = _scale_depth_to_original(outputs[0].squeeze(), height, width)
        check_deadline(deadline_monotonic)
        return DepthPrediction(
            depth_m=np.ascontiguousarray(depth, dtype=np.float32),
            source_height=height,
            source_width=width,
        )


def create_backends(
    config: Mapping,
) -> tuple[DepthBackend, InstanceSegmentationBackend]:
    indoor_engine = model_path(config, "indoor_depth_engine")
    vehicle_engine = model_path(config, "vehicle_depth_engine")
    segmentation_engine = model_path(config, "segmentation_engine")
    indoor_config = config.get("indoor", {})
    vehicle_config = config.get("vehicle", {})
    if not isinstance(indoor_config, Mapping):
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            "indoor configuration must be a mapping",
        )
    if not isinstance(vehicle_config, Mapping):
        vehicle_config = {}
    return (
        ZipDepthYoloTensorRTDepthBackend(
            indoor_engine,
            vehicle_engine,
            indoor_config=indoor_config,
            decision_threshold_m=config.get("decision_threshold_m", 1.0),
        ),
        NativeTensorRTSegBackend(
            segmentation_engine,
            allowed_classes=vehicle_config.get("allowed_classes"),
            min_confidence=vehicle_config.get("min_confidence"),
        ),
    )
