from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Protocol, Sequence, runtime_checkable


class SceneDomain(str, Enum):
    INDOOR = "indoor"
    VEHICLE = "vehicle"


class ErrorCode(str, Enum):
    INVALID_IMAGE = "invalid_image"
    MISSING_SCENE = "missing_scene"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    INVALID_DEPTH = "invalid_depth"
    NO_VALID_DEPTH = "no_valid_depth"
    NO_TARGET_INSTANCE = "no_target_instance"
    MISSING_CALIBRATION = "missing_calibration"
    INVALID_CALIBRATION = "invalid_calibration"


class ObstacleDistanceError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class DepthPrediction:
    depth_m: object
    source_height: int
    source_width: int
    uncertainty: object | None = None

    def __post_init__(self) -> None:
        if self.depth_m is None:
            raise ValueError("depth_m must not be None")
        if (
            type(self.source_height) is not int
            or self.source_height <= 0
            or type(self.source_width) is not int
            or self.source_width <= 0
        ):
            raise ValueError("source dimensions must be positive integers")


@dataclass(frozen=True)
class InstanceMask:
    class_name: str
    confidence: float
    mask: object

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be nonempty")
        if (
            not _is_finite_number(self.confidence)
            or self.confidence < 0
            or self.confidence > 1
        ):
            raise ValueError("confidence must be finite and between 0 and 1")
        if self.mask is None:
            raise ValueError("mask must not be None")


def _invalid_calibration(message: str) -> ObstacleDistanceError:
    return ObstacleDistanceError(ErrorCode.INVALID_CALIBRATION, message)


@dataclass(frozen=True)
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_ego: tuple[float, ...]
    bumper_xy: tuple[float, float]

    def __post_init__(self) -> None:
        if (
            not _is_finite_number(self.fx)
            or self.fx <= 0
            or not _is_finite_number(self.fy)
            or self.fy <= 0
        ):
            raise _invalid_calibration(
                "fx and fy must be finite positive numbers"
            )
        if not _is_finite_number(self.cx) or not _is_finite_number(self.cy):
            raise _invalid_calibration("cx and cy must be finite numbers")
        if (
            not isinstance(self.camera_to_ego, tuple)
            or len(self.camera_to_ego) != 16
            or not all(_is_finite_number(value) for value in self.camera_to_ego)
        ):
            raise _invalid_calibration(
                "camera_to_ego must contain 16 finite numbers"
            )
        if (
            not isinstance(self.bumper_xy, tuple)
            or len(self.bumper_xy) != 2
            or not all(_is_finite_number(value) for value in self.bumper_xy)
        ):
            raise _invalid_calibration(
                "bumper_xy must contain 2 finite numbers"
            )


@runtime_checkable
class DepthBackend(Protocol):
    def predict_depth(
        self,
        image_bytes: bytes,
        domain: SceneDomain,
        deadline_monotonic: float,
    ) -> DepthPrediction:
        ...


@runtime_checkable
class IndoorDistanceBackend(Protocol):
    """Direct indoor decision for non-metric depth models.

    Implementations return the protocol-facing distance directly instead of
    pretending that an affine-invariant depth map is measured in metres.
    """

    def predict_indoor_distance(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> float:
        ...


@runtime_checkable
class InstanceSegmentationBackend(Protocol):
    def predict_instances(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> Sequence[InstanceMask]:
        ...
