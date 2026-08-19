from __future__ import annotations

import math
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Mapping

from .backend_validation import (
    is_valid_depth_backend,
    is_valid_indoor_distance_backend,
    is_valid_segmentation_backend,
)
from .contracts import (
    CameraCalibration,
    DepthBackend,
    ErrorCode,
    InstanceMask,
    InstanceSegmentationBackend,
    ObstacleDistanceError,
    SceneDomain,
)
from .geometry import approximate_vehicle_distance_m, vehicle_distance_m
from .routing import resolve_scene


@dataclass(frozen=True)
class DistanceResult:
    distance_m: float
    near_obstacle: bool
    decision_threshold_m: float
    scene: str
    status: str
    error_code: str | None
    fallback: bool
    approximate_geometry: bool
    latency_ms: float
    timestamp: float


def _finite_number(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if positive and converted <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and converted < 0:
        raise ValueError(f"{name} must be nonnegative")
    return converted


def _invalid_calibration(message: str) -> ObstacleDistanceError:
    return ObstacleDistanceError(ErrorCode.INVALID_CALIBRATION, message)


def _time_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            "time source returned an invalid value",
        )
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            "time source returned an invalid value",
        ) from None
    if not math.isfinite(converted):
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            "time source returned an invalid value",
        )
    return converted


def _camera_calibration(value: object) -> CameraCalibration:
    if isinstance(value, CameraCalibration):
        return value
    if not isinstance(value, Mapping):
        raise _invalid_calibration("camera calibration is invalid")
    try:
        return CameraCalibration(
            fx=value["fx"],
            fy=value["fy"],
            cx=value["cx"],
            cy=value["cy"],
            camera_to_ego=tuple(value["camera_to_ego"]),
            bumper_xy=tuple(value["bumper_xy"]),
        )
    except ObstacleDistanceError:
        raise
    except Exception:
        raise _invalid_calibration("camera calibration is invalid") from None


class ObstacleDistanceEstimator:
    def __init__(
        self,
        depth_backend: DepthBackend | None,
        segmentation_backend: InstanceSegmentationBackend | None,
        config: Mapping[str, object],
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, Mapping):
            raise ValueError("estimator config must be a mapping")
        self._config = deepcopy(dict(config))
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._decision_threshold_m = _finite_number(
            self._config.get("decision_threshold_m", 1.0),
            name="decision_threshold_m",
            positive=True,
        )
        self._fallback_distance_m = _finite_number(
            self._config.get("fallback_distance_m", 3.0),
            name="fallback_distance_m",
            nonnegative=True,
        )
        self._soft_timeout_s = _finite_number(
            self._config.get("soft_timeout_s", 2.5),
            name="soft_timeout_s",
            positive=True,
        )

        if not is_valid_depth_backend(depth_backend):
            raise ValueError("estimator requires a depth backend")
        if not is_valid_segmentation_backend(segmentation_backend):
            raise ValueError("estimator requires a segmentation backend")

        self._depth_backend = depth_backend
        self._segmentation_backend = segmentation_backend

    def _latency_ms(
        self,
        started_monotonic: float | None,
        *,
        finished_monotonic: float | None,
        suppress_clock_errors: bool,
    ) -> float:
        if started_monotonic is None:
            return 0.0
        if finished_monotonic is None:
            try:
                finished_monotonic = _time_value(self._monotonic())
            except Exception:
                if suppress_clock_errors:
                    return 0.0
                raise ObstacleDistanceError(
                    ErrorCode.MODEL_ERROR,
                    "time source failed",
                ) from None
        elapsed = (finished_monotonic - started_monotonic) * 1000.0
        if elapsed < 0:
            return 0.0
        return elapsed

    def _result(
        self,
        *,
        distance_m: float,
        scene: str,
        status: str,
        error_code: str | None,
        fallback: bool,
        approximate_geometry: bool,
        started_monotonic: float | None,
        timestamp: float,
        finished_monotonic: float | None = None,
    ) -> DistanceResult:
        return DistanceResult(
            distance_m=distance_m,
            near_obstacle=distance_m < self._decision_threshold_m,
            decision_threshold_m=self._decision_threshold_m,
            scene=scene,
            status=status,
            error_code=error_code,
            fallback=fallback,
            approximate_geometry=approximate_geometry,
            latency_ms=self._latency_ms(
                started_monotonic,
                finished_monotonic=finished_monotonic,
                suppress_clock_errors=fallback,
            ),
            timestamp=timestamp,
        )

    def _fallback(
        self,
        *,
        code: ErrorCode,
        scene: str,
        started_monotonic: float | None,
        timestamp: float,
    ) -> DistanceResult:
        safe_code = code if isinstance(code, ErrorCode) else ErrorCode.MODEL_ERROR
        try:
            safe_timestamp = _time_value(timestamp)
        except Exception:
            safe_timestamp = 0.0
        return self._result(
            distance_m=self._fallback_distance_m,
            scene=scene,
            status="error",
            error_code=safe_code.value,
            fallback=True,
            approximate_geometry=False,
            started_monotonic=started_monotonic,
            timestamp=safe_timestamp,
        )

    def _check_deadline(self, deadline_monotonic: float) -> float:
        observed_monotonic = _time_value(self._monotonic())
        if observed_monotonic >= deadline_monotonic:
            raise ObstacleDistanceError(
                ErrorCode.TIMEOUT,
                "obstacle distance inference timed out",
            )
        return observed_monotonic

    @staticmethod
    def _validated_instances(value: object) -> tuple[InstanceMask, ...]:
        if (
            isinstance(value, (str, bytes, bytearray, memoryview, Mapping))
            or not isinstance(value, Sequence)
        ):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "segmentation output is invalid",
            )
        try:
            instances = tuple(value[index] for index in range(len(value)))
        except Exception:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "segmentation output is invalid",
            ) from None
        if any(not isinstance(instance, InstanceMask) for instance in instances):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "segmentation output is invalid",
            )
        return instances

    @staticmethod
    def _apply_distance_calibration(
        distance: float,
        section: Mapping,
    ) -> float:
        calibration = section.get("distance_calibration")
        if calibration is not None and not isinstance(calibration, Mapping):
            raise ObstacleDistanceError(
                ErrorCode.INVALID_DEPTH,
                "distance_calibration must be a mapping",
            )
        if isinstance(calibration, Mapping) and calibration:
            try:
                mode = calibration.get("mode", "affine")
                minimum_value = calibration.get("min_output_m")
                maximum_value = calibration.get("max_output_m")
                minimum = (
                    None if minimum_value is None else float(minimum_value)
                )
                maximum = (
                    None if maximum_value is None else float(maximum_value)
                )
            except (TypeError, ValueError, OverflowError):
                raise ObstacleDistanceError(
                    ErrorCode.INVALID_DEPTH,
                    "distance_calibration must contain numeric values",
                ) from None
            if not isinstance(mode, str) or mode not in {"affine", "power"}:
                raise ObstacleDistanceError(
                    ErrorCode.INVALID_DEPTH,
                    "distance_calibration mode is invalid",
                )
            if (
                (
                    minimum is not None
                    and (not math.isfinite(minimum) or minimum < 0)
                )
                or (
                    maximum is not None
                    and (not math.isfinite(maximum) or maximum < 0)
                )
                or (
                    minimum is not None
                    and maximum is not None
                    and maximum < minimum
                )
            ):
                raise ObstacleDistanceError(
                    ErrorCode.INVALID_DEPTH,
                    "distance_calibration values or output range are invalid",
                )
            try:
                if mode == "affine":
                    scale = float(calibration.get("scale", 1.0))
                    bias = float(calibration.get("bias", 0.0))
                    if (
                        not math.isfinite(scale)
                        or scale <= 0
                        or not math.isfinite(bias)
                    ):
                        raise ValueError
                    distance = scale * distance + bias
                else:
                    input_anchor = float(calibration["input_anchor_m"])
                    output_anchor = float(calibration["output_anchor_m"])
                    power = float(calibration["power"])
                    if (
                        not math.isfinite(distance)
                        or distance < 0
                        or not math.isfinite(input_anchor)
                        or input_anchor <= 0
                        or not math.isfinite(output_anchor)
                        or output_anchor <= 0
                        or not math.isfinite(power)
                        or power <= 0
                    ):
                        raise ValueError
                    distance = output_anchor * math.pow(
                        distance / input_anchor,
                        power,
                    )
                if not math.isfinite(distance):
                    raise ValueError
            except (KeyError, TypeError, ValueError, OverflowError):
                raise ObstacleDistanceError(
                    ErrorCode.INVALID_DEPTH,
                    "distance_calibration values are invalid",
                ) from None
            if minimum is not None:
                distance = max(distance, minimum)
            if maximum is not None:
                distance = min(distance, maximum)
        return distance

    def _vehicle_distance(self, prediction, instances) -> tuple[float, bool]:
        config = self._config.get("vehicle")
        if not isinstance(config, Mapping):
            raise ObstacleDistanceError(
                ErrorCode.INVALID_DEPTH,
                "vehicle configuration is invalid",
            )
        allowed_classes = config.get("allowed_classes")
        try:
            allowed = set(allowed_classes)
        except Exception:
            allowed = allowed_classes
        common = {
            "instances": instances,
            "allowed_classes": allowed,
            "min_confidence": config.get("min_confidence"),
            "percentile": config.get("percentile"),
            "min_depth_m": config.get("min_depth_m"),
            "max_depth_m": config.get("max_depth_m"),
        }

        calibration_value = config.get("calibration")
        calibration_missing = calibration_value is None or (
            isinstance(calibration_value, Mapping)
            and len(calibration_value) == 0
        )
        if calibration_missing:
            if config.get("allow_approximate_geometry") is not True:
                raise ObstacleDistanceError(
                    ErrorCode.MISSING_CALIBRATION,
                    "camera calibration is required for vehicle geometry",
                )
            distance = approximate_vehicle_distance_m(
                prediction.depth_m,
                camera_to_bumper_offset_m=config.get(
                    "camera_to_bumper_offset_m"
                ),
                **common,
            )
            return self._apply_distance_calibration(distance, config), True

        calibration = _camera_calibration(calibration_value)
        distance = vehicle_distance_m(
            prediction.depth_m,
            calibration=calibration,
            **common,
        )
        return self._apply_distance_calibration(distance, config), False

    @staticmethod
    def _validated_distance(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "estimated distance is invalid",
            )
        try:
            distance = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "estimated distance is invalid",
            ) from None
        if not math.isfinite(distance) or distance < 0:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "estimated distance is invalid",
            )
        return distance

    def estimate(
        self,
        image_bytes: bytes,
        scene_hint: SceneDomain | str | None = None,
        timestamp: float | None = None,
    ) -> DistanceResult:
        started_monotonic = None
        result_timestamp = 0.0
        scene = "unknown"
        try:
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise ObstacleDistanceError(
                    ErrorCode.INVALID_IMAGE,
                    "image bytes must be nonempty",
                )
            domain = resolve_scene(
                scene_hint=scene_hint,
                fixed_scene=self._config.get("fixed_scene"),
            )
            scene = domain.value
            started_monotonic = _time_value(self._monotonic())
            result_timestamp = _time_value(
                timestamp if timestamp is not None else self._wall_time()
            )
            deadline_monotonic = started_monotonic + self._soft_timeout_s

            approximate_geometry = False
            self._check_deadline(deadline_monotonic)
            if domain is SceneDomain.INDOOR:
                if not is_valid_indoor_distance_backend(self._depth_backend):
                    raise ObstacleDistanceError(
                        ErrorCode.MODEL_ERROR,
                        "indoor inference requires a direct distance backend",
                    )
                distance = self._depth_backend.predict_indoor_distance(
                    image_bytes,
                    deadline_monotonic,
                )
                self._check_deadline(deadline_monotonic)
            else:
                prediction = self._depth_backend.predict_depth(
                    image_bytes,
                    domain,
                    deadline_monotonic,
                )
                self._check_deadline(deadline_monotonic)

            if domain is SceneDomain.VEHICLE:
                self._check_deadline(deadline_monotonic)
                instances = self._segmentation_backend.predict_instances(
                    image_bytes,
                    deadline_monotonic,
                )
                self._check_deadline(deadline_monotonic)
                instances = self._validated_instances(instances)
                distance, approximate_geometry = self._vehicle_distance(
                    prediction,
                    instances,
                )
            distance = self._validated_distance(distance)
            finished_monotonic = self._check_deadline(deadline_monotonic)
            return self._result(
                distance_m=distance,
                scene=scene,
                status="ok",
                error_code=None,
                fallback=False,
                approximate_geometry=approximate_geometry,
                started_monotonic=started_monotonic,
                timestamp=result_timestamp,
                finished_monotonic=finished_monotonic,
            )
        except ObstacleDistanceError as error:
            return self._fallback(
                code=error.code,
                scene=scene,
                started_monotonic=started_monotonic,
                timestamp=result_timestamp,
            )
        except Exception:
            return self._fallback(
                code=ErrorCode.MODEL_ERROR,
                scene=scene,
                started_monotonic=started_monotonic,
                timestamp=result_timestamp,
            )
