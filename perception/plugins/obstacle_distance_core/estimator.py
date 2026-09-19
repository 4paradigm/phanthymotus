from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Mapping

from .contracts import ErrorCode, IndoorDistanceBackend, ObstacleDistanceError


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


class ObstacleDistanceEstimator:
    def __init__(
        self,
        depth_backend: IndoorDistanceBackend,
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
            self._config.get("decision_threshold_m", 2.0),
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

        if not callable(getattr(depth_backend, "predict_indoor_distance", None)):
            raise ValueError("estimator requires an indoor distance backend")
        self._depth_backend = depth_backend

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
        timestamp: float | None = None,
    ) -> DistanceResult:
        started_monotonic = None
        result_timestamp = 0.0
        scene = "indoor"
        try:
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise ObstacleDistanceError(
                    ErrorCode.INVALID_IMAGE,
                    "image bytes must be nonempty",
                )
            started_monotonic = _time_value(self._monotonic())
            result_timestamp = _time_value(
                timestamp if timestamp is not None else self._wall_time()
            )
            deadline_monotonic = started_monotonic + self._soft_timeout_s

            self._check_deadline(deadline_monotonic)
            distance = self._depth_backend.predict_indoor_distance(
                image_bytes, deadline_monotonic
            )
            distance = self._validated_distance(distance)
            finished_monotonic = self._check_deadline(deadline_monotonic)
            return self._result(
                distance_m=distance,
                scene=scene,
                status="ok",
                error_code=None,
                fallback=False,
                approximate_geometry=False,
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
