"""Pure conversion logic for G1 native locomotion state to planar odometry."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from .timestamp_contract import (
    DEFAULT_MAX_FUTURE_SKEW_NS,
    DEFAULT_MAX_SOURCE_AGE_NS,
    InvalidSourceTimestamp,
    validate_source_timestamp_ns,
)


class InvalidLocoState(ValueError):
    """Raised when a native G1 locomotion state cannot form odometry."""


@dataclass(frozen=True)
class PlanarOdometry:
    stamp: float
    source_stamp_ns: int
    source_frame: str
    timestamp_source: str
    frame_source: str
    source_schema: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _finite_vector(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < size:
        raise InvalidLocoState(f"{name} must contain at least {size} values")
    result = [float(item) for item in value[:size]]
    if not all(math.isfinite(item) for item in result):
        raise InvalidLocoState(f"{name} contains non-finite values")
    return result


def decode_state(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise InvalidLocoState("state is not valid JSON") from exc
    elif isinstance(payload, dict):
        value = payload
    else:
        raise InvalidLocoState("state must be JSON text or a mapping")
    if not isinstance(value, dict):
        raise InvalidLocoState("state root must be an object")
    return value


def _timestamp_and_frame(
    state: dict[str, Any],
    receive_stamp_ns: int | None,
    *,
    max_source_age_ns: int,
    max_future_skew_ns: int,
) -> tuple[int, str, str, str, str]:
    schema_version = state.get("schema_version")
    schema = state.get("schema")
    if schema_version is not None and (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 2
    ):
        raise InvalidLocoState(f"unsupported schema_version: {schema_version!r}")
    if schema not in (
        None,
        "unitree.g1.loco_state.legacy",
        "phanthy.g1.loco_state.v2",
    ):
        raise InvalidLocoState(f"unsupported schema: {schema!r}")
    if schema_version == 2 and schema == "unitree.g1.loco_state.legacy":
        raise InvalidLocoState("schema conflicts with schema_version=2")
    is_v2 = schema_version == 2 or schema == "phanthy.g1.loco_state.v2"
    if is_v2:
        timestamp_source = state.get("timestamp_source")
        if timestamp_source != "driver_receive":
            raise InvalidLocoState(
                "timestamp_source must be 'driver_receive' for loco_state.v2"
            )
        source_stamp_ns = state.get("source_stamp_ns")
        if isinstance(source_stamp_ns, bool) or not isinstance(source_stamp_ns, int):
            raise InvalidLocoState("source_stamp_ns must be an integer")
        try:
            validate_source_timestamp_ns(
                source_stamp_ns,
                receive_stamp_ns,
                max_source_age_ns=max_source_age_ns,
                max_future_skew_ns=max_future_skew_ns,
            )
        except InvalidSourceTimestamp as exc:
            raise InvalidLocoState(str(exc)) from exc
        source_frame = state.get("frame_id")
        if source_frame != "odom_source":
            raise InvalidLocoState("frame_id must be 'odom_source'")
        return (
            source_stamp_ns,
            source_frame,
            timestamp_source,
            "driver_payload",
            "phanthy.g1.loco_state.v2",
        )

    if isinstance(receive_stamp_ns, bool) or not isinstance(receive_stamp_ns, int):
        raise InvalidLocoState(
            "legacy loco_state requires an adapter receive timestamp"
        )
    if receive_stamp_ns <= 0:
        raise InvalidLocoState("receive_stamp_ns must be positive")
    return (
        receive_stamp_ns,
        "odom_source",
        "adapter_receive",
        "adapter_contract",
        "unitree.g1.loco_state.legacy",
    )


class OriginNormalizer:
    """Normalizes the native boot-relative pose to a card-local odom origin."""

    def __init__(
        self,
        reset_origin: bool = True,
        velocity_frame: str = "body",
        *,
        max_source_age_ns: int = DEFAULT_MAX_SOURCE_AGE_NS,
        max_future_skew_ns: int = DEFAULT_MAX_FUTURE_SKEW_NS,
    ):
        if velocity_frame not in {"body", "odom"}:
            raise ValueError("velocity_frame must be 'body' or 'odom'")
        if max_source_age_ns <= 0 or max_future_skew_ns < 0:
            raise ValueError("invalid source timestamp tolerances")
        self.reset_origin = reset_origin
        self.velocity_frame = velocity_frame
        self.max_source_age_ns = max_source_age_ns
        self.max_future_skew_ns = max_future_skew_ns
        self._origin: tuple[float, float, float] | None = None
        self._last_driver_source_stamp_ns: int | None = None

    @property
    def initialized(self) -> bool:
        return self._origin is not None

    def reset(self) -> None:
        self._origin = None
        self._last_driver_source_stamp_ns = None

    def convert(
        self,
        payload: str | dict[str, Any],
        *,
        receive_stamp_ns: int | None = None,
    ) -> PlanarOdometry:
        state = decode_state(payload)
        (
            source_stamp_ns,
            source_frame,
            timestamp_source,
            frame_source,
            source_schema,
        ) = _timestamp_and_frame(
            state,
            receive_stamp_ns,
            max_source_age_ns=self.max_source_age_ns,
            max_future_skew_ns=self.max_future_skew_ns,
        )
        if (
            timestamp_source == "driver_receive"
            and self._last_driver_source_stamp_ns is not None
            and source_stamp_ns < self._last_driver_source_stamp_ns
        ):
            raise InvalidLocoState("source_stamp_ns moved backwards")
        stamp = source_stamp_ns / 1_000_000_000.0
        position = _finite_vector(state.get("position"), 2, "position")
        velocity = _finite_vector(state.get("velocity"), 2, "velocity")
        imu = state.get("imu")
        if not isinstance(imu, dict):
            raise InvalidLocoState("imu must be an object")
        rpy = _finite_vector(imu.get("rpy"), 3, "imu.rpy")
        yaw_speed = float(state.get("yaw_speed", 0.0))
        if not math.isfinite(yaw_speed):
            raise InvalidLocoState("yaw_speed must be finite")

        raw_x, raw_y = position
        raw_yaw = wrap_angle(rpy[2])
        if self._origin is None:
            self._origin = (
                raw_x if self.reset_origin else 0.0,
                raw_y if self.reset_origin else 0.0,
                raw_yaw if self.reset_origin else 0.0,
            )

        origin_x, origin_y, origin_yaw = self._origin
        dx = raw_x - origin_x
        dy = raw_y - origin_y
        cos_origin = math.cos(origin_yaw)
        sin_origin = math.sin(origin_yaw)
        x = cos_origin * dx + sin_origin * dy
        y = -sin_origin * dx + cos_origin * dy
        yaw = wrap_angle(raw_yaw - origin_yaw)

        vx, vy = velocity
        if self.velocity_frame == "odom":
            cos_yaw = math.cos(raw_yaw)
            sin_yaw = math.sin(raw_yaw)
            vx, vy = (
                cos_yaw * velocity[0] + sin_yaw * velocity[1],
                -sin_yaw * velocity[0] + cos_yaw * velocity[1],
            )

        odometry = PlanarOdometry(
            stamp=stamp,
            source_stamp_ns=source_stamp_ns,
            source_frame=source_frame,
            timestamp_source=timestamp_source,
            frame_source=frame_source,
            source_schema=source_schema,
            x=x,
            y=y,
            yaw=yaw,
            vx=vx,
            vy=vy,
            wz=yaw_speed,
        )
        if timestamp_source == "driver_receive":
            self._last_driver_source_stamp_ns = source_stamp_ns
        return odometry
