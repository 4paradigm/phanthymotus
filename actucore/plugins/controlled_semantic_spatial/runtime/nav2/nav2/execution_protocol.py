"""ROS-free motion limits and shadow-command ownership checks."""

from __future__ import annotations

from dataclasses import dataclass
import math


MIN_EFFECTIVE_LINEAR_MPS = 0.30
MIN_EFFECTIVE_YAW_RADPS = 1.00
TURN_ONLY_YAW_THRESHOLD_RADPS = 0.20

_SHADOW_PRODUCING_STATUSES = {"starting", "navigating"}


class ProtocolError(ValueError):
    """Malformed or unsafe data-plane proposal."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Velocity:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @classmethod
    def zero(cls) -> "Velocity":
        return cls()

    def is_zero(self, *, tolerance: float = 1e-9) -> bool:
        return (
            abs(self.x) <= tolerance
            and abs(self.y) <= tolerance
            and abs(self.yaw) <= tolerance
        )

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


def shadow_context_is_current(
    active: dict | None,
    *,
    nav_id: str,
    attempt: int,
    status: str,
) -> bool:
    """Return whether an asynchronously shaped command still owns the goal."""

    return bool(
        active is not None
        and active.get("nav_id") == nav_id
        and active.get("attempt") == attempt
        and active.get("status") == status
    )


def shadow_context_is_publishable(
    active: dict | None,
    *,
    nav_id: str,
    attempt: int,
    status: str,
) -> bool:
    """Return whether a current goal still owns periodic motion output."""

    return status in _SHADOW_PRODUCING_STATUSES and shadow_context_is_current(
        active,
        nav_id=nav_id,
        attempt=attempt,
        status=status,
    )


@dataclass(frozen=True)
class MotionLimits:
    """Per-navigation motion deadbands and hard axis caps."""

    min_x_mps: float = MIN_EFFECTIVE_LINEAR_MPS
    max_x_mps: float = 1.0
    min_y_mps: float = 0.0
    max_y_mps: float = 0.0
    min_yaw_rps: float = MIN_EFFECTIVE_YAW_RADPS
    max_yaw_rps: float = 2.0

    @classmethod
    def from_payload(cls, payload) -> "MotionLimits":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ProtocolError(
                "invalid_velocity_limits", "velocity_limits must be an object"
            )
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unknown:
                details.append("unknown=" + ",".join(unknown))
            raise ProtocolError(
                "invalid_velocity_limits", "; ".join(details)
            )
        limits = cls(
            **{
                field: _finite_number(payload[field], f"velocity_limits.{field}")
                for field in expected
            }
        )
        limits.validate()
        return limits

    def validate(self) -> None:
        bounds = {
            "x": (self.min_x_mps, self.max_x_mps, 1.0),
            "y": (self.min_y_mps, self.max_y_mps, 1.0),
            "yaw": (self.min_yaw_rps, self.max_yaw_rps, 2.0),
        }
        for axis, (minimum, maximum, contract_maximum) in bounds.items():
            if not 0.0 <= minimum <= maximum <= contract_maximum:
                raise ProtocolError(
                    "invalid_velocity_limits",
                    f"{axis} limits must satisfy 0 <= min <= max <= "
                    f"{contract_maximum}",
                )

    def as_dict(self) -> dict:
        return {
            field: float(getattr(self, field))
            for field in self.__dataclass_fields__
        }


DEFAULT_MOTION_LIMITS = MotionLimits()


def limit_forward_velocity(
    velocity: Velocity, *, max_forward_mps: float
) -> Velocity:
    """Apply the forward-only per-navigation cap before publication.

    Negative X is suppressed at the output boundary so no Nav2 recovery,
    controller or smoother configuration can command reverse motion. Lateral
    and yaw components are preserved.
    """

    if isinstance(max_forward_mps, bool):
        raise ProtocolError(
            "invalid_speed_limit", "max_forward_mps must be positive and finite"
        )
    try:
        limit = float(max_forward_mps)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            "invalid_speed_limit", "max_forward_mps must be positive and finite"
        ) from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise ProtocolError(
            "invalid_speed_limit", "max_forward_mps must be positive and finite"
        )
    forward = min(max(float(velocity.x), 0.0), limit)
    return Velocity(x=forward, y=velocity.y, yaw=velocity.yaw)


def _apply_axis_deadband(value: float, minimum: float, maximum: float) -> float:
    threshold = min(minimum, maximum)
    if value == 0.0 or maximum == 0.0 or abs(value) < threshold:
        return 0.0
    magnitude = min(abs(value), maximum)
    return math.copysign(magnitude, value)


def apply_motion_limits(
    velocity: Velocity,
    *,
    limits: MotionLimits,
    max_forward_mps: float,
) -> Velocity:
    """Make translation/rotation exclusive and apply deadbands and caps.

    Exact zeros remain zeros so readiness, pause, terminal and watchdog stops
    preserve their fail-closed semantics. Values below a configured minimum are
    suppressed instead of being amplified, so Nav2 deceleration remains a
    deceleration. A mixed command turns in place only when its retained yaw
    demand is significant; otherwise it moves without yaw. Lateral and yaw
    signs are preserved, while negative X is suppressed. The navigation
    request's forward cap always wins over the configured X deadband.
    """

    limits.validate()
    limited = limit_forward_velocity(
        velocity,
        max_forward_mps=max_forward_mps,
    )
    x_max = limits.max_x_mps
    if limited.x > 0.0:
        x_max = min(x_max, float(max_forward_mps))
    x = _apply_axis_deadband(limited.x, limits.min_x_mps, x_max)
    y = _apply_axis_deadband(
        limited.y,
        limits.min_y_mps,
        limits.max_y_mps,
    )
    yaw = _apply_axis_deadband(
        limited.yaw,
        limits.min_yaw_rps,
        limits.max_yaw_rps,
    )
    if (x != 0.0 or y != 0.0) and yaw != 0.0:
        if abs(yaw) >= TURN_ONLY_YAW_THRESHOLD_RADPS:
            x = 0.0
            y = 0.0
        else:
            yaw = 0.0
    return Velocity(x=x, y=y, yaw=yaw)


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError("invalid_number", f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            "invalid_number", f"{field} must be a finite number"
        ) from exc
    if not math.isfinite(number):
        raise ProtocolError("invalid_number", f"{field} must be a finite number")
    return number


__all__ = [
    "DEFAULT_MOTION_LIMITS", "MIN_EFFECTIVE_LINEAR_MPS", "MIN_EFFECTIVE_YAW_RADPS",
    "MotionLimits", "TURN_ONLY_YAW_THRESHOLD_RADPS", "ProtocolError", "Velocity",
    "apply_motion_limits", "limit_forward_velocity", "shadow_context_is_current",
    "shadow_context_is_publishable",
]
