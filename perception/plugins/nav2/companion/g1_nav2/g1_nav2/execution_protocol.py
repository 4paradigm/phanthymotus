"""Structured, bounded velocity proposals emitted by the Nav2 companion.

This module has no ROS or robot SDK dependency.  It only defines the public
proposal envelope used by Nav2; trusted ownership, execution and stop
confirmation remain Driver responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time


SCHEMA_VERSION = 1
VELOCITY_PROPOSAL_SCHEMA = "phanthy.navigation.velocity_proposal.v1"
VELOCITY_PROPOSAL_TOPIC = "/ubuntu/navigation/nav2/velocity_proposal"
MIN_EFFECTIVE_LINEAR_MPS = 0.30
MIN_EFFECTIVE_YAW_RADPS = 1.00
TURN_ONLY_YAW_THRESHOLD_RADPS = 0.20
TERMINAL_XY_TOLERANCE_M = 0.18
TERMINAL_YAW_TOLERANCE_RAD = 0.45

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MOTION_STATUSES = {"planning", "navigating", "replanning", "running", "active"}
_IDLE_STATUSES = {"paused"}
_TERMINAL_STATUSES = {
    "arrived",
    "cancelled",
    "stopped",
    "error",
    "aborted",
    "rejected",
}


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


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


def _normalized_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def shape_terminal_approach(
    velocity: Velocity,
    *,
    current_pose: Pose2D,
    target_pose: Pose2D,
    xy_tolerance_m: float = TERMINAL_XY_TOLERANCE_M,
    yaw_tolerance_rad: float = TERMINAL_YAW_TOLERANCE_RAD,
) -> tuple[Velocity, str]:
    """Prevent configured motion floors from amplifying terminal corrections.

    This function is deliberately stateless: every proposal is evaluated from
    the current pose and target, so reaching one goal cannot lock a subsequent
    navigation.  Once position is within the inner tolerance, translation is
    suppressed while Nav2 finishes the requested heading.  Once both errors
    are within tolerance, the proposal is forced to exact zero and Nav2 remains
    responsible for reporting the action result.
    """

    values = (
        current_pose.x,
        current_pose.y,
        current_pose.yaw,
        target_pose.x,
        target_pose.y,
        target_pose.yaw,
        xy_tolerance_m,
        yaw_tolerance_rad,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ProtocolError(
            "non_finite_pose", "terminal pose and tolerances must be finite"
        )
    if xy_tolerance_m <= 0.0 or yaw_tolerance_rad <= 0.0:
        raise ProtocolError(
            "invalid_goal_tolerance", "terminal tolerances must be positive"
        )

    xy_error = math.hypot(
        target_pose.x - current_pose.x,
        target_pose.y - current_pose.y,
    )
    if xy_error > xy_tolerance_m:
        return velocity, "approach"

    yaw_error = abs(_normalized_angle(target_pose.yaw - current_pose.yaw))
    if yaw_error > yaw_tolerance_rad:
        return Velocity(yaw=velocity.yaw), "rotate"
    return Velocity.zero(), "reached"


def proposal_context_is_current(
    active: dict | None,
    *,
    nav_id: str,
    attempt: int,
    status: str,
) -> bool:
    """Return whether an asynchronously shaped proposal still owns the goal."""

    return bool(
        active is not None
        and active.get("nav_id") == nav_id
        and active.get("attempt") == attempt
        and active.get("status") == status
    )


@dataclass(frozen=True)
class MotionLimits:
    """Per-navigation nonzero magnitude floors and hard axis caps."""

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
    """Apply the per-navigation forward cap before a proposal is published.

    Reverse recovery remains governed by the separate global reverse limit;
    lateral and yaw components are preserved so the normal proposal validator
    can reject any unsafe Nav2 output instead of silently hiding it.
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
    if velocity.x <= limit:
        return velocity
    return Velocity(x=limit, y=velocity.y, yaw=velocity.yaw)


def _apply_axis_magnitude(value: float, minimum: float, maximum: float) -> float:
    if value == 0.0 or maximum == 0.0:
        return 0.0
    magnitude = min(max(abs(value), min(minimum, maximum)), maximum)
    return math.copysign(magnitude, value)


def apply_g1_motion_limits(
    velocity: Velocity,
    *,
    limits: MotionLimits,
    max_forward_mps: float,
) -> Velocity:
    """Make translation/rotation exclusive and enforce configured magnitudes.

    Exact zeros remain zeros so readiness, pause, terminal and watchdog stops
    preserve their fail-closed semantics.  A mixed Nav2 command turns in place
    when its raw yaw demand is significant; otherwise it moves without yaw.
    Axis signs are preserved.  The navigation request's forward cap always
    wins over the configured X floor.
    """

    limits.validate()
    limited = limit_forward_velocity(
        velocity,
        max_forward_mps=max_forward_mps,
    )
    x = limited.x
    y = limited.y
    yaw = limited.yaw
    if (x != 0.0 or y != 0.0) and yaw != 0.0:
        if abs(yaw) >= TURN_ONLY_YAW_THRESHOLD_RADPS:
            x = 0.0
            y = 0.0
        else:
            yaw = 0.0

    x_max = limits.max_x_mps
    if x > 0.0:
        x_max = min(x_max, float(max_forward_mps))
    return Velocity(
        x=_apply_axis_magnitude(x, limits.min_x_mps, x_max),
        y=_apply_axis_magnitude(y, limits.min_y_mps, limits.max_y_mps),
        yaw=_apply_axis_magnitude(
            yaw,
            limits.min_yaw_rps,
            limits.max_yaw_rps,
        ),
    )


def apply_g1_motion_floor(velocity: Velocity) -> Velocity:
    """Compatibility wrapper for the default card motion limits."""

    return apply_g1_motion_limits(
        velocity,
        limits=DEFAULT_MOTION_LIMITS,
        max_forward_mps=DEFAULT_MOTION_LIMITS.max_x_mps,
    )


@dataclass(frozen=True)
class VelocityLimits:
    min_x: float = -1.0
    max_x: float = 1.0
    max_abs_y: float = 1.0
    max_abs_yaw: float = 2.0
    max_planar_speed: float = math.sqrt(2.0)

    def validate(self, velocity: Velocity) -> None:
        values = (velocity.x, velocity.y, velocity.yaw)
        if any(not math.isfinite(value) for value in values):
            raise ProtocolError("non_finite_velocity", "velocity must be finite")
        if not self.min_x <= velocity.x <= self.max_x:
            raise ProtocolError(
                "velocity_limit",
                f"x must be within [{self.min_x}, {self.max_x}] m/s",
            )
        if abs(velocity.y) > self.max_abs_y:
            raise ProtocolError(
                "velocity_limit",
                f"abs(y) must not exceed {self.max_abs_y} m/s",
            )
        if abs(velocity.yaw) > self.max_abs_yaw:
            raise ProtocolError(
                "velocity_limit",
                f"abs(yaw) must not exceed {self.max_abs_yaw} rad/s",
            )
        if math.hypot(velocity.x, velocity.y) > self.max_planar_speed:
            raise ProtocolError(
                "velocity_limit",
                f"planar speed must not exceed {self.max_planar_speed} m/s",
            )


DEFAULT_VELOCITY_LIMITS = VelocityLimits()


def _identifier(value, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ProtocolError(
            "invalid_identifier",
            f"{field} must match {_IDENTIFIER.pattern}",
        )
    return value


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError("invalid_integer", f"{field} must be a positive integer")
    return value


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


@dataclass(frozen=True)
class VelocityProposal:
    nav_id: str
    sequence: int
    ttl_ms: int
    issued_at_unix_ms: int
    navigation_status: str
    velocity: Velocity
    reason: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        *,
        limits: VelocityLimits = DEFAULT_VELOCITY_LIMITS,
        max_ttl_ms: int = 250,
    ) -> "VelocityProposal":
        if not isinstance(payload, dict):
            raise ProtocolError("invalid_payload", "proposal must be a JSON object")
        if payload.get("schema") != VELOCITY_PROPOSAL_SCHEMA:
            raise ProtocolError(
                "schema_mismatch", f"schema must be {VELOCITY_PROPOSAL_SCHEMA}"
            )
        if payload.get("frame") != "base_link":
            raise ProtocolError("frame_mismatch", "frame must be base_link")
        if payload.get("shadow_only") is not True:
            raise ProtocolError("unsafe_flag", "shadow_only must be true")
        if payload.get("physical_execution") is not False:
            raise ProtocolError("unsafe_flag", "physical_execution must be false")
        if any(
            alias in payload
            for alias in ("status", "navigation_status", "navigation_state")
        ):
            raise ProtocolError(
                "unsupported_navigation_status_field",
                "nav_status is the only supported navigation status field",
            )

        nav_id = _identifier(payload.get("nav_id"), "nav_id")
        sequence = _positive_int(payload.get("sequence"), "sequence")
        ttl_ms = _positive_int(payload.get("ttl_ms"), "ttl_ms")
        if ttl_ms > max_ttl_ms:
            raise ProtocolError("ttl_limit", f"ttl_ms must not exceed {max_ttl_ms}")
        issued_at_unix_ms = _positive_int(
            payload.get("issued_at_unix_ms"), "issued_at_unix_ms"
        )
        navigation_status = payload.get("nav_status")
        allowed_statuses = _MOTION_STATUSES | _IDLE_STATUSES | _TERMINAL_STATUSES
        if navigation_status not in allowed_statuses:
            raise ProtocolError(
                "invalid_navigation_status",
                f"unsupported nav_status: {navigation_status}",
            )

        raw_velocity = payload.get("velocity")
        if not isinstance(raw_velocity, dict):
            raise ProtocolError("invalid_velocity", "velocity must be an object")
        velocity = Velocity(
            x=_finite_number(raw_velocity.get("x"), "velocity.x"),
            y=_finite_number(raw_velocity.get("y"), "velocity.y"),
            yaw=_finite_number(raw_velocity.get("yaw"), "velocity.yaw"),
        )
        limits.validate(velocity)
        if navigation_status not in _MOTION_STATUSES and not velocity.is_zero():
            raise ProtocolError(
                "unsafe_navigation_state",
                f"{navigation_status} proposals must carry zero velocity",
            )

        reason = payload.get("reason")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 256):
            raise ProtocolError(
                "invalid_reason", "reason must be a string of at most 256 characters"
            )
        return cls(
            nav_id=nav_id,
            sequence=sequence,
            ttl_ms=ttl_ms,
            issued_at_unix_ms=issued_at_unix_ms,
            navigation_status=navigation_status,
            velocity=velocity,
            reason=reason,
        )

    def as_payload(self) -> dict:
        payload = {
            "schema": VELOCITY_PROPOSAL_SCHEMA,
            "nav_id": self.nav_id,
            "sequence": self.sequence,
            "ttl_ms": self.ttl_ms,
            "issued_at_unix_ms": self.issued_at_unix_ms,
            "frame": "base_link",
            "nav_status": self.navigation_status,
            "velocity": self.velocity.as_dict(),
            "shadow_only": True,
            "physical_execution": False,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def build_velocity_proposal(
    *,
    nav_id: str,
    sequence: int,
    ttl_ms: int,
    navigation_status: str,
    velocity: Velocity,
    reason: str | None = None,
    issued_at_unix_ms: int | None = None,
) -> dict:
    """Build and self-validate one proposal before ROS publication."""

    proposal = VelocityProposal(
        nav_id=nav_id,
        sequence=sequence,
        ttl_ms=ttl_ms,
        issued_at_unix_ms=issued_at_unix_ms or time.time_ns() // 1_000_000,
        navigation_status=navigation_status,
        velocity=velocity,
        reason=reason,
    )
    payload = proposal.as_payload()
    VelocityProposal.from_payload(payload)
    return payload


__all__ = [
    "DEFAULT_MOTION_LIMITS",
    "DEFAULT_VELOCITY_LIMITS",
    "MIN_EFFECTIVE_LINEAR_MPS",
    "MIN_EFFECTIVE_YAW_RADPS",
    "MotionLimits",
    "Pose2D",
    "TERMINAL_XY_TOLERANCE_M",
    "TERMINAL_YAW_TOLERANCE_RAD",
    "TURN_ONLY_YAW_THRESHOLD_RADPS",
    "ProtocolError",
    "SCHEMA_VERSION",
    "VELOCITY_PROPOSAL_SCHEMA",
    "VELOCITY_PROPOSAL_TOPIC",
    "Velocity",
    "VelocityLimits",
    "VelocityProposal",
    "apply_g1_motion_floor",
    "apply_g1_motion_limits",
    "build_velocity_proposal",
    "limit_forward_velocity",
    "proposal_context_is_current",
    "shape_terminal_approach",
]
