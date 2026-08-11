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
class VelocityLimits:
    min_x: float = -0.05
    max_x: float = 0.15
    max_abs_y: float = 0.12
    max_abs_yaw: float = 0.35
    max_planar_speed: float = 0.18

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
    "DEFAULT_VELOCITY_LIMITS",
    "ProtocolError",
    "SCHEMA_VERSION",
    "VELOCITY_PROPOSAL_SCHEMA",
    "VELOCITY_PROPOSAL_TOPIC",
    "Velocity",
    "VelocityLimits",
    "VelocityProposal",
    "build_velocity_proposal",
]
