"""Pure, opt-in construction of a planar navigation control message.

This is not a publisher or a motion safety gate. The caller must establish the
observation's provenance and clock domain, negotiate the driver descriptor,
and enforce navigation ownership, velocity limits and a downstream watchdog.
"""

from __future__ import annotations

import math

from ...control_stream.message import build as build_control_message


def _integer(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _velocity(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def build_navigation_control_message(
    *,
    seq: int,
    source: str,
    stamp_ms: int,
    obs_stamp_ms: int,
    ttl_ms: int,
    priority: int,
    max_obs_age_ms: int,
    max_future_skew_ms: int,
    vx: float,
    vy: float,
    wz: float,
) -> dict:
    """Build a standard twist using caller-supplied Unix-millisecond times.

    ``stamp_ms`` is command generation time; ``obs_stamp_ms`` is the capture
    time of the actual trusted observation used to generate the command. Both
    must share a verified Unix clock domain. Observation age is checked at
    generation against explicit caller-selected bounds; both bounds are
    inclusive, and a small future skew is permitted only when explicitly given.
    The sink must independently check freshness at receipt.

    Positive timestamps and TTL are mandatory; seq and the two age bounds may
    be zero. Signed and zero velocities are representable. This function does not
    establish provenance or clamp/validate velocities against robot limits.
    """
    _integer("seq", seq, minimum=0)
    _integer("stamp_ms", stamp_ms, minimum=1)
    _integer("obs_stamp_ms", obs_stamp_ms, minimum=1)
    _integer("ttl_ms", ttl_ms, minimum=1)
    _integer("max_obs_age_ms", max_obs_age_ms, minimum=0)
    _integer("max_future_skew_ms", max_future_skew_ms, minimum=0)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ValueError("priority must be an integer")
    if not isinstance(source, str) or not source or source != source.strip():
        raise ValueError("source must be a non-empty, trimmed string")

    age_ms = stamp_ms - obs_stamp_ms
    if age_ms > max_obs_age_ms:
        raise ValueError("observation is older than max_obs_age_ms")
    if age_ms < -max_future_skew_ms:
        raise ValueError("observation exceeds max_future_skew_ms")

    return build_control_message(
        seq=seq,
        values=[_velocity("vx", vx), _velocity("vy", vy), 0.0, 0.0, 0.0,
                _velocity("wz", wz)],
        mode="twist",
        dof=6,
        source=source,
        stamp_ms=stamp_ms,
        obs_stamp_ms=obs_stamp_ms,
        ttl_ms=ttl_ms,
        priority=priority,
    )
