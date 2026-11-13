"""ROS-free Navi-compatible control stream helpers for the Nav2 subprocess.

This package is installed independently of ActuCore, so it cannot import the
card's control_stream helpers. Keep the wire format and rate/TTL rules aligned
with those helpers. Sensor header stamps describe readiness inputs, not proof
that the Nav2 controller consumed those particular samples.
"""
from __future__ import annotations

import json
import math

from .execution_protocol import Velocity

SCHEMA = "motus.control/1"
AXES = ["vx", "vy", "vz", "wx", "wy", "wz"]


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def configure(raw: str):
    """Validate a descriptor and apply Navi's effective-rate and TTL rules."""
    try:
        descriptor = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("control_interface_json must be a JSON object") from exc
    if not isinstance(descriptor, dict):
        raise ValueError("control_interface_json must be a JSON object")
    rate = descriptor.get("rate") or {}
    if not isinstance(rate, dict):
        raise ValueError("control descriptor rate must be an object")
    if descriptor:
        if descriptor.get("control_interface") != SCHEMA or descriptor.get("mode") != "twist":
            raise ValueError("control descriptor must declare motus.control/1 twist")
        dof = descriptor.get("dof")
        try:
            if dof is not None and int(dof) != 6:
                raise ValueError("control descriptor dof must match 6")
        except (TypeError, OverflowError) as exc:
            raise ValueError("control descriptor dof must match 6") from exc
        groups = descriptor.get("groups")
        if groups is not None and (not isinstance(groups, (list, tuple)) or any(not isinstance(g, dict) for g in groups)):
            raise ValueError("control descriptor groups must be a list of objects")
    # Match negotiate.check: a truthy max_hz is parsed and compared against the
    # producer capability; malformed truthy values fail rather than crash.
    raw_max = rate.get("max_hz")
    try:
        parsed_max = float(raw_max) if raw_max else 0.0
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid rate.max_hz") from exc
    if parsed_max < 10.0 and raw_max:
        raise ValueError("control producer capability 10 Hz exceeds downstream max_hz")
    candidates = [10.0]
    for key in ("expected_hz", "max_hz"):
        value = _finite(rate.get(key))
        if value is not None and value > 0:
            candidates.append(value)
    hz = min(candidates)
    # A pathological subnormal rate can overflow the timer period. Treat that
    # malformed value like other unrepresentable rates rather than crashing.
    if hz < 1e-300:
        hz = 10.0
    period_ms = 1000.0 / hz
    watchdog = _finite(rate.get("watchdog_ms"))
    ttl = max(50.0, period_ms * 2)
    if watchdog is not None and watchdog > 0:
        ttl = min(ttl, watchdog)
    ttl_ms = int(ttl) if math.isfinite(ttl) else 200
    max_obs_age_ms = _finite(rate.get("max_obs_age_ms"))
    if max_obs_age_ms is not None and max_obs_age_ms <= 0:
        max_obs_age_ms = None
    limits = descriptor.get("limits")
    limits = limits if isinstance(limits, dict) else {}
    bounds = [[-math.inf] * 6, [math.inf] * 6]
    lower, upper = limits.get("lower"), limits.get("upper")
    if isinstance(lower, list) and isinstance(upper, list):
        for i in range(min(6, len(lower), len(upper))):
            low, high = _finite(lower[i]), _finite(upper[i])
            # Never amplify or reverse a command to meet an incompatible bound.
            # Invalid pairs and pairs excluding zero make the axis stop-only.
            if low is None or high is None or low > 0 or high < 0:
                bounds[0][i] = bounds[1][i] = 0.0
            else:
                bounds[0][i], bounds[1][i] = low, high
    floors = []
    for key in ("min_magnitude", "min_magnitude_moving"):
        values = limits.get(key)
        if isinstance(values, list):
            row = []
            for i in range(6):
                value = _finite(values[i]) if i < len(values) else None
                row.append(abs(value) if value is not None else 0.0)
            floors.append(row)
        else:
            floors.append(None)
    return descriptor, hz, ttl_ms, max_obs_age_ms, bounds, floors


def limit_velocity(velocity: Velocity, bounds, floors) -> Velocity:
    """Fit descriptor ceilings/deadbands without amplifying a Nav2 proposal."""
    values = [velocity.x, velocity.y, 0.0, 0.0, 0.0, velocity.yaw]
    result = []
    for i, value in enumerate(values):
        selected = (floors[1] if i == 5 and floors[1] is not None and
                    (values[0] or values[1]) else floors[0])
        value = max(bounds[0][i], min(bounds[1][i], value))
        if selected is not None and abs(value) < selected[i]:
            value = 0.0
        result.append(value)
    return Velocity(x=result[0], y=result[1], yaw=result[5])


def observation_stamp_ms(odom_ns, obstacle_ns, *, ros_now_ns, unix_now_ns, max_age_ms):
    """Select the older ready input when timestamps plausibly share Unix time.

    Clock comparison detects obvious simulation/domain mismatch, not a dishonest
    or unsynchronised source whose header merely happens to look plausible.
    """
    stamps = (odom_ns, obstacle_ns)
    if any(isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in stamps):
        return None
    if abs(ros_now_ns - unix_now_ns) > 100_000_000:
        return None
    if any(s > ros_now_ns or s > unix_now_ns for s in stamps):
        return None
    oldest = min(stamps)
    if unix_now_ns - oldest > max_age_ms * 1_000_000:
        return None
    return oldest // 1_000_000


def _wire_json(value):
    """Remove non-JSON IEEE numbers without changing the internal descriptor."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _wire_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wire_json(item) for item in value]
    return value


def build(*, seq, stamp_ms, obs_stamp_ms, ttl_ms, velocity, descriptor, stop_fallback=False):
    payload = {
        "schema": SCHEMA, "seq": seq, "stamp_ms": stamp_ms,
        "obs_stamp_ms": obs_stamp_ms, "ttl_ms": ttl_ms,
        "source": "mcp__actucore__controlled_semantic_spatial", "priority": 50,
        "mode": "twist", "dof": 6,
        "values": [velocity.x, velocity.y, 0.0, 0.0, 0.0, velocity.yaw],
        "obs_stamp_kind": ("bridge_stop_generated" if stop_fallback else
                           "earliest_readiness_input_header"),
    }
    if seq % 50 == 1:
        payload["control_interface"] = _wire_json(descriptor) if descriptor else {
            "control_interface": SCHEMA, "mode": "twist", "dof": 6,
            "joint_names": list(AXES),
        }
    return payload
