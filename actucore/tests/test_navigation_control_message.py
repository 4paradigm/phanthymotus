"""Pure construction tests; no ROS nodes, clocks or robot operations."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navigation.planning.control_message import (  # noqa: E402
    build_navigation_control_message,
)


def _args(**changes):
    args = {
        "seq": 42,
        "source": "navigation/session-1",
        "stamp_ms": 1_700_000_000_100,
        "obs_stamp_ms": 1_700_000_000_050,
        "ttl_ms": 100,
        "priority": 37,
        "max_obs_age_ms": 100,
        "max_future_skew_ms": 10,
        "vx": 0.3,
        "vy": -0.1,
        "wz": -0.2,
    }
    args.update(changes)
    return args


def test_constructs_standard_twist_with_original_observation_timestamp():
    payload = build_navigation_control_message(**_args())
    assert payload == {
        "schema": "motus.control/1",
        "seq": 42,
        "source": "navigation/session-1",
        "stamp_ms": 1_700_000_000_100,
        "obs_stamp_ms": 1_700_000_000_050,
        "ttl_ms": 100,
        "priority": 37,
        "mode": "twist",
        "dof": 6,
        "values": [0.3, -0.1, 0.0, 0.0, 0.0, -0.2],
    }


def test_delegates_wire_construction_to_shared_builder():
    expected = object()
    with patch(
        "plugins.navigation.planning.control_message.build_control_message",
        return_value=expected,
    ) as shared:
        assert build_navigation_control_message(**_args()) is expected
    shared.assert_called_once_with(
        seq=42, values=[0.3, -0.1, 0.0, 0.0, 0.0, -0.2], mode="twist",
        dof=6, source="navigation/session-1", stamp_ms=1_700_000_000_100,
        obs_stamp_ms=1_700_000_000_050, ttl_ms=100, priority=37,
    )


def test_zero_command_and_zero_seq_are_valid():
    message = build_navigation_control_message(
        **_args(seq=0, vx=0, vy=0, wz=0, max_obs_age_ms=0,
                obs_stamp_ms=1_700_000_000_100, max_future_skew_ms=0)
    )
    assert message["values"] == [0.0] * 6
    assert message["seq"] == 0


@pytest.mark.parametrize("field", ["vx", "vy", "wz"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1", None, 10**400])
def test_invalid_velocity_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        build_navigation_control_message(**_args(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [("seq", -1), ("seq", True), ("seq", 1.5),
     ("stamp_ms", 0), ("stamp_ms", -1), ("stamp_ms", 1.5),
     ("obs_stamp_ms", 0), ("obs_stamp_ms", -1), ("obs_stamp_ms", None),
     ("ttl_ms", 0), ("ttl_ms", -1), ("ttl_ms", float("inf")),
     ("max_obs_age_ms", -1), ("max_obs_age_ms", True),
     ("max_future_skew_ms", -1), ("max_future_skew_ms", 1.0),
     ("priority", True), ("priority", float("nan"))],
)
def test_invalid_integer_fields_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        build_navigation_control_message(**_args(**{field: value}))


@pytest.mark.parametrize("source", ["", "  ", " navigation", "navigation ", None, 123])
def test_invalid_source_is_rejected(source):
    with pytest.raises(ValueError, match="source"):
        build_navigation_control_message(**_args(source=source))


def test_observation_age_bound_is_inclusive():
    args = _args(stamp_ms=1000, obs_stamp_ms=900, max_obs_age_ms=100)
    assert build_navigation_control_message(**args)["obs_stamp_ms"] == 900
    with pytest.raises(ValueError, match="older"):
        build_navigation_control_message(**{**args, "obs_stamp_ms": 899})


def test_observation_future_skew_bound_is_inclusive():
    args = _args(stamp_ms=1000, obs_stamp_ms=1010, max_future_skew_ms=10)
    assert build_navigation_control_message(**args)["obs_stamp_ms"] == 1010
    with pytest.raises(ValueError, match="future"):
        build_navigation_control_message(**{**args, "obs_stamp_ms": 1011})
    with pytest.raises(ValueError, match="future"):
        build_navigation_control_message(**{**args, "max_future_skew_ms": 0})


@pytest.mark.parametrize(
    "missing", ["seq", "source", "stamp_ms", "obs_stamp_ms", "ttl_ms", "priority",
                "max_obs_age_ms", "max_future_skew_ms", "vx", "vy", "wz"]
)
def test_required_fields_are_not_defaulted(missing):
    args = _args()
    del args[missing]
    with pytest.raises(TypeError):
        build_navigation_control_message(**args)
