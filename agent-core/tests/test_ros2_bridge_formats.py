"""`_resolve_msg_type` — the format table the dashboard's subscriptions rest on.

An unresolved format is not an error anyone sees: the bridge prints one line to
stderr and declines to subscribe. The topic stays registered, the producer keeps
publishing, and the canvas panel is empty for ever — which reads as a card that
produces nothing. That happened on r1_sz with `control/velocity`.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# std_msgs is not installed on a laptop; the table only needs the name back.
_std = sys.modules.setdefault("std_msgs", types.ModuleType("std_msgs"))
_msg = sys.modules.setdefault("std_msgs.msg", types.ModuleType("std_msgs.msg"))
for _n in ("String", "UInt8MultiArray"):
    setattr(_msg, _n, type(_n, (), {}))
setattr(_std, "msg", _msg)

import ros2_bridge  # noqa: E402


@pytest.mark.parametrize("fmt", [
    "control/velocity",     # motus.control/1 twist — navi → loco_servo
    "control/joint",        # motus.control/1 joint modes — vla → servo
    "state/odom",           # motus.odom/1
    "state/joint",
])
def test_control_and_state_formats_resolve_to_string(fmt):
    """Both carry JSON in a std_msgs/String, like the perception cards."""
    assert ros2_bridge._resolve_msg_type(fmt) is _msg.String


@pytest.mark.parametrize("fmt", ["data/json", "json", "sensor/imu"])
def test_the_formats_that_already_worked_still_do(fmt):
    assert ros2_bridge._resolve_msg_type(fmt) is _msg.String


def test_an_unknown_format_still_returns_none():
    """Declining is right for a format nobody can decode; the bug was the set
    of formats, not the declining."""
    assert ros2_bridge._resolve_msg_type("application/x-nonsense") is None
