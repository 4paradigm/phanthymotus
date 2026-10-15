"""`_resolve_msg_type` — the format table the dashboard's subscriptions rest on.

An unresolved format is not an error anyone sees: the bridge prints one line to
stderr and declines to subscribe. The topic stays registered, the producer keeps
publishing, and the canvas panel is empty for ever — which reads as a card that
produces nothing. That happened on r1_sz with `control/velocity`.
"""
from __future__ import annotations

import os
import asyncio
import json
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


@pytest.mark.parametrize("fmt", ["data/json", "json", "sensor/imu", "sensor/sound-direction"])
def test_the_formats_that_already_worked_still_do(fmt):
    assert ros2_bridge._resolve_msg_type(fmt) is _msg.String


def test_an_unknown_format_still_returns_none():
    """Declining is right for a format nobody can decode; the bug was the set
    of formats, not the declining."""
    assert ros2_bridge._resolve_msg_type("application/x-nonsense") is None


def test_dedicated_direction_delivers_standard_sensor_payload(monkeypatch):
    callbacks, received = [], []
    node = types.SimpleNamespace(create_subscription=lambda *args: callbacks.append(args) or object())
    monkeypatch.setattr(ros2_bridge, '_HAS_RCLPY', True)
    monkeypatch.setattr(ros2_bridge, '_running', True)
    monkeypatch.setattr(ros2_bridge, '_node_main', node)
    monkeypatch.setattr(ros2_bridge, '_executor', None)
    monkeypatch.setattr(ros2_bridge, '_low_lat_qos', lambda: None)
    monkeypatch.setattr(ros2_bridge, 'unsubscribe', lambda _: None)
    monkeypatch.setattr(ros2_bridge.asyncio, 'run_coroutine_threadsafe',
                        lambda coro, loop: asyncio.run(coro))

    async def receive(data, fmt):
        received.append((json.loads(data), fmt))

    ros2_bridge.subscribe('direction', '/robot/sound_direction/events',
                          'sensor/sound-direction', None, receive)
    assert callbacks[0][0] is _msg.String
    payload = {'state': 'fresh', 'angle': -30, 'angle_deg': -29.678,
               'audio_window_start_us': 1700000000000000,
               'audio_window_end_us': 1700000000020000}
    callbacks[0][2](types.SimpleNamespace(data=json.dumps(payload)))
    assert received == [(payload, 'sensor/sound-direction')]
