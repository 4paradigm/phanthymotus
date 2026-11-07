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


@pytest.mark.parametrize("fmt", ["data/json", "json", "sensor/imu", "sensor/tactile"])
def test_the_formats_that_already_worked_still_do(fmt):
    assert ros2_bridge._resolve_msg_type(fmt) is _msg.String


def test_an_unknown_format_still_returns_none():
    """Declining is right for a format nobody can decode; the bug was the set
    of formats, not the declining."""
    assert ros2_bridge._resolve_msg_type("application/x-nonsense") is None


def test_slam_preview_format_uses_binary_ros_message():
    assert ros2_bridge._resolve_msg_type('sensor/slam-preview') is _msg.UInt8MultiArray


def test_slam_preview_snapshot_is_relayed_as_binary(monkeypatch):
    import asyncio
    from api import inspection
    topic, raw = '/test/slam-preview', b'\xff\x00\x01'
    monkeypatch.setitem(inspection._topic_registry, topic, {'format': 'sensor/slam-preview'})
    monkeypatch.setitem(inspection._last_frame, topic, raw)
    monkeypatch.setattr(inspection, '_ensure_primary_sub', lambda *a: None)
    unsubscribed = []
    monkeypatch.setattr(ros2_bridge, 'unsubscribe', lambda key: unsubscribed.append(key))
    inspection._active_primary_subs.add(topic)
    class Socket:
        def __init__(self):
            self.binary = []
            self.text = []
        async def accept(self):
            pass
        async def send_text(self, message):
            self.text.append(message)
            if len(self.text) > 1:
                raise RuntimeError('wrongly relayed as text')
        async def send_bytes(self, data):
            self.binary.append(data)
            asyncio.get_running_loop().call_soon(
                lambda: inspection._topic_queues[topic][0].put_nowait(None))
    socket = Socket()
    asyncio.run(inspection.bus_ws(socket, topic.lstrip('/')))
    assert socket.binary == [raw] and len(socket.text) == 1
    assert unsubscribed == []
    assert inspection._last_frame[topic] == raw


def test_full_preview_queue_keeps_main_frame_order(monkeypatch):
    import asyncio
    from api import inspection
    topic = '/test/slow-preview'
    queue = asyncio.Queue(maxsize=2)
    monkeypatch.setitem(inspection._topic_registry, topic, {'format': 'sensor/slam-preview'})
    monkeypatch.setitem(inspection._topic_queues, topic, [queue])
    async def feed():
        push = inspection._push_factory(topic)
        await push(b'old', 'sensor/slam-preview')
        await push(b'middle', 'sensor/slam-preview')
        await push(b'latest', 'sensor/slam-preview')
        return [queue.get_nowait(), queue.get_nowait()]
    assert asyncio.run(feed()) == [b'old', b'middle']
