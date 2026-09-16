"""Two producers claiming one topic under different formats.

The consumer rule in register_topic_internal cannot catch this: both sides are
publishers, so each simply overwrote the other. Every overwrite tore down the
dashboard's subscription and rebuilt it against a message type the other
producer was not sending, which rclpy refuses as an incompatible type on an
existing topic name.

Observed on Tianyi: the driver's `camera_depth` published
/nvidia_desktop/camera/head/depth as image/depth-z16 while perception's
visual_depth derived the same name for its image/depth-zlib output. 303 rebuilds
later the panel had never received a frame, while both producers were publishing
perfectly well — it presented as visual_depth being stuck on its first frame.

Run: cd agent-core && python3 -m pytest tests/test_topic_collision.py -q
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import inspection  # noqa: E402

TOPIC = '/nvidia_desktop/camera/head/depth'


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    inspection._topic_registry.clear()
    inspection._active_primary_subs.clear()
    inspection._logged_topic_collisions.clear()
    # _ensure_primary_sub reaches for DDS; the registry decision is what matters.
    calls = []
    monkeypatch.setattr(inspection, '_ensure_primary_sub',
                        lambda topic, fmt, loop: calls.append((topic, fmt)))
    return calls


def _register(topic, fmt, mcp_id, producer=True):
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        inspection.register_topic_internal(topic, fmt, mcp_id, producer=producer))


def test_the_first_producer_keeps_the_topic():
    _register(TOPIC, 'image/depth-zlib', 'mcp-perception')
    _register(TOPIC, 'image/depth-z16', 'mcp-driver')
    assert inspection._topic_registry[TOPIC]['format'] == 'image/depth-zlib'
    assert inspection._topic_registry[TOPIC]['mcp_id'] == 'mcp-perception'


def test_the_rejected_producer_does_not_tear_down_the_subscription(clean):
    _register(TOPIC, 'image/depth-zlib', 'mcp-perception')
    inspection._active_primary_subs.add(TOPIC)
    _register(TOPIC, 'image/depth-z16', 'mcp-driver')
    assert TOPIC in inspection._active_primary_subs, \
        'the live subscription must survive a competing claim'


def test_flapping_settles_instead_of_rebuilding_forever(clean):
    # What actually happened: both sides re-register on every heartbeat.
    inspection._active_primary_subs.add(TOPIC)
    for _ in range(20):
        _register(TOPIC, 'image/depth-zlib', 'mcp-perception')
        _register(TOPIC, 'image/depth-z16', 'mcp-driver')
    assert inspection._topic_registry[TOPIC]['format'] == 'image/depth-zlib'
    assert TOPIC in inspection._active_primary_subs
    # And it says so once, not forty times.
    assert len(inspection._logged_topic_collisions) == 1


def test_a_producer_may_still_correct_its_own_format(clean):
    # Not a collision — the same publisher changing what it sends. This has to
    # keep rebuilding, because the message type really did change.
    _register(TOPIC, 'image/depth-z16', 'mcp-perception')
    inspection._active_primary_subs.add(TOPIC)
    _register(TOPIC, 'image/depth-zlib', 'mcp-perception')
    assert inspection._topic_registry[TOPIC]['format'] == 'image/depth-zlib'
    assert TOPIC not in inspection._active_primary_subs, \
        'a real format change must still rebuild'


def test_a_second_producer_agreeing_on_the_format_is_not_a_collision(clean):
    _register(TOPIC, 'image/depth-zlib', 'mcp-perception')
    _register(TOPIC, 'image/depth-zlib', 'mcp-driver')
    assert not inspection._logged_topic_collisions


def test_a_consumer_still_cannot_change_the_format(clean):
    # The pre-existing rule, unchanged: topic_in only says what a card wants to
    # eat, and must not rewrite what the publisher is sending.
    _register(TOPIC, 'image/depth-zlib', 'mcp-perception')
    _register(TOPIC, 'image/jpeg', 'mcp-ocr', producer=False)
    assert inspection._topic_registry[TOPIC]['format'] == 'image/depth-zlib'
