"""ROS-free regression for native PointCloud2 topic registration in Core.

The dashboard inspection path only forwards String/UInt8MultiArray-like payloads.
PointCloud2 must remain a registered canvas topic while the native ROS producer
and the Perception SLAM consumer connect directly without a spurious String
subscription from phanthy_bus_bridge.
"""
import asyncio
import os
from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import inspection  # noqa: E402


NATIVE = '/ubuntu/lidar/cloud_native'
ODOM = '/ubuntu/slam/mapping/odom'


@pytest.fixture(autouse=True)
def clean_inspection_state(monkeypatch):
    inspection._topic_registry.clear()
    inspection._active_primary_subs.clear()
    inspection._logged_topic_collisions.clear()
    inspection._last_frame.clear()
    inspection._topic_queues.clear()
    registered = {}
    subscribed, unsubscribed = [], []

    def subscribe(key, topic, fmt, loop, callback):
        subscribed.append((key, topic, fmt))
        registered[key] = (topic, fmt)

    def unsubscribe(key):
        unsubscribed.append(key)
        registered.pop(key, None)

    monkeypatch.setattr(inspection.ros2_bridge, 'subscribe', subscribe)
    monkeypatch.setattr(inspection.ros2_bridge, 'unsubscribe', unsubscribe)
    monkeypatch.setattr(inspection.ros2_bridge, 'get_dds_topics', lambda: {NATIVE, ODOM})
    monkeypatch.setattr(inspection.ros2_bridge, 'get_last_seen', lambda topic: 0.0)
    yield registered, subscribed, unsubscribed
    inspection._topic_registry.clear()
    inspection._active_primary_subs.clear()
    inspection._logged_topic_collisions.clear()
    inspection._last_frame.clear()
    inspection._topic_queues.clear()


def register(topic, fmt, *, producer=True, mcp_id='g1-driver'):
    asyncio.run(inspection.register_topic_internal(
        topic, fmt, mcp_id, producer=producer))


def test_native_pointcloud_is_registered_but_never_string_subscribed(clean_inspection_state):
    registered, subscribed, unsubscribed = clean_inspection_state
    register(NATIVE, 'sensor/pointcloud2')
    assert inspection._topic_registry[NATIVE]['format'] == 'sensor/pointcloud2'
    assert inspection._topic_registry[NATIVE]['mcp_id'] == 'g1-driver'
    assert NATIVE not in inspection._active_primary_subs
    assert registered == {} and subscribed == []
    assert '__primary__#' + NATIVE in unsubscribed  # removes stale legacy monitor
    # ROS publisher discovery, rather than dashboard reception, reports online.
    assert inspection._topic_status(NATIVE) == 'online'
    assert asyncio.run(inspection.topics_status())['data'][NATIVE] == 'online'


def test_repeated_registration_and_consumer_mismatch_do_not_create_subscription(
        clean_inspection_state):
    registered, subscribed, _ = clean_inspection_state
    for _ in range(3):
        register(NATIVE, 'sensor/pointcloud2')
    # A consumer cannot overwrite the producer's native wire format.
    register(NATIVE, 'data/json', producer=False, mcp_id='slam-consumer')
    assert inspection._topic_registry[NATIVE]['format'] == 'sensor/pointcloud2'
    assert subscribed == [] and registered == {}
    assert NATIVE not in inspection._active_primary_subs


def test_correcting_legacy_string_topic_to_native_removes_stale_reader(
        clean_inspection_state):
    registered, subscribed, unsubscribed = clean_inspection_state
    register(NATIVE, 'data/json')
    key = '__primary__#' + NATIVE
    assert registered[key] == (NATIVE, 'data/json')
    assert NATIVE in inspection._active_primary_subs
    inspection._last_frame[NATIVE] = b'stale string payload'
    register(NATIVE, 'sensor/pointcloud2')
    assert inspection._topic_registry[NATIVE]['format'] == 'sensor/pointcloud2'
    assert NATIVE not in inspection._active_primary_subs
    assert NATIVE not in inspection._last_frame
    assert registered == {}
    assert unsubscribed[-1] == key
    assert len(subscribed) == 1


def test_changing_native_topic_back_to_ordinary_string_restores_monitoring(
        clean_inspection_state):
    registered, subscribed, _ = clean_inspection_state
    register(NATIVE, 'sensor/pointcloud2')
    register(NATIVE, 'state/odom')
    key = '__primary__#' + NATIVE
    assert registered[key] == (NATIVE, 'state/odom')
    assert NATIVE in inspection._active_primary_subs
    assert subscribed[-1] == (key, NATIVE, 'state/odom')


@pytest.mark.parametrize('fmt', [
    'sensor/pointcloud', 'sensor/mapping', 'state/odom',
    'data/json', 'sensor/tactile', 'image/jpeg',
])
def test_existing_formats_keep_their_primary_subscriptions(fmt, clean_inspection_state):
    registered, subscribed, _ = clean_inspection_state
    register(ODOM, fmt)
    assert inspection._topic_registry[ODOM]['format'] == fmt
    assert ODOM in inspection._active_primary_subs
    assert registered['__primary__#' + ODOM] == (ODOM, fmt)
    assert len(subscribed) == 1


def test_failed_competing_producer_does_not_override_native_format(
        clean_inspection_state):
    registered, subscribed, _ = clean_inspection_state
    register(NATIVE, 'sensor/pointcloud2')
    register(NATIVE, 'data/json', mcp_id='other-driver')
    assert inspection._topic_registry[NATIVE]['format'] == 'sensor/pointcloud2'
    assert inspection._topic_registry[NATIVE]['mcp_id'] == 'g1-driver'
    assert len(inspection._logged_topic_collisions) == 1
    assert subscribed == [] and registered == {}
