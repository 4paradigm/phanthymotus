"""Real ROS message shape is decoded without inferring timestamps or axes."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from plugins.pointcloud_slam_ros import decode_cloud, SlamInputNode
from test_pointcloud_slam_stream import odom, stream


def cloud():
    points = np.array([[1, 2, 3]], dtype='<f4')
    return NS(header=NS(stamp=NS(sec=1, nanosec=0), frame_id='camera'),
              width=1, height=1, point_step=12, row_step=12, is_bigendian=False,
              fields=[NS(name=n, offset=i*4, datatype=7, count=1)
                      for i, n in enumerate('xyz')], data=points.tobytes())


def test_decodes_metric_xyz_and_preserves_header():
    stamp, frame, points = decode_cloud(cloud(), max_points=5)
    assert stamp == 1_000_000_000 and frame == 'camera'
    assert points.tolist() == [[1, 2, 3]]


def test_cloud_rejects_truncation_and_wrong_field_type():
    msg = cloud()
    msg.data = b'\0'
    with pytest.raises(ValueError):
        decode_cloud(msg, max_points=5)
    msg = cloud()
    msg.fields[1].offset = 1
    with pytest.raises(ValueError):
        decode_cloud(msg, max_points=5)
    msg = cloud()
    msg.fields[0].datatype = 2
    with pytest.raises(ValueError):
        decode_cloud(msg, max_points=5)


def test_two_subscriptions_feed_keyframe_store():
    import json
    from vision_stubs import _FakeExecutor
    s = stream()
    node = SlamInputNode('/dynamic/odom', '/dynamic/cloud', s, clock=lambda: 1.)
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/dynamic/odom'](NS(data=json.dumps(odom(1000))))
    callbacks['/dynamic/cloud'](cloud())
    assert len(s.store.segments[0].frames) == 1
    assert set(callbacks) == {'/dynamic/odom', '/dynamic/cloud'}
