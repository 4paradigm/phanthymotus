"""Keyframe storage must not silently join discontinuous trajectories."""
import numpy as np
import pytest

from plugins.pointcloud_slam_keyframes import KeyframeStore
from plugins.pointcloud_slam_odom import PoseSample


def pose(stamp, segment='a', frame='odom'):
    return PoseSample(stamp, frame, 'body', segment, (0., 0., 0.),
                      (0., 0., 0., 1.), None, 'robot-clock')


def add(store, sample, points):
    store.add(sample, points, cloud_stamp_ns=sample.stamp_ms * 1_000_000)


def test_preserves_cloud_capture_nanoseconds_and_checks_aligned_pose_time():
    store = KeyframeStore(max_frames=2, max_points=10)
    store.add(pose(1000), [[1., 2., 3.]], cloud_stamp_ns=1_000_123_456)
    assert store.segments[0].frames[0].cloud_stamp_ns == 1_000_123_456
    with pytest.raises(ValueError, match='timestamp'):
        store.add(pose(1100), [[1., 2., 3.]], cloud_stamp_ns=1_101_000_000)


def test_keeps_local_points_and_separates_new_segment():
    store = KeyframeStore(max_frames=3, max_points=10)
    points = np.array([[1., 2., 3.]])
    add(store, pose(1000), points)
    points[0, 0] = 99
    add(store, pose(1100, 'b'), [[4., 5., 6.]])
    assert [s.segment_id for s in store.segments] == ['a', 'b']
    assert store.segments[0].frames[0].points[0].tolist() == [1., 2., 3.]
    with pytest.raises(ValueError):
        store.segments[0].frames[0].points[0, 0] = 3.


def test_lost_blocks_old_segment_until_new_segment_arrives():
    store = KeyframeStore(max_frames=4, max_points=10)
    add(store, pose(1000), [[1., 2., 3.]])
    store.mark_lost()
    with pytest.raises(ValueError, match='segment'):
        add(store, pose(1100), [[1., 2., 3.]])
    add(store, pose(1200, 'b'), [[1., 2., 3.]])
    assert [s.segment_id for s in store.segments] == ['a', 'b']


def test_bounds_reject_new_frames_without_discarding_history():
    store = KeyframeStore(max_frames=2, max_points=2)
    add(store, pose(1000), [[1., 2., 3.]])
    with pytest.raises(ValueError, match='capacity'):
        add(store, pose(1100), [[1., 2., 3.], [4., 5., 6.]])
    add(store, pose(1200), [[4., 5., 6.]])
    with pytest.raises(ValueError, match='capacity'):
        add(store, pose(1300), [[7., 8., 9.]])
    assert len(store.segments[0].frames) == 2


def test_rejects_out_of_order_reused_segment_and_frame_change():
    store = KeyframeStore(max_frames=5, max_points=10)
    add(store, pose(1000), [[1., 2., 3.]])
    with pytest.raises(ValueError):
        add(store, pose(900), [[1., 2., 3.]])
    with pytest.raises(ValueError):
        add(store, pose(1100, frame='other'), [[1., 2., 3.]])
    add(store, pose(1200, 'b'), [[1., 2., 3.]])
    with pytest.raises(ValueError):
        add(store, pose(1300), [[1., 2., 3.]])


@pytest.mark.parametrize('points', [[], [[float('nan'), 0, 0]], [[1., 2.]]])
def test_rejects_invalid_cloud(points):
    store = KeyframeStore(max_frames=2, max_points=10)
    with pytest.raises(ValueError):
        add(store, pose(1000), points)
    assert not store.segments
