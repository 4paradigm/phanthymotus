"""Rebuild history from local keyframes, one coordinate system at a time."""
import numpy as np
import pytest

from plugins.pointcloud_slam_map import rebuild_segment
from test_pointcloud_slam_stream import odom, stream


def test_historical_points_move_when_optimized_pose_changes():
    s = stream()
    for t, x in [(1000, 0), (1100, 2)]:
        s.odom(odom(t, x), now=t / 1000)
        s.cloud(t * 1_000_000, 'camera', [[1, 0, 0]], now=t / 1000)
    segment = s.store.segments[0]
    original = s.graph()[0]['poses']
    corrected = [p.copy() for p in original]
    corrected[1][0, 3] = 1.5
    assert rebuild_segment(segment, original).tolist() == [[1., 0., 0.], [3., 0., 0.]]
    assert rebuild_segment(segment, corrected).tolist() == [[1., 0., 0.], [2.5, 0., 0.]]
    assert segment.frames[1].points.tolist() == [[1., 0., 0.]]


def test_rejects_wrong_pose_count_or_nonrigid_optimized_pose():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    segment = s.store.segments[0]
    with pytest.raises(ValueError):
        rebuild_segment(segment, [])
    invalid = np.eye(4)
    invalid[0, 0] = 2
    with pytest.raises(ValueError):
        rebuild_segment(segment, [invalid])


def test_two_segments_are_rebuilt_separately():
    s = stream()
    for t, segment, x in [(1000, 'a', 0), (1100, 'b', 10)]:
        s.odom(odom(t, x, segment), now=t / 1000)
        s.cloud(t * 1_000_000, 'camera', [[1, 0, 0]], now=t / 1000)
    maps = [rebuild_segment(segment, graph['poses'])
            for segment, graph in zip(s.store.segments, s.graph())]
    assert len(maps) == 2
    assert maps[0].tolist() == [[1., 0., 0.]]
    assert maps[1].tolist() == [[11., 0., 0.]]
