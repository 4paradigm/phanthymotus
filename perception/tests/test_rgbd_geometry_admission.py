"""Delayed depth observations retain capture identity and can authorize mapping."""
from copy import deepcopy

import numpy as np
import pytest

from plugins.rgbd_odometry_runtime import CaptureObservations
from plugins.pointcloud_slam_odom import decode_slam_odom
from test_rgbd_fusion import warmed, source, images
from test_rgbd_geometry_async import real_result


class Runner:
    def __init__(self, result):
        self.result, self.ready, self.calls = result, False, 0
    def submit(self, token, *args):
        self.token = token
        self.ready = False
        self.calls += 1
        return True
    def take(self, token):
        assert token == self.token
        return ('done', (self.result, None)) if self.ready else ('pending', None)


@pytest.fixture
def setup():
    fusion, stamp = warmed()
    result, depth = real_result()
    fusion.geometry = object()  # runner supplies an independently rendered registration
    runner = fusion.geometry_runner = Runner(result)
    return fusion, runner, CaptureObservations(fusion), stamp, images()[0], depth


def feed(fusion, start, end):
    for t in range(start+5_000_000, end+1, 5_000_000):
        fusion.observe_imu(source(t), t)


def submit(queue, stamp, color, depth):
    # Explicit native rejection resolves the join without a synthetic visual pose.
    from test_rgbd_odometry import messages
    queue.add_frame(stamp, color, depth, stamp)
    _, info = messages(stamp)
    info.lost = True
    return queue.result('info', info, stamp)


def test_pending_geometry_keeps_capture_order_without_rewriting_later_frames(setup):
    fusion, runner, queue, start, color, depth = setup
    first, second = start+50_000_000, start+100_000_000
    feed(fusion, start, second)
    assert submit(queue, first, color, depth) == []
    job = fusion.geometry_job
    assert submit(queue, second, color, depth) == []
    assert fusion.filter.stamp == first
    assert fusion.geometry_job is job and runner.calls == 1
    runner.ready = True
    out = queue.drain(second+10_000_000)
    assert len(out) == 1
    assert out[0]['vendor']['motus_slam']['stamp_ns'] == first
    assert decode_slam_odom(out[0]).stamp_ms == first//1_000_000
    assert runner.calls == 2 and fusion.filter.stamp == second
    # No repeat of the already published first capture.
    assert queue.drain(second+20_000_000) == []


def test_three_distinct_geometry_observations_restore_mapping_without_extending_visual_anchor(setup):
    fusion, runner, queue, stamp, color, depth = setup
    anchor = deepcopy(fusion.reference)
    fusion.good, fusion.state = 0, 'coasting'
    for i in range(3):
        next_stamp = stamp+50_000_000
        feed(fusion, stamp, next_stamp)
        assert submit(queue, next_stamp, color, depth) == []
        before = fusion.good
        assert queue.drain(next_stamp+5_000_000) == []
        assert fusion.good == before
        runner.ready = True
        out = queue.drain(next_stamp+10_000_000)
        assert len(out) == 1
        assert (out[0]['pose'] is not None) is (i == 2)
        assert out[0]['vendor']['fusion']['mapping_allowed'] is (i == 2)
        assert queue.drain(next_stamp+11_000_000) == []
        stamp = next_stamp
    assert fusion.last_visual == anchor[0]
    assert fusion.reference[0] == anchor[0]
    np.testing.assert_array_equal(fusion.reference[1], anchor[1])
    assert fusion.last_tracking[0] == stamp
    assert fusion.tracking_offset() is None  # no native pose at this geometry-only capture


def test_geometry_deadline_is_not_extended_by_waiting_frames(setup):
    fusion, runner, queue, start, color, depth = setup
    first, second = start+50_000_000, start+100_000_000
    feed(fusion, start, second)
    submit(queue, first, color, depth)
    deadline = fusion.geometry_job['deadline']
    submit(queue, second, color, depth)
    assert fusion.geometry_job['deadline'] == deadline
    out = queue.drain(deadline+1)
    assert out[0]['vendor']['motus_slam']['stamp_ns'] == first
    assert out[0]['pose'] is None
    assert next(frame for frame in fusion.frames if frame['capture_ns'] == first)['geometry'] == 'timeout_discarded'


def test_geometry_failure_breaks_recovery_evidence(setup):
    fusion, runner, queue, start, color, depth = setup
    fusion.good, fusion.state = 2, 'recovering'
    stamp = start+50_000_000
    feed(fusion, start, stamp)
    submit(queue, stamp, color, depth)
    runner.result, runner.ready = None, True
    out = queue.drain(stamp+10_000_000)
    assert fusion.good == 0 and out[0]['pose'] is None


def test_geometry_cannot_revive_lost_segment_or_extend_visual_deadline(setup):
    fusion, runner, queue, start, color, depth = setup
    stamp = start+50_000_000
    feed(fusion, start, stamp)
    submit(queue, stamp, color, depth)
    runner.ready = True
    # Sensor activity cannot renew the original visual-anchor lifetime.
    feed(fusion, stamp, start+1_010_000_000)
    out = queue.drain(start+1_010_000_000)
    assert fusion.state == 'lost' and out[0]['pose'] is None
    assert fusion.last_visual == start


def test_slam_admits_original_cloud_after_delayed_geometry_measurement(setup):
    from test_pointcloud_slam_stream import stream
    fusion, runner, queue, start, color, depth = setup
    slam = stream(cloud_frame='optical', child_frame='optical', clock_id='capture-clock')
    slam.odom(fusion.output(start), now=start/1e9)
    stamp = start+50_000_000
    feed(fusion, start, stamp)
    assert submit(queue, stamp, color, depth) == []
    cloud = [[1., 0., 2.]]
    slam.cloud(stamp, 'optical', cloud, now=(stamp+10_000_000)/1e9)
    assert slam.accepted == []
    runner.ready = True
    for out in queue.drain(stamp+100_000_000):
        slam.odom(out, now=(stamp+100_000_000)/1e9)
    assert len(slam.accepted) == 1
    assert slam.accepted[0].cloud_stamp_ns == stamp
    assert slam.accepted[0].cloud == cloud


def test_slam_replays_pending_capture_after_three_geometry_recoveries(setup):
    from test_pointcloud_slam_stream import stream
    fusion, runner, queue, stamp, color, depth = setup
    slam = stream(cloud_frame='optical', child_frame='optical', clock_id='capture-clock')
    slam.odom(fusion.output(stamp), now=stamp/1e9)
    fusion.good, fusion.state = 0, 'coasting'
    slam.odom(fusion.output(stamp), now=stamp/1e9)
    for i in range(4):
        next_stamp = stamp+50_000_000
        feed(fusion, stamp, next_stamp)
        assert submit(queue, next_stamp, color, depth) == []
        slam.cloud(next_stamp, 'optical', [[1., 0., 2.]], now=(next_stamp+5_000_000)/1e9)
        if i < 3:
            assert not slam.accepted and slam.pending_count == 1
        runner.ready = True
        out = queue.drain(next_stamp+10_000_000)
        assert len(out) == 1
        slam.odom(out[0], now=(next_stamp+10_000_000)/1e9)
        assert len(slam.accepted) == max(0, i-1)
        stamp = next_stamp
    assert slam.accepted[-1].cloud_stamp_ns == stamp


@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_delayed_geometry_obeys_exact_existing_deadline(setup, offset):
    fusion, runner, queue, start, color, depth = setup
    stamp = start+50_000_000
    feed(fusion, start, stamp)
    submit(queue, stamp, color, depth)
    deadline = fusion.geometry_job['deadline']
    runner.ready = True
    out = queue.drain(deadline+offset)
    assert len(out) == 1
    assert (out[0]['pose'] is not None) is (offset <= 0)
    assert out[0]['vendor']['motus_slam']['stamp_ns'] == stamp
