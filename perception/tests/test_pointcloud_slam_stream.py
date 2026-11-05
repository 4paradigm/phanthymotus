"""Exercise live-input state without a ROS installation or fabricated clocks."""
import numpy as np
import pytest

from plugins.pointcloud_slam_stream import SlamStream


def odom(ms, x=0, segment='a', status='tracking'):
    return {'schema': 'motus.odom/1', 'units': {'length': 'm'}, 'stamp_ms': ms,
            'pose': {'position': [x, 0, 0], 'quaternion': [0, 0, 0, 1],
                     'frame_id': 'odom', 'child_frame_id': 'body', 'covariance': None},
            'vendor': {'motus_slam': {'status': status, 'segment_id': segment,
                        'clock_id': 'clock', 'stamp_source': 'measurement'}}}


def stream(**kw):
    options = dict(cloud_frame='camera', child_frame='body', clock_id='clock',
                   child_from_cloud=np.eye(4), max_gap_ms=250, max_age_s=1,
                   max_pending=2, max_cloud_points=3, max_frames=4, max_points=10)
    options.update(kw)
    return SlamStream(**options)


def test_input_waits_for_bracketing_pose_and_stores_cloud_with_original_stamp():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_100_000_000, 'camera', [[1, 2, 3]], now=1.1)
    assert not s.store.segments
    s.odom(odom(1200, 2), now=1.2)
    frame = s.store.segments[0].frames[0]
    assert frame.pose.position == pytest.approx((1, 0, 0))
    assert frame.cloud_stamp_ns == 1_100_000_000
    assert frame.points.tolist() == [[1., 2., 3.]]


def test_calibrated_extrinsic_is_applied_before_storing_local_cloud():
    extrinsic = np.eye(4)
    extrinsic[0, 3] = 2
    s = stream(child_from_cloud=extrinsic)
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    assert s.store.segments[0].frames[0].points.tolist() == [[3., 0., 0.]]


def test_loss_discards_pending_and_recovery_starts_separate_graph():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.cloud(1_100_000_000, 'camera', [[2, 0, 0]], now=1.1)
    s.odom(odom(1100, status='lost'), now=1.1)
    s.odom(odom(1200, segment='b'), now=1.2)
    s.cloud(1_200_000_000, 'camera', [[3, 0, 0]], now=1.2)
    graph = s.graph()
    assert [(g['segment_id'], len(g['poses']), len(g['edges'])) for g in graph] == [
        ('a', 1, 0), ('b', 1, 0)]


def test_graph_uses_relative_odom_without_cross_segment_edge():
    s = stream()
    for t, x in [(1000, 0), (1100, 2)]:
        s.odom(odom(t, x), now=t/1000)
        s.cloud(t * 1_000_000, 'camera', [[1, 0, 0]], now=t/1000)
    edge = s.graph()[0]['edges'][0]
    assert edge['from'] == 0 and edge['to'] == 1
    assert edge['transform'][:3, 3] == pytest.approx([2, 0, 0])


def test_stale_lost_wrong_frame_and_capacity_never_fuse():
    s = stream(max_pending=1, max_cloud_points=1)
    s.odom(odom(1000), now=1.)
    with pytest.raises(ValueError):
        s.cloud(1_000_000_000, 'other', [[1, 0, 0]], now=1.)
    with pytest.raises(ValueError):
        s.cloud(1_000_000_000, 'camera', [[1, 0, 0], [2, 0, 0]], now=1.)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1)
    s.cloud(1_200_000_000, 'camera', [[1, 0, 0]], now=1.2)
    assert s.pending_count == 1
    with pytest.raises(ValueError, match='stale'):
        s.cloud(1_300_000_000, 'camera', [[1, 0, 0]], now=3.)
    assert not s.store.segments and s.pending_count == 0


def test_repeated_identical_odom_does_not_refresh_freshness_or_break_segment():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(odom(1000), now=1.5)
    s.odom(odom(1100, 1), now=1.6)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.6)
    assert len(s.store.segments[0].frames) == 2
    s.odom(odom(1100, 1), now=2.5)
    with pytest.raises(ValueError, match='stale'):
        s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=2.7)


def test_full_store_exposes_reason_instead_of_reporting_success():
    s = stream(max_frames=1)
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(odom(1100), now=1.1)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1)
    assert 'capacity' in s.last_error
    assert s.status(now=1.1)['integration'] == 'capacity_reached'


def test_measurement_age_rejects_old_and_future_data_even_when_received_now():
    s = stream(max_measurement_age_ms=200, max_future_skew_ms=20)
    s.odom(odom(1000), now=3., measurement_now_ns=3_000_000_000)
    with pytest.raises(ValueError, match='stale'):
        s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=3.,
                measurement_now_ns=3_000_000_000)
    s.odom(odom(3000), now=3., measurement_now_ns=3_000_000_000)
    with pytest.raises(ValueError, match='timestamp'):
        s.cloud(3_030_000_000, 'camera', [[1, 0, 0]], now=3.,
                measurement_now_ns=3_000_000_000)
    s.cloud(3_000_000_000, 'camera', [[1, 0, 0]], now=3.,
            measurement_now_ns=3_000_000_000)
    assert len(s.store.segments[0].frames) == 1


def test_measurement_clock_is_required_when_age_checks_are_enabled():
    s = stream(max_measurement_age_ms=200, max_future_skew_ms=20)
    s.odom(odom(1000), now=1.)
    assert 'measurement clock' in s.last_error
    with pytest.raises(ValueError):
        s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)


def test_pending_cloud_expires_by_measurement_time_before_next_pose():
    s = stream(max_measurement_age_ms=150, max_future_skew_ms=20)
    s.odom(odom(1000), now=1., measurement_now_ns=1_000_000_000)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1,
            measurement_now_ns=1_100_000_000)
    s.odom(odom(1200), now=1.3, measurement_now_ns=1_300_000_000)
    assert not s.store.segments and s.pending_count == 0


def test_status_exposes_bounded_queue_and_expires_odom_without_new_messages():
    s = stream(max_pending=1)
    assert s.status(now=0.)['odometry'] == 'waiting'
    s.odom(odom(1000), now=1.)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1)
    s.cloud(1_200_000_000, 'camera', [[1, 0, 0]], now=1.2)
    status = s.status(now=1.2)
    assert status['odometry'] == 'tracking'
    assert status['pending_clouds'] == 1 and status['dropped_pending'] == 1
    assert s.status(now=3.)['odometry'] == 'stale'
    assert s.pending_count == 1  # status is a read-only query
    with pytest.raises(ValueError, match='stale'):
        s.cloud(1_300_000_000, 'camera', [[1, 0, 0]], now=3.)
    assert s.pending_count == 0


def test_tracking_after_loss_does_not_claim_integration_resumed_in_old_segment():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(odom(1100, status='lost'), now=1.1)
    s.odom(odom(1200), now=1.2)
    status = s.status(now=1.2)
    assert status['odometry'] == 'tracking'
    assert status['integration'] == 'waiting_for_new_segment'


def test_explicit_segment_frame_template_accepts_reset_without_merging_history():
    s = stream(child_frame='{segment_id}_base')
    first = odom(1000, segment='run-a')
    first['pose']['child_frame_id'] = 'run-a_base'
    s.odom(first, now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(odom(1100, status='lost'), now=1.1)
    second = odom(1200, segment='run-b')
    second['pose']['child_frame_id'] = 'run-b_base'
    s.odom(second, now=1.2)
    s.cloud(1_200_000_000, 'camera', [[2, 0, 0]], now=1.2)
    assert [part.segment_id for part in s.store.segments] == ['run-a', 'run-b']
    assert [len(part.frames) for part in s.store.segments] == [1, 1]


def test_segment_frame_template_rejects_wrong_physical_reference():
    s = stream(child_frame='{segment_id}_base')
    value = odom(1000, segment='run-a')
    value['pose']['child_frame_id'] = 'other_base'
    s.odom(value, now=1.)
    assert 'child frame mismatch' in s.last_error
    assert not s.store.segments


def test_malformed_tracking_sample_is_barrier_but_does_not_lock_valid_segment():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    malformed = odom(1100)
    malformed['pose']['quaternion'] = [0, 0, 0, 0]
    s.odom(malformed, now=1.1)
    s.odom(odom(1200), now=1.2)
    s.cloud(1_150_000_000, 'camera', [[9, 0, 0]], now=1.2)
    assert len(s.store.segments[0].frames) == 1  # no interpolation through the bad sample
    s.cloud(1_200_000_000, 'camera', [[2, 0, 0]], now=1.2)
    assert len(s.store.segments[0].frames) == 2


def test_late_odom_is_ignored_without_refreshing_or_locking_segment():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(odom(900), now=1.1)
    s.odom(odom(1100), now=1.2)
    s.cloud(1_100_000_000, 'camera', [[2, 0, 0]], now=1.2)
    assert len(s.store.segments[0].frames) == 2


def test_cloud_freshness_requires_new_capture_not_repeated_frame():
    s = stream()
    s.odom(odom(1000), now=1.)
    assert s.status(now=1.)['cloud'] == 'waiting'
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    assert s.status(now=1.)['cloud'] == 'active'
    with pytest.raises(ValueError, match='timestamp'):
        s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.5)
    s.odom(odom(2100), now=2.1)
    assert s.status(now=2.1)['cloud'] == 'stale'
