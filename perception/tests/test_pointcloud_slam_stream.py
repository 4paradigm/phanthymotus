"""The admission gate retains provenance and poses, never map/keyframe history."""
from types import SimpleNamespace as NS

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
    accepted = []
    def receive(stamp, cloud, pose, matrix, context):
        accepted.append(NS(cloud_stamp_ns=stamp, cloud=cloud, pose=pose,
                           child_from_cloud=matrix, capture_context=context))
    options = dict(cloud_frame='camera', child_frame='body', clock_id='clock',
                   child_from_cloud=np.eye(4), max_gap_ms=250, max_age_s=1,
                   max_pending=2, max_cloud_points=3, on_frame=receive)
    options.update(kw)
    result = SlamStream(**options)
    result.accepted = accepted
    return result


def test_bracketing_pose_emits_untouched_cloud_and_exact_nanosecond_stamp():
    s = stream()
    s.odom(odom(1000), now=1.)
    raw = {'per_point_fields': b'xyzrgb-intensity-normal-time'}
    s.cloud(1_100_000_001, 'camera', raw, point_count=1, now=1.1)
    assert not s.accepted
    s.odom(odom(1200, 2), now=1.2)
    frame = s.accepted[0]
    assert frame.pose.position == pytest.approx((1.00000001, 0, 0))
    assert frame.cloud_stamp_ns == 1_100_000_001
    assert frame.cloud == raw
    assert not hasattr(s, 'store') and not hasattr(s, 'graph')


def test_extrinsic_is_capture_evidence_and_never_reprojects_cloud_bytes():
    matrix = np.eye(4); matrix[0, 3] = 2
    s = stream(child_from_cloud=matrix)
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    assert s.accepted[0].cloud == [[1, 0, 0]]
    assert s.accepted[0].child_from_cloud[0, 3] == 2


def test_loss_blocks_old_segment_and_retired_segments_cannot_return():
    s = stream()
    for ms, segment, state in [(1000, 'a', 'tracking'), (1100, 'a', 'lost'),
                                (1200, 'a', 'tracking'), (1300, 'b', 'tracking'),
                                (1400, 'a', 'tracking')]:
        s.odom(odom(ms, segment=segment, status=state), now=ms/1000)
        if state == 'tracking':
            s.cloud(ms*1_000_000, 'camera', [[1, 0, 0]], now=ms/1000)
    assert [f.pose.segment_id for f in s.accepted] == ['a', 'b']
    assert 'segment' in s.last_error


def test_same_segment_cannot_change_pose_frame_or_clock():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    changed = odom(1100); changed['pose']['frame_id'] = 'different_origin'
    s.odom(changed, now=1.1)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1)
    assert len(s.accepted) == 1
    assert s.status(now=1.1)['integration'] == 'waiting_for_new_segment'


def test_stale_inputs_and_bounded_pending_never_reach_runtime():
    s = stream(max_pending=1)
    s.odom(odom(1000), now=1.)
    for ms in [1100, 1200]:
        s.cloud(ms*1_000_000, 'camera', [[1, 0, 0]], now=ms/1000)
    assert s.pending_count == 1 and s.status(now=1.2)['dropped_pending'] == 1
    with pytest.raises(ValueError, match='stale'):
        s.cloud(1_300_000_000, 'camera', [[1, 0, 0]], now=3.)
    assert not s.accepted and s.pending_count == 0


def test_measurement_age_rejects_old_future_and_expired_pending_frames():
    s = stream(max_measurement_age_ms=150, max_future_skew_ms=20)
    s.odom(odom(1000), now=1., measurement_now_ns=1_000_000_000)
    with pytest.raises(ValueError, match='timestamp'):
        s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.,
                measurement_now_ns=1_000_000_000)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1,
            measurement_now_ns=1_100_000_000)
    s.odom(odom(1200), now=1.3, measurement_now_ns=1_300_000_000)
    assert not s.accepted and s.pending_count == 0
    assert s.status(now=2., measurement_now_ns=2_000_000_000)['odometry'] == 'stale'


def test_bad_tracking_is_pairing_barrier_without_poisoning_valid_segment():
    s = stream()
    s.odom(odom(1000), now=1.)
    bad = odom(1100); bad['pose']['quaternion'] = [0, 0, 0, 0]
    s.odom(bad, now=1.1)
    s.odom(odom(1200), now=1.2)
    s.cloud(1_150_000_000, 'camera', [[1, 0, 0]], now=1.2)
    assert not s.accepted
    s.cloud(1_200_000_000, 'camera', [[2, 0, 0]], now=1.2)
    assert len(s.accepted) == 1


def test_duplicates_and_late_odometry_do_not_refresh_receipt_age():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.odom(odom(900), now=1.5)
    s.odom(odom(1000), now=1.8)
    assert s.status(now=2.1)['odometry'] == 'stale'


@pytest.mark.parametrize('wrong', [dict(frame='wrong'), dict(point_count=0), dict(point_count=4)])
def test_cloud_frame_and_capacity_checks(wrong):
    s = stream(); s.odom(odom(1000), now=1.)
    args = dict(stamp_ns=1_000_000_000, frame='camera', cloud=[[1, 0, 0]], now=1.)
    args.update(wrong)
    with pytest.raises(ValueError):
        s.cloud(**args)


def paused(ms, status='coasting', segment='a'):
    sample = odom(ms, segment=segment, status=status)
    sample['pose'] = None
    sample['vendor']['fusion'] = dict(enabled=True, status=status, mapping_allowed=False,
                                      last_visual_ns=1_000_000_000)
    return sample


def test_explicit_short_fusion_pause_blocks_clouds_then_resumes_same_segment():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(paused(1100), now=1.1)
    with pytest.raises(ValueError):
        s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1)
    s.odom(paused(1200, 'recovering'), now=1.2)
    s.odom(odom(1300), now=1.3)
    s.cloud(1_300_000_000, 'camera', [[1, 0, 0]], now=1.3)
    assert len(s.accepted) == 2
    assert s.status(now=1.3)['integration'] == 'admitting'


@pytest.mark.parametrize('sample, resume_ms', [(paused(1100, segment='other'), 1300),
                                              (paused(2100), 2200), (paused(1100), 2101)])
def test_fusion_pause_cannot_bypass_segment_identity_or_one_second_limit(sample, resume_ms):
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(sample, now=sample['stamp_ms']/1000)
    s.odom(odom(resume_ms), now=resume_ms/1000)
    s.cloud(resume_ms*1_000_000, 'camera', [[1, 0, 0]], now=resume_ms/1000)
    assert len(s.accepted) == 1


def gate_paused(ms, status='recovering', segment='a', **pause):
    sample = odom(ms, segment=segment, status=status)
    sample['pose'] = None
    sample['vendor']['pause'] = dict(dict(source='visual_gate', status=status, mapping_allowed=False), **pause)
    return sample


def test_explicit_visual_gate_pause_resumes_same_segment_like_fusion_pause():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(gate_paused(1100), now=1.1)
    with pytest.raises(ValueError):
        s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1)
    s.odom(odom(1300), now=1.3)
    s.cloud(1_300_000_000, 'camera', [[1, 0, 0]], now=1.3)
    assert len(s.accepted) == 2 and s.status(now=1.3)['integration'] == 'admitting'


@pytest.mark.parametrize('sample, resume_ms', [
    (gate_paused(1100, mapping_allowed=True), 1300),
    (gate_paused(1100, status='recovering', segment='other'), 1300),
    (dict(gate_paused(1100), vendor=dict(gate_paused(1100)['vendor'],
          pause=dict(source='visual_gate', status='coasting', mapping_allowed=False))), 1300),
    (gate_paused(2100), 2200),
    (gate_paused(1100), 2101),
])
def test_visual_gate_pause_has_identical_bounds(sample, resume_ms):
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    s.odom(sample, now=sample['stamp_ms']/1000)
    s.odom(odom(resume_ms), now=resume_ms/1000)
    s.cloud(resume_ms*1_000_000, 'camera', [[1, 0, 0]], now=resume_ms/1000)
    assert len(s.accepted) == 1


def test_undeclared_pause_still_requires_a_new_segment():
    s = stream()
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)
    undeclared = odom(1100, status='recovering'); undeclared['pose'] = None
    s.odom(undeclared, now=1.1)
    s.odom(odom(1300), now=1.3)
    s.cloud(1_300_000_000, 'camera', [[1, 0, 0]], now=1.3)
    assert len(s.accepted) == 1
