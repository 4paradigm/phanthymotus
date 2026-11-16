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


def test_numeric_pose_status_and_segment_names_do_not_block_frames():
    s = stream()
    for ms, segment, state in [(1000, 'a', 'tracking'), (1100, 'a', 'lost'),
                                (1200, 'a', 'tracking'), (1300, 'b', 'tracking'),
                                (1400, 'a', 'tracking')]:
        s.odom(odom(ms, segment=segment, status=state), now=ms/1000)
        s.cloud(ms*1_000_000, 'different-name', [[1, 0, 0]], now=10000)
    assert [f.pose.segment_id for f in s.accepted] == ['a', 'a', 'a', 'b', 'a']


def test_bounded_pending_keeps_latest_clouds_without_age_expiry():
    s = stream(max_pending=1)
    s.odom(odom(1000), now=1)
    for ms in [1100, 1200]:
        s.cloud(ms*1_000_000, 'camera', [[1, 0, 0]], now=ms)
    assert s.pending_count == 1 and s.status(now=10000)['dropped_pending'] == 1
    s.odom(odom(1300), now=20000)
    assert [f.cloud_stamp_ns for f in s.accepted] == [1_200_000_000]


def test_bad_numeric_pose_is_pairing_barrier_and_valid_pose_recovers():
    s = stream(); s.odom(odom(1000), now=1)
    bad = odom(1100); bad['pose']['quaternion'] = [0, 0, 0, 0]
    s.odom(bad, now=2)
    s.odom(odom(1200), now=3)
    s.cloud(1_150_000_000, 'camera', [[1, 0, 0]], now=4)
    assert not s.accepted
    s.cloud(1_200_000_000, 'camera', [[1, 0, 0]], now=5)
    assert len(s.accepted) == 1


@pytest.mark.parametrize('wrong', [dict(point_count=0), dict(point_count=4), dict(stamp_ns=True)])
def test_cloud_numeric_format_and_capacity_checks(wrong):
    s = stream(); s.odom(odom(1000), now=1)
    args = dict(stamp_ns=1_000_000_000, frame='camera', cloud=[[1, 0, 0]], now=1)
    args.update(wrong)
    with pytest.raises(ValueError): s.cloud(**args)


@pytest.mark.parametrize('status', ['lost', 'coasting', 'recovering', 'unknown'])
def test_absent_numeric_pose_waits_and_resumes_same_segment(status):
    s = stream(); s.odom(odom(1000), now=1)
    missing = odom(1100, status=status); missing['pose'] = None
    s.odom(missing, now=2)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=3)
    assert not s.accepted and s.pending_count == 1
    s.odom(odom(1200), now=100)
    s.cloud(1_200_000_000, 'camera', [[1, 0, 0]], now=200)
    assert len(s.accepted) == 1
