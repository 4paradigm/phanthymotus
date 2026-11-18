"""The admission gate retains provenance and poses, never map/keyframe history."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins.pointcloud_slam_stream import SlamStream
from plugins.pointcloud_slam_odom import decode_slam_odom


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


def chassis(ms, x=0., segment='a'):
    sample = odom(ms, x, segment)
    sample['vendor']['motus_slam'].update(pose_source='chassis_odometry', pose_reference='body',
                                        source_topic='wheel_pose', stamp_ns=ms*1_000_000)
    return sample


def test_dense_chassis_run_has_one_continuity_token_at_aligned_scan_stamps():
    s = stream()
    for ms, x in [(1000, 0.), (1100, .03), (1200, .06), (1300, .09)]:
        s.odom(chassis(ms, x), now=ms/1000)
    for ms in (1000, 1150, 1250, 1300):
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=1.4)
    tokens = [frame.pose.chassis_continuity for frame in s.accepted]
    assert len(tokens) == 4 and len(tokens[0]) == 16 and len(set(tokens)) == 1


def test_delayed_chassis_delivery_can_leave_an_unposed_scan_inside_a_certified_run():
    unposed = []
    s = stream(max_pending=1, on_unlocalized_frame=lambda stamp, _, pose, *args: unposed.append((stamp, pose)))
    s.odom(chassis(1000), now=1.)
    for ms in (1000, 1100, 1200):
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=ms/1000)
    assert unposed == [(1_100_000_000, None)]
    s.odom(chassis(1250, .04), now=1.25)
    first, resumed = (frame.pose.chassis_continuity for frame in s.accepted)
    assert first and resumed == first


@pytest.mark.parametrize('second', [
    chassis(1200, 2.), chassis(1800), chassis(1200, segment='new-origin')
])
def test_chassis_gap_jump_and_new_origin_do_not_certify_interpolated_pose(second):
    s = stream()
    s.odom(chassis(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[0, 0, 1]], now=1.)
    s.odom(second, now=2.)
    s.cloud(1_100_000_000, 'camera', [[0, 0, 1]], now=2.)
    s.cloud(second['stamp_ms']*1_000_000, 'camera', [[0, 0, 1]], now=2.)
    old, middle, new = (frame.pose.chassis_continuity for frame in s.accepted)
    assert len(old) == len(new) == 16 and old != new
    assert middle is None


def test_missing_or_non_chassis_pose_cannot_extend_the_measured_run():
    s = stream()
    s.odom(chassis(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[0, 0, 1]], now=1.)
    s.odom(dict(chassis(1100), pose=None), now=1.1)
    s.odom(chassis(1200), now=1.2)
    s.cloud(1_200_000_000, 'camera', [[0, 0, 1]], now=1.2)
    s.odom(odom(1300), now=1.3)
    s.cloud(1_300_000_000, 'camera', [[0, 0, 1]], now=1.3)
    before, after, unauthenticated = (frame.pose.chassis_continuity for frame in s.accepted)
    assert before != after and unauthenticated is None


@pytest.mark.parametrize('metadata', [
    {'observation': {'time_basis': 'receipt_estimated', 'normalized_clock_id': 'clock'}},
    {'observation': {'time_basis': 'source_capture', 'normalized_clock_id': 'other'}},
    {'measurement': {'mapping': {'verified': False}}},
    {'measurement': {'stamp_ns': 1_050_000_000}},
])
def test_chassis_timing_contradiction_keeps_numeric_pose_without_continuity(metadata):
    s = stream()
    sample = chassis(1100, .03)
    if 'observation' in metadata:
        metadata = {'observation': dict(metadata['observation'], schema='motus.observation/1',
                                        observation_stamp_ns=1_100_000_000)}
    sample['vendor']['motus_slam'].update(metadata)
    s.odom(sample, now=1.1)
    s.cloud(1_100_000_000, 'camera', [[0, 0, 1]], now=1.1)
    assert s.accepted[0].pose.position == (.03, 0., 0.)
    assert s.accepted[0].pose.chassis_continuity is None


def test_chassis_measurement_epoch_change_starts_a_new_continuity_run():
    s = stream()
    for ms, epoch in ((1000, 'first'), (1100, 'second')):
        sample = chassis(ms)
        sample['vendor']['motus_slam']['measurement'] = {'clock_epoch': epoch}
        s.odom(sample, now=ms/1000)
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=ms/1000)
    tokens = [frame.pose.chassis_continuity for frame in s.accepted]
    assert all(tokens) and tokens[0] != tokens[1]


def test_inertial_prediction_is_only_a_hint_in_the_accepted_origin():
    s = stream()
    sample = odom(1200, .2, status='coasting')
    predicted = sample.pop('pose')
    predicted['covariance'] = (np.eye(6)*.01).ravel().tolist()
    sample['pose'] = None
    vendor = sample['vendor']
    vendor.update(source='rgbd_eskf', fusion=dict(prediction=predicted, translation_integrated=True,
        prediction_stamp_ns=1_200_000_000, prediction_anchor_ns=1_000_000_000, mapping_allowed=False))
    vendor['motus_slam']['stamp_ns'] = 1_200_000_000
    with pytest.raises(ValueError):
        decode_slam_odom(sample)  # startup requires a measured root
    s.odom(sample, now=1.2)
    assert s.status(now=1.2)['odometry'] == 'waiting'
    s.odom(odom(1000), now=1.)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=1.1)
    s.odom(sample, now=1.2)
    assert s.status(now=1.2)['odometry'] == 'coasting'
    assert s.accepted[0].pose.prediction_anchor_ns == 1_000_000_000
    assert s.accepted[0].pose.position == pytest.approx((.1, 0, 0))
    vendor['motus_slam']['segment_id'] = 'another-origin'
    s.odom(sample, now=1.21)
    assert s.status(now=1.21)['odometry'] == 'waiting'


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


def test_rgb_recovery_receives_unbracketed_clouds_when_external_pose_delivery_stops():
    unlocalized = []
    s = stream(on_unlocalized_frame=lambda *args: unlocalized.append(args))
    s.odom(odom(1000), now=1)
    for ms in range(1100, 1600, 100):
        s.cloud(ms*1_000_000, 'camera', [[1,0,3]], now=ms/1000)
    assert [frame[0] for frame in unlocalized] == [1_100_000_000, 1_200_000_000, 1_300_000_000]
    assert all(frame[2] is None for frame in unlocalized)
    assert s.pending_count == 2
    assert not s.accepted
