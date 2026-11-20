"""The admission gate retains provenance and poses, never map/keyframe history."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins.pointcloud_slam_stream import SlamStream
from plugins.pointcloud_slam_odom import decode_slam_odom
from plugins.spatial_time import FrameContext


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


def measured_rgbd(ms, x=0., segment='a', ingress='local'):
    sample = odom(ms, x, segment)
    stamp = ms*1_000_000
    parent = FrameContext(f'pair-{ms}', stamp, 'clock', 'source_capture', stamp, stamp, ingress,
                          transformations=('rgbd_association',))
    sample['pose']['covariance'] = (np.eye(6)*.01).ravel().tolist()
    sample['vendor'].update(source='rtabmap_rgbd_odometry', pose_reference='camera_optical',
                            registration=dict(status='tracking', native_lost=False, native_inliers=64))
    sample['vendor']['motus_slam'].update(stamp_ns=stamp, observation=parent.to_dict())
    return sample, parent


def test_dense_chassis_run_has_one_continuity_token_at_aligned_scan_stamps():
    s = stream()
    for ms, x in [(1000, 0.), (1100, .03), (1200, .06), (1300, .09)]:
        s.odom(chassis(ms, x), now=ms/1000)
    for ms in (1000, 1150, 1250, 1300):
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=1.4)
    tokens = [frame.pose.external_continuity for frame in s.accepted]
    assert len(tokens) == 4 and len(tokens[0]) == 16 and len(set(tokens)) == 1


def test_delayed_chassis_delivery_can_leave_an_unposed_scan_inside_a_certified_run():
    unposed = []
    s = stream(max_pending=1, on_unlocalized_frame=lambda stamp, _, pose, *args: unposed.append((stamp, pose)))
    s.odom(chassis(1000), now=1.)
    for ms in (1000, 1100, 1200):
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=ms/1000)
    assert unposed == [(1_100_000_000, None)]
    s.odom(chassis(1250, .04), now=1.25)
    first, resumed = (frame.pose.external_continuity for frame in s.accepted)
    assert first and resumed == first


@pytest.mark.parametrize('second', [
    chassis(1200, 2.), chassis(1800)
])
def test_chassis_gap_and_jump_do_not_certify_interpolated_pose(second):
    s = stream()
    s.odom(chassis(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[0, 0, 1]], now=1.)
    s.odom(second, now=2.)
    s.cloud(1_100_000_000, 'camera', [[0, 0, 1]], now=2.)
    s.cloud(second['stamp_ms']*1_000_000, 'camera', [[0, 0, 1]], now=2.)
    old, middle, new = (frame.pose.external_continuity for frame in s.accepted)
    assert len(old) == len(new) == 16 and old != new
    assert middle is None


@pytest.mark.parametrize('stamp_source', ['measurement', None])
def test_cloud_between_declared_chassis_origins_has_no_interpolated_pose(stamp_source):
    unposed = []
    s = stream(on_unlocalized_frame=lambda stamp, *args: unposed.append(stamp))
    s.odom(chassis(1000, 2.4, 'old'), now=1.)
    s.cloud(1_100_000_000, 'camera', [[0., 0., 1.]], now=1.1)
    reset = chassis(1200, 0., 'reset')
    reset['vendor']['motus_slam']['stamp_source'] = stamp_source
    s.odom(reset, now=1.2)
    assert not s.accepted
    assert unposed == [1_100_000_000]


def test_missing_or_non_chassis_pose_cannot_extend_the_measured_run():
    s = stream()
    s.odom(chassis(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[0, 0, 1]], now=1.)
    s.odom(dict(chassis(1100), pose=None), now=1.1)
    s.odom(chassis(1200), now=1.2)
    s.cloud(1_200_000_000, 'camera', [[0, 0, 1]], now=1.2)
    s.odom(odom(1300), now=1.3)
    s.cloud(1_300_000_000, 'camera', [[0, 0, 1]], now=1.3)
    before, after, unauthenticated = (frame.pose.external_continuity for frame in s.accepted)
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
    assert s.accepted[0].pose.external_continuity is None


@pytest.mark.parametrize('quality', [
    None,
    {'clock_id': 'another-label', 'stamp_source': 'receipt_estimated'},
    {'measurement': {'mapping': {'verified': False}}},
])
def test_static_cloud_brackets_numeric_poses_without_matching_provenance(quality):
    s = stream(on_unlocalized_frame=lambda *args: pytest.fail('cloud lost its numeric pose'))
    s.odom(odom(1000), now=1.)
    s.cloud(1_100_000_000, 'camera', [[0, 0, 1]], now=1.1)
    later = odom(1200, .2)
    if quality is None:
        del later['vendor']
    else:
        later['vendor']['motus_slam'].update(quality)
    s.odom(later, now=1.3)
    assert s.accepted[0].pose.position == pytest.approx((.1, 0., 0.))
    assert s.accepted[0].pose.external_continuity is None


def test_chassis_measurement_epoch_change_starts_a_new_continuity_run():
    s = stream()
    for ms, epoch in ((1000, 'first'), (1100, 'second')):
        sample = chassis(ms)
        sample['vendor']['motus_slam']['measurement'] = {'clock_epoch': epoch}
        s.odom(sample, now=ms/1000)
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=ms/1000)
    tokens = [frame.pose.external_continuity for frame in s.accepted]
    assert all(tokens) and tokens[0] != tokens[1]


@pytest.mark.parametrize('second,loss', [
    ((2501, .1), False), ((1100, 1.), False), ((2201, 3.), False), ((1100, .1), True),
    ((900, 0.), False),
])
def test_measured_visual_break_cannot_extend_previous_token(second, loss):
    s = stream()
    s.ingress_clock_id = 'local'

    def push(ms, x):
        sample, parent = measured_rgbd(ms, x)
        context = replace(parent, observation_id=f'derived-{ms}',
                          parent_observation_ids=(parent.observation_id,),
                          transformations=parent.transformations+('rgbd_odometry',))
        s.odom(sample, now=ms/1000, context=context)
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=ms/1000)

    push(1000, 0.)
    if loss:
        s.odom(dict(odom(1020, status='lost'), pose=None), now=1.02)
    push(*second)
    first, changed = [frame.pose.external_continuity for frame in s.accepted]
    assert first and changed and changed != first


@pytest.mark.parametrize('gap_ms,x', [(1068, .673), (1201, .752)])
def test_witnessed_vo_retains_token_across_brief_capture_gap(gap_ms, x):
    s = stream()
    s.ingress_clock_id = 'local'
    for ms, position in ((1000, 0.), (1000+gap_ms, x)):
        sample, parent = measured_rgbd(ms, position)
        context = replace(parent, observation_id=f'derived-{ms}',
                          parent_observation_ids=(parent.observation_id,),
                          transformations=parent.transformations+('rgbd_odometry',))
        s.odom(sample, now=ms/1000, context=context)
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=ms/1000)
    first, resumed = [frame.pose.external_continuity for frame in s.accepted]
    assert first and resumed == first


def _witnessed(s, ms, x, **extra):
    sample, parent = measured_rgbd(ms, x)
    sample['vendor'].update(extra)
    context = replace(parent, observation_id=f'derived-{ms}',
                      parent_observation_ids=(parent.observation_id,),
                      transformations=parent.transformations+('rgbd_odometry',))
    s.odom(sample, now=ms/1000, context=context)
    s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=ms/1000)


def test_unpaired_scan_does_not_interrupt_live_visual_pose_run():
    unposed = []
    s = stream(max_age_s=2, on_unlocalized_frame=lambda stamp, *args: unposed.append(stamp))
    s.ingress_clock_id = 'local'
    _witnessed(s, 1000, 0.)
    # One scan's capture is ahead of the live pose stream. Its bounded wait
    # expires while independently measured poses from the old run still arrive.
    s.cloud(3_200_000_000, 'camera', [[0, 0, 1]], now=1.1)
    _witnessed(s, 2100, .03)
    s.poll(now=3.11)
    assert not unposed
    assert s.status(now=3.11)['dropped_pending'] == 1
    first, latest = (frame.pose.external_continuity for frame in s.accepted)
    assert first and first == latest
    # A real pose outage must still hand an available scan to geometry.
    s.cloud(3_300_000_000, 'camera', [[0, 0, 1]], now=3.2)
    s.poll(now=3.61)
    assert unposed == [3_300_000_000]


@pytest.mark.parametrize('right_arrival, expected', [(2.569, True), (3.001, False)])
def test_live_visual_run_waits_for_measured_bracket_within_cloud_lifetime(right_arrival, expected):
    unposed = []
    s = stream(max_age_s=2, on_unlocalized_frame=lambda stamp, *args: unposed.append(stamp))
    s.ingress_clock_id = 'local'

    def receive(ms, now, x, ingress):
        sample, parent = measured_rgbd(ms, x)
        context = replace(parent, observation_id=f'pose-{ms}',
                          parent_observation_ids=(parent.observation_id,),
                          transformations=parent.transformations+('rgbd_odometry',))
        s.odom(sample, now=now, received=ingress, context=context)

    # Relative capture and receipt intervals from the first Tianyi outage.
    receive(900, .9, 0., .2)
    s.cloud(900_000_000, 'camera', [[0, 0, 1]], now=.91)
    s.cloud(1_734_000_000, 'camera', [[0, 0, 1]], now=1.)
    receive(1000, 1.209, 0., .36)
    s.poll(now=2.501)
    receive(2402, right_arrival, .5, 2.05)
    assert not unposed
    if not expected:
        assert len(s.accepted) == 1
        return
    first, resumed = s.accepted
    assert first.pose.external_continuity == resumed.pose.external_continuity
    assert resumed.pose.position[0] == pytest.approx(.5 * 734 / 1402)


def test_stale_visual_result_does_not_extend_pose_liveness():
    unposed = []
    s = stream(max_age_s=2, on_unlocalized_frame=lambda stamp, *args: unposed.append(stamp))
    s.ingress_clock_id = 'local'
    _witnessed(s, 1000, 0.)
    s.cloud(1_200_000_000, 'camera', [[0, 0, 1]], now=2.2)
    assert not unposed and s.pending_count == 1
    sample, parent = measured_rgbd(1300, .05)
    context = replace(parent, observation_id='late-result',
                      parent_observation_ids=(parent.observation_id,),
                      transformations=parent.transformations+('rgbd_odometry',))
    s.odom(sample, now=2.3, received=.2, context=context)
    assert unposed == [1_200_000_000]
    assert len(s.accepted) == 1


def relocalized(anchor_ms, **changes):
    proof = dict(method='native_f2m_landmarks', anchor_stamp_ns=anchor_ms*1_000_000,
                 shared_landmarks=44, outage_s=4.2, extended=True)
    proof.update(changes)
    return dict(relocalization=proof)


def test_measured_f2m_relocalization_resumes_the_suspended_token_after_a_pause():
    s = stream()
    s.ingress_clock_id = 'local'
    _witnessed(s, 1000, 0.)
    _witnessed(s, 1200, .05)
    for ms in (1500, 2600, 3900):
        s.odom(dict(odom(ms, status='recovering'), pose=None), now=ms/1000)
    assert s.continuity_token is None
    _witnessed(s, 5400, .12, **relocalized(1200))
    _witnessed(s, 5700, .14)
    tokens = [frame.pose.external_continuity for frame in s.accepted]
    assert tokens[0] and tokens == [tokens[0]]*4


@pytest.mark.parametrize('changes', [
    dict(method='other'), dict(shared_landmarks=19), dict(anchor_stamp_ns=1_100_000_000),
    dict(anchor_stamp_ns=2_800_000_000)])
def test_unverified_or_misanchored_relocalization_starts_a_new_token(changes):
    s = stream()
    s.ingress_clock_id = 'local'
    _witnessed(s, 1000, 0.)
    _witnessed(s, 1200, .05)
    s.odom(dict(odom(1500, status='recovering'), pose=None), now=1.5)
    _witnessed(s, 5400, .12, **relocalized(1200, **changes))
    first, same, resumed = [frame.pose.external_continuity for frame in s.accepted]
    assert first and same == first and resumed != first


@pytest.mark.parametrize('ms,x', [(5701, .12), (5400, 7.), (3000, .1)])
def test_relocalization_outside_the_witnessed_gap_or_motion_bound_starts_a_new_token(ms, x):
    s = stream()
    s.ingress_clock_id = 'local'
    _witnessed(s, 1000, 0.)
    _witnessed(s, 1200, .05)
    s.odom(dict(odom(1500, status='recovering'), pose=None), now=1.5)
    if ms == 3000:
        # A new segment origin can never carry the suspended token.
        sample, parent = measured_rgbd(ms, x, segment='b')
        sample['vendor'].update(relocalized(1200))
        context = replace(parent, observation_id=f'derived-{ms}',
                          parent_observation_ids=(parent.observation_id,),
                          transformations=parent.transformations+('rgbd_odometry',))
        s.odom(sample, now=ms/1000, context=context)
        s.cloud(ms*1_000_000, 'camera', [[0, 0, 1]], now=ms/1000)
    else:
        _witnessed(s, ms, x, **relocalized(1200))
    first, same, resumed = [frame.pose.external_continuity for frame in s.accepted]
    assert first and same == first and resumed != first


def test_suspended_token_is_consumed_by_the_first_measured_pose_only():
    s = stream()
    s.ingress_clock_id = 'local'
    _witnessed(s, 1000, 0.)
    s.odom(dict(odom(1500, status='recovering'), pose=None), now=1.5)
    _witnessed(s, 3000, .05)  # resumed without proof: new token
    s.odom(dict(odom(3200, status='recovering'), pose=None), now=3.2)
    _witnessed(s, 4000, .06, **relocalized(1000))  # proof names the first run, not the second
    first, second, third = [frame.pose.external_continuity for frame in s.accepted]
    assert len({first, second, third}) == 3


@pytest.mark.parametrize('sigma,accepted', [(.24, True), (.25, True), (.251, False)])
def test_inertial_prediction_is_only_a_bounded_hint_in_the_accepted_origin(sigma, accepted):
    s = stream()
    sample = odom(1200, .2, status='coasting')
    predicted = sample.pop('pose')
    predicted['covariance'] = (np.eye(6)*sigma**2).ravel().tolist()
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
    if not accepted:
        assert s.status(now=1.2)['odometry'] == 'waiting'
        assert not s.accepted
        return
    assert s.status(now=1.2)['odometry'] == 'coasting'
    assert s.accepted[0].pose.prediction_anchor_ns == 1_000_000_000
    assert s.accepted[0].pose.position == pytest.approx((.1, 0, 0))
    vendor['motus_slam']['segment_id'] = 'another-origin'
    s.odom(sample, now=1.21)
    assert s.status(now=1.21)['odometry'] == 'waiting'


def test_cloud_with_unbounded_inertial_gap_reaches_native_recovery_without_pose():
    unlocalized = []
    s = stream(on_unlocalized_frame=lambda *frame: unlocalized.append(frame))
    s.odom(odom(1000), now=1.)
    s.cloud(1_200_000_000, 'camera', [[1, 0, 0]], now=1.2)
    sample = odom(1450, .2, status='coasting')
    guess = sample['pose']
    sample['pose'] = None
    guess['covariance'] = (np.eye(6)*.01).ravel().tolist()
    sample['vendor'].update(source='rgbd_eskf', fusion=dict(
        prediction=guess, translation_integrated=True, prediction_stamp_ns=1_450_000_000,
        prediction_anchor_ns=1_000_000_000, mapping_allowed=False))
    sample['vendor']['motus_slam']['stamp_ns'] = 1_450_000_000
    s.odom(sample, now=1.45)

    assert not s.accepted
    assert s.pending_count == 0
    assert s.status(now=1.45)['admitted_frames'] == 0
    assert [(stamp, cloud, pose) for stamp, cloud, pose, *_ in unlocalized] == [
        (1_200_000_000, [[1, 0, 0]], None)]


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


def test_pose_loss_offers_newest_pending_scan_to_recovery_once():
    unlocalized = []
    s = stream(on_unlocalized_frame=lambda *frame: unlocalized.append(frame))
    s.odom(odom(1000), now=1.)
    s.cloud(1_200_000_000, 'camera', [[1, 0, 3]], now=1.2)
    s.cloud(1_100_000_000, 'camera', [[2, 0, 3]], now=1.25)  # delayed older scan
    loss = dict(odom(1300, status='recovering'), pose=None)
    s.odom(loss, now=1.3)
    s.odom(loss, now=1.4)
    assert [(stamp, cloud, pose) for stamp, cloud, pose, *_ in unlocalized] == [
        (1_200_000_000, [[1, 0, 3]], None)]
    assert s.pending_count == 0
    assert s.status(now=1.4)['admitted_frames'] == 0
