from types import SimpleNamespace as NS

import pytest

import vision_stubs
from plugins.rgbd_fusion import ShortTermFusion
from plugins.rgbd_odometry_runtime import CaptureObservations
from test_rgbd_fusion import K, images, source, warmed
from test_rgbd_odometry import messages


CAPTURE_PERIOD_NS = 100_000_000


def landmark_messages(stamp, frame, variance, defect=None):
    pose, info = messages(stamp)
    pose.child_frame_id = frame
    pose.pose.covariance[0] = variance
    info.type = 0
    ids = list(range(45))
    if defect == 'insufficient_support':
        ids = list(range(15)) + list(range(45, 75))
    shift = 1. if defect == 'mismatched_landmarks' else 0.
    info.inliers = len(ids)
    info.word_inliers = info.local_map_keys = ids
    info.local_map_values = [NS(x=float(key % 5)+shift, y=float(key // 5), z=1.) for key in ids]
    if defect == 'non_psd_covariance':
        pose.pose.covariance[7] = -.02
    elif defect == 'excess_rotation_uncertainty':
        pose.pose.covariance[35] = .15
    elif defect == 'correlated_rotation_uncertainty':
        # Rotational diagonals pass the ESKF bound. The .13 eigenvalue
        # of the first two angular axes exceeds the native bound.
        covariance = pose.pose.covariance
        covariance[21] = covariance[28] = .10
        covariance[22] = covariance[27] = .03
    return pose, info


def submit(captures, stamp, rgbd, pose, info):
    captures.add_frame(stamp, *rgbd, stamp+10_000_000)
    captures.result('pose', pose, stamp+19_000_000)
    captures.result('info', info, stamp+20_000_000)


@pytest.mark.parametrize('imu_covered', [False, True], ids=['no_imu', 'imu_covered'])
@pytest.mark.parametrize('defect,expected', [
    (None, [2, 3, 4, 5, 6]),
    ('mismatched_landmarks', [2, 6]),
    ('insufficient_support', [2, 6]),
    ('non_psd_covariance', [2, 6]),
    ('excess_rotation_uncertainty', [2, 6]),
])
def test_external_native_covariance_requires_landmarks_but_ignores_eskf_limit(
        imu_covered, defect, expected):
    fusion, anchor = (warmed() if imu_covered else
                      (ShortTermFusion('run', 'camera_optical', 'clock', K), 1_000_000_000))
    captures = CaptureObservations(fusion)
    forwarded = []
    captures.visual_sink = lambda *values: forwarded.append(values)
    rgbd = images()
    stamps = [anchor+(index+1)*CAPTURE_PERIOD_NS for index in range(7)]
    for index, stamp in enumerate(stamps):
        if imu_covered:
            for imu_stamp in range(stamp-95_000_000, stamp+1, 5_000_000):
                fusion.observe_imu(source(imu_stamp), imu_stamp+1_000_000)
        pose, info = landmark_messages(stamp, fusion.frame, .05 if index == 3 else .001,
                                       defect if index == 3 else None)
        if index == 3:
            pose.pose.pose.position.x = .01
        submit(captures, stamp, rgbd, pose, info)
        if index == 3:
            diagnostic = fusion.frames[-1]
            assert diagnostic['capture_ns'] == stamp
            assert diagnostic['visual'] == 'missing'
            assert diagnostic['visual_reject_reason'] == 'invalid visual pose or covariance'
            if imu_covered:
                assert fusion.filter.stamp == stamp
                assert fusion.last_visual == stamps[2]
            else:
                assert fusion.filter is None

    expected = [0, 1, *expected] if imu_covered else expected
    assert [stamp for stamp, _, _ in forwarded] == [stamps[index] for index in expected]
    assert all(segment == fusion.segment for _, segment, _ in forwarded)
    if defect is None:
        measurement = next(pose for stamp, _, pose in forwarded if stamp == stamps[3])
        assert measurement['frame_id'] == 'run_odom'
        assert measurement['child_frame_id'] == fusion.frame
        assert measurement['position'] == [.01, 0., 0.]
        assert measurement['quaternion'] == [0., 0., 0., 1.]
        assert measurement['covariance'] == [.05] + [.001 if i % 7 == 0 else 0. for i in range(1, 36)]


@pytest.mark.parametrize('defect,eskf_visual,after_discontinuity,delivered', [
    pytest.param('non_psd_covariance', 'missing', ('lost', 'visual'), [2], id='both_gates_reject'),
    pytest.param('correlated_rotation_uncertainty', 'offered', ('recovering', None), [2, 7],
                 id='native_only_rejects'),
])
def test_resume_after_native_covariance_rejection_preserves_segment_lifecycle(
        defect, eskf_visual, after_discontinuity, delivered):
    fusion = ShortTermFusion('run', 'camera_optical', 'clock', K)
    captures = CaptureObservations(fusion)
    forwarded = []
    captures.visual_sink = lambda stamp, *_: forwarded.append(stamp)
    rgbd = images()
    stamps = [1_000_000_000+index*CAPTURE_PERIOD_NS for index in range(8)]
    for stamp in stamps[:3]:
        submit(captures, stamp, rgbd, *landmark_messages(stamp, fusion.frame, .001))
    assert forwarded == [stamps[2]]

    rejected = stamps[3]
    submit(captures, rejected, rgbd, *landmark_messages(rejected, fusion.frame, .001, defect))
    assert captures.external_gate.state == 'recovering'
    assert fusion.frames[-1]['visual'] == eskf_visual

    discontinuous = stamps[4]
    pose, info = landmark_messages(discontinuous, fusion.frame, .001)
    pose.pose.pose.position.x = 10.
    submit(captures, discontinuous, rgbd, pose, info)
    assert (captures.external_gate.state, captures.external_gate.loss_kind) == after_discontinuity

    for stamp in stamps[5:]:
        submit(captures, stamp, rgbd, *landmark_messages(stamp, fusion.frame, .001))
    assert forwarded == [stamps[index] for index in delivered]


def test_wrong_frame_duplicate_supersedes_native_admission_while_waiting_for_imu():
    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    forwarded = []
    captures.visual_sink = lambda stamp, *_: forwarded.append(stamp)
    rgbd = images()
    first, inspected, later = [anchor+(index+1)*CAPTURE_PERIOD_NS for index in range(3)]
    for imu_stamp in range(anchor+5_000_000, first+1, 5_000_000):
        fusion.observe_imu(source(imu_stamp), imu_stamp+1_000_000)
    submit(captures, first, rgbd, *landmark_messages(first, fusion.frame, .001))
    submit(captures, inspected, rgbd, *landmark_messages(inspected, fusion.frame, .05))
    assert forwarded == [first, inspected]
    assert fusion.last_visual == first

    wrong, _ = messages(inspected)
    wrong.child_frame_id = 'wrong_camera'
    captures.result('pose', wrong, inspected+30_000_000)
    assert captures.external_gate.loss_kind == 'frame'
    assert fusion.state == 'tracking'
    submit(captures, later, rgbd, *landmark_messages(later, fusion.frame, .001))
    assert forwarded == [first, inspected]

    for imu_stamp in range(first+5_000_000, inspected+1, 5_000_000):
        fusion.observe_imu(source(imu_stamp), later+30_000_000)
    rejected, = captures.drain(later+30_000_000)
    assert rejected['pose'] is None
    assert fusion.loss_kind == 'frame'
    assert fusion.last_visual == first
