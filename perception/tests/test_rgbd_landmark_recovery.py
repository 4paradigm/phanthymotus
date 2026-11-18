"""Only measured correspondences to the old native F2M map can re-open its gauge."""
import math
from types import SimpleNamespace as NS

import pytest

from plugins.rgbd_odometry_runtime import EstimateGate
from test_rgbd_odometry import messages
from test_rgbd_odometry_status import sessions  # noqa: F401


def visual(gate, stamp, *, ids=range(24), x=0., angle=0., world_shift=0., lost=False, variance=.001, kind=0):
    msg, info = messages(stamp)
    msg.pose.covariance[0] = variance
    msg.pose.pose.position.x = x
    msg.pose.pose.orientation.z = math.sin(math.radians(angle)/2)
    msg.pose.pose.orientation.w = math.cos(math.radians(angle)/2)
    info.type, info.lost, info.inliers = kind, lost, 0 if lost else len(ids)
    info.local_map_keys = info.word_inliers = list(ids)
    info.local_map_values = [NS(x=float(i % 5) + world_shift, y=float(i // 5), z=1.) for i in ids]
    gate.input(stamp)
    return gate.accept(msg, info, stamp)


def anchored():
    g = EstimateGate('run', 'camera_optical', 'camera-clock')
    for i in range(3):
        sample = visual(g, 1_000_000_000 + i*100_000_000)
    assert sample is not None
    return g


def test_same_f2m_landmarks_can_relocalize_after_silence_without_any_gap_pose():
    g = anchored()
    assert g.status(3_000_000_000) == 'recovering'
    assert g.status(26_000_000_000) == 'recovering'
    assert visual(g, 27_000_000_000, x=.003) is None
    assert visual(g, 27_100_000_000, x=.004) is None
    s = visual(g, 27_200_000_000, x=.003)
    assert s['pose']['position'][0] == .003
    assert s['vendor']['motus_slam']['segment_id'] == 'run'
    assert s['vendor']['relocalization']['shared_landmarks'] >= 20


def test_expired_visual_pause_requires_old_landmarks_before_resuming():
    from plugins.spatial_recovery import RecoveryPolicy

    g = anchored()
    g.recovery_policy = RecoveryPolicy()
    g.input(1_300_000_000)
    g.degrade('visual rejection', 1_300_000_000)
    assert g.status(2_301_000_000) == 'recovering'
    assert len(g.relocalization_points) == 24
    assert visual(g, 2_400_000_000, ids=range(24, 48)) is None
    assert visual(g, 2_500_000_000, x=.003) is None
    assert visual(g, 2_600_000_000, x=.003) is None
    result = visual(g, 2_700_000_000, x=.003)
    assert result['vendor']['relocalization']['anchor_stamp_ns'] == 1_200_000_000


@pytest.mark.parametrize('changed', [
    dict(ids=range(5, 29)),  # 24 total inliers but only 19 old correspondences
    dict(ids=range(24, 48)), dict(ids=[1]*24), dict(world_shift=1.),
    dict(x=.051), dict(angle=5.1), dict(lost=True)])
def test_novel_recycled_ids_jump_and_lost_cannot_reopen_old_gauge(changed):
    g = anchored()
    g.status(3_000_000_000)
    for i in range(15):
        s = visual(g, 5_000_000_000 + i*100_000_000, **changed)
        assert s is None
    assert g.state == 'lost'


def test_output_timeout_with_no_returning_results_expires_while_inputs_are_fresh():
    g = anchored()
    g.status(3_000_000_000)
    for i in range(15):g.input(5_000_000_000 + i*100_000_000)
    assert g.state == 'lost' and g.loss_kind == 'output'


def test_sparse_candidates_cannot_extend_the_first_relocalization_deadline():
    gate = anchored()
    gate.status(3_000_000_000)
    for stamp in (5_000_000_000, 5_900_000_000, 6_010_000_000):
        assert visual(gate, stamp, x=.003) is None
    assert gate.state == 'lost'


def test_slow_native_cadence_can_verify_three_old_landmark_registrations():
    from plugins.spatial_recovery import RecoveryPolicy

    gate = anchored()
    gate.recovery_policy = RecoveryPolicy()
    gate.result_period_ns = 550_000_000
    gate.processing_p95_ns = 80_000_000
    assert gate.status(3_000_000_000) == 'recovering'
    assert visual(gate, 5_000_000_000, x=.003) is None
    assert visual(gate, 5_550_000_000, x=.003) is None
    result = visual(gate, 6_100_000_000, x=.003)
    assert result['vendor']['relocalization']['shared_landmarks'] == 24


def test_result_that_triggers_timeout_can_verify_the_old_landmarks():
    gate = anchored()
    stamp = 2_650_000_000
    gate.input(stamp)
    assert gate.state == 'tracking'
    msg, info = messages(stamp)
    anchor = gate.anchor_info
    for key in ('type', 'inliers', 'local_map_keys', 'local_map_values', 'word_inliers'):
        setattr(info, key, getattr(anchor, key))
    assert gate.accept(msg, info, stamp + 100_000_000) is None
    assert visual(gate, 2_950_000_000) is None
    assert visual(gate, 3_450_000_000)['vendor']['relocalization']['shared_landmarks'] == 24


def test_process_exit_during_relocalization_fails_closed():
    gate = anchored()
    gate.status(3_000_000_000)
    gate.input(5_000_000_000)
    gate.fault('native process exited', 'process')
    for i in range(3):
        assert visual(gate, 5_100_000_000 + i*100_000_000) is None
    assert gate.state == 'lost' and gate.loss_kind == 'process'


def test_failed_registration_to_old_map_restarts_without_switching_new_view_frontend():
    gate = anchored()
    gate.status(3_000_000_000)
    for i in range(3):
        assert visual(gate, 5_000_000_000+i*100_000_000, lost=True) is None
    assert gate.state == 'lost' and gate.loss_kind == 'relocalization'


def test_burst_keeps_two_measured_pairs_only_during_relocalization(sessions):
    from test_rgbd_degradation import received_images

    create, now, _ = sessions
    session = create()
    session.gate = anchored()
    session.gate.frame = 'camera_optical'
    now[0] = 4_000_000_000
    session.gate.status(now[0])
    for i in range(5):
        now[0] += 20_000_000
        received_images(session, now[0])
    assert session.pending.qsize() == 2
    buffered = [session.pending.get_nowait()[1][0] for _ in range(2)]
    assert buffered == [now[0]-20_000_000, now[0]]


@pytest.mark.parametrize('frontend', ['f2f', 'unpatched'])
def test_frontend_without_matched_f2m_points_keeps_existing_timeout(frontend):
    gate = anchored()
    if frontend == 'f2f':
        gate.anchor_info.type = 1
    else:
        gate.anchor_info.word_inliers = []
    assert gate.status(3_000_000_000) == 'lost'


def test_relocalization_expires_when_host_wall_clock_moves_back(monkeypatch):
    import time

    ticks = [100_000_000_000]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: ticks[0])
    gate = anchored()
    gate.status(3_000_000_000)
    gate.input(5_000_000_000)
    ticks[0] += 1_000_000_001
    assert gate.status(4_000_000_000) == 'lost'


def test_moderately_uncertain_visual_pose_requires_measured_repeat_landmarks_and_keeps_covariance():
    gate = EstimateGate('run', 'camera_optical', 'camera-clock')
    for i in range(4):
        sample = visual(gate, 1_000_000_000+i*250_000_000, x=.08*i, variance=.20)
        if i < 3:
            assert sample is None
    assert sample['pose']['position'][0] == .24
    assert sample['pose']['covariance'][0] == .20
    assert sample['vendor']['visual_uncertainty']['shared_landmarks'] == 24


@pytest.mark.parametrize('change', [dict(ids=range(5, 29)), dict(world_shift=.01),
                                    dict(x=1.), dict(kind=1), dict(variance=.26), dict(lost=True)])
def test_uncertain_covariance_rejects_weak_or_discontinuous_geometry(change):
    gate = anchored()
    assert visual(gate, 1_300_000_000, **dict(dict(variance=.20), **change)) is None


def test_uncertain_covariance_cannot_relocalize_a_frozen_world():
    gate = anchored()
    gate.status(3_000_000_000)
    for i in range(5):
        assert visual(gate, 5_000_000_000+i*100_000_000, variance=.20) is None

def test_short_depth_gap_before_first_pose_keeps_native_origin_alive():
    gate = EstimateGate('run', 'camera_optical', 'camera-clock')
    assert visual(gate, 1_000_000_000) is None
    assert gate.status(3_500_000_000) == 'initializing'
    assert visual(gate, 3_700_000_000) is None
    assert visual(gate, 4_000_000_000) is None
    assert visual(gate, 4_300_000_000)['pose'] is not None


def test_same_native_worker_can_return_before_its_receipt_timeout():
    gate = EstimateGate('run', 'camera_optical', 'camera-clock')
    for i in range(3):
        stamp = 1_000_000_000 + i*300_000_000
        gate.input(stamp, now_ns=stamp+100_000_000)
        gate.accept(*messages(stamp), stamp+400_000_000)
    gate.input(2_950_000_000, now_ns=3_050_000_000)
    sample = gate.accept(*messages(2_950_000_000), 3_300_000_000)
    assert sample is not None and sample['vendor']['motus_slam']['stamp_ns'] == 2_950_000_000


def test_recent_moving_f2m_pose_relocalizes_only_against_the_frozen_world():
    gate = anchored()
    gate.status(3_000_000_000)
    for i in range(3):
        sample = visual(gate, 3_200_000_000+i*120_000_000, x=.36+i*.04, variance=.2)
        if i < 2:
            assert sample is None
    assert sample['pose']['position'][0] == pytest.approx(.44)
    assert sample['pose']['covariance'][0] == .2
    assert sample['vendor']['relocalization']['shared_landmarks'] == 24
    assert sample['vendor']['visual_uncertainty']['shared_landmarks'] == 24


@pytest.mark.parametrize('change', [dict(ids=range(5, 29)), dict(world_shift=1.), dict(kind=1),
                                    dict(x=2.), dict(variance=.26), dict(lost=True)])
def test_recent_f2m_recovery_still_requires_bounded_measured_motion_and_old_geometry(change):
    gate = anchored()
    gate.status(3_000_000_000)
    for i in range(3):
        assert visual(gate, 3_200_000_000+i*120_000_000, **dict(dict(x=.36, variance=.20), **change)) is None

def test_moving_relocalization_requires_consecutive_measurements_to_agree():
    gate = anchored()
    gate.status(3_000_000_000)
    for i, x in enumerate((.4, -.4, .4)):
        assert visual(gate, 3_200_000_000+i*120_000_000, x=x, variance=.2) is None
