"""ESKF decision order: hard errors, then visual correction, then uncertainty.

Predicted covariance alone must not discard a legal visual correction, and no
missing/rejected observation can extend pure-inertial operation past its bounds.
"""
import numpy as np
import pytest

from plugins.pointcloud_slam_odom import decode_slam_odom
from test_rgbd_fusion import images, pose, source, warmed


def imu_until(c, start, seconds):
    for i in range(1, round(seconds/.005)+1):
        stamp = start+i*5_000_000
        c.observe_imu(source(stamp), stamp+1_000_000)
    return stamp


def noisy(x=0., variance=.04):
    value = pose(x)
    value['covariance'] = (np.eye(6)*variance).ravel().tolist()
    return value


@pytest.mark.parametrize('sigma,status', [(.15, 'coasting'), (.20, 'coasting'), (.201, 'lost')])
def test_default_position_sigma_limit_allows_up_to_twenty_cm_without_tracking_prediction(sigma, status):
    c, t = warmed()
    stamp = imu_until(c, t, .1)
    c._propagate(stamp)
    # Isolate the decision boundary from the engineering propagation priors.
    c.filter.P = np.zeros((15, 15))
    c.filter.P[:3, :3] = np.eye(3)*sigma**2
    sample = c.observe_frame(stamp, *images(), None, stamp+1_000_000, reason='visual missing')
    fusion = sample['vendor']['fusion']
    assert fusion['sigma_limit_m'] == .20
    assert fusion['last_frame']['predicted_sigma_m'] == pytest.approx(sigma)
    assert c.state == status
    assert sample['pose'] is None
    if status == 'lost':
        assert 'exceeds 0.20 m' in c.reason
    else:
        assert fusion['prediction'] is not None


def test_corrected_sigma_between_old_and_new_limits_can_recover_tracking():
    c, t = warmed()
    stamp = imu_until(c, t, .9)
    sample = c.observe_frame(stamp, *images(), noisy(), stamp+1_000_000)
    diagnostic = sample['vendor']['fusion']['last_frame']
    assert .10 < diagnostic['corrected_sigma_m'] <= .20
    assert diagnostic['visual'] == 'accepted'
    assert c.reason == 'waiting for three consistent visual observations'
    for _ in range(2):
        stamp = imu_until(c, stamp, .1)
        sample = c.observe_frame(stamp, *images(), noisy(), stamp+1_000_000)
    assert c.state == 'tracking' and sample['pose'] is not None


def test_predicted_sigma_over_limit_is_corrected_by_a_legal_visual_first():
    c, t = warmed()
    c.filter.P[:3, :3] = np.eye(3)*.25**2
    stamp = imu_until(c, t, .8)
    sample = c.observe_frame(stamp, *images(), pose(), stamp+1_000_000)
    fusion = sample['vendor']['fusion']
    assert c.state != 'lost', c.reason
    assert fusion['last_frame']['predicted_sigma_m'] > .20
    assert fusion['last_frame']['corrected_sigma_m'] <= .20
    assert fusion['last_frame']['visual'] == 'accepted'
    # A gap demotes tracking: one correction cannot re-establish tracking.
    assert sample['pose'] is None and c.state == 'recovering'
    for _ in range(3):
        stamp = imu_until(c, stamp, .1)
        sample = c.observe_frame(stamp, *images(), pose(), stamp+1_000_000)
    assert decode_slam_odom(sample).segment_id == 'run'


@pytest.mark.parametrize('visual,why', [(None, 'visual missing'), (pose(3.), 'absolute_residual_gate')])
def test_missing_or_bad_visual_cannot_extend_inertial_bound(visual, why):
    c, t = warmed()
    c.filter.P[:3, :3] = np.eye(3)*.25**2
    stamp = imu_until(c, t, .8)
    sample = c.observe_frame(stamp, *images(), visual, stamp+1_000_000, reason='visual missing')
    assert sample['pose'] is None and c.state == 'lost'
    assert 'predicted position uncertainty exceeds 0.20 m' in c.reason
    assert why in str(sample['vendor']['fusion']['last_frame'])


def test_corrected_uncertainty_over_limit_pauses_output_then_is_bounded():
    c, t = warmed()
    c.filter.P[:3, :3] = np.eye(3)*.5**2
    color, depth = images()
    color = np.where(color > 0, 16, 0).astype(np.uint8)
    weak = noisy()
    weak['quality_weight'] = 3.
    stamp = imu_until(c, t, .9)
    sample = c.observe_frame(stamp, color, depth, weak, stamp+1_000_000)
    assert sample['vendor']['fusion']['last_frame']['visual'] == 'accepted'
    assert sample['vendor']['fusion']['last_frame']['corrected_sigma_m'] > .20
    assert sample['pose'] is None and c.state == 'recovering'
    assert 'corrected position uncertainty' in c.reason
    # Accepted but over-uncertain corrections cannot keep the segment alive.
    for _ in range(12):
        stamp = imu_until(c, stamp, .1)
        sample = c.observe_frame(stamp, color, depth, weak, stamp+1_000_000)
        assert sample['pose'] is None
        if c.state == 'lost':
            break
    assert c.state == 'lost' and 'for more than 1 s' in c.reason


def test_info_poll_during_pipeline_latency_does_not_demote_or_reset_tracking():
    """Wall-clock polls lag capture time by native latency (130-240 ms seen on
    Tianyi). That must not demote tracking or reset recovery evidence."""
    c, t = warmed()
    stamp = imu_until(c, t, .133)
    c.info(t+300_000_000)
    assert c.state == 'tracking'
    sample = c.observe_frame(stamp, *images(), pose(), t+300_000_000)
    assert sample['pose'] is not None


def test_frame_silence_stops_claiming_tracking_without_resetting_evidence():
    c, t = warmed()
    imu_until(c, t, .6)
    assert c.status(t+400_000_000) == 'tracking'
    assert c.status(t+600_000_000) == 'coasting'
    assert c.good >= 3
    # Frames were only delayed: a contiguous capture resumes tracking...
    stamp = t+200_000_000
    assert c.observe_frame(stamp, *images(), pose(), t+610_000_000)['pose'] is not None
    # ...but a capture gap over 250 ms needs three corrections again.
    stamp += 300_000_000
    assert c.observe_frame(stamp, *images(), pose(), t+620_000_000)['pose'] is None


def test_per_frame_diagnostics_are_bounded_and_not_copied_into_each_sample():
    c, t = warmed()
    stamp = t
    for _ in range(40):
        stamp = imu_until(c, stamp, .1)
        sample = c.observe_frame(stamp, *images(), pose(), stamp+40_000_000, arrival_ns=stamp+15_000_000)
    history = c.info(stamp+40_000_000, history=True)['frames']
    assert len(history) == 16
    last = history[-1]
    assert last['capture_ns'] == stamp and last['arrival_ns'] == stamp+15_000_000
    assert last['latency_ms'] == pytest.approx(40.)
    assert 'frames' not in sample['vendor']['fusion']
    assert sample['vendor']['fusion']['last_frame'] == last
