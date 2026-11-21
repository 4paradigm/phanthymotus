"""Recovery must allow real native results to rebuild three-frame evidence."""
import pytest

from plugins.rgbd_odometry_runtime import EstimateGate
from plugins.spatial_recovery import RecoveryPolicy
from test_rgbd_odometry import messages


def slow_gate():
    gate = EstimateGate('run', 'camera_optical', 'camera-clock')
    gate.recovery_policy = RecoveryPolicy()
    gate.processing_p95_ns = 400_000_000
    next_input = 1_000_000_000
    for index in range(8):
        capture = 1_000_000_000 + index * 500_000_000
        while next_input <= capture:
            gate.input(next_input)
            next_input += 33_333_333
        sample = gate.accept(*messages(capture), capture + 400_000_000)
    assert sample['pose'] is not None
    return gate


def test_fast_camera_slow_native_recovers_without_restarting_its_origin():
    gate = slow_gate()
    msg, info = messages(5_000_000_000)
    info.lost = True
    gate.input(5_000_000_000)
    assert gate.accept(msg, info, 5_400_000_000) is None
    assert gate.state == 'recovering'
    # One missing native result plus three valid 2 Hz results must fit the
    # bounded pause even though camera callbacks arrive at approximately 30 Hz.
    for capture in (5_500_000_000, 6_000_000_000):
        gate.input(capture)
        assert gate.accept(*messages(capture), capture + 400_000_000) is None
        assert gate.state == 'recovering'
    gate.input(6_500_000_000)
    resumed = gate.accept(*messages(6_500_000_000), 6_900_000_000)
    assert resumed['pose']['frame_id'] == 'run_odom'
    assert resumed['pose']['child_frame_id'] == 'camera_optical'
    assert gate.state == 'tracking'


def test_slow_native_pause_is_frozen_and_persistent_loss_still_expires():
    gate = slow_gate()
    gate.degrade('temporary visual loss', 5_400_000_000)
    pause = gate.pause_context
    assert 2_200_000_000 <= pause.budget_ns <= 3_000_000_000
    gate.input_period_ns = 1_000_000_000
    gate.processing_p95_ns = 2_000_000_000
    gate.degrade('still lost', 5_500_000_000)
    assert gate.pause_context is pause
    gate.last_input = gate.last_result_stamp = 8_000_000_000
    assert gate.status(8_000_000_000) == 'lost'


@pytest.mark.parametrize('stamp', [4_500_000_000, 4_400_000_000, 8_000_000_000])
def test_duplicate_backwards_or_future_results_cannot_inflate_native_cadence(stamp):
    gate = slow_gate()
    assert gate.result_period_ns == 500_000_000
    assert gate.accept(*messages(stamp), 4_900_000_000) is None
    assert gate.result_period_ns == 500_000_000
