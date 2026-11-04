"""A missing or stale pose must be counted, not paired by arrival time."""
from tools.pointcloud_slam_timing_audit import audit


def sample(ms, status='tracking', segment='a'):
    return {'schema': 'motus.odom/1', 'stamp_ms': ms,
            'pose': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1],
                     'frame_id': 'odom', 'child_frame_id': 'camera', 'covariance': [0.] * 36},
            'vendor': {'status': status, 'segment_id': segment}}


def cloud(ns):
    return {'sec': ns // 10**9, 'nanosec': ns % 10**9}


def test_audit_counts_interpolation_and_rejects_long_gap_or_unbracketed():
    result = audit([sample(1000), sample(1100), sample(1500)],
                   [cloud(1_050_000_000), cloud(1_200_000_000), cloud(2_000_000_000)], 200)
    assert result == {'total_clouds': 3, 'aligned': 1, 'rejected': 2}


def test_lost_state_blocks_interpolation_even_when_times_are_nearby():
    result = audit([sample(1000), sample(1050, status='lost'), sample(1100)],
                   [cloud(1_075_000_000)], 200)
    assert result['aligned'] == 0
    assert result['rejected'] == 1


def test_bad_gap_is_rejected_even_with_no_clouds():
    import pytest
    with pytest.raises(ValueError):
        audit([], [], float('nan'))
