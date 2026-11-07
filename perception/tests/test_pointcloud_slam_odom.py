import copy

import pytest

from plugins.pointcloud_slam_odom import decode_bumi_odom


def sample():
    return {
        "schema": "motus.odom/1", "stamp_ms": 1234, "frame": "body",
        "twist": [None] * 6,
        "pose": {"position": [1, 2, 3], "quaternion": [0, 0, 0, 1],
                 "frame_id": "segment_odom", "child_frame_id": "segment_base",
                 "covariance": [0.0] * 36},
        "vendor": {"status": "tracking", "segment_id": "segment"},
    }


def test_pose_frames_come_from_pose_not_twist():
    result = decode_bumi_odom(sample())
    assert result.frame_id == "segment_odom"
    assert result.child_frame_id == "segment_base"
    assert result.stamp_ms == 1234
    assert result.segment_id == "segment"
    assert result.position == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("change", [
    lambda s: s.update(pose=None),
    lambda s: s.update(schema="wrong"),
    lambda s: s.update(stamp_ms=0),
    lambda s: s["vendor"].update(status="lost"),
    lambda s: s["vendor"].update(segment_id=""),
    lambda s: s["pose"].update(frame_id=""),
    lambda s: s["pose"].update(child_frame_id=""),
    lambda s: s["pose"].update(position=[float("nan"), 0, 0]),
    lambda s: s["pose"].update(quaternion=[0, 0, 0, 0]),
    lambda s: s["pose"].update(quaternion=[0, 0, 0, 2]),
    lambda s: s["pose"].update(covariance=[0] * 35),
    lambda s: s["pose"].update(covariance=[float("inf")] * 36),
])
def test_rejects_unusable_samples(change):
    data = copy.deepcopy(sample())
    change(data)
    with pytest.raises(ValueError):
        decode_bumi_odom(data)


def standard_sample():
    data = sample()
    data['units'] = {'length': 'm', 'linear': 'm/s', 'angular': 'rad/s'}
    data['vendor'] = {'motus_slam': {
        'status': 'tracking', 'segment_id': 'run-1',
        'stamp_source': 'measurement', 'clock_id': 'robot-clock',
    }}
    data['pose']['covariance'] = None
    return data


def test_standard_pose_accepts_unknown_covariance_and_preserves_clock():
    from plugins.pointcloud_slam_odom import decode_slam_odom
    result = decode_slam_odom(standard_sample())
    assert result.segment_id == 'run-1'
    assert result.clock_id == 'robot-clock'
    assert result.covariance is None


@pytest.mark.parametrize('change', [
    lambda s: s.update(pose=None),
    lambda s: s['units'].update(length='cm'),
    lambda s: s.update(units=None),
    lambda s: s['vendor']['motus_slam'].update(status='lost'),
    lambda s: s['vendor']['motus_slam'].update(stamp_source='received'),
    lambda s: s['vendor']['motus_slam'].update(clock_id=''),
    lambda s: s['vendor']['motus_slam'].update(segment_id=''),
])
def test_standard_pose_rejects_unusable_or_unproven_inputs(change):
    from plugins.pointcloud_slam_odom import decode_slam_odom
    data = standard_sample()
    change(data)
    with pytest.raises(ValueError):
        decode_slam_odom(data)
