import copy

import pytest

from plugins.pointcloud_slam_odom import decode_legacy_odom


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
    result = decode_legacy_odom(sample())
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
        decode_legacy_odom(data)


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


def test_standard_pose_requires_actual_numeric_pose():
    from plugins.pointcloud_slam_odom import decode_slam_odom
    data = standard_sample(); data['pose'] = None
    with pytest.raises(ValueError): decode_slam_odom(data)


@pytest.mark.parametrize('twist_stamp', [2000, None])
def test_slam_decoder_uses_independent_pose_measurement_time(twist_stamp):
    from plugins.pointcloud_slam_odom import decode_slam_odom
    value = standard_sample()
    value['stamp_ms'] = twist_stamp
    value['pose']['stamp_ms'] = 1234
    value['vendor']['motus_slam']['stamp_ns'] = 1_234_000_123
    result = decode_slam_odom(value)
    assert result.stamp_ms == 1234
    assert result.stamp_ns == 1_234_000_123
    assert value['stamp_ms'] == twist_stamp


def test_slam_decoder_keeps_legacy_top_level_pose_time():
    from plugins.pointcloud_slam_odom import decode_slam_odom
    value = standard_sample()
    result = decode_slam_odom(value)
    assert result.stamp_ms == 1234
    assert result.stamp_ns == 1_234_000_000


@pytest.mark.parametrize('offset,expected', [(0, True), (50_000_000, True), (50_000_001, False), (None, False)])
def test_height_matching_requires_capture_timed_body_gravity(offset, expected):
    from plugins.pointcloud_slam_odom import decode_slam_odom
    value = standard_sample()
    tick = 3_000_000_000
    value['vendor']['motus_slam'].update(pose_source='external_fusion', pose_reference='body', stamp_ns=tick)
    value['vendor']['external_fusion'] = {} if offset is None else {'body_gravity_stamp_ns': tick-offset}
    assert decode_slam_odom(value).gravity_aligned is expected
