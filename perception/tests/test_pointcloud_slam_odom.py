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
