import math
import struct
from dataclasses import replace

import numpy as np
import pytest

from plugins.pointcloud_slam_preview import encode_mapping
from plugins.pointcloud_slam_odom import decode_slam_odom
from plugins.pointcloud_slam_alignment import align_pose
from test_pointcloud_slam_stream import odom


@pytest.mark.parametrize('angle', [0., .7, -1.2, math.pi])
def test_optical_forward_and_body_forward_produce_same_map_heading(angle):
    c, s = np.cos(angle), np.sin(angle)
    body = np.eye(4); body[:2, :2] = [[c, -s], [s, c]]
    # Axis convention only: optical +Z forward, +X right, +Y down.
    optical = body.copy()
    optical[:3, :3] = body[:3, :3] @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    for reference, pose, expected_flags in [('body', body, 23), ('camera_optical', optical, 15)]:
        packet = encode_mapping([[1, 2, 3]], pose, max_points=1, pose_reference=reference)
        _, _, yaw, flags, _ = struct.unpack('<fffBI', packet[:17])
        assert yaw == pytest.approx(-angle)
        assert flags == expected_flags
        np.testing.assert_array_equal(np.frombuffer(packet[17:], '<f4'), [1, 2, 3])


@pytest.mark.parametrize('reference', [None, 'unknown', 'camera_optical'])
def test_unknown_or_vertical_heading_is_explicitly_unavailable(reference):
    packet = encode_mapping([[1, 2, 3]], np.eye(4), max_points=1, pose_reference=reference)
    _, _, yaw, flags, _ = struct.unpack('<fffBI', packet[:17])
    assert flags & 32 and flags & 4
    assert math.isfinite(yaw)


def test_pose_reference_survives_decode_and_interpolation_without_frame_name_guessing():
    sample = odom(1000)
    sample['vendor']['pose_reference'] = 'camera_optical'
    first = decode_slam_odom(sample)
    second = replace(first, stamp_ms=1200)
    assert align_pose([first, second], 1_100_000_000, max_gap_ms=250).pose_reference == 'camera_optical'
    with pytest.raises(ValueError):
        align_pose([first, replace(second, pose_reference='body')], 1_100_000_000, max_gap_ms=250)
    del sample['vendor']['pose_reference']
    sample['pose']['child_frame_id'] = 'base_link'
    assert decode_slam_odom(sample).pose_reference is None


def test_conflicting_pose_reference_is_rejected():
    sample = odom(1000)
    sample['vendor']['pose_reference'] = 'camera_optical'
    sample['vendor']['motus_slam']['pose_reference'] = 'body'
    with pytest.raises(ValueError, match='conflicting pose reference'):
        decode_slam_odom(sample)
