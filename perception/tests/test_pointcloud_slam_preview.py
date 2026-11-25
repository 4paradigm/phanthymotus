"""Mapping snapshots use the existing Core binary packet directly."""
import math
import struct

import numpy as np
import pytest

from plugins import pointcloud_slam_preview as preview


def test_mapping_packet_preserves_pose_coordinates_and_bounds_points():
    pose = np.array([[0, -1, 0, -2.5], [1, 0, 0, 4.25],
                     [0, 0, 1, -.75], [0, 0, 0, 1]])
    packet = preview.encode_mapping(np.arange(30).reshape(-1, 3), pose, max_points=3, pose_reference='body')
    expected = struct.pack('<fffBI9f', -2.5, 4.25, -math.pi / 2, 23, 3,
                           0, 1, 2, 12, 13, 14, 27, 28, 29)
    assert packet == expected


@pytest.mark.parametrize('ros_yaw', [math.pi / 2, -math.pi / 2, math.pi / 4, -math.pi / 4])
def test_mapping_display_yaw_matches_other_robot_cards(ros_yaw):
    pose = np.eye(4)
    c, s = math.cos(ros_yaw), math.sin(ros_yaw)
    pose[:2, :2] = [[c, -s], [s, c]]
    packet = preview.encode_mapping([[1., 2., 3.]], pose, max_points=1, pose_reference='body')
    x, y, display_yaw, flags, count = struct.unpack('<fffBI', packet[:17])
    assert (x, y, flags, count) == (0., 0., 23, 1)
    assert display_yaw == pytest.approx(-ros_yaw)


@pytest.mark.parametrize('points,limit', [([[float('nan'), 0, 0]], 1),
    ([[1e40, 0, 0]], 1), ([[1, 2]], 1), ([[0, 0, 0]], 80001),
    ([[0, 0, 0]], True), ([[0, 0, 0]], 0)])
def test_mapping_rejects_invalid_points_or_budget(points, limit):
    with pytest.raises(ValueError):
        preview.encode_mapping(points, np.eye(4), max_points=limit)


def test_mapping_supports_empty_saved_snapshot():
    packet = preview.encode_mapping(np.empty((0, 3)), np.eye(4), max_points=1)
    assert packet == struct.pack('<fffBI', 0, 0, 0, 39, 0)


def test_mapping_rejects_marker_outside_wire_float_range():
    pose = np.eye(4)
    pose[0, 3] = 1e40
    with pytest.raises(ValueError):
        preview.encode_mapping([[0, 0, 0]], pose, max_points=1, pose_reference='body')


def test_directed_place_uses_map_yaw_and_shares_the_preview_budget():
    base = preview.encode_mapping([[3., 4., 5.]], np.eye(4), max_points=2000)
    place = dict(x=1., y=-2., z=.3, yaw=math.pi / 2)
    packet = preview.with_places(base, [place], max_points=2000)
    points = np.frombuffer(packet, '<f4', offset=17).reshape(-1, 3)
    assert len(points) == struct.unpack_from('<I', packet, 13)[0] == 1 + 1218
    assert points[0].tolist() == [3., 4., 5.]
    assert np.max(points[1:, 1]) == pytest.approx(-2. + .55)
    assert np.max(points[1:, 2]) == pytest.approx(.3 + .82)
    bounded = preview.with_places(base, [place] * 6, max_points=20)
    assert len(bounded) == 17 + 12 * 20
