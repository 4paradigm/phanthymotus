"""Native preview metadata is removed only for the existing mapping consumer."""
import json
import math
import struct

import numpy as np
import pytest

from plugins import pointcloud_slam_preview as preview


@pytest.mark.parametrize('backend', ['raw', 'geometry_experimental'])
def test_mapping_body_preserves_bounded_xyz_and_pose_without_native_metadata(backend):
    points = [[-1.25, 2.5, -3.75], [4, -5, 6], [7.5, -8.25, 9.125],
              [-10, 11, -12], [-13.5, 14.25, -15.875]]
    pose = np.array([[0, -1, 0, -2.5], [1, 0, 0, 4.25],
                     [0, 0, 1, -.75], [0, 0, 0, 1]])
    metadata = {'segment_id': 'run-1', 'frame_id': 'odom', 'stamp_ns': '1234567890',
                'backend': backend, 'candidate_count': 2}
    packet = preview.encode_preview(points, pose, max_points=3, metadata=metadata)

    payload = preview.mapping_payload(packet)

    expected = struct.pack('<fffBI9f', -2.5, 4.25, math.pi / 2, 3, 3,
                           -1.25, 2.5, -3.75, 7.5, -8.25, 9.125,
                           -13.5, 14.25, -15.875)
    assert payload == expected
    assert np.isfinite(np.frombuffer(payload[17:], dtype='<f4')).all()
    metadata_length = struct.unpack('<I', packet[:4])[0]
    native_metadata = json.loads(packet[4:4 + metadata_length])
    assert native_metadata['schema'] == 'motus.slam-preview/1'
    assert native_metadata['backend'] == backend
    assert native_metadata['accepted_loops'] == 0
    assert packet == struct.pack('<I', metadata_length) + packet[4:4 + metadata_length] + payload
    assert b'motus.slam-preview/1' not in payload
