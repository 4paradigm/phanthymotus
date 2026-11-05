"""Diagnostic preview packets are bounded and carry no SLAM-success claim."""
import struct
import json
import numpy as np
import pytest

from plugins.pointcloud_slam_preview import encode_preview


META = {'segment_id': 'run-1', 'frame_id': 'odom', 'stamp_ns': '1234567890', 'backend': 'raw'}


def unpack(data):
    length = struct.unpack('<I', data[:4])[0]
    return json.loads(data[4:4+length]), data[4+length:]


def test_preview_packet_preserves_position_and_bounds_points():
    points = np.arange(30, dtype=float).reshape(-1, 3)
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    data = encode_preview(points, pose, max_points=4, metadata=META)
    meta, payload = unpack(data)
    assert meta['schema'] == 'motus.slam-preview/1' and meta['segment_id'] == 'run-1'
    assert meta['stamp_ns'] == '1234567890' and meta['accepted_loops'] == 0
    x, y, yaw, flags, count = struct.unpack('<fffBI', payload[:17])
    assert (x, y, yaw, flags, count) == (1., 2., 0., 3, 4)
    assert len(payload) == 17 + 12 * count
    decoded = np.frombuffer(payload[17:], dtype='<f4').reshape(-1, 3)
    assert decoded[0].tolist() == [0., 1., 2.]
    assert decoded[-1].tolist() == [27., 28., 29.]


def test_preview_rejects_invalid_coordinates_and_unbounded_count():
    with pytest.raises(ValueError):
        encode_preview([[float('nan'), 0, 0]], np.eye(4), max_points=10, metadata=META)
    with pytest.raises(ValueError):
        encode_preview([[0, 0, 0]], np.eye(4), max_points=80001, metadata=META)
    with pytest.raises(ValueError):
        encode_preview([[0, 0, 0]], np.eye(4), max_points=10, metadata={})


def test_geometry_preview_is_explicitly_unvalidated_and_cannot_claim_loops():
    meta = {**META, 'backend': 'geometry_experimental', 'candidate_count': 2}
    header, _ = unpack(encode_preview([[0, 0, 0]], np.eye(4), max_points=10, metadata=meta))
    assert header['kind'] == 'unvalidated_geometry_preview'
    assert header['candidate_count'] == 2 and header['accepted_loops'] == 0
    with pytest.raises(ValueError):
        encode_preview([[0, 0, 0]], np.eye(4), max_points=10,
                       metadata={**meta, 'candidate_count': -1})
