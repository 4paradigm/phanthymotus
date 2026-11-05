"""Bounded binary encoding for explicitly unvalidated odometry-map previews."""
import math
import json
import struct

import numpy as np

from .pointcloud_slam_stream import _rigid


def encode_preview(points, pose, *, max_points, metadata):
    """Encode bounded metadata followed by one diagnostic mapping packet."""
    if type(max_points) is not int or not 0 < max_points <= 80000:
        raise ValueError('preview point limit must be between 1 and 80000')
    if (not isinstance(metadata, dict) or
            any(not isinstance(metadata.get(k), str) or not metadata[k].strip() or len(metadata[k]) > 256
                for k in ('segment_id', 'frame_id', 'stamp_ns', 'backend')) or
            not metadata['stamp_ns'].isascii() or not metadata['stamp_ns'].isdigit() or
            int(metadata['stamp_ns']) <= 0 or metadata['backend'] not in ('raw', 'gtsam', 'geometry_experimental')):
        raise ValueError('preview segment, frame, capture time and backend required')
    geometry = metadata['backend'] == 'geometry_experimental'
    if geometry and (type(metadata.get('candidate_count')) is not int or metadata['candidate_count'] < 0):
        raise ValueError('geometry preview requires a nonnegative candidate count')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('preview needs finite Nx3 points')
    matrix = _rigid(pose)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    xyz = np.asarray(points, dtype='<f4')
    if not np.isfinite(xyz).all():
        raise ValueError('preview points exceed float32 range')
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    header = {'schema': 'motus.slam-preview/1',
              'kind': 'unvalidated_geometry_preview' if geometry else 'unvalidated_odometry_preview',
              'segment_id': metadata['segment_id'], 'frame_id': metadata['frame_id'],
              'stamp_ns': metadata['stamp_ns'], 'backend': metadata['backend'], 'accepted_loops': 0}
    if geometry:
        header['candidate_count'] = metadata['candidate_count']
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(header_bytes) > 2048:
        raise ValueError('preview metadata too large')
    body = struct.pack('<fffBI', matrix[0, 3], matrix[1, 3], yaw, 3, len(xyz)) + xyz.tobytes()
    return struct.pack('<I', len(header_bytes)) + header_bytes + body
