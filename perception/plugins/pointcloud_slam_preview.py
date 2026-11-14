"""Bounded snapshots in Core's existing sensor/mapping wire format."""
import math
import struct

import numpy as np

from .pointcloud_slam_transforms import rigid

# One stable canvas output for both live and archived maps.
PREVIEW_TOPIC = '/pointcloud_slam/map'


def encode_mapping(points, pose, *, max_points):
    if type(max_points) is not int or not 0 < max_points <= 80000:
        raise ValueError('preview point limit must be between 1 and 80000')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('preview needs finite Nx3 points')
    matrix = rigid(pose)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    if (np.any(np.abs(points) > np.finfo(np.float32).max) or
            np.any(np.abs(matrix[:2, 3]) > np.finfo(np.float32).max)):
        raise ValueError('preview points exceed float32 range')
    xyz = np.asarray(points, dtype='<f4')
    # The stock Core mapping renderer negates the packet's yaw. Match the
    # existing shared mapping renderer convention without changing RTAB-Map poses.
    ros_yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    display_yaw = -ros_yaw if ros_yaw != 0.0 else 0.0  # avoid negative-zero bytes
    return struct.pack('<fffBI', matrix[0, 3], matrix[1, 3], display_yaw, 3, len(xyz)) + xyz.tobytes()
