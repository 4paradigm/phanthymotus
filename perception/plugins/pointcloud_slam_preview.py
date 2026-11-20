"""Bounded snapshots in Core's existing sensor/mapping wire format."""
import math
import struct

import numpy as np

from .pointcloud_slam_transforms import rigid

# One stable canvas output for both live and archived maps.
PREVIEW_TOPIC = '/pointcloud_slam/map'

# Additive v2 flags; xyz payload and existing display-yaw convention are unchanged.
LAST_MAPPED, CAMERA_OPTICAL, BODY, HEADING_UNAVAILABLE = 4, 8, 16, 32


def encode_mapping(points, pose, *, max_points, pose_reference=None):
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
    flags = 3 | LAST_MAPPED
    axis = None
    if pose_reference == 'camera_optical':
        axis, flags = 2, flags | CAMERA_OPTICAL
    elif pose_reference == 'body':
        axis, flags = 0, flags | BODY
    if axis is None or np.linalg.norm(matrix[:2, axis]) < 1e-6:
        ros_yaw, flags = 0., flags | HEADING_UNAVAILABLE
    else:
        ros_yaw = math.atan2(matrix[1, axis], matrix[0, axis])
    display_yaw = -ros_yaw if ros_yaw != 0.0 else 0.0  # avoid negative-zero bytes
    return struct.pack('<fffBI', matrix[0, 3], matrix[1, 3], display_yaw, flags, len(xyz)) + xyz.tobytes()


def historical_mapping(packet):
    """Old snapshots lack axis evidence; retain the map but hide their heading."""
    flags = packet[12] | LAST_MAPPED
    if not flags & (CAMERA_OPTICAL | BODY):
        flags |= HEADING_UNAVAILABLE
    return packet[:12] + bytes([flags]) + packet[13:]
