"""Rebuild one experimental submap from retained child-frame keyframes."""
import numpy as np

from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_stream import _rigid


def map_from_odom(map_from_child, odom_from_child):
    """Compute T_map_odom for the same child and capture instant."""
    return _rigid(map_from_child) @ np.linalg.inv(_rigid(odom_from_child))


def rebuild_segment(segment, poses, *, max_points=None):
    """Apply one pose per keyframe without merging independent segments.

    The caller must supply poses in keyframe order in this segment's own map
    frame. This function does not decide whether an optimization is valid.
    """
    if not isinstance(segment, Segment) or len(poses) != len(segment.frames):
        raise ValueError('each segment keyframe needs one pose')
    if not segment.frames:
        raise ValueError('segment has no keyframes')
    matrices = [_rigid(pose) for pose in poses]
    total = sum(len(frame.points) for frame in segment.frames)
    if max_points is not None and (type(max_points) is not int or max_points <= 0):
        raise ValueError('positive preview point budget required')
    indices = (np.linspace(0, total - 1, max_points, dtype=int)
               if max_points is not None and total > max_points else None)
    chunks, offset = [], 0
    for frame, matrix in zip(segment.frames, matrices):
        points = frame.points
        if indices is not None:
            selected = indices[(indices >= offset) & (indices < offset + len(points))] - offset
            points = points[selected]
        chunks.append(points @ matrix[:3, :3].T + matrix[:3, 3])
        offset += len(frame.points)
    return np.concatenate(chunks, axis=0)
