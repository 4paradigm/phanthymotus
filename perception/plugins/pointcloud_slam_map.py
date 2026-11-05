"""Rebuild one experimental submap from retained child-frame keyframes."""
import numpy as np

from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_stream import _rigid


def map_from_odom(map_from_child, odom_from_child):
    """Compute T_map_odom for the same child and capture instant."""
    return _rigid(map_from_child) @ np.linalg.inv(_rigid(odom_from_child))


def rebuild_segment(segment, poses):
    """Apply one pose per keyframe without merging independent segments.

    The caller must supply poses in keyframe order in this segment's own map
    frame. This function does not decide whether an optimization is valid.
    """
    if not isinstance(segment, Segment) or len(poses) != len(segment.frames):
        raise ValueError('each segment keyframe needs one pose')
    if not segment.frames:
        raise ValueError('segment has no keyframes')
    matrices = [_rigid(pose) for pose in poses]
    return np.concatenate([frame.points @ matrix[:3, :3].T + matrix[:3, 3]
                           for frame, matrix in zip(segment.frames, matrices)], axis=0)
