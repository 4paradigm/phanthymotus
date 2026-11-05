"""Bounded, unmerged local keyframes for a future point-cloud SLAM backend.

The caller must provide tracking poses already validated and aligned to the
cloud timestamp, with points transformed into the pose child frame. This class
does not detect loops, optimize a graph, or establish timestamp synchronization.
"""
from dataclasses import dataclass

import numpy as np

from .pointcloud_slam_odom import PoseSample


class CapacityError(ValueError):
    """A frame does not fit without discarding retained history."""


@dataclass(frozen=True)
class Keyframe:
    pose: PoseSample
    cloud_stamp_ns: int
    points: np.ndarray


@dataclass(frozen=True)
class Segment:
    segment_id: str
    frames: tuple


class KeyframeStore:
    """Keep history until full; never merge segments or silently evict maps."""

    def __init__(self, *, max_frames, max_points):
        if any(type(v) is not int or v <= 0 for v in (max_frames, max_points)):
            raise ValueError('capacity must be a positive integer')
        self._max_frames, self._max_points = max_frames, max_points
        self._segments = []
        self._frame_count = self._point_count = 0
        self._lost = False

    @property
    def segments(self):
        return tuple(Segment(segment_id, tuple(frames)) for segment_id, frames in self._segments)

    @property
    def requires_new_segment(self):
        return self._lost and bool(self._segments)

    def mark_lost(self):
        self._lost = True

    def add(self, pose, points, *, cloud_stamp_ns):
        if not isinstance(pose, PoseSample):
            raise ValueError('validated tracking pose required')
        if (type(cloud_stamp_ns) is not int or cloud_stamp_ns <= 0 or
                cloud_stamp_ns // 1_000_000 != pose.stamp_ms):
            raise ValueError('cloud timestamp does not match aligned pose timestamp')
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError('local cloud must contain Nx3 points')
        try:
            points = np.asarray(points, dtype=np.float64)
            if not np.isfinite(points).all():
                raise ValueError('invalid local cloud')
        except (TypeError, OverflowError) as exc:
            raise ValueError('invalid local cloud') from exc
        if self._segments:
            current_id, frames = self._segments[-1]
            previous = frames[-1].pose
            if pose.stamp_ms <= previous.stamp_ms:
                raise ValueError('keyframe timestamps must increase')
            if pose.segment_id == current_id:
                if self._lost:
                    raise ValueError('recovery requires a new segment')
                if (pose.frame_id, pose.child_frame_id, pose.clock_id) != (
                        previous.frame_id, previous.child_frame_id, previous.clock_id):
                    raise ValueError('pose frames or clock changed within segment')
            elif any(segment_id == pose.segment_id for segment_id, _ in self._segments):
                raise ValueError('segment cannot be reused')
        points = self._retain(pose, points, cloud_stamp_ns)
        if points is None:
            return False
        if not self._segments or pose.segment_id != self._segments[-1][0]:
            self._segments.append((pose.segment_id, []))
        self._segments[-1][1].append(Keyframe(pose, cloud_stamp_ns, points))
        self._frame_count += 1
        self._point_count += len(points)
        self._lost = False
        return True

    def _retain(self, pose, points, stamp):
        if (self._frame_count >= self._max_frames or
                self._point_count + len(points) > self._max_points):
            raise CapacityError('keyframe capacity reached')
        # Immutable bytes prevent callers from mutating retained map history.
        return np.frombuffer(points.tobytes(), dtype=np.float64).reshape(-1, 3)
