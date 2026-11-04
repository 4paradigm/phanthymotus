"""Bounded live-input state and odometry graph; loop acceptance is not implemented.

The caller must establish the declared clock relation and calibrate the
extrinsic. ``now`` is monotonic receipt time, used only to detect stale input;
it is never substituted for a measurement timestamp. Calls must be serialized.
"""
from collections import deque
import math

import numpy as np

from .pointcloud_slam_alignment import align_pose
from .pointcloud_slam_keyframes import KeyframeStore
from .pointcloud_slam_odom import decode_slam_odom


def _rigid(value):
    m = np.asarray(value, dtype=float)
    if (m.shape != (4, 4) or not np.isfinite(m).all() or
            not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(m[:3, :3]), 1., atol=1e-6)):
        raise ValueError('extrinsic must be a rigid transform')
    return m.copy()


def _matrix(pose):
    x, y, z, w = pose.quaternion
    x, y, z, w = np.asarray([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    m = np.eye(4)
    m[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    m[:3, 3] = pose.position
    return m


class SlamStream:
    def __init__(self, *, cloud_frame, child_frame, clock_id, child_from_cloud,
                 max_gap_ms, max_age_s, max_pending, max_cloud_points,
                 max_frames, max_points):
        if any(not isinstance(s, str) or not s.strip() for s in
               (cloud_frame, child_frame, clock_id)):
            raise ValueError('cloud frame, child frame and verified clock required')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v <= 0 for v in (max_gap_ms, max_age_s)):
            raise ValueError('positive finite gap and age required')
        if any(type(v) is not int or v <= 0 for v in (max_pending, max_cloud_points)):
            raise ValueError('positive bounded capacities required')
        self.cloud_frame, self.child_frame, self.clock_id = cloud_frame, child_frame, clock_id
        self.extrinsic = _rigid(child_from_cloud)
        self.max_gap_ms, self.max_age_s = max_gap_ms, max_age_s
        self.max_pending, self.max_cloud_points = max_pending, max_cloud_points
        self.store = KeyframeStore(max_frames=max_frames, max_points=max_points)
        self._poses = deque(maxlen=512)
        self._pending = deque()
        self._last_received = None
        self.last_error = ''

    @property
    def pending_count(self):
        return len(self._pending)

    def _lose(self):
        self.store.mark_lost()
        self._poses.clear()
        self._pending.clear()
        self._last_received = None

    def odom(self, sample, *, now):
        try:
            pose = decode_slam_odom(sample)
            if pose.clock_id != self.clock_id or pose.child_frame_id != self.child_frame:
                raise ValueError('pose clock or child frame mismatch')
            if self._poses and pose == self._poses[-1]:
                return
            if self._poses and pose.stamp_ms <= self._poses[-1].stamp_ms:
                raise ValueError('pose timestamp did not increase')
        except ValueError as error:
            self.last_error = str(error)
            self._lose()
            return
        if self._last_received is not None and now - self._last_received > self.max_age_s:
            self._lose()
        if self._poses and pose.segment_id != self._poses[-1].segment_id:
            self._poses.clear()
            self._pending.clear()
        self._poses.append(pose)
        self._last_received = now
        self._drain(now)

    def cloud(self, stamp_ns, frame, points, *, now):
        if self._last_received is None or now - self._last_received > self.max_age_s:
            self._lose()
            raise ValueError('odometry stale or unavailable')
        if type(stamp_ns) is not int or stamp_ns <= 0 or frame != self.cloud_frame:
            raise ValueError('invalid cloud frame or timestamp')
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or not 0 < len(points) <= self.max_cloud_points:
            raise ValueError('invalid cloud size')
        points = np.asarray(points, dtype=float)
        if not np.isfinite(points).all():
            raise ValueError('cloud must be finite')
        if len(self._pending) == self.max_pending:
            self._pending.popleft()
        self._pending.append((stamp_ns, points.copy(), now))
        self._drain(now)

    def _drain(self, now):
        remaining = deque()
        while self._pending:
            stamp, points, received = self._pending.popleft()
            if now - received > self.max_age_s:
                continue
            if stamp > self._poses[-1].stamp_ms * 1_000_000:
                remaining.append((stamp, points, received))
                continue
            try:
                pose = align_pose(self._poses, stamp, max_gap_ms=self.max_gap_ms)
                local = points @ self.extrinsic[:3, :3].T + self.extrinsic[:3, 3]
                self.store.add(pose, local, cloud_stamp_ns=stamp)
            except ValueError as error:
                self.last_error = str(error)
                continue
        self._pending = remaining

    def graph(self):
        """Return separate odometry-only graphs with no claimed loop edges."""
        result = []
        for segment in self.store.segments:
            poses = [_matrix(frame.pose) for frame in segment.frames]
            edges = [{'from': i - 1, 'to': i,
                      'transform': np.linalg.inv(poses[i - 1]) @ poses[i]}
                     for i in range(1, len(poses))]
            result.append({'segment_id': segment.segment_id, 'poses': poses, 'edges': edges})
        return result
