"""Bounded live-input state and odometry graph; loop acceptance is not implemented.

The caller must establish the declared clock relation and calibrate the
extrinsic. ``now`` is monotonic receipt time, used only to detect stale input;
it is never substituted for a measurement timestamp. Calls must be serialized.
"""
from collections import deque
import math

import numpy as np

from .pointcloud_slam_alignment import align_pose
from .pointcloud_slam_keyframes import CapacityError, KeyframeStore
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
                 max_frames, max_points, max_measurement_age_ms=None,
                 max_future_skew_ms=None):
        if any(not isinstance(s, str) or not s.strip() for s in
               (cloud_frame, child_frame, clock_id)):
            raise ValueError('cloud frame, child frame and verified clock required')
        if (child_frame.count('{segment_id}') > 1 or
                '{' in child_frame.replace('{segment_id}', '') or
                '}' in child_frame.replace('{segment_id}', '')):
            raise ValueError('child frame may use only one {segment_id} placeholder')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v <= 0 for v in (max_gap_ms, max_age_s)):
            raise ValueError('positive finite gap and age required')
        if any(type(v) is not int or v <= 0 for v in (max_pending, max_cloud_points)):
            raise ValueError('positive bounded capacities required')
        if (max_measurement_age_ms is not None or max_future_skew_ms is not None):
            if (isinstance(max_measurement_age_ms, bool) or
                    not isinstance(max_measurement_age_ms, (int, float)) or
                    not math.isfinite(max_measurement_age_ms) or max_measurement_age_ms <= 0 or
                    isinstance(max_future_skew_ms, bool) or
                    not isinstance(max_future_skew_ms, (int, float)) or
                    not math.isfinite(max_future_skew_ms) or max_future_skew_ms < 0):
                raise ValueError('valid measurement age and future skew are required')
        self.max_measurement_age_ms = max_measurement_age_ms
        self.max_future_skew_ms = max_future_skew_ms
        self.cloud_frame, self.child_frame, self.clock_id = cloud_frame, child_frame, clock_id
        self.extrinsic = _rigid(child_from_cloud)
        self.max_gap_ms, self.max_age_s = max_gap_ms, max_age_s
        self.max_pending, self.max_cloud_points = max_pending, max_cloud_points
        self.store = KeyframeStore(max_frames=max_frames, max_points=max_points)
        self._poses = deque(maxlen=512)
        self._pending = deque()
        self._last_received = None
        self._last_cloud_received = None
        self._last_cloud_stamp_ns = None
        self.last_error = ''
        self._odom_state = 'waiting'
        self._dropped_pending = 0
        self._capacity_reached = False

    @property
    def pending_count(self):
        return len(self._pending)

    def _lose(self):
        self.store.mark_lost()
        self._poses.clear()
        self._pending.clear()
        self._last_received = None
        self._odom_state = 'lost'

    def status(self, *, now, measurement_now_ns=None):
        odom_state = self._odom_state
        last_error = self.last_error
        if self._last_received is not None:
            try:
                if now - self._last_received > self.max_age_s:
                    raise ValueError('odometry receipt stale')
                self._check_measurement_time(self._poses[-1].stamp_ms * 1_000_000,
                                             measurement_now_ns)
            except ValueError as error:
                last_error = str(error)
                odom_state = 'stale'
        segments = self.store.segments
        cloud_state = 'waiting'
        if self._last_cloud_received is not None:
            cloud_state = 'active'
            try:
                if now - self._last_cloud_received > self.max_age_s:
                    raise ValueError('pointcloud receipt stale')
                self._check_measurement_time(self._last_cloud_stamp_ns, measurement_now_ns)
            except ValueError:
                cloud_state = 'stale'
        integration = ('waiting_for_tracking' if odom_state != 'tracking' else
                       'waiting_for_new_segment' if self.store.requires_new_segment else
                       'capacity_reached' if self._capacity_reached else
                       'collecting_unverified')
        return {'odometry': odom_state, 'cloud': cloud_state, 'pending_clouds': len(self._pending),
                'integration': integration,
                'dropped_pending': self._dropped_pending, 'segments': len(segments),
                'keyframes': sum(len(segment.frames) for segment in segments),
                'last_error': last_error}

    def _check_measurement_time(self, stamp_ns, measurement_now_ns):
        if self.max_measurement_age_ms is None:
            return
        if type(measurement_now_ns) is not int or measurement_now_ns <= 0:
            raise ValueError('measurement clock unavailable')
        age_ns = measurement_now_ns - stamp_ns
        if (age_ns > self.max_measurement_age_ms * 1_000_000 or
                age_ns < -self.max_future_skew_ms * 1_000_000):
            raise ValueError('measurement timestamp stale or in the future')

    def odom(self, sample, *, now, measurement_now_ns=None):
        try:
            pose = decode_slam_odom(sample)
        except ValueError as error:
            self.last_error = str(error)
            vendor = sample.get('vendor') if isinstance(sample, dict) else None
            quality = vendor.get('motus_slam') if isinstance(vendor, dict) else None
            if isinstance(quality, dict) and quality.get('status') == 'tracking':
                # An invalid tracking sample is a pairing barrier. A later
                # validated sample may resume only if its segment remains valid.
                self._poses.clear()
                self._pending.clear()
                self._last_received = None
                self._odom_state = 'invalid'
            else:
                self._lose()
            return
        try:
            if (self._poses and pose.segment_id == self._poses[-1].segment_id and
                    pose.stamp_ms < self._poses[-1].stamp_ms):
                self.last_error = 'late odometry sample dropped'
                return
            self._check_measurement_time(pose.stamp_ms * 1_000_000, measurement_now_ns)
            expected_child = self.child_frame.replace('{segment_id}', pose.segment_id)
            if pose.clock_id != self.clock_id or pose.child_frame_id != expected_child:
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
        self._odom_state = 'tracking'
        self._drain(now, measurement_now_ns)

    def cloud(self, stamp_ns, frame, points, *, now, measurement_now_ns=None):
        if self._last_received is None or now - self._last_received > self.max_age_s:
            self._lose()
            raise ValueError('odometry stale or unavailable')
        if type(stamp_ns) is not int or stamp_ns <= 0 or frame != self.cloud_frame:
            raise ValueError('invalid cloud frame or timestamp')
        if self._last_cloud_stamp_ns is not None and stamp_ns <= self._last_cloud_stamp_ns:
            raise ValueError('cloud timestamp did not increase')
        self._check_measurement_time(stamp_ns, measurement_now_ns)
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or not 0 < len(points) <= self.max_cloud_points:
            raise ValueError('invalid cloud size')
        points = np.asarray(points, dtype=float)
        if not np.isfinite(points).all():
            raise ValueError('cloud must be finite')
        self._last_cloud_stamp_ns = stamp_ns
        self._last_cloud_received = now
        if len(self._pending) == self.max_pending:
            self._pending.popleft()
            self._dropped_pending += 1
        self._pending.append((stamp_ns, points.copy(), now))
        self._drain(now, measurement_now_ns)

    def _drain(self, now, measurement_now_ns):
        remaining = deque()
        while self._pending:
            stamp, points, received = self._pending.popleft()
            if now - received > self.max_age_s:
                continue
            try:
                self._check_measurement_time(stamp, measurement_now_ns)
            except ValueError as error:
                self.last_error = str(error)
                continue
            if stamp > self._poses[-1].stamp_ms * 1_000_000:
                remaining.append((stamp, points, received))
                continue
            try:
                pose = align_pose(self._poses, stamp, max_gap_ms=self.max_gap_ms)
                local = points @ self.extrinsic[:3, :3].T + self.extrinsic[:3, 3]
                self.store.add(pose, local, cloud_stamp_ns=stamp)
                self._capacity_reached = False
            except CapacityError as error:
                self.last_error = str(error)
                self._capacity_reached = True
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
