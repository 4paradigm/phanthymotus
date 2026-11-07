"""Bounded live-input state and odometry graph; loop acceptance is not implemented.

The caller must establish the declared clock relation and calibrate the
extrinsic. ``now`` is monotonic receipt time, used only to detect stale input;
it is never substituted for a measurement timestamp. Calls must be serialized.
"""
from collections import deque, OrderedDict
from copy import deepcopy
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
                 max_future_skew_ms=None, input_contract=None, extrinsic_mode='static'):
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
        self.extrinsic_mode = (input_contract['settings'].get('extrinsic_mode', extrinsic_mode)
                               if input_contract else extrinsic_mode)
        if self.extrinsic_mode not in ('static', 'dynamic'):
            raise ValueError('unsupported extrinsic mode')
        if self.extrinsic_mode == 'dynamic' and child_from_cloud is not None:
            raise ValueError('dynamic extrinsic cannot use a startup matrix')
        self.extrinsic = None if self.extrinsic_mode == 'dynamic' else _rigid(child_from_cloud)
        self.max_gap_ms, self.max_age_s = max_gap_ms, max_age_s
        self.max_pending, self.max_cloud_points = max_pending, max_cloud_points
        self.store = KeyframeStore(max_frames=max_frames, max_points=max_points)
        self._poses = deque(maxlen=512)
        self._pending = deque()
        self._last_received = None
        self._last_cloud_received = None
        self._last_cloud_stamp_ns = None
        self.input_contract = input_contract
        self._input_identity = None
        self._input_segment = None
        self._retired_identities = set()
        self._pose_measurement = None
        self._cloud_identity = None
        self._cloud_measurement = None
        self._cloud_segment = None
        self._retired_cloud_identities = set()
        self._required_new_segment = None
        self._cloud_metadata = OrderedDict()
        self._unmatched_clouds = OrderedDict()
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
        self._last_cloud_stamp_ns = None
        self._cloud_metadata.clear()
        self._unmatched_clouds.clear()
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
                if self._last_cloud_stamp_ns is None or now - self._last_cloud_received > self.max_age_s:
                    raise ValueError('pointcloud receipt stale')
                self._check_measurement_time(self._last_cloud_stamp_ns, measurement_now_ns)
            except ValueError:
                cloud_state = 'stale'
        integration = ('waiting_for_new_segment' if self._required_new_segment is not None else
                       'waiting_for_tracking' if odom_state != 'tracking' else
                       'waiting_for_new_segment' if self.store.requires_new_segment else
                       'storage_fault' if getattr(self.store, 'storage_error', '') else
                       'capacity_reached' if self._capacity_reached else
                       'waiting_for_cloud_metadata' if self.input_contract and self._unmatched_clouds else
                       'collecting_unverified')
        return {'odometry': odom_state, 'cloud': cloud_state, 'pending_clouds': len(self._pending),
                'integration': integration,
                'dropped_pending': self._dropped_pending, 'segments': len(segments),
                'unmatched_clouds': len(self._unmatched_clouds),
                'keyframes': sum(len(segment.frames) for segment in segments),
                'last_error': last_error,
                'storage': ({'mode': 'paged', 'disk_bytes': self.store.disk_bytes,
                             'skipped_stationary': self.store.skipped_stationary,
                             'active_frame_limit': self.store._max_frames}
                            if hasattr(self.store, 'disk_bytes') else {'mode': 'memory'})}

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
            if self.input_contract:
                from .pointcloud_slam_input import frame_identity
                quality = sample['vendor']['motus_slam']
                declared = next(d for d in self.input_contract['inputs'].values() if d['kind'] == 'pose')
                if declared.get('frame_id') and pose.frame_id != declared['frame_id'].replace('{segment_id}', pose.segment_id):
                    raise ValueError('pose frame differs from input declaration; restart inputs')
                mount = declared.get('extrinsic')
                if isinstance(mount, dict) and quality.get('extrinsic_version') != mount['version']:
                    raise ValueError('pose extrinsic_version changed or unknown; restart inputs')
                if declared.get('pose_reference') and quality.get('pose_reference') != declared['pose_reference']:
                    raise ValueError('pose_reference changed or unknown; restart inputs')
                measurement = quality.get('measurement')
                identity = frame_identity(measurement, clock_id=self.clock_id,
                    experimental=self.input_contract['experimental'],
                    relation_id=self.input_contract.get('relation_id'))
                if measurement['stamp_ns'] // 1_000_000 != pose.stamp_ms:
                    raise ValueError('pose measurement timestamp mismatch')
                if identity in self._retired_identities:
                    self.last_error = 'retired input epoch/generation dropped'
                    return


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
                self._cloud_metadata.clear()
                self._unmatched_clouds.clear()
                self._odom_state = 'invalid'
            else:
                self._lose()
            return
        try:
            self._check_measurement_time(pose.stamp_ms * 1_000_000, measurement_now_ns)
            expected_child = self.child_frame.replace('{segment_id}', pose.segment_id)
            if pose.clock_id != self.clock_id or pose.child_frame_id != expected_child:
                raise ValueError('pose clock or child frame mismatch')
            if self.input_contract:
                same_segment = pose.segment_id == self._input_segment
                changed = self._input_identity is not None and identity != self._input_identity
                mapping_changed = (identity == self._input_identity and self._pose_measurement is not None and
                    (measurement['mapping'] != self._pose_measurement['mapping'] or
                     measurement['source_clock'] != self._pose_measurement['source_clock']))
                if self._required_new_segment == pose.segment_id or (same_segment and (changed or mapping_changed)):
                    self._required_new_segment = self._input_segment
                    raise ValueError('input epoch/generation or clock mapping change requires a new segment')
            if (self._poses and pose.segment_id == self._poses[-1].segment_id and
                    pose.stamp_ms < self._poses[-1].stamp_ms):
                self.last_error = 'late odometry sample dropped'
                return
            if self.input_contract:
                if self._input_identity is not None and identity != self._input_identity:
                    self._retired_identities.add(self._input_identity)
                    self._lose()
                self._input_identity = identity
                self._input_segment = pose.segment_id
                self._pose_measurement = deepcopy(measurement)
                self._required_new_segment = None
            if self._poses and pose == self._poses[-1]:
                return
            if self._poses and pose.segment_id != self._poses[-1].segment_id:
                self._poses.clear()
                self._pending.clear()
                self._last_cloud_stamp_ns = None
                self._last_cloud_received = None
                self._cloud_metadata.clear()
                self._unmatched_clouds.clear()
            if self._poses and pose.stamp_ms <= self._poses[-1].stamp_ms:
                raise ValueError('pose timestamp did not increase')
        except ValueError as error:
            self.last_error = str(error)
            self._lose()
            return
        if self._last_received is not None and now - self._last_received > self.max_age_s:
            self._lose()
        self._poses.append(pose)
        self._last_received = now
        self._odom_state = 'tracking'
        self._drain(now, measurement_now_ns)

    def _capture_context(self, packet):
        declaration = self.input_contract.get('extrinsic', {})
        if self.extrinsic_mode == 'dynamic':
            evidence = packet.get('extrinsic')
            if not isinstance(evidence, dict):
                raise ValueError('dynamic extrinsic capture evidence is required')
            expected = dict(stamp_ns=packet['stamp_ns'], clock_id=packet['clock_id'],
                            clock_epoch=packet['clock_epoch'],
                            parent_frame=self.child_frame.replace('{segment_id}', self._input_segment or ''),
                            child_frame=packet['frame_id'], source=declaration['source'],
                            version=declaration['version'])
            if any(evidence.get(k) != v for k, v in expected.items()):
                raise ValueError('dynamic extrinsic timestamp/clock/frame/source/version mismatch')
            matrix = _rigid(evidence.get('matrix'))
        else:
            evidence, matrix = declaration, self.extrinsic
        return deepcopy(dict(child_from_cloud=matrix.tolist(), extrinsic=evidence,
                             measurement=packet))

    def cloud_metadata(self, packet, *, now, measurement_now_ns=None):
        if not self.input_contract:
            return
        from .pointcloud_slam_input import frame_identity
        try:
            identity = frame_identity(packet, clock_id=self.clock_id, payload=True,
                experimental=self.input_contract['experimental'],
                relation_id=self.input_contract.get('relation_id'))
            if packet['frame_id'] != self.cloud_frame:
                raise ValueError('cloud metadata frame mismatch')
            self._check_measurement_time(packet['stamp_ns'], measurement_now_ns)
            context = self._capture_context(packet)
            independent = self.input_contract.get('clock_relation_mode') == 'verified_relation'
            if not independent:
                if identity != self._input_identity:
                    raise ValueError('cloud epoch/generation does not match current pose')
                if (packet['mapping'] != self._pose_measurement['mapping'] or
                        packet['source_clock'] != self._pose_measurement['source_clock']):
                    raise ValueError('cloud and pose clock mapping evidence differs')
            else:
                if identity in self._retired_cloud_identities:
                    raise ValueError('retired cloud epoch/generation dropped')
                changed = self._cloud_identity is not None and identity != self._cloud_identity
                mapping_changed = (identity == self._cloud_identity and
                    self._cloud_measurement is not None and
                    (packet['mapping'] != self._cloud_measurement['mapping'] or
                     packet['source_clock'] != self._cloud_measurement['source_clock']))
                if mapping_changed and self._cloud_segment == self._input_segment:
                    self._required_new_segment = self._input_segment
                    self._lose()
                    raise ValueError('cloud clock mapping changed; a new segment is required')
                if changed:
                    self._retired_cloud_identities.add(self._cloud_identity)
                    self._pending.clear()
                    self._cloud_metadata.clear()
                    self._unmatched_clouds.clear()
                    self._last_cloud_stamp_ns = self._last_cloud_received = None
                    if self._cloud_segment == self._input_segment:
                        self._required_new_segment = self._input_segment
                        self._lose()
                self._cloud_identity = identity
                self._cloud_measurement = deepcopy(packet)
                if self._required_new_segment is not None:
                    raise ValueError('cloud source restart requires a new pose segment')
                if self._input_identity is None or identity[:2] != self._input_identity[:2]:
                    raise ValueError('cloud normalized clock epoch does not match current pose')
                self._cloud_segment = self._input_segment
            key = (packet['stamp_ns'], packet['frame_id'], packet['data_sha256'])
            self._cloud_metadata[key] = context
            while len(self._cloud_metadata) > self.max_pending:
                self._cloud_metadata.popitem(last=False)
            self._pair_cloud(key, now=now, measurement_now_ns=measurement_now_ns)
        except ValueError as error:
            self.last_error = str(error)

    def _pair_cloud(self, key, *, now, measurement_now_ns):
        if key not in self._cloud_metadata or key not in self._unmatched_clouds:
            if any(k[:2] == key[:2] and k[2] != key[2]
                   for k in (self._cloud_metadata if key in self._unmatched_clouds else self._unmatched_clouds)):
                self.last_error = 'cloud payload does not match measurement metadata'
            return
        points, received = self._unmatched_clouds.pop(key)
        context = self._cloud_metadata.pop(key)
        if now - received > self.max_age_s:
            self.last_error = 'pointcloud receipt stale while waiting for metadata'
            return
        self._accept_cloud(key[0], key[1], points, now=now, received=received,
                           measurement_now_ns=measurement_now_ns, capture_context=context)

    def cloud(self, stamp_ns, frame, points, *, now, measurement_now_ns=None, data_sha256=None,
              capture_context=None):
        if not self.input_contract:
            return self._accept_cloud(stamp_ns, frame, points, now=now,
                                      measurement_now_ns=measurement_now_ns, capture_context=capture_context)
        if type(stamp_ns) is not int or stamp_ns <= 0 or frame != self.cloud_frame:
            raise ValueError('invalid cloud frame or timestamp')
        if self._last_received is None or now - self._last_received > self.max_age_s:
            raise ValueError('odometry stale or unavailable')
        self._check_measurement_time(stamp_ns, measurement_now_ns)
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or
                not 0 < len(points) <= self.max_cloud_points or not np.isfinite(points).all()):
            raise ValueError('invalid cloud size or coordinates')
        if not isinstance(data_sha256, str) or len(data_sha256) != 64:
            raise ValueError('cloud payload digest is required')
        key = (stamp_ns, frame, data_sha256)
        self._unmatched_clouds[key] = (points.copy(), now)
        while len(self._unmatched_clouds) > self.max_pending:
            self._unmatched_clouds.popitem(last=False)
            self._dropped_pending += 1
        self._pair_cloud(key, now=now, measurement_now_ns=measurement_now_ns)

    def _accept_cloud(self, stamp_ns, frame, points, *, now, measurement_now_ns=None, received=None,
                      capture_context=None):
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
        if self.extrinsic_mode == 'dynamic' and capture_context is None:
            raise ValueError('dynamic extrinsic capture context is required')
        if capture_context is not None:
            _rigid(capture_context.get('child_from_cloud'))
        self._last_cloud_stamp_ns = stamp_ns
        self._last_cloud_received = now if received is None else received
        if len(self._pending) == self.max_pending:
            self._pending.popleft()
            self._dropped_pending += 1
        self._pending.append((stamp_ns, points.copy(), self._last_cloud_received, deepcopy(capture_context)))
        self._drain(now, measurement_now_ns)

    def _drain(self, now, measurement_now_ns):
        remaining = deque()
        while self._pending:
            stamp, points, received, context = self._pending.popleft()
            if now - received > self.max_age_s:
                continue
            try:
                self._check_measurement_time(stamp, measurement_now_ns)
            except ValueError as error:
                self.last_error = str(error)
                continue
            if stamp > self._poses[-1].stamp_ms * 1_000_000:
                remaining.append((stamp, points, received, context))
                continue
            try:
                pose = align_pose(self._poses, stamp, max_gap_ms=self.max_gap_ms)
                matrix = _rigid(context['child_from_cloud']) if context is not None else self.extrinsic
                local = points @ matrix[:3, :3].T + matrix[:3, 3]
                self.store.add(pose, local, cloud_stamp_ns=stamp, capture_context=context)
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
