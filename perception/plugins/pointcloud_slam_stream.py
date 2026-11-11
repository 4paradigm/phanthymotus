"""Bounded admission of raw clouds paired with validated external odometry.

The caller establishes comparable timestamp and frame coordinates before
admission. ``now`` is monotonic receipt time, used only to detect stale input;
it is never substituted for a measurement timestamp. Calls must be serialized.
Optional frame evidence is hash-checked when present; a producer's clock is
never promoted to verified merely because timestamps look plausible.
"""
from collections import deque, OrderedDict
from copy import deepcopy
import math

from .pointcloud_slam_alignment import align_pose
from .pointcloud_slam_odom import decode_slam_odom
from .pointcloud_slam_transforms import rigid as _rigid


class SlamStream:
    def __init__(self, *, cloud_frame, child_frame, clock_id, child_from_cloud,
                 max_gap_ms, max_age_s, max_pending, max_cloud_points,
                 on_frame, max_measurement_age_ms=None,
                 max_future_skew_ms=None, input_contract=None, extrinsic_mode='static'):
        if any(not isinstance(s, str) or not s.strip() for s in
               (cloud_frame, child_frame, clock_id)):
            raise ValueError('cloud frame, child frame and clock identity required')
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
        self.on_frame = on_frame
        self._segments = OrderedDict()
        self._requires_new_segment = False
        self._admitted = 0
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
        # Wait briefly for optional ROS sidecars; do not allow any known digest
        # mismatch to fall back to time-based alignment.
        self._metadata_grace_s = min(0.2, max_age_s / 4)
        self._metadata_fallbacks = 0
        self.last_error = ''
        self._odom_state = 'waiting'
        self._fusion_pause = None
        self._dropped_pending = 0

    @property
    def pending_count(self):
        return len(self._pending)

    def _lose(self):
        self._fusion_pause = None
        self._requires_new_segment = bool(self._segments)
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
                       'waiting_for_new_segment' if self._poses and self._poses[-1].segment_id in self._segments and
                           (self._requires_new_segment or self._poses[-1].segment_id != next(reversed(self._segments))) else
                       'waiting_for_cloud_metadata' if self.input_contract and self._unmatched_clouds else
                       'admitting')
        return {'odometry': odom_state, 'cloud': cloud_state, 'pending_clouds': len(self._pending),
                'integration': integration, 'dropped_pending': self._dropped_pending,
                'segments': len(self._segments), 'admitted_frames': self._admitted,
                'unmatched_clouds': len(self._unmatched_clouds),
                'frames_without_sidecar': self._metadata_fallbacks,
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

    def _pause_fusion(self, sample, now, measurement_now_ns):
        """Pause admission within one bounded fusion segment; retain no clouds."""
        vendor = sample.get('vendor', {}) if isinstance(sample, dict) else {}
        quality, fusion = vendor.get('motus_slam', {}), vendor.get('fusion', {})
        if not isinstance(quality, dict) or quality.get('status') not in ('coasting', 'recovering'):
            return False
        anchor = self._fusion_pause[0] if self._fusion_pause else self._poses[-1] if self._poses else None
        stamp = sample.get('stamp_ms')
        valid = (anchor is not None and not self._requires_new_segment and self._required_new_segment is None
                 and sample.get('schema') == 'motus.odom/1' and sample.get('pose') is None
                 and isinstance(fusion, dict) and fusion.get('enabled') is True
                 and fusion.get('mapping_allowed') is False and fusion.get('status') == quality['status']
                 and quality.get('segment_id') == anchor.segment_id and quality.get('clock_id') == self.clock_id
                 and type(stamp) is int and anchor.stamp_ms <= stamp <= anchor.stamp_ms+1000
                 and (self._fusion_pause is None or now-self._fusion_pause[1] <= 1.))
        if not valid:
            self._lose()
            self.last_error = 'invalid or expired fusion pause; new segment required'
            return True
        try:
            self._check_measurement_time(stamp*1_000_000, measurement_now_ns)
        except ValueError as error:
            self._lose()
            self.last_error = str(error)
            return True
        self._fusion_pause = (anchor, self._fusion_pause[1] if self._fusion_pause else now)
        self._poses.clear()
        self._pending.clear()
        self._last_received = None
        self._cloud_metadata.clear()
        self._unmatched_clouds.clear()
        self._odom_state = quality['status']
        self.last_error = 'map integration paused during short-term fusion'
        return True

    def odom(self, sample, *, now, measurement_now_ns=None):
        if self._pause_fusion(sample, now, measurement_now_ns):
            return
        try:
            pose = decode_slam_odom(sample)
            if self.input_contract:
                from .pointcloud_slam_input import frame_identity
                quality = sample['vendor']['motus_slam']
                declared = next(d for d in self.input_contract['inputs'].values() if d['kind'] == 'pose')
                expected_frame = declared.get('frame_id_template', declared.get('frame_id'))
                if expected_frame and pose.frame_id != expected_frame.replace('{segment_id}', pose.segment_id):
                    raise ValueError('pose frame differs from input declaration; restart inputs')
                mount = declared.get('extrinsic')
                if isinstance(mount, dict):
                    observed_mount = quality.get('extrinsic')
                    observed_version = (quality.get('extrinsic_version') or
                        (observed_mount.get('version') if isinstance(observed_mount, dict) else None))
                    if observed_version != mount['version']:
                        raise ValueError('pose extrinsic_version changed or unknown; restart inputs')
                observed_reference = quality.get('pose_reference') or sample['vendor'].get('pose_reference')
                if declared.get('pose_reference') and observed_reference != declared['pose_reference']:
                    raise ValueError('pose_reference changed or unknown; restart inputs')
                measurement = quality.get('measurement')
                identity = None
                if measurement is not None:
                    # Malformed claimed evidence cannot silently be ignored.
                    identity = frame_identity(measurement, clock_id=self.clock_id,
                        relation_id=self.input_contract.get('relation_id'))
                    if measurement['stamp_ns'] // 1_000_000 != pose.stamp_ms:
                        raise ValueError('pose measurement timestamp mismatch')
                    if identity in self._retired_identities:
                        self.last_error = 'retired input epoch/generation dropped'
                        return
                elif self._input_identity is not None:
                    raise ValueError('pose measurement evidence lost after prior verified segment')

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
                mapping_changed = (measurement is not None and identity == self._input_identity and
                    self._pose_measurement is not None and
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
        if self._fusion_pause is not None:
            anchor, started = self._fusion_pause
            if (pose.segment_id == anchor.segment_id and
                    (now-started > 1. or not anchor.stamp_ms < pose.stamp_ms <= anchor.stamp_ms+1000
                     or pose.frame_id != anchor.frame_id or pose.child_frame_id != anchor.child_frame_id)):
                self._lose()
            else:
                self._fusion_pause = None
        if self._last_received is not None and now - self._last_received > self.max_age_s:
            self._lose()
        self._poses.append(pose)
        self._last_received = now
        self._odom_state = 'tracking'
        self._drain(now, measurement_now_ns)
        self._release_without_metadata(now, measurement_now_ns)

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
                relation_id=self.input_contract.get('relation_id'))
            if packet['frame_id'] != self.cloud_frame:
                raise ValueError('cloud metadata frame mismatch')
            self._check_measurement_time(packet['stamp_ns'], measurement_now_ns)
            context = self._capture_context(packet)
            independent = self.input_contract.get('clock_relation_mode') == 'verified_relation'
            if not independent:
                if self._input_identity is not None:
                    if identity != self._input_identity:
                        raise ValueError('cloud epoch/generation does not match current pose')
                    if (packet['mapping'] != self._pose_measurement['mapping'] or
                            packet['source_clock'] != self._pose_measurement['source_clock']):
                        raise ValueError('cloud and pose clock mapping evidence differs')
                else:
                    # Same camera source: VO has segment and capture timestamp
                    # but does not need to mint a second motus.slam_frame packet.
                    # Cloud evidence, when available, still cannot change
                    # mapping or generation inside one pose segment.
                    if self._cloud_identity is not None and self._cloud_segment == self._input_segment:
                        if (identity != self._cloud_identity or
                            packet['mapping'] != self._cloud_measurement['mapping'] or
                            packet['source_clock'] != self._cloud_measurement['source_clock']):
                            self._required_new_segment = self._input_segment
                            self._lose()
                            raise ValueError('cloud clock mapping changed within segment')
                    self._cloud_identity = identity
                    self._cloud_measurement = deepcopy(packet)
                    self._cloud_segment = self._input_segment
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
            # A sidecar for this timestamp/frame is negative evidence for
            # any cloud with a different digest. It must NOT later bypass
            # the mismatch via the missing-sidecar fallback.
            mismatch = [other for other in self._unmatched_clouds
                        if other[:2] == key[:2] and
                        any(m[:2] == other[:2] and m[2] != other[2]
                            for m in self._cloud_metadata)]
            for other in mismatch:
                self._unmatched_clouds.pop(other, None)
                self._dropped_pending += 1
                self.last_error = 'cloud payload does not match measurement metadata'
            return
        cloud, received = self._unmatched_clouds.pop(key)
        context = self._cloud_metadata.pop(key)
        if now - received > self.max_age_s:
            self.last_error = 'pointcloud receipt stale while waiting for metadata'
            return
        self._accept_cloud(key[0], key[1], cloud, now=now, received=received,
                           measurement_now_ns=measurement_now_ns, capture_context=context)

    def _release_without_metadata(self, now, measurement_now_ns):
        """Use ordinary timestamp pairing when optional sidecar is missing.

        A known mismatch is never accepted. Dynamic camera frames require
        per-frame transforms and never take this static-only fallback.
        """
        if not self.input_contract or self.extrinsic_mode != 'static':
            return
        for key, (cloud, received) in list(self._unmatched_clouds.items()):
            if now - received < self._metadata_grace_s:
                continue
            self._unmatched_clouds.pop(key, None)
            if any(m[:2] == key[:2] for m in self._cloud_metadata):
                self._dropped_pending += 1
                self.last_error = 'cloud payload does not match measurement metadata'
                continue
            try:
                self._accept_cloud(key[0], key[1], cloud, now=now,
                                   measurement_now_ns=measurement_now_ns, received=received)
                self._metadata_fallbacks += 1
            except ValueError as error:
                self.last_error = str(error)

    def cloud(self, stamp_ns, frame, cloud, *, now, point_count=None, measurement_now_ns=None,
              data_sha256=None, capture_context=None):
        """Retain the opaque native message; XYZ validation belongs to the ROS adapter."""
        if type(stamp_ns) is not int or stamp_ns <= 0 or frame != self.cloud_frame:
            raise ValueError('invalid cloud frame or timestamp')
        if self._fusion_pause is not None:
            if now-self._fusion_pause[1] > 1.:
                self._lose()
            raise ValueError('cloud integration paused during short-term fusion')
        if self._last_received is None or now - self._last_received > self.max_age_s:
            self._lose()
            raise ValueError('odometry stale or unavailable')
        self._check_measurement_time(stamp_ns, measurement_now_ns)
        count = len(cloud) if point_count is None else point_count
        if type(count) is not int or not 0 < count <= self.max_cloud_points:
            raise ValueError('invalid cloud size or capacity')
        if not self.input_contract:
            return self._accept_cloud(stamp_ns, frame, deepcopy(cloud), now=now,
                measurement_now_ns=measurement_now_ns, capture_context=capture_context)
        if not isinstance(data_sha256, str) or len(data_sha256) != 64:
            raise ValueError('cloud payload digest is required')
        key = (stamp_ns, frame, data_sha256)
        self._unmatched_clouds[key] = (deepcopy(cloud), now)
        while len(self._unmatched_clouds) > self.max_pending:
            self._unmatched_clouds.popitem(last=False)
            self._dropped_pending += 1
        self._pair_cloud(key, now=now, measurement_now_ns=measurement_now_ns)

    def _accept_cloud(self, stamp_ns, frame, cloud, *, now, measurement_now_ns=None, received=None,
                      capture_context=None):
        if self._fusion_pause is not None:
            if now-self._fusion_pause[1] > 1.:
                self._lose()
            raise ValueError('cloud integration paused during short-term fusion')
        if self._last_received is None or now - self._last_received > self.max_age_s:
            self._lose()
            raise ValueError('odometry stale or unavailable')
        if self._last_cloud_stamp_ns is not None and stamp_ns <= self._last_cloud_stamp_ns:
            raise ValueError('cloud timestamp did not increase')
        self._check_measurement_time(stamp_ns, measurement_now_ns)
        if self.extrinsic_mode == 'dynamic' and capture_context is None:
            raise ValueError('dynamic extrinsic capture context is required')
        if capture_context is not None:
            _rigid(capture_context.get('child_from_cloud'))
        self._last_cloud_stamp_ns = stamp_ns
        self._last_cloud_received = now if received is None else received
        if len(self._pending) == self.max_pending:
            self._pending.popleft()
            self._dropped_pending += 1
        self._pending.append((stamp_ns, cloud, self._last_cloud_received, deepcopy(capture_context)))
        self._drain(now, measurement_now_ns)

    def _drain(self, now, measurement_now_ns):
        remaining = deque()
        while self._pending:
            stamp, cloud, received, context = self._pending.popleft()
            if now - received > self.max_age_s:
                continue
            try:
                self._check_measurement_time(stamp, measurement_now_ns)
                if not self._poses:
                    continue
                if stamp > self._poses[-1].stamp_ms * 1_000_000:
                    remaining.append((stamp, cloud, received, context))
                    continue
                pose = align_pose(self._poses, stamp, max_gap_ms=self.max_gap_ms)
                frames = (pose.frame_id, pose.child_frame_id, pose.clock_id)
                if pose.segment_id in self._segments:
                    if pose.segment_id != next(reversed(self._segments)):
                        raise ValueError('retired segment cannot be reused')
                    if self._requires_new_segment:
                        raise ValueError('recovery requires a new segment')
                    if self._segments[pose.segment_id] != frames:
                        self._requires_new_segment = True
                        raise ValueError('pose frames or clock changed within segment')
                matrix = _rigid(context['child_from_cloud']) if context is not None else self.extrinsic
                self.on_frame(stamp, cloud, pose, matrix.copy(), deepcopy(context))
                self._segments[pose.segment_id] = frames
                self._requires_new_segment = False
                self._admitted += 1
            except ValueError as error:
                self.last_error = str(error)
        self._pending = remaining
