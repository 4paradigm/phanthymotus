"""Bounded numeric cloud/pose association; producers own physical alignment."""
from collections import deque, OrderedDict
from copy import deepcopy
from dataclasses import replace
import math
import uuid

import numpy as np

from .pointcloud_slam_alignment import align_pose
from .pointcloud_slam_odom import decode_slam_odom, decode_slam_prediction
from .pointcloud_slam_transforms import rigid as _rigid


class SlamStream:
    def __init__(self, *, cloud_frame, child_frame, clock_id, child_from_cloud,
                 max_gap_ms, max_age_s, max_pending, max_cloud_points,
                 on_frame, max_measurement_age_ms=None,
                 max_future_skew_ms=None, input_contract=None, extrinsic_mode='static',
                 on_unlocalized_frame=None, on_observation=None):
        if any(type(v) is not int or v <= 0 for v in (max_pending, max_cloud_points)):
            raise ValueError('positive bounded capacities required')
        self.max_measurement_age_ms = max_measurement_age_ms
        self.max_future_skew_ms = max_future_skew_ms
        self.cloud_frame, self.child_frame, self.clock_id = cloud_frame, child_frame, clock_id
        self.extrinsic_mode = (input_contract['settings'].get('extrinsic_mode', extrinsic_mode)
                               if input_contract else extrinsic_mode)
        self.extrinsic = _rigid(child_from_cloud) if child_from_cloud is not None else np.eye(4)
        self._manual_extrinsic=bool(input_contract and input_contract.get('settings',{}).get('manual_extrinsic'))
        self.max_gap_ms, self.max_age_s = max_gap_ms, max_age_s
        self.max_pending, self.max_cloud_points = max_pending, max_cloud_points
        self.on_frame = on_frame
        self.on_unlocalized_frame = on_unlocalized_frame
        self.on_observation = on_observation
        self._segments = OrderedDict()
        self._admitted = 0
        self._poses = deque(maxlen=512)
        self._pose_anchor = None
        self._measured_run = None
        self._pending = deque()
        self._last_received = None
        self._last_cloud_received = None
        self._last_cloud_stamp_ns = None
        self.input_contract = input_contract
        self._cloud_metadata = OrderedDict()
        self._unmatched_clouds = OrderedDict()
        self._metadata_fallbacks = 0
        self.last_error = ''
        self._odom_state = 'waiting'
        self.ingress_clock_id = None
        self._dropped_pending = 0

    @property
    def pending_count(self):
        return len(self._pending)

    @property
    def continuity_token(self):
        return self._measured_run[1].external_continuity if self._measured_run is not None else None

    def _lose(self):
        """Numeric pose absence waits for a new pose, without retiring labels."""
        self._poses.clear()
        self._pose_anchor = None
        self._measured_run = None
        self._pending.clear()
        self._last_received = None
        self._odom_state = 'waiting'

    def status(self, *, now, measurement_now_ns=None):
        return {'odometry': self._odom_state,
                'cloud': 'active' if self._last_cloud_received is not None else 'waiting',
                'pending_clouds': len(self._pending),
                'integration': 'admitting' if self._poses else 'waiting_for_tracking',
                'dropped_pending': self._dropped_pending, 'segments': len(self._segments),
                'admitted_frames': self._admitted, 'unmatched_clouds': len(self._unmatched_clouds),
                'frames_without_sidecar': self._metadata_fallbacks, 'last_error': self.last_error}

    def _check_measurement_time(self, stamp_ns, measurement_now_ns):
        if type(stamp_ns) is not int:
            raise ValueError('measurement timestamp must be integer nanoseconds')

    def odom(self, sample, *, now, measurement_now_ns=None, received=None, context=None):
        try:
            pose = decode_slam_odom(sample)
        except ValueError as error:
            try:
                pose = decode_slam_prediction(sample, self._pose_anchor)
            except ValueError:
                self._lose()
                self.last_error = str(error)
                return
        if pose.prediction_anchor_ns is None:
            pose = self._track_measured_run(sample, pose, context)
            if self._pose_anchor is None or pose.stamp_ns >= self._pose_anchor.stamp_ns:
                self._pose_anchor = pose
            # Confirmed corrections supersede hypotheses beyond their capture.
            poses = {p.stamp_ns: p for p in self._poses
                     if p.prediction_anchor_ns is None or p.stamp_ns < pose.stamp_ns}
        else:
            self._measured_run = None
            poses = {p.stamp_ns: p for p in self._poses}
        poses[pose.stamp_ns] = pose
        self._poses = deque(sorted(poses.values(), key=lambda p: p.stamp_ns)[-512:], maxlen=512)
        self._last_received = now if received is None else received
        self._odom_state = 'coasting' if pose.prediction_anchor_ns is not None else 'tracking'
        self._drain(now, measurement_now_ns)

    def _track_measured_run(self, sample, pose, context):
        vendor = sample.get('vendor')
        quality = vendor.get('motus_slam') if isinstance(vendor, dict) else None
        chassis = (isinstance(quality, dict) and quality.get('pose_source') == 'chassis_odometry' and
                   quality.get('pose_reference') == 'body')
        visual = (isinstance(vendor, dict) and vendor.get('source') == 'rtabmap_rgbd_odometry' and
                  self._witnessed_visual(vendor, pose, context))
        origin = (pose.segment_id, pose.frame_id, pose.child_frame_id, pose.clock_id,
                  quality.get('source_topic') if chassis else vendor.get('source') if visual else None)
        if (not (chassis or visual) or not isinstance(quality, dict) or quality.get('status') != 'tracking' or
                quality.get('stamp_source') != 'measurement' or quality.get('stamp_ns') != pose.stamp_ns or
                pose.stamp_ns <= 0 or not all(isinstance(value, str) and value for value in origin)):
            self._measured_run = None
            return pose
        observation = pose.observation
        measurement = quality.get('measurement')
        measurement = measurement if isinstance(measurement, dict) else {}
        mapping = measurement.get('mapping')
        reference = quality.get('clock_reference')
        fusion = vendor.get('fusion')
        if ((observation is not None and
             (observation.time_basis != 'source_capture' or
              observation.normalized_clock_id not in ('', pose.clock_id) or
              (observation.mapping_offset_ns is not None and not observation.mapping_verified))) or
                measurement.get('stamp_ns', pose.stamp_ns) != pose.stamp_ns or
                measurement.get('clock_id', pose.clock_id) != pose.clock_id or
                measurement.get('frame_id', pose.frame_id) != pose.frame_id or
                (isinstance(mapping, dict) and mapping.get('verified') is False) or
                (isinstance(fusion, dict) and fusion.get('mapping_allowed') is False)):
            self._measured_run = None
            return pose
        origin += (measurement.get('clock_epoch'), measurement.get('source_generation'),
                   measurement.get('source_clock'), reference.get('domain_id') if isinstance(reference, dict) else None,
                   observation.source_id if observation is not None else None,
                   observation.source_epoch if observation is not None else None, chassis)
        token = None
        if self._measured_run is not None:
            last_origin, previous = self._measured_run
            dt = (pose.stamp_ns - previous.stamp_ns) / 1e9
            distance = math.dist(pose.position, previous.position)
            dot = abs(sum(a*b for a, b in zip(pose.quaternion, previous.quaternion)))
            angle = 2*math.acos(min(1., dot))
            # Measured VO can skip captures while its same-stamp registration
            # is still valid. Keep that observed walk bounded to one second.
            max_interval = .5 if chassis else 1.
            if last_origin == origin and (
                    (dt == 0 and distance <= 1e-6 and angle <= 1e-5) or
                    (0 < dt <= max_interval and distance <= .1 + 1.5*dt and
                     angle <= math.radians(10) + 2.5*dt)):
                token = previous.external_continuity
        # This token is carried only through an observed, bounded source run.
        # A new token requires independent registration in the native map.
        pose = replace(pose, external_continuity=token or uuid.uuid4().bytes)
        self._measured_run = origin, pose
        return pose

    def _witnessed_visual(self, vendor, pose, context):
        observation, report = pose.observation, vendor.get('registration')
        # The local registry, not a source's topic name or segment label,
        # attests that this output was produced by our RGB-D runtime.
        return (context is not None and observation is not None and self.ingress_clock_id is not None and
                context.ingress_clock_id == observation.ingress_clock_id == self.ingress_clock_id and
                context.time_basis == observation.time_basis == 'source_capture' and
                context.normalized_clock_id == observation.normalized_clock_id == pose.clock_id and
                context.observation_stamp_ns == pose.stamp_ns and
                context.parent_observation_ids == (observation.observation_id,) and
                context.transformations == observation.transformations+('rgbd_odometry',) and
                isinstance(report, dict) and report.get('status') == 'tracking' and
                report.get('native_lost') is False and type(report.get('native_inliers')) is int and
                report['native_inliers'] > 0 and pose.covariance is not None and
                all(pose.covariance[i] > 0. for i in (0, 7, 14, 21, 28, 35)))

    def _capture_context(self, packet):
        evidence = packet.get('extrinsic')
        matrix = self.extrinsic if self._manual_extrinsic else packet.get('child_from_cloud')
        if not self._manual_extrinsic and isinstance(evidence, dict) and evidence.get('matrix') is not None:
            matrix = evidence['matrix']
        if matrix is None:
            if self.extrinsic_mode == 'dynamic':
                raise ValueError('numeric dynamic extrinsic matrix is required')
            matrix = self.extrinsic
        context = dict(child_from_cloud=_rigid(matrix).tolist(), measurement=deepcopy(packet))
        if isinstance(evidence, dict):
            context['extrinsic'] = deepcopy(evidence)
        if isinstance(packet.get('observation'), dict):
            context['observation'] = deepcopy(packet['observation'])
        return context

    def _nearest_context(self,stamp):
        if not self._cloud_metadata:return None
        nearest=min(self._cloud_metadata,key=lambda t:abs(t-stamp))
        return deepcopy(self._cloud_metadata[nearest])

    def cloud_metadata(self, packet, *, now, measurement_now_ns=None):
        if not isinstance(packet, dict) or type(packet.get('stamp_ns')) is not int:
            self.last_error = 'invalid cloud metadata timestamp'
            return
        try:
            stamp = packet['stamp_ns']
            self._cloud_metadata[stamp] = self._capture_context(packet)
            while len(self._cloud_metadata) > self.max_pending:
                self._cloud_metadata.popitem(last=False)
            for cloud_stamp in list(self._unmatched_clouds):
                frame,cloud,received=self._unmatched_clouds.pop(cloud_stamp)
                self._accept_cloud(cloud_stamp,frame,cloud,now=now,received=received,
                                   capture_context=self._nearest_context(cloud_stamp))
        except ValueError as error:
            self.last_error = str(error)

    def cloud(self, stamp_ns, frame, cloud, *, now, point_count=None, measurement_now_ns=None,
              data_sha256=None, capture_context=None, received=None):
        """Retain opaque native data; the adapter checks its numeric XYZ layout."""
        self._check_measurement_time(stamp_ns, measurement_now_ns)
        count = len(cloud) if point_count is None else point_count
        if type(count) is not int or not 0 < count <= self.max_cloud_points:
            raise ValueError('invalid cloud size or capacity')
        if capture_context is None:
            capture_context = self._nearest_context(stamp_ns)
        if self._manual_extrinsic and capture_context is not None:
            capture_context=deepcopy(capture_context)
            capture_context['child_from_cloud']=self.extrinsic.tolist()
        if self.extrinsic_mode == 'dynamic' and capture_context is None:
            self._unmatched_clouds[stamp_ns] = (frame, deepcopy(cloud), now if received is None else received)
            while len(self._unmatched_clouds) > self.max_pending:
                self._unmatched_clouds.popitem(last=False)
                self._dropped_pending += 1
            return
        self._accept_cloud(stamp_ns, frame, deepcopy(cloud), now=now, received=received,
                           capture_context=capture_context)

    def _accept_cloud(self, stamp_ns, frame, cloud, *, now, measurement_now_ns=None, received=None,
                      capture_context=None):
        if capture_context is not None:
            _rigid(capture_context.get('child_from_cloud'))
        self._last_cloud_stamp_ns = stamp_ns
        self._last_cloud_received = now if received is None else received
        if self.on_observation is not None:
            matrix = _rigid(capture_context['child_from_cloud']) if capture_context is not None else self.extrinsic
            self.on_observation(stamp_ns, cloud, matrix)
        if len(self._pending) == self.max_pending:
            oldest = self._pending.popleft()
            if self.on_unlocalized_frame is not None:
                old_stamp, old_cloud, _, old_context = oldest
                matrix = _rigid(old_context['child_from_cloud']) if old_context is not None else self.extrinsic
                self.on_unlocalized_frame(old_stamp, old_cloud, None, matrix.copy(), deepcopy(old_context))
            else:
                self._dropped_pending += 1
        self._pending.append((stamp_ns, cloud, self._last_cloud_received, deepcopy(capture_context)))
        self._drain(now, measurement_now_ns)

    def _drain(self, now, measurement_now_ns):
        remaining = deque()
        while self._pending:
            stamp, cloud, received, context = self._pending.popleft()
            try:
                if not self._poses and self.on_unlocalized_frame is not None:
                    matrix = _rigid(context['child_from_cloud']) if context is not None else self.extrinsic
                    self.on_unlocalized_frame(stamp, cloud, None, matrix.copy(), deepcopy(context))
                    continue
                if not self._poses or stamp > self._poses[-1].stamp_ns or stamp < self._poses[0].stamp_ns:
                    remaining.append((stamp, cloud, received, context))
                    continue
                pose = align_pose(self._poses, stamp, max_gap_ms=self.max_gap_ms)
                matrix = _rigid(context['child_from_cloud']) if context is not None else self.extrinsic
                self.on_frame(stamp, cloud, pose, matrix.copy(), deepcopy(context))
                if pose.prediction_anchor_ns is None:
                    self._segments[pose.segment_id] = (pose.frame_id, pose.child_frame_id, pose.clock_id, pose.pose_reference)
                    self._admitted += 1
            except ValueError as error:
                self.last_error = str(error)
        self._pending = remaining
