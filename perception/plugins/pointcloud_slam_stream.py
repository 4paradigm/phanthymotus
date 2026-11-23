"""Bounded numeric cloud/pose association; producers own physical alignment."""
from collections import deque, OrderedDict
from copy import deepcopy
from dataclasses import replace
import math
import uuid

import numpy as np

from .camera_geometry import camera_transform
from .pointcloud_slam_alignment import align_pose
from .pointcloud_slam_contract import declares_measured_body
from .pointcloud_slam_odom import decode_slam_odom, decode_slam_prediction
from .pointcloud_slam_transforms import rigid as _rigid, within_measured_motion_bounds
from .pointcloud_slam_capture_mount import MappingMounts


class SlamStream:
    def __init__(self, *, cloud_frame, child_frame, clock_id, child_from_cloud,
                 max_gap_ms, max_age_s, max_pending, max_cloud_points,
                 on_frame, max_measurement_age_ms=None,
                 max_future_skew_ms=None, input_contract=None, extrinsic_mode='static',
                 on_unlocalized_frame=None, on_observation=None, on_pose_loss=None,
                 scan_origin_recovery=False, pose_capture_mount=False, color_from_depth=None):
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
        self.on_pose_loss = on_pose_loss
        self._scan_origin_recovery = scan_origin_recovery
        self._segments = OrderedDict()
        self._admitted = 0
        self._poses = deque(maxlen=512)
        self._pose_anchor = None
        self._measured_run = None
        self._suspended_run = None
        self._last_admitted_run = None
        self._pending = deque()
        self._last_received = None
        self._last_cloud_received = None
        self._last_cloud_stamp_ns = None
        self.input_contract = input_contract
        self._cloud_metadata = OrderedDict()
        self._pose_capture_mount = pose_capture_mount
        self._color_from_depth = color_from_depth
        self._pose_mounts = OrderedDict()
        self._external_body_frame = None
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

    @property
    def _live_visual_run(self):
        # The last origin field distinguishes external motion from visual-only.
        return (self._measured_run is not None and not self._measured_run[0][-1] and
                self.continuity_token == self._last_admitted_run)

    @property
    def _measured_body_run(self):
        return self._measured_run is not None and self._measured_run[0][-1]

    def _lose(self, now):
        """Retire unavailable poses and offer the latest fresh scan for registration."""
        latest = max(self._pending, key=lambda item: item[0], default=None)
        self._poses.clear()
        self._pose_anchor = None
        if self._measured_run is not None:
            # The run may resume only through the visual runtime's measured
            # relocalization against this exact last admitted pose.
            self._suspended_run = self._measured_run
        self._measured_run = None
        self._pending.clear()
        self._last_received = None
        self._odom_state = 'waiting'
        if self.on_pose_loss is not None:
            self.on_pose_loss()
        if latest is not None and self.on_unlocalized_frame is not None:
            self._offer_unlocalized(latest, now)

    def _offer_unlocalized(self, pending, now):
        live_body_run = (self._measured_body_run and
                            self.continuity_token == self._last_admitted_run)
        if self._live_visual_run or live_body_run:
            # A live, witnessed pose run can still lack a bracket for one scan.
            # Skip that scan without sending a false pose outage to RTAB-Map.
            # A real outage clears the run before offering scans for recovery.
            self._dropped_pending += 1
            return
        stamp, cloud, received, context = pending
        if now-received <= self.max_age_s:
            try:
                matrix = _rigid(context['child_from_cloud']) if context is not None else self.extrinsic
                self.on_unlocalized_frame(stamp, cloud, None, matrix.copy(), deepcopy(context))
            except ValueError as error:
                self.last_error = str(error)

    def poll(self, *, now):
        if self.on_unlocalized_frame is not None and self._pending:
            self._drain(now, None)

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
                self.last_error = str(error)
                self._lose(now)
                return
        if pose.prediction_anchor_ns is None:
            previous = self._measured_run
            pose = self._track_measured_run(sample, pose, context)
            if self._measured_body_run:
                # A queued measurement can still bracket a fresh scan. Detect
                # source silence from delivered progress, not time spent queued.
                progressed = (previous is None or previous[0] != self._measured_run[0] or
                              pose.stamp_ns > previous[1].stamp_ns)
                received = now if progressed else self._last_received
            if self._scan_origin_recovery and pose.declared_origin is None:
                # Scan-only built-in VO declares this same identity on the native wire.
                pose = replace(pose, visual_origin=(pose.segment_id, pose.frame_id))
            if self._pose_anchor is None or pose.stamp_ns >= self._pose_anchor.stamp_ns:
                self._pose_anchor = pose
                self._external_body_frame = pose.child_frame_id if pose.external_origin is not None else None
            # Confirmed corrections supersede hypotheses beyond their capture.
            poses = {p.stamp_ns: p for p in self._poses
                     if p.prediction_anchor_ns is None or p.stamp_ns < pose.stamp_ns}
        else:
            self._measured_run = None
            poses = {p.stamp_ns: p for p in self._poses}
        poses[pose.stamp_ns] = pose
        self._poses = deque(sorted(poses.values(), key=lambda p: p.stamp_ns)[-512:], maxlen=512)
        # Derived VO retains the camera ingress time. A fresh registration is
        # evidence of a live pose producer when its result actually arrives.
        fresh_visual = self._live_visual_run and received is not None and 0 <= now-received <= self.max_age_s
        self._last_received = now if received is None or fresh_visual else received
        self._odom_state = 'coasting' if pose.prediction_anchor_ns is not None else 'tracking'
        if pose.external_origin is not None and self._pose_capture_mount:
            self._remember_pose_mount(sample['vendor'], pose)
            self._drain_unmatched_clouds(now)
        self._drain(now, measurement_now_ns)

    def _remember_pose_mount(self, vendor, pose):
        mount = vendor.get('camera_mount')
        packet = mount.get('packet') if isinstance(mount, dict) else None
        if not isinstance(packet, dict):
            return
        stamp = packet.get('stamp_ns')
        parent = packet.get('parent_frame')
        child = packet.get('child_frame', self.cloud_frame)
        if (type(stamp) is not int or not isinstance(child, str) or
                (parent is not None and parent != pose.child_frame_id)):
            return
        history = self._pose_mounts.get(child)
        if history is None or history.declaration['parent_frame'] != pose.child_frame_id:
            history = MappingMounts(dict(topic='external_fusion', parent_frame=pose.child_frame_id,
                                         child_frame=child), child, None)
            self._pose_mounts[child] = history
        try:
            history.observe(packet, stamp)
        except ValueError as error:
            self.last_error = str(error)
        self._pose_mounts.move_to_end(child)
        while len(self._pose_mounts) > 4:
            self._pose_mounts.popitem(last=False)

    def _track_measured_run(self, sample, pose, context):
        """Track measured continuity in the active numeric odometry gauge."""
        vendor = sample.get('vendor')
        quality = vendor.get('motus_slam') if isinstance(vendor, dict) else None
        if pose.external_origin is not None:
            # The fusion card owns its body's local origin across visual resets.
            # Only actual motion samples carry its main pose.
            return self._remember_measured_run(pose, ('external_fusion', *pose.external_origin, True))
        chassis = declares_measured_body(quality)
        visual_source = isinstance(vendor, dict) and vendor.get('source') == 'rtabmap_rgbd_odometry'
        visual = visual_source and self._witnessed_visual(vendor, pose, context)
        mount = quality.get('extrinsic') if isinstance(quality, dict) else None
        if (visual_source and not visual and isinstance(mount, dict) and
                pose.segment_id and pose.frame_id and pose.child_frame_id and
                mount.get('parent_frame') == pose.child_frame_id):
            # A driver can declare its own coordinate origin without the
            # local RGB-D registry's evidence for continuity.
            pose = replace(pose, visual_origin=(pose.segment_id, pose.frame_id, pose.child_frame_id))
        origin = (pose.segment_id, pose.frame_id, pose.child_frame_id, pose.clock_id,
                  quality.get('source_topic') if chassis else vendor.get('source') if visual else None)
        if (not isinstance(quality, dict) or not (chassis or visual) or
                not all(isinstance(value, str) and value for value in origin)):
            self._measured_run = None
            return pose
        observation = pose.observation
        measurement = quality.get('measurement')
        measurement = measurement if isinstance(measurement, dict) else {}
        mapping = measurement.get('mapping')
        reference = quality.get('clock_reference')
        fusion = vendor.get('fusion')
        epoch = measurement.get('clock_epoch')
        source_stamp = measurement.get('source_stamp')
        source_clock = measurement.get('source_clock')
        unit = measurement.get('source_unit')
        ns_per_unit = {'ns': 1, 'us': 1000, 'ms': 1_000_000, 's': 1_000_000_000}
        # A clock mapping may change independently of the spatial generation.
        if (chassis and measurement.get('source_generation') == pose.segment_id and
                isinstance(source_clock, str) and source_clock and isinstance(unit, str) and unit in ns_per_unit and
                isinstance(epoch, str) and epoch and type(source_stamp) is int and source_stamp > 0 and
                isinstance(mapping, dict) and isinstance(mapping.get('method'), str) and mapping['method'] and
                mapping.get('verified') is True):
            pose = replace(pose, chassis_clock_epoch=epoch, device_stamp_ns=source_stamp * ns_per_unit[unit])
            # Keep the original native gauge, including for saved maps.
            epoch = pose.segment_id
        origin += (epoch, measurement.get('source_generation'),
                   measurement.get('source_clock'), reference.get('domain_id') if isinstance(reference, dict) else None,
                   observation.source_id if observation is not None else None,
                   observation.source_epoch if observation is not None else None, chassis)
        if (quality.get('status') != 'tracking' or quality.get('stamp_source') != 'measurement' or
                quality.get('stamp_ns') != pose.stamp_ns or pose.stamp_ns <= 0 or
                (observation is not None and
                 (observation.time_basis != 'source_capture' or
                  observation.normalized_clock_id not in ('', pose.clock_id) or
                  (observation.mapping_offset_ns is not None and not observation.mapping_verified))) or
                measurement.get('stamp_ns', pose.stamp_ns) != pose.stamp_ns or
                measurement.get('clock_id', pose.clock_id) != pose.clock_id or
                measurement.get('frame_id', pose.frame_id) != pose.frame_id or
                (isinstance(mapping, dict) and mapping.get('verified') is False) or
                (isinstance(fusion, dict) and fusion.get('mapping_allowed') is False)):
            self._measured_run = None
            # Native still needs the gauge identity to check this numeric pose.
            # Without a run token it verifies geometry before extending the map.
            return replace(pose, chassis_origin=origin) if chassis else pose
        return self._remember_measured_run(pose, origin, chassis=chassis, visual=visual, vendor=vendor)

    def _remember_measured_run(self, pose, origin, *, chassis=False, visual=False, vendor=None):
        token = None
        if self._measured_run is not None:
            last_origin, previous = self._measured_run
            dt = (pose.stamp_ns - previous.stamp_ns) / 1e9
            clock_switched = (pose.chassis_clock_epoch is not None and previous.chassis_clock_epoch is not None and
                              pose.chassis_clock_epoch != previous.chassis_clock_epoch)
            if clock_switched:
                # Discard old time brackets and queued work, keeping the spatial
                # witness only if source time and measured motion both agree.
                dt = (pose.device_stamp_ns - previous.device_stamp_ns) / 1e9
                self._poses.clear()
                self._pose_anchor = None
                self._dropped_pending += len(self._pending) + len(self._unmatched_clouds)
                self._pending.clear()
                self._unmatched_clouds.clear()
                if self.on_pose_loss is not None:
                    self.on_pose_loss()
            distance = math.dist(pose.position, previous.position)
            dot = abs(sum(a*b for a, b in zip(pose.quaternion, previous.quaternion)))
            angle = 2*math.acos(min(1., dot))
            # Measured VO can skip captures while its same-stamp registration
            # is still valid. Match the visual runtime's 1.5 s output bound.
            max_interval = 1.5 if visual or pose.external_origin is not None else .5
            if last_origin == origin and (
                    (not clock_switched and dt == 0 and distance <= 1e-6 and angle <= 1e-5) or
                    (0 < dt <= max_interval and within_measured_motion_bounds(distance, angle, dt))):
                token = previous.external_continuity
        elif self._suspended_run is not None and visual:
            token = self._resume_suspended_run(origin, pose, vendor)
        self._suspended_run = None
        # This token is carried only through an observed, bounded source run.
        # A new token requires independent registration in the native map.
        pose = replace(pose, external_continuity=token or uuid.uuid4().bytes,
                       chassis_origin=origin if chassis else None,
                       visual_origin=(pose.segment_id, pose.frame_id) if visual else None)
        self._measured_run = origin, pose
        return pose

    # The visual runtime keeps a paused segment open while native F2M still
    # re-registers to its retained local map, bounded by this same gap.
    RELOCALIZATION_MAX_S = 4.5

    def _resume_suspended_run(self, origin, pose, vendor):
        """Carry the old token only when VO proved the same F2M gauge after a pause."""
        last_origin, previous = self._suspended_run
        proof = vendor.get('relocalization')
        if last_origin != origin or not isinstance(proof, dict):
            return None
        anchor, shared = proof.get('anchor_stamp_ns'), proof.get('shared_landmarks')
        dt = (pose.stamp_ns - previous.stamp_ns) / 1e9
        distance = math.dist(pose.position, previous.position)
        dot = abs(sum(a*b for a, b in zip(pose.quaternion, previous.quaternion)))
        angle = 2*math.acos(min(1., dot))
        if (proof.get('method') != 'native_f2m_landmarks' or type(shared) is not int or shared < 20 or
                type(anchor) is not int or not previous.stamp_ns <= anchor <= previous.stamp_ns + 1_500_000_000 or
                not 0 < dt <= self.RELOCALIZATION_MAX_S or
                distance > .15 + 1.5*dt or angle > math.radians(10) + 2.5*dt):
            return None
        return previous.external_continuity

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

    def _mount_context(self, stamp, frame, explicit=None):
        if frame == self._external_body_frame:
            return dict(child_from_cloud=np.eye(4).tolist(), pose_capture_mount=self._pose_capture_mount)
        if explicit is not None:
            return explicit
        if self._pose_capture_mount and self._external_body_frame is not None:
            context = self._cloud_metadata.get(stamp)
            if context is not None and (context.get('extrinsic', {}).get('matrix') is not None or
                    context['measurement'].get('child_from_cloud') is not None):
                return dict(context, pose_capture_mount=True)
            history = self._pose_mounts.get(frame)
            color_from_depth = None
            if (history is None and self._color_from_depth is not None and
                    frame == self.cloud_frame and len(self._pose_mounts) == 1):
                history = next(iter(self._pose_mounts.values()))
                color_from_depth = self._color_from_depth
            if history is not None:
                matrix = history.matrix_at(stamp)
                evidence = None
                if matrix is None:
                    evidence = history.delayed_measurement()
                    if evidence:
                        matrix = _rigid(evidence['matrix'])
                if matrix is not None:
                    if color_from_depth is not None:
                        matrix = matrix @ camera_transform(color_from_depth)
                    result = dict(child_from_cloud=matrix.tolist(), pose_capture_mount=True)
                    if evidence:
                        result['extrinsic'] = deepcopy(evidence)
                    return result
            return None
        return self._nearest_context(stamp)

    def _drain_unmatched_clouds(self, now):
        for stamp, (frame, cloud, received) in list(self._unmatched_clouds.items()):
            if self._pose_capture_mount and now-received > self.max_age_s:
                self._unmatched_clouds.pop(stamp)
                self._dropped_pending += 1
                continue
            context = self._mount_context(stamp, frame)
            if context is not None:
                self._unmatched_clouds.pop(stamp)
                self._accept_cloud(stamp, frame, cloud, now=now, received=received, capture_context=context)

    def cloud_metadata(self, packet, *, now, measurement_now_ns=None):
        if not isinstance(packet, dict) or type(packet.get('stamp_ns')) is not int:
            self.last_error = 'invalid cloud metadata timestamp'
            return
        try:
            stamp = packet['stamp_ns']
            self._cloud_metadata[stamp] = self._capture_context(packet)
            while len(self._cloud_metadata) > self.max_pending:
                self._cloud_metadata.popitem(last=False)
            self._drain_unmatched_clouds(now)
        except ValueError as error:
            self.last_error = str(error)

    def cloud(self, stamp_ns, frame, cloud, *, now, point_count=None, measurement_now_ns=None,
              data_sha256=None, capture_context=None, received=None):
        """Retain opaque native data; the adapter checks its numeric XYZ layout."""
        self._check_measurement_time(stamp_ns, measurement_now_ns)
        count = len(cloud) if point_count is None else point_count
        if type(count) is not int or not 0 < count <= self.max_cloud_points:
            raise ValueError('invalid cloud size or capacity')
        capture_context = self._mount_context(stamp_ns, frame, capture_context)
        if self._manual_extrinsic and capture_context is not None and frame != self._external_body_frame:
            capture_context=deepcopy(capture_context)
            capture_context['child_from_cloud']=self.extrinsic.tolist()
        if capture_context is None and (self.extrinsic_mode == 'dynamic' or
                (self._pose_capture_mount and self._external_body_frame is not None)):
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
        self._pending.append((stamp_ns, cloud, self._last_cloud_received, deepcopy(capture_context)))
        self._drain(now, measurement_now_ns)

    def _drain(self, now, measurement_now_ns):
        can_localize = self.on_unlocalized_frame is not None
        # Give late visual results a bounded opportunity to pair with the scan.
        wait_s = min(1.5, self.max_age_s)
        # Projection and input delivery consume the same original ingress budget.
        # Measured body odometry can still bracket the scan within that budget.
        pose_wait_s = self.max_age_s if self._measured_body_run else wait_s
        scan_wait_s = self.max_age_s if self._live_visual_run else pose_wait_s
        if can_localize and self._last_received is not None and now-self._last_received >= pose_wait_s:
            self._lose(now)
            return
        if len(self._pending) > self.max_pending:
            oldest = self._pending.popleft()
            if can_localize:
                self._offer_unlocalized(oldest, now)
            else:
                self._dropped_pending += 1
        remaining = deque()
        while self._pending:
            pending = self._pending.popleft()
            stamp, cloud, received, context = pending
            if can_localize and (self._live_visual_run or self._measured_body_run) and now-received > self.max_age_s:
                self._dropped_pending += 1
                continue
            try:
                if not self._poses or stamp > self._poses[-1].stamp_ns or stamp < self._poses[0].stamp_ns:
                    if can_localize and (not self._poses or now-received >= scan_wait_s):
                        self._offer_unlocalized(pending, now)
                    else:
                        remaining.append(pending)
                    continue
                matrix = _rigid(context['child_from_cloud']) if context is not None else self.extrinsic
                try:
                    pose = align_pose(self._poses, stamp, max_gap_ms=self.max_gap_ms)
                except ValueError as error:
                    self.last_error = str(error)
                    if can_localize:
                        # Geometry can still verify this scan independently.
                        self._offer_unlocalized(pending, now)
                    continue
                self.on_frame(stamp, cloud, pose, matrix.copy(), deepcopy(context))
                if pose.prediction_anchor_ns is None:
                    self._last_admitted_run = pose.external_continuity
                    self._segments[pose.segment_id] = (pose.frame_id, pose.child_frame_id, pose.clock_id, pose.pose_reference)
                    self._admitted += 1
            except ValueError as error:
                self.last_error = str(error)
        self._pending = remaining
