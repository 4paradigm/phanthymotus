"""One supervised RTAB-Map estimator per calibrated RGB-D card instance."""
from collections import Counter, OrderedDict, deque
from array import array
from copy import deepcopy
from dataclasses import replace
import json
import logging
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import zlib

import numpy as np

from utils.dds_transport import receive_environment

from .rgbd_odometry_contract import PairBuffer, make_sample
from .rgbd_pairing import RgbdPairer, RgbdPair
from .spatial_time import TimePolicy
from .rgbd_imu import ImuAttitude
from .rgbd_association import RotationHints, association_enabled, require_rotation_hint_runtime
from .rgbd_recovery import (AuxiliaryHealthProbe, ASSISTANCE_RETRY_S,
                            VISUAL_INITIALIZATION_S, assistance_mode, visual_calibration)
from .rgbd_fusion import MAX_FRAME_AGE_NS, ShortTermFusion, log_rotation, observation_covariance, pose_values
from .rgbd_body_mount import CaptureMounts
from .rgbd_geometry import GeometryWorker, observation_quality
from .spatial_quality import MotionGate, QualityReport
from .spatial_recovery import RecoveryPolicy
from .spatial_input import latency_summary
from utils.ros_lifecycle import dispose_node


def require_runtime():
    from ament_index_python.packages import get_package_prefix
    path = Path(get_package_prefix('rtabmap_odom')) / 'lib/rtabmap_odom/rgbd_odometry'
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError('RTAB-Map RGB-D odometry runtime unavailable')
    return path


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def native_capture_stamp(color, depth):
    # Pinned RGBDOdometry::commonCallback compares rclcpp::Time::seconds(),
    # chooses the later image (depth on a double-precision tie), then retains
    # that original header. Never round the chosen stamp or fuzzy-join results.
    rgb, z = stamp_ns(color), stamp_ns(depth)
    return rgb if float(rgb)/1e9 > float(z)/1e9 else z


# pointcloud_slam pauses a segment for at most 1 s (wall time and capture
# stamps after its last pose). Resume in-segment only within 0.8 s of the last
# published capture, leaving margin for native latency jitter; otherwise the
# segment ends and supervised recovery starts a new one.
RECOVERY_NS, PAUSE_RESUME_NS = 1_000_000_000, 800_000_000
# Fused output differs from the native visual pose by ESKF corrections. A
# visual-only fallback may stay in the segment only within this (unvalidated)
# engineering bound at the last published fused capture.
FALLBACK_OFFSET_M, FALLBACK_OFFSET_RAD = .02, np.deg2rad(2.)
# Engineering bound for a head/body camera, not a validated motion model: a
# resumed estimate farther than this from the last published pose is treated as
# a native reset/relocalization and must start a new segment.
RESUME_TRANSLATION_M, RESUME_SPEED_MPS = .05, 1.
RESUME_ROTATION_RAD, RESUME_RATE_RADPS = np.deg2rad(5.), np.deg2rad(90.)
F2M_SHARED_INLIERS = 20  # pinned RTAB-Map Vis/MinInliers


class EstimateGate:
    """Hard errors end a segment; transient visual rejections only pause output.

    `recovering` keeps feeding the same native estimator but publishes no pose.
    Three consecutive valid estimates continuous with the last published pose
    resume tracking; persistent degradation or a discontinuity ends the segment.
    """
    def __init__(self, segment, frame, clock):
        self.segment, self.frame, self.clock = segment, frame, clock
        self.state, self.reason = 'initializing', None
        self.last_input = self.last_stamp = self.last_result_stamp = 0
        self.good = 0
        self.recover_by = None
        self.last_pose = None
        self.loss_kind = None
        self.rejection = None
        self.recovery_policy = self.pause_context = None
        self.input_period_ns = self.processing_p95_ns = None
        self.input_intervals = deque(maxlen=32)
        self.last_native_stamp = 0
        self.result_period_ns = None
        self.result_intervals = deque(maxlen=32)
        self.anchor_info = None
        self.relocalization_points = None
        self.relocalization_deadline = self.relocalization_deadline_mono = None
        self.relocalization_failure = 'output'
        self.relocalization_misses = 0

    def input(self, stamp, *, now_ns=None):
        self.status(stamp if now_ns is None else now_ns)
        if self.state != 'lost':
            if self.relocalization_points is not None and self.relocalization_deadline is None:
                self.relocalization_deadline = (stamp if now_ns is None else now_ns) + RECOVERY_NS
                self.relocalization_deadline_mono = time.monotonic_ns() + RECOVERY_NS
            if self.last_input and stamp > self.last_input:
                self.input_intervals.append(stamp-self.last_input)
                self.input_period_ns = int(np.median(self.input_intervals)) if len(self.input_intervals)>=5 else None
            self.last_input = stamp
            self.last_input_received = stamp if now_ns is None else now_ns

    def fault(self, reason, kind='other'):
        if self.state != 'lost':
            self.loss_kind = kind
        self.state, self.reason = 'lost', self.reason if self.state == 'lost' else str(reason)

    def degrade(self, reason, now, *, stamp=None):
        """Reject one estimate. Tracking pauses; evidence must be rebuilt."""
        self.rejection = str(reason)
        if self.state == 'lost':
            return
        if stamp is not None:
            self.last_result_stamp = stamp
        self.good = 0
        if self.relocalization_points is not None:
            self.relocalization_failure = 'relocalization'
            self.relocalization_misses += 1
            if self.relocalization_misses >= 3:
                self.fault('three native registrations failed against the old F2M map; restart required',
                           'relocalization')
            return
        if self.state in ('initializing', 'tracking', 'recovering'):
            if self.recover_by is None:
                if self.recovery_policy is not None:
                    periods = [v for v in (self.input_period_ns, self.result_period_ns) if v is not None]
                    self.pause_context = self.recovery_policy.begin(max(periods) if periods else None,
                        self.processing_p95_ns, time.monotonic_ns())
                self.recover_by = now + (self.pause_context.budget_ns if self.pause_context else RECOVERY_NS)
            self.state, self.reason = 'recovering', 'visual estimate rejected: ' + str(reason)

    @staticmethod
    def _matched_points(info):
        # The native lite message carries only inliers in the current F2M world.
        if info is None or getattr(info, 'type', None) != 0:
            return {}
        ids = getattr(info, 'local_map_keys', ())
        values = getattr(info, 'local_map_values', ())
        if len(ids) != len(values):
            return {}
        inliers = set(getattr(info, 'word_inliers', ()))
        return {i: (p.x, p.y, p.z) for i, p in zip(ids, values)
                if i in inliers and math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)}

    def _timeout(self, reason, kind):
        if self.state == 'tracking' or (self.state == 'recovering' and self.pause_context is not None):
            points = self._matched_points(self.anchor_info)
            if len(points) >= F2M_SHARED_INLIERS:
                self.relocalization_points = points
                self.relocalization_deadline = self.relocalization_deadline_mono = None
                self.relocalization_misses = 0
                self.relocalization_failure = 'output'
                self.good = 0
                self.pause_context = self.recover_by = None
                self.state, self.reason = 'recovering', 'waiting for native matches to the last tracked F2M landmarks'
                return
        self.fault(reason, kind)

    def status(self, now):
        if self.relocalization_points is not None:
            if self.relocalization_deadline is not None and (
                    now > self.relocalization_deadline or time.monotonic_ns() > self.relocalization_deadline_mono):
                self.fault('native F2M relocalization did not verify the old landmarks; restart required',
                           self.relocalization_failure)
            return self.state
        received = getattr(self, "last_input_received", self.last_input)
        if received and now - received > 1_500_000_000:
            self._timeout('RGB-D input stale; restart to begin a new segment', 'input')
            return self.state
        # Before any published anchor, fresh weak results prove liveness only.
        # An anchored segment still expires against its accepted visual stamps.
        visual_stamp = self.last_stamp if self.last_pose is not None and self.pause_context is None else self.last_result_stamp
        if visual_stamp and now - visual_stamp > 1_500_000_000:
            self._timeout('visual estimate stale; restart to begin a new segment', 'output')
            return self.state
        if self.state == 'recovering' and self.recover_by is not None and (
                now > self.recover_by or
                (self.pause_context is not None and self.pause_context.expired(time.monotonic_ns()))):
            self._timeout(f'visual degradation persisted beyond {(self.pause_context.budget_ns if self.pause_context else RECOVERY_NS)/1e9:g} s; restart to begin a new segment', 'visual')
        return self.state

    def _continuous(self, stamp, position, rotation):
        if self.last_pose is None:
            return True
        last_stamp, last_position, last_rotation = self.last_pose
        dt = max(0., (stamp - last_stamp) / 1e9)
        return (np.linalg.norm(position - last_position) <= RESUME_TRANSLATION_M + RESUME_SPEED_MPS*dt and
                np.linalg.norm(log_rotation(last_rotation.T @ rotation)) <=
                RESUME_ROTATION_RAD + RESUME_RATE_RADPS*dt)

    def accept(self, msg, info, now):
        if self.status(now) == 'lost' or stamp_ns(msg) != stamp_ns(info):
            return None
        stamp = stamp_ns(msg)
        if msg.header.frame_id != self.segment + '_odom' or msg.child_frame_id != self.frame:
            self.fault('native visual frame or segment changed; restart required', 'frame')
            return None
        if now - stamp < -50_000_000:
            self.fault('visual estimate capture time is in the future', 'clock')
            return None
        if stamp <= max(self.last_stamp, self.last_result_stamp) or now - stamp > 1_500_000_000:
            self.rejection = 'stale or out-of-order visual estimate'  # never counted as evidence
            return None
        if self.last_native_stamp and stamp > self.last_native_stamp:
            self.result_intervals.append(stamp-self.last_native_stamp)
            if len(self.result_intervals) >= 5:
                self.result_period_ns = int(np.median(self.result_intervals))
        self.last_native_stamp = stamp
        self.last_result_stamp = stamp
        try:
            if info.lost or info.inliers < 10:
                raise ValueError('visual tracking lost or insufficient inliers')
            covariance = np.asarray(msg.pose.covariance, dtype=float).reshape(6, 6)
            if (not np.isfinite(covariance).all() or
                    not np.allclose(covariance, covariance.T, atol=1e-8) or
                    np.any(np.diag(covariance) < 0)):
                raise ValueError('visual covariance exceeds tracking bounds')
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            pose = dict(position=[p.x, p.y, p.z], quaternion=[q.x, q.y, q.z, q.w],
                        frame_id=msg.header.frame_id, child_frame_id=self.frame,
                        covariance=covariance.ravel().tolist())
            sample = make_sample('tracking', stamp, self.segment, self.clock, pose)
            position, rotation = pose_values(pose)
        except (ValueError, TypeError, AttributeError) as error:
            self.degrade(error, now)
            return None
        if np.any(np.diag(covariance)[:3] > .04):
            self.degrade('visual covariance exceeds tracking bounds', now)
            if self.last_pose is None and np.linalg.eigvalsh(covariance).min() >= -1e-10:
                # No map/pose has been published in this segment. Keep the
                # native map alive while quality settles, without allowing a
                # weak pose onto the output or extending an old SLAM pause.
                self.state, self.recover_by = 'initializing', None
            return None
        shared = None
        if self.relocalization_points is not None:
            points = self._matched_points(info)
            unchanged = {p for i, p in points.items() if self.relocalization_points.get(i) == p}
            shared = len(unchanged)
            self.relocalization_misses = 0
            _, anchor_position, anchor_rotation = self.last_pose
            if (self.relocalization_deadline is None or shared < F2M_SHARED_INLIERS or
                    np.linalg.matrix_rank(np.array(list(unchanged)) - next(iter(unchanged))) < 2 or
                    np.linalg.norm(position - anchor_position) > RESUME_TRANSLATION_M or
                    np.linalg.norm(log_rotation(anchor_rotation.T @ rotation)) > RESUME_ROTATION_RAD):
                self.good = 0
                self.relocalization_failure = 'relocalization'
                return None
        elif (self.state == 'recovering' and self.last_pose is not None and
                stamp - self.last_pose[0] > (self.pause_context.budget_ns-200_000_000 if self.pause_context else PAUSE_RESUME_NS)):
            self._timeout(f'visual recovery exceeded the {(self.pause_context.budget_ns-200_000_000 if self.pause_context else PAUSE_RESUME_NS)/1e9:g} s in-segment pause bound; '
                       'restart to begin a new segment', 'visual')
            return None
        if self.state != 'tracking' and not self._continuous(stamp, position, rotation):
            self.fault('visual estimate resumed discontinuously; restart to begin a new segment', 'visual')
            return None
        self.rejection = None
        self.last_stamp = stamp
        self.good += 1
        if self.good < 3:
            return None
        self.state, self.reason, self.recover_by = 'tracking', None, None
        self.pause_context = None
        if shared is not None:
            sample['vendor']['relocalization'] = dict(method='native_f2m_landmarks',
                anchor_stamp_ns=self.last_pose[0], shared_landmarks=shared)
            self.relocalization_points = None
            self.relocalization_deadline = self.relocalization_deadline_mono = None
        self.anchor_info = info if len(getattr(info, 'word_inliers', ())) >= F2M_SHARED_INLIERS else None
        self.last_pose = (stamp, position, rotation)
        return sample


def estimator_environment(directory):
    """Give only the native child room for fragmented RGBDImage UDP bursts."""
    return dict(receive_environment(directory), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')


def estimator_parameters(calibration, segment):
    frontend = calibration.get('visual_frontend', 'f2m_feature')
    if frontend not in ('f2m_feature', 'f2f_optical_flow'):
        raise ValueError('unsupported RGB-D visual frontend')
    max_features = calibration.get('max_features', 1000)
    if type(max_features) is not int or not 100 <= max_features <= 5000:
        raise ValueError('invalid RGB-D max_features')
    # Pinned RTAB-Map OdometryF2M forcibly replaces Vis/CorType=1 with 0.
    # Optical flow must use F2F (Odom/Strategy=1), never a fake F2M toggle.
    optical = frontend == 'f2f_optical_flow'
    params = {'frame_id': calibration['frame_id'], 'odom_frame_id': segment + '_odom',
              'publish_tf': False, 'subscribe_rgbd': True, 'qos': 2,
              'topic_queue_size': 2, 'sync_queue_size': 2,
              'Odom/ResetCountdown': '0', 'Odom/Strategy': '1' if optical else '0',
              'Vis/CorType': '1' if optical else '0',
              'Vis/MaxFeatures': str(max_features)}
    if not optical and not (calibration.get('imu_topic') and not calibration.get('short_term_compensation')):
        params['motus_matched_landmarks'] = True
    if optical:
        params.update({'Vis/CorFlowWinSize': '16', 'Vis/CorFlowMaxLevel': '3',
                       'Vis/CorFlowIterations': '30'})
    if calibration.get('imu_topic') and not calibration.get('short_term_compensation'):
        params.update(wait_imu_to_init=True, **{'Odom/GuessSmoothingDelay': '0'})
    else:
        # Camera-start map axes: optical forward/right/down become +X/-Y/-Z.
        # This sets the world basis, not the camera child frame or a body mount.
        # Native attitude-only IMU initialization above owns gravity alignment;
        # fusion keeps this visual basis, including visual-only fallback.
        params['initial_pose'] = '0 0 0 -1.5707963267948966 0 -1.5707963267948966'
    if association_enabled(calibration):
        params.update(motus_rotation_hint=True, wait_imu_to_init=False, guess_frame_id='',
                      **{'Odom/GuessMotion': 'true', 'Odom/GuessSmoothingDelay': '0',
                         'Odom/FilteringStrategy': '0', 'Optimizer/GravitySigma': '0'})
    return params


def registration_report(sample, frontend, native=None, *, depth_enabled=False):
    """Optional diagnostics from the *same capture frame*, never a pose claim.

    OdomInfo inliers cover optical-flow or feature correspondences.
    Depth diagnostics come from this sample's fusion snapshot, never mutable
    filter state. Missing measurements remain None.
    """
    vendor = sample.get('vendor') or {}
    fusion = vendor.get('fusion') or {}
    quality = fusion.get('quality') or {}
    depth = quality.get('geometry') or {}
    native = native or {}
    image = ({key: quality.get(key) for key in
              ('visual_usable', 'depth_fraction', 'contrast', 'sharpness', 'reason')}
             if 'visual_usable' in quality else None)
    return {
        'frontend': frontend,
        'status': (vendor.get('motus_slam') or {}).get('status'),
        'native_inliers': native.get('inliers'),
        'native_lost': native.get('lost'),
        'capture_to_diagnostic_ms': native.get('latency_ms'),
        'image_quality': image,
        'depth_icp': {
            'enabled': bool(depth_enabled),
            'attempted': bool(depth.get('attempted', False)),
            'accepted': bool(depth.get('accepted', False)),
            'inliers': depth.get('inliers'),
            'rmse_m': depth.get('rmse'),
            'overlap': depth.get('overlap'),
            'condition': depth.get('condition'),
            'rank': depth.get('rank'),
            'full_pose_supported': depth.get('full_pose_supported'),
            'reject_reason': depth.get('reason')
        }
    }


class CaptureObservations:
    """Bounded exact-stamp joins with capture-order delivery and late-result discard."""
    def __init__(self, fusion):
        self.fusion = fusion
        self.frames = OrderedDict()
        self.last_added = 0
        self.geometry_capture = None
        self.last_tracking_info = None
        self.counts = Counter()

    def add_frame(self, stamp, color, depth, now):
        outputs = self.drain(now)
        if stamp <= self.last_added or len(self.frames) >= 8:
            self.fusion.fault('fusion capture queue repeated, reset or overflowed', 'input')
            return outputs
        self.last_added = stamp
        self.frames[stamp] = dict(color=color, depth=depth, added=now, deadline=now+250_000_000)
        self.counts['queued'] += 1
        return outputs

    def result(self, kind, msg, now):
        stamp = stamp_ns(msg)
        self.counts[kind+'_received'] += 1
        if stamp not in self.frames:
            # Includes drained captures and results never queued by this
            # consumer; it is not evidence of a native/DDS dropped frame.
            self.counts[kind+'_unqueued'] += 1
        elif now > self.frames[stamp]['deadline']:
            self.counts[kind+'_after_deadline'] += 1
        else:
            self.frames[stamp][kind] = msg
            self.counts[kind+'_joined'] += 1
        return self.drain(now)

    def _visual(self, stamp, entry):
        msg, info = entry.get('pose'), entry.get('info')
        if info is not None and info.lost:
            return None, 'native visual tracking lost'
        if info is None:
            return None, 'visual diagnostic missing' if msg is not None else 'visual pose and diagnostic missing'
        if info.inliers < 10:
            return None, 'visual inliers below 10'
        if msg is None:
            return None, 'visual pose missing'
        try:
            if msg.header.frame_id != self.fusion.segment+'_odom' or msg.child_frame_id != self.fusion.frame:
                self.fusion.fault('native visual frame/segment changed; restart required', 'frame')
                return None, 'visual frame changed'
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            pose = dict(position=[p.x, p.y, p.z], quaternion=[q.x, q.y, q.z, q.w],
                        frame_id=msg.header.frame_id, child_frame_id=msg.child_frame_id,
                        covariance=list(msg.pose.covariance), quality_weight=max(1., 50./info.inliers))
            pose_values(pose)
            observation_covariance(pose)
            return pose, None
        except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
            return None, 'invalid visual pose or covariance'

    def drain(self, now):
        outputs = []
        if self.geometry_capture is not None:
            self.fusion.poll_geometry(now)
            if self.fusion.geometry_job is not None:
                return outputs
            # Complete the measured capture before any newer filter state.
            outputs.append(self.fusion.output(self.geometry_capture))
            self.geometry_capture = None
        while self.frames:
            stamp, entry = next(iter(self.frames.items()))
            if now-stamp > MAX_FRAME_AGE_NS:
                # A delayed delivery is unusable without changing the capture clock.
                self.frames.pop(stamp)
                self.counts['stale_capture'] += 1
                continue
            ready = ('pose' in entry and 'info' in entry) or getattr(entry.get('info'), 'lost', False)
            coverage = bool(self.fusion.samples) and self.fusion.samples[-1][0] >= stamp
            if (not ready or not coverage) and now < entry['deadline']:
                break
            self.frames.pop(stamp)
            self.counts['completed'] += 1
            if not ready or not coverage:
                self.counts['deadline_expired'] += 1
            if not coverage:
                self.counts['imu_uncovered'] += 1
            visual, reason = self._visual(stamp, entry)
            sample = self.fusion.observe_frame(stamp, entry['color'], entry['depth'], visual, now,
                                               reason=reason, arrival_ns=entry['added'])
            if visual is not None and sample.get('pose') is not None:
                self.last_tracking_info = entry['info']
            if getattr(self.fusion, 'geometry_job', None) is not None:
                self.geometry_capture = stamp
                break
            outputs.append(sample)
        return outputs


def observation_calibration(calibration, ingress_clock_id):
    result = dict(calibration)
    if result.get('time_basis') == 'receipt_estimated':
        result['clock_id'] = ingress_clock_id
    return result


class RgbdSession:
    def __init__(self, calibration, executor, topic, *, publish_output=True, input_registry=None,
                 requested_calibration=None, mode_transition=None):
        self.input_registry, self.input_leases = input_registry, []
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from nav_msgs.msg import Odometry
        from rtabmap_msgs.msg import RGBDImage, OdomInfo
        self.requested_calibration = dict(calibration if requested_calibration is None else requested_calibration)
        self.time_policy = TimePolicy(input_registry.clock_id if input_registry else 'rgbd_ingress/' + uuid.uuid4().hex)
        self.contexts_by_stamp = OrderedDict()
        calibration = observation_calibration(calibration, self.time_policy.clock_id)
        self.assistance_unavailable = None
        if association_enabled(calibration):
            binary = require_runtime()
            try:
                require_rotation_hint_runtime(binary)
            except RuntimeError as error:
                calibration['inertial_association'] = False
                self.assistance_unavailable = str(error)
        self.calibration, self.executor, self.topic = calibration, executor, topic
        self.publish_output = publish_output
        self.segment = 'rgbd_' + uuid.uuid4().hex[:12]
        self.gate = EstimateGate(self.segment, calibration['frame_id'], calibration['clock_id'])
        self.gate.recovery_policy = RecoveryPolicy() if input_registry is not None else None
        self.motion_gate = MotionGate(calibration.get('motion_limits'))
        self.previous_motion = None
        self.quality_by_stamp = OrderedDict()
        self.processing_ages = deque(maxlen=32)
        self.processing_observations = deque(maxlen=32)
        self.stage_times={k:deque(maxlen=128) for k in ('decode','imu_wait')}
        self.fusion = (ShortTermFusion(self.segment, calibration['frame_id'], calibration['clock_id'],
                       calibration['K'], calibration.get('depth_geometry', False),
                       inertial_association=association_enabled(calibration))
                       if calibration.get('short_term_compensation') else None)
        if self.fusion is not None:
            self.fusion.recovery_policy = self.gate.recovery_policy
        self.observations = CaptureObservations(self.fusion) if self.fusion else None
        self.mounts = (CaptureMounts(calibration['body_mount'], calibration['frame_id'], calibration['clock_id'])
                       if calibration.get('body_mount') else None)
        self.imu = (self.fusion.imu if self.fusion else
                    ImuAttitude(calibration['frame_id'], calibration['clock_id']) if calibration.get('imu_topic') else None)
        self.auxiliary_probe = (AuxiliaryHealthProbe(calibration['frame_id'], calibration['clock_id'],
            require_translation=bool(self.requested_calibration.get('short_term_compensation')))
            if self.requested_calibration.get('imu_topic')
            else None)
        self.imu_coverage_timed_out = False
        self.imu_topic = (calibration.get('imu_topic') or self.requested_calibration.get('imu_topic')
                          if self.auxiliary_probe is not None else calibration.get('imu_topic'))
        self.lock = threading.RLock()
        self.imu_queue_lock = threading.Lock()
        self.imu_pending, self.imu_overflow = deque(), False
        self.pairs = RgbdPairer() if input_registry is not None or 'input_declarations' in calibration else PairBuffer()
        self.pending = queue.Queue(maxsize=2)
        self.probe_stamp = None
        self.probe_ack = threading.Event()
        self.latest_inputs = {}
        self.results = {'pose': OrderedDict(), 'info': OrderedDict()}
        self.registration_by_stamp = OrderedDict()
        self.last_registration = None
        self.status_publisher = None
        self.summary_origin = None
        self.last_summary_ns = self.last_summary_state = None
        self.native_reports = 0
        self.stop_event = threading.Event()
        self.process = self.worker = None
        self.spin_worker = None
        self.geometry_worker = None
        self.rotation_hints = None
        self.owns_executor = False
        self.directory = tempfile.TemporaryDirectory(prefix='motus-rgbd-')
        self.log = None
        self.counts = dict(pairs=0, estimates=0, dropped_pairs=0, rejected_inputs=0)
        self.input_rejection = None
        self.fallback = None
        self.started = time.monotonic()
        self.visual_initialization_started = self.mode_changed_at = self.started
        self.first_tracking_at = None
        self.initialization_timed_out = False
        self.mode_transition = deepcopy(mode_transition)
        self.node = Node(self.segment)
        try:
            if self.fusion is not None and self.fusion.geometry is not None:
                # ICP never runs under self.lock: IMU ingress, native results
                # and stop remain responsive while it computes.
                self.geometry_worker = GeometryWorker(self.fusion.geometry)
                self.fusion.geometry_runner = self.geometry_worker
            if self.imu_topic is not None:
                from rclpy.executors import SingleThreadedExecutor
                self.executor = SingleThreadedExecutor(context=self.node.context)
                self.owns_executor = True
            self.publisher = self.node.create_publisher(String, topic, qos_profile_sensor_data)
            self.status_publisher = self.node.create_publisher(
                String, (topic[:-11] if topic.endswith('/state/odom') else topic) + '/status',
                qos_profile_sensor_data)
            self.rgbd = self.node.create_publisher(RGBDImage, '/' + self.segment + '/rgbd_image', qos_profile_sensor_data)
            self.rotation_hints = RotationHints(calibration, self.node, self.segment, qos_profile_sensor_data)
            for kind in ('color', 'depth'):
                input_topic = calibration[kind + '_topic']
                if input_registry is None:
                    self.node.create_subscription(CompressedImage, input_topic,
                        lambda m, k=kind: self._image(k, m), qos_profile_sensor_data)
                else:
                    self.input_leases.append(input_registry.acquire(input_topic, CompressedImage,
                        calibration['input_declarations'].get(input_topic, {}),
                        lambda m, context, k=kind: self._image(k, m, context)))
            self.node.create_subscription(Odometry, '/' + self.segment + '/visual',
                                          lambda m: self._result('pose', m), qos_profile_sensor_data)
            self.node.create_subscription(OdomInfo, '/' + self.segment + '/diagnostic',
                                          lambda m: self._result('info', m), qos_profile_sensor_data)
            if self.imu_topic is not None:
                from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
                if self.imu is not None and self.fusion is None:
                    from sensor_msgs.msg import Imu
                    self.imu_pub = self.node.create_publisher(Imu, '/' + self.segment + '/imu', qos_profile_sensor_data)
                # Preserve capture intervals while a visual correction occupies
                # the callback group; admission still rejects samples over 750 ms old.
                imu_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
                self.imu_callback_group = MutuallyExclusiveCallbackGroup()
                self.node.create_subscription(String, self.imu_topic, self._enqueue_imu, imu_qos,
                                              callback_group=self.imu_callback_group)
            if self.mounts is not None:
                self.node.create_subscription(String, self.mounts.declaration['topic'], self._mount, qos_profile_sensor_data)
            self.node.create_timer(.05 if self.fusion else .2, self._watchdog)
            self.executor.add_node(self.node)
            params = estimator_parameters(calibration, self.segment)
            path = Path(self.directory.name) / 'parameters.json'
            path.write_text(json.dumps({'/**': {'ros__parameters': params}}))
            self.log = (Path(self.directory.name) / 'estimator.log').open('ab', buffering=0)
            cmd = [str(require_runtime()), '--ros-args', '--params-file', str(path),
                   '-r', '__ns:=/' + self.segment, '-r', '__node:=rgbd_odometry',
                   '-r', 'odom:=/' + self.segment + '/visual',
                   '-r', 'odom_info_lite:=/' + self.segment + '/diagnostic']
            if self.imu is not None and self.fusion is None:
                cmd.extend(['-r', 'imu:=/' + self.segment + '/imu'])
            if self.rotation_hints.enabled:
                cmd.extend(['-r', 'rotation_hint:=/' + self.segment + '/rotation_hint'])
            self.process = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                start_new_session=True, env=estimator_environment(self.directory.name))
            worker = threading.Thread(target=self._work, name=self.segment, daemon=True)
            worker.start()
            self.worker = worker
            if self.owns_executor:
                spin_worker = threading.Thread(target=self._spin, name=self.segment+'_ros', daemon=True)
                spin_worker.start()
                self.spin_worker = spin_worker
        except Exception:
            self.close(report_idle=False)
            raise

    def _spin(self):
        try:
            while not self.stop_event.is_set():
                self.executor.spin_once(timeout_sec=.05)
        except Exception as error:
            if not self.stop_event.is_set():
                with self.lock:
                    self.gate.fault('RGB-D callback executor: ' + str(error), 'process')
                    if self.fusion is not None:
                        self.fusion.fault(self.gate.reason)

    def _image(self, kind, msg, context=None):
        if self.stop_event.is_set():
            return
        with self.lock:
            self._drain_imu()
            try:
                now, stamp = time.time_ns(), stamp_ns(msg)
                if isinstance(self.pairs, RgbdPairer):
                    if context is None:
                        declaration = self.calibration.get('input_declarations', {}).get(self.calibration[kind + '_topic'], {})
                        source = dict(declaration.get('vendor') or {}, stamp_ns=stamp)
                        context = self.time_policy.select(source, now, time.monotonic_ns())
                    stamp = context.observation_stamp_ns
                self.latest_inputs[kind] = now
                if self.gate.status(now) == 'lost':
                    return
                try:
                    pair = (self.pairs.push(kind, msg, context) if isinstance(self.pairs, RgbdPairer) else
                            self.pairs.push(kind, stamp_ns(msg), msg, time.time_ns()))
                except ValueError as error:
                    # Only malformed payload structure is rejected here.
                    self.counts['rejected_inputs'] += 1
                    self.input_rejection = f'{kind}: {error}'
                    return
                if pair:
                    self.gate.input(pair[1][0], now_ns=time.time_ns())
                    # Keep the two measured successors needed after a slow first
                    # registration. Normal tracking still takes only the latest.
                    capacity = 2 if self.gate.relocalization_points is not None else 1
                    while self.pending.qsize() >= capacity:
                        try:
                            self.pending.get_nowait()
                        except queue.Empty:
                            break
                        self.counts['dropped_pairs'] += 1
                    self.pending.put_nowait(pair)
            except (ValueError, queue.Full) as error:
                self.gate.fault(error, 'frame')

    def has_fresh_inputs(self, now_ns):
        with self.lock:
            return all(kind in self.latest_inputs and
                       -50_000_000 <= now_ns-self.latest_inputs[kind] <= 1_500_000_000
                       for kind in ('color', 'depth'))

    def _probing_auxiliary(self):
        return (getattr(self, 'auxiliary_probe', None) is not None and
                (self.imu is None or self.gate.state == 'lost' or
                 (self.fusion is None and self.requested_calibration.get('short_term_compensation')) or
                 (getattr(self, 'fusion', None) is not None and self.fusion.state == 'lost')))

    def _mode_info(self):
        requested = assistance_mode(getattr(self, 'requested_calibration', self.calibration))
        effective = (self.fallback['mode'] if getattr(self, 'fallback', None) is not None
                     else assistance_mode(self.calibration))
        transition = getattr(self, 'mode_transition', None)
        return dict(requested=requested, effective=effective, degraded=requested != effective,
                    reason=(self.fallback['reason'] if getattr(self, 'fallback', None) is not None else
                            getattr(self, 'assistance_unavailable', None) or
                            (transition['reason'] if transition else None)),
                    transition=deepcopy(transition))

    def _auxiliary_info(self, now_ns, now_s):
        probe = getattr(self, 'auxiliary_probe', None)
        if probe is None:
            return dict(enabled=False, status='unavailable' if getattr(self, 'assistance_unavailable', None)
                        else 'disabled', reason=getattr(self, 'assistance_unavailable', None))
        if not self._probing_auxiliary():
            return dict(enabled=True, status='inactive', reason='requested assistance is active')
        return dict(probe.info(now_ns, now_s), restore_cooldown_remaining_s=round(max(
            0., ASSISTANCE_RETRY_S-(now_s-self.mode_changed_at)), 2))

    def _initialization_info(self, now_s):
        started = getattr(self, 'visual_initialization_started', getattr(self, 'started', now_s))
        finished = getattr(self, 'first_tracking_at', None)
        if finished is not None:
            phase = 'complete'
        elif getattr(self, 'initialization_timed_out', False):
            phase = 'timed_out'
        elif not (getattr(self.gate, "last_input_received", 0) or self.gate.last_input):
            phase = 'waiting_rgbd'
        elif self._fusing():
            phase = 'waiting_low_motion' if self.fusion.filter is None else 'waiting_consistency'
        elif self.imu is not None and self.imu.state == 'initializing':
            phase = 'waiting_imu'
        else:
            phase = 'waiting_visual'
        reason = self.fusion.reason if self._fusing() else self.gate.reason
        return dict(phase=phase, elapsed_s=round(max(0., (now_s if finished is None else finished)-started), 2),
                    budget_s=VISUAL_INITIALIZATION_S, reason=reason,
                    consecutive_visual_estimates=self.fusion.good if self._fusing() else self.gate.good,
                    required_visual_estimates=3,
                    fusion_visual_samples=len(self.fusion.warmup) if self._fusing() else None,
                    imu_samples=len(self.imu.history) if self.imu is not None else 0)

    def _check_initialization(self, now_s):
        if self.gate.state == 'lost' or getattr(self, 'first_tracking_at', None) is not None:
            return
        if self.gate.last_pose is not None or (self.gate.state == 'tracking' and not self._fusing()):
            return  # a tracking segment has separate bounded visual recovery rules
        if not (getattr(self.gate, "last_input_received", 0) or self.gate.last_input):
            return  # the watchdog's separate input timeout identifies missing images
        if now_s-getattr(self, 'visual_initialization_started', self.started) < VISUAL_INITIALIZATION_S:
            return
        if self._fusing():
            # Constructing an ESKF is not successful initialization. Repeated
            # short coasts may otherwise keep it below its first tracking pose
            # forever while each new visual sample refreshes the outage bound.
            self.fusion.fault(f'fusion initialization timed out after {VISUAL_INITIALIZATION_S:g} s '
                              'without an admitted tracking pose; visual fallback required')
            self._check_fusion(time.time_ns())
        elif self.imu is None or self.fusion is not None or self.imu.state != 'initializing':
            # An absent attitude source keeps the distinct 10 s IMU timeout.
            # A healthy attitude source does not extend visual warmup forever.
            self.initialization_timed_out = True
            # Only a fresh result rejected after the last valid one is visual evidence.
            kind = 'visual' if self.gate.last_result_stamp > self.gate.last_stamp else 'output'
            self.gate.fault(f'visual initialization timed out after {VISUAL_INITIALIZATION_S:g} s '
                            'without three valid estimates; restart required', kind)

    def recovery_target(self, now_s, now_ns):
        """Recommend a new segment; never enable an auxiliary filter in place."""
        with self.lock:
            mode = self._mode_info()
            source = getattr(self, 'restart_calibration', self.calibration)
            health = self._auxiliary_info(now_ns, now_s)
            if (mode['degraded'] and health.get('status') == 'ready' and
                    now_s-self.mode_changed_at >= ASSISTANCE_RETRY_S):
                calibration = dict(source)
                for key in ('imu_topic', 'imu_assisted', 'short_term_compensation',
                            'depth_geometry', 'inertial_association'):
                    if key in self.requested_calibration:
                        calibration[key] = self.requested_calibration[key]
                return dict(action='restore_assistance', calibration=calibration,
                            reason='auxiliary source passed numeric format checks')
            attitude = None
            fusion_lost = self.gate.state == 'lost' and self.gate.loss_kind == 'fusion'
            if fusion_lost or (self.fallback is not None and not self.rotation_hints.enabled):
                attitude = self.imu
            elif (self.imu is None and health.get('attitude_ready') and
                  now_s-self.mode_changed_at >= ASSISTANCE_RETRY_S):
                attitude = self.auxiliary_probe.validator
            if attitude is not None and attitude.covers(self.gate.last_input, now_ns):
                # Native gravity initialization owns a different world basis.
                # Switch through the supervisor so it publishes a fresh segment.
                return dict(action='degrade_assistance' if fusion_lost else 'restore_assistance', calibration=dict(source,
                    imu_topic=self.requested_calibration['imu_topic'], imu_assisted=True,
                    short_term_compensation=False, depth_geometry=False, inertial_association=False),
                    reason='ESKF unavailable; rotation available for native attitude assistance')
            auxiliary_loss = (self.imu is not None and
                              (self.gate.loss_kind in ('imu', 'fusion') or self.imu_coverage_timed_out))
            if self.gate.state == 'lost' and (auxiliary_loss or getattr(self, 'fallback', None) is not None):
                if self.rotation_hints.enabled:
                    # Optional hints never block images. Preserve their input in
                    # the new segment even if the next ESKF has no lever arm.
                    return dict(action='restart', calibration=source, reason=self.gate.reason)
                return dict(action='degrade_assistance', calibration=visual_calibration(source),
                            reason=self.gate.reason or mode['reason'])
            return None

    def _mount(self, msg):
        with self.lock:
            if self.stop_event.is_set():
                return
            try:
                self.mounts.observe(json.loads(msg.data), time.time_ns())
            except (ValueError, TypeError, AttributeError) as error:
                # Body-reference evidence is optional: camera VO remains valid.
                self.mounts.fault(error)

    def _enqueue_imu(self, msg):
        if self.stop_event.is_set():
            return
        with self.imu_queue_lock:
            if len(self.imu_pending) >= 100:
                self.imu_overflow = True
                self.imu_pending.popleft()
            self.imu_pending.append(msg)

    def _drain_imu(self):
        # Caller owns the estimator lock; ingress only touches its own queue.
        if not hasattr(self, 'imu_pending'):
            return
        with self.imu_queue_lock:
            pending = list(self.imu_pending)
            self.imu_pending.clear()
            self.imu_overflow = False
        for msg in pending:
            if self.stop_event.is_set():
                return
            self._imu(msg)

    def _imu(self, msg):
        with self.lock:
            if self.stop_event.is_set():
                return
            try:
                packet = json.loads(msg.data)
                if self._probing_auxiliary():
                    self.auxiliary_probe.observe(packet, time.time_ns(), time.monotonic())
                if self.gate.state == 'lost' or self.imu is None:
                    return
                if getattr(self, 'fusion', None) is not None:
                    if self.fusion.state != 'lost':
                        try:
                            self.fusion.observe_imu(packet, time.time_ns())
                        except ValueError:
                            # Each stage records its own fault. Missing translation
                            # stops ESKF but leaves a valid rotation available.
                            return
                    else:
                        self.imu.observe(packet, time.time_ns())
                    return
                sample = self.imu.observe(packet, time.time_ns())
                if sample is None:
                    return
                from sensor_msgs.msg import Imu
                out = Imu()
                out.header.frame_id = self.calibration['frame_id']
                out.header.stamp.sec, out.header.stamp.nanosec = divmod(sample['stamp_ns'], 10**9)
                (out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w) = sample['quaternion']
                out.orientation_covariance = [0.] * 9  # estimated orientation; covariance unknown
                (out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z) = sample['angular_velocity']
                (out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z) = sample['linear_acceleration']
                self.imu_pub.publish(out)
            except (ValueError, TypeError, AttributeError, RuntimeError) as error:
                if self._probing_auxiliary():
                    self.auxiliary_probe.reset(error)
                if self.gate.state == 'lost' or self.imu is None:
                    return
                self.imu.fault(error)
                if getattr(self, 'fusion', None) is not None:
                    self.fusion.fault(error)
                else:
                    self.gate.fault('IMU assistance: ' + str(error), 'imu')

    def _imu_frame_ready(self, stamp, now):
        if self.gate.status(now) == 'lost':
            return False
        if self.imu is None or getattr(self, 'fusion', None) is not None:
            return True  # only the attitude-prior native estimator waits for IMU coverage
        covered = self.imu.covers(stamp, now)
        if self.imu.state == 'lost':
            self.gate.fault('IMU assistance: ' + str(self.imu.reason), 'imu')
        return covered

    def _wait_for_imu(self, stamp, *, deadline=None):
        # Images and IMU travel independently. Allow bounded arrival skew,
        # releasing the estimator lock while waiting for actual IMU coverage.
        if deadline is None:
            deadline = time.monotonic() + .15
        while not self.stop_event.is_set():
            with self.lock:
                self._drain_imu()
                if self.fusion is not None or self._imu_frame_ready(stamp, time.time_ns()):
                    self.imu_coverage_timed_out = False
                    return True
                if self.gate.state == 'lost':
                    return False
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                with self.lock:
                    self.imu_coverage_timed_out = True
                return False
            self.stop_event.wait(min(.005, remaining))
        return False

    def _work(self):
        from sensor_msgs.msg import CompressedImage, CameraInfo
        from rtabmap_msgs.msg import RGBDImage
        import cv2
        while not self.stop_event.is_set():
            try:
                pair = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                (_, color), (_, depth) = pair
                # Native names identify the configured numeric optical frame.
                # Producer labels remain on the originals retained by the pair.
                color, depth = deepcopy(color), deepcopy(depth)
                color.header.frame_id = depth.header.frame_id = self.calibration['frame_id']
                if isinstance(pair, RgbdPair):
                    # Only native transport copies gain observation stamps. Source
                    # headers and both original contexts remain in the association.
                    for message, ctx in ((color, pair.color_time), (depth, pair.depth_time)):
                        message.header.stamp.sec, message.header.stamp.nanosec = divmod(ctx.observation_stamp_ns, 10**9)
                capture_stamp = native_capture_stamp(color, depth)
                if isinstance(pair, RgbdPair):
                    selected = pair.color_time if capture_stamp == pair.color_time.observation_stamp_ns else pair.depth_time
                    other=pair.depth_time if selected is pair.color_time else pair.color_time
                    selected=replace(selected,observation_id=uuid.uuid4().hex,
                        parent_observation_ids=(pair.color_context.observation_id,pair.depth_context.observation_id),
                        transformations=selected.transformations+('rgbd_association',),
                        **{k:getattr(selected,k) if getattr(selected,k) is not None else getattr(other,k)
                           for k in ('source_id','source_epoch','geometry_version','mapping_version')})
                    with self.lock:
                        if self.gate.last_result_stamp == 0:
                            self.gate.clock = selected.normalized_clock_id
                            self.calibration['clock_id'] = selected.normalized_clock_id
                        self.contexts_by_stamp[capture_stamp] = (selected, pair)
                        while len(self.contexts_by_stamp) > 32:self.contexts_by_stamp.popitem(last=False)
                # Prepare images while IMU arrives, retaining the original wait budget.
                imu_deadline = time.monotonic() + .15
                decode_started=time.monotonic_ns()
                image = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
                decoder = zlib.decompressobj()
                depth_width,depth_height=self.calibration.get('depth_size',(640,480))
                needed = depth_width * depth_height * 2
                raw = decoder.decompress(bytes(depth.data), needed + 1)
                color_width,color_height=self.calibration.get('color_size',(640,480))
                if (image is None or image.shape != (color_height, color_width, 3) or len(raw) != needed or
                        not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
                    raise ValueError('RGB-D payload does not match declared image dimensions')
                if (color_width,color_height)!=(depth_width,depth_height):
                    image=cv2.resize(image,(depth_width,depth_height),interpolation=cv2.INTER_AREA)
                    encoded,buffer=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
                    if not encoded:raise ValueError('registered RGB resize encoding failed')
                    color=deepcopy(color)
                    color.data=array('B',buffer.tobytes())
                z16 = np.frombuffer(raw, '<u2').reshape(depth_height, depth_width)
                image_quality = observation_quality(image, z16.astype(float)*.001)
                with self.lock:
                    if not hasattr(self, 'quality_by_stamp'):self.quality_by_stamp = OrderedDict()
                    self.quality_by_stamp[capture_stamp] = QualityReport(**{k:image_quality[k] for k in ('contrast','sharpness','depth_fraction')})
                    while len(self.quality_by_stamp) > 32:self.quality_by_stamp.popitem(last=False)
                out = RGBDImage()
                out.header = deepcopy(depth.header)
                # RTAB-Map supports lossless PNG depth in RGBDImage. Keep the
                # native inter-process message bounded instead of fragmenting
                # 1.5 MB of raw RGB/depth over best-effort DDS on every frame.
                ok, png = cv2.imencode('.png', z16, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok:
                    raise ValueError('RGB-D lossless depth encoding failed')
                out.rgb_compressed = deepcopy(color)
                out.depth_compressed = CompressedImage()
                out.depth_compressed.header = deepcopy(depth.header)
                out.depth_compressed.format = 'png'
                out.depth_compressed.data = array('B', png.tobytes())
                k = self.calibration['K'].ravel().tolist()
                for field in ('rgb_camera_info', 'depth_camera_info'):
                    info = CameraInfo()
                    info.header = deepcopy(out.header)
                    info.width, info.height = depth_width, depth_height
                    info.k, info.d, info.distortion_model = k, [0.] * 5, 'plumb_bob'
                    info.r = np.eye(3).ravel().tolist()
                    info.p = [k[0], 0., k[2], 0., 0., k[4], k[5], 0., 0., 0., 1., 0.]
                    setattr(out, field, info)
                if hasattr(self,'stage_times'):
                    with self.lock:self.stage_times['decode'].append(max(0,time.monotonic_ns()-decode_started))
                wait_started=time.monotonic_ns()
                covered=self._wait_for_imu(capture_stamp, deadline=imu_deadline)
                if hasattr(self,'stage_times'):
                    with self.lock:self.stage_times['imu_wait'].append(max(0,time.monotonic_ns()-wait_started))
                if not covered:
                    with self.lock:
                        self.counts['dropped_pairs'] += 1
                    continue
                probe_deadline = None
                with self.lock:
                    self._drain_imu()
                    ready = (self.gate.state != 'lost' if self.fusion is not None else
                             self._imu_frame_ready(capture_stamp, time.time_ns()))
                    if ready and not self.stop_event.is_set():
                        if self._fusing():
                            for sample in self.observations.add_frame(capture_stamp, image, z16, time.time_ns()):
                                self._publish(sample)
                                self.counts['estimates'] += 1
                            self._check_fusion(time.time_ns())
                        if self.gate.state == 'lost':
                            continue
                        if getattr(self, 'rotation_hints', None) is not None:
                            self.rotation_hints.offer(self.imu, capture_stamp, time.time_ns())
                        probe_deadline = getattr(self.gate, 'relocalization_deadline_mono', None)
                        if probe_deadline is not None:
                            self.probe_stamp = capture_stamp
                            self.probe_ack.clear()
                        self.rgbd.publish(out)
                        self.counts['pairs'] += 1
                if probe_deadline is not None and not self.stop_event.is_set():
                    # Native odometry has one processing slot. Release the
                    # session lock while it registers this actual capture.
                    self.probe_ack.wait(max(0., (probe_deadline - time.monotonic_ns()) / 1e9))
            except Exception as error:
                with self.lock:
                    self.gate.fault(error)

    def _fusing(self):
        return getattr(self, 'fusion', None) is not None and getattr(self, 'fallback', None) is None

    def _check_fusion(self, now):
        if (getattr(self, 'fusion', None) is not None and getattr(self, 'fallback', None) is None and
                self.fusion.state == 'lost' and self.gate.state != 'lost'):
            if self.fusion.loss_kind in ('frame', 'clock', 'input'):
                self.gate.fault(self.fusion.reason, self.fusion.loss_kind)
            else:
                self._enter_fallback(now)

    def _enter_fallback(self, now):
        """ESKF stopped: continue with the native visual estimate if continuous.

        Optional rotation hints can influence native association, but never
        reset its pose origin or replace visual measurements. Its poses share
        this segment's odom frame; fused corrections can introduce an offset.
        Keep the segment only if the offset at the last published fused capture
        is within the continuity bound. This proves no statistical independence.
        """
        reason = str(self.fusion.reason)
        offset = self.fusion.tracking_offset()
        if self.fusion.last_tracking is not None and offset is None:
            self.gate.fault('ESKF lost after geometry tracking; no capture-matched visual pose '
                            'to prove fallback continuity', 'fusion')
            return
        if offset is not None and (offset[0] > FALLBACK_OFFSET_M or offset[1] > FALLBACK_OFFSET_RAD):
            self.gate.fault(f'ESKF lost ({reason}); visual-only fallback cannot prove coordinate '
                            f'continuity (fused/visual offset {offset[0]:.3f} m, '
                            f'{np.rad2deg(offset[1]):.1f} deg)', 'fusion')
            return
        hints = getattr(self, 'rotation_hints', None)
        mode = 'visual' if hints and hints.enabled else 'visual_only'
        self.fallback = dict(mode=mode, reason=reason, since_ns=now,
                             fused_visual_offset_m=None if offset is None else round(offset[0], 4),
                             fused_visual_offset_deg=None if offset is None else round(float(np.rad2deg(offset[1])), 2))
        self.mode_changed_at = time.monotonic()
        self.mode_transition = dict(action='degrade_assistance', previous_segment=self.segment,
                                    from_mode=assistance_mode(self.calibration), to_mode=mode, reason=reason)
        if self.gate.last_pose is None and self.fusion.last_tracking is None:
            # The visual phase gets one bounded initialization window after
            # unsuccessful fusion warmup, never one window per weak frame.
            self.visual_initialization_started = self.mode_changed_at
        self.observations.counts['cancelled_on_fallback'] += len(self.observations.frames)
        self.observations.frames.clear()
        gate = self.gate
        gate.good = 0
        gate.last_stamp = max(gate.last_stamp, self.fusion.last_visual)
        if self.fusion.last_tracking is not None:
            # Something was published: resuming is a bounded recovery.
            stamp, fused, _ = self.fusion.last_tracking
            gate.last_pose = (stamp, *pose_values(fused))
            info = self.observations.last_tracking_info
            if info is not None and stamp_ns(info) == stamp:
                gate.anchor_info = info
            pause = self.fusion.pause_context
            if pause is not None:
                # Keep the pause already promised to downstream mapping.
                gate.pause_context = pause
                gate.recover_by = now + max(0, pause.deadline_monotonic_ns - time.monotonic_ns())
            gate.degrade(reason, now)
        gate.reason = mode.replace('_', '-') + ' fallback after ESKF loss: ' + reason
        logging.getLogger(__name__).warning('RGB-D %s: %s', self.segment, gate.reason)

    def _label_fallback(self, sample):
        vendor = sample['vendor']
        fallback = self.fallback['mode']
        vendor['fallback'] = fallback
        vendor['fusion'] = {'enabled': True, 'status': 'lost', 'reason': self.fallback['reason'],
                            'fallback': fallback, 'mapping_allowed': sample['pose'] is not None}
        if vendor['motus_slam']['status'] == 'tracking':
            vendor['reason'] = fallback.replace('_', '-') + ' fallback: ESKF ' + self.fallback['reason'][:80]
        return sample

    def _publish(self, sample):
        if not self.publish_output:
            return  # a replacement cannot advertise tracking before old cleanup succeeds
        from std_msgs.msg import String
        self._drain_imu()
        now = time.time_ns()
        if self.imu and getattr(self, 'fusion', None) is None and self.imu.status(now) == 'lost':
            self.gate.fault('IMU assistance: ' + str(self.imu.reason), 'imu')
        self._check_fusion(now)
        stamp = sample['vendor']['motus_slam']['stamp_ns']
        if self.gate.state == 'lost' or (self._fusing() and self.fusion.state == 'lost'):
            reason = self.fusion.reason if self._fusing() else self.gate.reason
            sample = make_sample('lost', stamp, self.segment, self.calibration['clock_id'], reason=reason)
        elif getattr(self, 'fallback', None) is not None:
            if sample['vendor'].get('source') == 'rgbd_eskf':  # never forward fused state after fallback
                sample = make_sample(self.gate.state, stamp, self.segment, self.calibration['clock_id'],
                                     reason=self.gate.reason)
            sample = self._label_fallback(sample)
        if (sample['vendor']['motus_slam']['status'] == 'recovering' and not self._fusing() and
                self.gate.state == 'recovering' and self.gate.last_pose is not None and
                self.gate.relocalization_points is None):
            # Declared, bounded in-segment pause the SLAM consumer can honour.
            sample['vendor']['pause'] = dict(source='visual_gate', status='recovering', mapping_allowed=False)
        association = getattr(self, 'contexts_by_stamp', {}).get(stamp)
        context = association[0] if association else None
        if getattr(self, 'mounts', None) is not None and (context is None or context.time_basis == 'source_capture'):
            sample = self.mounts.attach(sample, time.time_ns())
        if context is not None:
            self.published_context = context
            pair = association[1]
            slam = sample['vendor']['motus_slam']
            slam.update(profile='motus_slam/2', observation=context.to_dict(), clock_id=context.normalized_clock_id)
            if slam['status'] == 'tracking':
                slam['stamp_source'] = 'measurement' if context.time_basis == 'source_capture' else 'receipt_estimated'
            sample['vendor']['association'] = dict(id=pair.association_id, method=pair.method, skew_ns=pair.skew_ns,
                inputs=[pair.color_context.to_dict(), pair.depth_context.to_dict()])
        quality = getattr(self, 'quality_by_stamp', {}).get(stamp, QualityReport())
        sample['vendor']['observation_quality'] = quality.to_dict()
        if context is not None and context.observation_id not in self.processing_observations:
            self.processing_observations.append(context.observation_id)
            self.processing_ages.append(max(0, time.monotonic_ns()-context.ingress_monotonic_ns))
            self.gate.processing_p95_ns = int(np.percentile(self.processing_ages, 95))
            if self.fusion is not None:
                self.fusion.input_period_ns = self.gate.input_period_ns
                self.fusion.processing_p95_ns = self.gate.processing_p95_ns
        pause_owner = self.fusion if self._fusing() else self.gate
        pause = getattr(pause_owner, 'pause_context', None)
        if pause is not None and sample['vendor']['motus_slam']['status'] in ('coasting','recovering'):
            anchor = (self.fusion.last_tracking[0] if self._fusing() and self.fusion.last_tracking else
                      self.gate.last_pose[0] if self.gate.last_pose else None)
            if anchor is not None:
                sample['vendor']['pause_context'] = pause.to_dict(anchor_stamp_ns=anchor, ingress_clock_id=self.time_policy.clock_id)
        if sample['vendor']['motus_slam']['status'] == 'tracking' and hasattr(self, 'motion_gate'):
            decision = self.motion_gate.evaluate(self.previous_motion, sample, context, quality)
            sample['vendor']['motion_decision'] = dict(action=decision.action, reason=decision.reason)
            if decision.action != 'accept':
                if decision.action=='new_segment':
                    self.gate.fault(decision.reason,'visual')
                    if self.fusion is not None:self.fusion.fault(decision.reason)
                else:self.gate.degrade(decision.reason, now, stamp=stamp)
                sample['pose'] = None
                sample['vendor']['reason']=decision.reason
                sample['vendor']['motus_slam'].update(status=self.gate.state, stamp_source='unknown')
                sample['vendor']['pause'] = dict(status='recovering',mapping_allowed=False)
            else:self.previous_motion = deepcopy(sample)
        sample['vendor']['imu_assistance'] = self.imu.info(time.time_ns()) if self.imu else {'enabled': False}
        sample['vendor']['inertial_association'] = (self.rotation_hints.info(stamp)
            if getattr(self, 'rotation_hints', None) is not None else {'enabled': False})
        if sample['pose'] is not None and sample['vendor']['motus_slam']['status'] == 'tracking':
            if getattr(self, 'first_tracking_at', None) is None:
                self.first_tracking_at = time.monotonic()
        sample['vendor']['mode'] = self._mode_info()
        sample['vendor']['initialization'] = self._initialization_info(time.monotonic())
        sample['vendor']['auxiliary_recovery'] = self._auxiliary_info(now, time.monotonic())
        stamp = (sample['vendor'].get('motus_slam') or {}).get('stamp_ns')
        native = self.registration_by_stamp.pop(stamp, None)
        report = registration_report(
            sample, self.calibration.get('visual_frontend', 'f2m_feature'), native,
            depth_enabled=bool(self.fusion and self.fusion.geometry))
        sample['vendor']['registration'] = report
        self.last_registration = deepcopy(report)
        output = String(data=json.dumps(sample, allow_nan=False))
        if context is not None and getattr(self, 'input_registry', None) is not None:
            self.input_registry.register_derived(self.topic, output, context, 'rgbd_odometry')
        self.publisher.publish(output)
        self._publish_status(sample, now)

    def _publish_status(self, sample, now):
        """Bumi-shaped display summary; never promote a fusion prediction to pose."""
        from std_msgs.msg import String
        vendor = sample['vendor']
        slam = vendor['motus_slam']
        state = (slam['status'], vendor['reason'])
        tick = time.monotonic_ns()
        if (self.last_summary_ns is not None and tick-self.last_summary_ns < 200_000_000
                and state == self.last_summary_state):
            return
        position = sample['pose']['position'] if sample['pose'] is not None else None
        if position is not None and self.summary_origin is None:
            self.summary_origin = list(position)
        summary = dict(status=slam['status'], turn_angle_deg=None,
                       displacement_m=round(math.dist(position, self.summary_origin), 3)
                       if position is not None else None,
                       position_m=[round(v, 3) for v in position] if position is not None else None,
                       pose_reference=vendor['pose_reference'],
                       age_ms=round((now-slam['stamp_ns'])/1e6, 1)
                       if slam['stamp_source'] == 'measurement' else None,
                       reason=vendor['reason'],
                       mode=deepcopy(vendor.get('mode', self._mode_info())),
                       initialization=deepcopy(vendor.get('initialization', self._initialization_info(time.monotonic()))),
                       auxiliary_recovery=deepcopy(vendor.get('auxiliary_recovery',
                           self._auxiliary_info(now, time.monotonic()))))
        try:
            self.status_publisher.publish(String(data=json.dumps(summary, allow_nan=False)))
        except Exception as error:
            logging.getLogger(__name__).warning('RGB-D status unavailable: %s', error)
        self.last_summary_ns, self.last_summary_state = tick, state

    def _remember_native(self, msg):
        """Store only capture-time-matched diagnostics, with bounded memory."""
        stamp = stamp_ns(msg)
        if stamp <= 0:
            return
        inliers = getattr(msg, 'inliers', None)
        inliers = inliers if type(inliers) is int and inliers >= 0 else None
        lost = getattr(msg, 'lost', None)
        lost = lost if type(lost) is bool else None
        latency_ms = (time.time_ns() - stamp) / 1e6
        if not 0. <= latency_ms <= 1500.:
            latency_ms = None
        self.registration_by_stamp[stamp] = {
            'inliers': inliers, 'lost': lost, 'latency_ms': latency_ms}
        self.registration_by_stamp.move_to_end(stamp)
        while len(self.registration_by_stamp) > 32:
            self.registration_by_stamp.popitem(last=False)
        self.native_reports += 1

    def _result(self, kind, msg):
        with self.lock:
            if self.stop_event.is_set():
                return
            self._drain_imu()
            # Frame identity is a session boundary even when an IMU failure
            # has already stopped fusion and pending captures will be cleared.
            if kind == 'pose' and (msg.header.frame_id != self.gate.segment+'_odom' or
                                   msg.child_frame_id != self.gate.frame):
                self.gate.fault('native visual frame or segment changed; restart required', 'frame')
                if getattr(self, 'fusion', None) is not None:
                    self.fusion.fault(self.gate.reason, 'frame')
                return
            if kind == 'info' and hasattr(self, 'registration_by_stamp'):
                self._remember_native(msg)
                if stamp_ns(msg) == getattr(self, 'probe_stamp', None):
                    self.probe_ack.set()
            if self._fusing():
                if self.gate.state != 'lost':
                    for sample in self.observations.result(kind, msg, time.time_ns()):
                        self._publish(sample)
                        self.counts['estimates'] += 1
                    self._check_fusion(time.time_ns())
                return
            stamp, now = stamp_ns(msg), time.time_ns()
            if (kind == 'info' and msg.lost and
                    stamp > max(self.gate.last_stamp, self.gate.last_result_stamp) and
                    -50_000_000 <= now-stamp <= 1_500_000_000):
                # Pause before the matching pose arrives; the same estimator keeps running.
                self.gate.degrade('visual estimator reported lost tracking', now, stamp=stamp)
            if self.gate.state == 'lost':
                return
            stamp = stamp_ns(msg)
            self.results[kind][stamp] = msg
            while len(self.results[kind]) > 20:
                self.results[kind].popitem(last=False)
            if all(stamp in values for values in self.results.values()):
                pose, info = (self.results[k].pop(stamp) for k in ('pose', 'info'))
                now = time.time_ns()
                sample = self.gate.accept(pose, info, now) if self._imu_frame_ready(stamp, now) else None
                if sample:
                    self._publish(sample)
                    self.counts['estimates'] += 1
                elif (self.gate.state == 'recovering' and self.gate.last_pose is not None and
                        stamp > self.gate.last_pose[0]):
                    # Capture-stamped barrier: the map pauses before any later cloud.
                    self._publish(make_sample('recovering', stamp, self.segment,
                                              self.calibration['clock_id'], reason=self.gate.reason))

    def _watchdog(self):
        with self.lock:
            if self.stop_event.is_set():
                return
            self._drain_imu()
            if self.process is not None and self.process.poll() is not None:
                self.gate.fault('RGB-D estimator exited', 'process')
            if not (getattr(self.gate, "last_input_received", 0) or self.gate.last_input) and time.monotonic() - self.started > 10:
                self.gate.fault('waiting for calibrated RGB-D inputs timed out', 'input')
            self._check_initialization(time.monotonic())
            if self._fusing():
                now = time.time_ns()
                self.fusion.poll_geometry(now)
                for sample in self.observations.drain(now):
                    self._publish(sample)
                    self.counts['estimates'] += 1
                if self.gate.state == 'lost':
                    self.fusion.fault(self.gate.reason)
                self.fusion.status(now)
                self._check_fusion(now)
                if self._fusing():
                    # A bounded pending observation has no final verdict yet.
                    # A premature pause would clear SLAM's matching cloud.
                    if self.fusion.state != 'tracking' and self.observations.geometry_capture is None:
                        self._publish(self.fusion.watchdog_output(now))
                    return
            if self.imu is not None and getattr(self, 'fusion', None) is None:
                if self.imu.status(time.time_ns()) == 'lost':
                    self.gate.fault('IMU assistance: ' + str(self.imu.reason), 'imu')
                elif self.imu.state == 'initializing' and time.monotonic() - self.started > 10:
                    self.gate.fault('waiting for usable camera IMU attitude timed out', 'imu')
            if self.gate.status(time.time_ns()) != 'tracking':
                # A pause carries native capture evidence, not the watchdog's
                # receipt time; transport latency must not expire SLAM's segment.
                stamp = self.gate.last_result_stamp or self.gate.last_stamp or time.time_ns()
                self._publish(make_sample(self.gate.state, stamp, self.segment,
                                          self.calibration['clock_id'], reason=self.gate.reason))

    def _time_metadata(self):
        context=getattr(self,'published_context',None)
        if context is None:return {}
        result=dict(observation=context.to_dict())
        if context.time_basis=='receipt_estimated':
            from .spatial_contract import input_declaration
            result['input_metadata']=[input_declaration(self.topic,'pose',self.calibration['frame_id'],context)]
        return result

    def info(self):
        with self.lock:
            self._drain_imu()
            now = time.time_ns()
            if self.fusion:
                self.fusion.status(now)
                self._check_fusion(now)
            self._check_initialization(time.monotonic())
            fusing = self._fusing()
            return {'state': 'running', 'status': self.fusion.state if fusing else self.gate.status(now),
                    'reason': self.fusion.reason if fusing else self.gate.reason, 'segment_id': self.segment,
                    'pose_reference': 'camera_optical', **self.counts,
                    **self._time_metadata(),
                    'last_measurement_ns': self.gate.last_stamp,
                    'visual_rejection': self.gate.rejection,
                    'loss_kind': self.gate.loss_kind,
                    'input_rejections': {'count': self.counts['rejected_inputs'],
                                         'last_reason': self.input_rejection},
                    'fallback': deepcopy(self.fallback),
                    'mode': self._mode_info(),
                    'initialization': self._initialization_info(time.monotonic()),
                    'auxiliary_recovery': self._auxiliary_info(now, time.monotonic()),
                    'assistance_unavailable': self.assistance_unavailable,
                    'imu_assistance': self.imu.info(time.time_ns()) if self.imu else {'enabled': False},
                    'inertial_association': self.rotation_hints.info() if self.rotation_hints else {'enabled': False},
                    'fusion': self.fusion.info(time.time_ns(), history=True) if self.fusion else {'enabled': False},
                    'registration': deepcopy(self.last_registration),
                    'native_reports': self.native_reports,
                    'timing': dict(end_to_end=latency_summary(self.processing_ages),
                                   **{k:latency_summary(v) for k,v in self.stage_times.items()}),
                    'input_delivery': [lease.status() for lease in self.input_leases],
                    'native_join': dict(self.observations.counts, pending=len(self.observations.frames))
                                   if self.observations is not None else None,
                    'body_mount': self.mounts.info(time.time_ns()) if self.mounts else {'status': 'undeclared'},
                    'estimator_log': str(Path(self.directory.name) / 'estimator.log')}

    def close(self, *, report_idle=True):
        self.stop_event.set()
        if hasattr(self, 'probe_ack'):
            self.probe_ack.set()
        for lease in getattr(self, "input_leases", ()):
            lease.close()
        if self.spin_worker is not None:
            self.spin_worker.join(timeout=5)
            if self.spin_worker.is_alive():
                raise RuntimeError('RGB-D callback worker did not stop')
        if self.worker is not None:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                raise RuntimeError('RGB-D decoding worker did not stop')
        if getattr(self, 'geometry_worker', None) is not None and not self.geometry_worker.close(5):
            raise RuntimeError('RGB-D geometry worker did not stop')
        if self.process is not None:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                if self.process.poll() is not None:
                    break
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    self.process.wait(timeout=1)
                    break
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    continue
            self.process.wait(timeout=1)
        try:
            if report_idle and self.status_publisher is not None:
                self._publish_status(make_sample('idle', time.time_ns(), self.segment,
                    self.calibration['clock_id'], reason='RGB-D odometry stopped'), time.time_ns())
        finally:
            dispose_node(self.executor, self.node, label=self.segment)
            if self.owns_executor:
                self.executor.shutdown(timeout_sec=5)
            if self.log is not None:
                self.log.close()
            self.directory.cleanup()
