"""One supervised RTAB-Map estimator per calibrated RGB-D card instance."""
from collections import OrderedDict, deque
from array import array
from copy import deepcopy
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

from .rgbd_odometry_contract import PairBuffer, make_sample
from .rgbd_imu import ImuAttitude
from .rgbd_fusion import ShortTermFusion, log_rotation, observation_covariance, pose_values
from .rgbd_body_mount import CaptureMounts
from utils.ros_lifecycle import dispose_node


def require_runtime():
    from ament_index_python.packages import get_package_prefix
    path = Path(get_package_prefix('rtabmap_odom')) / 'lib/rtabmap_odom/rgbd_odometry'
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError('RTAB-Map RGB-D odometry runtime unavailable')
    return path


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


RECOVERY_NS = 1_500_000_000
# Fused output differs from the native visual pose by ESKF corrections. A
# visual-only fallback may stay in the segment only within this (unvalidated)
# engineering bound at the last published fused capture.
FALLBACK_OFFSET_M, FALLBACK_OFFSET_RAD = .02, np.deg2rad(2.)
# Engineering bound for a head/body camera, not a validated motion model: a
# resumed estimate farther than this from the last published pose is treated as
# a native reset/relocalization and must start a new segment.
RESUME_TRANSLATION_M, RESUME_SPEED_MPS = .05, 1.
RESUME_ROTATION_RAD, RESUME_RATE_RADPS = np.deg2rad(5.), np.deg2rad(90.)


class EstimateGate:
    """Hard errors end a segment; transient visual rejections only pause output.

    `recovering` keeps feeding the same native estimator but publishes no pose.
    Three consecutive valid estimates continuous with the last published pose
    resume tracking; persistent degradation or a discontinuity ends the segment.
    """
    def __init__(self, segment, frame, clock):
        self.segment, self.frame, self.clock = segment, frame, clock
        self.state, self.reason = 'initializing', None
        self.last_input = self.last_stamp = 0
        self.good = 0
        self.recover_by = None
        self.last_pose = None
        self.loss_kind = None
        self.rejection = None

    def input(self, stamp, *, now_ns=None):
        self.status(stamp if now_ns is None else now_ns)
        if self.state != 'lost':
            self.last_input = stamp

    def fault(self, reason, kind='other'):
        if self.state != 'lost':
            self.loss_kind = kind
        self.state, self.reason = 'lost', self.reason if self.state == 'lost' else str(reason)

    def degrade(self, reason, now):
        """Reject one estimate. Tracking pauses; evidence must be rebuilt."""
        self.rejection = str(reason)
        if self.state == 'lost':
            return
        self.good = 0
        if self.state in ('tracking', 'recovering'):
            if self.recover_by is None:
                self.recover_by = now + RECOVERY_NS
            self.state, self.reason = 'recovering', 'visual estimate rejected: ' + str(reason)

    def status(self, now):
        if self.last_input and now - self.last_input > 1_500_000_000:
            self.fault('RGB-D input stale; restart to begin a new segment', 'input')
        if self.last_stamp and now - self.last_stamp > 1_500_000_000:
            self.fault('visual estimate stale; restart to begin a new segment', 'visual')
        if self.state == 'recovering' and self.recover_by is not None and now > self.recover_by:
            self.fault('visual degradation persisted beyond 1.5 s; restart to begin a new segment', 'visual')
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
        if stamp <= self.last_stamp or now - stamp > 1_500_000_000:
            self.rejection = 'stale or out-of-order visual estimate'  # never counted as evidence
            return None
        try:
            if info.lost or info.inliers < 10:
                raise ValueError('visual tracking lost or insufficient inliers')
            covariance = np.asarray(msg.pose.covariance, dtype=float).reshape(6, 6)
            if (not np.isfinite(covariance).all() or
                    not np.allclose(covariance, covariance.T, atol=1e-8) or
                    np.any(np.diag(covariance) < 0) or np.any(np.diag(covariance)[:3] > .04)):
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
        if self.state != 'tracking' and not self._continuous(stamp, position, rotation):
            self.fault('visual estimate resumed discontinuously; restart to begin a new segment', 'visual')
            return None
        self.rejection = None
        self.last_stamp = stamp
        self.good += 1
        if self.good < 3:
            return None
        self.state, self.reason, self.recover_by = 'tracking', None, None
        self.last_pose = (stamp, position, rotation)
        return sample


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
    if optical:
        params.update({'Vis/CorFlowWinSize': '16', 'Vis/CorFlowMaxLevel': '3',
                       'Vis/CorFlowIterations': '30'})
    if calibration.get('imu_topic') and not calibration.get('short_term_compensation'):
        params.update(wait_imu_to_init=True, **{'Odom/GuessSmoothingDelay': '0'})
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
            'reject_reason': depth.get('reason')
        }
    }


class CaptureObservations:
    """Bounded exact-stamp joins with capture-order delivery and late-result discard."""
    def __init__(self, fusion):
        self.fusion = fusion
        self.frames = OrderedDict()
        self.last_added = 0

    def add_frame(self, stamp, color, depth, now):
        outputs = self.drain(now)
        if stamp <= self.last_added or len(self.frames) >= 8:
            self.fusion.fault('fusion capture queue repeated, reset or overflowed')
            return outputs
        self.last_added = stamp
        self.frames[stamp] = dict(color=color, depth=depth, added=now, deadline=now+250_000_000)
        return outputs

    def result(self, kind, msg, now):
        stamp = stamp_ns(msg)
        if stamp in self.frames and now <= self.frames[stamp]['deadline']:
            self.frames[stamp][kind] = msg
        return self.drain(now)

    def _visual(self, stamp, entry):
        msg, info = entry.get('pose'), entry.get('info')
        if msg is None or info is None or info.lost or info.inliers < 10:
            return None, 'visual pose missing, lost or insufficient inliers'
        try:
            if msg.header.frame_id != self.fusion.segment+'_odom' or msg.child_frame_id != self.fusion.frame:
                self.fusion.fault('native visual frame/segment changed; restart required')
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
        while self.frames:
            stamp, entry = next(iter(self.frames.items()))
            ready = ('pose' in entry and 'info' in entry) or getattr(entry.get('info'), 'lost', False)
            coverage = bool(self.fusion.samples) and self.fusion.samples[-1][0] >= stamp
            if (not ready or not coverage) and now < entry['deadline']:
                break
            self.frames.pop(stamp)
            visual, reason = self._visual(stamp, entry)
            outputs.append(self.fusion.observe_frame(stamp, entry['color'], entry['depth'], visual, now,
                                                     reason=reason, arrival_ns=entry['added']))
        return outputs


class RgbdSession:
    def __init__(self, calibration, executor, topic, *, publish_output=True):
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from nav_msgs.msg import Odometry
        from rtabmap_msgs.msg import RGBDImage, OdomInfo
        self.calibration, self.executor, self.topic = calibration, executor, topic
        self.publish_output = publish_output
        self.segment = 'rgbd_' + uuid.uuid4().hex[:12]
        self.gate = EstimateGate(self.segment, calibration['frame_id'], calibration['clock_id'])
        self.fusion = (ShortTermFusion(self.segment, calibration['frame_id'], calibration['clock_id'],
                       calibration['K'], calibration.get('depth_geometry', False))
                       if calibration.get('short_term_compensation') else None)
        self.observations = CaptureObservations(self.fusion) if self.fusion else None
        self.mounts = (CaptureMounts(calibration['body_mount'], calibration['frame_id'], calibration['clock_id'])
                       if calibration.get('body_mount') else None)
        self.imu = (self.fusion.imu if self.fusion else
                    ImuAttitude(calibration['frame_id'], calibration['clock_id']) if calibration.get('imu_topic') else None)
        self.lock = threading.RLock()
        self.imu_queue_lock = threading.Lock()
        self.imu_pending, self.imu_overflow = deque(), False
        self.pairs, self.pending = PairBuffer(), queue.Queue(maxsize=1)
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
        self.owns_executor = False
        self.directory = tempfile.TemporaryDirectory(prefix='motus-rgbd-')
        self.log = None
        self.counts = dict(pairs=0, estimates=0, dropped_pairs=0, rejected_inputs=0)
        self.input_rejection = None
        self.fallback = None
        self.started = time.monotonic()
        self.node = Node(self.segment)
        try:
            if self.imu is not None:
                from rclpy.executors import SingleThreadedExecutor
                self.executor = SingleThreadedExecutor(context=self.node.context)
                self.owns_executor = True
            self.publisher = self.node.create_publisher(String, topic, qos_profile_sensor_data)
            self.status_publisher = self.node.create_publisher(
                String, topic.removesuffix('/state/odom') + '/status', qos_profile_sensor_data)
            self.rgbd = self.node.create_publisher(RGBDImage, '/' + self.segment + '/rgbd_image', qos_profile_sensor_data)
            for kind in ('color', 'depth'):
                self.node.create_subscription(CompressedImage, calibration[kind + '_topic'],
                    lambda m, k=kind: self._image(k, m), qos_profile_sensor_data)
            self.node.create_subscription(Odometry, '/' + self.segment + '/visual',
                                          lambda m: self._result('pose', m), qos_profile_sensor_data)
            self.node.create_subscription(OdomInfo, '/' + self.segment + '/diagnostic',
                                          lambda m: self._result('info', m), qos_profile_sensor_data)
            if self.imu is not None:
                from sensor_msgs.msg import Imu
                from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
                if self.fusion is None:
                    self.imu_pub = self.node.create_publisher(Imu, '/' + self.segment + '/imu', qos_profile_sensor_data)
                # Preserve capture intervals while a visual correction occupies
                # the callback group; admission still rejects samples over 500 ms old.
                imu_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
                self.imu_callback_group = MutuallyExclusiveCallbackGroup()
                self.node.create_subscription(String, calibration['imu_topic'], self._enqueue_imu, imu_qos,
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
            self.process = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
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

    def _image(self, kind, msg):
        if self.stop_event.is_set():
            return
        with self.lock:
            self._drain_imu()
            try:
                if msg.header.frame_id != self.calibration['frame_id']:
                    raise ValueError('image optical frame differs from calibration')
                now, stamp = time.time_ns(), stamp_ns(msg)
                if stamp > self.latest_inputs.get(kind, 0) and -50_000_000 <= now-stamp <= 1_500_000_000:
                    self.latest_inputs[kind] = stamp
                if self.gate.status(now) == 'lost':
                    return
                try:
                    pair = self.pairs.push(kind, stamp_ns(msg), msg, time.time_ns())
                except ValueError as error:
                    # One late/repeated capture is dropped, never paired or
                    # used; persistent loss of input expires the segment.
                    self.counts['rejected_inputs'] += 1
                    self.input_rejection = f'{kind}: {error}'
                    return
                if pair:
                    self.gate.input(pair[1][0], now_ns=time.time_ns())
                    if self.pending.full():
                        self.pending.get_nowait()
                        self.counts['dropped_pairs'] += 1
                    self.pending.put_nowait(pair)
            except (ValueError, queue.Full) as error:
                self.gate.fault(error, 'frame')

    def has_fresh_inputs(self, now_ns):
        with self.lock:
            return all(kind in self.latest_inputs and
                       -50_000_000 <= now_ns-self.latest_inputs[kind] <= 1_500_000_000
                       for kind in ('color', 'depth'))

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
            else:
                self.imu_pending.append(msg)

    def _drain_imu(self):
        # Caller owns the estimator lock; ingress only touches its own queue.
        if not hasattr(self, 'imu_pending'):
            return
        with self.imu_queue_lock:
            pending = list(self.imu_pending)
            self.imu_pending.clear()
            overflow = self.imu_overflow
        if overflow:
            self.imu.fault('IMU ingress overflow; restart required')
            if self.fusion is not None:
                # The native estimator never consumed this IMU: only the ESKF stops.
                self.fusion.fault(self.imu.reason)
            else:
                self.gate.fault(self.imu.reason, 'imu')
            return
        for msg in pending:
            if self.stop_event.is_set():
                return
            self._imu(msg)

    def _imu(self, msg):
        from sensor_msgs.msg import Imu
        with self.lock:
            if self.stop_event.is_set() or self.gate.state == 'lost':
                return
            try:
                if getattr(self, 'fusion', None) is not None:
                    if self.fusion.state != 'lost':
                        self.fusion.observe_imu(json.loads(msg.data), time.time_ns())
                    return
                sample = self.imu.observe(json.loads(msg.data), time.time_ns())
                if sample is None:
                    return
                out = Imu()
                out.header.frame_id = self.calibration['frame_id']
                out.header.stamp.sec, out.header.stamp.nanosec = divmod(sample['stamp_ns'], 10**9)
                (out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w) = sample['quaternion']
                out.orientation_covariance = [0.] * 9  # estimated orientation; covariance unknown
                (out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z) = sample['angular_velocity']
                (out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z) = sample['linear_acceleration']
                self.imu_pub.publish(out)
            except (ValueError, TypeError, AttributeError, RuntimeError) as error:
                self.imu.fault(error)
                if getattr(self, 'fusion', None) is not None:
                    self.fusion.fault(error)  # visual-only fallback decides continuity
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
                with self.lock:
                    self._drain_imu()
                    if self.fusion is None and not self._imu_frame_ready(stamp_ns(depth), time.time_ns()):
                        self.counts['dropped_pairs'] += 1
                        continue
                image = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(bytes(depth.data), 640 * 480 * 2 + 1)
                if (image is None or image.shape != (480, 640, 3) or len(raw) != 640 * 480 * 2 or
                        not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
                    raise ValueError('RGB-D payload does not match declared 640x480 geometry')
                z16 = np.frombuffer(raw, '<u2').reshape(480, 640)
                if not self._fusing() and np.count_nonzero(z16) < 100:
                    raise ValueError('RGB-D depth has insufficient measured pixels')
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
                    info.width, info.height = 640, 480
                    info.k, info.d, info.distortion_model = k, [0.] * 5, 'plumb_bob'
                    info.r = np.eye(3).ravel().tolist()
                    info.p = [k[0], 0., k[2], 0., 0., k[4], k[5], 0., 0., 0., 1., 0.]
                    setattr(out, field, info)
                with self.lock:
                    self._drain_imu()
                    ready = (self.gate.state != 'lost' if self.fusion is not None else
                             self._imu_frame_ready(stamp_ns(depth), time.time_ns()))
                    if ready and not self.stop_event.is_set():
                        if self._fusing():
                            for sample in self.observations.add_frame(stamp_ns(depth), image, z16, time.time_ns()):
                                self._publish(sample)
                                self.counts['estimates'] += 1
                            self._check_fusion(time.time_ns())
                        if self.gate.state == 'lost':
                            continue
                        self.rgbd.publish(out)
                        self.counts['pairs'] += 1
            except Exception as error:
                with self.lock:
                    self.gate.fault(error)

    def _fusing(self):
        return getattr(self, 'fusion', None) is not None and getattr(self, 'fallback', None) is None

    def _check_fusion(self, now):
        if (getattr(self, 'fusion', None) is not None and getattr(self, 'fallback', None) is None and
                self.fusion.state == 'lost' and self.gate.state != 'lost'):
            self._enter_fallback(now)

    def _enter_fallback(self, now):
        """ESKF stopped: continue with the independent native visual estimate.

        The native estimator never received IMU in this mode. Its poses share
        this segment's odom frame, but fused output differed from them by the
        corrections. Keep the segment only if that offset at the last published
        fused capture is within the continuity bound; otherwise restart.
        """
        reason = str(self.fusion.reason)
        offset = self.fusion.tracking_offset()
        if offset is not None and (offset[0] > FALLBACK_OFFSET_M or offset[1] > FALLBACK_OFFSET_RAD):
            self.gate.fault(f'ESKF lost ({reason}); visual-only fallback cannot prove coordinate '
                            f'continuity (fused/visual offset {offset[0]:.3f} m, '
                            f'{np.rad2deg(offset[1]):.1f} deg)', 'fusion')
            return
        self.fallback = dict(mode='visual_only', reason=reason, since_ns=now,
                             fused_visual_offset_m=None if offset is None else round(offset[0], 4),
                             fused_visual_offset_deg=None if offset is None else round(float(np.rad2deg(offset[1])), 2))
        self.observations.frames.clear()
        gate = self.gate
        gate.good = 0
        gate.reason = 'visual-only fallback after ESKF loss: ' + reason
        gate.last_stamp = max(gate.last_stamp, self.fusion.last_visual)
        if self.fusion.last_tracking is not None:
            # Something was published: resuming is a bounded recovery.
            stamp, fused, _ = self.fusion.last_tracking
            gate.last_pose = (stamp, *pose_values(fused))
            gate.state, gate.recover_by = 'recovering', now + RECOVERY_NS
        logging.getLogger(__name__).warning('RGB-D %s: %s', self.segment, gate.reason)

    def _label_fallback(self, sample):
        vendor = sample['vendor']
        vendor['fallback'] = 'visual_only'
        vendor['fusion'] = {'enabled': True, 'status': 'lost', 'reason': self.fallback['reason'],
                            'fallback': 'visual_only', 'mapping_allowed': sample['pose'] is not None}
        if vendor['motus_slam']['status'] == 'tracking':
            vendor['reason'] = 'visual-only fallback: ESKF ' + self.fallback['reason'][:80]
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
        if getattr(self, 'mounts', None) is not None:
            sample = self.mounts.attach(sample, time.time_ns())
        sample['vendor']['imu_assistance'] = self.imu.info(time.time_ns()) if self.imu else {'enabled': False}
        stamp = (sample['vendor'].get('motus_slam') or {}).get('stamp_ns')
        native = self.registration_by_stamp.pop(stamp, None)
        report = registration_report(
            sample, self.calibration.get('visual_frontend', 'f2m_feature'), native,
            depth_enabled=bool(self.fusion and self.fusion.geometry))
        sample['vendor']['registration'] = report
        self.last_registration = deepcopy(report)
        self.publisher.publish(String(data=json.dumps(sample, allow_nan=False)))
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
                       reason=vendor['reason'])
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
            if kind == 'info' and hasattr(self, 'registration_by_stamp'):
                self._remember_native(msg)
            if self._fusing():
                if self.gate.state != 'lost':
                    for sample in self.observations.result(kind, msg, time.time_ns()):
                        self._publish(sample)
                        self.counts['estimates'] += 1
                    self._check_fusion(time.time_ns())
                return
            if kind == 'info' and msg.lost and stamp_ns(msg) > self.gate.last_stamp:
                # Pause before the matching pose arrives; the same estimator keeps running.
                self.gate.degrade('visual estimator reported lost tracking', time.time_ns())
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

    def _watchdog(self):
        with self.lock:
            if self.stop_event.is_set():
                return
            self._drain_imu()
            if self.process is not None and self.process.poll() is not None:
                self.gate.fault('RGB-D estimator exited', 'process')
            if not self.gate.last_input and time.monotonic() - self.started > 10:
                self.gate.fault('waiting for calibrated RGB-D inputs timed out', 'input')
            if self._fusing():
                now = time.time_ns()
                for sample in self.observations.drain(now):
                    self._publish(sample)
                    self.counts['estimates'] += 1
                if self.gate.state == 'lost':
                    self.fusion.fault(self.gate.reason)
                if (self.fusion.status(now) != 'lost' and self.fusion.filter is None and
                        time.monotonic()-self.started > 15):
                    self.fusion.fault('low-motion fusion initialization timed out; restart required')
                self._check_fusion(now)
                if self._fusing():
                    if self.fusion.state != 'tracking':
                        self._publish(self.fusion.output(self.fusion.last_frame or now))
                    return
            if self.imu is not None and getattr(self, 'fusion', None) is None:
                if self.imu.status(time.time_ns()) == 'lost':
                    self.gate.fault('IMU assistance: ' + str(self.imu.reason), 'imu')
                elif self.imu.state == 'initializing' and time.monotonic() - self.started > 10:
                    self.gate.fault('waiting for usable camera IMU attitude timed out', 'imu')
            if self.gate.status(time.time_ns()) != 'tracking':
                self._publish(make_sample(self.gate.state, time.time_ns(), self.segment,
                                          self.calibration['clock_id'], reason=self.gate.reason))

    def info(self):
        with self.lock:
            self._drain_imu()
            now = time.time_ns()
            if self.fusion:
                self.fusion.status(now)
                self._check_fusion(now)
            fusing = self._fusing()
            return {'state': 'running', 'status': self.fusion.state if fusing else self.gate.status(now),
                    'reason': self.fusion.reason if fusing else self.gate.reason, 'segment_id': self.segment,
                    'pose_reference': 'camera_optical', **self.counts,
                    'last_measurement_ns': self.gate.last_stamp,
                    'visual_rejection': self.gate.rejection,
                    'input_rejections': {'count': self.counts['rejected_inputs'],
                                         'last_reason': self.input_rejection},
                    'fallback': deepcopy(self.fallback),
                    'imu_assistance': self.imu.info(time.time_ns()) if self.imu else {'enabled': False},
                    'fusion': self.fusion.info(time.time_ns(), history=True) if self.fusion else {'enabled': False},
                    'registration': deepcopy(self.last_registration),
                    'native_reports': self.native_reports,
                    'body_mount': self.mounts.info(time.time_ns()) if self.mounts else {'status': 'undeclared'},
                    'estimator_log': str(Path(self.directory.name) / 'estimator.log')}

    def close(self, *, report_idle=True):
        self.stop_event.set()
        if self.spin_worker is not None:
            self.spin_worker.join(timeout=5)
            if self.spin_worker.is_alive():
                raise RuntimeError('RGB-D callback worker did not stop')
        if self.worker is not None:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                raise RuntimeError('RGB-D decoding worker did not stop')
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
