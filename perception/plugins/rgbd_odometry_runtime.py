"""One supervised RTAB-Map estimator per calibrated RGB-D card instance."""
from collections import OrderedDict, deque
from array import array
from copy import deepcopy
import json
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
from .rgbd_fusion import ShortTermFusion, observation_covariance, pose_values
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


class EstimateGate:
    """Loss ends a segment; only a newly started estimator can recover it."""
    def __init__(self, segment, frame, clock):
        self.segment, self.frame, self.clock = segment, frame, clock
        self.state, self.reason = 'initializing', None
        self.last_input = self.last_stamp = 0
        self.good = 0

    def input(self, stamp, *, now_ns=None):
        self.status(stamp if now_ns is None else now_ns)
        if self.state != 'lost':
            self.last_input = stamp

    def fault(self, reason):
        self.state, self.reason = 'lost', self.reason or str(reason)

    def status(self, now):
        if self.last_input and now - self.last_input > 1_500_000_000:
            self.fault('RGB-D input stale; restart to begin a new segment')
        if self.last_stamp and now - self.last_stamp > 1_500_000_000:
            self.fault('visual estimate stale; restart to begin a new segment')
        return self.state

    def accept(self, msg, info, now):
        if self.status(now) == 'lost' or stamp_ns(msg) != stamp_ns(info):
            return None
        stamp = stamp_ns(msg)
        try:
            if info.lost or info.inliers < 10:
                raise ValueError('visual tracking lost or insufficient inliers')
            if (msg.header.frame_id != self.segment + '_odom' or
                    msg.child_frame_id != self.frame or stamp <= self.last_stamp or
                    not -50_000_000 <= now - stamp <= 1_500_000_000):
                raise ValueError('invalid visual frames or capture time')
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
        except (ValueError, TypeError, AttributeError) as error:
            if self.state == 'tracking':
                self.fault(error)
            return None
        self.last_stamp = stamp
        self.good += 1
        if self.good < 3:
            return None
        self.state = 'tracking'
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
        self.frames[stamp] = dict(color=color, depth=depth, deadline=now+250_000_000)
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
            outputs.append(self.fusion.observe_frame(stamp, entry['color'], entry['depth'], visual, now, reason=reason))
        return outputs


class RgbdSession:
    def __init__(self, calibration, executor, topic):
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from nav_msgs.msg import Odometry
        from rtabmap_msgs.msg import RGBDImage, OdomInfo
        self.calibration, self.executor, self.topic = calibration, executor, topic
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
        self.results = {'pose': OrderedDict(), 'info': OrderedDict()}
        self.registration_by_stamp = OrderedDict()
        self.last_registration = None
        self.native_reports = 0
        self.stop_event = threading.Event()
        self.process = self.worker = None
        self.spin_worker = None
        self.owns_executor = False
        self.directory = tempfile.TemporaryDirectory(prefix='motus-rgbd-')
        self.log = None
        self.counts = dict(pairs=0, estimates=0, dropped_pairs=0)
        self.started = time.monotonic()
        self.node = Node(self.segment)
        try:
            if self.imu is not None:
                from rclpy.executors import SingleThreadedExecutor
                self.executor = SingleThreadedExecutor(context=self.node.context)
                self.owns_executor = True
            self.publisher = self.node.create_publisher(String, topic, qos_profile_sensor_data)
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
            self.worker = threading.Thread(target=self._work, name=self.segment, daemon=True)
            self.worker.start()
            if self.owns_executor:
                self.spin_worker = threading.Thread(target=self._spin, name=self.segment+'_ros', daemon=True)
                self.spin_worker.start()
        except Exception:
            self.close()
            raise

    def _spin(self):
        try:
            while not self.stop_event.is_set():
                self.executor.spin_once(timeout_sec=.05)
        except Exception as error:
            if not self.stop_event.is_set():
                with self.lock:
                    self.gate.fault('RGB-D callback executor: ' + str(error))
                    if self.fusion is not None:
                        self.fusion.fault(self.gate.reason)

    def _image(self, kind, msg):
        if self.stop_event.is_set():
            return
        with self.lock:
            self._drain_imu()
            if self.gate.status(time.time_ns()) == 'lost':
                return
            try:
                if msg.header.frame_id != self.calibration['frame_id']:
                    raise ValueError('image optical frame differs from calibration')
                pair = self.pairs.push(kind, stamp_ns(msg), msg, time.time_ns())
                if pair:
                    self.gate.input(pair[1][0], now_ns=time.time_ns())
                    if self.pending.full():
                        self.pending.get_nowait()
                        self.counts['dropped_pairs'] += 1
                    self.pending.put_nowait(pair)
            except (ValueError, queue.Full) as error:
                self.gate.fault(error)

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
            self.gate.fault(self.imu.reason)
            if self.fusion is not None:
                self.fusion.fault(self.imu.reason)
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
                    self.fusion.fault(error)
                self.gate.fault('IMU assistance: ' + str(error))

    def _imu_frame_ready(self, stamp, now):
        if self.gate.status(now) == 'lost':
            return False
        if self.imu is None:
            return True
        covered = self.imu.covers(stamp, now)
        if self.imu.state == 'lost':
            self.gate.fault('IMU assistance: ' + str(self.imu.reason))
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
                if self.fusion is None and np.count_nonzero(z16) < 100:
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
                        if self.observations is not None:
                            for sample in self.observations.add_frame(stamp_ns(depth), image, z16, time.time_ns()):
                                self._publish(sample)
                                self.counts['estimates'] += 1
                        if self.gate.state == 'lost':
                            continue
                        self.rgbd.publish(out)
                        self.counts['pairs'] += 1
            except Exception as error:
                with self.lock:
                    self.gate.fault(error)

    def _publish(self, sample):
        from std_msgs.msg import String
        self._drain_imu()
        now = time.time_ns()
        if self.imu and self.imu.status(now) == 'lost':
            self.gate.fault('IMU assistance: ' + str(self.imu.reason))
            if self.fusion is not None:
                self.fusion.fault(self.gate.reason)
        if self.gate.state == 'lost' or (self.fusion is not None and self.fusion.state == 'lost'):
            stamp = sample['vendor']['motus_slam']['stamp_ns']
            reason = self.fusion.reason if self.fusion is not None else self.gate.reason
            sample = make_sample('lost', stamp, self.segment, self.calibration['clock_id'], reason=reason)
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
            if getattr(self, 'fusion', None) is not None:
                if self.gate.state != 'lost':
                    for sample in self.observations.result(kind, msg, time.time_ns()):
                        self._publish(sample)
                        self.counts['estimates'] += 1
                return
            if kind == 'info' and msg.lost and self.gate.state == 'tracking':
                self.gate.fault('visual estimator reported lost tracking')
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
                self.gate.fault('RGB-D estimator exited')
            if not self.gate.last_input and time.monotonic() - self.started > 10:
                self.gate.fault('waiting for calibrated RGB-D inputs timed out')
            if getattr(self, 'fusion', None) is not None:
                now = time.time_ns()
                for sample in self.observations.drain(now):
                    self._publish(sample)
                    self.counts['estimates'] += 1
                if self.gate.state == 'lost':
                    self.fusion.fault(self.gate.reason)
                if self.fusion.status(now) == 'lost':
                    self.gate.fault(self.fusion.reason)
                elif self.fusion.filter is None and time.monotonic()-self.started > 15:
                    self.fusion.fault('low-motion fusion initialization timed out; restart required')
                    self.gate.fault(self.fusion.reason)
                if self.fusion.state != 'tracking':
                    self._publish(self.fusion.output(self.fusion.last_frame or now))
                return
            if self.imu is not None:
                if self.imu.status(time.time_ns()) == 'lost':
                    self.gate.fault('IMU assistance: ' + str(self.imu.reason))
                elif self.imu.state == 'initializing' and time.monotonic() - self.started > 10:
                    self.gate.fault('waiting for usable camera IMU attitude timed out')
            if self.gate.status(time.time_ns()) != 'tracking':
                self._publish(make_sample(self.gate.state, time.time_ns(), self.segment,
                                          self.calibration['clock_id'], reason=self.gate.reason))

    def info(self):
        with self.lock:
            self._drain_imu()
            return {'state': 'running', 'status': (self.fusion.status(time.time_ns()) if self.fusion else
                                                  self.gate.status(time.time_ns())),
                    'reason': self.fusion.reason if self.fusion else self.gate.reason, 'segment_id': self.segment,
                    'pose_reference': 'camera_optical', **self.counts,
                    'last_measurement_ns': self.gate.last_stamp,
                    'imu_assistance': self.imu.info(time.time_ns()) if self.imu else {'enabled': False},
                    'fusion': self.fusion.info(time.time_ns()) if self.fusion else {'enabled': False},
                    'registration': deepcopy(self.last_registration),
                    'native_reports': self.native_reports,
                    'body_mount': self.mounts.info(time.time_ns()) if self.mounts else {'status': 'undeclared'},
                    'estimator_log': str(Path(self.directory.name) / 'estimator.log')}

    def close(self):
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
        dispose_node(self.executor, self.node, label=self.segment)
        if self.owns_executor:
            self.executor.shutdown(timeout_sec=5)
        if self.log is not None:
            self.log.close()
        self.directory.cleanup()
