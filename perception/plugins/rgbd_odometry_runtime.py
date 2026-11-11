"""One supervised RTAB-Map estimator per calibrated RGB-D card instance."""
from collections import OrderedDict
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

    def input(self, stamp):
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
        if self.state == 'lost' or stamp_ns(msg) != stamp_ns(info):
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


class RgbdSession:
    def __init__(self, calibration, executor, topic):
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from nav_msgs.msg import Odometry
        from rtabmap_msgs.msg import RGBDImage, OdomInfo
        self.calibration, self.executor, self.topic = calibration, executor, topic
        self.segment = 'rgbd_' + uuid.uuid4().hex[:12]
        self.gate = EstimateGate(self.segment, calibration['frame_id'], calibration['clock_id'])
        self.lock = threading.RLock()
        self.pairs, self.pending = PairBuffer(), queue.Queue(maxsize=1)
        self.results = {'pose': OrderedDict(), 'info': OrderedDict()}
        self.stop_event = threading.Event()
        self.process = self.worker = None
        self.directory = tempfile.TemporaryDirectory(prefix='motus-rgbd-')
        self.log = None
        self.counts = dict(pairs=0, estimates=0, dropped_pairs=0)
        self.started = time.monotonic()
        self.node = Node(self.segment)
        try:
            self.publisher = self.node.create_publisher(String, topic, qos_profile_sensor_data)
            self.rgbd = self.node.create_publisher(RGBDImage, '/' + self.segment + '/rgbd_image', qos_profile_sensor_data)
            for kind in ('color', 'depth'):
                self.node.create_subscription(CompressedImage, calibration[kind + '_topic'],
                    lambda m, k=kind: self._image(k, m), qos_profile_sensor_data)
            self.node.create_subscription(Odometry, '/' + self.segment + '/visual',
                                          lambda m: self._result('pose', m), qos_profile_sensor_data)
            self.node.create_subscription(OdomInfo, '/' + self.segment + '/diagnostic',
                                          lambda m: self._result('info', m), qos_profile_sensor_data)
            self.node.create_timer(.2, self._watchdog)
            self.executor.add_node(self.node)
            params = {'frame_id': calibration['frame_id'], 'odom_frame_id': self.segment + '_odom',
                      'publish_tf': False, 'subscribe_rgbd': True, 'qos': 2,
                      'topic_queue_size': 2, 'sync_queue_size': 2,
                      'Odom/ResetCountdown': '0', 'Vis/MaxFeatures': '1000'}
            path = Path(self.directory.name) / 'parameters.json'
            path.write_text(json.dumps({'/**': {'ros__parameters': params}}))
            self.log = (Path(self.directory.name) / 'estimator.log').open('ab', buffering=0)
            cmd = [str(require_runtime()), '--ros-args', '--params-file', str(path),
                   '-r', '__ns:=/' + self.segment, '-r', '__node:=rgbd_odometry',
                   '-r', 'odom:=/' + self.segment + '/visual',
                   '-r', 'odom_info_lite:=/' + self.segment + '/diagnostic']
            self.process = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
            self.worker = threading.Thread(target=self._work, name=self.segment, daemon=True)
            self.worker.start()
        except Exception:
            self.close()
            raise

    def _image(self, kind, msg):
        if self.stop_event.is_set():
            return
        with self.lock:
            if self.gate.state == 'lost':
                return
            try:
                if msg.header.frame_id != self.calibration['frame_id']:
                    raise ValueError('image optical frame differs from calibration')
                pair = self.pairs.push(kind, stamp_ns(msg), msg, time.time_ns())
                if pair:
                    self.gate.input(pair[1][0])
                    if self.pending.full():
                        self.pending.get_nowait()
                        self.counts['dropped_pairs'] += 1
                    self.pending.put_nowait(pair)
            except (ValueError, queue.Full) as error:
                self.gate.fault(error)

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
                image = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(bytes(depth.data), 640 * 480 * 2 + 1)
                if (image is None or image.shape != (480, 640, 3) or len(raw) != 640 * 480 * 2 or
                        not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
                    raise ValueError('RGB-D payload does not match declared 640x480 geometry')
                z16 = np.frombuffer(raw, '<u2').reshape(480, 640)
                if np.count_nonzero(z16) < 100:
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
                out.depth_compressed.data = png.tobytes()
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
                    if self.gate.state != 'lost' and not self.stop_event.is_set():
                        self.rgbd.publish(out)
                        self.counts['pairs'] += 1
            except Exception as error:
                with self.lock:
                    self.gate.fault(error)

    def _publish(self, sample):
        from std_msgs.msg import String
        self.publisher.publish(String(data=json.dumps(sample, allow_nan=False)))

    def _result(self, kind, msg):
        with self.lock:
            if self.stop_event.is_set():
                return
            stamp = stamp_ns(msg)
            self.results[kind][stamp] = msg
            while len(self.results[kind]) > 20:
                self.results[kind].popitem(last=False)
            if all(stamp in values for values in self.results.values()):
                pose, info = (self.results[k].pop(stamp) for k in ('pose', 'info'))
                sample = self.gate.accept(pose, info, time.time_ns())
                if sample:
                    self._publish(sample)
                    self.counts['estimates'] += 1

    def _watchdog(self):
        with self.lock:
            if self.stop_event.is_set():
                return
            if self.process is not None and self.process.poll() is not None:
                self.gate.fault('RGB-D estimator exited')
            if not self.gate.last_input and time.monotonic() - self.started > 10:
                self.gate.fault('waiting for calibrated RGB-D inputs timed out')
            if self.gate.status(time.time_ns()) != 'tracking':
                self._publish(make_sample(self.gate.state, time.time_ns(), self.segment,
                                          self.calibration['clock_id'], reason=self.gate.reason))

    def info(self):
        with self.lock:
            return {'state': 'running', 'status': self.gate.status(time.time_ns()),
                    'reason': self.gate.reason, 'segment_id': self.segment,
                    'pose_reference': 'camera_optical', **self.counts,
                    'last_measurement_ns': self.gate.last_stamp,
                    'estimator_log': str(Path(self.directory.name) / 'estimator.log')}

    def close(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                raise RuntimeError('RGB-D decoding worker did not stop')
        if self.process is not None:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                if self.process.poll() is not None:
                    break
                os.killpg(self.process.pid, sig)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    continue
            self.process.wait(timeout=1)
        dispose_node(self.executor, self.node, label=self.segment)
        if self.log is not None:
            self.log.close()
        self.directory.cleanup()
