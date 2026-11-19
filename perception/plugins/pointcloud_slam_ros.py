"""Native RTAB-Map gateway for admitted PointCloud2 and existing external VO."""
import json
import hashlib
import math
import threading
import time
import logging
import struct
import zlib
from collections import deque, OrderedDict
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid
from copy import deepcopy
from array import array
from concurrent.futures import Future, TimeoutError
from contextlib import nullcontext

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage
from std_msgs.msg import String, UInt8MultiArray

from .pointcloud_slam_rtabmap import RtabmapProcess
from .pointcloud_slam_watchdog import BackendProgress
from .pointcloud_slam_snapshot import rebuild as rebuild_snapshot
from .pointcloud_slam_preview import PREVIEW_TOPIC
from .pointcloud_slam_stream import SlamStream
from .pointcloud_slam_transforms import rigid, pose_matrix
from .pointcloud_slam_motion import ScanMotion, MotionPending


_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                  history=HistoryPolicy.KEEP_LAST, depth=2,
                  durability=DurabilityPolicy.VOLATILE)
log = logging.getLogger(__name__)


def discover_input_bindings(input_topics):
    """Resolve native publisher types, allowing one second for DDS discovery."""
    if (not isinstance(input_topics, list) or len(input_topics) not in (2, 3) or
            any(not isinstance(topic, str) or not topic.strip() for topic in input_topics) or
            len(set(input_topics)) != len(input_topics)):
        raise ValueError('two required native ROS topics and at most one optional RGB topic required')
    formats = {'sensor_msgs/msg/PointCloud2': 'sensor/pointcloud2',
               'std_msgs/msg/String': 'state/odom',
               'sensor_msgs/msg/CompressedImage': 'image/jpeg'}
    node = Node('pointcloud_slam_input_discovery')
    try:
        deadline = time.monotonic() + 1.
        while True:
            bindings = {}
            for topic in input_topics:
                # Core may monitor even a native cloud through a String subscriber.
                # Only publisher endpoints identify the connected producer's type.
                types = {endpoint.topic_type for endpoint in node.get_publishers_info_by_topic(topic)}
                if not types:
                    continue
                if len(types) != 1 or not types <= formats.keys():
                    raise ValueError('ambiguous or unsupported native ROS input type: ' + topic)
                bindings[topic] = formats[next(iter(types))]
            if len(bindings) == len(input_topics):
                kinds = list(bindings.values())
                if (kinds.count('state/odom') != 1 or kinds.count('sensor/pointcloud2') != 1 or
                        kinds.count('image/jpeg') != len(input_topics)-2):
                    raise ValueError('native ROS inputs require one String pose and one PointCloud2 cloud')
                return bindings
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = ', '.join(topic for topic in input_topics if topic not in bindings)
                raise ValueError('native ROS input discovery timed out: ' + missing)
            time.sleep(min(.05, remaining))
    finally:
        node.destroy_node()


from .pointcloud_normalization import decode_cloud, normalize_cloud


_MAP_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
_RGB_WAIT_MS = 20
# A measured observation can become a durable visual anchor. Keep its own
# projection long enough for the measured Tianyi latency (~290 ms p95).
_MEASURED_RGB_WAIT_MS = 400
_FIRST_MAP_RGB_WAIT_MS = _MEASURED_RGB_WAIT_MS
# Allow one default RTAB-Map detection period for native startup and a usable root.
_MAP_ROOT_WINDOW_S = 1.
_NATIVE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=2, durability=DurabilityPolicy.VOLATILE)


def _native_pose(pose):
    return pose_matrix(SimpleNamespace(
        position=(pose.position.x, pose.position.y, pose.position.z),
        quaternion=(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)))


def _quaternion(matrix):
    """Convert an already validated rotation to an xyzw unit quaternion."""
    r = matrix[:3, :3]
    trace = np.trace(r)
    if trace > 0:
        scale = 2 * math.sqrt(trace + 1.)
        return ((r[2, 1]-r[1, 2])/scale, (r[0, 2]-r[2, 0])/scale,
                (r[1, 0]-r[0, 1])/scale, scale/4)
    i = int(np.argmax(np.diag(r)))
    j, k = (i+1) % 3, (i+2) % 3
    scale = 2 * math.sqrt(1 + r[i, i] - r[j, j] - r[k, k])
    values = [0., 0., 0., (r[k, j]-r[j, k])/scale]
    values[i], values[j], values[k] = scale/4, (r[j, i]+r[i, j])/scale, (r[k, i]+r[i, k])/scale
    return values


class _StorageLimit(RuntimeError):
    pass


class RtabmapNode(Node):
    """A short-lock input gate, native process worker and bounded map display.

    Each mapping session has a private ROS graph and native database.
    Process startup, shutdown, disk checks and map conversion never hold the
    capture lock. freeze_capture therefore remains a quick publication barrier.
    """
    def __init__(self, odom_topic, cloud_topic, settings, *, session_dir,
                 input_contract=None, capture_stop=None, input_registry=None, input_declarations=None,
                 restored_map=None):
        self._restored_map = deepcopy(restored_map)
        self._input_registry, self._input_leases = input_registry, []
        self._odom_topic = odom_topic
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        from rtabmap_msgs.msg import Info, MapGraph
        self._Odometry, self._TransformStamped, self._TFMessage = Odometry, TransformStamped, TFMessage
        self._Info, self._MapGraph = Info, MapGraph
        self._settings = deepcopy(settings)
        self._settings.update((input_contract or {}).get('settings', {}))
        self._rgb = None
        self._rgb_mode, self._rgb_reason = 'pointcloud', None
        self._rgb_subscription_error = None
        self._rgb_task = None
        self._root_deadline = None
        self._root_wait_done = threading.Event()
        self._rgb_preparation_ms = deque(maxlen=64)
        self._rgb_timeouts = self._rgb_published = 0
        if self._settings.get('rgb_input') is not None:
            from .pointcloud_slam_rgb import RgbLoopInput
            from rtabmap_msgs.msg import SensorData
            self._SensorData = SensorData
            self._rgb = RgbLoopInput(self._settings['rgb_input'], self._settings,
                                     max_pending=self._settings.get('max_pending', 10),
                                     on_decoded=self._reserve_decoded_rgb)
        for key, default in dict(voxel_size_m=.08, max_working_nodes=3000, max_update_ms=3000,
                max_disk_bytes=2*1024**3, reserve_bytes=1024**3,
                preview_interval_s=1., preview_max_points=40000).items():
            self._settings.setdefault(key, default)
        for key in ('max_working_nodes', 'max_disk_bytes', 'preview_max_points'):
            if type(self._settings[key]) is not int or self._settings[key] <= 0:
                raise ValueError(key + ' must be a positive integer')
        if type(self._settings['reserve_bytes']) is not int or self._settings['reserve_bytes'] < 0:
            raise ValueError('reserve_bytes must be a nonnegative integer')
        for key in ('voxel_size_m', 'max_update_ms', 'preview_interval_s'):
            value = self._settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + ' must be positive and finite')
        if self._settings['preview_max_points'] > 80000:
            raise ValueError('preview_max_points must not exceed 80000')
        self._lock, self._result_lock, self._close_lock = threading.RLock(), threading.RLock(), threading.Lock()
        self._capture_stop = threading.Event()
        self._project_stop = capture_stop if capture_stop is not None else threading.Event()
        self._worker_stop, self._status_stop, self._wake = threading.Event(), threading.Event(), threading.Event()
        self._queue = deque()
        self._motion_waiting = deque()
        self._motion_mounts = OrderedDict()
        self._motion_generation = 0
        self._admission_seq = self._published_seq = self._pose_loss_seq = 0
        self._motion = None
        self._cloud_ingress = OrderedDict()
        self._rgb_reservations = OrderedDict()
        self._map_decision = None
        self._covariance_source = None
        self._segments, self._active = [], None
        self._worker = self._status_worker = None
        self._started = self._closed = False
        self._start_done = threading.Event()
        self._start_done.set()
        self._runtime_error = ''
        self._stop_reason = ''
        self._forwarded = self._dropped = 0
        self._native_rejections = {'graph': 0, 'cloud': 0}
        self._native_rejection_streaks = {'graph': 0, 'cloud': 0}
        self._disk_bytes = 0
        self._session_dir = Path(session_dir).resolve()
        self._namespace = '/pointcloud_slam/session_' + uuid.uuid4().hex
        self._map_topic = PREVIEW_TOPIC
        self._receipt_clock = time.monotonic
        super().__init__('pointcloud_slam_gateway')
        if self._rgb is not None and self._rgb.camera is not None:
            self._motion = ScanMotion(self,self._namespace+'/motion',self._settings,
                                      self._session_dir/'motion',self._wake)
        self._measurement_clock_ns = lambda: getattr(self.get_clock().now(), 'nanoseconds', None)
        keys = ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud', 'max_gap_ms',
                'max_age_s', 'max_pending', 'max_cloud_points', 'max_measurement_age_ms', 'max_future_skew_ms')
        try:
            self.stream = SlamStream(**{key: self._settings[key] for key in keys},
                extrinsic_mode=self._settings.get('extrinsic_mode', 'static'),
                input_contract=input_contract, on_frame=self._queue_frame,
                on_observation=self._observe_motion if self._motion is not None else None,
                on_unlocalized_frame=self._queue_frame if self._rgb is not None else None,
                on_pose_loss=self._retire_pose_work)
            self.stream.ingress_clock_id = input_registry.clock_id if input_registry else None
            self._capture_clouds = None
            if self._settings.get('capture_mount'):
                if input_contract:
                    raise ValueError('capture mount cannot replace an explicit SLAM input contract')
                from .pointcloud_slam_capture_mount import CaptureClouds
                self._capture_clouds = CaptureClouds(self._settings['capture_mount'],
                                                    max_pending=self.stream.max_pending)
                self.create_subscription(String, self._settings['capture_mount']['declaration']['topic'],
                                         self._capture_mount, _QOS)
            self._map_pub = self.create_publisher(UInt8MultiArray, self._map_topic, _QOS)
            for topic, kind, callback in ((odom_topic, String, self._odom),
                                          (cloud_topic, PointCloud2, self._cloud)):
                if input_registry is None:
                    self.create_subscription(kind, topic, callback, _QOS)
                else:
                    declaration = (input_contract or {}).get("inputs", {}).get(topic) or (input_declarations or {}).get(topic, {})
                    if not declaration and topic==cloud_topic and self._settings.get('time_basis','source_capture')=='source_capture':
                        # Existing configured v1 path already requires capture
                        # headers in this clock. New receipt paths declare v2.
                        declaration={'vendor':{'clock_id':self.stream.clock_id,'timestamp_semantics':'source_capture_mapped'}}
                    self._input_leases.append(input_registry.acquire(topic, kind, declaration, callback))
            if input_contract and input_contract.get('frame_metadata_topic'):
                self.create_subscription(String, input_contract['frame_metadata_topic'], self._cloud_metadata, _QOS)
            if self._rgb is not None and self._rgb.camera is not None:
                topic = self._settings['rgb_input']['topic']
                try:
                    if input_registry is None:
                        self.create_subscription(CompressedImage, topic, self._rgb_image, _QOS)
                    else:
                        self._input_leases.append(input_registry.acquire(
                            topic, CompressedImage, self._rgb.camera, self._rgb_image))
                except Exception as error:
                    self._rgb_subscription_error = 'RGB subscription unavailable: ' + str(error)
        except Exception:
            for lease in self._input_leases:
                lease.close()
            self.destroy_node()
            raise

    def _rgb_image(self, msg, context=None):
        if not self._stopped():
            self._rgb.image(msg, context, now_ns=self._measurement_clock_ns())

    def _reserve_decoded_rgb(self, stamp):
        with self._lock:
            if self._stopped() or not (any(key[0] == stamp for key in self._cloud_ingress) or
                                      any(item[0] == stamp for item in self._queue)):
                return
            rgb = self._rgb.take(stamp, self._measurement_clock_ns(), exact=True)
            if rgb is not None:
                self._rgb_reservations[stamp] = rgb
                while len(self._rgb_reservations) > self.stream.max_pending*3:
                    self._rgb_reservations.popitem(last=False)

    def _stopped(self):
        return self._capture_stop.is_set() or self._project_stop.is_set()

    def _observe_motion(self, stamp, cloud, matrix):
        if not self._stopped() and self._motion is not None:
            _,_,points=decode_cloud(cloud,max_points=self.stream.max_cloud_points)
            self._motion.submit(stamp,points,matrix)

    def _measurement_now(self):
        return self._measurement_clock_ns() if self.stream.max_measurement_age_ms is not None else None

    def _odom(self, msg, context=None):
        try:
            from .spatial_contract import normalize_pose_time
            sample = normalize_pose_time(json.loads(msg.data), context)
        except (TypeError, ValueError):
            sample = None
        derived = (context if context is not None and self._input_registry is not None and
                   self._input_registry.is_derived(self._odom_topic, msg, context) else None)
        with self._lock:
            if not self._stopped():
                try:
                    if self._capture_clouds is not None:
                        self._capture_clouds.validate_pose(sample, self.stream)
                    self.stream.odom(sample, now=self._receipt_clock(), measurement_now_ns=self._measurement_now(),
                        received=context.ingress_monotonic_ns/1e9 if context is not None and
                        context.ingress_clock_id==self.stream.ingress_clock_id else None, context=derived)
                except ValueError as error:
                    self.stream.last_error = str(error)
        self._wake.set()

    def _retire_pose_work(self):
        # Retire work that depended on the old pose without cancelling recovery scans.
        self._pose_loss_seq = self._admission_seq

    def _capture_mount(self, msg):
        try:
            packet = json.loads(msg.data)
            with self._lock:
                if not self._stopped():
                    if self._rgb is not None:
                        self._rgb.mount(packet, self._measurement_clock_ns())
                    self._capture_clouds.mount(packet, self.stream, now=self._receipt_clock(),
                                               measurement_now_ns=self._measurement_clock_ns())
        except (TypeError, ValueError) as error:
            with self._lock:
                self.stream.last_error = str(error)
                self._queue.clear()
        self._wake.set()

    def _cloud_metadata(self, msg):
        try:
            packet = json.loads(msg.data)
            with self._lock:
                if not self._stopped():
                    self.stream.cloud_metadata(packet, now=self._receipt_clock(), measurement_now_ns=self._measurement_now())
        except (TypeError, ValueError) as error:
            with self._lock:
                self.stream.last_error = str(error)
        self._wake.set()

    def _cloud(self, msg, context=None):
        if self._stopped():
            return
        try:
            if context is not None:
                from .pointcloud_slam_capture_mount import validate_cloud_clock
                stamp = msg.header.stamp.sec*10**9 + msg.header.stamp.nanosec
                validate_cloud_clock(context, self.stream.clock_id, stamp, self._settings.get('capture_mount'))
                if context.time_basis=='receipt_estimated':
                    msg=deepcopy(msg)
                    msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(context.observation_stamp_ns,10**9)
            stamp, frame, _ = decode_cloud(msg, max_points=self.stream.max_cloud_points)
            received = (context.ingress_monotonic_ns/1e9 if context is not None and
                        context.ingress_clock_id == self.stream.ingress_clock_id else self._receipt_clock())
            digest = hashlib.sha256(bytes(msg.data)).hexdigest() if self.stream.input_contract else None
            with self._lock:
                if not self._stopped():
                    key = (stamp,frame,hashlib.sha256(msg.data).hexdigest())
                    # Keep measured depth with its cloud across the bounded pose wait.
                    depth = (self._input_registry.derived_depth(context)
                             if self._rgb is not None and context is not None and self._input_registry is not None
                             else None)
                    self._cloud_ingress[key] = (received, context, depth)
                    while len(self._cloud_ingress) > self.stream.max_pending*3:self._cloud_ingress.popitem(last=False)
                    if self._rgb is not None:
                        rgb = self._rgb.take(stamp, self._measurement_clock_ns(), exact=True)
                        if rgb is not None:
                            # Own this image with its waiting cloud, independently
                            # of the decoder's rolling cache and runtime backlog.
                            self._rgb_reservations[stamp] = rgb
                        while len(self._rgb_reservations) > self.stream.max_pending*3:
                            self._rgb_reservations.popitem(last=False)
                    if self._capture_clouds is not None:
                        self._capture_clouds.cloud(stamp, frame, msg, msg.width*msg.height, self.stream,
                            now=self._receipt_clock(), measurement_now_ns=self._measurement_clock_ns(), received=received)
                    else:
                        self.stream.cloud(stamp, frame, msg, point_count=msg.width*msg.height,
                            now=self._receipt_clock(), measurement_now_ns=self._measurement_now(), data_sha256=digest, received=received)
        except ValueError as error:
            with self._lock:
                self.stream.last_error = str(error)
        self._wake.set()

    def _queue_frame(self, stamp, cloud, pose, matrix, context):
        key = (stamp,cloud.header.frame_id,hashlib.sha256(cloud.data).hexdigest())
        ingress = self._cloud_ingress.pop(key, None)
        if ingress is None:
            raise ValueError('cloud original ingress context expired')
        received, observation, depth = ingress
        predicted = pose is not None and pose.prediction_anchor_ns is not None
        if predicted and (self._rgb is None or self._active is None):
            self._rgb_reservations.pop(stamp, None)
            self._dropped += 1
            return
        self._map_decision = dict(action='accept' if pose is not None else 'localize',
            reason='numeric cloud and pose' if pose is not None else 'external pose absent; native registration required')
        if predicted:
            self._map_decision = dict(action='register', reason='inertial guess requires native scan registration')
        normalized = normalize_cloud(cloud, observation, None,
                                     max_points=self.stream.max_cloud_points)
        cloud = normalized.message
        if len(self._queue) >= self.stream.max_pending:
            self._queue.popleft()
            self._dropped += 1
        self._admission_seq += 1
        # Allow the existing measured-pose wait and one bounded late result.
        rgb_deadline = time.monotonic() + (2 if pose is not None else 1) * _MEASURED_RGB_WAIT_MS / 1000.
        self._queue.append((stamp, cloud, pose, matrix, received, self._admission_seq, observation, rgb_deadline, depth))
        if pose is not None:
            # A usable newer observation supersedes optional unlocalized work;
            # never replay a delayed frame into an already advanced native map.
            self._dropped += len(self._motion_waiting)
            self._motion_generation += 1
            for pending in self._motion_waiting:
                self._rgb_reservations.pop(pending[0],None)
                self._motion_mounts.pop(pending[0],None)
            self._motion_waiting.clear()
            self._motion_mounts.clear()
        self._wake.set()

    def start(self):
        with self._lock:
            if self._started or self._closed or self._stopped():
                return
            self._started = True
            self._start_done.clear()
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                if self._stopped():
                    return
                self._worker = threading.Thread(target=self._run, name='rtabmap_mapping', daemon=True)
                self._status_worker = threading.Thread(target=self._status_loop, name='rtabmap_status', daemon=True)
                if self._rgb is not None:
                    self._rgb.start()
                self._worker.start()
                self._status_worker.start()
        finally:
            self._start_done.set()

    def freeze_capture(self):
        self._capture_stop.set()
        self._root_wait_done.set()
        with self._lock:
            if self._rgb_task is not None:
                # Wake the mapping worker. An in-progress projection may finish
                # off-lock but its result cannot be consumed after capture stops.
                self._rgb_task[1].cancel()
            self._queue.clear()
            self._motion_waiting.clear()
            self._motion_mounts.clear()
            self._rgb_reservations.clear()
            self._cloud_ingress.clear()
            self.stream._pending.clear()
            self.stream._unmatched_clouds.clear()
            if self._capture_clouds is not None:
                self._capture_clouds.pending.clear()
        self._wake.set()

    def _check_budget(self):
        used = 0
        for path in self._session_dir.rglob('*'):
            try:
                if path.is_file():
                    used += path.stat().st_size
            except FileNotFoundError:
                pass  # SQLite may remove its journal between discovery and stat.
        self._disk_bytes = used
        if used > self._settings['max_disk_bytes']:
            raise _StorageLimit('RTAB-Map session disk budget reached; native data retained')
        if shutil.disk_usage(self._session_dir).free < self._settings['reserve_bytes']:
            raise _StorageLimit('RTAB-Map reserved free disk space reached; native data retained')

    def _activate(self, pose):
        self._check_budget()
        if self._active is not None:
            with self._result_lock:
                self._active['closing'] = True
            self._active['process'].close()
            self._active['finalized'] = True
            for subscription in self._active['subscriptions']:
                self.destroy_subscription(subscription)
        if self._stopped():
            return None
        index = len(self._segments)
        namespace = self._namespace + '/segment_' + str(index)
        restored = self._restored_map if index == 0 else None
        database = (Path(restored['database']) if restored else
                    self._session_dir / ('segment_' + str(index)) / 'rtabmap.db')
        process = RtabmapProcess(database, namespace, self._settings,
                                **({'restored': True} if restored else {}))
        process.validate_runtime()
        segment = dict(segment_id=restored['segment_id'] if restored else pose.segment_id or 'segment_'+str(index),
            frame_id=restored['frame_id'] if restored else '', source_frame_id=pose.frame_id if pose else '',
            pose_reference=getattr(pose, 'pose_reference', None),
            process=process, snapshot=restored.get('snapshot') if restored else None,
            graph=None, cloud=None, finalized=False, closing=False,
            native_loop_closures=0, native_proximity_detections=0, native_updates=0,
            native_recovery={'state': 'unavailable'},
            last_info=None, subscriptions=[], last_preview=0., forwarded=0,
            progress=BackendProgress(timeout_s=max(10.,4*self._settings['max_update_ms']/1000)))
        with self._result_lock:
            self._segments.append(segment)
            self._active = segment
        segment['odom_pub'] = self.create_publisher(self._Odometry, namespace + '/odom', _NATIVE_QOS)
        segment['cloud_pub'] = self.create_publisher(
            self._SensorData if self._rgb is not None else PointCloud2,
            namespace + ('/sensor_data' if self._rgb is not None else '/scan_cloud'), _NATIVE_QOS)
        segment['tf_pub'] = self.create_publisher(self._TFMessage,
            namespace + ('/tf' if self.stream.extrinsic_mode == 'dynamic' else '/tf_static'),
            _NATIVE_QOS if self.stream.extrinsic_mode == 'dynamic' else _MAP_QOS)
        segment['odom_tf_pub'] = (segment['tf_pub'] if self.stream.extrinsic_mode == 'dynamic' else
            self.create_publisher(self._TFMessage, namespace + '/tf', _NATIVE_QOS))
        for message, suffix, callback, qos in [
                (PointCloud2, '/cloud_map', self._native_cloud, _MAP_QOS),
                (self._MapGraph, '/mapGraph', self._native_graph, _MAP_QOS),
                (self._Info, '/info', self._native_info, _NATIVE_QOS)]:
            segment['subscriptions'].append(self.create_subscription(message, namespace + suffix,
                lambda msg, target=segment, handler=callback: handler(msg, target), qos))
        process.start()
        # Native initialization can exceed ten seconds during model warmup.
        # Bound process discovery independently of input measurement timestamps.
        deadline = time.monotonic() + 30.
        while not self._stopped():
            if process.poll() is not None:
                raise RuntimeError('RTAB-Map exited during startup (exit ' + str(process.poll()) + ')')
            if segment['odom_pub'].get_subscription_count() and segment['cloud_pub'].get_subscription_count():
                return segment
            if time.monotonic() >= deadline:
                raise RuntimeError('RTAB-Map input discovery timed out')
            self._capture_stop.wait(.05)
        return segment

    def _retire_overtaken(self, item):
        if len(item) < 6:
            return False
        sequence = item[5]
        if sequence >= self._published_seq and (item[2] is None or sequence > self._pose_loss_seq):
            return False
        self._rgb_reservations.pop(item[0],None)
        self._motion_mounts.pop(item[0],None)
        self._dropped += 1
        return True

    def _publish_pair(self, segment, item, motion_generation=None):
        if motion_generation is None:
            motion_generation = self._motion_generation
        with self._lock:
            if self._retire_overtaken(item):
                return
        stamp, cloud, pose, matrix, received = item[:5]
        predicted = pose is not None and pose.prediction_anchor_ns is not None
        measured = pose if not predicted else None
        transform = self._TransformStamped()
        transform.header = deepcopy(cloud.header)
        transform.header.frame_id = segment['process'].namespace.lstrip('/') + '/base'
        transform.child_frame_id = segment['process'].namespace.lstrip('/') + '/cloud'
        for key, value in zip(('x', 'y', 'z'), matrix[:3, 3]):
            setattr(transform.transform.translation, key, float(value))
        for key, value in zip(('x', 'y', 'z', 'w'), _quaternion(matrix)):
            setattr(transform.transform.rotation, key, float(value))
        tf = self._TFMessage(); tf.transforms = [transform]
        odom = self._Odometry()
        odom.header = deepcopy(cloud.header)
        odom.header.frame_id = segment['process'].namespace.lstrip('/') + '/odom'
        odom.child_frame_id = transform.header.frame_id
        if measured is not None:
            for key, value in zip(('x', 'y', 'z'), pose.position):
                setattr(odom.pose.pose.position, key, value)
            for key, value in zip(('x', 'y', 'z', 'w'), pose.quaternion):
                setattr(odom.pose.pose.orientation, key, value)
        else:
            # Pinned transformFromPoseMsg maps an all-zero quaternion to null.
            # This is an explicit absent estimate, never the last valid pose.
            odom.pose.pose.orientation.w = 0.
        if measured is not None and pose.covariance is not None:
            odom.pose.covariance = list(pose.covariance)
        # Native odomUpdate looks up this exact pose even with an Odometry input.
        odom_transform = self._TransformStamped()
        odom_transform.header = deepcopy(odom.header)
        odom_transform.child_frame_id = odom.child_frame_id
        for key in ('x', 'y', 'z'):
            setattr(odom_transform.transform.translation, key, getattr(odom.pose.pose.position, key))
        odom_transform.transform.rotation = deepcopy(odom.pose.pose.orientation)
        odom_tf = self._TFMessage(); odom_tf.transforms = [odom_transform]
        if self.stream.extrinsic_mode == 'dynamic':
            odom_tf.transforms.append(transform)
        cloud.header.frame_id = transform.child_frame_id
        observation = item[6] if len(item) > 6 else None
        rgb_deadline = item[7] if len(item) > 7 else time.monotonic()
        depth = item[8] if len(item) > 8 else None
        native, rgb_generation = (self._sensor_data(cloud, pose, matrix, motion_generation, observation, rgb_deadline, depth)
                                  if self._rgb is not None else (cloud, None))
        if predicted:
            # The standard odometry input stays null. Even an older native
            # runtime cannot mistake this private hint for a tracking pose.
            payload = struct.pack('<8s7d', b'MOTUSIMU', *pose.position, *pose.quaternion)
            native.user_data = array('B', zlib.compress(payload)+struct.pack('<iii', 1, len(payload), 0))
        if (self._root_deadline and pose is not None and not native.left.data and
                (self._rgb_task is None or self._rgb_task[1].done()) and self._prefer_next_root_observation()):
            return
        with self._lock:
            if self._stopped() or self._retire_overtaken(item):
                return
            if pose is None and motion_generation != self._motion_generation:
                return
            if pose is not None and not self.stream._poses:
                return
            token = self.stream.continuity_token
            if (self._rgb is not None and measured is not None and token is not None and
                    measured.external_continuity == token):
                # Optional preparation can outlive the source run of a queued pose.
                payload = b'MOTUSOD1' + token
                native.user_data = array('B', zlib.compress(payload)+struct.pack('<iii', 1, len(payload), 0))
            if self.stream.extrinsic_mode != 'dynamic':
                segment['tf_pub'].publish(tf)
            if measured is not None:
                segment['odom_tf_pub'].publish(odom_tf)
            elif self.stream.extrinsic_mode == 'dynamic':
                # Keep the numerical mount, without publishing an invalid odom TF.
                segment['tf_pub'].publish(tf)
            segment['odom_pub'].publish(odom)
            # A decode fault may invalidate the optional image during projection;
            # the admitted scan and pose continue through the same native node.
            guard = self._rgb.publication(rgb_generation) if self._rgb is not None else nullcontext(True)
            with guard as valid_rgb:
                if self._rgb is not None and native.left.data and not valid_rgb:
                    native.left.data, native.right.data = b'', b''
                    native.left_camera_info, native.local_transform = [], []
                    self._rgb_mode, self._rgb_reason = 'pointcloud', 'RGB expired or invalidated before native publication'
                segment['cloud_pub'].publish(native)
                if self._rgb is not None and native.left.data:
                    self._rgb_published += 1
            if len(item)>5:
                self._published_seq = max(self._published_seq,item[5])
            self._forwarded += 1
            segment['forwarded'] += 1
            with self._result_lock:
                segment['progress'].forward(stamp,self._receipt_clock())
            self._covariance_source = ('inertial_guess' if predicted else 'native_visual' if pose is None else
                'upstream' if pose.covariance is not None else 'backend_default')
            startup_backlog = self._root_deadline and native.left.data and bool(self._queue)
        if startup_backlog:
            # Avoid flooding the just-discovered native subscriber with a burst
            # before it has accepted any root. Inputs continue to queue normally.
            remaining = min(_FIRST_MAP_RGB_WAIT_MS / 1000., self._root_deadline - time.monotonic())
            if remaining > 0:
                self._root_wait_done.wait(remaining)

    def _prefer_next_root_observation(self):
        while not self._root_wait_done.is_set() and not self._stopped():
            remaining = self._root_deadline - time.monotonic()
            if remaining <= 0:
                break
            self._wake.clear()
            with self._lock:
                if self._queue:
                    self._dropped += 1
                    self._wake.set()
                    return True
            self._wake.wait(remaining)
        return False  # One bounded fallback scan when RGB-D never becomes usable.

    def _sensor_data(self, cloud, pose, matrix, motion_generation=None, observation=None, rgb_deadline=0.,
                     measured_depth=None):
        """Scan is mandatory; optional source-timed imagery may fail independently."""
        if motion_generation is None:
            motion_generation = self._motion_generation
        stamp, _, points = decode_cloud(cloud, max_points=self.stream.max_cloud_points)
        data = self._SensorData()
        data.header = deepcopy(cloud.header)
        # This card has no IMU input. Geometry message defaults may be identity;
        # the native IMU converter uses a null local transform to mean absent.
        data.imu_local_transform.rotation.w = 0.
        # The native converter requires the scan enum to match its actual PCL
        # layout. Carry explicit XYZ geometry without changing the input cloud.
        data.laser_scan = deepcopy(cloud)
        data.laser_scan.fields = [PointField(name=name, offset=i*4, datatype=7, count=1)
                                 for i, name in enumerate(('x', 'y', 'z'))]
        data.laser_scan.width, data.laser_scan.height = len(points), 1
        data.laser_scan.point_step, data.laser_scan.row_step = 12, len(points)*12
        data.laser_scan.is_bigendian, data.laser_scan.is_dense = False, True
        data.laser_scan.data = array('B', np.ascontiguousarray(points, dtype='<f4').tobytes())
        data.laser_scan_format, data.laser_scan_max_pts, data.laser_scan_max_range = 5, 0, 0.
        data.laser_scan_local_transform = self._native_transform(matrix)
        self._rgb_mode, self._rgb_reason = 'pointcloud', self._rgb_subscription_error
        if pose is not None and pose.prediction_anchor_ns is not None:
            with self._lock:
                self._rgb_reservations.pop(stamp, None)
            return data, None  # Keep the guessed motion out of the measured scan.
        if self._rgb_subscription_error or self._rgb.camera is None:
            return data, None
        key = (pose.segment_id if pose is not None else None, stamp, id(cloud))
        # Seed a new map with visual evidence when a connected JPEG has arrived.
        # An existing database must establish its own map-to-odometry relation.
        if self._root_deadline is None:
            self._root_deadline = (time.monotonic() + _MAP_ROOT_WINDOW_S
                if self._restored_map is None and pose is not None and self._rgb.info()['received'] else 0.)
        budget_ms = _MEASURED_RGB_WAIT_MS if pose is not None else _RGB_WAIT_MS
        if not self._root_wait_done.is_set() and time.monotonic() < self._root_deadline:
            budget_ms = _FIRST_MAP_RGB_WAIT_MS
        try:
            # At most one optional job exists. A hung decoder/projection cannot
            # queue work or hold required scans beyond this small wait budget.
            task = self._rgb_task
            ready = task is not None and task[1].done()
            if ready and task[0] != key:
                with self._lock:
                    owned = any(frame[1] is task[2]
                        for pending in (self._queue, self._motion_waiting) for frame in pending)
                if owned:
                    raise MotionPending('completed RGB preparation is waiting for its observation')
            if task is None or ready and task[0] != key:
                future = Future()
                self._rgb_task = (key, future, cloud)

                def prepare():
                    started = time.perf_counter()
                    try:
                        depth = measured_depth
                        if depth is None and observation is not None and self._input_registry is not None:
                            depth = self._input_registry.derived_depth(observation)
                        result = self._rgb_attachment(cloud.header, stamp, pose, matrix, points,
                                                      motion_generation, depth)
                        with self._lock:
                            if not future.cancelled():
                                self._rgb_preparation_ms.append((time.perf_counter()-started)*1000.)
                                future.set_result(result)
                        self._wake.set()
                    except Exception as error:
                        with self._lock:
                            if not future.cancelled():
                                future.set_exception(error)

                threading.Thread(target=prepare, daemon=True, name='slam-optional-projection').start()
            if self._rgb_task[0] != key:
                raise TimeoutError('previous RGB preparation is still busy')
            if self._stopped():
                return data, None
            result = self._rgb_task[1].result(timeout=budget_ms / 1000.)
            if result is not None:
                left, right, info, local, generation = result
                data.left, data.right = left, right
                data.left_camera_info, data.local_transform = [info], [local]
                self._rgb_mode = 'image'
                return data, generation
        except MotionPending as error:
            if self._rgb_task is not None and self._rgb_task[0] == key:
                self._rgb_task = None
            if pose is None and time.monotonic() < rgb_deadline:
                raise
            with self._lock:
                self._rgb_reservations.pop(stamp, None)
                self._motion_mounts.pop(stamp, None)
            self._rgb_reason = str(error)
        except TimeoutError:
            with self._lock:
                self._rgb_timeouts += 1
            with self._result_lock:
                searching = self._active is not None and self._active['native_recovery']['state'] == 'searching'
            if time.monotonic() < rgb_deadline and (pose is None or (searching and self._rgb_task[0] == key)):
                # Give a late image one bounded chance to accompany its scan.
                raise MotionPending('optional RGB preparation is pending within the bounded worker')
            self._rgb_reason = f'optional RGB preparation exceeded {budget_ms} ms; scan forwarded without images'
        except Exception as error:
            self._rgb_reason = str(error)
        return data, None

    def _rgb_attachment(self, header, stamp, pose, matrix, points, motion_generation=None, measured_depth=None):
        from sensor_msgs.msg import Image, CameraInfo
        from .pointcloud_slam_alignment import align_pose
        from .pointcloud_slam_rgb import project_depth, project_measured_depth
        now_ns = self._measurement_clock_ns()
        with self._lock:
            rgb = self._rgb_reservations.pop(stamp, None)
            saved_mount = self._motion_mounts.pop(stamp, None)
        if rgb is None:
            # Decode may still be running after the cloud left ingress/queue.
            # Wait only for an already received matching JPEG.
            rgb = self._rgb.take(stamp, now_ns, wait_s=.02)
        if rgb is None:
            return None
        with self._lock:
            rgb_stamp, image, generation = rgb
            poses = list(self.stream._poses)
            mount = saved_mount if saved_mount is not None else self._rgb.capture_mount(rgb_stamp, matrix, now_ns)
            if mount is None:
                return None
        # Express the camera at its real capture time relative to the scan's
        # reference pose. Neither source header is retimed to manufacture sync.
        if pose is None:
            if rgb_stamp != stamp:
                try:
                    if self._motion is None:
                        raise MotionPending('measured motion unavailable without external poses')
                    local = self._motion.timeline.relative(rgb_stamp,stamp) @ mount
                except MotionPending:
                    with self._lock:
                        if not self._stopped() and motion_generation == self._motion_generation:
                            self._rgb_reservations[stamp] = rgb
                            self._motion_mounts[stamp] = mount.copy()
                            while len(self._motion_mounts)>self.stream.max_pending*3:
                                expired,_=self._motion_mounts.popitem(last=False)
                                self._rgb_reservations.pop(expired,None)
                    raise
            else:
                local = mount
        else:
            try:
                rgb_pose = align_pose(poses, rgb_stamp, max_gap_ms=self.stream.max_gap_ms)
                local = np.linalg.inv(pose_matrix(pose)) @ pose_matrix(rgb_pose) @ mount
            except ValueError:
                if self._motion is None:raise
                local = self._motion.timeline.relative(rgb_stamp,stamp) @ mount
        registration = np.linalg.inv(local) @ matrix
        if measured_depth is None:
            depth = project_depth(points[::max(1, (len(points)+99999)//100000)], registration,
                                  self._rgb.K, self._rgb.width, self._rgb.height)
        else:
            depth = project_measured_depth(measured_depth, registration, self._rgb.K,
                                           self._rgb.width, self._rgb.height)
        if not np.any(depth):
            raise ValueError('measured scan has no depth in the RGB field of view')
        left, right = Image(), Image()
        left.header = deepcopy(header)
        left.header.stamp.sec, left.header.stamp.nanosec = divmod(rgb_stamp, 10**9)
        left.height, left.width = self._rgb.height, self._rgb.width
        # ROS accepts typed uint8 arrays directly. Assigning bytes instead runs
        # Python validation per byte, which can exhaust the optional budget.
        left.encoding, left.step = 'bgr8', left.width*3
        left.data = array('B', image.tobytes())
        right.header = deepcopy(header)
        right.height, right.width = depth.shape
        right.encoding, right.step = '32FC1', right.width*4
        right.data = array('B', np.asarray(depth, dtype='<f4').tobytes())
        info = CameraInfo()
        info.header, info.height, info.width = deepcopy(left.header), left.height, left.width
        info.distortion_model, info.d = 'plumb_bob', [0.]*5
        info.k, info.r = self._rgb.K.reshape(-1).tolist(), np.eye(3).reshape(-1).tolist()
        info.p = np.column_stack((self._rgb.K, np.zeros(3))).reshape(-1).tolist()
        return left, right, info, self._native_transform(local), generation

    def _native_transform(self, matrix):
        transform = self._TransformStamped().transform
        for key, value in zip(('x', 'y', 'z'), matrix[:3, 3]):
            setattr(transform.translation, key, float(value))
        for key, value in zip(('x', 'y', 'z', 'w'), _quaternion(matrix)):
            setattr(transform.rotation, key, float(value))
        return transform

    def _run(self):
        while not self._worker_stop.is_set() and not self._stopped():
            self._wake.wait(.25)
            self._wake.clear()
            with self._lock:
                item = None
                if self._rgb_task is not None and self._rgb_task[1].done() and not self._stopped():
                    # Consume a completed attachment before another pending
                    # observation can replace its single preparation future.
                    for pending in (self._queue, self._motion_waiting):
                        ready = next((i for i, frame in enumerate(pending)
                            if frame[1] is self._rgb_task[2]), None)
                        if ready is not None:
                            item = pending[ready]
                            del pending[ready]
                            break
                if item is None:
                    pending = self._queue if self._queue else self._motion_waiting
                    item = pending.popleft() if pending and not self._stopped() else None
                if item is not None and self._retire_overtaken(item):
                    item = None
                generation = self._motion_generation
                if self._queue:
                    self._wake.set()
            if item is None:
                continue
            try:
                if self._active is None:
                    if item[2] is None and self._restored_map is None:
                        continue  # There is no existing map to relocalize in yet.
                    segment = self._activate(item[2])
                else:
                    segment = self._active
                if segment is None:
                    continue
                if segment['process'].poll() is not None:
                    raise RuntimeError('RTAB-Map process exited (exit ' + str(segment['process'].poll()) + ')')
                self._publish_pair(segment, item, generation)
            except MotionPending as error:
                with self._lock:
                    self._rgb_reason=str(error)
                    if not self._stopped() and generation == self._motion_generation:
                        if len(self._motion_waiting)>=self.stream.max_pending:
                            expired=self._motion_waiting.popleft()
                            self._rgb_reservations.pop(expired[0],None)
                            self._motion_mounts.pop(expired[0],None)
                            self._dropped+=1
                        self._motion_waiting.append(item)
                    else:
                        self._rgb_reservations.pop(item[0],None)
                        self._motion_mounts.pop(item[0],None)
                        self._dropped+=1
            except ValueError as error:
                with self._lock:
                    self.stream.last_error = str(error)
            except _StorageLimit as error:
                self._stop_reason = str(error)
                self.freeze_capture()
                break
            except Exception as error:
                self._runtime_error = str(error)
                self.freeze_capture()
                break

    def _status_loop(self):
        while not self._status_stop.wait(min(.25, self._settings['preview_interval_s'])):
            try:
                with self._lock:
                    if not self._stopped():
                        self.stream.poll(now=self._receipt_clock())
                self._check_budget()
                active = self._active
                if active is not None and not active['finalized'] and not active['closing'] and active['process'].poll() is not None and not self._stopped():
                    raise RuntimeError('RTAB-Map process exited (exit ' + str(active['process'].poll()) + ')')
            except _StorageLimit as error:
                self._stop_reason = str(error)
                self.freeze_capture()
            except (OSError, RuntimeError) as error:
                self._runtime_error = str(error)
                self.freeze_capture()

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _native_rejected(self, kind, error):
        """Report the first rejection, then every 100 additional consecutive errors."""
        with self._result_lock:
            self._native_rejections[kind] += 1
            self._native_rejection_streaks[kind] += 1
            total = self._native_rejections[kind]
            streak = self._native_rejection_streaks[kind]
        if (streak - 1) % 100 == 0:
            # Native input may be externally sourced; never log unbounded or multiline details.
            detail = ''.join(char if char.isprintable() else ' '
                             for char in str(error)[:200])
            log.warning('Invalid native RTAB-Map %s (streak=%d, total=%d): %s',
                        kind, streak, total, detail)

    def _native_accepted(self, kind):
        with self._result_lock:
            self._native_rejection_streaks[kind] = 0

    def _native_graph(self, msg, segment):
        try:
            if not msg.poses_id or len(msg.poses_id) != len(msg.poses):
                return
            indices = [i for i, node_id in enumerate(msg.poses_id) if node_id > 0]
            if not indices:
                return
            index = max(indices, key=lambda i: msg.poses_id[i])
            pose = _native_pose(msg.poses[index])
            stamp = self._stamp(msg)
            nodes = {msg.poses_id[i] for i in indices}
            links = [(link.from_id, link.to_id, link.type) for link in getattr(msg, 'links', ())]
            with self._result_lock:
                segment['graph'] = (stamp, msg.header.frame_id, pose)
                segment['progress'].graph(stamp, self._receipt_clock(), nodes, links)
            self._publish_snapshot(segment)
            self._native_accepted('graph')
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self._native_rejected('graph', error)

    def _native_cloud(self, msg, segment):
        try:
            stamp, frame, points = decode_cloud(msg, max_points=self._settings['preview_max_points'], sample=True)
            with self._result_lock:
                segment['cloud'] = (stamp, frame, points)
            self._publish_snapshot(segment)
            self._native_accepted('cloud')
        except ValueError as error:
            self._native_rejected('cloud', error)

    def _publish_snapshot(self, segment):
        from .pointcloud_slam_preview import encode_mapping
        with self._result_lock:
            graph, cloud = segment['graph'], segment['cloud']
            if graph is None or cloud is None or graph[:2] != cloud[:2]:
                return
            _, frame, pose = graph
            if not isinstance(frame, str) or not frame.strip():
                return
            packet = encode_mapping(cloud[2], pose, max_points=self._settings['preview_max_points'],
                                    pose_reference=segment.get('pose_reference'))
            segment['snapshot'], segment['frame_id'] = packet, frame
            if self._active is not segment or self._closed:
                return
            now = time.monotonic()
            if now - segment['last_preview'] < self._settings['preview_interval_s']:
                return
            segment['last_preview'] = now
            message = UInt8MultiArray(); message.data = array('B', packet)
            self._map_pub.publish(message)

    def _native_info(self, msg, segment):
        with self._result_lock:
            values = dict(zip(getattr(msg, 'stats_keys', []), getattr(msg, 'stats_values', [])))
            recovery = [values.get(key) for key in
                ('Motus/RecoveryState/', 'Motus/Relocalizations/', 'Motus/VisualInliers/')]
            valid_ref = type(msg.ref_id) is int and msg.ref_id > 0
            has_recovery_state = False
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in recovery):
                states = ('external', 'searching', 'localized', 'visual_tracking', 'inertial_registration', 'scan_tracking')
                if recovery[0] in range(len(states)):
                    has_recovery_state = True
                    segment['native_recovery'] = dict(state=states[int(recovery[0])],
                        relocalizations=int(recovery[1]), visual_inliers=int(recovery[2]))
                    reacquired = values.get('Motus/VisualReacquiredInliers/')
                    if recovery[0] in (0, 3) and isinstance(reacquired, (int, float)) and 0 < reacquired <= recovery[2]:
                        segment['native_recovery']['visual_reacquired_inliers'] = int(reacquired)
                    match = [values.get(key) for key in
                        ('Motus/RecoveryNode/', 'Motus/RecoveryAnchor/', 'Motus/RecoveryInliers/')]
                    if (recovery[0] == 2 and all(isinstance(v, (int, float)) and
                            math.isfinite(v) and v > 0 for v in match[:2]) and match[0] == msg.ref_id):
                        correspondences = values.get('Motus/ScanRelocalizationCorrespondences/')
                        if isinstance(correspondences, (int, float)) and math.isfinite(correspondences) and correspondences > 0:
                            segment['native_recovery'].update(relocalization_node=int(match[0]),
                                relocalization_anchor=int(match[1]), relocalization_source='scan',
                                relocalization_correspondences=int(correspondences))
                        elif isinstance(match[2], (int, float)) and math.isfinite(match[2]) and match[2] > 0:
                            segment['native_recovery'].update(relocalization_node=int(match[0]),
                                relocalization_anchor=int(match[1]), relocalization_inliers=int(match[2]))
            if valid_ref or has_recovery_state:
                try:
                    segment['progress'].activity(self._stamp(msg),self._receipt_clock())
                except AttributeError:
                    pass  # No stamped activity evidence in this native message.
            if not valid_ref:
                return
            if segment is self._active and segment['native_recovery']['state'] == 'external':
                self._root_wait_done.set()
                self._wake.set()
            identity = (msg.ref_id, msg.loop_closure_id, msg.proximity_detection_id)
            frame = getattr(getattr(msg, 'header', None), 'frame_id', None)
            if isinstance(frame, str) and frame.strip():
                segment['frame_id'] = frame
            if identity == segment['last_info']:
                return
            if segment['last_info'] is None or segment['last_info'][0] != msg.ref_id:
                segment['native_updates'] += 1
            if type(msg.loop_closure_id) is int and msg.loop_closure_id > 0:
                segment['native_loop_closures'] += 1
            if type(msg.proximity_detection_id) is int and msg.proximity_detection_id > 0:
                segment['native_proximity_detections'] += 1
            segment['last_info'] = identity

    def status(self):
        with self._lock:
            health = self.stream.status(now=self._receipt_clock(), measurement_now_ns=self._measurement_now())
            queued, stopped, forwarded, dropped = len(self._queue), self._stopped(), self._forwarded, self._dropped
            motion_pending = len(self._motion_waiting)
            samples = list(self._rgb_preparation_ms)
            rgb_preparation = dict(budget_ms=_RGB_WAIT_MS, measured_budget_ms=_MEASURED_RGB_WAIT_MS,
                first_map_budget_ms=_FIRST_MAP_RGB_WAIT_MS,
                timeouts=self._rgb_timeouts,
                published_images=self._rgb_published, sample_count=len(samples),
                p95_ms=float(np.percentile(samples, 95)) if samples else None,
                max_ms=max(samples) if samples else None)
            if self._capture_clouds is not None:
                health['capture_mount'] = self._capture_clouds.mounts.info(self._measurement_clock_ns())
                health['waiting_for_capture_mount'] = len(self._capture_clouds.pending)
        with self._result_lock:
            counts = {key: sum(segment[key] for segment in self._segments) for key in
                      ('native_loop_closures', 'native_proximity_detections', 'native_updates')}
            active = self._active
            frame = active['frame_id'] if active else None
            segment = active['segment_id'] if active else None
            progress = active['progress'].status(self._receipt_clock()) if active else {'state': 'idle'}
            rejections = dict(self._native_rejections)
            streaks = dict(self._native_rejection_streaks)
        rgb_info = self._rgb.info() if self._rgb is not None else {'state': 'disconnected', 'reason': None}
        if self._rgb_mode == 'pointcloud' and self._rgb_reason:
            rgb_info.update(state='fallback', reason=self._rgb_reason)
        return dict(health, backend='rtabmap', loop_closure_mode=self._rgb_mode, rgb_input=rgb_info,
            temporal_motion=self._motion.status() if self._motion is not None else {'state':'disconnected'},
            temporal_pending=motion_pending,
            rgb_preparation=rgb_preparation,
            native_recovery=deepcopy(active['native_recovery']) if active else {'state': 'unavailable'},
            state='error' if self._runtime_error else
            'storage_limit' if self._stop_reason else 'stopped' if stopped else health['odometry'],
            capture_stopped=stopped, stop_reason=self._stop_reason,
            accepted_clouds=health['admitted_frames'], forwarded_clouds=forwarded,
            upstream_covariance='known' if self._covariance_source=='upstream' else 'unknown',
            covariance_source=self._covariance_source or 'unknown',
            backend_progress=progress,
            segment_lifecycle=[dict(segment_id=s['segment_id'],database=str(s['process'].database_path),
                state='finalized' if s['finalized'] else 'closing' if s['closing'] else 'active',
                relation_to_previous='unknown_separate_origin') for s in self._segments],
            map_decision=self._map_decision, runtime_queue=queued, dropped_runtime_queue=dropped, segment_id=segment,
            frame_id=frame, disk_bytes=self._disk_bytes, **counts,
            native_graph_rejections=rejections['graph'],
            native_cloud_rejections=rejections['cloud'],
            native_graph_rejection_streak=streaks['graph'],
            native_cloud_rejection_streak=streaks['cloud'],
            accepted_loops=counts['native_loop_closures'] + counts['native_proximity_detections'],
            last_error=self._runtime_error or self._stop_reason or health['last_error'])

    def outputs(self):
        return [{'topic': self._map_topic, 'format': 'sensor/mapping',
                 'desc': 'Live or saved RTAB-Map preview'}]

    def close(self):
        self.freeze_capture()
        if self._motion is not None:self._motion.close()
        if self._rgb is not None:
            self._rgb.close()
        for lease in self._input_leases:
            lease.close()
        self._worker_stop.set(); self._status_stop.set(); self._wake.set()
        with self._close_lock:
            if self._closed:
                return
            self._start_done.wait()
            for worker in (self._worker, self._status_worker):
                if worker and worker is not threading.current_thread():
                    worker.join()
            self._rgb_task = None
            for segment in self._segments:
                if not segment['finalized']:
                    segment['closing'] = True
                    segment['process'].close()
                    segment['finalized'] = True
            if self._runtime_error:
                raise RuntimeError(self._runtime_error)
            self._closed = True

    def artifacts(self):
        if not self._closed:
            raise RuntimeError('Close RTAB-Map before reading finalized map artifacts')
        with self._result_lock:
            pending = [(segment['segment_id'], segment['frame_id'],
                        segment['process'].database_path, segment['snapshot'])
                       for segment in self._segments if segment['forwarded'] and
                       getattr(segment['process'], 'mapped_nodes', None) != 0]
        segments = []
        for segment_id, frame_id, database, snapshot in pending:
            if snapshot is not None:
                snapshot = rebuild_snapshot(database, max_points=self._settings['preview_max_points'],
                                            fallback=snapshot)
            segments.append({'segment_id': segment_id, 'frame_id': frame_id,
                             'database': str(database), 'snapshot': snapshot})
        return {'segments': segments, 'statistics': self.status()}
