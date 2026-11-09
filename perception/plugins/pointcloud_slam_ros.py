"""Native RTAB-Map gateway for admitted PointCloud2 and existing external VO."""
import json
import hashlib
import math
import threading
import time
import logging
from collections import deque
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid
from copy import deepcopy

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray

from .pointcloud_slam_rtabmap import RtabmapProcess
from .pointcloud_slam_snapshot import rebuild as rebuild_snapshot
from .pointcloud_slam_stream import SlamStream
from .pointcloud_slam_transforms import rigid, pose_matrix


_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                  history=HistoryPolicy.KEEP_LAST, depth=2,
                  durability=DurabilityPolicy.VOLATILE)
log = logging.getLogger(__name__)


def discover_input_bindings(input_topics):
    """Resolve native publisher types, allowing one second for DDS discovery."""
    if (not isinstance(input_topics, list) or len(input_topics) != 2 or
            any(not isinstance(topic, str) or not topic.strip() for topic in input_topics) or
            len(set(input_topics)) != 2):
        raise ValueError('two distinct native ROS input topics required')
    formats = {'sensor_msgs/msg/PointCloud2': 'sensor/pointcloud2',
               'std_msgs/msg/String': 'state/odom'}
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
            if len(bindings) == 2:
                if set(bindings.values()) != {'state/odom', 'sensor/pointcloud2'}:
                    raise ValueError('native ROS inputs require one String pose and one PointCloud2 cloud')
                return bindings
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = ', '.join(topic for topic in input_topics if topic not in bindings)
                raise ValueError('native ROS input discovery timed out: ' + missing)
            time.sleep(min(.05, remaining))
    finally:
        node.destroy_node()


def decode_cloud(msg, *, max_points, sample=False):
    """Read finite XYZ float32 points, including padded and organized clouds."""
    try:
        stamp = msg.header.stamp
        sec, nsec = stamp.sec, stamp.nanosec
        frame = msg.header.frame_id
        width, height, step, row = msg.width, msg.height, msg.point_step, msg.row_step
        if (type(sec) is not int or type(nsec) is not int or sec < 0 or
                not 0 <= nsec < 1_000_000_000 or sec * 1_000_000_000 + nsec <= 0 or
                not isinstance(frame, str) or not frame.strip()):
            raise ValueError('cloud needs capture timestamp and frame')
        if (any(type(v) is not int or v <= 0 for v in (width, height, step, row)) or
                (not sample and width * height > max_points) or row < width * step or len(msg.data) < row * height):
            raise ValueError('invalid cloud dimensions or capacity')
        fields = {}
        for field in msg.fields:
            if field.name in ('x', 'y', 'z'):
                if (field.name in fields or field.datatype != PointField.FLOAT32 or
                        field.count != 1 or type(field.offset) is not int or
                        field.offset < 0 or field.offset + 4 > step):
                    raise ValueError('unsupported XYZ fields')
                fields[field.name] = field.offset
        offsets = sorted(fields.values())
        if (set(fields) != {'x', 'y', 'z'} or
                any(b - a < 4 for a, b in zip(offsets, offsets[1:]))):
            raise ValueError('missing or overlapping XYZ fields')
        raw = memoryview(msg.data)
        dtype = '>f4' if msg.is_bigendian else '<f4'
        indices = np.linspace(0, width * height - 1, min(max_points, width * height), dtype=int)
        points = np.stack([np.ndarray((height, width), dtype=dtype, buffer=raw,
                                     offset=fields[key], strides=(row, step))[indices // width, indices % width]
                           for key in ('x', 'y', 'z')], axis=1)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError('cloud has no finite points')
        return sec * 1_000_000_000 + nsec, frame, points
    except (AttributeError, TypeError, BufferError) as error:
        raise ValueError('invalid PointCloud2') from error


_MAP_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
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

    Each external VO segment has a private ROS graph and native database.
    Process startup, shutdown, disk checks and map conversion never hold the
    capture lock. freeze_capture therefore remains a quick publication barrier.
    """
    def __init__(self, odom_topic, cloud_topic, settings, *, session_dir,
                 input_contract=None, capture_stop=None):
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        from rtabmap_msgs.msg import Info, MapGraph
        self._Odometry, self._TransformStamped, self._TFMessage = Odometry, TransformStamped, TFMessage
        self._Info, self._MapGraph = Info, MapGraph
        self._settings = deepcopy(settings)
        self._settings.update((input_contract or {}).get('settings', {}))
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
        self._segments, self._active = [], None
        self._worker = self._status_worker = None
        self._started = self._closed = False
        self._start_done = threading.Event()
        self._start_done.set()
        self._runtime_error = ''
        self._stop_reason = ''
        self._forwarded = self._dropped = 0
        self._disk_bytes = 0
        self._session_dir = Path(session_dir).resolve()
        self._namespace = '/pointcloud_slam/session_' + uuid.uuid4().hex
        self._map_topic = cloud_topic + '/slam_odom_preview'
        self._status_topic = cloud_topic + '/slam_odom_preview_status'
        self._receipt_clock = time.monotonic
        super().__init__('pointcloud_slam_gateway')
        self._measurement_clock_ns = lambda: getattr(self.get_clock().now(), 'nanoseconds', None)
        keys = ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud', 'max_gap_ms',
                'max_age_s', 'max_pending', 'max_cloud_points', 'max_measurement_age_ms', 'max_future_skew_ms')
        try:
            self.stream = SlamStream(**{key: self._settings[key] for key in keys},
                extrinsic_mode=self._settings.get('extrinsic_mode', 'static'),
                input_contract=input_contract, on_frame=self._queue_frame)
            self._map_pub = self.create_publisher(UInt8MultiArray, self._map_topic, _QOS)
            self._status_pub = self.create_publisher(String, self._status_topic, _QOS)
            self.create_subscription(String, odom_topic, self._odom, _QOS)
            self.create_subscription(PointCloud2, cloud_topic, self._cloud, _QOS)
            if input_contract:
                self.create_subscription(String, input_contract['frame_metadata_topic'], self._cloud_metadata, _QOS)
        except Exception:
            self.destroy_node()
            raise

    def _stopped(self):
        return self._capture_stop.is_set() or self._project_stop.is_set()

    def _measurement_now(self):
        return self._measurement_clock_ns() if self.stream.max_measurement_age_ms is not None else None

    def _odom(self, msg):
        try:
            sample = json.loads(msg.data)
        except (TypeError, ValueError):
            sample = None
        with self._lock:
            if not self._stopped():
                self.stream.odom(sample, now=self._receipt_clock(), measurement_now_ns=self._measurement_now())
                if self.stream._odom_state != 'tracking':
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

    def _cloud(self, msg):
        if self._stopped():
            return
        try:
            stamp, frame, _ = decode_cloud(msg, max_points=self.stream.max_cloud_points)
            # The pinned native converter reads float pointers and tightly packed rows.
            if msg.is_bigendian:
                raise ValueError('RTAB-Map input requires little-endian PointCloud2')
            if msg.row_step != msg.width * msg.point_step:
                raise ValueError('RTAB-Map input cannot contain PointCloud2 row padding')
            if len(msg.data) != msg.row_step * msg.height:
                raise ValueError('RTAB-Map input data size must exactly match row_step * height')
            digest = hashlib.sha256(bytes(msg.data)).hexdigest() if self.stream.input_contract else None
            with self._lock:
                if not self._stopped():
                    self.stream.cloud(stamp, frame, msg, point_count=msg.width*msg.height,
                        now=self._receipt_clock(), measurement_now_ns=self._measurement_now(), data_sha256=digest)
        except ValueError as error:
            with self._lock:
                self.stream.last_error = str(error)
        self._wake.set()

    def _queue_frame(self, stamp, cloud, pose, matrix, context):
        if len(self._queue) >= self.stream.max_pending:
            self._queue.popleft()
            self._dropped += 1
        self._queue.append((stamp, cloud, pose, matrix, self._receipt_clock()))
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
                self._worker.start()
                self._status_worker.start()
        finally:
            self._start_done.set()

    def freeze_capture(self):
        self._capture_stop.set()
        with self._lock:
            self._queue.clear()
        self._wake.set()

    def _check_budget(self):
        used = sum(path.stat().st_size for path in self._session_dir.rglob('*') if path.is_file())
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
        process = RtabmapProcess(self._session_dir / ('segment_' + str(index)) / 'rtabmap.db', namespace, self._settings)
        segment = dict(segment_id=pose.segment_id, frame_id='', source_frame_id=pose.frame_id,
            process=process, snapshot=None, graph=None, cloud=None, finalized=False, closing=False,
            native_loop_closures=0, native_proximity_detections=0, native_updates=0,
            last_info=None, subscriptions=[], last_preview=0., forwarded=0)
        with self._result_lock:
            self._segments.append(segment)
            self._active = segment
        segment['odom_pub'] = self.create_publisher(self._Odometry, namespace + '/odom', _NATIVE_QOS)
        segment['cloud_pub'] = self.create_publisher(PointCloud2, namespace + '/scan_cloud', _NATIVE_QOS)
        segment['tf_pub'] = self.create_publisher(self._TFMessage,
            namespace + ('/tf' if self.stream.extrinsic_mode == 'dynamic' else '/tf_static'),
            _NATIVE_QOS if self.stream.extrinsic_mode == 'dynamic' else _MAP_QOS)
        for message, suffix, callback, qos in [
                (PointCloud2, '/cloud_map', self._native_cloud, _MAP_QOS),
                (self._MapGraph, '/mapGraph', self._native_graph, _MAP_QOS),
                (self._Info, '/info', self._native_info, _NATIVE_QOS)]:
            segment['subscriptions'].append(self.create_subscription(message, namespace + suffix,
                lambda msg, target=segment, handler=callback: handler(msg, target), qos))
        process.start()
        deadline = time.monotonic() + 10.
        while not self._stopped():
            if process.poll() is not None:
                raise RuntimeError('RTAB-Map exited during startup (exit ' + str(process.poll()) + ')')
            if segment['odom_pub'].get_subscription_count() and segment['cloud_pub'].get_subscription_count():
                return segment
            if time.monotonic() >= deadline:
                raise RuntimeError('RTAB-Map input discovery timed out')
            self._capture_stop.wait(.05)
        return segment

    def _publish_pair(self, segment, item):
        stamp, cloud, pose, matrix, received = item
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
        for key, value in zip(('x', 'y', 'z'), pose.position):
            setattr(odom.pose.pose.position, key, value)
        for key, value in zip(('x', 'y', 'z', 'w'), pose.quaternion):
            setattr(odom.pose.pose.orientation, key, value)
        if pose.covariance is not None:
            odom.pose.covariance = list(pose.covariance)
        cloud.header.frame_id = transform.child_frame_id
        with self._lock:
            if self._stopped() or self._receipt_clock() - received > self.stream.max_age_s:
                return
            now_ns = self._measurement_now()
            self.stream._check_measurement_time(stamp, now_ns)
            health = self.stream.status(now=self._receipt_clock(), measurement_now_ns=now_ns)
            if (health['odometry'] != 'tracking' or health['integration'] != 'admitting' or
                    not self.stream._poses or self.stream._poses[-1].segment_id != pose.segment_id):
                return
            segment['tf_pub'].publish(tf)
            segment['odom_pub'].publish(odom)
            segment['cloud_pub'].publish(cloud)
            self._forwarded += 1
            segment['forwarded'] += 1

    def _run(self):
        while not self._worker_stop.is_set() and not self._stopped():
            self._wake.wait(.25)
            self._wake.clear()
            with self._lock:
                item = self._queue.popleft() if self._queue and not self._stopped() else None
                if self._queue:
                    self._wake.set()
            if item is None:
                continue
            try:
                if self._active is None or self._active['segment_id'] != item[2].segment_id:
                    segment = self._activate(item[2])
                else:
                    segment = self._active
                if segment is None:
                    continue
                if segment['process'].poll() is not None:
                    raise RuntimeError('RTAB-Map process exited (exit ' + str(segment['process'].poll()) + ')')
                self._publish_pair(segment, item)
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
            message = String(); message.data = json.dumps(self.status(), allow_nan=False)
            self._status_pub.publish(message)

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _native_graph(self, msg, segment):
        try:
            if not msg.poses_id or len(msg.poses_id) != len(msg.poses):
                return
            indices = [i for i, node_id in enumerate(msg.poses_id) if node_id > 0]
            if not indices:
                return
            index = max(indices, key=lambda i: msg.poses_id[i])
            pose = _native_pose(msg.poses[index])
            with self._result_lock:
                segment['graph'] = (self._stamp(msg), msg.header.frame_id, pose)
            self._publish_snapshot(segment)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            log.warning('Invalid native RTAB-Map graph: %s', error)

    def _native_cloud(self, msg, segment):
        try:
            stamp, frame, points = decode_cloud(msg, max_points=self._settings['preview_max_points'], sample=True)
            with self._result_lock:
                segment['cloud'] = (stamp, frame, points)
            self._publish_snapshot(segment)
        except ValueError as error:
            log.warning('Invalid native RTAB-Map cloud: %s', error)

    def _publish_snapshot(self, segment):
        from .pointcloud_slam_preview import encode_mapping
        with self._result_lock:
            graph, cloud = segment['graph'], segment['cloud']
            if graph is None or cloud is None or graph[:2] != cloud[:2]:
                return
            _, frame, pose = graph
            if not isinstance(frame, str) or not frame.strip():
                return
            packet = encode_mapping(cloud[2], pose, max_points=self._settings['preview_max_points'])
            segment['snapshot'], segment['frame_id'] = packet, frame
            if self._active is not segment or self._closed:
                return
            now = time.monotonic()
            if now - segment['last_preview'] < self._settings['preview_interval_s']:
                return
            segment['last_preview'] = now
            message = UInt8MultiArray(); message.data = packet
            self._map_pub.publish(message)

    def _native_info(self, msg, segment):
        if type(msg.ref_id) is not int or msg.ref_id <= 0:
            return
        identity = (msg.ref_id, msg.loop_closure_id, msg.proximity_detection_id)
        with self._result_lock:
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
        with self._result_lock:
            counts = {key: sum(segment[key] for segment in self._segments) for key in
                      ('native_loop_closures', 'native_proximity_detections', 'native_updates')}
            active = self._active
            frame = active['frame_id'] if active else None
            segment = active['segment_id'] if active else None
        return dict(health, backend='rtabmap', state='error' if self._runtime_error else
            'storage_limit' if self._stop_reason else 'stopped' if stopped else health['odometry'],
            capture_stopped=stopped, stop_reason=self._stop_reason,
            accepted_clouds=health['admitted_frames'], forwarded_clouds=forwarded,
            runtime_queue=queued, dropped_runtime_queue=dropped, segment_id=segment,
            frame_id=frame, disk_bytes=self._disk_bytes, **counts,
            accepted_loops=counts['native_loop_closures'] + counts['native_proximity_detections'],
            last_error=self._runtime_error or self._stop_reason or health['last_error'])

    def outputs(self):
        return [{'topic': self._map_topic, 'format': 'sensor/mapping', 'desc': 'Map snapshot; marker is last mapped pose'},
                {'topic': self._status_topic, 'format': 'data/json'}]

    def close(self):
        self.freeze_capture()
        self._worker_stop.set(); self._status_stop.set(); self._wake.set()
        with self._close_lock:
            if self._closed:
                return
            self._start_done.wait()
            for worker in (self._worker, self._status_worker):
                if worker and worker is not threading.current_thread():
                    worker.join()
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
