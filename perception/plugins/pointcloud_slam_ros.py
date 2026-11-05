"""ROS input adapter for the experimental, odometry-only SLAM graph."""
import json
import math
import threading
import time
import logging

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray


_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                  history=HistoryPolicy.KEEP_LAST, depth=2,
                  durability=DurabilityPolicy.VOLATILE)
log = logging.getLogger(__name__)


def decode_cloud(msg, *, max_points):
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
                width * height > max_points or row < width * step or len(msg.data) < row * height):
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
        points = np.stack([np.ndarray((height, width), dtype=dtype, buffer=raw,
                                     offset=fields[key], strides=(row, step)).reshape(-1)
                           for key in ('x', 'y', 'z')], axis=1)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError('cloud has no finite points')
        return sec * 1_000_000_000 + nsec, frame, points
    except (AttributeError, TypeError, BufferError) as error:
        raise ValueError('invalid PointCloud2') from error


class SlamInputNode(Node):
    """Subscribe only to explicitly resolved topics; does not control producers."""

    def __init__(self, odom_topic, cloud_topic, stream, *, clock=time.monotonic,
                 measurement_clock_ns=None, preview=None):
        if preview is not None:
            if not isinstance(preview, dict):
                raise ValueError('invalid diagnostic preview settings')
            interval, count = preview.get('interval_s'), preview.get('max_points')
            if (not isinstance(interval, (int, float)) or isinstance(interval, bool) or
                    not math.isfinite(interval) or not .1 <= interval <= 60. or
                    type(count) is not int or not 0 < count <= 80000 or
                    preview.get('backend') not in ('raw', 'gtsam') or
                    any(not isinstance(preview.get(k), str) or not preview[k].strip()
                        for k in ('topic', 'status_topic')) or preview['topic'] == preview['status_topic']):
                raise ValueError('invalid diagnostic preview settings')
            if preview['backend'] == 'gtsam':
                sigmas = np.asarray(preview.get('sigmas'), dtype=float)
                if sigmas.shape != (6,) or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
                    raise ValueError('gtsam preview requires six positive noise sigmas')
                import gtsam  # noqa: F401 — fail startup if target dependency is unavailable
        super().__init__('pointcloud_slam_input')
        self.stream = stream
        self._clock = clock
        self._measurement_clock_ns = measurement_clock_ns or (lambda: self.get_clock().now().nanoseconds)
        self._lock = threading.Lock()
        self._preview = preview
        self._preview_stop = threading.Event()
        self._preview_wake = threading.Event()
        self._preview_worker = None
        self._preview_errors = 0
        try:
            if preview:
                self._preview_pub = self.create_publisher(UInt8MultiArray, preview['topic'], _QOS)
                self._preview_status_pub = self.create_publisher(String, preview['status_topic'], _QOS)
            self.create_subscription(String, odom_topic, self._odom, _QOS)
            self.create_subscription(PointCloud2, cloud_topic, self._cloud, _QOS)
        except Exception:
            self.destroy_node()
            raise

    def _odom(self, msg):
        try:
            sample = json.loads(msg.data)
        except (TypeError, ValueError):
            sample = None
        with self._lock:
            measurement_now = (self._measurement_clock_ns()
                               if self.stream.max_measurement_age_ms is not None else None)
            self.stream.odom(sample, now=self._clock(), measurement_now_ns=measurement_now)
        self._preview_wake.set()

    def _cloud(self, msg):
        try:
            stamp, frame, points = decode_cloud(msg, max_points=self.stream.max_cloud_points)
            with self._lock:
                measurement_now = (self._measurement_clock_ns()
                                   if self.stream.max_measurement_age_ms is not None else None)
                self.stream.cloud(stamp, frame, points, now=self._clock(),
                                  measurement_now_ns=measurement_now)
        except ValueError as error:
            with self._lock:
                self.stream.last_error = str(error)
        self._preview_wake.set()

    def start_preview(self):
        if self._preview and self._preview_worker is None:
            self._preview_worker = threading.Thread(target=self._preview_loop, daemon=True,
                                                     name='pointcloud_slam_preview')
            self._preview_worker.start()
            self._preview_wake.set()

    def stop_preview(self):
        self._preview_stop.set()
        self._preview_wake.set()
        if self._preview_worker:
            self._preview_worker.join(timeout=3.)
            if self._preview_worker.is_alive():
                log.warning('SLAM preview worker still stopping')
        # A publisher already in progress holds this lock; wait before node disposal.
        with self._lock:
            pass

    def _preview_loop(self):
        last = 0.
        while not self._preview_stop.is_set():
            self._preview_wake.wait(timeout=self._preview['interval_s'])
            self._preview_wake.clear()
            if self._preview_stop.is_set():
                break
            delay = self._preview['interval_s'] - (time.monotonic() - last)
            if delay > 0 and self._preview_stop.wait(delay):
                break
            try:
                self.publish_preview_once()
                self._preview_errors = 0
            except Exception as error:
                with self._lock:
                    self.stream.last_error = 'preview error: ' + str(error)[:200]
                    if not self._preview_stop.is_set():
                        message = String()
                        message.data = json.dumps({'kind': 'unvalidated_odometry_preview',
                            'state': 'error', 'accepted_loops': 0,
                            'last_error': self.stream.last_error}, allow_nan=False)
                        self._preview_status_pub.publish(message)
                self._preview_errors += 1
                if self._preview_errors == 1 or self._preview_errors % 100 == 0:
                    log.exception('SLAM preview failed (%d)', self._preview_errors)
            last = time.monotonic()

    def publish_preview_once(self):
        """Publish a bounded diagnostic snapshot, never an accepted SLAM map."""
        from .pointcloud_slam_map import rebuild_segment
        from .pointcloud_slam_preview import encode_preview
        preview_subscribed = self._preview_pub.get_subscription_count() > 0
        with self._lock:
            measurement_now = (self._measurement_clock_ns()
                               if self.stream._poses and self.stream.max_measurement_age_ms is not None else None)
            status = self.stream.status(now=self._clock(), measurement_now_ns=measurement_now)
            segments = self.stream.store.segments
            active = self.stream._poses[-1].segment_id if self.stream._poses else None
            if (status['odometry'] != 'tracking' or status['cloud'] != 'active' or
                    status['integration'] != 'collecting_unverified' or
                    not segments or segments[-1].segment_id != active):
                graph = segment = None
            else:
                segment = segments[-1]
                graph = self.stream.graph()[-1] if preview_subscribed else None
        result = {'kind': 'unvalidated_odometry_preview',
                  'state': 'stale' if status['odometry'] == 'tracking' and status['cloud'] == 'stale' else status['odometry'],
                  'cloud_state': status['cloud'],
                  'integration': status['integration'], 'accepted_loops': 0,
                  'segment_id': segment.segment_id if segment else None,
                  'frame_id': segment.frames[0].pose.frame_id if segment else None,
                  'stamp_ns': str(segment.frames[-1].cloud_stamp_ns) if segment else None,
                  'backend': self._preview['backend'], 'last_error': status['last_error']}
        packet = None
        if segment is not None and preview_subscribed:
            poses = graph['poses']
            if self._preview['backend'] == 'gtsam':
                from .pointcloud_slam_backend import optimize_odometry_graph
                poses = optimize_odometry_graph(graph, sigmas=self._preview['sigmas'])
            packet = encode_preview(rebuild_segment(segment, poses), poses[-1],
                                    max_points=self._preview['max_points'], metadata=result)
        with self._lock:
            if self._preview_stop.is_set():
                return
            measurement_now = (self._measurement_clock_ns()
                               if self.stream._poses and self.stream.max_measurement_age_ms is not None else None)
            latest_status = self.stream.status(now=self._clock(), measurement_now_ns=measurement_now)
            if (latest_status['odometry'] != 'tracking' or latest_status['cloud'] != 'active' or
                    latest_status['integration'] != 'collecting_unverified'):
                packet = None
                state = ('stale' if latest_status['odometry'] == 'tracking' and latest_status['cloud'] == 'stale'
                         else latest_status['odometry'])
                result.update(state=state, cloud_state=latest_status['cloud'], integration=latest_status['integration'],
                              last_error=latest_status['last_error'], segment_id=None,
                              frame_id=None, stamp_ns=None)
            # Do not publish a snapshot after its segment/frame changed during work.
            current = self.stream.store.segments
            current_active = self.stream._poses[-1].segment_id if self.stream._poses else None
            if segment is not None and latest_status['odometry'] == 'tracking' and (
                    not current or current_active != segment.segment_id or
                    current[-1].frames[-1].cloud_stamp_ns != segment.frames[-1].cloud_stamp_ns):
                self._preview_wake.set()
                return
            if packet is not None:
                msg = UInt8MultiArray()
                msg.data = packet
                self._preview_pub.publish(msg)
            message = String()
            message.data = json.dumps(result, allow_nan=False)
            self._preview_status_pub.publish(message)

    def optimize_odometry(self, *, sigmas):
        """Return experimental per-segment snapshots; publish nothing."""
        from .pointcloud_slam_backend import optimize_odometry_graph
        from .pointcloud_slam_map import rebuild_segment
        with self._lock:
            graphs = self.stream.graph()
            segments = self.stream.store.segments
        result = []
        for graph, segment in zip(graphs, segments):
            poses = optimize_odometry_graph(graph, sigmas=sigmas)
            result.append({'segment_id': segment.segment_id, 'poses': poses,
                           'frame_id': segment.frames[0].pose.frame_id,
                           'last_cloud_stamp_ns': segment.frames[-1].cloud_stamp_ns,
                           'map_points': rebuild_segment(segment, poses)})
        return result

    def status(self):
        with self._lock:
            measurement_now = (self._measurement_clock_ns()
                               if self.stream._poses and self.stream.max_measurement_age_ms is not None else None)
            return self.stream.status(now=self._clock(), measurement_now_ns=measurement_now)
