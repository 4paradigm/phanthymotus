"""ROS input adapter for the experimental, odometry-only SLAM graph."""
import json
import hashlib
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
        loop_search = None
        if preview is not None:
            if not isinstance(preview, dict):
                raise ValueError('invalid diagnostic preview settings')
            interval, count = preview.get('interval_s'), preview.get('max_points')
            if (not isinstance(interval, (int, float)) or isinstance(interval, bool) or
                    not math.isfinite(interval) or not .1 <= interval <= 60. or
                    type(count) is not int or not 0 < count <= 80000 or
                    preview.get('backend') not in ('raw', 'gtsam', 'geometry_experimental') or
                    any(not isinstance(preview.get(k), str) or not preview[k].strip()
                        for k in ('topic', 'status_topic')) or preview['topic'] == preview['status_topic']):
                raise ValueError('invalid diagnostic preview settings')
            if preview['backend'] == 'gtsam':
                sigmas = np.asarray(preview.get('sigmas'), dtype=float)
                if sigmas.shape != (6,) or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
                    raise ValueError('gtsam preview requires six positive noise sigmas')
                import gtsam  # noqa: F401 — fail startup if target dependency is unavailable
            if preview['backend'] == 'geometry_experimental':
                import gtsam  # noqa: F401
                import open3d  # noqa: F401
                from .pointcloud_slam_loop_search import ExperimentalLoopSearch
                if not isinstance(preview.get('geometry'), dict):
                    raise ValueError('explicit geometry settings required')
                loop_search = ExperimentalLoopSearch(**preview['geometry'])
            if type(preview.get('relocalization', False)) is not bool or (
                    preview.get('relocalization') and preview['backend'] != 'geometry_experimental'):
                raise ValueError('relocalization requires experimental geometry backend')
        super().__init__('pointcloud_slam_input')
        self.stream = stream
        self._receipt_clock = clock
        self._measurement_clock_ns = measurement_clock_ns or (lambda: self.get_clock().now().nanoseconds)
        self._lock = threading.Lock()
        self._preview = preview
        self._loop_search = loop_search
        self._loop_segment = None
        self._segment_links = []
        self._segment_loops = {}
        self._rolling_searches = {}
        self._history_snapshots = {}
        self._history_publishers = {}
        self._history_published_at = {}
        self._preview_stop = threading.Event()
        self._preview_wake = threading.Event()
        self._preview_worker = None
        self._preview_errors = 0
        self._compute_started = None
        self._status_worker = None
        try:
            if preview:
                self._preview_pub = self.create_publisher(UInt8MultiArray, preview['topic'], _QOS)
                self._preview_status_pub = self.create_publisher(String, preview['status_topic'], _QOS)
                if preview['backend'] == 'geometry_experimental':
                    self._map_pose_pub = self.create_publisher(String, preview['topic'] + '/pose', _QOS)
                    self._map_correction_pub = self.create_publisher(String, preview['topic'] + '/map_to_odom', _QOS)
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
            self.stream.odom(sample, now=self._receipt_clock(), measurement_now_ns=measurement_now)
        self._preview_wake.set()

    def _cloud(self, msg):
        try:
            stamp, frame, points = decode_cloud(msg, max_points=self.stream.max_cloud_points)
            with self._lock:
                measurement_now = (self._measurement_clock_ns()
                                   if self.stream.max_measurement_age_ms is not None else None)
                self.stream.cloud(stamp, frame, points, now=self._receipt_clock(),
                                  measurement_now_ns=measurement_now)
        except ValueError as error:
            with self._lock:
                self.stream.last_error = str(error)
        self._preview_wake.set()

    def start_preview(self):
        if self._preview and self._preview_worker is None:
            self._preview_worker = threading.Thread(target=self._preview_loop, daemon=True,
                                                     name='pointcloud_slam_preview')
            self._status_worker = threading.Thread(target=self._status_loop, daemon=True,
                                                   name='pointcloud_slam_status')
            self._status_worker.start()
            self._preview_worker.start()
            self._preview_wake.set()

    def stop_preview(self):
        already_stopped = self._preview_stop.is_set()
        self._preview_stop.set()
        self._preview_wake.set()
        if self._status_worker:
            self._status_worker.join(timeout=3.)
        if self._preview_worker:
            self._preview_worker.join(timeout=3.)
            if self._preview_worker.is_alive():
                log.warning('SLAM preview worker still stopping')
        # A publisher already in progress holds this lock; wait before node disposal.
        with self._lock:
            if (not already_stopped and self._preview and
                    self._preview['backend'] == 'geometry_experimental'):
                message = String()
                message.data = json.dumps({'kind': 'unvalidated_geometry_pose',
                    'state': 'stopped', 'pose': None, 'accepted_loops': 0})
                self._map_pose_pub.publish(message)
                message = String()
                message.data = json.dumps({'kind': 'unvalidated_map_to_odom',
                    'state': 'stopped', 'transform': None, 'accepted_loops': 0})
                self._map_correction_pub.publish(message)

    def _status_loop(self):
        # Geometry can be slow; input health must remain observable independently.
        while not self._preview_stop.wait(min(1., self._preview['interval_s'])):
            with self._lock:
                compute_started = self._compute_started
                if compute_started is None or self._preview_stop.is_set():
                    continue
                now_ns = (self._measurement_clock_ns() if self.stream._poses and
                          self.stream.max_measurement_age_ms is not None else None)
                status = self.stream.status(now=self._receipt_clock(), measurement_now_ns=now_ns)
                message = String()
                message.data = json.dumps({**status, 'kind': 'slam_compute_status',
                    'state': status['odometry'], 'computing': True,
                    'compute_elapsed_ms': (time.perf_counter() - compute_started) * 1000,
                    'accepted_loops': 0}, allow_nan=False)
                self._preview_status_pub.publish(message)

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
            self._compute_started = time.perf_counter()
            try:
                self.publish_preview_once()
                self._preview_errors = 0
            except Exception as error:
                with self._lock:
                    from .pointcloud_slam_paged_store import StorageError
                    if isinstance(error, StorageError):
                        self.stream.store.storage_error = str(error)
                    self.stream.last_error = 'preview error: ' + str(error)[:200]
                    if not self._preview_stop.is_set():
                        message = String()
                        message.data = json.dumps({'kind': ('unvalidated_geometry_preview'
                            if self._preview['backend'] == 'geometry_experimental' else 'unvalidated_odometry_preview'),
                            'state': 'error', 'accepted_loops': 0,
                            'last_error': self.stream.last_error}, allow_nan=False)
                        self._preview_status_pub.publish(message)
                        if self._preview['backend'] == 'geometry_experimental':
                            self._loop_search = None
                            self._loop_segment = None
                            message = String()
                            message.data = json.dumps({'kind': 'unvalidated_geometry_pose',
                                'state': 'error', 'pose': None, 'accepted_loops': 0})
                            self._map_pose_pub.publish(message)
                            message = String()
                            message.data = json.dumps({'kind': 'unvalidated_map_to_odom',
                                'state': 'error', 'transform': None, 'accepted_loops': 0})
                            self._map_correction_pub.publish(message)
                self._preview_errors += 1
                if self._preview_errors == 1 or self._preview_errors % 100 == 0:
                    log.exception('SLAM preview failed (%d)', self._preview_errors)
            finally:
                self._compute_started = None
            last = time.monotonic()

    def history_topics(self):
        """Expose archived coordinate systems separately, never merge source segments."""
        if not self._preview or not self._preview.get('persistent_history'):
            return []
        with self._lock:
            active = (self.stream._poses[-1].segment_id if self.stream._poses and
                      not self.stream.store.requires_new_segment else None)
            topics = []
            for segment in self.stream.store.segments:
                if segment.segment_id == active:
                    continue
                topic = self._preview['topic'] + '/history/' + hashlib.sha256(
                    segment.segment_id.encode()).hexdigest()[:16]
                if topic not in self._history_publishers and not self._preview_stop.is_set():
                    self._history_publishers[topic] = self.create_publisher(UInt8MultiArray, topic, _QOS)
                topics.append({'topic': topic, 'format': 'sensor/slam-preview',
                               'segment_id': segment.segment_id})
            return topics

    def _publish_history_previews(self):
        from .pointcloud_slam_keyframes import Segment
        from .pointcloud_slam_stream import _matrix
        from .pointcloud_slam_map import rebuild_segment
        from .pointcloud_slam_preview import encode_preview
        for output in self.history_topics():
            topic, sid = output['topic'], output['segment_id']
            with self._lock:
                if self._preview_stop.is_set(): return
                pub = self._history_publishers[topic]
                if not pub.get_subscription_count():
                    self._history_published_at.pop(topic, None)
                    continue
                if time.monotonic() - self._history_published_at.get(topic, -10.) < 10.:
                    continue
                segment = next(s for s in self.stream.store.segments if s.segment_id == sid)
                snapshot = self._history_snapshots.get(sid)
            # A checkpoint may include a few inputs newer than its last optimized
            # snapshot. Display that snapshot exactly, without relabelling raw tail.
            if snapshot:
                poses = [np.asarray(p) for p in snapshot['poses']]
                segment = Segment(sid, segment.frames[:len(poses)])
            else:
                poses = [_matrix(f.pose) for f in segment.frames]
            metadata = {'segment_id': sid, 'frame_id': segment.frames[0].pose.frame_id + '_archive',
                        'stamp_ns': str(segment.frames[-1].cloud_stamp_ns),
                        'backend': 'geometry_experimental' if snapshot else 'raw',
                        'candidate_count': snapshot.get('candidate_count', 0) if snapshot else 0,
                        'state': 'historical_snapshot', 'input_validation': self._preview.get('input_validation')}
            points = rebuild_segment(segment, poses, max_points=self._preview['max_points'])
            packet = encode_preview(points, poses[-1], max_points=self._preview['max_points'], metadata=metadata)
            with self._lock:
                if self._preview_stop.is_set(): return
                message = UInt8MultiArray(); message.data = packet
                pub.publish(message)
                self._history_published_at[topic] = time.monotonic()

    def publish_preview_once(self):
        """Publish a bounded diagnostic snapshot, never an accepted SLAM map."""
        from .pointcloud_slam_map import rebuild_segment
        from .pointcloud_slam_preview import encode_preview
        started = time.perf_counter()
        self._publish_history_previews()
        preview_subscribed = self._preview_pub.get_subscription_count() > 0
        with self._lock:
            measurement_now = (self._measurement_clock_ns()
                               if self.stream._poses and self.stream.max_measurement_age_ms is not None else None)
            status = self.stream.status(now=self._receipt_clock(), measurement_now_ns=measurement_now)
            segments = self.stream.store.segments
            active = self.stream._poses[-1].segment_id if self.stream._poses else None
            if (status['odometry'] != 'tracking' or status['cloud'] != 'active' or
                    status['integration'] != 'collecting_unverified' or
                    not segments or segments[-1].segment_id != active):
                graph = segment = None
            else:
                segment = segments[-1]
                graphs = (self.stream.graph() if not self._preview.get('persistent_history') and
                          (preview_subscribed or self._preview['backend'] == 'geometry_experimental') else None)
                graph = graphs[-1] if graphs else {'poses': []}
        geometry = self._preview['backend'] == 'geometry_experimental'
        result = {'kind': 'unvalidated_geometry_preview' if geometry else 'unvalidated_odometry_preview',
                  'input_validation': self._preview.get('input_validation'),
                  'state': 'stale' if status['odometry'] == 'tracking' and status['cloud'] == 'stale' else status['odometry'],
                  'cloud_state': status['cloud'],
                  'integration': status['integration'], 'accepted_loops': 0,
                  'segment_id': segment.segment_id if segment else None,
                  'frame_id': segment.frames[0].pose.frame_id if segment else None,
                  'stamp_ns': str(segment.frames[-1].cloud_stamp_ns) if segment else None,
                  'backend': self._preview['backend'], 'last_error': status['last_error']}
        packet = None
        staged_links = None
        staged_loops = None
        staged_history = None
        if segment is None and geometry:
            self._loop_search = None
            self._loop_segment = None
        if segment is not None and (preview_subscribed or geometry):
            poses = graph['poses']
            if geometry:
                if self._preview.get('persistent_history'):
                    from .pointcloud_slam_rolling import RollingGeometryHistory
                    if segment.segment_id not in self._rolling_searches:
                        self._rolling_searches[segment.segment_id] = RollingGeometryHistory(
                            self._preview['geometry'], window_frames=self.stream.store._max_frames,
                            window_points=self.stream.store._max_points)
                    loop = self._rolling_searches[segment.segment_id].update(
                        segment, max_points=self._preview['max_points'])
                    staged_history = {'candidate_count': loop['candidate_count'],
                                      'poses': [p.tolist() for p in loop['poses']],
                                      'stamps': [f.cloud_stamp_ns for f in segment.frames]}
                    result.update(submaps=loop['submaps'], active_keyframes=loop['active_keyframes'],
                                  frozen_keyframes=loop['frozen_keyframes'],
                                  history_policy='frozen_submaps_no_global_loop_correction')
                else:
                    if self._loop_search is None or self._loop_segment != segment.segment_id:
                        from .pointcloud_slam_loop_search import ExperimentalLoopSearch
                        self._loop_search = ExperimentalLoopSearch(**self._preview['geometry'])
                        self._loop_segment = segment.segment_id
                    loop = self._loop_search.update(segment, graph, online=True,
                                                    max_points=self._preview['max_points'])
                poses = loop['poses']
                result['timings_ms'] = loop['timings_ms']
                result['descriptor_unindexed'] = loop['descriptor_unindexed']
                result.update(candidate_count=loop.get('candidate_count', len(loop['passed_candidates'])),
                              candidate_attempts=loop['candidate_attempts'],
                              descriptor_skips=loop['descriptor_skips'],
                              odom_frame=segment.frames[0].pose.frame_id,
                              frame_id=segment.frames[0].pose.frame_id + '_experimental_map',
                              map_pose=np.asarray(poses[-1]).tolist(),
                              map_from_odom=np.asarray(loop['map_from_odom']).tolist())
                points = loop['after_map']
                if self._preview.get('relocalization'):
                    staged_loops = dict(self._segment_loops)
                    staged_loops[segment.segment_id] = [
                        {'source': candidate['source'], 'target': candidate['target'],
                         'transform': candidate['diagnostic']['forward']['transform']}
                        for candidate in self._loop_search._graph._candidates]
                if self._preview.get('relocalization') and len(segments) > 1:
                    result['relocalization'] = 'searching'
                    links = list(self._segment_links)
                    connected = {segments[0].segment_id} | {link['source'] for link in links}
                    if segment.segment_id not in connected:
                        from .pointcloud_slam_relocalization import relocalize_pair
                        opts = self._preview['geometry']
                        for old, old_graph in zip(segments[:-1], graphs[:-1]):
                            if old.segment_id not in connected:
                                continue
                            merged = relocalize_pair(old, old_graph, segment, graph,
                                descriptor_voxel_m=opts['descriptor_voxel_m'],
                                max_candidates=opts['max_per_frame'],
                                thresholds=opts['thresholds'], odom_sigmas=opts['odom_sigmas'],
                                loop_sigmas=opts['loop_sigmas'], huber_k=opts['huber_k'],
                                voxel_m=opts['voxel_m'], max_distance_m=opts['max_distance_m'],
                                max_total_frames=self.stream.store._max_frames,
                                max_total_points=self.stream.store._max_points)
                            if merged['hypothesis_created']:
                                links.append({'target': old.segment_id, 'source': segment.segment_id,
                                    'target_index': merged['target_index'], 'source_index': merged['source_index'],
                                    'transform': merged['diagnostic']['forward']['transform']})
                                connected.add(segment.segment_id)
                                break
                    if segment.segment_id in connected:
                        from .pointcloud_slam_multigraph import optimize_multisegment
                        selected = [(s, g) for s, g in zip(segments, graphs) if s.segment_id in connected]
                        opts = self._preview['geometry']
                        merged = optimize_multisegment([s for s, _ in selected], [g for _, g in selected], links,
                            odom_sigmas=opts['odom_sigmas'], loop_sigmas=opts['loop_sigmas'],
                            huber_k=opts['huber_k'], max_total_frames=self.stream.store._max_frames,
                            max_total_points=self.stream.store._max_points,
                            segment_loops={key: value for key, value in staged_loops.items() if key in connected})
                        poses, points = merged['poses'][-1], merged['map_points']
                        staged_links = links
                        result.update(relocalization='unvalidated_hypothesis',
                            map_segments=merged['segment_ids'], segment_links=len(links),
                            frame_id=segments[0].frames[0].pose.frame_id + '_experimental_map',
                            map_pose=np.asarray(poses[-1]).tolist(),
                            map_from_odom=np.asarray(merged['map_from_active_odom']).tolist())
            elif self._preview['backend'] == 'gtsam':
                from .pointcloud_slam_backend import optimize_odometry_graph
                poses = optimize_odometry_graph(graph, sigmas=self._preview['sigmas'])
                points = rebuild_segment(segment, poses, max_points=self._preview['max_points'])
            else:
                points = rebuild_segment(segment, poses, max_points=self._preview['max_points'])
            if preview_subscribed:
                packet = encode_preview(points, poses[-1], max_points=self._preview['max_points'], metadata=result)
        result['compute_ms'] = (time.perf_counter() - started) * 1000
        with self._lock:
            if self._preview_stop.is_set():
                return
            measurement_now = (self._measurement_clock_ns()
                               if self.stream._poses and self.stream.max_measurement_age_ms is not None else None)
            latest_status = self.stream.status(now=self._receipt_clock(), measurement_now_ns=measurement_now)
            if (latest_status['odometry'] != 'tracking' or latest_status['cloud'] != 'active' or
                    latest_status['integration'] != 'collecting_unverified'):
                if geometry:
                    self._loop_search = None
                    self._loop_segment = None
                packet = None
                state = ('stale' if latest_status['odometry'] == 'tracking' and latest_status['cloud'] == 'stale'
                         else latest_status['odometry'])
                result.update(state=state, cloud_state=latest_status['cloud'], integration=latest_status['integration'],
                              last_error=latest_status['last_error'], segment_id=None,
                              frame_id=None, stamp_ns=None)
                result.pop('map_pose', None)
                result.pop('map_from_odom', None)
                result.pop('map_segments', None)
                result.pop('relocalization', None)
                result.pop('segment_links', None)
                staged_links = None
                staged_loops = None
                staged_history = None
            # Do not publish a snapshot after its segment/frame changed during work.
            current = self.stream.store.segments
            current_active = self.stream._poses[-1].segment_id if self.stream._poses else None
            if segment is not None and latest_status['odometry'] == 'tracking':
                if not current or current_active != segment.segment_id:
                    self._preview_wake.set()
                    return
                if current[-1].frames[-1].cloud_stamp_ns != segment.frames[-1].cloud_stamp_ns:
                    self._preview_wake.set()
                    if not geometry or measurement_now is None:
                        return
                if geometry and measurement_now is not None and result.get('map_pose') is not None:
                    try:
                        self.stream._check_measurement_time(segment.frames[-1].cloud_stamp_ns, measurement_now)
                    except ValueError:
                        # This is still a valid historical map, not a current pose.
                        # Keep its capture stamp; never substitute publish/arrival time.
                        result['state'] = 'historical_snapshot'
                        result.pop('map_pose', None)
                        result.pop('map_from_odom', None)
                        if packet is not None:
                            packet = encode_preview(points, poses[-1],
                                max_points=self._preview['max_points'], metadata=result)
            if staged_history is not None:
                self._history_snapshots[segment.segment_id] = staged_history
            if staged_links is not None:
                self._segment_links = staged_links
            if staged_loops is not None:
                self._segment_loops = staged_loops
            if packet is not None:
                msg = UInt8MultiArray()
                msg.data = packet
                self._preview_pub.publish(msg)
            message = String()
            message.data = json.dumps(result, allow_nan=False)
            self._preview_status_pub.publish(message)
            if geometry:
                common = {'state': result['state'], 'stamp_ns': result['stamp_ns'],
                          'input_validation': result['input_validation'],
                          'segment_id': result['segment_id'], 'map_frame': result['frame_id'],
                          'odom_frame': result.get('odom_frame'), 'accepted_loops': 0,
                          'candidate_count': result.get('candidate_count', 0)}
                message = String()
                message.data = json.dumps({**common, 'kind': 'unvalidated_geometry_pose',
                    'child_frame': segment.frames[-1].pose.child_frame_id if result['segment_id'] else None,
                    'pose': result.get('map_pose')}, allow_nan=False)
                self._map_pose_pub.publish(message)
                message = String()
                message.data = json.dumps({**common, 'kind': 'unvalidated_map_to_odom',
                    'transform': result.get('map_from_odom')}, allow_nan=False)
                self._map_correction_pub.publish(message)

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
            return self.stream.status(now=self._receipt_clock(), measurement_now_ns=measurement_now)
