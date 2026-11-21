"""Keep external fusion alive across changes to the visual tracking session."""
import hashlib
import json
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .external_odometry_fusion import ExternalOdometry, body_pose_observation, decode_motion
from .pointcloud_slam_transforms import rigid
from .rgbd_body_mount import CaptureMounts
from utils.ros_lifecycle import dispose_node


_LEARNING_BUDGET_MS = 50.
_OUTPUT_TIMEOUT_S = 1.5


class ExternalOdometryInput:
    def __init__(self, topic, calibration, *, learning=True):
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from std_msgs.msg import String

        self.topic = topic
        mount = calibration.get('body_mount')
        self.camera = calibration['frame_id']
        self.body = (mount or {}).get('parent_frame', 'body')
        self.fixed_mount = calibration.get('body_from_camera')
        self.mounts = CaptureMounts(mount, self.camera, calibration.get('clock_id')) if mount else None
        self.fuser = ExternalOdometry(learning=learning, body_frame=self.body, solve_learning=self._solve_learning)
        self.active = calibration.get('external_topic') is not None and (
            self.fixed_mount is not None or self.camera == self.body)
        self.motion_received = False
        self.selected = calibration.get('external_topic')
        self.events = queue.Queue(maxsize=256)
        self.closed = False
        self.last_error = None
        self.dropped = 0
        self.pending_visual = OrderedDict()
        self.learning_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rgbd-motion-learning')
        self.learning_work = None
        self.learning_results = OrderedDict()
        self.learning_budget_skips = 0
        self.visual_state = None
        self.status = dict(active=self.active, topic=self.selected, update=None)
        self.reported_stale = False
        self.node = Node(self.fuser.segment + '_external')
        # The card executor can be occupied replacing the native visual worker.
        # Keep the independent motion and mount subscriptions running through it.
        self.executor = SingleThreadedExecutor()
        self.publisher = self.node.create_publisher(String, topic, qos_profile_sensor_data)
        status_topic = topic[:-11] if topic.endswith('/state/odom') else topic
        self.status_publisher = self.node.create_publisher(String, status_topic + '/status', qos_profile_sensor_data)
        self.message_type = String
        self.worker = self.spin_worker = None
        try:
            for source in calibration['external_candidates']:
                if source == topic:
                    raise ValueError('external odometry cannot be wired to its own output')
                self.node.create_subscription(String, source,
                    lambda message, source=source: self._receive(source, message), qos_profile_sensor_data)
            if self.mounts is not None:
                self.node.create_subscription(String, mount['topic'], self._receive_mount, qos_profile_sensor_data)
            self.executor.add_node(self.node)
            worker = threading.Thread(target=self._work, name=self.fuser.segment, daemon=True)
            worker.start()
            self.worker = worker
            spin = threading.Thread(target=self._spin, name=self.fuser.segment+'_ros', daemon=True)
            spin.start()
            self.spin_worker = spin
        except BaseException:
            self.close()
            raise

    def _offer(self, event):
        if not self.closed:
            try:
                self.events.put_nowait(event)
            except queue.Full:
                self.dropped += 1

    def _receive(self, source, msg):
        now = time.time_ns()
        try:
            packet = json.loads(msg.data)
            if isinstance(packet, dict) and packet.get('schema') == 'motus.odom/1':
                self._offer(('motion', source, *decode_motion(packet, now)))
        except (ValueError, TypeError, AttributeError, ArithmeticError, np.linalg.LinAlgError) as error:
            self.last_error = str(error)

    def _receive_mount(self, msg):
        try:
            self._offer(('mount', json.loads(msg.data), time.time_ns()))
        except (ValueError, TypeError, AttributeError):
            return

    def visual(self, stamp, segment, pose):
        self._offer(('visual', stamp, segment, pose))

    def _camera_mount(self, stamp):
        if self.mounts is not None:
            return self.mounts.matrix_at(stamp)
        if self.fixed_mount is not None:
            return self.fixed_mount
        return np.eye(4) if self.camera == self.body else None

    def _apply(self, event):
        kind, *values = event
        if kind == 'mount':
            self.mounts.observe(*values)
            ready = [stamp for stamp in self.pending_visual if self._camera_mount(stamp) is not None]
            for stamp in sorted(ready):
                segment, pose = self.pending_visual.pop(stamp)
                self._apply_visual(stamp, segment, pose)
            if not ready:
                return
        elif kind == 'motion':
            source, stamp, motion = values
            if self.selected is not None and self.selected != source:
                return
            self.selected = source
            self.motion_received = True
            self.fuser.motion(stamp, motion)
        else:
            stamp, segment, pose = values
            if not self._apply_visual(stamp, segment, pose):
                if self.mounts is not None:
                    self.pending_visual[stamp] = segment, pose
                    if len(self.pending_visual) > 24:
                        self.pending_visual.popitem(last=False)
                return
        if not self.active and not self.motion_received:
            return
        state = self.fuser.current
        if (not self.active and state.visual is None and
                (state.mask is None or not state.mask[[0, 1, 5]].all())):
            # Preserve camera tracking while a partial input cannot establish a body track.
            return
        if state.last_update == 'external' and not state.mask[[0, 1, 5]].all():
            return
        sample = self.fuser.output()
        if sample is None:
            return
        self.active = True
        quality = sample['vendor']['motus_slam']
        stamp = quality['stamp_ns']
        mount = self._camera_mount(stamp)
        if mount is not None:
            packet = dict(matrix=mount.tolist(), parent_frame=self.body, child_frame=self.camera, stamp_ns=stamp)
            sample['vendor']['camera_mount'] = dict(status='available', packet=packet)
            if self.mounts is None:
                quality['extrinsic'] = dict(packet, mode='static')
        elif self.mounts is not None:
            # Motion can arrive ahead of the camera's capture transform.
            # SLAM buffers this evidence at its own stamp while pose stays live.
            capture = self.mounts.packets.get(self.mounts.last_stamp)
            if capture is not None:
                packet = dict(parent_frame=self.body, child_frame=self.camera)
                packet.update(capture)
                status = 'available' if packet.get('matrix') is not None else 'unavailable'
                sample['vendor']['camera_mount'] = dict(status=status, packet=packet)
        self.publisher.publish(self.message_type(data=json.dumps(sample, allow_nan=False)))
        self.status = dict(active=True, topic=self.selected, stamp_ns=stamp, published_at=time.monotonic(),
                           **sample['vendor']['external_fusion'])
        self.reported_stale = False
        self.status_publisher.publish(self.message_type(data=json.dumps(dict(
            status='tracking', pose_reference='body', position_m=sample['pose']['position'],
            visual_status=self.visual_state, external_fusion=self.info()), allow_nan=False)))

    def _apply_visual(self, stamp, segment, pose):
        mount = self._camera_mount(stamp)
        if mount is None:
            return False
        body, variance = body_pose_observation(pose, rigid(mount))
        self.fuser.vision(stamp, segment, body, variance)
        return True

    def _solve_learning(self, learner):
        # A result can be reused only for identical raw intervals and prior.
        # Late observations can invalidate both while a solve is still running.
        signature = hashlib.blake2b(digest_size=16)
        for value in (learner.parameters, learner.covariance, learner.observable_directions):
            signature.update(value.tobytes())
        for start, end, pieces, axes, target, noise in learner.window:
            signature.update(f'{start}:{end}:{len(pieces)};'.encode())
            for velocity, observed, duration in pieces:
                signature.update(velocity.tobytes() + observed.tobytes() + np.float64(duration).tobytes())
            signature.update(axes.tobytes() + target.tobytes() + noise.tobytes())
        key = (learner.last_solve_ns, learner.effective_after_ns, signature.digest())
        if key in self.learning_results:
            return self.learning_results[key]
        if self.learning_work is None:
            snapshot = learner.checkpoint()
            self.learning_work = (key, snapshot, self.learning_pool.submit(snapshot.fit))
        return None

    def _complete_learning(self):
        if self.learning_work is None or not self.learning_work[2].done():
            return
        key, result, future = self.learning_work
        self.learning_work = None
        future.result()
        if result.last_solve_ms > _LEARNING_BUDGET_MS:
            self.learning_budget_skips += 1
            result = None
        self.learning_results[key] = result
        if len(self.learning_results) > 64:
            self.learning_results.popitem(last=False)
        if result is not None:
            self.fuser.replay_learning(key[0])

    def _work(self):
        while not self.closed:
            try:
                self._complete_learning()
                try:
                    event = self.events.get(timeout=.05)
                except queue.Empty:
                    if not self.reported_stale:
                        info = self.info()
                        if info['status'] == 'lost':
                            self.status_publisher.publish(self.message_type(data=json.dumps(dict(
                                status='lost', reason=info['reason'], pose_reference='body',
                                visual_status=self.visual_state, external_fusion=info), allow_nan=False)))
                            self.reported_stale = True
                    continue
                self._apply(event)
            except (ValueError, TypeError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
                self.last_error = str(error)

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as error:
            if not self.closed:
                self.last_error = str(error)

    def info(self):
        result = dict(self.status, active=self.active, dropped_events=self.dropped, last_error=self.last_error,
                      learning_budget_skips=self.learning_budget_skips, segment_id=self.fuser.segment)
        published = result.pop('published_at', None)
        result.update(status='initializing', reason=None)
        if published is not None:
            if time.monotonic()-published <= _OUTPUT_TIMEOUT_S:
                result['status'] = 'tracking'
            else:
                result.update(status='lost', reason='no fresh visual or external body motion')
        return result

    def close(self):
        self.closed = True
        self.executor.shutdown(timeout_sec=2)
        for worker in (self.spin_worker, self.worker):
            if worker is not None:
                worker.join(timeout=2)
                if worker.is_alive():
                    raise RuntimeError('external odometry worker did not stop')
        self.learning_pool.shutdown()
        dispose_node(self.executor, self.node)
