"""Keep external fusion alive across changes to the visual tracking session."""
import hashlib
import json
import queue
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .external_odometry_fusion import (
    _HISTORY_NS as REPLAY_HISTORY_NS, _MAX_EVENTS as REPLAY_MAX_EVENTS,
    ExternalOdometry, body_pose_observation, decode_motion,
)
from .pointcloud_slam_transforms import rigid
from .rgbd_body_mount import _INTERPOLATION_NS as MOUNT_WINDOW_NS, CaptureMounts
from utils.ros_lifecycle import dispose_node


_FUTURE_TOLERANCE_NS = 50_000_000
_OUTPUT_TIMEOUT_S = 1.5
_OUTPUT_TIMEOUT_NS = int(_OUTPUT_TIMEOUT_S * 1_000_000_000)
_MAX_PENDING_EVENTS = 64


@dataclass
class _NativeVisual:
    segment: str
    pose: dict
    key: tuple
    projected_live: bool = False


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
        self.candidate_history = {}
        self.candidate_floors = {}
        self.latest_motion_stamp = 0
        self.stale_motion_stamp = None
        self.events = queue.Queue(maxsize=_MAX_PENDING_EVENTS)
        self.event_lock = threading.Lock()
        self.event_sequence = 0
        self.evicted_motion_boundaries = deque(maxlen=_MAX_PENDING_EVENTS)
        self.evicted_mounts = deque(maxlen=_MAX_PENDING_EVENTS)
        self.mount_eviction_fault = False
        self.motion_cutoffs = {}
        self.closed = False
        self.last_error = None
        self.dropped = 0
        self.pending_visual = OrderedDict()
        self.native_visual = OrderedDict()
        self.accepted_visual = set()
        self.learning_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rgbd-motion-learning')
        self.learning_work = None
        self.learning_pending_stamp = None
        self.learning_results = OrderedDict()
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
        with self.event_lock:
            if self.closed:
                return
            if self.events.full():
                try:
                    sequence, oldest = self.events.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.dropped += 1
                    if self._is_motion_boundary(oldest):
                        _, source, stamp, motion = oldest
                        # A dropped twist cannot stand in for the interval before it.
                        boundary = 'evicted_motion', source, stamp, replace(motion, reset=True)
                        self.evicted_motion_boundaries.append((sequence, boundary))
                    elif oldest[0] in ('mount', 'mount_fault'):
                        if len(self.evicted_mounts) == self.evicted_mounts.maxlen:
                            self.mount_eviction_fault = True
                        self.evicted_mounts.append((sequence, oldest))
            self.event_sequence += 1
            self.events.put_nowait((self.event_sequence, event))

    @staticmethod
    def _is_motion_boundary(event):
        return event[0] == 'motion' and (event[3].reset or event[3].twist is not None)

    def _take_ready(self):
        try:
            batch = [self.events.get(timeout=.05)]
        except queue.Empty:
            return []
        wall_now = time.time_ns()
        with self.event_lock:
            while not self.events.empty():
                batch.append(self.events.get_nowait())
            # The head was taken before acquiring this lock. A producer can
            # evict a later event in between; merge all retained ingress records
            # by their sequence instead of prepending that newer boundary.
            batch.extend(self.evicted_motion_boundaries)
            batch.extend(self.evicted_mounts)
            self.evicted_motion_boundaries.clear()
            self.evicted_mounts.clear()
            # A pose can still complete a retained interval after a worker wait.
            # Check motion eligibility when applying the batch's measured frontier.
            ready = [event for _, event in sorted(batch, key=lambda item: item[0])
                     if event[0] != 'visual' or self._visual_time_valid(event[1], wall_now)]
            self.dropped += len(batch) - len(ready)
            if self.mount_eviction_fault:
                # The lost packet may have invalidated any retained bracket.
                # Clear after this batch's numeric evidence, so its surviving
                # endpoints cannot silently bridge a missing boundary.
                ready.append(('mount_fault', 'body mount event history overflow'))
                self.mount_eviction_fault = False
        return ready

    def _receive(self, source, msg):
        now = time.time_ns()
        try:
            packet = json.loads(msg.data)
            if isinstance(packet, dict) and packet.get('schema') == 'motus.odom/1':
                stamp, motion = decode_motion(packet, now)
                if self._motion_time_valid(source, stamp, now, at_receipt=True):
                    self._offer(('motion', source, stamp, motion))
                else:
                    self._record_time_rejection(source, stamp, now)
        except (ValueError, TypeError, AttributeError, ArithmeticError, np.linalg.LinAlgError) as error:
            self.last_error = str(error)

    def _motion_time_valid(self, source, stamp, now, *, at_receipt=False):
        age = now - stamp
        if -_FUTURE_TOLERANCE_NS <= age <= _OUTPUT_TIMEOUT_NS:
            return True
        if age < -_FUTURE_TOLERANCE_NS or source != self.selected:
            return False
        # Old measurements can amend a selected timeline inside its replay
        # window. A fresh leading edge may still be ahead of us in the queue.
        if at_receipt:
            latest = max(self.latest_motion_stamp, self.fuser.current.stamp)
        else:
            # Only a surviving projection that was accepted live can lead
            # motion. Frozen owners are already behind the strict base cutoff.
            visual = max((t for t, owner in self.native_visual.items()
                          if owner.projected_live and t in self.fuser.raw_vision), default=0)
            latest = max(self.latest_motion_stamp, visual)
        oldest = max(latest, now-_OUTPUT_TIMEOUT_NS) - REPLAY_HISTORY_NS
        return (stamp >= oldest and stamp > self.fuser.base.stamp and
                (at_receipt or stamp <= latest))

    def _record_time_rejection(self, source, stamp, now):
        with self.event_lock:
            self.dropped += 1
            if source == self.selected and now-stamp > _OUTPUT_TIMEOUT_NS:
                self.stale_motion_stamp = max(self.stale_motion_stamp or 0, stamp)

    def _visual_time_valid(self, stamp, now):
        if type(stamp) is not int or stamp <= self.fuser.base.stamp:
            return False
        latest = max(self.latest_motion_stamp, self.fuser.current.stamp,
                     max(self.fuser.raw_vision, default=0))
        oldest = max(latest, now-_OUTPUT_TIMEOUT_NS) - REPLAY_HISTORY_NS
        return oldest <= stamp <= now+_FUTURE_TOLERANCE_NS

    def _select_candidate(self, source, stamp, motion, now):
        history = self.candidate_history.setdefault(source, {})
        prior = history.get(stamp)
        if prior is not None and prior.order_key() >= motion.order_key():
            return False
        history[stamp] = motion
        stamps = sorted(history)
        retained = [t for t in stamps[-_MAX_PENDING_EVENTS:] if now-t <= _OUTPUT_TIMEOUT_NS]
        if len(retained) < len(stamps):
            # The discarded prefix may contain a discontinuity. Reject later
            # amendments to that prefix even after this candidate is selected.
            self.candidate_floors[source] = retained[0]
            for t in stamps[:len(stamps)-len(retained)]:
                del history[t]
        if len(retained) < 2 or stamp not in history:
            return False
        begin, end = retained[-2:]
        left, right = history[begin], history[end]
        observed = left.mask & right.mask
        if (now-begin > _OUTPUT_TIMEOUT_NS or end-now > _FUTURE_TOLERANCE_NS or
                end-begin > _OUTPUT_TIMEOUT_NS or begin <= self.fuser.base.stamp or
                left.model is None or right.model != left.model or right.reset or
                left.origin != right.origin or not observed.any()):
            return False
        last_visual = max(self.fuser.raw_vision, default=None)
        visual_fresh = last_visual is not None and now-last_visual <= _OUTPUT_TIMEOUT_NS
        if not visual_fresh and not observed[[0, 1, 5]].all():
            return False  # A partial source alone cannot establish the initial body track.
        self.selected = source
        self.candidate_history.clear()
        # Preserve witnessed boundaries for delayed amendments to this wire.
        for t in retained:
            self.fuser.motion(t, history[t])
        self.motion_received = True
        self.latest_motion_stamp = end
        return True

    def _receive_mount(self, msg):
        try:
            self._offer(('mount', json.loads(msg.data), time.time_ns()))
        except (ValueError, TypeError, AttributeError) as error:
            self._offer(('mount_fault', 'invalid body mount JSON: '+str(error)))

    def visual(self, stamp, segment, pose):
        self._offer(('visual', stamp, segment, pose))

    def _camera_mount(self, stamp):
        if self.mounts is not None:
            return self.mounts.matrix_at(stamp)
        if self.fixed_mount is not None:
            return self.fixed_mount
        return np.eye(4) if self.camera == self.body else None

    def _apply(self, event, *, publish=True):
        kind, *values = event
        if kind in ('mount', 'mount_fault'):
            changed = None
            with self.fuser.batch():
                try:
                    if kind == 'mount_fault':
                        self.last_error = values[0]
                        self.mounts.fault(values[0])
                    else:
                        changed = {self.mounts.observe(*values)}
                finally:
                    self._replay_visuals(changed)
        elif kind in ('motion', 'evicted_motion'):
            source, stamp, motion = values
            if self.selected is not None and self.selected != source:
                return
            now = time.time_ns()
            if not self._motion_time_valid(source, stamp, now):
                self._record_time_rejection(source, stamp, now)
                return
            if kind == 'evicted_motion':
                # A current observation may have made this queued cut replayable.
                self.motion_cutoffs[source] = max(self.motion_cutoffs.get(source, 0), stamp)
            if (stamp < self.candidate_floors.get(source, 0) or
                    (not motion.reset and stamp < self.motion_cutoffs.get(source, 0))):
                # A late sample must not bridge an interval cut by an eviction.
                with self.event_lock:
                    self.dropped += 1
                return
            if self.selected is None:
                if not self._select_candidate(source, stamp, motion, now):
                    return
            elif self.fuser.motion(stamp, motion):
                self.motion_received = self.motion_received or motion.model is not None
                self.latest_motion_stamp = max(self.latest_motion_stamp, stamp)
        else:
            stamp, segment, pose = values
            if not self._apply_visual(stamp, segment, pose):
                if publish:
                    self._prune_visuals()
                return
        if publish:
            self._prune_visuals()
            self._publish()

    def _publishable_stamp(self, stamp):
        age = time.time_ns() - stamp
        if age > _OUTPUT_TIMEOUT_NS and self.active and 'stamp_ns' not in self.status:
            self.status = dict(self.status, stamp_ns=stamp)
        return -_FUTURE_TOLERANCE_NS <= age <= _OUTPUT_TIMEOUT_NS

    def _publish(self, state=None):
        if not self.active and not self.motion_received:
            return
        if state is None:
            state = self.fuser.current
        if (not self.active and state.visual is None and
                (state.mask is None or not state.mask[[0, 1, 5]].all())):
            # Preserve camera tracking while a partial input cannot establish a body track.
            return
        if state.last_update == 'external' and not state.mask[[0, 1, 5]].all():
            return
        if state.last_update is None:
            return
        stamp = state.stamp
        if not self._publishable_stamp(stamp):
            return
        sample = self.fuser.output(state)
        if sample is None:
            return
        quality = sample['vendor']['motus_slam']
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
        message = self.message_type(data=json.dumps(sample, allow_nan=False))
        if not self._publishable_stamp(stamp):
            return
        self.publisher.publish(message)
        self.active = True
        self.status = dict(active=True, topic=self.selected, stamp_ns=stamp, published_at=time.monotonic(),
                           **sample['vendor']['external_fusion'])
        self.reported_stale = False
        self.status_publisher.publish(self.message_type(data=json.dumps(dict(
            status='tracking', pose_reference='body', position_m=sample['pose']['position'],
            visual_status=self.visual_state, external_fusion=self.info()), allow_nan=False)))

    def _apply_visual(self, stamp, segment, pose):
        if not self._visual_time_valid(stamp, time.time_ns()):
            # Pending captures may have expired while waiting for their mount.
            # Consume the rejection instead of putting it back in that buffer.
            with self.event_lock:
                self.dropped += 1
            return True
        camera, _ = body_pose_observation(pose, np.eye(4))
        key = (segment, camera.tobytes())
        prior = self.native_visual.get(stamp)
        if prior is not None and prior.key >= key:
            return True
        self.native_visual[stamp] = _NativeVisual(segment, deepcopy(pose), key)
        if self.mounts is not None:
            self.mounts.retain((stamp,), replace=False)
        return self._project_visual(stamp)

    def _project_visual(self, stamp):
        owner = self.native_visual[stamp]
        mount = self._camera_mount(stamp)
        if mount is None:
            self.fuser.revise_vision(stamp, owner.segment, None)
            self.pending_visual[stamp] = (owner.segment, owner.pose)
            return False
        body, variance = body_pose_observation(owner.pose, rigid(mount))
        self.pending_visual.pop(stamp, None)
        age = time.time_ns() - stamp
        changed = self.fuser.revise_vision(stamp, owner.segment, body, variance)
        if changed and -_FUTURE_TOLERANCE_NS <= age <= _OUTPUT_TIMEOUT_NS:
            owner.projected_live = True
        self.accepted_visual.add(stamp)
        return True

    def _replay_visuals(self, changed=None):
        now = time.time_ns()
        for stamp in sorted(self.native_visual):
            if stamp <= self.fuser.base.stamp or (stamp not in self.accepted_visual and
                                                not self._visual_time_valid(stamp, now)):
                del self.native_visual[stamp]
                self.pending_visual.pop(stamp, None)
                self.accepted_visual.discard(stamp)
            elif changed is None or any(abs(stamp-value) <= MOUNT_WINDOW_NS for value in changed):
                self._project_visual(stamp)

    def _prune_visuals(self):
        now = time.time_ns()
        expired = {stamp for stamp in self.native_visual if stamp <= self.fuser.base.stamp or
                   (stamp not in self.accepted_visual and not self._visual_time_valid(stamp, now))}
        pending = sorted(stamp for stamp in self.native_visual
                         if stamp not in expired and stamp not in self.fuser.raw_vision)
        # Accepted projections retain their native owner throughout fuser replay.
        # Retraction does not turn an accepted owner into a wall-clock-expiring
        # new pending capture. Capacity can still evict unprojected captures,
        # oldest first, without forgetting any currently accepted projection.
        excess = max(0, len(self.native_visual)-len(expired)-REPLAY_MAX_EVENTS)
        for stamp in expired | set(pending[:excess]):
            del self.native_visual[stamp]
            self.pending_visual.pop(stamp, None)
            self.accepted_visual.discard(stamp)
        if self.mounts is not None:
            self.mounts.retain(self.native_visual)

    def _solve_learning(self, learner):
        # A result can be reused only for identical raw intervals and prior.
        # Late observations can invalidate both while a solve is still running.
        signature = hashlib.blake2b(digest_size=16)
        for value in (learner.parameters, learner.covariance, learner.observable_directions):
            signature.update(value.tobytes())
        for start, end, pieces, axes, target, noise in learner.window:
            signature.update(f'{start}:{end}:{len(pieces)};'.encode())
            for velocity, observed, duration, fixed_attitude in pieces:
                signature.update(velocity.tobytes() + observed.tobytes() + np.float64(duration).tobytes()
                                 + bytes((fixed_attitude,)))
            signature.update(axes.tobytes() + target.tobytes() + noise.tobytes())
        key = (learner.last_solve_ns, learner.effective_after_ns, signature.digest())
        if key in self.learning_results:
            return self.learning_results[key]
        if self.learning_work is None:
            snapshot = learner.checkpoint()
            self.learning_work = (key, snapshot, self.learning_pool.submit(snapshot.fit))
        else:
            # Revisions can introduce an earlier eligible solve or move it past
            # a withdrawn endpoint while the single worker is occupied.
            pending = self.learning_pending_stamp
            self.learning_pending_stamp = (learner.last_solve_ns if pending is None else
                                           min(pending, learner.last_solve_ns))
        return None

    def _complete_learning(self):
        if self.learning_work is None or not self.learning_work[2].done():
            return
        key, result, future = self.learning_work
        self.learning_work = None
        pending = self.learning_pending_stamp
        self.learning_pending_stamp = None
        future.result()
        # The worker's elapsed solve time is diagnostic. The replay lookup
        # checks the complete window signature before applying this snapshot.
        self.learning_results[key] = result
        if len(self.learning_results) > 64:
            self.learning_results.popitem(last=False)
        self.fuser.replay_learning(key[0] if pending is None else min(key[0], pending))

    def _apply_ready(self, events):
        if not events:
            return
        if self.mounts is not None:
            # First-seen captures also need the live evidence from before this
            # batch. Keep these temporary pins through every observation; the
            # normal post-batch pruning releases invalid/unselected captures.
            now = time.time_ns()
            captures = {event[1] for event in events if event[0] == 'visual' and
                        self._visual_time_valid(event[1], now)}
            self.mounts.retain(self.native_visual.keys() | captures)
        observations = []
        mounts_received = False
        changed_mounts = set()
        mount_fault = False
        for event in events:
            if event[0] not in ('mount', 'mount_fault'):
                observations.append(event)
                continue
            mounts_received = True
            try:
                if event[0] == 'mount_fault':
                    self.last_error = event[1]
                    self.mounts.fault(event[1])
                    mount_fault = True
                else:
                    changed_mounts.add(self.mounts.observe(*event[1:]))
            except (ValueError, TypeError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
                self.last_error = str(error)
                mount_fault = True
        current, delayed = [], []
        now = time.time_ns()
        for event in observations:
            stamp = event[1] if event[0] == 'visual' else event[2]
            if type(stamp) is int and now-stamp > _OUTPUT_TIMEOUT_NS:
                delayed.append((stamp, event))
            else:
                current.append(event)
        with self.fuser.batch():
            if mounts_received:
                # Revise accepted projections as well as captures still waiting
                # for evidence, against the complete mount batch.
                self._replay_visuals(None if mount_fault else changed_mounts)
            # An evicted live cut also applies to earlier ingress records.
            # Historical cuts are checked in capture order after the live batch.
            for event in current:
                if event[0] == 'evicted_motion':
                    _, source, stamp, _ = event
                    if self._motion_time_valid(source, stamp, time.time_ns()):
                        self.motion_cutoffs[source] = max(self.motion_cutoffs.get(source, 0), stamp)
            # Stage live observations first, then old corrections newest first.
            # Only accepted measurements may advance the replay floor while the
            # batch defers rebuilding fuser.current until all inputs are staged.
            delayed.sort(key=lambda item: item[0], reverse=True)
            current.extend(event for _, event in delayed)
            for event in current:
                try:
                    self._apply(event, publish=False)
                except (ValueError, TypeError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
                    self.last_error = str(error)
        self._prune_visuals()
        # Keep the measured samples needed to bracket cloud capture timestamps.
        for state in self.fuser.pending_output_states(time.time_ns()-_OUTPUT_TIMEOUT_NS):
            self._publish(state)

    def _work(self):
        while not self.closed:
            try:
                self._complete_learning()
                self._apply_ready(self._take_ready())
                if not self.reported_stale:
                    info = self.info()
                    if info['status'] == 'lost':
                        self.status_publisher.publish(self.message_type(data=json.dumps(dict(
                            status='lost', reason=info['reason'], pose_reference='body',
                            visual_status=self.visual_state, external_fusion=info), allow_nan=False)))
                        self.reported_stale = True
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
                      learning_budget_skips=0, segment_id=self.fuser.segment_id(self.fuser.current))
        published = result.pop('published_at', None)
        result.update(status='initializing', reason=None)
        if self.active and 'stamp_ns' not in result and self.stale_motion_stamp is not None:
            result['stamp_ns'] = self.stale_motion_stamp
        stamp = result.get('stamp_ns')
        if stamp is not None and time.time_ns() - stamp > _OUTPUT_TIMEOUT_NS:
            result.update(status='lost', reason='external fusion measurement is stale')
        elif published is not None:
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
