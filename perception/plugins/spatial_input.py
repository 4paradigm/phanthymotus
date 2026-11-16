"""One ingress per ROS stream, bounded independent delivery to each consumer."""
from collections import deque, OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import threading
import time
import uuid

from .spatial_time import TimePolicy, FrameContext


def _stamp(message):
    header = getattr(message, 'header', None)
    stamp = getattr(header, 'stamp', None)
    if stamp is None:return None
    if type(stamp.sec) is not int or type(stamp.nanosec) is not int:
        raise ValueError('invalid ROS timestamp')
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _payload(message):
    value = getattr(message, 'data', b'')
    return value.encode('utf8') if isinstance(value, str) else value


def _identity(topic, message):
    return (topic, getattr(getattr(message, 'header', None), 'frame_id', ''),
            _stamp(message), hashlib.sha256(_payload(message)).hexdigest())


def latency_summary(samples):
    values=sorted(samples)
    result={'count':len(values)}
    for name,percentile in (('p50_ms',.5),('p95_ms',.95),('p99_ms',.99)):
        result[name]=round(values[min(len(values)-1,int((len(values)-1)*percentile))]/1e6,3) if values else None
    return result


class SubscriptionLease:
    def __init__(self, registry, stream, consumer):
        self.registry, self.stream, self.consumer = registry, stream, consumer
        self._condition = threading.Condition()
        self._pending = deque()
        self._bytes = self._dropped = self._errors = 0
        self._closed = False
        self._last_error = None
        self._queue_times,self._callback_times=deque(maxlen=128),deque(maxlen=128)
        self._worker = threading.Thread(target=self._run, name='spatial-input-consumer', daemon=True)
        self._worker.start()

    def _offer(self, message, context):
        size = len(_payload(message))
        with self._condition:
            if self._closed:
                return
            if size > self.registry.max_bytes:
                self._dropped += 1
                return
            while self._pending and (len(self._pending) >= self.registry.max_pending or
                                     self._bytes + size > self.registry.max_bytes):
                self._bytes -= self._pending.popleft()[2]
                self._dropped += 1
            self._pending.append((message, context, size, time.monotonic_ns()))
            self._bytes += size
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending)
                if self._closed:
                    return
                message, context, size, offered = self._pending.popleft()
                self._bytes -= size
            began=time.monotonic_ns()
            with self._condition:self._queue_times.append(max(0,began-offered))
            try:
                # The canonical message is never lent to a mutating consumer.
                self.consumer(deepcopy(message), context)
            except Exception as error:
                with self._condition:
                    self._errors += 1
                    self._last_error = str(error)[:300]
            finally:
                with self._condition:self._callback_times.append(max(0,time.monotonic_ns()-began))

    def status(self):
        with self._condition:
            return dict(pending=len(self._pending), pending_bytes=self._bytes,
                        dropped=self._dropped, errors=self._errors, last_error=self._last_error,
                        timing=dict(queue_wait=latency_summary(self._queue_times),callback=latency_summary(self._callback_times)))

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._bytes = 0
            self._condition.notify_all()
        self.registry._release(self)
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=2)
            if self._worker.is_alive():
                raise RuntimeError('spatial consumer did not stop')


class InputAdapterRegistry:
    def __init__(self, executor=None, *, subscribe=None, max_pending=5,
                 max_bytes=16*1024*1024, max_age_ns=1_000_000_000):
        for value in (max_pending, max_bytes, max_age_ns):
            if type(value) is not int or value <= 0:
                raise ValueError('positive bounded input capacities required')
        self.executor, self._subscribe = executor, subscribe
        self.max_pending, self.max_bytes, self.max_age_ns = max_pending, max_bytes, max_age_ns
        self.clock_id = 'perception_ingress/' + uuid.uuid4().hex
        self.time_policy = TimePolicy(self.clock_id, max_age_ns=max_age_ns)
        self._lock = threading.RLock()
        self._streams, self._derived = {}, OrderedDict()
        self._node = None
        self._closed = False
        # A registry belongs to exactly one executor/ROS context, not all domains.
        self._domain = id(getattr(executor, 'context', executor))

    def _ros_subscribe(self, topic, message_type, callback):
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        if self._node is None:
            self._node = Node('spatial_input_' + uuid.uuid4().hex[:8],
                              context=getattr(self.executor, 'context', None))
            self.executor.add_node(self._node)
        subscription = self._node.create_subscription(message_type, topic, callback, qos_profile_sensor_data)
        return lambda: self._node.destroy_subscription(subscription)

    def acquire(self, topic, message_type, declaration, consumer):
        if not isinstance(topic, str) or not topic.strip() or not callable(consumer):
            raise ValueError('topic and consumer required')
        declaration = deepcopy(declaration or {})
        key = (self._domain, topic, message_type)
        with self._lock:
            if self._closed:
                raise RuntimeError('input registry is closed')
            stream = self._streams.get(key)
            if stream is not None:
                self._check_declarations(stream['declaration'], declaration)
            else:
                stream = dict(key=key, declaration=declaration, leases=set(), active=True,
                              received=0, rejected=0, last_error=None, close=None)
                self._streams[key] = stream
                try:
                    stream['close'] = (self._subscribe or self._ros_subscribe)(
                        topic, message_type, lambda m: self._receive(stream, m))
                except Exception:
                    del self._streams[key]
                    stream['active'] = False
                    raise
            lease = SubscriptionLease(self, stream, consumer)
            stream['leases'].add(lease)
            return lease

    @staticmethod
    def _check_declarations(first, second):
        return None  # Each consumer owns its numeric calibration.

    def _source(self, declaration, message):
        source = dict(declaration.get('vendor') or {})
        if declaration.get('schema') in ('motus.slam_input/1','motus.slam_input/2'):
            source.update(clock_id=declaration.get('clock_id'),timestamp_semantics='source_capture_mapped',
                          source_id=declaration.get('source_id'),source_epoch=declaration.get('source_epoch'))
        stamp = _stamp(message)
        if stamp is None and isinstance(getattr(message, 'data', None), str):
            sample = json.loads(message.data)
            quality = (sample.get('vendor') or {}).get('motus_slam') or {}
            if quality:
                source.update((k, v) for k, v in quality.items() if k in (
                    'clock_id', 'source_id', 'source_epoch'))
                stamp = quality.get('stamp_ns')
                source['timestamp_semantics'] = quality.get('stamp_source')
            if stamp is None and type(sample.get('stamp_ms')) is int:
                stamp=sample['stamp_ms']*1_000_000
        source['stamp_ns'] = stamp
        return source

    def _receive(self, stream, message):
        realtime, monotonic = time.time_ns(), time.monotonic_ns()
        with self._lock:
            if not stream['active'] or self._closed:
                return
            stream['received'] += 1
            try:
                if len(_payload(message)) > self.max_bytes:
                    raise ValueError('input payload exceeds byte budget')
                context = self._derived.get(_identity(stream['key'][1], message))
                if context is None and isinstance(getattr(message,'data',None),str):
                    sample=json.loads(message.data)
                    quality=(sample.get('vendor') or {}).get('motus_slam') or {}
                    if isinstance(quality.get('observation'),dict):
                        context=FrameContext.from_dict(quality['observation'])
                if context is None:
                    context = self.time_policy.select(self._source(stream['declaration'], message), realtime, monotonic)
            except (ValueError, TypeError, AttributeError, OverflowError) as error:
                stream['rejected'] += 1
                stream['last_error'] = str(error)[:300]
                return
            leases = tuple(stream['leases'])
        for lease in leases:
            lease._offer(message, context)

    def register_derived(self, topic, message, parent, stage):
        context = replace(parent, observation_id=uuid.uuid4().hex,
                          parent_observation_ids=(parent.observation_id,),
                          transformations=parent.transformations + (stage,))
        with self._lock:
            self._derived[_identity(topic, message)] = context
            while len(self._derived) > 256:
                self._derived.popitem(last=False)
        return context

    def _release(self, lease):
        with self._lock:
            stream = lease.stream
            stream['leases'].discard(lease)
            if not stream['leases'] and stream['active']:
                stream['active'] = False
                self._streams.pop(stream['key'], None)
                stream['close']()

    def status(self):
        with self._lock:
            return {str(k[1]): dict(received=s['received'], rejected=s['rejected'],
                    last_error=s['last_error'], consumers=[l.status() for l in s['leases']])
                    for k, s in self._streams.items()}

    def close(self):
        with self._lock:
            self._closed = True
            leases = [l for s in self._streams.values() for l in s['leases']]
        failures = []
        for lease in leases:
            try:
                lease.close()
            except RuntimeError as error:
                failures.append(str(error))
        if self._node is not None:
            from utils.ros_lifecycle import dispose_node
            dispose_node(self.executor, self._node, label='spatial-input')
            self._node = None
        if failures:
            raise RuntimeError('; '.join(failures))
