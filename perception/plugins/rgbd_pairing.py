"""Bounded RGB-D association without inventing a common capture identity."""
from collections import deque
from dataclasses import dataclass
import uuid

from .spatial_time import receipt_association


@dataclass(frozen=True)
class RgbdPair:
    color: object
    depth: object
    color_context: object
    depth_context: object
    association_id: str
    method: str
    skew_ns: int

    color_time: object
    depth_time: object

    def __iter__(self):
        return iter(((self.color_time.observation_stamp_ns, self.color),
                     (self.depth_time.observation_stamp_ns, self.depth)))

    def __getitem__(self, index):
        return tuple(self)[index]


class RgbdPairer:
    def __init__(self, max_pending=5, max_skew_ns=20_000_000,
                 max_bytes=16*1024*1024, max_wait_ns=1_000_000_000):
        self.queues = {k: deque() for k in ('color', 'depth')}
        self.max_pending, self.max_bytes = max_pending, max_bytes
        self.max_skew_ns, self.max_wait_ns = max_skew_ns, max_wait_ns
        self.sequence = 0
        self.dropped = 0
        self.dropped_by_side = dict(color=0, depth=0)
        self.last_skew_ns = None

    @property
    def pending_count(self):
        return sum(len(q) for q in self.queues.values())

    @property
    def pending_bytes(self):
        return sum(v[2] for q in self.queues.values() for v in q)

    def _drop(self, kind, index=0):
        del self.queues[kind][index]
        self.dropped += 1
        self.dropped_by_side[kind] += 1

    def _candidate(self, first, second):
        if (first.time_basis == second.time_basis == 'source_capture' and
                first.normalized_clock_id and first.normalized_clock_id == second.normalized_clock_id):
            skew = abs(first.observation_stamp_ns - second.observation_stamp_ns)
            identity = (first.source_capture_id and (first.source_epoch or first.source_id) and
                        first.source_capture_id == second.source_capture_id and
                        first.source_epoch == second.source_epoch and first.source_id == second.source_id)
            priority, method = (0, 'capture_id') if identity else (1, 'nearest_capture')
        elif first.ingress_clock_id and first.ingress_clock_id == second.ingress_clock_id:
            if abs(first.ingress_monotonic_ns - second.ingress_monotonic_ns) > self.max_skew_ns:
                return None
            skew = abs(first.ingress_realtime_ns - second.ingress_realtime_ns)
            priority, method = 2, 'receipt_estimated'
        else:
            return None
        if skew <= self.max_skew_ns:
            return priority, skew, method

    def push(self, kind, message, context):
        if kind not in self.queues or type(getattr(context, 'observation_stamp_ns', None)) is not int:
            raise ValueError('RGB-D requires an image role and integer timestamp')
        if not hasattr(message, 'data'):
            raise ValueError('RGB-D message payload missing')
        try:
            size = memoryview(message.data).nbytes
        except TypeError:
            raise ValueError('RGB-D message payload must be a byte buffer') from None
        if size > self.max_bytes:
            raise ValueError('RGB-D message exceeds byte budget')
        self.sequence += 1
        for side, queue in self.queues.items():
            for index in reversed(range(len(queue))):
                pending = queue[index][1]
                if (pending.ingress_clock_id == context.ingress_clock_id and
                        context.ingress_monotonic_ns - pending.ingress_monotonic_ns > self.max_wait_ns):
                    self._drop(side, index)
        q = self.queues[kind]
        q.append((message, context, size, self.sequence))
        if len(q) > self.max_pending:
            self._drop(kind)
        while self.pending_bytes > self.max_bytes:
            oldest = min((side for side, queue in self.queues.items() if queue),
                         key=lambda side: self.queues[side][0][3])
            self._drop(oldest)
        if not q:
            return None
        other = 'depth' if kind == 'color' else 'color'
        choices = []
        for index, (_, pending, _, _) in enumerate(self.queues[other]):
            candidate = self._candidate(context, pending)
            if candidate is not None:
                choices.append((*candidate, index))
        if not choices:
            return None
        _, skew, method, index = min(choices)
        for _ in range(index):
            self._drop(other)
        # Older frames on either stream have been superseded by this pair.
        while len(q) > 1:
            self._drop(kind)
        cm, a, _, _ = self.queues['color'].popleft()
        dm, b, _, _ = self.queues['depth'].popleft()
        color_time, depth_time = a, b
        if method == 'receipt_estimated':
            color_time, depth_time = receipt_association(a), receipt_association(b)
        self.last_skew_ns = skew
        return RgbdPair(cm, dm, a, b, uuid.uuid4().hex, method, skew, color_time, depth_time)
