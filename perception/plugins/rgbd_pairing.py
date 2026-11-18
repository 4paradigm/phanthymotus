"""Bounded RGB-D association without inventing a common capture identity."""
from collections import deque
from dataclasses import dataclass
import uuid


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
        self.sequence = 0
        self.dropped = 0

    @property
    def pending_count(self):
        return sum(len(q) for q in self.queues.values())

    @property
    def pending_bytes(self):
        return sum(v[2] for q in self.queues.values() for v in q)

    def _drop(self, q):
        q.popleft()
        self.dropped += 1

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
        q = self.queues[kind]
        q.append((message, context, size, self.sequence))
        while len(q) > self.max_pending:
            self._drop(q)
        while self.pending_bytes > self.max_bytes:
            oldest = min((v for v in self.queues.values() if v), key=lambda v: v[0][3])
            self._drop(oldest)
        color, depth = self.queues['color'], self.queues['depth']
        if color and depth:
            other = depth if kind == 'color' else color
            method = 'fifo'
            if len(other) > 1:
                # One stream can accumulate extra frames while the other drops
                # samples. Prefer the closest queued partner without waiting.
                stamp = context.observation_stamp_ns
                nearest = min(range(len(other)),
                              key=lambda i: abs(other[i][1].observation_stamp_ns - stamp))
                for _ in range(nearest):
                    self._drop(other)
                method = 'nearest_buffered'
            cm, a, _, _ = color.popleft()
            dm, b, _, _ = depth.popleft()
            return RgbdPair(cm, dm, a, b, uuid.uuid4().hex, method,
                            abs(a.observation_stamp_ns-b.observation_stamp_ns), a, b)
        return None
