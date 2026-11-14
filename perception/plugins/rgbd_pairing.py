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
        self.max_pending, self.max_skew_ns = max_pending, max_skew_ns
        self.max_bytes, self.max_wait_ns = max_bytes, max_wait_ns
        self.previous, self.identities = {}, {}
        self.signatures = {}
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
        signature = (context.time_basis, context.normalized_clock_id,
                     context.source_epoch, context.geometry_version, context.mapping_version, context.ingress_clock_id)
        previous=self.signatures.get(kind)
        other=self.signatures.get('depth' if kind=='color' else 'color')
        conflict=previous is not None and previous!=signature
        if other is not None:
            arrival = 'receipt_estimated' in (signature[0], other[0])
            clocks_conflict = signature[-1]!=other[-1] if arrival else signature[1]!=other[1]
            conflict=conflict or clocks_conflict or any(
                a is not None and b is not None and a!=b for a,b in zip(signature[2:5],other[2:5]))
        if conflict:
            for q in self.queues.values():q.clear()
            raise ValueError('RGB-D time/source/geometry context changed; new segment required')
        self.signatures[kind]=signature
        stamp = context.observation_stamp_ns
        if stamp <= self.previous.get(kind, 0):
            raise ValueError('repeated or backwards RGB-D observation')
        identity = context.source_capture_id
        if identity is not None and identity == self.identities.get(kind):
            raise ValueError('repeated RGB-D capture identity')
        self.previous[kind], self.identities[kind] = stamp, identity
        if not hasattr(message, 'data'):
            raise ValueError('RGB-D message payload missing')
        size = len(message.data)
        if size > self.max_bytes:
            raise ValueError('RGB-D message exceeds byte budget')
        for q in self.queues.values():
            while q and context.ingress_monotonic_ns-q[0][1].ingress_monotonic_ns > self.max_wait_ns:
                self._drop(q)
        q = self.queues[kind]
        q.append((message, context, size))
        while len(q) > self.max_pending:self._drop(q)
        while self.pending_bytes > self.max_bytes:
            oldest = min((v for v in self.queues.values() if v), key=lambda v:v[0][1].ingress_monotonic_ns)
            self._drop(oldest)
        color, depth = self.queues['color'], self.queues['depth']
        while color and depth:
            a, b = color[0][1], depth[0][1]
            arrival = 'receipt_estimated' in (a.time_basis, b.time_basis)
            at, bt = (receipt_association(a), receipt_association(b)) if arrival else (a, b)
            skew = at.observation_stamp_ns-bt.observation_stamp_ns
            ids = a.source_capture_id, b.source_capture_id
            if abs(skew) <= self.max_skew_ns and (None in ids or ids[0] == ids[1]):
                method = ('arrival_time' if arrival else
                          'source_identity' if None not in ids else 'source_time')
                cm, _, _ = color.popleft();dm, _, _ = depth.popleft()
                return RgbdPair(cm, dm, a, b, uuid.uuid4().hex, method, abs(skew), at, bt)
            self._drop(color if skew <= 0 else depth)
        return None
