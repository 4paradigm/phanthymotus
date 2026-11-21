"""Bounded RGB-D association without inventing a common capture identity."""
from collections import deque
from dataclasses import dataclass, replace
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
        self.estimated_capture_clock_id = 'rgbd_capture_estimated/' + uuid.uuid4().hex
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
        if (first.source_id and first.source_id == second.source_id and first.source_epoch and
                second.source_epoch and first.source_epoch != second.source_epoch):
            return None
        skew = abs(first.observation_stamp_ns - second.observation_stamp_ns)
        if skew > self.max_skew_ns:
            return None
        receipt = tuple(ctx.original_stamp is None and ctx.time_basis == 'receipt_estimated'
                        for ctx in (first, second))
        if any(receipt):
            if all(receipt) and abs(first.ingress_monotonic_ns-second.ingress_monotonic_ns) > self.max_skew_ns:
                return None
            return skew, 'receipt_estimated'
        if not first.normalized_clock_id or first.normalized_clock_id != second.normalized_clock_id:
            return skew, 'nearest_capture_estimated'
        identity = (first.source_capture_id and (first.source_epoch or first.source_id) and
                    first.source_capture_id == second.source_capture_id and
                    first.source_epoch == second.source_epoch and first.source_id == second.source_id)
        return skew, 'capture_id' if identity else 'nearest_capture'

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
                if context.ingress_monotonic_ns - pending.ingress_monotonic_ns > self.max_wait_ns:
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
                skew, method = candidate
                choices.append((skew, index, method))
        if not choices:
            return None
        skew, index, method = min(choices)
        selected = {kind: q.pop(), other: self.queues[other][index]}
        del self.queues[other][index]
        for side, queue in self.queues.items():
            stamp = selected[side][1].observation_stamp_ns
            for index in reversed(range(len(queue))):
                if queue[index][1].observation_stamp_ns < stamp:
                    self._drop(side, index)
        cm, a, _, _ = selected['color']
        dm, b, _, _ = selected['depth']
        color_time, depth_time = a, b
        if method == 'receipt_estimated':
            color_time, depth_time = (
                replace(receipt_association(ctx), observation_stamp_ns=ctx.observation_stamp_ns)
                for ctx in (a, b)
            )
        elif method == 'nearest_capture_estimated':
            # Retain capture values for visual processing on an unverified timeline.
            color_time, depth_time = (
                replace(ctx, observation_id=uuid.uuid4().hex,
                        normalized_clock_id=self.estimated_capture_clock_id, time_basis='capture_estimated',
                        source_capture_id=None, mapping_source=None, mapping_version=None,
                        mapping_offset_ns=None, mapping_verified=False, uncertainty_ns=None,
                        parent_observation_ids=(ctx.observation_id,),
                        transformations=ctx.transformations+('numeric_capture_alignment',))
                for ctx in (a, b)
            )
        self.last_skew_ns = skew
        return RgbdPair(cm, dm, a, b, uuid.uuid4().hex, method, skew, color_time, depth_time)
