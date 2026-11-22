"""Rate-limited latest-frame input and bounded, ordered publication."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace

from utils.latest_frame import LatestFrame

from .runtime import Analysis, Mask, _clip

MAX_BUFFER_BYTES = 64 * 1024 * 1024
MAX_INPUT_AGE_S = .5
MAX_PUBLICATION_SLACK_S = .5


@dataclass(frozen=True)
class InputFrame:
    jpeg: bytes
    stamp: int
    captured_at: float  # source stamp translated to monotonic time on receipt
    received_at: float
    header: object = None


@dataclass
class Pending:
    frame: InputFrame
    analysis: Analysis
    due: float
    compute_s: float
    corrected: bool = False

    @property
    def size(self):
        return self.analysis.image.nbytes


def _recent_fps(times, now):
    recent = [stamp for stamp in times if now - stamp <= 2.0]
    if len(recent) < 2 or now - recent[-1] > 1.0 or recent[0] == recent[-1]:
        return 0.
    return round((len(recent) - 1) / (recent[-1] - recent[0]), 2)


class MosaicStream:
    """One worker owns analysis and rendering. ROS callbacks only submit input."""

    def __init__(self, engine, config, publish, on_error):
        self.engine = engine
        self.cfg = config
        self.publish = publish
        self.on_error = on_error
        self.inbox = LatestFrame()
        self.stopping = threading.Event()
        self.pending = deque()
        self.pending_bytes = 0
        self.delay = config.publish_delay_ms / 1000
        self.last_stamp = 0
        self._gate = threading.Lock()
        self._processed_times = deque(maxlen=90)
        self._published_times = deque(maxlen=90)
        self._latencies = deque(maxlen=90)
        self._stats = dict(
            processed=0, published=0, face_frames=0, no_face_frames=0,
            expired_input=0, expired_output=0, dropped_capacity=0, processing_errors=0,
            corrected_frames=0, published_corrected_frames=0, discarded_on_stop=0,
            buffer_frames=0, buffer_bytes=0, peak_buffer_frames=0, peak_buffer_bytes=0,
        )

    def _count(self, name, amount=1):
        with self._gate:
            self._stats[name] += amount

    def snapshot(self):
        now = time.monotonic()
        with self._gate:
            latencies = sorted(self._latencies)
            return {
                **self._stats,
                "target_fps": self.cfg.target_fps,
                "actual_processing_fps": _recent_fps(self._processed_times, now),
                "output_fps": _recent_fps(self._published_times, now),
                "source_to_publish_p50_ms": self._percentile(latencies, .50),
                "source_to_publish_p95_ms": self._percentile(latencies, .95),
                "source_latency_samples": len(latencies),
                "input_overwritten": self.inbox.dropped,
                "detection_calls": self.engine.detection_calls,
                "lk_calls": self.engine.lk_calls,
                "unreliable_track_updates": self.engine.bad_tracks,
                "over_capacity_detections": self.engine.over_capacity,
                "backend": getattr(self.engine.detector, "backend", "test"),
            }

    @staticmethod
    def _percentile(sorted_values, fraction):
        if not sorted_values:
            return None
        return round(sorted_values[round((len(sorted_values) - 1) * fraction)] * 1000, 2)

    def stop(self):
        self.stopping.set()
        self.inbox.close()

    def _record_buffer(self):
        with self._gate:
            self._stats["buffer_frames"] = len(self.pending)
            self._stats["buffer_bytes"] = self.pending_bytes
            self._stats["peak_buffer_frames"] = max(self._stats["peak_buffer_frames"],
                                                   len(self.pending))
            self._stats["peak_buffer_bytes"] = max(self._stats["peak_buffer_bytes"],
                                                  self.pending_bytes)

    def _pop(self):
        item = self.pending.popleft()
        self.pending_bytes -= item.size
        return item

    def _expired(self, frame, now):
        return now - frame.captured_at > self.delay + MAX_PUBLICATION_SLACK_S

    def _apply_corrections(self, corrections, ready_at):
        if not corrections or not self.pending:
            return
        for item in self.pending:
            if item.due <= ready_at or self._expired(item.frame, ready_at):
                continue
            stamp = item.frame.stamp
            height, width = item.analysis.image.shape[:2]
            for correction in corrections:
                if not correction.anchor_stamp < stamp < correction.end_stamp:
                    continue
                mask = item.analysis.masks.get(correction.track_id)
                if mask is None or not mask.correctable:
                    continue
                fraction = ((stamp - correction.anchor_stamp) /
                            (correction.end_stamp - correction.anchor_stamp))
                box = _clip(tuple(
                    value + fraction * shift for value, shift in zip(mask.box, correction.delta)
                ), width, height)
                if box is None or (box == mask.box and not mask.expanded):
                    continue
                item.analysis.masks[correction.track_id] = Mask(box)
                if not item.corrected:
                    item.corrected = True
                    self._count("corrected_frames")

    def _emit(self, item):
        if self.stopping.is_set():
            return
        frame = item.frame
        if frame.stamp <= self.last_stamp or self._expired(frame, time.monotonic()):
            self._count("expired_output")
            return
        started = time.perf_counter()
        encoded, rectangles = self.engine.render(item.analysis)
        if self.engine.timing is not None:
            rendered_s = time.perf_counter() - started
            self.engine.timing("render", rendered_s)
            self.engine.timing("process", item.compute_s + rendered_s)
        now = time.monotonic()
        if self._expired(frame, now):
            self._count("expired_output")
            return
        if self.stopping.is_set() or self.publish(frame, encoded, rectangles) is False:
            return
        self.last_stamp = frame.stamp
        with self._gate:
            self._stats["published"] += 1
            self._stats["face_frames" if rectangles else "no_face_frames"] += 1
            self._stats["published_corrected_frames"] += int(item.corrected)
            self._published_times.append(now)
            self._latencies.append(max(0., now - frame.captured_at))

    def _report_error(self, exc, frame):
        self.engine.reset()
        self._count("processing_errors")
        self.on_error(exc, frame)

    def _flush(self):
        while self.pending and not self.stopping.is_set():
            now = time.monotonic()
            item = self.pending[0]
            if self._expired(item.frame, now):
                self._pop()
                self._count("expired_output")
                self._record_buffer()
            elif item.due <= now:
                self._pop()
                self._record_buffer()
                try:
                    self._emit(item)
                except Exception as exc:
                    self._report_error(exc, item.frame)
            else:
                break

    def _accept(self, frame):
        if time.monotonic() - frame.captured_at > MAX_INPUT_AGE_S:
            self._count("expired_input")
            return
        started = time.perf_counter()
        analysis = self.engine.analyze(frame.jpeg, frame.stamp, frame.captured_at)
        compute_s = time.perf_counter() - started
        if self.engine.timing is not None:
            self.engine.timing("analyze", compute_s)
        now = time.monotonic()
        with self._gate:
            self._stats["processed"] += 1
            self._processed_times.append(now)
        if self._expired(frame, now):
            self._count("expired_output")
            return
        if self.stopping.is_set():
            return
        # Buffered frames retain only the normalized image and the original header.
        item = Pending(replace(frame, jpeg=b""), analysis, frame.received_at + self.delay, compute_s)
        if not self.delay:
            self._emit(item)
            return
        self._apply_corrections(analysis.corrections, now)
        analysis.corrections.clear()
        self._flush()
        if (len(self.pending) >= self.cfg.max_buffer_frames
                or self.pending_bytes + item.size > MAX_BUFFER_BYTES):
            # Keep the oldest scheduled publication. Replacing it before its
            # deadline at every input would starve an undersized buffer.
            self._count("dropped_capacity")
            return
        self.pending.append(item)
        self.pending_bytes += item.size
        self._record_buffer()

    def run(self):
        interval = 1 / self.cfg.target_fps
        next_slot = time.monotonic()
        try:
            while not self.stopping.is_set():
                self._flush()
                now = time.monotonic()
                wait = max(0., (self.pending[0].due - now) if self.pending else .25)
                if now < next_slot:
                    self.stopping.wait(max(0., min(wait, next_slot - now)))
                    continue
                frame = self.inbox.pop(wait)
                if frame is None or self.stopping.is_set():
                    continue
                next_slot = max(next_slot + interval, time.monotonic() + .75 * interval)
                try:
                    self._accept(frame)
                except Exception as exc:
                    self._report_error(exc, frame)
        finally:
            self.stop()
            self._count("discarded_on_stop", len(self.pending))
            self.pending.clear()
            self.pending_bytes = 0
            self._record_buffer()
