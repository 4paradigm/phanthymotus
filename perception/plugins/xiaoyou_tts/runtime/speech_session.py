"""Bit-exact PCM rendering and ordered speech delivery with one EOF per request."""
from dataclasses import dataclass, field
import queue
import json
import logging
import os
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
FRAME_BYTES = 3200
# Current official main transport: one initial500ms cushion, then real time.
PREBUF_FRAMES = 5
FRAME_INTERVAL_S = .1
EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'
log = logging.getLogger(__name__)


PREFIX_SAMPLES = 8000
NATIVE_PREFIX_SAMPLES = 12023
JOIN_ZERO_KEEP_SAMPLES = 2000


def safe_prefix(prefix, *, head=False, tail=False):
    """Return an already join-trimmed prefix, or None when proof needs the tail."""
    if head and tail:
        raise ValueError('One nonempty span cannot be on both sides of the join')
    if not isinstance(prefix, bytes) or len(prefix) != 2 * PREFIX_SAMPLES:
        return None
    nonzero = np.flatnonzero(np.frombuffer(prefix, dtype='<i2'))
    if tail and (not nonzero.size or nonzero[-1] < PREFIX_SAMPLES - JOIN_ZERO_KEEP_SAMPLES - 1):
        return None
    if head and not nonzero.size:
        return None
    removed = max(0, int(nonzero[0]) - JOIN_ZERO_KEEP_SAMPLES) if head else 0
    return prefix[2 * removed:]


def trim_join_zeros(pcm, *, head=False, tail=False):
    """Keep up to 125 ms of zero PCM on each side of a sentence join."""
    if head and tail:
        raise ValueError('One nonempty span cannot be on both sides of the join')
    if not (head or tail) or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype='<i2')
    nonzero = np.flatnonzero(samples)
    head_zeros = int(nonzero[0]) if nonzero.size else len(samples)
    tail_zeros = len(samples) - int(nonzero[-1]) - 1 if nonzero.size else len(samples)
    head_cut = max(0, head_zeros - JOIN_ZERO_KEEP_SAMPLES) if head else 0
    tail_cut = max(0, tail_zeros - JOIN_ZERO_KEEP_SAMPLES) if tail else 0
    return pcm[2 * head_cut:len(pcm) - 2 * tail_cut]


def iter_exact_pcm(wave, *, resample, pcm16, head=False, tail=False, cancel=None):
    """Emit a safe prefix, followed by the rest of the whole-wave resample."""
    def cancelled():
        return cancel is not None and cancel.is_set()

    if cancelled():
        return
    emitted = b''
    if len(wave) >= NATIVE_PREFIX_SAMPLES:
        prefix = resample(wave[:NATIVE_PREFIX_SAMPLES])[:PREFIX_SAMPLES]
        admitted = safe_prefix(pcm16(prefix), head=head, tail=tail)
        if admitted is not None:
            if cancelled():
                return
            emitted = admitted
            yield emitted
    if cancelled():
        return
    # Preserve the original FIR phase and EOF padding.
    whole = trim_join_zeros(pcm16(resample(wave)), head=head, tail=tail)
    if not whole.startswith(emitted):
        raise RuntimeError('Prefix differs from complete reference PCM')
    rest = whole[len(emitted):]
    if rest and not cancelled():
        yield rest


@dataclass
class Request:
    text: str
    action_id: str
    language: str = 'auto'
    trace_id: str = ''
    cancel: threading.Event = field(default_factory=threading.Event)
    delivery_lock: threading.RLock = field(default_factory=threading.RLock)
    eof_sent: bool = False
    completed: bool = False
    bytes_sent: int = 0
    timing: dict = field(default=None, repr=False)
    auto_fallback: str = field(default=None, repr=False)


class SpeechSession:
    """Serialize whole calls; interruption cannot publish PCM after its EOF.

    Backend generation may be in flight; cancellation stops delivery immediately.
    It cannot preempt an already executing neural kernel.
    """
    def __init__(self, backend, publish, complete, *, ready=None, pace=True, max_queue=16):
        self.backend, self.publish, self.complete = backend, publish, complete
        self.ready, self.pace = ready, bool(pace)
        self._queue = queue.Queue(maxsize=max_queue)
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._thread = None
        self._active = None
        self.last_error = None
        self._timing_enabled = os.environ.get('XIAOYOU01_TTS_TIMING', '1') == '1'
        self._timing_counter = 0
        # Only this ordered session owns spoken auto-language context.
        self._auto_language = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self):
        with self._state_lock:
            if self.running:
                return
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError('Previous V6 inference is still stopping')
            self._stop.clear()
            self._wakeup.clear()
            self._thread = threading.Thread(target=self._worker, name='xiaoyou-v6-pcm', daemon=True)
            self._thread.start()

    def enqueue(self, request):
        with self._state_lock:
            if not self.running:
                raise RuntimeError('V6 session is not running')
            if not request.text.strip():
                raise ValueError('Nonempty text is required')
            if self._timing_enabled:
                self._timing_counter += 1
                request.timing = dict(origin_perf=time.perf_counter(), request_index=self._timing_counter,
                                      queue_wait_seconds=None, ready_wait_seconds=None,
                                      first_backend_chunk_seconds=None, first_publish_enter_seconds=None,
                                      first_publish_return_seconds=None, eof_seconds=None)
            self._queue.put_nowait(request)
            self._wakeup.set()

    def _finish(self, request, status, error=None, *, eof=True):
        timing = request.timing
        with request.delivery_lock:
            if eof and not request.eof_sent:
                self.publish(EOF_MAGIC)
                request.eof_sent = True
                if timing is not None:
                    timing['eof_seconds'] = time.perf_counter() - timing['origin_perf']
            if request.completed:
                return
            request.completed = True
            result = {'bytes': request.bytes_sent, 'frames': (request.bytes_sent + FRAME_BYTES - 1) // FRAME_BYTES,
                      'trace_id': request.trace_id, 'eof_sent': request.eof_sent}
            if error:
                result['error'] = str(error)
            completion = time.perf_counter() - timing['origin_perf'] if timing is not None else None
            self.complete(request.action_id, status, result)
        if timing is not None:
            if timing['queue_wait_seconds'] is None:
                timing['queue_wait_seconds'] = completion
            record = {'event': 'xiaoyou_request_timing', 'schema': 'xiaoyou_session_timing_v1',
                      'action_id': request.action_id[:96] if isinstance(request.action_id, str) else '',
                      'trace_id': request.trace_id[:128] if isinstance(request.trace_id, str) else '',
                      'pace': self.pace, 'bytes_sent': request.bytes_sent, 'status': status,
                      'eof_sent': request.eof_sent, 'completion_seconds': completion,
                      **{key: value for key, value in timing.items() if key != 'origin_perf'}}
            log.info('%s', json.dumps(record, separators=(',', ':')))

    def interrupt(self):
        with self._state_lock:
            active = self._active
            if active is not None:
                with active.delivery_lock:
                    active.cancel.set()
                    self._finish(active, 'cancelled')
            cleared = 0
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                pending.cancel.set()
                self._finish(pending, 'cancelled', eof=False)
                self._queue.task_done()
                cleared += 1
            return cleared

    def stop(self, timeout=30):
        self._stop.set()
        self._wakeup.set()
        self.interrupt()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError('V6 generation is still active; resources were not freed')

    def _send(self, request, frame, origin, frame_index):
        if self.pace:
            request.cancel.wait(max(0, origin + frame_index * FRAME_INTERVAL_S - time.monotonic()))
        with request.delivery_lock:
            if request.cancel.is_set() or self._stop.is_set() or request.eof_sent:
                return False
            timing = request.timing
            first = timing is not None and timing['first_publish_enter_seconds'] is None
            if first:
                timing['first_publish_enter_seconds'] = time.perf_counter() - timing['origin_perf']
            self.publish(frame)
            if first:
                timing['first_publish_return_seconds'] = time.perf_counter() - timing['origin_perf']
            request.bytes_sent += len(frame)
            return True

    def _deliver(self, request):
        timing = request.timing
        wait_for_subscriber = self.ready
        pending = bytearray()
        prebuf = []
        origin = None
        frame_index = 0

        def emit(frame):
            nonlocal origin, frame_index
            if origin is None:
                origin = time.monotonic()
            if not self._send(request, frame, origin, frame_index):
                return False
            frame_index += 1
            return True

        def flush_prebuf():
            nonlocal origin
            if not prebuf:
                return True
            # Backdate only once: the initial frames are all due now. Later
            # frames keep the same100ms schedule, with no rebase or speedup.
            origin = time.monotonic() - max(0, len(prebuf) - 1) * FRAME_INTERVAL_S
            for frame in prebuf:
                if not emit(frame):
                    return False
            prebuf.clear()
            return True

        options = dict(language=request.language, speed=1.0, cancel=request.cancel)
        if request.auto_fallback is not None:
            options['auto_fallback'] = request.auto_fallback
        chunks = self.backend.iter_pcm(request.text, **options)
        for chunk in chunks:
            if request.cancel.is_set() or self._stop.is_set():
                break
            if not isinstance(chunk, bytes) or len(chunk) % 2:
                raise ValueError('V6 backend must yield aligned PCM16 bytes')
            if timing is not None and timing['first_backend_chunk_seconds'] is None and chunk:
                timing['first_backend_chunk_seconds'] = time.perf_counter() - timing['origin_perf']
            if chunk and wait_for_subscriber is not None:
                # DDS discovery can run while the backend synthesizes the first chunk.
                started = time.perf_counter() if timing is not None else None
                try:
                    wait_for_subscriber(request.cancel)
                finally:
                    if timing is not None:
                        timing['ready_wait_seconds'] = time.perf_counter() - started
                wait_for_subscriber = None
            pending.extend(chunk)
            while len(pending) >= FRAME_BYTES:
                frame = bytes(pending[:FRAME_BYTES])
                del pending[:FRAME_BYTES]
                if self.pace and origin is None:
                    prebuf.append(frame)
                    if len(prebuf) >= PREBUF_FRAMES and not flush_prebuf():
                        return
                elif not emit(frame):
                    return
        if not request.cancel.is_set() and not self._stop.is_set():
            if pending:
                tail = bytes(pending)
                if self.pace and origin is None:
                    # A short utterance's final partial joins its EOF flush.
                    # No fabricated/silent frame or extra playback wait.
                    prebuf.append(tail)
                elif not emit(tail):
                    return
            if prebuf and not flush_prebuf():
                return
        if not request.bytes_sent and not request.cancel.is_set() and not self._stop.is_set():
            raise RuntimeError('V6 generated no PCM')

    def _worker(self):
        while not self._stop.is_set():
            # Dequeue and publish active ownership atomically against interrupt.
            with self._state_lock:
                try:
                    request = self._queue.get_nowait()
                except queue.Empty:
                    request = None
                    # enqueue uses this same lock, so its wake cannot be erased.
                    self._wakeup.clear()
                if request is not None:
                    if self._stop.is_set():
                        request.cancel.set()
                    self._active = request
                    if request.timing is not None:
                        request.timing['queue_wait_seconds'] = time.perf_counter() - request.timing['origin_perf']
            if request is None:
                # stop may have set its wake before the empty-queue clear.
                if not self._stop.is_set():
                    self._wakeup.wait()
                continue
            try:
                next_language = self._auto_language
                if not request.cancel.is_set():
                    if request.language in ('zh', 'en'):
                        request.auto_fallback = None
                        next_language = request.language
                    else:
                        from ..frontend.numbers import auto_context_language
                        request.auto_fallback = self._auto_language
                        detected, natural = auto_context_language(request.text, self._auto_language or 'zh')
                        if natural:
                            next_language = detected
                    self._deliver(request)
                status = 'cancelled' if request.cancel.is_set() or self._stop.is_set() else 'completed'
                self._finish(request, status)
                if status == 'completed' and request.bytes_sent and not request.cancel.is_set() and not self._stop.is_set():
                    self._auto_language = next_language
            except Exception as error:
                self.last_error = str(error)
                self._finish(request, 'cancelled', error=error)
            finally:
                with self._state_lock:
                    self._active = None
                self._queue.task_done()
