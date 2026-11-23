"""Xiaoyou adapter for the public tts tool and ordered ROS audio delivery."""
import array
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from .runtime.synthesis import acquire_gpu_backend, validate_options

log = logging.getLogger(__name__)
# Keep request timing under its existing logger for log consumers.
_session_log = logging.getLogger(__package__ + '.runtime.speech_session')
ENGINE = 'xiaoyou-zh-en'
MODEL_ID = 'xiaoyou01'
LANGUAGES = ('zh', 'en')
ACTIONS = ('start', 'stop', 'speak', 'info', 'config', 'interrupt', 'support_language')

# Keep only explicitly owned, stopped transports; no worker survives a stop.
MAX_IDLE_TRANSPORTS = 4

FRAME_BYTES = 3200
# Current official main transport: one initial500ms cushion, then real time.
PREBUF_FRAMES = 5
FRAME_INTERVAL_S = .1
EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'


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
            _session_log.info('%s', json.dumps(record, separators=(',', ':')))

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
                        from .frontend.numbers import auto_context_language
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


def _tts_output_topic(input_topic):
    return input_topic + '/tts' if input_topic else '/perception/tts'


def _node(input_topic, backend, suffix, cfg):
    # ROS is needed only for topic delivery, not native loading/raw inference.
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from audio_msgs.msg import AudioChunk
    from utils.acp import complete
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                     depth=200, durability=DurabilityPolicy.VOLATILE)

    class V6Node(Node):
        def __init__(self):
            super().__init__('xiaoyou_v6_' + suffix)
            self._input_topic = input_topic or ''
            self._output_topic = _tts_output_topic(self._input_topic)
            self._pub = self.create_publisher(AudioChunk, self._output_topic, qos)
            self._sub = self.create_subscription(String, self._input_topic, self._text_cb, qos) if self._input_topic else None
            self.session = SpeechSession(backend, self._publish, complete, ready=self._ready,
                                         pace=cfg.get('pace_audio', True))

        @property
        def state(self):
            return 'running' if self.session.running else 'idle'

        def _publish(self, frame):
            msg = AudioChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = 'audio/pcm-16k'
            msg.data = array.array('B', frame)
            self._pub.publish(msg)

        def _ready(self, cancel):
            timeout = float(cfg.get('subscriber_wait_s', 3.0))
            if timeout <= 0:
                return
            deadline = time.monotonic() + timeout
            while self._pub.get_subscription_count() < 1:
                if cancel.wait(.02):
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError('No V6 PCM subscriber became ready')

        def _text_cb(self, msg):
            try:
                body = json.loads(msg.data)
                text = body.get('text', '') if isinstance(body, dict) else str(body)
            except (ValueError, TypeError):
                text = msg.data
            if text and self.session.running:
                self.session.enqueue(Request(text, '', cfg.get('language', 'auto')))

    return V6Node()


class TTSPlugin:
    PREFIX = 'tts'

    def __init__(self, plugin_cfg, executor):
        self._cfg = dict(plugin_cfg)
        if self._cfg.get('engine', ENGINE) != ENGINE:
            raise ValueError('V6 plugin cannot use a different engine')
        self._cfg.setdefault('language', 'auto')
        validate_options(self._cfg['language'], self._cfg.get('speed', 1), self._cfg.get('speaker_id', 0))
        self._executor = executor
        self._nodes = {}
        self._idle_nodes = OrderedDict()
        self._lock = threading.RLock()
        self._backend = None
        self._shutdown = False
        self._startup_timing = {'warmup_enabled': bool(self._cfg.get('warmup', True)),
                                'ready_after_enabled_warmup': False}
        if self._cfg.get('backend', 'tensorrt_fp32') != 'tensorrt_fp32':
            raise ValueError('Xiaoyou requires the TensorRT FP32 backend')
        self._backend = acquire_gpu_backend(self._cfg.get('model_dir'),
            cpu_threads=self._cfg.get('cpu_threads', 2),
            expected_manifest_sha256=self._cfg.get('runtime_manifest_sha256'))
        try:
            loaded = time.perf_counter()
            self._startup_timing['asset_prepare_seconds'] = self._backend.asset_prepare_seconds
            self._startup_timing['backend_load_seconds'] = self._backend.backend_load_seconds
            if self._startup_timing['warmup_enabled']:
                self._backend.warmup()
            self._startup_timing['warmup_seconds'] = time.perf_counter()-loaded
            self._startup_timing['ready_after_enabled_warmup'] = True
            log.info('01yuewen startup ready: %s', self._startup_timing)
        except BaseException:
            self._backend.close()
            raise

    def _info(self, args):
        identity = {'name': ENGINE, 'model': MODEL_ID, 'model_id': MODEL_ID,
                    'engine': ENGINE, 'voice_id': '01yuewen'}
        identifier = args.get('instance_id')
        nodes = ([self._nodes[identifier]] if identifier in self._nodes else []) if identifier else list(self._nodes.values())
        metadata = getattr(self._backend, 'metadata', {})
        backend = metadata.get('backend', self._cfg.get('backend', 'tensorrt_fp32'))
        state = 'running' if any(n.state == 'running' for n in nodes) else 'idle'
        frontend_error = getattr(self._backend, 'frontend_error', None)
        if frontend_error:
            state = 'error'
            identity['message'] = frontend_error
        topics = [(node._input_topic, node._output_topic) for node in nodes]
        if not topics:
            # Canvas reads output ports before it starts the TTS instance.
            input_topic = args.get('input_topic') or ''
            topics = [(input_topic, _tts_output_topic(input_topic))]
        return {**identity, 'state': state,
                'backend': backend, 'accelerated': bool(metadata.get('accelerated', False)),
                'generation': metadata.get('streaming', 'natural-text-streaming'),
                'ttft_validated': bool(metadata.get('ttft_validated', False)),
                'jetson_validated': bool(metadata.get('jetson_validated', False)),
                'darvin_validated': bool(metadata.get('darvin_validated', False)),
                'three_process_instances_validated': bool(metadata.get('three_process_instances_validated', False)),
                'language': self._cfg['language'], 'speaker_id': 0, 'speed': 1.0,
                'startup_ready': not self._shutdown and not frontend_error,
                'startup_timing': dict(self._startup_timing),
                'topic_in': [{'topic': source, 'format': 'data/json'} for source, _ in topics if source],
                'topic_out': [{'topic': target, 'format': 'audio/pcm-16k'} for _, target in topics],
                'desc': 'xiaoyou01 FP32 GPU speech; deployment acceptance is reported explicitly.'}

    def _start(self, args):
        key = args.get('instance_id') or args.get('input_topic') or '_default'
        topic = args.get('input_topic') or ''
        if key in self._nodes and self._nodes[key]._input_topic != topic and topic:
            self._remove(key)
        if key not in self._nodes:
            if self._executor is None:
                raise RuntimeError('A ROS executor is required for V6 topic delivery')
            # One publisher owns any output topic within this plugin.
            output = _tts_output_topic(topic)
            for other in list(self._nodes):
                if self._nodes[other]._output_topic == output:
                    self._remove(other)
            suffix = ''.join(c if c.isalnum() else '_' for c in key)
            node = _node(topic, self._backend, suffix, self._cfg)
            try:
                self._executor.add_node(node)
            except Exception:
                node.destroy_node()
                raise
            self._nodes[key] = node
        node = self._nodes[key]
        node.session.start()
        self._idle_nodes.pop(key, None)
        return node

    def _stop_node(self, key):
        """Stop speech completely, retaining a bounded owned DDS transport.

        Reusing this same publisher avoids rediscovery at every start/stop.
        It remains registered and reachable, never a detached/orphan ROS node.
        Topic changes, eviction and shutdown still destroy the whole node.
        """
        self._nodes[key].session.stop()
        self._idle_nodes.pop(key, None)
        self._idle_nodes[key] = None
        while len(self._idle_nodes) > MAX_IDLE_TRANSPORTS:
            self._remove(next(iter(self._idle_nodes)))

    def _remove(self, key):
        node = self._nodes[key]
        node.session.stop()
        self._executor.remove_node(node)
        node.destroy_node()
        del self._nodes[key]
        self._idle_nodes.pop(key, None)

    def dispatch(self, name, args):
        action = args.get('action') if name == 'tts' else name
        if action not in ACTIONS:
            return None
        if action == 'support_language':
            return {'engine': ENGINE, 'languages': list(LANGUAGES)}
        if action == 'config' and any(
            args.get(key) not in (None, '', ENGINE) for key in ('tts_engine', 'engine')
        ):
            return {'state': 'error', 'message': 'xiaoyou01 does not support switching engines'}
        with self._lock:
            if action == 'info':
                return self._info(args)
            if self._shutdown:
                return {'state': 'error', 'message': 'V6 plugin has shut down'}
            try:
                if action == 'config':
                    language = args.get('tts_language', args.get('language', self._cfg['language']))
                    validate_options(language, args.get('speed', 1), args.get('speaker_id', 0))
                    self._cfg['language'] = language
                    return {'status': 'configured', 'engine': ENGINE, 'language': language,
                            'speed': 1.0, 'speaker_id': 0}
                if action == 'start':
                    node = self._start(args)
                    return {'state': node.state, 'topic_in': [{'topic': node._input_topic, 'format': 'data/json'}],
                            'topic_out': [{'topic': node._output_topic, 'format': 'audio/pcm-16k'}]}
                if action == 'speak':
                    text = args.get('text', '')
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError('Nonempty text is required')
                    if not args.get('instance_id') and not args.get('input_topic'):
                        running = [n for n in self._nodes.values() if n.state == 'running']
                        if len(running) > 1:
                            raise ValueError('Multiple V6 output nodes exist; select instance_id for speak')
                        node = running[0] if running else self._start(args)
                    else:
                        node = self._start(args)
                    action_id = 'speak-' + uuid.uuid4().hex[:12]
                    node.session.enqueue(Request(text, action_id, self._cfg['language'], args.get('_trace_id', '')))
                    return {'status': 'queued', 'action_id': action_id}
                keys = [args['instance_id']] if args.get('instance_id') in self._nodes else (
                    [] if args.get('instance_id') else list(self._nodes))
                if action == 'interrupt':
                    cleared = sum(self._nodes[k].session.interrupt() for k in keys)
                    return {'status': 'interrupted', 'nodes': len(keys), 'cleared': cleared}
                for key in keys:
                    if key in self._nodes:
                        self._stop_node(key)
                return {'state': 'idle'}
            except Exception as error:
                log.exception('V6 %s failed', action)
                return {'state': 'error', 'message': str(error)}

    def synthesize_raw(self, text):
        with self._lock:
            if self._shutdown or self._backend is None:
                raise RuntimeError('xiaoyou01 requested model is unavailable')
            language = self._cfg['language']
        return self._backend.synthesize(text, language=language, speed=1)

    def shutdown(self):
        with self._lock:
            if self._shutdown:
                return
            # A failed stop deliberately leaves the backend alive: an active
            # producer must finish before its GPU lease can be released.
            for key in list(self._nodes):
                self._remove(key)
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            self._shutdown = True
