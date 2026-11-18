"""Xiaoyou adapter for the public tts tool and ordered ROS audio delivery."""
import array
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict

from .runtime.speech_session import Request, SpeechSession
from .runtime.synthesis import acquire_gpu_backend, validate_options

log = logging.getLogger(__name__)
ENGINE = 'xiaoyou-zh-en'
MODEL_ID = 'xiaoyou01'
LANGUAGES = ('zh', 'en')
ACTIONS = ('start', 'stop', 'speak', 'info', 'config', 'interrupt', 'support_language')

# Keep only explicitly owned, stopped transports; no worker survives a stop.
MAX_IDLE_TRANSPORTS = 4

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
            self._output_topic = self._input_topic + '/tts' if self._input_topic else '/perception/tts'
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
                'topic_in': [{'topic': n._input_topic, 'format': 'data/json'} for n in nodes if n._input_topic],
                'topic_out': [{'topic': n._output_topic, 'format': 'audio/pcm-16k'} for n in nodes],
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
            output = topic + '/tts' if topic else '/perception/tts'
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
