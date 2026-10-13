"""audio_mixer — one speaker, several audio sources, speech always on top.

A speaker driver subscribes to exactly **one** topic and does not mix (checked
across G1, R1, Go2, T800, Bumi, U1 Pro). So the moment a robot has both a voice
and a background bed, something upstream has to sum them. That used to be the
music card, which is why it took TTS in on its own input and published the
combined stream. This card is that job, on its own.

Wiring: `TTS → audio_mixer`, `phanthy_music → audio_mixer`, `audio_mixer → Speaker`.

**One in-port, roles from config.** The canvas lets several connections land on
one card and `api/config.py:_resolve_input_topics` passes all of them — but it
filters connections on `toCardId` alone and never reads `toPortIdx`, so which
*port* a wire landed on does not survive to `start`. Declaring a "speech" port
and a "background" port would therefore be a lie resolved by a coin flip;
`echo_cancellation` refused a second in-port for exactly this reason. Roles come
from `speech_topics` instead, which defaults to asking the local TTS card what
it publishes.

**Two tiers, not an N-way priority lattice.** One speaker carries one foreground
voice over one background bed. A general priority matrix would be a setting
nobody can reason about and the hardware cannot honour anyway.
"""
from __future__ import annotations

import array
import logging
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from audio_msgs.msg import AudioChunk

from plugins.audio_mix import Mixer, bounded_number

log = logging.getLogger(__name__)

FORMAT = 'audio/pcm-16k'
OUTPUT_TOPIC = '/perception/audio/mixed'
SPEECH_AUTO = 'auto'
CONFIG_PARAMS = ['speech_topics', 'duck_gain', 'duck_ramp_ms', 'duck_hold_ms',
                 'background_gain']
# duck / unduck / flush_speech are reachable but undeclared to the model: mixing
# is wiring, not something an agent decides. flush_speech is bound to the
# interrupt hooks; duck/unduck exist for the dashboard and for a driver that
# knows about speech agent-core cannot see.
INTERNAL_ACTIONS = ['start', 'stop', 'info', 'config', 'duck', 'unduck',
                    'flush_speech']

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=20,
                 durability=DurabilityPolicy.VOLATILE)

TOOLS = [{
    'name': 'audio_mixer',
    'type': 'processor',
    'multiInstance': False,
    'description': '音频混音 — 把说话（TTS）和背景音乐混成一路送给扬声器，说话时自动压低音乐。'
                   '接法：TTS → 本卡、phanthy_music → 本卡、本卡 → Speaker。'
                   '说话永远不被压低、不被打断；音乐只被压低，不会暂停。',
    'inputSchema': {
        'type': 'object',
        'required': ['action'],
        'properties': {
            'action': {'type': 'string', 'enum': list(INTERNAL_ACTIONS)},
            'input_topic': {'type': 'string'},
            'input_topics': {'type': 'array', 'items': {'type': 'string'}},
        },
        # No x-action-params on purpose: there is no action here worth offering a
        # model. Everything it would want is on phanthy_music or tts.
        'x-resource': 'speaker_mix',
        'x-hooks': {
            'on_interrupt_speak': {'action': 'flush_speech'},
            'on_interrupt_all': {'action': 'flush_speech'},
        },
    },
    'configSchema': {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'speech_topics': {
                'type': 'string', 'default': SPEECH_AUTO, 'scope': 'shared',
                'title': '哪些输入算「说话」',
                # oneOf is rebuilt per get_tools() from the live TTS topics,
                # exactly like echo_cancellation.robot_voice_topic.
                'oneOf': [{'const': SPEECH_AUTO, 'title': '自动 — 跟随本机 TTS 卡'}],
                'description': '从接进本卡的话题里选哪些算「说话」：说话永远听得见、'
                               '不被压低，其余输入算背景音乐。auto = 跟随本机 TTS 卡。'
                               '下拉项在卡片启动、连线解析之后才会列全。'},
            'duck_gain': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 0.15,
                          'title': '说话时音乐压到', 'description': '0 = 全静音，1 = 不压低'},
            'duck_ramp_ms': {'type': 'integer', 'minimum': 20, 'maximum': 1000,
                             'default': 200, 'title': '压低/恢复渐变时长（毫秒）',
                             'description': '太短会有可听见的咔嗒声', 'advanced': True},
            'duck_hold_ms': {'type': 'integer', 'minimum': 0, 'maximum': 5000,
                             'default': 300, 'title': '说完后保持压低（毫秒）',
                             'description': '句与句之间不要来回起伏', 'advanced': True},
            'background_gain': {'type': 'number', 'minimum': 0, 'maximum': 100,
                                'default': 100, 'title': '背景总音量',
                                'description': '每首歌的音量由 phanthy_music 自己控制',
                                'advanced': True},
        }},
    'topic_in': [{'format': FORMAT, 'desc': 'speech and background sources'}],
    'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT,
                   'desc': 'speech mixed over ducked background'}],
}]


class MixerNode(Node):
    def __init__(self, topics, speech_topics, cfg):
        super().__init__('perception_audio_mixer')
        self.topics = list(topics)
        self.speech_topics = set(speech_topics)
        self.publisher = self.create_publisher(AudioChunk, OUTPUT_TOPIC, QOS)
        self.mixer = Mixer(
            self._publish,
            duck_gain=cfg.get('duck_gain', 0.15),
            duck_ramp_ms=cfg.get('duck_ramp_ms', 200),
            duck_hold_ms=cfg.get('duck_hold_ms', 300),
            background_gain=cfg.get('background_gain', 100),
            has_reader=lambda: self.publisher.get_subscription_count() > 0)
        self.subscriptions_by_topic = {}
        for topic in self.topics:
            speech = topic in self.speech_topics
            self.subscriptions_by_topic[topic] = self.create_subscription(
                AudioChunk, topic,
                (lambda message, _speech=speech: self._on_chunk(message, _speech)),
                QOS)

    def _publish(self, pcm):
        message = AudioChunk()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = FORMAT
        # array('B'), not list(): the ROS uint8[] setter validates element by
        # element, which costs ~10x on a 3200-byte frame published 10x a second.
        message.data = array.array('B', pcm)
        self.publisher.publish(message)

    def _on_chunk(self, message, speech):
        if message.format != FORMAT:
            self.mixer.error = f'{"Speech" if speech else "Background"} input must be {FORMAT}'
            return
        self.mixer.feed(bytes(message.data), speech=speech)

    def start(self):
        self.mixer.start()

    def retire(self):
        self.mixer.close()


class AudioMixerPlugin:
    PREFIX = 'audio_mixer'

    def __init__(self, plugin_cfg, executor, tts=None):
        self._cfg = {k: v for k, v in (plugin_cfg or {}).items() if k in CONFIG_PARAMS}
        self._executor = executor
        self._tts = tts
        self._node = None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        log.info('[audio_mixer] plugin init: speech_topics=%s, duck_gain=%s, tts_handle=%s',
                 self._cfg.get('speech_topics', SPEECH_AUTO),
                 self._cfg.get('duck_gain', 0.15), 'yes' if tts else 'no')

    # ── the speech role ──────────────────────────────────────────────────────
    def _tts_topics(self):
        """Audio topics the TTS card publishes, asked through its own `info`.

        Via the dispatch action the dashboard uses, **not** by reaching into
        TTSPlugin's attributes. `echo_cancellation._tts_audio_topics` already
        said why and I wrote it the other way anyway: TTSPlugin is a facade over
        a swappable engine, so it has no `_nodes` of its own — the node dict
        lives inside whichever engine is loaded, and the output topic is on
        `SherpaOnnxTTSPlugin`. Guessing attribute names returned nothing, so
        `auto` resolved to an empty speech set and **every** input was treated
        as background music, including the TTS. Measured on Orin5.

        Never raises: a TTS in a bad state must not stop this card answering.
        """
        if self._tts is None:
            return []
        try:
            info = self._tts.dispatch('tts', {'action': 'info'}) or {}
            topics = [str(entry.get('topic') or '')
                      for entry in (info.get('topic_out') or [])
                      if (entry.get('format') or '') == FORMAT]
        except Exception as error:   # noqa: BLE001
            log.debug('[audio_mixer] could not ask TTS for its topics: %s', error)
            return []
        return list(dict.fromkeys(t for t in topics if t))

    def _resolve_speech(self, bound_topics):
        """Which of the bound topics count as speech, and how we decided."""
        configured = str(self._cfg.get('speech_topics') or SPEECH_AUTO).strip()
        if configured and configured != SPEECH_AUTO:
            wanted = {t.strip() for t in configured.split(',') if t.strip()}
            return {t for t in bound_topics if t in wanted}, 'configured'
        return {t for t in bound_topics if t in set(self._tts_topics())}, SPEECH_AUTO

    def get_tools(self):
        """The dropdown lists the topics actually wired into this card.

        It used to list what the local TTS plugin publishes, which was a guess
        at *who the speech source is*. But the thing this card has to assign a
        role to is its **inputs** — and a TTS in another container, a driver's
        own voice card, or a second audio source never appears in that guess.
        The inputs are known exactly, from the connections the canvas resolved,
        so that is what the operator should be choosing among.
        """
        from copy import deepcopy
        tools = deepcopy(TOOLS)
        options = tools[0]['configSchema']['properties']['speech_topics']['oneOf']
        known = {SPEECH_AUTO}
        with self._lock:
            node = self._node
        speech = self._resolve_speech(node.topics)[0] if node else set()
        for topic in (node.topics if node else []):
            if topic not in known:
                # Say which one `auto` already picked, so choosing explicitly is
                # a confirmation rather than a guess.
                suffix = ' — auto 认为这是说话' if topic in speech else ''
                options.append({'const': topic, 'title': f'{topic}{suffix}'})
                known.add(topic)
        configured = str(self._cfg.get('speech_topics') or '').strip()
        if configured and configured not in known:
            # A topic set in config.yaml stays selectable even when its publisher
            # is down; otherwise the form silently resets it to `auto`.
            options.append({'const': configured, 'title': f'{configured}（已配置，当前未接入）'})
        return tools

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _topics_from(self, args):
        topics = args.get('input_topics')
        if isinstance(topics, list):
            clean = [t for t in dict.fromkeys(topics) if isinstance(t, str) and t]
        else:
            one = args.get('input_topic')
            clean = [one] if isinstance(one, str) and one else []
        if OUTPUT_TOPIC in clean:
            raise ValueError('audio_mixer cannot take its own output as an input')
        return clean

    def _start(self, args):
        topics = self._topics_from(args)
        if not topics:
            raise ValueError('audio_mixer needs at least one audio input; connect '
                             'TTS and phanthy_music to it')
        retired = None
        with self._lock:
            node = self._node
            if node is not None and node.topics == topics:
                return self._info()        # idempotent re-start
            if node is not None:
                retired, self._node = node, None
        if retired is not None:
            self._dispose(retired)

        speech, _how = self._resolve_speech(topics)
        node = MixerNode(topics, speech, dict(self._cfg))
        # Registered before started, so a concurrent stop can always find and
        # cancel it. A started-but-unregistered node keeps a live subscription
        # and an output thread nothing can ever reach.
        with self._lock:
            if self._node is not None:
                # Someone else won the race; theirs is already registered.
                node.destroy_node()
                return self._info()
            self._executor.add_node(node)
            self._node = node
        try:
            node.start()
        except Exception:
            with self._lock:
                if self._node is node:
                    self._node = None
            self._dispose(node)
            raise
        log.info('[audio_mixer] started: speech=%s background=%s',
                 sorted(speech), sorted(set(topics) - speech))
        return self._info()

    def _dispose(self, node):
        with self._lifecycle_lock:
            try:
                node.retire()
            finally:
                try:
                    self._executor.remove_node(node)
                finally:
                    # destroy_node, not just remove_node: otherwise the publisher
                    # and the ROS node name leak for the life of the process.
                    node.destroy_node()

    def _stop(self):
        with self._lock:
            node, self._node = self._node, None
        if node is not None:
            node.mixer.request_stop()      # signal before taking any slow path
            self._dispose(node)
        return self._info()

    def _info(self):
        with self._lock:
            node = self._node
        if node is None:
            return {'state': 'idle', 'topic_in': [], 'speech_topics': [],
                    'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT}]}
        speech, how = self._resolve_speech(node.topics)
        info = {
            'state': 'running',
            # Every bound topic must be reported, with its role. api/config.py's
            # _dropped_inputs fails a card's start when it was handed >=2 topics
            # and info() does not account for all of them — which is exactly the
            # check that makes a silently-ignored second connection visible.
            'topic_in': [{'topic': t, 'format': FORMAT,
                          'role': 'speech' if t in speech else 'background'}
                         for t in node.topics],
            'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT}],
            'speech_topics': sorted(speech),
            'speech_resolved_by': how,
            **node.mixer.status(),
        }
        if not speech:
            # The only way an operator discovers that `auto` resolved to nothing:
            # without a speech source every input is background, so the robot's
            # voice would be ducked by its own music.
            info['degraded'] = ('no speech source identified — every input is being '
                                'treated as background music')
        return info

    # ── MCP ──────────────────────────────────────────────────────────────────
    def dispatch(self, name, args):
        if name != self.PREFIX:
            return None
        action = str(args.get('action', 'info'))
        try:
            if action == 'start':
                return self._start(args)
            if action == 'stop':
                return self._stop()
            if action in ('info', 'status'):
                return self._info()
            if action == 'config':
                return self._config(args)
            with self._lock:
                node = self._node
            if node is None:
                # Idle-tolerant: a hook firing while nothing is wired is not an
                # error, and an interrupt must never raise.
                if action in ('duck', 'unduck', 'flush_speech'):
                    return self._info()
                raise ValueError('audio_mixer is not running; start the canvas '
                                 'project first')
            if action == 'flush_speech':
                node.mixer.flush_speech()
            elif action in ('duck', 'unduck'):
                node.mixer.duck(action == 'duck')
            else:
                raise ValueError(f'Unknown audio_mixer action: {action}')
            return self._info()
        except (ValueError, TypeError) as error:
            return {'error': {'code': 'invalid_argument', 'message': str(error)},
                    'state': 'error' if action == 'start' else self._info()['state']}

    def _config(self, args):
        unknown = [k for k in args
                   if k not in CONFIG_PARAMS + ['action', 'instance_id', '_trace_id']]
        if unknown:
            raise ValueError(f'Unknown audio_mixer config field: {unknown[0]}')
        with self._lock:
            for key in CONFIG_PARAMS:
                if args.get(key) is not None:
                    self._cfg[key] = args[key]
            node = self._node
        for name, (low, high) in (('duck_gain', (0, 1)), ('duck_ramp_ms', (20, 1000)),
                                  ('duck_hold_ms', (0, 5000)),
                                  ('background_gain', (0, 100))):
            if self._cfg.get(name) is not None:
                bounded_number(self._cfg[name], low, high, name)
        if node is None:
            return self._info()
        node.mixer.configure(**{k: self._cfg.get(k) for k in
                                ('duck_gain', 'duck_ramp_ms', 'duck_hold_ms',
                                 'background_gain')})
        # A role is baked into each subscription's callback at construction, so
        # changing speech_topics has to rebuild the node. Assigning
        # node.speech_topics would update the attribute and route nothing
        # differently — the card would report the new roles and mix by the old.
        if self._resolve_speech(node.topics)[0] != node.speech_topics:
            return self._start({'input_topics': list(node.topics)})
        return self._info()

    def stop(self):
        self._stop()
