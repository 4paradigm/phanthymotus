"""Music playback card: one optional TTS input, one mixed PCM output."""
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from audio_msgs.msg import AudioChunk

from plugins.music_player import MusicPlayer, bounded_number

FORMAT = 'audio/pcm-16k'
OUTPUT_TOPIC = '/perception/music/audio'
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=20,
                 durability=DurabilityPolicy.VOLATILE)

ACTIONS = {
    'start': ['input_topic'], 'stop': [], 'info': [], 'config': ['volume', 'duck_gain'],
    'play': ['url', 'audio_format', 'track_id', 'expires_at', 'usage'],
    'pause': [], 'resume': [], 'interrupt': [], 'status': [], 'set_volume': ['volume'],
    'duck': [], 'unduck': [], 'interrupt_voice': [],
}
TOOLS = [{
    'name': 'music', 'type': 'processor', 'multiInstance': False,
    'description': '播放 HTTPS mp3/wav 音乐；可暂停、恢复、停止和调节音乐音量。'
                   '将 TTS 接到本卡输入，再把本卡输出接 Speaker，可在音乐中对话。'
                   'play 返回后台播放会话，不代表播放完成。停歌用 interrupt，stop 会关闭整个卡片。',
    'inputSchema': {
        'type': 'object', 'required': ['action'],
        'properties': {
            'action': {'type': 'string', 'enum': list(ACTIONS)},
            'input_topic': {'type': 'string'},
            'url': {'type': 'string', 'description': '本次播放的公网 HTTPS 音频地址'},
            'audio_format': {'type': 'string', 'enum': ['mp3', 'wav'], 'default': 'mp3'},
            'track_id': {'type': 'string', 'description': '歌曲 ID，地址失效时用于向曲库刷新'},
            'expires_at': {'type': ['string', 'null'], 'description': '音频地址到期时间，ISO 8601；null 表示长期有效'},
            'usage': {'type': 'object', 'properties': {
                'scope': {'type': 'string', 'enum': ['personal_playback']},
                'attribution': {'type': 'string'}}, 'required': ['scope']},
            'volume': {'type': 'number', 'minimum': 0, 'maximum': 100},
            'duck_gain': {'type': 'number', 'minimum': 0, 'maximum': 1},
        },
        'x-action-params': {name: {'params': params} for name, params in ACTIONS.items()},
        # No full-song x-completion: music is a background session. Otherwise
        # finish()/the mouth barrier would hold the conversation for minutes.
        'x-resource': 'music',
        'x-hooks': {
            'on_hearing': {'action': 'duck'},
            'on_hearing_end': {'action': 'unduck'},
            'on_interrupt_speak': {'action': 'interrupt_voice'},
            'on_interrupt_all': {'action': 'interrupt_voice'},
            'on_interrupt_music': {'action': 'interrupt'},
        },
    },
    'configSchema': {'type': 'object', 'properties': {
        'volume': {'type': 'number', 'minimum': 0, 'maximum': 100, 'default': 50,
                   'scope': 'shared', 'description': '音乐音量，不改变 TTS 音量'},
        'duck_gain': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 0.15,
                      'scope': 'shared', 'description': '对话时音乐音量相对比例'},
    }},
    'topic_in': [{'format': FORMAT, 'desc': 'optional TTS speech'}],
    'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT, 'desc': 'speech + music'}],
}]


class MusicNode(Node):
    def __init__(self, input_topic, cfg):
        super().__init__('perception_music')
        self.input_topic = input_topic
        self.publisher = self.create_publisher(AudioChunk, OUTPUT_TOPIC, QOS)
        self.player = MusicPlayer(self._publish, volume=cfg.get('volume', 50),
                                  duck_gain=cfg.get('duck_gain', 0.15),
                                  has_reader=lambda: self.publisher.get_subscription_count() > 0)
        self.subscription = (self.create_subscription(AudioChunk, input_topic, self._voice, QOS)
                             if input_topic else None)

    def _publish(self, pcm):
        message = AudioChunk()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format, message.data = FORMAT, list(pcm)
        self.publisher.publish(message)

    def _voice(self, message):
        if message.format != FORMAT:
            self.player.error = 'TTS input must be audio/pcm-16k'
            return
        self.player.feed_voice(bytes(message.data))


class MusicPlugin:
    PREFIX = 'music'

    def __init__(self, plugin_cfg, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor
        self._node = None
        self._lock = threading.RLock()

    def get_tools(self):
        return TOOLS

    def _info(self):
        return {'state': 'running' if self._node else 'idle',
                'topic_in': ([{'topic': self._node.input_topic, 'format': FORMAT}]
                             if self._node and self._node.input_topic else []),
                'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT}],
                **(self._node.player.status() if self._node else {'playback_state': 'idle'})}

    def stop(self):
        with self._lock:
            node, self._node = self._node, None
            if node:
                node.player.close()
                self._executor.remove_node(node)
                node.destroy_node()

    def dispatch(self, name, args):
        action = args.get('action', 'info') if name == self.PREFIX else name
        with self._lock:
            try:
                if action in ('info', 'status'):
                    return self._info()
                if action == 'stop':
                    self.stop()
                    return self._info()
                if action == 'config':
                    cfg = dict(self._cfg)
                    for key, high in (('volume', 100), ('duck_gain', 1)):
                        if key in args:
                            cfg[key] = bounded_number(args[key], 0, high, key)
                    self._cfg = cfg
                    if self._node:
                        self._node.player.set_volume(cfg.get('volume', 50))
                        self._node.player.duck_gain = cfg.get('duck_gain', 0.15)
                    return self._info()
                if action == 'start':
                    topics = args.get('input_topics') or []
                    if len(set(topics)) > 1:
                        raise ValueError('music accepts one TTS input; connect its output only to Speaker')
                    topic = args.get('input_topic') or (topics[0] if topics else '')
                    if topic == OUTPUT_TOPIC:
                        raise ValueError('music input cannot subscribe to its own output')
                    if self._node and self._node.input_topic == topic:
                        return self._info()
                    self.stop()
                    node = MusicNode(topic, self._cfg)
                    added = False
                    try:
                        self._executor.add_node(node)
                        added = True
                        node.player.start()
                    except Exception:
                        node.player.close()
                        if added:
                            self._executor.remove_node(node)
                        node.destroy_node()
                        raise
                    self._node = node
                    return self._info()
                if not self._node:
                    if action in ('duck', 'unduck', 'interrupt', 'interrupt_voice'):
                        return self._info()  # Global hooks never auto-start a card.
                    raise ValueError('Music card is not running; start the canvas project first')
                player = self._node.player
                if action == 'play':
                    player.play(args.get('url', ''), args.get('audio_format', 'mp3'),
                                args.get('track_id', ''), args.get('expires_at'), args.get('usage'))
                elif action == 'set_volume':
                    player.set_volume(args.get('volume'))
                elif action in ('pause', 'resume', 'interrupt', 'interrupt_voice'):
                    getattr(player, action)()
                elif action in ('duck', 'unduck'):
                    player.hearing(action == 'duck')
                else:
                    return None
                return self._info()
            except (ValueError, TypeError) as error:
                return {'error': str(error), 'state': 'error' if action == 'start' else ('running' if self._node else 'idle')}
