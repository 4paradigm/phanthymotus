"""One phanthy_music card: catalogue search, background playback and speech mix."""
from copy import deepcopy
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from audio_msgs.msg import AudioChunk

from plugins.music_player import MusicPlayer, bounded_number
from plugins.music_catalog.contracts import MusicError
from plugins.music_catalog.service import MusicService

FORMAT = 'audio/pcm-16k'
OUTPUT_TOPIC = '/perception/music/audio'  # Existing AudioChunk/Speaker contract.
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=20,
                 durability=DurabilityPolicy.VOLATILE)
SEARCH_PARAMS = ['query', 'genre', 'language', 'vocal', 'tags', 'artist', 'top_k',
                 'exclude_ids', 'duration_min_s', 'duration_max_s']
CONFIG_PARAMS = ['catalogue_type', 'endpoint', 'api_key', 'timeout_ms', 'volume', 'duck_gain']
ACTIONS = {
    'search': SEARCH_PARAMS,
    'play_by_id': ['track_id'],
    'play_by_genre': [key for key in SEARCH_PARAMS if key != 'top_k'],
    'pause': [], 'resume': [], 'interrupt': [], 'status': [], 'set_volume': ['volume'],
}
INTERNAL_ACTIONS = ['start', 'stop', 'info', 'config', 'duck', 'unduck', 'interrupt_voice']
ACTION_DESCRIPTIONS = {
    'search': '只检索，不播放。换歌时保留条件并传 exclude_ids；如实说明 relaxed。',
    'play_by_id': '必须传 track_id；查询歌曲最新地址后后台播放。',
    'play_by_genre': '必须传 genre 曲风代码；按条件找一首并后台播放，返回选中的歌曲及 relaxed。',
    'interrupt': '立即取消取歌和本卡待播音乐，保留 TTS 通路。',
}
TOOLS = [{
    'name': 'phanthy_music', 'type': 'processor', 'multiInstance': False,
    'description': '在音乐曲库找歌并播放：search 后 play_by_id(track_id)，或 play_by_genre(genre)。'
                   '支持暂停、继续、停歌和音量；TTS → 本卡 → Speaker 支持音乐中对话。'
                   '播放操作返回后台会话，不代表播放完成。停歌用 interrupt。'
                   'personal_playback 只允许当次播放，不得保存或上传聊天附件。',
    'inputSchema': {
        'type': 'object', 'required': ['action'],
        'properties': {
            'action': {'type': 'string', 'enum': list(ACTIONS) + INTERNAL_ACTIONS},
            'input_topic': {'type': 'string'},
            **{key: {'type': 'string', 'description': description} for key, description in {
                'query': '情绪、场景、歌名或歌词片段，最多 200 字',
                'genre': '曲风代码，逗号分隔，如 pop,folk,rnb,rock,jazz；不确定留空',
                'language': '语种代码，逗号分隔：zh,en,ja,ko,th,pt,fr',
                'vocal': 'male/female/mixed/instrumental；不确定留空',
                'tags': '标签，逗号分隔；可用词表随结果返回，未知标签按自由文本处理',
                'artist': '歌手名或歌手 ID',
                'exclude_ids': '换歌时排除的歌曲 ID，逗号分隔，最多 50 个',
            }.items()},
            'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'default': 3},
            'duration_min_s': {'type': 'integer', 'minimum': 0, 'default': 0},
            'duration_max_s': {'type': 'integer', 'minimum': 0, 'default': 0},
            'track_id': {'type': 'string', 'minLength': 1, 'maxLength': 200,
                         'description': '歌曲 ID；play_by_id 必填，会查询最新详情'},
            'volume': {'type': 'number', 'minimum': 0, 'maximum': 100},
        },
        'x-action-params': {name: {'params': params, 'description': ACTION_DESCRIPTIONS.get(name, name)}
                            for name, params in ACTIONS.items()},
        # Background music never holds the mouth/full-song ACP barrier.
        'x-resource': 'music',
        'x-hooks': {
            'on_interrupt_speak': {'action': 'interrupt_voice'},
            'on_interrupt_all': {'action': 'interrupt_voice'},
            'on_interrupt_music': {'action': 'interrupt'},
        },
    },
    'configSchema': {'type': 'object', 'additionalProperties': False, 'properties': {
        'catalogue_type': {'type': 'string', 'enum': ['none', 'mock', 'motus_music'], 'default': 'none',
                           'description': '曲库类型；真实曲库的地址和凭据在本卡配置'},
        'endpoint': {'type': 'string', 'default': '', 'x-sensitive': True,
                     'description': '曲库 HTTPS 服务地址',
                     'x-show-when': {'catalogue_type': 'motus_music'}},
        'api_key': {'type': 'string', 'format': 'password', 'default': '',
                    'description': '曲库 API Key；仅在卡片配置填写，分享 Solution 时清空',
                    'x-show-when': {'catalogue_type': 'motus_music'}},
        'timeout_ms': {'type': 'integer', 'minimum': 100, 'maximum': 30000, 'default': 3000,
                       'description': '曲库请求总时限（毫秒，含能力查询和重试）',
                       'x-show-when': {'catalogue_type': 'motus_music'}},
        'volume': {'type': 'number', 'minimum': 0, 'maximum': 100, 'default': 50,
                   'description': '音乐音量，不改变 TTS 音量'},
        'duck_gain': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 0.15,
                      'description': '对话时音乐音量相对比例'},
    }},
    'topic_in': [{'format': FORMAT, 'desc': 'optional TTS speech'}],
    'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT, 'desc': 'speech + music'}],
}]


class MusicNode(Node):
    def __init__(self, input_topic, cfg):
        super().__init__('perception_phanthy_music')
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


class PhanthyMusicPlugin:
    PREFIX = 'phanthy_music'

    def __init__(self, plugin_cfg, executor):
        self._cfg = {key: value for key, value in plugin_cfg.items() if key in CONFIG_PARAMS}
        self._catalogue = MusicService(self._cfg)
        self._executor, self._node = executor, None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._catalogue_gate = threading.Lock()
        self._generation = 0
        self._pending_play = None

    def get_tools(self):
        return deepcopy(TOOLS)

    def _info(self):
        return {'state': 'running' if self._node else 'idle',
                'catalogue_type': self._catalogue.kind, 'catalogue_configured': self._catalogue.configured(),
                'resolving_track': self._pending_play is not None,
                'topic_in': ([{'topic': self._node.input_topic, 'format': FORMAT}]
                             if self._node and self._node.input_topic else []),
                'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT}],
                **(self._node.player.status() if self._node else {'playback_state': 'idle'})}

    def _cancel_pending(self):
        self._generation += 1
        if self._pending_play is not None:
            self._pending_play.set()
            self._pending_play = None

    def _dispose(self, node):
        if node:
            try:
                node.player.close()
            finally:
                self._executor.remove_node(node)
                node.destroy_node()

    def stop(self):
        # Signal first. Never wait for network/download/close while holding the
        # state lock needed by interrupt, duck and status.
        with self._lock:
            self._cancel_pending()
            node, self._node = self._node, None
            if node:
                node.player.request_stop()
        with self._lifecycle_lock:
            self._dispose(node)

    def _start(self, args):
        topics = args.get('input_topics') or []
        if not isinstance(topics, list) or any(not isinstance(t, str) for t in topics) or len(set(topics)) > 1:
            raise ValueError('phanthy_music accepts one TTS input')
        topic = args.get('input_topic') or (topics[0] if topics else '')
        if not isinstance(topic, str) or topic == OUTPUT_TOPIC:
            raise ValueError('phanthy_music needs a valid input other than its own output')
        with self._lifecycle_lock:
            with self._lock:
                if self._node and self._node.input_topic == topic:
                    return self._info()
                self._cancel_pending()
                generation = self._generation
                old, self._node = self._node, None
                if old:
                    old.player.request_stop()
            self._dispose(old)
            with self._lock:
                if generation != self._generation:
                    return MusicError('cancelled', '启动已取消。').result()
                node = MusicNode(topic, self._cfg)
                added = False
                try:
                    self._executor.add_node(node)
                    added = True
                    node.player.start()  # Thread start only; no network/model load.
                except Exception:
                    node.player.close()
                    if added:
                        self._executor.remove_node(node)
                    node.destroy_node()
                    raise
                self._node = node
                return self._info()

    def _query(self, catalogue, **args):
        if not self._catalogue_gate.acquire(blocking=False):
            return MusicError('busy', '曲库请求正在进行，请稍后重试。').result()
        try:
            return catalogue.execute(**args)
        finally:
            self._catalogue_gate.release()

    def _play(self, action, args):
        field = 'track_id' if action == 'play_by_id' else 'genre'
        if not isinstance(args.get(field), str) or not args[field].strip():
            raise ValueError(f'{action} requires {field}')
        if field == 'genre' and not any(value.strip() for value in args[field].split(',')):
            raise ValueError('play_by_genre requires genre')
        with self._lock:
            if not self._node:
                raise ValueError('phanthy_music is not running; start the canvas project first')
            self._cancel_pending()
            generation, node, catalogue = self._generation, self._node, self._catalogue
            cancel = self._pending_play = threading.Event()
        try:
            if action == 'play_by_id':
                result = self._query(catalogue, track_id=args['track_id'], cancel=cancel)
            else:
                filters = {k: args[k] for k in SEARCH_PARAMS if k in args and k != 'top_k'}
                result = self._query(catalogue, **filters, top_k=1, cancel=cancel)
            if 'error' in result:
                return result
            track = result.get('track') or next(iter(result.get('tracks', [])), None)
            if track is None:
                return {**result, **MusicError('not_found', '没有找到可播放的歌曲。').result(),
                        'request_id': result.get('request_id', '')}
            audio = track.get('audio')
            if not isinstance(audio, dict):
                return MusicError('not_playable', '该歌曲只有元数据，没有可播放音频。').result()
            with self._lock:
                if cancel.is_set() or generation != self._generation or self._node is not node:
                    return MusicError('cancelled', '播放请求已取消。').result()
                node.player.play(audio.get('url', ''), audio.get('format', 'mp3'), track['id'],
                                 audio.get('expires_at'), track.get('usage'))
                self._pending_play = None
                return {**result, 'track': track, **self._info()}
        finally:
            with self._lock:
                if self._pending_play is cancel:
                    self._pending_play = None

    def dispatch(self, name, args):
        # One registered tool with actions, using the existing card protocol.
        if name != self.PREFIX:
            return None
        action = args.get('action', 'info')
        try:
            if action == 'search':
                # No ROS node/Speaker needed, and no network under the state lock.
                with self._lock:
                    catalogue = self._catalogue
                return self._query(catalogue, **{k: args[k] for k in SEARCH_PARAMS if k in args})
            if action in ('play_by_id', 'play_by_genre'):
                return self._play(action, args)
            if action == 'start':
                return self._start(args)
            if action == 'stop':
                self.stop()
                with self._lock:
                    return self._info()
            with self._lock:
                if action in ('info', 'status'):
                    return self._info()
                if action == 'config':
                    if set(args) - set(CONFIG_PARAMS) - {'action', 'instance_id', '_trace_id'}:
                        raise ValueError('Unknown music config field')
                    cfg = {**self._cfg, **{k: args[k] for k in CONFIG_PARAMS if k in args}}
                    for key, high in (('volume', 100), ('duck_gain', 1)):
                        if key in cfg:
                            cfg[key] = bounded_number(cfg[key], 0, high, key)
                    catalogue = MusicService(cfg)
                    if any(cfg.get(k) != self._cfg.get(k) for k in ('catalogue_type', 'endpoint', 'api_key', 'timeout_ms')):
                        self._cancel_pending()
                        self._catalogue = catalogue
                    self._cfg = cfg
                    if self._node:
                        self._node.player.set_volume(cfg.get('volume', 50))
                        self._node.player.duck_gain = cfg.get('duck_gain', 0.15)
                    return self._info()
                if action == 'interrupt':
                    self._cancel_pending()
                if not self._node:
                    if action in ('duck', 'unduck', 'interrupt', 'interrupt_voice'):
                        return self._info()
                    raise ValueError('phanthy_music is not running; start the canvas project first')
                player = self._node.player
                if action == 'set_volume':
                    player.set_volume(args.get('volume'))
                elif action in ('pause', 'resume', 'interrupt', 'interrupt_voice'):
                    getattr(player, action)()
                elif action in ('duck', 'unduck'):
                    player.hearing(action == 'duck')
                else:
                    return None
                return self._info()
        except MusicError as error:
            return error.result()
        except (ValueError, TypeError) as error:
            return {'error': str(error), 'state': 'error' if action == 'start' else ('running' if self._node else 'idle')}
