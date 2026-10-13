"""One phanthy_music card: browse a catalogue, play a station, emit music PCM.

A pure actuator. It used to be the mixer as well — it took TTS in on its own
`topic_in`, ducked the song under it and published the combined stream, which is
why the wiring read `TTS → phanthy_music → Speaker`. Speech now belongs to
`audio_mixer`, and this card emits music and nothing else:
`phanthy_music → audio_mixer → Speaker`.

The action set is written for the two callers it actually has — a person
speaking, and the LLM relaying them — rather than for the catalogue's REST
shape. Three rules came out of that:

* **Never make the model guess a code.** `list_genres` returns the catalogue's
  own vocabularies; an unrecognised `genre` degrades to free text instead of
  erroring. Previously the only way to learn a valid code was to fail a search
  and read `error.allowed`.
* **Never make the model keep state.** `next` takes no arguments. The card
  remembers the criteria and what it has already played; `exclude_ids` used to
  be a list the model had to maintain by hand.
* **One error shape.** Every reply's `error` is absent or a dict with a `code`.
"""
import array
from collections import deque
from copy import deepcopy
import threading
import uuid

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from audio_msgs.msg import AudioChunk

from plugins.music_player import MusicPlayer, bounded_number
from plugins.music_catalog.contracts import MusicError
from plugins.music_catalog.service import MusicService
from utils import acp

# Fixed card integration; these values are not user-editable card settings.
CATALOGUE_ENDPOINT = 'https://api.phanthy.com/open'
CATALOGUE_API_KEY = 'pmk_315eeccf191e_NkXhERRw1L3squK-Grffb-a7ds-rXL5X1YhDhpx3-E0'

FORMAT = 'audio/pcm-16k'
OUTPUT_TOPIC = '/perception/music/audio'  # Existing AudioChunk contract.
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=20,
                 durability=DurabilityPolicy.VOLATILE)

# Filters a caller may express. `exclude_ids` is deliberately absent: it is the
# card's bookkeeping now, not the model's.
FILTER_PARAMS = ['query', 'genre', 'language', 'vocal', 'tags', 'artist',
                 'duration_min_s', 'duration_max_s']
CONFIG_PARAMS = ['catalogue_type', 'timeout_ms', 'volume', 'continuous']
HISTORY_LIMIT = 50

ACTIONS = {
    'list_genres': [],
    'play': FILTER_PARAMS + ['track_id'],
    'next': [],
    'previous': [],
    'pause': [],
    'resume': [],
    'stop_music': [],
    'now_playing': [],
    'set_volume': ['volume'],
    'search': FILTER_PARAMS + ['top_k'],
}
# Not in the declared enum, so they never reach the model. `present_to_llm`
# filters these four only for `type: 'processor'` cards, and this card is an
# `actuator` — so leaving them in the enum would hand the model a `stop` that
# tears the ROS node down, sitting right next to the `stop_music` it wants.
# That exact confusion was measured on Orin5 with `tts(action="stop")`.
SYSTEM_ACTIONS = ['start', 'stop', 'info', 'config']
# Old spellings keep dispatching: a stored canvas exec arg, a saved benchmark
# case or an x-hooks binding written against them must not break. Undeclared, so
# the model is only ever offered the new names.
ALIAS_ACTIONS = {'interrupt': 'stop_music', 'status': 'now_playing',
                 'play_by_id': 'play', 'play_by_genre': 'play'}

ACTION_DESCRIPTIONS = {
    'list_genres': '先查可用的曲风/语种/人声代码，再 play。代码来自曲库，不要凭译名猜。',
    'play': '点歌并开始播放：给 genre（见 list_genres）或 query（情绪/场景/歌名/歌词），'
            '也可给 track_id 精确点播；什么都不给 = 随便放。'
            '返回即表示已开始取歌，不代表播完；换歌用 next，停歌用 stop_music。',
    'next': '换下一首。不用传参数 —— 沿用上次 play 的条件，并自动排除刚放过的歌。',
    'previous': '回到上一首，从头开始播（记不住上一首播到哪儿了）。',
    'pause': '暂停当前这首，位置保留，resume 继续。',
    'resume': '继续播放暂停的那首。',
    'stop_music': '停止音乐。不影响说话。',
    'now_playing': '现在放的是什么：歌名、歌手、曲风、进度、音量。要向用户报歌名时先调它。',
    'set_volume': '音乐音量 0–100。不改变说话的音量。',
    'search': '只回答「有没有/有哪些」，不播放。想听直接用 play。',
}

TOOLS = [{
    'name': 'phanthy_music', 'type': 'actuator', 'multiInstance': False,
    'description': '音乐播放器：play 点歌（曲风或自由描述）、next 换歌、list_genres 查可用曲风、'
                   'now_playing 查当前曲目，另有 pause/resume/stop_music/set_volume。'
                   '只输出音乐音频到自己的话题；混音和「说话时压低音乐」由 audio_mixer 卡负责：'
                   'phanthy_music → audio_mixer → Speaker。'
                   'personal_playback 只允许当次播放，不得保存或上传聊天附件。',
    'inputSchema': {
        'type': 'object', 'required': ['action'],
        'properties': {
            'action': {'type': 'string', 'enum': list(ACTIONS)},
            **{key: {'type': 'string', 'description': description} for key, description in {
                'query': '情绪、场景、歌名或歌词片段，最多 200 字',
                'genre': '曲风代码，逗号分隔。取值先用 list_genres；'
                         '不在表里的词会被当作自由描述处理，并在回包的 genre_as_query 里说明',
                'language': '语种代码，逗号分隔；取值见 list_genres',
                'vocal': 'male/female/mixed/instrumental；不确定就留空',
                'tags': '标签，逗号分隔；可用词表见 list_genres',
                'artist': '歌手名或歌手 ID',
            }.items()},
            'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'default': 3},
            'duration_min_s': {'type': 'integer', 'minimum': 0, 'default': 0},
            'duration_max_s': {'type': 'integer', 'minimum': 0, 'default': 0},
            'track_id': {'type': 'string', 'minLength': 1, 'maxLength': 200,
                         'description': '精确点播某首歌；通常来自 search 的结果'},
            'volume': {'type': 'number', 'minimum': 0, 'maximum': 100},
        },
        'x-action-params': {name: {'params': params,
                                   'description': ACTION_DESCRIPTIONS.get(name, name)}
                            for name, params in ACTIONS.items()},
        # Its own channel, not `mouth`: music and speech are mixed downstream, so
        # they do not contend — a song must not make `speak` wait, nor vice versa.
        'x-resource': 'music',
        'x-completion': {
            # Only `play` opens a session. next/previous continue the same one,
            # so they must not register a second pending.
            'actions': ['play'],
            # Holds the `music` channel for "is it playing", but nobody waits on
            # it: a four-minute track must not block `finish`, nor the agent's
            # next call of any kind (pendings_to_wait_for's own-ordering branch).
            'background': True,
            # "The song finished" is a receipt, not a new message.
            'silent_completion': True,
            # 0 = no deadline; only the card's own /acp/complete ends it. A track
            # has no `text` to estimate from, and a static ceiling would reap the
            # pending mid-playback. `play` also returns expected_duration_s.
            'timeout': 0,
        },
        'x-hooks': {
            # Deliberately NOT on_interrupt_all: that fires on every new user
            # turn, and barge-in must stop the robot talking, not the music.
            # The cost, stated plainly: an e-stop does not silence the song.
            'on_interrupt_music': {'action': 'stop_music'},
        },
    },
    'configSchema': {'type': 'object', 'additionalProperties': False, 'properties': {
        'catalogue_type': {'type': 'string', 'enum': ['none', 'mock', 'motus_music'],
                           'default': 'motus_music',
                           'description': '默认使用内置曲库连接；mock 仅用于离线元数据测试'},
        'timeout_ms': {'type': 'integer', 'minimum': 100, 'maximum': 30000, 'default': 3000,
                       'description': '曲库请求总时限（毫秒，含能力查询和重试）',
                       'x-show-when': {'catalogue_type': 'motus_music'}},
        'volume': {'type': 'number', 'minimum': 0, 'maximum': 100, 'default': 50,
                   'description': '音乐音量，不改变说话音量'},
        'continuous': {'type': 'boolean', 'default': True,
                       'description': '一首放完自动续下一首（电台模式）；关掉则播完即停'},
    }},
    # No topic_in: this card no longer mixes. See the module docstring.
    'topic_in': [],
    'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT, 'desc': 'music only (no speech)'}],
}]


class MusicNode(Node):
    def __init__(self, cfg, on_drained=None):
        super().__init__('perception_phanthy_music')
        self.publisher = self.create_publisher(AudioChunk, OUTPUT_TOPIC, QOS)
        self.player = MusicPlayer(
            self._publish, volume=cfg.get('volume', 50), on_drained=on_drained,
            has_reader=lambda: self.publisher.get_subscription_count() > 0)

    def _publish(self, pcm):
        message = AudioChunk()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = FORMAT
        # array('B'), not list(): the ROS uint8[] setter validates element by
        # element, which costs ~10x on a 3200-byte frame ten times a second.
        message.data = array.array('B', pcm)
        self.publisher.publish(message)


class PhanthyMusicPlugin:
    PREFIX = 'phanthy_music'

    def __init__(self, plugin_cfg, executor):
        self._cfg = {key: value for key, value in (plugin_cfg or {}).items()
                     if key in CONFIG_PARAMS}
        self._cfg.setdefault('catalogue_type', 'motus_music')
        self._catalogue = self._new_catalogue(self._cfg)
        self._executor, self._node = executor, None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._catalogue_gate = threading.Lock()
        self._generation = 0
        self._pending_play = None
        # The station: the criteria of the last `play`, so `next` needs no
        # arguments, and what has been played, so it does not repeat itself.
        self._station = {}
        self._history = deque(maxlen=HISTORY_LIMIT)
        self._cursor = -1
        self._action_id = ''

    @staticmethod
    def _new_catalogue(cfg):
        # Spread after cfg so the built-in values always win, whatever a YAML or
        # a UI edit tried to put there.
        return MusicService({**cfg, 'endpoint': CATALOGUE_ENDPOINT,
                             'api_key': CATALOGUE_API_KEY})

    def get_tools(self):
        return deepcopy(TOOLS)

    # ── reporting ────────────────────────────────────────────────────────────
    def _now_playing(self):
        """What is on, in the words someone would use to say it out loud."""
        with self._lock:
            node, track = self._node, dict(self._station.get('track') or {})
            station = {k: v for k, v in (self._station.get('criteria') or {}).items() if v}
            history, cursor = list(self._history), self._cursor
        status = node.player.status() if node else {'playback_state': 'idle'}
        artist = track.get('artist') or {}
        duration = track.get('duration_s') or 0
        position = status.get('position_s', 0.0)
        return {
            'state': 'running' if node else 'idle',
            'playback_state': status.get('playback_state', 'idle'),
            'track': {
                'id': track.get('id', ''), 'title': track.get('title', ''),
                'artist': artist.get('name', '') if isinstance(artist, dict) else str(artist),
                'genre': track.get('genre', ''), 'language': track.get('language', ''),
                'vocal': track.get('vocal'), 'tags': track.get('tags') or [],
                'duration_s': duration, 'cover_url': track.get('cover_url', ''),
                'lyrics_excerpt': track.get('lyrics_excerpt', ''),
                'match_reason': track.get('match_reason', ''),
            } if track else None,
            'position_s': position,
            'remaining_s': round(max(0.0, duration - position), 2) if duration else None,
            'volume': status.get('volume', self._cfg.get('volume', 50)),
            'continuous': bool(self._cfg.get('continuous', True)),
            'station': station,
            'played_count': len(history),
            'can_previous': cursor > 0,
            'attribution': status.get('attribution', ''),
            'error': status.get('error', ''),
            'completion_scope': status.get(
                'completion_scope',
                'source_drained; speaker buffer may still contain audio'),
        }

    def _info(self):
        return {'catalogue_type': self._catalogue.kind,
                'catalogue_configured': self._catalogue.configured(),
                'resolving_track': self._pending_play is not None,
                # No topic_in: this card does not mix. Reported as an empty list
                # rather than omitted so the dashboard shows it has no input.
                'topic_in': [],
                'topic_out': [{'topic': OUTPUT_TOPIC, 'format': FORMAT}],
                **self._now_playing()}

    # ── cancellation ─────────────────────────────────────────────────────────
    def _cancel_pending(self):
        """Invalidate any in-flight catalogue call. Caller holds `_lock`.

        The generation counter is what stops a slow catalogue response from
        resurrecting a song the user already cancelled: the final guard in
        `_fetch_and_play` re-checks it after the network returns.
        """
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
        # state lock that interrupt, status and now_playing need.
        with self._lock:
            self._cancel_pending()
            node, self._node = self._node, None
            action_id = self._action_id
            self._action_id = ''
            if node:
                node.player.request_stop()
        with self._lifecycle_lock:
            self._dispose(node)
        if action_id:
            acp.complete(action_id, 'cancelled', {'reason': 'music card stopped'})

    def _start(self, args):
        # The card used to mix, so an old canvas has `TTS → phanthy_music →
        # Speaker`. Silently ignoring that input would leave a **mute robot
        # reporting 已就绪**: TTS would publish into nothing, and
        # api/config.py's _dropped_inputs cannot catch it (it only fires at two
        # or more topics). Failing with the fix in the message is far better
        # than a legacy mix path, which would keep alive the very code this
        # split exists to delete.
        if args.get('input_topic') or args.get('input_topics'):
            raise MusicError(
                'legacy_wiring',
                'phanthy_music 不再混音：请在 phanthy_music 和 Speaker 之间插入 '
                'audio_mixer 卡，并把 TTS 也接到 audio_mixer。')
        with self._lifecycle_lock:
            with self._lock:
                if self._node:
                    return self._info()        # idempotent re-start
                self._cancel_pending()
                generation = self._generation
            with self._lock:
                if generation != self._generation:
                    return MusicError('cancelled', '启动已取消。').result()
                node = MusicNode(self._cfg, on_drained=self._on_drained)
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

    # ── catalogue ────────────────────────────────────────────────────────────
    def _query(self, catalogue, **args):
        """One catalogue call at a time, and never under `_lock`.

        Non-blocking: a second concurrent request is refused rather than queued,
        so `now_playing`, `stop_music` and the hooks stay responsive while a
        slow lookup is in flight.
        """
        if not self._catalogue_gate.acquire(blocking=False):
            return MusicError('busy', '曲库请求正在进行，请稍后重试。').result()
        try:
            return catalogue.execute(**args)
        finally:
            self._catalogue_gate.release()

    def _vocabulary(self):
        with self._lock:
            catalogue = self._catalogue
        if not self._catalogue_gate.acquire(blocking=False):
            return MusicError('busy', '曲库请求正在进行，请稍后重试。').result()
        try:
            return catalogue.vocabulary()
        finally:
            self._catalogue_gate.release()

    def _known_values(self, field):
        """Values the catalogue accepts for a filter, or () when it cannot say."""
        try:
            with self._lock:
                catalogue = self._catalogue
            caps = catalogue.cached_capabilities()
        except Exception:   # noqa: BLE001
            return ()
        entries = ((caps or {}).get('filters') or {}).get(field) or []
        return tuple(e.get('value') for e in entries
                     if isinstance(e, dict) and e.get('value'))

    def _coerce_filters(self, args):
        """Keep a filter the catalogue knows; demote the rest to free text.

        An unrecognised `genre` used to be a hard `invalid_filter` error, so a
        model that said "city pop" got nothing and had no way to learn the real
        codes except by failing. The strict validation in `contracts` is
        deliberately untouched — the demotion happens here, above it — because a
        *declared* vocabulary violation is still a bug worth reporting.
        """
        filters = {k: args[k] for k in FILTER_PARAMS if args.get(k) not in (None, '')}
        demoted = []
        for field in ('genre', 'language', 'vocal'):
            raw = filters.get(field)
            if not isinstance(raw, str) or not raw.strip():
                continue
            known = self._known_values(field)
            if not known:
                continue
            kept, unknown = [], []
            for value in (v.strip() for v in raw.split(',')):
                if not value:
                    continue
                (kept if value in known else unknown).append(value)
            demoted += unknown
            if kept:
                filters[field] = ','.join(kept)
            else:
                filters.pop(field, None)
        if demoted:
            query = ' '.join(x for x in [filters.get('query', '')] + demoted if x)
            filters['query'] = query[:200]
        return filters, demoted

    def _fetch_and_play(self, *, track_id='', filters=None, remember=True):
        """Resolve one track and hand it to the player.

        The network call happens outside `_lock`, and the result is only acted
        on after re-checking the generation — otherwise a response that arrives
        after the user said "stop" would start the song anyway.
        """
        with self._lock:
            if not self._node:
                raise ValueError('phanthy_music is not running; '
                                 'start the canvas project first')
            self._cancel_pending()
            generation, node, catalogue = self._generation, self._node, self._catalogue
            cancel = self._pending_play = threading.Event()
            exclude = ','.join(list(self._history)[-HISTORY_LIMIT:])
        try:
            if track_id:
                result = self._query(catalogue, track_id=track_id, cancel=cancel)
            else:
                result = self._query(catalogue, **(filters or {}), top_k=1,
                                     exclude_ids=exclude, cancel=cancel)
            if 'error' in result:
                return result
            track = result.get('track') or next(iter(result.get('tracks', [])), None)
            if track is None:
                return {**result,
                        **MusicError('not_found', '没有找到符合条件的歌曲。',
                                     ).result(),
                        'hint': '放宽条件，或改用 query 自由描述',
                        'request_id': result.get('request_id', '')}
            audio = track.get('audio')
            if not isinstance(audio, dict):
                return {**MusicError('not_playable',
                                     '该歌曲只有元数据，没有可播放音频。').result(),
                        'hint': '换一首歌'}
            with self._lock:
                if (cancel.is_set() or generation != self._generation
                        or self._node is not node):
                    return MusicError('cancelled', '播放请求已取消。').result()
                node.player.play(audio.get('url', ''), audio.get('format', 'mp3'),
                                 track['id'], audio.get('expires_at'),
                                 track.get('usage'))
                self._pending_play = None
                if remember and (not self._history or self._history[-1] != track['id']):
                    self._history.append(track['id'])
                    self._cursor = len(self._history) - 1
                self._station['track'] = track
                # `track` last: _open_session folds in _now_playing(), whose own
                # `track` is the flattened display shape. For a play reply the
                # catalogue's full record is the more useful of the two — it is
                # what the model quotes the title and artist from.
                reply = {**result, **self._open_session(track), 'track': track}
            return reply
        finally:
            with self._lock:
                if self._pending_play is cancel:
                    self._pending_play = None

    # ── ACP session ──────────────────────────────────────────────────────────
    def _open_session(self, track):
        """Register this session with ACP. Caller holds `_lock`.

        One `action_id` per *session*, not per track: `next` continues the same
        listening session, and a second pending per skip would have the barrier
        believing two songs were playing at once.
        """
        duration = track.get('duration_s')
        if not self._action_id:
            self._action_id = f'music-{uuid.uuid4().hex[:8]}'
        info = self._info()
        info['action_id'] = self._action_id
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            # Lets agent-core size the pending's timeout to the real track
            # instead of a static ceiling that reaps it mid-playback.
            info['expected_duration_s'] = duration
        return info

    def _close_session(self, status, result=None):
        with self._lock:
            action_id, self._action_id = self._action_id, ''
        if action_id:
            acp.complete(action_id, status, result or {})

    def _on_drained(self, playback_id, state, error):
        """A track reached a terminal state. Called off the output thread.

        Station mode advances here. It must go through the same gate and the
        same generation guard as a manual `next`, or a stopped station comes
        back to life one track later — which is the bug the generation counter
        was introduced to prevent.
        """
        with self._lock:
            if not self._node or state not in ('completed', 'error'):
                return
            if error or not self._cfg.get('continuous', True) or not self._station:
                return
            filters = dict(self._station.get('criteria') or {})
        try:
            reply = self._fetch_and_play(filters=filters)
        except Exception:   # noqa: BLE001
            reply = {'error': {'code': 'internal', 'message': 'auto-advance failed'}}
        if 'error' in reply:
            self._close_session('completed', {'reason': 'station ended'})

    def dispatch(self, name, args):
        # One registered tool with actions, using the existing card protocol.
        if name != self.PREFIX:
            return None
        action = str(args.get('action', 'info'))
        action = ALIAS_ACTIONS.get(action, action)
        try:
            if action == 'list_genres':
                return self._vocabulary()
            if action == 'search':
                # No ROS node and no Speaker needed, and no network under the
                # state lock — "is there any jazz" must work with the card idle.
                with self._lock:
                    catalogue = self._catalogue
                filters, demoted = self._coerce_filters(args)
                reply = self._query(catalogue, **filters,
                                    **({'top_k': args['top_k']} if args.get('top_k') else {}))
                return {**reply, **({'genre_as_query': demoted} if demoted else {}),
                        'available_genres': list(self._known_values('genre'))}
            if action == 'play':
                track_id = args.get('track_id')
                if isinstance(track_id, str) and track_id.strip():
                    with self._lock:
                        self._station = {'criteria': {}}
                    return self._fetch_and_play(track_id=track_id.strip())
                filters, demoted = self._coerce_filters(args)
                with self._lock:
                    # No criteria at all is a valid station: "放点歌" means
                    # anything, and the catalogue picks.
                    self._station = {'criteria': filters}
                reply = self._fetch_and_play(filters=filters)
                return {**reply, **({'genre_as_query': demoted} if demoted else {})}
            if action == 'next':
                with self._lock:
                    filters = dict(self._station.get('criteria') or {})
                return self._fetch_and_play(filters=filters)
            if action == 'previous':
                with self._lock:
                    if self._cursor <= 0:
                        raise ValueError('这是本次播放的第一首，没有上一首了')
                    self._cursor -= 1
                    track_id = self._history[self._cursor]
                # By id, not from a cache: a signed audio URL expires, so going
                # back costs one catalogue call to refresh it.
                return self._fetch_and_play(track_id=track_id, remember=False)
            if action == 'start':
                return self._start(args)
            if action == 'stop':
                self.stop()
                with self._lock:
                    return self._info()
            with self._lock:
                if action in ('info', 'now_playing'):
                    return self._info()
                if action == 'config':
                    return self._config(args)
                if action == 'stop_music':
                    self._cancel_pending()
                    self._station = {}
                if not self._node:
                    # Idle-tolerant: an interrupt hook firing while nothing is
                    # wired is not an error, and must never raise.
                    if action == 'stop_music':
                        return self._info()
                    raise ValueError('phanthy_music is not running; '
                                     'start the canvas project first')
                player = self._node.player
                if action == 'set_volume':
                    player.set_volume(args.get('volume'))
                elif action == 'stop_music':
                    player.interrupt()
                elif action in ('pause', 'resume'):
                    getattr(player, action)()
                else:
                    raise ValueError(f'Unknown phanthy_music action: {action}')
                reply = self._info()
            if action == 'stop_music':
                self._close_session('cancelled', {'reason': 'stopped by request'})
            return reply
        except MusicError as error:
            return error.result()
        except (ValueError, TypeError) as error:
            # One error shape. This branch used to return `error` as a plain
            # string while MusicError returned a dict, so a caller could not
            # read `error.code` without first checking the type.
            return {**MusicError('invalid_argument', str(error)).result(),
                    'state': 'error' if action == 'start'
                             else ('running' if self._node else 'idle')}

    def _config(self, args):
        if set(args) - set(CONFIG_PARAMS) - {'action', 'instance_id', '_trace_id'}:
            raise ValueError('Unknown music config field')
        cfg = {**self._cfg, **{k: args[k] for k in CONFIG_PARAMS if k in args}}
        if 'volume' in cfg:
            cfg['volume'] = bounded_number(cfg['volume'], 0, 100, 'volume')
        catalogue = self._new_catalogue(cfg)
        if any(cfg.get(k) != self._cfg.get(k) for k in ('catalogue_type', 'timeout_ms')):
            self._cancel_pending()
            self._catalogue = catalogue
        self._cfg = cfg
        if self._node:
            self._node.player.set_volume(cfg.get('volume', 50))
        return self._info()
