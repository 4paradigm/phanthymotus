"""Configured catalogue and bounded capability cache, shared by main/subagent tools."""
import asyncio
from copy import deepcopy
import time

from music.contracts import FALLBACK, MusicError, capabilities, envelope, search_request, search_result
from music.providers import discover


class MusicService:
    def __init__(self):
        self._config = None
        self._provider = None
        self._caps = deepcopy(FALLBACK)
        self._expires = 0.0
        self._lock = asyncio.Lock()

    async def prepare(self):
        import config
        settings = await asyncio.to_thread(lambda: config.main.get('desktop_tools', {}).get('music', {}))
        async with self._lock:
            if settings != self._config:
                self._provider = None
                self._caps, self._expires = deepcopy(FALLBACK), 0.0
                kind = settings.get('type', 'none')
                if kind != 'none':
                    factory = discover().get(kind)
                    if factory is None:
                        raise MusicError('invalid_config', '未知音乐曲库 provider。')
                    self._provider = factory(settings)
                self._config = deepcopy(settings)
            if not self._provider:
                raise MusicError('not_configured', '未配置音乐曲库，请在决策核心配置中选择 provider。')
            if time.monotonic() >= self._expires:
                try:
                    self._caps = capabilities(await self._provider.capabilities())
                    self._expires = time.monotonic() + 86400
                except MusicError as exc:
                    # An incompatible protocol is never hidden by fallback metadata.
                    if exc.code in ('schema_mismatch', 'invalid_response', 'unauthorized', 'forbidden'):
                        raise
                    self._caps = deepcopy(FALLBACK)
                    self._expires = time.monotonic() + 60
            return self._provider, deepcopy(self._caps)

    async def execute(self, *, track_id='', **kwargs):
        try:
            provider, caps = await self.prepare()
            mock = self._config.get('type') == 'mock'
            if track_id:
                if not isinstance(track_id, str) or len(track_id) > 200:
                    raise MusicError('invalid_argument', 'track_id 最长 200 字。')
                data = envelope(await provider.track(track_id))
                # Both a versioned flat detail and a versioned {track: ...} envelope
                # are described explicitly in the client compatibility notes.
                track = data.get('track', data)
                if not isinstance(track, dict) or track.get('id') != track_id:
                    raise MusicError('invalid_response', '曲库返回的歌曲 ID 与请求不一致。')
                data = {**data, 'track': track} if 'track' in data else {
                    'schema': data['schema'], 'request_id': data.get('request_id', ''),
                    'track': {k: v for k, v in data.items() if k not in ('schema', 'request_id')}}
                search_result({'schema': data['schema'], 'tracks': [data['track']]},
                              {'top_k': 1, 'exclude_ids': [], 'filters': {}},
                              mock=mock)
            else:
                request = search_request(caps=caps, **kwargs)
                data = search_result(await provider.search(request), request,
                                     mock=mock)
            return {**data, 'usage_notice': 'personal_playback 仅限当次播放；不得下载保存、上传聊天附件或二次分发。',
                    'reply_notice': '如 relaxed 非空，必须如实说明放宽条件；空结果不能编造歌曲。'
                                    'vocal 为 null 表示人声未知，不得推断为男声、女声或纯音乐。',
                    'available_tags': caps['filters'].get('tags', {})}
        except MusicError as exc:
            return exc.result()

    async def tool_hint(self):
        try:
            _, caps = await self.prepare()
            return ' 可用标签（能力缓存）：' + str(caps['filters'].get('tags', {}))[:2500]
        except MusicError:
            return ''


service = MusicService()
