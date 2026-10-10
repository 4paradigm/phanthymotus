"""Per-card catalogue metadata and provider, independent of Agent Core and ROS."""
from copy import deepcopy
import threading
import time

from plugins.music_catalog.contracts import (
    FALLBACK, MusicError, capabilities, envelope, search_request, search_result)
from plugins.music_catalog.providers import discover
from plugins.music_catalog.providers.motus_music import MotusMusicProvider, check_cancel


class MusicService:
    def __init__(self, settings):
        self.kind = settings.get('catalogue_type', 'none')
        self.timeout_ms = settings.get('timeout_ms', 3000)
        if self.kind not in ('none', 'mock', 'motus_music'):
            raise ValueError('未知音乐曲库类型。')
        if type(self.timeout_ms) is not int or not 100 <= self.timeout_ms <= 30000:
            raise ValueError('曲库超时须为 100–30000 毫秒。')
        self._endpoint = settings.get('endpoint', '')
        self._key = settings.get('api_key', '')
        if not isinstance(self._endpoint, str) or not isinstance(self._key, str):
            raise ValueError('曲库地址和 API Key 须为字符串。')
        self._provider = (MotusMusicProvider({'endpoint': self._endpoint, 'api_key': self._key,
                                             'timeout_ms': self.timeout_ms})
                          if self.kind == 'motus_music' and self.configured() else None)
        self._caps = deepcopy(FALLBACK)
        self._expires = 0.0
        self._gate = threading.Lock()

    def configured(self):
        return self.kind == 'mock' or (self.kind == 'motus_music' and bool(
            self._endpoint and self._key))

    def _prepare(self, context):
        if self.kind == 'none':
            raise MusicError('not_configured', '未配置音乐曲库，请设置 phanthy_music 卡片。')
        if self._provider is None:
            if not self.configured():
                raise MusicError('not_configured', 'phanthy_music 内置曲库连接未就绪，请检查卡片版本。')
            self._provider = discover()[self.kind]({
                'endpoint': self._endpoint,
                'api_key': self._key,
                'timeout_ms': self.timeout_ms})
        if time.monotonic() >= self._expires:
            try:
                self._caps = capabilities(self._provider.capabilities(**context))
                self._expires = time.monotonic() + 86400
            except MusicError as exc:
                if exc.code in ('schema_mismatch', 'invalid_response', 'unauthorized', 'forbidden', 'cancelled'):
                    raise
                self._caps, self._expires = deepcopy(FALLBACK), time.monotonic() + 60
        return self._provider, self._caps

    def execute(self, *, track_id='', cancel=None, **kwargs):
        # Do not queue unbounded HTTP workers or block player controls behind
        # a slow catalogue. Capabilities and quota state have one owner.
        if not self._gate.acquire(blocking=False):
            return MusicError('busy', '曲库请求正在进行，请稍后重试。').result()
        try:
            check_cancel(cancel)
            context = {'deadline': time.monotonic() + self.timeout_ms / 1000, 'cancel': cancel}
            provider, caps = self._prepare(context)
            if track_id:
                if not isinstance(track_id, str) or len(track_id) > 200:
                    raise MusicError('invalid_argument', 'track_id 最长 200 字。')
                data = envelope(provider.track(track_id, **context))
                track = data.get('track', data)
                if not isinstance(track, dict) or track.get('id') != track_id:
                    raise MusicError('invalid_response', '曲库返回的歌曲 ID 与请求不一致。')
                data = {**data, 'track': track} if 'track' in data else {
                    'schema': data['schema'], 'request_id': data.get('request_id', ''),
                    'track': {k: v for k, v in data.items() if k not in ('schema', 'request_id')}}
                search_result({'schema': data['schema'], 'tracks': [data['track']]},
                              {'top_k': 1, 'exclude_ids': [], 'filters': {}}, mock=self.kind == 'mock')
            else:
                request = search_request(caps=caps, **kwargs)
                data = search_result(provider.search(request, **context), request, mock=self.kind == 'mock')
            check_cancel(cancel)
            return {**data, 'available_tags': deepcopy(caps['filters'].get('tags', {})),
                    'usage_notice': 'personal_playback 仅限当次播放；不得保存音频、上传聊天附件或二次分发。',
                    'reply_notice': 'relaxed 非空须如实说明放宽条件；空结果不能编造歌曲；vocal 为 null 表示人声未知。'}
        except MusicError as exc:
            return exc.result()
        finally:
            self._gate.release()
