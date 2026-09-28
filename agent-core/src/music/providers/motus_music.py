"""Generic HTTPS client for motus.music/1. No service address or credential defaults."""
import asyncio
import json
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import aiohttp

from music.contracts import MusicError, envelope, https_url


def retry_seconds(value):
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 1.0
    except (ValueError, TypeError):
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 1.0


class MotusMusicProvider:
    def __init__(self, config):
        self.endpoint = https_url(str(config.get('endpoint') or '')).rstrip('/')
        if '?' in self.endpoint:
            raise MusicError('invalid_config', '曲库 endpoint 不能包含 query。')
        self._key = str(config.get('api_key') or '')
        if not self._key or '\r' in self._key or '\n' in self._key:
            raise MusicError('invalid_config', '请配置音乐曲库 API Key。')
        timeout = config.get('timeout_ms', 3000)
        if type(timeout) is not int or not 100 <= timeout <= 30000:
            raise MusicError('invalid_config', '音乐曲库超时须为 100–30000 毫秒。')
        self.timeout = timeout / 1000
        self._retry_at = 0.0

    async def _request(self, method, path, payload=None):
        now = time.monotonic()
        if now < self._retry_at:
            raise MusicError('rate_limited', '曲库限流，请稍后重试。', retry_after=self._retry_at - now)
        deadline = now + self.timeout
        headers = {'Authorization': f'Bearer {self._key}', 'Accept': 'application/json'}
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                for attempt in range(2):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    async with session.request(method, self.endpoint + path, json=payload,
                                               allow_redirects=False,
                                               timeout=aiohttp.ClientTimeout(total=remaining)) as response:
                        raw = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            raw.extend(chunk)
                            if len(raw) > 1024 * 1024:
                                raise MusicError('invalid_response', '曲库响应超过 1 MiB。')
                        try:
                            data = json.loads(raw)
                        except (ValueError, UnicodeError):
                            data = {}
                        request_id = str(data.get('request_id', ''))[:200] if isinstance(data, dict) else ''
                        if 200 <= response.status < 300:
                            return envelope(data)
                        retry = response.status == 429 or 500 <= response.status < 600
                        delay = retry_seconds(response.headers.get('Retry-After')) if response.status == 429 else 0.2
                        if response.status == 429:
                            self._retry_at = time.monotonic() + delay
                        if retry and attempt == 0 and delay < deadline - time.monotonic():
                            await asyncio.sleep(delay)
                            continue
                        code = {400: 'schema_mismatch', 401: 'unauthorized', 403: 'forbidden',
                                404: 'not_found', 422: 'invalid_filter', 429: 'rate_limited'}.get(
                                    response.status, 'internal' if response.status >= 500 else 'http_error')
                        error = data.get('error') if isinstance(data, dict) else None
                        allowed = error.get('allowed') if isinstance(error, dict) else None
                        allowed = ([x[:80] for x in allowed[:100] if isinstance(x, str)]
                                   if isinstance(allowed, list) else None)
                        # Never expose request URLs, credentials or an unbounded server message.
                        raise MusicError(code, f'曲库请求失败（HTTP {response.status}），请勿编造歌曲。',
                                         request_id=request_id, allowed=allowed,
                                         retry_after=delay if response.status == 429 else None)
        except (asyncio.TimeoutError, aiohttp.ClientError):
            raise MusicError('unavailable', '曲库暂时不可用，请稍后重试。') from None

    async def capabilities(self):
        return await self._request('GET', '/v1/capabilities')

    async def search(self, request):
        return await self._request('POST', '/v1/music/search', request)

    async def track(self, track_id):
        return await self._request('GET', '/v1/music/tracks/' + quote(track_id, safe=''))


PROVIDER = MotusMusicProvider
