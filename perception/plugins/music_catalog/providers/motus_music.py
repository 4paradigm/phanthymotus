"""Bounded synchronous HTTPS client; called on the MCP request thread, not ROS."""
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from plugins.music_catalog.contracts import MusicError, envelope, https_url


def retry_seconds(value):
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 1.0
    except (ValueError, TypeError):
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 1.0


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise MusicError('cancelled', '音乐请求已取消。')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Even an HTTPS redirect must not forward the card credential.
        if fp is not None:
            fp.close()
        raise MusicError('http_error', '曲库重定向被拒绝，请联系卡片维护者检查内置地址。')


class MotusMusicProvider:
    def __init__(self, config):
        self.endpoint = https_url(config.get('endpoint', '')).rstrip('/')
        if '?' in self.endpoint:
            raise MusicError('invalid_config', '曲库 endpoint 不能包含 query。')
        self._key = config.get('api_key', '')
        if (not isinstance(self._key, str) or not self._key
                or len(self._key) > 4096 or any(not 33 <= ord(c) <= 126 for c in self._key)):
            raise MusicError('invalid_config', '曲库凭据无效，请联系卡片维护者。')
        timeout = config.get('timeout_ms', 3000)
        if type(timeout) is not int or not 100 <= timeout <= 30000:
            raise MusicError('invalid_config', '曲库超时须为 100–30000 毫秒。')
        self.timeout = timeout / 1000
        self._retry_at = 0.0

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self._key, '[redacted]')
        if isinstance(value, list):
            return [self._redact(x) for x in value]
        if isinstance(value, dict):
            return {self._redact(k): self._redact(v) for k, v in value.items()}
        return value

    def _request(self, method, path, payload=None, *, deadline=None, cancel=None):
        deadline = min(deadline or float('inf'), time.monotonic() + self.timeout)
        check_cancel(cancel)
        if time.monotonic() < self._retry_at:
            raise MusicError('rate_limited', '曲库限流，请稍后重试。',
                             retry_after=self._retry_at - time.monotonic())
        request = urllib.request.Request(self.endpoint + path, method=method,
            data=json.dumps(payload).encode('utf-8') if payload is not None else None,
            headers={'Authorization': 'Bearer ' + self._key, 'Accept': 'application/json',
                     'Content-Type': 'application/json; charset=utf-8'})
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            for attempt in range(2):
                check_cancel(cancel)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    response = opener.open(request, timeout=remaining)
                except urllib.error.HTTPError as exc:
                    response = exc  # Read the bounded error envelope and close it too.
                with response:
                    raw = bytearray()
                    while True:
                        check_cancel(cancel)
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError
                        # urllib's open timeout also applies to later reads;
                        # shrink it so a slow response cannot reset the budget.
                        sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
                        if sock is not None:
                            sock.settimeout(remaining)
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        raw.extend(chunk)
                        if len(raw) > 1024 * 1024:
                            raise MusicError('invalid_response', '曲库响应超过 1 MiB。')
                    try:
                        data = self._redact(json.loads(raw))
                    except (ValueError, UnicodeError):
                        data = {}
                    status = response.code
                    if 200 <= status < 300:
                        return envelope(data)
                    retry = status == 429 or 500 <= status < 600
                    delay = retry_seconds(response.headers.get('Retry-After')) if status == 429 else 0.2
                    if status == 429:
                        self._retry_at = time.monotonic() + delay
                    if retry and attempt == 0 and delay < deadline - time.monotonic():
                        if cancel is not None:
                            cancel.wait(delay)
                        else:
                            time.sleep(delay)
                        continue
                    code = {400: 'schema_mismatch', 401: 'unauthorized', 403: 'forbidden',
                            404: 'not_found', 422: 'invalid_filter', 429: 'rate_limited'}.get(
                                status, 'internal' if status >= 500 else 'http_error')
                    err = data.get('error') if isinstance(data, dict) else None
                    allowed = err.get('allowed') if isinstance(err, dict) else None
                    allowed = ([x[:80] for x in allowed[:100] if isinstance(x, str)]
                               if isinstance(allowed, list) else None)
                    request_id = str(data.get('request_id', ''))[:200] if isinstance(data, dict) else ''
                    raise MusicError(code, f'曲库请求失败（HTTP {status}），请勿编造歌曲。',
                                     request_id=request_id, allowed=allowed,
                                     retry_after=delay if status == 429 else None)
        except (OSError, http.client.HTTPException):
            check_cancel(cancel)
            raise MusicError('unavailable', '曲库暂时不可用，请稍后重试。') from None

    def capabilities(self, **context):
        return self._request('GET', '/v1/capabilities', **context)

    def search(self, request, **context):
        return self._request('POST', '/v1/music/search', request, **context)

    def track(self, track_id, **context):
        return self._request('GET', '/v1/music/tracks/' + quote(track_id, safe=''), **context)


PROVIDER = MotusMusicProvider
