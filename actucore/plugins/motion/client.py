"""Small, bounded HTTP boundary to the motion service. No robotics imports."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


class ServiceError(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MotionClient:
    MAX_RESPONSE = 4 * 1024 * 1024

    def __init__(self, endpoint: str = "", api_key: str = "", timeout: float = 8):
        if not isinstance(endpoint, str) or not isinstance(api_key, str):
            raise ServiceError("服务器配置必须是字符串")
        endpoint = endpoint.strip().rstrip("/")
        if endpoint:
            parts = urllib.parse.urlsplit(endpoint)
            if (parts.scheme not in ("http", "https") or not parts.hostname
                    or parts.username is not None or parts.password is not None
                    or parts.query or parts.fragment or any(c.isspace() for c in endpoint)):
                raise ServiceError("服务器地址须为 HTTP(S) 地址，不能包含凭证、查询参数或片段")
            try:
                parts.port
            except ValueError as exc:
                raise ServiceError("服务器端口无效") from exc
        if any(ord(c) < 32 or ord(c) > 126 for c in api_key):
            raise ServiceError("访问凭证格式无效")
        self.endpoint, self._api_key = endpoint, api_key
        self.timeout = max(0.5, min(float(timeout), 10.0))
        self._opener = urllib.request.build_opener(_NoRedirect())

    @property
    def configured(self):
        return bool(self.endpoint and self._api_key)

    @property
    def has_key(self):
        return bool(self._api_key)

    def _safe(self, message):
        value = str(message)
        return value.replace(self._api_key, "***") if self._api_key else value

    def call(self, action: str, **args):
        return self._request("/v1/call", {"action": action, **args})

    def health(self):
        return self._request("/health")

    def _request(self, path, payload=None):
        if not self.configured:
            raise ServiceError("请配置服务器地址和访问凭证")
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(
            self.endpoint + path, data=body,
            headers={"Authorization": "Bearer " + self._api_key,
                     "Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(self.MAX_RESPONSE + 1)
                if len(raw) > self.MAX_RESPONSE:
                    raise ServiceError("服务器响应过大")
                result = json.loads(raw)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise ServiceError("服务器返回重定向，请配置最终服务地址") from exc
            message = ""
            try:
                detail = json.loads(exc.read(8192))
                message = detail.get("message") or detail.get("error") or detail.get("detail") or ""
            except (ValueError, UnicodeError, AttributeError):
                pass
            raise ServiceError(self._safe(message or f"服务器请求失败（HTTP {exc.code}）")) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServiceError("无法连接服务器或请求超时") from exc
        except (ValueError, UnicodeError) as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError("服务器返回了无效 JSON") from exc
        if not isinstance(result, dict):
            raise ServiceError("服务器响应格式无效")
        if result.get("error") or result.get("state") == "error":
            raise ServiceError(self._safe(result.get("message") or result.get("error") or "服务操作失败"))
        return result
