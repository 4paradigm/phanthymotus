"""Small VLM client used by the vln card."""

from __future__ import annotations

import base64
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .manifest import (
    DEFAULT_VLM_BASE_URL,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TIMEOUT_SEC,
)


BASE_URL = DEFAULT_VLM_BASE_URL
MODEL = DEFAULT_VLM_MODEL
TIMEOUT_SEC = DEFAULT_VLM_TIMEOUT_SEC


def validate_configuration(
    base_url,
    api_key,
    model,
    timeout_sec,
) -> tuple[str, str, str, float]:
    """Validate and normalize one complete gear-config payload.

    The API key is deliberately returned only to the in-process caller. Error
    messages never include any supplied value, so they are safe for MCP replies
    and logs.
    """

    if not isinstance(base_url, str):
        raise ValueError("VLM API URL must be a string")
    normalized_url = base_url.strip().rstrip("/")
    if not normalized_url or len(normalized_url) > 2048:
        raise ValueError(
            "VLM API URL is required and must be at most 2048 characters"
        )
    if _has_control_character(normalized_url):
        raise ValueError("VLM API URL contains an invalid control character")
    try:
        parsed_url = urlsplit(normalized_url)
        hostname = parsed_url.hostname
    except ValueError as exc:
        raise ValueError("VLM API URL is invalid") from exc
    if parsed_url.scheme not in {"http", "https"} or not hostname:
        raise ValueError("VLM API URL must be an absolute HTTP or HTTPS URL")
    if parsed_url.username is not None or parsed_url.password is not None:
        raise ValueError("VLM API URL must not contain embedded credentials")
    if parsed_url.query or parsed_url.fragment:
        raise ValueError("VLM API URL must not contain a query string or fragment")

    if not isinstance(api_key, str):
        raise ValueError("VLM API Key must be a string")
    normalized_key = api_key.strip()
    if not normalized_key or normalized_key == "****":
        raise ValueError("VLM API Key is required")
    if len(normalized_key) > 4096 or _has_control_character(normalized_key):
        raise ValueError("VLM API Key is invalid")

    if not isinstance(model, str):
        raise ValueError("VLM model name must be a string")
    normalized_model = model.strip()
    if not normalized_model or len(normalized_model) > 256:
        raise ValueError(
            "VLM model name is required and must be at most 256 characters"
        )
    if _has_control_character(normalized_model):
        raise ValueError("VLM model name contains an invalid control character")

    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
        raise ValueError("VLM timeout must be a number")
    normalized_timeout = float(timeout_sec)
    if (
        not math.isfinite(normalized_timeout)
        or not 1.0 <= normalized_timeout <= 120.0
    ):
        raise ValueError("VLM timeout must be between 1 and 120 seconds")

    return normalized_url, normalized_key, normalized_model, normalized_timeout


class Client:
    def __init__(
        self,
        base_url: str = BASE_URL,
        api_key: str = "",
        model: str = MODEL,
        timeout: float = TIMEOUT_SEC,
    ):
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._url = (
            f"{self._base_url}/chat/completions" if self._base_url else ""
        )
        self._api_key = str(api_key or "").strip()
        self._model = str(model or "").strip()
        try:
            parsed_timeout = float(timeout)
        except (TypeError, ValueError):
            parsed_timeout = TIMEOUT_SEC
        self._timeout = (
            parsed_timeout
            if math.isfinite(parsed_timeout) and 1.0 <= parsed_timeout <= 120.0
            else TIMEOUT_SEC
        )

    @property
    def configured(self) -> bool:
        try:
            validate_configuration(
                self._base_url,
                self._api_key,
                self._model,
                self._timeout,
            )
        except (TypeError, ValueError):
            return False
        return True

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout_sec(self) -> float:
        return self._timeout

    @property
    def api_key_configured(self) -> bool:
        return bool(self._api_key)

    def complete_json(self, messages: list[dict]) -> dict:
        if not self.configured:
            raise RuntimeError("VLM is not configured")
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 1024,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        request = Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read())
        except HTTPError as exc:
            # Provider error bodies are intentionally not forwarded: some
            # gateways echo request metadata that may contain credentials.
            raise RuntimeError(f"VLM request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError("VLM endpoint is unavailable") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"VLM request timed out after {self._timeout:.1f}s"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("VLM returned invalid JSON over HTTP") from exc

        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError("VLM service returned an error response")
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("VLM returned an invalid response envelope") from exc
        return _parse_json(content)

    @staticmethod
    def image_url(image: bytes, mime_type: str = "image/jpeg") -> str:
        if not image:
            raise ValueError("image is empty")
        encoded = base64.b64encode(image).decode()
        return f"data:{mime_type};base64,{encoded}"


def _has_control_character(value: str) -> bool:
    return any(
        ord(character) < 32 or ord(character) == 127 for character in value
    )


def _parse_json(content) -> dict:
    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        raise RuntimeError("VLM response content is not text")

    decoder = json.JSONDecoder()
    objects: list[dict] = []
    index = 0
    while index < len(content):
        start = content.find("{", index)
        if start < 0:
            break
        try:
            result, consumed = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(result, dict):
            objects.append(result)
        index = start + max(consumed, 1)
    if len(objects) == 1:
        return objects[0]
    if not objects:
        raise RuntimeError("VLM response does not contain a valid JSON object")
    raise RuntimeError("VLM response contains multiple JSON objects")
