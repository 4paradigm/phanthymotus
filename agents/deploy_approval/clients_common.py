"""Shared HTTP helpers for the Deploy Approval Agent's clients.

Enforces a single policied transport: timeouts, redirect blocking, response
size limits and self-signed CAs come from one place so no client can silently
weaken it.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from urllib.parse import urlparse

import httpx

from .config import Config

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised for an out-of-policy target, image reference or transport."""


def build_client(
    config: Config, *, verify_tls: bool = True, ca_file: str = ""
) -> httpx.AsyncClient:
    """Construct the shared AsyncClient used by every outbound call.

    - ``follow_redirects=False`` everywhere (anti-SSRF / anti-auth-bypass).
    - explicit timeouts (connect/read/write/pool).
    - TLS via a CA file; plaintext HTTP only when ``allow_private_http`` is
      on and the host is in the private CIDR allowlist.
    """
    """Create the shared AsyncClient used by every outbound call.

    - ``follow_redirects=False`` everywhere (anti-SSRF / anti-auth-bypass).
    - explicit timeouts (connect/read/write/pool).
    - self-signed CAs via a CA file; plaintext HTTP only when
      ``allow_private_http`` is on and the host is in the private CIDR list.
    """
    timeout = httpx.Timeout(
        config.total_timeout,  # overall cap per request
        connect=config.connect_timeout,
        read=config.read_timeout,
        write=config.connect_timeout,
        pool=config.connect_timeout,
    )
    kwargs: dict = {
        "timeout": timeout,
        "follow_redirects": False,
        "limits": httpx.Limits(max_connections=50),
    }
    if not verify_tls:
        raise SecurityError("TLS verification must not be disabled")
    if ca_file:
        kwargs["verify"] = ca_file
    else:
        # Never disable TLS verification. Plaintext HTTP is gated separately by
        # require_http_policy; HTTPS always uses the system CA store.
        kwargs["verify"] = True
    return httpx.AsyncClient(**kwargs)


def is_allowed_private_host(host: str, allowed_cidrs: list[str]) -> bool:
    """True only when ``host`` is a literal IP in an allowed private CIDR."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    try:
        if not (ip.is_private or ip.is_loopback):
            return False
    except Exception:
        return False
    return any(
        ip in ipaddress.ip_network(c, strict=False) for c in allowed_cidrs
    )




AGENT_CORE_NODE_PORT = 15678


def require_http_policy(
    url: str, config: Config, *, allow_private: bool,
    agent_core_node: str = "",
) -> None:
    """Fail-closed transport policy for an outbound URL.

    - Only http/https; everything else refused.
    - HTTPS is always allowed.
    - HTTP to loopback (127.0.0.1 / ::1) is always allowed so a collocated
      Review Agent or local registry is reachable even when ALLOW_PRIVATE_HTTP
      is off — but only the literal loopback hosts, never hostnames.
    - Other private HTTP requires ``allow_private`` and a literal IP inside
      an allowed private CIDR (no DNS-based private guessing).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SecurityError(f"unsupported scheme {parsed.scheme!r}")
    if parsed.scheme == "https":
        return
    host = parsed.hostname or ""
    if host in ("127.0.0.1", "::1"):
        return
    if not allow_private:
        raise SecurityError("HTTP not permitted (HTTPS required by policy)")
    if not is_allowed_ip_host(host, config.http_allowed_cidrs):
        raise SecurityError("HTTP target not in the private CIDR allowlist")


def is_allowed_ip_host(host: str, allowed_cidrs: list[str]) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # hostnames resolved elsewhere; here we require a literal IP
    try:
        if not (ip.is_private or ip.is_loopback):
            return False
    except Exception:
        return False
    if not allowed_cidrs:
        return False
    return any(
        ip in ipaddress.ip_network(c, strict=False) for c in allowed_cidrs
    )


def enforce_max_size(content_length: str | None, limit: int) -> None:
    if content_length is None:
        return
    try:
        if int(content_length) > limit:
            raise SecurityError(
                f"response content-length {content_length} exceeds limit {limit}"
            )
    except ValueError:
        return


async def enforce_body_size(resp: httpx.Response, limit: int) -> httpx.Response:
    """Reads the response body and refuses it when it exceeds ``limit`` bytes.

    Uses ``aiter_bytes`` so a response that is in the middle of streaming is
    capped as it is read, instead of being buffered fully and only checked after
    the fact. ``aiter_bytes`` reads the (already-buffered) body for mock/closed
    responses and streams for live responses.
    """
    total = 0
    async for chunk in resp.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise SecurityError(
                f"response body exceeds limit {limit} (>{total} bytes)"
            )
    return resp


def sanitize_url(url: str) -> str:
    """Redact any userinfo before logging/audit."""
    parsed = urlparse(url)
    if parsed.netloc and "@" in parsed.netloc:
        safe = parsed.netloc.rsplit("@", 1)[-1]
        return url.replace(parsed.netloc, safe, 1)
    return url


def sanitize_headers(headers: dict) -> dict:
    """Redact common credential headers before persistence/audit."""
    sensitive = {
        "authorization",
        "x-registry-token",
        "cookie",
        "proxy-authorization",
    }
    return {
        k: ("***" if k.lower() in sensitive else v)
        for k, v in headers.items()
    }


async def stream_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    limit: int,
    *,
    headers=None,
    params=None,
    json=None,
    content=None,
    timeout: float = 0,
) -> httpx.Response:
    """Issue an outbound request and read the body with a hard byte cap.

    Uses ``client.stream`` so the response is read from the wire in chunks and
    refused as soon as it exceeds ``limit`` — it is never fully buffered before
    the size check (unlike ``client.get`` which eagerly buffers the whole body).
    The body is reconstructed into a fresh ``httpx.Response`` so callers can keep
    using ``.json()``, ``.headers`` and ``.aread()`` unchanged.

    ``timeout`` (seconds, 0 = disabled) is an absolute wall-clock total cap that
    spans connect + send + response headers + the *entire* streaming read,
    including a peer that drips bytes slowly (the per-read httpx timeout would
    otherwise keep resetting). It raises :class:`SecurityError`, which the clients
    map to their domain error. Redirects stay disabled at the shared client.
    """

    async def _read() -> httpx.Response:
        # The actual reading happens inside this coroutine, using the correct
        # `async with client.stream(...) as response` context manager. Reading the
        # body with aiter_bytes enforces the byte cap from the wire, then we
        # reassemble a plain httpx.Response for the callers.
        async with client.stream(
            method,
            url,
            headers=headers,
            params=params,
            json=json,
            content=content,
        ) as resp:
            total = 0
            chunks = []
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > limit:
                    raise SecurityError(
                        f"response body exceeds limit {limit} (>{total} bytes)"
                    )
                chunks.append(chunk)
            body = b"".join(chunks)
            # Rebuild headers without encoding/length that refer to the
            # *compressed* wire body.  aiter_bytes() already content-decoded,
            # so carrying Content-Encoding / Content-Length from the original
            # response causes httpx to attempt a second decode (DecodingError).
            new_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("content-encoding", "content-length")
            }
            return httpx.Response(
                status_code=resp.status_code,
                headers=new_headers,
                content=body,
                request=resp.request,
            )

    if timeout and timeout > 0:
        try:
            return await asyncio.wait_for(_read(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise SecurityError(
                f"request total timeout exceeded for {url}"
            ) from e
    return await _read()


def require_2xx(status_code: int, ctx: str) -> None:
    """Fail-closed transport check: only a 2xx status is acceptable.

    3xx is treated as an error too (redirects are already blocked at the client;
    a server answering 3xx without following is a protocol violation we must not
    interpret as success).
    """
    if not (200 <= status_code < 300):
        raise SecurityError(f"{ctx}: unexpected status {status_code}")
