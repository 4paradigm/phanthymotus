"""Tencent QQ official C2C text adapter for PhanthyMotus."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import os
from pathlib import Path
import re
import threading
import time

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage

_MAX_INBOUND = 2000
_MAX_OUTBOUND = 1800
_SEEN_TTL = 600.0
_SEEN_LIMIT = 1024
_APP_ID_RE = re.compile(r"^[0-9]{5,32}$")
_OPENID_RE = re.compile(r"^[A-Za-z0-9_-]{8,256}$")
_ENV_RE = re.compile(r"^QQ_OFFICIAL_[A-Z0-9_]{1,64}$")


def validate_qq_official_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict) or cfg.get("protocol") != "official":
        raise ValueError("QQ official protocol required")
    if not _APP_ID_RE.fullmatch(str(cfg.get("app_id", ""))):
        raise ValueError("QQ official app_id must be numeric")
    if cfg.get("private_only", True) is not True:
        raise ValueError("QQ official V1 is C2C only")
    sources = ("app_secret", "app_secret_env", "app_secret_file")
    if sum(bool(cfg.get(k)) for k in sources) != 1:
        raise ValueError("QQ official requires exactly one credential source")
    if cfg.get("app_secret_env") and not _ENV_RE.fullmatch(cfg["app_secret_env"]):
        raise ValueError("QQ official environment key invalid")
    if cfg.get("app_secret_file") and cfg["app_secret_file"] != "/run/qq-official-app-secret":
        raise ValueError("QQ official credential mount invalid")
    timeout = cfg.get("connect_timeout", 20)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 5 <= timeout <= 60:
        raise ValueError("QQ official connect_timeout out of range")
    return cfg


def _load_credential(cfg: dict) -> str:
    if cfg.get("app_secret_env"):
        value = os.environ.get(cfg["app_secret_env"], "")
    elif cfg.get("app_secret_file"):
        try:
            value = Path("/run/qq-official-app-secret").read_text(encoding="utf-8").strip()
        except OSError:
            raise ValueError("QQ official credential mount unreadable") from None
    else:
        value = cfg.get("app_secret", "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("QQ official credential not configured")
    return value.strip()


class QQOfficialAdapter(ChannelAdapter):
    """Official Gateway adapter; no group, media or proactive tools."""

    def __init__(self, channel_id: str, platform: str, config: dict, on_message):
        super().__init__(channel_id, platform, config, on_message)
        self._api = None
        self._ws = None
        self._loop = None
        self._connected = None
        self._last_error = ""
        self._session_lock = threading.Lock()
        self._session_id = None
        self._seq = None
        self._seen = OrderedDict()
        self._http = None
        self._attempted_replies = OrderedDict()
        self._reply_lock = asyncio.Lock()

    async def start(self) -> None:
        validate_qq_official_config(self.config)
        if self._running:
            return
        try:
            from qqbot_agent_sdk import QQApiClient, QQWebSocket, WSCallbacks, EventParser
        except ImportError:
            raise RuntimeError("QQ official SDK is not installed") from None
        import httpx
        self._loop = asyncio.get_running_loop()
        self._connected = asyncio.Event()
        self._parser = EventParser()
        self._last_error = ""
        credential = _load_credential(self.config)
        self._http = httpx.AsyncClient(timeout=15)
        self._api = QQApiClient(str(self.config["app_id"]), credential)
        self._api.setup(self._http)

        def signal(connected: bool):
            if self._loop is None or self._connected is None or self._loop.is_closed():
                return
            self._loop.call_soon_threadsafe(
                self._connected.set if connected else self._connected.clear)

        def get_session():
            with self._session_lock:
                return self._session_id, self._seq

        def set_session(session_id, seq):
            with self._session_lock:
                self._session_id, self._seq = session_id, seq

        async def on_event(event_type: str, raw: dict):
            if self._connected is None or not self._connected.is_set():
                return
            try:
                event = self._parser.parse(event_type, raw)
                msg = self._parse_message(event)
                if msg is not None:
                    await self._on_message(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "QQ official inbound failed (details suppressed)")

        def fatal(_code, _message):
            self._last_error = "QQ official Gateway rejected session"
            signal(False)

        callbacks = WSCallbacks(
            on_event,                 # inbound
            lambda: signal(True),     # ready
            lambda: signal(False),    # disconnected
            fatal,                    # fatal
            self._api.ensure_token_sync,
            get_session,
            set_session,
            lambda _value: None,      # heartbeat interval
            self._api.clear_token,
            lambda _reason: None,     # pending futures
            self._api.get_gateway_url_sync,
        )
        self._ws = QQWebSocket(callbacks=callbacks, log_tag="phanthymotus-qq")

        try:
            await self._api.ensure_token()
            url = await self._api.get_gateway_url()
            self._ws.start(url, self._loop)
            await asyncio.wait_for(
                self._connected.wait(),
                timeout=float(self.config.get("connect_timeout", 20)),
            )
            self._running = True
        except Exception:
            self._last_error = "QQ official authentication/Gateway READY failed"
            await self.stop()
            raise RuntimeError(self._last_error) from None

    async def stop(self) -> None:
        self._running = False
        if self._connected is not None:
            self._connected.clear()
        if self._ws is not None:
            try:
                await self._ws.async_stop()
            finally:
                self._ws = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._api = None
        self._loop = None

    def _parse_message(self, event) -> InboundMessage | None:
        if (event is None or event.chat_scope != "c2c"
                or event.event_type != "C2C_MESSAGE_CREATE"):
            return None
        if getattr(event, "attachments", None):
            return None
        # Tencent may include text-only msg_elements in an ordinary C2C message.
        # Reject embedded media, but do not discard valid plain-text elements.
        for element in (getattr(event, "msg_elements", None) or []):
            if (getattr(element, "attachments", None)
                    or not isinstance(getattr(element, "content", None), str)
                    or not element.content.strip()):
                return None
        user, chat, mid = event.user_id, event.chat_id, event.message_id
        content = event.content
        if (not isinstance(user, str) or not _OPENID_RE.fullmatch(user)
                or not isinstance(chat, str) or chat != user
                or not isinstance(mid, str) or not 0 < len(mid) <= 128
                or not isinstance(content, str)
                or not 0 < len(content.strip()) <= _MAX_INBOUND):
            return None
        now = time.monotonic()
        while self._seen:
            key, ts = next(iter(self._seen.items()))
            if now - ts <= _SEEN_TTL:
                break
            self._seen.popitem(last=False)
        dedup_key = f"{self.config['app_id']}:{mid}"
        if dedup_key in self._seen:
            return None
        self._seen[dedup_key] = now
        if len(self._seen) > _SEEN_LIMIT:
            self._seen.popitem(last=False)
        # Isolate official OpenIDs from numeric OneBot QQ IDs at ACL level.
        principal = f"official:{self.config['app_id']}:{user}"
        return InboundMessage(
            platform="qq", channel_id=self.channel_id, user_id=principal,
            chat_id=chat, display_name=(event.user_name or "QQ user")[:100],
            text=content.strip(), message_id=mid,
            chat_type="p2p", sender_type="user",
        )

    async def send_message(self, msg: OutboundMessage) -> None:
        if (not self._running or self._api is None or self._connected is None
                or not self._connected.is_set()):
            raise ConnectionError("QQ official Gateway is disconnected")
        if msg.files or msg.image_bytes:
            raise ValueError("QQ official V1 supports text only")
        if not isinstance(msg.text, str) or not 0 < len(msg.text) <= _MAX_OUTBOUND:
            raise ValueError("QQ official text length invalid")
        if not isinstance(msg.chat_id, str) or not _OPENID_RE.fullmatch(msg.chat_id):
            raise ValueError("QQ official target OpenID invalid")
        mid = msg.reply_to_message_id
        if not isinstance(mid, str) or not 0 < len(mid) <= 128:
            raise ValueError("QQ official requires a verified source message ID")
        # One passive reply per trigger. Mark an attempt *before* HTTP so that
        # concurrent requests or an unconfirmed timeout cannot deliver twice.
        key = (msg.chat_id, mid)
        async with self._reply_lock:
            now = time.monotonic()
            for seen, ts in list(self._attempted_replies.items()):
                if now - ts > _SEEN_TTL:
                    self._attempted_replies.pop(seen, None)
            if key in self._attempted_replies:
                raise ValueError("QQ official reply already attempted for this source")
            self._attempted_replies[key] = now
            if len(self._attempted_replies) > _SEEN_LIMIT:
                self._attempted_replies.popitem(last=False)
        try:
            # A timeout is NOT proof that sending failed. No blind retries.
            await self._api.send_text(
                "c2c", msg.chat_id, msg.text,
                reply_to=mid, markdown=False, retries=1,
            )
        except Exception:
            raise RuntimeError("QQ official reply failed or delivery unconfirmed") from None

    async def health_check(self) -> tuple[bool, str]:
        ok = bool(self._running and self._connected and self._connected.is_set())
        return ok, ("" if ok else (self._last_error or "QQ official Gateway disconnected"))

    def status(self) -> str:
        if self._running and self._connected and self._connected.is_set():
            return "connected"
        return "degraded" if self._running else "disconnected"
