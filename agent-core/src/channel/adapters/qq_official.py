"""Tencent QQ official Gateway Channel adapter for PhanthyMotus."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import os
from pathlib import Path
import re
import socket
import threading
import time

import httpx

from channel.qq_runtime_credentials import get as get_runtime_credential
from channel.adapter import (
    Attachment, ChannelAdapter, InboundMessage, OutboundMessage,
    KIND_IMAGE, KIND_VIDEO, KIND_AUDIO, KIND_FILE, PartialSendError,
)
from channel.adapters.qq_official_media import (
    receive_media, send_media, _validate_outbound_file,
)
from channel.adapters.qq_official_interactions import QQInteractionRegistry
from channel.adapters.qq_official_sessions import QQSessionState
from channel.adapters.qq_official_voice import read_voice_text

_MAX_INBOUND = 2000
_MAX_OUTBOUND = 1800
_SEEN_TTL = 3600.0  # C2C passive context can remain valid for 60 minutes
_REPLY_TTL = {"c2c": 3600.0, "group": 300.0, "guild": 300.0, "guild_dm": 300.0}
_SEEN_LIMIT = 1024
# Bounded retry only for proven pre-connection DNS failures.
_DNS_RETRY_DELAYS = (0.5, 1.5)
_APP_ID_RE = re.compile(r"^[0-9]{5,32}$")
_OPENID_RE = re.compile(r"^[A-Za-z0-9_-]{8,256}$")
_ENV_RE = re.compile(r"^QQ_OFFICIAL_[A-Z0-9_]{1,64}$")
_ALLOWED_SCOPES = frozenset(("c2c", "group", "guild", "guild_dm"))
_EVENT_SCOPE = {
    "C2C_MESSAGE_CREATE": "c2c",
    "GROUP_AT_MESSAGE_CREATE": "group",
    "GUILD_AT_MESSAGE_CREATE": "guild",
    "GUILD_MESSAGE_CREATE": "guild",
    "DIRECT_MESSAGE_CREATE": "guild_dm",
}


def _full_group_mentions_bot(raw: dict, bot_group_openid: str = "") -> bool:
    """Only deliver full-mode group traffic when it explicitly @s this bot.

    GROUP_MESSAGE_CREATE includes non-@ traffic. Unknown mention identity
    must fail closed so the LLM cannot reply to an entire busy QQ group.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("mentions"), list):
        return False
    for mention in raw["mentions"]:
        if not isinstance(mention, dict):
            continue
        if mention.get("is_you") is True:
            return True
        openid = mention.get("openid") or mention.get("id")
        if bot_group_openid and openid == bot_group_openid:
            return True
    return False


def validate_qq_official_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict) or cfg.get("protocol") != "official":
        raise ValueError("QQ official protocol required")
    if not _APP_ID_RE.fullmatch(str(cfg.get("app_id", ""))):
        raise ValueError("QQ official app_id must be numeric")
    scopes = cfg.get("enabled_scopes", ["c2c"])
    if (not isinstance(scopes, list) or not scopes
            or any(not isinstance(s, str) for s in scopes)
            or len(scopes) != len(set(scopes))
            or not set(scopes) <= _ALLOWED_SCOPES):
        raise ValueError("QQ official enabled_scopes must be a nonempty unique subset")
    if cfg.get("private_only", True) not in (True, False):
        raise ValueError("QQ official private_only must be boolean")
    if cfg.get("private_only", True) and set(scopes) != {"c2c"}:
        raise ValueError("QQ official expanded scopes require private_only=false")
    if cfg.get("private_only", True) is False and set(scopes) == {"c2c"}:
        raise ValueError("QQ official nonprivate mode requires expanded scopes")
    for flag in ("enable_inbound_media", "enable_outbound_media",
                 "enable_markdown", "enable_typing",
                 "enable_session_resume", "enable_interactions",
                 "enable_approvals", "enable_guild_plain_messages",
                 "enable_voice_transcription"):
        if flag in cfg and not isinstance(cfg[flag], bool):
            raise ValueError(f"QQ official {flag} must be boolean")
    if cfg.get("enable_approvals") and not cfg.get("enable_interactions"):
        raise ValueError("QQ approvals require interactive events")
    if cfg.get("enable_guild_plain_messages") and "guild" not in scopes:
        raise ValueError("QQ plain Guild events require the guild scope")
    if cfg.get("enable_voice_transcription") and not cfg.get("enable_inbound_media"):
        raise ValueError("QQ voice transcription requires inbound media")
    bot_group_openid = cfg.get("bot_group_openid", "")
    if (not isinstance(bot_group_openid, str)
            or (bot_group_openid and not _OPENID_RE.fullmatch(bot_group_openid))):
        raise ValueError("QQ bot_group_openid must be a valid OpenID")
    sources = ("app_secret", "app_secret_env", "app_secret_file", "app_secret_runtime")
    for key in sources:
        if key in cfg and not isinstance(cfg[key], str):
            raise ValueError(f"QQ official {key} must be a string")
    if sum(bool(cfg.get(k)) for k in sources) != 1:
        raise ValueError("QQ official requires exactly one credential source")
    if cfg.get("app_secret_env") and not _ENV_RE.fullmatch(cfg["app_secret_env"]):
        raise ValueError("QQ official environment key invalid")
    if cfg.get("app_secret_file") and cfg["app_secret_file"] != "/run/qq-official-app-secret":
        from channel.qq_credential_store import is_managed_reference
        if not is_managed_reference(str(cfg.get("app_id", "")), cfg["app_secret_file"]):
            raise ValueError("QQ official credential reference invalid")
    if cfg.get("app_secret_runtime") and not re.fullmatch(r"[A-Za-z0-9_-]{30,64}", cfg["app_secret_runtime"]):
        raise ValueError("QQ official runtime credential reference invalid")
    timeout = cfg.get("connect_timeout", 20)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 5 <= timeout <= 60:
        raise ValueError("QQ official connect_timeout out of range")
    return cfg


def _load_credential(cfg: dict) -> str:
    if cfg.get("app_secret_env"):
        value = os.environ.get(cfg["app_secret_env"], "")
    elif cfg.get("app_secret_file"):
        try:
            from channel.qq_credential_store import is_managed_reference, read
            path = cfg["app_secret_file"]
            value = (read(str(cfg["app_id"]), path)
                     if is_managed_reference(str(cfg["app_id"]), path)
                     else Path(path).read_text(encoding="utf-8").strip())
        except OSError:
            raise ValueError("QQ official credential mount unreadable") from None
    else:
        value = (get_runtime_credential(cfg["app_secret_runtime"]) if cfg.get("app_secret_runtime")
                 else cfg.get("app_secret", ""))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("QQ official credential not configured")
    return value.strip()


def _is_temporary_dns_failure(error: BaseException) -> bool:
    """Detect EAI_AGAIN through SDK wrapping, never a generic timeout."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, socket.gaierror) and current.errno == socket.EAI_AGAIN:
            return True
        if (isinstance(current, httpx.ConnectError)
                and "[Errno -3] Temporary failure in name resolution" in str(current)):
            return True
        current = current.__cause__ or current.__context__
    return False


class QQOfficialAdapter(ChannelAdapter):
    """Official Gateway adapter with scope-gated messages and rich media."""

    SUPPORTED_FILE_KINDS = (KIND_IMAGE, KIND_VIDEO, KIND_AUDIO, KIND_FILE)

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
        self._source_scopes = OrderedDict()
        self._source_actors = OrderedDict()
        self._prompted_sources = OrderedDict()
        self._interactions = QQInteractionRegistry()
        self._session_store = None
        self._http = None
        self._attempted_replies = OrderedDict()
        self._reply_lock = asyncio.Lock()
        self._group_event_count = 0

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
        if self.config.get("enable_session_resume", False):
            self._session_store = QQSessionState(str(self.config["app_id"]))
            self._session_id, self._seq = self._session_store.read()
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
                if self._session_store is not None:
                    self._session_store.save(session_id, seq)

        async def on_event(event_type: str, raw: dict):
            if self._connected is None or not self._connected.is_set():
                return
            try:
                is_group_event = event_type in (
                    "GROUP_AT_MESSAGE_CREATE", "GROUP_MESSAGE_CREATE")
                if is_group_event:
                    self._group_event_count += 1
                log_group = (is_group_event and
                             (self._group_event_count == 1 or
                              self._group_event_count % 100 == 0))
                if event_type == "GROUP_MESSAGE_CREATE":
                    if "group" not in self.config.get("enabled_scopes", ["c2c"]):
                        return
                    if not _full_group_mentions_bot(
                            raw, self.config.get("bot_group_openid", "")):
                        if log_group:
                            print("[qq-group] full_event_ignored_not_at_bot",
                                  flush=True)
                        return
                    # Same payload schema as GROUP_AT; normalize for SDK 1.2.2.
                    parsed_event_type = "GROUP_AT_MESSAGE_CREATE"
                else:
                    parsed_event_type = event_type
                if event_type == "GROUP_AT_MESSAGE_CREATE" and log_group:
                    print("[qq-group] gateway_event_received type=at",
                          flush=True)
                event = self._parser.parse(parsed_event_type, raw)
                msg = self._parse_message(event)
                if is_group_event and log_group:
                    print(
                        f"[qq-group] sdk_parsed={event is not None} "
                        f"adapter_accepted={msg is not None}",
                        flush=True,
                    )
                    if event is not None and msg is None:
                        # Log only validation categories, not user data.
                        elements = getattr(event, "msg_elements", None) or []
                        user_id = getattr(event, "user_id", None)
                        chat_id = getattr(event, "chat_id", None)
                        message_id = getattr(event, "message_id", None)
                        body = getattr(event, "content", None)
                        flags = {
                            "scope": getattr(event, "chat_scope", None) == "group",
                            "user_openid": isinstance(user_id, str) and bool(
                                _OPENID_RE.fullmatch(user_id)),
                            "group_openid": isinstance(chat_id, str) and bool(
                                _OPENID_RE.fullmatch(chat_id)),
                            "message_id": isinstance(message_id, str) and
                                0 < len(message_id) <= 512,
                            "content": isinstance(body, str) and
                                0 < len(body.strip()) <= _MAX_INBOUND,
                            "elements_valid": all(
                                isinstance(getattr(e, "content", None), str)
                                for e in elements),
                            "media_allowed": (
                                self.config.get("enable_inbound_media", False)
                                or not (getattr(event, "attachments", None)
                                    or any(getattr(e, "attachments", None)
                                           for e in elements))),
                        }
                        failures = [name for name, valid in flags.items()
                                    if not valid]
                        print(
                            "[qq-group] rejected_checks="
                            + ",".join(failures or ["duplicate_or_other"]),
                            flush=True,
                        )
                if msg is not None:
                    if self.config.get("enable_inbound_media", False):
                        # Reuse the same persistent channel store as Feishu:
                        # origin labeling, per-channel isolation and pruning.
                        from channel import store
                        cache = store.dir_for(self.channel_id).resolve()
                        msg.attachments = await receive_media(
                            self._http, event, str(cache))
                        if msg.attachments:
                            store.maybe_prune()
                        if (not msg.attachments
                                and (getattr(event, "attachments", None) or
                                     any(getattr(e, "attachments", None) for e in
                                         (getattr(event, "msg_elements", None) or [])))):
                            msg.text += " [QQ attachment unavailable]"
                    if self.config.get("enable_voice_transcription"):
                        transcript = await read_voice_text(
                            self._http, event, msg.attachments)
                        if transcript:
                            msg.text = (msg.text + " [QQ语音转写] " + transcript)[:_MAX_INBOUND]
                    if (msg.chat_type == "p2p" and self.config.get("enable_typing")
                            and self._api is not None):
                        try:
                            await self._api.send_typing(msg.chat_id, msg.message_id,
                                                        input_seconds=5)
                        except Exception:
                            pass
                    await self._on_message(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "QQ official inbound failed (details suppressed)")

        def fatal(_code, _message):
            if (str(_code) == "4014"
                    or "disallowed intents" in str(_message).lower()
                    or "intent not authorized" in str(_message).lower()):
                self._last_error = (
                    "QQ Gateway 4014: requested event intents are not authorized; "
                    "disable unsupported message scopes or interactions")
            else:
                self._last_error = "QQ official Gateway rejected session"
            with self._session_lock:
                self._session_id, self._seq = None, None
                if self._session_store is not None:
                    self._session_store.clear()
            signal(False)

        async def on_interaction(event_type: str, raw: dict):
            if (not self.config.get("enable_interactions")
                    or self._api is None or self._connected is None
                    or not self._connected.is_set()):
                return
            try:
                await self._interactions.handle(self._api, event_type, raw)
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "QQ official interaction handler failed (details suppressed)")

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
        callbacks.on_interaction_event = on_interaction
        from channel.adapters.qq_official_gateway import (
            QQScopedWebSocket, qq_gateway_intents,
        )
        self._ws = QQScopedWebSocket(
            callbacks=callbacks,
            intents=qq_gateway_intents(self.config),
            log_tag="phanthymotus-qq",
        )

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
            if not self._last_error:
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
        if event is None:
            return None
        scope = _EVENT_SCOPE.get(event.event_type)
        if (event.event_type == "GUILD_MESSAGE_CREATE"
                and not self.config.get("enable_guild_plain_messages", False)):
            return None
        if scope is None or event.chat_scope != ("dm" if scope == "guild_dm" else scope):
            return None
        if scope not in self.config.get("enabled_scopes", ["c2c"]):
            return None
        has_media = bool(getattr(event, "attachments", None))
        for element in (getattr(event, "msg_elements", None) or []):
            has_media = has_media or bool(getattr(element, "attachments", None))
            if not getattr(element, "attachments", None):
                element_text = getattr(element, "content", None)
                if not isinstance(element_text, str):
                    return None
                # The SDK may emit empty metadata segments for mentions
                # or media in any chat scope. The final content/media check
                # below rejects genuinely empty messages instead.
        media_enabled = self.config.get("enable_inbound_media", False)
        user, chat, mid = event.user_id, event.chat_id, event.message_id
        content = event.content
        if not isinstance(content, str):
            return None
        content = content.strip() or ("[QQ attachment]" if has_media else "")
        if has_media and not media_enabled:
            # Keep the safe text/reply path functional when media is disabled.
            # Never imply that an attachment was downloaded or inspected.
            content = (content + " [QQ attachment not enabled]")[:_MAX_INBOUND]
        if (not isinstance(user, str) or not _OPENID_RE.fullmatch(user)
                or not isinstance(chat, str) or not _OPENID_RE.fullmatch(chat)
                or (scope == "c2c" and chat != user)
                or not isinstance(mid, str) or not 0 < len(mid) <= 512
                or not 0 < len(content) <= _MAX_INBOUND):
            return None
        now = time.monotonic()
        for cache in (self._seen, self._source_scopes):
            while cache:
                _, item = next(iter(cache.items()))
                ts = item if cache is self._seen else item[1]
                if now - ts <= _SEEN_TTL:
                    break
                expired, _ = cache.popitem(last=False)
                if cache is self._source_scopes:
                    self._source_actors.pop(expired, None)
        dedup_key = (scope, chat, mid)
        if dedup_key in self._seen:
            return None
        self._seen[dedup_key] = now
        if len(self._seen) > _SEEN_LIMIT:
            self._seen.popitem(last=False)
        self._source_scopes[(chat, mid)] = (scope, now)
        self._source_actors[(chat, mid)] = user
        if len(self._source_scopes) > _SEEN_LIMIT:
            stale, _ = self._source_scopes.popitem(last=False)
            self._source_actors.pop(stale, None)
        # Keep each QQ group member's ACL identity separate; each chat
        # retains its own persistent conversation context in ChannelManager.
        principal = (f"official:{self.config['app_id']}:{user}" if scope == "c2c"
                     else f"official:{self.config['app_id']}:{scope}:{chat}:{user}")
        return InboundMessage(
            platform="qq", channel_id=self.channel_id, user_id=principal,
            chat_id=chat, display_name=(event.user_name or "QQ user")[:100],
            text=content, message_id=mid,
            chat_type="p2p" if scope in ("c2c", "guild_dm") else "group",
            sender_type="user",
        )

    async def send_message(self, msg: OutboundMessage) -> None:
        if (not self._running or self._api is None or self._connected is None
                or not self._connected.is_set()):
            raise ConnectionError("QQ official Gateway is disconnected")
        if msg.image_bytes:
            raise ValueError("QQ official byte images must be saved as a file first")
        if msg.files and not self.config.get("enable_outbound_media", False):
            raise ValueError("QQ official media sending is not enabled")
        if len(msg.files) > 3:
            raise ValueError("QQ official supports at most 3 media files per trigger")
        for attachment in msg.files:
            _validate_outbound_file(attachment)
        if (not isinstance(msg.text, str) or len(msg.text) > _MAX_OUTBOUND
                or (not msg.text and not msg.files)):
            raise ValueError("QQ official text length invalid")
        if not isinstance(msg.chat_id, str) or not _OPENID_RE.fullmatch(msg.chat_id):
            raise ValueError("QQ official target OpenID invalid")
        mid = msg.reply_to_message_id
        if not isinstance(mid, str) or not 0 < len(mid) <= 512:
            raise ValueError("QQ official requires a verified source message ID")
        # One passive reply per trigger. Mark an attempt *before* HTTP so that
        # concurrent requests or an unconfirmed timeout cannot deliver twice.
        key = (msg.chat_id, mid)
        source = self._source_scopes.get(key)
        if source is None:
            raise ValueError("QQ official unknown or expired reply source")
        scope = source[0]
        if time.monotonic() - source[1] > _REPLY_TTL[scope]:
            raise ValueError("QQ official unknown or expired reply source")
        if msg.files and scope not in ("c2c", "group"):
            raise ValueError("QQ SDK media supports only C2C and group")
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
        sent: list[str] = []
        for attachment in msg.files:
            try:
                await send_media(self._api, self._http, scope, msg.chat_id, mid, attachment)
                sent.append("media")
            except Exception:
                if sent:
                    raise PartialSendError(sent, ["QQ media delivery unconfirmed"]) from None
                raise RuntimeError(
                    "QQ official rich media failed or delivery unconfirmed") from None
        if not msg.text:
            return
        # One passive source can produce up to four actual messages:
        # <= 3 media files followed by text, below QQ's five-reply budget.
        # Generic timeouts must never be retried blindly.
        for attempt in range(len(_DNS_RETRY_DELAYS) + 1):
            try:
                if scope == "guild_dm":
                    # SDK 1.2.2 parses DMs but has no high-level DM send helper.
                    # Its authenticated generic REST client supports the v2 route.
                    await self._api.request(
                        "POST", f"/dms/{msg.chat_id}/messages",
                        {"content": msg.text, "msg_id": mid},
                    )
                else:
                    await self._api.send_text(
                        scope, msg.chat_id, msg.text,
                        reply_to=mid, markdown=self.config.get("enable_markdown", False),
                        retries=1,
                    )
                return
            except Exception as exc:
                if not _is_temporary_dns_failure(exc):
                    if sent:
                        raise PartialSendError(sent, ["QQ text delivery unconfirmed"]) from None
                    raise RuntimeError(
                        "QQ official reply failed or delivery unconfirmed") from None
                if attempt < len(_DNS_RETRY_DELAYS):
                    await asyncio.sleep(_DNS_RETRY_DELAYS[attempt])
                    continue
                # DNS never resolved, so no HTTP POST was transmitted.
                if not sent:
                    async with self._reply_lock:
                        if self._attempted_replies.get(key) == now:
                            self._attempted_replies.pop(key, None)
                    raise RuntimeError(
                        "QQ official DNS resolution failed before delivery; retry permitted") from None
                raise PartialSendError(sent, ["QQ text DNS failed before delivery"]) from None

    async def _interactive_source(self, chat_id: str, source_id: str,
                                  actor_id: str) -> str:
        if not self.config.get("enable_interactions"):
            raise ValueError("QQ interactive features are disabled")
        key = (chat_id, source_id)
        source = self._source_scopes.get(key)
        if (source is None or source[0] not in ("c2c", "group")
                or time.monotonic() - source[1] > _REPLY_TTL[source[0]]
                or self._source_actors.get(key) != actor_id):
            raise ValueError("QQ interactive source or actor not verified")
        async with self._reply_lock:
            now = time.monotonic()
            while self._prompted_sources:
                _, oldest = next(iter(self._prompted_sources.items()))
                if now - oldest <= _SEEN_TTL:
                    break
                self._prompted_sources.popitem(last=False)
            if key in self._prompted_sources:
                raise ValueError("QQ interactive prompt already attempted")
            self._prompted_sources[key] = now
            if len(self._prompted_sources) > _SEEN_LIMIT:
                self._prompted_sources.popitem(last=False)
        return source[0]

    async def request_approval(self, chat_id: str, source_id: str, actor_id: str,
                               title: str, description: str = "") -> str:
        """Server-side trusted workflow only: queue one-shot approval prompt.

        A received decision never grants Agent Core tool permissions directly.
        """
        if not self.config.get("enable_approvals"):
            raise ValueError("QQ interactive approvals are disabled")
        scope = await self._interactive_source(chat_id, source_id, actor_id)
        return await self._interactions.send_approval(
            self._api, scope, chat_id, actor_id, source_id, title, description)

    async def request_choice(self, chat_id: str, source_id: str, actor_id: str,
                             label: str, choices: dict[str, str]) -> str:
        """Server-side trusted workflow only: non-privileged inline buttons."""
        scope = await self._interactive_source(chat_id, source_id, actor_id)
        return await self._interactions.send_choices(
            self._api, scope, chat_id, actor_id, source_id, label, choices)

    def take_interaction_result(self, nonce: str) -> str | None:
        return self._interactions.take(nonce)

    async def health_check(self) -> tuple[bool, str]:
        ok = bool(self._running and self._connected and self._connected.is_set())
        return ok, ("" if ok else (self._last_error or "QQ official Gateway disconnected"))

    def status(self) -> str:
        if self._running and self._connected and self._connected.is_set():
            return "connected"
        return "degraded" if self._running else "disconnected"
