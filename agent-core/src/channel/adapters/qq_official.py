"""Tencent QQ Gateway and OpenAPI text transport."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import logging
import re
import time

import aiohttp
import httpx
from qqbot_agent_sdk import EventParser, Intent, OPCode, QQApiClient, QQWebSocket, WSCallbacks
from qqbot_agent_sdk.dto import MessageToCreate

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage
from channel.qq_credentials import read_secret

_ID = re.compile(r'[A-Za-z0-9_-]{1,128}')
_EVENTS = {'C2C_MESSAGE_CREATE': 'c2c', 'GROUP_AT_MESSAGE_CREATE': 'group'}
_MESSAGE_LIMIT = 2000


def quiet_sdk_logs():
    # SDK errors can contain raw QQ HTTP bodies, session IDs or one-time bind URLs.
    logging.getLogger('qqbot_agent_sdk').setLevel(logging.CRITICAL + 1)


class QQGateway(QQWebSocket):
    """Keep SDK heartbeats and resume, with exact intents and a finite outage budget."""

    failures = 0

    async def _send_identify(self):
        token = await asyncio.to_thread(self._cb.get_token)
        await self._send_ws_json({
            'op': OPCode.IDENTIFY,
            'd': {'token': f'QQBot {token}', 'intents': int(Intent.GROUP_MESSAGES),
                  'shard': [0, 1], 'properties': {'$os': 'linux', '$browser': 'qqbot-sdk',
                                                 '$device': 'qqbot-sdk'}},
        }, 'Identify')

    async def _reconnect(self, backoff_idx):
        # SDK 1.2.2 resets its counter on TCP connection, before READY/RESUMED.
        self.failures += 1
        if self.failures > 5:
            self._stop_requested = True
            self._cb.on_fatal_error('reconnect_exhausted', '')
            return False
        return await super()._reconnect(max(backoff_idx, self.failures - 1))


@dataclass
class ReplySource:
    scope: str
    target: str
    original_id: str
    chat_id: str
    expires: float
    attempted: bool = False


class QQOfficialAdapter(ChannelAdapter):
    # One Gateway session per AppID: a second concurrent start would kick the first.
    SINGLE_SESSION = True

    def __init__(self, channel_id, platform, config, on_message):
        super().__init__(channel_id, platform, config, on_message)
        self._ready = asyncio.Event()
        self._connected = False
        self._last_ack = 0.0
        self._reconnecting = False
        self._api = self._http = self._session = self._gateway = None
        self._resume = (None, None)
        self._sources: OrderedDict[str, ReplySource] = OrderedDict()

    @property
    def reconnecting(self):
        return self._reconnecting

    def _session_changed(self, session_id, seq):
        self._resume = (session_id, seq)

    def _ack(self):
        self._last_ack = time.monotonic()

    def _on_connected(self):
        self._ack()
        self._connected = True
        self._reconnecting = False
        self._gateway.failures = 0
        self._ready.set()

    def _on_disconnected(self):
        self._connected = False
        self._reconnecting = self._running

    def _on_fatal(self, _code, _message):
        self._connected = self._reconnecting = False
        self._ready.set()

    async def start(self):
        if self._running:
            return
        quiet_sdk_logs()
        self._resume = (None, None)
        self._ready.clear()
        app_id = self.config['app_id']
        try:
            self._http = httpx.AsyncClient(timeout=10)
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
            self._api = QQApiClient(app_id, read_secret(app_id))
            self._api.setup(self._http)
            self._gateway = QQGateway(WSCallbacks(
                on_message_event=self._receive, on_connected=self._on_connected,
                on_disconnected=self._on_disconnected, on_fatal_error=self._on_fatal,
                get_token=self._api.ensure_token_sync, get_session=lambda: self._resume,
                set_session=self._session_changed, set_heartbeat_interval=lambda _: None,
                clear_token=self._api.clear_token, fail_pending=lambda _: None,
                get_gateway_url=self._api.get_gateway_url_sync, on_heartbeat_ack=self._ack,
            ))
            async def connect():
                url = await self._api.get_gateway_url()
                if not isinstance(url, str) or not url.startswith('wss://'):
                    raise ValueError('invalid QQ Gateway URL')
                await self._gateway.open(url, self._session)
                self._gateway.start_listeners()
                await self._ready.wait()

            await asyncio.wait_for(connect(), 20)
            if not self._connected:
                raise ConnectionError('Gateway did not identify')
            self._running = True
        except asyncio.CancelledError:
            await self.stop()
            raise
        except Exception:
            await self.stop()
            raise ConnectionError('QQ Gateway connection failed. Check network and QQ permissions.') from None

    async def stop(self):
        self._running = self._connected = self._reconnecting = False
        if self._gateway:
            await self._gateway.stop()
        if self._session:
            await self._session.close()
        if self._http:
            await self._http.aclose()
        self._gateway = self._session = self._http = self._api = None

    def status(self):
        interval = self._gateway._heartbeat_interval if self._gateway else 30
        healthy = self._connected and time.monotonic() - self._last_ack < max(60, 3 * interval)
        return 'connected' if healthy else 'disconnected'

    async def health_check(self):
        ok = self.status() == 'connected'
        return ok, '' if ok else 'QQ Gateway disconnected'

    async def _receive(self, kind, raw):
        if not self._connected or kind not in _EVENTS or not isinstance(raw, dict):
            return
        scope = _EVENTS[kind]
        if scope == 'group' and self.config.get('group_enabled') is not True:
            return
        # The authenticated socket itself is AppID-scoped. When the event also
        # supplies an AppID, it must match that authenticated connection.
        for key in ('app_id', 'appid', 'bot_appid', 'bot_app_id'):
            if key in raw and str(raw[key]) != self.config['app_id']:
                return
        author = raw.get('author')
        if not isinstance(author, dict) or author.get('bot') or raw.get('attachments'):
            return
        user_key = 'member_openid' if scope == 'group' else 'user_openid'
        identifiers = (author.get(user_key), raw.get('group_openid')) if scope == 'group' else (author.get(user_key),)
        if not all(isinstance(value, str) and _ID.fullmatch(value) for value in identifiers):
            return
        if not isinstance(raw.get('content'), str):
            return
        original_id = raw.get('id')
        if not isinstance(original_id, str) or not 0 < len(original_id) <= 512 or not original_id.isprintable():
            return
        try:
            event = EventParser.parse(kind, raw)
        except (TypeError, AttributeError, ValueError):
            return
        if not event or event.attachments or event.msg_elements:
            return
        user, target = event.user_id, event.chat_id
        if not 0 < len(event.content) <= _MESSAGE_LIMIT:
            return

        chat_id = f'{scope}:{target}'
        source_id = hashlib.sha256(json.dumps([self.channel_id, chat_id, original_id]).encode()).hexdigest()
        now = time.monotonic()
        while self._sources and next(iter(self._sources.values())).expires <= now:
            self._sources.popitem(last=False)
        if source_id in self._sources:
            return
        if len(self._sources) >= 2048:
            return
        self._sources[source_id] = ReplySource(scope, target, original_id, chat_id, now + 300)
        try:
            await self._on_message(InboundMessage(
                self.platform, self.channel_id, f'{chat_id}:{user}', chat_id, 'QQ user',
                event.content, source_id, chat_type='group' if scope == 'group' else 'p2p',
            ))
        except Exception:
            logging.getLogger(__name__).warning('QQ message could not enter the Channel input')

    async def send_message(self, msg):
        if self.status() != 'connected' or self._api is None:
            raise ConnectionError('QQ Gateway disconnected')
        if (msg.files or msg.image_bytes is not None or msg.mention_open_id or msg.expect_reply
                or not isinstance(msg.text, str) or not 0 < len(msg.text.strip()) <= _MESSAGE_LIMIT):
            raise ValueError('QQ replies require 1–2000 characters of plain text')
        source = self._sources.get(msg.reply_to_message_id)
        if (source is None or source.chat_id != msg.chat_id or source.expires <= time.monotonic()
                or source.attempted):
            raise ValueError('QQ reply source is unknown, expired or already used')
        # Use one stable sequence and reserve before I/O. An HTTP timeout can
        # occur after QQ accepted the message, so never resend automatically.
        source.attempted = True
        post = self._api.post_group_message if source.scope == 'group' else self._api.post_c2c_message
        try:
            result = await post(source.target, MessageToCreate(
                content=msg.text.strip(), msg_type=0, msg_id=source.original_id, msg_seq=1,
            ))
            if not isinstance(result, dict) or not result.get('id'):
                raise ValueError('missing QQ delivery ID')
        except Exception:
            raise RuntimeError('QQ delivery failed or was unconfirmed') from None
