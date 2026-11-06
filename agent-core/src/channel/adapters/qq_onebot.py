"""QQ private-text Channel over NapCat's OneBot v11 forward WebSocket.

ChannelManager owns reconnection, ACL, Canvas routing, and persistence.
This adapter owns only the transport, OneBot parsing, and in-flight API calls.
"""
import asyncio
from collections import OrderedDict
import json
import logging
import os
import re
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage

_LOG = logging.getLogger(__name__)
_MAX_INBOUND = 2000
_MAX_OUTBOUND = 4000
_MAX_PENDING_DELIVERIES = 16
_SEEN_LIMIT = 1024
_SEEN_TTL = 600.0


def validate_qq_config(cfg: dict) -> dict:
    """Validate without ever exposing secrets in an exception."""
    if not isinstance(cfg, dict):
        raise ValueError('QQ config must be an object')
    url = str(cfg.get('ws_url', '')).strip()
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ValueError('QQ ws_url is invalid') from None
    if (parsed.scheme not in ('ws', 'wss') or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not (1 <= (port or (443 if parsed.scheme == 'wss' else 80)) <= 65535)):
        raise ValueError('QQ ws_url must be a ws(s) URL without credentials/query')
    if not (cfg.get('access_token') or cfg.get('access_token_env') or
            cfg.get('access_token_file')):
        raise ValueError('QQ access_token, access_token_env or access_token_file required')
    if cfg.get('access_token_file') not in (None, '', '/run/qq-token'):
        raise ValueError('QQ access_token_file must use the designated mounted secret')
    if sum(bool(cfg.get(k)) for k in ('access_token', 'access_token_env', 'access_token_file')) != 1:
        raise ValueError('Configure exactly one QQ credential source')
    token_env = cfg.get('access_token_env')
    if token_env and (not isinstance(token_env, str) or
                      not re.fullmatch(r'QQ_ONEBOT_[A-Z][A-Z0-9_]{0,63}', token_env)):
        raise ValueError('QQ access_token_env must use a QQ_ONEBOT_* variable')
    if cfg.get('private_only', True) is not True:
        raise ValueError('QQ V1 only supports private_only=true')
    for key, lo, hi in (
        ('connect_timeout', 1, 60), ('request_timeout', 1, 60),
        ('heartbeat_timeout', 10, 600),
    ):
        if key in cfg:
            v = cfg[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError(f'QQ {key} must be between {lo} and {hi}')
    return cfg


def _plain_text(raw) -> str:
    """Reject mixed media rather than interpreting text inside an unsupported command."""
    if isinstance(raw, str):
        if '[CQ:' in raw:
            return ''
        return raw.strip()
    if not isinstance(raw, list) or not raw:
        return ''
    parts = []
    for segment in raw:
        if not isinstance(segment, dict) or segment.get('type') != 'text':
            return ''
        data = segment.get('data')
        if not isinstance(data, dict) or not isinstance(data.get('text'), str):
            return ''
        parts.append(data['text'])
    return ''.join(parts).strip()


class QQOneBotAdapter(ChannelAdapter):
    """Forward-WS OneBot v11 adapter: private messages and text replies only."""

    def __init__(self, channel_id: str, platform: str, config: dict, on_message):
        super().__init__(channel_id, platform, config, on_message)
        self._session = None
        self._ws = None
        self._reader = None
        self._pending: dict[str, asyncio.Future] = {}
        self._deliveries: set[asyncio.Task] = set()
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._self_id = ''
        self._last_frame = 0.0
        self._last_error = ''
        self._closing = False

    async def start(self) -> None:
        validate_qq_config(self.config)
        if self._running:
            return
        self._closing = False
        # A WebSocket handshake is not proof of an authenticated QQ identity.
        # Never reuse the identity from a previous connection during login.
        self._self_id = ''
        token = str(self.config.get('access_token') or '')
        token_env = str(self.config.get('access_token_env') or '').strip()
        if token_env:
            if not token_env.isidentifier():
                raise ValueError('QQ access_token_env must name an environment variable')
            token = os.environ.get(token_env, '')
            if not token:
                raise ValueError('QQ access token environment variable is not set')
        if self.config.get('access_token_file'):
            try:
                token = Path('/run/qq-token').read_text(encoding='utf-8').strip()
            except OSError:
                raise RuntimeError('QQ mounted access token is not readable') from None
            if not token:
                raise RuntimeError('QQ mounted access token is empty')
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        connect_timeout = float(self.config.get('connect_timeout', 8))
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=connect_timeout))
        try:
            self._ws = await self._session.ws_connect(
                self.config['ws_url'], headers=headers, heartbeat=20,
                max_msg_size=128 * 1024)
            self._last_frame = time.monotonic()
            self._running = True
            self._reader = asyncio.create_task(self._read_loop())
            identity = await self._call('get_login_info', {})
            uid = (identity.get('data') or {}).get('user_id')
            if not str(uid or '').isdigit():
                raise RuntimeError('OneBot get_login_info returned no valid user_id')
            self._self_id = str(uid)
            self._last_error = ''
            _LOG.info('QQ Channel connected: %s', self.channel_id)
        except Exception as exc:
            # Do not expose ws URL, token, or messages in errors.
            if isinstance(exc, aiohttp.WSServerHandshakeError) and exc.status in (401, 403):
                self._last_error = 'OneBot WebSocket rejected credentials (HTTP 401/403)'
            else:
                self._last_error = 'OneBot authentication/connection/identity check failed'
            await self.stop()
            raise RuntimeError(self._last_error) from None

    async def stop(self) -> None:
        self._closing = True
        self._running = False
        self._self_id = ''
        if self._reader:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
            self._reader = None
        for task in list(self._deliveries):
            task.cancel()
        if self._deliveries:
            await asyncio.gather(*self._deliveries, return_exceptions=True)
        self._deliveries.clear()
        self._fail_pending('OneBot connection closed')
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._session:
            await self._session.close()
            self._session = None

    def _fail_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError(reason))
        self._pending.clear()

    async def _call(self, action: str, params: dict) -> dict:
        if not self._running or not self._ws or self._ws.closed:
            raise ConnectionError('QQ Channel is disconnected')
        echo = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[echo] = future
        try:
            await self._ws.send_json({'action': action, 'params': params, 'echo': echo})
            response = await asyncio.wait_for(
                future, timeout=float(self.config.get('request_timeout', 8)))
            if response.get('status') != 'ok' or response.get('retcode') not in (0, None):
                raise RuntimeError(
                    f'OneBot {action} failed (retcode={response.get("retcode", "unknown")})')
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(f'OneBot {action} timed out; delivery is unconfirmed') from None
        finally:
            self._pending.pop(echo, None)

    async def send_message(self, msg: OutboundMessage) -> None:
        if msg.files or msg.image_bytes:
            raise ValueError('QQ V1 does not support attachments')
        text = msg.text or ''
        if not isinstance(text, str) or not 0 < len(text) <= _MAX_OUTBOUND:
            raise ValueError('QQ text must contain 1-4000 characters')
        chat_id = str(msg.chat_id)
        if not chat_id.isdigit() or not 1 <= len(chat_id) <= 20 or int(chat_id) == 0:
            raise ValueError('QQ private chat_id must be a valid QQ user ID')
        await self._call('send_private_msg', {'user_id': int(chat_id), 'message': text})

    async def _read_loop(self) -> None:
        try:
            async for event in self._ws:
                if event.type != aiohttp.WSMsgType.TEXT:
                    if event.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        break
                    continue
                self._last_frame = time.monotonic()
                try:
                    data = json.loads(event.data)
                except (ValueError, TypeError):
                    _LOG.warning('QQ Channel received invalid JSON')
                    continue
                if not isinstance(data, dict):
                    continue
                echo = data.get('echo')
                if isinstance(echo, str) and echo in self._pending:
                    pending = self._pending[echo]
                    if not pending.done():
                        pending.set_result(data)
                elif data.get('post_type') == 'message':
                    msg = self._parse_message(data)
                    if msg and len(self._deliveries) < _MAX_PENDING_DELIVERIES:
                        task = asyncio.create_task(self._deliver(msg))
                        self._deliveries.add(task)
                        task.add_done_callback(self._deliveries.discard)
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOG.warning('QQ Channel receive loop failed')
        finally:
            self._running = False
            self._fail_pending('OneBot WebSocket disconnected')
            if not self._closing:
                self._last_error = 'OneBot WebSocket disconnected'

    async def _deliver(self, msg: InboundMessage) -> None:
        try:
            await self._on_message(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOG.warning('QQ Channel inbound delivery failed (details suppressed)')

    def _parse_message(self, event: dict) -> InboundMessage | None:
        if event.get('message_type') != 'private' or event.get('post_type') != 'message':
            return None
        if event.get('sub_type') == 'self' or event.get('self_id') is None:
            return None
        sender = str(event.get('user_id', ''))
        self_id = str(event.get('self_id', ''))
        # Reject events before get_login_info has completed and events claiming
        # another bot identity. This also covers stale frames after reconnection.
        if (not self._self_id or not sender.isdigit()
                or self_id != self._self_id or sender == self._self_id):
            return None
        mid = str(event.get('message_id', ''))
        if not mid or len(mid) > 96:
            return None
        # OneBot message may be a string or a segment array. If raw_message is a
        # non-text CQ representation, do not silently convert media to commands.
        text = _plain_text(event.get('message'))
        if not text or len(text) > _MAX_INBOUND:
            return None
        now = time.monotonic()
        for key, ts in list(self._seen.items()):
            if now - ts <= _SEEN_TTL:
                break
            self._seen.pop(key, None)
        key = f'{self_id}:{mid}'
        if key in self._seen:
            return None
        self._seen[key] = now
        while len(self._seen) > _SEEN_LIMIT:
            self._seen.popitem(last=False)
        sender_info = event.get('sender') or {}
        if not isinstance(sender_info, dict):
            sender_info = {}
        name = str(sender_info.get('card') or sender_info.get('nickname') or sender)[:100]
        return InboundMessage(
            platform='qq', channel_id=self.channel_id, user_id=sender,
            chat_id=sender, display_name=name, text=text,
            message_id=mid, chat_type='p2p', sender_type='user')

    async def health_check(self) -> tuple[bool, str]:
        if not self._running or not self._ws or self._ws.closed:
            return False, self._last_error or 'OneBot WebSocket is disconnected'
        if not self._self_id:
            return False, 'OneBot login identity is not verified'
        if self._reader is None or self._reader.done():
            return False, 'OneBot receive loop stopped'
        if time.monotonic() - self._last_frame > float(self.config.get('heartbeat_timeout', 90)):
            return False, 'OneBot heartbeat/event timeout'
        return True, ''

    def status(self) -> str:
        if not self._running or not self._ws or self._ws.closed or not self._self_id:
            return 'disconnected'
        if time.monotonic() - self._last_frame > float(self.config.get('heartbeat_timeout', 90)):
            return 'degraded'
        return 'connected'
