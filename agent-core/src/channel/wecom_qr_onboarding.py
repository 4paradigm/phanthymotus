"""Short-lived WeCom QR authorization sessions for the Dashboard.

Follows the official WeCom CLI's QR flow. Bot credentials stay on the server.
"""

import asyncio
import base64
from dataclasses import dataclass, field
import secrets
import sys
import time

import aiohttp
import qrcode
from qrcode.image.svg import SvgPathImage

from channel.manager import _get_channel_configs, add_channel_config, manager

_GENERATE_URL = 'https://work.weixin.qq.com/ai/qc/generate'
_QUERY_URL = 'https://work.weixin.qq.com/ai/qc/query_result'
_SOURCE = 'wecom_cli_external'
_TTL = 300
_SESSIONS = {}
_SETUP_LOCK = asyncio.Lock()


@dataclass
class Session:
    id: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    created: float = field(default_factory=time.monotonic)
    state: str = 'starting'
    qr: str = ''
    channel_id: str = ''
    task: asyncio.Task | None = None

    def snapshot(self):
        return {'session_id': self.id, 'status': self.state,
                'qr_data_url': self.qr, 'channel_id': self.channel_id}

    def cancel(self):
        if self.state in ('starting', 'waiting'):
            self.state = 'cancelled'
            self.qr = ''
            self.task.cancel()


async def _get_json(client, url, params):
    try:
        async with client.get(url, params=params, allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError('WeCom QR request failed')
            result = await response.json(content_type=None)
            if not isinstance(result, dict):
                raise ValueError('Invalid WeCom QR response')
            return result
    except asyncio.TimeoutError:
        raise ConnectionError('WeCom QR request timed out') from None


async def _scan(session):
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as client:
        plat = {'darwin': 1, 'win32': 2, 'linux': 3}.get(sys.platform, 0)
        result = await _get_json(client, _GENERATE_URL, {'source': _SOURCE, 'plat': plat})
        data = result.get('data') or {}
        if not isinstance(data, dict):
            raise ValueError('Invalid WeCom QR response')
        scode, url = data.get('scode'), data.get('auth_url')
        if not isinstance(scode, str) or not scode or not isinstance(url, str) or not url:
            raise ValueError('Invalid WeCom QR response')
        qr = qrcode.make(url, image_factory=SvgPathImage)
        session.qr = 'data:image/svg+xml;base64,' + base64.b64encode(qr.to_string()).decode('ascii')
        session.state = 'waiting'

        while True:
            result = await _get_json(client, _QUERY_URL, {'scode': scode})
            data = result.get('data') or {}
            if not isinstance(data, dict):
                raise ValueError('Invalid WeCom QR response')
            if data.get('status') == 'success':
                info = data.get('bot_info') or {}
                if not isinstance(info, dict):
                    raise ValueError('Missing WeCom bot credentials')
                bot_id, secret = info.get('botid'), info.get('secret')
                if (not isinstance(bot_id, str) or not bot_id.strip()
                        or not isinstance(secret, str) or not secret.strip()):
                    raise ValueError('Missing WeCom bot credentials')
                return bot_id.strip(), secret.strip()
            await asyncio.sleep(3)


def _save(bot_id, secret):
    for channel in _get_channel_configs():
        if channel['platform'] == 'wecom' and channel.get('config', {}).get('bot_id') == bot_id:
            return channel['id'], False
    channel_id = 'wecom_' + secrets.token_hex(8)
    add_channel_config(channel_id, 'wecom', {'bot_id': bot_id, 'secret': secret}, enabled=True)
    return channel_id, True


async def _run(session):
    try:
        bot_id, secret = await asyncio.wait_for(_scan(session), timeout=_TTL)
        session.qr = ''
        async with _SETUP_LOCK:
            session.state = 'configuring'
            channel_id, created = await asyncio.to_thread(_save, bot_id, secret)
            session.channel_id = channel_id
        del bot_id, secret
        if not created:
            session.state = 'duplicate'
            return
        session.state = 'connecting'
        await manager.restart_adapter(channel_id, retries=1)
        adapter = manager._adapters.get(channel_id)
        session.state = 'connected' if adapter and adapter.status() == 'connected' else 'saved'
    except asyncio.TimeoutError:
        session.state = 'expired' if not session.channel_id else 'saved'
    except Exception as exc:
        print(f'[wecom] QR setup failed: phase={session.state} error_type={type(exc).__name__}')
        session.state = 'failed' if not session.channel_id else 'saved'
    finally:
        session.qr = ''


def start():
    now = time.monotonic()
    for key, session in list(_SESSIONS.items()):
        if session.task.done() and (session.state == 'cancelled' or now - session.created > _TTL * 3):
            del _SESSIONS[key]
    if len(_SESSIONS) >= 64 or sum(not s.task.done() for s in _SESSIONS.values()) >= 4:
        return None
    session = Session()
    _SESSIONS[session.id] = session
    session.task = asyncio.create_task(_run(session))
    return session.snapshot()


def get(session_id):
    session = _SESSIONS.get(session_id)
    if session and time.monotonic() - session.created <= _TTL * 3:
        return session
    return None
