"""Short-lived Dashboard sessions for the official SDK's QR flow."""

import asyncio
import base64
from dataclasses import dataclass, field
import logging
import secrets
import time

import qrcode
from qrcode.image.svg import SvgPathImage
from qqbot_agent_sdk import OnboardExpiredError, start_onboard

from channel.adapters.qq_official import quiet_sdk_logs
from channel.manager import add_channel_config, get_channel_config, manager, update_channel_config
from channel import qq_canvas
from channel.qq_credentials import validate_app_id, write_secret

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
        status = self.state
        if status == 'saved':
            adapter = manager._adapters.get(self.channel_id)
            if not adapter or adapter.status() != 'connected':
                status = 'gateway_pending'
            elif qq_canvas.input_ready(self.channel_id):
                status = 'connected'
            elif qq_canvas.wired(self.channel_id):
                status = 'canvas_pending'
            else:
                status = 'canvas_setup'
        return {'session_id': self.id, 'status': status, 'qr_data_url': self.qr,
                'channel_id': self.channel_id}

    def cancel(self):
        if self.state in ('starting', 'waiting'):
            self.state = 'cancelled'
            self.qr = ''
            self.task.cancel()


def _save_credentials_config(app_id, secret, channel_id):
    """Keep synchronous credential and SQLite writes off the event loop."""
    existing = get_channel_config(channel_id)
    if existing and (existing['platform'] != 'qq' or existing['config'].get('app_id') != app_id):
        raise ValueError('QQ channel ID already in use')
    write_secret(app_id, secret)
    if existing:
        update_channel_config(channel_id, enabled=True, chat_only=True)
    else:
        add_channel_config(channel_id, 'qq', {'app_id': app_id, 'group_enabled': False}, True,
                           chat_only=True)


async def _run(session):
    def qr_ready(url):
        if session.state != 'starting':
            return
        qr = qrcode.make(url, image_factory=SvgPathImage)
        session.qr = 'data:image/svg+xml;base64,' + base64.b64encode(qr.to_string()).decode('ascii')
        session.state = 'waiting'

    quiet_sdk_logs()
    try:
        result = await asyncio.wait_for(
            start_onboard(on_qr_ready=qr_ready, poll_timeout=_TTL), timeout=_TTL,
        )
        session.state = 'configuring'
        session.qr = ''
        async with _SETUP_LOCK:
            app_id = validate_app_id(result.app_id)
            channel_id = f'qq_{app_id}'
            if channel_id in manager._starting:
                raise ValueError('QQ channel already connecting')
            await asyncio.to_thread(_save_credentials_config, app_id, result.client_secret, channel_id)
            del result
            session.channel_id = channel_id
            try:
                changed = await asyncio.to_thread(qq_canvas.wire, channel_id, notify=False)
                if changed is True:
                    from api.canvas import notify_layout_changed
                    notify_layout_changed()
            except Exception:
                # The channel is already durable. Its state can be checked
                # while the operator completes Canvas wiring manually.
                logging.getLogger(__name__).warning('QQ Canvas setup needs manual wiring')
            session.state = 'saved'
            manager._retry_at.pop(channel_id, None)
            manager._retry_backoff.pop(channel_id, None)
            await manager.restart_adapter(channel_id, retries=1)
    except (TimeoutError, OnboardExpiredError):
        session.state = 'expired' if not session.channel_id else 'saved'
    except Exception:
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
