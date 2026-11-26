"""Offline Dashboard setup: credentials, duplicate bots and cancellation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fastapi
import pytest

from api import channel as api
from channel import wecom_qr_onboarding as setup


@pytest.mark.parametrize('created,connected,state', [
    (True, True, 'connected'), (True, False, 'saved'), (False, True, 'duplicate'),
])
def test_setup_saves_once_and_reports_connection_state(monkeypatch, created, connected, state):
    monkeypatch.setattr(setup, '_SETUP_LOCK', asyncio.Lock())
    monkeypatch.setattr(setup, '_scan', AsyncMock(return_value=('test-bot', 'test-secret')))
    save = Mock(return_value=('wecom_test', created))
    monkeypatch.setattr(setup, '_save', save)
    restart = AsyncMock()
    monkeypatch.setattr(setup.manager, 'restart_adapter', restart)
    monkeypatch.setattr(setup.manager, '_adapters', {
        'wecom_test': SimpleNamespace(status=lambda: 'connected' if connected else 'disconnected'),
    })
    session = setup.Session(qr='temporary-qr')
    asyncio.run(setup._run(session))
    save.assert_called_once_with('test-bot', 'test-secret')
    assert session.state == state
    assert session.channel_id == 'wecom_test'
    assert not session.qr
    assert 'test-secret' not in str(session.snapshot())
    assert restart.await_count == int(created)


def test_duplicate_bot_does_not_create_another_channel(monkeypatch):
    monkeypatch.setattr(setup, '_get_channel_configs', lambda: [{
        'id': 'existing', 'platform': 'wecom', 'config': {'bot_id': 'bot'},
    }])
    add = Mock()
    monkeypatch.setattr(setup, 'add_channel_config', add)
    assert setup._save('bot', 'new-secret') == ('existing', False)
    add.assert_not_called()


def test_cancelled_scan_never_saves_credentials(monkeypatch):
    started = asyncio.Event()

    async def scan(_):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(setup, '_scan', scan)
    save = Mock()
    monkeypatch.setattr(setup, '_save', save)

    async def run():
        session = setup.Session()
        session.task = asyncio.create_task(setup._run(session))
        await started.wait()
        session.cancel()
        await asyncio.gather(session.task, return_exceptions=True)
        assert session.state == 'cancelled'
        assert not session.qr

    asyncio.run(run())
    save.assert_not_called()


def test_qr_api_requires_dashboard_authentication(monkeypatch):
    monkeypatch.setattr(api.auth, 'is_enabled', lambda: False)
    with pytest.raises(fastapi.HTTPException) as exc:
        asyncio.run(api.start_wecom_qr(fastapi.Response()))
    assert exc.value.status_code == 403
