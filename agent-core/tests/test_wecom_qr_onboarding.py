"""Offline coverage for WeCom QR authorization and credential handling."""

import asyncio
import json
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import config
from api import channel as channel_api
from channel import wecom_qr_onboarding as qr


BOT_ID = 'fake-bot-id'
SECRET = 'fake-secret-for-test'


class WecomQRTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {'channel_configs': []}
        self.manager = SimpleNamespace(_adapters={}, restart_adapter=AsyncMock())
        for target, name, value in (
            (config, 'main', self.config),
            (qr, 'manager', self.manager),
            (qr, '_SETUP_LOCK', asyncio.Lock()),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_authorized_credentials_are_saved_only_on_server(self):
        responses = [
            {'data': {'scode': 'fake-scode', 'auth_url': 'https://example.test/qr'}},
            {'data': {'status': 'waiting'}},
            {'data': {'status': 'success', 'bot_info': {'botid': BOT_ID, 'secret': SECRET}}},
        ]

        async def connect(channel_id, **_):
            self.manager._adapters[channel_id] = SimpleNamespace(status=lambda: 'connected')

        self.manager.restart_adapter.side_effect = connect
        session = qr.Session()
        with patch.object(qr, '_get_json', new=AsyncMock(side_effect=responses)) as get_json, \
             patch.object(qr.qrcode, 'make', return_value=SimpleNamespace(to_string=lambda: b'<svg/>')), \
             patch.object(qr.asyncio, 'sleep', new=AsyncMock()):
            await qr._run(session)

        channel = self.config['channel_configs'][0]
        self.assertEqual(channel['config'], {'bot_id': BOT_ID, 'secret': SECRET})
        self.assertTrue(channel['enabled'])
        self.assertEqual(session.state, 'connected')
        self.assertEqual(session.channel_id, channel['id'])
        self.assertEqual(get_json.await_args_list[0].args[1], qr._GENERATE_URL)
        self.assertEqual(get_json.await_args_list[0].args[2]['source'], qr._SOURCE)
        self.assertEqual(get_json.await_args_list[1].args[2], {'scode': 'fake-scode'})
        self.assertNotIn(SECRET, json.dumps(session.snapshot()))
        self.assertNotIn('fake-scode', json.dumps(session.snapshot()))
        self.assertEqual(session.qr, '')

    async def test_cancelled_scan_cannot_save_or_connect(self):
        waiting = asyncio.Event()

        async def get_json(_client, url, _params):
            if url == qr._GENERATE_URL:
                return {'data': {'scode': 'fake-scode', 'auth_url': 'https://example.test/qr'}}
            waiting.set()
            await asyncio.Event().wait()

        session = qr.Session()
        with patch.object(qr, '_get_json', side_effect=get_json), \
             patch.object(qr.qrcode, 'make', return_value=SimpleNamespace(to_string=lambda: b'<svg/>')):
            session.task = asyncio.create_task(qr._run(session))
            await waiting.wait()
            self.assertTrue(session.qr)
            session.cancel()
            await asyncio.gather(session.task, return_exceptions=True)

        self.assertEqual(session.state, 'cancelled')
        self.assertEqual(session.qr, '')
        self.assertEqual(self.config['channel_configs'], [])
        self.manager.restart_adapter.assert_not_awaited()

    async def test_duplicate_bot_does_not_overwrite_or_start_another_connection(self):
        existing = {'id': 'existing', 'platform': 'wecom',
                    'config': {'bot_id': BOT_ID, 'secret': 'existing-secret'}}
        self.config['channel_configs'].append(existing)
        session = qr.Session()
        with patch.object(qr, '_scan', new=AsyncMock(return_value=(BOT_ID, SECRET))):
            await qr._run(session)
        self.assertEqual(session.state, 'duplicate')
        self.assertEqual(session.channel_id, 'existing')
        self.assertEqual(self.config['channel_configs'], [existing])
        self.assertEqual(existing['config']['secret'], 'existing-secret')
        self.manager.restart_adapter.assert_not_awaited()

    async def test_timeout_and_connection_failure_are_reported_accurately(self):
        async def never(_session):
            await asyncio.Event().wait()

        session = qr.Session()
        with patch.object(qr, '_TTL', 0.01), patch.object(qr, '_scan', side_effect=never):
            await qr._run(session)
        self.assertEqual(session.state, 'expired')
        self.assertEqual(self.config['channel_configs'], [])

        session = qr.Session()
        with patch.object(qr, '_scan', new=AsyncMock(return_value=(BOT_ID, SECRET))):
            await qr._run(session)
        self.assertEqual(session.state, 'saved')
        self.assertEqual(len(self.config['channel_configs']), 1)

    async def test_request_timeout_is_not_reported_as_qr_expiry(self):
        class TimedOut:
            async def __aenter__(self):
                raise asyncio.TimeoutError()

            async def __aexit__(self, *_args):
                return False

        client = SimpleNamespace(get=lambda *_args, **_kwargs: TimedOut())

        async def scan(_session):
            await qr._get_json(client, qr._GENERATE_URL, {})

        session = qr.Session()
        with patch.object(qr, '_scan', side_effect=scan):
            await qr._run(session)
        self.assertEqual(session.state, 'failed')
        self.assertEqual(self.config['channel_configs'], [])

    async def test_success_without_credentials_does_not_create_channel(self):
        responses = [
            {'data': {'scode': 'fake-scode', 'auth_url': 'https://example.test/qr'}},
            {'data': {'status': 'success', 'bot_info': {'botid': BOT_ID}}},
        ]
        session = qr.Session()
        with patch.object(qr, '_get_json', new=AsyncMock(side_effect=responses)), \
             patch.object(qr.qrcode, 'make', return_value=SimpleNamespace(to_string=lambda: b'<svg/>')):
            await qr._run(session)
        self.assertEqual(session.state, 'failed')
        self.assertEqual(self.config['channel_configs'], [])
        self.manager.restart_adapter.assert_not_awaited()

    async def test_save_failure_logs_phase_without_credentials(self):
        session = qr.Session()
        with patch.object(qr, '_scan', new=AsyncMock(return_value=(BOT_ID, SECRET))), \
             patch.object(qr, '_save', side_effect=RuntimeError(SECRET)), \
             patch('builtins.print') as log:
            await qr._run(session)
        self.assertEqual(session.state, 'failed')
        self.assertEqual(self.config['channel_configs'], [])
        self.assertIn('phase=configuring', str(log.call_args_list))
        self.assertIn('error_type=RuntimeError', str(log.call_args_list))
        self.assertNotIn(SECRET, str(log.call_args_list))
        self.assertNotIn(SECRET, json.dumps(session.snapshot()))

    def test_scan_requires_dashboard_authentication(self):
        with patch.object(channel_api.auth, 'is_enabled', return_value=False):
            with self.assertRaises(channel_api.fastapi.HTTPException) as error:
                channel_api._wecom_setup()
        self.assertEqual(error.exception.status_code, 403)


if __name__ == '__main__':
    unittest.main()
