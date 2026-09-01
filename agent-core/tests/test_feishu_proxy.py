"""Feishu connections must honor the container proxy environment.

Run: cd agent-core && python3 -m pytest tests/test_feishu_proxy.py -q
"""

import asyncio
import pathlib
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from channel.adapters import feishu  # noqa: E402


async def _on_message(_message):
    return None


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {'Content-Type': 'application/json'}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, content_type=None):
        return self._payload


class _Session:
    def __init__(self, seen, payload, **kwargs):
        seen.append(kwargs)
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, _url, json):
        return _Response(self._payload)

    def request(self, _method, _url, **kwargs):
        return _Response(self._payload)


def _adapter():
    return feishu.FeishuAdapter(
        'test-feishu',
        'feishu',
        {'app_id': 'test-app', 'app_secret': 'test-secret'},
        _on_message,
    )


class FeishuProxyTest(unittest.TestCase):
    def test_tenant_token_session_honors_proxy_environment(self):
        seen = []
        payload = {'code': 0, 'tenant_access_token': 'token', 'expire': 7200}
        factory = lambda **kwargs: _Session(seen, payload, **kwargs)

        with mock.patch.object(feishu.aiohttp, 'ClientSession', factory):
            token = asyncio.run(_adapter()._tenant_token(force=True))

        self.assertEqual(token, 'token')
        self.assertTrue(seen)
        self.assertIs(seen[0]['trust_env'], True)

    def test_open_api_session_honors_proxy_environment(self):
        seen = []
        payload = {'code': 0, 'data': {'ok': True}}
        factory = lambda **kwargs: _Session(seen, payload, **kwargs)
        adapter = _adapter()
        adapter._token = 'token'
        adapter._token_expire = time.time() + 600

        with mock.patch.object(feishu.aiohttp, 'ClientSession', factory):
            result = asyncio.run(adapter._request('GET', '/open-apis/bot/v3/info'))

        self.assertEqual(result, {'ok': True})
        self.assertTrue(seen)
        self.assertIs(seen[0]['trust_env'], True)

    def test_websocket_sdk_proxy_patch_is_idempotent(self):
        import lark_oapi.ws.client as ws_mod

        with mock.patch.object(
            ws_mod, '_ws_connect_kwargs', return_value={'proxy': None, 'ping_interval': 30}
        ):
            feishu._enable_sdk_env_proxy(ws_mod)
            patched = ws_mod._ws_connect_kwargs
            feishu._enable_sdk_env_proxy(ws_mod)

            self.assertIs(ws_mod._ws_connect_kwargs, patched)
            self.assertEqual(patched(), {'ping_interval': 30})


if __name__ == '__main__':
    unittest.main()
