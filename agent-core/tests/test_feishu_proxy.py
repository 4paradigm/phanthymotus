"""Feishu connections must honor the container proxy environment.

Run: cd agent-core && python3 -m pytest tests/test_feishu_proxy.py -q
"""

import asyncio
import pathlib
import sys
import time
import types
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

class SdkProxyPatchTest(unittest.TestCase):
    """`_enable_sdk_env_proxy` is duck-typed — it only getattr/setattrs what it is
    handed. The earlier version of these tests imported `lark_oapi.ws.client` to
    get that object and then mocked the very attribute it came for, so the real
    SDK bought nothing but the attribute's existence — while making the whole
    file fail on any machine without the package installed. A stub carries the
    same information and runs everywhere.
    """

    @staticmethod
    def _stub(**attrs):
        return types.SimpleNamespace(**attrs)

    def test_patch_strips_proxy_and_keeps_everything_else(self):
        mod = self._stub(_ws_connect_kwargs=lambda: {'proxy': None, 'ping_interval': 30})
        feishu._enable_sdk_env_proxy(mod)
        self.assertEqual(mod._ws_connect_kwargs(), {'ping_interval': 30})

    def test_patch_is_idempotent(self):
        """Applied twice, the second call must be a no-op.

        Not cosmetic: each application wraps the previous function, so a
        re-import that patched again would nest a chain of closures whose depth
        grows with the number of reconnects.
        """
        mod = self._stub(_ws_connect_kwargs=lambda: {'proxy': None, 'ping_interval': 30})
        feishu._enable_sdk_env_proxy(mod)
        patched = mod._ws_connect_kwargs
        feishu._enable_sdk_env_proxy(mod)

        self.assertIs(mod._ws_connect_kwargs, patched)
        self.assertEqual(patched(), {'ping_interval': 30})

    def test_module_without_the_hook_is_left_alone(self):
        """An SDK version that renamed or dropped the function must not crash us —
        Feishu still has to connect, it just will not honour the proxy env."""
        mod = self._stub()
        feishu._enable_sdk_env_proxy(mod)
        self.assertFalse(hasattr(mod, '_ws_connect_kwargs'))

    def test_real_sdk_still_exposes_the_hook(self):
        """Canary for an upstream rename. Skips where the SDK is absent — this is
        the only assertion here that genuinely needs it, and its failing means
        lark_oapi changed, not that our patch is wrong."""
        try:
            import lark_oapi.ws.client as ws_mod
        except ImportError as e:
            self.skipTest(f'lark_oapi not installed ({e})')
        self.assertTrue(
            callable(getattr(ws_mod, '_ws_connect_kwargs', None)),
            'lark_oapi.ws.client._ws_connect_kwargs is gone — _enable_sdk_env_proxy '
            'now silently does nothing and Feishu will ignore HTTPS_PROXY',
        )


if __name__ == '__main__':
    unittest.main()
