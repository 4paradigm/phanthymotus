"""
test_peer_lan_adapter.py — lan ChannelAdapter 集成。

验证节点（按计划）：
9. health_check 对端不可达时返回 False
"""

import asyncio
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

import config  # noqa: E402
from peer import identity, store  # noqa: E402
from channel.adapters.lan import LanAdapter  # noqa: E402


class TestLanAdapter(unittest.TestCase):
    def setUp(self):
        # config.DB_PATH is a module-level constant read at import time, so the
        # os.environ assignment above only wins if this file happens to import config
        # first. Run under the full suite and it does not: store then reads whichever
        # database an earlier test set up, and `test_health_check_no_peers` finds that
        # test's peers instead of none. Patching the attribute is what actually
        # isolates us — the same trap is noted in test_peer_mutual.py.
        self._db = os.path.join(tempfile.mkdtemp(), 'peers.db')
        self._patch = mock.patch.object(config, 'DB_PATH', self._db)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        identity.reset_cache()
        identity.ensure_identity()

    def test_health_check_no_peers(self):
        """没有配对 peer 时，health_check 仍通过"""
        async def _on_msg(msg):
            pass

        adapter = LanAdapter('test_lan', 'lan', {}, _on_msg)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(adapter.start())
            ok, reason = loop.run_until_complete(adapter.health_check())
            self.assertTrue(ok)
            self.assertEqual(reason, 'no peers paired')
            loop.run_until_complete(adapter.stop())
        finally:
            loop.close()

    def test_health_check_peer_unreachable(self):
        """配对 peer 不可达时，health_check 返回 False（节点 9）"""
        # 配一个 peer，但没有真实端点
        store.upsert('unreachable_peer', identity.public_key_b64(), 'Unreachable',
                     role='viewer', endpoints=['https://192.0.2.1:15678'])

        async def _on_msg(msg):
            pass

        adapter = LanAdapter('test_lan', 'lan', {}, _on_msg)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(adapter.start())
            ok, reason = loop.run_until_complete(adapter.health_check())
            self.assertFalse(ok)
            self.assertIn('no paired peer reachable', reason)
            loop.run_until_complete(adapter.stop())
        finally:
            loop.close()

    def test_send_message_no_endpoints(self):
        """没有已知端点时，send_message 抛异常"""
        from channel.adapter import OutboundMessage

        store.upsert('peer_no_ep', identity.public_key_b64(), 'NoEP', role='viewer')

        async def _on_msg(msg):
            pass

        adapter = LanAdapter('test_lan', 'lan', {}, _on_msg)
        msg = OutboundMessage(
            chat_id='peer_no_ep',
            text='test',
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(adapter.start())
            with self.assertRaises(RuntimeError) as ctx:
                loop.run_until_complete(adapter.send_message(msg))
            self.assertIn('no known endpoint', str(ctx.exception))
            loop.run_until_complete(adapter.stop())
        finally:
            loop.close()


if __name__ == '__main__':
    unittest.main()
