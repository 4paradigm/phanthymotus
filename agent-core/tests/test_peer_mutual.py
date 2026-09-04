"""
test_peer_mutual.py — 单边配对必须看得出是单边的。

配对按方向各存一份：只在一台机器上确认，那台会把对方列为"已配对"，而对端会 403 掉
每个签名请求 —— 而这一侧界面看起来完全正常。现在：

  - 默认是"等待对方确认"（mutual_at=None）
  - 对端接受一个我们签名的请求（推送成功），或我们收到对端签名的请求，任一发生就
    标记为 mutual（双向完成）
  - 接口里带上 mutual 字段；界面上单边时显示"等待对方确认"，双向完成自动转为正常
"""

import os
import sys
import pathlib
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from peer import store, identity  # noqa: E402


class TestMutualTracking(unittest.TestCase):
    def setUp(self):
        # config.DB_PATH is read at import time, so patching os.environ does nothing —
        # every test would then share one database and an earlier test's touch()
        # would leave mutual_at set on the row this one expects to be fresh.
        # Patch the module attribute instead.
        import config
        self._db = os.path.join(tempfile.mkdtemp(), 'peers.db')
        self._patch = mock.patch.object(config, 'DB_PATH', self._db)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _peer(self):
        identity.reset_cache()
        identity.ensure_identity()
        return store.upsert('a' * 32, identity.public_key_b64(), 'Far',
                            role='viewer', endpoints=['https://10.0.0.5:15678'])

    def test_a_new_pairing_has_no_mutual_yet(self):
        """本机确认完，对端还没回过话 —— 界面应该显示"等待对方"，不是"配对成功"。"""
        p = self._peer()
        self.assertIsNone(p.get('mutual_at'), 'mutual_at 必须默认是 None，不是 0 或空字串')

    def test_an_inbound_authenticated_request_marks_mutual(self):
        """对端发来任何验过签的请求，就证明它有我们的记录。"""
        self._peer()
        store.touch('a' * 32, 'https://10.0.0.5:15678')
        p = store.get('a' * 32)
        self.assertIsNotNone(p['mutual_at'], 'touch 包含了这个证据，应该标上')

    def test_mark_mutual_is_idempotent(self):
        """重复调用不该出错、也不该让时间戳倒退。"""
        self._peer()
        store.mark_mutual('a' * 32)
        first = store.get('a' * 32)['mutual_at']
        import time
        time.sleep(0.01)
        store.mark_mutual('a' * 32)
        second = store.get('a' * 32)['mutual_at']
        self.assertGreaterEqual(second, first)

    def test_the_paired_endpoint_includes_mutual(self):
        """接口里要有这个字段，界面才能用它做判断。

        list_paired 是 async 且在函数体内 import 依赖，所以这里 patch 的是被 import
        的模块本身，而不是 api.peer 上的属性 —— 后者不存在，第一版就是这样写错的。
        """
        import asyncio
        from api import peer as peer_api

        self._peer()
        row = store.get('a' * 32)
        with mock.patch('peer.store.list_peers', return_value=[row]), \
             mock.patch('peer.liveness.liveness', return_value={
                 'online': True, 'agent_running': None, 'contact_age_s': 1.0}), \
             mock.patch('peer.naming.labels', return_value={row['peer_id']: 'Far'}), \
             mock.patch('peer.mcp_bridge.offered', {}), \
             mock.patch('peer.dds_state.push_errors', {}):
            resp = asyncio.run(peer_api.list_paired())
        self.assertIn('mutual', resp['peers'][0])
        self.assertFalse(resp['peers'][0]['mutual'], '还没有证据，应该是 False')
        self.assertIsNone(resp['peers'][0]['mutual_at'],
                          'mutual_at 原始值也要带上 —— 界面靠它区分"从未配对成功"和"对方解除了"')

        store.mark_mutual('a' * 32)
        row = store.get('a' * 32)
        with mock.patch('peer.store.list_peers', return_value=[row]), \
             mock.patch('peer.liveness.liveness', return_value={
                 'online': True, 'agent_running': None, 'contact_age_s': 1.0}), \
             mock.patch('peer.naming.labels', return_value={row['peer_id']: 'Far'}), \
             mock.patch('peer.mcp_bridge.offered', {}), \
             mock.patch('peer.dds_state.push_errors', {}):
            resp = asyncio.run(peer_api.list_paired())
        self.assertTrue(resp['peers'][0]['mutual'])
        self.assertIsNotNone(resp['peers'][0]['mutual_at'])


class TestSuccessfulPushMarksMutual(unittest.IsolatedAsyncioTestCase):
    """出向推送成功也是证据：对端接受了我们签名的请求。"""

    def setUp(self):
        self._db = os.path.join(tempfile.mkdtemp(), 'peers.db')
        import config
        self._patch = mock.patch.object(config, 'DB_PATH', self._db)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        identity.reset_cache()
        identity.ensure_identity()
        store.upsert('b' * 32, identity.public_key_b64(), 'Remote', role='viewer',
                     endpoints=['https://10.0.0.9:15678'])

    async def test_a_successful_push_marks_mutual(self):
        from peer import dds_state

        async def _post(eps, path, payload, **kw):
            return {'accepted': 1}, ''

        fake_bridge = mock.Mock(get_dds_topics=lambda: {'/x'})
        with mock.patch.dict(sys.modules, {'ros2_bridge': fake_bridge}), \
             mock.patch('peer.store.list_peers', return_value=[store.get('b' * 32)]), \
             mock.patch('peer.registry.registry.endpoints_for',
                        return_value=['https://10.0.0.9:15678']), \
             mock.patch('peer.transport.post_json', side_effect=_post):
            await dds_state.push_once()

        p = store.get('b' * 32)
        self.assertIsNotNone(p['mutual_at'], '推送成功就是对端接受了我们的签名，证据成立')

    async def test_a_failed_push_does_not_mark_mutual(self):
        from peer import dds_state

        async def _post(eps, path, payload, **kw):
            return None, 'HTTP 403 not paired'

        fake_bridge = mock.Mock(get_dds_topics=lambda: {'/x'})
        with mock.patch.dict(sys.modules, {'ros2_bridge': fake_bridge}), \
             mock.patch('peer.store.list_peers', return_value=[store.get('b' * 32)]), \
             mock.patch('peer.registry.registry.endpoints_for',
                        return_value=['https://10.0.0.9:15678']), \
             mock.patch('peer.transport.post_json', side_effect=_post):
            await dds_state.push_once()

        p = store.get('b' * 32)
        self.assertIsNone(p['mutual_at'], '403 说明对端根本没有我们，不该标记')


if __name__ == '__main__':
    unittest.main()
