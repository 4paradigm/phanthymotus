"""
test_peer_name_sync.py — 改名必须在 5 秒内同步到所有配对的机器人。

原来只在配对时交换一次名字，之后永远不更新 —— 除非解除配对重新配对。现在状态推送
（5 秒一轮）带上 display_name，接收方更新 peers 表，所以改名后 5 秒内所有机器人都能
看到新名字。
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


class TestDisplayNameSync(unittest.TestCase):
    def setUp(self):
        self._db = os.path.join(tempfile.mkdtemp(), 'peers.db')
        import config
        self._patch = mock.patch.object(config, 'DB_PATH', self._db)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _peer(self, name='OldName'):
        identity.reset_cache()
        identity.ensure_identity()
        return store.upsert('a' * 32, identity.public_key_b64(), name,
                            role='viewer', endpoints=['https://10.0.0.5:15678'])

    def test_update_display_name_changes_the_stored_name(self):
        self._peer('OldName')
        store.update_display_name('a' * 32, 'NewName')
        self.assertEqual(store.get('a' * 32)['display_name'], 'NewName')

    def test_empty_or_invalid_names_are_ignored(self):
        """不该因为对端发了个空字符串就把本地存的名字抹掉。"""
        self._peer('Valid')
        store.update_display_name('a' * 32, '')
        self.assertEqual(store.get('a' * 32)['display_name'], 'Valid')
        store.update_display_name('a' * 32, None)
        self.assertEqual(store.get('a' * 32)['display_name'], 'Valid')

    def test_name_is_trimmed_and_capped(self):
        self._peer()
        store.update_display_name('a' * 32, '  Padded  ')
        self.assertEqual(store.get('a' * 32)['display_name'], 'Padded')
        store.update_display_name('a' * 32, 'x' * 300)
        self.assertEqual(len(store.get('a' * 32)['display_name']), 200)


class TestStatePushCarriesDisplayName(unittest.IsolatedAsyncioTestCase):
    """出向推送必须带上本机的 display_name。"""

    async def test_the_payload_includes_our_name(self):
        from peer import dds_state
        import config

        config.main = {'peer_settings': {'display_name': 'TestBot'}}
        captured = []

        async def _post(eps, path, payload, **kw):
            captured.append(payload)
            return {'accepted': 1}, ''

        fake_bridge = mock.Mock(get_dds_topics=lambda: {'/test'})
        db = os.path.join(tempfile.mkdtemp(), 'peers.db')
        with mock.patch.object(config, 'DB_PATH', db), \
             mock.patch.dict(sys.modules, {'ros2_bridge': fake_bridge}), \
             mock.patch('peer.store.list_peers', return_value=[
                 {'peer_id': 'b' * 32, 'endpoints': ['https://x:15678']}]), \
             mock.patch('peer.registry.registry.endpoints_for',
                        return_value=['https://x:15678']), \
             mock.patch('peer.transport.post_json', side_effect=_post):
            await dds_state.push_once()

        self.assertEqual(len(captured), 1)
        self.assertIn('display_name', captured[0])
        self.assertEqual(captured[0]['display_name'], 'TestBot')


class TestInboundStateUpdatesDisplayName(unittest.IsolatedAsyncioTestCase):
    """入向状态推送收到新名字后必须更新 peers 表。"""

    def setUp(self):
        self._db = os.path.join(tempfile.mkdtemp(), 'peers.db')
        import config
        self._patch = mock.patch.object(config, 'DB_PATH', self._db)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        identity.reset_cache()
        identity.ensure_identity()
        store.upsert('c' * 32, identity.public_key_b64(), 'OldName',
                     role='viewer', endpoints=['https://10.0.0.9:15678'])

    async def test_a_new_name_is_written_to_the_peers_table(self):
        from api import peer as peer_api
        from unittest.mock import AsyncMock

        req = mock.Mock()
        req.method = 'POST'
        req.url.path = '/api/peer/inbox/state'
        req.headers = {}
        req.body = AsyncMock(return_value=b'{}')
        req.json = AsyncMock(return_value={
            'topics': ['/x'],
            'agent_running': False,
            'display_name': 'RenamedBot',
        })
        req.client = mock.Mock(host='10.0.0.9')

        with mock.patch('peer.transport.verify_signed_request',
                        return_value=('c' * 32, None)):
            await peer_api.inbox_state(req)

        self.assertEqual(store.get('c' * 32)['display_name'], 'RenamedBot')


if __name__ == '__main__':
    unittest.main()
