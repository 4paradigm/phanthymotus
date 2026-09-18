"""
test_peer_unpair_notify.py — 解除配对必须告诉对方，而且只能由对方解除对方自己。

背景（天轶，2026-09-11）：有人在天轶面板上点了「解除配对」。天轶删掉了自己那行，
Orin5 完全不知情 —— 它继续以为配着对，每 5 秒推一次状态，连续 7 天被 403
`unknown_peer` 拒绝 15209 次。两边的表就这样永久不一致，只能靠人去读数据库才发现。
`dds_state.py` 开头的注释早就描述过这个失败模式，但当时的应对只是把原因记进
`push_errors` 给面板看，没有任何东西去修复不一致本身。

这里覆盖三件事：

1. **通知在删除之前发**。要到达对方得先拿 endpoints，而 endpoints 来自那行本身。
2. **对方离线不能挡住删除**。操作员点了就得生效；响应里说清楚有没有通知到，
   界面才能诚实地讲，而不是暗示两边都干净了。
3. **一个 peer 只能删掉它自己**。`peer_id` 取自验过的签名而非请求体 —— 否则这个
   端点就成了「任意 peer 拆掉任意配对」。

Run: cd agent-core && python3 -m pytest tests/test_peer_unpair_notify.py
"""

import asyncio
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from peer import identity, store, transport  # noqa: E402
import api.peer as peer_api  # noqa: E402

PEER = 'a' * 32
UNPAIR_PATH = '/api/peer/inbox/unpair'


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _Base(unittest.TestCase):
    def setUp(self):
        # Same reason as test_peer_mutual: config.DB_PATH is read at import time,
        # so patch the module attribute, not os.environ.
        import config
        self._db = os.path.join(tempfile.mkdtemp(), 'peers.db')
        p = mock.patch.object(config, 'DB_PATH', self._db)
        p.start()
        self.addCleanup(p.stop)
        identity.reset_cache()
        identity.ensure_identity()
        from peer import backoff
        backoff.clear()

    def _pair(self, endpoints=('https://10.0.0.5:15678',)):
        return store.upsert(PEER, identity.public_key_b64(), 'Far',
                            role='viewer', endpoints=list(endpoints))


class TestPathIsPeerFacing(_Base):
    def test_the_new_path_authenticates_by_signature_not_access_token(self):
        """漏掉这一条，auth.py 会拿 ACCESS_TOKEN 去拦它，peer 请求全变 401。

        这个坑仓库里踩过一次 —— tool-proxy 和 delegation 的端点加在 '/inbox/'
        前缀之外，端点本身完全正确却每个请求都 401。PEER_FACING_PATHS 的注释
        写明了它是唯一事实来源。
        """
        self.assertIn(UNPAIR_PATH, transport.PEER_FACING_PATHS)
        self.assertTrue(transport.is_peer_facing(UNPAIR_PATH))


class TestOutboundUnpair(_Base):
    """DELETE /api/peer/paired/{id} —— 本机发起的解除。"""

    def _call(self, post_json):
        with mock.patch.object(transport, 'post_json', post_json):
            return _run(peer_api.unpair(PEER))

    def test_notifies_the_peer_then_deletes(self):
        seen = {}

        async def fake_post(endpoints, path, payload, **kw):
            # 关键断言：调用发生时那行还在，否则 endpoints 根本取不到。
            seen['row_present'] = store.get(PEER) is not None
            seen['endpoints'] = list(endpoints)
            seen['path'] = path
            return {}, ''

        self._pair()
        res = self._call(fake_post)

        self.assertTrue(seen['row_present'], '通知必须在删除之前发出')
        self.assertEqual(seen['path'], UNPAIR_PATH)
        self.assertEqual(seen['endpoints'], ['https://10.0.0.5:15678'])
        self.assertTrue(res['deleted'])
        self.assertTrue(res['notified'])
        self.assertIsNone(store.get(PEER), '本地那行应该已经删掉')

    def test_an_offline_peer_does_not_block_the_deletion(self):
        """R1 现在就是这个状态：关着机，通知发不出去。点了就得生效。"""
        async def fake_post(endpoints, path, payload, **kw):
            return None, 'https://10.0.0.5:15678 → ClientConnectorError: Cannot connect'

        self._pair()
        res = self._call(fake_post)

        self.assertTrue(res['deleted'])
        self.assertFalse(res['notified'])
        self.assertIn('ClientConnectorError', res['notify_error'])
        self.assertIsNone(store.get(PEER))

    def test_no_known_endpoint_is_reported_not_swallowed(self):
        async def fake_post(endpoints, path, payload, **kw):
            raise AssertionError('不该在没有 endpoint 时还去发请求')

        self._pair(endpoints=())
        with mock.patch.object(peer_api.registry, 'endpoints_for', lambda _pid: []):
            res = self._call(fake_post)

        self.assertTrue(res['deleted'])
        self.assertFalse(res['notified'])
        self.assertEqual(res['notify_error'], 'no_known_endpoint')

    def test_unknown_peer_is_404(self):
        import fastapi

        async def fake_post(*a, **kw):
            raise AssertionError('不该给一个不存在的 peer 发通知')

        with self.assertRaises(fastapi.HTTPException) as ctx:
            self._call(fake_post)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_backoff_entry_is_cleared_on_unpair(self):
        """别让一条已经删掉的记录在退避表里继续占位。"""
        from peer import backoff
        backoff.note_result(PEER, False, 'https://x → HTTP 403 unknown_peer')
        self.assertTrue(backoff.should_skip(PEER))

        async def fake_post(*a, **kw):
            return {}, ''

        self._pair()
        self._call(fake_post)
        self.assertFalse(backoff.should_skip(PEER))


class _Req:
    """最小的 Request 替身：verify_signed_request 在测试里是被打桩的。"""

    def __init__(self, path=UNPAIR_PATH):
        self.method = 'POST'
        self.headers = {}
        self.url = type('U', (), {'path': path})()

    async def body(self):
        return b''


class TestInboundUnpair(_Base):
    """POST /api/peer/inbox/unpair —— 对方发起的解除。"""

    def test_a_verified_peer_removes_itself(self):
        self._pair()
        with mock.patch.object(transport, 'verify_signed_request',
                               return_value=(PEER, '')):
            res = _run(peer_api.inbox_unpair(_Req()))
        self.assertTrue(res['removed'])
        self.assertIsNone(store.get(PEER))

    def test_a_bad_signature_changes_nothing(self):
        import fastapi
        self._pair()
        with mock.patch.object(transport, 'verify_signed_request',
                               return_value=('', 'bad_signature')):
            with self.assertRaises(fastapi.HTTPException) as ctx:
                _run(peer_api.inbox_unpair(_Req()))
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIsNotNone(store.get(PEER), '验签失败绝不能删掉任何东西')

    def test_a_peer_cannot_unpair_a_third_party(self):
        """身份只来自签名。请求体里写别人的 id 不该有任何效果 —— 否则这个端点
        就成了「任意 peer 拆掉任意配对」。"""
        other = 'b' * 32
        self._pair()
        store.upsert(other, identity.public_key_b64(), 'Third', role='viewer')

        class _EvilReq(_Req):
            async def body(self):
                import json
                return json.dumps({'peer_id': other}).encode()

        with mock.patch.object(transport, 'verify_signed_request',
                               return_value=(PEER, '')):
            _run(peer_api.inbox_unpair(_EvilReq()))

        self.assertIsNone(store.get(PEER), '签名者自己应该被删掉')
        self.assertIsNotNone(store.get(other), '第三方必须原封不动')

    def test_removing_an_already_absent_peer_is_not_an_error(self):
        """重发一次通知（或两边同时点）不该 500。"""
        with mock.patch.object(transport, 'verify_signed_request',
                               return_value=(PEER, '')):
            res = _run(peer_api.inbox_unpair(_Req()))
        self.assertFalse(res['removed'])


if __name__ == '__main__':
    unittest.main()
