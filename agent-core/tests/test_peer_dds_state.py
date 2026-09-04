"""
test_peer_dds_state.py — P3 状态共享（本机采集 + 签名 HTTPS 推送）。

这个模块原来跑一整套 rclpy 机制：自己的 node、publisher、对每个 peer 的订阅、
常驻 executor，以及"context 是不是我自己建的"这套所有权判断。旧测试守的就是
那些东西 —— 特别是一次真实崩溃：无条件 rclpy.init() 撞上 ros2_bridge 已经初始化
过的进程级 context，日志先打印 "started"、线程随即死掉、容器照常运行、功能静默
失效。

DDS 被锁到本机之后，跨机订阅永远收不到东西，那套机制连同它的测试一起删掉了。
这里测的是接替它的东西，其中两条不变量是从旧代码继承下来的教训：一轮失败不能
让循环结束；以及可用性判断不能只看 rclpy 能不能 import。
"""

import asyncio
import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from peer import dds_state  # noqa: E402


class TestLocalTopics(unittest.TestCase):
    def setUp(self):
        dds_state._peer_topics.clear()

    def test_unavailable_without_a_running_bridge(self):
        """rclpy 装着但 bridge 没起来时，必须报 unavailable。

        旧版问的是 rclpy 能不能 import + 有没有 ROS_DOMAIN_ID —— 这两者在
        bridge 没起来的容器里都成立，于是拓扑图永远是空的，却报告"可用"。
        """
        with mock.patch.dict(sys.modules, {'ros2_bridge': None}):
            self.assertFalse(dds_state.is_available())

    def test_local_topics_sorted_and_degrades_to_empty(self):
        fake = mock.Mock(get_dds_topics=lambda: {'/b', '/a'})
        with mock.patch.dict(sys.modules, {'ros2_bridge': fake}):
            self.assertEqual(dds_state.local_topics(), ['/a', '/b'])
        broken = mock.Mock(get_dds_topics=mock.Mock(side_effect=RuntimeError('no bus')))
        with mock.patch.dict(sys.modules, {'ros2_bridge': broken}):
            self.assertEqual(dds_state.local_topics(), [])


class TestPeerTopicStore(unittest.TestCase):
    def setUp(self):
        dds_state._peer_topics.clear()

    def test_records_and_returns(self):
        dds_state.record_peer_topics('abc', ['/x', '/y'])
        self.assertEqual(dds_state.get_peer_topics()['abc']['topics'], ['/x', '/y'])

    def test_prunes_stale_entries(self):
        """不再推送的 peer 应当自己从拓扑图里淡出。"""
        dds_state.record_peer_topics('gone', ['/x'])
        dds_state._peer_topics['gone']['last_seen'] = time.time() - dds_state.STALE_AFTER_S - 1
        dds_state.record_peer_topics('here', ['/y'])
        self.assertEqual(list(dds_state.get_peer_topics()), ['here'])


class TestPush(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        dds_state._peer_topics.clear()

    async def _push(self, *, peers, endpoints, topics=('/perception/tts',)):
        calls = []

        async def _post(eps, path, payload, **kw):
            calls.append({'endpoints': eps, 'path': path, 'payload': payload})
            return {'accepted': 1}, ''

        fake_bridge = mock.Mock(get_dds_topics=lambda: set(topics))
        with mock.patch.dict(sys.modules, {'ros2_bridge': fake_bridge}), \
             mock.patch('peer.store.list_peers', return_value=peers), \
             mock.patch('peer.registry.registry.endpoints_for',
                        side_effect=lambda pid: endpoints.get(pid, [])), \
             mock.patch('peer.transport.post_json', side_effect=_post):
            delivered = await dds_state.push_once()
        return delivered, calls

    async def test_pushes_to_each_paired_peer_over_the_signed_path(self):
        delivered, calls = await self._push(
            peers=[{'peer_id': 'a'}, {'peer_id': 'b'}],
            endpoints={'a': ['https://1'], 'b': ['https://2']},
        )
        self.assertEqual(delivered, 2)
        self.assertEqual({c['path'] for c in calls}, {dds_state.STATE_PATH})
        self.assertEqual(calls[0]['payload']['topics'], ['/perception/tts'])

    async def test_unreachable_peer_is_skipped_not_fatal(self):
        """关机的机器人是常态，不该让这一轮推送中断。"""
        delivered, calls = await self._push(
            peers=[{'peer_id': 'a'}, {'peer_id': 'off'}],
            endpoints={'a': ['https://1']},   # 'off' 没有任何可达地址
        )
        self.assertEqual(delivered, 1)
        self.assertEqual(len(calls), 1)

    async def test_nothing_is_pushed_when_there_are_no_local_topics(self):
        delivered, calls = await self._push(
            peers=[{'peer_id': 'a'}], endpoints={'a': ['https://1']}, topics=(),
        )
        self.assertEqual((delivered, calls), (0, []))

    async def test_a_failed_round_does_not_end_the_loop(self):
        """旧 DDS 版本让异常逃出去，共享就此静默死到进程结束。"""
        rounds = []

        async def boom():
            rounds.append(1)
            if len(rounds) == 1:
                raise RuntimeError('transient')

        with mock.patch.object(dds_state, 'push_once', side_effect=boom), \
             mock.patch.object(dds_state, 'PUSH_INTERVAL_S', 0.01):
            task = asyncio.create_task(dds_state._push_loop())
            await asyncio.sleep(0.05)
            task.cancel()
        self.assertGreater(len(rounds), 1, '第一轮失败后没有继续')


class TestOneSidedPairingIsVisible(unittest.IsolatedAsyncioTestCase):
    """单边配对必须看得出来。

    配对是按方向各存一份记录的：只在一台机器上确认，那台就会把对方列为"已配对"，
    而对端没有本机记录、会把每个签名请求 403 掉 —— 而这一侧的界面看起来完全正常。
    状态推送每 5s 一轮，它的失败原因是回答"对方到底批准了没有"最便宜的信号。
    """

    def setUp(self):
        dds_state.push_errors.clear()

    async def _push(self, resp, reason):
        async def _post(eps, path, payload, **kw):
            return resp, reason
        fake_bridge = mock.Mock(get_dds_topics=lambda: {'/x'})
        with mock.patch.dict(sys.modules, {'ros2_bridge': fake_bridge}), \
             mock.patch('peer.store.list_peers', return_value=[{'peer_id': 'p1'}]), \
             mock.patch('peer.registry.registry.endpoints_for', return_value=['https://x']), \
             mock.patch('peer.transport.post_json', side_effect=_post):
            return await dds_state.push_once()

    async def test_a_403_is_remembered(self):
        await self._push(None, 'https://x → HTTP 403 {"detail":"not paired"}')
        self.assertIn('403', dds_state.push_errors['p1'])

    async def test_success_clears_a_previous_failure(self):
        """对方批准之后，这个提示必须自己消失。"""
        await self._push(None, 'HTTP 403 nope')
        await self._push({'accepted': 1}, '')
        self.assertNotIn('p1', dds_state.push_errors)

    async def test_stop_clears_it(self):
        await self._push(None, 'HTTP 403 nope')
        dds_state.stop()
        self.assertEqual(dds_state.push_errors, {})


class TestPeerFacingPath(unittest.TestCase):
    def test_state_endpoint_is_exempt_from_access_token(self):
        """漏登记就是 401，而且日志看起来一切正常 —— 这个坑踩过一次。"""
        from peer import transport
        self.assertTrue(transport.is_peer_facing(dds_state.STATE_PATH))


if __name__ == '__main__':
    unittest.main()
