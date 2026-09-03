"""
test_peer_registry.py — 发现层与 Registry 归并。

验证节点（按计划）：
6. 多路发现归并：同一个 peer_id 从 mDNS + 静态清单发现，Registry 输出一条记录、两条链路
"""

import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

from peer.discovery.base import PeerAdvert  # noqa: E402
from peer.registry import PeerRegistry  # noqa: E402


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = PeerRegistry()
        self.reg.reset()

    def test_merge_multiple_sources(self):
        """同一个 peer 从两个 provider 发现时，归并为一条记录、多条链路（节点 6）"""
        peer_id = 'test_peer_123'

        # 从 mDNS 看到
        adv1 = PeerAdvert(
            peer_id=peer_id,
            display_name='Robot A',
            endpoints=['https://192.168.1.10:15678'],
            source='mdns',
            last_seen=time.time(),
        )
        self.reg.observe(adv1)

        # 从静态清单看到，同一个 peer_id，不同链路
        adv2 = PeerAdvert(
            peer_id=peer_id,
            display_name='Robot A',
            endpoints=['https://10.0.0.5:15678'],
            source='static',
            last_seen=time.time(),
        )
        self.reg.observe(adv2)

        discovered = self.reg.discovered(include_paired=True)
        self.assertEqual(len(discovered), 1)
        item = discovered[0]
        self.assertEqual(item['peer_id'], peer_id)
        # 两条链路
        self.assertIn('https://192.168.1.10:15678', item['endpoints'])
        self.assertIn('https://10.0.0.5:15678', item['endpoints'])
        # 两个来源
        sources = item['sources']
        self.assertIn('mdns', sources)
        self.assertIn('static', sources)

    def test_prune_stale(self):
        """过期记录被清理"""
        adv = PeerAdvert(
            peer_id='stale_peer',
            endpoints=['https://old'],
            source='mdns',
            last_seen=time.time() - 400,
        )
        self.reg.observe(adv)
        self.reg.prune()
        discovered = self.reg.discovered(include_paired=True)
        self.assertEqual(len(discovered), 0)

    def test_ignore_self(self):
        """不把自己列为 peer"""
        from peer import identity
        identity.reset_cache()
        my_id = identity.peer_id()
        adv = PeerAdvert(
            peer_id=my_id,
            endpoints=['https://self'],
            source='mdns',
            last_seen=time.time(),
        )
        self.reg.observe(adv)
        discovered = self.reg.discovered(include_paired=True)
        self.assertEqual(len(discovered), 0)



class TestDiscoveryFreshness(unittest.TestCase):
    """发现列表会不会自己饿死。

    真机上撞到的：两台机器互相都发现不到，但 provider 报 running=true。
    registry 会清掉 STALE_AFTER_S(300s) 内没更新的 advert，而 mDNS 对一个稳定
    服务的再通告周期远长于此（PTR 记录 TTL 默认数千秒）。于是**任何部署跑满
    5 分钟后发现列表都会清空**，跟配置无关。

    之前每次双机验证都在重启后 5 分钟内做完，所以从没暴露。
    """

    def test_refresh_interval_beats_staleness(self):
        from peer.discovery.mdns import REFRESH_INTERVAL_S
        from peer.registry import STALE_AFTER_S
        self.assertLess(
            REFRESH_INTERVAL_S, STALE_AFTER_S / 2,
            'mDNS refresh must run well inside the staleness window, or healthy '
            'peers age out of the live view and never return')

    def test_removed_is_handled(self):
        """收到 goodbye 要立刻忘掉，而不是等超时。"""
        import inspect
        from peer.discovery.mdns import MdnsProvider
        src = inspect.getsource(MdnsProvider._on_change)
        self.assertIn('_seen_names.discard', src,
                      'Removed must drop the service, not just return')

    def test_refresh_loop_stops_with_provider(self):
        """刷新任务必须随 stop() 取消，否则热重启会留下孤儿任务。"""
        import inspect
        from peer.discovery.mdns import MdnsProvider
        src = inspect.getsource(MdnsProvider.stop)
        self.assertIn('_refresh_task', src)
        self.assertIn('cancel', src)

if __name__ == '__main__':
    unittest.main()
