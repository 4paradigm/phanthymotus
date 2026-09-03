"""
test_dds_isolation.py — 隔离自检本身要靠得住。

存在的理由：DDS 隔离失效是**无症状**的。profile 文件没挂载、路径写错、XML 语法
错误，FastDDS 都会静默回退到所有网卡，机器人照常工作，直到某天另一台机器人替它
回答了指令才暴露。

这类"失败了但看起来正常"的模式在这套代码里已经出现过两次（rclpy 线程静默死亡、
画布门只写在注释里没实现），所以自检必须能真的分辨出隔离有没有生效——
一个永远报 OK 的自检比没有自检更糟。
"""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import dds_isolation  # noqa: E402


class TestDdsIsolation(unittest.TestCase):
    def test_unconfigured_is_reported_not_silently_ok(self):
        """没配 profile 时必须明确报未隔离，不能默认 OK。"""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('FASTRTPS_DEFAULT_PROFILES_FILE', None)
            r = dds_isolation.check_and_report()
        self.assertFalse(r['isolated'])
        self.assertFalse(r['profile_configured'])
        self.assertIn('未配置', r['detail'])

    def test_missing_file_is_caught(self):
        """配了但文件不存在——这正是"忘了挂载"的表现，必须抓住。"""
        with mock.patch.dict(os.environ,
                             {'FASTRTPS_DEFAULT_PROFILES_FILE': '/nonexistent/dds.xml'}):
            r = dds_isolation.check_and_report()
        self.assertFalse(r['isolated'])
        self.assertTrue(r['profile_configured'])
        self.assertFalse(r['profile_readable'])
        self.assertIn('不可读', r['detail'])

    def test_external_bind_defeats_the_check(self):
        """文件在、但仍绑外部地址时不能报 OK（XML 语法错会被静默忽略）。"""
        with tempfile.NamedTemporaryFile(suffix='.xml') as f:
            with mock.patch.dict(os.environ,
                                 {'FASTRTPS_DEFAULT_PROFILES_FILE': f.name}), \
                 mock.patch.object(dds_isolation, '_external_dds_binds',
                                   lambda: ['10.100.121.14:53642']):
                r = dds_isolation.check_and_report()
        self.assertFalse(r['isolated'])
        self.assertIn('10.100.121.14:53642', r['external_binds'])

    def test_isolated_when_only_loopback(self):
        """文件可读且无外部绑定 → 判定隔离生效。"""
        with tempfile.NamedTemporaryFile(suffix='.xml') as f:
            with mock.patch.dict(os.environ,
                                 {'FASTRTPS_DEFAULT_PROFILES_FILE': f.name}), \
                 mock.patch.object(dds_isolation, '_external_dds_binds', lambda: []):
                r = dds_isolation.check_and_report()
        self.assertTrue(r['isolated'])

    def test_multicast_group_is_not_counted_as_external(self):
        """多播组地址不能算外部绑定。

        加入哪个网卡由白名单决定，不能凭组地址判断；把 239.255.0.1 当成外部绑定
        会让自检在隔离**正常**时误报，那样它很快就会被无视。
        """
        fake_udp = (
            '  sl  local_address rem_address   st tx_queue rx_queue tr tm->when '
            'retrnsmt   uid  timeout inode\n'
            # 239.255.0.1:17900 —— 小端：0100FFEF
            '   1: 0100FFEF:45EC 00000000:0000 07 00000000:00000000 00:00000000 '
            '00000000  1000        0 12345\n'
            # 127.0.0.1:17900 —— 小端：0100007F
            '   2: 0100007F:45EC 00000000:0000 07 00000000:00000000 00:00000000 '
            '00000000  1000        0 12346\n'
        )
        m = mock.mock_open(read_data=fake_udp)
        with mock.patch.object(dds_isolation, 'glob') as g, \
             mock.patch('builtins.open', m), \
             mock.patch('os.readlink', side_effect=lambda p: 'socket:[12345]'):
            g.glob.return_value = ['/proc/1/fd/3']
            self.assertEqual(dds_isolation._external_dds_binds(), [])

    def test_mdns_is_not_mistaken_for_a_dds_leak(self):
        """mDNS 的 5353 不能算 DDS 泄漏。

        真机上撞到的：启动日志报"已隔离"、API 报"未隔离"，两者矛盾。根因是
        自检统计了进程**全部**非 loopback UDP socket，把 mDNS 的
        10.100.121.14:5353 当成了 DDS 绑到外网。而 mDNS 本来就必须走网络 ——
        peer 发现依赖它。会误报的自检很快会被无视，那还不如没有。
        """
        # domain 42 → DDS 端口段 17900-18149
        fake_udp = (
            '  sl  local_address rem_address st tx rx tr tm retr uid to inode\n'
            # 10.100.121.14:5353 (mDNS) —— 小端 IP 0E79640A，端口 14E9
            '   1: 0E79640A:14E9 00000000:0000 07 0:0 0:0 0 1000 0 111\n'
            # 10.100.121.14:17911 (DDS 用户单播) —— 端口 45F7，这个才该报
            '   2: 0E79640A:45F7 00000000:0000 07 0:0 0:0 0 1000 0 112\n'
        )
        with mock.patch.dict(os.environ, {'ROS_DOMAIN_ID': '42'}), \
             mock.patch.object(dds_isolation, 'glob') as g, \
             mock.patch('builtins.open', mock.mock_open(read_data=fake_udp)), \
             mock.patch('os.readlink', side_effect=lambda p: 'socket:[111]'
                        if p.endswith('3') else 'socket:[112]'):
            g.glob.return_value = ['/proc/1/fd/3', '/proc/1/fd/4']
            binds = dds_isolation._external_dds_binds()

        self.assertNotIn('10.100.121.14:5353', binds,
                         'mDNS was counted as a DDS leak — the false positive that '
                         'made the check contradict the startup log')
        self.assertIn('10.100.121.14:17911', binds,
                      'a real DDS socket on an external address must still be reported')

    def test_port_range_follows_the_domain(self):
        """端口段随 ROS_DOMAIN_ID 走，且是规范里的 250 个一块。"""
        with mock.patch.dict(os.environ, {'ROS_DOMAIN_ID': '42'}):
            self.assertEqual(dds_isolation._dds_port_range(), (17900, 18149))
        with mock.patch.dict(os.environ, {'ROS_DOMAIN_ID': '0'}):
            self.assertEqual(dds_isolation._dds_port_range(), (7400, 7649))

    def test_endpoint_exists(self):
        """/api/peer/dds_isolation 必须真的注册了，且属于 dashboard 侧。"""
        import api.peer
        from peer.transport import is_peer_facing
        paths = {'/api' + r.path for r in api.peer.router.routes}
        self.assertIn('/api/peer/dds_isolation', paths)
        self.assertFalse(is_peer_facing('/api/peer/dds_isolation'),
                         'operator-facing endpoint must still require ACCESS_TOKEN')


if __name__ == '__main__':
    unittest.main()
