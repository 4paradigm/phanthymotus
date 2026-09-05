"""
test_acp_complete_origin.py — /api/acp/complete 只接受本机。

这个端点原来完全免认证，而服务监听在 0.0.0.0。清掉一条 barrier 只需要知道 action_id，
而清掉它意味着 agent 认为动作已经完成 —— 局域网上任何人都可以告诉机器人"你的话播完了"
或"你已经导航到位了"。

Driver / perception 容器跑 host network，走 `https://localhost:15678`（AGENT_CORE_URL
的默认值），所以合法调用方全是 loopback。Peer 要送完成信号必须走签名路径。
"""
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import auth  # noqa: E402


def _req(host, path='/api/acp/complete', method='POST'):
    return types.SimpleNamespace(
        url=types.SimpleNamespace(path=path),
        method=method,
        client=types.SimpleNamespace(host=host) if host is not None else None,
    )


class TestLoopbackDetection(unittest.TestCase):
    def test_ipv4_loopback(self):
        self.assertTrue(auth._from_loopback(_req('127.0.0.1')))

    def test_ipv6_loopback(self):
        self.assertTrue(auth._from_loopback(_req('::1')))

    def test_ipv4_mapped_ipv6_loopback(self):
        """uvicorn 在双栈监听时会报这个形式。"""
        self.assertTrue(auth._from_loopback(_req('::ffff:127.0.0.1')))

    def test_office_lan_is_not_loopback(self):
        self.assertFalse(auth._from_loopback(_req('10.100.121.14')))

    def test_another_robot_is_not_loopback(self):
        self.assertFalse(auth._from_loopback(_req('10.100.121.16')))

    def test_missing_client_is_not_trusted(self):
        """拿不到来源时按不可信处理 —— 失败要往安全那边倒。"""
        self.assertFalse(auth._from_loopback(_req(None)))

    def test_almost_loopback_is_rejected(self):
        """127.0.0.1 的前缀相似地址不能混进来。"""
        for host in ('127.0.0.10', '1127.0.0.1', '127.0.0.1.evil.com', ''):
            self.assertFalse(auth._from_loopback(_req(host)), host)


class TestPeerCallbackHasNoUnusedExemption(unittest.TestCase):
    """签名回调端点还没有实现，就不该出现在免认证清单里。

    tests/test_peer_api_imports.py 已经在守这条（免认证路径必须有路由对应）。这里从反面
    再钉一次：一个"能置位 pending"的端点如果没有调用方，就是白送的攻击面。等 peer 工具
    调用改成异步、真的需要回调时再加。
    """

    def test_acp_complete_is_not_peer_facing_yet(self):
        from peer.transport import PEER_FACING_PATHS
        self.assertNotIn('/api/peer/inbox/acp_complete', PEER_FACING_PATHS)


if __name__ == '__main__':
    unittest.main()
