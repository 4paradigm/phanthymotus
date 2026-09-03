"""
test_peer_dds_state.py — P3 DDS 状态共享。

存在的原因：一次真实的运行时崩溃。dds_state 的后台线程无条件调用
rclpy.init()，而 ros2_bridge.py 启动时已经初始化过同一个进程级 context：

    RuntimeError: Context.init() must only be called once

表现很隐蔽 —— 日志先打印 "[peer] DDS state sharing started"，线程随即死掉，
容器照常运行，功能静默失效。

这里用假的 rclpy 覆盖两条路径：context 已存在时必须复用，不存在时才自己
初始化；并且只有自己初始化过才允许 shutdown（否则会把 ros2_bridge 的总线和
所有 /ws/bus 订阅一起拆掉）。
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from peer import dds_state  # noqa: E402


class _FakeNode:
    def __init__(self, name):
        self.name = name
        self.destroyed = False

    def create_publisher(self, *a, **k):
        return mock.MagicMock()

    def create_subscription(self, *a, **k):
        return mock.MagicMock()

    def get_topic_names_and_types(self):
        return [('/robot/status', ['std_msgs/msg/String'])]

    def destroy_node(self):
        self.destroyed = True


def _fake_rclpy(ok: bool):
    """A stand-in rclpy whose context is already initialised (ok=True) or not."""
    m = types.SimpleNamespace()
    m.state = {'init_calls': 0, 'shutdown_calls': 0, 'ok': ok}
    m.ok = lambda: m.state['ok']

    def _init():
        if m.state['ok']:
            raise RuntimeError('Context.init() must only be called once')
        m.state['init_calls'] += 1
        m.state['ok'] = True

    m.init = _init
    m.shutdown = lambda: m.state.__setitem__('shutdown_calls',
                                             m.state['shutdown_calls'] + 1)
    m.spin_once = lambda node, timeout_sec=None: None
    return m


class TestDdsState(unittest.TestCase):
    def setUp(self):
        dds_state._running = False
        dds_state._owns_rclpy = False
        dds_state._node = None
        dds_state._publisher = None
        dds_state._subscribers.clear()
        dds_state._peer_topics.clear()

    def _run_once(self, fake):
        """Run _run_loop for a single iteration with a fake rclpy."""
        node_mod = types.SimpleNamespace(Node=_FakeNode)
        msg_mod = types.SimpleNamespace(String=lambda: types.SimpleNamespace(data=''))
        modules = {
            'rclpy': fake,
            'rclpy.node': node_mod,
            'std_msgs': types.SimpleNamespace(msg=msg_mod),
            'std_msgs.msg': msg_mod,
        }
        with mock.patch.dict(sys.modules, modules):
            dds_state._running = True

            # Stop after the first spin so the loop terminates.
            def _spin(node, timeout_sec=None):
                dds_state._running = False

            fake.spin_once = _spin
            with mock.patch.object(dds_state, '_subscribe_to_peers', lambda: None):
                dds_state._run_loop()

    def test_reuses_existing_context(self):
        """ros2_bridge 已初始化时，绝不能再调用 rclpy.init()。"""
        fake = _fake_rclpy(ok=True)
        self._run_once(fake)
        self.assertEqual(fake.state['init_calls'], 0,
                         'called rclpy.init() on an already-initialised context')

    def test_does_not_shutdown_borrowed_context(self):
        """借用别人的 context 时不得 shutdown —— 那会拆掉整条 DDS 总线。"""
        fake = _fake_rclpy(ok=True)
        self._run_once(fake)
        self.assertEqual(fake.state['shutdown_calls'], 0,
                         'shut down a context owned by ros2_bridge')

    def test_initialises_when_no_context(self):
        """没有 bridge 时（rclpy 可用但未初始化）自己初始化。"""
        fake = _fake_rclpy(ok=False)
        self._run_once(fake)
        self.assertEqual(fake.state['init_calls'], 1)

    def test_shuts_down_own_context(self):
        """自己初始化的 context 才由自己关闭。"""
        fake = _fake_rclpy(ok=False)
        self._run_once(fake)
        self.assertEqual(fake.state['shutdown_calls'], 1)

    def test_is_available_without_rclpy(self):
        """rclpy 缺失时 is_available() 返回 False 而不是抛异常。"""
        self.assertIsInstance(dds_state.is_available(), bool)

    def test_get_peer_topics_prunes_stale(self):
        """超过 60s 的 peer 记录被清理。"""
        import time
        dds_state._peer_topics['fresh'] = {'topics': ['/a'], 'last_seen': time.time()}
        dds_state._peer_topics['stale'] = {'topics': ['/b'], 'last_seen': time.time() - 120}
        result = dds_state.get_peer_topics()
        self.assertIn('fresh', result)
        self.assertNotIn('stale', result)


if __name__ == '__main__':
    unittest.main()
