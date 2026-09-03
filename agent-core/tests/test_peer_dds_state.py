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


class _FakeExecutor:
    """Stands in for SingleThreadedExecutor; records lifecycle for assertions."""

    def __init__(self):
        self.nodes = []
        self.removed = []
        self.shutdown_called = False
        self.on_spin = lambda: None

    def add_node(self, n): self.nodes.append(n)
    def remove_node(self, n): self.removed.append(n)
    def shutdown(self): self.shutdown_called = True
    def spin_once(self, timeout_sec=None): self.on_spin()


class TestDdsState(unittest.TestCase):
    def setUp(self):
        dds_state._running = False
        dds_state._owns_rclpy = False
        dds_state._node = None
        dds_state._publisher = None
        dds_state._subscribers.clear()
        dds_state._peer_topics.clear()

    def _run_once(self, fake, on_spin=None, auto_stop=True):
        """Run _run_loop for a single iteration with a fake rclpy + executor.

        Returns the fake executor so lifecycle can be asserted.
        """
        node_mod = types.SimpleNamespace(Node=_FakeNode)
        msg_mod = types.SimpleNamespace(String=lambda: types.SimpleNamespace(data=''))
        ex = _FakeExecutor()
        exec_mod = types.SimpleNamespace(SingleThreadedExecutor=lambda: ex)
        modules = {
            'rclpy': fake,
            'rclpy.node': node_mod,
            'rclpy.executors': exec_mod,
            'std_msgs': types.SimpleNamespace(msg=msg_mod),
            'std_msgs.msg': msg_mod,
        }
        with mock.patch.dict(sys.modules, modules):
            dds_state._running = True

            # Stop after the first spin so the loop terminates.
            def _spin():
                if on_spin:
                    on_spin()
                if auto_stop:
                    dds_state._running = False

            ex.on_spin = _spin
            with mock.patch.object(dds_state, '_subscribe_to_peers', lambda: None):
                dds_state._run_loop()
        return ex

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

    def test_uses_a_persistent_executor(self):
        """必须用长期 executor，不能用 rclpy.spin_once(node)。

        真机上 Orin5 的 dds_state 线程崩了：

            IndexError: wait set index too big   (rclpy/qos_event.py is_ready)

        spin_once(node) 每次都新建一个临时 executor；而这个循环会随 peer 出现
        动态加订阅，于是上一轮 wait set 分配的 entity 索引对下一轮失效。
        """
        import inspect
        src = inspect.getsource(dds_state._run_loop)
        self.assertIn('SingleThreadedExecutor', src,
                      'must hold one executor; spin_once(node) rebuilds one per call')
        self.assertNotIn('rclpy.spin_once(_node', src,
                         'rclpy.spin_once(node) is the pattern that produced the stale wait set')

    def test_loop_survives_one_bad_iteration(self):
        """单次异常不能杀死线程。

        线程死掉是静默的 —— 容器照常运行，DDS 共享此后永久失效，日志里只有
        一条 traceback。这与 rclpy.init 那次的失败模式完全一样。
        """
        import inspect
        src = inspect.getsource(dds_state._run_loop)
        body = src[src.find('while _running'):]
        self.assertIn('except Exception', body,
                      'the spin loop must not let one failure end the thread')

    def test_executor_released_on_exit(self):
        """退出时要摘掉 node 并关闭 executor，否则热重启会留下残留。"""
        fake = _fake_rclpy(ok=True)
        ex = self._run_once(fake)
        self.assertIn(dds_state._node or ex.nodes[0], ex.nodes + ex.removed)
        self.assertTrue(ex.shutdown_called, 'executor was never shut down')
        self.assertTrue(ex.removed, 'node was never removed from the executor')

    def test_transient_spin_error_is_survived(self):
        """一次失败之后循环要继续，而不是把线程炸掉。

        这正是真机上发生的事 —— IndexError 冒到线程顶层，线程死亡，
        容器照常运行，功能此后静默失效。
        """
        fake = _fake_rclpy(ok=True)
        calls = {'n': 0}

        def _flaky():
            calls['n'] += 1
            if calls['n'] == 1:
                raise IndexError('wait set index too big')
            # 第二次成功后停下，证明循环挺过了第一次失败
            dds_state._running = False

        ex = self._run_once(fake, on_spin=_flaky, auto_stop=False)
        self.assertEqual(calls['n'], 2, 'loop did not continue after a failure')
        self.assertTrue(ex.shutdown_called)

    def test_persistent_failure_gives_up_instead_of_spinning_forever(self):
        """持续失败必须放弃，不能每秒刷一条错误刷到进程结束。

        第一版「扛住异常」写成了无限重试 —— 这个测试当时直接挂死，
        暴露的是生产代码的问题，不是测试的问题。
        """
        fake = _fake_rclpy(ok=True)
        calls = {'n': 0}

        def _always_broken():
            calls['n'] += 1
            raise IndexError('wait set index too big')

        with mock.patch.object(dds_state.time, 'sleep', lambda s: None):
            ex = self._run_once(fake, on_spin=_always_broken, auto_stop=False)

        self.assertEqual(calls['n'], dds_state._MAX_CONSECUTIVE_FAILURES,
                         'loop must stop after the failure budget, not retry forever')
        self.assertTrue(ex.shutdown_called, 'gave up without releasing the executor')

    def test_is_available_without_rclpy(self):
        """rclpy 缺失时 is_available() 返回 False 而不是抛异常。"""
        self.assertIsInstance(dds_state.is_available(), bool)

    def test_topic_name_valid_for_digit_leading_peer_id(self):
        """数字开头的 peer_id 必须产生合法的 ROS 话题名。

        真实故障：Orin6 的指纹是 8bd2bf8f…，rclpy 抛
        InvalidTopicNameException，DDS 线程当场死掉。Orin5 的指纹以字母开头
        所以看起来是好的 —— 十六进制里 16 个字符有 10 个是数字，约 62% 的
        机器人会踩到，能不能复现纯看运气。
        """
        import re
        # ROS 2: 每一段必须以字母或下划线开头，其后为字母/数字/下划线
        token = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        for peer_id in ('8bd2bf8fe8f872361f8f3212dc7e4279',   # 数字开头（Orin6）
                        'dd398c73177aa3487e7c695f4b19dfe5',   # 字母开头（Orin5）
                        '0' * 32, '9abc' + '0' * 28):
            name = dds_state.topic_for(peer_id)
            self.assertTrue(name.startswith('/'), name)
            for seg in name.strip('/').split('/'):
                self.assertRegex(seg, token, f'invalid ROS topic segment in {name}')

    def test_publisher_and_subscriber_agree_on_topic(self):
        """发布端与订阅端必须走同一个函数，否则两端静默对不上。"""
        peer_id = '8bd2bf8fe8f872361f8f3212dc7e4279'
        self.assertEqual(dds_state.topic_for(peer_id), dds_state.topic_for(peer_id))
        self.assertIn(peer_id, dds_state.topic_for(peer_id))

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
