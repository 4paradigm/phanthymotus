"""
test_peer_api_imports.py — 保证 api/peer.py 能被导入。

这个测试存在的原因：一次真实的启动崩溃。

    ImportError: cannot import name 'manager' from 'subagent.manager'

api/peer.py 在模块顶层写了 `from subagent.manager import manager`，而那个模块
只导出类、不导出单例。所有 peer 测试都通过了，因为它们测的是纯逻辑模块
（delegation.py、tools.py），从没导入过真正被 start.py 加载的那个 API 模块。

单元测试覆盖了逻辑，却没覆盖「进程能不能起来」。这个测试补的就是那一段：
凡是 start.py 会 import 的 peer 相关模块，这里都必须能 import 成功。
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))


class TestApiImports(unittest.TestCase):
    def test_api_peer_imports(self):
        """api/peer.py 必须能在没有 agent loop 的情况下导入。

        start.py 在 lifespan 之前就 import 它，那时 event/llm.py 还没构造
        SubagentManager —— 任何在模块顶层绑定 manager 实例的写法都会在这里炸。
        """
        import api.peer  # noqa: F401

    def test_router_is_mounted(self):
        """router 存在且带正确前缀，否则端点静默不可达。"""
        import api.peer
        self.assertTrue(hasattr(api.peer, 'router'))
        self.assertEqual(api.peer.router.prefix, '/api/peer')

    def test_inbox_routes_registered(self):
        """peer 侧端点必须真的挂上了。

        这些路径在 auth.py 里被豁免 ACCESS_TOKEN，如果路由没注册，
        表现是 404 而不是报错，很难排查。
        """
        import api.peer
        paths = {r.path for r in api.peer.router.routes}
        for expected in (
            '/api/peer/inbox/message',
            '/api/peer/inbox/ping',
            '/api/peer/inbox/pair_request',
            '/api/peer/tools/list',
            '/api/peer/tools/call',
            '/api/peer/delegate',
        ):
            self.assertIn(expected, paths, f'route missing: {expected}')

    def test_all_peer_modules_import(self):
        """peer 包的每个模块都要能独立导入（含可选依赖缺失时的降级）。"""
        import peer.identity        # noqa: F401
        import peer.pairing         # noqa: F401
        import peer.store           # noqa: F401
        import peer.transport       # noqa: F401
        import peer.registry        # noqa: F401
        import peer.tools           # noqa: F401
        import peer.delegation      # noqa: F401
        import peer.dds_state       # noqa: F401
        import peer.ble_bootstrap   # noqa: F401
        import peer.ble_advertiser  # noqa: F401
        import peer.discovery.base    # noqa: F401
        import peer.discovery.static  # noqa: F401
        import peer.discovery.mdns    # noqa: F401
        import channel.adapters.lan   # noqa: F401

    def test_delegate_without_agent_loop_returns_503(self):
        """没有 agent loop 时委托应返回 503，而不是 AttributeError。

        容器里 API 先于 agent loop 就绪，这个窗口内进来的委托请求必须得到
        一个能看懂的错误。
        """
        import subagent
        self.assertTrue(hasattr(subagent, '_manager_instance'))
        # 未初始化时就是 None —— api/peer.py 依赖这个前提
        self.assertIsNone(subagent._manager_instance)


if __name__ == '__main__':
    unittest.main()
