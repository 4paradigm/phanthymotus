"""
test_log_noise.py — 日志过滤只能删掉「成功的轮询」，绝不能删掉错误。

动机是排查成本：天轶上追那批 LLM 400 时，8 条真正有用的行埋在 15209 条 peer 403 和
几百条面板轮询里。`docker logs` 的可读性不是审美问题，它直接决定一次排查要花多久。

但「过滤日志」这件事本身很容易变成下一个 bug：一旦把失败的请求也滤掉，故障就变成了沉默。
所以这里逐条锁死边界 —— 只有**已知路径**上的 **2xx/3xx** 会被丢弃，其余一律保留。

Run: cd agent-core && python3 -m pytest tests/test_log_noise.py
"""

import logging
import os
import pathlib
import sys
import time
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'
START = SRC / 'start.py'


def _build_filter():
    """Rebuild the filter from start.py's own source.

    start.py runs uvicorn at import under `__main__`, so it cannot be imported
    here. The filter is lifted out of the source instead, which keeps this test
    honest — it fails if the real definition changes shape.
    """
    src = START.read_text(encoding='utf-8')
    block = src[src.index('_QUIET_POLLS = ('):src.index("logging.getLogger('uvicorn.access')")]
    ns: dict = {'logging': logging, 'time': time}
    exec(block.replace('\n    ', '\n'), ns)       # de-indent from the __main__ block
    return ns['_AccessPollFilter'](), ns['_QUIET_POLLS']


class _Rec(logging.LogRecord):
    def __init__(self, msg):
        super().__init__('uvicorn.access', logging.INFO, __file__, 0, msg, None, None)


class TestAccessFilter(unittest.TestCase):
    def setUp(self):
        self.filt, self.quiet = _build_filter()

    def _kept(self, msg):
        return self.filt.filter(_Rec(msg))

    # ── what it removes ──────────────────────────────────────────────────────

    def test_it_drops_successful_dashboard_polling(self):
        for path in ('GET /api/mcp ', 'GET /api/topics/status ',
                     'GET /api/canvas/edit-status?session_id=x ',
                     'POST /api/mcp '):
            msg = f'127.0.0.1:1 - "{path}HTTP/1.1" 200 OK'
            self.assertFalse(self._kept(msg), f'应被过滤: {path}')

    def test_a_3xx_poll_is_also_dropped(self):
        self.assertFalse(self._kept('1.2.3.4:5 - "GET /api/mcp HTTP/1.1" 304 Not Modified'))

    # ── what it must never remove ────────────────────────────────────────────

    def test_a_failing_poll_is_always_kept(self):
        """最重要的一条：把失败也滤掉，等于把故障变成沉默。"""
        for code in ('400 Bad Request', '401 Unauthorized', '403 Forbidden',
                     '404 Not Found', '423 Locked', '500 Internal Server Error'):
            msg = f'127.0.0.1:1 - "GET /api/mcp HTTP/1.1" {code}'
            self.assertTrue(self._kept(msg), f'{code} 必须保留')

    def test_an_unlisted_path_is_kept_even_when_successful(self):
        self.assertTrue(self._kept('127.0.0.1:1 - "POST /api/acp/complete HTTP/1.1" 200 OK'))
        self.assertTrue(self._kept('127.0.0.1:1 - "DELETE /api/peer/paired/abc HTTP/1.1" 200 OK'))

    def test_a_peer_call_is_kept(self):
        """peer 通过签名驱动本机执行器 —— 那是审计线索，不是噪音。"""
        self.assertTrue(self._kept('10.0.0.5:1 - "POST /api/peer/tools/call HTTP/1.1" 200 OK'))

    def test_a_non_access_line_is_untouched(self):
        self.assertTrue(self._kept('[decision] error in _one_turn: something broke'))
        self.assertTrue(self._kept('Traceback (most recent call last):'))

    def test_a_prefix_is_not_matched_loosely(self):
        """'GET /api/mcp ' 带尾空格，不能顺带吃掉 /api/mcp/<id>/call。"""
        self.assertTrue(
            self._kept('127.0.0.1:1 - "POST /api/mcp/mcp-123/call HTTP/1.1" 200 OK'),
            '工具调用是真实动作，不能当轮询丢掉')


class TestSuppressedVolumeIsStillVisible(unittest.TestCase):
    """删掉的是行，不是「这些请求发生过」这个事实。

    这条边界是有代价换来的：#234 那个泄漏的订阅**正是**靠驱动逐条记录请求才被发现的，
    线索是量的形状 —— 每小时 39k、53k、60k、67k、74k，一直在涨。把它滤掉，失控的客户端
    就隐形了，那比它消除的噪音更糟。所以行走、计数留。
    """

    def setUp(self):
        self.filt, _ = _build_filter()

    def _poll(self, n=1, path='GET /api/mcp '):
        for _ in range(n):
            self.filt.filter(_Rec(f'127.0.0.1:1 - "{path}HTTP/1.1" 200 OK'))

    def test_suppressed_polls_are_counted(self):
        self._poll(5)
        self.assertEqual(sum(self.filt._counts.values()), 5)

    def test_nothing_is_printed_before_the_window_elapses(self, ):
        import io, contextlib as _c
        buf = io.StringIO()
        with _c.redirect_stdout(buf):
            self._poll(50)
        self.assertEqual(buf.getvalue(), '', '不该每次都汇总，那又变成刷屏了')

    def test_a_summary_names_the_count_the_rate_and_the_busiest_paths(self):
        import io, contextlib as _c
        self._poll(120, 'GET /api/mcp ')
        self._poll(30, 'GET /api/topics/status ')
        # 把窗口推到过去，模拟十分钟已过
        self.filt._last_report -= 10_000
        buf = io.StringIO()
        with _c.redirect_stdout(buf):
            self._poll(1)
        out = buf.getvalue()
        self.assertIn('suppressed', out)
        self.assertIn('151', out)                 # 120 + 30 + 触发那一条
        self.assertIn('/api/mcp', out)
        self.assertIn('/s)', out, '速率是判断失控的那个量')

    def test_counters_reset_after_a_report(self):
        self._poll(10)
        self.filt._last_report -= 10_000
        import io, contextlib as _c
        with _c.redirect_stdout(io.StringIO()):
            self._poll(1)
        self.assertEqual(self.filt._counts, {}, '不清零就会把同一批重复报出去')

    def test_kept_lines_are_not_counted(self):
        self.filt.filter(_Rec('127.0.0.1:1 - "GET /api/mcp HTTP/1.1" 500 Boom'))
        self.assertEqual(self.filt._counts, {})


class TestSseAbsentIsSaidOnce(unittest.TestCase):
    def test_the_no_sse_line_is_deduplicated_per_device(self):
        import mcp_client
        src = (SRC / 'mcp_client.py').read_text(encoding='utf-8')
        self.assertIn('_sse_absent', src)
        self.assertTrue(hasattr(mcp_client, '_sse_absent'))
        # 按 url 记忆，换端口的设备应当重新报一次
        self.assertIn('_sse_absent[mcp_id] = sse_url', src)


class TestHeartbeatsLogEdges(unittest.TestCase):
    """perception 和 actucore 的注册心跳：报状态变化，不报每一拍。"""

    def _loop_src(self, path):
        """The whole `def _run()` that owns the heartbeat, not a byte window.

        A fixed-size window silently stopped covering the `except` branch the
        moment the loop grew — the test failed for the wrong reason. Slice on
        the function boundaries instead so it keeps covering what it claims to.
        """
        src = pathlib.Path(path).read_text(encoding='utf-8')
        start = src.rindex('def _run():', 0, src.index('[register] heartbeat ok'))
        end = src.index('threading.Thread(target=_run', start)
        return src[start:end]

    def test_both_guard_the_ok_line(self):
        root = SRC.parents[1]
        for p in (root / 'perception' / 'main.py', root / 'actucore' / 'main.py'):
            block = self._loop_src(p)
            self.assertIn('healthy is not True', block, f'{p.name} 仍在每拍都打日志')

    def test_a_slow_keepalive_keeps_proof_of_life(self):
        """只报边沿会让「健康」和「注册线程死了」在日志里长得一样。

        以前每 30 秒一条 ok 顺带就是存活证明：线程卡死，日志就安静下来，看得出来。
        改成只报变化之后，安静**就是**健康状态，于是两者无法区分 —— 这是拿一个盲点
        换了另一个。低频那条把这个信号以 1/60 的代价留下来。
        """
        root = SRC.parents[1]
        for p in (root / 'perception' / 'main.py', root / 'actucore' / 'main.py'):
            src = pathlib.Path(p).read_text(encoding='utf-8')
            self.assertIn('still registered', src, f'{p.name} 没有存活行')
            self.assertIn('REGISTER_ALIVE_INTERVAL_S', src)
            val = next(float(ln.split('=')[1]) for ln in src.splitlines()
                       if ln.startswith('REGISTER_ALIVE_INTERVAL_S ='))
            self.assertGreaterEqual(val, 600.0, '太频繁就退化回刷屏')
            self.assertLessEqual(val, 3600.0, '太稀疏就失去存活证明的意义')

    def test_the_keepalive_uses_a_monotonic_clock(self):
        """这些机器上的 wall clock 会跳（G1 的容器 StartedAt 是 1970）。"""
        root = SRC.parents[1]
        for p in (root / 'perception' / 'main.py', root / 'actucore' / 'main.py'):
            block = self._loop_src(p)
            self.assertIn('monotonic()', block)

    def test_failures_are_still_logged_every_time(self):
        """连接抖动本身是症状，折叠它会掩盖掉掉线的频率。"""
        root = SRC.parents[1]
        for p in (root / 'perception' / 'main.py', root / 'actucore' / 'main.py'):
            block = self._loop_src(p)
            fail = block[block.index('except Exception'):]
            self.assertIn('log.warning', fail)
            self.assertNotIn('if healthy', fail.split('log.warning')[0],
                             f'{p.name} 的失败分支被加了条件')


if __name__ == '__main__':
    unittest.main()
