"""面板必须自己等到数据，而不是要人关掉重开。

服务端在话题还没注册时会发一条 error 然后**关掉**连接（api/inspection.py 的
bus_ws）。前端原先只在 onclose 里写一句状态、不重连，于是面板永久停在"这个
topic 没有发出任何数据"上 —— 哪怕几秒后卡片启动了、话题注册了、数据开始流，它
也不会知道。

而"先打开面板、再启动卡片"恰恰是最自然的操作顺序，所以这个洞每次都会踩到。

没有 JS 测试框架、没有构建步骤（同 test_control_renderer_registered.py 的理由），
所以这里做文本检查：这不是算法问题，是一条缺失的边 —— 断开之后没有任何东西把它
接回去。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_detail_panel_reconnect.py
"""
import pathlib
import re
import unittest

WEB = pathlib.Path(__file__).resolve().parents[1] / 'web' / 'js'
SOURCE = (WEB / 'detail-panel.js').read_text()
SERVER = (pathlib.Path(__file__).resolve().parents[1]
          / 'src' / 'api' / 'inspection.py').read_text()


def _fn(name: str) -> str:
    match = re.search(rf'function {name}\(.*?\n\}}', SOURCE, re.S)
    assert match, f'{name} 不见了'
    return match.group(0)


class ReconnectTest(unittest.TestCase):

    def test_the_server_really_does_close_on_an_unregistered_topic(self):
        """这条测试是前端那几条的前提。服务端哪天改成保持连接，前端的重连就从
        必需变成多余，而这里会提醒下一个人去想这件事。"""
        block = SERVER[SERVER.index('async def bus_ws'):]
        block = block[:block.index('\n\n\n')] if '\n\n\n' in block else block
        self.assertIn('not registered', block)
        self.assertIn('websocket.close()', block)

    def test_a_closed_socket_is_retried(self):
        self.assertIn('_scheduleRetry', _fn('_connect'))

    def test_the_retry_is_dropped_when_the_panel_moved_on(self):
        """代次守卫。没有它，在途的重连会醒来往下一个话题的面板上写数据 ——
        而那正是 _cleanup 里清 handler 的那段注释已经踩过一次的坑。"""
        self.assertIn('session !== _session', _fn('_connect'))
        self.assertIn('session !== _session', _fn('_scheduleRetry'))

    def test_the_generation_advances_on_cleanup(self):
        """光清 timer 不够：_connect 可能已经在跑，timer 已经烧掉了。"""
        self.assertIn('_session += 1;', _fn('_cleanup'))
        self.assertIn('_retryTimer = null;', _fn('_cleanup'))

    def test_the_dead_end_wording_is_gone(self):
        """"连接已断开，这个 topic 没有发出任何数据"是一句结论。现在它只是在
        等，措辞得说出这一点，否则人还是会去关掉重开。"""
        self.assertNotIn('连接已断开，这个 topic 没有发出任何数据', SOURCE)
        self.assertIn('秒后重试', SOURCE)

    def test_only_onclose_reconnects(self):
        """onerror 之后总会跟一个 onclose。两边都重连就会连出双份 socket。"""
        body = _fn('_connect')
        onerror = body[body.index('_ws.onerror'):]
        self.assertNotIn('_scheduleRetry', onerror)


if __name__ == '__main__':
    unittest.main()
