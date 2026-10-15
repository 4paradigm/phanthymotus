"""barrier 丢掉的调用，工具结果必须让模型看出「没执行」。

天轶 2026-10-15，导览最后一轮：结束语的 `tts.speak` 在 13:18:57 发起，被 barrier
挡在上一段讲解后面；barrier 在 13:21:50 超时，于是这次调用**没有发给设备**
（`event/llm.py` 的 not_dispatched 分支刻意不派发）。模型拿到的结果是

    {"status": "not_dispatched",
     "reason": "barrier wait interrupted before dispatch (status=timeout)"}

下一轮它的原话是「结束语已播放完毕。全展厅导览已全部完成」，然后 finish()。那段
结束语一个字都没播出去。

所以 timeout 这条必须显式说明：没发声、没动作、不要当成已完成、确认需要就重发。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest agent-core/tests/test_not_dispatched_reads_as_failure.py -q
"""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from event.llm import _not_dispatched, _tool_content  # noqa: E402

SPEAK = 'mcp__mcp-1786695210__tts__speak'


class TestNotDispatched(unittest.TestCase):
    def test_timeout_is_an_error_and_says_nothing_happened(self):
        r = _not_dispatched(SPEAK, {'status': 'timeout', 'actions': ['tts-f78f92a2']})
        self.assertFalse(r['dispatched'])
        self.assertIn('error', r, '顶层没有 error，模型读不出这是失败')
        self.assertTrue(r.get('retry'))
        msg = r['error']
        self.assertIn('未执行', msg)
        self.assertIn('没有发声', msg)
        self.assertIn('tts-f78f92a2', msg, '没说在等谁，无法归因')
        self.assertIn(SPEAK, msg)

    def test_timeout_does_not_read_as_success(self):
        """这条是针对那次误读本身：结果文本里不能出现任何「完成/成功」的说法。"""
        out = _tool_content(_not_dispatched(SPEAK, {'status': 'timeout',
                                                    'actions': ['tts-1']}))
        for claim in ('已完成', '播放完毕', 'completed', 'success', '成功'):
            self.assertNotIn(claim, out, f'结果里出现了 {claim!r}')

    def test_cancelled_does_not_ask_for_a_blind_retry(self):
        """被用户打断和超时不是一回事：前者重发等于顶着用户说话。"""
        r = _not_dispatched(SPEAK, {'status': 'cancelled', 'actions': ['tts-1']})
        self.assertIn('打断', r['error'])
        self.assertNotIn('retry', r)

    def test_reconsidering_keeps_the_machine_readable_reason(self):
        r = _not_dispatched(SPEAK, {'status': 'reconsidering', 'actions': ['tts-1']})
        self.assertIn('reconsidering', r['reason'])
        self.assertIn('error', r)

    def test_missing_actions_does_not_crash(self):
        for acp in ({'status': 'timeout'}, {}, None):
            r = _not_dispatched(SPEAK, acp)
            self.assertFalse(r['dispatched'])
            self.assertIn('error', r)

    def test_result_survives_the_tool_content_funnel(self):
        """非字符串的工具结果曾让整轮请求 400（见 _tool_content 的注释）。"""
        out = _tool_content(_not_dispatched(SPEAK, {'status': 'timeout',
                                                    'actions': ['tts-1']}))
        self.assertIsInstance(out, str)
        self.assertEqual(json.loads(out)['status'], 'not_dispatched')


class TestDispatchSitesUseIt(unittest.TestCase):
    def test_both_barrier_branches_go_through_the_helper(self):
        """mcp__ 和系统工具两条分支都要用它 —— 漏一条就只在一边生效。

        以前两处各写了一份同样的 dict 字面量，这正是「改了一处以为改完了」的形状。
        """
        src = (pathlib.Path(__file__).resolve().parents[1]
               / 'src' / 'event' / 'llm.py').read_text(encoding='utf-8')
        self.assertEqual(
            src.count('_not_dispatched(name, _barrier_result)'), 2,
            '两条 barrier 分支没有都走 _not_dispatched')
        self.assertNotIn('"reason": f"barrier wait interrupted before dispatch "', src)


if __name__ == '__main__':
    unittest.main()
