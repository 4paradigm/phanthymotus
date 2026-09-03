"""
test_peer_tool_authorization.py — peer 能调本机哪些工具。

这是机器人之间的信任边界，而它此前有两个都"看着合理"的错误实现：

1. **按工具名猜类别**：判据是名字里有没有 move/grasp/speak/set_/execute/control。
   真实机器人的执行器叫 loco、led、speaker、switch_mode，一个都不匹配 —— 所以
   **viewer 角色也能驱动移动**。
2. **只信声明的 type**：type 区分不了"算数据"和"动硬件" —— actucore 把 vla 声明成
   processor，而 VLA 策略是会让机器人动的，navigation 的 goto 同理。

所以判定必须"层 + 类型"一起看：category=actucore 的一律算执行，category=perception
的一律算只读（ASR/TTS 合成/OCR 只产出话题数据），其余按 type。未声明 type 算执行 ——
信任边界要 fail closed。

角色决定能碰哪一类，tool_filter 只在角色允许的范围内再收窄。
"""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402
from peer import tools as peer_tools  # noqa: E402


# 真实机器上取到的形状：驱动的 loco/speaker 是 actuator，感知的 tts 是 processor。
_REGISTRY = {
    'mcp-drv': {
        'category': 'driver',
        'tool_meta': {
            'mcp__mcp-drv__loco':      {'type': 'actuator'},
            'mcp__mcp-drv__speaker':   {'type': 'actuator'},
            'mcp__mcp-drv__battery':   {'type': 'sensor'},
            'mcp__mcp-drv__model':     {'type': 'resource'},
            'mcp__mcp-drv__mystery':   {},                      # 驱动没声明
        },
    },
    'mcp-perc': {
        'category': 'perception',
        'tool_meta': {
            'mcp__mcp-perc__tts': {'type': 'processor'},
            'mcp__mcp-perc__ocr': {'type': 'processor'},
        },
    },
    'mcp-actu': {
        'category': 'actucore',
        'tool_meta': {
            # 声明成 processor，但它是执行层 —— 会让机器人移动。
            'mcp__mcp-actu__navigation': {'type': 'processor'},
            'mcp__mcp-actu__vla':        {'type': 'processor'},
        },
    },
}


def _peer(role='operator', tool_filter='*'):
    return {'peer_id': 'p1', 'display_name': 'Peer', 'role': role, 'tool_filter': tool_filter}


class _Registry(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.dict(mcp_client.registry, _REGISTRY, clear=True)
        self._patch.start()
        self.addCleanup(self._patch.stop)


class TestClassification(_Registry):
    def test_perception_processors_are_read_only(self):
        """ASR/TTS 合成/OCR 产出话题数据，不动硬件。"""
        self.assertTrue(peer_tools.is_read_only('mcp__mcp-perc__tts'))
        self.assertTrue(peer_tools.is_read_only('mcp__mcp-perc__ocr'))

    def test_actucore_acts_even_when_it_declares_processor(self):
        """执行层就是执行层 —— 只信 type 会把 vla 当成算数据的。"""
        self.assertFalse(peer_tools.is_read_only('mcp__mcp-actu__navigation'))
        self.assertFalse(peer_tools.is_read_only('mcp__mcp-actu__vla'))

    def test_driver_sensors_read_and_actuators_act(self):
        self.assertTrue(peer_tools.is_read_only('mcp__mcp-drv__battery'))
        self.assertTrue(peer_tools.is_read_only('mcp__mcp-drv__model'))
        self.assertFalse(peer_tools.is_read_only('mcp__mcp-drv__loco'))

    def test_undeclared_type_counts_as_acting(self):
        """fail closed：驱动什么都没声明的地方，恰恰是猜得最糟的地方。"""
        self.assertFalse(peer_tools.is_read_only('mcp__mcp-drv__mystery'))

    def test_a_name_that_would_have_fooled_the_old_keyword_guess(self):
        """loco/speaker 不含 move/speak 等关键词 —— 旧实现因此放行了 viewer。"""
        ok, why = peer_tools.check_tool_permission('p1', 'mcp__mcp-drv__loco')
        with mock.patch('peer.store.get', return_value=_peer(role='viewer')):
            ok, why = peer_tools.check_tool_permission('p1', 'mcp__mcp-drv__loco')
        self.assertFalse(ok)
        self.assertIn('viewer', why)


class TestRoleBoundary(_Registry):
    def _check(self, tool, role='operator', tool_filter='*'):
        with mock.patch('peer.store.get', return_value=_peer(role, tool_filter)):
            return peer_tools.check_tool_permission('p1', tool)

    def test_viewer_reads_only(self):
        self.assertTrue(self._check('mcp__mcp-drv__battery', role='viewer')[0])
        self.assertTrue(self._check('mcp__mcp-perc__tts', role='viewer')[0])
        self.assertFalse(self._check('mcp__mcp-drv__loco', role='viewer')[0])
        self.assertFalse(self._check('mcp__mcp-actu__navigation', role='viewer')[0])

    def test_operator_may_act(self):
        self.assertTrue(self._check('mcp__mcp-drv__loco', role='operator')[0])
        self.assertTrue(self._check('mcp__mcp-actu__navigation', role='operator')[0])

    def test_blocked_gets_nothing(self):
        self.assertFalse(self._check('mcp__mcp-drv__battery', role='blocked')[0])

    def test_filter_narrows_within_the_role(self):
        self.assertFalse(self._check('mcp__mcp-drv__loco', tool_filter='*battery*')[0])
        self.assertTrue(self._check('mcp__mcp-drv__loco', tool_filter='*loco*')[0])

    def test_refusal_says_what_to_do(self):
        _ok, why = self._check('mcp__mcp-drv__loco', role='viewer')
        self.assertIn('operator', why)
        self.assertIn('peer_delegate', why)


class TestListMatchesCall(_Registry):
    """列出来的必须调得动。

    列一个必然 403 的工具，会让对端的 LLM 围着一个不存在的能力做计划。
    """

    def _schemas(self):
        return [{'name': n} for info in _REGISTRY.values() for n in info['tool_meta']]

    def _listed(self, role, tool_filter='*'):
        with mock.patch('peer.store.get', return_value=_peer(role, tool_filter)):
            return {s['name'] for s in peer_tools.filter_schemas('p1', self._schemas())}

    def test_every_listed_tool_is_callable(self):
        for role in ('viewer', 'operator'):
            with mock.patch('peer.store.get', return_value=_peer(role)):
                for name in self._listed(role):
                    ok, why = peer_tools.check_tool_permission('p1', name)
                    self.assertTrue(ok, f'{role} 被列出了 {name} 却调不动: {why}')

    def test_nothing_callable_is_hidden(self):
        for role in ('viewer', 'operator'):
            listed = self._listed(role)
            with mock.patch('peer.store.get', return_value=_peer(role)):
                for s in self._schemas():
                    ok, _ = peer_tools.check_tool_permission('p1', s['name'])
                    if ok:
                        self.assertIn(s['name'], listed, f'{role} 调得动却没列出 {s["name"]}')

    def test_viewer_list_excludes_actuators(self):
        listed = self._listed('viewer')
        self.assertIn('mcp__mcp-perc__tts', listed)
        self.assertNotIn('mcp__mcp-drv__loco', listed)
        self.assertNotIn('mcp__mcp-actu__vla', listed)

    def test_blocked_lists_nothing(self):
        self.assertEqual(self._listed('blocked'), set())


class TestNotificationScope(_Registry):
    """只有会动的调用才通知。

    只读调用可能被高频轮询（电池、摄像头帧），全都推事件会把活动流刷爆；而通知的
    意义是「有人让这台机器人做了什么」，读状态不是那件事。
    """

    def test_acting_tools_are_announced(self):
        self.assertTrue(not peer_tools.is_read_only('mcp__mcp-drv__loco'))
        self.assertTrue(not peer_tools.is_read_only('mcp__mcp-actu__vla'))

    def test_read_only_tools_are_not(self):
        self.assertTrue(peer_tools.is_read_only('mcp__mcp-drv__battery'))
        self.assertTrue(peer_tools.is_read_only('mcp__mcp-perc__tts'))

    def test_the_endpoint_uses_that_same_predicate(self):
        """判据只有一处 —— 端点和授权逻辑漂移了，通知范围就会悄悄变。"""
        src = (pathlib.Path(__file__).resolve().parents[1] / 'src/api/peer.py').read_text()
        self.assertIn('peer_tools.is_read_only(tool_name)', src)


if __name__ == '__main__':
    unittest.main()
