"""
test_peer_visibility.py — agent 要能看见 peer 的存在。

起因是一次真实提问：Orin5 被问"能不能发现其他机器人"，它回答不能 —— 而它当时
确实配对着一台。整套 peer 机制都在跑，但没有任何东西把它交到 LLM 手里：没有工具
能列出 peer，prompt 里也一个字都没有。功能存在而 agent 不知道，等于不存在。

这里守两条：工具在两条路径上都注册了（主循环 + subagent），以及 L2 快照在有已
配对 peer 时会提到它们、没有时不会凭空多一段。
"""

import asyncio
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))


class TestPeerListTool(unittest.TestCase):
    def _run(self, peers, endpoints=None, discovered=(), **kw):
        from peer import delegation
        endpoints = endpoints or {}
        with mock.patch('peer.store.list_peers', return_value=peers), \
             mock.patch('peer.registry.registry.endpoints_for',
                        side_effect=lambda pid: endpoints.get(pid, [])), \
             mock.patch('peer.registry.registry.discovered', return_value=list(discovered)):
            return asyncio.run(delegation.peer_list(**kw))

    def test_names_a_paired_peer_and_its_role(self):
        out = self._run(
            [{'peer_id': 'dd39ab', 'display_name': 'orin6', 'role': 'operator'}],
            {'dd39ab': ['https://10.0.0.2:15678']},
        )
        self.assertIn('orin6', out)
        self.assertIn('dd39ab', out)
        self.assertIn('operator', out)
        self.assertIn('reachable', out)

    def test_says_so_when_a_paired_peer_has_no_address(self):
        """配对了但联系不上，和没配对是两回事，答案里必须能分清。"""
        out = self._run([{'peer_id': 'x', 'display_name': 'off', 'role': 'viewer'}])
        self.assertIn('no known address', out)

    def test_no_paired_peers_is_stated_plainly(self):
        self.assertIn('no paired agents', self._run([]))

    def test_unpaired_are_hidden_by_default_and_point_at_the_human(self):
        discovered = [{'peer_id': 'new1', 'display_name': 'stranger', 'source': 'mdns'}]
        hidden = self._run([], discovered=discovered)
        self.assertNotIn('stranger', hidden)
        shown = self._run([], discovered=discovered, include_unpaired=True)
        self.assertIn('stranger', shown)
        # 配对必须由人比对 6 位码，agent 不该以为自己能配对
        self.assertIn('human', shown)

    def test_states_that_a_peer_cannot_drive_actuators(self):
        """这条是安全铁律，agent 不该以为对方的请求可以直接执行。"""
        out = self._run([{'peer_id': 'a', 'display_name': 'b', 'role': 'operator'}],
                        {'a': ['https://1']})
        self.assertIn('actuator', out)


class TestToolIsRegistered(unittest.TestCase):
    def test_registered_in_the_main_loop(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / 'src/event/llm.py').read_text()
        self.assertIn("('peer_list', _peer_delegation.peer_list)", src)

    def test_reachable_from_a_subagent(self):
        """peer_delegate 就是因为漏在这个清单外面而变成过死代码。"""
        src = (pathlib.Path(__file__).resolve().parents[1] / 'src/subagent/agent.py').read_text()
        i = src.index('_DESKTOP_TOOLS = {')
        self.assertIn("'peer_list'", src[i:i + 400])


class TestPromptMentionsPeers(unittest.TestCase):
    def _snapshot(self, peers):
        import prompt
        with mock.patch('peer.store.list_peers', return_value=peers), \
             mock.patch('task_store.active_tasks', return_value=[]), \
             mock.patch('collector.get_available_sources', return_value=[]):
            return prompt._env_dynamic()

    def test_paired_peer_appears_in_the_snapshot(self):
        out = self._snapshot([{'peer_id': 'dd39ab', 'display_name': 'orin6', 'role': 'operator'}])
        self.assertIn('orin6', out)
        self.assertIn('peer_list', out, 'agent 得知道用什么去问详情')

    def test_no_peers_adds_no_section(self):
        """单机是常态，凭空一段 peers 只会让它谈论一个没有的能力。"""
        self.assertNotIn('<peers', self._snapshot([]))

    def test_blocked_peer_is_not_offered(self):
        out = self._snapshot([{'peer_id': 'z', 'display_name': 'banned', 'role': 'blocked'}])
        self.assertNotIn('banned', out)

    def test_a_broken_peer_store_does_not_break_the_prompt(self):
        """prompt 是每一轮推理的必经路径，不能被一个可选功能拖崩。"""
        import prompt
        with mock.patch('peer.store.list_peers', side_effect=RuntimeError('no db')), \
             mock.patch('task_store.active_tasks', return_value=[]), \
             mock.patch('collector.get_available_sources', return_value=[]):
            self.assertNotIn('<peers', prompt._env_dynamic())


if __name__ == '__main__':
    unittest.main()
