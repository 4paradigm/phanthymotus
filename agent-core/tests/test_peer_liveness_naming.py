"""
test_peer_liveness_naming.py — 在线状态与命名。

两件事都是被真实提问逼出来的：

1. **在线状态。** 配对是持久的，可达不是。快照里只有 id 和 role 时，agent 会计划把
   任务交给一台已经关机的机器人 —— 失败要等到超时才暴露，而它早就把计划说出口了。
2. **重名。** 快照里带名字而不是 peer_id（32 位十六进制单个就比整行其余部分还贵），
   但名字是人取的、会撞。撞了就是两行一模一样，agent 无从区分，委派只能被拒。

命名的渲染与解析必须是同一份契约，否则失败方式最难查：快照显示一个标签，工具却
不认它。
"""

import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from peer import liveness, naming  # noqa: E402


def _peer(pid, name='', last_seen=None, role='operator'):
    return {'peer_id': pid, 'display_name': name, 'role': role,
            'last_seen': last_seen if last_seen is not None else time.time()}


class _Advert:
    def __init__(self, age_s):
        self.last_seen = time.time() - age_s


class TestLiveness(unittest.TestCase):
    def _live(self, peer, advert_age=None, endpoints=('https://x',)):
        advert = _Advert(advert_age) if advert_age is not None else None
        with mock.patch('peer.registry.registry.get', return_value=advert), \
             mock.patch('peer.registry.registry.endpoints_for', return_value=list(endpoints)):
            return liveness.liveness(peer)

    def test_recent_contact_means_online(self):
        """状态推送 5s 一轮，刚联系过的必须算在线。"""
        self.assertTrue(self._live(_peer('a', 'A', time.time() - 3))['online'])

    def test_silent_peer_with_a_stale_advert_is_offline(self):
        """关机两分钟的机器人还留在 registry 里（保留 300s），不能因此算在线。"""
        live = self._live(_peer('a', 'A', time.time() - 600), advert_age=200)
        self.assertFalse(live['online'])

    def test_a_fresh_advert_alone_is_enough(self):
        """状态共享可能没开；mDNS 刚应答过也说明它活着。"""
        live = self._live(_peer('a', 'A', time.time() - 600), advert_age=10)
        self.assertTrue(live['online'])

    def test_never_contacted_is_distinguishable_from_long_ago(self):
        live = self._live(_peer('a', 'A', 0))
        self.assertIsNone(live['contact_age_s'])
        self.assertEqual(liveness.describe_age(None), 'never')

    def test_age_is_described_at_human_scale(self):
        self.assertEqual(liveness.describe_age(12), '12s ago')
        self.assertEqual(liveness.describe_age(300), '5min ago')
        self.assertEqual(liveness.describe_age(7200), '2.0h ago')


class TestNaming(unittest.TestCase):
    def test_unique_names_pay_nothing(self):
        peers = [_peer('aaaa1111' + '0' * 24, 'Orin5'), _peer('bbbb2222' + '0' * 24, 'Orin6')]
        self.assertEqual(set(naming.labels(peers).values()), {'Orin5', 'Orin6'})

    def test_colliding_names_get_an_id_suffix(self):
        a, b = _peer('8bd2bf8f' + '0' * 24, 'Orin6'), _peer('dd398c73' + '0' * 24, 'Orin6')
        labels = naming.labels([a, b])
        self.assertEqual(labels[a['peer_id']], 'Orin6#8bd2bf8f')
        self.assertEqual(labels[b['peer_id']], 'Orin6#dd398c73')

    def test_suffix_grows_when_the_prefix_also_collides(self):
        a = _peer('8bd2bf8f' + 'a' * 24, 'Orin6')
        b = _peer('8bd2bf8f' + 'b' * 24, 'Orin6')
        labels = naming.labels([a, b])
        self.assertNotEqual(labels[a['peer_id']], labels[b['peer_id']])

    def test_an_unnamed_peer_uses_the_same_prefix_convention(self):
        p = _peer('ffee0011' + '0' * 24, '')
        self.assertEqual(naming.labels([p])[p['peer_id']], 'ffee0011')

    def test_every_rendered_label_resolves_back(self):
        """渲染与解析是一份契约；漂移了就会显示工具不认的标签。"""
        peers = [_peer('8bd2bf8f' + '0' * 24, 'Orin6'),
                 _peer('dd398c73' + '0' * 24, 'Orin6'),
                 _peer('aa11bb22' + '0' * 24, 'Orin5'),
                 _peer('ffee0011' + '0' * 24, '')]
        for pid, label in naming.labels(peers).items():
            got, why = naming.resolve(label, peers)
            self.assertIsNotNone(got, f'{label} 解析不回来: {why}')
            self.assertEqual(got['peer_id'], pid)

    def test_a_bare_colliding_name_is_refused_with_the_options(self):
        peers = [_peer('8bd2bf8f' + '0' * 24, 'Orin6'), _peer('dd398c73' + '0' * 24, 'Orin6')]
        got, why = naming.resolve('Orin6', peers)
        self.assertIsNone(got, '不能替调用方在两台机器人之间挑一台')
        self.assertIn('Orin6#8bd2bf8f', why)
        self.assertIn('Orin6#dd398c73', why)

    def test_exact_peer_id_and_prefix_both_work(self):
        p = _peer('aa11bb22' + '0' * 24, 'Orin5')
        self.assertEqual(naming.resolve(p['peer_id'], [p])[0]['peer_id'], p['peer_id'])
        self.assertEqual(naming.resolve('aa11bb', [p])[0]['peer_id'], p['peer_id'])

    def test_unknown_peer_lists_what_is_available(self):
        peers = [_peer('aa11bb22' + '0' * 24, 'Orin5')]
        got, why = naming.resolve('Nope', peers)
        self.assertIsNone(got)
        self.assertIn('Orin5', why)


class TestAgentRunningIsSeparateFromReachable(unittest.TestCase):
    """可达 ≠ 能接活。

    这是真机上看到的：Orin6 的智能控制是关的，却在 Orin5 上显示在线 —— 因为状态
    推送与控制开关无关，每 5s 照推。而 /api/peer/delegate 需要随主事件循环才建起来
    的 subagent manager，所以那台机器会 503。只报 online 就等于让 agent 承诺一件
    做不到的事。
    """

    def _live(self, shared):
        with mock.patch('peer.registry.registry.get', return_value=_Advert(5)), \
             mock.patch('peer.registry.registry.endpoints_for', return_value=['https://x']), \
             mock.patch('peer.dds_state.get_peer_topics', return_value=shared):
            return liveness.liveness(_peer('a' * 32, 'A', time.time() - 2))

    def test_reports_the_peers_own_answer(self):
        self.assertIs(self._live({'a' * 32: {'agent_running': False}})['agent_running'], False)
        self.assertIs(self._live({'a' * 32: {'agent_running': True}})['agent_running'], True)

    def test_never_told_is_unknown_not_off(self):
        """旧版本 peer 不上报这个字段；把"没说"当成 off 会白白放弃一个可用的 peer。"""
        self.assertIsNone(self._live({})['agent_running'])
        self.assertIsNone(self._live({'a' * 32: {}})['agent_running'])

    def test_an_unreachable_peer_is_still_offline_regardless(self):
        with mock.patch('peer.registry.registry.get', return_value=None), \
             mock.patch('peer.registry.registry.endpoints_for', return_value=[]), \
             mock.patch('peer.dds_state.get_peer_topics', return_value={'a' * 32: {'agent_running': True}}):
            self.assertFalse(liveness.liveness(_peer('a' * 32, 'A', time.time() - 9999))['online'])


class TestPromptRendering(unittest.TestCase):
    def _snapshot(self, peers, adverts=None, shared=None):
        import prompt
        adverts = adverts or {}
        with mock.patch('peer.store.list_peers', return_value=peers), \
             mock.patch('peer.registry.registry.get', side_effect=lambda pid: adverts.get(pid)), \
             mock.patch('peer.registry.registry.endpoints_for', return_value=['https://x']), \
             mock.patch('peer.dds_state.get_peer_topics', return_value=shared or {}), \
             mock.patch('task_store.active_tasks', return_value=[]), \
             mock.patch('collector.get_available_sources', return_value=[]):
            return prompt._env_dynamic()

    def test_agent_off_is_marked_on_a_reachable_peer(self):
        out = self._snapshot([_peer('a' * 32, 'Live', time.time() - 2)],
                             shared={'a' * 32: {'agent_running': False}})
        self.assertIn('online="yes"', out)
        self.assertIn('agent="off"', out)

    def test_unknown_agent_state_adds_no_attribute(self):
        out = self._snapshot([_peer('a' * 32, 'Live', time.time() - 2)])
        self.assertIn('online="yes"', out)
        # 属性形式才算；hint 文案里本来就写着 agent=off 在解释这个属性的含义。
        self.assertNotIn('agent="', out)

    def test_online_and_offline_are_both_shown_and_marked(self):
        """离线的不能省掉：agent 要能回答"有一台但联系不上"，而不是"没有"。"""
        out = self._snapshot([_peer('a' * 32, 'Live', time.time() - 2),
                              _peer('b' * 32, 'Dead', time.time() - 9999)])
        self.assertIn('name="Live" role="operator" online="yes"', out)
        self.assertIn('name="Dead"', out)
        self.assertIn('online="no"', out)
        self.assertIn('last_contact=', out)

    def test_no_peer_id_in_the_snapshot(self):
        """32 位十六进制是这一行里最贵的东西，而 peer_delegate 接受名字。"""
        out = self._snapshot([_peer('a' * 32, 'Live', time.time() - 2)])
        self.assertNotIn('a' * 32, out)

    def test_online_peers_come_first(self):
        out = self._snapshot([_peer('b' * 32, 'Dead', time.time() - 9999),
                              _peer('a' * 32, 'Live', time.time() - 2)])
        self.assertLess(out.index('Live'), out.index('Dead'))

    def test_overflow_is_collapsed_rather_than_dumped(self):
        peers = [_peer(f'{i:02d}' + 'c' * 30, f'P{i}', time.time() - 9999) for i in range(9)]
        out = self._snapshot(peers)
        self.assertIn('peer_list', out)
        self.assertIn('另有 3 个', out)

    def test_colliding_names_are_distinguishable_in_the_snapshot(self):
        out = self._snapshot([_peer('8bd2bf8f' + '0' * 24, 'Orin6', time.time() - 2),
                              _peer('dd398c73' + '0' * 24, 'Orin6', time.time() - 2)])
        self.assertIn('Orin6#8bd2bf8f', out)
        self.assertIn('Orin6#dd398c73', out)


if __name__ == '__main__':
    unittest.main()
