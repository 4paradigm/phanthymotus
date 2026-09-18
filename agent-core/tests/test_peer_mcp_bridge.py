"""
test_peer_mcp_bridge.py — 出向工具代理：本机 LLM 调对端工具。

接收侧（/api/peer/tools/{list,call}）先建好了，发送侧一直是空的 —— 工具清单取得到，
却没有任何东西把它交给本机 LLM，"调另一台机器人的摄像头"根本无法表达。

这里守三件容易悄悄坏掉的事：

1. **合成 mcp_id 里不能有 `__`**。call_tool 用 `split('__', 2)` 拆全名，id 里带下划线
   会被从中间劈开，症状是"工具名格式错误"而不是"peer 不可达"。
2. **本地别名与远端名字的映射**。对端的 tools/list 报的是它自己的
   `mcp__<它的 id>__<tool>`，而它的 /tools/call 也只认这个名字；两边都翻译很容易漂移，
   所以远端名字要原样留着。
3. **peer 工具不受本机画布闸门约束**。画布是本机操作者对**本机**暴露面的授权；跑在
   对端的工具由对端把关。但入向 peer 请求点的是**本机**工具，绝不能被这个豁免带上。
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

import canvas_binding  # noqa: E402
import mcp_client  # noqa: E402
from peer import mcp_bridge  # noqa: E402


PEER_ID = '8bd2bf8fe8f872361f8f3212dc7e4279'
REMOTE_TOOL = 'mcp__mcp-1783770461__tts'


def _peer(role='operator'):
    return {'peer_id': PEER_ID, 'display_name': 'Orin6', 'role': role, 'tool_filter': '*'}


def _tools_list_response():
    return {'tools': [{'name': REMOTE_TOOL, 'description': 'TTS — speak text',
                       'parameters': {'type': 'object', 'properties': {}}}], 'count': 1}


class TestNaming(unittest.TestCase):
    def test_synthetic_id_has_no_double_underscore(self):
        """call_tool 按 `__` 拆全名，id 里带一个就会被劈开。"""
        mcp_id = mcp_bridge.mcp_id_for(PEER_ID)
        self.assertNotIn('__', mcp_id)
        full = f'mcp__{mcp_id}__tts'
        self.assertEqual(full.split('__', 2), ['mcp', mcp_id, 'tts'])

    def test_peer_id_resolves_from_the_prefix(self):
        with mock.patch('peer.store.list_peers', return_value=[_peer()]):
            self.assertEqual(mcp_bridge.peer_id_of(mcp_bridge.mcp_id_for(PEER_ID)), PEER_ID)
            self.assertEqual(mcp_bridge.peer_id_of('mcp-1234'), '')


class TestAliasCollision(unittest.TestCase):
    """同名工具必须各自可达。

    真机上 Orin6 同时提供 perception 的 tts 和驱动的 tts —— 两者短名相同。原来的别名
    直接用短名，后者把前者覆盖掉，前一个工具在本机就此不存在，而且没有任何报错。
    """

    def test_colliding_short_names_get_distinct_aliases(self):
        remotes = ['mcp__mcp-1783770461__tts', 'mcp__mcp-1783771428__tts']
        aliases = mcp_bridge._aliases(remotes)
        self.assertEqual(len(set(aliases.values())), 2, '两个 tts 必须映射到不同别名')
        for a in aliases.values():
            self.assertTrue(a.startswith('tts'))
            self.assertNotIn('__', a, '别名里出现 __ 会被 call_tool 从中间劈开')

    def test_unique_names_pay_nothing(self):
        aliases = mcp_bridge._aliases(['mcp__m1__camera_main', 'mcp__m1__battery'])
        self.assertEqual(set(aliases.values()), {'camera_main', 'battery'})

    def test_every_alias_maps_back_to_exactly_one_remote(self):
        remotes = ['mcp__a__tts', 'mcp__b__tts', 'mcp__b__ocr']
        aliases = mcp_bridge._aliases(remotes)
        self.assertEqual(len(set(aliases.values())), len(remotes))


class TestRefresh(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for k in [k for k in mcp_client.registry if k.startswith('peer:')]:
            mcp_client.registry.pop(k)

    async def _refresh(self, peer=None, resp=None, endpoints=('https://x',)):
        async def _get(eps, path, **kw):
            return (resp if resp is not None else _tools_list_response()), ''
        with mock.patch('peer.registry.registry.endpoints_for', return_value=list(endpoints)), \
             mock.patch('peer.transport.get_json', side_effect=_get):
            return await mcp_bridge.refresh_one(peer or _peer())

    async def test_two_tools_with_the_same_short_name_both_survive(self):
        resp = {'tools': [
            {'name': 'mcp__mcp-1783770461__tts', 'description': 'perception tts',
             'parameters': {'type': 'object', 'properties': {}}},
            {'name': 'mcp__mcp-1783771428__tts', 'description': 'driver tts',
             'parameters': {'type': 'object', 'properties': {}}},
        ]}
        n = await self._refresh(resp=resp)
        self.assertEqual(n, 2)
        entry = mcp_client.registry[mcp_bridge.mcp_id_for(PEER_ID)]
        self.assertEqual(len(entry['schemas']), 2, '一个 tts 被另一个覆盖了')
        self.assertEqual(len(set(entry['remote_names'].values())), 2)

    async def test_registers_the_peers_tools_under_a_local_alias(self):
        n = await self._refresh()
        self.assertEqual(n, 1)
        entry = mcp_client.registry[mcp_bridge.mcp_id_for(PEER_ID)]
        local = f'mcp__{mcp_bridge.mcp_id_for(PEER_ID)}__tts'
        self.assertIn(local, entry['schemas'])
        self.assertEqual(entry['remote_names'][local], REMOTE_TOOL,
                         '远端名字必须原样留着，它的 /tools/call 只认这个')
        self.assertEqual(entry['transport'], 'peer')
        self.assertTrue(entry['online'])

    async def test_acp_meta_crosses_the_boundary(self):
        """completion / resource 要跟着 tools/list 过来。

        `tools` 是 OpenAI function-calling schema，它的 `parameters` 不是 MCP 的
        `inputSchema`，所以 x-completion / x-resource 不在里面。以前 tool_meta 被填成
        `{}`，于是每个 peer 工具都算"未声明"，而未声明 = 与一切互斥 —— 调一个 peer 工具
        会挡住本机全部执行。
        """
        resp = {
            'tools': [{'name': REMOTE_TOOL, 'description': 'tts',
                       'parameters': {'type': 'object', 'properties': {}}}],
            'acp_meta': {REMOTE_TOOL: {'completion': {'actions': ['speak'],
                                                      'timeout': 60},
                                       'resource': ['mouth']}},
        }
        await self._refresh(resp=resp)
        entry = mcp_client.registry[mcp_bridge.mcp_id_for(PEER_ID)]
        local = f'mcp__{mcp_bridge.mcp_id_for(PEER_ID)}__tts'
        meta = entry['tool_meta'][local]
        self.assertEqual(meta['resource'], frozenset({'mouth'}))
        self.assertEqual(meta['completion']['actions'], ['speak'])
        # 声明了资源之后，说话不该再挡住底盘
        self.assertFalse(mcp_client.resources_conflict(
            frozenset({'base'}), meta['resource']))

    async def test_missing_acp_meta_stays_conservative(self):
        """旧版 peer 不发这个字段 —— 必须退回"与一切互斥"，不能当成"不冲突"。"""
        await self._refresh()   # 默认 fixture 不带 acp_meta
        entry = mcp_client.registry[mcp_bridge.mcp_id_for(PEER_ID)]
        local = f'mcp__{mcp_bridge.mcp_id_for(PEER_ID)}__tts'
        self.assertIsNone(entry['tool_meta'][local]['resource'])
        self.assertTrue(mcp_client.resources_conflict(
            frozenset({'base'}), entry['tool_meta'][local]['resource']))

    async def test_malformed_remote_resource_falls_back_to_conservative(self):
        """对端可以发任意内容；打错的声明不能静默解锁并行执行。"""
        resp = {
            'tools': [{'name': REMOTE_TOOL, 'description': 'tts',
                       'parameters': {'type': 'object', 'properties': {}}}],
            'acp_meta': {REMOTE_TOOL: {'resource': {'mouth': True}}},
        }
        await self._refresh(resp=resp)
        entry = mcp_client.registry[mcp_bridge.mcp_id_for(PEER_ID)]
        local = f'mcp__{mcp_bridge.mcp_id_for(PEER_ID)}__tts'
        self.assertIsNone(entry['tool_meta'][local]['resource'])

    async def test_description_says_which_robot(self):
        await self._refresh()
        entry = mcp_client.registry[mcp_bridge.mcp_id_for(PEER_ID)]
        self.assertIn('Orin6', next(iter(entry['schemas'].values()))['description'])

    async def test_blocked_peer_is_removed_not_merely_skipped(self):
        await self._refresh()
        await self._refresh(peer=_peer(role='blocked'))
        self.assertNotIn(mcp_bridge.mcp_id_for(PEER_ID), mcp_client.registry)

    async def test_unreachable_keeps_the_tools_but_marks_offline(self):
        """链路闪一下不该让模型重建计划 —— 联系不上 ≠ 没有工具。"""
        await self._refresh()
        await self._refresh(endpoints=())
        entry = mcp_client.registry[mcp_bridge.mcp_id_for(PEER_ID)]
        self.assertFalse(entry['online'])
        self.assertEqual(len(entry['schemas']), 1)

    async def test_a_failing_peer_does_not_abort_the_round(self):
        async def _boom(*a, **kw):
            raise RuntimeError('down')
        peers = [_peer(), {**_peer(), 'peer_id': 'ff' * 16, 'display_name': 'Other'}]
        with mock.patch('peer.store.list_peers', return_value=peers), \
             mock.patch('peer.registry.registry.endpoints_for', return_value=['https://x']), \
             mock.patch('peer.transport.get_json', side_effect=_boom):
            self.assertEqual(await mcp_bridge.refresh_all(), 0)


class TestCall(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mcp_id = mcp_bridge.mcp_id_for(PEER_ID)
        mcp_client.registry[self.mcp_id] = {
            'name': 'Orin6 (peer)', 'transport': 'peer', 'peer_id': PEER_ID,
            'online': True, 'schemas': {}, 'tools': ['tts'],
            'remote_names': {f'mcp__{self.mcp_id}__tts': REMOTE_TOOL},
        }
        self.addCleanup(mcp_client.registry.pop, self.mcp_id, None)

    async def _call(self, tool='tts', resp=None, reason='', endpoints=('https://x',)):
        seen = {}

        async def _post(eps, path, payload, **kw):
            seen.update({'path': path, 'payload': payload})
            return resp, reason
        with mock.patch('peer.store.list_peers', return_value=[_peer()]), \
             mock.patch('peer.registry.registry.endpoints_for', return_value=list(endpoints)), \
             mock.patch('peer.transport.post_json', side_effect=_post):
            out = await mcp_bridge.call(self.mcp_id, tool, {'action': 'speak'})
        return out, seen

    async def test_sends_the_remote_name_not_the_local_alias(self):
        out, seen = await self._call(resp={'result': 'ok', 'error': None})
        self.assertEqual(seen['path'], '/api/peer/tools/call')
        self.assertEqual(seen['payload']['tool_name'], REMOTE_TOOL)
        self.assertEqual(out, 'ok')

    async def test_a_refusal_from_the_far_side_is_passed_through(self):
        """对端的拒绝理由才说明是角色、过滤器还是它的画布挡下的。"""
        out, _ = await self._call(resp=None, reason='HTTP 403 tool call denied: viewer')
        self.assertIn('403', out)
        self.assertIn('viewer', out)

    async def test_an_error_field_is_reported(self):
        out, _ = await self._call(resp={'result': None, 'error': 'TTS not running'})
        self.assertIn('TTS not running', out)

    async def test_unknown_tool_says_the_list_may_have_changed(self):
        out, _ = await self._call(tool='nope')
        self.assertIn('nope', out)
        self.assertIn('peer_list', out)


class TestCanvasGateExemption(unittest.TestCase):
    def test_peer_tools_bypass_the_local_canvas_gate(self):
        with mock.patch.object(canvas_binding, 'bound_tool_names', return_value=set()):
            self.assertTrue(canvas_binding.is_bound('mcp__peer:8bd2bf8fe8f8__tts'))

    def test_local_tools_still_need_a_card(self):
        """入向 peer 请求点的是本机工具 —— 豁免绝不能带上它们。"""
        with mock.patch.object(canvas_binding, 'bound_tool_names', return_value=set()):
            self.assertFalse(canvas_binding.is_bound('mcp__mcp-drv__loco'))

    def test_the_predicate_only_matches_the_peer_prefix(self):
        self.assertTrue(canvas_binding.is_peer_tool('mcp__peer:abc__x'))
        self.assertFalse(canvas_binding.is_peer_tool('mcp__peering__x'))
        self.assertFalse(canvas_binding.is_peer_tool('mcp__mcp-1__peer'))


class TestPeerToolsAreNotExpandedIntoTheToolList(unittest.TestCase):
    """peer 工具**不能**进 LLM 的工具列表 —— 固定两个通用工具替代它。

    展开过一版，量出来不成立：单个 schema 实测 613 字符（≈200 token），一台连线完整的
    机器人绑 8–15 个工具，十台 peer 就是 ~100 个工具、每次请求多 ~20k token。而且在
    上下文上限之前先坏两件事 —— 工具列表太长导致选错，以及工具列表位于缓存前缀里，
    peer 上下线或重新广告都会把它改写、让缓存全部失效。

    所以改成 peer_tools(peer) + peer_call(peer, tool, arguments_json)：都带 peer 参数，
    无论多少台机器都只占两个 schema。
    """

    def _schemas(self, registry_patch):
        import config
        from event.llm import Event
        with mock.patch.dict(mcp_client.registry, registry_patch, clear=True), \
             mock.patch.object(config, 'main',
                               {'canvas_layout': {'cards': [], 'execConnections': []}}):
            return Event._get_bound_tool_schemas(Event.__new__(Event))

    def test_a_peers_tools_are_not_appended(self):
        mcp_id = 'peer:8bd2bf8fe8f8'
        local = f'mcp__{mcp_id}__tts'
        entry = {'transport': 'peer', 'online': True,
                 'schemas': {local: {'name': local, 'description': '[Orin6] TTS'}}}
        self.assertEqual(self._schemas({mcp_id: entry}), [])

    def test_local_tools_still_require_a_canvas_connection(self):
        entry = {'transport': 'http', 'online': True,
                 'schemas': {'mcp__mcp-1__loco': {'name': 'mcp__mcp-1__loco'}}}
        self.assertEqual(self._schemas({'mcp-1': entry}), [])

    def test_the_pair_is_registered_as_system_tools(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / 'src/event/llm.py').read_text()
        self.assertIn("('peer_tools', _peer_delegation.peer_tools)", src)
        self.assertIn("('peer_call', _peer_delegation.peer_call)", src)

    def test_the_cost_is_flat_in_fleet_size(self):
        """无论几台 peer，追加的 schema 数都是 0 —— 这就是"永远两个工具"的含义。"""
        many = {
            f'peer:{i:012d}': {'transport': 'peer', 'online': True,
                               'schemas': {f'mcp__peer:{i:012d}__t{j}': {'name': f't{j}'}
                                           for j in range(15)}}
            for i in range(30)
        }
        self.assertEqual(self._schemas(many), [])


class TestDispatchRouting(unittest.TestCase):
    def test_call_tool_routes_peer_transport_to_the_bridge(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / 'src/mcp_client.py').read_text()
        self.assertIn("if info.get('transport') == 'peer':", src)
        self.assertIn('mcp_bridge.call(mcp_id, tool_name, args)', src)


if __name__ == '__main__':
    unittest.main()


class TestGenericPeerTools(unittest.IsolatedAsyncioTestCase):
    """peer_tools / peer_call —— 固定两个工具覆盖任意规模的机群。"""

    def setUp(self):
        self.mcp_id = mcp_bridge.mcp_id_for(PEER_ID)
        local = f'mcp__{self.mcp_id}__tts'
        mcp_client.registry[self.mcp_id] = {
            'name': 'Orin6 (peer)', 'transport': 'peer', 'online': True,
            'tools': ['tts'],
            'schemas': {local: {'name': local, 'description': '[Orin6] TTS — speak text',
                                'parameters': {'type': 'object', 'properties': {
                                    'action': {'type': 'string', 'enum': ['speak']},
                                    'text': {'type': 'string'}}}}},
            'remote_names': {local: REMOTE_TOOL},
        }
        self.addCleanup(mcp_client.registry.pop, self.mcp_id, None)

    async def _tools(self, peer='Orin6'):
        from peer import delegation
        with mock.patch('peer.store.list_peers', return_value=[_peer()]):
            return await delegation.peer_tools(peer)

    async def _call(self, **kw):
        from peer import delegation
        seen = {}

        async def _ct(full_name, args):
            seen.update({'full_name': full_name, 'args': args})
            return 'ok'
        with mock.patch('peer.store.list_peers', return_value=[_peer()]), \
             mock.patch('mcp_client.call_tool', side_effect=_ct):
            out = await delegation.peer_call(**kw)
        return out, seen

    async def test_tools_lists_names_and_parameters(self):
        out = await self._tools()
        self.assertIn('tts', out)
        self.assertIn('parameters', out)
        self.assertIn('peer_call', out, '要告诉模型下一步用什么调它')

    async def test_tools_rejects_an_unknown_peer_with_the_options(self):
        out = await self._tools(peer='Nope')
        self.assertIn('Error', out)
        self.assertIn('Orin6', out)

    async def test_call_goes_through_call_tool_not_around_it(self):
        """走 call_tool 才能沿用同一套历史与 ACP 处理。"""
        out, seen = await self._call(peer='Orin6', tool='tts',
                                     arguments_json='{"action": "speak", "text": "hi"}')
        self.assertEqual(out, 'ok')
        self.assertEqual(seen['full_name'], f'mcp__{self.mcp_id}__tts')
        self.assertEqual(seen['args'], {'action': 'speak', 'text': 'hi'})

    async def test_call_explains_a_bad_json_payload(self):
        out, seen = await self._call(peer='Orin6', tool='tts', arguments_json='action=speak')
        self.assertIn('JSON', out)
        self.assertEqual(seen, {}, '解析失败不该发出请求')

    async def test_call_refuses_a_non_object_payload(self):
        out, _ = await self._call(peer='Orin6', tool='tts', arguments_json='["speak"]')
        self.assertIn('object', out)

    async def test_call_says_so_when_the_peer_offers_nothing(self):
        mcp_client.registry.pop(self.mcp_id, None)
        out, seen = await self._call(peer='Orin6', tool='tts', arguments_json='{}')
        self.assertIn('peer_tools', out)
        self.assertEqual(seen, {})

    async def test_default_arguments_are_an_empty_object(self):
        out, seen = await self._call(peer='Orin6', tool='tts')
        self.assertEqual(seen['args'], {})
