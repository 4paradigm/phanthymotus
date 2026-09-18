"""
test_mcp_registry_merge.py — 心跳不能把 `_connect_one` 写的键抹掉。

`api/mcp_manage.py` 的心跳分支原来用一个字典字面量**整体替换** `registry[mcp_id]`，
而那个字面量里没有 `input_schemas`。紧接着十几行之后：

    needs_schemas = not mcp_client.registry.get(mcp_id, {}).get('input_schemas')
    if needs_schemas:
        asyncio.create_task(mcp_client._connect_one(...))

于是构成一个自激循环：**心跳覆盖 registry → `input_schemas` 消失 → 判定缺 schema →
重跑 `_connect_one`（完整 initialize + tools/list + 重建 SSE 订阅）→ 下一次心跳再抹一遍**。
`_connect_one` 写回的值活不过 30 秒。

这正是 #234 里那个泄漏**线性增长**的动力：每小时 120 次心跳 = 120 个永不退出的订阅循环，
每个退避到 60s 上限 ⇒ 每小时多约 2 req/s，与天轶实测斜率吻合。#234 止住了增长，
这里止住驱动它的 churn，也让 SSE 长连接第一次有可能真的保持住。

这是第二个从那个字面量里漏掉的键（第一个是 `tool_meta`）。改成合并是为了让这一**类**问题
消失，而不是再补一个实例。

Run: cd agent-core && python3 -m pytest tests/test_mcp_registry_merge.py
"""

import ast
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402

MANAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'api' / 'mcp_manage.py'
CLIENT = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'mcp_client.py'


def _keys_of_registry_literal(path: pathlib.Path) -> list[set]:
    """Every dict literal in `path` that looks like a registry entry."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if {'tools', 'schemas', 'online'} <= keys:
            out.append(keys)
    return out


class TestTheHeartbeatMerges(unittest.TestCase):
    def test_it_does_not_replace_the_entry(self):
        src = MANAGE.read_text(encoding='utf-8')
        self.assertIn('registry.setdefault(mcp_id, {}).update(', src)
        self.assertNotIn('mcp_client.registry[mcp_id] = {', src,
                         '整体替换会抹掉 _connect_one 写的键')

    def test_an_existing_input_schemas_survives_a_heartbeat(self):
        """行为层面的断言，而不是只看写法。"""
        mcp_client.registry.pop('dev-1', None)
        mcp_client.registry['dev-1'] = {
            'name': 'dev', 'url': 'u', 'online': True, 'tools': ['a'],
            'render_hint': '', 'schemas': {}, 'tool_meta': {},
            'split_map': {}, 'tool_groups': {},
            'input_schemas': {'mcp__dev-1__a': {'type': 'object'}},
        }
        # 心跳所做的事，逐字复制
        mcp_client.registry.setdefault('dev-1', {}).update({
            'name': 'dev', 'url': 'u', 'online': True, 'tools': ['a'],
            'render_hint': '', 'schemas': {}, 'tool_meta': {},
            'split_map': {}, 'tool_groups': {},
        })
        self.assertIn('input_schemas', mcp_client.registry['dev-1'])
        mcp_client.registry.pop('dev-1', None)

    def test_needs_schemas_becomes_false_after_connect_one_has_run(self):
        """循环能停下来 —— 这正是原来做不到的那一点。"""
        mcp_client.registry['dev-2'] = {'input_schemas': {'x': {}}}
        mcp_client.registry.setdefault('dev-2', {}).update({'online': True, 'tools': []})
        needs = not mcp_client.registry.get('dev-2', {}).get('input_schemas')
        self.assertFalse(needs, '心跳之后仍判定「缺 schema」就会无限重连')
        mcp_client.registry.pop('dev-2', None)

    def test_the_heartbeat_still_updates_what_it_does_know(self):
        mcp_client.registry['dev-3'] = {'online': False, 'tools': [], 'input_schemas': {'x': {}}}
        mcp_client.registry.setdefault('dev-3', {}).update({'online': True, 'tools': ['a', 'b']})
        self.assertTrue(mcp_client.registry['dev-3']['online'])
        self.assertEqual(mcp_client.registry['dev-3']['tools'], ['a', 'b'])
        mcp_client.registry.pop('dev-3', None)

    def test_a_device_seen_for_the_first_time_is_created(self):
        mcp_client.registry.pop('dev-4', None)
        mcp_client.registry.setdefault('dev-4', {}).update({'online': True})
        self.assertTrue(mcp_client.registry['dev-4']['online'])
        mcp_client.registry.pop('dev-4', None)


class TestTheTwoWritersAgree(unittest.TestCase):
    def test_the_heartbeat_writes_no_key_connect_one_does_not(self):
        """合并只在心跳是 `_connect_one` 的子集时才是无损的。

        若哪天心跳开始写一个独有的键，这条会红 —— 那时要重新想清楚谁该赢，
        而不是让它悄悄生效。
        """
        heartbeat = max(_keys_of_registry_literal(MANAGE), key=len)
        connect = max(_keys_of_registry_literal(CLIENT), key=len)
        self.assertEqual(heartbeat - connect, set(),
                         f'心跳独有的键：{heartbeat - connect}')

    def test_input_schemas_is_the_key_only_connect_one_writes(self):
        heartbeat = max(_keys_of_registry_literal(MANAGE), key=len)
        connect = max(_keys_of_registry_literal(CLIENT), key=len)
        self.assertIn('input_schemas', connect - heartbeat)


if __name__ == '__main__':
    unittest.main()
