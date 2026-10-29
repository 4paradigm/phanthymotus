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

    def test_the_reconnect_loop_can_terminate(self):
        """循环能停下来 —— 这正是原来做不到的那一点。"""
        mcp_client.registry['dev-2'] = {'connected': True}
        mcp_client.registry.setdefault('dev-2', {}).update({'online': True, 'tools': []})
        self.assertTrue(mcp_client.registry['dev-2']['connected'],
                        '心跳之后仍判定「没连过」就会无限重连')
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

    def test_the_heartbeat_now_maintains_input_schemas_itself(self):
        """合并挡住了「被抹掉」，但没有解决「不再刷新」。

        合并之后 `needs_schemas` 恒为假，`_connect_one` 不再重跑，于是驱动改了某个工具的
        inputSchema 时，参数校验会一直用旧的那份，直到 agent-core 重启 —— 之前那个每 30 秒
        的 churn 恰好在充当刷新。所以心跳必须自己算出这个键，而不只是保住它。
        """
        heartbeat = max(_keys_of_registry_literal(MANAGE), key=len)
        self.assertIn('input_schemas', heartbeat)

    def test_connected_is_what_only_connect_one_writes(self):
        """触发全量连接的判据是「连过没有」，不是「缺哪个键」。

        旧判据（`input_schemas` 缺失）只在心跳先抹掉它时才成立。心跳现在会填它，
        那个判据就会在第一次 ping 时即为假 —— 而 `_connect_one` 是**唯一**启动 SSE
        订阅的地方，只靠心跳注册的设备将永远拿不到订阅。
        """
        heartbeat = max(_keys_of_registry_literal(MANAGE), key=len)
        connect = max(_keys_of_registry_literal(CLIENT), key=len)
        self.assertEqual(connect - heartbeat, {'connected'})
        src = MANAGE.read_text(encoding='utf-8')
        self.assertIn("get('connected')", src)
        # 只看可执行代码：旧判据的名字还留在解释历史的注释里，那是有意的。
        code = '\n'.join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith('#'))
        self.assertNotIn('needs_schemas =', code)


class TestConnectTriggerSurvivesAHeartbeatOnlyDevice(unittest.TestCase):
    """只靠心跳注册的设备也必须拿到一次全量连接。

    这是把判据从「缺 input_schemas」换成「没连过」的理由：心跳在检查之前就把
    registry 更新好了，所以任何「某个键在不在」的判据都会在第一次 ping 时即为假。
    """

    def _would_connect(self, entry):
        # 与 api/mcp_manage.py 中的判据保持一致
        return not (entry or {}).get('connected')

    def test_a_brand_new_device_triggers_a_connect(self):
        self.assertTrue(self._would_connect({}))

    def test_a_device_the_heartbeat_just_filled_still_triggers_one(self):
        """回归本身：心跳写了 input_schemas，不能因此就算「连过了」。"""
        self.assertTrue(self._would_connect({'input_schemas': {'x': {}}, 'online': True}))

    def test_a_connected_device_does_not_reconnect_every_heartbeat(self):
        self.assertFalse(self._would_connect({'connected': True, 'input_schemas': {'x': {}}}))

    def test_a_failed_connect_is_retried(self):
        """`connected` 取的是 online —— 没真正够到设备就不算连过。"""
        self.assertTrue(self._would_connect({'connected': False}))


if __name__ == '__main__':
    unittest.main()
