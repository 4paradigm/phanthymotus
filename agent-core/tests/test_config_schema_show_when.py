"""
test_config_schema_show_when.py — x-show-when 的值必须是字符串。

踩过的坑：给 decision_core 的两个主动播报阈值加了 `'x-show-when': {'auto_narration': True}`
（Python 布尔），字段在卡片上**永远不显示**，而且完全没有报错 —— 看起来就像功能没做。

原因在渲染侧（web/js/sidebar.js）：boolean 字段渲染成一个 `<select>`，option 的 value 是
**字符串** 'true'/'false'，而 `_condMatches` 做的是 `actual === condVal` 的严格相等。
条件值一旦序列化成 JSON 布尔 true，`"true" === true` 恒为 false，字段被永久隐藏。

在这之前 schema 里唯一的 x-show-when 比的是字符串（search_type: 'baidu_search'），
所以没人撞上过。

这条测试扫所有内建 configSchema，保证条件值都是字符串（或字符串数组 —— 渲染侧支持
`Array.includes(actual)` 表示"任一"）。隐藏的字段和没做的功能长得一模一样，不值得再靠
肉眼发现一次。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_config_schema_show_when.py
"""
import ast
import pathlib
import unittest

_SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'


def _literal_dicts_for_key(tree, key):
    """Yield every literal dict assigned to `key` anywhere in the module."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                try:
                    yield ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass


class TestShowWhenValuesAreStrings(unittest.TestCase):
    def _conditions(self):
        found = []
        for path in sorted(_SRC.rglob('*.py')):
            tree = ast.parse(path.read_text())
            for cond_key in ('x-show-when', 'x-hide-when'):
                for cond in _literal_dicts_for_key(tree, cond_key):
                    if isinstance(cond, dict):
                        found.append((path.relative_to(_SRC), cond_key, cond))
        return found

    def test_at_least_one_condition_exists(self):
        """扫不到任何条件说明这条测试本身失效了（比如 schema 搬了家）。"""
        self.assertTrue(self._conditions(), 'no x-show-when/x-hide-when found — test is dead')

    def test_all_condition_values_are_strings(self):
        bad = []
        for rel, cond_key, cond in self._conditions():
            for field, val in cond.items():
                vals = val if isinstance(val, list) else [val]
                for v in vals:
                    if not isinstance(v, str):
                        bad.append(f'{rel}: {cond_key}.{field} = {v!r} ({type(v).__name__})')
        self.assertEqual(bad, [], '条件值必须是字符串 —— sidebar.js 比的是 <select>.value，'
                                 '布尔 true 永远不等于字符串 "true"，字段会被静默隐藏：\n'
                                 + '\n'.join(bad))

    def test_decision_core_narration_fields_are_gated_on_the_string(self):
        """回归锚点：这个字段就是当初被布尔条件静默藏掉的那个。"""
        tree = ast.parse((_SRC / 'start.py').read_text())
        schemas = list(_literal_dicts_for_key(tree, 'configSchema'))
        props = {}
        for s in schemas:
            props.update(s.get('properties', {}))
        for key in ('narration_silence_seconds',):
            self.assertIn(key, props, f'{key} 不在 decision_core 的 configSchema 里')
            self.assertEqual(props[key].get('x-show-when'), {'auto_narration': 'true'})
            self.assertEqual(props[key]['type'], 'integer')


if __name__ == '__main__':
    unittest.main()
