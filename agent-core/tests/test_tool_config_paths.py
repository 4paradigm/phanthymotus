"""tool_config 的 saved-config 计划器：三条下发路径共用的一个决策点。

`plan_saved_config_calls` 是 canvas 保存补发之外所有 config 推送（重连恢复、
dashboard tools/call 自动补发、LLM 桥 start 补发）的唯一行加载/寻址/必填决策点。
此前这三处各写一遍，且已经漂移：恢复路径把 instance 键不带寻址地合并发（docstring
说不会），LLM 桥在行存在时完全跳过必填检查。这里钉住计划器的行为，而不是三个
调用点的 HTTP 细节——调用点现在薄到只剩「取计划、发请求」。
"""
import json

import pytest

import config
from tool_config import (config_result_adapter_ok, plan_saved_config_calls,
                         plan_config_calls)


SCHEMA = {
    'type': 'object',
    'properties': {
        'url': {'type': 'string'},
        'key': {'type': 'string'},
        'fps': {'type': 'number', 'scope': 'instance'},
        'retired_key': {'type': 'string'},  # 会在第二组用例里从 schema 删掉
    },
    'required': ['url'],
}

SCHEMA_NO_REQUIRED = {'type': 'object', 'properties': {'fps': {'type': 'number'}}}


@pytest.fixture
def rows(monkeypatch):
    """把 config.main 换成裸 dict——计划器只读 .get。"""
    fake = {}
    monkeypatch.setattr(config, 'main', fake)
    return fake


def _register(monkeypatch, tools):
    monkeypatch.setattr(config, 'main', {})  # rows() 先重置，再由 services.mcp 提供工具表
    monkeypatch.setattr(config, 'main', _ConfigFake(tools))


class _ConfigFake(dict):
    """services.mcp 提供工具表、其余当 config 行的 dict。"""

    def __init__(self, tools):
        super().__init__()
        self._tools = tools

    def get(self, key, default=None):
        if key == 'services':
            return {'mcp': self._tools}
        return super().get(key, default)


def _tool(mcp_id, name, schema):
    return {'id': mcp_id, 'tools': [{'name': name, 'configSchema': schema}]}


# ── 行加载与寻址 ─────────────────────────────────────────────────────────────

def test_instance_rows_are_addressed_when_the_call_has_an_instance(rows, monkeypatch):
    cfg = _ConfigFake([_tool('m1', 'cam', SCHEMA)])
    monkeypatch.setattr(config, 'main', cfg)
    cfg.update({
        'tool_config:m1:cam': {'url': 'http://x', 'key': 'k'},
        'tool_config:m1:cam:i7': {'fps': 12},
    })
    calls, missing, _ = plan_saved_config_calls('m1', 'cam', 'i7')
    assert calls == [
        ({'url': 'http://x', 'key': 'k'}, {}),
        ({'fps': 12}, {'instance_id': 'i7'}),
    ]
    assert missing == []


def test_without_an_instance_both_rows_merge_unaddressed(rows, monkeypatch):
    cfg = _ConfigFake([_tool('m1', 'cam', SCHEMA)])
    monkeypatch.setattr(config, 'main', cfg)
    cfg.update({
        'tool_config:m1:cam': {'url': 'http://x', 'fps': 1},
        'tool_config:m1:cam:i7': {'fps': 12},   # 没有 instance_id，不该读这行
    })
    calls, _missing, _ = plan_saved_config_calls('m1', 'cam')
    assert calls == [({'url': 'http://x', 'fps': 1}, {})]


def test_stale_keys_are_dropped_not_replayed(rows, monkeypatch):
    old_schema = dict(SCHEMA, properties=dict(SCHEMA['properties'], gone={'type': 'string'}))
    cfg = _ConfigFake([_tool('m1', 'cam', SCHEMA)])
    monkeypatch.setattr(config, 'main', cfg)
    cfg['tool_config:m1:cam'] = {'url': 'http://x', 'gone': '历史残留'}
    calls, _missing, _ = plan_saved_config_calls('m1', 'cam')
    assert calls == [({'url': 'http://x'}, {})]


# ── 必填检查：这是 LLM 桥曾经漏掉的那个分支 ────────────────────────────────

def test_missing_required_surfaces_even_when_a_row_exists(rows, monkeypatch):
    """行存在但必填键为空：修复前 LLM 桥看到行存在就跳过检查，照常启动。"""
    cfg = _ConfigFake([_tool('m1', 'cam', SCHEMA)])
    monkeypatch.setattr(config, 'main', cfg)
    cfg['tool_config:m1:cam'] = {'url': '', 'key': 'k'}
    _calls, missing, _ = plan_saved_config_calls('m1', 'cam')
    assert missing == ['url']


def test_no_schema_yields_no_calls_and_no_missing(rows, monkeypatch):
    cfg = _ConfigFake([_tool('m1', 'cam', None)])
    monkeypatch.setattr(config, 'main', cfg)
    cfg['tool_config:m1:cam'] = {'anything': 1}
    calls, missing, _ = plan_saved_config_calls('m1', 'cam')
    assert calls == [] and missing == []


def test_missing_is_computed_on_the_view_that_will_be_pushed(rows, monkeypatch):
    """计划要发的和必填要查的必须是同一份合并视图，否则拒绝与推送各说各话。"""
    cfg = _ConfigFake([_tool('m1', 'cam', SCHEMA)])
    monkeypatch.setattr(config, 'main', cfg)
    cfg.update({
        'tool_config:m1:cam': {'url': 'http://x'},
        'tool_config:m1:cam:i7': {'fps': 3},
    })
    calls, missing, _ = plan_saved_config_calls('m1', 'cam', 'i7')
    pushed = {}
    for body, _extra in calls:
        pushed.update(body)
    assert pushed == {'url': 'http://x', 'fps': 3}
    assert missing == []


# ── adapter_ok 判定 ──────────────────────────────────────────────────────────

def test_adapter_ok_false_is_detected():
    result = {'content': [{'type': 'text', 'text': json.dumps({'adapter_ok': False})}]}
    assert config_result_adapter_ok(result) is False


def test_garbage_acks_default_to_ok():
    """检查器自己解析失败不得发明一种新的启动失败。"""
    assert config_result_adapter_ok({'content': [{'text': 'not json'}]}) is True
    assert config_result_adapter_ok({}) is True
    assert config_result_adapter_ok(None) is True


def test_plan_config_calls_still_exported_for_the_canvas_save_path():
    """canvas 的保存补发走裸 plan_config_calls（推的是刚保存的体，不是存档行）。"""
    tool = {'configSchema': SCHEMA}
    assert plan_config_calls(tool, {'url': 'u'}, {}, '') == [({'url': 'u'}, {})]
