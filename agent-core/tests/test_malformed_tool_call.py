"""glm 偶发返回的空 tool_call 不能进历史，也不能让已中毒的历史一直 400。

真实样本来自 bumi（10.100.129.141）resource/log/llm.json 的 messages[24]：
finish 之后跟了一条 {"id": "", "function": {"name": "", "arguments": "{}"}}。
字段齐全但值是空串，所以旧的「key 在不在」检查放过去了；落盘后每次请求
都被服务端以 messages[24].tool_calls[1].function missing required field "name" 拒掉。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from client.llm import LLMErrorKind, _classify_error, _valid_tool_call  # noqa: E402
from event.llm import _sanitize, _scrub  # noqa: E402

GOOD_CALL = {'id': 'call_60582d0190364c928ea901b6', 'type': 'function', 'index': 0,
             'function': {'name': 'finish', 'arguments': '{}'}}
EMPTY_CALL = {'id': '', 'index': 0, 'function': {'name': '', 'arguments': '{}'}}


class _Err(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        self.status_code = status


# ── 结构校验 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('tc', [
    EMPTY_CALL,
    {'id': 'x', 'function': {'name': '', 'arguments': '{}'}},   # name 空串
    {'id': '', 'function': {'name': 'finish'}},                 # id 空串
    {'id': 'x', 'function': {'arguments': '{}'}},               # 缺 name
    {'id': 'x'},                                                # 缺 function
    {'id': 'x', 'function': None},
    'not-a-dict',
])
def test_invalid_tool_calls_rejected(tc):
    assert not _valid_tool_call(tc)


def test_valid_tool_call_accepted():
    assert _valid_tool_call(GOOD_CALL)


# ── 历史清洗 ──────────────────────────────────────────────────────────────────

def test_scrub_drops_empty_call_keeps_good_one():
    history = [
        {'role': 'user', 'content': '站起来'},
        {'role': 'assistant', 'content': '', 'tool_calls': [GOOD_CALL, EMPTY_CALL]},
        {'role': 'tool', 'tool_call_id': GOOD_CALL['id'], 'content': 'ok'},
    ]
    out = _scrub(history)
    assert [m['role'] for m in out] == ['user', 'assistant', 'tool']
    assert out[1]['tool_calls'] == [GOOD_CALL]


def test_scrub_drops_orphaned_tool_result():
    """空 id 的调用被 dispatch 过，于是历史里还有一条 tool_call_id="" 的结果。
    调用没了，结果也必须走，否则换成 orphan tool message 继续 400。"""
    history = [
        {'role': 'assistant', 'content': '', 'tool_calls': [GOOD_CALL, EMPTY_CALL]},
        {'role': 'tool', 'tool_call_id': GOOD_CALL['id'], 'content': 'ok'},
        {'role': 'tool', 'tool_call_id': '', 'content': 'Unknown tool: '},
    ]
    out = _scrub(history)
    assert [m.get('tool_call_id') for m in out if m['role'] == 'tool'] == [GOOD_CALL['id']]


def test_scrub_drops_message_with_no_content_and_no_valid_call():
    history = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': '', 'tool_calls': [EMPTY_CALL]},
    ]
    assert _scrub(history) == [{'role': 'user', 'content': 'hi'}]


def test_scrub_keeps_text_when_only_call_was_invalid():
    history = [{'role': 'assistant', 'content': '我想想', 'tool_calls': [EMPTY_CALL]}]
    out = _scrub(history)
    assert out == [{'role': 'assistant', 'content': '我想想'}]


def test_scrub_leaves_clean_history_untouched():
    history = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': '', 'tool_calls': [GOOD_CALL]},
        {'role': 'tool', 'tool_call_id': GOOD_CALL['id'], 'content': 'ok'},
    ]
    assert _scrub(history) == history


def test_scrub_then_sanitize_composes():
    """_scrub 之后 _sanitize 仍应砍掉末尾没有结果回应的调用。"""
    history = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': '', 'tool_calls': [GOOD_CALL, EMPTY_CALL]},
    ]
    assert _sanitize(_scrub(history)) == [{'role': 'user', 'content': 'hi'}]


# ── 错误分类 ──────────────────────────────────────────────────────────────────

def test_payload_400_is_not_retried():
    """指名某条历史消息的 400 每次都一样，重试只是多撞两次。"""
    err = _Err('Error code: 400 - InvalidParameter: messages[24].tool_calls[1].'
               'function missing required field "name"', 400)
    kind, delay = _classify_error(err)
    assert kind == LLMErrorKind.UNKNOWN
    assert delay is None


def test_bad_arguments_400_still_retried():
    """模型把 arguments 写成非 JSON 是随机的，这种仍然值得重试。"""
    err = _Err('Error code: 400 - arguments must be in json format', 400)
    kind, delay = _classify_error(err)
    assert kind == LLMErrorKind.SERVER_ERROR
    assert delay == 1.0
