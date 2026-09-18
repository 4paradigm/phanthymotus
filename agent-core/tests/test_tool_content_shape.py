"""
test_tool_content_shape.py — a tool role message's `content` must be a shape the
model service accepts.

Background (Tianyi, 2026-09-18): the agent loop built its tool messages as
`{'role': 'tool', 'tool_call_id': ..., 'content': r['result']}` with `r['result']`
straight from `_dispatch`. Most results are strings, but the barrier branches
return a dict — `{"status": "not_dispatched", "reason": ...}` — and that dict went
into `content` verbatim. The router answered 400 with
`UPSTREAM_PASSTHROUGH: The model service rejected this request.`, which names no
field, so the error reads like a flaky upstream.

It is neither flaky nor transient. It is deterministic, and it sticks: the bad
message stays in `turn_messages`, so every subsequent request in that turn carries
it and fails too — one interrupted barrier turns into a burst of 400s. Measured
over the full request log on that machine: 24356 requests, 35 of them carrying a
dict `content`, and **zero** of those 35 ever got a response.

Run: cd agent-core && python3 -m pytest tests/test_tool_content_shape.py
"""
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from event.llm import _tool_content  # noqa: E402


def test_string_result_passes_through_untouched():
    assert _tool_content('已播放完成') == '已播放完成'
    assert _tool_content('') == ''


def test_multimodal_list_passes_through_untouched():
    """Image results are a content-part array — also a legal shape, keep it."""
    parts = [{'type': 'text', 'text': 'see'},
             {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,AAAA'}}]
    assert _tool_content(parts) is parts


def test_not_dispatched_dict_becomes_a_json_string():
    """The exact payload that produced the 400s."""
    result = {'status': 'not_dispatched',
              'reason': 'barrier wait interrupted before dispatch (status=reconsidering)'}
    content = _tool_content(result)
    assert isinstance(content, str)
    assert json.loads(content) == result


def test_dict_serialisation_keeps_chinese_readable():
    """ensure_ascii=False — the model reads this text, \\u4f60\\u597d is not text."""
    content = _tool_content({'status': 'not_dispatched', 'reason': '新消息到达，本次 finish 已取消'})
    assert '新消息到达' in content


def test_none_becomes_empty_string_not_the_word_none():
    assert _tool_content(None) == ''


def test_unserialisable_object_still_yields_a_string():
    """A diagnostic must never be the thing that kills the turn."""
    class Weird:
        def __repr__(self):
            return '<weird>'

    assert _tool_content(Weird()) == '<weird>'


def test_every_result_shape_dispatch_can_return_is_a_legal_content():
    """Belt and braces: whatever comes back, the message is sendable."""
    for result in ['ok', '', None, 42, 1.5, True, {'a': 1}, [{'type': 'text', 'text': 'x'}]]:
        content = _tool_content(result)
        assert isinstance(content, (str, list)), f'{result!r} → {content!r}'
