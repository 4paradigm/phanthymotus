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

import sys as _sys  # noqa: E402
from event.llm import _sanitise_turn, _tool_content, get_recent_context_rich  # noqa: E402

# `import event.llm as _ev` binds the Event *instance* that event/__init__ exports,
# not the module — take the module out of sys.modules so monkeypatch hits the right object.
_ev = _sys.modules['event.llm']

BAD_RESULT = {'status': 'not_dispatched',
              'reason': 'barrier wait interrupted before dispatch (status=reconsidering)'}


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


# ── restore path ─────────────────────────────────────────────────────────────
#
# The dict was persisted to `chat_messages` before the fix existed, and
# `Event.__aenter__` reads the last 10 turns back into `_turns` on startup. Without
# a guard here, upgrading a robot does not cure it: the poisoned turn is restored
# and re-sent, so the 400s resume on the first turn after the restart.

def test_restored_turn_has_its_dict_tool_content_repaired():
    turn = [
        {'role': 'assistant', 'content': 'x', 'tool_calls': [{'id': 'c1'}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': BAD_RESULT},
    ]
    clean = _sanitise_turn(turn)
    assert isinstance(clean[1]['content'], str)
    assert json.loads(clean[1]['content']) == BAD_RESULT
    assert clean[1]['tool_call_id'] == 'c1'


def test_sanitise_turn_does_not_mutate_the_input():
    """The caller still holds the list that came out of the DB."""
    turn = [{'role': 'tool', 'tool_call_id': 'c1', 'content': BAD_RESULT}]
    _sanitise_turn(turn)
    assert turn[0]['content'] is BAD_RESULT


def test_sanitise_turn_leaves_healthy_messages_alone():
    turn = [{'role': 'user', 'content': 'hi'},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}]
    assert _sanitise_turn(turn) == turn


# ── get_recent_context_rich ──────────────────────────────────────────────────
#
# `content[:800]` on a dict raises `TypeError: unhashable type: 'slice'`. This is
# called from `_bg_trigger_loop`, whose `while True` had no try/except, so the
# exception killed the task outright — background sensor monitoring then stayed
# dead for the rest of the process with one "Task exception was never retrieved"
# line as the only trace. Measured on Tianyi: 40 minutes dead, cured by a restart.

class _FakeEvent:
    def __init__(self, turns):
        self._turns = turns


def test_recent_context_rich_survives_a_dict_tool_content(monkeypatch):
    turns = [[
        {'role': 'user', 'content': '讲讲这个'},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': BAD_RESULT},
    ]]
    monkeypatch.setattr(_ev, '_event_instance', _FakeEvent(turns))
    out = get_recent_context_rich(max_turns=10, max_chars=3000)
    assert '讲讲这个' in out
    assert 'not_dispatched' in out


def test_recent_context_rich_still_works_normally(monkeypatch):
    turns = [[{'role': 'user', 'content': 'hello'},
              {'role': 'assistant', 'content': 'hi'},
              {'role': 'tool', 'tool_call_id': 'c1', 'content': 'done'}]]
    monkeypatch.setattr(_ev, '_event_instance', _FakeEvent(turns))
    out = get_recent_context_rich()
    assert '[用户] hello' in out and '[助手] hi' in out and '[工具结果] done' in out


def test_recent_context_rich_skips_status_snapshots(monkeypatch):
    """Pre-existing behaviour must survive the new isinstance branch."""
    turns = [[{'role': 'user', 'content': '<status time="x">noise</status>'},
              {'role': 'user', 'content': 'real question'}]]
    monkeypatch.setattr(_ev, '_event_instance', _FakeEvent(turns))
    out = get_recent_context_rich()
    assert 'noise' not in out and 'real question' in out
