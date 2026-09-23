"""压缩历史时不能把 tool_call 结构压坏。

三条压缩路径各自把一份合法的 payload 压成服务端会拒绝的东西，而三条都是静默的：

- `_degrade_turn`（tier2）把 assistant 的 `tool_calls` 整个删掉，却留着后面几条
  `role: 'tool'` 结果 —— 每个滚进 tier2 的 turn 产出一批孤儿。实测样本：Orin5
  `/opt/phanthy-motus/data/llm_recent_request` 里 212 份留存请求有 25 份带孤儿，
  单份最多 12 条（`main_agent/260914_224012_ebc4d577.txt`，assistant 文本是
  `[调用: WebSearch, WebSearch]` —— 正是 `_degrade_turn` 的签名）。留存目录只存
  成功的请求，所以真实比例只高不低。
- `_compact_turn_messages`（turn 内）把 `arguments` 按字符切成半截 JSON。它是
  in-place 改 `self._current_turn`，损坏随 turn 进内存历史和 SQLite，之后每一轮
  请求都再带回去一次。
- 两者都能触发服务端 `messages[N]...` 形式的 400，而那类 400 在 `_classify_error`
  里判为不可重试（见 test_malformed_tool_call.py），于是整个 turn 被丢掉。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from event.llm import (  # noqa: E402
    _compact_turn_messages,
    _degrade_turn,
    _elide_arguments,
    _sanitize,
    _scrub,
)

LONG = '很长的一段播报词，' * 40


def _call(cid, name, args):
    return {'id': cid, 'type': 'function',
            'function': {'name': name, 'arguments': args}}


def _assert_payload_valid(messages):
    """服务端对一份 messages 的两条硬性要求。"""
    declared = set()
    for i, m in enumerate(messages):
        if m.get('role') == 'assistant':
            for tc in (m.get('tool_calls') or []):
                args = tc['function'].get('arguments')
                if args not in (None, ''):
                    json.loads(args)          # 必须是合法 JSON，否则这里抛
                declared.add(tc['id'])
        elif m.get('role') == 'tool':
            assert m['tool_call_id'] in declared, \
                f'messages[{i}] 是孤儿 tool 结果: {m["tool_call_id"]}'


# ── _elide_arguments ──────────────────────────────────────────────────────────

def test_elide_keeps_short_arguments_untouched():
    assert _elide_arguments('{"a": 1}') == '{"a": 1}'
    assert _elide_arguments('{}') == '{}'


def test_elide_output_is_always_valid_json():
    out = _elide_arguments(json.dumps({'text': LONG}, ensure_ascii=False),
                           value_chars=30, max_chars=100)
    assert json.loads(out)['text'].startswith(LONG[:30])
    assert len(out) < len(LONG)


def test_elide_keeps_field_names():
    """键名要留着 —— 模型靠它回忆自己调了什么。"""
    out = _elide_arguments(json.dumps({'text': LONG, 'voice': 'x' * 300}),
                           value_chars=10, max_chars=60)
    assert set(json.loads(out)) == {'text', 'voice'}


def test_elide_many_fields_falls_back_to_keys_only():
    args = json.dumps({f'k{i}': 'v' * 50 for i in range(20)})
    out = _elide_arguments(args, value_chars=10, max_chars=100)
    assert len(json.loads(out)) == 20        # 仍然合法，且键名都在


def test_elide_nested_structures_are_replaced_not_cut():
    out = _elide_arguments(json.dumps({'items': list(range(500))}), max_chars=50)
    json.loads(out)


def test_elide_unparseable_input_becomes_empty_object():
    """模型自己写歪的，压不动就退回合法空对象，不能留半截。"""
    assert _elide_arguments('{"text": "没闭合' + 'x' * 500) == '{}'
    assert _elide_arguments('x' * 500) == '{}'
    assert _elide_arguments(None) == '{}'


# ── _compact_turn_messages ────────────────────────────────────────────────────

def test_compact_turn_messages_keeps_arguments_parseable():
    big = json.dumps({'text': LONG}, ensure_ascii=False)
    msgs = [{'role': 'assistant', 'content': '', 'tool_calls': [_call('c1', 'tts', big)]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}]
    msgs += [{'role': 'user', 'content': f'{i}'} for i in range(12)]

    _compact_turn_messages(msgs, keep_recent=12)

    args = msgs[0]['tool_calls'][0]['function']['arguments']
    assert len(args) < len(big)
    json.loads(args)                          # 回归点：以前这里是半截 JSON
    _assert_payload_valid(msgs)


def test_compact_turn_messages_still_shrinks_tool_results():
    msgs = [{'role': 'tool', 'tool_call_id': 'c1', 'content': 'x' * 900}]
    msgs += [{'role': 'user', 'content': f'{i}'} for i in range(12)]
    _compact_turn_messages(msgs, keep_recent=12)
    assert len(msgs[0]['content']) < 200


# ── _degrade_turn ─────────────────────────────────────────────────────────────

def test_degrade_turn_does_not_orphan_tool_results():
    """回归点：tier2 曾经删掉 tool_calls 却留着结果。"""
    turn = [
        {'role': 'user', 'content': '查一下'},
        {'role': 'assistant', 'content': '我搜一下',
         'tool_calls': [_call('c1', 'WebSearch', '{"query": "百度 财报"}'),
                        _call('c2', 'WebSearch', '{"query": "百度 业务"}')]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'r1' * 200},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'r2' * 200},
    ]
    out = _degrade_turn(turn)
    _assert_payload_valid(out)
    assert {tc['id'] for tc in out[1]['tool_calls']} == {'c1', 'c2'}


def test_degrade_turn_still_shrinks():
    turn = [
        {'role': 'assistant', 'content': '', 'tool_calls': [_call('c1', 'tts', json.dumps({'text': LONG}))]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'x' * 900},
    ]
    before = len(json.dumps(turn, ensure_ascii=False))
    out = _degrade_turn(turn)
    assert len(json.dumps(out, ensure_ascii=False)) < before / 2
    _assert_payload_valid(out)


def test_degrade_turn_keeps_tool_names_visible():
    turn = [{'role': 'assistant', 'content': '', 'tool_calls': [_call('c1', 'WebSearch', '{}')]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}]
    assert _degrade_turn(turn)[0]['tool_calls'][0]['function']['name'] == 'WebSearch'


def test_degrade_turn_leaves_plain_messages_alone():
    turn = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': '你好'}]
    assert _degrade_turn(turn) == turn


# ── _scrub 的兜底（修已经落盘的坏历史）────────────────────────────────────────

def test_scrub_drops_orphan_tool_result_from_poisoned_history():
    """已部署机器的 data.db 里躺着旧版压缩写坏的记录，读回来的路上要洗掉。"""
    history = [
        {'role': 'assistant', 'content': '搜索中\n[调用: WebSearch]'},   # 旧版 degrade 的产物
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'r1'},
        {'role': 'user', 'content': '继续'},
    ]
    out = _scrub(history)
    assert [m['role'] for m in out] == ['assistant', 'user']


def test_scrub_repairs_broken_arguments_without_orphaning_result():
    """arguments 坏了要修成 '{}'，不能把调用丢掉 —— 丢了结果就成孤儿。"""
    history = [
        {'role': 'assistant', 'content': '', 'tool_calls': [_call('c1', 'tts', '{"text": "半截')]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
    ]
    out = _scrub(history)
    assert out[0]['tool_calls'][0]['function']['arguments'] == '{}'
    assert out[0]['tool_calls'][0]['function']['name'] == 'tts'
    _assert_payload_valid(out)


def test_scrub_strips_internal_fields_from_outbound_messages():
    """`_usage` 是给历史 modal 看的，不该出现在请求体里。

    Orin5 留存的 212 份请求里 139 份的 messages 带着它 —— 落盘时挂上去，
    读回来就一路跟进请求。按 OpenAI 的 message schema 它是未知字段。
    """
    history = [
        {'role': 'assistant', 'content': 'ok', '_usage': {'total_tokens': 12},
         'tool_calls': [_call('c1', 'tts', '{}')]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
    ]
    out = _scrub(history)
    assert all(not k.startswith('_') for m in out for k in m)
    assert out[0]['content'] == 'ok' and out[0]['tool_calls'][0]['id'] == 'c1'


def test_scrub_leaves_valid_history_untouched():
    history = [
        {'role': 'assistant', 'content': '', 'tool_calls': [_call('c1', 'tts', '{"text": "hi"}')]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
    ]
    assert _scrub(history) == history


# ── 端到端：走一遍真实的 tier2 + scrub + sanitize 组合 ────────────────────────

def test_degraded_history_survives_the_real_pipeline():
    turns = []
    for i in range(3):
        turns.append([
            {'role': 'user', 'content': f'任务 {i}'},
            {'role': 'assistant', 'content': '',
             'tool_calls': [_call(f'c{i}', 'WebSearch', json.dumps({'query': LONG}))]},
            {'role': 'tool', 'tool_call_id': f'c{i}', 'content': 'result' * 100},
        ])
    history = []
    for turn in turns:
        history.extend(_degrade_turn(turn))
    _assert_payload_valid(_sanitize(_scrub(history)))
