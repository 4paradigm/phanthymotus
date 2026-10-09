"""Subagent requests must contain a task even without inherited context.

Run: python3 -m pytest tests/test_subagent_context.py -q
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import client  # noqa: E402
from subagent.agent import Subagent  # noqa: E402
from subagent.context import SubagentContext  # noqa: E402
from subagent.protocol import SubagentSpec  # noqa: E402


@pytest.mark.parametrize('seed', ['', '机器人电量为 76%。'])
def test_first_request_contains_the_task_and_optional_context(seed):
    spec = SubagentSpec(goal='仅返回 JSON 格式的电量报告', context_seed=seed)
    messages = SubagentContext(spec).build_messages()
    assert [m['role'] for m in messages] == ['system', 'user']
    assert spec.goal in messages[1]['content']
    if seed:
        assert seed in messages[1]['content']


def test_followup_keeps_tool_history_without_restarting_the_task():
    context = SubagentContext(SubagentSpec(goal='读取电量', context_seed='robot-A'))
    turn = [
        {'role': 'assistant', 'tool_calls': [{
            'id': 'status-1', 'type': 'function',
            'function': {'name': 'read_status', 'arguments': '{}'},
        }]},
        {'role': 'tool', 'tool_call_id': 'status-1', 'content': '{"battery": 76}'},
    ]
    context.add_turn(turn)
    messages = context.build_messages([{'text': '只报告电量'}])
    assert messages[1:3] == turn
    assert messages[3] == {'role': 'user', 'content': '[来自主代理] 只报告电量'}
    assert len(messages) == 4


def test_checkpoint_restore_keeps_history_and_summary():
    context = SubagentContext(SubagentSpec(goal='继续整理报告'))
    turns = [[{'role': 'assistant', 'content': '已读取传感器'}]]
    context.restore_from_checkpoint(turns, '温度为 23.5 度')
    messages = context.build_messages()
    assert '23.5' in messages[1]['content']
    assert messages[-1] == turns[0][0]
    assert len(messages) == 4


def test_agent_loop_sends_a_user_task_without_a_context_seed(monkeypatch):
    sent = []

    async def accepting_client(message_list, tool_list, **kwargs):
        sent.append(message_list)
        assert any(m['role'] == 'user' and '仅回答 OK' in m['content']
                   for m in message_list), 'provider rejects system-only requests'
        return {'role': 'assistant', 'content': 'OK'}

    monkeypatch.setattr(client, 'call', accepting_client)
    agent = Subagent(SubagentSpec(goal='仅回答 OK', tool_filter=[],
                                 max_rounds=1, deliverable='report'))
    result = asyncio.run(agent.run())
    assert sent
    assert result.status == 'completed'
    assert result.output == 'OK'
