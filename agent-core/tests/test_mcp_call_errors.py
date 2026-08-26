"""Regression: an MCP JSON-RPC error must reach the model, not become `"{}"`.

Observed on R1 2026-08-19. The model called `mcp__mcp-1782801833__move`, but the
real name of a split tool has four segments — `mcp__mcp-1782801833__loco__move`.
The three-segment fallback took `tool_name='move'` and injected no `action`, so
the driver correctly answered `-32601 Unknown tool: move`… and `_jrpc` dropped
the `error` object, `call_tool` stringified `{}`, and the transcript showed
`← "{}"`. The model read that as success, said "好的，我要转身了", called
stop_move, and finished. The robot never moved, and when told so it retried the
identical bad call because nothing said it had failed.

Run: cd agent-core && python3 -m pytest tests/test_mcp_call_errors.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402

MCP_ID = 'mcp-1782801833'
LOCO_MOVE = f'mcp__{MCP_ID}__loco__move'


@pytest.fixture
def r1_registry(monkeypatch):
    """A registry shaped like R1's: `loco` split into move / stop_move."""
    split_map = {
        LOCO_MOVE: {'tool': 'loco', 'action': 'move'},
        f'mcp__{MCP_ID}__loco__stop_move': {'tool': 'loco', 'action': 'stop_move'},
    }
    reg = {MCP_ID: {
        'url': 'http://localhost:15702/mcp',
        'tools': ['loco', 'switch_mode', 'arm'],
        'split_map': split_map,
        'tool_meta': {LOCO_MOVE: {'type': 'actuator'}},
        'schemas': {}, 'input_schemas': {}, 'tool_groups': {},
    }}
    monkeypatch.setattr(mcp_client, 'registry', reg)
    return reg


def _stub_jrpc(monkeypatch, response, seen):
    async def fake(session, url, method, params, req_id=1):
        seen.append(params)
        return response
    monkeypatch.setattr(mcp_client, '_jrpc', fake)


def test_jrpc_error_is_surfaced_not_swallowed(r1_registry, monkeypatch):
    seen = []
    _stub_jrpc(monkeypatch, {mcp_client.JRPC_ERROR_KEY:
                             {'code': -32601, 'message': 'Unknown tool: move'}}, seen)

    out = asyncio.run(mcp_client.call_tool(LOCO_MOVE, {'vyaw': 1.5}))

    assert out != '{}', 'the exact symptom: an error rendered as an empty result'
    assert 'Unknown tool: move' in out
    assert '-32601' in out
    assert '未执行任何动作' in out, 'the model must be told nothing happened'


def test_unknown_tool_error_lists_the_valid_names(r1_registry, monkeypatch):
    seen = []
    _stub_jrpc(monkeypatch, {mcp_client.JRPC_ERROR_KEY:
                             {'code': -32601, 'message': 'Unknown tool: move'}}, seen)

    out = asyncio.run(mcp_client.call_tool(LOCO_MOVE, {}))

    assert LOCO_MOVE in out, 'so the model can correct itself instead of retrying blind'


def test_three_segment_name_resolves_to_the_split_tool(r1_registry, monkeypatch):
    """`mcp__<id>__move` → tool `loco`, action `move`, with `action` injected."""
    seen = []
    _stub_jrpc(monkeypatch, {'content': [{'type': 'text', 'text': '{"ret": 0}'}]}, seen)

    out = asyncio.run(mcp_client.call_tool(f'mcp__{MCP_ID}__move', {'vyaw': 1.5}))

    assert seen, 'the call must actually be dispatched'
    assert seen[0]['name'] == 'loco', 'previously sent name="move", which no driver has'
    assert seen[0]['arguments']['action'] == 'move', 'action was previously not injected'
    assert seen[0]['arguments']['vyaw'] == 1.5
    assert '{"ret": 0}' in out


def test_ambiguous_short_name_is_refused_with_the_options(monkeypatch):
    """Two tools sharing an action name must not be guessed between."""
    reg = {MCP_ID: {
        'url': 'http://x', 'tools': ['loco', 'arm'],
        'split_map': {
            f'mcp__{MCP_ID}__loco__stop': {'tool': 'loco', 'action': 'stop'},
            f'mcp__{MCP_ID}__arm__stop':  {'tool': 'arm',  'action': 'stop'},
        },
        'tool_meta': {}, 'schemas': {}, 'input_schemas': {}, 'tool_groups': {},
    }}
    monkeypatch.setattr(mcp_client, 'registry', reg)
    called = []
    _stub_jrpc(monkeypatch, {'content': []}, called)

    out = asyncio.run(mcp_client.call_tool(f'mcp__{MCP_ID}__stop', {}))

    assert not called, 'must not pick one of two tools at random'
    assert '不明确' in out
    assert f'mcp__{MCP_ID}__loco__stop' in out and f'mcp__{MCP_ID}__arm__stop' in out


def test_successful_call_is_unchanged(r1_registry, monkeypatch):
    seen = []
    _stub_jrpc(monkeypatch, {'content': [{'type': 'text',
                                          'text': '{"ret": 0, "vyaw": 1.5}'}]}, seen)

    out = asyncio.run(mcp_client.call_tool(LOCO_MOVE, {'vyaw': 1.5}))

    assert out == '{"ret": 0, "vyaw": 1.5}'
    assert seen[0]['arguments']['action'] == 'move'


def test_empty_result_without_an_error_still_stringifies(r1_registry, monkeypatch):
    """A driver legitimately returning no content keeps its old behaviour."""
    seen = []
    _stub_jrpc(monkeypatch, {}, seen)

    out = asyncio.run(mcp_client.call_tool(LOCO_MOVE, {}))

    assert out == '{}'
