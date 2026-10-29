"""Optional strict stop confirmation and legacy deletion semantics."""
import asyncio
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from api import config as api_config  # noqa: E402
from api import mcp_manage  # noqa: E402


CARD = {'id': 'nav', 'mcpId': 'mcp-1', 'toolName': 'navigation'}
OTHER = {'id': 'other', 'mcpId': 'mcp-1', 'toolName': 'tts'}


def install_responses(monkeypatch, responses, *, declaration=None):
    calls = []
    responses = iter(responses)

    async def call(mid, req, timeout_s=None):
        calls.append((mid, req.arguments['action'], req.arguments['instance_id']))
        return next(responses)

    monkeypatch.setattr(mcp_manage, 'mcp_call_tool', call)
    monkeypatch.setattr(api_config, 'find_tool', lambda _mid, _tool: declaration or {})
    monkeypatch.setattr(api_config, 'STOP_CARD_TIMEOUT_S', 0.04)
    monkeypatch.setattr(api_config, '_STOP_POLL_INTERVAL_S', 0.001)
    return calls


def strict_stop(cards):
    results = asyncio.run(api_config._stop_cards(cards, require_confirmation=True))
    return sum(result['ok'] for result in results), [r for r in results if not r['ok']]


@pytest.mark.parametrize('payload', [
    {'state': 'idle', 'status': 'error', 'terminal_confirmed': True},
    {'state': 'error', 'status': 'idle', 'terminal_confirmed': True},
    {'status': 'error', 'terminal_confirmed': True},
    {'state': 'idle', 'ok': False},
    {'state': 'idle', 'error': 'stop failed'},
    {'error': 'unknown action: stop'},
    {'error': 'unknown action: stop', 'terminal_confirmed': True},
    {'state': 'idle', 'status': 'refused', 'terminal_confirmed': True},
    {'state': 'unknown'},
    {'terminal_confirmed': True, 'status': 'mystery'},
    {'state': 'idle', 'status': 'unknown'},
    {'status': 'idle', 'state': 'unknown'},
    {'state': None, 'terminal_confirmed': True},
    {'state': [], 'status': 'idle'},
    {'state': {'unexpected': True}, 'terminal_confirmed': True},
    {'status': [], 'state': 'idle'},
    {'state': True, 'terminal_confirmed': True},
    {'state': [], 'status': 'running'},
    {'state': 'idle', 'status': ''},
    {},
])
def test_removed_card_rejects_error_or_unconfirmed_result(monkeypatch, payload):
    calls = install_responses(monkeypatch, [{'code': 200, 'data': payload}],
                              declaration={'inputSchema': {'x-stop': {
                                  'failed_states': ['refused']}}})
    count, failures = strict_stop([CARD])
    assert count == 0
    assert len(failures) == 1
    assert failures[0]['card_id'] == CARD['id']
    assert failures[0]['error']
    assert calls == [('mcp-1', 'stop', 'nav')]


@pytest.mark.parametrize('payload', [
    {'state': 'idle'},
    {'status': 'stopped'},
    {'state': 'disabled'},
    {'terminal_confirmed': True},
    {'status': 'stopped', 'state': 'idle'},
])
def test_removed_card_accepts_uncontradicted_confirmation(monkeypatch, payload):
    calls = install_responses(monkeypatch, [{'code': 200, 'data': payload}])
    assert strict_stop([CARD]) == (1, [])
    assert calls == [('mcp-1', 'stop', 'nav')]


@pytest.mark.parametrize('pending', [
    {'state': 'idle', 'status': 'running', 'terminal_confirmed': True},
    {'status': 'idle', 'state': 'stopping', 'terminal_confirmed': True},
    {'state': 'idle', 'status': 'releasing'},
    {'state': 'idle', 'authority_released': False},
])
def test_pending_takes_precedence_and_polls_info(monkeypatch, pending):
    calls = install_responses(monkeypatch, [
        {'code': 200, 'data': pending},
        {'code': 200, 'data': {'state': 'idle', 'authority_released': True}},
    ], declaration={'inputSchema': {'x-stop': {
        'pending_states': ['releasing'], 'pending_flags': {'authority_released': False}}}})
    assert strict_stop([CARD]) == (1, [])
    assert calls == [('mcp-1', 'stop', 'nav'), ('mcp-1', 'info', 'nav')]


def test_persistent_pending_times_out_without_resending_stop(monkeypatch):
    actions = []

    async def call(_mid, req, timeout_s=None):
        actions.append(req.arguments['action'])
        return {'code': 200, 'data': {'status': 'running', 'terminal_confirmed': True}}

    monkeypatch.setattr(mcp_manage, 'mcp_call_tool', call)
    monkeypatch.setattr(api_config, 'find_tool', lambda _mid, _tool: {})
    monkeypatch.setattr(api_config, 'STOP_CARD_TIMEOUT_S', 0.02)
    monkeypatch.setattr(api_config, '_STOP_POLL_INTERVAL_S', 0.002)
    count, failures = strict_stop([CARD])
    assert count == 0 and failures[0]['card_id'] == CARD['id']
    assert actions[0] == 'stop' and 'info' in actions
    assert actions.count('stop') == 1


def test_declared_failure_precedes_declared_pending(monkeypatch):
    calls = install_responses(monkeypatch, [{'code': 200, 'data': {
        'state': 'idle', 'status': 'handback', 'authority_released': False,
        'terminal_confirmed': True}}], declaration={'inputSchema': {'x-stop': {
            'failed_states': ['handback'], 'pending_states': ['handback'],
            'pending_flags': {'authority_released': False}}}})
    count, failures = strict_stop([CARD])
    assert count == 0 and failures[0]['card_id'] == CARD['id']
    assert calls == [('mcp-1', 'stop', 'nav')]


def test_pending_info_failure_cannot_be_overridden_by_terminal(monkeypatch):
    calls = install_responses(monkeypatch, [
        {'code': 200, 'data': {'state': 'stopping'}},
        {'code': 200, 'data': {'state': 'idle', 'status': 'error',
                               'terminal_confirmed': True}},
    ])
    count, failures = strict_stop([CARD])
    assert count == 0 and failures[0]['card_id'] == CARD['id']
    assert calls == [('mcp-1', 'stop', 'nav'), ('mcp-1', 'info', 'nav')]


def test_malformed_and_mcp_error_responses_are_unconfirmed(monkeypatch):
    for response in (
        {'code': 200, 'data': [{'text': 'not json'}]},
        {'code': 200, 'data': {'result': json.dumps({'state': 'idle'})}},
        {'code': 500, 'message': 'MCP isError', 'data': {'state': 'idle'}},
    ):
        install_responses(monkeypatch, [response])
        count, failures = strict_stop([CARD])
        assert count == 0 and failures[0]['card_id'] == CARD['id']


def test_one_failed_card_does_not_block_other_stop(monkeypatch):
    seen = []

    async def call(_mid, req, timeout_s=None):
        seen.append(req.arguments['instance_id'])
        if req.arguments['instance_id'] == 'nav':
            return {'code': 200, 'data': {'state': 'idle', 'status': 'error'}}
        return {'code': 200, 'data': {'state': 'idle'}}

    monkeypatch.setattr(mcp_manage, 'mcp_call_tool', call)
    monkeypatch.setattr(api_config, 'find_tool', lambda _mid, _tool: {})
    count, failures = strict_stop([CARD, OTHER])
    assert count == 1 and [f['card_id'] for f in failures] == ['nav']
    assert set(seen) == {'nav', 'other'}


def test_bad_metadata_is_isolated_and_other_card_still_stops(monkeypatch):
    seen = []

    def find_tool(_mid, name):
        if name == 'navigation':
            raise RuntimeError('bad metadata')
        return {}

    async def call(_mid, req, timeout_s=None):
        seen.append(req.arguments['instance_id'])
        return {'code': 200, 'data': {'state': 'idle'}}

    monkeypatch.setattr(api_config, 'find_tool', find_tool)
    monkeypatch.setattr(mcp_manage, 'mcp_call_tool', call)
    count, failures = strict_stop([CARD, OTHER])
    assert count == 1 and [f['card_id'] for f in failures] == ['nav']
    assert failures[0]['error'] == 'bad metadata'
    assert seen == ['other']


@pytest.mark.parametrize('payload', [
    {'error': 'unknown action: stop'},
    {'state': 'idle', 'error': 'last frame dropped'},
    {'state': 'idle', 'status': 'error'},
])
def test_ordinary_project_stop_retains_legacy_behavior(monkeypatch, payload):
    install_responses(monkeypatch, [{'code': 200, 'data': payload}])
    results = asyncio.run(api_config._stop_cards([CARD]))
    assert results[0]['ok'] is True
