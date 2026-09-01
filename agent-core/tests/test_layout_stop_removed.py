"""Regression: a card removed from the layout must have its instance stopped.

Observed on orin5 2026-08-24. The dashboard showed TTS running and silent. The ROS
graph had two vits2 nodes — `vits2_trt_card_mt4rkb752py8` publishing to
`/remote_control/message/tts`, and `vits2_trt_card_mt8rck6q87qd`, the live one, on
`/remote_control/mic/asr/tts`. The first belonged to a card the operator had
deleted. Nothing had stopped it: `_do_stop_project` walks
`config.main['canvas_layout']['cards']`, and that card was no longer in the list,
so it was unreachable the moment it was deleted. Its ROS node, its subscription and
its CUDA context lived until perception exited, and the audio panel was watching
the topic the orphan owned.

Every removal path — deleting a card, another editor's layout reload, loading a
solution — ends in a layout write, so the diff is enforced there.

Run: cd agent-core && python3 -m pytest tests/test_layout_stop_removed.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import canvas as canvas_api  # noqa: E402
from api import config as config_api  # noqa: E402

TTS_OLD = {'id': 'card_mt4rkb752py8', 'mcpId': 'mcp-1', 'toolName': 'tts'}
TTS_NEW = {'id': 'card_mt8rck6q87qd', 'mcpId': 'mcp-1', 'toolName': 'tts'}
ASR = {'id': 'card_asr', 'mcpId': 'mcp-1', 'toolName': 'asr'}


@pytest.fixture
def calls(monkeypatch):
    """Record every MCP call the layout write makes."""
    seen = []

    async def _fake_call(mcp_id, req):
        seen.append((mcp_id, req.tool, dict(req.arguments)))
        return {'state': 'idle'}

    class _Req:
        def __init__(self, tool, arguments):
            self.tool = tool
            self.arguments = arguments

    mod = type(sys)('api.mcp_manage')
    mod.mcp_call_tool = _fake_call
    mod.MCPCallRequest = _Req
    monkeypatch.setitem(sys.modules, 'api.mcp_manage', mod)
    return seen


def _run(old, new):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        config_api.stop_removed_cards(old, new))


def test_the_orphan_case_stops_exactly_the_deleted_card(calls):
    """The reported layout: the old TTS card is replaced by a new one."""
    assert _run([TTS_OLD, ASR], [TTS_NEW, ASR]) == 1
    assert calls == [('mcp-1', 'tts', {'action': 'stop', 'instance_id': 'card_mt4rkb752py8'})]


def test_an_unchanged_layout_calls_nothing(calls):
    """Layout writes happen on every drag; only removals may have side effects."""
    assert _run([TTS_OLD, ASR], [TTS_OLD, ASR]) == 0
    assert calls == []


def test_a_moved_card_is_not_a_removed_card(calls):
    """Identity is the card id, not the dict — x/y change on every drag."""
    moved = dict(TTS_OLD, x=400, y=120)
    assert _run([TTS_OLD], [moved]) == 0
    assert calls == []


def test_clearing_the_canvas_stops_every_card(calls):
    assert _run([TTS_OLD, ASR], []) == 2
    assert {c[2]['instance_id'] for c in calls} == {'card_mt4rkb752py8', 'card_asr'}


def test_adding_a_card_stops_nothing(calls):
    assert _run([ASR], [ASR, TTS_NEW]) == 0
    assert calls == []


def test_cards_without_a_device_are_skipped(calls):
    """A half-built card has no mcpId; there is nothing to stop and no crash."""
    assert _run([{'id': 'card_x'}, {'id': 'card_y', 'mcpId': '', 'toolName': 'tts'}], []) == 0
    assert calls == []


def test_a_card_with_no_id_is_ignored(calls):
    assert _run([{'mcpId': 'mcp-1', 'toolName': 'tts'}], []) == 0
    assert calls == []


def test_a_failing_stop_does_not_block_the_others(monkeypatch, calls):
    """The layout must still be saved even if one device is unreachable."""
    seen = []

    async def _flaky(mcp_id, req):
        seen.append(req.arguments['instance_id'])
        if req.arguments['instance_id'] == 'card_mt4rkb752py8':
            raise RuntimeError('connection refused')
        return {'state': 'idle'}

    sys.modules['api.mcp_manage'].mcp_call_tool = _flaky
    assert _run([TTS_OLD, ASR], []) == 1          # one of two succeeded
    assert seen == ['card_mt4rkb752py8', 'card_asr']


def test_none_layouts_are_tolerated(calls):
    """An empty ConfigDB gives `canvas_layout` as {} — .get('cards') is None."""
    assert _run(None, None) == 0
    assert _run(None, [TTS_NEW]) == 0
    assert calls == []


def test_processor_wiring_preserves_destination_port_and_live_source_topic():
    source = {
        'id': 'lidar-source',
        'topicOut': [{'port': 'cloud', 'topic': '/stale/cloud'}],
    }
    target = {
        'id': 'navigation',
        'topicIn': [
            {'port': 'lidar', 'topic': '/default/lidar'},
            {'port': 'imu', 'topic': '/default/imu'},
        ],
    }
    connections = [{
        'fromCardId': 'lidar-source',
        'fromPortIdx': 0,
        'fromTopic': '/cached/cloud',
        'toCardId': 'navigation',
        'toPortIdx': 0,
    }]

    single, multiple, bindings = config_api._resolve_processor_inputs(
        target,
        connections,
        [source, target],
        {'lidar-source': [{'port': 'cloud', 'topic': '/live/cloud'}]},
    )

    assert single == '/live/cloud'
    assert multiple == []
    assert bindings == [{'port': 'lidar', 'topic': '/live/cloud'}]


def test_port_aware_wiring_requires_an_explicit_tool_contract(monkeypatch):
    mcp_client = type(sys)('mcp_client')
    mcp_client.registry = {
        'actucore': {
            'tool_definitions': [{
                'name': 'ControlledSemanticSpatial',
                'inputSchema': {
                    'properties': {'input_bindings': {'type': 'array'}},
                },
            }],
        },
    }
    monkeypatch.setitem(sys.modules, 'mcp_client', mcp_client)

    assert config_api._tool_accepts_input_bindings(
        'actucore', 'ControlledSemanticSpatial'
    )
    assert not config_api._tool_accepts_input_bindings('actucore', 'legacy_tool')


def test_saving_a_running_layout_reconciles_topic_actions(monkeypatch):
    started = []

    class _TopicActions:
        async def start(self, layout):
            started.append(layout)

    topic_actions = type(sys)('topic_actions')
    topic_actions.manager = _TopicActions()
    monkeypatch.setitem(sys.modules, 'topic_actions', topic_actions)
    monkeypatch.setattr(config_api.config, 'main', {
        'core': {'project_running': True},
        'canvas_layout': {'cards': []},
    })
    monkeypatch.setattr(canvas_api, '_editor_session', 'editor-1')
    monkeypatch.setattr(canvas_api, '_live_sessions', {'editor-1': 1})
    monkeypatch.setattr(canvas_api, 'notify_layout_changed', lambda _session: None)

    result = asyncio.run(canvas_api.save_layout(canvas_api.CanvasLayout(
        cards=[TTS_NEW],
        connections=[],
        session_id='editor-1',
    )))

    assert result == {'code': 200}
    assert started == [{
        'cards': [TTS_NEW],
        'connections': [],
        'execConnections': [],
        'transform': {},
    }]


def test_project_stop_does_not_emit_terminal_state_until_every_card_confirms(
    monkeypatch, calls
):
    async def _pending(_mcp_id, _req):
        return {
            'ok': False,
            'state': 'error',
            'status': 'error',
            'error': 'mapping finalization is pending',
            'retryable': True,
        }

    events = []

    async def _push_event(event):
        events.append(event)

    class _TopicActions:
        async def stop(self):
            return None

    motus_stream = type(sys)('api.motus_stream')
    motus_stream.push_event = _push_event
    topic_actions = type(sys)('topic_actions')
    topic_actions.manager = _TopicActions()
    monkeypatch.setitem(sys.modules, 'api.motus_stream', motus_stream)
    monkeypatch.setitem(sys.modules, 'topic_actions', topic_actions)
    monkeypatch.setattr(sys.modules['api.mcp_manage'], 'mcp_call_tool', _pending)
    monkeypatch.setattr(config_api.config, 'main', {
        'core': {'project_running': True},
        'canvas_layout': {'cards': [TTS_OLD]},
    })

    result = asyncio.run(config_api._do_stop_project())

    assert result['ok'] is False
    assert result['failures'][0]['card_id'] == TTS_OLD['id']
    assert config_api.config.main['core']['project_running'] is False
    assert not any(event.get('type') == 'project_state' for event in events)
