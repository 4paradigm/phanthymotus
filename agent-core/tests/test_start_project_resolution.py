"""What _do_start_project actually passes each card, on the R1 layout.

Companion to test_start_project_order.py, which tests the order alone. This
drives the real function against a fake driver that behaves like perception's
ASR plugin — its `topic_out` is `{input_topic}/asr`, known only once it has been
started with an input — so the order is not merely asserted, it is load-bearing:
get it wrong and the fake cannot answer, exactly as on the robot.

Run: cd agent-core && python3 -m pytest tests/test_start_project_resolution.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import config as config_api  # noqa: E402
import config  # noqa: E402


# ── the R1 layout, as saved ──────────────────────────────────────────────────

def _card(cid, tool, topic_out=None):
    c = {'id': cid, 'toolName': tool, 'mcpId': 'mcp-perception'}
    if topic_out is not None:
        c['topicOut'] = topic_out
    return c


def _conn(src, dst, topic=''):
    return {'fromCardId': src, 'toCardId': dst, 'fromPortIdx': '0',
            'toPortIdx': '0', 'fromTopic': topic}


MIC, ASR, CORE, RM = 'card-mic', 'card-asr', 'card-core', 'card-rm'
# A card whose driver answers `info` with no topic at all — an offline plugin,
# or one whose output genuinely cannot be inferred. The fallback chain and the
# unresolved-input check are the only things that can speak for it.
SILENT = 'card-silent'

# decision_core precedes asr, as it does on R1: cards are listed in the order
# they were created and asr was added four days later.
R1_LAYOUT = {
    'cards': [
        _card(CORE, 'decision_core', [{'topic': '/decision_core', 'format': 'data/json'}]),
        _card(RM, 'remote_message', [{'topic': '/remote_control/message', 'format': 'data/json'}]),
        _card(MIC, 'mic', [{'topic': '/ubuntu/mic/audio', 'format': 'audio/pcm-16k'}]),
        # What a multiInstance tool's schema declares: a format, no topic.
        _card(ASR, 'asr', [{'format': 'data/json', 'desc': 'ASR result event'}]),
    ],
    'connections': [
        _conn(MIC, ASR, '/ubuntu/mic/audio'),
        _conn(ASR, CORE, ''),                        # never resolved in a browser
        _conn(RM, CORE, '/remote_control/message'),
    ],
}


@pytest.fixture
def driver(monkeypatch):
    """A fake perception whose ASR output is derived from its input.

    Returns the recorded `start` arguments per card id. `info` answers from the
    input the card was started with, which is the whole point: an ASR that was
    never started, or started without an input, has no output topic to report.
    """
    starts, started_input = {}, {}

    async def _call(mcp_id, req):
        args = dict(req.arguments)
        action, card_id = args.get('action'), args.get('instance_id')
        if action == 'start':
            starts[card_id] = args
            started_input[card_id] = args.get('input_topic') or ''
            return {'code': 200, 'data': {'state': 'running'}}
        if action == 'info':
            return {'code': 200, 'data': {'state': 'running',
                                          'topic_out': _topic_out(card_id, args)}}
        return {'code': 200, 'data': {'state': 'idle'}}

    def _topic_out(card_id, args):
        if card_id == ASR:
            # Derived. `info` is asked with the input the card was started with;
            # without one there is nothing to derive from and perception would
            # answer its fallback, which for ASR is no topic at all.
            src = args.get('input_topic') or started_input.get(card_id) or ''
            return [{'topic': f'{src}/asr', 'format': 'data/json'}] if src else []
        static = {
            MIC: [{'topic': '/ubuntu/mic/audio', 'format': 'audio/pcm-16k'}],
            RM: [{'topic': '/remote_control/message', 'format': 'data/json'}],
            CORE: [{'topic': '/decision_core', 'format': 'data/json'}],
        }
        # Anything else — SILENT included — answers with nothing, which is what
        # an offline driver or an uninferable output looks like.
        return static.get(card_id, [])

    class _Req:
        def __init__(self, tool, arguments):
            self.tool = tool
            self.arguments = arguments

    mcp = types.ModuleType('api.mcp_manage')
    mcp.mcp_call_tool = _call
    mcp.MCPCallRequest = _Req

    events = []

    async def _push(event):
        events.append(event)

    stream = types.ModuleType('api.motus_stream')
    stream.push_event = _push

    registered = []

    async def _register(topic, fmt, mcp_id):
        registered.append(topic)

    inspection = types.ModuleType('api.inspection')
    inspection.register_topic_internal = _register

    chan = types.ModuleType('channel.manager')
    chan.manager = types.SimpleNamespace(sync_from_canvas=lambda: None, _adapters={})
    chan._get_channel_configs = lambda: []

    for name, mod in (('api.mcp_manage', mcp), ('api.motus_stream', stream),
                      ('api.inspection', inspection), ('channel.manager', chan)):
        monkeypatch.setitem(sys.modules, name, mod)

    return types.SimpleNamespace(starts=starts, events=events, registered=registered)


def _start(layout):
    config.main['canvas_layout'] = layout
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        config_api._do_start_project())


def _errors(events):
    return [e['payload'] for e in events
            if e.get('type') == 'project_start_item'
            and e['payload'].get('status') == 'error']


# ── the reported failure ─────────────────────────────────────────────────────

def test_the_agent_loop_is_given_the_asr_topic(driver):
    assert _start(R1_LAYOUT) is True
    # Two inputs, so they arrive as input_topics rather than input_topic.
    assert sorted(driver.starts[CORE]['input_topics']) == [
        '/remote_control/message', '/ubuntu/mic/audio/asr']


def test_asr_is_started_with_the_mic_topic(driver):
    _start(R1_LAYOUT)
    assert driver.starts[ASR]['input_topic'] == '/ubuntu/mic/audio'


def test_a_derived_topic_reaches_the_bus(driver):
    """The dashboard subscribes to what start-project registers."""
    _start(R1_LAYOUT)
    assert '/ubuntu/mic/audio/asr' in driver.registered


def test_a_source_card_is_started_with_no_input_topic(driver):
    _start(R1_LAYOUT)
    assert 'input_topic' not in driver.starts[MIC]
    assert 'input_topics' not in driver.starts[MIC]


# ── the silence half of the bug ──────────────────────────────────────────────

def test_an_input_that_cannot_be_resolved_fails_its_card(driver):
    """A card fed by a source that reports no topic must not come up deaf.

    The source here is on the canvas but publishes nothing under any name, so
    no ordering and no fallback can supply the topic. Before this, decision_core
    started with its ASR input silently absent and reported 已就绪.
    """
    layout = {
        'cards': [_card(SILENT, 'mic', []), _card(CORE, 'decision_core')],
        'connections': [_conn(SILENT, CORE, '')],
    }
    assert _start(layout) is False
    errors = _errors(driver.events)
    assert len(errors) == 1
    assert errors[0]['tool'] == 'decision_core'
    assert 'mic' in errors[0]['message']
    assert CORE not in driver.starts


def test_one_unresolved_input_out_of_two_is_not_silently_dropped(driver):
    """The multi-input case: a partial answer used to look like a whole one."""
    layout = {
        'cards': [_card(SILENT, 'mic', []),
                  _card(RM, 'remote_message', [{'topic': '/remote_control/message',
                                                'format': 'data/json'}]),
                  _card(CORE, 'decision_core')],
        'connections': [_conn(SILENT, CORE, ''), _conn(RM, CORE, '/remote_control/message')],
    }
    assert _start(layout) is False
    assert CORE not in driver.starts


# ── fallbacks, for a source that cannot answer ───────────────────────────────

def test_a_persisted_fromTopic_still_covers_a_silent_source(driver):
    """The old fallback chain has to keep working for offline drivers."""
    layout = {
        'cards': [_card(SILENT, 'mic', []), _card(CORE, 'decision_core')],
        'connections': [_conn(SILENT, CORE, '/ubuntu/mic/audio')],
    }
    assert _start(layout) is True
    assert driver.starts[CORE]['input_topic'] == '/ubuntu/mic/audio'


def test_a_persisted_topicOut_covers_a_connection_with_no_fromTopic(driver):
    layout = {
        'cards': [_card(SILENT, 'mic', [{'topic': '/saved/mic', 'format': 'audio/pcm-16k'}]),
                  _card(CORE, 'decision_core')],
        'connections': [_conn(SILENT, CORE, '')],
    }
    assert _start(layout) is True
    assert driver.starts[CORE]['input_topic'] == '/saved/mic'


def test_a_live_answer_beats_a_stale_persisted_one(driver):
    """A running instance's topic wins over the canvas page's old snapshot."""
    layout = {
        'cards': [_card(MIC, 'mic'), _card(CORE, 'decision_core')],
        'connections': [_conn(MIC, CORE, '/remote_control/message/stale')],
    }
    assert _start(layout) is True
    assert driver.starts[CORE]['input_topic'] == '/ubuntu/mic/audio'
