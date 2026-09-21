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
# A card whose driver refuses the start with a bare `{"error": ...}` and no
# `state` — the shape unitree/{g1,r1,go2}/device.py use for "Missing
# input_topic". mcp_call_tool returns every ordinary tool result as code 200,
# so nothing about the transport marks this as a failure.
REFUSER = 'card-refuser'
# Two cards that differ only in how much of their input they actually take.
# GREEDY subscribes to every topic it was handed and reports all of them, the
# way decision_core does; PICKY takes the first and reports only that, the way
# every driver and perception's tts/ocr/face do.
GREEDY = 'card-greedy'
PICKY = 'card-picky'

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

    async def _call(mcp_id, req, timeout_s=None):
        # **`timeout_s` 要收下并断言，不能只是吞掉。** 一个把出问题的那个参数
        # 悄悄丢掉的假件，正是这一类 bug 能活下来的原因（画布那个 qos 也是）。
        # 只读的 `info()` 必须是有界的：无界时一个不回应的 MCP 会让整次启动永久
        # 挂死，而"取消"按钮对后端无效。`start` 相反，它合法地可以很慢。
        args = dict(req.arguments)
        if args.get('action') == 'info':
            assert timeout_s, 'info() 必须带超时，否则轮询循环的 deadline 是摆设'
        action, card_id = args.get('action'), args.get('instance_id')
        if action == 'start':
            starts[card_id] = args
            started_input[card_id] = args.get('input_topic') or ''
            # A Unitree speaker refuses a start it cannot bind, and says so the
            # way those drivers do: a bare `error` key, no `state`, and the
            # JSON-RPC call itself succeeds.
            if card_id == REFUSER:
                return {'code': 200, 'data': {'error': 'Missing input_topic'}}
            return {'code': 200, 'data': {'state': 'running'}}
        if action == 'info':
            data = {'state': 'running', 'topic_out': _topic_out(card_id, args, req.tool)}
            topic_in = _topic_in(card_id)
            if topic_in is not None:
                data['topic_in'] = topic_in
            return {'code': 200, 'data': data}
        return {'code': 200, 'data': {'state': 'idle'}}

    def _topic_in(card_id):
        """What the card reports having bound, or None to stay silent.

        Silence is the default because most of this file's fakes predate the
        question — and a tool that does not answer it must not be failed.
        """
        args = starts.get(card_id) or {}
        sent = list(args.get('input_topics') or
                    ([args['input_topic']] if args.get('input_topic') else []))
        if card_id == GREEDY:      # binds everything it was handed, as agentcore does
            return [{'topic': t, 'format': 'data/json'} for t in sent]
        if card_id == PICKY:       # binds the first and ignores the rest
            return [{'topic': t, 'format': 'data/json'} for t in sent[:1]]
        return None

    def _topic_out(card_id, args, tool=''):
        if tool == 'asr':
            # Derived from the input, exactly as perception does — and from the
            # *tool* name, with no trace of the instance, which is why two asr
            # cards on one source derive the same topic. `info` is asked with
            # the input the card was started with; without one there is nothing
            # to derive from and perception would answer no topic at all.
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


# ── a driver that refuses the start without setting `state` ──────────────────

def test_a_bare_error_answer_fails_the_card(driver):
    """`{"error": ...}` with no `state` is a refusal, not a successful start.

    Only `state` was read, so this answer fell through to the ready branch: the
    Unitree speaker came up bound to nothing, start-project reported all green,
    and the robot was silent with nothing in any log.
    """
    layout = {
        'cards': [_card(REFUSER, 'speaker')],
        'connections': [],
    }
    assert _start(layout) is False
    errors = _errors(driver.events)
    assert len(errors) == 1
    assert errors[0]['tool'] == 'speaker'
    assert 'Missing input_topic' in errors[0]['message']


def test_an_error_key_alongside_a_live_state_is_only_a_message(driver):
    """A running instance that also reports `error` must not be failed.

    Drivers use the key for both refusal and commentary, so letting it condemn
    a card that has explicitly said `state: running` would turn a dropped-frame
    warning into a rolled-back project.
    """
    state, message = config_api.tool_state_of(
        {'code': 200, 'data': {'state': 'running', 'error': 'last frame dropped'}})
    assert state == 'running'
    assert message == 'last frame dropped'


# ── a card handed more inputs than it consumes ───────────────────────────────

def _two_sources_into(card_id, tool):
    return {
        'cards': [_card(RM, 'remote_message', [{'topic': '/remote_control/message',
                                                'format': 'data/json'}]),
                  _card(MIC, 'mic', [{'topic': '/ubuntu/mic/audio',
                                      'format': 'audio/pcm-16k'}]),
                  _card(card_id, tool)],
        'connections': [_conn(RM, card_id, '/remote_control/message'),
                        _conn(MIC, card_id, '/ubuntu/mic/audio')],
    }


def test_a_card_that_binds_only_the_first_of_two_inputs_fails(driver):
    """The second connection did nothing, and the card reported 已就绪.

    No driver reads `input_topics`; perception's tts/ocr/face don't either.
    Drawing two lines into such a card produced a project that came up green
    with one of them inert.
    """
    assert _start(_two_sources_into(PICKY, 'tts')) is False
    errors = _errors(driver.events)
    assert len(errors) == 1
    assert errors[0]['tool'] == 'tts'
    # Names both halves: which input survived, and which one was ignored.
    assert '/remote_control/message' in errors[0]['message']
    assert '/ubuntu/mic/audio' in errors[0]['message']


def test_a_card_that_binds_both_inputs_succeeds(driver):
    """decision_core really does subscribe to all of them — no exemption needed."""
    assert _start(_two_sources_into(GREEDY, 'decision_core')) is True
    assert not _errors(driver.events)


def test_a_card_that_reports_no_bound_input_is_not_failed(driver):
    """Reticence is not evidence of dropping.

    Most tools report nothing useful under `topic_in`. Failing them on that
    would roll back projects that work.
    """
    assert _start(_two_sources_into(CORE, 'decision_core')) is True
    assert not _errors(driver.events)


def test_a_multi_input_card_is_also_given_the_singular_argument(driver):
    """The plural-only argument reached every driver as no input at all."""
    _start(_two_sources_into(GREEDY, 'decision_core'))
    assert driver.starts[GREEDY]['input_topic'] == \
        driver.starts[GREEDY]['input_topics'][0]


# ── two cards deriving the same output topic ─────────────────────────────────

def test_two_cards_of_one_tool_on_one_source_clash(driver):
    """`{input}/asr` carries no trace of the instance, so both cards claim it.

    The canvas allows the second card (the duplicate guard is skipped for
    multiInstance tools) and perception gives each its own node, so both publish
    to the one topic and every utterance arrives twice.
    """
    asr2 = 'card-asr-2'
    layout = {
        'cards': [_card(MIC, 'mic', [{'topic': '/ubuntu/mic/audio',
                                      'format': 'audio/pcm-16k'}]),
                  _card(ASR, 'asr', []), _card(asr2, 'asr', [])],
        'connections': [_conn(MIC, ASR, '/ubuntu/mic/audio'),
                        _conn(MIC, asr2, '/ubuntu/mic/audio')],
    }
    assert _start(layout) is False
    errors = _errors(driver.events)
    assert len(errors) == 1, 'reported once, on the second card'
    assert '/ubuntu/mic/audio/asr' in errors[0]['message']


def test_two_cards_of_one_tool_on_different_sources_are_fine(driver):
    """Different inputs derive different topics — the supported arrangement."""
    layout = {
        'cards': [_card(MIC, 'mic', [{'topic': '/ubuntu/mic/audio',
                                      'format': 'audio/pcm-16k'}]),
                  _card(RM, 'remote_message', [{'topic': '/remote_control/message',
                                                'format': 'data/json'}]),
                  _card(ASR, 'asr', []), _card('card-asr-2', 'asr', [])],
        'connections': [_conn(MIC, ASR, '/ubuntu/mic/audio'),
                        _conn(RM, 'card-asr-2', '/remote_control/message')],
    }
    assert _start(layout) is True
    assert not _errors(driver.events)


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
