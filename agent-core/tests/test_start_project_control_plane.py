"""What start-project does for a `control/*` link, which runs against its grain.

Everywhere else in _do_start_project a card learns from its **sources**, which
dependency order has already started. A producer of commands has the opposite
need: it must know the action space of its **consumer**, which has not started
yet — and a card that streams motion deliberately will not start itself, so
waiting for it is not an option either.

That works only because a descriptor is a declaration rather than runtime state:
a driver's command card answers `info()` with the same action space whether or
not it is running. These tests pin that down, plus the two policy decisions that
came with it:

  - a `control/*` topic may legitimately have several publishers, because
    motus.control/1 arbitrates by `source`/`priority` in the driver; everywhere
    else a second publisher on one topic is still an error
  - a producer wired to two disagreeing action spaces fails the start rather
    than driving one of them

Run: cd agent-core && python3 -m pytest tests/test_start_project_control_plane.py
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


VLA, ARM, ARM2, TELEOP = 'card-vla', 'card-arm', 'card-arm2', 'card-teleop'

ARM_DESCRIPTOR = {
    'control_interface': 'motus.control/1',
    'mode': 'joint_position',
    'dof': 7,
    'joint_names': [f'joint{i}' for i in range(1, 8)],
    'units': {'angle': 'rad'},
    'limits': {'lower': [-3.0] * 7, 'upper': [3.0] * 7},
    'rate': {'watchdog_ms': 200},
    'force_torque': None,
}

# Same robot family, different arm: one stream of 7-DOF joint commands cannot
# mean anything sensible on a 6-DOF one.
ARM2_DESCRIPTOR = {**ARM_DESCRIPTOR, 'dof': 6,
                   'joint_names': [f'joint{i}' for i in range(1, 7)]}


def _card(cid, tool, topic_out=None):
    c = {'id': cid, 'toolName': tool, 'mcpId': f'mcp-{tool}'}
    if topic_out is not None:
        c['topicOut'] = topic_out
    return c


def _conn(src, dst, fmt='control/joint', topic='/robot/arm/cmd'):
    return {'fromCardId': src, 'toCardId': dst, 'fromPortIdx': '0',
            'toPortIdx': '0', 'format': fmt, 'fromTopic': topic}


@pytest.fixture
def driver(monkeypatch):
    """Cards that answer `info` the way a driver command card does.

    The arm reports its `control_interface` unconditionally — that is the whole
    premise being tested — and reports no `topic_out`, because a card that
    consumes commands publishes none.
    """
    starts, infos = {}, []
    # Cards whose device is not answering — a test adds to this to simulate a
    # container that is restarting when the project starts.
    offline: set = set()

    descriptors = {ARM: ARM_DESCRIPTOR, ARM2: ARM2_DESCRIPTOR}
    topic_outs = {
        VLA: [{'topic': '/robot/arm/cmd', 'format': 'control/joint'}],
        TELEOP: [{'topic': '/robot/arm/cmd', 'format': 'control/joint'}],
        # Two cards of one tool publishing the same non-control topic — what
        # the clash check was written for.
        'card-dup-a': [{'topic': '/dup', 'format': 'data/json'}],
        'card-dup-b': [{'topic': '/dup', 'format': 'data/json'}],
    }

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
            return {'code': 200, 'data': {'state': 'running'}}
        if action == 'info':
            infos.append(card_id)
            if card_id in offline:
                raise RuntimeError('connection refused')
            data = {'state': 'idle', 'topic_out': topic_outs.get(card_id, [])}
            if card_id in descriptors:
                data['control_interface'] = descriptors[card_id]
            return {'code': 200, 'data': data}
        return {'code': 200, 'data': {'state': 'idle'}}

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

    inspection = types.ModuleType('api.inspection')

    async def _register(topic, fmt, mcp_id):
        return None

    inspection.register_topic_internal = _register

    chan = types.ModuleType('channel.manager')
    chan.manager = types.SimpleNamespace(sync_from_canvas=lambda: None, _adapters={})
    chan._get_channel_configs = lambda: []

    for name, mod in (('api.mcp_manage', mcp), ('api.motus_stream', stream),
                      ('api.inspection', inspection), ('channel.manager', chan)):
        monkeypatch.setitem(sys.modules, name, mod)

    return types.SimpleNamespace(starts=starts, events=events, infos=infos,
                                 offline=offline)


def _run(layout) -> bool:
    """Write the layout the way the canvas does, then run the real start.

    Returns the start's own verdict: False means at least one card failed and
    the project was rolled back.

    Assigning into config.main rather than patching its `get`: the start path
    also reads `core`, and a patch that answers only `canvas_layout` turns an
    expected failure into a crash in the rollback, which reads as the feature
    being broken.
    """
    config.main['canvas_layout'] = layout
    return asyncio.run(config_api._do_start_project_impl())


def _errors(events):
    return [e['payload'] for e in events
            if e.get('type') == 'project_start_item'
            and e['payload'].get('status') == 'error']


# ── the descriptor reaches the producer ──────────────────────────────────────

def test_the_arms_action_space_reaches_the_vla_card_on_start(driver):
    layout = {
        'cards': [_card(VLA, 'vla'), _card(ARM, 'servo')],
        'connections': [_conn(VLA, ARM)],
    }
    _run(layout)

    assert driver.starts[VLA]['control_interface'] == ARM_DESCRIPTOR


def test_it_is_asked_before_the_consumer_has_started(driver):
    """A card that streams motion does not start itself; waiting would deadlock."""
    layout = {
        'cards': [_card(VLA, 'vla'), _card(ARM, 'servo')],
        'connections': [_conn(VLA, ARM)],
    }
    _run(layout)

    # The VLA card was started with the descriptor in hand, and the arm's own
    # start came later — so the descriptor cannot have come from a started card.
    assert 'control_interface' in driver.starts[VLA]
    assert list(driver.starts).index(VLA) < list(driver.starts).index(ARM)


def test_no_control_link_means_no_extra_calls_and_no_argument(driver):
    layout = {
        'cards': [_card(VLA, 'vla'), _card(ARM, 'servo')],
        'connections': [_conn(VLA, ARM, fmt='data/json')],
    }
    _run(layout)

    assert 'control_interface' not in driver.starts[VLA]


def test_a_consumer_that_reports_no_descriptor_is_not_an_error(driver):
    """Most cards have never heard of this; silence must not fail a start."""
    layout = {
        'cards': [_card(VLA, 'vla'), _card('card-mute', 'speaker')],
        'connections': [_conn(VLA, 'card-mute')],
    }
    _run(layout)

    assert _errors(driver.events) == []
    assert 'control_interface' not in driver.starts[VLA]


# ── disagreeing action spaces ────────────────────────────────────────────────

def test_two_disagreeing_action_spaces_fail_the_start(driver):
    """One stream of commands cannot mean 7 joints here and 6 there."""
    layout = {
        'cards': [_card(VLA, 'vla'), _card(ARM, 'servo'), _card(ARM2, 'servo')],
        'connections': [_conn(VLA, ARM), _conn(VLA, ARM2)],
    }
    ok = _run(layout)

    assert ok is False                       # failed, and rolled the project back
    assert VLA not in driver.starts
    message = _errors(driver.events)[0]['message']
    assert 'servo' in message


def test_two_agreeing_action_spaces_are_fine(driver):
    """Identical mode/dof/joint order — the stricter driver refuses per command."""
    layout = {
        'cards': [_card(VLA, 'vla'), _card(ARM, 'servo'), _card('card-arm3', 'servo')],
        'connections': [_conn(VLA, ARM), _conn(VLA, 'card-arm3')],
    }
    _run(layout)

    assert _errors(driver.events) == []


# ── several publishers on one control topic ──────────────────────────────────

def test_two_control_sources_on_one_topic_are_allowed(driver):
    """A generic priority-controlled pendant shares the motus.control/1 sink.

    The dedicated teleop tool now uses control/teleop and exclusive binding;
    it is intentionally not this arbitrated generic control source.
    """
    layout = {
        'cards': [_card(VLA, 'vla'), _card(TELEOP, 'manual_pendant'), _card(ARM, 'servo')],
        'connections': [_conn(VLA, ARM), _conn(TELEOP, ARM)],
    }
    _run(layout)

    assert _errors(driver.events) == []
    assert VLA in driver.starts and TELEOP in driver.starts


def test_a_duplicate_publisher_on_a_normal_topic_is_still_an_error(driver):
    """The arbitration exemption must not become a general amnesty.

    Two `asr` cards publishing `/dup` on `data/json` — the pre-existing bug the
    clash check exists for, where the second card silently reassigns the bus
    registration and every utterance arrives twice.
    """
    layout = {
        'cards': [
            _card('card-dup-a', 'dup', [{'topic': '/dup', 'format': 'data/json'}]),
            _card('card-dup-b', 'dup', [{'topic': '/dup', 'format': 'data/json'}]),
        ],
        'connections': [],
    }
    ok = _run(layout)

    assert ok is False
    assert any('/dup' in e['message'] for e in _errors(driver.events))


# ── the helpers, directly ────────────────────────────────────────────────────

def test_control_topics_only_counts_control_formats():
    from api.config import _control_topics

    entries = [
        {'topic': '/a', 'format': 'control/joint'},
        {'topic': '/b', 'format': 'control/velocity'},
        {'topic': '/c', 'format': 'data/json'},
        {'format': 'control/joint'},                  # no topic
    ]
    assert _control_topics(entries) == {'/a', '/b'}


def test_descriptor_conflict_ignores_limits_but_not_shape():
    from api.config import _descriptor_conflict

    stricter = {**ARM_DESCRIPTOR,
                'limits': {'lower': [-1.0] * 7, 'upper': [1.0] * 7}}
    assert _descriptor_conflict([ARM_DESCRIPTOR, stricter]) is False
    assert _descriptor_conflict([ARM_DESCRIPTOR, ARM2_DESCRIPTOR]) is True
    assert _descriptor_conflict([ARM_DESCRIPTOR]) is False


def test_descriptor_conflict_catches_reordered_joints():
    """Same count, same mode — and every command means something different."""
    from api.config import _descriptor_conflict

    swapped = {**ARM_DESCRIPTOR,
               'joint_names': list(reversed(ARM_DESCRIPTOR['joint_names']))}
    assert _descriptor_conflict([ARM_DESCRIPTOR, swapped]) is True


# ── an unanswering consumer is not a wiring mistake ──────────────────────────

def test_a_consumer_that_cannot_be_reached_names_itself(driver):
    """The two failures used to be indistinguishable, and the wrong one showed.

    A downstream driver whose container is restarting throws on `info()`. That
    produced an empty error, so the producer card started with no
    `control_interface` and reported its own message — "nothing is connected" —
    against a canvas that was wired correctly. An operator then checks the
    wiring, which is fine, and not the device, which is not. Seen on a Tianyi.
    """
    driver.offline.add(ARM)
    layout = {'cards': [_card(VLA, 'vla'), _card(ARM, 'servo')],
              'connections': [_conn(VLA, ARM)]}

    assert _run(layout) is False

    message = _errors(driver.events)[0]['message']
    assert 'servo' in message          # which card
    assert '连线是对的' in message      # and where not to look
    assert VLA not in driver.starts     # it never started blind


def test_an_unreachable_consumer_does_not_hide_a_reachable_one(driver):
    """Two links, one device down: the arm that did answer still drives."""
    driver.offline.add(ARM2)
    layout = {'cards': [_card(VLA, 'vla'), _card(ARM, 'servo'),
                        _card(ARM2, 'servo2')],
              'connections': [_conn(VLA, ARM), _conn(VLA, ARM2)]}
    _run(layout)

    assert driver.starts[VLA]['control_interface'] == ARM_DESCRIPTOR


# ── 一次挂住的启动不能永久锁死"启动"这个动作 ─────────────────────────────────
#
# 真机实测 2026-09-21（G1）：perception 的 HTTP 服务线程被同进程一个空转线程饿死
# （GIL 争用，accept 队列 Recv-Q 6 > backlog 5），一次 `tts.start` 再没回来。此前
# 的实现是 `_start_project_lock = True` 加 `try/finally` —— 那个 finally 防的是
# **抛异常**，防不住 `await` **挂住**：finally 永远轮不到执行，标志位永久为真，
# 之后每次点启动都是 409，"取消启动"只是前端的，除了重启进程没有出路。
#
# 撞上的是 TTS，但任何一个 MCP 服务器变慢都走同一条路，所以这一组测的是**机制**，
# 不是 TTS。


def _hang_forever(monkeypatch):
    """让 `_do_start_project_impl` 永不返回 —— 模拟一个不回应的 MCP。"""
    started = asyncio.Event()

    async def _never():
        started.set()
        await asyncio.Event().wait()      # 永远等待

    monkeypatch.setattr(config_api, '_do_start_project_impl', _never)
    monkeypatch.setattr(config_api, '_start_project_task', None, raising=False)
    return started


def test_a_hung_start_gives_up_instead_of_wedging_the_flag(monkeypatch):
    """超时兜底加在**锁**上，而不是加在每一次 MCP 调用上。

    后者要逐个审计调用点，而且对 `start` 本来就不该限时（一张卡片的 start 合法地
    可能冷下载几个 GB）。加在锁上则不管哪个 await 挂住都成立。
    """
    _hang_forever(monkeypatch)
    monkeypatch.setattr(config_api, 'START_PROJECT_TIMEOUT_S', 0.05)

    async def _run():
        first = await config_api._do_start_project()
        # 超时之后那个任务必须是 done —— 否则下一次点启动还是被挡在门外。
        assert config_api._start_project_task.done()
        return first

    assert asyncio.run(_run()) is False


def test_the_cancel_button_actually_cancels(monkeypatch):
    """此前"取消启动"只去停卡片，挂住的那个协程**继续挂着**，标志位也不清。"""
    started = _hang_forever(monkeypatch)
    monkeypatch.setattr(config_api, 'START_PROJECT_TIMEOUT_S', 30)

    async def _run():
        task = asyncio.ensure_future(config_api._do_start_project())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert config_api._cancel_start_project() is True
        assert await task is False
        assert config_api._start_project_task.cancelled()
        # 没有在飞的启动时，取消是无害的空操作，不是异常。
        assert config_api._cancel_start_project() is False

    asyncio.run(_run())


def test_a_second_start_while_one_is_in_flight_is_still_refused(monkeypatch):
    """并发保护不能因为换了实现就丢掉 —— 两个重叠的启动会把前端的事件流搅乱。"""
    started = _hang_forever(monkeypatch)
    monkeypatch.setattr(config_api, 'START_PROJECT_TIMEOUT_S', 30)

    async def _run():
        task = asyncio.ensure_future(config_api._do_start_project())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert await config_api._do_start_project() is None      # 被挡住
        config_api._cancel_start_project()
        await task

    asyncio.run(_run())
