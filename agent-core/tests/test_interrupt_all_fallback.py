"""打断必须真的停住机器人，而不是「有绑定」就算数。

`_interrupt_active_outputs` 里 hook 与硬编码兜底覆盖的是**不相交**的两组卡片：
hook 覆盖自己声明了 `on_interrupt_all` 的卡，兜底覆盖叫 `tts`/`loco` 但没声明的卡。
以前写成 `if results: return`，而 `hooks.fire` 对每一个执行过的绑定都追加一条结果
——**包括抛异常的和什么都没做的** —— 于是一张卡的绑定就替全机队关掉了兜底。

Orin6 实测（2026-09-19）：actucore 的 `vla` 绑着 `on_interrupt_all`，卡片没在跑时
返回 `{"state":"idle","message":"卡片未在运行"}`，是那台机器上唯一的绑定。actucore
跑在每一台机器人上，所以插话时语音不停、导航不停，日志还写着 hook 已处理。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_interrupt_all_fallback.py -q
"""

import asyncio
import importlib
import os
import pathlib
import sys
import tempfile
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'interrupt-test.db'))

import hooks  # noqa: E402
import mcp_client  # noqa: E402
from event.llm import Event as DecisionLoop  # noqa: E402


class Rig:
    """真实形状：actucore 的 vla 绑了 hook 但什么都不做，感知的 tts 没绑任何东西。"""

    def __init__(self):
        self.calls = []          # (mcp_id, tool, action)
        self.replies = {}
        self.raises = set()

    async def call_tool_direct(self, mcp_id, tool_name, args):
        action = (args or {}).get('action', '')
        self.calls.append((mcp_id, tool_name, action))
        if (mcp_id, tool_name) in self.raises:
            raise RuntimeError('device offline')
        return self.replies.get((mcp_id, tool_name),
                                {'state': 'idle', 'stopped': True})

    @property
    def stopped(self):
        return {(m, t) for m, t, _ in self.calls}


@pytest.fixture
def rig(monkeypatch):
    r = Rig()
    monkeypatch.setattr(mcp_client, 'call_tool_direct', r.call_tool_direct)
    monkeypatch.setattr(mcp_client, 'registry', {
        # actucore：vla 绑了 on_interrupt_all，但卡片没在跑
        'mcp-actucore': {'online': True, 'url': 'http://x', 'tools': ['vla']},
        # 感知：真正会出声的 tts，没有声明任何 on_interrupt_all 绑定
        'mcp-perception': {'online': True, 'url': 'http://y', 'tools': ['tts', 'asr']},
        # 驱动：底盘
        'mcp-driver': {'online': True, 'url': 'http://z', 'tools': ['loco', 'switch_mode']},
    })
    monkeypatch.setattr(hooks, '_registry', {}, raising=False)
    monkeypatch.setattr(mcp_client, '_pending_actions', {}, raising=False)
    monkeypatch.setattr(mcp_client, '_pending_results', {}, raising=False)
    yield r
    hooks._registry.clear()


def bind_vla():
    """actucore 每台机器人都有，且默认声明这个绑定。"""
    hooks.register('mcp-actucore', 'vla', {'on_interrupt_all': {'action': 'interrupt'}})


def interrupt(reason=''):
    agent = DecisionLoop.__new__(DecisionLoop)
    # event.__init__ exports an Event instance as `llm`; resolve the module
    # explicitly so Python 3.10's mock resolver cannot patch the instance.
    with mock.patch.object(importlib.import_module('event.llm'), '_stop_countdown') as stop:
        asyncio.run(DecisionLoop._interrupt_active_outputs(agent, reason=reason))
    return stop


# ── 缺陷本体 ─────────────────────────────────────────────────────────────────

def test_a_binding_that_does_nothing_no_longer_suppresses_the_fallback(rig):
    """这条对着修复前的代码是失败的：vla 让 results 非空，于是提前 return。"""
    bind_vla()
    rig.replies[('mcp-actucore', 'vla')] = {'state': 'idle', 'message': '卡片未在运行'}

    interrupt()

    assert ('mcp-perception', 'tts') in rig.stopped, '插话没有停住语音'
    assert ('mcp-driver', 'loco') in rig.stopped, '插话没有停住底盘'


def test_the_hook_still_runs_too(rig):
    bind_vla()

    interrupt()

    assert ('mcp-actucore', 'vla', 'interrupt') in rig.calls


def test_a_card_already_stopped_by_the_hook_is_not_told_twice(rig):
    """去重按 (mcp_id, tool)：叫停是幂等的，但重复调用会把日志弄脏。"""
    hooks.register('mcp-driver', 'loco', {'on_interrupt_all': {'action': 'stop_move'}})

    interrupt()

    assert [c for c in rig.calls if c[:2] == ('mcp-driver', 'loco')] == [
        ('mcp-driver', 'loco', 'stop_move')]


def test_a_hook_binding_that_raised_is_retried_by_the_fallback(rig):
    """抛异常的那张卡并没有被叫停 —— 若它恰好也叫 tts/loco，兜底该再试一次。"""
    hooks.register('mcp-perception', 'tts', {'on_interrupt_all': {'action': 'interrupt'}})
    rig.raises.add(('mcp-perception', 'tts'))

    interrupt()

    assert len([c for c in rig.calls if c[:2] == ('mcp-perception', 'tts')]) == 2


def test_with_no_binding_at_all_the_fallback_still_runs(rig):
    interrupt()

    assert ('mcp-perception', 'tts') in rig.stopped
    assert ('mcp-driver', 'loco') in rig.stopped


# ── 兜底路径以前不做的事 ──────────────────────────────────────────────────────

def test_pending_actions_are_released_even_without_a_hook(rig):
    """以前只有 hook 路径清 pending，所以没有绑定的机器人插话后 barrier 一直卡着。"""
    event = asyncio.Event()
    mcp_client._pending_actions['act-1'] = event

    interrupt(reason='auto-interrupted by new user message')

    assert event.is_set()
    assert mcp_client._pending_results['act-1']['status'] == 'cancelled'


def test_the_silence_countdown_is_stopped_on_the_fallback_path_too(rig):
    stop = interrupt()

    stop.assert_called_once()


def test_no_reason_means_release_without_rewriting_the_result(rig):
    """TurnCancelled 只要放行 barrier；动作自己的终态由驱动的 ACP 回调写。"""
    event = asyncio.Event()
    mcp_client._pending_actions['act-1'] = event

    interrupt()

    assert event.is_set()
    assert 'act-1' not in mcp_client._pending_results


# ── 既有的边界不能被这次修改弄丢 ──────────────────────────────────────────────

def test_switch_mode_is_still_never_interrupted(rig):
    """半途中止姿态切换，是让一次受控下蹲变成摔倒。"""
    interrupt()

    assert all(tool != 'switch_mode' for _, tool, _ in rig.calls)


def test_peers_are_still_skipped(rig, monkeypatch):
    """别人的嘴不归我们关 —— Orin5/Orin6 曾互相掐断对方的话。"""
    mcp_client.registry['peer:abc'] = {'online': True, 'url': '', 'tools': ['tts', 'loco']}
    monkeypatch.setattr('peer.mcp_bridge.is_peer_mcp', lambda mid: mid.startswith('peer:'))

    interrupt()

    assert all(not m.startswith('peer:') for m, _, _ in rig.calls)


def test_offline_devices_are_skipped(rig):
    mcp_client.registry['mcp-perception']['online'] = False

    interrupt()

    assert ('mcp-perception', 'tts') not in rig.stopped


def test_nothing_to_interrupt_is_not_an_error(rig, monkeypatch):
    monkeypatch.setattr(mcp_client, 'registry', {})

    interrupt()        # 不抛异常就算过
