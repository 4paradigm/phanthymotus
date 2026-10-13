"""
test_stop_project_stages.py — 「停止智能控制」的两段,以及为什么必须是两段。

原来的顺序是**反的**:先逐卡 `stop`(串行、而且 `mcp_call_tool` 不传 timeout,默认
不限时),最后才把 `project_running` 翻成 False。于是只要有一张卡片的 MCP 接了连接
却不回应(被同进程的线程把 GIL 饿死、容器正在重启),停止就永远挂着,而**那段时间里
agent loop 还活着** —— 还会调 TTS、还会驱动执行器。界面上同时什么也不变,所以操作者
只会再点一次。

容器内实测:`timeout=None` 对着一个只 accept 不回的端口 25 秒仍未返回;给 5 秒则
5.3 秒失败。

所以这个文件守住四件事:

* **agent loop 先停,而且在任何一次 MCP 调用之前。** 这是安全属性,不是体验属性。
* **每张卡片各有死线,而且并行。** 一台坏掉的设备不能挡住别的设备停下来。
* **部分失败要报出来,而且报的是后果。** 「5/6」不是操作者要知道的事,「还有一台
  可能在跑」才是。
* **收尾在飞时再点停止不开第二轮**,否则界面上的逐项进度互相覆盖。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_stop_project_stages.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config as config_mod  # noqa: E402
from api import config as api_config  # noqa: E402


CARDS = [
    {'id': 'c-mic', 'mcpId': 'agentcore', 'toolName': 'remote_mic'},
    {'id': 'c-asr', 'mcpId': 'mcp-1', 'toolName': 'asr'},
    {'id': 'c-tts', 'mcpId': 'mcp-1', 'toolName': 'tts'},
]


class Recorder:
    """记下调用顺序 —— 顺序正是这里要守的东西。"""

    def __init__(self, hang_for=(), fail_for=()):
        self.calls = []
        self.events = []
        self.hang_for = set(hang_for)
        self.fail_for = set(fail_for)
        self.concurrent_peak = 0
        self._live = 0

    async def mcp_call_tool(self, mcp_id, req, timeout_s=None):
        self.calls.append((mcp_id, req.tool, req.arguments.get('action'), timeout_s))
        self._live += 1
        self.concurrent_peak = max(self.concurrent_peak, self._live)
        try:
            if req.tool in self.hang_for:
                # 没死线就永远挂着 —— 调用方必须自己带 timeout_s
                await asyncio.sleep(3600)
            if req.tool in self.fail_for:
                raise RuntimeError('mcp said no')
            await asyncio.sleep(0.01)
            return {'code': 200, 'data': {'result': {'state': 'idle'}}}
        finally:
            self._live -= 1

    async def push_event(self, event):
        self.events.append(event)

    def types(self):
        return [e['type'] for e in self.events]


@pytest.fixture
def rig(monkeypatch):
    rec = Recorder()
    config_mod.main['canvas_layout'] = {'cards': list(CARDS)}
    config_mod.main['core'] = {'project_running': True}

    def _install(recorder):
        monkeypatch.setattr('api.mcp_manage.mcp_call_tool', recorder.mcp_call_tool,
                            raising=False)
        monkeypatch.setattr('api.motus_stream.push_event', recorder.push_event,
                            raising=False)
        return recorder

    _install(rec)
    rec.install = _install
    monkeypatch.setattr(api_config, 'STOP_CARD_TIMEOUT_S', 0.2)
    monkeypatch.setattr(api_config, '_stop_teardown_task', None, raising=False)
    yield rec
    config_mod.main['canvas_layout'] = {}


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ── 第 1 段:agent loop ───────────────────────────────────────────────────────

def test_agent_loop_stops_before_any_mcp_call(rig):
    """这是安全属性:挂住的设备不能让 agent 多活几十秒。"""
    run(api_config._do_stop_project())
    assert rig.events[0]['type'] == 'project_state'
    assert rig.events[0]['payload']['running'] is False
    # 第一条 MCP 调用必须发生在那条广播之后
    assert rig.types().index('project_state') < rig.types().index('project_stop_begin')
    assert config_mod.main['core']['project_running'] is False


def test_a_hanging_device_does_not_delay_the_agent_stop(rig):
    """一张卡片挂死,`project_running` 仍然要立刻为 False。"""
    rig.install(Recorder(hang_for={'tts'}))

    async def scenario():
        await api_config._do_stop_project_agent()
        assert config_mod.main['core']['project_running'] is False
        # 收尾单独跑,慢是它的事
        await api_config._do_stop_project_teardown()

    run(asyncio.wait_for(scenario(), 5))


# ── 第 2 段:并行 + 死线 ──────────────────────────────────────────────────────

def test_every_card_gets_a_deadline(rig):
    run(api_config._stop_cards(CARDS))
    assert rig.calls, '一张卡片都没停'
    for _mcp, _tool, action, timeout_s in rig.calls:
        assert action == 'stop'
        assert timeout_s == pytest.approx(0.2), \
            '没传 timeout_s 的话默认是不限时,一个不回应的 MCP 会永远挂住停止'


def test_cards_stop_in_parallel(rig):
    run(api_config._stop_cards(CARDS))
    assert rig.concurrent_peak == len(CARDS), \
        f'串行了(峰值并发 {rig.concurrent_peak}),总耗时会是各卡之和'


def test_a_hanging_card_does_not_block_the_others(rig):
    rec = rig.install(Recorder(hang_for={'tts'}))
    results = run(asyncio.wait_for(api_config._stop_cards(CARDS), 5))
    ok = {r['tool'] for r in results if r['ok']}
    assert ok == {'remote_mic', 'asr'}, '坏掉的那张卡片不能带走别的'
    bad = [r for r in results if not r['ok']]
    assert len(bad) == 1 and bad[0]['tool'] == 'tts'
    assert '死线' in bad[0]['error'], bad[0]['error']
    assert rec.concurrent_peak == 3


def test_a_failing_card_is_reported_not_swallowed(rig):
    rig.install(Recorder(fail_for={'asr'}))
    results = run(api_config._stop_cards(CARDS))
    bad = [r for r in results if not r['ok']]
    assert len(bad) == 1
    assert bad[0]['tool'] == 'asr'
    assert bad[0]['error'], '错误信息不能是空的 —— 界面要显示它'


def test_cards_without_an_mcp_are_skipped(rig):
    results = run(api_config._stop_cards(
        CARDS + [{'id': 'note', 'mcpId': '', 'toolName': ''}]))
    assert len(results) == len(CARDS)


# ── 进度事件 ─────────────────────────────────────────────────────────────────

def test_progress_events_are_only_for_the_stop_button(rig):
    """删卡片触发的清理走同一个函数,但不该在界面上报「设备收尾」。"""
    run(api_config._stop_cards(CARDS))
    assert 'project_stop_item' not in rig.types()
    rig.events = []
    run(api_config._stop_cards(CARDS, progress=True))
    assert rig.types().count('project_stop_item') == len(CARDS)


def test_the_done_event_carries_what_failed(rig):
    rec = rig.install(Recorder(fail_for={'tts'}))
    run(api_config._do_stop_project())
    done = [e for e in rec.events if e['type'] == 'project_stop_done'][-1]
    assert done['payload']['total'] == 3
    assert done['payload']['stopped'] == 2
    assert [f['tool'] for f in done['payload']['failed']] == ['tts']
    assert done['payload']['failed'][0]['card_id'] == 'c-tts', \
        '界面要靠 card_id 做「重试这张」'


def test_begin_event_lists_the_cards_up_front(rig):
    """列表要在收尾开始前就给出来,否则界面只能一项项往上堆,看不出还剩几台。"""
    run(api_config._do_stop_project())
    begin = [e for e in rig.events if e['type'] == 'project_stop_begin'][0]
    assert [c['tool'] for c in begin['payload']['cards']] == \
        [c['toolName'] for c in CARDS]


# ── 在飞守卫 ─────────────────────────────────────────────────────────────────

def test_a_second_stop_does_not_start_a_second_teardown(rig):
    async def scenario():
        rig.install(Recorder(hang_for={'tts'}))
        first = await api_config.api_stop_project()
        assert first['ok'] is True
        assert api_config.teardown_running() is True
        second = await api_config.api_stop_project()
        assert second.get('teardown_running') is True
        task = api_config._stop_teardown_task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(scenario())


def test_the_endpoint_returns_before_the_teardown_finishes(rig):
    """响应只等第 1 段 —— 让它等完收尾,就等于让已经为真的「已停止」被一台坏设备拖着。"""
    async def scenario():
        rig.install(Recorder(hang_for={'tts'}))
        result = await asyncio.wait_for(api_config.api_stop_project(), 1.0)
        assert result['ok'] is True
        assert config_mod.main['core']['project_running'] is False
        assert api_config.teardown_running() is True
        api_config._stop_teardown_task.cancel()

    run(scenario())
