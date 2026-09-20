"""带 `test` 段的方案 = 一个基准测试用例。

用例横跨三层：画布与技能在 agent-core，世界与地图在驱动，断言在两者之上。卡片装不下
这个东西 —— 卡片住在某一个驱动里，驱动没装的时候卡片本身就不存在，「你缺这个驱动」
连说的地方都没有。做成方案包体的一段，载入之前就能分层把话说清楚。

这个文件钉的是那条路径：打包时拒绝跑不起来的用例，preflight 时分层报出缺什么，
载入后 `loaded_case()` 知道现在能跑哪个用例。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_solution_test_block.py -q
"""

import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'solution-test.db'))

import config  # noqa: E402
import mcp_client  # noqa: E402
from api import solutions  # noqa: E402

TEST_BLOCK = {
    'requires': {'drivers': ['simulator-generic'], 'assets': ['bj-2f']},
    'run': {'prompt': '带我转一下展区并给我介绍下',
            'injections': [{'after_arrival': 'P5', 'delay': 6.0, 'text': '先等一下'}]},
    'evaluate': {'expect': {'waypoint_order': ['P3', 'P4']},
                 'weights': {'orchestration': 60, 'long_horizon': 40}},
}


def payload(test=TEST_BLOCK):
    out = {'formatVersion': solutions.FORMAT_VERSION, 'canvas': {}, 'devices': []}
    if test is not None:
        out['test'] = test
    return out


@pytest.fixture
def simulator(monkeypatch):
    # 名字写进注册表，在线写进运行时 map —— 真机上就是这么分的两份。往 registry
    # 条目里塞 server_name 会把「名字该从哪儿查」这个 bug 遮住（Orin6 上抓到过）。
    monkeypatch.setitem(config.main, 'services', {'mcp': [
        {'id': 'mcp-sim', 'name': '仿真器', 'server_name': 'simulator-generic'}]})
    monkeypatch.setitem(mcp_client.registry, 'mcp-sim', {
        'url': 'http://localhost:15711/mcp', 'online': True,
        'tools': ['nav', 'tts', 'sim_scenario', 'sim_report'],
    })
    maps = {'maps': [{'name': 'bj-2f'}]}

    async def fake_call(mcp_id, tool_name, args):
        return maps

    monkeypatch.setattr(mcp_client, 'call_tool_direct', fake_call)
    return maps


@pytest.fixture(autouse=True)
def clean_current():
    yield
    config.main['solution_current'] = None


# ── preflight ────────────────────────────────────────────────────────────────

def test_a_plain_solution_reports_no_test_section():
    """普通方案不该长出一个基准测试的 UI。"""
    assert asyncio.run(solutions._test_preflight(payload(test=None))) is None


def test_a_runnable_case_reports_can_run(simulator):
    report = asyncio.run(solutions._test_preflight(payload()))

    assert report['isCase'] is True and report['canRun'] is True
    assert report['prompt'] == '带我转一下展区并给我介绍下'
    assert report['injections'] == 1


def test_a_case_missing_its_driver_says_which_one(simulator, monkeypatch):
    """用户要看到的是「没装 simulator 驱动」，不是画布上少了张卡片。"""
    monkeypatch.delitem(mcp_client.registry, 'mcp-sim')

    report = asyncio.run(solutions._test_preflight(payload()))

    assert report['canRun'] is False
    assert report['readiness']['missing_drivers'] == ['simulator-generic']


def test_a_case_missing_its_map_says_which_map(simulator):
    simulator['maps'] = [{'name': 'lab'}]

    report = asyncio.run(solutions._test_preflight(payload()))

    assert report['canRun'] is False
    assert report['readiness']['missing_assets'] == ['bj-2f']
    assert report['readiness']['missing_drivers'] == []


def test_a_malformed_case_reports_problems_not_readiness(simulator):
    """依赖齐了也不代表能跑 —— 空 prompt 的用例只会记一个 0 分。"""
    broken = {**TEST_BLOCK, 'run': {'prompt': ''}}

    report = asyncio.run(solutions._test_preflight(payload(broken)))

    assert report['canRun'] is False
    assert report['readiness']['ok'] is True
    assert any('prompt 为空' in p for p in report['problems'])


# ── 打包 ──────────────────────────────────────────────────────────────────────

def test_packing_refuses_a_case_that_could_not_run(simulator, monkeypatch):
    """拒绝在打包时发生，不在载入时。一个跑不起来的用例一旦流通出去，
    每一台载入它的机器都要重新发现同一件事。"""
    monkeypatch.setattr(solutions, '_layout', lambda: {
        'cards': [{'id': 'c1', 'mcpId': 'mcp-sim', 'toolName': 'nav', 'x': 0, 'y': 0}],
        'connections': [], 'execConnections': [],
    })
    monkeypatch.setattr(solutions, '_mcp_list', lambda: [
        {'id': 'mcp-sim', 'name': 'simulator', 'server_name': 'simulator-generic',
         'url': 'http://localhost:15711/mcp'}])
    req = solutions.PackRequest(name='坏用例', include=solutions.PackInclude(
        test={'run': {'prompt': ''}, 'evaluate': {'expect': {}}}))

    result = asyncio.run(solutions._build_payload(req, None))

    assert result['ok'] is False
    assert '不能作为测试用例' in result['error']


# ── 载入之后 ──────────────────────────────────────────────────────────────────

def test_the_loaded_case_is_the_one_from_the_applied_solution():
    config.main['solution_current'] = {'slug': 'bj-2f-tour', 'includes': ['canvas', 'test'],
                                       'test': TEST_BLOCK}

    assert solutions.loaded_case() == TEST_BLOCK


def test_a_solution_without_a_test_section_leaves_no_case_loaded():
    config.main['solution_current'] = {'slug': 'plain', 'includes': ['canvas'], 'test': None}

    assert solutions.loaded_case() is None
