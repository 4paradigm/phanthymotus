"""Benchmark storage and API.

Two properties this file exists to hold, both of which are about honesty rather
than correctness:

* **A score is stored with the configuration it was measured against.** Without
  the model, the image tag and the git sha, "did the number move when I changed
  the prompt" — the only question anyone actually asks of a benchmark — is
  unanswerable, and two rows of scores are not comparable.
* **n travels with the score.** An LLM is stochastic, so one run is one sample.
  A number shown without its sample size reads as a conclusion.

And one about layering: the simulator produces facts, agent-core judges them
(`benchmark_case.py`). The judge and the system under test have to be disjoint,
and the judge used to live in the simulator driver — which is part of the system
under test. This layer dispatches, collects, records, and reports what a case
still needs before it can run.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_api.py -q
"""

import asyncio
import os
import pathlib
import sys
import tempfile

import fastapi
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'benchmark-test.db'))

import benchmark_store  # noqa: E402
import config  # noqa: E402
import mcp_client  # noqa: E402
from api import benchmark  # noqa: E402


@pytest.fixture(autouse=True)
def clean_runs():
    for run in benchmark_store.list_runs(limit=500):
        benchmark_store.delete_run(run['id'])
    yield
    for run in benchmark_store.list_runs(limit=500):
        benchmark_store.delete_run(run['id'])


def _register(server_name, mcp_id='mcp-sim', name='仿真器'):
    """把驱动写进注册表（`/api/mcp` 读的那一份）。

    刻意**不**往 `mcp_client.registry` 里塞 `server_name` —— 真机上那份就没有这个
    字段，而测试里塞了它，就正好把「名字该从哪儿查」这个 bug 遮住。
    """
    import config
    services = dict(config.main.get('services') or {})
    services['mcp'] = [{'id': mcp_id, 'name': name, 'server_name': server_name}]
    config.main['services'] = services


@pytest.fixture
def simulator(monkeypatch):
    """A registry entry that looks like the simulator bundle, plus a fake driver."""
    calls = []
    replies = {}

    monkeypatch.setitem(mcp_client.registry, 'mcp-sim', {
        'url': 'http://localhost:15711/mcp', 'online': True,
        'tools': ['nav', 'tts', 'sim_scenario', 'sim_report'],
    })

    async def fake_call(mcp_id, tool_name, args):
        calls.append((mcp_id, tool_name, dict(args)))
        key = (tool_name, args.get('action') or args.get('what') or '')
        return replies.get(key, {})

    monkeypatch.setattr(mcp_client, 'call_tool_direct', fake_call)
    return {'calls': calls, 'replies': replies}


SUITE_SUMMARY = {
    'state': 'done', 'n': 2, 'scored': 2, 'mean': 87.5, 'stdev': 17.7,
    'outcomes': {'ok': 1, 'stalled': 1},
    'scenarios': {'exhibition_tour': {
        'n': 2, 'mean': 87.5, 'stdev': 17.7, 'pass_rate': 50.0,
        'by_dimension': {'orchestration': 100.0, 'long_horizon': 50.0},
    }},
    'cases': [
        {'scenario': 'exhibition_tour', 'repeat': 0, 'seed': 0, 'outcome': 'ok',
         'elapsed': 104.6, 'score': {'total': 100.0}, 'failures': []},
        {'scenario': 'exhibition_tour', 'repeat': 1, 'seed': 1, 'outcome': 'stalled',
         'elapsed': 90.0, 'score': {'total': 75.0}, 'failures': ['resume_correctness']},
    ],
}


# ── storage ──────────────────────────────────────────────────────────────────

def test_a_run_records_what_it_was_measured_against():
    run_id = benchmark_store.create_run(
        'exhibition_tour', n_repeats=3, llm_model='claude-opus-5',
        llm_provider='https://api.anthropic.com', host='orin6',
        image_tags={'agent_core': 'release.260918.abc1234'},
        git_shas={'agent_core': 'abc1234'})

    stored = benchmark_store.get_run(run_id)

    assert stored['llm_model'] == 'claude-opus-5'
    assert stored['image_tags']['agent_core'] == 'release.260918.abc1234'
    assert stored['git_shas']['agent_core'] == 'abc1234'
    assert stored['n_repeats'] == 3


def test_n_repeats_is_never_lost_even_when_it_is_one():
    run_id = benchmark_store.create_run('x', n_repeats=1)

    assert benchmark_store.get_run(run_id)['n_repeats'] == 1


def test_n_repeats_is_floored_at_one():
    run_id = benchmark_store.create_run('x', n_repeats=0)

    assert benchmark_store.get_run(run_id)['n_repeats'] == 1


def test_cases_attach_to_their_run_in_order():
    run_id = benchmark_store.create_run('x', n_repeats=2)
    benchmark_store.add_case(run_id, scenario='a', repeat_idx=0, ok=True, score=100.0,
                             outcome='ok', elapsed_ms=1200)
    benchmark_store.add_case(run_id, scenario='a', repeat_idx=1, ok=False, score=60.0,
                             outcome='stalled', assertions=['resume_correctness'])

    cases = benchmark_store.get_run(run_id)['cases']

    assert [c['repeat_idx'] for c in cases] == [0, 1]
    assert cases[0]['ok'] is True and cases[1]['ok'] is False
    assert cases[1]['assertions'] == ['resume_correctness']


def test_finishing_a_run_records_the_spread_not_only_the_mean():
    run_id = benchmark_store.create_run('x', n_repeats=3)
    benchmark_store.finish_run(run_id, score_total=82.0, score_stdev=9.5,
                               scores_by_dim={'orchestration': 90.0})

    stored = benchmark_store.get_run(run_id)

    assert stored['status'] == 'done'
    assert (stored['score_total'], stored['score_stdev']) == (82.0, 9.5)
    assert stored['scores_by_dim']['orchestration'] == 90.0
    assert stored['ended_at'] > stored['started_at'] - 1


def test_runs_come_back_newest_first():
    older = benchmark_store.create_run('a')
    newer = benchmark_store.create_run('b')

    ids = [run['id'] for run in benchmark_store.list_runs()]

    assert ids.index(newer) < ids.index(older)


def test_deleting_a_run_takes_its_cases_with_it():
    run_id = benchmark_store.create_run('x')
    benchmark_store.add_case(run_id, scenario='a')

    assert benchmark_store.delete_run(run_id) is True
    assert benchmark_store.get_run(run_id) is None
    assert benchmark_store.delete_run(run_id) is False


def test_trend_carries_the_model_so_a_change_can_be_attributed():
    """Two runs differ: the first question is always whether the code changed or
    the model did, and that is unanswerable without this field."""
    first = benchmark_store.create_run('tour', llm_model='claude-sonnet-5')
    benchmark_store.finish_run(first, score_total=70.0)
    second = benchmark_store.create_run('tour', llm_model='claude-opus-5')
    benchmark_store.finish_run(second, score_total=91.0)

    trend = benchmark_store.trend(suite='tour')

    assert [point['llm_model'] for point in trend] == ['claude-sonnet-5', 'claude-opus-5']
    assert [point['score_total'] for point in trend] == [70.0, 91.0]


def test_trend_skips_runs_that_never_finished():
    benchmark_store.create_run('tour')                      # still running
    done = benchmark_store.create_run('tour')
    benchmark_store.finish_run(done, score_total=50.0)

    assert [point['id'] for point in benchmark_store.trend(suite='tour')] == [done]


def test_malformed_json_columns_do_not_take_a_read_down():
    run_id = benchmark_store.create_run('x')
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET image_tags=? WHERE id=?', ('not json', run_id))
    conn.commit()

    assert benchmark_store.get_run(run_id)['image_tags'] == {}


# ── discovery ────────────────────────────────────────────────────────────────

def test_the_simulator_is_found_by_its_tools_not_by_its_name(simulator):
    """Device names, server_name and container names all change; tool names are
    part of the contract."""
    assert benchmark.find_simulator() == 'mcp-sim'


def test_a_device_without_both_cards_is_not_a_simulator(monkeypatch):
    monkeypatch.setattr(mcp_client, 'registry',
                        {'mcp-robot': {'online': True, 'tools': ['nav', 'tts', 'sim_report']}})

    assert benchmark.find_simulator() is None


def test_an_offline_simulator_is_not_offered(monkeypatch):
    monkeypatch.setattr(mcp_client, 'registry',
                        {'mcp-sim': {'online': False,
                                     'tools': ['sim_scenario', 'sim_report']}})

    assert benchmark.find_simulator() is None


# ── 申报探针 ──────────────────────────────────────────────────────────────────
#
# 这一组对着**真实形状的注册表**跑。`unmeasurable` 那边喂的是假 probe，所以探针自己
# 怎么查注册表，那边一条都盖不到 —— 第一版按裸名查 `tool_meta`，假 probe 一路绿，
# 到 R1 上才发现每张画布都被报成「没有任何申报了嘴的卡片」。

def registry_with(monkeypatch, server_name, mcp_id, tool, **meta):
    """两张表都要造，因为查一个工具真的要走两跳：

    `server_name → mcp_id` 住在 `config.main['services']['mcp']`（包体说的是驱动名，
    因为它要能换机器），`mcp_id → tool_meta` 住在运行时注册表，键是**全名**。
    """
    services = dict(config.main.get('services') or {})
    services['mcp'] = [{'id': mcp_id, 'server_name': server_name, 'name': server_name}]
    monkeypatch.setitem(config.main, 'services', services)
    monkeypatch.setitem(mcp_client.registry, mcp_id, {
        'online': True, 'tools': [tool],
        'tool_meta': {f'mcp__{mcp_id}__{tool}': meta},
    })


def test_the_probe_reads_a_tools_declared_mouth_and_acp(monkeypatch):
    registry_with(monkeypatch, 'perception-bundle', 'mcp-perc', 'tts',
                  resource=frozenset({'mouth'}), completion={'timeout': 120})

    assert benchmark.speech_probe('perception-bundle', 'tts') == {'mouth': True,
                                                                  'completion': True}


def test_the_probe_is_asked_by_driver_name_because_a_payload_has_no_mcp_id(monkeypatch):
    """包体里的卡片带 `deviceRef`，`mcpId` 是 None。拿 `mcp_id` 当入参，每个打包用例
    都会被报成「没有申报嘴的卡片」—— Orin6 上就是这么现形的。"""
    registry_with(monkeypatch, 'perception-bundle', 'mcp-perc', 'tts',
                  resource=frozenset({'mouth'}), completion={'timeout': 120})

    # 本机的 mcp_id 当驱动名传进去，应该什么都查不到。
    assert benchmark.speech_probe('mcp-perc', 'tts')['mouth'] is False


def test_a_tool_that_declares_no_channel_is_not_a_mouth(monkeypatch):
    """R1 的 `speaker` 就是这样：它是真的喇叭，但没申报通道，所以讲解事实不靠它 ——
    靠画布上那张申报了 mouth 的 `tts`。"""
    registry_with(monkeypatch, 'r1-device-bundle', 'mcp-r1', 'speaker',
                  resource=None, completion=None)

    assert benchmark.speech_probe('r1-device-bundle', 'speaker') == {'mouth': False,
                                                                     'completion': False}


def test_an_unknown_device_probes_false_rather_than_raising(monkeypatch):
    monkeypatch.setattr(mcp_client, 'registry', {})

    assert benchmark.speech_probe('nope', 'tts') == {'mouth': False, 'completion': False}


def test_the_panel_is_available_on_a_robot_with_no_simulator(monkeypatch):
    """R1 上抓到的：装了最新 agent-core，设置里却没有基准测试这一项，而且没有任何
    迹象说明为什么没有。

    入口原先挂在「有没有仿真器」上。但基准测试测的是**解决方案** —— 裁判一直是纯函数，
    事实流现在真机上也有（`benchmark_facts`），仿真器在不在场只决定世界能不能重置，
    以及有没有轨迹占用这类只有它算得出的量。两件都不是「入口该不该存在」。
    """
    monkeypatch.setattr(mcp_client, 'registry', {})

    result = asyncio.run(benchmark.available())

    assert result['available'] is True
    assert result['simulator'] is None      # 前端据此决定要不要走确认


# ── 被测配置 ────────────────────────────────────────────────────────────────

def test_the_environment_block_names_the_model_under_test(simulator):
    result = asyncio.run(benchmark.available())

    assert set(result['environment']) >= {'tier', 'llm_model', 'llm_provider',
                                          'host', 'image_tags', 'git_shas'}


# ── 用例依赖：分层报错 ────────────────────────────────────────────────────────

def test_a_driver_that_is_installed_and_online_is_not_reported_missing(simulator, monkeypatch):
    """在 Orin6 上抓到的：每个用例都报「缺驱动」，而驱动就在旁边跑着。

    名字住在注册表（`config.main['services']['mcp']`），在线与否住在
    `mcp_client.registry`，两份数据各答一半。原先只问运行时那份要 `server_name`，
    而它根本不存这个字段 —— 于是每个名字都比不上，全判成缺。这个 fixture 特意让
    registry 条目**没有** `server_name`，和真机上一模一样。
    """
    import config
    monkeypatch.setitem(config.main, 'services', {'mcp': [
        {'id': 'mcp-sim', 'name': '仿真器', 'server_name': 'simulator-generic-device-bundle'}]})

    ready = asyncio.run(benchmark.case_readiness({'drivers': ['simulator-generic-device-bundle']}))

    assert ready['missing_drivers'] == []
    assert ready['ok'] is True


def test_a_registered_but_offline_driver_still_counts_as_missing(simulator, monkeypatch):
    """装了但没起来，和没装一样跑不了 —— 要报出来。"""
    import config
    monkeypatch.setitem(config.main, 'services', {'mcp': [
        {'id': 'mcp-sim', 'name': '仿真器', 'server_name': 'simulator-generic'}]})
    monkeypatch.setitem(mcp_client.registry, 'mcp-sim',
                        {**mcp_client.registry['mcp-sim'], 'online': False})

    ready = asyncio.run(benchmark.case_readiness({'drivers': ['simulator-generic']}))

    assert ready['missing_drivers'] == ['simulator-generic']


def test_a_missing_driver_is_reported_before_assets_are_asked_about(simulator, monkeypatch):
    """驱动没装就不去问地图。

    那一问会走 MCP 超时，把「没装 simulator 驱动」这个清楚的结论，拖成一个
    「超时」的含糊结论 —— 而分层报错的全部意义就是让第一层先说话。
    """
    monkeypatch.delitem(mcp_client.registry, 'mcp-sim')

    ready = asyncio.run(benchmark.case_readiness(
        {'drivers': ['simulator-generic'], 'assets': ['bj-2f']}))

    assert ready['ok'] is False
    assert ready['missing_drivers'] == ['simulator-generic']
    assert ready['assets_checked'] is False
    assert simulator['calls'] == []


def test_a_missing_map_is_reported_once_the_driver_is_there(simulator):
    simulator['replies'][('sim_scenario', 'list_maps')] = {'maps': [{'name': 'lab'}]}
    _register('simulator-generic')

    ready = asyncio.run(benchmark.case_readiness(
        {'drivers': ['simulator-generic'], 'assets': ['bj-2f']}))

    assert ready['missing_drivers'] == []
    assert ready['missing_assets'] == ['bj-2f']
    assert ready['ok'] is False


def test_a_case_whose_dependencies_are_all_present_is_runnable(simulator):
    simulator['replies'][('sim_scenario', 'list_maps')] = {'maps': [{'name': 'bj-2f'}]}
    _register('simulator-generic')

    ready = asyncio.run(benchmark.case_readiness(
        {'drivers': ['simulator-generic'], 'assets': ['bj-2f']}))

    assert ready['ok'] is True and ready['missing_assets'] == []


def test_an_unreachable_simulator_is_not_reported_as_a_missing_asset(simulator):
    """问不到 ≠ 缺。报成缺失，一次临时故障就会看起来像装错了东西。"""
    simulator['replies'][('sim_scenario', 'list_maps')] = {'error': 'timeout'}
    _register('simulator-generic')

    ready = asyncio.run(benchmark.case_readiness(
        {'drivers': ['simulator-generic'], 'assets': ['bj-2f']}))

    assert ready['ok'] is True
    assert ready['missing_assets'] == [] and ready['assets_error'] == 'timeout'


# ── 覆盖前先存画布 ────────────────────────────────────────────────────────────

@pytest.fixture
def canvas_and_registry(monkeypatch):
    from api import solutions
    monkeypatch.setattr(solutions, '_layout', lambda: {
        'cards': [{'id': 'c1', 'mcpId': 'mcp-sim', 'toolName': 'nav', 'x': 0, 'y': 0}],
        'connections': [], 'execConnections': []})
    monkeypatch.setattr(solutions, '_mcp_list', lambda: [
        {'id': 'mcp-sim', 'name': 'simulator', 'server_name': 'simulator-generic',
         'url': 'http://localhost:15711/mcp'}])
    import config
    yield
    config.main[benchmark.SNAPSHOT_KEY] = None


def test_the_canvas_is_packed_before_a_case_overwrites_it(canvas_and_registry):
    """载入用例会覆盖画布，而用户自己搭的那套东西没有别的地方存着 —— 它就在画布上。"""
    result = asyncio.run(benchmark.snapshot(_request()))

    assert result['saved'] is True and result['cards'] == 1
    assert result['payload']['canvas']['cards'][0]['toolName'] == 'nav'


def test_the_snapshot_is_reported_but_never_restored_on_its_own(canvas_and_registry):
    """跑完自动把画布换回去，会在用户正看着结果的时候把画布抽走。还原是用户的决定。"""
    asyncio.run(benchmark.snapshot(_request()))

    info = asyncio.run(benchmark.snapshot_info())

    assert info['saved'] is True and info['cards'] == 1
    assert info['savedAt'] > 0


def test_restoring_without_a_snapshot_says_so_rather_than_doing_nothing():
    import config
    config.main[benchmark.SNAPSHOT_KEY] = None

    with pytest.raises(fastapi.HTTPException) as caught:
        asyncio.run(benchmark.snapshot_restore(_request()))

    assert caught.value.status_code == 404


def _request():
    class _Req:
        headers: dict = {}
        cookies: dict = {}
    return _Req()


# ── 跑完之后回看现场 ──────────────────────────────────────────────────────────

FACTS = {
    'events': [
        {'t': 1000.0, 'event': 'scenario_load'},
        {'t': 1020.0, 'event': 'nav_start', 'label': 'P3'},
        {'t': 1035.5, 'event': 'arrive', 'label': 'P3'},
        {'t': 1037.0, 'event': 'speak_start', 'text': 'P3 到了'},
        {'t': 1041.0, 'event': 'speak_end', 'text': 'P3 到了', 'status': 'completed'},
    ],
    'transcript': [{'text': 'P3 到了', 'status': 'completed'}],
    'acp_posts': [{'action_id': 'a1', 'status': 'completed'}],
}


def test_the_facts_survive_the_run_they_came_from():
    """仿真器的世界下一次跑动一开始就被重置 —— 不在写入时存一份，跑完就再也没法
    回答「分数为什么是这个」。"""
    run_id = benchmark_store.create_run('导览', n_repeats=1)
    benchmark_store.add_case(run_id, scenario='导览', repeat_idx=0, facts=FACTS)

    timeline = asyncio.run(benchmark.run_timeline(run_id))

    assert [w['kind'] for w in timeline['world']][:3] == [
        'scenario_load', 'nav_start', 'arrive']
    assert timeline['transcript'][0]['status'] == 'completed'


def test_world_times_are_relative_to_the_start_of_the_run():
    """世界那侧是仿真时钟，agent 那侧是墙钟 —— 两个钟相减没有意义，所以各自归一到
    「从本轮开始起算的秒数」，而不是编一个共同时间轴。"""
    run_id = benchmark_store.create_run('导览')
    benchmark_store.add_case(run_id, scenario='导览', facts=FACTS)

    world = asyncio.run(benchmark.run_timeline(run_id))['world']

    assert world[0]['at'] == 0.0
    assert world[2]['at'] == 35.5


def test_the_agent_track_is_grouped_by_turn(monkeypatch):
    """按轮取，不按消息取：一轮就是「被什么唤醒 → 想了什么 → 调了哪些工具」，
    排查时要看的正好是这个粒度。"""
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=? WHERE id=?', ('s1', run_id))
    conn.commit()

    import chat_history
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': stored['started_at'] + 2, 'updated_at': 0,
        'messages': [
            {'role': 'user', 'content': '带我转一下展区'},
            {'role': 'assistant', 'content': '好的，先去 P3。',
             'tool_calls': [{'function': {'name': 'mcp__m1__controlled_spatial',
                                          'arguments': '{"action":"navigate_to_tag"}'}}]},
        ]}])

    agent = asyncio.run(benchmark.run_timeline(run_id))['agent']

    assert agent[0]['trigger'] == '带我转一下展区'
    assert agent[0]['says'] == ['好的，先去 P3。']
    assert agent[0]['calls'][0]['name'] == 'controlled_spatial'
    assert agent[0]['at'] == 2.0


def test_turns_from_before_the_run_are_not_claimed_as_its_own(monkeypatch):
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=? WHERE id=?', ('s1', run_id))
    conn.commit()

    import chat_history
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [
        {'started_at': stored['started_at'] - 600, 'messages': [
            {'role': 'user', 'content': '上一场对话'}]},
        {'started_at': stored['started_at'] + 1, 'messages': [
            {'role': 'user', 'content': '这一场'}]},
    ])

    agent = asyncio.run(benchmark.run_timeline(run_id))['agent']

    assert [t['trigger'] for t in agent] == ['这一场']


def test_a_cleared_history_is_an_empty_track_not_an_error(monkeypatch):
    """会话被清过或压缩掉就返回空 —— 与其编一段出来，不如让前端说记录已经没有了。"""
    run_id = benchmark_store.create_run('导览')

    timeline = asyncio.run(benchmark.run_timeline(run_id))

    assert timeline['agent'] == []


def test_an_unknown_run_is_a_404():
    with pytest.raises(fastapi.HTTPException) as caught:
        asyncio.run(benchmark.run_timeline('nope'))

    assert caught.value.status_code == 404


def test_a_turn_that_began_before_the_run_but_grew_during_it_counts(monkeypatch):
    """Orin6 上抓到的：agent 那一栏整个是空的，而会话明明活跃在跑动窗口里。

    一轮是被反复重写的 —— `save_turn` 在 turn_index 已存在时走 UPDATE，`created_at`
    不变、`updated_at` 往后走。跑动期间写进去的内容，可能挂在一个**早就起了头**的
    轮次上。只看起始时刻，这一轮会被整个丢掉。
    """
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=?, ended_at=? WHERE id=?',
                 ('s1', stored['started_at'] + 400, run_id))
    conn.commit()

    import chat_history
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': stored['started_at'] - 4000,          # 跑动开始前就起了头
        'updated_at': stored['started_at'] + 100,           # 但内容写在跑动期间
        'messages': [{'role': 'user', 'content': '带我转一下展区'}]}])

    agent = asyncio.run(benchmark.run_timeline(run_id))['agent']

    assert [t['trigger'] for t in agent] == ['带我转一下展区']


def test_a_turn_that_ended_before_the_run_is_still_excluded(monkeypatch):
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=? WHERE id=?', ('s1', run_id))
    conn.commit()

    import chat_history
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': stored['started_at'] - 4000,
        'updated_at': stored['started_at'] - 3000,          # 整轮都结束在跑动之前
        'messages': [{'role': 'user', 'content': '上一场'}]}])

    assert asyncio.run(benchmark.run_timeline(run_id))['agent'] == []


def test_a_turns_time_comes_from_when_it_was_written(monkeypatch):
    """真机上排出过「第 11 轮 +-277189.7s」—— 三天前的时间戳。

    一轮的行是会被**覆盖**的：agent-core 重启后 `_turns` 从上一个会话重新载入，轮号
    从小往大重排，`save_turn` 撞上旧行就走 UPDATE —— `created_at` 停在几天前，内容
    却是刚写的。最后写入的时刻才是这轮真正发生的时刻。
    """
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=? WHERE id=?', ('s1', run_id))
    conn.commit()

    import chat_history
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': stored['started_at'] - 277189,        # 被覆盖的旧行
        'updated_at': stored['started_at'] + 12,            # 真正写入的时刻
        'messages': [{'role': 'assistant', 'content': '地图已加载。'}]}])

    agent = asyncio.run(benchmark.run_timeline(run_id))['agent']

    assert agent[0]['at'] == 12.0


def test_turns_are_numbered_from_the_start_of_this_run(monkeypatch):
    """打开的是一次跑动的现场，第一轮却写着「第 11 轮」，读的人会以为前面漏了十轮。
    会话里的序号留在 `sessionTurn`，要和历史面板对照时还用得上。"""
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=? WHERE id=?', ('s1', run_id))
    conn.commit()

    import chat_history
    old = {'started_at': stored['started_at'] - 9000,
           'updated_at': stored['started_at'] - 8000, 'messages': []}
    mine = {'started_at': stored['started_at'] + 5,
            'updated_at': stored['started_at'] + 9, 'messages': [
                {'role': 'assistant', 'content': '开始'}]}
    monkeypatch.setattr(chat_history, 'get_session_turns',
                        lambda sid: [old, old, old, old, old, old, old, old, old, old, mine])

    agent = asyncio.run(benchmark.run_timeline(run_id))['agent']

    assert agent[0]['turn'] == 0
    assert agent[0]['sessionTurn'] == 10


def test_a_time_outside_the_run_window_is_not_shown(monkeypatch):
    """真机上排出过「+1980s」——一次 7 分钟的跑动里，33 分钟处的一轮。

    一轮可能在跑动之前起头、在跑动之后还在被追写，两个时间戳都不在窗口里。那种情况
    宁可不给数，也不给一个看起来精确的假数。
    """
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=?, ended_at=? WHERE id=?',
                 ('s1', stored['started_at'] + 420, run_id))
    conn.commit()

    import chat_history
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': stored['started_at'] - 100,
        'updated_at': stored['started_at'] + 1980,      # 跑动结束 26 分钟之后还在追写
        'messages': [{'role': 'assistant', 'content': '地图可用了'}]}])

    turn = asyncio.run(benchmark.run_timeline(run_id))['agent'][0]

    assert turn['at'] is None
    assert turn['timing'] == 'outside'
    assert turn['says'] == ['地图可用了']       # 内容照常给，只是不编时间


def test_the_trigger_prefers_what_a_person_said(monkeypatch):
    """`<status …>` 是每轮都附的环境快照，几百字。拿它当「触发」既占满一屏，
    又什么都没说。"""
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=? WHERE id=?', ('s1', run_id))
    conn.commit()

    import chat_history
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': stored['started_at'], 'updated_at': stored['started_at'] + 3,
        'messages': [
            {'role': 'user', 'content': '<status time="2026-09-20"> <active_tasks>…'},
            {'role': 'user', 'content': '先等一下，我想先看看那边那个算力工厂'},
        ]}])

    assert asyncio.run(benchmark.run_timeline(run_id))['agent'][0]['trigger'] \
        == '先等一下，我想先看看那边那个算力工厂'


def test_a_turn_woken_only_by_a_status_refresh_says_so(monkeypatch):
    run_id = benchmark_store.create_run('导览')
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=? WHERE id=?', ('s1', run_id))
    conn.commit()

    import chat_history
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': stored['started_at'], 'updated_at': stored['started_at'] + 3,
        'messages': [{'role': 'user', 'content': '<status time="2026-09-20">…</status>'}]}])

    assert asyncio.run(benchmark.run_timeline(run_id))['agent'][0]['trigger'] == '（状态刷新）'


def test_the_agent_track_is_frozen_when_the_run_ends(monkeypatch):
    """会话里的轮次是**活的**：跑完之后 agent 接着工作，同一批行继续被改写。

    真机上同一条记录先显示 +32.8s，几分钟后变成「时间落在本轮之外」—— 那次跑动
    一个字都没变，变的是它读的那份数据。
    """
    run_id = benchmark_store.create_run('导览')
    benchmark_store.finish_run(run_id, score_total=80.0, agent_track=[
        {'turn': 0, 'at': 32.8, 'timing': 'exact', 'trigger': '带我转一下',
         'says': ['好的'], 'calls': []}])

    import chat_history
    # 之后会话被改写成另一个样子 —— 定格的那份不该受影响
    monkeypatch.setattr(chat_history, 'get_session_turns',
                        lambda sid: [{'started_at': 0, 'updated_at': 0, 'messages': []}])

    agent = asyncio.run(benchmark.run_timeline(run_id))['agent']

    assert agent[0]['at'] == 32.8
    assert agent[0]['says'] == ['好的']


def test_a_run_left_running_by_a_restart_is_marked_interrupted():
    """重启带走了那个在跑的 task，记录却留在原地 —— 不清理，面板每次打开都会报
    一次并不存在的跑动。"""
    run_id = benchmark_store.create_run('导览')
    assert benchmark_store.get_run(run_id)['status'] == 'running'

    benchmark_store.mark_stale_runs()

    stored = benchmark_store.get_run(run_id)
    assert stored['status'] == 'interrupted'
    assert stored['ended_at'] is not None


def test_marking_stale_runs_leaves_finished_ones_alone():
    done = benchmark_store.create_run('导览')
    benchmark_store.finish_run(done, score_total=90.0)

    benchmark_store.mark_stale_runs()

    assert benchmark_store.get_run(done)['status'] == 'done'
    assert benchmark_store.get_run(done)['score_total'] == 90.0


# ── 两栏对齐 ──────────────────────────────────────────────────────────────────

def _run_with_session(name='导览'):
    run_id = benchmark_store.create_run(name)
    stored = benchmark_store.get_run(run_id)
    conn = benchmark_store._get_conn()
    conn.execute('UPDATE benchmark_run SET session_id=?, ended_at=? WHERE id=?',
                 ('s1', stored['started_at'] + 300, run_id))
    conn.commit()
    return run_id, stored['started_at']


def test_a_turns_time_comes_from_the_spans_not_from_when_it_was_written(monkeypatch):
    """真机上左栏 +14.5s 的那次讲解，右栏 +8.7s 就开讲了 —— 会话历史只记「这一轮
    什么时候被写完」，而一轮里的动作发生在那之前，左栏整体晚一整轮的时长。"""
    run_id, started = _run_with_session()

    import chat_history
    import perf_log
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': started, 'updated_at': started + 14.5, 'messages': [
            {'role': 'user', 'content': '带我转一下展区'},
            {'role': 'assistant', 'content': '好的',
             'tool_calls': [{'function': {'name': 'mcp__m__tts', 'arguments': '{}'}}]}]}])
    monkeypatch.setattr(perf_log, 'turns_between', lambda a, b: [{
        'turn_id': 't1', 'trigger_text': '带我转一下展区',
        'spans': [{'span': 'turn_total', 'start_ts': started + 5.0},
                  {'span': 'tool:tts', 'start_ts': started + 8.7}]}])

    turn = asyncio.run(benchmark.run_timeline(run_id))['agent'][0]

    assert turn['at'] == 5.0 and turn['timing'] == 'exact'
    assert turn['calls'][0]['at'] == 8.7


def test_without_spans_the_time_is_marked_as_only_the_write_moment(monkeypatch):
    """没有 spans 就只知道写完的时刻 —— 它比实际做事晚一整轮，界面要看得出区别。"""
    run_id, started = _run_with_session()

    import chat_history
    import perf_log
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': started, 'updated_at': started + 14.5,
        'messages': [{'role': 'user', 'content': '带我转一下展区'}]}])
    monkeypatch.setattr(perf_log, 'turns_between', lambda a, b: [])

    turn = asyncio.run(benchmark.run_timeline(run_id))['agent'][0]

    assert turn['at'] == 14.5 and turn['timing'] == 'written'


def test_hook_fired_calls_do_not_break_the_match(monkeypatch):
    """perf 记每一次派发，会话只记 LLM 自己发起的调用 —— `on_notify` 钩子自动播报
    的那次 speak 有 span，却不在会话的 tool_calls 里。

    真机上八轮里六轮因此配不上：
        会话  ['navigate_to_tag', 'speak', 'task_update']
        perf  ['navigate_to_tag', 'speak', 'speak', 'task_update']
    所以是子序列，不是相等。
    """
    run_id, started = _run_with_session()

    import chat_history
    import perf_log
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': started, 'updated_at': started + 30, 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'function': {'name': 'mcp__m__controlled_spatial', 'arguments': '{}'}},
                {'function': {'name': 'mcp__m__tts', 'arguments': '{}'}},
                {'function': {'name': 'task_update', 'arguments': '{}'}}]}]}])
    monkeypatch.setattr(perf_log, 'turns_between', lambda a, b: [{
        'turn_id': 't1', 'trigger_text': 'x', 'spans': [
            {'span': 'turn_total', 'start_ts': started + 1},
            {'span': 'tool:controlled_spatial', 'start_ts': started + 2},
            {'span': 'tool:tts', 'start_ts': started + 5},
            {'span': 'tool:tts', 'start_ts': started + 9},     # 钩子播报，不在会话里
            {'span': 'tool:task_update', 'start_ts': started + 11}]}])

    turn = asyncio.run(benchmark.run_timeline(run_id))['agent'][0]

    assert turn['timing'] == 'exact'
    assert [c.get('at') for c in turn['calls']] == [2.0, 5.0, 11.0]


def test_a_turn_whose_calls_are_not_in_the_span_list_is_not_matched(monkeypatch):
    """多出来的可以忽略，缺掉的不行：会话里有、perf 里没有，就不是同一轮。"""
    run_id, started = _run_with_session()

    import chat_history
    import perf_log
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': started, 'updated_at': started + 10, 'messages': [
            {'role': 'user', 'content': '带我转一下'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'function': {'name': 'mcp__m__tts', 'arguments': '{}'}},
                {'function': {'name': 'mcp__m__loco', 'arguments': '{}'}}]}]}])
    monkeypatch.setattr(perf_log, 'turns_between', lambda a, b: [{
        'turn_id': 't1', 'trigger_text': '带我转一下',
        'spans': [{'span': 'turn_total', 'start_ts': started + 1},
                  {'span': 'tool:tts', 'start_ts': started + 3}]}])

    turn = asyncio.run(benchmark.run_timeline(run_id))['agent'][0]

    assert turn['timing'] == 'written'
    assert all('at' not in c for c in turn['calls'])


def test_matching_is_not_by_trigger_text(monkeypatch):
    """两边存的根本不是同一个串：perf 存原始的 `<event source=…>`，会话存 `<status …>`。

    真机上一比就知道 —— 而配不上的后果是整列悄悄退回「写完时」，看起来只是少了点
    精度，不像出错。所以配对靠调用名序列，不靠 trigger 文本。
    """
    run_id, started = _run_with_session()

    import chat_history
    import perf_log
    monkeypatch.setattr(chat_history, 'get_session_turns', lambda sid: [{
        'started_at': started, 'updated_at': started + 20, 'messages': [
            {'role': 'user', 'content': '<status time="2026-09-20 13:50:54">…'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'function': {'name': 'mcp__m__tts', 'arguments': '{}'}}]}]}])
    monkeypatch.setattr(perf_log, 'turns_between', lambda a, b: [{
        'turn_id': 't1',
        'trigger_text': '<event source="acp:sim-tts-bb312a896304" channel="sensor">',
        'spans': [{'span': 'turn_total', 'start_ts': started + 2},
                  {'span': 'tool:tts', 'start_ts': started + 4}]}])

    turn = asyncio.run(benchmark.run_timeline(run_id))['agent'][0]

    assert turn['timing'] == 'exact' and turn['at'] == 2.0
    assert turn['calls'][0]['at'] == 4.0
