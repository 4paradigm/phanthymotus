"""跑一个用例的执行底座。

三件事在这里定死，每一件都是这次架构调整的理由：

* **被测的 agent 不能重置测量。** 跑动由 agent-core 驱动，不是 LLM 可调用的一张卡片。
* **初始指令走用户消息那条路。** 用例问的就是「用户说了这句话之后会发生什么」。
* **画布上会动的真设备，开跑前必须经人确认。** 注入的文本和真实指令无法区分，
  所以这些设备会真的动起来。从前这里是「一律拒绝」，那在真机上等于永远不能跑 ——
  真机上每一张执行器卡都不属于仿真器。现在由现场的人决定，而**确认必须对得上服务端
  此刻看到的那一组设备**。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_runner.py -q
"""

import asyncio
import os
import pathlib
import sys
import tempfile

import fastapi
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'runner-test.db'))

import benchmark_runner  # noqa: E402
import benchmark_store  # noqa: E402
import config  # noqa: E402
import event_bus  # noqa: E402
import benchmark_judge  # noqa: E402
import mcp_client  # noqa: E402

CASE = {
    'name': 'bj-2f-tour',
    'requires': {'drivers': ['simulator-generic'], 'assets': ['bj-2f']},
    'run': {'prompt': '带我转一下展区并给我介绍下',
            'injections': [{'after_action': 1, 'delay': 0.05, 'text': '先等一下'}],
            'budget_seconds': 30, 'idle_seconds': 0.01},
    'requirements': [{'id': 'r0', 'text': '按顺序走完每一站', 'weight': 30,
                      'dimension': 'world_timing'}],
}


def card(mcp_id, tool):
    return {'id': f'{mcp_id}-{tool}', 'mcpId': mcp_id, 'toolName': tool, 'x': 0, 'y': 0}


@pytest.fixture
def canvas():
    """画布布局存在 ConfigDB 里，取出来的是副本 —— 改副本不算数，整份写回去才算。"""
    def put(*cards):
        config.main['canvas_layout'] = {'cards': list(cards), 'connections': [],
                                        'execConnections': []}
    yield put
    config.main['canvas_layout'] = {'cards': [], 'connections': [], 'execConnections': []}


@pytest.fixture
def registry(monkeypatch):
    # 层（category）记在注册表那一份里，不在运行时 map —— 真机上就是这么分的。
    monkeypatch.setitem(config.main, 'services', {'mcp': [
        {'id': 'mcp-sim', 'server_name': 'simulator-generic', 'category': 'driver'},
        {'id': 'mcp-real', 'server_name': 'x-humanoid-tianyi', 'category': 'driver'},
        {'id': 'mcp-perc', 'server_name': 'perception-bundle', 'category': 'perception'},
        {'id': 'agentcore', 'server_name': 'AgentCore', 'category': 'controller',
         'transport': 'internal'},
    ]})
    monkeypatch.setitem(mcp_client.registry, 'mcp-perc', {
        'online': True, 'name': 'Perception Stack',
        'tool_meta': {'mcp__mcp-perc__tts': {'type': 'actuator'}},
    })
    monkeypatch.setitem(mcp_client.registry, 'agentcore', {
        'online': True, 'name': 'AgentCore',
        'tool_meta': {'mcp__agentcore__decision_core': {'type': 'controller'}},
    })
    monkeypatch.setitem(mcp_client.registry, 'mcp-sim', {
        'online': True, 'name': 'simulator', 'server_name': 'simulator-generic',
        'category': 'driver',
        'tool_meta': {'mcp__mcp-sim__loco': {'type': 'actuator'},
                      'mcp__mcp-sim__odom': {'type': 'sensor'}},
    })
    monkeypatch.setitem(mcp_client.registry, 'mcp-real', {
        'online': True, 'name': '天轶', 'server_name': 'x-humanoid-tianyi',
        'category': 'driver',
        'tool_meta': {'mcp__mcp-real__loco': {'type': 'actuator'},
                      'mcp__mcp-real__controlled_spatial': {'type': 'actuator'},
                      'mcp__mcp-real__camera_head': {'type': 'sensor'}},
    })


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    benchmark_runner.set_current(None)
    yield
    benchmark_runner.set_current(None)
    for run in benchmark_store.list_runs(limit=200):
        benchmark_store.delete_run(run['id'])


# ── 安全闸 ────────────────────────────────────────────────────────────────────

def test_a_real_robots_actuator_on_the_canvas_blocks_the_run(canvas, registry):
    """一句「带我转一下展区」会让真机器人走起来 —— 没人确认过，也没人在旁边。"""
    canvas(card('mcp-sim', 'loco'), card('mcp-real', 'controlled_spatial'))

    unsafe = benchmark_runner.unsafe_cards('mcp-sim')

    assert [u['tool'] for u in unsafe] == ['controlled_spatial']
    assert unsafe[0]['device'] == '天轶'


def test_a_real_robots_sensor_does_not_block_the_run(canvas, registry):
    """读不会让任何东西动；把传感器也拦下来，等于没人能一边跑用例一边看真相机。"""
    canvas(card('mcp-sim', 'loco'), card('mcp-real', 'camera_head'))

    assert benchmark_runner.unsafe_cards('mcp-sim') == []


def test_the_simulators_own_actuators_are_fine(canvas, registry):
    canvas(card('mcp-sim', 'loco'))

    assert benchmark_runner.unsafe_cards('mcp-sim') == []


def test_agent_cores_own_pseudo_devices_never_block_a_run(canvas, registry):
    """Orin6 上抓到的：`decision_core` 被判成「会动的真实设备」，于是任何用例都跑不了。

    它按定义在每张画布上 —— 它就是被测的那个系统，不是一台会走起来的机器。这条闸
    要是把它算进去，就只剩一个永远亮着的红灯。"""
    canvas(card('agentcore', 'decision_core'), card('agentcore', 'remote_message'))

    assert benchmark_runner.unsafe_cards('mcp-sim') == []


def test_a_perception_tool_does_not_block_a_run(canvas, registry):
    """同一次跑动里抓到的第二条：perception 的 `tts` 申报 actuator，但它不驱动任何
    东西 —— 层比类型更有发言权，而层原先读的是一个从来没人写过的字段。"""
    canvas(card('mcp-perc', 'tts'))

    assert benchmark_runner.unsafe_cards('mcp-sim') == []


def test_an_undeclared_type_counts_as_acting(canvas, registry):
    """猜名字漏掉过 loco/led/speaker/switch_mode。没申报就按会动处理。"""
    canvas(card('mcp-real', 'whatever'))

    assert len(benchmark_runner.unsafe_cards('mcp-sim')) == 1


def test_a_split_sensor_is_not_listed_as_something_that_moves(canvas, registry,
                                                              monkeypatch):
    """带 action 枚举的工具会被按 action 拆开，`tool_meta` 里只有
    `mcp__<id>__mic__start` 这样的键，`mcp__<id>__mic` 根本不存在。

    画布卡片记的是**工具**名，所以拼出来的名字查不到申报 → `tool_type` 返回 '' →
    「没申报就算会动」→ 麦克风进了「会动的设备」清单。方向上安全，但一份把麦克风算
    进去的清单，人就不会认真读了 —— 而这份清单的全部作用就是让人认真读一遍。
    """
    monkeypatch.setitem(mcp_client.registry, 'mcp-real', {
        'online': True, 'name': '天轶', 'server_name': 'x-humanoid-tianyi',
        'category': 'driver',
        'tool_meta': {'mcp__mcp-real__mic__start': {'type': 'sensor'},
                      'mcp__mcp-real__mic__stop': {'type': 'sensor'},
                      'mcp__mcp-real__loco__walk': {'type': 'actuator'}},
    })
    canvas(card('mcp-real', 'mic'), card('mcp-real', 'loco'))

    assert [u['tool'] for u in benchmark_runner.unsafe_cards(None)] == ['loco']


def test_a_tool_with_no_declaration_anywhere_still_counts_as_acting(canvas, registry,
                                                                    monkeypatch):
    """兜底不能松：查不到就是查不到，按会动处理。"""
    monkeypatch.setitem(mcp_client.registry, 'mcp-real', {
        'online': True, 'name': '天轶', 'category': 'driver', 'tool_meta': {}})
    canvas(card('mcp-real', 'mystery'))

    assert [u['tool'] for u in benchmark_runner.unsafe_cards(None)] == ['mystery']


def test_without_a_simulator_every_actuator_needs_confirming(canvas, registry):
    """真机上没有仿真器可以代劳，所以画布上会动的东西**全部**要进确认清单。

    这也正是从前那条「非空就拒绝」在真机上的样子：清单永远非空，于是永远拒绝。
    """
    canvas(card('mcp-real', 'controlled_spatial'), card('mcp-real', 'whatever'),
           card('mcp-real', 'camera_head'), card('agentcore', 'decision_core'))

    moving = benchmark_runner.unsafe_cards(None)

    assert sorted(u['tool'] for u in moving) == ['controlled_spatial', 'whatever']


# ── 同一时刻只许一次跑动 ──────────────────────────────────────────────────────

def test_claiming_is_atomic_so_two_requests_cannot_both_win():
    """**检查与占位必须是同一步。**

    `/case/run` 在检查之后、真正接管之前有一个 await（依赖检查，还可能走 MCP 网络）。
    先查后占的话，两个并发请求会双双通过 —— 两次跑动同时往同一个 agent 注入用户消息、
    同时重置世界，两份事实流交织在一起，而分数看起来只是「莫名其妙地低」。
    """
    assert benchmark_runner.claim() is True
    assert benchmark_runner.claim() is False       # 第二个请求拿不到
    assert benchmark_runner.is_busy() is True

    benchmark_runner.release()

    assert benchmark_runner.is_busy() is False
    assert benchmark_runner.claim() is True
    benchmark_runner.release()


def test_a_failed_start_gives_the_slot_back():
    """占上了却没跑成而不还，面板会一直说「已经有一次基准测试在跑」，
    而实际上什么都没跑 —— 只能重启 agent-core 才能再跑。"""
    import asyncio

    from api import benchmark as bm_api

    with pytest.raises(fastapi.HTTPException):
        asyncio.run(bm_api.run_case(bm_api.CaseRunRequest(case_id='nope')))

    assert benchmark_runner.is_busy() is False


def test_handing_over_to_a_real_run_clears_the_claim():
    benchmark_runner.claim()
    run_id = benchmark_store.create_run('t')
    run = benchmark_runner.CaseRun(CASE, 'mcp-sim', 1, 0, run_id, {}, case_id='c1')

    benchmark_runner.set_current(run)

    assert benchmark_runner.is_busy() is True      # 现在忙的是真正的跑动
    assert benchmark_runner.current().snapshot()['case_id'] == 'c1'
    benchmark_runner.set_current(None)


# ── 开跑前的确认 ──────────────────────────────────────────────────────────────
#
# 这一组盯的是端点，不是分类。分类改对了而端点放行，等于一台真机器人在没人确认的
# 情况下走起来。

from api import benchmark as bm_api  # noqa: E402

MOVING = [{'mcpId': 'mcp-real', 'tool': 'controlled_spatial', 'device': '天轶'}]


def test_no_confirmation_means_no_run():
    with pytest.raises(fastapi.HTTPException) as caught:
        bm_api._check_confirmation(MOVING, None)

    assert caught.value.status_code == 409
    assert caught.value.detail['needs_confirmation'] is True
    assert caught.value.detail['moving_cards'] == MOVING


def test_a_confirmation_that_matches_lets_the_run_start():
    bm_api._check_confirmation(MOVING, ['mcp-real:controlled_spatial'])


def test_a_confirmation_for_a_different_set_is_refused():
    """画布在「弹窗弹出」和「点开始运行」之间是可以改的。

    人看着仿真器的 tts 按了确认，另一个标签页把卡片换成真机底盘 —— 那个勾就为一组
    他从没看见过的设备背了书。所以服务端重算，客户端送来的只用来核对。
    """
    with pytest.raises(fastapi.HTTPException) as caught:
        bm_api._check_confirmation(MOVING, ['mcp-sim:tts'])

    assert caught.value.status_code == 409
    assert '变了' in caught.value.detail['error']


def test_a_partial_confirmation_is_refused():
    """确认了一台、实际会动两台 —— 这不是「确认过了」。"""
    two = MOVING + [{'mcpId': 'mcp-real', 'tool': 'loco', 'device': '天轶'}]

    with pytest.raises(fastapi.HTTPException):
        bm_api._check_confirmation(two, ['mcp-real:controlled_spatial'])


def test_an_empty_confirmation_list_does_not_authorise_real_devices():
    """空列表是「我什么都没确认」，不是「确认了空集」。"""
    with pytest.raises(fastapi.HTTPException):
        bm_api._check_confirmation(MOVING, [])


def test_nothing_moves_so_nothing_is_asked():
    """仿真器在场时清单为空，不该弹窗 —— 行为和从前一字不差。"""
    bm_api._check_confirmation([], None)


# ── 触发时刻 ──────────────────────────────────────────────────────────────────

def test_an_arrival_triggered_injection_waits_for_the_arrival():
    """真机上 LLM 一轮 3-48 秒。按绝对时刻插话会落到完全不同的一段路上，
    于是「在去 P6 的路上被打断」测的其实是别的东西。"""
    injection = {'after_arrival': 'P5', 'delay': 6.0, 'text': '先等一下'}

    before = benchmark_runner._trigger_due(injection, [], 12.0)
    after = benchmark_runner._trigger_due(
        injection, [{'event': 'arrive', 'label': 'P5', 't': 40.0}], 41.0)

    assert before is None
    assert after == 47.0


def test_the_delay_is_counted_from_seeing_the_arrival_not_from_its_timestamp():
    """事件的 `t` 是驱动的时钟，加速倍率下和墙钟不是一回事 ——
    两个时钟相减出来的秒数没有意义。"""
    sped_up = [{'event': 'arrive', 'label': 'P5', 't': 300.0}]

    assert benchmark_runner._trigger_due(
        {'after_arrival': 'P5', 'delay': 6.0}, sped_up, 30.0) == 36.0


def test_an_injection_can_fire_after_the_nth_action_whatever_that_action_was():
    """`after_action` 是 `after_arrival` 的通用化 —— 「到达某一站」是展区导览的说法，
    「第 N 个动作做完」在任何用例里都成立，判据也一样在事实流里。"""
    two_done = [{'event': 'arrive', 't': 5.0}, {'event': 'speak_end', 't': 9.0}]

    assert benchmark_runner._trigger_due({'after_action': 3, 'delay': 1.0},
                                         two_done, 20.0) is None
    assert benchmark_runner._trigger_due({'after_action': 2, 'delay': 1.0},
                                         two_done, 20.0) == 21.0


def test_the_old_after_arrival_trigger_still_works():
    """旧用例还在用它。编辑器不再提供，但读到了要认。"""
    arrived = [{'event': 'arrive', 'label': 'P5', 't': 300.0}]

    assert benchmark_runner._trigger_due(
        {'after_arrival': 'P5', 'delay': 6.0}, arrived, 30.0) == 36.0


# ── 一次完整跑动 ──────────────────────────────────────────────────────────────

@pytest.fixture
def simulator(monkeypatch):
    """一个走完 P3→P4 的世界，外加记录它被要求做了什么。"""
    calls = []
    events = [{'event': 'nav_start', 'label': 'P3', 't': 0.0},
              {'event': 'arrive', 'label': 'P3', 't': 5.0},
              {'event': 'nav_start', 'label': 'P4', 't': 6.0},
              {'event': 'arrive', 'label': 'P4', 't': 9.0}]

    async def fake_call(mcp_id, tool, args):
        calls.append((tool, dict(args)))
        if tool == 'sim_report':
            return {'events': events, 'acp_posts': [], 'transcript': []}
        return {'ok': True}

    monkeypatch.setattr(mcp_client, 'call_tool_direct', fake_call)
    monkeypatch.setattr(benchmark_runner, '_POLL_SECONDS', 0.0)
    return calls


@pytest.fixture(autouse=True)
def judge(monkeypatch):
    """裁判打桩。

    `_one` 现在跑完会调 LLM 裁判 —— 不打桩的话，每一条跑动测试都会发真实请求，在没有
    网络的机器上表现为「运行失败」，而失败原因和被测的东西毫无关系。

    确定性那一半（默认目标对着指标判）照常真跑，本来就不该打桩 —— 那一半不花钱、
    不联网，正是「模糊判定要少」换来的好处。
    """
    calls = []

    async def fake(case, observations, facts, agent_track, model=None):
        calls.append({'observations': observations, 'facts': facts,
                      'agent_track': agent_track})
        return {'items': [], 'model': 'fake-judge', 'error': ''}

    monkeypatch.setattr(benchmark_judge, 'judge', fake)
    return calls


@pytest.fixture
def said(monkeypatch):
    messages = []

    async def fake_enqueue(source, text, payload=None):
        messages.append({'source': source, 'text': text, 'payload': payload or {}})

    monkeypatch.setattr(event_bus, 'enqueue', fake_enqueue)
    return messages


def test_the_prompt_enters_as_a_user_message(simulator, said):
    """不是新开一个入口：`message` 是 collector 认的 priority source，普通 source
    会返回 200 然后只进后台批 —— 看起来像发了但没反应。"""
    run = run_once()

    assert run.state == 'done', run.error
    assert said[0]['source'] == 'message'
    assert said[0]['text'] == '带我转一下展区并给我介绍下'


def test_the_injection_is_sent_even_when_the_world_finished_first(simulator, said):
    """世界跑得比 LLM 快时站点会先到齐。就这么收尾的话打断根本没发生过，
    而打断那几条断言会记在 agent 头上。"""
    run_once()

    texts = [m['text'] for m in said]
    assert texts == ['带我转一下展区并给我介绍下', '先等一下']


def test_every_spoken_line_also_lands_in_the_world_log(simulator, said):
    """判定用不着它，**人**要看：`sim_report` 是事后复盘的地方，而一条「机器人好好
    走着突然掉头」的记录里没有那句插话，读的人无从知道为什么。说话经由 collector，
    不经由仿真器，所以不补这一笔它就不在场。"""
    run_once()

    notes = [args.get('text') for tool, args in simulator
             if tool == 'sim_scenario' and args.get('action') == 'note']

    assert notes[0].startswith('[用例指令]'), '开场那句是指令，不是插话'
    assert any(n.startswith('[用例插话]') and '先等一下' in n for n in notes)


def test_a_failed_note_does_not_stop_the_run(simulator, said, monkeypatch):
    """记账失败不该让一次跑动停下来。"""
    async def flaky(mcp_id, tool, args):
        if tool == 'sim_scenario' and args.get('action') == 'note':
            raise RuntimeError('仿真器忙')
        return await _ok(mcp_id, tool, args)

    _ok = mcp_client.call_tool_direct
    monkeypatch.setattr(mcp_client, 'call_tool_direct', flaky)

    run = run_once()

    assert run.state == 'done', run.error


def test_a_case_that_names_no_map_leaves_the_world_where_it_is(simulator, said):
    """用例不再声明地图和出生点 —— **跑在当前世界上**，和「跑在当前画布上」是同一件事。

    仿真器仍然被 reset（世界要回到一个干净的起点），但 `map` 是空的，也就是「保持你
    现在这张图」。`world` 还写在包体里的旧用例照旧生效，只是编辑器不再提供这一项。
    """
    run_once()

    args = simulator[0][1]
    assert args['action'] == 'reset' and args['owner'] == benchmark_runner.OWNER
    assert args['map'] == '' and args['spawn'] == {}


def test_the_run_ends_when_the_world_goes_quiet(simulator, said):
    """收尾条件换过一版：原先是「所有期望站点都到过了」—— 那要求用例先声明一串站名，
    也就是把展区导览的词汇焊进了收尾逻辑。现在只剩预算耗尽和安静下来两条。"""
    run = run_once()

    assert run.state == 'done'
    assert run.cases[0]['elapsed'] < 30       # 远没跑满预算


def test_a_run_is_not_declared_over_while_an_action_is_still_pending(simulator, said,
                                                                     monkeypatch):
    """没有新事实常常只是因为机器人正走在半路上 —— 一段两分钟的导航期间事实流就是
    不动的。光看「没有新事实」会在半路上把运行判结束，然后给一个「什么都没做完」的分。
    """
    monkeypatch.setattr(mcp_client, 'get_pending_actions', lambda: ['a1'])
    monkeypatch.setitem(CASE['run'], 'budget_seconds', 0.3)

    run = run_once()

    monkeypatch.setitem(CASE['run'], 'budget_seconds', 30)
    # 安静了，但有动作没完 —— 只能等到预算耗尽。
    assert run.cases[0]['elapsed'] >= 0.3


def test_the_run_records_when_the_user_spoke_and_when_it_interrupted(simulator, said):
    """UX 那几个指标要知道这两个时刻，而事实流里没有 —— 它记的是机器人做了什么，
    不是用户什么时候说的话。"""
    run = run_once()

    marks = run._marks
    # 从「用户说完」起算，不从运行开始起算 —— 否则世界重置那几秒会被算进用户的等待。
    assert marks['prompt_at'] >= marks['started']
    assert len(marks['injections_at']) == 1
    assert marks['injections_at'][0] >= marks['prompt_at']


def test_a_legacy_case_that_still_carries_a_map_is_honoured(simulator, said):
    legacy = {**CASE, 'run': {**CASE['run'], 'world': {'map': 'bj-2f',
                                                       'spawn': {'x': 5.98}}}}
    run_once(case=legacy)

    args = simulator[0][1]
    assert args['map'] == 'bj-2f' and args['spawn']['x'] == 5.98


def test_the_score_is_stored_with_the_run(simulator, said):
    run = run_once()

    stored = benchmark_store.get_run(run.run_id)
    assert stored['status'] == 'done'
    assert stored['score_total'] == run.cases[0]['score']['total']


def test_repeats_each_get_their_own_seed(simulator, said):
    run = run_once(repeats=3, seed=10)

    assert [c['seed'] for c in run.cases] == [10, 11, 12]
    assert benchmark_store.get_run(run.run_id)['score_stdev'] is not None \
        or len({c['score']['total'] for c in run.cases}) == 1


def run_once(repeats=1, seed=0, case=None):
    run_id = benchmark_store.create_run('bj-2f-tour', n_repeats=repeats)
    run = benchmark_runner.CaseRun(case or CASE, 'mcp-sim', repeats, seed, run_id, {})
    benchmark_runner.set_current(run)
    asyncio.run(run._drive())
    return run


# ── 「开始智能控制」这道闸 ────────────────────────────────────────────────────

def test_the_background_route_is_closed_until_control_starts():
    """天轶上抓到的：容器一起来，还没点「开始智能控制」，后台 subagent 就每秒在刷。

    `project_running` 此前**不控制任何东西** —— `start.py` 只拿它清上次的残留，
    而 collector 与 run_forever 都是无条件起跑的。于是已有的 DDS 订阅和 MCP SSE
    一直在灌，一天 3668 个 turn，没人点过「开始」。
    """
    import collector
    config.main['core'] = {**(config.main.get('core') or {}), 'project_running': False}

    assert collector.project_running() is False

    config.main['core'] = {**(config.main.get('core') or {}), 'project_running': True}

    assert collector.project_running() is True


def test_running_a_case_before_control_starts_is_refused():
    """没开控制就跑，初始指令送进去没人处理，最后记一个 0 分 —— 那是基准测试在撒谎。"""
    import collector
    from api import benchmark as bm
    config.main['core'] = {**(config.main.get('core') or {}), 'project_running': False}
    try:
        with pytest.raises(fastapi.HTTPException) as caught:
            asyncio.run(bm.run_case(bm.CaseRunRequest(repeats=1)))
        assert caught.value.status_code == 409
    finally:
        config.main['core'] = {**(config.main.get('core') or {}), 'project_running': True}


def test_the_gate_announces_itself(capsys):
    """静默丢弃和坏掉的机器人从外面看一模一样 ——「为什么不理我」是排查时最贵的
    那类问题。进入和离开丢弃状态各说一次，并报出这期间扔了多少条。"""
    import collector
    collector._gated_since, collector._gated_count = None, 0

    collector._note_gated('dds:/camera/objects')
    collector._note_gated('dds:/camera/objects')
    first = capsys.readouterr().out

    collector._note_ungated()
    second = capsys.readouterr().out

    assert '智能控制未启动' in first and first.count('智能控制未启动') == 1  # 只说一次
    assert 'dds:/camera/objects' in first
    assert '丢弃 2 条' in second
