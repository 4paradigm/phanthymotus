"""用例的结构校验与裁判。

裁判搬到 agent-core，是为了让它和被测系统真的不相交：原先 `assertions.py` 住在
仿真器驱动里 —— 裁判住在被测系统内部。它本来就是纯函数
`(用例, 事件, ACP记录) → 判定`，没有驱动依赖，所以搬过来之后变成**驱动产出事实、
agent-core 做裁判**。顺带它也推广了：将来跑在真机器人上的用例，只要吐出同样形状的
事件流，这同一套断言直接可用。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_case.py -q
"""

import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'case-test.db'))

import benchmark_case as bc  # noqa: E402


def case(**evaluate):
    return {'formatVersion': 1, 'canvas': {}, 'test': {
        'requires': {'drivers': ['simulator-generic'], 'assets': ['bj-2f']},
        'run': {'prompt': '带我转一下展区并给我介绍下',
                'injections': [{'after_arrival': 'P5', 'delay': 6.0, 'text': '先等一下'}]},
        'evaluate': {'expect': {'waypoint_order': ['P3', 'P4']}, 'weights': {}, **evaluate},
    }}


def arrive(t, label):
    return {'t': t, 'event': 'arrive', 'label': label}


def nav_start(t, label=''):
    return {'t': t, 'event': 'nav_start', 'label': label}


# ── 结构 ──────────────────────────────────────────────────────────────────────

def test_a_solution_without_a_test_block_is_not_a_case():
    assert bc.is_case({'formatVersion': 1, 'canvas': {}}) is False
    assert bc.validate({'formatVersion': 1})[0].startswith('这个解决方案没有 test 段')


def test_a_well_formed_case_validates():
    assert bc.is_case(case()) is True
    assert bc.validate(case()) == []


def test_an_empty_prompt_is_refused():
    """没有初始指令，机器人不会动，最后会记一个 0 分 —— 那是基准测试在撒谎。"""
    broken = case()
    broken['test']['run']['prompt'] = '   '

    assert any('prompt 为空' in p for p in bc.validate(broken))


def test_an_injection_that_can_never_fire_is_refused():
    broken = case()
    broken['test']['run']['injections'] = [{'text': '喂'}]

    assert any('永远不会触发' in p for p in bc.validate(broken))


def test_a_case_with_no_assertions_is_refused():
    """没有断言的用例只会产生一个无意义的满分。"""
    broken = case()
    broken['test']['evaluate']['expect'] = {}

    assert any('没有断言' in p for p in bc.validate(broken))


def test_unknown_weight_dimensions_are_refused():
    broken = case()
    broken['test']['evaluate']['weights'] = {'orchestration': 30, 'vibes': 70}

    assert any('未知维度' in p and 'vibes' in p for p in bc.validate(broken))


def test_requires_is_what_preflight_reads():
    """用例声明依赖，preflight 才能分层报错 —— 这正是卡片做不到的那件事：
    驱动没装时卡片本身不存在，连「你缺这个驱动」都没地方说。"""
    needs = bc.requires(case())

    assert needs['drivers'] == ['simulator-generic']
    assert needs['assets'] == ['bj-2f']


# ── 裁判 ──────────────────────────────────────────────────────────────────────

def test_waypoint_order_compares_arrivals_in_order():
    good = bc.check_waypoint_order({'waypoint_order': ['P3', 'P4']},
                                   [arrive(1, 'P3'), arrive(2, 'P4')])
    bad = bc.check_waypoint_order({'waypoint_order': ['P3', 'P4']},
                                  [arrive(1, 'P4'), arrive(2, 'P3')])

    assert good['ok'] is True
    assert bad['ok'] is False and 'P3' in bad['detail']


def test_announcing_before_arriving_fails():
    events = [nav_start(0, 'P3'), arrive(5, 'P3'), nav_start(6, 'P4'), arrive(9, 'P4'),
              {'t': 9.5, 'event': 'speak_start'}, {'t': 11, 'event': 'speak_end',
                                                   'status': 'completed'}]

    result = bc.check_announce_after_arrive({'announce_after_arrive': True}, events)

    assert result['ok'] is False
    assert 'P3' in result['detail'] and '没讲' in result['detail']


def test_leaving_before_the_announcement_finishes_fails():
    events = [nav_start(0, 'P3'), arrive(5, 'P3'), {'t': 5.2, 'event': 'speak_start'},
              nav_start(6, 'P4'), {'t': 9, 'event': 'speak_end', 'status': 'completed'}]

    result = bc.check_announce_after_arrive({'announce_after_arrive': True}, events)

    assert result['ok'] is False and '没讲完就走' in result['detail']


def test_an_abandoned_leg_reported_completed_is_caught():
    """基准测试要抓的谎：被放弃的一段报 completed，等于告诉模型它到过一个没去过的地方。"""
    posts = [{'action_id': 'a1', 'status': 'completed',
              'result': {'label': 'P6', 'progress': {'fraction': 0.4}}}]

    result = bc.check_interrupted_leg(
        {'interrupted_leg': {'target': 'P6', 'acp_status': 'cancelled'}}, [], acp_posts=posts)

    assert result['ok'] is False and 'completed' in result['detail']


def test_a_full_progress_cancel_is_not_an_interruption():
    posts = [{'action_id': 'a1', 'status': 'cancelled',
              'result': {'label': 'P6', 'progress': {'fraction': 1.0}}}]

    result = bc.check_interrupted_leg(
        {'interrupted_leg': {'target': 'P6', 'min_progress': 0.02}}, [], acp_posts=posts)

    assert result['ok'] is False


def test_resuming_at_the_wrong_waypoint_is_caught():
    """长程任务里模型最常犯的错，任何单步断言都看不见。"""
    events = [{'t': 5, 'event': 'nav_cancelled', 'label': 'P6'}, nav_start(8, 'P7')]

    result = bc.check_resume_correctness({'resume_target': 'P6'}, events)

    assert result['ok'] is False and 'P7' in result['detail']


def test_duplicate_terminal_posts_are_caught():
    """重复上报在 agent-core 侧是静默的，除了这里没有地方会发现。"""
    posts = [{'action_id': 'a1'}, {'action_id': 'a1'}, {'action_id': 'a2'}]

    assert bc.check_exactly_one_terminal_post({}, [], acp_posts=posts)['ok'] is False


def test_checks_that_were_not_asserted_say_so():
    for check in (bc.check_waypoint_order, bc.check_interrupted_leg,
                  bc.check_resume_correctness, bc.check_max_wall_seconds):
        assert check({}, [], acp_posts=[])['detail'] == '未断言'


# ── 评分 ──────────────────────────────────────────────────────────────────────

def test_weights_come_from_the_case_not_from_code():
    payload = case()
    results = bc.evaluate(payload, [arrive(1, 'P3')], acp_posts=[])
    payload['test']['evaluate']['weights'] = {'orchestration': 1}
    only_orchestration = bc.score(payload, results)['total']
    payload['test']['evaluate']['weights'] = {'safety': 1}
    only_safety = bc.score(payload, results)['total']

    assert only_orchestration != only_safety


def test_an_unmeasured_dimension_scores_none_not_zero():
    """「没测」和「测了没过」是两件事；平均在一起，基准测试就开始撒谎。"""
    payload = case()
    result = bc.score(payload, bc.evaluate(payload, [], acp_posts=[]))

    assert result['by_dimension']['long_horizon'] is None
    assert result['by_dimension']['latency'] is None


def test_a_perfect_run_scores_one_hundred():
    payload = case()
    payload['test']['evaluate']['expect'] = {'waypoint_order': ['P3'], 'never_occupied': True}
    events = [nav_start(0, 'P3'), arrive(3, 'P3')]

    result = bc.score(payload, bc.evaluate(payload, events, acp_posts=[]))

    assert result['total'] == 100.0
    assert result['failures'] == []


def test_evaluate_reads_expect_out_of_the_case_payload():
    payload = case()
    payload['test']['evaluate']['expect'] = {'waypoint_order': ['P3', 'P4']}

    names = [r['name'] for r in bc.evaluate(payload, [arrive(1, 'P3')], acp_posts=[])]

    assert 'waypoint_order' in names
    assert len(names) == len(bc.CHECKS)
