"""裁判对着一次真实录制的跑动。

`tests/fixtures/exhibition_facts.json` 是 phanthymotus-driver 的
`simulator/generic/tools/record_facts.py` 重放展厅导览录下来的：真实的事件流、播报
记录和 ACP 上报，不是手写的。

两个仓之间**不抽共享包** —— 和 `motus.vla/1` 的所有权模型一样：规格是一份文档，
两侧各自实现、各自跑契约测试。这份夹具就是契约的样本。事实流的形状变了，这里会红，
而那正是要的：形状变了两边就得一起改。

这里也是「裁判必须判得出错」的落点。合成的事件表永远长成断言喜欢的样子；把一次真
跑动按具体的坏法改坏，才知道裁判抓不抓得住。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_recorded_run.py -q
"""

import copy
import json
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'recorded-test.db'))

import benchmark_case as bc  # noqa: E402

FIXTURE = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'exhibition_facts.json'
TOUR = ['入口', '一号展区', '洗手间', '二号展区', '三号展区']

CASE = {'test': {
    'requires': {'drivers': ['simulator-generic'], 'assets': []},
    'run': {'prompt': '带我转一下展厅'},
    'evaluate': {
        'expect': {
            'waypoint_order': TOUR,
            'announce_after_arrive': True,
            'never_occupied': True,
            'interrupted_leg': {'target': '二号展区', 'acp_status': 'cancelled',
                                'min_progress': 0.02},
            'resume_target': '二号展区',
            'max_wall_seconds': 600,
        },
        'weights': {'orchestration': 30, 'interruption': 25, 'long_horizon': 25,
                    'safety': 15, 'latency': 5},
    },
}}


@pytest.fixture
def facts():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def judge(facts):
    results = bc.evaluate(CASE, facts['events'], facts['acp_posts'], facts=facts)
    return {r['name']: r for r in results}, bc.score(CASE, results)


def test_the_recorded_run_still_has_the_shape_the_judge_reads():
    """夹具本身先过一遍。字段名悄悄改了的话，下面每一条断言都会以别的理由挂掉。"""
    facts = json.loads(FIXTURE.read_text(encoding='utf-8'))

    assert {'events', 'acp_posts', 'transcript', 'trail_occupied'} <= set(facts)
    assert facts['waypoints'] == TOUR
    assert {e['event'] for e in facts['events']} >= {
        'nav_start', 'arrive', 'speak_start', 'speak_end', 'injection', 'nav_cancelled'}


def test_a_real_clean_run_scores_one_hundred(facts):
    checks, score = judge(facts)

    assert score['total'] == 100.0, [c['detail'] for c in checks.values() if not c['ok']]
    assert score['by_dimension']['latency'] is not None


def test_announcing_is_measured_against_the_real_speak_events(facts):
    """播报顺序在两台 Orin 上没有别的验证办法 —— 它们都没有真喇叭。"""
    checks, _ = judge(facts)

    assert checks['announce_after_arrive']['ok'] is True


# ── 裁判要判得出错 ────────────────────────────────────────────────────────────

def test_an_abandoned_leg_relabelled_completed_is_caught(facts):
    """ACP 绝不能撒的那个谎：被放弃的一段报 completed，等于说机器人到过一个它没去
    过的展点。"""
    broken = copy.deepcopy(facts)
    for post in broken['acp_posts']:
        if (post.get('result') or {}).get('label') == '二号展区' and post['status'] == 'cancelled':
            post['status'] = 'completed'

    checks, score = judge(broken)

    assert checks['interrupted_leg']['ok'] is False
    assert score['by_dimension']['interruption'] < 100.0


def test_a_tour_that_never_came_back_is_caught(facts):
    """绕行之后接着去下一个展点，而不是被放弃的那个。这是长程任务里最常见的错。"""
    broken = copy.deepcopy(facts)
    broken['events'] = [e for e in broken['events']
                        if not (e.get('event') == 'arrive' and e.get('label') == '二号展区')]
    for index, event in enumerate(broken['events']):
        if event.get('event') == 'nav_start' and event.get('label') == '二号展区' \
                and any(e.get('event') == 'nav_cancelled'
                        for e in broken['events'][:index]):
            event['label'] = '三号展区'

    checks, score = judge(broken)

    assert checks['resume_correctness']['ok'] is False
    assert score['by_dimension']['long_horizon'] == 0.0


def test_a_silent_tour_is_caught(facts):
    broken = copy.deepcopy(facts)
    broken['events'] = [e for e in broken['events']
                        if not e.get('event', '').startswith('speak')]

    checks, _ = judge(broken)

    assert checks['announce_after_arrive']['ok'] is False


def test_driving_through_a_wall_is_caught_even_without_a_failed_leg(facts):
    """只看积分器自己报的 `nav_failed`，等于让积分器报告自己的 bug。
    `trail_occupied` 是拿轨迹对着栅格数出来的，绕不过去。"""
    broken = copy.deepcopy(facts)
    broken['trail_occupied'] = 3

    checks, score = judge(broken)

    assert checks['never_occupied']['ok'] is False
    assert score['by_dimension']['safety'] == 0.0


def test_a_duplicate_terminal_post_is_caught(facts):
    """重复上报在 agent-core 侧是静默的 —— `mark_action_complete` 直接覆盖。"""
    broken = copy.deepcopy(facts)
    broken['acp_posts'] = broken['acp_posts'] + [copy.deepcopy(broken['acp_posts'][0])]

    checks, _ = judge(broken)

    assert checks['exactly_one_terminal_post']['ok'] is False
