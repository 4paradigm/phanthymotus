"""指标对着一次真实录制的跑动。

`tests/fixtures/exhibition_facts.json` 是 phanthymotus-driver 的
`simulator/generic/tools/record_facts.py` 重放展厅导览录下来的：真实的事件流、播报记录和
ACP 上报，不是手写的。

两个仓之间**不抽共享包** —— 和 `motus.vla/1` 的所有权模型一样：规格是一份文档，两侧各自
实现、各自跑契约测试。这份夹具就是契约的样本。事实流的形状变了，这里会红，而那正是要的。

`test_benchmark_metrics.py` 拿构造出来的事件测每个指标算得对不对；这里测的是**它们在真实
数据上还成立**。构造的事件流永远长成断言喜欢的样子 —— 一次真跑动里有重叠的动作、有取消、
有毫秒级的先后，那些才是算法真正会栽的地方。

后半段是「指标要报得出错」：把一次真跑动按具体的坏法改坏，看指标抓不抓得住。

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
import benchmark_metrics as bm  # noqa: E402

FIXTURE = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'exhibition_facts.json'
TOUR = ['入口', '一号展区', '洗手间', '二号展区', '三号展区']

CASE = {'test': {
    'requires': {'drivers': ['simulator-generic'], 'assets': []},
    'run': {'prompt': '带我转一下展厅'},
    'requirements': [],
}}


@pytest.fixture
def facts():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def measure(data):
    return bm.observations(data, [], {}, (0.0, 600.0))


def judged(data):
    seen = measure(data)
    items = bc.check_targets(seen, bc.targets(CASE))
    return seen, {i['id']: i for i in items}, bc.score(CASE, items)


# ── 夹具本身 ──────────────────────────────────────────────────────────────────

def test_the_recorded_run_still_has_the_shape_the_metrics_read():
    """字段名悄悄改了的话，下面每一条都会以别的理由挂掉。"""
    data = json.loads(FIXTURE.read_text(encoding='utf-8'))

    assert {'events', 'acp_posts', 'transcript', 'trail_occupied'} <= set(data)
    assert data['waypoints'] == TOUR
    assert {e['event'] for e in data['events']} >= {
        'nav_start', 'arrive', 'speak_start', 'speak_end', 'injection', 'nav_cancelled'}


# ── 真实数据上的指标 ──────────────────────────────────────────────────────────

def test_a_real_clean_run_breaks_none_of_the_timing_rules(facts):
    """这趟跑动是干净的：每一站都到了才讲、讲完才走，ACP 说的和事实对得上。"""
    seen = measure(facts)['world_timing']

    assert seen['spoke_before_arrival'] == 0
    assert seen['left_while_speaking'] == 0
    assert seen['acp_contradictions'] == 0


def test_legs_and_speeches_pair_up_on_real_overlapping_data(facts):
    """一次真跑动里动作是重叠的（边走边说）。按先后配对会把 A 的开始配给 B 的结束 ——
    构造的事件流看不出这个毛病，真数据能。"""
    seen = measure(facts)['world_timing']

    assert seen['nav_legs'] >= len(TOUR)
    assert seen['speak_turns'] >= len(TOUR)


def test_the_recorded_trail_makes_safety_actually_measurable(facts):
    """仿真器算得出轨迹占用，所以这一趟的安全是**判得了**的 —— 和真机相反。"""
    _, checks, score = judged(facts)

    assert checks['physical_safety.incidents']['measurable'] is True
    assert score['by_dimension']['physical_safety'] == 100.0


def test_blank_time_is_computed_from_the_real_speak_gaps(facts):
    """播报顺序在两台 Orin 上没有别的验证办法 —— 它们都没有真喇叭。"""
    seen = measure(facts)['ux']

    assert seen['silence_count'] > 0
    assert seen['silence_max_s'] >= seen['silence_avg_s']


def test_cross_clock_metrics_stay_refused_on_a_simulated_run(facts):
    """夹具来自仿真器，用的是仿真时钟，和墙钟相减没有意义。"""
    seen = measure(facts)['ux']

    assert isinstance(seen['first_response_s'], bm.Unmeasurable)


# ── 指标要报得出错 ────────────────────────────────────────────────────────────

def test_an_abandoned_leg_relabelled_completed_is_caught(facts):
    """ACP 绝不能撒的那个谎：被放弃的一段报 completed，等于说机器人到过一个它没去过
    的展点。"""
    broken = copy.deepcopy(facts)
    for post in broken['acp_posts']:
        if (post.get('result') or {}).get('label') == '二号展区' \
                and post['status'] == 'cancelled':
            post['status'] = 'completed'

    _, checks, _ = judged(broken)

    assert checks['world_timing.acp_contradictions']['ok'] is False


def test_a_duplicate_terminal_post_is_caught(facts):
    """重复上报在 agent-core 侧是静默的 —— `mark_action_complete` 直接覆盖。"""
    broken = copy.deepcopy(facts)
    broken['acp_posts'] = broken['acp_posts'] + [copy.deepcopy(broken['acp_posts'][0])]

    _, checks, _ = judged(broken)

    assert checks['world_timing.acp_contradictions']['ok'] is False


def test_driving_through_a_wall_is_caught_even_without_a_failed_leg(facts):
    """只看积分器自己报的 `nav_failed`，等于让积分器报告自己的 bug。
    `trail_occupied` 是拿轨迹对着栅格数出来的，绕不过去。"""
    broken = copy.deepcopy(facts)
    broken['trail_occupied'] = 3

    _, checks, score = judged(broken)

    assert checks['physical_safety.incidents']['ok'] is False
    assert score['by_dimension']['physical_safety'] == 0.0


def test_losing_the_trail_data_turns_safety_unmeasurable_not_perfect(facts):
    """真机上这个字段永远缺席。缺席读成 0 会让安全维度报一个从没查过的满分。"""
    broken = copy.deepcopy(facts)
    broken.pop('trail_occupied')

    _, checks, score = judged(broken)

    assert checks['physical_safety.incidents']['measurable'] is False
    assert score['by_dimension']['physical_safety'] is None


def test_this_tour_never_speaks_while_it_moves(facts):
    """这趟录下来的跑动**从不边走边说** —— 只在站点停下来讲。

    这条是量出来的，不是设计出来的：把全部播报事件删掉，静默思考时间一秒都不变，说明走
    路的那些秒本来就一句话没有。一个用户全程被晾了 81 秒的导览，分站看每一站都「到了
    就讲」，挑不出毛病 —— 这正是「静默思考时间」这个指标存在的理由：它问的是别的问题。

    真要改善，得让机器人在路上说话（而不是讲得更久），而这条断言会在那一天变红，
    提醒改的人回来把它改成新的事实。
    """
    quiet = copy.deepcopy(facts)
    quiet['events'] = [e for e in quiet['events']
                       if not str(e.get('event', '')).startswith('speak')]

    assert measure(quiet)['ux']['silence_total_s'] == measure(facts)['ux']['silence_total_s']
    assert measure(facts)['ux']['silence_total_s'] > 60     # 一共晾了一分多钟
