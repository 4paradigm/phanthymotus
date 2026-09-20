"""用例的结构，以及分数怎么来。

判定本身不在这里了 —— 确定性那一半在 `test_benchmark_metrics.py`（指标算得对不对），
模糊那一半在 `test_benchmark_judge.py`（裁判的输入输出）。这个文件管的是中间那层：
用例声明了什么、默认目标怎么叠、一堆判决怎么变成分数。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_case.py -q
"""

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'case-test.db'))

import benchmark_case as bc  # noqa: E402
import benchmark_metrics as bm  # noqa: E402


def case(**over):
    block = {
        'requires': {'drivers': ['simulator-generic'], 'assets': []},
        'run': {'prompt': '带我转一下展区', 'injections': []},
        'requirements': [{'text': '到了再讲', 'weight': 25, 'dimension': 'world_timing'}],
    }
    block.update(over)
    return {'formatVersion': 1, 'canvas': {'cards': []}, 'devices': [], 'test': block}


def item(dimension, ok, weight=10, measurable=True, text='x'):
    return {'id': text, 'text': text, 'dimension': dimension, 'ok': ok,
            'weight': weight, 'measurable': measurable}


# ── 七条原则 ──────────────────────────────────────────────────────────────────

def test_no_dimension_key_collides_with_the_old_set():
    """旧的是 orchestration / interruption / long_horizon / **latency** / **safety**。

    新的里面也有「延时」和「安全」，但含义不同 —— 旧 `latency` 是「整趟跑完没超预算」，
    新 `llm_latency` 是「每轮推理多快」。沿用同一个键，趋势图会把两种度量悄悄画进同一
    条线。同名不同义在这个仓库已经坑过三次，这条是第五次的预防。
    """
    old = {'orchestration', 'interruption', 'long_horizon', 'latency', 'safety'}

    assert not (set(bc.DIMENSIONS) & old)
    assert set(bc.DIMENSION_LABELS) == set(bc.DIMENSIONS)


def test_total_task_duration_has_no_default_target():
    """三站的导览和十站的导览没有可比的标准。给一个拍脑袋的默认值，等于让每个长用例都
    无故扣分。它永远报出来、永远可看趋势，只在用例自己给了数时才参与判定。"""
    assert 'concurrency.total_seconds' not in bc.DEFAULT_TARGETS

    with_target = bc.targets(case(targets={'concurrency.total_seconds': 300}))

    assert with_target['concurrency.total_seconds']['max'] == 300


def test_speaking_on_the_way_is_not_penalised_by_default():
    """**两个维度的默认目标不能互相打架。**

    Orin6 上一跑就现原形：机器人在路上说了一句「正在带您前往一号展区，请跟我走」。
    同一次跑动里 `ux` 因为它少了一段空白，而 `world_timing` 曾因为它扣分 ——
    奖一次罚一次，等于没有意见。

    这个指标分不清「到达前宣布已到达」（该罚）和「路上说点什么」（该奖），而区分
    它们是内容问题，是裁判的活。所以它是可看的指标，不是默认目标。
    """
    assert 'world_timing.spoke_before_arrival' not in bc.DEFAULT_TARGETS

    seen_it = seen(**{'world_timing.spoke_before_arrival': 3})
    items = bc.check_targets(seen_it, bc.targets(case()))

    assert not any(i['id'] == 'world_timing.spoke_before_arrival' for i in items)


def test_leaving_mid_sentence_is_still_penalised():
    """「讲完再走」没有这种两义性：话说一半就走，任何维度都不想要。"""
    items = bc.check_targets(seen(**{'world_timing.left_while_speaking': 1}),
                             bc.targets(case()))

    left = next(i for i in items if i['id'] == 'world_timing.left_while_speaking')
    assert left['ok'] is False


def test_a_case_can_move_a_default_target_without_replacing_the_rest():
    spec = bc.targets(case(targets={'ux.blank_avg_s': 15}))

    assert spec['ux.blank_avg_s']['max'] == 15
    assert spec['llm_latency.median_s']['max'] == 10       # 其余不受影响


def test_overriding_a_lower_bound_target_keeps_it_a_lower_bound():
    """cache 命中是越高越好。覆盖它的时候当成上限，等于把「至少三成」改成「至多三成」。"""
    spec = bc.targets(case(targets={'cache_hit.ratio': 0.5}))

    assert spec['cache_hit.ratio'] == {'min': 0.5, 'label': 'cache 命中不低于三成'}


# ── 结构 ──────────────────────────────────────────────────────────────────────

def test_a_solution_without_a_test_block_is_not_a_case():
    assert bc.is_case({'formatVersion': 1, 'canvas': {}}) is False


def test_a_well_formed_case_validates():
    assert bc.validate(case()) == []


def test_an_empty_prompt_is_refused():
    """空指令跑出来的 0 分是基准测试在撒谎。"""
    payload = case(run={'prompt': '   ', 'injections': []})

    assert any('prompt 为空' in p for p in bc.validate(payload))


def test_an_injection_that_can_never_fire_is_refused():
    payload = case(run={'prompt': '走', 'injections': [{'text': '等一下'}]})

    assert any('永远不会触发' in p for p in bc.validate(payload))


def test_a_requirement_hung_on_an_unknown_principle_is_refused():
    payload = case(requirements=[{'text': '随便', 'dimension': '编排'}])

    assert any('未知原则' in p for p in bc.validate(payload))


def test_a_new_case_is_unrunnable_and_carries_no_canvas():
    """新用例故意是不合法的（没有初始指令），也**故意不带画布**。

    跑用例跑的是当前画布；自带画布只是可选的参考。新用例带一张空画布，「跑」就会把它
    刷进去 —— 那正是这一轮要修的那个 bug。
    """
    blank = {'test': bc.blank()}

    assert any('prompt 为空' in p for p in bc.validate(blank))
    assert 'canvas' not in bc.blank()


def test_requires_is_what_preflight_reads():
    assert bc.requires(case())['drivers'] == ['simulator-generic']


# ── 参考流程 ──────────────────────────────────────────────────────────────────

def test_a_procedure_becomes_one_requirement_under_world_timing():
    """一等的输入框，普通的计分方式。

    它不是新维度 —— 维度是固定的质量轴，而一条流程是这个用例的内容；「有没有按流程走」
    本来就是 `world_timing` 在问的事。
    """
    payload = case(procedure='1. 打开地图\n2. 导航', requirements=[])

    reqs = bc.requirements(payload)

    assert len(reqs) == 1
    assert reqs[0]['dimension'] == 'world_timing'
    assert reqs[0]['weight'] == bc.PROCEDURE_WEIGHT
    assert '打开地图' in reqs[0]['procedure']


def test_a_procedures_weight_and_home_can_both_be_changed():
    payload = case(procedure='1. 打开地图',
                   requirements=[{'id': bc.PROCEDURE_ID, 'weight': 60,
                                  'dimension': 'answer_quality'}])

    reqs = bc.requirements(payload)

    assert len(reqs) == 1 and reqs[0]['weight'] == 60
    assert reqs[0]['dimension'] == 'answer_quality'


def test_no_procedure_means_no_extra_requirement():
    assert len(bc.requirements(case(procedure='   '))) == 1   # 只有用例自己写的那条


def test_a_requirement_with_no_principle_lands_in_answer_quality():
    """没指定就归到「回答效果」—— 那是唯一没有可算指标、本来就靠裁判的一条。"""
    payload = case(requirements=[{'text': '讲得要有意思'}])

    assert bc.requirements(payload)[0]['dimension'] == 'answer_quality'


# ── 目标：确定性判定 ──────────────────────────────────────────────────────────

def seen(**over):
    base = bm.observations({'events': [], 'acp_posts': [], 'source': 'agent-core'},
                           [], {}, (0, 100))
    for path, value in over.items():
        group, key = path.split('.', 1)
        base[group][key] = value
    return base


def test_a_metric_inside_its_target_passes_without_any_llm():
    items = bc.check_targets(seen(**{'llm_latency.median_s': 4.0}),
                             bc.targets(case()))

    latency = next(i for i in items if i['id'] == 'llm_latency.median_s')
    assert latency['ok'] is True and latency['kind'] == 'target'


def test_a_metric_outside_its_target_fails_and_says_both_numbers():
    items = bc.check_targets(seen(**{'llm_latency.median_s': 23.4}),
                             bc.targets(case()))

    latency = next(i for i in items if i['id'] == 'llm_latency.median_s')
    assert latency['ok'] is False
    assert '23.4' in latency['detail'] and '10' in latency['detail']


def test_an_unmeasurable_metric_is_not_a_failure_and_carries_its_reason():
    items = bc.check_targets(seen(), bc.targets(case()))

    safety = next(i for i in items if i['id'] == 'physical_safety.incidents')
    assert safety['measurable'] is False
    assert '轨迹占用数据' in safety['detail']


def test_a_lower_bound_target_is_compared_the_other_way_round():
    items = bc.check_targets(seen(**{'cache_hit.ratio': 0.31}), bc.targets(case()))

    cache = next(i for i in items if i['id'] == 'cache_hit.ratio')
    assert cache['ok'] is True and '≥' in cache['detail']


# ── 分数 ──────────────────────────────────────────────────────────────────────

def test_a_dimension_scores_by_the_weight_of_what_passed():
    items = [item('world_timing', True, weight=30), item('world_timing', False, weight=10)]

    result = bc.score(case(), items)

    assert result['by_dimension']['world_timing'] == 75.0


def test_an_unmeasured_dimension_scores_none_not_zero():
    """「没测到」和「测了没过」是两件事。平均在一起，基准测试就开始撒谎了。"""
    items = [item('physical_safety', False, measurable=False)]

    result = bc.score(case(), items)

    assert result['by_dimension']['physical_safety'] is None
    assert result['failures'] == [] and result['unmeasured'] == ['x']


def test_dimension_weights_come_from_the_case_not_from_code():
    """权重是会被争论、会被调整的东西，改权重不该是改代码。"""
    payload = case(dimension_weights={'world_timing': 9, 'ux': 1})
    items = [item('world_timing', True), item('ux', False)]

    assert bc.score(payload, items)['total'] == 90.0


def test_an_unmeasured_dimension_drops_out_of_the_denominator():
    payload = case(dimension_weights={'world_timing': 1, 'physical_safety': 99})
    items = [item('world_timing', True), item('physical_safety', False, measurable=False)]

    assert bc.score(payload, items)['total'] == 100.0


def test_a_case_with_nothing_measurable_scores_none_rather_than_zero():
    result = bc.score(case(), [item('ux', False, measurable=False)])

    assert result['total'] is None


# ── 旧格式 ────────────────────────────────────────────────────────────────────

def legacy():
    return {'formatVersion': 1, 'canvas': {'cards': []}, 'devices': [], 'test': {
        'requires': {'drivers': ['simulator-generic'], 'assets': []},
        'run': {'prompt': '带我转一下展厅'},
        'evaluate': {
            'expect': {'waypoint_order': ['入口', '一号展区'],
                       'announce_after_arrive': True,
                       'resume_target': '二号展区',
                       'interrupted_leg': {'target': '二号展区'},
                       'max_wall_seconds': 600},
            'weights': {'orchestration': 30}}}}


def test_an_old_case_reads_back_as_prose_requirements():
    """不转的话，已经存在的用例打开就是空的 —— 用户看到的是「我的要求没了」，
    而不是「格式换了」。"""
    reqs = bc.requirements(bc.migrate(legacy()))
    texts = [r['text'] for r in reqs]

    assert any('依次到达每一站' in t and '入口' in t for t in texts)
    assert any('到了再讲' in t or '讲完再走' in t for t in texts)
    assert all(r['dimension'] in bc.DIMENSIONS for r in reqs)


def test_migrating_drops_the_old_evaluate_block():
    migrated = bc.migrate(legacy())

    assert 'evaluate' not in migrated['test']
    assert bc.validate(migrated) == []


def test_the_old_wall_clock_budget_becomes_the_run_budget():
    assert bc.migrate(legacy())['test']['run']['budget_seconds'] == 600


def test_never_occupied_does_not_become_a_requirement():
    """它在新结构里是**目标**，默认目标已经覆盖了。再转一遍会变成同一件事判两次。"""
    payload = legacy()
    payload['test']['evaluate']['expect']['never_occupied'] = True

    texts = [r['text'] for r in bc.requirements(bc.migrate(payload))]

    assert not any('占用' in t for t in texts)


def test_a_new_format_case_passes_through_migrate_untouched():
    assert bc.migrate(case()) == case()
