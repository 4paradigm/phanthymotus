"""本机用例库。

面板的主体是**看现有的方案**，不是打开一个文件 —— 所以用例得有个能列、能改、能留
下来的地方。存的是整个解决方案包体，不是只有 `test` 段：用例是解决方案 + 执行方案 +
评估方案，只存后两段的话，载入时拿不出画布，「用例自带画布」就不成立了。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_case_library.py -q
"""

import asyncio
import os
import pathlib
import sys
import tempfile

import fastapi
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'library-test.db'))

import benchmark_case  # noqa: E402
import benchmark_store  # noqa: E402
from api import benchmark  # noqa: E402

TEST_BLOCK = {
    'name': '北京2F展厅 · 完整导览',
    'requires': {'drivers': ['simulator-generic'], 'assets': ['bj-2f']},
    'run': {'prompt': '带我转一下展区并给我介绍下',
            'world': {'map': 'bj-2f'},
            'injections': [{'after_arrival': 'P5', 'delay': 6.0, 'text': '先等一下'}]},
    'evaluate': {'expect': {'waypoint_order': ['P3', 'P4']},
                 'weights': {'orchestration': 100}},
}


def payload(test=TEST_BLOCK, cards=1):
    return {'formatVersion': 1, 'devices': [],
            'canvas': {'cards': [{'id': f'c{i}'} for i in range(cards)]},
            'test': test}


@pytest.fixture(autouse=True)
def clean():
    for record in benchmark_store.list_cases():
        benchmark_store.delete_case(record['id'])
    for run in benchmark_store.list_runs(limit=500):
        benchmark_store.delete_run(run['id'])
    yield
    for record in benchmark_store.list_cases():
        benchmark_store.delete_case(record['id'])


# ── 存储 ──────────────────────────────────────────────────────────────────────

def test_a_case_survives_a_round_trip_whole():
    """存整个包体，不是只存 test 段 —— 画布那一半丢了，用例就载入不了了。"""
    case_id = benchmark_store.save_case(payload(cards=3), name='导览', origin='file')

    record = benchmark_store.get_case(case_id)

    assert record['name'] == '导览' and record['origin'] == 'file'
    assert len(record['payload']['canvas']['cards']) == 3
    assert record['payload']['test']['run']['prompt'] == '带我转一下展区并给我介绍下'


def test_saving_with_the_same_id_replaces_rather_than_duplicates():
    case_id = benchmark_store.save_case(payload(), name='一版')
    edited = payload({**TEST_BLOCK, 'run': {**TEST_BLOCK['run'], 'prompt': '改过了'}})

    benchmark_store.save_case(edited, name='二版', case_id=case_id)

    assert len(benchmark_store.list_cases()) == 1
    assert benchmark_store.get_case(case_id)['payload']['test']['run']['prompt'] == '改过了'


def test_the_library_lists_newest_first():
    benchmark_store.save_case(payload(), name='旧的')
    benchmark_store.save_case(payload(), name='新的')

    assert [r['name'] for r in benchmark_store.list_cases()][0] == '新的'


def test_deleting_a_case_leaves_the_scores_it_already_produced():
    """跑过的分数是已经发生的事实。删掉用例不会让它没发生过，而历史那一行还带着
    当时的模型与镜像 tag —— 那正是历史存在的理由。"""
    case_id = benchmark_store.save_case(payload(), name='导览')
    run_id = benchmark_store.create_run('导览', n_repeats=2, llm_model='claude-opus-5')
    benchmark_store.finish_run(run_id, score_total=87.5)

    benchmark_store.delete_case(case_id)

    stored = benchmark_store.get_run(run_id)
    assert stored['score_total'] == 87.5 and stored['llm_model'] == 'claude-opus-5'


def test_the_result_table_and_the_library_are_different_tables():
    """`benchmark_case` 存的是「跑出了什么」，`case_library` 存的是「要跑什么」。
    同名不同义在这一轮已经坑过三次，这条守着第四次。"""
    run_id = benchmark_store.create_run('导览')
    benchmark_store.add_case(run_id, scenario='导览', repeat_idx=0, score=90.0)
    benchmark_store.save_case(payload(), name='导览')

    assert len(benchmark_store.get_run(run_id)['cases']) == 1
    assert len(benchmark_store.list_cases()) == 1


# ── 端点 ──────────────────────────────────────────────────────────────────────

def test_the_list_carries_what_a_card_needs_without_unpacking_the_payload():
    benchmark_store.save_case(payload(), name='导览')

    card = asyncio.run(benchmark.list_cases())['cases'][0]

    assert card['prompt'] == '带我转一下展区并给我介绍下'
    assert card['injections'] == 1 and card['waypoints'] == 2
    assert card['problems'] == []


def test_a_new_case_is_created_unrunnable_and_says_why():
    """新建的用例故意是不合法的 —— 什么都还没写。一建出来就「合法」，等于说它已经
    能跑了，而空指令跑出来的 0 分是基准测试在撒谎。"""
    created = asyncio.run(benchmark.create_case(benchmark.CaseWrite()))

    problems = benchmark_case.validate(created['payload'])

    assert any('prompt 为空' in p for p in problems)


def test_saving_a_half_finished_edit_is_allowed_but_reported():
    """编辑分几次做完，存一半不该被拒 —— 但理由要当场说，不是等点「跑」才说。"""
    case_id = benchmark_store.save_case(payload(), name='导览')
    half = payload({**TEST_BLOCK, 'run': {**TEST_BLOCK['run'], 'prompt': ''}})

    result = asyncio.run(benchmark.update_case(
        case_id, benchmark.CaseWrite(payload=half, name='导览')))

    assert any('prompt 为空' in p for p in result['problems'])
    assert benchmark_store.get_case(case_id)['payload']['test']['run']['prompt'] == ''


def test_a_solution_without_a_test_section_is_refused_as_a_case():
    plain = {'formatVersion': 1, 'canvas': {'cards': []}, 'devices': []}

    with pytest.raises(fastapi.HTTPException) as caught:
        asyncio.run(benchmark.create_case(benchmark.CaseWrite(payload=plain)))

    assert caught.value.status_code == 422


def test_editing_a_case_that_does_not_exist_is_a_404():
    with pytest.raises(fastapi.HTTPException) as caught:
        asyncio.run(benchmark.update_case('nope', benchmark.CaseWrite(payload=payload())))

    assert caught.value.status_code == 404


def test_the_market_column_only_lists_solutions_that_are_cases(monkeypatch):
    """市场列表不返回包体（几十 KB），但 `includes` 它返回 —— 正好够认出用例。"""
    async def fake_market(search='', industry='all', limit=30):
        return {'code': 200, 'data': [
            {'slug': 'tour-case', 'name': '导览用例', 'includes': ['canvas', 'test']},
            {'slug': 'plain', 'name': '普通方案', 'includes': ['canvas', 'skills']},
        ]}

    from api import solutions
    monkeypatch.setattr(solutions, 'market', fake_market)

    result = asyncio.run(benchmark.market_cases())

    assert [c['slug'] for c in result['cases']] == ['tour-case']


def test_an_unreachable_market_is_an_empty_column_with_a_reason(monkeypatch):
    async def fake_market(search='', industry='all', limit=30):
        return {'code': 502, 'error': '连不上'}

    from api import solutions
    monkeypatch.setattr(solutions, 'market', fake_market)

    result = asyncio.run(benchmark.market_cases())

    assert result['cases'] == [] and result['error']


# ── summary / blank ───────────────────────────────────────────────────────────

def test_summary_is_where_the_payload_shape_is_known():
    """列表不该自己去解 `test.run.injections` 这种路径 —— 散到前端之后，改一次结构
    要追着它跑。"""
    view = benchmark_case.summary(payload(cards=4))

    assert view['name'] == '北京2F展厅 · 完整导览'
    assert view['map'] == 'bj-2f' and view['cards'] == 4


def test_summary_of_a_solution_that_is_not_a_case_is_empty_not_a_crash():
    view = benchmark_case.summary({'formatVersion': 1, 'canvas': {}})

    assert view['prompt'] == '' and view['injections'] == 0
