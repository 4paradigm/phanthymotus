"""两次运行之间，分数真的动了吗。

这一组守的是一条纪律，不是一个算法：**说不出显著差异的时候要说「测不出」**，
而不是把两个均值之差当成结论。

`tools/llm_bench` 的 README 记着这条的来历 ——「手工评测时正是这种点估计让我把 0.39s
的差异当成了结论，而它其实完全在波动范围内」。基准测试面板上原先只有 mean ± stdev，
照着它说「涨了」是同一个错误。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_compare.py -q
"""

import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'compare-test.db'))

import benchmark_store as bs  # noqa: E402


@pytest.fixture(autouse=True)
def clean():
    yield
    for run in bs.list_runs(limit=500):
        bs.delete_run(run['id'])


def run_with(scores):
    """一次运行，第 i 次重复得 scores[i] 分。"""
    run_id = bs.create_run('用例', n_repeats=len(scores))
    for index, score in enumerate(scores):
        bs.add_case(run_id, scenario='用例', repeat_idx=index, seed=index, score=score)
    bs.finish_run(run_id, score_total=sum(scores) / len(scores))
    return run_id


def test_a_real_improvement_is_called_one():
    before = run_with([40, 42, 41, 39, 43, 40])
    after = run_with([80, 82, 81, 79, 83, 80])

    result = bs.compare_runs(before, after)

    assert result['available'] is True
    assert result['significant'] is True and result['verdict'] == '更好'
    assert result['paired'] == 6


def test_a_real_regression_is_called_one():
    before = run_with([80, 82, 81, 79, 83, 80])
    after = run_with([40, 42, 41, 39, 43, 40])

    assert bs.compare_runs(before, after)['verdict'] == '更差'


def test_noise_is_not_called_an_improvement():
    """两个均值不同不等于有差别。这是这一组存在的全部理由。"""
    before = run_with([70, 74, 68, 72, 71, 69])
    after = run_with([71, 73, 70, 71, 72, 70])

    result = bs.compare_runs(before, after)

    assert result['significant'] is False
    assert result['verdict'] == '测不出显著差异'


def test_one_repeat_each_cannot_support_a_verdict():
    """n=1 是一次抛硬币。面板上那句「跑一次得到的是一个样本，不是结论」，
    在这里必须真的生效 —— 否则它只是一句免责声明。"""
    result = bs.compare_runs(run_with([40]), run_with([90]))

    assert result['available'] is False
    assert '无法判断显著性' in result['reason']
    assert 'verdict' not in result        # 不给结论，而不是给一个弱结论


def test_runs_pair_by_repeat_index_because_that_is_where_the_seed_matches():
    """第 i 次重复两边用的是同一个 seed —— seed 存在的理由就是让两次运行之间有东西
    可以配对。重复数不同的两次运行，只比对得上的那几次。"""
    before = run_with([40, 42, 41, 39])
    after = run_with([80, 82, 81, 79, 83, 80])

    assert bs.compare_runs(before, after)['paired'] == 4


def test_a_run_with_no_scores_at_all_is_not_a_comparison():
    empty = bs.create_run('用例', n_repeats=0)

    result = bs.compare_runs(empty, run_with([80, 81, 82]))

    assert result['available'] is False


def test_median_and_mean_pointing_opposite_ways_is_not_a_difference():
    """一方赢在多数请求、输在长尾。`llm_bench` 的 README 单列了这一条，因为它看起来
    最像「有提升」—— 中位数涨了。"""
    # 五次小幅变好，一次严重变差：中位 > 0 而均值 < 0。
    before = run_with([70, 70, 70, 70, 70, 70])
    after = run_with([72, 72, 72, 72, 72, 20])

    result = bs.compare_runs(before, after)

    assert result['sign_conflict'] is True
    assert result['significant'] is False
