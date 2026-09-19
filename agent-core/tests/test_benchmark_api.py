"""Benchmark storage and API.

Two properties this file exists to hold, both of which are about honesty rather
than correctness:

* **A score is stored with the configuration it was measured against.** Without
  the model, the image tag and the git sha, "did the number move when I changed
  the prompt" — the only question anyone actually asks of a benchmark — is
  unanswerable, and two rows of scores are not comparable.
* **n travels with the score.** An LLM is stochastic, so one run is one sample.
  A number shown without its sample size reads as a conclusion.

And one about layering: agent-core does not score. The simulator judges, because
the judge and the system under test have to be disjoint — otherwise a broken ACP
path quietly marks itself green. This layer only dispatches, collects and
records.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_api.py -q
"""

import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'benchmark-test.db'))

import benchmark_store  # noqa: E402
import mcp_client  # noqa: E402
from api import benchmark  # noqa: E402


@pytest.fixture(autouse=True)
def clean_runs():
    for run in benchmark_store.list_runs(limit=500):
        benchmark_store.delete_run(run['id'])
    yield
    for run in benchmark_store.list_runs(limit=500):
        benchmark_store.delete_run(run['id'])


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


def test_available_is_what_hides_the_panel_on_a_shipped_robot(monkeypatch):
    monkeypatch.setattr(mcp_client, 'registry', {})

    result = asyncio.run(benchmark.available())

    assert result['available'] is False
    assert result['mcp_id'] is None


# ── dispatch and collect ─────────────────────────────────────────────────────

def test_run_forwards_the_batch_and_opens_a_record(simulator):
    simulator['replies'][('sim_scenario', 'run_suite')] = {
        'state': 'running', 'scenarios': ['exhibition_tour'],
        'suite': {'state': 'loading', 'cases': 2}}

    result = asyncio.run(benchmark.run(benchmark.RunRequest(
        scenarios=['exhibition_tour'], repeats=2)))

    assert result['run_id'].startswith('bm-')
    assert ('mcp-sim', 'sim_scenario', {
        'action': 'run_suite', 'scenarios': ['exhibition_tour'],
        'repeats': 2, 'seed': 0}) in simulator['calls']
    assert benchmark_store.get_run(result['run_id'])['n_repeats'] == 2


def test_run_without_a_simulator_is_a_clear_409(monkeypatch):
    monkeypatch.setattr(mcp_client, 'registry', {})

    with pytest.raises(Exception) as caught:
        asyncio.run(benchmark.run(benchmark.RunRequest(scenarios=['x'])))

    assert getattr(caught.value, 'status_code', None) == 409


def test_a_driver_side_error_surfaces_rather_than_creating_an_empty_run(simulator):
    simulator['replies'][('sim_scenario', 'run_suite')] = {'error': 'no runnable scenario'}

    with pytest.raises(Exception) as caught:
        asyncio.run(benchmark.run(benchmark.RunRequest(scenarios=['nope'])))

    assert getattr(caught.value, 'status_code', None) == 400
    assert benchmark_store.list_runs() == []


def test_progress_is_polled_from_the_resource_card(simulator):
    simulator['replies'][('sim_report', 'suite')] = SUITE_SUMMARY

    result = asyncio.run(benchmark.progress())

    assert result['state'] == 'done'
    assert ('mcp-sim', 'sim_report', {'what': 'suite'}) in simulator['calls']


def test_collect_stores_the_verdicts_the_simulator_produced(simulator):
    """agent-core records; it does not judge. The judge and the system under test
    stay disjoint, or a broken ACP path marks itself green."""
    simulator['replies'][('sim_report', 'suite')] = SUITE_SUMMARY
    run_id = benchmark_store.create_run('exhibition_tour', n_repeats=2)

    stored = asyncio.run(benchmark.collect(run_id))

    assert stored['status'] == 'done'
    assert stored['score_total'] == 87.5
    assert stored['score_stdev'] == 17.7
    assert [c['outcome'] for c in stored['cases']] == ['ok', 'stalled']
    assert stored['cases'][1]['assertions'] == ['resume_correctness']
    assert stored['cases'][0]['elapsed_ms'] == 104600


def test_collect_keeps_a_partial_batch_marked_partial(simulator):
    simulator['replies'][('sim_report', 'suite')] = {**SUITE_SUMMARY, 'state': 'running'}
    run_id = benchmark_store.create_run('exhibition_tour', n_repeats=2)

    stored = asyncio.run(benchmark.collect(run_id))

    assert stored['status'] == 'partial'


def test_collect_on_an_unknown_run_is_a_404(simulator):
    with pytest.raises(Exception) as caught:
        asyncio.run(benchmark.collect('bm-nope'))

    assert getattr(caught.value, 'status_code', None) == 404


def test_scenarios_are_listed_through_the_report_card(simulator):
    simulator['replies'][('sim_report', 'list')] = {
        'scenarios': [{'slug': 'exhibition_tour', 'name': '展厅导览 · 基础巡游',
                       'waypoints': ['入口'], 'injections': [{'at': None}]}]}

    result = asyncio.run(benchmark.scenarios())

    assert result['scenarios'][0]['slug'] == 'exhibition_tour'
    assert result['mcp_id'] == 'mcp-sim'


def test_scenarios_without_a_simulator_returns_empty_not_an_error(monkeypatch):
    monkeypatch.setattr(mcp_client, 'registry', {})

    result = asyncio.run(benchmark.scenarios())

    assert result['scenarios'] == []
    assert 'error' in result


def test_the_environment_block_names_the_model_under_test(simulator):
    result = asyncio.run(benchmark.available())

    assert set(result['environment']) >= {'tier', 'llm_model', 'llm_provider',
                                          'host', 'image_tags', 'git_shas'}
