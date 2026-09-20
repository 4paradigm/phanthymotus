"""LLM 裁判：喂它什么、它回来的东西怎么变成分。

这里**不发真实请求**。要钉的不是「模型判得准不准」（那不是测试能回答的问题），而是
**裁判周围那圈约束真的生效**：不给工具、解析不出来不记 0 分、截断要写进材料里、漏判
算判不了。每一条都对应一种「测试全绿但基准测试在撒谎」的失败方式。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_judge.py -q
"""

import asyncio
import json
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'judge-test.db'))

import benchmark_judge as bj  # noqa: E402


def case(**over):
    block = {'run': {'prompt': '带我转一下展区'},
             'requirements': [{'id': 'r0', 'text': '到了再讲', 'weight': 25,
                               'dimension': 'world_timing'}]}
    block.update(over)
    return {'test': block}


def reply(content):
    """假的 client.call —— 只回一段文本。"""
    async def fake(messages, tools, **kwargs):
        fake.seen.append({'messages': messages, 'tools': tools, **kwargs})
        return {'content': content() if callable(content) else content}
    fake.seen = []
    return fake


def run(case_payload, fake, **kw):
    import client
    original = client.call
    client.call = fake
    try:
        return asyncio.run(bj.judge(case_payload, {'llm_latency': {'median_s': 4}},
                                    {'events': [], 'acp_posts': []}, [], **kw))
    finally:
        client.call = original


# ── 裁判不给工具 ──────────────────────────────────────────────────────────────

def test_the_judge_is_called_with_no_tools():
    """一个能调工具的裁判，可以在评判一次运行的过程中改变这次运行。"""
    fake = reply(json.dumps({'verdicts': [{'id': 'r0', 'verdict': 'pass',
                                           'reason': '每站都到了才讲'}]}))

    run(case(), fake)

    assert fake.seen[0]['tools'] == []


def test_the_call_is_attributed_so_the_judge_is_not_free():
    """裁判也要花钱。一次跑分里裁判花了多少，和被测系统花了多少一样该被看见。"""
    fake = reply(json.dumps({'verdicts': [{'id': 'r0', 'verdict': 'pass', 'reason': 'x'}]}))

    run(case(), fake)

    assert fake.seen[0]['caller_info'] == {'agent_type': 'benchmark_judge'}


def test_a_case_can_name_its_own_judge_model():
    fake = reply(json.dumps({'verdicts': [{'id': 'r0', 'verdict': 'pass', 'reason': 'x'}]}))

    result = run(case(judge={'model': 'some/other-model'}), fake)

    assert fake.seen[0]['model_override'] == 'some/other-model'
    assert result['model'] == 'some/other-model'


# ── 判决 → 计分项 ─────────────────────────────────────────────────────────────

def test_a_pass_becomes_a_scoring_item_with_its_reason():
    fake = reply(json.dumps({'verdicts': [{'id': 'r0', 'verdict': 'pass',
                                           'reason': '三站都是到达之后才开讲'}]}))

    item = run(case(), fake)['items'][0]

    assert item['ok'] is True and item['measurable'] is True
    assert item['dimension'] == 'world_timing' and item['weight'] == 25
    assert '到达之后' in item['detail']


def test_unmeasurable_is_not_a_failure():
    """「判不了」和「判了没过」是两件事 —— 混在一起，分母就错了。"""
    fake = reply(json.dumps({'verdicts': [{'id': 'r0', 'verdict': 'unmeasurable',
                                           'reason': '材料里没有播报记录'}]}))

    item = run(case(), fake)['items'][0]

    assert item['measurable'] is False and item['ok'] is False


def test_a_requirement_the_model_skipped_counts_as_unmeasurable():
    """漏判当成通过，等于让一个不肯回答的裁判给出满分。"""
    fake = reply(json.dumps({'verdicts': [{'id': 'nope', 'verdict': 'pass', 'reason': 'x'}]}))

    item = run(case(), fake)['items'][0]

    assert item['measurable'] is False
    assert '没有给出' in item['detail']


def test_an_unknown_verdict_word_is_dropped_without_dragging_down_the_others():
    """一条判决词不认识，不该连累认得出的那些 —— 但也不能猜它是 pass。"""
    two = case(requirements=[
        {'id': 'r0', 'text': '到了再讲', 'weight': 25, 'dimension': 'world_timing'},
        {'id': 'r1', 'text': '讲得清楚', 'weight': 10, 'dimension': 'answer_quality'}])
    fake = reply(json.dumps({'verdicts': [
        {'id': 'r0', 'verdict': '还行', 'reason': 'x'},
        {'id': 'r1', 'verdict': 'pass', 'reason': '条理清楚'}]}))

    items = {i['id']: i for i in run(two, fake)['items']}

    assert items['r0']['measurable'] is False      # 不认识 ⇒ 判不了，不是通过
    assert items['r1']['ok'] is True


def test_a_reply_where_no_verdict_is_usable_is_an_error():
    fake = reply(json.dumps({'verdicts': [{'id': 'r0', 'verdict': '还行', 'reason': 'x'}]}))

    result = run(case(), fake)

    assert result['items'] == [] and result['error']


# ── 解析 ──────────────────────────────────────────────────────────────────────

def test_a_fenced_json_block_is_accepted():
    """模型爱在 JSON 外面包一层 ```json —— 宽容包装，不宽容内容。"""
    verdicts = bj.parse('```json\n{"verdicts": [{"id": "r0", "verdict": "fail",'
                        ' "reason": "没到就讲了"}]}\n```')

    assert verdicts['r0']['verdict'] == 'fail'


def test_prose_around_the_json_is_tolerated():
    verdicts = bj.parse('好的，我的判决如下：\n{"verdicts": [{"id": "r0",'
                        ' "verdict": "pass", "reason": "ok"}]}\n希望有帮助。')

    assert verdicts['r0']['verdict'] == 'pass'


def test_garbage_raises_rather_than_returning_an_empty_verdict_table():
    with pytest.raises(ValueError):
        bj.parse('我觉得这次跑得不错。')


# ── 解析失败不记 0 分 ─────────────────────────────────────────────────────────

def test_a_reply_that_never_parses_is_an_error_not_a_zero():
    """0 分的意思是「测了，没过」，而这里发生的是「没测成」。把后者记成前者，
    基准测试就在撒谎 —— 而且撒的是最难发现的那种谎：一个看起来很正常的低分。"""
    fake = reply('这次跑得挺好的。')

    result = run(case(), fake)

    assert result['items'] == []
    assert 'JSON' in result['error']


def test_a_bad_reply_gets_exactly_one_more_chance():
    replies = iter(['不是 JSON',
                    json.dumps({'verdicts': [{'id': 'r0', 'verdict': 'pass',
                                              'reason': '第二次好了'}]})])
    fake = reply(lambda: next(replies))

    result = run(case(), fake)

    assert len(fake.seen) == 2
    assert result['items'][0]['ok'] is True and result['error'] == ''


def test_a_transport_failure_is_reported_not_retried_forever():
    async def boom(messages, tools, **kwargs):
        boom.calls += 1
        raise RuntimeError('连不上')
    boom.calls = 0

    import client
    original = client.call
    client.call = boom
    try:
        result = asyncio.run(bj.judge(case(), {}, {'events': [], 'acp_posts': []}, []))
    finally:
        client.call = original

    assert boom.calls == 1 and result['items'] == []
    assert '连不上' in result['error']


# ── 材料 ──────────────────────────────────────────────────────────────────────

def big_facts(n=4000):
    return {'events': [{'event': 'nav_start', 't': i, 'label': f'站{i}'} for i in range(n)],
            'acp_posts': [{'action_id': f'a{i}', 'status': 'completed'} for i in range(n)]}


def test_truncation_is_announced_inside_the_material():
    """截断说明必须在材料**里面**，不是在日志里 —— 裁判看不见日志。

    不说的话它会以为自己看到了全部，然后为一段它根本没见过的过程下判决。
    """
    material = bj.build_material(case(), {'llm_latency': {}}, big_facts(), [],
                                 budget=3000)

    assert len(material) <= 3200
    assert '已截断' in material and '不是全部' in material


def test_a_short_run_is_not_marked_as_truncated():
    material = bj.build_material(case(), {'llm_latency': {'median_s': 4}},
                                 {'events': [], 'acp_posts': []}, [])

    assert '已截断' not in material


def test_metrics_come_first_so_truncation_eats_the_transcript_not_the_numbers():
    """指标是算出来的、最可靠，而两条轨道很长。真被截了，至少那几个数还在。"""
    material = bj.build_material(case(), {'llm_latency': {'median_s': 8.4}},
                                 big_facts(), [], budget=3000)

    assert material.index('指标') < material.index('世界事件流')
    assert '8.4' in material


def test_the_procedure_is_handed_over_as_its_own_requirement():
    """参考流程要让裁判逐步比对，所以那几步得跟着要求一起送过去。"""
    messages = bj.build_request(case(procedure='1. 打开地图\n2. 导航', requirements=[]),
                                {}, {'events': [], 'acp_posts': []}, [])

    asked = messages[-1]['content']
    assert '打开地图' in asked and 'procedure' in asked


def test_the_system_prompt_forbids_guessing():
    """材料里没有的就判不了 —— 「看起来应该没问题」不是证据。"""
    assert 'unmeasurable' in bj.SYSTEM and '不要推测' in bj.SYSTEM
