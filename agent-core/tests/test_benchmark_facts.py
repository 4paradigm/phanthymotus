"""真机上的事实记录器。

驱动它的是 `mcp_client` 的两个订阅点，所以这里**真的走那两个回调**，不直接调
`Recorder` 的方法 —— 中间那段接线（订阅一次、按 `_current` 短路、结算时状态从哪查）
正是容易错的地方，绕过去测等于没测。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_facts.py -q
"""

import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'facts-test.db'))

import benchmark_facts  # noqa: E402
import mcp_client  # noqa: E402

TOUR = ['入口', '一号展区', '二号展区']

BASE = frozenset({'base'})
MOUTH = frozenset({'mouth'})


@pytest.fixture(autouse=True)
def recorder():
    """每条用例一个干净的记录器，且 mcp_client 的 pending 表不留残留。"""
    made = benchmark_facts.start(TOUR)
    yield made
    benchmark_facts.stop()
    for table in (mcp_client._pending_results, mcp_client._action_outcomes):
        table.clear()


def register(action_id, tool, args, resource):
    """走 mcp_client 的注册通知，而不是直接调记录器。"""
    mcp_client._notify_registered(action_id, tool, args, resource)


def complete(action_id, status='completed', result=None):
    """完成那条路：payload 留在 `_pending_results`，然后通知。"""
    mcp_client._pending_results[action_id] = {'status': status, 'result': result or {}}
    mcp_client._notify_settled(action_id)


def forget(action_id, outcome='timeout'):
    """超时 / 取消那条路：**先拆表再通知**，和 `_forget_pending` 的真实顺序一致。"""
    mcp_client._record_outcome(action_id, outcome)
    mcp_client._pending_results.pop(action_id, None)
    mcp_client._notify_settled(action_id)


def names(facts):
    return [e['event'] for e in facts['events']]


# ── 事实的形状 ────────────────────────────────────────────────────────────────

def test_a_navigation_becomes_nav_start_then_arrive(recorder):
    register('a1', 'controlled_spatial', {'action': 'navigate_to_tag',
                                          'tag_name': '一号展区'}, BASE)
    complete('a1')

    facts = recorder.facts()

    assert names(facts) == ['nav_start', 'arrive']
    assert [e.get('label') for e in facts['events']] == ['一号展区', '一号展区']


def test_speaking_becomes_speak_start_then_speak_end(recorder):
    register('s1', 'tts', {'text': '这里是一号展区'}, MOUTH)
    complete('s1')

    assert names(recorder.facts()) == ['speak_start', 'speak_end']


def test_a_cancelled_leg_is_not_an_arrival(recorder):
    """被打断的一段绝不能记成到达 —— 那等于说机器人到过一个它没去过的展位。"""
    register('a1', 'controlled_spatial', {'tag_name': '二号展区'}, BASE)
    complete('a1', status='cancelled')

    assert names(recorder.facts()) == ['nav_start', 'nav_cancelled']


def test_a_timeout_still_produces_an_ending(recorder):
    """超时路径上 `_forget_pending` 先拆表再通知，`_pending_results` 已经没了。

    只问那一张表的话，一次超时就变成「开始了但永远没有结局」，而事件流里
    「没到」和「记漏了」完全一样。
    """
    register('a1', 'controlled_spatial', {'tag_name': '二号展区'}, BASE)
    forget('a1', outcome='timeout')

    facts = recorder.facts()
    assert names(facts) == ['nav_start', 'nav_failed']
    assert facts['events'][-1]['status'] == 'timeout'


# ── 分类按申报，不按名字 ──────────────────────────────────────────────────────

def test_a_tool_that_declares_nothing_produces_no_facts(recorder):
    """没申报 `x-resource` 就不产出事实 —— 有意让它缺，不猜。

    一条猜错的事实会让裁判判错；缺一条至少还能在预检里说出来。
    """
    register('x1', 'switch_mode', {'mode': 'walk'}, None)
    complete('x1')

    assert recorder.facts()['events'] == []


def test_a_bare_string_channel_still_classifies(recorder):
    """驱动写的就是 `"x-resource": "mouth"`（一个字符串，不是列表）。

    注册表里存的是归一化过的 frozenset，但真让一个裸串走到这儿，`set("mouth")` 会
    变成一堆单字母，于是一条讲解事实都不产出 —— 方向安全，但**无声**。
    """
    register('s1', 'tts', {'text': '你好'}, 'mouth')
    complete('s1')

    assert names(recorder.facts()) == ['speak_start', 'speak_end']


def test_the_real_actuator_names_are_classified_by_channel(recorder):
    """`loco` 这个名字里没有 move / nav / goto，任何关键词表都抓不住它。

    `peer/tools.py` 就是在这儿栽过：查 `move`/`grasp`/`speak`，于是 viewer 能驱动底盘。
    """
    register('a1', 'loco', {'tag_name': '入口'}, BASE)
    complete('a1')

    assert names(recorder.facts()) == ['nav_start', 'arrive']


# ── 站名靠值匹配 ──────────────────────────────────────────────────────────────

def test_the_dispatch_arguments_are_recorded_verbatim(recorder):
    """各家参数名不同：天轶 `tag_name`、仿真器 `label`、下一家会是别的。挑一个名字读，
    等于把某一家的 schema 焊进事实层 —— 所以原样记下，不解释。

    要求是人话，判它的是裁判：「有没有按我说的顺序去那几个地方」是它读着参数就能回答
    的问题，而这里少做一层猜测，就少一处会猜错的地方。
    """
    register('a1', 'controlled_spatial', {'tag_name': '一号展区', 'speed': 0.5}, BASE)
    register('a2', 'nav', {'poi': '二号展区'}, BASE)

    args = [e.get('args') for e in recorder.facts()['events']]
    assert args == [{'tag_name': '一号展区'}, {'poi': '二号展区'}]


def test_numbers_are_left_out_of_the_recorded_arguments(recorder):
    """坐标、超时、重试次数对「它去了哪儿」没有帮助，却会把事实流撑大 ——
    而事实流是要整份喂给裁判的。"""
    register('a1', 'controlled_spatial', {'x': 5.98, 'y': 8.77, 'retries': 3}, BASE)

    assert recorder.facts()['events'][0].get('args') is None


def test_a_very_wide_parameter_table_is_capped(recorder):
    register('a1', 'nav', {f'k{i}': f'v{i}' for i in range(20)}, BASE)

    assert len(recorder.facts()['events'][0]['args']) == 6


# ── ACP 上报 ──────────────────────────────────────────────────────────────────

def test_acp_posts_are_kept_whole_and_gain_the_label_we_worked_out(recorder):
    """裁判的 `interrupted_leg` 读 `result.label`，而真机的 result 里没有这一项
    （天轶回的是 pose / elapsed_s）。补上我们自己认出来的，否则那条断言在真机上
    没有任何可比的东西。"""
    register('a1', 'controlled_spatial', {'tag_name': '二号展区'}, BASE)

    benchmark_facts.note_acp_post({'action_id': 'a1', 'status': 'cancelled',
                                   'result': {'pose': [1.0, 2.0], 'progress':
                                              {'fraction': 0.4}}})

    post = recorder.facts()['acp_posts'][0]
    assert post['result']['label'] == '二号展区'
    assert post['result']['pose'] == [1.0, 2.0]       # 原样，不重组


def test_a_post_that_arrives_after_the_settle_still_finds_its_leg(recorder):
    """上报和结算谁先到**不保证**。

    `/acp/complete` 里两句的先后可以改，而 `mark_action_complete` 还有 SSE 那条调用
    路径根本不经过那个端点。结算时就把这个 action 摘掉的话，晚到的上报认不出属于
    哪一段路，`result.label` 补不上，`interrupted_leg` 在真机上没有可比的东西。
    """
    register('a1', 'controlled_spatial', {'tag_name': '二号展区'}, BASE)
    complete('a1', status='cancelled')

    benchmark_facts.note_acp_post({'action_id': 'a1', 'status': 'cancelled',
                                   'result': {'progress': {'fraction': 0.4}}})

    assert recorder.facts()['acp_posts'][0]['result']['label'] == '二号展区'


def test_a_result_that_already_has_a_label_is_left_alone(recorder):
    """仿真器自己给了 label 就用它的 —— 它看得见世界，比我们从参数里认的可信。"""
    register('a1', 'controlled_spatial', {'tag_name': '入口'}, BASE)

    benchmark_facts.note_acp_post({'action_id': 'a1', 'status': 'completed',
                                   'result': {'label': '一号展区'}})

    assert recorder.facts()['acp_posts'][0]['result']['label'] == '一号展区'


# ── 缺的那一半要缺得明显 ──────────────────────────────────────────────────────

def test_facts_carry_no_trail_occupied_at_all(recorder):
    """**不能**放一个 0 进去。

    裁判把这个字段的缺席读成「不可测」，把 0 读成「一个点都没压到」。放 0 等于让真机
    每次都报一个从没查过的满分安全维度 —— 那正是刚修掉的那个缺陷。
    """
    register('a1', 'controlled_spatial', {'tag_name': '入口'}, BASE)
    complete('a1')

    assert 'trail_occupied' not in recorder.facts()


def test_the_judge_eats_what_the_recorder_produces(recorder):
    """**这条是这个模块存在的全部理由。**

    裁判的文档写着「只要能吐出同样形状的事件流，这同一套断言直接可用」。那个「只要」
    就是这里 —— 形状对不上的话，上面每一条都还是绿的，而真机跑分是错的。

    第一版就在这儿栽过：`speech_start` vs `speak_start`，差一个词，
    `announce_after_arrive` 恒为失败且全程零报错。
    """
    import benchmark_case

    tour = [('a1', '入口'), ('a2', '一号展区'), ('a3', '二号展区')]
    for index, (action_id, label) in enumerate(tour):
        register(action_id, 'controlled_spatial', {'tag_name': label}, BASE)
        complete(action_id)
        say = f's{index}'
        register(say, 'tts', {'text': f'这里是{label}'}, MOUTH)
        complete(say)

    import benchmark_metrics

    facts = recorder.facts()
    payload = {'test': {'requirements': []}}

    seen = benchmark_metrics.observations(facts, [], {}, (0, 60))
    items = benchmark_case.check_targets(seen, benchmark_case.targets(payload))
    score = benchmark_case.score(payload, items)

    # 每一站都是「到了才讲」，时序性这一条该满分。
    assert score['by_dimension']['world_timing'] == 100.0, \
        [i['detail'] for i in items if not i['ok']]
    # 真机测不到轨迹占用，安全维度必须是「不可测」而不是满分。
    assert score['by_dimension']['physical_safety'] is None
    assert any('占用格' in text for text in score['unmeasured'])


def test_nothing_is_recorded_once_the_run_is_over(recorder):
    """停了之后晚到的结算不该再写进去 —— 否则上一次运行的事实会漏进下一次。"""
    register('a1', 'controlled_spatial', {'tag_name': '入口'}, BASE)
    benchmark_facts.stop()
    complete('a1')

    assert names(recorder.facts()) == ['nav_start']
