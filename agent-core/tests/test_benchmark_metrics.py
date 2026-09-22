"""一次运行算出来的那些确切数字。

指标是整套评分的地基：原则分对着它判，裁判拿着它做判断。地基上有一个数算错了，
上面每一层都会跟着错，而且错得很像模像样 —— 一个算错的「静默思考时间 3.2s」不会报错，
只会让人以为体验很好。

所以这一组测的是**算得对不对**，不是「函数能跑」。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_benchmark_metrics.py -q
"""

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'metrics-test.db'))

import benchmark_metrics as bm  # noqa: E402

U = bm.Unmeasurable


def nav(t0, t1, label='P1', action_id='a1', end='arrive'):
    return [{'event': 'nav_start', 't': t0, 'label': label, 'action_id': action_id},
            {'event': end, 't': t1, 'label': label, 'action_id': action_id}]


def speak(t0, t1, action_id='s1'):
    return [{'event': 'speak_start', 't': t0, 'action_id': action_id},
            {'event': 'speak_end', 't': t1, 'action_id': action_id, 'status': 'completed'}]


def facts(events, source='agent-core', **extra):
    return {'events': events, 'acp_posts': [], 'source': source, **extra}


def span(name, start, end):
    return {'span': name, 'start_ts': start, 'end_ts': end}


# ── 物理世界时序性 ────────────────────────────────────────────────────────────

def test_speaking_while_still_driving_is_counted():
    """没到就开讲 —— 把没看见的东西说成看见了。"""
    events = nav(0, 20) + speak(5, 9)

    assert bm.world_timing(facts(events))['spoke_before_arrival'] == 1


def test_speaking_after_arrival_is_not_counted():
    events = nav(0, 20) + speak(21, 25)

    assert bm.world_timing(facts(events))['spoke_before_arrival'] == 0


def test_speaking_at_the_moment_of_arrival_is_not_counted():
    """到达那一刻开口，speak_start 和 arrive 可能只差几毫秒。

    按「开始时刻落在导航区间内」判，会把**每一次正常讲解**都算成违规 —— 一条恒为真的
    指标，比没有这条指标更糟。
    """
    events = nav(0, 20) + speak(19.98, 24)

    assert bm.world_timing(facts(events))['spoke_before_arrival'] == 0


def test_leaving_before_finishing_a_sentence_is_counted():
    events = speak(0, 10) + nav(4, 30)

    assert bm.world_timing(facts(events))['left_while_speaking'] == 1


def test_overlapping_legs_are_paired_by_action_id_not_by_order():
    """两段动作重叠时按先后配对，会把 A 的开始配给 B 的结束 —— 算出来的时长是负的。"""
    events = [
        {'event': 'nav_start', 't': 0, 'action_id': 'a1'},
        {'event': 'nav_start', 't': 1, 'action_id': 'a2'},
        {'event': 'arrive', 't': 30, 'action_id': 'a2'},
        {'event': 'arrive', 't': 40, 'action_id': 'a1'},
    ]

    assert bm.world_timing(facts(events))['nav_legs'] == 2


def test_a_duplicate_terminal_post_is_a_contradiction():
    """重复上报在 agent-core 侧是静默的 —— `mark_action_complete` 直接覆盖。"""
    data = facts(nav(0, 10))
    data['acp_posts'] = [{'action_id': 'a1', 'status': 'completed'},
                         {'action_id': 'a1', 'status': 'completed'}]

    assert bm.world_timing(data)['acp_contradictions'] == 1


def test_claiming_completed_for_a_leg_that_never_arrived_is_a_contradiction():
    """ACP 绝不能撒的那个谎：被放弃的一段报 completed，等于说机器人到过一个它没去过
    的地方。"""
    data = facts(nav(0, 10, end='nav_cancelled'))
    data['acp_posts'] = [{'action_id': 'a1', 'status': 'completed'}]

    assert bm.world_timing(data)['acp_contradictions'] == 1


# ── 同步执行效率 ──────────────────────────────────────────────────────────────

def test_total_seconds_comes_from_the_run_window():
    result = bm.concurrency(facts([]), [], (1000.0, 1186.5))

    assert result['total_seconds'] == 186.5


def test_overlapping_tools_show_parallelism_above_one():
    spans = [span('tool:loco', 0, 10), span('tool:tts', 2, 8)]

    assert bm.concurrency(facts([]), spans, (0, 10))['tool_parallelism'] == 1.6


def test_strictly_sequential_tools_show_parallelism_of_one():
    spans = [span('tool:loco', 0, 5), span('tool:tts', 5, 10)]

    assert bm.concurrency(facts([]), spans, (0, 10))['tool_parallelism'] == 1.0


def test_idle_time_is_what_nothing_was_recorded_for():
    """全程 100 秒，只有 20 秒在推理或调工具 —— 剩下 80 秒没有任何记录。"""
    spans = [span('llm_round_1', 0, 10), span('tool:loco', 10, 20)]

    assert bm.concurrency(facts([]), spans, (0, 100))['idle_seconds'] == 80.0


def test_walking_then_talking_counts_as_serialised():
    """走完才说，而嘴和底盘不是同一个部位 —— 这两件事本可以同时做。"""
    events = nav(0, 10) + speak(10.5, 14)

    assert bm.concurrency(facts(events), [], (0, 20))['serialised_cross_channel'] == 1


def test_walking_while_talking_is_not_serialised():
    events = nav(0, 10) + speak(2, 8)

    assert bm.concurrency(facts(events), [], (0, 20))['serialised_cross_channel'] == 0


# ── LLM 延时 ──────────────────────────────────────────────────────────────────

def test_latency_percentiles_come_from_the_round_spans():
    spans = [span(f'llm_round_{i}', 0, d) for i, d in enumerate([2, 4, 6, 8, 48])]

    result = bm.llm_latency(spans)

    assert result['median_s'] == 6.0 and result['max_s'] == 48.0
    assert result['rounds'] == 5 and result['complete'] is True


def test_failed_rounds_are_excluded_and_the_gap_is_stated():
    """幸存者偏差：失败往往是秒拒（0.06s 的 503），混进平均会让**更差的那组看起来更
    快**。分位数旁不标 (N/M)，读的人无从知道它只代表活下来的那些。"""
    spans = [span(f'llm_round_{i}', 0, 5) for i in range(10)]

    result = bm.llm_latency(spans, rounds_ok=7)

    assert result['complete'] is False
    assert '(7/10)' in result['note'] and '3 轮' in result['note']


def test_a_run_with_no_rounds_says_so_rather_than_reporting_zero():
    result = bm.llm_latency([])

    assert result['rounds'] == 0 and result['median_s'] is None


# ── cache ─────────────────────────────────────────────────────────────────────

def test_cache_ratio_is_cached_over_prompt():
    assert bm.cache_hit({'prompt_tokens': 1000, 'cached_tokens': 310})['ratio'] == 0.31


def test_no_tokens_at_all_is_unmeasurable_not_zero_percent():
    """0% 命中和「这次根本没调过 LLM」是两件事，混在一起趋势图就开始撒谎。"""
    assert isinstance(bm.cache_hit({})['ratio'], U)


# ── 用户体验 ──────────────────────────────────────────────────────────────────

def test_blank_time_is_measured_only_while_the_robot_is_busy():
    """站着不动不说话是正常的，正走着却一路不吭声才难受。"""
    events = nav(0, 30) + speak(10, 12)

    result = bm.ux(facts(events), {'started': 0})

    # 0→10 和 12→30 两段空白，18 秒那段是最长的。
    assert result['silence_max_s'] == 18.0
    assert result['silence_count'] == 2


def test_the_average_is_judged_and_the_max_is_still_reported():
    """只判最长会被一次离群值支配；只报平均会把一段 60 秒的空白稀释掉，而那一段正是
    用户真正会抱怨的。两个都要有。"""
    events = (nav(0, 60, action_id='n1') + speak(59, 60, action_id='s1')
              + nav(60, 62, action_id='n2') + speak(60.5, 62, action_id='s2'))

    result = bm.ux(facts(events), {'started': 0})

    assert result['silence_max_s'] == 59.0
    assert result['silence_avg_s'] < result['silence_max_s']


def test_first_response_is_measured_from_when_the_user_finished_speaking():
    events = speak(4.2, 9)

    result = bm.ux(facts(events), {'started': 1000.0, 'prompt_at': 1000.0})

    assert result['first_response_s'] == 4.2


def test_interrupt_response_is_the_worst_one_not_the_best():
    """有两次插话，一次 1 秒有反应、一次 8 秒 —— 报 8。报最好的那次等于替它遮丑。"""
    events = [{'event': 'speak_start', 't': 11.0, 'action_id': 'x'},
              {'event': 'nav_cancelled', 't': 28.0, 'action_id': 'y'}]

    result = bm.ux(facts(events), {'started': 0, 'injections_at': [10.0, 20.0]})

    assert result['interrupt_response_s'] == 8.0


def test_cross_clock_metrics_are_refused_when_the_world_is_simulated():
    """仿真器的事实流用仿真时钟（可以加速），而「用户说完」是墙钟。两个钟相减出来的
    秒数没有意义 —— 算一个看着像模像样、其实除以了错误单位的数，比不算更糟。"""
    events = nav(0, 30) + speak(10, 12)

    result = bm.ux(facts(events, source='simulator'),
                   {'started': 1000.0, 'prompt_at': 1000.0, 'injections_at': [1010.0]})

    assert isinstance(result['first_response_s'], U)
    assert isinstance(result['interrupt_response_s'], U)
    assert '仿真时钟' in result['first_response_s'].why
    # 单钟内部的指标不受影响 —— 它的单位是那个世界的秒，但相减是有意义的。
    assert result['silence_max_s'] == 18.0


# ── 安全 ──────────────────────────────────────────────────────────────────────

def test_a_crash_is_counted_with_its_reason():
    events = [{'event': 'nav_failed', 't': 3, 'reason': '前方占用'}]

    result = bm.physical_safety(facts(events))

    assert result['nav_failed'] == 1 and result['first_reason'] == '前方占用'


def test_missing_trail_data_is_unmeasurable_not_zero():
    """缺席读成 0 会让安全维度稳定报一个从没查过的满分 —— 这个缺陷真发生过。"""
    result = bm.physical_safety(facts([]))

    assert isinstance(result['trail_occupied'], U)


def test_trail_data_of_zero_really_means_zero():
    assert bm.physical_safety(facts([], trail_occupied=0))['trail_occupied'] == 0


def test_no_crashes_without_trail_data_proves_nothing():
    """**判定用的是 `incidents`，不是 `nav_failed`。**

    `nav_failed` 是积分器自己报的 —— 只看它，等于让积分器报告自己的 bug，这正是当初
    引入 `trail_occupied` 的理由。两者分开判的话，「没有轨迹数据时的 0 次撞停」会把安全
    维度顶成 100，而独立的那半根本没查过。旧代码修掉过一次这个缺陷，换个结构它原样
    回来了一次，所以这条守着第三次。
    """
    result = bm.physical_safety(facts([]))

    assert result['nav_failed'] == 0          # 数还是报出来的
    assert isinstance(result['incidents'], U)  # 但它不构成「没撞」的证据


def test_a_crash_is_positive_evidence_even_without_trail_data():
    """撞停谁报的都算数 —— 阳性证据不需要旁证。"""
    events = [{'event': 'nav_failed', 't': 3, 'reason': '前方占用'}]

    assert bm.physical_safety(facts(events))['incidents'] == 1


def test_a_clean_run_with_trail_data_is_actually_clean():
    assert bm.physical_safety(facts([], trail_occupied=0))['incidents'] == 0


# ── 总装 ──────────────────────────────────────────────────────────────────────

def test_every_principle_gets_a_block_even_when_there_is_nothing_to_report():
    """七条原则一条都不能少。少了的那条在面板上会变成「没这个维度」，而不是
    「这次没测到」—— 两者读起来完全不同。"""
    result = bm.observations(facts([]), [], {}, (0, 10))

    assert set(result) == {'world_timing', 'concurrency', 'llm_latency', 'cache_hit',
                           'answer_quality', 'ux', 'physical_safety'}


def test_answer_quality_has_no_metrics_by_design():
    """它是唯一一条真的没有可算指标的原则，所以也是唯一必须由裁判判的。"""
    assert bm.observations(facts([]), [], {}, (0, 10))['answer_quality'] == {}
