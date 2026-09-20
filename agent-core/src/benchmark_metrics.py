"""benchmark_metrics.py — 一次运行算出来的那些确切数字。

## 为什么指标在前，判定在后

「用户体验好不好」对大模型是个氛围词，判出来的分不可信也不可比。「懵逼时长 23.4 秒」
不是 —— 它是个数，两次运行之间能比，也能画趋势。

> **指标优先，模糊判定最少化。** 每条原则先有可计算的指标，LLM 只在指标之上做判断。

所以这个模块只做一件事：把一次运行的事实流、`perf_spans`、`token_usage` 换算成一组数。
它**不判分**。判分在 `benchmark_case`（拿数对目标）和 `benchmark_judge`（拿数做判断）。

这里也是原先那 7 条 CHECK 的去处 —— 它们被**降级成指标**。降级不是贬低：它们客观、便宜，
而且是 LLM 从对话记录里看不出来的（一条重复的 ACP 终态上报，长得和正常的一模一样）。

## 两个钟的问题

仿真器的事实流用的是**仿真时钟**（可以加速），而「用户说完这句话」「插话发出去」是墙钟。
两个钟相减出来的秒数没有意义 —— `api/benchmark.py::run_timeline` 的文档里已经记着这条，
那里的做法是两条轨道并排放而不强行合成一条。

这里照同一条规矩办：**跨钟的指标只在事实流本身就是墙钟时才算**（`facts['source'] ==
'agent-core'`，即 `benchmark_facts` 记的那一份）。仿真器给的事实走到这儿只会得到
「不可测」，附上理由 —— 而不是一个看着像模像样、其实除以了错误单位的数。

单钟内部的指标（比如两次播报之间隔了多久）不受影响，但它的单位是那个世界的秒。
"""

from __future__ import annotations

import statistics

# 指标算不出来时的统一形状。`None` 会被下游当成 0 或者当成"没有"，两种误读都出现过，
# 所以缺失是一个显式的对象，而且**必带理由** —— 「判不了」和「判了没过」是两件事，
# 说不出为什么判不了的话，读的人只能当它是后者。
class Unmeasurable:
    __slots__ = ('why',)

    def __init__(self, why: str):
        self.why = why

    def __repr__(self) -> str:
        return f'Unmeasurable({self.why!r})'

    def __eq__(self, other) -> bool:
        return isinstance(other, Unmeasurable) and other.why == self.why


def _pct(values: list[float], p: float) -> float:
    """分位。和 `tools/llm_bench/stats.py::pct` 同一套算法，照抄而不是 import ——
    那个包在 `tools/` 下，为一个三行函数把 `tools/` 塞进 `sys.path`，就会顺带把
    `llm_bench.config` 和 `src/config.py` 的抢名字问题带进运行时（`tests/test_llm_bench.py`
    开头记着这个坑）。显著性检验那种真有分量的才值得付这个代价。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """若干区间合并之后的总长。重叠的部分只算一次。"""
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


# ── 物理世界时序性 ────────────────────────────────────────────────────────────

# 结局由**事件名**表达，不由 `status` 字段。`arrive` 事件身上没有 `status` —— 去读它
# 只会得到空串，于是「到没到」永远判成「没到」：既漏掉所有「没到就开讲」，又把每一条
# 正常的 ACP 完成上报诬成撒谎。两条指标同时错，而且都错得不报错。
_OUTCOME = {'arrive': 'completed', 'nav_cancelled': 'cancelled',
            'nav_failed': 'failed', 'speak_end': 'completed'}


def _spans_of(events: list[dict], start_name: str, end_names: tuple) -> list[dict]:
    """把 `*_start` 和它的结局配成对。

    按 `action_id` 配，不按先后顺序配 —— 两段路重叠的时候（一边走一边说就是常态），
    按顺序配会把 A 的开始配给 B 的结束，算出来的时长可以是负的。
    """
    opened: dict[str, dict] = {}
    out: list[dict] = []
    for event in events:
        name = event.get('event')
        action_id = event.get('action_id') or ''
        if name == start_name:
            opened[action_id] = event
        elif name in end_names and action_id in opened:
            begin = opened.pop(action_id)
            out.append({'start': begin, 'end': event,
                        't0': begin.get('t', 0.0), 't1': event.get('t', 0.0),
                        'label': begin.get('label') or event.get('label') or '',
                        'status': event.get('status')
                                  or _OUTCOME.get(str(event.get('event')), '')})
    for leftover in opened.values():          # 开了没关的：跑到结束还没完成
        out.append({'start': leftover, 'end': None, 't0': leftover.get('t', 0.0),
                    't1': None, 'label': leftover.get('label', ''), 'status': 'open'})
    return out


def world_timing(facts: dict) -> dict:
    """到达前开讲、讲完前就走、ACP 说的和事实对不上。"""
    events = facts.get('events') or []
    navs = _spans_of(events, 'nav_start', ('arrive', 'nav_cancelled', 'nav_failed'))
    speaks = _spans_of(events, 'speak_start', ('speak_end',))

    # 没到就开讲：一段讲解**整个落在**一段导航的进行中，而那段导航最终到达了。
    # 「整个落在」是要紧的 —— 到达那一刻开口，speak_start 和 arrive 可能只差几毫秒，
    # 按「开始时刻在区间内」判会把每一次正常讲解都算成违规。
    spoke_early = 0
    for speak in speaks:
        for nav in navs:
            if nav['status'] != 'completed' and nav['end'] is not None:
                continue
            end = nav['t1']
            if end is None:
                continue
            if nav['t0'] <= speak['t0'] and (speak['t1'] or speak['t0']) < end:
                spoke_early += 1
                break

    # 讲完前就走：一段导航在一段讲解还没结束时就开始了。
    left_speaking = 0
    for nav in navs:
        for speak in speaks:
            if speak['t1'] is None:
                continue
            if speak['t0'] < nav['t0'] < speak['t1']:
                left_speaking += 1
                break

    return {
        'spoke_before_arrival': spoke_early,
        'left_while_speaking': left_speaking,
        'acp_contradictions': _acp_contradictions(facts, navs),
        'nav_legs': len(navs),
        'speak_turns': len(speaks),
    }


def _acp_contradictions(facts: dict, navs: list[dict]) -> int:
    """ACP 上报和事实流对不上的条数。

    两种都算：同一个 action 上报了不止一次终态（agent-core 侧是静默的 ——
    `mark_action_complete` 直接覆盖），以及上报说 `completed` 而事实流里那一段根本没到。
    后者是 ACP 唯一绝不能撒的谎：它等于告诉模型机器人到过一个它没去过的地方。
    """
    posts = facts.get('acp_posts') or []
    seen: dict[str, int] = {}
    for post in posts:
        action_id = str(post.get('action_id') or '')
        if action_id:
            seen[action_id] = seen.get(action_id, 0) + 1
    duplicates = sum(count - 1 for count in seen.values() if count > 1)

    arrived = {nav['start'].get('action_id') for nav in navs if nav['status'] == 'completed'}
    lying = sum(1 for post in posts
                if post.get('status') == 'completed'
                and str(post.get('action_id') or '') not in arrived
                and any(nav['start'].get('action_id') == post.get('action_id') for nav in navs))
    return duplicates + lying


# ── 同步执行效率 ──────────────────────────────────────────────────────────────

def concurrency(facts: dict, spans: list[dict], window: tuple) -> dict:
    """总任务完成时长，以及这段时间里有多少是真的在并行。

    **总时长故意没有默认目标** —— 三站的导览和十站的导览没有可比的标准。给一个拍脑袋
    的默认值，等于让每个长用例都无故扣分。它永远报出来、永远可看趋势，只在用例自己
    给了目标或写了要求时才参与判定。
    """
    started, ended = window
    total = max(0.0, float(ended or 0) - float(started or 0))

    tool_spans = [(s['start_ts'], s['end_ts']) for s in spans
                  if str(s.get('span', '')).startswith('tool:') and s.get('end_ts')]
    llm_spans = [(s['start_ts'], s['end_ts']) for s in spans
                 if str(s.get('span', '')).startswith('llm_round_') and s.get('end_ts')]

    tool_sum = sum(end - start for start, end in tool_spans)
    tool_union = _union_seconds(tool_spans)
    idle = max(0.0, total - _union_seconds(tool_spans + llm_spans))

    return {
        'total_seconds': round(total, 2),
        'llm_seconds': round(sum(end - start for start, end in llm_spans), 2),
        'tool_seconds': round(tool_sum, 2),
        # >1 表示真的有工具在同时跑；=1 表示全程一次只做一件事。
        'tool_parallelism': round(tool_sum / tool_union, 2) if tool_union else None,
        'idle_seconds': round(idle, 2),
        # 「这段时间里有多少是什么都没在做」。原本想量的是 barrier 等待占比，但
        # `perf_spans` 里**没有 barrier span** —— 采不到的指标不能承诺，所以量这个：
        # 空档里包含 barrier 等待，也包含别的停顿，含义更宽但是真的算得出来。
        'idle_ratio': round(idle / total, 3) if total > 0 else None,
        'serialised_cross_channel': _serialised_cross_channel(facts),
    }


def _serialised_cross_channel(facts: dict) -> int:
    """本可以同时做、却一前一后做了的次数。

    只看**不同物理通道**之间（走路和说话用的不是同一个部位，`x-resource` 就是这么
    申报的），所以「本可并行」不是猜的。同通道的当然只能排队 —— 一张嘴说不了两句话。
    """
    events = facts.get('events') or []
    navs = _spans_of(events, 'nav_start', ('arrive', 'nav_cancelled', 'nav_failed'))
    speaks = _spans_of(events, 'speak_start', ('speak_end',))
    serialised = 0
    for nav in navs:
        if nav['t1'] is None:
            continue
        for speak in speaks:
            if speak['t1'] is None:
                continue
            overlaps = nav['t0'] < speak['t1'] and speak['t0'] < nav['t1']
            adjacent = 0 <= speak['t0'] - nav['t1'] < 2.0 or 0 <= nav['t0'] - speak['t1'] < 2.0
            if adjacent and not overlaps:
                serialised += 1
    return serialised


# ── LLM 延时 ──────────────────────────────────────────────────────────────────

def llm_latency(spans: list[dict], rounds_ok: int | None = None) -> dict:
    """每轮推理的墙钟分布。

    **只统计成功的轮**，而且不满时强制标 `(N/M)` —— 这条是照搬 `tools/llm_bench`
    README 里那条「幸存者偏差」：失败往往是秒拒（0.06s 的 503），混进平均会让**更差的
    那一组看起来更快**。

    `rounds_ok` 是**推出来的，不是直接测的**：`perf_spans` 不记一轮成没成功，所以调用方
    用运行窗口内的 `token_usage` 行数当成功轮数（成功一次记一行）。这一点要说出来，
    免得被读成实测值。
    """
    durations = [float(s['end_ts']) - float(s['start_ts']) for s in spans
                 if str(s.get('span', '')).startswith('llm_round_') and s.get('end_ts')]
    total = len(durations)
    if not durations:
        return {'rounds': 0, 'median_s': None, 'p95_s': None, 'max_s': None,
                'complete': True, 'note': ''}

    ok = total if rounds_ok is None else min(int(rounds_ok), total)
    complete = ok >= total
    return {
        'rounds': total,
        'rounds_ok': ok,
        'median_s': round(statistics.median(durations), 2),
        'p95_s': round(_pct(durations, 0.95), 2),
        'max_s': round(max(durations), 2),
        'complete': complete,
        # 不完整时这句会被一路带到面板和裁判面前。分位数旁边不标 (N/M)，读的人无从知道
        # 它只代表活下来的那些。
        'note': '' if complete else f'({ok}/{total}) 有 {total - ok} 轮没有成功记录，'
                                    f'分位数只统计成功的轮',
    }


# ── cache 命中 ────────────────────────────────────────────────────────────────

def cache_hit(usage: dict) -> dict:
    prompt = float((usage or {}).get('prompt_tokens') or 0)
    cached = float((usage or {}).get('cached_tokens') or 0)
    if prompt <= 0:
        return {'prompt_tokens': 0, 'cached_tokens': 0,
                'ratio': Unmeasurable('这次运行没有记到任何 prompt token')}
    return {'prompt_tokens': int(prompt), 'cached_tokens': int(cached),
            'ratio': round(cached / prompt, 4)}


# ── 用户体验 ──────────────────────────────────────────────────────────────────

def ux(facts: dict, marks: dict) -> dict:
    """用户那一侧感觉到的几个数。

    **懵逼时长判平均、报最长。** 只判最长会被一次离群值支配；只报平均会把一段 60 秒的
    空白稀释掉，而那一段正是用户真正会抱怨的。两个都出，判定用平均。

    「懵逼」的定义是**机器人在忙却一声不吭**，不是单纯的静默：站着不动不说话是正常的，
    正走着却一路不吭声才难受。
    """
    events = facts.get('events') or []
    speaks = _spans_of(events, 'speak_start', ('speak_end',))
    navs = _spans_of(events, 'nav_start', ('arrive', 'nav_cancelled', 'nav_failed'))

    blanks = _blank_intervals(speaks, navs)
    out: dict = {
        'blank_avg_s': round(sum(blanks) / len(blanks), 2) if blanks else 0.0,
        'blank_max_s': round(max(blanks), 2) if blanks else 0.0,
        # 一共被晾了多久。平均和最长都答不了这个问题：十次 5 秒和一次 50 秒平均值不同、
        # 最长值也不同，但用户被晾的总时间一样长。
        'blank_total_s': round(sum(blanks), 2),
        'blank_count': len(blanks),
    }

    # 跨钟的两个：只有事实流本身就是墙钟时才算。见模块文档。
    if facts.get('source') != 'agent-core':
        why = '世界用的是仿真时钟，和墙钟记下的「用户说完」相减没有意义'
        out['first_response_s'] = Unmeasurable(why)
        out['interrupt_response_s'] = Unmeasurable(why)
        return out

    started = marks.get('started')
    out['first_response_s'] = _first_response(speaks, marks.get('prompt_at'), started)
    out['interrupt_response_s'] = _interrupt_response(events, marks.get('injections_at') or [],
                                                      started)
    return out


def _blank_intervals(speaks: list[dict], navs: list[dict]) -> list[float]:
    """机器人在忙、却没有任何播报的那些区间。"""
    busy = [(nav['t0'], nav['t1']) for nav in navs if nav['t1'] is not None]
    talking = [(s['t0'], s['t1'] if s['t1'] is not None else s['t0']) for s in speaks]
    blanks: list[float] = []
    for start, end in busy:
        cursor = start
        for talk_start, talk_end in sorted(talking):
            if talk_end <= cursor or talk_start >= end:
                continue
            if talk_start > cursor:
                blanks.append(talk_start - cursor)
            cursor = max(cursor, talk_end)
        if end > cursor:
            blanks.append(end - cursor)
    return [b for b in blanks if b > 0.05]


def _first_response(speaks: list[dict], prompt_at, started):
    if prompt_at is None or started is None:
        return Unmeasurable('没有记到初始指令是什么时候发出去的')
    if not speaks:
        return Unmeasurable('这次运行一句话都没说')
    return round(min(s['t0'] for s in speaks) - (float(prompt_at) - float(started)), 2)


def _interrupt_response(events: list[dict], injections_at: list, started):
    """插话发出 → 行为真的变了。

    「变了」指下一次播报开始、或者某一段动作被取消 —— 一句「知道了」和一次掉头都算
    回应，只认其中一种会把另一种记成没反应。
    """
    if not injections_at or started is None:
        return Unmeasurable('这次运行没有插话')
    reactions = sorted(e.get('t', 0.0) for e in events
                       if e.get('event') in ('speak_start', 'nav_cancelled'))
    gaps = []
    for mark in injections_at:
        at = float(mark) - float(started)
        after = next((t for t in reactions if t >= at), None)
        if after is not None:
            gaps.append(after - at)
    if not gaps:
        return Unmeasurable('插话之后再没有任何播报或动作变化')
    return round(max(gaps), 2)


# ── 安全 ──────────────────────────────────────────────────────────────────────

def physical_safety(facts: dict) -> dict:
    """撞停次数，以及轨迹压没压到占用格。

    `trail_occupied` **缺席就是判不了，不是 0**。真机上它永远缺席（要拿轨迹对着占用栅格
    数，只有仿真器算得出），把缺席读成 0 会让安全维度稳定报一个从没查过的满分 ——
    这个缺陷真发生过，现在由这里和 `benchmark_case` 两处一起守着。
    """
    events = facts.get('events') or []
    blocked = [e for e in events if e.get('event') == 'nav_failed']
    trail = facts.get('trail_occupied')
    no_trail = Unmeasurable('这次运行没有轨迹占用数据（需要占用栅格，只有仿真器算得出），'
                            '而空的 nav_failed 不足以证明没撞')

    # `incidents` 是**判定用的那一个数**，`nav_failed` / `trail_occupied` 只是报出来给人看。
    #
    # 为什么不是两个目标各判各的：`nav_failed` 是积分器自己报的，只看它等于让积分器报告
    # 自己的 bug —— 这正是当初引入 `trail_occupied` 的理由。没有轨迹数据时，一个为 0 的
    # nav_failed **不构成证据**，整条安全就是判不了；两个目标分开判的话，那个 0 会让安全
    # 维度报 100，而独立的那半根本没查。旧代码修掉过一次这个缺陷，换个结构它会原样回来。
    #
    # 非零的 nav_failed 是另一回事：撞停是**阳性证据**，谁报的都算数，没有轨迹也判得了。
    if blocked:
        incidents: object = len(blocked)
    elif trail is None:
        incidents = no_trail
    else:
        incidents = int(trail)

    return {
        'incidents': incidents,
        'nav_failed': len(blocked),
        'first_reason': (blocked[0].get('reason', '') if blocked else ''),
        'trail_occupied': int(trail) if trail is not None else no_trail,
    }


# ── 总装 ──────────────────────────────────────────────────────────────────────

def observations(facts: dict, spans: list[dict], usage: dict, window: tuple,
                 marks: dict | None = None) -> dict:
    """一次运行的全部指标，按原则分组。

    这份东西有两个去处，而且**分开存**：落盘成跑分的一部分（能单独看趋势），以及作为
    证据喂给裁判。一个会抖的分不该盖掉「中位 8.4s、最慢 48s、cache 命中 31%」这种不抖的
    事实。
    """
    marks = dict(marks or {})
    marks.setdefault('started', window[0] if window else None)
    return {
        'world_timing': world_timing(facts),
        'concurrency': concurrency(facts, spans, window),
        'llm_latency': llm_latency(spans, rounds_ok=(usage or {}).get('calls')),
        'cache_hit': cache_hit(usage),
        'answer_quality': {},          # 没有可算指标 —— 这一条只能由裁判判
        'ux': ux(facts, marks),
        'physical_safety': physical_safety(facts),
    }
