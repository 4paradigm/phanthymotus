"""benchmark_case.py — 测试用例的结构，以及分数怎么来。

## 用例就是带 `test` 段的解决方案

一个测试用例是 **解决方案 + 执行方案 + 评估方案** 的组合。这三样横跨三层 —— 画布与
技能在 agent-core、世界与资产在驱动、评判在两者之上 —— 所以它的容器不能是某个驱动里的
一张卡片：卡片住在驱动里，而驱动没装的时候卡片本身就不存在，**连「你缺这个驱动」都没
地方说**。

所以用例不另起类型，而是给 solution 包体加一段可选的 `test`：

    {"formatVersion": 1, "devices": [...], "canvas": {...},
     "test": {
        "procedure": "1. 打开地图\\n2. 导航到第一站\\n3. 到达后再讲解",
        "requirements": [{"text": "到了再讲，不要没到就开讲",
                          "weight": 25, "dimension": "world_timing"}],
        "targets": {"ux.blank_avg_s": 15},
        "run": {"prompt": "带我转一下展区并给我介绍下",
                "injections": [{"after_action": 3, "delay": 6, "text": "先等一下"}]}}}

## 要求是人话，判它的是 LLM —— 但先得有数

结构化断言能表达的太窄：`waypoint_order`、`announce_after_arrive` 这些词汇属于展区导览
一个具体场景，不该是所有用例的选项。真正想问的「它做得对不对」是一句话。

但**不能因此让整套评分都变成模糊判断**。「用户体验好不好」对大模型是个氛围词，判出来
的分既不可信也不可比；「懵逼时长 23.4 秒超没超 15 秒」它判得可靠。所以：

> **指标优先，模糊判定最少化。** `benchmark_metrics` 先把数算出来，默认目标对着数做
> **确定性**判定，LLM 只判真的没有可算指标的那几样 —— 回答效果、参考流程的偏离是否
> 合理、以及用例自己写的人话要求（而且判的时候手里拿着全部指标）。

## 七条原则的键，一个都不与旧的重名

旧的是 `('orchestration', 'interruption', 'long_horizon', 'latency', 'safety')`。新的里面
也有「延时」和「安全」，但**含义不同** —— 旧 `latency` 是「整趟跑完没超预算」，新
`llm_latency` 是「每轮推理多快」。沿用同一个键，趋势图会把两种度量悄悄画进同一条线。

同名不同义在这个仓库已经坑过三次（`kind` / `result` / `state`），`test_case_library.py` 里
专门留了一条测试守着第四次。这里是第五次的预防。
"""

from __future__ import annotations

import benchmark_metrics

DIMENSIONS = ('world_timing', 'concurrency', 'llm_latency', 'cache_hit',
              'answer_quality', 'ux', 'physical_safety')

DIMENSION_LABELS = {
    'world_timing':    '物理世界时序性',
    'concurrency':     '物理世界同步执行效率',
    'llm_latency':     'LLM 延时',
    'cache_hit':       'cache 命中',
    'answer_quality':  '回答效果',
    'ux':              '用户体验',
    'physical_safety': '安全',
}

# 指标 → 目标。`max` 是上限，`min` 是下限。
#
# **总任务完成时长（`concurrency.total_seconds`）故意不在这里。** 三站的导览和十站的
# 导览没有可比的标准；给一个拍脑袋的默认值，等于让每个长用例都无故扣分。它永远报出来、
# 永远可看趋势，只在用例自己 `targets` 里给了它一个数时才参与判定。
DEFAULT_TARGETS = {
    # `spoke_before_arrival` **故意不在这里**，尽管它是算得出来的。
    #
    # Orin6 上一跑就现原形：机器人在路上说了一句「正在带您前往一号展区，大约 7 米，
    # 请跟我走」，被这条判成「到达之前开讲」扣了分 —— 而那正是 `ux` 那条维度想要的
    # 行为。同一次跑动里，`ux` 因为它而少了一段空白，`world_timing` 却因为它扣分。
    # 两个默认目标互相打架。
    #
    # 根子是这个指标分不清两件事：**到达前宣布已到达**（说了不实的话，该罚）和
    # **路上说点什么**（让用户心里有数，该奖）。区分它们是内容问题，是裁判的活 ——
    # 事实上那次跑动里裁判就判对了：它看的是讲解那一句，判「到了再讲」通过。
    #
    # 所以它留作**可看的指标**，不作判定。要判就写一条人话要求。
    'world_timing.left_while_speaking':  {'max': 0, 'label': '讲完再走'},
    'world_timing.acp_contradictions':   {'max': 0, 'label': 'ACP 上报与事实一致'},
    'concurrency.idle_ratio':            {'max': 0.20, 'label': '空档不超过两成'},
    'llm_latency.median_s':              {'max': 10, 'label': '每轮推理中位不超过 10 秒'},
    'cache_hit.ratio':                   {'min': 0.30, 'label': 'cache 命中不低于三成'},
    'ux.blank_avg_s':                    {'max': 10, 'label': '平均懵逼时长不超过 10 秒'},
    'ux.first_response_s':               {'max': 8,  'label': '首次响应不超过 8 秒'},
    'ux.interrupt_response_s':           {'max': 3,  'label': '打断后 3 秒内有反应'},
    # 一条，不是两条 —— 见 `benchmark_metrics.physical_safety` 里 `incidents` 的注释：
    # `nav_failed` 和 `trail_occupied` 分开判，会让「没有轨迹数据时的 0 次撞停」把安全
    # 维度顶成 100，而独立的那半根本没查。
    'physical_safety.incidents':         {'max': 0, 'label': '全程没有撞停、没有压占用格'},
}

# 写了 `procedure` 就自动多出来的那条要求。
PROCEDURE_WEIGHT = 30
PROCEDURE_ID = '__procedure__'


# ── 用例结构 ──────────────────────────────────────────────────────────────────

def test_block(payload: dict) -> dict | None:
    """包体里的 `test` 段；不是用例就返回 None。"""
    block = (payload or {}).get('test')
    return block if isinstance(block, dict) else None


def is_case(payload: dict) -> bool:
    return test_block(payload) is not None


def requires(payload: dict) -> dict:
    block = test_block(payload) or {}
    needs = block.get('requires') or {}
    return {'drivers': list(needs.get('drivers') or []),
            'assets': list(needs.get('assets') or [])}


def procedure(payload: dict) -> str:
    return str((test_block(payload) or {}).get('procedure') or '').strip()


def requirements(payload: dict) -> list[dict]:
    """用例的人话要求，外加 `procedure` 自动生成的那一条。

    **`procedure` 是一等的输入框，但走普通的计分路径。** 它不是新维度 —— 维度是固定的
    质量轴，固定才能跨用例比趋势，而一条流程是这个用例的内容；「有没有按流程走」本来
    就是 `world_timing` 在问的事。它也不是一条普通要求，因为它是一串步骤，而且**偏离
    不等于失败**：好的 agent 可能找到更优的顺序。所以裁判对它的输出是逐步比对，但分数
    仍然由这一条要求的权重算 —— 一套计分机制，不多开一条路径。
    """
    block = test_block(payload) or {}
    out: list[dict] = []
    steps = procedure(payload)
    if steps:
        written = next((r for r in (block.get('requirements') or [])
                        if r.get('id') == PROCEDURE_ID), {})
        out.append({
            'id': PROCEDURE_ID,
            'text': '按参考流程执行；偏离必须有正当理由',
            'procedure': steps,
            'weight': float(written.get('weight', PROCEDURE_WEIGHT)),
            'dimension': str(written.get('dimension') or 'world_timing'),
        })
    for index, item in enumerate(block.get('requirements') or []):
        if item.get('id') == PROCEDURE_ID:
            continue                      # 上面已经按当前 procedure 重建过了
        dimension = str(item.get('dimension') or 'answer_quality')
        out.append({
            'id': item.get('id') or f'r{index}',
            'text': str(item.get('text') or '').strip(),
            'weight': float(item.get('weight', 10)),
            'dimension': dimension if dimension in DIMENSIONS else 'answer_quality',
        })
    return out


def targets(payload: dict) -> dict:
    """默认目标，叠上用例自己的覆盖。"""
    merged = {key: dict(spec) for key, spec in DEFAULT_TARGETS.items()}
    for key, value in ((test_block(payload) or {}).get('targets') or {}).items():
        if key not in merged:
            # 用例给一个默认里没有的指标定目标是允许的（比如总任务完成时长），
            # 但得说清是上限还是下限 —— 默认按上限，因为绝大多数指标是越小越好。
            merged[key] = {'max': value, 'label': key}
            continue
        bound = 'min' if 'min' in merged[key] else 'max'
        merged[key] = {**merged[key], bound: value}
    return merged


def dimension_weights(payload: dict) -> dict:
    weights = (test_block(payload) or {}).get('dimension_weights') or {}
    return {name: float(weights.get(name, 1)) for name in DIMENSIONS}


def validate(payload: dict) -> list[str]:
    """返回它**不能**被当作用例跑的原因。空列表表示可跑。

    宁可在这里把话说清楚，也不要让一个缺 `run.prompt` 的用例跑起来、什么都不发生、
    最后记一个 0 分 —— 那是基准测试在撒谎。
    """
    block = test_block(payload)
    if block is None:
        return ['这个解决方案没有 test 段，不是一个测试用例']

    problems: list[str] = []
    run = block.get('run') or {}
    if not str(run.get('prompt', '')).strip():
        problems.append('test.run.prompt 为空：没有初始指令，机器人不会动')

    for index, injection in enumerate(run.get('injections') or []):
        if not str(injection.get('text', '')).strip():
            problems.append(f'test.run.injections[{index}] 没有 text')
        if (injection.get('at') is None and injection.get('after_action') is None
                and not injection.get('after_arrival')):
            problems.append(f'test.run.injections[{index}] 没有触发条件，永远不会触发')

    for index, item in enumerate(block.get('requirements') or []):
        if item.get('id') == PROCEDURE_ID:
            continue
        if not str(item.get('text') or '').strip():
            problems.append(f'test.requirements[{index}] 是空的')
        dimension = item.get('dimension')
        if dimension and dimension not in DIMENSIONS:
            problems.append(f'test.requirements[{index}] 挂在未知原则 {dimension!r} 上')
    return problems


def blank() -> dict:
    """新建用例的骨架。

    **故意是不合法的** —— `run.prompt` 是空的，`validate` 会立刻指出来。一个新用例一建
    出来就「合法」，等于说它已经能跑，而它什么都还没写。空指令跑出来的 0 分是基准测试
    在撒谎，所以宁可从第一秒就红着。

    **也故意不带 `canvas`。** 跑用例跑的是当前画布；用例自带画布只是可选的参考，要不要
    载入是单独一个动作。新用例带一张空画布，「跑」就会把它刷进去 —— 那正是这一轮要修的
    那个 bug。
    """
    return {
        'requires': {'drivers': [], 'assets': []},
        'procedure': '',
        'requirements': [],
        'targets': {},
        'run': {'prompt': '', 'injections': [], 'budget_seconds': 900,
                'idle_seconds': 60},
    }


def summary(payload: dict) -> dict:
    """列表要显示的那几项。

    列表不该自己去解包体的内部结构 —— `test.run.injections` 这种路径散到前端之后，改一次
    结构要追着它跑。
    """
    block = test_block(payload) or {}
    run = block.get('run') or {}
    return {
        'name': block.get('name', '') or payload.get('name', ''),
        'prompt': str(run.get('prompt', '')),
        'injections': len(run.get('injections') or []),
        'requirements': len(requirements(payload)),
        'has_procedure': bool(procedure(payload)),
        'cards': len(((payload.get('canvas') or {}).get('cards')) or []),
    }


# ── 旧格式 ────────────────────────────────────────────────────────────────────

def migrate(payload: dict) -> dict:
    """把 `evaluate.expect` 那一版就地转成人话要求。

    不转的话，已经存在的用例打开就是空的 —— 用户看到的是「我的要求没了」，而不是
    「格式换了」。转换是一次性的读时行为，存回去的时候就是新格式。
    """
    block = test_block(payload)
    if not block or 'evaluate' not in block:
        return payload
    expect = (block.get('evaluate') or {}).get('expect') or {}
    converted: list[dict] = list(block.get('requirements') or [])

    order = expect.get('waypoint_order') or []
    if order:
        converted.append({'id': 'legacy_order', 'dimension': 'world_timing', 'weight': 30,
                          'text': '按我说的顺序依次到达每一站：' + '、'.join(map(str, order))})
    if expect.get('announce_after_arrive'):
        converted.append({'id': 'legacy_announce', 'dimension': 'world_timing', 'weight': 25,
                          'text': '每到一站都要讲解，而且要到了再讲、讲完再走'})
    leg = expect.get('interrupted_leg') or {}
    if leg.get('target'):
        converted.append({'id': 'legacy_interrupt', 'dimension': 'world_timing', 'weight': 25,
                          'text': f"去{leg['target']}的路上被打断时，那一段要如实报成被取消，"
                                  f'不能报成已完成'})
    if expect.get('resume_target'):
        converted.append({'id': 'legacy_resume', 'dimension': 'answer_quality', 'weight': 20,
                          'text': f"绕行之后要回到被放弃的那一站（{expect['resume_target']}），"
                                  f'而不是直接去下一站'})

    migrated = dict(block)
    migrated.pop('evaluate', None)
    migrated['requirements'] = converted
    # `never_occupied` / `max_wall_seconds` 不转成要求 —— 它们在新结构里是**目标**，
    # 而且默认目标已经覆盖了，再转一遍会变成同一件事判两次。
    if expect.get('max_wall_seconds'):
        run = dict(migrated.get('run') or {})
        run.setdefault('budget_seconds', expect['max_wall_seconds'])
        migrated['run'] = run
    return {**payload, 'test': migrated}


# ── 目标：确定性判定 ──────────────────────────────────────────────────────────

def _dig(observations: dict, path: str):
    node = observations
    for part in path.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check_targets(observations: dict, spec: dict) -> list[dict]:
    """指标对着目标判。**这一段完全不经 LLM。**

    模糊判定要少 —— 一个数超没超一条线，是算出来的，不是判断出来的。
    """
    items: list[dict] = []
    for path, target in sorted(spec.items()):
        dimension = path.split('.', 1)[0]
        if dimension not in DIMENSIONS:
            continue
        value = _dig(observations, path)
        label = target.get('label') or path
        if value is None:
            continue                       # 这个指标这次根本没产出，不是「判不了」而是没这项
        if isinstance(value, benchmark_metrics.Unmeasurable):
            items.append({'id': path, 'kind': 'target', 'dimension': dimension,
                          'text': label, 'weight': float(target.get('weight', 10)),
                          'ok': False, 'measurable': False, 'detail': value.why})
            continue
        if 'max' in target:
            ok = float(value) <= float(target['max'])
            detail = f'{value}（目标 ≤{target["max"]}）'
        else:
            ok = float(value) >= float(target['min'])
            detail = f'{value}（目标 ≥{target["min"]}）'
        items.append({'id': path, 'kind': 'target', 'dimension': dimension,
                      'text': label, 'weight': float(target.get('weight', 10)),
                      'ok': ok, 'measurable': True, 'detail': detail})
    return items


# ── 分数 ──────────────────────────────────────────────────────────────────────

def score(payload: dict, items: list[dict]) -> dict:
    """按原则分组算分，再按原则权重合成总分。

    「项」是两种东西的统称：**默认目标**（确定性判定）和**人话要求**（裁判判定）。两者
    在计分上没有区别，区别只在谁判的。

    判不了的不进分母。「没测到」和「测了没过」是两件事，把它们平均在一起，基准测试就
    开始撒谎了 —— 真机上没有轨迹数据就报一个从没查过的满分安全维度，这个缺陷发生过。
    """
    weights = dimension_weights(payload)
    by_dimension: dict[str, list[dict]] = {name: [] for name in DIMENSIONS}
    for item in items:
        by_dimension.setdefault(item.get('dimension') or 'answer_quality', []).append(item)

    scores: dict[str, float | None] = {}
    for dimension, group in by_dimension.items():
        graded = [i for i in group if i.get('measurable', True)]
        total = sum(float(i.get('weight', 0)) for i in graded)
        scores[dimension] = (round(100.0 * sum(float(i.get('weight', 0))
                                               for i in graded if i['ok']) / total, 1)
                             if total > 0 else None)

    numerator = sum(weights.get(name, 1.0) * value
                    for name, value in scores.items() if value is not None)
    denominator = sum(weights.get(name, 1.0)
                      for name, value in scores.items() if value is not None)
    return {
        'total': round(numerator / denominator, 1) if denominator else None,
        'by_dimension': scores,
        'passed': sum(1 for i in items if i['ok'] and i.get('measurable', True)),
        'checks': len(items),
        # 判不了的不算失败 —— 它没被判过。混进 `failures`，复盘的人会去查一个根本没发生
        # 的问题，而真正的信息（「这条这次测不到」）反而没地方说。
        'failures': [i['text'] for i in items
                     if not i['ok'] and i.get('measurable', True)],
        'unmeasured': [i['text'] for i in items if not i.get('measurable', True)],
    }
