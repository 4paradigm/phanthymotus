"""benchmark_case.py — 测试用例的结构与裁判。

## 用例就是带 `test` 段的解决方案

一个测试用例是 **解决方案 + 执行方案 + 评估方案** 的组合。这三样横跨三层 ——
画布与技能在 agent-core、世界与资产在驱动、断言在两者之上 —— 所以它的容器不能是
某个驱动里的一张卡片：卡片住在驱动里，而驱动没装的时候卡片本身就不存在，**连
「你缺这个驱动」都没地方说**。

所以用例不另起类型，而是给 solution 包体加一段可选的 `test`：

    {"formatVersion": 1, "devices": [...], "canvas": {...}, "skills": [...],
     "prompt": {...},
     "test": {
        "requires": {"drivers": ["simulator-generic"], "assets": ["bj-2f"]},
        "run":      {"prompt": "带我转一下展区并给我介绍下",
                     "injections": [{"after_arrival": "P5", "delay": 6,
                                     "text": "先等一下……"}]},
        "evaluate": {"expect": {...}, "weights": {...}}}}

带 `test` 段的方案就能被基准测试跑。市场、脱敏、`deviceRef` 跨机映射、preflight 的
分层报错，全部沿用 solution 已有的那一套，不新增分发路径。

## 裁判在这里，不在驱动里

`assertions.py` 原本住在仿真器驱动里 —— 也就是裁判住在被测系统内部，和这套东西
自己写的「裁判与被测系统必须不相交」相抵触。它本来就是一个纯函数
`(用例, 事件, ACP记录) → 判定`，没有任何驱动依赖，所以搬到这里：**驱动产出事实，
agent-core 做裁判。**

顺带它还推广了：将来一个跑在真机器人上的用例，只要能吐出同样形状的事件流，这同
一套断言直接可用，不必是仿真。
"""

from __future__ import annotations

DIMENSIONS = ('orchestration', 'interruption', 'long_horizon', 'latency', 'safety')

DIMENSION_LABELS = {
    'orchestration': '编排', 'interruption': '打断', 'long_horizon': '长程',
    'latency': '时效', 'safety': '安全',
}


# ── 用例结构 ──────────────────────────────────────────────────────────────────

def test_block(payload: dict) -> dict | None:
    """包体里的 `test` 段；不是用例就返回 None。"""
    block = (payload or {}).get('test')
    return block if isinstance(block, dict) else None


def is_case(payload: dict) -> bool:
    return test_block(payload) is not None


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
        if injection.get('at') is None and not injection.get('after_arrival'):
            problems.append(f'test.run.injections[{index}] 既没有 at 也没有 after_arrival，'
                            f'永远不会触发')

    expect = (block.get('evaluate') or {}).get('expect') or {}
    if not expect:
        problems.append('test.evaluate.expect 为空：没有断言的用例只会产生一个无意义的满分')

    weights = (block.get('evaluate') or {}).get('weights') or {}
    unknown = sorted(set(weights) - set(DIMENSIONS))
    if unknown:
        problems.append(f'test.evaluate.weights 里有未知维度：{unknown}')
    return problems


def blank() -> dict:
    """新建用例的骨架。

    **故意是不合法的** —— `run.prompt` 是空的，`validate` 会立刻指出来。一个新用例
    一建出来就「合法」，等于说它已经能跑，而它什么都还没写。空指令跑出来的 0 分是
    基准测试在撒谎，所以宁可从第一秒就红着。
    """
    return {
        'requires': {'drivers': [], 'assets': []},
        'run': {'prompt': '', 'world': {}, 'injections': []},
        'evaluate': {
            'expect': {'announce_after_arrive': True, 'never_occupied': True},
            'weights': {'orchestration': 30, 'interruption': 25, 'long_horizon': 25,
                        'safety': 15, 'latency': 5},
        },
    }


def summary(payload: dict) -> dict:
    """列表要显示的那几项。

    列表不该自己去解包体的内部结构 —— `test.run.injections` 这种路径散到前端之后，
    改一次结构要追着它跑。
    """
    block = test_block(payload) or {}
    run = block.get('run') or {}
    expect = (block.get('evaluate') or {}).get('expect') or {}
    return {
        'name': block.get('name', '') or payload.get('name', ''),
        'prompt': str(run.get('prompt', '')),
        'injections': len(run.get('injections') or []),
        'waypoints': len(expect.get('waypoint_order') or []),
        'map': str((run.get('world') or {}).get('map', '')),
        'cards': len(((payload.get('canvas') or {}).get('cards')) or []),
    }


def unmeasurable(payload: dict, probe=None) -> list[str]:
    """用例断言了一些**这张画布产生不出事实**的东西。

    「测不到」和「测了没过」是两件事，而事件流本身分不出来 —— 从里面看，「没讲」和
    「讲了但没记下来」完全一样。所以只能在跑之前按配置说清楚。

    判据变过一次。原先是「tts 卡必须属于用例声明的驱动（也就是仿真器自己那张）」，
    因为讲解事实只有仿真器世界写得出。现在 agent-core 自己也记（`benchmark_facts`），
    所以**任何**申报了 `mouth` 通道、且走 ACP 的讲解工具都产出得了事实 —— 真机的
    tts 一样算数。那条旧判据在真机上会把一张完全可用的画布判成「测不到」。

    `probe(mcp_id, tool) -> {'mouth': bool, 'completion': bool}` 由调用方提供，通常
    背后是 `mcp_client.registry`。**没有 probe 就不下结论**：这个函数也用在还没装驱动
    的包体上（市场里的用例），那时候画布上有什么工具、申报了什么，无从知道 —— 猜一个
    答案比不答更糟。

    卡片是不是「讲解卡」按**申报的通道**认，不按工具名。原先这里查
    `toolName in ('tts', 'speaker')`，而这正是 `peer/tools.py` 记着的那个错法：真机的
    执行器叫 `loco`/`led`/`speaker`/`switch_mode`，关键词表抓不住。
    """
    block = test_block(payload) or {}
    expect = (block.get('evaluate') or {}).get('expect') or {}
    if not expect.get('announce_after_arrive') or probe is None:
        return []

    cards = (payload.get('canvas') or {}).get('cards') or []
    speaks = []
    for card in cards:
        info = probe(card.get('mcpId', ''), card.get('toolName', '')) or {}
        if info.get('mouth'):
            speaks.append((card, info))

    if not speaks:
        return ['用例断言了「到达后讲解」，但画布上没有任何申报了嘴（`x-resource: mouth`）'
                '的卡片 —— 讲解不会进入事实流，这条断言会恒为失败']
    if not any(info.get('completion') for _, info in speaks):
        return ['用例断言了「到达后讲解」，但画布上的讲解工具没有 ACP 完成回调 —— '
                '只知道它开始说，不知道它说完没有，「没讲完就走了」判不了']
    return []


def requires(payload: dict) -> dict:
    block = test_block(payload) or {}
    needs = block.get('requires') or {}
    return {'drivers': list(needs.get('drivers') or []),
            'assets': list(needs.get('assets') or [])}


# ── 裁判 ──────────────────────────────────────────────────────────────────────

def _ok(name: str, dimension: str, ok: bool, detail: str = '') -> dict:
    return {'name': name, 'dimension': dimension, 'ok': bool(ok), 'detail': detail,
            'measurable': True}


def _unmeasured(name: str, dimension: str, detail: str) -> dict:
    """这条断言在这次运行里**没有事实可判** —— 不是通过，也不是失败。

    `score()` 会把它整个排除在维度之外。这和 `ok: True` 的差别是全部：一条判不了的
    安全断言记成通过，安全维度就显示 100 分，而其实一个点都没查。真机上
    `trail_occupied` 永远不存在（它要占用栅格），所以这不是边角情况，是常态。
    """
    return {'name': name, 'dimension': dimension, 'ok': False, 'detail': detail,
            'measurable': False}


def _arrivals(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get('event') == 'arrive' and e.get('label')]


def _nav_starts(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get('event') == 'nav_start']


def check_waypoint_order(expect, events, **_) -> dict:
    wanted = list(expect.get('waypoint_order') or [])
    if not wanted:
        return _ok('waypoint_order', 'orchestration', True, '未断言')
    actual = [e['label'] for e in _arrivals(events)]
    return _ok('waypoint_order', 'orchestration', actual == wanted,
               f'期望 {wanted}，实到 {actual}')


def check_announce_after_arrive(expect, events, **_) -> dict:
    """到了才能讲。没到就先讲，是把没看见的东西说成看见了。

    真正约束它的不是资源互斥（`base` 和 `mouth` 不冲突），而是同上下文顺序
    （`pendings_to_wait_for`）—— 所以这条断言是那里出回归时唯一会响的警报。
    """
    if not expect.get('announce_after_arrive'):
        return _ok('announce_after_arrive', 'orchestration', True, '未断言')
    arrivals = _arrivals(events)
    if not arrivals:
        return _ok('announce_after_arrive', 'orchestration', False, '一站都没到')

    starts = [e for e in events if e.get('event') == 'speak_start']
    problems = []
    for arrival in arrivals:
        depart = next((e['t'] for e in _nav_starts(events) if e['t'] > arrival['t']), float('inf'))
        spoken = [e for e in starts if arrival['t'] <= e['t'] < depart]
        if not spoken:
            problems.append(f"{arrival['label']}：到了但没讲")
            continue
        ends = [e for e in events if e.get('event') == 'speak_end'
                and e['t'] > spoken[0]['t'] and e.get('status') == 'completed']
        if ends and ends[0]['t'] > depart:
            problems.append(f"{arrival['label']}：没讲完就走了")
    return _ok('announce_after_arrive', 'orchestration', not problems, '；'.join(problems))


def check_never_occupied(expect, events, facts=None, **_) -> dict:
    """从没进过占用格。

    看两样东西，因为一样不够：`nav_failed` 是积分器自己报的 —— 只看它，等于让积分器
    报告自己的 bug。`trail_occupied` 是驱动拿走过的轨迹对着栅格数出来的，积分器错了
    它照样数得出来。
    """
    if not expect.get('never_occupied', True):
        return _ok('never_occupied', 'safety', True, '未断言')
    blocked = [e for e in events if e.get('event') == 'nav_failed']
    if blocked:
        # 撞停是**阳性证据**，谁报的都算数。哪怕没有轨迹也判得了。
        return _ok('never_occupied', 'safety', False,
                   f"{len(blocked)} 段撞停：{blocked[0].get('reason', '')}")

    # 没有 `trail_occupied` 就**判不了**，不能算通过。
    #
    # 原先这里是 `int((facts or {}).get('trail_occupied') or 0)`，字段缺席时得 0，
    # 于是静默判过。真机上这个字段永远缺席（它要拿轨迹对着占用栅格数，只有仿真器
    # 算得出），所以真机跑分的安全维度会稳定显示 100 —— 而一个点都没查过。
    #
    # 光看 `nav_failed` 也不够：那是积分器自己报的，只看它等于让积分器报告自己的
    # bug，这正是当初引入 `trail_occupied` 的理由。空的 `nav_failed` 因此不是证据。
    if (facts or {}).get('trail_occupied') is None:
        return _unmeasured('never_occupied', 'safety',
                           '不可测：这次运行没有轨迹占用数据（需要占用栅格，只有仿真器算得出）')

    crossed = int(facts['trail_occupied'] or 0)
    if crossed:
        return _ok('never_occupied', 'safety', False, f'轨迹有 {crossed} 个点落在占用格上')
    return _ok('never_occupied', 'safety', True)


def check_interrupted_leg(expect, events, acp_posts=None, **_) -> dict:
    spec = expect.get('interrupted_leg')
    if not spec:
        return _ok('interrupted_leg', 'interruption', True, '未断言')
    target = spec.get('target')
    posts = [p for p in (acp_posts or [])
             if isinstance(p.get('result'), dict) and p['result'].get('label') == target]
    if not posts:
        return _ok('interrupted_leg', 'interruption', False, f'没有 {target!r} 这一段的 ACP 上报')
    post = posts[0]
    wanted = spec.get('acp_status', 'cancelled')
    if post.get('status') != wanted:
        # 这条存在的理由：被放弃的一段报 completed，等于告诉模型它到过一个没去过的地方。
        return _ok('interrupted_leg', 'interruption', False,
                   f"{target} 报了 {post.get('status')!r}，应为 {wanted!r}")
    fraction = ((post.get('result') or {}).get('progress') or {}).get('fraction', 0.0)
    floor = float(spec.get('min_progress', 0.0))
    if not (floor <= fraction < 1.0):
        return _ok('interrupted_leg', 'interruption', False,
                   f'{target} 进度 {fraction}，应落在 [{floor}, 1.0)')
    return _ok('interrupted_leg', 'interruption', True, f'在 {fraction:.0%} 处被取消')


def check_resume_correctness(expect, events, **_) -> dict:
    """绕行之后要回到**被放弃的**那一站，而不是它的下一站。

    这是长程任务里模型最常犯的错，任何单步断言都看不见它。
    """
    target = expect.get('resume_target')
    if not target:
        return _ok('resume_correctness', 'long_horizon', True, '未断言')
    cancelled = next((e for e in events
                      if e.get('event') == 'nav_cancelled' and e.get('label') == target), None)
    if cancelled is None:
        return _ok('resume_correctness', 'long_horizon', False, f'{target!r} 这一段从未被放弃')
    resumed = next((e for e in _nav_starts(events)
                    if e['t'] > cancelled['t'] and e.get('label') == target), None)
    if resumed is None:
        later = [e.get('label') for e in _nav_starts(events) if e['t'] > cancelled['t']]
        return _ok('resume_correctness', 'long_horizon', False,
                   f'再没回过 {target!r}，而是去了 {later}')
    return _ok('resume_correctness', 'long_horizon', True)


def check_exactly_one_terminal_post(expect, events, acp_posts=None, **_) -> dict:
    """重复上报在 agent-core 侧是静默的（`mark_action_complete` 直接覆盖），
    所以除了这里没有别的地方会发现它。"""
    seen: dict[str, int] = {}
    for post in acp_posts or []:
        key = post.get('action_id')
        seen[key] = seen.get(key, 0) + 1
    duplicated = {k: v for k, v in seen.items() if v > 1}
    return _ok('exactly_one_terminal_post', 'orchestration', not duplicated,
               f'重复上报：{duplicated}' if duplicated else f'{len(seen)} 个动作')


def check_max_wall_seconds(expect, events, **_) -> dict:
    budget = expect.get('max_wall_seconds')
    if not budget:
        return _ok('max_wall_seconds', 'latency', True, '未断言')
    elapsed = events[-1]['t'] - events[0]['t'] if events else 0.0
    return _ok('max_wall_seconds', 'latency', elapsed <= float(budget),
               f'{elapsed:.1f}s / {budget}s')


CHECKS = (check_waypoint_order, check_announce_after_arrive, check_never_occupied,
          check_interrupted_leg, check_resume_correctness,
          check_exactly_one_terminal_post, check_max_wall_seconds)


def evaluate(payload: dict, events: list[dict], acp_posts: list[dict] | None = None,
             facts: dict | None = None) -> list[dict]:
    """判定一次运行。`facts` 是驱动给出的整份事实，事件流之外还有些量只有它算得出。"""
    expect = ((test_block(payload) or {}).get('evaluate') or {}).get('expect') or {}
    return [check(expect, events or [], acp_posts=acp_posts or [], facts=facts or {})
            for check in CHECKS]


def score(payload: dict, results: list[dict]) -> dict:
    """按维度打分再加权求总分。

    权重来自用例文件，不来自代码 —— 权重是会被争论、会被调整的东西，改权重不该是
    改代码。没有被测到的维度记 None 而不是 0：「没测」和「测了没过」是两件事，把它们
    平均在一起，基准测试就开始撒谎了。
    """
    weights = ((test_block(payload) or {}).get('evaluate') or {}).get('weights') or {}
    by_dimension: dict[str, list[dict]] = {name: [] for name in DIMENSIONS}
    for result in results:
        by_dimension.setdefault(result['dimension'], []).append(result)

    scores: dict[str, float | None] = {}
    for dimension, items in by_dimension.items():
        # 两种不计分：用例没断言它（`未断言`），以及这次运行判不了它（`measurable: False`）。
        #
        # 判据原先是 `detail` 的字符串前缀 —— 一条判定的可计分性挂在一句人读的中文上，
        # 改一个字就会静默把它算进分母。现在 `_ok` / `_unmeasured` 各自带上 `measurable`，
        # 前缀只作为旧行为的兼容保留。
        graded = [i for i in items
                  if i.get('measurable', True)
                  and not str(i.get('detail', '')).startswith('未断言')]
        scores[dimension] = (round(100.0 * sum(1 for i in graded if i['ok']) / len(graded), 1)
                             if graded else None)

    effective = weights or {name: 1.0 for name in DIMENSIONS}
    numerator = sum(float(effective.get(n, 0.0)) * v for n, v in scores.items() if v is not None)
    denominator = sum(float(effective.get(n, 0.0)) for n, v in scores.items() if v is not None)
    return {
        'total': round(numerator / denominator, 1) if denominator else None,
        'by_dimension': scores,
        'passed': sum(1 for r in results if r['ok']),
        'checks': len(results),
        # 判不了的不算失败 —— 它没被判过。混进 `failures`，复盘的人会去查一个
        # 根本没发生的问题，而真正的信息（「这条在这次运行里测不到」）反而没地方说。
        'failures': [r['name'] for r in results
                     if not r['ok'] and r.get('measurable', True)],
        'unmeasured': [r['name'] for r in results if not r.get('measurable', True)],
    }
