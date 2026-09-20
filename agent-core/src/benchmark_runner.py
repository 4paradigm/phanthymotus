"""benchmark_runner.py — 跑一个用例的执行底座。

## 为什么在这里，不在卡片里

跑一个用例要做的事，一半在 agent-core 这一侧：把初始指令当作**用户消息**送进
collector、在合适的时刻追加插话、按用例的权重算分、把分数连同被测配置落盘。仿真器
只管世界 —— 重置、推进、产出事实。

原先这些都在 `sim_scenario` 这张卡片上，于是有两个后果，都不是小事：

1. **被测的 agent 能重置正在测它的那次测量。** 卡片是 LLM 可调用的工具，`reset`
   和 `load` 就在它的 action 枚举里。
2. **裁判住在被测系统内部。** 一条坏掉的 ACP 路径会把自己判成绿的。

## 初始指令走的是用户消息那条路

不新开入口：`event_bus.enqueue(source='message')`。`collector._PRIORITY_SOURCES`
认这个 source，主 agent 才会被唤醒 —— 普通 source 会返回 200 然后只进后台批，看起
来像「发了但没反应」。更重要的是保真：用例要测的就是「用户说了这句话之后会发生
什么」，那它就该从用户说话的那条路进去。

## 一条安全闸

注入的文本和真实指令无法区分。所以开跑之前检查画布：**只要有一张会动的卡片不属于
仿真器，就拒绝。**否则一句「带我转一下展区」会让真机器人走起来 —— 没有人确认过，
也没有人在旁边。分类用 `peer.tools.is_read_only`，它按驱动申报的 type 和 layer 判，
不猜名字（猜名字漏掉过 `loco`/`led`/`speaker`/`switch_mode`）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import benchmark_case
import benchmark_store
import config
import event_bus
import mcp_client

SCENARIO_TOOL = 'sim_scenario'
REPORT_TOOL = 'sim_report'

# 轮询事实的间隔。事实是驱动侧累积的，问快了只是白问。
_POLL_SECONDS = 2.0

# 跑动期间世界的持有者。`sim_scenario` 的 `load`/`reset` 是 LLM 可调用的 action ——
# 没有这把锁，被测的 agent 能重置正在测它的那次测量。
OWNER = 'benchmark'


# ── 安全闸 ────────────────────────────────────────────────────────────────────

def unsafe_cards(simulator_mcp_id: str) -> list[dict]:
    """画布上不属于仿真器、且会动的卡片。

    非空就不能跑：用例的 prompt 会经由主 agent 变成真实的工具调用，而注入的文本
    和一句真实指令在 collector 眼里完全一样。
    """
    from peer.tools import is_read_only

    layout = config.main.get('canvas_layout') or {}
    internal = {m.get('id') for m in (config.main.get('services', {}).get('mcp') or [])
                if m.get('transport') == 'internal'}
    unsafe = []
    for card in layout.get('cards') or []:
        mcp_id = card.get('mcpId', '')
        tool = card.get('toolName', '')
        if not mcp_id or not tool or mcp_id == simulator_mcp_id:
            continue
        # agent-core 自己的伪设备（`decision_core`、`remote_message`、channel）不是
        # 会动的东西 —— 它们是被测的那个系统。`decision_core` 按定义在每张画布上，
        # 把它算进来，任何用例都跑不了，这条闸就只剩一个永远亮着的红灯。
        if mcp_id in internal:
            continue
        if is_read_only(f'mcp__{mcp_id}__{tool}'):
            continue
        entry = mcp_client.registry.get(mcp_id) or {}
        unsafe.append({'mcpId': mcp_id, 'tool': tool,
                       'device': entry.get('name') or entry.get('server_name') or mcp_id})
    return unsafe


# ── 一次跑动 ──────────────────────────────────────────────────────────────────

class CaseRun:
    """一个用例的 N 次重复。同一时刻只允许有一个。"""

    def __init__(self, case: dict, mcp_id: str, repeats: int, seed: int,
                 run_id: str, environment: dict):
        self.case = case
        self.mcp_id = mcp_id
        self.repeats = max(1, int(repeats))
        self.seed = int(seed)
        self.run_id = run_id
        self.environment = environment
        self.state = 'starting'
        self.repeat_idx = 0
        self.cases: list[dict] = []
        self.error = ''
        self._abort = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # -- 与仿真器说话 ---------------------------------------------------------

    async def _call(self, tool: str, args: dict) -> dict:
        result = await mcp_client.call_tool_direct(self.mcp_id, tool, args)
        return result if isinstance(result, dict) else {'error': str(result)}

    async def _facts(self) -> dict:
        """驱动产出的事实：事件流 + ACP 记录。判定不看驱动自己的 assertions。"""
        return await self._call(REPORT_TOOL, {'what': 'report'})

    # -- 执行 -----------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._drive())

    def abort(self) -> None:
        self._abort.set()

    async def _drive(self) -> None:
        try:
            for index in range(self.repeats):
                if self._abort.is_set():
                    break
                self.repeat_idx = index
                self.cases.append(await self._one(index))
            self.state = 'aborted' if self._abort.is_set() else 'done'
        except Exception as exc:                     # noqa: BLE001 — 落盘比崩掉有用
            self.state = 'error'
            self.error = str(exc)
        finally:
            # 把世界还给画布。不还，下一个人连 reset 都调不动，而错误信息会指向一次
            # 早就结束的跑动。
            await self._call(SCENARIO_TOOL, {'action': 'abort', 'owner': OWNER})
            await self._finish()

    async def _one(self, index: int) -> dict:
        run = self.case.get('run') or {}
        started = time.time()

        reset = await self._call(SCENARIO_TOOL, {
            'action': 'reset', 'map': (run.get('world') or {}).get('map', ''),
            'spawn': (run.get('world') or {}).get('spawn') or {},
            'seed': self.seed + index, 'owner': OWNER})
        if 'error' in reset:
            return self._case_row(index, started, {}, error=reset['error'])

        self.state = 'running'
        # 开场那句是**指令**，不是插话 —— 记错了，复盘的人会以为一开始就有人打断。
        await self._say(run.get('prompt', ''), label='指令')

        facts = await self._watch(run, started)
        results = benchmark_case.evaluate({'test': self.case},
                                          facts.get('events') or [],
                                          facts.get('acp_posts') or [], facts=facts)
        score = benchmark_case.score({'test': self.case}, results)
        return self._case_row(index, started, {'facts': facts, 'results': results,
                                               'score': score})

    async def _say(self, text: str, label: str = '插话') -> None:
        """把一句话当作用户说的送进去 —— 用例的保真度就在这里。

        同时在世界的事件日志里留一条。判定用不着它（打断看的是 ACP 上报与
        `nav_cancelled`），但**人**要看：`sim_report` 是事后复盘一次跑动的地方，
        而一条「机器人好好走着突然掉头」的记录里，如果没有那句插话，读的人无从知道
        为什么。说话经由 collector，不经由仿真器，所以不补这一笔它就不在场。
        """
        if not str(text).strip():
            return
        await event_bus.enqueue(source='message', text=str(text),
                                payload={'benchmark': True})
        try:
            await self._call(SCENARIO_TOOL, {'action': 'note', 'text': f'[用例{label}] {text}'})
        except Exception:
            pass      # 记账失败不该让一次跑动停下来

    async def _watch(self, run: dict, started: float) -> dict:
        """边轮询事实边按触发条件追加插话，直到用例结束或超时。"""
        expect = (self.case.get('evaluate') or {}).get('expect') or {}
        budget = float(expect.get('max_wall_seconds') or 900)
        pending = [dict(i) for i in (run.get('injections') or [])]
        armed: dict[int, float] = {}
        facts: dict = {}

        while not self._abort.is_set() and time.time() - started < budget:
            facts = await self._facts()
            events = facts.get('events') or []
            elapsed = time.time() - started

            for index, injection in enumerate(pending):
                if injection.get('done'):
                    continue
                if index not in armed:
                    due = _trigger_due(injection, events, elapsed)
                    if due is None:
                        continue
                    armed[index] = due
                if elapsed >= armed[index]:
                    await self._say(injection.get('text', ''))
                    injection['done'] = True

            # 已经触发、还没发出去的插话必须发完再收尾。世界跑得比 LLM 快的时候
            # （加速倍率、或者一段很短的路），站点会先到齐 —— 就这么收尾的话，打断
            # 根本没发生过，而打断那几条断言会记在 agent 头上。
            outstanding = any(index in armed and not injection.get('done')
                              for index, injection in enumerate(pending))
            if _finished(facts, expect) and not outstanding:
                break
            await asyncio.sleep(_POLL_SECONDS)

        return facts or await self._facts()

    # -- 落盘 -----------------------------------------------------------------

    def _case_row(self, index: int, started: float, payload: dict,
                  error: str = '') -> dict:
        score = payload.get('score') or {}
        results = payload.get('results') or []
        elapsed = time.time() - started
        row = {
            'repeat': index, 'seed': self.seed + index, 'elapsed': elapsed,
            'score': score, 'error': error,
            'failures': [r['name'] for r in results if not r['ok'] and r.get('detail') != '未断言'],
            'results': results,
        }
        benchmark_store.add_case(
            self.run_id, scenario=self.case.get('name', '') or 'case',
            repeat_idx=index, seed=self.seed + index,
            ok=bool(not error and not row['failures']),
            outcome='error' if error else ('ok' if not row['failures'] else 'failed'),
            score=score.get('total'), elapsed_ms=int(elapsed * 1000),
            assertions=row['failures'])
        return row

    async def _finish(self) -> None:
        scored = [c for c in self.cases if (c.get('score') or {}).get('total') is not None]
        totals = [c['score']['total'] for c in scored]
        mean = round(sum(totals) / len(totals), 1) if totals else None
        stdev = _stdev(totals)

        by_dimension: dict = {}
        for case in scored:
            for name, value in (case['score'].get('by_dimension') or {}).items():
                if value is not None:
                    by_dimension.setdefault(name, []).append(value)

        benchmark_store.finish_run(
            self.run_id,
            status={'done': 'done', 'aborted': 'aborted'}.get(self.state, 'error'),
            score_total=mean, score_stdev=stdev,
            scores_by_dim={k: round(sum(v) / len(v), 1) for k, v in by_dimension.items()},
            detail=self.error or f'n={len(self.cases)} scored={len(scored)}')

    def snapshot(self) -> dict:
        return {'state': self.state, 'run_id': self.run_id,
                'repeat': self.repeat_idx, 'repeats': self.repeats,
                'cases': self.cases, 'error': self.error}


# ── 触发与收尾判定 ────────────────────────────────────────────────────────────

def _trigger_due(injection: dict, events: list, elapsed: float) -> Optional[float]:
    """这条插话该在第几秒（跑动开始起算）发出；条件还没满足就返回 None。

    两件事：

    * **`after_arrival` 按事件触发，不按绝对时刻。** 真机上 LLM 一轮 3-48 秒，写死的
      偏移会落到完全不同的一段路上 —— 于是「在去 P6 的路上被打断」测的是别的东西。
    * **`delay` 从**看见**到达那一刻起算，不是从事件自己的时间戳起算。** 事件的 `t`
      是驱动的时钟，加速倍率下和墙钟不是一回事；两个时钟相减出来的秒数没有意义。
    """
    delay = float(injection.get('delay') or 0)
    label = injection.get('after_arrival')
    if label:
        seen = any(e.get('event') == 'arrive' and e.get('label') == label for e in events)
        return elapsed + delay if seen else None
    at = injection.get('at')
    return None if at is None else float(at) + delay


def _finished(facts: dict, expect: dict) -> bool:
    """所有期望到达的站点都到过了就算跑完，不必等满预算。"""
    order = expect.get('waypoint_order') or []
    if not order:
        return False
    arrived = [e.get('label') for e in (facts.get('events') or [])
               if e.get('event') == 'arrive']
    return all(label in arrived for label in order)


def _stdev(values: list) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return round((sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5, 1)


# ── 单例 ──────────────────────────────────────────────────────────────────────

_current: Optional[CaseRun] = None


def current() -> Optional[CaseRun]:
    return _current


def set_current(run: Optional[CaseRun]) -> None:
    global _current
    _current = run


def is_busy() -> bool:
    return _current is not None and _current.state in ('starting', 'running')
