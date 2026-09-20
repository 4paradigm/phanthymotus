"""benchmark_runner.py — 跑一个用例的执行底座。

## 为什么在这里，不在卡片里

跑一个用例要做的事，一半在 agent-core 这一侧：把初始指令当作**用户消息**送进
collector、在合适的时刻追加插话、按用例的权重算分、把分数连同被测配置落盘。世界那一半
交给 `SimulatorWorld` 或 `RealWorld` —— 重置、记账、产出事实。

原先这些都在 `sim_scenario` 这张卡片上，于是有两个后果，都不是小事：

1. **被测的 agent 能重置正在测它的那次测量。** 卡片是 LLM 可调用的工具，`reset`
   和 `load` 就在它的 action 枚举里。
2. **裁判住在被测系统内部。** 一条坏掉的 ACP 路径会把自己判成绿的。

## 初始指令走的是用户消息那条路

不新开入口：`event_bus.enqueue(source='message')`。`collector._PRIORITY_SOURCES`
认这个 source，主 agent 才会被唤醒 —— 普通 source 会返回 200 然后只进后台批，看起
来像「发了但没反应」。更重要的是保真：用例要测的就是「用户说了这句话之后会发生
什么」，那它就该从用户说话的那条路进去。

## 一条安全闸，以及它管不到的地方

注入的文本和真实指令无法区分。所以开始运行之前检查画布，列出会动、又不由仿真器代劳的
卡片 —— 一句「带我转一下展区」会让真机器人真的走起来。分类用 `peer.tools.is_read_only`，
它按驱动申报的 type 和 layer 判，不猜名字（猜名字漏掉过 `loco`/`led`/`speaker`/`switch_mode`）。

这条闸原先是「非空就拒绝」，而那在真机上等于**永远拒绝** —— 真机上每一张执行器卡都不
属于仿真器，于是一个面向解决方案的基准测试在真实机器人上一步也跑不了。现在这份清单交给
现场的人确认（`api/benchmark._check_confirmation`），**确认必须对得上服务端此刻重算的
那一组**：画布在弹窗与开跑之间是可以改的，拿上一次的勾为一组新设备背书是不行的。

**但画布并不能封死 agent 够得到什么。** 画布决定的是 LLM 被**告知**哪些工具
（`llm.py::_get_bound_tool_schemas` 从 execConnections 收集），而 `_dispatch` 直接把
工具名交给 `mcp_client.call_tool`，那里只解析 `mcp__<id>__<tool>` 然后调过去 ——
不核对这个工具在不在画布上。于是一个还留在对话历史里的旧工具名，照样调得通。
Orin6 上就撞见了：画布上的 tts 已经换成仿真器的，模型仍然调了感知栈那只，
因为那个名字还在它的上文里。

所以这条闸拦得住「画布上摆着真设备」，拦不住「历史里留着真设备的名字」。对照之下
peer 那条路**是**核画布绑定的（`canvas_binding.is_bound`）—— 本地 LLM 这条没有。
这是 agent-core 的既有行为，不在这里改；写在这儿是为了不让这条闸被当成它并不是的
那种保证。
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import benchmark_case
import benchmark_facts
import benchmark_store
import config
import event_bus
import mcp_client

SCENARIO_TOOL = 'sim_scenario'
REPORT_TOOL = 'sim_report'

# 轮询事实的间隔。事实是驱动侧累积的，问快了只是白问。
_POLL_SECONDS = 2.0

# 运行期间世界的持有者。`sim_scenario` 的 `load`/`reset` 是 LLM 可调用的 action ——
# 没有这把锁，被测的 agent 能重置正在测它的那次测量。
OWNER = 'benchmark'


# ── 安全闸 ────────────────────────────────────────────────────────────────────

def unsafe_cards(simulator_mcp_id: str | None = None) -> list[dict]:
    """画布上会动、且不由仿真器代劳的卡片 —— 也就是**这次运行会真的动起来的东西**。

    语义变了一次，值得写下来：原先是「非空就拒绝」。那条规则在仿真器在场时是对的，
    放到真机上却等于**永远拒绝** —— 真机上每一张执行器卡都不属于仿真器，于是一个
    面向解决方案的基准测试在真实机器人上一步也跑不了。

    现在它回答的是「要请人确认什么」。拒绝与否是调用方的事：仿真器在场时这个列表为空，
    行为和从前一字不差；真机上非空，`/case/run` 据此要求一次明确的人工确认。

    拦的东西没有变松：用例的 prompt 经主 agent 变成真实的工具调用，而注入的文本和
    一句真实指令在 collector 眼里完全一样。变的只是由谁决定 —— 从代码写死，改成现场
    那个人。
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


# ── 世界 ──────────────────────────────────────────────────────────────────────
#
# 用例测的是**解决方案**，而一个解决方案既能跑在仿真器上也能跑在真机器人上。区别只有
# 两件事：世界能不能被重置，以及事实从哪来。抽成两个实现，`CaseRun` 就不必知道自己
# 跑在哪一侧 —— 它原先整个是仿真器形状的（`sim_scenario` / `sim_report` 硬写在五处），
# 于是真机上连一步都走不了。


class SimulatorWorld:
    """仿真器持有的世界：可重置、可记账，事实由 `sim_report` 给出（含轨迹）。"""

    kind = 'simulator'
    resettable = True

    def __init__(self, mcp_id: str):
        self.mcp_id = mcp_id

    async def _call(self, tool: str, args: dict) -> dict:
        result = await mcp_client.call_tool_direct(self.mcp_id, tool, args)
        return result if isinstance(result, dict) else {'error': str(result)}

    async def reset(self, run: dict, seed: int) -> dict:
        world = run.get('world') or {}
        return await self._call(SCENARIO_TOOL, {
            'action': 'reset', 'map': world.get('map', ''),
            'spawn': world.get('spawn') or {}, 'seed': seed, 'owner': OWNER})

    async def release(self) -> None:
        await self._call(SCENARIO_TOOL, {'action': 'abort', 'owner': OWNER})

    async def note(self, text: str) -> None:
        await self._call(SCENARIO_TOOL, {'action': 'note', 'text': text})

    async def facts(self) -> dict:
        return await self._call(REPORT_TOOL, {'what': 'report'})


class RealWorld:
    """真实机器人所在的世界：重置不了，事实由 agent-core 自己记。

    `reset` 是**显式的空操作**，不是悄悄跳过：物理世界没有出生点可以回到，机器人就停
    在上一次运行结束的地方。这件事要进复盘 —— 否则第二次重复的起点和第一次不同，而
    记录里看不出任何差别，只会显示两次分数不一样。
    """

    kind = 'real'
    resettable = False

    def __init__(self, waypoints: list[str] | None = None):
        self.waypoints = waypoints or []
        self._recorder = None

    async def reset(self, run: dict, seed: int) -> dict:
        self._recorder = benchmark_facts.start(self.waypoints)
        return {'reset': False,
                'note': '真机不重置世界：机器人停在上一次运行结束的位置'}

    async def release(self) -> None:
        benchmark_facts.stop()

    async def note(self, text: str) -> None:
        return                      # 没有世界日志可记；插话已经在事实流里有 speak 事件

    async def facts(self) -> dict:
        recorder = self._recorder or benchmark_facts.current()
        return recorder.facts() if recorder else {'events': [], 'acp_posts': []}


# ── 一次运行 ──────────────────────────────────────────────────────────────────

class CaseRun:
    """一个用例的 N 次重复。同一时刻只允许有一个。"""

    def __init__(self, case: dict, mcp_id: str | None, repeats: int, seed: int,
                 run_id: str, environment: dict, world=None):
        self.case = case
        self.mcp_id = mcp_id
        # `mcp_id` 为空就是真机：没有仿真器持有世界。
        self.world = world or (SimulatorWorld(mcp_id) if mcp_id
                               else RealWorld(_expected_waypoints(case)))
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

    # -- 与世界说话 -----------------------------------------------------------

    async def _facts(self) -> dict:
        """这次运行的事实：事件流 + ACP 记录。判定不看驱动自己的 assertions。"""
        return await self.world.facts()

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
            # 早就结束的运行。真机这一侧还给的是事实记录器 —— 不停的话，运行结束后
            # 机器人继续做的事会漏进这次运行的事实里。
            await self.world.release()
            await self._finish()

    async def _one(self, index: int) -> dict:
        run = self.case.get('run') or {}
        started = time.time()

        reset = await self.world.reset(run, self.seed + index)
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
        `nav_cancelled`），但**人**要看：`sim_report` 是事后复盘一次运行的地方，
        而一条「机器人好好走着突然掉头」的记录里，如果没有那句插话，读的人无从知道
        为什么。说话经由 collector，不经由仿真器，所以不补这一笔它就不在场。
        """
        if not str(text).strip():
            return
        await event_bus.enqueue(source='message', text=str(text),
                                payload={'benchmark': True})
        try:
            await self.world.note(f'[用例{label}] {text}')
        except Exception:
            pass      # 记账失败不该让一次运行停下来

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
        # 判不了的不算失败（`measurable: False`）—— 否则真机上每次运行都会因为
        # 「没有轨迹占用数据」被标成 failed，而那不是 agent 做错了什么。
        failures = [r['name'] for r in results
                    if not r['ok'] and r.get('measurable', True)
                    and r.get('detail') != '未断言']
        name = self.case.get('name', '') or 'case'
        outcome = 'error' if error else ('ok' if not failures else 'failed')
        # `scenario` 和 `outcome` 也放进返回的行里：面板拿同一份数据渲染进度，
        # 少了它们那一行就显示成「#1 ·」和「undefined」。
        row = {
            'repeat': index, 'seed': self.seed + index, 'elapsed': elapsed,
            'scenario': name, 'outcome': outcome,
            'score': score, 'error': error,
            'failures': failures, 'results': results,
        }
        benchmark_store.add_case(
            self.run_id, scenario=name, repeat_idx=index, seed=self.seed + index,
            ok=bool(not error and not failures), outcome=outcome,
            score=score.get('total'), elapsed_ms=int(elapsed * 1000),
            assertions=failures, facts=payload.get('facts') or {})
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
            detail=self.error or f'n={len(self.cases)} scored={len(scored)}',
            agent_track=self._freeze_agent_track())

    def _freeze_agent_track(self) -> list:
        """把 agent 这一侧也定格。

        会话里的轮次是活的：运行结束后 agent 接着工作，同一批行继续被改写。不定格，
        同一条记录过几分钟再打开就换了个样子 —— 真机上先显示 +32.8s，后来变成
        「时间落在本轮之外」，而那次运行一个字都没变。
        """
        try:
            from api.benchmark import _agent_track
            stored = benchmark_store.get_run(self.run_id) or {}
            return _agent_track(stored.get('session_id', ''),
                                stored.get('started_at'), time.time())
        except Exception:
            return []

    def snapshot(self) -> dict:
        return {'state': self.state, 'run_id': self.run_id,
                'repeat': self.repeat_idx, 'repeats': self.repeats,
                'cases': self.cases, 'error': self.error}


# ── 触发与收尾判定 ────────────────────────────────────────────────────────────

def _expected_waypoints(case: dict) -> list[str]:
    """用例点名要去的那些站。

    真机的事实记录器靠这份名单认出一次派发的目标是哪一站 —— 它在参数值里找它们，
    因为参数**名**各家不同（见 `benchmark_facts` 的模块文档）。用例没点名站序的话，
    名单为空，导航事实就没有 label，而那些断言本来也没被断言。
    """
    expect = (case.get('evaluate') or {}).get('expect') or {}
    names = list(expect.get('waypoint_order') or [])
    for key in ('resume_target',):
        if expect.get(key):
            names.append(expect[key])
    leg = expect.get('interrupted_leg') or {}
    if leg.get('target'):
        names.append(leg['target'])
    seen: list[str] = []
    for name in names:
        if str(name) and str(name) not in seen:
            seen.append(str(name))
    return seen


def _trigger_due(injection: dict, events: list, elapsed: float) -> Optional[float]:
    """这条插话该在第几秒（运行开始起算）发出；条件还没满足就返回 None。

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
    """所有期望到达的站点都到过了就算运行结束，不必等满预算。"""
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
