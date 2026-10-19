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
import benchmark_judge
import benchmark_metrics
import benchmark_store
import config
import event_bus
import mcp_client
import perf_log

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

    def schema_name(mcp_id: str, tool: str) -> str:
        """画布上的工具名 → 注册表里真实存在的那个 schema 名。

        带 action 枚举的工具会被 `_connect_one` 按 action 拆开，`tool_meta` 的键变成
        `mcp__<id>__<tool>__<action>`，`mcp__<id>__<tool>` 不存在。而画布卡片记的是
        **工具**名，于是拼出来的名字查不到申报 —— `tool_type` 返回 ''，按「没申报就算
        会动」处理，麦克风就这么进了「会动的设备」清单。

        拆出来的几份 type 相同（都从父工具抄下来），取哪一份都一样。查不到就返回精确
        名：没申报仍然按会动处理，这条兜底不能松。

        只在这里解，不动 `peer/tools.py` —— 那是信任边界，而它拿到的名字来自
        `all_schemas()`，本来就是拆分后的名字，精确命中，不需要这一层。
        """
        exact = f'mcp__{mcp_id}__{tool}'
        metas = (mcp_client.registry.get(mcp_id) or {}).get('tool_meta') or {}
        if exact in metas:
            return exact
        return next((n for n in metas if n.startswith(f'{exact}__')), exact)

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
        if is_read_only(schema_name(mcp_id, tool)):
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
    """仿真器持有的世界：可重置、可记账，事实由 `sim_report` 给出（含轨迹）。

    **但不是只由 `sim_report` 给出。** 这里原先把 `sim_report` 当作唯一来源，理由是
    「仿真器看得见世界」—— 对，可它只看得见**仿真器自己那几张卡**。画布上一张卡若不是
    仿真器的（最常见的就是 perception 的 `tts`：它合成真实音频、发到 `/perception/tts`），
    仿真世界里不会留下任何痕迹。症状是运行日志左边 agent 明明在 `tts.speak`，右边
    「世界真的做了什么」那一列一句播报都没有，而两边都没有任何报错。

    所以两路都开：仿真器的事实为底（轨迹占用、到达这些只有它算得出），agent-core 自己
    记的那份补上仿真器看不见的部分。同一个动作两边都记时按 `action_id` 去重 —— 卡片返回
    给 agent-core 的 `action_id` 就是世界给这个动作的 id，两边对得上。
    """

    kind = 'simulator'
    resettable = True

    def __init__(self, mcp_id: str):
        self.mcp_id = mcp_id
        self._recorder = None
        # 记录器的 `t` 是「本次录制开始以来」，仿真器的 `t` 是「本场景开始以来」——
        # 两个零点不是一回事。合并前把前者搬到后者的时钟上，见 `_align`。
        self._t_offset = 0.0

    async def _call(self, tool: str, args: dict) -> dict:
        result = await mcp_client.call_tool_direct(self.mcp_id, tool, args)
        return result if isinstance(result, dict) else {'error': str(result)}

    async def reset(self, run: dict, seed: int) -> dict:
        self._recorder = benchmark_facts.start()
        world = run.get('world') or {}
        outcome = await self._call(SCENARIO_TOOL, {
            'action': 'reset', 'map': world.get('map', ''),
            'spawn': world.get('spawn') or {}, 'seed': seed, 'owner': OWNER})
        await self._align()
        return outcome

    async def _align(self) -> None:
        """把记录器的零点搬到仿真世界的**事件时钟**上。

        两边都用「相对秒数」，但相对的**不是同一个起点**：仿真器的事件 `t` 数的是
        它自己的进程时钟（一台跑了一个月的机器上就是三百多万秒），记录器数的是本次
        录制开始以来。合并后一排序，记录器那些几秒的 `t` 全被顶到最前面，而运行详情
        的右列拿最早那条事件当零点 —— 整列前移，看起来像机器人在被要求之前就开了口。

        锚点用仿真器自己在重置时记下的 `scenario_load`：那一刻就是记录器的零点附近
        （`start()` 紧挨着 `reset` 调用之前）。**不能用报告里的 `elapsed`** —— 它数的
        是本场景秒数，和事件 `t` 在仿真器内部就不是一个基准，拿它算出来的 offset 约等于
        0，于是这一列照样是错的，只是错得不那么显眼。

        找不到锚点就不搬（offset 留 0）：宁可两列差十几秒，也不要搬一个瞎猜的量 ——
        搬错了，「到了再讲」这类时序判定会拿错位的时间去比大小，而错位本身看不出来。
        """
        if self._recorder is None:
            return
        report = await self._call(REPORT_TOOL, {'what': 'report'})
        events = report.get('events') if isinstance(report, dict) else None
        if not events:
            return
        anchors = [e.get('t') for e in events if e.get('event') == 'scenario_load']
        anchor = anchors[-1] if anchors else min(
            (e.get('t') for e in events if e.get('t') is not None), default=None)
        if anchor is None:
            return
        try:
            self._t_offset = float(anchor)
        except (TypeError, ValueError):
            self._t_offset = 0.0

    async def release(self) -> None:
        benchmark_facts.stop()
        self._recorder = None
        await self._call(SCENARIO_TOOL, {'action': 'abort', 'owner': OWNER})

    async def note(self, text: str) -> None:
        await self._call(SCENARIO_TOOL, {'action': 'note', 'text': text})

    async def facts(self) -> dict:
        report = await self._call(REPORT_TOOL, {'what': 'report'})
        recorder = self._recorder or benchmark_facts.current()
        if recorder is None or not isinstance(report, dict):
            return report
        return _merge_facts(report, recorder.facts(), self._t_offset)


def _merge_facts(report: dict, mine: dict, t_offset: float = 0.0) -> dict:
    """仿真器的事实为底，补上它看不见的那些。

    **只合事件，不合 `acp_posts`。** 后者是 `exactly_one_terminal_post` 用来数
    「同一个动作上报了几次」的，按 `action_id` 去重会把真的重复上报一起抹掉 —— 那正是
    它要抓的东西。仿真器那份已经完整。
    """
    seen = {str(e.get('action_id')) for e in (report.get('events') or [])
            if e.get('action_id')}
    extra = [{**e, 't': round(float(e.get('t') or 0.0) + t_offset, 3)}
             for e in (mine.get('events') or [])
             if str(e.get('action_id') or '') not in seen]
    if not extra:
        return report
    events = sorted((report.get('events') or []) + extra,
                    key=lambda e: float(e.get('t') or 0.0))
    return {**report, 'events': events,
            'source': 'simulator+agent-core'}


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
                 run_id: str, environment: dict, world=None, case_id: str = ''):
        self.case = case
        # 哪个用例在跑。面板据此把那张卡的「运行」换成「停止」，其余置灰 —— 刷新页面
        # 之后也要对，所以它得从服务端来，不能只活在前端的一个变量里。
        self.case_id = case_id
        self.mcp_id = mcp_id
        # `mcp_id` 为空就是真机：没有仿真器持有世界。
        self.world = world or (SimulatorWorld(mcp_id) if mcp_id
                               else RealWorld())
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
        # UX 那几个指标要知道「用户是什么时候说完的」「插话是什么时候发出去的」。
        # 只有这里知道 —— 事实流里没有这两个时刻，它记的是机器人做了什么。
        self._marks = {'started': started, 'prompt_at': None, 'injections_at': []}

        reset = await self.world.reset(run, self.seed + index)
        if 'error' in reset:
            return self._case_row(index, started, {}, error=reset['error'])

        self.state = 'running'
        # 开场那句是**指令**，不是插话 —— 记错了，复盘的人会以为一开始就有人打断。
        await self._say(run.get('prompt', ''), label='指令')
        # 「用户说完」的时刻。首次响应时延从这里起算 —— 从 `started` 起算的话，
        # 会把世界重置那几秒也算进用户的等待里。
        self._marks['prompt_at'] = time.time()

        facts = await self._watch(run, started)
        payload = {'test': self.case}
        window = (started, time.time())
        seen = benchmark_metrics.observations(
            facts, perf_log.spans_between(*window), _usage_between(*window),
            window, self._marks)

        # 确定性那一半先判完 —— 它不花钱、不会抖，而且裁判要拿着这些数去判剩下那半。
        items = benchmark_case.check_targets(seen, benchmark_case.targets(payload))
        verdict = await benchmark_judge.judge(payload, seen, facts,
                                              self._agent_track_now(window))
        items += verdict['items']

        score = benchmark_case.score(payload, items)
        return self._case_row(index, started,
                              {'facts': facts, 'results': items, 'score': score,
                               'observations': seen, 'judge': verdict},
                              error=verdict['error'])

    def _session_id(self, live: bool = False) -> str:
        """这次运行对应哪一段对话，必要时现补。

        **开跑那一刻 agent 可能还没有会话** —— agent-core 刚重启、这一轮之前没人说过
        话，`event.llm._session_id` 就是空的，而会话恰恰是被这次运行的指令创建出来的。
        记成空之后再没人回头补，于是运行详情左栏永远是「这次运行的对话记录已经没有
        了」，尽管右栏的世界事实一条不少 —— 看起来像 agent 什么都没做。

        只在**这次运行还活着**的时候补：跑完的那条记录若补上「现在」的会话，等于把
        一段无关的对话安到一次历史运行上，而它看起来完全正常。
        """
        stored = benchmark_store.get_run(self.run_id) or {}
        known = str(stored.get('session_id') or '')
        if known or not (live or self.state in ('running', 'starting')):
            return known
        try:
            from api.benchmark import _current_session
            found = _current_session()
        except Exception:
            return ''
        if found:
            benchmark_store.set_session(self.run_id, found)
        return found

    def _agent_track_now(self, window: tuple) -> list:
        try:
            from api.benchmark import _agent_track
            return _agent_track(self._session_id(), window[0], window[1])
        except Exception:
            return []

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
        """边轮询事实边按触发条件追加插话，直到安静下来或者预算耗尽。

        **怎么算结束，这一版换了。** 原先是「所有期望站点都到过了」—— 那要求用例先声明
        一串站名，也就是把展区导览的词汇焊进了收尾逻辑。现在只剩两条通用的：

        * **预算耗尽**（`run.budget_seconds`）
        * **安静下来**：`idle_seconds` 内没有新事实，**而且**没有 pending 的 ACP 动作

        后半句不能少。没有新事实常常只是因为机器人正走在半路上 —— 一段两分钟的导航
        期间事实流就是不动的。光看「没有新事实」会在半路上把运行判结束，然后给一个
        「什么都没做完」的分数。
        """
        budget = float(run.get('budget_seconds') or 900)
        quiet_for = float(run.get('idle_seconds') or 60)
        pending = [dict(i) for i in (run.get('injections') or [])]
        armed: dict[int, float] = {}
        facts: dict = {}
        last_change = time.time()
        last_size = -1

        while not self._abort.is_set() and time.time() - started < budget:
            facts = await self._facts()
            events = facts.get('events') or []
            elapsed = time.time() - started

            if len(events) != last_size:
                last_size, last_change = len(events), time.time()

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
                    self._marks['injections_at'].append(time.time())
                    injection['done'] = True

            # 已经触发、还没发出去的插话必须发完再收尾。世界跑得比 LLM 快的时候
            # （加速倍率、或者一段很短的路），事情会先做完 —— 就这么收尾的话，打断
            # 根本没发生过，而那条要求会记在 agent 头上。
            outstanding = any(index in armed and not injection.get('done')
                              for index, injection in enumerate(pending))
            if (not outstanding and time.time() - last_change >= quiet_for
                    and not mcp_client.get_pending_actions()):
                break
            await asyncio.sleep(_POLL_SECONDS)

        return facts or await self._facts()

    # -- 落盘 -----------------------------------------------------------------

    def _case_row(self, index: int, started: float, payload: dict,
                  error: str = '') -> dict:
        score = payload.get('score') or {}
        results = payload.get('results') or []
        elapsed = time.time() - started
        # 直接用 `score()` 算好的那一份，不在这里重算一遍 —— 「判不了的不算失败」
        # 这条规则有两份实现的话，改一处就会悄悄分叉，而分叉的表现是某些运行被标成
        # failed 却列不出任何失败项。
        failures = list(score.get('failures') or [])
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
            assertions=failures, facts=payload.get('facts') or {},
            results=results, observations=payload.get('observations') or {})
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
            # `live=True`：定格发生在收尾那一刻，`state` 已经翻成 done/aborted，但
            # 「现在的会话」仍然就是这次运行的那段。这里不补，一次从头到尾没调过
            # `_agent_track_now` 的运行会把左栏永久定格成空。
            return _agent_track(self._session_id(live=True),
                                stored.get('started_at'), time.time())
        except Exception:
            return []

    def snapshot(self) -> dict:
        return {'state': self.state, 'run_id': self.run_id, 'case_id': self.case_id,
                'repeat': self.repeat_idx, 'repeats': self.repeats,
                'cases': self.cases, 'error': self.error}


# ── 触发与收尾判定 ────────────────────────────────────────────────────────────

def _trigger_due(injection: dict, events: list, elapsed: float) -> Optional[float]:
    """这条插话该在第几秒（运行开始起算）发出；条件还没满足就返回 None。

    两件事：

    * **`after_arrival` 按事件触发，不按绝对时刻。** 真机上 LLM 一轮 3-48 秒，写死的
      偏移会落到完全不同的一段路上 —— 于是「在去 P6 的路上被打断」测的是别的东西。
    * **`delay` 从**看见**到达那一刻起算，不是从事件自己的时间戳起算。** 事件的 `t`
      是驱动的时钟，加速倍率下和墙钟不是一回事；两个时钟相减出来的秒数没有意义。
    """
    delay = float(injection.get('delay') or 0)

    # 通用触发：第 N 个异步动作完成之后。这是 `after_arrival` 的推广 —— 「到达某一站」
    # 是展区导览的说法，而「第 N 个动作做完」在任何用例里都成立，判据也一样在事实流里。
    nth = injection.get('after_action')
    if nth is not None:
        done = sum(1 for e in events
                   if e.get('event') in ('arrive', 'speak_end', 'nav_cancelled',
                                         'nav_failed'))
        return elapsed + delay if done >= int(nth) else None

    label = injection.get('after_arrival')      # 旧用例还在用，继续认
    if label:
        seen = any(e.get('event') == 'arrive' and e.get('label') == label for e in events)
        return elapsed + delay if seen else None
    at = injection.get('at')
    return None if at is None else float(at) + delay


def _usage_between(start: float, end: float) -> dict:
    """这段时间里的 token 用量。cache 命中率就是从这儿来的。"""
    try:
        return perf_log.query_usage_summary(start, end)
    except Exception:
        return {}


def _stdev(values: list) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return round((sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5, 1)


# ── 单例 ──────────────────────────────────────────────────────────────────────

_current: Optional[CaseRun] = None
# 「有人已经占上了，但 CaseRun 还没造出来」。
#
# 没有这一格的话，`is_busy()` 检查和 `set_current()` 之间隔着一个 await（依赖检查，
# 还可能走 MCP 网络），两个并发请求会**双双通过**，于是两次跑动同时往同一个 agent
# 注入用户消息、同时重置世界。两份事实流交织在一起，而分数看起来只是「莫名其妙地低」。
_claimed = False


def current() -> Optional[CaseRun]:
    return _current


def claim() -> bool:
    """占位。检查与占位必须是同一步 —— 这就是这个函数存在的全部理由。

    占上了要么 `set_current` 接管，要么 `release()` 还回去；中途抛异常而不还，
    面板会一直说「已经有一次基准测试在跑」，而实际上什么都没跑。
    """
    global _claimed
    if is_busy():
        return False
    _claimed = True
    return True


def release() -> None:
    global _claimed
    _claimed = False


def set_current(run: Optional[CaseRun]) -> None:
    global _current, _claimed
    _current = run
    _claimed = False              # 占位交棒给真正的跑动


def is_busy() -> bool:
    return _claimed or (_current is not None and _current.state in ('starting', 'running'))
