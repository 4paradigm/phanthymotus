"""api/benchmark.py — 用例库、跑一个用例、看分数、看历史。

面板只在**检测到仿真器**（一个同时提供 `sim_scenario` 与 `sim_report` 的 MCP）时
才有内容 —— 出厂的机器人不该看到一个 Benchmark 标签。

## 主体是用例库，不是「打开一个文件」

进入一个用例的路径是**看现有的方案**：本机一份可编辑的用例库，加上市场里带 `test`
段的方案。文件导入还在，但它是导入路径，不是主体。

原先这里还有一套「勾几个场景跑一批」的端点（`/scenarios`、`/run`、`/progress`、
`/runs/{id}/collect`），已经删掉：那套的执行与评分都在仿真器卡片上，而卡片不该拥有
一个测试用例 —— 被测的 agent 能调到它，裁判也就住进了被测系统内部。场景现在描述的是
**世界**，不是一个可勾选的测试单位。

## 驱动产出事实，这一层做裁判

判定在 `benchmark_case.py`，不在仿真器里。裁判和被测系统必须不相交 —— 而裁判原先
住在仿真器驱动里，那正是被测系统的一部分。仿真器只负责产出事实（事件流、ACP 记录），
断言是 `(用例, 事件, ACP记录) → 判定` 的纯函数，搬上来之后顺带也能判真机跑出来的
同形状事件流。

除此之外这里做的事：跑用例（`benchmark_runner`）、把结果连同**被测配置**记下来，
以及在用例覆盖画布之前把画布存成一个解决方案包 —— 用户自己搭的那套东西没有别的地方
存着，它就在画布上。

## 为什么不按名字找仿真器

按 `tools` 里有没有 `sim_scenario` / `sim_report` 来认。设备名、`server_name`、
容器名都会变，而工具名是契约的一部分。
"""

import os
import platform
import time

import fastapi
from fastapi import APIRouter, Query
from pydantic import BaseModel

import benchmark_store
import config
import mcp_client

router = APIRouter(prefix='/benchmark', tags=['benchmark'])

SCENARIO_TOOL = 'sim_scenario'
REPORT_TOOL = 'sim_report'


def find_simulator() -> str | None:
    """第一个同时提供两张仿真卡片的在线 MCP。"""
    for mcp_id, entry in mcp_client.registry.items():
        if not entry.get('online'):
            continue
        tools = set(entry.get('tools') or [])
        if {SCENARIO_TOOL, REPORT_TOOL} <= tools:
            return mcp_id
    return None


def _environment() -> dict:
    """写进每条记录的「被测配置」。

    不带配置的分数是噪音 —— 真正的问题是「换了模型/改了 prompt 之后分数动没动」，
    没有这些字段就答不了。
    """
    llm = {}
    try:
        clients = (config.main.get('client') or {}).get('llm') or []
        if clients:
            llm = clients[0] or {}
    except Exception:
        llm = {}
    return {
        'tier': os.environ.get('SIM_TIER', 'fidelity'),
        'llm_model': str(llm.get('model', '')),
        'llm_provider': str(llm.get('url', '')),
        'host': os.environ.get('HOSTNAME') or platform.node(),
        'image_tags': {'agent_core': os.environ.get('IMAGE_TAG', '')},
        'git_shas': {'agent_core': os.environ.get('GIT_SHA', '')},
    }


def _current_session() -> str:
    """主 agent 当前的会话 id。

    跑动结束之后要回看「它当时怎么想的、调了什么」，而那份记录在 `chat_history` 里
    按会话存。不在开跑时记下来，事后就只能靠时间去猜是哪一段对话。
    """
    try:
        import event
        return str(getattr(event.llm, '_session_id', '') or '')
    except Exception:
        return ''


async def _call(mcp_id: str, tool: str, args: dict) -> dict:
    result = await mcp_client.call_tool_direct(mcp_id, tool, args)
    return result if isinstance(result, dict) else {'error': str(result)}


# ── 可用性 ────────────────────────────────────────────────────────────────────

def _driver_online(declared: str) -> bool:
    """本机有没有这个驱动，而且它在线。

    名字住在**注册表**（`config.main['services']['mcp']`，就是 `/api/mcp` 返回的那份），
    在线与否住在 `mcp_client.registry`。两份数据，各答一半。

    这里原先只问后者要 `server_name` —— 而运行时那份根本不存这个字段，于是每个名字
    都比不上，任何用例都报「缺驱动」，包括驱动就在旁边跑着、`available` 同时还答
    `true` 的时候。Orin6 上一跑就现原形。
    """
    entry = next((m for m in (config.main.get('services', {}).get('mcp') or [])
                  if m.get('server_name') == declared or m.get('name') == declared), None)
    if entry is None:
        return False
    return bool((mcp_client.registry.get(entry.get('id')) or {}).get('online'))


async def case_readiness(needs: dict) -> dict:
    """用例声明的依赖，在本机逐条对一遍。

    这是把用例做成**方案包体的一段**、而不是一张卡片，换来的那件事：卡片住在驱动
    里，驱动没装的时候卡片本身就不存在，于是「你缺这个驱动」这句话没有地方可说 ——
    用户只会看到画布上少了点什么。依赖写在包体里，载入之前就能逐层报出来：先说缺
    哪个驱动，驱动在了再说缺哪张地图。

    两层顺序是有意的：没装驱动就不去问地图。那一问会走 MCP 超时，把一个「没装」
    的清楚结论拖成一个「超时」的含糊结论。
    """
    drivers = [str(d) for d in (needs.get('drivers') or [])]
    assets = [str(a) for a in (needs.get('assets') or [])]
    missing_drivers = [d for d in drivers if not _driver_online(d)]

    ready = {'drivers': drivers, 'assets': assets,
             'missing_drivers': missing_drivers, 'missing_assets': [],
             'assets_checked': False}
    if missing_drivers or not assets:
        ready['ok'] = not missing_drivers
        return ready

    mcp_id = find_simulator()
    if mcp_id is None:
        ready['ok'] = not missing_drivers
        return ready
    result = await _call(mcp_id, SCENARIO_TOOL, {'action': 'list_maps'})
    if 'error' in result:
        # 问不到不等于缺 —— 报成缺失会让一个临时故障看起来像装错了东西。
        ready['ok'] = True
        ready['assets_error'] = result['error']
        return ready
    have = {str(m.get('name') or m) for m in (result.get('maps') or [])}
    ready['assets_checked'] = True
    ready['missing_assets'] = [a for a in assets if a not in have]
    ready['ok'] = not ready['missing_assets']
    return ready


@router.get('/case')
async def current_case():
    """当前已载入方案里的用例（没有则 `case: null`）。"""
    from api.solutions import loaded_case
    payload = loaded_case()
    if not payload:
        return {'case': None}
    import benchmark_case
    return {'case': payload,
            'problems': benchmark_case.validate({'test': payload}),
            'readiness': await case_readiness(
                benchmark_case.requires({'test': payload}))}


@router.get('/available')
async def available():
    """面板据此决定要不要出现。"""
    mcp_id = find_simulator()
    return {'available': mcp_id is not None, 'mcp_id': mcp_id,
            'environment': _environment()}


# ── 本机用例库 ────────────────────────────────────────────────────────────────
#
# 面板的主体是**看现有的方案**，不是「打开一个文件」—— 文件是导入路径。所以本机存
# 一份可编辑的用例库，市场那栏列出带 `test` 段的方案，收进来就能改。改一个权重不该
# 去走一遍发布审核。


class CaseWrite(BaseModel):
    payload: dict = {}
    name: str = ''
    origin: str = ''


def _case_view(record: dict, loaded: dict | None) -> dict:
    """列表里一张卡需要知道的一切，含「跑不了的理由」和上一次的分数。"""
    import benchmark_case
    payload = record['payload']
    block = benchmark_case.summary(payload)
    recent = benchmark_store.trend(suite=record['name'], limit=1)
    return {
        'id': record['id'], 'name': record['name'], 'origin': record['origin'],
        'updated_at': record['updated_at'],
        **block,
        'problems': benchmark_case.validate(payload),
        'isLoaded': bool(loaded and loaded == benchmark_case.test_block(payload)),
        'last': recent[0] if recent else None,
    }


@router.get('/cases')
async def list_cases():
    from api.solutions import loaded_case
    loaded = loaded_case()
    return {'cases': [_case_view(record, loaded)
                      for record in benchmark_store.list_cases()]}


@router.get('/cases/market')
async def market_cases(search: str = '', limit: int = Query(30, ge=1, le=50)):
    """市场上**带 test 段的**方案。

    过滤在这一层做，靠 `includes` 里有没有 `test` —— 市场列表不返回包体（几十 KB，
    列表页用不着），但 `includes` 是它返回的字段之一，正好够用。
    """
    from api.solutions import market as solutions_market

    result = await solutions_market(search=search, industry='all', limit=limit)
    if result.get('code') != 200:
        return {'cases': [], 'error': result.get('error', '连不上方案市场')}
    items = [s for s in (result.get('data') or []) if 'test' in (s.get('includes') or [])]
    return {'cases': items}


@router.get('/cases/{case_id}')
async def get_case(case_id: str):
    record = benchmark_store.get_case(case_id)
    if record is None:
        raise fastapi.HTTPException(status_code=404, detail='没有这个用例')
    return record


@router.post('/cases')
async def create_case(req: CaseWrite):
    """新建 / 从文件或市场导入。

    这里**不拒绝**不合法的用例：刚建出来的空用例本来就是不合法的（没有初始指令），
    拒绝它就没法新建了。问题随列表一起报出来，跑的时候才真正拦。
    """
    import benchmark_case
    payload = req.payload or {'formatVersion': 1, 'canvas': {}, 'devices': [],
                              'test': benchmark_case.blank()}
    if not benchmark_case.is_case(payload):
        raise fastapi.HTTPException(status_code=422, detail='这个解决方案没有 test 段，不是用例')
    case_id = benchmark_store.save_case(
        payload, name=req.name or benchmark_case.summary(payload)['name'],
        origin=req.origin)
    return benchmark_store.get_case(case_id)


@router.put('/cases/{case_id}')
async def update_case(case_id: str, req: CaseWrite):
    """保存编辑。

    保存**可以**存下一个还不能跑的用例 —— 编辑是分几次做完的，存一半不该被拒。
    但理由要一起回去，编辑器当场显示，而不是等到点「跑」才说。
    """
    import benchmark_case
    if benchmark_store.get_case(case_id) is None:
        raise fastapi.HTTPException(status_code=404, detail='没有这个用例')
    if not benchmark_case.is_case(req.payload):
        raise fastapi.HTTPException(status_code=422, detail='这个包体没有 test 段，不是用例')

    benchmark_store.save_case(req.payload, name=req.name, origin=req.origin,
                              case_id=case_id)
    record = benchmark_store.get_case(case_id)
    return {**record, 'problems': benchmark_case.validate(req.payload)}


@router.delete('/cases/{case_id}')
async def delete_case(case_id: str):
    return {'deleted': benchmark_store.delete_case(case_id)}


@router.post('/cases/{case_id}/preflight')
async def preflight_case(request: fastapi.Request, case_id: str, session_id: str = ''):
    """载入前检查：缺哪些驱动、会覆盖掉什么、用例本身还差什么。

    转给 `solutions.preflight`，不另写一套 —— 覆盖清单、`deviceRef` 映射、
    `test.readiness` 的分层报错都在那边，复制一份就会漏掉其中一项。
    """
    from api.solutions import LoadRequest, preflight
    record = benchmark_store.get_case(case_id)
    if record is None:
        raise fastapi.HTTPException(status_code=404, detail='没有这个用例')
    return await preflight(request, LoadRequest(
        payload=record['payload'], includes=['canvas', 'test'], session_id=session_id))


@router.post('/cases/{case_id}/apply')
async def apply_case(request: fastapi.Request, case_id: str, session_id: str = ''):
    """把这个用例载入成当前方案（会覆盖画布）。"""
    from api.solutions import LoadRequest, apply as apply_solution
    record = benchmark_store.get_case(case_id)
    if record is None:
        raise fastapi.HTTPException(status_code=404, detail='没有这个用例')
    result = await apply_solution(request, LoadRequest(
        payload=record['payload'], includes=['canvas', 'test'],
        confirm=True, session_id=session_id))
    if result.get('code') != 200:
        raise fastapi.HTTPException(status_code=result.get('code', 500),
                                    detail=result.get('error', '载入失败'))
    return {'applied': True, 'case': record['name']}


# ── 载入用例前：先把现在的画布存下来 ──────────────────────────────────────────

SNAPSHOT_KEY = 'benchmark_canvas_snapshot'


@router.post('/snapshot')
async def snapshot(request: fastapi.Request):
    """把当前画布打包成一个解决方案，存在本机并回给前端下载。

    载入用例会覆盖画布。用户自己搭的那套东西没有别的地方存着 —— 它就在画布上。所以
    在覆盖之前先打成一个包：一份留在机器上，一份用户可以下载，和用例一样是一个能在
    本地与市场上流通的文件。

    用 `solutions` 的打包路径，不另写一套：`deviceRef` 映射、`x-sensitive` 脱敏、
    版本记录都在那边，复制一遍就会漏掉脱敏。
    """
    from api.solutions import PackInclude, PackRequest, _build_payload, _get_rc_token

    built = await _build_payload(PackRequest(include=PackInclude()),
                                 _get_rc_token(request))
    if not built.get('ok'):
        raise fastapi.HTTPException(status_code=422, detail=built.get('error', '打包失败'))

    payload = built['payload']
    config.main[SNAPSHOT_KEY] = {
        'savedAt': int(time.time()),
        'cards': len(((payload.get('canvas') or {}).get('cards')) or []),
        'payload': payload,
    }
    return {'saved': True, 'cards': config.main[SNAPSHOT_KEY]['cards'],
            'payload': payload, 'includes': built['includes']}


@router.get('/snapshot')
async def snapshot_info():
    """存过的快照。**只报告，不自动还原** —— 跑完自动把画布换回去，会在用户正看着
    结果的时候把画布抽走；还原是用户的决定。"""
    import config
    stored = config.main.get(SNAPSHOT_KEY) or {}
    if not stored:
        return {'saved': False}
    return {'saved': True, 'savedAt': stored.get('savedAt'),
            'cards': stored.get('cards', 0)}


@router.post('/snapshot/restore')
async def snapshot_restore(request: fastapi.Request):
    """把画布换回快照。只在用户点的时候发生。"""
    from api.solutions import LoadRequest, apply as apply_solution

    stored = config.main.get(SNAPSHOT_KEY) or {}
    if not stored.get('payload'):
        raise fastapi.HTTPException(status_code=404, detail='本机没有存过画布快照')
    result = await apply_solution(request, LoadRequest(
        payload=stored['payload'], includes=['canvas'], confirm=True))
    if result.get('code') != 200:
        raise fastapi.HTTPException(status_code=result.get('code', 500),
                                    detail=result.get('error', '还原失败'))
    return {'restored': True, 'cards': stored.get('cards', 0)}


# ── 跑当前载入的用例 ──────────────────────────────────────────────────────────

class CaseRunRequest(BaseModel):
    repeats: int = 1
    seed: int = 0


@router.post('/case/run')
async def run_case(request: CaseRunRequest):
    """跑当前方案带的用例。

    四道门，顺序是有意的：有没有用例 → 用例本身合不合法 → 依赖齐不齐 → 画布上有没有
    真设备。每一道都给出不同的下一步动作，合成一个布尔就全丢了。
    """
    import benchmark_case
    import benchmark_runner
    from api.solutions import loaded_case

    if benchmark_runner.is_busy():
        raise fastapi.HTTPException(status_code=409, detail='已经有一次基准测试在跑')

    case = loaded_case()
    if not case:
        raise fastapi.HTTPException(
            status_code=409, detail='当前方案里没有 test 段，先载入一个测试用例')

    problems = benchmark_case.validate({'test': case})
    if problems:
        raise fastapi.HTTPException(status_code=422, detail='；'.join(problems))

    mcp_id = find_simulator()
    readiness = await case_readiness(benchmark_case.requires({'test': case}))
    if mcp_id is None or not readiness.get('ok'):
        raise fastapi.HTTPException(status_code=409, detail={
            'error': '用例的依赖还不齐', 'readiness': readiness})

    # 安全闸：注入的文本和真实指令无法区分，画布上任何一张会动的真卡片都会真的动。
    unsafe = benchmark_runner.unsafe_cards(mcp_id)
    if unsafe:
        raise fastapi.HTTPException(status_code=409, detail={
            'error': '画布上有会动的真实设备，基准测试会让它们真的动起来',
            'unsafe': unsafe})

    environment = _environment()
    repeats = max(1, int(request.repeats))
    run_id = benchmark_store.create_run(
        case.get('name', '') or 'case', n_repeats=repeats,
        tier=environment['tier'], llm_model=environment['llm_model'],
        llm_provider=environment['llm_provider'], host=environment['host'],
        image_tags=environment['image_tags'], git_shas=environment['git_shas'],
        session_id=_current_session())

    run = benchmark_runner.CaseRun(case, mcp_id, repeats, int(request.seed),
                                   run_id, environment)
    benchmark_runner.set_current(run)
    run.start()
    return {'run_id': run_id, 'repeats': repeats, 'mcp_id': mcp_id}


@router.get('/case/progress')
async def case_progress():
    import benchmark_runner
    run = benchmark_runner.current()
    return run.snapshot() if run else {'state': 'idle'}


@router.post('/case/abort')
async def case_abort():
    import benchmark_runner
    run = benchmark_runner.current()
    if not run:
        return {'state': 'idle'}
    run.abort()
    return {'state': 'aborting', 'run_id': run.run_id}


# ── 历史 ──────────────────────────────────────────────────────────────────────

@router.get('/runs')
async def runs(limit: int = Query(50, ge=1, le=500)):
    return {'runs': benchmark_store.list_runs(limit=limit)}


@router.get('/runs/{run_id}')
async def run_detail(run_id: str):
    stored = benchmark_store.get_run(run_id)
    if stored is None:
        raise fastapi.HTTPException(status_code=404, detail='unknown run')
    return stored


@router.get('/runs/{run_id}/timeline')
async def run_timeline(run_id: str):
    """一次跑动里到底发生了什么 —— 说了什么、想了什么、调了什么、世界怎么回应。

    分数只回答「好不好」，回答不了「哪儿坏了」。而这次跑动的现场**留不住**：仿真器
    的世界下一次跑动一开始就被重置，会话历史会被压缩。所以事实在写入时就存进了
    `benchmark_case.facts`，会话 id 存进了 `benchmark_run.session_id`。

    两条轨道**不强行合成一条**。agent 那侧的时间是墙钟，世界那侧是仿真时钟，两个钟
    相减没有意义 —— 各自归一到「从本轮开始起算的秒数」并排放，比编一个共同时间轴
    诚实。对齐基准两边都是跑动开始：世界的第一条事件是 `scenario_load`（reset 那一刻），
    agent 的第一轮是被那句初始指令唤醒的。
    """
    stored = benchmark_store.get_run(run_id)
    if stored is None:
        raise fastapi.HTTPException(status_code=404, detail='unknown run')

    cases = stored.get('cases') or []
    facts = next((c.get('facts') for c in cases if c.get('facts')), {}) or {}
    events = facts.get('events') or []
    base = events[0].get('t', 0) if events else 0

    world = [{
        'at': round(float(e.get('t', 0)) - base, 1),
        'kind': e.get('event', ''),
        'label': e.get('label', ''),
        'text': e.get('text', ''),
        'status': e.get('status', ''),
    } for e in events]

    return {
        'run': {k: stored.get(k) for k in
                ('id', 'suite', 'status', 'started_at', 'ended_at', 'n_repeats',
                 'llm_model', 'image_tags', 'score_total', 'score_stdev')},
        'cases': [{k: c.get(k) for k in
                   ('repeat_idx', 'seed', 'outcome', 'score', 'assertions')}
                  for c in cases],
        'world': world,
        'transcript': facts.get('transcript') or [],
        'acp': facts.get('acp_posts') or [],
        'agent': _agent_track(stored.get('session_id', ''),
                              stored.get('started_at'), stored.get('ended_at')),
    }


def _agent_track(session_id: str, started, ended) -> list:
    """这次跑动期间 agent 说了什么、调了什么。

    按**轮**取，不按消息取：一轮就是「被什么唤醒 → 想了什么 → 调了哪些工具」，而
    那正好是排查时要看的粒度。会话被清过或压缩掉就只能返回空 —— 与其编一段出来，
    不如让前端说「这次跑动的对话记录已经没有了」。
    """
    if not session_id:
        return []
    try:
        import chat_history
        turns = chat_history.get_session_turns(session_id)
    except Exception:
        return []

    track = []
    for index, turn in enumerate(turns):
        at = turn.get('started_at') or 0
        if started and at and at < float(started) - 5:
            continue          # 跑动开始之前的轮次，不是这次的事
        if ended and at and at > float(ended) + 5:
            continue
        says, calls, trigger = [], [], ''
        for message in turn.get('messages') or []:
            role = message.get('role')
            content = message.get('content')
            if role == 'user' and isinstance(content, str) and not trigger:
                trigger = content[:200]
            elif role == 'assistant':
                if isinstance(content, str) and content.strip():
                    says.append(content.strip())
                for call in message.get('tool_calls') or []:
                    fn = (call.get('function') or {})
                    calls.append({'name': str(fn.get('name', '')).split('__')[-1],
                                  'args': str(fn.get('arguments', ''))[:200]})
        track.append({
            'turn': index,
            'at': round(at - float(started), 1) if (started and at) else None,
            'trigger': trigger, 'says': says, 'calls': calls,
        })
    return track


@router.delete('/runs/{run_id}')
async def delete(run_id: str):
    if not benchmark_store.delete_run(run_id):
        raise fastapi.HTTPException(status_code=404, detail='unknown run')
    return {'ok': True, 'deleted': run_id}


@router.get('/trend')
async def trend(suite: str = Query(''), limit: int = Query(30, ge=1, le=200)):
    return {'trend': benchmark_store.trend(suite=suite, limit=limit)}
