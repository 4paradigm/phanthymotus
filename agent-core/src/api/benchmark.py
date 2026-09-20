"""api/benchmark.py — 浏览场景、起一批、看分数、看历史。

面板只在**检测到仿真器**（一个同时提供 `sim_scenario` 与 `sim_report` 的 MCP）时
才有内容 —— 出厂的机器人不该看到一个 Benchmark 标签。

## 驱动产出事实，这一层做裁判

判定在 `benchmark_case.py`，不在仿真器里。裁判和被测系统必须不相交 —— 而裁判原先
住在仿真器驱动里，那正是被测系统的一部分。仿真器只负责产出事实（事件流、ACP 记录），
断言是 `(用例, 事件, ACP记录) → 判定` 的纯函数，搬上来之后顺带也能判真机跑出来的
同形状事件流。

除此之外这里做三件事：把批次发过去、把结果收回来、把**被测配置**一起记下来。

## 为什么不按名字找仿真器

按 `tools` 里有没有 `sim_scenario` / `sim_report` 来认。设备名、`server_name`、
容器名都会变，而工具名是契约的一部分。
"""

import os
import platform

import fastapi
from fastapi import APIRouter, Query
from pydantic import BaseModel

import benchmark_store
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
        import config
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


async def _call(mcp_id: str, tool: str, args: dict) -> dict:
    result = await mcp_client.call_tool_direct(mcp_id, tool, args)
    return result if isinstance(result, dict) else {'error': str(result)}


# ── 可用性 ────────────────────────────────────────────────────────────────────

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

    online = {str(e.get('server_name') or ''): mid
              for mid, e in mcp_client.registry.items() if e.get('online')}
    missing_drivers = [d for d in drivers if d not in online]

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


@router.get('/scenarios')
async def scenarios():
    mcp_id = find_simulator()
    if mcp_id is None:
        return {'scenarios': [], 'error': 'no simulator registered'}
    result = await _call(mcp_id, REPORT_TOOL, {'what': 'list'})
    return {'mcp_id': mcp_id, 'scenarios': result.get('scenarios', []),
            **({'error': result['error']} if 'error' in result else {})}


# ── 起一批 ────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    scenarios: list[str] = []
    repeats: int = 1
    seed: int = 0


@router.post('/run')
async def run(request: RunRequest):
    mcp_id = find_simulator()
    if mcp_id is None:
        raise fastapi.HTTPException(status_code=409, detail='no simulator registered')

    result = await _call(mcp_id, SCENARIO_TOOL, {
        'action': 'run_suite', 'scenarios': request.scenarios,
        'repeats': max(1, int(request.repeats)), 'seed': int(request.seed),
    })
    if 'error' in result:
        raise fastapi.HTTPException(status_code=400, detail=result['error'])

    environment = _environment()
    suite = ','.join(result.get('scenarios') or request.scenarios) or 'all'
    run_id = benchmark_store.create_run(
        suite, n_repeats=max(1, int(request.repeats)),
        tier=environment['tier'], llm_model=environment['llm_model'],
        llm_provider=environment['llm_provider'], host=environment['host'],
        image_tags=environment['image_tags'], git_shas=environment['git_shas'])
    return {'run_id': run_id, 'mcp_id': mcp_id, 'suite': suite,
            'repeats': max(1, int(request.repeats)), 'started': result}


@router.post('/abort')
async def abort():
    mcp_id = find_simulator()
    if mcp_id is None:
        raise fastapi.HTTPException(status_code=409, detail='no simulator registered')
    return await _call(mcp_id, SCENARIO_TOOL, {'action': 'abort_suite'})


@router.get('/progress')
async def progress():
    """跑着的时候轮询。

    `sim_report` 是 `resource` 类型，`_needs_barrier` 豁免这一类 —— 所以一段 90 秒
    导航 pending 期间也查得到进度。写成 actuator 就查不了。
    """
    mcp_id = find_simulator()
    if mcp_id is None:
        return {'state': 'idle', 'error': 'no simulator registered'}
    return await _call(mcp_id, REPORT_TOOL, {'what': 'suite'})


@router.post('/runs/{run_id}/collect')
async def collect(run_id: str):
    """把仿真器给出的判定落盘。分数由仿真器算，这里只记录。"""
    stored = benchmark_store.get_run(run_id)
    if stored is None:
        raise fastapi.HTTPException(status_code=404, detail='unknown run')
    mcp_id = find_simulator()
    if mcp_id is None:
        raise fastapi.HTTPException(status_code=409, detail='no simulator registered')

    summary = await _call(mcp_id, REPORT_TOOL, {'what': 'suite'})
    if 'error' in summary:
        raise fastapi.HTTPException(status_code=400, detail=summary['error'])

    for case in summary.get('cases') or []:
        score = (case.get('score') or {}).get('total')
        elapsed = case.get('elapsed')
        benchmark_store.add_case(
            run_id, scenario=case.get('scenario', ''),
            repeat_idx=int(case.get('repeat', 0)), seed=int(case.get('seed', 0)),
            ok=case.get('outcome') == 'ok', outcome=case.get('outcome', ''),
            score=score, elapsed_ms=int(elapsed * 1000) if elapsed else None,
            assertions=case.get('failures') or [])

    by_dimension = {}
    for scenario in (summary.get('scenarios') or {}).values():
        for name, value in (scenario.get('by_dimension') or {}).items():
            by_dimension.setdefault(name, []).append(value)
    benchmark_store.finish_run(
        run_id,
        status='done' if summary.get('state') == 'done' else 'partial',
        score_total=summary.get('mean'), score_stdev=summary.get('stdev'),
        scores_by_dim={name: round(sum(v) / len(v), 1) for name, v in by_dimension.items()},
        detail=f"n={summary.get('n')} scored={summary.get('scored')}")
    return benchmark_store.get_run(run_id)


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


@router.delete('/runs/{run_id}')
async def delete(run_id: str):
    if not benchmark_store.delete_run(run_id):
        raise fastapi.HTTPException(status_code=404, detail='unknown run')
    return {'ok': True, 'deleted': run_id}


@router.get('/trend')
async def trend(suite: str = Query(''), limit: int = Query(30, ge=1, le=200)):
    return {'trend': benchmark_store.trend(suite=suite, limit=limit)}
