"""api/benchmark.py — 用例库、跑一个用例、看分数、看历史。

面板**恒在**。它测的是解决方案，不是仿真 —— 仿真器在不在场只决定两件事：世界能不能
被重置，以及有没有轨迹占用这类只有它算得出的量。

**「运行」跑的永远是当前画布。** 用例贡献的是初始指令、插话和评判标准；把它自带的画布
搬进来是另一个动作（`/cases/{id}/apply`），覆盖确认挂在那里。两件事原先绑在一起，于是
一个自带空画布的新用例，点一下「运行」就把用户手上的画布清空了。

## 主体是用例库，不是「打开一个文件」

进入一个用例的路径是**看现有的方案**：本机一份可编辑的用例库，加上市场里带 `test`
段的方案。文件导入还在，但它是导入路径，不是主体。

原先这里还有一套「勾几个场景跑一批」的端点（`/scenarios`、`/run`、`/progress`、
`/runs/{id}/collect`），已经删掉：那套的执行与评分都在仿真器卡片上，而卡片不该拥有
一个测试用例 —— 被测的 agent 能调到它，裁判也就住进了被测系统内部。场景现在描述的是
**世界**，不是一个可勾选的测试单位。

## 指标优先，模糊判定最少化

判定分两半，都不在仿真器里（裁判和被测系统必须不相交）：

* **算出来的**：`benchmark_metrics` 先把数算出来，`benchmark_case.check_targets` 拿数
  对默认目标做**确定性**判定，完全不经 LLM。
* **判出来的**：`benchmark_judge` 只判真的没有可算指标的那几样 —— 回答效果、参考流程的
  偏离是否合理、用例写的人话要求（而且判的时候手里拿着全部指标）。

「用户体验好不好」对大模型是个氛围词，判出来的分不可信也不可比；「懵逼时长 23.4 秒超没
超 15 秒」它判得可靠。

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

    运行结束之后要回看「它当时怎么想的、调了什么」，而那份记录在 `chat_history` 里
    按会话存。不在开始运行时记下来，事后就只能靠时间去猜是哪一段对话。
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
    """面板据此决定要不要出现 —— 现在恒为可用。

    从前这里是 `mcp_id is not None`：没装仿真器，入口整个消失。R1 上就是这样，一台
    部署了最新 agent-core 的机器人，设置里没有基准测试这一项，而且没有任何迹象说明
    为什么没有。

    但基准测试测的是**解决方案**，不是仿真。裁判从一开始就是纯函数，现在事实流在真机
    上也有了（`benchmark_facts`），所以仿真器在不在场只决定两件事：世界能不能重置，
    以及有没有轨迹占用这类只有它算得出的量。两件都不是「入口该不该存在」。

    `simulator` 照样返回，前端拿它提示当前是哪一侧、要不要走确认。
    """
    mcp_id = find_simulator()
    import benchmark_runner
    return {'available': True, 'mcp_id': mcp_id, 'simulator': mcp_id,
            # 这次运行会让什么动起来 —— **在点「运行」之前**就说。原先这份清单只在
            # 被拒绝的 409 里出现，于是面板在人做决定之前一个字都没提。
            'moving_cards': benchmark_runner.unsafe_cards(mcp_id),
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
    """把这个用例**自带的画布**载入成当前方案。

    这是唯一会覆盖画布的入口，所以「画布会被覆盖」的确认属于这里 —— 跑用例不会走到
    这条路。原先两件事是绑在一起的：想跑就得先载入，于是一个自带空画布的新用例，
    点一下「跑」就把现场的画布清空了。

    用例没有画布就拒绝，而不是载入一张空的。「没有可载入的东西」和「载入一张空画布」
    是两件完全不同的事，后者会毁掉用户手上的工作。
    """
    from api.solutions import LoadRequest, apply as apply_solution
    record = benchmark_store.get_case(case_id)
    if record is None:
        raise fastapi.HTTPException(status_code=404, detail='没有这个用例')
    if not ((record['payload'].get('canvas') or {}).get('cards') or []):
        raise fastapi.HTTPException(
            status_code=409,
            detail='这个用例没有自带画布，没有可载入的东西 —— 直接在当前画布上跑就行')
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
    """存过的快照。**只报告，不自动还原** —— 运行结束自动把画布换回去，会在用户正看着
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
    # 要跑哪个用例。**空 = 跑当前方案自带的那个。**
    #
    # 这个字段是这一轮修的那个 bug 的解药：原先想跑库里的用例，只能先「载入」它，而
    # 载入会把它自带的画布刷进来 —— 新建的用例画布是空的，于是点一下「跑」，当前画布
    # 就没了。跑从来不该改画布：**跑的永远是当前画布**，用例提供的只是指令、插话和
    # 评判标准。要把用例自带的画布搬进来，那是「载入」这个单独的动作。
    case_id: str = ''
    # 真机确认。`moving_cards` 是前端弹窗里那个人**看到并同意**的那一组设备，
    # 形如 `["mcp-123:loco", ...]`。服务端会重算一遍再比对 —— 见 `_check_confirmation`。
    confirm_moving_cards: list[str] | None = None


def card_key(card: dict) -> str:
    """一张会动的卡片在确认清单里的身份。设备名会变（改个昵称就变），`mcpId` 不会。"""
    return f"{card.get('mcpId', '')}:{card.get('tool', '')}"


def _check_confirmation(moving: list[dict], confirmed: list[str] | None) -> None:
    """真机上开跑前，确认必须对得上**服务端此刻看到的**那一组设备。

    比对而不是只看一个布尔位，是因为画布在「弹窗弹出」和「点开始运行」之间是可以改的：
    人看着「仿真器的 tts」按了确认，另一个标签页把卡片换成了真机的底盘，那个勾就为
    一组他从没看见过的设备背了书。所以清单由服务端重算，客户端送来的那份只用来核对。

    仿真器在场时 `moving` 为空，这里什么都不做 —— 行为和从前一字不差。
    """
    if not moving:
        return
    wanted = sorted(card_key(card) for card in moving)
    if confirmed is None:
        raise fastapi.HTTPException(status_code=409, detail={
            'error': '这次运行会驱动真实设备，需要现场确认',
            'needs_confirmation': True, 'moving_cards': moving})
    if sorted(str(entry) for entry in confirmed) != wanted:
        raise fastapi.HTTPException(status_code=409, detail={
            'error': '画布在确认之后变了，请重新确认',
            'needs_confirmation': True, 'moving_cards': moving})


def _case_to_run(case_id: str) -> dict | None:
    """要跑的那个 `test` 段 —— **不碰画布**。

    指名了就从本机用例库取，没指名就用当前方案自带的。两条路都只读 `test` 段：画布是
    现场那一张，用例只贡献指令、插话和评判标准。
    """
    import benchmark_case
    from api.solutions import loaded_case

    if not case_id:
        return loaded_case()
    record = benchmark_store.get_case(case_id)
    if record is None:
        raise fastapi.HTTPException(status_code=404, detail='没有这个用例')
    return benchmark_case.test_block(benchmark_case.migrate(record['payload']))


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

    case = _case_to_run(request.case_id)
    if not case:
        raise fastapi.HTTPException(
            status_code=409, detail='没有可跑的用例：在左边挑一个，或者给当前方案加一段 test')

    # 智能控制没开，事件进不了 collector（`collector.project_running` 那道闸），
    # 初始指令送进去也不会有人处理 —— 跑完会记一个 0 分，而那是基准测试在撒谎。
    import collector
    if not collector.project_running():
        raise fastapi.HTTPException(
            status_code=409, detail='还没「开始智能控制」——现在跑，指令送进去没人处理，只会记一个 0 分')

    problems = benchmark_case.validate({'test': case})
    if problems:
        raise fastapi.HTTPException(status_code=422, detail='；'.join(problems))

    # 仿真器不在场不是错误 —— 用例测的是解决方案，真机上照样能跑。
    mcp_id = find_simulator()
    readiness = await case_readiness(benchmark_case.requires({'test': case}))
    if not readiness.get('ok'):
        raise fastapi.HTTPException(status_code=409, detail={
            'error': '用例的依赖还不齐', 'readiness': readiness})

    # 安全闸：注入的文本和真实指令无法区分，画布上任何一张会动的真卡片都会真的动。
    # 仿真器在场时这组为空（它替所有会动的东西挡着），真机上非空，要人确认。
    _check_confirmation(benchmark_runner.unsafe_cards(mcp_id),
                        request.confirm_moving_cards)

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


@router.get('/runs/{run_id}/compare/{baseline_id}')
async def compare(run_id: str, baseline_id: str):
    """这次比上次，分数的差异站不站得住。

    面板据此决定**要不要**给涨跌结论。两个均值不同不等于有差别 —— LLM 是随机的，
    一次运行的分数是分布里的一个样本。「测不出显著差异」是一个结论，不是缺省值。
    """
    return benchmark_store.compare_runs(baseline_id, run_id)


@router.get('/runs/{run_id}')
async def run_detail(run_id: str):
    stored = benchmark_store.get_run(run_id)
    if stored is None:
        raise fastapi.HTTPException(status_code=404, detail='unknown run')
    return stored


@router.get('/runs/{run_id}/timeline')
async def run_timeline(run_id: str):
    """一次运行里到底发生了什么 —— 说了什么、想了什么、调了什么、世界怎么回应。

    分数只回答「好不好」，回答不了「哪儿坏了」。而这次运行的详情**留不住**：仿真器
    的世界下一次运行一开始就被重置，会话历史会被压缩。所以事实在写入时就存进了
    `benchmark_case.facts`，会话 id 存进了 `benchmark_run.session_id`。

    两条轨道**不强行合成一条**。agent 那侧的时间是墙钟，世界那侧是仿真时钟，两个钟
    相减没有意义 —— 各自归一到「从本轮开始起算的秒数」并排放，比编一个共同时间轴
    诚实。对齐基准两边都是运行开始：世界的第一条事件是 `scenario_load`（reset 那一刻），
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
        # 真机上没有 label（驱动的 ACP result 不带），目标只在派发参数里。
        # 不带上它，时间线里的「出发」就是一行没有目的地的字。
        'args': e.get('args') or {},
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
        # 定格的那一份优先：会话里的轮次运行结束之后还会被接着改写，现读一次，同一条
        # 记录过几分钟就换了个样子。老记录没有定格，只能现读。
        'agent': stored.get('agent_track') or _agent_track(
            stored.get('session_id', ''), stored.get('started_at'), stored.get('ended_at')),
    }


def _perf_turns(started, ended) -> list:
    """这段时间里每一轮的**真实**起止与逐次工具调用时刻。

    会话历史只记「这一轮是什么时候被写完的」，而一轮里的动作发生在那之前 —— 左栏
    于是整体比右栏晚一整轮的时长，两条轨道看着就是对不上（真机上左栏 +14.5s 的那次
    讲解，右栏在 +8.7s 就开讲了）。`perf_spans` 里有每次调用的真实起止，拿它对齐。
    """
    if not started:
        return []
    try:
        import perf_log
        return perf_log.turns_between(float(started) - 10, float(ended or started) + 600)
    except Exception:
        return []


def _tool_names(entry: dict) -> list:
    return [str(s['span'])[5:] for s in (entry.get('spans') or [])
            if str(s.get('span', '')).startswith('tool:')]


def _is_subsequence(needle: list, haystack: list) -> bool:
    iterator = iter(haystack)
    return all(name in iterator for name in needle)


def _match_perf(perf: list, calls: list) -> dict | None:
    """按**工具调用名的序列**配对，并消耗掉配上的那一条。

    不按 `trigger_text` 配：两边存的根本不是同一个串 —— perf 存原始的
    `<event source=…>`，会话存的是 `<status …>` 或格式化之后的通知。真机上一比就知道，
    而配不上的后果是整列悄悄退回「写完时」，看起来只是少了点精度，不像出错。

    也不按顺序配：一次运行期间可能有别的来源插进来，整列错位一格比没有时间更难发现。

    **是子序列，不是相等。** perf 记的是每一次派发，会话记的只是 LLM 自己发起的调用 ——
    `on_notify` 钩子自动播报的那次 speak 有 span，却不在会话的 tool_calls 里。真机上
    八轮里六轮因此配不上：

        会话  ['navigate_to_tag', 'speak', 'task_update']
        perf  ['navigate_to_tag', 'speak', 'speak', 'task_update']

    先找完全相等的（最可靠），再退到按顺序包含。
    """
    names = [c['name'] for c in calls]
    if not names:
        return None
    for index, entry in enumerate(perf):
        if _tool_names(entry) == names:
            return perf.pop(index)
    for index, entry in enumerate(perf):
        if _is_subsequence(names, _tool_names(entry)):
            return perf.pop(index)
    return None


def _timing_from_spans(entry: dict, started: float) -> dict:
    """把一轮的 spans 折算成「从本次运行开始起算」的秒数。"""
    spans = entry.get('spans') or []
    total = next((s for s in spans if s.get('span') == 'turn_total'), None)
    begin = (total or {}).get('start_ts') or min(
        (s.get('start_ts') for s in spans if s.get('start_ts')), default=None)
    calls = [{'name': str(s['span'])[5:],
              'at': round(float(s['start_ts']) - started, 1)}
             for s in spans
             if str(s.get('span', '')).startswith('tool:') and s.get('start_ts')]
    return {
        'at': round(float(begin) - started, 1) if begin else None,
        'callTimes': calls,
    }


def _with_times(calls: list, times: list) -> list:
    """给每次工具调用配上它真实发生的时刻。

    按**名字顺序**配：spans 记的是调用名与时刻，会话记的是调用名与参数，两边同名
    同序。名字对不上就不给时间 —— 与其配错一个，不如说不知道。
    """
    out, pool = [], list(times)
    for call in calls:
        match = next((i for i, t in enumerate(pool) if t['name'] == call['name']), None)
        if match is None:
            out.append(call)
            continue
        out.append({**call, 'at': pool.pop(match)['at']})
    return out


def _trigger_of(triggers: list) -> str:
    """这一轮是被什么唤醒的。

    优先挑**人说的那句话**：`<status …>` 是每轮都会附上的环境快照，几百字，拿它当
    「触发」既占满一屏又什么都没说。只有状态快照时才退回去，压成一行。
    """
    spoken = [t for t in triggers if not t.lstrip().startswith('<status')]
    if spoken:
        return spoken[0][:4000]
    return '（状态刷新）' if triggers else ''


def _agent_track(session_id: str, started, ended) -> list:
    """这次运行期间 agent 说了什么、调了什么。

    按**轮**取，不按消息取：一轮就是「被什么唤醒 → 想了什么 → 调了哪些工具」，而
    那正好是排查时要看的粒度。会话被清过或压缩掉就只能返回空 —— 与其编一段出来，
    不如让前端说「这次运行的对话记录已经没有了」。
    """
    if not session_id:
        return []
    try:
        import chat_history
        turns = chat_history.get_session_turns(session_id)
    except Exception:
        return []

    perf = _perf_turns(started, ended)
    window_from = float(started) - 5 if started else None
    window_to = float(ended) + 5 if ended else None

    track = []
    for index, turn in enumerate(turns):
        at = turn.get('started_at') or 0
        # 按**区间相交**判断，不按开始时刻。
        #
        # 一轮是被反复重写的：`save_turn` 在 turn_index 已存在时走 UPDATE，`created_at`
        # 保持不变、`updated_at` 往后走。所以一轮可能在运行**开始之前**就起了头，而
        # 它的内容是在运行**期间**写进去的。只看起始时刻，这一轮会被整个丢掉 ——
        # Orin6 上就是如此：会话明明活跃在运行窗口里（`ended_at` 落在窗口内），
        # 而每一轮的 `started_at` 都在窗口之前，于是 agent 那一栏是空的。
        until = turn.get('updated_at') or at
        if window_from is not None and until and until < window_from:
            continue          # 整轮都结束在运行之前
        if window_to is not None and at and at > window_to:
            continue          # 整轮都开始在运行之后
        says, calls, triggers = [], [], []
        for message in turn.get('messages') or []:
            role = message.get('role')
            content = message.get('content')
            if role == 'user' and isinstance(content, str) and content.strip():
                triggers.append(content.strip())
            elif role == 'assistant':
                if isinstance(content, str) and content.strip():
                    says.append(content.strip())
                for call in message.get('tool_calls') or []:
                    fn = (call.get('function') or {})
                    calls.append({'name': str(fn.get('name', '')).split('__')[-1],
                                  # 不在这里截断：界面要能展开看全文，砍在后端就永远
                                  # 看不到了。上限只挡住异常大的载荷。
                                  'args': str(fn.get('arguments', ''))[:4000]})
        # 时间取 `updated_at`，不取 `created_at`。
        #
        # 一轮的行是会被**覆盖**的：agent-core 重启后 `_turns` 从上一个会话重新载入，
        # 轮号从小往大重排，于是 `save_turn` 撞上几天前那一轮的行走了 UPDATE ——
        # `created_at` 留在几天前，内容却是刚刚写的。真机上因此排出了「第 11 轮
        # +-277189.7s」这种时间。最后写入的时刻才是这轮真正发生的时刻。
        # 时间优先用 spans 里的真实起止 —— 会话历史只有「写完」那一刻。
        spans_entry = _match_perf(perf, calls) if started else None
        timed = _timing_from_spans(spans_entry, float(started)) if spans_entry else {}
        written = turn.get('updated_at') or at
        offset = timed.get('at') if timed.get('at') is not None else (
            round(written - float(started), 1) if (started and written) else None)
        # 落在本轮运行窗口之外的时间**不显示**。
        #
        # 一轮可能在运行之前起头、在运行之后还在被追写（真机上见过一轮 `updated_at`
        # 在运行结束 26 分钟之后）。那种情况下两个时间戳都不在窗口里，排出来的
        # 「+1980s」放在一次 7 分钟的运行里，是个看起来精确的假数。宁可不给数。
        duration = (float(ended) - float(started)) if (started and ended) else None
        outside = (offset is not None and duration is not None
                   and not (-5 <= offset <= duration + 5))
        # 轮号从**这次运行**数起，不用会话里的序号 —— 打开的是一次运行的详情，
        # 第一轮却写着「第 11 轮」，读的人会以为前面漏了十轮。会话里的序号留在
        # `sessionTurn`，要和历史面板对照时还用得上。
        track.append({
            'turn': len(track),
            'sessionTurn': index,
            'at': None if outside else offset,
            # `exact` = 来自 spans 的真实时刻；`written` = 只知道这一轮写完的时刻，
            # 比它做的事晚一整轮。两者在界面上要看得出区别。
            'timing': ('outside' if outside
                       else ('exact' if timed.get('at') is not None else 'written')),
            'trigger': _trigger_of(triggers), 'says': says,
            'calls': _with_times(calls, timed.get('callTimes') or []),
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
