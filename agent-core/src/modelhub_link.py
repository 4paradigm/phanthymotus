"""
modelhub_link.py — 本机大模型（phanthymotus-modelhub）的自发现。

ModelHub 是和 perception / actucore 并列的一层：一张 `llm` 卡片在本机跑 vLLM，所有
模型统一挂在一个 OpenAI 兼容网关后面。

决策核心的「大模型」只有两个选择：云端，或本机。选本机时 agent-core **不选具体模型**
—— 用的永远是 ModelHub 当前加载的那一个。换模型是 ModelHub 模型管理页的事：加载一个模型要占
十几 GB 统一内存，那是 ModelHub 运维页上由人做的决定，不该在两个地方各有一个开关。

所以存的是引用 'modelhub' 而不是地址，每次请求前现查（有几秒缓存）：ModelHub 换了
端口、换了模型、重启过，agent-core 都不用重新配置。查的是卡片自己的 `info` —— 它的
`serving` 段就是为这里准备的：网关地址、在服务的模型、它一个会话能放多少 token。
"""

from __future__ import annotations

import json
import time

import httpx

import config

SOURCE_CLOUD = 'cloud'
SOURCE_MODELHUB = 'modelhub'

CACHE_TTL_S = 5.0          # 每次 LLM 请求都要解析一次引用，不能每次都打一遍 MCP
INFO_TIMEOUT_S = 3.0       # info 是毫秒级的；不回应就当它不在，别拖住一次推理

_cache: dict = {'at': 0.0, 'hubs': None}


def is_local(source: str | None) -> bool:
    return source == SOURCE_MODELHUB


def client_entries(source: str, url: str, key: str, model: str, think_mode: bool) -> list[dict]:
    """把卡片 / 设置页上的一组 LLM 配置翻成 client.llm 列表。

    选本机就**只有**本机那一条。云端的 URL / Key 照样存在 services.llm 里，切回云端
    不用重填 —— 但不拿来兜底：选本机的人可能就是不想让对话离开这台机器，悄悄换到
    云端的事不该发生。本机用不了时请求直接失败，原因写进日志和活动流。
    """
    if is_local(source):
        return [{'source': source, 'model': '', 'think_mode': think_mode}]
    return [{'url': url, 'key': key, 'model': model, 'think_mode': think_mode}]


def _hub_entries() -> list[dict]:
    mcps = (config.main.get('services') or {}).get('mcp') or []
    out = []
    for m in mcps:
        if m.get('category') != 'modelhub' or not m.get('url'):
            continue
        names = {t.get('name') if isinstance(t, dict) else t for t in (m.get('tools') or [])}
        if 'llm' in names:
            out.append(m)
    return out


async def _fetch_info(url: str) -> dict:
    payload = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
               'params': {'name': 'llm', 'arguments': {'action': 'info'}}}
    async with httpx.AsyncClient(timeout=INFO_TIMEOUT_S) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()
        result = (r.json() or {}).get('result') or {}
    for item in result.get('content') or []:
        if isinstance(item, dict) and item.get('type') == 'text':
            return json.loads(item.get('text') or '{}')
    return {}


# 测试替换这个就能不起 HTTP
fetch_info = _fetch_info


async def hubs(force: bool = False) -> list[dict]:
    """本机每个 ModelHub 的现状。查不到的那个照样列出来，带上 error。"""
    now = time.monotonic()
    if not force and _cache['hubs'] is not None and now - _cache['at'] < CACHE_TTL_S:
        return _cache['hubs']
    # 不加锁：并发的两次请求各查一次 info 无害，而模块级 asyncio.Lock 会绑死在
    # 第一次用它的事件循环上
    out = []
    for m in _hub_entries():
        hub = {'mcp_id': m.get('id', ''), 'name': m.get('name') or 'ModelHub',
               'base_url': '', 'loaded': [], 'error': ''}
        try:
            info = await fetch_info(m['url'])
            serving = info.get('serving') or {}
            hub['base_url'] = serving.get('base_url') or ''
            hub['loaded'] = list(serving.get('models') or [])
            # 本机已下载的模型；老版本 ModelHub 不报这一项，记为 None（说不清有没有）
            hub['installed'] = (list(serving.get('installed') or [])
                                if 'installed' in serving else None)
            hub['state'] = info.get('state', '')
            hub['message'] = info.get('message', '')
        except Exception as e:      # noqa: BLE001 — 一个不回应的 hub 不该拖垮推理
            hub['error'] = f'{type(e).__name__}: {e}'
        out.append(hub)
    _cache.update(at=time.monotonic(), hubs=out)
    return out


def invalidate() -> None:
    _cache.update(at=0.0, hubs=None)


async def status(force: bool = False) -> dict:
    """本机模型现在能不能用、用的是哪个 —— 卡片配置里「本机」那一栏显示的就是它。

    state：ready 能用；loading 正在下载或加载；error 启动失败；idle 没有加载模型；
           unreachable ModelHub 没回应；absent 这台机器没有 ModelHub。
    message 是给人看的一句话，说清楚现在怎样、要做什么。
    """
    found = await hubs(force=force)
    if not found:
        return {'state': 'absent', 'message': '这台机器没有 ModelHub，用不了本机模型。'}
    hub = found[0]          # 一台机器一个 ModelHub；多了也只认第一个
    if hub.get('error'):
        return {'state': 'unreachable',
                'message': f"ModelHub 没有回应（{hub['error']}）。"}
    ready = [m for m in hub['loaded'] if m.get('state') == 'ready']
    if ready and hub['base_url']:
        m = ready[0]
        return {'state': 'ready', 'model': m['id'], 'title': m.get('title') or m['id'],
                'context_tokens': m.get('context_tokens'), 'base_url': hub['base_url'],
                'message': ''}
    # 启动失败不是「在加载」：以前这里一律算 loading，于是决策核心显示一个黄点和一段
    # vLLM 堆栈，开启智能控制还要干等 15 分钟才报失败
    failed = [m for m in hub['loaded'] if m.get('state') == 'error']
    if failed or hub.get('state') == 'error':
        m = (failed or hub['loaded'] or [{}])[0]
        return {'state': 'error', 'title': m.get('title') or m.get('id') or '',
                'message': hub.get('message') or '本机模型启动失败，详情见模型管理。'}
    if hub['loaded'] or hub.get('state') == 'loading':
        m = (hub['loaded'] or [{}])[0]
        name = m.get('title') or m.get('id') or ''
        return {'state': 'loading', 'title': name,
                'message': hub.get('message') or f'{name} 正在加载。'}
    # 没加载模型时分两种：一个都没下载（得先下载），和下载了没启动（启动就行）
    installed = hub.get('installed')
    if installed == []:
        return {'state': 'idle', 'downloaded': 0,
                'message': 'ModelHub 里还没有模型。在侧边栏「模型」里打开模型管理，下载并启动一个。'}
    if installed:
        return {'state': 'idle', 'downloaded': len(installed),
                'message': f'已下载 {len(installed)} 个模型，还没有启动。在模型管理里启动一个。'}
    return {'state': 'idle',
            'message': 'ModelHub 还没有加载模型。在侧边栏「模型」里打开模型管理，启动一个模型。'}


async def resolve(source: str) -> tuple[dict | None, str]:
    """引用 → ({base_url, model, title, context_tokens}, '')，或 (None, 为什么现在用不了)。"""
    st = await status()
    if st['state'] != 'ready':
        return None, st['message']
    return {k: st[k] for k in ('base_url', 'model', 'title', 'context_tokens')}, ''


async def llm_readiness(force: bool = False) -> tuple[str, str]:
    """决策核心现在有没有大模型可用：('running' | 'loading' | 'error', 给人看的一句话)。

    开启智能控制时问它：以前大模型没配好也照样启动成功，机器人只是不说话 ——
    要停掉、配好、再开一次才能用。现在没配好就不让启动，并说清楚缺什么。

    本机模型还在加载算 loading，不算 error：开机自启时模型通常正在加载，那时把
    启动判失败就等于开机自启永远起不来。ModelHub 没回应也按 loading 等它：
    两个容器一起开机，agent-core 往往先起来。等不到时启动流程自己有 15 分钟上限。
    """
    entries = (config.main.get('client') or {}).get('llm') or []
    if any(is_local(e.get('source')) for e in entries):
        st = await status(force=force)
        if st['state'] == 'ready':
            return 'running', ''
        if st['state'] == 'loading':
            return 'loading', st.get('message') or '本机模型正在加载'
        if st['state'] == 'unreachable':
            return 'loading', '等待 ModelHub 响应…'
        if st['state'] == 'error':
            return 'error', f"本机模型 {st.get('title') or ''} 启动失败：{st.get('message')}"
        if st['state'] == 'absent':
            return 'error', ('大模型选了本机，但这台机器没有 ModelHub。'
                             '在决策核心的设置里改用云端，或者先安装 ModelHub。')
        return 'error', st.get('message') or '本机模型现在用不了'
    if any(e.get('url') and e.get('key') for e in entries):
        return 'running', ''
    return 'error', '还没有配置大模型。打开决策核心的设置，选云端并填好 URL 和 Key，或者选本机。'
