"""
modelhub_link.py — 本机大模型（phanthymotus-modelhub）的自发现。

ModelHub 是和 perception / actucore 并列的一层：一张 `llm` 卡片在本机跑 vLLM，所有
模型统一挂在一个 OpenAI 兼容网关后面。agent-core 不抄它的地址，而是存一个**引用**：

    'modelhub'           跟随 ModelHub 当前加载的那个模型
    'modelhub:<model>'   固定用这个模型；它没在服务时退回云端

每次请求前按引用现查（有几秒缓存），所以 ModelHub 换了端口、换了模型、重启过，
agent-core 都不用重新配置。查的是卡片自己的 `info` —— 它的 `serving` 段就是为这里
准备的：网关地址、在服务的模型和它一个会话能放多少 token、本机已下好的模型。

这里只读，不替用户加载模型：加载一个模型要占十几 GB 统一内存，是 ModelHub 运维页
上该由人做的决定，不该被一次 LLM 请求顺手触发。
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
    return bool(source) and (source == SOURCE_MODELHUB or source.startswith(SOURCE_MODELHUB + ':'))


def pinned_model(source: str | None) -> str:
    """'modelhub:qwen3-4b' → 'qwen3-4b'；跟随当前模型时为空串。"""
    if source and source.startswith(SOURCE_MODELHUB + ':'):
        return source.split(':', 1)[1]
    return ''


def client_entries(source: str, url: str, key: str, model: str, think_mode: bool) -> list[dict]:
    """把卡片 / 设置页上的一组 LLM 配置翻成 client.llm 列表。

    选了本机：本机那条在前、标 `source`，云端那条（填了才有）标 `role: fallback`
    —— client 先只问本机，本机不可用才问云端，而不是两边同时发、谁快用谁。
    没选本机：和以前一样就是一条云端配置。
    """
    cloud = {'url': url, 'key': key, 'model': model, 'think_mode': think_mode}
    if not is_local(source):
        return [cloud]
    local = {'source': source, 'model': pinned_model(source), 'think_mode': think_mode}
    return [local] + ([{**cloud, 'role': 'fallback'}] if url and key else [])


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
               'base_url': '', 'loaded': [], 'installed': [], 'error': ''}
        try:
            info = await fetch_info(m['url'])
            serving = info.get('serving') or {}
            hub['base_url'] = serving.get('base_url') or ''
            hub['loaded'] = list(serving.get('models') or [])
            hub['installed'] = list(serving.get('installed') or [])
            hub['state'] = info.get('state', '')
            hub['message'] = info.get('message', '')
        except Exception as e:      # noqa: BLE001 — 一个不回应的 hub 不该拖垮推理
            hub['error'] = f'{type(e).__name__}: {e}'
        out.append(hub)
    _cache.update(at=time.monotonic(), hubs=out)
    return out


def invalidate() -> None:
    _cache.update(at=0.0, hubs=None)


async def resolve(source: str) -> tuple[dict | None, str]:
    """引用 → ({base_url, model, context_tokens}, '')，或 (None, 为什么现在用不了)。"""
    want = pinned_model(source)
    found = await hubs()
    if not found:
        return None, '本机没有 ModelHub'
    reasons = []
    for hub in found:
        if hub.get('error'):
            reasons.append(f"{hub['name']} 没有回应（{hub['error']}）")
            continue
        for m in hub['loaded']:
            if want and m.get('id') != want:
                continue
            if m.get('state') != 'ready':
                reasons.append(f"本机模型 {m.get('title') or m.get('id')} 还没就绪（{m.get('state')}）")
                continue
            if not hub['base_url']:
                reasons.append(f"{hub['name']} 没报网关地址")
                continue
            return {'base_url': hub['base_url'], 'model': m['id'],
                    'title': m.get('title') or m['id'],
                    'context_tokens': m.get('context_tokens')}, ''
        if not hub['loaded']:
            reasons.append(hub.get('message') or f"{hub['name']} 没有加载模型")
        elif want and not any(m.get('id') == want for m in hub['loaded']):
            reasons.append(f'本机加载的不是 {want}')
    return None, '；'.join(reasons) or '本机模型不可用'


async def source_options(current: str = '') -> list[dict]:
    """卡片配置里「大模型」下拉的选项：云端、跟随本机当前模型、本机每个已下好的模型。"""
    opts = [{'value': SOURCE_CLOUD, 'label': '云端 API（下面填的地址）'}]
    found = await hubs(force=True)
    for hub in found:
        if hub.get('error'):
            opts.append({'value': SOURCE_MODELHUB, 'disabled': True,
                         'label': f"本机 {hub['name']}：没有回应"})
            continue
        ready = [m for m in hub['loaded'] if m.get('state') == 'ready']
        loaded_ids = {m.get('id') for m in hub['loaded']}
        if ready:
            m = ready[0]
            follow = f"本机模型 · 跟随当前加载的（{m.get('title') or m['id']}{_ctx(m)}）"
        elif hub['loaded']:
            m = hub['loaded'][0]
            follow = f"本机模型 · 跟随当前加载的（{m.get('title') or m['id']} 加载中）"
        else:
            follow = '本机模型 · 跟随当前加载的（现在没有加载，先去 ModelHub 卡片里启动一个）'
        opts.append({'value': SOURCE_MODELHUB, 'label': follow})
        for m in hub['installed']:
            mid = m.get('id')
            if m.get('unsupported'):
                opts.append({'value': f'{SOURCE_MODELHUB}:{mid}', 'disabled': True,
                              'label': f"本机模型 · {m.get('title') or mid}（{m['unsupported']}）"})
                continue
            state = '已加载' if mid in loaded_ids else '已下载，未加载'
            opts.append({'value': f'{SOURCE_MODELHUB}:{mid}',
                         'label': f"本机模型 · 固定 {m.get('title') or mid}（{state}）"})
        break       # 一台机器一个 ModelHub；多了也只认第一个，和 resolve 一致
    if not found:
        opts.append({'value': SOURCE_MODELHUB, 'disabled': True,
                     'label': '本机模型：这台机器没有装 ModelHub'})
    if current and not any(o['value'] == current for o in opts):
        opts.append({'value': current, 'label': f'{current}（已不在本机）'})
    return opts


def _ctx(m: dict) -> str:
    n = m.get('context_tokens')
    return f'，上下文 {round(n / 1000)}k' if n else ''
