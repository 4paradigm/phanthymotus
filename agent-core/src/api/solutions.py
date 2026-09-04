"""
api/solutions.py — 解决方案（Solution）打包 / 发布 / 载入。

一个解决方案是"这台机器上跑得起来的整套配置"，可打包的块：

    canvas          画布卡片 + 数据连线 + 执行连线 + 每张卡的 tool_config（必选）
    skills          当前激活、且已在技能广场上架的技能
    prompt.identity resource/memory/identity.md
    prompt.system   resource/memory/prompt_system.md
    prompt.memory   Memory Core 中的 Agent 长期记忆
    tasks           task_store 里的活跃任务（goal / check_cron / metadata）

包体格式（formatVersion 1）：

    {
      "formatVersion": 1,
      "coreVersion": "release.xxx",
      "devices": [{"ref": "d0", "serverName": "g1-device-bundle",
                   "category": "driver", "driverId": "unitree-g1",
                   "registryImage": "driver-unitree-g1",
                   "image": "…:release.260529.x", "name": "…", "port": 15701}],
      "canvas": {"cards": [{"id","deviceRef","toolName","x","y",
                            "topicIn","topicOut"}],
                 "connections": [], "execConnections": [], "transform": {},
                 "toolConfigs": {"d0:tool": {...}, "d0:tool:<cardId>": {...}},
                 "redactedFields": ["d0:tool:api_key"]},
      "skills": [{"slug","version","name","icon"}],
      "prompt": {"identity": "...", "system": "...", "memory": "..."},
      "tasks":  [{"goal","check_cron","metadata"}]
    }

为什么卡片存 deviceRef 而不是 mcpId：mcpId 是本机在注册设备时生成的
`mcp-<unix秒>`（见 api/mcp_manage.py:mcp_add），换一台机器一定对不上。
deviceRef 指向 devices[]，devices 用 MCP initialize 返回的 server_name
（如 `g1-device-bundle`）作为稳定标识，载入时再映射回本机 mcpId。

敏感字段脱敏是"声明式"的：卡片自己在 configSchema 的属性上标
`"x-sensitive": true`（或沿用已有的 `"format": "password"`），打包时把值清空
并把路径记进 redactedFields。反过来，固定的出厂密码这类"要打码但不是秘密"的
字段可以标 `"x-sensitive": false` 显式豁免。未声明的字段不会被自动清空 ——
打包接口会把候选清单给前端，作者可以通过 extraRedact 手工追加。

同样会被清空的是"本机专属"字段（`format: channel-select` /
`audio-input-device`）：它们存的是本机 channel id 或声卡设备名，换机器不是
泄密而是根本指不到东西，载入方必须重选。
"""

import os
import pathlib
import time
from typing import Any, Optional

import fastapi
from pydantic import BaseModel

import agent_memory
import config

router = fastapi.APIRouter(prefix='/solutions', tags=['solutions'])

FORMAT_VERSION = 1

# 可打包的块名。canvas 必选，其余由用户勾选。
BLOCK_CANVAS = 'canvas'
BLOCK_SKILLS = 'skills'
BLOCK_TASKS = 'tasks'
PROMPT_BLOCKS = ('prompt.identity', 'prompt.system', 'prompt.memory')

_CURRENT_KEY = 'solution_current'


# ── RC 代理（复用 api/skills.py 的 URL / token 约定）────────────────────────

def _get_rc_url() -> str:
    from api.skills import _get_rc_url as rc_url
    return rc_url()


def _get_rc_token(request: fastapi.Request) -> Optional[str]:
    """从请求头 X-RC-Token 中获取 RC bearer token（与技能一致）。"""
    return request.headers.get('x-rc-token')


def _rc_error(status: int, error: str) -> str:
    """Resource Center 报错中文化。鉴权类的原文是 Unauthorized，用户看不懂。"""
    from api.account import normalize_rc_error
    return normalize_rc_error(status, error)


_UNAUTHORIZED = '未登录或登录已过期，请在「我的」中登录'


async def _rc_request(method: str, path: str, token: Optional[str] = None,
                      json_body: Any = None, timeout: float = 20) -> dict:
    """向 Resource Center 发一个请求。

    返回 {'ok', 'status', 'data'|'error', 'payload'}。`data` 是 RC 的 `data`
    字段（大多数端点的约定），`payload` 是整个响应体 —— 登录与 session 这类
    端点把结果放在顶层（`{ok, token, userId}` / `{ok, user}`），调用方要自己从
    payload 里取。
    """
    url = f'{_get_rc_url()}{path}'
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.request(method, url, headers=headers, json=json_body)
            try:
                data = resp.json()
            except Exception:
                return {'ok': False, 'status': resp.status_code, 'payload': {},
                        'error': f'Resource Center 返回了非 JSON 响应 (HTTP {resp.status_code})'}
            if resp.status_code in (200, 201) and data.get('ok'):
                return {'ok': True, 'status': resp.status_code,
                        'data': data.get('data'), 'payload': data}
            return {'ok': False, 'status': resp.status_code, 'payload': data,
                    'error': data.get('error', f'请求失败 (HTTP {resp.status_code})')}
    except Exception as e:
        return {'ok': False, 'status': 502, 'payload': {},
                'error': f'无法连接 Resource Center: {e}'}


def _core_version() -> str:
    """当前 Agent Core 的镜像 tag（仅作记录，载入时不做比较）。"""
    try:
        return pathlib.Path('/work/VERSION').read_text().strip()
    except Exception:
        return os.environ.get('IMAGE_TAG', '')


# ── 本机状态读取 ─────────────────────────────────────────────────────────────

def _event_skills():
    """event.skills 模块本身。

    不能写 `import event.skills as skills_mod`：event/__init__.py 会把 `skills`
    这个属性重新绑定成 Tools 实例，于是 `as` 拿到的是那个实例而不是模块
    （api/skills.py 早就踩过，那里也是这么绕的）。
    """
    import sys
    import event.skills  # noqa: F401  —— 确保模块已被导入
    return sys.modules['event.skills']


def _mcp_list() -> list:
    return list(config.main.get('services', {}).get('mcp', []))


def _driver_manifest() -> list:
    return list(config.main.get('drivers') or [])


def _layout() -> dict:
    return config.main.get('canvas_layout', {}) or {}


def _tool_schema(mcp: dict, tool_name: str) -> dict:
    """某个工具的 configSchema（工具可能是纯字符串形式，那就没有 schema）。"""
    for tool in mcp.get('tools', []) or []:
        if isinstance(tool, dict) and tool.get('name') == tool_name:
            return tool.get('configSchema') or {}
    return {}


def _sensitive_props(config_schema: dict) -> set:
    """schema 里声明为敏感的属性名。

    判定顺序：
      1. `x-sensitive: true`  → 敏感（本文件引入的显式声明）
      2. `x-sensitive: false` → 不敏感，即使 format 是 password。给"输入框要打码
         但值本身不是秘密"的字段用，例如固定的出厂密码 —— 清掉只会让载入方
         被迫重新敲一遍同一个默认值。
      3. `format: password`   → 敏感（driver 侧早就在用的写法，见
         web/js/sidebar.js 用它渲染 password input）

    第 3 条是安全默认：没表态的密码框一律当秘密，否则已有驱动的密钥字段会
    原样进包体。要豁免就按第 2 条显式写出来。
    """
    props = (config_schema or {}).get('properties') or {}
    out = set()
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        declared = spec.get('x-sensitive')
        if declared is True:
            out.add(name)
        elif declared is False:
            continue
        elif spec.get('format') == 'password':
            out.add(name)
    return out


# 值只在本机有意义的字段格式：channel-select 存的是本机 channel_configs 里的
# id，audio-input-device 存的是本机声卡的设备名/序号。带到另一台机器上不是
# 泄密，而是根本指不到东西，所以同样清空并让载入方重选。
_LOCAL_ONLY_FORMATS = ('channel-select', 'audio-input-device')


def _local_only_props(config_schema: dict) -> set:
    """schema 里"值只对本机有效"的属性名。"""
    props = (config_schema or {}).get('properties') or {}
    return {name for name, spec in props.items()
            if isinstance(spec, dict) and spec.get('format') in _LOCAL_ONLY_FORMATS}


def _must_clear_props(config_schema: dict) -> tuple[set, set]:
    """(敏感字段, 本机专属字段) —— 两类都必须在打包时清空。"""
    return _sensitive_props(config_schema), _local_only_props(config_schema)


def _port_from_url(url: str) -> Optional[int]:
    try:
        from urllib.parse import urlparse
        return urlparse(url).port
    except Exception:
        return None


def _driver_entry_for(mcp: dict) -> dict:
    """把 MCP 条目对应回 drivers manifest 里的那一项。

    先按 mcp_url 完全相同匹配（drivers.py 给 perception/actucore 写死了
    mcp_url），退化到按端口匹配（硬件驱动的 manifest 只有 port）。
    """
    url = mcp.get('url', '')
    port = _port_from_url(url)
    manifest = _driver_manifest()
    for d in manifest:
        if d.get('mcp_url') and d['mcp_url'] == url:
            return d
    if port:
        for d in manifest:
            if d.get('port') == port or d.get('host_port') == port:
                return d
    return {}


def _device_descriptor(ref: str, mcp: dict) -> dict:
    """构造包体里的一条 devices 记录。"""
    driver = _driver_entry_for(mcp)
    server_name = mcp.get('server_name') or mcp.get('name') or ''
    return {
        'ref':           ref,
        'serverName':    server_name,
        'category':      mcp.get('category') or driver.get('category') or 'driver',
        'driverId':      driver.get('id', ''),
        'registryImage': driver.get('registry_image', ''),
        'image':         driver.get('image', ''),
        'name':          driver.get('name') or mcp.get('name') or server_name,
        'port':          _port_from_url(mcp.get('url', '')),
        # 机型：只有硬件驱动才有。适用机型（robotTypes）由它推出来，不让作者手填 ——
        # 手填的话方案会声称支持一台本机根本没有驱动的机器。
        'provider':      driver.get('provider', ''),
        'model':         driver.get('model', ''),
    }


def _robot_models(devices: list) -> list:
    """本机这套画布对应的机型列表 = 画布上硬件驱动的 model，去重保序。

    没有硬件驱动（纯 core + perception 的方案）就返回空 —— 前端显示 None。
    """
    models: list = []
    for d in devices:
        if d.get('category') != 'driver':
            continue
        model = (d.get('model') or '').strip()
        if model and model not in models:
            models.append(model)
    return models


def _canvas_devices() -> tuple[list, dict, list]:
    """画布上用到的设备。

    返回 (devices, mcp_id → ref, 无法解析的 mcp_id 列表)。
    """
    layout = _layout()
    mcps = {m.get('id'): m for m in _mcp_list()}
    devices: list = []
    ref_of: dict = {}
    unresolved: list = []

    for card in layout.get('cards', []) or []:
        mcp_id = card.get('mcpId', '')
        if not mcp_id or mcp_id in ref_of:
            continue
        mcp = mcps.get(mcp_id)
        if not mcp:
            unresolved.append(mcp_id)
            continue
        ref = f'd{len(devices)}'
        ref_of[mcp_id] = ref
        devices.append(_device_descriptor(ref, mcp))

    return devices, ref_of, unresolved


# ── 打包 ────────────────────────────────────────────────────────────────────

def _prompt_paths() -> dict:
    """Prompt paths, including the compatibility mirror for long-term memory."""
    import api.agent_definition as ad
    return {
        'identity': ad._IDENTITY_PATH,
        'system':   ad._SYSTEM_PATH,
        'memory':   ad._memory_path(),
    }


def _prompt_content(key: str, path: pathlib.Path) -> str:
    if key == 'memory':
        return agent_memory.snapshot().text
    return path.read_text()


def _prompt_exists(key: str, path: pathlib.Path) -> bool:
    if key == 'memory':
        return bool(agent_memory.snapshot().text)
    return path.exists()


def _config_field_paths(ref_of: dict) -> list:
    """画布上每个卡片配置字段的路径，标出哪些按 schema 声明是敏感的。

    前端用它渲染"将被清空的字段"清单：声明过的默认勾选且不可取消，其余留给
    作者手工勾选（extra_redact）—— 声明式脱敏管不到还没加标记的老驱动。
    """
    from api.canvas import all_tool_configs

    mcps = {m.get('id'): m for m in _mcp_list()}
    out = []
    for key, value in all_tool_configs().items():
        parts = key.split(':')
        if len(parts) < 2:
            continue
        mcp_id, tool_name = parts[0], parts[1]
        instance_id = parts[2] if len(parts) > 2 else ''
        ref = ref_of.get(mcp_id)
        if not ref or not isinstance(value, dict):
            continue
        sensitive, local_only = _must_clear_props(_tool_schema(mcps.get(mcp_id, {}), tool_name))
        pkey = f'{ref}:{tool_name}:{instance_id}' if instance_id else f'{ref}:{tool_name}'
        for prop in value:
            out.append({
                'path':       f'{pkey}:{prop}',
                'tool':       tool_name,
                'prop':       prop,
                'instanceId': instance_id,
                'sensitive':  prop in sensitive,
                'localOnly':  prop in local_only,
            })
    return sorted(out, key=lambda x: x['path'])


def _pack_canvas(ref_of: dict, extra_redact: set) -> tuple[dict, list]:
    """打包画布。返回 (canvas 块, 被清空的字段路径)。"""
    from api.canvas import all_tool_configs

    layout = _layout()
    mcps = {m.get('id'): m for m in _mcp_list()}

    cards = []
    for card in layout.get('cards', []) or []:
        ref = ref_of.get(card.get('mcpId', ''))
        if not ref:
            continue  # 设备已不在注册表里，连带这张卡一起丢掉
        cards.append({
            'id':        card.get('id', ''),
            'deviceRef': ref,
            'toolName':  card.get('toolName', ''),
            'x':         card.get('x', 0),
            'y':         card.get('y', 0),
            'topicIn':   card.get('topicIn') or [],
            'topicOut':  card.get('topicOut') or [],
        })

    card_ids = {c['id'] for c in cards}
    connections = [c for c in layout.get('connections', []) or []
                   if c.get('fromCardId') in card_ids and c.get('toCardId') in card_ids]
    exec_connections = []
    for c in layout.get('execConnections', []) or []:
        if c.get('fromCardId') not in card_ids or c.get('toCardId') not in card_ids:
            continue
        c = dict(c)
        # execConnections 缓存了目标卡的 mcpId；换机器后必须重新解析，
        # 留着会让载入侧拿到一个不存在的设备 id。
        c.pop('toMcpId', None)
        exec_connections.append(c)

    tool_configs: dict = {}
    redacted: list = []
    for key, value in all_tool_configs().items():
        parts = key.split(':')
        if len(parts) < 2:
            continue
        mcp_id, tool_name = parts[0], parts[1]
        instance_id = parts[2] if len(parts) > 2 else ''
        ref = ref_of.get(mcp_id)
        if not ref:
            continue
        if instance_id and instance_id not in card_ids:
            continue  # 属于已删除卡片的实例配置
        pkey = f'{ref}:{tool_name}:{instance_id}' if instance_id else f'{ref}:{tool_name}'

        if not isinstance(value, dict):
            tool_configs[pkey] = value
            continue

        sensitive, local_only = _must_clear_props(_tool_schema(mcps.get(mcp_id, {}), tool_name))
        packed = {}
        for prop, val in value.items():
            path = f'{pkey}:{prop}'
            if prop in sensitive or prop in local_only or path in extra_redact:
                packed[prop] = '' if isinstance(val, str) or val is None else type(val)()
                redacted.append(path)
            else:
                packed[prop] = val
        tool_configs[pkey] = packed

    return {
        'cards':           cards,
        'connections':     connections,
        'execConnections': exec_connections,
        'transform':       layout.get('transform', {}) or {},
        'toolConfigs':     tool_configs,
        'redactedFields':  sorted(redacted),
    }, sorted(redacted)


class PackInclude(BaseModel):
    canvas: bool = True                 # 必选，传 False 也会被拒
    skills: list[str] = []              # 要打包的技能 slug
    prompt: list[str] = []              # identity | system | memory
    tasks:  bool = False


class PackRequest(BaseModel):
    include: PackInclude = PackInclude()
    extra_redact: list[str] = []        # 作者手工追加要清空的字段路径


async def _market_skill_slugs(slugs: list[str]) -> tuple[list, list]:
    """把技能 slug 分成 (广场上已上架的, 广场上没有的)。

    只认公开可见（已上架）的技能：解决方案要能被别人载入，包体里引用一个
    只有作者自己能看到的草稿技能，别人载入时必然装不上。所以这里刻意不带
    RC token —— 带上会把作者的私有草稿也判成"有"。
    """
    from api.skills import fetch_rc_skill
    available, missing = [], []
    for slug in slugs:
        result = await fetch_rc_skill(slug, None)
        if result.get('ok') and (result['data'].get('status') == 'published'):
            available.append(slug)
        else:
            missing.append(slug)
    return available, missing


async def _build_payload(req: PackRequest, token: Optional[str]) -> dict:
    """构造包体。返回 {'ok': True, ...} 或 {'ok': False, 'error', 'detail'}。"""
    skills_mod = _event_skills()

    if not req.include.canvas:
        return {'ok': False, 'error': '画布是解决方案的必选部分，不能取消勾选'}

    devices, ref_of, unresolved = _canvas_devices()
    if not devices:
        return {'ok': False, 'error': '画布为空（或卡片引用的设备都已注销），无法打包'}
    if unresolved:
        return {'ok': False,
                'error': '画布上有卡片引用了未注册的设备，请先在画布上删除这些卡片',
                'detail': unresolved}

    includes = [BLOCK_CANVAS]
    payload: dict = {
        'formatVersion': FORMAT_VERSION,
        'coreVersion':   _core_version(),
        'devices':       devices,
    }

    canvas_block, redacted = _pack_canvas(ref_of, set(req.extra_redact))
    payload['canvas'] = canvas_block

    # 技能：必须是当前激活的，且已在技能广场上架
    if req.include.skills:
        installed = {s['slug']: s for s in skills_mod.installed_skills()}
        not_active = [s for s in req.include.skills
                      if not installed.get(s, {}).get('active')]
        if not_active:
            return {'ok': False,
                    'error': '只能打包当前已激活的技能',
                    'detail': not_active}
        available, off_market = await _market_skill_slugs(req.include.skills)
        if off_market:
            return {'ok': False,
                    'error': '只能保存技能广场中已上架的技能，请先把这些技能发布并通过审核',
                    'detail': off_market}
        payload['skills'] = [{
            'slug':    installed[s]['slug'],
            'name':    installed[s].get('name', ''),
            'icon':    installed[s].get('icon'),
            'version': installed[s].get('version', ''),
        } for s in available]
        includes.append(BLOCK_SKILLS)

    # Prompt：三项独立勾选
    if req.include.prompt:
        paths = _prompt_paths()
        prompt: dict = {}
        for key in req.include.prompt:
            if key not in paths:
                return {'ok': False, 'error': f'未知的 prompt 块: {key}'}
            path = paths[key]
            if not _prompt_exists(key, path):
                return {'ok': False, 'error': f'{path} 不存在，无法打包 prompt.{key}'}
            prompt[key] = _prompt_content(key, path)
            includes.append(f'prompt.{key}')
        payload['prompt'] = prompt

    # 任务
    if req.include.tasks:
        import task_store
        task_store.load_all()
        payload['tasks'] = [{
            'goal':       t.goal,
            'check_cron': t.check_cron,
            'metadata':   t.metadata or {},
        } for t in task_store.active_tasks()]
        includes.append(BLOCK_TASKS)

    return {
        'ok': True,
        'payload':         payload,
        'includes':        includes,
        'requiredDrivers': devices,
        'needsConfig':     redacted,
        'coreVersion':     payload['coreVersion'],
    }


# ── 设备解析（载入前） ───────────────────────────────────────────────────────

def _resolve_devices(devices: list) -> dict:
    """把包体的 devices 映射到本机。

    分三类：
      matched   已注册 MCP，可直接用（ref → mcpId）
      installable  drivers manifest 里有对应镜像，但还没起容器 → 可一键安装
      missing   本机连镜像都没有，必须先同步/安装驱动
    """
    mcps = _mcp_list()
    manifest = _driver_manifest()

    mapping: dict = {}
    matched, installable, missing = [], [], []

    for dev in devices:
        server_name = dev.get('serverName', '')
        registry_image = dev.get('registryImage', '')
        driver_id = dev.get('driverId', '')
        port = dev.get('port')

        # 1) 已注册 MCP：优先按 server_name（稳定），退化到端口，再退化到 name。
        #    name 兜底是给 agentcore / channel 这种内部伪设备用的：它们 url 为空、
        #    没有端口也没有 manifest 条目，一旦 server_name 对不上就会被误判成
        #    "本机缺驱动"，而实际上每台机器启动时都会注册它们。
        mcp = next((m for m in mcps if server_name and m.get('server_name') == server_name), None)
        if not mcp and port:
            mcp = next((m for m in mcps if _port_from_url(m.get('url', '')) == port), None)
        if not mcp and dev.get('name'):
            mcp = next((m for m in mcps if m.get('name') == dev['name']), None)
        if mcp:
            mapping[dev.get('ref')] = mcp.get('id')
            entry = _driver_entry_for(mcp)
            matched.append({**dev, 'mcpId': mcp.get('id'), 'mcpName': mcp.get('name', ''),
                            'localDriverId': entry.get('id', ''),
                            'localImage': entry.get('image', '')})
            continue

        # 2) 驱动清单里有：只是没部署 / 没注册
        entry = next((d for d in manifest
                      if (registry_image and d.get('registry_image') == registry_image)
                      or (driver_id and d.get('id') == driver_id)), None)
        if entry:
            installable.append({**dev, 'localDriverId': entry.get('id'),
                                'localImage': entry.get('image', '')})
        else:
            missing.append(dev)

    return {'mapping': mapping, 'matched': matched,
            'installable': installable, 'missing': missing}


# ── 版本对齐 ────────────────────────────────────────────────────────────────

def _image_tag(image_ref: str) -> str:
    """镜像 ref 的 tag 部分（没有 tag 则返回空串）。"""
    if not image_ref or ':' not in image_ref:
        return ''
    repo, tag = image_ref.rsplit(':', 1)
    return '' if '/' in tag else tag     # 端口号形如 host:5000/img，不是 tag


def _align_image_for(dev: dict) -> str:
    """把方案记录的 tag 套到本机的镜像仓库上。

    只换 tag、保留本机 repo：包体可能来自另一个 registry（作者的私有仓库），
    照搬整个 ref 会拉不到镜像；而"对齐版本"要的本来也只是 tag 一致。
    """
    package_tag = _image_tag(dev.get('image', ''))
    local_image = dev.get('localImage', '')
    if not package_tag or not local_image:
        return ''
    local_repo = local_image.rsplit(':', 1)[0] if _image_tag(local_image) else local_image
    return f'{local_repo}:{package_tag}'


async def _augment_versions(resolved: dict) -> None:
    """给 matched / installable 的设备补上版本对齐信息（原地修改）。

    只在 preflight 里调用：要问 Docker 当前跑的是哪个镜像，比较慢，而 apply
    只需要 ref → mcpId 的映射。
    """
    import asyncio as _asyncio
    from api.drivers import _get_status_sync, _load_manifest

    manifest = {d.get('id'): d for d in _load_manifest()}
    loop = _asyncio.get_event_loop()

    for dev in resolved['matched'] + resolved['installable']:
        dev['packageImage'] = dev.get('image', '')
        dev['packageTag'] = _image_tag(dev.get('image', ''))
        dev['alignImage'] = _align_image_for(dev)
        dev['runningImage'] = ''
        dev['runningTag'] = ''
        dev['aligned'] = None            # None = 问不到（没有 Docker / 容器没起）

        driver_id = dev.get('localDriverId') or ''
        if not driver_id:
            continue
        entry = manifest.get(driver_id, {})
        try:
            status = await loop.run_in_executor(
                None, _get_status_sync, driver_id, entry.get('container_name', ''))
        except Exception:
            continue
        running = status.get('running_image', '') or ''
        dev['runningImage'] = running
        dev['runningTag'] = _image_tag(running)
        if dev['alignImage'] and running:
            dev['aligned'] = (running == dev['alignImage'])


def _misaligned(resolved: dict) -> list:
    """需要重新部署才能对上方案记录版本的设备。

    `aligned is None`（问不到 Docker 状态）不算不一致 —— 那种情况下拒绝载入
    只会让没有 Docker socket 的部署方式完全用不了这个开关。
    """
    return [d for d in resolved['matched'] + resolved['installable']
            if d.get('aligned') is False]


def _overwrite_summary(includes: list) -> dict:
    """载入会覆盖掉的现状，逐项列给用户确认。"""
    skills_mod = _event_skills()
    import task_store

    layout = _layout()
    summary: dict = {}

    if BLOCK_CANVAS in includes:
        from api.canvas import all_tool_configs
        summary['canvas'] = {
            'cards':       len(layout.get('cards', []) or []),
            'connections': len(layout.get('connections', []) or [])
                           + len(layout.get('execConnections', []) or []),
            'toolConfigs': len(all_tool_configs()),
        }

    if BLOCK_SKILLS in includes:
        summary['skills'] = [
            {'slug': s['slug'], 'name': s.get('name', '')}
            for s in skills_mod.installed_skills() if s.get('active')
        ]

    prompt_files = []
    paths = _prompt_paths()
    for block in PROMPT_BLOCKS:
        key = block.split('.', 1)[1]
        if block in includes:
            path = paths[key]
            prompt_files.append({'block': block, 'path': str(path),
                                 'exists': _prompt_exists(key, path)})
    if prompt_files:
        summary['prompt'] = prompt_files

    if BLOCK_TASKS in includes:
        task_store.load_all()
        summary['tasks'] = [{'id': t.id, 'goal': t.goal}
                            for t in task_store.active_tasks()]

    return summary


# ── 端点：查看当前方案 ───────────────────────────────────────────────────────

@router.get('/current')
async def get_current():
    """当前已载入的解决方案（没有则 data 为 null）。"""
    return {'code': 200, 'data': config.main.get(_CURRENT_KEY, None)}


@router.delete('/current')
async def clear_current():
    """清除"当前方案"标记。只动标记，不回滚任何实际配置。"""
    config.main[_CURRENT_KEY] = None
    return {'code': 200}


# ── 端点：可打包内容 ─────────────────────────────────────────────────────────

@router.get('/packable')
async def packable(request: fastapi.Request):
    """当前机器上有哪些东西可以打包，以及哪些技能不能打包。"""
    skills_mod = _event_skills()
    import task_store

    devices, ref_of, unresolved = _canvas_devices()
    layout = _layout()

    active = [s for s in skills_mod.installed_skills() if s.get('active')]
    on_market, off_market = await _market_skill_slugs([s['slug'] for s in active])

    def _skill_brief(slug: str) -> dict:
        s = next((x for x in active if x['slug'] == slug), {})
        return {'slug': slug, 'name': s.get('name', ''), 'icon': s.get('icon'),
                'version': s.get('version', '')}

    paths = _prompt_paths()
    task_store.load_all()

    return {'code': 200, 'data': {
        'coreVersion': _core_version(),
        'canvas': {
            'cards':       len(layout.get('cards', []) or []),
            'devices':     devices,
            'unresolved':  unresolved,
        },
        # 适用机型：由画布上的硬件驱动推出，前端只展示不让改
        'robotTypes': _robot_models(devices),
        'skills': {
            'available': [_skill_brief(s) for s in on_market],
            'offMarket': [_skill_brief(s) for s in off_market],
        },
        'prompt': [
            {'block': f'prompt.{key}', 'path': str(path),
             'exists': _prompt_exists(key, path),
             'chars': len(_prompt_content(key, path)) if _prompt_exists(key, path) else 0}
            for key, path in paths.items()
        ],
        'tasks': [{'id': t.id, 'goal': t.goal, 'check_cron': t.check_cron}
                  for t in task_store.active_tasks()],
        'configFields': _config_field_paths(ref_of),
    }}


# ── 端点：打包 / 发布 ────────────────────────────────────────────────────────

@router.post('/pack')
async def pack(request: fastapi.Request, req: PackRequest):
    """只打包，不落盘、不上传 —— 前端用它做发布前预览。"""
    result = await _build_payload(req, _get_rc_token(request))
    if not result.get('ok'):
        return {'code': 422, 'error': result['error'], 'detail': result.get('detail')}
    return {'code': 200, 'data': {k: v for k, v in result.items() if k != 'ok'}}


class PublishMeta(BaseModel):
    name:        str
    slug:        str
    oneLiner:    str
    description: str
    industry:    str = 'general'
    icon:        Optional[str] = None
    coverImage:  Optional[str] = None
    tags:        list[str] = []
    robotTypes:  list[str] = []
    version:     str = '1.0.0'


class PublishRequest(BaseModel):
    meta:         PublishMeta
    include:      PackInclude = PackInclude()
    extra_redact: list[str] = []


@router.post('/publish')
async def publish(request: fastapi.Request, req: PublishRequest):
    """打包并作为草稿推到 Resource Center（之后由作者提交审核）。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': _UNAUTHORIZED}

    built = await _build_payload(
        PackRequest(include=req.include, extra_redact=req.extra_redact), token)
    if not built.get('ok'):
        return {'code': 422, 'error': built['error'], 'detail': built.get('detail')}

    body = {
        **req.meta.dict(),
        'includes':        built['includes'],
        'requiredDrivers': built['requiredDrivers'],
        'payload':         built['payload'],
        'needsConfig':     built['needsConfig'],
        'coreVersion':     built['coreVersion'],
    }
    result = await _rc_request('POST', '/api/solutions/mine', token, body, timeout=60)
    if not result['ok']:
        return {'code': result['status'],
                'error': _rc_error(result['status'], result['error'])}
    return {'code': 200, 'data': result['data']}


# ── 端点：载入前检查 ────────────────────────────────────────────────────────

class LoadRequest(BaseModel):
    slug:    str = ''
    payload: Optional[dict] = None      # 直接给包体（调试用），优先于 slug
    includes: list[str] = []            # 与 payload 搭配使用
    confirm: bool = False
    session_id: str = ''                # 画布编辑锁的持有者（前端传自己的）
    align_versions: bool = False        # 要求所有相关容器与包体记录的 tag 一致
    ref: str = ''                       # /align-device 用：要对齐哪个设备


async def _load_solution(req: LoadRequest, token: Optional[str]) -> dict:
    """取回要载入的方案：优先用请求里带的 payload，否则按 slug 拉 Resource Center。"""
    if req.payload:
        return {'ok': True, 'solution': {
            'slug':     req.slug,
            'payload':  req.payload,
            'includes': req.includes,
        }}
    if not req.slug:
        return {'ok': False, 'status': 422, 'error': 'slug 或 payload 至少要给一个'}

    result = await _rc_request('GET', f'/api/solutions/{req.slug}', token, timeout=60)
    if not result['ok']:
        return {'ok': False, 'status': result['status'], 'error': result['error']}
    return {'ok': True, 'solution': result['data']}


@router.post('/preflight')
async def preflight(request: fastapi.Request, req: LoadRequest):
    """检查能不能载入：缺哪些驱动、会覆盖掉什么。"""
    loaded = await _load_solution(req, _get_rc_token(request))
    if not loaded.get('ok'):
        return {'code': loaded['status'], 'error': loaded['error']}

    solution = loaded['solution']
    payload = solution.get('payload') or {}
    if payload.get('formatVersion') != FORMAT_VERSION:
        return {'code': 422,
                'error': f'不支持的包体版本 {payload.get("formatVersion")}，'
                         f'本机支持 {FORMAT_VERSION}'}

    includes = solution.get('includes') or []
    devices = _resolve_devices(payload.get('devices') or [])
    await _augment_versions(devices)

    return {'code': 200, 'data': {
        'solution': {
            'slug':        solution.get('slug', ''),
            'name':        solution.get('name', ''),
            'version':     solution.get('version', ''),
            'industry':    solution.get('industry', ''),
            'includes':    includes,
            'coreVersion': solution.get('coreVersion') or payload.get('coreVersion', ''),
            'needsConfig': solution.get('needsConfig')
                           or payload.get('canvas', {}).get('redactedFields', []),
        },
        'devices':   devices,
        'canLoad':   not devices['missing'] and not devices['installable'],
        # 版本不一致不阻止载入 —— 只有用户勾了"对齐版本"才需要先重新部署
        'misaligned':  _misaligned(devices),
        'selfVersion': _core_version(),
        'overwrite': _overwrite_summary(includes),
        'canvasEditor': _canvas_editor_conflict(req.session_id),
    }}


def _canvas_editor_conflict(session_id: str) -> Optional[str]:
    """别人正在编辑画布时返回其 session id。

    载入会直接改写 canvas_layout，绕过 api/canvas.py 的编辑锁；如果另一个
    浏览器正持有锁，它的下一次自动保存会把刚载入的画布覆盖回去。
    """
    from api.canvas import current_editor
    editor = current_editor()
    if editor and editor != session_id:
        return editor
    return None


@router.post('/align-device')
async def align_device(request: fastapi.Request, req: LoadRequest):
    """把某个设备的容器重新部署到方案记录的 tag（"对齐版本"逐个设备执行）。

    走 api/drivers.py 的 deploy 而不是另写一套：那边已经处理了 service.yml
    合并、日志轮转策略与旧容器清理。这里只负责决定"部署哪个镜像 ref"。
    前端逐个调用并轮询 preflight，因为拉镜像慢，需要给用户进度。
    """
    loaded = await _load_solution(req, _get_rc_token(request))
    if not loaded.get('ok'):
        return {'code': loaded['status'], 'error': loaded['error']}

    payload = loaded['solution'].get('payload') or {}
    resolved = _resolve_devices(payload.get('devices') or [])
    await _augment_versions(resolved)

    dev = next((d for d in resolved['matched'] + resolved['installable']
                if d.get('ref') == req.ref), None)
    if not dev:
        return {'code': 404, 'error': f'包体里没有 ref={req.ref} 的设备，或本机没有对应驱动'}

    align_image = dev.get('alignImage') or ''
    driver_id = dev.get('localDriverId') or ''
    if not driver_id:
        return {'code': 422, 'error': f'{dev.get("name") or dev.get("serverName")} 在本机驱动清单里没有对应条目'}
    if not align_image:
        return {'code': 422,
                'error': f'{dev.get("name") or dev.get("serverName")} 没有可对齐的版本：'
                         f'包体未记录镜像 tag'}

    from api.drivers import driver_deploy
    result = await driver_deploy(driver_id, {'image': align_image})
    if result.get('code') != 200:
        return {'code': 500, 'error': result.get('message', '部署失败'),
                'driverId': driver_id, 'image': align_image}
    return {'code': 200, 'data': {'driverId': driver_id, 'image': align_image,
                                  'deploy': result.get('data')}}


# ── 端点：载入 ──────────────────────────────────────────────────────────────

@router.post('/apply')
async def apply(request: fastapi.Request, req: LoadRequest):
    """载入解决方案：覆盖画布 / 技能 / prompt / 任务中包体声明的部分。"""
    if not req.confirm:
        return {'code': 428, 'error': '需要 confirm=true —— 载入会覆盖当前配置'}

    conflict = _canvas_editor_conflict(req.session_id)
    if conflict:
        return {'code': 423,
                'error': '有其他会话正在编辑画布，请先让对方退出编辑再载入',
                'editor': conflict}

    token = _get_rc_token(request)
    loaded = await _load_solution(req, token)
    if not loaded.get('ok'):
        return {'code': loaded['status'], 'error': loaded['error']}

    solution = loaded['solution']
    payload = solution.get('payload') or {}
    if payload.get('formatVersion') != FORMAT_VERSION:
        return {'code': 422,
                'error': f'不支持的包体版本 {payload.get("formatVersion")}'}

    includes = solution.get('includes') or []
    resolved = _resolve_devices(payload.get('devices') or [])
    if resolved['missing'] or resolved['installable']:
        return {'code': 409,
                'error': '所需驱动尚未就绪，请先安装并启动这些驱动',
                'devices': resolved}

    # 勾了"对齐版本"就在这里再核一遍：前端是逐个设备部署的，中间任何一个失败
    # 都不该让方案以混合版本落地。没勾则完全不看版本（打包时只记录、不判断）。
    if req.align_versions:
        await _augment_versions(resolved)
        misaligned = _misaligned(resolved)
        if misaligned:
            return {'code': 409,
                    'error': '以下容器的版本还没对齐到方案记录的 tag，请先完成对齐',
                    'misaligned': misaligned}

    # Apply prompts before every other solution side effect.  _apply_prompt
    # stores long-term memory before the file-backed prompts, so validation or
    # storage failures return without changing canvas, skills, or tasks.
    try:
        prompt_applied, prompt_warning = await _apply_prompt(
            payload.get('prompt') or {}, includes
        )
    except agent_memory.AgentMemoryValidationError:
        return {'code': 422, 'error': '方案中的长期记忆内容不能为空'}
    except agent_memory.AgentMemoryCommitUncertainError:
        return {'code': 503, 'error': '长期记忆写入结果无法确认，请检查存储状态'}
    except agent_memory.AgentMemoryError:
        return {'code': 503, 'error': '长期记忆存储暂不可用'}

    applied: dict = {}

    # 1) 画布 —— deviceRef 换回本机 mcpId
    if BLOCK_CANVAS in includes:
        applied['canvas'] = await _apply_canvas(payload.get('canvas') or {},
                                                resolved['mapping'])

    # 2) 技能
    if BLOCK_SKILLS in includes:
        applied['skills'] = await _apply_skills(payload.get('skills') or [], token)

    # 3) Prompt (already applied above; keep the response ordering stable)
    if prompt_applied:
        applied['prompt'] = prompt_applied

    # 4) 任务
    if BLOCK_TASKS in includes:
        applied['tasks'] = _apply_tasks(payload.get('tasks') or [])

    needs_config = (solution.get('needsConfig')
                    or (payload.get('canvas') or {}).get('redactedFields') or [])
    config.main[_CURRENT_KEY] = {
        'slug':        solution.get('slug', ''),
        'name':        solution.get('name', ''),
        'version':     solution.get('version', ''),
        'industry':    solution.get('industry', ''),
        'includes':    includes,
        'coreVersion': solution.get('coreVersion') or payload.get('coreVersion', ''),
        'needsConfig': needs_config,
        'appliedAt':   int(time.time()),
        'versionAligned': bool(req.align_versions),
        'devices':     payload.get('devices') or [],
    }

    # 记一次载入量（失败无所谓，别影响载入结果）
    if solution.get('slug') and not req.payload:
        await _rc_request('POST', f'/api/solutions/{solution["slug"]}', None, {})

    data = {'applied': applied, 'needsConfig': needs_config}
    if prompt_warning:
        data['warning'] = prompt_warning
    return {'code': 200, 'data': data}


async def _apply_canvas(canvas: dict, mapping: dict) -> dict:
    """写画布布局与卡片配置。"""
    from api.canvas import (apply_tool_config, delete_all_tool_configs,
                            notify_layout_changed, tool_config_key)

    cards = []
    for card in canvas.get('cards') or []:
        mcp_id = mapping.get(card.get('deviceRef'))
        if not mcp_id:
            continue
        cards.append({
            'id':       card.get('id', ''),
            'mcpId':    mcp_id,
            'toolName': card.get('toolName', ''),
            'x':        card.get('x', 0),
            'y':        card.get('y', 0),
            'topicIn':  card.get('topicIn') or [],
            'topicOut': card.get('topicOut') or [],
        })
    card_ids = {c['id'] for c in cards}

    connections = [c for c in canvas.get('connections') or []
                   if c.get('fromCardId') in card_ids and c.get('toCardId') in card_ids]
    exec_connections = []
    for c in canvas.get('execConnections') or []:
        if c.get('fromCardId') not in card_ids or c.get('toCardId') not in card_ids:
            continue
        c = dict(c)
        target = next((x for x in cards if x['id'] == c.get('toCardId')), None)
        if target:
            c['toMcpId'] = target['mcpId']
        exec_connections.append(c)

    old_cards = (config.main.get('canvas_layout', {}) or {}).get('cards', [])
    config.main['canvas_layout'] = {
        'cards':           cards,
        'connections':     connections,
        'execConnections': exec_connections,
        'transform':       canvas.get('transform') or {},
    }
    # 方案里的卡片是整套替换的，被换掉的那些卡片的实例不会再有人来停它
    from api.config import stop_removed_cards
    await stop_removed_cards(old_cards, cards)
    # 绕过编辑锁直接改写了布局，所有开着画布的客户端都得重新拉一次
    notify_layout_changed()

    # 卡片配置：先清空旧的，再写包体里的
    removed = delete_all_tool_configs()
    written = 0
    for key, value in (canvas.get('toolConfigs') or {}).items():
        parts = key.split(':')
        if len(parts) < 2:
            continue
        ref, tool_name = parts[0], parts[1]
        instance_id = parts[2] if len(parts) > 2 else ''
        mcp_id = mapping.get(ref)
        if not mcp_id:
            continue
        if instance_id and instance_id not in card_ids:
            continue
        config.main[tool_config_key(mcp_id, tool_name, instance_id)] = value
        apply_tool_config(mcp_id, tool_name, value, instance_id)
        written += 1

    return {'cards': len(cards),
            'connections': len(connections) + len(exec_connections),
            'toolConfigsWritten': written, 'toolConfigsRemoved': removed}


async def _apply_skills(skills: list, token: Optional[str]) -> dict:
    """安装（若缺）并激活包体里的技能。原有技能停用但不卸载。"""
    skills_mod = _event_skills()
    from api.skills import install_from_rc

    wanted = [s.get('slug') for s in skills if s.get('slug')]
    installed_now, failed = [], []

    for slug in wanted:
        result = await install_from_rc(slug, token)
        if result.get('ok'):
            if not result.get('already'):
                installed_now.append(slug)
        else:
            failed.append({'slug': slug, 'error': result.get('error', '安装失败')})

    # 先全停，再按包体激活 —— 否则上一套方案的技能会一起注入 prompt
    for s in skills_mod.installed_skills():
        if s.get('active') and s['slug'] not in wanted:
            skills_mod.active_skills.discard(s['slug'])
    for slug in wanted:
        if not any(f['slug'] == slug for f in failed):
            skills_mod.active_skills.add(slug)

    return {'activated': [s for s in wanted if not any(f['slug'] == s for f in failed)],
            'installed': installed_now, 'failed': failed}


async def _apply_prompt(prompt: dict, includes: list) -> tuple[list, str | None]:
    """Apply selected prompts, routing long-term memory through its owner."""
    paths = _prompt_paths()
    written = []
    warning = None

    # Store memory first so a durable-write failure cannot leave the remaining
    # prompt files looking like the whole solution was applied successfully.
    memory_block = 'prompt.memory'
    memory_content = prompt.get('memory')
    if memory_block in includes and memory_content is not None:
        snapshot = await agent_memory.replace(
            memory_content,
            actor_key='api:solutions',
            reason='solution_apply',
        )
        written.append(memory_block)
        if not snapshot.fallback_ready:
            warning = agent_memory.COMPATIBILITY_WARNING

    for block in PROMPT_BLOCKS:
        if block not in includes:
            continue
        key = block.split('.', 1)[1]
        if key == 'memory':
            continue
        content = prompt.get(key)
        if content is None:
            continue
        path = paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(block)
    return written, warning


def _apply_tasks(tasks: list) -> dict:
    """清空现有任务后按包体重建（新 id），并重新挂上 cron。"""
    import scheduler
    import task_store

    task_store.load_all()
    existing = task_store.active_tasks()
    for t in existing:
        scheduler.remove_job(f'task:{t.id}')
        task_store.done(t.id, summary='载入解决方案时清除')

    created = []
    for spec in tasks:
        goal = (spec.get('goal') or '').strip()
        if not goal:
            continue
        task = task_store.create(goal, spec.get('check_cron', '') or '',
                                 spec.get('metadata') or {})
        created.append(task.id)
        if task.check_cron:
            try:
                scheduler.add_job(
                    f'task:{task.id}', task.check_cron,
                    f'任务定时检查 [{task.id}]：{task.goal}。请查询实际状态并更新进展。')
            except Exception:
                pass  # cron 表达式坏了不该让整包载入失败

    return {'cleared': len(existing), 'created': created}


# ── 端点：RC 代理（与 api/skills.py 的 /rc/* 一一对应）──────────────────────

@router.get('/rc/mine')
async def rc_my_solutions(request: fastapi.Request):
    """代理获取用户在 Resource Center 上的解决方案列表。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': _UNAUTHORIZED}
    result = await _rc_request('GET', '/api/solutions/mine', token)
    if not result['ok']:
        return {'code': result['status'],
                'error': _rc_error(result['status'], result['error'])}
    return {'code': 200, 'data': result['data']}


@router.put('/rc/mine/{solution_id}')
async def rc_update_solution(solution_id: str, request: fastapi.Request,
                             body: dict = fastapi.Body(...)):
    """代理更新 Resource Center 上的解决方案（只改展示信息）。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': _UNAUTHORIZED}
    result = await _rc_request('PUT', f'/api/solutions/mine/{solution_id}', token, body)
    if not result['ok']:
        return {'code': result['status'],
                'error': _rc_error(result['status'], result['error'])}
    return {'code': 200, 'data': result['data']}


@router.delete('/rc/mine/{solution_id}')
async def rc_delete_solution(solution_id: str, request: fastapi.Request):
    """代理删除 Resource Center 上的解决方案。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': _UNAUTHORIZED}
    result = await _rc_request('DELETE', f'/api/solutions/mine/{solution_id}', token)
    if not result['ok']:
        return {'code': result['status'],
                'error': _rc_error(result['status'], result['error'])}
    return {'code': 200, 'data': {'id': solution_id}}


@router.post('/rc/mine/{solution_id}/submit')
async def rc_submit_solution(solution_id: str, request: fastapi.Request):
    """代理提交解决方案送审。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': _UNAUTHORIZED}
    result = await _rc_request('POST', f'/api/solutions/mine/{solution_id}/submit',
                               token, {})
    if not result['ok']:
        return {'code': result['status'],
                'error': _rc_error(result['status'], result['error'])}
    return {'code': 200, 'data': result['data']}


@router.get('/market')
async def market(search: str = '', industry: str = 'all', limit: int = 20):
    """代理方案市场列表。

    走后端代理而不是浏览器直连 RC：dashboard 走 HTTPS 自签证书，浏览器对
    跨域 + 混合内容的处理会因环境而异，而后端本来就有 RC 连接。
    """
    params = [f'limit={limit}']
    if search:
        from urllib.parse import quote
        params.append(f'search={quote(search)}')
    if industry and industry != 'all':
        params.append(f'industry={industry}')
    result = await _rc_request('GET', f'/api/solutions?{"&".join(params)}')
    if not result['ok']:
        return {'code': result['status'],
                'error': _rc_error(result['status'], result['error'])}
    return {'code': 200, 'data': result['data']}


@router.get('/market/{slug}')
async def market_detail(slug: str, request: fastapi.Request):
    """代理单个方案详情（含包体）。"""
    result = await _rc_request('GET', f'/api/solutions/{slug}',
                               _get_rc_token(request), timeout=60)
    if not result['ok']:
        return {'code': result['status'],
                'error': _rc_error(result['status'], result['error'])}
    return {'code': 200, 'data': result['data']}
