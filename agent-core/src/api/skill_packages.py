"""Skill-package transfer through one selected card, plus the local auto-execution policy.

Core is a transport and a policy store, nothing else. It does not know what is
inside a package, what the device does with it, or what kind of device it is:

- the **protocol identifier** is whatever the card declares in
  `inputSchema['x-skill-package']['schema']`, and a package is accepted only if
  its manifest names the same one. Core holds no identifier of its own.
- the **archive** needs a `manifest.json` and may carry any other entries; Core
  checks transport boundaries (sizes, compression, entry names) and nothing
  about their meaning.
- **`compatibility`** is opaque. Core reads exactly one field out of it --
  `tool`, to route the package to the right card -- and forwards the rest for
  the device to judge and the page to display.

Every earlier version of this file hardcoded one application's vocabulary --
a `motus.teaching.package/1` identifier, a `trajectory.jsonl` entry, a
`taught-` slug prefix, a 32-hex asset id, a 64-char binding. Each of those
made a second kind of device an edit to Core, which is the opposite of what
`x-skill-package` is for.

Three capabilities were built here and then removed before merge, because each
was scaffolding for something that does not exist:

- **A 30-minute session grant** (`session_challenge`/`authorize`/`revoke`, a
  one-use receipt store, and a loopback auth exemption to consume it) was
  described as backward compatibility for adapters predating the persistent
  setting. There are none: the only driver that ever implemented those actions
  is unmerged, so it was compatibility with a predecessor that never shipped.
- **A legacy slug-prefix resolver** mapped pre-`attachment` skills to their
  device. Exactly one test rig had such a row; a single DB update replaces two
  code paths, and a slug convention is not an identity mechanism anyway.
- **A whole-card override** granted auto-execution to every skill the card
  would *later* verify -- the widest grant in the feature -- and to do it, it
  intercepted the generic `PUT /api/canvas/tool-config` endpoint, did device
  I/O inside a config save, and stored a private field inside the user's own
  config blob. Per-skill consent covers the real need.

See docs/SKILL_WORKBENCH.md for what replaced them.
"""
import asyncio
import base64
import copy
import hashlib
import io
import json
import re
import secrets
import time
import zipfile

import fastapi
from fastapi.responses import Response

import config

router = fastapi.APIRouter(prefix='/package')

MAX_BYTES = 16 * 1024 * 1024          # whole archive, as uploaded
MAX_MANIFEST_BYTES = 16 * 1024        # manifest.json, uncompressed
MAX_ENTRY_BYTES = 64 * 1024 * 1024    # any single payload entry, uncompressed
MAX_UNPACKED_BYTES = 128 * 1024 * 1024  # all entries together, uncompressed
CHUNK_BYTES = 24576

# A whole-archive deadline. Each chunk already has its own 30s timeout, but
# 16 MiB is up to 683 chunks: without a deadline over the loop an upload can
# hold a request open for hours, and every other device-facing path in this
# codebase carries one.
TRANSFER_DEADLINE_S = 180.0

# Opaque device-side identifiers. Core compares and forwards them; it does not
# know whether they are hashes, UUIDs or row ids, so it bounds the length and
# the alphabet and stops there.
_OPAQUE = re.compile(r'[A-Za-z0-9_.:-]{1,128}')
_POLICY_KEY = 'skill_execution_policies'


# ── authentication ───────────────────────────────────────────────────────────

def require_browser(request):
    """Only a logged-in page may change what runs without asking.

    Fails closed when the deployment has no `ACCESS_TOKEN` at all, but says so.
    Telling an operator on an auth-disabled rig to "log in" sends them after a
    login that does not exist on that machine.
    """
    import auth
    token = auth.get_token()
    if not token:
        raise fastapi.HTTPException(
            403, '本机未配置 ACCESS_TOKEN，无法确认操作来自已登录网页；'
                 '请在 .env 中设置 ACCESS_TOKEN 后重启 Agent Core。')
    supplied = request.headers.get('Authorization', '').removeprefix('Bearer ')
    if not secrets.compare_digest(supplied, token):
        raise fastapi.HTTPException(401, '请登录网页后再修改自动执行设置')


# ── policy store ─────────────────────────────────────────────────────────────

def execution_key(target):
    return f"{target['mcp_id']}:{target['tool']}:{target['instance_id']}"


def read_binding(data, *, asset_id=''):
    """The three fields Core understands out of a device's binding reply.

    Explicit, because `execution_setting(**binding)` used to splat the device's
    reply straight into a three-parameter function: the moment a driver added
    the model or calibration summary its own protocol calls for, that was a
    `TypeError` and a 500. Extra fields are the device's business and are
    ignored here; a missing `device_binding` is a protocol error and says so.
    """
    binding = data if isinstance(data, dict) else {}
    device = binding.get('device_binding')
    if not isinstance(device, str) or not _OPAQUE.fullmatch(device):
        raise fastapi.HTTPException(502, '设备未返回有效的绑定标识')
    asset = binding.get('asset_binding', '')
    if not isinstance(asset, str) or len(asset) > 128:
        raise fastapi.HTTPException(502, '设备返回的资产绑定标识无效')
    return {'device_binding': device, 'asset_id': asset_id, 'asset_binding': asset}


def execution_setting(target, device_binding, asset_id='', asset_binding=''):
    """Whether this one skill may run on this one card without asking.

    Per skill, per card, per recording -- there is deliberately no card-wide
    form. The device reads this before every preview/run; MCP config arguments
    and imported Skill metadata cannot grant it.
    """
    policies = config.main.get(_POLICY_KEY, {})
    policy = policies.get(execution_key(target) + ':' + asset_id, {}) if asset_id else {}
    requested = policy.get('enabled') is True
    enabled = (requested and policy.get('device_binding') == device_binding
               and policy.get('asset_binding') == asset_binding)
    return {'enabled': enabled, 'source': 'skill' if enabled else 'manual',
            'binding_changed': requested and not enabled}


MANUAL = {'enabled': False, 'source': 'manual', 'binding_changed': False}


async def binding_of(target, asset_id=''):
    args = {'asset_id': asset_id} if asset_id else {}
    return read_binding(await call(target, 'execution_binding', **args), asset_id=asset_id)


@router.post('/execution/read')
async def read_execution(request: fastapi.Request, body: dict = fastapi.Body(...)):
    """What the device asks before every preview/run. Read-only, loopback only."""
    import auth
    if not auth.from_loopback(request):
        raise fastapi.HTTPException(403, '仅同机驱动可读取执行配置')
    known = targets()
    candidates = [t for t in known if t['tool'] == body.get('tool')
                  and t['capabilities'].get('persistent_execution')
                  and (not body.get('instance_id') or t['instance_id'] == body['instance_id'])]
    asset_id = body.get('asset_id') or ''
    if asset_id:
        owner = _owner_of_asset(asset_id, candidates, known)
        if owner is None:
            return {'code': 200, 'data': dict(MANUAL)}
        candidates = [owner]
    if len(candidates) != 1:
        return {'code': 200, 'data': dict(MANUAL)}
    device = body.get('device_binding')
    if not isinstance(device, str) or not _OPAQUE.fullmatch(device):
        raise fastapi.HTTPException(422, '设备绑定无效')
    return {'code': 200, 'data': execution_setting(
        candidates[0], device, asset_id, body.get('asset_binding', '') or '')}


def _owner_of_asset(asset_id, candidates, known):
    """Which of these cards owns the skill carrying `asset_id`, if exactly one."""
    for skill in config.main.get('skills', {}).get('installed', []):
        if (skill.get('attachment') or {}).get('asset_id') != asset_id:
            continue
        try:
            owner = resolve_skill(skill['slug'], known=known)[1]
        except fastapi.HTTPException:
            return None
        return owner if owner in candidates else None
    return None


@router.get('/execution/{slug}')
async def get_execution(slug: str):
    _, target, asset_id = resolve_skill(slug)
    if not target['capabilities'].get('persistent_execution'):
        return {'code': 200, 'data': {'supported': False}}
    binding = await binding_of(target, asset_id)
    return {'code': 200, 'data': {'supported': True, **execution_setting(target, **binding)}}


@router.post('/execution/{slug}')
async def set_execution(slug: str, request: fastapi.Request, body: dict = fastapi.Body(...)):
    require_browser(request)
    if type(body.get('enabled')) is not bool:
        raise fastapi.HTTPException(422, '请选择是否允许自动执行')
    _, target, asset_id = resolve_skill(slug)
    if not target['capabilities'].get('persistent_execution'):
        raise fastapi.HTTPException(409, '该设备驱动不支持持久的自动执行设置，请先升级驱动')
    binding = await binding_of(target, asset_id)
    if body['enabled']:
        # The generated instructions must learn the automatic-execution rule
        # before the skill is allowed to run unattended.
        await refresh_definition(slug)
    policies = dict(config.main.get(_POLICY_KEY, {}))
    policies[execution_key(target) + ':' + asset_id] = {**binding, 'enabled': body['enabled']}
    config.main[_POLICY_KEY] = policies
    return {'code': 200, 'data': {'supported': True, **execution_setting(target, **binding)}}


# ── card discovery ───────────────────────────────────────────────────────────

def targets():
    """Cards whose tool declares a skill-package protocol, whatever it is."""
    services = config.main.get('services', {}).get('mcp', [])
    cards = config.main.get('canvas_layout', {}).get('cards', [])
    result = []
    for card in cards:
        service = next((m for m in services if m['id'] == card.get('mcpId')), None)
        if not service or service.get('transport', 'http') != 'http':
            continue
        tool = next((t for t in service.get('tools', []) if isinstance(t, dict)
                     and t.get('name') == card.get('toolName')), None)
        capabilities = (tool or {}).get('inputSchema', {}).get('x-skill-package')
        schema = capabilities.get('schema') if isinstance(capabilities, dict) else None
        if not isinstance(schema, str) or not schema:
            continue
        result.append({'mcp_id': service['id'], 'tool': tool['name'], 'instance_id': card['id'],
                       'label': f"{service.get('name', service['id'])} / {tool['name']}",
                       'schema': schema, 'capabilities': capabilities,
                       'workbench': tool.get('inputSchema', {}).get('x-workbench')})
    return result


def target_for(card_id, known=None):
    target = next((t for t in (known if known is not None else targets())
                   if t['instance_id'] == card_id), None)
    if not target:
        raise fastapi.HTTPException(409, '请选择画布中支持技能包的设备卡片')
    return target


async def call(target, action, **args):
    from api.mcp_manage import mcp_call_tool, MCPCallRequest
    response = await mcp_call_tool(target['mcp_id'], MCPCallRequest(
        tool=target['tool'],
        arguments={**args, 'action': action, 'instance_id': target['instance_id']}), timeout_s=30)
    if response.get('code') != 200:
        raise fastapi.HTTPException(409, response.get('message') or '设备拒绝技能包请求')
    data = response['data']
    if isinstance(data, list):
        try:
            data = json.loads(next(c['text'] for c in data if c.get('type') == 'text'))
        except (StopIteration, ValueError, KeyError, TypeError):
            raise fastapi.HTTPException(502, '设备返回无效结果')
    if not isinstance(data, dict) or data.get('error'):
        raise fastapi.HTTPException(409, data.get('error') if isinstance(data, dict)
                                   else '设备返回无效结果')
    return data


# ── archive boundaries ───────────────────────────────────────────────────────

async def read_upload(file):
    data = await file.read(MAX_BYTES + 1)
    if not 1 <= len(data) <= MAX_BYTES:
        raise fastapi.HTTPException(413, f'技能包须小于 {MAX_BYTES // (1024 * 1024)} MiB')
    return data


def manifest_from(data, schema=None):
    """Transport-level validation of a package, and its manifest.

    What is *in* the package is the device's business. Core checks that the
    archive is an archive, that its entries cannot escape a directory or
    explode on extraction, and that the manifest names the protocol the
    receiving card speaks. It does not require, or know, any payload filename.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if 'manifest.json' not in names or len(names) != len(set(names)):
                raise ValueError()
            total = 0
            for item in archive.infolist():
                name = item.filename
                if (name.startswith('/') or '..' in name.split('/')
                        or item.flag_bits & 1 or item.compress_type not in (0, 8)):
                    raise ValueError()
                limit = MAX_MANIFEST_BYTES if name == 'manifest.json' else MAX_ENTRY_BYTES
                if item.file_size > limit:
                    raise ValueError()
                total += item.file_size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError()
            manifest = json.loads(archive.read(archive.getinfo('manifest.json')))
            declared = manifest.get('schema')
            if (not isinstance(declared, str) or not declared
                    or not isinstance(manifest.get('compatibility'), dict)
                    or not isinstance(manifest.get('skill'), dict)):
                raise ValueError()
            if schema is not None and declared != schema:
                raise fastapi.HTTPException(
                    409, f'技能包声明的协议 {declared} 与目标卡片的 {schema} 不一致')
            return manifest
    except fastapi.HTTPException:
        raise
    except (ValueError, KeyError, zipfile.BadZipFile, RuntimeError, UnicodeError,
            AttributeError, OSError):
        raise fastapi.HTTPException(422, '无效的技能包')


def manifest_tool(manifest):
    """The tool a package is for. The one field Core reads out of the manifest
    besides `schema`, because routing is Core's job and nothing else in there
    is."""
    for value in (manifest.get('tool'),
                  (manifest.get('compatibility') or {}).get('tool')):
        if isinstance(value, str) and value:
            return value
    return None


# ── listing and drafts ───────────────────────────────────────────────────────

@router.get('/targets')
async def list_targets():
    return {'code': 200, 'data': targets()}


@router.get('/pending')
async def pending_packages():
    async def read(target):
        if not target['capabilities'].get('pending'):
            return [], None
        try:
            result = await asyncio.wait_for(call(target, 'draft_list'), 5)
            return [{**d, 'target': target} for d in result['drafts']], None
        except Exception:     # noqa: BLE001 - an offline card is not an error here
            return [], target['label']
    result = await asyncio.gather(*(read(t) for t in targets()))
    return {'code': 200, 'data': {'drafts': [d for rows, _ in result for d in rows],
                                  'unavailable': [error for _, error in result if error]}}


@router.post('/resume')
async def resume_package(body: dict = fastapi.Body(...)):
    target = target_for(body.get('card_id'))
    await call(target, 'draft_open', draft_id=body.get('draft_id'))
    return {'code': 200, 'data': {'target': target, 'ui': await call(target, 'ui')}}


# ── skill identity ───────────────────────────────────────────────────────────

@router.post('/inspect')
async def inspect_package(file: fastapi.UploadFile = fastapi.File()):
    manifest = manifest_from(await read_upload(file))
    tool = manifest_tool(manifest)
    # Route here rather than in the page: which card can receive a package is a
    # transport question, and the browser should not need to know how a manifest
    # names its tool.
    usable = [t for t in targets() if t['schema'] == manifest['schema'] and t['workbench']
              and (tool is None or t['tool'] == tool)]
    return {'code': 200, 'data': {'manifest': manifest, 'targets': usable,
                                  'requires_local_trial': True}}


def resolve_skill(slug, card_id='', known=None):
    """The (skill, card, asset) a packaged skill belongs to.

    Identity comes from `attachment` and nothing else. A slug-prefix fallback
    for skills installed before that field existed lived here briefly; one test
    rig had such a row, and a slug naming convention is not an identity
    mechanism -- fix the row, not the resolver.
    """
    skill = next((s for s in config.main.get('skills', {}).get('installed', [])
                  if s['slug'] == slug), None)
    if not skill:
        raise fastapi.HTTPException(404, '技能未安装')
    attachment = skill.get('attachment') or {}
    tool, asset_id = attachment.get('tool'), attachment.get('asset_id')
    candidates = [t for t in (targets() if known is None else known) if t['tool'] == tool]
    origin = skill.get('origin') or {}
    if origin:
        candidates = [t for t in candidates if t['mcp_id'] == origin.get('mcp_id')
                      and t['instance_id'] == origin.get('instance_id')]
    if card_id:
        candidates = [t for t in candidates if t['instance_id'] == card_id]
    if not asset_id or len(candidates) != 1:
        raise fastapi.HTTPException(409, '无法唯一确定技能所属设备，请检查来源卡片的连接')
    return skill, candidates[0], asset_id


@router.post('/refresh/{slug}')
async def refresh_definition(slug: str):
    """Regenerate a skill's definition from its source device.

    Transitional: this exists because a skill has one mutable definition. Once
    the resource centre stores versioned packages, re-recording produces a new
    version instead of rewriting this one, and this route goes away.
    """
    skill, target, asset_id = resolve_skill(slug)
    original = copy.deepcopy(skill)
    result = await call(target, 'package_definition', asset_id=asset_id)
    from api.skills import validate_local_skill
    definition = validate_local_skill({**result['skill'], 'origin': {
        'mcp_id': target['mcp_id'], 'instance_id': target['instance_id']}})
    if definition['slug'] != slug or definition.get('attachment', {}).get('asset_id') != asset_id:
        raise fastapi.HTTPException(409, '设备返回的技能身份不匹配')
    settings = config.main.get('skills', {})
    installed = list(settings.get('installed', []))
    current = next((s for s in installed if s['slug'] == slug), None)
    if current != original:
        raise fastapi.HTTPException(409, '技能已被其他操作修改，请重新打开')
    installed[installed.index(current)] = {**current, **definition}
    config.main['skills'] = {**settings, 'installed': installed}
    return {'code': 200, 'data': {'slug': slug}}


# ── transfer ─────────────────────────────────────────────────────────────────
#
# Browser-side upload and download are a stopgap: they are how a skill moves
# between two devices while the resource centre cannot store an attachment.
# The destination is RC -> Core -> device, so nothing new should be built on
# `/export` or `/inspect`.

@router.get('/export/{slug}')
async def export_package(slug: str, card_id: str = ''):
    _, target, asset_id = resolve_skill(slug, card_id)
    content = await asyncio.wait_for(_read_all(target, asset_id), TRANSFER_DEADLINE_S)
    manifest_from(content, target['schema'])
    return Response(content, media_type='application/zip', headers={
        'Content-Disposition': f'attachment; filename="{_download_name(slug)}"',
        'Cache-Control': 'no-store'})


def _download_name(slug):
    safe = re.sub(r'[^A-Za-z0-9_.-]', '-', slug)[:80] or 'skill'
    return f'{safe}.motus-skill'


async def _read_all(target, asset_id):
    start = await call(target, 'package_export', asset_id=asset_id)
    if type(start.get('bytes')) is not int or not 1 <= start['bytes'] <= MAX_BYTES:
        raise fastapi.HTTPException(502, '设备返回的技能包大小无效')
    content = bytearray()
    while len(content) < start['bytes']:
        part = await call(target, 'package_read', transfer_id=start['transfer_id'],
                          offset=len(content))
        try:
            chunk = base64.b64decode(part['data'], validate=True)
        except (KeyError, ValueError, TypeError):
            raise fastapi.HTTPException(502, '设备返回无效技能包分块')
        if not chunk or len(chunk) > CHUNK_BYTES or len(content) + len(chunk) > start['bytes']:
            raise fastapi.HTTPException(502, '设备返回无效技能包长度')
        content.extend(chunk)
    if hashlib.sha256(content).hexdigest() != start.get('sha256'):
        raise fastapi.HTTPException(502, '技能包传输校验失败')
    return bytes(content)


@router.post('/install')
async def install_package(file: fastapi.UploadFile = fastapi.File(),
                          card_id: str = fastapi.Form()):
    known = targets()
    target = target_for(card_id, known)
    data = await read_upload(file)
    manifest = manifest_from(data, target['schema'])
    tool = manifest_tool(manifest)
    if tool is not None and tool != target['tool']:
        raise fastapi.HTTPException(409, f'技能包面向 {tool}，与目标卡片的 {target["tool"]} 不一致')
    result = await asyncio.wait_for(_write_all(target, data), TRANSFER_DEADLINE_S)
    if not result.get('reused'):
        # No installed Skill is created here. The device's normal trial/save
        # workflow issues a new local asset and uses the ordinary skill import
        # endpoint.
        return {'code': 200, 'data': {**result, 'target': target}}
    from api.skills import import_local_skill
    definition = {**result['skill'], 'origin': {'mcp_id': target['mcp_id'],
                                                'instance_id': target['instance_id']}}
    existing = next((s for s in config.main.get('skills', {}).get('installed', [])
                     if s['slug'] == definition['slug']), None)
    if existing:
        # Re-import must not reset visibility or overwrite edited metadata.
        _, original_target, original_asset = resolve_skill(existing['slug'], known=known)
        if original_target != target or original_asset != result['asset_id']:
            raise fastapi.HTTPException(409, '现有技能来源不匹配')
    else:
        await import_local_skill(definition)
    return {'code': 200, 'data': {'reused': True, 'installed': True,
                                  'slug': definition['slug'], 'target': target}}


async def _write_all(target, data):
    start = await call(target, 'package_begin', bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest())
    for offset in range(0, len(data), CHUNK_BYTES):
        await call(target, 'package_write', transfer_id=start['transfer_id'], offset=offset,
                   data=base64.b64encode(data[offset:offset + CHUNK_BYTES]).decode())
    return await call(target, 'package_finish', transfer_id=start['transfer_id'])
