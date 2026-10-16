"""Skill-package transfer through one selected card.

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
- the **legacy slug prefix** for skills installed before packages carried
  `attachment` is declared per card, not spelled here.

Every earlier version of this file hardcoded one application's vocabulary --
a `motus.teaching.package/1` identifier, a `trajectory.jsonl` entry, a
`taught-` slug prefix, a 32-hex asset id, a 64-char binding. Each of those
made a second kind of device an edit to Core, which is the opposite of what
`x-skill-package` is for.
"""
import asyncio
import base64
import copy
import hashlib
import io
import json
import re
import secrets
import threading
import time
import zipfile

import fastapi
from fastapi.responses import Response

import config
from api.canvas import tool_config_key

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

# `prepare_execution_config` refreshes the generated instructions of every
# skill the card owns. That is one device call each, so it needs a ceiling on
# both counts -- a card with a hundred skills must not turn a config save into
# a hundred serial round trips.
REFRESH_DEADLINE_S = 30.0
REFRESH_MAX_SKILLS = 50

# Opaque device-side identifiers. Core compares and forwards them; it does not
# know whether they are hashes, UUIDs or row ids, so it bounds the length and
# the alphabet and stops there.
_OPAQUE = re.compile(r'[A-Za-z0-9_.:-]{1,128}')
_POLICY_KEY = 'skill_execution_policies'

_approvals = {}
_approval_lock = threading.Lock()


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
    # This is the sole authority. The device reads it before every preview/run;
    # config MCP arguments and imported Skill metadata cannot grant permission.
    shared = config.main.get(tool_config_key(target['mcp_id'], target['tool']), {}) or {}
    own = config.main.get(tool_config_key(target['mcp_id'], target['tool'],
                                          target['instance_id']), {}) or {}
    settings = {**shared, **own}
    requested = settings.get('auto_execute_verified') is True
    device_enabled = requested and settings.get('_execution_binding') == device_binding
    policies = config.main.get(_POLICY_KEY, {})
    policy = policies.get(execution_key(target) + ':' + asset_id, {}) if asset_id else {}
    skill_enabled = (policy.get('enabled') is True
                     and policy.get('device_binding') == device_binding
                     and policy.get('asset_binding') == asset_binding)
    return {'enabled': device_enabled or skill_enabled, 'device_enabled': device_enabled,
            'skill_enabled': skill_enabled,
            'source': 'card' if device_enabled else 'skill' if skill_enabled else 'manual',
            'binding_changed': ((requested and not device_enabled)
                                or (policy.get('enabled') is True and not skill_enabled))}


async def binding_of(target, asset_id=''):
    args = {'asset_id': asset_id} if asset_id else {}
    return read_binding(await call(target, 'execution_binding', **args), asset_id=asset_id)


async def prepare_execution_config(mcp_id, tool_name, body, instance_id, request):
    """Gate and bind `auto_execute_verified` on the ordinary config-save path.

    Returns `body` untouched for every tool that does not declare the
    capability, which is all of them but one kind of card.
    """
    from tool_config import find_tool
    tool = find_tool(mcp_id, tool_name)
    if not tool.get('inputSchema', {}).get('x-skill-package', {}).get('persistent_execution'):
        return body
    require_browser(request)
    if not isinstance(body, dict) or type(body.get('auto_execute_verified', False)) is not bool:
        raise fastapi.HTTPException(422, '自动执行设置必须为布尔值')
    known = targets()
    candidates = [t for t in known if t['mcp_id'] == mcp_id and t['tool'] == tool_name
                  and (not instance_id or t['instance_id'] == instance_id)]
    if len(candidates) != 1:
        raise fastapi.HTTPException(409, '请在对应卡片的配置中设置自动执行')
    target = candidates[0]
    # `_execution_binding` is derived here and nowhere else. A caller that sends
    # one is either an old page replaying what it read back or someone trying to
    # pin a stale device; either way its value is discarded.
    clean = {k: v for k, v in body.items() if k != '_execution_binding'}
    if not clean.get('auto_execute_verified'):
        return clean
    previous = config.main.get(tool_config_key(mcp_id, tool_name, instance_id), {}) or {}
    clean['_execution_binding'] = (await binding_of(target))['device_binding']
    if (previous.get('auto_execute_verified') is True
            and previous.get('_execution_binding') == clean['_execution_binding']):
        return clean      # already on and still bound: nothing to migrate
    # Old generated instructions must learn the automatic-execution rule before
    # a whole card is enabled, so the user is not sent looking for a second
    # migration button. Bounded in both count and time: this runs inside a
    # config save, and one unreachable skill must not hold that save open.
    await _refresh_owned_definitions(target, known)
    return clean


async def _refresh_owned_definitions(target, known):
    legacy_prefix = target['capabilities'].get('legacy_slug_prefix') or ''
    pending, skipped = [], 0
    for skill in list(config.main.get('skills', {}).get('installed', [])):
        attachment = skill.get('attachment') or {}
        legacy = (not attachment and legacy_prefix
                  and skill.get('slug', '').startswith(legacy_prefix)
                  and skill.get('requiredTools') == [target['tool']])
        if attachment.get('tool') != target['tool'] and not legacy:
            continue
        try:
            if resolve_skill(skill['slug'], known=known)[1] != target:
                continue
        except fastapi.HTTPException:
            continue
        if len(pending) >= REFRESH_MAX_SKILLS:
            skipped += 1
            continue
        pending.append(skill['slug'])
    deadline = time.monotonic() + REFRESH_DEADLINE_S
    failed = []
    for slug in pending:
        if time.monotonic() >= deadline:
            skipped += len(pending) - pending.index(slug)
            break
        try:
            await refresh_definition(slug)
        except Exception:     # noqa: BLE001 - one bad skill must not block the save
            failed.append(slug)
    if failed or skipped:
        # Loud rather than silent: a skill whose instructions were not refreshed
        # still runs, it just does not mention the new rule yet.
        print(f'[skill-package] auto-execution enabled on {target["label"]}; '
              f'{len(pending) - len(failed)} definition(s) refreshed, '
              f'{len(failed)} failed, {skipped} not attempted')


@router.post('/execution/read')
async def read_execution(request: fastapi.Request, body: dict = fastapi.Body(...)):
    import auth
    if not auth.from_loopback(request):
        raise fastapi.HTTPException(403, '仅同机驱动可读取执行配置')
    known = targets()
    candidates = [t for t in known if t['tool'] == body.get('tool')
                  and t['capabilities'].get('persistent_execution')
                  and (not body.get('instance_id') or t['instance_id'] == body['instance_id'])]
    manual = {'code': 200, 'data': {'enabled': False, 'device_enabled': False,
                                    'skill_enabled': False, 'source': 'manual'}}
    asset_id = body.get('asset_id') or ''
    if asset_id:
        owner = _owner_of_asset(asset_id, candidates, known)
        if owner is None:
            return manual
        candidates = [owner]
    if len(candidates) != 1:
        return manual
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
    for target in candidates:
        prefix = target['capabilities'].get('legacy_slug_prefix')
        if not prefix:
            continue
        try:
            owner = resolve_skill(prefix + asset_id, known=known)[1]
        except fastapi.HTTPException:
            continue
        if owner in candidates:
            return owner
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
        await refresh_definition(slug)
    # Revoke **before** persisting, in both directions. The other order wrote
    # the policy first and then called a device that can refuse: turning the
    # setting off then reported a 409 while leaving the old temporary grant
    # alive -- the exact thing this call exists to prevent. Writing last means a
    # refusal leaves the stored setting untouched and the client correctly told
    # nothing changed.
    await call(target, 'session_revoke', asset_id=asset_id)
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


# ── listing, drafts, approvals ───────────────────────────────────────────────

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


@router.post('/approval/consume')
async def consume_approval(request: fastapi.Request, body: dict = fastapi.Body(...)):
    import auth
    if not auth.from_loopback(request):
        raise fastapi.HTTPException(403, '仅本机驱动可核验授权')
    ident = body.get('approval_id')
    if not isinstance(ident, str) or len(ident) != 64:
        raise fastapi.HTTPException(422, '无效的授权凭据')
    with _approval_lock:
        approval = _approvals.pop(ident, None)
    if not approval or time.monotonic() >= approval['expires']:
        raise fastapi.HTTPException(409, '授权凭据无效或已过期')
    return {'code': 200, 'data': approval['receipt']}


@router.get('/session/{slug}')
async def session_status(slug: str):
    _, target, asset_id = resolve_skill(slug)
    if not target['capabilities'].get('session_authorization'):
        return {'code': 200, 'data': {'supported': False, 'active': False}}
    return {'code': 200, 'data': {'supported': True,
                                  **await call(target, 'session_status', asset_id=asset_id)}}


@router.post('/session/{slug}')
async def authorize_session(slug: str, request: fastapi.Request, body: dict = fastapi.Body(...)):
    # This route is never a model tool. Require browser token authentication
    # even when a development server has its ordinary middleware disabled.
    require_browser(request)
    _, target, asset_id = resolve_skill(slug)
    if body.get('enabled') is False:
        return {'code': 200, 'data': await call(target, 'session_revoke', asset_id=asset_id)}
    if (body.get('enabled') is not True or body.get('workspace_clear') is not True
            or body.get('take_control') is not True):
        raise fastapi.HTTPException(422, '请确认本次路径、活动空间与控制权接管')
    challenge = await call(target, 'session_challenge', asset_id=asset_id)
    ident = secrets.token_hex(32)
    receipt = {'nonce': challenge['nonce'], 'asset_id': asset_id, 'tool': target['tool'],
               'instance_id': target['instance_id'], 'seconds': 1800}
    with _approval_lock:
        for key, value in list(_approvals.items()):
            if time.monotonic() >= value['expires']:
                _approvals.pop(key)
        if len(_approvals) >= 64:
            raise fastapi.HTTPException(429, '请稍后重试')
        _approvals[ident] = {'receipt': receipt, 'expires': time.monotonic() + 30}
    try:
        result = await call(target, 'session_authorize', asset_id=asset_id, approval_id=ident)
        # Existing installed definitions must learn the session-aware calling
        # contract too; do not make the user find another migration button.
        await refresh_definition(slug)
        return {'code': 200, 'data': result}
    except Exception:
        await call(target, 'session_revoke', asset_id=asset_id)
        raise
    finally:
        with _approval_lock:
            _approvals.pop(ident, None)


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
    skill = next((s for s in config.main.get('skills', {}).get('installed', [])
                  if s['slug'] == slug), None)
    if not skill:
        raise fastapi.HTTPException(404, '技能未安装')
    known = targets() if known is None else known
    attachment = skill.get('attachment') or {}
    tool = attachment.get('tool')
    asset_id = attachment.get('asset_id')
    if not attachment:
        # Skills installed before packages carried `attachment`. The prefix is
        # whichever one the card declares -- Core spells no application's
        # convention -- and only an exact prefix match on a uniquely connected
        # card resolves.
        deps = skill.get('requiredTools', [])
        tool = deps[0] if len(deps) == 1 else None
        prefixes = {t['capabilities'].get('legacy_slug_prefix') for t in known
                    if t['tool'] == tool}
        for prefix in sorted((p for p in prefixes if p), key=len, reverse=True):
            if slug.startswith(prefix) and len(slug) > len(prefix):
                asset_id = slug[len(prefix):]
                break
    candidates = [t for t in known if t['tool'] == tool]
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
