"""Skill-file transfer through the selected card. Robot validation stays in Driver."""
import base64
import copy
import hashlib
import io
import json
import zipfile
import asyncio
import secrets
import threading
import time

import fastapi
from fastapi.responses import Response
import config

router = fastapi.APIRouter(prefix='/package')
SCHEMA = 'motus.teaching.package/1'
MAX_BYTES = 16 * 1024 * 1024
_approvals = {}
_approval_lock = threading.Lock()


def require_browser(request):
    import auth
    supplied = request.headers.get('Authorization', '').removeprefix('Bearer ')
    if not auth.get_token() or not secrets.compare_digest(supplied, auth.get_token()):
        raise fastapi.HTTPException(401, '请登录网页后设置自动执行')


def execution_key(target):
    return f"{target['mcp_id']}:{target['tool']}:{target['instance_id']}"


def execution_setting(target, device_binding, asset_id='', asset_binding=''):
    # This is the sole authority. Driver reads it before every preview/run;
    # config MCP arguments and imported Skill metadata cannot grant permission.
    shared = config.main.get(f"tool_config:{target['mcp_id']}:{target['tool']}", {}) or {}
    own = config.main.get(f"tool_config:{execution_key(target)}", {}) or {}
    settings = {**shared, **own}
    requested = settings.get('auto_execute_verified') is True
    device_enabled = requested and settings.get('_execution_binding') == device_binding
    policies = config.main.get('teaching_execution_policies', {})
    policy = policies.get(execution_key(target)+':'+asset_id, {}) if asset_id else {}
    skill_enabled = (policy.get('enabled') is True and policy.get('device_binding') == device_binding
                     and policy.get('asset_binding') == asset_binding)
    return {'enabled': device_enabled or skill_enabled, 'device_enabled': device_enabled,
            'skill_enabled': skill_enabled, 'source': 'card' if device_enabled else 'skill' if skill_enabled else 'manual',
            'binding_changed': (requested and not device_enabled) or (policy.get('enabled') is True and not skill_enabled)}


async def prepare_execution_config(mcp_id, tool_name, body, instance_id, request):
    from tool_config import find_tool
    tool = find_tool(mcp_id, tool_name)
    if not tool.get('inputSchema', {}).get('x-skill-package', {}).get('persistent_execution'):
        return body
    require_browser(request)
    if not isinstance(body, dict) or type(body.get('auto_execute_verified', False)) is not bool:
        raise fastapi.HTTPException(422, '自动执行设置必须为布尔值')
    candidates = [t for t in targets() if t['mcp_id'] == mcp_id and t['tool'] == tool_name
                  and (not instance_id or t['instance_id'] == instance_id)]
    if len(candidates) != 1:
        raise fastapi.HTTPException(409, '请在对应示教卡片的配置中设置自动执行')
    target = candidates[0]
    clean = {k: v for k, v in body.items() if k != '_execution_binding'}
    if clean.get('auto_execute_verified'):
        binding = await call(target, 'execution_binding')
        clean['_execution_binding'] = binding['device_binding']
        # Old generated instructions must learn the automatic-execution rule
        # before enabling a whole card. Keep user metadata and visibility.
        for skill in list(config.main.get('skills', {}).get('installed', [])):
            attachment = skill.get('attachment') or {}
            legacy = not attachment and skill.get('slug', '').startswith('taught-') and skill.get('requiredTools') == [tool_name]
            if attachment.get('tool') != tool_name and not legacy:
                continue
            try:
                _, owner, _ = resolve_skill(skill['slug'])
            except fastapi.HTTPException:
                continue
            if owner == target:
                await refresh_definition(skill['slug'])
    return clean


@router.post('/execution/read')
async def read_execution(request: fastapi.Request, body: dict = fastapi.Body(...)):
    import auth
    if not auth._from_loopback(request):
        raise fastapi.HTTPException(403, '仅同机 Driver 可读取执行配置')
    candidates = [t for t in targets() if t['tool'] == body.get('tool')
                  and t.get('capabilities', {}).get('persistent_execution')
                  and (not body.get('instance_id') or t['instance_id'] == body['instance_id'])]
    if body.get('asset_id'):
        try:
            _, owner, _ = resolve_skill('taught-'+body['asset_id'])
            candidates = [t for t in candidates if t == owner]
        except fastapi.HTTPException:
            return {'code': 200, 'data': {'enabled': False, 'device_enabled': False, 'skill_enabled': False, 'source': 'manual'}}
    if len(candidates) != 1:
        return {'code': 200, 'data': {'enabled': False, 'device_enabled': False, 'skill_enabled': False, 'source': 'manual'}}
    if not isinstance(body.get('device_binding'), str) or len(body['device_binding']) != 64:
        raise fastapi.HTTPException(422, '设备绑定无效')
    return {'code': 200, 'data': execution_setting(candidates[0], body['device_binding'],
        body.get('asset_id', ''), body.get('asset_binding', ''))}


@router.get('/execution/{slug}')
async def get_execution(slug: str):
    _, target, asset_id = resolve_skill(slug)
    if not target['capabilities'].get('persistent_execution'):
        return {'code': 200, 'data': {'supported': False}}
    binding = await call(target, 'execution_binding', asset_id=asset_id)
    return {'code': 200, 'data': {'supported': True, **execution_setting(target, **binding)}}


@router.post('/execution/{slug}')
async def set_execution(slug: str, request: fastapi.Request, body: dict = fastapi.Body(...)):
    require_browser(request)
    if type(body.get('enabled')) is not bool:
        raise fastapi.HTTPException(422, '请选择是否允许自动执行')
    _, target, asset_id = resolve_skill(slug)
    if not target['capabilities'].get('persistent_execution'):
        raise fastapi.HTTPException(409, '请更新示教 Driver')
    key = execution_key(target)+':'+asset_id
    binding = await call(target, 'execution_binding', asset_id=asset_id)
    if body['enabled']:
        await refresh_definition(slug)
    policies = dict(config.main.get('teaching_execution_policies', {}))
    policies[key] = {**binding, 'enabled': body['enabled']}
    config.main['teaching_execution_policies'] = policies
    # Disabling permanent permission must not leave an old temporary grant.
    await call(target, 'session_revoke', asset_id=asset_id)
    return {'code': 200, 'data': {'supported': True, **execution_setting(target, **binding)}}


def targets():
    services = config.main.get('services', {}).get('mcp', [])
    cards = config.main.get('canvas_layout', {}).get('cards', [])
    result = []
    for card in cards:
        service = next((m for m in services if m['id'] == card.get('mcpId')), None)
        if not service or service.get('transport', 'http') != 'http':
            continue
        tool = next((t for t in service.get('tools', []) if isinstance(t, dict)
                     and t.get('name') == card.get('toolName')), None)
        if tool and tool.get('inputSchema', {}).get('x-skill-package', {}).get('schema') == SCHEMA:
            result.append({'mcp_id': service['id'], 'tool': tool['name'], 'instance_id': card['id'],
                           'label': f"{service.get('name', service['id'])} / {tool['name']}",
                           'capabilities': tool['inputSchema']['x-skill-package'],
                           'workbench': tool.get('inputSchema', {}).get('x-workbench')})
    return result


def target_for(card_id):
    target = next((t for t in targets() if t['instance_id'] == card_id), None)
    if not target:
        raise fastapi.HTTPException(409, '请选择画布中支持动作技能包的设备卡片')
    return target


async def call(target, action, **args):
    from api.mcp_manage import mcp_call_tool, MCPCallRequest
    response = await mcp_call_tool(target['mcp_id'], MCPCallRequest(tool=target['tool'],
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
        raise fastapi.HTTPException(409, data.get('error') if isinstance(data, dict) else '设备返回无效结果')
    return data


async def read_upload(file):
    data = await file.read(MAX_BYTES+1)
    if not 1 <= len(data) <= MAX_BYTES:
        raise fastapi.HTTPException(413, '动作技能包须小于 16 MiB')
    return data


def manifest_from(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sorted(archive.namelist()) != ['manifest.json', 'trajectory.jsonl']:
                raise ValueError()
            entry = archive.getinfo('manifest.json')
            for item in archive.infolist():
                limit = 16384 if item.filename == 'manifest.json' else 64 * 1024 * 1024
                if item.file_size > limit or item.flag_bits & 1 or item.compress_type not in (0, 8):
                    raise ValueError()
            manifest = json.loads(archive.read(entry))
            if (manifest.get('schema') != SCHEMA or not isinstance(manifest.get('compatibility'), dict)
                    or not isinstance(manifest.get('skill'), dict)):
                raise ValueError()
            return manifest
    except (ValueError, KeyError, zipfile.BadZipFile, RuntimeError, UnicodeError, AttributeError):
        raise fastapi.HTTPException(422, '无效的动作技能包')


@router.get('/targets')
async def list_targets():
    return {'code': 200, 'data': targets()}


@router.get('/pending')
async def pending_packages():
    async def read(target):
        if not target.get('capabilities', {}).get('pending'):
            return [], None
        try:
            result = await asyncio.wait_for(call(target, 'draft_list'), 5)
            return [{**d, 'target': target} for d in result['drafts']], None
        except Exception:
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
    if not auth._from_loopback(request):
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
    if not target.get('capabilities', {}).get('session_authorization'):
        return {'code': 200, 'data': {'supported': False, 'active': False}}
    return {'code': 200, 'data': {'supported': True, **await call(target, 'session_status', asset_id=asset_id)}}


@router.post('/session/{slug}')
async def authorize_session(slug: str, request: fastapi.Request, body: dict = fastapi.Body(...)):
    import auth
    # This route is never a model tool. Require browser token authentication
    # even when a development server has its ordinary middleware disabled.
    supplied = request.headers.get('Authorization', '').removeprefix('Bearer ')
    if not auth.get_token() or not secrets.compare_digest(supplied, auth.get_token()):
        raise fastapi.HTTPException(401, '请登录网页后设置会话授权')
    _, target, asset_id = resolve_skill(slug)
    if body.get('enabled') is False:
        return {'code': 200, 'data': await call(target, 'session_revoke', asset_id=asset_id)}
    if body.get('enabled') is not True or body.get('workspace_clear') is not True or body.get('take_control') is not True:
        raise fastapi.HTTPException(422, '请确认本次路径、活动空间与控制权接管')
    challenge = await call(target, 'session_challenge', asset_id=asset_id)
    ident = secrets.token_hex(32)
    receipt = {'nonce': challenge['nonce'], 'asset_id': asset_id, 'tool': target['tool'],
               'instance_id': target['instance_id'], 'seconds': 1800}
    with _approval_lock:
        for key, value in list(_approvals.items()):
            if time.monotonic() >= value['expires']: _approvals.pop(key)
        if len(_approvals) >= 64:
            raise fastapi.HTTPException(429, '请稍后重试')
        _approvals[ident] = {'receipt': receipt, 'expires': time.monotonic()+30}
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
        with _approval_lock: _approvals.pop(ident, None)


@router.post('/inspect')
async def inspect_package(file: fastapi.UploadFile = fastapi.File()):
    manifest = manifest_from(await read_upload(file))
    return {'code': 200, 'data': {'manifest': manifest, 'targets': targets(), 'requires_local_trial': True}}


def resolve_skill(slug, card_id=''):
    skill = next((s for s in config.main.get('skills', {}).get('installed', []) if s['slug'] == slug), None)
    if not skill:
        raise fastapi.HTTPException(404, '技能未安装')
    attachment = skill.get('attachment') or {}
    tool = attachment.get('tool')
    asset_id = attachment.get('asset_id')
    # Existing locally taught skills predate package metadata. Resolve only
    # their exact asset slug and a uniquely connected teaching card.
    if not attachment and slug.startswith('taught-'):
        asset_id = slug.removeprefix('taught-')
        deps = skill.get('requiredTools', [])
        tool = deps[0] if len(deps) == 1 else None
    candidates = [t for t in targets() if t['tool'] == tool]
    origin = skill.get('origin') or {}
    if origin:
        candidates = [t for t in candidates if t['mcp_id'] == origin.get('mcp_id')
                      and t['instance_id'] == origin.get('instance_id')]
    if card_id:
        candidates = [t for t in candidates if t['instance_id'] == card_id]
    if not asset_id or len(candidates) != 1:
        raise fastapi.HTTPException(409, '无法唯一确定动作所属设备，请检查原示教卡连接')
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


@router.get('/export/{slug}')
async def export_package(slug: str, card_id: str = ''):
    _, target, asset_id = resolve_skill(slug, card_id)
    start = await call(target, 'package_export', asset_id=asset_id)
    if type(start.get('bytes')) is not int or not 1 <= start['bytes'] <= MAX_BYTES:
        raise fastapi.HTTPException(502, '设备返回的技能包大小无效')
    content = bytearray()
    while len(content) < start['bytes']:
        part = await call(target, 'package_read', transfer_id=start['transfer_id'], offset=len(content))
        try:
            chunk = base64.b64decode(part['data'], validate=True)
        except (KeyError, ValueError):
            raise fastapi.HTTPException(502, '设备返回无效技能包分块')
        if not chunk or len(chunk) > 24576 or len(content)+len(chunk) > start['bytes']:
            raise fastapi.HTTPException(502, '设备返回无效技能包长度')
        content.extend(chunk)
    if hashlib.sha256(content).hexdigest() != start.get('sha256'):
        raise fastapi.HTTPException(502, '动作包传输校验失败')
    manifest_from(content)
    return Response(bytes(content), media_type='application/zip', headers={
        'Content-Disposition': 'attachment; filename="action.motus-skill"', 'Cache-Control': 'no-store'})


@router.post('/install')
async def install_package(file: fastapi.UploadFile = fastapi.File(), card_id: str = fastapi.Form()):
    target = target_for(card_id)
    data = await read_upload(file)
    manifest_from(data)
    start = await call(target, 'package_begin', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    for offset in range(0, len(data), 24576):
        await call(target, 'package_write', transfer_id=start['transfer_id'], offset=offset,
                   data=base64.b64encode(data[offset:offset+24576]).decode())
    result = await call(target, 'package_finish', transfer_id=start['transfer_id'])
    if result.get('reused'):
        from api.skills import import_local_skill
        definition = {**result['skill'], 'origin': {'mcp_id': target['mcp_id'], 'instance_id': target['instance_id']}}
        existing = next((s for s in config.main.get('skills', {}).get('installed', []) if s['slug'] == definition['slug']), None)
        if existing:
            # Re-import must not reset visibility or overwrite edited metadata.
            _, original_target, original_asset = resolve_skill(existing['slug'])
            if original_target != target or original_asset != result['asset_id']:
                raise fastapi.HTTPException(409, '现有技能来源不匹配')
        else:
            await import_local_skill(definition)
        return {'code': 200, 'data': {'reused': True, 'installed': True, 'slug': definition['slug'], 'target': target}}
    # No installed Skill is created here. Driver's normal trial/save workflow
    # issues a new local asset and uses the ordinary skill import endpoint.
    return {'code': 200, 'data': {**result, 'target': target}}
