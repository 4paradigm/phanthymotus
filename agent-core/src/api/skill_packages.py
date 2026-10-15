"""Skill-file transfer through the selected card. Robot validation stays in Driver."""
import base64
import copy
import hashlib
import io
import json
import zipfile

import fastapi
from fastapi.responses import Response
import config

router = fastapi.APIRouter(prefix='/package')
SCHEMA = 'motus.teaching.package/1'
MAX_BYTES = 16 * 1024 * 1024


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
    # No installed Skill is created here. Driver's normal trial/save workflow
    # issues a new local asset and uses the ordinary skill import endpoint.
    return {'code': 200, 'data': {**result, 'target': target}}
