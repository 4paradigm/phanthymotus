"""Whether a skill may run unattended, and which device it belongs to.

Core stores one decision per (skill, card, recording): has the operator allowed
this skill to run without being asked each time? The device reads it over a
loopback-only, read-only endpoint before every preview and run; nothing else can
grant it -- not an MCP `config` argument, not metadata inside an imported skill.

It also resolves a skill's identity (slug -> which card, which asset) and lists
drafts that are installed but not yet verified on this machine.

**What used to live here and no longer does.** This file was `api/skill_packages`
and carried a second file transport: `package_begin` / `package_write` /
`package_finish` in 24 KiB base64 chunks, plus the zip validation that went with
forwarding a whole archive, plus a bespoke import dialog in the skill panel. All
of it existed for one reason -- no driver implemented `POST /file/upload`, so
there was no way to get bytes into a driver container.

That gap is closed (phanthymotus-driver#417 wires the endpoint into all 15 HTTP
bundles, from a copy of the module perception already used), so the generic path
applies: a tool declares `format: file` on a parameter, the canvas renders a file
picker for it, and `mcp_client._transfer_file_args` pushes the file to whichever
service that tool belongs to and rewrites the argument. The tool the caller
invoked *is* the destination -- no protocol declaration, no card selection
dialog, no second transport.

Two transports for one job is the thing worth not shipping. See
`phanthymotus/README.md` § File Arguments and § Skill Files, and that repo's
README_dev.md § File Arguments for the per-bundle glue.
"""
import copy
import re
import secrets

import asyncio
import fastapi

import config

router = fastapi.APIRouter(prefix='/package')

# Opaque device-side identifiers. Core compares and forwards them; it does not
# know whether they are hashes, UUIDs or row ids, so it bounds the length and
# the alphabet and stops there.
_OPAQUE = re.compile(r'[A-Za-z0-9_.:-]{1,128}')
_POLICY_KEY = 'skill_execution_policies'

MANUAL = {'enabled': False, 'source': 'manual', 'binding_changed': False}


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


# ── the policy ───────────────────────────────────────────────────────────────

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
    form. One existed: a single checkbox authorised every skill the card would
    *later* verify, and buying it cost an interception of the generic
    `PUT /api/canvas/tool-config` endpoint, device I/O inside a config save, and
    a private field stored in the user's own config blob.
    """
    policies = config.main.get(_POLICY_KEY, {})
    policy = policies.get(execution_key(target) + ':' + asset_id, {}) if asset_id else {}
    requested = policy.get('enabled') is True
    enabled = (requested and policy.get('device_binding') == device_binding
               and policy.get('asset_binding') == asset_binding)
    return {'enabled': enabled, 'source': 'skill' if enabled else 'manual',
            'binding_changed': requested and not enabled}


async def binding_of(target, asset_id=''):
    args = {'asset_id': asset_id} if asset_id else {}
    return read_binding(await call(target, 'execution_binding', **args), asset_id=asset_id)


@router.post('/execution/read')
async def read_execution(request: fastapi.Request, body: dict = fastapi.Body(...)):
    """What the device asks before every preview/run. Read-only, loopback only.

    It can observe a permission, never create one, which is why same-host origin
    is enough here while every write below needs a browser token.
    """
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
    """Cards whose tool declares a skill-package protocol, whatever it is.

    The protocol identifier is the card's own (`x-skill-package.schema`); Core
    holds none of its own and does not interpret it. What Core needs it for is
    knowing which cards are in this family at all.
    """
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
        import json
        try:
            data = json.loads(next(c['text'] for c in data if c.get('type') == 'text'))
        except (StopIteration, ValueError, KeyError, TypeError):
            raise fastapi.HTTPException(502, '设备返回无效结果')
    if not isinstance(data, dict) or data.get('error'):
        raise fastapi.HTTPException(409, data.get('error') if isinstance(data, dict)
                                   else '设备返回无效结果')
    return data


# ── drafts awaiting local verification ───────────────────────────────────────

@router.get('/pending')
async def pending_packages():
    """Drafts installed on a card but not yet verified here.

    Aggregated by Core rather than the page: finding them means walking the
    canvas layout against the MCP registry, which is the same reason routing
    does not live in the browser.
    """
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


# ── identity ─────────────────────────────────────────────────────────────────

def resolve_skill(slug, card_id='', known=None):
    """The (skill, card, asset) a packaged skill belongs to.

    Identity comes from `attachment` and nothing else. A slug-prefix fallback
    for skills installed before that field existed lived here briefly; one test
    rig had such a row, and a naming convention is not an identity mechanism --
    fix the row, not the resolver.
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

    Transitional: a skill has one mutable definition, so re-recording rewrites
    it in place. Once the resource centre stores versioned packages
    (resource-center!66), re-recording produces a new version and this goes away.
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
