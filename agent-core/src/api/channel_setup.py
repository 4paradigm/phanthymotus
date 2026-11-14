"""QQ guided setup, policy and verified administrator pairing.

The onboarding password is process-memory only. Managed deployments can use a
preconfigured environment key or an already-mounted read-only secret file.
"""
import asyncio
import os
import re
import fastapi
from pydantic import BaseModel

import auth
from channel.manager import (
    add_channel_config, get_channel_config, update_channel_config,
    _get_channel_configs, manager,
)
from channel import qq_onboarding, qq_runtime_credentials
from channel.permissions import assign_preset
from channel.adapters.qq_official import validate_qq_official_config
from channel.qq_default_flow import ensure_default_flow

router = fastapi.APIRouter(prefix='/channel', tags=['channel'])


def _setup_auth():
    # Dashboard must be authenticated before granting remote-control rights.
    if not auth.is_enabled():
        raise fastapi.HTTPException(403, 'Dashboard ACCESS_TOKEN is required for QQ setup')


class QQSetup(BaseModel):
    app_id: str
    app_secret: str = ''
    source: str = 'persistent'   # persistent (default), runtime, legacy env/file
    env_name: str = ''
    receive_images: bool = True


class QQRuntimeRefresh(BaseModel):
    channel_id: str
    app_secret: str


class Pairing(BaseModel):
    channel_id: str


class Preset(BaseModel):
    platform: str
    user_id: str
    preset: str


class AccessPolicy(BaseModel):
    default_permission: str = 'chat'


def _policy(value: AccessPolicy) -> dict:
    if value.default_permission != 'chat':
        raise fastapi.HTTPException(400, 'Ordinary QQ users are chat-only')
    return {'default_permission': value.default_permission}


def _credential_ready(cfg: dict) -> bool:
    ref = cfg.get('app_secret_runtime')
    if ref:
        return qq_runtime_credentials.available(ref)
    env_name = cfg.get('app_secret_env')
    if env_name:
        return bool(os.environ.get(env_name))
    if cfg.get('app_secret_file'):
        from pathlib import Path
        path = Path(cfg['app_secret_file'])
        return path.is_file() and os.access(path, os.R_OK)
    return False


def _qq(channel_id: str) -> dict:
    channel = get_channel_config(channel_id)
    if channel is None or channel.get('platform') != 'qq':
        raise fastapi.HTTPException(404, 'QQ channel not found')
    return channel


@router.get('/qq/setup/status')
def guided_status():
    _setup_auth()
    from channel import qq_runtime_credentials as runtime
    manager.sync_from_canvas()
    channels = []
    owner_grant = __import__('config').main.get('qq_c2c_owner_grant', {})
    for ch in _get_channel_configs():
        if ch.get('platform') != 'qq':
            continue
        cfg = ch.get('config', {})
        reference = cfg.get('app_secret_runtime', '')
        from channel.qq_credential_store import is_managed_reference
        file_path = cfg.get('app_secret_file', '')
        mode = ('runtime' if reference
                else 'persistent' if file_path and is_managed_reference(
                    str(cfg.get('app_id', '')), file_path)
                else 'env' if cfg.get('app_secret_env')
                else 'mounted_file' if file_path else 'legacy')
        channels.append({
            'channel_id': ch['id'], 'app_id': cfg.get('app_id', ''),
            'enabled': ch.get('enabled', False),
            'status': ('connected' if manager._health.get(ch['id'], (False, ''))[0]
                       else ch.get('status', 'disconnected')),
            'credential_mode': mode,
            'group_scope_enabled': 'group' in cfg.get('enabled_scopes', ['c2c']),
            'owner_bound': bool(isinstance(owner_grant, dict)
                                and owner_grant.get('enabled')
                                and str(owner_grant.get('principal', '')).startswith(
                                    'official:' + str(cfg.get('app_id', '')) + ':')),
            'credential_ready': _credential_ready(cfg),
            'flow_ready': ch['id'] in manager.active_input_channels
                          and ch['id'] in manager.active_output_channels,
        })
    return {'channels': channels, 'credential_policy': 'persistent_default_runtime_optional'}


@router.post('/qq/guided-setup')
async def guided_setup(req: QQSetup):
    _setup_auth()
    app_id = req.app_id.strip()
    if not re.fullmatch(r'[0-9]{5,32}', app_id):
        raise fastapi.HTTPException(400, 'QQ AppID must contain 5–32 digits')
    for existing in _get_channel_configs():
        if (existing.get('platform') == 'qq'
                and str(existing.get('config', {}).get('app_id')) == app_id):
            raise fastapi.HTTPException(409, {
                'message': 'QQ 机器人已存在，请恢复原有渠道而不是重新创建。',
                'channel_id': existing['id'],
            })
    channel_id = 'qq_official_' + app_id
    cfg = {'protocol': 'official', 'app_id': app_id,
           'enabled_scopes': ['c2c'], 'private_only': True,
           'enable_inbound_media': req.receive_images,
           'access_policy': {'default_permission': 'chat',
                             'default_permission': 'chat'}}
    ref = ''
    if req.source in ('runtime', 'persistent'):
        try:
            ref = qq_runtime_credentials.put(req.app_secret)
        except ValueError as exc:
            raise fastapi.HTTPException(400, str(exc)) from None
        cfg['app_secret_runtime'] = ref
    elif req.source == 'env':
        if not re.fullmatch(r'QQ_OFFICIAL_[A-Z0-9_]{1,64}', req.env_name):
            raise fastapi.HTTPException(400, 'Invalid environment variable name')
        if not os.environ.get(req.env_name, '').strip():
            raise fastapi.HTTPException(400,
                '环境变量尚未在 Agent Core 运行环境中配置。请使用默认的直接输入 AppSecret，'
                '或者先让部署管理员配置这个环境变量。')
        cfg['app_secret_env'] = req.env_name
    elif req.source == 'mounted_file':
        from pathlib import Path
        secret = Path('/run/qq-official-app-secret')
        if not secret.is_file() or not os.access(secret, os.R_OK):
            raise fastapi.HTTPException(400,
                'Agent Core 中没有可读取的密钥挂载。请改选默认的直接输入 AppSecret。')
        try:
            if not secret.read_text(encoding='utf-8').strip():
                raise fastapi.HTTPException(400, '挂载的密钥文件为空，请检查部署配置。')
        except (OSError, UnicodeError):
            raise fastapi.HTTPException(400, '无法读取挂载的密钥文件，请检查部署配置。') from None
        cfg['app_secret_file'] = str(secret)
    else:
        raise fastapi.HTTPException(400, 'Unsupported credential source')
    try:
        validate_qq_official_config(cfg)
        await asyncio.to_thread(add_channel_config, channel_id, 'qq', cfg, True)
    except Exception:
        if ref:
            qq_runtime_credentials.revoke(ref)
        raise
    # Only an empty Canvas is automatically populated; existing work is kept.
    flow = ensure_default_flow(channel_id)
    await manager.restart_adapter(channel_id, retries=1)
    ok, reason = manager._health.get(channel_id, (False, 'Not connected'))
    if ok and req.source == 'persistent':
        from api.qq_settings import promote_verified_credential
        try:
            await promote_verified_credential(channel_id, cfg, ref, req.app_secret)
        except Exception:
            # Do not misreport a temporary connection as durably saved.
            return {'channel_id': channel_id, 'connected': False,
                    'detail': 'QQ connection established, but durable credential saving failed. Retry setup.',
                    'default_flow': flow, 'credential_mode': 'runtime'}
    return {'channel_id': channel_id, 'connected': bool(ok),
            'detail': 'Connected' if ok else 'Not connected: check QQ credentials, network and platform permissions.',
            'default_flow': flow, 'credential_mode': req.source}


@router.post('/qq/setup/runtime-credential')
async def refresh_runtime_credential(req: QQRuntimeRefresh):
    _setup_auth()
    ch = _qq(req.channel_id)
    old_ref = ch.get('config', {}).get('app_secret_runtime', '')
    if not old_ref:
        raise fastapi.HTTPException(400, 'Channel is not configured for runtime credentials')
    try:
        new_ref = qq_runtime_credentials.put(req.app_secret)
    except ValueError as exc:
        raise fastapi.HTTPException(400, str(exc)) from None
    try:
        cfg = dict(ch['config'], app_secret_runtime=new_ref)
        await asyncio.to_thread(update_channel_config, req.channel_id, config=cfg)
        await manager.restart_adapter(req.channel_id, retries=1)
    except Exception:
        qq_runtime_credentials.revoke(new_ref)
        raise
    qq_runtime_credentials.revoke(old_ref)
    ok, reason = manager._health.get(req.channel_id, (False, 'Not connected'))
    return {'connected': bool(ok), 'detail': 'Connected' if ok else 'Not connected: check QQ credentials, network and platform permissions.'}


@router.post('/qq/setup/default-flow')
def default_flow(req: Pairing):
    _setup_auth()
    _qq(req.channel_id)
    return ensure_default_flow(req.channel_id)


@router.get('/qq/setup/diagnostics')
def diagnostics(channel_id: str):
    _setup_auth()
    ch = _qq(channel_id)
    manager.sync_from_canvas()
    cfg = ch.get('config', {})
    owner = __import__('config').main.get('qq_c2c_owner_grant', {})
    return {
        'gateway': ('connected' if manager._health.get(channel_id, (False, ''))[0]
                    else ch.get('status', 'disconnected')),
        'flow_input': channel_id in manager.active_input_channels,
        'flow_output': channel_id in manager.active_output_channels,
        'credential_ready': _credential_ready(cfg),
        'owner_bound': bool(isinstance(owner, dict) and owner.get('enabled') and
                            owner.get('principal', '').startswith('official:' + cfg.get('app_id', '') + ':')),
        'note': 'Verify actual QQ send/receive separately; no physical Bumi is required',
    }


@router.post('/qq/pair/start')
def pair_start(req: Pairing):
    _setup_auth()
    ch = _qq(req.channel_id)
    if not ch.get('enabled'):
        raise fastapi.HTTPException(400, 'Enable QQ channel before pairing')
    return {'code': qq_onboarding.start(req.channel_id), 'expires_in_seconds': 300}


@router.get('/qq/pair/status')
def pair_status(channel_id: str):
    _setup_auth()
    _qq(channel_id)
    return qq_onboarding.status(channel_id)


@router.post('/qq/pair/confirm')
def pair_confirm(req: Pairing):
    _setup_auth()
    _qq(req.channel_id)
    try:
        return qq_onboarding.confirm(req.channel_id)
    except ValueError as exc:
        raise fastapi.HTTPException(400, str(exc)) from None


class PairRevocation(BaseModel):
    channel_id: str
    confirmation: str


@router.post('/qq/pair/revoke')
def pair_revoke(req: PairRevocation):
    _setup_auth()
    _qq(req.channel_id)
    if req.confirmation != '解除管理员绑定':
        raise fastapi.HTTPException(400, 'Explicit confirmation is required')
    try:
        return qq_onboarding.revoke(req.channel_id)
    except ValueError as exc:
        raise fastapi.HTTPException(400, str(exc)) from None


@router.get('/{channel_id}/access-policy')
def get_access_policy(channel_id: str):
    _setup_auth()
    ch = _qq(channel_id)
    return {'policy': {'default_permission': 'chat'}}


@router.put('/{channel_id}/access-policy')
def set_access_policy(channel_id: str, req: AccessPolicy):
    _setup_auth()
    ch = _qq(channel_id)
    cfg = dict(ch.get('config') or {})
    cfg['access_policy'] = _policy(req)
    update_channel_config(channel_id, config=cfg)
    return {'policy': cfg['access_policy']}


class QQGroupScope(BaseModel):
    channel_id: str
    enabled: bool
    platform_permission_confirmed: bool = False


@router.put('/qq/groups/scope')
async def set_group_scope(req: QQGroupScope):
    """Opt in to Tencent-authorized group events without affecting other scopes."""
    _setup_auth()
    ch = _qq(req.channel_id)
    if req.enabled and not req.platform_permission_confirmed:
        raise fastapi.HTTPException(
            400, 'Confirm QQ platform group-event permission before enabling')
    old_cfg = dict(ch.get('config') or {})
    scopes = list(old_cfg.get('enabled_scopes', ['c2c']))
    if req.enabled and 'group' not in scopes:
        scopes.append('group')
    if not req.enabled:
        scopes = [scope for scope in scopes if scope != 'group']
    if not scopes:
        raise fastapi.HTTPException(400, 'At least one QQ event scope is required')
    cfg = dict(old_cfg, enabled_scopes=scopes,
               private_only=set(scopes) == {'c2c'})
    try:
        validate_qq_official_config(cfg)
    except ValueError as exc:
        raise fastapi.HTTPException(400, str(exc)) from None
    # QQ-only atomic apply: a rejected Gateway subscription must not
    # leave the previously working C2C bot with an invalid saved scope.
    from api.qq_settings import _save_config, _locks
    lock = _locks.setdefault(req.channel_id, asyncio.Lock())
    if lock.locked():
        raise fastapi.HTTPException(409, 'Another QQ settings update is in progress')
    async with lock:
        result = await _save_config(req.channel_id, old_cfg, cfg)
    return {'group_scope_enabled': 'group' in scopes,
            'gateway_connected': result['connected'],
            'note': 'QQ groups are discovered automatically after the first authenticated group @message'}


class QQGroupChange(BaseModel):
    channel_id: str
    group_id: str
    default_permission: str = 'chat'
    mention_only: bool = True
    enabled: bool = True


@router.get('/qq/groups')
def group_list(channel_id: str):
    _setup_auth()
    _qq(channel_id)
    from channel import qq_groups
    return {'groups': qq_groups.list_groups(channel_id)}


@router.put('/qq/groups/permission')
def group_permission(req: QQGroupChange):
    _setup_auth()
    _qq(req.channel_id)
    from channel import qq_groups
    try:
        return qq_groups.update(req.channel_id, req.group_id,
                                default_permission=req.default_permission,
                                mention_only=req.mention_only, enabled=req.enabled)
    except ValueError as exc:
        raise fastapi.HTTPException(400, str(exc)) from None


@router.post('/users/preset')
def user_preset(req: Preset):
    _setup_auth()
    if req.platform != 'qq':
        raise fastapi.HTTPException(400, 'Only QQ permission presets are supported')
    if req.preset != 'chat':
        raise fastapi.HTTPException(400, 'Ordinary QQ users are chat-only')
    try:
        result = assign_preset(req.platform, req.user_id, req.preset)
        return {'user': result}
    except ValueError as exc:
        raise fastapi.HTTPException(400, str(exc)) from None
