"""QQ-only standard/advanced settings with bounded, reversible Gateway apply.

Do not change credentials in this API. SDK capability subscription and human
authorization are separate; other messaging platforms are never touched.
"""
import asyncio
from typing import Literal

from channel import qq_runtime_credentials, qq_credential_store

import fastapi
from pydantic import BaseModel, Field

import auth
from channel import acl
from channel.adapters.qq_official import validate_qq_official_config
from channel.manager import get_channel_config, update_channel_config, manager
from channel.permissions import CHAT_FILTER, DAILY_FILTER, OWNER_FILTER
from channel.tool_policy import _qq_owner_dm_identity

router = fastapi.APIRouter(prefix='/channel/qq/manage', tags=['qq management'])

SCOPES = ('c2c', 'group', 'guild', 'guild_dm')
FLAGS = (
    'enable_inbound_media', 'enable_outbound_media', 'enable_typing',
    'enable_session_resume', 'enable_markdown', 'enable_voice_transcription',
    'enable_guild_plain_messages', 'enable_interactions', 'enable_approvals',
)
# No new Gateway intent is needed for these optional SDK transport capabilities.
STANDARD_FLAGS = ('enable_inbound_media', 'enable_outbound_media',
                  'enable_typing', 'enable_session_resume')
_locks: dict[str, asyncio.Lock] = {}


def _channel(channel_id):
    if not auth.is_enabled():
        raise fastapi.HTTPException(403, 'Dashboard authentication must be configured')
    channel = get_channel_config(channel_id)
    if channel is None or channel.get('platform') != 'qq':
        raise fastapi.HTTPException(404, 'QQ channel not found')
    return channel


def _credential_available(cfg):
    try:
        from channel.adapters.qq_official import _load_credential
        return bool(_load_credential(cfg))
    except (OSError, ValueError):
        return False


def _owner(channel):
    import config
    cfg = channel.get('config') or {}
    grant = config.main.get('qq_c2c_owner_grant', {})
    uid = grant.get('principal', '') if isinstance(grant, dict) else ''
    verified = (bool(grant.get('enabled')) if isinstance(grant, dict) else False)
    verified = (verified and _qq_owner_dm_identity(uid)
                and uid.split(':')[1] == str(cfg.get('app_id', '')))
    user = acl.get_user('qq', uid) if verified else None
    return uid if verified and user and user.get('role') == 'owner' else '', user


class QQSettingsChange(BaseModel):
    mode: Literal['standard', 'advanced']
    group_enabled: bool | None = None
    enabled_scopes: list[str] | None = None
    flags: dict[str, bool] = Field(default_factory=dict)


class QQCredentialRecovery(BaseModel):
    app_secret: str
    restore_basic: bool = True
    persist: bool = True


class QQOwnerChange(BaseModel):
    preset: Literal['full', 'daily', 'chat', 'custom']
    capabilities: list[Literal['speech', 'sensing', 'motion', 'system']] = Field(default_factory=list)


OWNER_CAPABILITY_PATTERNS = {
    'speech': ['mcp__*__tts', 'mcp__*__speaker*'],
    'sensing': ['mcp__*__camera*', 'mcp__*__watch*', 'mcp__*__sensor*'],
    'motion': ['mcp__*__loco*', 'mcp__*__motion*', 'mcp__*__move*'],
    'system': ['Bash', 'mcp__*__shell*', 'mcp__*__file*'],
}


@router.get('/{channel_id}/settings')
def get_settings(channel_id: str):
    channel = _channel(channel_id)
    cfg = channel.get('config') or {}
    uid, user = _owner(channel)
    from channel.qq_credential_store import is_managed_reference
    file_path = cfg.get('app_secret_file', '')
    credential = ('runtime' if cfg.get('app_secret_runtime') else
                  'persistent' if file_path and is_managed_reference(
                      str(cfg.get('app_id', '')), file_path) else
                  'env' if cfg.get('app_secret_env') else
                  'mounted_file' if file_path else 'legacy')
    return {
        'mode': cfg.get('qq_config_mode', 'standard'),
        'app_id': cfg.get('app_id'),
        'enabled_scopes': cfg.get('enabled_scopes', ['c2c']),
        'flags': {flag: bool(cfg.get(flag)) for flag in FLAGS},
        'credential_mode': credential,
        'credential_ready': _credential_available(cfg),
        'credential_source_unchanged': True,
        'connected': bool(manager._health.get(channel_id, (False, ''))[0]),
        'owner': {'bound': bool(uid), 'suffix': uid[-6:] if uid else '',
                  'filter': user.get('tool_filter') if user else '',
                  'role': 'owner' if uid else ''},
    }


async def _save_config(channel_id: str, original: dict, new_cfg: dict):
    """Apply new subscription with fail-safe rollback on start failure.

    Persisting a failed subscription is worse than keeping previously working
    private messages. Never catch BaseException: cancellations propagate.
    """
    if original == new_cfg:
        return {'saved': True, 'changed': False,
                'connected': bool(manager._health.get(channel_id, (False, ''))[0])}
    channel = _channel(channel_id)
    if not channel.get('enabled'):
        await asyncio.to_thread(update_channel_config, channel_id, config=new_cfg)
        return {'saved': True, 'changed': True, 'connected': False,
                'note': 'Channel is stopped; changes will apply on next start'}
    await asyncio.to_thread(update_channel_config, channel_id, config=new_cfg)
    reason = ''
    try:
        timeout = min(max(float(new_cfg.get('connect_timeout', 20)) + 12, 18), 75)
        await asyncio.wait_for(manager.restart_adapter(channel_id, retries=1),
                               timeout=timeout)
        healthy, health_reason = manager._health.get(channel_id, (False, ''))
        if not healthy:
            reason = str(health_reason)[:160] or 'QQ Gateway did not connect'
        else:
            return {'saved': True, 'changed': True, 'connected': True}
    except (asyncio.TimeoutError, Exception) as exc:
        reason = 'Gateway restart timed out or failed: ' + type(exc).__name__
    # Always restore exactly the previous configuration. No secret changes.
    await asyncio.to_thread(update_channel_config, channel_id, config=original)
    recovery = False
    try:
        await asyncio.wait_for(manager.restart_adapter(channel_id, retries=1),
                               timeout=min(float(original.get('connect_timeout', 20)) + 12, 75))
        recovery = bool(manager._health.get(channel_id, (False, ''))[0])
    except Exception:
        pass
    raise fastapi.HTTPException(
        409, {'message': 'QQ Gateway rejected the requested configuration; old settings restored',
              'recovered': recovery, 'reason': reason})


async def promote_verified_credential(channel_id: str, current_cfg: dict,
                                      runtime_ref: str, value: str) -> dict:
    """Promote a Gateway-verified credential to a private persisted file.

    The earlier credential/config stay intact until the file-backed Gateway
    restart succeeds, allowing rollback without losing the working connection.
    """
    app_id = str(current_cfg['app_id'])
    path = await asyncio.to_thread(qq_credential_store.persist, app_id, value)
    target = dict(current_cfg)
    target.pop('app_secret_runtime', None)
    target.pop('app_secret_env', None)
    target.pop('app_secret', None)
    target.pop('app_secret_file', None)
    target['app_secret_file'] = path
    try:
        validate_qq_official_config(target)
        result = await _save_config(channel_id, current_cfg, target)
        if not result.get('connected'):
            raise fastapi.HTTPException(409, 'QQ connection not verified; credential was not persisted')
    except Exception:
        await asyncio.to_thread(qq_credential_store.delete, app_id, path)
        raise
    qq_runtime_credentials.revoke(runtime_ref)
    old_path = current_cfg.get('app_secret_file', '')
    if old_path and old_path != path:
        await asyncio.to_thread(qq_credential_store.delete, app_id, old_path)
    return {**result, 'credential_mode': 'persistent',
            'restart_requires_reentry': False}


@router.post('/{channel_id}/credential')
async def recover_qq_credential(channel_id: str, req: QQCredentialRecovery):
    """Replace a broken credential with process-memory-only secret.

    Default to the smallest known-good C2C event subscription; the Owner may
    enable platform-approved group events independently after reconnection.
    """
    _channel(channel_id)
    lock = _locks.setdefault(channel_id, asyncio.Lock())
    if lock.locked():
        raise fastapi.HTTPException(409, 'Another QQ settings update is in progress')
    async with lock:
        previous = dict((_channel(channel_id).get('config') or {}))
        try:
            ref = qq_runtime_credentials.put(req.app_secret)
        except ValueError as exc:
            raise fastapi.HTTPException(400, str(exc)) from None
        cfg = dict(previous)
        old_ref = cfg.pop('app_secret_runtime', '')
        for key in ('app_secret', 'app_secret_env', 'app_secret_file'):
            cfg.pop(key, None)
        cfg['app_secret_runtime'] = ref
        if req.restore_basic:
            cfg['enabled_scopes'] = ['c2c']
            cfg['private_only'] = True
            cfg['enable_guild_plain_messages'] = False
            cfg['enable_interactions'] = False
            cfg['enable_approvals'] = False
            cfg['qq_config_mode'] = 'standard'
        try:
            validate_qq_official_config(cfg)
            result = await _save_config(channel_id, previous, cfg)
        except Exception:
            qq_runtime_credentials.revoke(ref)
            raise
        if not result.get('connected') and req.persist:
            return {**result, 'credential_mode': 'runtime',
                    'restart_requires_reentry': True,
                    'note': 'Connect the QQ Gateway before saving this credential'}
        if req.persist:
            try:
                saved = await promote_verified_credential(
                    channel_id, cfg, ref, req.app_secret)
            except Exception:
                # The verified temporary reference remains usable on rollback.
                # Do not pretend the key was durably stored.
                raise fastapi.HTTPException(
                    409, 'QQ Gateway connected, but persistent credential storage failed; '
                         'temporary connection remains available') from None
            old_path = previous.get('app_secret_file', '')
            if old_path:
                await asyncio.to_thread(qq_credential_store.delete,
                                        str(cfg['app_id']), old_path)
            result = saved
        else:
            result = {**result, 'credential_mode': 'runtime',
                      'restart_requires_reentry': True}
        if old_ref and old_ref != ref:
            qq_runtime_credentials.revoke(old_ref)
        return result


@router.put('/{channel_id}/settings')
async def save_settings(channel_id: str, req: QQSettingsChange):
    channel = _channel(channel_id)
    lock = _locks.setdefault(channel_id, asyncio.Lock())
    if lock.locked():
        raise fastapi.HTTPException(409, 'Another QQ settings update is in progress')
    async with lock:
        old = dict((_channel(channel_id).get('config') or {}))
        cfg = dict(old)
        cfg['qq_config_mode'] = req.mode
        if req.mode == 'standard':
            # Existing platform-scoped events are preserved; no unsupported
            # Guild/group permission is silently added during setup.
            for flag in STANDARD_FLAGS:
                cfg[flag] = True
        else:
            if req.enabled_scopes is not None:
                if not req.enabled_scopes or len(req.enabled_scopes) != len(set(req.enabled_scopes)):
                    raise fastapi.HTTPException(400, 'Select unique QQ message scopes')
                if any(scope not in SCOPES for scope in req.enabled_scopes):
                    raise fastapi.HTTPException(400, 'Unsupported QQ event scope')
                cfg['enabled_scopes'] = req.enabled_scopes
            if any(key not in FLAGS or not isinstance(value, bool)
                   for key, value in req.flags.items()):
                raise fastapi.HTTPException(400, 'Unsupported QQ capability setting')
            cfg.update(req.flags)
        if req.group_enabled is not None:
            scopes = [x for x in cfg.get('enabled_scopes', ['c2c']) if x != 'group']
            if req.group_enabled:
                scopes.append('group')
            cfg['enabled_scopes'] = scopes
        scopes = cfg.get('enabled_scopes', ['c2c'])
        cfg['private_only'] = scopes == ['c2c']
        try:
            validate_qq_official_config(cfg)
        except ValueError as exc:
            raise fastapi.HTTPException(400, str(exc)) from None
        if old != cfg and channel.get('enabled') and not _credential_available(old):
            raise fastapi.HTTPException(
                409, 'QQ credential unavailable; use Restore QQ connection before saving SDK options')
        result = await _save_config(channel_id, old, cfg)
        return {**result, 'mode': req.mode,
                'group_enabled': 'group' in cfg['enabled_scopes']}


@router.put('/{channel_id}/owner')
def update_owner(channel_id: str, req: QQOwnerChange):
    ch = _channel(channel_id)
    uid, user = _owner(ch)
    if not uid or not user:
        raise fastapi.HTTPException(409, 'Bind and verify a QQ Owner first')
    if req.preset == 'full':
        filt = OWNER_FILTER
    elif req.preset == 'daily':
        filt = DAILY_FILTER
    elif req.preset == 'chat':
        filt = CHAT_FILTER
    else:
        if not req.capabilities or len(set(req.capabilities)) != len(req.capabilities):
            raise fastapi.HTTPException(400, 'Choose distinct capability categories')
        patterns = [CHAT_FILTER]
        for category in req.capabilities:
            patterns += OWNER_CAPABILITY_PATTERNS[category]
        filt = ','.join(patterns)
    updated = acl.upsert_user('qq', uid, user.get('display_name', ''),
                              role='owner', tool_filter=filt)
    return {'role': 'owner', 'user_suffix': uid[-6:],
            'tool_filter': updated['tool_filter'], 'preset': req.preset}
