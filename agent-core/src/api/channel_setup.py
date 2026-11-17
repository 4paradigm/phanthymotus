"""QQ setup, policy and onboarding APIs.

Legacy manual credential entry may use process-memory references; the QR flow
stores the Tencent-issued credential in the managed private secret directory.
"""
import asyncio
import os
import re
import fastapi
from pydantic import BaseModel

import auth
from channel.manager import (
    add_channel_config, get_channel_config, update_channel_config,
    _get_channel_configs, channel_request_topic, manager,
)
from channel import (qq_onboarding, qq_runtime_credentials,
                     qq_credential_store, qq_qr_onboarding)
from channel.qq_capabilities import apply_safe_defaults, describe as describe_capabilities
from channel.qq_diagnostics import (snapshot as qq_telemetry_snapshot,
                                    classify_gateway_failure)
from channel.permissions import assign_preset
from channel.adapters.qq_official import validate_qq_official_config
from channel.qq_default_flow import ensure_default_flow

router = fastapi.APIRouter(prefix='/channel', tags=['channel'])
_QQ_QR_CONFIG_LOCKS: dict[str, asyncio.Lock] = {}


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
            'admin_pairing_required': False,
            'credential_ready': _credential_ready(cfg),
            'flow_ready': ch['id'] in manager.active_input_channels
                          and ch['id'] in manager.active_output_channels,
            'input_subscription_ready': manager.is_input_subscription_ready(ch['id']),
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
    cfg = apply_safe_defaults({
        'protocol': 'official', 'app_id': app_id,
        'enabled_scopes': ['c2c', 'group'], 'private_only': False,
        'access_policy': {'default_permission': 'chat'},
    })
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


async def _configure_scanned_qq(app_id: str, app_secret: str) -> dict:
    """Persist a Tencent-scanned credential and attach it to the existing Core.

    The credential file is committed before starting the Gateway so a transient
    network outage cannot leave a successfully scanned bot with a process-only
    secret that disappears on restart. Existing Channel settings are preserved.
    """
    app_id = str(app_id or '').strip()
    app_secret = str(app_secret or '').strip()
    if not re.fullmatch(r'[0-9]{5,32}', app_id):
        raise qq_qr_onboarding.QRSessionError('QQ 平台返回的机器人信息无效，请重新扫码。')
    if not 8 <= len(app_secret) <= 512:
        raise qq_qr_onboarding.QRSessionError('QQ 授权凭证无效，请重新扫码。')

    matches = [
        item for item in _get_channel_configs()
        if item.get('platform') == 'qq'
        and str((item.get('config') or {}).get('app_id', '')) == app_id
    ]
    if len(matches) > 1:
        raise qq_qr_onboarding.QRSessionError(
            '发现重复的 QQ 通道配置，请先在渠道列表清理重复项。')
    channel_id = matches[0]['id'] if matches else 'qq_official_' + app_id
    lock = _QQ_QR_CONFIG_LOCKS.setdefault(channel_id, asyncio.Lock())

    async with lock:
        # Re-check after acquiring the per-AppID lock; two simultaneous scans
        # for the same bot must not create two Channel records.
        current = get_channel_config(channel_id)
        is_existing = current is not None
        old_cfg = dict((current or {}).get('config') or {})
        if current and current.get('platform') != 'qq':
            raise qq_qr_onboarding.QRSessionError(
                '当前渠道编号已被其他平台占用，请检查渠道配置。')
        if current and str(old_cfg.get('app_id', '')) != app_id:
            raise qq_qr_onboarding.QRSessionError(
                'QQ 机器人编号与已有渠道不一致，未覆盖原有配置。')
        secret_path = await asyncio.to_thread(
            qq_credential_store.persist, app_id, app_secret)

        cfg = dict(old_cfg) if is_existing else {
            'protocol': 'official',
            'app_id': app_id,
            'enabled_scopes': ['c2c', 'group'],
            'private_only': False,
            'access_policy': {'default_permission': 'chat'},
        }
        for key in ('app_secret', 'app_secret_env', 'app_secret_runtime',
                    'app_secret_file'):
            cfg.pop(key, None)
        cfg['protocol'] = 'official'
        cfg['app_id'] = app_id
        cfg['app_secret_file'] = secret_path
        cfg = apply_safe_defaults(
            cfg,
            standard_mode=cfg.get('qq_config_mode', 'standard') == 'standard',
        )

        try:
            validate_qq_official_config(cfg)
            if is_existing:
                updated = await asyncio.to_thread(
                    update_channel_config, channel_id, config=cfg, enabled=True)
                if updated is None:
                    raise RuntimeError('QQ Channel disappeared during QR setup')
            else:
                await asyncio.to_thread(
                    add_channel_config, channel_id, 'qq', cfg, True)
        except Exception:
            await asyncio.to_thread(qq_credential_store.delete, app_id, secret_path)
            raise

        # The existing Canvas is never replaced. If it contains multiple
        # decision cores, preserve it and report that manual selection is needed.
        flow_error = False
        try:
            flow = ensure_default_flow(channel_id)
        except Exception as exc:
            flow_error = True
            flow = {'ready': False, 'reason': type(exc).__name__}
        # ensure_default_flow synchronizes Canvas card configuration. Reconcile
        # the live DDS subscription separately: a saved card is not a live input.
        subscription_state = manager.sync_channel_subscriptions()
        input_subscription_ready = manager.is_input_subscription_ready(channel_id)
        input_topic = channel_request_topic(channel_id)
        subscription_pending = input_topic in subscription_state.get('pending_subscriptions', [])
        flow_ready = bool(
            flow.get('ready')
            and channel_id in manager.active_input_channels
            and channel_id in manager.active_output_channels
        )

        # Credentials are already stored in the persistent private file.
        # A temporary Gateway/network failure is recoverable by the normal
        # ChannelManager watchdog and must not force the user to scan again.
        try:
            await asyncio.wait_for(
                manager.restart_adapter(channel_id, retries=1), timeout=60)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print('[qq-onboard] adapter start pending: ' + type(exc).__name__,
                  flush=True)
        connected = bool(manager._health.get(channel_id, (False, ''))[0])

        if is_existing:
            old_ref = old_cfg.get('app_secret_runtime', '')
            if old_ref:
                qq_runtime_credentials.revoke(old_ref)
            old_path = old_cfg.get('app_secret_file', '')
            if old_path and old_path != secret_path:
                await asyncio.to_thread(
                    qq_credential_store.delete, app_id, old_path)

        if not flow_ready:
            if flow_error:
                message = (
                    'QQ 已授权并安全保存，但自动连接助手流程时发生错误。'
                    '原有画布未被覆盖，请在 Dashboard 检查 Canvas 后重试。'
                )
            else:
                message = (
                    'QQ 已授权并安全保存，但当前画布无法自动确定唯一助手。'
                    '为保护已有项目，没有改动现有画布；请在 Dashboard 选择一个助手。'
                )
            if not connected:
                message += ' QQ Gateway 连接也在自动重试。'
        elif not connected:
            message = (
                'QQ 已授权，配置已安全保存。当前网络或 Gateway 尚未就绪，'
                '系统会自动重试；可稍后在渠道列表查看状态。'
            )
        elif input_subscription_ready:
            message = 'QQ 已连接，消息输入订阅已就绪，可以正常收发消息。'
        elif subscription_pending:
            message = (
                'QQ 已连接，已加入当前助手流程。智能控制启动时会自动激活消息输入订阅。'
            )
        else:
            message = (
                'QQ 已连接，但 Agent Core 的消息输入订阅尚未就绪。'
                '请启动智能控制检查订阅状态；无需重新扫码。'
            )
        return {
            'channel_id': channel_id,
            'connected': connected,
            'flow_ready': flow_ready,
            'input_subscription_ready': input_subscription_ready,
            'subscription_pending': subscription_pending,
            'configured': True,
            'message': message,
        }


@router.post('/qq/onboard/start')
async def start_qq_qr_onboard(response: fastapi.Response):
    _setup_auth()
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    try:
        return await qq_qr_onboarding.create_session(_configure_scanned_qq)
    except qq_qr_onboarding.QRSessionError as exc:
        raise fastapi.HTTPException(429, str(exc)) from None


@router.get('/qq/onboard/status')
def qq_qr_onboard_status(session_id: str, response: fastapi.Response):
    _setup_auth()
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    try:
        return qq_qr_onboarding.get_session(session_id)
    except qq_qr_onboarding.QRSessionError as exc:
        raise fastapi.HTTPException(404, str(exc)) from None


class QQQRSessionCancel(BaseModel):
    session_id: str


@router.post('/qq/onboard/cancel')
def cancel_qq_qr_onboard(req: QQQRSessionCancel, response: fastapi.Response):
    _setup_auth()
    response.headers['Cache-Control'] = 'no-store'
    try:
        return qq_qr_onboarding.cancel_session(req.session_id)
    except qq_qr_onboarding.QRSessionError as exc:
        raise fastapi.HTTPException(404, str(exc)) from None


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
    telemetry = qq_telemetry_snapshot(channel_id)
    gateway_ready, gateway_reason = manager._health.get(channel_id, (False, ''))
    gateway_ready = bool(gateway_ready)
    credential_ready = _credential_ready(cfg)
    flow_input = channel_id in manager.active_input_channels
    flow_output = channel_id in manager.active_output_channels
    return {
        'gateway': 'connected' if gateway_ready else ch.get('status', 'disconnected'),
        'flow_input': flow_input,
        'flow_output': flow_output,
        'credential_ready': credential_ready,
        'connection_issue': ('NONE' if gateway_ready else classify_gateway_failure(
            gateway_reason, credential_ready=credential_ready)),
        'admin_pairing_required': False,
        'capabilities': describe_capabilities(
            cfg, gateway=gateway_ready,
            flow_ready=flow_input and flow_output, recent=telemetry),
        'events': telemetry,
        'note': 'Platform accepted is not proof the user read the message. '
                'Counters reset when Agent Core restarts.',
    }


@router.get('/qq/setup/test-status')
def qq_test_status(channel_id: str, since: float = 0):
    """Observe any QQ C2C greeting after setup, without binding an owner.

    A reply row is written only after the QQ SDK send call returned normally;
    this is platform acceptance, NOT proof that a human received or read it.
    The UI requests human confirmation separately. Only booleans leave this API.
    """
    _setup_auth()
    _qq(channel_id)
    import config
    import math
    import time
    now = time.time()
    if not math.isfinite(since):
        raise fastapi.HTTPException(400, 'Invalid verification start time')
    lower_bound = max(now - 900, min(float(since), now + 5))
    per_channel = (config.main.get('channel_chat_history') or {}).get(channel_id, {})
    if not isinstance(per_channel, dict):
        return {'message_received': False, 'reply_sent': False}
    received, accepted = False, False
    for chat_id, history in list(per_channel.items())[-128:]:
        if not isinstance(chat_id, str) or not chat_id.startswith('p2p:'):
            continue
        if not isinstance(history, list):
            continue
        greetings = [
            float(row['ts']) for row in history if isinstance(row, dict)
            and row.get('role') == 'user'
            and isinstance(row.get('text'), str)
            and row['text'].strip() == '你好'
            and isinstance(row.get('ts'), (int, float))
            and math.isfinite(row['ts']) and row['ts'] >= lower_bound
        ]
        if not greetings:
            continue
        received = True
        latest = max(greetings)
        if any(isinstance(row, dict) and row.get('role') == 'assistant'
               and isinstance(row.get('ts'), (int, float))
               and math.isfinite(row['ts']) and row['ts'] >= latest
               for row in history):
            accepted = True
            break
    return {'message_received': received, 'reply_sent': accepted}


@router.post('/qq/pair/start')
def pair_start(req: Pairing):
    _setup_auth()
    _qq(req.channel_id)
    raise fastapi.HTTPException(410, 'QQ administrator pairing is retired; all chat users can participate')


@router.get('/qq/pair/status')
def pair_status(channel_id: str):
    _setup_auth()
    _qq(channel_id)
    return qq_onboarding.status(channel_id)


@router.post('/qq/pair/confirm')
def pair_confirm(req: Pairing):
    _setup_auth()
    _qq(req.channel_id)
    raise fastapi.HTTPException(410, 'QQ administrator pairing is retired')


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
    # A valid QQ Gateway event itself proves reachability; no separate
    # group authorization step is required on our side.
    old_cfg = dict(ch.get('config') or {})
    scopes = list(old_cfg.get('enabled_scopes', ['c2c']))
    if req.enabled and 'group' not in scopes:
        scopes.append('group')
    if not req.enabled:
        scopes = [scope for scope in scopes if scope != 'group']
    if not scopes:
        raise fastapi.HTTPException(400, 'At least one QQ event scope is required')
    cfg = dict(old_cfg, enabled_scopes=scopes,
               private_only=set(scopes) == {'c2c'},
               qq_group_scope_opt_out=not req.enabled)
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
