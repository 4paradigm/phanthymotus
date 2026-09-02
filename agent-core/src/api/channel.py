"""
api/channel.py — Channel 管理 REST API。

端点：
- GET  /api/channel/list          — 列出所有 channel 及状态
- POST /api/channel/add           — 添加 channel 配置
- PUT  /api/channel/{id}          — 更新 channel 配置
- DELETE /api/channel/{id}        — 删除 channel
- POST /api/channel/{id}/restart  — 重启 adapter
- GET  /api/channel/users         — 列出用户
- POST /api/channel/users         — 添加/更新用户
- DELETE /api/channel/users       — 删除用户
- GET  /api/channel/settings      — 获取 channel 全局设置
- PUT  /api/channel/settings      — 更新 channel 全局设置
"""

import asyncio

import fastapi
from pydantic import BaseModel, Field

from channel.manager import (
    manager, get_channel_config, add_channel_config,
    update_channel_config, delete_channel_config,
    _get_channel_configs, supports_bot_to_bot, uses_trusted_bots,
)
from channel import acl
import config

router = fastapi.APIRouter(prefix='/channel', tags=['channel'])


# ── Channel CRUD ─────────────────────────────────────────────────────────────

@router.get('/list')
def list_channels():
    return {'channels': manager.get_status()}


class AddChannelReq(BaseModel):
    id: str
    platform: str
    config: dict = {}
    enabled: bool = False
    bot_to_bot_enabled: bool = False
    trusted_bots: list[dict] = Field(default_factory=list)


@router.post('/add')
async def add_channel(req: AddChannelReq):
    if req.bot_to_bot_enabled and not supports_bot_to_bot(req.platform):
        raise fastapi.HTTPException(
            400, f'bot_to_bot_enabled is not supported by platform "{req.platform}"')
    if req.trusted_bots and not uses_trusted_bots(req.platform):
        raise fastapi.HTTPException(
            400,
            f'trusted_bots is not used by platform "{req.platform}"'
            + (' — lan peers are trusted via pairing (Settings → Peers), not this list.'
               if req.platform == 'lan' else ''))
    try:
        entry = await asyncio.to_thread(
            add_channel_config,
            req.id,
            req.platform,
            req.config,
            req.enabled,
            req.bot_to_bot_enabled,
            req.trusted_bots,
        )
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))
    # 如果 enabled，立即启动
    if req.enabled:
        await manager.restart_adapter(req.id)
    return {'channel': entry}


class UpdateChannelReq(BaseModel):
    platform: str | None = None
    config: dict | None = None
    enabled: bool | None = None
    bot_to_bot_enabled: bool | None = None
    trusted_bots: list[dict] | None = None


@router.put('/{channel_id}')
async def update_channel(channel_id: str, req: UpdateChannelReq):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise fastapi.HTTPException(400, 'No fields to update')
    current = await asyncio.to_thread(get_channel_config, channel_id)
    if current is None:
        raise fastapi.HTTPException(404, f'Channel not found: {channel_id}')
    target_platform = updates.get('platform', current['platform'])
    if updates.get('bot_to_bot_enabled') and not supports_bot_to_bot(target_platform):
        raise fastapi.HTTPException(
            400, f'bot_to_bot_enabled is not supported by platform "{target_platform}"')
    if updates.get('trusted_bots') and not uses_trusted_bots(target_platform):
        raise fastapi.HTTPException(
            400, f'trusted_bots is not used by platform "{target_platform}"')
    if not uses_trusted_bots(target_platform):
        updates['trusted_bots'] = []
    if not supports_bot_to_bot(target_platform):
        updates['bot_to_bot_enabled'] = False
    try:
        result = await asyncio.to_thread(update_channel_config, channel_id, **updates)
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))
    # 重启 adapter 以应用新配置
    await manager.restart_adapter(channel_id)
    return {'channel': result}


@router.delete('/{channel_id}')
async def delete_channel(channel_id: str):
    # 先停止 adapter
    if channel_id in manager._adapters:
        await manager._adapters[channel_id].stop()
        del manager._adapters[channel_id]
    if not await asyncio.to_thread(delete_channel_config, channel_id):
        raise fastapi.HTTPException(404, f'Channel not found: {channel_id}')
    return {'deleted': channel_id}


@router.post('/{channel_id}/restart')
async def restart_channel(channel_id: str):
    ch = await asyncio.to_thread(get_channel_config, channel_id)
    if ch is None:
        raise fastapi.HTTPException(404, f'Channel not found: {channel_id}')
    # Restart implies "I want this running" — otherwise restart_adapter is a no-op
    # for a previously stopped (disabled) channel.
    if not ch.get('enabled'):
        await asyncio.to_thread(update_channel_config, channel_id, enabled=True)
    await manager.restart_adapter(channel_id)
    return {'status': 'ok'}


@router.post('/{channel_id}/stop')
async def stop_channel(channel_id: str):
    ch = await asyncio.to_thread(get_channel_config, channel_id)
    if ch is None:
        raise fastapi.HTTPException(404, f'Channel not found: {channel_id}')
    # Persist the intent: the manager watchdog reconnects any *enabled* channel,
    # so without this a stopped channel would come back within 30s.
    await asyncio.to_thread(update_channel_config, channel_id, enabled=False)
    if channel_id in manager._adapters:
        await manager._adapters[channel_id].stop()
        del manager._adapters[channel_id]
    from channel.manager import _update_status
    _update_status(channel_id, 'stopped')
    return {'status': 'stopped'}


# ── User Management ──────────────────────────────────────────────────────────

@router.get('/users')
def list_users(platform: str = None):
    return {'users': acl.list_users(platform)}


class UpsertUserReq(BaseModel):
    platform: str
    user_id: str
    display_name: str = ''
    role: str = 'viewer'
    tool_filter: str = '*'


@router.post('/users')
def upsert_user(req: UpsertUserReq):
    try:
        user = acl.upsert_user(req.platform, req.user_id, req.display_name, req.role, req.tool_filter)
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))
    return {'user': user}


class DeleteUserReq(BaseModel):
    platform: str
    user_id: str


@router.delete('/users')
def delete_user(req: DeleteUserReq):
    if not acl.delete_user(req.platform, req.user_id):
        raise fastapi.HTTPException(404, 'User not found')
    return {'deleted': True}


# ── Channel Settings ─────────────────────────────────────────────────────────

@router.get('/settings')
def get_settings():
    settings = config.main.get('channel_settings', {
        'default_role': 'viewer',
        'auto_approve': True,
        'require_actuator_confirm': True,
    })
    return {'settings': settings}


@router.put('/settings')
def update_settings(settings: dict):
    config.main['channel_settings'] = settings
    return {'settings': settings}
