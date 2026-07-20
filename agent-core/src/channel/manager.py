"""
channel/manager.py — Channel 生命周期管理器。

职责：
- 管理 channel_configs（CRUD）
- 启动/停止 adapters
- 入站消息 → ACL 检查 → event_bus
- 出站回复路由（trigger source 为 channel:* 时）
"""

import asyncio
import json
import time

import config
import event_bus
from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage
from channel import acl


# ── Channel Config 持久化 ────────────────────────────────────────────────────

_CONFIG_KEY = 'channel_configs'


def _get_channel_configs() -> list[dict]:
    return config.main.get(_CONFIG_KEY, [])


def _save_channel_configs(configs: list[dict]):
    config.main[_CONFIG_KEY] = configs


def get_channel_config(channel_id: str) -> dict | None:
    for ch in _get_channel_configs():
        if ch['id'] == channel_id:
            return ch
    return None


def add_channel_config(channel_id: str, platform: str, cfg: dict, enabled: bool = False) -> dict:
    configs = _get_channel_configs()
    # 检查 ID 唯一
    if any(c['id'] == channel_id for c in configs):
        raise ValueError(f'Channel ID already exists: {channel_id}')
    entry = {
        'id': channel_id,
        'platform': platform,
        'enabled': enabled,
        'config': cfg,
        'status': 'disconnected',
        'updated_at': time.time(),
    }
    configs.append(entry)
    _save_channel_configs(configs)
    return entry


def update_channel_config(channel_id: str, **updates) -> dict | None:
    configs = _get_channel_configs()
    for ch in configs:
        if ch['id'] == channel_id:
            for k, v in updates.items():
                if k in ('platform', 'config', 'enabled'):
                    ch[k] = v
            ch['updated_at'] = time.time()
            _save_channel_configs(configs)
            return ch
    return None


def delete_channel_config(channel_id: str) -> bool:
    configs = _get_channel_configs()
    new_configs = [c for c in configs if c['id'] != channel_id]
    if len(new_configs) == len(configs):
        return False
    _save_channel_configs(new_configs)
    return True


# ── Adapter Registry ─────────────────────────────────────────────────────────

_ADAPTER_CLASSES: dict[str, type] = {}


def register_adapter(platform: str, cls: type):
    """注册平台适配器类。"""
    _ADAPTER_CLASSES[platform] = cls


# ── Manager ──────────────────────────────────────────────────────────────────

class ChannelManager:
    """管理所有 channel adapter 的生命周期和消息路由。"""

    def __init__(self):
        self._adapters: dict[str, ChannelAdapter] = {}  # channel_id → adapter
        self._active_input_channels: set[str] = set()
        self._active_output_channels: set[str] = set()

    def sync_from_canvas(self):
        """从 canvas layout 读取 channel_msg_input/output 卡片的 instance config，
        确定哪些 channel 处于活跃状态。"""
        layout = config.main.get('canvas_layout', {})
        cards = layout.get('cards', [])

        input_channels = set()
        output_channels = set()

        for card in cards:
            if card.get('mcpId') != 'agentcore':
                continue
            tool_name = card.get('toolName', '')
            card_id = card.get('id', '')
            if tool_name not in ('channel_request', 'channel_reply'):
                continue

            # 读取 instance config 获取 channel_id
            instance_key = f'tool_config:agentcore:{tool_name}:{card_id}'
            instance_cfg = config.main.get(instance_key, None)
            channel_id = None
            if instance_cfg:
                channel_id = instance_cfg.get('channel_id', '')
            if not channel_id:
                continue

            if tool_name == 'channel_request':
                input_channels.add(channel_id)
            else:
                output_channels.add(channel_id)

        self._active_input_channels = input_channels
        self._active_output_channels = output_channels

    @property
    def active_input_channels(self) -> set[str]:
        return self._active_input_channels

    @property
    def active_output_channels(self) -> set[str]:
        return self._active_output_channels

    async def start(self):
        """启动所有 enabled 的 channel adapters。"""
        for ch_cfg in _get_channel_configs():
            if ch_cfg.get('enabled'):
                await self._start_adapter(ch_cfg)
        print(f'[channel] manager started, {len(self._adapters)} adapters running')

    async def stop(self):
        """关闭所有运行中的 adapters。"""
        for adapter in list(self._adapters.values()):
            try:
                await adapter.stop()
            except Exception as e:
                print(f'[channel] stop adapter {adapter.channel_id} error: {e}')
        self._adapters.clear()
        print('[channel] manager stopped')

    async def _start_adapter(self, ch_cfg: dict):
        """为单个 channel 配置启动 adapter。"""
        channel_id = ch_cfg['id']
        platform = ch_cfg['platform']

        cls = _ADAPTER_CLASSES.get(platform)
        if cls is None:
            print(f'[channel] no adapter class for platform: {platform}')
            return

        adapter = cls(
            channel_id=channel_id,
            platform=platform,
            config=ch_cfg.get('config', {}),
            on_message=self._on_inbound_message,
        )
        try:
            await adapter.start()
            self._adapters[channel_id] = adapter
            _update_status(channel_id, 'connected')
        except Exception as e:
            print(f'[channel] failed to start {channel_id}: {e}')
            _update_status(channel_id, f'error:{e}')

    async def restart_adapter(self, channel_id: str):
        """重启指定 adapter（配置更新后调用）。"""
        # Stop existing
        if channel_id in self._adapters:
            await self._adapters[channel_id].stop()
            del self._adapters[channel_id]
        # Start fresh
        ch_cfg = get_channel_config(channel_id)
        if ch_cfg and ch_cfg.get('enabled'):
            await self._start_adapter(ch_cfg)

    # ── Inbound ──────────────────────────────────────────────────────────────

    async def _on_inbound_message(self, msg: InboundMessage):
        """Handle incoming platform message."""
        # Sync active channels from canvas
        self.sync_from_canvas()

        # 1. Check if this channel is activated on canvas (Input connection)
        if msg.channel_id not in self.active_input_channels:
            return  # Not connected to AgentCore, discard

        # 2. ACL — ensure user exists, otherwise auto-register
        user = acl.get_user(msg.platform, msg.user_id)
        if user is None:
            channel_settings = self._get_channel_settings()
            default_role = channel_settings.get('default_role', 'viewer')
            auto_approve = channel_settings.get('auto_approve', True)
            if auto_approve:
                acl.upsert_user(msg.platform, msg.user_id, msg.display_name, role=default_role)
                user = acl.get_user(msg.platform, msg.user_id)
            else:
                adapter = self._adapters.get(msg.channel_id)
                if adapter:
                    await adapter.send_message(OutboundMessage(
                        chat_id=msg.chat_id,
                        text='Pending approval. An admin has been notified.'
                    ))
                return

        # 3. ACL — 检查是否 blocked
        if user['role'] == 'blocked':
            return  # 静默丢弃

        # 4. 注入 event_bus
        source = f"channel:{msg.platform}:{msg.user_id}"
        await event_bus.enqueue(
            source=source,
            text=f"[{msg.platform} @{msg.display_name}] {msg.text}",
            payload={
                'platform': msg.platform,
                'channel_id': msg.channel_id,
                'chat_id': msg.chat_id,
                'user_id': msg.user_id,
                'display_name': msg.display_name,
                'user_role': user['role'],
            }
        )

        # 5. Broadcast to frontend
        from api.motus_stream import push_event
        await push_event({
            'type': 'trigger',
            'mcp_id': source,
            'payload': {
                'text': msg.text,
                'platform': msg.platform,
                'user': msg.display_name,
            }
        })

    # ── Outbound (Reply Routing) ─────────────────────────────────────────────

    async def route_reply(self, trigger_event: dict, reply_text: str):
        """
        在 agent turn 结束后调用：如果 trigger 来自 channel，把回复发回对应平台。

        trigger_event.payload 包含 channel_id, chat_id 等路由信息。
        """
        source = trigger_event.get('source', '')
        if not source.startswith('channel:'):
            return

        payload = trigger_event.get('payload', {})
        channel_id = payload.get('channel_id', '')
        chat_id = payload.get('chat_id', '')

        if not channel_id or not chat_id:
            return

        # 检查 output channel 是否在 canvas 中激活
        if channel_id not in self.active_output_channels:
            return

        adapter = self._adapters.get(channel_id)
        if adapter is None:
            return

        try:
            await adapter.send_message(OutboundMessage(chat_id=chat_id, text=reply_text))
        except Exception as e:
            print(f'[channel] reply failed ({channel_id}→{chat_id}): {e}')

    # ── Status ───────────────────────────────────────────────────────────────

    def get_status(self) -> list[dict]:
        """获取所有 channel 的状态。"""
        result = []
        for ch_cfg in _get_channel_configs():
            channel_id = ch_cfg['id']
            adapter = self._adapters.get(channel_id)
            result.append({
                'id': channel_id,
                'platform': ch_cfg['platform'],
                'enabled': ch_cfg.get('enabled', False),
                'status': adapter.status() if adapter else 'disconnected',
                'active_input': channel_id in self.active_input_channels,
                'active_output': channel_id in self.active_output_channels,
            })
        return result

    def _get_channel_settings(self) -> dict:
        return config.main.get('channel_settings', {
            'default_role': 'viewer',
            'auto_approve': True,
            'require_actuator_confirm': True,
        })


def _update_status(channel_id: str, status: str):
    """更新 channel_configs 中的 status 字段。"""
    configs = _get_channel_configs()
    for ch in configs:
        if ch['id'] == channel_id:
            ch['status'] = status
            ch['updated_at'] = time.time()
            break
    _save_channel_configs(configs)


# ── 全局单例 ─────────────────────────────────────────────────────────────────

manager = ChannelManager()
