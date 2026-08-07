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

    # ── Persistent conversation context ──────────────────────────────────────

    def _get_last_context(self) -> dict:
        """Get all channel contexts from persistent storage."""
        return config.main.get('channel_last_context', {})

    def _set_last_context(self, channel_id: str, chat_id: str, user_id: str):
        """Persist conversation context for a channel."""
        ctx = config.main.get('channel_last_context', {})
        ctx[channel_id] = {'chat_id': chat_id, 'user_id': user_id}
        config.main['channel_last_context'] = ctx

    def sync_from_canvas(self):
        """从 canvas layout 读取 channel_msg_input/output 卡片的 instance config，
        确定哪些 channel 处于活跃状态。"""
        layout = config.main.get('canvas_layout', {})
        cards = layout.get('cards', [])

        input_channels = set()
        output_channels = set()

        for card in cards:
            if card.get('mcpId') not in ('agentcore', 'channel'):
                continue
            tool_name = card.get('toolName', '')
            card_id = card.get('id', '')
            if tool_name not in ('channel_request', 'channel_reply'):
                continue

            # 读取 instance config 获取 channel_id (check both old and new MCP id)
            mcp_id = card.get('mcpId', 'channel')
            instance_key = f'tool_config:{mcp_id}:{tool_name}:{card_id}'
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

    async def _start_adapter(self, ch_cfg: dict, retries: int = 3, delay: float = 5.0):
        """为单个 channel 配置启动 adapter，带重试。"""
        channel_id = ch_cfg['id']
        platform = ch_cfg['platform']

        cls = _ADAPTER_CLASSES.get(platform)
        if cls is None:
            msg = f'[channel] No adapter for platform: {platform}. Supported: telegram, slack, feishu'
            print(msg)
            await self._push_error(msg)
            return

        adapter = cls(
            channel_id=channel_id,
            platform=platform,
            config=ch_cfg.get('config', {}),
            on_message=self._on_inbound_message,
        )

        for attempt in range(retries):
            try:
                await adapter.start()
                self._adapters[channel_id] = adapter
                _update_status(channel_id, 'connected')
                return
            except Exception as e:
                if attempt < retries - 1:
                    print(f'[channel] {channel_id} start failed (attempt {attempt + 1}/{retries}), '
                          f'retrying in {delay}s: {e}')
                    await asyncio.sleep(delay)
                else:
                    error_msg = f'[channel] Failed to start {channel_id} ({platform}) after {retries} attempts: {e}'
                    print(error_msg)
                    await self._push_error(error_msg)
                    _update_status(channel_id, 'error')

    async def _push_error(self, message: str):
        """Push error to frontend activity stream."""
        try:
            from api.motus_stream import push_event
            await push_event({
                'type': 'error',
                'payload': {'message': message},
            })
        except Exception:
            pass

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

        # 4. Store last conversation context for reply routing (persisted)
        self._set_last_context(msg.channel_id, msg.chat_id, msg.user_id)

        # 5. Publish to topic for dashboard and canvas data flow
        from api.inspection import publish_to_topic
        topic_id = msg.channel_id.replace(' ', '_')
        topic = f'/channel/request/{topic_id}'
        topic_data = json.dumps({
            'platform': msg.platform,
            'user': msg.display_name,
            'user_id': msg.user_id,
            'chat_id': msg.chat_id,
            'text': msg.text,
            'user_role': user['role'],
        }, ensure_ascii=False)
        await publish_to_topic(topic, topic_data)

        # 5. Broadcast to frontend activity stream
        from api.motus_stream import push_event
        source = f"channel:{msg.platform}:{msg.user_id}"
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
        Send reply back to the channel that triggered this event.
        Simple passthrough — no canvas connection check needed.
        """
        source = trigger_event.get('source', '')
        if not source.startswith('channel:'):
            return

        payload = trigger_event.get('payload', {})
        channel_id = payload.get('channel_id', '')
        chat_id = payload.get('chat_id', '')

        if not channel_id or not chat_id:
            return

        adapter = self._adapters.get(channel_id)
        if adapter is None:
            return

        try:
            await adapter.send_message(OutboundMessage(chat_id=chat_id, text=reply_text))
        except Exception as e:
            print(f'[channel] reply failed ({channel_id}→{chat_id}): {e}')

    async def send_to_channel(self, channel_id: str, text: str) -> str:
        """Send a message to a channel using the last known chat context.
        Called by channel_reply tool dispatch."""
        ctx = self._get_last_context().get(channel_id)
        if not ctx:
            return (
                f'Error: No conversation context for channel "{channel_id}".\n'
                f'Cause: The bot has not received any message from a user yet in this channel.\n'
                f'Solution: Ask a user to send a message to the bot in Feishu (private chat or @bot in group), '
                f'then the bot can reply to that conversation.'
            )

        adapter = self._adapters.get(channel_id)
        if not adapter:
            return (
                f'Error: Channel "{channel_id}" is not running.\n'
                f'Cause: The channel adapter failed to start, was stopped, or the connection dropped.\n'
                f'Solution: Go to Settings → Channels and click Restart for this channel.'
            )

        chat_id = ctx['chat_id']
        try:
            await adapter.send_message(OutboundMessage(chat_id=chat_id, text=text))
            return f'Reply sent ({len(text)} chars)'
        except Exception as e:
            error_msg = str(e)
            if '99991672' in error_msg or 'Permission denied' in error_msg or 'Access denied' in error_msg:
                result = (
                    f'Error: Permission denied when sending message.\n'
                    f'Cause: The bot lacks "im:message:send_as_bot" permission.\n'
                    f'Solution: Grant the permission in Feishu Developer Console, '
                    f'then publish a new app version.'
                )
            elif 'not initialized' in error_msg or 'not running' in error_msg:
                result = (
                    f'Error: Channel adapter is not connected.\n'
                    f'Cause: The WebSocket connection may have dropped silently.\n'
                    f'Solution: Go to Settings → Channels and click Restart.'
                )
            else:
                result = f'Error sending reply: {error_msg}'
            await self._push_error(result)
            return result

    async def send_to_channel_any(self, text: str) -> str:
        """Send to the most recent channel with conversation context."""
        all_ctx = self._get_last_context()
        if not all_ctx:
            return 'No active conversation. A user must send a message first.'
        # Use the most recently updated channel
        channel_id = list(all_ctx.keys())[-1]
        return await self.send_to_channel(channel_id, text)

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
