"""
channel/manager.py — Channel 生命周期管理器。

职责：
- 管理 channel_configs（CRUD）
- 启动/停止 adapters，并用 watchdog 周期探测健康度、自动重连
- 入站消息 → ACL 检查 → topic 发布（附件已落盘，随事件带本地路径）
- 出站回复路由：卡片实例配置的 channel → 最近会话上下文
"""

import asyncio
import json
import time

import config
from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage, PartialSendError
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
        self._watchdog_task: asyncio.Task | None = None
        self._health: dict[str, tuple[bool, str]] = {}   # channel_id → (ok, reason)
        self._retry_at: dict[str, float] = {}            # channel_id → 下次重连时间
        self._retry_backoff: dict[str, float] = {}       # channel_id → 当前退避秒数
        self._inactive_logged: set[str] = set()

    # ── Persistent conversation context ──────────────────────────────────────

    def _get_last_context(self) -> dict:
        """Get all channel contexts from persistent storage."""
        return config.main.get('channel_last_context', {})

    def _set_last_context(self, channel_id: str, chat_id: str, user_id: str):
        """Persist conversation context for a channel.

        带 ts —— 「最近一次会话」必须靠时间戳判断：dict 就地更新不改插入顺序，
        用 list(ctx)[-1] 取到的是最早写入的那个 channel。
        """
        ctx = config.main.get('channel_last_context', {})
        ctx[channel_id] = {'chat_id': chat_id, 'user_id': user_id, 'ts': time.time()}
        config.main['channel_last_context'] = ctx

    def _latest_context_channel(self) -> str:
        """返回最近有过会话的 channel_id（按 ts 最大），无则空串。"""
        ctx = self._get_last_context()
        if not ctx:
            return ''
        return max(ctx.items(), key=lambda kv: (kv[1] or {}).get('ts', 0))[0]

    def resolve_target_channel(self, instance_id: str = '') -> tuple[str, str]:
        """解析回复目标 channel，返回 (channel_id, error)。

        优先级：
        1. 输出卡片实例配置里选的 channel（画布上的连接才真正生效）
        2. 最近有过会话的 channel（按 ts）
        """
        if instance_id:
            cfg = config.main.get(f'tool_config:channel:channel_reply:{instance_id}', None)
            channel_id = (cfg or {}).get('channel_id', '')
            if channel_id:
                if get_channel_config(channel_id) is None:
                    return '', (
                        f'Error: Channel "{channel_id}" configured on this card no longer exists.\n'
                        f'Solution: pick an existing channel in the card config, '
                        f'or re-add it in Settings → Channels.'
                    )
                return channel_id, ''

        channel_id = self._latest_context_channel()
        if channel_id:
            return channel_id, ''
        return '', (
            'Error: No target channel.\n'
            'Cause: the output card has no channel selected and no user has messaged the bot yet.\n'
            'Solution: select a channel in the card config (Settings → Channels to add one), '
            'or ask a user to message the bot first.'
        )

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
        """启动所有 enabled 的 channel adapters，并拉起 watchdog。"""
        for ch_cfg in _get_channel_configs():
            if ch_cfg.get('enabled'):
                await self._start_adapter(ch_cfg)
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog())
        # 清理历史遗留的入站媒体文件
        try:
            from channel import store
            store.prune()
        except Exception as e:
            print(f'[channel] media prune skipped: {e}')
        print(f'[channel] manager started, {len(self._adapters)} adapters running')

    async def stop(self):
        """关闭所有运行中的 adapters。"""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        for adapter in list(self._adapters.values()):
            try:
                await adapter.stop()
            except Exception as e:
                print(f'[channel] stop adapter {adapter.channel_id} error: {e}')
        self._adapters.clear()
        print('[channel] manager stopped')

    # ── Watchdog ─────────────────────────────────────────────────────────────

    async def _watchdog(self, interval: float = 30.0):
        """周期探测每个 enabled channel 的真实连通性，失败则退避重连。

        WS 长连接会静默断开，SDK 的 auto_reconnect 也可能永久失败——没有这一层，
        状态会一直停在 connected，而消息早就收不到了。
        """
        while True:
            try:
                await asyncio.sleep(interval)
                for ch_cfg in _get_channel_configs():
                    if not ch_cfg.get('enabled'):
                        continue
                    await self._check_one(ch_cfg['id'])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f'[channel] watchdog error: {e}')

    async def _check_one(self, channel_id: str):
        adapter = self._adapters.get(channel_id)
        if adapter is None:
            ok, reason = False, 'adapter not started'
        else:
            try:
                ok, reason = await adapter.health_check()
            except Exception as e:
                ok, reason = False, f'health check raised: {e}'

        prev_ok = self._health.get(channel_id, (None, ''))[0]
        self._health[channel_id] = (ok, reason)

        if ok:
            if prev_ok is False:
                print(f'[channel] {channel_id} recovered')
            self._retry_backoff.pop(channel_id, None)
            self._retry_at.pop(channel_id, None)
            _update_status(channel_id, 'connected')
            return

        if prev_ok is not False:
            msg = f'[channel] {channel_id} unhealthy: {reason}'
            print(msg)
            await self._push_error(msg)
        _update_status(channel_id, 'error')

        # 指数退避重连（30s → 60s → … → 300s）
        now = time.time()
        if now < self._retry_at.get(channel_id, 0):
            return
        backoff = min(300.0, max(30.0, self._retry_backoff.get(channel_id, 0) * 2 or 30.0))
        self._retry_backoff[channel_id] = backoff
        self._retry_at[channel_id] = now + backoff
        print(f'[channel] {channel_id} reconnecting (next retry in {backoff:.0f}s if it fails)')
        try:
            # retries=1：watchdog 自己带退避，不要在这里堆内层重试拖住其他 channel 的探测
            await self.restart_adapter(channel_id, retries=1)
        except Exception as e:
            print(f'[channel] {channel_id} reconnect failed: {e}')

    async def ensure_connected(self, channel_id: str, timeout: float = 8.0) -> tuple[bool, str]:
        """确保某 channel 的 adapter 已连接，必要时先拉起。供启动自检使用。

        返回 (ok, 原因)。原因区分「未配置 / 已删除 / 未启用 / 凭据或网络失败」，
        这样启动控制的报错能直接指向要改的地方。
        """
        if not channel_id:
            return False, ('no channel selected on this card — pick one in the card config '
                           '(add channels in Settings → Channels)')

        ch_cfg = get_channel_config(channel_id)
        if ch_cfg is None:
            return False, (f'channel "{channel_id}" no longer exists — '
                           f're-add it in Settings → Channels or fix the card config')
        if not ch_cfg.get('enabled'):
            # 画布上明确要用这个 channel，就是「我要它跑起来」——顺手启用，
            # 而不是让启动控制卡在一个用户早就忘了自己 Stop 过的 channel 上。
            print(f'[channel] {channel_id} was disabled; enabling it because a canvas card requires it')
            update_channel_config(channel_id, enabled=True)

        if channel_id not in self._adapters:
            # 允许自愈：Stop 过或上次启动失败的 adapter 在这里被拉起
            self._retry_at.pop(channel_id, None)
            self._retry_backoff.pop(channel_id, None)
            try:
                await self.restart_adapter(channel_id, retries=1)
            except Exception as e:
                return False, f'channel "{channel_id}" failed to start: {e}'

        deadline = time.time() + timeout
        last_reason = 'adapter not connected'
        while True:
            adapter = self._adapters.get(channel_id)
            if adapter is not None:
                try:
                    ok, reason = await adapter.health_check()
                except Exception as e:
                    ok, reason = False, f'health check raised: {e}'
                self._health[channel_id] = (ok, reason)
                if ok:
                    return True, ''
                last_reason = reason or last_reason
            if time.time() >= deadline:
                return False, f'channel "{channel_id}" not connected: {last_reason}'
            await asyncio.sleep(0.5)

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
                self._health[channel_id] = (True, '')
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
                    self._health[channel_id] = (False, str(e))
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

    async def restart_adapter(self, channel_id: str, retries: int = 3):
        """重启指定 adapter（配置更新后调用）。"""
        # Stop existing
        if channel_id in self._adapters:
            await self._adapters[channel_id].stop()
            del self._adapters[channel_id]
        # Start fresh
        ch_cfg = get_channel_config(channel_id)
        if ch_cfg and ch_cfg.get('enabled'):
            await self._start_adapter(ch_cfg, retries=retries)

    # ── Inbound ──────────────────────────────────────────────────────────────

    async def _on_inbound_message(self, msg: InboundMessage):
        """Handle incoming platform message."""
        # Sync active channels from canvas
        self.sync_from_canvas()

        # 1. Check if this channel is activated on canvas (Input connection)
        if msg.channel_id not in self.active_input_channels:
            # 静默丢弃会让「消息发了但机器人没反应」完全无法排查，至少记一次
            if msg.channel_id not in self._inactive_logged:
                self._inactive_logged.add(msg.channel_id)
                print(f'[channel] {msg.channel_id} message discarded: no channel_request card '
                      f'on the canvas is configured for this channel')
            return
        self._inactive_logged.discard(msg.channel_id)

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
        #    附件已由 adapter 落盘到持久化目录，这里只带容器内本地路径给 LLM
        from api.inspection import publish_to_topic
        files = [a.to_dict() for a in (msg.attachments or [])]
        topic_id = msg.channel_id.replace(' ', '_')
        topic = f'/channel/request/{topic_id}'
        payload = {
            'platform': msg.platform,
            'channel_id': msg.channel_id,
            'message_id': msg.message_id,
            'user': msg.display_name,
            'user_id': msg.user_id,
            'chat_id': msg.chat_id,
            'text': msg.text,
            'user_role': user['role'],
        }
        if files:
            payload['files'] = files
        await publish_to_topic(topic, json.dumps(payload, ensure_ascii=False))

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
                'files': files,
            }
        })

    # ── Outbound (Reply Routing) ─────────────────────────────────────────────

    async def send_to_channel(self, channel_id: str, text: str = '',
                              files: list | None = None) -> str:
        """Send text and/or attachments to a channel using its last chat context.
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

        # 出站附件：路径白名单 + 大小/类别校验（失败的逐条回报，不当作发送成功）
        attachments = []
        file_errors: list[str] = []
        if files:
            from channel import media
            attachments, file_errors = media.resolve_outbound(files, adapter.platform)
            unsupported = [a for a in attachments
                           if a.kind not in adapter.SUPPORTED_FILE_KINDS]
            for a in unsupported:
                file_errors.append(f'{a.path}: {adapter.platform} adapter cannot send {a.kind}')
            attachments = [a for a in attachments if a.kind in adapter.SUPPORTED_FILE_KINDS]
            if not attachments and not text:
                return 'Error: nothing sent.\n' + '\n'.join(file_errors)

        chat_id = ctx['chat_id']
        try:
            await adapter.send_message(OutboundMessage(chat_id=chat_id, text=text,
                                                       files=attachments))
            parts = []
            if text:
                parts.append(f'{len(text)} chars')
            if attachments:
                parts.append(f'{len(attachments)} file(s): ' +
                             ', '.join(a.name for a in attachments))
            result = f'Reply sent to {channel_id} ({"; ".join(parts) or "empty"})'
            if file_errors:
                result += '\nSome files were NOT sent:\n' + '\n'.join(file_errors)
            return result
        except PartialSendError as e:
            # 部分成功：必须让 LLM 知道哪部分已经到了，否则它会把文本重发一遍
            print(f'[channel] {channel_id} partial send: {e}')
            result = f'部分发送成功（目标 {channel_id}）：\n{e}'
            if file_errors:
                result += '\n其它未通过校验的文件：\n' + '\n'.join(file_errors)
            await self._push_error(f'[channel] {channel_id} partial send: {e}')
            return result
        except Exception as e:
            error_msg = str(e)
            # 平台自己的报错通常已经写明缺哪个权限、给了授权链接 —— 原文透传，
            # 不要用猜的原因盖掉它（曾经把「缺上传权限」说成「缺 im:message:send_as_bot」，
            # LLM 就照着这句错话跟用户解释）。
            if 'not initialized' in error_msg or 'not running' in error_msg:
                result = (
                    f'Error: Channel adapter is not connected.\n'
                    f'Cause: The WebSocket connection may have dropped silently.\n'
                    f'Solution: Go to Settings → Channels and click Restart.'
                )
            else:
                result = f'Error sending reply to {channel_id}: {error_msg}'
            if file_errors:
                result += '\nAlso rejected before sending:\n' + '\n'.join(file_errors)
            print(f'[channel] send failed ({channel_id}→{chat_id}): {error_msg}')
            await self._push_error(result)
            return result

    async def send_reply(self, instance_id: str = '', text: str = '',
                         files: list | None = None) -> str:
        """channel_reply 的统一出口：先按卡片实例配置解析目标 channel，再发送。

        画布上给输出卡片选的 channel 必须真正决定回复去向——之前两条 dispatch
        路径都用「最后一个有上下文的 channel」猜目标，卡片配置形同装饰。
        """
        channel_id, err = self.resolve_target_channel(instance_id)
        if err:
            return err
        return await self.send_to_channel(channel_id, text=text, files=files)

    # ── Status ───────────────────────────────────────────────────────────────

    def get_status(self) -> list[dict]:
        """获取所有 channel 的状态。"""
        result = []
        for ch_cfg in _get_channel_configs():
            channel_id = ch_cfg['id']
            adapter = self._adapters.get(channel_id)
            health = self._health.get(channel_id)
            result.append({
                'id': channel_id,
                'platform': ch_cfg['platform'],
                'enabled': ch_cfg.get('enabled', False),
                'status': adapter.status() if adapter else 'disconnected',
                'health_error': '' if not health or health[0] else health[1],
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
    """更新 channel_configs 中的 status 字段（状态未变则不落库——watchdog 每 30s 会调）。"""
    configs = _get_channel_configs()
    for ch in configs:
        if ch['id'] == channel_id:
            if ch.get('status') == status:
                return
            ch['status'] = status
            ch['updated_at'] = time.time()
            break
    _save_channel_configs(configs)


# ── 全局单例 ─────────────────────────────────────────────────────────────────

manager = ChannelManager()
