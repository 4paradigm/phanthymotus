"""
channel/adapter.py — Channel Adapter ABC。

每个消息平台（Telegram、Slack 等）实现此接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Awaitable


@dataclass
class InboundMessage:
    """适配器解析后的统一入站消息格式。"""
    platform: str           # 'telegram', 'slack', ...
    channel_id: str         # 配置 ID（channel_configs.id）
    user_id: str            # 平台用户 ID
    chat_id: str            # 会话/频道 ID（用于回复路由）
    display_name: str       # 用户显示名
    text: str               # 消息文本
    attachments: list = field(default_factory=list)  # 附件列表（future）


@dataclass
class OutboundMessage:
    """发送给平台的统一出站消息格式。"""
    chat_id: str            # 目标会话 ID
    text: str = ''          # 文本内容
    image_bytes: bytes | None = None  # 图片（可选）
    image_caption: str = ''


# 收到消息时的回调签名
OnMessageCallback = Callable[[InboundMessage], Awaitable[None]]


class ChannelAdapter(ABC):
    """消息平台适配器基类。"""

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        self.channel_id = channel_id
        self.platform = platform
        self.config = config
        self._on_message = on_message
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    async def start(self) -> None:
        """启动适配器（开始接收消息）。"""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """优雅关闭。"""
        ...

    @abstractmethod
    async def send_message(self, msg: OutboundMessage) -> None:
        """发送消息到平台。"""
        ...

    def status(self) -> str:
        """返回当前状态。"""
        return 'connected' if self._running else 'disconnected'
