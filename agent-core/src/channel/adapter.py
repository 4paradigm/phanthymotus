"""
channel/adapter.py — Channel Adapter ABC。

每个消息平台（Telegram、Slack、飞书 等）实现此接口。

## 附件（Attachment）

入站与出站共用 `Attachment` 描述媒体：入站由平台下载后落盘（见 channel/store.py），
`path` 指向容器内持久化目录；出站由 LLM 给出容器内路径（见 channel/media.py 做白名单校验），
adapter 负责上传到平台。

## 健康状态

`status()` 不能只看「start() 有没有抛异常」——WS 长连接可能静默断开、凭据可能被吊销。
每个 adapter 应重写 `health_check()` 返回真实可观测的连通性；`ChannelManager` 的 watchdog
和「启动控制」时的卡片自检都依赖它。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Awaitable


# 附件类别 → 各平台据此选择上传/发送接口
KIND_IMAGE = 'image'
KIND_VIDEO = 'video'
KIND_AUDIO = 'audio'
KIND_FILE  = 'file'


@dataclass
class Attachment:
    """一个媒体文件（入站已落盘 / 出站待上传）。"""
    kind: str = KIND_FILE       # image | video | audio | file
    path: str = ''              # 容器内绝对路径
    name: str = ''              # 展示用文件名
    mime: str = ''              # MIME 类型（可空）
    size: int = 0               # 字节数
    caption: str = ''           # 出站时的附带说明（部分平台支持）

    def to_dict(self) -> dict:
        return {
            'kind': self.kind,
            'path': self.path,
            'name': self.name,
            'mime': self.mime,
            'size': self.size,
        }


@dataclass
class InboundMessage:
    """适配器解析后的统一入站消息格式。"""
    platform: str           # 'telegram', 'slack', 'feishu', ...
    channel_id: str         # 配置 ID（channel_configs.id）
    user_id: str            # 平台用户 ID
    chat_id: str            # 会话/频道 ID（用于回复路由）
    display_name: str       # 用户显示名
    text: str               # 消息文本（无文本时为附件占位描述，如 "[图片]"）
    message_id: str = ''    # 平台消息 ID（用于去重）
    attachments: list[Attachment] = field(default_factory=list)
    sender_type: str = 'user'  # user | bot（部分旧事件使用 app）
    chat_type: str = ''        # p2p | group
    mentions: list[dict] = field(default_factory=list)
    expect_reply: bool | None = None  # 机器人协作消息是否明确要求继续 @


@dataclass
class OutboundMessage:
    """发送给平台的统一出站消息格式。"""
    chat_id: str            # 目标会话 ID
    text: str = ''          # 文本内容
    files: list[Attachment] = field(default_factory=list)  # 附件（图片/视频/文件）
    mention_open_id: str = ''  # Feishu 群聊中的目标机器人 open_id
    expect_reply: bool = False  # @ 消息是否要求目标机器人继续回复
    # 兼容旧调用方：直接给字节的图片
    image_bytes: bytes | None = None
    image_caption: str = ''


# 收到消息时的回调签名
OnMessageCallback = Callable[[InboundMessage], Awaitable[None]]


class PartialSendError(RuntimeError):
    """一条出站消息里，部分内容发出去了、部分失败了。

    文本与每个附件在平台侧是**独立的**几次调用（飞书要先上传再发一条 image/file 消息）。
    如果附件失败就整体抛异常，上层只看到「失败」，LLM 会把已经送达的文本重发一遍
    —— 用户那边收到两条一样的话。带上 sent 让上层能如实回报「文本已达、文件没成功」。
    """

    def __init__(self, sent: list[str], failures: list[str]):
        self.sent = sent
        self.failures = failures
        parts = []
        if sent:
            parts.append('已发送：' + '、'.join(sent) + '（不要重发这部分）')
        if failures:
            parts.append('失败：\n' + '\n'.join(failures))
        super().__init__('\n'.join(parts))


class ChannelAdapter(ABC):
    """消息平台适配器基类。"""

    # 该平台支持发送的附件类别；send_message 遇到不支持的类别应抛异常而非静默丢弃
    SUPPORTED_FILE_KINDS: tuple[str, ...] = ()

    # 该平台能不能做 Bot↔Bot（A2A）协作。
    # 由 adapter 自己声明，而不是在 api/channel.py 里写死平台名 —— 否则每加一个
    # 支持 A2A 的平台都要回去改校验分支，而漏改的表现是「配置存下来了但静默失效」。
    SUPPORTS_BOT_TO_BOT: bool = False

    # 对端 Bot 的信任是否来自 channel 配置里的 `trusted_bots`（open_id + chat_id）。
    # 飞书是；局域网 peer 不是 —— 它的信任来自 `peers` 表里配对时钉住的公钥，
    # 所以 lan channel 不该出现 trusted_bots 字段。
    USES_TRUSTED_BOTS: bool = False

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
        """启动适配器（开始接收消息）。

        实现约定：凭据无效 / 无法连接时必须抛异常，不要「起个线程就返回」——
        上层据此决定是否重试、是否把状态标成 error。
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """优雅关闭。"""
        ...

    @abstractmethod
    async def send_message(self, msg: OutboundMessage) -> None:
        """发送消息到平台。失败必须抛异常（上层据此回报给 LLM/用户）。"""
        ...

    async def health_check(self) -> tuple[bool, str]:
        """返回 (是否连通, 失败原因)。默认只看 _running，子类应重写为真实探测。"""
        return (self._running, '' if self._running else 'adapter not running')

    def status(self) -> str:
        """返回当前状态：connected / degraded / disconnected。

        默认实现无法区分 degraded；重写 health_check() 的子类应同时重写本方法。
        """
        return 'connected' if self._running else 'disconnected'
