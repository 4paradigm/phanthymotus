"""
channel — 消息平台集成模块。

通过 Channel Adapter 接收外部消息（Telegram/Slack 等），
经 ACL 鉴权后注入 event_bus，Agent 回复自动路由回对应渠道。
"""

from channel.manager import register_adapter


def _register_builtin_adapters():
    """注册内置平台适配器（延迟导入避免缺少依赖时启动失败）。"""
    try:
        from channel.adapters.telegram import TelegramAdapter
        register_adapter('telegram', TelegramAdapter)
    except ImportError:
        print('[channel] telegram adapter unavailable (missing python-telegram-bot)')

    try:
        from channel.adapters.slack import SlackAdapter
        register_adapter('slack', SlackAdapter)
    except ImportError:
        print('[channel] slack adapter unavailable (missing slack-sdk)')

    try:
        from channel.adapters.feishu import FeishuAdapter
        register_adapter('feishu', FeishuAdapter)
    except ImportError as e:
        print(f'[channel] feishu adapter unavailable ({e})')

    try:
        from channel.adapters.lan import LanAdapter
        register_adapter('lan', LanAdapter)
    except ImportError as e:
        print(f'[channel] lan adapter unavailable ({e})')


_register_builtin_adapters()
