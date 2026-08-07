"""
channel/adapters/telegram.py — Telegram Bot 适配器。

使用 long-polling (getUpdates) 模式，无需公网 IP。
依赖：python-telegram-bot (>=21.0)
"""

import asyncio

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage, OnMessageCallback


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot API 适配器（long-polling）。"""

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._app = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        bot_token = self.config.get('bot_token', '')
        if not bot_token:
            raise ValueError('Telegram bot_token is required')

        from telegram import Update
        from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

        self._app = ApplicationBuilder().token(bot_token).build()

        async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not update.message or not update.message.text:
                return
            user = update.message.from_user
            msg = InboundMessage(
                platform='telegram',
                channel_id=self.channel_id,
                user_id=str(user.id),
                chat_id=str(update.message.chat_id),
                display_name=user.username or user.first_name or str(user.id),
                text=update.message.text,
            )
            await self._on_message(msg)

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

        # 启动 polling（在后台 task 中运行）
        await self._app.initialize()
        await self._app.start()
        self._task = asyncio.create_task(self._poll_loop())
        self._running = True
        print(f'[telegram] adapter started: {self.channel_id}')

    async def _poll_loop(self):
        """运行 telegram updater polling。"""
        try:
            updater = self._app.updater
            await updater.start_polling(
                poll_interval=1.0,
                timeout=30,
                drop_pending_updates=True,
            )
            # 保持运行直到被 cancel
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f'[telegram] poll error: {e}')
            self._running = False

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._app:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                print(f'[telegram] shutdown error: {e}')
        self._app = None
        print(f'[telegram] adapter stopped: {self.channel_id}')

    async def send_message(self, msg: OutboundMessage) -> None:
        if not self._app or not self._app.bot:
            return
        bot = self._app.bot
        if msg.image_bytes:
            from io import BytesIO
            await bot.send_photo(
                chat_id=int(msg.chat_id),
                photo=BytesIO(msg.image_bytes),
                caption=msg.image_caption or msg.text or '',
            )
        elif msg.text:
            await bot.send_message(
                chat_id=int(msg.chat_id),
                text=msg.text,
            )
