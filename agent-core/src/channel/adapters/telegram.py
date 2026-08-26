"""
channel/adapters/telegram.py — Telegram Bot 适配器。

使用 long-polling (getUpdates) 模式，无需公网 IP。
依赖：python-telegram-bot (>=21.0)

支持收发图片 / 视频 / 语音 / 文件：入站附件下载后落盘到持久化目录
（channel/store.py），出站按 Attachment.kind 选择 send_photo / send_video /
send_document。
"""

import asyncio

from channel.adapter import (
    Attachment, ChannelAdapter, InboundMessage, OnMessageCallback, OutboundMessage,
    PartialSendError, KIND_AUDIO, KIND_FILE, KIND_IMAGE, KIND_VIDEO,
)

_TEXT_CHUNK = 4000  # Telegram 单条上限 4096


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot API 适配器（long-polling）。"""

    SUPPORTED_FILE_KINDS = (KIND_IMAGE, KIND_VIDEO, KIND_AUDIO, KIND_FILE)

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
            if not update.message:
                return
            m = update.message
            user = m.from_user
            text, attachments = await self._extract(m)
            if not text and not attachments:
                return
            msg = InboundMessage(
                platform='telegram',
                channel_id=self.channel_id,
                user_id=str(user.id),
                chat_id=str(m.chat_id),
                display_name=user.username or user.first_name or str(user.id),
                text=text,
                message_id=str(m.message_id),
                attachments=attachments,
            )
            await self._on_message(msg)

        media_filter = (filters.PHOTO | filters.VIDEO | filters.Document.ALL
                        | filters.VOICE | filters.AUDIO)
        self._app.add_handler(MessageHandler(
            (filters.TEXT & ~filters.COMMAND) | media_filter, _handle_message))

        # 启动 polling（在后台 task 中运行）
        await self._app.initialize()
        await self._app.start()
        # 凭据校验：getMe 失败说明 token 无效，此时必须抛出而不是假装启动成功
        await self._app.bot.get_me()
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

    # ── 健康状态 ─────────────────────────────────────────────────────────────

    async def health_check(self) -> tuple[bool, str]:
        if not self._running or not self._app or not self._app.bot:
            return False, 'adapter not running'
        if self._app.updater and not self._app.updater.running:
            return False, 'polling stopped — restart the channel'
        try:
            await self._app.bot.get_me()
            return True, ''
        except Exception as e:
            return False, f'getMe failed: {e}'

    # ── 发送 ─────────────────────────────────────────────────────────────────

    async def send_message(self, msg: OutboundMessage) -> None:
        if not self._app or not self._app.bot:
            raise RuntimeError(
                '[telegram] Cannot send: adapter not initialized. '
                'Solution: check bot_token in Channel settings and restart the channel.'
            )
        bot = self._app.bot
        chat_id = int(msg.chat_id)

        files = list(msg.files)
        if msg.image_bytes:
            from channel import store
            files.append(store.save_bytes(self.channel_id, msg.image_bytes,
                                          kind=KIND_IMAGE, name='image.jpg',
                                          mime='image/jpeg', fallback_ext='.jpg'))
            if msg.image_caption and not msg.text:
                files[-1].caption = msg.image_caption

        if msg.text:
            for chunk in _chunks(msg.text, _TEXT_CHUNK):
                await bot.send_message(chat_id=chat_id, text=chunk)

        if not files:
            return

        # 逐个附件汇报成败：附件失败不该让上层以为已送达的文本也失败了
        sent = ['文本'] if msg.text else []
        failures = []
        for att in files:
            try:
                with open(att.path, 'rb') as f:
                    payload = f.read()
                if att.kind == KIND_IMAGE:
                    await bot.send_photo(chat_id=chat_id, photo=payload, caption=att.caption or '')
                elif att.kind == KIND_VIDEO:
                    await bot.send_video(chat_id=chat_id, video=payload, caption=att.caption or '')
                elif att.kind == KIND_AUDIO:
                    await bot.send_audio(chat_id=chat_id, audio=payload, caption=att.caption or '')
                else:
                    await bot.send_document(chat_id=chat_id, document=payload,
                                            filename=att.name or None, caption=att.caption or '')
                sent.append(att.name or att.path)
            except Exception as e:
                print(f'[telegram] send attachment failed ({att.name or att.path}): {e}')
                failures.append(f'- {att.name or att.path}: {e}')
        if failures:
            raise PartialSendError(sent, failures)

    # ── 接收 ─────────────────────────────────────────────────────────────────

    async def _extract(self, m) -> tuple[str, list[Attachment]]:
        """从 telegram Message 提取文本与附件（附件下载后落盘）。"""
        attachments: list[Attachment] = []
        text = m.text or m.caption or ''

        if m.photo:
            # photo 是同一张图的多个尺寸，取最大的
            biggest = max(m.photo, key=lambda p: p.file_size or 0)
            att = await self._download(biggest.file_id, KIND_IMAGE,
                                       name='photo.jpg', fallback_ext='.jpg')
            if att:
                attachments.append(att)
            text = text or '[图片]'
        elif m.video:
            att = await self._download(m.video.file_id, KIND_VIDEO,
                                       name=m.video.file_name or 'video.mp4',
                                       mime=m.video.mime_type or '', fallback_ext='.mp4')
            if att:
                attachments.append(att)
            text = text or '[视频]'
        elif m.voice:
            att = await self._download(m.voice.file_id, KIND_AUDIO,
                                       name='voice.ogg',
                                       mime=m.voice.mime_type or '', fallback_ext='.ogg')
            if att:
                attachments.append(att)
            text = text or '[语音]'
        elif m.audio:
            att = await self._download(m.audio.file_id, KIND_AUDIO,
                                       name=m.audio.file_name or 'audio.mp3',
                                       mime=m.audio.mime_type or '', fallback_ext='.mp3')
            if att:
                attachments.append(att)
            text = text or '[音频]'
        elif m.document:
            d = m.document
            kind = KIND_IMAGE if (d.mime_type or '').startswith('image/') else KIND_FILE
            att = await self._download(d.file_id, kind, name=d.file_name or 'file',
                                       mime=d.mime_type or '')
            if att:
                attachments.append(att)
            text = text or f'[文件] {d.file_name or ""}'.strip()

        return text, attachments

    async def _download(self, file_id: str, kind: str, *, name: str = '',
                        mime: str = '', fallback_ext: str = '') -> Attachment | None:
        try:
            tg_file = await self._app.bot.get_file(file_id)
            payload = await tg_file.download_as_bytearray()
        except Exception as e:
            print(f'[telegram] download failed ({file_id}): {e}')
            return None
        from channel import store
        return store.save_bytes(self.channel_id, bytes(payload), kind=kind,
                                name=name, mime=mime, fallback_ext=fallback_ext)


def _chunks(text: str, size: int):
    while text:
        if len(text) <= size:
            yield text
            return
        cut = text.rfind('\n', 0, size)
        if cut < size // 2:
            cut = size
        yield text[:cut]
        text = text[cut:].lstrip('\n')
