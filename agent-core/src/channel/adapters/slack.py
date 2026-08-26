"""
channel/adapters/slack.py — Slack Socket Mode 适配器。

使用 Socket Mode（WebSocket），无需公网 IP。
依赖：slack-sdk (>=3.27)

支持收发图片 / 视频 / 文件：出站统一走 files_upload_v2；入站从 event['files'] 拿
url_private_download，用 bot token 以 Bearer 认证下载后落盘（channel/store.py）。
需要权限：files:read（下载）、files:write（上传）、chat:write。
"""

import asyncio
import mimetypes
import os

from channel.adapter import (
    Attachment, ChannelAdapter, InboundMessage, OnMessageCallback, OutboundMessage,
    PartialSendError, KIND_AUDIO, KIND_FILE, KIND_IMAGE, KIND_VIDEO,
)


class SlackAdapter(ChannelAdapter):
    """Slack Socket Mode 适配器。"""

    SUPPORTED_FILE_KINDS = (KIND_IMAGE, KIND_VIDEO, KIND_AUDIO, KIND_FILE)

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._handler = None
        self._client = None
        self._socket_client = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        bot_token = self.config.get('bot_token', '')
        app_token = self.config.get('app_token', '')
        if not bot_token or not app_token:
            raise ValueError('Slack bot_token and app_token are required')

        from slack_sdk.web.async_client import AsyncWebClient
        from slack_sdk.socket_mode.aiohttp import SocketModeClient
        from slack_sdk.socket_mode.request import SocketModeRequest
        from slack_sdk.socket_mode.response import SocketModeResponse

        self._client = AsyncWebClient(token=bot_token)
        # 凭据前置校验：token 无效时必须抛出，让上层状态变成 error
        await self._client.auth_test()
        self._socket_client = SocketModeClient(app_token=app_token, web_client=self._client)

        async def _handle_event(client, req: SocketModeRequest):
            # Acknowledge
            await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

            if req.type == 'events_api' and req.payload:
                event = req.payload.get('event', {})
                if event.get('type') != 'message':
                    return
                # 只跳过非 file_share 的 subtype（编辑、加入频道等）
                subtype = event.get('subtype')
                if subtype and subtype != 'file_share':
                    return
                if event.get('bot_id'):
                    return
                text = event.get('text', '')
                user_id = event.get('user', '')
                chat = event.get('channel', '')
                if not user_id:
                    return

                attachments = await self._download_files(event.get('files', []) or [])
                if not text and not attachments:
                    return
                if not text and attachments:
                    text = f'[{attachments[0].kind}] {attachments[0].name}'

                msg = InboundMessage(
                    platform='slack',
                    channel_id=self.channel_id,
                    user_id=user_id,
                    chat_id=chat,
                    display_name=user_id,  # Could resolve via users.info API
                    text=text,
                    message_id=event.get('client_msg_id', '') or event.get('ts', ''),
                    attachments=attachments,
                )
                await self._on_message(msg)

        self._socket_client.socket_mode_request_listeners.append(_handle_event)
        self._task = asyncio.create_task(self._run())
        self._running = True
        print(f'[slack] adapter started: {self.channel_id}')

    async def _run(self):
        try:
            await self._socket_client.connect()
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f'[slack] socket mode error: {e}')
            self._running = False

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._socket_client:
            try:
                await self._socket_client.disconnect()
            except Exception:
                pass
        self._socket_client = None
        self._client = None
        print(f'[slack] adapter stopped: {self.channel_id}')

    # ── 健康状态 ─────────────────────────────────────────────────────────────

    async def health_check(self) -> tuple[bool, str]:
        if not self._running or not self._client:
            return False, 'adapter not running'
        try:
            await self._client.auth_test()
            return True, ''
        except Exception as e:
            return False, f'auth_test failed: {e}'

    # ── 发送 ─────────────────────────────────────────────────────────────────

    async def send_message(self, msg: OutboundMessage) -> None:
        if not self._client:
            raise RuntimeError(
                '[slack] Cannot send: adapter not initialized. '
                'Solution: check bot_token/app_token in Channel settings and restart the channel.'
            )

        files = list(msg.files)
        if msg.image_bytes:
            from channel import store
            files.append(store.save_bytes(self.channel_id, msg.image_bytes,
                                          kind=KIND_IMAGE, name='image.jpg',
                                          mime='image/jpeg', fallback_ext='.jpg'))
            if msg.image_caption and not msg.text:
                files[-1].caption = msg.image_caption

        if msg.text:
            await self._client.chat_postMessage(channel=msg.chat_id, text=msg.text)

        if not files:
            return

        # 逐个附件汇报成败：附件失败不该让上层以为已送达的文本也失败了
        sent = ['文本'] if msg.text else []
        failures = []
        for att in files:
            try:
                await self._client.files_upload_v2(
                    channel=msg.chat_id,
                    file=att.path,
                    filename=att.name or os.path.basename(att.path),
                    initial_comment=att.caption or '',
                )
                sent.append(att.name or att.path)
            except Exception as e:
                print(f'[slack] send attachment failed ({att.name or att.path}): {e}')
                failures.append(f'- {att.name or att.path}: {e}')
        if failures:
            raise PartialSendError(sent, failures)

    # ── 接收 ─────────────────────────────────────────────────────────────────

    async def _download_files(self, slack_files: list) -> list[Attachment]:
        if not slack_files:
            return []
        import aiohttp
        from channel import media, store

        token = self.config.get('bot_token', '')
        out: list[Attachment] = []
        timeout = aiohttp.ClientTimeout(total=120)
        headers = {'Authorization': f'Bearer {token}'}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as s:
            for f in slack_files:
                url = f.get('url_private_download') or f.get('url_private') or ''
                if not url:
                    continue
                name = f.get('name', '') or 'file'
                mime = f.get('mimetype', '') or mimetypes.guess_type(name)[0] or ''
                try:
                    async with s.get(url) as resp:
                        if resp.status != 200:
                            print(f'[slack] file download failed ({name}): HTTP {resp.status}'
                                  f' — check the files:read scope')
                            continue
                        payload = await resp.read()
                except Exception as e:
                    print(f'[slack] file download failed ({name}): {e}')
                    continue
                out.append(store.save_bytes(self.channel_id, payload,
                                            kind=media.infer_kind(name, mime),
                                            name=name, mime=mime))
        return out
