"""
channel/adapters/slack.py — Slack Socket Mode 适配器。

使用 Socket Mode（WebSocket），无需公网 IP。
依赖：slack-sdk (>=3.27)
"""

import asyncio

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage, OnMessageCallback


class SlackAdapter(ChannelAdapter):
    """Slack Socket Mode 适配器。"""

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._handler = None
        self._client = None
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
        self._socket_client = SocketModeClient(app_token=app_token, web_client=self._client)

        async def _handle_event(client, req: SocketModeRequest):
            # Acknowledge
            await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

            if req.type == 'events_api' and req.payload:
                event = req.payload.get('event', {})
                if event.get('type') == 'message' and not event.get('subtype'):
                    text = event.get('text', '')
                    user_id = event.get('user', '')
                    channel_id_slack = event.get('channel', '')
                    if text and user_id:
                        msg = InboundMessage(
                            platform='slack',
                            channel_id=self.channel_id,
                            user_id=user_id,
                            chat_id=channel_id_slack,
                            display_name=user_id,  # Could resolve via users.info API
                            text=text,
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

    async def send_message(self, msg: OutboundMessage) -> None:
        if not self._client:
            return
        if msg.image_bytes:
            await self._client.files_upload_v2(
                channel=msg.chat_id,
                content=msg.image_bytes,
                filename='image.jpg',
                initial_comment=msg.image_caption or msg.text or '',
            )
        elif msg.text:
            await self._client.chat_postMessage(
                channel=msg.chat_id,
                text=msg.text,
            )
