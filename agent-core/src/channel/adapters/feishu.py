"""
channel/adapters/feishu.py — Feishu (Lark) adapter using WebSocket long connection.

Uses lark-oapi SDK's built-in WebSocket mode — outbound connection to Feishu servers,
no public IP or webhook needed. Same pattern as Telegram long-polling and Slack Socket Mode.

Requires: pip install lark-oapi
Config: {app_id, app_secret}

Required Feishu permissions:
- im:message — receive messages
- im:message:send_as_bot — send messages as bot
- im:chat:readonly — list chats (optional)

Event subscription:
- Event: im.message.receive_v1
- Mode: 长连接 (WebSocket long connection)
"""

import asyncio
import json
import threading

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage, OnMessageCallback

# Common Feishu error codes and actionable messages
_FEISHU_ERROR_HINTS = {
    10003: 'Invalid app_id. Check your Feishu app credentials in Channel settings.',
    10014: 'Invalid app_secret. Check your Feishu app credentials in Channel settings.',
    99991663: 'Tenant token invalid. The adapter will attempt to reconnect automatically. If this persists, restart the channel.',
    99991668: 'Tenant token expired. The SDK auto-refreshes tokens; if this persists, check app_id/app_secret in Channel settings.',
    99991672: 'Permission denied. Grant the required permission in Feishu Developer Console: https://open.feishu.cn/app/{app_id}/auth',
    230001: 'Bot not in this chat. Add the bot to the chat first, or the user needs to message the bot directly.',
    230002: 'Bot has been removed from chat. Re-add the bot.',
    230006: 'Message send failed: bot not activated. Publish your app version in Feishu Developer Console.',
    230014: 'Message too long. Maximum 4096 characters.',
}


class FeishuAdapter(ChannelAdapter):
    """Feishu/Lark adapter using SDK WebSocket long connection."""

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._client = None
        self._task: asyncio.Task | None = None
        self._api_client = None
        self._thread: threading.Thread | None = None

    async def start(self) -> None:
        app_id = self.config.get('app_id', '')
        app_secret = self.config.get('app_secret', '')
        if not app_id or not app_secret:
            raise ValueError(
                'Feishu app_id and app_secret are required. '
                'Configure them in Settings → Channels.'
            )

        import lark_oapi as lark

        # Store the main event loop for cross-thread scheduling
        self._loop = asyncio.get_event_loop()

        # Create API client for sending messages
        self._api_client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .build()

        # Create event handler
        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._handle_message_event) \
            .build()

        # Create WebSocket long connection client
        self._client = lark.ws.Client(
            app_id=app_id,
            app_secret=app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
            auto_reconnect=True,
        )

        # Hook into SDK reconnect lifecycle for status tracking
        self._client.on_reconnecting = self._on_reconnecting
        self._client.on_reconnected = self._on_reconnected

        # Start in background thread — SDK's start() blocks (contains _select())
        self._running = True
        self._thread = threading.Thread(target=self._thread_target, daemon=True, name='feishu-ws')
        self._thread.start()
        print(f'[feishu] adapter started (WebSocket mode): {self.channel_id}')

    def _thread_target(self):
        """Run lark SDK client.start() in a dedicated thread.

        SDK's start() does:
        1. _connect() — establishes WS, starts _receive_message_loop task
        2. _ping_loop() — periodic keepalive pings
        3. _select() — keeps event loop alive forever

        _receive_message_loop has built-in auto_reconnect on disconnect.
        """
        # Lark SDK stores the event loop in a module-level variable (lark_oapi.ws.client.loop)
        # which captures the main uvloop at import time. Replace it with a fresh loop
        # so SDK operations run independently of the main event loop.
        import lark_oapi.ws.client as ws_mod
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        ws_mod.loop = new_loop
        try:
            self._client.start()
        except Exception as e:
            err_msg = str(e)
            if 'invalid' in err_msg.lower() and ('app_id' in err_msg.lower() or 'secret' in err_msg.lower()):
                print(f'[feishu] Connection failed: invalid app credentials. '
                      f'Check app_id and app_secret in Channel settings.')
            else:
                print(f'[feishu] WebSocket connection error: {e}')
            self._running = False

    def _on_reconnecting(self):
        print(f'[feishu] connection lost, reconnecting... ({self.channel_id})')

    def _on_reconnected(self):
        print(f'[feishu] reconnected successfully ({self.channel_id})')

    async def stop(self) -> None:
        self._running = False

        # Stop the SDK's event loop to break out of _select()
        import lark_oapi.ws.client as ws_mod
        if ws_mod.loop and ws_mod.loop.is_running():
            ws_mod.loop.call_soon_threadsafe(ws_mod.loop.stop)

        # Wait for thread to exit
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self._client = None
        self._thread = None
        print(f'[feishu] adapter stopped: {self.channel_id}')

    async def send_message(self, msg: OutboundMessage) -> None:
        """Send message via Feishu Open API."""
        if not self._api_client:
            raise RuntimeError(
                '[feishu] Cannot send: adapter not initialized. '
                'Cause: The adapter failed to start or has been stopped. '
                'Solution: Check app_id/app_secret in Channel settings and restart the channel.'
            )

        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        body = CreateMessageRequestBody.builder() \
            .receive_id(msg.chat_id) \
            .msg_type('text') \
            .content(json.dumps({'text': msg.text})) \
            .build()

        request = CreateMessageRequest.builder() \
            .receive_id_type('chat_id') \
            .request_body(body) \
            .build()

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, self._api_client.im.v1.message.create, request
            )
            if not response.success():
                hint = _FEISHU_ERROR_HINTS.get(response.code, '')
                app_id = self.config.get('app_id', '')
                if hint:
                    hint = hint.format(app_id=app_id)
                error_detail = f'[feishu] send_message failed (code={response.code}): {response.msg}'
                if hint:
                    error_detail += f'\n  → {hint}'
                print(error_detail)
                raise RuntimeError(error_detail)
        except RuntimeError:
            raise
        except Exception as e:
            error_str = str(e)
            if 'timeout' in error_str.lower() or 'connect' in error_str.lower():
                raise RuntimeError(
                    f'[feishu] send_message network error: {e}\n'
                    f'  → Cause: Network unreachable or Feishu API rate-limited.\n'
                    f'  → Solution: Check network connectivity. Retry in a few seconds.'
                )
            raise RuntimeError(f'[feishu] send_message exception: {e}')

    def _handle_message_event(self, data):
        """Handle im.message.receive_v1 event from SDK callback (runs in SDK thread)."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # Skip bot messages
            if sender.sender_type == 'app':
                return

            # Extract text
            msg_type = message.message_type
            text = ''
            if msg_type == 'text':
                content = json.loads(message.content)
                text = content.get('text', '')
            else:
                text = f'[{msg_type}]'

            if not text:
                return

            sender_id = sender.sender_id.open_id or sender.sender_id.user_id or ''
            chat_id = message.chat_id

            msg = InboundMessage(
                platform='feishu',
                channel_id=self.channel_id,
                user_id=sender_id,
                chat_id=chat_id,
                display_name=sender_id,
                text=text,
            )

            # Schedule coroutine from SDK thread to main event loop
            asyncio.run_coroutine_threadsafe(self._on_message(msg), self._loop)

        except Exception as e:
            print(f'[feishu] handle message error: {e}')
            print(f'  If messages are not being received, ensure:')
            print(f'  1. Event "im.message.receive_v1" is subscribed in Feishu Developer Console')
            print(f'  2. Subscription mode is set to "长连接" (WebSocket long connection)')
            print(f'  3. App has "im:message" permission and is published')
