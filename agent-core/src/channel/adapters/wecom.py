"""
channel/adapters/wecom.py — WeCom (企业微信) 智能机器人长连接适配器。

与第一代 webhook/relay 方案不同，这里走企业微信官方「智能机器人 · API 模式 ·
使用长连接」：本机作为 WebSocket 客户端主动连 `wss://openws.work.weixin.qq.com`，
只需出网，不需要公网 IP 或回调域名。图片素材需按消息中的密钥解密。

与 feishu.py 一样，由官方 SDK 接收长连接消息，适配器负责归一化消息、
去重、附件落盘及 ChannelAdapter 生命周期。企微回复走 SDK 的 reply_stream。
图片 HTTP 下载保留 aiohttp 的代理、超时与大小限制，解密调用 SDK 的 decrypt_file。
Requires: wecom-aibot-python-sdk >= 1.0.2

Config: {bot_id, secret, bot_name?}
  - bot_id / secret 是长连接专用凭据，与 URL 回调模式的 Token/EncodingAESKey 不同
  - bot_name 可选，用于剥掉群聊正文里的「@机器人名」

平台能力通过类属性声明，框架据此自动适配（不需要写 if platform == 'wecom'）：
  - SUPPORTED_FILE_KINDS = (KIND_IMAGE,) → 支持图片，其它附件由 manager 过滤
  - SUPPORTS_BOT_TO_BOT = False / USES_TRUSTED_BOTS = False
                             → Bot@Bot / trusted_bots 配置由 api/channel.py 自动拒收

SDK 原生使用 asyncio，无需复制飞书 SDK 专用线程和模块级事件循环补丁。
认证成功后才报告 connected；普通断线交给 ChannelManager watchdog 重启，
被其他进程接管后停止自动重连。旧版临时程序曾验证单聊和群聊文本回显；
SDK 版独立适配器的单聊图文回复已验证；完整 Agent 链路和群聊图片仍待验证。
详见 docs/wecom-channel-setup.md。

统一帧形态： `{cmd, headers: {req_id}, body}`
ACK 形态：   `{errcode, errmsg, headers: {req_id}}`
"""

import asyncio
import base64
import binascii
import hashlib
import re
import time
from collections import OrderedDict
from urllib.parse import urljoin, urlsplit

import aiohttp
from yarl import URL

from channel.adapter import (
    KIND_IMAGE, ChannelAdapter, InboundMessage, OnMessageCallback, OutboundMessage, PartialSendError,
)

SUBSCRIBE_TIMEOUT = 15           # 等订阅 ACK 的超时
REPLY_ACK_TIMEOUT = 10           # SDK 自带 5s ACK 超时；此处限制整个调用
DISCONNECT_TIMEOUT = 6           # SDK 的 WebSocket close_timeout 为 5s
DEDUP_MAX = 512                  # 去重窗口，照抄 feishu.py
REQ_ID_MAX = 512                 # msgid → req_id 映射容量
REPLY_WINDOW = 24 * 60 * 60      # 24 小时回复窗口
IMAGE_MAX_BYTES = 10 * 1024 * 1024
IMAGE_MAX_ITEMS = 10
IMAGE_UPLOAD_CHUNK_BYTES = 512 * 1024
IMAGE_DOWNLOAD_TIMEOUT = 10

# 群聊正文里的 @ 前缀，例如 "@Bumi 助手 挥个手"
_AT_PREFIX_RE = re.compile(r'^\s*@[^\s]+\s*')


def _frame_req_id(frame: dict) -> str:
    headers = frame.get('headers', {})
    return str(headers.get('req_id') or '') if isinstance(headers, dict) else ''


def _sdk_error_detail(error: Exception) -> str:
    """SDK errors can contain platform payloads; expose only a code or type."""
    text = str(error)
    match = re.match(r'^Reply ack error: errcode=(-?\d{1,10})(?:,|$)', text)
    if text.startswith('Authentication failed:'):
        match = re.search(r'\(code: (-?\d{1,10})\)$', text)
    return f'errcode={match[1]}' if match else f'error_type={type(error).__name__}'


class _SdkLogger:
    """The SDK logs whole frames even on warnings; never forward raw payloads."""

    def __init__(self):
        self._warnings = 0
        self._errors = 0

    def debug(self, message, *args):
        pass

    def info(self, message, *args):
        pass

    def warn(self, message, *args):
        self._warnings += 1
        if self._warnings == 1 or self._warnings % 100 == 0:
            print(f'[wecom] SDK warnings: count={self._warnings}; payload omitted')

    def error(self, message, *args):
        self._errors += 1
        if self._errors == 1 or self._errors % 100 == 0:
            print(f'[wecom] SDK errors: count={self._errors}; payload omitted')


class WecomError(RuntimeError):
    """企业微信长连接返回了非 0 errcode，或凭据/连接不可用。"""


def _image_type(data: bytes) -> tuple[str, str]:
    """Return the actual format, rather than trusting a URL or file extension."""
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg', '.jpg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png', '.png'
    raise WecomError('WeCom images must be JPEG or PNG.')


def _image_failure_reason(error: Exception) -> str:
    """Describe expected failures without logging the signed URL or AES key."""
    if not isinstance(error, WecomError):
        return f'error_type={type(error).__name__}'
    text = str(error)
    status = re.fullmatch(r'WeCom image download failed \(HTTP (\d{3})\)\.', text)
    if status:
        return f'http_status={status[1]}'
    if 'missing its URL or decryption key' in text or 'Invalid WeCom image metadata' in text:
        return 'missing_metadata'
    if 'exceeds the 10 MiB limit' in text:
        return 'size_limit'
    if 'decrypt' in text or 'encrypted data' in text:
        return 'decrypt_failed'
    if 'must be JPEG or PNG' in text:
        return 'unsupported_format'
    if 'download failed or timed out' in text:
        return 'network_or_timeout'
    if 'HTTPS URL' in text or 'redirect' in text:
        return 'invalid_download_url'
    if 'session is unavailable' in text:
        return 'download_session_unavailable'
    return 'image_error'


def _progress_failure_reason(error: Exception) -> str:
    """Return a safe platform error code when available for troubleshooting."""
    if isinstance(error, WecomError):
        message = str(error)
        code = re.fullmatch(r'WeCom reply rejected \(errcode=(-?\d{1,10})\)\.', message)
        if code:
            return f'errcode={code[1]}'
        if 'ACK timed out' in message:
            return 'ack_timeout'
        if 'not connected' in message or 'connection closed' in message:
            return 'disconnected'
    return f'error_type={type(error).__name__}'


def _decrypt_image(data: bytes, aeskey: str) -> bytes:
    """Validate the envelope, then use the official SDK's public decryptor."""
    from aibot import decrypt_file

    try:
        encoded = aeskey.encode('ascii')
        key = base64.b64decode(encoded + b'=' * (-len(encoded) % 4), validate=True)
    except (AttributeError, UnicodeEncodeError, ValueError, binascii.Error):
        raise WecomError('Invalid WeCom image decryption key.') from None
    # SDK 1.0.2 pads truncated ciphertext with zeros; reject it before decryption.
    if len(key) != 32 or not data or len(data) % 16:
        raise WecomError('Invalid WeCom image key or encrypted data.')
    try:
        return decrypt_file(data, aeskey)
    except (ValueError, RuntimeError):
        raise WecomError('WeCom image decryption failed.') from None


class WecomAdapter(ChannelAdapter):
    """企业微信智能机器人（API 模式 · 长连接）适配器。"""

    SUPPORTED_FILE_KINDS = (KIND_IMAGE,)
    SUPPORTS_BOT_TO_BOT = False
    USES_TRUSTED_BOTS = False

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._client = None                       # official aibot.WSClient
        self._session: aiohttp.ClientSession | None = None
        self._ready = asyncio.Event()
        self._authenticated = False
        self._message_tasks: set[asyncio.Task] = set()
        self._reply_locks: dict[str, asyncio.Lock] = {}
        self._reply_lock_users: dict[str, int] = {}
        self._sdk_logger = _SdkLogger()
        self._taken_over = False
        self._last_error = ''
        self._progress_failures = 0
        self._image_download_failures = 0
        self._message_failures = 0
        self._invalid_message_drops = 0
        self._received_counts: dict[tuple[str, str], int] = {}

        # 去重（照抄 feishu.py:575-590 的有界 LRU）
        self._seen_ids: list[str] = []
        self._seen_set: set[str] = set()
        self._duplicate_drops = 0

        # msgid -> (req_id, ts, stream_id); only authorized messages are stored.
        self._msgid_to_req: OrderedDict[str, tuple[str, float, str]] = OrderedDict()

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def _create_client(self, bot_id: str, secret: str):
        # 与飞书一样延迟导入 SDK，未启用的渠道不影响其他平台加载。
        from aibot import WSClient, WSClientOptions

        return WSClient(WSClientOptions(
            bot_id=bot_id, secret=secret,
            max_reconnect_attempts=0,  # 项目 watchdog 统一重启，避免双重重连
            logger=self._sdk_logger,
        ))

    async def start(self) -> None:
        """As in Feishu, validate credentials before reporting startup success."""
        bot_id = str(self.config.get('bot_id', '') or '').strip()
        secret = str(self.config.get('secret', '') or '').strip()
        if not bot_id or not secret:
            raise WecomError(
                'WeCom bot_id and secret are required. '
                'Configure the API-mode long-connection credentials in Settings → Channels.'
            )
        if self._client is not None:
            await self.stop()
        self._taken_over = False
        self._last_error = ''
        self._authenticated = False
        self._ready.clear()
        client = self._create_client(bot_id, secret)
        self._client = client

        def current(callback):
            def handle(*args):
                # Ignore callbacks from a client which is stopping/replaced.
                if self._client is client:
                    callback(*args)
            return handle

        client.on('authenticated', current(self._on_authenticated))
        client.on('disconnected', current(self._on_disconnected))
        client.on('error', current(self._on_sdk_error))
        client.on('message', current(self._handle_message_event))
        client.on('event', current(self._handle_event))

        async def connect_and_authenticate():
            await client.connect()
            # connect() only opens the socket; authentication arrives separately.
            await self._ready.wait()
            if not self._authenticated or not self._running or not client.is_connected:
                raise WecomError(self._last_error or 'WeCom authentication failed.')

        try:
            self._session = aiohttp.ClientSession(trust_env=True)
            await asyncio.wait_for(connect_and_authenticate(), timeout=SUBSCRIBE_TIMEOUT)
        except asyncio.TimeoutError:
            await self.stop()
            raise WecomError(
                'WeCom subscription timed out. Check outbound access to '
                'wss://openws.work.weixin.qq.com:443 and the bot API mode setting.'
            ) from None
        except (WecomError, asyncio.CancelledError):
            await self.stop()
            raise
        except Exception as e:
            await self.stop()
            raise WecomError(f'WeCom SDK startup failed ({_sdk_error_detail(e)}).') from None
        print(f'[wecom] adapter started (official SDK): {self.channel_id}')

    def _on_authenticated(self) -> None:
        if self._taken_over or (self._ready.is_set() and not self._running):
            return
        self._authenticated = True
        self._running = True
        self._last_error = ''
        self._ready.set()

    def _on_sdk_error(self, error: Exception) -> None:
        # Keep the original failure: disabling SDK reconnect emits another error.
        if self._taken_over or (self._ready.is_set() and not self._running):
            return
        self._last_error = f'WeCom SDK connection failed ({_sdk_error_detail(error)}).'
        self._running = False
        self._authenticated = False
        self._ready.set()
        print(f'[wecom] {self._last_error}')

    def _on_disconnected(self, reason: str) -> None:
        if self._taken_over:
            return
        if self._running:
            print(f'[wecom] connection lost: {self.channel_id}; watchdog will restart it')
        self._running = False
        self._authenticated = False
        self._last_error = 'WeCom SDK connection closed.'
        self._ready.set()

    async def _close_client(self) -> None:
        self._running = False
        self._authenticated = False
        client, self._client = self._client, None
        if client is not None:
            client.disconnect()
            # SDK disconnect() schedules shutdown and returns immediately.
            deadline = asyncio.get_running_loop().time() + DISCONNECT_TIMEOUT
            while client.is_connected:
                if asyncio.get_running_loop().time() >= deadline:
                    raise WecomError('WeCom SDK disconnect timed out.')
                await asyncio.sleep(0.05)

    async def stop(self) -> None:
        """Stop the SDK and drain only this adapter's message tasks."""
        self._running = False
        self._authenticated = False
        self._ready.set()
        try:
            await self._close_client()
        finally:
            tasks = [task for task in self._message_tasks
                     if task is not asyncio.current_task() and not task.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._message_tasks.clear()
            if self._session is not None:
                await self._session.close()
                self._session = None
            self._msgid_to_req.clear()
        print(f'[wecom] adapter stopped: {self.channel_id}')

    # ── 入站 ────────────────────────────────────────────────────────────────

    def _handle_message_event(self, frame: dict) -> None:
        """Schedule business processing without blocking SDK callbacks/ACKs."""
        if not self._running:
            return
        body = frame.get('body') or {}
        if isinstance(body, dict):
            chat_type = body.get('chattype') if body.get('chattype') in ('single', 'group') else 'other'
            msg_type = body.get('msgtype') if body.get('msgtype') in ('text', 'image', 'mixed', 'voice') else 'other'
            key = (chat_type, msg_type)
            count = self._received_counts.get(key, 0) + 1
            self._received_counts[key] = count
            if count == 1 or count % 100 == 0:
                print(f'[wecom] received callback: chat_type={chat_type} msg_type={msg_type} count={count}')
        task = asyncio.create_task(self._dispatch_message(frame))
        self._message_tasks.add(task)
        task.add_done_callback(self._message_tasks.discard)

    async def _dispatch_message(self, frame: dict) -> None:
        try:
            body = frame.get('body')
            if isinstance(body, dict):
                await self._process_message(frame, body)
        except Exception as e:
            self._message_failures += 1
            if self._message_failures == 1 or self._message_failures % 100 == 0:
                print('[wecom] process message failed: '
                      f'count={self._message_failures} error_type={type(e).__name__}')

    def _handle_event(self, frame: dict) -> None:
        body = frame.get('body', {})
        event = body.get('event', {}) if isinstance(body, dict) else {}
        if not isinstance(event, dict) or event.get('eventtype') != 'disconnected_event':
            return
        if self._taken_over:
            return
        # Disable watchdog reconnect immediately, before scheduling socket close.
        self._taken_over = True
        self._running = False
        self._authenticated = False
        self._last_error = (
            'long connection taken over by another process for the same Bot ID; '
            'automatic reconnect disabled'
        )
        self._ready.set()
        print(f'[wecom] {self.channel_id}: {self._last_error}. '
              'Stop the other process, then Restart this channel.')
        if self._client is not None:
            self._client.disconnect()

    def _is_duplicate(self, message_id: str) -> bool:
        if not message_id:
            return False
        if message_id in self._seen_set:
            self._duplicate_drops += 1
            if self._duplicate_drops == 1 or self._duplicate_drops % 100 == 0:
                print(f'[wecom] duplicate messages dropped: '
                      f'count={self._duplicate_drops} latest_id={str(message_id)[:128]!r}')
            return True
        self._seen_set.add(message_id)
        self._seen_ids.append(message_id)
        if len(self._seen_ids) > DEDUP_MAX:
            self._seen_set.discard(self._seen_ids.pop(0))
        return False

    def _extract_inbound(self, frame: dict, body: dict) -> dict:
        """Normalize the official SDK message payload, without guessed aliases."""
        sender = body.get('from') or {}
        user_id = str(sender.get('userid') or '')
        chat_type = 'group' if body.get('chattype') == 'group' else 'p2p'
        chat_id = str(body.get('chatid') or (user_id if chat_type == 'p2p' else ''))
        msgtype = body.get('msgtype', '')
        text = ''
        images = []
        if msgtype == 'text':
            text = str((body.get('text') or {}).get('content') or '')
        elif msgtype == 'voice':
            text = str((body.get('voice') or {}).get('content') or '')
        elif msgtype == 'image':
            images.append(body.get('image') or {})
        elif msgtype == 'mixed':
            parts = []
            for item in (body.get('mixed') or {}).get('msg_item', []):
                if item.get('msgtype') == 'text':
                    parts.append(str((item.get('text') or {}).get('content') or ''))
                elif item.get('msgtype') == 'image':
                    images.append(item.get('image') or {})
            text = ''.join(parts)
        if not text:
            text = '[图片]' if images else '[收到不支持的非文本消息]'
        return {
            'message_id': str(body.get('msgid') or ''),
            'user_id': user_id,
            'chat_id': chat_id,
            'chat_type': chat_type,
            'text': text,
            'msgtype': msgtype,
            'images': images,
        }

    async def _download_image(self, image: dict):
        """Download and decrypt one WeCom image; never log its signed URL or key."""
        from channel import store

        if not isinstance(image, dict):
            raise WecomError('Invalid WeCom image metadata.')
        url, aeskey = image.get('url'), image.get('aeskey')
        if not isinstance(url, str) or not url or (aeskey is not None and not isinstance(aeskey, str)):
            raise WecomError('Invalid WeCom image metadata.')
        if self._session is None or self._session.closed:
            raise WecomError('WeCom image download session is unavailable.')

        try:
            for _ in range(4):
                parsed = urlsplit(url)
                if (parsed.scheme != 'https' or not parsed.hostname or
                        parsed.username is not None or parsed.password is not None):
                    raise WecomError('WeCom image download requires an HTTPS URL.')
                async with self._session.get(
                    # WeCom media URLs can be pre-signed: do not let aiohttp
                    # requote the path/query before sending the request.
                    URL(url, encoded=True), timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                    allow_redirects=False, auto_decompress=False,
                ) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get('Location')
                        if not location:
                            raise WecomError('WeCom image redirect has no destination.')
                        url = urljoin(url, location)
                        continue
                    if response.status != 200:
                        raise WecomError(f'WeCom image download failed (HTTP {response.status}).')
                    encrypted = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        encrypted.extend(chunk)
                        if len(encrypted) > IMAGE_MAX_BYTES + 32:
                            raise WecomError('WeCom image exceeds the 10 MiB limit.')
                    break
            else:
                raise WecomError('Too many WeCom image redirects.')
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            raise WecomError('WeCom image download failed or timed out.') from e

        # The SDK's download_file also permits media without an AES key. Only
        # accept the result after checking its actual JPEG/PNG signature below.
        data = _decrypt_image(bytes(encrypted), aeskey) if aeskey else bytes(encrypted)
        if not data or len(data) > IMAGE_MAX_BYTES:
            raise WecomError('WeCom image is empty or exceeds the 10 MiB limit.')
        mime, ext = _image_type(data)
        return store.save_bytes(self.channel_id, data, kind=KIND_IMAGE,
                                name=f'image{ext}', mime=mime, fallback_ext=ext)

    async def _prepare_images(self, msg: InboundMessage, images: list) -> None:
        """Materialize media only after the channel and sender are authorized."""
        failures = 0
        for image in images[:IMAGE_MAX_ITEMS]:
            try:
                msg.attachments.append(await self._download_image(image))
            except Exception as e:
                failures += 1
                self._image_download_failures += 1
                if self._image_download_failures == 1 or self._image_download_failures % 100 == 0:
                    print('[wecom] image download failed: '
                          f'count={self._image_download_failures} reason={_image_failure_reason(e)}')
        failures += max(0, len(images) - IMAGE_MAX_ITEMS)
        if failures:
            msg.text += f' [图片处理失败或超限：{failures} 张]'

    async def _process_message(self, frame: dict, body: dict) -> None:
        parsed = self._extract_inbound(frame, body)
        req_id = _frame_req_id(frame)
        msg_id = parsed['message_id']
        if not msg_id or not req_id or not parsed['user_id'] or not parsed['chat_id']:
            self._invalid_message_drops += 1
            if self._invalid_message_drops == 1 or self._invalid_message_drops % 100 == 0:
                missing = [name for name, value in (
                    ('msgid', msg_id), ('req_id', req_id),
                    ('userid', parsed['user_id']), ('chatid', parsed['chat_id']),
                ) if not value]
                print('[wecom] invalid message dropped: '
                      f'count={self._invalid_message_drops} missing={",".join(missing)}')
            return
        if self._is_duplicate(msg_id):
            return

        text = parsed['text']
        if parsed['chat_type'] == 'group':
            # 剥掉群聊正文里的 @机器人名；配置了 bot_name 时只剥那一个，否则剥任意 @前缀
            bot_name = str(self.config.get('bot_name', '') or '').strip()
            if bot_name and text.lstrip().startswith('@' + bot_name):
                text = text.lstrip()[len('@' + bot_name):].lstrip()
            else:
                text = _AT_PREFIX_RE.sub('', text, count=1)

        msg = InboundMessage(
            platform='wecom',
            channel_id=self.channel_id,
            user_id=parsed['user_id'],
            chat_id=parsed['chat_id'],
            display_name=parsed['user_id'],
            text=text,
            message_id=msg_id,
            attachments=[],
            sender_type='user',
            chat_type=parsed['chat_type'],
            mentions=[],
            expect_reply=None,
        )

        from aibot import generate_req_id
        stream_id = generate_req_id('stream')

        async def prepare_authorized_message():
            # Called by ChannelManager only after channel activation and ACL.
            self._msgid_to_req[msg_id] = (req_id, time.time(), stream_id)
            while len(self._msgid_to_req) > REQ_ID_MAX:
                self._msgid_to_req.popitem(last=False)
            try:
                await self._send_progress(msg_id)
            except Exception as e:
                # The callback still reaches the normal message path; that path
                # will report any subsequent reply failure to its caller.
                self._progress_failures += 1
                if self._progress_failures == 1 or self._progress_failures % 100 == 0:
                    print('[wecom] progress reply failed: '
                          f'count={self._progress_failures} reason={_progress_failure_reason(e)}')
            if parsed['images']:
                await self._prepare_images(msg, parsed['images'])

        async def notify_pending_approval():
            notice = OutboundMessage(chat_id=msg.chat_id, text='Pending approval. An admin has been notified.',
                                     source_message_id=msg_id)
            await self._send_reply(req_id, notice, stream_id=stream_id)

        msg.prepare = prepare_authorized_message
        msg.notify_pending_approval = notify_pending_approval
        await self._on_message(msg)

    # ── 出站 ────────────────────────────────────────────────────────────────

    def _lookup_reply_context(self, msg: OutboundMessage) -> tuple[str, str]:
        # 方案 B：manager.send_to_channel 已把触发事件的 message_id 透传进来
        source_id = str(getattr(msg, 'source_message_id', '') or '')
        entry = self._msgid_to_req.get(source_id) if source_id else None
        if entry:
            req_id, ts, stream_id = entry
            if time.time() - ts > REPLY_WINDOW:
                raise WecomError(
                    'Reply window expired (24h). Re-trigger from a current '
                    'message instead of reusing an old source_message_id.'
                )
            return req_id, stream_id
        raise WecomError(
            'No reply context for this message: the inbound req_id is unknown or '
            'expired. The user must send a new message first.'
        )

    async def _send_progress(self, message_id: str) -> None:
        msg = OutboundMessage(chat_id='', text='正在处理…', source_message_id=message_id)
        req_id, stream_id = self._lookup_reply_context(msg)
        await self._send_reply(req_id, msg, stream_id=stream_id, finish=False)

    async def send_message(self, msg: OutboundMessage) -> None:
        """回复原消息。失败必须抛异常，让上层如实回报给 LLM / 用户。"""
        if self.status() != 'connected':
            raise WecomError(
                'Cannot send: WeCom adapter is not connected. '
                'Check bot_id/secret in Settings → Channels and Restart.'
            )
        if any(att.kind != KIND_IMAGE for att in msg.files):
            raise WecomError('WeCom currently supports image attachments only.')
        count = len(msg.files) + (1 if msg.image_bytes is not None else 0)
        if count > IMAGE_MAX_ITEMS:
            raise WecomError('WeCom supports at most 10 images in one reply.')
        if not msg.text and not count and not msg.image_caption:
            return
        req_id, stream_id = self._lookup_reply_context(msg)

        def read_image(path: str) -> bytes:
            with open(path, 'rb') as source:
                return source.read(IMAGE_MAX_BYTES + 1)

        images = []
        for att in msg.files:
            images.append(await asyncio.to_thread(read_image, att.path))
        if msg.image_bytes is not None:
            images.append(msg.image_bytes)
        extensions = []
        for data in images:
            if not data or len(data) > IMAGE_MAX_BYTES:
                raise WecomError('WeCom image is empty or exceeds the 10 MiB limit.')
            _, extension = _image_type(data)
            extensions.append(extension)
        captions = [att.caption for att in msg.files if att.caption]
        if msg.image_caption:
            captions.append(msg.image_caption)
        content = '\n'.join(part for part in [msg.text, *captions] if part)
        reply = OutboundMessage(chat_id=msg.chat_id, text=content,
                                source_message_id=msg.source_message_id)
        # Finish the existing progress stream before sending standalone media;
        # otherwise the client's "processing" indicator remains visible.
        await self._send_reply(req_id, reply, stream_id=stream_id)
        sent = ['text'] if content else []
        failures = []
        for index, (data, extension) in enumerate(zip(images, extensions), 1):
            try:
                media_id = await self._upload_reply_image(data, extension)
                await self._send_body(req_id, {'msgtype': 'image', 'image': {'media_id': media_id}})
                sent.append(f'image {index}')
            except WecomError as exc:
                failures.append(f'image {index}: {exc}')
                if self.status() != 'connected':
                    failures.extend(f'image {remaining}: not sent (disconnected)'
                                    for remaining in range(index + 1, len(images) + 1))
                    break
        if failures:
            if sent:
                raise PartialSendError(sent, failures)
            raise WecomError('; '.join(failures))

    async def _send_reply(self, req_id: str, msg: OutboundMessage, *, stream_id: str,
                          finish: bool = True) -> None:
        await self._send_request(
            req_id,
            lambda client: client.reply_stream(
                {'headers': {'req_id': req_id}}, stream_id=stream_id,
                content=msg.text, finish=finish,
            ),
        )

    async def _send_body(self, req_id: str, body: dict, cmd: str | None = None) -> dict:
        return await self._send_request(
            req_id,
            lambda client: client.reply({'headers': {'req_id': req_id}}, body, cmd=cmd),
        )

    async def _upload_reply_image(self, data: bytes, extension: str) -> str:
        # The official Node SDK uses these WebSocket commands. Python SDK 1.0.2
        # exposes them through its public generic reply() method.
        from aibot import generate_req_id

        async def command(name: str, body: dict) -> dict:
            return await self._send_body(generate_req_id(name), body, cmd=name)

        chunks = (len(data) + IMAGE_UPLOAD_CHUNK_BYTES - 1) // IMAGE_UPLOAD_CHUNK_BYTES
        result = await command('aibot_upload_media_init', {
            'type': 'image', 'filename': f'image{extension}', 'total_size': len(data),
            'total_chunks': chunks, 'md5': hashlib.md5(data, usedforsecurity=False).hexdigest(),
        })
        response_body = result.get('body')
        upload_id = response_body.get('upload_id') if isinstance(response_body, dict) else None
        if not isinstance(upload_id, str) or not upload_id:
            raise WecomError('WeCom image upload did not return an upload ID.')
        for index in range(chunks):
            chunk = data[index * IMAGE_UPLOAD_CHUNK_BYTES:(index + 1) * IMAGE_UPLOAD_CHUNK_BYTES]
            await command('aibot_upload_media_chunk', {
                'upload_id': upload_id, 'chunk_index': index,
                'base64_data': base64.b64encode(chunk).decode('ascii'),
            })
        result = await command('aibot_upload_media_finish', {'upload_id': upload_id})
        response_body = result.get('body')
        media_id = response_body.get('media_id') if isinstance(response_body, dict) else None
        if not isinstance(media_id, str) or not media_id:
            raise WecomError('WeCom image upload did not return a media ID.')
        return media_id

    async def _send_request(self, req_id: str, send) -> dict:
        client = self._client
        if self.status() != 'connected':
            raise WecomError('Cannot send: WeCom adapter is not connected.')
        lock = self._reply_locks.setdefault(req_id, asyncio.Lock())
        self._reply_lock_users[req_id] = self._reply_lock_users.get(req_id, 0) + 1
        try:
            # Keep one SDK call per req_id in flight: a timeout/cancellation must
            # close this connection before a later call can consume a late ACK.
            async with lock:
                if self._client is not client or self.status() != 'connected':
                    raise WecomError('Cannot send: WeCom adapter is not connected.')
                try:
                    ack = await asyncio.wait_for(send(client), timeout=REPLY_ACK_TIMEOUT)
                except (asyncio.TimeoutError, TimeoutError):
                    self._last_error = 'WeCom reply ACK timed out; delivery is unknown.'
                    await self._close_client()
                    raise WecomError(self._last_error + ' Check the chat before retrying.') from None
                except asyncio.CancelledError:
                    if self._client is client:
                        self._last_error = 'WeCom reply cancelled before its ACK; delivery is unknown.'
                        await self._close_client()
                    raise
                except Exception as e:
                    if self._client is not client or not self._running:
                        raise WecomError('WeCom connection closed before the reply ACK.') from None
                    raise WecomError(f'WeCom reply rejected ({_sdk_error_detail(e)}).') from None
                if not isinstance(ack, dict) or ack.get('errcode') != 0 or _frame_req_id(ack) != req_id:
                    raise WecomError('WeCom reply returned an invalid or rejected ACK.')
                return ack
        finally:
            remaining = self._reply_lock_users[req_id] - 1
            if remaining:
                self._reply_lock_users[req_id] = remaining
            else:
                self._reply_lock_users.pop(req_id, None)
                self._reply_locks.pop(req_id, None)

    # ── 健康状态 ────────────────────────────────────────────────────────────

    def should_auto_reconnect(self) -> bool:
        return not self._taken_over

    async def health_check(self) -> tuple[bool, str]:
        if self.status() != 'connected':
            return False, self._last_error or 'WeCom SDK is not authenticated/connected.'
        return True, ''

    def status(self) -> str:
        # Heartbeats and dead-socket detection belong to the SDK. Idle chats do
        # not imply failure; is_connected alone is insufficient before auth.
        if (self._taken_over or not self._running or not self._authenticated or
                self._client is None or not self._client.is_connected):
            return 'disconnected'
        return 'connected'
