"""
channel/adapters/wecom.py — WeCom (企业微信) 智能机器人长连接适配器。

与第一代 webhook/relay 方案不同，这里走企业微信官方「智能机器人 · API 模式 ·
使用长连接」：本机作为 WebSocket 客户端主动连 `wss://openws.work.weixin.qq.com`，
只需出网，不需要公网 IP 或回调域名。图片素材需按消息中的密钥解密。

设计上刻意不引入官方 Python SDK：使用 aiohttp 和 pycryptodome，
部署新代码仍需构建或获取包含它的新镜像。帧格式可直接调试
（飞书链路曾因 SDK 版本升级打断过，见 feishu.py 的 _enable_sdk_env_proxy）。

Config: {bot_id, secret, bot_name?}
  - bot_id / secret 是长连接专用凭据，与 URL 回调模式的 Token/EncodingAESKey 不同
  - bot_name 可选，用于剥掉群聊正文里的「@机器人名」

平台能力通过类属性声明，框架据此自动适配（不需要写 if platform == 'wecom'）：
  - SUPPORTED_FILE_KINDS = (KIND_IMAGE,) → 支持图片，其它附件由 manager 过滤
  - SUPPORTS_BOT_TO_BOT = False / USES_TRUSTED_BOTS = False
                             → Bot@Bot / trusted_bots 配置由 api/channel.py 自动拒收

## 协议帧字段（待 L0 实测核对）

官方文档 101463 / 60904 的正文是动态渲染的，静态抓取只能拿到目录导航，
拿不到逐字 JSON 示例。下列字段来自官方开发前必读 101805、帮助中心 21677
与官方 SDK 的行为描述，仍需真实帧样本验证。协议常量和主要字段解析集中在下面的
`FIELD_*` / `CMD_*` 常量区与 `_extract_*` 函数里；订阅和回复回执也需联调核对。
详见 docs/wecom-channel-setup.md。

统一帧形态： `{cmd, headers: {req_id}, body}`
ACK 形态：   `{errcode, errmsg, headers: {req_id}}`
"""

import asyncio
import base64
import binascii
import hashlib
import json
import re
import time
import uuid
from collections import OrderedDict
from urllib.parse import urljoin, urlsplit

import aiohttp
from Crypto.Cipher import AES

from channel.adapter import (
    KIND_IMAGE, ChannelAdapter, InboundMessage, OnMessageCallback, OutboundMessage,
)

# ── 协议常量区（L0 实测后集中调整） ──────────────────────────────────────────
WSS_URL = 'wss://openws.work.weixin.qq.com'

CMD_SUBSCRIBE = 'aibot_subscribe'
CMD_MSG_CALLBACK = 'aibot_msg_callback'
CMD_EVENT_CALLBACK = 'aibot_event_callback'
CMD_RESPOND = 'aibot_respond_msg'
CMD_PING = 'ping'

# 订阅携带的凭据字段名
FIELD_BOT_ID = 'bot_id'
FIELD_SECRET = 'secret'

# 单连接被接管的事件名；可能出现在 cmd 或 body 的 event_type 里
EVENT_DISCONNECTED = 'disconnected_event'

# 回复体形态。'stream' 为长连接文档描述的流式回复（单帧 finish=true 即全量文本）；
# 若 L0 表明普通文本回复即可，改成 'text'，_build_reply_frame 会自动切换。
REPLY_MODE = 'stream'

HEARTBEAT_INTERVAL = 30          # 秒，官方建议 30s ping
SUBSCRIBE_TIMEOUT = 15           # 等订阅 ACK 的超时
REPLY_ACK_TIMEOUT = 10           # 等待平台确认回复
STALE_AFTER = 90                 # 超过 90s 未收到任何帧 → degraded
DEDUP_MAX = 512                  # 去重窗口，照抄 feishu.py
REQ_ID_MAX = 512                 # msgid → req_id 映射容量
REPLY_WINDOW = 24 * 60 * 60      # 24 小时回复窗口
IMAGE_MAX_BYTES = 10 * 1024 * 1024
IMAGE_MAX_ITEMS = 10
IMAGE_DOWNLOAD_TIMEOUT = 10

# 入站消息体字段的候选名（不同版本客户端/文档写法不一，逐个兜底）
_MSG_ID_KEYS = ('msgid', 'msg_id', 'message_id')
_CHAT_ID_KEYS = ('chatid', 'chat_id', 'chatId')
_CHAT_TYPE_KEYS = ('chattype', 'chat_type', 'chatType')
_USER_ID_KEYS = ('userid', 'user_id', 'userId', 'open_id')

# 群聊正文里的 @ 前缀，例如 "@Bumi 助手 挥个手"
_AT_PREFIX_RE = re.compile(r'^\s*@[^\s]+\s*')


def _first(d: dict, keys, default=''):
    """从 dict 里按候选键取第一个非空值。"""
    if not isinstance(d, dict):
        return default
    for key in keys:
        value = d.get(key)
        if value not in (None, '', [], {}):
            return value
    return default


def _frame_cmd(frame: dict) -> str:
    """命令名可能落在 cmd、command、body.cmd 或 msgtype 上。"""
    if not isinstance(frame, dict):
        return ''
    cmd = frame.get('cmd') or frame.get('command') or ''
    if not cmd and isinstance(frame.get('body'), dict):
        cmd = frame['body'].get('cmd') or frame['body'].get('command') or ''
    return str(cmd or '')


def _frame_req_id(frame: dict) -> str:
    headers = frame.get('headers') if isinstance(frame, dict) else None
    if isinstance(headers, dict):
        return str(headers.get('req_id') or headers.get('reqId') or '')
    return ''


def _frame_errcode(frame: dict) -> int | None:
    """ACK 帧才有 errcode；普通回调返回 None。"""
    if not isinstance(frame, dict):
        return None
    code = frame.get('errcode')
    if code is None and isinstance(frame.get('body'), dict):
        code = frame['body'].get('errcode')
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


class WecomError(RuntimeError):
    """企业微信长连接返回了非 0 errcode，或凭据/连接不可用。"""


def _image_type(data: bytes) -> tuple[str, str]:
    """Return the actual format, rather than trusting a URL or file extension."""
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg', '.jpg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png', '.png'
    raise WecomError('WeCom images must be JPEG or PNG.')


def _decrypt_image(data: bytes, aeskey: str) -> bytes:
    """WeCom's AES-256-CBC uses PKCS#7 padding to a 32-byte multiple."""
    try:
        encoded = aeskey.encode('ascii')
        key = base64.b64decode(encoded + b'=' * (-len(encoded) % 4), validate=True)
    except (AttributeError, UnicodeEncodeError, ValueError, binascii.Error) as e:
        raise WecomError('Invalid WeCom image decryption key.') from e
    if len(key) != 32 or not data or len(data) % AES.block_size:
        raise WecomError('Invalid WeCom image key or encrypted data.')
    plain = AES.new(key, AES.MODE_CBC, iv=key[:16]).decrypt(data)
    padding = plain[-1]
    if not 1 <= padding <= 32 or padding > len(plain) or plain[-padding:] != bytes([padding]) * padding:
        raise WecomError('WeCom image decryption failed.')
    return plain[:-padding]


class WecomAdapter(ChannelAdapter):
    """企业微信智能机器人（API 模式 · 长连接）适配器。"""

    SUPPORTED_FILE_KINDS = (KIND_IMAGE,)
    SUPPORTS_BOT_TO_BOT = False
    USES_TRUSTED_BOTS = False

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._recv_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._message_tasks: set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()
        self._reply_locks: dict[str, asyncio.Lock] = {}
        self._reply_lock_users: dict[str, int] = {}
        self._pending_replies: dict[str, asyncio.Future] = {}

        self._last_frame_at = 0.0
        self._taken_over = False
        self._last_error = ''

        # 去重（照抄 feishu.py:575-590 的有界 LRU）
        self._seen_ids: list[str] = []
        self._seen_set: set[str] = set()
        self._duplicate_drops = 0

        # msgid -> (req_id, ts)。用 msgid 而非 chat_id 作 key，才能精确回复原消息。
        self._msgid_to_req: OrderedDict[str, tuple[str, float]] = OrderedDict()

    # ── 生命周期 ────────────────────────────────────────────────────────────

    async def _connect(self) -> tuple[aiohttp.ClientSession, aiohttp.ClientWebSocketResponse]:
        """建立 WS。单独成方法，单测可直接 patch 掉真实网络。"""
        # 与飞书 HTTP 客户端一致，容器配置的 HTTP(S)_PROXY 也适用于长连接。
        session = aiohttp.ClientSession(trust_env=True)
        try:
            ws = await session.ws_connect(WSS_URL, heartbeat=None)
        except Exception:
            await session.close()
            raise
        return session, ws

    async def start(self) -> None:
        """连接 + 订阅 + 前置校验，全部通过才算启动成功。

        adapter.py 的契约要求：凭据无效或连不上必须抛异常，不能「起个任务就返回」，
        否则 manager._start_adapter 的 3 次重试不会生效，status() 会显示假的
        connected（飞书 #125 修过的正是这个问题）。
        """
        bot_id = str(self.config.get(FIELD_BOT_ID, '') or '').strip()
        secret = str(self.config.get(FIELD_SECRET, '') or '').strip()
        if not bot_id or not secret:
            raise WecomError(
                'WeCom bot_id and secret are required. '
                'Create the bot in API mode (long connection) and configure them '
                'in Settings → Channels.'
            )

        self._loop = asyncio.get_running_loop()
        self._taken_over = False
        self._last_error = ''
        self._session, self._ws = await self._connect()

        sub_req_id = uuid.uuid4().hex
        subscribe = {
            'cmd': CMD_SUBSCRIBE,
            'headers': {'req_id': sub_req_id},
            'body': {FIELD_BOT_ID: bot_id, FIELD_SECRET: secret},
        }
        try:
            await self._send_frame(subscribe)
            ack = await asyncio.wait_for(self._receive_frame(), timeout=SUBSCRIBE_TIMEOUT)
        except asyncio.TimeoutError as e:
            await self._close_ws()
            raise WecomError(
                'WeCom subscription timed out. Check outbound access to '
                'wss://openws.work.weixin.qq.com:443 and the bot API mode setting.'
            ) from e
        except Exception as e:
            await self._close_ws()
            raise WecomError(f'WeCom subscription failed: {type(e).__name__}: {e}') from e

        errcode = _frame_errcode(ack)
        if _frame_req_id(ack) != sub_req_id or errcode is None:
            await self._close_ws()
            raise WecomError('WeCom subscription returned an invalid or unmatched ACK.')
        if errcode != 0:
            # 不要把 secret 带进异常文本
            errmsg = ack.get('errmsg', '') if isinstance(ack, dict) else ''
            await self._close_ws()
            raise WecomError(
                f'WeCom subscribe rejected (errcode={errcode}, errmsg={errmsg}). '
                'Check bot_id/secret and that the bot uses "long connection" API mode.'
            )

        self._running = True
        self._last_frame_at = time.time()
        self._recv_task = asyncio.create_task(
            self._recv_loop(), name=f'wecom-recv-{self.channel_id}')
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name=f'wecom-ping-{self.channel_id}')
        print(f'[wecom] adapter started (long connection): {self.channel_id}')

    async def stop(self) -> None:
        """幂等关闭：取消接收、心跳与消息任务，关闭 ws 与 session。"""
        self._running = False
        self._finish_pending_replies()
        tasks = [task for task in (self._recv_task, self._heartbeat_task, *self._message_tasks)
                 if task and task is not asyncio.current_task() and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._recv_task = None
        self._heartbeat_task = None
        self._message_tasks.clear()
        await self._close_ws()
        self._msgid_to_req.clear()
        print(f'[wecom] adapter stopped: {self.channel_id}')

    async def _close_ws(self) -> None:
        ws, session = self._ws, self._session
        self._ws, self._session = None, None
        if ws is not None and not ws.closed:
            try:
                await ws.close()
            except Exception:
                pass
        if session is not None and not session.closed:
            try:
                await session.close()
            except Exception:
                pass

    # ── 帧 IO ───────────────────────────────────────────────────────────────

    async def _send_frame(self, frame: dict) -> None:
        """所有出站帧走同一把锁：心跳与回复可能并发写 ws。"""
        if self._ws is None or self._ws.closed:
            raise WecomError('WeCom long connection is not available. Restart the channel.')
        async with self._send_lock:
            await self._ws.send_json(frame)

    async def _receive_frame(self) -> dict:
        """读一帧文本并解析成 dict。非文本帧（PING/PONG/CLOSE）在此处理。"""
        if self._ws is None:
            raise WecomError('WeCom long connection is not available.')
        while True:
            message = await self._ws.receive()
            mtype = message.type
            if mtype in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                         aiohttp.WSMsgType.CLOSING):
                raise WecomError('WeCom long connection closed by server.')
            if mtype == aiohttp.WSMsgType.ERROR:
                raise WecomError('WeCom WebSocket error.')
            if mtype in (aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG):
                continue
            data = message.data
            if isinstance(data, bytes):
                data = data.decode('utf-8', errors='replace')
            if not data:
                continue
            try:
                frame = json.loads(data)
            except (TypeError, ValueError):
                print(f'[wecom] ignoring non-JSON frame: {str(data)[:120]!r}')
                continue
            self._last_frame_at = time.time()
            return frame if isinstance(frame, dict) else {'body': frame}

    async def _recv_loop(self) -> None:
        try:
            while self._running:
                frame = await self._receive_frame()
                if self._is_message_callback(frame):
                    # 与飞书一样，业务处理不能阻塞收帧；审批提示本身也可能等待回复 ACK。
                    task = asyncio.create_task(self._dispatch_message(frame))
                    self._message_tasks.add(task)
                    task.add_done_callback(self._message_tasks.discard)
                else:
                    await self._handle_frame(frame)
        except asyncio.CancelledError:
            raise
        except WecomError as e:
            if self._running:
                self._last_error = str(e)
                print(f'[wecom] receive loop ended: {e}')
        except Exception as e:
            self._last_error = f'{type(e).__name__}: {e}'
            print(f'[wecom] receive loop error: {self._last_error}')
        finally:
            self._finish_pending_replies()
            if self._running and not self._taken_over:
                # 普通网络断开：标记掉线，让 watchdog / Restart 拉起；不清 req_id 映射
                self._running = False
                print(f'[wecom] connection lost: {self.channel_id}. '
                      'Use Settings → Channels → Restart.')

    async def _heartbeat_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if not self._running:
                    break
                await self._send_frame({'cmd': CMD_PING, 'headers': {'req_id': uuid.uuid4().hex}})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._last_error = f'heartbeat: {type(e).__name__}: {e}'
            print(f'[wecom] {self._last_error}')

    # ── 入站 ────────────────────────────────────────────────────────────────

    def _is_message_callback(self, frame: dict) -> bool:
        cmd = _frame_cmd(frame)
        body = frame.get('body') if isinstance(frame.get('body'), dict) else {}
        return cmd == CMD_MSG_CALLBACK or (
            not cmd and _frame_errcode(frame) is None and self._looks_like_message(body)
        )

    async def _dispatch_message(self, frame: dict) -> None:
        try:
            await self._handle_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f'[wecom] process message failed: {type(e).__name__}: {e}')

    async def _handle_frame(self, frame: dict) -> None:
        """分发一帧：事件回调 / 消息回调 / ACK / 心跳响应。"""
        cmd = _frame_cmd(frame)
        body = frame.get('body') if isinstance(frame.get('body'), dict) else {}

        if self._is_disconnected_event(cmd, body, frame):
            await self._on_taken_over()
            return

        # 订阅 ACK、发送 ACK、pong 等：只更新健康状态，不触发 Agent
        errcode = _frame_errcode(frame)
        if errcode is not None:
            pending = self._pending_replies.get(_frame_req_id(frame))
            if pending is not None and not pending.done():
                pending.set_result(frame)
                return
            if errcode != 0:
                self._last_error = f'errcode={errcode}: {frame.get("errmsg", "")}'
                print(f'[wecom] frame rejected: {self._last_error}')
            return
        if cmd in ('pong', 'ping_ack') or cmd == CMD_PING:
            return

        if cmd == CMD_EVENT_CALLBACK:
            # 一期不处理模板卡片按钮等其他事件
            return

        if self._is_message_callback(frame):
            await self._process_message(frame, body)

    @staticmethod
    def _is_disconnected_event(cmd: str, body: dict, frame: dict) -> bool:
        if EVENT_DISCONNECTED in (cmd or ''):
            return True
        candidates = [
            body.get('event_type'), body.get('eventType'),
            body.get('event'), frame.get('event_type'),
        ]
        nested = body.get('event') if isinstance(body.get('event'), dict) else {}
        candidates.append(nested.get('type') or nested.get('event_type') or nested.get('eventtype'))
        for value in candidates:
            if isinstance(value, str) and EVENT_DISCONNECTED in value:
                return True
            if isinstance(value, dict) and EVENT_DISCONNECTED in str(value):
                return True
        return False

    @staticmethod
    def _looks_like_message(body: dict) -> bool:
        """cmd 缺失但 body 里明显是消息时也能识别（不同文档版本写法不一）。"""
        if not isinstance(body, dict):
            return False
        return bool(_first(body, _MSG_ID_KEYS) or
                    body.get('text') or body.get('msgtype') or body.get('msg_type'))

    async def _on_taken_over(self) -> None:
        """同一 BotID 只能有一条长连接。被接管后必须停重连，否则两边无限互踢。"""
        self._taken_over = True
        self._running = False
        self._last_error = (
            'long connection taken over by another process for the same Bot ID; '
            'automatic reconnect disabled'
        )
        print(f'[wecom] {self.channel_id}: {self._last_error}. '
              'Stop the other process, then Restart this channel.')
        for task in (self._heartbeat_task,):
            if task and not task.done():
                task.cancel()
        self._finish_pending_replies()
        await self._close_ws()

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
        """把企微回调帧归一化成构造 InboundMessage 所需的字段。

        独立成纯函数，便于单测逐项核对字段名；L0 帧样本回来后只改这里。
        """
        # 部分版本把消息内容再嵌一层 message / msg
        inner = body
        for key in ('message', 'msg'):
            candidate = body.get(key)
            if isinstance(candidate, dict):
                inner = candidate
                break

        sender = inner.get('from') if isinstance(inner.get('from'), dict) else {}
        if not sender:
            sender = body.get('from') if isinstance(body.get('from'), dict) else {}
        user_id = str(_first(sender, _USER_ID_KEYS) or
                      _first(inner, ('from_userid', 'from_user', 'sender_id')) or
                      _first(body, ('from_userid', 'from_user', 'sender_id')) or '')

        msg_id = str(_first(inner, _MSG_ID_KEYS) or _first(body, _MSG_ID_KEYS) or
                     _first(frame, _MSG_ID_KEYS) or '')
        chat_type_raw = str(_first(inner, _CHAT_TYPE_KEYS) or
                            _first(body, _CHAT_TYPE_KEYS) or 'single').lower()
        chat_type = 'group' if chat_type_raw in ('group', '2', 'chat') else 'p2p'

        # 单聊回调不返回 chatid，用发送者 userid 作会话标识
        chat_id = str(_first(inner, _CHAT_ID_KEYS) or
                      _first(body, _CHAT_ID_KEYS) or user_id)

        msgtype = str(inner.get('msgtype') or inner.get('msg_type') or
                      body.get('msgtype') or body.get('msg_type') or 'text').lower()
        text = ''
        text_obj = inner.get('text') or body.get('text')
        if isinstance(text_obj, dict):
            text = str(text_obj.get('content') or text_obj.get('text') or '')
        elif isinstance(text_obj, str):
            text = text_obj
        if not text and isinstance(inner.get('content'), str):
            text = inner['content']
        image_items = []
        if msgtype == 'image':
            image_items.append(inner.get('image') or body.get('image') or {})
        mixed = inner.get('mixed') or body.get('mixed') or {}
        items = mixed.get('msg_item', []) if isinstance(mixed, dict) else []
        if not isinstance(items, list):
            items = []
        if msgtype == 'mixed':
            image_items.extend(item.get('image') or {} for item in items
                               if isinstance(item, dict) and item.get('msgtype') == 'image')
        # mixed 消息：拼接所有 text 片段
        if not text and msgtype == 'mixed':
            parts = []
            for item in items:
                if isinstance(item, dict) and item.get('msgtype') == 'text':
                    parts.append(str((item.get('text') or {}).get('content', '')))
            text = ''.join(parts)
        # voice 已由平台转成文字
        if not text and msgtype == 'voice':
            voice = inner.get('voice') or {}
            if isinstance(voice, dict):
                text = str(voice.get('content') or '')
        if not text:
            text = '[图片]' if image_items else '[收到不支持的非文本消息]'

        return {
            'message_id': msg_id,
            'user_id': user_id,
            'chat_id': chat_id,
            'chat_type': chat_type,
            'text': text,
            'msgtype': msgtype,
            'images': image_items,
        }

    async def _download_image(self, image: dict):
        """Download and decrypt one WeCom image; never log its signed URL or key."""
        from channel import store

        if not isinstance(image, dict):
            raise WecomError('Invalid WeCom image metadata.')
        url, aeskey = image.get('url'), image.get('aeskey')
        if not isinstance(url, str) or not isinstance(aeskey, str) or not aeskey:
            raise WecomError('WeCom image is missing its URL or decryption key.')
        if self._session is None or self._session.closed:
            raise WecomError('WeCom image download session is unavailable.')

        try:
            for _ in range(4):
                parsed = urlsplit(url)
                if (parsed.scheme != 'https' or not parsed.hostname or
                        parsed.username is not None or parsed.password is not None):
                    raise WecomError('WeCom image download requires an HTTPS URL.')
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
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

        data = _decrypt_image(bytes(encrypted), aeskey)
        if not data or len(data) > IMAGE_MAX_BYTES:
            raise WecomError('WeCom image is empty or exceeds the 10 MiB limit.')
        mime, ext = _image_type(data)
        return store.save_bytes(self.channel_id, data, kind=KIND_IMAGE,
                                name=f'image{ext}', mime=mime, fallback_ext=ext)

    async def _prepare_images(self, msg: InboundMessage, images: list) -> None:
        """Acknowledge after ACL, then materialize the authorized message's media."""
        try:
            await self._send_progress(msg.message_id)
        except Exception as e:
            # The connection or platform may reject the placeholder; still pass
            # the message onward and let the normal reply path report failures.
            print(f'[wecom] image progress reply failed: {type(e).__name__}')
        failures = 0
        for image in images[:IMAGE_MAX_ITEMS]:
            try:
                msg.attachments.append(await self._download_image(image))
            except Exception as e:
                failures += 1
                print(f'[wecom] image download failed: {type(e).__name__}')
        failures += max(0, len(images) - IMAGE_MAX_ITEMS)
        if failures:
            msg.text += f' [图片处理失败或超限：{failures} 张]'

    async def _process_message(self, frame: dict, body: dict) -> None:
        parsed = self._extract_inbound(frame, body)
        req_id = _frame_req_id(frame)
        msg_id = parsed['message_id'] or uuid.uuid4().hex
        if self._is_duplicate(msg_id):
            return

        if req_id:
            self._msgid_to_req[msg_id] = (req_id, time.time())
            while len(self._msgid_to_req) > REQ_ID_MAX:
                self._msgid_to_req.popitem(last=False)
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
        if parsed['images']:
            msg.prepare = lambda: self._prepare_images(msg, parsed['images'])
        try:
            await self._on_message(msg)
        except Exception as e:
            print(f'[wecom] deliver inbound message failed: {type(e).__name__}: {e}')

    # ── 出站 ────────────────────────────────────────────────────────────────

    def _lookup_req_id(self, msg: OutboundMessage) -> str:
        # 方案 B：manager.send_to_channel 已把触发事件的 message_id 透传进来
        source_id = str(getattr(msg, 'source_message_id', '') or '')
        entry = self._msgid_to_req.get(source_id) if source_id else None
        if entry:
            req_id, ts = entry
            if time.time() - ts > REPLY_WINDOW:
                raise WecomError(
                    'Reply window expired (24h). Re-trigger from a current '
                    'message instead of reusing an old source_message_id.'
                )
            return req_id
        raise WecomError(
            'No reply context for this message: the inbound req_id is unknown or '
            'expired. The user must send a new message first.'
        )

    def _build_reply_frame(self, req_id: str, msg: OutboundMessage, *,
                           finish: bool = True, image_items: list | None = None) -> dict:
        if REPLY_MODE == 'stream':
            body = {
                'msgtype': 'stream',
                'stream': {
                    'id': str(getattr(msg, 'source_message_id', '') or uuid.uuid4().hex),
                    'finish': finish,
                    'content': msg.text,
                },
            }
            if image_items:
                body['stream']['msg_item'] = image_items
        else:
            if image_items or not finish:
                raise WecomError('WeCom image replies require stream mode.')
            body = {'msgtype': 'text', 'text': {'content': msg.text}}
        return {'cmd': CMD_RESPOND, 'headers': {'req_id': req_id}, 'body': body}

    async def _send_progress(self, message_id: str) -> None:
        msg = OutboundMessage(chat_id='', text='正在处理图片…', source_message_id=message_id)
        req_id = self._lookup_req_id(msg)
        await self._send_reply(req_id, self._build_reply_frame(req_id, msg, finish=False))

    async def send_message(self, msg: OutboundMessage) -> None:
        """回复原消息。失败必须抛异常，让上层如实回报给 LLM / 用户。"""
        if not self._running or self._ws is None or self._ws.closed:
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
        req_id = self._lookup_req_id(msg)

        def read_image(path: str) -> bytes:
            with open(path, 'rb') as source:
                return source.read(IMAGE_MAX_BYTES + 1)

        images = []
        for att in msg.files:
            images.append(await asyncio.to_thread(read_image, att.path))
        if msg.image_bytes is not None:
            images.append(msg.image_bytes)
        items = []
        for data in images:
            if not data or len(data) > IMAGE_MAX_BYTES:
                raise WecomError('WeCom image is empty or exceeds the 10 MiB limit.')
            _image_type(data)
            items.append({'msgtype': 'image', 'image': {
                'base64': base64.b64encode(data).decode('ascii'),
                'md5': hashlib.md5(data, usedforsecurity=False).hexdigest(),
            }})
        captions = [att.caption for att in msg.files if att.caption]
        if msg.image_caption:
            captions.append(msg.image_caption)
        content = '\n'.join(part for part in [msg.text, *captions] if part)
        reply = OutboundMessage(chat_id=msg.chat_id, text=content,
                                source_message_id=msg.source_message_id)
        await self._send_reply(req_id, self._build_reply_frame(req_id, reply, image_items=items))

    async def _send_reply(self, req_id: str, frame: dict) -> None:
        if not self._running or self._ws is None or self._ws.closed:
            raise WecomError('Cannot send: WeCom adapter is not connected.')
        lock = self._reply_locks.setdefault(req_id, asyncio.Lock())
        self._reply_lock_users[req_id] = self._reply_lock_users.get(req_id, 0) + 1
        try:
            # 同一 req_id 的多次回复串行等待回执，避免回执关联到错误的发送。
            async with lock:
                if not self._running:
                    raise WecomError('Cannot send: WeCom adapter is not connected.')
                future = asyncio.get_running_loop().create_future()
                self._pending_replies[req_id] = future
                try:
                    await self._send_frame(frame)
                    try:
                        ack = await asyncio.wait_for(future, timeout=REPLY_ACK_TIMEOUT)
                    except asyncio.TimeoutError as e:
                        # 超时后的迟到 ACK 无法与同一 req_id 的下一次发送区分。
                        # 关闭旧连接，让 watchdog 建立新连接，避免误报下一条发送成功。
                        self._running = False
                        self._last_error = 'WeCom reply ACK timed out.'
                        self._finish_pending_replies()
                        await self._close_ws()
                        raise WecomError(
                            'WeCom reply ACK timed out; delivery is unknown. '
                            'Check the chat before retrying.'
                        ) from e
                    if ack is None:
                        raise WecomError('WeCom connection closed before the reply ACK.')
                    errcode = _frame_errcode(ack)
                    if errcode != 0:
                        raise WecomError(
                            f'WeCom reply rejected (errcode={errcode}, '
                            f'errmsg={ack.get("errmsg", "")}).'
                        )
                except asyncio.CancelledError:
                    if self._running:
                        # 发送可能已到达平台，旧连接上的迟到 ACK 不能用于下一次回复。
                        self._running = False
                        self._last_error = 'WeCom reply cancelled before its ACK; delivery is unknown.'
                        self._finish_pending_replies()
                        await self._close_ws()
                    raise
                finally:
                    if self._pending_replies.get(req_id) is future:
                        self._pending_replies.pop(req_id, None)
        finally:
            remaining = self._reply_lock_users[req_id] - 1
            if remaining:
                self._reply_lock_users[req_id] = remaining
            else:
                self._reply_lock_users.pop(req_id, None)
                self._reply_locks.pop(req_id, None)

    def _finish_pending_replies(self) -> None:
        for future in self._pending_replies.values():
            if not future.done():
                future.set_result(None)

    # ── 健康状态 ────────────────────────────────────────────────────────────

    def should_auto_reconnect(self) -> bool:
        return not self._taken_over

    async def health_check(self) -> tuple[bool, str]:
        """真实可观测的连通性，不看「start() 有没有抛异常」。"""
        if self._taken_over:
            return False, self._last_error or 'long connection taken over'
        if not self._running:
            return False, self._last_error or 'adapter not running'
        if self._ws is None or self._ws.closed:
            return False, 'WebSocket is closed — restart the channel'
        if self._recv_task is None or self._recv_task.done():
            return False, 'receive task is dead — restart the channel'
        if self._last_frame_at and time.time() - self._last_frame_at > STALE_AFTER:
            return False, 'heartbeat stalled: no frame received recently'
        return True, ''

    def status(self) -> str:
        """connected / degraded / disconnected 三态，供前端轮询。"""
        if self._taken_over or not self._running:
            return 'disconnected'
        if self._ws is None or self._ws.closed:
            return 'disconnected'
        if self._recv_task is None or self._recv_task.done():
            return 'disconnected'
        if self._last_frame_at and time.time() - self._last_frame_at > STALE_AFTER:
            return 'degraded'
        return 'connected'
