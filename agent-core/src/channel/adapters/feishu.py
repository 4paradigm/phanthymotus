"""
channel/adapters/feishu.py — Feishu (Lark) adapter。

**接收**用 lark-oapi SDK 的 WebSocket 长连接（出网连接，无需公网 IP / webhook），
**所有 REST 调用**（发消息、上传/下载资源、健康探测）走 aiohttp 直连开放平台，
不经过 SDK 的同步客户端——这样错误码可以统一处理，也不用为每个调用开线程池。

Requires: pip install lark-oapi
Config: {app_id, app_secret, domain?}

Required Feishu permissions:
- im:message                — 接收消息
- im:message:send_as_bot    — 以机器人身份发消息（只发文本时够了）
- im:resource               — 上传/下载图片与文件（收发附件必需；仅发送也可用
                              im:resource:upload。缺它时文本能发出去、附件一律 99991672）
- im:chat:readonly          — 列出会话（可选）

Event subscription:
- Event: im.message.receive_v1
- Mode: 长连接 (WebSocket long connection)
"""

import asyncio
import json
import os
import threading
import time

import aiohttp

from channel.adapter import (
    Attachment, ChannelAdapter, InboundMessage, OnMessageCallback, OutboundMessage,
    PartialSendError, KIND_AUDIO, KIND_FILE, KIND_IMAGE, KIND_VIDEO,
)

# Common Feishu error codes and actionable messages
_FEISHU_ERROR_HINTS = {
    10003: 'Invalid app_id. Check your Feishu app credentials in Channel settings.',
    10014: 'Invalid app_secret. Check your Feishu app credentials in Channel settings.',
    99991663: 'Tenant token invalid. The adapter will attempt to reconnect automatically. If this persists, restart the channel.',
    99991668: 'Tenant token expired. Tokens auto-refresh; if this persists, check app_id/app_secret in Channel settings.',
    99991672: 'Permission denied. Grant the required permission in Feishu Developer Console: https://open.feishu.cn/app/{app_id}/auth',
    230001: 'Bot not in this chat. Add the bot to the chat first, or the user needs to message the bot directly.',
    230002: 'Bot has been removed from chat. Re-add the bot.',
    230006: 'Message send failed: bot not activated. Publish your app version in Feishu Developer Console.',
    230014: 'Message too long. Maximum 4096 characters.',
}

_DEFAULT_DOMAIN = 'https://open.feishu.cn'

# 单条文本上限 4096 字符，留些余量分片
_TEXT_CHUNK = 3500

# 探测结果缓存时长（秒）——status() 可能被前端高频轮询
_PROBE_TTL = 15

# 去重窗口：飞书可能重投递同一事件
_DEDUP_MAX = 512

# 上传时的 file_type（飞书只认这几种，其余用 stream）
_FILE_TYPE_BY_EXT = {
    '.opus': 'opus', '.mp4': 'mp4', '.pdf': 'pdf',
    '.doc': 'doc', '.docx': 'doc', '.xls': 'xls', '.xlsx': 'xls',
    '.ppt': 'ppt', '.pptx': 'ppt',
}

# SDK 把事件循环存在模块级变量里，多个飞书 channel 同时启动会互相覆盖
_ws_loop_lock = threading.Lock()


class FeishuError(RuntimeError):
    """飞书 API 返回了非 0 的 code。"""

    def __init__(self, code: int, msg: str, hint: str = ''):
        self.code = code
        self.msg = msg
        text = f'[feishu] API error (code={code}): {msg}'
        if hint:
            text += f'\n  → {hint}'
        super().__init__(text)


class FeishuAdapter(ChannelAdapter):
    """Feishu/Lark adapter using SDK WebSocket long connection."""

    SUPPORTED_FILE_KINDS = (KIND_IMAGE, KIND_VIDEO, KIND_AUDIO, KIND_FILE)

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._client = None                       # lark ws client
        self._thread: threading.Thread | None = None
        self._sdk_loop: asyncio.AbstractEventLoop | None = None
        self._loop: asyncio.AbstractEventLoop | None = None   # main app loop
        self._token = ''
        self._token_expire = 0.0
        self._token_lock = asyncio.Lock()
        self._probe_ok = False
        self._probe_err = ''
        self._probe_ts = 0.0
        self._last_event_ts = 0.0
        self._seen_ids: list[str] = []
        self._seen_set: set[str] = set()

    # ── 基础设施：token / REST ────────────────────────────────────────────────

    @property
    def _domain(self) -> str:
        return (self.config.get('domain') or _DEFAULT_DOMAIN).rstrip('/')

    async def _tenant_token(self, force: bool = False) -> str:
        async with self._token_lock:
            if not force and self._token and time.time() < self._token_expire:
                return self._token
            payload = {
                'app_id': self.config.get('app_id', ''),
                'app_secret': self.config.get('app_secret', ''),
            }
            url = f'{self._domain}/open-apis/auth/v3/tenant_access_token/internal'
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(url, json=payload) as resp:
                    data = await resp.json(content_type=None)
            code = data.get('code', -1)
            if code != 0:
                raise FeishuError(code, data.get('msg', 'token request failed'),
                                  self._hint(code))
            self._token = data.get('tenant_access_token', '')
            # 官方 expire 单位为秒，提前 60s 过期
            self._token_expire = time.time() + max(60, int(data.get('expire', 7200))) - 60
            return self._token

    def _hint(self, code: int) -> str:
        hint = _FEISHU_ERROR_HINTS.get(code, '')
        return hint.format(app_id=self.config.get('app_id', '')) if hint else ''

    async def _request(self, method: str, path: str, *, json_body: dict | None = None,
                       params: dict | None = None, data=None,
                       raw: bool = False, retry_auth: bool = True):
        """调用开放平台 REST。raw=True 时返回 (bytes, headers)，否则返回 data 字段。"""
        token = await self._tenant_token()
        url = f'{self._domain}{path}'
        headers = {'Authorization': f'Bearer {token}'}
        timeout = aiohttp.ClientTimeout(total=120 if (data is not None or raw) else 20)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.request(method, url, headers=headers, json=json_body,
                                     params=params, data=data) as resp:
                    ctype = resp.headers.get('Content-Type', '')
                    if raw and 'application/json' not in ctype:
                        return await resp.read(), resp.headers
                    body = await resp.json(content_type=None)
        except FeishuError:
            raise
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            raise RuntimeError(
                f'[feishu] network error calling {path}: {e}\n'
                f'  → Cause: Feishu API unreachable or rate-limited.\n'
                f'  → Solution: check connectivity, then retry.'
            )

        code = body.get('code', -1)
        if code != 0:
            # token 失效 → 强制刷新后重试一次
            if retry_auth and code in (99991663, 99991668, 99991661):
                await self._tenant_token(force=True)
                return await self._request(method, path, json_body=json_body, params=params,
                                          data=data, raw=raw, retry_auth=False)
            raise FeishuError(code, body.get('msg', ''), self._hint(code))
        return body.get('data', {})

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        app_id = self.config.get('app_id', '')
        app_secret = self.config.get('app_secret', '')
        if not app_id or not app_secret:
            raise ValueError(
                'Feishu app_id and app_secret are required. '
                'Configure them in Settings → Channels.'
            )

        import lark_oapi as lark

        self._loop = asyncio.get_running_loop()

        # 凭据前置校验：起线程前先确认 app_id/app_secret 与网络可用。
        # 否则「起个线程就返回成功」，凭据错了状态依然显示 connected。
        await self._tenant_token(force=True)
        ok, err = await self._probe(force=True)
        if not ok:
            raise RuntimeError(f'Feishu credential check failed: {err}')

        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._handle_message_event) \
            .build()

        self._client = lark.ws.Client(
            app_id=app_id,
            app_secret=app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
            auto_reconnect=True,
        )

        self._running = True
        self._thread = threading.Thread(target=self._thread_target, daemon=True,
                                        name=f'feishu-ws-{self.channel_id}')
        self._thread.start()
        print(f'[feishu] adapter started (WebSocket mode): {self.channel_id}')

    def _thread_target(self):
        """Run lark SDK client.start() in a dedicated thread.

        SDK's start() does:
        1. _connect() — establishes WS, starts _receive_message_loop task
        2. _ping_loop() — periodic keepalive pings
        3. _select() — keeps event loop alive forever

        _receive_message_loop has built-in auto_reconnect on disconnect.

        SDK 把事件循环放在模块级 `lark_oapi.ws.client.loop`（import 时捕获主 uvloop）。
        每个 adapter 换成自己的 loop，并只停自己的那个——共用模块变量时，
        第二个 channel 启动会覆盖它，停第一个就会停错 loop。
        """
        import lark_oapi.ws.client as ws_mod
        new_loop = asyncio.new_event_loop()
        self._sdk_loop = new_loop
        asyncio.set_event_loop(new_loop)
        with _ws_loop_lock:
            ws_mod.loop = new_loop
        try:
            self._client.start()
        except Exception as e:
            err_msg = str(e)
            if 'invalid' in err_msg.lower() and ('app_id' in err_msg.lower() or 'secret' in err_msg.lower()):
                print('[feishu] Connection failed: invalid app credentials. '
                      'Check app_id and app_secret in Channel settings.')
            else:
                print(f'[feishu] WebSocket connection error: {e}')
        finally:
            self._running = False
            print(f'[feishu] ws thread exited: {self.channel_id}')

    async def stop(self) -> None:
        self._running = False

        # 停自己的 SDK loop 以打断 _select()
        if self._sdk_loop and self._sdk_loop.is_running():
            try:
                self._sdk_loop.call_soon_threadsafe(self._sdk_loop.stop)
            except RuntimeError:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self._client = None
        self._thread = None
        self._sdk_loop = None
        print(f'[feishu] adapter stopped: {self.channel_id}')

    # ── 健康状态 ─────────────────────────────────────────────────────────────

    async def _probe(self, force: bool = False) -> tuple[bool, str]:
        """轻量 API 探测（bot info），结果缓存 _PROBE_TTL 秒。

        探测同时覆盖：凭据是否有效（换 token）+ 网络是否通 + 应用是否已发布。
        应用没开 bot info 权限时不算故障——token 换到了就说明凭据没问题。
        """
        if not force and time.time() - self._probe_ts < _PROBE_TTL:
            return self._probe_ok, self._probe_err
        try:
            await self._request('GET', '/open-apis/bot/v3/info')
            self._probe_ok, self._probe_err = True, ''
        except FeishuError as e:
            if e.code == 99991672:
                self._probe_ok, self._probe_err = True, ''
            else:
                self._probe_ok, self._probe_err = False, str(e)
        except Exception as e:
            self._probe_ok, self._probe_err = False, str(e)
        self._probe_ts = time.time()
        return self._probe_ok, self._probe_err

    async def health_check(self) -> tuple[bool, str]:
        if not self._running:
            return False, 'adapter not running'
        if self._thread is None or not self._thread.is_alive():
            return False, 'WebSocket thread is dead — restart the channel'
        return await self._probe()

    def status(self) -> str:
        """同步状态：线程死了就是 disconnected；线程活着但最近探测失败为 degraded。

        探测本身是异步的（health_check），这里只读缓存结果，避免前端轮询打 API。
        """
        if not self._running or self._thread is None or not self._thread.is_alive():
            return 'disconnected'
        if self._probe_ts and not self._probe_ok:
            return 'degraded'
        return 'connected'

    # ── 发送 ─────────────────────────────────────────────────────────────────

    async def send_message(self, msg: OutboundMessage) -> None:
        """Send text and/or attachments via Feishu Open API."""
        if not self._running:
            raise RuntimeError(
                '[feishu] Cannot send: adapter not running. '
                'Cause: the adapter failed to start or has been stopped. '
                'Solution: check app_id/app_secret in Channel settings and restart the channel.'
            )

        files = list(msg.files)
        if msg.image_bytes:
            # 兼容旧调用方：字节图片先落盘再走统一路径
            from channel import store
            files.append(store.save_bytes(self.channel_id, msg.image_bytes,
                                          kind=KIND_IMAGE, name='image.jpg',
                                          mime='image/jpeg', fallback_ext='.jpg'))

        if msg.text:
            for chunk in _chunks(msg.text, _TEXT_CHUNK):
                await self._send_raw(msg.chat_id, 'text', {'text': chunk})

        if not files:
            return

        # 逐个附件独立汇报成败：一个文件失败不该让上层以为整条消息（含已送达的文本）失败
        sent = ['文本'] if msg.text else []
        failures = []
        for att in files:
            try:
                await self._send_attachment(msg.chat_id, att)
                sent.append(att.name or att.path)
            except Exception as e:
                print(f'[feishu] send attachment failed ({att.name or att.path}): {e}')
                failures.append(f'- {att.name or att.path}: {e}')
        if failures:
            raise PartialSendError(sent, failures)

    async def _send_raw(self, chat_id: str, msg_type: str, content: dict) -> None:
        await self._request(
            'POST', '/open-apis/im/v1/messages',
            params={'receive_id_type': 'chat_id'},
            json_body={
                'receive_id': chat_id,
                'msg_type': msg_type,
                'content': json.dumps(content, ensure_ascii=False),
            },
        )

    async def _send_attachment(self, chat_id: str, att: Attachment) -> None:
        if att.kind == KIND_IMAGE:
            image_key = await self._upload_image(att.path)
            await self._send_raw(chat_id, 'image', {'image_key': image_key})
        elif att.kind == KIND_VIDEO:
            file_key = await self._upload_file(att.path, att.name)
            await self._send_raw(chat_id, 'media', {'file_key': file_key})
        elif att.kind == KIND_AUDIO and att.path.lower().endswith('.opus'):
            file_key = await self._upload_file(att.path, att.name)
            await self._send_raw(chat_id, 'audio', {'file_key': file_key})
        else:
            file_key = await self._upload_file(att.path, att.name)
            await self._send_raw(chat_id, 'file', {'file_key': file_key})
        if att.caption:
            await self._send_raw(chat_id, 'text', {'text': att.caption})

    async def _upload_image(self, path: str) -> str:
        with open(path, 'rb') as f:
            payload = f.read()
        form = aiohttp.FormData()
        form.add_field('image_type', 'message')
        form.add_field('image', payload, filename=os.path.basename(path),
                       content_type='application/octet-stream')
        data = await self._upload('/open-apis/im/v1/images', form)
        return data.get('image_key', '')

    async def _upload_file(self, path: str, name: str = '') -> str:
        ext = os.path.splitext(path)[1].lower()
        file_type = _FILE_TYPE_BY_EXT.get(ext, 'stream')
        fname = name or os.path.basename(path)
        with open(path, 'rb') as f:
            payload = f.read()
        form = aiohttp.FormData()
        form.add_field('file_type', file_type)
        form.add_field('file_name', fname)
        form.add_field('file', payload, filename=fname,
                       content_type='application/octet-stream')
        data = await self._upload('/open-apis/im/v1/files', form)
        return data.get('file_key', '')

    async def _upload(self, path: str, form: 'aiohttp.FormData') -> dict:
        """上传资源。缺权限是这里最常见的失败，且与「发消息」的权限是两码事 ——
        只有 im:message:send_as_bot 时文本照发、附件全 99991672，所以要说清是上传权限。"""
        try:
            return await self._request('POST', path, data=form)
        except FeishuError as e:
            if e.code == 99991672:
                raise FeishuError(
                    e.code, e.msg,
                    '上传资源需要 im:resource（或 im:resource:upload）权限，与发送文本的 '
                    'im:message:send_as_bot 是两个权限。在飞书开发者后台开通后**发布新版本**：'
                    f'https://open.feishu.cn/app/{self.config.get("app_id", "")}/auth'
                    '?q=im:resource:upload,im:resource'
                ) from None
            raise

    # ── 接收 ─────────────────────────────────────────────────────────────────

    def _handle_message_event(self, data):
        """Handle im.message.receive_v1 (runs in the SDK thread).

        SDK 线程里只做纯数据提取，下载/落盘/回调都丢到主 loop 的协程里完成。
        """
        try:
            event = data.event
            message = event.message
            sender = event.sender

            if sender.sender_type == 'app':
                return  # 跳过机器人自己的消息

            raw = {
                'message_id': getattr(message, 'message_id', '') or '',
                'message_type': getattr(message, 'message_type', '') or '',
                'content': getattr(message, 'content', '') or '',
                'chat_id': getattr(message, 'chat_id', '') or '',
                'sender_id': (sender.sender_id.open_id or sender.sender_id.user_id or ''),
            }

            self._last_event_ts = time.time()

            if self._loop is None:
                print('[feishu] event dropped: adapter loop not ready')
                return
            asyncio.run_coroutine_threadsafe(self._process_event(raw), self._loop)

        except Exception as e:
            print(f'[feishu] handle message error: {e}')
            print('  If messages are not being received, ensure:')
            print('  1. Event "im.message.receive_v1" is subscribed in Feishu Developer Console')
            print('  2. Subscription mode is set to "长连接" (WebSocket long connection)')
            print('  3. App has "im:message" permission and is published')

    def _is_duplicate(self, message_id: str) -> bool:
        """飞书可能重投递同一事件；用有界 LRU 去重。"""
        if not message_id:
            return False
        if message_id in self._seen_set:
            return True
        self._seen_set.add(message_id)
        self._seen_ids.append(message_id)
        if len(self._seen_ids) > _DEDUP_MAX:
            self._seen_set.discard(self._seen_ids.pop(0))
        return False

    async def _process_event(self, raw: dict) -> None:
        """在主 loop 上解析消息、下载附件、回调 manager。"""
        try:
            if self._is_duplicate(raw['message_id']):
                print(f'[feishu] duplicate message dropped: {raw["message_id"]}')
                return

            text, attachments = await self._parse_content(raw)
            if not text and not attachments:
                return

            msg = InboundMessage(
                platform='feishu',
                channel_id=self.channel_id,
                user_id=raw['sender_id'],
                chat_id=raw['chat_id'],
                display_name=raw['sender_id'],
                text=text,
                message_id=raw['message_id'],
                attachments=attachments,
            )
            await self._on_message(msg)
        except Exception as e:
            print(f'[feishu] process event failed: {e}')

    async def _parse_content(self, raw: dict) -> tuple[str, list[Attachment]]:
        msg_type = raw['message_type']
        try:
            content = json.loads(raw['content'] or '{}')
        except json.JSONDecodeError:
            content = {}

        message_id = raw['message_id']
        attachments: list[Attachment] = []

        if msg_type == 'text':
            return content.get('text', ''), []

        if msg_type in ('image', 'sticker'):
            key = content.get('image_key', '')
            att = await self._download(message_id, key, 'image', KIND_IMAGE,
                                       name='image.jpg', fallback_ext='.jpg')
            if att:
                attachments.append(att)
            return ('[表情]' if msg_type == 'sticker' else '[图片]'), attachments

        if msg_type in ('file', 'media', 'audio'):
            key = content.get('file_key', '')
            name = content.get('file_name', '') or ''
            kind = {'media': KIND_VIDEO, 'audio': KIND_AUDIO}.get(msg_type, KIND_FILE)
            ext = {'media': '.mp4', 'audio': '.opus'}.get(msg_type, '')
            att = await self._download(message_id, key, 'file', kind,
                                       name=name or f'{msg_type}{ext}', fallback_ext=ext)
            if att:
                attachments.append(att)
            label = {'media': '[视频]', 'audio': '[语音]'}.get(msg_type, '[文件]')
            return f'{label} {att.name}' if att else label, attachments

        if msg_type == 'post':
            # 富文本：content = {"title":..,"content":[[{tag:text|img|a,...}]]}
            parts = []
            for block in content.get('content', []) or []:
                for el in block or []:
                    tag = el.get('tag', '')
                    if tag == 'text':
                        parts.append(el.get('text', ''))
                    elif tag == 'a':
                        parts.append(f'{el.get("text", "")}({el.get("href", "")})')
                    elif tag == 'img':
                        att = await self._download(message_id, el.get('image_key', ''),
                                                   'image', KIND_IMAGE,
                                                   name='image.jpg', fallback_ext='.jpg')
                        if att:
                            attachments.append(att)
                            parts.append('[图片]')
                parts.append('\n')
            title = content.get('title', '')
            text = (f'{title}\n' if title else '') + ''.join(parts).strip()
            return text, attachments

        # 其它类型（分享、位置…）：给出可读占位，不静默丢弃
        return f'[{msg_type}]', []

    async def _download(self, message_id: str, key: str, res_type: str,
                        kind: str, *, name: str = '', fallback_ext: str = '') -> Attachment | None:
        """下载消息资源并落盘。res_type: 'image' 用于图片，'file' 用于文件/视频/语音。"""
        if not message_id or not key:
            return None
        try:
            result = await self._request(
                'GET', f'/open-apis/im/v1/messages/{message_id}/resources/{key}',
                params={'type': res_type}, raw=True,
            )
        except Exception as e:
            print(f'[feishu] resource download failed ({key}): {e}\n'
                  f'  → 确认应用已获得 "im:resource" 权限并发布新版本')
            return None

        # raw=True 时返回 (bytes, headers)；若平台回了 JSON（code==0 但无二进制体）则放弃
        if not (isinstance(result, tuple) and isinstance(result[0], (bytes, bytearray))):
            print(f'[feishu] resource download returned no binary data ({key}): {result!r:.200}')
            return None
        payload, headers = result

        from channel import store
        mime = headers.get('Content-Type', '').split(';')[0] if headers else ''
        return store.save_bytes(self.channel_id, bytes(payload), kind=kind,
                                name=name, mime=mime, fallback_ext=fallback_ext)


def _chunks(text: str, size: int):
    """按 size 切分长文本（尽量在换行处断开）。"""
    while text:
        if len(text) <= size:
            yield text
            return
        cut = text.rfind('\n', 0, size)
        if cut < size // 2:
            cut = size
        yield text[:cut]
        text = text[cut:].lstrip('\n')
