"""Robot-independent Fanshi+ canvas card, hosted by Agent Core.

Direct sends await their final result; no background ACP callback is needed.
Credentials are read only from saved canvas settings or a provisioned relay file.
"""
import asyncio
import base64
from collections import OrderedDict, defaultdict, deque
import copy
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp

from fanshi_relay import RelayClient, RelayError


def _reply(status, message, **extra):
    result = {"ok": status in ("accepted", "queued", "sending", "ready", "idle", "configured"),
              "status": status, "message": message, **extra}
    if not result["ok"]:
        result.update(state="error", error=message)
    return result


TOOL_NAME = 'fanshi_message_send'
TOOL = {
    'name': TOOL_NAME,
    'type': 'actuator',
    'multiInstance': False,
    'description': (
        '向预配置的范式+群主动发送文字、单张图片和可选 @ 成员；适用于所有机器人。'
        'send 等待发送结果，图文依次发送两条；只有 accepted 表示全部被平台接受。'
        'partial 表示部分成功，unknown 表示结果不确定：先核对 parts 和群聊，勿整组重发。'
        'result 用 action_id 查询本卡片最近的记录。图片必须使用实际拍照返回且 '
        'Agent Core 可读取的路径。群和 Webhook 由管理员预配置，发送参数不能改群。'
    ),
    'inputSchema': {
        'type': 'object',
        'properties': {
            'action': {'type': 'string', 'enum': ['send', 'result']},
            'text': {'type': 'string', 'description': '要发送的文字'},
            'image_path': {
                'type': 'string', 'format': 'file', 'accept': '.jpg,.jpeg,.png',
                'uploadDir': '/tmp/uploads',
                'description': 'Agent Core 可读取的 PNG/JPEG 路径；文字和图片至少提供一种',
            },
            'at_userid': {'type': 'string', 'description': '可选：一个真实成员 userid'},
            'action_id': {'type': 'string', 'description': 'result 查询使用发送返回的 action_id'},
        },
        'required': ['action'],
        'x-action-params': {
            'send': {'params': ['text', 'image_path', 'at_userid'], 'description': '发送群消息并等待结果'},
            'result': {'params': ['action_id'], 'description': '查询最近发送结果'},
        },
    },
    'configSchema': {
        'type': 'object',
        'properties': {
            'mode': {
                'type': 'string', 'scope': 'shared', 'enum': ['auto', 'direct', 'relay'],
                'default': 'auto',
                'description': 'auto 优先使用已下发的统一服务接入文件，否则使用下方 Webhook',
            },
            'webhook_url': {
                'type': 'string', 'scope': 'shared', 'format': 'password', 'x-sensitive': True,
                'description': '直连模式：完整 HTTPS Webhook；统一服务模式无需填写',
            },
            'group_id': {
                'type': 'string', 'scope': 'shared', 'pattern': '^[0-9]+$', 'x-sensitive': True,
                'description': '直连模式：目标群的数字 ID，禁止留空；统一服务固定群由服务端配置',
            },
        },
    },
}

_IMAGE_ROOTS = (Path('/work/resource/images'), Path('/work/resource/channel_files'), Path('/tmp/uploads'))
_active = {}
_generation = defaultdict(int)
_busy = set()
_locks = {}
_recent = defaultdict(deque)
_jobs = defaultdict(OrderedDict)
_job_relays = {}


def _read_settings():
    import config
    cfg = config.main.get(f'tool_config:channel:{TOOL_NAME}') or {}
    if not isinstance(cfg, dict):
        raise ValueError('配置必须是对象，请重新保存配置。')
    mode = cfg.get('mode', 'auto')
    if mode not in ('auto', 'direct', 'relay'):
        raise ValueError('发送模式无效。')
    if mode != 'direct':
        relay = RelayClient.load()
        if relay is not None:
            return ('relay', relay.identity, relay, 1)
        if mode == 'relay':
            raise ValueError('缺少统一服务接入文件，请先下发设备配置。')
    url, group = cfg.get('webhook_url', ''), cfg.get('group_id', '')
    if not isinstance(url, str) or any(ord(c) < 32 for c in url):
        raise ValueError('请填写完整 HTTPS Webhook 和正整数群 ID。')
    url = url.strip()
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == 'https' and parsed.hostname and not parsed.username
                 and not parsed.password and not parsed.fragment and parsed.port != 0)
    except ValueError:
        valid = False
    if (not valid or isinstance(group, bool)
            or not re.fullmatch(r'[0-9]{1,20}', str(group)) or int(group) <= 0):
        raise ValueError('请填写完整 HTTPS Webhook 和正整数群 ID。')
    group = int(group)
    identity = hashlib.sha256(json.dumps([url, group]).encode()).hexdigest()
    return ('direct', identity, url, group)


def _image_base64(raw_path):
    if not isinstance(raw_path, str) or not raw_path.startswith('/'):
        raise ValueError('图片必须是 Agent Core 可读取的绝对路径。')
    host_prefix = '/opt/phanthy-motus/data/images/'
    if raw_path.startswith(host_prefix):
        raw_path = '/work/resource/images/' + raw_path[len(host_prefix):]
    path = Path(raw_path).resolve(strict=True)
    if not any(path.is_relative_to(root.resolve()) for root in _IMAGE_ROOTS):
        raise ValueError('图片必须位于共享 images、channel_files 或 /tmp/uploads 目录。')
    if not path.is_file() or path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
        raise ValueError('仅支持 PNG/JPEG 图片。')
    with path.open('rb') as stream:
        raw = stream.read(750_001)
    if not (raw.startswith(b'\x89PNG\r\n\x1a\n') or raw.startswith(b'\xff\xd8\xff')):
        raise ValueError('文件内容不是 PNG/JPEG。')
    encoded = base64.b64encode(raw).decode('ascii')
    if len(encoded) >= 1_000_000:
        raise ValueError('图片 Base64 必须小于 1,000,000 字节，请先缩小图片。')
    return encoded


def _payload(group, args):
    text, path, userid = (args.get(k, '') for k in ('text', 'image_path', 'at_userid'))
    if not all(isinstance(v, str) for v in (text, path, userid)):
        raise ValueError('文字、图片路径和 userid 必须为字符串。')
    body = []
    if text.strip():
        body.append({'type': 'TEXT', 'content': text})
    if path:
        body.append({'type': 'IMAGE', 'content': _image_base64(path)})
    if not body:
        raise ValueError('文字和图片至少填写一种。')
    if userid.strip():
        body.append({'type': 'AT', 'atuserids': [userid.strip()], 'atall': False})
    return {'message': {'header': {'toid': [group]}, 'body': body}}


async def _post(url, payload):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), trust_env=True) as session:
        async with session.post(url, json=payload, allow_redirects=False) as response:
            chunks, size = [], 0
            async for chunk in response.content.iter_chunked(16384):
                chunks.append(chunk)
                size += len(chunk)
                if size > 65536:
                    break
            return response.status, b''.join(chunks)


def _current(instance, settings, generation):
    if _generation[instance] != generation or _active.get(instance) != settings[:2]:
        return False
    try:
        return _read_settings()[:2] == settings[:2]
    except (ValueError, OSError, RelayError):
        return False


async def _send_direct(instance, settings, generation, action_id, payload):
    planned = _split_payload(payload)
    parts = []
    for kind, request in planned:
        if not _current(instance, settings, generation):
            outcome = _reply('cancelled', '提交前卡片停止或配置改变，本部分未发送。')
            attempted = False
        else:
            attempted = True
            try:
                outcome = _interpret(*await _post(settings[2], request))
            except (aiohttp.ClientError, OSError, ValueError):
                outcome = _reply('unknown', '网络错误或超时；先核对群聊，勿自动重发。')
        parts.append({**outcome, 'part': kind, 'attempted': attempted})
        _jobs[instance][action_id]['parts'] = copy.deepcopy(parts)
        if outcome['status'] != 'accepted':
            break
    for kind, _ in planned[len(parts):]:
        parts.append({**_reply('not_sent', '前一部分未确认成功，本部分未发送。'),
                      'part': kind, 'attempted': False})
    return _combine_results(parts)


def _remember_result(instance, action_id, result):
    previous = _jobs[instance].get(action_id, {})
    # A concurrent gateway query may already have confirmed delivery. A stale
    # poll or a local timeout must not downgrade that confirmation to unknown.
    if previous.get('status') in ('accepted', 'partial', 'failed', 'cancelled'):
        return copy.deepcopy(previous)
    result = {**result, 'action_id': action_id}
    if action_id in _jobs[instance]:
        _jobs[instance][action_id] = copy.deepcopy(result)
    return result


async def handle(args):
    """Shared by canvas calls and model calls; never accepts credential overrides."""
    instance = args.get('instance_id', '')
    action = args.get('action', 'send')
    if action == 'config' and not instance:
        # Shared canvas settings are applied without an instance_id.
        try:
            settings = _read_settings()
        except (ValueError, OSError, RelayError):
            for running in list(_active):
                _generation[running] += 1
            _active.clear()
            return _reply('failed', '配置无效，请检查 Webhook、群 ID 或统一服务接入文件。', adapter_ok=False)
        for running in list(_active):
            if _active[running] != settings[:2]:
                _active.pop(running, None)
                _generation[running] += 1
        return _reply('configured', '配置已校验，未发送测试消息。', adapter_ok=True)
    if not isinstance(instance, str) or not instance:
        return _reply('failed', '缺少卡片实例 ID，请从画布或已绑定的 Skill 调用。')
    if action == 'stop':
        _active.pop(instance, None)
        _generation[instance] += 1
        return _reply('idle', '已停止新发送，已提交消息无法撤回。', state='idle')
    if action == 'info':
        return _reply('ready' if instance in _active else 'idle', '就绪不代表网络或群投递已验证。',
                      state='running' if instance in _active else 'idle',
                      busy=instance in _busy, topic_in=[], topic_out=[])
    if action == 'result':
        action_id = args.get('action_id', '')
        if not isinstance(action_id, str) or action_id not in _jobs[instance]:
            return _reply('unknown', '记录不存在、已过期或服务已重启；先核对群聊，勿自动重发。')
        cached = copy.deepcopy(_jobs[instance][action_id])
        relay = _job_relays.get((instance, action_id))
        if relay is not None and _jobs[instance][action_id]['status'] in ('queued', 'sending', 'unknown'):
            try:
                # Query the original gateway; never submit again or change gateways.
                result = await asyncio.to_thread(relay.result, action_id)
                return _remember_result(instance, action_id, result)
            except RelayError:
                pass
        return copy.deepcopy(_jobs[instance].get(action_id, cached))
    if action not in ('start', 'config', 'send'):
        return _reply('failed', '不支持的动作。')
    try:
        settings = _read_settings()
    except (ValueError, OSError, RelayError):
        _active.pop(instance, None)
        _generation[instance] += 1
        return _reply('failed', '配置无效：请保存 HTTPS Webhook 和群 ID，或检查统一服务接入文件。')
    if action in ('start', 'config'):
        if _active.get(instance) != settings[:2]:
            _active.pop(instance, None)
            _generation[instance] += 1
        if action == 'start':
            _active[instance] = settings[:2]
        return _reply('ready' if instance in _active else 'configured',
                      '配置已校验，未发送测试消息。',
                      state='running' if instance in _active else 'idle', transport=settings[0])
    if _active.get(instance) != settings[:2]:
        return _reply('failed', '请保存配置并启动画布项目；配置改变后需要重新启动卡片。')
    if instance in _busy:
        return _reply('failed', '上一条消息仍在处理，请先查询结果。')
    generation = _generation[instance]
    _busy.add(instance)
    try:
        # Validate the complete group before posting even its first text part.
        try:
            payload = await asyncio.to_thread(_payload, settings[3], args)
        except ValueError as exc:
            return _reply('failed', str(exc))
        except OSError:
            return _reply('failed', '图片不存在或不可读，请使用 Agent Core 中可访问的实际路径。')
        if not _current(instance, settings, generation):
            return _reply('cancelled', '准备发送期间卡片停止或配置改变，未提交消息。')
        key = settings[:2]
        async with _locks.setdefault(key, asyncio.Lock()):
            if not _current(instance, settings, generation):
                return _reply('cancelled', '等待期间卡片停止或配置改变，未提交消息。')
            if settings[0] == 'direct':
                recent = _recent[key]
                now = time.monotonic()
                while recent and now - recent[0] >= 60:
                    recent.popleft()
                cost = len(_split_payload(payload))
                if len(recent) + cost > 30:
                    return _reply('failed', '达到本进程每目标每分钟 30 次请求限制；图文占两次。')
                recent.extend([now] * cost)
            action_id = 'fanshi_' + uuid4().hex
            _jobs[instance][action_id] = _reply('sending', '正在发送。', action_id=action_id)
            while len(_jobs[instance]) > 32:
                expired, _ = _jobs[instance].popitem(last=False)
                _job_relays.pop((instance, expired), None)
            try:
                if settings[0] == 'relay':
                    _job_relays[(instance, action_id)] = settings[2]
                    loop = asyncio.get_running_loop()
                    # Relay polling runs outside the event loop so canvas stop remains usable.
                    def record_progress(value):
                        job = _jobs[instance].get(action_id)
                        if job and job['status'] in ('queued', 'sending'):
                            job.update(value)
                    def progress(value):
                        loop.call_soon_threadsafe(record_progress, copy.deepcopy(value))
                    result = await asyncio.to_thread(
                        settings[2].run, action_id, payload,
                        lambda: not _current(instance, settings, generation), progress)
                else:
                    result = await _send_direct(instance, settings, generation, action_id, payload)
            except asyncio.CancelledError:
                _generation[instance] += 1
                _remember_result(instance, action_id, _reply('unknown', '调用已中断；先核对群聊，勿重发。',
                                 parts=_jobs[instance][action_id].get('parts', [])))
                raise
            except Exception:
                result = _reply('unknown', '发送结果未确认，请查询并核对群聊，勿自动重发。',
                                parts=_jobs[instance][action_id].get('parts', []))
            return _remember_result(instance, action_id, result)
    finally:
        _busy.discard(instance)

def _interpret(status, raw):
    if status >= 500 or status == 408:
        return _reply("unknown", "平台响应不确定，请先核对群聊，不要自动重发。")
    if not 200 <= status < 300:
        return _reply("failed", f"HTTP {status}，请检查平台权限、地址或限流。")
    try:
        if len(raw) > 65_536:
            raise ValueError()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError()
        codes = [data[key] for key in ("errcode", "errorcode") if key in data]
        if not codes or any(type(code) is not int for code in codes) or len(set(codes)) != 1:
            raise ValueError()
        detail = data.get("data")
        if detail is not None and not isinstance(detail, dict):
            raise ValueError()
    except (ValueError, UnicodeError):
        return _reply("unknown", "平台响应格式异常，请先核对群内是否收到。")
    failures = (detail or {}).get("fail")
    if codes[0] != 0 or failures:
        # Keep numeric per-group diagnostics; never echo a request or Webhook.
        failed_groups = {}
        if isinstance(failures, dict):
            for group, value in list(failures.items())[:32]:
                if re.fullmatch(r"[0-9]{1,20}", str(group)):
                    if type(value) is int:
                        failed_groups[str(group)] = value
                    elif isinstance(value, str) and re.fullmatch(r"-?[0-9]{1,12}", value):
                        failed_groups[str(group)] = int(value)
                    else:
                        failed_groups[str(group)] = "detail_unavailable"
        message = ("平台拒绝请求，请结合 platform_code 排查。"
                   if codes[0] != 0 else
                   "平台返回群投递失败，请查看 failed_groups 错误码；不能仅凭此认定为权限问题。")
        return _reply("failed", message, platform_code=codes[0],
                      has_group_failures=bool(failures), failed_groups=failed_groups)
    return _reply("accepted", "平台接受消息且无失败群；最终到达及图片展示以群内为准。")

def _split_payload(payload):
    """Plan requests before sending; no mixed request or fallback retry."""
    body = payload["message"]["body"]
    has_text = any(item["type"] == "TEXT" for item in body)
    has_image = any(item["type"] == "IMAGE" for item in body)
    if not (has_text and has_image):
        return [("image" if has_image else "text", payload)]
    requests = []
    for kind, items in (
        ("text", [item for item in body if item["type"] != "IMAGE"]),
        ("image", [item for item in body if item["type"] == "IMAGE"]),
    ):
        request = copy.deepcopy(payload)
        request["message"]["body"] = copy.deepcopy(items)
        requests.append((kind, request))
    return requests

def _combine_results(parts):
    accepted = sum(part["status"] == "accepted" for part in parts)
    if len(parts) == 1:
        result = {key: value for key, value in parts[0].items()
                  if key not in ("part", "attempted")}
    elif accepted == len(parts):
        result = _reply("accepted", "文字和图片已分别被平台接受，群内将显示为两条消息。")
    elif any(part["status"] == "unknown" for part in parts):
        result = _reply("unknown", "部分结果未确认；查看 parts 和群聊，勿整组重发。")
    elif accepted:
        result = _reply("partial", "仅部分消息被平台接受；查看 parts，勿重复发送已成功部分。")
    elif parts[0]["status"] == "cancelled":
        result = _reply("cancelled", "发送前卡片停止或配置改变，本组消息未发送。")
    else:
        result = _reply("failed", "发送失败，后续部分未发送；请查看 parts 的具体错误。")
    result.update(parts=parts, accepted_parts=accepted, total_parts=len(parts),
                  request_count=sum(part["attempted"] for part in parts))
    return result
