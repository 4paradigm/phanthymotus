"""Dashboard-owned PICO installation, with fixed paths and pinned capture TLS.

Public short links grant only a bounded APK download, not pairing or motion.
Pairing payloads live in the URL fragment and never reach the public server.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import secrets
import tempfile
import time
from urllib.parse import urlsplit

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

router = APIRouter(prefix='/teleop-install', tags=['teleop'])
public_router = APIRouter(prefix='/pico', tags=['teleop-install'])
_tickets = {}
_TTL = 900
_MAX_APK = 512 * 1024 * 1024
_HEADERS = {'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
            'X-Content-Type-Options': 'nosniff'}


def qr_svg(value):
    # qrcode 8.2: BSD-3-Clause; SVG needs no Pillow or external service.
    import qrcode
    from qrcode.exceptions import DataOverflowError
    from qrcode.image.svg import SvgPathFillImage
    try:
        image = qrcode.make(value, image_factory=SvgPathFillImage, box_size=5, border=4)
    except DataOverflowError:
        raise HTTPException(422, '连接资料过长，无法生成二维码') from None
    out = io.BytesIO()
    image.save(out)
    return out.getvalue().decode('utf-8')


async def _call(mcp_id, action, **arguments):
    from api.mcp_manage import MCPCallRequest, mcp_call_tool
    from api.config import payload_of
    result = await mcp_call_tool(mcp_id, MCPCallRequest(tool='teleop', arguments={'action': action, **arguments}), timeout_s=10)
    if result.get('code') != 200:
        raise HTTPException(503, '遥操服务未确认安装或邀请操作')
    value = payload_of(result)
    if value.get('error') or value.get('state') in ('error', 'fault'):
        raise HTTPException(409, '遥操服务拒绝安装或邀请操作')
    return value


def _endpoint(info):
    origin = info.get('capture_origin', '')
    pin = info.get('certificate_sha256', '')
    try:
        parsed = urlsplit(origin)
        valid = (parsed.scheme == 'https' and bool(parsed.hostname)
                 and parsed.path in ('', '/') and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
                 and (parsed.port is None or 0 < parsed.port < 65536)
                 and isinstance(pin, str) and re.fullmatch(r'[0-9a-fA-F]{64}', pin))
    except (TypeError, ValueError):
        valid = False
    if not valid or info.get('package_path') != '/onboarding/package' or info.get('apk_path') != '/onboarding/apk':
        raise HTTPException(503, '安装地址或证书指纹未配置')
    return origin.rstrip('/'), aiohttp.Fingerprint(bytes.fromhex(pin))


async def _package(info):
    origin, fingerprint = _endpoint(info)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12), trust_env=False) as session:
            async with session.get(origin + '/onboarding/package', ssl=fingerprint, allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(503, '当前镜像没有可下载的 PICO 安装包')
                raw = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    raw.extend(chunk)
                    if len(raw) > 65536:
                        raise HTTPException(502, '安装包元数据过大')
                package = json.loads(raw)
    except (aiohttp.ClientError, TimeoutError, ValueError):
        raise HTTPException(502, '安装服务不可达、证书不匹配或元数据无效') from None
    if (not isinstance(package, dict) or package.get('available') is not True
            or not re.fullmatch(r'[0-9a-f]{64}', str(package.get('sha256', '')))
            or type(package.get('size_bytes')) is not int or not 0 < package['size_bytes'] <= _MAX_APK):
        raise HTTPException(503, '当前安装包尚未就绪')
    return package


def _ticket(token):
    entry = _tickets.get(token)
    if not entry or entry['expires'] <= time.monotonic():
        _tickets.pop(token, None)
        raise HTTPException(410, '安装链接已过期，请在遥操卡片重新生成')
    return entry


@router.post('/{mcp_id}')
async def prepare_installation(mcp_id: str, request: Request):
    info = await _call(mcp_id, 'installation_info')
    package = await _package(info)
    now = time.monotonic()
    for token, entry in list(_tickets.items()):
        if entry['expires'] <= now:
            del _tickets[token]
    if len(_tickets) >= 128:
        raise HTTPException(429, '安装链接过多，请稍后再试')
    token = secrets.token_urlsafe(12)
    _tickets[token] = {'expires': now + _TTL, 'mcp_id': mcp_id, 'info': info, 'package': package}
    url = str(request.base_url).rstrip('/') + '/pico/' + token
    return {'code': 200, 'data': {'ticket': token, 'url': url, 'expires_in_seconds': _TTL,
                                 'package': package, 'qr_svg': qr_svg(url)}}


@router.post('/{mcp_id}/invitation/{token}')
async def prepare_invitation(mcp_id: str, token: str, request: Request):
    entry = _ticket(token)
    if entry['mcp_id'] != mcp_id:
        raise HTTPException(403, '安装链接不属于此遥操服务')
    invitation = await _call(mcp_id, 'create_invitation')
    link = invitation.get('deep_link', '')
    prefix = 'motus-teleop://connect#'
    if not isinstance(link, str) or not link.startswith(prefix) or not re.fullmatch(r'[A-Za-z0-9_-]{1,8192}', link[len(prefix):]):
        raise HTTPException(502, '邀请格式无效')
    url = str(request.base_url).rstrip('/') + '/pico/' + token + '#' + link[len(prefix):]
    return {'code': 200, 'data': {**invitation, 'url': url, 'qr_svg': qr_svg(url)}}


@public_router.get('/{token}', response_class=HTMLResponse)
async def installation_page(token: str):
    _ticket(token)
    # No template interpolation with server/peer text; hashes stay client-only.
    return HTMLResponse('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PICO 遥操安装</title>
<body style="font:18px/1.7 system-ui;max-width:640px;margin:40px auto;padding:20px">
<h1>连接机器人</h1><p>在 PICO 浏览器下载并安装应用，然后返回本页连接。</p>
<p><a id="apk">下载 PICO 安装包</a></p><p><a id="connect" hidden>打开应用并连接此机器人</a></p>
<p id="hint">已有应用时，请在 Canvas 生成一次性连接邀请。安装不会启动机器人。</p>
<script>const p=location.pathname.replace(/\\/$/,'');document.querySelector('#apk').href=p+'/apk';
const payload=location.hash.slice(1);if(/^[A-Za-z0-9_-]{1,8192}$/.test(payload)){
const a=document.querySelector('#connect');a.href='motus-teleop://connect#'+payload;a.hidden=false;
document.querySelector('#hint').textContent='点击连接会预填机器人资料并使用一次性邀请；不会开始运动。';}
</script></body></html>''', headers={**_HEADERS, 'Content-Security-Policy': "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'"})


@public_router.get('/{token}/apk')
async def download_apk(token: str):
    entry = _ticket(token)
    origin, fingerprint = _endpoint(entry['info'])
    expected = entry['package']
    data = tempfile.TemporaryFile()
    try:
        digest, size = hashlib.sha256(), 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180, connect=5, sock_read=30), trust_env=False) as session:
            async with session.get(origin + '/onboarding/apk', ssl=fingerprint, allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(502, '安装包下载失败，请重新生成链接')
                async for chunk in response.content.iter_chunked(256 * 1024):
                    size += len(chunk)
                    if size > expected['size_bytes']:
                        raise HTTPException(502, '安装包大小已变化，请重新生成链接')
                    digest.update(chunk)
                    data.write(chunk)
        if size != expected['size_bytes'] or digest.hexdigest() != expected['sha256']:
            raise HTTPException(502, '安装包校验失败，请重新生成链接')
        data.seek(0)
    except BaseException as error:
        data.close()
        if isinstance(error, (aiohttp.ClientError, TimeoutError)):
            raise HTTPException(502, '安装包连接失败或证书不匹配') from None
        raise

    def chunks():
        try:
            while chunk := data.read(256 * 1024):
                yield chunk
        finally:
            data.close()
    return StreamingResponse(chunks(), media_type='application/vnd.android.package-archive',
                             headers={**_HEADERS, 'Content-Length': str(size),
                                      'Content-Disposition': 'attachment; filename="motus-pico.apk"'})
