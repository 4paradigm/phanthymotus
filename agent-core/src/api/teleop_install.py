"""Dashboard-owned PICO installation, with fixed paths and pinned capture TLS.

Public short links grant only a bounded APK download, not pairing or motion.
Pairing payloads live in the URL fragment and never reach the public server.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import secrets
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from urllib.parse import urlsplit

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

import config

router = APIRouter(prefix='/teleop-install', tags=['teleop'])
public_router = APIRouter(prefix='/pico', tags=['teleop-install'])
_TTL = 900
_MAX_TICKETS = 128
_MAX_DOWNLOADS = 3
_DOWNLOAD_INTERVAL = 2
_DOWNLOAD_LEASE = 240
_MAX_APK = 512 * 1024 * 1024
_CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
_INSTALL_PAGE = Path(__file__).resolve().parents[2] / 'web/pico-install.html'
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


def _ticket_time():
    # Absolute expiry survives process restarts; an observed backward jump before
    # creation invalidates a ticket instead of extending its lifetime.
    return time.time()


@contextmanager
def _ticket_db():
    """Bounded private table in the existing shared Core DB, not Canvas config.

    Workers/containers must share this same DB volume, as for other Core state.
    Only download-token hashes and public endpoint/package metadata are stored;
    pairing invitations and management credentials never enter this table.
    """
    try:
        db = sqlite3.connect(config.DB_PATH, timeout=2)
        try:
            db.execute('CREATE TABLE IF NOT EXISTS teleop_install_tickets ('
                       'digest TEXT PRIMARY KEY, mcp_id TEXT UNIQUE NOT NULL, '
                       'created REAL NOT NULL, expires REAL NOT NULL, '
                       'info TEXT NOT NULL, package TEXT NOT NULL, '
                       'downloads INTEGER NOT NULL DEFAULT 0, next_download REAL NOT NULL DEFAULT 0, '
                       'active_until REAL NOT NULL DEFAULT 0, active_id TEXT)')
            with db:
                db.execute('BEGIN IMMEDIATE')
                yield db
        finally:
            db.close()
    except sqlite3.Error:
        raise HTTPException(503, '安装链接存储暂不可用，请稍后重试') from None


def _ticket_digest(token):
    if not isinstance(token, str) or not re.fullmatch(r'(?:[A-HJ-NP-Z2-9]{12}|[A-Za-z0-9_-]{16})', token):
        raise HTTPException(410, '安装链接已过期或撤销，请在遥操卡片重新生成')
    return hashlib.sha256(token.encode()).hexdigest()


def _create_ticket(mcp_id, info, package):
    # Download-only capability: 12 unambiguous symbols (~59 bits), 15 minutes,
    # at most three downloads. Pairing retains its independent 256-bit invitation.
    token, now = ''.join(secrets.choice(_CODE_ALPHABET) for _ in range(12)), _ticket_time()
    public_info = {k: info[k] for k in ('capture_origin', 'certificate_sha256', 'package_path', 'apk_path')}
    with _ticket_db() as db:
        db.execute('DELETE FROM teleop_install_tickets WHERE expires <= ? OR created > ?', (now, now))
        # Generating a replacement is also the UI's revocation operation.
        db.execute('DELETE FROM teleop_install_tickets WHERE mcp_id = ?', (mcp_id,))
        if db.execute('SELECT COUNT(*) FROM teleop_install_tickets').fetchone()[0] >= _MAX_TICKETS:
            raise HTTPException(429, '安装链接过多，请稍后再试')
        db.execute('INSERT INTO teleop_install_tickets (digest,mcp_id,created,expires,info,package) '
                   'VALUES (?,?,?,?,?,?)', (_ticket_digest(token), mcp_id, now, now+_TTL,
                                          json.dumps(public_info), json.dumps(package)))
    return token


def _ticket(token, *, reserve_download=False, download_id=None):
    digest, now = _ticket_digest(token), _ticket_time()
    with _ticket_db() as db:
        row = db.execute('SELECT mcp_id,created,expires,info,package,downloads,next_download,active_until,active_id '
                         'FROM teleop_install_tickets WHERE digest = ?', (digest,)).fetchone()
        if not row or not row[1] <= now < row[2]:
            raise HTTPException(410, '安装链接已过期或撤销，请在遥操卡片重新生成')
        if download_id is not None and (row[8] != download_id or now >= row[7]):
            raise HTTPException(410, '下载已中断，请重试或重新生成链接')
        if reserve_download:
            if row[5] >= _MAX_DOWNLOADS:
                raise HTTPException(429, '此链接已达到三次下载限制，请在遥操卡片重新生成')
            until = max(row[6], row[7])
            if now < until:
                raise HTTPException(429, '下载正在进行或请求过快，请稍后重试',
                                    headers={'Retry-After': str(max(1, int(until-now)+1))})
            download_id = secrets.token_hex(16)
            db.execute('UPDATE teleop_install_tickets SET downloads=downloads+1,next_download=?,active_until=?,active_id=? '
                       'WHERE digest=?', (now+_DOWNLOAD_INTERVAL, min(row[2], now+_DOWNLOAD_LEASE), download_id, digest))
        return {'mcp_id': row[0], 'created': row[1], 'expires': row[2],
                'info': json.loads(row[3]), 'package': json.loads(row[4]), 'download_id': download_id}


def _release_download(token, download_id):
    with _ticket_db() as db:
        db.execute('UPDATE teleop_install_tickets SET active_until=0,active_id=NULL WHERE digest=? AND active_id=?',
                   (_ticket_digest(token), download_id))


@router.post('/{mcp_id}')
async def prepare_installation(mcp_id: str, request: Request):
    info = await _call(mcp_id, 'installation_info')
    package = await _package(info)
    token = _create_ticket(mcp_id, info, package)
    url = str(request.base_url).rstrip('/') + '/pico/' + token
    return {'code': 200, 'data': {'ticket': token, 'url': url, 'expires_in_seconds': _TTL,
                                 'entry_url': str(request.base_url).rstrip('/') + '/pico',
                                 'install_code': '-'.join(token[i:i+4] for i in range(0, len(token), 4)),
                                 'package': package, 'qr_svg': qr_svg(url)}}


@router.delete('/{mcp_id}/{token}')
async def revoke_installation(mcp_id: str, token: str):
    with _ticket_db() as db:
        row = db.execute('SELECT mcp_id FROM teleop_install_tickets WHERE digest=?', (_ticket_digest(token),)).fetchone()
        if row and row[0] != mcp_id:
            raise HTTPException(403, '安装链接不属于此遥操服务')
        db.execute('DELETE FROM teleop_install_tickets WHERE digest=?', (_ticket_digest(token),))
    return {'code': 200, 'revoked': True}


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


def _installation_page(status=200):
    return HTMLResponse(_INSTALL_PAGE.read_text(encoding='utf-8'), status_code=status,
                        headers={**_HEADERS, 'Content-Security-Policy':
                                 "default-src 'none'; script-src 'self'; connect-src 'self'; "
                                 "style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'"})


@public_router.get('', response_class=HTMLResponse)
@public_router.get('/', response_class=HTMLResponse)
async def installation_entry():
    """Bookmarkable entry; a download code never enrolls a headset."""
    return _installation_page()


@public_router.get('/{token}', response_class=HTMLResponse)
async def installation_page(token: str):
    try:
        _ticket(token)
    except HTTPException as exc:
        if exc.status_code != 410:
            raise
        # Render recovery instructions instead of a raw JSON error in the headset.
        return _installation_page(410)
    return _installation_page()


@public_router.get('/{token}/package')
async def installation_package(token: str):
    entry = _ticket(token)
    package = entry['package']
    return JSONResponse({k: package[k] for k in ('version', 'size_bytes', 'sha256')}, headers=_HEADERS)


@public_router.get('/{token}/apk')
async def download_apk(token: str):
    entry = _ticket(token, reserve_download=True)
    origin, fingerprint = _endpoint(entry['info'])
    expected = entry['package']
    data = tempfile.TemporaryFile()
    closed = False
    def close_download():
        nonlocal closed
        if not closed:
            closed = True
            data.close()
            _release_download(token, entry['download_id'])
    try:
        digest, size = hashlib.sha256(), 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180, connect=5, sock_read=30), trust_env=False) as session:
            async with session.get(origin + '/onboarding/apk', ssl=fingerprint, allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(502, '安装包下载失败，请重新生成链接')
                async for chunk in response.content.iter_chunked(256 * 1024):
                    _ticket(token, download_id=entry['download_id'])
                    size += len(chunk)
                    if size > expected['size_bytes']:
                        raise HTTPException(502, '安装包大小已变化，请重新生成链接')
                    digest.update(chunk)
                    data.write(chunk)
        if size != expected['size_bytes'] or digest.hexdigest() != expected['sha256']:
            raise HTTPException(502, '安装包校验失败，请重新生成链接')
        data.seek(0)
    except BaseException as error:
        close_download()
        if isinstance(error, (aiohttp.ClientError, TimeoutError)):
            raise HTTPException(502, '安装包连接失败或证书不匹配') from None
        raise

    def chunks():
        try:
            while chunk := data.read(256 * 1024):
                _ticket(token, download_id=entry['download_id'])
                yield chunk
        finally:
            close_download()
    return StreamingResponse(chunks(), media_type='application/vnd.android.package-archive',
                             headers={**_HEADERS, 'Content-Length': str(size),
                                      'Content-Disposition': 'attachment; filename="motus-pico.apk"'},
                             background=BackgroundTask(close_download))
