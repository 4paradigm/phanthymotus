"""Real localhost TLS + HTTP proxy tests; no robot, Internet, or release APK."""
import asyncio
import base64
import datetime
import hashlib
import io
import json
import importlib.util
from pathlib import Path
import ssl
import subprocess
import sys
import time
from contextlib import asynccontextmanager

import aiohttp.web
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI

from test_teleop_project_lifecycle import config  # isolated DB/module import setup
from api import teleop_install as api
import auth


APK = b'local signed-artifact placeholder: proxy checksum contract only'


@pytest.fixture(autouse=True)
def installation_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path/'core.db'))


@asynccontextmanager
async def capture(tmp_path, *, wrong_hash=False, redirect=False, download_gate=None):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-datetime.timedelta(minutes=1)).not_valid_after(now+datetime.timedelta(hours=1))
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path/'capture.pem', tmp_path/'capture.key'
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    package = {'available': True, 'version': 'test', 'version_code': 1, 'filename': 'pico.apk',
               'sha256': '0'*64 if wrong_hash else hashlib.sha256(APK).hexdigest(), 'size_bytes': len(APK)}
    app = aiohttp.web.Application()
    seen = []
    async def metadata(request):
        seen.append(request.path)
        if redirect:
            raise aiohttp.web.HTTPFound('/untrusted')
        return aiohttp.web.json_response(package)
    async def artifact(request):
        seen.append(request.path)
        if download_gate:
            download_gate[0].set()
            await download_gate[1].wait()
        return aiohttp.web.Response(body=APK)
    app.router.add_get('/onboarding/package', metadata)
    app.router.add_get('/onboarding/apk', artifact)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, '127.0.0.1', 0, ssl_context=context)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    info = {'capture_origin': f'https://127.0.0.1:{port}', 'certificate_sha256': cert.fingerprint(hashes.SHA256()).hex(),
            'package_path': '/onboarding/package', 'apk_path': '/onboarding/apk'}
    try:
        yield info, seen
    finally:
        await runner.cleanup()


def client(monkeypatch, info, *, stub_mcp=True):
    async def call(mid, action, **kwargs):
        assert mid == 'registered-ac'
        if action == 'installation_info':
            return info
        assert action == 'create_invitation'
        payload = base64.urlsafe_b64encode(json.dumps({'token': 'one-use-fixture'}).encode()).decode().rstrip('=')
        return {'deep_link': 'motus-teleop://connect#' + payload, 'invitation_id': 'invitation-fixture'}
    if stub_mcp:
        monkeypatch.setattr(api, '_call', call)
    monkeypatch.setattr(auth, '_auth_enabled', True)
    monkeypatch.setattr(auth, '_token', 'dashboard-fixture')
    app = FastAPI()
    app.middleware('http')(auth.auth_middleware)
    nested = FastAPI()
    nested.include_router(api.router)
    app.mount('/api', nested)
    app.include_router(api.public_router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://core.example',
                             headers={'Authorization': 'Bearer dashboard-fixture'})


def test_pinned_proxy_download_and_fragment_invitation(tmp_path, monkeypatch):
    async def run():
        async with capture(tmp_path) as (info, seen), client(monkeypatch, info) as c:
            result = await c.post('/api/teleop-install/registered-ac')
            assert result.status_code == 200, result.text
            data = result.json()['data']
            assert data['url'] == 'https://core.example/pico/' + data['ticket']
            path = '/pico/' + data['ticket']
            c.headers.clear()
            page = await c.get(path)
            assert page.status_code == 200 and 'location.hash' in page.text
            assert 'one-use-fixture' not in page.text
            download = await c.get(path+'/apk')
            assert download.status_code == 200 and download.content == APK
            assert download.headers['cache-control'] == 'no-store'
            assert (await c.post('/api/teleop-install/registered-ac')).status_code == 401
            c.headers['Authorization'] = 'Bearer dashboard-fixture'
            invitation = await c.post('/api/teleop-install/registered-ac/invitation/' + data['ticket'])
            invited = invitation.json()['data']
            assert invitation.status_code == 200 and '#' in invited['url']
            assert invited['url'].split('#')[1] == invited['deep_link'].split('#')[1]
            assert 'token' not in api._ticket(data['ticket'])
            assert (await c.post('/api/teleop-install/other/invitation/' + data['ticket'])).status_code == 403
            monkeypatch.setattr(api, '_ticket_time', lambda: time.time()+901)
            assert (await c.get(path)).status_code == 410
            assert seen == ['/onboarding/package', '/onboarding/apk']
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['pin', 'redirect', 'hash'])
def test_proxy_refuses_untrusted_or_changed_artifacts(tmp_path, monkeypatch, failure):
    async def run():
        async with capture(tmp_path, wrong_hash=failure=='hash', redirect=failure=='redirect') as (info, seen):
            if failure == 'pin':
                info['certificate_sha256'] = '0'*64
            async with client(monkeypatch, info) as c:
                result = await c.post('/api/teleop-install/registered-ac')
                if failure == 'hash':
                    assert result.status_code == 200
                    result = await c.get('/pico/'+result.json()['data']['ticket']+'/apk')
                assert result.status_code in (502, 503)
                assert '/untrusted' not in seen
    asyncio.run(run())


def test_real_svg_qr_decodes_exact_fragment_without_remote_service():
    # Decoder/rendering are test tools only; the production dependency is qrcode.
    cairosvg = pytest.importorskip('cairosvg')
    zxingcpp = pytest.importorskip('zxingcpp')
    from PIL import Image
    url = 'https://core.example/pico/download-only#eyJ0b2tlbiI6Im9uZS11c2UifQ'
    svg = api.qr_svg(url)
    png = cairosvg.svg2png(bytestring=svg.encode(), scale=3)
    decoded = zxingcpp.read_barcode(Image.open(io.BytesIO(png)))
    assert decoded and decoded.text == url


@pytest.mark.parametrize('change', [{'capture_origin':'http://127.0.0.1'}, {'capture_origin':'https://user:secret@127.0.0.1'},
                                  {'package_path':'/other'}, {'apk_path':'/onboarding/apk?path=anything'},
                                  {'certificate_sha256':'bad'}])
def test_fixed_origin_and_paths(change):
    from fastapi import HTTPException
    value = {'capture_origin':'https://127.0.0.1:15741','certificate_sha256':'a'*64,
             'package_path':'/onboarding/package','apk_path':'/onboarding/apk', **change}
    with pytest.raises(HTTPException):
        api._endpoint(value)


def test_ticket_survives_new_process_and_two_workers_share_download_budget(tmp_path, monkeypatch):
    info = {'capture_origin':'https://localhost:15741', 'certificate_sha256':'a'*64,
            'package_path':'/onboarding/package', 'apk_path':'/onboarding/apk',
            'package': {'ignored':'summary'}, 'token':'must-not-persist'}
    package = {'available':True,'sha256':hashlib.sha256(APK).hexdigest(),'size_bytes':len(APK)}
    token = api._create_ticket('registered-ac', info, package)
    # A new interpreter has no inherited module/global ticket state.
    code = ('import config; config.DB_PATH=__import__("sys").argv[1]; '
            'from api.teleop_install import _ticket; import json; '
            'print(json.dumps(_ticket(__import__("sys").argv[2])))')
    result = subprocess.run([sys.executable, '-c', code, config.DB_PATH, token],
                            env={**__import__('os').environ, 'PYTHONPATH':str(Path(__file__).parents[1]/'src')},
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['package'] == package
    spec = importlib.util.spec_from_file_location('teleop_install_second_worker', api.__file__)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    lease = api._ticket(token, reserve_download=True)
    with pytest.raises(api.HTTPException) as caught:
        worker._ticket(token, reserve_download=True)
    assert caught.value.status_code == 429
    api._release_download(token, lease['download_id'])
    assert worker._ticket(token)['mcp_id'] == 'registered-ac'
    with api._ticket_db() as db:
        saved = db.execute('SELECT digest,info,package,downloads FROM teleop_install_tickets').fetchone()
    assert saved[0] == hashlib.sha256(token.encode()).hexdigest()
    assert saved[3] == 1 and token not in str(saved) and 'must-not-persist' not in str(saved)


def test_repeat_downloads_are_bounded_rotation_and_revocation_are_authorized(tmp_path, monkeypatch):
    async def run():
        clock = [time.time()]
        monkeypatch.setattr(api, '_ticket_time', lambda: clock[0])
        async with capture(tmp_path) as (info, seen), client(monkeypatch, info) as c:
            data = (await c.post('/api/teleop-install/registered-ac')).json()['data']
            path = '/pico/'+data['ticket']
            for attempt in range(3):
                response = await c.get(path+'/apk')
                assert response.status_code == 200 and response.content == APK
                if attempt == 0:
                    limited = await c.get(path+'/apk')
                    assert limited.status_code == 429 and 'retry-after' in limited.headers
                clock[0] += 3
            assert (await c.get(path+'/apk')).status_code == 429
            assert seen.count('/onboarding/apk') == 3
            replacement = (await c.post('/api/teleop-install/registered-ac')).json()['data']
            assert (await c.get(path)).status_code == 410
            assert (await c.get(path+'/apk')).status_code == 410
            assert (await c.post('/api/teleop-install/registered-ac/invitation/'+data['ticket'])).status_code == 410
            token = replacement['ticket']
            c.headers.clear()
            assert (await c.delete('/api/teleop-install/registered-ac/'+token)).status_code == 401
            assert (await c.post('/api/teleop-install/registered-ac/invitation/'+token)).status_code == 401
            c.headers['Authorization'] = 'Bearer dashboard-fixture'
            assert (await c.delete('/api/teleop-install/other/'+token)).status_code == 403
            assert (await c.delete('/api/teleop-install/registered-ac/'+token)).status_code == 200
            assert (await c.get('/pico/'+token)).status_code == 410
            assert (await c.get('/pico/'+token+'/apk')).status_code == 410
    asyncio.run(run())


def test_ticket_capacity_expiry_and_backward_clock_are_bounded(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(api, '_ticket_time', lambda: clock[0])
    info = {'capture_origin':'https://localhost:15741','certificate_sha256':'a'*64,
            'package_path':'/onboarding/package','apk_path':'/onboarding/apk'}
    for i in range(api._MAX_TICKETS):
        api._create_ticket(str(i), info, {})
    with pytest.raises(api.HTTPException) as caught:
        api._create_ticket('overflow', info, {})
    assert caught.value.status_code == 429
    token = api._create_ticket('0', info, {})  # replacement still works at capacity
    clock[0] = 999.
    with pytest.raises(api.HTTPException) as caught:
        api._ticket(token)
    assert caught.value.status_code == 410
    clock[0] = 1900.
    with pytest.raises(api.HTTPException):
        api._ticket(token)
    api._create_ticket('fresh', info, {})
    with api._ticket_db() as db:
        assert db.execute('SELECT COUNT(*) FROM teleop_install_tickets').fetchone()[0] == 1


def test_active_download_can_be_revoked_and_parallel_attempt_never_hits_upstream(tmp_path, monkeypatch):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async with capture(tmp_path, download_gate=(started,release)) as (info, seen), client(monkeypatch, info) as c:
            token = (await c.post('/api/teleop-install/registered-ac')).json()['data']['ticket']
            downloading = asyncio.create_task(c.get('/pico/'+token+'/apk'))
            await asyncio.wait_for(started.wait(), 2)
            try:
                assert (await c.get('/pico/'+token+'/apk')).status_code == 429
                assert (await c.delete('/api/teleop-install/registered-ac/'+token)).status_code == 200
            finally:
                release.set()
            result = await downloading
            assert result.status_code == 410
            assert APK not in result.content and seen.count('/onboarding/apk') == 1
    asyncio.run(run())


def test_crashed_download_lease_recovers_without_extending_ticket(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(api,'_ticket_time',lambda:clock[0])
    info = {'capture_origin':'https://localhost:15741','certificate_sha256':'a'*64,
            'package_path':'/onboarding/package','apk_path':'/onboarding/apk'}
    token = api._create_ticket('local',info,{})
    first = api._ticket(token,reserve_download=True)
    clock[0] += api._DOWNLOAD_LEASE+1
    second = api._ticket(token,reserve_download=True)
    assert second['expires'] == first['expires'] == 1900.
    api._release_download(token,first['download_id'])
    assert api._ticket(token,download_id=second['download_id'])
    with pytest.raises(api.HTTPException) as caught:
        api._ticket(token,download_id=first['download_id'])
    assert caught.value.status_code == 410
