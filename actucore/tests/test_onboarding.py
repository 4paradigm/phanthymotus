"""Enrollment capabilities and fixed-artifact serving; no ROS or devices."""
import asyncio
import base64
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
sys.path.insert(0, str(Path(__file__).parents[1]/'plugins'))
from teleop.enrollment import Enrollment
from teleop.capture import CaptureError
from teleop.onboarding import package_metadata


class Capture:
    paired = False
    issued = 0
    async def status(self):
        return {'paired_devices': int(self.paired)}
    async def create_pairing(self):
        await asyncio.sleep(0)
        if self.paired:
            raise CaptureError('revoke_existing_headset_first', status=409)
        self.issued += 1
        return {'pairing_id': str(self.issued)}


def enrollment():
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'test')])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(1).not_valid_before(now-datetime.timedelta(minutes=1))
        .not_valid_after(now+datetime.timedelta(days=1)).sign(key, hashes.SHA256()))
    clock = [1.0]
    return Enrollment(Capture(), base64.b64encode(certificate.public_bytes(serialization.Encoding.PEM)),
        clock=lambda:clock[0], public_wss_url='wss://robot.local:15741/ws/teleop-capture'), clock


def request(invitation):
    return {k:invitation[k] for k in ('invitation_id','token','device_id')} | {'device_name':'PICO 4 Ultra'}


def test_single_use_invitation_concurrent_replay_and_no_status_secret():
    async def run():
        e, _ = enrollment()
        invite = await e.create_invitation()
        assert invite['expires_in_seconds'] == 900
        fragment = invite['deep_link'].split('#')[1]
        payload = json.loads(base64.urlsafe_b64decode(fragment+'='*(-len(fragment)%4)))
        assert payload['device_id'] == payload['certificate_sha256'] == e.device_id
        assert invite['token'] not in json.dumps(e.status())
        results = await asyncio.gather(e.redeem_invitation(request(invite)), e.redeem_invitation(request(invite)), return_exceptions=True)
        assert sum(isinstance(x, dict) for x in results) == 1
        assert sum(isinstance(x, CaptureError) for x in results) == 1
        assert e.capture.issued == 1
        assert e.status()['invitation'] is None
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['expired','revoked','rotated','wrong_robot','wrong_token','paired','restart','manual_pairing'])
def test_invitation_fail_closed(failure):
    async def run():
        e, clock = enrollment()
        invite = await e.create_invitation()
        data = request(invite)
        if failure == 'expired': clock[0] += 900
        elif failure == 'revoked': e.revoke_invitation()
        elif failure == 'rotated': await e.create_invitation()
        elif failure == 'wrong_robot': data['device_id'] = '0'*64
        elif failure == 'wrong_token': data['token'] = 'x'*43
        elif failure == 'paired': e.capture.paired = True
        elif failure == 'restart': e, _ = enrollment()
        elif failure == 'manual_pairing': await e.open()
        with pytest.raises(CaptureError): await e.redeem_invitation(data)
        assert e.capture.issued == 0
    asyncio.run(run())


def artifact(tmp_path):
    apk = tmp_path/'pico.apk'
    apk.write_bytes(b'verified-fixture-not-an-installable-apk')
    manifest = {'filename':'pico.apk','application_id':'com.phanthymotus.picocapture',
        'build_type':'release','version':'test','version_code':1,'sha256':hashlib.sha256(apk.read_bytes()).hexdigest(),
        'signing_certificate_sha256':'a'*64,'size_bytes':apk.stat().st_size}
    (tmp_path/'package.json').write_text(json.dumps(manifest))
    return manifest


def test_artifact_tamper_and_symlinks_rejected(tmp_path):
    artifact(tmp_path)
    assert package_metadata(tmp_path)['available']
    (tmp_path/'pico.apk').write_bytes(b'changed')
    assert not package_metadata(tmp_path)['available']
    artifact(tmp_path)
    (tmp_path/'pico.apk').rename(tmp_path/'other.apk')
    (tmp_path/'pico.apk').symlink_to(tmp_path/'other.apk')
    assert not package_metadata(tmp_path)['available']


def test_fetch_cached_fixed_artifact_and_no_insecure_source(tmp_path):
    manifest = artifact(tmp_path)
    script = Path(__file__).parents[1]/'openxr_capture_native/scripts/fetch_apk.py'
    spec = importlib.util.spec_from_file_location('fetch_apk', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path/'source.json'
    manifest['source_url'] = 'https://example.invalid/fixed/pico.apk'
    source.write_text(json.dumps(manifest))
    assert module.fetch(source, tmp_path) == manifest['sha256']
    manifest['source_url'] = 'http://example.invalid/pico.apk'
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): module.fetch(source, tmp_path)


@pytest.mark.parametrize('failure',[None,'source_hash','apk_hash','expanded_size'])
def test_compressed_build_artifact_is_bounded_and_verified_before_install(tmp_path,monkeypatch,failure):
    import gzip
    import io
    manifest=artifact(tmp_path)
    data=(tmp_path/'pico.apk').read_bytes()
    compressed=gzip.compress(data,mtime=0)
    manifest.update(source_url='https://example.invalid/container-apk.gz',source_encoding='gzip',
        source_sha256=hashlib.sha256(compressed).hexdigest(),source_size_bytes=len(compressed))
    if failure=='source_hash':manifest['source_sha256']='0'*64
    if failure=='apk_hash':manifest['sha256']='0'*64
    if failure=='expanded_size':manifest['size_bytes']-=1
    script=Path(__file__).parents[1]/'openxr_capture_native/scripts/fetch_apk.py'
    spec=importlib.util.spec_from_file_location('fetch_apk_compressed',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Response(io.BytesIO):
        def geturl(self):return manifest['source_url']
    monkeypatch.setattr(module,'urlopen',lambda *a,**k:Response(compressed))
    source=tmp_path/'source.json';source.write_text(json.dumps(manifest))
    destination=tmp_path/'download'
    if failure:
        with pytest.raises(ValueError):module.fetch(source,destination)
        assert list(destination.iterdir())==[]
    else:
        assert module.fetch(source,destination)==manifest['sha256']
        assert (destination/'pico.apk').read_bytes()==data
        assert not any(k.startswith('source_') for k in json.loads((destination/'package.json').read_text()))


def test_http_download_and_invitation_routes(tmp_path, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer
    import teleop.capture_server as server
    async def run():
        artifact(tmp_path)
        monkeypatch.setattr(server, 'APK_DIRECTORY', tmp_path)
        monkeypatch.setattr(server, 'package_metadata', lambda: package_metadata(tmp_path))
        e, _ = enrollment()
        async with TestClient(TestServer(server.create_capture_app(e.capture, e))) as client:
            assert (await (await client.get('/onboarding/package')).json())['available']
            response = await client.get('/onboarding/apk')
            assert response.status == 200
            assert await response.read() == (tmp_path/'pico.apk').read_bytes()
            assert (await client.get('/onboarding/apk?path=/etc/passwd')).status == 400
            invite = await e.create_invitation()
            assert (await client.post('/pairing/invite', json=request(invite))).status == 200
            assert (await client.post('/pairing/invite', json=request(invite))).status == 403
            (tmp_path/'pico.apk').write_bytes(b'changed')
            assert (await client.get('/onboarding/apk')).status == 503
    asyncio.run(run())


def test_invitation_cannot_replace_pairing_that_wins_handshake_race():
    from teleop.capture import CaptureManager
    from teleop.descriptor import CAPTURE_PROTOCOL, RTC_FRAME_PROTOCOL
    async def run():
        e, _ = enrollment()
        manager = CaptureManager(None, None, None)
        e.capture = manager
        old_pairing = await manager.create_pairing()
        invite = await e.create_invitation()
        hello = dict(type='pair', pairing_id=old_pairing['pairing_id'],
            pairing_code=old_pairing['pairing_code'], capture_protocol=CAPTURE_PROTOCOL,
            frame_protocol=RTC_FRAME_PROTOCOL, client_kind='native_openxr', app_version='test')
        # Queue an already-issued pairing handshake first, then the invitation
        # redemption. Neither a stale status read nor a new token can replace it.
        await manager._lock.acquire()
        connecting = asyncio.create_task(manager.connect(hello))
        await asyncio.sleep(0)
        redeeming = asyncio.create_task(e.redeem_invitation(request(invite)))
        await asyncio.sleep(0)
        manager._lock.release()
        connection, ack = await connecting
        with pytest.raises(CaptureError, match='revoke_existing_headset_first'):
            await redeeming
        status = await manager.status()
        assert status['paired_devices'] == 1
        assert status['capture_id'] == ack['capture_id'] == connection.capture_id
        assert status['pending_pairings'] == 0
        assert e.status()['invitation'] is None
        with pytest.raises(CaptureError, match='revoke_existing_headset_first'):
            await e.create_invitation()
    asyncio.run(run())
