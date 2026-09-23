"""JVM production parser/TLS + real Python enrollment, on localhost only.

Requires JDK 17, Android compile stubs and a test-only org.json runtime jar.
Only Android Base64 is adapted to the JDK; no Android Activity/UI is executed.
This is not PICO installation, permission, browser deep-link or XR acceptance.
The jar is supplied explicitly; this test never downloads or installs tools.
"""
import asyncio
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile

from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'plugins'))
from teleop.enrollment import Enrollment
from teleop.capture import CaptureError


async def main():
    android = Path(os.environ['MOTUS_ANDROID_JAR']).resolve()
    json_jar = Path(os.environ['MOTUS_JAVA_JSON_JAR']).resolve()
    java = Path(os.environ['JAVA_HOME']) / 'bin'
    for path in (android, json_jar, java/'java', java/'javac'):
        assert path.is_file(), path
    # This test fixture's JSON runtime is fixed; it is not an APK dependency.
    assert hashlib.sha256(json_jar.read_bytes()).hexdigest() == os.environ['MOTUS_JAVA_JSON_SHA256']
    checks = 0
    with tempfile.TemporaryDirectory(prefix='motus-invitation-') as directory:
        temp = Path(directory)
        shim = temp/'Base64.java'
        shim.write_text('''package android.util;
public final class Base64 {
 public static final int DEFAULT=0,NO_WRAP=2,NO_PADDING=1,URL_SAFE=8;
 public static byte[] decode(String s,int f) {
  return ((f&URL_SAFE)!=0?java.util.Base64.getUrlDecoder():java.util.Base64.getMimeDecoder()).decode(s);
 }
 public static String encodeToString(byte[] b,int f) {
  java.util.Base64.Encoder e=(f&URL_SAFE)!=0?java.util.Base64.getUrlEncoder():java.util.Base64.getEncoder();
  return ((f&NO_PADDING)!=0?e.withoutPadding():e).encodeToString(b);
 }
}''')
        source = ROOT/'app/src/main/java/com/phanthymotus/capture'
        classpath = os.pathsep.join(map(str, (temp, json_jar, android)))
        subprocess.run([str(java/'javac'), '-encoding', 'UTF-8', '-cp', classpath, '-d', str(temp), str(shim),
            str(source/'ConnectionInvitation.java'), str(source/'ConnectionActivity.java'),
            str(ROOT/'tests/InvitationContract.java')], check=True)
        async def run(link, *, valid=True, mode='post'):
            nonlocal checks
            process = await asyncio.create_subprocess_exec(str(java/'java'), '-cp', classpath,
                'com.phanthymotus.capture.InvitationContract', stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            output, error = await asyncio.wait_for(process.communicate(
                json.dumps({'uri': link, 'mode': mode}).encode()), timeout=15)
            # Failure detail can contain endpoint/URI data; never echo it.
            assert (process.returncode == 0) == valid, (checks, mode, process.returncode)
            if valid and mode == 'post':assert output.strip() == b'APPROVED'
            checks += 1

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
        now = datetime.datetime.now(datetime.timezone.utc)
        certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-datetime.timedelta(minutes=1))
            .not_valid_after(now+datetime.timedelta(hours=1)).sign(key, hashes.SHA256()))
        cert = certificate.public_bytes(serialization.Encoding.PEM)
        (temp/'cert.pem').write_bytes(cert)
        (temp/'key.pem').write_bytes(key.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(temp/'cert.pem', temp/'key.pem')

        class Capture:
            issued = 0
            async def status(self):return {'paired_devices': 0}
            async def create_pairing(self):
                self.issued += 1
                return {'pairing_id': 'fixture', 'pairing_code': 'fixture',
                    'wss_url': enrollment.public_wss_url, 'ca_certificate_base64': base64.b64encode(cert).decode()}

        clock = [1.]
        capture = Capture()
        enrollment = Enrollment(capture, base64.b64encode(cert), clock=lambda: clock[0])
        received = []
        async def redeem(request):
            try:
                value = await enrollment.redeem_invitation(await request.json())
                received.append((request.path, 200))
                return web.json_response(value)
            except CaptureError as exc:
                received.append((request.path, exc.status))
                return web.json_response({'error': exc.code}, status=exc.status)
        app = web.Application()
        app.router.add_post('/pairing/invite', redeem)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0, ssl_context=context)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        enrollment.public_wss_url = f'wss://127.0.0.1:{port}/ws/teleop-capture'
        def altered(link, **changes):
            encoded = link.split('#')[1]
            payload = json.loads(base64.urlsafe_b64decode(encoded+'='*(-len(encoded)%4)))
            payload.update(changes)
            return link.split('#')[0]+'#'+base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
        try:
            link = (await enrollment.create_invitation())['deep_link']
            await run(link, mode='parse')
            for bad in (link.replace('motus-teleop:', 'https:'), link.replace('#', '?unsafe=1#'),
                        altered(link, device_id='0'*64), altered(link, endpoint='user@127.0.0.1:443'),
                        altered(link, endpoint='127.0.0.1:443/path'), altered(link, extra='unsupported')):
                await run(bad, valid=False, mode='parse')
            await run(link)
            assert capture.issued == 1
            await run(link, valid=False)  # Actual HTTP replay cannot mint another pairing.
            assert capture.issued == 1 and received[-1] == ('/pairing/invite', 403)
            for failure in ('expired', 'revoked', 'rotated', 'token', 'pin'):
                link = (await enrollment.create_invitation())['deep_link']
                if failure == 'expired':clock[0] += 901
                elif failure == 'revoked':enrollment.revoke_invitation()
                elif failure == 'rotated':await enrollment.create_invitation()
                elif failure == 'token':link = altered(link, token='x'*43)
                else:link = altered(link, device_id='0'*64, certificate_sha256='0'*64)
                before = len(received)
                await run(link, valid=False)
                if failure == 'pin':assert len(received) == before, 'TLS pin failed after sending capability'
                else:assert received[-1] == ('/pairing/invite', 403)
                assert capture.issued == 1
            await run((await enrollment.create_invitation())['deep_link'])
            assert capture.issued == 2  # A failed invitation never wedges later enrollment.
        finally:
            await runner.cleanup()
    print(f'INVITATION CONTRACT PASS: {checks} JVM/Python checks; localhost only; no Activity or robot')


if __name__ == '__main__':
    asyncio.run(main())
