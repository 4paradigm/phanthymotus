"""Loopback WebXR enrollment/transport with RecordingAdapter, never a robot.

The optional Chromium run trusts only a generated test site's TLS via its test
context. Production code has no certificate bypass. XR poses are synthetic;
browser WebRTC, enrollment, TLS/WSS and the Python runtime are real. The input
clock is synthetic too: desktop timer scheduling is not an XR frame deadline.
"""
import asyncio
import base64
import datetime
import ipaddress
import os
from contextlib import asynccontextmanager
from pathlib import Path
import ssl
import sys
import uuid

import aiohttp
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

sys.path.insert(0, str(Path(__file__).parents[1] / 'plugins'))
from teleop.capture import CaptureManager, CaptureError
from teleop.capture_server import CaptureWssServer, capture_certificate_base64
from teleop.descriptor import CAPTURE_PROTOCOL, RTC_FRAME_PROTOCOL
from teleop.dispatch import RecordingAdapter
from teleop.protocol import TicketCodec, TicketVerifier
from teleop.rtc import RtcManager
from teleop.runtime import TeleopRuntime


@asynccontextmanager
async def site(tmp_path):
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'localhost')])
    now=datetime.datetime.now(datetime.timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=1))
          .not_valid_after(now+datetime.timedelta(hours=1))
          .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),False)
          .sign(key,hashes.SHA256()))
    certificate,private=tmp_path/'cert.pem',tmp_path/'key.pem'
    certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    private.chmod(0o600)
    config={'discovery_enabled':False,'port':0,'bind_host':'127.0.0.1',
            'public_wss_url':'wss://127.0.0.1:0/ws/teleop-capture','tls_cert_file':str(certificate),'tls_key_file':str(private)}
    runtime=TeleopRuntime(mode='shadow',adapter=RecordingAdapter(),lease_timeout_ms=15000,pose_timeout_ms=1000,auto_watchdog=False)
    codec=TicketCodec('webxr-loopback-only-'+str(uuid.uuid4()))
    rtc=RtcManager(runtime,TicketVerifier(codec))
    manager=CaptureManager(runtime,rtc,codec,state_file=tmp_path/'state.json',public_wss_url=config['public_wss_url'],
                           ca_certificate_base64=capture_certificate_base64(config),presence_interval_ms=250,presence_timeout_ms=2000)
    server=CaptureWssServer(manager,config)
    try:
        await server.start()
        port=server._site._server.sockets[0].getsockname()[1]
        origin=f'https://127.0.0.1:{port}'
        manager._public_wss_url=server.enrollment.public_wss_url=origin.replace('https:','wss:')+'/ws/teleop-capture'
        runtime.prepare_local_session()
        yield origin,server,manager,runtime,ssl.create_default_context(cafile=str(certificate))
    finally:
        await server.close();await rtc.close_all();runtime.close()


def hello(kind,**credentials):
    return {**credentials,'capture_protocol':CAPTURE_PROTOCOL,'frame_protocol':RTC_FRAME_PROTOCOL,
            'client_kind':kind,'app_version':'webxr-0.1-operator1-ikview2'}


def test_assets_config_and_cross_origin_enrollment_are_bounded(tmp_path):
    async def run():
        async with site(tmp_path) as (origin,server,manager,runtime,tls),aiohttp.ClientSession() as client:
            for path in ('/webxr','/webxr/','/webxr/app.mjs','/webxr/capture.mjs','/webxr/frame.mjs','/webxr/view.mjs','/webxr/manifest.webmanifest','/webxr/icon.svg'):
                async with client.get(origin+path,ssl=tls) as r:
                    assert r.status==200
                    assert r.headers['Cache-Control']=='no-store'
                    assert "frame-ancestors 'none'" in r.headers['Content-Security-Policy']
                    assert 'Access-Control-Allow-Origin' not in r.headers
                    assert await r.read()
            async with client.get(origin+'/webxr/config',ssl=tls) as r:
                data=await r.json()
                assert data['origin']==origin
                assert base64.b64decode(data['certificate_der_base64'])==server.enrollment.certificate
                assert set(data)=={'schema','origin','wss_url','device_id','certificate_der_base64'}
            assert (await client.get(origin+'/webxr/unknown.js',ssl=tls)).status==404
            assert (await client.get(origin+'/webxr/app.mjs?path=private',ssl=tls)).status==404
            await server.enrollment.open()
            for foreign in ('https://untrusted.example','null',origin+'.untrusted.example'):
                async with client.post(origin+'/pairing/request',json={},headers={'Origin':foreign},ssl=tls) as r:
                    assert r.status==403 and (await r.json())['error']=='pairing_origin_invalid'
                with pytest.raises(aiohttp.WSServerHandshakeError) as error:
                    await client.ws_connect(origin.replace('https:','wss:')+'/ws/teleop-capture',headers={'Origin':foreign},ssl=tls)
                assert error.value.status==403
            assert server.enrollment.status()['pending'] is None
            assert (await manager.status())['paired_devices']==0
            assert not runtime.status()['publisher_present']
    asyncio.run(run())


def test_webxr_credentials_persist_and_cannot_impersonate_native(tmp_path):
    async def run():
        async with site(tmp_path) as (_,server,manager,runtime,_):
            pairing=await manager.create_pairing()
            connection,ack=await manager.connect(hello('webxr',type='pair',pairing_id=pairing['pairing_id'],pairing_code=pairing['pairing_code']))
            await manager.disconnect(connection)
            # Loading the same state retains the browser kind, instead of accepting a native hello.
            restored=CaptureManager(runtime,manager._rtc,manager._ticket_codec,state_file=tmp_path/'state.json')
            credential={'type':'credential','capture_id':ack['capture_id'],'capture_credential':ack['capture_credential']}
            with pytest.raises(CaptureError,match='capture_credential_invalid'):
                await restored.connect(hello('native_openxr',**credential))
            connection,result=await restored.connect(hello('webxr',**credential))
            assert result['type']=='connected'
            await restored.disconnect(connection)
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('RUN_WEBXR_BROWSER')!='1',reason='requires host Chromium/Playwright; never a headset')
def test_chromium_pair_real_rtc_frames_and_focus_loss(tmp_path,monkeypatch):
    from playwright.async_api import async_playwright
    import aioice.ice
    monkeypatch.setattr(aioice.ice,'get_host_addresses',lambda use_ipv4,use_ipv6:['127.0.0.1'] if use_ipv4 else [])
    async def wait_until(predicate,timeout=8):
        deadline=asyncio.get_running_loop().time()+timeout
        while not predicate():
            if asyncio.get_running_loop().time()>deadline:raise AssertionError('condition timeout')
            await asyncio.sleep(.02)
    async def browser_wait(page, expression, timeout=15):
        deadline=asyncio.get_running_loop().time()+timeout
        while not await page.evaluate(expression):
            if asyncio.get_running_loop().time()>deadline or await page.evaluate('window.fixtureLoss===true'):
                raise AssertionError('browser condition timeout: '+await page.evaluate("window.transport?.status || document.querySelector('#status').textContent"))
            await asyncio.sleep(.05)
    async def run():
        async with site(tmp_path) as (origin,server,manager,runtime,_),async_playwright() as playwright:
            browser=await playwright.chromium.launch(headless=True,executable_path=os.environ.get('CHROME_PATH'),
                args=['--disable-features=WebRtcHideLocalIpsWithMdns',
                      '--disable-background-timer-throttling','--disable-renderer-backgrounding',
                      '--disable-backgrounding-occluded-windows'])
            try:
                context=await browser.new_context(ignore_https_errors=True)
                page=await context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                await page.goto(origin+'/webxr/')
                await browser_wait(page, "document.querySelector('#support').textContent.includes('不支持')")
                assert await page.locator('#pair').is_disabled()
                # Feature probe only. No XRSession or real head/controller API is emulated.
                await context.add_init_script("Object.defineProperty(navigator,'xr',{value:{isSessionSupported:async()=>true}})")
                from teleop.operator_session import OperatorCommands
                commands=[]
                def execute(action,cancel,connected):
                    assert connected()
                    commands.append(action)
                    return {'state':'ready'}
                manager.operator_commands=OperatorCommands(manager,execute)
                manager.visualization_provider=lambda:{'available':False,'operator':{'armed':True,'state':'ready','mode':'shadow'}}
                await server.enrollment.open()
                await page.reload()
                await page.locator('#pair').click()
                await wait_until(lambda:server.enrollment.status()['pending'] is not None)
                pending=server.enrollment.status()['pending']
                await page.locator('#confirm').wait_for(state='visible')
                assert (await page.locator('#fingerprint').inner_text()).replace(' ','')==pending['fingerprint']
                await page.locator('#confirm').click()
                server.enrollment.decide(pending['request_id'],pending['fingerprint'],True)
                await browser_wait(page, "!!localStorage.getItem('motus.webxr.capture.v1')")
                await page.screenshot(path=str(tmp_path/'webxr-paired.png'),full_page=True)
                await page.locator('#disconnect').click()
                await wait_until(lambda:manager._connection is None)
                # Synthetic samples use a controlled input clock: headless 2D
                # scheduling can stall while initializing RTC. Production uses
                # performance.now(); webxr.test.mjs checks its 250 ms boundary.
                # Use the production CaptureClient with actual browser WebRTC;
                # only head/controller samples are synthetic, with no navigator.xr shim.
                await page.evaluate('''async () => {
                  const {CaptureClient}=await import('/webxr/capture.mjs');
                  const {poseOf,controllerOf}=await import('/webxr/frame.mjs');
                  const config=await (await fetch('/webxr/config')).json();
                  window.fixtureGrip=0;window.fixtureLoss=false;window.fixtureTime=100;
                  const pose={emulatedPosition:false,transform:{position:{x:0,y:1,z:0},orientation:{x:0,y:0,z:0,w:1}}};
                  const controller=hand=>controllerOf({handedness:hand,targetRayMode:'tracked-pointer',gripSpace:{},gamepad:{mapping:'xr-standard',connected:true,axes:[0,0,0,0],buttons:[{value:0,pressed:false},{value:fixtureGrip,pressed:fixtureGrip===1},...Array.from({length:4},()=>({value:0,pressed:false}))]}},pose,hand);
                  window.transport=new CaptureClient({url:config.wss_url,now:()=>fixtureTime,onLoss:()=>{fixtureLoss=true;}});
                  transport.connect(JSON.parse(localStorage.getItem('motus.webxr.capture.v1')).credentials);
                  window.pump=setInterval(()=>{fixtureTime+=20;transport.submit({head:poseOf(pose),left:controller('left'),right:controller('right')});},20);
                }''')
                await browser_wait(page, 'transport.authenticated')
                await page.evaluate('transport.focus(true)')
                await browser_wait(page, "transport.pose?.readyState==='open' && transport.control?.readyState==='open'")
                assert not commands  # Pairing, reconnect and RTC setup never issue Start.
                await browser_wait(page, 'transport.operator?.armed===true')
                assert await page.evaluate("transport.command('start')")
                await browser_wait(page, 'transport.allowed')
                assert commands==['start']
                await asyncio.sleep(.1);await page.evaluate('fixtureGrip=1')
                await wait_until(lambda:runtime.status()['state']=='active_shadow')
                assert not runtime.status()['publisher_present']
                await page.evaluate('transport.focus(false);clearInterval(pump)')
                await wait_until(lambda:runtime.status()['state']=='hold')
                assert await page.evaluate('fixtureLoss && !transport.allowed && !transport.socket')
                # Compile and draw the production HUD using real WebGL. The XR
                # framebuffer/views are fixtures; this is not headset rendering evidence.
                assert await page.evaluate("""async () => {
                  const {XrPanel}=await import('/webxr/view.mjs');
                  const canvas=document.createElement('canvas');canvas.width=800;canvas.height=400;
                  const hud=new XrPanel(canvas),matrix=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
                  hud.draw({renderState:{baseLayer:{framebuffer:null,getViewport:()=>({x:0,y:0,width:800,height:400})}}},
                    {transform:{matrix},views:[{projectionMatrix:matrix,transform:{inverse:{matrix}}}]},transport,[]);
                  const ok=hud.gl.getError()===hud.gl.NO_ERROR;hud.dispose();return ok;
                }""")
                assert not errors
            finally:
                await browser.close()
    asyncio.run(run())
