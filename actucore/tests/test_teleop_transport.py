"""Real TLS/WSS and local aiortc channels; RecordingAdapter, no ROS or robot."""
import asyncio
import base64
import hashlib
import datetime
import ipaddress
import json
from pathlib import Path
import ssl
import sys
import time
import uuid
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'plugins'))
from aiohttp import ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from cryptography import x509
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.x509.oid import NameOID
from teleop.capture import CaptureManager, CaptureError
from teleop.capture_server import CaptureWssServer, capture_certificate_base64
from teleop.descriptor import CAPTURE_PROTOCOL,RTC_FRAME_PROTOCOL
from teleop.dispatch import RecordingAdapter
from teleop.protocol import TicketCodec,TicketVerifier
from teleop.rtc import RtcManager
from teleop.runtime import TeleopRuntime
from test_teleop import frame


def test_omitted_capture_port_matches_public_contract_for_tls_and_listener(tmp_path,monkeypatch):
    from test_teleop_site import site_configuration
    import teleop.capture_server as server_module
    config=site_configuration(tmp_path,monkeypatch)['capture']
    config.pop('port')
    requested=[]
    class Site:
        def __init__(self,runner,host,port,ssl_context):
            assert isinstance(ssl_context,ssl.SSLContext)
            requested.append(port)
        async def start(self):pass
        async def stop(self):pass
    monkeypatch.setattr(server_module.web,'TCPSite',Site)
    async def run():
        server=CaptureWssServer(CaptureManager(None,None,None),config)
        await server.start()
        await server.close()
    asyncio.run(run())
    assert requested==[15741]


@pytest.mark.parametrize('robot_profile',['tianyi2','g1_23'])
@pytest.mark.parametrize('visualization',[False,True,'error','stream'])
def test_real_wss_pair_rtc_pose_disconnect_and_credential_reconnect(tmp_path,robot_profile,visualization,monkeypatch):
    # Local protocol coverage uses real TLS/ICE/DTLS/SCTP sockets, not host
    # LAN/VPN routing. Default aioice discovery excludes loopback and can select
    # interfaces on which OS policy rejects self-addressed UDP. Only discovery
    # is scoped here; hardware/LAN acceptance remains a separate test.
    import aioice.ice
    monkeypatch.setattr(aioice.ice, 'get_host_addresses',
                        lambda use_ipv4, use_ipv6: ['127.0.0.1'] if use_ipv4 else [])
    async def run():
        display_received=[]
        async def control_message(ws, timeout):
            while True:
                value=await ws.receive_json(timeout=timeout)
                if value['type']=='visualization':
                    assert visualization=='stream'
                    display_received.append(value)
                    continue
                return value
        key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        subject=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'localhost')])
        now=datetime.datetime.now(datetime.timezone.utc)
        cert=(x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=1))
              .not_valid_after(now+datetime.timedelta(days=1))
              .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),False)
              .sign(key,hashes.SHA256()))
        certificate=tmp_path/'cert.pem';private=tmp_path/'key.pem'
        certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        private.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
        private.chmod(0o600)
        cfg={'discovery_enabled':False,'port':0,'bind_host':'127.0.0.1','public_wss_url':'wss://127.0.0.1:0/ws/teleop-capture',
             'tls_cert_file':str(certificate),'tls_key_file':str(private)}
        from teleop.g1 import CAPABILITIES_G1
        runtime=TeleopRuntime(mode='shadow',adapter=RecordingAdapter(),capabilities=CAPABILITIES_G1 if robot_profile=='g1_23' else None,lease_timeout_ms=15000,
                              pose_timeout_ms=1000,auto_watchdog=False)
        codec=TicketCodec('offline-test-signing-key-'+str(uuid.uuid4()))
        rtc=RtcManager(runtime,TicketVerifier(codec))
        manager=CaptureManager(runtime,rtc,codec,state_file=tmp_path/'state.json',
            public_wss_url=cfg['public_wss_url'],ca_certificate_base64=capture_certificate_base64(cfg),
            presence_interval_ms=250,presence_timeout_ms=10000)
        def visual():
            if visualization=='error':raise RuntimeError('display_only_failure')
            return {'schema':'motus.g1-visualization.v1','available':False,'reason':'calibration_missing'}
        manager.visualization_provider=visual
        # The real released client must retain both display and operator controls.
        # No commands are sent in this transport test; the sentinel enables negotiation.
        manager.operator_commands=object()
        # Exercise the version actually sent by Android, not a synthetic opt-in.
        # V039 updated Gradle but left the native hello at a non-visual version.
        native_root=Path(__file__).parents[1]/'openxr_capture_native'
        native_source=(native_root/'app/src/main/cpp/android_main.cpp').read_text()
        native_version=native_source.split('constexpr char kAppVersion[] = "',1)[1].split('"',1)[0]
        gradle_version=(native_root/'app/build.gradle.kts').read_text().split('versionName = "',1)[1].split('"',1)[0]
        assert native_version == gradle_version
        app_version=native_version if visualization=='stream' else '0.3.11-ikview1' if visualization else 'test'
        server=CaptureWssServer(manager,cfg);peer=RTCPeerConnection(RTCConfiguration(iceServers=[]))
        heartbeat=None
        try:
            await server.start();port=server._site._server.sockets[0].getsockname()[1]
            runtime.prepare_local_session()
            ssl_client=ssl.create_default_context(cafile=str(certificate))
            async with ClientSession() as client:
                async def enroll(operation, payload):
                    async with client.post(f'https://127.0.0.1:{port}/pairing/{operation}',json=payload,ssl=ssl_client) as response:
                        return response.status, await response.json()
                pub=ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(serialization.Encoding.DER,serialization.PublicFormat.SubjectPublicKeyInfo)
                nonce=b'n'*32
                request={'device_name':'PICO test','public_key':base64.b64encode(pub).decode(),'nonce':base64.b64encode(nonce).decode()}
                assert (await enroll('request',request))[0]==403
                await server.enrollment.open()
                code,pending=await enroll('request',request);assert code==200
                assert (await enroll('request',request))[0]==429
                transcript=b'motus-enrollment-v1\0'+hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).digest()+hashlib.sha256(pub).digest()+nonce+base64.b64decode(pending['server_nonce'])
                fingerprint=hashlib.sha256(transcript).hexdigest()[:32].upper()
                assert fingerprint==pending['fingerprint']
                assert 'ticket' not in json.dumps(server.enrollment.status())
                poll={'request_id':pending['request_id'],'ticket':pending['ticket'],'confirm':True,'fingerprint':fingerprint}
                assert (await enroll('poll',{**poll,'ticket':'wrong'}))[0]==403
                assert (await enroll('poll',poll))[1]['state']=='pending'
                server.enrollment.decide(pending['request_id'],fingerprint,True)
                code,pairing=await enroll('poll',poll);assert code==200 and pairing['state']=='approved'
                assert (await enroll('poll',poll))[0]==403
                async with client.ws_connect(f'wss://127.0.0.1:{port}/ws/teleop-capture',ssl=ssl_client) as ws:
                    await ws.send_json(dict(type='pair',pairing_id=pairing['pairing_id'],pairing_code=pairing['pairing_code'],
                        capture_protocol=CAPTURE_PROTOCOL,frame_protocol=RTC_FRAME_PROTOCOL,client_kind='native_openxr',app_version=app_version))
                    paired=await control_message(ws, timeout=3);assert paired['type']=='paired',paired
                    assert ('operator_control' in paired) == (visualization=='stream')
                    if visualization=='stream': assert paired['operator_control']['version']==1
                    await ws.send_json(dict(type='presence',state='xr_standby',assignment_id=None))
                    assignment=None
                    for _ in range(3):
                        msg=await control_message(ws, timeout=3)
                        if msg['type']=='assignment':assignment=msg['assignment'];break
                    assert assignment and assignment['mode']=='shadow'
                    control=peer.createDataChannel('teleop-control',ordered=True)
                    pose=peer.createDataChannel('teleop-pose',ordered=False,maxRetransmits=0)
                    await peer.setLocalDescription(await peer.createOffer())
                    await ws.send_json({'type':'signaling_offer','assignment_id':assignment['id'],
                        'offer':{'type':'offer','sdp':peer.localDescription.sdp}})
                    answer=None
                    for _ in range(4):
                        msg=await control_message(ws, timeout=12)
                        if msg['type']=='signaling_answer':answer=msg;break
                    assert answer, msg
                    await peer.setRemoteDescription(RTCSessionDescription(**answer['answer']))
                    deadline=time.monotonic()+5
                    while pose.readyState!='open' and time.monotonic()<deadline:await asyncio.sleep(.01)
                    assert pose.readyState=='open' and control.readyState=='open'
                    pose.send(json.dumps(frame(0)))
                    await asyncio.sleep(.05)
                    pose.send(json.dumps(frame(1,True,1)))
                    deadline=time.monotonic()+1
                    while runtime.status()['state']=='prepared_shadow' and time.monotonic()<deadline:
                        await asyncio.sleep(.005)
                    assert runtime.status()['state']=='active_shadow',runtime.status()
                    assert not runtime.status()['publisher_present']
                await asyncio.sleep(.1)
                assert runtime.status()['state']=='hold'
                runtime.release_local()
                runtime.prepare_local_session()
                runtime.bind_capture(paired['capture_id'])
                runtime._clock = lambda: time.monotonic() + 60
                assert runtime.watchdog_tick()['reason']=='lease_timeout'
                async with client.ws_connect(f'wss://127.0.0.1:{port}/ws/teleop-capture',ssl=ssl_client) as ws:
                    await ws.send_json(dict(type='credential',capture_id=paired['capture_id'],
                        capture_credential=paired['capture_credential'],capture_protocol=CAPTURE_PROTOCOL,
                        frame_protocol=RTC_FRAME_PROTOCOL,client_kind='native_openxr',app_version=app_version))
                    result=await control_message(ws, timeout=3);assert result['type']=='connected',result
                    assert runtime.status()['state']=='hold'  # Authentication cannot re-enable.
                    for _ in range(3):
                        await ws.send_json(dict(type='presence',state='xr_standby',assignment_id=None))
                        result=await control_message(ws, timeout=3)
                        assert result['type']=='presence_ack',result
                        assert ('visualization' in result)==(bool(visualization) and visualization!='stream')
                        if visualization and visualization!='stream':
                            assert result['visualization']['reason']==('visualization_unavailable' if visualization=='error' else 'calibration_missing')
                        assert not runtime.status()['authority_valid']
                        assert not runtime.status()['publisher_present']
                    if visualization=='stream':
                        # Stream works without waiting for another heartbeat.
                        for _ in range(5):
                            item=await ws.receive_json(timeout=.2)
                            assert item['type']=='visualization'
                            display_received.append(item)
                        assert len(display_received)>=5
                    runtime.release_local()
                    runtime.prepare_local_session()
                    await manager.issue_assignment_if_connected()
                    result=await control_message(ws, timeout=3)
                    assert result['type']=='assignment',result
                    await manager.revoke_headset()
                    assert (await manager.status())['paired_devices']==0
                clock=[time.monotonic()+10]
                server.enrollment.clock=lambda:clock[0]
                await server.enrollment.open()
                _,pending=await enroll('request',request)
                with pytest.raises(CaptureError):
                    server.enrollment.decide(pending['request_id'],'0'*32,True)
                poll={'request_id':pending['request_id'],'ticket':pending['ticket'],'confirm':True,'fingerprint':pending['fingerprint']}
                assert (await enroll('poll',{**poll,'fingerprint':'0'*32}))[0]==403
                server.enrollment.decide(pending['request_id'],pending['fingerprint'],False)
                assert (await enroll('poll',poll))[0]==403
                clock[0]+=3
                await server.enrollment.open()
                _,pending=await enroll('request',request)
                clock[0]+=121
                assert (await enroll('poll',{'request_id':pending['request_id'],'ticket':pending['ticket']}))[0]==403
                assert server.enrollment.status()['pending'] is None
        finally:
            await peer.close();await server.close();await rtc.close_all();runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize('presence_state',['browser_ready','xr_ended','error'])
def test_focus_loss_requires_new_xr_readiness_before_assignment(presence_state):
    from unittest.mock import AsyncMock
    from teleop.capture import CaptureConnection
    async def run():
        manager=CaptureManager(None,None,None)
        connection=CaptureConnection('test-capture','test-connection',ready_for_assignment=True)
        manager._connection=connection
        manager._bind_and_assign_locked=AsyncMock(return_value=True)
        await manager.presence(connection,dict(type='presence',state=presence_state,assignment_id=None))
        await manager.issue_assignment_if_connected()
        manager._bind_and_assign_locked.assert_not_awaited()
        assert not connection.ready_for_assignment
        await manager.presence(connection,dict(type='presence',state='xr_standby',assignment_id=None))
        manager._bind_and_assign_locked.assert_awaited_once_with(connection)
    asyncio.run(run())


def test_focus_loss_during_binding_does_not_issue_assignment():
    from types import SimpleNamespace
    from unittest.mock import Mock,patch
    from teleop.capture import CaptureConnection
    async def run():
        runtime=SimpleNamespace(bind_capture=Mock(),capture_hold=Mock())
        manager=CaptureManager(runtime,None,None)
        connection=CaptureConnection('test-capture','test-connection',ready_for_assignment=True)
        manager._connection=connection
        async def bind(*args):
            await manager.presence(connection,dict(type='presence',state='browser_ready',assignment_id=None))
            return {},7
        with patch('teleop.capture.asyncio.to_thread',side_effect=bind):
            await manager.issue_assignment_if_connected()
        assert connection.assignment is None and connection.events.empty()
        runtime.capture_hold.assert_called_once_with('test-capture',7,'capture_not_ready')
    asyncio.run(run())
