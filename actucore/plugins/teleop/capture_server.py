"""Dedicated TLS WebSocket listener for G1 Capture control."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aiohttp import web
from cryptography import x509

from .enrollment import Enrollment
from .webxr import browser_origin_allowed, register_webxr
from .onboarding import APK_DIRECTORY, APK_FILENAME, MIME_TYPE, package_metadata

from .capture import (
    MAX_CAPTURE_CA_PEM_BYTES,
    MAX_CAPTURE_MESSAGE_BYTES,
    CaptureConnection,
    CaptureError,
    CaptureManager,
)


class CaptureTlsError(RuntimeError):
    """Capture listener cannot establish its authenticated TLS boundary."""


def validated_capture_wss_url(value: Any, *, expected_port: int | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise CaptureTlsError("Capture public URL must be a bounded wss:// URL")
    parsed = urlsplit(value)
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise CaptureTlsError("Capture public URL contains an invalid port") from exc
    if (
        parsed.scheme != "wss"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/ws/teleop-capture"
        or parsed_port is None
        or (expected_port is not None and parsed_port != expected_port)
    ):
        raise CaptureTlsError(
            "Capture public URL must be wss://<host>:<port>/ws/teleop-capture"
        )
    return value


def capture_certificate_base64(config: dict) -> str:
    certificate_path = Path(str(config.get(
        "tls_cert_file",
        "/etc/motus-teleop-capture-tls/cert.pem",
    )))
    if not certificate_path.is_file() or certificate_path.is_symlink():
        raise CaptureTlsError(f"Capture TLS certificate is missing: {certificate_path}")
    try:
        with certificate_path.open("rb") as certificate_file:
            certificate = certificate_file.read(MAX_CAPTURE_CA_PEM_BYTES + 1)
    except OSError as exc:
        raise CaptureTlsError(
            f"Capture TLS certificate is unreadable: {certificate_path}"
        ) from exc
    if (
        not certificate
        or len(certificate) > MAX_CAPTURE_CA_PEM_BYTES
        or b"-----BEGIN CERTIFICATE-----" not in certificate
        or b"PRIVATE KEY" in certificate
    ):
        raise CaptureTlsError(
            "Capture TLS certificate must contain only a bounded public PEM chain"
        )
    return base64.b64encode(certificate).decode("ascii")


def build_capture_ssl_context(config: dict) -> ssl.SSLContext:
    """Validate SAN and load the independent Capture certificate/key pair."""

    port = int(config.get("port", 15741))
    public_wss_url = validated_capture_wss_url(
        config.get("public_wss_url"),
        expected_port=port,
    )
    capture_certificate_base64(config)
    certificate_path = Path(str(config.get(
        "tls_cert_file",
        "/etc/motus-teleop-capture-tls/cert.pem",
    )))
    key_path = Path(str(config.get(
        "tls_key_file",
        "/etc/motus-teleop-capture-tls/key.pem",
    )))
    forbidden_core_roots = (
        Path("/etc/motus-core-certs"),
        Path("/opt/phanthy-motus/data/certs"),
    )
    for candidate in (certificate_path, key_path):
        for core_key_root in forbidden_core_roots:
            try:
                candidate.resolve(strict=False).relative_to(core_key_root)
            except ValueError:
                continue
            raise CaptureTlsError(
                "Capture TLS material must not reuse the Agent Core certificate directory"
            )
    if not key_path.is_file() or key_path.is_symlink():
        raise CaptureTlsError(f"Capture TLS private key is missing: {key_path}")
    if certificate_path.resolve() == key_path.resolve():
        raise CaptureTlsError("Capture TLS certificate and private key must be separate files")
    try:
        certificate = x509.load_pem_x509_certificate(certificate_path.read_bytes())
        alternatives = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
    except (OSError, ValueError, x509.ExtensionNotFound) as exc:
        raise CaptureTlsError(
            "Capture TLS certificate must contain a valid SAN extension"
        ) from exc
    hostname = urlsplit(public_wss_url).hostname
    assert hostname is not None
    try:
        requested_ip = ipaddress.ip_address(hostname)
    except ValueError:
        requested_ip = None
    if requested_ip is not None:
        matches_host = requested_ip in alternatives.get_values_for_type(x509.IPAddress)
    else:
        normalized_host = hostname.rstrip(".").lower()
        matches_host = any(
            normalized_host == candidate.rstrip(".").lower()
            or (
                candidate.startswith("*.")
                and normalized_host.endswith(candidate[1:].lower())
                and normalized_host.count(".") == candidate.count(".")
            )
            for candidate in alternatives.get_values_for_type(x509.DNSName)
        )
    if not matches_host:
        raise CaptureTlsError(
            "Capture TLS certificate SAN does not match capture.public_wss_url host"
        )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        context.load_cert_chain(str(certificate_path), str(key_path))
    except (OSError, ssl.SSLError) as exc:
        raise CaptureTlsError(
            "Capture TLS certificate/key pair is invalid"
        ) from exc
    return context


class _DuplicateJsonField(ValueError):
    pass


def capture_json(raw: str) -> dict:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_CAPTURE_MESSAGE_BYTES:
        raise CaptureError("capture_message_invalid")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJsonField(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (
        UnicodeEncodeError,
        json.JSONDecodeError,
        _DuplicateJsonField,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise CaptureError("capture_message_invalid") from exc
    if not isinstance(value, dict):
        raise CaptureError("capture_message_invalid")
    return value


async def _ws_text(
    websocket: web.WebSocketResponse,
    *,
    timeout: float | None = None,
) -> str:
    try:
        message = await websocket.receive(timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise CaptureError("capture_auth_timeout", status=408) from exc
    if message.type != web.WSMsgType.TEXT or not isinstance(message.data, str):
        if message.type in {
            web.WSMsgType.CLOSE,
            web.WSMsgType.CLOSING,
            web.WSMsgType.CLOSED,
        }:
            raise EOFError
        raise CaptureError("capture_message_invalid")
    return message.data


CAPTURE_KEY = web.AppKey("teleop_capture_manager", CaptureManager)


async def visualization_stream(websocket, provider):
    """Independent 30 Hz latest snapshot; one computation in flight, no backlog.

    Slow rendering reduces display rate only; heartbeat and pose handling never
    await FK. No catch-up frames, command publishing, or authority mutation.
    """
    loop = asyncio.get_running_loop()
    while not websocket.closed:
        started = loop.time()
        try:
            visual = await asyncio.to_thread(provider)
        except Exception:
            visual = {'schema': 'motus.tianyi-visualization.v1',
                      'available': False, 'reason': 'visualization_unavailable'}
        if websocket.closed:
            return
        try:
            await asyncio.wait_for(websocket.send_json(
                {'type': 'visualization', 'visualization': visual}), timeout=.1)
        except (asyncio.TimeoutError, ConnectionError, RuntimeError):
            # Stop optional diagnostics if the socket cannot drain. Do not
            # enqueue history or interfere with the control connection.
            return
        await asyncio.sleep(max(0., 1/30 - (loop.time()-started)))


async def capture_websocket_handler(request: web.Request) -> web.StreamResponse:
    enrollment = request.app.get(ENROLLMENT_KEY)
    if request.headers.get("Origin") is not None and (enrollment is None or not browser_origin_allowed(request, enrollment)):
        raise web.HTTPForbidden()
    manager = request.app[CAPTURE_KEY]
    websocket = web.WebSocketResponse(max_msg_size=MAX_CAPTURE_MESSAGE_BYTES)
    await websocket.prepare(request)
    if request.query_string:
        await websocket.send_json({"type": "error", "code": "capture_query_forbidden"})
        await websocket.close(code=4400, message=b"capture_query_forbidden")
        return websocket

    connection: CaptureConnection | None = None
    receive_task: asyncio.Task | None = None
    event_task: asyncio.Task | None = None
    visual_task: asyncio.Task | None = None
    try:
        first = capture_json(await _ws_text(websocket, timeout=5.0))
        connection, acknowledgement = await manager.connect(first)
        version = first.get('app_version', '')
        # This release changed its version label, not its existing wire features.
        # Keep opt-in exact for unsuffixed releases: older clients reject extra messages.
        onboarding_client = version == '0.3.18-onboarding1'
        if (onboarding_client or version.endswith('-operator1-ikview2')) and getattr(manager,'operator_commands',None) is not None:
            acknowledgement['operator_control']={'version':1,'connection_id':connection.connection_id}
        await websocket.send_json(acknowledgement)
        provider = getattr(manager, 'visualization_provider', None)
        if (onboarding_client or version.endswith('-ikview2')) and provider is not None:
            visual_task = asyncio.create_task(visualization_stream(websocket, provider))
        receive_task = asyncio.create_task(_ws_text(websocket))
        event_task = asyncio.create_task(connection.events.get())
        while True:
            done, _ = await asyncio.wait(
                {receive_task, event_task},
                timeout=0.25,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                if manager.presence_expired(connection):
                    manager.disconnect_immediate(connection)
                    await websocket.send_json({
                        "type": "error",
                        "code": "capture_presence_timeout",
                    })
                    await websocket.close(
                        code=4408,
                        message=b"capture_presence_timeout",
                    )
                    break
                continue
            if event_task in done:
                event = event_task.result()
                await websocket.send_json(event)
                if event.get("type") in {"capture_revoked", "capture_stale"}:
                    await websocket.close(code=4403, message=b"capture_revoked")
                    break
                event_task = asyncio.create_task(connection.events.get())
            if receive_task in done:
                incoming = capture_json(receive_task.result())
                message_type = incoming.get("type")
                if message_type == "presence":
                    response = await manager.presence(connection, incoming)
                    # Version-tag opt-in preserves the strict wire contract of old APKs.
                    provider = getattr(manager, 'visualization_provider', None)
                    if first.get('app_version', '').endswith('-ikview1') and provider is not None:
                        try:
                            visual = await asyncio.wait_for(asyncio.to_thread(provider), timeout=.025)
                        except Exception:
                            # Diagnostics are optional and must never break presence/authority.
                            visual = {'schema':'motus.g1-visualization.v1','available':False,
                                      'reason':'visualization_unavailable'}
                        if visual is not None:
                            response['visualization'] = visual
                elif message_type == "operator_command" and getattr(manager,'operator_commands',None) is not None:
                    try:response=await manager.operator_commands.submit(connection,incoming)
                    except ValueError as exc:
                        response={'type':'operator_result','request_id':incoming.get('request_id'),
                                  'state':'failed','error':str(exc)}
                elif message_type == "signaling_offer":
                    response = await manager.signaling_offer(connection, incoming)
                else:
                    raise CaptureError("capture_message_invalid")
                await websocket.send_json(response)
                receive_task = asyncio.create_task(_ws_text(websocket))
    except EOFError:
        pass
    except asyncio.CancelledError:
        raise
    except CaptureError as exc:
        if connection is not None:
            manager.disconnect_immediate(connection)
        if not websocket.closed:
            await websocket.send_json({"type": "error", "code": exc.code})
            await websocket.close(
                code=4403 if exc.status in {401, 403} else 4400,
                message=exc.code.encode("ascii", errors="ignore")[:120],
            )
    finally:
        if connection is not None:
            manager.disconnect_immediate(connection)
        tasks = [
            task
            for task in (receive_task, event_task, visual_task)
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if connection is not None:
            await manager.disconnect(connection)
    return websocket


ENROLLMENT_KEY = web.AppKey("teleop_enrollment", Enrollment)


async def enrollment_handler(request):
    try:
        if request.query_string or request.content_length is None or request.content_length > 4096:
            raise CaptureError('pairing_request_invalid')
        enrollment = request.app[ENROLLMENT_KEY]
        if not browser_origin_allowed(request, enrollment):
            raise CaptureError('pairing_origin_invalid', status=403)
        data = capture_json(await request.text())
        operation = request.match_info['operation']
        method = {'request': enrollment.request, 'poll': enrollment.poll, 'invite': enrollment.redeem_invitation}[operation]
        result = await method(data)
        return web.json_response(result, headers={'Cache-Control': 'no-store'})
    except CaptureError as exc:
        return web.json_response({'error': exc.code}, status=exc.status)


async def package_handler(request):
    if request.query_string:
        raise web.HTTPBadRequest()
    metadata = await asyncio.to_thread(package_metadata)
    if request.path == '/onboarding/package':
        return web.json_response(metadata, headers={'Cache-Control': 'no-store'})
    if not metadata['available']:
        return web.json_response(metadata, status=503, headers={'Cache-Control': 'no-store'})
    return web.FileResponse(APK_DIRECTORY / APK_FILENAME, headers={
        'Content-Type': MIME_TYPE, 'Content-Disposition': 'attachment; filename="pico.apk"',
        'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})


def create_capture_app(manager: CaptureManager, enrollment=None) -> web.Application:
    app = web.Application(client_max_size=MAX_CAPTURE_MESSAGE_BYTES)
    app[CAPTURE_KEY] = manager
    app.router.add_get('/onboarding/package', package_handler)
    app.router.add_get('/onboarding/apk', package_handler)
    if enrollment is not None:
        app[ENROLLMENT_KEY] = enrollment
        register_webxr(app, enrollment)
        app.router.add_post('/pairing/{operation:request|poll|invite}', enrollment_handler)
    app.router.add_get("/ws/teleop-capture", capture_websocket_handler)
    return app


class CaptureWssServer:
    """Lifecycle wrapper started on the teleop service's asyncio loop."""

    def __init__(self, manager: CaptureManager, config: dict):
        self._manager = manager
        self._config = dict(config)
        self._ssl_context = build_capture_ssl_context(self._config)
        self.enrollment = Enrollment(manager, capture_certificate_base64(config), public_wss_url=config.get("public_wss_url"))
        self._discovery = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    def installation_info(self):
        parsed = urlsplit(self._config['public_wss_url'])
        return {'capture_origin': 'https://'+parsed.netloc,
            'certificate_sha256': self.enrollment.device_id,
            'package_path': '/onboarding/package', 'apk_path': '/onboarding/apk',
            'package': package_metadata(), 'webxr_url': 'https://'+parsed.netloc+'/webxr/'}

    async def start(self) -> None:
        if self._runner is not None:
            return
        runner = web.AppRunner(create_capture_app(self._manager, self.enrollment))
        await runner.setup()
        try:
            site = web.TCPSite(
                runner,
                str(self._config.get("bind_host", "0.0.0.0")),
                int(self._config.get("port", 15741)),
                ssl_context=self._ssl_context,
            )
            await site.start()
            if self._config.get('discovery_enabled', True):
                from zeroconf import ServiceInfo
                from zeroconf.asyncio import AsyncZeroconf
                import socket
                hostname = urlsplit(self._config['public_wss_url']).hostname
                addresses = list({socket.inet_pton(family, item[4][0])
                    for item in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
                    for family in (item[0],)})
                service_type = '_motus-teleop._tcp.local.'
                self._advert = ServiceInfo(service_type,
                    self.enrollment.device_id[:24]+'.'+service_type,
                    addresses=addresses, port=int(self._config.get('port',15741)),
                    properties={'id':self.enrollment.device_id,
                        'name':str(self._config.get('display_name','Tianyi Teleop'))[:64], 'version':'1'})
                self._discovery = AsyncZeroconf()
                await self._discovery.async_register_service(self._advert)
        except Exception:
            if self._discovery:
                await self._discovery.async_close()
                self._discovery = None
            await runner.cleanup()
            raise
        self._runner = runner
        self._site = site

    async def close(self) -> None:
        runner = self._runner
        self._runner = None
        self._site = None
        self.enrollment.revoke_invitation()
        self.enrollment.pending = None
        self.enrollment.deadline = 0
        if self._discovery:
            try:
                await self._discovery.async_unregister_service(self._advert)
            finally:
                await self._discovery.async_close()
                self._discovery = None
        if runner is not None:
            await runner.cleanup()


__all__ = [
    "CAPTURE_KEY",
    "CaptureTlsError",
    "CaptureWssServer",
    "build_capture_ssl_context",
    "capture_certificate_base64",
    "capture_json",
    "capture_websocket_handler",
    "create_capture_app",
    "validated_capture_wss_url",
]
