"""Same-origin WebXR assets on the existing Capture TLS listener.

Browser trust comes from a valid HTTPS certificate. No TLS bypass, arbitrary
proxy target, additional authority or independent control service is introduced.
"""
import base64
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

ASSETS = Path(__file__).with_name('webxr')
FILES = {'index.html': 'text/html', 'app.mjs': 'text/javascript',
         'capture.mjs': 'text/javascript', 'frame.mjs': 'text/javascript',
         'view.mjs': 'text/javascript', 'manifest.webmanifest': 'application/manifest+json',
         'icon.svg': 'image/svg+xml'}
HEADERS = {'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
           'X-Content-Type-Options': 'nosniff',
           'Permissions-Policy': 'xr-spatial-tracking=(self)',
           'Content-Security-Policy': "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; "
                                     "connect-src 'self'; img-src 'self'; manifest-src 'self'; "
                                     "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"}


def browser_origin_allowed(request, enrollment):
    """Native clients omit Origin. Browser requests must match the configured site."""
    origin = request.headers.get('Origin')
    if origin is None:
        return True
    parsed = urlsplit(enrollment.public_wss_url or '')
    expected = 'https://' + parsed.netloc
    return bool(parsed.netloc) and origin == expected


def register_webxr(app, enrollment):
    async def asset(request):
        name = request.match_info.get('asset', 'index.html')
        if name not in FILES or request.query_string:
            raise web.HTTPNotFound()
        # Fixed allowlist; no directory listing or user-selected filesystem paths.
        return web.Response(body=(ASSETS / name).read_bytes(), content_type=FILES[name], headers=HEADERS)

    async def config(request):
        parsed = urlsplit(enrollment.public_wss_url or '')
        if parsed.scheme != 'wss' or not parsed.netloc or request.query_string:
            raise web.HTTPServiceUnavailable()
        return web.json_response({'schema': 'motus.teleop.webxr.v1',
            'origin': 'https://' + parsed.netloc, 'wss_url': enrollment.public_wss_url,
            'device_id': enrollment.device_id,
            'certificate_der_base64': base64.b64encode(enrollment.certificate).decode('ascii')}, headers=HEADERS)

    app.router.add_get('/webxr/config', config)
    app.router.add_get('/webxr', asset)
    app.router.add_get('/webxr/', asset)
    app.router.add_get('/webxr/{asset}', asset)
