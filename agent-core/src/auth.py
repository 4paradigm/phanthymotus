"""
auth.py — Access token authentication for Dashboard and API.

Token source: /opt/phanthy-motus/.env file (volume-mounted from host).
Read once at startup. Restart container to apply .env changes.
"""

import pathlib

from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse


_ENV_PATH = pathlib.Path('/opt/phanthy-motus/.env')
# Fallback for local dev
_ENV_PATH_DEV = pathlib.Path('.env')

_token: str = ''
_auth_enabled: bool = False


def _read_token_from_env() -> str:
    """Read ACCESS_TOKEN from .env file."""
    env_file = _ENV_PATH if _ENV_PATH.exists() else _ENV_PATH_DEV
    if not env_file.exists():
        return ''
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith('ACCESS_TOKEN=') and not line.startswith('#'):
            return line.split('=', 1)[1].strip()
    return ''


def init():
    """Read token from .env and cache it. Called once at startup."""
    global _token, _auth_enabled
    _token = _read_token_from_env()
    _auth_enabled = bool(_token)
    if _auth_enabled:
        print(f'[auth] Token authentication enabled')
    else:
        print(f'[auth] No ACCESS_TOKEN in .env — authentication disabled')


def get_token() -> str:
    return _token


def is_enabled() -> bool:
    return _auth_enabled


def verify(token: str) -> bool:
    if not _auth_enabled:
        return True
    return bool(token) and token == _token


async def auth_middleware(request: Request, call_next):
    """FastAPI middleware: enforce token auth on /api/* and /ws/* paths."""
    if not _auth_enabled:
        return await call_next(request)

    path = request.url.path

    # Static files and HTML — no auth needed
    if not path.startswith('/api/') and not path.startswith('/ws/'):
        return await call_next(request)

    # Exempt paths
    if path == '/api/auth/verify':
        return await call_next(request)
    if path.startswith('/api/channel/webhook/'):
        return await call_next(request)
    # MCP registration from driver containers
    if path == '/api/mcp' and request.method == 'POST':
        return await call_next(request)
    # /ws/mic stays open (internal browser mic)
    if path == '/ws/mic':
        return await call_next(request)

    # Check token
    token = _extract_token(request)
    if not verify(token):
        return JSONResponse(status_code=401, content={'detail': 'Unauthorized'})

    return await call_next(request)


def check_ws_token(websocket: WebSocket) -> bool:
    """Check token in WebSocket query params."""
    if not _auth_enabled:
        return True
    token = websocket.query_params.get('token', '')
    return verify(token)


def _extract_token(request: Request) -> str:
    """Extract token from Authorization header or query param."""
    auth = request.headers.get('authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:]
    return request.query_params.get('token', '')

