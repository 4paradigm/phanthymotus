"""
Authentication and role checks for the dashboard, API, and trusted drivers.

Human principals are configured with:

* ``ACCESS_TOKEN`` — backwards-compatible owner token.
* ``MOTUS_OPERATOR_TOKENS`` — named operator tokens.
* ``MOTUS_VIEWER_TOKENS`` — named viewer tokens.

Named token settings accept either a JSON object (recommended), for example
``{"alice": "secret"}``, or ``alice:secret,bob:secret``.

Drivers use either the backwards-compatible ``MOTUS_DRIVER_TOKEN`` credential
or an exact ``driver_id`` → token mapping in ``MOTUS_DRIVER_TOKENS``.  Dedicated
credentials take precedence over the legacy credential, so one Driver cannot
authenticate or receive Core credentials for another Driver identity.

Core-to-Driver WebRTC signaling likewise selects a dedicated value from
``MOTUS_TELEOP_TICKET_SECRETS`` or the legacy
``MOTUS_TELEOP_TICKET_SECRET`` fallback.  These secrets sign short-lived,
one-use offer tickets and are never returned to a browser.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import secrets
from dataclasses import dataclass
from typing import Mapping, Optional

from fastapi import HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse

_ENV_PATH = pathlib.Path('/opt/phanthy-motus/.env')
_ENV_PATH_DEV = pathlib.Path('.env')

_HUMAN_KEYS = ('ACCESS_TOKEN', 'MOTUS_OPERATOR_TOKENS', 'MOTUS_VIEWER_TOKENS')
_ALL_KEYS = (
    *_HUMAN_KEYS,
    'MOTUS_DRIVER_TOKEN',
    'MOTUS_DRIVER_TOKENS',
    'MOTUS_ENFORCE_DRIVER_AUTH',
    'MOTUS_TELEOP_TICKET_SECRET',
    'MOTUS_TELEOP_TICKET_SECRETS',
    'MOTUS_ENABLE_UNSAFE_DESKTOP_CODE_TOOLS',
)
_ROLE_LEVEL = {'viewer': 1, 'operator': 2, 'owner': 3}
_DRIVER_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$')
_DRIVER_BEARER_RE = re.compile(r'^[A-Za-z0-9._~+/=-]{24,4096}$')


@dataclass(frozen=True)
class Principal:
    """An authenticated dashboard/API identity."""

    id: str
    role: str

    def as_dict(self) -> dict:
        return {'id': self.id, 'role': self.role}


_principals_by_token: dict[str, Principal] = {}
_tokens_by_principal_id: dict[str, str] = {}
_driver_token: str = ''
_driver_tokens_by_id: dict[str, str] = {}
_driver_auth_enforced: bool = False
_teleop_ticket_secret: bytes = b''
_teleop_ticket_secrets_by_driver: dict[str, bytes] = {}
_auth_enabled: bool = False
_unsafe_desktop_code_tools_enabled: bool = False


def _secret_text_equal(left: object, right: object) -> bool:
    """Compare UTF-8 secrets without ``compare_digest`` Unicode failures."""

    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        return secrets.compare_digest(left.encode('utf-8'), right.encode('utf-8'))
    except UnicodeEncodeError:
        return False


def _valid_driver_id(driver_id: object) -> bool:
    return bool(
        isinstance(driver_id, str)
        and _DRIVER_ID_RE.fullmatch(driver_id)
    )


def _read_env_file() -> dict[str, str]:
    env_file = _ENV_PATH if _ENV_PATH.exists() else _ENV_PATH_DEV
    if not env_file.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip()
    return values


def _load_settings() -> dict[str, str]:
    values = _read_env_file()
    for key in _ALL_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def _parse_named_tokens(raw: str, role: str) -> list[tuple[str, str]]:
    if not raw.strip():
        return []

    def reject_duplicate_json_names(pairs):
        parsed_object = {}
        for name, token in pairs:
            if name in parsed_object:
                raise ValueError(f'{role} principal id is configured more than once: {name}')
            parsed_object[name] = token
        return parsed_object

    try:
        parsed = json.loads(raw, object_pairs_hook=reject_duplicate_json_names)
    except json.JSONDecodeError:
        parsed = None

    pairs: list[tuple[str, str]] = []
    if isinstance(parsed, dict):
        pairs = [(str(name).strip(), str(token).strip()) for name, token in parsed.items()]
    else:
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            separator = ':' if ':' in item else '=' if '=' in item else ''
            if not separator:
                raise ValueError(
                    f'{role} token entry must be name:token or a JSON object'
                )
            name, token = item.split(separator, 1)
            pairs.append((name.strip(), token.strip()))

    for name, token in pairs:
        if not name or not token:
            raise ValueError(f'{role} token entries require non-empty names and tokens')
    return pairs


def _register_human_token(token: str, principal: Principal) -> None:
    existing = _principals_by_token.get(token)
    if existing:
        if existing != principal:
            raise ValueError(
                f'authentication token is assigned to both {existing.id} and {principal.id}'
            )
        raise ValueError(f'authentication token is configured more than once: {principal.id}')
    existing_token = _tokens_by_principal_id.get(principal.id)
    if existing_token:
        raise ValueError(f'principal id is configured more than once: {principal.id}')
    _principals_by_token[token] = principal
    _tokens_by_principal_id[principal.id] = token


def _parse_driver_secret_map(
    raw: str,
    setting_name: str,
    *,
    minimum_bytes: int,
    secret_pattern: re.Pattern[str] | None = None,
) -> dict[str, str]:
    """Parse a strict JSON object of stable Driver identities to secrets."""

    if not isinstance(raw, str):
        raise TypeError(f'{setting_name} must be a JSON object string')
    if not raw.strip():
        return {}

    def reject_duplicate_driver_ids(pairs):
        parsed_object = {}
        for driver_id, token in pairs:
            if driver_id in parsed_object:
                raise ValueError(
                    f'driver credential id is configured more than once: {driver_id}'
                )
            parsed_object[driver_id] = token
        return parsed_object

    try:
        parsed = json.loads(raw, object_pairs_hook=reject_duplicate_driver_ids)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{setting_name} must be a valid JSON object') from exc
    if not isinstance(parsed, dict):
        raise TypeError(f'{setting_name} must be a JSON object')

    credentials: dict[str, str] = {}
    token_owners: dict[str, str] = {}
    for raw_driver_id, raw_token in parsed.items():
        if not isinstance(raw_driver_id, str) or not isinstance(raw_token, str):
            raise TypeError(f'{setting_name} requires string ids and secrets')
        driver_id = raw_driver_id.strip()
        token = raw_token.strip()
        if driver_id != raw_driver_id or not _DRIVER_ID_RE.fullmatch(driver_id):
            raise ValueError(f'invalid driver credential id: {raw_driver_id!r}')
        if (
            not token
            or token != raw_token
            or len(token) > 4096
            or '\r' in raw_token
            or '\n' in raw_token
        ):
            raise ValueError(f'invalid {setting_name} secret for driver id: {driver_id}')
        try:
            token_bytes = token.encode('utf-8')
        except UnicodeEncodeError:
            raise ValueError(
                f'invalid {setting_name} secret for driver id: {driver_id}'
            ) from None
        if len(token_bytes) < minimum_bytes:
            raise ValueError(
                f'{setting_name} secret for {driver_id} must contain at least '
                f'{minimum_bytes} bytes'
            )
        if secret_pattern is not None and not secret_pattern.fullmatch(token):
            raise ValueError(
                f'{setting_name} secret for {driver_id} must use 24-4096 '
                'restricted ASCII Bearer characters'
            )
        existing_owner = token_owners.get(token)
        if existing_owner is not None:
            raise ValueError(
                'driver credential is assigned to more than one id: '
                f'{existing_owner}, {driver_id}'
            )
        credentials[driver_id] = token
        token_owners[token] = driver_id
    return credentials


def _parse_driver_tokens(raw: str) -> dict[str, str]:
    return _parse_driver_secret_map(
        raw,
        'MOTUS_DRIVER_TOKENS',
        minimum_bytes=24,
        secret_pattern=_DRIVER_BEARER_RE,
    )


def _parse_driver_ticket_secrets(raw: str) -> dict[str, bytes]:
    parsed = _parse_driver_secret_map(
        raw,
        'MOTUS_TELEOP_TICKET_SECRETS',
        minimum_bytes=32,
    )
    return {driver_id: value.encode('utf-8') for driver_id, value in parsed.items()}


def init(settings: Optional[Mapping[str, str]] = None):
    """Load and cache authentication settings.

    ``settings`` is intentionally injectable so focused tests can avoid reading
    developer or deployment credentials from disk.
    """

    global _driver_token, _driver_auth_enforced, _teleop_ticket_secret, _auth_enabled
    global _unsafe_desktop_code_tools_enabled
    values = dict(settings) if settings is not None else _load_settings()
    snapshot = {
        'principals': dict(_principals_by_token),
        'principal_tokens': dict(_tokens_by_principal_id),
        'driver_token': _driver_token,
        'driver_tokens': dict(_driver_tokens_by_id),
        'driver_enforced': _driver_auth_enforced,
        'ticket_secret': _teleop_ticket_secret,
        'ticket_secrets': dict(_teleop_ticket_secrets_by_driver),
        'auth_enabled': _auth_enabled,
        'unsafe_code_tools': _unsafe_desktop_code_tools_enabled,
    }
    try:
        _apply_settings(values)
    except Exception:
        _principals_by_token.clear()
        _principals_by_token.update(snapshot['principals'])
        _tokens_by_principal_id.clear()
        _tokens_by_principal_id.update(snapshot['principal_tokens'])
        _driver_token = snapshot['driver_token']
        _driver_tokens_by_id.clear()
        _driver_tokens_by_id.update(snapshot['driver_tokens'])
        _driver_auth_enforced = snapshot['driver_enforced']
        _teleop_ticket_secret = snapshot['ticket_secret']
        _teleop_ticket_secrets_by_driver.clear()
        _teleop_ticket_secrets_by_driver.update(snapshot['ticket_secrets'])
        _auth_enabled = snapshot['auth_enabled']
        _unsafe_desktop_code_tools_enabled = snapshot['unsafe_code_tools']
        raise


def _apply_settings(values: Mapping[str, str]) -> None:
    """Validate and install one settings snapshot; ``init`` owns rollback."""

    global _driver_token, _driver_auth_enforced, _teleop_ticket_secret, _auth_enabled
    global _unsafe_desktop_code_tools_enabled

    _unsafe_desktop_code_tools_enabled = False
    _teleop_ticket_secret = b''
    _teleop_ticket_secrets_by_driver.clear()
    _principals_by_token.clear()
    _tokens_by_principal_id.clear()
    owner_token = values.get('ACCESS_TOKEN', '').strip()
    if owner_token:
        _register_human_token(owner_token, Principal(id='owner:legacy', role='owner'))

    for name, token in _parse_named_tokens(values.get('MOTUS_OPERATOR_TOKENS', ''), 'operator'):
        _register_human_token(token, Principal(id=f'operator:{name}', role='operator'))
    for name, token in _parse_named_tokens(values.get('MOTUS_VIEWER_TOKENS', ''), 'viewer'):
        _register_human_token(token, Principal(id=f'viewer:{name}', role='viewer'))

    _driver_tokens_by_id.clear()
    _driver_tokens_by_id.update(
        _parse_driver_tokens(values.get('MOTUS_DRIVER_TOKENS', ''))
    )
    _driver_token = values.get('MOTUS_DRIVER_TOKEN', '').strip()
    if _driver_token and authenticate(_driver_token) is not None:
        raise ValueError('MOTUS_DRIVER_TOKEN must not reuse a human access token')
    for driver_id, token in _driver_tokens_by_id.items():
        if authenticate(token) is not None:
            raise ValueError(
                f'driver credential must not reuse a human access token: {driver_id}'
            )
        if _driver_token and _secret_text_equal(token, _driver_token):
            raise ValueError(
                f'MOTUS_DRIVER_TOKEN must not reuse dedicated credential: {driver_id}'
            )
    enforce_raw = values.get('MOTUS_ENFORCE_DRIVER_AUTH', '').strip().lower()
    if enforce_raw not in ('', '0', 'false', 'no', 'off', '1', 'true', 'yes', 'on'):
        raise ValueError('MOTUS_ENFORCE_DRIVER_AUTH must be true or false')
    _driver_auth_enforced = enforce_raw in ('1', 'true', 'yes', 'on')
    if _driver_auth_enforced and not (_driver_token or _driver_tokens_by_id):
        raise ValueError(
            'MOTUS_ENFORCE_DRIVER_AUTH requires MOTUS_DRIVER_TOKEN or '
            'MOTUS_DRIVER_TOKENS'
        )
    ticket_secret = values.get('MOTUS_TELEOP_TICKET_SECRET', '')
    if not isinstance(ticket_secret, str):
        raise TypeError('MOTUS_TELEOP_TICKET_SECRET must be a string')
    try:
        ticket_secret_bytes = ticket_secret.encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError('MOTUS_TELEOP_TICKET_SECRET must be valid UTF-8') from None
    if ticket_secret_bytes and len(ticket_secret_bytes) < 32:
        raise ValueError('MOTUS_TELEOP_TICKET_SECRET must contain at least 32 bytes')
    dedicated_ticket_secrets = _parse_driver_ticket_secrets(
        values.get('MOTUS_TELEOP_TICKET_SECRETS', '')
    )
    for driver_id, dedicated_secret in dedicated_ticket_secrets.items():
        dedicated_text = dedicated_secret.decode('utf-8')
        if authenticate(dedicated_text) is not None:
            raise ValueError(
                'teleop ticket secret must not reuse a human access token: '
                f'{driver_id}'
            )
        if _driver_token and _secret_text_equal(
            dedicated_text,
            _driver_token,
        ):
            raise ValueError(
                'teleop ticket secret must not reuse legacy Driver credential: '
                f'{driver_id}'
            )
        bearer_owner = next(
            (
                owner
                for owner, token in _driver_tokens_by_id.items()
                if _secret_text_equal(dedicated_text, token)
            ),
            None,
        )
        if bearer_owner is not None:
            raise ValueError(
                'teleop ticket secret must not reuse dedicated Driver credential: '
                f'{driver_id}, {bearer_owner}'
            )
        if ticket_secret_bytes and secrets.compare_digest(
            dedicated_secret,
            ticket_secret_bytes,
        ):
            raise ValueError(
                'MOTUS_TELEOP_TICKET_SECRET must not reuse dedicated secret: '
                f'{driver_id}'
            )
    if ticket_secret_bytes:
        if authenticate(ticket_secret) is not None:
            raise ValueError(
                'MOTUS_TELEOP_TICKET_SECRET must not reuse a human access token'
            )
        if _driver_token and _secret_text_equal(ticket_secret, _driver_token):
            raise ValueError(
                'MOTUS_TELEOP_TICKET_SECRET must not reuse legacy Driver credential'
            )
        conflicting_bearer = next(
            (
                driver_id
                for driver_id, token in _driver_tokens_by_id.items()
                if _secret_text_equal(ticket_secret, token)
            ),
            None,
        )
        if conflicting_bearer is not None:
            raise ValueError(
                'MOTUS_TELEOP_TICKET_SECRET must not reuse dedicated Driver '
                f'credential: {conflicting_bearer}'
            )
    _teleop_ticket_secrets_by_driver.update(dedicated_ticket_secrets)
    _teleop_ticket_secret = ticket_secret_bytes
    # Service credentials turn otherwise anonymous API routes into confused
    # deputies: an unauthenticated caller could make Core send a trusted
    # Driver Bearer or mint a WebRTC ticket. Enable the API authentication
    # boundary whenever any such credential exists, even if the deployment
    # forgot to configure a human principal. With no human token this fails
    # closed (management APIs return 401) until the environment is corrected.
    _auth_enabled = bool(
        _principals_by_token
        or _driver_token
        or _driver_tokens_by_id
        or _teleop_ticket_secret
        or _teleop_ticket_secrets_by_driver
    )
    unsafe_raw = values.get(
        'MOTUS_ENABLE_UNSAFE_DESKTOP_CODE_TOOLS', ''
    ).strip().lower()
    if unsafe_raw not in ('', '0', 'false', 'no', 'off', '1', 'true', 'yes', 'on'):
        raise ValueError(
            'MOTUS_ENABLE_UNSAFE_DESKTOP_CODE_TOOLS must be true or false'
        )
    requested_unsafe_code_tools = unsafe_raw in ('1', 'true', 'yes', 'on')
    if requested_unsafe_code_tools and (
        _principals_by_token
        or _driver_token
        or _driver_tokens_by_id
        or _teleop_ticket_secret
        or _teleop_ticket_secrets_by_driver
    ):
        raise ValueError(
            'MOTUS_ENABLE_UNSAFE_DESKTOP_CODE_TOOLS cannot be enabled when '
            'authentication or teleop credentials are configured'
        )
    _unsafe_desktop_code_tools_enabled = requested_unsafe_code_tools

    if _principals_by_token:
        print(f'[auth] Principal authentication enabled ({len(_principals_by_token)} identities)')
    elif _auth_enabled:
        print(
            '[auth] Service credentials configured without human tokens '
            '— protected management APIs are locked until an ACCESS_TOKEN '
            'owner is configured'
        )
    else:
        print('[auth] No human tokens configured — legacy API authentication disabled')
    if not (_driver_token or _driver_tokens_by_id):
        print('[auth] Driver credentials not configured — driver registrations are untrusted')
    elif _driver_auth_enforced:
        print(
            '[auth] Driver authentication enforcement enabled '
            f'({_driver_credential_summary()})'
        )
    else:
        print(
            '[auth] Driver credentials configured in compatibility mode '
            f'({_driver_credential_summary()}) — legacy services remain untrusted'
        )
    if not (_teleop_ticket_secret or _teleop_ticket_secrets_by_driver):
        print('[auth] Teleop ticket secrets not configured — WebRTC signaling disabled')
    else:
        print(
            '[auth] Teleop WebRTC ticket signing enabled '
            f'({len(_teleop_ticket_secrets_by_driver)} dedicated, '
            f'{"legacy fallback enabled" if _teleop_ticket_secret else "no legacy fallback"})'
        )


def get_token() -> str:
    """Return the legacy owner token for backwards compatibility."""
    for token, principal in _principals_by_token.items():
        if principal.role == 'owner':
            return token
    return ''


def is_enabled() -> bool:
    return _auth_enabled


def _driver_credential_summary() -> str:
    dedicated = len(_driver_tokens_by_id)
    legacy = 'legacy fallback enabled' if _driver_token else 'no legacy fallback'
    return f'{dedicated} dedicated, {legacy}'


def has_any_driver_credentials() -> bool:
    """Return whether any Core → Driver bearer credential is configured."""

    return bool(_driver_token or _driver_tokens_by_id)


def is_driver_auth_configured(driver_id: str) -> bool:
    """Return whether Core can authenticate to the exact Driver identity."""

    if not _valid_driver_id(driver_id):
        return False
    if driver_id in _driver_tokens_by_id:
        return True
    return bool(_driver_token)


def driver_runtime_credential_available(driver_id: str) -> bool:
    """Require an outbound credential for every persisted trusted record."""

    return is_driver_auth_configured(driver_id)


def driver_credential_binding(driver_id: str) -> str:
    """Return a non-secret binding for the currently selected dedicated token."""

    if not _valid_driver_id(driver_id):
        return ''
    token = _driver_tokens_by_id.get(driver_id)
    if token is None:
        return ''
    digest = hashlib.sha256(
        b'motus.driver.credential-binding.v1\0'
        + driver_id.encode('ascii')
        + b'\0'
        + token.encode('ascii')
    ).hexdigest()
    return f'sha256:{digest}'


def driver_record_credential_available(
    driver_id: str,
    credential_binding: object,
) -> bool:
    """Require re-registration after a dedicated credential is added or rotated."""

    if not is_driver_auth_configured(driver_id):
        return False
    selected_binding = driver_credential_binding(driver_id)
    if selected_binding:
        return bool(
            isinstance(credential_binding, str)
            and _secret_text_equal(credential_binding, selected_binding)
        )
    # A stale dedicated binding must not silently downgrade to the legacy
    # fallback. A valid legacy-token registration removes the old binding.
    return credential_binding is None or credential_binding == ''


def is_driver_auth_enforced() -> bool:
    return _driver_auth_enforced


def unsafe_desktop_code_tools_enabled() -> bool:
    """Return whether an administrator explicitly enabled in-process code tools."""

    return _unsafe_desktop_code_tools_enabled


def has_dedicated_teleop_ticket_secret(driver_id: str) -> bool:
    return bool(
        _valid_driver_id(driver_id)
        and driver_id in _teleop_ticket_secrets_by_driver
    )


def teleop_ticket_secret(driver_id: str) -> bytes:
    """Return the server-only RTC ticket secret for one exact Driver."""

    if not _valid_driver_id(driver_id):
        return b''
    return _teleop_ticket_secrets_by_driver.get(driver_id, _teleop_ticket_secret)


def teleop_ticket_credential_available(driver_id: str) -> bool:
    """Return whether Core can sign an offer ticket for one exact Driver."""

    return bool(teleop_ticket_secret(driver_id))


def authenticate(token: str) -> Optional[Principal]:
    if not token:
        return None
    # Iterate and compare in constant time instead of using a direct dict lookup.
    for configured, principal in _principals_by_token.items():
        if _secret_text_equal(token, configured):
            return principal
    return None


def verify(token: str) -> bool:
    # Preserve the pre-teleop behavior for existing non-teleop APIs.
    if not _auth_enabled:
        return True
    return authenticate(token) is not None


def has_dedicated_driver_token(driver_id: str) -> bool:
    return bool(
        _valid_driver_id(driver_id)
        and driver_id in _driver_tokens_by_id
    )


def driver_token_identity(token: str) -> str | None:
    """Return the dedicated identity owning ``token`` without exposing a secret."""

    if not isinstance(token, str) or not token:
        return None
    matched_identity = None
    for driver_id, configured in _driver_tokens_by_id.items():
        if _secret_text_equal(token, configured):
            matched_identity = driver_id
    return matched_identity


def is_known_driver_token(token: str) -> bool:
    """Return whether ``token`` is any configured Driver bearer credential."""

    if not isinstance(token, str) or not token:
        return False
    matched = bool(_driver_token and _secret_text_equal(token, _driver_token))
    for configured in _driver_tokens_by_id.values():
        matched = _secret_text_equal(token, configured) or matched
    return matched


def verify_driver_token(token: str, driver_id: str) -> bool:
    """Verify a registration credential against one claimed stable identity."""

    if not isinstance(token, str) or not token or not _valid_driver_id(driver_id):
        return False
    dedicated = _driver_tokens_by_id.get(driver_id)
    if dedicated is not None:
        return _secret_text_equal(token, dedicated)
    return bool(_driver_token and _secret_text_equal(token, _driver_token))


def driver_request_headers(driver_id: str) -> dict[str, str]:
    """Return the service credential for Core → Driver MCP requests.

    Callers must never log this mapping.  An empty mapping preserves legacy
    deployments that have not enabled driver authentication.
    """
    if not _valid_driver_id(driver_id):
        return {}
    token = _driver_tokens_by_id.get(driver_id)
    if token is None:
        token = _driver_token
    if not token:
        return {}
    return {'Authorization': f'Bearer {token}'}


def _extract_token(request: Request) -> str:
    auth_header = request.headers.get('authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]
    return request.query_params.get('token', '')


def extract_driver_token(request: Request) -> str:
    token = request.headers.get('x-motus-driver-token', '')
    if token:
        return token
    auth_header = request.headers.get('authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]
    return ''


def request_principal(request: Request) -> Optional[Principal]:
    cached = getattr(request.state, 'principal', None)
    if cached is not None:
        return cached
    principal = authenticate(_extract_token(request))
    request.state.principal = principal
    return principal


def require_role(request: Request, required_role: str = 'viewer') -> Principal:
    principal = request_principal(request)
    if principal is None:
        raise HTTPException(status_code=401, detail='Authentication required')
    if _ROLE_LEVEL.get(principal.role, 0) < _ROLE_LEVEL.get(required_role, 0):
        raise HTTPException(
            status_code=403,
            detail=f'{required_role} role required',
        )
    return principal


async def auth_middleware(request: Request, call_next):
    """Enforce legacy API authentication and attach a trusted Principal."""

    token = _extract_token(request)
    request.state.principal = authenticate(token)

    if not _auth_enabled:
        return await call_next(request)

    path = request.url.path
    if not path.startswith('/api/') and not path.startswith('/ws/'):
        return await call_next(request)

    if path == '/api/auth/verify':
        return await call_next(request)

    principal = request.state.principal
    # A recognized human identity is always governed by its role, including on
    # service callback routes that remain open for legacy machine callers.
    if principal is not None and principal.role != 'owner':
        if path.startswith('/api/teleop/'):
            return await call_next(request)
        return JSONResponse(status_code=403, content={'detail': 'Owner role required'})

    if path.startswith('/api/channel/webhook/'):
        return await call_next(request)
    # Driver/perception registration authenticates and records trust itself.
    if path == '/api/mcp' and request.method == 'POST':
        return await call_next(request)
    if path == '/api/acp/complete' and request.method == 'POST':
        return await call_next(request)
    # System hooks fire (internal/driver calls)
    if path == '/api/hooks/fire' and request.method == 'POST':
        return await call_next(request)
    # /ws/mic stays open (internal browser mic)
    if path == '/ws/mic':
        return await call_next(request)

    if principal is None:
        return JSONResponse(status_code=401, content={'detail': 'Unauthorized'})

    return await call_next(request)


def check_ws_token(websocket: WebSocket) -> bool:
    if not _auth_enabled:
        return True
    principal = authenticate(websocket.query_params.get('token', ''))
    return bool(principal and principal.role == 'owner')
