"""HMAC capabilities for the optional browser Evidence surface."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import re
from urllib.parse import urlsplit

from cryptography.fernet import Fernet
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

SESSION_COOKIE = 'evidence-capability'
SESSION_SECONDS = 3600
PURPOSE = 'deploy-evidence'


def _internal_http_host(host):
    if host == 'localhost' or host.endswith('.localhost'):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (not address.is_unspecified and not address.is_multicast
            and (address.is_private or address.is_loopback or address.is_link_local))


def _public_base_reason(base):
    try:
        if (not isinstance(base, str)
                or any(ord(char) < 33 or ord(char) == 127 for char in base)):
            return 'evidence_public_url_required'
        parsed = urlsplit(base if isinstance(base, str) else '')
        if (parsed.scheme not in {'https', 'http'} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/') or parsed.netloc.endswith(':')
                or any(c.isspace() or ord(c) < 33 or ord(c) == 127 for c in parsed.netloc)):
            return 'evidence_public_url_required'
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            return 'evidence_public_url_required'
        if parsed.scheme == 'http' and not _internal_http_host(parsed.hostname.lower()):
            return 'evidence_public_url_required'
        return ''
    except (TypeError, ValueError):
        return 'evidence_public_url_required'


def disabled_reason(config):
    if not getattr(config, 'deploy_evidence_enabled', False):
        return 'evidence_disabled'
    reason = _public_base_reason(getattr(config, 'evidence_public_base_url', ''))
    if reason:
        return reason
    try:
        key = config.evidence_session_key.encode('ascii')
        Fernet(key)
    except (AttributeError, UnicodeError, TypeError, ValueError):
        return 'evidence_session_key_required'
    return ''


def require_enabled(request):
    reason = disabled_reason(request.app.state.config)
    if reason or getattr(request.app.state, 'evidence_service', None) is None:
        raise HTTPException(503, detail=reason or 'evidence_service_unavailable')


def _key(config):
    return config.evidence_session_key.encode('ascii')


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def issue_capability(config, repo, pr_number, head_sha, round_id, purpose=PURPOSE):
    claims = {'repo': repo, 'pr': pr_number, 'head': head_sha,
              'round': round_id, 'purpose': purpose}
    body = _b64(json.dumps(claims, sort_keys=True, separators=(',', ':')).encode())
    signature = _b64(hmac.new(_key(config), body.encode('ascii'), hashlib.sha256).digest())
    return body + '.' + signature


def verify_capability(config, token, repo, pr_number):
    try:
        if not isinstance(token, str) or len(token) > 2048:
            raise ValueError()
        body, signature = token.split('.', 1)
        expected = _b64(hmac.new(_key(config), body.encode('ascii'), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError()
        raw = base64.urlsafe_b64decode(body + '=' * (-len(body) % 4))
        claims = json.loads(raw)
        if (not isinstance(claims, dict) or claims.get('repo') != repo
                or type(claims.get('pr')) is not int or claims['pr'] != pr_number
                or claims.get('purpose') != PURPOSE
                or not re.fullmatch(r'[0-9a-f]{40}', str(claims.get('head', '')))
                or not re.fullmatch(r'[0-9a-f]{40}-[0-9]+', str(claims.get('round', '')))
                or not claims['round'].startswith(claims['head'] + '-')):
            raise ValueError()
        return claims
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, binascii.Error):
        raise HTTPException(403, detail='invalid_evidence_capability') from None


def csrf_token(config, token):
    return _b64(hmac.new(_key(config), b'evidence-csrf:' + token.encode('ascii'),
                         hashlib.sha256).digest())


def _secure(request):
    return request.url.scheme.lower() == 'https'


def _set_cookie(response, request, token):
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_SECONDS, path='/',
                        secure=_secure(request), httponly=True, samesite='lax')


def bind_page_capability(request, repo, pr_number):
    """Exchange a Summary query capability for an HttpOnly cookie and clean URL."""
    require_enabled(request)
    token = request.query_params.get('cap')
    if token:
        verify_capability(request.app.state.config, token, repo, pr_number)
        response = RedirectResponse(request.url.path, status_code=303,
                                    headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})
        _set_cookie(response, request, token)
        return response
    token = request.cookies.get(SESSION_COOKIE, '')
    verify_capability(request.app.state.config, token, repo, pr_number)
    return None


async def verified_principal(request: Request, repo: str, pr_number: int, *, write=False):
    require_enabled(request)
    config = request.app.state.config
    token = request.query_params.get('cap') or request.cookies.get(SESSION_COOKIE, '')
    claims = verify_capability(config, token, repo, pr_number)
    csrf = csrf_token(config, token)
    if write:
        parsed = urlsplit(config.evidence_public_base_url.rstrip('/'))
        host = parsed.hostname or ''
        if ':' in host:
            host = '[' + host + ']'
        port = parsed.port
        if port is not None and not (parsed.scheme == 'http' and port == 80) and not (
                parsed.scheme == 'https' and port == 443):
            host += ':' + str(port)
        origin = f'{parsed.scheme}://{host}'
        supplied = request.headers.get('x-csrf-token', '')
        if (request.headers.get('origin', '') != origin or not supplied
                or not hmac.compare_digest(supplied, csrf)):
            raise HTTPException(403, detail='evidence_csrf_failed')
    actor = request.headers.get('x-evidence-submitter', 'browser tester')
    if not isinstance(actor, str) or not actor.strip():
        actor = 'browser tester'
    actor = ' '.join(actor.split())[:100]
    actor_id = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big') or 1
    return {'id': actor_id, 'login': actor, 'actor': actor,
            'capability': claims, 'capability_token': token}, csrf
