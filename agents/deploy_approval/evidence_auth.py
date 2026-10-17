"""Read-only public capabilities and GitHub user OAuth for Evidence writes."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import time
from urllib.parse import urlencode, urlsplit

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

CAPABILITY_COOKIE = 'evidence-capability'
USER_COOKIE = 'evidence-user-session'
FLOW_COOKIE = 'evidence-oauth-flow'
SESSION_COOKIE = CAPABILITY_COOKIE  # compatibility for existing page links
SESSION_SECONDS = 3600
FLOW_SECONDS = 600
PURPOSE = 'deploy-evidence'  # previously published public capability; read-only now
DOWNLOAD_PURPOSE = 'deploy-evidence-download'
_PURPOSES = {PURPOSE, DOWNLOAD_PURPOSE}
_USED_STATES: dict[str, float] = {}


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
        if not isinstance(base, str) or any(ord(c) < 33 or ord(c) == 127 for c in base):
            return 'evidence_public_url_required'
        parsed = urlsplit(base)
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
        Fernet(config.evidence_session_key.encode('ascii'))
    except (AttributeError, UnicodeError, TypeError, ValueError):
        return 'evidence_session_key_required'
    return ''


def oauth_enabled(config):
    return (not disabled_reason(config)
            and bool(getattr(config, 'evidence_oauth_client_id', '').strip())
            and bool(getattr(config, 'evidence_oauth_client_secret', '').strip())
            and _callback_allowed(config))


def _callback_allowed(config):
    try:
        parsed = urlsplit(config.evidence_public_base_url.rstrip('/'))
        return parsed.scheme == 'https' or (
            parsed.scheme == 'http' and parsed.hostname
            and (parsed.hostname == 'localhost' or parsed.hostname.endswith('.localhost')
                 or ipaddress.ip_address(parsed.hostname).is_loopback)
        )
    except (ValueError, AttributeError):
        return False


def require_enabled(request):
    reason = disabled_reason(request.app.state.config)
    if reason or getattr(request.app.state, 'evidence_service', None) is None:
        raise HTTPException(503, detail=reason or 'evidence_service_unavailable')


def require_oauth(request):
    require_enabled(request)
    if not oauth_enabled(request.app.state.config):
        raise HTTPException(503, detail='evidence_write_authorization_unavailable')


def _key(config):
    return config.evidence_session_key.encode('ascii')


def _fernet(config):
    return Fernet(_key(config))


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def _seal(config, value):
    return _fernet(config).encrypt(json.dumps(value, separators=(',', ':')).encode()).decode('ascii')


def _unseal(config, token, max_age):
    try:
        raw = _fernet(config).decrypt(token.encode('ascii'), ttl=max_age)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (InvalidToken, ValueError, TypeError, UnicodeError, binascii.Error):
        return None


def issue_capability(config, repo, pr_number, head_sha, round_id, purpose=PURPOSE, *, resource=None):
    if purpose not in _PURPOSES:
        raise ValueError('invalid evidence capability purpose')
    claims = {'repo': repo, 'pr': pr_number, 'head': head_sha,
              'round': round_id, 'purpose': purpose}
    if purpose == DOWNLOAD_PURPOSE:
        if not isinstance(resource, str) or not 1 <= len(resource) <= 160 or any(ord(c) < 33 for c in resource):
            raise ValueError('download capability resource required')
        claims['resource'] = resource
    elif resource is not None:
        raise ValueError('upload capability cannot carry a resource')
    body = _b64(json.dumps(claims, sort_keys=True, separators=(',', ':')).encode())
    signature = _b64(hmac.new(_key(config), body.encode('ascii'), hashlib.sha256).digest())
    return body + '.' + signature


def verify_capability(config, token, repo, pr_number, *, purpose=None, resource=None):
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
                or claims.get('purpose') not in _PURPOSES
                or (purpose is not None and claims.get('purpose') != purpose)
                or (claims.get('purpose') == DOWNLOAD_PURPOSE
                    and (not isinstance(resource, str) or claims.get('resource') != resource))
                or (claims.get('purpose') == PURPOSE and 'resource' in claims)
                or not re.fullmatch(r'[0-9a-f]{40}', str(claims.get('head', '')))
                or not re.fullmatch(r'[0-9a-f]{40}-[0-9]+', str(claims.get('round', '')))
                or not claims['round'].startswith(claims['head'] + '-')):
            raise ValueError()
        return claims
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, binascii.Error):
        raise HTTPException(403, detail='invalid_evidence_capability') from None


def csrf_token(config, token):
    return _b64(hmac.new(_key(config), b'evidence-csrf:' + token.encode('ascii'), hashlib.sha256).digest())


def _secure(request):
    return request.url.scheme.lower() == 'https'


def _cookie(response, request, name, token, max_age):
    response.set_cookie(name, token, max_age=max_age, path='/', secure=_secure(request),
                        httponly=True, samesite='lax')


def bind_page_capability(request, repo, pr_number):
    """Exchange a Summary query capability for an HttpOnly read-only cookie."""
    require_enabled(request)
    token = request.query_params.get('cap')
    if token:
        verify_capability(request.app.state.config, token, repo, pr_number)
        response = RedirectResponse(request.url.path, status_code=303,
                                    headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})
        _cookie(response, request, CAPABILITY_COOKIE, token, SESSION_SECONDS)
        return response
    token = request.cookies.get(CAPABILITY_COOKIE, '')
    if token:
        verify_capability(request.app.state.config, token, repo, pr_number)
    elif not _read_user_session(request.app.state.config, request.cookies.get(USER_COOKIE, ''), repo, pr_number):
        raise HTTPException(403, detail='invalid_evidence_capability')
    return None


def _read_user_session(config, token, repo, pr_number):
    value = _unseal(config, token, SESSION_SECONDS) if token else None
    if (not value or value.get('repo') != repo or value.get('pr') != pr_number
            or type(value.get('id')) is not int or value['id'] <= 0
            or not isinstance(value.get('login'), str) or not value.get('login')
            or not isinstance(value.get('access_token'), str) or not value['access_token']
            or not isinstance(value.get('csrf'), str)):
        return None
    return value


async def _github_get(url, token=None):
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'Deploy-Approval-Evidence'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), follow_redirects=False) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError:
        raise HTTPException(503, detail='github_user_validation_unavailable') from None
    if response.status_code != 200:
        raise HTTPException(403, detail='github_user_validation_failed')
    try:
        return response.json()
    except ValueError:
        raise HTTPException(403, detail='github_user_validation_failed') from None


async def verified_principal(request: Request, repo: str, pr_number: int, *, write=False,
                             purpose=PURPOSE, resource=None):
    require_enabled(request)
    config = request.app.state.config
    session_token = request.cookies.get(USER_COOKIE, '')
    session = _read_user_session(config, session_token, repo, pr_number)
    if write:
        require_oauth(request)
        if not session:
            raise HTTPException(401, detail='github_user_authentication_required')
        parsed = urlsplit(config.evidence_public_base_url.rstrip('/'))
        host = parsed.hostname or ''
        if ':' in host:
            host = '[' + host + ']'
        port = parsed.port
        if port is not None and not (parsed.scheme == 'http' and port == 80) and not (parsed.scheme == 'https' and port == 443):
            host += ':' + str(port)
        origin = f'{parsed.scheme}://{host}'
        supplied = request.headers.get('x-csrf-token', '')
        if (request.headers.get('origin', '') != origin or not supplied
                or not hmac.compare_digest(supplied, session['csrf'])):
            raise HTTPException(403, detail='evidence_csrf_failed')
    if session:
        user = await _github_get('https://api.github.com/user', session['access_token'])
        uid, login = user.get('id'), user.get('login')
        if type(uid) is not int or uid != session['id'] or not isinstance(login, str) or login.lower() != session['login'].lower():
            raise HTTPException(403, detail='github_user_identity_changed')
        principal = {'id': uid, 'login': login, 'actor': login,
                     'github_user_authenticated': True, 'access_token': session['access_token'],
                     'csrf': session['csrf']}
        return principal, session['csrf']
    token = request.query_params.get('cap') or request.cookies.get(CAPABILITY_COOKIE, '')
    claims = None
    if purpose == DOWNLOAD_PURPOSE:
        try:
            claims = verify_capability(config, token, repo, pr_number, purpose=DOWNLOAD_PURPOSE, resource=resource)
        except HTTPException:
            # Capabilities published before per-attachment URLs remain read-only.
            claims = verify_capability(config, token, repo, pr_number, purpose=PURPOSE)
    else:
        claims = verify_capability(config, token, repo, pr_number, purpose=purpose, resource=resource)
    if write:
        raise HTTPException(403, detail='github_user_authentication_required')
    actor_id = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big') or 1
    return {'id': actor_id, 'login': 'evidence-capability', 'actor': 'evidence-capability',
            'capability': claims, 'capability_token': token,
            'github_user_authenticated': False}, ''


def oauth_callback_url(config):
    return config.evidence_public_base_url.rstrip('/') + '/evidence/auth/callback'


def oauth_login(request, repo, pr_number, return_to):
    require_oauth(request)
    cap = request.cookies.get(CAPABILITY_COOKIE, '')
    verify_capability(request.app.state.config, cap, repo, pr_number, purpose=PURPOSE)
    if not isinstance(return_to, str) or not re.fullmatch(r'/evidence/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pulls/[1-9][0-9]*', return_to):
        raise HTTPException(400, detail='invalid_return_path')
    if return_to.lower() != f'/evidence/{repo}/pulls/{pr_number}'.lower():
        raise HTTPException(400, detail='invalid_return_path')
    config = request.app.state.config
    state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
    flow = {'state': state, 'verifier': verifier, 'repo': repo, 'pr': pr_number,
            'return_to': return_to, 'issued': int(time.time())}
    challenge = _b64(hashlib.sha256(verifier.encode()).digest())
    callback = oauth_callback_url(config)
    query = urlencode({'client_id': config.evidence_oauth_client_id,
                       'redirect_uri': callback, 'state': state,
                       'code_challenge': challenge, 'code_challenge_method': 'S256',
                       })
    response = RedirectResponse('https://github.com/login/oauth/authorize?' + query, status_code=302,
                                headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})
    _cookie(response, request, FLOW_COOKIE, _seal(config, flow), FLOW_SECONDS)
    return response


async def oauth_callback(request):
    require_oauth(request)
    config = request.app.state.config
    flow = _unseal(config, request.cookies.get(FLOW_COOKIE, ''), FLOW_SECONDS)
    state = request.query_params.get('state', '')
    code = request.query_params.get('code', '')
    now = time.time()
    for key, expires in list(_USED_STATES.items()):
        if expires < now:
            _USED_STATES.pop(key, None)
    if (not flow or not state or not hmac.compare_digest(state, str(flow.get('state', '')))
            or not code or len(code) > 1024 or state in _USED_STATES
            or now - flow.get('issued', 0) > FLOW_SECONDS):
        raise HTTPException(403, detail='invalid_or_replayed_oauth_state')
    _USED_STATES[state] = now + FLOW_SECONDS
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), follow_redirects=False) as client:
            response = await client.post('https://github.com/login/oauth/access_token',
                headers={'Accept': 'application/json', 'User-Agent': 'Deploy-Approval-Evidence'},
                data={'client_id': config.evidence_oauth_client_id,
                      'client_secret': config.evidence_oauth_client_secret,
                      'code': code, 'redirect_uri': oauth_callback_url(config),
                      'code_verifier': flow['verifier']})
    except httpx.HTTPError:
        raise HTTPException(503, detail='github_oauth_exchange_unavailable') from None
    if response.status_code != 200:
        raise HTTPException(403, detail='github_oauth_exchange_failed')
    try:
        token_response = response.json()
        access_token = token_response.get('access_token') if isinstance(token_response, dict) else None
    except (ValueError, TypeError):
        access_token = None
    if not isinstance(access_token, str) or not access_token or len(access_token) > 4096:
        raise HTTPException(403, detail='github_oauth_exchange_failed')
    user = await _github_get('https://api.github.com/user', access_token)
    uid, login = user.get('id'), user.get('login')
    if type(uid) is not int or uid <= 0 or not isinstance(login, str) or not re.fullmatch(r'[A-Za-z0-9-]{1,39}', login):
        raise HTTPException(403, detail='github_user_validation_failed')
    session = {'id': uid, 'login': login, 'access_token': access_token,
               'csrf': secrets.token_urlsafe(32), 'repo': flow['repo'], 'pr': flow['pr']}
    result = RedirectResponse(flow['return_to'], status_code=303,
                              headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})
    _cookie(result, request, USER_COOKIE, _seal(config, session), SESSION_SECONDS)
    result.delete_cookie(FLOW_COOKIE, path='/', secure=_secure(request), httponly=True, samesite='lax')
    return result
