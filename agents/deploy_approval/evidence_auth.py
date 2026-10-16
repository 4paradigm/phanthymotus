"""GitHub user OAuth, encrypted short sessions, and same-origin write checks."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
from urllib.parse import urlencode, urlsplit

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

SESSION_COOKIE = '__Host-evidence-session'
FLOW_COOKIE = '__Host-evidence-oauth'
SESSION_SECONDS = 3600


def disabled_reason(config):
    if not config.deploy_evidence_enabled:
        return 'evidence_disabled'
    base = config.evidence_public_base_url
    try:
        if not isinstance(base, str) or any(ord(c) < 33 for c in base):
            return 'evidence_public_url_required'
        parsed = urlsplit(base)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in ('', '/')
                or not parsed.port and parsed.netloc.endswith(':')):
            return 'evidence_public_url_required'
    except ValueError:
        return 'evidence_public_url_required'
    if not all(isinstance(v, str) and v.strip() for v in (
            config.evidence_oauth_client_id, config.evidence_oauth_client_secret)):
        return 'evidence_oauth_required'
    try:
        if not isinstance(config.evidence_session_key, str):
            return 'evidence_session_key_required'
        Fernet(config.evidence_session_key.encode('ascii'))
    except (ValueError, TypeError, UnicodeError):
        return 'evidence_session_key_required'
    return ''


def require_enabled(request):
    reason = disabled_reason(request.app.state.config)
    if reason or getattr(request.app.state, 'evidence_service', None) is None:
        raise HTTPException(503, detail=reason or 'evidence_service_unavailable')


def _cipher(config):
    return Fernet(config.evidence_session_key.encode('ascii'))


def _seal(config, value):
    return _cipher(config).encrypt(json.dumps(value, separators=(',', ':')).encode()).decode('ascii')


def _open(config, value, ttl):
    try:
        result = json.loads(_cipher(config).decrypt((value or '').encode('ascii'), ttl=ttl))
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except (InvalidToken, ValueError, TypeError, UnicodeError):
        raise HTTPException(401, detail='evidence_login_required') from None


def _cookie(response, name, value, max_age):
    response.set_cookie(name, value, max_age=max_age, path='/', secure=True,
                        httponly=True, samesite='lax')


def _return_path(value):
    # Only the stable PR page may be a login target; no external redirect.
    if not isinstance(value, str) or not re.fullmatch(
            r'/evidence/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+/pulls/[1-9][0-9]*', value):
        raise HTTPException(400, detail='invalid_return_path')
    return value


async def login(request: Request):
    require_enabled(request)
    config = request.app.state.config
    target = _return_path(request.query_params.get('return_to', ''))
    state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    params = {'client_id': config.evidence_oauth_client_id,
              'redirect_uri': config.evidence_public_base_url.rstrip('/') + '/evidence/auth/callback',
              'state': state, 'code_challenge': challenge, 'code_challenge_method': 'S256'}
    response = RedirectResponse('https://github.com/login/oauth/authorize?' + urlencode(params), status_code=303)
    _cookie(response, FLOW_COOKIE, _seal(config, {'state': state, 'verifier': verifier, 'return_to': target}), 600)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


async def _get_user(http, token):
    response = await http.get('https://api.github.com/user',
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
    response.raise_for_status()
    user = response.json()
    if (not isinstance(user.get('id'), int) or isinstance(user['id'], bool) or user['id'] < 1
            or not isinstance(user.get('login'), str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}', user['login'])
            or user.get('type') != 'User'):
        raise ValueError('invalid GitHub user')
    return {'id': user['id'], 'login': user['login']}


async def callback(request: Request):
    require_enabled(request)
    config = request.app.state.config
    flow = _open(config, request.cookies.get(FLOW_COOKIE), 600)
    state = request.query_params.get('state', '')
    if not state or not hmac.compare_digest(state, flow.get('state', '')):
        raise HTTPException(400, detail='oauth_state_mismatch')
    code = request.query_params.get('code', '')
    if not code or len(code) > 1024 or request.query_params.get('error'):
        raise HTTPException(400, detail='oauth_authorization_failed')
    try:
        async with httpx.AsyncClient(verify=True, trust_env=False, timeout=20, follow_redirects=False) as http:
            response = await http.post('https://github.com/login/oauth/access_token',
                headers={'Accept': 'application/json'}, data={
                    'client_id': config.evidence_oauth_client_id,
                    'client_secret': config.evidence_oauth_client_secret,
                    'code': code, 'code_verifier': flow['verifier'],
                    'redirect_uri': config.evidence_public_base_url.rstrip('/') + '/evidence/auth/callback'})
            response.raise_for_status()
            token = response.json().get('access_token')
            if not isinstance(token, str) or not token:
                raise ValueError()
            user = await _get_user(http, token)
    except (httpx.HTTPError, ValueError, KeyError):
        raise HTTPException(401, detail='oauth_identity_verification_failed') from None
    session = {**user, 'token': token, 'csrf': secrets.token_urlsafe(32)}
    encrypted = _seal(config, session)
    if len(encrypted) > 3800:
        raise HTTPException(401, detail='oauth_session_too_large')
    response = RedirectResponse(_return_path(flow['return_to']), status_code=303)
    response.delete_cookie(FLOW_COOKIE, path='/', secure=True, httponly=True, samesite='lax')
    _cookie(response, SESSION_COOKIE, encrypted, SESSION_SECONDS)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


async def verified_principal(request: Request, repo: str, *, write=False):
    require_enabled(request)
    config = request.app.state.config
    if not config.auth_valid or repo not in config.active_repos:
        raise HTTPException(403, detail='repository_not_authorized')
    session = _open(config, request.cookies.get(SESSION_COOKIE), SESSION_SECONDS)
    if write:
        expected = config.evidence_public_base_url.rstrip('/')
        origin, csrf = request.headers.get('origin', ''), request.headers.get('x-csrf-token', '')
        if origin != expected or not csrf or not hmac.compare_digest(csrf, session.get('csrf', '')):
            raise HTTPException(403, detail='evidence_csrf_failed')
    try:
        async with httpx.AsyncClient(verify=True, trust_env=False, timeout=20, follow_redirects=False) as http:
            user = await _get_user(http, session['token'])
            if user['id'] != session['id'] or user['login'].lower() != session['login'].lower():
                raise ValueError()
            # A user credential independently proves repository access; App
            # installation access cannot stand in for the logged-in user.
            access = await http.get('https://api.github.com/repos/' + repo,
                headers={'Authorization': 'Bearer ' + session['token'],
                         'Accept': 'application/vnd.github+json'})
            access.raise_for_status()
            if access.json().get('full_name', '').lower() != repo.lower():
                raise ValueError()
    except (httpx.HTTPError, KeyError, ValueError):
        raise HTTPException(403, detail='evidence_user_access_denied') from None
    # A uses this credential only to recheck user access; it is never part of
    # an operation record or a store request.
    return {**user, 'user_access_token': session['token']}, session['csrf']
