"""Read-only capabilities, scoped upload invitations, and proof-bound writes."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

from .config import is_evidence_ingress_url, is_stable_public_base_url

CAPABILITY_COOKIE = 'evidence-capability'
SESSION_COOKIE = CAPABILITY_COOKIE
SESSION_SECONDS = 3600
PURPOSE = 'deploy-evidence'
DOWNLOAD_PURPOSE = 'deploy-evidence-download'
UPLOAD_PURPOSE = 'deploy-evidence-upload'
_PURPOSES = {PURPOSE, DOWNLOAD_PURPOSE, UPLOAD_PURPOSE}
UPLOAD_COOKIE = 'evidence-upload-capability'
UPLOAD_MAX_FILES = 20
UPLOAD_MAX_BYTES = 10 * 1024 * 1024 * 1024
UPLOAD_MAX_LIFETIME_SECONDS = 30 * 24 * 60 * 60
PROOF_HEADER = 'x-evidence-secret'
NONCE_HEADER = 'x-evidence-nonce'
PROOF_SECONDS = 24 * 60 * 60


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
    return '' if is_stable_public_base_url(base) else 'evidence_public_url_required'


def disabled_reason(config):
    if not cos_ready(config):
        return 'COS_NOT_READY'
    return ''


def cos_ready(config):
    return bool(getattr(config, 'deploy_evidence_enabled', False)) and all(
        isinstance(getattr(config, name, None), str) and getattr(config, name).strip()
        for name in ('cos_region', 'cos_bucket', 'cos_secret_id', 'cos_secret_key'))


def public_https_ingress_ready(config):
    base = getattr(config, 'evidence_public_base_url', '')
    return is_evidence_ingress_url(base)


def _configured_https_authority(config):
    parsed = urlsplit(config.evidence_public_base_url)
    host = parsed.hostname.lower()
    if ':' in host:
        host = '[' + host + ']'
    port = parsed.port
    if port is not None and port != 443:
        host += ':' + str(port)
    return host


def _configured_https_request(request):
    """Accept direct TLS or the HTTPS marker from the loopback-only proxy path.

    Docker publishes the application port on host loopback only; the existing
    Nginx listener is also loopback-bound and sets X-Forwarded-Proto itself.
    Never use forwarded host data to select or validate the public origin.
    """
    config = request.app.state.config
    if not public_https_ingress_ready(config):
        return False
    parsed = urlsplit(config.evidence_public_base_url)
    supplied_host = request.headers.get('host', '').lower()
    if supplied_host not in {parsed.netloc.lower(), _configured_https_authority(config)}:
        return False
    if request.scope.get('scheme') == 'https':
        return True
    if request.scope.get('scheme') != 'http':
        return False
    forwarded_proto = [value.strip().lower() for name, value in request.scope.get('headers', [])
                       if name.lower() == b'x-forwarded-proto']
    peer = getattr(request, 'client', None)
    try:
        peer_ip = ipaddress.ip_address(peer.host) if peer and peer.host else None
    except ValueError:
        peer_ip = None
    trusted_proxy_peer = bool(peer_ip and (
        peer_ip.is_loopback or peer_ip.is_private or peer_ip.is_link_local))
    return forwarded_proto == [b'https'] and trusted_proxy_peer


def readiness(config, evidence_service=None, request=None):
    storage = cos_ready(config)
    backend = storage and evidence_service is not None and bool(getattr(config, 'auth_valid', False))
    ingress = public_https_ingress_ready(config)
    local_tunnel = bool(backend and request is not None and _is_local_tunnel_request(
        request, require_origin=False))
    return {
        'COS_READY': storage,
        'EVIDENCE_BACKEND_READY': backend,
        'PUBLIC_HTTPS_INGRESS_READY': ingress,
        'STABLE_PUBLIC_HTTPS_INGRESS_READY': is_stable_public_base_url(
            getattr(config, 'evidence_public_base_url', ''),
        ),
        'LOCAL_SSH_TUNNEL_READY': local_tunnel,
        # A loopback SSH tunnel is useful for local development, but it cannot
        # back Summary links that must remain usable after the tunnel ends.
        'BROWSER_UPLOAD_READY': backend and ingress,
        'status': '' if backend and ingress else (
            'EVIDENCE_BACKEND_PENDING' if not backend else 'PUBLIC_INGRESS_PENDING'),
    }


def _is_local_tunnel_request(request, *, require_origin=True):
    """Recognize only the documented loopback SSH forward; ignore proxy headers."""
    headers = request.headers
    origin = headers.get('origin', '').lower()
    return (
        request.scope.get('scheme') == 'http'
        and headers.get('host', '').lower() == '127.0.0.1:25001'
        and (origin == 'http://127.0.0.1:25001'
             or (not origin and not require_origin))
        and headers.get('sec-fetch-site', '').lower() == 'same-origin'
    )


def require_enabled(request):
    reason = disabled_reason(request.app.state.config)
    if reason or getattr(request.app.state, 'evidence_service', None) is None:
        raise HTTPException(503, detail=reason or 'EVIDENCE_BACKEND_PENDING')


def _key(config):
    secret = getattr(config, 'cos_secret_key', '')
    if not isinstance(secret, str) or not secret:
        raise ValueError('Evidence key unavailable')
    # Domain separation keeps capability signatures independent from COS use.
    return hmac.new(secret.encode('utf-8'),
                    b'phanthymotus/deploy-approval/evidence-capability/v1',
                    hashlib.sha256).digest()


def _legacy_key(config):
    """Verify already-published read links while issuing only derived-key links."""
    value = getattr(config, 'evidence_session_key', '')
    if not isinstance(value, str) or not value:
        return None
    try:
        return value.encode('ascii')
    except UnicodeError:
        return None


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def issue_capability(config, repo, pr_number, head_sha, round_id, purpose=PURPOSE, *, resource=None,
                     expires_at=None):
    if purpose not in _PURPOSES:
        raise ValueError('invalid evidence capability purpose')
    if purpose == UPLOAD_PURPOSE:
        now = int(time.time())
        if (type(pr_number) is not int or pr_number <= 0
                or not isinstance(repo, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo)
                or not re.fullmatch(r'[0-9a-f]{40}', str(head_sha))
                or not re.fullmatch(re.escape(str(head_sha)) + r'-[0-9]+', str(round_id))
                or type(expires_at) is not int or not now < expires_at <= now + UPLOAD_MAX_LIFETIME_SECONDS):
            raise ValueError('upload capability expiry required')
    elif expires_at is not None:
        raise ValueError('expiry is only valid for upload capabilities')
    claims = {'repo': repo, 'pr': pr_number, 'head': head_sha,
              'round': round_id, 'purpose': purpose}
    if purpose == UPLOAD_PURPOSE:
        claims.update(expires_at=expires_at, status='testing')
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
        raw = base64.urlsafe_b64decode(body + '=' * (-len(body) % 4))
        claims = json.loads(raw)
        keys = [_key(config)]
        legacy = None if isinstance(claims, dict) and claims.get('purpose') == UPLOAD_PURPOSE else _legacy_key(config)
        if legacy and legacy not in keys:
            keys.append(legacy)
        if not any(hmac.compare_digest(
                signature, _b64(hmac.new(key, body.encode('ascii'), hashlib.sha256).digest()))
                for key in keys):
            raise ValueError()
        if (not isinstance(claims, dict) or claims.get('repo') != repo
                or type(claims.get('pr')) is not int or claims['pr'] != pr_number
                or claims.get('purpose') not in _PURPOSES
                or (purpose is not None and claims.get('purpose') != purpose)
                or (claims.get('purpose') == DOWNLOAD_PURPOSE
                    and (not isinstance(resource, str) or claims.get('resource') != resource))
                or (claims.get('purpose') in {PURPOSE, UPLOAD_PURPOSE} and 'resource' in claims)
                or (claims.get('purpose') == UPLOAD_PURPOSE
                    and (type(claims.get('expires_at')) is not int
                         or not int(time.time()) < claims['expires_at'] <=
                         int(time.time()) + UPLOAD_MAX_LIFETIME_SECONDS
                         or claims.get('status') != 'testing'))
                or (claims.get('purpose') != UPLOAD_PURPOSE and 'expires_at' in claims)
                or not re.fullmatch(r'[0-9a-f]{40}', str(claims.get('head', '')))
                or not re.fullmatch(r'[0-9a-f]{40}-[0-9]+', str(claims.get('round', '')))
                or not claims['round'].startswith(claims['head'] + '-')):
            raise ValueError()
        return claims
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, binascii.Error):
        raise HTTPException(403, detail='invalid_evidence_capability') from None


def capability_lookup_target(token, repo, pr_number):
    """Read only a bounded, unverified capability payload as a lookup hint.

    The returned values are never authorization. Callers must match them to a
    trusted lifecycle state or verified round archive before minting a new
    read capability.
    """
    try:
        if not isinstance(token, str) or len(token) > 2048:
            return None
        body, _signature = token.split('.', 1)
        raw = base64.urlsafe_b64decode(body + '=' * (-len(body) % 4))
        claims = json.loads(raw)
        if not isinstance(claims, dict):
            return None
        head, round_id = claims.get('head'), claims.get('round')
        if (claims.get('repo') != repo
                or type(claims.get('pr')) is not int or claims['pr'] != pr_number
                or claims.get('purpose') not in _PURPOSES
                or not isinstance(head, str) or not re.fullmatch(r'[0-9a-f]{40}', head)
                or not isinstance(round_id, str)
                or not re.fullmatch(re.escape(head) + r'-[0-9]+', round_id)):
            return None
        return {'head': head, 'round': round_id}
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, binascii.Error):
        return None


def _verify_page_capability(config, token, repo, pr_number):
    try:
        return verify_capability(config, token, repo, pr_number, purpose=PURPOSE)
    except HTTPException:
        return verify_capability(config, token, repo, pr_number,
                                 purpose=DOWNLOAD_PURPOSE, resource='page')


def _verify_upload_capability(config, token, repo, pr_number):
    return verify_capability(config, token, repo, pr_number, purpose=UPLOAD_PURPOSE)


def _secure(request):
    config = request.app.state.config
    base = getattr(config, 'evidence_public_base_url', '')
    return public_https_ingress_ready(config) and urlsplit(base).scheme == 'https'


def _cookie(response, request, name, token, max_age):
    response.set_cookie(name, token, max_age=max_age, path='/', secure=_secure(request),
                        httponly=True, samesite='lax')


def bind_page_capability(request, repo, pr_number, *, trusted_summary_capability=None,
                         read_only=False):
    """Exchange a Summary read or upload capability for an HttpOnly cookie."""
    require_enabled(request)
    if read_only:
        # view=read is an explicit read-only mode. Never inspect the query cap
        # or upload cookie here; only a page-scoped read capability can bind it.
        token = trusted_summary_capability or request.cookies.get(CAPABILITY_COOKIE, '')
        if not token:
            raise HTTPException(403, detail='invalid_evidence_capability')
        claims = _verify_page_capability(request.app.state.config, token, repo, pr_number)
        requested_head = request.query_params.get('head_sha')
        requested_round = request.query_params.get('round_id')
        if ((requested_head and requested_head != claims.get('head'))
                or (requested_round and requested_round != claims.get('round'))):
            raise HTTPException(403, detail='invalid_evidence_capability')
        target_query = urlencode({'view': 'read', 'head_sha': claims['head'],
                                  'round_id': claims['round']})
        expected_query = {'view': 'read', 'head_sha': claims['head'],
                          'round_id': claims['round']}
        response = None
        if (dict(request.query_params) != expected_query or 'cap' in request.query_params):
            response = RedirectResponse(
                request.url.path + '?' + target_query, status_code=303,
                headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'},
            )
        elif token != request.cookies.get(CAPABILITY_COOKIE, ''):
            response = RedirectResponse(
                request.url.path + '?' + target_query, status_code=303,
                headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'},
            )
        if response is not None and token != request.cookies.get(CAPABILITY_COOKIE, ''):
            _cookie(response, request, CAPABILITY_COOKIE, token, SESSION_SECONDS)
        return response
    token = request.query_params.get('cap')
    if 'cap' in request.query_params and not token:
        raise HTTPException(403, detail='invalid_evidence_capability')
    if 'cap' not in request.query_params and trusted_summary_capability:
        # The page handler supplies a read capability recovered from the
        # trusted Summary or a verified upload cookie; neither overrides an
        # explicitly supplied query capability.
        token = trusted_summary_capability
    if token:
        try:
            claims = _verify_upload_capability(request.app.state.config, token, repo, pr_number)
            upload = True
        except HTTPException:
            claims = _verify_page_capability(request.app.state.config, token, repo, pr_number)
            upload = False
        if upload:
            if not _configured_https_request(request):
                raise HTTPException(403, detail='evidence_origin_failed')
        target_path = request.url.path if upload else request.url.path + '?view=read'
        response = RedirectResponse(target_path, status_code=303,
                                    headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})
        if upload:
            _cookie(response, request, UPLOAD_COOKIE, token,
                    max(1, claims['expires_at'] - int(time.time())))
            response.delete_cookie(
                CAPABILITY_COOKIE, path='/', secure=_secure(request),
                httponly=True, samesite='lax',
            )
        else:
            _cookie(response, request, CAPABILITY_COOKIE, token, SESSION_SECONDS)
        return response
    token = request.cookies.get(CAPABILITY_COOKIE, '')
    if token:
        try:
            _verify_page_capability(request.app.state.config, token, repo, pr_number)
            return None
        except HTTPException:
            pass
    token = request.cookies.get(UPLOAD_COOKIE, '')
    if token:
        try:
            _verify_upload_capability(request.app.state.config, token, repo, pr_number)
            if not _configured_https_request(request):
                raise HTTPException(403, detail='evidence_origin_failed')
            return None
        except HTTPException:
            pass
    raise HTTPException(403, detail='invalid_evidence_capability')


def _trusted_origin(request, *, require_header=True):
    config = request.app.state.config
    if _is_local_tunnel_request(request, require_origin=require_header):
        return 'http://127.0.0.1:25001'
    if request.headers.get('host', '').lower() == '127.0.0.1:25001':
        raise HTTPException(403, detail='evidence_origin_failed')
    base = getattr(config, 'evidence_public_base_url', '')
    if not public_https_ingress_ready(config):
        raise HTTPException(503, detail='PUBLIC_INGRESS_PENDING')
    parsed = urlsplit(base)
    origin = f'{parsed.scheme}://{_configured_https_authority(config)}'.lower()
    if not _configured_https_request(request):
        raise HTTPException(403, detail='evidence_origin_failed')
    supplied = request.headers.get('origin', '').rstrip('/').lower()
    if (require_header and not supplied) or (supplied and supplied != origin):
        raise HTTPException(403, detail='evidence_origin_failed')
    if request.headers.get('sec-fetch-site', '').lower() not in ('', 'same-origin'):
        raise HTTPException(403, detail='evidence_origin_failed')
    return origin


def _trusted_upload_origin(request):
    """Require the configured HTTPS origin for bearer-capability writes."""
    if not public_https_ingress_ready(request.app.state.config):
        raise HTTPException(503, detail='PUBLIC_INGRESS_PENDING')
    parsed = urlsplit(request.app.state.config.evidence_public_base_url)
    expected_origin = f'https://{_configured_https_authority(request.app.state.config)}'.lower()
    origin = request.headers.get('origin', '').rstrip('/').lower()
    if (not _configured_https_request(request)
            or not origin or origin != expected_origin
            or request.headers.get('sec-fetch-site', '').lower() not in ('', 'same-origin')):
        raise HTTPException(403, detail='evidence_origin_failed')


def _comment_timestamp(comment):
    value = comment.get('created_at') if isinstance(comment, dict) else None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


async def _comment_proof_principal(request, repo, pr_number):
    config = request.app.state.config
    proxy = request.app.state.proxy
    secret = request.headers.get(PROOF_HEADER, '')
    nonce = request.headers.get(NONCE_HEADER, '')
    if (not re.fullmatch(r'[0-9a-f]{64}', secret)
            or not re.fullmatch(r'[0-9a-f]{64}', nonce)):
        raise HTTPException(401, detail='evidence_comment_proof_required')
    proof = hashlib.sha256(bytes.fromhex(secret)).hexdigest()
    try:
        pr = await proxy.get_pr(repo, pr_number)
    except Exception as exc:
        if getattr(exc, 'status_code', None) == 404:
            raise HTTPException(404, detail='evidence_pr_not_found') from None
        raise HTTPException(503, detail='evidence_pr_unavailable') from None
    if not isinstance(pr, dict) or pr.get('state') not in {'open', 'closed'}:
        raise HTTPException(503, detail='evidence_pr_status_unavailable')
    head = pr.get('head')
    head_sha = head.get('sha') if isinstance(head, dict) else None
    base = pr.get('base')
    base_repo = base.get('repo') if isinstance(base, dict) else None
    if pr.get('draft'):
        raise HTTPException(409, detail='evidence_pr_draft')
    if (not isinstance(head_sha, str) or not re.fullmatch(r'[0-9a-f]{40}', head_sha)
            or not isinstance(base_repo, dict)
            or base_repo.get('full_name', '').casefold() != repo.casefold()):
        raise HTTPException(409, detail='evidence_pr_context_mismatch')
    state = await proxy.read_hidden_state(repo, pr_number)
    if not isinstance(state, dict):
        raise HTTPException(409, detail='evidence_state_unavailable')
    from .evidence_models import round_id_for_state
    round_id = round_id_for_state(state)
    if (state.get('head_sha') != head_sha or state.get('status') != 'testing'
            or not isinstance(round_id, str)):
        raise HTTPException(409, detail='evidence_round_changed')
    command = f'/authorize_evidence head={head_sha} round={round_id} proof={proof}'
    comments = await proxy.get_issue_comments(repo, pr_number)
    matches = [row for row in comments if isinstance(row, dict) and row.get('body') == command]
    if len(matches) != 1:
        raise HTTPException(403, detail='evidence_comment_proof_not_found')
    row = matches[0]
    comment_id = row.get('id')
    if type(comment_id) is not int or comment_id <= 0:
        raise HTTPException(403, detail='evidence_comment_proof_invalid')
    created = _comment_timestamp(row)
    updated = row.get('updated_at')
    now = time.time()
    if (created is None or now - created > PROOF_SECONDS or created > now + 30
            or (updated and updated != row.get('created_at'))):
        raise HTTPException(403, detail='evidence_comment_proof_expired')
    current = await proxy.get_comment(repo, comment_id)
    if (not isinstance(current, dict) or current.get('id') != comment_id
            or current.get('body') != command
            or current.get('issue_url') != f'{config.github_api_url.rstrip("/")}/repos/{repo}/issues/{pr_number}'):
        raise HTTPException(403, detail='evidence_comment_proof_invalid')
    current_created = _comment_timestamp(current)
    if (current_created is None or abs(current_created - created) > 1
            or (current.get('updated_at') and current.get('updated_at') != current.get('created_at'))):
        raise HTTPException(403, detail='evidence_comment_proof_invalid')
    uid, login = await proxy.comment_identity(repo, comment_id)
    if not isinstance(uid, str) or not uid.isdigit() or int(uid) <= 0:
        raise HTTPException(403, detail='evidence_comment_identity_invalid')
    if not isinstance(login, str) or not re.fullmatch(r'[A-Za-z0-9-]{1,39}', login):
        raise HTTPException(403, detail='evidence_comment_identity_invalid')
    # Every authenticated read/write after proof issuance is same-origin and
    # tied to the configured HTTPS ingress; Host and forwarded headers are ignored.
    _trusted_origin(request, require_header=request.method in ('POST', 'PUT', 'DELETE'))
    return {
        'id': int(uid), 'login': login, 'actor': login,
        'auth_method': 'github_comment_proof', 'upload_method': 'github_comment_proof',
        'actor_verified': True, 'github_user_authenticated': True,
        'evidence_auth': {'comment_id': comment_id, 'repo': repo,
                          'pr_number': pr_number, 'head_sha': head_sha,
                          'round_id': round_id, 'proof': proof,
                          'nonce': nonce, 'created_at': row.get('created_at')},
    }


async def verified_principal(request: Request, repo: str, pr_number: int, *, write=False,
                             purpose=PURPOSE, resource=None):
    require_enabled(request)
    if (not write and request.method == 'GET'
            and request.url.path.endswith('/context')
            and '1' in request.query_params.getlist('read_view')):
        page_token = request.cookies.get(CAPABILITY_COOKIE, '')
        if page_token:
            try:
                claims = _verify_page_capability(
                    request.app.state.config, page_token, repo, pr_number,
                )
            except HTTPException:
                claims = None
            requested_head = request.query_params.get('head_sha')
            requested_round = request.query_params.get('round_id')
            if (claims is not None
                    and requested_head == claims.get('head')
                    and requested_round == claims.get('round')):
                actor_id = int.from_bytes(
                    hashlib.sha256(page_token.encode()).digest()[:8], 'big',
                ) or 1
                return ({
                    'id': actor_id, 'login': 'evidence-capability',
                    'actor': 'evidence-capability', 'capability': claims,
                    'capability_token': page_token, 'auth_method': 'capability',
                    'actor_verified': False, 'github_user_authenticated': False,
                }, '')
        # An explicit read_view request may not fall through to an upload
        # invitation or comment proof when its read capability is absent,
        # expired, or bound to another round.
        raise HTTPException(403, detail='invalid_evidence_capability')
    if request.headers.get(PROOF_HEADER):
        if write and request.method not in ('POST', 'PUT', 'DELETE'):
            raise HTTPException(405, detail='evidence_write_method_required')
        if write:
            _trusted_origin(request)
        return await _comment_proof_principal(request, repo, pr_number), ''
    invite_token = request.cookies.get(UPLOAD_COOKIE, '')
    if invite_token and purpose in {PURPOSE, UPLOAD_PURPOSE}:
        try:
            claims = _verify_upload_capability(request.app.state.config, invite_token, repo, pr_number)
        except HTTPException:
            claims = None
        if (claims is not None and not write and request.method == 'GET'
                and request.url.path.endswith('/context')):
            requested_head = request.query_params.get('head_sha')
            requested_round = request.query_params.get('round_id')
            if ((requested_head and requested_head != claims.get('head'))
                    or (requested_round and requested_round != claims.get('round'))):
                claims = None
        if claims is not None:
            if write:
                if request.method not in ('POST', 'PUT', 'DELETE'):
                    raise HTTPException(405, detail='evidence_write_method_required')
                _trusted_upload_origin(request)
            client_key = request.headers.get('x-evidence-client-key', '')
            if not re.fullmatch(r'[0-9a-f]{64}', client_key):
                if (request.method == 'GET' and request.url.path.startswith('/evidence/')
                        and not request.url.path.startswith('/api/')):
                    return {'id': 0, 'login': 'shared-link', 'actor': 'shared-link',
                            'upload_invite': claims, 'upload_method': 'signed_invite',
                            'auth_method': 'signed_invite', 'actor_verified': False,
                            'github_user_authenticated': False}, ''
                raise HTTPException(401, detail='evidence_client_key_required')
            scope = '\0'.join((repo, str(pr_number), claims['head'], claims['round']))
            identity = hmac.new(_key(request.app.state.config),
                                b'phanthymotus/evidence-browser/v1\0'
                                + scope.encode('utf-8') + b'\0' + bytes.fromhex(client_key),
                                hashlib.sha256).digest()
            pseudo_id = int.from_bytes(identity[:16], 'big') or 1
            return {'id': pseudo_id, 'login': 'shared-link', 'actor': 'shared-link',
                    'upload_invite': claims, 'upload_method': 'signed_invite',
                    'auth_method': 'signed_invite', 'actor_verified': False,
                    'github_user_authenticated': False}, ''
    if (not write and request.method == 'GET'
            and request.url.path.endswith('/context')):
        page_token = request.cookies.get(CAPABILITY_COOKIE, '')
        if page_token:
            try:
                claims = _verify_page_capability(
                    request.app.state.config, page_token, repo, pr_number,
                )
            except HTTPException:
                pass
            else:
                requested_head = request.query_params.get('head_sha')
                requested_round = request.query_params.get('round_id')
                if ((requested_head and requested_head != claims.get('head'))
                        or (requested_round and requested_round != claims.get('round'))):
                    page_token = ''
                else:
                    actor_id = int.from_bytes(
                        hashlib.sha256(page_token.encode()).digest()[:8], 'big',
                    ) or 1
                    return ({
                        'id': actor_id, 'login': 'evidence-capability',
                        'actor': 'evidence-capability', 'capability': claims,
                        'capability_token': page_token, 'auth_method': 'capability',
                        'actor_verified': False, 'github_user_authenticated': False,
                    }, '')
    if write:
        if purpose == UPLOAD_PURPOSE:
            _trusted_upload_origin(request)
        else:
            _trusted_origin(request)
        raise HTTPException(401, detail='evidence_comment_proof_required')
    config = request.app.state.config
    token = request.query_params.get('cap') or request.cookies.get(CAPABILITY_COOKIE, '')
    if (purpose in {PURPOSE, UPLOAD_PURPOSE} and request.method == 'GET'
            and request.url.path.endswith('/context')):
        claims = _verify_page_capability(config, token, repo, pr_number)
    elif purpose == DOWNLOAD_PURPOSE:
        try:
            claims = verify_capability(config, token, repo, pr_number,
                                       purpose=DOWNLOAD_PURPOSE, resource=resource)
        except HTTPException:
            # Existing unscoped public capabilities remain strictly read-only.
            claims = verify_capability(config, token, repo, pr_number, purpose=PURPOSE)
    else:
        claims = verify_capability(config, token, repo, pr_number,
                                   purpose=purpose, resource=resource)
    if (purpose in {PURPOSE, UPLOAD_PURPOSE} and request.method == 'GET'
            and request.url.path.endswith('/context')
            and ((request.query_params.get('head_sha')
                  and request.query_params.get('head_sha') != claims.get('head'))
                 or (request.query_params.get('round_id')
                     and request.query_params.get('round_id') != claims.get('round')))):
        raise HTTPException(403, detail='invalid_evidence_capability')
    actor_id = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big') or 1
    return {'id': actor_id, 'login': 'evidence-capability', 'actor': 'evidence-capability',
            'capability': claims, 'capability_token': token,
            'auth_method': 'capability', 'actor_verified': False,
            'github_user_authenticated': False}, ''
