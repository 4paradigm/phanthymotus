"""Stable authenticated evidence HTTP contract; business mutations belong to A."""
from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import parse_qs, urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from . import evidence_auth
from .config import validate_repository_names
from .evidence_store import EvidenceStoreError, safe_filename

router = APIRouter()
_PREFIX = '/api/evidence/{owner}/{repo}/pulls/{pr}'
_ID = re.compile(r'[A-Za-z0-9_-]{1,128}')
_BASE = {'head_sha', 'round_id', 'idempotency_key'}


def _identity(value, label):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise HTTPException(400, detail='invalid_' + label)
    return value


def _repo(owner, repo, pr):
    try:
        canonical = validate_repository_names([f'{owner}/{repo}'])[0]
    except ValueError:
        raise HTTPException(400, detail='invalid_repository') from None
    if pr < 1:
        raise HTTPException(400, detail='invalid_pr')
    return canonical


def _byte_length(value):
    try:
        return len(value.encode('utf-8'))
    except UnicodeError:
        raise HTTPException(400, detail='invalid_text_encoding') from None


def _validate(payload, action, config):
    extras = {'init': {'filename', 'size', 'content_type', 'note', 'replaces_id', 'sha256', 'submitter'},
              'complete': {'sha256', 'parts'}, 'note': {'text', 'note', 'replaces_id', 'submitter'}}.get(action, set())
    route_fields = {'status': {'upload_id'}, 'resume': {'upload_id'}, 'sign_part': {'upload_id', 'part_number'},
                    'put_part': {'upload_id', 'part_number'},
                    'complete': {'upload_id'}, 'download': {'attachment_id'},
                    'preview': {'attachment_id'},
                    'delete': {'attachment_id'}}.get(action, set())
    if not isinstance(payload, dict) or payload.keys() - (_BASE | extras | route_fields):
        raise HTTPException(400, detail='invalid_evidence_payload')
    if 'round_id' in payload:
        _identity(payload['round_id'], 'round_id')
    if 'head_sha' in payload:
        if not isinstance(payload.get('head_sha'), str) or not re.fullmatch('[0-9a-f]{40}', payload['head_sha']):
            raise HTTPException(400, detail='invalid_head_sha')
        _identity(payload.get('round_id'), 'round_id')

    if 'idempotency_key' in payload:
        idem = payload['idempotency_key']
        if not isinstance(idem, str) or not 1 <= len(idem) <= 1024 or any(ord(c) < 32 for c in idem):
            raise HTTPException(400, detail='invalid_idempotency_key')
        payload['idempotency_key'] = hashlib.sha256(idem.encode('utf-8')).hexdigest()
    if 'parts' in payload:
        # COS ListParts is authoritative; browser ETags are only bounded hints.
        if not isinstance(payload['parts'], list) or len(payload['parts']) > 10000:
            raise HTTPException(400, detail='invalid_parts')
        payload.pop('parts')
    if action == 'init':
        try:
            safe_filename(payload.get('filename'))
        except ValueError:
            raise HTTPException(400, detail='invalid_filename') from None
        size = payload.get('size')
        video_limit = getattr(config, 'evidence_max_video_bytes', config.evidence_max_file_bytes)
        is_video = isinstance(payload.get('content_type'), str) and payload['content_type'].lower().startswith('video/')
        limit = video_limit if is_video else config.evidence_max_file_bytes
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= limit:
            raise HTTPException(413, detail='invalid_file_size')
        content_type = payload.get('content_type')
        if not isinstance(content_type, str) or not re.fullmatch(r'[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+', content_type):
            raise HTTPException(400, detail='invalid_content_type')
    for field in ('note', 'text'):
        if field in payload and (not isinstance(payload[field], str) or
                               len(payload[field]) > 5000 or
                               _byte_length(payload[field]) > config.evidence_max_note_bytes):
            raise HTTPException(413, detail='invalid_note_size')
    if action == 'note' and not (payload.get('text') or payload.get('note')):
        raise HTTPException(400, detail='note_required')
    for field in ('upload_id', 'attachment_id', 'replaces_id'):
        if payload.get(field) is not None:
            _identity(payload[field], field)
    if 'sha256' in payload and (not isinstance(payload['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', payload['sha256'])):
        raise HTTPException(400, detail='invalid_sha256')
    if 'part_number' in payload and not 1 <= payload['part_number'] <= 10000:
        raise HTTPException(400, detail='invalid_part_number')
    if action == 'note':
        text = payload.get('text', '')
        note = payload.get('note', '')
        if note:
            text = (text + '\n\n' + note) if text else note
        if (not 1 <= len(text) <= 5000
                or _byte_length(text) > config.evidence_max_note_bytes):
            raise HTTPException(413, detail='invalid_note_size')
        payload.update(text=text, size=_byte_length(text),
                       sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    return payload


async def _payload(request, action, **path):
    if request.method in ('POST', 'DELETE'):
        # Bound text input before parsing; uploads themselves go directly to COS.
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 512 * 1024:
                raise HTTPException(413, detail='evidence_request_too_large')
        try:
            payload = json.loads(data) if data else {}
        except (ValueError, UnicodeError):
            raise HTTPException(400, detail='invalid_json') from None
    else:
        payload = dict(request.query_params)
        payload.pop('cap', None)
        if action == 'put_part' and payload.keys() - {'head_sha', 'round_id'}:
            raise HTTPException(400, detail='invalid_evidence_payload')
    if not isinstance(payload, dict) or any(field in payload for field in path):
        raise HTTPException(400, detail='invalid_evidence_payload')
    # Older browser JS sends this display-only value. It is discarded before
    # validation/dispatch and can never become a persisted actor identity.
    payload.pop('submitter', None)
    if request.method in ('POST', 'DELETE') and not _BASE <= payload.keys():
        raise HTTPException(400, detail='evidence_context_required')
    payload.update(path)
    return _validate(payload, action, request.app.state.config)


def _public_attachment(row, *, operation_context=None):
    allowed = {'attachment_id', 'filename', 'size', 'sha256', 'note', 'display_note',
               'display_note_complete', 'evidence_kind', 'round_id', 'head_sha',
               'status', 'created_at', 'updated_at', 'completed_at', 'deleted_at', 'replaces_id',
               'actor', 'actor_login', 'actor_verified', 'cos_content_verified',
               'submitter', 'upload_method', 'download_capability', 'can_delete'}
    result = {k: v for k, v in row.items() if k in allowed}
    upload_ref = row.get('upload_ref')
    if (operation_context is not None and isinstance(upload_ref, str)
            and _ID.fullmatch(upload_ref)
            and row.get('repo') == operation_context.get('repo')
            and row.get('pr_number') == operation_context.get('pr_number')
            and row.get('head_sha') == operation_context.get('head_sha')
            and row.get('round_id') == operation_context.get('round_id')):
        # This is the durable, scoped operation handle; the COS multipart
        # UploadId and other internal upload facts are never exposed.
        result['upload_id'] = upload_ref
    return result


async def _summary_updated(request, repo, pr, result):
    # This is a read-only comparison against the trusted lifecycle comment.
    # Store verification alone does not prove the watcher updated Summary.
    try:
        from .github_state_proxy import HIDDEN_STATE_MARKER, _extract_hidden_state
        comment = await request.app.state.proxy.find_trusted_lifecycle_comment(repo, pr)
        if comment is None:
            return False
        body = comment['body']
        state = _extract_hidden_state(body)
        if not state or state.get('head_sha') != result['head_sha']:
            return False
        visible = body.split(HIDDEN_STATE_MARKER, 1)[0].rstrip()
        expected = await request.app.state.controller._render_evidence_projection(repo, pr, state, visible)
        return expected == visible and '<!-- evidence-projection:start -->' in visible
    except Exception:
        return False


async def _trusted_summary_page_capability(request, repo, pr):
    """Recover only a currently valid page capability from the trusted Summary."""
    config = request.app.state.config
    if (not evidence_auth.public_https_ingress_ready(config)
            or not evidence_auth._configured_https_request(request)):
        return None
    proxy = request.app.state.proxy
    try:
        comment = await proxy.find_trusted_lifecycle_comment(repo, pr)
    except Exception:
        raise HTTPException(503, detail='evidence_pr_unavailable') from None
    if not isinstance(comment, dict):
        return None
    expected_issue_url = (
        f'{config.github_api_url.rstrip("/")}/repos/{repo}/issues/{pr}'
    )
    if comment.get('issue_url') != expected_issue_url:
        return None
    body = comment.get('body')
    if not isinstance(body, str):
        return None
    from .github_state_proxy import HIDDEN_STATE_MARKER, _extract_hidden_state, _validate_hidden_state
    marker_pos = body.find(HIDDEN_STATE_MARKER)
    if marker_pos < 0:
        return None
    try:
        state = _validate_hidden_state(_extract_hidden_state(body))
    except Exception:
        raise HTTPException(503, detail='evidence_state_unavailable') from None
    if not isinstance(state, dict):
        raise HTTPException(503, detail='evidence_state_unavailable')
    try:
        current_pr = await proxy.get_pr(repo, pr)
    except Exception as exc:
        if getattr(exc, 'status_code', None) == 404:
            raise HTTPException(404, detail='evidence_pr_not_found') from None
        raise HTTPException(503, detail='evidence_pr_unavailable') from None
    if not isinstance(current_pr, dict) or current_pr.get('state') not in {'open', 'closed'}:
        raise HTTPException(503, detail='evidence_pr_status_unavailable')
    base = current_pr.get('base')
    base_repository = base.get('repo') if isinstance(base, dict) else None
    base_name = base_repository.get('full_name') if isinstance(base_repository, dict) else None
    if current_pr.get('draft') or not isinstance(base_name, str) or base_name.casefold() != repo.casefold():
        raise HTTPException(409, detail='evidence_pr_context_changed')

    from .evidence_models import round_id_for_state
    try:
        head, round_id = state.get('head_sha'), round_id_for_state(state)
    except Exception:
        raise HTTPException(503, detail='evidence_state_unavailable') from None
    if (not isinstance(head, str) or not re.fullmatch(r'[0-9a-f]{40}', head)
            or state.get('status') not in {
                'review-required', 'reviewing', 'deploy-ready', 'deploy-requested',
                'deploying', 'testing', 'succeeded', 'failed',
            }):
        return None
    pr_head = current_pr.get('head')
    pr_head_sha = pr_head.get('sha') if isinstance(pr_head, dict) else None
    if pr_head_sha != head:
        raise HTTPException(409, detail='evidence_pr_context_changed')
    configured_base = urlsplit(config.evidence_public_base_url)
    expected_authorities = {
        configured_base.netloc.casefold(),
        evidence_auth._configured_https_authority(config).casefold(),
    }
    page_path = '/evidence/' + repo + '/pulls/' + str(pr)
    visible = body[:marker_pos]
    page_capability = None
    for match in re.finditer(r'\[打开附件页\]\(<(?P<url>https://[^<>\s]+)>\)', visible):
        candidate_text = match.group('url')
        try:
            candidate = urlsplit(candidate_text)
            if (candidate.scheme != 'https' or candidate.username or candidate.password
                    or candidate.fragment or candidate.path != page_path
                    or candidate.netloc.casefold() not in expected_authorities
                    or candidate.port not in (None, 443)):
                continue
            query = parse_qs(candidate.query, strict_parsing=True)
            if set(query) != {'cap'} or len(query['cap']) != 1 or not query['cap'][0]:
                continue
            token = query['cap'][0]
        except (ValueError, TypeError):
            continue
        claims = None
        is_upload = False
        try:
            claims = evidence_auth.verify_capability(
                config, token, repo, pr, purpose=evidence_auth.UPLOAD_PURPOSE,
            )
            is_upload = True
        except HTTPException:
            for purpose, resource in ((evidence_auth.DOWNLOAD_PURPOSE, 'page'),
                                      (evidence_auth.PURPOSE, None)):
                try:
                    claims = evidence_auth.verify_capability(
                        config, token, repo, pr, purpose=purpose, resource=resource,
                    )
                    break
                except HTTPException:
                    continue
        if (not isinstance(claims, dict) or claims.get('head') != head
                or claims.get('round') != round_id):
            continue
        if is_upload:
            # A bare URL is never upgraded to the bearer upload invitation.
            # The explicit Summary URL can bind that invite; URL recovery only
            # restores read-only history access.
            if (state.get('status') != 'testing' or pr_head_sha != head
                    or current_pr.get('state') != 'open' or current_pr.get('merged')
                    or current_pr.get('draft') or claims.get('status') != 'testing'):
                continue
            continue
        if page_capability is None:
            page_capability = token
    if page_capability:
        return page_capability
    # If the trusted Summary has no usable invitation, a page-only token may
    # still provide read access. Its purpose is explicit and cannot authorize
    # any Evidence write.
    try:
        token = evidence_auth.issue_capability(
            config, repo, pr, head, round_id,
            purpose=evidence_auth.DOWNLOAD_PURPOSE, resource='page',
        )
        evidence_auth.verify_capability(
            config, token, repo, pr,
            purpose=evidence_auth.DOWNLOAD_PURPOSE, resource='page',
        )
        return token
    except (TypeError, ValueError, HTTPException):
        return None


async def _dispatch(request, owner, repo, pr, action, **path):
    canonical = _repo(owner, repo, pr)
    purpose = (evidence_auth.DOWNLOAD_PURPOSE if action in {'download', 'preview', 'log_download'}
               else evidence_auth.UPLOAD_PURPOSE if action in {
                   'context', 'status', 'init', 'resume', 'sign_part', 'put_part',
                   'complete', 'note', 'delete'} else evidence_auth.PURPOSE)
    resource = None
    if action in {'download', 'preview'}:
        resource = 'attachment:' + path.get('attachment_id', '')
    elif action == 'log_download':
        resource = 'log:' + path.get('round_id', '')
    principal, _ = await evidence_auth.verified_principal(
        request, canonical, pr, write=request.method in ('POST', 'PUT', 'DELETE'),
        purpose=purpose, resource=resource)
    # The EvidenceService reads and validates the live PR before any operation.
    # Do not double-fetch it here: additional GitHub calls delay uploads and
    # turn transient PR reads into misleading 'evidence_pr_unavailable' errors.
    payload = await _payload(request, action, **path)
    claims = principal.get('capability') or principal.get('upload_invite') or {
        'head': payload.get('head_sha'), 'round': payload.get('round_id')}
    if action == 'context':
        # The page capability selects the current round or a specific archived
        # round. The EvidenceService still validates that binding against the
        # trusted lifecycle snapshot before returning any records.
        payload.setdefault('head_sha', claims.get('head'))
        payload.setdefault('round_id', claims.get('round'))
    if 'upload_id' in payload:
        # The public upload handle is A's durable operation identity, rather
        # than an unauthenticated client-selected COS UploadId.
        payload['operation_id'] = payload.pop('upload_id')
    if action in {'download', 'preview'} and 'round_id' not in payload:
        records = await request.app.state.proxy.read_evidence_records(canonical, pr)
        candidates = [r['data'] for r in records if r['kind'] == 'attachment'
                      and r['data'].get('attachment_id') == payload['attachment_id']]
        if candidates:
            row = candidates[-1]
            payload.update(head_sha=row['head_sha'], round_id=row['round_id'])
    if action in {'resume', 'download', 'preview', 'log_download'}:
        if action == 'resume' and not _BASE <= payload.keys():
            raise HTTPException(400, detail='evidence_context_required')
    if action in {'init', 'resume', 'sign_part', 'put_part', 'complete', 'note', 'delete'}:
        if action == 'put_part' and not {'head_sha', 'round_id'} <= payload.keys():
            raise HTTPException(400, detail='evidence_context_required')
        payload.setdefault('head_sha', claims['head'])
        payload.setdefault('round_id', claims['round'])
        # Compatibility with the current browser UI: a display name is not
        # an identity source. The service records only the verified /user login.
        payload.pop('submitter', None)
    if action in {'download', 'preview', 'log_download'}:
        payload.setdefault('idempotency_key', hashlib.sha256(
            json.dumps([action, path], sort_keys=True).encode()).hexdigest())
    try:
        if action == 'put_part':
            cap = min(8 * 1024 * 1024, request.app.state.config.evidence_part_bytes)
            content_length = request.headers.get('content-length')
            if content_length:
                try:
                    if int(content_length) > cap:
                        raise HTTPException(413, detail='evidence_part_too_large')
                except ValueError:
                    raise HTTPException(400, detail='invalid_content_length') from None
            payload['_body_stream'] = request.stream()
            result = await request.app.state.evidence_service.dispatch(
                action, repo=canonical, pr_number=pr, principal=principal, payload=payload)
        else:
            result = await request.app.state.evidence_service.dispatch(
                action, repo=canonical, pr_number=pr, principal=principal, payload=payload)
    except EvidenceStoreError:
        raise HTTPException(503, detail='evidence_object_unavailable') from None
    except Exception as exc:
        # Only the service's explicitly public error code crosses HTTP. SDK and
        # network diagnostics can contain signed URLs or authentication headers.
        from .evidence_models import EvidenceError
        if isinstance(exc, EvidenceError):
            raise HTTPException(exc.status_code, detail=exc.code) from None
        raise HTTPException(503, detail='evidence_operation_unavailable') from None
    if not isinstance(result, dict):
        raise HTTPException(503, detail='invalid_evidence_service_response')
    # Internal recovery facts (COS keys, actor IDs) never need to reach the
    # browser. A durable operation ID is the only browser upload handle.
    if action in {'init', 'resume', 'complete', 'note'}:
        result = {**result, 'upload_id': path.get('upload_id') or result.get('source_operation_id', result.get('operation_id')),
                  'part_bytes': result.get('part_bytes', request.app.state.config.evidence_part_bytes),
                  'uploaded_parts': result.get('uploaded_parts', [])}
    if action == 'status':
        # Status is a deliberately small public projection. In particular, a
        # COS multipart UploadId and signed URLs are never public state.
        result = {k: result[k] for k in ('status', 'uploaded_parts', 'size', 'filename',
                                         'head_sha', 'round_id', 'verified', 'sha256') if k in result}
    if action == 'context':
        def scoped(rows):
            from .evidence_auth import issue_capability
            output = []
            for row in rows:
                item = dict(row)
                head = item.get('head_sha')
                round_value = item.get('round_id')
                digest = item.get('sha256')
                if (item.get('status') == 'ready' and not item.get('deleted_at')
                        and isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest)
                        and isinstance(head, str) and re.fullmatch(r'[0-9a-f]{40}', head)
                        and isinstance(round_value, str)
                        and re.fullmatch(re.escape(head) + r'-[0-9]+', round_value)
                        and isinstance(item.get('attachment_id'), str)
                        and _ID.fullmatch(item['attachment_id'])):
                    item['download_capability'] = issue_capability(
                        request.app.state.config, canonical, pr, head, round_value,
                        purpose=evidence_auth.DOWNLOAD_PURPOSE,
                        resource='attachment:' + item['attachment_id'],
                    )
                output.append(_public_attachment(item))
            return output
        rows = result.get('attachments', result.get('records', []))
        attachments = scoped(rows)
        rounds = []
        for row in result.get('rounds', []):
            round_value = {**{k: v for k, v in row.items() if k in {
                'head_sha', 'round_id', 'round_number', 'status', 'created_at', 'started_at',
                'round_started_at', 'ended_at', 'archived_at', 'acceptance_pending',
                'display_deployments'}},
                'attachments': scoped(row.get('attachments', [])),
                'log_available': bool(row.get('cos', {}).get('object_key'))}
            head, round_id = round_value.get('head_sha'), round_value.get('round_id')
            if isinstance(head, str) and isinstance(round_id, str) and head == round_id.rsplit('-', 1)[0]:
                from .evidence_auth import issue_capability
                round_value['capability'] = issue_capability(
                    request.app.state.config, canonical, pr, head, round_id,
                    purpose=evidence_auth.DOWNLOAD_PURPOSE,
                    resource='log:' + round_id,
                )
            rounds.append(round_value)
        result['rounds'] = rounds
        operation_context = {key: result.get(key) for key in ('repo', 'pr_number', 'head_sha', 'round_id')}
        result['operations'] = [
            _public_attachment(row, operation_context=operation_context)
            for row in result.get('operations', [])
        ]
        result['summary_updated'] = await _summary_updated(request, canonical, pr, result)
        result = {**result, 'authenticated': True, 'enabled': True,
                  'github_user_authenticated': bool(principal.get('github_user_authenticated')),
                  'actor_verified': principal.get('actor_verified') is True,
                  'auth_method': principal.get('auth_method', 'github_comment_proof'),
                  'readiness': evidence_auth.readiness(
                      request.app.state.config, request.app.state.evidence_service, request=request),
                  'attachments': attachments, 'repo': canonical, 'pr_number': pr,
                  # PR state is returned from the EvidenceService's validated
                  # snapshot; no extra GitHub request is needed here.
                  'max_file_bytes': request.app.state.config.evidence_max_file_bytes,
                  'part_bytes': request.app.state.config.evidence_part_bytes}
        if not {'head_sha', 'round_id', 'status', 'can_write', 'attachments', 'archive_links'} <= result.keys():
            raise HTTPException(503, detail='incomplete_evidence_context')
    for name in ('object_key', 'upload_ref', 'operation_id', 'source_operation_id',
                 'actor_id', 'actor_login', 'deleted_by', 'records', 'request_sha256'):
        result.pop(name, None)
    return result


def _json(result):
    return JSONResponse(result, headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
                                         'X-Content-Type-Options': 'nosniff'})


@router.get(_PREFIX + '/context')
async def context(request: Request, owner: str, repo: str, pr: int):
    return _json(await _dispatch(request, owner, repo, pr, 'context'))


@router.post(_PREFIX + '/uploads')
async def init(request: Request, owner: str, repo: str, pr: int):
    return _json(await _dispatch(request, owner, repo, pr, 'init'))


@router.get(_PREFIX + '/uploads/{upload_id}')
async def upload_status(request: Request, owner: str, repo: str, pr: int, upload_id: str):
    return _json(await _dispatch(request, owner, repo, pr, 'status', upload_id=upload_id))


@router.post(_PREFIX + '/uploads/{upload_id}/resume')
async def resume(request: Request, owner: str, repo: str, pr: int, upload_id: str):
    return _json(await _dispatch(request, owner, repo, pr, 'resume', upload_id=upload_id))


@router.post(_PREFIX + '/uploads/{upload_id}/parts/{part_number}/url')
async def sign_part(request: Request, owner: str, repo: str, pr: int, upload_id: str, part_number: int):
    return _json(await _dispatch(request, owner, repo, pr, 'sign_part', upload_id=upload_id, part_number=part_number))


@router.put(_PREFIX + '/uploads/{upload_id}/parts/{part_number}/data')
async def upload_part(request: Request, owner: str, repo: str, pr: int, upload_id: str, part_number: int):
    return _json(await _dispatch(request, owner, repo, pr, 'put_part', upload_id=upload_id, part_number=part_number))


@router.post(_PREFIX + '/uploads/{upload_id}/complete')
async def complete(request: Request, owner: str, repo: str, pr: int, upload_id: str):
    return _json(await _dispatch(request, owner, repo, pr, 'complete', upload_id=upload_id))


@router.post(_PREFIX + '/notes')
async def note(request: Request, owner: str, repo: str, pr: int):
    return _json(await _dispatch(request, owner, repo, pr, 'note'))


@router.delete(_PREFIX + '/attachments/{attachment_id}')
async def delete(request: Request, owner: str, repo: str, pr: int, attachment_id: str):
    return _json(await _dispatch(request, owner, repo, pr, 'delete', attachment_id=attachment_id))


async def _download(request, owner, repo, pr, action, **path):
    result = await _dispatch(request, owner, repo, pr, action, **path)
    url = result.get('url')
    if not isinstance(url, str) or not url.startswith('https://'):
        raise HTTPException(503, detail='evidence_download_unavailable')
    return RedirectResponse(url, status_code=303,
                            headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})


@router.get(_PREFIX + '/attachments/{attachment_id}/download')
async def download(request: Request, owner: str, repo: str, pr: int, attachment_id: str):
    return await _download(request, owner, repo, pr, 'download', attachment_id=attachment_id)


@router.get(_PREFIX + '/attachments/{attachment_id}/preview')
async def preview(request: Request, owner: str, repo: str, pr: int, attachment_id: str):
    return await _download(request, owner, repo, pr, 'preview', attachment_id=attachment_id)


@router.get(_PREFIX + '/rounds/{round_id}/log/download')
async def log_download(request: Request, owner: str, repo: str, pr: int, round_id: str):
    return await _download(request, owner, repo, pr, 'log_download', round_id=round_id)


@router.get('/evidence/{owner}/{repo}/pulls/{pr}', response_class=HTMLResponse)
async def page(request: Request, owner: str, repo: str, pr: int):
    headers = {'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
               'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY'}
    canonical = None
    try:
        evidence_auth.require_enabled(request)
        canonical = _repo(owner, repo, pr)
        explicit_capability = 'cap' in request.query_params
        page_cookie = request.cookies.get(evidence_auth.CAPABILITY_COOKIE, '')
        upload_cookie = request.cookies.get(evidence_auth.UPLOAD_COOKIE, '')
        cookie_token = page_cookie or upload_cookie
        cookie_claims = None
        if cookie_token:
            try:
                cookie_claims = (
                    evidence_auth._verify_page_capability(
                        request.app.state.config, page_cookie, canonical, pr,
                    ) if page_cookie else evidence_auth._verify_upload_capability(
                        request.app.state.config, upload_cookie, canonical, pr,
                    )
                )
            except HTTPException:
                pass
        summary_capability = None
        summary_checked = False
        if not explicit_capability and (cookie_claims is None or bool(page_cookie)):
            try:
                summary_capability = await _trusted_summary_page_capability(request, canonical, pr)
            except HTTPException as exc:
                if cookie_claims is None or exc.status_code != 503:
                    raise
            summary_checked = True
            if cookie_claims and summary_capability:
                try:
                    summary_claims = evidence_auth.verify_capability(
                        request.app.state.config, summary_capability, canonical, pr,
                    )
                except HTTPException:
                    summary_claims = None
                if (not isinstance(summary_claims, dict)
                        or (summary_claims.get('purpose') != evidence_auth.UPLOAD_PURPOSE
                            and summary_claims.get('head') == cookie_claims.get('head')
                            and summary_claims.get('round') == cookie_claims.get('round'))):
                    # Keep an already-valid cookie when Summary offers no
                    # newer upload invite or a different read-only binding.
                    summary_capability = None
        try:
            redirect = evidence_auth.bind_page_capability(
                request, canonical, pr, trusted_summary_capability=summary_capability,
            )
        except HTTPException as exc:
            if explicit_capability or summary_checked or exc.status_code != 403:
                raise
            summary_capability = await _trusted_summary_page_capability(request, canonical, pr)
            summary_checked = True
            if not summary_capability:
                raise
            redirect = evidence_auth.bind_page_capability(
                request, canonical, pr, trusted_summary_capability=summary_capability,
            )
        if redirect is not None:
            return redirect
        cookie_token = request.cookies.get(evidence_auth.CAPABILITY_COOKIE, '')
        if cookie_token:
            claims = evidence_auth._verify_page_capability(
                request.app.state.config, cookie_token, canonical, pr,
            )
        else:
            cookie_token = request.cookies.get(evidence_auth.UPLOAD_COOKIE, '')
            claims = evidence_auth._verify_upload_capability(
                request.app.state.config, cookie_token, canonical, pr,
            )
        result = {
            'repo': canonical,
            'head_sha': claims['head'],
            'round_id': claims['round'],
        }
    except HTTPException as exc:
        # Keep the authorization status and never echo exception detail,
        # capability input, cookies, or backend diagnostics into HTML.
        messages = {
            401: '附件链接身份验证失败，请从当前 PR Summary 重新打开。',
            403: '附件链接已失效或无权访问，请从当前 PR Summary 重新打开。',
            409: 'PR 或测试轮次已变化，请刷新当前 PR Summary 后重新打开。',
            503: '附件服务暂时不可用，请稍后从当前 PR Summary 重新打开。',
        }
        message = messages.get(exc.status_code, '附件链接暂不可用，请从当前 PR Summary 重新打开。')
        summary_link = ('<p><a href="https://github.com/' + canonical + '/pull/' + str(pr)
                        + '">打开当前 PR Summary</a></p>') if canonical else ''
        html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>附件链接不可用</title><main><h1>附件链接不可用</h1><p>'
                + message + '</p>' + summary_link + '</main></html>')
        return HTMLResponse(html, status_code=exc.status_code, headers=headers)
    from . import evidence_ui
    page_html = evidence_ui.render_upload_page(
        result['repo'], pr, result['head_sha'], result['round_id'])
    return HTMLResponse(page_html, headers=headers)


@router.get('/evidence/assets/app.js')
async def javascript(request: Request):
    evidence_auth.require_enabled(request)
    from . import evidence_ui
    return Response(evidence_ui.javascript_source(), media_type='application/javascript',
                    headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
