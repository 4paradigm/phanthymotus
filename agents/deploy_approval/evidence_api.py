"""Stable authenticated evidence HTTP contract; business mutations belong to A."""
from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlencode

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
    extras = {'init': {'filename', 'size', 'content_type', 'note', 'replaces_id', 'sha256'},
              'complete': {'sha256', 'parts'}, 'note': {'text', 'note', 'replaces_id'}}.get(action, set())
    route_fields = {'status': {'upload_id'}, 'resume': {'upload_id'}, 'sign_part': {'upload_id', 'part_number'},
                    'complete': {'upload_id'}, 'download': {'attachment_id'},
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
        if not 1 <= _byte_length(text) <= config.evidence_max_note_bytes:
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
    if not isinstance(payload, dict) or any(field in payload for field in path):
        raise HTTPException(400, detail='invalid_evidence_payload')
    if request.method in ('POST', 'DELETE') and not _BASE <= payload.keys():
        raise HTTPException(400, detail='evidence_context_required')
    payload.update(path)
    return _validate(payload, action, request.app.state.config)


def _public_attachment(row):
    allowed = {'attachment_id', 'filename', 'size', 'sha256', 'note', 'round_id', 'head_sha',
               'status', 'created_at', 'completed_at', 'deleted_at', 'replaces_id', 'actor', 'actor_login'}
    result = {k: v for k, v in row.items() if k in allowed}
    if row.get('upload_ref'):
        result['upload_id'] = row['upload_ref']
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


async def _dispatch(request, owner, repo, pr, action, **path):
    canonical = _repo(owner, repo, pr)
    principal, csrf = await evidence_auth.verified_principal(
        request, canonical, write=request.method in ('POST', 'DELETE'))
    # This read is deliberately before dispatch, including historic downloads;
    # a closed PR must never trigger a lifecycle write or a comment reply.
    try:
        snapshot = await request.app.state.github.get_pr(canonical, pr)
    except Exception:
        raise HTTPException(503, detail='evidence_pr_unavailable') from None
    if (snapshot.get('state') != 'open' or snapshot.get('merged') or
            snapshot.get('base', {}).get('repo', {}).get('full_name', '').lower() != canonical):
        raise HTTPException(409, detail='pr_closed_or_mismatched')
    payload = await _payload(request, action, **path)
    if 'upload_id' in payload:
        # The public upload handle is A's durable operation identity, rather
        # than an unauthenticated client-selected COS UploadId.
        payload['operation_id'] = payload.pop('upload_id')
    if action in {'resume', 'download', 'log_download'}:
        if action == 'resume' and not _BASE <= payload.keys():
            raise HTTPException(400, detail='evidence_context_required')
    if action in {'download', 'log_download'}:
        payload.setdefault('idempotency_key', hashlib.sha256(
            json.dumps([action, path], sort_keys=True).encode()).hexdigest())
    if action == 'download' and 'round_id' not in payload:
        records = await request.app.state.proxy.read_evidence_records(canonical, pr)
        candidates = [r['data'] for r in records if r['kind'] == 'attachment'
                      and r['data'].get('attachment_id') == payload['attachment_id']]
        if candidates:
            payload.update(head_sha=candidates[-1]['head_sha'], round_id=candidates[-1]['round_id'])
    try:
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
    if action in {'init', 'resume', 'complete'}:
        result = {**result, 'upload_id': path.get('upload_id') or result.get('source_operation_id', result.get('operation_id')),
                  'part_bytes': result.get('part_bytes', request.app.state.config.evidence_part_bytes),
                  'uploaded_parts': result.get('uploaded_parts', [])}
    if action == 'status':
        # Status is a deliberately small public projection. In particular, a
        # COS multipart UploadId and signed URLs are never public state.
        result = {k: result[k] for k in ('status', 'uploaded_parts', 'size', 'filename',
                                         'head_sha', 'round_id', 'verified') if k in result}
    if action == 'context':
        rows = result.get('attachments', result.get('records', []))
        attachments = [_public_attachment(row) for row in rows]
        rounds = []
        for row in result.get('rounds', []):
            rounds.append({**{k: v for k, v in row.items() if k in {
                'head_sha', 'round_id', 'status', 'created_at', 'started_at', 'ended_at', 'archived_at'}},
                'attachments': [_public_attachment(a) for a in row.get('attachments', [])],
                'log_available': bool(row.get('cos', {}).get('object_key'))})
        result['rounds'] = rounds
        result['operations'] = [_public_attachment(row) for row in result.get('operations', [])]
        result['summary_updated'] = await _summary_updated(request, canonical, pr, result)
        result = {**result, 'authenticated': True, 'enabled': True,
                  'can_write': result.get('can_write', False) and result['status'] == 'testing'
                               and result['head_sha'] == snapshot.get('head', {}).get('sha'),
                  'attachments': attachments, 'repo': canonical, 'pr_number': pr,
                  'max_file_bytes': request.app.state.config.evidence_max_file_bytes,
                  'part_bytes': request.app.state.config.evidence_part_bytes, 'csrf_token': csrf}
        if not {'head_sha', 'round_id', 'status', 'can_write', 'attachments', 'archive_links'} <= result.keys():
            raise HTTPException(503, detail='incomplete_evidence_context')
    for name in ('object_key', 'actor_id', 'actor_login', 'deleted_by', 'records', 'request_sha256'):
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
    try:
        result = await _dispatch(request, owner, repo, pr, action, **path)
    except HTTPException as exc:
        if exc.status_code == 401:
            target = f'/evidence/{owner}/{repo}/pulls/{pr}'
            return RedirectResponse('/evidence/auth/login?' + urlencode({'return_to': target}), status_code=303,
                                    headers={'Cache-Control': 'no-store'})
        raise
    url = result.get('url')
    if not isinstance(url, str) or not url.startswith('https://'):
        raise HTTPException(503, detail='evidence_download_unavailable')
    return RedirectResponse(url, status_code=303,
                            headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})


@router.get(_PREFIX + '/attachments/{attachment_id}/download')
async def download(request: Request, owner: str, repo: str, pr: int, attachment_id: str):
    return await _download(request, owner, repo, pr, 'download', attachment_id=attachment_id)


@router.get(_PREFIX + '/rounds/{round_id}/log/download')
async def log_download(request: Request, owner: str, repo: str, pr: int, round_id: str):
    return await _download(request, owner, repo, pr, 'log_download', round_id=round_id)


@router.get('/evidence/{owner}/{repo}/pulls/{pr}', response_class=HTMLResponse)
async def page(request: Request, owner: str, repo: str, pr: int):
    evidence_auth.require_enabled(request)
    try:
        result = await _dispatch(request, owner, repo, pr, 'context')
    except HTTPException as exc:
        if exc.status_code == 401:
            return RedirectResponse('/evidence/auth/login?' + urlencode(
                {'return_to': f'/evidence/{owner}/{repo}/pulls/{pr}'}), status_code=303)
        raise
    from . import evidence_ui
    return HTMLResponse(evidence_ui.render_upload_page(
        result['repo'], pr, result['head_sha'], result['round_id']),
        headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
                 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY'})


@router.get('/evidence/assets/app.js')
async def javascript(request: Request):
    evidence_auth.require_enabled(request)
    from . import evidence_ui
    return Response(evidence_ui.javascript_source(), media_type='application/javascript',
                    headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})


router.add_api_route('/evidence/auth/login', evidence_auth.login, methods=['GET'])
router.add_api_route('/evidence/auth/callback', evidence_auth.callback, methods=['GET'])
