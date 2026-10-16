"""Attachment operations persisted as immutable, trusted GitHub documents.

Store adapter contract (B): async dispatch(action, *, operation, payload) -> dict.
init/note must be idempotent by operation_id; resume must locate an uncertain
initialization without creating another upload. complete/note return verified,
object_key, size, sha256 after bounded streaming verification. sign_part and
(download/log_download) return ephemeral URLs only. Tokens/URLs/file contents
are never accepted into durable records. C renders references via the watcher.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import PurePosixPath

from .evidence_models import EvidenceError, round_id_for_state, utc_now, validate_context


class EvidenceService:
    def __init__(self, *, config, github, proxy, policy, controller, store):
        self.config, self.github, self.proxy = config, github, proxy
        self.policy, self.controller, self.store = policy, controller, store
        controller.evidence_service = self

    async def _authorize(self, repo, principal, state, *, check_machine=True):
        if (not self.config.auth_valid or repo not in self.config.github_repos
                or repo not in self.config.active_repos):
            raise EvidenceError('repository_not_authorized', 403)
        authorized = await self.github.list_installation_repositories()
        if repo not in authorized:
            raise EvidenceError('app_access_revoked', 403)
        uid, login, token = (principal.get(k) for k in ('id', 'login', 'user_access_token'))
        if type(uid) is not int or uid <= 0 or not isinstance(login, str) or not login or not isinstance(token, str) or not token:
            raise EvidenceError('identity_required', 401)
        # Principal is an internal authenticated value, but every operation also
        # checks current GitHub identity and current repository access.
        for path in ('/user', f'/repos/{repo}'):
            response = await self.github.http.get(
                self.config.github_api_url.rstrip('/') + path,
                headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'},
                timeout=self.config.total_timeout, follow_redirects=False,
            )
            if response.status_code != 200 or len(response.content) > self.config.max_response_bytes:
                raise EvidenceError('github_access_unavailable', 403 if response.status_code in (401, 403, 404) else 503)
            data = response.json()
            if path == '/user':
                if type(data.get('id')) is not int or data['id'] != uid or data.get('login') != login:
                    raise EvidenceError('identity_changed', 401)
            elif data.get('full_name', '').lower() != repo.lower() or data.get('permissions', {}).get('pull') is not True:
                raise EvidenceError('repository_access_denied', 403)
        if check_machine:
            try:
                await self.controller._check_record_test_permissions(repo, login, state)
            except Exception:
                raise EvidenceError('actor_not_authorized', 403) from None
        if not self.config.auth_valid or repo not in self.config.active_repos:
            raise EvidenceError('app_access_revoked', 403)

    async def _save(self, repo, pr_number, kind, operation, phase):
        key = operation['operation_id'] + ':' + phase
        record = await self.proxy.write_evidence_record(repo, pr_number, kind, key, operation)
        canonical = copy.deepcopy(record['data'])
        operation.clear()
        operation.update(canonical)
        return record

    async def _store(self, action, operation, payload):
        if self.store is None or not callable(getattr(self.store, 'dispatch', None)):
            raise EvidenceError('store_adapter_unavailable', 503)
        result = await self.store.dispatch(action, operation=copy.deepcopy(operation), payload=payload)
        if not isinstance(result, dict):
            raise EvidenceError('store_result_invalid', 502)
        return result

    async def dispatch(self, action: str, *, repo: str, pr_number: int, principal: dict, payload: dict) -> dict:
        if action not in {'context', 'status', 'init', 'resume', 'sign_part', 'complete', 'note', 'delete', 'download', 'log_download'}:
            raise EvidenceError('unknown_action', 400)
        if type(pr_number) is not int or pr_number <= 0 or not isinstance(payload, dict) or not isinstance(principal, dict):
            raise EvidenceError('invalid_request', 400)
        if (not self.config.auth_valid or repo not in self.config.github_repos
                or repo not in self.config.active_repos):
            raise EvidenceError('repository_not_authorized', 403)
        # A process-local serialization lock is scheduling, not persistence.
        # All operation identities and recovery facts live in GitHub comments.
        async with self.controller._evidence_lock:
            pr = await self.proxy.get_pr(repo, pr_number)
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if not isinstance(state, dict):
                raise EvidenceError('state_unavailable', 409)
            await self._authorize(repo, principal, state, check_machine=False)
            if pr.get('state') != 'open' or pr.get('merged'):
                raise EvidenceError('pr_closed', 409)
            records = await self.proxy.read_evidence_records(repo, pr_number)
            active_round = round_id_for_state(state)
            payload = dict(payload)
            if action in {'resume', 'status'}:
                upload_ref = payload.get('operation_id') or payload.get('upload_id')
                if not isinstance(upload_ref, str) or not 1 <= len(upload_ref) <= 128:
                    raise EvidenceError('invalid_upload_id', 400)
                source = next((r['data'] for r in reversed(records) if r['kind'] == 'operation'
                               and r['data'].get('upload_ref') == upload_ref), None)
                if not source:
                    raise EvidenceError('operation_not_found', 404)
                if source.get('actor_id') != principal.get('id'):
                    raise EvidenceError('upload_owner_mismatch', 403)
                if source.get('repo') != repo or source.get('pr_number') != pr_number:
                    raise EvidenceError('operation_not_found', 404)
                # Both endpoints bind to the durable operation. A historical
                # upload remains readable in its archived round; only POST
                # resume may perform recovery or persist a new phase.
                payload.setdefault('head_sha', source['head_sha'])
                payload.setdefault('round_id', source['round_id'])
            if action == 'resume':
                payload.setdefault('idempotency_key', hashlib.sha256(('resume:' + source['upload_ref']).encode()).hexdigest())
            if action in {'init', 'sign_part', 'complete', 'note', 'delete'}:
                if 'head_sha' not in payload or 'round_id' not in payload:
                    raise EvidenceError('explicit_context_required', 400)
            round_id = payload.get('round_id', active_round)
            if action == 'download' and 'round_id' not in payload:
                targets = [r['data'] for r in records if r['kind'] == 'attachment'
                           and r['data'].get('attachment_id') == payload.get('attachment_id')]
                if targets:
                    round_id = targets[-1]['round_id']
            selected = state
            if round_id != active_round:
                archives = [r['data'] for r in records if r['kind'] == 'round' and r['data'].get('round_id') == round_id
                            and r['data'].get('repo') == repo and r['data'].get('pr_number') == pr_number]
                if not archives:
                    raise EvidenceError('round_not_found', 404)
                selected = archives[-1]['snapshot']
            await self._authorize(repo, principal, selected)
            write = action in {'init', 'resume', 'sign_part', 'complete', 'note', 'delete'}
            expected_head = payload.get('head_sha', selected.get('head_sha'))
            validate_context(pr, selected, expected_head, round_id, write=write)
            if write and round_id != active_round:
                raise EvidenceError('round_changed', 409)
            latest = {}
            operations = {}
            for record in records:
                data = record['data']
                if record['kind'] == 'attachment':
                    latest[data['attachment_id']] = data
                if record['kind'] == 'operation':
                    operations[data['operation_id']] = data
            if action == 'context':
                return {'repo': repo, 'pr_number': pr_number, 'head_sha': selected['head_sha'], 'round_id': round_id,
                        'status': selected['status'], 'authenticated': True, 'enabled': True,
                        'can_write': selected['status'] == 'testing' and round_id == active_round
                                     and pr.get('head', {}).get('sha') == selected['head_sha'],
                        'attachments': [r for r in latest.values() if r['round_id'] == round_id],
                        'rounds': [{**r['data']['snapshot'], 'round_id': r['data']['round_id'],
                                    'attachments': [a for a in latest.values() if a['round_id'] == r['data']['round_id']]}
                                   for r in records if r['kind'] == 'round'],
                        'operations': [{k: v for k, v in r.items() if k != 'upload_id'}
                                       for r in operations.values() if r['round_id'] == round_id
                                       and r['actor_id'] == principal['id']], 'records': [r for r in latest.values() if r['round_id'] == round_id],
                        'archive_links': [{'round_id': r['data']['round_id'], 'url': r['url']} for r in records if r['kind'] == 'round']}
            if action == 'status':
                if source.get('head_sha') != expected_head or source.get('round_id') != round_id:
                    raise EvidenceError('round_changed', 409)
                if source.get('status') == 'invalid':
                    raise EvidenceError('round_changed', 409)
                saved = latest.get(source.get('attachment_id'))
                if saved and saved.get('status') == 'deleted':
                    raise EvidenceError('attachment_deleted', 410)
                operation = copy.deepcopy(saved if saved and saved.get('status') == 'ready' else source)
                result = {'uploaded_parts': []}
                # Status is a read-only reconciliation. In particular, an
                # uncertain complete is probed for the final object and then
                # the existing multipart upload; it is never submitted again.
                if operation.get('status') in {'completing', 'verifying', 'processing'}:
                    try:
                        result = await self.store.verify_object(
                            key=operation['object_key'], size=operation['size'],
                            metadata=self.store._identity_metadata(operation),
                            sha256=operation.get('sha256') or None,
                        )
                    except Exception:
                        if not operation.get('upload_id'):
                            result = {'uploaded_parts': [], 'completion_unknown': True}
                        else:
                            result = {'uploaded_parts': await self.store.list_parts(
                                key=operation['object_key'], upload_id=operation['upload_id']),
                                'completion_unknown': True}
                    else:
                        result = {**result, 'uploaded_parts': [], 'completion_unknown': False}
                elif operation.get('status') == 'uploading':
                    if not operation.get('object_key') or not operation.get('upload_id'):
                        result = {'uploaded_parts': [], 'initialization_pending': True}
                    else:
                        result = {'uploaded_parts': await self.store.list_parts(
                            key=operation['object_key'], upload_id=operation['upload_id'])}
                elif operation.get('status') == 'ready':
                    result = {'uploaded_parts': [], 'verified': True}
                public_fields = {'attachment_id', 'filename', 'size', 'sha256', 'note',
                                 'round_id', 'head_sha', 'status', 'created_at',
                                 'completed_at', 'actor', 'actor_login', 'part_bytes'}
                value = {k: copy.deepcopy(v) for k, v in operation.items() if k in public_fields}
                value['upload_id'] = source.get('upload_ref', source['operation_id'])
                value['uploaded_parts'] = result.get('uploaded_parts', [])
                value['parts'] = value['uploaded_parts']
                value['historical'] = (round_id != active_round
                                       or pr.get('head', {}).get('sha') != expected_head)
                if result.get('completion_unknown'):
                    value['completion_unknown'] = True
                if result.get('initialization_pending'):
                    value['initialization_pending'] = True
                if result.get('verified') is True:
                    value['status'] = 'ready'
                    value['sha256'] = result.get('sha256', value.get('sha256', ''))
                return value
            if action in {'download', 'log_download'}:
                if action == 'log_download':
                    obj = selected.get('cos', {})
                    if not self.controller._valid_cos_metadata(repo, pr_number, expected_head, obj):
                        raise EvidenceError('log_not_ready', 404)
                    operation = {'repo': repo, 'pr_number': pr_number, 'head_sha': expected_head, 'round_id': round_id, **obj}
                else:
                    operation = latest.get(payload.get('attachment_id'))
                    if not operation or operation['round_id'] != round_id:
                        raise EvidenceError('attachment_not_found', 404)
                    if operation['status'] == 'deleted':
                        raise EvidenceError('attachment_deleted', 410)
                    if operation['status'] != 'ready':
                        raise EvidenceError('attachment_not_ready', 409)
                # Store may sign ONLY the server-loaded object, never payload key.
                result = await self._store(action, operation, {})
                fresh = await self.proxy.get_pr(repo, pr_number)
                fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                await self._authorize(repo, principal, selected)
                validate_context(fresh, selected, expected_head, round_id, write=False)
                if action == 'download':
                    revisions = await self.proxy.read_evidence_records(repo, pr_number)
                    deleted = any(r['kind'] == 'attachment' and r['data'].get('attachment_id') == operation['attachment_id']
                                  and r['data'].get('status') == 'deleted' for r in revisions)
                    if deleted:
                        raise EvidenceError('attachment_deleted', 410)
                return result
            idem = payload.get('idempotency_key')
            if not isinstance(idem, str) or not 1 <= len(idem) <= 128:
                raise EvidenceError('idempotency_key_required', 400)
            opid = hashlib.sha256(json.dumps([repo, pr_number, principal['id'], action, idem]).encode()).hexdigest()
            request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            existing = operations.get(opid)
            if existing and (existing.get('request_sha256') != request_hash
                             or existing.get('head_sha') != expected_head or existing.get('round_id') != round_id):
                raise EvidenceError('idempotency_conflict', 409)

            def public(operation, result=None):
                value = copy.deepcopy(operation)
                # HTTP upload_id is a stable operation reference, never a key
                # supplied by a browser. COS UploadId remains a recovery fact.
                value['upload_id'] = operation.get('upload_ref', operation['operation_id'])
                value['part_bytes'] = operation.get('part_bytes', self.config.evidence_part_bytes)
                value['uploaded_parts'] = (result or {}).get('uploaded_parts', [])
                value['parts'] = value['uploaded_parts']
                return value

            async def current():
                fresh_pr = await self.proxy.get_pr(repo, pr_number)
                fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                validate_context(fresh_pr, fresh_state, expected_head, round_id, write=True)
                await self._authorize(repo, principal, fresh_state)

            if action == 'resume':
                if source.get('actor_id') != principal['id']:
                    raise EvidenceError('upload_owner_mismatch', 403)
                if source['status'] == 'invalid':
                    raise EvidenceError('round_changed', 409)
                saved = latest.get(source['attachment_id'])
                if saved and saved['status'] == 'deleted':
                    raise EvidenceError('attachment_deleted', 410)
                if saved and saved['status'] == 'ready':
                    return public(saved)
                if source['status'] == 'ready':
                    await current()
                    await self._save(repo, pr_number, 'attachment', source, 'ready')
                    return public(source)
                result = await self._store('resume', source, {})
                try:
                    await current()
                except ValueError:
                    fresh_pr = await self.proxy.get_pr(repo, pr_number)
                    if fresh_pr.get('state') == 'open' and not fresh_pr.get('merged'):
                        await self._save(repo, pr_number, 'operation',
                                         {**source, 'status': 'invalid', 'invalidated_at': utc_now()}, 'invalid')
                    raise EvidenceError('round_changed', 409) from None
                recovered = copy.deepcopy(source)
                for field in ('object_key', 'upload_id'):
                    if field in result:
                        value = result[field]
                        if (not isinstance(value, str) or not value or '://' in value
                                or value.startswith('/') or '\\' in value or '..' in value.split('/')
                                or '?' in value or '#' in value or any(ord(c) < 32 for c in value)
                                or (source.get(field) and source[field] != value)):
                            raise EvidenceError('store_recovery_conflict', 409)
                        recovered[field] = value
                if result.get('verified') is True:
                    sha = result.get('sha256')
                    if (not recovered.get('object_key') or type(result.get('size')) is not int
                            or result['size'] != source['size'] or not isinstance(sha, str)
                            or not re.fullmatch('[0-9a-f]{64}', sha)
                            or (source.get('sha256') and sha != source['sha256'])):
                        raise EvidenceError('object_not_verified', 409)
                    recovered.update(status='ready', sha256=sha, completed_at=utc_now())
                    await self._save(repo, pr_number, 'operation', recovered, 'ready')
                    await self._save(repo, pr_number, 'attachment', recovered, 'ready')
                else:
                    if not recovered.get('object_key') or not recovered.get('upload_id'):
                        raise EvidenceError('upload_not_recoverable', 502)
                    recovered['status'] = 'uploading'
                    await self._save(repo, pr_number, 'operation', recovered, 'uploading')
                return public(recovered, result)

            if existing and existing['status'] in {'ready', 'deleted', 'invalid'}:
                if existing['status'] == 'invalid':
                    raise EvidenceError('round_changed', 409)
                await current()
                await self._save(repo, pr_number, 'attachment', existing, existing['status'])
                return public(existing)
            if action in {'init', 'note'}:
                text = payload.get('text') or payload.get('note', '')
                if action == 'note':
                    if not isinstance(text, str) or not text:
                        raise EvidenceError('text_required', 400)
                    body = text.encode('utf-8')
                    if len(body) > self.config.evidence_max_note_bytes:
                        raise EvidenceError('text_too_large', 413)
                    filename, size, sha = 'note.txt', len(body), hashlib.sha256(body).hexdigest()
                else:
                    filename, size, sha = payload.get('filename'), payload.get('size'), payload.get('sha256', '')
                    if sha and (not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)):
                        raise EvidenceError('invalid_sha256', 400)
                if (not isinstance(filename, str) or not filename or len(filename) > 255
                        or PurePosixPath(filename).name != filename or '\\' in filename or filename in {'.', '..'}
                        or any(ord(c) < 32 for c in filename)
                        or type(size) is not int or not 1 <= size <= self.config.evidence_max_file_bytes):
                    raise EvidenceError('invalid_file_metadata', 400)
                replaces = payload.get('replaces_id') or ''
                if replaces and (replaces not in latest or latest[replaces]['round_id'] != round_id):
                    raise EvidenceError('replacement_not_found', 404)
                from .evidence_builder import _redact_secrets
                note = payload.get('note', '')
                if not isinstance(note, str):
                    raise EvidenceError('invalid_note', 400)
                operation = existing or {
                    'operation_id': opid, 'upload_ref': opid, 'attachment_id': opid, 'repo': repo, 'pr_number': pr_number,
                    'head_sha': expected_head, 'round_id': round_id, 'actor_id': principal['id'],
                    'actor_login': principal['login'], 'actor': principal['login'],
                    'created_at': utc_now(), 'filename': filename, 'size': size, 'sha256': sha,
                    'note': _redact_secrets(text if action == 'note' else note)[:256],
                    'replaces_id': replaces, 'status': 'initializing', 'request_sha256': request_hash,
                    'part_bytes': self.config.evidence_part_bytes,
                }
                if not existing:
                    await current()
                    await self._save(repo, pr_number, 'operation', operation, 'initializing')
                elif operation['status'] == 'uploading':
                    return public(operation)
                store_action = 'resume' if existing else action
                store_payload = {'text': text} if action == 'note' and not existing else {}
            else:
                ref = payload.get('operation_id') or payload.get('upload_id')
                source = next((r['data'] for r in reversed(records) if r['kind'] == 'operation'
                               and r['data'].get('upload_ref') == ref), operations.get(ref))
                if action == 'delete':
                    source = latest.get(payload.get('attachment_id'))
                if not source or source.get('round_id') != round_id or source.get('head_sha') != expected_head:
                    raise EvidenceError('operation_not_found', 404)
                current_attachment = latest.get(source['attachment_id'])
                if source.get('status') == 'invalid':
                    raise EvidenceError('round_changed', 409)
                if current_attachment and current_attachment.get('status') == 'deleted':
                    raise EvidenceError('attachment_deleted', 410)
                if source.get('actor_id') != principal['id'] and action != 'delete':
                    raise EvidenceError('upload_owner_mismatch', 403)
                if action != 'delete' and current_attachment and current_attachment.get('status') == 'ready':
                    return public(current_attachment)
                operation = existing or {**source, 'operation_id': opid,
                                         'source_operation_id': source['operation_id'], 'request_sha256': request_hash}
                if action == 'delete':
                    operation.update(status='deleted', deleted_at=utc_now(), deleted_by=principal['id'], deleted_by_login=principal['login'])
                    await current()
                    await self._save(repo, pr_number, 'operation', operation, 'deleted')
                    await self._save(repo, pr_number, 'attachment', operation, 'deleted')
                    return public(operation)
                if action == 'sign_part':
                    part = payload.get('part_number')
                    if type(part) is not int or not 1 <= part <= 10000 or source.get('status') != 'uploading':
                        raise EvidenceError('invalid_part_request', 409)
                    result = await self._store(action, source, {'part_number': part})
                    try:
                        await current()
                    except ValueError:
                        fresh_pr = await self.proxy.get_pr(repo, pr_number)
                        if fresh_pr.get('state') == 'open' and not fresh_pr.get('merged'):
                            await self._save(repo, pr_number, 'operation',
                                             {**operation, 'status': 'invalid', 'invalidated_at': utc_now()}, 'invalid')
                        raise EvidenceError('round_changed', 409) from None
                    return result
                if action == 'complete':
                    sha = payload.get('sha256')
                    if sha is not None and (not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)):
                        raise EvidenceError('invalid_sha256', 400)
                    if sha and source.get('sha256') and source['sha256'] != sha:
                        raise EvidenceError('sha256_conflict', 409)
                    if sha:
                        operation['sha256'] = sha
                    if not existing:
                        operation['status'] = 'completing'
                        await current()
                        await self._save(repo, pr_number, 'operation', operation, 'completing')
                store_action = 'resume' if existing or action == 'resume' else 'complete'
                store_payload = {}
            result = await self._store(store_action, operation, store_payload)
            try:
                await current()
            except ValueError:
                fresh_pr = await self.proxy.get_pr(repo, pr_number)
                invalid = {**operation, 'status': 'invalid', 'invalidated_at': utc_now()}
                if fresh_pr.get('state') == 'open' and not fresh_pr.get('merged'):
                    await self._save(repo, pr_number, 'operation', invalid, 'invalid')
                raise EvidenceError('round_changed', 409) from None
            # Only server-verified, whitelisted store facts enter GitHub.
            for key in ('object_key', 'upload_id'):
                if key in result:
                    value = result[key]
                    if (not isinstance(value, str) or not value or '\\' in value or '..' in value.split('/')
                            or value.startswith('/') or '://' in value or '?' in value or '#' in value
                            or any(ord(c) < 32 for c in value)):
                        raise EvidenceError('store_result_invalid', 502)
                    if operation.get(key) and operation[key] != value:
                        raise EvidenceError('store_object_conflict', 409)
                    operation[key] = value
            ready = result.get('verified') is True
            if action in {'complete', 'note'} or ready:
                actual_sha = result.get('sha256')
                if (not ready or not operation.get('object_key') or type(result.get('size')) is not int
                        or result['size'] != operation['size'] or not isinstance(actual_sha, str)
                        or not re.fullmatch('[0-9a-f]{64}', actual_sha)
                        or (operation.get('sha256') and actual_sha != operation['sha256'])):
                    raise EvidenceError('object_not_verified', 409)
                operation.update(status='ready', sha256=actual_sha, completed_at=utc_now())
                await self._save(repo, pr_number, 'operation', operation, 'ready')
                await self._save(repo, pr_number, 'attachment', operation, 'ready')
            else:
                if not operation.get('upload_id') or not operation.get('object_key'):
                    raise EvidenceError('upload_not_recoverable', 502)
                operation['status'] = 'uploading'
                await self._save(repo, pr_number, 'operation', operation, 'uploading')
            return public(operation, result)
