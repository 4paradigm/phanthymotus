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
import asyncio
import hashlib
import ipaddress
import json
import re
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit

from .evidence_auth import DOWNLOAD_PURPOSE, PURPOSE
from .evidence_models import EvidenceError, round_id_for_state, utc_now, validate_context


class EvidenceService:
    def __init__(self, *, config, github, proxy, policy, controller, store):
        self.config, self.github, self.proxy = config, github, proxy
        self.policy, self.controller, self.store = policy, controller, store
        self._used_auth_nonces = {}
        controller.evidence_service = self

    async def _authorize(self, repo, principal, state, *, check_machine=True, pr=None):
        if (not self.config.auth_valid or repo not in self.config.github_repos
                or repo not in self.config.active_repos):
            raise EvidenceError('repository_not_authorized', 403)
        authorized = await self.github.list_installation_repositories()
        if repo not in authorized:
            raise EvidenceError('app_access_revoked', 403)
        uid, login = principal.get('id'), principal.get('login')
        if type(uid) is not int or uid <= 0 or not isinstance(login, str) or not login:
            raise EvidenceError('identity_required', 401)
        if not self.config.auth_valid or repo not in self.config.active_repos:
            raise EvidenceError('app_access_revoked', 403)
        if not (principal.get('actor_verified') is True
                and principal.get('github_user_authenticated') is True):
            return False
        # Browser write identity is accepted only after the HTTP layer has
        # revalidated GitHub /user for this request. Authorization remains
        # tied to the live PR author, deployed machine owners, or repo ACL.
        author = (pr or {}).get('user', {})
        if (isinstance(author, dict) and
                type(author.get('id')) is int and author.get('id') == uid):
            return True
        deployed = {d.get('machine') for d in state.get('deployments', [])
                    if isinstance(d, dict) and d.get('phase') == 'deployed'}
        for alias in deployed:
            machine = self.policy.get_machine(alias) if isinstance(alias, str) else None
            if machine and any(isinstance(owner, str) and owner.lower() == login.lower()
                               for owner in (machine.owners or [])):
                return True
        try:
            permission = await self.proxy.collaborator_permission(repo, login)
        except Exception:
            permission = ''
        return self.policy.collaborator_can_approve(permission)

    @staticmethod
    def _validate_proof_context(principal, repo, pr_number, head_sha, round_id):
        """Require the HTTP layer's comment-proof identity to match this write."""
        proof = principal.get('evidence_auth')
        if (principal.get('actor_verified') is not True or not isinstance(proof, dict)
                or proof.get('repo') != repo or proof.get('pr_number') != pr_number
                or type(proof.get('comment_id')) is not int or proof['comment_id'] <= 0
                or proof.get('head_sha') != head_sha or proof.get('round_id') != round_id
                or not re.fullmatch(r'[0-9a-f]{64}', str(proof.get('proof', '')))
                or not re.fullmatch(r'[0-9a-f]{64}', str(proof.get('nonce', '')))):
            raise EvidenceError('evidence_authorization_invalid', 403)

    async def _consume_auth_nonce(self, principal):
        """Consume request replay protection in process memory, not GitHub comments.

        Business mutations remain idempotent through their durable operation IDs;
        short-lived per-request nonces must not create permanent records for every
        signed multipart part URL.
        """
        proof = principal['evidence_auth']
        nonce_key = hashlib.sha256(
            f"{principal['id']}:{proof['nonce']}".encode('utf-8')
        ).hexdigest()
        now = time.monotonic()
        for key, expires in list(self._used_auth_nonces.items()):
            if expires <= now:
                self._used_auth_nonces.pop(key, None)
        if nonce_key in self._used_auth_nonces:
            raise EvidenceError('evidence_request_replayed', 409)
        if len(self._used_auth_nonces) >= 100_000:
            raise EvidenceError('evidence_request_replay_cache_full', 503)
        self._used_auth_nonces[nonce_key] = now + 24 * 60 * 60

    async def _save(self, repo, pr_number, kind, operation, phase):
        operation['updated_at'] = utc_now()
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

    @staticmethod
    def _native_asset_id(source_url):
        """Accept only GitHub's native PR attachment URL shapes."""
        parsed = urlsplit(source_url)
        if (parsed.scheme != 'https' or parsed.hostname != 'github.com'
                or parsed.port not in (None, 443) or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise EvidenceError('attachment_url_not_allowed', 400)
        if not parsed.path.startswith('/user-attachments/'):
            raise EvidenceError('attachment_url_not_allowed', 400)
        parts = [unquote(part) for part in parsed.path.split('/') if part]
        if len(parts) < 3 or parts[0] != 'user-attachments':
            raise EvidenceError('attachment_url_not_allowed', 400)
        kind = parts[1]
        if kind == 'assets' and len(parts) == 3 and re.fullmatch(r'[A-Fa-f0-9-]{16,}', parts[2]):
            asset_id = parts[2]
        elif (kind == 'files' and len(parts) == 4
              and re.fullmatch(r'[1-9][0-9]*', parts[2])
              and re.fullmatch(r'[A-Za-z0-9._~-]+', parts[3])):
            asset_id = parts[2]
        else:
            raise EvidenceError('attachment_url_not_allowed', 400)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', asset_id):
            raise EvidenceError('attachment_url_not_allowed', 400)
        return asset_id

    @staticmethod
    def _attachment_urls(body):
        found = []
        for value in re.findall(r'https://github\.com/user-attachments/(?:assets|files)/[^\s<>\])]+', body):
            value = value.rstrip('.,;')
            try:
                EvidenceService._native_asset_id(value)
            except EvidenceError:
                continue
            if value not in found:
                found.append(value)
        return found

    @staticmethod
    def _safe_download_host(host):
        if host == 'github.com':
            return True
        if host == 'githubusercontent.com' or host.endswith('.githubusercontent.com'):
            return True
        return bool(re.fullmatch(
            r'github-production-user-asset-[0-9a-f]{6}\.s3\.amazonaws\.com', host))

    @classmethod
    def _download_native_asset(cls, source_url, target, limit):
        """Stream a public GitHub attachment without forwarding credentials."""
        import requests

        current = source_url
        session = requests.Session()
        session.trust_env = False
        total = 0
        digest = hashlib.sha256()
        filename = ''
        started = time.monotonic()
        try:
            for redirect_count in range(4):
                parsed = urlsplit(current)
                host = (parsed.hostname or '').lower()
                if (parsed.scheme != 'https' or not cls._safe_download_host(host)
                        or parsed.port not in (None, 443) or parsed.username or parsed.password):
                    raise EvidenceError('attachment_redirect_not_allowed', 400)
                try:
                    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
                except OSError:
                    raise EvidenceError('attachment_download_unavailable', 502) from None
                if (not addresses or any(not ipaddress.ip_address(item[4][0]).is_global
                                         for item in addresses)):
                    raise EvidenceError('attachment_redirect_not_allowed', 400)
                response = session.get(current, stream=True, allow_redirects=False,
                                       timeout=(10, 30), headers={'User-Agent': 'Deploy-Approval-Evidence/1.0',
                                                                  'Accept': 'application/octet-stream'})
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get('Location', '')
                    response.close()
                    if redirect_count >= 3 or not location:
                        raise EvidenceError('attachment_redirect_not_allowed', 400)
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    response.close()
                    raise EvidenceError('attachment_download_unavailable', 502)
                length = response.headers.get('Content-Length')
                if length:
                    try:
                        declared_size = int(length)
                    except ValueError:
                        response.close()
                        raise EvidenceError('attachment_download_invalid', 502) from None
                    if declared_size > limit:
                        response.close()
                        raise EvidenceError('attachment_too_large', 413)
                disposition = response.headers.get('Content-Disposition', '')
                try:
                    from email.message import Message
                    header = Message()
                    header['Content-Disposition'] = disposition
                    filename = header.get_filename() or ''
                except (TypeError, ValueError):
                    filename = ''
                if not filename:
                    filename = unquote(urlsplit(source_url).path.rsplit('/', 1)[-1])
                try:
                    from .evidence_store import safe_filename
                    filename = safe_filename(filename)
                except (ValueError, TypeError):
                    filename = 'attachment-' + hashlib.sha256(source_url.encode()).hexdigest()[:12]
                try:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > limit:
                            raise EvidenceError('attachment_too_large', 413)
                        if time.monotonic() - started > 300:
                            raise EvidenceError('attachment_download_timeout', 502)
                        target.write(chunk)
                        digest.update(chunk)
                finally:
                    response.close()
                if total < 1:
                    raise EvidenceError('attachment_empty', 400)
                target.flush()
                target.seek(0)
                return {'filename': filename, 'size': total, 'sha256': digest.hexdigest()}
            raise EvidenceError('attachment_redirect_not_allowed', 400)
        finally:
            session.close()

    async def _native_identity(self, repo, pr_number, comment_id, expected_id, expected_login,
                               expected_body_hash=''):
        comment = await self.proxy.get_comment(repo, comment_id)
        issue_url = comment.get('issue_url', '') if isinstance(comment, dict) else ''
        body = comment.get('body', '') if isinstance(comment, dict) else ''
        if (not isinstance(comment, dict) or comment.get('id') != comment_id
                or not isinstance(issue_url, str)
                or issue_url.rstrip('/').split('/')[-1] != str(pr_number)
                or not isinstance(body, str)
                or (expected_body_hash and hashlib.sha256(body.encode()).hexdigest() != expected_body_hash)):
            raise EvidenceError('comment_unavailable', 409)
        actor_id, actor_login = await self.proxy.comment_identity(repo, comment_id)
        if (not actor_id or not actor_login or str(actor_id) != str(expected_id)
                or actor_login.casefold() != expected_login.casefold()):
            raise EvidenceError('comment_identity_changed', 403)
        return comment, int(actor_id), actor_login

    async def _native_authorization(self, repo, pr_number, actor_id, actor_login):
        if (not self.config.auth_valid or repo not in self.config.github_repos
                or repo not in self.config.active_repos):
            raise EvidenceError('repository_not_authorized', 403)
        if repo not in await self.github.list_installation_repositories():
            raise EvidenceError('app_access_revoked', 403)
        pr = await self.proxy.get_pr(repo, pr_number)
        state = await self.proxy.read_hidden_state(repo, pr_number)
        if (not isinstance(pr, dict) or pr.get('state') != 'open' or pr.get('merged')
                or pr.get('draft') or not isinstance(state, dict)):
            raise EvidenceError('pr_closed', 409)
        author = pr.get('user', {})
        if isinstance(author, dict) and str(author.get('id', '')) == str(actor_id):
            return pr, state, 'pr_author', ''
        deployed = {row.get('machine') for row in state.get('deployments', [])
                    if isinstance(row, dict) and row.get('phase') == 'deployed'}
        for alias in deployed:
            machine = self.policy.get_machine(alias) if isinstance(alias, str) else None
            if machine and any(isinstance(owner, str) and owner.casefold() == actor_login.casefold()
                               for owner in (machine.owners or [])):
                return pr, state, 'machine_owner', alias
        permission = await self.proxy.collaborator_permission(repo, actor_login)
        if self.policy.collaborator_can_approve(permission):
            return pr, state, 'collaborator', permission
        raise EvidenceError('evidence_write_permission_required', 403)

    @staticmethod
    def _comment_created_at(comment):
        value = comment.get('created_at')
        if not isinstance(value, str):
            raise EvidenceError('comment_time_invalid', 409)
        try:
            stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                raise ValueError
            return stamp.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        except ValueError:
            raise EvidenceError('comment_time_invalid', 409) from None

    async def handle_native_comment(self, **kwargs):
        """Serialize native comment writes with the existing Evidence writer."""
        async with self.controller._evidence_lock:
            return await self._handle_native_comment_locked(**kwargs)

    async def _handle_native_comment_locked(self, *, repo, pr_number, comment_id, actor_id,
                                            actor_login, action, arguments):
        """Transfer native PR attachments and append immutable evidence documents."""
        if action not in {'submit_evidence', 'remove_evidence'}:
            raise EvidenceError('unknown_action', 400)
        if repo not in {'4paradigm/phanthymotus', '4paradigm/phanthymotus-driver'}:
            raise EvidenceError('repository_not_authorized', 403)
        comment, actor_id, actor_login = await self._native_identity(
            repo, pr_number, comment_id, actor_id, actor_login)
        pr, state, role, role_value = await self._native_authorization(
            repo, pr_number, actor_id, actor_login)
        body = comment.get('body', '')
        if not isinstance(body, str):
            raise EvidenceError('comment_unavailable', 409)
        from .commands import parse_command
        parsed = parse_command(body)
        if parsed.kind != action or parsed.evidence_args != arguments:
            raise EvidenceError('comment_command_changed', 409)
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        head_sha = arguments.get('head', '')
        round_id = arguments.get('round', '')
        if (not re.fullmatch(r'[0-9a-f]{40}', head_sha)
                or state.get('head_sha') != head_sha or pr.get('head', {}).get('sha') != head_sha
                or round_id_for_state(state) != round_id):
            raise EvidenceError('round_changed', 409)
        if state.get('status') != 'testing':
            raise EvidenceError('context_not_writable', 409)
        created_at = self._comment_created_at(comment)
        async def recheck_current():
            _, fresh_id, fresh_login = await self._native_identity(
                repo, pr_number, comment_id, actor_id, actor_login, body_hash)
            fresh_pr, fresh_state, _, _ = await self._native_authorization(
                repo, pr_number, fresh_id, fresh_login)
            if (fresh_pr.get('head', {}).get('sha') != head_sha
                    or fresh_state.get('head_sha') != head_sha
                    or round_id_for_state(fresh_state) != round_id
                    or fresh_state.get('status') != 'testing'):
                raise EvidenceError('round_changed', 409)
            return fresh_pr, fresh_state

        records = await self.proxy.read_evidence_records(repo, pr_number)
        latest = {}
        prepared_operations = {}
        for record in records:
            if record['kind'] == 'attachment':
                latest[record['data'].get('attachment_id')] = record['data']
            elif record['kind'] == 'operation':
                data = record['data']
                if data.get('native_attachment') is True:
                    prepared_operations[data.get('attachment_id')] = data
        if action == 'remove_evidence':
            target_id = arguments.get('id', '')
            target = latest.get(target_id)
            if (not target or target.get('round_id') != round_id or target.get('head_sha') != head_sha
                    or target.get('status') == 'deleted'):
                raise EvidenceError('attachment_not_found', 404)
            uploader = str(target.get('actor_id')) == str(actor_id)
            await recheck_current()
            if not uploader:
                permission = await self.proxy.collaborator_permission(repo, actor_login)
                if permission not in {'maintain', 'admin'}:
                    raise EvidenceError('evidence_delete_permission_required', 403)
            tombstone = {**target, 'status': 'deleted', 'deleted_at': created_at,
                         'deleted_by': actor_login, 'deleted_by_id': actor_id,
                         'deletion_comment_id': comment_id, 'updated_at': created_at}
            await self.proxy.write_evidence_record(
                repo, pr_number, 'attachment', 'native-delete:' + str(comment_id) + ':' + target_id,
                tombstone)
            return tombstone

        urls = self._attachment_urls(body)
        if len(urls) > 10:
            raise EvidenceError('too_many_attachments', 413)
        replaces = arguments.get('replaces', '')
        if replaces and (len(urls) != 1 or replaces not in latest
                         or latest[replaces].get('round_id') != round_id):
            raise EvidenceError('replacement_not_found', 404)
        if arguments.get('machine'):
            alias = arguments['machine']
            deployed = {row.get('machine') for row in state.get('deployments', [])
                        if isinstance(row, dict) and row.get('phase') == 'deployed'}
            if alias not in deployed or self.policy.get_machine(alias) is None:
                raise EvidenceError('machine_not_deployed', 409)
            machine = self.policy.get_machine(alias)
            if (role == 'machine_owner' and role_value != alias
                    and not any(owner.casefold() == actor_login.casefold()
                                for owner in machine.owners or [])):
                raise EvidenceError('machine_owner_mismatch', 403)
        note = arguments.get('note', '').strip()
        if not note:
            remaining = body.split('\n', 1)[1] if '\n' in body else ''
            remaining = re.sub(
                r'!?\[[^\]]*\]\(\s*https://github\.com/user-attachments/(?:assets|files)/[^)]*\)',
                '', remaining)
            remaining = re.sub(r'https://github\.com/user-attachments/(?:assets|files)/[^\s<>\])]+', '', remaining)
            note = remaining.strip()
        if len(note.encode('utf-8')) > self.config.evidence_max_note_bytes:
            raise EvidenceError('note_too_large', 413)
        if not urls:
            if not note:
                raise EvidenceError('evidence_content_required', 400)
            attachment_id = 'note-' + hashlib.sha256(
                f'{repo}:{pr_number}:{comment_id}'.encode()).hexdigest()[:48]
            data = {'repo': repo, 'pr_number': pr_number, 'source_comment_id': comment_id,
                    'source_url': '', 'attachment_id': attachment_id, 'filename': 'note.txt',
                    'size': len(note.encode('utf-8')), 'sha256': hashlib.sha256(note.encode('utf-8')).hexdigest(),
                    'object_key': '', 'head_sha': head_sha, 'round_id': round_id,
                    'actor_id': actor_id, 'actor_login': actor_login, 'actor_verified': True,
                    'machine': arguments.get('machine', ''), 'note': note,
                    'created_at': created_at, 'updated_at': created_at, 'status': 'text'}
            await recheck_current()
            await self.proxy.write_evidence_record(
                repo, pr_number, 'attachment', 'native:' + attachment_id, data)
            return data

        limit = min(int(getattr(self.config, 'evidence_max_file_bytes', 100 * 1024 * 1024)),
                    100 * 1024 * 1024)
        completed = []
        for source_url in urls:
            asset_id = self._native_asset_id(source_url)
            attachment_id = 'native-' + hashlib.sha256(
                f'{repo}:{pr_number}:{comment_id}:{asset_id}'.encode()).hexdigest()[:48]
            existing = latest.get(attachment_id)
            if existing:
                if (existing.get('source_comment_id') != comment_id
                        or existing.get('source_url') != source_url
                        or existing.get('head_sha') != head_sha or existing.get('round_id') != round_id):
                    raise EvidenceError('attachment_identity_conflict', 409)
                if existing.get('status') == 'ready':
                    completed.append(existing)
                    continue
            prepared = prepared_operations.get(attachment_id)
            metadata = {'repo': repo, 'pr_number': pr_number, 'head_sha': head_sha,
                        'round_id': round_id, 'attachment_id': attachment_id}
            if prepared and (prepared.get('source_url') != source_url
                             or prepared.get('source_comment_id') != comment_id
                             or prepared.get('actor_id') != actor_id
                             or prepared.get('head_sha') != head_sha
                             or prepared.get('round_id') != round_id
                             or prepared.get('machine', '') != arguments.get('machine', '')
                             or prepared.get('note', '') != note
                             or prepared.get('replaces_id', '') != replaces):
                raise EvidenceError('attachment_identity_conflict', 409)
            if prepared:
                record = {k: v for k, v in prepared.items() if k not in {'native_attachment', 'operation_id'}}
                key, size, sha = record['object_key'], record['size'], record['sha256']
                try:
                    result = await self.store.verify_object(
                        key=key, size=size, metadata=metadata, sha256=sha)
                except Exception:
                    result = None
            else:
                record = None
                result = None
            if result is None:
                if not callable(getattr(self.store, 'put_native_file', None)):
                    raise EvidenceError('store_adapter_unavailable', 503)
                with tempfile.TemporaryFile() as temp:
                    facts = await asyncio.to_thread(
                        self._download_native_asset, source_url, temp, limit)
                    from .evidence_store import build_attachment_key
                    key = build_attachment_key(repo, pr_number, head_sha, round_id,
                                               attachment_id, created_at, facts['filename'])
                    record = {'repo': repo, 'pr_number': pr_number, 'source_comment_id': comment_id,
                              'source_url': source_url, 'attachment_id': attachment_id,
                              'filename': facts['filename'], 'size': facts['size'], 'sha256': facts['sha256'],
                              'object_key': key, 'head_sha': head_sha, 'round_id': round_id,
                              'actor_id': actor_id, 'actor_login': actor_login, 'actor_verified': True,
                              'machine': arguments.get('machine', ''), 'note': note,
                              'created_at': created_at, 'updated_at': created_at, 'status': 'prepared'}
                    if replaces:
                        record['replaces_id'] = replaces
                    if prepared and any(record.get(k) != prepared.get(k)
                                        for k in ('filename', 'size', 'sha256', 'object_key')):
                        raise EvidenceError('attachment_retry_conflict', 409)
                    if not prepared:
                        await recheck_current()
                        prepared_data = {**record, 'native_attachment': True,
                                         'operation_id': attachment_id}
                        await self.proxy.write_evidence_record(
                            repo, pr_number, 'operation', 'native:' + attachment_id + ':prepared',
                            prepared_data)
                    await recheck_current()
                    result = await self.store.put_native_file(
                        key=key, metadata=metadata, size=facts['size'],
                        sha256=facts['sha256'], body=temp)
            if (not isinstance(result, dict) or result.get('verified') is not True
                    or result.get('size') != record['size'] or result.get('sha256') != record['sha256']
                    or result.get('object_key') != record['object_key']):
                raise EvidenceError('object_not_verified', 409)
            await recheck_current()
            record.update(status='ready', updated_at=created_at)
            await self.proxy.write_evidence_record(
                repo, pr_number, 'attachment', 'native:' + attachment_id, record)
            completed.append(record)
        return {'attachments': completed}

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
            if pr.get('state') != 'open' or pr.get('merged') or pr.get('draft'):
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
            authorized_writer = await self._authorize(repo, principal, selected, pr=pr)
            write = action in {'init', 'resume', 'sign_part', 'complete', 'note', 'delete'}
            if write and not authorized_writer:
                raise EvidenceError('evidence_write_permission_required', 403)
            expected_head = payload.get('head_sha', selected.get('head_sha'))
            validate_context(pr, selected, expected_head, round_id, write=write)
            if write and round_id != active_round:
                raise EvidenceError('round_changed', 409)
            if write:
                self._validate_proof_context(
                    principal, repo, pr_number, expected_head, round_id,
                )
            capability = principal.get('capability', {})
            resource = (('attachment:' + str(payload.get('attachment_id', '')))
                        if action == 'download' else ('log:' + round_id)
                        if action == 'log_download' else None)
            if not principal.get('github_user_authenticated'):
                purpose = capability.get('purpose')
                context_capability = (
                    action == 'context'
                    and (purpose == PURPOSE and 'resource' not in capability
                         or purpose == DOWNLOAD_PURPOSE
                         and capability.get('resource') == 'page')
                )
                scoped_read_capability = (
                    action in {'download', 'log_download'}
                    and ((purpose == PURPOSE and 'resource' not in capability)
                         or (purpose == DOWNLOAD_PURPOSE
                             and capability.get('resource') == resource))
                )
                legacy_read_capability = (
                    action == 'status' and purpose == PURPOSE
                    and 'resource' not in capability
                )
                if (capability.get('repo') != repo or capability.get('pr') != pr_number
                        or capability.get('head') != expected_head
                        or capability.get('round') != round_id
                        or not (context_capability or scoped_read_capability
                                or legacy_read_capability)):
                    raise EvidenceError('capability_context_mismatch', 403)
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
                        'github_user_authenticated': bool(
                            principal.get('actor_verified') is True
                            and principal.get('github_user_authenticated') is True),
                        'can_write': bool(authorized_writer and selected.get('status') == 'testing'
                                          and selected.get('head_sha') == pr.get('head', {}).get('sha')),
                        'attachments': [r for r in latest.values() if r['round_id'] == round_id],
                        'rounds': [{**r['data']['snapshot'], 'round_id': r['data']['round_id'],
                                    'attachments': [a for a in latest.values() if a['round_id'] == r['data']['round_id']]}
                                   for r in records if r['kind'] == 'round'],
                        'operations': [{k: v for k, v in r.items() if k != 'upload_id'}
                                       for r in operations.values() if not r.get('auth_nonce_marker')
                                       and not r.get('native_attachment')
                                       and r.get('round_id') == round_id
                                       and r.get('actor_id') == principal['id']], 'records': [r for r in latest.values() if r['round_id'] == round_id],
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
                    value['verified'] = True
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
                await self._authorize(repo, principal, selected, pr=fresh)
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

            nonce_consumed = False

            async def current():
                nonlocal nonce_consumed
                fresh_pr = await self.proxy.get_pr(repo, pr_number)
                fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                validate_context(fresh_pr, fresh_state, expected_head, round_id, write=True)
                if not await self._authorize(repo, principal, fresh_state, pr=fresh_pr):
                    raise EvidenceError('evidence_write_permission_required', 403)
                if not nonce_consumed:
                    await self._consume_auth_nonce(principal)
                    nonce_consumed = True

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
                await current()
                result = await self._store('resume', source, {})
                try:
                    await current()
                except ValueError as exc:
                    if isinstance(exc, EvidenceError) and exc.code not in {'round_changed', 'context_not_writable'}:
                        raise
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
                submitter = principal['login']
                operation = existing or {
                    'operation_id': opid, 'upload_ref': opid, 'attachment_id': opid, 'repo': repo, 'pr_number': pr_number,
                    'head_sha': expected_head, 'round_id': round_id, 'actor_id': principal['id'],
                    'actor_login': submitter, 'actor': submitter, 'submitter': submitter,
                    'actor_verified': True, 'created_by': submitter,
                    'created_at': utc_now(), 'updated_at': utc_now(), 'filename': filename, 'size': size, 'sha256': sha,
                    'note': _redact_secrets(text if action == 'note' else note)[:256],
                    'replaces_id': replaces, 'status': 'initializing', 'request_sha256': request_hash,
                    'part_bytes': self.config.evidence_part_bytes,
                }
                if not existing:
                    await current()
                    await self._save(repo, pr_number, 'operation', operation, 'initializing')
                elif operation['status'] == 'uploading':
                    return public(operation)
                store_action = 'note' if action == 'note' else ('resume' if existing else action)
                store_payload = {'text': text} if action == 'note' else {}
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
                    operation.update(status='deleted', deleted_at=utc_now(), deleted_by=principal['login'],
                                     deleted_by_login=principal['login'], deleted_by_id=principal['id'],
                                     actor_verified=True, updated_at=utc_now())
                    await current()
                    await self._save(repo, pr_number, 'operation', operation, 'deleted')
                    await self._save(repo, pr_number, 'attachment', operation, 'deleted')
                    return public(operation)
                if action == 'sign_part':
                    part = payload.get('part_number')
                    if type(part) is not int or not 1 <= part <= 10000 or source.get('status') != 'uploading':
                        raise EvidenceError('invalid_part_request', 409)
                    await current()
                    result = await self._store(action, source, {'part_number': part})
                    try:
                        await current()
                    except ValueError as exc:
                        if isinstance(exc, EvidenceError) and exc.code not in {'round_changed', 'context_not_writable'}:
                            raise
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
            await current()
            result = await self._store(store_action, operation, store_payload)
            try:
                await current()
            except ValueError as exc:
                if isinstance(exc, EvidenceError) and exc.code not in {'round_changed', 'context_not_writable'}:
                    raise
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
