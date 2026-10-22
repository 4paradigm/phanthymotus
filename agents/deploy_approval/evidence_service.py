"""Attachment operations persisted as immutable, trusted Git Notes records.

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

from .evidence_auth import (
    DOWNLOAD_PURPOSE, PURPOSE, UPLOAD_MAX_BYTES, UPLOAD_MAX_FILES, UPLOAD_PURPOSE,
)
from .evidence_models import EvidenceError, round_id_for_state, utc_now, validate_context

UPLOAD_MAX_NOTES = 20


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
        if isinstance(principal.get('upload_invite'), dict):
            return False
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
        content_type = ''
        prefix = bytearray()
        from .evidence_store import safe_filename

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
                content_type = response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
                try:
                    from email.message import Message
                    header = Message()
                    header['Content-Disposition'] = disposition
                    filename = header.get_filename() or ''
                except (TypeError, ValueError):
                    filename = ''
                if not filename:
                    filename = unquote(urlsplit(current).path.rsplit('/', 1)[-1])
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
                        if len(prefix) < 4096:
                            prefix.extend(chunk[:4096 - len(prefix)])
                        target.write(chunk)
                        digest.update(chunk)
                finally:
                    response.close()
                if total < 1:
                    raise EvidenceError('attachment_empty', 400)
                sample = bytes(prefix).lstrip(b'\xef\xbb\xbf \t\r\n').lower()
                error_markers = (
                    b'accessdenied', b'access denied', b'signaturedoesnotmatch',
                    b'request has expired', b'nosuchkey', b'no such key',
                    b'403 forbidden', b'404 not found', b'502 bad gateway',
                    b'503 service unavailable', b'request blocked',
                    b'internal server error', b'<title>error', b'<error',
                )
                if (sample.startswith((b'<!doctype html', b'<html', b'<?xml'))
                        and any(marker in sample for marker in error_markers)):
                    raise EvidenceError('attachment_source_error_content', 400)
                if (sample.startswith((b'<!doctype html', b'<html'))
                        and content_type not in {'text/html', 'application/xhtml+xml'}):
                    raise EvidenceError('attachment_source_error_content', 400)
                if any(sample.startswith(marker) for marker in error_markers):
                    raise EvidenceError('attachment_source_error_content', 400)

                suffix = PurePosixPath(filename).suffix.lower()
                if (sample.startswith((b'<!doctype html', b'<html'))
                        and content_type in {'text/html', 'application/xhtml+xml'}
                        and suffix not in {'.html', '.htm'}):
                    raise EvidenceError('attachment_source_error_content', 400)
                webm_signature = bytes(prefix).startswith(b'\x1a\x45\xdf\xa3')
                ftyp_signature = len(prefix) >= 8 and prefix[4:8] == b'ftyp'
                mp4_declared = (content_type == 'video/mp4'
                                or (suffix == '.mp4' and content_type != 'video/webm'))
                is_mp4 = mp4_declared or ftyp_signature
                if mp4_declared and not ftyp_signature:
                    raise EvidenceError('attachment_video_invalid', 400)
                if is_mp4:
                    target.flush()
                    offset = 0
                    box_count = 0
                    has_ftyp = False
                    has_media = False
                    while offset < total and box_count < 10000:
                        target.seek(offset)
                        box_header = target.read(8)
                        if len(box_header) != 8:
                            raise EvidenceError('attachment_video_invalid', 400)
                        box_size = int.from_bytes(box_header[:4], 'big')
                        box_type = box_header[4:8]
                        header_size = 8
                        if box_size == 1:
                            extended_size = target.read(8)
                            if len(extended_size) != 8:
                                raise EvidenceError('attachment_video_invalid', 400)
                            box_size = int.from_bytes(extended_size, 'big')
                            header_size = 16
                        elif box_size == 0:
                            box_size = total - offset
                        if (box_size < header_size or box_size > total - offset
                                or (box_count == 0 and box_type != b'ftyp')):
                            raise EvidenceError('attachment_video_invalid', 400)
                        if box_type == b'ftyp':
                            target.seek(offset + header_size)
                            major_brand = target.read(4)
                            if (box_size < header_size + 8
                                    or (box_size - header_size - 8) % 4
                                    or len(major_brand) != 4
                                    or any(char < 0x20 or char > 0x7e for char in major_brand)):
                                raise EvidenceError('attachment_video_invalid', 400)
                            has_ftyp = True
                        if box_type == b'mdat':
                            if box_size <= header_size:
                                raise EvidenceError('attachment_video_invalid', 400)
                            has_media = True
                        offset += box_size
                        box_count += 1
                    if (offset != total or (box_count >= 10000 and offset < total)
                            or not has_ftyp or not has_media):
                        raise EvidenceError('attachment_video_invalid', 400)

                extension_by_type = {
                    'video/mp4': '.mp4', 'video/webm': '.webm',
                    'video/quicktime': '.mov', 'image/jpeg': '.jpg',
                    'image/png': '.png', 'image/gif': '.gif',
                    'image/webp': '.webp', 'text/plain': '.txt',
                    'application/pdf': '.pdf', 'audio/mpeg': '.mp3',
                    'audio/wav': '.wav', 'audio/ogg': '.ogg',
                }
                inferred_extension = (
                    '.mp4' if is_mp4 else '.webm' if webm_signature
                    else extension_by_type.get(content_type, '')
                )
                if inferred_extension and not suffix:
                    filename += inferred_extension
                elif (inferred_extension and suffix != inferred_extension
                      and (is_mp4 or webm_signature or content_type.startswith('video/'))):
                    filename = filename[:-len(suffix)] + inferred_extension
                filename = safe_filename(filename)
                target.flush()
                target.seek(0)
                return {'filename': filename, 'size': total, 'sha256': digest.hexdigest(),
                        'content_type': content_type}
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
                                            actor_login, action, arguments, automatic=False):
        """Transfer native PR attachments and append immutable evidence documents."""
        if action not in {'submit_evidence', 'remove_evidence'}:
            raise EvidenceError('unknown_action', 400)
        if automatic and action != 'submit_evidence':
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
        if automatic:
            from .commands import command_starts_line_any
            if command_starts_line_any(body):
                raise EvidenceError('comment_command_changed', 409)
            arguments = {}
        else:
            from .commands import parse_command
            parsed = parse_command(body)
            if parsed.kind != action or parsed.evidence_args != arguments:
                raise EvidenceError('comment_command_changed', 409)
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        if automatic:
            head_sha = state.get('head_sha', '')
            round_id = round_id_for_state(state)
            arguments = {
                'head': head_sha, 'round': round_id,
                'machine': role_value if role == 'machine_owner' else '',
                'note': '', 'replaces': '',
            }
        else:
            head_sha = arguments.get('head', '')
            round_id = arguments.get('round', '')
        if (not re.fullmatch(r'[0-9a-f]{40}', head_sha)
                or state.get('head_sha') != head_sha or pr.get('head', {}).get('sha') != head_sha
                or round_id_for_state(state) != round_id):
            raise EvidenceError('round_changed', 409)
        if state.get('status') != 'testing':
            raise EvidenceError('context_not_writable', 409)
        created_at = self._comment_created_at(comment)

        def require_current_round_comment(current_state):
            if not automatic:
                return
            try:
                started_at = current_state.get('round_started_at')
                started = datetime.fromisoformat(str(started_at).replace('Z', '+00:00'))
                created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                if started.tzinfo is None or created.tzinfo is None or created <= started:
                    raise ValueError
            except (TypeError, ValueError):
                raise EvidenceError('comment_before_round', 409) from None

        require_current_round_comment(state)

        async def recheck_current():
            fresh_comment, fresh_id, fresh_login = await self._native_identity(
                repo, pr_number, comment_id, actor_id, actor_login, body_hash)
            if automatic and fresh_comment.get('created_at') != comment.get('created_at'):
                raise EvidenceError('comment_time_invalid', 409)
            fresh_pr, fresh_state, _, _ = await self._native_authorization(
                repo, pr_number, fresh_id, fresh_login)
            if (fresh_pr.get('head', {}).get('sha') != head_sha
                    or fresh_state.get('head_sha') != head_sha
                    or round_id_for_state(fresh_state) != round_id
                    or fresh_state.get('status') != 'testing'):
                raise EvidenceError('round_changed', 409)
            require_current_round_comment(fresh_state)
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
        if automatic:
            note = re.sub(
                r'!?\[[^\]]*\]\(\s*https://github\.com/user-attachments/(?:assets|files)/[^)]*\)',
                '', body)
            note = re.sub(
                r'https://github\.com/user-attachments/(?:assets|files)/[^\s<>\])]+',
                '', note,
            ).strip()
        else:
            note = arguments.get('note', '').strip()
        if not note and not automatic:
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
            existing = latest.get(attachment_id)
            if existing:
                if (existing.get('source_comment_id') != comment_id
                        or existing.get('head_sha') != head_sha or existing.get('round_id') != round_id
                        or existing.get('actor_id') != actor_id or existing.get('note') != note
                        or existing.get('status') != 'text'):
                    raise EvidenceError('attachment_identity_conflict', 409)
                await recheck_current()
                return existing
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
                if existing.get('status') in {'deleted', 'invalid'}:
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
        if action not in {'context', 'status', 'init', 'resume', 'sign_part', 'put_part', 'complete', 'note', 'delete', 'download', 'preview', 'log_download'}:
            raise EvidenceError('unknown_action', 400)
        if type(pr_number) is not int or pr_number <= 0 or not isinstance(payload, dict) or not isinstance(principal, dict):
            raise EvidenceError('invalid_request', 400)
        if (not self.config.auth_valid or repo not in self.config.github_repos
                or repo not in self.config.active_repos):
            raise EvidenceError('repository_not_authorized', 403)
        # A process-local serialization lock is scheduling, not persistence.
        # Operation identities and recovery facts live in Git Notes records;
        # the lifecycle comment remains the sole business-state projection.
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
                if (source.get('actor_id') != principal.get('id')
                        or isinstance(principal.get('upload_invite'), dict)
                        and source.get('upload_method') != 'signed_invite'):
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
            if action in {'init', 'sign_part', 'put_part', 'complete', 'note', 'delete'}:
                if 'head_sha' not in payload or 'round_id' not in payload:
                    raise EvidenceError('explicit_context_required', 400)
            round_id = payload.get('round_id', active_round)
            if action in {'download', 'preview'} and 'round_id' not in payload:
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
            expected_head = payload.get('head_sha', selected.get('head_sha'))
            authorized_writer = await self._authorize(repo, principal, selected, pr=pr)
            write = action in {'init', 'resume', 'sign_part', 'put_part', 'complete', 'note', 'delete'}
            invite = principal.get('upload_invite')

            def invite_context_matches(target_pr, target_state, target_round):
                return bool(
                    isinstance(invite, dict)
                    and invite.get('purpose') == UPLOAD_PURPOSE
                    and principal.get('auth_method') == 'signed_invite'
                    and principal.get('upload_method') == 'signed_invite'
                    and principal.get('actor_verified') is False
                    and principal.get('github_user_authenticated') is False
                    and type(principal.get('id')) is int
                    and (principal['id'] > 0 or action == 'context' and principal['id'] == 0)
                    and invite.get('repo') == repo and invite.get('pr') == pr_number
                    and invite.get('head') == expected_head
                    and invite.get('round') == target_round
                    and invite.get('status') == 'testing'
                    and type(invite.get('expires_at')) is int
                    and invite['expires_at'] > int(time.time())
                    and target_pr.get('state') == 'open' and not target_pr.get('merged')
                    and not target_pr.get('draft')
                    and target_pr.get('head', {}).get('sha') == expected_head
                    and target_state.get('head_sha') == expected_head
                    and target_state.get('status') == 'testing'
                    and round_id_for_state(target_state) == target_round
                )

            invite_allowed = (action in {'context', 'status'} or write) and invite_context_matches(
                pr, selected, round_id)
            if write and not (authorized_writer or invite_allowed):
                raise EvidenceError('evidence_write_permission_required', 403)
            validate_context(pr, selected, expected_head, round_id, write=write)
            if write and round_id != active_round:
                raise EvidenceError('round_changed', 409)
            if write and not invite_allowed:
                self._validate_proof_context(
                    principal, repo, pr_number, expected_head, round_id,
                )
            capability = principal.get('capability', {})
            resource = (('attachment:' + str(payload.get('attachment_id', '')))
                        if action in {'download', 'preview'} else ('log:' + round_id)
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
                    action in {'download', 'preview', 'log_download'}
                    and ((purpose == PURPOSE and 'resource' not in capability)
                         or (purpose == DOWNLOAD_PURPOSE
                             and capability.get('resource') == resource))
                )
                legacy_read_capability = (
                    action == 'status' and purpose == PURPOSE
                    and 'resource' not in capability
                )
                if (not invite_allowed and (capability.get('repo') != repo or capability.get('pr') != pr_number
                        or capability.get('head') != expected_head
                        or capability.get('round') != round_id
                        or not (context_capability or scoped_read_capability
                                or legacy_read_capability))):
                    raise EvidenceError('capability_context_mismatch', 403)
            latest = {}
            operations = {}
            for record in records:
                data = record['data']
                if record['kind'] == 'attachment':
                    latest[data['attachment_id']] = data
                if record['kind'] == 'operation':
                    operation_id = data.get('operation_id')
                    upload_ref = data.get('upload_ref')
                    head_sha = data.get('head_sha')
                    operation_round = data.get('round_id')
                    attachment_id = data.get('attachment_id')
                    native = data.get('native_attachment') is True
                    status = data.get('status')

                    # A one-time server-side diagnostic wrote these legacy
                    # records under a separate namespace without creating a
                    # resumable EvidenceService operation. Keep them out of
                    # the operation index while leaving attachment records
                    # (and their Summary projection) untouched.
                    legacy_server_operation = (
                        isinstance(record.get('key'), str)
                        and re.fullmatch(r'server:[A-Za-z0-9._:-]{1,128}', record['key']) is not None
                        and 'operation_id' not in data and 'upload_ref' not in data
                        and not native and data.get('repo') == repo
                        and data.get('pr_number') == pr_number
                        and isinstance(attachment_id, str)
                        and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', attachment_id) is not None
                        and isinstance(data.get('upload_id'), str)
                        and 1 <= len(data['upload_id']) <= 1024
                        and all(32 <= ord(char) < 127 for char in data['upload_id'])
                        and isinstance(head_sha, str)
                        and re.fullmatch(r'[0-9a-f]{40}', head_sha) is not None
                        and isinstance(operation_round, str)
                        and re.fullmatch(re.escape(head_sha) + r'-[0-9]+', operation_round) is not None
                        and isinstance(status, str)
                        and status in {'initializing', 'uploading', 'completing', 'verifying',
                                       'processing', 'ready', 'deleted', 'invalid'}
                    )
                    if legacy_server_operation:
                        continue

                    valid_attachment = (isinstance(attachment_id, str)
                                        and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', attachment_id) is not None)
                    valid_id = (valid_attachment and operation_id == attachment_id if native else
                                isinstance(operation_id, str)
                                and re.fullmatch(r'[0-9a-f]{64}', operation_id) is not None)
                    valid_context = (
                        data.get('repo') == repo and data.get('pr_number') == pr_number
                        and isinstance(head_sha, str)
                        and re.fullmatch(r'[0-9a-f]{40}', head_sha) is not None
                        and isinstance(operation_round, str)
                        and re.fullmatch(re.escape(head_sha) + r'-[0-9]+', operation_round) is not None
                    )
                    valid_status = (isinstance(status, str)
                                    and status in {'prepared', 'initializing', 'uploading',
                                                   'completing', 'verifying', 'processing',
                                                   'ready', 'deleted', 'invalid'})
                    valid_reference = (native and operation_id == attachment_id
                                       or not native and isinstance(upload_ref, str)
                                       and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', upload_ref) is not None)
                    if not (valid_id and valid_attachment and valid_context and valid_status
                            and valid_reference and data.get('native_attachment') in (None, True)):
                        raise EvidenceError('operation_record_invalid', 503)

                    previous = operations.get(operation_id)
                    if previous is not None and any(
                            previous.get(field) != data.get(field)
                            for field in ('repo', 'pr_number', 'head_sha', 'round_id',
                                          'attachment_id', 'upload_ref', 'actor_id')):
                        raise EvidenceError('operation_identity_conflict', 503)
                    operations[operation_id] = data
            if action == 'put_part':
                ref = payload.get('operation_id')
                source = next((r['data'] for r in reversed(records)
                               if r['kind'] == 'operation'
                               and r['data'].get('upload_ref') == ref), None)
                if (not source or source.get('repo') != repo
                        or source.get('pr_number') != pr_number
                        or source.get('head_sha') != expected_head
                        or source.get('round_id') != round_id):
                    raise EvidenceError('operation_not_found', 404)
                if (source.get('actor_id') != principal.get('id')
                        or isinstance(principal.get('upload_invite'), dict)
                        and source.get('upload_method') != 'signed_invite'):
                    raise EvidenceError('upload_owner_mismatch', 403)
                saved = latest.get(source.get('attachment_id'))
                if (source.get('status') != 'uploading' or saved
                        and saved.get('status') in {'deleted', 'invalid'}):
                    raise EvidenceError('upload_not_writable', 409)
                if not source.get('object_key') or not source.get('upload_id'):
                    raise EvidenceError('upload_not_recoverable', 503)
                part = payload.get('part_number')
                chunk = source.get('part_bytes', self.config.evidence_part_bytes)
                if (type(part) is not int or not 1 <= part <= 10000
                        or type(chunk) is not int or not 1 <= chunk <= 8 * 1024 * 1024
                        or type(source.get('size')) is not int
                        or not 1 <= source['size'] <= min(
                            512 * 1024 * 1024, self.config.evidence_max_file_bytes)
                        or part > (source['size'] + chunk - 1) // chunk):
                    raise EvidenceError('invalid_part_request', 400)
                expected_length = min(chunk, source['size'] - (part - 1) * chunk)
                body_stream = payload.get('_body_stream')
                if not hasattr(body_stream, '__aiter__'):
                    raise EvidenceError('invalid_part_body', 400)

                nonce_consumed = False

                async def current_part():
                    nonlocal nonce_consumed
                    fresh_pr = await self.proxy.get_pr(repo, pr_number)
                    fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                    validate_context(fresh_pr, fresh_state, expected_head, round_id, write=True)
                    fresh_authorized = await self._authorize(repo, principal, fresh_state, pr=fresh_pr)
                    if not (fresh_authorized or invite_allowed and invite_context_matches(
                            fresh_pr, fresh_state, round_id)):
                        raise EvidenceError('evidence_write_permission_required', 403)
                    if not nonce_consumed and not invite_allowed:
                        await self._consume_auth_nonce(principal)
                        nonce_consumed = True

                await current_part()
                with tempfile.SpooledTemporaryFile(max_size=256 * 1024, mode='w+b') as body:
                    length = 0
                    async for block in body_stream:
                        if not isinstance(block, bytes):
                            raise EvidenceError('invalid_part_body', 400)
                        length += len(block)
                        if length > min(8 * 1024 * 1024, chunk):
                            raise EvidenceError('evidence_part_too_large', 413)
                        body.write(block)
                    if length != expected_length:
                        raise EvidenceError(
                            'evidence_part_too_large' if length > expected_length
                            else 'invalid_part_size',
                            413 if length > expected_length else 400)
                    body.seek(0)
                    await current_part()
                    result = await self._store('put_part', source, {
                        'part_number': part, 'body': body, 'content_length': length,
                    })
                await current_part()
                if result.get('conflict') is True:
                    raise EvidenceError('part_conflict', 409)
                return {'uploaded': True, 'part_number': part, 'size': length}
            if action == 'context':
                def context_attachment(row):
                    return {**row, 'can_delete': bool(
                        authorized_writer or invite_allowed
                        and row.get('actor_id') == principal['id']
                        and row.get('upload_method') == 'signed_invite')}

                return {'repo': repo, 'pr_number': pr_number, 'head_sha': selected['head_sha'], 'round_id': round_id,
                        'status': selected['status'], 'authenticated': True, 'enabled': True,
                        'github_user_authenticated': bool(
                            principal.get('actor_verified') is True
                            and principal.get('github_user_authenticated') is True),
                        'can_write': bool((authorized_writer or invite_allowed) and selected.get('status') == 'testing'
                                          and selected.get('head_sha') == pr.get('head', {}).get('sha')),
                        'upload_invite': bool(invite_allowed),
                        'upload_identity_label': ('分享链接上传，GitHub 身份未验证'
                                                  if invite_allowed else ''),
                        'auth_method': principal.get('auth_method', 'github_comment_proof'),
                        'actor_verified': principal.get('actor_verified') is True,
                        'upload_method': principal.get('upload_method', 'github_comment_proof'),
                        'upload_max_files': (UPLOAD_MAX_FILES if invite_allowed else None),
                        'upload_max_bytes': (UPLOAD_MAX_BYTES if invite_allowed else None),
                        'attachments': [context_attachment(r) for r in latest.values()
                                        if r['round_id'] == round_id],
                        'rounds': [{**r['data']['snapshot'], 'round_id': r['data']['round_id'],
                                    'attachments': [context_attachment(a) for a in latest.values()
                                                    if a['round_id'] == r['data']['round_id']]}
                                   for r in records if r['kind'] == 'round'],
                        'operations': [{k: v for k, v in r.items() if k != 'upload_id'}
                                       for r in operations.values() if not r.get('auth_nonce_marker')
                                       and not r.get('native_attachment')
                                       and r.get('round_id') == round_id
                                       and r.get('actor_id') == principal['id']], 'records': [r for r in latest.values() if r['round_id'] == round_id],
                        # Evidence history is projected from verified round data;
                        # storage comment URLs/IDs are intentionally not exposed.
                        'archive_links': []}
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
            if action in {'download', 'preview', 'log_download'}:
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
                # Recheck the live PR and hidden state immediately before
                # verifying/signing. Historical rounds remain readable from
                # their trusted archived snapshot while the PR itself is open.
                before_pr = await self.proxy.get_pr(repo, pr_number)
                before_state = await self.proxy.read_hidden_state(repo, pr_number)
                if (before_pr.get('state') != 'open' or before_pr.get('merged')
                        or before_pr.get('draft')
                        or before_pr.get('base', {}).get('repo', {}).get('full_name', '').lower() != repo):
                    raise EvidenceError('pr_closed', 409)
                if not isinstance(before_state, dict):
                    raise EvidenceError('state_unavailable', 409)
                if round_id == active_round:
                    validate_context(before_pr, before_state, expected_head, round_id, write=False)
                # Store may sign ONLY the server-loaded object, never payload key.
                result = await self._store(action, operation, {})
                fresh = await self.proxy.get_pr(repo, pr_number)
                fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                if (fresh.get('state') != 'open' or fresh.get('merged') or fresh.get('draft')
                        or fresh.get('base', {}).get('repo', {}).get('full_name', '').lower() != repo
                        or not isinstance(fresh_state, dict)):
                    raise EvidenceError('pr_closed', 409)
                if round_id == active_round:
                    validate_context(fresh, fresh_state, expected_head, round_id, write=False)
                await self._authorize(repo, principal, selected, pr=fresh)
                validate_context(fresh, selected, expected_head, round_id, write=False)
                if action in {'download', 'preview'}:
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

            if invite_allowed and action in {'init', 'note'} and existing is None:
                invite_rows = {}
                for row in [*latest.values(), *operations.values()]:
                    if (row.get('upload_method') != 'signed_invite' or row.get('round_id') != round_id
                            or not isinstance(row.get('attachment_id'), str)):
                        continue
                    # Older note operations predate this marker; note.txt was
                    # their fixed filename. New operations carry an explicit
                    # kind so an uploaded file named note.txt still counts as a file.
                    row_kind = row.get('evidence_kind')
                    if row_kind not in {'file', 'note'}:
                        legacy_note_size = row.get('size')
                        row_kind = ('note' if row.get('filename') == 'note.txt'
                                    and type(legacy_note_size) is int and 1 <= legacy_note_size
                                    <= int(self.config.evidence_max_note_bytes) else 'file')
                    if row_kind not in {'file', 'note'}:
                        row_kind = 'file'
                    if row_kind != ('note' if action == 'note' else 'file'):
                        continue
                    attachment_id = row['attachment_id']
                    size = row.get('size', 0) if type(row.get('size')) is int else 0
                    invite_rows[attachment_id] = max(invite_rows.get(attachment_id, 0), size)

                if action == 'init':
                    requested_size = payload.get('size', 0)
                    if (type(requested_size) is not int
                            or not 1 <= requested_size <= 512 * 1024 * 1024):
                        raise EvidenceError('invalid_file_size', 413)
                    if len(invite_rows) >= UPLOAD_MAX_FILES:
                        raise EvidenceError('upload_invitation_file_quota_exceeded', 429)
                    if sum(invite_rows.values()) + requested_size > UPLOAD_MAX_BYTES:
                        raise EvidenceError('upload_invitation_byte_quota_exceeded', 413)
                else:
                    note_text = payload.get('text') or payload.get('note', '')
                    requested_size = len(note_text.encode('utf-8')) if isinstance(note_text, str) else 0
                    note_limit = int(self.config.evidence_max_note_bytes)
                    if len(invite_rows) >= UPLOAD_MAX_NOTES:
                        raise EvidenceError('upload_invitation_note_quota_exceeded', 429)
                    if (requested_size <= note_limit
                            and sum(invite_rows.values()) + requested_size > UPLOAD_MAX_BYTES):
                        raise EvidenceError('upload_invitation_note_byte_quota_exceeded', 413)

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
                fresh_authorized = await self._authorize(repo, principal, fresh_state, pr=fresh_pr)
                if not (fresh_authorized or invite_allowed and invite_context_matches(
                        fresh_pr, fresh_state, round_id)):
                    raise EvidenceError('evidence_write_permission_required', 403)
                if not nonce_consumed and not invite_allowed:
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
                    recovered.update(status='ready', sha256=sha, cos_content_verified=True,
                                     completed_at=utc_now())
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
                if (invite_allowed and replaces
                        and latest[replaces].get('actor_id') != principal['id']):
                    raise EvidenceError('upload_owner_mismatch', 403)
                from .evidence_builder import _redact_secrets
                note = payload.get('note', '')
                if not isinstance(note, str):
                    raise EvidenceError('invalid_note', 400)
                submitter = principal['login']
                operation = existing or {
                    'operation_id': opid, 'upload_ref': opid, 'attachment_id': opid, 'repo': repo, 'pr_number': pr_number,
                    'head_sha': expected_head, 'round_id': round_id, 'actor_id': principal['id'],
                    'actor_login': submitter, 'actor': 'shared-link' if invite_allowed else submitter,
                    'submitter': submitter,
                    'actor_verified': principal.get('actor_verified') is True,
                    'github_user_authenticated': principal.get('github_user_authenticated') is True,
                    'upload_method': ('signed_invite' if invite_allowed else 'github_comment_proof'),
                    'evidence_kind': 'note' if action == 'note' else 'file',
                    'upload_link_authorized': bool(invite_allowed),
                    'created_by': submitter,
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
                if ((source.get('actor_id') != principal['id']
                     or invite_allowed and source.get('upload_method') != 'signed_invite')
                        and action != 'delete'):
                    raise EvidenceError('upload_owner_mismatch', 403)
                if action != 'delete' and current_attachment and current_attachment.get('status') == 'ready':
                    return public(current_attachment)
                operation = existing or {**source, 'operation_id': opid,
                                         'source_operation_id': source['operation_id'], 'request_sha256': request_hash}
                if action == 'delete':
                    if (invite_allowed and (source.get('actor_id') != principal['id']
                                            or source.get('upload_method') != 'signed_invite')):
                        raise EvidenceError('upload_owner_mismatch', 403)
                    operation.update(status='deleted', deleted_at=utc_now(), deleted_by=principal['login'],
                                     deleted_by_login=principal['login'], deleted_by_id=principal['id'],
                                     actor_verified=principal.get('actor_verified') is True,
                                     github_user_authenticated=principal.get('github_user_authenticated') is True,
                                     updated_at=utc_now())
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
                operation.update(status='ready', sha256=actual_sha, cos_content_verified=True,
                                 completed_at=utc_now())
                await self._save(repo, pr_number, 'operation', operation, 'ready')
                await self._save(repo, pr_number, 'attachment', operation, 'ready')
            else:
                if not operation.get('upload_id') or not operation.get('object_key'):
                    raise EvidenceError('upload_not_recoverable', 502)
                operation['status'] = 'uploading'
                await self._save(repo, pr_number, 'operation', operation, 'uploading')
            return public(operation, result)
