"""Pure round identities and fail-closed evidence context validation."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone


class EvidenceError(ValueError):
    def __init__(self, code: str, status_code: int = 409):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def make_round_id(head_sha: str, request_comment_id: int) -> str:
    if (not isinstance(head_sha, str) or not re.fullmatch(r'[0-9a-f]{40}', head_sha)
            or type(request_comment_id) is not int or request_comment_id < 0):
        raise EvidenceError('invalid_round', 400)
    return f'{head_sha}-{request_comment_id}'


def round_id_for_state(state: dict) -> str:
    if not isinstance(state, dict):
        raise EvidenceError('invalid_state', 400)
    if 'round_id' in state:
        value = state['round_id']
        if (not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{40}-[0-9]+', value)
                or not value.startswith(str(state.get('head_sha')) + '-')):
            raise EvidenceError('invalid_round', 400)
        return value
    evidence = state.get('review_evidence', {})
    command = state.get('command', {})
    if not isinstance(evidence, dict) or not isinstance(command, dict):
        raise EvidenceError('invalid_state', 400)
    identifier = (command.get('comment_id', 0) if command.get('kind') == 'request_deploy'
                  else evidence.get('build_comment_id') or command.get('comment_id', 0))
    return make_round_id(state.get('head_sha'), identifier)


def archive_snapshot(repo, pr_number, state, archived_at, reason) -> dict:
    if (not isinstance(repo, str) or not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', repo)
            or type(pr_number) is not int or pr_number <= 0 or not isinstance(state, dict)
            or not isinstance(reason, str) or not reason or not isinstance(archived_at, str)):
        raise EvidenceError('invalid_archive', 400)
    try:
        if datetime.fromisoformat(archived_at.replace('Z', '+00:00')).utcoffset() is None:
            raise ValueError('timezone required')
    except ValueError:
        raise EvidenceError('invalid_archive_time', 400) from None
    rid = round_id_for_state(state)
    return {
        'repo': repo, 'pr_number': pr_number, 'head_sha': state['head_sha'],
        'round_id': rid, 'started_at': state.get('round_started_at'),
        'ended_at': archived_at, 'archived_at': archived_at, 'reason': reason,
        'snapshot': copy.deepcopy(state),
    }


def trusted_round_numbers(records, state, repo, pr_number) -> dict[str, int]:
    """Return round ordinals only when the complete immutable archive chain verifies."""
    if (not isinstance(records, (list, tuple)) or not isinstance(state, dict)
            or not isinstance(repo, str) or type(pr_number) is not int or pr_number <= 0):
        return {}
    try:
        current_round = round_id_for_state(state)
    except (EvidenceError, TypeError, ValueError):
        return {}
    current_head = state.get('head_sha')
    if (not isinstance(current_head, str) or not re.fullmatch(r'[0-9a-f]{40}', current_head)
            or not current_round.startswith(current_head + '-')):
        return {}

    def valid_link(value):
        if not isinstance(value, dict):
            return False
        if set(value) == {'round_id', 'sha256'}:
            return True
        return (set(value) == {'round_id', 'sha256', 'comment_id', 'url'}
                and type(value.get('comment_id')) is int and value['comment_id'] > 0
                and isinstance(value.get('url'), str)
                and value['url'].startswith('https://github.com/'))

    round_records = []
    other_rounds = set()
    for record in records:
        if not isinstance(record, dict):
            return {}
        data = record.get('data')
        if isinstance(data, dict) and isinstance(data.get('round_id'), str):
            if data['round_id'] != current_round:
                other_rounds.add(data['round_id'])
        if record.get('kind') == 'round':
            round_records.append(record)

    link = state.get('round_archive')
    if link is None:
        return ({current_round: 1}
                if 'round_archive' in state and not round_records and not other_rounds else {})
    if not valid_link(link):
        return {}

    by_sha = {}
    seen_ids = set()
    seen_keys = set()
    for record in round_records:
        digest, key, data = record.get('sha256'), record.get('key'), record.get('data')
        if (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
                or not isinstance(key, str) or not re.fullmatch(r'[0-9a-f]{64}', key)
                or not isinstance(data, dict) or digest in by_sha or key in seen_keys
                or not isinstance(data.get('round_id'), str)
                or data['round_id'] in seen_ids):
            return {}
        try:
            canonical = json.dumps(data, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode('utf-8')
        except (TypeError, ValueError, UnicodeEncodeError):
            return {}
        if hashlib.sha256(canonical).hexdigest() != digest:
            return {}
        snapshot = data.get('snapshot')
        attachments = data.get('attachment_records')
        if not isinstance(snapshot, dict) or not isinstance(attachments, list):
            return {}
        try:
            content = json.dumps(
                {'snapshot': snapshot, 'attachment_records': attachments},
                sort_keys=True, ensure_ascii=False,
            ).encode('utf-8')
        except (TypeError, ValueError, UnicodeEncodeError):
            return {}
        if hashlib.sha256(content).hexdigest() != key:
            return {}
        rid = data['round_id']
        head = data.get('head_sha')
        if (data.get('repo') != repo or type(data.get('pr_number')) is not int
                or data.get('pr_number') != pr_number
                or not isinstance(head, str) or not re.fullmatch(r'[0-9a-f]{40}', head)
                or not re.fullmatch(re.escape(head) + r'-[0-9]+', rid)
                or snapshot.get('head_sha') != head
                or (snapshot.get('repo') is not None and snapshot.get('repo') != repo)
                or (snapshot.get('pr_number') is not None
                    and (type(snapshot.get('pr_number')) is not int
                         or snapshot.get('pr_number') != pr_number))):
            return {}
        try:
            if round_id_for_state(snapshot) != rid:
                return {}
        except (EvidenceError, TypeError, ValueError):
            return {}
        nested_seen = set()
        for nested in attachments:
            if not isinstance(nested, dict):
                return {}
            nested_kind, nested_key = nested.get('kind'), nested.get('key')
            nested_data, nested_digest = nested.get('data'), nested.get('sha256')
            if (nested_kind not in {'operation', 'attachment'}
                    or not isinstance(nested_key, str) or not nested_key
                    or len(nested_key) > 512
                    or (nested_kind, nested_key) in nested_seen
                    or not isinstance(nested_data, dict)
                    or not isinstance(nested_digest, str)
                    or not re.fullmatch(r'[0-9a-f]{64}', nested_digest)
                    or nested_data.get('repo') != repo
                    or type(nested_data.get('pr_number')) is not int
                    or nested_data.get('pr_number') != pr_number
                    or nested_data.get('head_sha') != head
                    or nested_data.get('round_id') != rid):
                return {}
            try:
                nested_canonical = json.dumps(
                    nested_data, ensure_ascii=False, sort_keys=True,
                    separators=(',', ':'), allow_nan=False,
                ).encode('utf-8')
            except (TypeError, ValueError, UnicodeEncodeError):
                return {}
            if hashlib.sha256(nested_canonical).hexdigest() != nested_digest:
                return {}
            nested_seen.add((nested_kind, nested_key))
        by_sha[digest] = data
        seen_ids.add(rid)
        seen_keys.add(key)

    newest_first = []
    visited_sha = set()
    visited_ids = set()
    while True:
        digest, rid = link.get('sha256'), link.get('round_id')
        if (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
                or not isinstance(rid, str) or digest in visited_sha or rid in visited_ids):
            return {}
        archive = by_sha.get(digest)
        if archive is None or archive.get('round_id') != rid:
            return {}
        visited_sha.add(digest)
        visited_ids.add(rid)
        newest_first.append(rid)
        snapshot = archive['snapshot']
        if 'round_archive' not in snapshot:
            return {}
        link = snapshot['round_archive']
        if link is None:
            break
        if not valid_link(link):
            return {}

    if (visited_sha != set(by_sha) or current_round in visited_ids
            or other_rounds - visited_ids):
        return {}
    numbers = {rid: index + 1 for index, rid in enumerate(reversed(newest_first))}
    numbers[current_round] = len(newest_first) + 1
    return numbers


def validate_context(pr, state, expected_head, round_id, *, write: bool) -> None:
    if (not isinstance(pr, dict) or pr.get('state') not in {'open', 'closed'}
            or (write and pr.get('draft'))):
        raise EvidenceError('pr_closed', 409)
    if not isinstance(expected_head, str) or not re.fullmatch(r'[0-9a-f]{40}', expected_head):
        raise EvidenceError('invalid_head', 400)
    if not isinstance(state, dict):
        raise EvidenceError('state_unavailable', 409)
    if state.get('head_sha') != expected_head or round_id_for_state(state) != round_id:
        raise EvidenceError('round_changed', 409)
    if write:
        head = pr.get('head')
        if not isinstance(head, dict) or state.get('status') != 'testing' or head.get('sha') != expected_head:
            raise EvidenceError('context_not_writable', 409)
