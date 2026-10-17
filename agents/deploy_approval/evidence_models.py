"""Pure round identities and fail-closed evidence context validation."""
from __future__ import annotations

import copy
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


def validate_context(pr, state, expected_head, round_id, *, write: bool) -> None:
    if (not isinstance(pr, dict) or pr.get('state') not in {'open', 'closed'}
            or (write and (pr.get('state') != 'open' or pr.get('merged') or pr.get('draft')))):
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
