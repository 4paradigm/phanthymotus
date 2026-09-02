"""
peer/store.py — the `peers` table: who we have paired with, and what they may do.

Deliberately mirrors channel/acl.py: a peer is just another principal, and
reusing that role ladder means "what does operator mean" is defined once rather
than drifting between humans and robots.

A newly paired peer is `viewer` (read-only sensors). Granting an actuator role
is a separate, explicit act — and even then the actuator double gate in
event/llm.py still applies. See README § Multi-Agent Peers.
"""

import base64
import json
import time

import config
from channel.acl import ROLES


def _conn():
    return config._get_conn()


def _row_to_dict(row) -> dict:
    return {
        'peer_id': row[0],
        'display_name': row[1],
        'public_key': row[2],
        'role': row[3],
        'tool_filter': row[4],
        'endpoints': json.loads(row[5]) if row[5] else [],
        'capabilities': json.loads(row[6]) if row[6] else [],
        'paired_at': row[7],
        'last_seen': row[8],
    }


_COLS = ('peer_id, display_name, public_key, role, tool_filter, endpoints, '
         'capabilities, paired_at, last_seen')


def get(peer_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            f'SELECT {_COLS} FROM peers WHERE peer_id=?', (peer_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_peers() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(f'SELECT {_COLS} FROM peers ORDER BY paired_at').fetchall()
    return [_row_to_dict(r) for r in rows]


def upsert(peer_id: str, public_key_b64: str, display_name: str = '',
           role: str = 'viewer', tool_filter: str = '*',
           endpoints: list[str] | None = None,
           capabilities: list[str] | None = None) -> dict:
    if role not in ROLES:
        raise ValueError(f'Invalid role: {role}. Must be one of {ROLES}')
    if role == 'owner':
        # owner implies user management; a remote machine has no business there.
        raise ValueError('Peers cannot be owner')
    now = time.time()
    with _conn() as conn:
        conn.execute(
            'INSERT INTO peers (peer_id, display_name, public_key, role, tool_filter, '
            'endpoints, capabilities, paired_at, last_seen) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) '
            'ON CONFLICT(peer_id) DO UPDATE SET '
            'display_name=excluded.display_name, role=excluded.role, '
            'tool_filter=excluded.tool_filter, endpoints=excluded.endpoints, '
            'capabilities=excluded.capabilities',
            (peer_id, display_name, public_key_b64, role, tool_filter,
             json.dumps(endpoints or []), json.dumps(capabilities or []), now, now)
        )
        conn.commit()
    return get(peer_id)


def update(peer_id: str, **fields) -> dict | None:
    allowed = {'display_name', 'role', 'tool_filter', 'endpoints', 'capabilities'}
    sets, values = [], []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == 'role':
            if v not in ROLES:
                raise ValueError(f'Invalid role: {v}. Must be one of {ROLES}')
            if v == 'owner':
                raise ValueError('Peers cannot be owner')
        sets.append(f'{k}=?')
        values.append(json.dumps(v) if k in ('endpoints', 'capabilities') else v)
    if not sets:
        return get(peer_id)
    values.append(peer_id)
    with _conn() as conn:
        conn.execute(f'UPDATE peers SET {", ".join(sets)} WHERE peer_id=?', values)
        conn.commit()
    return get(peer_id)


def touch(peer_id: str) -> None:
    with _conn() as conn:
        conn.execute('UPDATE peers SET last_seen=? WHERE peer_id=?', (time.time(), peer_id))
        conn.commit()


def delete(peer_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute('DELETE FROM peers WHERE peer_id=?', (peer_id,))
        conn.commit()
        return cur.rowcount > 0


def public_key_bytes(peer_id: str) -> bytes | None:
    """Raw 32-byte public key of a *paired* peer, or None.

    This is the pin: a request claiming to be `peer_id` is verified against the
    key recorded at pairing time, never against a key the request supplies.
    """
    row = get(peer_id)
    if not row:
        return None
    try:
        return base64.b64decode(row['public_key'])
    except (ValueError, TypeError):
        return None


def check_permission(peer_id: str, required_role: str = 'viewer') -> tuple[bool, str]:
    """Same ladder as channel/acl.py, evaluated for a peer."""
    peer = get(peer_id)
    if peer is None:
        return False, 'unknown_peer'
    if peer['role'] == 'blocked':
        return False, 'blocked'
    level = {'owner': 3, 'operator': 2, 'viewer': 1, 'blocked': 0}
    if level.get(peer['role'], 0) >= level.get(required_role, 0):
        return True, 'ok'
    return False, f'insufficient_role:{peer["role"]}<{required_role}'
