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
        # None until the peer proves it has us too — see mark_mutual().
        'mutual_at': row[9] if len(row) > 9 else None,
    }


_COLS = ('peer_id, display_name, public_key, role, tool_filter, endpoints, '
         'capabilities, paired_at, last_seen, mutual_at')


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
    # (重新)配对是「情况变了」的那一刻，所以别让上一轮的退避再拖十几分钟才恢复。
    # 单向配对修好之后，人在界面上点完就该立刻看到对方上线。
    from peer import backoff
    backoff.reset(peer_id)
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


def touch(peer_id: str, endpoint: str = '') -> None:
    """Mark a peer as heard from just now, and learn its address if we lack one.

    The address matters because pairing is per-direction: the side that received
    the pair request may never have discovered the other (mDNS does not cross
    subnets), and it then stored no endpoints — so state pushes, tool calls and
    delegation could only ever travel one way. Measured between two real machines:
    43 pushes arrived in four minutes, none went back, and the peer read as
    offline on the side that had no address.

    Only filled in when empty. An address the peer advertised for itself, or one
    confirmed at pairing time, is better evidence than the source of one request,
    which a proxy or NAT would misreport.
    """
    with _conn() as conn:
        if endpoint:
            row = conn.execute('SELECT endpoints FROM peers WHERE peer_id=?',
                               (peer_id,)).fetchone()
            existing = json.loads(row[0]) if row and row[0] else []
            if not existing:
                conn.execute('UPDATE peers SET endpoints=? WHERE peer_id=?',
                             (json.dumps([endpoint]), peer_id))
                print(f'[peer] learned endpoint for {peer_id[:12]}: {endpoint}')
        # An authenticated inbound request is itself the evidence: a peer only
        # talks to agents it has paired.
        conn.execute('UPDATE peers SET last_seen=?, mutual_at=? WHERE peer_id=?',
                     (time.time(), time.time(), peer_id))
        conn.commit()


def mark_mutual(peer_id: str) -> None:
    """Record evidence that the peer has us in its own peers table.

    Two things count, and both mean the far side accepted a signed exchange:
    a state push of ours that it answered, and any authenticated request it sent
    us (it only pushes to peers it has). Until one of those happens, "paired" here
    is one-sided — the operator confirmed on this screen and nobody confirmed on
    the other, which used to look exactly like a finished pairing.
    """
    with _conn() as conn:
        conn.execute('UPDATE peers SET mutual_at=? WHERE peer_id=?', (time.time(), peer_id))
        conn.commit()


def update_display_name(peer_id: str, display_name: str) -> None:
    """Update the stored display name for a paired peer.

    Called when the peer pushes state with a new name — so a rename on one machine
    syncs to all others within one push interval (5s), rather than staying stale
    until the pairing is redone.
    """
    if not display_name or not isinstance(display_name, str):
        return
    with _conn() as conn:
        conn.execute('UPDATE peers SET display_name=? WHERE peer_id=?',
                     (display_name.strip()[:200], peer_id))
        conn.commit()


def delete(peer_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute('DELETE FROM peers WHERE peer_id=?', (peer_id,))
        conn.commit()
        return cur.rowcount > 0


# ── 审计 ──────────────────────────────────────────────────────────────────────

# 保留多少条。配对关系的变更是罕见事件 —— 几百条就能覆盖一台机器整个生命周期，
# 而无上限的表迟早会变成另一个需要有人去清的东西。
AUDIT_MAX_ROWS = 500

EVENT_PAIRED = 'paired'
EVENT_UNPAIRED_LOCAL = 'unpaired_local'
EVENT_UNPAIRED_BY_PEER = 'unpaired_by_peer'
EVENT_ROLE_CHANGED = 'role_changed'


def record_audit(peer_id: str, event: str, *, display_name: str = '',
                 actor: str = '', detail: str = '') -> None:
    """记一条 peer 关系变更。只增不改，失败不抛。

    `peers` 表回答「现在是什么」，删掉一行之后那段关系就不再有任何痕迹。这张表
    回答「发生过什么」—— 写它的直接理由是：有人在天轶上解除了和 Orin5 的配对，
    一周后要查是谁、什么时候，`docker logs` 已经轮换过去，活动流只在内存里，
    数据库里只剩一张空表，于是这个问题根本没有答案。

    审计写失败绝不能让配对/解除本身失败：一条查不到的记录，远好过一次做不成的操作。
    """
    try:
        with _conn() as conn:
            conn.execute(
                'INSERT INTO peer_audit (ts, peer_id, display_name, event, actor, detail) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (time.time(), peer_id, display_name or '', event, actor or '',
                 str(detail or '')[:500]))
            # 按 id 裁剪而不是按时间：id 是单调的，而 ts 来自 wall clock，机器上
            # 时钟跳变过（G1 上的容器启动时间是 1970），按 ts 排序会留错行。
            conn.execute(
                'DELETE FROM peer_audit WHERE id <= '
                '(SELECT MAX(id) FROM peer_audit) - ?', (AUDIT_MAX_ROWS,))
            conn.commit()
    except Exception as e:
        print(f'[peer] audit write failed ({type(e).__name__}: {e}) — ignored')


def list_audit(limit: int = 50, peer_id: str = '') -> list[dict]:
    """最近的关系变更，最新在前。"""
    limit = max(1, min(int(limit or 50), AUDIT_MAX_ROWS))
    sql = ('SELECT ts, peer_id, display_name, event, actor, detail FROM peer_audit')
    args: list = []
    if peer_id:
        sql += ' WHERE peer_id=?'
        args.append(peer_id)
    sql += ' ORDER BY id DESC LIMIT ?'
    args.append(limit)
    with _conn() as conn:
        rows = conn.execute(sql, tuple(args)).fetchall()
    return [{'ts': r[0], 'peer_id': r[1], 'display_name': r[2],
             'event': r[3], 'actor': r[4], 'detail': r[5]} for r in rows]


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
