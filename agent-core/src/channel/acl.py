"""
channel/acl.py — 访问控制层。

管理渠道用户的角色和权限：
- owner: 全部控制 + 用户管理
- operator: 所有 canvas 绑定工具（含执行器）
- viewer: 只读（状态查询、传感器数据）
- blocked: 拒绝一切
"""

import time
import sqlite3
import json

import config


ROLES = ('owner', 'operator', 'viewer', 'blocked')


def _conn() -> sqlite3.Connection:
    return config._get_conn()


def get_user(platform: str, user_id: str) -> dict | None:
    """查找用户，不存在返回 None。"""
    with _conn() as conn:
        row = conn.execute(
            'SELECT platform, platform_user_id, display_name, role, tool_filter, alert_subscriptions, created_at '
            'FROM channel_users WHERE platform=? AND platform_user_id=?',
            (platform, user_id)
        ).fetchone()
    if not row:
        return None
    return {
        'platform': row[0],
        'user_id': row[1],
        'display_name': row[2],
        'role': row[3],
        'tool_filter': row[4],
        'alert_subscriptions': json.loads(row[5]) if row[5] else [],
        'created_at': row[6],
    }


def upsert_user(platform: str, user_id: str, display_name: str = '',
                role: str = 'viewer', tool_filter: str = '*') -> dict:
    """创建或更新用户。"""
    if role not in ROLES:
        raise ValueError(f'Invalid role: {role}. Must be one of {ROLES}')
    now = time.time()
    with _conn() as conn:
        conn.execute(
            'INSERT INTO channel_users (platform, platform_user_id, display_name, role, tool_filter, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?) '
            'ON CONFLICT(platform, platform_user_id) DO UPDATE SET '
            'display_name=excluded.display_name, role=excluded.role, tool_filter=excluded.tool_filter',
            (platform, user_id, display_name, role, tool_filter, now)
        )
        conn.commit()
    return get_user(platform, user_id)


def delete_user(platform: str, user_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute(
            'DELETE FROM channel_users WHERE platform=? AND platform_user_id=?',
            (platform, user_id)
        )
        conn.commit()
        return cur.rowcount > 0


def list_users(platform: str = None) -> list[dict]:
    """列出用户，可按平台过滤。"""
    with _conn() as conn:
        if platform:
            rows = conn.execute(
                'SELECT platform, platform_user_id, display_name, role, tool_filter, alert_subscriptions, created_at '
                'FROM channel_users WHERE platform=? ORDER BY created_at',
                (platform,)
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT platform, platform_user_id, display_name, role, tool_filter, alert_subscriptions, created_at '
                'FROM channel_users ORDER BY created_at'
            ).fetchall()
    return [
        {
            'platform': r[0], 'user_id': r[1], 'display_name': r[2],
            'role': r[3], 'tool_filter': r[4],
            'alert_subscriptions': json.loads(r[5]) if r[5] else [],
            'created_at': r[6],
        }
        for r in rows
    ]


def check_permission(platform: str, user_id: str, required_role: str = 'viewer') -> tuple[bool, str]:
    """
    检查用户是否具有所需权限。
    返回 (allowed: bool, reason: str)。
    """
    user = get_user(platform, user_id)
    if user is None:
        return False, 'unknown_user'

    if user['role'] == 'blocked':
        return False, 'blocked'

    role_level = {'owner': 3, 'operator': 2, 'viewer': 1, 'blocked': 0}
    user_level = role_level.get(user['role'], 0)
    required_level = role_level.get(required_role, 0)

    if user_level >= required_level:
        return True, 'ok'
    return False, f'insufficient_role:{user["role"]}<{required_role}'
