#!/usr/bin/env python3
"""Offline, target-only Motus skill management. Never creates a missing DB."""
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from contextlib import closing

from guide import FIELDS, ROOT, SLUG, digest, load_json


def connect(path, readonly=False):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('数据库必须是已存在的普通文件')
    return sqlite3.connect(path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'),
                           uri=True, timeout=5, isolation_level=None)


def schema(conn):
    if conn.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
        raise ValueError('数据库完整性检查失败')
    columns = conn.execute('PRAGMA table_info(config)').fetchall()
    if (len(columns) != 2 or [(c[1], c[2].upper(), c[5]) for c in columns]
            != [('key', 'TEXT', 1), ('value', 'TEXT', 0)] or not columns[1][3]):
        raise ValueError('config 表结构不是 key TEXT PRIMARY KEY, value TEXT NOT NULL')


def read_config(conn):
    row = conn.execute('SELECT value FROM config WHERE key=?', ('skills',)).fetchone()
    cfg = load_value(row[0]) if row else {'installed': []}
    if not isinstance(cfg, dict) or not isinstance(cfg.get('installed'), list):
        raise ValueError('skills 必须为包含 installed 数组的对象')
    if any(not isinstance(s, dict) or not isinstance(s.get('slug'), str)
           for s in cfg['installed']):
        raise ValueError('installed 含无效技能对象')
    if sum(s['slug'] == SLUG for s in cfg['installed']) > 1:
        raise ValueError('发现多条相同目标 slug，停止修改')
    return cfg


def load_value(text):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('skills JSON 含重复字段')
            d[k] = v
        return d
    return json.loads(text, object_pairs_hook=unique)


def target(cfg):
    return next((s for s in cfg['installed'] if s['slug'] == SLUG), None)


def definition(path):
    value = load_json(path)
    limits = dict(slug=100, name=100, oneLiner=300, description=4000,
                  instruction=20000, category=80, version=40)
    if not isinstance(value, dict) or set(value) != set(FIELDS) or value.get('slug') != SLUG:
        raise ValueError('仅接受本工具生成的便携站台技能定义；不接受完整数据库对象')
    for k, limit in limits.items():
        if not isinstance(value[k], str) or len(value[k]) > limit:
            raise ValueError(f'无效定义字段: {k}')
    if any(not value[k].strip() for k in ('name', 'oneLiner', 'instruction')):
        raise ValueError('名称、简介和指令不能为空')
    tools = value['requiredTools']
    if (not isinstance(tools, list) or len(tools) > 32 or
            any(not isinstance(t, str) or not re.fullmatch(r'[\w:.-]{1,200}', t) for t in tools)):
        raise ValueError('无效工具依赖')
    if len(json.dumps(value, ensure_ascii=False).encode()) > 32768:
        raise ValueError('定义超过32KiB')
    return value


def backup_database(path):
    path = Path(path).resolve(strict=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    fd, name = tempfile.mkstemp(prefix=path.name + '.' + stamp + '.', suffix='.bak', dir=path.parent)
    os.close(fd)  # mkstemp creates mode 0600; full DB backups may contain secrets.
    # Separate reader: backing up the connection holding BEGIN IMMEDIATE can hang.
    with closing(connect(path, readonly=True)) as src, closing(sqlite3.connect(name)) as dst:
        src.backup(dst)
        if dst.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('备份完整性检查失败，未修改原库')
    return name


def manage(db, action='check', skill=None, dry_run=False, core_stopped=False,
           backup=None, expect_current=None):
    writing = action != 'check' and not dry_run
    if writing and not core_stopped:
        raise ValueError('离线修改前须核实并停止实际 Agent Core，然后传 --core-stopped；该参数不是停止服务命令')
    proposed = definition(skill or ROOT / 'skill.json') if action in ('install', 'update') else None
    old_target = None
    if action == 'restore':
        if not backup or not expect_current:
            raise ValueError('恢复必须提供 --backup 和 --expect-current（本次操作输出的 after_sha256）')
        if Path(backup).resolve() == Path(db).resolve():
            raise ValueError('备份不能是当前数据库自身')
        with closing(connect(backup, readonly=True)) as saved:
            schema(saved)
            old_target = target(read_config(saved))
    with closing(connect(db, readonly=not writing)) as conn:
        schema(conn)
        conn.execute('BEGIN IMMEDIATE' if writing else 'BEGIN')
        try:
            before = read_config(conn)
            previous = target(before)
            after = copy.deepcopy(before)
            current = target(after)
            desired = current
            if action == 'install':
                if current is not None:
                    if any(current.get(k) != v for k, v in proposed.items()):
                        raise ValueError('相同 slug 定义冲突；请明确使用 update')
                else:
                    desired = {**proposed, 'active': True,
                               'installedAt': dt.datetime.now(dt.timezone.utc).isoformat()}
            elif action == 'update':
                if current is None:
                    raise ValueError('技能尚未安装；请使用 install')
                desired = {**current, **proposed}  # Preserve installedAt, active and unknown metadata.
            elif action == 'uninstall':
                desired = None
            elif action == 'restore':
                if digest(previous) != expect_current:
                    raise ValueError('目标技能已发生后续变化，拒绝覆盖；先人工核对')
                desired = old_target
            elif action != 'check':
                raise ValueError('未知操作')
            if current is not None:
                index = next(i for i, s in enumerate(after['installed']) if s['slug'] == SLUG)
                if desired is None:
                    after['installed'].pop(index)
                else:
                    after['installed'][index] = desired
            elif desired is not None:
                after['installed'].append(desired)
            changed = before != after
            report = dict(ok=True, action=action, changed=changed, dry_run=dry_run,
                          installed=desired is not None, before_sha256=digest(previous),
                          after_sha256=digest(desired), backup=None)
            if changed and writing:
                report['backup'] = backup_database(db)
                other_rows = conn.execute('SELECT key,value FROM config WHERE key<>? ORDER BY key', ('skills',)).fetchall()
                conn.execute('INSERT INTO config(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                             ('skills', json.dumps(after, ensure_ascii=False)))
                if read_config(conn) != after or conn.execute(
                        'SELECT key,value FROM config WHERE key<>? ORDER BY key', ('skills',)).fetchall() != other_rows:
                    raise ValueError('写后验证失败，事务已回滚')
                conn.execute('COMMIT')
                if read_config(conn) != after:
                    raise ValueError('提交后验证不一致；停止服务并核对备份')
            else:
                conn.execute('ROLLBACK')
            return report
        except BaseException:
            if conn.in_transaction:
                conn.execute('ROLLBACK')
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', default='check', choices=['check', 'install', 'update', 'uninstall', 'restore'])
    p.add_argument('--db', type=Path, required=True)
    p.add_argument('--skill', type=Path, default=ROOT / 'skill.json')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--core-stopped', action='store_true')
    p.add_argument('--backup', type=Path)
    p.add_argument('--expect-current')
    a = p.parse_args()
    try:
        result = manage(a.db, a.action, a.skill, a.dry_run, a.core_stopped, a.backup, a.expect_current)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, sqlite3.Error) as e:
        # Do not dump skills JSON or other configuration values.
        p.exit(1, f'失败：{e}\n')


if __name__ == '__main__':
    main()
