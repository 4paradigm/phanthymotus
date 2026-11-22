#!/usr/bin/env python3
"""
plugins/table_db_store.py — the SQLite layer behind the `table_db` card.

One SQLite file per *library* (`db`, one per skill by convention), several user
tables per library. Standard library only.

Layout of a library file
------------------------

    _tables    (id, name, created_at)                 user table names
    _columns   (table_id, pos, name, type, phys)      user column names and types
    _files     (id, sha256, name, mime, size, ...)    one row per stored attachment
    _blobs     (sha256, data)                          file bytes, deduplicated
    _requests  (request_id, digest, result, ...)      idempotent retries
    t_<id>     (_id, _created_at, _updated_at, _deleted_at, c_1, c_2, ...)

Two column types only: `text`, and `file` (anything that is not text — a photo,
a PDF, a recording). A file column holds an integer `_files.id`; the bytes live
in `_blobs`, so a record and its attachments commit or roll back together and
the whole library is one file to back up.

Why user names never reach SQL
------------------------------

Table and column names come from a spreadsheet header or a model, so they are
data, not identifiers: they are stored in `_tables` / `_columns` and every
physical name (`t_3`, `c_5`) is generated here. Values always go through
parameter binding. There is no path by which caller text becomes SQL.

Concurrency
-----------

perception answers tool calls on separate threads. Each operation opens its own
connection (WAL, `busy_timeout` 5 s); writes to one library are additionally
serialised by an in-process lock and run in `BEGIN IMMEDIATE`, so a writer
either sees a committed state or waits. A lock held past the timeout surfaces as
`sqlite3.OperationalError`, which the card reports as `busy`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

TEXT = "text"
FILE = "file"
COLUMN_TYPES = (TEXT, FILE)

_DB_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FILE_ID = re.compile(r"^f_(\d{1,12})$")
MAX_NAME_CHARS = 64
MAX_IN_VALUES = 500
MAX_GROUPS = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS _tables(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS _columns(
    table_id INTEGER NOT NULL, pos INTEGER NOT NULL, name TEXT NOT NULL,
    type TEXT NOT NULL, phys TEXT NOT NULL, PRIMARY KEY(table_id, name));
CREATE TABLE IF NOT EXISTS _blobs(sha256 TEXT PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS _files(
    id INTEGER PRIMARY KEY AUTOINCREMENT, sha256 TEXT NOT NULL, name TEXT NOT NULL,
    mime TEXT NOT NULL, size INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS _requests(
    request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, result TEXT NOT NULL,
    created_at TEXT NOT NULL);
"""


class TableDbError(Exception):
    """A refusal the caller can act on: `reason` to branch on, `detail` to say."""

    def __init__(self, reason: str, detail: str, row: Optional[int] = None,
                 column: Optional[str] = None, **extra: Any):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.row = row
        self.column = column
        self.extra = extra

    def as_result(self) -> dict:
        out: Dict[str, Any] = {"ok": False, "reason": self.reason, "detail": self.detail}
        if self.row is not None:
            out["row"] = self.row
        if self.column is not None:
            out["column"] = self.column
        out.update(self.extra)
        # Every failure is all-or-nothing; saying so on every reply is what lets
        # a skill tell the user "nothing was saved" without checking.
        out["written"] = 0
        return out


def bad_request(detail: str, **extra: Any) -> TableDbError:
    return TableDbError("bad_request", detail, **extra)


@dataclass
class FileValue:
    """An attachment on its way into the store."""

    data: bytes
    name: str
    mime: str


@dataclass
class Column:
    name: str
    type: str
    phys: str

    def describe(self) -> dict:
        return {"name": self.name, "type": self.type}


@dataclass
class StoreLimits:
    max_file_bytes: int = 8 * 1024 * 1024
    max_db_bytes: int = 2 * 1024 * 1024 * 1024
    max_rows: int = 20000
    max_columns: int = 64
    max_tables: int = 50
    max_cell_chars: int = 2000


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def check_db_name(db: Any) -> str:
    value = "default" if db in (None, "") else str(db).strip()
    if not _DB_NAME.match(value):
        raise bad_request("db 只能用字母、数字、- 和 _，1–64 个字符（建议填 skill 的 slug），"
                          "收到 %r" % (db,))
    return value


def check_name(value: Any, what: str) -> str:
    """A user-facing table or column name: 1-64 chars, no leading `_`."""
    if not isinstance(value, str):
        raise bad_request("%s 要是字符串，收到 %r" % (what, value))
    name = value.strip()
    if not name:
        raise bad_request("%s 不能为空" % what)
    if len(name) > MAX_NAME_CHARS:
        raise bad_request("%s「%s…」超过 %d 个字符" % (what, name[:16], MAX_NAME_CHARS))
    if name.startswith("_"):
        raise bad_request("%s「%s」不能以 _ 开头（_ 开头的名字留给系统字段）" % (what, name))
    if any(ord(char) < 32 for char in name):
        raise bad_request("%s「%s」含有控制字符" % (what, name))
    return name


def check_columns(columns: Any, limits: StoreLimits) -> List[Tuple[str, str]]:
    """`[{"name": .., "type": "text"|"file"}]` -> [(name, type)], validated."""
    if not isinstance(columns, list) or not columns:
        raise bad_request('columns 要是非空数组，如 [{"name":"姓名","type":"text"},'
                          '{"name":"照片","type":"file"}]')
    if len(columns) > limits.max_columns:
        raise TableDbError("too_large", "最多 %d 列，收到 %d 列" % (limits.max_columns,
                                                                len(columns)))
    out: List[Tuple[str, str]] = []
    seen = set()
    for item in columns:
        if isinstance(item, str):
            name, kind = item, TEXT
        elif isinstance(item, dict):
            name, kind = item.get("name"), str(item.get("type") or TEXT).strip().lower()
        else:
            raise bad_request("columns 的每一项要是 {\"name\":..,\"type\":..}，收到 %r" % (item,))
        name = check_name(name, "列名")
        if kind not in COLUMN_TYPES:
            raise bad_request("列「%s」的类型 %r 不认识：只有 text（文字）和 file（非文字："
                              "图片/PDF/音频等）" % (name, kind), column=name)
        if name in seen:
            raise bad_request("列名「%s」重复" % name, column=name)
        seen.add(name)
        out.append((name, kind))
    return out


def text_value(value: Any, column: str, limits: StoreLimits, where: str) -> Optional[str]:
    """Coerce a value for a text column. "" and None both mean empty."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise bad_request("%s「%s」的值是布尔 %r；文字列请传字符串，如 \"是\" / \"否\""
                          % (where, column, value), column=column)
    if isinstance(value, float):
        value = int(value) if value.is_integer() else repr(value)
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        raise bad_request("%s「%s」的值要是字符串或数字，收到 %s" % (
            where, column, type(value).__name__), column=column)
    if value == "":
        return None
    if len(value) > limits.max_cell_chars:
        raise TableDbError("too_large", "%s「%s」有 %d 个字符，超过上限 %d" % (
            where, column, len(value), limits.max_cell_chars), column=column)
    return value


def parse_file_id(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    match = _FILE_ID.match(str(value or "").strip())
    if not match:
        raise bad_request("file_id 的格式是 f_数字（例如 f_42，来自 query 的返回），收到 %r"
                          % (value,))
    return int(match.group(1))


def file_info(row: Sequence[Any]) -> dict:
    return {"file_id": "f_%d" % row[0], "name": row[1], "mime": row[2], "size": row[3]}


class TableStore:
    """All SQL lives here. Every public method validates, then acts atomically."""

    def __init__(self, data_dir: str, limits: Optional[StoreLimits] = None):
        self.data_dir = data_dir
        self.limits = limits or StoreLimits()
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # ── connections ──────────────────────────────────────────────────────────

    def db_path(self, db: str) -> str:
        return os.path.join(self.data_dir, "%s.db" % check_db_name(db))

    def databases(self) -> List[str]:
        try:
            names = sorted(os.listdir(self.data_dir))
        except OSError:
            return []
        return [name[:-3] for name in names
                if name.endswith(".db") and _DB_NAME.match(name[:-3])]

    def _connect(self, db: str, create: bool) -> Optional[sqlite3.Connection]:
        path = self.db_path(db)
        if not create and not os.path.isfile(path):
            return None
        os.makedirs(self.data_dir, exist_ok=True)
        conn = sqlite3.connect(path, timeout=5.0, isolation_level=None,
                               check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        return conn

    def _lock(self, db: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(db, threading.Lock())

    @contextmanager
    def _read(self, db: str):
        conn = self._connect(db, create=False)
        try:
            yield conn
        finally:
            if conn is not None:
                conn.close()

    @contextmanager
    def _write(self, db: str):
        with self._lock(db):
            conn = self._connect(db, create=True)
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def _check_size(self, db: str, extra: int) -> None:
        path = self.db_path(db)
        size = sum(os.path.getsize(part) for part in (path, path + "-wal")
                   if os.path.exists(part))
        if size + extra > self.limits.max_db_bytes:
            raise TableDbError("too_large", "库 %s 已有 %d 字节，再写 %d 字节会超过上限 %d"
                               % (db, size, extra, self.limits.max_db_bytes))

    # ── metadata ─────────────────────────────────────────────────────────────

    @staticmethod
    def _table_names(conn: Optional[sqlite3.Connection]) -> List[str]:
        if conn is None:
            return []
        return [row[0] for row in conn.execute("SELECT name FROM _tables ORDER BY id")]

    def _meta(self, conn: Optional[sqlite3.Connection], table: str) -> Tuple[int, List[Column]]:
        row = None
        if conn is not None:
            row = conn.execute("SELECT id FROM _tables WHERE name=?", (table,)).fetchone()
        if row is None:
            names = self._table_names(conn)
            raise TableDbError("unknown_table", "没有表「%s」。现有的表：%s" % (
                table, "、".join(names) or "（无，先 create 或 import）"), tables=names)
        columns = [Column(*item) for item in conn.execute(
            "SELECT name, type, phys FROM _columns WHERE table_id=? ORDER BY pos", (row[0],))]
        return row[0], columns

    @staticmethod
    def _column(columns: List[Column], name: Any) -> Column:
        for column in columns:
            if column.name == name:
                return column
        raise TableDbError("unknown_column", "没有列「%s」。现有的列：%s" % (
            name, "、".join(column.name for column in columns)),
            column=str(name), columns=[column.describe() for column in columns])

    def columns(self, db: str, table: str) -> Optional[List[Column]]:
        """The table's columns, or None when the table does not exist."""
        with self._read(db) as conn:
            try:
                return self._meta(conn, table)[1]
            except TableDbError:
                return None

    def tables(self, db: str) -> List[dict]:
        with self._read(db) as conn:
            if conn is None:
                return []
            out = []
            for tid, name in conn.execute("SELECT id, name FROM _tables ORDER BY id").fetchall():
                columns = self._meta(conn, name)[1]
                rows = conn.execute('SELECT COUNT(*) FROM "t_%d" WHERE _deleted_at IS NULL'
                                    % tid).fetchone()[0]
                out.append({"table": name, "rows": rows,
                            "columns": [column.describe() for column in columns]})
            return out

    def overview(self, db: str) -> List[dict]:
        """Tables with columns, live rows, attached files and last change."""
        with self._read(db) as conn:
            if conn is None:
                return []
            out = []
            for tid, name, created in conn.execute(
                    "SELECT id, name, created_at FROM _tables ORDER BY id").fetchall():
                columns = self._meta(conn, name)[1]
                live = ' WHERE _deleted_at IS NULL'
                rows, updated = conn.execute(
                    'SELECT COUNT(*), MAX(_updated_at) FROM "t_%d"%s' % (tid, live)).fetchone()
                files = 0
                for column in columns:
                    if column.type == FILE:
                        files += conn.execute('SELECT COUNT("%s") FROM "t_%d"%s' % (
                            column.phys, tid, live)).fetchone()[0]
                out.append({"table": name, "rows": rows, "files": files,
                            "columns": [column.describe() for column in columns],
                            "created_at": created, "updated_at": updated or created})
            return out

    def db_bytes(self, db: str) -> int:
        path = self.db_path(db)
        return sum(os.path.getsize(part) for part in (path, path + "-wal")
                   if os.path.exists(part))

    def drop_table(self, db: str, table: str) -> dict:
        """Remove a table, its metadata and any file no other row still uses."""
        with self._write(db) as conn:
            tid, columns = self._meta(conn, table)
            rows = conn.execute('SELECT COUNT(*) FROM "t_%d" WHERE _deleted_at IS NULL'
                                % tid).fetchone()[0]
            file_ids: List[int] = []
            for column in columns:
                if column.type == FILE:
                    file_ids += [row[0] for row in conn.execute(
                        'SELECT "%s" FROM "t_%d" WHERE "%s" IS NOT NULL'
                        % (column.phys, tid, column.phys))]
            conn.execute('DROP TABLE "t_%d"' % tid)
            conn.execute("DELETE FROM _columns WHERE table_id=?", (tid,))
            conn.execute("DELETE FROM _tables WHERE id=?", (tid,))
            for start in range(0, len(file_ids), MAX_IN_VALUES):
                chunk = file_ids[start:start + MAX_IN_VALUES]
                conn.execute("DELETE FROM _files WHERE id IN (%s)" % ",".join("?" * len(chunk)),
                             chunk)
            conn.execute("DELETE FROM _blobs WHERE sha256 NOT IN (SELECT sha256 FROM _files)")
            # Forget the requests that wrote this table: replaying "imported 5"
            # for a table that no longer exists would tell a skill its re-import
            # succeeded when nothing was written.
            stale = []
            for request_id, result in conn.execute("SELECT request_id, result FROM _requests"):
                try:
                    if json.loads(result).get("table") == table:
                        stale.append(request_id)
                except ValueError:
                    continue
            for request_id in stale:
                conn.execute("DELETE FROM _requests WHERE request_id=?", (request_id,))
            return {"rows": rows, "files": len(file_ids)}

    def remove_if_empty(self, db: str) -> bool:
        """Delete a library that holds no table (left behind by a failed first write)."""
        with self._lock(db):
            with self._read(db) as conn:
                if conn is None:
                    return False
                if conn.execute("SELECT COUNT(*) FROM _tables").fetchone()[0]:
                    return False
            path = self.db_path(db)
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(path + suffix)
                except FileNotFoundError:
                    continue
            return True

    def drop_db(self, db: str) -> dict:
        """Delete a whole library file (and its WAL/SHM). Returns what it held."""
        path = self.db_path(db)
        with self._lock(db):
            tables = self.overview(db)
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(path + suffix)
                except FileNotFoundError:
                    continue
        return {"tables": len(tables), "table_list": [item["table"] for item in tables],
                "rows": sum(item["rows"] for item in tables),
                "files": sum(item["files"] for item in tables)}

    def quick_check(self, db: str) -> str:
        with self._read(db) as conn:
            if conn is None:
                return "missing"
            return str(conn.execute("PRAGMA quick_check").fetchone()[0])

    def _create_in(self, conn: sqlite3.Connection, table: str,
                   columns: List[Tuple[str, str]]) -> Tuple[int, List[Column]]:
        if conn.execute("SELECT 1 FROM _tables WHERE name=?", (table,)).fetchone():
            raise TableDbError("table_exists", "表「%s」已经存在。追加数据用 import(mode=\"append\") "
                               "或 append；要新表请换个表名" % table)
        count = conn.execute("SELECT COUNT(*) FROM _tables").fetchone()[0]
        if count >= self.limits.max_tables:
            raise TableDbError("too_large", "一个库最多 %d 张表" % self.limits.max_tables)
        tid = conn.execute("INSERT INTO _tables(name, created_at) VALUES(?, ?)",
                           (table, now_text())).lastrowid
        out = []
        for pos, (name, kind) in enumerate(columns, 1):
            column = Column(name, kind, "c_%d" % pos)
            conn.execute("INSERT INTO _columns(table_id, pos, name, type, phys) "
                         "VALUES(?, ?, ?, ?, ?)", (tid, pos, name, kind, column.phys))
            out.append(column)
        definitions = ", ".join('"%s" %s' % (column.phys,
                                             "INTEGER" if column.type == FILE else "TEXT")
                                for column in out)
        conn.execute('CREATE TABLE "t_%d"(_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                     '_created_at TEXT NOT NULL, _updated_at TEXT NOT NULL, '
                     '_deleted_at TEXT, %s)' % (tid, definitions))
        return tid, out

    # ── idempotency ──────────────────────────────────────────────────────────

    @staticmethod
    def _replay(conn: sqlite3.Connection, request: Optional[Tuple[str, str]]) -> Optional[dict]:
        if not request:
            return None
        found = conn.execute("SELECT digest, result FROM _requests WHERE request_id=?",
                             (request[0],)).fetchone()
        if found is None:
            return None
        if found[0] != request[1]:
            raise TableDbError("request_conflict", "request_id %r 之前用于另一批内容；"
                               "这是一次新的写入就换一个新的 request_id" % request[0])
        result = json.loads(found[1])
        result["replayed"] = True
        return result

    def lookup_request(self, db: str, request: Optional[Tuple[str, str]]) -> Optional[dict]:
        """The stored result of an earlier identical request, before any pre-check.

        A retried `import mode=create` must replay, not fail with table_exists
        because its own first attempt created the table.
        """
        if not request:
            return None
        with self._read(db) as conn:
            if conn is None:
                return None
            return self._replay(conn, request)

    @staticmethod
    def _remember(conn: sqlite3.Connection, request: Optional[Tuple[str, str]],
                  result: dict) -> None:
        if request:
            conn.execute("INSERT INTO _requests(request_id, digest, result, created_at) "
                         "VALUES(?, ?, ?, ?)", (request[0], request[1],
                                                json.dumps(result, ensure_ascii=False),
                                                now_text()))

    # ── files ────────────────────────────────────────────────────────────────

    def _put_file(self, conn: sqlite3.Connection, value: FileValue,
                  column: str, row: Optional[int]) -> dict:
        size = len(value.data)
        if size > self.limits.max_file_bytes:
            raise TableDbError("too_large", "「%s」里的文件 %s 有 %d 字节，超过单文件上限 %d"
                               % (column, value.name, size, self.limits.max_file_bytes),
                               row=row, column=column)
        if size == 0:
            raise bad_request("「%s」里的文件 %s 是空的" % (column, value.name),
                              row=row, column=column)
        digest = hashlib.sha256(value.data).hexdigest()
        conn.execute("INSERT OR IGNORE INTO _blobs(sha256, data) VALUES(?, ?)",
                     (digest, sqlite3.Binary(value.data)))
        fid = conn.execute("INSERT INTO _files(sha256, name, mime, size, created_at) "
                           "VALUES(?, ?, ?, ?, ?)",
                           (digest, value.name, value.mime, size, now_text())).lastrowid
        return {"file_id": "f_%d" % fid, "name": value.name, "mime": value.mime,
                "size": size, "_fid": fid}

    def read_files(self, db: str, ids: Iterable[int]) -> Dict[int, Tuple[str, str, bytes]]:
        wanted = sorted(set(ids))
        out: Dict[int, Tuple[str, str, bytes]] = {}
        with self._read(db) as conn:
            if conn is None:
                return out
            for start in range(0, len(wanted), MAX_IN_VALUES):
                chunk = wanted[start:start + MAX_IN_VALUES]
                for fid, name, mime, data in conn.execute(
                        "SELECT f.id, f.name, f.mime, b.data FROM _files f JOIN _blobs b "
                        "ON b.sha256 = f.sha256 WHERE f.id IN (%s)"
                        % ",".join("?" * len(chunk)), chunk):
                    out[fid] = (name, mime, bytes(data))
        return out

    # ── where ────────────────────────────────────────────────────────────────

    def _where(self, columns: List[Column], where: Any) -> Tuple[str, List[Any]]:
        if where in (None, {}):
            return "", []
        if not isinstance(where, dict):
            raise bad_request('where 要是对象，如 {"状态":"未归还","借用人":["张三","李四"],'
                              '"归还时间":null}')
        parts: List[str] = []
        params: List[Any] = []
        for key, value in where.items():
            if key == "_id":
                expression, column = '"_id"', None
            else:
                column = self._column(columns, key)
                expression = '"%s"' % column.phys
            if value is None or value == "":
                parts.append(expression + " IS NULL")
                continue
            values = value if isinstance(value, list) else [value]
            if not values or len(values) > MAX_IN_VALUES:
                raise bad_request("where「%s」的数组要有 1–%d 个值" % (key, MAX_IN_VALUES),
                                  column=key)
            bound = [self._where_value(column, key, item) for item in values]
            if len(bound) == 1:
                parts.append(expression + " = ?")
            else:
                parts.append("%s IN (%s)" % (expression, ",".join("?" * len(bound))))
            params.extend(bound)
        return " AND " + " AND ".join(parts), params

    def _where_value(self, column: Optional[Column], key: str, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            raise bad_request(
                "where「%s」只支持三种写法：一个值（等于）、数组（等于其中之一）、null（为空）；"
                "不支持 %s" % (key, json.dumps(value, ensure_ascii=False)), column=key)
        if column is None:                               # _id
            try:
                if isinstance(value, bool):
                    raise ValueError
                return int(value)
            except (TypeError, ValueError):
                raise bad_request("where 里 _id 要是整数，收到 %r" % (value,), column=key)
        if column.type == FILE:
            return parse_file_id(value)
        return text_value(value, key, self.limits, "where")

    # ── records ──────────────────────────────────────────────────────────────

    def _records(self, conn: sqlite3.Connection, columns: List[Column],
                 rows: List[Sequence[Any]]) -> List[dict]:
        file_ids = {row[3 + index] for row in rows
                    for index, column in enumerate(columns)
                    if column.type == FILE and row[3 + index] is not None}
        infos: Dict[int, dict] = {}
        ordered = sorted(file_ids)
        for start in range(0, len(ordered), MAX_IN_VALUES):
            chunk = ordered[start:start + MAX_IN_VALUES]
            for item in conn.execute("SELECT id, name, mime, size FROM _files WHERE id IN (%s)"
                                     % ",".join("?" * len(chunk)), chunk):
                infos[item[0]] = file_info(item)
        out = []
        for row in rows:
            record: Dict[str, Any] = {"_id": row[0], "_created_at": row[1],
                                      "_updated_at": row[2]}
            for index, column in enumerate(columns):
                value = row[3 + index]
                if value is None:
                    continue
                record[column.name] = (infos.get(value, {"file_id": "f_%d" % value,
                                                         "missing": True})
                                       if column.type == FILE else value)
            out.append(record)
        return out

    @staticmethod
    def _select(tid: int, columns: List[Column]) -> str:
        physical = "".join(', "%s"' % column.phys for column in columns)
        return 'SELECT _id, _created_at, _updated_at%s FROM "t_%d"' % (physical, tid)

    # ── public operations ────────────────────────────────────────────────────

    def create(self, db: str, table: str, columns: List[Tuple[str, str]],
               request: Optional[Tuple[str, str]] = None) -> dict:
        with self._write(db) as conn:
            replay = self._replay(conn, request)
            if replay:
                return replay
            _, created = self._create_in(conn, table, columns)
            result = {"ok": True, "db": db, "table": table,
                      "columns": [column.describe() for column in created]}
            self._remember(conn, request, result)
            return result

    def insert(self, db: str, table: str, records: List[Dict[str, Any]],
               create_columns: Optional[List[Tuple[str, str]]] = None,
               request: Optional[Tuple[str, str]] = None,
               make_reply: Optional[Callable[[List[dict]], dict]] = None,
               row_numbers: Optional[List[int]] = None, row_word: str = "条") -> dict:
        """Insert all records or none. Values: str/number/None, or FileValue."""
        extra = sum(len(value.data) for record in records for value in record.values()
                    if isinstance(value, FileValue))
        self._check_size(db, extra)
        with self._write(db) as conn:
            replay = self._replay(conn, request)
            if replay:
                return replay
            if create_columns is not None:
                tid, columns = self._create_in(conn, table, create_columns)
            else:
                tid, columns = self._meta(conn, table)
            existing = conn.execute('SELECT COUNT(*) FROM "t_%d"' % tid).fetchone()[0]
            if existing + len(records) > self.limits.max_rows:
                raise TableDbError("too_large", "表「%s」已有 %d 行，再加 %d 行会超过上限 %d"
                                   % (table, existing, len(records), self.limits.max_rows))
            stamp = now_text()
            stored: List[dict] = []
            for index, record in enumerate(records):
                number = row_numbers[index] if row_numbers else index + 1
                label = "第 %d %s" % (number, row_word)
                if not isinstance(record, dict) or not record:
                    raise bad_request("%s记录要是非空对象，收到 %r" % (label, record),
                                      row=number)
                names, values = [], []
                shown: Dict[str, Any] = {}
                for key, value in record.items():
                    column = self._column(columns, key)
                    if column.type == FILE:
                        if value is None:
                            continue
                        if not isinstance(value, FileValue):
                            raise bad_request(
                                "%s「%s」是文件列，只能通过 import 或 attach_file 写入文件"
                                % (label, key), row=number, column=key)
                        info = self._put_file(conn, value, key, number)
                        names.append(column.phys)
                        values.append(info.pop("_fid"))
                        shown[key] = info
                        continue
                    if isinstance(value, FileValue):
                        raise bad_request("%s「%s」是文字列，不能放文件" % (label, key),
                                          row=number, column=key)
                    try:
                        text = text_value(value, key, self.limits, label)
                    except TableDbError as error:
                        error.row = number
                        raise
                    if text is not None:
                        names.append(column.phys)
                        values.append(text)
                        shown[key] = text
                placeholders = "".join(", ?" for _ in names)
                physical = "".join(', "%s"' % name for name in names)
                rid = conn.execute('INSERT INTO "t_%d"(_created_at, _updated_at%s) '
                                   'VALUES(?, ?%s)' % (tid, physical, placeholders),
                                   [stamp, stamp] + values).lastrowid
                stored.append(dict({"_id": rid, "_created_at": stamp, "_updated_at": stamp},
                                   **shown))
            result = (make_reply(stored) if make_reply else
                      {"ok": True, "db": db, "table": table, "inserted": len(stored),
                       "ids": [item["_id"] for item in stored]})
            self._remember(conn, request, result)
            return result

    def query(self, db: str, table: str, where: Any = None, offset: int = 0,
              limit: int = 20) -> Tuple[List[Column], int, List[dict]]:
        with self._read(db) as conn:
            tid, columns = self._meta(conn, table)
            clause, params = self._where(columns, where)
            base = ' WHERE _deleted_at IS NULL' + clause
            total = conn.execute('SELECT COUNT(*) FROM "t_%d"%s' % (tid, base),
                                 params).fetchone()[0]
            rows = conn.execute(self._select(tid, columns) + base +
                                " ORDER BY _id LIMIT ? OFFSET ?",
                                params + [limit, offset]).fetchall()
            return columns, total, self._records(conn, columns, rows)

    def _live_row(self, conn: sqlite3.Connection, tid: int, table: str, rid: Any) -> int:
        try:
            if isinstance(rid, bool):
                raise ValueError
            number = int(rid)
        except (TypeError, ValueError):
            raise bad_request("_id 要是整数（来自 query / append 的返回），收到 %r" % (rid,))
        found = conn.execute('SELECT _deleted_at FROM "t_%d" WHERE _id=?' % tid,
                             (number,)).fetchone()
        if found is None:
            raise TableDbError("not_found", "表「%s」里没有 _id=%d 的记录" % (table, number))
        if found[0] is not None:
            raise TableDbError("not_found", "_id=%d 的记录已在 %s 删除" % (number, found[0]))
        return number

    def update(self, db: str, table: str, rid: Any, changes: Any) -> dict:
        if not isinstance(changes, dict) or not changes:
            raise bad_request('set 要是非空对象，如 {"状态":"已归还"}；值为 null 表示清空')
        with self._write(db) as conn:
            tid, columns = self._meta(conn, table)
            number = self._live_row(conn, tid, table, rid)
            assignments, values = [], []
            for key, value in changes.items():
                column = self._column(columns, key)
                if column.type == FILE:
                    if value is not None:
                        raise bad_request("「%s」是文件列：换文件用 attach_file，解除文件传 null"
                                          % key, column=key)
                    assignments.append('"%s" = NULL' % column.phys)
                    continue
                assignments.append('"%s" = ?' % column.phys)
                values.append(text_value(value, key, self.limits, "set"))
            stamp = now_text()
            conn.execute('UPDATE "t_%d" SET %s, _updated_at=? WHERE _id=?' % (
                tid, ", ".join(assignments)), values + [stamp, number])
            row = conn.execute(self._select(tid, columns) + " WHERE _id=?",
                               (number,)).fetchone()
            return self._records(conn, columns, [row])[0]

    def delete(self, db: str, table: str, rid: Any) -> int:
        with self._write(db) as conn:
            tid, _ = self._meta(conn, table)
            number = self._live_row(conn, tid, table, rid)
            conn.execute('UPDATE "t_%d" SET _deleted_at=? WHERE _id=?' % tid,
                         (now_text(), number))
            return number

    def stats(self, db: str, table: str, where: Any = None,
              group_by: Any = None) -> dict:
        with self._read(db) as conn:
            tid, columns = self._meta(conn, table)
            clause, params = self._where(columns, where)
            base = ' WHERE _deleted_at IS NULL' + clause
            total = conn.execute('SELECT COUNT(*) FROM "t_%d"%s' % (tid, base),
                                 params).fetchone()[0]
            result: Dict[str, Any] = {"ok": True, "db": db, "table": table, "total": total}
            if group_by in (None, ""):
                return result
            column = self._column(columns, group_by)
            if column.type == FILE:
                raise bad_request("「%s」是文件列，不能按它分组" % group_by, column=group_by)
            rows = conn.execute('SELECT "%s", COUNT(*) FROM "t_%d"%s GROUP BY 1 '
                                'ORDER BY 2 DESC, 1 LIMIT ?' % (column.phys, tid, base),
                                params + [MAX_GROUPS + 1]).fetchall()
            result["group_by"] = group_by
            result["groups"] = [{"value": value, "count": count}
                                for value, count in rows[:MAX_GROUPS]]
            if len(rows) > MAX_GROUPS:
                result["warnings"] = ["分组超过 %d 个，只返回数量最多的 %d 个"
                                      % (MAX_GROUPS, MAX_GROUPS)]
            return result

    def attach(self, db: str, table: str, rid: Any, column_name: Any,
               value: FileValue) -> Tuple[int, dict]:
        self._check_size(db, len(value.data))
        with self._write(db) as conn:
            tid, columns = self._meta(conn, table)
            column = self._column(columns, column_name)
            if column.type != FILE:
                raise bad_request("「%s」是文字列，不能挂文件；文件要挂在 file 类型的列上"
                                  % column_name, column=str(column_name))
            number = self._live_row(conn, tid, table, rid)
            info = self._put_file(conn, value, column.name, None)
            conn.execute('UPDATE "t_%d" SET "%s"=?, _updated_at=? WHERE _id=?' % (
                tid, column.phys), (info.pop("_fid"), now_text(), number))
            return number, info
