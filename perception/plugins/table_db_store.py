#!/usr/bin/env python3
"""
plugins/table_db_store.py — the SQLite storage layer behind the `table_db` card.

Single SQLite file (`table_db.db`) under `data_dir`; tables are the top-level
unit. Standard library only.

Layout of `table_db.db`
-----------------------

    _tables    (id, name, created_at)                 user table names
    _columns   (table_id, pos, name, type, phys)      user column names & SQLite types
    _files     (id, sha256, name, mime, size, ...)    one row per stored BLOB cell
    _blobs     (sha256, data)                         BLOB bytes, deduplicated
    t_<id>     (rowid, c_1, c_2, ...)                 physical user table

Column types follow SQLite's standard storage classes: `TEXT`, `INTEGER`,
`REAL`, `BLOB`. Every table and column name from callers stays data in
`_tables` / `_columns`; physical SQL identifiers (`t_<id>`, `c_<pos>`) are
generated internally and all values use parameter binding.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

TEXT = "TEXT"
INTEGER = "INTEGER"
REAL = "REAL"
BLOB = "BLOB"
COLUMN_TYPES = (TEXT, INTEGER, REAL, BLOB)

_RESERVED_COLS = frozenset(("rowid", "_rowid_", "oid"))
MAX_NAME_CHARS = 64
MAX_IN_VALUES = 500
MAX_GROUPS = 200
DB_FILENAME = "table_db.db"

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
"""


class TableDbError(Exception):
    """A structured error returned to callers without raising."""

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
        out["written"] = 0
        return out


def bad_request(detail: str, **extra: Any) -> TableDbError:
    return TableDbError("bad_request", detail, **extra)


@dataclass
class FileValue:
    """Binary content read from a file on its way into a BLOB column."""

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


def check_name(value: Any, what: str, *, allow_rowid: bool = False) -> str:
    """Validate a user-facing table or column name (1-64 chars, no leading `_`)."""
    if not isinstance(value, str):
        raise bad_request("%s 必须为字符串，收到 %r" % (what, value))
    name = value.strip()
    if not name:
        raise bad_request("%s 不能为空" % what)
    if len(name) > MAX_NAME_CHARS:
        raise bad_request("%s「%s…」超过 %d 个字符" % (what, name[:16], MAX_NAME_CHARS))
    if name.startswith("_"):
        raise bad_request("%s「%s」不能以 _ 开头（_ 保留给系统内部表）" % (what, name))
    if not allow_rowid and name.lower() in _RESERVED_COLS:
        raise bad_request("%s「%s」为 SQLite 保留列名" % (what, name))
    if any(ord(char) < 32 for char in name):
        raise bad_request("%s「%s」含有控制字符" % (what, name))
    return name


def check_columns(columns: Any, limits: StoreLimits) -> List[Tuple[str, str]]:
    """Validate `[{"name": .., "type": "TEXT"|"INTEGER"|"REAL"|"BLOB"}]`."""
    if not isinstance(columns, list) or not columns:
        raise bad_request("columns 必须为非空数组，格式 [{'name':'列名','type':'TEXT'}]")
    if len(columns) > limits.max_columns:
        raise TableDbError("too_large", "最多 %d 列，收到 %d 列" % (limits.max_columns, len(columns)))
    out: List[Tuple[str, str]] = []
    seen = set()
    for item in columns:
        if isinstance(item, str):
            name, kind = item, TEXT
        elif isinstance(item, dict):
            name = item.get("name")
            kind = str(item.get("type") or TEXT).strip().upper()
        else:
            raise bad_request("columns 的每一项必须为 {'name':'列名','type':'类型'}，收到 %r" % (item,))
        name = check_name(name, "列名")
        if kind not in COLUMN_TYPES:
            raise bad_request("列「%s」的类型 %r 无效；SQLite 列类型仅支持：%s"
                              % (name, kind, " / ".join(COLUMN_TYPES)), column=name)
        if name in seen:
            raise bad_request("列名「%s」重复" % name, column=name)
        seen.add(name)
        out.append((name, kind))
    return out


def coerce_value(value: Any, col_type: str, column: str, limits: StoreLimits,
                 where: str) -> Any:
    """Coerce a scalar cell value according to its SQLite column type."""
    if value is None:
        return None
    if col_type == TEXT:
        if isinstance(value, bool):
            raise bad_request("%s「%s」为 TEXT 列，不支持布尔值 %r，请传字符串"
                              % (where, column, value), column=column)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise bad_request("%s「%s」包含无效浮点数" % (where, column), column=column)
            value = int(value) if value.is_integer() else repr(value)
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, str):
            raise bad_request("%s「%s」为 TEXT 列，值必须为字符串或数字，收到 %s"
                              % (where, column, type(value).__name__), column=column)
        if value == "":
            return None
        if len(value) > limits.max_cell_chars:
            raise TableDbError("too_large", "%s「%s」长度为 %d 字符，超过上限 %d"
                               % (where, column, len(value), limits.max_cell_chars), column=column)
        return value
    if col_type == INTEGER:
        if isinstance(value, bool):
            raise bad_request("%s「%s」为 INTEGER 列，不支持布尔值 %r"
                              % (where, column, value), column=column)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if math.isfinite(value) and value.is_integer():
                return int(value)
            raise bad_request("%s「%s」为 INTEGER 列，需要整数，收到 %r"
                              % (where, column, value), column=column)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                try:
                    num = float(stripped)
                    if math.isfinite(num) and num.is_integer():
                        return int(num)
                except ValueError:
                    pass
                raise bad_request("%s「%s」为 INTEGER 列，无法解析为整数：%r"
                                  % (where, column, value), column=column)
        raise bad_request("%s「%s」为 INTEGER 列，需要整数，收到 %s"
                          % (where, column, type(value).__name__), column=column)
    if col_type == REAL:
        if isinstance(value, bool):
            raise bad_request("%s「%s」为 REAL 列，不支持布尔值 %r"
                              % (where, column, value), column=column)
        if isinstance(value, (int, float)):
            num = float(value)
            if not math.isfinite(num):
                raise bad_request("%s「%s」包含无效浮点数" % (where, column), column=column)
            return num
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                num = float(stripped)
            except ValueError:
                raise bad_request("%s「%s」为 REAL 列，无法解析为实数：%r"
                                  % (where, column, value), column=column)
            if not math.isfinite(num):
                raise bad_request("%s「%s」包含无效浮点数" % (where, column), column=column)
            return num
        raise bad_request("%s「%s」为 REAL 列，需要数值，收到 %s"
                          % (where, column, type(value).__name__), column=column)
    raise bad_request("未知列类型 %r" % (col_type,), column=column)


def _parse_rowid(value: Any, what: str = "rowid") -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        num = int(value)
        if num <= 0:
            raise ValueError
        return num
    except (TypeError, ValueError):
        raise bad_request("%s 必须为正整数，收到 %r" % (what, value), column="rowid")


class TableStore:
    """Single-database SQLite store. Every mutation runs in one serialized transaction."""

    def __init__(self, data_dir: str, limits: Optional[StoreLimits] = None):
        self.data_dir = data_dir
        self.limits = limits or StoreLimits()
        self._lock = threading.Lock()

    def db_path(self) -> str:
        return os.path.join(self.data_dir, DB_FILENAME)

    def _connect(self, create: bool) -> Optional[sqlite3.Connection]:
        path = self.db_path()
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

    @contextmanager
    def _read(self):
        conn = self._connect(create=False)
        try:
            yield conn
        finally:
            if conn is not None:
                conn.close()

    @contextmanager
    def _write(self):
        with self._lock:
            conn = self._connect(create=True)
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

    def _check_size(self, conn: sqlite3.Connection, extra: int) -> None:
        path = self.db_path()
        size = sum(os.path.getsize(part) for part in (path, path + "-wal", path + "-shm")
                   if os.path.exists(part))
        # Subtract SQLite freelist pages that subsequent writes can reuse in place.
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
        effective = max(0, size - page_size * freelist)
        projected = extra + max(4096, extra // 10) if extra > 0 else 0
        if effective + projected > self.limits.max_db_bytes:
            raise TableDbError("too_large", "数据库有效占用约 %d 字节，再写入约 %d 字节会超过上限 %d"
                               % (effective, projected, self.limits.max_db_bytes))

    def db_bytes(self) -> int:
        path = self.db_path()
        return sum(os.path.getsize(part) for part in (path, path + "-wal", path + "-shm")
                   if os.path.exists(part))

    def quick_check(self) -> str:
        with self._read() as conn:
            if conn is None:
                return "empty"
            return str(conn.execute("PRAGMA quick_check").fetchone()[0])

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
            raise TableDbError("unknown_table", "表「%s」不存在。现有表：%s" % (
                table, "、".join(names) or "（无）"), tables=names)
        columns = [Column(*item) for item in conn.execute(
            "SELECT name, type, phys FROM _columns WHERE table_id=? ORDER BY pos", (row[0],))]
        return row[0], columns

    @staticmethod
    def _column(columns: List[Column], name: Any) -> Column:
        for column in columns:
            if column.name == name:
                return column
        raise TableDbError("unknown_column", "列「%s」不存在。现有列：%s" % (
            name, "、".join(column.name for column in columns)),
            column=str(name), columns=[column.describe() for column in columns])

    def columns(self, table: str) -> Optional[List[Column]]:
        with self._read() as conn:
            try:
                return self._meta(conn, table)[1]
            except TableDbError:
                return None

    def tables(self) -> List[dict]:
        with self._read() as conn:
            if conn is None:
                return []
            out = []
            for tid, name in conn.execute("SELECT id, name FROM _tables ORDER BY id").fetchall():
                cols = self._meta(conn, name)[1]
                rows = conn.execute('SELECT COUNT(*) FROM "t_%d"' % tid).fetchone()[0]
                out.append({"table": name, "rows": rows,
                            "columns": [column.describe() for column in cols]})
            return out

    # ── BLOB storage & garbage collection ────────────────────────────────────

    def _put_file(self, conn: sqlite3.Connection, value: FileValue, column: str,
                  row: Optional[int]) -> int:
        if len(value.data) > self.limits.max_file_bytes:
            raise TableDbError("too_large", "列「%s」的文件大小 %d 字节超过单文件上限 %d 字节"
                               % (column, len(value.data), self.limits.max_file_bytes),
                               row=row, column=column)
        digest = hashlib.sha256(value.data).hexdigest()
        conn.execute("INSERT OR IGNORE INTO _blobs(sha256, data) VALUES(?, ?)",
                     (digest, sqlite3.Binary(value.data)))
        fid = conn.execute("INSERT INTO _files(sha256, name, mime, size, created_at) "
                           "VALUES(?, ?, ?, ?, ?)",
                           (digest, value.name, value.mime, len(value.data), now_text())).lastrowid
        return int(fid)

    @staticmethod
    def _gc_files(conn: sqlite3.Connection, file_ids: Sequence[int]) -> None:
        """Delete orphaned `_files` rows and any unreferenced `_blobs`."""
        ids = [int(fid) for fid in file_ids if fid is not None]
        if not ids:
            return
        for start in range(0, len(ids), MAX_IN_VALUES):
            chunk = ids[start:start + MAX_IN_VALUES]
            conn.execute("DELETE FROM _files WHERE id IN (%s)" % ",".join("?" * len(chunk)), chunk)
        conn.execute("DELETE FROM _blobs WHERE sha256 NOT IN (SELECT sha256 FROM _files)")

    def read_files(self, file_ids: Sequence[int]) -> Dict[int, Tuple[str, str, bytes]]:
        wanted = sorted({int(fid) for fid in file_ids if fid is not None})
        if not wanted:
            return {}
        out: Dict[int, Tuple[str, str, bytes]] = {}
        with self._read() as conn:
            if conn is None:
                return out
            for start in range(0, len(wanted), MAX_IN_VALUES):
                chunk = wanted[start:start + MAX_IN_VALUES]
                sql = ("SELECT f.id, f.name, f.mime, b.data FROM _files f "
                       "JOIN _blobs b ON f.sha256 = b.sha256 WHERE f.id IN (%s)"
                       % ",".join("?" * len(chunk)))
                for fid, name, mime, blob in conn.execute(sql, chunk):
                    out[fid] = (name, mime, bytes(blob))
        return out

    def read_cell_blob(self, table: str, rowid: int, column_name: str
                       ) -> Optional[Tuple[str, str, bytes]]:
        """Read one BLOB cell `(table, rowid, column_name)` directly from SQLite."""
        with self._read() as conn:
            tid, cols = self._meta(conn, table)
            col = self._column(cols, column_name)
            if col.type != BLOB:
                raise bad_request("列「%s」的类型为 %s，不是 BLOB 列" % (col.name, col.type),
                                  column=col.name)
            row = conn.execute('SELECT "%s" FROM "t_%d" WHERE rowid=?' % (col.phys, tid),
                               (rowid,)).fetchone()
            if row is None or row[0] is None:
                return None
            fid = int(row[0])
            found = conn.execute(
                "SELECT f.name, f.mime, b.data FROM _files f "
                "JOIN _blobs b ON f.sha256 = b.sha256 WHERE f.id=?",
                (fid,),
            ).fetchone()
            if found is None:
                return None
            return str(found[0]), str(found[1]), bytes(found[2])

    # ── WHERE & ORDER BY ─────────────────────────────────────────────────────

    def _where(self, columns: List[Column], where: Any) -> Tuple[str, List[Any]]:
        if where is None or where == {}:
            return "", []
        if not isinstance(where, dict):
            raise bad_request("where 必须为对象，格式 {'列名':'值'}", column="where")
        parts: List[str] = []
        params: List[Any] = []
        for key, value in where.items():
            if key == "rowid":
                phys, col = "rowid", None
            else:
                col = self._column(columns, key)
                if col.type == BLOB and value is not None:
                    raise bad_request("BLOB 列「%s」在 where 中仅支持 null（判空）" % key,
                                      column=key)
                phys = '"%s"' % col.phys
            if value is None:
                parts.append("%s IS NULL" % phys)
            elif isinstance(value, list):
                if not value:
                    parts.append("0")
                    continue
                if len(value) > MAX_IN_VALUES:
                    raise TableDbError("too_large", "where「%s」数组最多 %d 项" % (key, MAX_IN_VALUES),
                                       column=key)
                coerced = [self._where_scalar(col, key, item) for item in value]
                non_null = [item for item in coerced if item is not None]
                clauses = []
                if non_null:
                    clauses.append("%s IN (%s)" % (phys, ",".join("?" * len(non_null))))
                    params.extend(non_null)
                if len(non_null) < len(coerced):
                    clauses.append("%s IS NULL" % phys)
                parts.append("(%s)" % " OR ".join(clauses))
            else:
                coerced = self._where_scalar(col, key, value)
                if coerced is None:
                    parts.append("%s IS NULL" % phys)
                else:
                    parts.append("%s = ?" % phys)
                    params.append(coerced)
        return (" WHERE " + " AND ".join(parts)) if parts else "", params

    def _where_scalar(self, column: Optional[Column], key: str, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            raise bad_request("where「%s」只支持单值、数组或 null，收到 %s"
                              % (key, json.dumps(value, ensure_ascii=False)), column=key)
        if column is None:
            return _parse_rowid(value, "where.rowid")
        return coerce_value(value, column.type, key, self.limits, "where")

    def _order_by(self, columns: List[Column], order_by: Any) -> str:
        if order_by in (None, ""):
            return " ORDER BY rowid ASC"
        if isinstance(order_by, str):
            raw_terms = [part.strip() for part in order_by.split(",") if part.strip()]
        elif isinstance(order_by, list):
            raw_terms = [str(part).strip() for part in order_by if str(part).strip()]
        else:
            raise bad_request("order_by 格式为 '列名' 或 '列名 DESC'", column="order_by")
        if not raw_terms:
            return " ORDER BY rowid ASC"
        sql_terms = []
        for term in raw_terms:
            tokens = term.split()
            if len(tokens) == 1:
                col_name, direction = tokens[0], "ASC"
            elif len(tokens) == 2 and tokens[1].upper() in ("ASC", "DESC"):
                col_name, direction = tokens[0], tokens[1].upper()
            else:
                raise bad_request("order_by 项 %r 无效，格式为 '列名' 或 '列名 ASC/DESC'" % term,
                                  column="order_by")
            if col_name.lower() == "rowid":
                sql_terms.append("rowid %s" % direction)
            else:
                col = self._column(columns, col_name)
                if col.type == BLOB:
                    raise bad_request("不能按 BLOB 列「%s」排序" % col.name, column="order_by")
                sql_terms.append('"%s" %s' % (col.phys, direction))
        return " ORDER BY " + ", ".join(sql_terms)

    # ── row formatting ───────────────────────────────────────────────────────

    def _format_rows(self, conn: sqlite3.Connection, columns: List[Column],
                     rows: List[Sequence[Any]]) -> List[dict]:
        file_ids = {row[1 + index] for row in rows
                    for index, column in enumerate(columns)
                    if column.type == BLOB and row[1 + index] is not None}
        infos: Dict[int, dict] = {}
        ordered = sorted(file_ids)
        for start in range(0, len(ordered), MAX_IN_VALUES):
            chunk = ordered[start:start + MAX_IN_VALUES]
            for fid, name, mime, size in conn.execute(
                    "SELECT id, name, mime, size FROM _files WHERE id IN (%s)"
                    % ",".join("?" * len(chunk)), chunk):
                infos[fid] = {"type": BLOB, "_fid": fid, "name": name, "mime": mime, "size": size}
        out = []
        for row in rows:
            record: Dict[str, Any] = {"rowid": row[0]}
            for index, column in enumerate(columns):
                value = row[1 + index]
                if value is None:
                    record[column.name] = None
                elif column.type == BLOB:
                    info = infos.get(value)
                    record[column.name] = dict(info) if info else None
                else:
                    record[column.name] = value
            out.append(record)
        return out

    @staticmethod
    def _select_sql(tid: int, columns: List[Column]) -> str:
        physical = "".join(', "%s"' % column.phys for column in columns)
        return 'SELECT rowid%s FROM "t_%d"' % (physical, tid)

    # ── DDL: CREATE TABLE / DROP TABLE ───────────────────────────────────────

    def _create_in(self, conn: sqlite3.Connection, table: str,
                   columns: List[Tuple[str, str]]) -> Tuple[int, List[Column]]:
        if conn.execute("SELECT 1 FROM _tables WHERE name=?", (table,)).fetchone():
            raise TableDbError("table_exists", "表「%s」已存在" % table)
        count = conn.execute("SELECT COUNT(*) FROM _tables").fetchone()[0]
        if count >= self.limits.max_tables:
            raise TableDbError("too_large", "表数量已达上限 %d" % self.limits.max_tables)
        tid = conn.execute("INSERT INTO _tables(name, created_at) VALUES(?, ?)",
                           (table, now_text())).lastrowid
        created: List[Column] = []
        col_defs: List[str] = []
        for pos, (name, kind) in enumerate(columns, 1):
            phys = "c_%d" % pos
            conn.execute("INSERT INTO _columns(table_id, pos, name, type, phys) "
                         "VALUES(?, ?, ?, ?, ?)", (tid, pos, name, kind, phys))
            sql_type = INTEGER if kind == BLOB else kind
            col_defs.append('"%s" %s' % (phys, sql_type))
            created.append(Column(name=name, type=kind, phys=phys))
        conn.execute('CREATE TABLE "t_%d" (rowid INTEGER PRIMARY KEY AUTOINCREMENT, %s)'
                     % (tid, ", ".join(col_defs)))
        return int(tid), created

    def create_table(self, table: str, columns: List[Tuple[str, str]]) -> dict:
        with self._write() as conn:
            _, created = self._create_in(conn, table, columns)
            return {"ok": True, "table": table,
                    "columns": [column.describe() for column in created]}

    def drop_table(self, table: str) -> dict:
        with self._write() as conn:
            tid, cols = self._meta(conn, table)
            rows = conn.execute('SELECT COUNT(*) FROM "t_%d"' % tid).fetchone()[0]
            file_ids: List[int] = []
            for col in cols:
                if col.type == BLOB:
                    file_ids.extend(
                        row[0] for row in conn.execute(
                            'SELECT "%s" FROM "t_%d" WHERE "%s" IS NOT NULL'
                            % (col.phys, tid, col.phys)
                        )
                    )
            conn.execute('DROP TABLE "t_%d"' % tid)
            conn.execute("DELETE FROM _columns WHERE table_id=?", (tid,))
            conn.execute("DELETE FROM _tables WHERE id=?", (tid,))
            self._gc_files(conn, file_ids)
            return {"ok": True, "table": table, "dropped": True, "rows": rows}

    # ── DML: INSERT / SELECT / UPDATE / DELETE ───────────────────────────────

    def insert(self, table: str, records: List[Dict[str, Any]],
               create_columns: Optional[List[Tuple[str, str]]] = None,
               make_reply: Optional[Callable[[List[dict]], dict]] = None,
               row_numbers: Optional[List[int]] = None, row_word: str = "条") -> dict:
        """Insert all rows or none in one SQLite transaction."""
        extra = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            for value in record.values():
                if isinstance(value, FileValue):
                    extra += len(value.data)
                elif isinstance(value, str):
                    extra += len(value.encode("utf-8"))
        with self._write() as conn:
            self._check_size(conn, extra)
            if create_columns is not None:
                tid, cols = self._create_in(conn, table, create_columns)
            else:
                tid, cols = self._meta(conn, table)
            existing = conn.execute('SELECT COUNT(*) FROM "t_%d"' % tid).fetchone()[0]
            if existing + len(records) > self.limits.max_rows:
                raise TableDbError("too_large", "表「%s」已有 %d 行，再写入 %d 行会超过上限 %d"
                                   % (table, existing, len(records), self.limits.max_rows))
            stored: List[dict] = []
            for index, record in enumerate(records):
                number = row_numbers[index] if row_numbers else index + 1
                label = "第 %d %s" % (number, row_word)
                if not isinstance(record, dict) or not record:
                    raise bad_request("%s必须为非空对象 {'列名':'值'}，收到 %r" % (label, record),
                                      row=number)
                names, values = [], []
                shown: Dict[str, Any] = {}
                for key, value in record.items():
                    col = self._column(cols, key)
                    if col.type == BLOB:
                        if value is None:
                            continue
                        if not isinstance(value, FileValue):
                            raise bad_request("%s「%s」为 BLOB 列，请传入文件路径"
                                              % (label, key), row=number, column=key)
                        fid = self._put_file(conn, value, key, number)
                        names.append(col.phys)
                        values.append(fid)
                        shown[key] = {"type": BLOB, "_fid": fid, "name": value.name,
                                      "mime": value.mime, "size": len(value.data)}
                        continue
                    if isinstance(value, FileValue):
                        raise bad_request("%s「%s」为 %s 列，不能写入文件"
                                          % (label, key, col.type), row=number, column=key)
                    try:
                        coerced = coerce_value(value, col.type, key, self.limits, label)
                    except TableDbError as error:
                        error.row = number
                        raise
                    if coerced is not None:
                        names.append(col.phys)
                        values.append(coerced)
                        shown[key] = coerced
                if names:
                    physical = ", ".join('"%s"' % n for n in names)
                    placeholders = ", ".join("?" * len(names))
                    rid = conn.execute('INSERT INTO "t_%d" (%s) VALUES (%s)'
                                       % (tid, physical, placeholders), values).lastrowid
                else:
                    rid = conn.execute('INSERT INTO "t_%d" DEFAULT VALUES' % tid).lastrowid
                stored.append(dict({"rowid": int(rid)}, **shown))
            if make_reply:
                return make_reply(stored)
            return {"ok": True, "table": table, "inserted": len(stored),
                    "rowids": [item["rowid"] for item in stored]}

    def select(self, table: str, columns: Optional[List[str]] = None,
               where: Any = None, group_by: Optional[str] = None,
               order_by: Any = None, offset: int = 0, limit: int = 20
               ) -> Tuple[List[Column], int, List[dict], bool]:
        """Run SELECT. Returns `(selected_cols, total, rows, is_aggregate)`."""
        with self._read() as conn:
            tid, all_cols = self._meta(conn, table)
            clause, params = self._where(all_cols, where)
            total = conn.execute('SELECT COUNT(*) FROM "t_%d"%s' % (tid, clause),
                                 params).fetchone()[0]

            # GROUP BY aggregation: SELECT <group_by>, COUNT(*) ... GROUP BY <group_by>
            if group_by not in (None, ""):
                gcol = self._column(all_cols, group_by)
                if gcol.type == BLOB:
                    raise bad_request("不能按 BLOB 列「%s」分组" % gcol.name, column=gcol.name)
                rows = conn.execute(
                    'SELECT "%s", COUNT(*) FROM "t_%d"%s GROUP BY 1 '
                    'ORDER BY 2 DESC, 1 ASC LIMIT ? OFFSET ?' % (gcol.phys, tid, clause),
                    params + [min(limit, MAX_GROUPS), offset],
                ).fetchall()
                result_rows = [{gcol.name: val, "COUNT(*)": cnt} for val, cnt in rows]
                return [gcol], total, result_rows, True

            # SELECT COUNT(*)
            if columns and len(columns) == 1 and columns[0].strip().upper() == "COUNT(*)":
                return [], total, [{"COUNT(*)": total}], True

            # Column projection
            if columns and not (len(columns) == 1 and columns[0].strip() == "*"):
                selected: List[Column] = []
                seen = set()
                for name in columns:
                    cleaned = name.strip()
                    if cleaned.lower() == "rowid":
                        continue
                    col = self._column(all_cols, cleaned)
                    if col.name not in seen:
                        seen.add(col.name)
                        selected.append(col)
            else:
                selected = all_cols

            order_sql = self._order_by(all_cols, order_by)
            rows = conn.execute(
                self._select_sql(tid, selected) + clause + order_sql + " LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            return selected, total, self._format_rows(conn, selected, rows), False

    def update(self, table: str, changes: Dict[str, Any], where: Any) -> dict:
        """SQLite UPDATE ... SET ... WHERE ... with BLOB replacement and GC."""
        if not isinstance(changes, dict) or not changes:
            raise bad_request("set 必须为非空对象，格式 {'列名':'新值'}")
        if not isinstance(where, dict) or not where:
            raise bad_request("update 必须指定非空 where 条件（例如 {'rowid':1} 或 {'列名':'值'}）",
                              column="where")
        extra = sum(
            len(v.data) if isinstance(v, FileValue)
            else (len(v.encode("utf-8")) if isinstance(v, str) else 0)
            for v in changes.values()
        )
        with self._write() as conn:
            self._check_size(conn, extra)
            tid, cols = self._meta(conn, table)
            clause, wparams = self._where(cols, where)
            matched = [row[0] for row in conn.execute(
                'SELECT rowid FROM "t_%d"%s ORDER BY rowid' % (tid, clause), wparams
            ).fetchall()]
            if not matched:
                return {"ok": True, "table": table, "updated": 0, "rowids": [], "rows": []}

            # Collect old BLOB file_ids on columns being replaced/cleared so we can GC them.
            replaced_blob_cols = [self._column(cols, k) for k in changes
                                  if self._column(cols, k).type == BLOB]
            old_fids: List[int] = []
            for bcol in replaced_blob_cols:
                for start in range(0, len(matched), MAX_IN_VALUES):
                    chunk = matched[start:start + MAX_IN_VALUES]
                    old_fids.extend(
                        row[0] for row in conn.execute(
                            'SELECT "%s" FROM "t_%d" WHERE rowid IN (%s) AND "%s" IS NOT NULL'
                            % (bcol.phys, tid, ",".join("?" * len(chunk)), bcol.phys), chunk
                        )
                    )

            # Validate scalar columns first (same value for all matched rows);
            # BLOB columns get a dedicated `_files` row per updated row so GC is exact.
            scalar_assignments: List[str] = []
            scalar_values: List[Any] = []
            blob_changes: List[Tuple[Column, Optional[FileValue]]] = []
            for key, value in changes.items():
                col = self._column(cols, key)
                if col.type == BLOB:
                    if value is not None and not isinstance(value, FileValue):
                        raise bad_request("列「%s」为 BLOB 列，请传入文件路径或 null" % key,
                                          column=key)
                    blob_changes.append((col, value))
                    continue
                if isinstance(value, FileValue):
                    raise bad_request("列「%s」为 %s 列，不能写入文件" % (key, col.type),
                                      column=key)
                coerced = coerce_value(value, col.type, key, self.limits, "set")
                scalar_assignments.append('"%s" = ?' % col.phys)
                scalar_values.append(coerced)

            for rid in matched:
                assignments = list(scalar_assignments)
                values = list(scalar_values)
                for bcol, fval in blob_changes:
                    if fval is None:
                        assignments.append('"%s" = NULL' % bcol.phys)
                    else:
                        fid = self._put_file(conn, fval, bcol.name, None)
                        assignments.append('"%s" = ?' % bcol.phys)
                        values.append(fid)
                conn.execute('UPDATE "t_%d" SET %s WHERE rowid = ?'
                             % (tid, ", ".join(assignments)), values + [rid])

            self._gc_files(conn, old_fids)

            updated_rows: List[Sequence[Any]] = []
            for start in range(0, len(matched), MAX_IN_VALUES):
                chunk = matched[start:start + MAX_IN_VALUES]
                updated_rows.extend(conn.execute(
                    self._select_sql(tid, cols) + " WHERE rowid IN (%s) ORDER BY rowid"
                    % ",".join("?" * len(chunk)), chunk
                ).fetchall())
            return {"ok": True, "table": table, "updated": len(matched),
                    "rowids": matched, "rows": self._format_rows(conn, cols, updated_rows)}

    def delete(self, table: str, where: Any) -> dict:
        """SQLite DELETE FROM ... WHERE ... with immediate BLOB GC."""
        if not isinstance(where, dict) or not where:
            raise bad_request("delete 必须指定非空 where 条件（例如 {'rowid':1} 或 {'列名':'值'}）",
                              column="where")
        with self._write() as conn:
            tid, cols = self._meta(conn, table)
            clause, wparams = self._where(cols, where)
            matched = [row[0] for row in conn.execute(
                'SELECT rowid FROM "t_%d"%s ORDER BY rowid' % (tid, clause), wparams
            ).fetchall()]
            if not matched:
                return {"ok": True, "table": table, "deleted": 0, "rowids": []}

            old_fids: List[int] = []
            for bcol in (c for c in cols if c.type == BLOB):
                for start in range(0, len(matched), MAX_IN_VALUES):
                    chunk = matched[start:start + MAX_IN_VALUES]
                    old_fids.extend(
                        row[0] for row in conn.execute(
                            'SELECT "%s" FROM "t_%d" WHERE rowid IN (%s) AND "%s" IS NOT NULL'
                            % (bcol.phys, tid, ",".join("?" * len(chunk)), bcol.phys), chunk
                        )
                    )
            for start in range(0, len(matched), MAX_IN_VALUES):
                chunk = matched[start:start + MAX_IN_VALUES]
                conn.execute('DELETE FROM "t_%d" WHERE rowid IN (%s)'
                             % (tid, ",".join("?" * len(chunk)),), chunk)
            self._gc_files(conn, old_fids)
            return {"ok": True, "table": table, "deleted": len(matched), "rowids": matched}
