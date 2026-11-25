#!/usr/bin/env python3
"""
plugins/table_db.py — SQLite table storage card (Perception processor).

Exposes standard SQLite table primitives over a single SQLite database
(`table_db.db`), with CSV / XLSX table import and export:

  tables        list tables and their SQLite column schemas (.tables / .schema)
  create_table  CREATE TABLE with SQLite types (TEXT / INTEGER / REAL / BLOB)
  drop_table    DROP TABLE
  insert        INSERT INTO (BLOB columns accept file paths, readfile semantics)
  select        SELECT ... WHERE ... GROUP BY ... ORDER BY ... LIMIT ... OFFSET
  update        UPDATE ... SET ... WHERE ... (supports BLOB file replacement / NULL)
  delete        DELETE FROM ... WHERE ...
  import        import a .csv or .xlsx file into a table
  export        export a table (or filtered query) to .csv or .xlsx

Standard library only (sqlite3, zipfile, csv, xml.etree).
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import mimetypes
import os
import posixpath
import re
import shutil
import socket
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

from plugins.image_input import BadInput, check_under_roots
from plugins.table_db_parse import (
    Limits,
    ParseError,
    SUPPORTED_SUFFIXES,
    column_letter,
    open_bounded_zip,
    parse_document,
    resolve_layout,
    write_csv,
    write_xlsx,
)
from plugins.table_db_store import (
    BLOB,
    COLUMN_TYPES,
    INTEGER,
    REAL,
    TEXT,
    FileValue,
    StoreLimits,
    TableDbError,
    TableStore,
    bad_request,
    check_columns,
    check_name,
)
from utils.log_sampling import escape_log_text

log = logging.getLogger(__name__)

PREFIX = "table_db"

DEFAULT_DATA_DIR = "/models/table_db"
DEFAULT_HTTP_BASE = "http://localhost:15720"
HTTP_BLOB_PREFIX = "/table_db/blob"
HTTP_EXPORT_PREFIX = "/table_db/export"
DEFAULT_ROOTS = ("/models", "/tmp", "/work")
PAGE_ROWS = 20
MAX_PAGE_ROWS = 200
MAX_INSERT = 500
CACHE_TTL_S = 24 * 3600
DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024

_IMPORT_SUFFIXES = (".xlsx", ".csv")
_TABLE_FILES = ".xlsx,.csv"
_EXPORT_FILENAME = re.compile(r"^export-[0-9a-f]{12}\.(csv|xlsx)$")
_IMAGE_MIMES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/webp": "webp",
}
_MIME_BY_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"GIF87a", "image/gif", ".gif"),
    (b"GIF89a", "image/gif", ".gif"),
    (b"BM", "image/bmp", ".bmp"),
    (b"%PDF", "application/pdf", ".pdf"),
    (b"ID3", "audio/mpeg", ".mp3"),
)

# Canvas schema rules:
# * Dropdown shows bare SQLite action names (`oneOf.const` + `description`, no `title`).
# * NO ASCII double quote (") in any property description (canvas.js `_esc` does
#   not escape quotes in placeholder="..." / title="...").
# * All placeholders use generic notation (`表名`, `列名`, `值`) without domain examples.

ACTIONS = [
    ("tables", "查看当前所有数据表及其列结构与行数（对应 SQLite .tables / .schema）"),
    ("create_table", "新建数据表并定义列名与 SQLite 类型（对应 CREATE TABLE）"),
    ("drop_table", "删除数据表及其全部数据（对应 DROP TABLE）"),
    ("insert", "向表中插入一条或多条记录，BLOB 列可传文件路径（对应 INSERT INTO）"),
    ("select", "查询记录、条件筛选、分页、排序与分组计数（对应 SELECT）"),
    ("update", "按条件修改记录，可更新或置空 BLOB 列文件（对应 UPDATE）"),
    ("delete", "按条件删除记录（对应 DELETE FROM）"),
    ("import", "导入 .csv 或 .xlsx 表格文件到数据表（对应 .import）"),
    ("export", "将数据表或查询结果导出为 .csv 或 .xlsx 文件"),
]

TOOLS: List[Dict[str, Any]] = [{
    "name": PREFIX,
    "type": "processor",
    "multiInstance": False,
    "description": (
        "SQLite 数据表存储卡片：提供标准 SQLite 表级操作（tables / create_table / drop_table / "
        "insert / select / update / delete）以及 .csv / .xlsx 数据表导入导出（import / export）。"
        "列类型遵循 SQLite 标准（TEXT / INTEGER / REAL / BLOB），文件以 BLOB 存储，"
        "insert/update 可通过文件路径写入 BLOB，select 返回 BLOB 下载链接。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "info"] + [name for name, _ in ACTIONS],
                "oneOf": [{"const": name, "description": text} for name, text in ACTIONS],
                "description": "要执行的 SQLite 操作，各操作说明见 oneOf.description",
            },
            "table": {
                "type": "string",
                "description": "表名，例如 table_name；查看现有表请先执行 tables",
            },
            "columns": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(COLUMN_TYPES)},
                }},
                "description": (
                    "列定义或查询列：create_table 填 列名1,列名2:INTEGER,列名3:BLOB"
                    "（类型支持 TEXT/INTEGER/REAL/BLOB，默认 TEXT）；"
                    "select/export 可填要返回的列名，如 列名1,列名2 或 COUNT(*)"
                ),
            },
            "values": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    "插入的行数据：[{'列名1':'值1','列名2':'值2'}] 或单行对象 {'列名':'值'}；"
                    "BLOB 列填文件路径或 '$file'（配合 file 参数）；一次最多 500 行"
                ),
            },
            "set": {
                "type": "object",
                "description": (
                    "要更新的列与新值：{'列名':'新值'}；置空写 {'列名':null}；"
                    "BLOB 列填文件路径或 '$file'（配合 file 参数）"
                ),
            },
            "where": {
                "type": "object",
                "description": (
                    "筛选条件：{'列名':'值'}；多值 {'列名':['值1','值2']}；"
                    "为空 {'列名':null}；按行号 {'rowid':1}；单双引号均可"
                ),
            },
            "group_by": {
                "type": "string",
                "description": "分组列名，格式 列名；配合 select 按该列分组并返回 COUNT(*)",
            },
            "order_by": {
                "type": "string",
                "description": "排序规则：列名 或 列名 DESC（支持 rowid），多列用逗号分隔",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "最多返回行数，整数；不填默认 20，最大 200",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "跳过前几行用于翻页，整数；不填默认 0",
            },
            "path": {
                "type": "string",
                "format": "file",
                "accept": _TABLE_FILES,
                "uploadTo": "mcp",
                "description": "要导入的表格文件（仅支持 .csv 和 .xlsx）",
            },
            "file": {
                "type": "string",
                "format": "file",
                "uploadTo": "mcp",
                "description": (
                    "要写入 BLOB 列的文件（配合 insert/update 使用；"
                    "在 values/set 中填 '$file'，或当表仅有唯一 BLOB 列时自动绑定）"
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["create", "append"],
                "description": "导入模式：create（建新表，默认）或 append（追加到已有表）",
            },
            "header_row": {
                "type": "integer",
                "minimum": 0,
                "description": "表头所在行号（从 1 开始）；不填自动识别；0 表示无表头",
            },
            "format": {
                "type": "string",
                "enum": ["xlsx", "csv"],
                "description": "导出格式：xlsx（默认，支持嵌入图片 BLOB）或 csv",
            },
        },
        "required": ["action"],
        "x-action-params": {
            "tables": {
                "params": [],
                "required": [],
                "description": "查看当前所有数据表、各表的列定义（TEXT/INTEGER/REAL/BLOB）和行数。",
            },
            "create_table": {
                "params": ["table", "columns"],
                "required": ["table", "columns"],
                "description": "新建数据表（CREATE TABLE）。列类型支持 TEXT / INTEGER / REAL / BLOB。",
            },
            "drop_table": {
                "params": ["table"],
                "required": ["table"],
                "description": "删除数据表及其全部数据和关联 BLOB（DROP TABLE）。",
            },
            "insert": {
                "params": ["table", "values", "file"],
                "required": ["table", "values"],
                "description": (
                    "插入一条或多条记录（INSERT INTO，整批事务提交或回滚），返回 rowids。"
                    "BLOB 列可直接填文件路径、{'readfile':'路径'} 或配合 file 填 '$file'。"
                ),
            },
            "select": {
                "params": ["table", "columns", "where", "group_by", "order_by", "limit", "offset"],
                "required": ["table"],
                "description": (
                    "查询记录（SELECT）。支持条件 where、列选择 columns（含 COUNT(*)）、"
                    "分组计数 group_by、排序 order_by 和分页 limit/offset；"
                    "BLOB 列返回 {type, name, mime, size, url}。"
                ),
            },
            "update": {
                "params": ["table", "where", "set", "file"],
                "required": ["table", "where"],
                "description": (
                    "更新符合 where 条件的记录（UPDATE ... SET ... WHERE ...）。"
                    "BLOB 列可传文件路径、'$file'（或仅选 file 自动绑定唯一 BLOB 列）或 null 清空。"
                ),
            },
            "delete": {
                "params": ["table", "where"],
                "required": ["table", "where"],
                "description": "删除符合 where 条件的记录（DELETE FROM ... WHERE ...）并回收关联 BLOB。",
            },
            "import": {
                "params": ["table", "path", "mode", "header_row"],
                "required": ["table", "path"],
                "description": (
                    "导入 .csv 或 .xlsx 表格文件到数据表。.xlsx 中的单元格图片自动存为 BLOB 列。"
                    "mode=create（默认）建新表，mode=append 追加到已有表；整批成功或整批回滚。"
                ),
            },
            "export": {
                "params": ["table", "format", "where", "columns", "order_by"],
                "required": ["table"],
                "description": (
                    "将数据表（或按 where/columns/order_by 筛选的结果）导出为 .xlsx 或 .csv 文件，"
                    "返回本地 path 与 HTTP 下载 url。"
                ),
            },
            "info": {
                "params": [],
                "description": "返回数据表概览、容量上限与 SQLite 完整性检查结果。",
            },
        },
    },
}]

_JSON_PARAMS = ("where", "set", "values", "columns")

_FULLWIDTH = str.maketrans({
    "，": ",", "：": ":", "“": '"', "”": '"', "‘": "'", "’": "'",
    "【": "[", "】": "]", "｛": "{", "｝": "}", "、": ",",
})


class TableDbPlugin:
    """MCP plugin backed by a single SQLite database (`table_db.db`)."""

    PREFIX = PREFIX

    def __init__(self, plugin_cfg: dict, executor=None):
        cfg = dict(plugin_cfg or {})
        self._data_dir = str(cfg.get("data_dir") or DEFAULT_DATA_DIR)
        self._cache_dir = os.path.join(self._data_dir, "cache")
        self._max_cache_bytes = int(cfg.get("max_cache_bytes") or DEFAULT_MAX_CACHE_BYTES)
        self._cache_lock = threading.Lock()
        self._roots = tuple(str(root) for root in (cfg.get("image_roots") or DEFAULT_ROOTS))
        self._max_import_bytes = int(cfg.get("max_import_bytes") or 64 * 1024 * 1024)
        limits = StoreLimits(
            max_file_bytes=int(cfg.get("max_file_bytes") or 8 * 1024 * 1024),
            max_db_bytes=int(cfg.get("max_db_bytes") or 2 * 1024 * 1024 * 1024),
            max_rows=int(cfg.get("max_rows") or 20000),
            max_columns=int(cfg.get("max_columns") or 64),
            max_tables=int(cfg.get("max_tables") or 50),
            max_cell_chars=int(cfg.get("max_cell_chars") or 2000),
        )
        self._store = TableStore(self._data_dir, limits)
        self._http_base = str(cfg.get("http_base") or DEFAULT_HTTP_BASE).rstrip("/")
        self._public_base = str(cfg.get("public_base") or "").rstrip("/")
        self._parse_limits = Limits(
            max_rows=limits.max_rows + 1,
            max_columns=limits.max_columns,
            max_tables=20,
            max_paragraphs=5000,
            max_text_chars=limits.max_cell_chars + 1,
        )
        self._calls = 0
        self._errors = 0
        self._counter_lock = threading.Lock()
        log.info("[table_db] init: data_dir=%s", self._data_dir)

    # ── tool surface ─────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> Optional[dict]:
        args = args if isinstance(args, dict) else {}
        action = args.get("action") if name == PREFIX else name
        handlers = {
            "tables": self._do_tables,
            "create_table": self._do_create_table,
            "drop_table": self._do_drop_table,
            "insert": self._do_insert,
            "select": self._do_select,
            "update": self._do_update,
            "delete": self._do_delete,
            "import": self._do_import,
            "export": self._do_export,
        }
        if action in ("start", "stop"):
            return {"ok": True, "state": "idle", "action": action}
        if action == "info":
            return self._safe(self._info, args)
        handler = handlers.get(action)
        if handler is None:
            return None
        with self._counter_lock:
            self._calls += 1
        return self._safe(lambda given: handler(self._prepare(action, given)), args)

    @staticmethod
    def _prepare(action: str, args: dict) -> dict:
        out = {key: value for key, value in args.items()
               if not (isinstance(value, str) and not value.strip())}
        for key in _JSON_PARAMS:
            value = out.get(key)
            if not isinstance(value, str):
                continue
            out[key] = _parse_text_param(action, key, value.strip())
        spec = TOOLS[0]["inputSchema"]["x-action-params"].get(action, {})
        missing = [key for key in spec.get("required", []) if key not in out]
        if missing:
            hint = "；查看现有表请执行 tables" if "table" in missing else ""
            raise TableDbError("bad_request", "%s 需要填写：%s%s" % (
                action, "、".join(missing), hint), missing=missing)
        return out

    def _safe(self, handler, args: dict) -> dict:
        try:
            return handler(args)
        except TableDbError as error:
            return self._failed(error.as_result())
        except BadInput as error:
            result = error.as_result()
            result["written"] = 0
            return self._failed(result)
        except ParseError as error:
            result = error.as_result()
            result["written"] = 0
            return self._failed(result)
        except sqlite3.OperationalError as error:
            log.warning("[table_db] sqlite operational error: %s", error)
            reason = "busy" if "locked" in str(error).lower() or "busy" in str(error).lower() else "internal_error"
            detail = "数据表正在被其他请求写入，请稍后重试" if reason == "busy" else str(error)[:200]
            return self._failed({"ok": False, "reason": reason, "detail": detail, "written": 0})
        except Exception as error:  # noqa: BLE001
            log.exception("[table_db] unexpected error")
            return self._failed({"ok": False, "reason": "internal_error",
                                 "detail": str(error)[:200], "written": 0})

    def _failed(self, result: dict) -> dict:
        with self._counter_lock:
            self._errors += 1
        log.info("[table_db] refused: reason=%s detail=%s",
                 result.get("reason"), escape_log_text(str(result.get("detail", ""))[:160]))
        return result

    # ── file reading (readfile semantics) ────────────────────────────────────

    def _read_local(self, raw_path: Any, cap: int, field_name: str = "path") -> Tuple[bytes, str]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise bad_request("%s 必须为非空文件路径" % field_name, column=field_name)
        resolved = check_under_roots(raw_path.strip(), {"image_roots": self._roots},
                                     url_action="table_db")
        if not os.path.isfile(resolved):
            raise TableDbError("not_found", "文件不存在：%s" % raw_path, column=field_name)
        size = os.path.getsize(resolved)
        if size == 0:
            raise TableDbError("empty", "文件为空：%s" % os.path.basename(resolved), column=field_name)
        if size > cap:
            raise TableDbError("too_large", "文件大小 %d 字节超过上限 %d 字节" % (size, cap),
                               column=field_name)
        with open(resolved, "rb") as handle:
            return handle.read(), os.path.basename(resolved)

    def _resolve_blob_value(self, raw: Any, upload_file: Optional[str],
                            column: str, row: Optional[int] = None) -> Optional[FileValue]:
        if raw is None:
            return None
        path: Any = raw
        if isinstance(raw, dict) and "readfile" in raw and len(raw) == 1:
            path = raw["readfile"]
        if isinstance(path, str) and path.strip() == "$file":
            if not upload_file:
                raise bad_request("列「%s」使用了 '$file'，但未提供 file 参数" % column,
                                  row=row, column=column)
            path = upload_file
        if not isinstance(path, str) or not path.strip():
            raise bad_request("BLOB 列「%s」请传入文件路径、{'readfile':'路径'} 或 null" % column,
                              row=row, column=column)
        try:
            data, name = self._read_local(path, self._store.limits.max_file_bytes, column)
        except TableDbError as error:
            error.row = row
            raise
        except BadInput as error:
            raise TableDbError(error.reason, error.detail, row=row, column=column) from error
        return FileValue(data=data, name=name, mime=_mime(data, name))

    # ── URL formatting & signing ─────────────────────────────────────────────

    def _public_url(self) -> str:
        if self._public_base:
            return self._public_base
        port = self._http_base.rsplit(":", 1)[-1] if ":" in self._http_base[6:] else "15720"
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect(("10.255.255.255", 1))
                address = probe.getsockname()[0]
            finally:
                probe.close()
        except OSError:
            address = ""
        if not address or address.startswith("127.") or address == "0.0.0.0":
            return self._http_base
        return "http://%s:%s" % (address, port)

    @staticmethod
    def _token() -> str:
        from utils.file_intake import access_token
        return access_token()

    def _signature(self, path: str, exp: int) -> str:
        token = self._token()
        if not token:
            return ""
        payload = ("%s?exp=%d" % (path, exp)).encode("utf-8")
        return hmac.new(token.encode("utf-8"), payload,
                        hashlib.sha256).hexdigest()[:32]

    def _signed_path(self, path: str, ttl_s: int = CACHE_TTL_S) -> str:
        if not self._token():
            return path
        exp = int(time.time()) + int(ttl_s)
        sig = self._signature(path, exp)
        return "%s?exp=%d&sig=%s" % (path, exp, sig)

    def _blob_url(self, table: str, rowid: int, column: str) -> str:
        route = "%s/%s/%d/%s" % (HTTP_BLOB_PREFIX, quote(table, safe=""),
                                 int(rowid), quote(column, safe=""))
        return self._public_url() + self._signed_path(route)

    def _export_url(self, filename: str) -> str:
        route = "%s/%s" % (HTTP_EXPORT_PREFIX, filename)
        return self._public_url() + self._signed_path(route)

    def _decorate_rows(self, table: str, rows: List[dict]) -> List[dict]:
        """Replace internal BLOB `_fid` with clean public download `url`."""
        out: List[dict] = []
        for row in rows:
            rowid = row.get("rowid")
            decorated: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, dict) and value.get("type") == BLOB and "_fid" in value:
                    decorated[key] = {
                        "type": BLOB,
                        "name": value["name"],
                        "mime": value["mime"],
                        "size": value["size"],
                        "url": self._blob_url(table, rowid, key),
                    }
                else:
                    decorated[key] = value
            out.append(decorated)
        return out

    # ── actions: tables / create_table / drop_table ──────────────────────────

    def _do_tables(self, _args: dict) -> dict:
        items = self._store.tables()
        return {
            "ok": True,
            "count": len(items),
            "table_names": [item["table"] for item in items],
            "tables": items,
        }

    def _do_create_table(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        columns = check_columns(args.get("columns"), self._store.limits)
        return self._store.create_table(table, columns)

    def _do_drop_table(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        return self._store.drop_table(table)

    # ── actions: insert / select / update / delete ───────────────────────────

    def _do_insert(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        raw_values = args.get("values")
        if isinstance(raw_values, dict):
            rows = [raw_values]
        elif isinstance(raw_values, list) and raw_values:
            rows = raw_values
        else:
            raise bad_request("values 必须为非空数组 [{'列名':'值'}] 或单行对象 {'列名':'值'}")
        if len(rows) > MAX_INSERT:
            raise TableDbError("too_large", "单次 insert 最多 %d 行，收到 %d 行"
                               % (MAX_INSERT, len(rows)))

        cols = self._store.columns(table)
        if cols is None:
            names = [item["table"] for item in self._store.tables()]
            raise TableDbError("unknown_table", "表「%s」不存在。现有表：%s"
                               % (table, "、".join(names) or "（无）"), tables=names)
        col_types = {col.name: col.type for col in cols}
        blob_cols = [col.name for col in cols if col.type == BLOB]
        upload_file = args.get("file")

        prepared: List[Dict[str, Any]] = []
        for idx, item in enumerate(rows, 1):
            if not isinstance(item, dict) or not item:
                raise bad_request("第 %d 行必须为非空对象 {'列名':'值'}" % idx, row=idx)
            row_copy = dict(item)
            if upload_file and len(rows) == 1 and len(blob_cols) == 1:
                only_blob = blob_cols[0]
                if only_blob not in row_copy:
                    row_copy[only_blob] = "$file"
            for key, value in list(row_copy.items()):
                if col_types.get(key) == BLOB:
                    row_copy[key] = self._resolve_blob_value(value, upload_file, key, row=idx)
            prepared.append(row_copy)

        return self._store.insert(table, prepared)

    def _do_select(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        raw_cols = args.get("columns")
        selected_names: Optional[List[str]] = None
        if raw_cols is not None:
            if isinstance(raw_cols, list):
                selected_names = [
                    (item["name"] if isinstance(item, dict) and "name" in item else str(item))
                    for item in raw_cols
                ]
            else:
                raise bad_request("columns 格式为 ['列名1','列名2'] 或 '列名1,列名2'",
                                  column="columns")
        group_by = args.get("group_by")
        order_by = args.get("order_by")
        offset = _int_param(args.get("offset"), 0, 0, 10 ** 9, "offset")
        limit = _int_param(args.get("limit"), PAGE_ROWS, 1, MAX_PAGE_ROWS, "limit")
        cols, total, rows, is_agg = self._store.select(
            table, columns=selected_names, where=args.get("where"),
            group_by=group_by, order_by=order_by, offset=offset, limit=limit,
        )
        if is_agg:
            reply: Dict[str, Any] = {
                "ok": True,
                "table": table,
                "total": total,
                "count": len(rows),
                "rows": rows,
            }
            if group_by:
                reply["group_by"] = group_by
            return reply
        decorated = self._decorate_rows(table, rows)
        return {
            "ok": True,
            "table": table,
            "columns": [col.describe() for col in cols],
            "total": total,
            "offset": offset,
            "count": len(decorated),
            "has_more": offset + len(decorated) < total,
            "rows": decorated,
        }

    def _do_update(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        where = args.get("where")
        raw_set = args.get("set")
        upload_file = args.get("file")

        cols = self._store.columns(table)
        if cols is None:
            names = [item["table"] for item in self._store.tables()]
            raise TableDbError("unknown_table", "表「%s」不存在。现有表：%s"
                               % (table, "、".join(names) or "（无）"), tables=names)
        col_types = {col.name: col.type for col in cols}
        blob_cols = [col.name for col in cols if col.type == BLOB]

        changes: Dict[str, Any] = dict(raw_set) if isinstance(raw_set, dict) else {}
        if raw_set is not None and not isinstance(raw_set, dict):
            raise bad_request("set 必须为对象，格式 {'列名':'新值'}", column="set")

        if upload_file:
            has_placeholder = any(
                isinstance(v, str) and v.strip() == "$file" for v in changes.values()
            )
            if not has_placeholder:
                if len(blob_cols) == 1 and blob_cols[0] not in changes:
                    changes[blob_cols[0]] = "$file"
                elif not blob_cols:
                    raise bad_request("表「%s」没有 BLOB 列，无法写入 file" % table, column="file")
                elif len(blob_cols) > 1 and not any(k in blob_cols for k in changes):
                    raise bad_request(
                        "表「%s」有多个 BLOB 列（%s），请在 set 中用 {'列名':'$file'} 指定写入列"
                        % (table, "、".join(blob_cols)),
                        column="set",
                    )

        if not changes:
            raise bad_request("update 需要填写 set（或在表有唯一 BLOB 列时传入 file）", column="set")

        for key, value in list(changes.items()):
            if col_types.get(key) == BLOB:
                changes[key] = self._resolve_blob_value(value, upload_file, key)

        result = self._store.update(table, changes, where)
        result["rows"] = self._decorate_rows(table, result.get("rows", []))
        return result

    def _do_delete(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        return self._store.delete(table, args.get("where"))

    # ── actions: import / export (.csv and .xlsx only) ───────────────────────

    def _do_import(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        mode = str(args.get("mode") or "create").strip().lower()
        if mode not in ("create", "append"):
            raise bad_request("mode 只能是 create 或 append，收到 %r" % mode, column="mode")
        header_row = args.get("header_row")
        if header_row is not None:
            header_row = _int_param(header_row, 0, 0, 1000, "header_row")

        data, filename = self._read_local(args.get("path"), self._max_import_bytes, "path")
        suffix = posixpath.splitext(filename.lower())[1]
        if suffix not in _IMPORT_SUFFIXES:
            raise TableDbError("unsupported_format",
                               "不支持的文件格式 %r，仅支持 .csv 和 .xlsx" % (suffix or "（无扩展名）"))

        document = parse_document(data, filename, self._parse_limits)
        if not document.tables:
            raise TableDbError("empty", "文件中没有可读的表格")

        source = document.tables[0]
        warnings = list(document.notes)
        if len(document.tables) > 1:
            warnings.append("工作簿包含 %d 个工作表，仅导入第一个「%s」"
                            % (len(document.tables), source.name))
        if source.truncated:
            raise TableDbError("too_large", "表格超过单表上限（最多 %d 行、%d 列）"
                               % (self._store.limits.max_rows, self._store.limits.max_columns))

        layout = resolve_layout(source, fields=None, header_row=header_row)
        if layout.header_row is None and header_row != 0:
            raise TableDbError("invalid_header",
                               "未识别出表头行，请指定 header_row（表头行号，从 1 开始；0 表示无表头）")

        header_number = layout.header_sheet_row(source)
        data_start = 0 if layout.header_row is None else layout.header_row + 1
        columns, col_map = self._import_columns(source, layout, header_number, data_start)

        existing = self._store.columns(table)
        if mode == "create" and existing is not None:
            raise TableDbError("table_exists",
                               "表「%s」已存在；追加数据请使用 mode=append，或先执行 drop_table"
                               % table)
        if mode == "append":
            if existing is None:
                raise TableDbError("unknown_table",
                                   "表「%s」不存在；append 只能追加到已有表，首次导入请用 mode=create"
                                   % table)
            known = {col.name: col.type for col in existing}
            for name, kind in columns:
                if name not in known:
                    raise TableDbError("unknown_column",
                                       "文件中的列「%s」不在表「%s」中。现有列：%s"
                                       % (name, table, "、".join(known)),
                                       row=header_number, column=name)
                if kind == BLOB and known[name] != BLOB:
                    raise TableDbError("invalid_value",
                                       "列「%s」在文件中包含图片，但表结构中类型为 %s（非 BLOB）"
                                       % (name, known[name]),
                                       row=header_number, column=name)

        archive = open_bounded_zip(data) if document.kind == "xlsx" else None
        kinds = dict(columns)
        if existing is not None:
            kinds.update({col.name: col.type for col in existing})
        records, numbers, file_count = self._import_records(
            source, data_start, col_map, kinds, archive, warnings
        )
        if not records:
            raise TableDbError("empty", "表头下方没有可导入的数据行")

        def make_reply(stored: List[dict]) -> dict:
            reply: Dict[str, Any] = {
                "ok": True,
                "table": table,
                "mode": mode,
                "format": document.kind,
                "columns": [{"name": n, "type": t} for n, t in (
                    [(c.name, c.type) for c in existing] if existing is not None else columns
                )],
                "imported": len(stored),
                "blobs": file_count,
                "rowids": [item["rowid"] for item in stored[:20]],
            }
            if warnings:
                reply["warnings"] = warnings
            return reply

        return self._store.insert(
            table,
            records,
            create_columns=columns if mode == "create" else None,
            make_reply=make_reply,
            row_numbers=numbers,
            row_word="行",
        )

    def _import_columns(self, source, layout, header_number: int, data_start: int
                        ) -> Tuple[List[Tuple[str, str]], Dict[int, str]]:
        seen: Dict[str, int] = {}
        columns: List[Tuple[str, str]] = []
        col_map: Dict[int, str] = {}
        data_rows = source.rows[data_start:]
        col_indices = sorted(layout.keys)
        for col_index in col_indices:
            raw_name = layout.labels.get(col_index) or layout.keys.get(col_index) or ""
            label = raw_name.strip() or ("col_%d" % col_index)
            try:
                name = check_name(label, "表头列名")
            except TableDbError as error:
                raise TableDbError("invalid_header", error.detail,
                                   row=header_number, column=label) from error
            if name in seen:
                raise TableDbError(
                    "invalid_header",
                    "第 %d 行表头「%s」重复（%s 列与 %s 列）"
                    % (header_number, name, column_letter(seen[name]), column_letter(col_index)),
                    row=header_number, column=name,
                )
            seen[name] = col_index
            has_media = any(col_index in row and row[col_index].media for row in data_rows)
            columns.append((name, BLOB if has_media else TEXT))
            col_map[col_index] = name
        if len(columns) > self._store.limits.max_columns:
            raise TableDbError("too_large", "表格共 %d 列，超过上限 %d 列"
                               % (len(columns), self._store.limits.max_columns))
        return columns, col_map

    def _import_records(self, source, data_start: int, col_map: Dict[int, str],
                        kinds: Dict[str, str], archive, warnings: List[str]
                        ) -> Tuple[List[Dict[str, Any]], List[int], int]:
        records: List[Dict[str, Any]] = []
        numbers: List[int] = []
        file_count = 0
        for offset, row in enumerate(source.rows[data_start:]):
            number = source.sheet_rows[data_start + offset]
            if not any(col in row and (row[col].text or row[col].media) for col in col_map):
                continue
            item: Dict[str, Any] = {}
            for col_index, name in col_map.items():
                cell = row.get(col_index)
                if cell is None:
                    continue
                if cell.media:
                    if kinds.get(name) != BLOB:
                        raise TableDbError("invalid_value",
                                           "第 %d 行「%s」包含图片，但该列不是 BLOB 列" % (number, name),
                                           row=number, column=name)
                    value = _archive_file(archive, cell.media, number, name,
                                          self._store.limits.max_file_bytes)
                    if cell.text:
                        warnings.append("第 %d 行「%s」同时含文字与图片，已保留图片" % (number, name))
                    item[name] = value
                    file_count += 1
                elif cell.text:
                    if kinds.get(name) == BLOB:
                        raise TableDbError("invalid_value",
                                           "第 %d 行「%s」为 BLOB 列，单元格包含纯文字 %r"
                                           % (number, name, cell.text[:30]),
                                           row=number, column=name)
                    item[name] = cell.text
            if item:
                records.append(item)
                numbers.append(number)
        return records, numbers, file_count

    def _do_export(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        fmt = str(args.get("format") or "xlsx").strip().lower()
        if fmt not in ("xlsx", "csv"):
            raise bad_request("format 仅支持 xlsx 或 csv，收到 %r" % fmt, column="format")

        raw_cols = args.get("columns")
        selected_names: Optional[List[str]] = None
        if raw_cols is not None:
            if isinstance(raw_cols, list):
                selected_names = [
                    (item["name"] if isinstance(item, dict) and "name" in item else str(item))
                    for item in raw_cols
                ]
            else:
                raise bad_request("columns 格式为 ['列名1','列名2'] 或 '列名1,列名2'",
                                  column="columns")

        cols, _, rows, _ = self._store.select(
            table,
            columns=selected_names,
            where=args.get("where"),
            order_by=args.get("order_by"),
            offset=0,
            limit=self._store.limits.max_rows,
        )
        headers = [col.name for col in cols]
        if fmt == "csv":
            csv_rows: List[List[Any]] = []
            est_bytes = sum(len(h.encode("utf-8")) + 2 for h in headers)
            for row in rows:
                out_row = []
                for col in cols:
                    val = row.get(col.name)
                    if col.type == BLOB and isinstance(val, dict):
                        cell_val = self._blob_url(table, row["rowid"], col.name)
                        est_bytes += len(cell_val.encode("utf-8")) + 2
                        out_row.append(cell_val)
                    else:
                        if val is not None:
                            est_bytes += len(str(val).encode("utf-8")) + 2
                        out_row.append(val)
                if est_bytes > self._max_cache_bytes:
                    raise TableDbError(
                        "too_large",
                        "本次导出预计超过缓存上限 %d 字节；请用 where 分批导出"
                        % self._max_cache_bytes,
                    )
                csv_rows.append(out_row)
            payload = write_csv(headers, csv_rows)
        else:
            # Preflight total image/text bytes BEFORE reading BLOB contents into RAM.
            image_fids: List[int] = []
            est_bytes = sum(len(h.encode("utf-8")) for h in headers)
            for row in rows:
                for col in cols:
                    val = row.get(col.name)
                    if col.type == BLOB and isinstance(val, dict) and "_fid" in val:
                        if val.get("mime") in _IMAGE_MIMES:
                            est_bytes += int(val.get("size") or 0)
                            image_fids.append(val["_fid"])
                        else:
                            est_bytes += len(str(val.get("name") or "").encode("utf-8"))
                    elif val is not None:
                        est_bytes += len(str(val).encode("utf-8"))
            if est_bytes > self._max_cache_bytes:
                raise TableDbError(
                    "too_large",
                    "本次导出预计至少 %d 字节，超过缓存上限 %d 字节；请用 where 分批导出"
                    % (est_bytes, self._max_cache_bytes),
                )
            blobs = self._store.read_files(image_fids) if image_fids else {}
            xlsx_rows: List[List[Any]] = []
            images: Dict[Tuple[int, int], Tuple[bytes, str]] = {}
            for r_idx, row in enumerate(rows, 2):
                out_row = []
                for c_idx, col in enumerate(cols, 1):
                    val = row.get(col.name)
                    if col.type == BLOB and isinstance(val, dict) and "_fid" in val:
                        img_ext = _IMAGE_MIMES.get(val.get("mime", ""))
                        entry = blobs.get(val["_fid"]) if img_ext else None
                        if entry is not None and img_ext:
                            images[(r_idx, c_idx)] = (entry[2], img_ext)
                            out_row.append(None)
                        else:
                            out_row.append(val.get("name"))
                    else:
                        out_row.append(val)
                xlsx_rows.append(out_row)
            payload = write_xlsx(table, headers, xlsx_rows, images=images)

        digest = hashlib.sha1(payload).hexdigest()[:12]
        filename = "export-%s.%s" % (digest, fmt)
        path = self._write_export_cache(filename, payload)
        return {
            "ok": True,
            "table": table,
            "format": fmt,
            "rows": len(rows),
            "size": len(payload),
            "path": path,
            "url": self._export_url(filename),
            "expires_in_s": CACHE_TTL_S,
        }

    # ── HTTP GET handler (/table_db/blob/... and /table_db/export/...) ───────

    def _check_http_auth(self, route: str, query: str,
                         headers: Optional[Dict[str, str]]) -> Optional[Tuple[int, bytes, str, str]]:
        token = self._token()
        if not token:
            return None
        header = ""
        for key, value in (headers or {}).items():
            if key.lower() == "x-access-token":
                header = value
                break
        if header and hmac.compare_digest(header, token):
            return None
        params: Dict[str, str] = {}
        for pair in query.split("&"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                params[key] = value
        try:
            exp = int(params.get("exp", ""))
        except ValueError:
            return 401, b"missing or invalid signature", "text/plain; charset=utf-8", ""
        given = params.get("sig", "")
        expected = self._signature(route, exp)
        if not hmac.compare_digest(given, expected):
            return 401, b"missing or invalid signature", "text/plain; charset=utf-8", ""
        if time.time() > exp:
            return 401, b"link expired", "text/plain; charset=utf-8", ""
        return None

    def handle_http_get(self, url_path: str,
                        headers: Optional[Dict[str, str]] = None
                        ) -> Optional[Tuple[int, bytes, str, str]]:
        """Serve `/table_db/blob/<table>/<rowid>/<col>` or `/table_db/export/<file>`."""
        route, _, query = url_path.partition("?")
        if route.startswith(HTTP_BLOB_PREFIX + "/"):
            denied = self._check_http_auth(route, query, headers)
            if denied is not None:
                return denied
            parts = route[len(HTTP_BLOB_PREFIX) + 1:].split("/")
            try:
                if len(parts) != 3:
                    raise bad_request("expected %s/<table>/<rowid>/<column>" % HTTP_BLOB_PREFIX)
                table = check_name(unquote(parts[0]), "表名")
                rowid = int(parts[1])
                if rowid <= 0:
                    raise ValueError
                column = check_name(unquote(parts[2]), "列名")
                blob = self._store.read_cell_blob(table, rowid, column)
            except (TableDbError, ValueError) as error:
                detail = error.detail if isinstance(error, TableDbError) else "invalid blob route"
                return 400, detail.encode("utf-8"), "text/plain; charset=utf-8", ""
            except Exception as error:  # noqa: BLE001
                log.exception("[table_db] blob download failed")
                return 500, str(error)[:200].encode("utf-8"), "text/plain; charset=utf-8", ""
            if blob is None:
                return 404, b"blob not found", "text/plain; charset=utf-8", ""
            name, mime, data = blob
            return 200, data, mime, name

        if route.startswith(HTTP_EXPORT_PREFIX + "/"):
            denied = self._check_http_auth(route, query, headers)
            if denied is not None:
                return denied
            filename = route[len(HTTP_EXPORT_PREFIX) + 1:]
            if not _EXPORT_FILENAME.match(filename):
                return 400, b"invalid export filename", "text/plain; charset=utf-8", ""
            path = os.path.join(self._cache_dir, filename)
            if not os.path.isfile(path):
                return 404, b"export not found or expired", "text/plain; charset=utf-8", ""
            try:
                with open(path, "rb") as handle:
                    data = handle.read()
            except OSError as error:
                return 500, str(error)[:200].encode("utf-8"), "text/plain; charset=utf-8", ""
            mime = ("text/csv; charset=utf-8" if filename.endswith(".csv")
                    else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            return 200, data, mime, filename

        return None

    # ── cache management (quota + TTL) ───────────────────────────────────────

    def _write_export_cache(self, filename: str, payload: bytes) -> str:
        path = os.path.join(self._cache_dir, filename)
        with self._cache_lock:
            if os.path.isfile(path) and os.path.getsize(path) == len(payload):
                os.utime(path, None)
                self._prune_cache()
                return path
            self._make_cache_room(len(payload), keep=path)
            os.makedirs(self._cache_dir, exist_ok=True)
            partial = path + ".part"
            try:
                with open(partial, "wb") as handle:
                    handle.write(payload)
                os.replace(partial, path)
            except Exception:
                try:
                    os.unlink(partial)
                except OSError:
                    pass
                raise
        return path

    def _make_cache_room(self, needed: int, keep: Optional[str] = None) -> None:
        if needed > self._max_cache_bytes:
            raise TableDbError(
                "too_large",
                "本次导出需要 %d 字节，超过缓存上限 %d 字节；请用 where 分批导出"
                % (needed, self._max_cache_bytes),
            )
        self._prune_cache()
        try:
            entries = list(os.scandir(self._cache_dir))
        except OSError:
            return
        items: List[Tuple[float, str, bool, int]] = []
        total = 0
        for entry in entries:
            try:
                if entry.is_file():
                    size = entry.stat().st_size
                    total += size
                    items.append((entry.stat().st_mtime, entry.path, False, size))
                elif entry.is_dir():
                    size = _dir_bytes(entry.path)
                    total += size
                    items.append((entry.stat().st_mtime, entry.path, True, size))
            except OSError:
                continue
        if total + needed <= self._max_cache_bytes:
            return
        items.sort(key=lambda row: row[0])
        for _, path, is_dir, size in items:
            if total + needed <= self._max_cache_bytes:
                break
            if keep and os.path.abspath(path) == os.path.abspath(keep):
                continue
            try:
                if is_dir:
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.unlink(path)
                total = max(0, total - size)
            except OSError:
                continue
        if total + needed > self._max_cache_bytes:
            raise TableDbError(
                "too_large",
                "缓存空间不足：已有 %d 字节，本次需要 %d 字节，上限 %d 字节"
                % (total, needed, self._max_cache_bytes),
            )

    def _prune_cache(self) -> None:
        cutoff = time.time() - CACHE_TTL_S
        try:
            entries = list(os.scandir(self._cache_dir))
        except OSError:
            return
        for entry in entries:
            try:
                if entry.stat().st_mtime >= cutoff:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry.path, ignore_errors=True)
                elif entry.is_file():
                    os.unlink(entry.path)
            except OSError:
                continue

    # ── info ─────────────────────────────────────────────────────────────────

    def _info(self, _args: dict) -> dict:
        limits = self._store.limits
        return {
            "ok": True,
            "name": "Table DB",
            "manufacture": "Embodied",
            "model": "table_db",
            "state": "idle",
            "desc": "SQLite 数据表卡片：增删改查与 CSV/XLSX 导入导出",
            "data_dir": self._data_dir,
            "tables": self._store.tables(),
            "integrity": self._store.quick_check(),
            "db_bytes": self._store.db_bytes(),
            "import_formats": list(_IMPORT_SUFFIXES),
            "export_formats": ["xlsx", "csv"],
            "limits": {
                "max_file_bytes": limits.max_file_bytes,
                "max_import_bytes": self._max_import_bytes,
                "max_cache_bytes": self._max_cache_bytes,
                "max_db_bytes": limits.max_db_bytes,
                "max_rows": limits.max_rows,
                "max_columns": limits.max_columns,
                "max_tables": limits.max_tables,
                "max_cell_chars": limits.max_cell_chars,
                "page_rows_max": MAX_PAGE_ROWS,
                "insert_max": MAX_INSERT,
            },
            "calls": self._calls,
            "errors": self._errors,
        }


# ── helpers ──────────────────────────────────────────────────────────────────

def _dir_bytes(path: str) -> int:
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _int_param(value: Any, default: int, low: int, high: int, name: str) -> int:
    if value is None or value == "":
        return default
    try:
        if isinstance(value, bool):
            raise ValueError
        number = int(value)
    except (TypeError, ValueError):
        raise bad_request("%s 必须为整数，收到 %r" % (name, value), column=name)
    if number < low or number > high:
        raise bad_request("%s 必须在 %d–%d 之间，收到 %d" % (name, low, high, number), column=name)
    return number


def _parse_text_param(action: str, key: str, text: str) -> Any:
    """Parse JSON or shorthand strings typed on the canvas."""
    folded = text.translate(_FULLWIDTH)
    for candidate in (text, folded):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    if key in ("where", "set", "values"):
        quoted = _try_single_quoted_json(folded)
        if quoted is not None:
            return quoted
    if key == "columns":
        items = [part.strip() for part in folded.strip("[]").split(",") if part.strip()]
        if items:
            if action == "create_table":
                return [_split_column_spec(item) for item in items]
            return items
    raise bad_request(_param_hint(action, key, text), column=key)


def _try_single_quoted_json(text: str) -> Any:
    if "'" not in text:
        return None
    out: List[str] = []
    in_single = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_single and i + 1 < len(text):
            nxt = text[i + 1]
            out.append("'" if nxt == "'" else "\\" + nxt)
            i += 2
            continue
        if ch == "'":
            in_single = not in_single
            out.append('"')
        elif ch == '"' and in_single:
            out.append('\\"')
        else:
            out.append(ch)
        i += 1
    if in_single:
        return None
    try:
        return json.loads("".join(out))
    except ValueError:
        return None


def _split_column_spec(spec: str) -> dict:
    name, sep, kind = spec.partition(":")
    if not sep:
        return {"name": name.strip(), "type": TEXT}
    return {"name": name.strip(), "type": (kind.strip().upper() or TEXT)}


def _param_hint(action: str, key: str, raw: str) -> str:
    shown = raw if len(raw) <= 60 else raw[:57] + "..."
    if key == "columns":
        if action == "create_table":
            return ("columns 格式错误，收到 %r。写法：列名1,列名2:INTEGER,列名3:BLOB"
                    " 或 [{'name':'列名','type':'TEXT'}]" % shown)
        return "columns 格式错误，收到 %r。写法：列名1,列名2 或 ['列名1','列名2'] 或 COUNT(*)" % shown
    if key == "where":
        return "where 格式错误，收到 %r。写法：{'列名':'值'} 或 {'rowid':1}" % shown
    if key == "set":
        return "set 格式错误，收到 %r。写法：{'列名':'新值'}，置空写 null" % shown
    if key == "values":
        return "values 格式错误，收到 %r。写法：[{'列名1':'值1','列名2':'值2'}]" % shown
    return "%s 格式错误，收到 %r" % (key, shown)


def _archive_file(archive, media: str, row: int, column: str,
                  max_bytes: int) -> FileValue:
    if archive is None:
        raise TableDbError("invalid_value", "第 %d 行「%s」引用了图片，但文件不是 .xlsx 格式"
                           % (row, column), row=row, column=column)
    try:
        info = archive.getinfo(media)
    except KeyError:
        raise TableDbError("unreadable", "第 %d 行「%s」的图片在 .xlsx 中找不到（%s）"
                           % (row, column, media), row=row, column=column)
    if info.file_size > max_bytes:
        raise TableDbError(
            "too_large",
            "第 %d 行「%s」的图片大小 %d 字节超过单文件上限 %d 字节"
            % (row, column, info.file_size, max_bytes),
            row=row, column=column,
        )
    try:
        data = archive.read(info, max_bytes=max_bytes)
    except TypeError:
        data = archive.read(info)
    name = posixpath.basename(media)
    return FileValue(data=data, name=name, mime=_mime(data, name))


def _mime(data: bytes, name: str) -> str:
    for magic, mime, _ in _MIME_BY_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    guessed = mimetypes.guess_type(name)[0]
    return guessed or "application/octet-stream"
