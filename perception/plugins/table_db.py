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

# What the model sees. agent-core splits this card into one tool per
# `x-action-params` entry (`table_db__select`, ...), each carrying the card
# description plus its own, and **every action shares one `properties` map** — so
# a property's description must hold for every action that lists it, and a
# property used two ways (column definitions vs. columns to return) has to be
# two properties. Examples are real JSON: canvas.js escapes quotes now.
#
# `readOnly` per action: the perception layer is read-only to a `viewer` peer
# unless the card says otherwise, and this card deletes data.

ACTIONS = [
    ("list_tables", "列出所有数据表及行数（.tables）"),
    ("describe", "查看一张表的列名和类型（.schema）"),
    ("create_table", "新建数据表（CREATE TABLE）"),
    ("add_column", "给已有表加列（ALTER TABLE ADD COLUMN）"),
    ("drop_table", "删除整张表（DROP TABLE），不可恢复"),
    ("insert", "插入记录（INSERT INTO）"),
    ("select", "查询、筛选、排序、分组统计（SELECT）"),
    ("update", "修改符合条件的记录（UPDATE）"),
    ("delete", "删除符合条件的记录（DELETE FROM）"),
    ("import", "导入 .csv / .xlsx 文件为数据表（.import）"),
    ("export", "把表或查询结果导出为 .xlsx / .csv 文件"),
]

_WHERE_DESC = (
    '筛选条件，多个条件之间是 AND。每个条件写成 [列名, 运算符, 值]。'
    '运算符：= != > >= < <= like、not like（% 匹配任意字符）、in、not in（值为数组）、'
    'between（值为 [下限, 上限]）、is null、is not null（不写值）。'
    '例：[["金额", ">", 100], ["姓名", "like", "%张%"], ["状态", "in", ["待处理", "处理中"]]]。'
    '行号列名为 rowid。'
)

TOOLS: List[Dict[str, Any]] = [{
    "name": PREFIX,
    "type": "processor",
    "multiInstance": False,
    "annotations": {"readOnlyHint": False, "destructiveHint": True},
    "description": "SQLite 数据表：表格数据的增删改查、统计和 .csv/.xlsx 导入导出。不确定表名或列名时先 list_tables / describe。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "info"] + [name for name, _ in ACTIONS],
                "oneOf": [{"const": name, "description": text} for name, text in ACTIONS],
                "description": "要执行的操作",
            },
            "table": {"type": "string", "description": "表名"},
            "columns": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(COLUMN_TYPES)},
                }, "required": ["name"]},
                "description": (
                    '列定义，type 取 TEXT（文本，默认）/ INTEGER（整数）/ REAL（小数）/ BLOB（图片或文件）。'
                    '例：[{"name": "姓名"}, {"name": "金额", "type": "REAL"}, {"name": "照片", "type": "BLOB"}]'
                ),
            },
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    '要返回的列，不填返回全部列。可写聚合 COUNT(*)、COUNT(列)、COUNT(DISTINCT 列)、'
                    'SUM(列)、AVG(列)、MIN(列)、MAX(列)。例：["类别", "SUM(金额)", "COUNT(*)"]'
                ),
            },
            "values": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    '要插入的行，每行一个对象，未写的列为空；一次最多 500 行。'
                    'BLOB 列写本机文件路径。例：[{"姓名": "张三", "金额": 12.5, "照片": "/work/photos/a.jpg"}]'
                ),
            },
            "set": {
                "type": "object",
                "description": (
                    '要改成的新值。null 表示清空；BLOB 列写新文件路径即替换，写 null 即删除该文件。'
                    '例：{"状态": "已完成", "照片": null}'
                ),
            },
            "where": {"type": "array", "items": {"type": "array"}, "description": _WHERE_DESC},
            "group_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    '分组列，每组返回一行；fields 里的普通列必须出现在这里，其余写聚合。'
                    '去重也用它。例：["类别"]'
                ),
            },
            "order_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": '排序，每项写 列名 或 列名 DESC；分组查询可按聚合排序。例：["SUM(金额) DESC", "类别"]',
            },
            "limit": {
                "type": "integer", "minimum": 1, "maximum": MAX_PAGE_ROWS,
                "description": "最多返回行数，默认 %d，最大 %d；结果里 has_more 为 true 时用 offset 翻页"
                               % (PAGE_ROWS, MAX_PAGE_ROWS),
            },
            "offset": {"type": "integer", "minimum": 0, "description": "跳过前几行，用于翻页，默认 0"},
            "confirm": {
                "type": "boolean", "const": True,
                "description": "确认删除整张表；只在用户明确要求删表时填 true",
            },
            "path": {
                "type": "string",
                "format": "file",
                "accept": _TABLE_FILES,
                "uploadTo": "mcp",
                "description": "要导入的 .csv 或 .xlsx 文件路径",
            },
            "file": {
                "type": "string",
                "format": "file",
                "uploadTo": "mcp",
                "description": (
                    '画布上传文件用：表里只有一个 BLOB 列时自动写入它，否则在 values/set 里写 "$file"。'
                    '对话中直接在 values/set 里写文件路径即可'
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["create", "append"],
                "description": "create：新建表（默认）；append：追加到已有表",
            },
            "header_row": {
                "type": "integer",
                "minimum": 0,
                "description": "表头在第几行（从 1 开始），不填自动识别，0 表示没有表头",
            },
            "format": {
                "type": "string",
                "enum": ["xlsx", "csv"],
                "description": "导出格式：xlsx（默认，图片嵌入单元格）或 csv",
            },
        },
        "required": ["action"],
        "x-action-params": {
            "list_tables": {
                "params": [], "required": [], "readOnly": True,
                "description": "列出所有数据表的表名和行数。",
            },
            "describe": {
                "params": ["table"], "required": ["table"], "readOnly": True,
                "description": "查看一张表的列名、列类型和行数。写 where、fields 前先确认列名。",
            },
            "create_table": {
                "params": ["table", "columns"], "required": ["table", "columns"], "readOnly": False,
                "description": "新建数据表。",
            },
            "add_column": {
                "params": ["table", "columns"], "required": ["table", "columns"], "readOnly": False,
                "description": "给已有表加一列或多列，已有行的新列为空。",
            },
            "drop_table": {
                "params": ["table", "confirm"], "required": ["table", "confirm"], "readOnly": False,
                "description": "删除整张表及其全部数据和文件，不可恢复。只在用户明确要求删表时使用，confirm 填 true。",
            },
            "insert": {
                "params": ["table", "values", "file"], "required": ["table", "values"], "readOnly": False,
                "description": "插入一行或多行，整批成功或整批失败，返回新行的 rowid。",
            },
            "select": {
                "params": ["table", "fields", "where", "group_by", "order_by", "limit", "offset"],
                "required": ["table"], "readOnly": True,
                "description": (
                    "查询记录。返回 total（符合条件的总行数）、rows 和 has_more。"
                    "要统计数量、求和、平均就用 fields 里的聚合，不要取出全部行自己算。"
                    "BLOB 列返回 {name, mime, size, url}。"
                ),
            },
            "update": {
                "params": ["table", "set", "where", "file"], "required": ["table", "where"],
                "readOnly": False,
                "description": "把符合 where 的记录改成 set 中的值，返回改了几行。where 必填。",
            },
            "delete": {
                "params": ["table", "where"], "required": ["table", "where"], "readOnly": False,
                "description": "删除符合 where 的记录及其文件，返回删了几行。where 必填。",
            },
            "import": {
                "params": ["table", "path", "mode", "header_row"], "required": ["table", "path"],
                "readOnly": False,
                "description": ".csv 或 .xlsx 导入为数据表，.xlsx 里的图片存为 BLOB 列；整批成功或整批失败。",
            },
            "export": {
                "params": ["table", "format", "fields", "where", "order_by"], "required": ["table"],
                "readOnly": True,
                "description": "把表（或按 fields/where/order_by 筛选后的结果）导出为文件，返回 path 和下载 url。",
            },
        },
    },
}]

_JSON_PARAMS = ("where", "set", "values", "columns", "fields", "group_by", "order_by")
# Names a canvas card saved before the rename; still accepted, never advertised.
_ACTION_ALIASES = {"tables": "list_tables"}

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
        action = _ACTION_ALIASES.get(action, action)
        handlers = {
            "list_tables": self._do_list_tables,
            "describe": self._do_describe,
            "create_table": self._do_create_table,
            "add_column": self._do_add_column,
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
        if action in ("select", "export") and "fields" not in out and "columns" in out:
            out["fields"] = out.pop("columns")   # cards saved before `fields` existed
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

    def _do_list_tables(self, _args: dict) -> dict:
        items = self._store.tables()
        return {"ok": True, "count": len(items),
                "tables": [{"table": item["table"], "rows": item["rows"]} for item in items]}

    def _do_describe(self, args: dict) -> dict:
        return self._store.describe(check_name(args.get("table"), "表名"))

    def _do_add_column(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        return self._store.add_column(table, check_columns(args.get("columns"), self._store.limits))

    def _do_create_table(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        columns = check_columns(args.get("columns"), self._store.limits)
        return self._store.create_table(table, columns)

    def _do_drop_table(self, args: dict) -> dict:
        table = check_name(args.get("table"), "表名")
        if args.get("confirm") not in (True, "true", "True"):
            raise bad_request("删除整张表不可恢复，需要 confirm: true；只删部分行请用 delete",
                              column="confirm")
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
        offset = _int_param(args.get("offset"), 0, 0, 10 ** 9, "offset")
        limit = _int_param(args.get("limit"), PAGE_ROWS, 1, MAX_PAGE_ROWS, "limit")
        cols, total, rows, is_agg = self._store.select(
            table, fields=args.get("fields"), where=args.get("where"),
            group_by=args.get("group_by"), order_by=args.get("order_by"),
            offset=offset, limit=limit,
        )
        if not is_agg:
            rows = self._decorate_rows(table, rows)
        return {
            "ok": True,
            "table": table,
            "columns": [col.describe() for col in cols],
            "total": total,
            "offset": offset,
            "count": len(rows),
            "has_more": offset + len(rows) < total,
            "rows": rows,
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

        cols, _, rows, is_agg = self._store.select(
            table,
            fields=args.get("fields"),
            where=args.get("where"),
            group_by=args.get("group_by"),
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
    if key in ("columns", "fields", "group_by", "order_by"):
        items = [part.strip() for part in folded.strip("[]").split(",") if part.strip()]
        if items:
            if key == "columns":
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
    hints = {
        "columns": '[{"name": "列名", "type": "TEXT"}]，或 列名1,列名2:INTEGER',
        "fields": '["列名", "COUNT(*)"]，或 列名1,SUM(列名2)',
        "where": '[["列名", "=", "值"], ["列名", ">", 100]]',
        "set": '{"列名": "新值"}，清空写 null',
        "values": '[{"列名1": "值1", "列名2": "值2"}]',
        "group_by": '["列名"]',
        "order_by": '["列名 DESC"]',
    }
    if key in hints:
        return "%s 格式错误，收到 %r。写法：%s" % (key, shown, hints[key])
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
