#!/usr/bin/env python3
"""
plugins/table_db.py — the `table_db` card: one table store every skill can share.

Why this card exists
--------------------

Check-in, lending, inventory and visitor skills each kept their data as a JSON
file the model rewrote whole with Read/Write: ids came from the model counting,
totals from the model adding up, and a roster arriving from Feishu as an .xlsx
with photos needed a parsing script pasted into the skill. This card is that
data layer, once, on the platform: a skill names a table, imports a spreadsheet
into it, and reads/writes it through structured JSON calls.

Contract
--------

* Every action takes `table` (and optional `db`, one library per skill).
  Parameters and replies are JSON; every reply has `ok`.
* Two column types: `text`, and `file` (anything that is not text). Files are
  stored as BLOBs in the same SQLite file, so a record and its files commit or
  roll back together. Records show a file as {file_id, name, mime, size};
  `get_file` / `export_files` write it out as a path the vision cards can open.
* Every write is one transaction. A failure answers
  `{"ok": false, "reason", "detail", "row"?, "column"?, "written": 0}` and
  leaves the store as it was. `request_id` makes import/append safe to retry.
* It stores and counts; it does not interpret. What "overdue" means, whether to
  enrol a face, what to tell the user — that is the calling skill's job.

Why `processor` in the perception container
-------------------------------------------

Files handed out by `get_file` must be readable by face_recognition, ocr and
vop, which only open paths under this container's /models — a hard constraint
on placement. `processor` gives agent-core's 60 s timeout (a 5 000-row import)
and hides start/stop/info from the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import posixpath
import re
import shutil
import sqlite3
import threading
import time
import zipfile
import io
from typing import Any, Dict, List, Optional, Tuple

from plugins.image_input import BadInput, check_under_roots
from plugins.table_db_parse import (
    LEGACY_SUFFIXES,
    ParseError,
    SUPPORTED_SUFFIXES,
    Limits,
    column_letter,
    parse_document,
    resolve_layout,
)
from plugins.table_db_store import (
    FILE,
    TEXT,
    FileValue,
    StoreLimits,
    TableDbError,
    TableStore,
    bad_request,
    check_columns,
    check_db_name,
    check_name,
    parse_file_id,
)
from utils.log_sampling import escape_log_text

log = logging.getLogger(__name__)

PREFIX = "table_db"

DEFAULT_DATA_DIR = "/models/table_db"
DEFAULT_ROOTS = ("/models", "/tmp", "/work")
PAGE_ROWS = 20
MAX_PAGE_ROWS = 200
MAX_APPEND = 500
SAMPLE_ROWS = 3
MAX_SKIPPED = 50
EXPORT_BATCH = 200            # face_recognition's default max_batch
CACHE_TTL_S = 24 * 3600
MAX_REQUEST_ID = 128

_IMPORT_SUFFIXES = (".xlsx", ".xlsm", ".csv", ".tsv", ".docx", ".docm")
_MIME_BY_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"GIF87a", "image/gif", ".gif"),
    (b"GIF89a", "image/gif", ".gif"),
    (b"BM", "image/bmp", ".bmp"),
    (b"%PDF", "application/pdf", ".pdf"),
    (b"ID3", "audio/mpeg", ".mp3"),
)

_TABLE_DESC = "表名（1–64 字，可中文，不能以 _ 开头）"
_DB_DESC = "库名，默认 default。建议填你的 skill slug，不同 skill 数据互不干扰"
_WHERE_DESC = ('筛选：{"列":"值"} 等于；{"列":["值1","值2"]} 等于其一；{"列":null} 为空；'
               '多个键同时满足；可用 "_id"')
_REQ_DESC = "可选的请求编号（≤128 字）。同一编号重试返回上次结果，不会重复写入"

TOOLS: List[Dict[str, Any]] = [{
    "name": PREFIX,
    "type": "processor",
    "multiInstance": False,
    "description": (
        "表格数据库：每个 skill 用自己的表保存条目，可导入飞书发来的 Excel/CSV（含图片），"
        "运行中增删改查、分组计数。列只有 text（文字）和 file（图片/PDF/音频等非文字）。"
        "编号 _id 和计数由本卡保证，不要自己读写 JSON 文件。只存取，不判断业务含义。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "info", "create", "import", "append", "query",
                         "update", "delete", "stats", "attach_file", "get_file",
                         "export_files"],
                "description": "Action to perform",
            },
            "db": {"type": "string", "description": _DB_DESC},
            "table": {"type": "string", "description": _TABLE_DESC},
            "columns": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["text", "file"]}}},
                "description": '列定义，如 [{"name":"姓名","type":"text"},{"name":"照片","type":"file"}]',
            },
            "path": {
                "type": "string",
                "format": "file",
                "uploadTo": "mcp",
                "description": "文件路径。飞书发来的文件直接填消息里的 files[].path",
            },
            "mode": {
                "type": "string",
                "enum": ["create", "append"],
                "description": "create（默认）= 按表头建新表；append = 追加到已有表（列名要对得上）",
            },
            "header_row": {
                "type": "integer", "minimum": 0,
                "description": "表头在 Excel 第几行（自动识别错了才填）；0 = 没有表头",
            },
            "dry_run": {"type": "boolean",
                        "description": "true = 只检查并预览，不写库。先给用户看，确认后再正式导入"},
            "request_id": {"type": "string", "description": _REQ_DESC},
            "records": {
                "type": "array", "items": {"type": "object"},
                "description": '要新增的记录，按列名填值（只填文字列），如 [{"姓名":"张三","状态":"借出"}]',
            },
            "where": {"type": "object", "description": _WHERE_DESC},
            "offset": {"type": "integer", "minimum": 0, "description": "跳过几条，默认 0"},
            "limit": {"type": "integer", "minimum": 1,
                      "description": "最多返回几条，默认 20，上限 200"},
            "_id": {"type": "integer", "description": "记录编号（来自 query / append 的返回）"},
            "set": {"type": "object",
                    "description": '要修改的文字列，如 {"状态":"已归还"}；值为 null 表示清空'},
            "group_by": {"type": "string", "description": "分组列名；不填只返回总数"},
            "column": {"type": "string", "description": "文件列（type=file）的列名"},
            "file_id": {"type": "string", "description": "文件编号，如 f_42（来自 query 的返回）"},
            "name_column": {"type": "string", "description": "姓名所在的文字列"},
            "profile_columns": {
                "type": "array", "items": {"type": "string"},
                "description": "一起导出的其它文字列（写进人脸档案 profile）",
            },
        },
        "required": ["action"],
        "x-action-params": {
            "create": {
                "params": ["db", "table", "columns"],
                "required": ["table", "columns"],
                "description": "建空表，列类型只有 text / file。表已存在会报 table_exists。",
            },
            "import": {
                "params": ["db", "table", "path", "mode", "header_row", "dry_run",
                           "request_id"],
                "required": ["table", "path"],
                "description": (
                    "导入表格文件（xlsx/csv，兼容 Word 里的表格；单元格里的图片自动存成 file 列）。"
                    "整批成功或整批不写。返回识别出的列、导入条数、空行、前 3 条样例。"
                    "建议先 dry_run=true 给用户确认。"
                ),
            },
            "append": {
                "params": ["db", "table", "records", "request_id"],
                "required": ["table", "records"],
                "description": "新增记录（一次 ≤500 条），返回每条的 _id。只写文字列，文件用 attach_file。",
            },
            "query": {
                "params": ["db", "table", "where", "offset", "limit"],
                "required": ["table"],
                "description": (
                    "查询记录（分页）。返回 total、has_more 和 records；文件列显示为 "
                    "{file_id,name,mime,size}，要给视觉卡用就调 get_file。"
                ),
            },
            "update": {
                "params": ["db", "table", "_id", "set"],
                "required": ["table", "_id", "set"],
                "description": "按 _id 修改一条记录的文字列，返回修改后的整条记录。",
            },
            "delete": {
                "params": ["db", "table", "_id"],
                "required": ["table", "_id"],
                "description": "按 _id 删除一条记录（标记删除，查询和统计自动跳过）。",
            },
            "stats": {
                "params": ["db", "table", "where", "group_by"],
                "required": ["table"],
                "description": "计数：符合 where 的总条数；给 group_by 时返回每组条数。数数用它，不要自己数。",
            },
            "attach_file": {
                "params": ["db", "table", "_id", "column", "path"],
                "required": ["table", "_id", "column", "path"],
                "description": "给一条记录的文件列挂上/更换文件（如 vision_capture 刚拍的照片、PDF、录音）。",
            },
            "get_file": {
                "params": ["db", "table", "file_id"],
                "required": ["table", "file_id"],
                "description": (
                    "取出文件，返回本容器里的临时路径（24 小时有效），可直接作为 "
                    "face_recognition / ocr / vop 的 image_path。"
                ),
            },
            "export_files": {
                "params": ["db", "table", "column", "name_column", "profile_columns", "where"],
                "required": ["table", "column", "name_column"],
                "description": (
                    "把「姓名 + 文件」批量导出成一个目录（文件 + manifest.json），返回 dir。"
                    "名单录人脸：接着调 face_recognition.register_by_corpus(package=dir)。"
                    "没有文件或没有姓名的行在 skipped 里；一次最多 200 条。"
                ),
            },
            "info": {"params": ["db"], "description": "诊断：库列表、表、上限、库完整性。"},
        },
    },
}]


class TableDbPlugin:
    """MCP plugin. Stateless between calls apart from the SQLite files."""

    PREFIX = PREFIX

    def __init__(self, plugin_cfg: dict, executor=None):
        cfg = dict(plugin_cfg or {})
        self._data_dir = str(cfg.get("data_dir") or DEFAULT_DATA_DIR)
        self._cache_dir = os.path.join(self._data_dir, "cache")
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
        # The parser truncates long cells; asking it for one character more than
        # the store accepts means an over-long cell arrives *over* the limit and
        # is refused with its row and column, instead of being silently cut.
        # The parser also reads one row/column past the store limits, so an
        # oversized sheet is reported as truncated rather than quietly clipped.
        # max_paragraphs stays generous: the Word reader stops at that many body
        # blocks, and a roster's title and instructions come *before* its table.
        self._parse_limits = Limits(max_rows=limits.max_rows + 1,
                                    max_columns=limits.max_columns,
                                    max_tables=20, max_paragraphs=5000,
                                    max_text_chars=limits.max_cell_chars + 1)
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
            "create": self._do_create, "import": self._do_import,
            "append": self._do_append, "query": self._do_query,
            "update": self._do_update, "delete": self._do_delete,
            "stats": self._do_stats, "attach_file": self._do_attach,
            "get_file": self._do_get_file, "export_files": self._do_export,
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
        return self._safe(handler, args)

    def _safe(self, handler, args: dict) -> dict:
        """The outermost guard: no exception ever leaves the card."""
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
            text = str(error)
            if "locked" in text or "busy" in text:
                return self._failed({"ok": False, "reason": "busy",
                                     "detail": "数据库正被另一个写操作占用，稍后重试",
                                     "written": 0})
            log.exception("[table_db] sqlite error")
            return self._failed({"ok": False, "reason": "internal_error",
                                 "detail": "数据库错误：%s" % text[:200], "written": 0})
        except Exception as error:  # noqa: BLE001 - the card must answer, not crash
            log.exception("[table_db] unexpected error")
            return self._failed({"ok": False, "reason": "internal_error",
                                 "detail": "%s: %s" % (type(error).__name__, str(error)[:200]),
                                 "written": 0})

    def _failed(self, result: dict) -> dict:
        with self._counter_lock:
            self._errors += 1
        return result

    def start(self) -> None:
        """Inert: nothing to subscribe to."""

    def stop(self) -> None:
        """Inert."""

    def shutdown(self) -> None:
        """Nothing held open: every call opens and closes its own connection."""

    # ── common argument handling ─────────────────────────────────────────────

    @staticmethod
    def _db_table(args: dict) -> Tuple[str, str]:
        return check_db_name(args.get("db")), check_name(args.get("table"), "表名 table")

    @staticmethod
    def _request(args: dict, payload: Any) -> Optional[Tuple[str, str]]:
        request_id = args.get("request_id")
        if request_id in (None, ""):
            return None
        if not isinstance(request_id, str) or len(request_id) > MAX_REQUEST_ID:
            raise bad_request("request_id 要是不超过 %d 字的字符串" % MAX_REQUEST_ID)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                           default=str).encode("utf-8")).hexdigest()
        return request_id, digest

    @staticmethod
    def _int(args: dict, key: str, default: int, minimum: int, maximum: int) -> int:
        value = args.get(key)
        if value in (None, ""):
            return default
        try:
            if isinstance(value, bool):
                raise ValueError
            number = int(value)
        except (TypeError, ValueError):
            raise bad_request("%s 要是整数，收到 %r" % (key, value))
        if number < minimum:
            raise bad_request("%s 不能小于 %d" % (key, minimum))
        return min(number, maximum)

    @staticmethod
    def _bool(args: dict, key: str) -> bool:
        value = args.get(key)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes")
        return bool(value)

    def _read_upload(self, args: dict, max_bytes: int, what: str) -> Tuple[bytes, str]:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            raise bad_request("需要 path：%s的文件路径（飞书文件填 files[].path）" % what)
        resolved = check_under_roots(path.strip(), {"image_roots": self._roots},
                                     url_action="upload through the card")
        if not os.path.isfile(resolved):
            raise bad_request("找不到文件 %r" % path)
        size = os.path.getsize(resolved)
        if size == 0:
            raise TableDbError("empty", "%r 是空文件" % os.path.basename(resolved))
        if size > max_bytes:
            raise TableDbError("too_large", "文件 %d 字节，超过上限 %d" % (size, max_bytes))
        with open(resolved, "rb") as handle:
            return handle.read(), os.path.basename(resolved)

    # ── create ───────────────────────────────────────────────────────────────

    def _do_create(self, args: dict) -> dict:
        db, table = self._db_table(args)
        columns = check_columns(args.get("columns"), self._store.limits)
        return self._store.create(db, table, columns)

    # ── import ───────────────────────────────────────────────────────────────

    def _do_import(self, args: dict) -> dict:
        started = time.time()
        db, table = self._db_table(args)
        mode = str(args.get("mode") or "create").strip().lower()
        if mode not in ("create", "append"):
            raise bad_request("mode 只能是 create（建新表）或 append（追加到已有表），收到 %r"
                              % args.get("mode"))
        header_row = args.get("header_row")
        if header_row not in (None, ""):
            header_row = self._int(args, "header_row", 0, 0, 10 ** 7)
        else:
            header_row = None
        dry_run = self._bool(args, "dry_run")

        data, filename = self._read_upload(args, self._max_import_bytes, "表格")
        request = None
        if not dry_run:
            request = self._request(args, {"op": "import", "db": db, "table": table,
                                           "mode": mode, "header_row": header_row,
                                           "file": hashlib.sha256(data).hexdigest()})
            replay = self._store.lookup_request(db, request)
            if replay:
                return replay
        suffix = posixpath.splitext(filename.lower())[1]
        if suffix in LEGACY_SUFFIXES:
            raise TableDbError("legacy_format", "这是旧版格式 %s：请用 Excel / WPS 另存为 "
                               ".xlsx 或 .csv 再发一次" % suffix)
        document = parse_document(data, filename, self._parse_limits)
        if not document.tables:
            raise TableDbError("empty", "文件里没有表格。table_db 只导入表格（xlsx / csv / "
                               "Word 里的表格），正文段落不导入")
        source = document.tables[0]
        if source.truncated:
            raise TableDbError("too_large", "表格超过 %d 行或 %d 列的上限；请拆成几个文件导入"
                               % (self._store.limits.max_rows, self._store.limits.max_columns))
        layout = resolve_layout(source, None, header_row)
        if layout.header_row is None:
            raise TableDbError("invalid_header", "没有找到表头。请确认第一行是列名，或用 "
                               "header_row 指定表头在第几行")

        warnings: List[str] = []
        if len(document.tables) > 1:
            warnings.append("文件里有 %d 张表，只导入第一张「%s」"
                            % (len(document.tables), source.name))
        warnings.extend(note for note in document.notes if note not in warnings)
        header_number = layout.header_sheet_row(source)
        columns, col_map = self._import_columns(source, layout, header_number, warnings)

        existing = self._store.columns(db, table)
        if mode == "create" and existing is not None:
            raise TableDbError("table_exists", "表「%s」已经存在。追加用 mode=\"append\"，"
                               "或换一个表名" % table)
        if mode == "append":
            if existing is None:
                raise TableDbError("unknown_table", "表「%s」不存在；第一次导入用 mode=\"create\""
                                   % table)
            known = {column.name: column.type for column in existing}
            for name, kind in columns:
                if name not in known:
                    raise TableDbError("unknown_column", "文件里的列「%s」在表「%s」里没有。"
                                       "表里的列：%s" % (name, table, "、".join(known)),
                                       row=header_number, column=name)
                if kind == FILE and known[name] != FILE:
                    raise TableDbError("invalid_value", "列「%s」在文件里是图片，但表里是文字列"
                                       % name, row=header_number, column=name)

        archive = None
        if document.kind in ("xlsx", "docx"):
            archive = zipfile.ZipFile(io.BytesIO(data))
        kinds = dict(columns)
        if existing is not None:
            kinds.update({column.name: column.type for column in existing})
        records, numbers, skipped, file_count = self._import_records(
            source, layout, col_map, kinds, archive, warnings)
        if not records:
            raise TableDbError("empty", "表头下面没有数据行")

        preview_columns = ([{"name": name, "type": kind} for name, kind in columns])
        reply_base = {
            "ok": True, "db": db, "table": table, "mode": mode,
            "source": {"name": filename, "kind": document.kind, "sheet": source.name},
            "header_row": header_number,
            "columns": preview_columns,
            "rows": len(records),
            "files": file_count,
        }
        if skipped:
            reply_base["skipped_rows"] = skipped[:MAX_SKIPPED]

        if dry_run:
            sample = [{key: (_file_preview(value) if isinstance(value, FileValue) else value)
                       for key, value in record.items()}
                      for record in records[:SAMPLE_ROWS]]
            reply = dict(reply_base, dry_run=True, imported=0, sample=sample)
            if warnings:
                reply["warnings"] = warnings
            return reply

        def make_reply(stored: List[dict]) -> dict:
            reply = dict(reply_base, dry_run=False, imported=len(stored),
                         sample=[_with_row(item, numbers[index])
                                 for index, item in enumerate(stored[:SAMPLE_ROWS])])
            if warnings:
                reply["warnings"] = warnings
            return reply

        result = self._store.insert(db, table, records,
                                    create_columns=columns if mode == "create" else None,
                                    request=request, make_reply=make_reply,
                                    row_numbers=numbers, row_word="行")
        log.info("[table_db] import %s/%s from %s: %s rows in %d ms", db, table,
                 escape_log_text(filename), result.get("imported"),
                 (time.time() - started) * 1000)
        return result

    @staticmethod
    def _import_columns(table, layout, header_number: int,
                        warnings: List[str]) -> Tuple[List[Tuple[str, str]], Dict[int, str]]:
        """(columns as [(name, type)], sheet column index -> column name).

        Returned, not stored on self: concurrent imports run on separate threads.
        """
        start = layout.header_row + 1
        has_media = {col for row in table.rows[start:] for col, cell in row.items()
                     if cell.media}
        has_value = {col for row in table.rows[start:] for col, cell in row.items()
                     if cell.text or cell.media}
        labels = {col: label for col, label in layout.labels.items()}
        out: List[Tuple[str, str]] = []
        seen: Dict[str, int] = {}
        for col in sorted(set(labels) | has_value):
            label = (labels.get(col) or "").strip()
            if not label:
                label = column_letter(col)
                warnings.append("第 %d 行 %s 列没有表头但有数据，列名用「%s」"
                                % (header_number, label, label))
            label = re.sub(r"\s+", " ", label).strip("*※★☆:： ")[:64] or column_letter(col)
            if label.startswith("_"):
                label = label.lstrip("_") or column_letter(col)
            if label in seen:
                raise TableDbError("invalid_header", "第 %d 行表头「%s」重复（%s 列和 %s 列）"
                                   % (header_number, label, column_letter(seen[label]),
                                      column_letter(col)), row=header_number, column=label)
            seen[label] = col
            out.append((label, FILE if col in has_media else TEXT))
        return out, {seen[name]: name for name, _ in out}

    @staticmethod
    def _import_records(table, layout, col_map: Dict[int, str], kinds: Dict[str, str],
                        archive, warnings: List[str]
                        ) -> Tuple[List[dict], List[int], List[int], int]:
        records: List[dict] = []
        numbers: List[int] = []
        skipped: List[int] = []
        files = 0
        multi = 0
        cache: Dict[str, FileValue] = {}
        start = layout.header_row + 1
        for position in range(start, len(table.rows)):
            row = table.rows[position]
            number = table.sheet_rows[position]
            record: Dict[str, Any] = {}
            for col, cell in sorted(row.items()):
                name = col_map.get(col)
                if name is None:
                    continue
                if kinds.get(name) == FILE:
                    if cell.media:
                        if name in record:
                            multi += 1
                            continue
                        value = cache.get(cell.media)
                        if value is None:
                            value = _archive_file(archive, cell.media, number, name)
                            cache[cell.media] = value
                        record[name] = value
                        files += 1
                    elif cell.text:
                        raise TableDbError("invalid_value", "第 %d 行「%s」是图片列，但这一格是文字"
                                           "「%s」" % (number, name, cell.text[:20]),
                                           row=number, column=name)
                elif cell.text:
                    record[name] = cell.text
            if record:
                records.append(record)
                numbers.append(number)
            else:
                skipped.append(number)
        if multi:
            warnings.append("有 %d 个单元格放了多张图片，每格只保留第一张" % multi)
        return records, numbers, skipped, files

    # ── append / query / update / delete / stats ─────────────────────────────

    def _do_append(self, args: dict) -> dict:
        db, table = self._db_table(args)
        records = args.get("records")
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or not records:
            raise bad_request('records 要是非空数组，如 [{"姓名":"张三","状态":"借出"}]')
        if len(records) > MAX_APPEND:
            raise TableDbError("too_large", "一次最多新增 %d 条，收到 %d 条"
                               % (MAX_APPEND, len(records)))
        request = self._request(args, {"op": "append", "db": db, "table": table,
                                       "records": records})
        return self._store.insert(db, table, records, request=request)

    def _do_query(self, args: dict) -> dict:
        db, table = self._db_table(args)
        offset = self._int(args, "offset", 0, 0, 10 ** 9)
        limit = self._int(args, "limit", PAGE_ROWS, 1, MAX_PAGE_ROWS)
        columns, total, records = self._store.query(db, table, args.get("where"),
                                                    offset, limit)
        reply = {"ok": True, "db": db, "table": table,
                 "columns": [column.describe() for column in columns],
                 "total": total, "offset": offset, "count": len(records),
                 "has_more": offset + len(records) < total, "records": records}
        if reply["has_more"]:
            reply["next_offset"] = offset + len(records)
        return reply

    def _do_update(self, args: dict) -> dict:
        db, table = self._db_table(args)
        if "_id" not in args:
            raise bad_request("需要 _id：要修改的记录编号")
        record = self._store.update(db, table, args.get("_id"), args.get("set"))
        return {"ok": True, "db": db, "table": table, "record": record}

    def _do_delete(self, args: dict) -> dict:
        db, table = self._db_table(args)
        if "_id" not in args:
            raise bad_request("需要 _id：要删除的记录编号")
        return {"ok": True, "db": db, "table": table,
                "deleted": self._store.delete(db, table, args.get("_id"))}

    def _do_stats(self, args: dict) -> dict:
        db, table = self._db_table(args)
        return self._store.stats(db, table, args.get("where"), args.get("group_by"))

    # ── files ────────────────────────────────────────────────────────────────

    def _do_attach(self, args: dict) -> dict:
        db, table = self._db_table(args)
        if "_id" not in args:
            raise bad_request("需要 _id：要挂文件的记录编号")
        column = check_name(args.get("column"), "列名 column")
        data, filename = self._read_upload(args, self._store.limits.max_file_bytes, "要挂的")
        mime = _mime(data, filename)
        number, info = self._store.attach(db, table, args.get("_id"), column,
                                          FileValue(data, filename, mime))
        return {"ok": True, "db": db, "table": table, "_id": number, "column": column,
                "file": info}

    def _do_get_file(self, args: dict) -> dict:
        db, table = self._db_table(args)
        fid = parse_file_id(args.get("file_id"))
        if self._store.columns(db, table) is None:
            raise TableDbError("unknown_table", "没有表「%s」" % table)
        found = self._store.read_files(db, [fid])
        if fid not in found:
            raise TableDbError("not_found", "库 %s 里没有文件 f_%d" % (db, fid))
        name, mime, data = found[fid]
        path = self._write_cache(db, fid, name, mime, data)
        return {"ok": True, "file_id": "f_%d" % fid, "path": path, "name": name,
                "mime": mime, "size": len(data), "expires_in_s": CACHE_TTL_S}

    def _do_export(self, args: dict) -> dict:
        db, table = self._db_table(args)
        column = check_name(args.get("column"), "文件列 column")
        name_column = check_name(args.get("name_column"), "姓名列 name_column")
        profile = args.get("profile_columns") or []
        if not isinstance(profile, list) or not all(isinstance(item, str) for item in profile):
            raise bad_request('profile_columns 要是列名数组，如 ["准考证号","身份证号"]')
        columns = self._store.columns(db, table)
        if columns is None:
            raise TableDbError("unknown_table", "没有表「%s」" % table)
        kinds = {item.name: item.type for item in columns}
        for name in [column, name_column] + profile:
            if name not in kinds:
                raise TableDbError("unknown_column", "没有列「%s」。现有的列：%s"
                                   % (name, "、".join(kinds)), column=name)
        if kinds[column] != FILE:
            raise bad_request("「%s」是文字列；column 要填文件列" % column, column=column)
        if kinds[name_column] != TEXT:
            raise bad_request("「%s」是文件列；name_column 要填姓名所在的文字列" % name_column,
                              column=name_column)

        _, total, records = self._store.query(db, table, args.get("where"), 0,
                                              self._store.limits.max_rows)
        chosen, skipped = [], []
        for record in records:
            if not isinstance(record.get(column), dict) or record[column].get("missing"):
                skipped.append({"_id": record["_id"], "reason": "没有文件"})
            elif not str(record.get(name_column) or "").strip():
                skipped.append({"_id": record["_id"], "reason": "没有姓名"})
            else:
                chosen.append(record)
        if len(chosen) > EXPORT_BATCH:
            raise TableDbError("too_large", "有 %d 条可导出，一次最多 %d 条（人脸批量导入的上限）；"
                               "用 where 分批导出" % (len(chosen), EXPORT_BATCH))
        if not chosen:
            return {"ok": True, "db": db, "table": table, "dir": None, "count": 0,
                    "skipped": skipped[:MAX_SKIPPED],
                    "detail": "没有同时具备姓名和文件的记录，没有导出任何内容"}

        files = self._store.read_files(db, [parse_file_id(item[column]["file_id"])
                                            for item in chosen])
        tag = hashlib.sha1(("%s/%s/%s/%s" % (db, table, column, time.time())).encode()
                           ).hexdigest()[:10]
        directory = os.path.join(self._cache_dir, "export-%s" % tag)
        os.makedirs(directory, exist_ok=True)
        manifest = []
        for record in chosen:
            fid = parse_file_id(record[column]["file_id"])
            name, mime, data = files[fid]
            filename = "r%06d%s" % (record["_id"], _suffix(name, mime))
            with open(os.path.join(directory, filename), "wb") as handle:
                handle.write(data)
            item: Dict[str, Any] = {"file": filename, "name": record[name_column],
                                    "person": "%s-%d" % (table, record["_id"])}
            extra = {key: record[key] for key in profile
                     if isinstance(record.get(key), str)}
            if extra:
                item["profile"] = extra
            manifest.append(item)
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=1)
        self._prune_cache()
        reply = {"ok": True, "db": db, "table": table, "dir": directory,
                 "count": len(manifest), "skipped": skipped[:MAX_SKIPPED],
                 "expires_in_s": CACHE_TTL_S,
                 "next": "face_recognition.register_by_corpus(package=\"%s\")" % directory}
        return reply

    def _write_cache(self, db: str, fid: int, name: str, mime: str, data: bytes) -> str:
        directory = os.path.join(self._cache_dir, db)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "f_%d%s" % (fid, _suffix(name, mime)))
        if not (os.path.isfile(path) and os.path.getsize(path) == len(data)):
            partial = path + ".part"
            with open(partial, "wb") as handle:
                handle.write(data)
            os.replace(partial, path)
        else:
            os.utime(path, None)
        self._prune_cache()
        return path

    def _prune_cache(self) -> None:
        cutoff = time.time() - CACHE_TTL_S
        try:
            entries = list(os.scandir(self._cache_dir))
        except OSError:
            return
        for entry in entries:
            try:
                if entry.is_dir() and entry.name.startswith("export-"):
                    if entry.stat().st_mtime < cutoff:
                        shutil.rmtree(entry.path, ignore_errors=True)
                elif entry.is_dir():
                    for item in os.scandir(entry.path):
                        if item.is_file() and item.stat().st_mtime < cutoff:
                            os.unlink(item.path)
            except OSError:
                continue

    # ── info ─────────────────────────────────────────────────────────────────

    def _info(self, args: dict) -> dict:
        limits = self._store.limits
        reply: Dict[str, Any] = {
            "ok": True, "name": "Table DB", "manufacture": "Embodied", "model": "table_db",
            "state": "idle",
            "desc": "表格数据库：导入 Excel/CSV（含图片），增删改查与计数，文件以 BLOB 存储",
            "data_dir": self._data_dir,
            "databases": self._store.databases(),
            "import_formats": list(_IMPORT_SUFFIXES),
            "limits": {"max_file_bytes": limits.max_file_bytes,
                       "max_import_bytes": self._max_import_bytes,
                       "max_db_bytes": limits.max_db_bytes, "max_rows": limits.max_rows,
                       "max_columns": limits.max_columns, "max_tables": limits.max_tables,
                       "max_cell_chars": limits.max_cell_chars, "page_rows_max": MAX_PAGE_ROWS,
                       "append_max": MAX_APPEND, "export_batch": EXPORT_BATCH},
            "calls": self._calls, "errors": self._errors,
        }
        if args.get("db"):
            db = check_db_name(args.get("db"))
            reply["db"] = db
            reply["tables"] = self._store.tables(db)
            reply["integrity"] = self._store.quick_check(db)
        return reply


# ── helpers ──────────────────────────────────────────────────────────────────

def _archive_file(archive, media: str, row: int, column: str) -> FileValue:
    if archive is None:
        raise TableDbError("invalid_value", "第 %d 行「%s」引用了图片，但文件不是 zip 格式"
                           % (row, column), row=row, column=column)
    try:
        data = archive.read(media)
    except KeyError:
        raise TableDbError("unreadable", "第 %d 行「%s」的图片在文件里找不到（%s）"
                           % (row, column, media), row=row, column=column)
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


def _suffix(name: str, mime: str) -> str:
    suffix = os.path.splitext(name)[1].lower()
    if re.match(r"^\.[a-z0-9]{1,5}$", suffix):
        return suffix
    for _, known, extension in _MIME_BY_MAGIC:
        if known == mime:
            return extension
    return mimetypes.guess_extension(mime) or ".bin"


def _file_preview(value: FileValue) -> dict:
    return {"name": value.name, "mime": value.mime, "size": len(value.data)}


def _with_row(record: dict, row: int) -> dict:
    out = dict(record)
    out["_source_row"] = row
    return out
