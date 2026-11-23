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
  leaves the store as it was.
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
import hmac
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
import socket
import zipfile
import io
from typing import Any, Dict, List, Optional, Tuple

from plugins.image_input import BadInput, check_under_roots
from plugins.table_db_parse import (
    LEGACY_SUFFIXES,
    ParseError,
    SUPPORTED_SUFFIXES,
    open_bounded_zip,
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
DEFAULT_HTTP_BASE = "http://localhost:15720"
HTTP_FILE_PREFIX = "/table_db/file"
DEFAULT_ROOTS = ("/models", "/tmp", "/work")
PAGE_ROWS = 20
MAX_PAGE_ROWS = 200
MAX_APPEND = 500
SAMPLE_ROWS = 3
MAX_SKIPPED = 50
EXPORT_BATCH = 200            # face_recognition's default max_batch
CACHE_TTL_S = 24 * 3600

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

# The canvas renders this schema as ONE form: an `action` dropdown, one field
# per property (in this order), and a single ▶ 执行 button. Fields not listed in
# the selected action's `params` are hidden. What that means for this schema
# (agent-core/web/js/canvas.js, _buildCardEl / _executeCard):
#
# * The grey placeholder is the first 40 characters of `description`, so every
#   description starts with "例 <sample>" followed by the shortest rule; the
#   full text shows when hovering the field name.
# * Object / array fields are plain text boxes and arrive as text; dispatch()
#   parses them (`_JSON_PARAMS`) and accepts the short forms written there.
# * Per-action `required` is not checked by the canvas; dispatch() checks it.
# * Dropdowns always send a value, so no enum here may change the meaning of an
#   action that does not list it (`mode` is read by import only).
# * NO ASCII double quote (") in a property description: the canvas puts the
#   text into placeholder="..." / title="..." with an escaper that does not
#   escape quotes (canvas.js _esc), so `例 {"姓名"...` showed as `例 {`.
#   Examples use single quotes, which _parse_text_param accepts.
# * `accept` greys out other file types in the 选择文件 dialog, which is why
#   import (`path`, spreadsheets) and attach_file (`file`, anything) use two
#   different fields.

ACTIONS = [
    ("list", "查看本机所有库、每个库里的表和列（不用填参数，先点这个）"),
    ("query", "查询某张表的记录（可按条件筛选、分页）"),
    ("stats", "计数：符合条件的总条数，或按某一列分组计数"),
    ("import", "导入 Excel/CSV/Word 表格，单元格里的图片存成文件列"),
    ("create", "建一张空表（自己定表头）；库不存在会自动新建"),
    ("append", "往已有的表里新增记录，返回每条的 _id"),
    ("update", "按 _id 修改一条记录的某些列"),
    ("delete", "按 _id 删除一条记录"),
    ("attach_file", "给一条记录的文件列挂上或更换文件（图片/PDF/音频等）"),
    ("get_file", "按 _id 取出这条记录的文件：返回 url（粘贴到人脸/OCR 卡的 *_by_url，或浏览器打开下载）"),
    ("export_files", "把「姓名+照片」导出成一个目录，用于批量录入人脸库"),
    ("drop_table", "删除一张表及其文件（不可恢复），confirm 填表名"),
    ("drop_db", "删除整个库：库里所有表和文件一起删除（不可恢复），confirm 填库名"),
]

_TABLE_FILES = ".xlsx,.xlsm,.csv,.tsv,.docx,.docm"

TOOLS: List[Dict[str, Any]] = [{
    "name": PREFIX,
    "type": "processor",
    "multiInstance": False,
    "description": (
        "表格数据库：每个 skill 用自己的表保存条目，可导入飞书发来的 Excel/CSV（含图片），"
        "运行中增删改查、分组计数。列只有 text（文字）和 file（图片/PDF/音频等非文字）。"
        "编号 _id 和计数由本卡保证，不要自己读写 JSON 文件。只存取，不判断业务含义。"
        "不知道库名和表名时先调 list。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "info"] + [name for name, _ in ACTIONS],
                # No `title`: the canvas dropdown shows the bare action name;
                # the description is what the detail panel shows.
                "oneOf": [{"const": name, "description": text} for name, text in ACTIONS],
                "description": "要做的操作。每个操作的作用见 oneOf 里的 description",
            },
            "db": {"type": "string",
                   "description": "例 exam。库名，不填=default；有哪些库先执行 list"},
            "table": {"type": "string",
                      "description": "例 名单。表名；不知道有哪些表，先执行 list"},
            "path": {
                "type": "string",
                "format": "file",
                "accept": _TABLE_FILES,
                "uploadTo": "mcp",
                "description": "要导入的表格（xlsx/csv/docx）。飞书发来的文件直接填消息里的 files[].path",
            },
            "file": {
                "type": "string",
                "format": "file",
                "uploadTo": "mcp",
                "description": "要挂上的文件（图片/PDF/音频等任意类型）。飞书文件填 files[].path",
            },
            "where": {"type": "object",
                      "description": ("例 {'姓名':'张三'} 只要姓名=张三的行;不填=全部。"
                                      "多个条件同时满足：{'状态':'借出','借用人':'张三'}；"
                                      "等于其中之一：{'状态':['借出','逾期']}；为空：{'归还时间':null}；"
                                      "可用 _id：{'_id':3}。单引号双引号都可以")},
            "_id": {"type": "integer",
                    "description": "例 3。记录编号，见 query/append 返回的 _id"},
            "set": {"type": "object",
                    "description": ("例 {'状态':'已归还'} 把状态改成已归还,其它列不变。"
                                    "可同时改多列：{'状态':'已归还','归还时间':'11-20'}；"
                                    "值写 null 表示清空这一格；文件列换文件用 attach_file")},
            "records": {
                "type": "array", "items": {"type": "object"},
                "description": ("例 [{'姓名':'张三','状态':'借出'}] 方括号里一个{}是一条。"
                                "多条：[{'姓名':'张三'},{'姓名':'李四'}]；键是列名、值是内容，"
                                "没写的列留空；只能填文字列（文件用 attach_file）；一次≤500条"),
            },
            "columns": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["text", "file"]}}},
                "description": ("例 姓名,状态,照片:file 逗号分隔表头;默认文字列,:file=文件列。"
                                "表头用英文逗号隔开；类型只有 text（文字，默认，可不写）和 "
                                "file（图片/PDF/音频等），写在列名后面、用英文冒号隔开"),
            },
            "group_by": {"type": "string",
                         "description": "例 借用人。按这一列分组计数；不填只给总数"},
            "column": {"type": "string",
                       "description": ("例 照片。文件列的列名；get_file 时表里只有一个文件列可不填。"
                                       "文件列就是 list 里 type 为 file 的列")},
            "name_column": {"type": "string",
                            "description": "例 姓名。每张照片对应的人名所在的列"},
            "profile_columns": {
                "type": "array", "items": {"type": "string"},
                "description": ("例 准考证号,身份证号。可不填；这些列会跟人脸一起存进人脸档案，"
                                "以后认出这个人时一并返回"),
            },
            "mode": {
                "type": "string",
                "enum": ["create", "append"],
                "description": ("create（默认）：建新表，表已存在就报错不写；"
                                "append：追加到已有的库和表，库或表不存在就报错不写"),
            },
            "header_row": {
                "type": "integer", "minimum": 0,
                "description": "例 2。表头在 Excel 第几行，一般不填；0 = 没有表头",
            },
            "dry_run": {"type": "boolean",
                        "description": "例 true。填 true 只预览不写库；不填=正式导入"},
            "confirm": {"type": "string",
                        "description": "例 名单。drop_table 再填一遍表名，drop_db 再填一遍库名"},
            "offset": {"type": "integer", "minimum": 0,
                       "description": "例 20。跳过前几条，用于翻页；不填=0"},
            "limit": {"type": "integer", "minimum": 1,
                      "description": "例 20。最多返回几条；不填=20，最多200"},
        },
        "required": ["action"],
        "x-action-params": {
            "list": {
                "params": [],
                "required": [],
                "description": (
                    "查看本机所有库（db）、每个库里的表、每张表的列（text/file）、行数和文件数。"
                    "不知道库名表名时先调它。表里的具体记录用 query。"
                ),
            },
            "create": {
                "params": ["db", "table", "columns"],
                "required": ["table", "columns"],
                "description": ("建空表，列类型只有 text / file。db 填一个还没有的名字会新建这个库"
                                "（返回 db_created=true）；表已存在会报 table_exists。"),
            },
            "import": {
                "params": ["db", "table", "path", "mode", "header_row", "dry_run"],
                "required": ["table", "path"],
                "description": (
                    "导入表格文件（xlsx/csv，兼容 Word 里的表格；单元格里的图片自动存成 file 列）。"
                    "mode=create（默认）建新表，db 是新名字会新建库，表已存在报 table_exists；"
                    "mode=append 追加到已有的表，库不存在报 unknown_db、表不存在报 unknown_table。"
                    "整批成功或整批不写。返回识别出的列、导入条数、前 3 条样例。"
                    "建议先 dry_run=true 给用户确认。"
                ),
            },
            "append": {
                "params": ["db", "table", "records"],
                "required": ["table", "records"],
                "description": "新增记录（一次 ≤500 条），返回每条的 _id。只写文字列，文件用 attach_file。",
            },
            "query": {
                "params": ["db", "table", "where", "offset", "limit"],
                "required": ["table"],
                "description": (
                    "查询记录（分页）。返回 total、has_more 和 records；文件列显示为 "
                    "{file_id,name,mime,size}，要用文件本身就调 get_file。"
                ),
            },
            "update": {
                "params": ["db", "table", "_id", "set"],
                "required": ["table", "_id", "set"],
                "description": ("按 _id 修改一条记录：set 里写要改的列和新值，没写的列保持不变；"
                                "返回修改后的整条记录。"),
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
                "params": ["db", "table", "_id", "column", "file"],
                "required": ["table", "_id", "column", "file"],
                "description": "给一条记录的文件列挂上/更换文件（如 vision_capture 刚拍的照片、PDF、录音）。",
            },
            "get_file": {
                "params": ["db", "table", "_id", "column"],
                "required": ["table", "_id"],
                "description": (
                    "按记录 _id 取出它的文件（表里只有一个文件列时不用填 column）。返回："
                    "url = 文件的网址：粘贴到 face_recognition / ocr / vop 的 *_by_url 动作的 url，"
                    "或在浏览器打开下载到电脑；path = 机器人上的路径，skill 调视觉卡时可直接作为 image_path；"
                    "agent_core.download = 模型要自己看这个文件时，用 Bash 执行它再 Read local_path。"
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
            "drop_table": {
                "params": ["db", "table", "confirm"],
                "required": ["table", "confirm"],
                "description": (
                    "删除整张表及其文件（不可恢复）。confirm 必须再填一遍表名。"
                    "只在用户明确要求删除或重新导入时使用。"
                ),
            },
            "drop_db": {
                "params": ["db", "confirm"],
                "required": ["db", "confirm"],
                "description": (
                    "删除整个库（库里所有表、记录和文件一起删，不可恢复）。confirm 必须再填一遍库名。"
                    "只在用户明确要求时使用。"
                ),
            },
            "info": {"params": ["db"], "description": "诊断：库列表、表、上限、库完整性。"},
        },
    },
}]

# Object / array parameters the canvas sends as text. A model sends real JSON;
# both are accepted. columns / profile_columns also take "a,b,c" (see _prepare).
_JSON_PARAMS = ("where", "set", "records", "columns", "profile_columns")

# Full-width punctuation typed by a Chinese IME, folded before parsing the
# short forms and as a second chance for JSON.
_FULLWIDTH = str.maketrans({"，": ",", "：": ":", "“": '"', "”": '"', "‘": "'", "’": "'",
                            "【": "[", "】": "]", "｛": "{", "｝": "}", "、": ","})


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
        # Where agent-core reaches this container's HTTP server (host network on
        # the robot, so localhost); used only to build get_file's download URL.
        self._http_base = str(cfg.get("http_base") or DEFAULT_HTTP_BASE).rstrip("/")
        # Address people and other cards use for get_file's url: the robot's
        # LAN IP, so the same link opens in a browser on the operator's PC and
        # is fetched by face_recognition / ocr / vop *_by_url (perception runs
        # on the host network, so it reaches its own LAN address too).
        self._public_base = str(cfg.get("public_base") or "").rstrip("/")
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
            "list": self._do_list, "drop_table": self._do_drop_table,
            "drop_db": self._do_drop_db,
            "create": self._do_create, "import": self._do_import,
            "append": self._do_append, "query": self._do_query,
            "update": self._do_update, "delete": self._do_delete,
            "stats": self._do_stats, "attach_file": self._do_attach,
            "get_file": self._do_get_file, "export_files": self._do_export,
        }
        if action == "show":                 # the name used before `list`
            action = "list"
        if action in ("start", "stop"):
            return {"ok": True, "state": "idle", "action": action}
        if action == "info":
            return self._safe(self._info, args)
        handler = handlers.get(action)
        if handler is None:
            return None
        with self._counter_lock:
            self._calls += 1
        return self._safe(lambda given: handler(self._checked(action, self._prepare(action, given))),
                          args)

    # Actions that may bring a library into existence. Everything else names a
    # library that must already be there: a typo in `db` must answer "no such
    # library", not silently create an empty one.
    _MAY_CREATE_DB = ("list", "create", "import")

    def _checked(self, action: str, args: dict) -> dict:
        if action in self._MAY_CREATE_DB and not (action == "import" and
                                                   str(args.get("mode") or "").strip().lower()
                                                   == "append"):
            return args
        db = check_db_name(args.get("db"))
        names = self._store.databases()
        if db not in names:
            raise TableDbError("unknown_db", "没有库 %s。现有的库：%s。用 list 查看有哪些库和表；"
                               "新建库请用 create 或 import（mode=create）" % (
                                   db, "、".join(names) or "（还没有任何库）"),
                               databases=names)
        return args

    @staticmethod
    def _prepare(action: str, args: dict) -> dict:
        """Normalise one call's arguments the same way for model and canvas.

        * Blank strings count as "not given" (a canvas field left empty).
        * Object/array params given as text are parsed: JSON, or the short
          forms `姓名,状态,照片:file` (columns) and `准考证号,身份证号`
          (profile_columns). Full-width punctuation from a Chinese IME is
          folded first. A text that is neither is a bad_request with the field
          and an example of each accepted form.
        * The action's required params are checked here, so a missing `table`
          answers "fill in table (try list)" rather than a None further down.
        * `request_id` is no longer part of the interface and is ignored.
        """
        out = {key: value for key, value in args.items()
               if not (isinstance(value, str) and not value.strip())}
        out.pop("request_id", None)
        for key in _JSON_PARAMS:
            value = out.get(key)
            if not isinstance(value, str):
                continue
            out[key] = _parse_text_param(key, value.strip())
        spec = TOOLS[0]["inputSchema"]["x-action-params"].get(action, {})
        missing = [key for key in spec.get("required", []) if key not in out]
        if action == "attach_file" and "file" in missing and "path" in out:
            missing.remove("file")           # the field's name before `file`
        if action == "get_file" and "_id" in missing and "file_id" in out:
            missing.remove("_id")            # the form before get_file took _id
        if missing:
            hint = "；不知道库名或表名就先执行 list" if "table" in missing else ""
            raise TableDbError("bad_request", "%s 需要填写：%s%s" % (
                action, "、".join(missing), hint), missing=missing)
        return out

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

    def _read_upload(self, args: dict, max_bytes: int, what: str,
                     key: str = "path") -> Tuple[bytes, str]:
        path = args.get(key)
        if key != "path" and not path:
            path = args.get("path")          # attach_file's field was `path` before `file`
        if not isinstance(path, str) or not path.strip():
            raise bad_request("需要 %s：%s的文件路径（飞书文件填 files[].path）" % (key, what))
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

    # ── list / drop_table ────────────────────────────────────────────────────

    def _do_list(self, args: dict) -> dict:
        """Every library, its tables, their columns and sizes — the "where am I".

        No parameters: the records themselves are `query`'s job.
        """
        databases = []
        for name in self._store.databases():
            databases.append({"db": name, "tables": self._store.overview(name),
                              "bytes": self._store.db_bytes(name),
                              "file": self._store.db_path(name)})
        # Summary first — counts and names in a few lines a person can read on
        # the card — then the full nested detail.
        reply: Dict[str, Any] = {
            "ok": True,
            "databases_count": len(databases),
            "databases_list": [item["db"] for item in databases],
            "tables_list": [{"db": item["db"], "count": len(item["tables"]),
                             "table_list": [table["table"] for table in item["tables"]]}
                            for item in databases],
            "count": len(databases),
            "databases": databases,
        }
        if not databases:
            reply["detail"] = ("还没有任何库。用 import 导入一个表格，或 create 建表；"
                               "db 不填就是 default，填新名字会自动建库")
        else:
            first_db = next((item for item in databases if item["tables"]), None)
            if first_db:
                reply["next"] = ("看某张表的记录：action 选 query，db 填 %s，table 填 %s"
                                 % (first_db["db"], first_db["tables"][0]["table"]))
        return reply

    def _do_drop_table(self, args: dict) -> dict:
        db, table = self._db_table(args)
        if str(args.get("confirm") or "").strip() != table:
            raise bad_request("confirm 要再填一遍表名「%s」才会删除（防止误删）" % table)
        removed = self._store.drop_table(db, table)
        return {"ok": True, "db": db, "table": table, "dropped": True,
                "rows": removed["rows"], "files": removed["files"]}

    def _do_drop_db(self, args: dict) -> dict:
        db = check_db_name(args.get("db"))
        if str(args.get("confirm") or "").strip() != db:
            raise bad_request("confirm 要再填一遍库名「%s」才会删除整个库（防止误删）" % db)
        removed = self._store.drop_db(db)
        shutil.rmtree(os.path.join(self._cache_dir, db), ignore_errors=True)
        return {"ok": True, "db": db, "dropped": True, "tables": removed["tables"],
                "table_list": removed["table_list"], "rows": removed["rows"],
                "files": removed["files"]}

    # ── create ───────────────────────────────────────────────────────────────

    def _do_create(self, args: dict) -> dict:
        db, table = self._db_table(args)
        columns = check_columns(args.get("columns"), self._store.limits)
        new_db = db not in self._store.databases()
        try:
            reply = self._store.create(db, table, columns)
        except Exception:
            if new_db:
                self._store.remove_if_empty(db)
            raise
        if new_db:
            # A library is just a name: it exists as soon as its first table does.
            reply["db_created"] = True
        return reply

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
            raise TableDbError("table_exists", "库 %s 里已经有表「%s」，没有写入。要往里追加用 "
                               "mode=append；要整表重来先 drop_table；或者换一个表名" % (db, table))
        if mode == "append":
            warnings.append("append 不去重：同一份文件追加两次，表里就会有两份")
            if existing is None:
                raise TableDbError("unknown_table", "库 %s 里没有表「%s」，没有写入。append 只能追加到"
                                   "已有的表；第一次导入请用 mode=create" % (db, table))
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
            archive = open_bounded_zip(data)
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

        new_db = db not in self._store.databases()
        try:
            result = self._store.insert(db, table, records,
                                        create_columns=columns if mode == "create" else None,
                                        make_reply=make_reply,
                                        row_numbers=numbers, row_word="行")
        except Exception:
            # The transaction rolled the table back, but opening the library
            # created its file: a failed first import must not leave an empty
            # library behind in `list`.
            if new_db:
                self._store.remove_if_empty(db)
            raise
        if new_db:
            result["db_created"] = True
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
        return self._store.insert(db, table, records)

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
        data, filename = self._read_upload(args, self._store.limits.max_file_bytes, "要挂的",
                                           key="file")
        mime = _mime(data, filename)
        number, info = self._store.attach(db, table, args.get("_id"), column,
                                          FileValue(data, filename, mime))
        return {"ok": True, "db": db, "table": table, "_id": number, "column": column,
                "file": info}

    def _do_get_file(self, args: dict) -> dict:
        """The file of one record, named by the record's _id (and column).

        A record is what a caller already holds (query / append return _id);
        file_id was a second handle for the same thing. The old `file_id`
        argument is still accepted so calls written against it keep working.
        """
        db, table = self._db_table(args)
        columns = self._store.columns(db, table)
        if columns is None:
            raise TableDbError("unknown_table", "库 %s 里没有表「%s」。用 list 查看有哪些表" % (db, table))
        file_columns = [column.name for column in columns if column.type == FILE]
        record = None
        column = args.get("column")
        if "_id" not in args and args.get("file_id"):          # legacy form
            fid = parse_file_id(args.get("file_id"))
        else:
            if not file_columns:
                raise bad_request("表「%s」没有文件列（list 里 type 为 file 的列），没有文件可取" % table)
            if column in (None, ""):
                if len(file_columns) > 1:
                    raise bad_request("表「%s」有 %d 个文件列：%s，请用 column 指定取哪一列"
                                      % (table, len(file_columns), "、".join(file_columns)),
                                      columns=file_columns)
                column = file_columns[0]
            column = check_name(column, "列名 column")
            if column not in file_columns:
                kinds = {item.name: item.type for item in columns}
                if column in kinds:
                    raise bad_request("「%s」是文字列，column 要填文件列：%s"
                                      % (column, "、".join(file_columns)), column=column)
                raise TableDbError("unknown_column", "没有列「%s」。文件列有：%s"
                                   % (column, "、".join(file_columns)), column=column)
            rid = args.get("_id")
            try:
                if isinstance(rid, bool):
                    raise ValueError
                rid = int(rid)
            except (TypeError, ValueError):
                raise bad_request("_id 要是整数（来自 query / append 的返回），收到 %r" % (rid,))
            _, _, found_records = self._store.query(db, table, {"_id": rid}, 0, 1)
            if not found_records:
                raise TableDbError("not_found", "表「%s」里没有 _id=%d 的记录（或已删除）"
                                   % (table, rid))
            record = found_records[0]
            info = record.get(column)
            if not isinstance(info, dict) or info.get("missing"):
                raise TableDbError("not_found", "_id=%d 这条记录的「%s」列没有文件。"
                                   "可以用 attach_file 挂一个" % (rid, column))
            fid = parse_file_id(info["file_id"])
        found = self._store.read_files(db, [fid])
        if fid not in found:
            raise TableDbError("not_found", "库 %s 里没有文件 f_%d" % (db, fid))
        name, mime, data = found[fid]
        path = self._write_cache(db, fid, name, mime, data)
        route = self._signed_route(db, fid)
        url = self._public_url() + route
        local = "/tmp/table_db_%s_%s" % (db, os.path.basename(path))
        reply: Dict[str, Any] = {
            "ok": True, "db": db, "table": table,
            "file": {"file_id": "f_%d" % fid, "name": name, "mime": mime, "size": len(data)},
            "url": url,
            "path": path,
            "expires_in_s": CACHE_TTL_S,
            "how_to_use": ("画布上：复制 url，粘贴到 face_recognition 的 recognize_by_url / "
                           "register_by_url（或 ocr / vop 的 recognize_by_url）的 url 格子执行；"
                           "也可以在浏览器打开 url 把文件下载到电脑。skill 里调视觉卡时直接用 path "
                           "作为 image_path"),
            # `path` lives in the perception container; agent-core's Read cannot
            # open it (it only reads its own /work and /tmp), so the model fetches
            # the file over localhost HTTP first.
            "agent_core": {
                "download": "curl -sf -o %s '%s%s'" % (local, self._http_base, route),
                "local_path": local,
                "note": "模型自己要看这个文件：用 Bash 执行 download，再 Read local_path",
            },
        }
        if record is not None:
            reply["_id"] = record["_id"]
            reply["column"] = column
        return reply

    def _public_url(self) -> str:
        """http://<robot LAN IP>:<port> for links a person or another card opens."""
        if self._public_base:
            return self._public_base
        port = self._http_base.rsplit(":", 1)[-1] if ":" in self._http_base[6:] else "15720"
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # No packet is sent: connect() on UDP only picks the interface
                # the default route would use, which is the robot's LAN address.
                probe.connect(("10.255.255.255", 1))
                address = probe.getsockname()[0]
            finally:
                probe.close()
        except OSError:
            address = ""
        if not address or address.startswith("127.") or address == "0.0.0.0":
            return self._http_base
        return "http://%s:%s" % (address, port)

    # ── file download over HTTP (GET /table_db/file/<db>/<file_id>) ──────────
    #
    # Authorisation is in the URL itself, not in a header: the consumers of this
    # link — a browser, face/ocr/vop *_by_url, a curl the model runs — cannot add
    # an X-Access-Token header, so a header check made every advertised link a
    # 401 the moment ACCESS_TOKEN was set. When a token is configured, get_file
    # signs the path with it (HMAC-SHA256, `?sig=`), and only a link this card
    # issued opens. Without a token the service is gated by reachability, like
    # /file/upload and tools/call, and the link carries no signature.

    @staticmethod
    def _token() -> str:
        from utils.file_intake import access_token
        return access_token()

    def _signature(self, path: str) -> str:
        token = self._token()
        if not token:
            return ""
        return hmac.new(token.encode("utf-8"), path.encode("utf-8"),
                        hashlib.sha256).hexdigest()[:32]

    def _signed_route(self, db: str, fid: int) -> str:
        path = "%s/%s/f_%d" % (HTTP_FILE_PREFIX, db, fid)
        sig = self._signature(path)
        return path + ("?sig=" + sig if sig else "")

    def handle_http_get(self, url_path: str,
                        headers: Optional[Dict[str, str]] = None
                        ) -> Optional[Tuple[int, bytes, str, str]]:
        """Serve one stored file, or None if the path is not ours.

        Returns (status, body, content_type, filename). Only `<db>/<f_N>` is
        accepted — both validated exactly as the tool would — so the URL names
        a row in the store, never a filesystem path. When ACCESS_TOKEN is set,
        a valid `sig` (from get_file) or the X-Access-Token header is required.
        """
        route, _, query = url_path.partition("?")
        if not route.startswith(HTTP_FILE_PREFIX + "/"):
            return None
        token = self._token()
        if token:
            given = ""
            for pair in query.split("&"):
                if pair.startswith("sig="):
                    given = pair[4:]
            expected = self._signature(route)
            header = (headers or {}).get("X-Access-Token", "")
            if not (hmac.compare_digest(given, expected)
                    or (header and hmac.compare_digest(header, token))):
                return 401, b"missing or invalid signature: get a fresh url from get_file", \
                    "text/plain; charset=utf-8", ""
        parts = route[len(HTTP_FILE_PREFIX) + 1:].split("/")
        try:
            if len(parts) != 2:
                raise bad_request("expected %s/<db>/<file_id>" % HTTP_FILE_PREFIX)
            db = check_db_name(parts[0])
            fid = parse_file_id(parts[1])
            found = self._store.read_files(db, [fid])
        except TableDbError as error:
            return 400, error.detail.encode("utf-8"), "text/plain; charset=utf-8", ""
        except Exception as error:  # noqa: BLE001 - an HTTP handler must answer
            log.exception("[table_db] file download failed")
            return 500, str(error)[:200].encode("utf-8"), "text/plain; charset=utf-8", ""
        if fid not in found:
            return 404, ("no file f_%d in %s" % (fid, db)).encode(), "text/plain", ""
        name, mime, data = found[fid]
        return 200, data, mime, name

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
                 "next": ("下一步（录入人脸库）：复制上面的 dir，打开 face_recognition 卡，action 选 "
                          "register_by_corpus，把 dir 粘贴到 package，点执行。会真的写入人脸库，"
                          "返回里 registered 是成功人数；dir 24 小时内有效"),
                 "next_call": "face_recognition.register_by_corpus(package=\"%s\")" % directory}
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

_PARAM_EXAMPLES = {
    "where": "{'姓名':'张三'}",
    "set": "{'状态':'已归还'}",
    "records": "[{'姓名':'张三','状态':'借出'}]",
    "columns": '姓名,状态,照片:file  或  [{"name":"姓名","type":"text"}]',
    "profile_columns": '准考证号,身份证号',
}


def _split_list(text: str) -> List[str]:
    """`a,b,c`, `[a,b,c]`, `["a","b"]` -> ["a","b","c"] (quotes and brackets dropped)."""
    text = text.strip().strip("[]")
    return [part.strip().strip("\"'").strip() for part in text.split(",")
            if part.strip().strip("\"'").strip()]


def _parse_text_param(key: str, text: str) -> Any:
    """One object/array parameter that arrived as text (always, from the canvas).

    JSON first, then JSON with full-width punctuation folded (a Chinese IME
    turns `,` into `，`), then — for the two list-of-names parameters — the
    short comma form. Only if all of that fails is it a bad_request, and the
    message shows the accepted forms rather than a bare parser position.
    """
    folded = text.translate(_FULLWIDTH)
    for candidate in (text, folded, folded.replace("'", '"')):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if key == "columns" and isinstance(value, list):
            value = [item if isinstance(item, dict) else _column_spec(str(item))
                     for item in value]
        return value
    if key in ("columns", "profile_columns"):
        names = _split_list(text.translate(_FULLWIDTH))
        if names:
            return [_column_spec(name) for name in names] if key == "columns" else names
    raise bad_request("%s 的格式不对，收到 %r。正确写法：%s（逗号、冒号用英文半角；引号单双都可以）"
                      % (key, text[:60], _PARAM_EXAMPLES[key]), column=key)


def _column_spec(text: str) -> dict:
    """`照片:file` -> {"name": "照片", "type": "file"}; no `:type` means text."""
    name, _, kind = text.partition(":")
    return {"name": name.strip(), "type": (kind.strip().lower() or TEXT)}


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
