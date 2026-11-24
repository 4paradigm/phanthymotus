#!/usr/bin/env python3
"""
plugins/table_db_parse.py — read and write .xlsx / .csv tables (standard library only).

Supports:
* .csv (UTF-8 / UTF-8-BOM / GB18030)
* .xlsx (values, inline/shared strings, dates, WPS DISPIMG cell images,
  floating drawing images, Excel 365 place-in-cell images)
* Export back to .csv (UTF-8-BOM) and .xlsx (including embedded image BLOBs).

No third-party dependencies, no ROS, no models.
"""

from __future__ import annotations

import csv
import io
import math
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape as _xml_escape

# ── reasons ──────────────────────────────────────────────────────────────────

REASON_UNSUPPORTED = "unsupported_format"
REASON_UNREADABLE = "unreadable"
REASON_TOO_LARGE = "too_large"
REASON_EMPTY = "empty"
REASON_NO_DOC = "unknown_document"
REASON_NO_TABLE = "unknown_table"

SUPPORTED_SUFFIXES = (".xlsx", ".csv")

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# A cell is one string; the cap keeps a stray 2 MB cell (a pasted log, an
# embedded base64 blob) from becoming the reply's whole payload.
DEFAULT_MAX_TEXT_CHARS = 2000
MAX_TEXT_CHARS = 20000

REASON_BAD_HEADER = "bad_header_row"

# Keys a record already uses; a header label that collides gets a suffix rather
# than overwriting the position information every record carries.
ROW_KEY = "_row"
RESERVED_KEYS = (ROW_KEY, "image", "images")

# How far down a sheet the header is looked for. Rosters put one or two title
# rows above it (`2026年第三期培训考试名单`, a date line); ten covers every real
# template seen so far without letting a data row win on a headerless sheet.
HEADER_PROBE_ROWS = 10

_DATE_BUILTIN_IDS = frozenset(
    (14, 15, 16, 17, 18, 19, 20, 21, 22,          # date / time / datetime
     27, 28, 29, 30, 31, 32, 33, 34, 35, 36,      # CJK era / date formats
     45, 46, 47,                                   # mm:ss / [h]:mm:ss / mmss.0
     50, 51, 52, 53, 54, 55, 56, 57, 58)          # CJK date variants
)


class ParseError(Exception):
    """A document could not be read. Carries the caller-facing detail."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail

    def as_result(self) -> dict:
        return {"ok": False, "reason": self.reason, "detail": self.detail}


# ── limits ───────────────────────────────────────────────────────────────────

@dataclass
class Limits:
    """Bounds on what one document may occupy in memory.

    Every one of these is a *memory* bound, not a policy: the card runs in the
    perception container next to two TensorRT engines on a 7.4 GB Orin, and a
    90 000-row workbook read without a ceiling is how that container gets OOM
    killed for a document nobody meant to read in full. When a bound is hit the
    table is marked `truncated` and a note says which one.
    """

    max_rows: int = 20000
    max_columns: int = 64
    max_tables: int = 20
    max_paragraphs: int = 5000
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS


# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class Cell:
    """One cell: its text, and the archive path of a picture in it."""

    text: str = ""
    media: str = ""          # e.g. "word/media/image1.png" — inside the archive
    lossy: bool = False      # a long number Excel stored with only 15 digits


@dataclass
class Table:
    """A worksheet, a Word table, or a CSV file — raw cells, no interpretation.

    `rows` are dicts keyed by 1-based column index rather than dense lists: real
    spreadsheets are ragged, and a 40-column sheet with 3 populated columns
    should not cost 40 cells per row to store.

    Deliberately holds no header state. Which row is the header and what each
    column is called depends on the *caller* (`fields`, `header_row`), and the
    same cached table is read by concurrent calls with different answers to
    that question — so it is computed per call by `resolve_layout`, never
    stored here.
    """

    name: str
    rows: List[Dict[int, Cell]] = field(default_factory=list)
    sheet_rows: List[int] = field(default_factory=list)   # 1-based row in the sheet
    truncated: bool = False
    hidden: bool = False

    def image_count(self) -> int:
        return sum(1 for row in self.rows
                   for cell in row.values() if cell.media)

    def rows_with_images(self) -> int:
        return sum(1 for row in self.rows
                   if any(cell.media for cell in row.values()))


@dataclass
class Paragraph:
    """One block of a Word body, in document order.

    `table` is set on a placeholder that marks where a Word table sat between
    paragraphs, so a reader of `text` knows "the roster is here" without the
    card having to inline the table into prose.
    """

    text: str = ""
    heading: int = 0          # 1..9 for a heading, 0 for body text
    style: str = ""
    media: List[str] = field(default_factory=list)
    table: str = ""


@dataclass
class Document:
    kind: str                                  # xlsx | docx | csv
    tables: List[Table] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def image_count(self) -> int:
        return (sum(table.image_count() for table in self.tables)
                + sum(len(item.media) for item in self.paragraphs))

    def media_paths(self) -> List[str]:
        """Every picture reference, in first-seen order, deduplicated."""
        seen: Dict[str, None] = {}
        for table in self.tables:
            for row in table.rows:
                for col in sorted(row):
                    if row[col].media:
                        seen.setdefault(row[col].media, None)
        for item in self.paragraphs:
            for media in item.media:
                seen.setdefault(media, None)
        return list(seen)

    def remap_media(self, mapping: Dict[str, str]) -> None:
        """Swap archive paths for extracted file paths; drop unmapped ones."""
        for table in self.tables:
            for row in table.rows:
                for cell in row.values():
                    if cell.media:
                        cell.media = mapping.get(cell.media, "")
        for item in self.paragraphs:
            item.media = [mapping[m] for m in item.media if mapping.get(m)]


# ── small XML helpers ────────────────────────────────────────────────────────

def _local(tag: Any) -> str:
    """Local name of an ElementTree tag, ignoring any namespace URI.

    Matching on local names rather than on one hardcoded namespace URI is what
    makes this work across the namespace variants in the wild: Word 2003 XML
    compatibility files, WPS's `cellimages.xml`, LibreOffice's export, and the
    "strict" OOXML namespace all use different URIs for the same elements.
    """
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _rid(element) -> str:
    """The `r:id` attribute, whatever prefix the document bound it to."""
    for key, value in element.attrib.items():
        if key.endswith("}id") and "relationships" in key:
            return value or ""
    return ""


def _embed(element) -> str:
    """The `r:embed` attribute on an `<a:blip>`, whatever prefix bound it."""
    for key, value in element.attrib.items():
        if key.endswith("}embed") and "relationships" in key:
            return value or ""
    return ""


def _attr(element, name: str, default: str = "") -> str:
    """Read an attribute by local name, namespaced or not.

    WordprocessingML qualifies almost every attribute (`w:val`, `w:type`,
    `w:styleId`), while SpreadsheetML mostly leaves them bare (`r="A1"`,
    `t="n"`). Both spellings mean the same thing, and which one a producer
    writes is an implementation detail — reading only the bare form returned
    `None` for every heading in a Word document and every style id in a sheet.
    """
    value = element.get(name)
    if value is not None:
        return value
    suffix = "}" + name
    for key, candidate in element.attrib.items():
        if key.endswith(suffix):
            return candidate or ""
    return default


def _children(element, name: str) -> Iterable[Any]:
    for child in element:
        if _local(child.tag) == name:
            yield child


def _child(element, name: str):
    for child in element:
        if _local(child.tag) == name:
            return child
    return None


def _iter_named(element, name: str) -> Iterable[Any]:
    for child in element.iter():
        if _local(child.tag) == name:
            yield child


def _parse_xml(blob: bytes, what: str):
    try:
        return ET.fromstring(blob)
    except ET.ParseError as error:
        raise ParseError(REASON_UNREADABLE,
                         "%s 不是有效的 XML：%s" % (what, error)) from error


def _rels(zf: zipfile.ZipFile, part: str) -> Dict[str, str]:
    """Id -> target for a part's `.rels` file, or {} when it has none."""
    rels_path = posixpath.join(posixpath.dirname(part), "_rels",
                               posixpath.basename(part) + ".rels")
    try:
        blob = zf.read(rels_path)
    except KeyError:
        return {}
    out: Dict[str, str] = {}
    for element in _parse_xml(blob, rels_path):
        if _local(element.tag) != "Relationship":
            continue
        identifier = _attr(element, "Id")
        if _attr(element, "TargetMode").lower() == "external":
            # A linked picture lives on the web; this module only reads what is
            # inside the archive, so it resolves to "" and the caller counts it
            # as "not embedded".
            out[identifier] = ""
            continue
        out[identifier] = _attr(element, "Target")
    return out


def _resolve(base: str, target: str) -> str:
    """Resolve a relationship target to an archive path (POSIX, no leading /)."""
    target = (target or "").replace("\\", "/")
    if not target:
        return ""
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(base, target))


def _rich_text(element, skip=("rPh", "phoneticPr")) -> str:
    """Concatenate `<t>` text, skipping phonetic (`rPh`) runs.

    A shared string with furigana carries the reading twice; including it puts
    `とうきょう` next to `東京` in every cell of a Japanese sheet.
    """
    parts: List[str] = []

    def walk(node) -> None:
        for child in node:
            name = _local(child.tag)
            if name in skip:
                continue
            if name == "t":
                parts.append(child.text or "")
            else:
                walk(child)

    walk(element)
    return "".join(parts)


# ── header mapping ───────────────────────────────────────────────────────────

# Built-in aliases. They do two jobs, and neither of them is renaming columns
# behind the caller's back:
#
# 1. **Finding the header row.** A row that contains `姓名` or `身份证号` is a
#    header, wherever it sits under the title rows.
# 2. **Filling in `fields`.** A caller asking for `{"name": []}` gets every
#    spelling below matched for free; one asking `{"name": ["借用人"]}` gets
#    their own word first and these after it.
#
# Without `fields`, a record is keyed by the sheet's own header text — what the
# user sees in Excel is what the model reads.
DEFAULT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "name": ("姓名", "名字", "考生姓名", "学员姓名", "学生姓名", "员工姓名",
             "人员", "人名", "name", "fullname", "full name", "student",
             "student name", "employee"),
    "id": ("身份证号", "身份证号码", "身份证", "证件号", "证件号码", "身份证件号",
           "id", "id number", "idcard", "id no", "identity", "identity number"),
    "photo": ("照片", "相片", "头像", "图片", "考生照片", "证件照", "photo",
              "picture", "avatar", "image"),
    "ticket": ("准考证号", "准考证", "考号", "学号", "工号", "编号",
               "ticket", "ticket number", "no", "number", "student id"),
    "seat": ("座位", "座位号", "座位编号", "位置", "考场座位", "座号",
             "seat", "seat number", "position"),
    "phone": ("手机号", "手机号码", "手机", "电话", "联系电话", "联系方式",
              "phone", "mobile", "tel", "telephone"),
    "email": ("邮箱", "电子邮箱", "邮件", "email", "e-mail", "mail"),
    "dept": ("部门", "单位", "班级", "院系", "科室", "组织", "车间",
             "department", "dept", "team", "class", "unit"),
    "role": ("职务", "职位", "岗位", "角色", "工种", "role", "title", "job"),
    "item": ("物品", "物品名称", "名称", "物资", "设备", "资产", "品名",
             "item", "goods", "asset", "device"),
    "count": ("数量", "件数", "个数", "qty", "quantity", "count", "amount"),
    "date": ("日期", "时间", "签到时间", "考试日期", "date", "time", "datetime"),
    "note": ("备注", "说明", "注释", "note", "notes", "remark", "comment"),
}

_DECORATION = "*※★☆#▲●:：.,，、;； \t\u3000"
_PARENS = re.compile(r"[（(][^）)]*[）)]")
_VALUE_LIKE = re.compile(r"^[\d\s.,:/\-+%年月日时分秒]+$")
_COLUMN_LETTERS = re.compile(r"^[A-Z]{1,3}$")


def normalise_header(text: str) -> str:
    """Fold a header label to the form aliases are compared in.

    Templates decorate headers — `*姓名`, `姓名（必填）`, `姓名:`, `身份证号（18位）`
    — and every one of those has to land on `姓名` or the alias table is only
    useful for hand-made files. Parenthesised hints are stripped first because
    they can contain the decoration characters themselves.
    """
    value = _PARENS.sub("", text or "")
    value = re.sub(r"\s+", "", value)
    value = value.strip(_DECORATION)
    return value.lower()


def column_letter(index: int) -> str:
    """1 -> A, 27 -> AA. The key of a column that has no header text."""
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters or "A"


def _column_index(letters: str) -> int:
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - 64)
    return index


def label_key(label: str) -> str:
    """Record key for a header cell: its own text, tidied, never invented.

    Only the decoration a template adds (`*`, a trailing colon, whitespace and
    line breaks inside the cell) is removed; `身份证号（18位）` stays as written
    because that is what the user will search their spreadsheet for.
    """
    value = re.sub(r"\s+", "", (label or "")).strip(_DECORATION)
    return value[:32]


def field_candidates(key: str, aliases: Iterable[str] = ()) -> List[str]:
    """Normalised labels that satisfy one `fields` entry, in priority order.

    The caller's own aliases first, then the key itself (`{"姓名": []}` means
    "the column called 姓名"), then the built-in spellings when the key is one
    of ours (`{"name": []}` picks up 姓名 / 名字 / 考生姓名 ...).
    """
    out: List[str] = []
    for word in list(aliases or ()) + [key] + list(DEFAULT_ALIASES.get(key, ())):
        if not isinstance(word, str):
            continue
        norm = normalise_header(word)
        if norm and norm not in out:
            out.append(norm)
    return out


def clean_fields(value: Any) -> Dict[str, List[str]]:
    """Accept the shapes a model or a script actually sends for `fields`.

    `{"name": ["姓名"]}` is the documented shape; `{"name": "姓名"}` and
    `["name", "id"]` (built-in aliases only) are what gets typed by hand, and
    rejecting them would be pedantry over an unambiguous request.
    """
    out: Dict[str, List[str]] = {}
    if isinstance(value, (list, tuple)):
        value = {str(item): [] for item in value if str(item).strip()}
    if not isinstance(value, dict):
        return out
    for key, words in value.items():
        key = str(key).strip()
        if not key:
            continue
        if isinstance(words, str):
            words = [words] if words.strip() else []
        if not isinstance(words, (list, tuple)):
            words = []
        out[key] = [str(word) for word in words if str(word).strip()]
    return out


@dataclass
class Layout:
    """How one call reads one table: where the header is, what columns are called.

    `header_row` is an index into `table.rows` (None = no header row). `keys`
    maps a column to its record key; in `fields` mode it holds only the
    requested columns, and `unmatched` lists the fields no column satisfied.
    """

    header_row: Optional[int] = None
    keys: Dict[int, str] = field(default_factory=dict)
    labels: Dict[int, str] = field(default_factory=dict)
    selected: bool = False
    unmatched: List[str] = field(default_factory=list)

    def header_sheet_row(self, table: Table) -> int:
        """1-based sheet row of the header, 0 when there is none."""
        if self.header_row is None or self.header_row >= len(table.sheet_rows):
            return 0
        return table.sheet_rows[self.header_row]

    def columns(self) -> List[str]:
        return [self.keys[col] for col in sorted(self.keys)]


def _looks_like_value(text: str) -> bool:
    """A cell that reads as data (a number, a date, an id), not as a label."""
    return bool(_VALUE_LIKE.match(text)) or len(text) > 40


def _detect_header(table: Table, wanted: List[List[str]]) -> Optional[int]:
    """Index of the header row, or None for a sheet that has none.

    Three passes, strongest evidence first:

    1. the row matching the most of the caller's `fields`;
    2. the row matching the most built-in aliases (`姓名`, `身份证号` ...);
    3. structure: the first row of short, distinct, non-numeric labels that is
       at least half as wide as the table — which is what a header looks like
       in a sheet whose columns are all business-specific (`借用人 | 事项`).

    A one-column list of names has no header by this test, and that is the
    right answer: guessing would consume the first name.
    """
    probe = table.rows[:HEADER_PROBE_ROWS]
    if not probe:
        return None

    def best(candidate_sets: List[List[str]]) -> Optional[int]:
        best_index, best_score = None, 0
        for index, row in enumerate(probe):
            labels = {normalise_header(cell.text) for cell in row.values() if cell.text}
            score = sum(1 for options in candidate_sets
                        if any(option in labels for option in options))
            if score > best_score:
                best_index, best_score = index, score
        return best_index

    if wanted:
        found = best(wanted)
        if found is not None:
            return found
    found = best([[normalise_header(word) for word in words]
                  for words in DEFAULT_ALIASES.values()])
    if found is not None:
        return found

    widest = max(len(row) for row in probe)
    if widest < 2:
        return None
    for index, row in enumerate(probe):
        texts = [cell.text for cell in row.values() if cell.text]
        if any(cell.media for cell in row.values()):
            continue
        if (len(texts) >= max(2, (widest + 1) // 2)
                and len(set(texts)) == len(texts)
                and not any(_looks_like_value(text) for text in texts)):
            return index
    return None


def resolve_layout(table: Table, fields: Optional[Dict[str, List[str]]] = None,
                   header_row: Optional[int] = None) -> Layout:
    """Decide the header row and the record key of every column, for one call.

    `fields` (cleaned by `clean_fields`) switches to *selected* mode: only the
    requested keys appear in records, each bound to the first column whose
    label matches one of its candidates. An alias written as a column letter
    (`{"name": ["B"]}`) binds that column when no label matched — the escape
    hatch for a sheet without a header.

    `header_row` is the 1-based row number the user sees in Excel; 0 means "this
    sheet has no header". Raises ParseError(bad_header_row) for a row that is
    not in the table.
    """
    fields = fields or {}
    layout = Layout(selected=bool(fields))
    wanted = [field_candidates(key, aliases) for key, aliases in fields.items()]

    if header_row is not None:
        if header_row == 0:
            layout.header_row = None
        elif header_row in table.sheet_rows:
            layout.header_row = table.sheet_rows.index(header_row)
        else:
            raise ParseError(
                REASON_BAD_HEADER,
                "表格 %r 里没有第 %d 行（有内容的行是第 %s 行）"
                % (table.name, header_row,
                   "、".join(str(row) for row in table.sheet_rows[:8])
                   + (" …" if len(table.sheet_rows) > 8 else "")))
    else:
        layout.header_row = _detect_header(table, wanted)

    header_cells = (table.rows[layout.header_row]
                    if layout.header_row is not None else {})
    labels = {col: cell.text.strip() for col, cell in header_cells.items() if cell.text}
    layout.labels = dict(labels)

    if fields:
        normalised = {col: normalise_header(text) for col, text in labels.items()}
        claimed: Dict[int, str] = {}
        for key, aliases in fields.items():
            column = None
            for candidate in field_candidates(key, aliases):
                column = next((col for col in sorted(normalised)
                               if normalised[col] == candidate and col not in claimed),
                              None)
                if column is not None:
                    break
            if column is None:
                for alias in aliases:
                    if _COLUMN_LETTERS.match(alias.strip()):
                        index = _column_index(alias.strip())
                        if index not in claimed:
                            column = index
                            break
            if column is None:
                layout.unmatched.append(key)
            else:
                claimed[column] = key
        layout.keys = claimed
        return layout

    columns = set(labels)
    start = 0 if layout.header_row is None else layout.header_row + 1
    for row in table.rows[start:]:
        columns.update(col for col, cell in row.items() if cell.text)
    used: Dict[str, int] = {}
    for col in sorted(columns):
        key = label_key(labels.get(col, "")) or column_letter(col)
        if key in RESERVED_KEYS:
            key = key + "_col"
        if key in used:
            used[key] += 1
            key = "%s_%d" % (key, used[key])
        else:
            used[key] = 1
        layout.keys[col] = key
    return layout


def table_records(table: Table, layout: Layout) -> List[Dict[str, Any]]:
    """Data rows as records under `layout`, empty cells omitted.

    Every record carries `_row` — the row number the user sees in Excel (or the
    table row in Word), so "row 7 has no ID number" can be checked by a human.
    The row's pictures are `image` (the first) and `images` (all, only when
    there are several), wherever in the row they sit. In selected mode a
    requested column that holds a picture rather than text gets the picture's
    path as its value, so `{"photo": ["照片"]}` reads back as a path.

    Omitted rather than emitted as `""`: a 12-column roster where each person
    fills 5 columns would otherwise spend most of the reply on blanks.
    """
    start = 0 if layout.header_row is None else layout.header_row + 1
    out: List[Dict[str, Any]] = []
    for position in range(start, len(table.rows)):
        row = table.rows[position]
        sheet_row = (table.sheet_rows[position]
                     if position < len(table.sheet_rows) else position + 1)
        record: Dict[str, Any] = {ROW_KEY: sheet_row}
        values = 0
        for col in sorted(layout.keys):
            cell = row.get(col)
            if cell is None:
                continue
            if cell.text:
                record[layout.keys[col]] = cell.text
                values += 1
            elif cell.media and layout.selected:
                record[layout.keys[col]] = cell.media
                values += 1
        images = [row[col].media for col in sorted(row) if row[col].media]
        if images:
            record["image"] = images[0]
            if len(images) > 1:
                record["images"] = images
        if values or images:
            out.append(record)
    return out


# ── number formats (dates only) ──────────────────────────────────────────────

def _is_date_format(num_fmt_id: int, code: str) -> bool:
    """True when a number format renders as a date or time.

    Built-in ids cover every date format Excel writes by default, including the
    CJK era formats (27-36, 50-58) that a Chinese-locale Excel emits. A custom
    format is judged on the letters that survive stripping the parts which are
    literal text: `0"m"` is a metre suffix, `mm-dd-yy` is a date.
    """
    if num_fmt_id in _DATE_BUILTIN_IDS:
        return True
    if num_fmt_id < 164 or not code:
        return False                      # built-in non-date, or General
    cleaned = re.sub(r'"[^"]*"', "", code)        # quoted literals
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)  # [Red] / [$-409] / [h]
    cleaned = re.sub(r"\\.", "", cleaned)         # backslash-escaped char
    cleaned = cleaned.lower()
    return any(char in cleaned for char in "ymdhs")


def _serial_to_text(serial: float, date1904: bool = False) -> str:
    """Excel's day serial -> ISO-ish text.

    The 1900 system's epoch is 1899-12-30, not 1900-01-01: Excel carries Lotus
    1-2-3's phantom 1900-02-29, so serial 60 is a date that never existed and
    every later serial is one day ahead of a naive epoch.
    """
    base = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
    days = int(math.floor(serial))
    fraction = serial - days
    try:
        moment = base + timedelta(days=days)
    except (OverflowError, ValueError):
        return ""
    seconds = int(round(fraction * 86400))
    if seconds >= 86400:                   # 23:59:59.6 rounds into tomorrow
        seconds -= 86400
        moment += timedelta(days=1)
    if seconds == 0:
        return moment.strftime("%Y-%m-%d")
    return (moment + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


# ── xlsx ─────────────────────────────────────────────────────────────────────

_CELL_REF = re.compile(r"^([A-Za-z]{1,3})(\d{1,7})$")
_DISPIMG = re.compile(r'DISPIMG\(\s*"([^"]+)"')


def _cell_position(ref: str) -> Tuple[int, int]:
    """`"AB123"` -> (123, 28). Returns (0, 0) for an unusable reference."""
    match = _CELL_REF.match((ref or "").strip())
    if not match:
        return 0, 0
    col = 0
    for char in match.group(1).upper():
        col = col * 26 + (ord(char) - 64)
    return int(match.group(2)), col


def _shared_strings(zf: zipfile.ZipFile, names) -> List[str]:
    if "xl/sharedStrings.xml" not in names:
        return []
    root = _parse_xml(zf.read("xl/sharedStrings.xml"), "xl/sharedStrings.xml")
    return [_rich_text(item) for item in _iter_named(root, "si")]


def _cell_images(zf: zipfile.ZipFile, names) -> Dict[str, str]:
    """WPS `DISPIMG` id -> media path.

    WPS writes the picture into `xl/cellimages.xml` and refers to it from a cell
    formula by id, so this map plus the formula is the only row-level link.
    Matched on local names because WPS's namespace URIs are not the ones
    drawingml uses elsewhere.
    """
    out: Dict[str, str] = {}
    for part in ("xl/cellimages.xml", "xl/media/cellimages.xml"):
        if part not in names:
            continue
        rels = _rels(zf, part)
        root = _parse_xml(zf.read(part), part)
        base = posixpath.dirname(part)
        for picture in _iter_named(root, "pic"):
            name_element = None
            blip = None
            for element in picture.iter():
                local = _local(element.tag)
                if local == "cNvPr" and name_element is None:
                    name_element = element
                elif local == "blip" and blip is None:
                    blip = element
            if name_element is None or blip is None:
                continue
            media = _resolve(base, rels.get(_embed(blip), ""))
            if media:
                out[_attr(name_element, "name")] = media
    return out


def _date_styles(zf: zipfile.ZipFile, names) -> Dict[int, bool]:
    """cellXfs index -> is-date. Empty when the workbook has no styles part."""
    if "xl/styles.xml" not in names:
        return {}
    root = _parse_xml(zf.read("xl/styles.xml"), "xl/styles.xml")
    custom: Dict[int, str] = {}
    for element in _iter_named(root, "numFmt"):
        try:
            custom[int(_attr(element, "numFmtId") or -1)] = _attr(element, "formatCode")
        except (TypeError, ValueError):
            continue
    out: Dict[int, bool] = {}
    for element in _iter_named(root, "cellXfs"):
        for index, xf in enumerate(child for child in element
                                   if _local(child.tag) == "xf"):
            try:
                fmt_id = int(_attr(xf, "numFmtId") or 0)
            except (TypeError, ValueError):
                fmt_id = 0
            out[index] = _is_date_format(fmt_id, custom.get(fmt_id, ""))
    return out


def _date1904(zf: zipfile.ZipFile, names) -> bool:
    if "xl/workbook.xml" not in names:
        return False
    root = _parse_xml(zf.read("xl/workbook.xml"), "xl/workbook.xml")
    for element in _iter_named(root, "workbookPr"):
        return _attr(element, "date1904").lower() in ("1", "true")
    return False


def _xlsx_sheets(zf: zipfile.ZipFile, names) -> List[Tuple[str, str, bool]]:
    """[(name, part path, hidden)] in workbook order."""
    if "xl/workbook.xml" not in names:
        raise ParseError(REASON_UNREADABLE, "不是 Excel 工作簿：缺少 xl/workbook.xml")
    root = _parse_xml(zf.read("xl/workbook.xml"), "xl/workbook.xml")
    rels = _rels(zf, "xl/workbook.xml")
    out: List[Tuple[str, str, bool]] = []
    for element in _iter_named(root, "sheet"):
        name = _attr(element, "name") or ("Sheet%d" % (len(out) + 1))
        path = _resolve("xl", rels.get(_rid(element), ""))
        if not path or path not in names:
            continue
        out.append((name, path, _attr(element, "state") == "hidden"))
    return out


def _read_xlsx_cell(element, shared: List[str], cell_images: Dict[str, str],
                    date_styles: Dict[int, bool], date1904: bool,
                    max_chars: int, rich_images: Optional[Dict[int, str]] = None
                    ) -> Optional[Cell]:
    # Excel 365 "Place in cell" picture: the cell is an error value (#VALUE!)
    # carrying `vm="N"`, and N leads through metadata.xml to an image in
    # xl/richData. Checked first, because its <v> text is a placeholder that
    # would otherwise be read as the cell's content.
    vm = _attr(element, "vm")
    if vm and rich_images:
        try:
            media = rich_images.get(int(vm), "")
        except ValueError:
            media = ""
        if media:
            return Cell(text="", media=media)

    formula = _child(element, "f")
    if formula is not None and "DISPIMG" in (formula.text or ""):
        match = _DISPIMG.search(formula.text or "")
        media = cell_images.get(match.group(1), "") if match else ""
        return Cell(text="", media=media) if media else None

    cell_type = _attr(element, "t").strip()
    inline = _child(element, "is")
    value = _child(element, "v")
    text = ""

    if inline is not None:
        text = _rich_text(inline)
    elif value is not None:
        raw = value.text or ""
        if cell_type == "s":
            try:
                text = shared[int(raw)] if raw.strip().isdigit() else raw
            except (ValueError, IndexError):
                text = raw
        elif cell_type == "b":
            text = "TRUE" if raw.strip() in ("1", "true", "TRUE") else "FALSE"
        elif cell_type in ("str", "e", "d"):
            text = raw
        else:
            text = raw.strip()
            converted_date = False
            if text and date_styles:
                try:
                    style = int(_attr(element, "s") or 0)
                except (TypeError, ValueError):
                    style = 0
                if date_styles.get(style):
                    try:
                        serial = float(text)
                    except ValueError:
                        serial = None
                    if serial is not None and 0 < serial < 2958466:
                        converted = _serial_to_text(serial, date1904)
                        if converted:
                            text = converted
                            converted_date = True
            if text and not converted_date:
                text, lossy = _plain_number(text)
                if lossy:
                    return Cell(text=text[:max_chars] if max_chars > 0 else text, lossy=True)
    if not text:
        return Cell(text="", media="")
    return Cell(text=text[:max_chars] if max_chars > 0 else text)


def _plain_number(text: str) -> Tuple[str, bool]:
    """A numeric cell as Excel displays it: no exponent, no trailing `.0`.

    Excel keeps 15 significant digits, so an 18-digit ID typed into a number
    cell is stored as `1.1010119900101101E+17` and its last digits are already
    gone from the file. Formatting to 15 significant digits gives what the
    user sees in Excel (`110101199001011000`) instead of an exponent or an
    invented tail; the second value says digits were lost, so the caller can
    warn instead of storing a wrong ID silently.
    """
    if not re.match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$", text):
        return text, False
    if "e" not in text.lower() and not text.endswith(".0"):
        return text, False
    try:
        value = float(text)
    except ValueError:
        return text, False
    if not math.isfinite(value) or value != int(value):
        return text, False
    integral = abs(int(value))
    sign = "-" if value < 0 else ""
    if integral < 10 ** 15:
        return sign + str(integral), False
    digits = "%.15g" % integral                     # what Excel keeps
    # Decimal, not float: float("1.10101199001011e+17") is not exactly
    # representable and would come back as ...011008.
    rendered = sign + str(int(Decimal(digits)))
    return rendered, True


def _rich_images(zf: zipfile.ZipFile, names) -> Dict[int, str]:
    """Excel 365 "Place in cell" pictures: cell `vm` index -> media path.

    The chain, all 0-based except `vm`:

        cell vm=N  ->  metadata.xml valueMetadata/bk[N-1]/rc(t, v)
                   ->  futureMetadata "XLRICHVALUE" bk[v] / rvb i
                   ->  richData/rdrichvalue.xml rv[i]  (structure s, values)
                   ->  value at the `_rvRel:LocalImageIdentifier` key
                   ->  richData/richValueRel.xml rel[k] r:id
                   ->  richData/_rels/richValueRel.xml.rels -> ../media/imageK

    Any missing link degrades to "vm N is rich value N-1", which is what Excel
    writes when every rich value is a picture. Returns {} for a workbook with
    no rich data, so the common case costs one membership test.
    """
    values_part = next((name for name in ("xl/richData/rdrichvalue.xml",)
                        if name in names), "")
    rel_part = "xl/richData/richValueRel.xml"
    if not values_part or rel_part not in names:
        return {}
    try:
        rel_ids = [_rid(element) for element in _iter_named(
            _parse_xml(zf.read(rel_part), rel_part), "rel")]
        targets = _rels(zf, rel_part)
        base = posixpath.dirname(rel_part)
        rel_media = [_resolve(base, targets.get(rid, "")) for rid in rel_ids]

        structures: List[List[str]] = []
        structure_part = "xl/richData/rdrichvaluestructure.xml"
        if structure_part in names:
            for item in _iter_named(_parse_xml(zf.read(structure_part), structure_part), "s"):
                structures.append([_attr(key, "n") for key in _children(item, "k")])

        rich_values: List[str] = []
        for item in _iter_named(_parse_xml(zf.read(values_part), values_part), "rv"):
            values = [child.text or "" for child in _children(item, "v")]
            try:
                keys = structures[int(_attr(item, "s") or 0)]
            except (ValueError, IndexError):
                keys = []
            position = next((index for index, key in enumerate(keys)
                             if key == "_rvRel:LocalImageIdentifier"), 0)
            media = ""
            try:
                media = rel_media[int(values[position])]
            except (ValueError, IndexError):
                media = ""
            rich_values.append(media)

        cell_to_value: List[int] = []
        if "xl/metadata.xml" in names:
            root = _parse_xml(zf.read("xl/metadata.xml"), "xl/metadata.xml")
            type_names = [_attr(item, "name") for item in _iter_named(root, "metadataType")]
            future: Dict[str, List[int]] = {}
            for block in _iter_named(root, "futureMetadata"):
                indices = []
                for bk in _children(block, "bk"):
                    rvb = next(iter(_iter_named(bk, "rvb")), None)
                    try:
                        indices.append(int(_attr(rvb, "i")) if rvb is not None else -1)
                    except ValueError:
                        indices.append(-1)
                future[_attr(block, "name")] = indices
            for block in _iter_named(root, "valueMetadata"):
                for bk in _children(block, "bk"):
                    rc = _child(bk, "rc")
                    value = -1
                    if rc is not None:
                        try:
                            kind = type_names[int(_attr(rc, "t")) - 1]
                            value = future.get(kind, [])[int(_attr(rc, "v"))]
                        except (ValueError, IndexError):
                            value = -1
                    cell_to_value.append(value)
    except (KeyError, ParseError):
        return {}

    out: Dict[int, str] = {}
    count = len(cell_to_value) or len(rich_values)
    for vm in range(1, count + 1):
        index = cell_to_value[vm - 1] if cell_to_value else vm - 1
        if 0 <= index < len(rich_values) and rich_values[index]:
            out[vm] = rich_values[index]
    return out


def _read_worksheet(zf: zipfile.ZipFile, part: str, name: str, limits: Limits,
                    row_budget: int, shared: List[str],
                    cell_images: Dict[str, str], date_styles: Dict[int, bool],
                    date1904: bool, hidden: bool,
                    notes: List[str], rich_images: Optional[Dict[int, str]] = None) -> Table:
    root = _parse_xml(zf.read(part), part)
    rows: Dict[int, Dict[int, Cell]] = {}
    truncated = False
    max_chars = max(0, min(limits.max_text_chars, MAX_TEXT_CHARS))
    max_cols = max(1, limits.max_columns)
    lossy: List[str] = []

    for element in _iter_named(root, "c"):
        sheet_row, column = _cell_position(_attr(element, "r"))
        if not sheet_row:
            continue
        if sheet_row not in rows and len(rows) >= max(1, row_budget):
            truncated = True
            break
        if column > max_cols:
            truncated = True
            continue
        cell = _read_xlsx_cell(element, shared, cell_images, date_styles,
                               date1904, max_chars, rich_images)
        if cell is None or (not cell.text and not cell.media):
            continue
        if cell.lossy:
            lossy.append(_attr(element, "r"))
        rows.setdefault(sheet_row, {})[column] = cell

    if lossy:
        notes.append(
            "工作表 %r 有 %d 个超过 15 位的数字（%s%s）在 Excel 里存成了数值，第 15 位之后已经"
            "丢失、显示为 0（身份证号常见）。请把这一列设为「文本」格式、重新填写后再导入"
            % (name, len(lossy), "、".join(lossy[:5]), " 等" if len(lossy) > 5 else ""))

    # Pictures anchored to a cell. Kept separate from the cell loop because the
    # drawing part is a sibling of the sheet's cells, not of any one cell.
    _attach_drawing_images(zf, root, part, rows, notes)

    ordered = sorted(rows)
    table = Table(
        name=name,
        rows=[{col: rows[row][col] for col in sorted(rows[row])} for row in ordered],
        sheet_rows=ordered,
        truncated=truncated,
        hidden=hidden,
    )
    if truncated:
        notes.append("工作表 %r 超过本卡的行/列上限，只读了前一部分" % name)
    return table


def _attach_drawing_images(zf: zipfile.ZipFile, sheet_root, sheet_part: str,
                           rows: Dict[int, Dict[int, Cell]],
                           notes: List[str]) -> None:
    sheet_rels = _rels(zf, sheet_part)
    base = posixpath.dirname(sheet_part)
    known_rows = set(rows)
    for drawing in _children(sheet_root, "drawing"):
        target = _resolve(base, sheet_rels.get(_rid(drawing), ""))
        if not target:
            continue
        try:
            drawing_root = _parse_xml(zf.read(target), target)
        except (KeyError, ParseError):
            notes.append("图片锚点 %s 无法读取，这一页的图片略过" % target)
            continue
        drawing_rels = _rels(zf, target)
        drawing_base = posixpath.dirname(target)
        for anchor in drawing_root:
            if not _local(anchor.tag).endswith("Anchor"):
                continue
            start = _child(anchor, "from")
            if start is None:
                # `absoluteAnchor`: positioned in EMUs relative to the sheet, not
                # in a cell, so there is no row to attach it to. Counting it
                # would claim the caller can reach it by row, which is the one
                # thing they cannot do.
                continue
            row_element = _child(start, "row")
            col_element = _child(start, "col")
            if row_element is None or col_element is None:
                continue
            try:
                sheet_row = int(row_element.text or "") + 1
                column = int(col_element.text or "") + 1
            except (TypeError, ValueError):
                continue
            media = _first_blip(anchor, drawing_base, drawing_rels)
            if not media or sheet_row not in known_rows:
                continue          # anchored below the row budget that was read
            cell = rows[sheet_row].setdefault(column, Cell())
            if not cell.media:
                cell.media = media


def _first_blip(element, base: str, rels: Dict[str, str]) -> str:
    for candidate in _iter_named(element, "blip"):
        media = _resolve(base, rels.get(_embed(candidate), ""))
        if media:
            return media
    return ""


def read_xlsx(data: bytes, limits: Limits) -> Document:
    zf = _open_zip(data)
    names = set(zf.namelist())
    shared = _shared_strings(zf, names)
    cell_images = _cell_images(zf, names)
    rich_images = _rich_images(zf, names)
    date_styles = _date_styles(zf, names)
    date1904 = _date1904(zf, names)

    notes: List[str] = []
    tables: List[Table] = []
    budget = max(0, limits.max_rows)
    sheets = _xlsx_sheets(zf, names)
    for index, (sheet_name, part, hidden) in enumerate(sheets):
        if index >= max(1, limits.max_tables):
            notes.append("工作簿有 %d 个工作表，本卡只读了前 %d 个"
                         % (len(sheets), limits.max_tables))
            break
        if budget <= 0:
            notes.append("已达到本卡的总行数上限，后面的工作表没有读")
            break
        table = _read_worksheet(zf, part, sheet_name, limits, budget, shared,
                                cell_images, date_styles, date1904, hidden, notes,
                                rich_images)
        budget -= len(table.rows)
        tables.append(table)

    if not any(table.rows for table in tables):
        raise ParseError(REASON_EMPTY,
                         "工作簿里没有可读的单元格。如果是"
                         "「显示公式」或者受密码保护的工作簿，请另存为普通 xlsx 再试")
    return Document(kind="xlsx", tables=tables, notes=notes)


# ── csv ──────────────────────────────────────────────────────────────────────

def _decode(data: bytes) -> str:
    """Best-effort text decode.

    utf-8-sig first: Excel's "CSV UTF-8" writes a BOM and reading it as gb18030
    would put a zero-width character in front of the first header. gb18030
    second because that is what Excel writes on a Chinese Windows box, and it
    round-trips nearly every byte sequence, so a wrong guess here degrades to
    mojibake rather than an exception.
    """
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def read_csv(data: bytes, limits: Limits) -> Document:
    text = _decode(data)
    if not text.strip():
        raise ParseError(REASON_EMPTY, "CSV 文件是空的")
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|")
    except csv.Error:
        # One column, or too little text to sniff. csv's default (comma) is the
        # right answer for both, and refusing would reject a valid single-column
        # list of names.
        dialect = None
    reader = (csv.reader(io.StringIO(text), dialect) if dialect is not None
              else csv.reader(io.StringIO(text)))
    max_chars = max(0, min(limits.max_text_chars, MAX_TEXT_CHARS))
    max_cols = max(1, limits.max_columns)
    rows: List[Dict[int, Cell]] = []
    sheet_rows: List[int] = []
    truncated = False
    notes: List[str] = []
    for index, values in enumerate(reader, 1):
        row: Dict[int, Cell] = {}
        for column, value in enumerate(values, 1):
            if column > max_cols:
                truncated = True
                break
            cleaned = (value or "").strip()
            if cleaned:
                row[column] = Cell(text=cleaned[:max_chars])
        if row:
            rows.append(row)
            sheet_rows.append(index)
        if len(rows) >= max(1, limits.max_rows):
            truncated = True
            notes.append("CSV 超过本卡的行数上限，只读了前 %d 行" % len(rows))
            break
    if not rows:
        raise ParseError(REASON_EMPTY, "CSV 里没有可读的内容")
    return Document(kind="csv",
                    tables=[Table(name="csv", rows=rows, sheet_rows=sheet_rows,
                                  truncated=truncated)],
                    notes=notes)


# ── entry point ──────────────────────────────────────────────────────────────

# Bounds on what an OOXML archive may expand to. The import cap (64 MiB) is on
# the *compressed* upload; a tiny, highly compressible xlsx can declare
# gigabytes of XML, and reading one member whole would OOM this container.
# Real workbooks sit far below every one of these: a 20 000-row sheet is tens
# of MB of XML, and Office compresses XML around 10-20x.
MAX_ZIP_MEMBERS = 10000
MAX_ZIP_MEMBER_BYTES = 256 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_ZIP_RATIO = 200          # uncompressed / compressed, per member over 1 MiB


class _BoundedZip(zipfile.ZipFile):
    """ZipFile whose read() never returns more than a member's declared size.

    The declared sizes are checked up front (`_open_zip`); this enforces them,
    so an archive that lies in its headers is cut off at the declared size
    instead of being trusted. zipfile itself raises BadZipFile on a CRC or
    length mismatch at the end of a member, but only after inflating it.
    """

    def read(self, name, pwd=None):  # noqa: D401 - same contract as ZipFile.read
        info = name if isinstance(name, zipfile.ZipInfo) else self.getinfo(name)
        limit = min(info.file_size, MAX_ZIP_MEMBER_BYTES)
        with self.open(info, "r", pwd=pwd) as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise ParseError(REASON_UNREADABLE,
                             "文件里的 %s 解压后超过声明的大小，文件可能已损坏或被篡改" % info.filename)
        return data


def check_zip_bounds(zf: zipfile.ZipFile) -> None:
    """Refuse an archive whose declared expansion exceeds the bounds above."""
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ParseError(REASON_TOO_LARGE, "文件里有 %d 个部件，超过上限 %d，不像正常的 Office 文档"
                         % (len(infos), MAX_ZIP_MEMBERS))
    total = 0
    for info in infos:
        if info.file_size > MAX_ZIP_MEMBER_BYTES:
            raise ParseError(REASON_TOO_LARGE, "文件里的 %s 解压后 %d 字节，超过单个部件上限 %d"
                             % (info.filename, info.file_size, MAX_ZIP_MEMBER_BYTES))
        if (info.file_size > 1024 * 1024
                and info.file_size > MAX_ZIP_RATIO * max(info.compress_size, 1)):
            raise ParseError(REASON_TOO_LARGE, "文件里的 %s 压缩比异常（%d 倍），拒绝解压"
                             % (info.filename, info.file_size // max(info.compress_size, 1)))
        total += info.file_size
    if total > MAX_ZIP_TOTAL_BYTES:
        raise ParseError(REASON_TOO_LARGE, "文件解压后共 %d 字节，超过上限 %d"
                         % (total, MAX_ZIP_TOTAL_BYTES))


def _open_zip(data: bytes) -> zipfile.ZipFile:
    try:
        zf = _BoundedZip(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ParseError(
            REASON_UNREADABLE,
            "文件不是有效的 Office 文档（zip 打不开：%s）。"
            "常见原因：文件扩展名被改过、下载不完整、或者其实是网页/错误页"
            % error,
        ) from error
    check_zip_bounds(zf)
    return zf


open_bounded_zip = _open_zip


def sniff_kind(data: bytes, filename: str) -> str:
    """Identify .xlsx or .csv by content / extension."""
    if data[:8] == _OLE_MAGIC:
        return "unsupported"
    if data[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return "unreadable"
        if "xl/workbook.xml" in names:
            return "xlsx"
        if "word/document.xml" in names:
            return "unsupported"
        return "unreadable"
    suffix = posixpath.splitext((filename or "").lower())[1]
    if suffix == ".csv":
        return "csv"
    return "unknown"


def parse_document(data: bytes, filename: str,
                   limits: Optional[Limits] = None) -> Document:
    """Read one .xlsx or .csv table file. Raises ParseError on failure."""
    limits = limits or Limits()
    suffix = posixpath.splitext((filename or "").lower())[1]
    if suffix and suffix not in SUPPORTED_SUFFIXES:
        raise ParseError(
            REASON_UNSUPPORTED,
            "不支持的文件格式 %r，仅支持 .csv 和 .xlsx" % suffix,
        )
    kind = sniff_kind(data, filename)

    if kind == "xlsx":
        return read_xlsx(data, limits)
    if kind == "csv":
        return read_csv(data, limits)
    if kind == "unsupported":
        raise ParseError(
            REASON_UNSUPPORTED,
            "不支持的文件格式，仅支持 .csv 和 .xlsx",
        )
    if suffix in SUPPORTED_SUFFIXES:
        raise ParseError(REASON_UNREADABLE,
                         "扩展名是 %s，但内容无法按该格式解析，文件可能已损坏" % suffix)
    raise ParseError(
        REASON_UNSUPPORTED,
        "不支持的文件格式 %r，仅支持 .csv 和 .xlsx" % (suffix or "（无扩展名）"),
    )


# ── export writers (.csv / .xlsx) ────────────────────────────────────────────

_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_IMAGE_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
}


def _xml_text(value: str) -> str:
    return _xml_escape(_XML_ILLEGAL.sub("", value))


def write_csv(headers: List[str], rows: List[List[Any]]) -> bytes:
    """Serialize headers + rows into UTF-8-BOM CSV bytes."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(["" if val is None else str(val) for val in row])
    return buf.getvalue().encode("utf-8-sig")


def write_xlsx(sheet_name: str, headers: List[str], rows: List[List[Any]],
               images: Optional[Dict[Tuple[int, int], Tuple[bytes, str]]] = None) -> bytes:
    """Serialize headers + rows (+ optional embedded cell images) into a valid .xlsx.

    `images` maps `(row_1based, col_1based)` (where row 1 is the header row and
    data rows start at 2) to `(image_bytes, ext)` where `ext` is e.g. `"png"` or `"jpeg"`.
    """
    clean_sheet = re.sub(r"[:\\/?*\[\]]", "_", (sheet_name or "Sheet1")).strip()[:31] or "Sheet1"
    images = images or {}

    row_xml: List[str] = []
    # Row 1: header
    header_cells = []
    for col_idx, name in enumerate(headers, 1):
        ref = "%s1" % column_letter(col_idx)
        header_cells.append('<c r="%s" t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>'
                            % (ref, _xml_text(str(name))))
    row_xml.append('<row r="1">%s</row>' % "".join(header_cells))

    # Data rows (starting at row 2)
    for r_offset, row in enumerate(rows, 2):
        cells = []
        for col_idx, val in enumerate(row, 1):
            if val is None:
                continue
            ref = "%s%d" % (column_letter(col_idx), r_offset)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                cells.append('<c r="%s" t="n"><v>%s</v></c>' % (ref, val))
            else:
                cells.append('<c r="%s" t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>'
                             % (ref, _xml_text(str(val))))
        if cells:
            row_xml.append('<row r="%d">%s</row>' % (r_offset, "".join(cells)))

    drawing_tag = '<drawing r:id="rId1"/>' if images else ""
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetData>%s</sheetData>%s</worksheet>'
    ) % ("".join(row_xml), drawing_tag)

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="%s" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    ) % _xml_text(clean_sheet)

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )

    used_exts = set()
    anchors = []
    drawing_rels = []
    media_files: List[Tuple[str, bytes]] = []
    for idx, ((row_num, col_num), (img_bytes, raw_ext)) in enumerate(sorted(images.items()), 1):
        ext = (raw_ext or "png").lstrip(".").lower()
        if ext not in _IMAGE_MIME_BY_EXT:
            ext = "png"
        used_exts.add(ext)
        media_name = "image%d.%s" % (idx, ext)
        media_files.append(("xl/media/" + media_name, img_bytes))
        rid = "rId%d" % idx
        drawing_rels.append(
            '<Relationship Id="%s" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="../media/%s"/>' % (rid, media_name)
        )
        c0 = max(0, col_num - 1)
        r0 = max(0, row_num - 1)
        anchors.append(
            '<xdr:twoCellAnchor>'
            '<xdr:from><xdr:col>%d</xdr:col><xdr:colOff>0</xdr:colOff>'
            '<xdr:row>%d</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>'
            '<xdr:to><xdr:col>%d</xdr:col><xdr:colOff>0</xdr:colOff>'
            '<xdr:row>%d</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>'
            '<xdr:pic>'
            '<xdr:nvPicPr><xdr:cNvPr id="%d" name="%s"/><xdr:cNvPicPr/></xdr:nvPicPr>'
            '<xdr:blipFill><a:blip r:embed="%s"/>'
            '<a:stretch><a:fillRect/></a:stretch></xdr:blipFill>'
            '<xdr:spPr><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></xdr:spPr>'
            '</xdr:pic><xdr:clientData/>'
            '</xdr:twoCellAnchor>' % (c0, r0, c0 + 1, r0 + 1, idx + 1, media_name, rid)
        )

    ext_defaults = "".join(
        '<Default Extension="%s" ContentType="%s"/>' % (ext, _IMAGE_MIME_BY_EXT[ext])
        for ext in sorted(used_exts)
    )
    drawing_override = (
        '<Override PartName="/xl/drawings/drawing1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
        if images else ""
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '%s'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '%s'
        '</Types>'
    ) % (ext_defaults, drawing_override)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        if images:
            sheet_rels = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
                'Target="../drawings/drawing1.xml"/>'
                '</Relationships>'
            )
            drawing_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '%s</xdr:wsDr>'
            ) % "".join(anchors)
            drawing_rels_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '%s</Relationships>'
            ) % "".join(drawing_rels)
            zf.writestr("xl/worksheets/_rels/sheet1.xml.rels", sheet_rels)
            zf.writestr("xl/drawings/drawing1.xml", drawing_xml)
            zf.writestr("xl/drawings/_rels/drawing1.xml.rels", drawing_rels_xml)
            for media_path, img_bytes in media_files:
                zf.writestr(media_path, img_bytes)
    return buf.getvalue()
