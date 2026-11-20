"""
tests/ooxml_fixtures.py — build real .xlsx / .docx archives with the stdlib.

Why generate instead of committing samples
------------------------------------------

A committed .xlsx is a binary in a repo whose reviewer flags binaries on sight,
and it is opaque: nobody reviewing a parser can see that the fixture has a title
row above the header, a WPS `DISPIMG` picture, or a date in a Chinese number
format. These builders spell all of that out in the test that uses it.

The archives are real: same parts, same namespaces, same relationship graph a
producer writes. They are read back through the same zip and the same XML a
robot sees, so the only thing being simulated is the producer's choices — which
is exactly the set of things that varies in the wild.

The pictures are valid PNGs, built here rather than embedded as base64, so a
test that hands one to a decoder is handing it a real image.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib

# ── namespaces ───────────────────────────────────────────────────────────────

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_XDR = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_CT = "http://schemas.openxmlformats.org/package/2006/content-types"

_HEADER = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def tiny_png(red: int = 0x10, green: int = 0x80, blue: int = 0x40) -> bytes:
    """A valid 1x1 PNG. Real bytes, so a downstream decoder can open it."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixel = bytes((0x00, red & 0xFF, green & 0xFF, blue & 0xFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(pixel))
            + chunk(b"IEND", b""))


PNG_BYTES = tiny_png()
# EMF magic: Word writes these for pasted shapes, and neither cv2 nor PIL in
# this image can decode one.
EMF_BYTES = b"\x01\x00\x00\x00" + b"\x00" * 40


# ── xlsx cell constructors ───────────────────────────────────────────────────

def xs(text: str) -> tuple:
    """An inline-string cell (`<is><t>`), what openpyxl and WPS write."""
    return ("inline", text)


def sh(index: int) -> tuple:
    """A shared-string cell (`t="s"` + `<v>index`), what Excel writes."""
    return ("shared", index)


def num(value, style: int = 0) -> tuple:
    """A numeric cell, optionally with an xf index (0 = General)."""
    return ("number", value, style)


def dispimg(image_id: str) -> tuple:
    """A WPS embedded-cell picture: `=DISPIMG("ID_x",1)` in the cell."""
    return ("dispimg", image_id)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


# ── xlsx ─────────────────────────────────────────────────────────────────────

def build_xlsx(sheets=None, rows=None, shared=(), cell_xfs=(), num_fmts=(),
               anchors=(), cell_images=(), media=None, date1904=False) -> bytes:
    """Build a workbook.

    Exactly one of `sheets` / `rows` must be given, and they are named rather
    than positional on purpose: `build_xlsx([[cell, cell], [cell, cell]])`
    cannot say whether that is one sheet of two rows or two sheets of one row,
    and the friendly guess ("each element is a sheet") silently produces a
    three-sheet workbook when the caller meant one. That mistake is invisible in
    the test until an assertion about `tables[0]` fails for the wrong reason.

    `sheets`:     [(name, rows)] — two or more sheets, each with a visible name.
    `rows`:       [row] — one sheet. A row is a list of cell tuples.
    `shared`:     sharedStrings, for `sh(i)` cells.
    `cell_xfs`:   [numFmtId] — index i is the style `s="i"`.
    `num_fmts`:   [(numFmtId, formatCode)] — custom formats.
    `anchors`:    [(row0, col0, media_name, "twoCell"|"oneCell")] — a floating
                  picture whose top-left corner sits in that cell (0-based).
    `cell_images`: [("ID_x", media_name)] — a WPS embedded-cell picture.
    `media`:      {name: bytes}, default one PNG at "image1.png".
    """
    if (sheets is None) == (rows is None):
        raise TypeError("build_xlsx needs exactly one of sheets= or rows=")
    if rows is not None:
        named = [("Sheet1", rows)]
    else:
        named = []
        for entry in sheets:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("each sheets entry must be (name, rows): %r"
                                % (entry,))
            named.append((entry[0], entry[1]))
    media = dict(media or {"image1.png": PNG_BYTES})
    parts = {}

    parts["[Content_Types].xml"] = (
        _HEADER + '<Types xmlns="%s">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        "</Types>" % _CT)
    parts["_rels/.rels"] = (
        _HEADER + '<Relationships xmlns="%s">'
        '<Relationship Id="rId1" Type="%s/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>' % (_PKG_REL, _REL))

    sheet_entries = []
    workbook_rels = []
    for index, (name, _rows) in enumerate(named, 1):
        sheet_entries.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>'
                             % (_escape(name), index, index))
        workbook_rels.append('<Relationship Id="rId%d" Type="%s/worksheet" '
                             'Target="worksheets/sheet%d.xml"/>'
                             % (index, _REL, index))
    next_rid = len(named) + 1
    if shared:
        workbook_rels.append('<Relationship Id="rId%d" Type="%s/sharedStrings" '
                             'Target="sharedStrings.xml"/>' % (next_rid, _REL))
        next_rid += 1
    if cell_xfs:
        workbook_rels.append('<Relationship Id="rId%d" Type="%s/styles" '
                             'Target="styles.xml"/>' % (next_rid, _REL))
        next_rid += 1

    parts["xl/workbook.xml"] = (
        _HEADER + '<workbook xmlns="%s" xmlns:r="%s"><workbookPr%s/><sheets>%s'
        "</sheets></workbook>"
        % (_MAIN, _REL, ' date1904="1"' if date1904 else "", "".join(sheet_entries)))
    parts["xl/_rels/workbook.xml.rels"] = (
        _HEADER + '<Relationships xmlns="%s">%s</Relationships>'
        % (_PKG_REL, "".join(workbook_rels)))

    for index, (_name, rows) in enumerate(named, 1):
        # A drawing is related from the *sheet*, not from the workbook, and the
        # fixture puts it on the first sheet only — the ordinary shape.
        drawing = bool(anchors) and index == 1
        parts["xl/worksheets/sheet%d.xml" % index] = _sheet_xml(
            rows, "rId1" if drawing else "")
        if drawing:
            parts["xl/worksheets/_rels/sheet%d.xml.rels" % index] = (
                _HEADER
                + ('<Relationships xmlns="%s">' % _PKG_REL)
                + '<Relationship Id="rId1" Type="%s/drawing" '
                  'Target="../drawings/drawing1.xml"/>' % _REL
                + "</Relationships>")

    if shared:
        items = "".join("<si><t>%s</t></si>" % _escape(item) for item in shared)
        parts["xl/sharedStrings.xml"] = (
            _HEADER + '<sst xmlns="%s" count="%d" uniqueCount="%d">%s</sst>'
            % (_MAIN, len(shared), len(shared), items))

    if cell_xfs:
        formats = "".join('<numFmt numFmtId="%d" formatCode="%s"/>'
                          % (fmt_id, _escape(code)) for fmt_id, code in num_fmts)
        xfs = "".join('<xf numFmtId="%d" fontId="0" fillId="0" borderId="0"/>'
                      % int(entry) for entry in cell_xfs)
        parts["xl/styles.xml"] = (
            _HEADER + '<styleSheet xmlns="%s">'
            '<numFmts count="%d">%s</numFmts>'
            '<cellXfs count="%d">%s</cellXfs></styleSheet>'
            % (_MAIN, len(num_fmts), formats, len(cell_xfs), xfs))

    if cell_images:
        parts["xl/cellimages.xml"] = _cellimages_xml(cell_images)
        parts["xl/_rels/cellimages.xml.rels"] = (
            _HEADER
            + ('<Relationships xmlns="%s">' % _PKG_REL)
            + "".join('<Relationship Id="rId%d" Type="%s/image" Target="media/%s"/>'
                      % (index, _REL, name)
                      for index, (_id, name) in enumerate(cell_images, 1))
            + "</Relationships>")

    if anchors:
        parts["xl/drawings/drawing1.xml"] = _drawing_xml(anchors)
        parts["xl/drawings/_rels/drawing1.xml.rels"] = (
            _HEADER
            + ('<Relationships xmlns="%s">' % _PKG_REL)
            + "".join('<Relationship Id="rId%d" Type="%s/image" '
                      'Target="../media/%s"/>' % (index, _REL, name)
                      for index, (_row, _col, name, _kind)
                      in enumerate(anchors, 1))
            + "</Relationships>")

    for name, blob in media.items():
        parts["xl/media/%s" % name] = blob

    return _zip(parts)


def _cell_xml(ref: str, cell) -> str:
    kind = cell[0] if isinstance(cell, tuple) else "inline"
    if kind == "inline":
        text = cell[1] if isinstance(cell, tuple) else cell
        return ('<c r="%s" t="inlineStr"><is><t>%s</t></is></c>'
                % (ref, _escape(text)))
    if kind == "shared":
        return '<c r="%s" t="s"><v>%d</v></c>' % (ref, int(cell[1]))
    if kind == "number":
        value, style = cell[1], cell[2]
        attr = ' s="%d"' % style if style else ""
        return '<c r="%s"%s><v>%s</v></c>' % (ref, attr, value)
    if kind == "dispimg":
        return ('<c r="%s"><f>DISPIMG("%s",1)</f><v>0</v></c>'
                % (ref, _escape(cell[1])))
    raise AssertionError("unknown cell kind %r" % (kind,))


def _sheet_xml(rows, drawing_rid: str = "") -> str:
    parts = [_HEADER,
             '<worksheet xmlns="%s" xmlns:r="%s"><sheetData>' % (_MAIN, _REL)]
    for row_index, row in enumerate(rows, 1):
        parts.append('<row r="%d">' % row_index)
        for col_index, cell in enumerate(row, 1):
            parts.append(_cell_xml("%s%d" % (_column_letter(col_index), row_index),
                                   cell))
        parts.append("</row>")
    parts.append("</sheetData>")
    if drawing_rid:
        parts.append('<drawing xmlns:r="%s" r:id="%s"/>' % (_REL, drawing_rid))
    parts.append("</worksheet>")
    return "".join(parts)


def _cellimages_xml(cell_images) -> str:
    """WPS's `xl/cellimages.xml`: one picture per `DISPIMG` id, keyed by name.

    The root is WPS's own namespace, which is exactly why the reader matches on
    local names instead of on a URI.
    """
    pictures = []
    for index, (image_id, _name) in enumerate(cell_images, 1):
        pictures.append(
            '<xdr:pic><xdr:nvPicPr><xdr:cNvPr id="%d" name="%s"/></xdr:nvPicPr>'
            '<xdr:blipFill><a:blip r:embed="rId%d"/></xdr:blipFill></xdr:pic>'
            % (index, _escape(image_id), index))
    return (_HEADER
            + '<etc:cellImages xmlns:etc="http://www.wps.cn/officeDocument/2017/etCustomData"'
            ' xmlns:a="%s" xmlns:r="%s" xmlns:xdr="%s">%s</etc:cellImages>'
            % (_A, _REL, _XDR, "".join(pictures)))


def _drawing_xml(anchors) -> str:
    blocks = []
    for index, (row, col, _name, kind) in enumerate(anchors, 1):
        one_cell = kind == "oneCell"
        tag = "xdr:oneCellAnchor" if one_cell else "xdr:twoCellAnchor"
        if one_cell:
            extent = '<xdr:ext cx="9525" cy="9525"/>'
        else:
            extent = ("<xdr:to><xdr:row>%d</xdr:row><xdr:col>%d</xdr:col>"
                      "</xdr:to>" % (row + 1, col + 1))
        blocks.append(
            "<%s><xdr:from><xdr:row>%d</xdr:row><xdr:col>%d</xdr:col></xdr:from>%s"
            '<xdr:pic><xdr:nvPicPr><xdr:cNvPr id="%d" name="Picture %d"/>'
            "</xdr:nvPicPr>"
            '<xdr:blipFill><a:blip r:embed="rId%d"/></xdr:blipFill></xdr:pic>'
            "</%s>"
            % (tag, row, col, extent, index, index, index, tag))
    return (_HEADER + '<xdr:wsDr xmlns:xdr="%s" xmlns:a="%s" xmlns:r="%s">%s'
            "</xdr:wsDr>" % (_XDR, _A, _REL, "".join(blocks)))


# ── docx ─────────────────────────────────────────────────────────────────────

def paragraph(text: str = "", style: str = "", image: str = "") -> dict:
    return {"type": "p", "text": text, "style": style, "image": image}


def table(rows) -> dict:
    """A table; each cell is a string, or {"text": ..., "image": media}."""
    return {"type": "tbl", "rows": rows}


def build_docx(blocks, styles=(), media=None) -> bytes:
    """Build a Word document.

    `styles`: [(styleId, styleName)] in styles.xml — a heading is recognised by
              the style *name* (`heading 1` / `标题 1`), which is what the reader
              matches on.
    `media`:  {name: bytes}, default one PNG at "image1.png".
    """
    media = dict(media or {"image1.png": PNG_BYTES})
    parts = {}
    parts["[Content_Types].xml"] = (
        _HEADER + '<Types xmlns="%s">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        "</Types>" % _CT)
    parts["_rels/.rels"] = (
        _HEADER + '<Relationships xmlns="%s">'
        '<Relationship Id="rId1" Type="%s/officeDocument" '
        'Target="word/document.xml"/></Relationships>' % (_PKG_REL, _REL))

    body = []
    rels = []
    media_ids = {}

    def rel_for(target: str) -> str:
        if target not in media_ids:
            rid = "rId%d" % (100 + len(media_ids))
            media_ids[target] = rid
            rels.append('<Relationship Id="%s" Type="%s/image" Target="media/%s"/>'
                        % (rid, _REL, target))
        return media_ids[target]

    for block in blocks:
        if block.get("type") == "p":
            body.append(_docx_paragraph(block, rel_for))
        elif block.get("type") == "tbl":
            body.append(_docx_table(block["rows"], rel_for))
        else:
            raise AssertionError("unknown block %r" % (block.get("type"),))

    parts["word/document.xml"] = (
        _HEADER + '<w:document xmlns:w="%s" xmlns:r="%s" xmlns:a="%s" '
        'xmlns:wp="%s"><w:body>%s</w:body></w:document>'
        % (_W, _REL, _A, _WP, "".join(body)))
    parts["word/_rels/document.xml.rels"] = (
        _HEADER + '<Relationships xmlns="%s">%s</Relationships>'
        % (_PKG_REL, "".join(rels)))

    if styles:
        entries = "".join(
            '<w:style w:type="paragraph" w:styleId="%s"><w:name w:val="%s"/>'
            "</w:style>" % (_escape(style_id), _escape(name))
            for style_id, name in styles)
        parts["word/styles.xml"] = (
            _HEADER + '<w:styles xmlns:w="%s">%s</w:styles>' % (_W, entries))

    for name, blob in media.items():
        parts["word/media/%s" % name] = blob

    return _zip(parts)


def _picture_run(rid: str) -> str:
    return ('<w:r><w:drawing><wp:inline><a:graphic><a:graphicData>'
            '<a:blip r:embed="%s"/>'
            "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>" % rid)


def _docx_paragraph(block, rel_for) -> str:
    style = block.get("style") or ""
    properties = ('<w:pPr><w:pStyle w:val="%s"/></w:pPr>' % _escape(style)
                  if style else "")
    runs = []
    if block.get("text"):
        runs.append("<w:r><w:t>%s</w:t></w:r>" % _escape(block["text"]))
    if block.get("image"):
        runs.append(_picture_run(rel_for(block["image"])))
    return "<w:p>%s%s</w:p>" % (properties, "".join(runs))


def _docx_table(rows, rel_for) -> str:
    out = ["<w:tbl>"]
    for row in rows:
        out.append("<w:tr>")
        for cell in row:
            if isinstance(cell, str):
                text, image = cell, ""
            else:
                text, image = cell.get("text", ""), cell.get("image", "")
            inner = ""
            if text:
                inner += "<w:r><w:t>%s</w:t></w:r>" % _escape(text)
            if image:
                inner += _picture_run(rel_for(image))
            out.append("<w:tc><w:p>%s</w:p></w:tc>" % inner)
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


# ── shared ───────────────────────────────────────────────────────────────────

def _zip(parts) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            archive.writestr(name, content)
    return buffer.getvalue()


__all__ = [
    "EMF_BYTES",
    "PNG_BYTES",
    "build_docx",
    "build_xlsx",
    "dispimg",
    "num",
    "paragraph",
    "sh",
    "table",
    "tiny_png",
    "xs",
]
