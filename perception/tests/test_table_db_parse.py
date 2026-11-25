"""
tests/test_table_db_parse.py — reading .xlsx / .docx / .csv with no dependencies.

The archives come from `ooxml_fixtures`, which builds real OOXML parts, so these
tests exercise the same zip, the same shared-string table and the same drawing
anchors a robot sees. Nothing here needs a model, ROS, or a third-party library.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

from datetime import date

import pytest

from ooxml_fixtures import (
    EMF_BYTES,
    PNG_BYTES,
    build_docx,
    build_xlsx,
    dispimg,
    num,
    paragraph,
    sh,
    table,
    xs,
)

from plugins.table_db_parse import (
    Limits,
    ParseError,
    clean_fields,
    column_letter,
    normalise_header,
    parse_document,
    resolve_layout,
    sniff_kind,
    table_records,
    write_csv,
    write_xlsx,
)


def _read(sheet, fields=None, header_row=None):
    layout = resolve_layout(sheet, clean_fields(fields), header_row)
    return layout, table_records(sheet, layout)

# The 1900 system's epoch, restated here so the test pins it: Excel counts from
# 1899-12-30, not 1900-01-01 (Lotus 1-2-3's phantom 29 February 1900).
EPOCH = date(1899, 12, 30)


def _sheet_2001(value=date(2001, 2, 3)) -> int:
    return (value - EPOCH).days


# ── xlsx ─────────────────────────────────────────────────────────────────────

def test_xlsx_reads_inline_and_shared_strings():
    data = build_xlsx(rows=[
        [xs("姓名"), xs("身份证号")],
        [xs("张三"), sh(0)],
    ], shared=["110101199001011234"])

    document = parse_document(data, "roster.xlsx")
    layout, records = _read(document.tables[0])

    assert document.kind == "xlsx"
    assert layout.header_row == 0
    assert layout.columns() == ["姓名", "身份证号"]
    assert records == [{"_row": 2, "姓名": "张三", "身份证号": "110101199001011234"}]


def test_xlsx_header_below_a_title_row_is_still_found():
    """Rosters open with a merged title, so the header is not row 1.

    The alias test is what makes this work: "row 2 is all text" would also match
    the data, and "row 1 is non-empty" would consume the title as the header.
    """
    data = build_xlsx(rows=[
        [xs("2026年第三期培训考试名单")],
        [xs("姓名"), xs("身份证号")],
        [xs("张三"), xs("110101199001011234")],
    ])

    sheet = parse_document(data, "roster.xlsx").tables[0]
    layout, records = _read(sheet)

    assert layout.header_sheet_row(sheet) == 2
    assert records == [{"_row": 3, "姓名": "张三", "身份证号": "110101199001011234"}]


def test_header_with_only_business_words_is_found_by_its_shape():
    """`借用人 | 事项 | 物品` matches no built-in alias for the first two columns;
    a row of short distinct labels above data is still a header."""
    data = build_xlsx(rows=[
        [xs("借还台账")],
        [xs("借用人"), xs("事项"), xs("器材")],
        [xs("张三"), xs("借出"), xs("万用表")],
    ])

    sheet = parse_document(data, "ledger.xlsx").tables[0]
    layout, records = _read(sheet)

    assert layout.header_sheet_row(sheet) == 2
    assert records[0] == {"_row": 3, "借用人": "张三", "事项": "借出", "器材": "万用表"}


def test_xlsx_header_decorations_are_tidied_but_not_renamed():
    data = build_xlsx(rows=[
        [xs("*姓名"), xs("身份证号（18位）"), xs("照片：")],
        [xs("李四"), xs("110101199002021235"), xs("")],
    ])

    layout, records = _read(parse_document(data, "roster.xlsx").tables[0])

    assert layout.columns() == ["姓名", "身份证号（18位）", "照片"]
    assert records[0]["姓名"] == "李四"


def test_fields_pick_and_rename_columns_with_builtin_aliases():
    data = build_xlsx(rows=[
        [xs("*姓名"), xs("身份证号（18位）"), xs("考场"), xs("座位号")],
        [xs("王五"), xs("110101199003031236"), xs("三号"), xs("12")],
    ])

    layout, records = _read(parse_document(data, "roster.xlsx").tables[0],
                            fields={"name": [], "id": [], "seat": ["座位号"]})

    assert layout.columns() == ["name", "id", "seat"]
    assert records == [{"_row": 2, "name": "王五", "id": "110101199003031236",
                        "seat": "12"}]


def test_fields_own_alias_wins_and_unmatched_is_reported():
    data = build_xlsx(rows=[
        [xs("借用人"), xs("物品")],
        [xs("张三"), xs("万用表")],
    ])

    layout, records = _read(parse_document(data, "ledger.xlsx").tables[0],
                            fields={"name": ["借用人"], "item": [], "phone": []})

    assert records == [{"_row": 2, "name": "张三", "item": "万用表"}]
    assert layout.unmatched == ["phone"]


def test_fields_accept_a_column_letter_for_a_headerless_sheet():
    data = build_xlsx(rows=[
        [xs("张三"), xs("110101199001011234")],
        [xs("李四"), xs("110101199002021235")],
    ])

    layout, records = _read(parse_document(data, "plain.xlsx").tables[0],
                            fields={"name": ["A"], "id": ["B"]}, header_row=0)

    assert [record["name"] for record in records] == ["张三", "李四"]
    assert records[1]["id"] == "110101199002021235"


def test_fields_list_shorthand_means_builtin_aliases():
    assert clean_fields(["name", "id"]) == {"name": [], "id": []}
    assert clean_fields({"name": "姓名"}) == {"name": ["姓名"]}
    assert clean_fields("nonsense") == {}


def test_explicit_header_row_overrides_detection_and_is_validated():
    data = build_xlsx(rows=[
        [xs("姓名"), xs("备注")],          # looks like a header, but is not
        [xs("name"), xs("note")],
        [xs("张三"), xs("x")],
    ])
    sheet = parse_document(data, "odd.xlsx").tables[0]

    layout, records = _read(sheet, header_row=2)
    assert layout.columns() == ["name", "note"]
    assert records == [{"_row": 3, "name": "张三", "note": "x"}]

    with pytest.raises(ParseError) as caught:
        _read(sheet, header_row=9)
    assert caught.value.reason == "bad_header_row"


def test_picture_column_in_fields_reads_back_as_the_path():
    data = build_xlsx(rows=[
        [xs("姓名"), xs("照片")],
        [xs("张三"), dispimg("ID_ABC")],
    ], cell_images=[("ID_ABC", "wps1.png")], media={"wps1.png": PNG_BYTES})

    _, records = _read(parse_document(data, "wps.xlsx").tables[0],
                       fields={"name": [], "photo": []})

    assert records == [{"_row": 2, "name": "张三", "photo": "xl/media/wps1.png",
                        "image": "xl/media/wps1.png"}]


def test_xlsx_date_in_a_builtin_and_a_chinese_custom_format():
    serial = _sheet_2001()
    data = build_xlsx(rows=[
        [xs("日期")],
        [num(serial, 1)],          # built-in 14 = mm-dd-yy
        [num(serial, 2)],          # custom 164 = yyyy"年"m"月"d"日"
        [num(3.5)],                # General: untouched
    ], cell_xfs=[0, 14, 164],
        num_fmts=[(164, 'yyyy"年"m"月"d"日"')])

    sheet = parse_document(data, "dates.xlsx").tables[0]

    assert [row[1].text for row in sheet.rows] == ["日期", "2001-02-03",
                                                   "2001-02-03", "3.5"]


def test_xlsx_1904_date_system_is_honoured():
    serial = (date(2001, 2, 3) - date(1904, 1, 1)).days
    data = build_xlsx(rows=[[num(serial, 1)]], cell_xfs=[0, 14], date1904=True)

    assert parse_document(data, "mac.xlsx").tables[0].rows[0][1].text == "2001-02-03"


def test_xlsx_floating_picture_lands_on_the_anchors_row():
    """Excel anchors a picture in the row it starts in, and that is the row it
    belongs to — even when the picture's height spills over several rows."""
    data = build_xlsx(rows=[
        [xs("姓名"), xs("照片")],
        [xs("张三"), xs("")],
        [xs("李四"), xs("")],
    ], anchors=[(1, 1, "image1.png", "twoCell")])

    sheet = parse_document(data, "roster.xlsx").tables[0]
    assert sheet.rows[1][2].media == "xl/media/image1.png"
    assert sheet.rows[2].get(2) is None


def test_xlsx_one_cell_anchor_picture_is_read_too():
    data = build_xlsx(rows=[
        [xs("姓名")],
        [xs("张三")],
    ], anchors=[(1, 1, "image1.png", "oneCell")])

    sheet = parse_document(data, "roster.xlsx").tables[0]
    assert sheet.rows[1][2].media == "xl/media/image1.png"


def test_xlsx_wps_embedded_cell_picture():
    """WPS stores the picture once and refers to it from a `DISPIMG` formula;
    the formula is the only row-level link there is."""
    data = build_xlsx(rows=[
        [xs("姓名"), xs("照片")],
        [xs("张三"), dispimg("ID_ABC")],
        [xs("李四"), dispimg("ID_DEF")],
    ], cell_images=[("ID_ABC", "wps1.png"), ("ID_DEF", "wps2.png")],
        media={"wps1.png": PNG_BYTES, "wps2.png": PNG_BYTES})

    sheet = parse_document(data, "wps.xlsx").tables[0]
    assert sheet.rows[1][2].media == "xl/media/wps1.png"
    assert sheet.rows[2][2].media == "xl/media/wps2.png"
    assert sheet.rows[1][1].text == "张三"   # the DISPIMG cell has no text


def test_xlsx_multiple_sheets_keep_their_names_and_order():
    data = build_xlsx(sheets=[
        ("名单", [[xs("姓名")], [xs("张三")]]),
        ("签到", [[xs("姓名")], [xs("张三"), xs("到")]]),
    ])

    document = parse_document(data, "book.xlsx")
    assert [sheet.name for sheet in document.tables] == ["名单", "签到"]
    assert [len(sheet.rows) for sheet in document.tables] == [2, 2]


def test_xlsx_row_budget_truncates_and_says_so():
    rows = [[xs("第%d行" % index)] for index in range(1, 11)]
    document = parse_document(build_xlsx(rows=rows), "long.xlsx", Limits(max_rows=4))

    sheet = document.tables[0]
    assert len(sheet.rows) == 4
    assert sheet.truncated is True
    assert any("上限" in note for note in document.notes)


def test_xlsx_column_budget_marks_the_sheet_truncated():
    wide = [[xs("c%d" % index) for index in range(1, 6)]]
    document = parse_document(build_xlsx(rows=wide), "wide.xlsx", Limits(max_columns=3))

    assert document.tables[0].truncated is True
    assert sorted(document.tables[0].rows[0]) == [1, 2, 3]


# ── export writers (.csv / .xlsx round-trip) ─────────────────────────────────

def test_write_csv_and_read_back_round_trip():
    raw = write_csv(["col1", "col2"], [["a", 1], ["b,c", None]])
    doc = parse_document(raw, "out.csv")
    layout, records = _read(doc.tables[0])
    assert layout.columns() == ["col1", "col2"]
    assert records[0]["col1"] == "a" and records[0]["col2"] == "1"
    assert records[1]["col1"] == "b,c" and "col2" not in records[1]


def test_write_xlsx_with_embedded_images_round_trip():
    raw = write_xlsx(
        "sheet_a",
        ["col1", "col2", "img"],
        [["row1", 42, None], ["row2", 3.5, None]],
        images={(2, 3): (PNG_BYTES, "png")},
    )
    doc = parse_document(raw, "out.xlsx")
    layout, records = _read(doc.tables[0])
    assert layout.columns() == ["col1", "col2", "img"]
    assert records[0]["col1"] == "row1" and records[0]["col2"] == "42"
    assert records[0]["image"] == "xl/media/image1.png"
    assert records[1]["col1"] == "row2" and records[1]["col2"] == "3.5"
    assert "image" not in records[1]


# ── csv ───────────────────────────────────────────────────────────────────────

def test_csv_utf8_bom_is_not_part_of_the_first_header():
    data = "姓名,身份证号\n张三,110101199001011234\n".encode("utf-8-sig")

    layout, records = _read(parse_document(data, "roster.csv").tables[0])

    assert layout.columns() == ["姓名", "身份证号"]
    assert records[0]["姓名"] == "张三"


def test_csv_in_gb18030_from_a_chinese_excel():
    data = "姓名,身份证号\n张三,110101199001011234\n".encode("gb18030")

    _, records = _read(parse_document(data, "roster.csv").tables[0])

    assert records[0]["姓名"] == "张三"


def test_csv_semicolon_delimiter_is_sniffed():
    data = "姓名;身份证号\n张三;110101199001011234\n".encode("utf-8")

    _, records = _read(parse_document(data, "roster.csv").tables[0], fields=["name", "id"])

    assert records[0] == {"_row": 2, "name": "张三", "id": "110101199001011234"}


def test_csv_single_column_with_a_known_header():
    data = "姓名\n张三\n李四\n".encode("utf-8")

    _, records = _read(parse_document(data, "names.csv").tables[0])

    assert [record["姓名"] for record in records] == ["张三", "李四"]


def test_csv_single_column_without_header_keeps_every_name():
    data = "张三\n李四\n".encode("utf-8")

    layout, records = _read(parse_document(data, "names.csv").tables[0])

    assert layout.header_row is None
    assert [record["A"] for record in records] == ["张三", "李四"]


# ── refusals and shapes ──────────────────────────────────────────────────────

@pytest.mark.parametrize("filename,reason", [
    ("old.xls", "unsupported_format"),
    ("old.doc", "unsupported_format"),
    ("doc.docx", "unsupported_format"),
    ("data.tsv", "unsupported_format"),
    ("scan.pdf", "unsupported_format"),
    ("slides.pptx", "unsupported_format"),
    ("archive.zip", "unsupported_format"),
])
def test_unsupported_formats_name_the_reason(filename, reason):
    if filename.endswith((".xls", ".doc")):
        payload = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
    elif filename.endswith(".docx"):
        payload = build_docx([paragraph("x")])
    else:
        payload = b"not that kind of file"

    with pytest.raises(ParseError) as caught:
        parse_document(payload, filename)

    assert caught.value.reason == reason
    assert ".csv" in caught.value.detail and ".xlsx" in caught.value.detail


def test_a_broken_zip_says_the_file_is_damaged():
    with pytest.raises(ParseError) as caught:
        parse_document(b"PK\x03\x04garbage", "broken.xlsx")

    assert caught.value.reason == "unreadable"


def test_a_zip_that_is_not_xlsx_is_refused():
    assert sniff_kind(build_docx([paragraph("x")]), "x.docx") == "unsupported"
    with pytest.raises(ParseError) as caught:
        parse_document(_zip_without_office_parts(), "thing.xlsx")

    assert caught.value.reason == "unreadable"


def _zip_without_office_parts() -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "not a document")
    return buffer.getvalue()


def test_empty_csv_is_reported_as_empty():
    with pytest.raises(ParseError) as caught:
        parse_document(b"   \n\n", "empty.csv")

    assert caught.value.reason == "empty"


@pytest.mark.parametrize("label", [
    "姓名", "*姓名", "姓名:", "姓名 ", "姓名（必填）", " 姓名 ",
])
def test_header_normalisation_covers_template_decorations(label):
    assert normalise_header(label) == "姓名"


def test_header_key_does_not_shadow_the_position_fields():
    """A column literally called "_row" or "image" must not overwrite the
    row number or the picture path every record carries."""
    data = build_xlsx(rows=[
        [xs("姓名"), xs("_row"), xs("image")],
        [xs("张三"), xs("7"), xs("http://x/y.png")],
    ])

    _, records = _read(parse_document(data, "odd.xlsx").tables[0])

    assert records[0]["_row"] == 2
    assert records[0]["_row_col"] == "7"
    assert records[0]["image_col"] == "http://x/y.png"
    assert "image" not in records[0]


def test_duplicate_header_labels_get_suffixed_keys():
    data = build_xlsx(rows=[
        [xs("姓名"), xs("姓名")],
        [xs("张三"), xs("李四")],
    ])

    layout, _ = _read(parse_document(data, "dup.xlsx").tables[0])

    assert layout.columns() == ["姓名", "姓名_2"]


def test_headerless_sheet_keys_by_column_letter():
    data = build_xlsx(rows=[
        [xs("张三"), xs("110101199001011234")],
        [xs("李四"), xs("110101199002021235")],
    ])

    layout, records = _read(parse_document(data, "plain.xlsx").tables[0])

    assert layout.header_row is None
    assert records[0] == {"_row": 1, "A": "张三", "B": "110101199001011234"}
    assert column_letter(28) == "AB"


def test_empty_cells_and_blank_rows_are_omitted_but_row_numbers_are_real():
    data = build_xlsx(rows=[
        [xs("姓名"), xs("身份证号")],
        [xs("张三"), xs("")],
        [xs(""), xs("")],
        [xs("李四"), xs("110101199002021235")],
    ])

    _, records = _read(parse_document(data, "sparse.xlsx").tables[0])

    # `_row` is the row number the user sees in Excel, so "第 4 行是谁" is
    # answerable; the blank row in between is not a record.
    assert records == [{"_row": 2, "姓名": "张三"},
                       {"_row": 4, "姓名": "李四", "身份证号": "110101199002021235"}]


def test_emf_picture_is_reported_by_the_reader_as_media():
    """The runtime hands back the archive path even for an EMF image."""
    data = build_xlsx(
        rows=[[xs("col1"), xs("col2")], [xs("v1"), xs("")]],
        anchors=[(1, 1, "stamp.emf", "twoCell")],
        media={"stamp.emf": EMF_BYTES},
    )

    sheet = parse_document(data, "stamp.xlsx").tables[0]

    assert sheet.rows[1][2].media == "xl/media/stamp.emf"


# ── numbers as Excel shows them ──────────────────────────────────────────────

@pytest.mark.parametrize("raw,shown,lossy", [
    ("1.1010119900101101E+17", "110101199001011000", True),   # 18-digit ID typed as a number
    ("13800138000", "13800138000", False),                     # 11-digit phone: exact
    ("12.0", "12", False),
    ("3.5", "3.5", False),
    ("1E+3", "1000", False),
    ("abc", "abc", False),
])
def test_numbers_read_as_excel_shows_them(raw, shown, lossy):
    from plugins.table_db_parse import _plain_number

    assert _plain_number(raw) == (shown, lossy)


def test_lossy_long_numbers_are_reported_in_notes():
    data = build_xlsx(rows=[[xs("身份证号")], [num("1.1010119900101101E+17")]])
    document = parse_document(data, "ids.xlsx")

    assert document.tables[0].rows[1][1].text == "110101199001011000"
    assert any("A2" in note and "15" in note for note in document.notes)


def test_excel365_place_in_cell_picture_is_media_not_value_error():
    from ooxml_fixtures import build_xlsx_place_in_cell

    data = build_xlsx_place_in_cell(
        [["姓名", "照片"], ["张三", ("pic", 0)], ["李四", ("pic", 1)]],
        [("a.jpeg", PNG_BYTES), ("b.png", PNG_BYTES)])
    sheet = parse_document(data, "365.xlsx").tables[0]

    assert sheet.rows[1][2].media == "xl/media/a.jpeg" and sheet.rows[1][2].text == ""
    assert sheet.rows[2][2].media == "xl/media/b.png"
