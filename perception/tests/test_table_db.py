"""
tests/test_table_db.py — the `table_db` card as skills use it.

Each test pins one promise:

* import turns a spreadsheet (with pictures) into rows + stored files, all or
  nothing, with the row and column of whatever was wrong;
* append / query / update / delete / stats behave like a small database whose
  ids and counts the model never has to compute;
* files are stored as BLOBs, come back byte-identical via get_file, and
  export_files produces the package face_recognition.register_by_corpus reads;
* bad input never writes; nothing raises.

No ROS, model or network. Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

import pytest

from ooxml_fixtures import (PNG_BYTES, build_docx, build_xlsx, dispimg, paragraph, table,
                            xs)

from plugins.table_db import TOOLS, TableDbPlugin

PNG_B = PNG_BYTES + b"\x00b"

ROSTER = build_xlsx(rows=[
    [xs("2026年第三期考试名单")],
    [xs("姓名"), xs("准考证号"), xs("照片")],
    [xs("张三"), xs("0012"), xs("")],
    [xs("李四"), xs("0013"), xs("")],
    [xs("王五"), xs("0014"), xs("")],
], anchors=[(2, 2, "image1.png", "twoCell"), (3, 2, "image2.png", "oneCell")],
    media={"image1.png": PNG_BYTES, "image2.png": PNG_B})


def _card(tmp_path, **overrides) -> TableDbPlugin:
    options = {"data_dir": str(tmp_path / "db"), "image_roots": [str(tmp_path)]}
    options.update(overrides)
    return TableDbPlugin(options, executor=None)


def _write(tmp_path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def _call(card, action, **args):
    args.setdefault("db", "t")
    reply = card.dispatch("table_db", dict(action=action, **args))
    assert reply is not None
    return reply


def _ledger(card):
    _call(card, "create", table="台账", columns=[
        {"name": "借用人", "type": "text"}, {"name": "物品", "type": "text"},
        {"name": "状态", "type": "text"}, {"name": "照片", "type": "file"}])
    return _call(card, "append", table="台账", records=[
        {"借用人": "张三", "物品": "万用表", "状态": "借出"},
        {"借用人": "张三", "物品": "网线钳", "状态": "借出"},
        {"借用人": "李四", "物品": "万用表", "状态": "已归还"},
    ])


# ── import ───────────────────────────────────────────────────────────────────

def test_import_roster_with_pictures(tmp_path):
    card = _card(tmp_path)
    reply = _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))

    assert reply["ok"] is True and reply["imported"] == 3 and reply["files"] == 2
    assert reply["header_row"] == 2
    assert reply["columns"] == [{"name": "姓名", "type": "text"},
                                {"name": "准考证号", "type": "text"},
                                {"name": "照片", "type": "file"}]
    first = reply["sample"][0]
    assert first["_id"] == 1 and first["_source_row"] == 3
    assert first["准考证号"] == "0012"                 # leading zero kept
    assert first["照片"]["mime"] == "image/png" and first["照片"]["file_id"].startswith("f_")


def test_import_dry_run_writes_nothing(tmp_path):
    card = _card(tmp_path)
    reply = _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER),
                  dry_run=True)

    assert reply["ok"] is True and reply["dry_run"] is True and reply["imported"] == 0
    assert reply["rows"] == 3 and reply["sample"][0]["照片"]["size"] == len(PNG_BYTES)
    assert _call(card, "query", table="名单")["reason"] == "unknown_db"
    assert card.dispatch("table_db", {"action": "list"})["databases_count"] == 0


def test_import_wps_embedded_picture_and_csv(tmp_path):
    card = _card(tmp_path)
    wps = build_xlsx(rows=[[xs("姓名"), xs("照片")], [xs("张三"), dispimg("ID_A")]],
                     cell_images=[("ID_A", "a.png")], media={"a.png": PNG_BYTES})
    assert _call(card, "import", table="wps", path=_write(tmp_path, "w.xlsx", wps))["files"] == 1

    csv_path = _write(tmp_path, "b.csv", "书名,价格\n活着,55.00元\n兄弟,39\n".encode("gb18030"))
    reply = _call(card, "import", table="书单", path=csv_path)
    assert reply["imported"] == 2
    assert reply["columns"] == [{"name": "书名", "type": "text"}, {"name": "价格", "type": "text"}]


def test_import_word_table(tmp_path):
    card = _card(tmp_path)
    data = build_docx([table([["姓名", "照片"], ["王五", {"image": "image1.png"}]])])
    reply = _call(card, "import", table="名单", path=_write(tmp_path, "r.docx", data))

    assert reply["imported"] == 1 and reply["files"] == 1


def test_import_word_table_after_title_and_instructions(tmp_path):
    """The real Q5 roster: a heading and an instruction paragraph, then the table.
    Found on the robot — a reader that stopped after the first paragraph saw no
    table at all."""
    card = _card(tmp_path)
    data = build_docx([
        paragraph("考生名单（模板）", style="Heading1"),
        paragraph("每位考生一行；照片请插入到「照片」列。"),
        paragraph("第三段说明。"),
        table([["姓名", "准考证号", "照片"], ["王五", "ZK003", {"image": "image1.png"}]]),
    ], styles=[("Heading1", "heading 1")])
    reply = _call(card, "import", table="名单", path=_write(tmp_path, "r.docx", data))

    assert reply["ok"] is True and reply["imported"] == 1 and reply["files"] == 1


def test_import_duplicate_header_is_located_and_writes_nothing(tmp_path):
    card = _card(tmp_path)
    data = build_xlsx(rows=[[xs("姓名"), xs("电话"), xs("姓名")], [xs("张三"), xs("1"), xs("x")]])
    reply = _call(card, "import", table="名单", path=_write(tmp_path, "d.xlsx", data))

    assert reply["ok"] is False and reply["reason"] == "invalid_header"
    assert reply["row"] == 1 and reply["column"] == "姓名" and reply["written"] == 0
    assert "A 列和 C 列" in reply["detail"]


def test_import_failure_mid_batch_rolls_back_everything(tmp_path):
    card = _card(tmp_path, max_cell_chars=10)
    data = build_xlsx(rows=[[xs("姓名")], [xs("张三")], [xs("李四")], [xs("超" * 50)]])
    reply = _call(card, "import", table="名单", path=_write(tmp_path, "l.xlsx", data))

    assert reply["reason"] == "too_large" and reply["row"] == 4 and reply["column"] == "姓名"
    # The table was created inside the failed transaction, and the library
    # file that opening it created is removed too: nothing is left behind.
    assert _call(card, "query", table="名单")["reason"] == "unknown_db"
    assert card.dispatch("table_db", {"action": "list"})["databases_list"] == []


def test_import_append_checks_columns(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    more = build_xlsx(rows=[[xs("姓名"), xs("准考证号")], [xs("赵六"), xs("0015")]])
    ok = _call(card, "import", table="名单", path=_write(tmp_path, "m.xlsx", more), mode="append")
    assert ok["imported"] == 1
    assert _call(card, "stats", table="名单")["total"] == 4

    odd = build_xlsx(rows=[[xs("姓名"), xs("座位")], [xs("钱七"), xs("12")]])
    bad = _call(card, "import", table="名单", path=_write(tmp_path, "o.xlsx", odd), mode="append")
    assert bad["reason"] == "unknown_column" and bad["column"] == "座位"
    assert _call(card, "stats", table="名单")["total"] == 4


def test_import_refusals(tmp_path):
    card = _card(tmp_path)
    ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
    assert _call(card, "import", table="a", path=_write(tmp_path, "o.xls", ole))["reason"] == "legacy_format"
    assert _call(card, "import", table="a", path=_write(tmp_path, "e.csv", b""))["reason"] == "empty"
    assert _call(card, "import", table="a", path=_write(tmp_path, "x.xlsx", b"PK\x03\x04junk"))["reason"] == "unreadable"
    text_only = build_docx([])
    assert _call(card, "import", table="a", path=str(tmp_path / "missing.xlsx"))["reason"] == "bad_request"
    outside = _card(tmp_path, image_roots=[str(tmp_path / "only")])
    assert _call(outside, "import", table="a", path=_write(tmp_path, "r.xlsx", ROSTER))["reason"] == "bad_input"
    del text_only


# ── append / query / update / delete / stats ─────────────────────────────────

def test_append_query_and_paging(tmp_path):
    card = _card(tmp_path)
    assert _ledger(card)["ids"] == [1, 2, 3]

    page = _call(card, "query", table="台账", where={"借用人": "张三"}, limit=1)
    assert (page["total"], page["count"], page["has_more"], page["next_offset"]) == (2, 1, True, 1)
    assert page["records"][0]["物品"] == "万用表"
    either = _call(card, "query", table="台账", where={"物品": ["网线钳", "螺丝刀"]})
    assert [item["_id"] for item in either["records"]] == [2]
    no_photo = _call(card, "query", table="台账", where={"照片": None})
    assert no_photo["total"] == 3


def test_append_coerces_numbers_and_rejects_the_rest(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    assert _call(card, "append", table="台账", records={"借用人": 7})["ok"] is True

    bad = _call(card, "append", table="台账", records=[{"借用人": "a"}, {"借用人": True}])
    assert bad["reason"] == "bad_request" and bad["row"] == 2 and bad["column"] == "借用人"
    assert _call(card, "stats", table="台账")["total"] == 4      # nothing from the bad batch
    assert _call(card, "append", table="台账", records=[{"不存在": "x"}])["reason"] == "unknown_column"
    assert _call(card, "append", table="台账", records=[{"照片": "a.jpg"}])["reason"] == "bad_request"
    assert _call(card, "append", table="没有的表", records=[{"a": "b"}])["reason"] == "unknown_table"


def test_update_and_delete(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    updated = _call(card, "update", table="台账", _id=1, set={"状态": "已归还"})
    assert updated["record"]["状态"] == "已归还" and updated["record"]["借用人"] == "张三"

    cleared = _call(card, "update", table="台账", _id=1, set={"状态": None})
    assert "状态" not in cleared["record"]

    assert _call(card, "delete", table="台账", _id=2)["deleted"] == 2
    assert _call(card, "stats", table="台账")["total"] == 2
    gone = _call(card, "update", table="台账", _id=2, set={"状态": "x"})
    assert gone["reason"] == "not_found" and "删除" in gone["detail"]
    assert _call(card, "delete", table="台账", _id=99)["reason"] == "not_found"
    assert _call(card, "update", table="台账", _id="abc", set={"状态": "x"})["reason"] == "bad_request"


def test_stats_counts_and_groups(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    lent = _call(card, "stats", table="台账", where={"状态": "借出"}, group_by="借用人")

    assert lent["total"] == 2
    assert lent["groups"] == [{"value": "张三", "count": 2}]
    assert _call(card, "stats", table="台账", group_by="照片")["reason"] == "bad_request"


def test_where_rejects_operators_and_sql(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    assert _call(card, "query", table="台账", where={"状态": {"ne": "x"}})["reason"] == "bad_request"
    assert _call(card, "query", table="台账", where="1=1")["reason"] == "bad_request"
    injected = _call(card, "query", table="台账", where={"借用人": "张三' OR '1'='1"})
    assert injected["ok"] is True and injected["total"] == 0


def test_hostile_names_are_data_not_sql(tmp_path):
    card = _card(tmp_path)
    reply = _call(card, "create", table='x"; DROP TABLE _tables; --',
                  columns=[{"name": 'a" TEXT); --', "type": "text"}])
    assert reply["ok"] is True
    assert _call(card, "append", table='x"; DROP TABLE _tables; --',
                 records=[{'a" TEXT); --': "v"}])["inserted"] == 1
    assert _call(card, "create", table="_secret", columns=["a"])["reason"] == "bad_request"
    assert _call(card, "query", table="x", db="../etc")["reason"] == "bad_request"


def test_create_validation(tmp_path):
    card = _card(tmp_path)
    assert _call(card, "create", table="a", columns=[])["reason"] == "bad_request"
    assert _call(card, "create", table="a", columns=[{"name": "x", "type": "image"}])["reason"] == "bad_request"
    assert _call(card, "create", table="a", columns=["x", "x"])["reason"] == "bad_request"
    assert _call(card, "create", table="a", columns=["x"])["ok"] is True
    assert _call(card, "create", table="a", columns=["x"])["reason"] == "table_exists"


def test_databases_are_isolated(tmp_path):
    card = _card(tmp_path)
    _call(card, "create", table="名单", columns=["姓名"], db="skill-a")
    assert _call(card, "query", table="名单", db="skill-b")["reason"] == "unknown_db"


# ── files ────────────────────────────────────────────────────────────────────

def test_get_file_by_record_id_returns_identical_bytes(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    rid = _call(card, "query", table="名单", where={"姓名": "李四"})["records"][0]["_id"]
    got = _call(card, "get_file", table="名单", _id=rid)            # no column: only one file column

    assert got["ok"] is True and got["_id"] == rid and got["column"] == "照片"
    assert got["file"]["mime"] == "image/png"
    with open(got["path"], "rb") as handle:
        assert handle.read() == PNG_B
    assert got["path"].startswith(str(tmp_path / "db"))
    assert _call(card, "get_file", table="名单", _id=999)["reason"] == "not_found"
    no_photo = _call(card, "get_file", table="名单", _id=3)        # 王五 has no picture
    assert no_photo["reason"] == "not_found" and "attach_file" in no_photo["detail"]
    assert _call(card, "get_file", table="名单", _id="x")["reason"] == "bad_request"
    assert _call(card, "get_file", table="名单", _id=1, column="姓名")["reason"] == "bad_request"
    assert _call(card, "get_file", table="名单", _id=1, column="不存在")["reason"] == "unknown_column"
    assert _call(card, "get_file", table="名单")["missing"] == ["_id"]


def test_get_file_needs_column_when_a_table_has_two_file_columns(tmp_path):
    card = _card(tmp_path)
    _call(card, "create", table="档案", columns="姓名,照片:file,合同:file")
    rid = _call(card, "append", table="档案", records=[{"姓名": "张三"}])["ids"][0]
    _call(card, "attach_file", table="档案", _id=rid, column="合同",
          file=_write(tmp_path, "c.pdf", b"%PDF-1.4 x"))

    ask = _call(card, "get_file", table="档案", _id=rid)
    assert ask["reason"] == "bad_request" and ask["columns"] == ["照片", "合同"]
    got = _call(card, "get_file", table="档案", _id=rid, column="合同")
    assert got["file"]["mime"] == "application/pdf"
    assert _call(card, "create", table="纯文字", columns="a")["ok"]
    _call(card, "append", table="纯文字", records=[{"a": "1"}])
    assert _call(card, "get_file", table="纯文字", _id=1)["reason"] == "bad_request"


def test_get_file_still_accepts_the_old_file_id(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    fid = _call(card, "query", table="名单")["records"][0]["照片"]["file_id"]
    got = _call(card, "get_file", table="名单", file_id=fid)
    assert got["ok"] is True and got["file"]["file_id"] == fid
    assert _call(card, "get_file", table="名单", file_id="f_999")["reason"] == "not_found"


def test_get_file_url_is_on_the_robot_address_and_serves_the_file(tmp_path):
    card = _card(tmp_path, public_base="http://10.1.2.3:15720")
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    got = _call(card, "get_file", table="名单", _id=2)

    assert got["url"].startswith("http://10.1.2.3:15720/table_db/file/t/f_")
    assert "recognize_by_url" in got["how_to_use"]
    # The model's download goes to localhost, not the LAN address.
    assert "http://localhost:15720/table_db/file/t/" in got["agent_core"]["download"]
    status, body, mime, _ = card.handle_http_get(got["url"][len("http://10.1.2.3:15720"):])
    assert (status, body, mime) == (200, PNG_B, "image/png")


def test_get_file_url_falls_back_when_no_lan_address(tmp_path, monkeypatch):
    import plugins.table_db as module

    class NoNetwork:
        def __init__(self, *args):
            raise OSError("no route")

    monkeypatch.setattr(module.socket, "socket", NoNetwork)
    card = _card(tmp_path)
    assert card._public_url() == "http://localhost:15720"


def test_attach_any_kind_of_file_and_replace(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    pdf = b"%PDF-1.4\n" + os.urandom(2000)
    attached = _call(card, "attach_file", table="台账", _id=1, column="照片",
                     file=_write(tmp_path, "收据.pdf", pdf))
    assert attached["file"]["mime"] == "application/pdf" and attached["file"]["size"] == len(pdf)

    wav = b"RIFF\x00\x00\x00\x00WAVEfmt " + os.urandom(100)
    replaced = _call(card, "attach_file", table="台账", _id=1, column="照片",
                     file=_write(tmp_path, "note.wav", wav))
    record = _call(card, "query", table="台账", where={"_id": 1})["records"][0]
    assert record["照片"]["file_id"] == replaced["file"]["file_id"]
    assert record["照片"]["mime"] == "audio/wav"

    assert _call(card, "attach_file", table="台账", _id=1, column="物品",
                 file=_write(tmp_path, "a.png", PNG_BYTES))["reason"] == "bad_request"
    big = _card(tmp_path, max_file_bytes=10)
    assert _call(big, "attach_file", table="台账", _id=1, column="照片",
                 file=_write(tmp_path, "b.png", PNG_BYTES))["reason"] == "too_large"

    detached = _call(card, "update", table="台账", _id=1, set={"照片": None})
    assert "照片" not in detached["record"]


def test_identical_files_are_stored_once(tmp_path):
    import sqlite3

    card = _card(tmp_path)
    same = build_xlsx(rows=[[xs("姓名"), xs("照片")], [xs("a"), xs("")], [xs("b"), xs("")]],
                      anchors=[(1, 1, "image1.png", "oneCell"), (2, 1, "image1.png", "oneCell")])
    _call(card, "import", table="名单", path=_write(tmp_path, "s.xlsx", same))
    conn = sqlite3.connect(str(tmp_path / "db" / "t.db"))
    assert conn.execute("SELECT COUNT(*) FROM _blobs").fetchone()[0] == 1
    conn.close()


def test_export_files_builds_a_face_corpus(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    reply = _call(card, "export_files", table="名单", column="照片", name_column="姓名",
                  profile_columns=["准考证号"])

    assert reply["ok"] is True and reply["count"] == 2
    assert reply["skipped"] == [{"_id": 3, "reason": "没有文件"}]
    assert reply["dir"] in reply["next_call"]
    with open(os.path.join(reply["dir"], "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest[0] == {"file": "r000001.png", "name": "张三", "person": "名单-1",
                           "profile": {"准考证号": "0012"}}
    assert sorted(os.listdir(reply["dir"])) == ["manifest.json", "r000001.png", "r000002.png"]

    only = _call(card, "export_files", table="名单", column="照片", name_column="姓名",
                 where={"姓名": "王五"})
    assert only["count"] == 0 and only["dir"] is None
    assert _call(card, "export_files", table="名单", column="姓名",
                 name_column="姓名")["reason"] == "bad_request"


# ── show / drop_table / canvas-style input ───────────────────────────────────

def test_list_shows_every_library_and_table_without_parameters(tmp_path):
    card = _card(tmp_path)
    assert card.dispatch("table_db", {"action": "list"})["databases"] == []

    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    _call(card, "create", table="签到", columns=["姓名"], db="other")
    reply = card.dispatch("table_db", {"action": "list"})

    assert reply["ok"] is True and reply["count"] == 2
    by_db = {item["db"]: item for item in reply["databases"]}
    roster = by_db["t"]["tables"][0]
    assert roster["table"] == "名单" and roster["rows"] == 3 and roster["files"] == 2
    assert {"name": "照片", "type": "file"} in roster["columns"]
    assert by_db["other"]["tables"][0]["table"] == "签到"
    assert "query" in reply["next"]
    # The old name still works, so a skill written against `show` keeps running.
    assert card.dispatch("table_db", {"action": "show"})["count"] == 2


def test_create_in_a_new_library_says_it_was_created(tmp_path):
    card = _card(tmp_path)
    first = _call(card, "create", db="newlib", table="a", columns="姓名")
    second = _call(card, "create", db="newlib", table="b", columns="姓名")
    assert first["db_created"] is True and "db_created" not in second


def test_missing_required_field_says_which_and_points_at_show(tmp_path):
    card = _card(tmp_path)
    # Exactly what the canvas sent: an empty table field is simply absent, and
    # the always-filled `mode` dropdown and instance_id come along.
    reply = card.dispatch("table_db", {"action": "query", "mode": "create",
                                       "instance_id": "card-x", "table": "  "})
    assert reply["ok"] is False and reply["reason"] == "bad_request"
    assert reply["missing"] == ["table"] and "list" in reply["detail"]


def test_canvas_json_text_fields_are_parsed(tmp_path):
    card = _card(tmp_path)
    created = card.dispatch("table_db", {"action": "create", "db": "t", "table": "台账",
                                         "columns": "借用人,物品,照片:file"})
    assert created["columns"][2] == {"name": "照片", "type": "file"}
    _call(card, "append", table="台账", records='[{"借用人":"张三","物品":"万用表"}]')
    found = _call(card, "query", table="台账", where='{"借用人":"张三"}')
    assert found["total"] == 1
    assert _call(card, "update", table="台账", _id="1", set='{"物品":"电钻"}')["ok"] is True
    bad = _call(card, "query", table="台账", where="{借用人:张三}")
    assert bad["reason"] == "bad_request" and bad["column"] == "where" and "{'姓名':'张三'}" in bad["detail"]
    assert _call(card, "import", table="x", path=_write(tmp_path, "r.xlsx", ROSTER),
                 dry_run="true")["dry_run"] is True


def test_drop_table_needs_confirmation_and_frees_files(tmp_path):
    import sqlite3

    card = _card(tmp_path)
    path = _write(tmp_path, "r.xlsx", ROSTER)
    _call(card, "import", table="名单", path=path)
    assert _call(card, "drop_table", table="名单", confirm="名")["reason"] == "bad_request"

    dropped = _call(card, "drop_table", table="名单", confirm="名单")
    assert dropped["dropped"] is True and dropped["rows"] == 3 and dropped["files"] == 2
    assert _call(card, "query", table="名单")["reason"] == "unknown_table"
    conn = sqlite3.connect(str(tmp_path / "db" / "t.db"))
    assert conn.execute("SELECT COUNT(*) FROM _blobs").fetchone()[0] == 0
    conn.close()
    again = _call(card, "import", table="名单", path=path)
    assert again["imported"] == 3
    assert _call(card, "stats", table="名单")["total"] == 3


# ── the real Feishu roster (数据名单.xlsx, 2026-11-20) ─────────────────────────

def test_import_excel365_place_in_cell_pictures_and_long_ids(tmp_path):
    """The file that failed on the robot: pictures placed *in* cells (Excel 365,
    shown as #VALUE! to an old reader) and 18-digit IDs typed as numbers."""
    from ooxml_fixtures import build_xlsx_place_in_cell

    data = build_xlsx_place_in_cell([
        ["姓名", "身份证号", "准考证号", "照片"],
        ["游宇辰", ("num", "1.1010119900101101E+17"), "ZK2026001", ("pic", 0)],
        ["刘简恺", ("num", "1.10101199202022E+17"), "ZK2026002", ("pic", 1)],
    ], [("image1.jpeg", PNG_BYTES), ("image2.jpeg", PNG_B)])
    card = _card(tmp_path)
    reply = _call(card, "import", table="名单", path=_write(tmp_path, "数据名单.xlsx", data))

    assert reply["ok"] is True and reply["imported"] == 2 and reply["files"] == 2
    assert {"name": "照片", "type": "file"} in reply["columns"]
    first = reply["sample"][0]
    assert "#VALUE!" not in json.dumps(reply, ensure_ascii=False)
    assert first["身份证号"] == "110101199001011000"            # not 1.10...E+17
    assert any("15 位" in note for note in reply["warnings"])
    got = _call(card, "get_file", table="名单", _id=reply["sample"][1]["_id"])
    with open(got["path"], "rb") as handle:
        assert handle.read() == PNG_B                          # row 3 got picture 2


# ── robustness ───────────────────────────────────────────────────────────────

def test_data_survives_a_restart(tmp_path):
    _call(_card(tmp_path), "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    restarted = _card(tmp_path)

    reply = _call(restarted, "query", table="名单")
    assert reply["total"] == 3
    got = _call(restarted, "get_file", table="名单", _id=reply["records"][0]["_id"])
    with open(got["path"], "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == hashlib.sha256(PNG_BYTES).hexdigest()


def test_concurrent_appends_get_distinct_ids(tmp_path):
    card = _card(tmp_path)
    _call(card, "create", table="签到", columns=["姓名"])
    results = []

    def worker(index):
        results.append(_call(card, "append", table="签到", records=[{"姓名": "p%d" % index}]))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(item["ok"] for item in results)
    assert sorted(item["ids"][0] for item in results) == list(range(1, 21))
    assert _call(card, "stats", table="签到")["total"] == 20


def test_bad_arguments_never_raise(tmp_path):
    card = _card(tmp_path)
    for args in ({}, {"table": 5}, {"table": ""}, {"table": "a", "limit": "x"},
                 {"table": "a", "offset": -1}):
        reply = card.dispatch("table_db", dict(action="query", **args))
        assert reply["ok"] is False and reply["written"] == 0
    # No action at all, or an unknown one: declined (perception reports it), no crash.
    assert card.dispatch("table_db", None) is None
    assert card.dispatch("table_db", {"action": "nope"}) is None


def test_internal_errors_are_answered_not_raised(tmp_path, monkeypatch):
    card = _card(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk on fire")

    _call(card, "create", table="a", columns="x")
    monkeypatch.setattr(card._store, "stats", boom)
    reply = _call(card, "stats", table="a")
    assert reply["reason"] == "internal_error" and "disk on fire" in reply["detail"]


def test_info_reports_tables_and_integrity(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    info = _call(card, "info")
    assert info["ok"] is True and info["databases"] == ["t"]
    assert info["tables"][0]["table"] == "台账" and info["tables"][0]["rows"] == 3
    assert info["integrity"] == "ok"


# ── schema ───────────────────────────────────────────────────────────────────

def test_schema_declares_actions_params_and_required():
    schema = TOOLS[0]["inputSchema"]
    split = schema["x-action-params"]
    visible = [action for action in schema["properties"]["action"]["enum"]
               if action not in ("start", "stop", "info")]

    # `show` first: it is what the canvas dropdown selects when the card opens.
    assert visible == ["list", "query", "stats", "import", "create", "append", "update",
                       "delete", "attach_file", "get_file", "export_files", "drop_table", "drop_db"]
    options = schema["properties"]["action"]["oneOf"]
    assert {item["const"] for item in options} == set(visible)
    # Bare names in the canvas dropdown; the explanation lives in description.
    assert all("title" not in item and item["description"] for item in options)
    assert split["list"] == {**split["list"], "params": [], "required": []}
    for action in visible:
        spec = split[action]
        if action not in ("list", "drop_db"):
            assert "table" in spec["params"] and "table" in spec["required"]
        for param in spec["params"]:
            assert param in schema["properties"], (action, param)
        assert set(spec["required"]) <= set(spec["params"])
    # agent-core pushes a file to this container only for `format: file` params.
    assert schema["properties"]["path"]["format"] == "file"


def test_every_field_placeholder_starts_with_something_useful():
    """The canvas shows only the first 40 characters of a description."""
    props = TOOLS[0]["inputSchema"]["properties"]
    for key in ("where", "set", "records", "columns", "_id", "column", "table"):
        head = props[key]["description"][:40]
        assert head.startswith("例"), (key, head)


def test_schema_is_valid_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft7Validator.check_schema(TOOLS[0]["inputSchema"])


# ── the 2026-11-20 canvas review ─────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("姓名,状态,现场照:file", ["text", "text", "file"]),
    ("姓名，状态，现场照：file", ["text", "text", "file"]),        # Chinese IME punctuation
    ("[姓名,状态,现场照:file]", ["text", "text", "file"]),        # brackets, no quotes
    ('["姓名","状态"]', ["text", "text"]),
    ('[{"name":"姓名","type":"text"},{"name":"照片","type":"file"}]', ["text", "file"]),
    ('[{“name”:“姓名”,“type”:“text”}]', ["text"]),               # curly quotes
])
def test_columns_accept_every_way_people_type_them(tmp_path, text, expected):
    card = _card(tmp_path)
    reply = card.dispatch("table_db", {"action": "create", "table": "t", "columns": text})
    assert reply["ok"] is True, reply
    assert [item["type"] for item in reply["columns"]] == expected


def test_request_id_is_ignored_not_an_error(tmp_path):
    card = _card(tmp_path)
    path = _write(tmp_path, "r.xlsx", ROSTER)
    assert _call(card, "import", table="名单", path=path, request_id="x")["imported"] == 3
    assert "request_id" not in TOOLS[0]["inputSchema"]["properties"]


def test_import_and_attach_use_separate_file_fields(tmp_path):
    props = TOOLS[0]["inputSchema"]["properties"]
    assert props["path"]["accept"] == ".xlsx,.xlsm,.csv,.tsv,.docx,.docm"
    assert "accept" not in props["file"] and props["file"]["format"] == "file"
    split = TOOLS[0]["inputSchema"]["x-action-params"]
    assert "path" in split["import"]["params"] and "file" in split["attach_file"]["params"]
    # The old field name still works for attach_file.
    card = _card(tmp_path)
    _ledger(card)
    old = _call(card, "attach_file", table="台账", _id=1, column="照片",
                path=_write(tmp_path, "a.png", PNG_BYTES))
    assert old["ok"] is True


def test_get_file_explains_how_agent_core_can_read_it(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    got = _call(card, "get_file", table="名单", _id=2)

    hint = got["agent_core"]
    assert hint["local_path"].startswith("/tmp/") and hint["local_path"] in hint["download"]
    url = hint["download"].split()[-1].strip("'")
    assert url.startswith("http://localhost:15720/table_db/file/t/")
    status, body, mime, name = card.handle_http_get(url[len("http://localhost:15720"):])
    assert (status, body, mime) == (200, PNG_B, "image/png") and name

    assert card.handle_http_get("/other/route") is None
    assert card.handle_http_get("/table_db/file/t/f_999")[0] == 404
    assert card.handle_http_get("/table_db/file/../etc/passwd")[0] == 400
    assert card.handle_http_get("/table_db/file/t/../../x")[0] == 400


def test_export_reply_says_in_chinese_what_to_do_next(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    reply = _call(card, "export_files", table="名单", column="照片", name_column="姓名")
    assert "register_by_corpus" in reply["next"] and "package" in reply["next"]
    assert reply["dir"] in reply["next_call"]


# ── 2026-11-22: drop_db, list summary, db must exist, mode, quotes ───────────

def test_list_starts_with_a_short_summary(tmp_path):
    card = _card(tmp_path)
    _call(card, "create", db="a", table="t1", columns="x")
    _call(card, "create", db="a", table="t2", columns="x")
    _call(card, "create", db="b", table="名单", columns="姓名")
    reply = card.dispatch("table_db", {"action": "list"})

    assert list(reply)[:4] == ["ok", "databases_count", "databases_list", "tables_list"]
    assert reply["databases_count"] == 2 and reply["databases_list"] == ["a", "b"]
    assert reply["tables_list"] == [{"db": "a", "count": 2, "table_list": ["t1", "t2"]},
                                    {"db": "b", "count": 1, "table_list": ["名单"]}]
    assert reply["databases"][0]["tables"][0]["columns"] == [{"name": "x", "type": "text"}]


def test_drop_db_removes_every_table_and_needs_the_name_twice(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", db="exam", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    _call(card, "create", db="exam", table="签到", columns="姓名")
    fid = _call(card, "query", db="exam", table="名单")["records"][0]["照片"]["file_id"]
    _call(card, "get_file", db="exam", table="名单", file_id=fid)        # fills the cache

    assert _call(card, "drop_db", db="exam")["missing"] == ["confirm"]
    assert _call(card, "drop_db", db="exam", confirm="exa")["reason"] == "bad_request"
    assert card.dispatch("table_db", {"action": "drop_db", "confirm": "default"})["missing"] == ["db"]
    dropped = _call(card, "drop_db", db="exam", confirm="exam")

    assert dropped["dropped"] is True and dropped["tables"] == 2
    assert dropped["table_list"] == ["名单", "签到"] and dropped["rows"] == 3 and dropped["files"] == 2
    assert card.dispatch("table_db", {"action": "list"})["databases_list"] == []
    assert not (tmp_path / "db" / "exam.db").exists()
    assert not (tmp_path / "db" / "cache" / "exam").exists()
    assert _call(card, "query", db="exam", table="名单")["reason"] == "unknown_db"
    assert _call(card, "drop_db", db="exam", confirm="exam")["reason"] == "unknown_db"


@pytest.mark.parametrize("action,extra", [
    ("query", {}), ("stats", {}), ("append", {"records": "[{'x':'1'}]"}),
    ("update", {"_id": 1, "set": "{'x':'2'}"}), ("delete", {"_id": 1}),
    ("get_file", {"_id": 1}), ("drop_table", {"confirm": "t"}),
    ("export_files", {"column": "照片", "name_column": "姓名"}),
])
def test_only_create_and_import_may_create_a_library(tmp_path, action, extra):
    card = _card(tmp_path)
    reply = card.dispatch("table_db", dict(action=action, db="typo", table="t", **extra))
    assert reply["ok"] is False and reply["reason"] == "unknown_db" and "list" in reply["detail"]
    assert reply["written"] == 0
    assert not (tmp_path / "db" / "typo.db").exists()


def test_attach_file_to_a_missing_library_is_refused(tmp_path):
    card = _card(tmp_path)
    reply = _call(card, "attach_file", db="nope", table="t", _id=1, column="c",
                  file=_write(tmp_path, "a.png", PNG_BYTES))
    assert reply["reason"] == "unknown_db"


def test_import_mode_matrix(tmp_path):
    card = _card(tmp_path)
    path = _write(tmp_path, "r.xlsx", ROSTER)

    # append: library missing -> unknown_db; table missing -> unknown_table; nothing written
    assert _call(card, "import", db="exam", table="名单", path=path,
                 mode="append")["reason"] == "unknown_db"
    created = _call(card, "import", db="exam", table="名单", path=path)            # mode=create
    assert created["imported"] == 3 and created["db_created"] is True
    assert _call(card, "import", db="exam", table="别的表", path=path,
                 mode="append")["reason"] == "unknown_table"
    # create on an existing table: refused, says what to do instead
    again = _call(card, "import", db="exam", table="名单", path=path, mode="create")
    assert again["reason"] == "table_exists" and "append" in again["detail"]
    # append to it: works, and warns that it does not deduplicate
    more = _call(card, "import", db="exam", table="名单", path=path, mode="append")
    assert more["imported"] == 3 and "db_created" not in more
    assert any("不去重" in item for item in more["warnings"])
    assert _call(card, "stats", db="exam", table="名单")["total"] == 6


@pytest.mark.parametrize("key,text,check", [
    ("where", "{'借用人':'张三'}", lambda r: r["total"] == 2),
    ("where", "{'状态':['借出','逾期']}", lambda r: r["total"] == 2),
    ("where", "{’借用人‘:’李四‘}", lambda r: r["total"] == 1),
])
def test_where_accepts_single_quotes(tmp_path, key, text, check):
    card = _card(tmp_path)
    _ledger(card)
    reply = card.dispatch("table_db", {"action": "query", "db": "t", "table": "台账",
                                       "where": text.replace("’", "'").replace("‘", "'")
                                       if "’" in text else text})
    assert reply["ok"] is True, reply
    assert check(reply)


def test_set_and_records_accept_single_quotes(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    added = card.dispatch("table_db", {"action": "append", "db": "t", "table": "台账",
                                       "records": "[{'借用人':'王五','物品':'电钻'},{'借用人':'赵六'}]"})
    assert added["inserted"] == 2
    changed = card.dispatch("table_db", {"action": "update", "db": "t", "table": "台账",
                                         "_id": added["ids"][0], "set": "{'状态':'借出','物品':null}"})
    assert changed["record"]["状态"] == "借出" and "物品" not in changed["record"]


def test_no_double_quote_in_any_field_description():
    """The canvas inserts descriptions into placeholder="..." without escaping
    quotes; one ASCII double quote cut the hint off at `例 {`."""
    for key, spec in TOOLS[0]["inputSchema"]["properties"].items():
        assert '"' not in spec.get("description", ""), key


# ── review of PR #429: signed links, zip bombs ───────────────────────────────

def _served(card, url, headers=None):
    route = url[url.index("/table_db/file/"):]
    return card.handle_http_get(route, headers or {})


def test_file_links_open_without_a_token_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    import utils.file_intake as intake
    monkeypatch.setattr(intake, "_ENV_CANDIDATES", ())
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    got = _call(card, "get_file", table="名单", _id=2)

    assert "sig=" not in got["url"]
    assert _served(card, got["url"])[0] == 200


def test_with_access_token_issued_links_are_signed_and_open(tmp_path, monkeypatch):
    """Every consumer of the link (browser, *_by_url, the model's curl) gets 200
    with a token configured — none of them can send X-Access-Token."""
    monkeypatch.setenv("ACCESS_TOKEN", "s3cret")
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    got = _call(card, "get_file", table="名单", _id=2)

    assert "sig=" in got["url"]
    assert (_served(card, got["url"])[0], _served(card, got["url"])[1]) == (200, PNG_B)
    curl_url = got["agent_core"]["download"].split()[-1].strip("'")
    assert "sig=" in curl_url and _served(card, curl_url)[0] == 200

    bare = got["url"].split("?")[0]
    assert _served(card, bare)[0] == 401                                   # unsigned
    assert _served(card, bare + "?sig=" + "0" * 32)[0] == 401              # forged
    other = bare.rsplit("f_", 1)[0] + "f_1?" + got["url"].split("?")[1]
    assert _served(card, other)[0] == 401                                  # sig of f_2 on f_1
    assert _served(card, bare, {"X-Access-Token": "s3cret"})[0] == 200     # header still works
    assert _served(card, bare, {"X-Access-Token": "wrong"})[0] == 401


def _zip_with(member, size, compress=True):
    import io as _io
    import zipfile

    buffer = _io.BytesIO()
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", mode) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr(member, b"<" * size)
    return buffer.getvalue()


def test_zip_bomb_is_refused_before_it_is_inflated(tmp_path, monkeypatch):
    import plugins.table_db_parse as parse

    # A member far over the declared ratio: 40 MB of one byte compresses ~1000x.
    bomb = _zip_with("xl/worksheets/sheet1.xml", 40 * 1024 * 1024)
    assert len(bomb) < 200 * 1024
    card = _card(tmp_path)
    reply = _call(card, "import", table="x", path=_write(tmp_path, "bomb.xlsx", bomb))
    assert reply["ok"] is False and reply["reason"] == "too_large" and reply["written"] == 0
    assert "压缩比" in reply["detail"]

    # Per-member and total ceilings, checked from the central directory alone.
    monkeypatch.setattr(parse, "MAX_ZIP_MEMBER_BYTES", 1000)
    big = _zip_with("xl/worksheets/sheet1.xml", 5000, compress=False)
    with pytest.raises(parse.ParseError) as caught:
        parse.open_bounded_zip(big)
    assert caught.value.reason == "too_large"
    monkeypatch.setattr(parse, "MAX_ZIP_MEMBER_BYTES", 10 ** 9)
    monkeypatch.setattr(parse, "MAX_ZIP_TOTAL_BYTES", 1000)
    with pytest.raises(parse.ParseError):
        parse.open_bounded_zip(big)


def test_zip_member_lying_about_its_size_is_cut_off(tmp_path):
    import plugins.table_db_parse as parse

    data = bytearray(_zip_with("xl/worksheets/sheet1.xml", 5000, compress=False))
    # Rewrite the declared uncompressed size of the last member (central
    # directory and local header) to 100 bytes, keeping the real 5000 inside.
    import io
    import struct
    import zipfile
    archive = zipfile.ZipFile(io.BytesIO(bytes(data)))
    info = archive.getinfo("xl/worksheets/sheet1.xml")
    archive.close()
    local = info.header_offset
    struct.pack_into("<I", data, local + 22, 100)
    central = bytes(data).rfind(b"PK\x01\x02")
    struct.pack_into("<I", data, central + 24, 100)
    zf = parse.open_bounded_zip(bytes(data))
    with pytest.raises((parse.ParseError, zipfile.BadZipFile)):
        zf.read("xl/worksheets/sheet1.xml")


def test_real_workbooks_still_open_under_the_bounds(tmp_path):
    import plugins.table_db_parse as parse

    zf = parse.open_bounded_zip(ROSTER)
    assert "xl/workbook.xml" in zf.namelist()
    card = _card(tmp_path)
    assert _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))["imported"] == 3
