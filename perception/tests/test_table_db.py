"""
tests/test_table_db.py — the `table_db` card as skills use it.

Each test pins one promise:

* import turns a spreadsheet (with pictures) into rows + stored files, all or
  nothing, with the row and column of whatever was wrong;
* append / query / update / delete / stats behave like a small database whose
  ids and counts the model never has to compute;
* files are stored as BLOBs, come back byte-identical via get_file, and
  export_files produces the package face_recognition.register_by_corpus reads;
* request_id makes retries safe; bad input never writes; nothing raises.

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
    assert _call(card, "query", table="名单")["reason"] == "unknown_table"


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
    # The table itself was created inside the failed transaction: gone too.
    assert _call(card, "query", table="名单")["reason"] == "unknown_table"


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


def test_import_retry_with_request_id_replays(tmp_path):
    card = _card(tmp_path)
    path = _write(tmp_path, "r.xlsx", ROSTER)
    first = _call(card, "import", table="名单", path=path, request_id="imp-1")
    again = _call(card, "import", table="名单", path=path, request_id="imp-1")

    assert again["replayed"] is True and again["imported"] == first["imported"]
    assert _call(card, "stats", table="名单")["total"] == 3

    other = _write(tmp_path, "o.csv", "姓名\n赵六\n".encode())
    conflict = _call(card, "import", table="名单", path=other, mode="append",
                     request_id="imp-1")
    assert conflict["reason"] == "request_conflict"


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


def test_append_retry_with_request_id(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    record = [{"借用人": "王五", "物品": "电钻", "状态": "借出"}]
    first = _call(card, "append", table="台账", records=record, request_id="a-1")
    again = _call(card, "append", table="台账", records=record, request_id="a-1")

    assert again["replayed"] is True and again["ids"] == first["ids"]
    assert _call(card, "stats", table="台账")["total"] == 4


def test_databases_are_isolated(tmp_path):
    card = _card(tmp_path)
    _call(card, "create", table="名单", columns=["姓名"], db="skill-a")
    assert _call(card, "query", table="名单", db="skill-b")["reason"] == "unknown_table"


# ── files ────────────────────────────────────────────────────────────────────

def test_get_file_returns_identical_bytes(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    record = _call(card, "query", table="名单", where={"姓名": "李四"})["records"][0]
    got = _call(card, "get_file", table="名单", file_id=record["照片"]["file_id"])

    assert got["ok"] is True and got["mime"] == "image/png"
    with open(got["path"], "rb") as handle:
        assert handle.read() == PNG_B
    assert got["path"].startswith(str(tmp_path / "db"))
    assert _call(card, "get_file", table="名单", file_id="f_999")["reason"] == "not_found"
    assert _call(card, "get_file", table="名单", file_id="nope")["reason"] == "bad_request"


def test_attach_any_kind_of_file_and_replace(tmp_path):
    card = _card(tmp_path)
    _ledger(card)
    pdf = b"%PDF-1.4\n" + os.urandom(2000)
    attached = _call(card, "attach_file", table="台账", _id=1, column="照片",
                     path=_write(tmp_path, "收据.pdf", pdf))
    assert attached["file"]["mime"] == "application/pdf" and attached["file"]["size"] == len(pdf)

    wav = b"RIFF\x00\x00\x00\x00WAVEfmt " + os.urandom(100)
    replaced = _call(card, "attach_file", table="台账", _id=1, column="照片",
                     path=_write(tmp_path, "note.wav", wav))
    record = _call(card, "query", table="台账", where={"_id": 1})["records"][0]
    assert record["照片"]["file_id"] == replaced["file"]["file_id"]
    assert record["照片"]["mime"] == "audio/wav"

    assert _call(card, "attach_file", table="台账", _id=1, column="物品",
                 path=_write(tmp_path, "a.png", PNG_BYTES))["reason"] == "bad_request"
    big = _card(tmp_path, max_file_bytes=10)
    assert _call(big, "attach_file", table="台账", _id=1, column="照片",
                 path=_write(tmp_path, "b.png", PNG_BYTES))["reason"] == "too_large"

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
    assert reply["dir"] in reply["next"]
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


# ── robustness ───────────────────────────────────────────────────────────────

def test_data_survives_a_restart(tmp_path):
    _call(_card(tmp_path), "import", table="名单", path=_write(tmp_path, "r.xlsx", ROSTER))
    restarted = _card(tmp_path)

    reply = _call(restarted, "query", table="名单")
    assert reply["total"] == 3
    got = _call(restarted, "get_file", table="名单", file_id=reply["records"][0]["照片"]["file_id"])
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

    assert visible == ["create", "import", "append", "query", "update", "delete", "stats",
                       "attach_file", "get_file", "export_files"]
    for action in visible:
        spec = split[action]
        assert "table" in spec["params"] and "table" in spec["required"]
        for param in spec["params"]:
            assert param in schema["properties"], (action, param)
        assert set(spec["required"]) <= set(spec["params"])
    # agent-core pushes a file to this container only for `format: file` params.
    assert schema["properties"]["path"]["format"] == "file"


def test_schema_is_valid_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft7Validator.check_schema(TOOLS[0]["inputSchema"])
