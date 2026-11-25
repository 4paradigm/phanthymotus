"""
tests/test_table_db.py — unit & integration tests for the SQLite `table_db` card.

Covers all 9 SQLite actions (tables / create_table / drop_table / insert /
select / update / delete / import / export), SQLite types (TEXT / INTEGER /
REAL / BLOB), BLOB readfile/download/GC, CSV/XLSX round-trips, zip-bomb bounds,
signed expiring links, cache quota eviction, concurrent max_db_bytes locking,
and main.make_handler() HTTP integration.

No ROS, model, or network. Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests/test_table_db.py -q
"""

from __future__ import annotations

import email.message
import io
import os
import sqlite3
import struct
import threading
import time
import types
import zipfile

import pytest

from ooxml_fixtures import PNG_BYTES, build_docx, build_xlsx, dispimg, paragraph, xs
from plugins.table_db import TOOLS, TableDbPlugin

PNG_B = PNG_BYTES + b"\x00b"

SAMPLE_XLSX = build_xlsx(
    rows=[
        [xs("title_row")],
        [xs("c_name"), xs("c_code"), xs("c_pic")],
        [xs("row_a"), xs("0012"), xs("")],
        [xs("row_b"), xs("0013"), xs("")],
        [xs("row_c"), xs("0014"), xs("")],
    ],
    anchors=[(2, 2, "image1.png", "twoCell"), (3, 2, "image2.png", "oneCell")],
    media={"image1.png": PNG_BYTES, "image2.png": PNG_B},
)


def _card(tmp_path, **overrides) -> TableDbPlugin:
    options = {"data_dir": str(tmp_path / "data"), "image_roots": [str(tmp_path)]}
    options.update(overrides)
    return TableDbPlugin(options, executor=None)


def _write(tmp_path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def _call(card: TableDbPlugin, action: str, **args) -> dict:
    reply = card.dispatch("table_db", dict(action=action, **args))
    assert reply is not None
    return reply


def _served(card: TableDbPlugin, url: str, headers=None):
    for prefix in ("/table_db/blob/", "/table_db/export/"):
        if prefix in url:
            route = url[url.index(prefix):]
            return card.handle_http_get(route, headers or {})
    raise AssertionError("unexpected url: " + url)


def _blob_counts(card: TableDbPlugin):
    conn = sqlite3.connect(card._store.db_path())
    try:
        files = conn.execute("SELECT COUNT(*) FROM _files").fetchone()[0]
        blobs = conn.execute("SELECT COUNT(*) FROM _blobs").fetchone()[0]
        return files, blobs
    finally:
        conn.close()


# ── schema & contract checks ─────────────────────────────────────────────────

def _llm_tools():
    """The tools agent-core actually hands the model: one per action."""
    import importlib.util, pathlib, sys as _sys, types as _types
    src = pathlib.Path(__file__).resolve().parents[2] / "agent-core" / "src" / "mcp_client.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("def _to_openai_schema(")
    end = text.index("\ndef declared_read_only(")
    scope: dict = {}
    exec(text[start:end], scope)
    return {s["name"].rsplit("__", 1)[-1]: s for s in scope["_to_openai_schema"]("m", TOOLS[0])}


def test_actions_read_like_sql():
    props = TOOLS[0]["inputSchema"]["properties"]
    assert [item["const"] for item in props["action"]["oneOf"]] == [
        "list_tables", "describe", "create_table", "add_column", "drop_table",
        "insert", "select", "update", "delete", "import", "export",
    ]
    tools = _llm_tools()
    # info is the canvas's status poll, not something the model should call.
    assert "info" not in tools and "start" not in tools
    assert set(tools["select"]["parameters"]["properties"]) == {
        "table", "fields", "where", "group_by", "order_by", "limit", "offset"}
    assert tools["drop_table"]["parameters"]["required"] == ["table", "confirm"]


def test_one_property_one_meaning():
    """Every action shares one `properties` map, so a property used two ways
    showed the model create_table's syntax inside select. Column definitions
    and columns-to-return are now separate properties with consistent types."""
    props = TOOLS[0]["inputSchema"]["properties"]
    assert props["columns"]["items"]["type"] == "object"
    assert props["fields"]["items"]["type"] == "string"
    assert props["values"]["type"] == "array" and props["where"]["type"] == "array"
    tools = _llm_tools()
    assert "columns" not in tools["select"]["parameters"]["properties"]
    assert "fields" not in tools["create_table"]["parameters"]["properties"]


def test_examples_are_real_json():
    """The canvas used to forbid double quotes (its escaper did not escape them),
    which left the model reading single-quoted pseudo-JSON."""
    import json as _json, re as _re
    props = TOOLS[0]["inputSchema"]["properties"]
    for key in ("columns", "fields", "values", "set", "where", "group_by", "order_by"):
        example = _re.search(r"例：(.+?)(?:。|$)", props[key]["description"]).group(1)
        _json.loads(example)


def test_descriptions_are_generic():
    """Examples use placeholder names, never a business's columns or a person.

    A model copies example column names: shown ["金额", ">", 100] it writes that
    against a table that has no 金额. Placeholders teach the syntax only.
    """
    import re as _re
    schema = TOOLS[0]["inputSchema"]
    texts = [TOOLS[0]["description"]]
    texts += [spec.get("description", "") for spec in schema["properties"].values()]
    texts += [spec.get("description", "") for spec in schema["x-action-params"].values()]
    texts += [item["description"] for item in schema["properties"]["action"]["oneOf"]]
    allowed = {"列A", "列B", "列C", "文件列", "值1", "值2", "新值", "列名", "rowid"}
    for text in texts:
        for word in ("张三", "李四", "姓名", "金额", "照片", "类别", "状态", "客户", "订单",
                     "聊天", "飞书", "QQ", "人脸", "访客"):
            assert word not in text, (word, text)
        # Quoted Chinese in an example is a column name or a value: placeholders only.
        for name in _re.findall(r'"([^"\s，。：；、$%]*[\u4e00-\u9fff][^"\s，。：；、$%]*)"', text):
            name = _re.sub(r"^(SUM|COUNT|AVG|MIN|MAX)\((.+)\)( DESC)?$", r"\2", name)
            assert name in allowed, (name, text)


def test_write_actions_are_not_read_only_to_peers():
    """Perception is read-only to a viewer peer unless the card says otherwise."""
    params = TOOLS[0]["inputSchema"]["x-action-params"]
    reads = {name for name, spec in params.items() if spec["readOnly"]}
    assert reads == {"list_tables", "describe", "select", "export"}
    assert TOOLS[0]["annotations"]["readOnlyHint"] is False


# ── DDL ──────────────────────────────────────────────────────────────────────

def test_create_describe_add_column_and_drop_table(tmp_path):
    card = _card(tmp_path)
    assert _call(card, "list_tables") == {"ok": True, "count": 0, "tables": []}

    created = _call(card, "create_table", table="t1", columns=[
        {"name": "c_text"}, {"name": "c_int", "type": "INTEGER"},
        {"name": "c_real", "type": "REAL"}, {"name": "c_blob", "type": "BLOB"}])
    assert created["ok"] is True
    assert [c["type"] for c in created["columns"]] == ["TEXT", "INTEGER", "REAL", "BLOB"]
    _call(card, "insert", table="t1", values=[{"c_text": "x"}])

    added = _call(card, "add_column", table="t1", columns=[{"name": "备注"}])
    assert added["ok"] is True and added["columns"][-1] == {"name": "备注", "type": "TEXT"}
    described = _call(card, "describe", table="t1")
    assert described["rows"] == 1 and described["columns"][-1]["name"] == "备注"
    assert _call(card, "select", table="t1")["rows"][0]["备注"] is None
    assert _call(card, "add_column", table="t1", columns=[{"name": "备注"}])["reason"] == "column_exists"

    assert _call(card, "list_tables")["tables"] == [{"table": "t1", "rows": 1}]
    assert _call(card, "tables")["tables"] == [{"table": "t1", "rows": 1}]   # old canvas name

    dup = _call(card, "create_table", table="t1", columns="a")
    assert dup["ok"] is False and dup["reason"] == "table_exists" and dup["written"] == 0
    assert _call(card, "create_table", table="t2", columns="a:file")["reason"] == "bad_request"

    refused = _call(card, "drop_table", table="t1")
    assert refused["ok"] is False and "confirm" in refused["detail"]
    assert _call(card, "drop_table", table="t1", confirm=False)["column"] == "confirm"
    assert _call(card, "drop_table", table="t1", confirm=True) == {
        "ok": True, "table": "t1", "dropped": True, "rows": 1}
    assert _call(card, "list_tables")["count"] == 0


# ── SELECT ───────────────────────────────────────────────────────────────────

@pytest.fixture
def orders(tmp_path):
    card = _card(tmp_path)
    _call(card, "create_table", table="orders", columns=[
        {"name": "客户"}, {"name": "类别"}, {"name": "金额", "type": "REAL"},
        {"name": "数量", "type": "INTEGER"}])
    assert _call(card, "insert", table="orders", values=[
        {"客户": "张三", "类别": "书", "金额": 30, "数量": 1},
        {"客户": "李四", "类别": "书", "金额": 120, "数量": 2},
        {"客户": "张小明", "类别": "文具", "金额": 15.5, "数量": 5},
        {"客户": "王五", "类别": "文具", "金额": 200, "数量": "3"},
        {"客户": "赵六"},
    ])["rowids"] == [1, 2, 3, 4, 5]
    return card


def _rows(card, **args):
    reply = _call(card, "select", table="orders", **args)
    assert reply["ok"] is True, reply
    return reply


def test_where_operators(orders):
    def who(where):
        return [r["客户"] for r in _rows(orders, where=where, order_by=["rowid"])["rows"]]
    assert who([["金额", ">", 100]]) == ["李四", "王五"]
    assert who([["金额", ">=", 30], ["金额", "<", 200]]) == ["张三", "李四"]
    assert who([["客户", "like", "张%"]]) == ["张三", "张小明"]
    assert who([["客户", "not like", "%张%"], ["金额", "is not null"]]) == ["李四", "王五"]
    assert who([["类别", "!=", "书"]]) == ["张小明", "王五"]
    assert who([["类别", "in", ["书"]]]) == ["张三", "李四"]
    assert who([["类别", "not in", ["书"]]]) == ["张小明", "王五"]
    assert who([["金额", "between", [20, 150]]]) == ["张三", "李四"]
    assert who([["金额", "is null"]]) == ["赵六"]
    assert who([["rowid", "=", 2]]) == ["李四"]
    assert who(["客户", "=", "王五"]) == ["王五"]            # a single condition, unwrapped
    assert who({"类别": "书"}) == ["张三", "李四"]           # old object form: equality


def test_where_errors_say_how_to_write_it(orders):
    bad = _call(orders, "select", table="orders", where=[["金额", "≈", 1]])
    assert bad["reason"] == "bad_request" and "like" in bad["detail"]
    unknown = _call(orders, "select", table="orders", where=[["价格", ">", 1]])
    assert unknown["reason"] == "unknown_column" and "金额" in unknown["detail"]


def test_aggregates_and_group_by(orders):
    whole = _rows(orders, fields=["COUNT(*)", "SUM(金额)", "MAX(数量)", "COUNT(金额)"])
    assert whole["rows"] == [{"COUNT(*)": 5, "SUM(金额)": 365.5, "MAX(数量)": 5, "COUNT(金额)": 4}]

    by_kind = _rows(orders, fields=["类别", "SUM(金额)", "COUNT(*)"], group_by=["类别"],
                    where=[["类别", "is not null"]], order_by=["SUM(金额) DESC"])
    assert by_kind["rows"] == [{"类别": "文具", "SUM(金额)": 215.5, "COUNT(*)": 2},
                               {"类别": "书", "SUM(金额)": 150.0, "COUNT(*)": 2}]
    assert by_kind["total"] == 2
    assert [c["name"] for c in by_kind["columns"]] == ["类别", "SUM(金额)", "COUNT(*)"]

    # group_by alone: the distinct values with their counts, ordered by the group.
    assert _rows(orders, group_by="类别", where=[["类别", "is not null"]])["rows"] == [
        {"类别": "书", "COUNT(*)": 2}, {"类别": "文具", "COUNT(*)": 2}]
    assert _rows(orders, fields=["COUNT(DISTINCT 类别)"])["rows"] == [{"COUNT(DISTINCT 类别)": 2}]

    ungrouped = _call(orders, "select", table="orders", fields=["客户", "SUM(金额)"])
    assert ungrouped["ok"] is False and "group_by" in ungrouped["detail"]
    text_sum = _call(orders, "select", table="orders", fields=["SUM(客户)"])
    assert text_sum["ok"] is False and "INTEGER / REAL" in text_sum["detail"]


def test_projection_order_and_paging(orders):
    page = _rows(orders, fields=["客户", "金额"], order_by=["金额 DESC"], limit=2,
                 where=[["金额", "is not null"]])
    assert page["total"] == 4 and page["has_more"] is True
    assert page["rows"] == [{"rowid": 4, "客户": "王五", "金额": 200.0},
                            {"rowid": 2, "客户": "李四", "金额": 120.0}]
    # Canvas cards saved before `fields` existed still send `columns`.
    old = _call(orders, "select", table="orders", columns="客户", where={"rowid": 1})
    assert old["rows"] == [{"rowid": 1, "客户": "张三"}]


def test_update_and_delete_take_condition_lists(orders):
    upd = _call(orders, "update", table="orders", set={"类别": "大单"}, where=[["金额", ">", 100]])
    assert upd["updated"] == 2 and upd["rowids"] == [2, 4]
    gone = _call(orders, "delete", table="orders", where=[["金额", "is null"]])
    assert gone["deleted"] == 1
    no_where = _call(orders, "delete", table="orders", where=[])
    assert no_where["ok"] is False and no_where["column"] == "where"


def test_insert_failure_rolls_back_entire_batch(tmp_path):
    card = _card(tmp_path)
    _call(card, "create_table", table="t", columns="k,n:INTEGER")
    bad = _call(card, "insert", table="t", values=[
        {"k": "ok", "n": 1},
        {"k": "bad", "n": "not_an_int"},
    ])
    assert bad["ok"] is False and bad["reason"] == "bad_request" and bad["written"] == 0
    assert _call(card, "select", table="t")["total"] == 0


def test_blob_insert_update_download_and_gc_on_replace_clear_delete(tmp_path):
    card = _card(tmp_path)
    _call(card, "create_table", table="docs", columns="title,data:BLOB")
    p1 = _write(tmp_path, "one.png", PNG_BYTES)
    p2 = _write(tmp_path, "two.png", PNG_B)
    pdf = _write(tmp_path, "doc.pdf", b"%PDF-1.4\nhello")

    # Insert via direct file path (readfile semantics) + dict {"readfile": ...}
    ins = _call(card, "insert", table="docs", values=[
        {"title": "r1", "data": p1},
        {"title": "r2", "data": {"readfile": p2}},
    ])
    assert ins["ok"] is True and ins["rowids"] == [1, 2]
    assert _blob_counts(card) == (2, 2)

    # Select returns BLOB descriptor with working HTTP download URL
    sel = _call(card, "select", table="docs", where={"rowid": 1})
    blob_cell = sel["rows"][0]["data"]
    assert blob_cell["type"] == "BLOB" and blob_cell["size"] == len(PNG_BYTES)
    status, body, mime, fname = _served(card, blob_cell["url"])
    assert (status, body, mime, fname) == (200, PNG_BYTES, "image/png", "one.png")

    # Update replaces BLOB via canvas `file` upload auto-binding -> old BLOB is GC'd!
    upd = _call(card, "update", table="docs", where={"rowid": 1}, file=pdf)
    assert upd["ok"] is True and upd["updated"] == 1
    assert _blob_counts(card) == (2, 2)
    status, body, mime, fname = _served(card, upd["rows"][0]["data"]["url"])
    assert (status, body, mime, fname) == (200, b"%PDF-1.4\nhello", "application/pdf", "doc.pdf")

    # Update sets BLOB to null -> BLOB is GC'd!
    clr = _call(card, "update", table="docs", where={"rowid": 1}, set={"data": None})
    assert clr["ok"] is True and clr["rows"][0]["data"] is None
    assert _blob_counts(card) == (1, 1)

    # Delete remaining row -> last BLOB is GC'd!
    deleted = _call(card, "delete", table="docs", where={"rowid": 2})
    assert deleted == {"ok": True, "table": "docs", "deleted": 1, "rowids": [2]}
    assert _blob_counts(card) == (0, 0)


def test_canvas_shorthand_single_quotes_and_file_placeholder(tmp_path):
    card = _card(tmp_path)
    _call(card, "create_table", table="t", columns="k,v,f:BLOB")
    p = _write(tmp_path, "a.png", PNG_BYTES)

    ins = _call(card, "insert", table="t", values="[{'k':'k1','v':'v1','f':'$file'}]", file=p)
    assert ins["ok"] is True and ins["rowids"] == [1]

    upd = _call(card, "update", table="t", where="{'rowid':1}", set="{'v':'v2'}")
    assert upd["ok"] is True and upd["rows"][0]["v"] == "v2"


def test_hostile_names_and_values_cannot_inject_sql(tmp_path):
    card = _card(tmp_path)
    evil_table = "t'; DROP TABLE _tables; --"
    evil_col = 'c" OR 1=1 --'
    assert _call(card, "create_table", table=evil_table,
                 columns=[{"name": evil_col, "type": "TEXT"}])["ok"] is True
    assert _call(card, "insert", table=evil_table,
                 values=[{evil_col: "'; DELETE FROM _tables; --"}])["ok"] is True
    got = _call(card, "select", table=evil_table, where={evil_col: "'; DELETE FROM _tables; --"})
    assert got["total"] == 1
    assert _call(card, "list_tables")["count"] == 1


# ── import & export (.csv and .xlsx only) ────────────────────────────────────

def test_import_xlsx_and_csv_and_refuse_docx_and_xls(tmp_path):
    card = _card(tmp_path)
    xlsx_path = _write(tmp_path, "sample.xlsx", SAMPLE_XLSX)
    imp = _call(card, "import", table="t_xlsx", path=xlsx_path)
    assert imp["ok"] is True and imp["imported"] == 3 and imp["blobs"] == 2
    assert [c["type"] for c in imp["columns"]] == ["TEXT", "TEXT", "BLOB"]

    # Append mode on existing table
    app = _call(card, "import", table="t_xlsx", path=xlsx_path, mode="append")
    assert app["ok"] is True and app["imported"] == 3
    assert _call(card, "select", table="t_xlsx")["total"] == 6

    # CSV import
    csv_path = _write(tmp_path, "sample.csv", "col1,col2\na,1\nb,2\n".encode("utf-8-sig"))
    c_imp = _call(card, "import", table="t_csv", path=csv_path)
    assert c_imp["ok"] is True and c_imp["imported"] == 2

    # Refusals: .docx, .xls, .tsv all rejected
    docx_path = _write(tmp_path, "bad.docx", build_docx([paragraph("x")]))
    xls_path = _write(tmp_path, "bad.xls", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32)
    tsv_path = _write(tmp_path, "bad.tsv", b"a\tb\n1\t2\n")
    for p in (docx_path, xls_path, tsv_path):
        rep = _call(card, "import", table="x", path=p)
        assert rep["ok"] is False and rep["reason"] == "unsupported_format" and rep["written"] == 0


def test_import_wps_dispimg_and_export_xlsx_and_csv_round_trip(tmp_path):
    wps = build_xlsx(
        rows=[[xs("col1"), xs("col2")], [xs("v1"), dispimg("ID_1")]],
        cell_images=[("ID_1", "image1.png")],
        media={"image1.png": PNG_BYTES},
    )
    card = _card(tmp_path)
    assert _call(card, "import", table="src", path=_write(tmp_path, "wps.xlsx", wps))["blobs"] == 1

    # Export to XLSX -> download -> re-import into a second table losslessly!
    exp_xlsx = _call(card, "export", table="src", format="xlsx")
    assert exp_xlsx["ok"] is True and exp_xlsx["rows"] == 1
    status, body, mime, fname = _served(card, exp_xlsx["url"])
    assert status == 200 and fname.endswith(".xlsx") and "spreadsheetml" in mime

    reimp_path = _write(tmp_path, "roundtrip.xlsx", body)
    reimp = _call(card, "import", table="dst", path=reimp_path)
    assert reimp["ok"] is True and reimp["imported"] == 1 and reimp["blobs"] == 1
    sel = _call(card, "select", table="dst")
    assert sel["rows"][0]["col1"] == "v1"
    st2, img_bytes, _, _ = _served(card, sel["rows"][0]["col2"]["url"])
    assert st2 == 200 and img_bytes == PNG_BYTES

    # Export to CSV -> download
    exp_csv = _call(card, "export", table="src", format="csv")
    assert exp_csv["ok"] is True and exp_csv["format"] == "csv"
    st3, csv_bytes, mime3, fname3 = _served(card, exp_csv["url"])
    assert st3 == 200 and fname3.endswith(".csv") and "text/csv" in mime3
    text = csv_bytes.decode("utf-8-sig")
    assert "col1,col2" in text and "v1," in text


# ── security: signed expiring URLs, cache quota, zip bomb, locked size check ─

def test_signed_links_expiry_and_main_http_handler_integration(tmp_path, monkeypatch):
    import sys
    try:
        import rclpy  # noqa: F401
    except ImportError:
        try:
            from vision_stubs import PERCEPTION_ROOT  # noqa: F401
        except ImportError:
            rclpy_mod = types.ModuleType("rclpy")
            exec_mod = types.ModuleType("rclpy.executors")
            rclpy_mod.executors = exec_mod
            monkeypatch.setitem(sys.modules, "rclpy", rclpy_mod)
            monkeypatch.setitem(sys.modules, "rclpy.executors", exec_mod)
    pytest.importorskip("yaml")
    import main
    import plugins.table_db as mod

    now = [1_700_000_000.0]
    monkeypatch.setattr(mod.time, "time", lambda: now[0])
    monkeypatch.setenv("ACCESS_TOKEN", "s3cret")

    card = _card(tmp_path)
    _call(card, "import", table="表一", path=_write(tmp_path, "s.xlsx", SAMPLE_XLSX))
    sel = _call(card, "select", table="表一", where={"rowid": 2})
    blob_url = sel["rows"][0]["c_pic"]["url"]
    assert "exp=" in blob_url and "sig=" in blob_url

    signed_route = blob_url[blob_url.index("/table_db/blob/"):]
    bare_route = signed_route.split("?", 1)[0]

    monkeypatch.setattr(main, "_bundle", types.SimpleNamespace(_plugins=[card]))
    handler_cls = main.make_handler()

    def run_get(path, req_headers=None):
        h = handler_cls.__new__(handler_cls)
        h.path = path
        h.command = "GET"
        h.request_version = "HTTP/1.1"
        h.requestline = "GET %s HTTP/1.1" % path
        h.client_address = ("127.0.0.1", 0)
        msg = email.message.Message()
        for k, v in (req_headers or {}).items():
            msg[k] = v
        h.headers = msg
        h.wfile = io.BytesIO()
        h.do_GET()
        raw = h.wfile.getvalue()
        head, _, body = raw.partition(b"\r\n\r\n")
        lines = head.decode("iso-8859-1").split("\r\n")
        status = int(lines[0].split(" ", 2)[1])
        resp_headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                resp_headers[k.strip().lower()] = v.strip()
        return status, resp_headers, body

    status, headers, body = run_get(signed_route)
    assert status == 200 and headers.get("content-type") == "image/png" and body == PNG_B

    assert run_get(bare_route)[0] == 401
    status, _, body = run_get(bare_route, {"x-access-token": "s3cret"})
    assert status == 200 and body == PNG_B

    # Tampered exp -> 401
    qs = signed_route.split("?", 1)[1]
    assert run_get(bare_route + "?exp=9999999999&" + qs.split("&", 1)[1])[0] == 401

    # Expired -> 401
    now[0] += 86401
    status, _, body = run_get(signed_route)
    assert status == 401 and b"expired" in body

    assert run_get("/not-ours")[0] == 404


def test_export_cache_reuses_evicts_and_refuses_oversized(tmp_path):
    card = _card(tmp_path)
    _call(card, "import", table="t", path=_write(tmp_path, "s.xlsx", SAMPLE_XLSX))
    first = _call(card, "export", table="t", format="csv", where={"c_name": "row_a"})
    again = _call(card, "export", table="t", format="csv", where={"c_name": "row_a"})
    assert first["ok"] is True and again["path"] == first["path"]

    sub = tmp_path / "bounded"
    sub.mkdir()
    bounded = _card(sub, max_cache_bytes=first["size"] + 20)
    assert _call(bounded, "import", table="t", path=_write(sub, "s2.xlsx", SAMPLE_XLSX))["ok"]
    e1 = _call(bounded, "export", table="t", format="csv", where={"c_name": "row_a"})
    assert e1["ok"] is True and os.path.isfile(e1["path"])
    os.utime(e1["path"], (time.time() - 60, time.time() - 60))

    e2 = _call(bounded, "export", table="t", format="csv", where={"c_name": "row_b"})
    assert e2["ok"] is True and os.path.isfile(e2["path"])
    assert not os.path.exists(e1["path"])

    too_big = _call(bounded, "export", table="t", format="xlsx")
    assert too_big["ok"] is False and too_big["reason"] == "too_large"
    assert os.path.isfile(e2["path"])


def test_concurrent_blob_updates_cannot_bypass_max_db_bytes(tmp_path):
    blob = b"\x89PNG\r\n\x1a\n" + os.urandom(60 * 1024)
    p1 = _write(tmp_path, "a.png", blob)
    p2 = _write(tmp_path, "b.png", blob + b"x")
    card = _card(tmp_path)
    _call(card, "create_table", table="t", columns="k,f:BLOB")
    ids = _call(card, "insert", table="t", values=[{"k": "1"}, {"k": "2"}])["rowids"]

    base_size = card._store.db_bytes()
    card._store.limits.max_db_bytes = base_size + 120 * 1024

    barrier = threading.Barrier(2)
    results = []

    def worker(rid, path):
        barrier.wait()
        results.append(_call(card, "update", table="t", where={"rowid": rid}, set={"f": path}))

    threads = [threading.Thread(target=worker, args=(ids[0], p1)),
               threading.Thread(target=worker, args=(ids[1], p2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok_count = sum(1 for r in results if r["ok"])
    too_large = [r for r in results if not r["ok"] and r["reason"] == "too_large"]
    assert ok_count == 1 and len(too_large) == 1


def _zip_with(member, size, compress=True):
    buf = io.BytesIO()
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buf, "w", mode) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr(member, b"<" * size)
    return buf.getvalue()


def test_zip_bomb_and_lying_header_are_refused(tmp_path, monkeypatch):
    import plugins.table_db_parse as parse

    bomb = _zip_with("xl/worksheets/sheet1.xml", 40 * 1024 * 1024)
    card = _card(tmp_path)
    reply = _call(card, "import", table="x", path=_write(tmp_path, "bomb.xlsx", bomb))
    assert reply["ok"] is False and reply["reason"] == "too_large" and reply["written"] == 0

    data = bytearray(_zip_with("xl/worksheets/sheet1.xml", 5000, compress=False))
    archive = zipfile.ZipFile(io.BytesIO(bytes(data)))
    info = archive.getinfo("xl/worksheets/sheet1.xml")
    archive.close()
    struct.pack_into("<I", data, info.header_offset + 22, 100)
    central = bytes(data).rfind(b"PK\x01\x02")
    struct.pack_into("<I", data, central + 24, 100)
    zf = parse.open_bounded_zip(bytes(data))
    with pytest.raises((parse.ParseError, zipfile.BadZipFile)):
        zf.read("xl/worksheets/sheet1.xml")


def test_csv_export_escapes_formula_injection_prefixes(tmp_path):
    """Exported CSV neutralizes spreadsheet formula prefixes (=, +, -, @, tab)
    on TEXT cells while preserving genuine numeric INTEGER/REAL values."""
    import csv as _csv

    card = _card(tmp_path)
    _call(card, "create_table", table="f_test", columns="txt,n:INTEGER,r:REAL")
    dangerous = [
        "=HYPERLINK(\"http://evil\",\"click\")",
        "+1+1",
        "-2+3",
        "@SUM(A1:A2)",
        "\t=cmd|' /C calc'!A0",
        " =1+2",
    ]
    _call(card, "insert", table="f_test", values=[
        {"txt": val, "n": -42, "r": -1.5} for val in dangerous
    ])

    exp = _call(card, "export", table="f_test", format="csv")
    assert exp["ok"] is True
    st, raw, _, _ = _served(card, exp["url"])
    assert st == 200
    rows = list(_csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    assert rows[0] == ["txt", "n", "r"]
    for idx, orig in enumerate(dangerous, 1):
        assert rows[idx][0] == "'" + orig
        assert rows[idx][1] == "-42"
        assert rows[idx][2] == "-1.5"


def test_import_rejects_oversized_xlsx_media_before_inflating(tmp_path, monkeypatch):
    """An embedded XLSX image exceeding max_file_bytes is rejected using its
    ZipInfo.file_size header without opening or inflating the media entry."""
    import plugins.table_db_parse as parse

    big_png = PNG_BYTES + b"\x00" * 4096
    xlsx = build_xlsx(
        rows=[[xs("col1"), xs("img")], [xs("v1"), xs("")]],
        anchors=[(1, 1, "big.png", "twoCell")],
        media={"big.png": big_png},
    )
    opened_media = []
    orig_open = parse._BoundedZip.open

    def spy_open(self, name, *args, **kwargs):
        member = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        if member.startswith("xl/media/"):
            opened_media.append(member)
        return orig_open(self, name, *args, **kwargs)

    monkeypatch.setattr(parse._BoundedZip, "open", spy_open)
    card = _card(tmp_path, max_file_bytes=512)
    reply = _call(card, "import", table="t_big", path=_write(tmp_path, "big.xlsx", xlsx))
    assert reply["ok"] is False and reply["reason"] == "too_large" and reply["written"] == 0
    assert opened_media == []
    assert _call(card, "tables")["count"] == 0


def test_over_budget_xlsx_export_refuses_before_reading_blobs(tmp_path, monkeypatch):
    """Preflight rejects an XLSX export whose BLOB sizes exceed max_cache_bytes
    without calling store.read_files() or materializing media in memory."""
    big_png = PNG_BYTES + b"\x00" * 4096
    p1 = _write(tmp_path, "a.png", big_png)
    p2 = _write(tmp_path, "b.png", big_png + b"1")
    card = _card(tmp_path, max_cache_bytes=5000)
    _call(card, "create_table", table="photos", columns="k,img:BLOB")
    assert _call(card, "insert", table="photos", values=[
        {"k": "1", "img": p1},
        {"k": "2", "img": p2},
    ])["ok"] is True

    read_calls = []
    orig_read_files = card._store.read_files

    def spy_read_files(fids):
        read_calls.append(list(fids))
        return orig_read_files(fids)

    monkeypatch.setattr(card._store, "read_files", spy_read_files)
    reply = _call(card, "export", table="photos", format="xlsx")
    assert reply["ok"] is False and reply["reason"] == "too_large"
    assert read_calls == []


def test_deleted_blob_freelist_pages_are_reused_under_max_db_bytes(tmp_path):
    """After deleting a BLOB row, SQLite freelist pages are subtracted from the
    file size check so subsequent writes can reuse reclaimed space."""
    blob_a = b"\x89PNG\r\n\x1a\n" + os.urandom(80 * 1024)
    blob_b = b"\x89PNG\r\n\x1a\n" + os.urandom(80 * 1024)
    p1 = _write(tmp_path, "a.png", blob_a)
    p2 = _write(tmp_path, "b.png", blob_b)

    card = _card(tmp_path)
    _call(card, "create_table", table="t", columns="k,f:BLOB")
    base_size = card._store.db_bytes()
    # Fits one 80 KiB BLOB, not two simultaneously.
    card._store.limits.max_db_bytes = base_size + 150 * 1024

    ins1 = _call(card, "insert", table="t", values=[{"k": "1", "f": p1}])
    assert ins1["ok"] is True
    # Second simultaneous BLOB exceeds max_db_bytes.
    ins2_full = _call(card, "insert", table="t", values=[{"k": "2", "f": p2}])
    assert ins2_full["ok"] is False and ins2_full["reason"] == "too_large"

    # Delete row 1 -> BLOB pages move to SQLite freelist -> row 2 now fits!
    assert _call(card, "delete", table="t", where={"rowid": ins1["rowids"][0]})["deleted"] == 1
    ins2_after = _call(card, "insert", table="t", values=[{"k": "2", "f": p2}])
    assert ins2_after["ok"] is True


def test_files_travel_in_a_file_param(tmp_path):
    """agent-core pushes only `format: file` params to the card's container; a
    path written inside `values` is a path in agent-core's container, which this
    card cannot open. The schema has to steer the model to `file` + "$file"."""
    props = TOOLS[0]["inputSchema"]["properties"]
    assert props["file"]["format"] == "file" and props["path"]["format"] == "file"
    assert '"$file"' in props["values"]["description"]
    card = _card(tmp_path)
    _call(card, "create_table", table="t", columns=[{"name": "照片", "type": "BLOB"}])
    missing = _call(card, "insert", table="t", values=[{"照片": "/opt/phanthy-motus/x.jpg"}])
    assert missing["reason"] == "not_found" and "file 参数" in missing["detail"]
