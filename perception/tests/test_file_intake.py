"""
tests/test_file_intake.py — HTTP file intake (utils/file_intake.py).

This is how a photo reaches the perception container at all. agent-core serves
the browser but shares no filesystem with perception, so a path minted there is
meaningless here — the first face-enrolment attempt failed with an `image_path`
that really existed, in agent-core. agent-core now proxies the upload to
`POST /file/upload` on this service, which writes the bytes where it can see
them and returns its own absolute path.

Stdlib only, so the drivers (bare ThreadingHTTPServer, no FastAPI) can reuse it.

Run: python -m pytest perception/tests/test_file_intake.py -q
"""

from __future__ import annotations

import glob
import os

import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (puts perception on sys.path)

from utils.file_intake import (  # noqa: E402
    IntakeError,
    access_token,
    check_token,
    handle_upload,
    parse_multipart,
    prune_old,
    safe_filename,
    safe_subdir,
    store_stream,
)

BOUNDARY = "----boundary9d8f"
JPEG = b"\xff\xd8\xff\xe0" + b"payload-bytes" * 500 + b"\xff\xd9"


def _multipart(filename: str = "alice.jpg", payload: bytes = JPEG,
               extra: dict | None = None) -> tuple[dict, bytes]:
    parts = []
    for key, value in (extra or {}).items():
        parts.append(
            f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode()
        + payload + f"\r\n--{BOUNDARY}--\r\n".encode()
    )
    return ({"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
            b"".join(parts))


# ── filename safety ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expect_not", [
    ("../../etc/passwd", ".."),
    ("../../../root/.ssh/authorized_keys", ".."),
    ("/etc/shadow", "/"),
    ("C:\\windows\\system32\\x.jpg", "\\"),
    ("a/b/c.jpg", "/"),
])
def test_traversal_cannot_survive_the_filename(raw, expect_not):
    """A client-supplied filename is joined onto a path, so it has to collapse
    to a single component."""
    cleaned = safe_filename(raw)
    assert expect_not not in cleaned
    assert os.path.basename(cleaned) == cleaned


def test_chinese_filenames_are_kept():
    """The people using this name their files in Chinese; mangling those to
    underscores would make the upload directory unreadable."""
    assert safe_filename("戴文渊.jpg") == "戴文渊.jpg"
    assert safe_filename("陈雨强-正面.png") == "陈雨强-正面.png"


def test_leading_dots_and_empty_names_get_a_fallback():
    assert not safe_filename("...hidden").startswith(".")
    assert safe_filename("") == "upload.bin"
    assert safe_filename("..") == "upload.bin"
    assert safe_filename("/") == "upload.bin"


def test_absurdly_long_names_are_bounded_but_keep_the_extension():
    """Many filesystems cap a component at 255 bytes, and a CJK name is 3 bytes
    per character."""
    cleaned = safe_filename("超长" * 200 + ".jpg")
    assert len(cleaned.encode("utf-8")) <= 255
    assert cleaned.endswith(".jpg")


def test_macos_and_linux_uploads_of_one_name_agree():
    """NFD vs NFC would otherwise produce two files that look identical."""
    assert safe_filename("é.jpg") == safe_filename("e\u0301.jpg")


def test_subdir_is_one_component_or_refused():
    """Refused, not sanitised: turning "../escape" into ".._escape" made the
    write succeed somewhere the caller did not ask for, silently."""
    assert safe_subdir("") == ""
    assert safe_subdir("faces") == "faces"
    for bad in ("../escape", "..", ".", "a/b", "a\\b", "../../etc"):
        with pytest.raises(IntakeError):
            safe_subdir(bad)


# ── auth ──────────────────────────────────────────────────────────────────────

def test_token_check_is_a_noop_when_none_is_configured():
    """Matches agent-core's own behaviour: no ACCESS_TOKEN means auth disabled,
    so a dev deployment works untouched. Verified on Tianyi that the perception
    container is not given one."""
    check_token(None, None)
    check_token(None, "")
    check_token("anything", "")


def test_a_configured_token_is_enforced():
    check_token("secret", "secret")
    for bad in (None, "", "wrong", "secre", "secrets"):
        with pytest.raises(IntakeError) as caught:
            check_token(bad, "secret")
        assert caught.value.status == 401


def test_access_token_prefers_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ACCESS_TOKEN", "from-env")
    assert access_token() == "from-env"


def test_access_token_is_empty_when_there_is_no_source(monkeypatch):
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    monkeypatch.setattr("utils.file_intake._ENV_CANDIDATES", ("/nonexistent/.env",))
    assert access_token() == ""


# ── storing ───────────────────────────────────────────────────────────────────

def test_a_stored_file_is_byte_identical_and_reports_its_own_path(tmp_path):
    headers, body = _multipart()
    status, result = handle_upload(headers, body, str(tmp_path))

    assert status == 200 and result["ok"]
    assert os.path.isabs(result["path"])
    assert result["path"].startswith(str(tmp_path))
    assert result["bytes"] == len(JPEG)
    with open(result["path"], "rb") as handle:
        assert handle.read() == JPEG, "the bytes must survive the round trip"


def test_files_land_in_a_date_directory(tmp_path):
    import time
    headers, body = _multipart()
    _status, result = handle_upload(headers, body, str(tmp_path))
    assert time.strftime("%Y-%m-%d") in result["path"]


def test_a_subdir_is_honoured(tmp_path):
    headers, body = _multipart(extra={"subdir": "faces"})
    _status, result = handle_upload(headers, body, str(tmp_path))
    assert os.sep + "faces" + os.sep in result["path"]


def test_no_partial_file_is_left_under_the_final_name(tmp_path):
    """A reader listing the directory must never see half a file."""
    headers, body = _multipart()
    status, result = handle_upload(headers, body, str(tmp_path), max_bytes=64)
    assert status == 413 and not result["ok"]
    assert glob.glob(os.path.join(str(tmp_path), "*", "*")) == []
    assert glob.glob(os.path.join(str(tmp_path), "*", ".*.part")) == []


def test_the_size_cap_is_enforced_while_writing_not_after(tmp_path):
    """So an oversized body is never fully committed to disk or memory."""
    big = b"x" * 5_000_000
    headers, body = _multipart(payload=big)
    status, result = handle_upload(headers, body, str(tmp_path), max_bytes=1_000_000)
    assert status == 413
    assert "limit" in result["error"]


def test_an_empty_file_is_refused(tmp_path):
    headers, body = _multipart(payload=b"")
    status, result = handle_upload(headers, body, str(tmp_path))
    assert status == 400 and "empty" in result["error"]


def test_a_traversing_filename_stays_inside_the_base_dir(tmp_path):
    headers, body = _multipart(filename="../../escaped.jpg")
    _status, result = handle_upload(headers, body, str(tmp_path))
    assert result["ok"]
    assert os.path.realpath(result["path"]).startswith(os.path.realpath(str(tmp_path)))
    assert not os.path.exists(tmp_path.parent / "escaped.jpg")


def test_store_stream_directly(tmp_path):
    import io
    result = store_stream(io.BytesIO(b"abc"), "x.bin", str(tmp_path))
    assert result["bytes"] == 3


# ── request parsing ───────────────────────────────────────────────────────────

def test_a_non_multipart_body_is_refused(tmp_path):
    status, result = handle_upload(
        {"Content-Type": "application/json"}, b'{"file": "x"}', str(tmp_path))
    assert status == 400
    assert "multipart" in result["error"]


def test_a_multipart_body_without_a_file_part_is_refused(tmp_path):
    body = (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n"
            f"Alice\r\n--{BOUNDARY}--\r\n").encode()
    status, result = handle_upload(
        {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        body, str(tmp_path))
    assert status == 400
    assert "file" in result["error"]


def test_binary_containing_the_boundary_marker_survives(tmp_path):
    """Hand-rolled boundary splitting mangles exactly this; cgi.FieldStorage
    handles it, which is why the parser is not hand-written."""
    tricky = b"\xff\xd8" + BOUNDARY.encode() + b"\r\n--not-the-end\r\n" + b"\xff\xd9"
    headers, body = _multipart(payload=tricky)
    _status, result = handle_upload(headers, body, str(tmp_path))
    assert result["ok"]
    with open(result["path"], "rb") as handle:
        assert handle.read() == tricky


def test_extra_form_fields_are_exposed_not_dropped():
    headers, body = _multipart(extra={"subdir": "faces", "note": "hello"})
    filename, _stream, fields = parse_multipart(headers, body)
    assert filename == "alice.jpg"
    assert fields["subdir"] == "faces"
    assert fields["note"] == "hello"


# ── retention ─────────────────────────────────────────────────────────────────

def test_old_date_directories_are_pruned(tmp_path):
    """Uploads are a transfer buffer, not storage: enrolment keeps the embedding
    and never the photo, so this directory would grow forever on eMMC."""
    for day in ("2020-01-01", "2020-06-15"):
        (tmp_path / day).mkdir()
        (tmp_path / day / "old.jpg").write_bytes(b"x")
    import time
    today = time.strftime("%Y-%m-%d")
    (tmp_path / today).mkdir(exist_ok=True)
    (tmp_path / today / "new.jpg").write_bytes(b"x")

    assert prune_old(str(tmp_path), retention_days=7) == 2
    assert (tmp_path / today / "new.jpg").exists()
    assert not (tmp_path / "2020-01-01").exists()


def test_prune_leaves_directories_it_does_not_recognise(tmp_path):
    (tmp_path / "not-a-date").mkdir()
    (tmp_path / "not-a-date" / "keep.jpg").write_bytes(b"x")
    assert prune_old(str(tmp_path), retention_days=1) == 0
    assert (tmp_path / "not-a-date" / "keep.jpg").exists()


def test_prune_is_disabled_by_a_zero_retention(tmp_path):
    (tmp_path / "2020-01-01").mkdir()
    assert prune_old(str(tmp_path), retention_days=0) == 0
    assert (tmp_path / "2020-01-01").exists()


def test_prune_on_a_missing_directory_is_harmless(tmp_path):
    assert prune_old(str(tmp_path / "nope")) == 0
