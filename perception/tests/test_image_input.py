"""
tests/test_image_input.py — the shared image-source rules.

These are security properties, not conveniences: the MCP server has no
authentication and runs as root in the container, so the path confinement here
is what stops a LAN caller probing the filesystem. They were tested through
plugins/face.py before the helpers were extracted for vop and ocr; testing them
directly means the next plugin to use them inherits the coverage too.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import pytest

import vision_stubs  # noqa: F401  (puts the perception root on sys.path)

from plugins import image_input  # noqa: E402
from plugins.image_input import (  # noqa: E402
    BadInput,
    check_under_roots,
    load_image_bytes,
    read_local,
)


@pytest.fixture
def cfg(tmp_path):
    return {"image_roots": [str(tmp_path)], "max_image_bytes": 1024}


# ── path confinement ─────────────────────────────────────────────────────────

def test_a_path_inside_a_root_is_accepted(tmp_path, cfg):
    target = tmp_path / "a.jpg"
    target.write_bytes(b"xx")
    assert check_under_roots(str(target), cfg) == str(target.resolve())


def test_a_path_outside_the_roots_is_refused(cfg):
    with pytest.raises(BadInput) as excinfo:
        check_under_roots("/etc/passwd", cfg)
    # The refusal has to say how to hand the file over, or the caller just
    # retries with another path they cannot reach either.
    assert "file/upload" in excinfo.value.detail


def test_a_symlink_escaping_a_root_is_refused(tmp_path, cfg):
    """Resolving after the check would let a link inside a root walk out of it."""
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"secret")
    link = tmp_path / "link.jpg"
    link.symlink_to(outside)
    with pytest.raises(BadInput):
        check_under_roots(str(link), cfg)


def test_a_root_prefix_is_not_a_root(tmp_path):
    """/modelsX must not pass because it starts with /models."""
    cfg = {"image_roots": [str(tmp_path / "models")]}
    (tmp_path / "models").mkdir()
    sneaky = tmp_path / "modelsX"
    sneaky.mkdir()
    with pytest.raises(BadInput):
        check_under_roots(str(sneaky / "a.jpg"), cfg)


# ── transfer caps ────────────────────────────────────────────────────────────

def test_an_oversized_file_is_refused(tmp_path, cfg):
    big = tmp_path / "big.jpg"
    big.write_bytes(b"x" * 2048)          # cap is 1024
    with pytest.raises(BadInput, match="transfer cap"):
        read_local(str(big), cfg, 1024)


def test_a_directory_is_not_an_image(tmp_path, cfg):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(BadInput, match="directory"):
        read_local(str(d), cfg, 1024)


# ── input channels ───────────────────────────────────────────────────────────

def test_base64_is_refused_with_a_pointer_to_what_works(cfg):
    """It was removed because an LLM truncated it in production, twice."""
    with pytest.raises(BadInput) as excinfo:
        load_image_bytes({"image_b64": "AAAA"}, cfg)
    assert "file/upload" in excinfo.value.detail


def test_one_of_path_or_url_is_required(cfg):
    with pytest.raises(BadInput, match="image_path or url"):
        load_image_bytes({}, cfg)


def test_a_local_path_is_read(tmp_path, cfg):
    target = tmp_path / "a.jpg"
    target.write_bytes(b"hello")
    data, source = load_image_bytes({"image_path": str(target)}, cfg)
    assert data == b"hello"
    assert source == str(target)


def test_a_url_is_fetched(monkeypatch, cfg):
    monkeypatch.setattr(image_input, "fetch_url", lambda url, max_bytes: b"bytes")
    data, source = load_image_bytes({"url": "https://example.com/a.jpg"}, cfg)
    assert data == b"bytes"
    assert source == "https://example.com/a.jpg"


def test_only_http_urls_are_accepted(cfg):
    with pytest.raises(BadInput, match="http"):
        load_image_bytes({"url": "file:///etc/passwd"}, cfg)


# ── the rejection names the caller's own action ──────────────────────────────

@pytest.mark.parametrize("action", ["register_by_url", "recognize_by_url"])
def test_rejections_name_the_action_the_caller_should_use(cfg, action):
    """"Use the _by_url action" makes the caller go and find which one."""
    with pytest.raises(BadInput) as excinfo:
        load_image_bytes({"image_b64": "AAAA"}, cfg, url_action=action)
    assert action in excinfo.value.detail

    with pytest.raises(BadInput) as excinfo:
        load_image_bytes({"image_path": "/etc/passwd"}, cfg, url_action=action)
    assert action in excinfo.value.detail


def test_bad_input_renders_as_a_dispatch_result():
    result = BadInput("nope", "/x.jpg").as_result()
    assert result == {"ok": False, "reason": "bad_input",
                      "detail": "nope", "source": "/x.jpg"}


def test_a_308_redirect_is_followed():
    """Python 3.8's HTTPRedirectHandler has no http_error_308.

    Without this, an ordinary image URL — ultralytics.com/images/bus.jpg among
    them — comes back as "cannot fetch ...: HTTP Error 308" rather than the
    image. face.py's register_by_url had the same gap before these helpers were
    shared; fixing it here fixes it for all three plugins.
    """
    handler = image_input._Redirect308()
    assert hasattr(handler, "http_error_308")
    # The opener the module actually uses must carry it, not just the class.
    assert any(isinstance(h, image_input._Redirect308)
               for h in image_input._OPENER.handlers)
