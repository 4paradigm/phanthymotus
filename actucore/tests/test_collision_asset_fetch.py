"""Exercise the real download path without an external service or robot."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "fetch_g1_collision", Path(__file__).resolve().parents[2] / "deploy/fetch_g1_collision.py")
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


@pytest.fixture
def asset(tmp_path):
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    content = b"verified original mesh"
    (manifest / "sha256.json").write_text(json.dumps({
        "torso_link.STL": hashlib.sha256(content).hexdigest()}))
    return manifest, tmp_path / "assets", content


def test_verified_download_and_offline_cache(monkeypatch, asset):
    manifest, dest, content = asset
    urls = []
    def download(url, timeout):
        urls.append((url, timeout))
        return io.BytesIO(content)
    monkeypatch.setattr(fetcher, "urlopen", download)
    assert fetcher.fetch(manifest, dest) == 1
    assert urls == [(fetcher.SOURCE + "torso_link.STL", 30)]
    monkeypatch.setattr(fetcher, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    assert fetcher.fetch(manifest, dest) == fetcher.fetch(manifest, dest, True) == 1
    assert (dest / "torso_link.STL").read_bytes() == content


@pytest.mark.parametrize("failure", ["corrupt", "timeout", "too_large"])
def test_failed_download_never_publishes_partial_file(monkeypatch, asset, failure):
    manifest, dest, content = asset
    dest.mkdir()
    target = dest / "torso_link.STL"
    target.write_bytes(b"old invalid file")
    def download(*args, **kwargs):
        if failure == "timeout":
            raise TimeoutError("fixture timeout")
        return io.BytesIO(b"wrong bytes" if failure == "corrupt" else content)
    monkeypatch.setattr(fetcher, "urlopen", download)
    if failure == "too_large":
        monkeypatch.setattr(fetcher, "MAX_ASSET_BYTES", 4)
    with pytest.raises((ValueError, TimeoutError)):
        fetcher.fetch(manifest, dest)
    assert target.read_bytes() == b"old invalid file"
    assert list(dest.glob("*.part")) == []


def test_check_does_not_create_or_download(monkeypatch, asset):
    manifest, dest, _ = asset
    monkeypatch.setattr(fetcher, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="missing or corrupt"):
        fetcher.fetch(manifest, dest, True)
    assert not dest.exists()


def test_manifest_cannot_escape_destination(asset):
    manifest, dest, content = asset
    (manifest / "sha256.json").write_text(json.dumps({
        "../escape.STL": hashlib.sha256(content).hexdigest()}))
    with pytest.raises(ValueError, match="invalid collision asset manifest"):
        fetcher.fetch(manifest, dest)
    assert not dest.exists()
