"""Pinned Windows artifacts are verified without querying mutable metadata."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1] / "deploy" / "windows_bridge"
spec = importlib.util.spec_from_file_location("visitor_offline_test", ROOT / "build_offline.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_bad_download_is_deleted_instead_of_being_uploaded(tmp_path):
    response = io.BytesIO(b"tampered payload")
    response.headers = {"Content-Length": "16"}
    target = tmp_path / "wheel.whl"
    with patch.object(builder.urllib.request, "urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="hash mismatch"):
            builder.download("https://example.test/wheel.whl", target, "0" * 64)
    assert not target.exists()


def test_build_uses_source_pins_without_fetching_package_metadata(tmp_path):
    calls = []
    pins = json.loads((ROOT / "offline-wheels.json").read_text(encoding="utf-8"))
    by_name = {item["filename"]: item for item in pins}
    def download(urls, target, expected):
        calls.append((urls, target.name, expected))
        with target.open("wb") as output:
            output.truncate(by_name.get(target.name, {}).get("size", 1))
        return expected
    with patch.object(builder, "download_from_mirrors", side_effect=download), patch.object(builder.urllib.request, "urlopen") as network:
        builder.build(tmp_path)
    network.assert_not_called()
    assert len(calls) == 5
    for urls, name, digest in calls[1:]:
        assert urls[-1] == by_name[name]["url"]
        assert digest == by_name[name]["sha256"]
    manifest = json.loads((tmp_path / "offline" / "manifest.json").read_text())
    assert len(manifest) == 5
    for name, digest in manifest.items():
        assert digest == hashlib.sha256((tmp_path / "offline" / name).read_bytes()).hexdigest()


def test_wheel_size_mismatch_stops_provisioning(tmp_path):
    def short_download(urls, target, expected):
        target.write_bytes(b"short")
    with patch.object(builder, "download_from_mirrors", side_effect=short_download):
        with pytest.raises(RuntimeError, match="size mismatch"):
            builder.build(tmp_path)
    assert not (tmp_path / "offline" / "playwright-1.51.0-py3-none-win_amd64.whl").exists()
