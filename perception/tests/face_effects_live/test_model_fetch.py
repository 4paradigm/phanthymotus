"""Offline regression tests for the opt-in, verified model-fetch CLI."""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading

import pytest

from tools.face_effects_live import fetch_models


MODEL = b"synthetic fixed test model bytes"


def spec():
    return {
        "file": "local-model.onnx",
        "url": "https://example.org/models/upstream-name.onnx",
        "bytes": len(MODEL),
        "sha256": hashlib.sha256(MODEL).hexdigest(),
        "license": "TEST",
    }


def fake_downloader(calls):
    def ensure_verified_bundle(name, model_dir, base_url, files, progress_cb=None):
        calls.append((name, base_url, files))
        assert base_url == "https://example.org/models"
        assert files == {"upstream-name.onnx": {
            "size": len(MODEL), "sha256": hashlib.sha256(MODEL).hexdigest(),
        }}
        root = Path(model_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "upstream-name.onnx"
        path.write_bytes(MODEL)
        if progress_cb is not None:
            progress_cb(50, 1, 2)
        return {"upstream-name.onnx": str(path)}

    return ensure_verified_bundle


def test_shared_verified_downloader_reports_progress_and_publishes_alias(tmp_path, monkeypatch):
    calls, statuses = [], []
    monkeypatch.setattr(fetch_models, "ensure_verified_bundle", fake_downloader(calls))
    target = fetch_models.fetch_one("yunet", spec(), tmp_path, statuses.append)
    assert target == tmp_path / "local-model.onnx"
    assert target.read_bytes() == MODEL
    assert len(calls) == 1
    assert any("50%" in message for message in statuses)
    assert any("downloaded and verified" in message for message in statuses)
    cached = tmp_path / ".verified" / "yunet" / "upstream-name.onnx"
    assert cached.exists()
    assert target.stat().st_ino == cached.stat().st_ino


def test_existing_valid_file_is_reused_without_network(tmp_path, monkeypatch):
    target = tmp_path / spec()["file"]
    target.write_bytes(MODEL)

    def unexpected_download(**kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(fetch_models, "ensure_verified_bundle", unexpected_download)
    assert fetch_models.fetch_one("yunet", spec(), tmp_path) == target


def test_existing_corrupted_file_is_never_overwritten(tmp_path, monkeypatch):
    target = tmp_path / spec()["file"]
    target.write_bytes(b"not the model")
    monkeypatch.setattr(fetch_models, "ensure_verified_bundle",
                        lambda **kwargs: pytest.fail("should not download"))
    with pytest.raises(RuntimeError, match="size mismatch"):
        fetch_models.fetch_one("yunet", spec(), tmp_path)
    assert target.read_bytes() == b"not the model"


def test_corrupt_downloader_result_cannot_be_published(tmp_path, monkeypatch):
    def corrupt(name, model_dir, base_url, files, progress_cb=None):
        root = Path(model_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "upstream-name.onnx"
        path.write_bytes(b"corrupt")
        return {"upstream-name.onnx": str(path)}

    monkeypatch.setattr(fetch_models, "ensure_verified_bundle", corrupt)
    with pytest.raises(RuntimeError, match="size mismatch"):
        fetch_models.fetch_one("yunet", spec(), tmp_path)
    assert not (tmp_path / spec()["file"]).exists()


def test_concurrent_publication_keeps_one_verified_alias(tmp_path, monkeypatch):
    source = tmp_path / ".verified" / "yunet" / "upstream-name.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(MODEL)
    barrier = threading.Barrier(2)

    def verified(name, model_dir, base_url, files, progress_cb=None):
        barrier.wait(timeout=5)
        return {"upstream-name.onnx": str(source)}

    monkeypatch.setattr(fetch_models, "ensure_verified_bundle", verified)
    outputs, failures = [], []

    def run():
        try:
            outputs.append(fetch_models.fetch_one("yunet", spec(), tmp_path))
        except Exception as error:
            failures.append(error)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not failures
    assert len(outputs) == 2
    assert all(out.read_bytes() == MODEL for out in outputs)


def test_invalid_source_and_local_path_are_rejected(tmp_path):
    bad = spec()
    bad["file"] = "../escape.onnx"
    with pytest.raises(ValueError, match="local model filename"):
        fetch_models.fetch_one("yunet", bad, tmp_path)

    bad = spec()
    bad["url"] = "http://example.org/model.onnx"
    with pytest.raises(ValueError, match="HTTPS"):
        fetch_models.fetch_one("yunet", bad, tmp_path)
