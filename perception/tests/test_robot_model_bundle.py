"""The card cannot load an unpinned or missing model artifact."""

import hashlib

import pytest

from utils import model_downloader as downloader


def _pin(data):
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_missing_robot_model_artifact_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "require_models_subpath", lambda path: path)
    with pytest.raises(RuntimeError, match="robot model bundle"):
        downloader.ensure_robot_model_bundle(str(tmp_path))


def test_local_bundle_is_verified_before_use(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "require_models_subpath", lambda path: path)
    files = {"clip_image.onnx": b"local encoder",
             "candidates.npz": b"local candidates"}
    monkeypatch.setattr(downloader, "ROBOT_MODEL_BUNDLES", {
        "clip-vit-b32": {"files": {name: _pin(data)
                                   for name, data in files.items()}}
    })
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)

    paths = downloader.ensure_robot_model_bundle(str(tmp_path))
    assert set(paths) == set(files)

    (tmp_path / "candidates.npz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="verification failed"):
        downloader.ensure_robot_model_bundle(str(tmp_path))
