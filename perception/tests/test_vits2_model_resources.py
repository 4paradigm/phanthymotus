"""VITS2 public release selection and original archive layout compatibility."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from urllib.parse import parse_qs, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import model_downloader as md


@pytest.mark.parametrize("family", ["jp61", "jp511"])
def test_public_release_url_is_immutable_and_personal(family):
    entry = md.VITS2_MODEL_ARCHIVES[family]
    url = md._source_url(md.VITS2_MODEL_BASE, entry["archive"])
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "www.modelscope.cn"
    assert parsed.path == "/api/v1/models/lanxijiang/TTSV4_Ch_Eng/repo"
    assert query["Revision"] == [md.VITS2_MODEL_REVISION]
    assert query["FilePath"] == [entry["archive"]]
    assert len(entry["sha256"]) == 64


@pytest.mark.parametrize("family", ["jp61", "jp511"])
def test_original_archive_layout_is_installed_and_reused(family, tmp_path, monkeypatch):
    """Exercise real verified extraction and warm reuse, without network/GPU."""
    root_name = md.VITS2_MODEL_ARCHIVES[family]["release_root"]
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "release.tar.gz"
    payloads = {
        "config.json": b'{"data":{"sampling_rate":16000}}',
        "engines/manifest.json": json.dumps({"family": family}).encode(),
        "engines/decoder.plan": b"original engine bytes",
        "tn_cache/zh_tn_tagger.fst": b"original TN bytes",
        "frontend_data/cmudict.rep": b"original pronunciations",
    }
    with tarfile.open(archive, "w:gz") as tf:
        for relative, data in payloads.items():
            member = tarfile.TarInfo(f"{root_name}/{relative}")
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))
    blob = archive.read_bytes()
    entry = {
        "archive": archive.name, "release_root": root_name,
        "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
    }
    monkeypatch.setitem(md.VITS2_MODEL_ARCHIVES, family, entry)
    monkeypatch.setattr(md, "VITS2_MODEL_BASE", source.as_uri())
    guard = md.require_models_subpath
    monkeypatch.setattr(md, "require_models_subpath",
                        lambda path: guard(path, root=str(tmp_path)))
    destination = tmp_path / "models"
    engine_dir = md.ensure_vits2_model(str(destination), family=family)
    release_root = Path(md.vits2_release_root(engine_dir))
    assert release_root == destination / root_name
    for relative, data in payloads.items():
        assert (release_root / relative).read_bytes() == data

    # Even after the source disappears, the pinned installed marker permits
    # the same release to start without another fetch or an engine rebuild.
    archive.unlink()
    assert md.ensure_vits2_model(str(destination), family=family) == engine_dir


@pytest.mark.parametrize("engine_dir, expected", [
    ("/models/vits2/release/engines", "/models/vits2/release"),
    ("/models/vits2/engines/jp61", "/models/vits2"),
    ("/models/vits2/engines/jp511/", "/models/vits2"),
])
def test_release_root_supports_both_installed_layouts(engine_dir, expected):
    assert md.vits2_release_root(engine_dir) == expected


def test_unknown_engine_layout_is_not_guessed():
    with pytest.raises(ValueError, match="directory layout"):
        md.vits2_release_root("/models/vits2/unrecognized")
