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


def test_gold_lexicon_has_an_independent_public_checksum_pin():
    filename = "frontend/en_gold_pronunciations.json"
    query = parse_qs(urlsplit(md._source_url(md.VITS2_MODEL_BASE, filename)).query)
    assert query["Revision"] == [md.VITS2_MODEL_REVISION]
    assert query["FilePath"] == [filename]
    assert md.VITS2_FRONTEND_FILES[filename] == {
        "size": 672106,
        "sha256": "0c6fd2adb51aeae0fd83968f1902dedb16c73db695a3e3ad088bfc23cc05d57a",
    }


@pytest.fixture
def gold_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    filename = "frontend/en_gold_pronunciations.json"
    path = source / filename
    path.parent.mkdir()
    payload = b'{"SQL":["EH2","S","K","Y","UW2","EH1","L"]}'
    path.write_bytes(payload)
    monkeypatch.setattr(md, "VITS2_FRONTEND_FILES", {
        filename: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
    })
    monkeypatch.setattr(md, "VITS2_MODEL_BASE", source.as_uri())
    guard = md.require_models_subpath
    monkeypatch.setattr(md, "require_models_subpath",
                        lambda path: guard(path, root=str(tmp_path)))
    monkeypatch.setattr(md.time, "sleep", lambda _: None)
    return path, payload


@pytest.mark.parametrize("family", ["jp61", "jp511"])
def test_original_archive_layout_is_installed_and_reused(
    family, tmp_path, monkeypatch, gold_source,
):
    """Exercise real verified extraction and warm reuse, without network/GPU."""
    root_name = md.VITS2_MODEL_ARCHIVES[family]["release_root"]
    source = tmp_path / "source"
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
    destination = tmp_path / "models"
    engine_dir = md.ensure_vits2_model(str(destination), family=family)
    release_root = Path(md.vits2_release_root(engine_dir))
    assert release_root == destination / root_name
    for relative, data in payloads.items():
        assert (release_root / relative).read_bytes() == data
    gold_path, gold_bytes = gold_source
    assert (release_root / "frontend" / gold_path.name).read_bytes() == gold_bytes

    # Even after the source disappears, the pinned installed marker permits
    # the same release to start without another fetch or an engine rebuild.
    archive.unlink()
    gold_path.unlink()
    # A warm startup must not need the network for either resource.
    monkeypatch.setattr(md, "urlopen", lambda *a, **k: pytest.fail("warm cache fetched"))
    assert md.ensure_vits2_model(str(destination), family=family) == engine_dir


@pytest.mark.parametrize("family", ["jp61", "jp511"])
def test_cached_archive_still_installs_or_repairs_gold(
    family, tmp_path, monkeypatch, gold_source,
):
    """Upgrade existing installations, including a same-size corrupted overlay."""
    monkeypatch.setattr(md, "ensure_verified_archive", lambda *a, **k: None)
    destination = tmp_path / "models"
    progress = []
    engine_dir = md.ensure_vits2_model(
        str(destination), family=family, progress_cb=lambda *args: progress.append(args),
    )
    target = Path(md.vits2_release_root(engine_dir)) / "frontend/en_gold_pronunciations.json"
    assert target.read_bytes() == gold_source[1]
    assert progress[-1][0] == 100
    target.write_bytes(b"x" * len(gold_source[1]))
    md.ensure_vits2_model(str(destination), family=family)
    assert target.read_bytes() == gold_source[1]


def test_bad_gold_download_fails_before_a_release_can_be_returned(
    tmp_path, monkeypatch, gold_source,
):
    monkeypatch.setattr(md, "ensure_verified_archive", lambda *a, **k: None)
    gold_source[0].write_bytes(b"x" * len(gold_source[1]))
    destination = tmp_path / "models"
    with pytest.raises(RuntimeError, match="en_gold_pronunciations") as error:
        md.ensure_vits2_model(str(destination), family="jp61")
    assert "SHA256 mismatch" in str(error.value.__cause__)
    target = destination / md.VITS2_MODEL_ARCHIVES["jp61"]["release_root"]
    assert not (target / "frontend/en_gold_pronunciations.json").exists()


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
