"""
tests/test_identity_db_artifacts.py — per-sample source artifacts.

An embedding is only meaningful inside the network that produced it, so a model
switch can never be migrated from the embeddings alone — only re-enrolled by hand.
Storing the aligned crop each sample was computed from is what makes a backfill
possible at all.

Every test here is about one hazard: the artifact must stay attached to the right
sample. Rows are reindexed by the max-samples trim, by `forget`, by capacity
eviction and by the orphan filter on load, and an artifact filed under a row index
would — after any of those — be served up as a different person's face. That is
worse than having no artifact, because it is wrong rather than absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (installs stubs, sets sys.path)

import plugins.identity_db as identity_db  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_tmp_dirs(monkeypatch):
    """The real guard confines db_dir to /models; tests write to tmp_path."""
    monkeypatch.setattr(identity_db, "require_models_subpath",
                        lambda path, root=None: path)


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    raw = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return raw / np.linalg.norm(raw)


def _db(tmp_path, suffix=".jpg", **kw):
    return identity_db.IdentityDB(str(tmp_path), dim=8, artifact_suffix=suffix,
                                  label="test", **kw)


def _payloads(db):
    return [open(item["path"], "rb").read() for item in db.sample_artifacts()]


def test_disabled_by_default_so_voiceprints_do_not_pay_for_it(tmp_path):
    db = identity_db.IdentityDB(str(tmp_path), dim=8, label="test")
    assert db.artifacts_enabled is False
    db.add("a", [_vec(1)], artifacts=[b"X"])
    assert db.sample_artifacts() == []
    assert db.stats()["samples_with_artifact"] == 0
    assert not os.path.isdir(os.path.join(str(tmp_path), "samples"))


def test_an_artifact_is_stored_and_readable_back(tmp_path):
    db = _db(tmp_path)
    db.add("alice", [_vec(1), _vec(2)], artifacts=[b"A1", b"A2"])
    assert _payloads(db) == [b"A1", b"A2"]
    stats = db.stats()
    assert stats["artifacts_enabled"] is True
    assert stats["samples_with_artifact"] == stats["samples"] == 2


def test_a_sample_without_an_artifact_is_allowed(tmp_path):
    """The encoder is best-effort; a missing crop must not fail enrolment."""
    db = _db(tmp_path)
    db.add("alice", [_vec(1), _vec(2)], artifacts=[b"A1", None])
    assert _payloads(db) == [b"A1"]
    assert db.stats()["samples"] == 2
    assert db.stats()["samples_with_artifact"] == 1


def test_fewer_artifacts_than_embeddings_is_padded_not_misaligned(tmp_path):
    """A short list must not shift artifacts onto the wrong rows."""
    db = _db(tmp_path)
    db.add("alice", [_vec(1), _vec(2), _vec(3)], artifacts=[b"A1"])
    items = db.sample_artifacts()
    assert [item["row"] for item in items] == [0]
    assert _payloads(db) == [b"A1"]


def test_the_max_samples_trim_drops_the_oldest_artifacts(tmp_path):
    db = _db(tmp_path, max_samples_per_person=3)
    db.add("alice", [_vec(1), _vec(2)], artifacts=[b"A1", b"A2"])
    db.add_samples("p-1", [_vec(3), _vec(4)], artifacts=[b"A3", b"A4"])
    # oldest-first eviction, so A1 goes and the newest three survive
    assert _payloads(db) == [b"A2", b"A3", b"A4"]
    assert all(item["person_id"] == "p-1" for item in db.sample_artifacts())
    # and the dropped file is gone from disk, not merely unreferenced
    on_disk = os.listdir(os.path.join(str(tmp_path), "samples"))
    assert len(on_disk) == 3, on_disk


def test_forget_sweeps_that_persons_artifacts_only(tmp_path):
    db = _db(tmp_path)
    db.add("alice", [_vec(1)], artifacts=[b"A1"])
    db.add("bob", [_vec(2)], artifacts=[b"B1"])
    db.forget("p-1")
    assert _payloads(db) == [b"B1"]
    assert len(os.listdir(os.path.join(str(tmp_path), "samples"))) == 1


def test_capacity_eviction_sweeps_artifacts(tmp_path):
    db = _db(tmp_path, unknown_capacity=1)
    db.enroll_unknown(_vec(1), artifact=b"U1")
    db.enroll_unknown(_vec(2), artifact=b"U2")
    # one unknown evicted by the second enrolment; only the survivor's file remains
    assert _payloads(db) == [b"U2"]
    assert len(os.listdir(os.path.join(str(tmp_path), "samples"))) == 1


def test_associations_survive_a_reopen(tmp_path):
    db = _db(tmp_path)
    db.add("alice", [_vec(1)], artifacts=[b"A1"])
    db.add("bob", [_vec(2)], artifacts=[b"B1"])
    reopened = _db(tmp_path)
    pairs = [(item["person_id"], open(item["path"], "rb").read())
             for item in reopened.sample_artifacts()]
    assert pairs == [("p-1", b"A1"), ("p-2", b"B1")]


def test_a_retired_sample_id_is_never_reused(tmp_path):
    """The hazard: a new sample adopting a deleted sample's file.

    Ids come from a counter that only increases and is restored from disk on load,
    so a fresh enrolment after a reopen cannot land on a retired id.
    """
    db = _db(tmp_path)
    db.add("alice", [_vec(1)], artifacts=[b"A1"])
    first = db.sample_artifacts()[0]["sample_id"]
    db.forget("p-1")
    reopened = _db(tmp_path)
    reopened.add("bob", [_vec(2)], artifacts=[b"B1"])
    item = reopened.sample_artifacts()[0]
    assert item["sample_id"] != first
    assert open(item["path"], "rb").read() == b"B1"


def test_a_database_written_before_artifacts_existed_still_opens(tmp_path):
    """Back-compat: no sample_rows means no artifacts, not a guess."""
    db = identity_db.IdentityDB(str(tmp_path), dim=8, label="test")
    db.add("alice", [_vec(1), _vec(2)])
    reopened = _db(tmp_path)
    assert reopened.sample_artifacts() == []
    assert reopened.stats()["samples"] == 2
    assert reopened.stats()["samples_with_artifact"] == 0
    # and it can still take new samples, which do get artifacts
    reopened.add_samples("p-1", [_vec(3)], artifacts=[b"A3"])
    assert _payloads(reopened) == [b"A3"]


def test_a_sample_rows_of_the_wrong_length_is_ignored_not_trusted(tmp_path):
    """A hand-edited persons.json must not attach the wrong face to a person."""
    import json
    db = _db(tmp_path)
    db.add("alice", [_vec(1), _vec(2)], artifacts=[b"A1", b"A2"])
    path = os.path.join(str(tmp_path), "persons.json")
    with open(path) as handle:
        state = json.load(handle)
    state["sample_rows"] = state["sample_rows"][:1]      # now disagrees with rows
    with open(path, "w") as handle:
        json.dump(state, handle)
    reopened = _db(tmp_path)
    assert reopened.sample_artifacts() == []
    assert reopened.stats()["samples"] == 2


def test_an_unreferenced_file_is_swept_on_the_next_commit(tmp_path):
    """What a crash between writing an artifact and committing leaves behind."""
    db = _db(tmp_path)
    db.add("alice", [_vec(1)], artifacts=[b"A1"])
    stray = os.path.join(str(tmp_path), "samples", "s-9999.jpg")
    with open(stray, "wb") as handle:
        handle.write(b"orphan")
    db.add("bob", [_vec(2)], artifacts=[b"B1"])          # any commit sweeps
    assert not os.path.exists(stray)
    assert sorted(_payloads(db)) == [b"A1", b"B1"]
