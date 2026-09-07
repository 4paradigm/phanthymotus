"""
tests/test_face_db.py — FaceDB persistence, id allocation, capacity, meta CRUD.

Pure host-side: no models, no ROS. Everything here is disk + numpy.
Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (puts perception on sys.path)

import plugins.face_db as face_db_module  # noqa: E402
from plugins.face_db import EMBEDDING_DIM, FaceDB, FaceDBError, is_unknown_id  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_tmp_db(monkeypatch):
    """FaceDB confines db_dir to /models; tests write to tmp_path instead.

    Patching the guard rather than the path keeps the production invariant
    (`require_models_subpath`) intact and covered by its own tests.
    """
    monkeypatch.setattr(
        face_db_module, "require_models_subpath", lambda path, root="/models": str(path)
    )


def _vector(seed: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """A deterministic unit vector. Distinct seeds are near-orthogonal."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    return raw / np.linalg.norm(raw)


def _nudge(vector: np.ndarray, amount: float, seed: int = 99) -> np.ndarray:
    """A vector close to `vector` — stands in for the same face, second photo."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(vector.shape).astype(np.float32)
    noise -= noise @ vector * vector
    noise /= np.linalg.norm(noise)
    mixed = vector + amount * noise
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


# ── basics ────────────────────────────────────────────────────────────────────

def test_add_and_match_roundtrip(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    record = db.add("Alice", [_vector(1)], meta={"team": "ops"})
    assert record["id"] == "p-1"
    assert record["named"] is True
    assert record["samples"] == 1

    matched, score = db.match(_vector(1), 0.35)
    assert matched == "p-1"
    assert score == pytest.approx(1.0, abs=1e-5)

    missed, best = db.match(_vector(2), 0.35)
    assert missed is None
    assert best < 0.35


def test_match_on_empty_db_reports_no_best_score(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    matched, score = db.match(_vector(1), 0.35)
    assert matched is None
    assert score == -1.0


def test_best_of_samples_not_the_centroid(tmp_path):
    """A person's score is their best sample, so extra poses only help.

    With a mean vector, enrolling a second, different-looking photo would pull
    the centroid away from both and could push a real match below threshold.
    """
    db = FaceDB(db_dir=str(tmp_path))
    front = _vector(1)
    profile_pose = _nudge(front, 1.4, seed=7)
    record = db.add("Alice", [front, profile_pose])
    assert record["samples"] == 2

    for probe in (front, profile_pose):
        matched, score = db.match(probe, 0.35)
        assert matched == "p-1"
        assert score == pytest.approx(1.0, abs=1e-5)


def test_rejects_wrong_dimension_and_zero_norm(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    with pytest.raises(ValueError):
        db.add("Bad", [np.ones(7, dtype=np.float32)])
    with pytest.raises(ValueError):
        db.add("Bad", [np.zeros(EMBEDDING_DIM, dtype=np.float32)])
    with pytest.raises(ValueError):
        db.add("Bad", [])


# ── persistence ───────────────────────────────────────────────────────────────

def test_survives_reload(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(1)], meta={"team": "ops"})
    db.enroll_unknown(_vector(5))

    reopened = FaceDB(db_dir=str(tmp_path))
    assert reopened.stats()["persons"] == 2
    assert reopened.get_person("p-1")["meta"] == {"team": "ops"}
    matched, _ = reopened.match(_vector(5), 0.35)
    assert matched == "unknown-1"


def test_persons_json_is_the_commit_point(tmp_path):
    """persons.json names the embeddings file, so the pair is never mismatched.

    Simulates a crash *after* the embeddings write and *before* the metadata
    replace by deleting the new persons.json: the previous committed state must
    load cleanly rather than pairing new rows with old owners.
    """
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(1)])
    first = json.loads((tmp_path / "persons.json").read_text())
    assert first["embeddings_file"] == "embeddings-1.npy"
    assert len(first["rows"]) == 1

    db.add("Bob", [_vector(2)])
    second = json.loads((tmp_path / "persons.json").read_text())
    assert second["embeddings_file"] == "embeddings-2.npy"
    # The superseded generation is only unlinked after the commit succeeds.
    assert not (tmp_path / "embeddings-1.npy").exists()

    # Roll the metadata back to generation 1 while generation 2's matrix is on
    # disk: the loader must follow persons.json, and generation 1's file is
    # gone, so this is a genuinely broken database and must say so.
    (tmp_path / "persons.json").write_text(json.dumps(first))
    with pytest.raises(FaceDBError):
        FaceDB(db_dir=str(tmp_path))


def test_row_count_mismatch_is_refused(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(1)])
    state = json.loads((tmp_path / "persons.json").read_text())
    state["rows"].append("p-1")            # one more owner than there are rows
    (tmp_path / "persons.json").write_text(json.dumps(state))
    with pytest.raises(FaceDBError):
        FaceDB(db_dir=str(tmp_path))


def test_orphaned_rows_are_dropped_not_misattributed(tmp_path):
    """A row owned by a deleted person must not be silently reassigned."""
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(1)])
    db.add("Bob", [_vector(2)])
    state = json.loads((tmp_path / "persons.json").read_text())
    state["persons"] = [p for p in state["persons"] if p["id"] != "p-2"]
    (tmp_path / "persons.json").write_text(json.dumps(state))

    reopened = FaceDB(db_dir=str(tmp_path))
    assert reopened.stats()["persons"] == 1
    assert reopened.stats()["samples"] == 1
    matched, _ = reopened.match(_vector(2), 0.35)
    assert matched is None


# ── ids ───────────────────────────────────────────────────────────────────────

def test_forgotten_id_is_never_reused(tmp_path):
    """A retired id must not resolve to a different person later.

    An `unknown-3` may already have been published on the activity stream and
    recorded in the agent's history; handing that id to somebody else would
    silently rewrite who those sightings were about.
    """
    db = FaceDB(db_dir=str(tmp_path))
    first = db.enroll_unknown(_vector(1))["id"]
    assert first == "unknown-1"
    assert db.forget(first) is True

    second = db.enroll_unknown(_vector(2))["id"]
    assert second == "unknown-2"

    reopened = FaceDB(db_dir=str(tmp_path))
    third = reopened.enroll_unknown(_vector(3))["id"]
    assert third == "unknown-3"


def test_forget_unknown_id_helper():
    assert is_unknown_id("unknown-4")
    assert not is_unknown_id("p-4")


def test_forget_missing_person_is_false(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    assert db.forget("p-999") is False


# ── promotion ─────────────────────────────────────────────────────────────────

def test_promote_keeps_the_id(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    unknown = db.enroll_unknown(_vector(1))["id"]
    promoted = db.promote(unknown, "Carol", meta={"floor": 3})

    assert promoted["id"] == unknown          # id preserved, deliberately
    assert promoted["named"] is True
    assert promoted["profile"] == "Carol"
    assert promoted["meta"] == {"floor": 3}
    stats = db.stats()
    assert (stats["named"], stats["unknown"]) == (1, 0)


def test_setting_a_blank_profile_does_not_promote(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    unknown = db.enroll_unknown(_vector(1))["id"]
    record = db.update_person(unknown, profile="   ")
    assert record["named"] is False


# ── meta CRUD ─────────────────────────────────────────────────────────────────

def test_meta_merge_replace_and_delete(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(1)], meta={"team": "ops", "floor": 3})

    merged = db.update_person("p-1", meta={"floor": 4, "badge": "A7"})
    assert merged["meta"] == {"team": "ops", "floor": 4, "badge": "A7"}

    replaced = db.update_person("p-1", meta={"only": "this"}, merge=False)
    assert replaced["meta"] == {"only": "this"}

    trimmed = db.update_person("p-1", meta={"keep": 1}, meta_delete=["only"])
    assert trimmed["meta"] == {"keep": 1}

    assert FaceDB(db_dir=str(tmp_path)).get_person("p-1")["meta"] == {"keep": 1}


def test_meta_must_be_a_json_object(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(1)])
    with pytest.raises(ValueError):
        db.update_person("p-1", meta=["not", "an", "object"])
    with pytest.raises(ValueError):
        db.update_person("p-1", meta={"bad": {1, 2}})     # a set is not JSON
    # The rejected write must not have changed anything.
    assert db.get_person("p-1")["meta"] == {}


def test_update_and_get_missing_person_raise_keyerror(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    with pytest.raises(KeyError):
        db.get_person("p-1")
    with pytest.raises(KeyError):
        db.update_person("p-1", profile="x")
    with pytest.raises(KeyError):
        db.add_samples("p-1", [_vector(1)])


# ── capacity ──────────────────────────────────────────────────────────────────

def test_unknown_capacity_evicts_least_recently_seen(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), unknown_capacity=3)
    ids = [db.enroll_unknown(_vector(seed))["id"] for seed in range(3)]
    # unknown-1 is the least recently seen once the others are touched.
    db.touch(ids[1], when=5_000.0)
    db.touch(ids[2], when=6_000.0)
    db.touch(ids[0], when=1_000.0)

    fourth = db.enroll_unknown(_vector(50))["id"]
    remaining = {p["id"] for p in db.list_persons(named="unknown")["persons"]}
    assert ids[0] not in remaining
    assert remaining == {ids[1], ids[2], fourth}


def test_lowering_capacity_evicts_now_and_spares_named(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), unknown_capacity=10)
    db.add("Alice", [_vector(900)])
    for seed in range(6):
        db.touch(db.enroll_unknown(_vector(seed))["id"], when=1_000.0 + seed)

    evicted = db.set_unknown_capacity(2)
    assert evicted == 4
    stats = db.stats()
    assert stats["unknown"] == 2
    assert stats["named"] == 1                    # never a candidate
    assert db.get_person("p-1")["profile"] == "Alice"


def test_zero_capacity_refuses_to_enrol_unknowns(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), unknown_capacity=0)
    with pytest.raises(FaceDBError):
        db.enroll_unknown(_vector(1))
    assert db.stats()["unknown"] == 0


def test_forget_unknowns_leaves_named_intact(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(900)])
    db.enroll_unknown(_vector(1))
    db.enroll_unknown(_vector(2))

    assert db.forget_unknowns() == 2
    stats = db.stats()
    assert (stats["named"], stats["unknown"], stats["samples"]) == (1, 0, 1)
    matched, _ = db.match(_vector(900), 0.35)
    assert matched == "p-1"


def test_samples_are_capped_keeping_the_newest(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), max_samples_per_person=2)
    oldest = _vector(1)
    db.add("Alice", [oldest])
    newer = _vector(2)
    newest = _vector(3)
    record = db.add_samples("p-1", [newer])
    record = db.add_samples("p-1", [newest])

    assert record["samples"] == 2
    assert db.match(oldest, 0.9)[0] is None       # evicted
    assert db.match(newer, 0.9)[0] == "p-1"
    assert db.match(newest, 0.9)[0] == "p-1"


# ── listing ───────────────────────────────────────────────────────────────────

def test_list_persons_filters_and_pages(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice from ops", [_vector(1)], meta={"badge": "A7"})
    db.add("Bob from sales", [_vector(2)])
    db.enroll_unknown(_vector(3))

    assert db.list_persons()["total"] == 3
    assert db.list_persons(named="named")["total"] == 2
    assert db.list_persons(named="unknown")["total"] == 1
    assert db.list_persons(query="sales")["total"] == 1
    assert db.list_persons(query="a7")["total"] == 1            # matches meta
    assert db.list_persons(query="unknown-")["total"] == 1      # matches id

    page = db.list_persons(limit=1, offset=1)
    assert len(page["persons"]) == 1 and page["total"] == 3
    assert "embedding" not in json.dumps(page)


# ── concurrency ───────────────────────────────────────────────────────────────

def test_concurrent_enrolment_keeps_every_person_and_row(tmp_path):
    """Registration runs on arbitrary ThreadingHTTPServer threads."""
    db = FaceDB(db_dir=str(tmp_path))
    errors: list[Exception] = []

    def worker(seed: int):
        try:
            db.add(f"person-{seed}", [_vector(seed)])
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    stats = db.stats()
    assert stats["persons"] == 12
    assert stats["samples"] == 12
    assert len({p["id"] for p in db.list_persons(limit=50)["persons"]}) == 12

    reloaded = FaceDB(db_dir=str(tmp_path))
    assert reloaded.stats()["samples"] == 12
    for seed in range(12):
        assert reloaded.match(_vector(seed), 0.9)[0] is not None
