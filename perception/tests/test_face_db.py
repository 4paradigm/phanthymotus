"""
tests/test_face_db.py — FaceDB persistence, id allocation, capacity, profile CRUD.

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
    record = db.add("Alice", [_vector(1)], profile={"team": "ops"})
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
    db.add("Alice", [_vector(1)], profile={"team": "ops"})
    db.enroll_unknown(_vector(5))

    reopened = FaceDB(db_dir=str(tmp_path))
    assert reopened.stats()["persons"] == 2
    assert reopened.get_person("p-1")["profile"] == {"team": "ops"}
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
    promoted = db.promote(unknown, "Carol", profile={"floor": 3})

    assert promoted["id"] == unknown          # id preserved, deliberately
    assert promoted["named"] is True
    assert promoted["name"] == "Carol"
    assert promoted["profile"] == {"floor": 3}
    stats = db.stats()
    assert (stats["named"], stats["unknown"]) == (1, 0)


def test_setting_a_blank_name_does_not_promote(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    unknown = db.enroll_unknown(_vector(1))["id"]
    record = db.update_person(unknown, name="   ")
    assert record["named"] is False


# ── profile CRUD ─────────────────────────────────────────────────────────────────

def test_profile_merge_replace_and_delete(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(1)], profile={"team": "ops", "floor": 3})

    merged = db.update_person("p-1", profile={"floor": 4, "badge": "A7"})
    assert merged["profile"] == {"team": "ops", "floor": 4, "badge": "A7"}

    replaced = db.update_person("p-1", profile={"only": "this"}, merge=False)
    assert replaced["profile"] == {"only": "this"}

    trimmed = db.update_person("p-1", profile={"keep": 1}, profile_delete=["only"])
    assert trimmed["profile"] == {"keep": 1}

    assert FaceDB(db_dir=str(tmp_path)).get_person("p-1")["profile"] == {"keep": 1}


def test_profile_must_be_a_json_object(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    db.add("Alice", [_vector(1)])
    with pytest.raises(ValueError):
        db.update_person("p-1", profile=["not", "an", "object"])
    with pytest.raises(ValueError):
        db.update_person("p-1", profile={"bad": {1, 2}})     # a set is not JSON
    # The rejected write must not have changed anything.
    assert db.get_person("p-1")["profile"] == {}


def test_update_and_get_missing_person_raise_keyerror(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    with pytest.raises(KeyError):
        db.get_person("p-1")
    with pytest.raises(KeyError):
        db.update_person("p-1", name="x")
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
    assert db.get_person("p-1")["name"] == "Alice"


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
    db.add("Alice from ops", [_vector(1)], profile={"badge": "A7"})
    db.add("Bob from sales", [_vector(2)])
    db.enroll_unknown(_vector(3))

    assert db.list_persons()["total"] == 3
    assert db.list_persons(named="named")["total"] == 2
    assert db.list_persons(named="unknown")["total"] == 1
    assert db.list_persons(query="sales")["total"] == 1
    assert db.list_persons(query="a7")["total"] == 1            # matches profile
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


# ── visit log (访问记录表) ────────────────────────────────────────────────────

def _seen(db, person_id, at, topic="/cam"):
    db.record_sighting(person_id, at, topic)


def test_a_continuous_presence_is_one_visit_not_one_per_frame(tmp_path):
    """At 1 detection/second a per-frame log would be 86 400 rows a day."""
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=600.0)
    db.add("Alice", [_vector(1)])
    for offset in range(0, 300, 1):          # 5 minutes of sightings
        _seen(db, "p-1", 1_000_000 + offset)

    # Nothing written yet: the visit is still open.
    assert not (tmp_path / "visits.jsonl").exists()
    assert db.list_visits()["total"] == 1
    open_visit = db.list_visits()["visits"][0]
    assert open_visit["open"] is True
    assert open_visit["sightings"] == 300

    # Still open before the gap elapses, closed after. The last sighting is at
    # +299, so the gap is measured from there.
    last_seen = 1_000_000 + 299
    assert db.close_stale_visits(now=last_seen + 599) == 0
    assert db.close_stale_visits(now=last_seen + 601) == 1

    lines = (tmp_path / "visits.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1, "one visit, not one row per sighting"
    record = json.loads(lines[0])
    assert record["person_id"] == "p-1"
    assert record["first_seen"] == 1_000_000
    assert record["last_seen"] == 1_000_000 + 299
    assert record["sightings"] == 300


def test_a_short_absence_does_not_split_the_visit(tmp_path):
    """Someone turning their head must not fragment an afternoon."""
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=600.0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", 2_000_000)
    db.close_stale_visits(now=2_000_000 + 120)     # 2 min gap: still the same visit
    _seen(db, "p-1", 2_000_000 + 121)
    db.close_stale_visits(now=2_000_000 + 800)     # now it has really been quiet

    visits = db.list_visits()["visits"]
    assert len(visits) == 1
    assert visits[0]["sightings"] == 2


def test_separate_visits_when_the_gap_is_exceeded(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=60.0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", 3_000_000)
    db.close_stale_visits(now=3_000_000 + 100)
    _seen(db, "p-1", 3_000_000 + 200)
    db.close_stale_visits(now=3_000_000 + 400)
    assert db.list_visits()["total"] == 2


def test_list_visits_filters_by_overlap_not_containment(tmp_path):
    """Someone there 14:50-15:10 was there at 15:00."""
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=60.0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", 1_000)
    _seen(db, "p-1", 2_000)
    db.close_stale_visits(force=True)

    assert db.list_visits(since=1_500, until=1_600)["total"] == 1   # inside
    assert db.list_visits(since=500, until=1_200)["total"] == 1     # overlaps start
    assert db.list_visits(since=1_900, until=5_000)["total"] == 1   # overlaps end
    assert db.list_visits(since=3_000)["total"] == 0                # after
    assert db.list_visits(until=500)["total"] == 0                  # before


def test_list_visits_accepts_iso_timestamps(tmp_path):
    from datetime import datetime
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=60.0)
    db.add("Alice", [_vector(1)])
    when = datetime(2026, 9, 7, 15, 0, 0).timestamp()
    _seen(db, "p-1", when)
    db.close_stale_visits(force=True)

    assert db.list_visits(since="2026-09-07T14:00", until="2026-09-07T16:00")["total"] == 1
    assert db.list_visits(since="2026-09-07T16:00")["total"] == 0
    with pytest.raises(ValueError):
        db.list_visits(since="last tuesday")


def test_list_visits_filters_by_person_and_resolves_the_current_name(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=60.0)
    unknown = db.enroll_unknown(_vector(1))["id"]
    db.add("Bob", [_vector(2)])
    _seen(db, unknown, 10_000)
    _seen(db, "p-1", 10_000)
    db.close_stale_visits(force=True)

    assert db.list_visits(person_id=unknown)["total"] == 1
    assert db.list_visits()["total"] == 2

    # The visit was logged while they were anonymous; naming them later must
    # make the history readable rather than leaving a blank.
    db.update_person(unknown, name="Late-named Carol")
    entry = db.list_visits(person_id=unknown)["visits"][0]
    assert entry["name"] == "Late-named Carol"


def test_open_visits_are_checkpointed_against_power_loss(tmp_path):
    """A 10-minute gap means an all-afternoon visit lives in RAM for hours."""
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=600.0, visit_checkpoint_s=0.0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", 4_000_000)
    _seen(db, "p-1", 4_000_100)
    assert db.checkpoint_open_visits() is True
    assert (tmp_path / "visits-open.json").exists()

    # Simulate a power cut: a brand-new FaceDB over the same directory.
    recovered = FaceDB(db_dir=str(tmp_path), visit_gap_s=600.0)
    visits = recovered.list_visits()["visits"]
    assert len(visits) == 1
    assert visits[0]["sightings"] == 2
    assert visits[0]["first_seen"] == 4_000_000


def test_a_stale_checkpointed_visit_is_closed_on_recovery(tmp_path):
    """If the person left while we were down, the visit is completed, not resumed."""
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=1.0, visit_checkpoint_s=0.0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", 100.0)                # long in the past
    db.checkpoint_open_visits(force=True)

    recovered = FaceDB(db_dir=str(tmp_path), visit_gap_s=1.0)
    assert recovered.stats()["open_visits"] == 0
    lines = (tmp_path / "visits.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["recovered"] is True


def test_a_fresh_checkpointed_visit_is_resumed_not_duplicated(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=600.0, visit_checkpoint_s=0.0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", _now_for_test := __import__("time").time())
    db.checkpoint_open_visits(force=True)

    recovered = FaceDB(db_dir=str(tmp_path), visit_gap_s=600.0)
    assert recovered.stats()["open_visits"] == 1
    _seen(recovered, "p-1", _now_for_test + 1)
    recovered.close_stale_visits(force=True)
    lines = (tmp_path / "visits.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1, "the resumed visit must not become a second record"
    assert json.loads(lines[0])["sightings"] == 2


def test_checkpoint_is_throttled_and_cleared_when_nothing_is_open(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=60.0, visit_checkpoint_s=3600.0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", 5_000_000)
    assert db.checkpoint_open_visits() is True          # first one always writes
    assert db.checkpoint_open_visits() is False         # throttled
    db.close_stale_visits(force=True)
    assert not (tmp_path / "visits-open.json").exists()


def test_visit_log_is_trimmed_to_its_cap(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=1.0, visit_log_max=5)
    db.add("Alice", [_vector(1)])
    for index in range(12):
        _seen(db, "p-1", 6_000_000 + index * 100)
        db.close_stale_visits(now=6_000_000 + index * 100 + 50)
    lines = (tmp_path / "visits.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5
    # The newest are the ones kept.
    assert json.loads(lines[-1])["first_seen"] == 6_000_000 + 11 * 100


def test_a_torn_line_does_not_make_the_history_unreadable(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=1.0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", 7_000_000)
    db.close_stale_visits(force=True)
    with open(tmp_path / "visits.jsonl", "a") as handle:
        handle.write('{"person_id": "p-1", "first_se\n')     # crash mid-append
    assert db.list_visits()["total"] == 1


def test_visit_log_disabled_by_a_zero_cap(tmp_path):
    db = FaceDB(db_dir=str(tmp_path), visit_gap_s=1.0, visit_log_max=0)
    db.add("Alice", [_vector(1)])
    _seen(db, "p-1", 8_000_000)
    db.close_stale_visits(force=True)
    assert not (tmp_path / "visits.jsonl").exists()


# ── record shape / migration ─────────────────────────────────────────────────

def test_record_separates_name_from_free_form_profile(tmp_path):
    db = FaceDB(db_dir=str(tmp_path))
    record = db.add("小王", [_vector(1)], profile={"gender": "male", "team": "ops"})
    assert record["name"] == "小王"
    assert record["profile"] == {"gender": "male", "team": "ops"}
    assert "registered_at" in record and "last_seen_at" in record
    assert "created_at" not in record and "meta" not in record


def test_a_string_profile_is_kept_as_a_note(tmp_path):
    """An LLM will occasionally send prose where an object is expected."""
    db = FaceDB(db_dir=str(tmp_path))
    record = db.add("小王", [_vector(1)], profile="爱穿蓝色外套")
    assert record["profile"] == {"note": "爱穿蓝色外套"}


def test_version_1_database_migrates_profile_to_name(tmp_path):
    """The field split must not orphan a database written by the older build."""
    db = FaceDB(db_dir=str(tmp_path))
    db.add("ignored", [_vector(1)])
    state = json.loads((tmp_path / "persons.json").read_text())
    state["version"] = 1
    state["persons"][0] = {
        "id": "p-1",
        "profile": "Alice from ops",          # v1: free text label
        "meta": {"badge": "A7"},              # v1: structured bag
        "named": True,
        "created_at": 111.0,
        "updated_at": 222.0,
        "last_seen_at": 333.0,
    }
    (tmp_path / "persons.json").write_text(json.dumps(state))

    migrated = FaceDB(db_dir=str(tmp_path)).get_person("p-1")
    assert migrated["name"] == "Alice from ops"
    assert migrated["profile"] == {"badge": "A7"}
    assert migrated["registered_at"] == 111.0
    assert migrated["last_seen_at"] == 333.0
