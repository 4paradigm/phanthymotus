"""
tests/test_face_roster_migration.py — migrating a roster to another model.

The background pass reads the source model's stored crops, runs them through the
model currently loaded, and writes into *that* model's database. It is neither an
overwrite nor an extra column: each model owns its own store, so the source is left
byte-for-byte intact.

There used to be a `reembed_from` action driving the same work synchronously. It is
gone, and these tests were rewritten against the background path — nobody needed to
trigger a migration by hand, and the cost of offering it was that agent-core splits
every `x-action-params` entry into an LLM-callable function unconditionally, so a
maintenance operation that blocks for tens of seconds ended up in the robot's own
toolbox. `auto_migrate_roster` is now the only entry point.

The embedding step is faked here. What these tests are for is everything around it:
that identity is carried across (ids, names, unnamed-ness, enrolment time), that the
source is not touched, that samples with no crop are *reported* rather than silently
dropped, that the pass does not gate service, and that it stops cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (installs stubs, sets sys.path)

import plugins.face as face_plugin  # noqa: E402
import plugins.identity_db as identity_db  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_tmp_dirs(monkeypatch):
    monkeypatch.setattr(identity_db, "require_models_subpath",
                        lambda path, root=None: path)


def _vec(seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).standard_normal(
        face_plugin.EMBEDDING_DIM).astype(np.float32)
    return raw / np.linalg.norm(raw)


class _FakeAnalyzer:
    """Stands in for the ORT worker: one deterministic vector per crop."""

    def __init__(self):
        self.calls = []

    def embed_aligned(self, crops):
        self.calls.append(list(crops))
        out = []
        for crop in crops:
            if crop == b"CORRUPT":
                out.append(None)
                continue
            out.append(_vec(len(crop) + sum(crop[:4])))
        return out


class _Engine:
    def __init__(self, db, analyzer):
        self.db = db
        self.analyzer = analyzer


def _source_db(tmp_path):
    """A buffalo_sc roster: two named, one unnamed, one with no crop at all."""
    db = identity_db.IdentityDB(
        str(tmp_path / "face_db"), dim=face_plugin.EMBEDDING_DIM,
        model="buffalo_sc", label="src", artifact_suffix=".jpg")
    db.add("alice", [_vec(1), _vec(2)], artifacts=[b"ALICE-1", b"ALICE-2"],
           registered_at=1000.0)
    db.add("bob", [_vec(3)], artifacts=[b"BOB-1"], registered_at=2000.0)
    db.add("", [_vec(4)], named=False, artifacts=[b"ANON-1"])
    db.add("carol-no-crop", [_vec(5)], artifacts=[None], registered_at=3000.0)
    return db


def _plugin(tmp_path, target_model="scrfd25g_edgeface_s"):
    plugin = face_plugin.FaceRecognitionPlugin.__new__(
        face_plugin.FaceRecognitionPlugin)
    import threading
    plugin._state_lock = threading.RLock()
    plugin._migration = None          # __init__ is bypassed by __new__ here
    plugin._load_generation = 0
    plugin._plugin_cfg = {"model": target_model,
                          "db_dir": face_plugin.DEFAULT_DB_DIR}
    target = identity_db.IdentityDB(
        str(tmp_path / "face_db-target"), dim=face_plugin.EMBEDDING_DIM,
        model=target_model, label="dst", artifact_suffix=".jpg")
    analyzer = _FakeAnalyzer()
    engine = _Engine(target, analyzer)
    plugin._require_engine = lambda: engine
    return plugin, engine



def _run_pass(plugin, engine, generation=1, timeout=5.0, **cfg_extra):
    """Run the background migration to completion and return its progress dict.

    The pass is deliberately asynchronous, so a test that wants its *result* has to
    wait for it. Tests about the non-blocking property assert on the state right
    after the call instead — see test_the_background_pass_does_not_gate_service.
    """
    import time
    plugin._load_generation = generation
    cfg = {"model": "scrfd25g_edgeface_s", "db_dir": face_plugin.DEFAULT_DB_DIR}
    cfg.update(cfg_extra)
    plugin._maybe_start_roster_migration(generation, engine, cfg,
                                         cfg["model"])
    if plugin._migration is None:
        return None
    deadline = time.time() + timeout
    while time.time() < deadline and plugin._migration["state"] == "running":
        time.sleep(0.01)
    return plugin._migration


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = _source_db(tmp_path)
    plugin, engine = _plugin(tmp_path)
    # The plugin resolves each model's directory by name; point both at tmp rather
    # than /models. Model-aware on purpose: a stub returning one path for every
    # model would collide source with target, which the guard correctly refuses —
    # and the background pass would then find nothing to do for the wrong reason.
    monkeypatch.setattr(
        face_plugin, "face_db_dir_for",
        lambda model, db_dir: str(tmp_path / ("face_db" if model == "buffalo_sc"
                                              else "face_db-target")))
    return plugin, engine, source


# ── the background startup pass ──────────────────────────────────────────────
#
# Migration is part of coming up, not a precondition for it: the engine is already
# published and recognising when this starts, and simply recognises more people as
# it proceeds. These tests are about that property and about stopping cleanly.


def test_the_background_pass_fills_the_roster_without_being_waited_on(setup):
    import time
    plugin, engine, source = setup
    plugin._load_generation = 7
    cfg = {"model": "scrfd25g_edgeface_s", "db_dir": face_plugin.DEFAULT_DB_DIR,
           "auto_migrate_roster": True}

    plugin._maybe_start_roster_migration(7, engine, cfg, "scrfd25g_edgeface_s")
    # the call returns immediately — it must not have done the work inline
    assert plugin._migration["state"] in ("running", "done")

    deadline = time.time() + 5
    while time.time() < deadline and plugin._migration["state"] == "running":
        time.sleep(0.02)
    assert plugin._migration["state"] == "done", plugin._migration
    assert plugin._migration["persons_done"] == 3
    assert plugin._migration["samples_done"] == 4
    assert {person["id"] for person
            in engine.db.list_persons(limit=0)["persons"]} == {"p-1", "p-2", "p-3"}


def test_the_background_pass_is_cancelled_when_the_engine_is_replaced(setup):
    import time
    plugin, engine, source = setup
    plugin._load_generation = 1
    cfg = {"model": "scrfd25g_edgeface_s", "db_dir": face_plugin.DEFAULT_DB_DIR}

    # a config change or stop bumps the generation; the pass must notice and stop
    # rather than keep writing into a database the plugin has moved on from
    slow = engine.analyzer.embed_aligned

    def stall(crops):
        plugin._load_generation = 2        # simulate the rebuild mid-pass
        return slow(crops)
    engine.analyzer.embed_aligned = stall

    plugin._maybe_start_roster_migration(1, engine, cfg, "scrfd25g_edgeface_s")
    deadline = time.time() + 5
    while time.time() < deadline and plugin._migration["state"] == "running":
        time.sleep(0.02)
    assert plugin._migration["state"] == "cancelled", plugin._migration


def test_auto_migrate_roster_false_starts_nothing(setup):
    plugin, engine, source = setup
    plugin._load_generation = 1
    plugin._maybe_start_roster_migration(
        1, engine, {"model": "scrfd25g_edgeface_s",
                    "db_dir": face_plugin.DEFAULT_DB_DIR,
                    "auto_migrate_roster": False}, "scrfd25g_edgeface_s")
    assert plugin._migration is None
    assert engine.db.stats()["persons"] == 0


def test_no_source_with_crops_means_no_pass(tmp_path, monkeypatch):
    """A fresh robot has nothing to migrate and must not report a migration."""
    plugin, engine = _plugin(tmp_path)
    plugin._load_generation = 1
    monkeypatch.setattr(face_plugin, "face_db_dir_for",
                        lambda model, db_dir: str(tmp_path / "absent"))
    plugin._maybe_start_roster_migration(
        1, engine, {"model": "scrfd25g_edgeface_s",
                    "db_dir": face_plugin.DEFAULT_DB_DIR}, "scrfd25g_edgeface_s")
    assert plugin._migration is None


def test_a_rerun_of_the_background_pass_is_also_idempotent(setup):
    """Restarting perception must not re-migrate what is already across.

    This is the switching question: every (crop, model) pair is computed once and
    kept in that model's own database, so coming back to a model costs nothing.
    """
    import time
    plugin, engine, source = setup
    cfg = {"model": "scrfd25g_edgeface_s", "db_dir": face_plugin.DEFAULT_DB_DIR}
    for generation in (1, 2, 3):
        plugin._load_generation = generation
        plugin._maybe_start_roster_migration(generation, engine, cfg,
                                             "scrfd25g_edgeface_s")
        deadline = time.time() + 5
        while time.time() < deadline and plugin._migration["state"] == "running":
            time.sleep(0.02)
        assert plugin._migration["state"] == "done"
    assert engine.db.stats()["samples"] == 4        # not 12
    payloads = [open(item["path"], "rb").read()
                for item in engine.db.sample_artifacts()]
    assert len(payloads) == len(set(payloads))
    # the model ran exactly once per crop across all three passes
    embedded = [crop for call in engine.analyzer.calls for crop in call]
    assert len(embedded) == 4, embedded


# ── what the migration itself must guarantee ────────────────────────────────
#
# These were written against the old synchronous action. The entry point changed;
# the guarantees did not, so they are re-expressed here rather than dropped.


def test_identity_is_carried_across(setup):
    plugin, engine, source = setup
    progress = _run_pass(plugin, engine)
    assert progress["state"] == "done"
    assert progress["persons_done"] == 3          # alice, bob, the anon
    assert progress["samples_done"] == 4          # 2 + 1 + 1

    people = {person["id"]: person
              for person in engine.db.list_persons(limit=0)["persons"]}
    assert set(people) == {"p-1", "p-2", "p-3"}   # ids preserved for visits.jsonl
    assert people["p-1"]["name"] == "alice"
    assert people["p-2"]["name"] == "bob"
    # unnamed stays unnamed rather than arriving as a blank-named person
    assert people["p-3"]["named"] is False
    # enrolment time survives, so nobody looks newly met against the visit log
    assert people["p-1"]["registered_at"] == 1000.0
    assert people["p-2"]["registered_at"] == 2000.0


def test_the_embeddings_are_recomputed_not_copied(setup):
    """Every crop goes through the *target* model; nothing is carried over raw."""
    plugin, engine, source = setup
    _run_pass(plugin, engine)
    embedded = [crop for call in engine.analyzer.calls for crop in call]
    assert sorted(embedded) == [b"ALICE-1", b"ALICE-2", b"ANON-1", b"BOB-1"]


def test_a_person_without_a_crop_is_left_behind_and_counted(setup):
    """p-4 has no crop, so there is nothing to re-embed — and it must show."""
    plugin, engine, source = setup
    progress = _run_pass(plugin, engine)
    assert "p-4" not in {person["id"] for person
                         in engine.db.list_persons(limit=0)["persons"]}
    # the honest denominator: the pass only ever claims the people it could do
    assert progress["persons_total"] == 3
    assert progress["source_crops"] == 4
    assert source.stats()["samples"] == 5


def test_the_source_database_is_not_touched(setup):
    plugin, engine, source = setup
    before = source.stats()
    before_people = source.list_persons(limit=0)["persons"]
    _run_pass(plugin, engine)
    reopened = identity_db.IdentityDB(
        source.path, dim=face_plugin.EMBEDDING_DIM, model="buffalo_sc",
        label="src", artifact_suffix=".jpg")
    assert reopened.stats()["samples"] == before["samples"]
    assert reopened.list_persons(limit=0)["persons"] == before_people
    assert len(reopened.sample_artifacts()) == 4


def test_a_rerun_does_not_duplicate_samples_or_evict_fresh_captures(setup):
    """The defect this guards, measured before it was fixed.

    A second pass re-embedded the same crops and appended them, taking one person
    from 5 samples to 8 with three duplicates. Once that fills
    max_samples_per_person, the oldest-first trim starts evicting the genuinely new
    captures made *since* the migration — the migration eating real data.
    """
    plugin, engine, source = setup
    _run_pass(plugin, engine, generation=1)
    after_first = engine.db.stats()["samples"]

    # a real capture enrolled under the new model after migrating
    engine.db.add_samples("p-1", [_vec(777)], artifacts=[b"FRESH-CAPTURE"])
    assert engine.db.stats()["samples"] == after_first + 1

    for generation in (2, 3, 4):
        _run_pass(plugin, engine, generation=generation)

    assert engine.db.stats()["samples"] == after_first + 1, "a rerun duplicated samples"
    payloads = [open(item["path"], "rb").read()
                for item in engine.db.sample_artifacts()]
    assert b"FRESH-CAPTURE" in payloads, "a rerun evicted a real capture"
    assert len(payloads) == len(set(payloads)), "duplicate crops stored"


def test_a_crop_that_fails_to_reembed_does_not_abandon_the_pass(tmp_path, monkeypatch):
    """One corrupt file must not stop a 500-person migration."""
    source = identity_db.IdentityDB(
        str(tmp_path / "face_db"), dim=face_plugin.EMBEDDING_DIM,
        model="buffalo_sc", label="src", artifact_suffix=".jpg")
    source.add("dave", [_vec(9)], artifacts=[b"CORRUPT"])
    source.add("erin", [_vec(10)], artifacts=[b"FINE"])
    plugin, engine = _plugin(tmp_path)
    monkeypatch.setattr(
        face_plugin, "face_db_dir_for",
        lambda model, db_dir: str(tmp_path / ("face_db" if model == "buffalo_sc"
                                              else "face_db-target")))
    progress = _run_pass(plugin, engine)
    assert progress["state"] == "done"
    assert progress["persons_done"] == 1            # erin made it, dave did not
    assert {person["name"] for person
            in engine.db.list_persons(limit=0)["persons"]} == {"erin"}


def test_the_background_pass_does_not_gate_service(setup):
    """The property the whole design exists for: serving does not wait on it.

    `_maybe_start_roster_migration` must return before the work is done, so the
    engine is already answering while the roster fills in behind it.
    """
    import threading
    plugin, engine, source = setup
    plugin._load_generation = 1
    released = threading.Event()
    original = engine.analyzer.embed_aligned

    def blocked(crops):
        released.wait(5)
        return original(crops)
    engine.analyzer.embed_aligned = blocked

    plugin._maybe_start_roster_migration(
        1, engine, {"model": "scrfd25g_edgeface_s",
                    "db_dir": face_plugin.DEFAULT_DB_DIR},
        "scrfd25g_edgeface_s")
    # returned while the embedding is still stuck, and said so
    assert plugin._migration["state"] == "running"
    assert plugin._migration["persons_done"] == 0
    released.set()
