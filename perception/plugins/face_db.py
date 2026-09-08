#!/usr/bin/env python3
"""
plugins/face_db.py — 人脸身份持久化存储。

Holds every enrolled identity (both named and unnamed) plus their face embeddings,
on disk under a `/models` subdirectory — the only host-mounted writable path the
perception container has (see `perception/deploy/service.yml`).

Three files, and **`persons.json` is the commit point**:

    persons.json        metadata + row ownership + the name of the embeddings
                        file this metadata belongs to
    embeddings-<n>.npy  float32 [rows, 512], L2-normalised, one row per sample
    visits.jsonl        append-only sighting log, one line per *visit*

A person record separates the structured fields from the free-form ones:

    id              p-N (all persons use the same ID format)
    name            str  — structured. A non-blank name is what makes an entry
                    "named"; publishing it is how the agent addresses someone.
                    When blank, the person is recognized but not yet identified.
    profile         object — non-structured: gender, appearance, notes, tags.
                    Whatever the operator wants to carry, published alongside
                    the name so the agent has it in context on every sighting.
    named           bool — whether this person has been given a name
    registered_at   when the identity was created
    last_seen_at    most recent sighting

`registered_at` and `last_seen_at` are deliberately **not** in the per-frame
payload: they change on every frame (or never), and the answer to "when was this
person around" belongs in the visit log, which can express it properly.

**The visit log records one line per visit, not per frame.** A visit is a
contiguous presence: it opens on the first sighting, absorbs every later one,
and is closed and appended once the person has not been seen for
`visit_gap_s` — **10 minutes by default**. At the default 1 detection/second a
per-frame log would be 86 400 writes a day per person onto eMMC for no extra
information, and a short gap would fragment one afternoon in the office into
dozens of rows every time somebody turned their head. A visit carries
`first_seen`, `last_seen` and a sighting count, which is what "who was here at
3pm" actually needs. The cost of the long gap is latency: a visit is only
queryable as a *closed* record 10 minutes after the person leaves — which is
why `list_visits` also reports the still-open ones, flagged `open: true`.

The embeddings file is written under a *fresh* name first and `persons.json`
replaced last, so the two can never be observed out of step: until the new
`persons.json` lands, the old pair is still the committed state, and a crash at
any point leaves a consistent database. Superseded `.npy` files are removed
only after the commit succeeds. Writing `embeddings.npy` in place instead —
two `os.replace` calls, either order — has a window where the row count and the
owner list disagree, which silently misattributes every identity after the
missing row.

Matching is a single `matrix @ embedding`: both sides are L2-normalised, so the
dot product *is* the cosine similarity and no `sklearn` is needed (it is not in
the image). Rows are per *sample*, not per person, and a person's score is the
best of its samples — a mean-vector centroid blurs the pose variation that
several enrolment photos exist to capture.

Thread safety: `register_*` runs on arbitrary `ThreadingHTTPServer` threads
(see `perception/README.md` § "Plugin Concurrency") while the recognition
worker calls `match()` at frame rate. Every public method takes an `RLock`, and
every *write* additionally takes an `fcntl` lock on a file in `db_dir`, the same
belt-and-braces `utils/model_downloader.py` uses for a shared `/models`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Iterable

import numpy as np

from utils.log_sampling import escape_log_text
from utils.model_downloader import require_models_subpath

try:  # Linux only; the perception images are Linux, dev hosts may not be.
    import fcntl
except ImportError:  # pragma: no cover - Windows/macOS dev hosts
    fcntl = None

log = logging.getLogger(__name__)

DEFAULT_DB_DIR = "/models/face_db"
EMBEDDING_DIM = 512

DEFAULT_UNKNOWN_CAPACITY = 500
DEFAULT_MAX_SAMPLES_PER_PERSON = 8
DEFAULT_VISIT_GAP_S = 600.0   # 10 minutes
DEFAULT_VISIT_LOG_MAX = 20000
DEFAULT_VISIT_CHECKPOINT_S = 60.0

_PERSONS_FILE = "persons.json"
_VISITS_FILE = "visits.jsonl"
_OPEN_VISITS_FILE = "visits-open.json"
_LOCK_FILE = ".face_db.lock"
_EMBEDDINGS_PREFIX = "embeddings-"
_EMBEDDINGS_SUFFIX = ".npy"

# All person IDs use the same format now: p-N
# Whether a person is "named" is determined by the `named` field in their record,
# not by the ID prefix. This keeps IDs stable when a person is recognized and named.
PERSON_PREFIX = "p-"


class FaceDBError(RuntimeError):
    """The database on disk exists but cannot be used as-is."""


def is_unknown_id(person_id: str) -> bool:
    """Deprecated: all IDs are now p-N format. Check record['named'] instead."""
    # Kept for backward compatibility during migration, always returns False
    return False


def _now() -> float:
    return time.time()


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Return a float32 unit vector, so a dot product is a cosine."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size != EMBEDDING_DIM:
        raise ValueError(
            f"embedding must have {EMBEDDING_DIM} dims, got {array.size}"
        )
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        raise ValueError("embedding has zero norm")
    return (array / norm).astype(np.float32)


def _clean_profile(profile: Any) -> dict:
    """Accept only a JSON-serialisable object for the free-form profile.

    `profile` is the *non-structured* half of a person record — gender,
    appearance, notes, tags, whatever the operator wants to carry. It comes
    from an MCP caller and is written straight to disk, so a value that cannot
    be serialised would fail at save time, i.e. *after* the in-memory state had
    already changed. Validate on the way in instead.
    """
    if profile is None:
        return {}
    if isinstance(profile, str):
        # Tolerated because it is the shape the field had before the split into
        # name + profile, and because an LLM will occasionally send prose here.
        text = profile.strip()
        return {"note": text} if text else {}
    if not isinstance(profile, dict):
        raise ValueError("profile must be a JSON object")
    try:
        json.dumps(profile, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"profile is not JSON-serialisable: {error}") from error
    return dict(profile)


def parse_time(value: Any) -> float | None:
    """Coerce an epoch number or an ISO-8601 string to epoch seconds.

    Visit queries are the one place a caller naturally thinks in wall-clock
    ("who was here after 15:00"), and an LLM will send a string. Accept both
    rather than making every caller convert.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    from datetime import datetime
    normalised = text.replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(normalised)
    except ValueError as error:
        raise ValueError(
            f"cannot parse {value!r} as epoch seconds or ISO-8601"
        ) from error
    if moment.tzinfo is None:
        # Naive input means local time, which is what an operator reading a
        # wall clock next to the robot means.
        return moment.timestamp()
    return moment.timestamp()


class FaceDB:
    """Persistent store of enrolled identities and their face embeddings."""

    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        unknown_capacity: int = DEFAULT_UNKNOWN_CAPACITY,
        max_samples_per_person: int = DEFAULT_MAX_SAMPLES_PER_PERSON,
        visit_gap_s: float = DEFAULT_VISIT_GAP_S,
        visit_log_max: int = DEFAULT_VISIT_LOG_MAX,
        visit_checkpoint_s: float = DEFAULT_VISIT_CHECKPOINT_S,
    ):
        # db_dir arrives over MCP config and this process runs as root in the
        # container; the same validation model_downloader applies to model_dir.
        self._dir = require_models_subpath(db_dir)
        self._unknown_capacity = max(0, int(unknown_capacity))
        self._max_samples = max(1, int(max_samples_per_person))

        self._lock = threading.RLock()
        self._persons: dict[str, dict] = {}
        self._row_owners: list[str] = []
        self._matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        # Single counter for all person IDs (p-N format)
        self._next_ids = {"named": 1}
        self._generation = 0

        self._visit_gap = max(0.0, float(visit_gap_s))
        self._visit_log_max = max(0, int(visit_log_max))
        self._visit_checkpoint_s = max(0.0, float(visit_checkpoint_s))
        self._visits_dirty = False
        self._last_checkpoint_at = 0.0
        # person_id -> the visit currently in progress. Held in memory until the
        # person has been absent for visit_gap_s, then appended to visits.jsonl.
        self._open_visits: dict[str, dict] = {}

        self._load()
        self._recover_open_visits()

    # ── persistence ───────────────────────────────────────────────────────

    @property
    def path(self) -> str:
        return self._dir

    def _persons_path(self) -> str:
        return os.path.join(self._dir, _PERSONS_FILE)

    def _embeddings_name(self, generation: int) -> str:
        return f"{_EMBEDDINGS_PREFIX}{generation}{_EMBEDDINGS_SUFFIX}"

    def _load(self) -> None:
        persons_path = self._persons_path()
        if not os.path.exists(persons_path):
            log.info("[face_db] no database at %s; starting empty", self._dir)
            return
        try:
            with open(persons_path, encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, ValueError) as error:
            raise FaceDBError(
                f"{persons_path} is unreadable: {escape_log_text(error)}"
            ) from error

        embeddings_file = state.get("embeddings_file")
        rows = list(state.get("rows") or [])
        matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        if embeddings_file:
            matrix_path = os.path.join(self._dir, os.path.basename(embeddings_file))
            try:
                matrix = np.load(matrix_path).astype(np.float32, copy=False)
            except (OSError, ValueError) as error:
                # The commit point named a file that is gone or corrupt. Do not
                # silently continue with no embeddings: recognition would report
                # every enrolled person as unknown while the dashboard showed a
                # populated database. Surfaced as the plugin's error state.
                raise FaceDBError(
                    f"embeddings file {matrix_path} named by {_PERSONS_FILE} is "
                    f"unusable: {escape_log_text(error)}"
                ) from error
        if matrix.ndim != 2 or matrix.shape[0] != len(rows) or (
            matrix.size and matrix.shape[1] != EMBEDDING_DIM
        ):
            raise FaceDBError(
                f"database at {self._dir} is inconsistent: {len(rows)} row owners "
                f"but embeddings shape {matrix.shape}"
            )

        persons: dict[str, dict] = {}
        for entry in state.get("persons") or []:
            person_id = str(entry.get("id") or "").strip()
            if not person_id:
                continue
            # Migration from version 1, where `profile` was the free-text label
            # and `meta` the structured bag. The two swapped roles: `name` is now
            # the structured label and `profile` the free-form bag.
            name = entry.get("name")
            raw_profile = entry.get("profile")
            if name is None and isinstance(raw_profile, str):
                name = raw_profile
                raw_profile = entry.get("meta")
            elif raw_profile is None:
                raw_profile = entry.get("meta")
            persons[person_id] = {
                "id": person_id,
                "name": str(name or ""),
                "profile": raw_profile if isinstance(raw_profile, dict) else {},
                "named": bool(entry.get("named", not is_unknown_id(person_id))),
                "registered_at": float(
                    entry.get("registered_at") or entry.get("created_at") or _now()
                ),
                "updated_at": float(entry.get("updated_at") or _now()),
                "last_seen_at": float(entry.get("last_seen_at") or 0.0),
            }

        # A row owned by a person that is not in the table would make
        # `_person_scores` attribute a sample to nobody; drop those rows rather
        # than carry a matrix the metadata cannot explain.
        keep = [index for index, owner in enumerate(rows) if owner in persons]
        if len(keep) != len(rows):
            log.warning("[face_db] dropping %d orphaned embedding row(s)",
                        len(rows) - len(keep))
            rows = [rows[index] for index in keep]
            matrix = matrix[keep] if matrix.size else matrix

        next_ids = state.get("next_ids") or {}
        self._persons = persons
        self._row_owners = [str(owner) for owner in rows]
        self._matrix = matrix if matrix.size else np.zeros(
            (0, EMBEDDING_DIM), dtype=np.float32
        )
        # Migrate from old dual-counter format to single counter
        # Take the max of both counters to ensure no ID collisions
        old_named = max(1, int(next_ids.get("named", 1)))
        old_unknown = max(1, int(next_ids.get("unknown", 1)))
        self._next_ids = {
            "named": max(old_named, old_unknown),
        }
        self._generation = max(0, int(state.get("generation") or 0))
        log.info("[face_db] loaded %d person(s), %d sample(s) from %s",
                 len(self._persons), len(self._row_owners), self._dir)

    def _save_locked(self) -> None:
        """Persist the current state. Caller holds `self._lock`.

        The embeddings file is written under a new generation name and
        `persons.json` — which names it — is replaced last. See the module
        docstring for why the order matters.
        """
        os.makedirs(self._dir, exist_ok=True)
        lock_path = os.path.join(self._dir, _LOCK_FILE)
        with open(lock_path, "a+b") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                generation = self._generation + 1
                matrix_name = self._embeddings_name(generation)
                matrix_path = os.path.join(self._dir, matrix_name)

                # np.save() appends ".npy" to any path that lacks it, which
                # would silently write beside a ".tmp" staging name and leave
                # the file we then rename empty. Passing an open handle makes
                # it write exactly where we asked.
                tmp_matrix = os.path.join(self._dir, f".{matrix_name}.tmp")
                try:
                    with open(tmp_matrix, "wb") as handle:
                        np.save(handle, self._matrix, allow_pickle=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(tmp_matrix, matrix_path)
                finally:
                    if os.path.exists(tmp_matrix):
                        os.unlink(tmp_matrix)

                state = {
                    "version": 2,
                    "generation": generation,
                    "embeddings_file": matrix_name,
                    "next_ids": dict(self._next_ids),
                    "rows": list(self._row_owners),
                    "persons": [
                        dict(person) for person in self._persons.values()
                    ],
                }
                persons_path = self._persons_path()
                tmp_persons = f"{persons_path}.tmp"
                with open(tmp_persons, "w", encoding="utf-8") as handle:
                    json.dump(state, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_persons, persons_path)   # ← commit point
                self._generation = generation

                # Only now is the previous embeddings file unreachable.
                for name in os.listdir(self._dir):
                    if (
                        name.startswith(_EMBEDDINGS_PREFIX)
                        and name.endswith(_EMBEDDINGS_SUFFIX)
                        and name != matrix_name
                    ):
                        try:
                            os.unlink(os.path.join(self._dir, name))
                        except OSError:  # noqa: PERF203 - best-effort cleanup
                            pass
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # ── ids ───────────────────────────────────────────────────────────────

    def _allocate_id_locked(self, named: bool) -> str:
        """Return a fresh id. Counters only ever increase, so an id retired by
        `forget()` is never handed to a different person later — a stale
        reference in a conversation history or a peer's notes must not resolve
        to somebody else.

        All IDs use the same p-N format now, regardless of named status.
        The `named` parameter still determines which counter to use for backward
        compatibility, but both produce p-N formatted IDs.
        """
        # Use a single counter for all person IDs
        key = "named"  # Always use the named counter
        while True:
            number = self._next_ids[key]
            self._next_ids[key] = number + 1
            candidate = f"{PERSON_PREFIX}{number}"
            if candidate not in self._persons:
                return candidate

    # ── matching ──────────────────────────────────────────────────────────

    def _person_scores_locked(self, embedding: np.ndarray) -> dict[str, float]:
        if not self._row_owners:
            return {}
        scores = self._matrix @ embedding
        best: dict[str, float] = {}
        for owner, score in zip(self._row_owners, scores):
            value = float(score)
            if value > best.get(owner, -2.0):
                best[owner] = value
        return best

    def match(
        self, embedding: np.ndarray, threshold: float
    ) -> tuple[str | None, float]:
        """Best matching person id above `threshold`, plus the score.

        Returns `(None, best_score)` when nothing clears the threshold, so the
        caller can log how close it came — the number an operator needs when
        tuning `match_threshold`. `best_score` is `-1.0` on an empty database.
        """
        vector = _normalize(embedding)
        with self._lock:
            scores = self._person_scores_locked(vector)
            if not scores:
                return None, -1.0
            person_id = max(scores, key=lambda key: scores[key])
            best = scores[person_id]
            if best < float(threshold):
                return None, best
            return person_id, best

    # ── writes ────────────────────────────────────────────────────────────

    def _add_rows_locked(self, person_id: str, vectors: list[np.ndarray]) -> None:
        """Append samples for one person, capped at `max_samples_per_person`.

        Oldest-first eviction: rows are appended in enrolment order, so keeping
        the tail keeps the most recent captures — the ones most likely to match
        how the person looks now.
        """
        if not vectors:
            return
        stacked = np.stack(vectors).astype(np.float32, copy=False)
        self._matrix = (
            np.concatenate([self._matrix, stacked])
            if self._matrix.size
            else stacked
        )
        self._row_owners.extend([person_id] * len(vectors))

        owned = [i for i, owner in enumerate(self._row_owners) if owner == person_id]
        if len(owned) > self._max_samples:
            drop = set(owned[: len(owned) - self._max_samples])
            keep = [i for i in range(len(self._row_owners)) if i not in drop]
            self._row_owners = [self._row_owners[i] for i in keep]
            self._matrix = self._matrix[keep]

    def _samples_locked(self, person_id: str) -> int:
        """Sample count, always derived from the row owners rather than stored.

        A cached count is one more thing that can drift out of step with the
        matrix after an eviction; this is an O(rows) scan over a few thousand
        strings and runs only on the reporting paths, not per frame.
        """
        return sum(1 for owner in self._row_owners if owner == person_id)

    def _record_locked(self, person_id: str) -> dict:
        person = self._persons[person_id]
        return {
            "id": person["id"],
            "name": person["name"],
            "profile": dict(person["profile"]),
            "named": person["named"],
            "samples": self._samples_locked(person_id),
            "registered_at": person["registered_at"],
            "updated_at": person["updated_at"],
            "last_seen_at": person["last_seen_at"],
        }

    def add(
        self,
        name: str,
        embeddings: Iterable[np.ndarray],
        named: bool = True,
        profile: Any = None,
        person_id: str | None = None,
    ) -> dict:
        """Enrol a new identity and return its record."""
        vectors = [_normalize(item) for item in embeddings]
        if not vectors:
            raise ValueError("at least one embedding is required")
        clean_profile = _clean_profile(profile)
        with self._lock:
            if person_id:
                if person_id in self._persons:
                    raise ValueError(f"person {person_id!r} already exists")
                new_id = str(person_id)
            else:
                new_id = self._allocate_id_locked(named)
            timestamp = _now()
            self._persons[new_id] = {
                "id": new_id,
                "name": str(name or ""),
                "profile": clean_profile,
                "named": bool(named),
                "registered_at": timestamp,
                "updated_at": timestamp,
                "last_seen_at": timestamp,
            }
            self._add_rows_locked(new_id, vectors)
            if not named:
                self._evict_unknowns_locked()
            record = self._record_locked(new_id) if new_id in self._persons else None
            self._save_locked()
        if record is None:
            # Capacity 0: the entry was evicted by the same call that made it.
            raise FaceDBError("unknown_capacity is 0; cannot enrol unknown faces")
        log.info("[face_db] enrolled %s (named=%s, samples=%d): %s",
                 new_id, named, record["samples"],
                 escape_log_text(record["name"]))
        return record

    def add_samples(self, person_id: str, embeddings: Iterable[np.ndarray]) -> dict:
        """Attach more face samples to an existing person."""
        vectors = [_normalize(item) for item in embeddings]
        if not vectors:
            raise ValueError("at least one embedding is required")
        with self._lock:
            if person_id not in self._persons:
                raise KeyError(person_id)
            self._add_rows_locked(person_id, vectors)
            self._persons[person_id]["updated_at"] = _now()
            record = self._record_locked(person_id)
            self._save_locked()
        return record

    def enroll_unknown(self, embedding: np.ndarray) -> dict:
        """Create an anonymous `unknown-N` entry for a face nobody has named."""
        return self.add("", [embedding], named=False)

    def promote(
        self, person_id: str, name: str, profile: Any = None, merge: bool = True
    ) -> dict:
        """Turn an `unknown-N` entry into a named person, **keeping its id**.

        The id is deliberately preserved rather than reissued as `p-N`: it is
        what already appeared on the activity stream and in the agent's
        conversation history for every earlier sighting, and rewriting it would
        orphan all of that.
        """
        return self.update_person(person_id, name=name, profile=profile, merge=merge)

    def update_person(
        self,
        person_id: str,
        name: str | None = None,
        profile: Any = None,
        profile_delete: Iterable[str] | None = None,
        merge: bool = True,
    ) -> dict:
        """Edit a person's name and/or free-form profile."""
        clean_profile = _clean_profile(profile) if profile is not None else None
        delete_keys = [str(key) for key in (profile_delete or [])]
        with self._lock:
            person = self._persons.get(person_id)
            if person is None:
                raise KeyError(person_id)
            changed = False
            if name is not None:
                person["name"] = str(name)
                # A name is what makes an entry a *named* person; setting one on
                # an unknown promotes it in place, keeping the id.
                if str(name).strip() and not person["named"]:
                    person["named"] = True
                    log.info("[face_db] %s promoted to a named person", person_id)
                changed = True
            if clean_profile is not None:
                person["profile"] = (
                    {**person["profile"], **clean_profile} if merge else clean_profile
                )
                changed = True
            for key in delete_keys:
                if key in person["profile"]:
                    del person["profile"][key]
                    changed = True
            if changed:
                person["updated_at"] = _now()
            record = self._record_locked(person_id)
            if changed:
                self._save_locked()
        return record

    def touch(self, person_id: str, when: float | None = None) -> None:
        """Record a sighting. Drives `unknown_capacity` eviction order.

        Cheap and deliberately not persisted on its own: this runs at frame
        rate, and rewriting the database per frame would hammer the flash. The
        value reaches disk with the next real write.
        """
        with self._lock:
            person = self._persons.get(person_id)
            if person is not None:
                person["last_seen_at"] = float(when if when is not None else _now())

    def flush(self) -> None:
        """Persist pending in-memory changes (e.g. `touch` timestamps)."""
        with self._lock:
            self._save_locked()

    def _drop_locked(self, ids) -> list[str]:
        """Remove these people and their embedding rows. Caller holds the lock.

        Does **not** save: the caller decides when to commit, so a batch delete
        is one rewrite rather than one per person. That matters more than it
        sounds — every save writes a fresh `embeddings-<n>.npy` and rewrites
        `persons.json`, so clearing 50 strangers one at a time meant 50 full
        rewrites of the entire database onto eMMC.

        Takes an *ordered* iterable and returns the removals in that order.
        Iterating a set here instead made `forget_many`'s result order
        hash-dependent, so a caller could not line its request up against the
        response.

        Also drops any visit in progress for those people, which would
        otherwise be appended later under an id that no longer exists.
        """
        removed = [
            pid for pid in dict.fromkeys(ids) if pid in self._persons
        ]
        if not removed:
            return []
        for pid in removed:
            del self._persons[pid]
            self._open_visits.pop(pid, None)
        doomed = set(removed)
        keep = [i for i, owner in enumerate(self._row_owners) if owner not in doomed]
        if len(keep) != len(self._row_owners):
            self._row_owners = [self._row_owners[i] for i in keep]
            self._matrix = (
                self._matrix[keep] if keep
                else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            )
        return removed

    def forget(self, person_id: str) -> bool:
        """Delete one person, their samples and their embedding rows."""
        with self._lock:
            if not self._drop_locked([person_id]):
                return False
            self._save_locked()
        log.info("[face_db] forgot %s", person_id)
        return True

    def forget_many(self, person_ids) -> dict:
        """Delete several people in one commit.

        Returns `{"forgotten": [...], "missing": [...]}` rather than a count:
        given a list of ids the caller needs to know *which* were not there, and
        a partial result is the normal case — a stale id, a typo, or an entry
        another operator already removed.
        """
        wanted = [str(pid) for pid in person_ids if str(pid).strip()]
        unique = list(dict.fromkeys(wanted))          # de-dup, preserve order
        with self._lock:
            # `unique` is already ordered, so the response lines up with the
            # request; passing a set here made the result order hash-dependent.
            removed = self._drop_locked(unique)
            if removed:
                self._save_locked()
        removed_set = set(removed)
        missing = [pid for pid in unique if pid not in removed_set]
        if removed:
            log.info("[face_db] forgot %d person(s) in one commit", len(removed))
        return {"forgotten": removed, "missing": missing}

    def forget_unknowns(self) -> int:
        """Delete every anonymous entry. Named people are untouched."""
        with self._lock:
            # A list, not a set: roster order is insertion order, which keeps
            # the return value deterministic.
            targets = [
                pid for pid, person in self._persons.items()
                if not person["named"]
            ]
            removed = self._drop_locked(targets)
            if removed:
                self._save_locked()
        return len(removed)

    # ── capacity ──────────────────────────────────────────────────────────

    def _evict_unknowns_locked(self) -> int:
        """Trim anonymous entries to `unknown_capacity`, oldest sighting first.

        Named people are never candidates: the cap exists to bound automatic
        enrolment, not to expire people somebody deliberately registered.
        """
        unknowns = [
            person for person in self._persons.values() if not person["named"]
        ]
        excess = len(unknowns) - self._unknown_capacity
        if excess <= 0:
            return 0
        unknowns.sort(key=lambda person: (person["last_seen_at"], person["registered_at"]))
        doomed = {person["id"] for person in unknowns[:excess]}
        for person_id in doomed:
            del self._persons[person_id]
        keep = [
            index for index, owner in enumerate(self._row_owners)
            if owner not in doomed
        ]
        self._row_owners = [self._row_owners[i] for i in keep]
        self._matrix = (
            self._matrix[keep] if keep
            else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        )
        log.info("[face_db] evicted %d unknown entr(ies) over capacity %d",
                 len(doomed), self._unknown_capacity)
        return len(doomed)

    def set_unknown_capacity(self, capacity: int) -> int:
        """Change the ceiling and apply it now. Returns how many were evicted."""
        with self._lock:
            self._unknown_capacity = max(0, int(capacity))
            evicted = self._evict_unknowns_locked()
            if evicted:
                self._save_locked()
        return evicted

    @property
    def unknown_capacity(self) -> int:
        return self._unknown_capacity

    # ── reads ─────────────────────────────────────────────────────────────

    def get_person(self, person_id: str) -> dict:
        with self._lock:
            if person_id not in self._persons:
                raise KeyError(person_id)
            return self._record_locked(person_id)

    def list_persons(
        self,
        named: str = "all",
        query: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """Page through the roster. Never returns embeddings."""
        wanted = str(named or "all").lower()
        needle = str(query or "").strip().lower()
        with self._lock:
            records = [
                self._record_locked(person_id) for person_id in self._persons
            ]
        if wanted == "named":
            records = [r for r in records if r["named"]]
        elif wanted == "unknown":
            records = [r for r in records if not r["named"]]
        if needle:
            def matches(record: dict) -> bool:
                haystack = " ".join([
                    record["id"], record["name"],
                    json.dumps(record["profile"], ensure_ascii=False),
                ]).lower()
                return needle in haystack
            records = [r for r in records if matches(r)]
        records.sort(key=lambda record: (not record["named"], -record["updated_at"]))
        total = len(records)
        start = max(0, int(offset))
        end = start + max(0, int(limit)) if limit else total
        return {
            "total": total,
            "offset": start,
            "limit": int(limit),
            "persons": records[start:end],
        }

    # ── visit log ─────────────────────────────────────────────────────────

    def _visits_path(self) -> str:
        return os.path.join(self._dir, _VISITS_FILE)

    def record_sighting(self, person_id: str, when: float, topic: str = "") -> None:
        """Note that `person_id` was seen at `when`.

        Called at detection rate, so it must stay cheap and must not write: it
        updates `last_seen_at` in memory and either opens a visit or extends the
        open one. A visit only reaches disk when `close_stale_visits` decides
        the person has left.
        """
        with self._lock:
            person = self._persons.get(person_id)
            if person is not None:
                person["last_seen_at"] = float(when)
            visit = self._open_visits.get(person_id)
            if visit is None:
                self._open_visits[person_id] = {
                    "person_id": person_id,
                    "name": person["name"] if person else "",
                    "first_seen": float(when),
                    "last_seen": float(when),
                    "sightings": 1,
                    "topic": topic,
                }
            else:
                visit["last_seen"] = max(visit["last_seen"], float(when))
                visit["sightings"] += 1
                if person is not None and person["name"]:
                    # A visit that began while the person was still unknown gets
                    # their name once they are enrolled mid-visit.
                    visit["name"] = person["name"]
            self._visits_dirty = True

    def _open_visits_path(self) -> str:
        return os.path.join(self._dir, _OPEN_VISITS_FILE)

    def checkpoint_open_visits(self, force: bool = False) -> bool:
        """Persist visits still in progress, so a power cut cannot lose them.

        With a 10-minute `visit_gap_s`, somebody present all afternoon is one
        visit held in memory for hours — and a robot that loses power would
        lose the whole record, not just the tail. This writes the open visits to
        a small separate file at most every `visit_checkpoint_s` (default 60 s),
        which bounds the loss to a minute of `last_seen`/`sightings` rather than
        the entire visit.

        Deliberately *not* the append-only log: an open visit is still changing,
        so it must be overwritten rather than appended, and mixing the two would
        mean rewriting history on every checkpoint.
        """
        moment = _now()
        with self._lock:
            if not force:
                if not self._visits_dirty:
                    return False
                if moment - self._last_checkpoint_at < self._visit_checkpoint_s:
                    return False
            snapshot = [dict(visit) for visit in self._open_visits.values()]
            self._last_checkpoint_at = moment
            self._visits_dirty = False
        path = self._open_visits_path()
        try:
            if not snapshot:
                if os.path.exists(path):
                    os.unlink(path)
                return True
            os.makedirs(self._dir, exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump({"saved_at": moment, "visits": snapshot}, handle,
                          ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            return True
        except OSError:
            log.warning("[face_db] could not checkpoint open visits",
                        exc_info=True)
            return False

    def _recover_open_visits(self) -> None:
        """Reload visits that were in progress when the process last stopped.

        A visit whose subject has since been absent longer than `visit_gap_s` is
        closed and appended straight away — the person left while we were down.
        One that is still fresh is resumed, so a restart does not split an
        ongoing presence into two records.
        """
        path = self._open_visits_path()
        try:
            with open(path, encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, ValueError):
            return
        moment = _now()
        resumed, closed = 0, []
        for visit in state.get("visits") or []:
            person_id = str(visit.get("person_id") or "")
            if not person_id:
                continue
            last_seen = float(visit.get("last_seen") or 0.0)
            if moment - last_seen >= self._visit_gap:
                closed.append({**visit, "recovered": True})
            else:
                self._open_visits[person_id] = dict(visit)
                resumed += 1
        if closed:
            with self._lock:
                self._append_visits_locked(closed)
        if resumed or closed:
            log.info("[face_db] recovered open visits: %d resumed, %d closed",
                     resumed, len(closed))
        try:
            if not self._open_visits and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass

    def close_stale_visits(self, now: float | None = None, force: bool = False) -> int:
        """Append every visit whose subject has been absent for `visit_gap_s`.

        `force` closes all open visits regardless — used when an instance stops,
        so a visit in progress is not lost.
        """
        moment = _now() if now is None else float(now)
        with self._lock:
            due = [
                person_id for person_id, visit in self._open_visits.items()
                if force or moment - visit["last_seen"] >= self._visit_gap
            ]
            if not due:
                return 0
            closing = [self._open_visits.pop(person_id) for person_id in due]
            self._append_visits_locked(closing)
            self._visits_dirty = True
        # The checkpoint must follow the append, so a crash in between replays a
        # closed visit rather than dropping it: `list_visits` can show one twice
        # in the worst case, which is recoverable; losing it is not.
        self.checkpoint_open_visits(force=True)
        return len(closing)

    def _append_visits_locked(self, visits: list[dict]) -> None:
        """Append closed visits to visits.jsonl. Caller holds `self._lock`.

        JSON Lines, not the persons.json blob: appends are O(1) and cannot
        rewrite (or corrupt) the identity table, and a time-range query is a
        forward scan. Its own file also means the commit-point ordering of
        persons.json/embeddings is untouched by sighting traffic.
        """
        if not visits or self._visit_log_max == 0:
            return
        os.makedirs(self._dir, exist_ok=True)
        lock_path = os.path.join(self._dir, _LOCK_FILE)
        try:
            with open(lock_path, "a+b") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    with open(self._visits_path(), "a", encoding="utf-8") as handle:
                        for visit in visits:
                            handle.write(
                                json.dumps(visit, ensure_ascii=False) + "\n"
                            )
                        handle.flush()
                        os.fsync(handle.fileno())
                    self._trim_visits_locked()
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            # A full or read-only /models must not take recognition down; the
            # identities still work, only the history is lost.
            log.warning("[face_db] could not append to the visit log",
                        exc_info=True)

    def _trim_visits_locked(self) -> None:
        """Keep the newest `visit_log_max` lines. Rewrites only when over."""
        path = self._visits_path()
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return
        if len(lines) <= self._visit_log_max:
            return
        keep = lines[-self._visit_log_max:]
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.writelines(keep)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        log.info("[face_db] visit log trimmed to the newest %d entries",
                 self._visit_log_max)

    def list_visits(
        self,
        person_id: str = "",
        since: Any = None,
        until: Any = None,
        limit: int = 100,
        offset: int = 0,
        include_open: bool = True,
    ) -> dict:
        """Visits overlapping [since, until], newest first.

        Overlap rather than containment: someone who arrived at 14:50 and left
        at 15:10 *was* there at 15:00, and a query for 15:00-15:05 has to say so.
        """
        start = parse_time(since)
        end = parse_time(until)
        records: list[dict] = []
        try:
            with open(self._visits_path(), encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        # One torn line (a crash mid-append) must not make the
                        # whole history unreadable.
                        continue
        except OSError:
            records = []

        if include_open:
            with self._lock:
                records.extend(
                    {**visit, "open": True} for visit in self._open_visits.values()
                )

        def overlaps(visit: dict) -> bool:
            first = float(visit.get("first_seen") or 0.0)
            last = float(visit.get("last_seen") or first)
            if start is not None and last < start:
                return False
            if end is not None and first > end:
                return False
            return True

        if person_id:
            records = [r for r in records if r.get("person_id") == person_id]
        records = [r for r in records if overlaps(r)]
        records.sort(key=lambda r: float(r.get("last_seen") or 0.0), reverse=True)

        total = len(records)
        begin = max(0, int(offset))
        stop = begin + max(0, int(limit)) if limit else total
        page = records[begin:stop]
        # Names change; resolve them at read time so old lines are not stale.
        with self._lock:
            for visit in page:
                person = self._persons.get(visit.get("person_id", ""))
                if person is not None:
                    visit["name"] = person["name"]
        return {
            "total": total,
            "offset": begin,
            "limit": int(limit),
            "since": start,
            "until": end,
            "visits": page,
        }

    def stats(self) -> dict:
        with self._lock:
            named = sum(1 for person in self._persons.values() if person["named"])
            return {
                "persons": len(self._persons),
                "named": named,
                "unknown": len(self._persons) - named,
                "samples": len(self._row_owners),
                "unknown_capacity": self._unknown_capacity,
                "open_visits": len(self._open_visits),
                "db_dir": self._dir,
            }


__all__ = [
    "DEFAULT_DB_DIR",
    "DEFAULT_VISIT_CHECKPOINT_S",
    "DEFAULT_VISIT_GAP_S",
    "DEFAULT_VISIT_LOG_MAX",
    "parse_time",
    "DEFAULT_MAX_SAMPLES_PER_PERSON",
    "DEFAULT_UNKNOWN_CAPACITY",
    "EMBEDDING_DIM",
    "FaceDB",
    "FaceDBError",
    "is_unknown_id",
]
