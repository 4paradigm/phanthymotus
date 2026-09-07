#!/usr/bin/env python3
"""
plugins/face_db.py — 人脸身份持久化存储。

Holds every enrolled identity (named people and auto-assigned `unknown-N`
entries) plus their face embeddings, on disk under a `/models` subdirectory —
the only host-mounted writable path the perception container has (see
`perception/deploy/service.yml`).

Two files, and **`persons.json` is the commit point**:

    persons.json        metadata + row ownership + the name of the embeddings
                        file this metadata belongs to
    embeddings-<n>.npy  float32 [rows, 512], L2-normalised, one row per sample

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

_PERSONS_FILE = "persons.json"
_LOCK_FILE = ".face_db.lock"
_EMBEDDINGS_PREFIX = "embeddings-"
_EMBEDDINGS_SUFFIX = ".npy"

UNKNOWN_PREFIX = "unknown-"
NAMED_PREFIX = "p-"


class FaceDBError(RuntimeError):
    """The database on disk exists but cannot be used as-is."""


def is_unknown_id(person_id: str) -> bool:
    return str(person_id).startswith(UNKNOWN_PREFIX)


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


def _clean_meta(meta: Any) -> dict:
    """Accept only a JSON-serialisable object for the free-form meta field.

    Meta comes from an MCP caller and is written straight to disk; a value that
    cannot be serialised would fail at save time, i.e. *after* the in-memory
    state had already changed. Validate on the way in instead.
    """
    if meta is None:
        return {}
    if not isinstance(meta, dict):
        raise ValueError("meta must be a JSON object")
    try:
        json.dumps(meta, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"meta is not JSON-serialisable: {error}") from error
    return dict(meta)


class FaceDB:
    """Persistent store of enrolled identities and their face embeddings."""

    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        unknown_capacity: int = DEFAULT_UNKNOWN_CAPACITY,
        max_samples_per_person: int = DEFAULT_MAX_SAMPLES_PER_PERSON,
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
        self._next_ids = {"named": 1, "unknown": 1}
        self._generation = 0

        self._load()

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
            persons[person_id] = {
                "id": person_id,
                "profile": str(entry.get("profile") or ""),
                "named": bool(entry.get("named", not is_unknown_id(person_id))),
                "meta": entry.get("meta") if isinstance(entry.get("meta"), dict) else {},
                "created_at": float(entry.get("created_at") or _now()),
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
        self._next_ids = {
            "named": max(1, int(next_ids.get("named", 1))),
            "unknown": max(1, int(next_ids.get("unknown", 1))),
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
                    "version": 1,
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
        to somebody else."""
        key = "named" if named else "unknown"
        prefix = NAMED_PREFIX if named else UNKNOWN_PREFIX
        while True:
            number = self._next_ids[key]
            self._next_ids[key] = number + 1
            candidate = f"{prefix}{number}"
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
            "profile": person["profile"],
            "named": person["named"],
            "meta": dict(person["meta"]),
            "samples": self._samples_locked(person_id),
            "created_at": person["created_at"],
            "updated_at": person["updated_at"],
            "last_seen_at": person["last_seen_at"],
        }

    def add(
        self,
        profile: str,
        embeddings: Iterable[np.ndarray],
        named: bool = True,
        meta: Any = None,
        person_id: str | None = None,
    ) -> dict:
        """Enrol a new identity and return its record."""
        vectors = [_normalize(item) for item in embeddings]
        if not vectors:
            raise ValueError("at least one embedding is required")
        clean_meta = _clean_meta(meta)
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
                "profile": str(profile or ""),
                "named": bool(named),
                "meta": clean_meta,
                "created_at": timestamp,
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
                 escape_log_text(record["profile"]))
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
        self, person_id: str, profile: str, meta: Any = None, merge: bool = True
    ) -> dict:
        """Turn an `unknown-N` entry into a named person, **keeping its id**.

        The id is deliberately preserved rather than reissued as `p-N`: it is
        what already appeared on the activity stream and in the agent's
        conversation history for every earlier sighting, and rewriting it would
        orphan all of that.
        """
        return self.update_person(person_id, profile=profile, meta=meta, merge=merge)

    def update_person(
        self,
        person_id: str,
        profile: str | None = None,
        meta: Any = None,
        meta_delete: Iterable[str] | None = None,
        merge: bool = True,
    ) -> dict:
        """Edit a person's profile and/or free-form meta."""
        clean_meta = _clean_meta(meta) if meta is not None else None
        delete_keys = [str(key) for key in (meta_delete or [])]
        with self._lock:
            person = self._persons.get(person_id)
            if person is None:
                raise KeyError(person_id)
            changed = False
            if profile is not None:
                person["profile"] = str(profile)
                # A profile is what makes an entry a *named* person; setting one
                # on an unknown promotes it in place.
                if str(profile).strip() and not person["named"]:
                    person["named"] = True
                    log.info("[face_db] %s promoted to a named person", person_id)
                changed = True
            if clean_meta is not None:
                person["meta"] = (
                    {**person["meta"], **clean_meta} if merge else clean_meta
                )
                changed = True
            for key in delete_keys:
                if key in person["meta"]:
                    del person["meta"][key]
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

    def forget(self, person_id: str) -> bool:
        """Delete one person, their samples and their embedding rows."""
        with self._lock:
            if person_id not in self._persons:
                return False
            del self._persons[person_id]
            keep = [
                index for index, owner in enumerate(self._row_owners)
                if owner != person_id
            ]
            if len(keep) != len(self._row_owners):
                self._row_owners = [self._row_owners[i] for i in keep]
                self._matrix = (
                    self._matrix[keep] if keep
                    else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
                )
            self._save_locked()
        log.info("[face_db] forgot %s", person_id)
        return True

    def forget_unknowns(self) -> int:
        """Delete every anonymous entry. Named people are untouched."""
        with self._lock:
            targets = [
                person_id for person_id, person in self._persons.items()
                if not person["named"]
            ]
            for person_id in targets:
                del self._persons[person_id]
            if targets:
                dropped = set(targets)
                keep = [
                    index for index, owner in enumerate(self._row_owners)
                    if owner not in dropped
                ]
                self._row_owners = [self._row_owners[i] for i in keep]
                self._matrix = (
                    self._matrix[keep] if keep
                    else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
                )
                self._save_locked()
        return len(targets)

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
        unknowns.sort(key=lambda person: (person["last_seen_at"], person["created_at"]))
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
                    record["id"], record["profile"],
                    json.dumps(record["meta"], ensure_ascii=False),
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

    def stats(self) -> dict:
        with self._lock:
            named = sum(1 for person in self._persons.values() if person["named"])
            return {
                "persons": len(self._persons),
                "named": named,
                "unknown": len(self._persons) - named,
                "samples": len(self._row_owners),
                "unknown_capacity": self._unknown_capacity,
                "db_dir": self._dir,
            }


__all__ = [
    "DEFAULT_DB_DIR",
    "DEFAULT_MAX_SAMPLES_PER_PERSON",
    "DEFAULT_UNKNOWN_CAPACITY",
    "EMBEDDING_DIM",
    "FaceDB",
    "FaceDBError",
    "is_unknown_id",
]
