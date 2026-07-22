from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from .atomic_writer import fsync_directory
from .layout import card_storage_slug, instance_storage_slug, safe_component
from .ledger import SegmentLedger
from .models import SegmentState


log = logging.getLogger(__name__)


class RetentionSweeper:
    def __init__(
        self,
        *,
        ledger: SegmentLedger,
        card_id: str,
        data_root: str | Path | None = None,
        interval_seconds: float = 30,
        artifact_sweep_interval_seconds: float = 3600,
        on_critical: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.ledger = ledger
        self.card_id = card_id
        self.data_root = Path(data_root) if data_root is not None else None
        self.interval_seconds = float(interval_seconds)
        self.artifact_sweep_interval_ns = int(float(artifact_sweep_interval_seconds) * 1_000_000_000)
        self.on_critical = on_critical
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.purged = 0
        self.corrupt_files_purged = 0
        self.empty_dirs_pruned = 0
        self._next_artifact_sweep_ns = 0
        self.last_error = ""
        self.critical_instances: dict[str, dict[str, int]] = {}
        self.pressure_instances: dict[str, dict[str, float | int | str]] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{self.card_id}-retention")
        self._thread.start()

    def stop(self, *, timeout: float = 5) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        stopped = self._thread is None or not self._thread.is_alive()
        if stopped:
            self._thread = None
        return stopped

    def _prune_empty_parents(self, start: Path) -> int:
        if self.data_root is None:
            return 0
        base = self.data_root.resolve()
        current = start.resolve()
        try:
            current.relative_to(base)
        except ValueError:
            return 0
        pruned = 0
        while current != base:
            parent = current.parent
            try:
                current.rmdir()
            except (FileNotFoundError, OSError):
                break
            pruned += 1
            if parent.exists():
                fsync_directory(parent)
            current = parent
        return pruned

    def _purge_record(self, record: dict) -> tuple[int, int]:
        segment_id = str(record["segment_id"])
        if record["state"] != SegmentState.RETENTION_ELIGIBLE.value:
            self.ledger.mark_retention_eligible(segment_id)
        removed_bytes = 0
        parents: set[Path] = set()
        for field in ("local_path", "metadata_path"):
            path = Path(record[field])
            try:
                removed_bytes += path.stat().st_size
            except FileNotFoundError:
                pass
            path.unlink(missing_ok=True)
            parents.add(path.parent)
        for parent in parents:
            if parent.exists():
                fsync_directory(parent)
        pruned_dirs = sum(self._prune_empty_parents(parent) for parent in parents)
        self.ledger.mark_purged_local(segment_id)
        self.purged += 1
        self.empty_dirs_pruned += pruned_dirs
        return removed_bytes, pruned_dirs

    @staticmethod
    def _corrupt_group(corrupt_path: Path) -> tuple[Path, Path, Path]:
        marker = ".part.corrupt"
        text = str(corrupt_path)
        if not text.endswith(marker):
            raise ValueError(f"unexpected corrupt artifact path: {corrupt_path}")
        return (
            corrupt_path,
            Path(text + ".json"),
            Path(text[:-len(marker)] + ".open.json"),
        )

    def _instance_roots(self, instance_id: str, input_topic: str) -> list[Path]:
        if self.data_root is None:
            return []
        roots = [
            self.data_root / safe_component(self.card_id) / safe_component(instance_id),
            self.data_root / card_storage_slug(self.card_id) / instance_storage_slug(instance_id, input_topic),
        ]
        return list(dict.fromkeys(roots))

    def _purge_expired_corrupt(
        self,
        instance_id: str,
        input_topic: str,
        *,
        cutoff_ns: int,
    ) -> tuple[int, int]:
        if self.data_root is None:
            return 0, 0
        candidates: set[Path] = set()
        for instance_root in self._instance_roots(instance_id, input_topic):
            if not instance_root.exists():
                continue
            for path in instance_root.rglob("*.part.corrupt*"):
                text = str(path)
                marker_index = text.rfind(".part.corrupt")
                if marker_index >= 0:
                    candidates.add(Path(text[:marker_index + len(".part.corrupt")]))
        removed_files = 0
        pruned_dirs = 0
        for corrupt_path in sorted(candidates):
            related = [path for path in self._corrupt_group(corrupt_path) if path.exists()]
            if not related:
                continue
            try:
                newest_mtime_ns = max(path.stat().st_mtime_ns for path in related)
            except FileNotFoundError:
                continue
            if newest_mtime_ns > cutoff_ns:
                continue
            parents: set[Path] = set()
            for path in related:
                try:
                    path.unlink()
                    removed_files += 1
                    parents.add(path.parent)
                except FileNotFoundError:
                    pass
            for parent in parents:
                if parent.exists():
                    fsync_directory(parent)
            pruned_dirs += sum(self._prune_empty_parents(parent) for parent in parents)
        self.corrupt_files_purged += removed_files
        self.empty_dirs_pruned += pruned_dirs
        return removed_files, pruned_dirs

    def _prune_existing_empty_dirs(self) -> int:
        if self.data_root is None:
            return 0
        pruned = 0
        card_roots = {
            self.data_root / safe_component(self.card_id),
            self.data_root / card_storage_slug(self.card_id),
        }
        for card_root in card_roots:
            if not card_root.exists():
                continue
            for directory, _children, _files in os.walk(card_root, topdown=False):
                path = Path(directory)
                try:
                    path.rmdir()
                except (FileNotFoundError, OSError):
                    continue
                pruned += 1
                if path.parent.exists():
                    fsync_directory(path.parent)
        self.empty_dirs_pruned += pruned
        return pruned

    def _filesystem_ratio(self) -> float:
        if self.data_root is None:
            return 0.0
        try:
            usage = shutil.disk_usage(self.data_root)
        except OSError:
            return 0.0
        return 1.0 - (usage.free / usage.total) if usage.total else 0.0

    def sweep_once(self, *, now_ns: int | None = None) -> dict[str, int]:
        now_ns = time.time_ns() if now_ns is None else int(now_ns)
        stats = {"purged": 0, "critical": 0, "corrupt_files_purged": 0, "empty_dirs_pruned": 0}
        self.critical_instances = {}
        self.pressure_instances = {}
        saved_instances = self.ledger.list_instance_states(card_id=self.card_id)
        for saved in saved_instances:
            instance_id = str(saved["instance_id"])
            config = dict(saved.get("config") or {})
            storage_mode = str(config.get(
                "storage_mode",
                "local_and_cos" if bool(config.get("upload_enabled", True)) else "local_ring",
            ))
            retention_hours = float(config.get("local_retention_hours", 24))
            max_bytes = int(float(config.get("local_max_gb", 4)) * 1024 * 1024 * 1024)
            cutoff_ns = now_ns - int(retention_hours * 3600 * 1_000_000_000)
            local_bytes = self.ledger.summary(card_id=self.card_id, instance_id=instance_id)["local_bytes"]
            filesystem_ratio = self._filesystem_ratio()
            candidates = self.ledger.list_cleanup_candidates(
                card_id=self.card_id,
                instance_id=instance_id,
                include_unuploaded=storage_mode == "local_ring",
            )
            for record in candidates:
                already_eligible = record["state"] == SegmentState.RETENTION_ELIGIBLE.value
                expired = int(record["created_at_ns"]) <= cutoff_ns
                over_budget = local_bytes >= max_bytes * 0.85 or filesystem_ratio >= 0.85
                if not (already_eligible or expired or over_budget):
                    continue
                try:
                    removed_bytes, pruned_dirs = self._purge_record(record)
                    local_bytes = max(0, local_bytes - removed_bytes)
                    filesystem_ratio = self._filesystem_ratio()
                    stats["purged"] += 1
                    stats["empty_dirs_pruned"] += pruned_dirs
                    self.last_error = ""
                except Exception as exc:
                    self.last_error = str(exc)
                    log.exception("failed to purge local segment %s", record["segment_id"])
            local_ratio = local_bytes / max(1, max_bytes)
            filesystem_ratio = self._filesystem_ratio()
            effective_ratio = max(local_ratio, filesystem_ratio)
            if effective_ratio >= 0.95:
                pressure = "critical"
            elif effective_ratio >= 0.85:
                pressure = "high"
            elif effective_ratio >= 0.70:
                pressure = "warning"
            else:
                pressure = "normal"
            self.pressure_instances[instance_id] = {
                "storage_mode": storage_mode,
                "local_bytes": local_bytes,
                "max_bytes": max_bytes,
                "local_ratio": local_ratio,
                "filesystem_ratio": filesystem_ratio,
                "pressure": pressure,
            }
            if effective_ratio >= 0.95:
                self.critical_instances[instance_id] = {"local_bytes": local_bytes, "max_bytes": max_bytes}
                stats["critical"] += 1
                if self.on_critical is not None:
                    self.on_critical(instance_id, local_bytes, max_bytes)
        if self.data_root is not None and now_ns >= self._next_artifact_sweep_ns:
            for saved in saved_instances:
                instance_id = str(saved["instance_id"])
                config = dict(saved.get("config") or {})
                retention_hours = float(config.get("corrupt_retention_hours", 24))
                cutoff_ns = now_ns - int(retention_hours * 3600 * 1_000_000_000)
                removed_files, pruned_dirs = self._purge_expired_corrupt(
                    instance_id,
                    str(saved.get("input_topic", "")),
                    cutoff_ns=cutoff_ns,
                )
                stats["corrupt_files_purged"] += removed_files
                stats["empty_dirs_pruned"] += pruned_dirs
            stats["empty_dirs_pruned"] += self._prune_existing_empty_dirs()
            self._next_artifact_sweep_ns = now_ns + self.artifact_sweep_interval_ns
        return stats

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sweep_once()
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("retention sweep failed")
            self._stop.wait(self.interval_seconds)

    def stats(self) -> dict:
        return {
            "retention_purged": self.purged,
            "corrupt_files_purged": self.corrupt_files_purged,
            "empty_dirs_pruned": self.empty_dirs_pruned,
            "retention_last_error": self.last_error,
            "critical_instances": dict(self.critical_instances),
            "pressure_instances": dict(self.pressure_instances),
        }
