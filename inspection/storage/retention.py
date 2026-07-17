from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from .atomic_writer import fsync_directory
from .ledger import SegmentLedger
from .models import SegmentState


log = logging.getLogger(__name__)


class RetentionSweeper:
    def __init__(
        self,
        *,
        ledger: SegmentLedger,
        card_id: str,
        interval_seconds: float = 30,
        on_critical: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.ledger = ledger
        self.card_id = card_id
        self.interval_seconds = float(interval_seconds)
        self.on_critical = on_critical
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.purged = 0
        self.last_error = ""
        self.critical_instances: dict[str, dict[str, int]] = {}

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

    def _purge_record(self, record: dict) -> int:
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
        self.ledger.mark_purged_local(segment_id)
        self.purged += 1
        return removed_bytes

    def sweep_once(self, *, now_ns: int | None = None) -> dict[str, int]:
        now_ns = time.time_ns() if now_ns is None else int(now_ns)
        stats = {"purged": 0, "critical": 0}
        self.critical_instances = {}
        for saved in self.ledger.list_instance_states(card_id=self.card_id):
            instance_id = str(saved["instance_id"])
            config = dict(saved.get("config") or {})
            retention_hours = float(config.get("local_retention_hours", 24))
            max_bytes = int(float(config.get("local_max_gb", 4)) * 1024 * 1024 * 1024)
            cutoff_ns = now_ns - int(retention_hours * 3600 * 1_000_000_000)
            local_bytes = self.ledger.summary(card_id=self.card_id, instance_id=instance_id)["local_bytes"]
            candidates = self.ledger.list_cleanup_candidates(card_id=self.card_id, instance_id=instance_id)
            for record in candidates:
                already_eligible = record["state"] == SegmentState.RETENTION_ELIGIBLE.value
                expired = int(record["created_at_ns"]) <= cutoff_ns
                over_budget = local_bytes > max_bytes
                if not (already_eligible or expired or over_budget):
                    continue
                try:
                    local_bytes = max(0, local_bytes - self._purge_record(record))
                    stats["purged"] += 1
                    self.last_error = ""
                except Exception as exc:
                    self.last_error = str(exc)
                    log.exception("failed to purge local segment %s", record["segment_id"])
            if local_bytes > max_bytes:
                self.critical_instances[instance_id] = {"local_bytes": local_bytes, "max_bytes": max_bytes}
                stats["critical"] += 1
                if self.on_critical is not None:
                    self.on_critical(instance_id, local_bytes, max_bytes)
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
            "retention_last_error": self.last_error,
            "critical_instances": dict(self.critical_instances),
        }
