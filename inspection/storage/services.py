from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .cos_backend import COSUploadCoordinator, describe_cos_error
from .ledger import SegmentLedger
from .retention import RetentionSweeper


log = logging.getLogger(__name__)


class DurableServices:
    """Long-lived upload and retention workers shared by one Inspector tool."""

    def __init__(
        self,
        *,
        ledger: SegmentLedger,
        data_root: str | Path,
        card_id: str,
        on_critical: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.ledger = ledger
        self.data_root = Path(data_root)
        self.card_id = card_id
        self.uploader: COSUploadCoordinator | None = None
        self.retention = RetentionSweeper(
            ledger=ledger,
            card_id=card_id,
            data_root=self.data_root,
            on_critical=on_critical,
        )
        self.retention.start()
        self.last_error = ""
        self._config_signature = ""

    @staticmethod
    def _signature(config: dict[str, Any]) -> str:
        keys = (
            "storage_mode", "upload_enabled", "credential_profile", "cos_region", "cos_bucket",
            "cos_prefix", "device_id", "upload_concurrency",
            "multipart_threshold_mb", "multipart_stale_hours", "retry_max_seconds",
        )
        return repr(tuple((key, config.get(key)) for key in keys))

    def configure(
        self,
        config: dict[str, Any],
        *,
        backend: Any | None = None,
        validate_remote: bool = False,
    ) -> bool:
        upload_enabled = (
            str(config.get("storage_mode")) == "local_and_cos"
            if config.get("storage_mode") is not None
            else bool(config.get("upload_enabled", True))
        )
        signature = self._signature(config)
        if (
            backend is None
            and signature == self._config_signature
            and (self.uploader is not None or not upload_enabled)
        ):
            if validate_remote and self.uploader is not None:
                try:
                    self.uploader.test_upload()
                except Exception as exc:
                    self.last_error = str(exc)
                    return False
            self.last_error = ""
            return True
        if self.uploader is not None:
            if not self.uploader.stop():
                self.last_error = "previous COS uploader did not stop before reconfiguration"
                return False
            self.uploader = None
        self.last_error = ""
        if not upload_enabled:
            self._config_signature = signature
            return True
        try:
            candidate = COSUploadCoordinator(
                ledger=self.ledger,
                data_root=self.data_root,
                card_id=self.card_id,
                config=config,
                backend=backend,
            )
            if validate_remote:
                candidate.test_upload()
            candidate.start()
            self.uploader = candidate
            self._config_signature = signature
            return True
        except Exception as exc:
            self.last_error = (
                str(exc) if str(exc).startswith("COS ") else describe_cos_error(
                    exc,
                    bucket=str(config.get("cos_bucket", "")),
                    region=str(config.get("cos_region", "ap-beijing")),
                )
            )
            self.uploader = None
            log.warning("%s uploader is not ready: %s", self.card_id, exc)
            return False

    def restore_latest(self, *, overrides: dict[str, Any] | None = None) -> None:
        saved = self.ledger.list_instance_states(card_id=self.card_id)
        if saved:
            restored = dict(saved[-1].get("config") or {})
            restored.update(overrides or {})
            self.configure(restored)

    def test_upload(self) -> dict[str, Any]:
        if self.uploader is None:
            raise RuntimeError(self.last_error or "COS uploader is not configured")
        return self.uploader.test_upload()

    def stats(self) -> dict[str, Any]:
        stats = self.retention.stats()
        stats["upload_service_error"] = self.last_error
        if self.uploader is not None:
            stats.update(self.uploader.stats())
        return stats

    def stop(self) -> bool:
        uploader_stopped = True
        if self.uploader is not None:
            uploader_stopped = self.uploader.stop()
            if uploader_stopped:
                self.uploader = None
        retention_stopped = self.retention.stop()
        return uploader_stopped and retention_stopped
