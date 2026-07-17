from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .ledger import SegmentLedger


log = logging.getLogger(__name__)


class COSObjectConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class COSCredentials:
    secret_id: str
    secret_key: str
    token: str = ""


def load_cos_credentials(profile: str, *, secret_root: str | Path = "/run/secrets/phanthymotus/cos") -> COSCredentials:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", profile) or profile in {".", ".."}:
        raise ValueError("credential_profile contains unsupported path characters")
    secret_path = Path(secret_root) / f"{profile}.json"
    payload: dict[str, Any] = {}
    if secret_path.exists():
        payload = json.loads(secret_path.read_text(encoding="utf-8"))
    secret_id = str(payload.get("secret_id") or payload.get("SecretId") or os.environ.get("COS_SECRET_ID", ""))
    secret_key = str(payload.get("secret_key") or payload.get("SecretKey") or os.environ.get("COS_SECRET_KEY", ""))
    token = str(payload.get("token") or payload.get("Token") or os.environ.get("COS_SESSION_TOKEN", ""))
    if not secret_id or not secret_key:
        raise ValueError(f"COS credential profile {profile!r} is unavailable")
    return COSCredentials(secret_id=secret_id, secret_key=secret_key, token=token)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header(response: dict[str, Any], name: str) -> str:
    expected = name.lower()
    for key, value in response.items():
        if str(key).lower() == expected:
            return str(value)
    return ""


class TencentCOSBackend:
    def __init__(
        self,
        *,
        region: str,
        credentials: COSCredentials,
        multipart_threshold_mb: int,
        upload_concurrency: int,
    ) -> None:
        from qcloud_cos import CosConfig, CosS3Client

        config_kwargs = dict(
            Region=region,
            SecretId=credentials.secret_id,
            SecretKey=credentials.secret_key,
            Scheme="https",
        )
        if credentials.token:
            config_kwargs["Token"] = credentials.token
        config = CosConfig(**config_kwargs)
        self._client = CosS3Client(config)
        self.multipart_threshold = int(multipart_threshold_mb) * 1024 * 1024
        self.upload_concurrency = int(upload_concurrency)

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        status_getter = getattr(exc, "get_status_code", None)
        if callable(status_getter):
            try:
                return int(status_getter()) == 404
            except (TypeError, ValueError):
                pass
        return "404" in str(exc) and "not" in str(exc).lower()

    def head(self, *, bucket: str, key: str) -> dict[str, Any] | None:
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise
        size_text = _header(response, "content-length")
        sha256 = _header(response, "x-cos-meta-sha256") or _header(response, "sha256")
        return {"size": int(size_text or 0), "sha256": sha256}

    def upload_file(self, *, bucket: str, key: str, path: Path, sha256: str) -> None:
        metadata = {"x-cos-meta-sha256": sha256}
        if path.stat().st_size >= self.multipart_threshold:
            self._client.upload_file(
                Bucket=bucket,
                Key=key,
                LocalFilePath=str(path),
                PartSize=10,
                MAXThread=self.upload_concurrency,
                EnableMD5=False,
                Metadata=metadata,
            )
            return
        with path.open("rb") as handle:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=handle,
                EnableMD5=False,
                Metadata=metadata,
            )

    def put_bytes(self, *, bucket: str, key: str, body: bytes, sha256: str) -> None:
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            EnableMD5=False,
            Metadata={"x-cos-meta-sha256": sha256},
        )

    def cleanup_stale_multipart(self, *, bucket: str, prefix: str, stale_hours: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(stale_hours))
        key_marker = ""
        upload_id_marker = ""
        aborted = 0
        while True:
            response = self._client.list_multipart_uploads(
                Bucket=bucket,
                Prefix=prefix,
                KeyMarker=key_marker,
                UploadIdMarker=upload_id_marker,
                MaxUploads=1000,
            )
            uploads = response.get("Upload") or []
            if isinstance(uploads, dict):
                uploads = [uploads]
            for upload in uploads:
                initiated_text = str(upload.get("Initiated", "")).replace("Z", "+00:00")
                try:
                    initiated = datetime.fromisoformat(initiated_text)
                except ValueError:
                    continue
                if initiated.tzinfo is None:
                    initiated = initiated.replace(tzinfo=timezone.utc)
                if initiated > cutoff:
                    continue
                self._client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=str(upload["Key"]),
                    UploadId=str(upload["UploadId"]),
                )
                aborted += 1
            if str(response.get("IsTruncated", "false")).lower() != "true":
                break
            key_marker = str(response.get("NextKeyMarker", ""))
            upload_id_marker = str(response.get("NextUploadIdMarker", ""))
            if not key_marker and not upload_id_marker:
                break
        return aborted


class COSUploadCoordinator:
    def __init__(
        self,
        *,
        ledger: SegmentLedger,
        data_root: str | Path,
        card_id: str,
        config: dict[str, Any],
        backend: Any | None = None,
    ) -> None:
        self.ledger = ledger
        self.data_root = Path(data_root)
        self.card_id = card_id
        self.config = dict(config)
        self.bucket = str(config.get("cos_bucket", ""))
        self.prefix = str(config.get("cos_prefix", "inspection-data")).strip("/")
        self.device_id = str(config.get("device_id", "unknown"))
        if not self.bucket:
            raise ValueError("cos_bucket is required")
        if backend is None:
            credentials = load_cos_credentials(str(config.get("credential_profile", "default")))
            backend = TencentCOSBackend(
                region=str(config.get("cos_region", "ap-beijing")),
                credentials=credentials,
                multipart_threshold_mb=int(config.get("multipart_threshold_mb", 64)),
                upload_concurrency=int(config.get("upload_concurrency", 2)),
            )
        self.backend = backend
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.uploaded_verified = 0
        self.failed = 0
        self.conflicts = 0
        self.multipart_aborted = 0
        self.multipart_cleanup_error = ""
        self.last_error = ""
        self.last_retry_delay_seconds = 0.0

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for index in range(max(1, int(self.config.get("upload_concurrency", 2)))):
            thread = threading.Thread(target=self._worker, args=(index,), daemon=True, name=f"{self.card_id}-cos-{index}")
            thread.start()
            self._threads.append(thread)

    def stop(self, *, timeout: float = 10) -> bool:
        self._stop.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        return not self._threads

    def _object_key(self, path: Path) -> str:
        relative = path.relative_to(self.data_root)
        if len(relative.parts) < 5:
            raise ValueError(f"unexpected inspection path: {path}")
        card_id, instance_id, date, hour = relative.parts[:4]
        year, month, day = date.split("-", 2)
        return "/".join(filter(None, (
            self.prefix,
            self.device_id,
            card_id,
            instance_id,
            year,
            month,
            day,
            hour,
            path.name,
        )))

    @staticmethod
    def _verify_head(info: dict[str, Any], *, size: int, sha256: str, key: str) -> None:
        if int(info.get("size", -1)) != int(size) or str(info.get("sha256", "")) != sha256:
            raise COSObjectConflict(
                f"COS object conflict for {key}: remote size/hash does not match local immutable segment"
            )

    def _ensure_file(self, path: Path) -> str:
        size = path.stat().st_size
        sha256 = _sha256(path)
        key = self._object_key(path)
        existing = self.backend.head(bucket=self.bucket, key=key)
        if existing is not None:
            self._verify_head(existing, size=size, sha256=sha256, key=key)
            return key
        self.backend.upload_file(bucket=self.bucket, key=key, path=path, sha256=sha256)
        uploaded = self.backend.head(bucket=self.bucket, key=key)
        if uploaded is None:
            raise RuntimeError(f"COS HEAD returned not found after upload: {key}")
        self._verify_head(uploaded, size=size, sha256=sha256, key=key)
        return key

    def run_once(self) -> bool:
        record = self.ledger.claim_next_upload(card_id=self.card_id)
        if record is None:
            return False
        segment_id = str(record["segment_id"])
        try:
            media_path = Path(record["local_path"])
            metadata_path = Path(record["metadata_path"])
            if not media_path.exists() or not metadata_path.exists():
                raise COSObjectConflict(f"local segment pair is incomplete: {media_path}, {metadata_path}")
            media_key = self._ensure_file(media_path)
            self._ensure_file(metadata_path)
            self.ledger.mark_upload_verified(segment_id, object_key=media_key)
            self.uploaded_verified += 1
            self.last_error = ""
        except COSObjectConflict as exc:
            self.conflicts += 1
            self.last_error = str(exc)
            self.ledger.mark_conflict(segment_id, error=self.last_error)
        except Exception as exc:
            self.failed += 1
            self.last_error = str(exc)
            self.last_retry_delay_seconds = self._retry_delay(exc, attempts=int(record["attempts"]))
            self.ledger.mark_upload_retry(
                segment_id,
                error=self.last_error,
                retry_after_seconds=self.last_retry_delay_seconds,
            )
        return True

    def _retry_delay(self, exc: Exception, *, attempts: int) -> float:
        retry_max = max(1, int(self.config.get("retry_max_seconds", 300)))
        status_getter = getattr(exc, "get_status_code", None)
        if callable(status_getter):
            try:
                if int(status_getter()) in {401, 403}:
                    return float(retry_max)
            except (TypeError, ValueError):
                pass
        return float(min(retry_max, 2 ** min(max(0, attempts - 1), 16)))

    def _cleanup_stale_multipart(self) -> None:
        cleanup = getattr(self.backend, "cleanup_stale_multipart", None)
        if not callable(cleanup):
            return
        try:
            self.multipart_aborted = int(cleanup(
                bucket=self.bucket,
                prefix="/".join(filter(None, (self.prefix, self.device_id, self.card_id))),
                stale_hours=int(self.config.get("multipart_stale_hours", 24)),
            ))
            self.multipart_cleanup_error = ""
        except Exception as exc:
            self.multipart_cleanup_error = str(exc)
            log.warning("stale multipart cleanup skipped: %s", exc)

    def _worker(self, index: int) -> None:
        if index == 0:
            self._cleanup_stale_multipart()
        retry_max = max(1, int(self.config.get("retry_max_seconds", 300)))
        idle_wait = 1.0
        while not self._stop.is_set():
            worked = self.run_once()
            if worked and not self.last_error:
                idle_wait = 0.1
            elif worked:
                idle_wait = min(float(retry_max), max(1.0, idle_wait * 2))
            else:
                idle_wait = min(5.0, max(0.5, idle_wait * 1.5))
            self._stop.wait(idle_wait)

    def test_upload(self) -> dict[str, Any]:
        started = time.monotonic()
        body = json.dumps({"ok": True, "device_id": self.device_id, "card_id": self.card_id}).encode("utf-8")
        sha256 = hashlib.sha256(body).hexdigest()
        key = "/".join(filter(None, (
            self.prefix,
            self.device_id,
            self.card_id,
            "_health",
            f"{time.time_ns()}.json",
        )))
        self.backend.put_bytes(bucket=self.bucket, key=key, body=body, sha256=sha256)
        info = self.backend.head(bucket=self.bucket, key=key)
        if info is None:
            raise RuntimeError("COS health object not found after upload")
        self._verify_head(info, size=len(body), sha256=sha256, key=key)
        return {
            "state": "verified",
            "verified": True,
            "object_key": key,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "uploaded_verified_worker": self.uploaded_verified,
            "upload_failed": self.failed,
            "upload_conflicts": self.conflicts,
            "upload_last_error": self.last_error,
            "upload_retry_delay_seconds": self.last_retry_delay_seconds,
            "multipart_aborted": self.multipart_aborted,
            "multipart_cleanup_error": self.multipart_cleanup_error,
        }
