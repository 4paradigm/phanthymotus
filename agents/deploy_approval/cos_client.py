"""COS (Tencent Cloud Object Storage) client for evidence uploads (final alignment).

Uses cos-python-sdk-v5 for authenticated uploads. Credentials are fully
separate from GitHub / Agent Core tokens.
"""

from __future__ import annotations

import asyncio
import re
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .config import Config

logger = logging.getLogger(__name__)

# Fixed root path for all COS evidence objects.
# Not configurable — aligns with the frozen COS layout contract.
COS_ROOT = "phanthymotus_pr/"
EVIDENCE_MAX_ARCHIVE_BYTES = 10 * 1024 * 1024

EVIDENCE_PRESIGNED_URL_TTL_SECONDS = 120



# Repo directory mapping: full repo -> short directory name.
_REPO_DIR_MAP = {
    "4paradigm/phanthymotus": "phanthymotus",
    "4paradigm/phanthymotus-driver": "phanthymotus-driver",
}


class CosError(Exception):
    pass


class CosClient:
    """Minimal COS client for uploading deploy artifacts.

    In production, this uses cos-python-sdk-v5. For testing, a fake
    implementation is provided.
    """

    def __init__(self, config: Config, _fake: bool = False):
        self.config = config
        self._fake = _fake
        self._uploads: list[dict] = []

    def _has_credentials(self) -> bool:
        values = (
            self.config.cos_region,
            self.config.cos_bucket,
            self.config.cos_secret_id,
            self.config.cos_secret_key,
        )
        return all(
            isinstance(value, str) and bool(value.strip())
            for value in values
        )

    def generate_evidence_download_url(
        self,
        object_key: str,
    ) -> str:
        """Generate a short-lived HTTPS presigned GET URL for an evidence object.

        Returns empty string on failure (fail-closed).  Never leaks secrets.
        """
        if not object_key or not isinstance(object_key, str):
            logger.warning(
                "COS_EVIDENCE_PRESIGN=INVALID_KEY",
            )
            return ""
        if object_key.startswith("/") or ".." in object_key or "\\" in object_key:
            logger.warning(
                "COS_EVIDENCE_PRESIGN=INVALID_KEY",
            )
            return ""
        if not object_key.startswith(COS_ROOT):
            logger.warning(
                "COS_EVIDENCE_PRESIGN=INVALID_KEY",
            )
            return ""
        if not self._has_credentials():
            logger.warning(
                "COS_EVIDENCE_PRESIGN=NO_CREDS",
            )
            return ""
        try:
            from qcloud_cos import CosConfig, CosS3Client  # type: ignore

            sdk_config = CosConfig(
                Region=self.config.cos_region,
                SecretId=self.config.cos_secret_id,
                SecretKey=self.config.cos_secret_key,
                Token=None,
            )
            client = CosS3Client(sdk_config)
            url = client.get_presigned_url(
                Method="GET",
                Bucket=self.config.cos_bucket,
                Key=object_key,
                Expired=EVIDENCE_PRESIGNED_URL_TTL_SECONDS,
            )
            if not url:
                logger.warning(
                    "COS_EVIDENCE_PRESIGN=EMPTY_URL",
                )
                return ""
            if not url.startswith("https://"):
                logger.warning(
                    "COS_EVIDENCE_PRESIGN=NOT_HTTPS",
                )
                return ""
            import urllib.parse as _urllib
            _parsed = _urllib.urlparse(url)
            if not _parsed.scheme or _parsed.scheme != "https":
                logger.warning(
                    "COS_EVIDENCE_PRESIGN=INVALID_SCHEME",
                )
                return ""
            if not _parsed.hostname:
                logger.warning(
                    "COS_EVIDENCE_PRESIGN=NO_HOST",
                )
                return ""
            if _parsed.username is not None or _parsed.password is not None:
                logger.warning(
                    "COS_EVIDENCE_PRESIGN=CREDENTIALS_IN_URL",
                )
                return ""
            return url
        except Exception as exc:
            logger.warning(
                "COS_EVIDENCE_PRESIGN=FAILED error=%s",
                type(exc).__name__,
            )
            return ""

    def build_object_key(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        *,
        now: datetime | None = None,
    ) -> str:
        """Build a deterministic COS object key for evidence.log.gz.

        Layout: phanthymotus_pr/<repo-dir>/<YYYY-MM>/<YYYY-MM-DD>/pr-<N>/evidence-<40hex>.log.gz

        Args:
            repo: Full repo name, e.g. "4paradigm/phanthymotus".
            pr_number: Positive integer PR number (bool rejected).
            head_sha: Exact full 40-char lowercase hex commit SHA.
            now: Optional datetime for date segments. Defaults to now in Asia/Shanghai.

        Returns:
            Deterministic COS object key string.

        Raises:
            ValueError: On invalid pr_number, head_sha, or repo.
        """
        repo_dir = _REPO_DIR_MAP.get(repo)
        if repo_dir is None:
            raise ValueError(
                f"repo {repo!r} is not a supported repo for COS evidence"
            )

        # PR number: must be int, not bool, and > 0
        if isinstance(pr_number, bool) or not isinstance(pr_number, int):
            raise ValueError(f"pr_number must be a positive integer, got {pr_number!r}")
        if pr_number <= 0:
            raise ValueError(f"pr_number must be > 0, got {pr_number!r}")

        # head_sha: must be str, exact 40 lowercase hex chars
        if not isinstance(head_sha, str):
            raise ValueError(f"head_sha must be a string, got {type(head_sha).__name__}")
        if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
            raise ValueError(
                f"head_sha must be exactly 40 lowercase hex chars, got {head_sha!r}"
            )

        if now is None:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
        else:
            now = now.astimezone(ZoneInfo("Asia/Shanghai"))

        month = now.strftime("%Y-%m")
        day = now.strftime("%Y-%m-%d")

        return (
            f"{COS_ROOT}{repo_dir}/{month}/{day}/"
            f"pr-{pr_number}/evidence-{head_sha}.log.gz"
        )

    async def upload_evidence_archive(
        self, object_key: str, archive_bytes: bytes
    ) -> bool:
        """Upload the evidence.log.gz as bytes.

        Returns True on success, False on failure.
        """
        if self._fake:
            self._uploads.append({
                "key": object_key,
                "size": len(archive_bytes),
            })
            return True

        if not self._has_credentials():
            return False

        try:
            from qcloud_cos import CosConfig, CosS3Client  # type: ignore
        except Exception:
            logger.error("COS SDK import failed: import error")
            return False

        try:
            config = CosConfig(
                Region=self.config.cos_region,
                SecretId=self.config.cos_secret_id,
                SecretKey=self.config.cos_secret_key,
                Token=None,
            )
            client = CosS3Client(config)
            await asyncio.to_thread(
                client.put_object,
                Bucket=self.config.cos_bucket,
                Body=archive_bytes,
                Key=object_key,
            )
            return True
        except Exception as exc:
            logger.error("COS upload failed: %s", type(exc).__name__)
            return False

    @staticmethod
    def validate_object_key(
        repo: str, pr_number: int, head_sha: str, object_key: str,
    ) -> str:
        expected = CosClient._expected_object_key_prefix(repo, pr_number, head_sha)
        if not isinstance(object_key, str) or ".." in object_key or "\\" in object_key:
            raise ValueError("invalid COS object key")
        parts = object_key.split("/")
        if len(parts) != 6 or parts[0] != "phanthymotus_pr":
            raise ValueError("invalid COS object key layout")
        if parts[1] != expected[0] or parts[4] != f"pr-{pr_number}":
            raise ValueError("COS object key does not match requested identity")
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}", parts[2]):
            raise ValueError("invalid COS object key date")
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", parts[3]):
            raise ValueError("invalid COS object key date")
        try:
            parsed_day = datetime.strptime(parts[3], "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("invalid COS object key date") from exc
        if parsed_day.strftime("%Y-%m-%d") != parts[3] or parsed_day.strftime("%Y-%m") != parts[2]:
            raise ValueError("invalid COS object key date")
        if parts[5] != f"evidence-{head_sha}.log.gz":
            raise ValueError("COS object key does not match requested HEAD")
        return object_key

    @staticmethod
    def _expected_object_key_prefix(repo: str, pr_number: int, head_sha: str) -> tuple[str, str]:
        repo_dir = _REPO_DIR_MAP.get(repo)
        if repo_dir is None:
            raise ValueError("unsupported repo")
        if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
            raise ValueError("invalid PR number")
        if not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", head_sha):
            raise ValueError("invalid HEAD SHA")
        return repo_dir, f"{COS_ROOT}{repo_dir}/"

    async def download_evidence_archive(
        self, object_key: str, max_bytes: int,
    ) -> bytes:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self._fake:
            raise CosError("fake COS download is not available")
        if not self._has_credentials():
            raise CosError("COS credentials are unavailable")
        try:
            from qcloud_cos import CosConfig, CosS3Client  # type: ignore
            config = CosConfig(
                Region=self.config.cos_region,
                SecretId=self.config.cos_secret_id,
                SecretKey=self.config.cos_secret_key,
                Token=None,
            )
            client = CosS3Client(config)
            head = await asyncio.to_thread(
                client.head_object, Bucket=self.config.cos_bucket, Key=object_key,
            )
            content_length = head.get("Content-Length", head.get("content-length"))
            if isinstance(content_length, bool):
                raise CosError("invalid COS Content-Length")
            content_length = int(content_length)
            if content_length <= 0 or content_length > max_bytes:
                raise CosError("COS object size is outside the allowed bound")
            response = await asyncio.to_thread(
                client.get_object, Bucket=self.config.cos_bucket, Key=object_key,
            )
            stream = response["Body"].get_raw_stream()
            chunks: list[bytes] = []
            total = 0
            try:
                while total <= max_bytes:
                    chunk = await asyncio.to_thread(stream.read, min(1024 * 1024, max_bytes + 1 - total))
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise CosError("COS stream returned invalid data")
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        raise CosError("COS object exceeded the allowed bound")
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    await asyncio.to_thread(close)
            if total != content_length:
                raise CosError("COS object size changed during download")
            return b"".join(chunks)
        except CosError:
            raise
        except Exception as exc:
            logger.error("COS download failed: %s", type(exc).__name__)
            raise CosError("COS download failed") from exc
