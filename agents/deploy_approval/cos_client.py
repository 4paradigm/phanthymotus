"""COS (Tencent Cloud Object Storage) client for evidence uploads (final alignment).

Uses cos-python-sdk-v5 for authenticated uploads. Credentials are fully
separate from GitHub / Agent Core tokens.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import re
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Config, validate_repository_names

logger = logging.getLogger(__name__)

# Fixed root path for all COS evidence objects.
# Not configurable — aligns with the frozen COS layout contract.
COS_ROOT = "phanthymotus_pr/"
EVIDENCE_MAX_ARCHIVE_BYTES = 10 * 1024 * 1024

EVIDENCE_PRESIGNED_URL_TTL_SECONDS = 120



# Immutable historical object directories. This map is NOT an allowlist.
_REPO_DIR_MAP = {
    "4paradigm/phanthymotus": "phanthymotus",
    "4paradigm/phanthymotus-driver": "phanthymotus-driver",
}


def _repo_directory(repo: str) -> str:
    canonical = validate_repository_names([repo])[0]
    if canonical in _REPO_DIR_MAP:
        return _REPO_DIR_MAP[canonical]
    owner, name = canonical.split("/")
    # Reversible, injective encoding. No slashes, dots, or ambiguous separators;
    # the reserved prefix cannot collide with either historical short directory.
    return f"repo-v1-{owner.encode('ascii').hex()}-{name.encode('ascii').hex()}"


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

    def _is_controlled_key(self, object_key: str) -> bool:
        try:
            repo = self._repo_from_key(object_key)
            if repo not in validate_repository_names(self.config.github_repos):
                return False
            self.validate_object_key(repo, self._pr_number_from_key(object_key),
                                     self._head_sha_from_key(object_key), object_key)
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    def generate_evidence_download_url(
        self,
        object_key: str,
    ) -> str:
        """Generate a short-lived HTTPS presigned GET URL for an evidence object.

        Returns empty string on failure (fail-closed).  Never leaks secrets.
        """
        if not self._is_controlled_key(object_key):
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
        canonical = validate_repository_names([repo])[0]
        if canonical not in validate_repository_names(self.config.github_repos):
            raise ValueError(
                "repo is not configured for COS evidence"
            )
        repo_dir = _repo_directory(canonical)

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

        tz = ZoneInfo("Asia/Shanghai")

        if now is None:
            now = datetime.now(tz)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=tz)
        else:
            now = now.astimezone(tz)
        month = now.strftime("%Y-%m")
        day = now.strftime("%Y-%m-%d")

        return (
            f"{COS_ROOT}{repo_dir}/{month}/{day}/"
            f"pr-{pr_number}/evidence-{head_sha}.log.gz"
        )

    async def upload_evidence_archive(
        self, object_key: str, archive_bytes: bytes, *, expected_result: str = ""
    ) -> bool:
        """Upload the evidence.log.gz as bytes with pre-upload overwrite guard.

        Before PUT:
        1. HEAD the object.
        2. 404 -> safe to upload new object.
        3. Exists -> download, verify identity (SHA256/gzip/repo/pr/head/result).
           - If identity correct (including result match if provided) -> return True
             (reuse existing, no PUT).
           - If identity wrong or corrupted -> return False (reject overwrite).
        4. HEAD error (403, timeout, network) -> UNKNOWN -> return False.

        Uses Tencent COS ForbidOverwrite parameter ONLY when bucket
        versioning is confirmed absent (empty/None response).
        Enabled, Suspended, or unknown versioning -> fail closed.
        """
        if not self._is_controlled_key(object_key):
            return False
        if self._fake:
            # T1LAST Issue 2: Validate incoming archive_bytes BEFORE any existing lookup.
            # Prevents passing invalid bytes through when existing object happens to match.
            if not isinstance(archive_bytes, bytes):
                return False
            archive_sha = hashlib.sha256(archive_bytes).hexdigest()
            validation_err = CosClient._validate_evidence_identity(
                object_key, archive_bytes,
                expected_sha256=archive_sha,
                expected_size=len(archive_bytes),
                repo=self._repo_from_key(object_key),
                pr_number=self._pr_number_from_key(object_key),
                head_sha=self._head_sha_from_key(object_key),
                expected_result=expected_result,
            )
            if validation_err is not None:
                logger.warning(
                    "Fake COS %s: incoming archive identity invalid (%s) — "
                    "rejecting",
                    object_key, validation_err,
                )
                return False
            # Prevent duplicate uploads for the same key (idempotent reuse)
            for upload in self._uploads:
                if upload.get("key") == object_key:
                    # Existing entry found — verify full identity matches
                    # before reusing.  Do NOT require identical compression
                    # length: timestamps / log content may differ while the
                    # repo/pr/head/result identity remains the same.
                    existing = upload.get("bytes")
                    if not isinstance(existing, bytes) or len(existing) == 0:
                        logger.warning(
                            "Fake COS %s: existing key has empty bytes — "
                            "rejecting overwrite",
                            object_key,
                        )
                        return False
                    # Verify SHA256 and full identity of existing object
                    existing_sha = hashlib.sha256(existing).hexdigest()
                    err = CosClient._validate_evidence_identity(
                        object_key, existing,
                        expected_sha256=existing_sha,
                        expected_size=len(existing),
                        repo=self._repo_from_key(object_key),
                        pr_number=self._pr_number_from_key(object_key),
                        head_sha=self._head_sha_from_key(object_key),
                        expected_result=expected_result,
                    )
                    if err is None:
                        return True
                    # Identity mismatch or corruption — reject overwrite.
                    logger.warning(
                        "Fake COS %s: existing key identity invalid (%s) — "
                        "rejecting overwrite",
                        object_key, err,
                    )
                    return False
            # Key does not exist — validate local archive then store.
            # Validate BEFORE any write to prevent uploading invalid evidence.
            archive_sha = hashlib.sha256(archive_bytes).hexdigest()
            validation_err = CosClient._validate_evidence_identity(
                object_key, archive_bytes,
                expected_sha256=archive_sha,
                expected_size=len(archive_bytes),
                repo=self._repo_from_key(object_key),
                pr_number=self._pr_number_from_key(object_key),
                head_sha=self._head_sha_from_key(object_key),
                expected_result=expected_result,
            )
            if validation_err is not None:
                logger.warning(
                    "Fake COS %s: local archive identity invalid (%s) — "
                    "refusing upload",
                    object_key, validation_err,
                )
                return False
            self._uploads.append({
                "key": object_key,
                "size": len(archive_bytes),
                "bytes": archive_bytes,
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


            # Pre-validate local archive identity BEFORE any COS operation.
            # Prevents uploading invalid evidence even when object doesn't exist.
            archive_sha = hashlib.sha256(archive_bytes).hexdigest()
            validation_err = CosClient._validate_evidence_identity(
                object_key, archive_bytes,
                expected_sha256=archive_sha,
                expected_size=len(archive_bytes),
                repo=self._repo_from_key(object_key),
                pr_number=self._pr_number_from_key(object_key),
                head_sha=self._head_sha_from_key(object_key),
                expected_result=expected_result,
            )
            if validation_err is not None:
                logger.warning(
                    "COS %s: local archive identity invalid (%s) — "
                    "refusing upload",
                    object_key, validation_err,
                )
                return False
            # ── Pre-check: does the object already exist? ──
            existing_length = None
            try:
                head = await asyncio.to_thread(
                    client.head_object,
                    Bucket=self.config.cos_bucket,
                    Key=object_key,
                )
                cl = head.get("Content-Length", head.get("content-length"))
                # HEAD 200 requires a valid, positive Content-Length.
                # Missing, bool, non-numeric, negative, 0, or exceeding limit
                # means object exists but is unverifiable — reject PUT.
                if isinstance(cl, bool) or not cl:
                    logger.warning(
                        "COS %s: HEAD 200 but Content-Length missing/invalid — "
                        "rejecting PUT to avoid overwrite",
                        object_key,
                    )
                    return False
                try:
                    cl_int = int(cl)
                except (ValueError, TypeError):
                    logger.warning(
                        "COS %s: HEAD 200 but Content-Length non-numeric — "
                        "rejecting PUT to avoid overwrite",
                        object_key,
                    )
                    return False
                if cl_int <= 0 or cl_int > EVIDENCE_MAX_ARCHIVE_BYTES:
                    logger.warning(
                        "COS %s: HEAD 200 Content-Length=%d out of bounds — "
                        "rejecting PUT to avoid overwrite",
                        object_key, cl_int,
                    )
                    return False
                existing_length = cl_int
            except Exception as exc:
                # Use CosServiceError.get_status_code() for precise 404 detection ONLY.
                try:
                    from qcloud_cos import CosServiceError  # type: ignore
                    if isinstance(exc, CosServiceError):
                        status = exc.get_status_code()
                        if status == 404:
                            pass  # safe to proceed with new upload
                        else:
                            logger.warning(
                                "COS head_object status=%s for %s — "
                                "rejecting PUT to avoid overwrite",
                                status, object_key,
                            )
                            return False
                    else:
                        # Non-CosServiceError (network, timeout, etc.) -> UNKNOWN
                        logger.warning(
                            "COS head_object error for %s — "
                            "rejecting PUT to avoid overwrite: %s",
                            object_key, type(exc).__name__,
                        )
                        return False
                except Exception:
                    # CosServiceError unavailable -> UNKNOWN, reject PUT.
                    # Do NOT fall back to string parsing which could misidentify
                    # unknown exceptions as 404.
                    logger.warning(
                        "COS head_object error for %s — "
                        "rejecting PUT to avoid overwrite: %s",
                        object_key, type(exc).__name__,
                    )
                    return False

            if existing_length is not None:
                # Object exists — verify identity before deciding to overwrite.
                # Read with bounded size (compression limit).
                try:
                    download_resp = await asyncio.to_thread(
                        client.get_object,
                        Bucket=self.config.cos_bucket,
                        Key=object_key,
                    )
                    stream = download_resp["Body"].get_raw_stream()
                    existing_bytes = b""
                    try:
                        remaining = EVIDENCE_MAX_ARCHIVE_BYTES
                        while remaining > 0:
                            chunk_size = min(1024 * 1024, remaining)
                            chunk = await asyncio.to_thread(stream.read, chunk_size)
                            if not chunk:
                                break
                            if not isinstance(chunk, bytes):
                                raise CosError("invalid chunk type")
                            existing_bytes += chunk
                            remaining -= len(chunk)
                            # Strict truncation check: verify stream exhaustion at limit
                            if remaining <= 0:
                                try:
                                    peek = await asyncio.to_thread(stream.read, 1)
                                    if peek:
                                        raise CosError("existing object truncated at limit")
                                except CosError:
                                    raise
                                except Exception:
                                    # Stream error during truncation check — reject
                                    raise CosError("existing object stream error at limit")
                    finally:
                        close = getattr(stream, "close", None)
                        if close is not None:
                            await asyncio.to_thread(close)
                except CosError:
                    raise
                except Exception:
                    logger.warning(
                        "COS get_object failed for %s — "
                        "cannot verify existing object, rejecting PUT",
                        object_key,
                    )
                    return False

                # Strictly compare HEAD Content-Length with actual read bytes.
                if existing_length is not None:
                    if len(existing_bytes) != existing_length:
                        logger.warning(
                            "COS %s: HEAD length %d != actual %d — "
                            "rejecting existing object",
                            object_key, existing_length, len(existing_bytes),
                        )
                        return False
                # Verify SHA256 of compressed bytes against existing data.
                existing_sha = hashlib.sha256(existing_bytes).hexdigest()

                # Verify gzip integrity of existing object with bounded decompression.
                try:
                    decompressed_size = 0
                    with gzip.GzipFile(fileobj=io.BytesIO(existing_bytes)) as gf:
                        while True:
                            chunk = gf.read(65536)
                            if not chunk:
                                break
                            decompressed_size += len(chunk)
                            if decompressed_size > EVIDENCE_MAX_ARCHIVE_BYTES:
                                break
                except Exception:
                    logger.warning(
                        "COS %s: existing object gzip corrupted — "
                        "rejecting reuse",
                        object_key,
                    )
                    return False

                # Validate full object key layout
                try:
                    CosClient.validate_object_key(
                        archive_bytes and self._repo_from_key(object_key) or "",
                        self._pr_number_from_key(object_key),
                        self._head_sha_from_key(object_key),
                        object_key,
                    )
                except ValueError:
                    logger.warning(
                        "COS %s: object key layout invalid — rejecting",
                        object_key,
                    )
                    return False

                # Verify identity fields from existing object
                identity = CosClient._parse_evidence_identity(existing_bytes)
                validation_err = CosClient._validate_evidence_identity(
                    object_key,
                    existing_bytes,
                    expected_sha256=existing_sha,
                    expected_size=len(existing_bytes),
                    repo=archive_bytes and self._repo_from_key(object_key) or "",
                    pr_number=self._pr_number_from_key(object_key),
                    head_sha=self._head_sha_from_key(object_key),
                    expected_result=expected_result,
                )
                if validation_err is not None:
                    logger.warning(
                        "COS %s: existing identity invalid (%s) — "
                        "rejecting overwrite",
                        object_key, validation_err,
                    )
                    return False

                # Existing object has valid identity — reuse it (idempotent).
                # Different compressed bytes are acceptable when identity is valid.
                logger.info(
                    "COS %s: existing object verified (idempotent reuse)",
                    object_key,
                )
                return True

            # ── Object does not exist (404) — proceed with guarded PUT ──

            # Check bucket versioning: ForbidOverwrite only valid on non-versioned buckets.
            forbid_overwrite = False
            versioning_ok = False
            try:
                versioning_result = await asyncio.to_thread(
                    client.get_bucket_versioning,
                    Bucket=self.config.cos_bucket,
                )
                # Must be a dict response to trust any status.
                # None, non-dict, Enabled, Suspended, 0, unknown status
                # all mean versioning state is not confirmed absent — reject PUT.
                if not isinstance(versioning_result, dict):
                    logger.warning(
                        "COS %s: versioning returned non-dict %r — "
                        "refusing PUT to avoid overwrite",
                        object_key, type(versioning_result).__name__,
                    )
                elif "Status" not in versioning_result:
                    # Key missing ({}) → bucket is non-versioned, allow PUT.
                    forbid_overwrite = True
                    versioning_ok = True
                else:
                    raw_status = versioning_result["Status"]
                    # Explicit Status=None or non-string → reject.
                    if raw_status is None:
                        logger.warning(
                            "COS %s: versioning Status is explicitly None — "
                            "refusing PUT to avoid overwrite",
                            object_key,
                        )
                    elif not isinstance(raw_status, str):
                        logger.warning(
                            "COS %s: versioning Status is %r (not string) — "
                            "refusing PUT to avoid overwrite",
                            object_key, raw_status,
                        )
                    else:
                        status = raw_status or ""
                        if status in ("Enabled", "Suspended"):
                            logger.warning(
                                "COS %s: bucket versioning is %s — "
                                "refusing ForbidOverwrite, fail closed",
                                object_key, status,
                            )
                        elif status == "":
                            # Empty status string in a valid dict response
                            # confirms the bucket is non-versioned.
                            forbid_overwrite = True
                            versioning_ok = True
                        else:
                            # Unknown status value -> fail closed
                            logger.warning(
                                "COS %s: unknown versioning status %r — "
                                "refusing ForbidOverwrite, fail closed",
                                object_key, status,
                            )
            except Exception as exc:
                # Versioning query error -> fail closed, reject PUT entirely
                logger.warning(
                    "COS %s: versioning check failed (%s) — "
                    "refusing PUT to avoid overwrite",
                    object_key, type(exc).__name__,
                )

            # Only proceed with PUT when versioning is confirmed absent.
            if not versioning_ok:
                logger.warning(
                    "COS %s: cannot confirm bucket is non-versioned — "
                    "refusing PUT to avoid overwrite",
                    object_key,
                )
                return False
            try:
                kwargs: dict = {
                    "Bucket": self.config.cos_bucket,
                    "Body": archive_bytes,
                    "Key": object_key,
                }
                if forbid_overwrite:
                    kwargs["ForbidOverwrite"] = "true"
                await asyncio.to_thread(client.put_object, **kwargs)
                return True
            except Exception as exc:
                logger.error("COS upload failed: %s", type(exc).__name__)
                return False
        except Exception as exc:
            logger.error("COS upload failed: %s", type(exc).__name__)
            return False

    @staticmethod
    def _repo_from_key(object_key: str) -> str:
        """Resolve full repo name from COS object key short directory."""
        parts = object_key.split("/")
        if len(parts) >= 2:
            short_dir = parts[1]
            for full_repo, repo_dir in _REPO_DIR_MAP.items():
                if repo_dir == short_dir:
                    return full_repo
            if short_dir.startswith("repo-v1-"):
                try:
                    owner_hex, name_hex = short_dir[len("repo-v1-"):].split("-")
                    repo = bytes.fromhex(owner_hex).decode("ascii") + "/" + bytes.fromhex(name_hex).decode("ascii")
                    canonical = validate_repository_names([repo])[0]
                    if _repo_directory(canonical) == short_dir:
                        return canonical
                except (ValueError, UnicodeDecodeError):
                    pass
        return ""

    @staticmethod
    def _pr_number_from_key(object_key: str) -> int:
        """Extract PR number from key path segment pr-N."""
        parts = object_key.split("/")
        for part in parts:
            if part.startswith("pr-"):
                try:
                    val = int(part[3:])
                    if val > 0:
                        return val
                except ValueError:
                    pass
        return 0

    @staticmethod
    def _head_sha_from_key(object_key: str) -> str:
        """Extract HEAD SHA from key filename evidence-<sha>.log.gz."""
        filename = object_key.split("/")[-1]
        if filename.startswith("evidence-") and filename.endswith(".log.gz"):
            sha_part = filename[len("evidence-"):-len(".log.gz")]
            if re.fullmatch(r"[0-9a-f]{40}", sha_part):
                return sha_part
        return ""

    @staticmethod
    def _parse_evidence_identity(archive_bytes: bytes) -> dict:
        """Parse repo/pr/head/result binding fields from evidence bytes.

        Only reads fields from the [metadata] section (between
        [metadata] and the next section header or EOF).
        Uses bounded decompression to prevent excessive memory use.
        Returns empty dict when the body cannot be read or lacks
        the expected identity fields, so callers can fail closed.
        """
        try:
            decompressed_size = 0
            chunks: list[bytes] = []
            with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes)) as gf:
                while True:
                    chunk = gf.read(65536)
                    if not chunk:
                        break
                    decompressed_size += len(chunk)
                    if decompressed_size > EVIDENCE_MAX_ARCHIVE_BYTES:
                        return {}
                    chunks.append(chunk)
            text = b"".join(chunks).decode("utf-8", "strict")
        except Exception:
            return {}
        identity: dict[str, str] = {}
        in_metadata = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "[metadata]":
                in_metadata = True
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                # Entering a new section — stop reading metadata
                if in_metadata:
                    break
                continue
            if not in_metadata:
                continue
            for field in ("repo", "pr", "head", "result"):
                key = f"{field}="
                if stripped.lower().startswith(key):
                    identity[field] = stripped[len(key):].strip()
        return identity

    @staticmethod
    def _validate_evidence_identity(
        key: str,
        archive_bytes: bytes,
        *,
        expected_sha256: str,
        expected_size: int,
        repo: str,
        pr_number: int,
        head_sha: str,
        expected_result: str = "",
    ) -> str | None:
        """Return an error reason string when the evidence is invalid, else None.

        Enforces:
        - type checks for all parameters,
        - compressed size <= 10MiB,
        - len(archive_bytes) == expected_size,
        - non-empty 64-char lowercase hex SHA256,
        - SHA256 of real compressed bytes MUST match expected_sha256,
        - gzip magic/CRC/trailer integrity,
        - decompressed size <= 10MiB,
        - repo/pr/head/result identity binding,
        - key layout identity (repo dir, PR number, HEAD).
        """

        # --- Type checks first ---
        if (
            not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        ):
            return "invalid_expected_sha256"
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
            return "invalid_expected_size"
        if not isinstance(archive_bytes, bytes):
            return "invalid_archive_bytes"
        # --- Size checks ---
        if len(archive_bytes) > EVIDENCE_MAX_ARCHIVE_BYTES:
            return "compressed_too_large"
        if len(archive_bytes) != expected_size:
            return "size_mismatch"
        # --- Gzip magic ---
        if not archive_bytes.startswith(b'\x1f\x8b'):
            return "gzip_magic_missing"


        # Verify SHA256 of real compressed bytes matches expected.
        actual_sha = hashlib.sha256(archive_bytes).hexdigest()
        if actual_sha != expected_sha256:
            return "sha256_mismatch"

        # Verify full gzip stream (CRC/trailer) with bounded decompression.
        # Track actual decompressed bytes (not iteration count).
        decompressed_size = 0
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes)) as gf:
                while True:
                    chunk = gf.read(65536)
                    if not chunk:
                        break
                    decompressed_size += len(chunk)
                    if decompressed_size > EVIDENCE_MAX_ARCHIVE_BYTES:
                        return "decompressed_too_large"
        except Exception:
            return "gzip_corrupted"
        if decompressed_size > EVIDENCE_MAX_ARCHIVE_BYTES:
            return "decompressed_too_large"

        identity = CosClient._parse_evidence_identity(archive_bytes)
        if not identity.get("repo") or not identity.get("pr") or not identity.get("head"):
            return "identity_fields_missing"

        # Use strict validate_object_key() to check the FULL key layout.
        # This prevents substring mismatches (e.g. INVALID-10 directory,
        # path traversal, wrong year/month segments) from being accepted.
        try:
            CosClient.validate_object_key(repo, pr_number, head_sha, key)
        except ValueError:
            return "key_layout_invalid"

        # Strictly compare identity metadata fields against requested values.
        # Key layout may be valid but identity content could differ.
        if identity.get("repo") != repo:
            return "repo_mismatch"
        if identity.get("head") != head_sha:
            return "head_mismatch"

        # identity["pr"] should be plain decimal (e.g. "42")
        try:
            identity_pr = int(identity["pr"])
        except (ValueError, TypeError):
            return "pr_mismatch"
        if identity_pr != pr_number:
            return "pr_mismatch"

        # Result binding: compare identity result to expected result if provided
        if expected_result:
            if identity.get("result") != expected_result:
                return "result_mismatch"
        elif identity.get("result") not in ("pass", "fail"):
            return "result_unrecognized"

        return None

    async def verify_upload(
        self, object_key: str, expected_sha256: str, expected_size: int,
    ) -> bool:
        """Verify an uploaded COS object exists and matches expected metadata.

        Uses head_object to check existence and Content-Length,
        then downloads the object and validates SHA256, gzip integrity,
        and repo/PR/HEAD/result identity binding.

        Upload-time pre-check: if a matching object already exists with
        matching metadata, return True immediately (idempotent retry).
        """
        if not self._is_controlled_key(object_key):
            return False
        if self._fake:
            # In fake mode, find the upload and verify real bytes, SHA, size,
            # gzip integrity, and identity binding — same contract as real COS.
            target = None
            for upload in self._uploads:
                if upload.get("key") == object_key:
                    target = upload
                    break
            if target is None:
                return False
            uploaded_bytes = target.get("bytes")
            if not isinstance(uploaded_bytes, bytes) or len(uploaded_bytes) == 0:
                return False
            if len(uploaded_bytes) != expected_size:
                return False
            if hashlib.sha256(uploaded_bytes).hexdigest() != expected_sha256:
                return False
            if not uploaded_bytes.startswith(b'\x1f\x8b'):
                return False
            try:
                decompressed_size = 0
                with gzip.GzipFile(fileobj=io.BytesIO(uploaded_bytes)) as gf:
                    while True:
                        chunk = gf.read(65536)
                        if not chunk:
                            break
                        decompressed_size += len(chunk)
                        if decompressed_size > EVIDENCE_MAX_ARCHIVE_BYTES:
                            return False
            except Exception:
                return False
            # Resolve repo/pr/head from key for identity validation
            key_repo = CosClient._repo_from_key(object_key)
            key_pr = CosClient._pr_number_from_key(object_key)
            key_head = CosClient._head_sha_from_key(object_key)
            err = CosClient._validate_evidence_identity(
                object_key, uploaded_bytes,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                repo=key_repo,
                pr_number=key_pr,
                head_sha=key_head,
                expected_result="",
            )
            if err is not None:
                return False
            return True

        if not self._has_credentials():
            return False

        try:
            from qcloud_cos import CosConfig, CosS3Client  # type: ignore
            config = CosConfig(
                Region=self.config.cos_region,
                SecretId=self.config.cos_secret_id,
                SecretKey=self.config.cos_secret_key,
                Token=None,
            )
            client = CosS3Client(config)

            # 1. HEAD pre-check: verify object existence.
            # HEAD exception -> always return False (403/404/timeout/network).
            # Only HEAD 200 with valid Content-Length proceeds.
            try:
                head = await asyncio.to_thread(
                    client.head_object, Bucket=self.config.cos_bucket, Key=object_key,
                )
            except Exception:
                # HEAD failed (403, 404, timeout, network, SDK error) — reject.
                logger.warning(
                    "COS head_object failed for %s — rejecting verification",
                    object_key,
                )
                return False

            # HEAD 200 — validate Content-Length strictly.
            content_length_raw = head.get("Content-Length", head.get("content-length"))
            if isinstance(content_length_raw, bool) or not content_length_raw:
                logger.warning(
                    "COS verify %s: HEAD 200 but Content-Length missing/invalid — rejecting",
                    object_key,
                )
                return False
            try:
                existing_length = int(content_length_raw)
            except (ValueError, TypeError):
                logger.warning(
                    "COS verify %s: HEAD 200 but Content-Length non-numeric — rejecting",
                    object_key,
                )
                return False
            if existing_length <= 0 or existing_length > EVIDENCE_MAX_ARCHIVE_BYTES:
                logger.warning(
                    "COS verify %s: HEAD 200 Content-Length=%d out of bounds — rejecting",
                    object_key, existing_length,
                )
                return False

            # Size comparison — if existing object has different size, fail closed.
            if existing_length is not None and expected_size > 0:
                if existing_length != expected_size:
                    logger.warning(
                        "COS verify %s: existing size %d != expected %d — fail closed",
                        object_key, existing_length, expected_size,
                    )
                    return False
                # Size matches — object likely already valid, proceed to
                # SHA256 verification via download.


            # 2. Download and validate identity.
            # expected_sha256 must be non-empty — we never trust an
            # upload without a known SHA.
            if not expected_sha256:
                logger.warning(
                    "COS verify %s: empty expected_sha256 — rejecting",
                    object_key,
                )
                return False

            try:
                archive_bytes = await self.download_evidence_archive(
                    object_key, EVIDENCE_MAX_ARCHIVE_BYTES,
                )
            except CosError:
                return False

            # Resolve repo/pr/head from key using static helpers
            key_repo = CosClient._repo_from_key(object_key)
            key_pr = CosClient._pr_number_from_key(object_key)
            key_head = CosClient._head_sha_from_key(object_key)

            error_reason = CosClient._validate_evidence_identity(
                object_key,
                archive_bytes,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                repo=key_repo,
                pr_number=key_pr,
                head_sha=key_head,
                expected_result="",
            )
            if error_reason is not None:
                logger.warning(
                    "COS verify %s: identity invalid (%s)",
                    object_key, error_reason,
                )
                return False

            return True
        except CosError:
            return False
        except Exception as exc:
            logger.warning("COS verify failed for %s: %s", object_key, type(exc).__name__)
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
        repo_dir = _repo_directory(repo)
        if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
            raise ValueError("invalid PR number")
        if not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", head_sha):
            raise ValueError("invalid HEAD SHA")
        return repo_dir, f"{COS_ROOT}{repo_dir}/"

    async def download_evidence_archive(
        self, object_key: str, max_bytes: int,
    ) -> bytes:
        if not self._is_controlled_key(object_key):
            raise CosError("COS object key is not configured or has invalid identity")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self._fake:
            # Find the upload entry for this key and return its bytes
            for upload in self._uploads:
                if upload.get("key") == object_key:
                    uploaded_bytes = upload.get("bytes")
                    if isinstance(uploaded_bytes, bytes) and len(uploaded_bytes) > 0:
                        if len(uploaded_bytes) > max_bytes:
                            raise CosError("COS object size exceeds allowed bound")
                        return uploaded_bytes
            raise CosError("fake COS object not found")
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

    async def ensure_evidence_archive(
        self, *, object_key: str, archive_bytes: bytes,
        repo: str, pr_number: int, head_sha: str, result: str,
    ) -> dict:
        """Ensure an evidence archive exists in COS with identity validation.

        Before PUT: verify existing object identity (SHA256/gzip/repo/pr/head/result).
        Reuse if identity matches; reject overwrite if different or corrupted.
        After PUT: verify uploaded object matches expected metadata.

        Returns metadata dict on success:
            {"object_key": str, "sha256": str (64-hex), "size": int (>0)}
        Returns empty metadata on any failure.
        """
        if not archive_bytes or not self._is_controlled_key(object_key):
            return {"object_key": "", "sha256": "", "size": 0}

        try:
            # Build the archive first to get its hash/size
            archive_sha = hashlib.sha256(archive_bytes).hexdigest()
            archive_size = len(archive_bytes)

            # Validate identity before any COS operation
            validation_err = CosClient._validate_evidence_identity(
                object_key, archive_bytes,
                expected_sha256=archive_sha,
                expected_size=archive_size,
                repo=repo,
                pr_number=pr_number,
                head_sha=head_sha,
                expected_result=result,
            )
            if validation_err is not None:
                logger.warning(
                    "ensure_evidence_archive %s: local identity invalid (%s)",
                    object_key, validation_err,
                )
                return {"object_key": "", "sha256": "", "size": 0}

            # Attempt upload (returns bool: True=success/reuse, False=failure)
            ok = await self.upload_evidence_archive(
                object_key, archive_bytes, expected_result=result,
            )
            if not ok:
                return {"object_key": "", "sha256": "", "size": 0}

            # Download actual COS object to get real SHA256 and size
            try:
                actual_bytes = await self.download_evidence_archive(
                    object_key, EVIDENCE_MAX_ARCHIVE_BYTES,
                )
                actual_sha = hashlib.sha256(actual_bytes).hexdigest()
                actual_size = len(actual_bytes)

                # Verify the actual object is valid
                key_repo = CosClient._repo_from_key(object_key)
                key_pr = CosClient._pr_number_from_key(object_key)
                key_head = CosClient._head_sha_from_key(object_key)
                err = CosClient._validate_evidence_identity(
                    object_key, actual_bytes,
                    expected_sha256=actual_sha,
                    expected_size=actual_size,
                    repo=key_repo,
                    pr_number=key_pr,
                    head_sha=key_head,
                    expected_result=result,
                )
                if err is not None:
                    logger.warning(
                        "ensure_evidence_archive %s: actual object invalid (%s)",
                        object_key, err,
                    )
                    return {"object_key": "", "sha256": "", "size": 0}

                return {
                    "object_key": object_key,
                    "sha256": actual_sha,
                    "size": actual_size,
                }
            except CosError:
                return {"object_key": "", "sha256": "", "size": 0}
            except Exception as e:
                logger.warning(
                    "ensure_evidence_archive %s: post-upload download failed (%s)",
                    object_key, e,
                )
                return {"object_key": "", "sha256": "", "size": 0}
        except Exception as e:
            logger.warning(
                "ensure_evidence_archive %s: %s", object_key, e,
            )
            return {"object_key": "", "sha256": "", "size": 0}
