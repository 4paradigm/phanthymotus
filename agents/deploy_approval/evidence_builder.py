"""Minimal COS evidence bundle builder for Deploy Approval."""

from __future__ import annotations

import gzip
import hashlib
import re

from .config import Config


def _redact_secrets(text: str) -> str:
    """Comprehensive secret redaction on free-text content.

    Processing order:
    collect -> redact -> U+FFFD/BOM normalize -> combine -> newest-tail <= 10MiB
    -> strict UTF-8 -> gzip
    """
    if not isinstance(text, str):
        return text
    # GitHub tokens — classic PAT, fine-grained PAT, OAuth, App user/install/refresh tokens
    text = re.sub(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9._-]+", "[REDACTED_GITHUB_TOKEN]", text)
    text = re.sub(r"github_pat_[A-Za-z0-9_]+", "[REDACTED_GITHUB_TOKEN]", text)

    # 2. PEM private key blocks
    text = re.sub(
        r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----.*?-----END \1-----",
        "[REDACTED_PRIVATE_KEY]",
        text,
        flags=re.DOTALL,
    )

    # Authorization and bearer credentials
    text = re.sub(
        r"(Authorization:\s*Bearer\s+)[^\s]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(\bBearer\s+)[^\s]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    # token/password/secret/cookie key-value credentials
    text = re.sub(
        r"(token\s*[=:]\s*)[^\s&;]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(password\s*[=:]\s*)[^\s&;]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(secret\s*[=:]\s*)[^\s&;]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(cookie\s*[=:]\s*)[^\s&;]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    # Registry Basic credentials
    text = re.sub(
        r"(Authorization:\s*Basic\s+)[A-Za-z0-9+/=]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    # COS / S3 presigned query credentials
    text = re.sub(
        r"((?:X-Amz-Signature|X-Amz-Credential|X-Amz-Algorithm|X-Amz-Signature|sig|AWSAccessKeyId|X-Goog-Signature)=[^&]+)",
        "[REDACTED_PRESIGNED]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:q-ak|q-signature|q-sign-time|q-key-time|q-sign-algorithm|q-header-list|q-url-param-list)=[^&\s]+",
        "[REDACTED_COS_QUERY]",
        text,
        flags=re.IGNORECASE,
    )


    return text

def _case_id_for_target(target: str) -> str:
    if target == "perception":
        return "perception-health-check"
    if target == "actucore":
        return "actucore-health-check"
    if target == "driver":
        return "driver-health-check"
    return ""


def _sanitize_text(value: str) -> str:
    text = str(value or "")
    # Apply comprehensive secret redaction
    text = _redact_secrets(text)
    return text


def _sanitize_for_evidence(value: str) -> str:
    """Prepare a string for inclusion in evidence files.

    - Replace Unicode replacement character U+FFFD with [INVALID_UTF8].
    - Apply secret redaction.
    """
    if not isinstance(value, str):
        value = str(value or "")
    value = _redact_secrets(value)
    # Normalize replacement character and BOM after redaction
    value = value.replace("\ufffd", "[INVALID_UTF8]")
    # Replace BOM
    value = value.replace("\ufeff", "[INVALID_UTF8]")
    return value


def _short_digest(ref: str) -> str:
    """Return a compact display for an image ref: short digest or truncated tag."""
    value = str(ref or "").strip()
    if "@sha256:" in value:
        digest = value.rsplit("@sha256:", 1)[-1]
        if len(digest) < 12:
            return ""
        return f"@sha256:{digest[:12]}"
    # Tag form
    if ":" in value:
        tag = value.rsplit(":", 1)[-1]
        if len(tag) > 40:
            return tag[:40] + "…"
        return tag
    return ""


def _compact_running_image(ref: str) -> str:
    """Compact display for running_image evidence."""
    result = _short_digest(ref)
    if result:
        return result
    return "occupied"


EVIDENCE_MAX_ARCHIVE_BYTES = 10 * 1024 * 1024
_EVIDENCE_MAX_BYTES = EVIDENCE_MAX_ARCHIVE_BYTES
_MARKER_PREFIX = "[truncated: showing last 10 MiB]\n"


def _bound_text(text: str, limit: int | None = None) -> str:
    """Byte-bound text to *limit* bytes.

    New contract (v11): keep the *newest tail*, discard old head.
    Marker is always "[truncated: showing last 10 MiB]" as the first line.
    When limit is None it defaults to ``_EVIDENCE_MAX_BYTES``.

    If the text fits within the limit, it is returned unchanged (no marker).
    If it exceeds the limit, the marker + newest tail is returned, and the
    total UTF-8 byte length is guaranteed <= limit.
    """
    if limit is None:
        limit = _EVIDENCE_MAX_BYTES
    encoded = text.encode("utf-8", errors="strict")
    if len(encoded) <= limit:
        return text
    marker_bytes = len(_MARKER_PREFIX.encode("utf-8"))
    budget = limit - marker_bytes
    if budget <= 0:
        return _MARKER_PREFIX
    # Take the last `budget` bytes (newest tail).
    tail_bytes = encoded[-budget:]
    # Fix leading partial UTF-8 codepoint.
    while len(tail_bytes) > 0 and (tail_bytes[0] & 0xC0) == 0x80:
        tail_bytes = tail_bytes[1:]
    # Try to start at a whole-line boundary: find first newline in tail.
    nl_pos = tail_bytes.find(b"\n")
    if nl_pos >= 0:
        tail_bytes = tail_bytes[nl_pos + 1:]
    result = _MARKER_PREFIX + tail_bytes.decode("utf-8", errors="strict")
    # Final guarantee: total bytes <= limit.
    result_bytes = result.encode("utf-8")
    if len(result_bytes) > limit:
        # Trim tail_bytes to fit exactly.
        excess = len(result_bytes) - limit
        # We need to trim `excess` bytes from the tail portion.
        tail_bytes = tail_bytes[:-excess] if excess < len(tail_bytes) else b""
        # Re-fix UTF-8 leading partial codepoint.
        while len(tail_bytes) > 0 and (tail_bytes[0] & 0xC0) == 0x80:
            tail_bytes = tail_bytes[1:]
        nl_pos = tail_bytes.find(b"\n")
        if nl_pos >= 0:
            tail_bytes = tail_bytes[nl_pos + 1:]
        result = _MARKER_PREFIX + tail_bytes.decode("utf-8", errors="strict")
    return result


class EvidenceBuilder:
    def __init__(self, config: Config):
        self.config = config

    async def build_evidence(
        self,
        *,
        repo: str,
        pr_number: int,
        head_sha: str,
        state: dict,
        result: str,
        summary: str = "",
        deploy_error: str = "",
        runtime_logs: dict[str, str] | None = None,
    ) -> tuple[bytes, str, int]:
        """Build evidence.log, gzip it, return (gzip_bytes, sha256, size)."""
        fixed_header = self._build_fixed_header(
            repo, pr_number, head_sha, state, result, summary, deploy_error,
        )
        variable_body = self._build_variable_body(runtime_logs)
        gzip_bytes = self._fit_evidence_archive(fixed_header, variable_body)

        sha256_hex = hashlib.sha256(gzip_bytes).hexdigest()
        return gzip_bytes, sha256_hex, len(gzip_bytes)

    def _build_fixed_header(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        state: dict,
        result: str,
        summary: str,
        deploy_error: str = "",
    ) -> str:
        safe_summary = _bound_text(_sanitize_for_evidence(summary), 4096)
        lines: list[str] = [
            "[metadata]",
            f"repo={repo}",
            f"pr={pr_number}",
            f"head={head_sha}",
            f"review_build_comment_id={state.get('review_evidence', {}).get('build_comment_id', '')}",
            f"review_test_comment_id={state.get('review_evidence', {}).get('test_comment_id', 0)}",
            f"review_code_review_comment_id={state.get('review_evidence', {}).get('code_review_comment_id', '')}",
            f"review_commit_prefix={state.get('review_evidence', {}).get('commit_prefix', '')}",
            f"review_resolved_head_sha={state.get('review_evidence', {}).get('resolved_head_sha', '')}",
            f"review_author_id={state.get('review_evidence', {}).get('review_author_id', '')}",
            f"result={result}",
            "",
            "[deploy]",
        ]
        components = state.get("components", [])
        if isinstance(components, list) and components:
            comp_text = ", ".join(
                f"{str(c.get('component_id', ''))}:{str(c.get('target', ''))}"
                for c in components
                if isinstance(c, dict) and c.get("component_id")
            )
            if comp_text:
                lines.append(f"components={comp_text}")
        approve_attempts = state.get("approve_attempts", [])
        if isinstance(approve_attempts, list):
            for attempt in approve_attempts:
                if not isinstance(attempt, dict):
                    continue
                lines.append(f"comment={attempt.get('comment_id', 0)} actor={attempt.get('actor', '')} machine={attempt.get('machine', '')} outcome={attempt.get('outcome', '')}")
                for item in attempt.get("preflight", []) or []:
                    if not isinstance(item, dict):
                        continue
                    lines.append(f"runtime={item.get('runtime_id', '')} running_image={_compact_running_image(item.get('running_image', ''))} component={item.get('component_id', '')}")
        if deploy_error:
            lines.append(f"terminal_error={deploy_error}")
        lines.extend(("", "[case]"))
        case_results = state.get("case_results", {})
        if isinstance(case_results, dict):
            for cid, case_result in case_results.items():
                lines.append(f"component={cid} result={case_result} advisory=true")
        if state.get("command", {}).get("kind") == "record_test":
            lines.extend(("", "[test]", f"result={result or state.get('test_result', '')}", f"summary={safe_summary}"))
        return _sanitize_for_evidence("\n".join(lines))

    def _build_variable_body(self, runtime_logs: dict[str, str] | None) -> str:
        lines: list[str] = []
        if runtime_logs:
            for source, log_text in runtime_logs.items():
                lines.extend(("", "[runtime-log]", f"source={source}"))
                lines.append(_sanitize_for_evidence(log_text))
        return "\n".join(lines)

    def _fit_evidence_archive(self, fixed_header: str, variable_body: str) -> bytes:
        fixed_bytes = fixed_header.encode("utf-8", errors="strict")
        if len(fixed_bytes) > EVIDENCE_MAX_ARCHIVE_BYTES:
            raise ValueError("fixed evidence header exceeds 10 MiB")
        variable_bytes = variable_body.encode("utf-8", errors="strict")
        low, high = 0, len(variable_bytes)
        best = b""
        while low <= high:
            candidate_size = (low + high) // 2
            candidate_tail = _bound_text(variable_bytes[-candidate_size:].decode("utf-8", errors="ignore"), candidate_size) if candidate_size else ""
            combined = fixed_bytes + (b"\n" + candidate_tail.encode("utf-8", errors="strict") if candidate_tail else b"")
            if len(combined) <= EVIDENCE_MAX_ARCHIVE_BYTES:
                archive = gzip.compress(combined, mtime=0)
                if len(archive) <= EVIDENCE_MAX_ARCHIVE_BYTES:
                    best = archive
                    low = candidate_size + 1
                    continue
            high = candidate_size - 1
        if not best:
            best = gzip.compress(fixed_bytes, mtime=0)
            if len(best) > EVIDENCE_MAX_ARCHIVE_BYTES:
                raise ValueError("fixed evidence archive exceeds 10 MiB")
        return best
