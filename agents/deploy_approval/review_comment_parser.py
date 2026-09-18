"""Parse Review Agent output from GitHub PR conversation comments.

This module replaces the previous Review Agent HTTP API dependency.
Deploy Approval reads GitHub PR comments written by the trusted Review
Agent operator and extracts build / test / code-review evidence from
Markdown-formatted comments.

Comment protocol (written by agents/pr_review, NOT modified here):
  Marker: ``<!-- pr-review-agent -->``
  Build Result heading: ``## PR Review Agent — Build Result``
  Test Results heading: ``## PR Review Agent — Test Results``
  Code Review heading:  ``## PR Review Agent — Code Review``

All parsing is strict and fail-closed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

PR_REVIEW_MARKER = "<!-- pr-review-agent -->"
BUILD_HEADING = "## PR Review Agent — Build Result"
TEST_HEADING = "## PR Review Agent — Test Results"
CODE_REVIEW_HEADING = "## PR Review Agent — Code Review"

_COMMIT_PREFIX_RE = re.compile(r"^Commit:\s*(.+)$")
_TABLE_ROW_RE = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$")
_IMAGE_SECTION_RE = re.compile(
    r"\*\*(.+?)\*\*\s*\n\s*```(?:\n|(.*?))\n\s*(.+?)\s*```",
    re.DOTALL,
)
_IMAGE_REF_RE = re.compile(
    r"^([^\s:/]+(?:/[^\s:/]+)+)/([^:\s]+(?::(.+))?)$"
)
_VERSION_RE = re.compile(r"^`(.+?)`$")
_SHORT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ReviewBuild:
    """Parsed build row from a Review Agent Build Result comment."""
    target: str = ""
    driver_path: str = ""
    variant: str = ""
    success: bool = False
    version: str = ""
    image_tag: str = ""  # mutable image:tag from comment
    image_ref: str = ""  # full image reference (target + version)
    took: str = ""

    def label(self) -> str:
        return self.driver_path or self.target


@dataclass
class ReviewCommentEvidence:
    """Complete review evidence extracted from PR comments for one HEAD."""
    head_sha: str = ""
    commit_prefix: str = ""
    review_author_id: str = ""
    review_author_login: str = ""
    builds: list[ReviewBuild] = field(default_factory=list)
    build_comment_id: int = 0
    build_comment_created_at: str = ""
    build_comment_updated_at: str = ""
    test_comment_id: int = 0
    test_comment_created_at: str = ""
    test_passed: int = 0
    test_failed: int = 0
    test_skipped: bool = False
    code_review_comment_id: int = 0
    code_review_created_at: str = ""
    code_review_text: str = ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_image_ref(text: str) -> tuple[str, str]:
    """Parse image reference into (repo, tag)."""
    text = text.strip()
    if ":" in text:
        parts = text.rsplit(":", 1)
        return parts[0], parts[1]
    return text, ""


def _normalize_variant(variant: str) -> str:
    """Normalize legacy variant names to canonical forms."""
    v = variant.strip().lower()
    if v == "jetson-jp5.11":
        return "5.11"
    if v == "jetson-jp6.1":
        return "6.1"
    # Keep canonical forms as-is
    if v in ("5.11", "6.1"):
        return v
    return variant.strip()


# ------------------------------------------------------------------
# Comment identification
# ------------------------------------------------------------------

def _has_marker(body: str) -> bool:
    """Check if comment body contains the Review Agent marker."""
    return PR_REVIEW_MARKER in body


def _find_section(body: str, heading: str) -> str | None:
    """Find text between `heading` and the next heading (##) or end of body."""
    idx = body.find(heading)
    if idx < 0:
        return None
    rest = body[idx + len(heading):]
    # Find next ## heading
    next_idx = rest.find("\n## ")
    if next_idx >= 0:
        return rest[:next_idx].strip()
    return rest.strip()


def _extract_commit_prefix(body: str) -> str:
    """Extract commit prefix from 'Commit: <sha>' line."""
    for line in body.splitlines():
        m = _COMMIT_PREFIX_RE.match(line.strip())
        if m:
            return m.group(1).strip().strip("`")
    return ""


# ------------------------------------------------------------------
# Build Result parser
# ------------------------------------------------------------------

def _parse_build_table(text: str) -> list[dict[str, str]]:
    """Parse the target/status/version/took table from Build Result."""
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        m = _TABLE_ROW_RE.match(line.strip())
        if m:
            rows.append({
                "target": m.group(1).strip(),
                "status": m.group(2).strip(),
                "version": m.group(3).strip(),
                "took": m.group(4).strip(),
            })
    return rows


def _parse_build_images(text: str) -> dict[str, str]:
    """Parse the Images section: **target** ```image_ref```."""
    images: dict[str, str] = {}
    # Match **target**\n```\nimage_ref\n```
    for m in re.finditer(
        r"\*\*(.+?)\*\*\s*\n```\n([\s\S]*?)\n```", text
    ):
        target = m.group(1).strip()
        image_ref = m.group(2).strip()
        if target and image_ref:
            images[target] = image_ref
    return images


def parse_build_result(
    body: str,
) -> ReviewBuild | None:
    """Parse a single Build Result section and return merged ReviewBuild.

    Returns None on any malformed input.
    """
    if not _has_marker(body):
        return None
    if BUILD_HEADING not in body:
        return None

    section = _find_section(body, BUILD_HEADING)
    if section is None:
        return None

    # Extract commit prefix
    commit_prefix = _extract_commit_prefix(section)
    if not commit_prefix:
        return None

    # Parse table
    table_rows = _parse_build_table(section)
    if not table_rows:
        return None

    # Parse images section
    images = _parse_build_images(section)

    # Merge all rows into a single ReviewBuild per target
    # We collect per-target info and return list of ReviewBuild
    builds: list[ReviewBuild] = []
    seen_targets: set[str] = set()

    for row in table_rows:
        target = row["target"]
        if not target:
            continue
        # Skip duplicate targets
        if target in seen_targets:
            return None  # duplicate => fail
        seen_targets.add(target)

        status_raw = row["status"]
        # Success indicators
        is_success = (
            ":white_check_mark:" in status_raw
            or status_raw.lower() == "success"
        )
        is_failed = (
            ":x:" in status_raw
            or status_raw.lower() in ("failed", "failure")
        )
        is_killed = (
            status_raw.lower() in ("killed",)
        )

        version_raw = row["version"].strip("`")
        version = row["version"]

        # Get image ref from Images section
        image_ref = images.get(target, "")
        if not image_ref and is_success:
            return None  # success without image => fail

        image_tag = ""
        if image_ref:
            image_tag = image_ref.split("/")[-1] if "/" in image_ref else image_ref

        build = ReviewBuild(
            target=target,
            driver_path=target if "/" in target else "",
            variant=_normalize_variant(
                next(
                    (r["version"] for r in table_rows if r["target"] == target),
                    "",
                )
            ),
            success=is_success and not is_failed and not is_killed,
            version=version,
            image_tag=image_tag,
            image_ref=image_ref,
            took=row.get("took", ""),
        )
        builds.append(build)

    return builds[0] if len(builds) == 1 else None


def parse_all_build_results(
    body: str,
) -> list[ReviewBuild]:
    """Parse all build rows from a Build Result comment.

    Returns a list of ReviewBuild (one per target).
    Returns empty list on malformed input (caller decides fail closed).
    """
    if not _has_marker(body) or BUILD_HEADING not in body:
        return []

    section = _find_section(body, BUILD_HEADING)
    if section is None:
        return []

    commit_prefix = _extract_commit_prefix(section)
    if not commit_prefix:
        return []

    table_rows = _parse_build_table(section)
    if not table_rows:
        return []

    images = _parse_build_images(section)

    builds: list[ReviewBuild] = []
    seen_targets: set[str] = set()

    for row in table_rows:
        target = row["target"]
        if not target:
            continue
        if target in seen_targets:
            return []  # duplicate => fail closed
        seen_targets.add(target)

        status_raw = row["status"]
        is_success = (
            ":white_check_mark:" in status_raw
            or status_raw.lower() == "success"
        )
        is_failed = (
            ":x:" in status_raw
            or status_raw.lower() in ("failed", "failure")
        )
        is_killed = status_raw.lower() in ("killed",)

        version = row["version"]
        image_ref = images.get(target, "")
        image_tag = image_ref.split("/")[-1] if "/" in image_ref and image_ref else ""

        build = ReviewBuild(
            target=target,
            driver_path=target if "/" in target else "",
            variant=_normalize_variant(version.strip("`")),
            success=is_success and not is_failed and not is_killed,
            version=version.strip("`"),
            image_tag=image_tag,
            image_ref=image_ref,
            took=row.get("took", ""),
        )
        builds.append(build)

    return builds


# ------------------------------------------------------------------
# Test Results parser
# ------------------------------------------------------------------

def parse_test_results(
    body: str,
) -> tuple[int, int, bool]:
    """Parse Test Results comment.

    Returns (passed, failed, skipped).
    Returns (0, 0, True) if no test suite ran.
    """
    if not _has_marker(body) or TEST_HEADING not in body:
        return (0, 0, False)

    section = _find_section(body, TEST_HEADING)
    if section is None:
        return (0, 0, False)

    # Check for skip-tests
    if "skip-tests" in section.lower() or "skipped" in section.lower():
        # Check if there are actual results
        passed = 0
        failed = 0
        for line in section.splitlines():
            if "passed" in line.lower() and ":" in line:
                try:
                    val = line.split(":")[-1].strip()
                    passed = int(val)
                except (ValueError, IndexError):
                    pass
            if "failed" in line.lower() and ":" in line:
                try:
                    val = line.split(":")[-1].strip()
                    failed = int(val)
                except (ValueError, IndexError):
                    pass
        if passed == 0 and failed == 0:
            return (0, 0, True)
        return (passed, failed, False)

    # Parse table rows
    passed = 0
    failed = 0
    for line in section.splitlines():
        m = _TABLE_ROW_RE.match(line.strip())
        if m:
            result = m.group(2).strip().lower()
            if ":white_check_mark:" in result or result == "pass":
                passed += 1
            elif ":x:" in result or result in ("fail", "failure"):
                failed += 1

    return (passed, failed, False)


# ------------------------------------------------------------------
# Code Review parser
# ------------------------------------------------------------------

def parse_code_review(
    body: str,
) -> str | None:
    """Parse Code Review comment body.

    Returns the review text or None if not found/malformed.
    """
    if not _has_marker(body) or CODE_REVIEW_HEADING not in body:
        return None

    section = _find_section(body, CODE_REVIEW_HEADING)
    if section is None:
        return None

    # Remove footer
    section = re.sub(r"<sub>.*?</sub>", "", section, flags=re.DOTALL)
    section = section.strip()

    if not section:
        return None

    return section


# ------------------------------------------------------------------
# Full evidence extraction from PR comments
# ------------------------------------------------------------------

def extract_review_evidence(
    comments: list[dict],
    trusted_author_id: str,
    trusted_author_login: str = "",
) -> ReviewCommentEvidence | None:
    """Extract the latest complete, unambiguous review evidence from PR comments.

    Arguments:
        comments: List of PR comment dicts with keys: id, user, body,
                  created_at, updated_at.
        trusted_author_id: The GitHub user ID that must match comment.user.id.
        trusted_author_login: Optional login for additional verification.

    Returns:
        ReviewCommentEvidence or None if no complete evidence found.
    """
    # Filter comments by trusted author
    trusted_comments: list[dict] = []
    for c in comments:
        user = c.get("user", {})
        if not isinstance(user, dict):
            continue
        cid = str(user.get("id", ""))
        if cid != trusted_author_id:
            continue
        login = str(user.get("login", ""))
        if trusted_author_login and login != trusted_author_login:
            continue
        trusted_comments.append(c)

    if not trusted_comments:
        return None

    # Sort by created_at for stable ordering
    trusted_comments.sort(
        key=lambda c: str(c.get("created_at", "")),
    )

    # Classify comments
    build_comments: list[dict] = []
    test_comments: list[dict] = []
    code_review_comments: list[dict] = []

    for c in trusted_comments:
        body = str(c.get("body", "") or "")
        if BUILD_HEADING in body:
            build_comments.append(c)
        elif TEST_HEADING in body:
            test_comments.append(c)
        elif CODE_REVIEW_HEADING in body:
            code_review_comments.append(c)

    if not build_comments:
        return None

    # Find the latest build comment with valid content
    latest_build: dict | None = None
    latest_build_idx = -1

    for idx, c in enumerate(build_comments):
        body = str(c.get("body", "") or "")
        builds = parse_all_build_results(body)
        if not builds:
            continue
        # Check that at least one build succeeded
        if not any(b.success for b in builds):
            continue
        latest_build = c
        latest_build_idx = idx

    if latest_build is None:
        return None

    build_body = str(latest_build.get("body", "") or "")
    build_commit = _extract_commit_prefix(build_body)
    build_created = str(latest_build.get("created_at", ""))
    build_updated = str(latest_build.get("updated_at", ""))
    build_builds = parse_all_build_results(build_body)

    if not build_builds:
        return None

    # Check all builds succeeded
    for b in build_builds:
        if not b.success:
            return None

    # Find test results (after build)
    test_evidence = (0, 0, False)
    test_comment_id = 0
    test_created = ""
    for c in test_comments:
        c_created = str(c.get("created_at", ""))
        if c_created >= build_created:
            test_evidence = parse_test_results(str(c.get("body", "") or ""))
            test_comment_id = int(c.get("id", 0) or 0)
            test_created = c_created

    # Find code review (after build, unambiguous)
    code_review_text = ""
    code_review_comment_id = 0
    code_review_created = ""

    # Filter code reviews that come after the build
    candidate_reviews = [
        c for c in code_review_comments
        if str(c.get("created_at", "")) >= build_updated
    ]

    if candidate_reviews:
        # Check for overlapping jobs: if there's another build comment between
        # this build and the code review, correlation is ambiguous
        for cr in candidate_reviews:
            cr_created = str(cr.get("created_at", ""))
            # Check for intervening build comments
            intervening = False
            for bc in build_comments:
                bc_created = str(bc.get("created_at", ""))
                if bc_created > build_updated and bc_created < cr_created:
                    intervening = True
                    break
            if not intervening:
                review_body = str(cr.get("body", "") or "")
                review_text = parse_code_review(review_body)
                if review_text:
                    code_review_text = review_text
                    code_review_comment_id = int(cr.get("id", 0) or 0)
                    code_review_created = cr_created
                    break

    if not code_review_text:
        return None

    # Gather successful build info
    successful_builds = [b for b in build_builds if b.success]
    if not successful_builds:
        return None

    evidence = ReviewCommentEvidence(
        commit_prefix=build_commit,
        review_author_id=trusted_author_id,
        review_author_login=str(
            latest_build.get("user", {}).get("login", "")
        ),
        builds=successful_builds,
        build_comment_id=int(latest_build.get("id", 0) or 0),
        build_comment_created_at=build_created,
        build_comment_updated_at=build_updated,
        test_comment_id=test_comment_id,
        test_comment_created_at=test_created,
        test_passed=test_evidence[0],
        test_failed=test_evidence[1],
        test_skipped=test_evidence[2],
        code_review_comment_id=code_review_comment_id,
        code_review_created_at=code_review_created,
        code_review_text=code_review_text,
    )

    return evidence
