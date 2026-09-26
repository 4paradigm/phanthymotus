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

@dataclass
class ReviewJobAnchor:
    comment_id: int
    commit_prefix: str
    created_at: str
    updated_at: str
    state: str
    body: str = field(default="", repr=False)


def _extract_review_job_commit_prefix(body: str) -> str:
    """Extract commit prefix from 'Commit: <sha>' or '| Commit | <sha> |'."""
    for line in body.splitlines():
        m = _COMMIT_PREFIX_RE.match(line.strip())
        if m:
            return m.group(1).strip().strip("`")
        m2 = _TABLE_COMMIT_PREFIX_RE.match(line.strip())
        if m2:
            return m2.group(1).strip().strip("`")
    return ""


def extract_latest_review_job_anchor(
    comments: list[dict],
    trusted_author_id: str,
    trusted_author_login: str = "",
) -> ReviewJobAnchor | None:
    """Scan ALL trusted comments with pr-review-agent marker.

    Returns the latest ReviewJobAnchor by (created_at, comment_id) among
    comments that contain a commit prefix.  Supports both
    `Commit: sha` and `| Commit | sha |` formats.

    LATEST RUN ALWAYS WINS.  Never falls back to an older completed run.
    If the latest run is still Queued/Building/Generating review,
    returns anchor with state="reviewing".
    """
    # Filter trusted comments
    trusted: list[dict] = []
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
        trusted.append(c)

    if not trusted:
        return None

    # Build anchors from ALL trusted comments that have a marker and commit prefix.
    # Progress comments (Queued/Building) that lack a Build Result table still form anchors.
    # Test Results and Code Review are FOLLOW-UP evidence, NOT run anchors.
    anchors: list[ReviewJobAnchor] = []
    for c in trusted:
        body = str(c.get("body", "") or "")
        if not _has_marker(body):
            continue
        commit_prefix = _extract_review_job_commit_prefix(body)
        if not commit_prefix:
            continue
        # Exclude follow-up evidence comments from anchor candidates
        if TEST_HEADING in body or CODE_REVIEW_HEADING in body:
            continue
        # Determine anchor state based on comment content
        anchor_state = "reviewing"
        if BUILD_HEADING in body:
            # This is a Build Result comment — parse rows for state determination
            rows = _parse_build_table(body)
            images = _parse_build_images(body)
            if rows:
                all_success = True
                has_exact_image = True
                for row in rows:
                    status_raw = row.get("status", "")
                    is_success = (
                        ":white_check_mark:" in status_raw
                        or status_raw.lower() == "success"
                    )
                    if not is_success:
                        all_success = False
                    target = row.get("target", "")
                    if is_success and not images.get(target):
                        has_exact_image = False
                if all_success and has_exact_image:
                    anchor_state = "build-succeeded"
                elif not all_success:
                    anchor_state = "terminal"
            # If rows is empty for a BUILD_HEADING comment, state stays "reviewing"
        # If BUILD_HEADING not in body (progress comment), state stays "reviewing"
        anchors.append(ReviewJobAnchor(
            comment_id=int(c.get("id", 0) or 0),
            commit_prefix=commit_prefix,
            created_at=str(c.get("created_at", "")),
            updated_at=str(c.get("updated_at", "")),
            state=anchor_state,
            body=body,
        ))

    if not anchors:
        return None

    anchors.sort(key=lambda a: (a.created_at, a.comment_id))
    return anchors[-1]


_COMMIT_PREFIX_RE = re.compile(r"^Commit:\s*(.+)$")
_TABLE_ROW_RE = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$")
_TEST_TABLE_ROW_RE = re.compile(
    r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$"
)
_IMAGE_SECTION_RE = re.compile(
    r"\*\*(.+?)\*\*\s*\n\s*```(?:\n|(.*?))\n\s*(.+?)\s*```",
    re.DOTALL,
)
_IMAGE_REF_RE = re.compile(
    r"^([^\s:/]+(?:/[^\s:/]+)+)/([^:\s]+(?::(.+))?)$"
)
_VERSION_RE = re.compile(r"^`(.+?)`$")
_TABLE_COMMIT_PREFIX_RE = re.compile(r'^\|\s*(?:Commit|commit)\s*\|\s*(.+?)\s*\|')
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
    test_comment_updated_at: str = ""
    test_passed: int = 0
    test_failed: int = 0
    test_skipped: bool = False
    code_review_comment_id: int = 0
    code_review_created_at: str = ""
    code_review_comment_updated_at: str = ""
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


def _normalize_build_label(raw_label: str) -> dict | None:
    """Normalize a raw Review Agent build display label into canonical fields.

    Raw display labels and their canonical fields:
        core                          -> target="core",          driver_path="",  variant=""
        perception (jetson-jp5.11)   -> target="perception",    driver_path="",  variant="5.11"
        perception (jetson-jp6.1)    -> target="perception",    driver_path="",  variant="6.1"
        actucore (jetson-jp5.11)     -> target="actucore",      driver_path="",  variant="5.11"
        actucore (jetson-jp6.1)      -> target="actucore",      driver_path="",  variant="6.1"
        unitree/g1                   -> target="driver",         driver_path="unitree/g1", variant=""

    Returns None for unknown or malformed labels (fail closed).
    """
    raw = raw_label.strip()
    if not raw:
        return None

    # Driver: path-shaped label (contains '/')
    if "/" in raw:
        return {"target": "driver", "driver_path": raw, "variant": ""}

    # Known targets with optional variant parenthesised suffix
    known_targets = ("core", "perception", "actucore")
    for target in known_targets:
        prefix = target + " ("
        if raw.startswith(prefix):
            suffix = raw[len(prefix):]
            if suffix.endswith(")"):
                version_in_parens = suffix[:-1]
                variant = _normalize_variant(version_in_parens)
                if not variant:
                    return None
                return {"target": target, "driver_path": "", "variant": variant}

    # Plain known target without variant suffix
    if raw in known_targets:
        return {"target": raw, "driver_path": "", "variant": ""}

    # Unknown label => fail closed
    return None


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
    """Parse the target/status/version/took table from Build Result.

    Skips markdown table header rows (e.g. ``| Target | Status | Version |
    Took |``) and separator rows (e.g. ``| --- | --- | --- | --- |``).
    """

    def _is_header_row(groups: re.Match) -> bool:
        """Return True if the row is a table header (all cells alphabetic)."""
        for g in (groups.group(1), groups.group(2), groups.group(3), groups.group(4)):
            cell = g.strip()
            if not cell or not cell.isalpha():
                return False
        return True

    def _is_separator(target: str) -> bool:
        """Return True if *target* looks like a markdown table separator."""
        return bool(re.fullmatch(r"[- :	]+", target)) and len(target) > 2

    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        m = _TABLE_ROW_RE.match(line.strip())
        if m:
            # Skip header rows (all cells purely alphabetic)
            if _is_header_row(m):
                continue
            target = m.group(1).strip()
            # Skip markdown table separator rows
            if _is_separator(target):
                continue
            rows.append({
                "target": target,
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
            if target in images:
                # Duplicate target => conflicting images => fail closed
                return {}
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
    seen_keys: set[tuple[str, str, str]] = set()

    for row in table_rows:
        raw_label = row["target"]
        normalized = _normalize_build_label(raw_label)
        if normalized is None:
            return None  # unknown/malformed => fail closed

        target = normalized["target"]
        driver_path = normalized["driver_path"]
        variant = normalized["variant"]

        dup_key = (target, driver_path, variant)
        if dup_key in seen_keys:
            return None  # duplicate => fail
        seen_keys.add(dup_key)

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

        # Use raw display label for image section lookup
        image_ref = images.get(raw_label, "")
        if not image_ref and is_success:
            return None  # success without image => fail

        image_tag = image_ref  # full mutable ref: registry.example/path/image:tag

        build = ReviewBuild(
            target=target,
            driver_path=driver_path,
            variant=variant,
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
    seen_keys: set[tuple[str, str, str]] = set()

    for row in table_rows:
        raw_label = row["target"]
        normalized = _normalize_build_label(raw_label)
        if normalized is None:
            return []  # unknown/malformed => fail closed

        target = normalized["target"]
        driver_path = normalized["driver_path"]
        variant = normalized["variant"]

        dup_key = (target, driver_path, variant)
        if dup_key in seen_keys:
            return []  # duplicate => fail closed
        seen_keys.add(dup_key)

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

        # Use raw display label for image section lookup (labels must match exactly)
        image_ref = images.get(raw_label, "")

        # Success build without an image reference => fail closed
        if is_success and not is_failed and not is_killed and not image_ref:
            return []

        # Duplicate target in images section => fail closed
        if is_success and image_ref and images.get(raw_label, "") and images.get(raw_label, "") != image_ref:
            return []

        image_tag = image_ref  # full mutable ref: registry.example/path/image:tag

        build = ReviewBuild(
            target=target,
            driver_path=driver_path,
            variant=variant,
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

    A real Test Results table takes precedence over any informational
    ``skip-tests`` text in the comment footer.

    Passed and failed counts are accumulated independently across every
    concrete suite row because a failed suite can still contain many
    passing tests.
    """
    if not _has_marker(body) or TEST_HEADING not in body:
        return (0, 0, False)

    section = _find_section(body, TEST_HEADING)
    if section is None:
        return (0, 0, False)

    passed = 0
    failed = 0
    saw_result_row = False

    for line in section.splitlines():
        m = _TEST_TABLE_ROW_RE.match(line.strip())
        if not m:
            continue

        suite = m.group(1).strip()

        if not suite:
            continue

        if suite.lower() == "suite":
            continue

        if re.fullmatch(r"[- :\t]+", suite) and len(suite) > 2:
            continue

        try:
            row_passed = int(m.group(3).strip())
            row_failed = int(m.group(4).strip())
        except (ValueError, IndexError):
            continue

        saw_result_row = True
        passed += row_passed
        failed += row_failed

    if saw_result_row:
        return (passed, failed, False)

    # Review Agent normally emits no Test Results comment at all when
    # nothing ran. Preserve compatibility for an explicit skipped-only
    # Test Results body, but never let the standard footer override a
    # real table above.
    lowered = section.lower()
    if (
        "tests skipped" in lowered
        or "test skipped" in lowered
        or "not run" in lowered
        or "skip-tests" in lowered
    ):
        return (0, 0, True)

    return (0, 0, False)


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

    # Treat lines that are only dashes/spaces (e.g. "---") as empty
    cleaned = re.sub(r"[-*_\s]+", "", section)
    if not cleaned:
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

    Latest review run wins: selects newest run by (created_at, comment_id).
    Does NOT fall back to older runs.
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

    # Classify comments into build, test, code review
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

    # Use canonical latest anchor selection — do not duplicate logic.
    anchor = extract_latest_review_job_anchor(
        trusted_comments,
        trusted_author_id,
        trusted_author_login,
    )
    if anchor is None:
        return None

    # If latest run is terminal/fail, never fall back
    if anchor.state == "terminal":
        return None

    # If latest run is still reviewing (Queued/Building/incomplete),
    # complete evidence is not yet available.
    if anchor.state == "reviewing":
        return None

    # Parse builds from the selected anchor
    build_builds = parse_all_build_results(anchor.body)
    if not build_builds:
        return None

    # Build commit prefix must match fresh HEAD (checked later by caller)
    build_commit = anchor.commit_prefix

    # Select Test Results: trusted test comments created after selected Build
    # Lower bound: selected Build Result anchor.updated_at
    test_candidates: list[dict] = []
    for c in test_comments:
        c_created = str(c.get("created_at", ""))
        if c_created < anchor.updated_at:
            continue
        c_body = str(c.get("body", "") or "")
        test_commit = _extract_review_job_commit_prefix(c_body)
        if test_commit and test_commit != build_commit:
            return None  # commit mismatch => fail closed
        if not test_commit:
            return None  # commit missing => fail closed
        test_candidates.append(c)

    test_comment_id = 0
    test_created = ""
    test_updated = ""
    if len(test_candidates) == 1:
        tc = test_candidates[0]
        test_comment_id = int(tc.get("id", 0) or 0)
        test_created = str(tc.get("created_at", ""))
        test_updated = str(tc.get("updated_at", ""))
    elif len(test_candidates) > 1:
        # Ambiguous => fail closed
        return None

    # Select Code Review: trusted CR comments after the selected run
    # Lower bound: latest of selected Build Result updated_at, selected Test updated_at
    lb = anchor.updated_at
    if test_comment_id > 0 and test_updated:
        lb = test_updated

    cr_candidates: list[dict] = []
    for c in code_review_comments:
        c_created = str(c.get("created_at", ""))
        if c_created < lb:
            continue
        cr_candidates.append(c)

    if len(cr_candidates) != 1:
        return None  # 0 or >1 => fail closed

    cr_body = str(cr_candidates[0].get("body", "") or "")
    cr_text = parse_code_review(cr_body)
    if cr_text is None:
        return None

    cr_comment = cr_candidates[0]
    code_review_comment_id = int(cr_comment.get("id", 0) or 0)
    code_review_created = str(cr_comment.get("created_at", ""))
    code_review_updated = str(cr_comment.get("updated_at", ""))

    # Test results from the selected test comment
    test_evidence = (0, 0, False)
    if test_comment_id > 0:
        test_body = str(test_candidates[0].get("body", "") or "")
        test_evidence = parse_test_results(test_body)

    # Gather successful build info
    successful_builds = [b for b in build_builds if b.success]
    if not successful_builds:
        return None

    # Derive review_author_login from the selected trusted comment's user.login
    selected_author_login = ""
    for c in trusted_comments:
        if int(c.get("id", 0) or 0) == anchor.comment_id:
            user = c.get("user", {})
            if isinstance(user, dict):
                selected_author_login = str(user.get("login", ""))
            break
    if not selected_author_login:
        selected_author_login = trusted_author_login  # fallback

    evidence = ReviewCommentEvidence(
        commit_prefix=build_commit,
        review_author_id=trusted_author_id,
        review_author_login=selected_author_login,
        builds=successful_builds,
        build_comment_id=anchor.comment_id,
        build_comment_created_at=anchor.created_at,
        build_comment_updated_at=anchor.updated_at,
        test_comment_id=test_comment_id,
        test_comment_created_at=test_created,
        test_comment_updated_at=test_updated,
        test_passed=test_evidence[0],
        test_failed=test_evidence[1],
        test_skipped=test_evidence[2],
        code_review_comment_id=code_review_comment_id,
        code_review_created_at=code_review_created,
        code_review_comment_updated_at=code_review_updated,
        code_review_text=cr_text,
    )

    return evidence