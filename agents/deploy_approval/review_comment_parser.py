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
from datetime import datetime

from .image_ref import validate_image_ref

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

PR_REVIEW_MARKER = "<!-- pr-review-agent -->"
BUILD_HEADING = "## PR Review Agent — Build Result"
TEST_HEADING = "## PR Review Agent — Test Results"
CODE_REVIEW_HEADING = "## PR Review Agent — Code Review"
ERROR_HEADING = "## PR Review Agent — Error"
# Published by agents/pr_review/models.py DEFAULT_JP_VERSION. The isolated
# Controller image does not ship that package; keep the protocol default here.
_DEFAULT_REQUEST_VARIANT = "5.11"

@dataclass
class ReviewJobAnchor:
    comment_id: int
    commit_prefix: str
    created_at: str
    updated_at: str
    state: str
    body: str = field(default="", repr=False)


def _extract_review_job_commit_prefix(body: str) -> str:
    """Read one explicit Commit; malformed/duplicate metadata is not absence."""
    prefixes = []
    for line in body.splitlines():
        m = _COMMIT_PREFIX_RE.match(line.strip())
        m2 = _TABLE_COMMIT_PREFIX_RE.match(line.strip())
        if m or m2:
            prefix = (m or m2).group(1).strip().strip("`")
            if not _SHORT_SHA_RE.fullmatch(prefix):
                return "!ambiguous"
            prefixes.append(prefix)
        elif re.match(r"^\|\s*Commit\s*\|", line.strip(), re.I):
            return "!ambiguous"
    if not prefixes:
        return ""
    return prefixes[0] if len(prefixes) == 1 else "!ambiguous"


def _is_review_job_anchor_body(body: str) -> bool:
    if not _has_marker(body):
        return False
    if BUILD_HEADING in body:
        return True
    if TEST_HEADING in body or CODE_REVIEW_HEADING in body:
        return False
    return bool(_extract_review_job_commit_prefix(body)
                or re.search(r"(?m)^## PR Review Agent\s*$", body))


def _is_terminal_review_error_body(body: str) -> bool:
    """Recognize the Review Agent's explicit terminal error protocol."""
    return (
        _has_marker(body)
        and body.count(ERROR_HEADING) == 1
        and not any(heading in body for heading in (BUILD_HEADING, TEST_HEADING, CODE_REVIEW_HEADING))
        and bool(_SHORT_SHA_RE.fullmatch(_extract_review_job_commit_prefix(body)))
        and "Push a fix and comment `/request_bot_review` again to retrigger." in body
    )


def extract_latest_review_job_anchor(
    comments: list[dict],
    trusted_author_id: str,
    trusted_author_login: str = "",
) -> ReviewJobAnchor | None:
    """Scan ALL trusted comments with pr-review-agent marker.

    Returns the latest ReviewJobAnchor by (created_at, comment_id).
    Malformed newer anchors still supersede older evidence. Supports both
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

    # Build anchors from ALL trusted run comments, including malformed new runs.
    # Progress comments (Queued/Building) that lack a Build Result table still form anchors.
    # Test Results and Code Review are FOLLOW-UP evidence, NOT run anchors.
    anchors: list[ReviewJobAnchor] = []
    for c in trusted:
        body = str(c.get("body", "") or "")
        if not _is_review_job_anchor_body(body):
            continue
        commit_prefix = _extract_review_job_commit_prefix(body)
        if _comment_position(c) is None:
            return None  # Invalid run provenance must never fall back to an older run.
        # Determine anchor state based on comment content
        anchor_state = "reviewing"
        if _is_terminal_review_error_body(body):
            anchor_state = "terminal"
        elif BUILD_HEADING in body:
            # This is a Build Result comment — parse rows for state determination
            rows = _parse_build_table(body)
            images = _parse_build_images(body)
            if rows:
                all_success = True
                has_exact_image = True
                for row in rows:
                    status_raw = row.get("status", "")
                    is_success = _success_status(status_raw)
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

    anchors.sort(key=lambda a: (datetime.fromisoformat(a.created_at.replace("Z", "+00:00")), a.comment_id))
    return anchors[-1]


_COMMIT_PREFIX_RE = re.compile(r"^Commit:\s*(.*?)\s*$", re.I)
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
_TABLE_COMMIT_PREFIX_RE = re.compile(r'^\|\s*Commit\s*\|\s*(.*?)\s*\|$', re.I)
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
    """Review evidence for one HEAD; free-text code review needs human review.

    Comment completeness is not a machine-readable code-quality approval.
    The developer must review the findings before requesting deployment,
    followed by the existing machine owner/collaborator approval gate.
    """
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
    # Parsed evidence always supplies these explicitly. Defaults retain the
    # internal API for callers constructing already validated historical data.
    test_status: str = "VALID"
    skip_tests_requested: bool = False
    run_binding_valid: bool = True
    request_comment: dict | None = field(default=None, repr=False)
    code_review_binding: str = ""

    @property
    def test_results_block_deployment(self) -> bool:
        """Reject invalid review evidence, not reported project test failures.

        Valid failure counts remain evidence for manual review. Deployment
        success is gated separately by the Controller's Automated Cases.
        """
        return (
            not self.run_binding_valid
            or self.test_status not in ("VALID", "SKIPPED")
            or (self.test_status == "SKIPPED" and not self.skip_tests_requested)
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
                   for count in (self.test_passed, self.test_failed))
            or not isinstance(self.test_skipped, bool)
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


def _comment_position(comment: dict) -> tuple[datetime, int] | None:
    cid = comment.get("id")
    if isinstance(cid, bool) or not str(cid).isdigit() or int(cid) <= 0:
        return None
    try:
        created = datetime.fromisoformat(str(comment.get("created_at", "")).replace("Z", "+00:00"))
        updated = datetime.fromisoformat(str(comment.get("updated_at", "")).replace("Z", "+00:00"))
        if created.tzinfo is None or updated.tzinfo is None or updated < created:
            return None
    except ValueError:
        return None
    return created, int(cid)


def _parse_review_request(body: str) -> dict | None:
    """Read the actual command, never PR prose or the PR body.

    Expanded matrix tokens express operator intent. In particular jp511/jp61
    and actucore are NOT evidence that the unchanged upstream supports them.
    """
    lines = [line.strip().lstrip(">*- \t") for line in body.splitlines()]
    commands = [line for line in lines if re.match(r"^/request_bot_review(?:\s|$)", line, re.I)]
    if len(commands) != 1:
        return None
    targets, variants = [], []
    skip_tests = False
    unsupported = False
    for raw in commands[0].split()[1:]:
        token = raw.strip("`,").lower()
        if token in ("force", "--force", "-f"):
            continue
        if token == "skip-tests":
            skip_tests = True
        elif token in ("core", "perception", "actucore"):
            if token in targets:
                return None
            targets.append(token)
        elif token in ("jp511", "jp61", "jp5.11", "jp6.1", "5.11", "6.1",
                        "jetson-jp5.11", "jetson-jp6.1", "jetson-5.11", "jetson-6.1"):
            variant = "5.11" if token in ("jp511", "jp5.11", "5.11", "jetson-jp5.11", "jetson-5.11") else "6.1"
            if variant in variants:
                return None
            variants.append(variant)
        elif re.fullmatch(r"[a-z0-9_.-]+(?:/[a-z0-9_.-]+)+", token) and all(
            part not in (".", "..") for part in token.split("/")
        ):
            if token in targets:
                return None
            targets.append(token)
        else:
            unsupported = True  # Unknown flags/versions must never silently disappear.
    if variants and not any(t in ("perception", "actucore") for t in targets):
        targets.append("perception")
    if any(t in ("perception", "actucore") for t in targets) and not variants:
        # Reuse the producer's actual default instead of inventing a platform.
        variants = [_DEFAULT_REQUEST_VARIANT]
    expected = set()
    for target in targets:
        if target in ("perception", "actucore"):
            expected.update((target, "", variant) for variant in variants)
        elif target == "core":
            expected.add(("core", "", ""))
        else:
            expected.add(("driver", target, ""))
    return {"expected": expected or None, "skip_tests": skip_tests, "valid": not unsupported}


def _same_commit(value: str, expected: str) -> bool:
    return bool(_SHORT_SHA_RE.fullmatch(value) and _SHORT_SHA_RE.fullmatch(expected)
                and (value.lower().startswith(expected.lower()) or expected.lower().startswith(value.lower())))


def _run_id(body: str) -> str:
    # Optional explicit Markdown metadata, not an invented GitHub API field.
    lines = re.findall(r"^[ \t]*(?:Review )?Run(?: ID)?:[ \t]*([^\r\n]*)$", body, re.M | re.I)
    if not lines:
        return ""
    value = lines[0].strip().strip("`")
    return value if len(lines) == 1 and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) else "!ambiguous"


def _success_status(status: str) -> bool:
    return bool(re.fullmatch(r"(?:success|:white_check_mark:(?:\s*success)?)", status.strip(), re.I))


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
    """Read the whole result, including tables after Markdown subheadings."""
    idx = body.find(heading)
    if idx < 0 or body.count(heading) != 1:
        return None
    rest = body[idx + len(heading):]
    if any(other in rest for other in (BUILD_HEADING, TEST_HEADING, CODE_REVIEW_HEADING)
           if other != heading):
        return None
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
        """Only the actual header is a header; alphabetic results are data."""
        return tuple(groups.group(i).strip().lower() for i in range(1, 5)) == (
            "target", "status", "version", "took",
        )

    def _is_separator(target: str) -> bool:
        """Return True if *target* looks like a markdown table separator."""
        return bool(re.fullmatch(r"[- :	]+", target)) and len(target) > 2

    rows: list[dict[str, str]] = []
    fenced = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = _TABLE_ROW_RE.match(line.strip())
        if m:
            # Skip the exact protocol header.
            if _is_header_row(m):
                continue
            target = m.group(1).strip()
            # Skip markdown table separator rows
            if all(_is_separator(m.group(i).strip()) for i in range(1, 5)):
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

    commit_prefix = _extract_review_job_commit_prefix(section)
    if not _SHORT_SHA_RE.fullmatch(commit_prefix):
        return []

    # Malformed rows must not vanish before the requested-set comparison. Ignore
    # fenced image/log payloads, which are not build-result table evidence.
    fenced = False
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if not stripped.startswith("|"):
            if "|" in stripped:
                return []
            continue
        if _TABLE_COMMIT_PREFIX_RE.match(stripped):
            continue
        if not stripped.endswith("|") or len(stripped[1:-1].split("|")) != 4:
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
        is_success = _success_status(status_raw)
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

        if is_success:
            try:
                validate_image_ref(image_ref)
            except ValueError:
                return []
            # Producer tags may explicitly attest a JetPack version. Never
            # accept a contradictory tag; older refs without this suffix retain
            # their trusted build-label binding instead of inventing metadata.
            image_variant = re.search(r"-jetson-jp(\d+\.\d+)$", image_ref)
            if image_variant and image_variant.group(1) != variant:
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

@dataclass
class TestResultEvidence:
    status: str
    passed: int = 0
    failed: int = 0

    def __iter__(self):
        # Preserve historical tuple unpacking without hiding the explicit state.
        yield self.passed
        yield self.failed
        yield self.status == "SKIPPED"


def parse_test_results(body: str) -> TestResultEvidence:
    if TEST_HEADING not in body:
        return TestResultEvidence("MISSING")
    if not _has_marker(body):
        return TestResultEvidence("MALFORMED")
    if _extract_review_job_commit_prefix(body) == "!ambiguous":
        return TestResultEvidence("MALFORMED")
    section = _find_section(body, TEST_HEADING)
    if section is None:
        return TestResultEvidence("MALFORMED")
    # A footer instruction is not a verdict. A table inside a footer is invalid
    # evidence, and must not disappear together with its reported failures.
    malformed = any("|" in footer for footer in re.findall(
        r"<sub>.*?</sub>", section, flags=re.DOTALL,
    ))
    section = re.sub(r"<sub>[^|]*?</sub>", "", section, flags=re.DOTALL)
    suites = set()
    passed = failed = 0
    header_seen = False
    separator_seen = False
    for line in section.splitlines():
        if not line.strip().startswith("|"):
            if "|" in line:
                malformed = True
            continue
        if not line.strip().endswith("|"):
            malformed = True
            continue
        cells = [cell.strip() for cell in line.strip()[1:-1].split("|")]
        if cells and cells[0].lower() == "suite":
            if (header_seen or len(cells) != 5 or [c.lower() for c in cells[2:4]] != ["passed", "failed"]
                    or cells[1].lower() not in ("result", "status")):
                malformed = True
            header_seen = True
            continue
        if cells and all(re.fullmatch(r"[- :]+", cell) for cell in cells):
            if separator_seen or suites or not header_seen:
                malformed = True
            separator_seen = len(cells) == 5
            if not separator_seen:
                malformed = True
            continue
        if len(cells) == 2 and cells[0].lower() == "commit" and _SHORT_SHA_RE.fullmatch(cells[1].strip("`")):
            continue
        if len(cells) != 5 or not cells[0]:
            malformed = True
            continue
        suite = cells[0].casefold()
        if suite in suites:
            malformed = True
        suites.add(suite)
        counts = []
        for raw in cells[2:4]:
            if not re.fullmatch(r"[0-9]{1,10}", raw) or int(raw) > 2147483647:
                malformed = True
                counts.append(None)
            else:
                counts.append(int(raw))
        if counts[0] is not None:
            passed += counts[0]
        if counts[1] is not None:
            failed += counts[1]
        status = cells[1].lower()
        is_failure = bool(re.fullmatch(r"(?::x:\s*)?(?:[0-9]+\s+)?(?:failed|failure)", status))
        is_pass = bool(re.fullmatch(r"(?::white_check_mark:\s*)?(?:[0-9]+\s+)?(?:passed|pass|success)", status))
        if not (is_failure or is_pass) or (is_failure and counts[1] == 0) or (is_pass and counts[1] not in (None, 0)):
            malformed = True
        if not cells[4] or passed > 2147483647 or failed > 2147483647:
            malformed = True
    if malformed or (suites and not (header_seen and separator_seen)):
        return TestResultEvidence("MALFORMED", passed, failed)
    if suites:
        return TestResultEvidence("VALID", passed, failed)
    prose = re.sub(r"(?m)^Commit:.*$", "", section).strip()
    prose = re.sub(r"(?im)^(?:Review )?Run(?: ID)?:.*$", "", prose).strip()
    prose = re.sub(r"(?m)^\s*---+\s*$", "", prose).strip()
    if re.fullmatch(r"(?:tests? skipped|tests? not run)(?:[.!])?", prose, re.I):
        return TestResultEvidence("SKIPPED")
    return TestResultEvidence("MALFORMED")


# ------------------------------------------------------------------
# Code Review parser
# ------------------------------------------------------------------

def parse_code_review(
    body: str,
) -> str | None:
    """Parse Code Review comment body.

    Returns review text for manual inspection, not an approval verdict, or
    None if not found/malformed. Free-text keywords never establish approval.
    """
    if not _has_marker(body) or body.count(CODE_REVIEW_HEADING) != 1:
        return None

    # format_review embeds arbitrary Markdown, including ## sections, in this
    # one comment. Those headings belong to the review, not to another result.
    section = body.split(CODE_REVIEW_HEADING, 1)[1].strip()

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
    *,
    head_sha: str = "",
) -> ReviewCommentEvidence | None:
    """Extract the latest complete, unambiguous review evidence from PR comments.

    Latest review run wins: selects newest run by (created_at, comment_id).
    Does NOT fall back to older runs.
    """
    if not trusted_author_id or not trusted_author_login:
        return None
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
    expected_commit = head_sha or build_commit
    if not _same_commit(build_commit, expected_commit):
        return None
    anchor_comment = next(c for c in trusted_comments if int(c.get("id", 0) or 0) == anchor.comment_id)
    anchor_position = _comment_position(anchor_comment)
    if anchor_position is None:
        return None

    other_anchors = [c for c in trusted_comments if c is not anchor_comment
                     and _is_review_job_anchor_body(str(c.get("body", "")))]
    prior_positions = [_comment_position(c) for c in other_anchors]
    if any(position is None or position >= anchor_position for position in prior_positions):
        return None
    previous_position = max(prior_positions) if prior_positions else None
    request_candidates = []
    request_positions = []
    run_valid = True
    for c in comments:
        body = c.get("body", "")
        if not isinstance(body, str) or not re.search(r"(?im)^\s*[>*-]*\s*/request_bot_review(?:\s|$)", body):
            continue
        position = _comment_position(c)
        if position is None:
            run_valid = False
            continue
        request_positions.append(position)
        if previous_position is not None and position <= previous_position:
            continue
        if position >= anchor_position:
            run_valid = False  # A newer request supersedes this chain; no fallback.
            continue
        request_candidates.append(c)
    request = request_candidates[0] if len(request_candidates) == 1 else None
    command = _parse_review_request(str(request.get("body", ""))) if request else None
    if (request is None or command is None or not command["valid"]
            or int(request.get("id", 0)) >= anchor.comment_id
            or request.get("updated_at") != request.get("created_at")):
        run_valid = False
    request_commit = _extract_review_job_commit_prefix(str(request.get("body", ""))) if request else ""
    if request_commit and not _same_commit(request_commit, expected_commit):
        run_valid = False
    user = request.get("user") if request else None
    if (not isinstance(user, dict) or isinstance(user.get("id"), bool)
            or not str(user.get("id", "")).isdigit() or int(user.get("id", 0)) <= 0
            or not isinstance(user.get("login"), str) or not user.get("login")):
        run_valid = False
    keys = {(b.target, b.driver_path, b.variant) for b in build_builds}
    if (not all(b.success for b in build_builds)
            or len(keys) != len(build_builds)
            or any(b.target in ("perception", "actucore") and b.variant not in ("5.11", "6.1")
                   for b in build_builds)
            or any(b.target == "core" and b.variant for b in build_builds)
            or (command and command["expected"] is not None and keys != command["expected"])):
        run_valid = False
    skip_requested = bool(command and command["skip_tests"])
    anchor_run = _run_id(anchor.body)
    if anchor_run == "!ambiguous":
        run_valid = False

    # Select Test Results: trusted test comments created after selected Build
    # Lower bound: selected Build Result anchor.updated_at
    anchor_updated = datetime.fromisoformat(anchor.updated_at.replace("Z", "+00:00"))
    test_candidates: list[dict] = []
    for c in test_comments:
        position = _comment_position(c)
        if position is None:
            return None
        if position[0] < anchor_updated:
            if position > anchor_position:
                return None  # A result inside an unfinished run is ambiguous.
            continue
        c_body = str(c.get("body", "") or "")
        test_commit = _extract_review_job_commit_prefix(c_body)
        if test_commit and not _same_commit(test_commit, expected_commit):
            return None  # commit mismatch => fail closed
        if not test_commit:
            return None  # commit missing => fail closed
        if position[1] <= anchor.comment_id:
            return None
        explicit_run = _run_id(c_body)
        if explicit_run and (not anchor_run or explicit_run != anchor_run):
            return None
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
    lb = anchor_updated
    if test_comment_id > 0 and test_updated:
        lb = datetime.fromisoformat(test_updated.replace("Z", "+00:00"))

    cr_candidates: list[dict] = []
    for c in code_review_comments:
        position = _comment_position(c)
        if position is None:
            return None
        if position[0] < lb:
            if position > anchor_position:
                return None
            continue
        if position[1] <= max(anchor.comment_id, test_comment_id):
            return None
        body = str(c.get("body", ""))
        explicit_commit = _extract_review_job_commit_prefix(body)
        if explicit_commit and not _same_commit(explicit_commit, expected_commit):
            return None
        explicit_run = _run_id(body)
        if explicit_run and (not anchor_run or explicit_run != anchor_run):
            return None
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

    # Prior runs must be closed before this run starts. Without this, a delayed
    # SHA-less review could belong to an older/overlapping worker.
    for older in other_anchors:
        old_builds = parse_all_build_results(str(older.get("body", "")))
        old_position = _comment_position(older)
        old_updated = datetime.fromisoformat(older["updated_at"].replace("Z", "+00:00"))
        # Each prior run needs its own closure before the next request/anchor;
        # one delayed review cannot close several overlapping runs.
        boundary = min(position for position in
                       [anchor_position, *prior_positions, *request_positions]
                       if position > old_position)
        # The Review Agent's explicit error comment closes that run even
        # though failed runs have neither a Build Result nor a Code Review.
        old_body = str(older.get("body", ""))
        if (_is_terminal_review_error_body(old_body)
                and old_updated < boundary[0]
                and old_position < boundary):
            continue
        # A failed build is a real terminal run; upstream emits no Code Review
        # for it. It must have finished before the new request, not overlap it.
        old_rows = _parse_build_table(str(older.get("body", "")))
        terminal_status = lambda status: bool(re.fullmatch(
            r"(?:(?::x:\s*)?(?:failed|failure)|(?::hourglass:\s*)?killed)", status.strip(), re.I))
        if (old_builds and old_rows and any(terminal_status(row["status"]) for row in old_rows)
                and all(_success_status(row["status"]) or terminal_status(row["status"]) for row in old_rows)
                and old_updated < boundary[0]):
            continue
        closed = []
        for c in code_review_comments:
            position = _comment_position(c)
            if position is None or position <= old_position or position >= boundary:
                continue
            if (position[0] < old_updated or position[1] <= old_position[1]
                    or datetime.fromisoformat(c["updated_at"].replace("Z", "+00:00")) >= boundary[0]):
                run_valid = False
                continue
            body = str(c.get("body", ""))
            commit = _extract_review_job_commit_prefix(body)
            if commit and not _same_commit(commit, _extract_review_job_commit_prefix(str(older.get("body", "")))):
                run_valid = False
                continue
            run = _run_id(body)
            if run and (run == "!ambiguous" or run != _run_id(str(older.get("body", "")))):
                run_valid = False
                continue
            if parse_code_review(body) is not None:
                closed.append(c)
            else:
                run_valid = False
        if len(closed) != 1:
            run_valid = False

    # Test results from the selected test comment
    test_evidence = TestResultEvidence("SKIPPED" if skip_requested else "MISSING")
    if test_comment_id > 0:
        test_body = str(test_candidates[0].get("body", "") or "")
        test_evidence = parse_test_results(test_body)

    # Preserve the entire requested matrix; failed rows must never disappear.
    successful_builds = build_builds

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
        test_passed=test_evidence.passed,
        test_failed=test_evidence.failed,
        test_skipped=test_evidence.status == "SKIPPED",
        test_status=test_evidence.status,
        skip_tests_requested=skip_requested,
        run_binding_valid=run_valid,
        request_comment=request,
        code_review_binding=("DIRECT_SHA" if _extract_review_job_commit_prefix(cr_body)
                             else "LEGACY_CONTIGUOUS_CHAIN_NO_DIRECT_SHA"),
        code_review_comment_id=code_review_comment_id,
        code_review_created_at=code_review_created,
        code_review_comment_updated_at=code_review_updated,
        code_review_text=cr_text,
    )

    return evidence
