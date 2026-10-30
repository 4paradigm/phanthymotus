"""GitHub state and immutable evidence proxy for Deploy Approval.

The GitHubStateProxy provides a narrow, read/write boundary for GitHub PR
state and Git Notes evidence records. It is the ONLY module that reads/writes
the GitHub lifecycle comment and labels. It MUST NOT import or call business logic
(RegistryClient, AgentCoreClient, CaseRunner, CosClient,
EvidenceBuilder, DeploymentService internals).

Responsibilities:
- fresh GET PR metadata
- fresh full HEAD SHA
- list/read PR comments
- fresh read of command comment actor identity
- repo collaborator permission read
- locate trusted Deploy Approval lifecycle comment
- parse hidden JSON state
- create/update the ONE lifecycle comment
- read current labels
- project status:* label with runtime self-heal
- preserve non-status labels
- enforce trusted identity on lifecycle hidden state
- persist cursor while preserving exact visible markdown (no stale state write)
- persist immutable per-PR evidence in a dedicated Git Notes ref
"""

from __future__ import annotations

import asyncio
import json
import base64
import hashlib
import logging
import re
import time
from datetime import datetime
from urllib.parse import parse_qs, urlsplit
from typing import Any

from .config import Config
from .image_ref import validate_image_ref
from .github_client import GitHubClient, GitHubError
from .comments import beijing_now_str

logger = logging.getLogger(__name__)

# Hidden state marker
HIDDEN_STATE_MARKER = "<!-- deploy-approval-state:v1\n"

# Allowed lifecycle status labels
_ALLOWED_STATUS_LABELS = frozenset({
    "status: review-required",
    "status: reviewing",
    "status: deploy-ready",
    "status: deploy-requested",
    "status: testing",
    "status: acceptance-required",
    "status: succeeded",
    "status: failed",
})

STATUS_PREFIX = "status:"

# Maximum hidden state JSON size (8 KB)
_MAX_HIDDEN_STATE_BYTES = 8192

# Maximum lifecycle comment body size (64 KB)
_MAX_COMMENT_BODY_BYTES = 65536
_MAX_EVIDENCE_DATA_BYTES = 8 * 1024 * 1024
_EVIDENCE_SINGLE_RECORD_MAX_BYTES = 24000
_EVIDENCE_PART_BYTES = 24000
_EVIDENCE_MAX_PARTS = 512

# Maximum visible lifecycle markdown soft budget (48 KiB)
_MAX_VISIBLE_LIFECYCLE_BYTES = 48 * 1024

# Visible history section markers
VISIBLE_HISTORY_START_MARKER = "<!-- deploy-approval-visible-history:start -->"
VISIBLE_HISTORY_END_MARKER = "<!-- deploy-approval-visible-history:end -->"

# History archive marker prefix
HISTORY_ARCHIVE_MARKER_PREFIX = "<!-- deploy-approval-history:"

# History event delimiter marker
HISTORY_EVENT_DELIMITER = "<!-- deploy-approval-history-event -->"


def _history_archive_marker(repo: str, pr_number: int, page: int) -> str:
    """Return the exact archive marker string for a given page."""
    return f"{HISTORY_ARCHIVE_MARKER_PREFIX}{repo}:{pr_number}:{page} -->"


def _next_history_archive_page(comments: list[dict], repo: str, pr_number: int) -> int:
    """Discover the next archive page number from existing comments.

    Parses markers of the form:
        <!-- deploy-approval-history:<repo>:<pr>:<page> -->
    Returns the smallest positive integer greater than any existing page.
    """
    page_num = 1
    prefix = f"{HISTORY_ARCHIVE_MARKER_PREFIX}{repo}:{pr_number}:"
    for c in comments:
        cbody = c.get("body", "")
        if not isinstance(cbody, str):
            continue
        if prefix not in cbody:
            continue
        # Extract page number after the prefix
        try:
            after_prefix = cbody.split(prefix, 1)[1]
            page_str = after_prefix.split(" -->", 1)[0].strip()
            p = int(page_str)
            if p >= page_num:
                page_num = p + 1
        except (ValueError, IndexError):
            pass
    return page_num



class GitHubStateProxyError(Exception):
    pass


class MultipleTrustedCommentsError(GitHubStateProxyError):
    """More than one trusted lifecycle comment found — fail closed."""
    pass


class MalformedHiddenStateError(GitHubStateProxyError):
    """Hidden state JSON is malformed or fails validation."""
    pass


class TrustedIdentityRequiredError(GitHubStateProxyError):
    """No trusted lifecycle comment author identity configured."""
    pass


def _is_valid_full_sha(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.match(r"^[0-9a-f]{40}$", value))


def _is_valid_image_ref(value: Any) -> bool:
    """Validate image_ref using the canonical validator.

    Accepts legacy digest or new tag form.
    Returns True if valid, False otherwise.
    """
    if not isinstance(value, str):
        return False
    try:
        validate_image_ref(value)
        return True
    except ValueError:
        return False



def _validate_review_evidence(data: dict) -> dict:
    """Strictly validate review_evidence dict. Returns validated data or raises.

    Required provenance fields:
      build_comment_id, build_comment_updated_at,
      commit_prefix, resolved_head_sha,
      test_comment_id, test_comment_updated_at,
      code_review_comment_id, code_review_comment_updated_at,
      review_author_id

    New snapshots may also carry the parsed test totals.  They are optional
    so existing lifecycle comments remain readable during migration.
    """
    if not isinstance(data, dict):
        raise MalformedHiddenStateError("review_evidence is not a dict")
    required = {
        "build_comment_id", "build_comment_updated_at",
        "commit_prefix", "resolved_head_sha",
        "test_comment_id", "test_comment_updated_at",
        "code_review_comment_id", "code_review_comment_updated_at",
        "review_author_id",
    }
    optional = {"test_passed", "test_failed", "test_skipped"}
    extra = set(data.keys()) - required - optional
    if extra:
        raise MalformedHiddenStateError(f"extra review_evidence keys: {', '.join(sorted(extra))}")
    for k in required:
        if k not in data:
            raise MalformedHiddenStateError(f"review_evidence missing key: {k!r}")
    bcid = data["build_comment_id"]
    if isinstance(bcid, bool) or not isinstance(bcid, int) or bcid <= 0:
        raise MalformedHiddenStateError("review_evidence.build_comment_id must be a positive int")
    bcut = data["build_comment_updated_at"]
    if not isinstance(bcut, str) or not bcut:
        raise MalformedHiddenStateError("review_evidence.build_comment_updated_at must be non-empty str")
    cp = data["commit_prefix"]
    if not isinstance(cp, str) or not re.fullmatch(r"[0-9a-f]{7,40}", cp):
        raise MalformedHiddenStateError("review_evidence.commit_prefix must be 7-40 lowercase hex")
    rhs = data["resolved_head_sha"]
    if not _is_valid_full_sha(rhs):
        raise MalformedHiddenStateError("review_evidence.resolved_head_sha must be 40 lowercase hex")
    tcid = data["test_comment_id"]
    if isinstance(tcid, bool) or not isinstance(tcid, int) or tcid < 0:
        raise MalformedHiddenStateError("review_evidence.test_comment_id must be a non-negative int")
    tcut = data["test_comment_updated_at"]
    if tcid == 0:
        if tcut:
            raise MalformedHiddenStateError(
                "review_evidence.test_comment_updated_at must be empty when test_comment_id is 0"
            )
    elif tcid > 0:
        if not isinstance(tcut, str) or not tcut:
            raise MalformedHiddenStateError(
                "review_evidence.test_comment_updated_at must be non-empty when test_comment_id > 0"
            )
    crc = data["code_review_comment_id"]
    if isinstance(crc, bool) or not isinstance(crc, int) or crc <= 0:
        raise MalformedHiddenStateError("review_evidence.code_review_comment_id must be a positive int")
    crcut = data["code_review_comment_updated_at"]
    if not isinstance(crcut, str) or not crcut:
        raise MalformedHiddenStateError(
            "review_evidence.code_review_comment_updated_at must be non-empty str"
        )
    rai = data["review_author_id"]
    if not isinstance(rai, str) or not rai or not rai.isdigit():
        raise MalformedHiddenStateError("review_evidence.review_author_id must be a non-empty numeric string")
    test_passed = data.get("test_passed", 0)
    test_failed = data.get("test_failed", 0)
    test_skipped = data.get("test_skipped", False)
    for key, value in (("test_passed", test_passed), ("test_failed", test_failed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MalformedHiddenStateError(
                f"review_evidence.{key} must be a non-negative int"
            )
    if not isinstance(test_skipped, bool):
        raise MalformedHiddenStateError(
            "review_evidence.test_skipped must be a bool"
        )
    return data


def _validate_hidden_state(data: dict) -> dict:
    """Strictly validate hidden state JSON. Returns validated data or raises."""
    if not isinstance(data, dict):
        raise MalformedHiddenStateError("hidden state is not a dict")

    allowed_keys = {
        "version", "head_sha", "status", "review_evidence",
        "components", "deployments", "approve_attempts", "approve_attempts_total",
        "approve_attempts_truncated",
        "case_results", "test_result", "cos", "command", "last_processed_comment_id",
        "round_id", "round_started_at", "round_archive",
        "acceptance_request",
    }
    extra_keys = set(data.keys()) - allowed_keys
    if extra_keys:
        raise MalformedHiddenStateError(
            f"extra keys not allowed: {', '.join(sorted(extra_keys))}"
        )

    if "round_id" in data:
        from .evidence_models import round_id_for_state
        try:
            rid = round_id_for_state(data)
            if not rid.startswith(str(data.get("head_sha")) + "-"):
                raise ValueError("round head mismatch")
        except ValueError as exc:
            raise MalformedHiddenStateError("invalid round_id") from exc
    if "round_started_at" in data:
        from datetime import datetime
        value = data["round_started_at"]
        try:
            if not isinstance(value, str) or datetime.fromisoformat(value.replace("Z", "+00:00")).utcoffset() is None:
                raise ValueError("timestamp requires timezone")
        except ValueError as exc:
            raise MalformedHiddenStateError("invalid round_started_at") from exc
    if "round_archive" in data:
        link = data["round_archive"]
        legacy_archive = isinstance(link, dict) and set(link) == {"round_id", "comment_id", "url", "sha256"}
        note_archive = isinstance(link, dict) and set(link) == {"round_id", "sha256"}
        if (link is not None and (not (legacy_archive or note_archive)
                or (legacy_archive and (type(link["comment_id"]) is not int or link["comment_id"] <= 0
                    or not isinstance(link["url"], str) or not link["url"].startswith("https://github.com/")))
                or not isinstance(link["round_id"], str)
                or not re.fullmatch(r"[0-9a-f]{40}-[0-9]+", link["round_id"])
                or not isinstance(link["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", link["sha256"]))):
            raise MalformedHiddenStateError("invalid round_archive")

    if "acceptance_request" in data:
        request = data["acceptance_request"]
        base_fields = {
            "head_sha", "round_id", "status", "requested_by",
            "requested_by_id", "comment_id", "requested_at",
        }
        completed_fields = base_fields | {
            "accepted_by", "accepted_by_id", "acceptance_comment_id",
            "accepted_at", "result", "summary",
        }
        if (not isinstance(request, dict)
                or request.get("status") not in {"pending", "completed"}
                or set(request) != (base_fields if request.get("status") == "pending"
                                    else completed_fields)
                or request.get("head_sha") != data.get("head_sha")
                or request.get("round_id") != data.get("round_id")
                or request.get("status") == "pending" and data.get("status") != "testing"
                or request.get("status") == "completed"
                and data.get("status") not in {"succeeded", "failed"}):
            raise MalformedHiddenStateError("invalid acceptance_request")

        def valid_login(value):
            return (isinstance(value, str)
                    and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", value)
                    is not None)

        def valid_timestamp(value):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return isinstance(value, str) and parsed.utcoffset() is not None
            except (AttributeError, TypeError, ValueError):
                return False

        if (not valid_login(request.get("requested_by"))
                or type(request.get("requested_by_id")) is not int
                or request["requested_by_id"] <= 0
                or type(request.get("comment_id")) is not int
                or request["comment_id"] <= 0
                or not valid_timestamp(request.get("requested_at"))):
            raise MalformedHiddenStateError("invalid acceptance_request identity")
        if request.get("status") == "completed":
            if (not valid_login(request.get("accepted_by"))
                    or type(request.get("accepted_by_id")) is not int
                    or request["accepted_by_id"] <= 0
                    or type(request.get("acceptance_comment_id")) is not int
                    or request["acceptance_comment_id"] <= 0
                    or request["acceptance_comment_id"] <= request["comment_id"]
                    or not valid_timestamp(request.get("accepted_at"))
                    or request.get("result") not in {"pass", "fail"}
                    or request.get("result") != data.get("test_result")
                    or data.get("status") != ("succeeded" if request.get("result") == "pass" else "failed")
                    or not isinstance(request.get("summary"), str)
                    or len(request["summary"]) > 5000):
                raise MalformedHiddenStateError("invalid acceptance_request completion")

    # Explicitly reject unbounded history fields — history must persist
    # in visible markdown only, never in hidden JSON state.
    if "history" in data:
        raise MalformedHiddenStateError(
            "hidden state must not contain 'history' field; "
            "history persists in visible lifecycle markdown only"
        )
    if "history_events" in data:
        raise MalformedHiddenStateError(
            "hidden state must not contain 'history_events' field; "
            "history persists in visible lifecycle markdown only"
        )

    if data.get("version") != 1:
        raise MalformedHiddenStateError(f"unsupported version: {data.get('version')!r}")
    head_sha = data.get("head_sha", "")
    if not _is_valid_full_sha(head_sha):
        raise MalformedHiddenStateError(f"invalid head_sha: {head_sha!r}")
    status = data.get("status", "")
    allowed_statuses = {
        "review-required", "reviewing", "deploy-ready",
        "deploy-requested", "testing", "succeeded",
        "failed",
    }
    if status not in allowed_statuses:
        raise MalformedHiddenStateError(f"invalid status: {status!r}")

    review_evidence = data.get("review_evidence", {})
    if not isinstance(review_evidence, dict):
        raise MalformedHiddenStateError("review_evidence must be a dict")
    if review_evidence:
        _validate_review_evidence(review_evidence)

    components = data.get("components", [])
    if not isinstance(components, list):
        raise MalformedHiddenStateError("components must be a list")
    known_cids: list[str] = []
    for comp in components:
        if not isinstance(comp, dict):
            raise MalformedHiddenStateError("component must be a dict")
        comp_keys = set(comp.keys())
        if comp_keys not in (
            {"component_id", "target", "driver_path", "variant", "review_image_tag", "image_ref", "resolved_platform"},
            {"component_id", "target", "driver_path", "variant", "review_image_tag", "image_ref", "resolved_platform", "runtime_id"},
        ):
            raise MalformedHiddenStateError("component keys must match the canonical schema")
        cid = comp.get("component_id", "")
        if not isinstance(cid, str) or not cid:
            raise MalformedHiddenStateError("component_id missing")
        if cid in known_cids:
            raise MalformedHiddenStateError("duplicate component_id values")
        known_cids.append(cid)
        review_image_tag = comp.get("review_image_tag", "")
        if not isinstance(review_image_tag, str) or not review_image_tag:
            raise MalformedHiddenStateError("review_image_tag missing")
        target = comp.get("target")
        if target not in {"perception", "actucore", "driver", "core"}:
            raise MalformedHiddenStateError(f"invalid component target: {comp.get('target')!r}")
        if not isinstance(comp.get("driver_path"), str):
            raise MalformedHiddenStateError("driver_path must be a string")
        if not isinstance(comp.get("variant"), str):
            raise MalformedHiddenStateError("variant must be a string")
        image_ref = comp.get("image_ref", "")
        if not _is_valid_image_ref(image_ref):
            raise MalformedHiddenStateError(f"invalid image_ref: {image_ref!r}")
        resolved_platform = comp.get("resolved_platform", "")
        if not isinstance(resolved_platform, str) or not resolved_platform:
            raise MalformedHiddenStateError("resolved_platform must be a non-empty string")
        runtime_id = comp.get("runtime_id", "")
        if runtime_id != "" and (not isinstance(runtime_id, str) or not runtime_id):
            raise MalformedHiddenStateError("runtime_id must be an empty or non-empty string")
        if target == "core":
            if comp_keys != {
                "component_id", "target", "driver_path", "variant",
                "review_image_tag", "image_ref", "resolved_platform", "runtime_id",
            }:
                raise MalformedHiddenStateError(
                    "core component must use the canonical runtime-bound schema"
                )
            if comp.get("driver_path") != "":
                raise MalformedHiddenStateError("core driver_path must be empty")
            if comp.get("variant") != "":
                raise MalformedHiddenStateError("core variant must be empty")
            if runtime_id != "core":
                raise MalformedHiddenStateError("core runtime_id must be 'core'")

    deployments = data.get("deployments", [])
    if not isinstance(deployments, list):
        raise MalformedHiddenStateError("deployments must be a list")
    deployed_cids: set[str] = set()
    for dep in deployments:
        if not isinstance(dep, dict):
            raise MalformedHiddenStateError("deployment must be a dict")
        if set(dep.keys()) not in (
                {"machine", "component_ids", "phase"},
                {"machine", "component_ids", "phase", "ip"}):
            raise MalformedHiddenStateError("deployment keys must match the canonical schema")
        machine = dep.get("machine", "")
        if not isinstance(machine, str) or not machine:
            raise MalformedHiddenStateError("deployment must have non-empty machine")
        if "ip" in dep and (
                not isinstance(dep["ip"], str) or not dep["ip"]
                or len(dep["ip"]) > 255
                or any(ord(char) < 33 or ord(char) == 127 for char in dep["ip"])):
            raise MalformedHiddenStateError("deployment.ip must be a bounded host value")
        if dep.get("phase") != "deployed":
            raise MalformedHiddenStateError(f"invalid deployment phase: {dep.get('phase')!r}")
        cids = dep.get("component_ids", [])
        if not isinstance(cids, list) or not cids:
            raise MalformedHiddenStateError("deployment must have non-empty component_ids")
        seen_in_dep: set[str] = set()
        for cid in cids:
            if not isinstance(cid, str) or not cid:
                raise MalformedHiddenStateError("deployment component_ids must be strings")
            if cid not in known_cids:
                raise MalformedHiddenStateError(f"deployment references unknown component {cid!r}")
            if cid in deployed_cids or cid in seen_in_dep:
                raise MalformedHiddenStateError(f"component {cid!r} deployed more than once")
            seen_in_dep.add(cid)
            deployed_cids.add(cid)

    approve_attempts = data.get("approve_attempts", [])
    if not isinstance(approve_attempts, list):
        raise MalformedHiddenStateError("approve_attempts must be a list")
    for attempt in approve_attempts:
        if not isinstance(attempt, dict):
            raise MalformedHiddenStateError("approve_attempt must be a dict")
        required_attempt_keys = {"comment_id", "actor", "machine", "preflight", "outcome", "health"}
        if set(attempt.keys()) != required_attempt_keys:
            raise MalformedHiddenStateError("approve_attempt keys must match the canonical schema")
        comment_id = attempt.get("comment_id", -1)
        if isinstance(comment_id, bool) or not isinstance(comment_id, int) or comment_id < 0:
            raise MalformedHiddenStateError("approve_attempt.comment_id must be a non-negative int")
        for key in ("actor", "machine", "outcome"):
            val = attempt.get(key, "")
            if not isinstance(val, str) or not val:
                raise MalformedHiddenStateError(f"approve_attempt.{key} must be a non-empty string")
        if attempt.get("outcome") not in {"blocked_occupied", "approval_revoked", "deployed", "failed", "uncertain"}:
            raise MalformedHiddenStateError("approve_attempt.outcome invalid")
        preflight = attempt.get("preflight", [])
        health = attempt.get("health", [])
        if not isinstance(preflight, list) or not isinstance(health, list):
            raise MalformedHiddenStateError("approve_attempt preflight/health must be lists")
        for h in health:
            if not isinstance(h, dict):
                raise MalformedHiddenStateError("approve_attempt health entries must be dicts")
            # Legacy schema (deprecated but still allowed for backward compat)
            legacy_keys = {"component_id", "runtime_id", "running_image", "passed"}
            # New schema — post-deploy verification (verified == True/False)
            new_schema_keys = {"component_id", "runtime_id", "status", "running_image", "target_image", "verified"}
            # New schema with optional error (uncertain post-deploy)
            new_schema_with_error_keys = new_schema_keys | {"error"}
            # Core self-update verification schema.  Core has no running_image
            # or driver status; it is verified through update-check current_tag.
            core_schema_keys = {
                "component_id", "runtime_id", "current_tag", "target_tag", "verified",
            }
            core_schema_optional = {"already_target", "error"}
            h_keys = set(h.keys())
            is_core_schema = (
                core_schema_keys <= h_keys
                and h_keys - core_schema_keys <= core_schema_optional
            )
            if (
                h_keys != legacy_keys
                and h_keys not in (new_schema_keys, new_schema_with_error_keys)
                and not is_core_schema
            ):
                raise MalformedHiddenStateError(
                    f"approve_attempt health keys must match canonical schema, got {h_keys}"
                )
            if is_core_schema:
                component_id = h.get("component_id")
                if not isinstance(component_id, str) or not component_id:
                    raise MalformedHiddenStateError(
                        "core health component_id must be a non-empty string"
                    )
                if h.get("runtime_id") != "core":
                    raise MalformedHiddenStateError(
                        "core health runtime_id must be 'core'"
                    )
                if not isinstance(h.get("current_tag"), str):
                    raise MalformedHiddenStateError(
                        "core health current_tag must be a string"
                    )
                if not isinstance(h.get("target_tag"), str) or not h.get("target_tag"):
                    raise MalformedHiddenStateError(
                        "core health target_tag must be a non-empty string"
                    )
                if not isinstance(h.get("verified"), bool):
                    raise MalformedHiddenStateError(
                        "core health verified must be a bool"
                    )
                if h.get("verified") is True:
                    if not h.get("current_tag"):
                        raise MalformedHiddenStateError(
                            "core health verified=true requires non-empty current_tag"
                        )
                    if h.get("current_tag") != h.get("target_tag"):
                        raise MalformedHiddenStateError(
                            "core health verified=true requires current_tag==target_tag"
                        )
                if "already_target" in h:
                    if not isinstance(h["already_target"], bool):
                        raise MalformedHiddenStateError(
                            "core health already_target must be a bool"
                        )
                    if h["already_target"] is True and (
                        h.get("verified") is not True
                        or h.get("current_tag") != h.get("target_tag")
                    ):
                        raise MalformedHiddenStateError(
                            "core health already_target=true requires verified target"
                        )
                if "error" in h:
                    if not isinstance(h["error"], str) or not h["error"]:
                        raise MalformedHiddenStateError(
                            "core health error must be a non-empty string when present"
                        )
                    if h.get("verified") is not False:
                        raise MalformedHiddenStateError(
                            "core health error implies verified=false"
                        )
                continue
            if h.get("runtime_id") == "core":
                raise MalformedHiddenStateError(
                    "core health must use the canonical update-check schema"
                )
            if not isinstance(h.get("component_id", ""), str):
                raise MalformedHiddenStateError("approve_attempt health component_id must be a string")
            if not isinstance(h.get("runtime_id", ""), str):
                raise MalformedHiddenStateError("approve_attempt health runtime_id must be a string")
            if not isinstance(h.get("running_image", ""), str):
                raise MalformedHiddenStateError("approve_attempt health running_image must be a string")
            if "passed" in h and not isinstance(h.get("passed"), bool):
                raise MalformedHiddenStateError("approve_attempt health passed must be a bool")
            if "verified" in h and not isinstance(h.get("verified"), bool):
                raise MalformedHiddenStateError("approve_attempt health verified must be a bool")
            # Semantic validation for new schema
            if h_keys in (new_schema_keys, new_schema_with_error_keys):
                if h.get("verified") is True:
                    if h.get("status") != "running":
                        raise MalformedHiddenStateError(
                            "health verified=true requires status==running"
                        )
                    if h.get("running_image") != h.get("target_image"):
                        raise MalformedHiddenStateError(
                            "health verified=true requires running_image==target_image"
                        )
                    if not h.get("target_image"):
                        raise MalformedHiddenStateError(
                            "health verified=true requires non-empty target_image"
                        )
                if "error" in h:
                    if not isinstance(h["error"], str) or not h["error"]:
                        raise MalformedHiddenStateError(
                            "health error must be a non-empty string when present"
                        )
                    if h.get("verified") is not False:
                        raise MalformedHiddenStateError(
                            "health error implies verified=false"
                        )
    approve_attempts_total = data.get("approve_attempts_total", len(approve_attempts))
    if isinstance(approve_attempts_total, bool) or not isinstance(approve_attempts_total, int):
        raise MalformedHiddenStateError("approve_attempts_total must be a non-negative int")
    if approve_attempts_total < 0:
        raise MalformedHiddenStateError("approve_attempts_total must be a non-negative int")
    if approve_attempts_total < len(approve_attempts):
        raise MalformedHiddenStateError(
            "approve_attempts_total must be >= len(approve_attempts)"
        )
    approve_attempts_truncated = data.get("approve_attempts_truncated", False)
    if not isinstance(approve_attempts_truncated, bool):
        raise MalformedHiddenStateError("approve_attempts_truncated must be a bool")

    case_results = data.get("case_results", {})
    if not isinstance(case_results, dict):
        raise MalformedHiddenStateError("case_results must be a dict")
    for cid, val in case_results.items():
        if cid not in known_cids:
            raise MalformedHiddenStateError(f"case_results key {cid!r} is not a known component_id")
        if val not in {"running", "pass", "fail", "n/a"}:
            raise MalformedHiddenStateError(
                f"case_results value {val!r} must be running/pass/fail/n/a"
            )

    test_result = data.get("test_result", "")
    if test_result not in {"", "pass", "fail"}:
        raise MalformedHiddenStateError("test_result must be '', 'pass', or 'fail'")

    cos = data.get("cos", {})
    if not isinstance(cos, dict):
        raise MalformedHiddenStateError("cos must be a dict")
    if set(cos.keys()) != {"object_key", "sha256", "size"}:
        raise MalformedHiddenStateError("cos keys must match the canonical schema")
    object_key = cos.get("object_key", "")
    sha256 = cos.get("sha256", "")
    size = cos.get("size", 0)
    if not isinstance(object_key, str):
        raise MalformedHiddenStateError("cos.object_key must be a string")
    if not isinstance(sha256, str):
        raise MalformedHiddenStateError("cos.sha256 must be a string")
    if sha256 and not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise MalformedHiddenStateError("cos.sha256 must be empty or 64 lowercase hex")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise MalformedHiddenStateError("cos.size must be a non-negative int")
    if bool(object_key) != bool(sha256):
        raise MalformedHiddenStateError("cos.object_key and cos.sha256 must be set together")
    if not object_key and size != 0:
        raise MalformedHiddenStateError("cos.size must be 0 when object_key is empty")

    command = data.get("command", {})
    if not isinstance(command, dict):
        raise MalformedHiddenStateError("command must be a dict")
    if set(command.keys()) != {"comment_id", "kind", "phase", "args"}:
        raise MalformedHiddenStateError("command keys must match the canonical schema")
    comment_id = command.get("comment_id", -1)
    if isinstance(comment_id, bool) or not isinstance(comment_id, int) or comment_id < 0:
        raise MalformedHiddenStateError("command.comment_id must be a non-negative int")
    if command.get("kind", "") not in {"", "request_deploy", "approve_deploy", "request_acceptance", "record_test", "deploy_status", "deploy_help"}:
        raise MalformedHiddenStateError(f"command.kind invalid: {command.get('kind')!r}")
    if command.get("phase") not in {"completed", "executing", "uncertain"}:
        raise MalformedHiddenStateError(f"command.phase invalid: {command.get('phase')!r}")
    args = command.get("args")
    if not isinstance(args, dict):
        raise MalformedHiddenStateError("command.args must be a dict")
    if command.get("phase") in {"executing", "uncertain"} and command.get("kind") == "approve_deploy":
        machine = args.get("machine", "")
        if not isinstance(machine, str) or not machine:
            raise MalformedHiddenStateError("approve_deploy executing/uncertain requires machine arg")
    if command.get("phase") == "uncertain" and status != "deploy-requested":
        raise MalformedHiddenStateError("uncertain command phase requires status deploy-requested")

    last_processed = data.get("last_processed_comment_id")
    if isinstance(last_processed, bool) or not isinstance(last_processed, int) or last_processed < 0:
        raise MalformedHiddenStateError("last_processed_comment_id must be a non-negative int")
    acceptance_request = data.get("acceptance_request")
    if (isinstance(acceptance_request, dict)
            and (acceptance_request["comment_id"] > last_processed
                 or acceptance_request.get("status") == "completed"
                 and acceptance_request["acceptance_comment_id"] > last_processed)):
        raise MalformedHiddenStateError("acceptance_request comments exceed the persisted cursor")

    if status in {"deploy-ready", "deploy-requested", "testing", "succeeded", "failed"}:
        if not review_evidence:
            raise MalformedHiddenStateError(f"review_evidence must be non-empty for status {status!r}")
    if status in {"deploy-requested", "testing", "succeeded", "failed"}:
        if not components:
            raise MalformedHiddenStateError(f"components must be non-empty for status {status!r}")
    if status == "succeeded" and test_result != "pass":
        raise MalformedHiddenStateError("test_result must be 'pass' when status is succeeded")
    if status == "failed" and test_result not in {"", "fail"}:
        raise MalformedHiddenStateError("test_result must be 'fail' when status is failed")

    return data


def _extract_hidden_state(body: str) -> dict | None:
    """Extract and parse hidden JSON state from comment body."""
    marker = HIDDEN_STATE_MARKER
    start = body.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = body.find("\n-->", start)
    if end < 0:
        return None
    raw = body[start:end].strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _build_hidden_state_body(visible_markdown: str, state: dict) -> str:
    """Build full comment body with visible markdown and hidden JSON state."""
    hidden_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    if len(hidden_json.encode("utf-8")) > _MAX_HIDDEN_STATE_BYTES:
        raise GitHubStateProxyError(
            f"hidden state JSON exceeds {_MAX_HIDDEN_STATE_BYTES} bytes"
        )
    hidden_block = f"{HIDDEN_STATE_MARKER}{hidden_json}\n-->"
    if visible_markdown:
        # Collapse all trailing newlines to a single newline before adding
        # the standard two-newline separator.  This keeps the boundary before
        # the hidden-state marker perfectly idempotent when persist_cursor
        # re-reads body[:idx].rstrip() and re-serialises.
        vm = visible_markdown.rstrip("\n") + "\n"
        return vm + "\n\n" + hidden_block
    return hidden_block



# ── Visible history helpers ──────────────────────────────────────────────────


def _parse_visible_history(body: str) -> tuple[str, list[dict]]:
    """Parse visible lifecycle body into (body_without_history, events).

    Returns the body text stripped of the generated history section,
    plus a list of event dicts (newest first).
    """
    start = body.find(VISIBLE_HISTORY_START_MARKER)
    end = body.find(VISIBLE_HISTORY_END_MARKER)
    if start < 0 or end < 0 or end <= start:
        # No generated history section — whole body is visible
        return body.rstrip(), []
    # Strip history section from body
    before = body[:start].rstrip()
    after = body[end + len(VISIBLE_HISTORY_END_MARKER):].lstrip()
    clean_body = (before + "\n" + after).strip() + "\n"

    # Parse events from the history section
    history_section = body[start + len(VISIBLE_HISTORY_START_MARKER):end]
    events: list[dict] = []
    # Split by optional delimiter
    blocks = re.split(r"<!-- deploy-approval-history-event -->", history_section)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        evt: dict[str, Any] = {}
        for line in block.splitlines():
            line = line.strip()
            # Extract timestamp from header #### 2026-09-24 10:00:00 \u2014 Title
            ts_match = re.match(r"^####\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\u2014\s+(.*)$", line)
            if ts_match:
                evt["timestamp"] = ts_match.group(1)
                evt["event"] = ts_match.group(2)
                continue
            if line.startswith("**Lifecycle:**"):
                evt["lifecycle"] = line.replace("**Lifecycle:**", "").strip()
            elif line.startswith("**Machine:**"):
                m = re.search(r"`([^`]*)`", line)
                if m:
                    evt["machine"] = m.group(1)
                ip_m = re.search("\u00b7\\s+\\*\\*IP:\\*\\*\\s+`([^`]*)`", line)
                if ip_m:
                    evt["ip"] = ip_m.group(1)
            elif line.startswith("**Components:**"):
                evt["components"] = line.replace("**Components:**", "").strip()
            elif line.startswith("**Result:**"):
                evt["result"] = line.replace("**Result:**", "").strip()
        if evt:
            events.append(evt)
    return clean_body, events


def _build_history_block(events: list[dict]) -> str:
    """Build the generated history block, folded by default in GitHub."""
    lines = [
        VISIBLE_HISTORY_START_MARKER,
        "<details>",
        "<summary>History · 历史事件</summary>",
        "",
    ]
    for evt in events:
        ts = evt.get("timestamp", beijing_now_str())
        title = evt.get("event", "Event")
        lines.append(f"#### {ts} \u2014 {title}")
        lines.append("")
        lifecycle = evt.get("lifecycle", "")
        if lifecycle:
            lines.append(f"**Lifecycle:** {lifecycle}")
            lines.append("")
        machine = evt.get("machine", "")
        ip = evt.get("ip", "")
        if machine or ip:
            if machine and ip:
                lines.append(f"**Machine:** `{machine}` \u00b7 **IP:** `{ip}`")
            elif machine:
                lines.append(f"**Machine:** `{machine}`")
            else:
                lines.append(f"**IP:** `{ip}`")
            lines.append("")
        components = evt.get("components", "")
        if components:
            lines.append(f"**Components:** {components}")
            lines.append("")
        result = evt.get("result", "")
        if result:
            lines.append(f"**Result:** {result}")
            lines.append("")
        lines.append(HISTORY_EVENT_DELIMITER)
        lines.append("")
    lines.extend(["</details>", VISIBLE_HISTORY_END_MARKER])
    return "\n".join(lines)


def _insert_history_into_visible(visible: str, event: dict) -> str:
    """Insert a new history event into the visible lifecycle markdown.

    Preserves existing generated history section; prepends new event.
    Ensures exactly one "### History" heading exists.
    """
    start = visible.find(VISIBLE_HISTORY_START_MARKER)
    end = visible.find(VISIBLE_HISTORY_END_MARKER)
    if start >= 0 and end >= 0 and end > start:
        # Parse existing events, prepend new event
        _, existing_events = _parse_visible_history(visible)
        new_events = [event] + existing_events
        new_history = _build_history_block(new_events)
        # Replace existing history section
        before_history = re.sub(
            r"(?:^|\n)#{2,6}[ \t]+History[ \t]*\n[ \t]*$", "\n",
            visible[:start], flags=re.IGNORECASE,
        )
        after_history = visible[end + len(VISIBLE_HISTORY_END_MARKER):]
        return before_history + new_history + after_history
    else:
        # No existing history section — append one with "### History" heading
        existing_events: list[dict] = []
        new_events = [event] + existing_events
        new_history = _build_history_block(new_events)
        return visible.rstrip() + "\n\n" + new_history + "\n"




def _render_history_event(evt: dict) -> str:
    """Render a single history event dict into markdown lines.

    Used by both _build_history_block (main) and archive creation,
    ensuring archived events are semantically identical to main history.
    """
    lines = []
    ts = evt.get("timestamp", beijing_now_str())
    title = evt.get("event", "Event")
    lines.append(f"#### {ts} — {title}")
    lines.append("")
    lifecycle = evt.get("lifecycle", "")
    if lifecycle:
        lines.append(f"**Lifecycle:** {lifecycle}")
        lines.append("")
    machine = evt.get("machine", "")
    ip = evt.get("ip", "")
    if machine or ip:
        from .comments import _escape
        if machine and ip:
            lines.append(f"**Machine:** `{_escape(machine)}` · **IP:** `{ip}`")
        elif machine:
            lines.append(f"**Machine:** `{_escape(machine)}`")
        else:
            lines.append(f"**IP:** `{ip}`")
        lines.append("")
    components = evt.get("components", "")
    if components:
        lines.append(f"**Components:** {components}")
        lines.append("")
    result = evt.get("result", "")
    if result:
        lines.append(f"**Result:** {result}")
        lines.append("")
    return "\n".join(lines)


def _count_visible_bytes(visible: str) -> int:
    """Count UTF-8 encoded bytes of visible lifecycle markdown."""
    return len(visible.encode("utf-8"))


def _truncate_events_to_fit(
    events: list[dict], target_bytes: int,
) -> tuple[list[dict], list[dict]]:
    """Truncate oldest events until total history block fits target_bytes.

    Returns (kept_events, archived_events).
    Never silently drops events — archived events must be moved to an archive.
    Every popped event goes into archived; kept + archived == original (no dup, no loss).
    """
    # Try keeping all events
    candidate = _build_history_block(events)
    if _count_visible_bytes(candidate) <= target_bytes:
        return events, []
    # Remove oldest events one by one until it fits
    kept = list(events)
    archived: list[dict] = []
    while kept:
        removed = kept.pop()  # remove oldest (last in newest-first list)
        archived.insert(0, removed)
        candidate = _build_history_block(kept)
        if _count_visible_bytes(candidate) <= target_bytes:
            break
    return kept, archived


def _build_archive_body(
    repo: str,
    pr_number: int,
    page_num: int,
    events: list[dict],
    extra_text: str = "",
) -> str:
    """Build an immutable history archive comment body.

    Always starts with BOT_MARKER + archive marker.
    """
    from .comments import BOT_MARKER
    archive_marker = _history_archive_marker(repo, pr_number, page_num)
    archive_lines = [
        BOT_MARKER,
        archive_marker,
        f"### Deploy Approval — History Archive Page {page_num}",
        "",
    ]
    if extra_text:
        archive_lines.append(extra_text)
        archive_lines.append("")
    for evt in events:
        event_text = _render_history_event(evt)
        archive_lines.append(event_text)
        archive_lines.append(HISTORY_EVENT_DELIMITER)
        archive_lines.append("")
    return "\n".join(archive_lines)

class GitHubStateProxy:
    """GitHub-only state proxy for Deploy Approval.

    Reads and writes state through the GitHub API only.
    Must NOT import business logic modules.
    """

    def __init__(
        self,
        config: Config,
        github: GitHubClient,
        github_app_id: str = "",
    ):
        """Initialize the GitHub state proxy.

        Parameters
        ----------
        github_app_id : str
            The GitHub App ID used for lazy provenance validation of
            lifecycle comments.  Trust is established per-comment
            by checking ``performed_via_github_app.id == github_app_id``.
        """
        self.config = config
        self._github = github
        self._github_app_id = github_app_id
        # Status labels are an optional projection. Cache successful checks
        # briefly and back off failed projections so a missing label cannot
        # generate an API request on every watcher pass.
        self._status_label_projection: dict[tuple[str, int], dict[str, Any]] = {}

    # ── GitHub API passthrough ──

    async def get_pr(self, repo: str, pr_number: int) -> dict:
        return await self._github.get_pr(repo, pr_number)

    async def get_open_prs(self, repo: str) -> list[dict]:
        return await self._github.list_open_prs(repo)

    async def get_issue_comments(self, repo: str, pr_number: int) -> list[dict]:
        return await self._github.get_issue_comments(repo, pr_number)

    async def get_comment(self, repo: str, comment_id: int) -> dict:
        return await self._github.get_comment(repo, comment_id)

    async def post_issue_comment(self, repo: str, pr_number: int,
                                 body: str) -> dict:
        return await self._github.post_issue_comment(repo, pr_number, body)

    async def update_comment(self, repo: str, comment_id: int,
                             body: str) -> dict:
        return await self._github.update_comment(repo, comment_id, body)

    async def get_issue_labels(self, repo: str, issue_number: int) -> list[str]:
        return await self._github.get_issue_labels(repo, issue_number)

    async def add_issue_label(
        self,
        repo: str,
        issue_number: int,
        label: str,
    ) -> None:
        await self._github.add_issue_label(
            repo, issue_number, label,
        )

    async def remove_issue_label(
        self,
        repo: str,
        issue_number: int,
        label: str,
    ) -> None:
        await self._github.remove_issue_label(
            repo, issue_number, label,
        )

    async def comment_identity(self, repo: str,
                               comment_id: int) -> tuple[str, str]:
        """Return (author_id, author_login) for a comment."""
        comment = await self._github.get_comment(repo, comment_id)
        if not comment:
            return ("", "")
        user = comment.get("user", {})
        uid = user.get("id")
        login = user.get("login")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
            return ("", "")
        if not isinstance(login, str) or not login:
            return ("", "")
        author_id = str(uid)
        author_login = login
        return (author_id, author_login)

    async def collaborator_permission(self, repo: str,
                                      actor: str) -> str:
        """Get collaborator permission level for an actor.

        Returns permission string (admin, write, read, etc.) or empty string.
        Fail closed on error.
        """
        try:
            return await self._github.collaborator_permission(repo, actor)
        except Exception as e:
            logger.warning(
                "collaborator_permission %s %s: %s", repo, actor, e,
            )
            return ""

    # ── Lifecycle comment management ──

    async def find_trusted_lifecycle_comment(
        self, repo: str, pr_number: int
    ) -> dict | None:
        """Find the ONE trusted lifecycle comment for this PR.

        Lazy provenance: trust is established per-comment by checking
        ``performed_via_github_app.id == github_app_id``.  No startup
        bot-identity binding is required.

        A comment is trusted only when ALL of:
        1. body contains the exact deploy-approval state marker
        2. performed_via_github_app exists, is dict, id is non-bool positive int
        3. performed_via_github_app.id == configured github_app_id
        4. user dict exists with non-bool positive id and non-empty login
           (if user.type is present, it must be "Bot")

        Returns the comment dict, or None if no trusted comment exists.
        Raises MultipleTrustedCommentsError if >1 trusted comments exist.
        """
        if not self._github_app_id:
            raise TrustedIdentityRequiredError(
                "github_app_id is required before reading lifecycle state"
            )
        comments = await self.get_issue_comments(repo, pr_number)
        trusted: list[dict] = []
        for c in comments:
            body = c.get("body", "")
            if not isinstance(body, str):
                continue
            if HIDDEN_STATE_MARKER not in body:
                continue
            # Check performed_via_github_app provenance
            pvga = c.get("performed_via_github_app")
            if not isinstance(pvga, dict):
                continue
            app_id = pvga.get("id")
            if not (isinstance(app_id, int) and not isinstance(app_id, bool) and app_id > 0):
                continue
            if str(app_id) != self._github_app_id:
                logger.warning(
                    "ignoring state marker from mismatched app id %s",
                    app_id,
                )
                continue
            # Also validate user dict exists
            user = c.get("user")
            if not isinstance(user, dict):
                continue
            uid = user.get("id", "")
            if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
                continue
            login = user.get("login", "")
            if not isinstance(login, str) or not login:
                continue
            # If user type is provided, it must be Bot
            utype = user.get("type")
            if utype is not None and utype != "Bot":
                logger.warning(
                    "ignoring state marker from non-bot user type %r", utype,
                )
                continue
            trusted.append(c)

        if len(trusted) > 1:
            raise MultipleTrustedCommentsError(
                f"found {len(trusted)} trusted lifecycle comments for {repo}#{pr_number}"
            )
        if not trusted:
            return None
        return trusted[0]

    async def read_hidden_state(
        self, repo: str, pr_number: int
    ) -> dict | None:
        """Read and validate hidden state from the trusted lifecycle comment.

        Returns validated state dict, or None if no trusted comment exists.
        Raises on malformed state or multiple trusted comments.
        """
        comment = await self.find_trusted_lifecycle_comment(repo, pr_number)
        if comment is None:
            return None
        body = comment.get("body", "")
        if not isinstance(body, str):
            return None
        data = _extract_hidden_state(body)
        if data is None:
            return None
        return _validate_hidden_state(data)

    def is_bot_comment(self, comment: dict) -> bool:
        """Check if a comment is authored by the configured GitHub App.

        Only relies on performed_via_github_app.id provenance.
        """
        if self._github_app_id:
            pvga = comment.get("performed_via_github_app")
            if isinstance(pvga, dict):
                app_id = pvga.get("id")
                if isinstance(app_id, int) and not isinstance(app_id, bool):
                    if str(app_id) == self._github_app_id:
                        return True
        return False

    async def write_hidden_state(
        self,
        repo: str,
        pr_number: int,
        visible_markdown: str,
        state: dict,
        *, expected_comment: tuple[int | None, str | None] | None = None,
    ) -> dict | None:
        """Write hidden state to the trusted lifecycle comment.

        If no trusted comment exists, creates a new one.
        Returns the comment dict, or None on failure. When expected_comment
        is supplied, require the same ID/body at the final read before PATCH;
        (None, None) requires that no lifecycle comment exists yet.

        Trust is established per-comment via lazy provenance in
        find_trusted_lifecycle_comment.  The installation token used
        to create/update comments is already trusted; we only require
        that github_app_id is configured for consistency.
        """
        if not self._github_app_id:
            raise TrustedIdentityRequiredError(
                "github_app_id is required before writing lifecycle state"
            )
        # Validate state before writing
        _validate_hidden_state(state)

        body = _build_hidden_state_body(visible_markdown, state)
        if len(body.encode("utf-8")) > _MAX_COMMENT_BODY_BYTES:
            raise GitHubStateProxyError("comment body exceeds max size")

        existing = await self.find_trusted_lifecycle_comment(repo, pr_number)
        if expected_comment is not None:
            observed = ((existing.get("id"), existing.get("body"))
                        if existing is not None else (None, None))
            if observed != expected_comment:
                raise GitHubStateProxyError("Lifecycle comment changed before persistence")
        if existing is not None:
            comment_id = existing.get("id")
            if isinstance(comment_id, int):
                await self.update_comment(repo, comment_id, body)
                return existing
        # Create new comment
        result = await self.post_issue_comment(repo, pr_number, body)
        return result

    def _evidence_repo_guard(self, repo: str) -> None:
        if (not self.config.auth_valid or repo not in self.config.github_repos
                or repo not in self.config.active_repos or not self._github_app_id):
            raise GitHubStateProxyError("Evidence repository is not authorized")

    @staticmethod
    def _evidence_envelope(comment: dict) -> dict | None:
        marker = "<!-- deploy-approval-evidence:v1\n"
        body = comment.get("body", "")
        if not isinstance(body, str) or marker not in body:
            return None
        # Exactly one marker and one terminal envelope are accepted. Visible
        # Markdown may precede the hidden document; it never contributes data.
        if body.count(marker) != 1 or not body.endswith("\n-->"):
            raise GitHubStateProxyError("Malformed trusted evidence envelope")
        start = body.index(marker) + len(marker)
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        try:
            data = json.loads(body[start:-len("\n-->")], object_pairs_hook=unique_object)
            if not isinstance(data, dict):
                return None
            kind = data.get("kind")
            common = {"repo", "pr_number", "kind", "key", "sha256", "type"}
            if (not isinstance(data.get("repo"), str) or not data["repo"]
                    or type(data.get("pr_number")) is not int or data["pr_number"] <= 0
                    or kind not in {"round", "operation", "attachment"}
                    or not isinstance(data.get("key"), str) or not data["key"]
                    or len(data["key"]) > 512
                    or not isinstance(data.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", data["sha256"]) is None):
                raise ValueError("invalid evidence envelope identity")
            envelope_type = data.get("type")
            if envelope_type == "record":
                if set(data) != common | {"data"} or not isinstance(data["data"], str):
                    raise ValueError("invalid single-record envelope")
            elif envelope_type == "part":
                if (set(data) != common | {"index", "data"}
                        or type(data.get("index")) is not int or data["index"] < 0
                        or not isinstance(data.get("data"), str)):
                    raise ValueError("invalid evidence part envelope")
            elif envelope_type == "manifest":
                ids = data.get("parts")
                if (set(data) != common | {"parts"} or not isinstance(ids, list)
                        or not 1 <= len(ids) <= _EVIDENCE_MAX_PARTS
                        or any(type(cid) is not int or cid <= 0 for cid in ids)
                        or len(set(ids)) != len(ids)):
                    raise ValueError("invalid evidence manifest envelope")
            else:
                raise ValueError("unknown evidence envelope type")
            return data
        except ValueError:
            raise GitHubStateProxyError("Malformed trusted evidence envelope")

    @staticmethod
    def _evidence_visible_description(envelope: dict) -> str:
        """Keep internal record comments folded while retaining the v1 marker."""
        return "<details>\n<summary>自动保存的内部证据记录（无需操作）</summary>\n</details>"

    async def ensure_visible_evidence_archive_comments(
        self, repo: str, pr_number: int, limit: int = 10,
    ) -> int:
        """Keep legacy evidence comments read-only while preserving the call contract."""
        self._evidence_repo_guard(repo)
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a non-negative int")
        return 0

    async def retire_legacy_evidence_link_replies(
        self, repo: str, pr_number: int,
        validated_summary_urls: set[str], limit: int = 10,
    ) -> int:
        """Retire old manual-link replies only when Summary has a valid COS URL."""
        if not isinstance(validated_summary_urls, set) or not validated_summary_urls:
            return 0
        self._evidence_repo_guard(repo)
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a non-negative int")
        if limit == 0:
            return 0
        expected_url = f"{self.config.github_api_url.rstrip('/')}/repos/{repo}/issues/{pr_number}"
        replacement_body = "This legacy reply is obsolete. Open the current PR Summary and click the verified COS download link. The link is maintained automatically; no manual command is needed."
        legacy_expiry_prompt = re.compile(
            r"(?:expires\s+within\s+900\s+seconds|"
            r"valid\s+for\s+up\s+to\s+900\s+seconds|"
            r"after\s+expiry|after\s+it\s+expires)",
            re.IGNORECASE,
        )
        old_command = re.compile(r"/evidence_link\s+id=(?:<attachment_id>|[A-Za-z0-9_-]+)")

        async def current_summary_cos_links() -> set[str]:
            pr = await self.get_pr(repo, pr_number)
            state = await self.read_hidden_state(repo, pr_number)
            lifecycle = await self.find_trusted_lifecycle_comment(repo, pr_number)
            if (not isinstance(pr, dict) or pr.get("state") != "open"
                    or pr.get("merged") or pr.get("draft")
                    or not isinstance(state, dict) or not isinstance(lifecycle, dict)
                    or (pr.get("head") or {}).get("sha") != state.get("head_sha")):
                return set()
            body = lifecycle.get("body", "")
            if not isinstance(body, str) or HIDDEN_STATE_MARKER not in body:
                return set()
            visible = body.split(HIDDEN_STATE_MARKER, 1)[0]
            urls = set(re.findall(r"<(https://[^<>\s]+)>", visible))
            bucket = str(getattr(self.config, "cos_bucket", "") or "").strip().lower()
            region = str(getattr(self.config, "cos_region", "") or "").strip().lower()
            expected_host = f"{bucket}.cos.{region}.myqcloud.com" if bucket and region else ""
            valid = set()
            now = int(time.time())
            for url in urls:
                try:
                    parsed = urlsplit(url)
                    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                    def one(name):
                        values = query.get(name, [])
                        return values[0] if len(values) == 1 else ""
                    signature = one("q-signature")
                    signed_time = one("q-sign-time")
                    time_match = re.fullmatch(r"([0-9]+);([0-9]+)", signed_time)
                    signed_params = {part.lower() for part in one("q-url-param-list").split(";") if part}
                    if (parsed.scheme != "https" or parsed.hostname != expected_host
                            or parsed.username is not None or parsed.password is not None
                            or parsed.port is not None or parsed.fragment or not parsed.path.strip("/")
                            or not re.fullmatch(r"[0-9a-fA-F]{40,128}", signature)
                            or len(signature) % 2 or not time_match
                            or one("q-key-time") != signed_time
                            or one("q-sign-algorithm") not in {"sha1", "sha256"}
                            or not one("q-ak")
                            or one("q-ak") != getattr(self.config, "cos_secret_id", "")
                            or "response-content-disposition" not in signed_params
                            or not one("response-content-disposition")):
                        continue
                    start, expiry = map(int, time_match.groups())
                    duration = expiry - start
                    if (not 31536000 <= duration <= 31536120
                            or start > now or expiry <= now):
                        continue
                    valid.add(url)
                except (TypeError, ValueError):
                    continue
            valid.intersection_update(validated_summary_urls)
            return valid

        if not await current_summary_cos_links():
            return 0
        comments = await self.get_issue_comments(repo, pr_number)
        retired = 0
        for listed in comments:
            if retired >= limit:
                break
            body = listed.get("body")
            comment_id = listed.get("id")
            if (not self.is_bot_comment(listed) or listed.get("issue_url") != expected_url
                    or type(comment_id) is not int or not isinstance(body, str)
                    or not legacy_expiry_prompt.search(body)
                    or not old_command.search(body)):
                continue
            fresh = await self.get_comment(repo, comment_id)
            listed_app = listed.get("performed_via_github_app")
            fresh_app = fresh.get("performed_via_github_app") if isinstance(fresh, dict) else None
            if (not isinstance(fresh, dict) or fresh.get("id") != comment_id
                    or fresh.get("issue_url") != expected_url or fresh.get("body") != body
                    or not self.is_bot_comment(fresh)
                    or not isinstance(listed_app, dict) or not isinstance(fresh_app, dict)
                    or fresh_app.get("id") != listed_app.get("id")
                    or not isinstance(listed.get("user"), dict)
                    or not isinstance(fresh.get("user"), dict)
                    or fresh["user"].get("id") != listed["user"].get("id")):
                continue
            fresh_pr = await self.get_pr(repo, pr_number)
            if (fresh_pr.get("state") != "open" or fresh_pr.get("merged")
                    or fresh_pr.get("draft")):
                return retired
            if not await current_summary_cos_links():
                return retired
            fresh = await self.get_comment(repo, comment_id)
            if (not isinstance(fresh, dict) or fresh.get("id") != comment_id
                    or fresh.get("body") != body or fresh.get("issue_url") != expected_url
                    or not self.is_bot_comment(fresh)
                    or not isinstance(fresh.get("user"), dict)
                    or fresh["user"].get("id") != listed["user"].get("id")
                    or (fresh.get("performed_via_github_app") or {}).get("id")
                    != listed_app.get("id")):
                continue
            self._evidence_repo_guard(repo)
            await self.update_comment(repo, comment_id, replacement_body)
            confirmed = await self.get_comment(repo, comment_id)
            if (not isinstance(confirmed, dict) or confirmed.get("id") != comment_id
                    or confirmed.get("issue_url") != expected_url
                    or not self.is_bot_comment(confirmed)
                    or confirmed.get("body") != replacement_body
                    or not isinstance(confirmed.get("user"), dict)
                    or confirmed["user"].get("id") != listed["user"].get("id")
                    or (confirmed.get("performed_via_github_app") or {}).get("id")
                    != listed_app.get("id")
                    or legacy_expiry_prompt.search(confirmed.get("body", ""))
                    or old_command.search(confirmed.get("body", ""))):
                raise GitHubStateProxyError("Legacy evidence reply retirement readback failed")
            retired += 1
        return retired

    @staticmethod
    def _evidence_note_ref(pr_number: int) -> str:
        return f"notes/deploy-approval-evidence-v1/pr-{pr_number}"

    @staticmethod
    def _canonical_evidence_data(data: dict) -> bytes:
        return json.dumps(data, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")

    @staticmethod
    def _same_evidence_operation(kind: str, left: dict, right: dict) -> bool:
        """Ignore retry-only timestamps while comparing immutable business fields."""
        volatile = {
            "created_at", "updated_at", "completed_at", "deleted_at",
            "invalidated_at", "archived_at", "ended_at",
        }
        if kind == "round":
            volatile.add("reason")
        return ({key: value for key, value in left.items() if key not in volatile}
                == {key: value for key, value in right.items() if key not in volatile})

    @classmethod
    def _evidence_note_target(cls, kind: str, key: str, data: dict) -> tuple[bytes, str]:
        raw = cls._canonical_evidence_data({"kind": kind, "key": key, "data": data})
        git_blob_sha = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        return raw, git_blob_sha

    @staticmethod
    def _evidence_note_data(repo: str, pr_number: int, payload: dict) -> tuple[bytes, dict]:
        def unique_object(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("duplicate JSON key")
                result[name] = value
            return result

        if not isinstance(payload, dict):
            raise GitHubStateProxyError("Malformed Git Notes evidence record")
        try:
            envelope = json.loads(json.dumps(payload, ensure_ascii=False),
                                  object_pairs_hook=unique_object)
            required = {"version", "repo", "pr_number", "kind", "key", "head_sha",
                        "round_id", "sha256", "data"}
            if (set(envelope) != required or type(envelope.get("version")) is not int
                    or envelope.get("version") != 1):
                raise ValueError("invalid keys")
            kind, key, data = envelope["kind"], envelope["key"], envelope["data"]
            if (envelope["repo"] != repo or type(envelope["pr_number"]) is not int
                    or envelope["pr_number"] != pr_number
                    or kind not in {"round", "operation", "attachment"}
                    or not isinstance(key, str) or not key or len(key) > 512
                    or not isinstance(data, dict) or data.get("repo") != repo
                    or type(data.get("pr_number")) is not int or data.get("pr_number") != pr_number):
                raise ValueError("invalid identity")
            head_sha, round_id = envelope["head_sha"], envelope["round_id"]
            if (not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", head_sha)
                    or not isinstance(round_id, str)
                    or not re.fullmatch(re.escape(head_sha) + r"-[0-9]+", round_id)
                    or data.get("head_sha") != head_sha or data.get("round_id") != round_id):
                raise ValueError("invalid round binding")
            raw = json.dumps(data, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            if envelope.get("sha256") != digest:
                raise ValueError("digest mismatch")
            return raw, envelope
        except (TypeError, ValueError, KeyError):
            raise GitHubStateProxyError("Git Notes evidence identity or SHA256 is invalid") from None

    async def _read_git_note_evidence_records(self, repo: str, pr_number: int) -> list[dict]:
        ref = self._evidence_note_ref(pr_number)
        try:
            ref_data = await self._github.get_git_ref(repo, ref)
        except GitHubError as exc:
            status = getattr(exc, "status_code", None)
            if status == 404:
                # This PR has no legacy Git Notes ref. Continue reading its
                # complete trusted GitHub comment history in the caller.
                return []
            if status == 403:
                return await self._read_public_git_note_records(repo, pr_number, ref, allow_missing=True)
            raise GitHubStateProxyError("Git Notes evidence ref could not be read") from exc
        target = ref_data.get("object") if isinstance(ref_data, dict) else None
        commit_sha = target.get("sha") if isinstance(target, dict) else None
        if (not isinstance(target, dict) or target.get("type") != "commit"
                or not isinstance(commit_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", commit_sha)):
            raise GitHubStateProxyError("Git Notes evidence ref is malformed")
        try:
            return await self._read_git_note_commit_records(repo, pr_number, commit_sha)
        except GitHubStateProxyError:
            # Some installations can read refs but not commit/tree/blob data.
            # Re-verify the public repository and the same ref anonymously;
            # never reinterpret a failed authenticated read as empty history.
            return await self._read_public_git_note_records(repo, pr_number, ref, allow_missing=False)

    async def _public_git_data(self, repo: str, path: str, *, params: dict | None = None) -> dict:
        reader = getattr(self._github, "get_public_git_data", None)
        if not callable(reader):
            raise GitHubStateProxyError("Public Git Data read interface is unavailable")
        try:
            value = await reader(repo, path, params=params)
        except Exception as exc:
            raise GitHubStateProxyError("Public Git Data read could not be verified") from exc
        if not isinstance(value, dict):
            raise GitHubStateProxyError("Public Git Data response is malformed")
        return value

    async def _read_public_git_note_records(
        self, repo: str, pr_number: int, ref: str, *, allow_missing: bool,
    ) -> list[dict]:
        self._evidence_repo_guard(repo)
        if (type(pr_number) is not int or pr_number <= 0
                or ref != self._evidence_note_ref(pr_number)):
            raise GitHubStateProxyError("Git Notes evidence ref identity is invalid")
        try:
            ref_data = await self._public_git_data(repo, f"git/ref/{ref}")
        except GitHubStateProxyError as exc:
            cause = exc
            while cause is not None and getattr(cause, "status_code", None) != 404:
                cause = cause.__cause__
            if allow_missing and cause is not None:
                return []
            raise GitHubStateProxyError("Git Notes evidence ref could not be verified publicly") from exc
        target = ref_data.get("object") if isinstance(ref_data, dict) else None
        commit_sha = target.get("sha") if isinstance(target, dict) else None
        if (not isinstance(target, dict) or target.get("type") != "commit"
                or not isinstance(commit_sha, str)
                or re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None):
            raise GitHubStateProxyError("Public Git Notes evidence ref is malformed")
        return await self._read_git_note_commit_records(
            repo, pr_number, commit_sha, public=True,
        )

    async def _read_git_note_commit_records(
        self, repo: str, pr_number: int, commit_sha: str, *, public: bool = False,
    ) -> list[dict]:
        try:
            if public:
                commit = await self._public_git_data(repo, f"git/commits/{commit_sha}")
            else:
                commit = await self._github.get_git_commit(repo, commit_sha)
            tree_sha = (commit.get("tree") or {}).get("sha")
            if not isinstance(tree_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
                raise ValueError("invalid tree sha")
            if public:
                tree = await self._public_git_data(
                    repo, f"git/trees/{tree_sha}", params={"recursive": "1"},
                )
            else:
                tree = await self._github.get_git_tree(repo, tree_sha, recursive=True)
            entries = tree.get("tree")
            if tree.get("truncated") is not False or not isinstance(entries, list):
                raise ValueError("truncated or malformed tree")
            result = []
            seen_paths = set()
            seen_identities = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("malformed tree entry")
                path = entry.get("path")
                if (not isinstance(path, str)
                        or not re.fullmatch(r"[0-9a-f]{40}", path)
                        or path in seen_paths or entry.get("type") != "blob"
                        or entry.get("mode") != "100644"
                        or not isinstance(entry.get("sha"), str)
                        or not re.fullmatch(r"[0-9a-f]{40}", entry["sha"])):
                    raise ValueError("unexpected evidence tree entry")
                seen_paths.add(path)
                if public:
                    blob = await self._public_git_data(
                        repo, f"git/blobs/{entry['sha']}"
                    )
                else:
                    blob = await self._github.get_git_blob(repo, entry["sha"])
                if blob.get("encoding") != "base64" or not isinstance(blob.get("content"), str):
                    raise ValueError("malformed evidence blob")
                encoded_blob = re.sub(r"\s+", "", blob["content"])
                raw_blob = base64.b64decode(encoded_blob, validate=True)
                if len(raw_blob) > _MAX_EVIDENCE_DATA_BYTES:
                    raise ValueError("oversized evidence blob")

                def unique_object(pairs):
                    decoded = {}
                    for name, value in pairs:
                        if name in decoded:
                            raise ValueError("duplicate JSON key")
                        decoded[name] = value
                    return decoded

                payload = json.loads(raw_blob, object_pairs_hook=unique_object)
                raw, envelope = self._evidence_note_data(repo, pr_number, payload)
                _, expected_path = self._evidence_note_target(
                    envelope["kind"], envelope["key"], envelope["data"])
                if path != expected_path:
                    raise ValueError("evidence path does not match identity")
                identity = (envelope["kind"], envelope["key"])
                if identity in seen_identities:
                    raise ValueError("duplicate evidence identity in Git Notes")
                seen_identities[identity] = raw
                result.append({"kind": envelope["kind"], "key": envelope["key"],
                               "data": envelope["data"], "sha256": envelope["sha256"]})
            return result
        except GitHubStateProxyError:
            raise
        except GitHubError as exc:
            if getattr(exc, "status_code", None) == 403:
                raise GitHubStateProxyError(
                    "Git Notes evidence read denied: Git Data API permission is insufficient"
                ) from exc
            raise GitHubStateProxyError("Git Notes evidence integrity validation failed") from exc
        except Exception as exc:
            raise GitHubStateProxyError("Git Notes evidence integrity validation failed") from exc

    async def read_evidence_records(self, repo: str, pr_number: int) -> list[dict]:
        """Read Git Notes records and trusted legacy comment records."""
        self._evidence_repo_guard(repo)
        if type(pr_number) is not int or pr_number <= 0:
            raise GitHubStateProxyError("Evidence PR identity is invalid")
        result = await self._read_git_note_evidence_records(repo, pr_number)
        seen = {(row["kind"], row["key"]): row["data"]
                for row in result}
        comments = await self.get_issue_comments(repo, pr_number)
        expected_url = f"{self.config.github_api_url.rstrip('/')}/repos/{repo}/issues/{pr_number}"
        trusted = {c["id"]: c for c in comments
                   if type(c.get("id")) is int and c["id"] > 0
                   and c.get("issue_url") == expected_url and self.is_bot_comment(c)}

        def decode_document(envelope):
            if (envelope.get("repo") != repo or envelope.get("pr_number") != pr_number):
                raise GitHubStateProxyError("Evidence document repository mismatch")
            if (envelope.get("kind") not in {"round", "operation", "attachment"}
                    or not isinstance(envelope.get("key"), str)
                    or not envelope["key"] or len(envelope["key"]) > 512):
                raise GitHubStateProxyError("Evidence document kind or key is invalid")
            try:
                raw = base64.b64decode(envelope["data"], validate=True)
            except (ValueError, KeyError, TypeError):
                raise GitHubStateProxyError("Invalid evidence document encoding")
            if (len(raw) > _MAX_EVIDENCE_DATA_BYTES
                    or hashlib.sha256(raw).hexdigest() != envelope.get("sha256")):
                raise GitHubStateProxyError("Evidence document integrity mismatch")
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                raise GitHubStateProxyError("Invalid evidence document JSON")
            if (not isinstance(data, dict) or data.get("repo") != repo
                    or type(data.get("pr_number")) is not int
                    or data.get("pr_number") != pr_number):
                raise GitHubStateProxyError("Evidence record identity mismatch")
            head_sha, round_id = data.get("head_sha"), data.get("round_id")
            if (not isinstance(head_sha, str)
                    or re.fullmatch(r"[0-9a-f]{40}", head_sha) is None
                    or not isinstance(round_id, str)
                    or re.fullmatch(re.escape(head_sha) + r"-[0-9]+", round_id) is None):
                raise GitHubStateProxyError("Evidence record HEAD/round identity is invalid")
            return raw, data

        def add_record(envelope, comment, raw=None, data=None):
            if raw is None or data is None:
                raw, data = decode_document(envelope)
            key = (envelope["kind"], envelope["key"])
            if key in seen:
                if not self._same_evidence_operation(key[0], seen[key], data):
                    raise GitHubStateProxyError("Conflicting immutable evidence documents")
                return
            seen[key] = data
            result.append({"kind": key[0], "key": key[1], "data": data,
                           "comment_id": comment["id"], "url": comment.get("html_url", ""),
                           "sha256": envelope["sha256"]})

        for comment in trusted.values():
            envelope = self._evidence_envelope(comment)
            if not envelope:
                continue
            if envelope["type"] == "record":
                add_record(envelope, comment)
                continue
            if envelope["type"] != "manifest":
                continue
            ids = envelope["parts"]
            chunks = []
            for index, cid in enumerate(ids):
                part = self._evidence_envelope(trusted.get(cid, {}))
                if (not part or part.get("type") != "part" or part.get("index") != index
                        or any(part.get(k) != envelope.get(k)
                               for k in ("repo", "pr_number", "kind", "key", "sha256"))):
                    raise GitHubStateProxyError("Untrusted or incomplete evidence document")
                try:
                    chunks.append(base64.b64decode(part["data"], validate=True))
                except (ValueError, KeyError, TypeError):
                    raise GitHubStateProxyError("Invalid evidence document encoding")
            raw = b"".join(chunks)
            if (len(raw) > _MAX_EVIDENCE_DATA_BYTES
                    or hashlib.sha256(raw).hexdigest() != envelope.get("sha256")):
                raise GitHubStateProxyError("Evidence document integrity mismatch")
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                raise GitHubStateProxyError("Invalid evidence document JSON")
            if (not isinstance(data, dict) or data.get("repo") != repo
                    or type(data.get("pr_number")) is not int
                    or data.get("pr_number") != pr_number):
                raise GitHubStateProxyError("Evidence record identity mismatch")
            head_sha, round_id = data.get("head_sha"), data.get("round_id")
            if (not isinstance(head_sha, str)
                    or re.fullmatch(r"[0-9a-f]{40}", head_sha) is None
                    or not isinstance(round_id, str)
                    or re.fullmatch(re.escape(head_sha) + r"-[0-9]+", round_id) is None):
                raise GitHubStateProxyError("Evidence record HEAD/round identity is invalid")
            add_record(envelope, comment, raw, data)
        return sorted(result, key=lambda record: (
            str(record.get("data", {}).get("updated_at")
                or record.get("data", {}).get("deleted_at")
                or record.get("data", {}).get("completed_at")
                or record.get("data", {}).get("created_at", "")),
            record.get("comment_id", 0), record["kind"], record["key"],
        ))

    async def write_evidence_record(
        self, repo: str, pr_number: int, kind: str, key: str, data: dict,
    ) -> dict:
        """Persist one immutable record as a trusted, bounded GitHub comment.

        Git Notes remain readable for migration compatibility, but writing an
        evidence record does not require Git Data API write permission.
        """
        self._evidence_repo_guard(repo)
        if (not isinstance(repo, str) or type(pr_number) is not int or pr_number <= 0
                or not isinstance(kind, str) or kind not in {"round", "operation", "attachment"}
                or not isinstance(key, str) or not key or len(key) > 512
                or not isinstance(data, dict) or data.get("repo") != repo
                or type(data.get("pr_number")) is not int or data.get("pr_number") != pr_number):
            raise GitHubStateProxyError("Evidence record identity is invalid")
        head_sha, round_id = data.get("head_sha"), data.get("round_id")
        if (not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", head_sha)
                or not isinstance(round_id, str)
                or not re.fullmatch(re.escape(head_sha) + r"-[0-9]+", round_id)):
            raise GitHubStateProxyError("Evidence record HEAD/round identity is invalid")

        def contains_private_material(value):
            if isinstance(value, dict):
                for name, child in value.items():
                    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
                    sensitive_fragments = (
                        "secret", "authorization", "accesstoken", "githubtoken",
                        "capability", "signedurl", "presignedurl", "downloadurl",
                        "clientkey", "filebytes", "videobytes", "assetbytes",
                        "filebase64", "videobase64", "contentbase64", "objectbody",
                    )
                    if (any(fragment in normalized for fragment in sensitive_fragments)
                            or contains_private_material(child)):
                        return True
            elif isinstance(value, list):
                return any(contains_private_material(child) for child in value)
            elif isinstance(value, str):
                return bool(re.search(
                    r"(?i)(?:q-signature=|x-amz-signature=|x-cos-security-token=|"
                    r"authorization:\s*bearer\s+|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)",
                    value,
                ))
            return False

        if contains_private_material(data):
            raise GitHubStateProxyError("Evidence record contains credential or binary payload material")
        try:
            raw = self._canonical_evidence_data(data)
        except (TypeError, ValueError, UnicodeEncodeError):
            raise GitHubStateProxyError("Evidence record structure is invalid") from None
        if len(raw) > _MAX_EVIDENCE_DATA_BYTES:
            raise GitHubStateProxyError("Evidence snapshot exceeds safe archive capacity")
        digest = hashlib.sha256(raw).hexdigest()

        expected_url = f"{self.config.github_api_url.rstrip('/')}/repos/{repo}/issues/{pr_number}"

        async def assert_writable_pr() -> None:
            self._evidence_repo_guard(repo)
            pr = await self.get_pr(repo, pr_number)
            if (not isinstance(pr, dict) or pr.get("state") != "open"
                    or pr.get("merged") is not False or pr.get("draft") is not False):
                raise GitHubStateProxyError("Evidence records require an Open, unmerged, non-draft PR")
            if kind != "round" and (pr.get("head") or {}).get("sha") != head_sha:
                raise GitHubStateProxyError("Evidence record HEAD no longer matches the PR")

        # Read Notes before comments. A permission failure is not equivalent
        # to an empty ref: fail closed rather than forking an unreadable history.
        existing_records = await self.read_evidence_records(repo, pr_number)
        for record in existing_records:
            if (record.get("kind"), record.get("key")) == (kind, key):
                if not self._same_evidence_operation(kind, record.get("data"), data):
                    raise GitHubStateProxyError("Immutable evidence record conflicts")
                return record

        comments = await self.get_issue_comments(repo, pr_number)
        trusted = {
            item["id"]: item for item in comments
            if type(item.get("id")) is int and item["id"] > 0
            and item.get("issue_url") == expected_url and self.is_bot_comment(item)
        }
        pending: dict[str, dict[int, dict]] = {}
        for comment in trusted.values():
            envelope = self._evidence_envelope(comment)
            if (envelope and envelope.get("type") == "part"
                    and envelope.get("repo") == repo
                    and envelope.get("pr_number") == pr_number
                    and envelope.get("kind") == kind and envelope.get("key") == key):
                digest_value, index = envelope.get("sha256"), envelope.get("index")
                if (not isinstance(digest_value, str)
                        or re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
                        or type(index) is not int or not 0 <= index < _EVIDENCE_MAX_PARTS):
                    raise GitHubStateProxyError("Malformed pending evidence part")
                indexed = pending.setdefault(digest_value, {})
                previous = indexed.get(index)
                if previous is not None and previous != envelope:
                    raise GitHubStateProxyError("Conflicting duplicate evidence parts")
                indexed[index] = envelope

        def decode_complete_parts(part_digest: str, indexed_parts: dict[int, dict]):
            indices = sorted(indexed_parts)
            if not indices or indices != list(range(len(indices))):
                return None
            try:
                chunks = [base64.b64decode(indexed_parts[index]["data"], validate=True)
                          for index in indices]
                if (any(len(chunk) != _EVIDENCE_PART_BYTES for chunk in chunks[:-1])
                        or not 0 < len(chunks[-1]) <= _EVIDENCE_PART_BYTES):
                    return None
                candidate_raw = b"".join(chunks)
                if (len(candidate_raw) > _MAX_EVIDENCE_DATA_BYTES
                        or hashlib.sha256(candidate_raw).hexdigest() != part_digest):
                    return None

                def unique_object(pairs):
                    value = {}
                    for name, item in pairs:
                        if name in value:
                            raise ValueError("duplicate evidence JSON key")
                        value[name] = item
                    return value

                candidate = json.loads(candidate_raw, object_pairs_hook=unique_object)
                if (not isinstance(candidate, dict)
                        or self._canonical_evidence_data(candidate) != candidate_raw
                        or candidate.get("repo") != repo
                        or type(candidate.get("pr_number")) is not int
                        or candidate.get("pr_number") != pr_number):
                    return None
                candidate_head, candidate_round = candidate.get("head_sha"), candidate.get("round_id")
                if (not isinstance(candidate_head, str)
                        or re.fullmatch(r"[0-9a-f]{40}", candidate_head) is None
                        or not isinstance(candidate_round, str)
                        or re.fullmatch(re.escape(candidate_head) + r"-[0-9]+", candidate_round) is None):
                    return None
                return candidate_raw, candidate
            except (ValueError, TypeError, KeyError, UnicodeDecodeError):
                return None

        parts = pending.get(digest, {})
        part_count = (len(raw) + _EVIDENCE_PART_BYTES - 1) // _EVIDENCE_PART_BYTES
        if part_count > _EVIDENCE_MAX_PARTS:
            raise GitHubStateProxyError("Evidence record exceeds safe part count")
        for index, envelope in parts.items():
            expected_chunk = raw[index * _EVIDENCE_PART_BYTES:(index + 1) * _EVIDENCE_PART_BYTES]
            try:
                chunk = base64.b64decode(envelope["data"], validate=True)
            except (ValueError, KeyError, TypeError):
                raise GitHubStateProxyError("Malformed pending evidence part data") from None
            if index >= part_count or chunk != expected_chunk:
                raise GitHubStateProxyError("Pending evidence part does not match immutable record")

        recovered = None
        for pending_digest, indexed_parts in pending.items():
            if pending_digest == digest:
                continue
            candidate = decode_complete_parts(pending_digest, indexed_parts)
            if candidate is None or not self._same_evidence_operation(kind, candidate[1], data):
                raise GitHubStateProxyError(
                    "Incomplete evidence parts cannot be safely matched to this retry"
                )
            if recovered is not None and not self._same_evidence_operation(
                    kind, recovered[1], candidate[1]):
                raise GitHubStateProxyError("Conflicting completed evidence parts")
            recovered = (pending_digest, indexed_parts, candidate)
        if not parts and recovered is not None:
            digest, parts, (raw, data) = recovered[0], recovered[1], recovered[2]
            part_count = len(parts)

        async def publish(envelope: dict) -> dict:
            body = (self._evidence_visible_description(envelope) + "\n\n"
                    + "<!-- deploy-approval-evidence:v1\n"
                    + json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n-->")
            if len(body.encode("utf-8")) > _MAX_COMMENT_BODY_BYTES:
                raise GitHubStateProxyError("Evidence comment exceeds safe body limit")

            found = next((item for item in comments
                          if self.is_bot_comment(item)
                          and item.get("issue_url") == expected_url
                          and self._evidence_envelope(item) == envelope), None)
            if found is None:
                await assert_writable_pr()
                try:
                    found = await self.post_issue_comment(repo, pr_number, body)
                except Exception as post_error:
                    # A timed-out POST may have committed on GitHub. Re-read
                    # before deciding whether retrying could duplicate it.
                    try:
                        latest_comments = await self.get_issue_comments(repo, pr_number)
                    except Exception as read_error:
                        raise GitHubStateProxyError(
                            "Evidence comment POST outcome is unknown; readback unavailable"
                        ) from read_error
                    candidate = next((item for item in latest_comments
                                      if self.is_bot_comment(item)
                                      and item.get("issue_url") == expected_url
                                      and self._evidence_envelope(item) == envelope), None)
                    if candidate is None:
                        raise GitHubStateProxyError(
                            "Evidence comment POST failed and no persisted record was found"
                        ) from post_error
                    comments[:] = latest_comments
                    found = candidate
            if not isinstance(found, dict) or type(found.get("id")) is not int or found["id"] <= 0:
                raise GitHubStateProxyError("Evidence comment write was not confirmed")
            if (found.get("issue_url") != expected_url
                    or not self.is_bot_comment(found)
                    or self._evidence_envelope(found) != envelope):
                raise GitHubStateProxyError("Evidence comment write identity failed")
            comments[:] = [item for item in comments if item.get("id") != found["id"]]
            comments.append(found)
            trusted[found["id"]] = found
            return found

        common = {"repo": repo, "pr_number": pr_number, "kind": kind,
                  "key": key, "sha256": digest}
        expected_stored_envelope = None
        if len(raw) <= _EVIDENCE_SINGLE_RECORD_MAX_BYTES and not parts:
            envelope = {**common, "type": "record",
                        "data": base64.b64encode(raw).decode("ascii")}
            expected_stored_envelope = envelope
            stored_comment = await publish(envelope)
        else:
            part_ids = []
            for index in range(part_count):
                envelope = parts.get(index)
                if envelope is None:
                    start = index * _EVIDENCE_PART_BYTES
                    envelope = {
                        **common, "type": "part", "index": index,
                        "data": base64.b64encode(
                            raw[start:start + _EVIDENCE_PART_BYTES]
                        ).decode("ascii"),
                    }
                    confirmed_part = await publish(envelope)
                else:
                    comment_id = next((cid for cid, item in trusted.items()
                                       if self._evidence_envelope(item) == envelope), None)
                    if comment_id is None:
                        raise GitHubStateProxyError("Pending evidence part is not linked to a trusted comment")
                    confirmed_part = trusted.get(comment_id)
                    if (not isinstance(confirmed_part, dict)
                            or not self.is_bot_comment(confirmed_part)
                            or confirmed_part.get("issue_url") != expected_url
                            or self._evidence_envelope(confirmed_part) != envelope):
                        raise GitHubStateProxyError("Pending evidence part readback failed")
                part_ids.append(confirmed_part["id"])
            manifest = {**common, "type": "manifest", "parts": part_ids}
            expected_stored_envelope = manifest
            stored_comment = await publish(manifest)

        readback = await self.get_comment(repo, stored_comment["id"])
        stored_envelope = self._evidence_envelope(readback) if isinstance(readback, dict) else None
        if (not isinstance(readback, dict) or readback.get("id") != stored_comment["id"]
                or readback.get("issue_url") != expected_url
                or not self.is_bot_comment(readback)
                or stored_envelope != expected_stored_envelope):
            raise GitHubStateProxyError("Evidence comment write readback failed")

        verified = await self.read_evidence_records(repo, pr_number)
        result = next((item for item in verified
                       if item.get("kind") == kind and item.get("key") == key), None)
        if (result is None or result.get("sha256") != digest
                or self._canonical_evidence_data(result.get("data")) != raw):
            raise GitHubStateProxyError("Evidence comment persistence readback failed")
        if type(result.get("comment_id")) is not int:
            raise GitHubStateProxyError("Evidence comment record identity is missing")
        return result

    async def persist_cursor(
        self,
        repo: str,
        pr_number: int,
        comment_id: int,
    ) -> dict | None:
        """Persist cursor to GitHub hidden state without overwriting visible markdown.

        Safely advances last_processed_comment_id without regenerating or
        blanking the existing lifecycle visible markdown.

        Behavior:
        - Freshly finds the trusted lifecycle comment.
        - Parses fresh hidden state from that comment.
        - Preserves the exact existing visible markdown.
        - Sets only last_processed_comment_id = max(old, comment_id).
        - Never lowers the cursor.
        - Never accepts an arbitrary stale caller state dict.
        - Returns fresh state dict, or None on failure.

        This is the ONLY safe way to advance the cursor from the watcher
        without corrupting Controller-persisted business state.
        """
        if not self._github_app_id:
            raise TrustedIdentityRequiredError(
                "github_app_id is required before persisting cursor"
            )
        try:
            comment = await self.find_trusted_lifecycle_comment(repo, pr_number)
            if comment is None:
                return None
            first_comment_id = comment.get("id")
            body = comment.get("body", "")
            if not isinstance(body, str):
                return None

            # Locate hidden-state block to replace only the JSON payload.
            # Preserving body[:idx] byte-for-byte so the visible History
            # markdown is never re-serialised or normalised.
            idx = body.find(HIDDEN_STATE_MARKER)
            if idx < 0:
                return None

            # Parse and validate fresh hidden state from the existing comment.
            fresh_state = _extract_hidden_state(body)
            if fresh_state is None:
                return None
            fresh_state = _validate_hidden_state(fresh_state)

            # Advance cursor without lowering it — strictly increasing only.
            # Reject bool (subclass of int) and non-int cursors outright.
            if not isinstance(comment_id, int) or isinstance(comment_id, bool):
                logger.warning(
                    "persist_cursor %s#%s rejected non-int cursor: %r",
                    repo, pr_number, comment_id,
                )
                return fresh_state
            old_cursor = fresh_state.get("last_processed_comment_id", 0)
            if comment_id <= old_cursor:
                # Cursor already at or past target — silent no-op.
                # Prevents replay of stale commands from lowering the cursor.
                return fresh_state
            fresh_state["last_processed_comment_id"] = comment_id

            # Build the new hidden block and splice it into the existing body.
            hidden_json = json.dumps(
                fresh_state, ensure_ascii=False, separators=(",", ":"),
            )
            if len(hidden_json.encode("utf-8")) > _MAX_HIDDEN_STATE_BYTES:
                raise GitHubStateProxyError(
                    f"hidden state JSON exceeds {_MAX_HIDDEN_STATE_BYTES} bytes"
                )
            new_hidden_block = (
                f"{HIDDEN_STATE_MARKER}{hidden_json}\n-->"
            )
            new_body = body[:idx] + new_hidden_block
            if len(new_body.encode("utf-8")) > _MAX_COMMENT_BODY_BYTES:
                raise GitHubStateProxyError("comment body exceeds max size")

            # Update existing comment in-place.
            # TOCTOU guard (A2-04/A2-05): re-fetch the trusted comment and
            # verify BOTH comment ID and body have not changed since we read
            # them above.  If either differs, new_body built from the stale
            # read would corrupt a concurrently-updated lifecycle comment —
            # abort safely (fail-closed).  GitHub PATCH has no CAS; we rely
            # on the single-writer constraint plus this identity check.
            existing = await self.find_trusted_lifecycle_comment(
                repo, pr_number,
            )
            if existing is not None:
                existing_id = existing.get("id")
                existing_body = existing.get("body", "")
                if (
                    isinstance(existing_id, int)
                    and isinstance(existing_body, str)
                    and existing_id == first_comment_id
                    and existing_body == body
                ):
                    # update_comment raises on HTTP non-2xx / network error;
                    # returns None on HTTP 2xx success.  No exception =>
                    # write succeeded.  Do NOT check result is not None.
                    await self.update_comment(
                        repo, existing_id, new_body,
                    )
                    return fresh_state
            return None
        except TrustedIdentityRequiredError:
            raise
        except Exception as e:
            logger.warning(
                "persist_cursor %s#%s cid=%s: %s",
                repo, pr_number, comment_id, e,
            )
            return None

    async def project_status_label(
        self, repo: str, issue_number: int, hidden_status: str
    ) -> None:
        """Project hidden status onto exactly one status:* label.

        Idempotent runtime self-heal:
        1. Fresh GET current labels
        2. If desired already present → skip create/add entirely
        3. Delete ONLY canonical status:* labels that actually exist on PR
        4. Add desired label only if not present
        5. Preserve every non-status label

        If current labels are already exactly the desired state:
        ZERO label create, ZERO add, ZERO delete.

        Any create/add/remove failure is warning-only; hidden lifecycle state
        is never rolled back and business flow is never blocked.

        Non-status labels (bug/documentation/enhancement, etc.) are never
        touched.
        """
        desired_label = f"{STATUS_PREFIX} {hidden_status}"
        desired_fold = desired_label.casefold()
        allowed_folds = {name.casefold() for name in _ALLOWED_STATUS_LABELS}
        if desired_fold not in allowed_folds:
            logger.warning("unknown status label for %s: %s", hidden_status, desired_label)
            return
        key = (repo, issue_number)
        now = time.monotonic()
        cached = self._status_label_projection.get(key)
        if (cached and cached.get("label") == desired_fold
                and now < float(cached.get("retry_at", 0.0))):
            return
        if cached and cached.get("label") != desired_fold:
            self._status_label_projection.pop(key, None)
        projection_deadline = now + 4.0

        async def bounded(awaitable):
            remaining = projection_deadline - time.monotonic()
            if remaining <= 0:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()
                raise asyncio.TimeoutError
            return await asyncio.wait_for(awaitable, timeout=remaining)

        def remember_failure(reason: str) -> None:
            previous = self._status_label_projection.get(key, {})
            failures = min(8, int(previous.get("failures", 0)) + 1) \
                if previous.get("label") == desired_fold else 1
            delay = min(600.0, 15.0 * (2 ** (failures - 1)))
            self._status_label_projection.pop(key, None)
            self._status_label_projection[key] = {
                "label": desired_fold,
                "failures": failures,
                "retry_at": time.monotonic() + delay,
            }
            while len(self._status_label_projection) > 4096:
                self._status_label_projection.pop(next(iter(self._status_label_projection)))
            logger.warning(
                "status label projection deferred repo=%s pr=%s label=%s reason=%s retry_seconds=%.0f",
                repo, issue_number, desired_label, reason, delay,
            )

        try:
            current_labels = await bounded(self.get_issue_labels(repo, issue_number))
            if (not isinstance(current_labels, list)
                    or any(not isinstance(label, str) for label in current_labels)):
                remember_failure("invalid_issue_labels_response")
                return
        except Exception as exc:
            remember_failure(f"read_{type(exc).__name__}")
            return

        desired_existing = next(
            (label for label in current_labels if label.casefold() == desired_fold),
            None,
        )
        if desired_existing is None:
            try:
                repository_label = await bounded(
                    self._ensure_label_exists(repo, desired_label),
                )
                await bounded(self.add_issue_label(
                    repo, issue_number, repository_label,
                ))
                desired_existing = repository_label
                current_labels = [*current_labels, repository_label]
            except Exception as add_exc:
                # Do not issue a second API request after rate limit/transient
                # failure. The bounded retry begins with a fresh label read, so
                # an add that succeeded before a lost response remains idempotent.
                remember_failure(f"add_{type(add_exc).__name__}")
                return

        # The new status label is now confirmed present. Remove only other
        # Deploy Approval lifecycle labels, using GitHub's actual spelling;
        # arbitrary user labels, including other `status:*` labels, survive.
        removal_failed = False
        for label in current_labels:
            folded = label.casefold()
            if folded in allowed_folds and folded != desired_fold:
                try:
                    await bounded(self.remove_issue_label(
                        repo, issue_number, label,
                    ))
                except Exception as exc:
                    removal_failed = True
                    logger.warning(
                        "status label removal deferred repo=%s pr=%s label=%s reason=%s",
                        repo, issue_number, label, type(exc).__name__,
                    )
        if removal_failed:
            remember_failure("remove_failed")
            return

        self._status_label_projection.pop(key, None)
        self._status_label_projection[key] = {
            "label": desired_fold,
            "failures": 0,
            "retry_at": time.monotonic() + 60.0,
        }
        while len(self._status_label_projection) > 4096:
            self._status_label_projection.pop(next(iter(self._status_label_projection)))

    async def _ensure_label_exists(self, repo: str, label: str) -> str:
        """Return the existing case-insensitive label name or create it."""
        def equivalent_name(labels) -> str | None:
            if not isinstance(labels, list):
                return None
            for item in labels:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if isinstance(name, str) and name.casefold() == label.casefold():
                    return name
            return None

        names = await self._github.list_repository_labels(repo)
        existing = equivalent_name(names)
        if existing is not None:
            return existing
        # Failure is retried in the bounded projection window; that attempt
        # first reads and reuses any label created before the response was lost.
        await self._github.create_repository_label(
            repo, label, "6cc2dc", "Deploy Approval lifecycle status",
        )
        return label
