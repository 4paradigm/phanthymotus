"""GitHub-only state proxy for Deploy Approval (pre-merge validation).

The GitHubStateProxy provides a narrow, read/write boundary for GitHub PR
state. It is the ONLY module that reads/writes the GitHub lifecycle comment
and labels. It MUST NOT import or call any Deploy Approval business logic
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
"""

from __future__ import annotations

import json
import logging
import re
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
    "status: succeeded",
    "status: failed",
})

STATUS_PREFIX = "status:"

# Maximum hidden state JSON size (8 KB)
_MAX_HIDDEN_STATE_BYTES = 8192

# Maximum lifecycle comment body size (64 KB)
_MAX_COMMENT_BODY_BYTES = 65536

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
    }
    extra_keys = set(data.keys()) - allowed_keys
    if extra_keys:
        raise MalformedHiddenStateError(
            f"extra keys not allowed: {', '.join(sorted(extra_keys))}"
        )

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
        if set(dep.keys()) != {"machine", "component_ids", "phase"}:
            raise MalformedHiddenStateError("deployment keys must match the canonical schema")
        machine = dep.get("machine", "")
        if not isinstance(machine, str) or not machine:
            raise MalformedHiddenStateError("deployment must have non-empty machine")
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
    if command.get("kind", "") not in {"", "request_deploy", "approve_deploy", "record_test", "deploy_status", "deploy_help"}:
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
        return visible_markdown.rstrip() + "\n\n" + hidden_block
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
                ip_m = re.search(r"\u00b7\s+\**IP:**\s+`([^`]*)`", line)
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
    """Build the generated visible history section string."""
    lines = [VISIBLE_HISTORY_START_MARKER]
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
    lines.append(VISIBLE_HISTORY_END_MARKER)
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
        before_history = visible[:start]
        after_history = visible[end + len(VISIBLE_HISTORY_END_MARKER):]
        return before_history + new_history + after_history
    else:
        # No existing history section — append one with "### History" heading
        existing_events: list[dict] = []
        new_events = [event] + existing_events
        new_history = _build_history_block(new_events)
        return visible.rstrip() + "\n\n### History\n\n" + new_history + "\n"




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
        f"### Deploy Approval — History Archive #{page_num}",
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
    ) -> dict | None:
        """Write hidden state to the trusted lifecycle comment.

        If no trusted comment exists, creates a new one.
        Returns the comment dict, or None on failure.

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
        if existing is not None:
            comment_id = existing.get("id")
            if isinstance(comment_id, int):
                await self.update_comment(repo, comment_id, body)
                return existing
        # Create new comment
        result = await self.post_issue_comment(repo, pr_number, body)
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
            body = comment.get("body", "")
            if not isinstance(body, str):
                return None

            # Extract existing visible markdown (everything before the marker)
            idx = body.find(HIDDEN_STATE_MARKER)
            if idx < 0:
                return None
            visible_markdown = body[:idx].rstrip()

            # Parse fresh hidden state
            fresh_state = _extract_hidden_state(body)
            if fresh_state is None:
                return None
            fresh_state = _validate_hidden_state(fresh_state)

            # Advance cursor without lowering it
            old_cursor = fresh_state.get("last_processed_comment_id", 0)
            if comment_id > old_cursor:
                fresh_state["last_processed_comment_id"] = comment_id

            # Write back with preserved visible markdown
            await self.write_hidden_state(
                repo, pr_number, visible_markdown, fresh_state,
            )
            return fresh_state
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
        if desired_label not in _ALLOWED_STATUS_LABELS:
            logger.warning("unknown status label for %s: %s", hidden_status, desired_label)
            return
        try:
            # 1. Fresh read of current labels
            current_labels: list[str] = []
            try:
                current_labels = await self.get_issue_labels(repo, issue_number)
            except Exception as read_exc:
                logger.warning(
                    "label fresh read failed for %s#%s: %s (best-effort continuing)",
                    repo, issue_number, read_exc,
                )

            # 2. If desired already exists → skip create/add
            if desired_label in current_labels:
                # 3. Remove only OTHER canonical status:* labels that actually exist
                for label in _ALLOWED_STATUS_LABELS:
                    if label != desired_label and label in current_labels:
                        try:
                            await self.remove_issue_label(repo, issue_number, label)
                        except Exception as rem_exc:
                            logger.warning(
                                "label remove failed for %s#%s %s: %s (best-effort, next reconcile will heal)",
                                repo, issue_number, label, rem_exc,
                            )
                return

            # 4. Desired missing → best-effort ensure + add
            try:
                await self._ensure_label_exists(repo, desired_label)
            except Exception as create_exc:
                logger.warning(
                    "label ensure/create failed for %s#%s %s: %s (continuing)",
                    repo, issue_number, desired_label, create_exc,
                )

            try:
                await self.add_issue_label(repo, issue_number, desired_label)
            except Exception as add_exc:
                logger.warning(
                    "label add failed for %s#%s %s: %s (continuing)",
                    repo, issue_number, desired_label, add_exc,
                )

            # 5. Remove only OTHER canonical status:* labels that actually exist
            for label in _ALLOWED_STATUS_LABELS:
                if label != desired_label and label in current_labels:
                    try:
                        await self.remove_issue_label(repo, issue_number, label)
                    except Exception as rem_exc:
                        logger.warning(
                            "label remove failed for %s#%s %s: %s (best-effort, next reconcile will heal)",
                            repo, issue_number, label, rem_exc,
                        )
        except Exception as e:
            logger.warning(
                "label projection failed for %s#%s: %s",
                repo, issue_number, e,
            )

    async def _ensure_label_exists(self, repo: str, label: str) -> None:
        """Best-effort create a repository label if it does not exist.

        409 Conflict (already exists) is treated as success.
        """
        try:
            await self._github.create_repository_label(
                repo, label, "6cc2dc", "Deploy Approval lifecycle status",
            )
        except GitHubError as exc:
            # 409 Conflict means the label already exists
            msg = str(exc)
            if "409" in msg or "Already exists" in msg or "already_exists" in msg:
                return
            raise
