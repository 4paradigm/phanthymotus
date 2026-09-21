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
from .github_client import GitHubClient, GitHubError

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


def _is_valid_digest_ref(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.match(r"^[A-Za-z0-9._-]+(?::[0-9]+)?/[a-z0-9]+(?:(?:[._]|__+|[-]+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:[._]|__+|[-]+)[a-z0-9]+)*)*@sha256:[0-9a-f]{64}$", value))


def _validate_review_evidence(data: dict) -> dict:
    """Strictly validate review_evidence dict. Returns validated data or raises.

    Exact 9 fields:
      build_comment_id, build_comment_updated_at,
      commit_prefix, resolved_head_sha,
      test_comment_id, test_comment_updated_at,
      code_review_comment_id, code_review_comment_updated_at,
      review_author_id
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
    extra = set(data.keys()) - required
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
        if comp.get("target") not in {"perception", "actucore", "driver"}:
            raise MalformedHiddenStateError(f"invalid component target: {comp.get('target')!r}")
        if not isinstance(comp.get("driver_path"), str):
            raise MalformedHiddenStateError("driver_path must be a string")
        if not isinstance(comp.get("variant"), str):
            raise MalformedHiddenStateError("variant must be a string")
        image_ref = comp.get("image_ref", "")
        if not _is_valid_digest_ref(image_ref):
            raise MalformedHiddenStateError(f"invalid image_ref: {image_ref!r}")
        resolved_platform = comp.get("resolved_platform", "")
        if not isinstance(resolved_platform, str) or not resolved_platform:
            raise MalformedHiddenStateError("resolved_platform must be a non-empty string")
        runtime_id = comp.get("runtime_id", "")
        if runtime_id != "" and (not isinstance(runtime_id, str) or not runtime_id):
            raise MalformedHiddenStateError("runtime_id must be an empty or non-empty string")

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
            if set(h.keys()) != {"component_id", "runtime_id", "running_image", "passed"}:
                raise MalformedHiddenStateError("approve_attempt health keys must match the canonical schema")
            if not isinstance(h.get("component_id", ""), str):
                raise MalformedHiddenStateError("approve_attempt health component_id must be a string")
            if not isinstance(h.get("runtime_id", ""), str):
                raise MalformedHiddenStateError("approve_attempt health runtime_id must be a string")
            if not isinstance(h.get("running_image", ""), str):
                raise MalformedHiddenStateError("approve_attempt health running_image must be a string")
            if not isinstance(h.get("passed"), bool):
                raise MalformedHiddenStateError("approve_attempt health passed must be a bool")
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

        Runtime self-heal: fresh read → create desired if missing → fresh
        verify → add → remove other canonical status:* labels.

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

            # 2. Desired canonical status label missing → best-effort create
            desired_applied = desired_label in current_labels
            if not desired_applied:
                try:
                    await self._ensure_label_exists(repo, desired_label)
                except Exception as create_exc:
                    logger.warning(
                        "label ensure/create failed for %s#%s %s: %s (continuing)",
                        repo, issue_number, desired_label, create_exc,
                    )

            if not desired_applied:
                # Fresh verify after create attempt.
                verify_labels: list[str] = []
                try:
                    verify_labels = await self.get_issue_labels(repo, issue_number)
                except Exception:
                    pass
                desired_applied = desired_label in verify_labels

            # Add desired label only if fresh verification did not confirm it.
            if not desired_applied:
                try:
                    await self.add_issue_label(repo, issue_number, desired_label)
                    desired_applied = True
                except Exception as add_exc:
                    logger.warning(
                        "label add failed for %s#%s %s: %s (continuing)",
                        repo, issue_number, desired_label, add_exc,
                    )

            if desired_applied:
                for label in _ALLOWED_STATUS_LABELS:
                    if label != desired_label:
                        try:
                            await self.remove_issue_label(repo, issue_number, label)
                        except Exception as rem_exc:
                            logger.warning(
                                "label remove failed for %s#%s %s: %s (best-effort, next reconcile will heal)",
                                repo, issue_number, label, rem_exc,
                            )
            else:
                logger.warning(
                    "desired label was not applied for %s#%s; preserving existing status labels",
                    repo, issue_number,
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
