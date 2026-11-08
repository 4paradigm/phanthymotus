"""Deploy Controller — stateless command handler for Deploy Approval.

The Deploy Controller is stateless between commands. Its ONLY persistence is
the GitHub lifecycle comment hidden state, read/written through GitHubStateProxy.

Commands:
  - /request_deploy — PR Author requests deployment for current HEAD
  - /approve_deploy — Machine Owner approves and binds machine
  - /record_test — Overall validation verdict

State machine (hidden state only):
  review-required -> reviewing -> deploy-ready -> deploy-requested
  -> deploying -> testing -> succeeded | failed
"""

from __future__ import annotations

import hashlib
import re
import inspect
import ipaddress

import asyncio
import logging
import time
import copy
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
from zoneinfo import ZoneInfo

from . import comments as comments_mod
from .agent_core_client import (
    AgentCoreClient,
    AgentCoreDeployOutcomeUncertain,
    AgentCoreError,
)
from .case_runner import CaseRunner
from .config import (
    Config, DEFAULT_GITHUB_REPOS, is_evidence_ingress_url,
    is_stable_public_base_url, validate_repository_names, _LOGIN_RE,
)
from .cos_client import CosClient
from .evidence_builder import EvidenceBuilder
from .evidence_models import (
    archive_snapshot, make_round_id, round_id_for_state, trusted_round_numbers, utc_now,
)
from .github_client import GitHubClient, GitHubError
from .github_state_proxy import (
    GitHubStateProxy,
    GitHubStateProxyError,
    _validate_hidden_state,
    _extract_hidden_state,
    _build_hidden_state_body,
    _insert_history_into_visible,
    _count_visible_bytes,
    _truncate_events_to_fit,
    _MAX_VISIBLE_LIFECYCLE_BYTES,
    HISTORY_ARCHIVE_MARKER_PREFIX,
    HISTORY_EVENT_DELIMITER,
    _next_history_archive_page,
    _build_archive_body,
    _parse_visible_history,
    _build_history_block,
    VISIBLE_HISTORY_START_MARKER,
    VISIBLE_HISTORY_END_MARKER,
)
from .models import BuildInfo
from .policy import Policy, PolicyError
from .review_comment_parser import (
    extract_review_evidence,
    ReviewCommentEvidence,
)
from .image_ref import validate_image_ref, get_deploy_platform

logger = logging.getLogger(__name__)

AGENT_CORE_STATUS_PROBE_TIMEOUT_SECONDS = 4.0


class DeployControllerError(Exception):
    pass


class DeployOutcomeUncertain(DeployControllerError):
    pass


class DeployPreflightUnavailable(DeployControllerError):
    pass


def _short(sha: str) -> str:
    return (sha or "")[:7]



def _is_supported_target(target: str) -> bool:
    return target in ("core", "perception", "actucore", "driver")


def _image_repository(ref: str) -> str:
    ref = str(ref or "").strip()
    if not ref:
        return ""
    if "@sha256:" in ref:
        return ref.split("@sha256:", 1)[0]
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    if colon > slash:
        return ref[:colon]
    return ref


def _target_tag_from_image_ref(ref: str) -> str:
    """Extract the tag portion from an image reference.

    Returns empty string if no tag can be resolved (e.g. repo@sha256:...).
    For image:tag returns *tag*.  For repo@sha256: returns "".
    """
    ref = str(ref or "").strip()
    if not ref:
        return ""
    if "@sha256:" in ref:
        return ""
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    if colon > slash:
        return ref[colon + 1:]
    return ""


_CANONICAL_VARIANTS = {"5.11", "6.1"}
_LEGACY_VARIANTS = {
    "jetson-jp5.11": "5.11",
    "jetson-jp6.1": "6.1",
}

MAX_RECENT_APPROVE_ATTEMPTS = 4

_REVIEW_EVIDENCE_IDENTITY_FIELDS = frozenset({
    "build_comment_id",
    "build_comment_updated_at",
    "commit_prefix",
    "resolved_head_sha",
    "test_comment_id",
    "test_comment_updated_at",
    "code_review_comment_id",
    "code_review_comment_updated_at",
    "review_author_id",
})


def _normalize_variant(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in _CANONICAL_VARIANTS:
        return raw
    if raw in _LEGACY_VARIANTS:
        return _LEGACY_VARIANTS[raw]
    raise DeployControllerError(f"unsupported variant: {raw!r}")


def _is_deployable_build(repo: str, build, configured_repos: list[str] | None = None) -> bool:
    target = str(build.target or "").strip()
    if not _is_supported_target(target):
        return False
    if not build.success:
        return False
    if not build.image_tag:
        return False
    # The protected configuration selects the main repository, not its owner.
    # Additional controlled repositories retain the driver-only contract.
    try:
        repos = validate_repository_names(list(DEFAULT_GITHUB_REPOS) if configured_repos is None else configured_repos)
    except ValueError:
        return False
    repo_lower = (repo or "").lower()
    if repo_lower in repos[1:]:
        # Driver family: only driver with non-empty driver_path is deployable
        if target != "driver":
            return False
        driver_path = getattr(build, "driver_path", "") or ""
        if not driver_path.strip():
            return False
        return True
    if repo_lower == repos[0]:
        # Core family: core, perception, and actucore are deployable; driver is NOT
        if target == "driver":
            return False
        if target in ("core", "perception", "actucore"):
            return True
        return False
    # Unknown repository: nothing is deployable
    return False

def _component_id_set(deployments: list[dict]) -> set[str]:
    deployed: set[str] = set()
    for dep in deployments:
        if not isinstance(dep, dict) or dep.get("phase") != "deployed":
            continue
        for cid in dep.get("component_ids", []):
            if isinstance(cid, str) and cid:
                deployed.add(cid)
    return deployed


_REVIEW_ACTIVE_STATUSES = {
    "queued",
    "running",
    "retrying",
    "build_success",
}

_REVIEW_FAILED_STATUSES = {
    "build_failed",
    "cancelled",
    "timeout",
    "error",
}


class DeployController:
    """Stateless Deploy Controller.

    Reads/writes hidden state through GitHubStateProxy.
    No SQLite, no DeploymentStore, no in-memory lifecycle state.
    """

    def __init__(
        self,
        config: Config,
        proxy: GitHubStateProxy,
        policy: Policy,
        github: GitHubClient,
        agent_core_factory: object = None,
    ):
        self.config = config
        self.proxy = proxy
        self.policy = policy
        self.github = github
        self._agent_core_factory = agent_core_factory
        self._core_clients: dict[tuple[str, str], AgentCoreClient] = {}
        self._core_client_fingerprints: dict[tuple[str, str], tuple[str, str, str]] = {}
        self.cos = CosClient(config)
        self._case_runner: CaseRunner | None = None
        self._evidence_lock = asyncio.Lock()
        self.evidence_service = None  # B attaches the shared EvidenceService instance.
        # Retry scheduling only; durable terminal state identifies pending COS
        # after restart. At most five attempts per binding per process.
        self._cos_retries: dict[tuple[str, int, str, int], tuple[int, float]] = {}

    async def aclose(self) -> None:
        for core in self._core_clients.values():
            try:
                await core.aclose()
            except Exception:
                pass
        self._core_clients.clear()
        self._core_client_fingerprints.clear()

    # ── Core helpers ──

    async def _core_for_node(
        self, node_id: str, repo: str, *, timeout: float | None = None,
    ) -> AgentCoreClient:
        repo_key = repo.casefold()
        try:
            machines = self.policy.get_machines(repo=repo)
        except Exception as exc:
            await self._close_core_clients_for_repo(repo_key)
            raise DeployControllerError(
                f"cannot verify current machine policy for {repo}: {type(exc).__name__}"
            ) from exc
        if not isinstance(machines, list):
            await self._close_core_clients_for_repo(repo_key)
            raise DeployControllerError(
                f"cannot verify current machine policy for {repo}"
            )

        current_bindings: dict[tuple[str, str], tuple[str, str, str]] = {}
        for current in machines:
            alias = getattr(current, "alias", "")
            current_node_id = getattr(current, "node_id", "")
            node_host = getattr(current, "node_host", "")
            if not all(isinstance(value, str) and value for value in
                       (alias, current_node_id, node_host)):
                continue
            token = self._agent_core_token(alias)
            credential_fingerprint = self._credential_fingerprint(token)
            if self._agent_core_factory is not None and not credential_fingerprint:
                credential_fingerprint = "injected-factory"
            current_bindings[(repo_key, alias)] = (
                current_node_id, node_host, credential_fingerprint,
            )

        # Policy may refresh machine IPs, node IDs, aliases, or credentials
        # while this controller remains alive. Drop any cached connection whose
        # repository binding no longer matches the current public policy view.
        for key in list(self._core_clients):
            if key[0] != repo_key:
                continue
            if self._core_client_fingerprints.get(key) != current_bindings.get(key):
                await self._close_core_client(key)

        matches = [machine for machine in machines
                   if getattr(machine, "node_id", None) == node_id]
        if len(matches) != 1:
            detail = "not present" if not matches else "ambiguous"
            raise DeployControllerError(
                f"machine node {node_id!r} is {detail} in the current policy for {repo}; no Agent Core request was sent"
            )
        machine = matches[0]
        alias = getattr(machine, "alias", "")
        node_host = getattr(machine, "node_host", "")
        token = self._agent_core_token(alias)
        if not isinstance(node_host, str) or not node_host:
            raise DeployControllerError(
                f"machine {alias!r} has no current node_host in {repo}; no Agent Core request was sent"
            )
        if self._agent_core_factory is None and not token:
            raise DeployControllerError(
                f"no current agent_core_token is configured for machine {alias!r} in {repo}; no Agent Core request was sent"
            )
        self._assert_repo_machine_credential_scope(repo, machine)
        fingerprint = (
            node_id, node_host,
            self._credential_fingerprint(token) or "injected-factory",
        )
        key = (repo_key, alias)
        cached = self._core_clients.get(key)
        if cached is not None and self._core_client_fingerprints.get(key) == fingerprint:
            return cached
        if cached is not None:
            await self._close_core_client(key)

        client = await self._resolve_core_client(
            node_id, repo, machine=machine, token=token, timeout=timeout,
        )
        if any(
            key[0] != repo_key and existing is client
            for key, existing in self._core_clients.items()
        ):
            raise DeployControllerError(
                f"Agent Core client factory reused one connection across repositories for machine {alias!r}; no request was sent"
            )
        # Fail closed if the policy/credential binding changed while the new
        # client was being verified. Never cache or use a stale destination.
        try:
            refreshed = self.policy.get_machines(repo=repo)
        except Exception as exc:
            await self._close_client(client)
            raise DeployControllerError(
                f"cannot confirm current machine policy for {repo}: {type(exc).__name__}"
            ) from exc
        if not isinstance(refreshed, list):
            await self._close_client(client)
            raise DeployControllerError(
                f"cannot confirm current machine policy for {repo}"
            )
        refreshed_matches = [item for item in refreshed
                             if getattr(item, "alias", None) == alias]
        if (len(refreshed_matches) != 1
                or self._machine_connection_fingerprint(refreshed_matches[0]) != fingerprint
                or self._credential_fingerprint(self._agent_core_token(alias))
                != self._credential_fingerprint(token)):
            await self._close_client(client)
            raise DeployControllerError(
                f"machine configuration or credentials changed while verifying {alias!r} in {repo}; no deployment was started"
            )
        self._core_clients[key] = client
        self._core_client_fingerprints[key] = fingerprint
        return client

    async def _resolve_core_client(
        self, node_id: str, repo: str, *, machine=None, token: str = "",
        timeout: float | None = None,
    ) -> AgentCoreClient:
        if self._agent_core_factory is not None:
            client = self._agent_core_factory(node_id)
            if client is not None:
                return client
        if (machine is None or machine.node_id != node_id
                or not isinstance(machine.node_host, str) or not machine.node_host):
            raise DeployControllerError(
                f"machine node {node_id!r} is missing a configured node_host"
            )
        if not token:
            raise DeployControllerError(
                f"no agent_core_token configured for machine {machine.alias!r} in {repo}"
            )
        base_url = f"https://{machine.node_host}:15678"
        # This deployment path keeps the policy-bound legacy HTTPS + token contract.
        core_config = copy.copy(self.config)
        core_config.agent_core_tls_mode = "legacy"
        core = AgentCoreClient(
            core_config,
            base_url,
            node_host=machine.node_host,
            machine_alias=machine.alias,
            access_token=token,
        )
        try:
            if timeout is None:
                await core.verify()
            else:
                await asyncio.wait_for(core.verify(), timeout=timeout)
        except asyncio.TimeoutError as e:
            await core.aclose()
            raise DeployControllerError(
                f"Agent Core {node_id} verification timed out"
            ) from e
        except AgentCoreError as e:
            await core.aclose()
            raise DeployControllerError(
                f"Agent Core {node_id} unreachable: {e}"
            )
        return core

    @staticmethod
    def _credential_fingerprint(token: str | None) -> str:
        if not isinstance(token, str) or not token:
            return ""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _agent_core_token(self, alias: str) -> str:
        tokens = getattr(self.config, "agent_core_tokens", {})
        token = tokens.get(alias) if isinstance(tokens, dict) else None
        return token if isinstance(token, str) and token else ""

    def _assert_repo_machine_credential_scope(self, repo: str, machine) -> None:
        """Reject alias-only credentials when another repo maps that alias elsewhere."""
        active_repos = getattr(self.config, "active_repos", [])
        if not isinstance(active_repos, list):
            raise DeployControllerError("cannot verify active repository machine credential scope")
        for other_repo in active_repos:
            if not isinstance(other_repo, str) or other_repo.casefold() == repo.casefold():
                continue
            try:
                status = self.policy.repo_policy_status(other_repo)
                if not isinstance(status, dict):
                    raise ValueError("invalid repository policy status")
                if not status.get("machines_configured"):
                    # An unconfigured repository has no machine binding to
                    # collide with. Its own get_machines(repo=...) remains
                    # empty and therefore fails closed independently.
                    continue
                other_machines = self.policy.get_machines(repo=other_repo)
            except DeployControllerError:
                raise
            except Exception as exc:
                raise DeployControllerError(
                    f"cannot confirm machine alias credential scope for {repo}: {type(exc).__name__}"
                ) from exc
            if not isinstance(other_machines, list):
                raise DeployControllerError(
                    f"cannot confirm machine alias credential scope for {repo}"
                )
            for other in other_machines:
                if (getattr(other, "alias", "").casefold()
                        != getattr(machine, "alias", "").casefold()):
                    continue
                # The same physical machine may be independently authorized by
                # both repositories. Credentials are keyed by the exact alias,
                # so refuse any alias spelling or endpoint identity drift.
                if (
                    getattr(other, "alias", "") == getattr(machine, "alias", "")
                    and getattr(other, "node_id", "") == getattr(machine, "node_id", "")
                    and getattr(other, "node_host", "") == getattr(machine, "node_host", "")
                ):
                    continue
                raise DeployControllerError(
                    f"machine alias {machine.alias!r} is shared by {repo} and {other_repo} with conflicting machine identity; Agent Core credentials are keyed only by alias"
                )

    def _machine_connection_fingerprint(self, machine) -> tuple[str, str, str]:
        alias = getattr(machine, "alias", "")
        return (
            str(getattr(machine, "node_id", "") or ""),
            str(getattr(machine, "node_host", "") or ""),
            self._credential_fingerprint(self._agent_core_token(alias))
            or ("injected-factory" if self._agent_core_factory is not None else ""),
        )

    async def _close_core_client(self, key: tuple[str, str]) -> None:
        client = self._core_clients.pop(key, None)
        self._core_client_fingerprints.pop(key, None)
        if client is not None:
            await self._close_client(client)

    @staticmethod
    async def _close_client(client) -> None:
        try:
            await client.aclose()
        except Exception:
            pass

    async def _close_core_clients_for_repo(self, repo_key: str) -> None:
        for key in list(self._core_clients):
            if key[0] == repo_key:
                await self._close_core_client(key)

    def _get_case_runner(self) -> CaseRunner:
        if self._case_runner is None:
            self._case_runner = CaseRunner(self.config)
        return self._case_runner

    def _extract_trusted_review_evidence(self, comments: list[dict], *, head_sha: str = ""):
        # Even an injected/incomplete Config must not turn exact login matching
        # into the parser's optional ID-only compatibility path.
        author_id = self.config.review_comment_author_id
        author_login = self.config.review_comment_author_login
        if not isinstance(author_id, str) or not author_id or not isinstance(author_login, str) or not author_login:
            return None
        return extract_review_evidence(comments, author_id, author_login, head_sha=head_sha)

    async def _review_evidence_from_comments(
        self, repo: str, pr_number: int, head_sha: str, comments: list[dict],
        *, recheck_latest_comments: bool = True,
        require_exact_pr_context: bool = False,
        resolved_head: str | None = None,
        expected_review_comment_fingerprints: dict[int, dict] | None = None,
        failure_reason: list[str] | None = None,
    ):
        """Authenticate the real triggering comment and recheck HEAD after awaits.

        Parsed unbound evidence remains available for inspection, but its
        blocking flag forbids readiness, requests and pre-POST revalidation.
        """
        if failure_reason is not None:
            failure_reason.clear()

        def reject(reason: str):
            if failure_reason is not None:
                failure_reason[:] = [reason]
            if evidence is not None:
                evidence.run_binding_valid = False
            return evidence

        evidence = self._extract_trusted_review_evidence(comments, head_sha=head_sha)
        if evidence is not None and evidence.test_results_block_deployment:
            logger.warning("REVIEW_EVIDENCE_BLOCKED TEST_RESULTS_STATE=%s TEST_FAILED=%s REQUEST_RUN_BOUND=%s",
                           evidence.test_status, evidence.test_failed, evidence.run_binding_valid)
        if evidence is None:
            if failure_reason is not None:
                failure_reason[:] = ["review_incomplete_or_ambiguous"]
            return None
        if evidence.request_comment is None or not evidence.run_binding_valid:
            if failure_reason is not None:
                failure_reason[:] = ["review_source_or_run_untrusted"]
            return evidence
        request = evidence.request_comment
        original_evidence = copy.deepcopy(evidence)
        evidence.run_binding_valid = False
        try:
            fresh = await self.proxy.get_comment(repo, int(request["id"]))
            if not isinstance(fresh, dict):
                return reject("review_request_unavailable")
            for key in ("id", "body", "created_at", "updated_at", "user"):
                if fresh.get(key) != request.get(key):
                    return reject("review_request_changed")
            user = fresh["user"]
            pr = await self.proxy.get_pr(repo, pr_number)
            if not isinstance(pr, dict):
                return reject("pr_unavailable")
            if (require_exact_pr_context
                    and not self._review_recovery_pr_matches(
                        pr, repo, pr_number, head_sha,
                    )):
                return reject("pr_identity_or_status_mismatch")
            if (not isinstance(pr.get("head"), dict)
                    or pr["head"].get("sha") != head_sha or pr.get("state") != "open"
                    or pr.get("merged") or pr.get("draft") is True):
                return reject("pr_head_or_status_mismatch")
            author = pr.get("user", {})
            if (str(author.get("id", "")) != str(user["id"])
                    or author.get("login", "").casefold() != user["login"].casefold()):
                permission = await self.proxy.collaborator_permission(repo, user["login"])
                if not Policy.collaborator_can_approve(permission):
                    return reject("review_request_unauthorized")
            if (resolved_head is not None and resolved_head != head_sha):
                return reject("review_commit_resolution_failed")
            if (resolved_head is None
                    and await self._resolve_commit_prefix_for_head(
                        repo, head_sha, evidence.commit_prefix,
                    ) != head_sha):
                return reject("review_commit_resolution_failed")
            # Permission/commit lookups await GitHub too. A newer run, deleted
            # evidence or an edited result during those awaits must revoke this
            # snapshot even when the PR HEAD itself has not changed.
            if recheck_latest_comments:
                latest_comments = await self.proxy.get_issue_comments(repo, pr_number)
                if not isinstance(latest_comments, list):
                    return reject("review_comments_unavailable")
                if expected_review_comment_fingerprints is not None:
                    latest_by_id = {
                        comment.get("id"): comment for comment in latest_comments
                        if isinstance(comment, dict)
                    }
                    for comment_id, expected_fingerprint in expected_review_comment_fingerprints.items():
                        if self._review_recovery_comment_fingerprint(
                            latest_by_id.get(comment_id),
                        ) != expected_fingerprint:
                            return reject("review_comment_changed_during_check")
                if (require_exact_pr_context
                        and self._has_untrusted_newer_review_comment(
                            latest_comments, original_evidence.build_comment_id,
                        )):
                    return reject("untrusted_newer_review_run")
                latest_evidence = self._extract_trusted_review_evidence(latest_comments, head_sha=head_sha)
                if latest_evidence != original_evidence:
                    return reject("review_updated_during_check")
            final_pr = await self.proxy.get_pr(repo, pr_number)
            if (require_exact_pr_context
                    and not self._review_recovery_pr_matches(
                        final_pr, repo, pr_number, head_sha,
                    )):
                return reject("pr_identity_or_status_mismatch")
            if (not isinstance(final_pr, dict)
                    or not isinstance(final_pr.get("head"), dict)
                    or final_pr["head"].get("sha") != head_sha
                    or final_pr.get("state") != "open" or final_pr.get("merged")
                    or final_pr.get("draft") is True):
                return reject("pr_changed_during_check")
        except Exception:
            return reject("temporary_read_failure")
        evidence.head_sha = head_sha
        evidence.run_binding_valid = True
        if evidence.code_review_binding == "LEGACY_CONTIGUOUS_CHAIN_NO_DIRECT_SHA":
            logger.info("CODE_REVIEW_HEAD_BINDING=INDIRECT_LEGACY_CHAIN DIRECT_SHA_PROOF=NO build_comment_id=%s", evidence.build_comment_id)
        return evidence

    @staticmethod
    def _empty_cos() -> dict[str, Any]:
        return {"object_key": "", "sha256": "", "size": 0}

    @staticmethod
    def _review_evidence_snapshot(
        evidence: "ReviewCommentEvidence",
        resolved_head: str,
    ) -> dict:
        """Return the canonical review evidence snapshot."""
        return {
            "build_comment_id": evidence.build_comment_id,
            "build_comment_updated_at": evidence.build_comment_updated_at,
            "commit_prefix": evidence.commit_prefix,
            "resolved_head_sha": resolved_head,
            "test_comment_id": evidence.test_comment_id,
            "test_comment_updated_at": evidence.test_comment_updated_at,
            "code_review_comment_id": evidence.code_review_comment_id,
            "code_review_comment_updated_at": evidence.code_review_comment_updated_at,
            "review_author_id": evidence.review_author_id,
            "test_passed": evidence.test_passed,
            "test_failed": evidence.test_failed,
            "test_skipped": evidence.test_skipped,
        }

    @staticmethod
    def _review_evidence_identity(snapshot: dict) -> dict:
        """Return provenance fields used to identify the reviewed run.

        Test totals are evidence metadata, not a new review run.  Ignoring
        them here keeps old nine-field lifecycle snapshots migration-safe.
        """
        return {
            key: snapshot.get(key)
            for key in _REVIEW_EVIDENCE_IDENTITY_FIELDS
            if key in snapshot
        }

    def _review_recovery_state_is_eligible(
        self, state: dict, current_head: str,
    ) -> bool:
        """Allow review recovery only before any approval/deployment activity."""
        if not isinstance(state, dict) or state.get("status") != "deploy-requested":
            return False
        if state.get("head_sha") != current_head:
            return False
        review = state.get("review_evidence")
        if (not isinstance(review, dict)
                or type(review.get("build_comment_id")) is not int
                or review.get("build_comment_id", 0) <= 0
                or review.get("resolved_head_sha") != current_head
                or type(review.get("code_review_comment_id")) is not int
                or review.get("code_review_comment_id", 0) <= 0
                or not isinstance(state.get("components"), list)
                or not state.get("components")):
            return False
        command = state.get("command")
        if (not isinstance(command, dict)
                or command.get("kind") != "request_deploy"
                or command.get("phase") != "completed"
                or not isinstance(command.get("args"), dict)
                or command.get("args") != {}):
            return False
        request_comment_id = command.get("comment_id")
        cursor = state.get("last_processed_comment_id")
        if (type(request_comment_id) is not int or request_comment_id <= 0
                or type(cursor) is not int or cursor < request_comment_id):
            return False
        try:
            if round_id_for_state(state) != make_round_id(current_head, request_comment_id):
                return False
        except (TypeError, ValueError):
            return False
        if (state.get("deployments") != []
                or state.get("approve_attempts") != []
                or type(state.get("approve_attempts_total")) is not int
                or state.get("approve_attempts_total") != 0
                or state.get("approve_attempts_truncated") is not False
                or state.get("deployment_logs") not in (None, {})
                or state.get("case_results") != {}
                or state.get("test_result") != ""
                or state.get("acceptance_request") is not None
                or state.get("cos") != self._empty_cos()):
            return False
        return True

    @staticmethod
    def _review_recovery_position(comment: dict) -> tuple[datetime, int] | None:
        if (not isinstance(comment, dict) or type(comment.get("id")) is not int
                or comment["id"] <= 0):
            return None
        try:
            created = datetime.fromisoformat(
                str(comment.get("created_at", "")).replace("Z", "+00:00"),
            )
            updated = datetime.fromisoformat(
                str(comment.get("updated_at", "")).replace("Z", "+00:00"),
            )
        except (TypeError, ValueError):
            return None
        if (created.tzinfo is None or created.utcoffset() is None
                or updated.tzinfo is None or updated.utcoffset() is None
                or updated < created):
            return None
        return created, comment["id"]

    def _review_recovery_comment_is_scoped(
        self,
        repo: str,
        pr_number: int,
        comment_id: int,
        comment: dict,
        expected_author_id: str,
        expected_author_login: str,
        *,
        allow_edited: bool = False,
    ) -> bool:
        """Verify comment scope and identity, with edits allowed only by caller."""
        if (type(comment_id) is not int or comment_id <= 0
                or not isinstance(comment, dict)):
            return False
        api_base = self.config.github_api_url.rstrip("/")
        expected_url = f"{api_base}/repos/{repo}/issues/comments/{comment_id}"
        expected_issue_url = f"{api_base}/repos/{repo}/issues/{pr_number}"
        user = comment.get("user")
        return (
            type(comment.get("id")) is int and comment.get("id") == comment_id
            and isinstance(comment.get("body"), str)
            and isinstance(comment.get("created_at"), str)
            and isinstance(comment.get("updated_at"), str)
            and (allow_edited
                 or comment.get("created_at") == comment.get("updated_at"))
            and comment.get("url") == expected_url
            and comment.get("issue_url") == expected_issue_url
            and self._review_recovery_position(comment) is not None
            and isinstance(user, dict)
            and str(user.get("id", "")) == str(expected_author_id)
            and user.get("login") == expected_author_login
        )

    @staticmethod
    def _review_recovery_comment_fingerprint(comment: dict | None) -> dict | None:
        """Capture fields that must remain byte-for-byte stable across a reread."""
        if (not isinstance(comment, dict)
                or DeployController._review_recovery_position(comment) is None
                or not isinstance(comment.get("body"), str)
                or not isinstance(comment.get("user"), dict)):
            return None
        return {
            "id": comment.get("id"),
            "body": comment.get("body"),
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "url": comment.get("url"),
            "issue_url": comment.get("issue_url"),
            "user": copy.deepcopy(comment.get("user")),
        }

    def _has_untrusted_newer_review_comment(
        self, comments: list[dict], frozen_build_comment_id: int,
    ) -> bool:
        """Do not let newer untrusted Review protocol comments expose an older run."""
        from .review_comment_parser import (
            CODE_REVIEW_HEADING,
            ERROR_HEADING,
            PR_REVIEW_MARKER,
            TEST_HEADING,
            _is_review_job_anchor_body,
        )

        trusted_id = self.config.review_comment_author_id
        trusted_login = self.config.review_comment_author_login
        frozen_comment = next((comment for comment in comments
                               if isinstance(comment, dict)
                               and type(comment.get("id")) is int
                               and comment.get("id") == frozen_build_comment_id), None)
        frozen_position = self._review_recovery_position(frozen_comment)
        if frozen_position is None:
            return True
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            body = str(comment.get("body", "") or "")
            is_review_protocol_comment = (
                _is_review_job_anchor_body(body)
                or (PR_REVIEW_MARKER in body and any(
                    heading in body for heading in (
                        TEST_HEADING, CODE_REVIEW_HEADING, ERROR_HEADING,
                    )
                ))
            )
            if not is_review_protocol_comment:
                continue
            user = comment.get("user", {})
            is_trusted = (
                isinstance(user, dict)
                and str(user.get("id", "")) == str(trusted_id)
                and user.get("login") == trusted_login
            )
            if is_trusted:
                continue
            position = self._review_recovery_position(comment)
            try:
                updated = datetime.fromisoformat(
                    str(comment.get("updated_at", "")).replace("Z", "+00:00"),
                )
            except (TypeError, ValueError):
                updated = None
            if (position is None
                    or comment.get("id", 0) > frozen_build_comment_id
                    or position > frozen_position
                    or (updated is not None and updated.tzinfo is not None
                        and updated.utcoffset() is not None
                        and updated > frozen_position[0])):
                return True
        return False

    @staticmethod
    def _review_recovery_pr_matches(
        pr: dict, repo: str, pr_number: int, expected_head: str,
    ) -> bool:
        if (not isinstance(pr, dict) or type(pr.get("number")) is not int
                or pr.get("number") != pr_number or pr.get("state") != "open"
                or pr.get("merged") is not False or pr.get("draft") is not False):
            return False
        base = pr.get("base")
        base_repo = base.get("repo") if isinstance(base, dict) else None
        if (not isinstance(base_repo, dict)
                or not isinstance(base_repo.get("full_name"), str)
                or base_repo["full_name"].casefold() != repo.casefold()):
            return False
        head = pr.get("head")
        sha = head.get("sha") if isinstance(head, dict) else None
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            return False
        user = pr.get("user")
        if (not isinstance(user, dict) or not str(user.get("id", "")).isdigit()
                or not isinstance(user.get("login"), str) or not user["login"]):
            return False
        return sha == expected_head

    async def _review_recovery_candidate(
        self,
        repo: str,
        pr_number: int,
        state: dict,
        expected_head: str,
        *,
        final_recheck: bool = True,
    ) -> tuple[str, dict | None]:
        """Return a fully revalidated newer Review Run, or a fail-closed reason."""
        if not self._review_recovery_state_is_eligible(state, expected_head):
            return "lifecycle_not_safely_recoverable", None
        try:
            comments = await self.proxy.get_issue_comments(repo, pr_number)
        except Exception:
            return "temporary_read_failure", None
        if not isinstance(comments, list):
            return "temporary_read_failure", None

        from .commands import _iter_top_level_command_lines, parse_command
        from .review_comment_parser import (
            _parse_review_request,
            extract_latest_review_job_anchor,
        )
        cursor = state["last_processed_comment_id"]
        for comment in comments:
            if (not isinstance(comment, dict) or type(comment.get("id")) is not int
                    or comment["id"] <= cursor
                    or not isinstance(comment.get("body"), str)):
                continue
            if parse_command(comment["body"]).kind == "approve_deploy":
                return "approval_command_pending", None
        author_id = self.config.review_comment_author_id
        author_login = self.config.review_comment_author_login
        if (not isinstance(author_id, str) or not author_id
                or not isinstance(author_login, str) or not author_login):
            return "review_agent_identity_unconfigured", None
        saved_review = state["review_evidence"]
        old_build_id = saved_review["build_comment_id"]
        if self._has_untrusted_newer_review_comment(comments, old_build_id):
            return "untrusted_newer_review_run", None

        latest_anchor = extract_latest_review_job_anchor(
            comments, author_id, author_login,
        )
        if latest_anchor is None:
            return "no_trusted_review_run", None
        review_comments = {
            comment["id"]: comment for comment in comments
            if isinstance(comment, dict) and type(comment.get("id")) is int
        }
        old_comment = review_comments.get(old_build_id)
        if latest_anchor.comment_id == old_build_id:
            if (old_comment is not None
                    and old_comment.get("updated_at")
                    == saved_review.get("build_comment_updated_at")):
                return "no_new_review_run", None
            return "frozen_review_comment_changed", None
        if latest_anchor.state == "reviewing":
            return "latest_review_still_running", None
        if latest_anchor.state == "terminal":
            return "latest_review_failed", None
        if latest_anchor.state != "build-succeeded":
            return "latest_review_state_ambiguous", None
        if old_comment is None:
            return "frozen_review_origin_unverifiable", None

        old_anchor = extract_latest_review_job_anchor(
            [old_comment], author_id, author_login,
        )
        old_position = self._review_recovery_position(old_comment)
        new_anchor_comment = review_comments.get(latest_anchor.comment_id)
        new_position = self._review_recovery_position(new_anchor_comment)
        if (old_anchor is None or old_position is None or new_position is None
                or old_anchor.comment_id != old_build_id
                or old_anchor.state != "build-succeeded"
                or old_anchor.commit_prefix != saved_review.get("commit_prefix")
                or old_comment.get("updated_at") != saved_review.get("build_comment_updated_at")
                or new_anchor_comment.get("updated_at") != latest_anchor.updated_at
                or latest_anchor.comment_id <= max(
                    state["command"]["comment_id"], old_build_id,
                )
                or not old_position < new_position):
            return "review_run_identity_mismatch", None

        try:
            pr = await self.proxy.get_pr(repo, pr_number)
        except Exception:
            return "temporary_read_failure", None
        pr_head = pr.get("head", {}).get("sha") if isinstance(pr, dict) else None
        if pr_head != expected_head:
            return "head_changed", None
        if not self._review_recovery_pr_matches(pr, repo, pr_number, expected_head):
            return "pr_identity_or_status_mismatch", None

        request_id = state["command"]["comment_id"]
        try:
            original_request = await self.proxy.get_comment(repo, request_id)
        except Exception:
            return "temporary_read_failure", None
        original_body = original_request.get("body", "") if isinstance(original_request, dict) else ""
        request_lines = [
            line for line in _iter_top_level_command_lines(original_body)
            if re.match(r"^/request_deploy(?:\s|$)", line, re.I)
        ] if isinstance(original_body, str) else []
        parsed_original = parse_command(original_body) if isinstance(original_body, str) else None
        pr_author = pr.get("user", {})
        if (not isinstance(original_request, dict)
                or type(original_request.get("id")) is not int
                or original_request.get("id") != request_id
                or not request_lines or len(request_lines) != 1
                or parsed_original is None or parsed_original.kind != "request_deploy"
                or self.proxy.is_bot_comment(original_request)
                or original_request.get("url")
                != f"{self.config.github_api_url.rstrip('/')}/repos/{repo}/issues/comments/{request_id}"
                or original_request.get("issue_url")
                != f"{self.config.github_api_url.rstrip('/')}/repos/{repo}/issues/{pr_number}"
                or original_request.get("created_at") != original_request.get("updated_at")
                or self._review_recovery_position(original_request) is None
                or not isinstance(original_request.get("user"), dict)
                or str(original_request["user"].get("id", "")) != str(pr_author.get("id", ""))
                or not isinstance(original_request["user"].get("login"), str)
                or not isinstance(pr_author.get("login"), str)
                or original_request["user"]["login"].casefold()
                != pr_author["login"].casefold()):
            return "original_request_identity_unverifiable", None
        original_request_position = self._review_recovery_position(original_request)
        if (original_request_position is None
                or not original_request_position < new_position):
            return "review_run_order_mismatch", None

        try:
            resolved_head = await self.github.resolve_commit_sha(
                repo, latest_anchor.commit_prefix,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code in {400, 404, 422}:
                return "review_commit_identity_mismatch", None
            return "temporary_read_failure", None
        if not isinstance(resolved_head, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved_head):
            return "review_commit_identity_mismatch", None
        if resolved_head != expected_head:
            return "review_head_mismatch", None

        evidence = self._extract_trusted_review_evidence(comments, head_sha=expected_head)
        if evidence is None:
            return "latest_review_incomplete_or_ambiguous", None
        if (not evidence.run_binding_valid
                or evidence.build_comment_id != latest_anchor.comment_id
                or evidence.review_author_id != author_id
                or evidence.review_author_login != author_login
                or evidence.commit_prefix != latest_anchor.commit_prefix
                or evidence.test_results_block_deployment):
            return "review_source_or_matrix_untrusted", None
        request_comment = evidence.request_comment
        parsed_request = _parse_review_request(
            str((request_comment or {}).get("body", "")),
        )
        request_position = self._review_recovery_position(request_comment)
        if (parsed_request is None or not parsed_request.get("valid")
                or not isinstance(request_comment, dict)
                or type(request_comment.get("id")) is not int
                or request_comment.get("id") <= request_id
                or request_position is None
                or not original_request_position < request_position
                or not request_position < new_position):
            return "review_request_identity_mismatch", None

        if (not evidence.test_comment_id
                and not (evidence.skip_tests_requested and evidence.test_skipped)):
            return "review_test_evidence_incomplete", None
        if (request_comment.get("created_at") != request_comment.get("updated_at")
                or request_comment.get("id") not in review_comments
                or not isinstance(request_comment.get("user"), dict)):
            return "review_request_was_edited_or_mismatched", None

        build_comment = review_comments.get(evidence.build_comment_id)
        if (not isinstance(build_comment, dict)
                or build_comment.get("id") != latest_anchor.comment_id
                or build_comment.get("created_at") != evidence.build_comment_created_at
                or build_comment.get("updated_at") != evidence.build_comment_updated_at
                or build_comment.get("updated_at") != latest_anchor.updated_at
                or self._review_recovery_position(build_comment) is None):
            return "review_build_identity_or_time_mismatch", None

        immutable_result_sources = [
            (evidence.code_review_comment_id, evidence.code_review_created_at,
             evidence.code_review_comment_updated_at),
        ]
        if evidence.test_comment_id:
            immutable_result_sources.append((
                evidence.test_comment_id, evidence.test_comment_created_at,
                evidence.test_comment_updated_at,
            ))
        elif not (evidence.skip_tests_requested and evidence.test_skipped):
            return "review_test_evidence_incomplete", None
        for comment_id, created_at, updated_at in immutable_result_sources:
            listed = review_comments.get(comment_id)
            if (not isinstance(listed, dict)
                    or listed.get("created_at") != created_at
                    or listed.get("updated_at") != updated_at
                    or created_at != updated_at):
                return "review_result_was_edited_or_mismatched", None

        scoped_ids = [
            (request_comment["id"], request_comment["user"].get("id"),
             request_comment["user"].get("login"), False),
            (old_build_id, author_id, author_login, True),
            (evidence.build_comment_id, author_id, author_login, True),
        ]
        scoped_ids.extend(
            (comment_id, author_id, author_login, False)
            for comment_id, _, _ in immutable_result_sources
        )
        for comment_id, expected_id, expected_login, allow_edited in scoped_ids:
            listed = review_comments.get(comment_id)
            if (not isinstance(listed, dict)
                    or not self._review_recovery_comment_is_scoped(
                        repo, pr_number, comment_id, listed,
                        str(expected_id), expected_login,
                        allow_edited=allow_edited,
                    )):
                return "review_comment_identity_mismatch", None

        prior_result_sources = [
            (saved_review.get("code_review_comment_id"),
             saved_review.get("code_review_comment_updated_at")),
        ]
        if saved_review.get("test_comment_id"):
            prior_result_sources.append((
                saved_review.get("test_comment_id"),
                saved_review.get("test_comment_updated_at"),
            ))
        for comment_id, persisted_updated_at in prior_result_sources:
            prior = review_comments.get(comment_id)
            if (type(comment_id) is not int or not isinstance(prior, dict)
                    or prior.get("updated_at") != persisted_updated_at
                    or prior.get("created_at") != persisted_updated_at
                    or not self._review_recovery_comment_is_scoped(
                        repo, pr_number, comment_id, prior, author_id, author_login,
                    )):
                return "frozen_review_result_was_edited_or_mismatched", None

        expected_comment_fingerprints = {}
        for comment_id in {
            request_comment["id"], old_build_id, evidence.build_comment_id,
            evidence.code_review_comment_id,
            *([evidence.test_comment_id] if evidence.test_comment_id else []),
        }:
            fingerprint = self._review_recovery_comment_fingerprint(
                review_comments.get(comment_id),
            )
            if fingerprint is None:
                return "review_comment_identity_mismatch", None
            expected_comment_fingerprints[comment_id] = fingerprint

        validation_reason: list[str] = []
        validated = await self._review_evidence_from_comments(
            repo, pr_number, expected_head, comments,
            recheck_latest_comments=final_recheck,
            require_exact_pr_context=True,
            resolved_head=resolved_head,
            expected_review_comment_fingerprints=expected_comment_fingerprints,
            failure_reason=validation_reason,
        )
        if validated is None:
            return (validation_reason[0] if validation_reason
                    else "latest_review_incomplete_or_ambiguous"), None
        if not validated.run_binding_valid or validated.head_sha != expected_head:
            reason = validation_reason[0] if validation_reason else "review_source_or_run_untrusted"
            if reason in {
                "temporary_read_failure", "review_request_unavailable",
                "pr_unavailable", "review_comments_unavailable",
                "review_commit_resolution_failed",
            }:
                return "temporary_read_failure", None
            if reason == "review_updated_during_check":
                return reason, None
            return reason, None
        evidence = validated

        builds = self._review_build_infos(repo, evidence)
        try:
            components = await self._build_component_snapshot(
                repo, pr_number, expected_head, builds,
            )
        except Exception:
            return "review_component_identity_unavailable", None
        if (not components or any(
                item.get("image_ref") != item.get("review_image_tag")
                or item.get("resolved_platform") != get_deploy_platform()
                for item in components)):
            return "review_component_matrix_or_image_mismatch", None
        snapshot = self._review_evidence_snapshot(evidence, resolved_head)
        fingerprint = self._review_recovery_fingerprint(
            evidence, snapshot, components, expected_comment_fingerprints,
        )
        return "ready", {
            "evidence": evidence,
            "snapshot": snapshot,
            "components": components,
            "builds": builds,
            "fingerprint": fingerprint,
            "head_sha": expected_head,
        }

    @staticmethod
    def _review_recovery_fingerprint(
        evidence: "ReviewCommentEvidence", snapshot: dict,
        components: list[dict],
        comment_fingerprints: dict[int, dict],
    ) -> dict:
        request = evidence.request_comment or {}
        user = request.get("user", {}) if isinstance(request, dict) else {}
        return {
            "snapshot": copy.deepcopy(snapshot),
            "request": {
                "id": request.get("id"),
                "body": request.get("body"),
                "created_at": request.get("created_at"),
                "updated_at": request.get("updated_at"),
                "url": request.get("url"),
                "issue_url": request.get("issue_url"),
                "user_id": user.get("id") if isinstance(user, dict) else None,
                "user_login": user.get("login") if isinstance(user, dict) else None,
            },
            "build_created_at": evidence.build_comment_created_at,
            "test_created_at": evidence.test_comment_created_at,
            "code_review_created_at": evidence.code_review_created_at,
            "code_review_text": evidence.code_review_text,
            "code_review_binding": evidence.code_review_binding,
            "skip_tests_requested": evidence.skip_tests_requested,
            "builds": [
                (build.target, build.driver_path, build.variant,
                 build.success, build.image_tag)
                for build in evidence.builds
            ],
            "components": DeployController._canonical_component_snapshot(components),
            "comments": {
                comment_id: copy.deepcopy(comment_fingerprints[comment_id])
                for comment_id in sorted(comment_fingerprints)
            },
        }

    async def _recover_deploy_requested_review(
        self, repo: str, pr_number: int, state: dict, current_head: str,
    ) -> dict:
        """Supersede an untouched frozen request after a verified newer Review Run."""
        status, candidate = await self._review_recovery_candidate(
            repo, pr_number, state, current_head, final_recheck=False,
        )
        if status != "ready" or candidate is None:
            if status not in {"no_new_review_run", "lifecycle_not_safely_recoverable"}:
                logger.debug(
                    "review recovery deferred %s#%s reason=%s",
                    repo, pr_number, status,
                )
            return state

        evidence = candidate["evidence"]
        next_state = copy.deepcopy(state)
        review_comment_ids = [
            evidence.build_comment_id,
            evidence.code_review_comment_id,
            evidence.request_comment["id"],
        ]
        if evidence.test_comment_id:
            review_comment_ids.append(evidence.test_comment_id)
        cursor = max(
            int(state.get("last_processed_comment_id", 0) or 0),
            *review_comment_ids,
        )
        next_state["status"] = "deploy-ready"
        next_state["review_evidence"] = candidate["snapshot"]
        next_state["components"] = candidate["components"]
        next_state["last_processed_comment_id"] = cursor
        next_state["round_id"] = make_round_id(
            current_head, evidence.build_comment_id,
        )
        next_state["round_started_at"] = utc_now()
        next_state["command"] = {
            "comment_id": cursor,
            "kind": "",
            "phase": "completed",
            "args": {},
        }
        # Consume commands posted before the complete Review result, while
        # preserving monotonicity. The next deploy request must follow the
        # newest reviewed comment. The history-aware writer archives the
        # frozen request once and links it to the existing chain.
        event = {
            "event": "New trusted Review Run invalidated the frozen deployment request",
            "lifecycle": "`deploy-requested` → `deploy-ready`",
            "review_build_comment_id": evidence.build_comment_id,
            "review_code_comment_id": evidence.code_review_comment_id,
            "timestamp": comments_mod.beijing_now_str(),
        }

        async def guard_latest_review() -> bool:
            guard_status, latest = await self._review_recovery_candidate(
                repo, pr_number, state, current_head,
            )
            return (
                guard_status == "ready" and latest is not None
                and latest["fingerprint"] == candidate["fingerprint"]
            )

        machine_groups = self._get_machine_groups_for_components(
            candidate["components"], repo,
        )
        markdown = comments_mod.deploy_ready(
            repo, pr_number, current_head, candidate["builds"],
            review_evidence=candidate["snapshot"],
            machine_groups=machine_groups,
            components=candidate["components"],
        )
        try:
            written = await self._write_lifecycle_with_history(
                repo, pr_number, next_state, markdown, event=event,
                expected_lifecycle_state=state, write_guard=guard_latest_review,
                archive_reason=(
                    "New trusted Review Run invalidated frozen deployment request"
                ),
            )
        except Exception:
            logger.warning("review recovery write failed %s#%s", repo, pr_number)
            return state
        if not written:
            return state
        await self._post_deploy_ready_notice(
            repo, pr_number, head_sha=current_head,
            round_id=round_id_for_state(next_state),
            review_evidence=candidate["snapshot"],
            components=candidate["components"],
        )
        return next_state

    async def _resolve_commit_prefix_for_head(
        self,
        repo: str,
        fresh_head: str,
        commit_prefix: str,
    ) -> str | None:
        """Resolve commit_prefix to full SHA and verify == fresh_head."""
        if not isinstance(fresh_head, str) or not re.fullmatch(r"[0-9a-f]{40}", fresh_head):
            return None
        if not isinstance(commit_prefix, str) or not re.fullmatch(r"[0-9a-f]{7,40}", commit_prefix):
            return None
        try:
            resolved = await self.github.resolve_commit_sha(repo, commit_prefix)
        except Exception as e:
            logger.warning("resolve_commit_sha %s %s: %s", repo, commit_prefix, e)
            return None
        if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
            logger.warning("resolve_commit_sha returned malformed sha: %r", resolved)
            return None
        if resolved != fresh_head:
            return None
        return resolved

    def _init_hidden_state(
        self,
        *,
        head_sha: str,
        status: str,
        review_evidence: dict | None = None,
        components: list[dict] | None = None,
        deployments: list[dict] | None = None,
    ) -> dict:
        state = {
            "version": 1,
            "head_sha": head_sha,
            "status": status,
            "review_evidence": dict(review_evidence or {}),
            "components": list(components or []),
            "deployments": list(deployments or []),
            "case_results": {},
            "test_result": "",
            "cos": self._empty_cos(),
            "approve_attempts": [],
            "approve_attempts_total": 0,
            "approve_attempts_truncated": False,
            "command": {
                "comment_id": 0,
                "kind": "",
                "phase": "completed",
                "args": {},
            },
            "last_processed_comment_id": 0,
            "round_archive": None,
        }
        self._validate_hidden_state(state)
        return state

    def _record_approve_attempt(self, state: dict, attempt: dict) -> None:
        attempts = list(state.get("approve_attempts", []))
        attempts_total = int(state.get("approve_attempts_total", 0) or 0)
        attempts_total += 1
        attempts.append(attempt)
        if len(attempts) > MAX_RECENT_APPROVE_ATTEMPTS:
            attempts = attempts[-MAX_RECENT_APPROVE_ATTEMPTS:]
        state["approve_attempts"] = attempts
        state["approve_attempts_total"] = attempts_total
        state["approve_attempts_truncated"] = attempts_total > len(attempts)

    @staticmethod
    def _reset_review_lifecycle_state(
        state: dict,
        *,
        head_sha: str,
        status: str,
        review_evidence: dict | None = None,
    ) -> None:
        state["head_sha"] = head_sha
        state["round_id"] = make_round_id(
            head_sha, int(state.get("last_processed_comment_id", 0) or 0),
        )
        state["round_started_at"] = utc_now()
        state.pop("deployment_logs", None)
        state["status"] = status
        state["review_evidence"] = dict(review_evidence or {})
        state["components"] = []
        state["deployments"] = []
        state["approve_attempts"] = []
        state["approve_attempts_total"] = 0
        state["approve_attempts_truncated"] = False
        state["case_results"] = {}
        state["test_result"] = ""
        state.pop("acceptance_request", None)
        state["cos"] = {"object_key": "", "sha256": "", "size": 0}
        state["command"] = {
            "comment_id": int(state.get("last_processed_comment_id", 0) or 0),
            "kind": "",
            "phase": "completed",
            "args": {},
        }

    @staticmethod
    def _canonical_component_snapshot(components: list[dict]) -> list[dict]:
        canonical: list[dict] = []
        for component in components or []:
            if not isinstance(component, dict):
                continue
            canonical.append({
                "component_id": str(component.get("component_id", "") or ""),
                "target": str(component.get("target", "") or ""),
                "driver_path": str(component.get("driver_path", "") or ""),
                "variant": str(component.get("variant", "") or ""),
                "review_image_tag": str(component.get("review_image_tag", "") or ""),
                "image_ref": str(component.get("image_ref", "") or ""),
                "resolved_platform": str(component.get("resolved_platform", "") or ""),
            })
        canonical.sort(key=lambda comp: comp["component_id"])
        return canonical

    @staticmethod
    def _fresh_component_runtime_binding(component: dict) -> dict:
        """Normalize runtime identity for a fresh component snapshot.

        Core is its own runtime and therefore carries a static binding from
        the review snapshot onward.  Other targets are dynamically resolved
        during preflight; a fresh snapshot must not retain an old binding.
        """
        item = dict(component)
        if item.get("target") == "core":
            item["runtime_id"] = "core"
        else:
            item.pop("runtime_id", None)
        return item

    @staticmethod
    def _components_with_preserved_runtime_bindings(
        fresh_components: list[dict],
        old_components: list[dict],
        deployments: list[dict],
    ) -> list[dict] | None:
        deployed_component_ids: set[str] = set()
        for deployment in deployments or []:
            if not isinstance(deployment, dict):
                return None
            if deployment.get("phase") != "deployed":
                continue
            component_ids = deployment.get("component_ids", [])
            if not isinstance(component_ids, list) or not component_ids:
                return None
            for component_id in component_ids:
                if not isinstance(component_id, str) or not component_id:
                    return None
                deployed_component_ids.add(component_id)

        old_runtime_bindings: dict[str, str] = {}
        for component in old_components or []:
            if not isinstance(component, dict):
                return None
            component_id = component.get("component_id", "")
            if not isinstance(component_id, str) or not component_id:
                return None
            runtime_id = component.get("runtime_id")
            if component_id in deployed_component_ids and component.get("target") != "core":
                if not isinstance(runtime_id, str) or not runtime_id:
                    return None
                old_runtime_bindings[component_id] = runtime_id

        rebuilt: list[dict] = []
        for component in fresh_components or []:
            if not isinstance(component, dict):
                return None
            item = dict(component)
            component_id = item.get("component_id", "")
            if not isinstance(component_id, str) or not component_id:
                return None
            item = DeployController._fresh_component_runtime_binding(item)
            if component_id in deployed_component_ids and item.get("target") != "core":
                runtime_id = old_runtime_bindings.get(component_id)
                if not isinstance(runtime_id, str) or not runtime_id:
                    return None
                item["runtime_id"] = runtime_id
            rebuilt.append(item)
        return rebuilt

    @staticmethod
    def _component_semantic_key(component: dict) -> tuple:
        """Stable semantic key for migration compatibility.

        Based ONLY on identity fields, NOT on image_ref/digest/runtime_id.
        """
        return (
            component.get("target", ""),
            component.get("driver_path", ""),
            component.get("variant", ""),
            component.get("review_image_tag", ""),
        )

    def _review_build_infos(self, repo: str, evidence: "ReviewCommentEvidence") -> list[BuildInfo]:
        """Keep deployment candidates while ignoring the legacy CORE summary row.

        ``CORE`` (uppercase) is a non-deployable aggregate row in the legacy
        alignment scenario. It must never be normalized into the deployable
        ``core`` target. Keeping it beside a real component rejected the entire
        snapshot, even when the requested deployable matrix was valid.
        All other rows remain subject to the existing strict target/build
        checks in ``_build_component_snapshot``.
        """
        return [
            BuildInfo(
                target=build.target,
                driver_path=build.driver_path,
                variant=build.variant,
                success=build.success,
                image_tag=build.image_tag,
                deployable=_is_deployable_build(repo, build, self.config.github_repos),
            )
            for build in evidence.builds
            if not (
                str(getattr(build, "target", "") or "").strip() == "CORE"
                and not str(getattr(build, "driver_path", "") or "").strip()
                and not str(getattr(build, "variant", "") or "").strip()
            )
        ]

    async def _build_component_snapshot(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        builds: list[BuildInfo],
    ) -> list[dict] | None:
        """Build component snapshot from trusted Review Agent build evidence.

        No Registry access.  Image references come directly from the trusted
        GitHub comment.  Platform is policy-derived (linux/arm64).
        """
        if not builds:
            return None
        snapshot: list[dict] = []
        seen_semantic_keys: set[tuple] = set()
        for build in builds:
            # A requested matrix is indivisible: never deploy its successful subset.
            if (not build.success or not build.deployable
                    or not _is_deployable_build(repo, build, self.config.github_repos)):
                return None
            review_image_tag = str(build.image_tag or "")
            if not review_image_tag:
                return None
            try:
                validated_tag = validate_image_ref(review_image_tag)
            except ValueError:
                return None
            resolved_platform = get_deploy_platform()
            if not resolved_platform:
                return None
            component_id = hashlib.sha256(
                f"{build.target}|{build.driver_path}|{build.variant}|{validated_tag}".encode()
            ).hexdigest()[:16]
            skey = (
                build.target,
                build.driver_path,
                build.variant,
            )
            if skey in seen_semantic_keys:
                return None
            seen_semantic_keys.add(skey)
            component = {
                "component_id": component_id,
                "target": build.target,
                "driver_path": build.driver_path,
                "variant": build.variant,
                "review_image_tag": validated_tag,
                "image_ref": validated_tag,
                "resolved_platform": resolved_platform,
            }
            if build.target == "core":
                component["driver_path"] = ""
                component["variant"] = ""
                component["runtime_id"] = "core"
            snapshot.append(component)
        return snapshot

    async def _rebind_terminal_cos_if_current(
        self,
        repo: str,
        pr_number: int,
        *,
        expected_head: str,
        expected_terminal_status: str,
        expected_comment_id: int,
        expected_command_kind: str,
        expected_test_result: str = "",
        cos_metadata: dict[str, Any] | None = None,
        markdown: str = "",
        expected_state: dict | None = None,
    ) -> bool:
        fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
        if fresh_state is None:
            return False
        if expected_state is not None and fresh_state != expected_state:
            return False
        if fresh_state.get("head_sha", "") != expected_head:
            return False
        if fresh_state.get("status", "") != expected_terminal_status:
            return False
        cmd = fresh_state.get("command", {})
        if not isinstance(cmd, dict):
            return False
        if cmd.get("comment_id", 0) != expected_comment_id:
            return False
        if cmd.get("kind", "") != expected_command_kind:
            return False
        if cmd.get("phase", "") != "completed":
            return False
        if expected_test_result and fresh_state.get("test_result", "") != expected_test_result:
            return False

        # T4: expected_comment_id must be a valid positive integer (not bool)
        if isinstance(expected_comment_id, bool) or not isinstance(expected_comment_id, int):
            return False
        if expected_comment_id <= 0:
            return False

        # Read-only commands may have advanced the cursor after finalization.
        # A full expected_state snapshot still guards every field during rebind.
        cursor_val = int(fresh_state.get("last_processed_comment_id", 0) or 0)
        if cursor_val < expected_comment_id or (expected_state is None and cursor_val != expected_comment_id):
            return False

        # T4: command.args.result must match expected_test_result
        if expected_test_result:
            cmd_args = cmd.get("args", {})
            if not isinstance(cmd_args, dict):
                cmd_args = {}
            if cmd_args.get("result", "") != expected_test_result:
                return False

        metadata = cos_metadata or {}
        object_key = str(metadata.get("object_key", "") or "")
        sha256_val = str(metadata.get("sha256", "") or "")
        size_val = metadata.get("size", 0)

        if not object_key:
            return False

        # Strict object_key layout validation
        try:
            from .cos_client import CosClient
            CosClient.validate_object_key(
                repo, pr_number, expected_head, object_key,
            )
        except Exception:
            return False

        # SHA256 must be 64-char lowercase hex
        if not sha256_val or not re.fullmatch(r"[0-9a-f]{64}", sha256_val):
            return False

        # Size must be int (not bool) in range 1..10MiB
        if not (isinstance(size_val, int) and not isinstance(size_val, bool)
                and 1 <= size_val <= 10485760):
            return False

        # Existing COS conflict check — do not overwrite different evidence
        existing_cos = fresh_state.get("cos", {}) or {}
        existing_key = str(existing_cos.get("object_key", "") or "")
        if existing_key:
            if existing_key != object_key:
                return False
            if existing_cos.get("sha256", "") != sha256_val:
                return False
            existing_size = existing_cos.get("size", 0)
            if not (isinstance(existing_size, int) and not isinstance(existing_size, bool)
                    and existing_size == size_val):
                return False

        expected_terminal_state = copy.deepcopy(fresh_state)
        fresh_state = copy.deepcopy(fresh_state)
        fresh_state["cos"] = {
            "object_key": object_key,
            "sha256": sha256_val,
            "size": size_val,
        }
        if markdown:
            # Terminal rebind must NOT bypass the history-aware lifecycle writer:
            # a direct write_hidden_state would replace the visible lifecycle and
            # drop the existing visible History (Test recorded, deployments, ...).
            # event=None: refresh terminal rendering + state.cos only — the
            # history-preserving writer carries existing events over verbatim.
            await self._preflight_archive_for_event(repo, pr_number, markdown)
            return await self._write_lifecycle_with_history(
                repo, pr_number, fresh_state, markdown, event=None,
                expected_terminal_state=expected_terminal_state,
            )
        return False  # No lifecycle write was performed.

    def _resolve_component_runtime(
        self,
        drivers: list[dict],
        component: dict,
    ) -> dict[str, str] | None:
        """Resolve the exact Agent Core runtime id for one component.

        Perception: runtime id must be EXACT perception.
        ActuCore: runtime id must be EXACT actucore.
        Driver: exact 1-to-1 match on image repository from Agent Core catalog
        entry image metadata.  running_image is never used for
        resolution — the deploy state is the authoritative reference.

        No fallback, no substring, no fuzzy matching.
        """
        target = str(component.get("target", "") or "")
        if target in {"perception", "actucore"}:
            exact = [
                d for d in drivers
                if isinstance(d, dict)
                and str(d.get("id", "") or "") == target
            ]
            if len(exact) != 1:
                return None
            runtime_id = str(exact[0].get("id") or "")
            if not runtime_id:
                return None
            return {
                "runtime_id": runtime_id,
                "runtime_repo": _image_repository(
                    str(exact[0].get("image", "") or "")
                ),
            }

        if target != "driver":
            return None

        want_repo = _image_repository(str(component.get("image_ref", "") or ""))
        if not want_repo:
            return None

        matches: list[dict] = []
        for driver in drivers:
            if not isinstance(driver, dict):
                continue
            if str(driver.get("category", "") or "").strip() != "driver":
                continue
            # Agent Core entry must have image repository metadata
            driver_image = str(driver.get("image", "") or "").strip()
            if not driver_image:
                continue
            runtime_repo = _image_repository(driver_image)
            if not runtime_repo or runtime_repo != want_repo:
                continue
            matches.append(driver)

        if len(matches) == 1:
            runtime_id = str(matches[0].get("id") or "")
            if not runtime_id:
                return None
            return {
                "runtime_id": runtime_id,
                "runtime_repo": want_repo,
            }

        return None

    # ── Command handlers ──

    async def handle_request_deploy(
        self, repo: str, pr_number: int, comment_id: int,
    ) -> bool:
        try:
            pr_data = await self.proxy.get_pr(repo, pr_number)
            pr_state = pr_data.get("state", "")
            pr_merged = pr_data.get("merged", False)
            if pr_state != "open" or pr_merged or pr_data.get("draft"):
                return True
            state = await self.proxy.read_hidden_state(repo, pr_number)
            comment_author_id, comment_author_login = await self.proxy.comment_identity(
                repo, comment_id
            )
            if not comment_author_id:
                await self._post_error(
                    repo, pr_number,
                    "Cannot verify `/request_deploy` author identity.",
                    comment_id=comment_id,
                )
                return True

            if state is None:
                await self._post_error(
                    repo, pr_number,
                    "No deploy approval lifecycle found. Wait for review lifecycle reconciliation first.",
                    comment_id=comment_id,
                )
                return True
            pr_head = pr_data.get("head", {}).get("sha", "")
            if not isinstance(pr_head, str) or not re.fullmatch(r"[0-9a-f]{40}", pr_head):
                await self._post_error(repo, pr_number, "Cannot verify the current PR HEAD.", comment_id=comment_id)
                return True
            if pr_head != state.get("head_sha", ""):
                await self._supersede_head_drift(
                    repo, pr_number, state, pr_head, comment_id,
                    command_kind="request_deploy",
                )
                return True
            _current_status = state.get("status", "")
            if _current_status in ("deploy-ready", "succeeded", "failed"):
                if comment_id <= state.get("last_processed_comment_id", 0):
                    return True
            elif _current_status in ("deploy-requested", "deploying", "testing"):
                logger.warning(
                    "STALE_COMMAND_IGNORED kind=request_deploy comment_id=%s current_status=%s",
                    comment_id, _current_status,
                )
                await self.proxy.persist_cursor(repo, pr_number, comment_id)
                approval_guidance = await self._approval_command_guidance(
                    repo, pr_number,
                )
                await self._post_command_reply(
                    repo, pr_number, comment_id,
                    f"`/request_deploy` was not applied because this round is already `{_current_status}`. "
                    + approval_guidance,
                    topic="request-deploy-already-processed",
                )
                return True
            else:
                await self._post_command_not_ready(
                    repo, pr_number, _current_status,
                    "Wait for the PR to reach `deploy-ready` status before requesting deployment.",
                    comment_id=comment_id,
                )
                return True
            pr_user = pr_data.get("user", {})
            pr_author_id = str(pr_user.get("id") or "") if isinstance(pr_user, dict) else ""
            if not pr_author_id:
                await self._post_error(
                    repo, pr_number,
                    "Cannot verify PR author identity.",
                    comment_id=comment_id,
                )
                return True
            if comment_author_id != pr_author_id:
                await self._post_error(
                    repo, pr_number,
                    "Only the PR author can use `/request_deploy`.",
                    comment_id=comment_id,
                )
                return True

            # Get build evidence from PR comments
            try:
                comments = await self.github.get_issue_comments(repo, pr_number)
                if not isinstance(comments, list):
                    comments = []
                persisted_build_id = state.get("review_evidence", {}).get("build_comment_id")
                if (comments and type(persisted_build_id) is int
                        and self._has_untrusted_newer_review_comment(
                            comments, persisted_build_id,
                        )):
                    await self._post_error(
                        repo, pr_number,
                        "A newer Review Agent protocol comment is not from the configured trusted "
                        "Review Agent. No deployment request was created.",
                        comment_id=comment_id,
                    )
                    return True
                evidence = await self._review_evidence_from_comments(repo, pr_number, pr_head, comments)
            except Exception as e:
                logger.warning(
                    "request_deploy comment evidence %s#%s: %s",
                    repo, pr_number, e,
                )
                evidence = None

            if evidence is None:
                await self._post_error(
                    repo, pr_number,
                    "No completed review evidence found for this HEAD. "
                    "Wait for Review Agent to complete.",
                    comment_id=comment_id,
                )
                return True

            if not evidence.run_binding_valid:
                await self._post_error(
                    repo, pr_number,
                    "Review request, required build matrix or run provenance is missing or ambiguous. "
                    "No deployment can be requested from unbound review evidence.",
                    comment_id=comment_id,
                )
                return True
            # Convert evidence builds to BuildInfo
            builds = self._review_build_infos(repo, evidence)

            components = await self._build_component_snapshot(
                repo, pr_number, pr_head, builds,
            )
            if components is None:
                await self._post_error(
                    repo, pr_number,
                    "Failed to resolve component snapshot for this HEAD.",
                    comment_id=comment_id,
                )
                return True
            if not components:
                await self._post_error(
                    repo, pr_number,
                    "No deployable builds found for this HEAD.",
                    comment_id=comment_id,
                )
                return True

            # Resolve evidence commit prefix to full SHA
            resolved_head = await self._resolve_review_evidence_for_head(
                repo, pr_head, evidence, pr_number=pr_number,
            )
            if resolved_head is None:
                await self._post_error(
                    repo, pr_number,
                    "Review evidence commit could not be resolved to the current PR HEAD.",
                    comment_id=comment_id,
                )
                return True

            # Use the same canonical snapshot as reconcile and recovery.
            review_evidence_data = self._review_evidence_snapshot(
                evidence, resolved_head,
            )

            # A ready lifecycle is the developer-visible review snapshot. If
            # the latest trusted run changed before this command was handled,
            # consume this request without silently treating the new findings
            # as already reviewed. Reconciliation will publish the new result;
            # direct approval can then proceed from the refreshed deploy-ready state.
            if _current_status == "deploy-ready":
                persisted_evidence = state.get("review_evidence", {})
                persisted_components = state.get("components", [])
                if (self._review_evidence_identity(review_evidence_data)
                        != self._review_evidence_identity(persisted_evidence)
                        or (persisted_components and
                            self._canonical_component_snapshot(components)
                            != self._canonical_component_snapshot(persisted_components))):
                    await self._post_error(
                        repo, pr_number,
                        "Review evidence changed after this `deploy-ready` snapshot. "
                        "Read the latest Review Agent result and wait for lifecycle reconciliation. "
                        "When the current Summary returns to `deploy-ready`, approve directly; `/request_deploy` is optional.",
                        comment_id=comment_id,
                    )
                    return True

            # Determine compatible machine groups
            machine_groups = self._get_machine_groups_for_components(components, repo)

            # Preserve the original round before replacing the active snapshot.
            previous_state = copy.deepcopy(state)
            state = self._init_hidden_state(
                head_sha=pr_head,
                status="deploy-requested",
                review_evidence=review_evidence_data,
                components=components,
            )
            state["round_id"] = make_round_id(pr_head, comment_id)
            state["round_started_at"] = utc_now()
            state["command"] = {
                "comment_id": comment_id,
                "kind": "request_deploy",
                "phase": "completed",
                "args": {},
            }
            state["last_processed_comment_id"] = comment_id

            # Validate state before persisting
            self._validate_hidden_state(state)

            markdown = comments_mod.deploy_requested(
                repo, pr_number, pr_head, components, machine_groups,
            )
            event = {
                "event": "Deployment requested",
                "lifecycle": "`deploy-ready` \u2192 `deploy-requested`",
                "timestamp": comments_mod.beijing_now_str(),
            }
            written = await self._write_lifecycle_with_history(
                repo, pr_number, state, markdown, event=event,
                expected_lifecycle_state=previous_state,
                archive_reason="New deployment request",
            )
            if not written:
                return False
            await self._project_status_best_effort(repo, pr_number, "deploy-requested")
            command_lines = self._machine_notice_lines(
                machine_groups, repo, mention_owners=True,
                test_only=self._is_test_only_pr(pr_data),
            )
            if not command_lines:
                command_lines = [
                    "当前没有适配且可访问的机器，无法生成可执行的 IP 审批命令。"
                    "请检查当前仓库机器配置和 Agent Core 访问。"
                ]
            await self._post_bound_notice(
                repo, pr_number, head_sha=pr_head,
                round_id=round_id_for_state(state), comment_id=comment_id,
                topic="deployment-requested-next-approver",
                expected_command_kind="request_deploy",
                message="\n".join([
                    f"已为 HEAD `{_short(pr_head)}` 的本轮 `{round_id_for_state(state)}` 请求部署审批。",
                    "以下是已匹配机器的完整审批命令及其负责人：",
                    *command_lines,
                ]),
            )
            return True

        except Exception as e:
            logger.error(
                "handle_request_deploy %s#%s: %s", repo, pr_number, e,
            )
            raise

    async def handle_approve_deploy(
        self, repo: str, pr_number: int, comment_id: int,
        machine_alias: str, actor: str, actor_id: str,
    ) -> bool:
        try:
            pr_data = await self.proxy.get_pr(repo, pr_number)
            if pr_data.get("state") != "open" or pr_data.get("merged") or pr_data.get("draft"):
                return True
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if state is None:
                await self._post_error(
                    repo, pr_number,
                    "No trusted lifecycle is available for this HEAD. Wait for `deploy-ready`; then you can approve directly. `/request_deploy` is optional.",
                    comment_id=comment_id,
                )
                return True

            pr_head = pr_data.get("head", {}).get("sha", "")
            if not isinstance(pr_head, str) or not re.fullmatch(r"[0-9a-f]{40}", pr_head):
                await self._post_error(repo, pr_number, "Cannot verify the current PR HEAD.", comment_id=comment_id)
                return True
            if pr_head != state.get("head_sha", ""):
                await self._supersede_head_drift(
                    repo, pr_number, state, pr_head, comment_id,
                )
                return True
            initial_lifecycle_state = copy.deepcopy(state)

            _current_status = state.get("status", "")
            if _current_status == "deploy-ready":
                if comment_id <= state.get("last_processed_comment_id", 0):
                    return True
                if (not state.get("review_evidence") or not state.get("components")
                        or not await self._fresh_review_evidence_matches_state(
                            repo, pr_number, pr_head, state.get("review_evidence", {}),
                            expected_components=state.get("components", []),
                        )):
                    await self._post_error(
                        repo, pr_number,
                        "Current HEAD build or Review Agent evidence is missing, stale, or changed. "
                        "Wait for lifecycle reconciliation to publish trusted `deploy-ready` evidence.",
                        comment_id=comment_id,
                    )
                    return True
                ready_state = copy.deepcopy(state)
                state = self._init_hidden_state(
                    head_sha=pr_head, status="deploy-requested",
                    review_evidence=ready_state.get("review_evidence", {}),
                    components=ready_state.get("components", []),
                )
                state["round_id"] = make_round_id(pr_head, comment_id)
                state["round_started_at"] = utc_now()
                state["round_archive"] = ready_state.get("round_archive")
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "completed",
                    "args": {"machine": "", "actor": actor},
                }
                state["last_processed_comment_id"] = comment_id
                _current_status = "deploy-requested"

            if _current_status == "deploying":
                phase = state.get("command", {}).get("phase")
                if phase == "executing":
                    await self._handle_uncertain_if_needed(repo, pr_number, state)
                elif phase == "uncertain":
                    await self._recover_uncertain_deployment_tick(
                        repo, pr_number, state, pr_data,
                    )
                await self._post_command_reply(
                    repo, pr_number, comment_id,
                    "A deployment is already in progress or its POST outcome is being verified. "
                    "No retry was started; wait for the current deployment status to be confirmed.",
                    topic="deployment-already-in-progress",
                )
                return True

            if _current_status == "deploy-requested":
                pass  # normal path continues below
            elif _current_status in ("testing", "succeeded", "failed"):
                logger.warning(
                    "STALE_COMMAND_IGNORED kind=approve_deploy comment_id=%s current_status=%s",
                    comment_id, _current_status,
                )
                await self.proxy.persist_cursor(repo, pr_number, comment_id)
                await self._post_command_reply(
                    repo, pr_number, comment_id,
                    f"`/approve_deploy` was not started because the current round is already `{_current_status}`.",
                    topic="approve-deploy-already-processed",
                )
                return True
            else:
                await self._post_command_not_ready(
                    repo, pr_number, _current_status,
                    "Wait for trusted Review Agent build and review evidence. Once the lifecycle reaches `deploy-ready`, its Summary will show the current executable machine commands; `/request_deploy` remains optional.",
                    comment_id=comment_id,
                )
                return True

            # Check PR is still valid
            pr_state = pr_data.get("state", "")
            pr_merged = pr_data.get("merged", False)
            if pr_state != "open" or pr_merged:
                await self._post_error(
                    repo, pr_number,
                    "PR is not open. Deploy only when PR is open and unmerged.",
                    comment_id=comment_id,
                )
                return True

            if pr_data.get("draft") is True:
                await self._post_error(
                    repo, pr_number,
                    "Deployment requires PR ready-for-review / non-draft.",
                    comment_id=comment_id,
                )
                return True

            phase = state.get("command", {}).get("phase")
            if phase == "executing":
                await self._handle_uncertain_if_needed(repo, pr_number, state)
                return True
            if phase == "uncertain":
                await self._refresh_uncertain_state(repo, pr_number, state)
                # Recovery is read-only with respect to Agent Core. A later
                # approval may deploy only components proven never attempted.
                return True

            # Resolve machine selector (alias or IPv4) to canonical MachineInfo
            machine_selector = machine_alias
            if not machine_alias:
                remaining_for_selection = [
                    component for component in state.get("components", [])
                    if component.get("component_id") not in _component_id_set(
                        state.get("deployments", []),
                    )
                ]
                groups = self._get_machine_groups_for_components(
                    remaining_for_selection, repo,
                )
                remaining_ids = {
                    component.get("component_id") for component in remaining_for_selection
                    if isinstance(component, dict)
                }
                full_groups = [
                    group for group in groups
                    if remaining_ids and remaining_ids.issubset(set(group.get("component_ids", [])))
                ]
                if len(full_groups) == 1:
                    options = full_groups
                elif len(full_groups) > 1:
                    options = full_groups
                else:
                    options = groups
                if len(options) != 1 or not full_groups:
                    command_lines = self._machine_notice_lines(
                        options, repo, mention_owners=False,
                        test_only=self._is_test_only_pr(pr_data),
                    )
                    if not command_lines:
                        explanation = (
                            "No deployable machine matches the remaining components' target, platform, "
                            "variant, and driver path, or its Agent Core access is unavailable. "
                            "Check this repository's machine policy and machine access configuration."
                        )
                    elif full_groups:
                        explanation = "Several fully compatible machines are available. Choose one explicitly:"
                    else:
                        explanation = (
                            "No single machine covers every remaining component. Choose an adapted "
                            "machine explicitly for partial deployment:"
                        )
                    reply = "\n".join([
                        explanation,
                        *command_lines,
                        "Use one of the complete IP commands above; no machine was selected automatically.",
                    ])
                    await self._post_command_reply(
                        repo, pr_number, comment_id, reply,
                        topic="approve-deploy-machine-selection",
                    )
                    return True
                machine_alias = str(options[0].get("ip", "") or "")

            try:
                machine_map = {item.alias: item for item in self.policy.get_machines(repo=repo)}
                machine = Policy.resolve_machine_selector(machine_alias, machine_map)
            except PolicyError as e:
                deployed_ids = _component_id_set(state.get("deployments", []))
                remaining_components = [
                    component for component in state.get("components", [])
                    if isinstance(component, dict)
                    and component.get("component_id") not in deployed_ids
                ]
                try:
                    available_commands = self._machine_notice_lines(
                        self._get_machine_groups_for_components(remaining_components, repo),
                        repo, mention_owners=False,
                        test_only=self._is_test_only_pr(pr_data),
                    )
                except Exception as exc:
                    logger.warning("compatible machine list unavailable %s#%s exception=%s",
                                   repo, pr_number, type(exc).__name__)
                    available_commands = []
                suggestion = (
                    "\nAvailable compatible commands:\n" + "\n".join(available_commands)
                    if available_commands else "\nNo compatible deployable machine is currently available; check repository machine policy and machine access."
                )
                await self._post_error(
                    repo, pr_number,
                    f"{e} Choose an exact command from the compatible machine list below.{suggestion}",
                    comment_id=comment_id,
                )
                return True

            # Preserve the ORIGINAL selector (IPv4 or alias) from the comment
            # for exact revalidation. Canonical alias is used for state/persistence.
            # Retain the selector from the original comment (empty for automatic selection).
            # Use canonical alias for all state persistence
            machine_alias = machine.alias
            approved_machine_binding = (
                machine.alias, machine.node_id, machine.node_host,
            )

            # Check permissions async
            await self._check_approval_permissions(
                machine, actor, repo,
            )

            # Determine which component_ids are assigned to this machine group
            components = state.get("components", [])
            existing_deployments = state.get("deployments", [])
            existing_deployed = _component_id_set(existing_deployments)

            # Determined from ALL undeployed components — not from compatible IDs.
            undeployed_components = [
                c for c in components
                if c.get("component_id") not in existing_deployed
            ]
            if not undeployed_components:
                await self._post_error(
                    repo, pr_number,
                    f"No remaining components to deploy for machine `{machine_alias}`.",
                    comment_id=comment_id,
                )
                return True


            compatible_remaining_ids = set(
                self._get_component_ids_for_machine(
                    machine_alias, undeployed_components, repo,
                )
            )

            # Multi-machine partial coverage: select only compatible, undeployed components.
            selected_components = [
                c for c in undeployed_components
                if c.get("component_id", "") in compatible_remaining_ids
            ]

            if not selected_components:
                # Selected machine covers zero remaining components.
                if not self._machine_is_deployable(machine):
                    selection_problem = (
                        f"Machine `{machine_alias}` is not deployable because its Agent Core endpoint or access is unavailable."
                    )
                else:
                    selection_problem = (
                        f"Machine `{machine_alias}` does not match any remaining component's target, platform, variant, and driver path."
                    )
                state["status"] = "deploy-requested"
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "completed",
                    "args": {"machine": machine_alias, "actor": actor},
                }
                state["last_processed_comment_id"] = comment_id
                markdown = comments_mod.deploy_requested(
                    repo, pr_number, pr_head,
                    state["components"],
                    self._get_machine_groups_for_components(undeployed_components, repo),
                    gate_note=[
                        selection_problem,
                        "ZERO deploy POST.",
                        "Use one of the complete IP approval commands in this Summary for a compatible machine.",
                    ],
                    deployments=self._enrich_deployments_for_render(state["deployments"], repo),
                )
                event = {
                    "event": f"Machine `{machine_alias}` selected",
                    "machine": machine_alias,
                    "timestamp": comments_mod.beijing_now_str(),
                }
                written = await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                    expected_lifecycle_state=initial_lifecycle_state,
                )
                if not written:
                    return False
                await self._project_status_best_effort(repo, pr_number, "deploy-requested")
                compatible_groups = self._get_machine_groups_for_components(
                    undeployed_components, repo,
                )
                command_lines = self._machine_notice_lines(
                    compatible_groups, repo, mention_owners=False,
                    test_only=self._is_test_only_pr(pr_data),
                )
                await self._post_command_reply(
                    repo, pr_number, comment_id,
                    selection_problem + " "
                    + ("Try one of these commands:\n" + "\n".join(command_lines)
                       if command_lines else "No compatible machine is configured; check this repository's machine policy."),
                    topic="approve-deploy-incompatible-machine",
                )
                return True


            try:
                preflight_written = await self._write_lifecycle_with_history(
                    repo, pr_number, state,
                    self._deployment_progress_markdown(
                        repo, pr_number, state, machine, selected_components, "Approval received; checking",
                    ), event=None,
                    expected_lifecycle_state=initial_lifecycle_state,
                )
                if not preflight_written:
                    return False
                initial_lifecycle_state = copy.deepcopy(state)
                node_id = machine.node_id
                core = await self._core_for_node(node_id, repo)
                preflight = await self._preflight_running_images(core, selected_components)
            except (AgentCoreError, DeployControllerError, TimeoutError, OSError) as exc:
                # This try block contains read-only preflight and no deploy POST.
                safe_reason = type(exc).__name__
                if (isinstance(exc, DeployControllerError)
                        and ("Agent Core credentials are keyed only by alias" in str(exc)
                             or "cannot confirm whether alias" in str(exc)
                             or "cannot confirm machine alias credential scope" in str(exc))):
                    safe_reason = str(exc)
                fresh = await self.proxy.read_hidden_state(repo, pr_number)
                if fresh != state:
                    return False
                state["command"] = {
                    "comment_id": comment_id, "kind": "approve_deploy", "phase": "completed",
                    "args": {"machine": machine_alias, "actor": actor},
                }
                state["last_processed_comment_id"] = comment_id
                markdown = self._deployment_progress_markdown(
                    repo, pr_number, state, machine, selected_components,
                    f"Preflight failed ({safe_reason}); target machine/runtime unavailable or invalid. ZERO deploy POST. Send a NEW approval after recovery.",
                )
                preflight_failure_written = await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown,
                    event={"event": "Preflight failed: ZERO deploy POST", "machine": machine_alias,
                           "ip": machine.node_host, "reason": type(exc).__name__,
                           "timestamp": comments_mod.beijing_now_str()},
                    expected_lifecycle_state=fresh,
                )
                if not preflight_failure_written:
                    return False
                await self._post_command_reply(
                    repo, pr_number, comment_id,
                    f"Deployment did not start because machine `{machine_alias}` failed its read-only preflight ({safe_reason}). "
                    "No deploy POST was sent. Resolve the machine/runtime issue, then use a current complete IP command from the Lifecycle Summary.",
                    topic="deployment-preflight-check-failed",
                )
                return True
            for item in preflight:
                item["component"]["runtime_id"] = item["runtime_id"]
            approve_attempt = {
                "comment_id": comment_id,
                "actor": actor,
                "machine": machine_alias,
                "preflight": [
                    {
                        "component_id": item["component"].get("component_id", ""),
                        "runtime_id": item["runtime_id"],
                        "running_image": item["running_image"],
                    }
                    for item in preflight
                ],
                "outcome": "",
                "health": [],
            }

            # Fresh PR re-read before unsafe POST
            fresh_pr = await self.proxy.get_pr(repo, pr_number)
            fresh_state = fresh_pr.get("state", "")
            fresh_merged = fresh_pr.get("merged", False)
            fresh_head = fresh_pr.get("head", {}).get("sha", "")
            if fresh_state != "open" or fresh_merged or fresh_pr.get("draft") is True or fresh_head != state.get("head_sha", ""):
                if fresh_pr.get("draft") is True:
                    await self._post_error(
                        repo, pr_number,
                        "Deployment requires PR ready-for-review / non-draft.",
                        comment_id=comment_id,
                    )
                await self._invalidate_review_required(
                    repo,
                    pr_number,
                    state,
                    fresh_head or state.get("head_sha", ""),
                    comment_id,
                    "PR changed after runtime preflight.",
                )
                return True

            # Fresh exact approval comment re-validation
            validated_comment = await self._revalidate_approve_comment(
                repo, comment_id, actor_id, machine_selector,
            )
            if validated_comment is None:
                state["status"] = "deploy-requested"
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "completed",
                    "args": {"machine": machine_alias, "actor": actor},
                }
                state["last_processed_comment_id"] = comment_id
                approve_attempt["outcome"] = "approval_revoked"
                self._record_approve_attempt(state, approve_attempt)
                markdown = self._deployment_progress_markdown(
                    repo, pr_number, state, machine, selected_components,
                    "Approval comment was deleted, edited, or changed; a NEW approval is required. ZERO deploy POST.",
                )
                event = {
                    "event": "Approval revoked",
                    "machine": machine_alias,
                    "timestamp": comments_mod.beijing_now_str(),
                }
                revoked_written = await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                    expected_lifecycle_state=initial_lifecycle_state,
                )
                if not revoked_written:
                    return False
                return True

            # Fresh actor permission re-check
            await self._check_approval_permissions(
                machine, actor, repo,
            )

            # Final fresh PR read — authoritative gate after approval+permission revalidation
            final_pr = await self.proxy.get_pr(repo, pr_number)
            final_state = final_pr.get("state", "")
            final_merged = final_pr.get("merged", False)
            final_head = final_pr.get("head", {}).get("sha", "")

            if final_state != "open" or final_merged or final_pr.get("draft") is True:
                if final_pr.get("draft") is True:
                    await self._post_error(
                        repo, pr_number,
                        "Deployment requires PR ready-for-review / non-draft.",
                        comment_id=comment_id,
                    )
                await self._invalidate_review_required(
                    repo,
                    pr_number,
                    state,
                    final_head or state.get("head_sha", ""),
                    comment_id,
                    "PR changed after approval re-validation.",
                )
                return True

            if final_head != state.get("head_sha", ""):
                await self._invalidate_review_required(
                    repo,
                    pr_number,
                    state,
                    final_head,
                    comment_id,
                    "HEAD drifted after approval re-validation.",
                )
                return True

            # Capture frozen component + deployment snapshots before hidden state validation
            expected_component_snapshot = self._canonical_component_snapshot(
                state.get("components", []),
            )
            expected_deployments = list(state.get("deployments", []))

            # Fresh hidden state re-read before unsafe POST
            validated_state = await self._revalidate_hidden_state(
                repo, pr_number, final_head,
                state.get("review_evidence", {}),
                expected_component_snapshot=expected_component_snapshot,
                expected_deployments=expected_deployments,
            )
            if validated_state is None:
                await self._invalidate_review_required(
                    repo, pr_number, state, final_head, comment_id,
                    "Hidden state changed after final pre-deploy validation.",
                )
                return True


            # Fresh review evidence revalidation before unsafe POST
            if not await self._fresh_review_evidence_matches_state(
                repo, pr_number, final_head,
                state.get("review_evidence", {}),
                expected_components=state.get("components", []),
            ):
                await self._invalidate_review_required(
                    repo, pr_number, state, final_head, comment_id,
                    "Review evidence changed after approval.",
                )
                return True

            # Non-core components deploy first; core restarts Agent Core and
            # therefore remains last. Track only the component whose POST may
            # be in flight, not every component selected for this approval.
            ordered_components = self._order_components_for_deploy(selected_components)

            # Reserve visible history capacity before marking the command as
            # in flight. Archive writes are safe, but an archive failure must
            # not leave a false deploying state for a POST that never started.
            prospective = comments_mod.deploy_requested(
                repo, pr_number, pr_head,
                state.get("components", []),
                self._get_machine_groups_for_components(state.get("components", []), repo),
                deployments=state.get("deployments", []),
                last_lifecycle_event=comments_mod.beijing_now_str(),
            )
            deploy_reservation = {
                "event": "Machine `<alias>` deployed",
                "machine": "<alias>",
                "ip": "<ip>",
                "timestamp": comments_mod.beijing_now_str(),
            }
            await self._preflight_archive_for_event(
                repo, pr_number, prospective,
                reserved_event=deploy_reservation,
            )

            # Write executing state only after ALL gates pass.
            # Persist the complete template before any deploy POST.
            state["status"] = "deploying"
            state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "executing",
                "args": {
                    "machine": machine_alias,
                    "actor": actor,
                    "node_id": machine.node_id,
                    "node_host": machine.node_host,
                    "pending_component_ids": [ordered_components[0]["component_id"]],
                },
            }
            state["last_processed_comment_id"] = comment_id
            executing_written = await self._write_lifecycle_with_history(
                repo, pr_number, state,
                self._deployment_progress_markdown(
                    repo, pr_number, state, machine, selected_components, "Deploying",
                ), event=None,
                expected_lifecycle_state=validated_state,
            )
            if not executing_written:
                return False
            persisted_lifecycle_state = copy.deepcopy(state)
            try:
                await self.proxy.project_status_label(repo, pr_number, "deploying")
            except Exception as exc:
                logger.warning("deploying label projection %s#%s exception=%s",
                               repo, pr_number, type(exc).__name__)
            round_id = round_id_for_state(state)
            deployment_started_notice_posted = False

            new_deployments = []
            deploy_error = None
            deploy_outcome_uncertain = False
            deploy_preflight_unavailable = False
            # Sequential per-component deploy; POST success followed by
            # post-deploy runtime verification before recording as deployed.
            for comp in ordered_components:
                image_ref = comp["image_ref"]
                runtime_id = str(comp.get("runtime_id") or "")
                component_id = comp.get("component_id", "")
                comp_target = comp.get("target", "")
                if not runtime_id:
                    deploy_error = f"missing runtime id for {comp_target}"
                    break

                # Persist verified earlier components and narrow restart
                # recovery to the next POST that may actually be attempted.
                state["deployments"] = list(existing_deployments) + list(new_deployments)
                command_args = dict(state["command"].get("args", {}))
                command_args["pending_component_ids"] = [component_id]
                state["command"]["args"] = command_args
                pending_written = await self._write_lifecycle_with_history(
                    repo, pr_number, state,
                    self._deployment_progress_markdown(
                        repo, pr_number, state, machine, [comp],
                        "Deploying and verifying Core" if comp_target == "core" else "Deploying component",
                    ), event=None,
                    expected_lifecycle_state=persisted_lifecycle_state,
                )
                if not pending_written:
                    return False
                persisted_lifecycle_state = copy.deepcopy(state)

                # Machine policy refreshes independently of the controller.
                # Recheck endpoint, compatibility, ownership, and the cached
                # connection binding immediately before each possible POST.
                try:
                    current_machine = self.policy.get_machine(machine_alias, repo=repo)
                except Exception as exc:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        f"Current machine policy could not be verified ({type(exc).__name__}).",
                    )
                if current_machine is None:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        "The approved machine is no longer present in this repository's current policy.",
                    )
                if (current_machine.alias, current_machine.node_id, current_machine.node_host) != approved_machine_binding:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        "The machine alias now points to a different node ID or IP than the one that passed preflight.",
                    )
                try:
                    current_compatible_ids = set(self._get_component_ids_for_machine(
                        machine_alias, [comp], repo,
                    ))
                except Exception as exc:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        f"Current machine compatibility could not be verified ({type(exc).__name__}).",
                    )
                if component_id not in current_compatible_ids:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        "The current machine policy no longer adapts to this component's target, platform, variant, or driver path.",
                    )
                try:
                    await self._check_approval_permissions(current_machine, actor, repo)
                except PolicyError as exc:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        f"Approval authority could not be revalidated ({str(exc)}).",
                    )
                try:
                    machine_after_permission = self.policy.get_machine(machine_alias, repo=repo)
                except Exception as exc:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        f"Machine policy changed while approval authority was checked ({type(exc).__name__}).",
                    )
                if (machine_after_permission is None
                        or (machine_after_permission.alias,
                            machine_after_permission.node_id,
                            machine_after_permission.node_host) != approved_machine_binding
                        or tuple(sorted(str(owner).casefold() for owner in
                                        (machine_after_permission.owners or [])))
                        != tuple(sorted(str(owner).casefold() for owner in
                                        (current_machine.owners or [])))):
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        "Machine ownership or endpoint policy changed during final authorization; submit a new approval.",
                    )
                current_core_key = (repo.casefold(), machine_alias)
                if (self._core_clients.get(current_core_key) is not core
                        or self._core_client_fingerprints.get(current_core_key)
                        != self._machine_connection_fingerprint(machine_after_permission)):
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        "The Agent Core target or configured credential changed after preflight; review the current configuration and submit a new approval.",
                    )
                machine = machine_after_permission
                node_id = machine.node_id

                if not deployment_started_notice_posted:
                    await self._post_bound_notice(
                        repo, pr_number, head_sha=pr_head, round_id=round_id,
                        comment_id=comment_id, topic="deployment-started",
                        expected_command_kind="approve_deploy",
                        message=(f"Deployment is in progress for HEAD `{_short(pr_head)}` in round `"
                                 f"{round_id}`. The Controller is completing final pre-POST checks for "
                                 f"machine `{machine.node_host}`."),
                    )
                    deployment_started_notice_posted = True

                    # Posting the notice awaits GitHub. Recheck the synchronous
                    # policy snapshot before entering the deploy client.
                    try:
                        latest_machine = self.policy.get_machine(machine_alias, repo=repo)
                    except Exception as exc:
                        return await self._block_machine_policy_change(
                            repo, pr_number, persisted_lifecycle_state,
                            persisted_lifecycle_state, comment_id, machine_alias, actor,
                            f"Machine policy could not be confirmed after posting the deployment notice ({type(exc).__name__}); no deploy POST was sent.",
                        )
                    if (latest_machine is None
                            or (latest_machine.alias, latest_machine.node_id,
                                latest_machine.node_host) != approved_machine_binding
                            or tuple(sorted(str(owner).casefold() for owner in
                                            (latest_machine.owners or [])))
                            != tuple(sorted(str(owner).casefold() for owner in
                                            (machine.owners or [])))
                            or self._core_clients.get(current_core_key) is not core
                            or self._core_client_fingerprints.get(current_core_key)
                            != self._machine_connection_fingerprint(latest_machine)):
                        return await self._block_machine_policy_change(
                            repo, pr_number, persisted_lifecycle_state,
                            persisted_lifecycle_state, comment_id, machine_alias, actor,
                            "Machine policy or credentials changed while the deployment notice was being posted; no deploy POST was sent.",
                        )
                # The lifecycle and start notice both await GitHub. Recheck the
                # PR, the exact approval comment, permissions, and the current
                # policy binding immediately before each possible POST.
                try:
                    execution_pr = await self.proxy.get_pr(repo, pr_number)
                except Exception:
                    logger.warning("final deployment PR check unavailable %s#%s", repo, pr_number)
                    return True
                execution_head = execution_pr.get("head") if isinstance(execution_pr, dict) else None
                if (not isinstance(execution_pr, dict)
                        or execution_pr.get("state") != "open"
                        or execution_pr.get("merged")
                        or execution_pr.get("draft") is True
                        or not isinstance(execution_head, dict)
                        or execution_head.get("sha") != pr_head):
                    return True
                if await self._revalidate_approve_comment(
                    repo, comment_id, actor_id, machine_selector,
                ) is None:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        "The approval comment changed after final authorization; no deploy POST was sent. Submit a new approval.",
                    )
                try:
                    await self._check_approval_permissions(machine, actor, repo)
                except PolicyError as exc:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        f"Approval authority changed before the deploy POST ({exc}); no deploy POST was sent.",
                    )
                try:
                    final_machine = self.policy.get_machine(machine_alias, repo=repo)
                except Exception as exc:
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        f"Current machine policy could not be verified immediately before deployment ({type(exc).__name__}).",
                    )
                if (final_machine is None
                        or (final_machine.alias, final_machine.node_id,
                            final_machine.node_host) != approved_machine_binding
                        or tuple(sorted(str(owner).casefold() for owner in
                                        (final_machine.owners or [])))
                        != tuple(sorted(str(owner).casefold() for owner in
                                        (machine.owners or [])))
                        or self._core_clients.get(current_core_key) is not core
                        or self._core_client_fingerprints.get(current_core_key)
                        != self._machine_connection_fingerprint(final_machine)):
                    return await self._block_machine_policy_change(
                        repo, pr_number, persisted_lifecycle_state,
                        persisted_lifecycle_state, comment_id, machine_alias, actor,
                        "Machine endpoint, ownership, or credentials changed during final deployment checks; no deploy POST was sent.",
                    )
                machine = final_machine
                node_id = machine.node_id
                # Route core through self-update path; non-core via existing path
                if comp_target == "core":
                    # Core has no automated case; runtime_id is always "core"
                    # Use unified idempotent entry point
                    try:
                        verified, health_evidence, post_performed = (
                            await self._deploy_or_adopt_core_component(
                                core, node_id, component_id, image_ref,
                            )
                        )
                    except DeployOutcomeUncertain as e:
                        deploy_outcome_uncertain = True
                        deploy_error = str(e)
                        break
                    except DeployPreflightUnavailable as e:
                        deploy_preflight_unavailable = True
                        deploy_error = str(e)
                        break
                    except DeployControllerError as e:
                        deploy_error = str(e)
                        break

                    approve_attempt["health"].append(health_evidence)
                    if not verified:
                        deploy_outcome_uncertain = True
                        deploy_error = (
                            f"post-deploy verify timeout for core: "
                            f"target image not observed running"
                        )
                        break

                    # Deploy success verified (or adopted as already deployed): record deployment
                    new_deployments.append({
                        "machine": machine_alias,
                        "ip": machine.node_host,
                        "component_ids": [component_id],
                        "phase": "deployed",
                    })
                else:
                    try:
                        await self._deploy_component(
                            core, node_id, image_ref, runtime_id,
                        )
                    except DeployOutcomeUncertain as e:
                        deploy_outcome_uncertain = True
                        deploy_error = str(e)
                        break
                    except DeployControllerError as e:
                        deploy_error = str(e)
                        break

                    verifying_written = await self._write_lifecycle_with_history(
                        repo, pr_number, state,
                        self._deployment_progress_markdown(
                            repo, pr_number, state, machine, [comp], "Verifying target image",
                        ), event=None,
                        expected_lifecycle_state=persisted_lifecycle_state,
                    )
                    if not verifying_written:
                        return False
                    persisted_lifecycle_state = copy.deepcopy(state)
                    # Post-deploy verification: confirm target image is actually running
                    verified, health_evidence = await self._verify_deployed_runtime(
                        core, node_id, component_id, runtime_id, image_ref,
                    )
                    approve_attempt["health"].append(health_evidence)
                    if not verified:
                        deploy_outcome_uncertain = True
                        deploy_error = (
                            f"post-deploy verify timeout for {comp_target!r}: "
                            f"target image not observed running"
                        )
                        break

                    # Deploy success verified: record deployment
                    new_deployments.append({
                        "machine": machine_alias,
                        "ip": machine.node_host,
                        "component_ids": [component_id],
                        "phase": "deployed",
                    })

            if deploy_preflight_unavailable:
                state["deployments"] = list(existing_deployments) + list(new_deployments)
                state["status"] = "deploy-requested"
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "completed",
                    "args": {"machine": machine_alias, "actor": actor},
                }
                state["last_processed_comment_id"] = comment_id
                approve_attempt["outcome"] = "preflight_unavailable"
                self._record_approve_attempt(state, approve_attempt)
                markdown = comments_mod.deploy_requested(
                    repo, pr_number, pr_head, state.get("components", []),
                    self._get_machine_groups_for_components([
                        c for c in state.get("components", [])
                        if c.get("component_id", "") not in _component_id_set(state["deployments"])
                    ], repo),
                    gate_note=[
                        "Agent Core status could not be confirmed; no POST was sent for the remaining component.",
                        "Deployment remains pending. Send a NEW approval after status checks recover.",
                    ],
                    deployments=self._enrich_deployments_for_render(state["deployments"], repo),
                )
                deferred_written = await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown,
                    event={"event": "Deploy preflight deferred: zero POST for remaining component",
                           "machine": machine_alias, "timestamp": comments_mod.beijing_now_str()},
                    expected_lifecycle_state=persisted_lifecycle_state,
                )
                if not deferred_written:
                    return False
                await self._project_status_best_effort(repo, pr_number, "deploy-requested")
                await self._post_command_reply(
                    repo, pr_number, comment_id,
                    "Deployment did not start because Agent Core preflight could not confirm the target. "
                    "No deploy POST was sent; after the machine recovers, post a new approval command.",
                    topic="deployment-preflight-failed",
                )
                return True

            if deploy_outcome_uncertain:
                state["deployments"] = list(existing_deployments) + list(new_deployments)
                state["status"] = "deploying"
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "uncertain",
                    "args": {
                        "machine": machine_alias,
                        "actor": actor,
                        "node_id": node_id,
                        "node_host": machine.node_host,
                        "pending_component_ids": [component_id],
                    },
                }
                state["last_processed_comment_id"] = comment_id
                approve_attempt["outcome"] = "uncertain"
                self._record_approve_attempt(state, approve_attempt)
                markdown = self._deploy_requested_markdown(
                    repo,
                    pr_number,
                    pr_head,
                    state.get("components", []),
                    self._get_machine_groups_for_components([
                        c for c in state.get("components", [])
                        if c.get("component_id", "") not in _component_id_set(state["deployments"])
                    ], repo),
                    gate_note=[
                        "### Restart Recovery",
                        "",
                        "A deploy POST was attempted but its outcome is uncertain.",
                        "The Controller will periodically verify the target status without repeating the POST.",
                        "Keep this deployment pending until the exact target is confirmed.",
                    ],
                    deployments=self._enrich_deployments_for_render(state["deployments"], repo),
                    display_status="deploying",
                )
                event = {
                    "event": "Deploy outcome uncertain",
                    "result": "uncertain",
                    "machine": machine_alias,
                    "timestamp": comments_mod.beijing_now_str(),
                }
                uncertain_written = await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                    expected_lifecycle_state=persisted_lifecycle_state,
                )
                if not uncertain_written:
                    return False
                await self._project_status_best_effort(repo, pr_number, "deploying")
                await self._post_bound_notice(
                    repo, pr_number, head_sha=pr_head,
                    round_id=round_id_for_state(state), comment_id=comment_id,
                    topic="deployment-outcome-uncertain",
                    expected_command_kind="approve_deploy",
                    message=("The deployment POST outcome is uncertain. The controller will check "
                             "the machine status and will not repeat the POST automatically."),
                )
                return True

            if deploy_error is not None:
                state["deployments"] = list(existing_deployments) + list(new_deployments)
                # Deploy failed - persist terminal state FIRST, then upload evidence
                state["status"] = "failed"
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "completed",
                    "args": {"machine": machine_alias, "actor": actor},
                }
                state["last_processed_comment_id"] = comment_id

                markdown = comments_mod.failed_comment(
                    repo, pr_number, pr_head, error=deploy_error,
                )
                approve_attempt["outcome"] = "failed"
                self._record_approve_attempt(state, approve_attempt)
                event = {
                    "event": "Deploy failed",
                    "result": "failed",
                    "machine": machine_alias,
                    "timestamp": comments_mod.beijing_now_str(),
                }
                failed_written = await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                    expected_lifecycle_state=persisted_lifecycle_state,
                )
                if not failed_written:
                    return False
                await self._project_status_best_effort(repo, pr_number, "failed")
                await self._post_bound_notice(
                    repo, pr_number, head_sha=pr_head,
                    round_id=round_id_for_state(state), comment_id=comment_id,
                    topic="deployment-failed", expected_command_kind="approve_deploy",
                    message=(f"Deployment failed for HEAD `{_short(pr_head)}` in round "
                             f"`{round_id_for_state(state)}`. Final status: `failed`."),
                )

                # One-shot runtime log snapshot for failed path
                runtime_logs = await self._snapshot_terminal_runtime_logs(
                    repo,
                    state.get("components", []),
                    state.get("deployments", []),
                )

                # Best-effort evidence upload after failed
                try:
                    cos_metadata = await self._upload_evidence(
                        repo, pr_number, pr_head, state, "fail",
                        deploy_error=deploy_error,
                        runtime_logs=runtime_logs,
                    )
                except Exception as e:
                    logger.warning(
                        "failed deploy evidence upload %s#%s: %s",
                        repo, pr_number, e,
                    )
                    return True

                if cos_metadata.get("object_key"):
                    # Step 1: rebind terminal state with COS metadata FIRST
                    rebound = await self._rebind_terminal_cos_if_current(
                        repo,
                        pr_number,
                        expected_head=pr_head,
                        expected_terminal_status="failed",
                        expected_comment_id=comment_id,
                        expected_command_kind="approve_deploy",
                        cos_metadata=cos_metadata,
                        markdown=comments_mod.failed_comment(
                            repo, pr_number, pr_head,
                            error=deploy_error,
                            cos_object_key=str(cos_metadata.get("object_key", "") or ""),
                            cos_bundle_sha256=str(cos_metadata.get("sha256", "") or ""),
                            cos_bundle_size=int(cos_metadata.get("size", 0) or 0),
                        ),
                    )
                    if not rebound:
                        return False
                    # Step 2: generate presigned URL AFTER successful rebind
                    cos_download_url = ""
                    try:
                        cos_download_url = self.cos.generate_evidence_download_url(
                            cos_metadata["object_key"],
                        )
                    except Exception:
                        logger.warning(
                            "COS_EVIDENCE_PRESIGN=FAILED repo=%s pr=%s",
                            repo, pr_number,
                        )
                    # Step 3: rebind final comment with download URL
                    rebound = await self._rebind_terminal_cos_if_current(
                        repo,
                        pr_number,
                        expected_head=pr_head,
                        expected_terminal_status="failed",
                        expected_comment_id=comment_id,
                        expected_command_kind="approve_deploy",
                        cos_metadata=cos_metadata,
                        markdown=comments_mod.failed_comment(
                            repo, pr_number, pr_head,
                            error=deploy_error,
                            cos_object_key=str(cos_metadata.get("object_key", "") or ""),
                            cos_bundle_sha256=str(cos_metadata.get("sha256", "") or ""),
                            cos_bundle_size=int(cos_metadata.get("size", 0) or 0),
                            cos_download_url=cos_download_url,
                        ),
                    )
                    if not rebound:
                        return False
                return True

            # Merge into existing deployments
            state["deployments"] = list(existing_deployments) + list(new_deployments)

            state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "completed",
                "args": {"machine": machine_alias, "actor": actor},
            }
            state["last_processed_comment_id"] = comment_id
            approve_attempt["outcome"] = "deployed"
            self._record_approve_attempt(state, approve_attempt)
            # Persist this machine's log task atomically with its verified
            # deployed fact, before testing projection or other evidence reads.
            deployment_logs = state.setdefault("deployment_logs", {})
            prior_log = deployment_logs.get(machine_alias)
            if not (isinstance(prior_log, dict)
                    and prior_log.get("head_sha") == pr_head
                    and prior_log.get("round_id") == round_id_for_state(state)
                    and prior_log.get("status") == "ready"):
                deployment_logs[machine_alias] = {
                    "head_sha": pr_head, "round_id": round_id_for_state(state),
                    "status": "pending", "collected_at": "",
                    "object_key": "", "sha256": "", "size": 0,
                }

            # Check completion: all required components durably deployed -> testing; else -> stay deploy-requested
            all_component_ids = {c.get("component_id", "") for c in components}
            deployed_component_ids = set()
            for dep in state["deployments"]:
                if dep.get("phase") == "deployed":
                    for cid in dep.get("component_ids", []):
                        deployed_component_ids.add(cid)

            if deployed_component_ids != all_component_ids:
                # Partial coverage achieved; more machines needed.
                state["status"] = "deploy-requested"
                remaining = [
                    c for c in components
                    if c.get("component_id", "") not in deployed_component_ids
                ]
                remaining_groups = self._get_machine_groups_for_components(remaining, repo)
                remaining_commands = self._machine_notice_lines(
                    remaining_groups, repo, mention_owners=False,
                    test_only=self._is_test_only_pr(pr_data),
                )
                markdown = comments_mod.deploy_requested(
                    repo, pr_number, pr_head,
                    state["components"],
                    remaining_groups,
                    gate_note=[
                        "### Partial coverage completed",
                        "",
                        f"Machine `{machine_alias}` deployed its compatible components.",
                        f"Remaining components need additional machine approval.",
                        *(remaining_commands or ["No trusted compatible machine with a unique IPv4 address is available for the remaining components."]),
                    ],
                    deployments=self._enrich_deployments_for_render(state["deployments"], repo),
                )
                machine_info = self.policy.get_machine(machine_alias, repo=repo)
                deploy_event = {
                    "event": f"Machine `{machine_alias}` deployed",
                    "machine": machine_alias,
                    "ip": machine_info.node_host if machine_info else "",
                    "timestamp": comments_mod.beijing_now_str(),
                }
                partial_written = await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=deploy_event,
                    expected_lifecycle_state=persisted_lifecycle_state,
                )
                if not partial_written:
                    return False
                await self._save_deployed_machine_log(repo, pr_number, state, machine_alias)
                await self._project_status_best_effort(repo, pr_number, "deploy-requested")
                notice_lines = self._machine_notice_lines(
                    remaining_groups, repo, mention_owners=True,
                    test_only=self._is_test_only_pr(pr_data),
                )
                await self._post_bound_notice(
                    repo, pr_number, head_sha=pr_head,
                    round_id=round_id_for_state(state), comment_id=comment_id,
                    topic="partial-deployment-next-approver",
                    expected_command_kind="approve_deploy",
                    message="\n".join([
                        "Partial deployment completed. The remaining components need another machine approval.",
                        *notice_lines,
                    ]),
                )
                return True

            # All components deployed -> durable testing state FIRST
            state["status"] = "testing"
            state.pop("acceptance_request", None)

            # Build testing markdown WITHOUT advisory case result
            testing_markdown = comments_mod.testing(
                repo, pr_number, pr_head, case_result="",
                deployments=self._enrich_deployments_for_render(state.get("deployments", []), repo),
                components=state.get("components", []),
                deployments_complete=(bool(all_component_ids)
                                      and all_component_ids.issubset(deployed_component_ids)),
            )
            # The LAST machine also gets its own per-machine history event,
            # mirroring the partial-coverage path, so every successful machine
            # appears exactly once in the visible History (newest-first).
            machine_info = self.policy.get_machine(machine_alias, repo=repo)
            last_machine_event = {
                "event": f"Machine `{machine_alias}` deployed",
                "machine": machine_alias,
                "ip": machine_info.node_host if machine_info else "",
                "timestamp": comments_mod.beijing_now_str(),
            }
            event_all = {
                "event": "All components deployed",
                "lifecycle": "`deploy-requested` \u2192 `testing`",
                "timestamp": comments_mod.beijing_now_str(),
            }
            # Atomic: write BOTH "All components deployed" and "Machine deployed"
            # in a single hidden-state/comment write so a crash cannot leave
            # incomplete History.  Newest-first: All components, then Machine.
            testing_written = await self._write_lifecycle_with_history(
                repo, pr_number, state, testing_markdown,
                events=[event_all, last_machine_event],
                expected_lifecycle_state=persisted_lifecycle_state,
            )
            if not testing_written:
                return False
            # Project status label "testing" SECOND
            await self._save_deployed_machine_log(repo, pr_number, state, machine_alias)
            await self._project_status_best_effort(repo, pr_number, "testing")
            upload_url = await self._current_testing_upload_url(
                repo, pr_number, head_sha=pr_head,
                round_id=round_id_for_state(state),
                approval_comment_id=comment_id,
            )
            upload_message = (
                f"本轮测试视频/图片上传入口：[打开上传页面](<{upload_url}>)。"
                if upload_url else
                "当前轮安全上传入口不可用，请从 PR Summary 查看上传状态；不要使用旧轮次链接。"
            )
            await self._post_bound_notice(
                repo, pr_number, head_sha=pr_head,
                round_id=round_id_for_state(state), comment_id=comment_id,
                topic="deployment-complete-testing-next",
                expected_command_kind="approve_deploy",
                message=(f"部署已完成，Lifecycle 已进入 `testing`。当前 HEAD 为 `{_short(pr_head)}`，"
                         f"round 为 `{round_id_for_state(state)}`。下一步请进行测试。\n\n"
                         f"{upload_message}\n\n"
                         "测试完成后，请由本轮 Machine Owner 执行 `/request_acceptance`。"),
            )

            return True

        except PolicyError as e:
            await self._post_error(repo, pr_number, str(e), comment_id=comment_id)
            return True
        except Exception as e:
            logger.error(
                "handle_approve_deploy %s#%s: %s",
                repo, pr_number, e,
            )
            raise

    def _all_required_components_deployed(self, state: dict, repo: str) -> bool:
        """Validate every required component has a trusted deployment record."""
        components = state.get("components")
        deployments = state.get("deployments")
        if (not isinstance(components, list) or not components
                or not isinstance(deployments, list)):
            return False
        seen = set()
        for component in components:
            if not isinstance(component, dict):
                return False
            cid = component.get("component_id")
            target = component.get("target")
            runtime_id = component.get("runtime_id")
            image_ref = component.get("image_ref")
            if (not isinstance(cid, str) or not cid or cid in seen
                    or not _is_supported_target(target)
                    or not isinstance(runtime_id, str) or not runtime_id
                    or not isinstance(image_ref, str) or not image_ref):
                return False
            seen.add(cid)
            if target == "core" and runtime_id != "core":
                return False
            records = [dep for dep in deployments if isinstance(dep, dict)
                       and dep.get("phase") == "deployed"
                       and isinstance(dep.get("component_ids"), list)
                       and cid in dep.get("component_ids", [])]
            if len(records) != 1:
                return False
            alias = records[0].get("machine")
            if not isinstance(alias, str) or not alias or self.policy.get_machine(alias, repo=repo) is None:
                return False
        return True

    async def _restart_record_test_comment_is_current(
        self, repo: str, state: dict, expected_comment: dict, *, consumed: bool = False,
    ) -> bool:
        """Recheck unedited comment provenance and current authority, fail closed."""
        if (not self.config.auth_valid or repo not in self.config.active_repos
                or repo not in self.config.github_repos):
            return False
        cid = expected_comment.get("id")
        user = expected_comment.get("user", {})
        if not isinstance(user, dict):
            return False
        uid, login = user.get("id"), user.get("login")
        if (not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0
                or (not consumed and cid <= state.get("last_processed_comment_id", 0))
                or (consumed and (cid != state.get("command", {}).get("comment_id")
                                  or cid > state.get("last_processed_comment_id", 0)))
                or not isinstance(uid, int) or isinstance(uid, bool) or uid <= 0
                or not isinstance(login, str) or not login
                or self.proxy.is_bot_comment(expected_comment)
                or not isinstance(expected_comment.get("body"), str)
                or not isinstance(expected_comment.get("created_at"), str)
                or not expected_comment.get("created_at")
                or expected_comment.get("updated_at") != expected_comment.get("created_at")):
            return False
        from .commands import parse_command
        parsed = parse_command(expected_comment.get("body", ""))
        try:
            await self._check_record_test_permissions(
                repo, login, state, actor_id=uid, comment_id=cid,
                result=parsed.result,
            )
        except PolicyError:
            return False
        fresh_comment = await self.proxy.get_comment(repo, cid)
        if not isinstance(fresh_comment, dict):
            return False
        fresh_user = fresh_comment.get("user", {})
        if not isinstance(fresh_user, dict):
            return False
        # Compare API identity, not response shape (pin/reactions may differ).
        # Bind both snapshots to the explicitly requested repository and ID.
        comment_url = f"{self.config.github_api_url.rstrip('/')}/repos/{repo}/issues/comments/{cid}"
        for comment in (expected_comment, fresh_comment):
            app = comment.get("performed_via_github_app")
            if app is not None and (
                not isinstance(app, dict)
                or not isinstance(app.get("id"), int) or isinstance(app.get("id"), bool)
                or app["id"] <= 0
            ):
                return False
        return (type(fresh_comment.get("id")) is int and fresh_comment["id"] == cid
                and expected_comment.get("url") == comment_url
                and fresh_comment.get("url") == comment_url
                and fresh_comment.get("body") == expected_comment["body"]
                and type(fresh_user.get("id")) is int and fresh_user["id"] == uid
                and fresh_user.get("login") == login
                and fresh_user.get("type") == user.get("type")
                and fresh_comment.get("created_at") == expected_comment["created_at"]
                and fresh_comment.get("updated_at") == expected_comment["updated_at"]
                and (fresh_comment.get("performed_via_github_app") or {}).get("id")
                == (expected_comment.get("performed_via_github_app") or {}).get("id")
                and not self.proxy.is_bot_comment(fresh_comment)
                and self.config.auth_valid
                and repo in self.config.active_repos and repo in self.config.github_repos)

    async def resume_record_test(
        self, repo: str, pr_number: int, expected_state: dict, expected_comment: dict,
    ) -> bool:
        """Resume the business verdict only; never dispatch deployment commands."""
        from .commands import parse_command

        cmd = parse_command(expected_comment.get("body", ""))
        if (cmd.kind != "record_test" or cmd.result not in ("pass", "fail")
                or expected_state.get("status") != "testing"
                or not self._all_required_components_deployed(expected_state, repo)
                or not await self._restart_record_test_comment_is_current(
                    repo, expected_state, expected_comment)):
            return False
        pr = await self.proxy.get_pr(repo, pr_number)
        if (pr.get("state") != "open" or pr.get("merged") or pr.get("draft")
                or pr.get("head", {}).get("sha") != expected_state.get("head_sha")):
            return False
        user = expected_comment["user"]
        handled = await self.handle_record_test(
            repo, pr_number, expected_comment["id"], cmd.result, cmd.summary,
            user["login"], str(user["id"]),
            _recovery_state=expected_state, _recovery_comment=expected_comment,
        )
        if not handled:
            return False
        # Policy/not-ready responses are handled commands in normal operation,
        # but never authorize advancing a startup baseline over unfinished work.
        fresh = await self.proxy.read_hidden_state(repo, pr_number)
        return (isinstance(fresh, dict)
                and fresh.get("head_sha") == expected_state.get("head_sha")
                and fresh.get("status") == ("succeeded" if cmd.result == "pass" else "failed")
                and fresh.get("command", {}).get("kind") == "record_test"
                and fresh.get("command", {}).get("comment_id") == expected_comment["id"]
                and fresh.get("command", {}).get("phase") == "completed"
                and fresh.get("last_processed_comment_id") == expected_comment["id"])

    async def _revalidate_record_test_state(
        self, repo: str, pr_number: int, original: dict, result: str,
        expected_comment: dict | None = None,
    ) -> dict | None:
        """Fresh-read GitHub after awaits before publishing the result."""
        pr_data = await self.proxy.get_pr(repo, pr_number)
        if inspect.isawaitable(pr_data):
            pr_data = await pr_data
        fresh = await self.proxy.read_hidden_state(repo, pr_number)
        if (not isinstance(pr_data, dict) or not isinstance(fresh, dict)
                or pr_data.get("state") != "open" or pr_data.get("merged")
                or pr_data.get("draft")):
            return None
        pr_head = pr_data.get("head", {}).get("sha", "")
        if (not isinstance(pr_head, str) or not re.fullmatch(r"[0-9a-f]{40}", pr_head)
                or pr_head != original.get("head_sha", "")):
            return None
        if (fresh.get("status") != "testing" or not isinstance(fresh.get("command"), dict)
                or fresh.get("command", {}).get("phase") != "completed"
                or fresh.get("command") != original.get("command")):
            return None
        if int(fresh.get("last_processed_comment_id", 0) or 0) != original.get("last_processed_comment_id", 0):
            return None
        if (fresh.get("head_sha") != original.get("head_sha")
                or fresh.get("cos") != original.get("cos")):
            return None
        if not self._all_required_components_deployed(fresh, repo):
            return None
        if result == "pass":
            for key in (
                "head_sha", "components", "deployments", "case_results", "review_evidence",
                "approve_attempts", "approve_attempts_total", "approve_attempts_truncated",
            ):
                if fresh.get(key) != original.get(key):
                    return None
        if expected_comment is not None:
            if (pr_head != original.get("head_sha")
                    or fresh != original
                    or not await self._restart_record_test_comment_is_current(
                        repo, fresh, expected_comment)):
                return None
        return copy.deepcopy(fresh)

    @staticmethod
    def _valid_cos_metadata(repo: str, pr_number: int, head_sha: str, metadata: dict) -> bool:
        try:
            CosClient.validate_object_key(repo, pr_number, head_sha, metadata.get("object_key", ""))
            size = metadata.get("size")
            return (isinstance(metadata.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]) is not None
                    and isinstance(size, int) and not isinstance(size, bool)
                    and 1 <= size <= 10485760)
        except (ValueError, TypeError, AttributeError):
            return False

    async def _terminal_cos_context_is_current(
        self, repo: str, pr_number: int, state: dict, comment: dict,
    ) -> bool:
        if not await self._restart_record_test_comment_is_current(
            repo, state, comment, consumed=True,
        ):
            return False
        pr = await self.proxy.get_pr(repo, pr_number)
        fresh = await self.proxy.read_hidden_state(repo, pr_number)
        return (fresh == state and pr.get("state") == "open" and not pr.get("merged")
                and not pr.get("draft")
                and pr.get("head", {}).get("sha") == state.get("head_sha")
                and self.config.auth_valid and repo in self.config.active_repos
                and repo in self.config.github_repos)

    async def _retry_terminal_cos(
        self, repo: str, pr_number: int, state: dict, *, refresh_url: bool = False,
    ) -> None:
        """Best-effort evidence maintenance in the existing serial watcher.

        Missing metadata is the durable restart signal. Retry timing is local:
        five attempts per command/process, 60s exponential backoff capped at
        900s. No business fields/events/labels are changed by this path.
        """
        try:
            state = copy.deepcopy(state)
            status = state.get("status")
            cmd = state.get("command", {})
            result = state.get("test_result")
            cid = cmd.get("comment_id")
            head = state.get("head_sha", "")
            if (status not in ("succeeded", "failed") or result not in ("pass", "fail")
                    or status != ("succeeded" if result == "pass" else "failed")
                    or cmd.get("kind") != "record_test" or cmd.get("phase") != "completed"
                    or not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0
                    or state.get("last_processed_comment_id", 0) < cid):
                return
            metadata = state.get("cos", {}) or {}
            if self._valid_cos_metadata(repo, pr_number, head, metadata) and not refresh_url:
                return
            key = (repo, pr_number, head, cid)
            attempts, next_attempt = self._cos_retries.get(key, (0, 0.0))
            now = time.monotonic()
            if attempts >= 5 or now < next_attempt:
                return
            self._cos_retries[key] = (attempts + 1, now + min(60 * 2 ** attempts, 900))

            # Old terminal states lacking the original command/deployment
            # evidence remain readable; they never authorize invented evidence.
            args = cmd.get("args", {})
            if (args.get("result") != result or not args.get("actor")
                    or not isinstance(args.get("summary"), str)
                    or not state.get("components") or not state.get("deployments")
                    or not state.get("review_evidence")
                    or not self._all_required_components_deployed(state, repo)):
                logger.warning("COS retry lacks original evidence %s#%s", repo, pr_number)
                return
            comment = await self.proxy.get_comment(repo, cid)
            if not isinstance(comment, dict):
                return
            from .commands import parse_command
            parsed = parse_command(comment.get("body", ""))
            if (parsed.kind != "record_test" or parsed.result != result
                    or parsed.summary != args["summary"]
                    or comment.get("user", {}).get("login") != args["actor"]
                    or not await self._terminal_cos_context_is_current(repo, pr_number, state, comment)):
                return

            # A stable date keeps uncertain PUTs on the same key across retries
            # and restarts. This candidate key is never published as metadata.
            object_key = metadata.get("object_key") or self.cos.build_object_key(
                repo, pr_number, head,
                now=datetime.fromisoformat(comment["created_at"].replace("Z", "+00:00")),
            )
            runtime_logs = await self._snapshot_terminal_runtime_logs(
                repo,
                state.get("components", []), state.get("deployments", []),
            )
            if not await self._terminal_cos_context_is_current(repo, pr_number, state, comment):
                return
            uploaded = await self._upload_evidence(
                repo, pr_number, head, state, result, args["summary"],
                runtime_logs=runtime_logs, object_key=object_key,
            )
            if not self._valid_cos_metadata(repo, pr_number, head, uploaded):
                logger.warning("COS retry not verified %s#%s attempt=%s", repo, pr_number, attempts + 1)
                return
            if not await self._terminal_cos_context_is_current(repo, pr_number, state, comment):
                return
            # Existing references are immutable even when they are incomplete.
            if metadata.get("object_key") and metadata != uploaded:
                logger.warning("COS retry metadata conflict %s#%s", repo, pr_number)
                return
            try:
                download_url = self.cos.generate_evidence_download_url(uploaded["object_key"])
            except Exception:
                download_url = ""
                logger.warning("COS_EVIDENCE_PRESIGN=FAILED repo=%s pr=%s", repo, pr_number)
            render_kwargs = {
                "cos_object_key": uploaded["object_key"],
                "cos_bundle_sha256": uploaded["sha256"],
                "cos_bundle_size": uploaded["size"],
                "cos_download_url": download_url,
            }
            if result == "pass":
                markdown = comments_mod.succeeded_comment(
                    repo, pr_number, head,
                    deployments=self._enrich_deployments_for_render(state["deployments"], repo),
                    components=state["components"], **render_kwargs,
                )
            else:
                markdown = comments_mod.failed_comment(repo, pr_number, head, **render_kwargs)
            if await self._rebind_terminal_cos_if_current(
                repo, pr_number, expected_head=head, expected_terminal_status=status,
                expected_comment_id=cid, expected_command_kind="record_test",
                expected_test_result=result, cos_metadata=uploaded, markdown=markdown,
                expected_state=state,
            ):
                self._cos_retries.pop(key, None)
        except Exception as exc:
            # No exception from optional persistence may strand a consumed
            # record_test or prevent subsequent GitHub command processing.
            logger.warning("COS retry failed %s#%s exception=%s", repo, pr_number, type(exc).__name__)

    async def handle_request_acceptance(
        self, repo: str, pr_number: int, comment_id: int,
        actor: str, actor_id: str,
    ) -> bool:
        """Persist a current-round independent acceptance request from a deployed owner."""
        try:
            pr = await self.proxy.get_pr(repo, pr_number)
            if (not isinstance(pr, dict) or pr.get("state") != "open"
                    or pr.get("merged") or pr.get("draft")):
                await self._post_error(
                    repo, pr_number,
                    "/request_acceptance 仅适用于开放、未合并且非 Draft 的 PR。",
                    comment_id=comment_id,
                )
                return True
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if not isinstance(state, dict):
                await self._post_error(repo, pr_number, "当前没有可验收的部署记录。", comment_id=comment_id)
                return True
            state = copy.deepcopy(state)
            head = pr.get("head", {}).get("sha", "")
            round_id = round_id_for_state(state)
            if isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head) and head != state.get("head_sha"):
                await self._supersede_head_drift(
                    repo, pr_number, state, head, comment_id,
                    command_kind="request_acceptance",
                )
                return True
            if (state.get("status") != "testing" or state.get("head_sha") != head
                    or not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head)
                    or not self._all_required_components_deployed(state, repo)):
                await self._post_command_not_ready(
                    repo, pr_number, state.get("status", "unknown"),
                    "所有本轮部署完成并进入 testing 后，Machine Owner 才能请求独立验收。",
                    comment_id=comment_id,
                )
                return True
            try:
                owner_id = int(actor_id)
            except (TypeError, ValueError):
                owner_id = 0
            if (owner_id <= 0 or not isinstance(actor, str)
                    or not self._is_deployed_machine_owner(actor, state, repo)):
                await self._post_error(
                    repo, pr_number,
                    "只有本轮实际部署机器的 Machine Owner 可以执行 /request_acceptance。",
                    comment_id=comment_id,
                )
                return True
            if type(comment_id) is not int or comment_id <= state.get("last_processed_comment_id", 0):
                return True
            comment = await self.proxy.get_comment(repo, comment_id)
            from .commands import parse_command
            parsed = parse_command(comment.get("body", "") if isinstance(comment, dict) else "")
            user = comment.get("user", {}) if isinstance(comment, dict) else {}
            api_base = self.config.github_api_url.rstrip("/")
            expected_comment_url = f"{api_base}/repos/{repo}/issues/comments/{comment_id}"
            expected_issue_url = f"{api_base}/repos/{repo}/issues/{pr_number}"
            if (not isinstance(comment, dict) or comment.get("id") != comment_id
                    or not isinstance(user, dict)
                    or parsed.kind != "request_acceptance"
                    or user.get("login") != actor or str(user.get("id")) != str(owner_id)
                    or type(user.get("id")) is not int or user["id"] <= 0
                    or comment.get("url") != expected_comment_url
                    or comment.get("issue_url") != expected_issue_url
                    or self.proxy.is_bot_comment(comment)
                    or not isinstance(comment.get("created_at"), str)
                    or comment.get("updated_at") != comment.get("created_at")):
                return False

            existing = state.get("acceptance_request")
            refresh_pending_request = (
                isinstance(existing, dict) and existing.get("status") == "pending"
                and existing.get("head_sha") == head and existing.get("round_id") == round_id
            )

            original = copy.deepcopy(state)
            if not refresh_pending_request:
                state["acceptance_request"] = {
                    "head_sha": head,
                    "round_id": round_id,
                    "status": "pending",
                    "requested_by": actor,
                    "requested_by_id": owner_id,
                    "comment_id": comment_id,
                    "requested_at": comment["created_at"],
                }

            reviewers_configured, eligible_reviewers = self._eligible_acceptance_reviewers(
                repo, pr,
            )

            state["command"] = {
                "comment_id": comment_id,
                "kind": "request_acceptance",
                "phase": "completed",
                "args": {"actor": actor, "round_id": round_id},
            }
            state["last_processed_comment_id"] = comment_id
            markdown = comments_mod.testing(
                repo, pr_number, head,
                deployments=self._enrich_deployments_for_render(state.get("deployments", []), repo),
                components=state.get("components", []),
                deployments_complete=True,
                acceptance_pending=True,
                acceptance_reviewers_configured=reviewers_configured,
                acceptance_reviewers=eligible_reviewers,
            )
            event = None
            if not refresh_pending_request:
                event = {
                    "event": "Independent acceptance requested",
                    "lifecycle": "`testing` · 测试完成 · 待独立验收",
                    "machine": next((d.get("machine", "") for d in state.get("deployments", [])
                                     if isinstance(d, dict) and d.get("phase") == "deployed"), ""),
                    "timestamp": comments_mod.beijing_now_str(),
                }
                await self._preflight_archive_for_event(repo, pr_number, markdown, reserved_event=event)
            written = await self._write_lifecycle_with_history(
                repo, pr_number, state, markdown, event=event,
                expected_acceptance_state=original,
                expected_acceptance_comment=comment,
            )
            if not written:
                return False
            try:
                await self.proxy.project_status_label(repo, pr_number, "acceptance-required")
            except Exception as exc:
                logger.warning("acceptance label projection %s#%s exception=%s",
                               repo, pr_number, type(exc).__name__)
            reviewer_notice = self._acceptance_reviewer_guidance(
                repo, pr, reviewers_configured, eligible_reviewers,
            )
            record_test_instruction = (
                "当前仓库验收白名单中的获授权账号可执行 `/record_test result=pass` 或 `/record_test result=fail`。"
                if reviewers_configured else
                "配置有效的当前仓库验收名单后，获授权账号才能执行 `/record_test result=pass` 或 `/record_test result=fail`。"
            )
            await self._post_bound_notice(
                repo, pr_number, head_sha=head, round_id=round_id,
                comment_id=comment_id, topic="acceptance-requested",
                expected_command_kind="request_acceptance",
                message="\n".join([
                    f"本轮测试已完成，当前等待独立验收：`{round_id}`。",
                    reviewer_notice,
                    record_test_instruction,
                ]),
            )
            return True
        except PolicyError as exc:
            await self._post_error(repo, pr_number, str(exc), comment_id=comment_id)
            return True
        except Exception as exc:
            logger.error("handle_request_acceptance %s#%s: %s",
                         repo, pr_number, type(exc).__name__)
            raise

    async def handle_record_test(
        self, repo: str, pr_number: int, comment_id: int,
        result: str, summary: str, actor: str, actor_id: str,
        *, _recovery_state: dict | None = None, _recovery_comment: dict | None = None,
    ) -> bool:
        try:
            pr_data = await self.proxy.get_pr(repo, pr_number)
            if pr_data.get("state") != "open" or pr_data.get("merged") or pr_data.get("draft"):
                return True
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if _recovery_state is not None and state != _recovery_state:
                return False
            if state is None:
                await self._post_error(
                    repo, pr_number,
                    "当前 PR 没有有效的 Deploy Approval Lifecycle。请先查看 PR Summary，等待生命周期状态更新后再记录测试结果。",
                    comment_id=comment_id,
                )
                return True

            # Freeze the accepted bindings before permissions/logging awaits.
            state = copy.deepcopy(state)
            pr_head = pr_data.get("head", {}).get("sha", "")
            if not isinstance(pr_head, str) or not re.fullmatch(r"[0-9a-f]{40}", pr_head):
                await self._post_error(repo, pr_number, "Cannot verify the current PR HEAD.", comment_id=comment_id)
                return True
            if pr_head != state.get("head_sha", ""):
                if _recovery_comment is not None:
                    return False
                await self._supersede_head_drift(
                    repo, pr_number, state, pr_head, comment_id,
                    command_kind="record_test",
                )
                return True

            _current_status = state.get("status", "")
            if _current_status == "testing":
                pass  # normal path continues below
            elif _current_status in ("succeeded", "failed"):
                terminal_command = state.get("command", {})
                if (terminal_command.get("kind") == "record_test"
                        and terminal_command.get("comment_id") == comment_id
                        and terminal_command.get("phase") == "completed"
                        and _current_status == ("succeeded" if result == "pass" else "failed")
                        and state.get("test_result") == result
                        and terminal_command.get("args", {}).get("result") == result):
                    if not self._all_required_components_deployed(state, repo):
                        return False
                    await self._check_record_test_permissions(
                        repo, actor, state, actor_id=actor_id,
                        comment_id=comment_id, result=result,
                    )
                    await self._retry_terminal_cos(repo, pr_number, state, refresh_url=True)
                    return True
                logger.warning(
                    "STALE_COMMAND_IGNORED kind=record_test comment_id=%s current_status=%s",
                    comment_id, _current_status,
                )
                await self.proxy.persist_cursor(repo, pr_number, comment_id)
                await self._post_command_reply(
                    repo, pr_number, comment_id,
                    f"`/record_test` was already finalized for this round. Current status: `{_current_status}`.",
                    topic="record-test-already-processed",
                )
                return True
            else:
                if _current_status in {"review-required", "reviewing"}:
                    not_ready_message = (
                        "当前 HEAD 尚无完整可信的审核和构建结果，请等待 Review Agent 完成并更新 PR Summary。"
                    )
                elif _current_status == "deploy-ready":
                    approval_guidance = await self._approval_command_guidance(
                        repo, pr_number,
                    )
                    not_ready_message = (
                        "当前 HEAD 的审核和构建已完成，Lifecycle 为 `deploy-ready`。"
                        "可以直接审批适配机器，无需先执行 `/request_deploy`。\n"
                        + approval_guidance
                    )
                elif _current_status == "deploy-requested":
                    not_ready_message = (
                        "当前 Lifecycle 为 `deploy-requested`，正在等待机器审批。请查看 PR Summary 中的可执行机器命令并等待部署完成。"
                    )
                elif _current_status == "deploying":
                    not_ready_message = (
                        "当前部署正在执行或确认结果，请等待部署状态更新后再记录测试结果。"
                    )
                else:
                    not_ready_message = (
                        f"当前 Lifecycle 状态为 `{_current_status or 'unknown'}`，尚不能记录测试结果。请查看当前 PR Summary。"
                    )
                await self._post_command_not_ready(
                    repo, pr_number, _current_status,
                    not_ready_message,
                    comment_id=comment_id,
                )
                return True

            # Check PR is still valid
            pr_state = pr_data.get("state", "")
            pr_merged = pr_data.get("merged", False)
            if pr_state != "open" or pr_merged or pr_data.get("draft"):
                await self._post_error(
                    repo, pr_number,
                    "PR is not open. Record test only when PR is open and unmerged.",
                    comment_id=comment_id,
                )
                return True

            # Check all components are deployed before accepting record_test
            components = state.get("components", [])
            deployed_component_ids = set()
            for dep in state.get("deployments", []):
                if dep.get("phase") == "deployed":
                    for cid in dep.get("component_ids", []):
                        deployed_component_ids.add(cid)
            all_component_ids = {c.get("component_id", "") for c in components}
            if (not all_component_ids or not all_component_ids.issubset(deployed_component_ids)
                    or not self._all_required_components_deployed(state, repo)):
                await self._post_error(
                    repo, pr_number,
                    "Not all components are deployed yet. "
                    "Complete all machine approvals before recording test result.",
                    comment_id=comment_id,
                )
                return True

            await self._check_record_test_permissions(
                repo, actor, state, actor_id=actor_id,
                comment_id=comment_id, result=result,
            )

            original_state = copy.deepcopy(state)
            if _recovery_comment is None:
                _recovery_comment = await self.proxy.get_comment(repo, comment_id)
            if not isinstance(_recovery_comment, dict):
                return False
            from .commands import parse_command
            accepted = parse_command(_recovery_comment.get("body", ""))
            user = _recovery_comment.get("user", {})
            if (accepted.kind != "record_test" or accepted.result != result
                    or accepted.summary != summary or _recovery_comment.get("id") != comment_id
                    or user.get("login") != actor or str(user.get("id")) != actor_id
                    or not await self._restart_record_test_comment_is_current(
                        repo, state, _recovery_comment)):
                return False
            fresh_state = await self._revalidate_record_test_state(
                repo, pr_number, original_state, result,
                expected_comment=_recovery_comment,
            )
            if fresh_state is None:
                return False
            state = fresh_state
            # COS is independent of the business verdict. Retain any published
            # metadata, but expose it only when its existing binding is valid.
            metadata = state.get("cos", {}) or {}
            render_cos = metadata if self._valid_cos_metadata(repo, pr_number, pr_head, metadata) else {}

            # Persist the business verdict before any COS or runtime-log request.
            state["status"] = "succeeded" if result == "pass" else "failed"
            state["test_result"] = "pass" if result == "pass" else "fail"
            state["command"] = {
                "comment_id": comment_id,
                "kind": "record_test",
                "phase": "completed",
                "args": {"result": result, "summary": summary, "actor": actor},
            }
            state["last_processed_comment_id"] = comment_id
            acceptance_request = copy.deepcopy(state.get("acceptance_request", {}))
            acceptance_request.update({
                "status": "completed",
                "accepted_by": actor,
                "accepted_by_id": int(actor_id),
                "acceptance_comment_id": comment_id,
                "accepted_at": utc_now(),
                "result": result,
                "summary": summary,
            })
            state["acceptance_request"] = acceptance_request

            # Write the terminal visible lifecycle comment
            if result == "pass":
                enriched = self._enrich_deployments_for_render(
                    state.get("deployments", []), repo,
                )
                terminal_markdown = comments_mod.succeeded_comment(
                    repo, pr_number, pr_head,
                    deployments=enriched,
                    components=state.get("components", []),
                    cos_object_key=str(render_cos.get("object_key", "") or ""),
                    cos_bundle_sha256=str(render_cos.get("sha256", "") or ""),
                    cos_bundle_size=int(render_cos.get("size", 0) or 0),
                )
                event = {
                    "event": "Test recorded",
                    "lifecycle": "`testing` \u2192 `succeeded`",
                    "result": "pass",
                    "timestamp": comments_mod.beijing_now_str(),
                }
            else:
                terminal_markdown = comments_mod.failed_comment(
                    repo, pr_number, pr_head,
                    cos_object_key=str(render_cos.get("object_key", "") or ""),
                    cos_bundle_sha256=str(render_cos.get("sha256", "") or ""),
                    cos_bundle_size=int(render_cos.get("size", 0) or 0),
                )
                event = {
                    "event": "Test recorded",
                    "lifecycle": "`testing` \u2192 `failed`",
                    "result": "fail",
                    "timestamp": comments_mod.beijing_now_str(),
                }

            await self._preflight_archive_for_event(
                repo, pr_number, terminal_markdown, reserved_event=event,
            )
            terminal_written = await self._write_lifecycle_with_history(
                repo, pr_number, state, terminal_markdown, event=event,
                expected_record_test_state=original_state,
                expected_record_test_comment=_recovery_comment,
            )
            if not terminal_written:
                return False
            try:
                await self.proxy.project_status_label(repo, pr_number, state["status"])
            except Exception as exc:
                logger.warning("record_test label projection %s#%s exception=%s",
                               repo, pr_number, type(exc).__name__)
            await self._post_bound_notice(
                repo, pr_number, head_sha=pr_head,
                round_id=round_id_for_state(state), comment_id=comment_id,
                topic="test-result-recorded", expected_command_kind="record_test",
                message=(f"Test result recorded as `{result}` for HEAD `{_short(pr_head)}` "
                         f"in round `{round_id_for_state(state)}`. Final status: `{state['status']}`."),
            )
            # Failures are logged/backed off internally and cannot undo or block
            # consumption of this durably completed business command.
            await self._retry_terminal_cos(repo, pr_number, state)
            return True



        except PolicyError as e:
            await self._post_error(repo, pr_number, str(e), comment_id=comment_id)
            return True
        except Exception as e:
            logger.error(
                "handle_record_test %s#%s: %s",
                repo, pr_number, e,
            )
            raise

    async def handle_evidence_link(
        self, repo: str, pr_number: int, comment_id: int, actor: str, actor_id: str,
    ) -> bool:
        """Consume deprecated requests without signing or replying with links."""
        return True

    async def handle_deploy_status(
        self, repo: str, pr_number: int, comment_id: int,
    ) -> bool:
        try:
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if state is None:
                await self._post_error(
                    repo, pr_number,
                    "No deploy approval state found for this PR.",
                    comment_id=comment_id,
                )
                return True

            head_sha = state.get("head_sha", "")
            status = state.get("status", "review-required")
            components = state.get("components", [])
            deployments = state.get("deployments", [])

            # Refresh 120s COS URL for terminal states
            cos_object_key = ""
            cos_bundle_sha256 = ""
            cos_bundle_size = 0
            cos_download_url = ""
            cos_state = state.get("cos", {})
            if status in ("succeeded", "failed"):
                cos_object_key = str(cos_state.get("object_key", "") or "")
                cos_bundle_sha256 = str(cos_state.get("sha256", "") or "")
                cos_bundle_size = int(cos_state.get("size", 0) or 0)
                if cos_object_key:
                    try:
                        cos_download_url = self.cos.generate_evidence_download_url(
                            cos_object_key,
                        )
                    except Exception:
                        logger.warning(
                            "COS_EVIDENCE_PRESIGN=FAILED repo=%s pr=%s",
                            repo, pr_number,
                        )

            markdown = comments_mod.deploy_status_comment(
                status, head_sha, repo, pr_number,
                components=components,
                deployments=deployments,
                cos_object_key=cos_object_key,
                cos_bundle_sha256=cos_bundle_sha256,
                cos_bundle_size=cos_bundle_size,
                cos_download_url=cos_download_url,
            )
            await self.proxy.post_issue_comment(
                repo, pr_number, markdown,
            )
            return True
        except Exception as e:
            logger.error(
                "handle_deploy_status %s#%s: %s", repo, pr_number, e,
            )
            raise

    async def handle_deploy_help(
        self, repo: str, pr_number: int, comment_id: int,
        topic: str = "",
    ) -> bool:
        try:
            markdown = comments_mod.deploy_help_text(topic)
            await self.proxy.post_issue_comment(
                repo, pr_number, markdown,
            )
            return True
        except Exception as e:
            logger.error(
                "handle_deploy_help %s#%s: %s", repo, pr_number, e,
            )
            raise

    # ── Build helpers ──


    def _get_component_ids_for_machine(
        self, machine_alias: str, components: list[dict], repo: str,
    ) -> list[str]:
        """Return component_ids compatible with this machine.

        Fail-closed rules:
        - If machine has platforms, component must have non-empty resolved_platform
          matching one of machine.platforms.
        - If machine has variants, perception/actucore components must have
          non-empty variant matching one of machine.variants.
        - For driver target, if machine has driver_paths, component must have
          non-empty driver_path matching one of machine.driver_paths.
        - No empty-value bypass.
        """
        machine = self.policy.get_machine(machine_alias, repo=repo)
        if machine is None or not self._machine_is_deployable(machine):
            return []
        result = []
        for comp in components:
            if not self.policy.machine_supports_target(
                    machine_alias, comp.get("target", ""), repo=repo):
                continue
            comp_platform = comp.get("resolved_platform", "")
            if not comp_platform or not machine.platforms or comp_platform not in machine.platforms:
                continue
            comp_variant = comp.get("variant", "")
            if comp.get("target") not in ("core", "driver") and machine.variants:
                if not comp_variant or comp_variant not in machine.variants:
                    continue
            if comp.get("target") == "driver":
                comp_driver_path = comp.get("driver_path", "")
                if not comp_driver_path or not machine.driver_paths or comp_driver_path not in machine.driver_paths:
                    continue
            result.append(comp.get("component_id", ""))
        return result

    def _machine_is_deployable(self, machine) -> bool:
        if (machine is None or not getattr(machine, "node_id", "")
                or not getattr(machine, "node_host", "")):
            return False
        if self._agent_core_factory is not None:
            return True
        tokens = getattr(self.config, "agent_core_tokens", {})
        return isinstance(tokens, dict) and bool(tokens.get(getattr(machine, "alias", "")))

    def _get_machine_groups_for_components(
        self, components: list[dict], repo: str,
    ) -> list[dict]:
        """Determine compatible machine groups for the given components.

        Returns machines whose compatible component_ids are a non-empty
        subset of the complete set of component_ids passed in.

        Machines with partial coverage ARE included — this supports
        multi-machine deployment where no single machine covers all
        remaining components.
        """
        machines = self.policy.get_machines(repo=repo)
        ip_counts: dict[str, int] = {}
        for machine in machines:
            try:
                address = ipaddress.ip_address(getattr(machine, "node_host", ""))
            except ValueError:
                continue
            if address.version == 4:
                ip_counts[str(address)] = ip_counts.get(str(address), 0) + 1
        required_ids = {
            comp.get("component_id", "")
            for comp in components
            if isinstance(comp, dict) and comp.get("component_id")
        }
        groups = []
        for m in machines:
            if not self._machine_is_deployable(m):
                continue
            try:
                address = ipaddress.ip_address(m.node_host)
            except ValueError:
                continue
            if address.version != 4 or ip_counts.get(str(address)) != 1:
                # Approval suggestions use literal IPs. An IP shared by
                # multiple aliases cannot be resolved safely by that command.
                continue
            compatible = []
            for comp in components:
                if not self.policy.machine_supports_target(
                    m.alias, comp.get("target", ""), repo=repo,
                ):
                    continue
                comp_platform = comp.get("resolved_platform", "")
                if not comp_platform or not m.platforms or comp_platform not in m.platforms:
                    continue
                comp_variant = comp.get("variant", "")
                if comp.get("target") not in ("core", "driver") and m.variants:
                    if not comp_variant or comp_variant not in m.variants:
                        continue
                if comp.get("target") == "driver":
                    comp_driver_path = comp.get("driver_path", "")
                    if not comp_driver_path or not m.driver_paths or comp_driver_path not in m.driver_paths:
                        continue
                compatible.append(comp.get("component_id", ""))
            # PARTIAL COVERAGE OK: any machine with non-empty compatible set is shown
            if compatible and set(compatible) <= required_ids:
                groups.append({
                    "alias": m.alias,
                    "node_id": m.node_id,
                    "ip": m.node_host,
                    "component_ids": compatible,
                })
        return groups

    def _enrich_deployments_for_render(
        self, deployments: list[dict], repo: str, *, historical: bool = False,
        historical_machine_groups: list[dict] | None = None,
    ) -> list[dict]:
        """Enrich deployment rows with machine IP for render-only display.

        Prefer the IP captured with a deployment snapshot. Legacy rows without
        one can still use current policy data for the active round. Historical
        rows may use only a unique IP captured in that archive's machine groups.

        Hidden-state authoritative identity remains the alias only.
        """
        historical_ips: dict[str, set[str]] = {}
        historical_deployment_ips: dict[str, set[str]] = {}
        if historical and isinstance(historical_machine_groups, list):
            for group in historical_machine_groups:
                if not isinstance(group, dict):
                    continue
                alias, ip = group.get("alias"), group.get("ip")
                if (isinstance(alias, str) and alias.strip()
                        and isinstance(ip, str) and ip.strip()):
                    historical_ips.setdefault(alias, set()).add(ip.strip())
        if historical:
            for deployment in deployments:
                if not isinstance(deployment, dict):
                    continue
                alias, ip = deployment.get("machine"), deployment.get("ip")
                if (isinstance(alias, str) and alias.strip()
                        and isinstance(ip, str) and ip.strip()):
                    historical_deployment_ips.setdefault(alias, set()).add(ip.strip())
        enriched: list[dict] = []
        for dep in deployments:
            if not isinstance(dep, dict):
                enriched.append(dep)
                continue
            machine_alias = dep.get("machine", "")
            ip = dep.get("ip", "")
            if historical:
                alias = machine_alias if isinstance(machine_alias, str) else ""
                deployment_matches = historical_deployment_ips.get(alias, set())
                if len(deployment_matches) == 1:
                    ip = next(iter(deployment_matches))
                elif len(deployment_matches) > 1:
                    ip = "历史未记录"
                else:
                    group_matches = historical_ips.get(alias, set())
                    ip = next(iter(group_matches)) if len(group_matches) == 1 else "历史未记录"
            elif (not historical and not (isinstance(ip, str) and ip.strip())
                  and isinstance(machine_alias, str) and machine_alias):
                machine_info = self.policy.get_machine(machine_alias, repo=repo)
                if machine_info is not None:
                    ip = machine_info.node_host or ""
            enriched.append({
                **dep,
                "ip": ip,
            })
        return enriched

    async def _preflight_running_images(
        self,
        core: AgentCoreClient,
        components: list[dict],
    ) -> list[dict]:
        """Read running_image for every selected component before any deploy POST.

        Also validates that no two selected components resolve to the same
        runtime_id — duplicates would cause double-deploy of the same runtime.

        Core components use /api/system/update-check instead of driver Status.
        list_drivers() is only called when there are non-core components.
        """
        deadline = time.monotonic() + float(self.config.total_timeout)

        def remaining_probe_time() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            return remaining

        has_non_core = any(c.get("target") != "core" for c in components)
        if has_non_core:
            drivers = await asyncio.wait_for(
                core.list_drivers(), timeout=remaining_probe_time(),
            )
            if not isinstance(drivers, list):
                raise DeployControllerError("Agent Core list_drivers returned an invalid payload")
        else:
            drivers = []

        cached_statuses: dict[str, dict] = {}
        preflight: list[dict] = []
        seen_runtime_ids: dict[str, str] = {}  # runtime_id -> component_id
        for component in components:
            comp_target = component.get("target", "")
            if comp_target == "core":
                # Core preflight: use update-check instead of driver status
                try:
                    update_info = await asyncio.wait_for(
                        core.core_update_check(), timeout=remaining_probe_time(),
                    )
                except AgentCoreError as e:
                    raise DeployControllerError(
                        f"core preflight update-check failed: {e}"
                    ) from e
                current_tag = str(update_info.get("current_tag", "") or "")
                target_tag = _target_tag_from_image_ref(
                    str(component.get("image_ref", "") or ""),
                )
                if not target_tag:
                    raise DeployControllerError(
                        "core image has no verifiable tag — refusing unsafe POST"
                    )
                runtime_id = "core"
                if runtime_id in seen_runtime_ids:
                    raise DeployControllerError(
                        f"duplicate runtime_id {runtime_id!r} for components "
                        f"{seen_runtime_ids[runtime_id]!r} and "
                        f"{component.get('component_id', '')!r} "
                        f"— refusing to deploy the same runtime twice"
                    )
                seen_runtime_ids[runtime_id] = component.get("component_id", "")
                already_target = bool(
                    current_tag and current_tag == target_tag,
                )
                preflight.append({
                    "component": component,
                    "runtime_id": runtime_id,
                    "running_image": current_tag,
                    "current_tag": current_tag,
                    "target_tag": target_tag,
                    "already_target": already_target,
                    "runtime_repo": "",
                })
                continue

            resolved = self._resolve_component_runtime(drivers, component)
            if resolved is None:
                target = component.get("target", "")
                raise DeployControllerError(
                    f"cannot uniquely resolve runtime for component target {target!r}"
                )
            runtime_id = str(resolved.get("runtime_id") or "")
            if not runtime_id:
                raise DeployControllerError(
                    f"runtime for component target {component.get('target', '')!r} has no driver id"
                )
            # Fail on duplicate runtime_id in the same approval group
            if runtime_id in seen_runtime_ids:
                raise DeployControllerError(
                    f"duplicate runtime_id {runtime_id!r} for components "
                    f"{seen_runtime_ids[runtime_id]!r} and {component.get('component_id', '')!r} "
                    f"— refusing to deploy the same runtime twice"
                )
            seen_runtime_ids[runtime_id] = component.get("component_id", "")
            if runtime_id not in cached_statuses:
                status = await asyncio.wait_for(
                    core.driver_status(runtime_id),
                    timeout=remaining_probe_time(),
                )
                if not isinstance(status, dict):
                    raise DeployControllerError(
                        f"Agent Core driver_status for {runtime_id} returned invalid data"
                    )
                cached_statuses[runtime_id] = status
            running_image = str(cached_statuses[runtime_id].get("running_image", "") or "")
            preflight.append({
                "component": component,
                "runtime_id": runtime_id,
                "running_image": running_image,
                "runtime_repo": resolved.get("runtime_repo", ""),
            })
        return preflight

    async def _verify_deployed_runtime(
        self,
        core,
        node_id: str,
        component_id: str,
        runtime_id: str,
        target_image_ref: str,
    ) -> tuple[bool, dict]:
        """Make one short observation; an unconfirmed POST remains uncertain."""
        last_status = ""
        last_running_image = ""
        last_error: str | None = None
        try:
            status_data = await asyncio.wait_for(
                core.driver_status(runtime_id),
                timeout=min(float(self.config.total_timeout),
                            AGENT_CORE_STATUS_PROBE_TIMEOUT_SECONDS),
            )
            if isinstance(status_data, dict):
                status_val = status_data.get("status", "")
                running_image = status_data.get("running_image", "")
                last_status = status_val if isinstance(status_val, str) else ""
                last_running_image = running_image if isinstance(running_image, str) else ""
                if last_status == "running" and last_running_image == target_image_ref:
                    return True, {
                        "component_id": component_id,
                        "runtime_id": runtime_id,
                        "status": "running",
                        "running_image": target_image_ref,
                        "target_image": target_image_ref,
                        "verified": True,
                    }
                last_error = "target image not yet observed running"
            else:
                last_error = "invalid runtime status response"
        except (AgentCoreError, asyncio.TimeoutError) as exc:
            last_error = type(exc).__name__

        result: dict[str, Any] = {
            "component_id": component_id,
            "runtime_id": runtime_id,
            "status": last_status,
            "running_image": last_running_image,
            "target_image": target_image_ref,
            "verified": False,
        }
        result["error"] = last_error or "runtime status unavailable"
        return False, result

    async def _snapshot_terminal_runtime_logs(
        self, repo: str, components: list[dict], deployments: list[dict],
    ) -> dict[str, str]:
        """Snapshot terminal runtime logs for deployed components.

        Executed AFTER durable terminal lifecycle state has been written.
        Never used as a health gate; never changes lifecycle status.
        Returns dict[source_key: log_text] where source_key is
        "machine_alias/runtime_id" for each distinct deployed runtime.
        """
        result: dict[str, str] = {}
        seen: set[tuple[str, str]] = set()
        for dep in deployments:
            if dep.get("phase") != "deployed":
                continue
            machine_alias = dep.get("machine", "")
            for cid in dep.get("component_ids", []):
                comp = next(
                    (c for c in components if c.get("component_id", "") == cid),
                    None,
                )
                if comp is None:
                    continue
                runtime_id = str(comp.get("runtime_id") or "")
                comp_target = comp.get("target", "")
                # Core has no driver_status endpoint; never fabricate runtime text.
                if comp_target == "core":
                    continue
                if not runtime_id:
                    continue
                seen_key = (machine_alias, runtime_id)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                source_key = f"{machine_alias}/{runtime_id}"
                machine = self.policy.get_machine(machine_alias, repo=repo)
                if machine is None:
                    result[source_key] = "[RUNTIME_LOG_SNAPSHOT_UNAVAILABLE]"
                    continue
                try:
                    core = await self._core_for_node(machine.node_id, repo)
                    status = await core.driver_status(runtime_id)
                    logs = status.get("logs", "") if isinstance(status, dict) else ""
                    if isinstance(logs, str) and logs:
                        result[source_key] = logs
                    else:
                        result[source_key] = "[RUNTIME_LOG_SNAPSHOT_UNAVAILABLE]"
                except NameError:
                    raise
                except Exception:
                    result[source_key] = "[RUNTIME_LOG_SNAPSHOT_UNAVAILABLE]"
        return result

    async def _save_deployed_machine_log(self, repo: str, pr_number: int,
                                        state: dict, machine_alias: str) -> None:
        """Persist and best-effort upload logs for this newly verified machine."""
        try:
            head = state.get("head_sha", "")
            round_id = round_id_for_state(state)
            deployments = [d for d in state.get("deployments", [])
                           if d.get("phase") == "deployed" and d.get("machine") == machine_alias]
            if not deployments:
                return
            current = await self.proxy.read_hidden_state(repo, pr_number)
            if not isinstance(current, dict) or current.get("head_sha") != head or round_id_for_state(current) != round_id:
                return
            deployments = [d for d in current.get("deployments", [])
                           if d.get("phase") == "deployed" and d.get("machine") == machine_alias]
            if not deployments:
                return
            logs = current.get("deployment_logs", {})
            prior = logs.get(machine_alias) if isinstance(logs, dict) else None
            if isinstance(prior, dict) and (prior.get("head_sha"), prior.get("round_id")) == (head, round_id):
                if prior.get("status") != "pending":
                    return
                task = copy.deepcopy(prior)
            else:
                task = {"head_sha": head, "round_id": round_id, "status": "pending",
                        "collected_at": "",
                        "object_key": "", "sha256": "", "size": 0}
                if not await self._write_machine_log_state(repo, pr_number, current, machine_alias, task):
                    return
            # The pending CAS is durable; refresh its exact result before any
            # log collection or subsequent state transition.
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if (not isinstance(state, dict) or state.get("head_sha") != head
                    or round_id_for_state(state) != round_id
                    or state.get("deployment_logs", {}).get(machine_alias) != task
                    or not any(d.get("phase") == "deployed" and d.get("machine") == machine_alias
                               for d in state.get("deployments", []))):
                return
            if task.get("collected_at"):
                object_key = self.cos.build_object_key(
                    repo, pr_number, head, machine_alias=machine_alias, round_id=round_id,
                    now=datetime.fromisoformat(task["collected_at"].replace("Z", "+00:00")),
                )
                try:
                    existing_bytes = await self.cos.download_evidence_archive(object_key, 10 * 1024 * 1024)
                except Exception as exc:
                    existing_bytes = None
                    cause = exc
                    while cause is not None:
                        getter = getattr(cause, "get_status_code", None)
                        try:
                            status = getter() if callable(getter) else None
                        except Exception:
                            status = None
                        if status == 404:
                            break
                        cause = getattr(cause, "__cause__", None)
                    else:
                        if self.cos._fake and str(exc) == "fake COS object not found":
                            cause = None
                        else:
                            # A failed read is not evidence that the object is
                            # absent. Keep the fixed task pending without recapture.
                            return
                if existing_bytes is not None:
                    existing_sha = hashlib.sha256(existing_bytes).hexdigest()
                    identity_error = CosClient._validate_evidence_identity(
                        object_key, existing_bytes, expected_sha256=existing_sha,
                        expected_size=len(existing_bytes), repo=repo, pr_number=pr_number,
                        head_sha=head, expected_result="deployed", machine_alias=machine_alias,
                        round_id=round_id,
                    )
                    if identity_error is None:
                        ready = copy.deepcopy(task)
                        ready.update(status="ready", object_key=object_key,
                                     sha256=existing_sha, size=len(existing_bytes))
                        if await self._write_machine_log_state(repo, pr_number, state, machine_alias, ready):
                            return
                        return
                    # A key occupied by different or unverifiable content is
                    # immutable. Never attempt a replacement upload.
                    return
                if existing_bytes is None and task.get("collected_at"):
                    # A confirmed 404 permits a fresh collection only after
                    # clearing the former collection identity via CAS.
                    task = copy.deepcopy(task)
                    task["collected_at"] = ""
                    if not await self._write_machine_log_state(repo, pr_number, state, machine_alias, task):
                        return
                    state = await self.proxy.read_hidden_state(repo, pr_number)
                    if (not isinstance(state, dict) or state.get("head_sha") != head
                            or round_id_for_state(state) != round_id
                            or state.get("deployment_logs", {}).get(machine_alias) != task):
                        return
            all_logs = await asyncio.wait_for(
                self._snapshot_terminal_runtime_logs(
                    repo, state.get("components", []), deployments,
                ),
                timeout=20,
            )
            machine_logs = {k: v for k, v in all_logs.items() if k.startswith(machine_alias + "/")}
            machine_logs = {k: v for k, v in machine_logs.items()
                            if isinstance(v, str) and v.strip()
                            and "SNAPSHOT_UNAVAILABLE" not in v}
            if not task.get("collected_at"):
                task = copy.deepcopy(task)
                task["collected_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                if not await self._write_machine_log_state(repo, pr_number, state, machine_alias, task):
                    return
                state = await self.proxy.read_hidden_state(repo, pr_number)
                if (not isinstance(state, dict) or state.get("head_sha") != head
                        or round_id_for_state(state) != round_id
                        or state.get("deployment_logs", {}).get(machine_alias) != task):
                    return
            object_key = self.cos.build_object_key(
                repo, pr_number, head, machine_alias=machine_alias, round_id=round_id,
                now=datetime.fromisoformat(task["collected_at"].replace("Z", "+00:00")),
            )
            pr_now = await self.proxy.get_pr(repo, pr_number)
            state_now = await self.proxy.read_hidden_state(repo, pr_number)
            if (not isinstance(pr_now, dict) or pr_now.get("state") != "open"
                    or pr_now.get("merged") or pr_now.get("draft")
                    or pr_now.get("head", {}).get("sha") != head
                    or state_now != state or not self.config.auth_valid
                    or repo not in self.config.active_repos or repo not in self.config.github_repos):
                return
            meta = await asyncio.wait_for(
                self._upload_evidence(repo, pr_number, head, state, "deployed",
                                      runtime_logs=machine_logs, object_key=object_key,
                                      machine_alias=machine_alias, round_id=round_id,
                                      collected_at=task["collected_at"]),
                timeout=20,
            )
            if (not meta or meta.get("object_key") != object_key
                    or not isinstance(meta.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", meta["sha256"])
                    or type(meta.get("size")) is not int or meta["size"] <= 0):
                unavailable = copy.deepcopy(task)
                unavailable["status"] = "unavailable"
                await self._write_machine_log_state(repo, pr_number, state, machine_alias, unavailable)
                return
            ready = copy.deepcopy(task)
            ready.update(status="ready", object_key=meta["object_key"],
                         sha256=meta["sha256"], size=meta["size"])
            await self._write_machine_log_state(repo, pr_number, state, machine_alias, ready)
        except Exception:
            logger.warning("machine deployment log save failed %s#%s machine=%s", repo, pr_number, machine_alias)

    async def _write_machine_log_state(self, repo: str, pr_number: int, expected: dict,
                                       machine_alias: str, task: dict) -> bool:
        """CAS one machine-log transition into the existing trusted Lifecycle comment."""
        try:
            pr = await self.proxy.get_pr(repo, pr_number)
            comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
            fresh = await self.proxy.read_hidden_state(repo, pr_number)
            if (not isinstance(pr, dict) or pr.get("state") != "open" or pr.get("merged")
                    or pr.get("draft") or pr.get("head", {}).get("sha") != expected.get("head_sha")
                    or fresh != expected or not self.config.auth_valid
                    or repo not in self.config.active_repos or repo not in self.config.github_repos
                    or not isinstance(comment, dict)):
                return False
            dep_exists = any(isinstance(d, dict) and d.get("phase") == "deployed"
                             and d.get("machine") == machine_alias for d in fresh.get("deployments", []))
            if not dep_exists or task.get("head_sha") != fresh.get("head_sha") \
                    or task.get("round_id") != round_id_for_state(fresh):
                return False
            new_state = copy.deepcopy(fresh)
            new_state.setdefault("deployment_logs", {})[machine_alias] = copy.deepcopy(task)
            from .github_state_proxy import HIDDEN_STATE_MARKER
            visible = comment.get("body", "").split(HIDDEN_STATE_MARKER, 1)[0].rstrip()
            written = await self.proxy.write_hidden_state(
                repo, pr_number, visible, new_state,
                expected_comment=(comment.get("id"), comment.get("body")),
            )
            if not isinstance(written, dict):
                return False
            after = await self.proxy.read_hidden_state(repo, pr_number)
            return after == new_state
        except Exception:
            return False

    async def _check_approval_permissions(
        self, machine, actor: str, repo: str,
    ) -> None:
        """Check if actor is authorized to approve deploy on this machine.

        Authorized if:
        1. actor is in machine owners list, OR
        2. actor has write/maintain/admin GitHub collaborator permission.

        Fail closed on permission lookup error.
        """
        actor_lower = actor.strip().lower() if actor else ""
        owners_lower = [o.lower() for o in (machine.owners or [])]
        if actor_lower in owners_lower:
            return

        # Fresh collaborator permission check
        try:
            permission = await self.proxy.collaborator_permission(repo, actor)
        except Exception as e:
            raise PolicyError(
                f"Cannot verify collaborator permission for {actor}: {e}"
            )

        if not Policy.collaborator_can_approve(permission):
            raise PolicyError(
                f"Actor {actor} is not a machine owner and has insufficient "
                f"collaborator permission ({permission}). "
                "Write, maintain, or admin access required."
            )

    def _is_deployed_machine_owner(self, actor: str, state: dict, repo: str) -> bool:
        actor_lower = actor.strip().casefold() if isinstance(actor, str) else ""
        if not actor_lower:
            return False
        for dep in state.get("deployments", []):
            if not isinstance(dep, dict) or dep.get("phase") != "deployed":
                continue
            machine = self.policy.get_machine(dep.get("machine", ""), repo=repo)
            if machine and any(
                isinstance(owner, str) and owner.casefold() == actor_lower
                for owner in (machine.owners or [])
            ):
                return True
        return False

    async def _check_record_test_permissions(
        self, repo: str, actor: str, state: dict, *, actor_id=None,
        comment_id=None, result="",
    ) -> None:
        """Authorize a configured allowlisted GitHub user for this round."""
        reviewers = self.policy.reviewers_for_repo(repo)
        valid_reviewers = {
            reviewer.strip().casefold()
            for reviewer in reviewers if isinstance(reviewer, str)
            and _LOGIN_RE.fullmatch(reviewer.strip())
            and not reviewer.strip().casefold().endswith("[bot]")
        } if isinstance(reviewers, list) else set()
        if not valid_reviewers:
            raise PolicyError(
                f"当前仓库 `{repo}` 没有有效的验收人白名单，已拒绝记录测试结果。"
                f"请检查该仓库 main 分支的 `test_reviewers.yaml`；不会使用其他仓库的名单。"
            )
        actor_lower = actor.strip().casefold() if isinstance(actor, str) else ""
        if not actor_lower or actor_lower not in valid_reviewers:
            raise PolicyError(
                f"只有当前仓库 `{repo}` 的 `test_reviewers.yaml` 白名单用户可以执行 `/record_test`。"
            )
        request = state.get("acceptance_request")
        try:
            expected_round = round_id_for_state(state)
        except Exception:
            expected_round = ""
        if (not isinstance(request, dict)
                or request.get("head_sha") != state.get("head_sha")
                or request.get("round_id") != expected_round):
            raise PolicyError("本轮尚无有效的验收请求。")
        try:
            numeric_actor_id = int(actor_id)
        except (TypeError, ValueError):
            numeric_actor_id = 0
        if numeric_actor_id <= 0 or type(comment_id) is not int or comment_id <= 0:
            raise PolicyError("无法验证验收人的 GitHub 评论身份。")
        if request.get("status") == "pending":
            if state.get("status") != "testing":
                raise PolicyError("当前轮次尚未处于可记录验收结果的 testing 状态。")
            return
        if (request.get("status") == "completed"
                and request.get("accepted_by_id") == numeric_actor_id
                and request.get("acceptance_comment_id") == comment_id
                and request.get("accepted_by", "").casefold() == actor_lower
                and request.get("result") == result
                and state.get("status") == ("succeeded" if result == "pass" else "failed")):
            return
        raise PolicyError("本轮验收请求已完成或不再有效。")

    async def _deploy_component(
        self, core: AgentCoreClient, node_id: str,
        image_ref: str, runtime_id: str,
    ) -> dict:
        """Deploy a single component to a machine via Agent Core."""
        try:
            result = await core.deploy_driver(runtime_id, image_ref)
            return {"result": result}
        except AgentCoreDeployOutcomeUncertain as e:
            raise DeployOutcomeUncertain(
                f"Deploy outcome uncertain for {runtime_id} on node {node_id}: {e}"
            ) from e
        except AgentCoreError as e:
            raise DeployControllerError(
                f"Deploy failed for {runtime_id} on node {node_id}: {e}"
            )

    async def _deploy_core_component(
        self,
        core: AgentCoreClient,
        node_id: str,
        image_ref: str,
    ) -> dict:
        """Deploy core via Agent Core self-update POST /api/system/update.

        Must be called AFTER all non-core components have been deployed,
        because core update restarts Agent Core and would interrupt further
        API calls.

        Returns {"result": data_dict} on UPDATE_ACCEPTED.
        Raises DeployOutcomeUncertain on transport uncertainty.
        Raises DeployControllerError on confirmed failure.
        """
        try:
            result = await core.update_core(image_ref)
            return {"result": result}
        except AgentCoreDeployOutcomeUncertain as e:
            raise DeployOutcomeUncertain(
                f"Core deploy outcome uncertain for node {node_id}: {e}"
            ) from e
        except AgentCoreError as e:
            raise DeployControllerError(
                f"Core deploy failed for node {node_id}: {e}"
            )

    async def _verify_core_update(
        self,
        core: AgentCoreClient,
        component_id: str,
        target_image_ref: str,
    ) -> tuple[bool, dict]:
        """Make one short update-check; only exact current_tag confirms success."""
        target_tag = _target_tag_from_image_ref(target_image_ref)
        if not target_tag:
            # No verifiable tag — cannot confirm deployment
            return False, {
                "component_id": component_id,
                "runtime_id": "core",
                "current_tag": "",
                "target_tag": target_tag,
                "verified": False,
                "error": "core image has no verifiable tag",
            }

        last_current_tag = ""
        last_error: str | None = None
        try:
            update_info = await asyncio.wait_for(
                core.core_update_check(),
                timeout=min(float(self.config.total_timeout),
                            AGENT_CORE_STATUS_PROBE_TIMEOUT_SECONDS),
            )
            current_tag = update_info.get("current_tag", "") if isinstance(update_info, dict) else ""
            last_current_tag = current_tag if isinstance(current_tag, str) else ""
            if last_current_tag and last_current_tag == target_tag:
                return True, {
                    "component_id": component_id,
                    "runtime_id": "core",
                    "current_tag": last_current_tag,
                    "target_tag": target_tag,
                    "verified": True,
                }
            last_error = "target core tag not yet observed"
        except (AgentCoreError, asyncio.TimeoutError) as exc:
            last_error = type(exc).__name__

        result: dict[str, Any] = {
            "component_id": component_id,
            "runtime_id": "core",
            "current_tag": last_current_tag,
            "target_tag": target_tag,
            "verified": False,
        }
        result["error"] = last_error or "core update status unavailable"
        return False, result

    @staticmethod
    def _order_components_for_deploy(components: list[dict]) -> list[dict]:
        """Order components so non-core deploy first, core last.

        Preserves relative order within each group.
        """
        non_core = [c for c in components if c.get("target") != "core"]
        core_comp = [c for c in components if c.get("target") == "core"]
        return non_core + core_comp


    async def _deploy_or_adopt_core_component(
        self,
        core: AgentCoreClient,
        node_id: str,
        component_id: str,
        image_ref: str,
    ) -> tuple[bool, dict, bool]:
        """Deploy core via self-update, or adopt if already at target.

        Returns (verified, health_evidence, post_performed).

        This is the single entry point for core deployment in
        handle_approve_deploy — it implements the idempotent
        "already-target → zero POST" protection.
        """
        target_tag = _target_tag_from_image_ref(image_ref)
        if not target_tag:
            raise DeployControllerError(
                "core image has no verifiable tag — refusing unsafe POST"
            )

        # Step 1: fresh already-target check
        try:
            update_info = await asyncio.wait_for(
                core.core_update_check(),
                timeout=float(self.config.total_timeout),
            )
        except (AgentCoreError, asyncio.TimeoutError) as e:
            raise DeployPreflightUnavailable(
                f"core already-target check failed: {e}"
            ) from e

        current_tag = str(update_info.get("current_tag", "") or "")

        # Step 2: if already at target, adopt without POST
        if current_tag and current_tag == target_tag:
            return True, {
                "component_id": component_id,
                "runtime_id": "core",
                "current_tag": current_tag,
                "target_tag": target_tag,
                "verified": True,
                "already_target": True,
            }, False

        # Step 3: perform the unsafe POST then verify
        await self._deploy_core_component(core, node_id, image_ref)
        verified, health_evidence = await self._verify_core_update(
            core, component_id, image_ref,
        )
        if verified:
            health_evidence["already_target"] = False
        return verified, health_evidence, True
    async def _run_automated_case(
        self, repo: str, pr_number: int, head_sha: str,
        components: list[dict],
        deployments: list[dict],
    ) -> dict[str, str]:
        """Run automated cases per component.

        Returns dict of component_id -> result ("pass", "fail", "n/a").
        Only runs after all components are deployed.

        For each component, binds to the actual deployed machine.
        Uses the explicit deployments list from the caller.
        """
        if not deployments:
            deployments = []
        results: dict[str, str] = {}
        for comp in components:
            cid = comp.get("component_id", "")
            if not cid:
                continue
            target = comp.get("target", "")
            variant = comp.get("variant", "")
            image_ref = comp.get("image_ref", "")

            # Find the deployment record for this component
            deployment_record = None
            for dep in deployments:
                if dep.get("phase") == "deployed" and cid in dep.get("component_ids", []):
                    deployment_record = dep
                    break

            if deployment_record is None:
                results[cid] = "fail"
                continue

            machine_alias = deployment_record.get("machine", "")
            machine = self.policy.get_machine(machine_alias, repo=repo)
            if machine is None:
                results[cid] = "fail"
                continue

            node_id = machine.node_id

            # Select case for this component before network access
            runner = self._get_case_runner()
            try:
                case_id = runner.select_case(target, variant, node_id)
            except Exception as e:
                logger.warning("case select %s: %s", cid, e)
                results[cid] = "fail"
                continue
            if case_id is None:
                results[cid] = "n/a"
                continue

            # Resolve core client (only for components with cases)
            try:
                core = await self._core_for_node(node_id, repo)
            except Exception as e:
                logger.warning("case core resolve %s: %s", cid, e)
                results[cid] = "fail"
                continue

            # Use deploy-pinned runtime_id from the component, never re-resolve
            # from list_drivers(). The runtime_id was pinned during approve_deploy
            # preflight and is the authoritative binding.
            driver_id = str(comp.get("runtime_id") or "")
            if not driver_id:
                logger.warning(
                    "case component %s has no pinned runtime_id; "
                    "deploy must have set runtime_id during preflight",
                    cid,
                )
                results[cid] = "fail"
                continue

            # Run case with actual binding
            deployment_dict = {
                "component_id": cid,
                "target": target,
                "variant": variant,
                "driver_path": comp.get("driver_path", ""),
                "runtime_id": driver_id,
                "node_id": node_id,
                "node_host": machine.node_host,
                "image_ref": image_ref,
                "machine_alias": machine_alias,
                "_core": core,
                "_driver_id": driver_id,
            }
            try:
                case_result = await runner.run_case(case_id, deployment_dict)
                passed = case_result.get("passed", False)
                results[cid] = "pass" if passed else "fail"
            except Exception as e:
                logger.warning(
                    "case %s for %s: %s", case_id, cid, e,
                )
                results[cid] = "fail"

        return results

    # _get_current_state removed - state is passed explicitly

    async def _upload_evidence(
        self, repo: str, pr_number: int, head_sha: str,
        state: dict, result: str, summary: str = "",
        deploy_error: str = "",
        runtime_logs: dict | None = None,
        object_key: str = "",
        machine_alias: str = "",
        round_id: str = "",
        collected_at: str = "",
    ) -> dict:
        """Upload COS evidence bundle. Returns metadata dict.

        Builds evidence in memory, uploads to COS, returns metadata.
        Failure does not raise.
        """
        try:
            # Retries keep the originally published key, including its date.
            # ensure_evidence_archive verifies/reuses it without replacing bytes.
            object_key = object_key or (state.get("cos", {}) or {}).get("object_key") or self.cos.build_object_key(
                repo, pr_number, head_sha, machine_alias=machine_alias, round_id=round_id,
                now=(datetime.fromisoformat(collected_at.replace("Z", "+00:00")) if collected_at else None),
            )
            CosClient.validate_object_key(
                repo, pr_number, head_sha, object_key,
                machine_alias=machine_alias, round_id=round_id,
            )
            archive_bytes = b""
            sha256 = ""
            size = 0

            # Build evidence bundle
            try:
                builder = EvidenceBuilder(self.config)
                archive_bytes, sha256, size = await builder.build_evidence(
                    repo=repo,
                    pr_number=pr_number,
                    head_sha=head_sha,
                    state=state,
                    result=result,
                    summary=summary,
                    deploy_error=deploy_error,
                    runtime_logs=runtime_logs,
                    machine_alias=machine_alias, round_id=round_id, collected_at=collected_at,
                )
            except Exception as e:
                logger.warning(
                    "evidence build %s#%s: %s", repo, pr_number, e,
                )

            if archive_bytes:
                return await self.cos.ensure_evidence_archive(
                    object_key=object_key,
                    archive_bytes=archive_bytes,
                    repo=repo,
                    pr_number=pr_number,
                    head_sha=head_sha,
                    result=result,
                    machine_alias=machine_alias, round_id=round_id,
                )
            return {"object_key": "", "sha256": "", "size": 0}
        except Exception as e:
            logger.warning(
                "evidence upload %s#%s: %s", repo, pr_number, e,
            )
            return {"object_key": "", "sha256": "", "size": 0}

    async def _ensure_history_archive(
        self, repo: str, pr_number: int, existing_comments: list[dict],
        *, extra_text: str,
    ) -> dict:
        """Reuse the identical trusted archive after an uncertain GitHub write."""
        prefix = f"{HISTORY_ARCHIVE_MARKER_PREFIX}{repo}:{pr_number}:"
        for comment in existing_comments:
            body = comment.get("body", "")
            if not isinstance(body, str) or prefix not in body or not self.proxy.is_bot_comment(comment):
                continue
            page = body.split(prefix, 1)[1].split(" -->", 1)[0]
            if not re.fullmatch(r"[1-9][0-9]*", page):
                continue
            if body == _build_archive_body(repo, pr_number, int(page), [], extra_text=extra_text):
                return {"page": int(page), "url": comment.get("html_url", "")}
        page = _next_history_archive_page(existing_comments, repo, pr_number)
        body = _build_archive_body(repo, pr_number, page, [], extra_text=extra_text)
        comment = await self.proxy.post_issue_comment(repo, pr_number, body)
        if not isinstance(comment, dict) or not comment.get("id"):
            raise GitHubStateProxyError("History archive write was not confirmed")
        # The archive must be visible as a trusted comment before the caller
        # removes its events from the lifecycle document.
        readback = await self.proxy.get_issue_comments(repo, pr_number)
        confirmed = next((row for row in readback if isinstance(row, dict)
                         and row.get("id") == comment.get("id")
                         and row.get("body") == body
                         and self.proxy.is_bot_comment(row)), None)
        if confirmed is None:
            raise GitHubStateProxyError("History archive readback failed")
        return {"page": page, "url": confirmed.get("html_url", comment.get("html_url", ""))}

    async def _preflight_archive_for_event(
        self,
        repo: str,
        pr_number: int,
        prospective_visible: str,
        reserved_event: dict | None = None,
    ) -> None:
        """Pre-flight visible lifecycle capacity and create archive if needed.

        Computes the REAL prospective visible body:
          prospective_renderer_output
          + existing generated history events
          + reserved/new event (if any)
          + archive links
        and measures its UTF-8 bytes.

        Must be called BEFORE any unsafe side effect (Agent Core deploy POST).
        If archive creation fails, raise to prevent the unsafe action.
        If prospective body exceeds 48 KiB and there is no archivables history,
        raise explicitly (fail closed).
        """
        from .github_state_proxy import (
            _parse_visible_history,
            _build_history_block,
            _insert_history_into_visible,
            VISIBLE_HISTORY_START_MARKER,
            VISIBLE_HISTORY_END_MARKER,
        )

        # 1. Fresh-read current lifecycle comment
        comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
        if comment is None:
            # No existing lifecycle comment: compute prospective body size.
            # Build real prospective visible: base + reserved_event
            from .github_state_proxy import (
                _build_history_block,
                _insert_history_into_visible,
            )
            if reserved_event is not None:
                prospective_with_history = _insert_history_into_visible(
                    prospective_visible, reserved_event,
                )
            else:
                prospective_with_history = prospective_visible
            body_bytes = _count_visible_bytes(prospective_with_history)
            if body_bytes <= _MAX_VISIBLE_LIFECYCLE_BYTES:
                # Under budget: no archive needed, caller will create lifecycle comment
                return
            else:
                # Over budget with no existing comment to archive into: fail closed
                raise GitHubStateProxyError(
                    "preflight archive: visible lifecycle exceeds 48 KiB "
                    "soft limit and no trusted lifecycle comment exists to host archives"
                )

        current_body = comment.get("body", "")
        if not isinstance(current_body, str):
            raise GitHubStateProxyError(
                "preflight archive: comment body is not a string"
            )

        # Keep raw events: parsing/re-rendering would discard historical fields.
        from .github_state_proxy import HIDDEN_STATE_MARKER
        hidden_idx = current_body.find(HIDDEN_STATE_MARKER)
        if hidden_idx < 0:
            raise GitHubStateProxyError("preflight archive: cannot find hidden state marker")
        existing_visible_part = current_body[:hidden_idx].rstrip()
        h_start = existing_visible_part.find(VISIBLE_HISTORY_START_MARKER)
        h_end = existing_visible_part.find(VISIBLE_HISTORY_END_MARKER)
        raw_events = []
        tail = ""
        if h_start >= 0 or h_end >= 0:
            if (h_start < 0 or h_end <= h_start
                    or existing_visible_part.count(VISIBLE_HISTORY_START_MARKER) != 1
                    or existing_visible_part.count(VISIBLE_HISTORY_END_MARKER) != 1):
                raise GitHubStateProxyError("preflight archive: malformed History markers")
            content = existing_visible_part[h_start + len(VISIBLE_HISTORY_START_MARKER):h_end]
            parts = content.split(HISTORY_EVENT_DELIMITER)
            raw_events = [part + HISTORY_EVENT_DELIMITER for part in parts[:-1]]
            tail = parts[-1]
            if tail.strip():
                raw_events.append(tail)
                tail = ""
        legacy_history = None
        if h_start < 0 and h_end < 0:
            legacy_history = re.search(r"(?ms)^### History[ \t]*\n.*?(?=^### |\Z)", existing_visible_part)
            if legacy_history:
                raw_events = [legacy_history.group(0).partition("\n")[2]]
        old_archives = re.search(r"(?ms)^### Archived history[ \t]*\n.*?(?=^### |\Z)", existing_visible_part)
        archive_links_text = old_archives.group(0) if old_archives else ""
        reserved_inner = ""
        if reserved_event is not None:
            reserved_inner = _build_history_block([reserved_event])[
                len(VISIBLE_HISTORY_START_MARKER):-len(VISIBLE_HISTORY_END_MARKER)
            ]

        def prospective_size(kept_events: list[str]) -> int:
            return _count_visible_bytes(
                prospective_visible.rstrip() + "\n\n### History\n\n"
                + VISIBLE_HISTORY_START_MARKER + reserved_inner
                + "".join(kept_events) + tail + VISIBLE_HISTORY_END_MARKER
                + "\n\n" + archive_links_text
            )

        # 4. Count real prospective visible bytes
        body_bytes = prospective_size(raw_events)
        if body_bytes <= _MAX_VISIBLE_LIFECYCLE_BYTES:
            return

        # 5. Need to archive oldest events
        if not raw_events:
            raise GitHubStateProxyError(
                "preflight archive: visible lifecycle exceeds 48 KiB "
                "soft limit and no history events available to archive"
            )

        # 6. Truncate oldest events to fit
        kept = list(raw_events)
        archived = []
        while kept and prospective_size(kept) > _MAX_VISIBLE_LIFECYCLE_BYTES - 2048:
            archived.insert(0, kept.pop())
        if prospective_size(kept) > _MAX_VISIBLE_LIFECYCLE_BYTES - 2048:
            raise GitHubStateProxyError("preflight archive: prospective lifecycle cannot fit after archival")

        if not archived:
            raise GitHubStateProxyError(
                "preflight archive: unable to fit visible lifecycle even "
                "after attempting to archive oldest events"
            )

        # 7. Create archive comment
        existing_comments = await self.proxy.get_issue_comments(repo, pr_number)
        archive = await self._ensure_history_archive(
            repo, pr_number, existing_comments,
            extra_text=("Archived oldest history events to preserve main lifecycle capacity.\n\n"
                        + VISIBLE_HISTORY_START_MARKER + "".join(archived) + tail
                        + VISIBLE_HISTORY_END_MARKER),
        )

        # 8. Rewrite the SAME lifecycle comment to remove archived events
        new_history_block = (VISIBLE_HISTORY_START_MARKER + "".join(kept) + tail
                             + VISIBLE_HISTORY_END_MARKER)

        # Reconstruct visible with shrunk history
        h_start = existing_visible_part.find(VISIBLE_HISTORY_START_MARKER)
        h_end = existing_visible_part.find(VISIBLE_HISTORY_END_MARKER)
        if h_start >= 0 and h_end >= 0:
            before_hist = existing_visible_part[:h_start]
            after_hist = existing_visible_part[h_end + len(VISIBLE_HISTORY_END_MARKER):]
            new_visible = before_hist + new_history_block + after_hist
        elif legacy_history is not None:
            new_visible = (existing_visible_part[:legacy_history.start()]
                           + "### History\n\n" + new_history_block + "\n\n"
                           + existing_visible_part[legacy_history.end():])
        else:
            new_visible = existing_visible_part.rstrip() + "\n\n" + new_history_block + "\n"

        # Publish the new archive reference atomically with removal from main.
        archive_line = comments_mod._archives_link_block([archive]).split("\n", 2)[2]
        old_links = re.search(r"(?ms)^### Archived history[ \t]*\n.*?(?=^### |\Z)", new_visible)
        if old_links:
            if str(archive["page"]) not in re.findall(r"History Archive Page ([0-9]+)", old_links.group(0)):
                new_visible = (new_visible[:old_links.end()].rstrip() + "\n" + archive_line
                               + "\n\n" + new_visible[old_links.end():])
        else:
            new_visible += "\n\n" + comments_mod._archives_link_block([archive])

        # 9. Verify the shrunk visible fits
        shrunk_bytes = _count_visible_bytes(new_visible)
        if shrunk_bytes > _MAX_VISIBLE_LIFECYCLE_BYTES:
            # Should not happen if truncation logic is correct, but guard anyway
            raise GitHubStateProxyError(
                "preflight archive: shrunk visible lifecycle still exceeds 48 KiB soft limit"
            )

        # 10. Parse fresh hidden state and write back
        fresh_state = _extract_hidden_state(current_body)
        if fresh_state is None:
            raise GitHubStateProxyError("preflight archive: cannot extract hidden state")
        fresh_state = _validate_hidden_state(fresh_state)

        final_body = _build_hidden_state_body(new_visible, fresh_state)
        comment_id = comment.get("id")
        fresh_comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
        if (fresh_comment is None or fresh_comment.get("body") != current_body
                or fresh_comment.get("id") != comment_id):
            raise GitHubStateProxyError("preflight archive: lifecycle changed during archival")
        if not isinstance(comment_id, int) or isinstance(comment_id, bool) or comment_id <= 0:
            raise GitHubStateProxyError("preflight archive: invalid lifecycle comment id")
        written = await self.proxy.update_comment(repo, comment_id, final_body)
        if not isinstance(written, dict):
            raise GitHubStateProxyError("preflight archive: lifecycle write was not confirmed")
        confirmed = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
        if (confirmed is None or confirmed.get("id") != comment_id
                or confirmed.get("body") != final_body):
            raise GitHubStateProxyError("preflight archive: lifecycle readback failed")

    @staticmethod
    def _set_lifecycle_summary_status(markdown: str, status: str) -> str:
        if status not in {"deploy-requested", "deploying"}:
            raise DeployControllerError(f"unsupported lifecycle display status {status!r}")
        target = f"**Status:** `{status}`"
        if target in markdown:
            return markdown
        alternate = "**Status:** `deploying`" if status == "deploy-requested" else "**Status:** `deploy-requested`"
        if markdown.count(alternate) != 1:
            raise DeployControllerError("lifecycle summary status marker is missing or ambiguous")
        return markdown.replace(alternate, target, 1)

    def _deploy_requested_markdown(
        self, repo, pr_number, head_sha, components, machine_groups, *,
        display_status: str = "deploy-requested", **kwargs,
    ) -> str:
        renderer = comments_mod.deploy_requested
        try:
            parameters = inspect.signature(renderer).parameters
            supports_display_status = (
                "display_status" in parameters
                or any(item.kind == inspect.Parameter.VAR_KEYWORD
                       for item in parameters.values())
            )
        except (TypeError, ValueError):
            supports_display_status = False
        if supports_display_status:
            return renderer(
                repo, pr_number, head_sha, components, machine_groups,
                display_status=display_status, **kwargs,
            )
        markdown = renderer(repo, pr_number, head_sha, components, machine_groups, **kwargs)
        return self._set_lifecycle_summary_status(markdown, display_status)

    def _deployment_progress_markdown(self, repo, pr_number, state, machine, components, phase):
        deployed = _component_id_set(state.get("deployments", []))
        remaining = [c for c in state.get("components", []) if c.get("component_id") not in deployed]
        targets = ", ".join(str(c.get("component_id", "")) for c in components)
        # Progress is presentation-only: hidden deployment rows retain their
        # canonical deployed schema until a POST has been verified.
        display_deployments = list(state.get("deployments", [])) + [{
            "machine": machine.alias,
            "ip": machine.node_host,
            "component_ids": [c.get("component_id", "") for c in components],
            "phase": ("preflight" if str(phase).lower().startswith(
                ("approval received", "preflight failed")) else "deploying"),
            "timestamp": comments_mod.beijing_now_str(),
        }]
        display_status = "deploying" if state.get("status") == "deploying" else "deploy-requested"
        return self._deploy_requested_markdown(
            repo, pr_number, state["head_sha"], state.get("components", []),
            self._get_machine_groups_for_components(remaining, repo),
            deployments=self._enrich_deployments_for_render(display_deployments, repo),
            gate_note=[f"{phase}: `{machine.alias}` / `{machine.node_host}`; components: {targets}; "
                       f"{comments_mod.beijing_now_str()}"],
            display_status=display_status,
        )

    async def _archive_round(self, repo, pr_number, state, reason):
        async with self._evidence_lock:
            return await self._archive_round_unlocked(repo, pr_number, state, reason)

    async def _archive_round_unlocked(self, repo, pr_number, state, reason):
        rid = round_id_for_state(state)
        records = await self.proxy.read_evidence_records(repo, pr_number)
        attachments = [{"kind": r["kind"], "key": r["key"],
                        "data": copy.deepcopy(r["data"]), "sha256": r["sha256"]}
                       for r in records if r["kind"] in ("operation", "attachment")
                       and r["data"].get("round_id") == rid]
        snapshot = archive_snapshot(repo, pr_number, state, utc_now(), reason)
        snapshot["attachment_records"] = attachments
        # Archive identity is the round snapshot and its evidence, not the
        # wall-clock time of an attempt to archive it. A retry after an
        # uncertain lifecycle write must resolve to the same immutable record.
        identity_snapshot = copy.deepcopy(snapshot)
        identity_snapshot.pop("ended_at", None)
        identity_snapshot.pop("archived_at", None)
        content = {"snapshot": identity_snapshot["snapshot"], "attachment_records": attachments}
        import json
        key = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        existing = next((r for r in records if r["kind"] == "round" and r["key"] == key), None)
        if existing is None:
            # Reuse records made by the previous timestamp-based key scheme
            # when they already contain this same snapshot and attachment set.
            existing = next((r for r in records if r["kind"] == "round"
                             and r["data"].get("round_id") == rid
                             and r["data"].get("reason") == reason
                             and r["data"].get("attachment_records") == attachments
                             and {k: v for k, v in r["data"].get("snapshot", {}).items()
                                  if k not in {"ended_at", "archived_at"}}
                             == identity_snapshot.get("snapshot")), None)
        record = existing or await self.proxy.write_evidence_record(repo, pr_number, "round", key, snapshot)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise GitHubStateProxyError("Round archive write was not confirmed")
        # Do not let Lifecycle point at an archive until its immutable evidence
        # record is observable through the public reader with the same digest.
        verified = await self.proxy.read_evidence_records(repo, pr_number)
        readback = next((r for r in verified if r["kind"] == "round"
                         and r["key"] == record.get("key", key)
                         and r.get("sha256") == record["sha256"]), None)
        if (readback is None or readback["data"].get("round_id") != rid
                or readback["data"].get("attachment_records") != attachments):
            raise GitHubStateProxyError("Round archive readback failed")
        result = {"round_id": rid, "sha256": record["sha256"]}
        if type(record.get("comment_id")) is int and isinstance(record.get("url"), str):
            result.update(comment_id=record["comment_id"], url=record["url"])
        return result

    @staticmethod
    def _replace_markdown_section(markdown, heading, content, next_heading):
        """Replace one heading-delimited lifecycle section idempotently."""
        heading_re = re.compile(r"(?m)^" + re.escape(heading) + r"[ \t]*$")
        matches = list(heading_re.finditer(markdown))
        if len(matches) != 1:
            return markdown
        content_start = matches[0].end()
        next_match = None
        if isinstance(next_heading, str) and next_heading:
            next_re = re.compile(r"(?m)^" + re.escape(next_heading) + r"[ \t]*$")
            next_match = next_re.search(markdown, content_start)
            if next_match is None:
                return markdown
        end = next_match.start() if next_match is not None else len(markdown)
        replacement = markdown[:content_start] + "\n\n" + content.rstrip()
        if end < len(markdown):
            return replacement + "\n\n" + markdown[end:]
        if markdown.endswith(("\n", "\r")):
            replacement += "\n"
        return replacement

    @staticmethod
    def _unwrap_history_details(history_content: str) -> str:
        """Remove one generated outer History wrapper without touching its body."""
        if not isinstance(history_content, str):
            return ""
        opening = re.match(r"\A\s*<details\b[^>]*>", history_content, re.IGNORECASE)
        if opening is None:
            return history_content
        summary_open = re.match(
            r"\s*<summary\b[^>]*>", history_content[opening.end():], re.IGNORECASE,
        )
        if summary_open is None:
            return history_content
        summary_start = opening.end() + summary_open.end()
        summary_close = re.search(r"</summary\s*>", history_content[summary_start:], re.IGNORECASE)
        if summary_close is None:
            return history_content
        summary_end = summary_start + summary_close.start()
        title = history_content[summary_start:summary_end].strip()
        if title.casefold() != "history · 历史事件":
            return history_content
        body_start = summary_start + summary_close.end()
        outer_close = re.search(r"</details\s*>\s*\Z", history_content, re.IGNORECASE)
        if outer_close is None or outer_close.start() < body_start:
            return history_content
        return history_content[body_start:outer_close.start()]

    @staticmethod
    def _fold_history_content(history_content: str) -> str:
        """Normalize only the outer History block; preserve nested event markup."""
        if not isinstance(history_content, str) or not history_content.strip():
            return ""
        content = history_content
        while True:
            unwrapped = DeployController._unwrap_history_details(content)
            if unwrapped == content:
                break
            content = unwrapped
        header = "<details>\n<summary>History · 历史事件</summary>"
        if content and not content.startswith("\n"):
            header += "\n\n"
        closing_separator = "" if not content or content.endswith("\n") else "\n"
        return (
            header + content + closing_separator + "</details>"
        )

    @staticmethod
    def _extract_visible_history(visible):
        """Remove recognized History sections for safe repositioning verbatim."""
        if not isinstance(visible, str):
            return visible, ""
        history_start = VISIBLE_HISTORY_START_MARKER
        history_end = VISIBLE_HISTORY_END_MARKER
        if visible.count(history_start) != visible.count(history_end):
            raise GitHubStateProxyError("Malformed visible History markers")
        history_spans = []
        if history_start in visible:
            cursor = 0
            while True:
                start = visible.find(history_start, cursor)
                end_before_start = visible.find(history_end, cursor)
                if start < 0 and end_before_start < 0:
                    break
                if end_before_start >= 0 and (start < 0 or end_before_start < start):
                    raise GitHubStateProxyError("Malformed visible History markers")
                end = visible.find(history_end, start + len(history_start))
                if end < 0 or visible.find(history_start, start + len(history_start), end) >= 0:
                    raise GitHubStateProxyError("Malformed visible History markers")
                history_spans.append((start, end + len(history_end)))
                cursor = end + len(history_end)

        heading_re = re.compile(
            r"(?mi)^(#{2,6})[ \t]+(?:历史记录|history|archived history)[ \t]*$",
        )
        heading_matches = list(heading_re.finditer(visible))
        all_heading_matches = list(re.finditer(r"(?m)^(#{1,6})[ \t]+", visible))
        for heading in heading_matches:
            level = len(heading.group(1))
            end = len(visible)
            for following in all_heading_matches:
                if following.start() > heading.start() and len(following.group(1)) <= level:
                    end = min(end, following.start())
                    break
            projection_start = visible.find("<!-- evidence-projection:start -->", heading.end())
            if projection_start >= 0:
                end = min(end, projection_start)
            history_spans.append((heading.start(), end))

        if not history_spans:
            return visible, ""
        history_spans.sort()
        merged = []
        for start, end in history_spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        pieces = []
        cursor = 0
        history_blocks = []
        for start, end in merged:
            pieces.append(visible[cursor:start])
            block = visible[start:end]
            block = re.sub(r"\A[ \t]*#{2,6}[ \t]+(?:history|历史记录)[ \t]*\r?\n+",
                           "", block, count=1, flags=re.IGNORECASE)
            if block.strip():
                # Collapse any pre-existing generated marker pair before
                # joining sections so the final Summary contains one marker
                # pair and one History details block, never nested wrappers.
                block = block.replace(history_start, "").replace(history_end, "")
                block = DeployController._unwrap_history_details(block)
                if block.strip():
                    history_blocks.append(block)
            cursor = end
        pieces.append(visible[cursor:])
        if not history_blocks:
            return "".join(pieces), ""
        history_content = "\n\n".join(history_blocks)
        history_content = DeployController._fold_history_content(history_content)
        if not history_content.endswith("\n"):
            history_content += "\n"
        history = (
            VISIBLE_HISTORY_START_MARKER + "\n" + history_content
            + VISIBLE_HISTORY_END_MARKER
        )
        return "".join(pieces), history

    @classmethod
    def _strip_legacy_evidence_blocks(cls, visible):
        """Remove duplicate evidence presentation before History is isolated.

        Operate on the whole visible body first so obsolete evidence nested
        under a legacy History heading cannot be copied into the preserved
        History block. Only presentation rows are removed; archive links and
        lifecycle prose remain intact.
        """
        if not isinstance(visible, str):
            return visible
        projection_start = "<!-- evidence-projection:start -->"
        projection_end = "<!-- evidence-projection:end -->"
        starts, ends = visible.count(projection_start), visible.count(projection_end)
        if starts != ends:
            raise GitHubStateProxyError("Malformed evidence projection")
        if starts:
            cleaned = []
            cursor = 0
            for _ in range(starts):
                start = visible.find(projection_start, cursor)
                end_before_start = visible.find(projection_end, cursor)
                if start < 0 or end_before_start >= 0 and end_before_start < start:
                    raise GitHubStateProxyError("Malformed evidence projection")
                end = visible.find(projection_end, start + len(projection_start))
                if start < 0 or end < 0 or visible.find(projection_start, start + len(projection_start), end) >= 0:
                    raise GitHubStateProxyError("Malformed evidence projection")
                cleaned.append(visible[cursor:start])
                cursor = end + len(projection_end)
            cleaned.append(visible[cursor:])
            visible = "".join(cleaned)

        lines = visible.splitlines(keepends=True)
        result = []
        index = 0
        heading_re = re.compile(
            r"^(#{2,6})[ \t]+(?:\*\*)?(?:测试附件|附件)(?:\*\*)?[ \t]*#*[ \t]*(?:\r?\n)?$",
        )
        legacy_link_re = re.compile(
            r"\[(?:查看测试附件|打开附件页|上传或查看附件|查看 COS 证据与历史|上传到 COS)\]"
            r"\((?:<[^>]*>|[^)]*)\)"
        )
        cos_link_re = re.compile(
            r"\[[^\]]*\]\((?:<)?https://[^<>\s)]*\.cos\.[^<>\s)]*\.myqcloud\.com/[^<>\s)]*(?:>)?\)",
            re.IGNORECASE,
        )
        cos_url_re = re.compile(
            r"https?://[^\s<>()\]]*\.cos\.[^\s<>()\]]*\.myqcloud\.com/[^\s<>()\]]*",
            re.IGNORECASE,
        )
        cos_download_field_re = re.compile(
            r'''(?i)\bcos_download_url\b[ \t]*[:=][ \t]*(?:"[^"]*"|'[^']*'|[^|,}\s]+)'''
        )
        metadata_value_re = re.compile(
            r"[\"']?(?:object_key|sha256|size|cos_object_key|cos_bundle_sha256|cos_bundle_size|cos_download_url|cos)"
            r"[\"']?[ \t]*[:=][ \t]*(\"[^\"]*\"|'[^']*'|[0-9]+|null|None)",
            re.IGNORECASE,
        )
        empty_metadata_assignment_re = re.compile(
            r"(?i)(?:\*{1,2}cos(?:_object_key|_bundle_sha256|_bundle_size)?\*{1,2}"
            r"|[\"']?cos(?:_object_key|_bundle_sha256|_bundle_size)?[\"']?)"
            r"[ \t]*[:：=][ \t]*(?:[|·*_`-]*)(?:\r?\n)?$",
        )
        while index < len(lines):
            heading = heading_re.match(lines[index])
            if heading:
                index += 1
                # Remove the old heading and its immediate Markdown table,
                # while retaining following History prose/events.
                while index < len(lines) and not lines[index].strip():
                    index += 1
                if index < len(lines) and lines[index].lstrip().startswith("|"):
                    while index < len(lines) and lines[index].lstrip().startswith("|"):
                        index += 1
                continue
            line = lines[index]
            legacy_link = legacy_link_re.search(line)
            cos_link = cos_link_re.search(line)
            cos_url = cos_url_re.search(line)
            metadata_values = metadata_value_re.findall(line)
            empty_cos_metadata = (
                bool(metadata_values) and all(
                    value.strip("\\\"' ").lower() in {"", "0", "null", "none"}
                    for value in metadata_values
                ) and bool(re.search(r"(?i)cos|object_key|sha256", line))
            ) or bool(empty_metadata_assignment_re.search(line))
            old_cos_download_row = (
                bool(re.search(r"(?i)\b(?:COS[ \t]+下载|下载[ \t]+COS|verified[ \t]+COS[ \t]+download)\b", line))
                and (".cos." in line.lower() or "cos_download_url" in line.lower())
            )
            if cos_download_field_re.search(line):
                line = cos_download_field_re.sub("COS 下载链接已省略", line)
                cos_link = cos_link_re.search(line)
                cos_url = cos_url_re.search(line)
            if legacy_link or cos_link or cos_url or old_cos_download_row or empty_cos_metadata:
                remainder = line
                if legacy_link:
                    remainder = legacy_link_re.sub("", remainder)
                if cos_link:
                    remainder = cos_link_re.sub("", remainder)
                if cos_url:
                    remainder = cos_url_re.sub("", remainder)
                if old_cos_download_row and re.fullmatch(
                        r"[ \t|·:*_`-]*(?:COS[ \t]+)?(?:下载|Download(?:[ \t]+COS)?|"
                        r"下载已验证归档文件)[ \t:：|·*_`-]*\r?\n?",
                        remainder, re.IGNORECASE):
                    remainder = ""
                if empty_cos_metadata or not remainder.strip(" \t\r\n|·-*"):
                    index += 1
                    continue
                line = remainder
            result.append(line)
            index += 1
        visible = "".join(result).rstrip()
        return visible

    @staticmethod
    def _strip_legacy_lifecycle_display_fields(
        visible, *, remove_current_deployment=False,
    ):
        """Remove recognized top-level legacy display fields only."""
        if not isinstance(visible, str):
            return visible
        lines = visible.splitlines()
        result = []
        index = 0
        details_depth = 0
        code_fence = None
        deployment_heading = re.compile(r"^### Current Deployment[ \t]*$")
        table_rule = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")
        last_event = re.compile(
            r"^\s*(?:\*\*)?Last lifecycle event:(?:\*\*)?(?:\s+.*)?$",
        )
        known_empty_deployment = {
            "Deployment records exist; details unavailable.",
            "No deployment has been performed yet.",
            "No deployment record is available for this round yet.",
            "Deployment completed; deployment records unavailable.",
        }

        def update_details_depth(line):
            nonlocal details_depth
            details_depth += len(re.findall(r"<details\b[^>]*>", line, re.IGNORECASE))
            details_depth -= len(re.findall(r"</details\s*>", line, re.IGNORECASE))
            details_depth = max(0, details_depth)

        while index < len(lines):
            fence_match = re.match(r"^\s*(`{3,}|~{3,})", lines[index])
            if code_fence is not None:
                result.append(lines[index])
                if (fence_match and fence_match.group(1)[0] == code_fence[0]
                        and len(fence_match.group(1)) >= code_fence[1]):
                    code_fence = None
                index += 1
                continue
            if fence_match:
                result.append(lines[index])
                code_fence = (fence_match.group(1)[0], len(fence_match.group(1)))
                index += 1
                continue
            if details_depth == 0 and last_event.fullmatch(lines[index]):
                index += 1
                continue
            if (remove_current_deployment and details_depth == 0
                    and deployment_heading.fullmatch(lines[index])):
                following = index + 1
                while following < len(lines) and not lines[following].strip():
                    following += 1
                end = None
                if following < len(lines) and lines[following].lstrip().startswith("|"):
                    cells = [cell.strip() for cell in lines[following].strip().strip("|").split("|")]
                    if (len(cells) >= 2 and cells[0].casefold() == "machine"
                            and cells[1].casefold() == "ip"
                            and following + 1 < len(lines)
                            and table_rule.fullmatch(lines[following + 1])):
                        end = following + 2
                        while end < len(lines) and lines[end].lstrip().startswith("|"):
                            end += 1
                elif (following < len(lines)
                      and lines[following].strip() in known_empty_deployment):
                    end = following + 1
                if end is None:
                    logger.warning(
                        "Unrecognized top-level Current Deployment section; preserving it",
                    )
                else:
                    index = end
                    while index < len(lines) and not lines[index].strip():
                        index += 1
                    if result and index < len(lines):
                        result.append("")
                    continue
            result.append(lines[index])
            update_details_depth(lines[index])
            index += 1
        return "\n".join(result).strip()

    async def _render_evidence_projection(self, repo, pr_number, state, visible):
        renderer = getattr(comments_mod, "render_evidence_section", None)
        if renderer is None:
            visible, history = self._extract_visible_history(visible)
            visible = self._strip_legacy_lifecycle_display_fields(visible)
            return visible.rstrip() + ("\n\n" + history if history else "")
        previous_visible = visible
        records = await self.proxy.read_evidence_records(repo, pr_number)
        archives = [r["data"] for r in records if r["kind"] == "round"]
        attachments = [r["data"] for r in records if r["kind"] == "attachment"]
        trusted_records_by_key = {
            (record.get("kind"), record.get("key")): record
            for record in records
            if isinstance(record, dict) and isinstance(record.get("key"), str)
        }
        latest_by_identity = {}
        deleted_by_identity = {}
        attachment_id_identities: dict[str, set[tuple[str, int, str, str, str]]] = {}

        def attachment_is_deleted(row):
            return bool(row.get("deleted_at")) or row.get("status") == "deleted"

        for row in attachments:
            attachment_id = row.get("attachment_id") if isinstance(row, dict) else None
            if isinstance(attachment_id, str):
                # read_evidence_records is chronologically ordered; this is
                # the same Latest rule used by EvidenceService.
                identity = (
                    row.get("repo"), row.get("pr_number"), row.get("head_sha"),
                    row.get("round_id"), attachment_id,
                )
                attachment_id_identities.setdefault(attachment_id, set()).add(identity)
                latest_by_identity[identity] = row
                if attachment_is_deleted(row):
                    # The download API treats an attachment ID as permanently
                    # deleted once a trusted tombstone exists for that identity.
                    deleted_by_identity[identity] = row
        current_round = round_id_for_state(state)
        current_rows = []
        for identity, row in latest_by_identity.items():
            record_repo, record_pr, head_sha, round_id, attachment_id = identity
            if ((record_repo, record_pr, head_sha, round_id)
                    != (repo, pr_number, state.get("head_sha"), current_round)):
                continue
            current_rows.append(deleted_by_identity.get(identity, row))

        def attachment_key_inputs(row):
            object_key = row.get("object_key")
            filename = row.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                filename = (object_key.rsplit("/", 1)[-1]
                            if isinstance(object_key, str) else "")
            uploaded_at = None
            for field in ("created_at", "uploaded_at"):
                value = row.get(field)
                if not isinstance(value, str):
                    continue
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                    uploaded_at = value
                    break
            display_uploaded_at = uploaded_at
            if uploaded_at is None and isinstance(object_key, str):
                parts = object_key.split("/")
                if len(parts) == 10 and parts[0] == "phanthymotus_pr":
                    try:
                        # This date-only fallback is used solely to verify the
                        # canonical object path; it is never exposed as upload time.
                        uploaded_at = datetime.fromisoformat(
                            parts[3] + "T00:00:00+08:00",
                        ).isoformat()
                    except ValueError:
                        pass
            return uploaded_at, filename, display_uploaded_at

        def verified_evidence_row(
            row, head_sha, round_id, *, archived_invite_provenance=False,
        ):
            if not isinstance(row, dict):
                return False
            if (row.get("repo") != repo or type(row.get("pr_number")) is not int
                    or row.get("pr_number") != pr_number):
                return False
            invitation_upload = (
                row.get("upload_method") == "signed_invite"
                and row.get("upload_link_authorized") is True
                and row.get("actor_verified") is False
                and row.get("github_user_authenticated") is False
                and row.get("cos_content_verified") is True
            )
            archived_invitation_upload = (
                archived_invite_provenance
                and row.get("upload_method") == "signed_invite"
                and row.get("actor") == "shared-link"
            )
            invitation_upload = invitation_upload or archived_invitation_upload
            if row.get("upload_method") == "signed_invite" and not invitation_upload:
                return False
            if (row.get("head_sha") != head_sha
                    or row.get("round_id") != round_id
                    or not isinstance(head_sha, str)
                    or not isinstance(round_id, str)
                    or re.fullmatch(re.escape(head_sha) + r"-[0-9]+", round_id) is None
                    or not (row.get("actor_verified") is True or invitation_upload)):
                return False
            if row.get("status") == "text":
                source_id = row.get("source_comment_id")
                digest = row.get("sha256")
                return (type(source_id) is int and source_id > 0
                        and isinstance(digest, str)
                        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None)
            if (row.get("status") != "ready" or row.get("deleted_at")
                    or type(row.get("size")) is not int
                    or not 1 <= row["size"] <= 512 * 1024 * 1024):
                return False
            digest = row.get("sha256")
            source_id = row.get("source_comment_id")
            if source_id is not None and (type(source_id) is not int or source_id <= 0):
                return False
            if (not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                return False
            if (row.get("upload_method") == "signed_invite"
                    and not archived_invite_provenance
                    and row.get("cos_content_verified") is not True):
                return False
            uploaded_at, filename, _ = attachment_key_inputs(row)
            if not isinstance(uploaded_at, str) or not isinstance(filename, str) or not filename:
                return False
            try:
                from .evidence_store import build_attachment_key
                expected_key = build_attachment_key(
                    repo, pr_number, head_sha, round_id, row.get("attachment_id", ""),
                    uploaded_at, filename,
                )
            except (ValueError, TypeError, AttributeError):
                return False
            return row.get("object_key") == expected_key

        confirmed_current = any(
            verified_evidence_row(row, state.get("head_sha"), current_round)
            for row in current_rows
        )
        if state.get("status") == "testing":
            component_ids = {c.get("component_id") for c in state.get("components", [])
                             if isinstance(c, dict) and c.get("component_id")}
            deployed_ids = _component_id_set(state.get("deployments", []))
            deployments_complete = bool(component_ids) and component_ids.issubset(deployed_ids)
            workflow = comments_mod._workflow_lines(
                "testing", evidence_complete=confirmed_current,
                deployments_complete=deployments_complete,
                record_test_complete=(
                    state.get("test_result") in {"pass", "fail"}
                    and state.get("command", {}).get("kind") == "record_test"
                    and state.get("command", {}).get("phase") == "completed"
                    and state.get("command", {}).get("args", {}).get("result")
                    == state.get("test_result")
                ),
            )
            visible = self._replace_markdown_section(
                visible, "### Workflow", workflow, "### Next action",
            )
        # Presentation flags are deliberately ephemeral. Hidden state has a
        # strict schema and must never learn about browser feature config.
        render_state = copy.deepcopy(state)
        store = getattr(self.evidence_service, "store", None)
        try:
            pr = await self.proxy.get_pr(repo, pr_number)
        except Exception as exc:
            logger.warning("Evidence PR context unavailable %s#%s exception=%s",
                           repo, pr_number, type(exc).__name__)
            raise GitHubStateProxyError(
                "Evidence projection could not verify current PR context",
            ) from None
        if (not isinstance(pr, dict) or pr.get("state") not in {"open", "closed"}
                or pr.get("base", {}).get("repo", {}).get("full_name", "").lower() != repo.lower()
                or not isinstance(pr.get("head"), dict)
                or pr.get("head", {}).get("sha") != state.get("head_sha")):
            raise GitHubStateProxyError(
                "Evidence projection could not verify current PR head",
            )
        if state.get("status") == "testing":
            acceptance = state.get("acceptance_request")
            acceptance_pending = (
                isinstance(acceptance, dict)
                and acceptance.get("status") == "pending"
                and acceptance.get("head_sha") == state.get("head_sha")
                and acceptance.get("round_id") == current_round
            )
            reviewers_configured, eligible_reviewers = self._eligible_acceptance_reviewers(
                repo, pr,
            )
            fresh_testing = comments_mod.testing(
                repo, pr_number, state.get("head_sha", ""),
                deployments=self._enrich_deployments_for_render(
                    state.get("deployments", []), repo,
                ),
                components=state.get("components", []),
                evidence_complete=confirmed_current,
                deployments_complete=deployments_complete,
                acceptance_pending=acceptance_pending,
                acceptance_reviewers_configured=reviewers_configured,
                acceptance_reviewers=eligible_reviewers,
            )
            action_heading = re.search(r"(?m)^### Next action[ \t]*$", fresh_testing)
            if action_heading is not None:
                action_start = action_heading.end()
                next_template_heading = re.search(
                    r"(?m)^#{1,6}[ \t]+[^\r\n]+", fresh_testing[action_start:],
                )
                action_end = (action_start + next_template_heading.start()
                              if next_template_heading else len(fresh_testing))
                next_action = fresh_testing[action_start:action_end].strip()
                if acceptance_pending and not eligible_reviewers:
                    reviewer_guidance = self._acceptance_reviewer_guidance(
                        repo, pr, reviewers_configured, eligible_reviewers,
                    )
                    if reviewers_configured:
                        next_action = "\n\n".join((
                            reviewer_guidance,
                            "由当前仓库验收白名单中的获授权账号输入 `/record_test result=pass` 或 `/record_test result=fail`。",
                        ))
                    else:
                        next_action = "\n\n".join((
                            reviewer_guidance,
                            "配置有效的当前仓库验收账号后，获授权验收人才能输入 `/record_test result=pass` 或 `/record_test result=fail`。",
                        ))
                action_start = visible.find("### Next action")
                action_content_start = (action_start + len("### Next action")
                                        if action_start >= 0 else -1)
                next_heading = "### Current Deployment"
                if (action_content_start >= 0
                        and visible.find(next_heading, action_content_start) < 0):
                    candidates = [
                        match.start() + action_content_start
                        for match in re.finditer(
                            r"(?m)^#{1,6}[ \t]+[^\r\n]+",
                            visible[action_content_start:],
                        )
                    ]
                    candidates.extend(
                        position for marker in (
                            "<!-- evidence-projection:start -->",
                            VISIBLE_HISTORY_START_MARKER,
                        )
                        if (position := visible.find(marker, action_content_start)) >= 0
                    )
                    next_heading = (
                        visible[min(candidates):].splitlines()[0]
                        if candidates else None
                    )
                visible = self._replace_markdown_section(
                    visible, "### Next action", next_action, next_heading,
                )
        try:
            from .evidence_auth import readiness
            status = readiness(self.config, self.evidence_service)
            backend_ready = (status["EVIDENCE_BACKEND_READY"]
                             and callable(getattr(store, "dispatch", None)))
            browser_upload_ready = (
                backend_ready and status["PUBLIC_HTTPS_INGRESS_READY"]
                and state.get("status") == "testing"
                and pr.get("state") == "open" and not pr.get("merged")
                and not pr.get("draft")
            )
        except Exception:
            status = {"status": "EVIDENCE_BACKEND_PENDING",
                      "PUBLIC_HTTPS_INGRESS_READY": False}
            backend_ready = False
            browser_upload_ready = False
        has_evidence_history = any(
            isinstance(row, dict) and isinstance(row.get("head_sha"), str)
            and isinstance(row.get("round_id"), str)
            and re.fullmatch(re.escape(row["head_sha"]) + r"-[0-9]+", row["round_id"])
            for row in attachments
        ) or any(
            isinstance(archive.get("snapshot"), dict)
            and isinstance(archive["snapshot"].get("attachments"), list)
            and bool(archive["snapshot"]["attachments"])
            for archive in archives if isinstance(archive, dict)
        ) or any(
            isinstance(archive, dict)
            and isinstance(archive.get("attachment_records"), list)
            and any(
                isinstance(wrapper, dict)
                and wrapper.get("kind") == "attachment"
                and isinstance(wrapper.get("data"), dict)
                and wrapper["data"].get("repo") == repo
                and type(wrapper["data"].get("pr_number")) is int
                and wrapper["data"].get("pr_number") == pr_number
                and wrapper["data"].get("head_sha") == archive.get("head_sha")
                and wrapper["data"].get("round_id") == archive.get("round_id")
                for wrapper in archive["attachment_records"]
            )
            for archive in archives
        )
        attachment_access_allowed = (
            state.get("status") in {"testing", "succeeded", "failed"}
            or has_evidence_history
        )
        render_state.update(
            evidence_enabled=bool(backend_ready and attachment_access_allowed),
            browser_upload_ready=bool(browser_upload_ready),
            evidence_readiness=status,
            enabled=bool(backend_ready and attachment_access_allowed),
            pr_state=pr.get("state"),
            merged=bool(pr.get("merged")),
            attachments=current_rows,
        )
        round_numbers = trusted_round_numbers(records, state, repo, pr_number)
        verified_current_round = (
            state.get("status") in {"testing", "succeeded", "failed"}
            and current_round in round_numbers
        )
        if verified_current_round:
            # This display-only copy may use current authorized machine data
            # when a legacy current-round deployment row has no captured IP.
            render_state["deployments"] = self._enrich_deployments_for_render(
                state.get("deployments", []), repo,
            )
        if current_round in round_numbers:
            render_state["round_number"] = round_numbers[current_round]
        bundles = []
        seen_rounds = set()
        configured_public_base = getattr(self.config, "evidence_public_base_url", "")
        public_base_url = (configured_public_base
                           if is_stable_public_base_url(configured_public_base) else "")
        upload_public_base_url = (
            configured_public_base
            if is_evidence_ingress_url(configured_public_base) else ""
        )
        historical_cos_sign_budget = [24]
        from .evidence_auth import (
            DOWNLOAD_PURPOSE, UPLOAD_MAX_LIFETIME_SECONDS, UPLOAD_PURPOSE,
            issue_capability, verify_capability,
        )

        # Reuse only long-lived COS URLs from the previous trusted lifecycle
        # projection. They are matched to their exact decoded object key and
        # checked for the configured HTTPS bucket host and COS signature fields.
        # Signed URLs remain ephemeral presentation data; they are never saved
        # into hidden state or evidence records.
        store = getattr(self.evidence_service, "store", None)
        cached_cos_urls = {}
        inferred_cos_download_urls = set()
        if backend_ready and store is not None:
            for match in re.finditer(r"<(?P<url>https://[^<>\s]+)>", previous_visible):
                candidate = match.group("url")
                try:
                    from urllib.parse import unquote, urlsplit
                    object_key = unquote(urlsplit(candidate).path.lstrip("/"))
                    expiry = store.download_url_expiry(candidate, key=object_key)
                except (AttributeError, TypeError, ValueError):
                    continue
                if expiry is not None and expiry > cached_cos_urls.get(object_key, (0, ""))[0]:
                    cached_cos_urls[object_key] = (expiry, candidate)

        async def stable_cos_url(
            object_key, filename, *, attachment_id=None, sign_budget=None,
        ):
            if not backend_ready or store is None:
                return ""
            cached = cached_cos_urls.get(object_key)
            now = int(time.time())
            if cached:
                checked_expiry = store.download_url_expiry(
                    cached[1], key=object_key, filename=filename,
                    attachment_id=attachment_id,
                )
                if checked_expiry is not None:
                    cached = (checked_expiry, cached[1])
                    cached_cos_urls[object_key] = cached
                    if checked_expiry - now > 7 * 24 * 60 * 60:
                        return cached[1]
                else:
                    cached = None
            if sign_budget is not None and sign_budget[0] <= 0:
                return cached[1] if cached and cached[0] > now else ""
            if sign_budget is not None:
                sign_budget[0] -= 1
            try:
                candidate = await store._io(
                    store.issue_download_url, key=object_key, filename=filename,
                )
                expiry = store.download_url_expiry(
                    candidate, key=object_key, filename=filename,
                    attachment_id=attachment_id,
                )
                if expiry is not None and expiry > now:
                    cached_cos_urls[object_key] = (expiry, candidate)
                    return candidate
            except Exception as exc:
                logger.warning("COS evidence link refresh failed %s#%s exception=%s",
                               repo, pr_number, type(exc).__name__)
            # A transient signing failure must not erase a still-valid link,
            # including one inside the renewal window.
            if cached and cached[0] > now:
                return cached[1]
            return ""

        async def bind_download_capabilities(
            rows, head_sha, round_id, *, include_cos_urls=True,
            cos_sign_budget=None, legacy_signed_invite_identities=None,
        ):
            bound = []
            legacy_signed_invite_identities = legacy_signed_invite_identities or set()
            for row in rows:
                item = copy.deepcopy(row)
                for time_field in ("created_at", "uploaded_at"):
                    value = item.get(time_field)
                    if value is None:
                        continue
                    try:
                        parsed_time = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except (AttributeError, TypeError, ValueError):
                        item.pop(time_field, None)
                        continue
                    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
                        item.pop(time_field, None)
                _, _, trusted_display_time = attachment_key_inputs(item)
                if trusted_display_time is None:
                    item.pop("created_at", None)
                    item.pop("uploaded_at", None)
                attachment_id = item.get("attachment_id")
                identity = (repo, pr_number, head_sha, round_id, attachment_id)
                valid_row = (
                    verified_evidence_row(
                        item, head_sha, round_id,
                        archived_invite_provenance=(
                            identity in legacy_signed_invite_identities
                        ),
                    )
                    and item.get("head_sha") == head_sha
                    and item.get("round_id") == round_id
                    and item.get("status") == "ready"
                    and isinstance(item.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")) is not None
                )
                if valid_row:
                    uploaded_at, filename, display_uploaded_at = attachment_key_inputs(item)
                    try:
                        from .evidence_store import build_attachment_key
                        expected_key = build_attachment_key(
                            repo, pr_number, head_sha, round_id, attachment_id,
                            uploaded_at, filename,
                        )
                    except (ValueError, TypeError, AttributeError):
                        valid_row = False
                    else:
                        valid_row = item.get("object_key") == expected_key
                        if valid_row:
                            if (not isinstance(item.get("filename"), str)
                                    or not item["filename"].strip()):
                                item["filename"] = filename
                            if display_uploaded_at is None:
                                # These fields are presentation timestamps; a
                                # path-derived date must never be presented as
                                # the real upload time.
                                item.pop("created_at", None)
                                item.pop("uploaded_at", None)
                item["cos_projection_verified"] = bool(valid_row)
                if valid_row and identity in legacy_signed_invite_identities:
                    # Render-only evidence from the paired trusted archive
                    # operation authorizes this compatibility projection.
                    item["cos_content_verified"] = True
                if (valid_row and isinstance(attachment_id, str)
                        and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attachment_id)):
                    controller_download_compatible = False
                    created_at = item.get("created_at")
                    if (len(attachment_id_identities.get(attachment_id, set())) == 1
                            and isinstance(created_at, str)):
                        try:
                            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            if created.tzinfo is not None and created.utcoffset() is not None:
                                from .evidence_store import build_attachment_key
                                controller_download_compatible = (
                                    build_attachment_key(
                                        repo, pr_number, head_sha, round_id,
                                        attachment_id, created_at, filename,
                                    ) == expected_key
                                )
                        except (ValueError, TypeError, AttributeError):
                            controller_download_compatible = False
                    if controller_download_compatible:
                        item["download_capability"] = issue_capability(
                            self.config, repo, pr_number, head_sha, round_id,
                            purpose=DOWNLOAD_PURPOSE,
                            resource="attachment:" + attachment_id,
                        )
                    else:
                        # The Controller download API reconstructs object keys
                        # from created_at. For older records without that
                        # trusted timestamp, use the verified object key below.
                        item.pop("download_capability", None)
                    if include_cos_urls or not controller_download_compatible:
                        cos_url = await stable_cos_url(
                            expected_key, item.get("filename", ""),
                            attachment_id=attachment_id,
                            sign_budget=cos_sign_budget,
                        )
                        if cos_url:
                            item["cos_download_url"] = cos_url
                            if not controller_download_compatible:
                                item["_projection_cos_fallback"] = True
                            if display_uploaded_at is None:
                                # The renderer needs a timestamp to reproduce
                                # the dated key. This render-only date is
                                # removed from the visible upload-time cell.
                                item["created_at"] = uploaded_at
                                item["_projection_date_only"] = True
                                inferred_cos_download_urls.add(cos_url)
                        else:
                            item.pop("cos_download_url", None)
                    else:
                        # Historical downloads resolve through the existing
                        # attachment-scoped API when clicked; do not sign every
                        # archived COS object on each Summary projection.
                        item.pop("cos_download_url", None)
                else:
                    item.pop("download_capability", None)
                    item.pop("cos_download_url", None)
                bound.append(item)
            return bound

        async def bind_log_url(target, head_sha, round_id):
            metadata = target.get("cos") if isinstance(target.get("cos"), dict) else {}
            if (not backend_ready or not isinstance(metadata, dict)
                    or not self._valid_cos_metadata(repo, pr_number, head_sha, metadata)):
                target.pop("cos_log_download_url", None)
                return
            object_key = metadata["object_key"]
            filename = object_key.rsplit("/", 1)[-1]
            # Lifecycle COS metadata is persisted only after the existing
            # upload verification. Reuse or sign a URL from that trusted record
            # without downloading historical logs again on every projection.
            url = await stable_cos_url(object_key, filename)
            if url:
                target["cos_log_download_url"] = url
            else:
                target.pop("cos_log_download_url", None)

        def remove_attachment_page_download_links(rendered):
            """Keep read-page links and inferred upload dates out of file rows."""
            if not isinstance(rendered, str):
                return rendered
            page_link = re.compile(
                r"\[([^\]]+)\]\(<https?://[^<>]*[?&]view=read(?:[&#][^<>]*)?>\)",
            )
            lines = []
            for line in rendered.split("\n"):
                if line.lstrip().startswith("|"):
                    if any(url in line for url in inferred_cos_download_urls):
                        cells = line.split("|")
                        if len(cells) >= 5:
                            cells[-2] = " 时间未记录 "
                            line = "|".join(cells)
                    line = page_link.sub(r"\1", line)
                lines.append(line)
            return "\n".join(lines)

        if backend_ready and attachment_access_allowed:
            render_state["capability"] = issue_capability(
                self.config, repo, pr_number, state["head_sha"], current_round,
                purpose=DOWNLOAD_PURPOSE, resource="page",
            )
            if browser_upload_ready and state.get("status") == "testing":
                upload_capability = ""
                page_path = comments_mod._evidence_path(repo, pr_number).replace(
                    "/api/evidence/", "/evidence/", 1,
                )
                try:
                    configured_origin = urlsplit(upload_public_base_url)
                    now = int(time.time())
                    for match in re.finditer(
                            r"\[打开附件页\]\(<(?P<url>https://[^<>\s]+)>\)", previous_visible):
                        candidate = urlsplit(match.group("url"))
                        if (candidate.scheme != "https"
                                or candidate.netloc.casefold() != configured_origin.netloc.casefold()
                                or candidate.path != page_path or candidate.fragment):
                            continue
                        query = parse_qs(candidate.query, strict_parsing=True)
                        values = query.get("cap")
                        if set(query) != {"cap"} or not isinstance(values, list) or len(values) != 1:
                            continue
                        token = values[0]
                        claims = verify_capability(
                            self.config, token, repo, pr_number, purpose=UPLOAD_PURPOSE,
                        )
                        if (claims.get("head") == state.get("head_sha")
                                and claims.get("round") == current_round
                                and claims.get("status") == "testing"
                                and claims.get("expires_at", 0) > now):
                            upload_capability = token
                            break
                except Exception:
                    # Old, expired or malformed Summary capabilities are never
                    # trusted; issue a fresh one bound to the current round.
                    upload_capability = ""
                if not upload_capability:
                    try:
                        upload_capability = issue_capability(
                            self.config, repo, pr_number, state["head_sha"], current_round,
                            purpose=UPLOAD_PURPOSE,
                            expires_at=int(time.time()) + UPLOAD_MAX_LIFETIME_SECONDS,
                        )
                    except (OverflowError, TypeError, ValueError):
                        logger.warning("Evidence upload capability unavailable %s#%s",
                                       repo, pr_number)
                if upload_capability:
                    render_state["upload_capability"] = upload_capability
            render_state["log_capability"] = issue_capability(
                self.config, repo, pr_number, state["head_sha"], current_round,
                purpose=DOWNLOAD_PURPOSE, resource="log:" + current_round,
            )
        if backend_ready and attachment_access_allowed:
            current_rows = await bind_download_capabilities(current_rows, state["head_sha"], current_round)
            render_state["attachments"] = current_rows
            await bind_log_url(render_state, state["head_sha"], current_round)
            render_state["deployment_logs"] = self.evidence_service._trusted_deployment_logs(
                render_state, repo, pr_number, state["head_sha"], current_round,
            )
        trusted_archives = []
        testing_statuses = {"testing", "succeeded", "failed"}
        for archive in archives:
            if not isinstance(archive, dict):
                continue
            archive_round = archive.get("round_id")
            if not isinstance(archive_round, str):
                continue
            archive_head = archive.get("head_sha")
            snapshot_value = archive.get("snapshot")
            round_number = round_numbers.get(archive_round)
            if (type(round_number) is not int or round_number <= 0
                    or not isinstance(archive_head, str)
                    or not isinstance(snapshot_value, dict)
                    or snapshot_value.get("head_sha") != archive_head
                    or snapshot_value.get("status") not in testing_statuses
                    or archive_round == current_round):
                continue
            try:
                snapshot_round = round_id_for_state(snapshot_value)
            except Exception:
                continue
            if (snapshot_round != archive_round
                    or not archive_round.startswith(archive_head + "-")
                    or archive.get("repo") != repo
                    or type(archive.get("pr_number")) is not int
                    or archive.get("pr_number") != pr_number):
                continue
            trusted_archives.append(archive)

        for archive in trusted_archives:
            if archive["round_id"] in seen_rounds or archive["round_id"] == current_round:
                continue
            seen_rounds.add(archive["round_id"])
            snapshot = copy.deepcopy(archive["snapshot"])
            if backend_ready and attachment_access_allowed:
                snapshot["log_capability"] = issue_capability(
                    self.config, repo, pr_number, snapshot["head_sha"], archive["round_id"],
                    purpose=DOWNLOAD_PURPOSE, resource="log:" + archive["round_id"],
                )
                snapshot["deployment_logs"] = self.evidence_service._trusted_deployment_logs(
                    snapshot, repo, pr_number, archive["head_sha"], archive["round_id"],
                )
            snapshot["deployments"] = self._enrich_deployments_for_render(
                snapshot.get("deployments", []), repo, historical=True,
                historical_machine_groups=snapshot.get("machine_groups"),
            )
            archive_head = archive["head_sha"]
            archive_round = archive["round_id"]
            archive_attachment_rows = {}
            archive_deleted_rows = {}
            archive_attachment_wrappers = {}
            archive_operation_rows = {}
            archive_operation_wrappers = {}
            for wrapper in archive.get("attachment_records", []):
                if (not isinstance(wrapper, dict)
                        or wrapper.get("kind") not in {"attachment", "operation"}):
                    continue
                data = wrapper.get("data")
                if not isinstance(data, dict):
                    continue
                record_kind = wrapper["kind"]
                attachment_id = data.get("attachment_id")
                if (not isinstance(attachment_id, str)
                        or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attachment_id) is None
                        or not isinstance(wrapper.get("key"), str)
                        or not isinstance(wrapper.get("sha256"), str)
                        or re.fullmatch(r"[0-9a-f]{64}", wrapper["sha256"]) is None
                        or data.get("repo") != repo
                        or type(data.get("pr_number")) is not int
                        or data.get("pr_number") != pr_number
                        or data.get("head_sha") != archive_head
                        or data.get("round_id") != archive_round):
                    continue
                identity = (repo, pr_number, archive_head, archive_round, attachment_id)
                if record_kind == "attachment":
                    # Archive wrappers were captured in read_evidence_records
                    # order, so overwriting mirrors EvidenceService Latest.
                    archive_attachment_rows[identity] = data
                    archive_attachment_wrappers[identity] = wrapper
                    if attachment_is_deleted(data):
                        archive_deleted_rows[identity] = data
                else:
                    archive_operation_rows[identity] = data
                    archive_operation_wrappers[identity] = wrapper

            def wrapper_matches_trusted_record(wrapper, kind):
                if not isinstance(wrapper, dict):
                    return False
                record = trusted_records_by_key.get((kind, wrapper.get("key")))
                return bool(
                    record is not None
                    and record.get("sha256") == wrapper.get("sha256")
                    and record.get("data") == wrapper.get("data")
                )

            def archived_invite_proof(identity, attachment_row):
                """Require paired, integrity-bound archive records for legacy invite rows."""
                attachment_wrapper = archive_attachment_wrappers.get(identity)
                operation_wrapper = archive_operation_wrappers.get(identity)
                operation = archive_operation_rows.get(identity)
                if (not wrapper_matches_trusted_record(attachment_wrapper, "attachment")
                        or not wrapper_matches_trusted_record(operation_wrapper, "operation")
                        or not isinstance(operation, dict)
                        or operation.get("status") != "ready"
                        or operation.get("upload_method") != "signed_invite"
                        or operation.get("upload_link_authorized") is not True
                        or operation.get("actor_verified") is not False
                        or operation.get("github_user_authenticated") is not False
                        or operation.get("actor") != "shared-link"
                        or operation.get("cos_content_verified") is not True
                        or attachment_row.get("upload_method") != "signed_invite"
                        or attachment_row.get("actor") != "shared-link"):
                    return False
                # Missing legacy fields are supplied only by the paired,
                # trusted operation record. Any present contradiction rejects
                # the compatibility path; absence itself is never a grant.
                for field, expected in (
                    ("upload_link_authorized", True),
                    ("actor_verified", False),
                    ("github_user_authenticated", False),
                    ("cos_content_verified", True),
                ):
                    if field in attachment_row and attachment_row[field] is not expected:
                        return False
                _, filename, _ = attachment_key_inputs(attachment_row)
                return all(
                    operation.get(field) == value
                    for field, value in (
                        ("repo", repo), ("pr_number", pr_number),
                        ("head_sha", identity[2]), ("round_id", identity[3]),
                        ("attachment_id", identity[4]),
                        ("object_key", attachment_row.get("object_key")),
                        ("filename", filename), ("size", attachment_row.get("size")),
                        ("sha256", attachment_row.get("sha256")),
                    )
                )

            snapshot_attachment_map = {}
            legacy_signed_invite_identities = set()
            identities = list(archive_attachment_rows)
            identities.extend(
                identity for identity in latest_by_identity
                if identity[:4] == (repo, pr_number, archive_head, archive_round)
            )
            identities = dict.fromkeys(identities)
            for identity in identities:
                archived_row = archive_attachment_rows.get(identity)
                selected = (
                    deleted_by_identity.get(identity)
                    or archive_deleted_rows.get(identity)
                    or latest_by_identity.get(identity)
                    or archived_row
                )
                if selected is None:
                    continue
                attachment_id = identity[4]
                snapshot_attachment_map[identity] = copy.deepcopy(selected)
                if (archived_row is not None and selected == archived_row
                        and archived_invite_proof(identity, selected)):
                    legacy_signed_invite_identities.add(identity)
            snapshot_attachments = list(snapshot_attachment_map.values())
            snapshot["attachments"] = (await bind_download_capabilities(
                snapshot_attachments, snapshot["head_sha"], archive["round_id"],
                # Prefer the scoped Controller download. Signed COS links are
                # a bounded fallback when no stable HTTPS Controller origin is
                # available for historical attachment downloads.
                include_cos_urls=not bool(public_base_url or upload_public_base_url),
                cos_sign_budget=historical_cos_sign_budget,
                legacy_signed_invite_identities=legacy_signed_invite_identities,
            ) if backend_ready and attachment_access_allowed else snapshot_attachments)
            # The archive's nested wrappers were reconciled with current
            # Latest records into this render-only snapshot. Do not let the
            # renderer append stale immutable revisions a second time.
            bundle = {
                key: value for key, value in archive.items()
                if key != "attachment_records"
            }
            bundle["snapshot"] = snapshot
            if archive.get("round_id") in round_numbers:
                bundle["round_number"] = round_numbers[archive["round_id"]]
            bundles.append(bundle)
        download_url_expiry = getattr(store, "download_url_expiry", None)
        def validate_cos_download_url(url, *, key, filename, attachment_id):
            if not callable(download_url_expiry):
                return None
            return download_url_expiry(
                url, key=key, filename=filename, attachment_id=attachment_id,
            )
        section = renderer(repo, pr_number, render_state, current_rows[-20:], bundles,
                           public_base_url=public_base_url,
                           upload_public_base_url=(
                               upload_public_base_url
                               if backend_ready and attachment_access_allowed else ""
                           ),
                           cos_download_url_expiry=validate_cos_download_url)
        if not isinstance(section, str):
            raise GitHubStateProxyError("Evidence renderer must return markdown")
        section = remove_attachment_page_download_links(section)
        # Refresh existing sections before removing duplicate top-level
        # presentation blocks. History and the old projection marker provide
        # safe boundaries for older Summaries whose deployment table is gone.
        visible = self._strip_legacy_evidence_blocks(visible)
        visible, history = self._extract_visible_history(visible)
        visible = self._strip_legacy_lifecycle_display_fields(
            visible,
            remove_current_deployment=verified_current_round or bool(trusted_archives),
        )
        start, end = "<!-- evidence-projection:start -->", "<!-- evidence-projection:end -->"
        def projection_body(rendered_section):
            projection = start + "\n" + rendered_section + "\n" + end
            result = visible.rstrip() + "\n\n" + projection
            if history:
                result += "\n\n" + history
            return result

        body = projection_body(section)
        if _count_visible_bytes(body) > _MAX_VISIBLE_LIFECYCLE_BYTES:
            # Keep every trusted test round addressable while reducing repeated
            # attachment notes, deployment tables, and COS links. The existing
            # read-only page reconstructs the full round from immutable records.
            compact_current_rows = []
            for row in current_rows[-20:]:
                item = copy.deepcopy(row)
                if item.pop("_projection_date_only", False):
                    item.pop("created_at", None)
                    item.pop("uploaded_at", None)
                for field in (
                    "note", "text", "display_note", "display_note_complete",
                    "evidence_kind", "cos_download_url",
                ):
                    item.pop(field, None)
                compact_current_rows.append(item)
            compact_state = copy.deepcopy(render_state)
            compact_state["attachments"] = compact_current_rows
            compact_machines = sorted({
                deployment.get("machine") for deployment in render_state.get("deployments", [])
                if isinstance(deployment, dict)
                and deployment.get("phase") == "deployed"
                and isinstance(deployment.get("machine"), str)
            })
            compact_state["deployments"] = [
                {"machine": machine, "phase": "deployed"}
                for machine in compact_machines
            ]
            source_machine_logs = render_state.get("deployment_logs", {})
            compact_log_machines = {
                machine for machine, record in source_machine_logs.items()
                if isinstance(source_machine_logs, dict)
                and isinstance(machine, str) and isinstance(record, dict)
                and record.get("head_sha") == state["head_sha"]
                and record.get("round_id") == current_round
                and record.get("status") in {"pending", "ready", "unavailable"}
            } if isinstance(source_machine_logs, dict) else set()
            compact_state["deployment_logs"] = {
                machine: {
                    field: source_machine_logs[machine][field]
                    for field in ("head_sha", "round_id", "status", "capability")
                    if field in source_machine_logs[machine]
                }
                for machine in compact_log_machines
            }
            compact_state["_compact_projection"] = True
            compact_state["components"] = []
            compact_state["machine_groups"] = []
            compact_state["case_results"] = {}
            # The compact current card names its role first; retain the real
            # ordinal explicitly in the Summary immediately below it.
            compact_state.pop("round_number", None)
            compact_section = renderer(
                repo, pr_number, compact_state, compact_current_rows, [],
                public_base_url=public_base_url,
                upload_public_base_url=(
                    upload_public_base_url
                    if backend_ready and attachment_access_allowed else ""
                ),
                cos_download_url_expiry=validate_cos_download_url,
            )
            if not isinstance(compact_section, str):
                raise GitHubStateProxyError(
                    "Compact Evidence renderer must return markdown",
                )
            compact_section = remove_attachment_page_download_links(compact_section)
            compact_round_number = round_numbers.get(current_round)
            if type(compact_round_number) is int and compact_round_number > 0:
                compact_section = compact_section.replace(
                    "<summary>当前轮（当前轮次） ·",
                    f"<summary>当前轮次（第 {compact_round_number} 轮） ·",
                    1,
                )
            evidence_end = "<!-- deploy-approval-evidence:end -->"
            end_pos = compact_section.rfind(evidence_end)
            if end_pos < 0:
                raise GitHubStateProxyError(
                    "Compact Evidence renderer omitted its end marker",
                )

            def archive_time(archive):
                snapshot_value = archive.get("snapshot", {})
                for value in (
                    snapshot_value.get("round_started_at"),
                    archive.get("started_at"), snapshot_value.get("started_at"),
                    archive.get("archived_at"), archive.get("ended_at"),
                ):
                    if not isinstance(value, str):
                        continue
                    try:
                        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                        return parsed
                return None

            def archive_time_label(parsed):
                if parsed is None:
                    return "时间未知"
                return parsed.astimezone(ZoneInfo("Asia/Shanghai")).strftime(
                    "%Y-%m-%d %H:%M（北京时间）",
                )

            def archive_sort_key(archive):
                parsed = archive_time(archive)
                timestamp = parsed.timestamp() if parsed is not None else float("-inf")
                return timestamp, round_numbers[archive["round_id"]]

            compact_archives = sorted(
                trusted_archives, key=archive_sort_key, reverse=True,
            )
            compatibility_downloads_by_round = {}

            def compatibility_download_lines(rows, round_label):
                output = []
                seen = set()
                for row in rows:
                    if (not isinstance(row, dict)
                            or row.get("_projection_cos_fallback") is not True
                            or row.get("status") != "ready"
                            or row.get("deleted_at")
                            or row.get("cos_projection_verified") is not True
                            or (row.get("cos_content_verified") is not True
                                and not (row.get("actor_verified") is True
                                         and row.get("upload_method") != "signed_invite"))
                            or not isinstance(row.get("sha256"), str)
                            or re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) is None
                            or type(row.get("size")) is not int
                            or not 1 <= row["size"] <= 512 * 1024 * 1024):
                        continue
                    attachment_id = row.get("attachment_id")
                    filename = row.get("filename")
                    object_key = row.get("object_key")
                    url = row.get("cos_download_url")
                    if (not isinstance(attachment_id, str)
                            or not isinstance(filename, str)
                            or not isinstance(object_key, str)
                            or not isinstance(url, str)
                            or validate_cos_download_url(
                                url, key=object_key, filename=filename,
                                attachment_id=attachment_id,
                            ) is None):
                        continue
                    identity = (row.get("repo"), row.get("pr_number"),
                                row.get("head_sha"), row.get("round_id"), attachment_id)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    label = comments_mod._escape(filename)
                    label = re.sub(r"([\\\[\]])", r"\\\1", label)
                    output.append(
                        f"- {round_label}：[{label}](<{url}>)",
                    )
                return output

            compatibility_downloads_by_round[current_round] = compatibility_download_lines(
                current_rows,
                (f"第 {round_numbers[current_round]} 轮"
                 if current_round in round_numbers else "当前轮"),
            )
            bundle_snapshots_by_round = {}
            for bundle in bundles:
                bundle_round = bundle.get("round_id")
                bundle_snapshot = bundle.get("snapshot")
                if not isinstance(bundle_snapshot, dict):
                    continue
                bundle_snapshots_by_round[bundle_round] = bundle_snapshot
                compatibility_downloads_by_round[bundle_round] = compatibility_download_lines(
                    bundle_snapshot.get("attachments", []),
                    f"第 {round_numbers[bundle_round]} 轮",
                )
            page_path = comments_mod._evidence_path(repo, pr_number).replace(
                "/api/evidence/", "/evidence/", 1,
            )
            read_origin = (public_base_url or upload_public_base_url).rstrip("/")
            if not read_origin or not page_path.startswith("/evidence/"):
                raise GitHubStateProxyError(
                    "Evidence projection cannot compact details without a trusted read page",
                )
            compact_history = [""] if compact_archives else []
            current_round_url = ""
            if read_origin and page_path.startswith("/evidence/"):
                current_round_url = read_origin + page_path + "?" + urlencode({
                    "view": "read", "head_sha": state["head_sha"],
                    "round_id": current_round,
                })

            def ensure_read_page_entry(rendered):
                if not current_round_url or "[打开附件页]" in rendered:
                    return rendered
                current_end = "<!-- deploy-approval-evidence:end -->"
                current_end_pos = rendered.rfind(current_end)
                if current_end_pos >= 0:
                    return (
                        rendered[:current_end_pos].rstrip()
                        + "\n[查看附件页](<"
                        + current_round_url + ">)\n"
                        + rendered[current_end_pos:]
                    )
                return rendered

            compact_section = ensure_read_page_entry(compact_section)
            display_status = {
                "testing": "测试中", "succeeded": "已通过", "failed": "未通过",
            }
            for archive in compact_archives:
                archive_round = archive["round_id"]
                archive_head = archive["head_sha"]
                snapshot_value = archive["snapshot"]
                round_number = round_numbers[archive_round]
                parsed_time = archive_time(archive)
                read_url = ""
                if read_origin and page_path.startswith("/evidence/"):
                    read_url = read_origin + page_path + "?" + urlencode({
                        "view": "read", "head_sha": archive_head,
                        "round_id": archive_round,
                    })
                title = (
                    f"第 {round_number} 轮 · "
                    f"{archive_time_label(parsed_time)} · "
                    f"{display_status[snapshot_value['status']]}"
                )
                compact_history.extend([
                    "<details>",
                    f"<summary>{comments_mod._escape(title)}</summary>",
                    "",
                    f"**Commit：** `{comments_mod._escape(archive_head)}`",
                ])
                if read_url:
                    compact_history.append(
                        f"[查看本轮附件、备注和部署记录](<{read_url}>)",
                    )
                else:
                    compact_history.append(
                        "本轮附件、备注和部署记录仍保存在可信归档中。",
                    )
                archive_url = comments_mod._trusted_archive_url(
                    archive.get("url", ""), repo, pr_number, public_base_url,
                )
                if archive_url:
                    compact_history.append(
                        f"[查看原始归档](<{archive_url}>)",
                    )
                compact_history.extend(
                    compatibility_downloads_by_round.get(archive_round, []),
                )
                historical_machine_logs = comments_mod._machine_log_lines(
                    bundle_snapshots_by_round.get(archive_round, {}),
                    (archive_head, archive_round), comments_mod._evidence_path(repo, pr_number),
                    read_origin,
                )
                if historical_machine_logs:
                    compact_history.extend(["", "**每机器部署日志：**", *historical_machine_logs])
                compact_history.extend(["", "</details>", ""])
            end_pos = compact_section.rfind(evidence_end)
            if end_pos < 0:
                raise GitHubStateProxyError(
                    "Compact Evidence renderer omitted its end marker after trimming",
                )
            current_fallback_lines = compatibility_downloads_by_round.get(current_round, [])
            current_fallback_block = (
                ["", "<details>", "<summary>当前轮兼容下载链接</summary>", "",
                 *current_fallback_lines, "", "</details>", ""]
                if current_fallback_lines else []
            )
            compact_section = (
                compact_section[:end_pos].rstrip()
                + ("\n" + "\n".join(current_fallback_block)
                   if current_fallback_block else "")
                + ("\n" + "\n".join(compact_history) if compact_history else "")
                + "\n" + compact_section[end_pos:]
            )
            body = projection_body(compact_section)
            if _count_visible_bytes(body) > _MAX_VISIBLE_LIFECYCLE_BYTES:
                # If even the per-round index is too large, keep an explicit
                # count and a link to the read-only page, whose context includes
                # every trusted round and its attachment/deployment records.
                minimal_state = copy.deepcopy(compact_state)
                minimal_state["attachments"] = []
                minimal_section = renderer(
                    repo, pr_number, minimal_state, [], [],
                    public_base_url=public_base_url,
                    upload_public_base_url=(
                        upload_public_base_url
                        if backend_ready and attachment_access_allowed else ""
                    ),
                    cos_download_url_expiry=validate_cos_download_url,
                )
                if not isinstance(minimal_section, str):
                    raise GitHubStateProxyError(
                        "Minimal Evidence renderer must return markdown",
                    )
                minimal_section = remove_attachment_page_download_links(minimal_section)
                if type(compact_round_number) is int and compact_round_number > 0:
                    minimal_section = minimal_section.replace(
                        "<summary>当前轮（当前轮次） ·",
                        f"<summary>当前轮次（第 {compact_round_number} 轮） ·",
                        1,
                    )
                for empty_label in (
                    "本轮暂无附件，上传后会显示",
                    "本轮未保存附件",
                    "该轮未保存附件",
                ):
                    minimal_section = minimal_section.replace(
                        empty_label, "附件和部署明细已精简，请打开附件页查看。",
                    )
                minimal_section = ensure_read_page_entry(minimal_section)
                minimal_end = minimal_section.rfind(evidence_end)
                if minimal_end < 0:
                    raise GitHubStateProxyError(
                        "Minimal Evidence renderer omitted its end marker",
                    )
                minimal_lines = [""]
                if compact_archives:
                    minimal_lines.extend([
                        "",
                        f"可信历史测试轮次：{len(compact_archives)} 个。",
                        "完整轮次编号、HEAD、附件身份和部署记录均保留在附件页。",
                    ])
                all_compatibility_downloads = [
                    line for archive in compact_archives
                    for line in compatibility_downloads_by_round.get(
                        archive["round_id"], [],
                    )
                ]
                all_compatibility_downloads.extend(current_fallback_lines)
                if all_compatibility_downloads:
                    minimal_lines.extend([
                        "", "<details>", "<summary>历史附件兼容下载链接</summary>", "",
                        *all_compatibility_downloads, "", "</details>",
                    ])
                all_machine_log_lines = []
                for archive in compact_archives:
                    archive_round = archive["round_id"]
                    archive_head = archive["head_sha"]
                    label = f"第 {round_numbers[archive_round]} 轮"
                    for line in comments_mod._machine_log_lines(
                            bundle_snapshots_by_round.get(archive_round, {}),
                            (archive_head, archive_round),
                            comments_mod._evidence_path(repo, pr_number), read_origin):
                        all_machine_log_lines.append(line.replace("- `", f"- {label} · `", 1))
                if all_machine_log_lines:
                    minimal_lines.extend([
                        "", "<details>", "<summary>历史机器部署日志</summary>", "",
                        *all_machine_log_lines, "", "</details>",
                    ])
                minimal_section = (
                    minimal_section[:minimal_end].rstrip()
                    + "\n" + "\n".join(minimal_lines)
                    + "\n" + minimal_section[minimal_end:]
                )
                minimal_section = ensure_read_page_entry(minimal_section)
                body = projection_body(minimal_section)
                if _count_visible_bytes(body) > _MAX_VISIBLE_LIFECYCLE_BYTES:
                    raise GitHubStateProxyError(
                        "Evidence projection cannot fit the 48 KiB visible comment budget; "
                        "trusted round data was not omitted",
                    )
        return body

    async def project_evidence(self, repo: str, pr_number: int) -> None:
        """Watcher-only projection. HTTP handlers never call this writer."""
        if self.evidence_service is None:
            return
        pr = await self.proxy.get_pr(repo, pr_number)
        if (not isinstance(pr, dict) or pr.get("state") not in {"open", "closed"}
                or pr.get("draft")
                or pr.get("base", {}).get("repo", {}).get("full_name", "").lower() != repo.lower()):
            return
        comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
        if comment is None:
            return
        body = comment.get("body", "")
        state = _extract_hidden_state(body)
        if state is None:
            raise GitHubStateProxyError("Evidence projection has no trusted state")
        if pr.get("head", {}).get("sha") != state.get("head_sha"):
            return
        from .github_state_proxy import HIDDEN_STATE_MARKER
        visible = body.split(HIDDEN_STATE_MARKER, 1)[0].rstrip()
        await self._write_lifecycle_with_history(
            repo, pr_number, state, visible, evidence_projection_only=True,
        )

    async def _write_lifecycle_with_history(
        self,
        repo: str,
        pr_number: int,
        state: dict,
        new_visible_markdown: str,
        event: dict | None = None,
        events: list[dict] | None = None,
        expected_record_test_state: dict | None = None,
        expected_terminal_state: dict | None = None,
        expected_record_test_comment: dict | None = None,
        expected_acceptance_state: dict | None = None,
        expected_acceptance_comment: dict | None = None,
        evidence_projection_only: bool = False,
        expected_lifecycle_state: dict | None = None,
        write_guard: Any | None = None,
        archive_reason: str = "Lifecycle snapshot replaced",
    ) -> bool:
        """Write lifecycle comment with automatic history management.

        If event is provided, it is appended to the visible History section.
        Legacy v1 migration is handled automatically on first meaningful event.
        Preserves existing History events across lifecycle rewrites.
        """
        archive_source = None
        projection_state_expected = copy.deepcopy(state) if evidence_projection_only else None
        async def write_if_current(visible: str) -> bool:
            async with self._evidence_lock:
                if expected_lifecycle_state is not None:
                    fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                    if fresh_state != expected_lifecycle_state:
                        return False
                if archive_source is not None:
                    # Capture the prior round once under the evidence lock,
                    # after the expected lifecycle state has been checked.
                    state["round_archive"] = await self._archive_round_unlocked(
                        repo, pr_number, archive_source, archive_reason,
                    )
                if expected_lifecycle_state is not None:
                    fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                    if fresh_state != expected_lifecycle_state:
                        return False
                # History/archive discovery also awaits GitHub. Recheck after all
                # of those awaits, immediately before a terminal record_test write.
                if expected_record_test_state is not None:
                    fresh = await self._revalidate_record_test_state(
                        repo, pr_number, expected_record_test_state,
                        "pass" if state.get("status") == "succeeded" else "fail",
                        expected_comment=expected_record_test_comment,
                    )
                    if fresh is None:
                        return False
                if expected_acceptance_state is not None:
                    fresh_pr = await self.proxy.get_pr(repo, pr_number)
                    fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                    expected_comment_id = expected_acceptance_comment.get("id")
                    fresh_command_comment = await self.proxy.get_comment(
                        repo, expected_comment_id,
                    )
                    expected_user = expected_acceptance_comment.get("user", {})
                    fresh_user = (fresh_command_comment.get("user", {})
                                  if isinstance(fresh_command_comment, dict) else {})
                    fresh_base = (fresh_pr.get("base", {})
                                  if isinstance(fresh_pr, dict) else {})
                    fresh_base_repo = (fresh_base.get("repo", {})
                                       if isinstance(fresh_base, dict) else {})
                    fresh_base_name = (fresh_base_repo.get("full_name", "")
                                       if isinstance(fresh_base_repo, dict) else "")
                    from .commands import parse_command
                    fresh_command = parse_command(
                        fresh_command_comment.get("body", "")
                        if isinstance(fresh_command_comment, dict) else ""
                    )
                    fresh_pr_head = (fresh_pr.get("head", {}).get("sha", "")
                                     if isinstance(fresh_pr, dict) else "")
                    api_base = self.config.github_api_url.rstrip("/")
                    if (not isinstance(fresh_pr, dict)
                            or fresh_pr.get("state") != "open"
                            or fresh_pr.get("merged") or fresh_pr.get("draft")
                            or not isinstance(fresh_base_name, str)
                            or fresh_base_name.casefold() != repo.casefold()
                            or fresh_pr_head != expected_acceptance_state.get("head_sha")
                            or fresh_state != expected_acceptance_state
                            or not isinstance(fresh_command_comment, dict)
                            or not isinstance(fresh_user, dict)
                            or fresh_command_comment.get("id") != expected_comment_id
                            or fresh_command_comment.get("url")
                            != f"{api_base}/repos/{repo}/issues/comments/{expected_comment_id}"
                            or fresh_command_comment.get("issue_url")
                            != f"{api_base}/repos/{repo}/issues/{pr_number}"
                            or fresh_command_comment.get("body") != expected_acceptance_comment.get("body")
                            or fresh_command_comment.get("created_at") != expected_acceptance_comment.get("created_at")
                            or fresh_command_comment.get("updated_at") != expected_acceptance_comment.get("updated_at")
                            or fresh_user.get("id") != expected_user.get("id")
                            or fresh_user.get("login") != expected_user.get("login")
                            or fresh_command.kind != "request_acceptance"
                            or self.proxy.is_bot_comment(fresh_command_comment)):
                        return False
                if expected_terminal_state is not None:
                    pr = await self.proxy.get_pr(repo, pr_number)
                    fresh = await self.proxy.read_hidden_state(repo, pr_number)
                    if (fresh != expected_terminal_state
                            or pr.get("head", {}).get("sha") != state.get("head_sha")
                            or pr.get("state") != "open" or pr.get("merged") or pr.get("draft")):
                        return False
                current_pr = await self.proxy.get_pr(repo, pr_number)
                if evidence_projection_only:
                    current_pr_valid = (
                        isinstance(current_pr, dict)
                        and current_pr.get("state") in {"open", "closed"}
                        and not current_pr.get("draft")
                        and current_pr.get("base", {}).get("repo", {}).get("full_name", "").lower()
                        == repo.lower()
                        and current_pr.get("head", {}).get("sha") == state.get("head_sha")
                    )
                else:
                    current_pr_valid = (
                        current_pr.get("state") == "open" and not current_pr.get("merged")
                        and not current_pr.get("draft")
                        and current_pr.get("head", {}).get("sha") == state.get("head_sha")
                    )
                if not current_pr_valid:
                    raise GitHubStateProxyError("Lifecycle PR context changed before write")
                current_comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
                if ((comment is None) != (current_comment is None)
                        or (comment is not None and (
                            current_comment.get("id") != comment.get("id")
                            or current_comment.get("body") != comment.get("body")))):
                    raise GitHubStateProxyError("Lifecycle History changed before write")
                if expected_lifecycle_state is not None or write_guard is not None:
                    # Archiving awaited GitHub. Re-read the lifecycle CAS
                    # inputs and PR identity before replacing the comment.
                    if expected_lifecycle_state is not None:
                        fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
                        if fresh_state != expected_lifecycle_state:
                            return False
                    final_pr = await self.proxy.get_pr(repo, pr_number)
                    final_base = final_pr.get("base", {}) if isinstance(final_pr, dict) else {}
                    final_base_repo = final_base.get("repo", {}) if isinstance(final_base, dict) else {}
                    final_pr_valid = (
                        isinstance(final_pr, dict)
                        and type(final_pr.get("number")) is int
                        and final_pr.get("number") == pr_number
                        and final_pr.get("state") == "open"
                        and not final_pr.get("merged")
                        and not final_pr.get("draft")
                        and final_pr.get("head", {}).get("sha") == state.get("head_sha")
                        and isinstance(final_base_repo, dict)
                        and isinstance(final_base_repo.get("full_name"), str)
                        and final_base_repo["full_name"].casefold() == repo.casefold()
                    )
                    if not final_pr_valid:
                        return False
                    final_comment = await self.proxy.find_trusted_lifecycle_comment(
                        repo, pr_number,
                    )
                    if ((comment is None) != (final_comment is None)
                            or (comment is not None and (
                                final_comment.get("id") != comment.get("id")
                                or final_comment.get("body") != comment.get("body")))):
                        return False
                if write_guard is not None:
                    # Archive creation can take several GitHub calls. Recheck
                    # the complete Review provenance after that wait and just
                    # before the lifecycle CAS so a superseded run cannot win.
                    guard_result = write_guard()
                    if inspect.isawaitable(guard_result):
                        guard_result = await guard_result
                    if guard_result is not True:
                        return False
                if evidence_projection_only:
                    current_state = await self.proxy.read_hidden_state(repo, pr_number)
                    if current_state != projection_state_expected:
                        raise GitHubStateProxyError(
                            "Lifecycle state changed before evidence projection write",
                        )
                    final_pr = await self.proxy.get_pr(repo, pr_number)
                    if (not isinstance(final_pr, dict)
                            or final_pr.get("state") not in {"open", "closed"}
                            or final_pr.get("draft")
                            or final_pr.get("base", {}).get("repo", {}).get("full_name", "").lower()
                            != repo.lower()
                            or final_pr.get("head", {}).get("sha") != state.get("head_sha")):
                        raise GitHubStateProxyError(
                            "Lifecycle PR context changed before evidence projection write",
                        )
                written = await self.proxy.write_hidden_state(
                    repo, pr_number, visible, state,
                    expected_comment=((comment.get("id"), comment.get("body"))
                                      if comment is not None else (None, None)),
                )
                if not isinstance(written, dict):
                    raise GitHubStateProxyError("Lifecycle state write was not confirmed")
                return True

        comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
        if comment is not None:
            old = _extract_hidden_state(comment.get("body", ""))
            if old is None:
                raise GitHubStateProxyError("Cannot replace unreadable prior round")
            old = _validate_hidden_state(old)
            if (expected_lifecycle_state is not None
                    and old != expected_lifecycle_state):
                return False
            if evidence_projection_only:
                if old != state:
                    raise GitHubStateProxyError("Lifecycle state changed before evidence projection")
                projected = await self._render_evidence_projection(
                    repo, pr_number, old, new_visible_markdown,
                )
                if projected == new_visible_markdown:
                    return True
                return await write_if_current(projected)
            old_round = round_id_for_state(old)
            head_changed = old.get("head_sha") != state.get("head_sha")
            if "round_id" not in state and not head_changed:
                state["round_id"] = old_round
            round_changed = state.get("round_id") != old_round
            lost_fields = any(old.get(k) and not state.get(k) for k in
                              ("components", "deployments", "case_results", "review_evidence", "approve_attempts", "test_result"))
            lost_attempt = any(a not in state.get("approve_attempts", []) for a in old.get("approve_attempts", []))
            destructive = (head_changed or round_changed or lost_fields or lost_attempt
                           or (old.get("cos", {}).get("object_key") and any(
                               old["cos"].get(k) != state.get("cos", {}).get(k)
                               for k in ("object_key", "sha256", "size"))))
            if destructive:
                archive_source = copy.deepcopy(old)
                guarded_write = (
                    expected_lifecycle_state is not None or write_guard is not None
                )
                if not guarded_write:
                    # Preserve the original behavior for legacy lifecycle
                    # callers. Only guarded transitions defer archive capture.
                    state["round_archive"] = await self._archive_round(
                        repo, pr_number, old, archive_reason,
                    )
                if head_changed or (lost_fields and not round_changed):
                    state["round_id"] = make_round_id(state["head_sha"], state.get("command", {}).get("comment_id", 0))
                    state["round_started_at"] = utc_now()
                    state.pop("deployment_logs", None)
                    state.pop("acceptance_request", None)
            if "round_started_at" not in state and old.get("round_started_at") and not head_changed and not round_changed:
                state["round_started_at"] = old["round_started_at"]
        if comment is None:
            if evidence_projection_only or expected_lifecycle_state is not None:
                return False
            if events is not None:
                # Multiple events: newest-first; then insert existing fresh markdown
                built = new_visible_markdown
                for evt in reversed(events):
                    built = _insert_history_into_visible(built, evt)
                final_visible = built
            elif event is not None:
                final_visible = _insert_history_into_visible(new_visible_markdown, event)
            else:
                final_visible = new_visible_markdown
            return await write_if_current(final_visible)

        existing_body = comment.get("body", "")
        if not isinstance(existing_body, str):
            if events is not None:
                built = new_visible_markdown
                for evt in reversed(events):
                    built = _insert_history_into_visible(built, evt)
                final_visible = built
            elif event is not None:
                final_visible = _insert_history_into_visible(new_visible_markdown, event)
            else:
                final_visible = new_visible_markdown
            return await write_if_current(final_visible)

        from .github_state_proxy import HIDDEN_STATE_MARKER
        idx = existing_body.find(HIDDEN_STATE_MARKER)
        if idx < 0:
            if events is not None:
                built = new_visible_markdown
                for evt in reversed(events):
                    built = _insert_history_into_visible(built, evt)
                final_visible = built
            elif event is not None:
                final_visible = _insert_history_into_visible(new_visible_markdown, event)
            else:
                final_visible = new_visible_markdown
            return await write_if_current(final_visible)

        existing_visible = existing_body[:idx].rstrip()

        # Check if this is a legacy comment (missing new-format markers)
        # Only migrate a recognizable old Deploy Approval lifecycle. Transient
        # statuses such as "Deploying..." are not lifecycle documents.
        is_legacy = (
            VISIBLE_HISTORY_START_MARKER not in existing_visible
            and "### Workflow" not in existing_visible
            and "<!-- deploy-approval-agent -->" in existing_visible
            and "Deploy Approval" in existing_visible
            and ("**Status:**" in existing_visible or "### Deploy Approval" in existing_visible)
        )

        created_archives = []
        if is_legacy and (event is not None or events is not None):
            # First meaningful event on a legacy comment
            # 1. Preserve legacy visible markdown in an archive
            legacy_snapshot = existing_visible

            existing_comments = await self.proxy.get_issue_comments(repo, pr_number)
            archive = await self._ensure_history_archive(
                repo, pr_number, existing_comments,
                extra_text=(
                    "Legacy lifecycle snapshot preserved before history-enabled renderer migration.\n\n"
                    + legacy_snapshot
                ),
            )
            created_archives.append(archive)

        # Preserve the original event bytes, including fields the legacy parser
        # does not understand. Render only newly appended events.
        h_start = existing_visible.find(VISIBLE_HISTORY_START_MARKER)
        h_end = existing_visible.find(VISIBLE_HISTORY_END_MARKER)
        history_block = ""
        if h_start >= 0 or h_end >= 0:
            if (h_start < 0 or h_end <= h_start
                    or existing_visible.count(VISIBLE_HISTORY_START_MARKER) != 1
                    or existing_visible.count(VISIBLE_HISTORY_END_MARKER) != 1):
                raise GitHubStateProxyError("Cannot safely preserve malformed History markers")
            history_block = existing_visible[h_start:h_end + len(VISIBLE_HISTORY_END_MARKER)]
        elif event is not None or events:
            # Wrap unmarked legacy events when appending new History.
            old_history = re.search(r"(?ms)^### History[ \t]*\n.*?(?=^### |\Z)", existing_visible)
            if old_history:
                history_block = (VISIBLE_HISTORY_START_MARKER
                                 + old_history.group(0).partition("\n")[2]
                                 + VISIBLE_HISTORY_END_MARKER)
        else:
            # Projection-only rewrites also normalize a legacy unmarked History
            # section. Keep its original event text intact inside the wrapper.
            old_history = re.search(r"(?ms)^### History[ \t]*\n.*?(?=^### |\Z)", existing_visible)
            if old_history:
                folded = self._fold_history_content(
                    old_history.group(0).partition("\n")[2],
                )
                if folded:
                    history_block = (
                        VISIBLE_HISTORY_START_MARKER + "\n" + folded + "\n"
                        + VISIBLE_HISTORY_END_MARKER
                    )
        incoming = list(events) if events is not None else ([event] if event is not None else [])
        if incoming:
            new_block = _build_history_block(incoming)
            if history_block:
                def event_content(block):
                    content = block[
                        len(VISIBLE_HISTORY_START_MARKER):
                        -len(VISIBLE_HISTORY_END_MARKER)
                    ].strip("\n")
                    return self._unwrap_history_details(content)

                new_events_body = event_content(new_block).rstrip("\n")
                old_events_body = event_content(history_block).lstrip("\n")
                history_block = (
                    VISIBLE_HISTORY_START_MARKER
                    + "\n<details>\n<summary>History · 历史事件</summary>\n\n"
                    + new_events_body
                    + ("\n\n" + old_events_body if old_events_body else "")
                    + "\n\n</details>\n"
                    + VISIBLE_HISTORY_END_MARKER
                )
            else:
                history_block = new_block
        final_visible = new_visible_markdown.rstrip()
        if history_block:
            final_visible += "\n\n" + history_block + "\n"
        elif not incoming:
            # Retain an unmarked History section only when no marked block was
            # available to relocate above.
            old_history = re.search(r"(?ms)^### History[ \t]*\n.*?(?=^### |\Z)", existing_visible)
            if old_history:
                folded = self._fold_history_content(
                    old_history.group(0).partition("\n")[2],
                )
                if folded:
                    final_visible += (
                        "\n\n" + VISIBLE_HISTORY_START_MARKER + "\n"
                        + folded + "\n" + VISIBLE_HISTORY_END_MARKER
                    )

        if history_block:
            history_start = VISIBLE_HISTORY_START_MARKER
            history_end = VISIBLE_HISTORY_END_MARKER
            if (history_block.count(history_start) != 1
                    or history_block.count(history_end) != 1
                    or history_block.find(history_end) < history_block.find(history_start)):
                raise GitHubStateProxyError("Cannot safely normalize malformed History markers")
            history_body = history_block[
                history_block.find(history_start) + len(history_start):
                history_block.find(history_end)
            ]
            folded_history = self._fold_history_content(history_body)
            history_block = (
                history_start + "\n" + folded_history + "\n" + history_end
            )

        # Keep published archive links even if fresh discovery is unavailable.
        old_archives = re.search(r"(?ms)^### Archived history[ \t]*\n.*?(?=^### |\Z)", existing_visible)
        archives_block = old_archives.group(0).rstrip() if old_archives else ""
        archive_links = list(created_archives)
        try:
            existing_comments = await self.proxy.get_issue_comments(repo, pr_number)
            archive_links += self._discover_trusted_archive_links(
                existing_comments, repo, pr_number,
            )
        except Exception:
            pass
        seen_pages = set(re.findall(r"History Archive Page ([0-9]+)", archives_block))
        for archive in archive_links:
            if str(archive["page"]) in seen_pages:
                continue
            rendered = comments_mod._archives_link_block([archive])
            archives_block += ("\n" + rendered.split("\n", 2)[2]) if archives_block else rendered
            seen_pages.add(str(archive["page"]))
        if archives_block:
            h_start = final_visible.find("### History")
            if h_start >= 0:
                final_visible = final_visible[:h_start] + archives_block.rstrip() + "\n\n" + final_visible[h_start:]
            else:
                final_visible += "\n\n" + archives_block

        # Projection is supplied only from trusted independent records.
        final_visible = await self._render_evidence_projection(repo, pr_number, state, final_visible)
        return await write_if_current(final_visible)

    def _discover_trusted_archive_links(
        self, comments: list[dict], repo: str, pr_number: int,
    ) -> list[dict]:
        """Discover trusted History Archive comments from GitHub.

        Only accepts comments with both BOT_MARKER, the archive marker,
        AND GitHub App provenance via self.proxy.is_bot_comment().
        Never stores archive list in hidden state.
        Returns dicts with page and url for the archive link block.
        """
        from .comments import BOT_MARKER
        links: list[dict] = []
        seen_pages: set[int] = set()
        for c in comments:
            if not isinstance(c, dict):
                continue
            cbody = c.get("body", "")
            if not isinstance(cbody, str):
                continue
            # Must contain bot marker
            if BOT_MARKER not in cbody:
                continue
            # Must be authored by the configured GitHub App
            if not self.proxy.is_bot_comment(c):
                continue
            marker_prefix = f"{HISTORY_ARCHIVE_MARKER_PREFIX}{repo}:{pr_number}:"
            if marker_prefix not in cbody:
                continue
            # Extract page number
            try:
                after_prefix = cbody.split(marker_prefix, 1)[1]
                page_str = after_prefix.split(" -->", 1)[0].strip()
                page_num = int(page_str)
                if page_num <= 0 or page_num in seen_pages:
                    continue
                seen_pages.add(page_num)
            except (ValueError, IndexError):
                continue
            links.append({
                "page": page_num,
                "url": c.get("html_url", ""),
            })
        # Sort by page ascending
        links.sort(key=lambda x: x["page"])
        return links

    async def _supersede_head_drift(
        self, repo: str, pr_number: int, state: dict,
        new_head: str, comment_id: int, command_kind: str = "approve_deploy",
    ) -> None:
        """Handle HEAD drift by superseding the current deployment."""
        pr = await self.proxy.get_pr(repo, pr_number)
        if (pr.get("state") != "open" or pr.get("merged") or pr.get("draft")
                or not re.fullmatch(r"[0-9a-f]{40}", str(new_head))
                or pr.get("head", {}).get("sha") != new_head):
            return
        old_head = state.get("head_sha", "")
        old_status = state.get("status", "review-required")
        archived = await self._archive_round(
            repo, pr_number, copy.deepcopy(state), "HEAD changed",
        )
        pr = await self.proxy.get_pr(repo, pr_number)
        if (pr.get("state") != "open" or pr.get("merged") or pr.get("draft")
                or pr.get("head", {}).get("sha") != new_head):
            return
        state["status"] = "review-required"
        state["head_sha"] = new_head
        state["round_id"] = make_round_id(
            new_head, comment_id,
        )
        state.pop("acceptance_request", None)
        state["round_started_at"] = utc_now()
        state.pop("deployment_logs", None)
        state["review_evidence"] = {}
        state["components"] = []
        state["deployments"] = []
        state["approve_attempts"] = []
        state["approve_attempts_total"] = 0
        state["approve_attempts_truncated"] = False
        state["case_results"] = {}
        state["test_result"] = ""
        state["cos"] = {"object_key": "", "sha256": "", "size": 0}
        state["round_archive"] = archived
        state["command"] = {
            "comment_id": comment_id,
            "kind": command_kind,
            "phase": "completed",
            "args": {"superseded": True},
        }
        state["last_processed_comment_id"] = comment_id

        markdown = comments_mod.superseded_comment(
            repo, pr_number, old_head, new_head,
        )
        event = {
            "event": "HEAD drift detected",
            "lifecycle": f"`{old_status}` \u2192 `review-required`",
            "timestamp": comments_mod.beijing_now_str(),
        }
        await self._write_lifecycle_with_history(
            repo, pr_number, state, markdown, event=event,
        )
        await self.proxy.project_status_label(repo, pr_number, "review-required")
        await self._post_bound_notice(
            repo, pr_number, head_sha=new_head,
            round_id=round_id_for_state(state), comment_id=comment_id,
            topic="pr-head-updated",
            message=(f"The PR HEAD changed from `{_short(old_head)}` to `{_short(new_head)}`. "
                     "The previous review and deployment approval are invalid. Wait for trusted review evidence on the new HEAD."),
        )

    async def _invalidate_review_required(
        self,
        repo: str,
        pr_number: int,
        state: dict,
        head_sha: str,
        comment_id: int,
        reason: str,
    ) -> None:
        """Invalidate the current validation snapshot and return to review-required."""
        pr = await self.proxy.get_pr(repo, pr_number)
        if (pr.get("state") != "open" or pr.get("merged") or pr.get("draft")
                or not re.fullmatch(r"[0-9a-f]{40}", str(head_sha))
                or pr.get("head", {}).get("sha") != head_sha):
            return
        head_changed = state.get("head_sha") != head_sha
        archived = None
        if head_changed:
            archived = await self._archive_round(
                repo, pr_number, copy.deepcopy(state), "HEAD changed",
            )
            pr = await self.proxy.get_pr(repo, pr_number)
            if (pr.get("state") != "open" or pr.get("merged") or pr.get("draft")
                    or not re.fullmatch(r"[0-9a-f]{40}", str(head_sha))
                    or pr.get("head", {}).get("sha") != head_sha):
                return
        state["status"] = "review-required"
        if head_changed:
            state["head_sha"] = head_sha
            state["round_id"] = make_round_id(head_sha, comment_id)
            state["round_started_at"] = utc_now()
            state.pop("deployment_logs", None)
            state["round_archive"] = archived
            state.pop("acceptance_request", None)
        state["review_evidence"] = {}
        state["components"] = []
        state["deployments"] = []
        state["approve_attempts"] = []
        state["approve_attempts_total"] = 0
        state["approve_attempts_truncated"] = False
        state["case_results"] = {}
        state["test_result"] = ""
        state["cos"] = {"object_key": "", "sha256": "", "size": 0}
        state["command"] = {
            "comment_id": comment_id,
            "kind": "approve_deploy",
            "phase": "completed",
            "args": {"reason": reason},
        }
        state["last_processed_comment_id"] = comment_id

        markdown = comments_mod.review_required(repo, pr_number, head_sha)
        event = {
            "event": reason,
            "lifecycle": "`deploy-requested` \u2192 `review-required`",
            "timestamp": comments_mod.beijing_now_str(),
        }
        await self._write_lifecycle_with_history(
            repo, pr_number, state, markdown, event=event,
        )
        await self.proxy.project_status_label(repo, pr_number, "review-required")
        await self._post_bound_notice(
            repo, pr_number, head_sha=head_sha,
            round_id=round_id_for_state(state), comment_id=comment_id,
            topic="predeploy-validation-invalidated",
            message=(f"Pre-deployment validation failed ({reason}). No new deploy POST was started. "
                     "Wait for current HEAD review evidence, then submit a new approval."),
        )

    async def _current_testing_upload_url(
        self, repo: str, pr_number: int, *, head_sha: str, round_id: str,
        approval_comment_id: int,
    ) -> str:
        """Return only the current Summary's verified upload link for this testing round."""
        try:
            pr = await self.proxy.get_pr(repo, pr_number)
            state = await self.proxy.read_hidden_state(repo, pr_number)
            base = pr.get("base", {}) if isinstance(pr, dict) else {}
            base_repo = base.get("repo", {}) if isinstance(base, dict) else {}
            full_name = base_repo.get("full_name") if isinstance(base_repo, dict) else None
            if (not isinstance(pr, dict) or pr.get("state") != "open"
                    or pr.get("merged") or pr.get("draft")
                    or not isinstance(full_name, str)
                    or full_name.casefold() != repo.casefold()
                    or (pr.get("head") or {}).get("sha") != head_sha
                    or not isinstance(state, dict)
                    or state.get("status") != "testing"
                    or state.get("head_sha") != head_sha
                    or round_id_for_state(state) != round_id):
                return ""
            command = state.get("command", {})
            if (not isinstance(command, dict)
                    or command.get("kind") != "approve_deploy"
                    or command.get("comment_id") != approval_comment_id):
                return ""

            try:
                from .evidence_auth import UPLOAD_PURPOSE, readiness, verify_capability
            except Exception:
                return ""
            evidence_status = readiness(self.config, self.evidence_service)
            store = getattr(self.evidence_service, "store", None)
            if (not isinstance(evidence_status, dict)
                    or evidence_status.get("EVIDENCE_BACKEND_READY") is not True
                    or evidence_status.get("PUBLIC_HTTPS_INGRESS_READY") is not True
                    or not callable(getattr(store, "dispatch", None))):
                return ""

            configured_origin = str(
                getattr(self.config, "evidence_public_base_url", "") or "",
            ).rstrip("/")
            if not is_evidence_ingress_url(configured_origin):
                return ""
            configured_parts = urlsplit(configured_origin)

            comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
            body = comment.get("body") if isinstance(comment, dict) else None
            if not isinstance(body, str):
                return ""
            from .github_state_proxy import HIDDEN_STATE_MARKER
            visible = body.split(HIDDEN_STATE_MARKER, 1)[0]
            embedded_state = _extract_hidden_state(body)
            if embedded_state != state:
                return ""
            link_match = re.search(
                r"(?m)^\[deploy-approval-upload\]:[ \t]*<(?P<url>https://[^<>\s]+)>[ \t]*$",
                visible,
            )
            if link_match is None:
                return ""
            candidate = comments_mod._trusted_upload_page_url(
                link_match.group("url"), repo, pr_number,
            )
            if not candidate:
                return ""
            parsed = urlsplit(candidate)
            if (parsed.scheme != "https"
                    or parsed.netloc.casefold() != configured_parts.netloc.casefold()):
                return ""
            query = parse_qs(parsed.query, strict_parsing=True)
            values = query.get("cap")
            if set(query) != {"cap"} or not isinstance(values, list) or len(values) != 1:
                return ""
            claims = verify_capability(
                self.config, values[0], repo, pr_number,
                purpose=UPLOAD_PURPOSE,
            )
            if (claims.get("head") != head_sha
                    or claims.get("round") != round_id
                    or claims.get("status") != "testing"):
                return ""
            return candidate
        except Exception as exc:
            logger.warning("current testing upload link unavailable %s#%s exception=%s",
                           repo, pr_number, type(exc).__name__)
            return ""

    async def _post_bound_notice(
        self, repo: str, pr_number: int, *, head_sha: str, round_id: str,
        comment_id: int, topic: str, message: str,
        expected_command_kind: str = "",
        allow_bot_origin: bool = False,
        allow_missing_state: bool = False,
    ) -> bool:
        """Post one command-bound public reply, confirming uncertain POSTs by readback."""
        if (not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", head_sha)
                or not isinstance(round_id, str)
                or not re.fullmatch(re.escape(head_sha) + r"-[0-9]+", round_id)
                or type(comment_id) is not int or comment_id <= 0):
            return False
        try:
            pr = await self.proxy.get_pr(repo, pr_number)
            state = await self.proxy.read_hidden_state(repo, pr_number)
            origin = await self.proxy.get_comment(repo, comment_id)
            comments = await self.proxy.get_issue_comments(repo, pr_number)
        except Exception as exc:
            logger.warning("notice preflight %s#%s failed: %s",
                           repo, pr_number, type(exc).__name__)
            return False
        api_base = self.config.github_api_url.rstrip("/")
        expected_issue_url = f"{api_base}/repos/{repo}/issues/{pr_number}"
        expected_comment_url = f"{api_base}/repos/{repo}/issues/comments/{comment_id}"
        if (not isinstance(pr, dict) or pr.get("state") != "open" or pr.get("merged")
                or pr.get("draft") or (pr.get("head") or {}).get("sha") != head_sha
                or not isinstance(origin, dict) or origin.get("id") != comment_id
                or origin.get("url") != expected_comment_url
                or origin.get("issue_url") != expected_issue_url
                or not isinstance(origin.get("created_at"), str)
                or origin.get("updated_at") != origin.get("created_at")
                or (self.proxy.is_bot_comment(origin) and not allow_bot_origin)
                or not isinstance(comments, list)):
            return False
        if isinstance(state, dict) and state.get("head_sha") == head_sha:
            if round_id_for_state(state) != round_id:
                return False
        elif (not allow_missing_state or expected_command_kind
              or round_id != make_round_id(head_sha, comment_id)):
            return False
        if expected_command_kind:
            command = state.get("command", {})
            if (not isinstance(command, dict)
                    or command.get("kind") != expected_command_kind
                    or command.get("comment_id") != comment_id):
                return False
        identity = "\0".join((repo, str(pr_number), head_sha, round_id,
                               str(comment_id), topic))
        marker = "<!-- deploy-approval-notice:v1:" + hashlib.sha256(
            identity.encode("utf-8"),
        ).hexdigest() + " -->"

        def found_marker(items):
            return next((item for item in items
                         if isinstance(item, dict)
                         and isinstance(item.get("body"), str)
                         and marker in item["body"]
                         and self.proxy.is_bot_comment(item)), None)

        if found_marker(comments):
            return True
        payload = message.strip()
        if payload.startswith(comments_mod.BOT_MARKER):
            payload = payload[len(comments_mod.BOT_MARKER):].lstrip()
        body = "\n".join((comments_mod.BOT_MARKER, marker, "", payload))
        try:
            posted = await self.proxy.post_issue_comment(repo, pr_number, body)
        except Exception as post_error:
            # A lost response may still have created the comment. Read before
            # returning so a retry can never blindly duplicate the notice.
            try:
                latest = await self.proxy.get_issue_comments(repo, pr_number)
            except Exception as read_error:
                logger.warning("notice POST outcome unknown %s#%s topic=%s",
                               repo, pr_number, topic)
                return False
            return bool(isinstance(latest, list) and found_marker(latest))
        if isinstance(posted, dict) and posted.get("id"):
            body_value = posted.get("body", "")
            if marker in body_value and self.proxy.is_bot_comment(posted):
                return True
        try:
            latest = await self.proxy.get_issue_comments(repo, pr_number)
        except Exception:
            return False
        return bool(isinstance(latest, list) and found_marker(latest))

    async def _post_deploy_ready_notice(
        self, repo: str, pr_number: int, *, head_sha: str, round_id: str,
        review_evidence: dict, components: list[dict],
    ) -> bool:
        """Notify once for one freshly verified Review Agent result."""
        build_comment_id = review_evidence.get("build_comment_id")
        if type(build_comment_id) is not int or build_comment_id <= 0:
            return False
        event_fingerprint = hashlib.sha256(
            repr(sorted(self._review_evidence_identity(review_evidence).items())).encode(),
        ).hexdigest()
        notice_identity = "\0".join((
            repo, str(pr_number), head_sha, round_id, "deploy-ready",
            str(build_comment_id), event_fingerprint,
        ))
        marker = "<!-- deploy-approval-notice:v1:" + hashlib.sha256(
            notice_identity.encode("utf-8"),
        ).hexdigest() + " -->"

        async def context_is_current() -> bool:
            try:
                pr = await self.proxy.get_pr(repo, pr_number)
                state = await self.proxy.read_hidden_state(repo, pr_number)
            except Exception:
                return False
            if not isinstance(pr, dict):
                return False
            base = pr.get("base", {}) if isinstance(pr, dict) else {}
            base_repo = base.get("repo", {}) if isinstance(base, dict) else {}
            head = pr.get("head")
            full_name = base_repo.get("full_name") if isinstance(base_repo, dict) else None
            if (pr.get("state") != "open"
                    or pr.get("merged") or pr.get("draft")
                    or pr.get("number") != pr_number
                    or not isinstance(head, dict) or head.get("sha") != head_sha
                    or not isinstance(full_name, str)
                    or full_name.casefold() != repo.casefold()
                    or not isinstance(state, dict)
                    or state.get("status") != "deploy-ready"
                    or state.get("head_sha") != head_sha
                    or round_id_for_state(state) != round_id
                    or not isinstance(state.get("review_evidence"), dict)
                    or self._review_evidence_identity(state.get("review_evidence", {}))
                    != self._review_evidence_identity(review_evidence)
                    or self._canonical_component_snapshot(state.get("components", []))
                    != self._canonical_component_snapshot(components)):
                return False
            return await self._fresh_review_evidence_matches_state(
                repo, pr_number, head_sha, review_evidence,
                expected_components=components,
            )

        def marker_exists(comments) -> bool:
            if not isinstance(comments, list):
                return False
            return any(
                isinstance(comment, dict)
                and isinstance(comment.get("body"), str)
                and marker in comment["body"]
                and self.proxy.is_bot_comment(comment)
                for comment in comments
            )

        if not await context_is_current():
            return False
        try:
            comments = await self.proxy.get_issue_comments(repo, pr_number)
        except Exception:
            return False
        if not isinstance(comments, list):
            return False
        if marker_exists(comments):
            return True
        # The trusted review may have changed while the marker read waited.
        if not await context_is_current():
            return False

        groups = self._get_machine_groups_for_components(components, repo)
        commands = self._machine_notice_lines(
            groups, repo, mention_owners=False,
        )
        required_ids = {
            component.get("component_id") for component in components
            if isinstance(component, dict) and isinstance(component.get("component_id"), str)
        }
        full_groups = [
            group for group in groups
            if required_ids and required_ids.issubset(set(group.get("component_ids", [])))
        ]
        message_lines = [
            f"当前 Commit `{_short(head_sha)}` 的构建和审核已完成，Lifecycle 已进入 `deploy-ready`。",
        ]
        if commands:
            message_lines.extend([
                "现在可以审批部署。可用机器及完整 IP 审批命令：",
                *commands,
            ])
            if len(full_groups) == 1:
                message_lines.extend([
                    "只有一台机器完整适配全部组件，也可以直接执行 `/approve_deploy`。",
                ])
            elif len(full_groups) > 1:
                message_lines.append("有多台机器完整适配，请从上面的 IP 命令中明确选择一台。")
            elif required_ids:
                message_lines.append("没有单台机器覆盖全部组件；可按上面的命令分批审批适配机器。")
        else:
            message_lines.append(
                "审核和构建已完成，但当前没有适配且可访问的机器，暂时无法部署。请检查当前仓库的机器配置和 Agent Core 访问。"
            )
        message_lines.append("无需先执行 `/request_deploy`；该命令仍可用于显式申请部署。")
        body = "\n".join((comments_mod.BOT_MARKER, marker, "", *message_lines))
        if not await context_is_current():
            return False
        try:
            posted = await self.proxy.post_issue_comment(repo, pr_number, body)
        except Exception:
            try:
                latest = await self.proxy.get_issue_comments(repo, pr_number)
            except Exception:
                return False
            return bool(isinstance(latest, list) and marker_exists(latest))
        if (isinstance(posted, dict) and marker in posted.get("body", "")
                and self.proxy.is_bot_comment(posted)):
            return True
        try:
            latest = await self.proxy.get_issue_comments(repo, pr_number)
        except Exception:
            return False
        return bool(isinstance(latest, list) and marker_exists(latest))

    async def _post_command_reply(
        self, repo: str, pr_number: int, comment_id: int, message: str,
        *, topic: str = "command-error",
    ) -> None:
        """Bind an operation reply to the current PR HEAD, round, and source comment."""
        try:
            pr = await self.proxy.get_pr(repo, pr_number)
            state = await self.proxy.read_hidden_state(repo, pr_number)
        except Exception as exc:
            logger.warning("command reply context %s#%s unavailable: %s",
                           repo, pr_number, type(exc).__name__)
            return
        head = ((pr.get("head") or {}).get("sha")
                if isinstance(pr, dict) else "")
        if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
            return
        has_current_head_state = (
            isinstance(state, dict) and state.get("head_sha") == head
        )
        if has_current_head_state:
            round_id = round_id_for_state(state)
        else:
            # No persisted lifecycle round exists for the current HEAD. Bind
            # this rejected command to a deterministic event-scoped identity.
            round_id = make_round_id(head, comment_id)
        await self._post_bound_notice(
            repo, pr_number, head_sha=head, round_id=round_id,
            comment_id=comment_id, topic=topic, message=message,
            allow_missing_state=not has_current_head_state,
        )

    @staticmethod
    def _is_test_only_pr(pr: dict | None) -> bool:
        title = pr.get("title", "") if isinstance(pr, dict) else ""
        return isinstance(title, str) and "[TEST ONLY]" in title.upper()

    def _eligible_acceptance_reviewers(
        self, repo: str, pr: dict | None,
    ) -> tuple[bool, list[str]]:
        try:
            configured_reviewers = self.policy.reviewers_for_repo(repo)
        except Exception as exc:
            logger.warning("acceptance reviewer policy unavailable %s exception=%s",
                           repo, type(exc).__name__)
            return False, []
        valid_reviewers = []
        seen = set()
        for reviewer in configured_reviewers if isinstance(configured_reviewers, list) else []:
            if (not isinstance(reviewer, str)
                    or not _LOGIN_RE.fullmatch(reviewer.strip())
                    or reviewer.strip().casefold().endswith("[bot]")):
                continue
            key = reviewer.strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            valid_reviewers.append(reviewer.strip())
        eligible = valid_reviewers
        if self._is_test_only_pr(pr):
            # Test PRs suppress real reviewer mentions. Mention Haohao-end only
            # when this repository's verified policy explicitly allows it.
            eligible = ["Haohao-end" for name in eligible
                        if name.casefold() == "haohao-end"]
        return bool(valid_reviewers), eligible

    def _acceptance_reviewer_guidance(
        self, repo: str, pr: dict | None, reviewers_configured: bool,
        eligible_reviewers: list[str],
    ) -> str:
        """Describe repo-scoped acceptance authorization without TEST ONLY leaks."""
        if self._is_test_only_pr(pr):
            if any(reviewer.casefold() == "haohao-end" for reviewer in eligible_reviewers):
                return "测试验收账号 @Haohao-end 已列入当前仓库验收白名单。"
            if reviewers_configured:
                return (
                    f"当前仓库 `{repo}` 已配置有效验收白名单。为避免打扰，[TEST ONLY] PR 不 @真实验收人；"
                    "Haohao-end 不在该仓库白名单中，未获授权执行 `/record_test`。"
                )
            return (
                f"当前仓库 `{repo}` 没有可确认的有效验收白名单。Haohao-end 也未获授权，"
                "请检查该仓库 main 分支的 `test_reviewers.yaml`。"
            )
        if eligible_reviewers:
            return "当前仓库有效验收人：" + " ".join(
                f"@{reviewer}" for reviewer in eligible_reviewers
            )
        if not reviewers_configured:
            return (
                f"当前仓库 `{repo}` 没有可确认的有效验收白名单，请检查该仓库 main 分支的 `test_reviewers.yaml`。"
            )
        return f"当前仓库 `{repo}` 的验收白名单中没有可用的 GitHub 账号。"

    def _machine_notice_lines(
        self, groups: list[dict], repo: str, *, mention_owners: bool,
        test_only: bool = False,
    ) -> list[str]:
        lines: list[str] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            ip = str(group.get("ip", "") or "").strip()
            try:
                parsed_ip = ipaddress.ip_address(ip)
            except ValueError:
                continue
            if parsed_ip.version != 4:
                continue
            lines.append(f"`/approve_deploy machine={parsed_ip}`")
            if mention_owners and not test_only:
                machine = self.policy.get_machine(str(group.get("alias", "")), repo=repo)
                owners = getattr(machine, "owners", []) if machine is not None else []
                valid_owners = sorted({
                    owner.strip() for owner in owners
                    if isinstance(owner, str) and _LOGIN_RE.fullmatch(owner.strip())
                    and not owner.strip().casefold().endswith("[bot]")
                }, key=str.casefold)
                if valid_owners:
                    lines.append("对应机器负责人：" + " ".join(f"@{owner}" for owner in valid_owners))
        return lines

    async def _approval_command_guidance(self, repo: str, pr_number: int) -> str:
        """Build executable approval guidance from the current trusted lifecycle."""
        try:
            pr = await self.proxy.get_pr(repo, pr_number)
            if (not isinstance(pr, dict) or pr.get("state") != "open"
                    or pr.get("merged") or pr.get("draft")):
                return "Deployment approval requires an open, unmerged, non-draft PR."
            head = pr.get("head", {}).get("sha", "")
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if (not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head)
                    or not isinstance(state, dict) or state.get("head_sha") != head):
                return "The PR HEAD or lifecycle changed; wait for the current Lifecycle Summary to update before retrying."
            status = state.get("status", "")
            if status not in {"deploy-ready", "deploy-requested"}:
                return f"Current lifecycle status is `{status or 'unknown'}`; wait until deployment approval is available."
            review_evidence = state.get("review_evidence")
            components = state.get("components")
            if (not isinstance(review_evidence, dict) or not isinstance(components, list)
                    or not await self._fresh_review_evidence_matches_state(
                        repo, pr_number, head, review_evidence,
                        expected_components=components,
                    )):
                return "Current HEAD does not have fresh trusted Review Agent evidence; wait for lifecycle reconciliation."
            deployed_ids = _component_id_set(state.get("deployments", []))
            remaining = [
                component for component in components
                if isinstance(component, dict)
                and component.get("component_id") not in deployed_ids
            ]
            groups = self._get_machine_groups_for_components(remaining, repo)
            commands = self._machine_notice_lines(
                groups, repo, mention_owners=False,
                test_only=self._is_test_only_pr(pr),
            )
            if not commands:
                return "No trusted compatible machine with a unique IPv4 address is available; check this repository's machine policy and Agent Core access."
            lines = ["Current executable machine commands:", *commands]
            required_ids = {
                component.get("component_id") for component in remaining
                if isinstance(component, dict) and isinstance(component.get("component_id"), str)
            }
            full_groups = [
                group for group in groups
                if required_ids and required_ids.issubset(set(group.get("component_ids", [])))
            ]
            if status == "deploy-ready" and len(full_groups) == 1:
                lines.append("Exactly one fully compatible machine is available; `/approve_deploy` also works without a machine argument.")
            elif len(full_groups) > 1:
                lines.append("Several fully compatible machines are available; select one of the IP commands above.")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("approval guidance unavailable %s#%s exception=%s",
                           repo, pr_number, type(exc).__name__)
            return "The current machine list could not be verified. Check the Lifecycle Summary before retrying."

    async def _project_status_best_effort(
        self, repo: str, pr_number: int, status: str,
    ) -> None:
        try:
            await self.proxy.project_status_label(repo, pr_number, status)
        except Exception as exc:
            logger.warning("status label projection %s#%s status=%s exception=%s",
                           repo, pr_number, status, type(exc).__name__)

    async def _block_machine_policy_change(
        self, repo: str, pr_number: int, state: dict, expected_state: dict,
        comment_id: int, machine_alias: str, actor: str, reason: str,
    ) -> bool:
        """Persist a no-POST result when machine policy changes mid-approval."""
        try:
            pr = await self.proxy.get_pr(repo, pr_number)
        except Exception:
            return False
        if (not isinstance(pr, dict) or pr.get("state") != "open"
                or pr.get("merged") or pr.get("draft")
                or (pr.get("head") or {}).get("sha") != state.get("head_sha")):
            return False

        blocked_state = copy.deepcopy(state)
        blocked_state["status"] = "deploy-requested"
        blocked_state["command"] = {
            "comment_id": comment_id,
            "kind": "approve_deploy",
            "phase": "completed",
            "args": {"machine": machine_alias, "actor": actor},
        }
        blocked_state["last_processed_comment_id"] = max(
            int(blocked_state.get("last_processed_comment_id", 0) or 0),
            comment_id,
        )
        deployed_ids = _component_id_set(blocked_state.get("deployments", []))
        remaining = [
            component for component in blocked_state.get("components", [])
            if isinstance(component, dict)
            and component.get("component_id") not in deployed_ids
        ]
        try:
            machine_groups = self._get_machine_groups_for_components(remaining, repo)
        except Exception as exc:
            logger.warning("machine policy changed during approval %s#%s exception=%s",
                           repo, pr_number, type(exc).__name__)
            machine_groups = []
        markdown = comments_mod.deploy_requested(
            repo, pr_number, blocked_state.get("head_sha", ""),
            blocked_state.get("components", []), machine_groups,
            gate_note=[reason, "No deploy POST was sent for the pending component.",
                       "Review the current machine list and submit a NEW approval command."],
            deployments=self._enrich_deployments_for_render(
                blocked_state.get("deployments", []), repo,
            ),
        )
        written = await self._write_lifecycle_with_history(
            repo, pr_number, blocked_state, markdown,
            event={"event": "Machine policy changed before deployment",
                   "machine": machine_alias,
                   "timestamp": comments_mod.beijing_now_str()},
            expected_lifecycle_state=expected_state,
        )
        if not written:
            return False
        await self._project_status_best_effort(repo, pr_number, "deploy-requested")
        command_lines = self._machine_notice_lines(
            machine_groups, repo, mention_owners=False,
            test_only=self._is_test_only_pr(pr),
        )
        message_lines = [reason, "No deploy POST was sent for the pending component."]
        if command_lines:
            message_lines.extend(["Current compatible commands:", *command_lines])
        else:
            message_lines.append("No compatible deployable machine is currently available.")
        await self._post_bound_notice(
            repo, pr_number, head_sha=blocked_state["head_sha"],
            round_id=round_id_for_state(blocked_state), comment_id=comment_id,
            topic="machine-policy-changed-before-deploy",
            expected_command_kind="approve_deploy",
            message="\n".join(message_lines),
        )
        return True

    async def _post_command_not_ready(
        self, repo: str, pr_number: int, current_status: str, next_action_text: str,
        *, comment_id: int = 0,
    ) -> None:
        """Post a non-error 'command not ready' reply comment."""
        body = "\n".join([
            comments_mod.BOT_MARKER,
            "### Deploy Approval — Command not ready",
            "",
            f"Current lifecycle: `{current_status}`",
            "",
            "### Next action",
            "",
            next_action_text,
        ])
        if comment_id:
            await self._post_command_reply(
                repo, pr_number, comment_id, body, topic="command-not-ready",
            )
        else:
            try:
                await self.proxy.post_issue_comment(repo, pr_number, body)
            except GitHubError as e:
                logger.warning("post command not ready comment %s#%s: %s", repo, pr_number, e)

    async def _post_error(
        self, repo: str, pr_number: int, message: str, *, comment_id: int = 0,
    ) -> None:
        """Post an error reply comment."""
        body = "\n".join([
            comments_mod.BOT_MARKER,
            "### Deploy Approval \u2014 Error",
            "",
            message,
        ])
        if comment_id:
            topic = "command-error-" + hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
            await self._post_command_reply(
                repo, pr_number, comment_id, body, topic=topic,
            )
        else:
            try:
                await self.proxy.post_issue_comment(repo, pr_number, body)
            except GitHubError as e:
                logger.warning("post error comment %s#%s: %s", repo, pr_number, e)

    # ── Reconcile ──

    async def reconcile_pr(self, repo: str, pr_number: int) -> None:
        """Reconcile lifecycle state for a PR.

        Called by the watcher before processing commands each cycle.
        Handles uncertain command state and status label projection.
        """
        state = await self.proxy.read_hidden_state(repo, pr_number)
        pr_data = await self.proxy.get_pr(repo, pr_number)
        if inspect.isawaitable(pr_data):
            pr_data = await pr_data
        if not isinstance(pr_data, dict):
            return
        pr_state = pr_data.get("state", "")
        pr_merged = pr_data.get("merged", False)
        current_head = str(pr_data.get("head", {}).get("sha", "") or "")

        if pr_state == "open" and pr_data.get("draft") is True:
            # Draft PRs need an explicit status label, but they are not
            # lifecycle evidence. Never initialize or advance review status
            # from comments until GitHub marks the PR ready for review.
            label_status = "review-required"
            if (isinstance(state, dict)
                    and re.fullmatch(r"[0-9a-f]{40}", current_head)
                    and state.get("head_sha") == current_head):
                label_status = state.get("status", "review-required")
                acceptance = state.get("acceptance_request")
                if (label_status == "testing"
                        and isinstance(acceptance, dict)
                        and acceptance.get("status") == "pending"
                        and acceptance.get("head_sha") == current_head
                        and acceptance.get("round_id") == round_id_for_state(state)):
                    label_status = "acceptance-required"
            if label_status in {
                "review-required", "reviewing", "deploy-ready",
                "deploy-requested", "deploying", "testing", "acceptance-required",
                "succeeded", "failed",
            }:
                try:
                    await self.proxy.project_status_label(
                        repo, pr_number, label_status,
                    )
                except Exception as exc:
                    logger.warning(
                        "reconcile draft PR label projection %s#%s exception=%s",
                        repo, pr_number, type(exc).__name__,
                    )
            return

        if pr_state != "open" or pr_merged or pr_data.get("draft"):
            # Label repair is a presentation-only projection of trusted state.
            # It never rewrites Lifecycle or replays commands for a closed PR.
            if (isinstance(state, dict)
                    and re.fullmatch(r"[0-9a-f]{40}", current_head)
                    and state.get("head_sha") == current_head):
                try:
                    label_status = state.get("status", "")
                    acceptance = state.get("acceptance_request")
                    if (label_status == "testing"
                            and isinstance(acceptance, dict)
                            and acceptance.get("status") == "pending"
                            and acceptance.get("head_sha") == current_head
                            and acceptance.get("round_id") == round_id_for_state(state)):
                        label_status = "acceptance-required"
                    if label_status in {
                        "review-required", "reviewing", "deploy-ready",
                        "deploy-requested", "deploying", "testing", "acceptance-required",
                        "succeeded", "failed",
                    }:
                        await self.proxy.project_status_label(
                            repo, pr_number, label_status,
                        )
                except Exception as exc:
                    logger.warning(
                        "reconcile closed PR label projection %s#%s exception=%s",
                        repo, pr_number, type(exc).__name__,
                    )
            return
        if not re.fullmatch(r"[0-9a-f]{40}", current_head):
            return

        if state is not None:
            pending_logs = state.get("deployment_logs", {})
            if isinstance(pending_logs, dict):
                for machine_alias, log_task in list(pending_logs.items()):
                    if (isinstance(log_task, dict) and log_task.get("status") == "pending"
                            and log_task.get("head_sha") == state.get("head_sha")
                            and log_task.get("round_id") == state.get("round_id")):
                        await self._save_deployed_machine_log(repo, pr_number, state, machine_alias)
                        refreshed = await self.proxy.read_hidden_state(repo, pr_number)
                        if not isinstance(refreshed, dict) or refreshed.get("head_sha") != current_head:
                            return
                        state = refreshed
            cmd = state.get("command", {})
            cmd_phase = cmd.get("phase", "")
            same_head = current_head == state.get("head_sha", "")
            if (cmd_phase == "executing" and same_head
                    and state.get("status") in {"deploying", "deploy-requested"}):
                await self._handle_uncertain_if_needed(repo, pr_number, state)
                return
            if (cmd_phase == "uncertain" and same_head
                    and state.get("status") in {"deploying", "deploy-requested"}):
                await self._recover_uncertain_deployment_tick(
                    repo, pr_number, state, pr_data,
                )
                return

            if current_head and current_head != state.get("head_sha", ""):
                old_status = state.get("status", "review-required")
                archived = await self._archive_round(
                    repo, pr_number, copy.deepcopy(state), "HEAD changed",
                )
                self._reset_review_lifecycle_state(
                    state,
                    head_sha=current_head,
                    status="review-required",
                    review_evidence={},
                )
                state["round_archive"] = archived
                state["command"] = {
                    "comment_id": int(state.get("last_processed_comment_id", 0) or 0),
                    "kind": "",
                    "phase": "completed",
                    "args": {},
                }
                try:
                    markdown = comments_mod.review_required(repo, pr_number, current_head)
                    event = {
                        "event": "HEAD drift detected",
                        "lifecycle": f"`{old_status}` \u2192 `review-required`",
                        "timestamp": comments_mod.beijing_now_str(),
                    }
                    await self._write_lifecycle_with_history(
                        repo, pr_number, state, markdown, event=event,
                    )
                    await self.proxy.project_status_label(repo, pr_number, "review-required")
                except Exception as e:
                    logger.warning(
                        "reconcile head drift %s#%s: %s", repo, pr_number, e,
                    )
                return

            if state.get("status") in ("succeeded", "failed"):
                try:
                    await self.proxy.project_status_label(repo, pr_number, state["status"])
                except Exception as exc:
                    logger.warning("reconcile label projection %s#%s exception=%s",
                                   repo, pr_number, type(exc).__name__)
                await self._retry_terminal_cos(repo, pr_number, state)
                return

            if state.get("status") == "deploy-requested":
                try:
                    state = await self._recover_deploy_requested_review(
                        repo, pr_number, state, current_head,
                    )
                except Exception as e:
                    # A read or identity failure leaves the frozen request
                    # untouched. The approval path independently keeps its
                    # fresh-evidence gate closed.
                    logger.warning(
                        "reconcile review recovery %s#%s: %s",
                        repo, pr_number, type(e).__name__,
                    )
                try:
                    await self.proxy.project_status_label(
                        repo, pr_number, state.get("status", "review-required"),
                    )
                except Exception as e:
                    logger.warning(
                        "reconcile label projection %s#%s: %s", repo, pr_number, e,
                    )
                return

            # Deployment and manual testing are decoupled. Preserve any legacy
            # case_results as history, but never launch automated Cases here.
            if state.get("status") == "testing":
                try:
                    acceptance = state.get("acceptance_request")
                    pending_acceptance = (
                        isinstance(acceptance, dict)
                        and acceptance.get("status") == "pending"
                        and acceptance.get("head_sha") == state.get("head_sha")
                        and acceptance.get("round_id") == round_id_for_state(state)
                    )
                    await self.proxy.project_status_label(
                        repo, pr_number,
                        "acceptance-required" if pending_acceptance else "testing",
                    )
                except Exception as e:
                    logger.warning(
                        "reconcile label projection %s#%s: %s", repo, pr_number, e,
                    )
                return

            if state.get("status") == "deploying":
                try:
                    await self.proxy.project_status_label(repo, pr_number, "deploying")
                except Exception as exc:
                    logger.warning("reconcile deploying label %s#%s exception=%s",
                                   repo, pr_number, type(exc).__name__)
                return

        if pr_state != "open" or pr_merged:
            if state is not None:
                try:
                    await self.proxy.project_status_label(
                        repo, pr_number, state.get("status", "review-required"),
                    )
                except Exception as e:
                    logger.warning(
                        "reconcile label projection %s#%s: %s", repo, pr_number, e,
                    )
            return

        if not current_head:
            return

        # Fetch PR comments and extract review evidence from GitHub comments
        components: list[dict] = []
        try:
            comments = await self.github.get_issue_comments(repo, pr_number)
            if not isinstance(comments, list):
                raise GitHubStateProxyError(
                    "GitHub PR comments response was not a list",
                )

            from .review_comment_parser import extract_latest_review_job_anchor

            # Step 1: Get canonical latest anchor FIRST
            latest_anchor = extract_latest_review_job_anchor(
                comments,
                self.config.review_comment_author_id,
                self.config.review_comment_author_login,
            )

            if latest_anchor is None:
                desired_status = "review-required"
                review_evidence_data: dict = {}
                build_infos: list[BuildInfo] = []
            else:
                resolved_head = await self._resolve_commit_prefix_for_head(
                    repo, current_head, latest_anchor.commit_prefix,
                )
                if resolved_head is None:
                    # Cannot resolve anchor commit prefix to fresh HEAD
                    desired_status = "review-required"
                    review_evidence_data = {}
                    build_infos = []
                elif latest_anchor.state == "terminal":
                    # Build failed / malformed — must NOT fall back
                    desired_status = "review-required"
                    review_evidence_data = {}
                    build_infos = []
                elif latest_anchor.state == "reviewing":
                    # Queued / Building / incomplete — project reviewing, NEVER fall back
                    desired_status = "reviewing"
                    review_evidence_data = {}
                    build_infos = []
                elif latest_anchor.state == "build-succeeded":
                    # Build completed successfully — get full evidence for deploy-ready check
                    evidence = await self._review_evidence_from_comments(repo, pr_number, current_head, comments)
                    if evidence is None:
                        # Build succeeded but Test/Code Review not yet complete
                        desired_status = "reviewing"
                        review_evidence_data = {}
                        build_infos = []
                    else:
                        resolved_head_ev = await self._resolve_review_evidence_for_head(
                            repo, current_head, evidence, pr_number=pr_number,
                        )
                        build_infos = self._review_build_infos(repo, evidence)
                        components = await self._build_component_snapshot(
                            repo, pr_number, current_head, build_infos,
                        ) if resolved_head_ev == current_head else None
                        if resolved_head_ev != current_head or not components:
                            desired_status = "review-required"
                            review_evidence_data = {}
                            build_infos = []
                        else:
                            desired_status = "deploy-ready"
                            review_evidence_data = self._review_evidence_snapshot(evidence, resolved_head_ev)
                else:
                    desired_status = "review-required"
                    review_evidence_data = {}
                    build_infos = []
        except Exception as e:
            logger.warning(
                "reconcile comment evidence %s#%s head=%s: %s",
                repo, pr_number, current_head, e,
            )
            # An incomplete/transient comment read is not evidence that prior
            # review approval disappeared. Never initialize from an unknown
            # response, nor clear a trusted current-HEAD deployment or result.
            # A label repair may still reflect the verified hidden-state value.
            if (isinstance(state, dict)
                    and state.get("head_sha") == current_head):
                try:
                    label_status = state.get("status", "")
                    acceptance = state.get("acceptance_request")
                    if (label_status == "testing"
                            and isinstance(acceptance, dict)
                            and acceptance.get("status") == "pending"
                            and acceptance.get("head_sha") == current_head
                            and acceptance.get("round_id") == round_id_for_state(state)):
                        label_status = "acceptance-required"
                    if label_status in {
                        "review-required", "reviewing", "deploy-ready",
                        "deploy-requested", "deploying", "testing", "acceptance-required",
                        "succeeded", "failed",
                    }:
                        await self.proxy.project_status_label(
                            repo, pr_number, label_status,
                        )
                except Exception as label_exc:
                    logger.warning(
                        "reconcile fallback label projection %s#%s exception=%s",
                        repo, pr_number, type(label_exc).__name__,
                    )
            return

        lifecycle_before_update = copy.deepcopy(state) if isinstance(state, dict) else None
        state_was_none = state is None
        if state is None:
            state = self._init_hidden_state(
                head_sha=current_head,
                status=desired_status,
                review_evidence=review_evidence_data,
                components=components if desired_status == "deploy-ready" else [],
                deployments=[],
            )
            state["command"] = {
                "comment_id": 0,
                "kind": "",
                "phase": "completed",
                "args": {},
            }
            state["last_processed_comment_id"] = 0
        else:
            prev_status = state.get("status", "review-required")
            prev_head = state.get("head_sha", "")
            prev_evidence = state.get("review_evidence", {})
            state["head_sha"] = current_head
            state["status"] = desired_status
            state["review_evidence"] = review_evidence_data
            state["components"] = components if desired_status == "deploy-ready" else []
            state["deployments"] = []
            state["approve_attempts"] = []
            state["approve_attempts_total"] = 0
            state["approve_attempts_truncated"] = False
            state["case_results"] = {}
            state["test_result"] = ""
            state["cos"] = {"object_key": "", "sha256": "", "size": 0}
            state["command"] = {
                "comment_id": int(state.get("last_processed_comment_id", 0) or 0),
                "kind": "",
                "phase": "completed",
                "args": {},
            }

        if desired_status == "deploy-ready":
            build_comment_id = review_evidence_data.get("build_comment_id")
            if type(build_comment_id) is int and build_comment_id > 0:
                state["round_id"] = make_round_id(current_head, build_comment_id)
                state["round_started_at"] = state.get("round_started_at") or utc_now()

        # No-churn: only write lifecycle/history if there is a real change
        # (prev_status/prev_head/prev_evidence captured BEFORE state mutation above)
        if state_was_none:
            meaningful_change = True
        else:
            meaningful_change = (
                prev_status != desired_status
                or prev_head != current_head
                or prev_evidence != review_evidence_data
            )

        if not meaningful_change:
            # Same state, no meaningful change — only project status label
            try:
                await self.proxy.project_status_label(repo, pr_number, desired_status)
            except Exception as e:
                logger.warning(
                    "reconcile label projection %s#%s: %s", repo, pr_number, e,
                )
            if desired_status == "deploy-ready":
                await self._post_deploy_ready_notice(
                    repo, pr_number, head_sha=current_head,
                    round_id=round_id_for_state(state),
                    review_evidence=state.get("review_evidence", {}),
                    components=state.get("components", []),
                )
        else:
            try:
                if desired_status == "reviewing":
                    markdown = comments_mod.reviewing(repo, pr_number, current_head)
                elif desired_status == "review-required":
                    markdown = comments_mod.review_required(repo, pr_number, current_head)
                else:
                    markdown = comments_mod.deploy_ready(
                        repo, pr_number, current_head, build_infos,
                        review_evidence=review_evidence_data,
                        machine_groups=self._get_machine_groups_for_components(components, repo),
                        components=components,
                    )
                if state_was_none:
                    event = {
                        "event": "Lifecycle initialized",
                        "lifecycle": f"`none` \u2192 `{desired_status}`",
                        "timestamp": comments_mod.beijing_now_str(),
                    }
                else:
                    event = {
                        "event": "Review lifecycle transitioned",
                        "lifecycle": f"`{prev_status}` \u2192 `{desired_status}`",
                        "timestamp": comments_mod.beijing_now_str(),
                    }
                async def guard_deploy_ready_review() -> bool:
                    return await self._fresh_review_evidence_matches_state(
                        repo, pr_number, current_head, review_evidence_data,
                        expected_components=components,
                    )

                written = await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                    expected_lifecycle_state=lifecycle_before_update,
                    write_guard=(guard_deploy_ready_review
                                 if desired_status == "deploy-ready" else None),
                )
                if not written:
                    return
                await self._project_status_best_effort(
                    repo, pr_number, desired_status,
                )
                if desired_status == "deploy-ready":
                    await self._post_deploy_ready_notice(
                        repo, pr_number, head_sha=current_head,
                        round_id=round_id_for_state(state),
                        review_evidence=review_evidence_data,
                        components=components,
                    )
            except Exception as e:
                logger.warning(
                    "reconcile lifecycle %s#%s: %s", repo, pr_number, e,
                )

    async def _handle_uncertain_if_needed(
        self, repo: str, pr_number: int, state: dict,
    ) -> None:
        """On startup, if hidden command.phase is executing, mark as uncertain."""
        original_state = copy.deepcopy(state)
        cmd = state.get("command", {})
        if cmd.get("phase") != "executing":
            return

        logger.warning(
            "startup: found executing command %s (comment %s) in %s#%s "
            "- marking as uncertain",
            cmd.get("kind", "?"), cmd.get("comment_id", "?"),
            repo, pr_number,
        )

        cmd["phase"] = "uncertain"
        state["status"] = "deploying"
        state["command"] = cmd
        comment_id = int(cmd.get("comment_id") or 0)
        old_cursor = int(state.get("last_processed_comment_id", 0) or 0)
        if comment_id > old_cursor:
            state["last_processed_comment_id"] = comment_id
        # Extract canonical machine alias from command args (safe read)
        cmd_args = cmd.get("args", {})
        machine_alias = ""
        if isinstance(cmd_args, dict):
            raw_machine = cmd_args.get("machine", "")
            if isinstance(raw_machine, str):
                machine_alias = raw_machine


        markdown = self._set_lifecycle_summary_status(
            comments_mod.uncertain_comment(repo, pr_number, state.get("head_sha", "")),
            "deploying",
        )
        markdown += (
            "\n\nThe Controller will periodically verify the exact target status. "
            "It will not repeat the deployment POST."
        )
        event = {
            "event": "Deploy outcome uncertain",
            "result": "uncertain",
            "timestamp": comments_mod.beijing_now_str(),
        }
        if machine_alias:
            event["machine"] = machine_alias
        written = await self._write_lifecycle_with_history(
            repo, pr_number, state, markdown, event=event,
            expected_lifecycle_state=original_state,
        )
        if not written:
            return
        await self.proxy.project_status_label(
            repo, pr_number, "deploying",
        )
        if comment_id:
            try:
                await self._post_bound_notice(
                    repo, pr_number, head_sha=state.get("head_sha", ""),
                    round_id=round_id_for_state(state), comment_id=comment_id,
                    topic="deployment-recovery-waiting",
                    expected_command_kind="approve_deploy",
                    message=("Deployment status is still being verified after restart. "
                             "No deploy POST will be repeated until the current result is confirmed."),
                )
            except Exception as exc:
                logger.warning("uncertain notice %s#%s exception=%s",
                               repo, pr_number, type(exc).__name__)

    async def _recover_uncertain_deployments(
        self, repo: str, old_state: dict, refreshed_state: dict,
    ) -> bool:
        """Observe attempted components within one short budget; never repost."""
        old_command = old_state.get("command", {})
        args = old_command.get("args", {}) if isinstance(old_command, dict) else {}
        if not isinstance(args, dict):
            return False
        machine_alias = args.get("machine", "")
        if not isinstance(machine_alias, str) or not machine_alias:
            return False
        machine = self.policy.get_machine(machine_alias, repo=repo)
        if machine is None:
            return False
        expected_node_id = args.get("node_id")
        expected_node_host = args.get("node_host")
        if (not isinstance(expected_node_id, str) or not expected_node_id
                or not isinstance(expected_node_host, str) or not expected_node_host
                or machine.node_id != expected_node_id
                or machine.node_host != expected_node_host):
            logger.warning(
                "uncertain deployment machine binding changed; refusing to infer completion %s machine=%s",
                repo, machine_alias,
            )
            return False
        old_components = old_state.get("components", [])
        if not isinstance(old_components, list):
            return False
        fresh_components = refreshed_state.get("components", [])
        if not isinstance(fresh_components, list):
            return False
        fresh_command = refreshed_state.get("command", {})
        old_args = args
        fresh_args = fresh_command.get("args", {}) if isinstance(fresh_command, dict) else {}
        if (
            old_state.get("status") not in {"deploying", "deploy-requested"}
            or refreshed_state.get("status") not in {"deploying", "deploy-requested"}
            or old_state.get("head_sha") != refreshed_state.get("head_sha")
            or round_id_for_state(old_state) != round_id_for_state(refreshed_state)
            or old_state.get("round_archive") != refreshed_state.get("round_archive")
            or old_state.get("deployments") != refreshed_state.get("deployments")
            or not isinstance(old_command, dict)
            or not isinstance(fresh_command, dict)
            or old_command.get("kind") != "approve_deploy"
            or fresh_command.get("kind") != "approve_deploy"
            or old_command.get("phase") != "uncertain"
            or fresh_command.get("phase") not in {"uncertain", "completed"}
            or old_command.get("comment_id") != fresh_command.get("comment_id")
            or type(old_command.get("comment_id")) is not int
            or old_command.get("comment_id", 0) <= 0
            or type(old_state.get("last_processed_comment_id")) is not int
            or old_state.get("last_processed_comment_id", 0) < old_command.get("comment_id", 0)
            or type(refreshed_state.get("last_processed_comment_id")) is not int
            or refreshed_state.get("last_processed_comment_id", 0) < old_command.get("comment_id", 0)
            or not isinstance(old_args, dict)
            or not isinstance(fresh_args, dict)
            or not isinstance(old_args.get("machine"), str)
            or not old_args.get("machine")
            or not isinstance(old_args.get("actor"), str)
            or not old_args.get("actor")
            or old_args.get("machine") != fresh_args.get("machine")
            or old_args.get("actor") != fresh_args.get("actor")
            or old_args.get("node_id") != fresh_args.get("node_id")
            or old_args.get("node_host") != fresh_args.get("node_host")
            or old_args.get("pending_component_ids") != fresh_args.get("pending_component_ids")
            or self._review_evidence_identity(old_state.get("review_evidence", {}))
            != self._review_evidence_identity(refreshed_state.get("review_evidence", {}))
        ):
            return False
        deployed = _component_id_set(old_state.get("deployments", []))
        pending = args.get("pending_component_ids")
        old_by_id: dict[str, dict] = {}
        fresh_by_id: dict[str, dict] = {}
        for components, index in ((old_components, old_by_id), (fresh_components, fresh_by_id)):
            for component in components:
                if not isinstance(component, dict):
                    return False
                component_id = component.get("component_id")
                if not isinstance(component_id, str) or not component_id or component_id in index:
                    return False
                index[component_id] = component
        if (not isinstance(pending, list) or not pending
                or any(not isinstance(cid, str) or not cid for cid in pending)
                or len(set(pending)) != len(pending)
                or not set(pending).issubset(old_by_id)
                or not set(pending).issubset(fresh_by_id)
                or set(pending).intersection(deployed)):
            return False
        recovered = []
        unresolved = []
        deadline = time.monotonic() + AGENT_CORE_STATUS_PROBE_TIMEOUT_SECONDS
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            core = await self._core_for_node(machine.node_id, repo, timeout=remaining)
            for component_id in dict.fromkeys(pending):
                if component_id in deployed:
                    continue
                old = old_by_id.get(component_id)
                component = fresh_by_id.get(component_id)
                if not isinstance(old, dict) or not isinstance(component, dict):
                    unresolved.append(component_id)
                    continue
                identity_fields = (
                    "component_id", "target", "driver_path", "variant",
                    "review_image_tag", "image_ref",
                )
                if any(
                    not isinstance(old.get(field), str)
                    or old.get(field) != component.get(field)
                    for field in identity_fields
                ):
                    unresolved.append(component_id)
                    continue
                if (not old.get("target") or not old.get("review_image_tag")
                        or not old.get("image_ref")
                        or component_id not in self._get_component_ids_for_machine(
                            machine_alias, [component], repo,
                        )):
                    unresolved.append(component_id)
                    continue
                old_runtime_id = old.get("runtime_id")
                fresh_runtime_id = component.get("runtime_id")
                if old.get("target") == "core":
                    if old_runtime_id != "core" or fresh_runtime_id != "core":
                        unresolved.append(component_id)
                        continue
                    runtime_id = "core"
                else:
                    if not isinstance(old_runtime_id, str) or not old_runtime_id:
                        unresolved.append(component_id)
                        continue
                    if ("runtime_id" in component
                            and fresh_runtime_id != old_runtime_id):
                        unresolved.append(component_id)
                        continue
                    runtime_id = old_runtime_id
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    unresolved.append(component_id)
                    continue
                image_ref = old.get("image_ref")
                try:
                    if component.get("target") == "core":
                        update_info = await asyncio.wait_for(
                            core.core_update_check(), timeout=remaining,
                        )
                        target_tag = _target_tag_from_image_ref(image_ref)
                        verified = bool(
                            isinstance(update_info, dict) and target_tag
                            and update_info.get("current_tag") == target_tag
                        )
                        runtime_id = "core"
                    else:
                        status = await asyncio.wait_for(
                            core.driver_status(runtime_id), timeout=remaining,
                        )
                        verified = bool(
                            isinstance(status, dict) and status.get("status") == "running"
                            and status.get("running_image") == image_ref
                        )
                except (AgentCoreError, asyncio.TimeoutError):
                    verified = False
                if verified:
                    recovered.append((component, image_ref, runtime_id))
                else:
                    unresolved.append(component_id)
        except Exception as exc:
            logger.warning("uncertain deployment observation unavailable: %s", type(exc).__name__)
            recovered_component_ids = {component["component_id"] for component, _, _ in recovered}
            unresolved = [
                cid for cid in pending
                if cid not in deployed and cid not in recovered_component_ids
            ]
        # A stale or empty pending list is not proof that any deployment
        # completed. Preserve the uncertain command until an exact target
        # status supplies new deployment evidence.
        if not recovered:
            return False
        already_deployed = _component_id_set(refreshed_state.get("deployments", []))
        refreshed_state["deployments"] = list(refreshed_state.get("deployments", []))
        for component, image_ref, runtime_id in recovered:
            if component["component_id"] in already_deployed:
                continue
            component["runtime_id"] = runtime_id
            component["image_ref"] = image_ref
            refreshed_state["deployments"].append({
                "machine": machine_alias,
                "ip": machine.node_host,
                "component_ids": [component["component_id"]],
                "phase": "deployed",
            })
        updated_command = copy.deepcopy(refreshed_state.get("command", old_command))
        if not isinstance(updated_command, dict):
            return False
        updated_args = updated_command.get("args", {})
        if not isinstance(updated_args, dict):
            updated_args = {}
        updated_args = dict(updated_args)
        updated_args["pending_component_ids"] = unresolved
        updated_command["args"] = updated_args
        updated_command["phase"] = "uncertain" if unresolved else "completed"
        refreshed_state["command"] = updated_command
        all_ids = {c["component_id"] for c in refreshed_state.get("components", [])}
        if all_ids and all_ids == _component_id_set(refreshed_state.get("deployments", [])):
            refreshed_state["status"] = "testing"
            refreshed_state.pop("acceptance_request", None)
        elif not unresolved:
            refreshed_state["status"] = "deploy-requested"
        else:
            refreshed_state["status"] = "deploying"
        return not unresolved

    async def _recover_uncertain_deployment_tick(
        self, repo: str, pr_number: int, state: dict, pr_data: dict,
    ) -> None:
        """Run one short restart-safe observation tick and persist only changes."""
        command = state.get("command", {})
        if (not isinstance(command, dict) or command.get("kind") != "approve_deploy"
                or command.get("phase") != "uncertain"
                or state.get("status") not in {"deploying", "deploy-requested"}
                or pr_data.get("head", {}).get("sha") != state.get("head_sha")):
            return
        refreshed = copy.deepcopy(state)
        if refreshed.get("status") == "deploy-requested":
            refreshed["status"] = "deploying"
        try:
            await self._recover_uncertain_deployments(repo, state, refreshed)
        except Exception as exc:
            logger.warning("uncertain deployment recovery tick %s#%s failed: %s",
                           repo, pr_number, type(exc).__name__)
            return
        if refreshed == state:
            await self._project_status_best_effort(
                repo, pr_number, state.get("status", "deploying"),
            )
            return
        refreshed["last_processed_comment_id"] = max(
            int(state.get("last_processed_comment_id", 0) or 0),
            int(command.get("comment_id", 0) or 0),
        )
        args = command.get("args", {}) if isinstance(command.get("args"), dict) else {}
        machine_alias = str(args.get("machine", "") or "")
        recovered_ids = _component_id_set(refreshed.get("deployments", [])) - _component_id_set(
            state.get("deployments", []),
        )
        required_ids = {
            item.get("component_id") for item in refreshed.get("components", [])
            if isinstance(item, dict) and isinstance(item.get("component_id"), str)
        }
        deployments_complete = bool(required_ids) and required_ids.issubset(
            _component_id_set(refreshed.get("deployments", [])),
        )
        if refreshed.get("status") == "testing":
            markdown = comments_mod.testing(
                repo, pr_number, refreshed.get("head_sha", ""), case_result="",
                deployments=self._enrich_deployments_for_render(refreshed.get("deployments", []), repo),
                components=refreshed.get("components", []),
                deployments_complete=deployments_complete,
            )
        else:
            unresolved = refreshed.get("command", {}).get("args", {}).get(
                "pending_component_ids", [],
            )
            gate_note = ([
                "### Restart Recovery", "",
                "A prior deployment is still being verified. No deployment POST will be repeated.",
            ] if unresolved else [
                "### Restart Recovery", "",
                "The attempted component was verified. Remaining components require a NEW approval.",
            ])
            markdown = self._deploy_requested_markdown(
                repo, pr_number, refreshed.get("head_sha", ""),
                refreshed.get("components", []),
                self._get_machine_groups_for_components([
                    item for item in refreshed.get("components", [])
                    if item.get("component_id") not in _component_id_set(refreshed.get("deployments", []))
                ], repo),
                gate_note=gate_note,
                deployments=self._enrich_deployments_for_render(refreshed.get("deployments", []), repo),
                display_status=refreshed.get("status", "deploy-requested"),
            )
        event = None
        if recovered_ids:
            event = {
                "event": "Deployment status recovered",
                "result": "verified",
                "machine": machine_alias,
                "timestamp": comments_mod.beijing_now_str(),
            }
        written = await self._write_lifecycle_with_history(
            repo, pr_number, refreshed, markdown, event=event,
            expected_terminal_state=state,
        )
        if written:
            if recovered_ids and machine_alias:
                try:
                    await self._save_deployed_machine_log(repo, pr_number, refreshed, machine_alias)
                except Exception:
                    logger.warning("recovered machine log save failed %s#%s", repo, pr_number)
            try:
                await self.proxy.project_status_label(repo, pr_number, refreshed["status"])
            except Exception as exc:
                logger.warning("uncertain recovery label %s#%s exception=%s",
                               repo, pr_number, type(exc).__name__)
            if (refreshed.get("status") == "testing"
                    and state.get("status") != "testing"):
                await self._post_bound_notice(
                    repo, pr_number, head_sha=refreshed["head_sha"],
                    round_id=round_id_for_state(refreshed),
                    comment_id=int(command.get("comment_id", 0) or 0),
                    topic="deployment-recovered-complete",
                    expected_command_kind="approve_deploy",
                    message=(f"The deployment is confirmed complete for HEAD `{_short(refreshed['head_sha'])}` "
                             f"in round `{round_id_for_state(refreshed)}`. Next, run testing and upload the test video."),
                )
            elif recovered_ids and refreshed.get("status") == "deploy-requested":
                remaining = [
                    item for item in refreshed.get("components", [])
                    if item.get("component_id") not in _component_id_set(refreshed.get("deployments", []))
                ]
                lines = self._machine_notice_lines(
                    self._get_machine_groups_for_components(remaining, repo), repo,
                    mention_owners=True, test_only=self._is_test_only_pr(pr_data),
                )
                await self._post_bound_notice(
                    repo, pr_number, head_sha=refreshed["head_sha"],
                    round_id=round_id_for_state(refreshed),
                    comment_id=int(command.get("comment_id", 0) or 0),
                    topic="deployment-recovered-partial",
                    expected_command_kind="approve_deploy",
                    message="\n".join(["A partial deployment was confirmed. The remaining components need another approval.", *lines]),
                )

    async def _refresh_uncertain_state(
        self, repo: str, pr_number: int, state: dict,
    ) -> str:
        """Refresh an uncertain command without replaying the old comment.

        Returns:
        - "deploy-requested" when a fresh validation snapshot was rebuilt.
        - "testing" when every deployment was observed successful without POST.
        - "review-required" when the head drifted.
        - "uncertain" when the rebuild could not complete but must stay pending.
        - "noop" when the PR is no longer open/active.
        """
        original_state = copy.deepcopy(state)
        cmd = state.get("command", {})
        machine_alias = str(cmd.get("args", {}).get("machine", "") or "")
        comment_id = int(cmd.get("comment_id", 0) or 0)
        old_cursor = int(state.get("last_processed_comment_id", 0) or 0)

        pr_data = await self.proxy.get_pr(repo, pr_number)
        pr_state = pr_data.get("state", "")
        pr_merged = pr_data.get("merged", False)
        current_head = pr_data.get("head", {}).get("sha", "")

        if pr_state != "open" or pr_merged or pr_data.get("draft"):
            return "noop"

        if not current_head:
            return "uncertain"
        if not re.fullmatch(r"[0-9a-f]{40}", str(current_head)):
            return "uncertain"
        if current_head != state.get("head_sha", ""):
            archived = await self._archive_round(
                repo, pr_number, copy.deepcopy(state), "HEAD changed",
            )
            old_status = state.get("status", "review-required")
            state["status"] = "review-required"
            state["round_id"] = make_round_id(current_head, comment_id)
            state["round_started_at"] = utc_now()
            state.pop("deployment_logs", None)
            state.pop("acceptance_request", None)
            state["review_evidence"] = {}
            state["components"] = []
            state["deployments"] = []
            state["approve_attempts"] = []
            state["approve_attempts_total"] = 0
            state["approve_attempts_truncated"] = False
            state["case_results"] = {}
            state["test_result"] = ""
            state["cos"] = {"object_key": "", "sha256": "", "size": 0}
            state["round_archive"] = archived
            state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "completed",
                "args": dict(cmd.get("args", {}) or {}),
            }
            state["last_processed_comment_id"] = max(old_cursor, comment_id)
            markdown = comments_mod.review_required(
                repo, pr_number, current_head or state.get("head_sha", ""),
            )
            event = {
                "event": "HEAD drift detected",
                "lifecycle": f"`{old_status}` \u2192 `review-required`",
                "timestamp": comments_mod.beijing_now_str(),
            }
            written = await self._write_lifecycle_with_history(
                repo, pr_number, state, markdown, event=event,
                expected_lifecycle_state=original_state,
            )
            if not written:
                return "uncertain"
            await self.proxy.project_status_label(repo, pr_number, "review-required")
            return "review-required"

        # Get build evidence from PR comments
        try:
            comments = await self.github.get_issue_comments(repo, pr_number)
            if not isinstance(comments, list):
                comments = []
            evidence = await self._review_evidence_from_comments(repo, pr_number, current_head, comments)
        except Exception as e:
            logger.warning(
                "refresh_uncertain comment evidence %s#%s: %s",
                repo, pr_number, e,
            )
            evidence = None

        if evidence is None:
            # Missing evidence cannot erase the fact that a POST may have run.
            return "uncertain"

        # Convert evidence builds to BuildInfo
        resolved_head = await self._resolve_review_evidence_for_head(
            repo, current_head, evidence, pr_number=pr_number,
        )
        if resolved_head is None:
            return "uncertain"
        builds = self._review_build_infos(repo, evidence)

        review_evidence_data = self._review_evidence_snapshot(evidence, resolved_head)
        fresh_components = await self._build_component_snapshot(
            repo, pr_number, current_head, builds,
        )
        if fresh_components is None:
            logger.warning(
                "uncertain recovery snapshot rebuild unavailable %s#%s head=%s",
                repo, pr_number, current_head,
            )
            state["status"] = "deploying"
            state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "uncertain",
                "args": dict(cmd.get("args", {}) or {}),
            }
            state["last_processed_comment_id"] = max(old_cursor, comment_id)
            markdown = self._set_lifecycle_summary_status(
                comments_mod.uncertain_comment(repo, pr_number, current_head),
                "deploying",
            )
            if machine_alias:
                markdown += "\n\n" + "\n".join([
                    "### Restart Recovery",
                    "",
                    "Validation facts are temporarily unavailable.",
                    f"Try a new `/approve_deploy machine={machine_alias}` later.",
                ])
            event = {
                "event": "Restart recovery snapshot unavailable",
                "result": "uncertain",
                "machine": machine_alias,
                "timestamp": comments_mod.beijing_now_str(),
            }
            await self._write_lifecycle_with_history(
                repo, pr_number, state, markdown, event=event,
                expected_lifecycle_state=original_state,
            )
            return "uncertain"

        old_review_evidence = state.get("review_evidence", {})
        old_components = state.get("components", [])

        # Build semantic-key maps for migration compatibility.
        # A semantic key change (target/driver_path/variant/image_tag) means
        # a genuinely different build — reset everything.
        old_semantic_key_list = [
            self._component_semantic_key(oc)
            for oc in old_components
            if isinstance(oc, dict)
        ]
        old_semantic_duplicate = (
            len(old_semantic_key_list) != len(set(old_semantic_key_list))
        )
        old_semantic_keys = set(old_semantic_key_list)
        fresh_semantic_keys = {
            self._component_semantic_key(fc)
            for fc in fresh_components
            if isinstance(fc, dict)
        }

        # A true evidence/tag change resets everything.
        evidence_changed = (
            self._review_evidence_identity(old_review_evidence)
            != self._review_evidence_identity(review_evidence_data)
        )
        # Symmetric set comparison: detects added, removed, and changed components.
        semantic_changed = old_semantic_duplicate or old_semantic_keys != fresh_semantic_keys

        snapshot_reset = evidence_changed or semantic_changed
        if snapshot_reset:
            # A command may have sent a POST whose result is still unknown.
            # New evidence cannot release that machine/component binding or
            # make the old POST safe to repeat.
            return "uncertain"
        # Same evidence AND same semantic keys — apply migration rules.
        deployed_component_ids: set[str] = set()
        for dep in state.get("deployments", []):
            if isinstance(dep, dict) and dep.get("phase") == "deployed":
                for cid in dep.get("component_ids", []):
                    if isinstance(cid, str) and cid:
                        deployed_component_ids.add(cid)

        # Fail-closed: deployed components must have non-empty runtime_id.
        # If not, keep the outcome uncertain instead of releasing its binding.
        _migration_ok = True
        for old_comp in old_components:
            if not isinstance(old_comp, dict):
                continue
            old_cid = old_comp.get("component_id", "")
            if (old_cid in deployed_component_ids and old_comp.get("target") != "core"):
                runtime_id = old_comp.get("runtime_id", "")
                if not isinstance(runtime_id, str) or not runtime_id:
                    _migration_ok = False
                    break

        if not _migration_ok:
            return "uncertain"
        semantic_map: dict[tuple, dict] = {}
        for fc in fresh_components:
            if isinstance(fc, dict):
                semantic_map[self._component_semantic_key(fc)] = fc

        migrated: list[dict] = []
        for old_comp in old_components:
            if not isinstance(old_comp, dict):
                continue
            skey = self._component_semantic_key(old_comp)
            fresh = semantic_map.get(skey)
            if not fresh:
                continue
            old_cid = old_comp.get("component_id", "")
            is_deployed = old_cid in deployed_component_ids
            if is_deployed:
                migrated_component = {
                    "component_id": old_cid,
                    "target": fresh["target"],
                    "driver_path": fresh["driver_path"],
                    "variant": fresh["variant"],
                    "review_image_tag": fresh["review_image_tag"],
                    "image_ref": old_comp.get("image_ref", ""),
                    "resolved_platform": fresh["resolved_platform"],
                }
                migrated_component = self._fresh_component_runtime_binding(
                    migrated_component
                )
                if fresh["target"] != "core":
                    migrated_component["runtime_id"] = old_comp.get("runtime_id", "")
                migrated.append(migrated_component)
            else:
                migrated.append(self._fresh_component_runtime_binding({
                    "component_id": old_cid,
                    "target": fresh["target"],
                    "driver_path": fresh["driver_path"],
                    "variant": fresh["variant"],
                    "review_image_tag": fresh["review_image_tag"],
                    "image_ref": fresh["image_ref"],
                    "resolved_platform": fresh["resolved_platform"],
                }))

        updated_state = dict(state)
        updated_state["review_evidence"] = review_evidence_data
        updated_state["status"] = "deploying"
        updated_state["command"] = {
            "comment_id": comment_id,
            "kind": "approve_deploy",
            "phase": "completed",
            "args": dict(cmd.get("args", {}) or {}),
        }
        updated_state["last_processed_comment_id"] = max(old_cursor, comment_id)
        updated_state["components"] = migrated

        if snapshot_reset:
            # A fresh build must never adopt old runtime/image provenance as
            # this round's deployment. Only a new approval may deploy it.
            build_id = review_evidence_data.get("build_comment_id") or comment_id
            updated_state["round_id"] = make_round_id(current_head, build_id)
            updated_state["round_started_at"] = utc_now()
            updated_state.pop("deployment_logs", None)
            updated_state.pop("acceptance_request", None)
        old_deployment_facts = copy.deepcopy(state.get("deployments", []))
        if not snapshot_reset and not await self._recover_uncertain_deployments(
            repo, state, updated_state,
        ):
            logger.warning("uncertain deployment remains pending; refusing repeated POST %s#%s", repo, pr_number)
            return "uncertain"
        recovered_machines = set()
        if not snapshot_reset:
            old_facts = {
                (dep.get("machine"), tuple(dep.get("component_ids", ())))
                for dep in old_deployment_facts
                if isinstance(dep, dict) and dep.get("phase") == "deployed"
            }
            recovered_machines = {
                dep.get("machine")
                for dep in updated_state.get("deployments", [])
                if isinstance(dep, dict) and dep.get("phase") == "deployed"
                and isinstance(dep.get("machine"), str) and dep.get("machine")
                and (dep.get("machine"), tuple(dep.get("component_ids", ()))) not in old_facts
            }
        if not await self._fresh_review_evidence_matches_state(
            repo, pr_number, current_head, review_evidence_data,
            expected_components=updated_state.get("components", []),
        ):
            return "uncertain"

        state.clear()
        state.update(updated_state)

        gate_note = [
            "### Restart Recovery",
            "",
            "Validation facts were refreshed from the current HEAD.",
        ]
        if machine_alias:
            gate_note.append(
                f"Send a NEW `/approve_deploy machine={machine_alias}`."
            )
        gate_note.append("fresh HEAD + actor -> FINAL PRE-DEPLOY VALIDATION")
        markdown = self._deploy_requested_markdown(
            repo, pr_number, current_head,
            state.get("components", []),
            self._get_machine_groups_for_components([
                c for c in state.get("components", [])
                if c.get("component_id", "") not in _component_id_set(state.get("deployments", []))
            ], repo),
            gate_note=gate_note,
            deployments=self._enrich_deployments_for_render(state.get("deployments", []), repo),
            display_status=state.get("status", "deploy-requested"),
        )
        if state["status"] == "testing":
            required_component_ids = {
                c.get("component_id") for c in state.get("components", [])
                if isinstance(c, dict) and c.get("component_id")
            }
            markdown = comments_mod.testing(
                repo, pr_number, current_head, case_result="",
                deployments=self._enrich_deployments_for_render(state.get("deployments", []), repo),
                components=state.get("components", []),
                deployments_complete=(bool(required_component_ids)
                                      and required_component_ids.issubset(
                                          _component_id_set(state.get("deployments", [])))),
            )
        event = {
            "event": "Restart recovery: evidence refreshed",
            "timestamp": comments_mod.beijing_now_str(),
        }
        lifecycle_written = await self._write_lifecycle_with_history(
            repo, pr_number, state, markdown, event=event,
            expected_lifecycle_state=original_state,
        )
        if not lifecycle_written:
            return "uncertain"
        if lifecycle_written:
            for recovered_machine in sorted(recovered_machines):
                try:
                    await self._save_deployed_machine_log(
                        repo, pr_number, state, recovered_machine,
                    )
                except Exception:
                    logger.warning("refreshed machine log save failed %s#%s", repo, pr_number)
            await self.proxy.project_status_label(repo, pr_number, state["status"])
        return state["status"]

    async def _resolve_review_evidence_for_head(
        self, repo: str, fresh_head: str, evidence: "ReviewCommentEvidence",
        *, pr_number: int | None = None,
    ) -> str | None:
        """Resolve evidence.commit_prefix to full SHA and verify == fresh_head.

        Returns resolved 40-hex SHA on success, or None on any failure.
        """
        from .review_comment_parser import ReviewCommentEvidence
        if not isinstance(evidence, ReviewCommentEvidence):
            return None
        if evidence.head_sha and evidence.head_sha != fresh_head:
            return None
        builds = self._review_build_infos(repo, evidence)
        if not builds or any(not build.success or not build.deployable for build in builds):
            return None
        commit_prefix = evidence.commit_prefix
        if not isinstance(commit_prefix, str) or not re.fullmatch(r"[0-9a-f]{7,40}", commit_prefix):
            logger.warning(
                "resolve evidence commit_prefix invalid: %r", commit_prefix
            )
            return None
        try:
            resolved = await self.github.resolve_commit_sha(repo, commit_prefix)
        except Exception as e:
            logger.warning(
                "resolve_commit_sha %s %s: %s", repo, commit_prefix, e
            )
            return None
        if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
            logger.warning(
                "resolve_commit_sha returned malformed sha: %r", resolved
            )
            return None
        if resolved != fresh_head:
            logger.warning(
                "resolved sha %s != fresh_head %s", resolved, fresh_head
            )
            return None
        if pr_number is not None:
            # Commit resolution awaits GitHub after the run-binding HEAD check.
            try:
                final_pr = await self.proxy.get_pr(repo, pr_number)
            except Exception:
                return None
            if (not isinstance(final_pr, dict)
                    or final_pr.get("head", {}).get("sha") != fresh_head
                    or final_pr.get("state") != "open" or final_pr.get("merged")
                    or final_pr.get("draft") is True):
                return None
        return resolved

        # ── Hidden state validation ──


    async def _revalidate_approve_comment(self, repo: str, comment_id: int, expected_actor_id: str, expected_machine_alias: str):
        """Re-read the exact approval comment from GitHub and validate it.

        Returns the comment dict on success, None on any failure.
        """
        try:
            comment = await self.proxy.get_comment(repo, comment_id)
        except Exception:
            return None
        if not isinstance(comment, dict):
            return None
        cid = comment.get("id")
        if not isinstance(cid, int) or cid <= 0 or isinstance(cid, bool):
            return None
        if cid != comment_id:
            return None
        user = comment.get("user")
        if not isinstance(user, dict):
            return None
        uid = user.get("id")
        if not isinstance(uid, int) or uid <= 0 or isinstance(uid, bool):
            return None
        if str(uid) != expected_actor_id:
            return None
        body = comment.get("body")
        if not isinstance(body, str):
            return None
        from .commands import parse_command
        parsed = parse_command(body)
        if not parsed or parsed.kind != "approve_deploy":
            return None
        if parsed.machine_alias != expected_machine_alias:
            return None
        return comment

    async def _revalidate_hidden_state(
        self,
        repo: str,
        pr_number: int,
        expected_head: str,
        expected_review_evidence: dict,
        expected_status: str = "deploy-requested",
        expected_component_snapshot: list | None = None,
        expected_deployments: list | None = None,
    ):
        """Re-read hidden state before unsafe POST.

        Requires status == "deploy-requested" AND command.phase == "completed".
        Returns the state dict on match, None on any mismatch.
        """
        state = await self.proxy.read_hidden_state(repo, pr_number)
        if state is None:
            return None
        if state.get("status") != expected_status:
            return None
        if state.get("head_sha") != expected_head:
            return None
        if state.get("review_evidence") != expected_review_evidence:
            return None
        if state.get("command", {}).get("phase") != "completed":
            return None
        if expected_component_snapshot is not None:
            if self._canonical_component_snapshot(
                state.get("components", []),
            ) != expected_component_snapshot:
                return None
        if expected_deployments is not None:
            if state.get("deployments", []) != expected_deployments:
                return None
        return state


    async def _fresh_review_evidence_matches_state(
        self, repo: str, pr_number: int, fresh_head: str,
        persisted_review_evidence: dict,
        *, expected_components: list[dict] | None = None,
    ):
        """Fresh-read PR Conversation and validate evidence matches persisted snapshot."""
        try:
            comments = await self.proxy.get_issue_comments(repo, pr_number)
        except Exception:
            return False
        if not isinstance(comments, list):
            return False
        persisted_build_id = persisted_review_evidence.get("build_comment_id")
        if (type(persisted_build_id) is int
                and self._has_untrusted_newer_review_comment(
                    comments, persisted_build_id,
                )):
            return False
        evidence = await self._review_evidence_from_comments(repo, pr_number, fresh_head, comments)
        if evidence is None:
            return False
        resolved_head = await self._resolve_review_evidence_for_head(
            repo, fresh_head, evidence, pr_number=pr_number,
        )
        if resolved_head != fresh_head:
            return False
        fresh_components = await self._build_component_snapshot(
            repo, pr_number, fresh_head, self._review_build_infos(repo, evidence),
        )
        if not fresh_components:
            return False
        if expected_components is not None:
            # Compare semantic identities so historical digest/runtime bindings
            # survive, while a truncated or duplicated matrix cannot reach POST.
            if sorted(self._component_semantic_key(c) for c in fresh_components) != sorted(
                self._component_semantic_key(c) for c in expected_components
            ):
                return False
            if any(c.get("resolved_platform") != get_deploy_platform() for c in expected_components):
                return False
        fresh_snapshot = self._review_evidence_snapshot(evidence, resolved_head)
        return (
            self._review_evidence_identity(fresh_snapshot)
            == self._review_evidence_identity(persisted_review_evidence)
        )


    def _validate_hidden_state(self, state: dict) -> None:
        """Validate hidden state before persisting."""
        try:
            _validate_hidden_state(state)
        except Exception as e:
            raise DeployControllerError(str(e)) from e

    # ── Command dispatcher ──

    async def on_command(
        self, cmd, repo: str, pr_number: int, comment_id: int,
    ) -> bool:
        """Dispatch a parsed command to the appropriate handler.

        Returns True if the command was handled (cursor advances).
        """
        # Re-read the comment to verify author identity
        author_id, author_login = await self.proxy.comment_identity(
            repo, comment_id
        )
        if not author_id and not author_login:
            logger.warning(
                "on_command: cannot verify author for %s#%s cid=%s",
                repo, pr_number, comment_id,
            )
            await self._post_error(
                repo, pr_number,
                "Cannot verify this command's GitHub author, so no action was taken. Check the account identity and post a new command.",
                comment_id=comment_id,
            )
            return True

        intent = getattr(cmd, "intent", "") or cmd.kind
        if getattr(cmd, "error", "") or getattr(cmd, "intent", ""):
            approval_guidance = (
                "\n" + await self._approval_command_guidance(repo, pr_number)
                if intent == "approve_deploy" else ""
            )
            if getattr(cmd, "intent", ""):
                suggestions = {
                    "approve_deploy": "Use `/approve_deploy` with a complete machine command from the current Lifecycle Summary.",
                    "request_deploy": "Use `/request_deploy`.",
                    "request_acceptance": "Use `/request_acceptance`.",
                    "record_test": "Use `/record_test result=pass` or `/record_test result=fail`.",
                }
                message = (f"Unrecognized deployment command. {suggestions.get(intent, 'Check `/deploy_help`.')}")
            elif intent == "approve_deploy" and getattr(cmd, "error", "") == "missing_machine_value":
                message = "`machine=` is empty. Use a configured machine IP from the current compatible command list, or omit the argument only when exactly one fully compatible machine is available."
            elif intent == "approve_deploy":
                message = "Invalid `/approve_deploy` arguments. Use a complete machine command shown below, or omit `machine` only when exactly one fully compatible machine is available."
            elif intent == "request_deploy":
                message = "`/request_deploy` takes no parameters. Remove extra arguments and try again."
            elif intent == "request_acceptance":
                message = "`/request_acceptance` takes no parameters. Remove extra arguments and try again."
            elif intent == "record_test":
                message = "`/record_test` requires `result=pass` or `result=fail`."
            else:
                message = f"Invalid `/{intent}` command syntax. Check `/deploy_help {intent}` and try again."
            message += approval_guidance
            await self._post_error(
                repo, pr_number, message, comment_id=comment_id,
            )
            return True

        if cmd.kind == "request_deploy":
            return await self.handle_request_deploy(
                repo, pr_number, comment_id,
            )

        if cmd.kind == "approve_deploy":
            machine_alias = cmd.machine_alias
            return await self.handle_approve_deploy(
                repo, pr_number, comment_id,
                machine_alias, author_login, author_id,
            )

        if cmd.kind == "record_test":
            result = cmd.result
            summary = cmd.summary
            if result not in ("pass", "fail"):
                await self._post_error(
                    repo, pr_number,
                    "`/record_test` requires `result=pass` or `result=fail`.",
                    comment_id=comment_id,
                )
                return True
            return await self.handle_record_test(
                repo, pr_number, comment_id,
                result, summary, author_login, author_id,
            )

        if cmd.kind == "request_acceptance":
            return await self.handle_request_acceptance(
                repo, pr_number, comment_id, author_login, author_id,
            )

        if cmd.kind == "deploy_status":
            return await self.handle_deploy_status(
                repo, pr_number, comment_id,
            )

        if cmd.kind == "deploy_help":
            return await self.handle_deploy_help(
                repo, pr_number, comment_id,
                topic=cmd.help_topic,
            )

        if cmd.kind == "evidence_link":
            return await self.handle_evidence_link(
                repo, pr_number, comment_id, author_login, author_id,
            )

        if cmd.kind in {"submit_evidence", "remove_evidence"}:
            evidence = self.evidence_service
            if evidence is None:
                await self._post_error(
                    repo, pr_number,
                    "Evidence storage is unavailable; the comment was not accepted.",
                    comment_id=comment_id,
                )
                return True
            try:
                await evidence.handle_native_comment(
                    repo=repo, pr_number=pr_number, comment_id=comment_id,
                    actor_id=int(author_id), actor_login=author_login,
                    action=cmd.kind, arguments=cmd.evidence_args or {},
                )
            except Exception as exc:
                # Business rejections are terminal comments; network/API/COS
                # exceptions escape so the watcher retries this exact comment.
                if getattr(exc, "status_code", None) in {400, 403, 404, 409, 410, 413}:
                    await self._post_error(
                        repo, pr_number,
                        f"Evidence comment rejected: {getattr(exc, 'code', 'invalid_request')}.",
                        comment_id=comment_id,
                    )
                    return True
                raise
            return True

        logger.warning("on_command: unknown kind %s", cmd.kind)
        return True


# Backward-compatible aliases
DeployServiceError = DeployControllerError
DeploymentService = DeployController
