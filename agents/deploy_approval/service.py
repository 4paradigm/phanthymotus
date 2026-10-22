"""Deploy Controller — stateless command handler for Deploy Approval.

The Deploy Controller is stateless between commands. Its ONLY persistence is
the GitHub lifecycle comment hidden state, read/written through GitHubStateProxy.

Commands:
  - /request_deploy — PR Author requests deployment for current HEAD
  - /approve_deploy — Machine Owner approves and binds machine
  - /record_test — Overall validation verdict

State machine (hidden state only):
  review-required -> reviewing -> deploy-ready -> deploy-requested
  -> testing -> succeeded | failed
"""

from __future__ import annotations

import hashlib
import re
import inspect

import asyncio
import logging
import time
import copy
from datetime import datetime, timedelta
from typing import Any

from . import comments as comments_mod
from .agent_core_client import (
    AgentCoreClient,
    AgentCoreDeployOutcomeUncertain,
    AgentCoreError,
)
from .case_runner import CaseRunner
from .config import Config, DEFAULT_GITHUB_REPOS, validate_repository_names
from .cos_client import CosClient
from .evidence_builder import EvidenceBuilder
from .evidence_models import archive_snapshot, make_round_id, round_id_for_state, utc_now
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


class DeployControllerError(Exception):
    pass


class DeployOutcomeUncertain(DeployControllerError):
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
        self._core_clients: dict[str, AgentCoreClient] = {}
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

    # ── Core helpers ──

    async def _core_for_node(self, node_id: str) -> AgentCoreClient:
        if node_id in self._core_clients:
            return self._core_clients[node_id]
        client = await self._resolve_core_client(node_id)
        self._core_clients[node_id] = client
        return client

    async def _resolve_core_client(self, node_id: str) -> AgentCoreClient:
        if self._agent_core_factory is not None:
            client = self._agent_core_factory(node_id)
            if client is not None:
                return client
        machine = self.policy.get_machine_by_node_id(node_id)
        if machine is None or not machine.node_host:
            raise DeployControllerError(
                f"machine node {node_id!r} is missing a configured node_host"
            )
        token = self.config.agent_core_tokens.get(machine.alias)
        if not token:
            raise DeployControllerError(
                f"no agent_core_token configured for machine {machine.alias!r}"
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
            await core.verify()
        except AgentCoreError as e:
            await core.aclose()
            raise DeployControllerError(
                f"Agent Core {node_id} unreachable: {e}"
            )
        return core

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

    async def _review_evidence_from_comments(self, repo: str, pr_number: int, head_sha: str, comments: list[dict]):
        """Authenticate the real triggering comment and recheck HEAD after awaits.

        Parsed unbound evidence remains available for inspection, but its
        blocking flag forbids readiness, requests and pre-POST revalidation.
        """
        evidence = self._extract_trusted_review_evidence(comments, head_sha=head_sha)
        if evidence is not None and evidence.test_results_block_deployment:
            logger.warning("REVIEW_EVIDENCE_BLOCKED TEST_RESULTS_STATE=%s TEST_FAILED=%s REQUEST_RUN_BOUND=%s",
                           evidence.test_status, evidence.test_failed, evidence.run_binding_valid)
        if evidence is None or evidence.request_comment is None or not evidence.run_binding_valid:
            return evidence
        request = evidence.request_comment
        original_evidence = copy.deepcopy(evidence)
        evidence.run_binding_valid = False
        try:
            fresh = await self.proxy.get_comment(repo, int(request["id"]))
            for key in ("id", "body", "created_at", "updated_at", "user"):
                if fresh.get(key) != request.get(key):
                    return evidence
            user = fresh["user"]
            pr = await self.proxy.get_pr(repo, pr_number)
            if (pr.get("head", {}).get("sha") != head_sha or pr.get("state") != "open"
                    or pr.get("merged") or pr.get("draft") is True):
                return evidence
            author = pr.get("user", {})
            if (str(author.get("id", "")) != str(user["id"])
                    or author.get("login", "").casefold() != user["login"].casefold()):
                permission = await self.proxy.collaborator_permission(repo, user["login"])
                if not Policy.collaborator_can_approve(permission):
                    return evidence
            if await self._resolve_commit_prefix_for_head(repo, head_sha, evidence.commit_prefix) != head_sha:
                return evidence
            # Permission/commit lookups await GitHub too. A newer run, deleted
            # evidence or an edited result during those awaits must revoke this
            # snapshot even when the PR HEAD itself has not changed.
            latest_comments = await self.proxy.get_issue_comments(repo, pr_number)
            if not isinstance(latest_comments, list):
                return evidence
            latest_evidence = self._extract_trusted_review_evidence(latest_comments, head_sha=head_sha)
            if latest_evidence != original_evidence:
                return evidence
            final_pr = await self.proxy.get_pr(repo, pr_number)
            if (final_pr.get("head", {}).get("sha") != head_sha
                    or final_pr.get("state") != "open" or final_pr.get("merged")
                    or final_pr.get("draft") is True):
                return evidence
        except Exception:
            return evidence
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
        state["status"] = status
        state["review_evidence"] = dict(review_evidence or {})
        state["components"] = []
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
        """Keep every parsed row; build validity is independent of project tests."""
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
                )
                return True

            if state is None:
                await self._post_error(
                    repo, pr_number,
                    "No deploy approval lifecycle found. Wait for review lifecycle reconciliation first.",
                )
                return True
            pr_head = pr_data.get("head", {}).get("sha", "")
            if not isinstance(pr_head, str) or not re.fullmatch(r"[0-9a-f]{40}", pr_head):
                await self._post_error(repo, pr_number, "Cannot verify the current PR HEAD.")
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
            elif _current_status in ("deploy-requested", "testing"):
                logger.warning(
                    "STALE_COMMAND_IGNORED kind=request_deploy comment_id=%s current_status=%s",
                    comment_id, _current_status,
                )
                await self.proxy.persist_cursor(repo, pr_number, comment_id)
                return True
            else:
                await self._post_command_not_ready(
                    repo, pr_number, _current_status,
                    "Wait for the PR to reach `deploy-ready` status before requesting deployment.",
                )
                return True
            pr_user = pr_data.get("user", {})
            pr_author_id = str(pr_user.get("id") or "") if isinstance(pr_user, dict) else ""
            if not pr_author_id:
                await self._post_error(
                    repo, pr_number,
                    "Cannot verify PR author identity.",
                )
                return True
            if comment_author_id != pr_author_id:
                await self._post_error(
                    repo, pr_number,
                    "Only the PR author can use `/request_deploy`.",
                )
                return True

            # Get build evidence from PR comments
            try:
                comments = await self.github.get_issue_comments(repo, pr_number)
                if not isinstance(comments, list):
                    comments = []
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
                )
                return True

            if not evidence.run_binding_valid:
                await self._post_error(
                    repo, pr_number,
                    "Review request, required build matrix or run provenance is missing or ambiguous. "
                    "No deployment can be requested from unbound review evidence.",
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
                )
                return True
            if not components:
                await self._post_error(
                    repo, pr_number,
                    "No deployable builds found for this HEAD.",
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
                )
                return True

            # Use the same canonical snapshot as reconcile and recovery.
            review_evidence_data = self._review_evidence_snapshot(
                evidence, resolved_head,
            )

            # Determine compatible machine groups
            machine_groups = self._get_machine_groups_for_components(components)

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
            state["round_archive"] = await self._archive_round(
                repo, pr_number, previous_state, "New deployment request",
            )
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
            await self._write_lifecycle_with_history(
                repo, pr_number, state, markdown, event=event,
            )
            await self.proxy.project_status_label(repo, pr_number, "deploy-requested")
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
                    "No active deployment found. Use `/request_deploy` first.",
                )
                return True

            pr_head = pr_data.get("head", {}).get("sha", "")
            if not isinstance(pr_head, str) or not re.fullmatch(r"[0-9a-f]{40}", pr_head):
                await self._post_error(repo, pr_number, "Cannot verify the current PR HEAD.")
                return True
            if pr_head != state.get("head_sha", ""):
                await self._supersede_head_drift(
                    repo, pr_number, state, pr_head, comment_id,
                )
                return True

            _current_status = state.get("status", "")
            if _current_status == "deploy-requested":
                pass  # normal path continues below
            elif _current_status in ("testing", "succeeded", "failed"):
                logger.warning(
                    "STALE_COMMAND_IGNORED kind=approve_deploy comment_id=%s current_status=%s",
                    comment_id, _current_status,
                )
                await self.proxy.persist_cursor(repo, pr_number, comment_id)
                return True
            else:
                await self._post_command_not_ready(
                    repo, pr_number, _current_status,
                    "Deploy Approval is not yet in `deploy-requested` state. Wait for Review Agent to complete and Developer to run `/request_deploy`.",
                )
                return True

            # Check PR is still valid
            pr_state = pr_data.get("state", "")
            pr_merged = pr_data.get("merged", False)
            if pr_state != "open" or pr_merged:
                await self._post_error(
                    repo, pr_number,
                    "PR is not open. Deploy only when PR is open and unmerged.",
                )
                return True

            if pr_data.get("draft") is True:
                await self._post_error(
                    repo, pr_number,
                    "Deployment requires PR ready-for-review / non-draft.",
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
            try:
                machine = Policy.resolve_machine_selector(machine_alias, self.policy.machines)
            except PolicyError as e:
                await self._post_error(repo, pr_number, str(e))
                return True

            # Preserve the ORIGINAL selector (IPv4 or alias) from the comment
            # for exact revalidation. Canonical alias is used for state/persistence.
            machine_selector = machine_alias
            # Use canonical alias for all state persistence
            machine_alias = machine.alias

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
                )
                return True


            compatible_remaining_ids = set(
                self._get_component_ids_for_machine(
                    machine_alias, undeployed_components,
                )
            )

            # Multi-machine partial coverage: select only compatible, undeployed components.
            selected_components = [
                c for c in undeployed_components
                if c.get("component_id", "") in compatible_remaining_ids
            ]

            if not selected_components:
                # Selected machine covers zero remaining components.
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
                    self._get_machine_groups_for_components(undeployed_components),
                    gate_note=[
                        f"Machine `{machine_alias}` does not cover any remaining component.",
                        "ZERO deploy POST.",
                        "Send a NEW `/approve_deploy machine=<alias-or-ip>` for a compatible machine.",
                    ],
                    deployments=self._enrich_deployments_for_render(state["deployments"]),
                )
                event = {
                    "event": f"Machine `{machine_alias}` selected",
                    "machine": machine_alias,
                    "timestamp": comments_mod.beijing_now_str(),
                }
                await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                )
                await self.proxy.project_status_label(repo, pr_number, "deploy-requested")
                return True


            try:
                await self._write_lifecycle_with_history(
                    repo, pr_number, state,
                    self._deployment_progress_markdown(
                        repo, pr_number, state, machine, selected_components, "Approval received; checking",
                    ), event=None,
                )
                node_id = machine.node_id
                core = await self._core_for_node(node_id)
                preflight = await self._preflight_running_images(core, selected_components)
            except (AgentCoreError, DeployControllerError, TimeoutError, OSError) as exc:
                # This try block contains read-only preflight and no deploy POST.
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
                    f"Preflight failed ({type(exc).__name__}); target machine/runtime unavailable or invalid. ZERO deploy POST. Send a NEW approval after recovery.",
                )
                await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown,
                    event={"event": "Preflight failed: ZERO deploy POST", "machine": machine_alias,
                           "ip": machine.node_host, "reason": type(exc).__name__,
                           "timestamp": comments_mod.beijing_now_str()},
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
                await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                )
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

            # Write executing state only after ALL gates pass.
            # Persist the complete template before any deploy POST.
            state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "executing",
                "args": {
                    "machine": machine_alias,
                    "actor": actor,
                    # A restart cannot prove which selected POST was sent.
                    "pending_component_ids": [c["component_id"] for c in selected_components],
                },
            }
            await self._write_lifecycle_with_history(
                repo, pr_number, state,
                self._deployment_progress_markdown(
                    repo, pr_number, state, machine, selected_components, "Deploying",
                ), event=None,
            )

            new_deployments = []
            deploy_error = None
            deploy_outcome_uncertain = False
            # Pre-flight visible history capacity before unsafe deploy POST
            prospective = comments_mod.deploy_requested(
                repo, pr_number, pr_head,
                state.get("components", []),
                self._get_machine_groups_for_components(state.get("components", [])),
                deployments=state.get("deployments", []),
                last_lifecycle_event=comments_mod.beijing_now_str(),
            )
            # Pass a conservative deployment event reservation so preflight
            # measures: base renderer + existing history + this event
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

            # Order components: non-core first, core last (core restarts agent-core)
            ordered_components = self._order_components_for_deploy(selected_components)

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

                await self._write_lifecycle_with_history(
                    repo, pr_number, state,
                    self._deployment_progress_markdown(
                        repo, pr_number, state, machine, [comp],
                        "Deploying and verifying Core" if comp_target == "core" else "Deploying component",
                    ), event=None,
                )
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

                    await self._write_lifecycle_with_history(
                        repo, pr_number, state,
                        self._deployment_progress_markdown(
                            repo, pr_number, state, machine, [comp], "Verifying target image",
                        ), event=None,
                    )
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
                        "component_ids": [component_id],
                        "phase": "deployed",
                    })

            if deploy_outcome_uncertain:
                state["deployments"] = list(existing_deployments) + list(new_deployments)
                state["status"] = "deploy-requested"
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "uncertain",
                    "args": {
                        "machine": machine_alias,
                        "actor": actor,
                        "pending_component_ids": [component_id],
                    },
                }
                state["last_processed_comment_id"] = comment_id
                approve_attempt["outcome"] = "uncertain"
                self._record_approve_attempt(state, approve_attempt)
                markdown = comments_mod.deploy_requested(
                    repo,
                    pr_number,
                    pr_head,
                    state.get("components", []),
                    self._get_machine_groups_for_components([
                        c for c in state.get("components", [])
                        if c.get("component_id", "") not in _component_id_set(state["deployments"])
                    ]),
                    gate_note=[
                        "### Restart Recovery",
                        "",
                        "A deploy POST was attempted but its outcome is uncertain.",
                        f"Send a NEW `/approve_deploy machine={machine_alias}`.",
                        "No further deploy POSTs are allowed in this cycle.",
                    ],
                    deployments=self._enrich_deployments_for_render(state["deployments"]),
                )
                event = {
                    "event": "Deploy outcome uncertain",
                    "result": "uncertain",
                    "machine": machine_alias,
                    "timestamp": comments_mod.beijing_now_str(),
                }
                await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                )
                await self.proxy.project_status_label(repo, pr_number, "deploy-requested")
                return True

            if deploy_error is not None:
                state["deployments"] = list(state.get("deployments", [])) + list(new_deployments)
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
                await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                )
                await self.proxy.project_status_label(repo, pr_number, "failed")

                # One-shot runtime log snapshot for failed path
                runtime_logs = await self._snapshot_terminal_runtime_logs(
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
                markdown = comments_mod.deploy_requested(
                    repo, pr_number, pr_head,
                    state["components"],
                    self._get_machine_groups_for_components(remaining),
                    gate_note=[
                        "### Partial coverage completed",
                        "",
                        f"Machine `{machine_alias}` deployed its compatible components.",
                        f"Remaining components need additional machine approval.",
                        "Send a NEW `/approve_deploy machine=<alias-or-ip>`.",
                    ],
                    deployments=self._enrich_deployments_for_render(state["deployments"]),
                )
                machine_info = self.policy.get_machine(machine_alias)
                deploy_event = {
                    "event": f"Machine `{machine_alias}` deployed",
                    "machine": machine_alias,
                    "ip": machine_info.node_host if machine_info else "",
                    "timestamp": comments_mod.beijing_now_str(),
                }
                await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=deploy_event,
                )
                await self.proxy.project_status_label(repo, pr_number, "deploy-requested")
                return True

            # All components deployed -> durable testing state FIRST
            state["status"] = "testing"

            # Build testing markdown WITHOUT advisory case result
            testing_markdown = comments_mod.testing(
                repo, pr_number, pr_head, case_result="",
                deployments=self._enrich_deployments_for_render(state.get("deployments", [])),
                components=state.get("components", []),
                deployments_complete=(bool(all_component_ids)
                                      and all_component_ids.issubset(deployed_component_ids)),
            )
            # The LAST machine also gets its own per-machine history event,
            # mirroring the partial-coverage path, so every successful machine
            # appears exactly once in the visible History (newest-first).
            machine_info = self.policy.get_machine(machine_alias)
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
            await self._write_lifecycle_with_history(
                repo, pr_number, state, testing_markdown,
                events=[event_all, last_machine_event],
            )
            # Project status label "testing" SECOND
            await self.proxy.project_status_label(repo, pr_number, "testing")

            return True

        except PolicyError as e:
            await self._post_error(repo, pr_number, str(e))
            return True
        except Exception as e:
            logger.error(
                "handle_approve_deploy %s#%s: %s",
                repo, pr_number, e,
            )
            raise

    def _all_required_components_deployed(self, state: dict) -> bool:
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
            if not isinstance(alias, str) or not alias or self.policy.get_machine(alias) is None:
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
        try:
            await self._check_record_test_permissions(repo, login, state)
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
                or not self._all_required_components_deployed(expected_state)
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
        if not self._all_required_components_deployed(fresh):
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
                    or not self._all_required_components_deployed(state)):
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
                    deployments=self._enrich_deployments_for_render(state["deployments"]),
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
                    "No active deployment found. Use `/request_deploy` first.",
                )
                return True

            # Freeze the accepted bindings before permissions/logging awaits.
            state = copy.deepcopy(state)
            pr_head = pr_data.get("head", {}).get("sha", "")
            if not isinstance(pr_head, str) or not re.fullmatch(r"[0-9a-f]{40}", pr_head):
                await self._post_error(repo, pr_number, "Cannot verify the current PR HEAD.")
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
                    if not self._all_required_components_deployed(state):
                        return False
                    await self._check_record_test_permissions(repo, actor, state)
                    await self._retry_terminal_cos(repo, pr_number, state, refresh_url=True)
                    return True
                logger.warning(
                    "STALE_COMMAND_IGNORED kind=record_test comment_id=%s current_status=%s",
                    comment_id, _current_status,
                )
                await self.proxy.persist_cursor(repo, pr_number, comment_id)
                return True
            else:
                await self._post_command_not_ready(
                    repo, pr_number, _current_status,
                    "Deployment has not yet reached `testing` state. Complete all required component deployments first.",
                )
                return True

            # Check PR is still valid
            pr_state = pr_data.get("state", "")
            pr_merged = pr_data.get("merged", False)
            if pr_state != "open" or pr_merged or pr_data.get("draft"):
                await self._post_error(
                    repo, pr_number,
                    "PR is not open. Record test only when PR is open and unmerged.",
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
                    or not self._all_required_components_deployed(state)):
                await self._post_error(
                    repo, pr_number,
                    "Not all components are deployed yet. "
                    "Complete all machine approvals before recording test result.",
                )
                return True

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

            # Write the terminal visible lifecycle comment
            if result == "pass":
                enriched = self._enrich_deployments_for_render(
                    state.get("deployments", []),
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
            # Failures are logged/backed off internally and cannot undo or block
            # consumption of this durably completed business command.
            await self._retry_terminal_cos(repo, pr_number, state)
            return True



        except PolicyError as e:
            await self._post_error(repo, pr_number, str(e))
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
        self, machine_alias: str, components: list[dict],
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
        machine = self.policy.get_machine(machine_alias)
        if machine is None:
            return []
        result = []
        for comp in components:
            if not self.policy.machine_supports_target(machine_alias, comp.get("target", "")):
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

    def _get_machine_groups_for_components(
        self, components: list[dict],
    ) -> list[dict]:
        """Determine compatible machine groups for the given components.

        Returns machines whose compatible component_ids are a non-empty
        subset of the complete set of component_ids passed in.

        Machines with partial coverage ARE included — this supports
        multi-machine deployment where no single machine covers all
        remaining components.
        """
        machines = self.policy.get_machines()
        required_ids = {
            comp.get("component_id", "")
            for comp in components
            if isinstance(comp, dict) and comp.get("component_id")
        }
        groups = []
        for m in machines:
            compatible = []
            for comp in components:
                if not self.policy.machine_supports_target(
                    m.alias, comp.get("target", ""),
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

    def _enrich_deployments_for_render(self, deployments: list[dict]) -> list[dict]:
        """Enrich deployment rows with machine IP for render-only display.

        Resolves each deployment's canonical machine alias against the
        configured policy to obtain node_host. This works even when the
        machine is no longer among the remaining-compatible machine groups.

        Hidden-state authoritative identity remains the alias only.
        """
        enriched: list[dict] = []
        for dep in deployments:
            if not isinstance(dep, dict):
                enriched.append(dep)
                continue
            machine_alias = dep.get("machine", "")
            ip = ""
            if machine_alias:
                machine_info = self.policy.get_machine(machine_alias)
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
        has_non_core = any(c.get("target") != "core" for c in components)
        if has_non_core:
            drivers = await core.list_drivers()
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
                    update_info = await core.core_update_check()
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
                status = await core.driver_status(runtime_id)
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
        """Verify that a deployed runtime is actually running the target image.

        Bounded polling using self.config.total_timeout as the verification budget.
        Success requires BOTH status=="running" AND running_image == target_image_ref
        (exact string comparison).

        Returns (True, health_evidence_dict) on success.
        Returns (False, health_evidence_dict) on timeout/mismatch — caller must treat as uncertain.
        """
        deadline = time.monotonic() + self.config.total_timeout
        poll_interval = 2  # seconds between polls
        last_status = ""
        last_running_image = ""
        last_error: str | None = None

        while True:
            if time.monotonic() >= deadline:
                break
            try:
                status_data = await core.driver_status(runtime_id)
            except AgentCoreError as e:
                # Transient error — continue polling until deadline
                last_error = str(e)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(poll_interval, remaining))
                continue

            status_val = status_data.get("status", "")
            running_image = status_data.get("running_image", "")
            last_status = status_val if isinstance(status_val, str) else ""
            last_running_image = running_image if isinstance(running_image, str) else ""

            if (isinstance(status_val, str) and status_val == "running"
                    and isinstance(running_image, str)
                    and running_image == target_image_ref):
                # Success: exact target image observed and running
                return True, {
                    "component_id": component_id,
                    "runtime_id": runtime_id,
                    "status": "running",
                    "running_image": target_image_ref,
                    "target_image": target_image_ref,
                    "verified": True,
                }

            # Still polling: status not yet running or image not yet converged
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval, remaining))

        # Deadline reached — outcome is uncertain (POST already happened)
        result: dict[str, Any] = {
            "component_id": component_id,
            "runtime_id": runtime_id,
            "status": last_status,
            "running_image": last_running_image,
            "target_image": target_image_ref,
            "verified": False,
        }
        if last_error:
            result["error"] = last_error
        return False, result

    async def _snapshot_terminal_runtime_logs(
        self, components: list[dict], deployments: list[dict],
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
                # Core has no driver_status; use sentinel log
                if comp_target == "core":
                    runtime_id = "core"
                    seen_key = (machine_alias, runtime_id)
                    if seen_key in seen:
                        continue
                    seen.add(seen_key)
                    source_key = f"{machine_alias}/{runtime_id}"
                    result[source_key] = "[CORE_RUNTIME_LOG_SNAPSHOT_UNAVAILABLE]"
                    continue
                if not runtime_id:
                    continue
                seen_key = (machine_alias, runtime_id)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                source_key = f"{machine_alias}/{runtime_id}"
                machine = self.policy.get_machine(machine_alias)
                if machine is None:
                    result[source_key] = "[RUNTIME_LOG_SNAPSHOT_UNAVAILABLE]"
                    continue
                try:
                    core = await self._core_for_node(machine.node_id)
                    status = await core.driver_status(runtime_id)
                    logs = status.get("logs", "") if isinstance(status, dict) else ""
                    if isinstance(logs, str) and logs:
                        result[source_key] = logs
                    else:
                        result[source_key] = "[RUNTIME_LOG_SNAPSHOT_UNAVAILABLE]"
                except Exception:
                    result[source_key] = "[RUNTIME_LOG_SNAPSHOT_UNAVAILABLE]"
        return result

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

    async def _check_record_test_permissions(
        self, repo: str, actor: str, state: dict,
    ) -> None:
        """Check if actor is authorized to record test result.

        Authorized if:
        1. Actor has write/maintain/admin GitHub collaborator permission, OR
        2. Actor is an owner of any machine actually used in deployments.

        Fail closed on permission lookup error.
        """
        actor_lower = actor.strip().lower() if actor else ""

        # Check if actor is owner of any deployed machine
        deployed_machines = set()
        for dep in state.get("deployments", []):
            if dep.get("phase") == "deployed":
                machine_alias = dep.get("machine", "")
                if machine_alias:
                    deployed_machines.add(machine_alias)

        for machine_alias in deployed_machines:
            machine = self.policy.get_machine(machine_alias)
            if machine:
                owners_lower = [o.lower() for o in (machine.owners or [])]
                if actor_lower in owners_lower:
                    return

        # Check collaborator permission
        try:
            permission = await self.proxy.collaborator_permission(repo, actor)
        except Exception as e:
            raise PolicyError(
                f"Cannot verify collaborator permission for {actor}: {e}"
            )

        if not Policy.collaborator_can_approve(permission):
            raise PolicyError(
                f"Actor {actor} is not a deployed machine owner and has "
                f"insufficient collaborator permission ({permission}). "
                "Write, maintain, or admin access required."
            )

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
        """Poll /api/system/update-check until current_tag matches target_tag.

        Success condition: current_tag is a non-empty string AND equals target_tag.
        up_to_date==True alone is NOT sufficient (Agent Core may return it
        incorrectly on error paths).

        Connection failures during restart are treated as transient — the
        bounded polling budget continues until deadline.

        Returns (True, evidence) on verified success.
        Returns (False, evidence) on timeout/uncertainty.
        """
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

        deadline = time.monotonic() + self.config.total_timeout
        poll_interval = 2  # seconds between polls
        last_current_tag = ""
        last_error: str | None = None

        while True:
            if time.monotonic() >= deadline:
                break
            try:
                update_info = await core.core_update_check()
            except AgentCoreError as e:
                # Transient (restart in progress) — continue polling
                last_error = str(e)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(poll_interval, remaining))
                continue

            current_tag = str(update_info.get("current_tag", "") or "")
            last_current_tag = current_tag

            if current_tag and current_tag == target_tag:
                # Exact match — deployment confirmed
                return True, {
                    "component_id": component_id,
                    "runtime_id": "core",
                    "current_tag": current_tag,
                    "target_tag": target_tag,
                    "verified": True,
                }

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval, remaining))

        # Deadline reached — outcome uncertain (POST already happened)
        result: dict[str, Any] = {
            "component_id": component_id,
            "runtime_id": "core",
            "current_tag": last_current_tag,
            "target_tag": target_tag,
            "verified": False,
        }
        if last_error:
            result["error"] = last_error
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
            update_info = await core.core_update_check()
        except AgentCoreError as e:
            raise DeployControllerError(
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
            machine = self.policy.get_machine(machine_alias)
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
                core = await self._core_for_node(node_id)
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
    ) -> dict:
        """Upload COS evidence bundle. Returns metadata dict.

        Builds evidence in memory, uploads to COS, returns metadata.
        Failure does not raise.
        """
        try:
            # Retries keep the originally published key, including its date.
            # ensure_evidence_archive verifies/reuses it without replacing bytes.
            object_key = object_key or (state.get("cos", {}) or {}).get("object_key") or self.cos.build_object_key(
                repo, pr_number, head_sha,
            )
            CosClient.validate_object_key(repo, pr_number, head_sha, object_key)
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
        return {"page": page, "url": comment.get("html_url", "")}

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
        return comments_mod.deploy_requested(
            repo, pr_number, state["head_sha"], state.get("components", []),
            self._get_machine_groups_for_components(remaining),
            deployments=self._enrich_deployments_for_render(display_deployments),
            gate_note=[f"{phase}: `{machine.alias}` / `{machine.node_host}`; components: {targets}; "
                       f"{comments_mod.beijing_now_str()}"],
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
        content = {"snapshot": snapshot["snapshot"], "attachment_records": attachments}
        import json
        key = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        existing = next((r for r in records if r["kind"] == "round" and r["key"] == key), None)
        record = existing or await self.proxy.write_evidence_record(repo, pr_number, "round", key, snapshot)
        result = {"round_id": rid, "sha256": record["sha256"]}
        if type(record.get("comment_id")) is int and isinstance(record.get("url"), str):
            result.update(comment_id=record["comment_id"], url=record["url"])
        return result

    @staticmethod
    def _replace_markdown_section(markdown, heading, content, next_heading):
        """Replace one heading-delimited lifecycle section idempotently."""
        start = markdown.find(heading)
        if start < 0 or markdown.find(heading, start + len(heading)) >= 0:
            return markdown
        content_start = start + len(heading)
        end = markdown.find(next_heading, content_start)
        if end < 0:
            return markdown
        return markdown[:content_start] + "\n\n" + content.rstrip() + "\n\n" + markdown[end:]

    async def _render_evidence_projection(self, repo, pr_number, state, visible):
        renderer = getattr(comments_mod, "render_evidence_section", None)
        if renderer is None:
            return visible
        records = await self.proxy.read_evidence_records(repo, pr_number)
        archives = [r["data"] for r in records if r["kind"] == "round"]
        attachments = [r["data"] for r in records if r["kind"] == "attachment"]
        latest = {r["attachment_id"]: r for r in attachments}
        current_round = round_id_for_state(state)
        current_rows = [r for r in latest.values()
                        if r.get("round_id") == current_round
                        and r.get("head_sha") == state.get("head_sha")]

        def verified_evidence_row(row, head_sha, round_id):
            if not isinstance(row, dict):
                return False
            invitation_upload = (
                row.get("upload_method") == "signed_invite"
                and row.get("upload_link_authorized") is True
                and row.get("actor_verified") is False
                and row.get("github_user_authenticated") is False
                and row.get("cos_content_verified") is True
            )
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
            if row.get("status") != "ready":
                return False
            digest = row.get("sha256")
            source_id = row.get("source_comment_id")
            if source_id is not None and (type(source_id) is not int or source_id <= 0):
                return False
            if (not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                return False
            if invitation_upload and row.get("cos_content_verified") is not True:
                return False
            try:
                from .evidence_store import build_attachment_key
                expected_key = build_attachment_key(
                    repo, pr_number, head_sha, round_id, row.get("attachment_id", ""),
                    row.get("created_at", ""), row.get("filename", ""),
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
            visible = self._replace_markdown_section(
                visible, "### Next action",
                comments_mod._testing_next_action(confirmed_current),
                "### Current Deployment",
            )
        # Presentation flags are deliberately ephemeral. Hidden state has a
        # strict schema and must never learn about browser feature config.
        render_state = copy.deepcopy(state)
        pr = {}
        try:
            from .evidence_auth import readiness
            pr = await self.proxy.get_pr(repo, pr_number)
            store = getattr(self.evidence_service, "store", None)
            status = readiness(self.config, self.evidence_service)
            backend_ready = (status["EVIDENCE_BACKEND_READY"]
                             and callable(getattr(store, "dispatch", None))
                             and pr.get("state") == "open" and not pr.get("merged")
                             and not pr.get("draft")
                             and pr.get("head", {}).get("sha") == state.get("head_sha"))
            browser_upload_ready = backend_ready and status["PUBLIC_HTTPS_INGRESS_READY"]
        except Exception:
            status = {"status": "EVIDENCE_BACKEND_PENDING",
                      "PUBLIC_HTTPS_INGRESS_READY": False}
            backend_ready = False
            browser_upload_ready = False
        render_state.update(
            evidence_enabled=bool(backend_ready),
            browser_upload_ready=bool(browser_upload_ready),
            evidence_readiness=status,
            enabled=bool(backend_ready),
            pr_state=(pr.get("state") if backend_ready else "closed"),
            merged=(bool(pr.get("merged")) if backend_ready else False),
            attachments=current_rows,
        )
        bundles = []
        seen_rounds = set()
        from .evidence_auth import DOWNLOAD_PURPOSE, UPLOAD_PURPOSE, issue_capability

        # Reuse only long-lived COS URLs from the previous trusted lifecycle
        # projection. They are matched to their exact decoded object key and
        # checked for the configured HTTPS bucket host and COS signature fields.
        # Signed URLs remain ephemeral presentation data; they are never saved
        # into hidden state or evidence records.
        store = getattr(self.evidence_service, "store", None)
        cached_cos_urls = {}
        if backend_ready and store is not None:
            for match in re.finditer(r"<(?P<url>https://[^<>\s]+)>", visible):
                candidate = match.group("url")
                try:
                    from urllib.parse import unquote, urlsplit
                    object_key = unquote(urlsplit(candidate).path.lstrip("/"))
                    expiry = store.download_url_expiry(candidate, key=object_key)
                except (AttributeError, TypeError, ValueError):
                    continue
                if expiry is not None and expiry > cached_cos_urls.get(object_key, (0, ""))[0]:
                    cached_cos_urls[object_key] = (expiry, candidate)

        async def stable_cos_url(object_key, filename, *, attachment_id=None):
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

        async def bind_download_capabilities(rows, head_sha, round_id):
            bound = []
            for row in rows:
                item = copy.deepcopy(row)
                attachment_id = item.get("attachment_id")
                valid_row = (
                    verified_evidence_row(item, head_sha, round_id)
                    and item.get("head_sha") == head_sha
                    and item.get("round_id") == round_id
                    and item.get("status") == "ready"
                    and isinstance(item.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")) is not None
                )
                if valid_row:
                    try:
                        from .evidence_store import build_attachment_key
                        expected_key = build_attachment_key(
                            repo, pr_number, head_sha, round_id, attachment_id,
                            item.get("created_at", ""), item.get("filename", ""),
                        )
                    except (ValueError, TypeError, AttributeError):
                        valid_row = False
                    else:
                        valid_row = item.get("object_key") == expected_key
                item["cos_projection_verified"] = bool(valid_row)
                if (valid_row and isinstance(attachment_id, str)
                        and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attachment_id)):
                    item["download_capability"] = issue_capability(
                        self.config, repo, pr_number, head_sha, round_id,
                        purpose=DOWNLOAD_PURPOSE, resource="attachment:" + attachment_id,
                    )
                    item["cos_download_url"] = await stable_cos_url(
                        expected_key, item.get("filename", ""),
                        attachment_id=attachment_id,
                    )
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

        if backend_ready:
            render_state["capability"] = issue_capability(
                self.config, repo, pr_number, state["head_sha"], current_round,
                purpose=DOWNLOAD_PURPOSE, resource="page",
            )
            if browser_upload_ready and state.get("status") == "testing":
                started_at = state.get("round_started_at")
                try:
                    if not isinstance(started_at, str):
                        raise ValueError("round start unavailable")
                    parsed_started_at = datetime.fromisoformat(
                        started_at.replace("Z", "+00:00"),
                    )
                    if parsed_started_at.utcoffset() is None:
                        raise ValueError("round start timezone unavailable")
                    upload_expiry = int((parsed_started_at + timedelta(days=30)).timestamp())
                    if upload_expiry > int(time.time()):
                        render_state["upload_capability"] = issue_capability(
                            self.config, repo, pr_number, state["head_sha"], current_round,
                            purpose=UPLOAD_PURPOSE, expires_at=upload_expiry,
                        )
                except (OverflowError, TypeError, ValueError):
                    logger.warning("Evidence upload capability unavailable %s#%s: invalid round start time",
                                   repo, pr_number)
            render_state["log_capability"] = issue_capability(
                self.config, repo, pr_number, state["head_sha"], current_round,
                purpose=DOWNLOAD_PURPOSE, resource="log:" + current_round,
            )
        if backend_ready:
            current_rows = await bind_download_capabilities(current_rows, state["head_sha"], current_round)
            render_state["attachments"] = current_rows
            await bind_log_url(render_state, state["head_sha"], current_round)
        for archive in reversed(archives):
            if archive["round_id"] in seen_rounds or archive["round_id"] == current_round:
                continue
            seen_rounds.add(archive["round_id"])
            snapshot = copy.deepcopy(archive["snapshot"])
            if backend_ready:
                snapshot["log_capability"] = issue_capability(
                    self.config, repo, pr_number, snapshot["head_sha"], archive["round_id"],
                    purpose=DOWNLOAD_PURPOSE, resource="log:" + archive["round_id"],
                )
            snapshot["deployments"] = self._enrich_deployments_for_render(snapshot.get("deployments", []))
            snapshot_attachments = [r for r in latest.values() if r["round_id"] == archive["round_id"]][-10:]
            snapshot["attachments"] = (await bind_download_capabilities(
                snapshot_attachments, snapshot["head_sha"], archive["round_id"],
            ) if backend_ready else snapshot_attachments)
            if backend_ready:
                await bind_log_url(snapshot, snapshot["head_sha"], archive["round_id"])
            bundles.append({**archive, "snapshot": snapshot})
            if len(bundles) == 3:
                break
        public_base_url = (getattr(self.config, "evidence_public_base_url", "")
                           if browser_upload_ready else "")
        section = renderer(repo, pr_number, render_state, current_rows[-20:] + bundles, archives[-4:],
                           public_base_url=public_base_url)
        if not isinstance(section, str):
            raise GitHubStateProxyError("Evidence renderer must return markdown")
        section += ("\n\nAttachment preview is bounded; complete records and archive chains remain in trusted GitHub "
                    "comments and the capability-protected evidence page.")
        start, end = "<!-- evidence-projection:start -->", "<!-- evidence-projection:end -->"
        if start in visible or end in visible:
            if visible.count(start) != 1 or visible.count(end) != 1 or visible.index(start) >= visible.index(end):
                raise GitHubStateProxyError("Malformed evidence projection")
            before, rest = visible.split(start, 1)
            _, after = rest.split(end, 1)
            visible = before.rstrip() + after
        return visible.rstrip() + "\n\n" + start + "\n" + section + "\n" + end

    async def project_evidence(self, repo: str, pr_number: int) -> None:
        """Watcher-only projection. HTTP handlers never call this writer."""
        if self.evidence_service is None:
            return
        pr = await self.proxy.get_pr(repo, pr_number)
        if pr.get("state") != "open" or pr.get("merged") or pr.get("draft"):
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
        evidence_projection_only: bool = False,
    ) -> bool:
        """Write lifecycle comment with automatic history management.

        If event is provided, it is appended to the visible History section.
        Legacy v1 migration is handled automatically on first meaningful event.
        Preserves existing History events across lifecycle rewrites.
        """
        archive_source = None
        async def write_if_current(visible: str) -> bool:
            async with self._evidence_lock:
                if archive_source is not None:
                    # Final recapture includes any attachment committed while
                    # the initial archive/projection was being assembled.
                    state["round_archive"] = await self._archive_round_unlocked(
                        repo, pr_number, archive_source, "Lifecycle snapshot replaced",
                    )
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
                if expected_terminal_state is not None:
                    pr = await self.proxy.get_pr(repo, pr_number)
                    fresh = await self.proxy.read_hidden_state(repo, pr_number)
                    if (fresh != expected_terminal_state
                            or pr.get("head", {}).get("sha") != state.get("head_sha")
                            or pr.get("state") != "open" or pr.get("merged") or pr.get("draft")):
                        return False
                current_pr = await self.proxy.get_pr(repo, pr_number)
                if (current_pr.get("state") != "open" or current_pr.get("merged")
                        or current_pr.get("draft")
                        or current_pr.get("head", {}).get("sha") != state.get("head_sha")):
                    raise GitHubStateProxyError("Lifecycle PR context changed before write")
                current_comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
                if ((comment is None) != (current_comment is None)
                        or (comment is not None and (
                            current_comment.get("id") != comment.get("id")
                            or current_comment.get("body") != comment.get("body")))):
                    raise GitHubStateProxyError("Lifecycle History changed before write")
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
                state["round_archive"] = await self._archive_round(repo, pr_number, old, "Lifecycle snapshot replaced")
                if head_changed or (lost_fields and not round_changed):
                    state["round_id"] = make_round_id(state["head_sha"], state.get("command", {}).get("comment_id", 0))
                    state["round_started_at"] = utc_now()
            if "round_started_at" not in state and old.get("round_started_at") and not head_changed and not round_changed:
                state["round_started_at"] = old["round_started_at"]
            if evidence_projection_only:
                projected = await self._render_evidence_projection(repo, pr_number, state, new_visible_markdown)
                if projected == new_visible_markdown:
                    return True
                return await write_if_current(projected)
        if comment is None:
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
            # Only wrap unmarked legacy events when appending. COS-only refresh
            # retains the original unmarked History section verbatim below.
            old_history = re.search(r"(?ms)^### History[ \t]*\n.*?(?=^### |\Z)", existing_visible)
            if old_history:
                history_block = (VISIBLE_HISTORY_START_MARKER
                                 + old_history.group(0).partition("\n")[2]
                                 + VISIBLE_HISTORY_END_MARKER)
        incoming = list(events) if events is not None else ([event] if event is not None else [])
        if incoming:
            new_block = _build_history_block(incoming)
            if history_block:
                history_block = (new_block[:-len(VISIBLE_HISTORY_END_MARKER)]
                                 + history_block[len(VISIBLE_HISTORY_START_MARKER):])
            else:
                history_block = new_block
        final_visible = new_visible_markdown.rstrip()
        if history_block:
            final_visible += "\n\n### History\n\n" + history_block + "\n"
        elif not incoming:
            # Older terminal comments can have an unmarked History section.
            old_history = re.search(r"(?ms)^### History[ \t]*\n.*?(?=^### |\Z)", existing_visible)
            if old_history:
                final_visible += "\n\n" + old_history.group(0)

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
        state["round_started_at"] = utc_now()
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
            state["round_archive"] = archived
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

    async def _post_command_not_ready(
        self, repo: str, pr_number: int, current_status: str, next_action_text: str,
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
        try:
            await self.proxy.post_issue_comment(repo, pr_number, body)
        except GitHubError as e:
            logger.warning("post command not ready comment %s#%s: %s", repo, pr_number, e)

    async def _post_error(
        self, repo: str, pr_number: int, message: str,
    ) -> None:
        """Post an error reply comment."""
        body = "\n".join([
            comments_mod.BOT_MARKER,
            "### Deploy Approval \u2014 Error",
            "",
            message,
        ])
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

        if pr_state != "open" or pr_merged or pr_data.get("draft"):
            return
        if not re.fullmatch(r"[0-9a-f]{40}", current_head):
            return

        if state is not None:
            cmd = state.get("command", {})
            cmd_phase = cmd.get("phase", "")
            if cmd_phase == "executing":
                await self._handle_uncertain_if_needed(repo, pr_number, state)
                return
            if cmd_phase == "uncertain":
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
                    await self.proxy.project_status_label(repo, pr_number, "testing")
                except Exception as e:
                    logger.warning(
                        "reconcile label projection %s#%s: %s", repo, pr_number, e,
                    )
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
        try:
            comments = await self.github.get_issue_comments(repo, pr_number)
            if not isinstance(comments, list):
                comments = []

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
            desired_status = "review-required"
            review_evidence_data: dict = {}
            build_infos = []

        state_was_none = state is None
        if state is None:
            state = self._init_hidden_state(
                head_sha=current_head,
                status=desired_status,
                review_evidence=review_evidence_data,
                components=[],
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
            state["components"] = []
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
        else:
            try:
                if desired_status == "reviewing":
                    markdown = comments_mod.reviewing(repo, pr_number, current_head)
                elif desired_status == "review-required":
                    markdown = comments_mod.review_required(repo, pr_number, current_head)
                else:
                    markdown = comments_mod.deploy_ready(repo, pr_number, current_head, build_infos, review_evidence=review_evidence_data)
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
                await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                )
                await self.proxy.project_status_label(repo, pr_number, desired_status)
            except Exception as e:
                logger.warning(
                    "reconcile lifecycle %s#%s: %s", repo, pr_number, e,
                )

    async def _handle_uncertain_if_needed(
        self, repo: str, pr_number: int, state: dict,
    ) -> None:
        """On startup, if hidden command.phase is executing, mark as uncertain."""
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


        markdown = comments_mod.uncertain_comment(
            repo, pr_number, state.get("head_sha", ""),
        )
        event = {
            "event": "Deploy outcome uncertain",
            "result": "uncertain",
            "timestamp": comments_mod.beijing_now_str(),
        }
        if machine_alias:
            event["machine"] = machine_alias
        await self._write_lifecycle_with_history(
            repo, pr_number, state, markdown, event=event,
        )
        await self.proxy.project_status_label(
            repo, pr_number, state.get("status", "review-required"),
        )

    async def _recover_uncertain_deployments(self, old_state: dict, refreshed_state: dict) -> bool:
        """Observe possibly attempted components; never repeat their deploy POSTs."""
        args = old_state.get("command", {}).get("args", {})
        machine_alias = args.get("machine", "")
        machine = self.policy.get_machine(machine_alias)
        if machine is None:
            return False
        old_components = old_state.get("components", [])
        deployed = _component_id_set(old_state.get("deployments", []))
        pending = args.get("pending_component_ids")
        if pending is None:
            # Older executing states have no per-component POST provenance.
            # Conservatively observe every remaining component for this machine.
            pending = self._get_component_ids_for_machine(machine_alias, old_components)
        if (not isinstance(pending, list) or not pending
                or any(not isinstance(cid, str) or not cid for cid in pending)
                or not set(pending).issubset({c.get("component_id") for c in old_components})):
            return False
        fresh_by_key = {
            self._component_semantic_key(c): c for c in refreshed_state.get("components", [])
        }
        recovered = []
        try:
            core = await self._core_for_node(machine.node_id)
            for old in old_components:
                if old["component_id"] not in pending or old["component_id"] in deployed:
                    continue
                component = fresh_by_key.get(self._component_semantic_key(old))
                if component is None:
                    # A genuinely different artifact is not a retry of this POST.
                    continue
                if component["component_id"] not in self._get_component_ids_for_machine(machine_alias, [component]):
                    return False
                runtime_id = str(old.get("runtime_id") or "")
                if component.get("target") == "core":
                    verified, _ = await self._verify_core_update(
                        core, component["component_id"], old["image_ref"],
                    )
                    runtime_id = "core"
                else:
                    if not runtime_id:
                        preflight = await self._preflight_running_images(core, [old])
                        runtime_id = preflight[0]["runtime_id"]
                    verified, _ = await self._verify_deployed_runtime(
                        core, machine.node_id, component["component_id"], runtime_id, old["image_ref"],
                    )
                if not verified:
                    return False
                recovered.append((component, old["image_ref"], runtime_id))
        except Exception as e:
            logger.warning("uncertain deployment observation unavailable: %s", e)
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
                "component_ids": [component["component_id"]],
                "phase": "deployed",
            })
        all_ids = {c["component_id"] for c in refreshed_state.get("components", [])}
        if all_ids and all_ids == _component_id_set(refreshed_state.get("deployments", [])):
            refreshed_state["status"] = "testing"
        return True

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
            await self._write_lifecycle_with_history(
                repo, pr_number, state, markdown, event=event,
            )
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
            state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "uncertain",
                "args": dict(cmd.get("args", {}) or {}),
            }
            state["last_processed_comment_id"] = max(old_cursor, comment_id)
            markdown = comments_mod.uncertain_comment(
                repo, pr_number, current_head,
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
            # Genuinely different build/evidence — reset to fresh snapshot.
            fresh_reset_components: list[dict] = []
            for component in fresh_components:
                fresh_reset_components.append(
                    self._fresh_component_runtime_binding(component)
                )
            updated_state = dict(state)
            updated_state["review_evidence"] = review_evidence_data
            updated_state["status"] = "deploy-requested"
            updated_state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "completed",
                "args": dict(cmd.get("args", {}) or {}),
            }
            updated_state["last_processed_comment_id"] = max(old_cursor, comment_id)
            updated_state["components"] = fresh_reset_components
            updated_state["deployments"] = []
            updated_state["approve_attempts"] = []
            updated_state["approve_attempts_total"] = 0
            updated_state["approve_attempts_truncated"] = False
            updated_state["case_results"] = {}
            updated_state["test_result"] = ""
            updated_state["cos"] = {"object_key": "", "sha256": "", "size": 0}
        else:
            # Same evidence AND same semantic keys — apply migration rules.
            deployed_component_ids: set[str] = set()
            for dep in state.get("deployments", []):
                if isinstance(dep, dict) and dep.get("phase") == "deployed":
                    for cid in dep.get("component_ids", []):
                        if isinstance(cid, str) and cid:
                            deployed_component_ids.add(cid)

            # Fail-closed: deployed components must have non-empty runtime_id.
            # If a deployed component is missing runtime_id, fall through to the
            # reset branch instead of proceeding with migration.
            _migration_ok = True
            for old_comp in old_components:
                if not isinstance(old_comp, dict):
                    continue
                old_cid = old_comp.get("component_id", "")
                if (
                    old_cid in deployed_component_ids
                    and old_comp.get("target") != "core"
                ):
                    # Non-core runtime ids are dynamic and must be present
                    # before preserving a deployed binding.  Core is static
                    # and is normalized to "core" below.
                    runtime_id = old_comp.get("runtime_id", "")
                    if not isinstance(runtime_id, str) or not runtime_id:
                        # Missing runtime_id for deployed component — reset to fresh.
                        _migration_ok = False
                        break

            if not _migration_ok:
                snapshot_reset = True
                # Fall through to the reset branch below.
                fresh_reset_components: list[dict] = []
                for component in fresh_components:
                    fresh_reset_components.append(
                        self._fresh_component_runtime_binding(component)
                    )
                updated_state = dict(state)
                updated_state["review_evidence"] = review_evidence_data
                updated_state["status"] = "deploy-requested"
                updated_state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "completed",
                    "args": dict(cmd.get("args", {}) or {}),
                }
                updated_state["last_processed_comment_id"] = max(old_cursor, comment_id)
                updated_state["components"] = fresh_reset_components
                updated_state["deployments"] = []
                updated_state["approve_attempts"] = []
                updated_state["approve_attempts_total"] = 0
                updated_state["approve_attempts_truncated"] = False
                updated_state["case_results"] = {}
                updated_state["test_result"] = ""
                updated_state["cos"] = {"object_key": "", "sha256": "", "size": 0}
            else:
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
                        # Rule B: preserve historical artifact
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
                            migrated_component["runtime_id"] = old_comp.get(
                                "runtime_id", ""
                            )
                        migrated.append(migrated_component)
                    else:
                        # Rule A: migrate to tag, reuse old component_id
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
                updated_state["status"] = "deploy-requested"
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
        if not snapshot_reset and not await self._recover_uncertain_deployments(state, updated_state):
            logger.warning("uncertain deployment remains pending; refusing repeated POST %s#%s", repo, pr_number)
            return "uncertain"
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
        markdown = comments_mod.deploy_requested(
            repo, pr_number, current_head,
            state.get("components", []),
            self._get_machine_groups_for_components([
                c for c in state.get("components", [])
                if c.get("component_id", "") not in _component_id_set(state.get("deployments", []))
            ]),
            gate_note=gate_note,
            deployments=self._enrich_deployments_for_render(state.get("deployments", [])),
        )
        if state["status"] == "testing":
            required_component_ids = {
                c.get("component_id") for c in state.get("components", [])
                if isinstance(c, dict) and c.get("component_id")
            }
            markdown = comments_mod.testing(
                repo, pr_number, current_head, case_result="",
                deployments=self._enrich_deployments_for_render(state.get("deployments", [])),
                components=state.get("components", []),
                deployments_complete=(bool(required_component_ids)
                                      and required_component_ids.issubset(
                                          _component_id_set(state.get("deployments", [])))),
            )
        event = {
            "event": "Restart recovery: evidence refreshed",
            "timestamp": comments_mod.beijing_now_str(),
        }
        await self._write_lifecycle_with_history(
            repo, pr_number, state, markdown, event=event,
        )
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
            return True

        if cmd.kind == "request_deploy":
            return await self.handle_request_deploy(
                repo, pr_number, comment_id,
            )

        if cmd.kind == "approve_deploy":
            machine_alias = cmd.machine_alias
            if not machine_alias:
                await self._post_error(
                    repo, pr_number,
                    "`/approve_deploy` requires `machine=<alias-or-ip>`.",
                )
                return True
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
                )
                return True
            return await self.handle_record_test(
                repo, pr_number, comment_id,
                result, summary, author_login, author_id,
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
                    )
                    return True
                raise
            return True

        logger.warning("on_command: unknown kind %s", cmd.kind)
        return True


# Backward-compatible aliases
DeployServiceError = DeployControllerError
DeploymentService = DeployController
