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
import os
import inspect

import asyncio
import logging
import time
from typing import Any

from . import comments as comments_mod
from .agent_core_client import (
    AgentCoreClient,
    AgentCoreDeployOutcomeUncertain,
    AgentCoreError,
)
from .case_runner import CaseRunner
from .config import Config
from .cos_client import CosClient
from .evidence_builder import EvidenceBuilder, _case_id_for_target
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
    _MAX_COMMENT_BODY_BYTES,
    HISTORY_ARCHIVE_MARKER_PREFIX,
    _next_history_archive_page,
    _build_archive_body,
    _parse_visible_history,
    VISIBLE_HISTORY_START_MARKER,
    VISIBLE_HISTORY_END_MARKER,
    HIDDEN_STATE_MARKER,
)
from .models import BuildInfo, new_id, utc_now
from .policy import Policy, PolicyError
from .review_comment_parser import (
    extract_review_evidence,
    ReviewCommentEvidence,
    _parse_build_table,
    _parse_build_images,
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
    return target in ("perception", "actucore", "driver")


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


_CANONICAL_VARIANTS = {"5.11", "6.1"}
_LEGACY_VARIANTS = {
    "jetson-jp5.11": "5.11",
    "jetson-jp6.1": "6.1",
}

MAX_RECENT_APPROVE_ATTEMPTS = 4


def _normalize_variant(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in _CANONICAL_VARIANTS:
        return raw
    if raw in _LEGACY_VARIANTS:
        return _LEGACY_VARIANTS[raw]
    raise DeployControllerError(f"unsupported variant: {raw!r}")


def _is_deployable_build(repo: str, build) -> bool:
    target = str(build.target or "").strip()
    if not _is_supported_target(target):
        return False
    if not build.success:
        return False
    if not build.image_tag:
        return False
    # Repo-aware deployability matrix — exact known identities only
    repo_lower = (repo or "").lower()
    if repo_lower in ("4paradigm/phanthymotus-driver", "haohao-end/phanthymotus-driver"):
        # Driver family: only driver with non-empty driver_path is deployable
        if target != "driver":
            return False
        driver_path = getattr(build, "driver_path", "") or ""
        if not driver_path.strip():
            return False
        return True
    if repo_lower in ("4paradigm/phanthymotus", "haohao-end/phanthymotus"):
        # Core family: perception and actucore are deployable; driver is NOT
        if target != "driver":
            return True
        return False
    # Unknown repository: nothing is deployable
    return False

def _trim_approve_attempts(
    state: dict,
    attempt: dict,
) -> tuple[list[dict], int, bool]:
    attempts = list(state.get("approve_attempts", []))
    total = state.get("approve_attempts_total")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        total = len(attempts)
    total += 1
    truncated = False
    attempts.append(attempt)
    if len(attempts) > MAX_RECENT_APPROVE_ATTEMPTS:
        truncated = True
        attempts = attempts[-MAX_RECENT_APPROVE_ATTEMPTS:]
    return attempts, total, truncated


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
    No SQLite, no DeploymentStore, no in-memory state.
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
        core = AgentCoreClient(
            self.config,
            base_url,
            node_host=machine.node_host,
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

    @staticmethod
    def _empty_cos() -> dict[str, Any]:
        return {"object_key": "", "sha256": "", "size": 0}

    @staticmethod
    def _empty_approve_attempts() -> dict[str, Any]:
        return {
            "approve_attempts": [],
            "approve_attempts_total": 0,
            "approve_attempts_truncated": False,
        }

    @staticmethod
    def _review_evidence_snapshot(
        evidence: "ReviewCommentEvidence",
        resolved_head: str,
    ) -> dict:
        """Return the canonical 9-field review_evidence snapshot."""
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
            if component_id in deployed_component_ids:
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
            item.pop("runtime_id", None)
            if component_id in deployed_component_ids:
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
        snapshot: list[dict] = []
        seen_semantic_keys: set[tuple] = set()
        for build in builds or []:
            if not build.success or not build.deployable:
                continue
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
                validated_tag,
            )
            if skey in seen_semantic_keys:
                return None
            seen_semantic_keys.add(skey)
            snapshot.append({
                "component_id": component_id,
                "target": build.target,
                "driver_path": build.driver_path,
                "variant": build.variant,
                "review_image_tag": validated_tag,
                "image_ref": validated_tag,
                "resolved_platform": resolved_platform,
            })
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
    ) -> bool:
        fresh_state = await self.proxy.read_hidden_state(repo, pr_number)
        if fresh_state is None:
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

        metadata = cos_metadata or {}
        object_key = str(metadata.get("object_key", "") or "")
        sha256 = str(metadata.get("sha256", "") or "")
        size = int(metadata.get("size", 0) or 0)
        if not object_key:
            return False

        fresh_state["cos"] = {
            "object_key": object_key,
            "sha256": sha256,
            "size": size,
        }
        if markdown:
            await self.proxy.write_hidden_state(repo, pr_number, markdown, fresh_state)
        return True

    @staticmethod
    def _deployment_component_ids(state: dict) -> set[str]:
        deployed_component_ids: set[str] = set()
        for dep in state.get("deployments", []):
            if dep.get("phase") != "deployed":
                continue
            for cid in dep.get("component_ids", []):
                if isinstance(cid, str) and cid:
                    deployed_component_ids.add(cid)
        return deployed_component_ids

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

            pr_data = await self.proxy.get_pr(repo, pr_number)
            pr_state = pr_data.get("state", "")
            pr_merged = pr_data.get("merged", False)
            pr_head = pr_data.get("head", {}).get("sha", "")
            if state is None:
                await self._post_error(
                    repo, pr_number,
                    "No deploy approval lifecycle found. Wait for review lifecycle reconciliation first.",
                )
                return True
            _current_status = state.get("status", "")
            if _current_status == "deploy-ready":
                pass  # normal path continues below
            elif _current_status in ("deploy-requested", "testing", "succeeded", "failed"):
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
            if pr_head != state.get("head_sha", ""):
                # HEAD drift: zero deploy, clear old validation snapshot, reset to review-required
                old_head = state.get("head_sha", "")
                state["status"] = "review-required"
                state["head_sha"] = pr_head
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
                    "kind": "request_deploy",
                    "phase": "completed",
                    "args": {},
                }
                state["last_processed_comment_id"] = comment_id
                markdown = comments_mod.superseded_comment(
                    repo, pr_number, old_head, pr_head,
                )
                event = {
                    "event": "HEAD drift detected",
                    "lifecycle": f"`{state.get('status', 'review-required')}` \u2192 `review-required`",
                    "timestamp": comments_mod.beijing_now_str(),
                }
                await self._write_lifecycle_with_history(
                    repo, pr_number, state, markdown, event=event,
                )
                await self.proxy.project_status_label(repo, pr_number, "review-required")
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

            if pr_state != "open" or pr_merged:
                await self._post_error(
                    repo, pr_number,
                    "PR is not open. Deploy only when PR is open and unmerged.",
                )
                return True

            # Get build evidence from PR comments
            try:
                comments = await self.github.get_issue_comments(repo, pr_number)
                if not isinstance(comments, list):
                    comments = []
                evidence = extract_review_evidence(
                    comments,
                    self.config.review_comment_author_id,
                    self.config.review_comment_author_login,
                )
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

            # Convert evidence builds to BuildInfo
            builds = []
            for eb in evidence.builds:
                bi = BuildInfo(
                    idx=0,
                    target=eb.target,
                    driver_path=eb.driver_path,
                    variant=eb.variant,
                    success=eb.success,
                    image_tag=eb.image_tag,
                    deployable=_is_deployable_build(
                        repo,
                        type("_build", (), {
                            "target": eb.target,
                            "success": eb.success,
                            "image_tag": eb.image_tag,
                            "driver_path": eb.driver_path,
                        })()
                    ),
                )
                builds.append(bi)

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
                repo, pr_head, evidence,
            )
            if resolved_head is None:
                await self._post_error(
                    repo, pr_number,
                    "Review evidence commit could not be resolved to the current PR HEAD.",
                )
                return True

            # Build review_evidence snapshot with all 9 canonical fields
            review_evidence_data = {
                "build_comment_id": evidence.build_comment_id,
                "build_comment_updated_at": evidence.build_comment_updated_at,
                "commit_prefix": evidence.commit_prefix,
                "resolved_head_sha": resolved_head,
                "test_comment_id": evidence.test_comment_id,
                "test_comment_updated_at": evidence.test_comment_updated_at,
                "code_review_comment_id": evidence.code_review_comment_id,
                "code_review_comment_updated_at": evidence.code_review_comment_updated_at,
                "review_author_id": evidence.review_author_id,
            }

            # Determine compatible machine groups
            machine_groups = self._get_machine_groups_for_components(components)

            # Build hidden state
            state = self._init_hidden_state(
                head_sha=pr_head,
                status="deploy-requested",
                review_evidence=review_evidence_data,
                components=components,
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
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if state is None:
                await self._post_error(
                    repo, pr_number,
                    "No active deployment found. Use `/request_deploy` first.",
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
            pr_data = await self.proxy.get_pr(repo, pr_number)
            pr_state = pr_data.get("state", "")
            pr_merged = pr_data.get("merged", False)
            pr_head = pr_data.get("head", {}).get("sha", "")

            if pr_state != "open" or pr_merged:
                await self._post_error(
                    repo, pr_number,
                    "PR is not open. Deploy only when PR is open and unmerged.",
                )
                return True

            # Check HEAD drift
            if pr_head and pr_head != state.get("head_sha", ""):
                await self._supersede_head_drift(
                    repo, pr_number, state, pr_head, comment_id,
                )
                return True

            if state.get("command", {}).get("phase") == "uncertain":
                refresh_result = await self._refresh_uncertain_state(
                    repo, pr_number, state,
                )
                if refresh_result == "deploy-requested":
                    refreshed_state = await self.proxy.read_hidden_state(repo, pr_number)
                    if refreshed_state is None:
                        return True
                    state = refreshed_state
                else:
                    return True

            # Resolve machine selector (alias or IPv4) to canonical MachineInfo
            try:
                machine = Policy.resolve_machine_selector(machine_alias, self.policy.machines)
            except PolicyError as e:
                await self._post_error(repo, pr_number, str(e))
                return True

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


            required_remaining_ids = {
                c.get("component_id", "") for c in undeployed_components
            }
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
            # machine_alias added conditionally below
                }
                state["last_processed_comment_id"] = comment_id
                markdown = comments_mod.deploy_requested(
                    repo, pr_number, pr_head,
                    undeployed_components,
                    self._get_machine_groups_for_components(undeployed_components),
                    gate_note=[
                        f"Machine `{machine_alias}` does not cover any remaining component.",
                        "ZERO deploy POST.",
                        "Send a NEW `/approve_deploy machine=<alias>` for a compatible machine.",
                    ],
                    deployments=[],
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


            node_id = machine.node_id
            core = await self._core_for_node(node_id)

            # Preflight: read running_image for every SELECTED component before
            # any deploy POST. running_image is evidence, not a block — Agent Core
            # handles old-container replacement via its own deploy contract.
            preflight = await self._preflight_running_images(core, selected_components)
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
            if fresh_state != "open" or fresh_merged or fresh_head != state.get("head_sha", ""):
                await self._invalidate_review_required(
                    repo,
                    pr_number,
                    state,
                    fresh_head or state.get("head_sha", ""),
                    comment_id,
                    "PR changed after clean gate.",
                )
                return True

            # Fresh exact approval comment re-validation
            validated_comment = await self._revalidate_approve_comment(
                repo, comment_id, actor_id, machine_alias,
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
                markdown = (
                    "Approval comment was deleted, edited, or changed.\n"
                    "A NEW approval comment is required."
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

            if final_state != "open" or final_merged:
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
                    "Hidden state changed after clean gate.",
                )
                return True


            # Fresh review evidence revalidation before unsafe POST
            if not await self._fresh_review_evidence_matches_state(
                repo, pr_number, final_head,
                state.get("review_evidence", {}),
            ):
                await self._invalidate_review_required(
                    repo, pr_number, state, final_head, comment_id,
                    "Review evidence changed after approval.",
                )
                return True

            # Write executing state only after ALL gates pass.
            state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "executing",
                "args": {"machine": machine_alias, "actor": actor},
            }
            await self.proxy.write_hidden_state(
                repo, pr_number, "Deploying...", state
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

            # Sequential per-component deploy; POST success followed by
            # post-deploy runtime verification before recording as deployed.
            for comp in selected_components:
                image_ref = comp["image_ref"]
                runtime_id = str(comp.get("runtime_id") or "")
                if not runtime_id:
                    deploy_error = f"missing runtime id for {comp.get('target', '')}"
                    break
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

                # Post-deploy verification: confirm target image is actually running
                verified, health_evidence = await self._verify_deployed_runtime(
                    core, node_id, comp.get("component_id", ""), runtime_id, image_ref,
                )
                approve_attempt["health"].append(health_evidence)
                if not verified:
                    deploy_outcome_uncertain = True
                    deploy_error = (
                        f"post-deploy verify timeout for {comp.get('target', '')!r}: "
                        f"target image not observed running"
                    )
                    break

                # Deploy success verified: record deployment
                new_deployments.append({
                    "machine": machine_alias,
                    "component_ids": [comp["component_id"]],
                    "phase": "deployed",
                })

            if deploy_outcome_uncertain:
                state["deployments"] = list(existing_deployments) + list(new_deployments)
                state["status"] = "deploy-requested"
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "uncertain",
                    "args": {"machine": machine_alias, "actor": actor},
                }
                state["last_processed_comment_id"] = comment_id
                approve_attempt["outcome"] = "uncertain"
                self._record_approve_attempt(state, approve_attempt)
                markdown = comments_mod.deploy_requested(
                    repo,
                    pr_number,
                    pr_head,
                    state.get("components", []),
                    self._get_machine_groups_for_components(state.get("components", [])),
                    gate_note=[
                        "### Restart Recovery",
                        "",
                        "A deploy POST was attempted but its outcome is uncertain.",
                        f"Send a NEW `/approve_deploy machine={machine_alias}`.",
                        "No further deploy POSTs are allowed in this cycle.",
                    ],
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
                    await self._rebind_terminal_cos_if_current(
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
                    await self._rebind_terminal_cos_if_current(
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

            # Check full coverage: all components deployed -> testing; else -> stay deploy-requested
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
                    remaining,
                    self._get_machine_groups_for_components(remaining),
                    gate_note=[
                        "### Partial coverage completed",
                        "",
                        f"Machine `{machine_alias}` deployed its compatible components.",
                        f"Remaining components need additional machine approval.",
                        "Send a NEW `/approve_deploy machine=<alias>`.",
                    ],
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
            )
            event = {
                "event": "All components deployed",
                "lifecycle": "`deploy-requested` \u2192 `testing`",
                "timestamp": comments_mod.beijing_now_str(),
            }
            await self._write_lifecycle_with_history(
                repo, pr_number, state, testing_markdown, event=event,
            )
            # Project status label "testing" SECOND
            await self.proxy.project_status_label(repo, pr_number, "testing")

            # NOW run advisory Case only after durable testing state exists
            case_results = await self._run_automated_case(
                repo, pr_number, pr_head, components,
                state.get("deployments", []),
            )

            # Case is advisory only; always re-read fresh state before merging
            fresh = await self.proxy.read_hidden_state(repo, pr_number)
            fresh_command = (
                fresh.get("command", {})
                if isinstance(fresh, dict)
                else {}
            )
            if not isinstance(fresh_command, dict):
                fresh_command = {}
            fresh_args = fresh_command.get("args", {})
            if not isinstance(fresh_args, dict):
                fresh_args = {}
            fresh_ok = (
                isinstance(fresh, dict)
                and fresh.get("head_sha") == pr_head
                and fresh.get("status") == "testing"
                and fresh_command.get("comment_id") == comment_id
                and fresh_command.get("kind") == "approve_deploy"
                and fresh_command.get("phase") == "completed"
                and fresh_args.get("machine") == machine_alias
                and fresh_args.get("actor") == actor
            )
            if case_results and fresh_ok:
                fresh["case_results"] = fresh.get("case_results", {})
                fresh["case_results"].update(case_results)
                case_result_str = ", ".join(
                    f"{k}={v}" for k, v in case_results.items()
                )
                case_markdown = comments_mod.testing(
                    repo, pr_number, pr_head, case_result=case_result_str,
                )
                await self.proxy.write_hidden_state(repo, pr_number, case_markdown, fresh)
            elif case_results:
                logger.warning(
                    "CASE_RESULT_DROPPED repo=%s pr=%s: fresh state mismatch, "
                    "not overwriting newer GitHub state",
                    repo, pr_number,
                )

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

    async def handle_record_test(
        self, repo: str, pr_number: int, comment_id: int,
        result: str, summary: str, actor: str, actor_id: str,
    ) -> bool:
        try:
            state = await self.proxy.read_hidden_state(repo, pr_number)
            if state is None:
                await self._post_error(
                    repo, pr_number,
                    "No active deployment found. Use `/request_deploy` first.",
                )
                return True

            _current_status = state.get("status", "")
            if _current_status == "testing":
                pass  # normal path continues below
            elif _current_status in ("succeeded", "failed"):
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
            pr_data = await self.proxy.get_pr(repo, pr_number)
            pr_state = pr_data.get("state", "")
            pr_merged = pr_data.get("merged", False)
            pr_head = pr_data.get("head", {}).get("sha", "")
            if pr_state != "open" or pr_merged:
                await self._post_error(
                    repo, pr_number,
                    "PR is not open. Record test only when PR is open and unmerged.",
                )
                return True

            # Check HEAD drift
            if pr_head and pr_head != state.get("head_sha", ""):
                await self._supersede_head_drift(
                    repo, pr_number, state, pr_head, comment_id,
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
            if all_component_ids and not all_component_ids.issubset(deployed_component_ids):
                await self._post_error(
                    repo, pr_number,
                    "Not all components are deployed yet. "
                    "Complete all machine approvals before recording test result.",
                )
                return True

            # Authorize actor: collaborator OR owner of any deployed machine
            await self._check_record_test_permissions(
                repo, actor, state,
            )

            # Write terminal state FIRST, before COS upload
            state["command"] = {
                "comment_id": comment_id,
                "kind": "record_test",
                "phase": "completed",
                "args": {"result": result, "summary": summary, "actor": actor},
            }
            state["last_processed_comment_id"] = comment_id

            if result == "pass":
                state["status"] = "succeeded"
                state["test_result"] = "pass"
            else:
                state["status"] = "failed"
                state["test_result"] = "fail"

            # Write a REAL terminal visible lifecycle comment immediately
            if result == "pass":
                enriched = self._enrich_deployments_for_render(
                    state.get("deployments", []),
                )
                terminal_markdown = comments_mod.succeeded_comment(
                    repo, pr_number, pr_head,
                    deployments=enriched,
                    components=state.get("components", []),
                )
                event = {
                    "event": "Test recorded",
                    "lifecycle": "`testing` \u2192 `succeeded`",
                    "result": "pass",
                    "timestamp": comments_mod.beijing_now_str(),
                }
            else:
                terminal_markdown = comments_mod.failed_comment(repo, pr_number, pr_head)
                event = {
                    "event": "Test recorded",
                    "lifecycle": "`testing` \u2192 `failed`",
                    "result": "fail",
                    "timestamp": comments_mod.beijing_now_str(),
                }

            # Terminal state is written to GitHub before COS upload
            await self._write_lifecycle_with_history(
                repo, pr_number, state, terminal_markdown, event=event,
            )
            await self.proxy.project_status_label(repo, pr_number, state["status"])

            # One-shot runtime log snapshot for record_test terminal evidence
            runtime_logs = await self._snapshot_terminal_runtime_logs(
                state.get("components", []),
                state.get("deployments", []),
            )

            # Upload COS evidence (failure does not roll back terminal state)
            state["cos"] = self._empty_cos()
            try:
                cos_metadata = await self._upload_evidence(
                    repo, pr_number, pr_head, state, result, summary,
                    deploy_error="",
                    runtime_logs=runtime_logs,
                )
            except Exception as e:
                logger.warning(
                    "record_test evidence upload failed %s#%s: %s",
                    repo, pr_number, e,
                )
                return True

            if cos_metadata.get("object_key"):
                # Step 1: rebind terminal state with COS metadata FIRST
                markdown_no_url = comments_mod.succeeded_comment(
                    repo, pr_number, pr_head,
                    cos_object_key=str(cos_metadata.get("object_key", "") or ""),
                    cos_bundle_sha256=str(cos_metadata.get("sha256", "") or ""),
                    cos_bundle_size=int(cos_metadata.get("size", 0) or 0),
                ) if result == "pass" else comments_mod.failed_comment(
                    repo, pr_number, pr_head,
                    cos_object_key=str(cos_metadata.get("object_key", "") or ""),
                    cos_bundle_sha256=str(cos_metadata.get("sha256", "") or ""),
                    cos_bundle_size=int(cos_metadata.get("size", 0) or 0),
                )
                await self._rebind_terminal_cos_if_current(
                    repo,
                    pr_number,
                    expected_head=pr_head,
                    expected_terminal_status=state["status"],
                    expected_comment_id=comment_id,
                    expected_command_kind="record_test",
                    expected_test_result=result,
                    cos_metadata=cos_metadata,
                    markdown=markdown_no_url,
                )
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
                markdown_with_url = comments_mod.succeeded_comment(
                    repo, pr_number, pr_head,
                    cos_object_key=str(cos_metadata.get("object_key", "") or ""),
                    cos_bundle_sha256=str(cos_metadata.get("sha256", "") or ""),
                    cos_bundle_size=int(cos_metadata.get("size", 0) or 0),
                    cos_download_url=cos_download_url,
                ) if result == "pass" else comments_mod.failed_comment(
                    repo, pr_number, pr_head,
                    cos_object_key=str(cos_metadata.get("object_key", "") or ""),
                    cos_bundle_sha256=str(cos_metadata.get("sha256", "") or ""),
                    cos_bundle_size=int(cos_metadata.get("size", 0) or 0),
                    cos_download_url=cos_download_url,
                )
                await self._rebind_terminal_cos_if_current(
                    repo,
                    pr_number,
                    expected_head=pr_head,
                    expected_terminal_status=state["status"],
                    expected_comment_id=comment_id,
                    expected_command_kind="record_test",
                    expected_test_result=result,
                    cos_metadata=cos_metadata,
                    markdown=markdown_with_url,
                )
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
            if comp.get("target") != "driver" and machine.variants:
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
                if comp.get("target") != "driver" and m.variants:
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
        """
        drivers = await core.list_drivers()
        if not isinstance(drivers, list):
            raise DeployControllerError("Agent Core list_drivers returned an invalid payload")

        cached_statuses: dict[str, dict] = {}
        preflight: list[dict] = []
        seen_runtime_ids: dict[str, str] = {}  # runtime_id -> component_id
        for component in components:
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

            # Resolve core client
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

            # Select case for this component
            runner = self._get_case_runner()
            case_id = runner.select_case(target, variant, node_id)
            if case_id is None:
                results[cid] = "n/a"
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
    ) -> dict:
        """Upload COS evidence bundle. Returns metadata dict.

        Builds evidence in memory, uploads to COS, returns metadata.
        Failure does not raise.
        """
        try:
            object_key = self.cos.build_object_key(
                repo, pr_number, head_sha,
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
                )
            except Exception as e:
                logger.warning(
                    "evidence build %s#%s: %s", repo, pr_number, e,
                )

            if archive_bytes:
                ok = await self.cos.upload_evidence_archive(
                    object_key, archive_bytes,
                )
                if ok:
                    return {
                        "object_key": object_key,
                        "sha256": sha256,
                        "size": size,
                    }
            return {"object_key": "", "sha256": "", "size": 0}
        except Exception as e:
            logger.warning(
                "evidence upload %s#%s: %s", repo, pr_number, e,
            )
            return {"object_key": "", "sha256": "", "size": 0}

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

        # 2. Parse existing visible history events
        existing_visible, existing_events = _parse_visible_history(current_body)

        # 3. Build the real prospective visible body
        #    base renderer + existing history + new event
        if reserved_event is not None:
            # Merge existing events into prospective base, then prepend new event
            if existing_events:
                existing_block = _build_history_block(existing_events)
                temp_base = prospective_visible.rstrip() + "\n\n" + existing_block + "\n"
            else:
                temp_base = prospective_visible
            prospective_with_history = _insert_history_into_visible(temp_base, reserved_event)
        else:
            prospective_with_history = prospective_visible

        # 4. Count real prospective visible bytes
        body_bytes = _count_visible_bytes(prospective_with_history)
        if body_bytes <= _MAX_VISIBLE_LIFECYCLE_BYTES:
            return

        # 5. Need to archive oldest events
        if not existing_events:
            raise GitHubStateProxyError(
                "preflight archive: visible lifecycle exceeds 48 KiB "
                "soft limit and no history events available to archive"
            )

        # 6. Truncate oldest events to fit
        kept, archived = _truncate_events_to_fit(
            existing_events,
            _MAX_VISIBLE_LIFECYCLE_BYTES - 2048,  # 2 KiB headroom
        )

        if not archived:
            raise GitHubStateProxyError(
                "preflight archive: unable to fit visible lifecycle even "
                "after attempting to archive oldest events"
            )

        # 7. Create archive comment
        existing_comments = await self.proxy.get_issue_comments(repo, pr_number)
        page_num = _next_history_archive_page(existing_comments, repo, pr_number)

        archive_body = _build_archive_body(
            repo, pr_number, page_num, archived,
            extra_text="Archived oldest history events to preserve main lifecycle capacity.",
        )

        await self.proxy.post_issue_comment(repo, pr_number, archive_body)

        # 8. Rewrite the SAME lifecycle comment to remove archived events
        from .github_state_proxy import HIDDEN_STATE_MARKER
        hidden_idx = current_body.find(HIDDEN_STATE_MARKER)
        if hidden_idx < 0:
            raise GitHubStateProxyError("preflight archive: cannot find hidden state marker")

        existing_visible_part = current_body[:hidden_idx].rstrip()

        final_events = kept if kept else []

        # Build the new history block with kept events only
        new_history_block = _build_history_block(final_events)

        # Reconstruct visible with shrunk history
        h_start = existing_visible_part.find(VISIBLE_HISTORY_START_MARKER)
        h_end = existing_visible_part.find(VISIBLE_HISTORY_END_MARKER)
        if h_start >= 0 and h_end >= 0:
            before_hist = existing_visible_part[:h_start]
            after_hist = existing_visible_part[h_end + len(VISIBLE_HISTORY_END_MARKER):]
            new_visible = before_hist + new_history_block + after_hist
        else:
            new_visible = existing_visible_part.rstrip() + "\n\n" + new_history_block + "\n"

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
        if isinstance(comment_id, int):
            await self.proxy.update_comment(repo, comment_id, final_body)

    async def _write_lifecycle_with_history(
        self,
        repo: str,
        pr_number: int,
        state: dict,
        new_visible_markdown: str,
        event: dict | None = None,
    ) -> None:
        """Write lifecycle comment with automatic history management.

        If event is provided, it is appended to the visible History section.
        Legacy v1 migration is handled automatically on first meaningful event.
        Preserves existing History events across lifecycle rewrites.
        """
        comment = await self.proxy.find_trusted_lifecycle_comment(repo, pr_number)
        if comment is None:
            if event is not None:
                final_visible = _insert_history_into_visible(new_visible_markdown, event)
            else:
                final_visible = new_visible_markdown
            await self.proxy.write_hidden_state(repo, pr_number, final_visible, state)
            return

        existing_body = comment.get("body", "")
        if not isinstance(existing_body, str):
            if event is not None:
                final_visible = _insert_history_into_visible(new_visible_markdown, event)
            else:
                final_visible = new_visible_markdown
            await self.proxy.write_hidden_state(repo, pr_number, final_visible, state)
            return

        from .github_state_proxy import HIDDEN_STATE_MARKER
        idx = existing_body.find(HIDDEN_STATE_MARKER)
        if idx < 0:
            if event is not None:
                final_visible = _insert_history_into_visible(new_visible_markdown, event)
            else:
                final_visible = new_visible_markdown
            await self.proxy.write_hidden_state(repo, pr_number, final_visible, state)
            return

        existing_visible = existing_body[:idx].rstrip()

        # Check if this is a legacy comment (missing new-format markers)
        is_legacy = (
            VISIBLE_HISTORY_START_MARKER not in existing_visible
            and "### Workflow" not in existing_visible
        )

        if is_legacy and event is not None:
            # First meaningful event on a legacy comment
            # 1. Preserve legacy visible markdown in an archive
            legacy_snapshot = existing_visible

            existing_comments = await self.proxy.get_issue_comments(repo, pr_number)
            page_num = _next_history_archive_page(existing_comments, repo, pr_number)

            archive_body = _build_archive_body(
                repo, pr_number, page_num, [],
                extra_text=(
                    "Legacy lifecycle snapshot preserved before history-enabled renderer migration.\n\n"
                    + legacy_snapshot
                ),
            )
            await self.proxy.post_issue_comment(repo, pr_number, archive_body)

        # 2. Collect existing history events from the current lifecycle comment
        existing_events: list[dict] = []
        if not is_legacy:
            _, existing_events = _parse_visible_history(existing_visible)

        # 3. Build the prospective new visible markdown with event + existing events
        if event is not None:
            # _insert_history_into_visible reads existing events from visible markdown.
            # new_visible_markdown is a fresh renderer output with no history section.
            # So we inject existing events first via a temporary placeholder,
            # then let _insert_history_into_visible prepend the new event on top.
            if existing_events:
                existing_history_block = _build_history_block(existing_events)
                temp_visible = new_visible_markdown.rstrip() + "\n\n" + existing_history_block + "\n"
            else:
                temp_visible = new_visible_markdown
            final_visible = _insert_history_into_visible(temp_visible, event)
        else:
            final_visible = new_visible_markdown

        # 3.5. Discover trusted archive comments and add archive links if any
        try:
            existing_comments = await self.proxy.get_issue_comments(repo, pr_number)
            archive_links = self._discover_trusted_archive_links(
                existing_comments, repo, pr_number,
            )
            if archive_links:
                archives_block = comments_mod._archives_link_block(archive_links)
                # Insert archive links before the History section
                h_start = final_visible.find("### History")
                if h_start >= 0:
                    final_visible = (
                        final_visible[:h_start]
                        + archives_block
                        + final_visible[h_start:]
                    )
                else:
                    # No History section yet; append before hidden state marker
                    final_visible = final_visible.rstrip() + "\n\n" + archives_block
        except Exception:
            # Archive link discovery is best-effort
            pass

        # 4. Write the final state
        await self.proxy.write_hidden_state(repo, pr_number, final_visible, state)

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
        new_head: str, comment_id: int,
    ) -> None:
        """Handle HEAD drift by superseding the current deployment."""
        old_head = state.get("head_sha", "")
        old_status = state.get("status", "review-required")
        state["status"] = "review-required"
        state["head_sha"] = new_head
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
        state["status"] = "review-required"
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

        if state is None and (pr_state != "open" or pr_merged):
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
                self._reset_review_lifecycle_state(
                    state,
                    head_sha=current_head,
                    status="review-required",
                    review_evidence={},
                )
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
                        "lifecycle": f"`{state.get('status', 'review-required')}` \u2192 `review-required`",
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

            if state.get("status") in {"deploy-requested", "testing", "succeeded", "failed"}:
                try:
                    await self.proxy.project_status_label(
                        repo, pr_number, state.get("status", "review-required"),
                    )
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
                    evidence = extract_review_evidence(
                        comments,
                        self.config.review_comment_author_id,
                        self.config.review_comment_author_login,
                    )
                    if evidence is None:
                        # Build succeeded but Test/Code Review not yet complete
                        desired_status = "reviewing"
                        review_evidence_data = {}
                        build_infos = []
                    else:
                        resolved_head_ev = await self._resolve_commit_prefix_for_head(
                            repo, current_head, evidence.commit_prefix,
                        )
                        if resolved_head_ev != current_head:
                            desired_status = "review-required"
                            review_evidence_data = {}
                            build_infos = []
                        else:
                            desired_status = "deploy-ready"
                            review_evidence_data = self._review_evidence_snapshot(evidence, resolved_head_ev)
                            build_infos = []
                            for eb in evidence.builds:
                                bi = BuildInfo(
                                    idx=0,
                                    target=eb.target,
                                    driver_path=eb.driver_path,
                                    variant=eb.variant,
                                    success=eb.success,
                                    image_tag=eb.image_tag,
                                    deployable=_is_deployable_build(
                                        repo,
                                        type("_build", (), {
                                            "target": eb.target,
                                            "success": eb.success,
                                            "image_tag": eb.image_tag,
                                            "driver_path": eb.driver_path,
                                        })()
                                    ),
                                )
                                build_infos.append(bi)
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
                    markdown = comments_mod.deploy_ready(repo, pr_number, current_head, build_infos)
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

    async def _refresh_uncertain_state(
        self, repo: str, pr_number: int, state: dict,
    ) -> str:
        """Refresh an uncertain command without replaying the old comment.

        Returns:
        - "deploy-requested" when a fresh validation snapshot was rebuilt.
        - "review-required" when the head drifted or no exact review job exists.
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

        if pr_state != "open" or pr_merged:
            return "noop"

        if not current_head or current_head != state.get("head_sha", ""):
            state["status"] = "review-required"
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
                "args": dict(cmd.get("args", {}) or {}),
            }
            state["last_processed_comment_id"] = max(old_cursor, comment_id)
            markdown = comments_mod.review_required(
                repo, pr_number, current_head or state.get("head_sha", ""),
            )
            event = {
                "event": "HEAD drift detected",
                "lifecycle": f"`{state.get('status', 'review-required')}` \u2192 `review-required`",
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
            evidence = extract_review_evidence(
                comments,
                self.config.review_comment_author_id,
                self.config.review_comment_author_login,
            )
        except Exception as e:
            logger.warning(
                "refresh_uncertain comment evidence %s#%s: %s",
                repo, pr_number, e,
            )
            evidence = None

        if evidence is None:
            state["status"] = "review-required"
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
                "args": dict(cmd.get("args", {}) or {}),
            }
            state["last_processed_comment_id"] = max(old_cursor, comment_id)
            markdown = comments_mod.review_required(
                repo, pr_number, current_head,
            )
            event = {
                "event": "Review evidence unavailable",
                "lifecycle": f"`{state.get('status', 'review-required')}` \u2192 `review-required`",
                "timestamp": comments_mod.beijing_now_str(),
            }
            await self._write_lifecycle_with_history(
                repo, pr_number, state, markdown, event=event,
            )
            await self.proxy.project_status_label(repo, pr_number, "review-required")
            return "review-required"

        # Convert evidence builds to BuildInfo
        builds = []
        resolved_head = await self._resolve_review_evidence_for_head(repo, current_head, evidence)
        if resolved_head is None:
            # resolved_head is None => fail closed, cannot write None into canonical review_evidence
            state["status"] = "review-required"
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
                "args": dict(cmd.get("args", {}) or {}),
            }
            state["last_processed_comment_id"] = max(old_cursor, comment_id)
            markdown = comments_mod.review_required(
                repo, pr_number, current_head,
            )
            event = {
                "event": "Review evidence unresolved",
                "lifecycle": f"`{state.get('status', 'review-required')}` \u2192 `review-required`",
                "timestamp": comments_mod.beijing_now_str(),
            }
            await self._write_lifecycle_with_history(
                repo, pr_number, state, markdown, event=event,
            )
            await self.proxy.project_status_label(repo, pr_number, "review-required")
            return "review-required"
        for eb in evidence.builds:
            bi = BuildInfo(
                idx=0,
                target=eb.target,
                driver_path=eb.driver_path,
                variant=eb.variant,
                success=eb.success,
                image_tag=eb.image_tag,
                deployable=_is_deployable_build(
                    repo,
                    type("_build", (), {
                        "target": eb.target,
                        "success": eb.success,
                        "image_tag": eb.image_tag,
                        "driver_path": eb.driver_path,
                    })()
                ),
            )
            builds.append(bi)

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
        evidence_changed = old_review_evidence != review_evidence_data
        # Symmetric set comparison: detects added, removed, and changed components.
        semantic_changed = old_semantic_duplicate or old_semantic_keys != fresh_semantic_keys

        if evidence_changed or semantic_changed:
            # Genuinely different build/evidence — reset to fresh snapshot.
            fresh_reset_components: list[dict] = []
            for component in fresh_components:
                item = dict(component)
                item.pop("runtime_id", None)
                fresh_reset_components.append(item)
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
                if old_cid in deployed_component_ids:
                    runtime_id = old_comp.get("runtime_id", "")
                    if not isinstance(runtime_id, str) or not runtime_id:
                        # Missing runtime_id for deployed component — reset to fresh.
                        _migration_ok = False
                        break

            if not _migration_ok:
                # Fall through to the reset branch below.
                fresh_reset_components: list[dict] = []
                for component in fresh_components:
                    item = dict(component)
                    item.pop("runtime_id", None)
                    fresh_reset_components.append(item)
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
                        migrated.append({
                            "component_id": old_cid,
                            "target": fresh["target"],
                            "driver_path": fresh["driver_path"],
                            "variant": fresh["variant"],
                            "review_image_tag": fresh["review_image_tag"],
                            "image_ref": old_comp.get("image_ref", ""),
                            "resolved_platform": fresh["resolved_platform"],
                            "runtime_id": old_comp.get("runtime_id", ""),
                        })
                    else:
                        # Rule A: migrate to tag, reuse old component_id
                        migrated.append({
                            "component_id": old_cid,
                            "target": fresh["target"],
                            "driver_path": fresh["driver_path"],
                            "variant": fresh["variant"],
                            "review_image_tag": fresh["review_image_tag"],
                            "image_ref": fresh["image_ref"],
                            "resolved_platform": fresh["resolved_platform"],
                        })

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
        gate_note.append("fresh HEAD + actor -> CLEAN GATE")
        markdown = comments_mod.deploy_requested(
            repo, pr_number, current_head,
            state.get("components", []),
            self._get_machine_groups_for_components(state.get("components", [])),
            gate_note=gate_note,
            deployments=state.get("deployments", []),
        )
        event = {
            "event": "Restart recovery: evidence refreshed",
            "timestamp": comments_mod.beijing_now_str(),
        }
        await self._write_lifecycle_with_history(
            repo, pr_number, state, markdown, event=event,
        )
        await self.proxy.project_status_label(repo, pr_number, "deploy-requested")
        return "deploy-requested"

    async def _resolve_review_evidence_for_head(
        self, repo: str, fresh_head: str, evidence: "ReviewCommentEvidence",
    ) -> str | None:
        """Resolve evidence.commit_prefix to full SHA and verify == fresh_head.

        Returns resolved 40-hex SHA on success, or None on any failure.
        """
        from .review_comment_parser import ReviewCommentEvidence
        if not isinstance(evidence, ReviewCommentEvidence):
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
    ):
        """Fresh-read PR Conversation and validate evidence matches persisted snapshot."""
        from .review_comment_parser import extract_review_evidence
        try:
            comments = await self.proxy.get_issue_comments(repo, pr_number)
        except Exception:
            return False
        review_trust = self.config.review_comment_author_id or ""
        review_login = self.config.review_comment_author_login or ""
        evidence = extract_review_evidence(comments, review_trust, review_login)
        if evidence is None:
            return False
        resolved_head = await self._resolve_review_evidence_for_head(
            repo, fresh_head, evidence,
        )
        if resolved_head != fresh_head:
            return False
        fresh_snapshot = self._review_evidence_snapshot(evidence, resolved_head)
        return fresh_snapshot == persisted_review_evidence


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
                    "`/approve_deploy` requires `machine=<alias>`.",
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

        logger.warning("on_command: unknown kind %s", cmd.kind)
        return True


def _extract_visible_markdown(body: str) -> str:
    """Extract visible markdown before the hidden state marker."""
    from .github_state_proxy import HIDDEN_STATE_MARKER
    idx = body.find(HIDDEN_STATE_MARKER)
    if idx < 0:
        return body
    return body[:idx].rstrip()


# Backward-compatible aliases
DeployServiceError = DeployControllerError
DeploymentService = DeployController
