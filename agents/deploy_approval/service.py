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
import os
import inspect

import logging
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
from .github_state_proxy import GitHubStateProxy, _validate_hidden_state
from .models import BuildInfo, new_id, utc_now
from .policy import Policy, PolicyError
from .review_comment_parser import (
    extract_review_evidence,
    ReviewCommentEvidence,
)
from .registry_client import RegistryClient, parse_reference

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


def _is_deployable_build(build) -> bool:
    target = str(build.target or "").strip()
    if not _is_supported_target(target):
        return False
    if not build.success:
        return False
    if not build.image_tag:
        return False
    return True


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
        registry: object = None,
        agent_core_factory: object = None,
    ):
        self.config = config
        self.proxy = proxy
        self.policy = policy
        self.github = github
        self.registry = registry
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
        base_url = f"http://{machine.node_host}:15678"
        core = AgentCoreClient(
            self.config,
            base_url,
            node_host=machine.node_host,
        )
        try:
            await core.verify()
        except AgentCoreError as e:
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

    async def _build_component_snapshot(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        builds: list[BuildInfo],
    ) -> list[dict] | None:
        snapshot: list[dict] = []
        for build in builds or []:
            if not build.success or not build.deployable:
                continue
            resolved = await self._resolve_image_ref(
                repo,
                pr_number,
                head_sha,
                build,
            )
            if resolved is None:
                return None
            image_ref, resolved_platform = resolved
            if not resolved_platform:
                return None
            component_id = hashlib.sha256(
                f"{build.target}|{build.driver_path}|{build.variant}|{image_ref}".encode()
            ).hexdigest()[:16]
            snapshot.append({
                "component_id": component_id,
                "target": build.target,
                "driver_path": build.driver_path,
                "variant": build.variant,
                "review_image_tag": build.image_tag,
                "image_ref": image_ref,
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
            if state.get("status") != "deploy-ready":
                await self._post_error(
                    repo, pr_number,
                    f"Current status is `{state.get('status')}`. Expected `deploy-ready`.",
                )
                return True
            if pr_head != state.get("head_sha", ""):
                await self._post_error(
                    repo, pr_number,
                    "PR HEAD changed. Refresh review lifecycle before requesting deploy.",
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
                        type("_build", (), {
                            "target": eb.target,
                            "success": eb.success,
                            "image_tag": eb.image_tag,
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

            # Build review_evidence snapshot with resolved full SHA
            review_evidence_data = {
                "build_comment_id": evidence.build_comment_id,
                "build_comment_updated_at": evidence.build_comment_updated_at,
                "commit_prefix": evidence.commit_prefix,
                "resolved_head_sha": resolved_head,
                "test_comment_id": evidence.test_comment_id,
                "code_review_comment_id": evidence.code_review_comment_id,
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
            await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
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

            if state.get("status") != "deploy-requested":
                await self._post_error(
                    repo, pr_number,
                    f"Current status is `{state.get('status')}`. "
                    "Expected `deploy-requested`.",
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

            # Get machine info
            machine = self.policy.get_machine(machine_alias)
            if machine is None:
                await self._post_error(
                    repo, pr_number,
                    f"Unknown machine alias `{machine_alias}`. "
                    "Check machine owners configuration.",
                )
                return True

            # Check permissions async
            await self._check_approval_permissions(
                machine, actor, repo,
            )

            # Determine which component_ids are assigned to this machine group
            components = state.get("components", [])
            existing_deployments = list(state.get("deployments", []))
            compatible_comp_ids = self._get_component_ids_for_machine(
                machine_alias, components
            )
            # Skip already-deployed component_ids
            existing_deployed = _component_id_set(existing_deployments)
            remaining = [c for c in components if c.get("component_id") in compatible_comp_ids and c.get("component_id") not in existing_deployed]

            if not remaining:
                await self._post_error(
                    repo, pr_number,
                    f"No remaining components to deploy for machine `{machine_alias}`.",
                )
                return True

            node_id = machine.node_id
            core = await self._core_for_node(node_id)

            # Clean gate: read running_image for every selected component before
            # any deploy POST. Status fields are ignored.
            preflight = await self._preflight_running_images(core, remaining)
            occupied = [item for item in preflight if item["running_image"]]
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
            if occupied:
                state["status"] = "deploy-requested"
                state["command"] = {
                    "comment_id": comment_id,
                    "kind": "approve_deploy",
                    "phase": "completed",
                    "args": {"machine": machine_alias, "actor": actor},
                }
                state["last_processed_comment_id"] = comment_id
                markdown = comments_mod.approve_deploy_occupied_comment(
                    repo,
                    pr_number,
                    pr_head,
                    machine_alias,
                    [item["component"] for item in occupied],
                    {
                        item["component"].get("component_id", ""): item["running_image"]
                        for item in occupied
                    },
                )
                approve_attempt["outcome"] = "blocked_occupied"
                self._record_approve_attempt(state, approve_attempt)
                await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
                await self.proxy.project_status_label(repo, pr_number, "deploy-requested")
                return True

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

            # Write executing state only after the clean gate passes.
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
            health_records = []
            deploy_error = None
            deploy_outcome_uncertain = False
            # Sequential per-component: deploy then immediately health check
            for comp in remaining:
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

                # Immediately health check this exact component
                health_result = await self._wait_for_deploy_health(
                    core, runtime_id, image_ref,
                    f"{comp.get('target', '')}/{comp.get('component_id', '')[:8]}",
                )
                health_records.append({
                    "runtime_id": runtime_id,
                    "running_image": health_result.get("running_image", ""),
                    "passed": health_result.get("passed", False),
                    "component_id": comp.get("component_id", ""),
                })
                if not health_result.get("passed", False):
                    deploy_error = (
                        f"health check failed for {comp.get('target', '')} "
                        f"runtime={runtime_id}: running_image={health_result.get('running_image', '')} "
                        f"(expected {image_ref})"
                    )
                    break

                # Health PASS: record deployment immediately
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
                approve_attempt["health"] = health_records
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
                await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
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
                approve_attempt["health"] = health_records
                self._record_approve_attempt(state, approve_attempt)
                await self.proxy.write_hidden_state(
                    repo, pr_number, markdown, state
                )
                await self.proxy.project_status_label(repo, pr_number, "failed")

                # Best-effort evidence upload after failed
                try:
                    cos_metadata = await self._upload_evidence(
                        repo, pr_number, pr_head, state, "fail",
                        context={
                            "approve_attempts": state.get("approve_attempts", []),
                            "approve_attempts_total": state.get("approve_attempts_total", 0),
                            "actor": actor,
                            "comment_id": comment_id,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "failed deploy evidence upload %s#%s: %s",
                        repo, pr_number, e,
                    )
                    return True

                if cos_metadata.get("object_key"):
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
                return True

            # Merge into existing deployments
            state["deployments"] = list(existing_deployments) + list(new_deployments)

            # Check if every component_id is now deployed at least once
            all_component_ids = {c.get("component_id", "") for c in components}
            deployed_component_ids = _component_id_set(state["deployments"])
            all_deployed = all_component_ids.issubset(deployed_component_ids) if all_component_ids else False

            state["command"] = {
                "comment_id": comment_id,
                "kind": "approve_deploy",
                "phase": "completed",
                "args": {"machine": machine_alias, "actor": actor},
            }
            state["last_processed_comment_id"] = comment_id
            approve_attempt["outcome"] = "deployed"
            approve_attempt["health"] = health_records
            self._record_approve_attempt(state, approve_attempt)

            if all_deployed:
                state["status"] = "testing"

                # Run automated case only after ALL components deployed
                case_results = await self._run_automated_case(
                    repo, pr_number, pr_head, components,
                    state.get("deployments", []),
                )
                if case_results:
                    state["case_results"].update(case_results)

                case_result_str = ", ".join(
                    f"{k}={v}" for k, v in (case_results or {}).items()
                )
                markdown = comments_mod.testing(
                    repo, pr_number, pr_head, case_result=case_result_str or "",
                )
                await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
                await self.proxy.project_status_label(repo, pr_number, "testing")
            else:
                # Still deploy-requested, show remaining machine commands
                remaining_components = [
                    c for c in components
                    if c.get("component_id", "") not in deployed_component_ids
                ]
                remaining_groups = self._get_machine_groups_for_components(remaining_components)
                markdown = comments_mod.deploy_requested(
                    repo, pr_number, pr_head, remaining_components, remaining_groups,
                )
                await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
                await self.proxy.project_status_label(repo, pr_number, "deploy-requested")
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

            if state.get("status") != "testing":
                await self._post_error(
                    repo, pr_number,
                    f"Current status is `{state.get('status')}`. "
                    "Expected `testing`.",
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
                terminal_markdown = comments_mod.succeeded_comment(
                    repo, pr_number, pr_head,
                )
            else:
                terminal_markdown = comments_mod.failed_comment(repo, pr_number, pr_head)

            # Terminal state is written to GitHub before COS upload
            await self.proxy.write_hidden_state(
                repo, pr_number, terminal_markdown, state,
            )
            await self.proxy.project_status_label(repo, pr_number, state["status"])

            # Upload COS evidence (failure does not roll back terminal state)
            state["cos"] = self._empty_cos()
            try:
                cos_metadata = await self._upload_evidence(
                    repo, pr_number, pr_head, state, result, summary,
                    context={
                        "approve_attempts": state.get("approve_attempts", []),
                        "approve_attempts_total": state.get("approve_attempts_total", 0),
                        "actor": actor,
                        "comment_id": comment_id,
                        "case": [
                            {
                                "component_id": cid,
                                "case_id": _case_id_for_target(
                                    next((c.get("target", "") for c in components if c.get("component_id", "") == cid), "")
                                ),
                                "result": res,
                                "advisory": True,
                            }
                            for cid, res in (state.get("case_results", {}) or {}).items()
                        ],
                    },
                )
            except Exception as e:
                logger.warning(
                    "record_test evidence upload failed %s#%s: %s",
                    repo, pr_number, e,
                )
                return True

            if cos_metadata.get("object_key"):
                markdown = comments_mod.succeeded_comment(
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
                    markdown=markdown,
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

            markdown = comments_mod.deploy_status_comment(
                status, head_sha, repo, pr_number,
                components=components,
                deployments=deployments,
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


    async def _resolve_image_ref(
        self, repo: str, pr_number: int, head_sha: str,
        build: BuildInfo,
    ) -> tuple[str, str] | None:
        """Resolve the exact Review Agent image fact to immutable digest + platform.

        Returns (image_ref, platform) or None.
        """
        try:
            review_image_tag = str(build.image_tag or "")
            if not review_image_tag:
                return None
            family, _tag = parse_reference(review_image_tag)
            resolved = await self.registry.resolve(
                review_image_tag,
                platform="",
                allowed_prefixes=[family],
            )
            if resolved is None:
                return None
            image_ref = resolved.image_ref or ""
            resolved_platform = resolved.platform or ""
            return (image_ref, resolved_platform)
        except Exception as e:
            logger.warning(
                "resolve_image_ref %s#%s: %s", repo, pr_number, e,
            )
            return None

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
        """Determine compatible machine groups from components.

        Returns a list of machine groups, each with the machine alias and
        the component_ids it can host. A machine may host multiple components.

        Uses same fail-closed rules as _get_component_ids_for_machine.
        """
        machines = self.policy.get_machines()
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
            if compatible:
                groups.append({
                    "alias": m.alias,
                    "node_id": m.node_id,
                    "component_ids": compatible,
                })
        return groups

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

    def _occupied_gate_note(
        self, machine_alias: str, component: dict, running_image: str
    ) -> list[str]:
        target = str(component.get("target", "") or "")
        lines = [
            "### Clean Gate",
            "",
            f"Component `{target}` on machine `{machine_alias}` is occupied.",
            f"running_image: `{running_image}`",
            "",
            "running_image != \"\" -> ZERO deploy POST",
            "status: deploy-requested",
            "Machine Owner must clear the occupied runtime image manually.",
            f"Then send `/approve_deploy machine={machine_alias}`.",
        ]
        return lines

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

    async def _wait_for_deploy_health(
        self, core: AgentCoreClient, runtime_id: str,
        image_ref: str, component_label: str,
    ) -> dict:
        """Bounded poll for a deployed component to bind the exact image.

        CLEAN GATE:
        - pinned runtime_id
        - POST deploy immutable image
        - bounded poll existing /api/drivers/{runtime_id}/status
        - running_image == exact immutable image_ref

        Returns {"passed": bool, "running_image": str}.
        """
        import asyncio
        import time

        deadline = time.time() + self.config.health_timeout_seconds
        interval = self.config.health_poll_interval_seconds
        last_running = ""
        while time.time() < deadline:
            try:
                status = await core.driver_status(runtime_id)
                last_running = str(status.get("running_image", "") or "")
                if last_running == image_ref:
                    return {
                        "passed": True,
                        "running_image": last_running,
                    }
            except Exception as e:
                logger.warning(
                    "health poll %s runtime=%s: %s", component_label, runtime_id, e,
                )
            await asyncio.sleep(interval)
        logger.error(
            "health timeout %s runtime=%s after %ss: running_image=%s (expected %s)",
            component_label, runtime_id, self.config.health_timeout_seconds,
            last_running, image_ref,
        )
        return {
            "passed": False,
            "running_image": last_running,
        }

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
        context: dict | None = None,
    ) -> dict:
        """Upload COS evidence bundle. Returns metadata dict.

        Builds evidence in memory, uploads to COS, returns metadata.
        Failure does not raise.
        """
        try:
            object_key = self.cos.build_object_key(
                repo, pr_number, head_sha,
                deployment_id=state.get("head_sha", ""),
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
                    context=context,
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

    async def _supersede_head_drift(
        self, repo: str, pr_number: int, state: dict,
        new_head: str, comment_id: int,
    ) -> None:
        """Handle HEAD drift by superseding the current deployment."""
        old_head = state.get("head_sha", "")
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
        await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
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
        await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
        await self.proxy.project_status_label(repo, pr_number, "review-required")

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
                    await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
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
            evidence = extract_review_evidence(
                comments,
                self.config.review_comment_author_id,
                self.config.review_comment_author_login,
            )
            if evidence is None:
                desired_status = "review-required"
                review_evidence_data: dict = {}
                build_infos: list[BuildInfo] = []
            else:
                # Convert evidence builds to BuildInfo
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
                            type("_build", (), {
                                "target": eb.target,
                                "success": eb.success,
                                "image_tag": eb.image_tag,
                            })()
                        ),
                    )
                    build_infos.append(bi)
                desired_status = "deploy-ready"
                review_evidence_data = {
                    "build_comment_id": evidence.build_comment_id,
                    "build_comment_updated_at": evidence.build_comment_updated_at,
                    "commit_prefix": evidence.commit_prefix,
                    "resolved_head_sha": evidence.resolved_head_sha,
                    "test_comment_id": evidence.test_comment_id,
                    "code_review_comment_id": evidence.code_review_comment_id,
                    "review_author_id": evidence.review_author_id,
                }
        except Exception as e:
            logger.warning(
                "reconcile comment evidence %s#%s head=%s: %s",
                repo, pr_number, current_head, e,
            )
            desired_status = "review-required"
            review_evidence_data: dict = {}
            build_infos = []

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

        try:
            markdown = (
                comments_mod.review_required(repo, pr_number, current_head)
                if desired_status == "review-required"
                else comments_mod.reviewing(repo, pr_number, current_head)
                if desired_status == "reviewing"
                else comments_mod.deploy_ready(repo, pr_number, current_head, build_infos)
            )
            await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
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

        markdown = comments_mod.uncertain_comment(
            repo, pr_number, state.get("head_sha", ""),
        )
        await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
        await self.proxy.project_status_label(
            repo, pr_number, state.get("status", "review-required"),
        )

    async def _refresh_uncertain_state(
        self, repo: str, pr_number: int, state: dict,
    ) -> str:
        """Refresh an uncertain command without replaying the old comment.

        Returns:
        - "deploy-requested" when a fresh immutable snapshot was rebuilt.
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
            await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
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
            await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
            await self.proxy.project_status_label(repo, pr_number, "review-required")
            return "review-required"

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
                    type("_build", (), {
                        "target": eb.target,
                        "success": eb.success,
                        "image_tag": eb.image_tag,
                    })()
                ),
            )
            builds.append(bi)

        review_evidence_data = {
            "build_comment_id": evidence.build_comment_id,
            "build_comment_updated_at": evidence.build_comment_updated_at,
            "commit_prefix": evidence.commit_prefix,
            "resolved_head_sha": evidence.resolved_head_sha,
            "test_comment_id": evidence.test_comment_id,
            "code_review_comment_id": evidence.code_review_comment_id,
            "review_author_id": evidence.review_author_id,
        }
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
            await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
            return "uncertain"

        old_review_evidence = state.get("review_evidence", {})
        old_components = self._canonical_component_snapshot(state.get("components", []))
        fresh_canonical = self._canonical_component_snapshot(fresh_components)
        same_snapshot = (
            old_review_evidence == review_evidence_data and old_components == fresh_canonical
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
        if not same_snapshot:
            fresh_reset_components: list[dict] = []
            for component in fresh_components:
                item = dict(component)
                item.pop("runtime_id", None)
                fresh_reset_components.append(item)
            updated_state["components"] = fresh_reset_components
            updated_state["deployments"] = []
            updated_state["approve_attempts"] = []
            updated_state["approve_attempts_total"] = 0
            updated_state["approve_attempts_truncated"] = False
            updated_state["case_results"] = {}
            updated_state["test_result"] = ""
            updated_state["cos"] = {"object_key": "", "sha256": "", "size": 0}
        else:
            preserved_components = self._components_with_preserved_runtime_bindings(
                fresh_components,
                state.get("components", []),
                state.get("deployments", []),
            )
            if preserved_components is None:
                logger.warning(
                    "uncertain recovery runtime binding unavailable %s#%s head=%s",
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
                await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
                return "uncertain"
            updated_state["components"] = preserved_components
            updated_state["deployments"] = list(state.get("deployments", []))

        state.clear()
        state.update(updated_state)

        gate_note = [
            "### Restart Recovery",
            "",
            "Validation facts were refreshed from the current HEAD.",
        ]
        if same_snapshot:
            gate_note.append("Previously confirmed deployments were preserved.")
        else:
            gate_note.append("Old validation snapshot was replaced.")
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
        )
        await self.proxy.write_hidden_state(repo, pr_number, markdown, state)
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
