"""Focused regressions for the final source-alignment contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import inspect
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ..agent_core_client import AgentCoreClient, AgentCoreError
from ..case_runner import CaseRunner
from ..config import Config
from ..models import MachineInfo
from ..policy import Policy
from ..registry_client import RegistryClient, RegistryError, ResolvedImage
from ..service import DeployController
from ..github_state_proxy import GitHubStateProxy
from .conftest import make_config


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _request_state() -> dict:
    return {
        "version": 1,
        "head_sha": "a" * 40,
        "status": "deploy-ready",
        "review_evidence": {"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z", "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z", "review_author_id": "7950763"},
        "components": [],
        "deployments": [],
        "case_results": {},
        "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "approve_attempts": [],
        "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    }


def _component(**overrides) -> dict:
    component = {
        "component_id": "comp-001",
        "target": "perception",
        "driver_path": "",
        "variant": "5.11",
        "review_image_tag": "registry.example/repo:v1",
        "image_ref": "registry.example/repo@sha256:" + "a" * 64,
        "resolved_platform": "linux/arm64",
        "runtime_id": "perception",
    }
    component.update(overrides)
    return component



def test_runtime_env_names_are_upstream_existing_only():
    files = {
        path: _text(path)
        for path in (
            "agents/deploy_approval/config.py",
            "agents/deploy_approval/server.py",
            "agents/deploy_approval/agent_core_client.py",
            "deploy/deploy-approval/deploy.sh",
            "deploy/deploy-approval/docker-compose.yml",
        )
    }
    forbidden = (
        "DA_DEPLOY_ENV",
        "DA_REVIEW_ENV",
        "DA_AGENT_CORE_ENV",
        "DA_ENV_FILE",
        "GITHUB_COMMAND_POLL_INTERVAL_SECONDS",
        "AGENT_CORE_TOKEN",
        "API_TOKEN",
        "MACHINE_OWNERS_FILE",
        "MACHINE_OWNERS_HOST_FILE",
        "REVIEW_AGENT_BASE_URL",
        "ALLOW_PRIVATE_HTTP",
        "HTTP_ALLOWED_CIDRS",
        "HTTP_CONNECT_TIMEOUT",
        "HTTP_READ_TIMEOUT",
        "HTTP_TOTAL_TIMEOUT",
        "HEALTH_POLL_INTERVAL_SECONDS",
        "HEALTH_TIMEOUT_SECONDS",
        "COS_REGION",
        "COS_BUCKET",
        "COS_SECRET_ID",
        "COS_SECRET_KEY",

        "COS_PREFIX",
        "REGISTRY_USER_ENV",
        "REGISTRY_PASSWORD_ENV",
        "REGISTRY_AUTH_HOST_ALLOWLIST",
    )
    for name in forbidden:
        env_read = re.compile(rf'os\.getenv\(\s*[\"\']{re.escape(name)}[\"\']')
        shell_read = re.compile(rf"\${{{re.escape(name)}}}")
        compose_env = re.compile(rf"^\s*{re.escape(name)}:\s*\"", re.MULTILINE)
        assert not any(
            env_read.search(text)
            or shell_read.search(text)
            or compose_env.search(text)
            for text in files.values()
        ), name
    all_text = "\n".join(files.values())
    for name in (
        "GITHUB_APP_ID",
        "GITHUB_INSTALLATION_ID",
        "GITHUB_APP_PRIVATE_KEY_FILE",
        "GITHUB_REPOS",
        "POLL_ENABLED",
        "POLL_INTERVAL_SECONDS",
        "WEBHOOK_ENABLED",
        "GITHUB_WEBHOOK_SECRET",
        "REGISTRY",
        "REGISTRY_USER",
        "REGISTRY_PASSWORD",
    ):
        assert name in all_text


def test_poll_reuses_poll_interval_seconds():
    text = _text("agents/deploy_approval/github_command_watcher.py")
    assert "self.config.poll_interval_seconds" in text
    assert "GITHUB_COMMAND_POLL_INTERVAL_SECONDS" not in text


def test_no_da_environment_names():
    text = "\n".join(
        _text(path)
        for path in (
            "agents/deploy_approval/config.py",
            "agents/deploy_approval/server.py",
            "deploy/deploy-approval/deploy.sh",
            "deploy/deploy-approval/docker-compose.yml",
        )
    )
    assert "DA_" not in text


def test_no_custom_agent_core_token_env():
    text = "\n".join(
        _text(path)
        for path in (
            "agents/deploy_approval/config.py",
            "agents/deploy_approval/agent_core_client.py",
            "agents/deploy_approval/service.py",
            "deploy/deploy-approval/deploy.sh",
        )
    )
    assert "AGENT_CORE_TOKEN" not in text
    assert "os.getenv(\"ACCESS_TOKEN\")" not in _text("agents/deploy_approval/agent_core_client.py")


def test_no_deploy_approval_api_token_runtime_state():
    text = "\n".join(
        _text(path)
        for path in (
            "agents/deploy_approval/config.py",
            "agents/deploy_approval/server.py",
            "deploy/deploy-approval/deploy.sh",
            "deploy/deploy-approval/docker-compose.yml",
        )
    )
    assert "API_TOKEN" not in text
    assert "config.api_token" not in _text("agents/deploy_approval/server.py")


def test_machine_and_cos_config_are_fixed_read_only_files():
    config = Config()
    assert config.machine_owners_file == "/run/deploy-approval/machines.yaml"
    assert config.secrets_file == "/run/deploy-approval/secrets.yaml"

    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    compose = _text("deploy/deploy-approval/docker-compose.yml")
    assert "./machines.yaml" in deploy_sh
    assert "./secrets.yaml" in deploy_sh
    assert "./machines.yaml:/run/deploy-approval/machines.yaml:ro" in compose
    assert "./secrets.yaml:/run/deploy-approval/secrets.yaml:ro" in compose


def test_deploy_script_has_no_stateful_purge():
    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    assert "PURGE-DEPLOY-DATA" not in deploy_sh
    assert "down -v" not in deploy_sh


def test_dockerfile_has_no_deploy_approval_data_state_dir():
    dockerfile = _text("agents/deploy_approval/Dockerfile")
    assert "/data" not in dockerfile


def test_source_of_truth_contract_is_explicit():
    docs = "\n".join(
        _text(path)
        for path in (
            "DEPLOY_APPROVAL_AGENT.md",
            "docs/deploy-approval-github-driven-architecture.md",
        )
    )
    assert "GitHub hidden lifecycle JSON" in docs
    assert "GitHub PR comments are the source of Review Agent Build/Test/Code Review evidence" in docs
    assert "Registry only verifies/resolves that exact Review Agent image tag" in docs
    assert "Agent Core only supplies runtime identity, current `running_image`, and MCP evidence" in docs


def test_single_replica_restart_safe_contract_is_documented():
    text = "\n".join(
        _text(path)
        for path in (
            "deploy/deploy-approval/docker-compose.yml",
            "docs/deploy-approval-github-driven-architecture.md",
        )
    )
    lowered = text.lower()
    assert "single-replica" in lowered or "single replica" in lowered
    assert "single-writer" in lowered or "single writer" in lowered
    assert "replicas >1" in text or "multiple concurrent Deploy Controller replicas are unsupported" in text


def test_architecture_doc_explicitly_documents_single_replica_single_writer():
    docs = "\n".join(
        _text(path)
        for path in (
            "DEPLOY_APPROVAL_AGENT.md",
            "docs/deploy-approval-github-driven-architecture.md",
        )
    )
    lowered = docs.lower()
    assert "restart-safe stateless" in lowered
    assert "single-replica / single-writer" in lowered
    assert "githubcommandwatcher" in lowered
    assert "serially processes mutating commands" in lowered


def test_architecture_doc_rejects_multi_replica_claim():
    docs = _text("docs/deploy-approval-github-driven-architecture.md")
    lowered = docs.lower()
    assert "multiple concurrent deploy controller replicas are unsupported" in lowered
    assert "no cas/distributed lock" in lowered
    assert "replicas >1" in lowered


def test_docs_use_poll_interval_seconds_not_hardcoded_60_second_polling():
    docs = "\n".join(
        _text(path)
        for path in (
            "DEPLOY_APPROVAL_AGENT.md",
            "docs/deploy-approval-github-driven-architecture.md",
            "agents/deploy_approval/github_command_watcher.py",
        )
    )
    assert "POLL_INTERVAL_SECONDS" in docs
    assert "30 seconds" in docs
    assert "60-second polling" not in docs
    assert "every 60 seconds" not in docs


def test_deploy_script_is_executable():
    mode = Path("deploy/deploy-approval/deploy.sh").stat().st_mode
    assert os.access("deploy/deploy-approval/deploy.sh", os.X_OK)
    assert mode & 0o111


def test_config_has_no_api_token_field():
    assert "api_token" not in Config.__dataclass_fields__


def test_config_has_no_legacy_github_command_poll_interval_field():
    assert "github_command_poll_interval_seconds" not in Config.__dataclass_fields__


def test_agent_core_client_has_no_token_env_parameter():
    sig = inspect.signature(AgentCoreClient.__init__)
    params = sig.parameters
    assert "token_env" not in params
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


def test_agent_core_client_rejects_unknown_constructor_kwargs():
    with pytest.raises(TypeError):
        AgentCoreClient(
            Config(),
            base_url="https://192.0.2.1:15678",
            node_host="192.0.2.1",
            typo_option="must-fail",
        )


def test_dead_registry_env_selector_fields_removed_when_unused():
    assert "registry_user_env" not in Config.__dataclass_fields__
    assert "registry_password_env" not in Config.__dataclass_fields__


def test_sparse_review_env_does_not_emit_empty_optional_overrides():
    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    assert "dotenv_has_key" in deploy_sh
    assert "if [ -n \"$value\" ]" in deploy_sh
    assert "printf 'GITHUB_REPOS=%s\\n'" not in deploy_sh
    assert "printf 'POLL_ENABLED=%s\\n'" not in deploy_sh


def test_poll_defaults_to_30_when_upstream_poll_interval_is_absent():
    compose = _text("deploy/deploy-approval/docker-compose.yml")
    assert 'POLL_INTERVAL_SECONDS: "${POLL_INTERVAL_SECONDS:-30}"' in compose
    assert 'POLL_ENABLED: "${POLL_ENABLED:-true}"' in compose


def test_optional_webhook_env_absence_keeps_default_false():
    compose = _text("deploy/deploy-approval/docker-compose.yml")
    assert 'WEBHOOK_ENABLED: "${WEBHOOK_ENABLED:-false}"' in compose


def test_machine_policy_symlink_rejected():
    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    lowered = deploy_sh.lower()
    assert '! -L "$MACHINES_FILE"' in deploy_sh
    assert "machine policy file must be a regular file" in lowered


def test_machine_policy_duplicate_node_id_rejected():
    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    assert "duplicate node_id" in deploy_sh


def test_machine_policy_empty_owner_rejected():
    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    assert "owners list" in deploy_sh
    assert "invalid owner entry" in deploy_sh


def test_secrets_symlink_rejected():
    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    assert '! -L "$SECRETS_FILE"' in deploy_sh
    assert "COS secrets file must be a regular file" in deploy_sh


def test_secrets_requires_version_one():
    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    assert "secrets file must be version 1" in deploy_sh


def test_secrets_rejects_non_string_secret_fields():
    deploy_sh = _text("deploy/deploy-approval/deploy.sh")
    assert "must be a string" in deploy_sh




def test_architecture_markdown_includes_machine_owner():
    """DEPLOY_APPROVAL_AGENT.md Architecture must include Machine Owner."""
    md = Path("DEPLOY_APPROVAL_AGENT.md").read_text(encoding="utf-8")
    assert "Machine Owner" in md
    # Registry and GitHub State Proxy must not be top-level actors
    # They may appear in lower sections but not as architecture actors
    arch_section = md.split("## Architecture")[1].split("```")[1] if "## Architecture" in md else ""
    if arch_section:
        assert "GitHub State Proxy" not in arch_section



# ═══════════════════════════════════════════════════════════════════════
# MIGRATED from test_v8_contract.py
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_phanthymotus_core_only_is_not_deployable(controller, proxy, mock_github):
    mock_github.get_comment.return_value = {"user": {"id": 111, "login": "alice"}, "body": "/request_deploy"}
    mock_github.get_pr.return_value = {"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}}
    from ..review_comment_parser import ReviewBuild, ReviewCommentEvidence
    mock_evidence = ReviewCommentEvidence(
        head_sha="a" * 40,
        commit_prefix="abcdef1",
        review_author_id="7950763",
        review_author_login="review-bot",
        builds=[
            ReviewBuild(
                target="CORE", driver_path="", variant="",
                success=True, image_tag="registry.example/repo:tag-core",
            )
        ],
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        code_review_comment_id=1003,
        code_review_created_at="2026-09-18T03:57:00Z",
        code_review_text="Review completed.",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=mock_evidence):
        proxy.post_issue_comment = AsyncMock()
        proxy.write_hidden_state = AsyncMock()
        proxy.project_status_label = AsyncMock()

        await controller.handle_request_deploy("4paradigm/phanthymotus", 1, 99)

    proxy.write_hidden_state.assert_not_called()
    proxy.post_issue_comment.assert_called()


@pytest.mark.asyncio
async def test_phanthymotus_core_plus_perception_deploys_only_perception(controller, proxy, mock_github):
    mock_github.get_comment.return_value = {"user": {"id": 111, "login": "alice"}, "body": "/request_deploy"}
    mock_github.get_pr.return_value = {"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}}
    from ..review_comment_parser import ReviewBuild, ReviewCommentEvidence
    mock_evidence = ReviewCommentEvidence(
        head_sha="a" * 40,
        commit_prefix="abcdef1",
        review_author_id="7950763",
        review_author_login="review-bot",
        builds=[
            ReviewBuild(target="CORE", driver_path="", variant="", success=True, image_tag="registry.example/repo:tag-core"),
            ReviewBuild(target="perception", driver_path="", variant="5.11", success=True, image_tag="registry.example/repo:tag-perception"),
        ],
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-18T03:55:54Z",
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-18T03:57:00Z",
        code_review_text="Review completed.",
    )
    proxy.read_hidden_state = AsyncMock(return_value={
        "version": 1,
        "head_sha": "a" * 40,
        "status": "deploy-ready",
        "review_evidence": {"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z", "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z", "review_author_id": "7950763"},
        "components": [],
        "deployments": [],
        "approve_attempts": [],
        "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "case_results": {},
        "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    })
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=mock_evidence):
        controller.registry.resolve = AsyncMock(
            return_value=SimpleNamespace(
                image_ref="registry.example/perception@sha256:" + "b" * 64,
                platform="linux/arm64",
            )
        )
        proxy.write_hidden_state = AsyncMock()
        proxy.project_status_label = AsyncMock()
        mock_github.resolve_commit_sha.return_value = "a" * 40

        await controller.handle_request_deploy("4paradigm/phanthymotus", 1, 100)

    written_state = proxy.write_hidden_state.call_args.args[3]
    assert [c["target"] for c in written_state["components"]] == ["perception"]
    assert written_state["review_evidence"].get("build_comment_id") == 1001





# ══════════════════════════════════════════════════════════════════════════════
# MIGRATED from test_v10_contract.py
# ══════════════════════════════════════════════════════════════════════════════

# ── Fixtures and helpers for runtime resolution tests (migrated) ──


@pytest.fixture
def mock_github():
    github = MagicMock()
    github.get_pr = AsyncMock()
    github.get_comment = AsyncMock()
    github.get_issue_comments = AsyncMock()
    github.resolve_commit_sha = AsyncMock()
    github.collaborator_permission = AsyncMock()
    github.comment_identity = AsyncMock()
    github.post_issue_comment = AsyncMock()
    github.update_comment = AsyncMock()
    github.get_issue_labels = AsyncMock()
    github.list_open_prs = AsyncMock()
    github.get_repo = AsyncMock()
    return github


@pytest.fixture
def proxy(config, mock_github):
    return GitHubStateProxy(config, mock_github, github_app_id="12345")


@pytest.fixture
def policy(config):
    p = Policy(config)
    p.machines = {
        "perception-machine": MachineInfo(
            alias="perception-machine",
            node_id="node-1",
            owners=["owner1"],
            node_host="10.0.0.1",
            targets=["perception", "actucore"],
            platforms=["linux/arm64"],
            variants=["5.11", "6.1"],
        ),
        "driver-machine": MachineInfo(
            alias="driver-machine",
            node_id="node-2",
            owners=["owner1"],
            node_host="10.0.0.2",
            targets=["driver"],
            platforms=["linux/arm64"],
            variants=["5.11"],
            driver_paths=["unitree/g1"],
        ),
        "multi-machine": MachineInfo(
            alias="multi-machine",
            node_id="node-3",
            owners=["owner1"],
            node_host="10.0.0.3",
            targets=["perception", "actucore", "driver"],
            platforms=["linux/arm64"],
            variants=["5.11", ""],
            driver_paths=["custom/driver"],
        ),
    }
    return p


@pytest.fixture
def controller(config, proxy, policy, mock_github):
    registry = MagicMock()
    registry.resolve = AsyncMock()
    return DeployController(config, proxy, policy, mock_github, registry)


def _component(**overrides):
    component = {
        "component_id": "comp-001",
        "target": "perception",
        "driver_path": "",
        "variant": "5.11",
        "review_image_tag": "registry/repo:v1",
        "image_ref": "registry/repo@sha256:" + "a" * 64,
        "resolved_platform": "linux/arm64",
        "runtime_id": "perception",
    }
    component.update(overrides)
    return component


def _state(**overrides):
    state = {
        "version": 1,
        "head_sha": "a" * 40,
        "status": "testing",
        "review_evidence": {"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z", "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z", "review_author_id": "7950763"},
        "components": [_component()],
        "deployments": [{"machine": "perception-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
        "case_results": {},
        "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    }
    state.update(overrides)
    return state


class TestRuntimeResolution:
    """Exact runtime id resolution, no fallback, no fuzzy."""

    def test_service_runtime_requires_exact_perception_id(self, controller):
        """Perception runtime id must be EXACT 'perception'."""
        drivers = [
            {"id": "perception", "target": "perception", "image": "registry/perception:v1"},
        ]
        result = controller._resolve_component_runtime(
            drivers, {"target": "perception", "image_ref": "registry/perception@sha256:" + "a" * 64}
        )
        assert result is not None
        assert result["runtime_id"] == "perception"

        # No "perception" in drivers
        drivers2 = [{"id": "runtime-1", "target": "perception", "image": "registry/perception:v1"}]
        result2 = controller._resolve_component_runtime(
            drivers2, {"target": "perception", "image_ref": "registry/perception@sha256:" + "a" * 64}
        )
        assert result2 is None

    def test_service_runtime_requires_exact_actucore_id(self, controller):
        """ActuCore runtime id must be EXACT 'actucore'."""
        drivers = [
            {"id": "actucore", "target": "actucore", "image": "registry/actucore:v1"},
        ]
        result = controller._resolve_component_runtime(
            drivers, {"target": "actucore", "image_ref": "registry/actucore@sha256:" + "a" * 64}
        )
        assert result is not None
        assert result["runtime_id"] == "actucore"

        drivers2 = [{"id": "runtime-2", "target": "actucore", "image": "registry/actucore:v1"}]
        result2 = controller._resolve_component_runtime(
            drivers2, {"target": "actucore", "image_ref": "registry/actucore@sha256:" + "a" * 64}
        )
        assert result2 is None

    def test_driver_runtime_missing_repository_metadata_fails_closed(self, controller):
        """Driver resolver must fail closed when Agent Core entry has no image repository metadata."""
        drivers = [
            {"id": "unitree-g1", "category": "driver", "image": ""},
        ]
        result = controller._resolve_component_runtime(
            drivers,
            {
                "target": "driver",
                "image_ref": "registry/unitree/g1@sha256:" + "a" * 64,
            },
        )
        assert result is None

    def test_driver_runtime_never_falls_back_to_driver_path(self, controller):
        """Driver resolution must never use driver_path as runtime id."""
        drivers = [
            {"id": "unitree-g1", "category": "driver",
             "image": "registry/unitree/g1:v1"},
        ]
        result = controller._resolve_component_runtime(
            drivers,
            {
                "target": "driver",
                "driver_path": "unitree/g1",
                "image_ref": "registry/unitree/g1@sha256:" + "a" * 64,
            },
        )
        assert result is not None
        assert result["runtime_id"] == "unitree-g1"
        # driver_path was never used as a fallback for id

    def test_driver_runtime_never_falls_back_to_single_driver(self, controller):
        """When there is exactly one driver in the catalog, but no image match, fail closed."""
        drivers = [
            {"id": "only-driver", "category": "driver",
             "image": "registry/other:v1"},
        ]
        result = controller._resolve_component_runtime(
            drivers,
            {
                "target": "driver",
                "image_ref": "registry/unitree/g1@sha256:" + "a" * 64,
            },
        )
        assert result is None

    def test_unknown_review_service_variant_fails_closed(self):
        """A non-empty variant that is not 5.11 or 6.1 must fail closed."""
        from ..service import _normalize_variant, DeployControllerError
        with pytest.raises(DeployControllerError, match="unsupported variant"):
            _normalize_variant("unknown-variant")



class TestCaseContract:
    """Case must use pinned runtime_id from deploy, not re-resolve."""

    @pytest.mark.asyncio
    async def test_case_uses_pinned_runtime_id_from_deploy(self, controller):
        """The automated case must use the deploy-pinned runtime_id, not re-list drivers."""
        core = AsyncMock()
        core.list_drivers = AsyncMock(return_value=[
            {"id": "perception", "target": "perception", "mcp_url": "http://mcp/runtime-1"},
        ])
        controller._core_for_node = AsyncMock(return_value=core)
        captured = {}

        runner = MagicMock()
        runner.select_case.return_value = "perception-health-check"

        async def _capture(case_id, deployment):
            captured["runtime_id"] = deployment.get("runtime_id", "")
            return {"passed": True, "case_id": case_id, "logs": [], "error": ""}

        runner.run_case = AsyncMock(side_effect=_capture)
        controller._get_case_runner = MagicMock(return_value=runner)

        result = await controller._run_automated_case(
            "repo",
            1,
            "a" * 40,
            [_component(runtime_id="perception")],
            [{"machine": "perception-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
        )
        assert result == {"comp-001": "pass"}
        assert captured["runtime_id"] == "perception"

    @pytest.mark.asyncio
    async def test_case_does_not_reresolve_runtime_to_different_driver(self, controller):
        """The case must not re-resolve the runtime by calling list_drivers again."""
        core = AsyncMock()
        core.list_drivers = AsyncMock(return_value=[
            {"id": "different-driver", "target": "perception", "mcp_url": "http://mcp/runtime-1"},
        ])
        controller._core_for_node = AsyncMock(return_value=core)
        captured = {}

        runner = MagicMock()
        runner.select_case.return_value = "perception-health-check"

        async def _capture(case_id, deployment):
            captured["runtime_id"] = deployment.get("runtime_id", "")
            captured["_driver_id"] = deployment.get("_driver_id", "")
            return {"passed": True, "case_id": case_id, "logs": [], "error": ""}

        runner.run_case = AsyncMock(side_effect=_capture)
        controller._get_case_runner = MagicMock(return_value=runner)

        # The component has runtime_id="perception" pinned from deploy
        result = await controller._run_automated_case(
            "repo",
            1,
            "a" * 40,
            [_component(runtime_id="perception")],
            [{"machine": "perception-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
        )
        assert result == {"comp-001": "pass"}
        # The case used the pinned runtime_id, not the different one from list_drivers
        assert captured["runtime_id"] == "perception"
        assert captured["_driver_id"] == "perception"
        # list_drivers may have been called but the result was not used for resolution
        # (the pinned runtime_id from deploy takes precedence)
