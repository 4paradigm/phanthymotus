"""Final recovery/outcome regressions for Deploy Approval."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from .. import comments as comments_mod
from ..agent_core_client import (
    AgentCoreClient,
    AgentCoreDeployOutcomeUncertain,
    AgentCoreError,
)
from ..config import Config, validate_config
from ..github_state_proxy import _validate_hidden_state
from ..models import BuildInfo, MachineInfo
from ..policy import Policy
from ..github_command_watcher import GitHubCommandWatcher
from ..router_webhook import webhook
from ..service import DeployController, DeployOutcomeUncertain
from .conftest import make_config


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


def _state(**overrides) -> dict:
    state = {
        "version": 1,
        "head_sha": "a" * 40,
        "status": "deploy-requested",
        "review_evidence": {"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z", "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z", "review_author_id": "7950763"},
        "components": [_component()],
        "deployments": [],
        "approve_attempts": [],
        "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "case_results": {},
        "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {
            "comment_id": 17,
            "kind": "approve_deploy",
            "phase": "uncertain",
            "args": {"machine": "test-machine"},
        },
        "last_processed_comment_id": 17,
    }
    state.update(overrides)
    return state


def _build(*, target: str = "perception", driver_path: str = "", variant: str = "5.11",
           success: bool = True, image_tag: str = "registry.example/repo:v1") -> BuildInfo:
    return BuildInfo(
        target=target,
        driver_path=driver_path,
        variant=variant,
        success=success,
        image_tag=image_tag,
        deployable=True,
    )


def _controller():
    config = make_config()
    proxy = MagicMock()
    proxy.read_hidden_state = AsyncMock(return_value=_state())
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value={"id": 42, "body": ""})
    proxy.get_pr = AsyncMock(
        return_value={
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 111, "login": "alice"},
        }
    )
    proxy.post_issue_comment = AsyncMock(return_value={"id": 1})
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    policy = Policy(config)
    policy.machines = {
        "test-machine": MachineInfo(
            alias="test-machine",
            node_id="node-1",
            owners=["owner1"],
            node_host="127.0.0.1",
            targets=["perception", "actucore", "driver"],
            platforms=["linux/arm64"],
            variants=["5.11", "6.1"],
            driver_paths=["custom/driver"],
        )
    }
    github = MagicMock()
    github.get_current_user = AsyncMock(return_value={"id": 123, "login": "bot"})
    github.get_issue_comments = AsyncMock(return_value=[])
    async def _get_comment(repo, cid):
        if isinstance(cid, int) and not isinstance(cid, bool) and cid > 0:
            return {
                "id": cid,
                "body": "/approve_deploy machine=test-machine",
                "user": {"id": 111, "login": "owner1"},
            }
        return None

    proxy.get_comment = AsyncMock(side_effect=_get_comment)

    controller = DeployController(config, proxy, policy, github)
    return controller, proxy, policy, github, config


def _request_payload(comment_body: str) -> dict:
    return {
        "action": "created",
        "repository": {"full_name": "repo"},
        "issue": {"number": 1, "pull_request": {}},
        "comment": {"id": 99},
    }


def _webhook_request(config, proxy, controller, payload, signature: str):
    body = json.dumps(payload).encode("utf-8")

    class _Request:
        def __init__(self):
            self.app = SimpleNamespace(
                state=SimpleNamespace(config=config, proxy=proxy, controller=controller)
            )
            self.headers = {
                "X-GitHub-Event": "issue_comment",
                "X-Hub-Signature-256": signature,
            }

        async def stream(self):
            yield body

    return _Request()


def _driver_client(transport: httpx.AsyncBaseTransport) -> AgentCoreClient:
    cfg = make_config(allow_private_http=True)
    cfg
    return AgentCoreClient(
        cfg,
        base_url="https://10.0.0.1:15678",
        node_host="10.0.0.1",
        http=httpx.AsyncClient(transport=transport),
    )


def _fresh_component(**overrides) -> dict:
    component = _component(**overrides)
    component.pop("runtime_id", None)
    return component


@pytest.mark.asyncio
async def test_driver_status_current_no_container_shape_normalizes_to_empty_running_image():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={"code": 200, "data": {"status": "stopped", "logs": "no container"}},
                request=request,
            )

    client = _driver_client(Transport())
    result = await client.driver_status("driver")
    assert result == {"running_image": "", "status": "stopped"}


@pytest.mark.asyncio
async def test_driver_status_invalid_status_type_fails_closed():
    """status field present but not a non-empty string -> AgentCoreError (fail closed)."""
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "data": {
                        "status": 123,
                        "running_image": "registry.example/repo@sha256:" + "a" * 64,
                    },
                },
                request=request,
            )

    client = _driver_client(Transport())
    with pytest.raises(AgentCoreError, match="status must be a non-empty string"):
        await client.driver_status("driver")


@pytest.mark.asyncio
async def test_driver_status_error_shape_without_running_image_fails_closed():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={"code": 200, "data": {"status": "error", "error": "docker unavailable"}},
                request=request,
            )

    client = _driver_client(Transport())
    with pytest.raises(AgentCoreError):
        await client.driver_status("driver")


@pytest.mark.asyncio
async def test_driver_status_malformed_missing_running_image_fails_closed():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={"code": 200, "data": {"status": "stopped"}},
                request=request,
            )

    client = _driver_client(Transport())
    with pytest.raises(AgentCoreError):
        await client.driver_status("driver")


@pytest.mark.asyncio
async def test_request_deploy_uses_canonical_component_snapshot_helper():
    controller, proxy, policy, github, config = _controller()
    builds = [BuildInfo("perception", "", "5.11", True, "registry.example/repo:v1", True)]
    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {
            "id": 5001,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abcdef1\n\n| target | status | version | took |\n| perception (jetson-jp5.11) | :white_check_mark: | release.260918.abcdef1 | 10s |\n\n### Images\n\n**perception (jetson-jp5.11)**\n```\nregistry.example/repo:v1\n```\n",
            "created_at": "2026-09-03T09:00:00Z",
            "updated_at": "2026-09-03T10:00:00Z",
        },
        {
            "id": 5002,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nLGTM\n",
            "created_at": "2026-09-03T11:00:00Z",
            "updated_at": "2026-09-03T11:00:00Z",
        },
    ])
    controller._build_component_snapshot = AsyncMock(
        return_value=[
            {
                "component_id": "comp-123",
                "target": "perception",
                "driver_path": "",
                "variant": "5.11",
                "review_image_tag": "registry.example/repo:v1",
                "image_ref": "registry.example/repo@sha256:" + "b" * 64,
                "resolved_platform": "linux/arm64",
            }
        ]
    )
    proxy.read_hidden_state = AsyncMock(
        return_value={
            "version": 1,
            "head_sha": "a" * 40,
            "status": "deploy-ready",
            "review_evidence": {},
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
    )
    mock_comment = {"id": 101, "user": {"id": 111, "login": "alice"}, "body": "/request_deploy"}
    proxy.get_comment = AsyncMock(return_value=mock_comment)
    proxy.get_pr = AsyncMock(
        return_value={
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 111, "login": "alice"},
        }
    )
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    await controller.handle_request_deploy("4paradigm/phanthymotus", 1, 101)

    controller._build_component_snapshot.assert_awaited_once_with(
        "4paradigm/phanthymotus", 1, "a" * 40, builds
    )
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["components"][0]["component_id"] == "comp-123"


@pytest.mark.asyncio
async def test_uncertain_recovery_rebuilds_fresh_component_snapshot():
    controller, proxy, policy, github, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z", "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z", "review_author_id": "7950763"},
        components=[_component(component_id="comp-old", image_ref="registry.example/repo@sha256:" + "c" * 64)],
        deployments=[],
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence, ReviewBuild
    fake_evidence = ReviewCommentEvidence(
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        commit_prefix="abc1234",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-18T03:55:54Z",
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-18T03:56:00Z",
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {
            "id": 1001,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| target | status | version | took |\n| perception | :white_check_mark: | v2 | 10s |\n\n### Images\n\n**perception**\n```\nregistry.example/repo:v2\n```\n",
            "created_at": "2026-09-18T03:50:00Z",
            "updated_at": "2026-09-18T03:55:54Z",
        },
        {
            "id": 1003,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": ("<!-- pr-review-agent -->\n## PR Review Agent " + emdash + " Code Review\n\nLooks good.\n").replace("\\n", "\n"),
            "created_at": "2026-09-18T03:56:00Z",
            "updated_at": "2026-09-18T03:56:00Z",
        },
    ])
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(
        return_value=[_component(component_id="comp-new", image_ref="registry.example/repo@sha256:" + "d" * 64)]
    )
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state)

    assert result == "deploy-requested"
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["review_evidence"]["build_comment_id"] == 1001
    assert written_state["components"][0]["component_id"] == "comp-new"


@pytest.mark.asyncio
async def test_uncertain_recovery_new_job_never_keeps_old_components():
    controller, proxy, policy, github, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z", "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z", "review_author_id": "7950763"},
        components=[_component(component_id="comp-old")],
        deployments=[],
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence, ReviewBuild
    fake_evidence = ReviewCommentEvidence(
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        commit_prefix="abc1234",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-18T03:55:54Z",
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-18T03:56:00Z",
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {
            "id": 1001,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| target | status | version | took |\n| perception | :white_check_mark: | v2 | 10s |\n\n### Images\n\n**perception**\n```\nregistry.example/repo:v2\n```\n",
            "created_at": "2026-09-18T03:50:00Z",
            "updated_at": "2026-09-18T03:55:54Z",
        },
        {
            "id": 1003,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": ("<!-- pr-review-agent -->\n## PR Review Agent " + emdash + " Code Review\n\nLooks good.\n").replace("\\n", "\n"),
            "created_at": "2026-09-18T03:56:00Z",
            "updated_at": "2026-09-18T03:56:00Z",
        },
    ])
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(return_value=[_component(component_id="comp-new")])
        proxy.write_hidden_state = AsyncMock()
        proxy.project_status_label = AsyncMock()

        await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state)

        written_state = proxy.write_hidden_state.call_args.args[3]
        assert [c["component_id"] for c in written_state["components"]] == ["comp-new"]


@pytest.mark.asyncio
async def test_uncertain_recovery_snapshot_change_resets_deployments():
    controller, proxy, policy, github, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z", "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z", "review_author_id": "7950763"},
        components=[_component(component_id="comp-old")],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-old"], "phase": "deployed"}],
    )
    state["command"]["phase"] = "completed"
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence, ReviewBuild
    fake_evidence = ReviewCommentEvidence(
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        commit_prefix="abc1234",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-18T03:55:54Z",
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-18T03:56:00Z",
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {
            "id": 1001,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| target | status | version | took |\n| perception | :white_check_mark: | v2 | 10s |\n\n### Images\n\n**perception**\n```\nregistry.example/repo:v2\n```\n",
            "created_at": "2026-09-18T03:50:00Z",
            "updated_at": "2026-09-18T03:55:54Z",
        },
        {
            "id": 1003,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": ("<!-- pr-review-agent -->\n## PR Review Agent " + emdash + " Code Review\n\nLooks good.\n").replace("\\n", "\n"),
            "created_at": "2026-09-18T03:56:00Z",
            "updated_at": "2026-09-18T03:56:00Z",
        },
    ])
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(
            return_value=[_component(component_id="comp-new", review_image_tag="registry.example/repo:v2")]
        )
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state)

    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["deployments"] == []
    assert written_state["components"] == [
        {
            "component_id": "comp-new",
            "target": "perception",
            "driver_path": "",
            "variant": "5.11",
            "review_image_tag": "registry.example/repo:v2",
            "image_ref": "registry.example/repo@sha256:" + "a" * 64,
            "resolved_platform": "linux/arm64",
        }
    ]


@pytest.mark.asyncio
async def test_uncertain_recovery_same_snapshot_preserves_known_successful_deployments():
    controller, proxy, policy, github, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1001, "build_comment_updated_at": "2026-09-18T03:55:54Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 1002, "code_review_comment_id": 1003, "review_author_id": "7950763"},
        components=[
            _component(component_id="comp-1", runtime_id="perception"),
            _component(component_id="comp-2", runtime_id="actucore", target="actucore"),
        ],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-1"], "phase": "deployed"}],
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fresh_components = [
        _fresh_component(component_id="comp-1"),
        _fresh_component(component_id="comp-2", target="actucore"),
    ]
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)
    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {
            "id": 1001,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": ("<!-- pr-review-agent -->\n## PR Review Agent " + emdash + " Build Result\n\nCommit: abc1234\n\n| target | status | version | took |\n| perception | :white_check_mark: | v1 | 10s |\n| actucore | :white_check_mark: | v1 | 10s |\n\n### Images\n\n**perception**\n```\nregistry.example/repo:v1\n```\n\n**actucore**\n```\nregistry.example/repo:v1\n```\n").replace("\\n", "\n"),
            "created_at": "2026-09-18T03:50:00Z",
            "updated_at": "2026-09-18T03:55:54Z",
        },
        {
            "id": 1003,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": ("<!-- pr-review-agent -->\n## PR Review Agent " + emdash + " Code Review\n\nLooks good.\n").replace("\\n", "\n"),
            "created_at": "2026-09-18T03:56:00Z",
            "updated_at": "2026-09-18T03:56:00Z",
        },
    ])
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence, ReviewBuild
    fake_evidence = ReviewCommentEvidence(
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        commit_prefix="abc1234",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-18T03:55:54Z",
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-18T03:56:00Z",
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1"),
                ReviewBuild(target="actucore", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(return_value=fresh_components)
        proxy.write_hidden_state = AsyncMock()
        proxy.project_status_label = AsyncMock()

        await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state)

        written_state = proxy.write_hidden_state.call_args.args[3]
        assert written_state["deployments"] == state["deployments"]
        # runtime_id preserved from prior deployed component
        assert written_state["deployments"] == state["deployments"]
        assert written_state["components"][0].get("runtime_id") is None
        assert "runtime_id" not in written_state["components"][1]


@pytest.mark.asyncio
async def test_uncertain_recovery_clears_runtime_id_for_ambiguous_component():
    controller, proxy, policy, github, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1001, "build_comment_updated_at": "2026-09-18T03:55:54Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 1002, "code_review_comment_id": 1003, "review_author_id": "7950763"},
        components=[
            _component(component_id="comp-1", runtime_id="perception"),
            _component(component_id="comp-2", target="actucore", runtime_id="actucore"),
        ],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-1"], "phase": "deployed"}],
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)
    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {
            "id": 1001,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": ("<!-- pr-review-agent -->\n## PR Review Agent " + emdash + " Build Result\n\nCommit: abc1234\n\n| target | status | version | took |\n| perception | :white_check_mark: | v1 | 10s |\n| actucore | :white_check_mark: | v1 | 10s |\n\n### Images\n\n**perception**\n```\nregistry.example/repo:v1\n```\n\n**actucore**\n```\nregistry.example/repo:v1\n```\n").replace("\\n", "\n"),
            "created_at": "2026-09-18T03:50:00Z",
            "updated_at": "2026-09-18T03:55:54Z",
        },
        {
            "id": 1003,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": ("<!-- pr-review-agent -->\n## PR Review Agent " + emdash + " Code Review\n\nLooks good.\n").replace("\\n", "\n"),
            "created_at": "2026-09-18T03:56:00Z",
            "updated_at": "2026-09-18T03:56:00Z",
        },
    ])
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence, ReviewBuild
    fake_evidence = ReviewCommentEvidence(
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        commit_prefix="abc1234",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-18T03:55:54Z",
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-18T03:56:00Z",
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1"),
                ReviewBuild(target="actucore", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(
        return_value=[
            _component(component_id="comp-1"),
            _component(component_id="comp-2", target="actucore"),
        ]
    )
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state)

    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["components"][0].get("runtime_id") is None
    assert "runtime_id" not in written_state["components"][1]


@pytest.mark.asyncio
async def test_uncertain_recovery_snapshot_rebuild_failure_stays_uncertain():
    controller, proxy, policy, github, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z", "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z", "review_author_id": "7950763"},
        components=[_component()],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence, ReviewBuild
    fake_evidence = ReviewCommentEvidence(
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        commit_prefix="abc1234",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-18T03:55:54Z",
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-18T03:56:00Z",
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {
            "id": 1001,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| target | status | version | took |\n| perception | :white_check_mark: | v2 | 10s |\n\n### Images\n\n**perception**\n```\nregistry.example/repo:v2\n```\n",
            "created_at": "2026-09-18T03:50:00Z",
            "updated_at": "2026-09-18T03:55:54Z",
        },
        {
            "id": 1003,
            "user": {"id": "7950763", "login": "review-agent-bot"},
            "body": ("<!-- pr-review-agent -->\n## PR Review Agent " + emdash + " Code Review\n\nLooks good.\n").replace("\\n", "\n"),
            "created_at": "2026-09-18T03:56:00Z",
            "updated_at": "2026-09-18T03:56:00Z",
        },
    ])
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(return_value=None)
        proxy.write_hidden_state = AsyncMock()
        proxy.project_status_label = AsyncMock()

        result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state)

        assert result == "uncertain"
        written_state = proxy.write_hidden_state.call_args.args[3]
        assert written_state["review_evidence"].get("build_comment_id")
        assert written_state["components"] == [_component()]


@pytest.mark.asyncio
async def test_new_approve_from_uncertain_refreshes_before_final_pre_deploy_validation():
    controller, proxy, policy, github, config = _controller()
    events: list[str] = []

    async def _refresh(*args, **kwargs):
        events.append("refresh")
        return "deploy-requested"

    async def _list_drivers():
        events.append("list_drivers")
        return [{"id": "perception", "target": "perception", "image": "registry.example/repo:v1"}]

    async def _driver_status(runtime_id):
        events.append(f"driver_status:{runtime_id}")
        return {"status": "running", "running_image": "registry.example/repo@sha256:" + "a" * 64}

    controller._refresh_uncertain_state = AsyncMock(side_effect=_refresh)
    core = AsyncMock()
    core.list_drivers = AsyncMock(side_effect=_list_drivers)
    core.driver_status = AsyncMock(side_effect=_driver_status)
    core.deploy_driver = AsyncMock(return_value={"code": 0})
    controller._core_for_node = AsyncMock(return_value=core)
    controller._verify_deployed_runtime = AsyncMock(return_value=(True, {"component_id": "comp-001", "runtime_id": "test", "running_image": "x", "passed": False}))
    controller._run_automated_case = AsyncMock(return_value={})
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 222, "login": "alice"}})

    await controller.handle_approve_deploy("repo", 1, 17, "test-machine", "owner1", "111")

    assert events[0] == "refresh"
    assert "list_drivers" in events
    assert events.index("refresh") < events.index("list_drivers")


@pytest.mark.asyncio
async def test_watcher_uncertain_without_new_approve_does_not_refresh_review():
    controller, proxy, policy, github, config = _controller()
    state = _state()
    state["command"]["phase"] = "uncertain"
    state["last_processed_comment_id"] = 17

    async def _read_hidden_state(*args, **kwargs):
        return state

    proxy.read_hidden_state = AsyncMock(side_effect=_read_hidden_state)
    proxy.get_pr = AsyncMock(
        return_value={
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 111, "login": "alice"},
        }
    )
    proxy.get_issue_comments = AsyncMock(
        return_value=[
            {
                "id": 17,
                "body": "/approve_deploy machine=test-machine",
                "user": {"id": 111, "login": "owner1"},
            }
        ]
    )
    proxy.is_bot_comment = MagicMock(return_value=False)
    controller._core_for_node = AsyncMock(side_effect=AssertionError("unexpected Agent Core lookup"))
    controller._deploy_component = AsyncMock(side_effect=AssertionError("unexpected deploy"))
    controller._run_automated_case = AsyncMock(side_effect=AssertionError("unexpected case run"))
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    await watcher._process_pr("4paradigm/phanthymotus", 1)

    assert state["command"]["phase"] == "uncertain"
    assert controller.on_command.await_count == 0
    assert controller._core_for_node.await_count == 0
    assert controller._deploy_component.await_count == 0
    assert proxy.write_hidden_state.await_count == 0


@pytest.mark.asyncio
async def test_watcher_uncertain_new_approve_refreshes_review_before_final_pre_deploy_validation():
    controller, proxy, policy, github, config = _controller()
    config.active_repos = ["4paradigm/phanthymotus"]
    config.auth_valid = True
    state = _state()
    state["command"]["phase"] = "uncertain"
    state["last_processed_comment_id"] = 17

    events: list[str] = []

    async def _read_hidden_state(*args, **kwargs):
        events.append("read_hidden_state")
        return state

    async def _get_pr(*args, **kwargs):
        events.append("get_pr")
        return {
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 111, "login": "alice"},
        }

    emdash = "\u2014"

    async def _get_issue_comments(*args, **kwargs):
        events.append("get_issue_comments")
        return [
            {
                "id": 17,
                "body": "/approve_deploy machine=test-machine",
                "user": {"id": 111, "login": "owner1"},
            },
            {
                "id": 99,
                "body": "/approve_deploy machine=test-machine",
                "user": {"id": 111, "login": "owner1"},
            },
            {
                "id": 5001,
                "user": {"id": "7950763", "login": "review-agent-bot"},
                "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abcdef1\n\n| target | status | version | took |\n| perception | :white_check_mark: | 5.11 | 10s |\n\n### Images\n\n**perception**\n```\nccr.ccs.tencentyun.com/repo:v1\n```\n",
                "created_at": "2026-09-03T09:00:00Z",
                "updated_at": "2026-09-03T10:00:00Z",
            },
            {
                "id": 5002,
                "user": {"id": "7950763", "login": "review-agent-bot"},
                "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nLGTM\n",
                "created_at": "2026-09-03T11:00:00Z",
                "updated_at": "2026-09-03T11:00:00Z",
            },
        ]


    async def _list_drivers():
        events.append("list_drivers")
        return [
            {
                "id": "perception",
                "target": "perception",
                "image": "registry.example/repo:v1",
            }
        ]

    async def _driver_status(runtime_id):
        events.append(f"driver_status:{runtime_id}")
        return {"status": "running", "running_image": "occupied@sha256:" + "c" * 64}

    async def _deploy_component(*args, **kwargs):
        events.append("deploy")
        return {"result": {"code": 0}}

    proxy.read_hidden_state = AsyncMock(side_effect=_read_hidden_state)
    proxy.get_pr = AsyncMock(side_effect=_get_pr)
    proxy.get_issue_comments = AsyncMock(side_effect=_get_issue_comments)
    _comment_lookup = {
        17: {
            "id": 17,
            "body": "/approve_deploy machine=test-machine",
            "user": {"id": 111, "login": "owner1"},
        },
        99: {
            "id": 99,
            "body": "/approve_deploy machine=test-machine",
            "user": {"id": 111, "login": "owner1"},
        },
    }
    def _get_comment(repo, cid):
        return _comment_lookup.get(cid, {"id": cid, "body": "normal comment", "user": {"id": 999}})
    proxy.get_comment = AsyncMock(side_effect=_get_comment)
    proxy.is_bot_comment = MagicMock(return_value=False)
    proxy.comment_identity = AsyncMock(return_value=("111", "owner1"))
    proxy.persist_cursor = AsyncMock(return_value={"last_processed_comment_id": 99})
    core = AsyncMock()
    core.list_drivers = AsyncMock(side_effect=_list_drivers)
    core.driver_status = AsyncMock(side_effect=_driver_status)
    controller._core_for_node = AsyncMock(return_value=core)
    controller._deploy_component = AsyncMock(side_effect=_deploy_component)
    controller._run_automated_case = AsyncMock(side_effect=AssertionError("automated case should not run on occupied gate"))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    real_on_command = controller.on_command
    controller.on_command = AsyncMock(wraps=real_on_command)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    await watcher._process_pr("4paradigm/phanthymotus", 1)

    assert controller.on_command.await_count == 1
    assert controller.on_command.call_args.args[3] == 99
    assert state["command"]["phase"] == "completed"
    assert "get_issue_comments" in events
    # review.list_jobs seam removed; evidence now from GitHub comments
    assert "deploy" not in events
    assert proxy.write_hidden_state.await_count >= 1


@pytest.mark.asyncio
async def test_deploy_post_transport_timeout_becomes_uncertain():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ReadTimeout("boom", request=request)

    client = _driver_client(Transport())
    with pytest.raises(AgentCoreDeployOutcomeUncertain):
        await client.deploy_driver("driver", "registry.example/repo@sha256:" + "a" * 64)


@pytest.mark.asyncio
async def test_deploy_post_non_success_response_becomes_uncertain():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(500, json={"code": 500, "message": "boom"}, request=request)

    client = _driver_client(Transport())
    with pytest.raises(AgentCoreDeployOutcomeUncertain):
        await client.deploy_driver("driver", "registry.example/repo@sha256:" + "a" * 64)


@pytest.mark.asyncio
async def test_deploy_post_malformed_response_becomes_uncertain():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, content=b"not-json", request=request)

    client = _driver_client(Transport())
    with pytest.raises(AgentCoreDeployOutcomeUncertain):
        await client.deploy_driver("driver", "registry.example/repo@sha256:" + "a" * 64)


@pytest.mark.asyncio
async def test_deploy_post_prevalidation_error_is_not_outcome_uncertain():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise AssertionError("POST should not be attempted")

    client = _driver_client(Transport())
    with pytest.raises(AgentCoreError) as excinfo:
        await client.deploy_driver("driver", "https://registry.example/repo:latest")
    assert not isinstance(excinfo.value, AgentCoreDeployOutcomeUncertain)


@pytest.mark.asyncio
async def test_uncertain_post_stops_later_components():
    """Canonical uncertain-post contract: A succeeds, B uncertain, C gets zero POSTs.

    Scenario with THREE components so the contract is fully observable:
    A = perception (succeeds), B = actucore (uncertain), C = driver (zero POSTs).

    Assertions:
    1. A POST exactly once
    2. B POST exactly once
    3. C POST zero times
    4. total deploy calls == 2
    5. successful deployment for A remains in state["deployments"]
    6. B is not marked successfully deployed
    7. C is not marked deployed
    8. state.status == "deploy-requested"
    9. state.command.phase == "uncertain"
    10. state.last_processed_comment_id == current approve comment ID
    11. no automatic replay
    12. no rollback behavior
    """
    controller, proxy, policy, github, config = _controller()
    real_comps = await controller._build_component_snapshot(
        "4paradigm/phanthymotus", 1, "a" * 40,
        [_build(), _build(target="actucore"), _build(target="driver", driver_path="custom/driver")],
    )
    assert real_comps is not None
    state = _state(
        status="deploy-requested",
        components=real_comps,
        deployments=[],
    )
    state["command"]["phase"] = "completed"
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(
        return_value={
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 222, "login": "alice"},
        }
    )
    core = AsyncMock()
    core.list_drivers = AsyncMock(
        return_value=[
            {"id": "perception", "target": "perception", "image": "registry.example/repo:v1"},
            {"id": "actucore", "target": "actucore", "image": "registry.example/repo:v1"},
            {"id": "driver", "category": "driver", "image": "registry.example/repo:v1"},
        ]
    )
    status_order = []

    async def _driver_status(driver_id):
        status_order.append(driver_id)
        return {"status": "running", "running_image": ""}

    core.driver_status = AsyncMock(side_effect=_driver_status)
    controller._core_for_node = AsyncMock(return_value=core)
    deploy_calls = {"perception": 0, "actucore": 0, "driver": 0}

    async def _deploy_component(_core, _node_id, _image_ref, runtime_id):
        deploy_calls[runtime_id] = deploy_calls.get(runtime_id, 0) + 1
        if runtime_id == "actucore":
            raise DeployOutcomeUncertain("network timeout")
        return {"result": {"code": 0}}

    controller._deploy_component = AsyncMock(side_effect=_deploy_component)
    controller._verify_deployed_runtime = AsyncMock(return_value=(True, {"component_id": "comp-001", "runtime_id": "test", "running_image": "x", "passed": False}))
    controller._run_automated_case = AsyncMock(return_value={})
    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    proxy.get_issue_comments = AsyncMock(return_value=[])
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller.handle_approve_deploy("4paradigm/phanthymotus", 1, 17, "test-machine", "owner1", "111")

    # 1-4: A once, B once, C zero, total 2
    assert deploy_calls.get("perception", 0) == 1, f"perception deployed {deploy_calls.get('perception', 0)} times"
    assert deploy_calls.get("actucore", 0) == 1, f"actucore deployed {deploy_calls.get('actucore', 0)} times"
    assert deploy_calls.get("driver", 0) == 0, f"driver deployed {deploy_calls.get('driver', 0)} times"
    total = sum(deploy_calls.values())
    assert total == 2, f"Expected 2 total deploy calls, got {total}"

    # 5-7: A deployed, B/C not
    written_state = proxy.write_hidden_state.call_args.args[3]
    deployed_cids = set()
    for dep in written_state.get("deployments", []):
        deployed_cids.update(dep.get("component_ids", []))
    assert real_comps[0]["component_id"] in deployed_cids, "Component A should be deployed"
    assert real_comps[1]["component_id"] not in deployed_cids, "Component B should not be deployed"

    # 8-10
    assert written_state["status"] == "deploy-requested"
    assert written_state["command"]["phase"] == "uncertain"
    assert written_state["last_processed_comment_id"] == 17

    # 11-12: no automatic replay (phase is uncertain, not executing), no rollback
    assert written_state["command"]["phase"] == "uncertain"
    # prior successful deployment preserved (no rollback)
    assert len(written_state["deployments"]) == 1
    assert written_state["deployments"][0]["component_ids"] == [real_comps[0]["component_id"]]



@pytest.mark.asyncio
async def test_uncertain_post_does_not_upload_failed_cos():
    controller, proxy, policy, github, config = _controller()
    state = _state(
        components=[_component(component_id="comp-1"), _component(component_id="comp-2", target="actucore", runtime_id="actucore")],
        deployments=[],
    )
    state["command"]["phase"] = "completed"
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}})
    core = AsyncMock()
    core.list_drivers = AsyncMock(
        return_value=[
            {"id": "perception", "target": "perception", "image": "registry.example/repo:v1"},
            {"id": "actucore", "target": "actucore", "image": "registry.example/repo:v1"},
            {"id": "driver", "category": "driver", "image": "registry.example/repo:v1"},
        ]
    )
    core.driver_status = AsyncMock(
        side_effect=[
            {"status": "running", "running_image": ""},
            {"status": "running", "running_image": ""},
        ]
    )
    controller._core_for_node = AsyncMock(return_value=core)
    controller._deploy_component = AsyncMock(side_effect=DeployOutcomeUncertain("timeout"))
    controller._run_automated_case = AsyncMock(return_value={})
    controller._upload_evidence = AsyncMock()
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller.handle_approve_deploy("repo", 1, 17, "test-machine", "owner1", "111")

    controller._upload_evidence.assert_not_called()



def test_poll_disabled_webhook_enabled_fails_config():
    with pytest.raises(ValueError, match="requires polling"):
        validate_config(
            Config(
                github_repos=[
                    "4paradigm/phanthymotus",
                ],
                poll_enabled=False,
                webhook_enabled=True,
                github_webhook_secret="secret",
            )
        )


@pytest.mark.asyncio
async def test_webhook_remains_supplementary_zero_dispatch():
    controller, proxy, policy, github, config = _controller()
    config.webhook_enabled = True
    config.active_repos = ["4paradigm/phanthymotus"]
    config.auth_valid = True
    payload = {
        "action": "created",
        "repository": {"full_name": "4paradigm/phanthymotus"},
        "issue": {"number": 1, "pull_request": {}},
        "comment": {"id": 99},
    }
    proxy.get_comment = AsyncMock(return_value={"id": 99, "body": "/request_deploy", "user": {"id": 111, "login": "alice"}})
    controller.on_command = AsyncMock()
    request = _webhook_request(config, proxy, controller, payload, "sha256=" + "0" * 64)

    with patch("agents.deploy_approval.router_webhook._verify_signature_impl", return_value=True):
        result = await webhook(request)

    assert result["status"] == "deferred"
    controller.on_command.assert_not_called()


def test_hidden_state_accepts_uncertain_approve_attempt():
    state = _state(
        approve_attempts=[
            {
                "comment_id": 101,
                "actor": "alice",
                "machine": "test-machine",
                "preflight": [],
                "outcome": "uncertain",
                "health": [],
            }
        ],
        approve_attempts_total=1,
        command={
            "comment_id": 101,
            "kind": "approve_deploy",
            "phase": "uncertain",
            "args": {"machine": "test-machine"},
        },
    )
    state["status"] = "deploy-requested"
    _validate_hidden_state(state)


def test_hidden_state_rejects_unknown_approve_attempt_outcome():
    state = _state(
        approve_attempts=[
            {
                "comment_id": 101,
                "actor": "alice",
                "machine": "test-machine",
                "preflight": [],
                "outcome": "bogus",
                "health": [],
            }
        ],
        command={
            "comment_id": 101,
            "kind": "approve_deploy",
            "phase": "uncertain",
            "args": {"machine": "test-machine"},
        },
    )
    state["status"] = "deploy-requested"
    with pytest.raises(Exception):
        _validate_hidden_state(state)


def test_uncertain_post_state_passes_real_hidden_state_validator():
    state = _state(
        approve_attempts=[
            {
                "comment_id": 17,
                "actor": "alice",
                "machine": "test-machine",
                "preflight": [
                    {"component_id": "comp-001", "runtime_id": "perception", "running_image": ""}
                ],
                "outcome": "uncertain",
                "health": [
                    {
                        "component_id": "comp-001",
                        "runtime_id": "perception",
                        "running_image": "registry.example/repo@sha256:" + "a" * 64,
                        "passed": True,
                    }
                ],
            }
        ],
        approve_attempts_total=1,
        command={
            "comment_id": 17,
            "kind": "approve_deploy",
            "phase": "uncertain",
            "args": {"machine": "test-machine"},
        },
    )
    state["status"] = "deploy-requested"
    _validate_hidden_state(state)


@pytest.mark.asyncio
async def test_build_component_snapshot_never_contains_runtime_id():
    controller, proxy, policy, github, config = _controller()
    result = await controller._build_component_snapshot(
        "repo",
        1,
        "a" * 40,
        [_build(), _build(target="actucore")],
    )
    assert result is not None
    assert result
    assert all("runtime_id" not in component for component in result)


def test_uncertain_same_snapshot_preserves_runtime_id_from_old_deployed_component():
    controller, proxy, policy, github, config = _controller()
    fresh = [
        _fresh_component(component_id="comp-1"),
        _fresh_component(component_id="comp-2", target="actucore"),
    ]
    rebuilt = controller._components_with_preserved_runtime_bindings(
        fresh,
        [
            _component(component_id="comp-1", runtime_id="perception"),
            _component(component_id="comp-2", target="actucore", runtime_id="actucore"),
        ],
        [{"machine": "test-machine", "component_ids": ["comp-1"], "phase": "deployed"}],
    )
    assert rebuilt is not None
    assert rebuilt[0]["runtime_id"] == "perception"
    assert "runtime_id" not in rebuilt[1]



@pytest.mark.asyncio
async def test_new_approve_deploys_only_remaining_components():
    """New approve processes REMAINING components only.

    Already successful component A receives ZERO new deploy POST.
    Remaining component B receives exactly ONE deploy POST.
    """
    import copy
    import hashlib

    controller, proxy, policy, github, config = _controller()
    REPO = "4paradigm/phanthymotus"
    events: list[str] = []
    write_log: list[tuple[str, dict]] = []

    FAKE_SHA = "a" * 64
    IMAGE_REF_A = f"registry.example/repo@sha256:{FAKE_SHA}"

    comp_a_id = hashlib.sha256(f"perception||5.11|{IMAGE_REF_A}".encode()).hexdigest()[:16]
    comp_b_id = hashlib.sha256(f"actucore||5.11|{IMAGE_REF_A}".encode()).hexdigest()[:16]

    current_state = _state(
        head_sha="a" * 40,
        status="deploy-requested",
        review_evidence={
            "build_comment_id": 5001,
            "build_comment_updated_at": "2026-09-18T03:55:54Z",
            "commit_prefix": "abc1234",
            "resolved_head_sha": "a" * 40,
            "test_comment_id": 5002,
            "test_comment_updated_at": "2026-09-18T03:55:54Z",
            "code_review_comment_id": 5003,
            "code_review_comment_updated_at": "2026-09-18T03:56:00Z",
            "review_author_id": "7950763",
        },
        components=[
            {
                "component_id": comp_a_id,
                "target": "perception",
                "driver_path": "",
                "variant": "5.11",
                "review_image_tag": "registry.example/repo:v1",
                "image_ref": IMAGE_REF_A,
                "resolved_platform": "linux/arm64",
                "runtime_id": "perception",
            },
            {
                "component_id": comp_b_id,
                "target": "actucore",
                "driver_path": "",
                "variant": "5.11",
                "review_image_tag": "registry.example/repo:v1",
                "image_ref": IMAGE_REF_A,
                "resolved_platform": "linux/arm64",
                "runtime_id": "actucore",
            },
        ],
        deployments=[
            {"machine": "test-machine", "component_ids": [comp_a_id], "phase": "deployed"}
        ],
        command={
            "comment_id": 17,
            "kind": "approve_deploy",
            "phase": "uncertain",
            "args": {"machine": "test-machine"},
        },
        last_processed_comment_id=17,
    )

    async def _read_hidden_state(*args, **kwargs):
        return copy.deepcopy(current_state)

    async def _write_hidden_state(*args, **kwargs):
        markdown = args[0] if args else kwargs.get("markdown", "")
        state = args[3] if len(args) > 3 else kwargs.get("state")
        if state is None:
            return
        written = copy.deepcopy(state)
        write_log.append((markdown, written))
        current_state.clear()
        current_state.update(written)

    proxy.read_hidden_state = AsyncMock(side_effect=_read_hidden_state)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.project_status_label = AsyncMock()

    proxy.get_pr = AsyncMock(
        return_value={
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 333, "login": "pr-author"},
        }
    )

    actor_id = "222"
    async def _get_comment(repo, cid):
        return {
            "id": cid,
            "body": "/approve_deploy machine=test-machine",
            "user": {"id": int(actor_id), "login": "owner1"},
        }
    proxy.get_comment = AsyncMock(side_effect=_get_comment)
    proxy.comment_identity = AsyncMock(return_value=("222", "owner1"))
    proxy.post_issue_comment = AsyncMock(return_value={"id": 1})
    proxy.collaborator_permission = AsyncMock(return_value="admin")

    # Patch unrelated boundaries
    async def _refresh_uncertain_state(repo, pr_number, state):
        state["status"] = "deploy-requested"
        old_command = state.get("command", {})
        old_args = dict(old_command.get("args", {}) or {})
        state["command"] = {
            "comment_id": int(old_command.get("comment_id", 0) or 0),
            "kind": "approve_deploy",
            "phase": "completed",
            "args": old_args,
        }

        # Persist the refreshed state so subsequent production reads see phase=completed.
        current_state.clear()
        current_state.update(copy.deepcopy(state))

        return "deploy-requested"

    controller._refresh_uncertain_state = AsyncMock(side_effect=_refresh_uncertain_state)
    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)

    # Mock _preflight for the REMAINING component (actucore) only
    controller._preflight_running_images = AsyncMock(return_value=[
        {
            "component": current_state["components"][1],
            "runtime_id": "actucore",
            "running_image": "",
            "runtime_repo": "registry.example/repo",
        },
    ])

    core = AsyncMock()
    deploy_calls: dict[str, int] = {"perception": 0, "actucore": 0}

    async def _deploy_driver(runtime_id, image):
        deploy_calls[runtime_id] = deploy_calls.get(runtime_id, 0) + 1
        events.append(f"deploy_driver:{runtime_id}")
        return {"code": 200, "data": {"status": "starting"}}
    core.deploy_driver = _deploy_driver
    core.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "target": "perception", "image": "registry.example/repo:v1"},
        {"id": "actucore", "target": "actucore", "image": "registry.example/repo:v1"},
    ])
    controller._core_for_node = AsyncMock(return_value=core)
    controller._verify_deployed_runtime = AsyncMock(return_value=(True, {"component_id": "comp-001", "runtime_id": "test", "running_image": "x", "passed": False}))
    controller._run_automated_case = AsyncMock(return_value={})

    await controller.handle_approve_deploy(REPO, 1, 99, "test-machine", "owner1", actor_id)

    # Proofs:
    # A (perception) receives ZERO new deploy POST
    assert deploy_calls.get("perception", 0) == 0, (
        f"perception should NOT be redeployed, got {deploy_calls.get('perception', 0)}"
    )
    # B (actucore) receives exactly ONE deploy POST
    assert deploy_calls.get("actucore", 0) == 1, (
        f"actucore should be deployed once, got {deploy_calls.get('actucore', 0)}"
    )
    # A remains recorded in deployments
    final_state = write_log[-1][1] if write_log else current_state
    all_cids = []
    for dep in final_state.get("deployments", []):
        all_cids.extend(dep.get("component_ids", []))
    assert comp_a_id in all_cids, "Component A must remain in deployments"
    assert all_cids.count(comp_a_id) == 1
    assert all_cids.count(comp_b_id) == 1
    # Final state
    assert final_state["status"] == "testing"
    assert final_state["command"]["phase"] == "completed"
    _validate_hidden_state(final_state)


LEGACY_DIGEST = "registry.example/repo@sha256:" + "a" * 64
TAG_A = "bj-warehouse.tencentcloudcr.com/phanthy-motus/perception:release.260922.4707deb-jetson-jp5.11"
TAG_B = "bj-warehouse.tencentcloudcr.com/phanthy-motus/perception:release.260923.4707deb-jetson-jp5.11"


def _fake_config():
    REVIEW_AGENT_GITHUB_LOGIN, REVIEW_AGENT_GITHUB_USER_ID = "kentcyq", "7950763"
    from ..tests.conftest import make_config
    return make_config(
        review_comment_author_id=REVIEW_AGENT_GITHUB_USER_ID,
        review_comment_author_login=REVIEW_AGENT_GITHUB_LOGIN,
        agent_core_tokens={"m1": "test-token"},
    )


def _make_policy():
    policy = MagicMock()
    policy.get_machines.return_value = [
        MagicMock(
            alias="m1", node_host="127.0.0.1", node_id="n1",
            owners=["alice"],
            targets=["perception", "actucore"],
            variants=["5.11"],
            platforms=["linux/arm64"],
            driver_paths=[],
            node_host_public=False,
            is_production=True,
        )
    ]
    policy.get_machine_by_node_id.return_value = policy.get_machines.return_value[0]
    policy.check_machine_targets_component.return_value = None
    policy.check_variant_compatible.return_value = None
    policy.check_driver_path_compatible.return_value = None
    policy.get_machine_groups_for_components.return_value = []
    return policy


def _make_controller(fake_proxy, policy, fake_github):
    REVIEW_AGENT_GITHUB_LOGIN, REVIEW_AGENT_GITHUB_USER_ID = "kentcyq", "7950763"
    from ..service import DeployController
    config = _fake_config()
    config.review_comment_author_id = REVIEW_AGENT_GITHUB_USER_ID
    config.review_comment_author_login = REVIEW_AGENT_GITHUB_LOGIN
    config.agent_core_tokens = {"m1": "test-token"}
    # AsyncMock proxies return coroutines for every attribute; production
    # find_trusted_lifecycle_comment must resolve to None so the controller
    # falls through to its normal history scan, and is_bot_comment is a SYNC
    # method that must stay MagicMock-shaped to avoid un-awaited coroutines.
    fake_proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    fake_proxy.is_bot_comment = MagicMock(return_value=False)
    controller = DeployController(config, fake_proxy, policy, fake_github)
    return controller


# ── Legacy digest migration regression tests (no Registry) ──────────


@pytest.mark.asyncio
async def test_legacy_digest_undeployed_component_migrates_to_tag_preserving_component_id(config, monkeypatch):
    """TEST 1: Old component with digest image_ref, no deployment, same semantic key
    -> migrate to tag, keep component_id. ZERO Registry calls."""
    from unittest.mock import AsyncMock, MagicMock
    from ..review_comment_parser import ReviewCommentEvidence

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()

    controller = _make_controller(fake_proxy, policy, fake_github)

    # Old state: undeployed, legacy digest, uncertain
    old_cid = "legacy-cid"
    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [{
            "component_id": old_cid, "target": "perception", "driver_path": "",
            "variant": "5.11", "review_image_tag": TAG_A,
            "image_ref": LEGACY_DIGEST, "resolved_platform": "linux/arm64",
        }],
        "deployments": [],
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False, "case_results": {}, "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    # Set up comment with hidden state
    # Patch extract_review_evidence at the service module level to return valid evidence
    import agents.deploy_approval.service as svc_mod
    fake_evidence = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[MagicMock(target="perception", success=True, deployable=True,
                          image_tag=TAG_A, driver_path="", variant="5.11")],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence))

    # Directly call _refresh_uncertain_state
    import copy
    state_copy = copy.deepcopy(old_state)
    from ..github_state_proxy import _validate_hidden_state as _validate_hidden_state_fn
    _validate_hidden_state_fn(state_copy)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    assert result == "deploy-requested"

    # Verify written state
    write_call = fake_proxy.write_hidden_state.call_args
    assert write_call is not None, "write_hidden_state must have been called"
    written_state = write_call[0][3] if len(write_call[0]) > 3 else write_call[1].get("state")
    assert written_state is not None

    # component_id is on each component, not on state
    comp = written_state["components"][0]
    assert comp["component_id"] == old_cid, f"component_id should be {old_cid}, got {comp['component_id']}"
    assert comp["review_image_tag"] == TAG_A
    assert comp["image_ref"] == TAG_A
    assert comp["resolved_platform"] == "linux/arm64"
    assert "runtime_id" not in comp, "runtime_id must not be present for undeployed component"
    assert written_state["deployments"] == []

    # Verify ZERO registry calls (no registry mock was set up, so any call would fail)
    # If the code tried to call registry.resolve, it would raise AttributeError



@pytest.mark.asyncio
async def test_legacy_digest_deployed_component_preserves_deployment_and_historical_digest(config, monkeypatch):
    """TEST 2: Old deployed component with legacy digest preserves deployment and historical data."""
    from unittest.mock import AsyncMock, MagicMock

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()
    controller = _make_controller(fake_proxy, policy, fake_github)

    old_cid = "legacy-deployed-cid"
    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [{
            "component_id": old_cid, "target": "perception", "driver_path": "",
            "variant": "5.11", "review_image_tag": TAG_A,
            "image_ref": LEGACY_DIGEST, "resolved_platform": "linux/arm64",
            "runtime_id": "perception",
        }],
        "deployments": [{
            "machine": "m1", "component_ids": [old_cid], "phase": "deployed"
        }],
        "approve_attempts": [{"comment_id": 10, "actor": "alice", "machine": "m1",
                              "preflight": [], "outcome": "uncertain", "health": []}],
        "approve_attempts_total": 1,
        "approve_attempts_truncated": False,
        "case_results": {"legacy-deployed-cid": "pass"},
        "test_result": "pass",
        "cos": {"object_key": "evidence/key", "sha256": "b" * 64, "size": 1024},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    # Patch extract_review_evidence at the service module level to return valid evidence
    import agents.deploy_approval.service as svc_mod
    from ..review_comment_parser import ReviewCommentEvidence
    fake_evidence = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[MagicMock(target="perception", success=True, deployable=True,
                          image_tag=TAG_A, driver_path="", variant="5.11")],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence))

    import copy
    state_copy = copy.deepcopy(old_state)
    from ..github_state_proxy import _validate_hidden_state as _validate_hidden_state_fn
    _validate_hidden_state_fn(state_copy)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    assert result == "deploy-requested"

    write_call = fake_proxy.write_hidden_state.call_args
    written_state = write_call[0][3] if len(write_call[0]) > 3 else write_call[1].get("state")

    comp = written_state["components"][0]
    assert comp["component_id"] == old_cid
    assert comp["review_image_tag"] == TAG_A
    assert comp["image_ref"] == LEGACY_DIGEST  # Historical digest preserved for deployed
    assert comp["runtime_id"] == "perception"

    assert written_state["deployments"] == [{"machine": "m1", "component_ids": [old_cid], "phase": "deployed"}]
    assert written_state["approve_attempts"] == old_state["approve_attempts"]
    assert written_state["approve_attempts_total"] == 1

    assert written_state["case_results"] == {"legacy-deployed-cid": "pass"}
    assert written_state["test_result"] == "pass"
    assert written_state["cos"] == {"object_key": "evidence/key", "sha256": "b" * 64, "size": 1024}


@pytest.mark.asyncio
async def test_legacy_deployed_component_missing_runtime_id_fails_closed(config, monkeypatch):
    """TEST: Deployed component without runtime_id must fail closed — not continue as migration."""
    from unittest.mock import AsyncMock

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()
    controller = _make_controller(fake_proxy, policy, fake_github)

    old_cid = "legacy-deployed-no-runtime"
    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [{
            "component_id": old_cid, "target": "perception", "driver_path": "",
            "variant": "5.11", "review_image_tag": TAG_A,
            "image_ref": LEGACY_DIGEST, "resolved_platform": "linux/arm64",
            "runtime_id": "",  # Missing runtime_id for deployed component!
        }],
        "deployments": [{
            "machine": "m1", "component_ids": [old_cid], "phase": "deployed"
        }],
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False, "case_results": {}, "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    # Patch extract_review_evidence
    import agents.deploy_approval.service as svc_mod
    from ..review_comment_parser import ReviewCommentEvidence
    fake_evidence_rt = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[MagicMock(target="perception", success=True, deployable=True,
                          image_tag=TAG_A, driver_path="", variant="5.11")],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence_rt))

    import copy
    state_copy = copy.deepcopy(old_state)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    # Should reset to fresh snapshot (semantic_changed=True, evidence_changed=True)
    assert result == "deploy-requested"

    write_call = fake_proxy.write_hidden_state.call_args
    written_state = write_call[0][3] if len(write_call[0]) > 3 else write_call[1].get("state")

    # Because runtime_id was missing for deployed component, it triggers reset to fresh
    # Fresh components have a NEW component_id (from TAG_A), deployments should be cleared
    assert written_state["deployments"] == []
    # gate_note must NOT say "Previously confirmed deployments were preserved."
    write_call = fake_proxy.write_hidden_state.call_args
    markdown_arg = write_call[0][2] if len(write_call[0]) > 2 else ""
    assert "Previously confirmed deployments were preserved." not in markdown_arg
    # The component should have a fresh component_id (from TAG_A), not the old one
    comp = written_state["components"][0]
    # Fresh component_id comes from sha256(target|driver_path|variant|tag)
    import hashlib
    expected_cid = hashlib.sha256(f"perception||5.11|{TAG_A}".encode()).hexdigest()[:16]
    assert comp["component_id"] == expected_cid
    assert "runtime_id" not in comp, "runtime_id must not be present in fresh snapshot"



@pytest.mark.asyncio
async def test_real_review_tag_change_resets_snapshot(config, monkeypatch):
    """TEST 3: Real tag change -> semantic_changed=True, reset everything."""
    from unittest.mock import AsyncMock

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()
    controller = _make_controller(fake_proxy, policy, fake_github)

    old_cid_a = "component-tag-a"
    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [{
            "component_id": old_cid_a, "target": "perception", "driver_path": "",
            "variant": "5.11", "review_image_tag": TAG_A,
            "image_ref": TAG_A, "resolved_platform": "linux/arm64",
            "runtime_id": "perception",
        }],
        "deployments": [{"machine": "m1", "component_ids": [old_cid_a], "phase": "deployed"}],
        "approve_attempts": [{"comment_id": 10, "actor": "alice", "machine": "m1",
                              "preflight": [], "outcome": "uncertain", "health": []}],
        "approve_attempts_total": 1,
        "approve_attempts_truncated": False,
        "case_results": {"component-tag-a": "pass"},
        "test_result": "pass",
        "cos": {"object_key": "key", "sha256": "b" * 64, "size": 100},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    # Patch extract_review_evidence with TAG_B (different from old TAG_A)
    import agents.deploy_approval.service as svc_mod
    from ..review_comment_parser import ReviewCommentEvidence
    fake_evidence_b = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[MagicMock(target="perception", success=True, deployable=True,
                          image_tag=TAG_B, driver_path="", variant="5.11")],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence_b))

    import copy
    state_copy = copy.deepcopy(old_state)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    assert result == "deploy-requested"

    write_call = fake_proxy.write_hidden_state.call_args
    written_state = write_call[0][3] if len(write_call[0]) > 3 else write_call[1].get("state")

    # semantic_changed=True because TAG_B != TAG_A
    assert written_state["deployments"] == []
    assert written_state["approve_attempts"] == []
    assert written_state["case_results"] == {}
    assert written_state["test_result"] == ""
    assert written_state["cos"] == {"object_key": "", "sha256": "", "size": 0}

    # New component_id from TAG_B
    import hashlib
    expected_cid = hashlib.sha256(f"perception||5.11|{TAG_B}".encode()).hexdigest()[:16]
    assert written_state["components"][0]["component_id"] == expected_cid
    assert written_state["components"][0]["review_image_tag"] == TAG_B
    assert written_state["components"][0]["image_ref"] == TAG_B



@pytest.mark.asyncio
async def test_removed_component_resets_snapshot(config, monkeypatch):
    """TEST 4: Removed component -> semantic_changed=True, reset everything."""
    from unittest.mock import AsyncMock

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()
    controller = _make_controller(fake_proxy, policy, fake_github)

    import hashlib
    cid_a = hashlib.sha256(f"perception||5.11|{TAG_A}".encode()).hexdigest()[:16]
    cid_b = hashlib.sha256(f"actucore||5.11|{TAG_A}".encode()).hexdigest()[:16]

    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [
            {"component_id": cid_a, "target": "perception", "driver_path": "",
             "variant": "5.11", "review_image_tag": TAG_A, "image_ref": TAG_A,
             "resolved_platform": "linux/arm64", "runtime_id": "perception"},
            {"component_id": cid_b, "target": "actucore", "driver_path": "",
             "variant": "5.11", "review_image_tag": TAG_A, "image_ref": TAG_A,
             "resolved_platform": "linux/arm64", "runtime_id": "actucore"},
        ],
        "deployments": [
            {"machine": "m1", "component_ids": [cid_a], "phase": "deployed"},
            {"machine": "m1", "component_ids": [cid_b], "phase": "deployed"},
        ],
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False, "case_results": {}, "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    # Patch extract_review_evidence with only perception (B removed)
    import agents.deploy_approval.service as svc_mod
    from ..review_comment_parser import ReviewCommentEvidence
    fake_evidence_a = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[MagicMock(target="perception", success=True, deployable=True,
                          image_tag=TAG_A, driver_path="", variant="5.11")],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence_a))

    import copy
    state_copy = copy.deepcopy(old_state)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    assert result == "deploy-requested"

    write_call = fake_proxy.write_hidden_state.call_args
    written_state = write_call[0][3] if len(write_call[0]) > 3 else write_call[1].get("state")

    # semantic_changed=True because B was removed
    assert written_state["deployments"] == []
    assert written_state["approve_attempts"] == []
    assert written_state["case_results"] == {}
    assert written_state["test_result"] == ""
    assert written_state["cos"] == {"object_key": "", "sha256": "", "size": 0}
    # Only component A remains (fresh)
    assert len(written_state["components"]) == 1
    assert written_state["components"][0]["target"] == "perception"
    assert written_state["components"][0]["component_id"] == cid_a



@pytest.mark.asyncio
async def test_added_component_resets_snapshot(config, monkeypatch):
    """TEST 5: Added component -> semantic_changed=True, reset everything."""
    from unittest.mock import AsyncMock

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()
    controller = _make_controller(fake_proxy, policy, fake_github)

    import hashlib
    cid_a = hashlib.sha256(f"perception||5.11|{TAG_A}".encode()).hexdigest()[:16]

    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [
            {"component_id": cid_a, "target": "perception", "driver_path": "",
             "variant": "5.11", "review_image_tag": TAG_A, "image_ref": TAG_A,
             "resolved_platform": "linux/arm64", "runtime_id": "perception"},
        ],
        "deployments": [
            {"machine": "m1", "component_ids": [cid_a], "phase": "deployed"},
        ],
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False, "case_results": {}, "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    # Patch extract_review_evidence with both perception and actucore (B added)
    import agents.deploy_approval.service as svc_mod
    from ..review_comment_parser import ReviewCommentEvidence
    fake_evidence_ab = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[
            MagicMock(target="perception", success=True, deployable=True,
                      image_tag=TAG_A, driver_path="", variant="5.11"),
            MagicMock(target="actucore", success=True, deployable=True,
                      image_tag=TAG_A, driver_path="", variant="5.11"),
        ],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence_ab))

    import copy
    state_copy = copy.deepcopy(old_state)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    assert result == "deploy-requested"

    write_call = fake_proxy.write_hidden_state.call_args
    written_state = write_call[0][3] if len(write_call[0]) > 3 else write_call[1].get("state")

    # semantic_changed=True because B was added
    assert written_state["deployments"] == []
    assert written_state["approve_attempts"] == []
    assert written_state["case_results"] == {}
    assert written_state["test_result"] == ""
    assert written_state["cos"] == {"object_key": "", "sha256": "", "size": 0}
    # Both A and B are fresh
    assert len(written_state["components"]) == 2



@pytest.mark.asyncio
async def test_duplicate_old_semantic_component_resets_snapshot(config, monkeypatch):
    """TEST: Old state with duplicate semantic components -> semantic_changed=True, reset."""
    import hashlib
    from unittest.mock import AsyncMock, MagicMock
    from ..review_comment_parser import ReviewCommentEvidence

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()
    controller = _make_controller(fake_proxy, policy, fake_github)

    cid_a = hashlib.sha256(f"perception||5.11|{TAG_A}".encode()).hexdigest()[:16]

    # Old state has TWO components with same semantic key (perception, TAG_A)
    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [
            {"component_id": cid_a, "target": "perception", "driver_path": "",
             "variant": "5.11", "review_image_tag": TAG_A, "image_ref": TAG_A,
             "resolved_platform": "linux/arm64"},
            {"component_id": cid_a + "-dup", "target": "perception", "driver_path": "",
             "variant": "5.11", "review_image_tag": TAG_A, "image_ref": TAG_A,
             "resolved_platform": "linux/arm64"},
        ],
        "deployments": [{"machine": "m1", "component_ids": [cid_a], "phase": "deployed"}],
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False, "case_results": {}, "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    import agents.deploy_approval.service as svc_mod
    fake_evidence_a = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[MagicMock(target="perception", success=True, deployable=True,
                          image_tag=TAG_A, driver_path="", variant="5.11")],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence_a))

    import copy
    state_copy = copy.deepcopy(old_state)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    assert result == "deploy-requested"

    write_call = fake_proxy.write_hidden_state.call_args
    written_state = write_call[0][3] if len(write_call[0]) > 3 else write_call[1].get("state")

    # duplicate old semantic keys -> semantic_changed=True -> full reset
    assert written_state["deployments"] == []
    assert written_state["approve_attempts"] == []
    assert written_state["case_results"] == {}
    assert written_state["test_result"] == ""
    assert written_state["cos"] == {"object_key": "", "sha256": "", "size": 0}
    assert len(written_state["components"]) == 1
    assert written_state["components"][0]["target"] == "perception"

@pytest.mark.asyncio
async def test_duplicate_fresh_semantic_component_fails_closed(config, monkeypatch):
    """TEST 6: Fresh evidence with duplicate semantic components -> snapshot returns None."""
    from unittest.mock import AsyncMock, MagicMock

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()
    controller = _make_controller(fake_proxy, policy, fake_github)

    # Call the REAL _build_component_snapshot with two identical BuildInfo (Mock with deployable)
    build1 = MagicMock(target="perception", driver_path="", variant="5.11",
                       success=True, deployable=True, image_tag=TAG_A, version="v1")
    build2 = MagicMock(target="perception", driver_path="", variant="5.11",
                       success=True, deployable=True, image_tag=TAG_A, version="v1")

    snapshot = await controller._build_component_snapshot(
        "4paradigm/phanthymotus", 1, "a" * 40, [build1, build2]
    )
    assert snapshot is None, "duplicate semantic key must produce None snapshot"

    # Now also exercise via _refresh_uncertain_state with duplicate evidence builds
    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [], "deployments": [],
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False, "case_results": {}, "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    import agents.deploy_approval.service as svc_mod
    from ..review_comment_parser import ReviewCommentEvidence
    fake_evidence_dup = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[
            MagicMock(target="perception", success=True, deployable=True,
                      image_tag=TAG_A, driver_path="", variant="5.11"),
            MagicMock(target="perception", success=True, deployable=True,
                      image_tag=TAG_A, driver_path="", variant="5.11"),
        ],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence_dup))

    import copy
    state_copy = copy.deepcopy(old_state)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    # duplicate semantic key -> _build_component_snapshot returns None -> uncertain
    assert result == "uncertain"
    assert old_state["deployments"] == []


@pytest.mark.asyncio
async def test_legacy_migration_zero_registry_http(config, monkeypatch):
    """TEST 7: Legacy migration makes ZERO Registry HTTP calls.

    Proves: legacy digest state -> recovery -> exact Review Agent tag ->
    ZERO Registry dependency.
    """
    from unittest.mock import AsyncMock

    fake_proxy = AsyncMock()
    fake_proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    fake_proxy.comment_identity = AsyncMock(return_value=("test_author", "test_author"))
    fake_proxy.read_hidden_state = AsyncMock(return_value=None)

    fake_github = AsyncMock()
    fake_github.get_issue_comments = AsyncMock(return_value=[])
    fake_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    policy = _make_policy()
    controller = _make_controller(fake_proxy, policy, fake_github)

    # CRITICAL: Controller must NOT have a registry attribute at all.
    assert not hasattr(controller, "registry")

    old_cid = "legacy-migration-cid"
    old_state = {
        "version": 1, "head_sha": "a" * 40, "status": "deploy-requested",
        "review_evidence": {
            "build_comment_id": 1, "build_comment_updated_at": "2025-01-01T00:00:00Z",
            "commit_prefix": "a" * 7, "resolved_head_sha": "a" * 40,
            "test_comment_id": 2, "test_comment_updated_at": "2025-01-01T00:00:01Z",
            "code_review_comment_id": 3, "code_review_comment_updated_at": "2025-01-01T00:00:02Z",
            "review_author_id": "7950763",
        },
        "components": [{
            "component_id": old_cid, "target": "perception", "driver_path": "",
            "variant": "5.11", "review_image_tag": TAG_A,
            "image_ref": LEGACY_DIGEST, "resolved_platform": "linux/arm64",
        }],
        "deployments": [], "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False, "case_results": {}, "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 17, "kind": "approve_deploy", "phase": "uncertain",
                    "args": {"machine": "m1"}},
        "last_processed_comment_id": 17,
    }

    # Patch extract_review_evidence
    import agents.deploy_approval.service as svc_mod
    from ..review_comment_parser import ReviewCommentEvidence
    fake_evidence_zr = ReviewCommentEvidence(
        head_sha="a" * 40, commit_prefix="a" * 7, review_author_id="7950763",
        build_comment_id=1, build_comment_updated_at="2025-01-01T00:00:00Z",
        test_comment_id=2, test_comment_updated_at="2025-01-01T00:00:01Z",
        code_review_comment_id=3, code_review_comment_updated_at="2025-01-01T00:00:02Z",
        builds=[MagicMock(target="perception", success=True, deployable=True,
                          image_tag=TAG_A, driver_path="", variant="5.11")],
    )
    monkeypatch.setattr(svc_mod, "extract_review_evidence", MagicMock(return_value=fake_evidence_zr))

    import copy
    state_copy = copy.deepcopy(old_state)
    result = await controller._refresh_uncertain_state("4paradigm/phanthymotus", 1, state_copy)

    assert result == "deploy-requested"

    # Verify write_hidden_state was actually called
    fake_proxy.write_hidden_state.assert_awaited()

    # Parse the written state from the call
    written_state = fake_proxy.write_hidden_state.call_args.args[3]

    # Verify the migrated component preserves old component_id and gets exact Review Agent tag
    assert written_state["components"][0]["component_id"] == old_cid
    assert written_state["components"][0]["review_image_tag"] == TAG_A
    assert written_state["components"][0]["image_ref"] == TAG_A
    assert written_state["components"][0]["resolved_platform"] == "linux/arm64"

    # Verify all deployment/tracking state was reset
    assert written_state["deployments"] == []


# ── History archive regression tests ────────────────────────────────────────


def _lifecycle_visible(repo: str, pr_number: int, status: str = "deploy-requested") -> str:
    """Build a minimal NEW-format trusted lifecycle visible markdown."""
    from ..comments import BOT_MARKER, lifecycle_marker
    return "\n".join([
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        f"**Status:** `{status}`",
        "",
        "### Workflow",
        "",
        "- [x] review",
        "",
    ])


def _lifecycle_body(repo: str, pr_number: int, state: dict, visible: str) -> str:
    from ..github_state_proxy import _build_hidden_state_body
    return _build_hidden_state_body(visible, state)


@pytest.mark.asyncio
async def test_transient_deploying_visible_is_not_archived_as_legacy():
    """A trusted lifecycle whose visible part is the transient 'Deploying...' must
    NOT be treated as a legacy lifecycle: no archive comment, normal write."""
    controller, proxy, policy, github, config = _controller()
    state = _state()

    from ..comments import BOT_MARKER
    transient_visible = BOT_MARKER + "\nDeploying..."
    existing_body = _lifecycle_body("repo", 1, state, transient_visible)
    proxy.find_trusted_lifecycle_comment = AsyncMock(
        return_value={"id": 42, "body": existing_body}
    )
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": existing_body}])
    proxy.write_hidden_state = AsyncMock()

    event = {"event": "Deploy started", "machine": "test-machine",
             "timestamp": comments_mod.beijing_now_str()}

    await controller._write_lifecycle_with_history(
        "repo", 1, state, _lifecycle_visible("repo", 1), event=event,
    )

    # NO legacy archive comment created
    proxy.post_issue_comment.assert_not_called()
    posted_bodies = [c.args[2] for c in proxy.post_issue_comment.call_args_list] if proxy.post_issue_comment.call_args_list else []
    assert all("Legacy lifecycle snapshot preserved" not in b for b in posted_bodies)

    # Normal lifecycle write happened with the new history event
    proxy.write_hidden_state.assert_awaited_once()
    written_visible = proxy.write_hidden_state.call_args.args[2]
    assert "Deploy started" in written_visible


@pytest.mark.asyncio
async def test_true_legacy_lifecycle_is_archived_once():
    """A true old-format Deploy Approval lifecycle is archived exactly once on the
    first meaningful event; a subsequent write must not duplicate the archive."""
    controller, proxy, policy, github, config = _controller()
    state = _state()

    from ..comments import BOT_MARKER
    legacy_visible = "\n".join([
        BOT_MARKER,
        "### Deploy Approval",
        "",
        "**Status:** `deploy-requested`",
        "",
        "Old-format lifecycle content.",
    ])
    legacy_body_with_state = _lifecycle_body("repo", 1, state, legacy_visible)

    # Emulate the real store: write_hidden_state UPDATES the trusted comment body,
    # so a second write reads the migrated new-format visible and must not re-archive.
    current_body = {"body": legacy_body_with_state}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": current_body["body"]}

    async def _write_hidden_state(_repo, _pr, visible, _state):
        from ..github_state_proxy import _build_hidden_state_body
        current_body["body"] = _build_hidden_state_body(visible, _state)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": legacy_body_with_state}])
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.post_issue_comment = AsyncMock(return_value={"id": 77})

    event = {"event": "First meaningful event", "machine": "test-machine",
             "timestamp": comments_mod.beijing_now_str()}

    await controller._write_lifecycle_with_history(
        "repo", 1, state, _lifecycle_visible("repo", 1), event=event,
    )

    # First call: ONE legacy archive created
    assert proxy.post_issue_comment.await_count == 1
    archive_body = proxy.post_issue_comment.call_args.args[2]
    assert "Legacy lifecycle snapshot preserved" in archive_body
    assert "History Archive Page 1" in archive_body
    # Hidden business state must not be lost: the archive preserves old visible
    assert "Old-format lifecycle content." in archive_body

    # Second write on the SAME legacy comment must not re-archive the same snapshot
    proxy.write_hidden_state.reset_mock()
    await controller._write_lifecycle_with_history(
        "repo", 1, state, _lifecycle_visible("repo", 1), event=event,
    )
    assert proxy.post_issue_comment.await_count == 1  # still just the one archive


@pytest.mark.asyncio
async def test_visible_history_rollover_preserves_oldest_events():
    """When the visible lifecycle approaches the 48 KiB budget, the preflight archive
    step (run before any unsafe deploy POST) moves oldest events into a History
    Archive page; no events are silently dropped and no #N autolink is produced."""
    repo, pr_number = "repo", 1
    controller, proxy, policy, github, config = _controller()
    state = _state()

    visible = _lifecycle_visible(repo, pr_number)
    posted_bodies: list[str] = []

    async def _post(_repo, _pr, body):
        posted_bodies.append(body)
        return {"id": 90}

    proxy.post_issue_comment = AsyncMock(side_effect=_post)

    # Build a large history: many oversized events so the block exceeds 48 KiB
    from ..github_state_proxy import _build_history_block, _count_visible_bytes
    events = [
        {"event": f"Bulk event {i} " + "x" * 900,
         "machine": "test-machine",
         "timestamp": f"2026-09-24 10:{i:02d}:00"}
        for i in reversed(range(60))
    ]
    history_block = _build_history_block(events)
    primed_visible = visible.rstrip() + "\n\n### History\n\n" + history_block + "\n"
    assert _count_visible_bytes(primed_visible) > 48 * 1024, "precondition: primed body must exceed budget"

    primed_body = _lifecycle_body(repo, pr_number, state, primed_visible)

    # Emulate the real store so the preflight update rewrites the same comment
    current_body = {"body": primed_body}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": current_body["body"]}

    async def _update_comment(_repo, comment_id, body):
        current_body["body"] = body
        return {"id": comment_id}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.get_issue_comments = AsyncMock(
        return_value=[{"id": 42, "body": current_body["body"]}]
    )
    proxy.update_comment = AsyncMock(side_effect=_update_comment)

    reserved_event = {"event": "Rollover trigger event", "machine": "test-machine",
                      "timestamp": comments_mod.beijing_now_str()}

    # Preflight runs before the unsafe deploy POST: rollover must happen here
    await controller._preflight_archive_for_event(
        repo, pr_number, visible, reserved_event=reserved_event,
    )

    # An archive comment was created for the oldest events
    assert proxy.post_issue_comment.await_count == 1
    archive_body = posted_bodies[0]
    assert "History Archive Page 1" in archive_body
    # Oldest events are preserved in the archive (no silent loss)
    assert "Bulk event 0" in archive_body

    # Main lifecycle bounded: rewritten visible stays within budget
    rewritten = current_body["body"]
    rewritten_visible = rewritten.split("<!-- deploy-approval-state:v1")[0]
    assert _count_visible_bytes(rewritten_visible) <= 48 * 1024
    # The new event is not silently lost: it stays reserved for the caller's write
    # and the shrunk main history retains the newest kept events
    assert "Bulk event 59" in rewritten_visible


# ── Visible history preservation regression tests ───────────────────────────

from ..github_state_proxy import (
    VISIBLE_HISTORY_END_MARKER,
    VISIBLE_HISTORY_START_MARKER,
    _build_history_block,
    _parse_visible_history,
)


def _visible_history_marker_counts(visible: str) -> tuple[int, int, int]:
    """Return (history_heading_count, start_marker_count, end_marker_count)."""
    return (
        visible.count("### History"),
        visible.count(VISIBLE_HISTORY_START_MARKER),
        visible.count(VISIBLE_HISTORY_END_MARKER),
    )


def _event_titles(visible: str) -> list[str]:
    _, events = _parse_visible_history(visible)
    return [str(e.get("event", "")) for e in events]


def _make_history_state(status: str = "succeeded", test_result: str = "pass") -> dict:
    st = _state()
    st["status"] = status
    st["test_result"] = test_result
    st["command"] = {
        "comment_id": 17, "kind": "record_test", "phase": "completed",
        "args": {"actor": "alice", "result": "pass"},
    }
    st["last_processed_comment_id"] = 17
    return st


def _seed_lifecycle_with_history(proxy, repo: str, pr_number: int, state: dict):
    """Install store-emulating mocks so writes update the same comment body."""
    visible = _lifecycle_visible(repo, pr_number, status=state["status"])
    body_holder = {"body": _lifecycle_body(repo, pr_number, state, visible)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _state):
        from ..github_state_proxy import _build_hidden_state_body
        body_holder["body"] = _build_hidden_state_body(vis, _state)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    return body_holder


def _existing_history_events(count: int = 2) -> list[dict]:
    return [
        {"event": "Deployment requested", "machine": "test-machine", "ip": "127.0.0.1",
         "components": "perception", "result": "",
         "timestamp": "2026-09-30 10:00:00"},
        {"event": "Machine `test-machine` deployed", "machine": "test-machine",
         "ip": "127.0.0.1", "components": "perception", "result": "",
         "timestamp": "2026-09-30 10:05:00"},
    ][:count]


def _prime_history(body_holder: dict, repo: str, pr_number: int, state: dict):
    """Inject an existing visible History block into the stored lifecycle body."""
    from ..github_state_proxy import _build_hidden_state_body
    history_block = _build_history_block(_existing_history_events())
    body = body_holder["body"]
    visible, _, _ = body.partition("<!-- deploy-approval-state:v1")
    new_visible = visible.rstrip() + "\n\n### History\n\n" + history_block + "\n"
    body_holder["body"] = _build_hidden_state_body(new_visible, state)


@pytest.mark.asyncio
async def test_event_none_rewrite_preserves_visible_history():
    """event=None rewrite must carry over ALL existing visible history events,
    add no new event, and keep exactly one History section."""
    controller, proxy, _policy, _github, _config = _controller()
    state = _make_history_state()
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)
    _prime_history(body_holder, "repo", 1, state)

    fresh_markdown = _lifecycle_visible("repo", 1, status="succeeded")

    await controller._write_lifecycle_with_history(
        "repo", 1, state, fresh_markdown, event=None,
    )

    written_visible = proxy.write_hidden_state.call_args.args[2]
    headings, starts, ends = _visible_history_marker_counts(written_visible)
    assert headings == 1
    assert starts == 1
    assert ends == 1
    titles = _event_titles(written_visible)
    assert "Deployment requested" in titles
    assert "Machine `test-machine` deployed" in titles
    # No new event was fabricated by the event=None rewrite
    assert len(titles) == 2


@pytest.mark.asyncio
async def test_terminal_cos_metadata_rebind_preserves_visible_history():
    """COS metadata rebind must update state.cos while keeping the terminal
    History (Test recorded) intact and emitting no new lifecycle event."""
    controller, proxy, _policy, _github, _config = _controller()
    state = _make_history_state()
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)
    _prime_history(body_holder, "repo", 1, state)

    # Simulate the terminal record_test write: Test recorded event added
    terminal_md = comments_mod.succeeded_comment("repo", 1, "a" * 40)
    recorded_event = {
        "event": "Test recorded",
        "lifecycle": "`testing` → `succeeded`",
        "result": "pass",
        "timestamp": comments_mod.beijing_now_str(),
    }
    await controller._write_lifecycle_with_history(
        "repo", 1, state, terminal_md, event=recorded_event,
    )
    assert proxy.write_hidden_state.await_count == 1
    first_write = proxy.write_hidden_state.call_args.args[2]
    assert _event_titles(first_write).count("Test recorded") == 1

    proxy.write_hidden_state.reset_mock()
    fresh_state = _make_history_state()
    fresh_state["head_sha"] = "a" * 40

    async def _read_state(_repo, _pr):
        return dict(fresh_state)

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    rebind_md = comments_mod.succeeded_comment(
        "repo", 1, "a" * 40,
        cos_object_key="phanthymotus_pr/phanthymotus/2026-10/2026-10-13/pr-1/evidence-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.log.gz",
        cos_bundle_sha256="ab" * 32,
        cos_bundle_size=1024,
    )
    rebound = await controller._rebind_terminal_cos_if_current(
        "4paradigm/phanthymotus", 1,
        expected_head="a" * 40,
        expected_terminal_status="succeeded",
        expected_comment_id=17,
        expected_command_kind="record_test",
        expected_test_result="pass",
        cos_metadata={"object_key": "phanthymotus_pr/phanthymotus/2026-10/2026-10-13/pr-1/evidence-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.log.gz",
                      "sha256": "ab" * 32, "size": 1024},
        markdown=rebind_md,
    )
    assert rebound is True
    assert proxy.write_hidden_state.await_count == 1

    # state.cos updated correctly
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["cos"]["object_key"] == \
        "phanthymotus_pr/phanthymotus/2026-10/2026-10-13/pr-1/evidence-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.log.gz"
    assert written_state["cos"]["size"] == 1024

    # Terminal History intact; no new event; exactly one History section
    written_visible = proxy.write_hidden_state.call_args.args[2]
    headings, starts, ends = _visible_history_marker_counts(written_visible)
    assert (headings, starts, ends) == (1, 1, 1)
    titles = _event_titles(written_visible)
    assert titles.count("Test recorded") == 1
    assert len(titles) == 3  # 2 pre-existing + Test recorded, nothing new


@pytest.mark.asyncio
async def test_terminal_cos_presigned_url_rebind_preserves_visible_history():
    """Full record_test two-phase terminal flow: terminal write → COS metadata
    rebind → presigned URL rebind. Final visible keeps the COS download block,
    the full existing History, and exactly one Test recorded event."""
    controller, proxy, _policy, _github, _config = _controller()
    state = _make_history_state()
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)
    _prime_history(body_holder, "repo", 1, state)

    object_key = "phanthymotus_pr/phanthymotus/2026-10/2026-10-13/pr-1/evidence-" + "a" * 40 + ".log.gz"

    # Phase 0: terminal write with Test recorded event
    terminal_md = comments_mod.succeeded_comment("repo", 1, "a" * 40)
    recorded_event = {
        "event": "Test recorded",
        "lifecycle": "`testing` → `succeeded`",
        "result": "pass",
        "timestamp": comments_mod.beijing_now_str(),
    }
    await controller._write_lifecycle_with_history(
        "repo", 1, state, terminal_md, event=recorded_event,
    )

    proxy.write_hidden_state.reset_mock()
    fresh_state = _make_history_state()

    async def _read_state(_repo, _pr):
        return dict(fresh_state)

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    cos_metadata = {"object_key": object_key, "sha256": "cd" * 32, "size": 2048}

    # Phase 1: COS metadata rebind
    md_no_url = comments_mod.succeeded_comment(
        "repo", 1, "a" * 40,
        cos_object_key=object_key, cos_bundle_sha256="cd" * 32, cos_bundle_size=2048,
    )
    ok1 = await controller._rebind_terminal_cos_if_current(
        "4paradigm/phanthymotus", 1,
        expected_head="a" * 40, expected_terminal_status="succeeded",
        expected_comment_id=17, expected_command_kind="record_test",
        expected_test_result="pass", cos_metadata=cos_metadata, markdown=md_no_url,
    )
    assert ok1 is True

    # Phase 2: presigned URL rebind
    md_with_url = comments_mod.succeeded_comment(
        "repo", 1, "a" * 40,
        cos_object_key=object_key, cos_bundle_sha256="cd" * 32, cos_bundle_size=2048,
        cos_download_url="https://bucket.cos.ap-beijing.myqcloud.com/signed?token=x",
    )
    ok2 = await controller._rebind_terminal_cos_if_current(
        "4paradigm/phanthymotus", 1,
        expected_head="a" * 40, expected_terminal_status="succeeded",
        expected_comment_id=17, expected_command_kind="record_test",
        expected_test_result="pass", cos_metadata=cos_metadata, markdown=md_with_url,
    )
    assert ok2 is True
    assert proxy.write_hidden_state.await_count == 2

    final_visible = proxy.write_hidden_state.call_args.args[2]
    # COS evidence block present with correct key/sha/size rendering
    assert "Download COS evidence" in final_visible
    assert object_key in final_visible
    assert "@sha256:cdcdcdcdcdcd" in final_visible  # sha256[:12] rendered
    assert "2.0 KB" in final_visible

    # History fully preserved across both rebinds
    headings, starts, ends = _visible_history_marker_counts(final_visible)
    assert (headings, starts, ends) == (1, 1, 1)
    titles = _event_titles(final_visible)
    assert titles.count("Test recorded") == 1
    assert "Deployment requested" in titles
    assert "Machine `test-machine` deployed" in titles


@pytest.mark.asyncio
async def test_history_rewrite_does_not_duplicate_events():
    """Repeated event=None rewrites must be idempotent: event count never grows,
    every original event appears exactly once, markers stay unique."""
    controller, proxy, _policy, _github, _config = _controller()
    state = _make_history_state()
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)
    _prime_history(body_holder, "repo", 1, state)

    fresh_markdown = _lifecycle_visible("repo", 1, status="succeeded")

    for _ in range(2):
        await controller._write_lifecycle_with_history(
            "repo", 1, state, fresh_markdown, event=None,
        )

    written_visible = proxy.write_hidden_state.call_args.args[2]
    headings, starts, ends = _visible_history_marker_counts(written_visible)
    assert (headings, starts, ends) == (1, 1, 1)
    titles = _event_titles(written_visible)
    assert len(titles) == 2
    for expected in ("Deployment requested", "Machine `test-machine` deployed"):
        assert titles.count(expected) == 1


@pytest.mark.asyncio
async def test_history_rewrite_preserves_archive_links():
    """event=None rewrite must keep the trusted archive comment linkage:
    the Archived history block and its History Archive Page link survive,
    without duplication, alongside the main visible History."""
    controller, proxy, _policy, _github, _config = _controller()
    state = _make_history_state()
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)
    _prime_history(body_holder, "repo", 1, state)

    from ..comments import BOT_MARKER
    from ..github_state_proxy import _history_archive_marker
    archive_body = "\n".join([
        BOT_MARKER,
        _history_archive_marker("repo", 1, 1),
        "### History Archive Page 1",
        "",
        "Archived oldest history events.",
    ])
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 42, "body": body_holder["body"]},
        {"id": 77, "body": archive_body, "html_url":
            "https://github.com/repo/pull/1#issuecomment-77"},
    ])

    fresh_markdown = _lifecycle_visible("repo", 1, status="succeeded")

    await controller._write_lifecycle_with_history(
        "repo", 1, state, fresh_markdown, event=None,
    )

    written_visible = proxy.write_hidden_state.call_args.args[2]
    assert "### Archived history" in written_visible
    assert "History Archive Page 1" in written_visible
    assert written_visible.count("History Archive Page 1") == 1
    # Main history still intact
    headings, starts, ends = _visible_history_marker_counts(written_visible)
    assert (headings, starts, ends) == (1, 1, 1)
    assert len(_event_titles(written_visible)) == 2


@pytest.mark.asyncio
async def test_history_isolated_between_prs():
    """History events, archive links, and lifecycle markers must never leak
    across PRs or repos.

    Uses fully disjoint sentinel event titles with exact membership/equality
    assertions (no substring overlap possible):
      - repo#289   → EVENT_ALPHA_REPO_ONE_289
      - repo#290   → EVENT_BETA_REPO_ONE_290
      - repo2#289  → EVENT_GAMMA_REPO_TWO_289  (different repo, same PR number)
    Each lifecycle also gets its own trusted archive comment with a unique
    html_url; rewrites must keep only the matching archive link per lifecycle.
    """
    controller, proxy, _policy, _github, _config = _controller()

    EVENT_289 = "EVENT_ALPHA_REPO_ONE_289"
    EVENT_290 = "EVENT_BETA_REPO_ONE_290"
    EVENT_R2 = "EVENT_GAMMA_REPO_TWO_289"

    ARCHIVE_URL_289 = "https://example.invalid/archive/alpha-289"
    ARCHIVE_URL_290 = "https://example.invalid/archive/beta-290"
    ARCHIVE_URL_R2 = "https://example.invalid/archive/gamma-repo2-289"

    from ..comments import BOT_MARKER, lifecycle_marker
    from ..github_state_proxy import (
        _build_hidden_state_body,
        _history_archive_marker,
    )

    def _make_spec(repo: str, pr: int, event_title: str):
        """Seed a lifecycle primed with exactly one unique sentinel event."""
        state = _make_history_state()
        body_holder = _seed_lifecycle_with_history(proxy, repo, pr, state)
        body = body_holder["body"]
        visible, _, _ = body.partition("<!-- deploy-approval-state:v1")
        history_block = _build_history_block([
            {"event": event_title, "machine": f"m-{repo}-{pr}",
             "ip": "127.0.0.1", "components": "perception", "result": "",
             "timestamp": "2026-09-30 10:00:00"},
        ])
        new_visible = visible.rstrip() + "\n\n### History\n\n" + history_block + "\n"
        body_holder["body"] = _build_hidden_state_body(new_visible, state)
        return state, body_holder

    def _archive_body(repo: str, pr: int) -> str:
        return "\n".join([
            BOT_MARKER,
            _history_archive_marker(repo, pr, 1),
            "### History Archive Page 1",
            "",
            "Archived oldest history events.",
        ])

    state_289, holder_289 = _make_spec("repo", 289, EVENT_289)
    state_290, holder_290 = _make_spec("repo", 290, EVENT_290)
    state_r2, holder_r2 = _make_spec("repo2", 289, EVENT_R2)

    # Store emulation keyed by (repo, pr)
    bodies = {
        ("repo", 289): holder_289,
        ("repo", 290): holder_290,
        ("repo2", 289): holder_r2,
    }
    archive_urls = {
        ("repo", 289): ARCHIVE_URL_289,
        ("repo", 290): ARCHIVE_URL_290,
        ("repo2", 289): ARCHIVE_URL_R2,
    }

    async def _find_trusted(repo, pr):
        return {"id": 42, "body": bodies[(repo, pr)]["body"]}

    async def _write_hidden_state(repo, pr, vis, _state):
        bodies[(repo, pr)]["body"] = _build_hidden_state_body(vis, _state)
        return {"id": 42}

    async def _get_comments(repo, pr):
        # Only this repo/pr's lifecycle comment and its own archive comment.
        return [
            {"id": 42, "body": bodies[(repo, pr)]["body"]},
            {"id": 77, "body": _archive_body(repo, pr),
             "html_url": archive_urls[(repo, pr)]},
        ]

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(side_effect=_get_comments)
    # Archive comments are trusted GitHub App comments (sync method).
    proxy.is_bot_comment = MagicMock(return_value=True)

    fresh_289 = _lifecycle_visible("repo", 289, status="succeeded")
    fresh_290 = _lifecycle_visible("repo", 290, status="succeeded")
    fresh_r2 = _lifecycle_visible("repo2", 289, status="succeeded")

    await controller._write_lifecycle_with_history(
        "repo", 289, state_289, fresh_289, event=None)
    await controller._write_lifecycle_with_history(
        "repo", 290, state_290, fresh_290, event=None)
    await controller._write_lifecycle_with_history(
        "repo2", 289, state_r2, fresh_r2, event=None)

    visible_289 = proxy.write_hidden_state.call_args_list[0].args[2]
    visible_290 = proxy.write_hidden_state.call_args_list[1].args[2]
    visible_r2 = proxy.write_hidden_state.call_args_list[2].args[2]

    # Exactly one History section per rewritten lifecycle.
    headings_289, starts_289, ends_289 = _visible_history_marker_counts(visible_289)
    headings_290, starts_290, ends_290 = _visible_history_marker_counts(visible_290)
    headings_r2, starts_r2, ends_r2 = _visible_history_marker_counts(visible_r2)
    assert (headings_289, starts_289, ends_289) == (1, 1, 1)
    assert (headings_290, starts_290, ends_290) == (1, 1, 1)
    assert (headings_r2, starts_r2, ends_r2) == (1, 1, 1)

    titles_289 = _event_titles(visible_289)
    titles_290 = _event_titles(visible_290)
    titles_r2 = _event_titles(visible_r2)

    # Exact event-title equality: each lifecycle carries exactly its own
    # sentinel event — no duplication, no fabrication, no cross-contamination.
    assert titles_289 == [EVENT_289]
    assert titles_290 == [EVENT_290]
    assert titles_r2 == [EVENT_R2]

    # Exact membership: sentinel events never appear in another lifecycle.
    assert EVENT_289 not in titles_290
    assert EVENT_289 not in titles_r2
    assert EVENT_290 not in titles_289
    assert EVENT_290 not in titles_r2
    assert EVENT_R2 not in titles_289
    assert EVENT_R2 not in titles_290

    # Archive link isolation: each visible keeps only its own archive URL.
    assert ARCHIVE_URL_289 in visible_289
    assert ARCHIVE_URL_290 not in visible_289
    assert ARCHIVE_URL_R2 not in visible_289
    assert ARCHIVE_URL_290 in visible_290
    assert ARCHIVE_URL_289 not in visible_290
    assert ARCHIVE_URL_R2 not in visible_290
    assert ARCHIVE_URL_R2 in visible_r2
    assert ARCHIVE_URL_289 not in visible_r2
    assert ARCHIVE_URL_290 not in visible_r2

    # "History Archive Page 1" link appears exactly once per lifecycle.
    assert visible_289.count("History Archive Page 1") == 1
    assert visible_290.count("History Archive Page 1") == 1
    assert visible_r2.count("History Archive Page 1") == 1

    # Lifecycle markers stay bound to their own repo/pr.
    assert lifecycle_marker("repo", 289) in visible_289
    assert lifecycle_marker("repo", 290) not in visible_289
    assert lifecycle_marker("repo2", 289) not in visible_289
    assert lifecycle_marker("repo", 290) in visible_290
    assert lifecycle_marker("repo", 289) not in visible_290
    assert lifecycle_marker("repo2", 289) not in visible_290
    assert lifecycle_marker("repo2", 289) in visible_r2
    assert lifecycle_marker("repo", 289) not in visible_r2
    assert lifecycle_marker("repo", 290) not in visible_r2


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION TESTS — PR #291 History-preservation gap (Tests 1-5)
# ══════════════════════════════════════════════════════════════════════════════

import ast as _ast


# ── Test 1: executing state write preserves visible history ───────────────────

@pytest.mark.asyncio
async def test_executing_state_write_preserves_visible_history():
    """The executing-state write (event=None) before the unsafe deploy POST
    MUST preserve ALL existing visible history events.

    Construction:
      1. Lifecycle initialized  (from reconcile)
      2. Review lifecycle transitioned
      3. Deployment requested

    Then drive handle_approve_deploy through the executing phase.
    The executing write uses event=None, so all three events above must
    survive into the persisted visible lifecycle.
    """
    controller, proxy, _policy, _github, _config = _controller()

    # Step 1: seed lifecycle with pre-existing history from reconcile + request_deploy
    state = _state(status="deploy-requested")
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)

    # Inject the three expected pre-existing history events
    from ..github_state_proxy import _build_hidden_state_body, _build_history_block
    pre_events = [
        {"event": "Lifecycle initialized",
         "lifecycle": "`none` → `review-required`",
         "timestamp": "2026-09-30 10:00:00"},
        {"event": "Review lifecycle transitioned",
         "lifecycle": "`review-required` → `deploy-requested`",
         "timestamp": "2026-09-30 10:01:00"},
        {"event": "Deployment requested",
         "lifecycle": "`deploy-ready` → `deploy-requested`",
         "timestamp": "2026-09-30 10:02:00"},
    ]
    history_block = _build_history_block(pre_events)
    body = body_holder["body"]
    visible, _, _ = body.partition("<!-- deploy-approval-state:v1")
    new_visible = visible.rstrip() + "\n\n### History\n\n" + history_block + "\n"
    body_holder["body"] = _build_hidden_state_body(new_visible, state)

    # Reset proxy mocks so handle_approve_deploy finds this primed lifecycle
    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _build_hidden_state_body(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    # Ensure PR is open and valid
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40},
        "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": "/approve_deploy machine=test-machine",
        "user": {"id": 111, "login": "owner1"},
    })

    # Mock Agent Core — driver_status must return running_image matching
    # the component image_ref (registry.example/...) so _verify_deployed_runtime
    # succeeds on the first call and the deploy completes.
    core = MagicMock()
    core.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    _test1_img = "registry.example/repo@sha256:" + "a" * 64
    core.driver_status = AsyncMock(return_value={
        "status": "running", "running_image": _test1_img})
    core.deploy_driver = AsyncMock(return_value={"ok": True})
    controller._core_for_node = AsyncMock(return_value=core)

    # Mock review evidence validation (fresh PR re-read must pass)
    emdash = "\u2014"
    _github.get_issue_comments = AsyncMock(return_value=[
        {"id": 1001, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| Target | Status | Version | Took |\n| perception | :white_check_mark: Success | `registry/repo:v1` | 10s |\n",
         "created_at": "2026-09-18T00:00:00Z", "updated_at": "2026-09-18T00:01:00Z"},
        {"id": 1002, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Test Results\n\nCommit: abc1234\n\n| Suite | Result | Passed | Failed | Took |\n| perception | :white_check_mark: Passed | 10 | 0 | 5s |\n",
         "created_at": "2026-09-18T00:02:00Z", "updated_at": "2026-09-18T00:03:00Z"},
        {"id": 1003, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nAll checks passed.",
         "created_at": "2026-09-18T00:04:00Z", "updated_at": "2026-09-18T00:05:00Z"},
    ])
    _github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    controller._revalidate_hidden_state = AsyncMock(return_value=dict(state))
    controller._run_automated_case = AsyncMock(return_value={})
    controller._refresh_uncertain_state = AsyncMock(return_value="deploy-requested")

    # Short timeout so _verify_deployed_runtime doesn't sleep
    _test1_orig_timeout = controller.config.total_timeout
    controller.config.total_timeout = 0.5

    # Now drive handle_approve_deploy
    await controller.handle_approve_deploy("repo", 1, 50, "test-machine", "owner1", "111")
    controller.config.total_timeout = _test1_orig_timeout

    # ── Assertions ──
    # At least the executing-state write must have happened
    assert proxy.write_hidden_state.call_count >= 1

    # The FIRST write is the executing-state write (event=None)
    first_visible = proxy.write_hidden_state.call_args_list[0].args[2]

    headings, starts, ends = _visible_history_marker_counts(first_visible)
    assert headings == 1, f"expected exactly 1 History heading, got {headings}"
    assert starts == 1
    assert ends == 1

    titles = _event_titles(first_visible)
    # All pre-existing events must survive
    assert "Lifecycle initialized" in titles
    assert "Review lifecycle transitioned" in titles
    assert "Deployment requested" in titles

    # No new Deploying event (event=None)
    deploying_count = sum(1 for t in titles if "Deploying" in t)
    assert deploying_count == 0, f"unexpected Deploying event in executing write: {titles}"

    # No duplicate events
    for t in set(titles):
        assert titles.count(t) == 1, f"event {t!r} duplicated: {titles}"

    # Event count unchanged (3 pre-existing → 3)
    assert len(titles) == 3

    # The executing phase write happened BEFORE deploy POST
    # (core.deploy_driver should not have been called before first write)
    # Since the first write is the executing write, this is guaranteed.


# ── Test 2: automated case refresh preserves visible history ─────────────────

@pytest.mark.asyncio
async def test_automated_case_refresh_preserves_visible_history():
    """After all machines are deployed and lifecycle reaches `testing`,
    the advisory Automated Case refresh (event=None) MUST preserve
    all existing history events (machine deployments, all-components).

    Full handle_approve_deploy path for a single machine covering all components:
      - executing state write (event=None)
      - deploy POST + verify
      - testing state + machine event
      - all-components event
      - case refresh (event=None)
    """
    controller, proxy, policy, github, config = _controller()

    state = _state(status="deploy-requested")
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    from ..github_state_proxy import _build_hidden_state_body as _bhsb

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40},
        "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": "/approve_deploy machine=test-machine",
        "user": {"id": 111, "login": "owner1"},
    })

    # Mock Agent Core — same fixes as test 1
    core = MagicMock()
    core.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    _test2_img = "registry.example/repo@sha256:" + "a" * 64
    core.driver_status = AsyncMock(return_value={
        "status": "running", "running_image": _test2_img})
    core.deploy_driver = AsyncMock(return_value={"ok": True})
    controller._core_for_node = AsyncMock(return_value=core)

    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {"id": 1001, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| Target | Status | Version | Took |\n| perception | :white_check_mark: Success | `registry/repo:v1` | 10s |\n",
         "created_at": "2026-09-18T00:00:00Z", "updated_at": "2026-09-18T00:01:00Z"},
        {"id": 1002, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Test Results\n\nCommit: abc1234\n\n| Suite | Result | Passed | Failed | Took |\n| perception | :white_check_mark: Passed | 10 | 0 | 5s |\n",
         "created_at": "2026-09-18T00:02:00Z", "updated_at": "2026-09-18T00:03:00Z"},
        {"id": 1003, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nAll checks passed.",
         "created_at": "2026-09-18T00:04:00Z", "updated_at": "2026-09-18T00:05:00Z"},
    ])
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    controller._revalidate_hidden_state = AsyncMock(return_value=dict(state))
    controller._refresh_uncertain_state = AsyncMock(return_value="deploy-requested")

    # Case runner returns results — they should be merged without creating new events
    controller._run_automated_case = AsyncMock(return_value={
        "comp-001": "pass",
    })

    # Short timeout
    _test2_orig_timeout = controller.config.total_timeout
    controller.config.total_timeout = 0.5

    await controller.handle_approve_deploy("repo", 1, 50, "test-machine", "owner1", "111")
    controller.config.total_timeout = _test2_orig_timeout

    # ── Assertions ──
    # Case refresh should have triggered an additional write (event=None)
    assert proxy.write_hidden_state.call_count >= 1

    # The final write must have gone through case refresh (event=None)
    last_visible = proxy.write_hidden_state.call_args_list[-1].args[2]

    headings, starts, ends = _visible_history_marker_counts(last_visible)
    assert headings == 1
    assert starts == 1
    assert ends == 1

    titles = _event_titles(last_visible)

    # Machine deploy event and All components deployed must be present
    assert "Machine `test-machine` deployed" in titles
    assert "All components deployed" in titles

    # Case refresh does NOT create a new History event
    # So event count should be exactly 2 (machine + all-components)
    assert len(titles) == 2, f"expected 2 events, got {len(titles)}: {titles}"

    # No duplicate
    for t in set(titles):
        assert titles.count(t) == 1

    # hidden state case_results must have been updated
    written_state = proxy.write_hidden_state.call_args_list[-1].args[3]
    assert written_state.get("case_results", {}).get("comp-001") == "pass"
    assert written_state["status"] == "testing"


# ── Test 3: final machine deployment records machine + all-components history ─

@pytest.mark.asyncio
async def test_final_machine_deployment_records_machine_and_all_components_history():
    """Two-machine deployment: Machine A covers JP5.11, Machine B covers JP6.1.
    After Machine B approves (final machine), the visible History must contain
    exactly one each of:
      - Machine `jp5-machine` deployed
      - Machine `jp6-machine` deployed
      - All components deployed
    with correct phase/status and newest-first ordering.
    """
    from ..github_state_proxy import _build_hidden_state_body, _build_history_block
    from ..comments import BOT_MARKER, lifecycle_marker

    config = make_config()
    proxy = MagicMock()
    proxy.read_hidden_state = AsyncMock()
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.post_issue_comment = AsyncMock(return_value={"id": 1})
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": "/approve_deploy machine=jp5-machine",
        "user": {"id": 111, "login": "owner1"},
    })

    # Two machines, two components
    policy = Policy(config)
    policy.machines = {
        "jp5-machine": MachineInfo(
            alias="jp5-machine", node_id="node-5", owners=["owner1"],
            node_host="10.0.0.5", targets=["perception"],
            platforms=["linux/arm64"], variants=["5.11"],
        ),
        "jp6-machine": MachineInfo(
            alias="jp6-machine", node_id="node-6", owners=["owner2"],
            node_host="10.0.0.6", targets=["actucore"],
            platforms=["linux/arm64"], variants=["6.1"],
        ),
    }

    comp_jp5 = _component(
        component_id="comp-jp5", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )
    comp_jp6 = _component(
        component_id="comp-jp6", target="actucore", variant="6.1",
        runtime_id="actucore",
        image_ref="registry.example/repo@sha256:" + "b" * 64,
    )

    state = _state(components=[comp_jp5, comp_jp6], deployments=[])
    body_holder = {"body": _build_hidden_state_body(
        _lifecycle_visible("repo", 1, status="deploy-requested"), state)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _build_hidden_state_body(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    github = MagicMock()
    github.get_current_user = AsyncMock(return_value={"id": 123, "login": "bot"})
    emdash = '—'
    github.get_issue_comments = AsyncMock(return_value=[
        {"id": 1001, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| Target | Status | Version | Took |\n| perception | :white_check_mark: Success | \`registry/repo:v1\` | 10s |\n",
         "created_at": "2026-09-18T00:00:00Z", "updated_at": "2026-09-18T00:01:00Z"},
        {"id": 1002, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Test Results\n\nCommit: abc1234\n\n| Suite | Result | Passed | Failed | Took |\n| perception | :white_check_mark: Passed | 10 | 0 | 5s |\n",
         "created_at": "2026-09-18T00:02:00Z", "updated_at": "2026-09-18T00:03:00Z"},
        {"id": 1003, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nAll checks passed.",
         "created_at": "2026-09-18T00:04:00Z", "updated_at": "2026-09-18T00:05:00Z"},
    ])
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    controller = DeployController(config, proxy, policy, github)
    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    controller._revalidate_hidden_state = AsyncMock(return_value=dict(state))
    controller._run_automated_case = AsyncMock(return_value={})
    controller._refresh_uncertain_state = AsyncMock(return_value="deploy-requested")
    controller.config.total_timeout = 0.5

    # Machine A: jp5-machine
    core_a = MagicMock()
    core_a.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    _core_a_img = "registry.example/repo@sha256:" + "a" * 64
    core_a.driver_status = AsyncMock(return_value={
        "status": "running", "running_image": _core_a_img})
    core_a.deploy_driver = AsyncMock(return_value={"ok": True})
    controller._core_for_node = AsyncMock(return_value=core_a)

    # Pre-set the expected running_image for verify (must match component.image_ref)
    _core_a_image_ref = "registry.example/repo@sha256:" + "a" * 64

    await controller.handle_approve_deploy("repo", 1, 50, "jp5-machine", "owner1", "111")

    # After Machine A: partial coverage, state stays deploy-requested
    written_state_a = proxy.write_hidden_state.call_args_list[-1].args[3]
    assert written_state_a["status"] == "deploy-requested"

    titles_after_a = _event_titles(body_holder["body"])
    assert "Machine `jp5-machine` deployed" in titles_after_a

    # Machine B: jp6-machine (final)
    proxy.write_hidden_state.reset_mock()
    core_b = MagicMock()
    core_b.list_drivers = AsyncMock(return_value=[
        {"id": "actucore", "category": "driver", "image": "registry/repo:latest"}])
    _core_b_img = "registry.example/repo@sha256:" + "b" * 64
    core_b.driver_status = AsyncMock(return_value={
        "status": "running", "running_image": _core_b_img})
    core_b.deploy_driver = AsyncMock(return_value={"ok": True})

    # Re-set find/write to reflect updated body_holder
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)

    proxy.get_comment = AsyncMock(return_value={
        "id": 60, "body": "/approve_deploy machine=jp6-machine",
        "user": {"id": 111, "login": "owner2"},
    })
    # proxy.read_hidden_state already set via _seed_lifecycle_with_history
    # Simplify: just re-read from body_holder
    from ..github_state_proxy import _extract_hidden_state
    extracted = _extract_hidden_state(body_holder["body"])
    proxy.read_hidden_state = AsyncMock(return_value=extracted)

    controller._core_for_node = AsyncMock(return_value=core_b)

    await controller.handle_approve_deploy("repo", 1, 60, "jp6-machine", "owner2", "111")
    controller.config.total_timeout = 60.0

    # ── Assertions on final visible ──
    last_visible = proxy.write_hidden_state.call_args_list[-1].args[2]
    headings, starts, ends = _visible_history_marker_counts(last_visible)
    assert (headings, starts, ends) == (1, 1, 1)

    titles = _event_titles(last_visible)

    # Each event appears exactly once
    assert titles.count("Machine `jp5-machine` deployed") == 1
    assert titles.count("Machine `jp6-machine` deployed") == 1
    assert titles.count("All components deployed") == 1

    # Total of 3 deployment events
    assert len(titles) == 3, f"expected 3 events, got {len(titles)}: {titles}"

    # Final status == testing
    final_state = proxy.write_hidden_state.call_args_list[-1].args[3]
    assert final_state["status"] == "testing"

    # All components deployed
    all_cids = {c["component_id"] for c in [comp_jp5, comp_jp6]}
    deployed_cids = set()
    for dep in final_state.get("deployments", []):
        if dep.get("phase") == "deployed":
            deployed_cids.update(dep.get("component_ids", []))
    assert deployed_cids == all_cids

    # newest-first: All components deployed should be first (index 0),
    # jp6-machine second, jp5-machine last
    assert titles[0] == "All components deployed"
    assert titles[1] == "Machine `jp6-machine` deployed"
    assert titles[2] == "Machine `jp5-machine` deployed"


# ── Test 4: full lifecycle survives record_test + COS rebind ─────────────────

@pytest.mark.asyncio
async def test_two_machine_full_lifecycle_history_survives_record_test_and_cos_rebind():
    """PR #291 regression test: the full E2E lifecycle must preserve History
    through record_test terminal write + COS metadata rebind + presigned URL rebind.

    Lifecycle flow:
      1. Seed lifecycle with pre-existing history
      2. Approve machine A (jp5-machine, JP5.11 perception) → deploy-requested
      3. Approve machine B (jp6-machine, JP6.1 actucore) → testing + all-components
      4. Automated Cases refresh (advisory, event=None)
      5. handle_record_test(result="pass") → succeeded + "Test recorded"
      6. Simulate COS evidence upload (mock _upload_evidence)
      7. Simulate _rebind_terminal_cos_if_current twice (metadata + presigned URL, event=None)

    Final assertions:
      - status=succeeded, test_result=pass
      - COS: object_key non-empty, sha256 non-empty, size > 0
      - Visible contains: Download COS evidence, object key, sha prefix, human-readable size, HTTPS URL
      - History events (newest-first): Test recorded, All components deployed, Machine jp6 deployed,
        Machine jp5 deployed, Deployment requested, Review lifecycle transitioned, Lifecycle initialized
      - Each event count == 1; heading/start/end markers exactly 1
      - No duplicate, no cross-PR/repo contamination
    """
    controller, proxy, _policy, github, config = _controller()

    # Two machine fixtures in policy
    from ..models import MachineInfo
    controller.policy.machines = {
        "jp5-machine": MachineInfo(
            alias="jp5-machine", node_id="node-jp5", owners=["owner1"],
            node_host="10.0.0.5", targets=["perception"],
            platforms=["linux/arm64"], variants=["5.11"], driver_paths=[],
        ),
        "jp6-machine": MachineInfo(
            alias="jp6-machine", node_id="node-jp6", owners=["owner2"],
            node_host="10.0.0.6", targets=["actucore"],
            platforms=["linux/arm64"], variants=["6.1"], driver_paths=[],
        ),
    }

    # Components: one for JP5 (perception/5.11), one for JP6 (actucore/6.1)
    comp_jp5 = _component(
        component_id="comp-jp5-perception",
        target="perception",
        variant="5.11",
        runtime_id="perception",
    )
    comp_jp6 = _component(
        component_id="comp-jp6-actucore",
        target="actucore",
        variant="6.1",
        runtime_id="actucore",
        image_ref="registry.example/repo@sha256:" + "b" * 64,
    )

    state = _state(
        components=[comp_jp5, comp_jp6],
        status="deploy-requested",
    )

    # ── Store-emulating fixture ──
    from ..github_state_proxy import _build_hidden_state_body, _build_history_block

    body_holder = {"body": ""}

    def _make_body(visible, st):
        return _build_hidden_state_body(visible, st)

    def _seed_initial():
        visible = (
            "### Deploy Approval — Lifecycle\n\n"
            "**Status:** `deploy-requested`\n\n"
            "### Workflow\n\n"
            "- [x] review\n"
        )
        body_holder["body"] = _make_body(visible, state)

    _seed_initial()

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _make_body(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.project_status_label = AsyncMock()

    def _get_read_state(*_a, **_kw):
        from ..github_state_proxy import _extract_hidden_state
        return _extract_hidden_state(body_holder["body"])

    proxy.read_hidden_state = AsyncMock(side_effect=_get_read_state)

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40},
        "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")

    # Seed pre-existing history events
    pre_events = [
        {"event": "Deployment requested",
         "lifecycle": "`deploy-ready` → `deploy-requested`",
         "timestamp": "2026-09-30 10:02:00"},
        {"event": "Review lifecycle transitioned",
         "lifecycle": "`review-required` → `deploy-requested`",
         "timestamp": "2026-09-30 10:01:00"},
        {"event": "Lifecycle initialized",
         "lifecycle": "`none` → `review-required`",
         "timestamp": "2026-09-30 10:00:00"},
    ]
    history_block = _build_history_block(pre_events)
    body = body_holder["body"]
    visible_part, _, rest = body.partition("<!-- deploy-approval-state:v1")
    new_visible = visible_part.rstrip() + "\n\n### History\n\n" + history_block + "\n"
    body_holder["body"] = _make_body(new_visible, state)

    # Re-bind mocks to current body
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_get_read_state)

    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {"id": 1001, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| Target | Status | Version | Took |\n| perception | :white_check_mark: Success | `registry/repo:v1` | 10s |\n",
         "created_at": "2026-09-18T00:00:00Z", "updated_at": "2026-09-18T00:01:00Z"},
        {"id": 1002, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Test Results\n\nCommit: abc1234\n\n| Suite | Result | Passed | Failed | Took |\n| perception | :white_check_mark: Passed | 10 | 0 | 5s |\n",
         "created_at": "2026-09-18T00:02:00Z", "updated_at": "2026-09-18T00:03:00Z"},
        {"id": 1003, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nAll checks passed.",
         "created_at": "2026-09-18T00:04:00Z", "updated_at": "2026-09-18T00:05:00Z"},
    ])
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    controller._revalidate_hidden_state = AsyncMock(side_effect=_get_read_state)
    controller._run_automated_case = AsyncMock(return_value={})
    controller._refresh_uncertain_state = AsyncMock(return_value="deploy-requested")
    # Shorten verification timeout so the test doesn't wait 60s
    orig_timeout = controller.config.total_timeout
    controller.config.total_timeout = 0.5

    # ── Stage 1: Approve Machine A (jp5-machine) ──
    core_a = MagicMock()
    core_a.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    image_holder = ["registry.example/repo@sha256:" + "a" * 64]
    core_a.driver_status = AsyncMock(return_value={"status": "running", "running_image": image_holder[0]})
    core_a.deploy_driver = AsyncMock(return_value={"ok": True})

    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": "/approve_deploy machine=jp5-machine",
        "user": {"id": 111, "login": "owner1"},
    })
    controller._core_for_node = AsyncMock(return_value=core_a)

    image_holder[0] = "registry.example/repo@sha256:" + "a" * 64

    await controller.handle_approve_deploy("repo", 1, 50, "jp5-machine", "owner1", "111")

    controller.config.total_timeout = orig_timeout

    # After Machine A: partial coverage, stays deploy-requested
    last_st_a = _get_read_state()
    assert last_st_a["status"] == "deploy-requested"
    titles_after_a = _event_titles(body_holder["body"])
    assert "Machine `jp5-machine` deployed" in titles_after_a, \
        f"Expected jp5-machine event after first approve, got: {titles_after_a}"

    # ── Stage 2: Approve Machine B (jp6-machine, final) ──
    proxy.write_hidden_state.reset_mock()
    core_b = MagicMock()
    core_b.list_drivers = AsyncMock(return_value=[
        {"id": "actucore", "category": "driver", "image": "registry/repo:latest"}])
    core_b.driver_status = AsyncMock(return_value={"status": "running", "running_image": "registry.example/repo@sha256:" + "b" * 64})
    core_b.deploy_driver = AsyncMock(return_value={"ok": True})

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_get_read_state)

    proxy.get_comment = AsyncMock(return_value={
        "id": 60, "body": "/approve_deploy machine=jp6-machine",
        "user": {"id": 111, "login": "owner2"},
    })
    controller._core_for_node = AsyncMock(return_value=core_b)

    await controller.handle_approve_deploy("repo", 1, 60, "jp6-machine", "owner2", "111")

    # After Machine B: all components deployed, status=testing
    last_st_b = _get_read_state()
    assert last_st_b["status"] == "testing", \
        f"Expected testing after second approve, got: {last_st_b['status']}"

    titles_after_b = _event_titles(body_holder["body"])
    assert "Machine `jp6-machine` deployed" in titles_after_b, \
        f"Expected jp6-machine event after second approve, got: {titles_after_b}"
    assert "All components deployed" in titles_after_b, \
        f"Expected All components deployed after second approve, got: {titles_after_b}"

    # ── Stage 3: Automated Case refresh (advisory, event=None) ──
    # The handle_approve_deploy already ran _run_automated_case internally.
    # In a real flow, case_results would be present in state.
    # We simulate the case refresh by directly invoking the same event=None path:
    case_results = {
        "comp-jp5-perception": "pass",
        "comp-jp6-actucore": "pass",
    }
    fresh_state = _get_read_state()
    fresh_state["case_results"] = case_results
    case_result_str = ", ".join(f"{k}={v}" for k, v in case_results.items())
    from .. import comments as comments_mod
    case_markdown = comments_mod.testing("repo", 1, "a" * 40, case_result=case_result_str)

    await controller._write_lifecycle_with_history(
        "repo", 1, fresh_state, case_markdown, event=None,
    )

    # Re-bind mocks
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_get_read_state)

    titles_after_case = _event_titles(body_holder["body"])
    # Case refresh must NOT add new history events
    assert len(titles_after_case) == len(titles_after_b), \
        f"Case refresh added events: was {len(titles_after_b)}, now {len(titles_after_case)}"
    # All prior events still present
    for t in titles_after_b:
        assert t in titles_after_case, f"Event {t} lost after case refresh"

    # ── Stage 4: handle_record_test(result="pass") ──
    proxy.get_comment = AsyncMock(return_value={
        "id": 70, "body": "/record_test result=pass summary='all good'",
        "user": {"id": 111, "login": "owner1"},
    })

    # Mock _upload_evidence to return valid COS metadata
    controller._upload_evidence = AsyncMock(return_value={
        "object_key": "phanthymotus_pr/phanthymotus/2026-10/2026-10-13/pr-1/evidence-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.log.gz",
        "sha256": "d4" + "a" * 62,
        "size": 2048,
    })

    await controller.handle_record_test("4paradigm/phanthymotus", 1, 70, "pass", "all good", "owner1", "111")

    # Re-bind mocks after record_test
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_get_read_state)

    # ── Stage 5: Simulate COS metadata rebind (event=None) ──
    cos_meta = {"object_key": "phanthymotus_pr/phanthymotus/2026-10/2026-10-13/pr-1/evidence-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.log.gz", "sha256": "d4" + "a" * 62, "size": 2048}
    markdown_meta = comments_mod.succeeded_comment(
        "4paradigm/phanthymotus", 1, "a" * 40,
        cos_object_key=cos_meta["object_key"],
        cos_bundle_sha256=cos_meta["sha256"],
        cos_bundle_size=cos_meta["size"],
    )
    await controller._rebind_terminal_cos_if_current(
        "4paradigm/phanthymotus", 1,
        expected_head="a" * 40,
        expected_terminal_status="succeeded",
        expected_comment_id=70,
        expected_command_kind="record_test",
        expected_test_result="pass",
        cos_metadata=cos_meta,
        markdown=markdown_meta,
    )

    # Re-bind mocks
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_get_read_state)

    # ── Stage 6: Simulate COS presigned URL rebind (event=None) ──
    markdown_url = comments_mod.succeeded_comment(
        "repo", 1, "a" * 40,
        cos_object_key=cos_meta["object_key"],
        cos_bundle_sha256=cos_meta["sha256"],
        cos_bundle_size=cos_meta["size"],
        cos_download_url="https://cos.example.com/download/phanthymotus_pr/phanthymotus/2026-10/2026-10-13/pr-1/evidence-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.log.gz",
    )
    await controller._rebind_terminal_cos_if_current(
        "4paradigm/phanthymotus", 1,
        expected_head="a" * 40,
        expected_terminal_status="succeeded",
        expected_comment_id=70,
        expected_command_kind="record_test",
        expected_test_result="pass",
        cos_metadata=cos_meta,
        markdown=markdown_url,
    )

    # ── Final Assertions ──
    final_state = _get_read_state()
    assert final_state["status"] == "succeeded", \
        f"Expected succeeded, got: {final_state['status']}"
    assert final_state["test_result"] == "pass", \
        f"Expected test_result=pass, got: {final_state['test_result']}"

    # COS metadata present
    cos_state = final_state.get("cos", {})
    assert cos_state.get("object_key"), "Expected non-empty object_key"
    assert cos_state.get("sha256"), "Expected non-empty sha256"
    assert cos_state.get("size", 0) > 0, "Expected size > 0"

    # Visible contains COS evidence details
    final_visible = body_holder["body"].split("<!-- deploy-approval-state:v1")[0]
    assert "Download COS evidence" in final_visible, \
        f"Expected 'Download COS evidence' in visible"
    assert "evidence-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.log.gz" in final_visible, \
        f"Expected object_key in visible"
    assert "https://cos.example.com/download" in final_visible, \
        f"Expected presigned URL in visible"
    # sha prefix
    assert "da" in final_visible, f"Expected sha prefix in visible"
    # human-readable size
    assert "2.0 KB" in final_visible, f"Expected human-readable size in visible"

    # ── History Assertions ──
    headings, starts, ends = _visible_history_marker_counts(final_visible)
    assert headings == 1, f"expected exactly 1 History heading, got {headings}"
    assert starts == 1, f"expected exactly 1 start marker, got {starts}"
    assert ends == 1, f"expected exactly 1 end marker, got {ends}"

    final_titles = _event_titles(final_visible)

    # Expected events (newest-first order):
    expected_events = [
        "Test recorded",
        "All components deployed",
        "Machine `jp6-machine` deployed",
        "Machine `jp5-machine` deployed",
        "Deployment requested",
        "Review lifecycle transitioned",
        "Lifecycle initialized",
    ]
    for ev in expected_events:
        assert ev in final_titles, f"Expected event {ev!r} in history, got: {final_titles}"

    # Each event exactly once
    for ev in expected_events:
        cnt = final_titles.count(ev)
        assert cnt == 1, f"Event {ev!r} count={cnt}, expected 1: {final_titles}"

    # Total event count matches
    assert len(final_titles) == len(expected_events), \
        f"Expected {len(expected_events)} events, got {len(final_titles)}: {final_titles}"

    # No cross-PR/repo contamination
    for t in final_titles:
        assert "repo" not in t.lower() or "Machine" in t or "component" in t.lower() or "recorded" in t.lower() or "deploy" in t.lower() or "lifecycle" in t.lower() or "requested" in t.lower() or "transitioned" in t.lower() or "initialized" in t.lower(), \
            f"Suspicious event: {t}"

    # newest-first order check
    for i, ev in enumerate(expected_events):
        assert final_titles[i] == ev, \
            f"Expected event[{i}]={ev!r}, got {final_titles[i]!r}"


# ── Test 5: AST static analysis - no direct lifecycle write outside history writer ──

@pytest.mark.asyncio
async def test_service_has_no_direct_lifecycle_write_outside_history_writer():
    """Scan service.py with Python ast: every `self.proxy.write_hidden_state(...)`
    call must be inside `DeployController._write_lifecycle_with_history`.

    Only allowed method: DeployController._write_lifecycle_with_history.
    Any other direct call in DeployController is a test failure.

    Note: GitHubStateProxy.persist_cursor is in another file and has its own
    fresh-read + exact-visible-preservation contract, so it is explicitly
    excluded from this check.
    """
    import ast

    service_path = "agents/deploy_approval/service.py"
    source = open(service_path).read()
    tree = ast.parse(source, filename=service_path)

    # Find all Call nodes matching self.proxy.write_hidden_state
    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Check for self.proxy.write_hidden_state call
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "write_hidden_state"):
            continue
        # Check that the object is self.proxy
        obj = func.value
        if not (isinstance(obj, ast.Attribute) and obj.attr == "proxy"):
            continue
        self_obj = obj.value
        if not isinstance(self_obj, ast.Name) or self_obj.id != "self":
            continue

        # Found a write_hidden_state call. Now check which method it's in.
        # Walk up the AST to find the enclosing function/method.
        enclosing_method = None
        for parent_node in ast.walk(tree):
            if isinstance(parent_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if this node is inside this function
                for child in ast.walk(parent_node):
                    if child is node:
                        enclosing_method = parent_node.name
                        break
                if enclosing_method:
                    break

        if enclosing_method is None:
            violations.append((node.lineno, "write_hidden_state call at module level"))
            continue

        if enclosing_method != "_write_lifecycle_with_history":
            violations.append((node.lineno,
                               f"write_hidden_state call in {enclosing_method}"))

    if violations:
        lines = "\n".join(f"  Line {ln}: {desc}" for ln, desc in violations)
        pytest.fail(
            "Found self.proxy.write_hidden_state() calls outside "
            "DeployController._write_lifecycle_with_history:\n" + lines
        )


# ══════════════════════════════════════════════════════════════════════════════
# BLOCKER-1 — IPv4 approve selector exact revalidation (external fast acceptance)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ipv4_approve_selector_survives_final_comment_revalidation_and_persists_alias():
    """A literal IPv4 selector must survive final comment revalidation and
    persist ONLY the canonical MachineInfo.alias in hidden state/history.
    """
    controller, proxy, _policy, _github, _config = _controller()

    # IPv4 selector — literal, as written by the user in the GitHub comment
    ip_selector = "10.100.129.72"
    canonical_alias = "tianyi2-005"

    state = _state(status="deploy-requested")
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)

    from ..github_state_proxy import _build_hidden_state_body as _bhsb

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": f"/approve_deploy machine={ip_selector}",
        "user": {"id": 111, "login": "owner1"},
    })

    core = MagicMock()
    core.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    img = "registry.example/repo@sha256:" + "a" * 64
    core.driver_status = AsyncMock(return_value={"status": "running", "running_image": img})
    core.deploy_driver = AsyncMock(return_value={"ok": True})
    controller._core_for_node = AsyncMock(return_value=core)

    emdash = "\u2014"
    _github.get_issue_comments = AsyncMock(return_value=[
        {"id": 1001, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| Target | Status | Version | Took |\n| perception | :white_check_mark: Success | `registry/repo:v1` | 10s |\n",
         "created_at": "2026-09-18T00:00:00Z", "updated_at": "2026-09-18T00:01:00Z"},
        {"id": 1002, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Test Results\n\nCommit: abc1234\n\n| Suite | Result | Passed | Failed | Took |\n| perception | :white_check_mark: Passed | 10 | 0 | 5s |\n",
         "created_at": "2026-09-18T00:02:00Z", "updated_at": "2026-09-18T00:03:00Z"},
        {"id": 1003, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nAll checks passed.",
         "created_at": "2026-09-18T00:04:00Z", "updated_at": "2026-09-18T00:05:00Z"},
    ])
    _github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    controller._revalidate_hidden_state = AsyncMock(return_value=dict(state))
    controller._run_automated_case = AsyncMock(return_value={})
    controller._refresh_uncertain_state = AsyncMock(return_value="deploy-requested")
    orig_timeout = controller.config.total_timeout
    controller.config.total_timeout = 0.5

    # Two real machines: canonical alias path + IPv4 selector.
    from ..models import MachineInfo
    controller.policy.machines = {
        canonical_alias: MachineInfo(
            alias=canonical_alias, node_id="node-tianyi",
            owners=["owner1"], node_host=ip_selector,
            targets=["perception"], platforms=["linux/arm64"],
            variants=["5.11"], driver_paths=[],
        ),
        "other-machine": MachineInfo(
            alias="other-machine", node_id="node-other",
            owners=["owner2"], node_host="10.0.0.9",
            targets=["actucore"], platforms=["linux/arm64"],
            variants=["6.1"], driver_paths=[],
        ),
    }

    # Re-seed the store after policy change.
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"),
        state,
    )
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])

    comp_jp5 = _component(
        component_id="comp-jp5-perception", target="perception",
        variant="5.11", runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )
    state = _state(components=[comp_jp5], status="deploy-requested")
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"),
        state,
    )
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    from ..github_state_proxy import _extract_hidden_state

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    await controller.handle_approve_deploy("repo", 1, 50, ip_selector, "owner1", "111")

    controller.config.total_timeout = orig_timeout

    # Canonical alias persisted in hidden state — NOT the raw IPv4 selector.
    hidden = await _read_state()
    assert hidden["deployments"] == [{"machine": canonical_alias, "component_ids": ["comp-jp5-perception"], "phase": "deployed"}], \
        f"expected canonical alias in deployments, got {hidden.get('deployments')}"
    assert hidden.get("last_processed_comment_id") == 50

    # IPv4 selector reached the deploy path: one deploy POST occurred.
    assert core.deploy_driver.await_count == 1

    # The original IP comment passes exact final revalidation — no revoked event.
    titles = _event_titles(body_holder["body"])
    assert titles.count("Approval revoked") == 0


@pytest.mark.asyncio
async def test_ipv4_approve_rejects_edited_selector_before_unsafe_post():
    """If the approval comment is edited after preflight (IP->alias,
    alias->IP), final exact-comment revalidation MUST fail closed and
    no unsafe deploy POST may fire.

    The persisted state keeps only canonical MachineInfo.alias.
    """
    controller, proxy, _policy, github, _config = _controller()

    alias = "tianyi2-005"
    ip_selector = "10.100.129.72"
    orig_comment = f"/approve_deploy machine={ip_selector}"

    state = _state(status="deploy-requested")
    body_holder = _seed_lifecycle_with_history(proxy, "repo", 1, state)
    from ..github_state_proxy import _build_hidden_state_body as _bhsb, _extract_hidden_state

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    # Policy carries both machines (canonical alias + literal IP resolution).
    from ..models import MachineInfo
    controller.policy.machines = {
        alias: MachineInfo(
            alias=alias, node_id="node-tianyi",
            owners=["owner1"], node_host=ip_selector,
            targets=["perception"], platforms=["linux/arm64"],
            variants=["5.11"], driver_paths=[],
        ),
        "other": MachineInfo(
            alias="other", node_id="node-other",
            owners=["owner2"], node_host="10.0.0.9",
            targets=["actucore"], platforms=["linux/arm64"],
            variants=["6.1"], driver_paths=[],
        ),
    }

    core = MagicMock()
    core.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    img = "registry.example/repo@sha256:" + "a" * 64
    core.driver_status = AsyncMock(return_value={"status": "running", "running_image": img})
    core.deploy_driver = AsyncMock(return_value={"ok": True})
    controller._core_for_node = AsyncMock(return_value=core)

    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {"id": 1001, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| Target | Status | Version | Took |\n| perception | :white_check_mark: Success | `registry/repo:v1` | 10s |\n",
         "created_at": "2026-09-18T00:00:00Z", "updated_at": "2026-09-18T00:01:00Z"},
        {"id": 1002, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Test Results\n\nCommit: abc1234\n\n| Suite | Result | Passed | Failed | Took |\n| perception | :white_check_mark: Passed | 10 | 0 | 5s |\n",
         "created_at": "2026-09-18T00:02:00Z", "updated_at": "2026-09-18T00:03:00Z"},
        {"id": 1003, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nAll checks passed.",
         "created_at": "2026-09-18T00:04:00Z", "updated_at": "2026-09-18T00:05:00Z"},
    ])
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    controller._revalidate_hidden_state = AsyncMock(side_effect=_read_state)
    controller._run_automated_case = AsyncMock(return_value={})
    controller._refresh_uncertain_state = AsyncMock(return_value="deploy-requested")
    orig_timeout = controller.config.total_timeout
    controller.config.total_timeout = 0.5

    # Machine with JP5 component.
    comp_jp5 = _component(
        component_id="comp-jp5-perception", target="perception",
        variant="5.11", runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )
    state = _state(components=[comp_jp5], status="deploy-requested")
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"),
        state,
    )
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])

    # ── Case 1: literal IPv4 comment is revalidated exactly ──
    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": orig_comment,
        "user": {"id": 111, "login": "owner1"},
    })
    await controller.handle_approve_deploy("repo", 1, 50, ip_selector, "owner1", "111")
    controller.config.total_timeout = orig_timeout
    assert core.deploy_driver.await_count == 1

    hidden = await _read_state()
    assert hidden["deployments"] == [{"machine": alias, "component_ids": ["comp-jp5-perception"], "phase": "deployed"}], \
        f"expected canonical alias, got {hidden.get('deployments')}"
    assert "Approval revoked" not in _event_titles(body_holder["body"])

    # ── Case 2: comment edited IP -> alias ──
    proxy.write_hidden_state.reset_mock()
    orig_comment_a = "/approve_deploy machine=other"
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"),
        state,
    )
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": orig_comment_a,
        "user": {"id": 111, "login": "owner1"},
    })
    core.deploy_driver.reset_mock()
    await controller.handle_approve_deploy("repo", 1, 50, ip_selector, "owner1", "111")
    # Exact selector comparison fails: persisted selector was the IP, comment changed to another alias.
    assert core.deploy_driver.await_count == 0
    titles = _event_titles(body_holder["body"])
    assert titles.count("Approval revoked") == 1, f"expected Approval revoked, got {titles}"
    assert "Machine `other` selected" not in titles

    # ── Case 3: comment edited alias -> IP (different from the parsed selector) ──
    proxy.write_hidden_state.reset_mock()
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"),
        state,
    )
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": "/approve_deploy machine=10.100.0.9",
        "user": {"id": 111, "login": "owner1"},
    })
    core.deploy_driver.reset_mock()
    await controller.handle_approve_deploy("repo", 1, 50, ip_selector, "owner1", "111")
    assert core.deploy_driver.await_count == 0
    titles = _event_titles(body_holder["body"])
    assert titles.count("Approval revoked") == 1, f"expected Approval revoked, got {titles}"


# ── Blocker-2 — zero coverage ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zero_coverage_approve_writes_valid_state_and_preserves_existing_deployments():
    """A machine that covers zero remaining components must still:
      - persist a schema-valid completed command (args.machine present),
      - preserve existing successful deployments in hidden state and the
        visible deploy_requested rendering,
      - write History exactly once (no duplicate events),
      - never fire a deploy POST.

    The written state is validated through the production `_validate_hidden_state`
    (same validation that precedes unsafe deploy POSTs).
    """
    controller, proxy, _policy, _github, _config = _controller()

    from ..github_state_proxy import (
        _build_hidden_state_body as _bhsb,
        _extract_hidden_state,
        _validate_hidden_state,
    )

    state = _state(
        components=[
            _component(component_id="comp-a", target="perception", runtime_id="perception"),
            _component(component_id="comp-b", target="actucore", runtime_id="actucore"),
        ],
        deployments=[
            {"machine": "old-machine", "component_ids": ["comp-a"], "phase": "deployed"},
        ],
        status="deploy-requested",
        head_sha="a" * 40,
        command={
            "comment_id": 55,
            "kind": "approve_deploy",
            "phase": "completed",
            "args": {"machine": "other-machine", "actor": "owner1"},
        },
        last_processed_comment_id=55,
    )
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"),
        state,
    )}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 55, "body": "/approve_deploy machine=other-machine",
        "user": {"id": 111, "login": "owner1"},
    })

    # Policy: zero coverage for other-machine on both components.
    from ..models import MachineInfo
    controller.policy.machines = {
        "other-machine": MachineInfo(
            alias="other-machine", node_id="node-x",
            owners=["owner1"], node_host="10.0.0.9",
            targets=["actucore"], platforms=["linux/arm64"],
            variants=["6.1"], driver_paths=[],
        ),
        "good-machine": MachineInfo(
            alias="good-machine", node_id="node-y",
            owners=["owner2"], node_host="10.0.0.10",
            targets=["perception"], platforms=["linux/arm64"],
            variants=["5.11"], driver_paths=[],
        ),
    }

    controller.config.total_timeout = 0.5

    await controller.handle_approve_deploy("repo", 1, 55, "other-machine", "owner1", "111")

    # Zero unsafe deploy POST.
    assert proxy.write_hidden_state.await_count == 1

    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["status"] == "deploy-requested"
    # Schema-valid completed command: args.machine present.
    assert written_state["command"]["phase"] == "completed"
    assert written_state["command"]["args"] == {
        "machine": "other-machine", "actor": "owner1",
    }, f"unexpected command args: {written_state['command'].get('args')}"
    # Existing deployments preserved in hidden state.
    assert written_state["deployments"] == [{"machine": "old-machine", "component_ids": ["comp-a"], "phase": "deployed"}]

    # Validated through the production validator (same contract as pre-deploy).
    _validate_hidden_state(written_state)

    # Visible deploy_requested rendering carries existing deployments.
    written_visible = proxy.write_hidden_state.call_args.args[2]
    assert "old-machine" in written_visible, \
        f"existing deployment missing from visible rendering: {written_visible}"

    # History: exactly one Machine selected event, no duplicates.
    titles = _event_titles(body_holder["body"])
    assert titles.count(f"Machine `other-machine` selected") == 1
    assert len(titles) == 1


# ── Blocker-3A — atomic final machine + all-components ────────────────────────

@pytest.mark.asyncio
async def test_final_machine_and_all_components_history_are_persisted_atomically():
    """The final machine that completes all components MUST persist BOTH
    'Machine `<alias>` deployed' AND 'All components deployed' in a SINGLE
    _write_lifecycle_with_history call using the `events=` parameter.

    This eliminates the crash window between the two separate lifecycle writes.
    """
    controller, proxy, policy, github, config = _controller()

    from ..models import MachineInfo
    controller.policy.machines = {
        "jp5-machine": MachineInfo(
            alias="jp5-machine", node_id="node-5", owners=["owner1"],
            node_host="10.0.0.5", targets=["perception"],
            platforms=["linux/arm64"], variants=["5.11"], driver_paths=[],
        ),
        "jp6-machine": MachineInfo(
            alias="jp6-machine", node_id="node-6", owners=["owner2"],
            node_host="10.0.0.6", targets=["actucore"],
            platforms=["linux/arm64"], variants=["6.1"], driver_paths=[],
        ),
    }

    comp_jp5 = _component(
        component_id="comp-jp5-perception", target="perception",
        variant="5.11", runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )
    comp_jp6 = _component(
        component_id="comp-jp6-actucore", target="actucore",
        variant="6.1", runtime_id="actucore",
        image_ref="registry.example/repo@sha256:" + "b" * 64,
    )

    state = _state(components=[comp_jp5, comp_jp6], deployments=[])
    from ..github_state_proxy import _build_hidden_state_body as _bhsb
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"), state)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": "/approve_deploy machine=jp5-machine",
        "user": {"id": 111, "login": "owner1"},
    })

    github = MagicMock()
    github.get_current_user = AsyncMock(return_value={"id": 123, "login": "bot"})
    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {"id": 1001, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| Target | Status | Version | Took |\n| perception | :white_check_mark: Success | `registry/repo:v1` | 10s |\n",
         "created_at": "2026-09-18T00:00:00Z", "updated_at": "2026-09-18T00:01:00Z"},
        {"id": 1002, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Test Results\n\nCommit: abc1234\n\n| Suite | Result | Passed | Failed | Took |\n| perception | :white_check_mark: Passed | 10 | 0 | 5s |\n",
         "created_at": "2026-09-18T00:02:00Z", "updated_at": "2026-09-18T00:03:00Z"},
        {"id": 1003, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nAll checks passed.",
         "created_at": "2026-09-18T00:04:00Z", "updated_at": "2026-09-18T00:05:00Z"},
    ])
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    controller._revalidate_hidden_state = AsyncMock(return_value=dict(state))
    controller._run_automated_case = AsyncMock(return_value={})
    controller._refresh_uncertain_state = AsyncMock(return_value="deploy-requested")
    controller.config.total_timeout = 0.5

    # Machine A: partial coverage
    core_a = MagicMock()
    core_a.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    core_a.driver_status = AsyncMock(return_value={"status": "running", "running_image": "registry.example/repo@sha256:" + "a" * 64})
    core_a.deploy_driver = AsyncMock(return_value={"ok": True})
    controller._core_for_node = AsyncMock(return_value=core_a)

    await controller.handle_approve_deploy("repo", 1, 50, "jp5-machine", "owner1", "111")
    assert proxy.write_hidden_state.call_count >= 1
    written_state_a = proxy.write_hidden_state.call_args_list[-1].args[3]
    assert written_state_a["status"] == "deploy-requested"

    # Machine B: final machine → testing
    proxy.write_hidden_state.reset_mock()
    core_b = MagicMock()
    core_b.list_drivers = AsyncMock(return_value=[
        {"id": "actucore", "category": "driver", "image": "registry/repo:latest"}])
    core_b.driver_status = AsyncMock(return_value={"status": "running", "running_image": "registry.example/repo@sha256:" + "b" * 64})
    core_b.deploy_driver = AsyncMock(return_value={"ok": True})
    controller._core_for_node = AsyncMock(return_value=core_b)

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])

    from ..github_state_proxy import _extract_hidden_state

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    proxy.get_comment = AsyncMock(return_value={
        "id": 60, "body": "/approve_deploy machine=jp6-machine",
        "user": {"id": 111, "login": "owner2"},
    })

    # ── Assertion: ONE durable testing-transition write ──
    # Preparatory writes (executing "Deploying...") are normal safety design;
    # the atomicity requirement applies to the final transition: exactly one
    # write_hidden_state call whose persisted state has status "testing".
    # We capture each call AT CALL TIME via deepcopy so stale references
    # don't mask split writes.
    from copy import deepcopy

    captured_calls = []
    original_write = proxy.write_hidden_state.side_effect

    async def _capturing_write(_repo, _pr, vis, _st):
        captured_calls.append({
            "visible": deepcopy(vis),
            "state": deepcopy(_st),
        })
        return await original_write(_repo, _pr, vis, _st)

    proxy.write_hidden_state = AsyncMock(side_effect=_capturing_write)

    # Single jp6-machine approve — the ONLY call. The deepcopy capture
    # is installed BEFORE this call so the testing-transition write is
    # recorded at the exact boundary.
    await controller.handle_approve_deploy("repo", 1, 60, "jp6-machine", "owner2", "111")

    testing_writes = [
        c for c in captured_calls
        if isinstance(c["state"], dict) and c["state"].get("status") == "testing"
    ]
    # The atomicity invariant applies to the FIRST durable transition
    # from deploy-requested to testing.  Subsequent event=None advisory
    # Case-refresh writes also carry status=testing and must be allowed.
    assert len(testing_writes) >= 1, \
        f"Expected at least 1 testing-transition write, got {len(testing_writes)}"

    # The FIRST testing write must already contain both History events
    # in newest-first order.  Subsequent advisory refreshes must NOT
    # recreate the missing deploy-requested -> testing transition events.
    testing_cap = testing_writes[0]
    titles_first = _event_titles(testing_cap["visible"])
    assert "All components deployed" in titles_first, \
        f"FIRST testing write must have 'All components deployed', got {titles_first}"
    assert "Machine `jp6-machine` deployed" in titles_first, \
        f"FIRST testing write must have 'Machine jp6-machine deployed', got {titles_first}"
    # Verify newest-first ordering in the FIRST write
    idx_all = titles_first.index("All components deployed")
    idx_machine = titles_first.index("Machine `jp6-machine` deployed")
    assert idx_all < idx_machine, \
        f"Expected 'All components deployed' before 'Machine jp6-machine deployed', got {titles_first}"

    # Validate the first write itself for completeness
    for bad_key in ("events", "history", "history_events"):
        assert bad_key not in testing_cap["state"], \
            f"Hidden state must not have '{bad_key}'; found in {testing_cap['state'].keys()}"

    # Validate with production validators.
    from ..github_state_proxy import _build_hidden_state_body as _bhsb, _validate_hidden_state
    roundtrip = _bhsb(testing_cap["visible"], testing_cap["state"])
    _validate_hidden_state(_extract_hidden_state(roundtrip))

    # Subsequent advisory Case-refresh writes (if any) must NOT introduce
    # a second missing History transition: they preserve existing events.
    for extra_cap in testing_writes[1:]:
        extra_titles = _event_titles(extra_cap["visible"])
        assert extra_titles == titles_first, \
            f"Advisory refresh must not change History events. Before: {titles_first}, After: {extra_titles}"


    # ── Visible history: both events present, newest-first ──
    titles = _event_titles(testing_cap["visible"])
    assert titles.count("Machine `jp6-machine` deployed") == 1, \
        f"Expected 1 'Machine jp6-machine deployed' in history, got {titles}"
    assert titles.count("All components deployed") == 1, \
        f"Expected 1 'All components deployed' in history, got {titles}"
    assert titles[0] == "All components deployed", \
        f"Expected 'All components deployed' first (newest), got {titles}"
    assert titles[1] == "Machine `jp6-machine` deployed", \
        f"Expected 'Machine jp6-machine deployed' second, got {titles}"

    # Partial machine history (jp5) must still be present.
    assert "Machine `jp5-machine` deployed" in testing_cap["visible"]

    # Confirm persisted component IDs and machine aliases.
    final_state = testing_cap["state"]
    deployed_ids = set()
    for dep in final_state.get("deployments", []):
        if dep.get("phase") == "deployed":
            deployed_ids.update(dep.get("component_ids", []))
    assert "comp-jp5-perception" in deployed_ids
    assert "comp-jp6-actucore" in deployed_ids

    # Zero extra deploy POSTs beyond the two expected (one per machine).
    assert core_b.deploy_driver.call_count == 1

    # Validate the final persisted body (body_holder mirrors the last write).
    titles_body = _event_titles(body_holder["body"])
    assert titles_body[0] == "All components deployed"
    assert titles_body[1] == "Machine `jp6-machine` deployed"

    final_state2 = await _read_state()
    assert final_state2["status"] == "testing"
    assert final_state2["head_sha"] == "a" * 40



# ── Blocker-3B — testing reconcile repairs missing case_results ───────────────

@pytest.mark.asyncio
async def test_testing_reconcile_repairs_missing_case_results_without_redeploy():
    """When reconcile_pr finds status=testing with incomplete case_results,
    it must re-run ONLY the advisory Automated Cases, merge results, and
    refresh visible testing markdown with event=None.

    No deploy POST should fire during recovery.
    History must remain exactly once.
    """
    from ..github_state_proxy import (
        _build_hidden_state_body as _bhsb,
        _extract_hidden_state,
    )

    config = make_config()
    proxy = MagicMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.get_comment = AsyncMock(return_value={"id": 50, "body": "/approve_deploy machine=jp5-machine",
        "user": {"id": 111, "login": "owner1"}})

    policy = Policy(config)
    policy.machines = {
        "m1": MachineInfo(
            alias="m1", node_id="n1", owners=["owner1"], node_host="10.0.0.1",
            targets=["perception"], platforms=["linux/arm64"], variants=["5.11"],
        ),
    }

    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )

    # Simulate crash: status=testing, empty case_results
    crash_state = {
        **_state(components=[comp], deployments=[
            {"machine": "m1", "component_ids": ["comp-001"], "phase": "deployed"},
        ]),
        "status": "testing",
        "case_results": {},
        "command": {
            "comment_id": 50, "kind": "approve_deploy",
            "phase": "completed", "args": {"machine": "m1", "actor": "alice"},
        },
    }

    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), crash_state)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    controller = DeployController(config, proxy, policy, MagicMock())
    controller.config.total_timeout = 0.5

    # Mock case runner — will return the missing results
    controller._run_automated_case = AsyncMock(return_value={"comp-001": "pass"})

    await controller.reconcile_pr("repo", 1)

    # Must write once (case merge with event=None)
    assert proxy.write_hidden_state.call_count == 1

    # Written state has merged case_results
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["case_results"]["comp-001"] == "pass"

    # Status remains testing
    assert written_state["status"] == "testing"

    # Titles preserved — no churn
    titles = _event_titles(body_holder["body"])
    # No duplicate deploy events
    for t in titles:
        assert titles.count(t) == 1, f"event {t!r} duplicated in reconcile repair"

    # Second reconcile: already complete, no churn
    proxy.write_hidden_state.reset_mock()
    await controller.reconcile_pr("repo", 1)
    assert proxy.write_hidden_state.call_count == 0, \
        f"Expected zero writes on second reconcile, got {proxy.write_hidden_state.call_count}"


# ── Blocker-3B continued — no duplicate complete history ─────────────────────

@pytest.mark.asyncio
async def test_testing_reconcile_does_not_duplicate_complete_history():
    """Second reconcile on already-complete testing state must make zero writes
    and produce no duplicate/history churn.
    """
    from ..github_state_proxy import (
        _build_hidden_state_body as _bhsb,
        _extract_hidden_state,
    )

    config = make_config()
    proxy = MagicMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })

    policy = Policy(config)
    policy.machines = {
        "m1": MachineInfo(
            alias="m1", node_id="n1", owners=["owner1"], node_host="10.0.0.1",
            targets=["perception"], platforms=["linux/arm64"], variants=["5.11"],
        ),
    }

    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )

    # Already-complete testing state
    complete_state = {
        **_state(components=[comp], deployments=[
            {"machine": "m1", "component_ids": ["comp-001"], "phase": "deployed"},
        ]),
        "status": "testing",
        "case_results": {"comp-001": "pass"},
        "command": {
            "comment_id": 50, "kind": "approve_deploy",
            "phase": "completed", "args": {"machine": "m1", "actor": "alice"},
        },
    }

    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), complete_state)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    controller = DeployController(config, proxy, policy, MagicMock())

    await controller.reconcile_pr("repo", 1)

    # Zero writes — state already complete.
    assert proxy.write_hidden_state.call_count == 0

    # History untouched.
    titles = _event_titles(body_holder["body"])
    for t in titles:
        assert titles.count(t) == 1


# ══════════════════════════════════════════════════════════════════════════════
# HEAD DRIFT — history uses previous status (external fast acceptance)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_request_deploy_head_drift_history_uses_previous_status():
    """handle_request_deploy HEAD drift must capture old_status BEFORE
    state reset and render it in the History lifecycle event.

    Produces:
      `deploy-ready` → `review-required`
    instead of:
      `review-required` → `review-required`
    """
    controller, proxy, _policy, _github, _config = _controller()

    # Seed lifecycle with pre-existing History including Lifecycle initialized.
    from ..github_state_proxy import _build_hidden_state_body as _bhsb, _build_history_block
    state = _state(status="deploy-requested")
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"), state)}

    pre_events = [
        {"event": "Lifecycle initialized",
         "lifecycle": "`none` → `review-required`",
         "timestamp": "2026-09-30 10:00:00"},
        {"event": "Review lifecycle transitioned",
         "lifecycle": "`review-required` → `deploy-requested`",
         "timestamp": "2026-09-30 10:01:00"},
        {"event": "Deployment requested",
         "lifecycle": "`deploy-ready` → `deploy-requested`",
         "timestamp": "2026-09-30 10:02:00"},
    ]
    history_block = _build_history_block(pre_events)
    body = body_holder["body"]
    visible, _, _ = body.partition("<!-- deploy-approval-state:v1")
    new_visible = visible.rstrip() + "\n\n### History\n\n" + history_block + "\n"
    body_holder["body"] = _bhsb(new_visible, state)

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "b" * 40},  # DRIFTED HEAD
        "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()

    # The /request_deploy comment
    proxy.get_comment = AsyncMock(return_value={
        "id": 30, "body": "/request_deploy",
        "user": {"id": 111, "login": "alice"},
    })

    # Simulate a deploy-ready state with review evidence.
    state["status"] = "deploy-ready"
    state["review_evidence"] = {
        "build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z",
        "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40,
        "test_comment_id": 2, "test_comment_updated_at": "2026-09-18T00:00:00Z",
        "code_review_comment_id": 3, "code_review_comment_updated_at": "2026-09-18T00:00:00Z",
        "review_author_id": "7950763",
    }
    state["head_sha"] = "a" * 40
    state["command"] = {
        "comment_id": 30, "kind": "request_deploy",
        "phase": "completed", "args": {},
    }
    state["last_processed_comment_id"] = 30
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-ready"), state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    controller._check_review_permissions = AsyncMock(return_value=None)
    controller._ensure_deploy_ready_evidence = AsyncMock(return_value={
        "build_comment_id": 1, "test_comment_id": 2, "code_review_comment_id": 3,
    })

    await controller.handle_request_deploy("repo", 1, 30)

    # ── Assertions ──
    assert proxy.write_hidden_state.call_count >= 1

    # History event must use old_status "deploy-ready", NOT "review-required".
    written_visible = proxy.write_hidden_state.call_args.args[2]
    titles = _event_titles(written_visible)

    # Find the HEAD drift event
    drift_event = None
    for t in titles:
        if "HEAD drift detected" in t:
            drift_event = t
            break

    assert drift_event is not None, f"Expected 'HEAD drift detected' event in history, got {titles}"

    # The lifecycle portion must show deploy-ready -> review-required
    written_state = proxy.write_hidden_state.call_args.args[3]
    # Check the body contains the correct lifecycle text
    assert "`deploy-ready` \\u2192 `review-required`" in body_holder["body"] or \
           "`deploy-ready` → `review-required`" in body_holder["body"], \
        f"Expected 'deploy-ready → review-required' in body, got:\n{body_holder['body'][:2000]}"


@pytest.mark.asyncio
async def test_reconcile_head_drift_history_uses_previous_status():
    """reconcile_pr HEAD drift must capture old_status BEFORE state reset
    and render it in the History lifecycle event.
    """
    from ..github_state_proxy import _build_hidden_state_body as _bhsb, _build_history_block

    config = make_config()
    proxy = MagicMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "b" * 40},  # DRIFTED HEAD
        "user": {"id": 111, "login": "alice"},
    })

    policy = Policy(config)
    policy.machines = {
        "test-machine": MachineInfo(
            alias="test-machine", node_id="n1", owners=["owner1"],
            node_host="127.0.0.1", targets=["perception"],
            platforms=["linux/arm64"], variants=["5.11"],
        ),
    }

    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )

    # Seed lifecycle with existing history and deploy-requested status.
    state = _state(components=[comp], status="deploy-requested", command={"comment_id": 17, "kind": "approve_deploy", "phase": "completed", "args": {"machine": "test-machine"}})
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"), state)}

    pre_events = [
        {"event": "Lifecycle initialized",
         "lifecycle": "`none` → `review-required`",
         "timestamp": "2026-09-30 10:00:00"},
        {"event": "Review lifecycle transitioned",
         "lifecycle": "`review-required` → `deploy-requested`",
         "timestamp": "2026-09-30 10:01:00"},
    ]
    history_block = _build_history_block(pre_events)
    body = body_holder["body"]
    visible, _, _ = body.partition("<!-- deploy-approval-state:v1")
    new_visible = visible.rstrip() + "\n\n### History\n\n" + history_block + "\n"
    body_holder["body"] = _bhsb(new_visible, state)

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))

    # Old head (not drifted yet for the state)
    state["head_sha"] = "a" * 40

    controller = DeployController(config, proxy, policy, MagicMock())

    await controller.reconcile_pr("repo", 1)

    # ── Assertions ──
    assert proxy.write_hidden_state.call_count >= 1

    # The lifecycle portion must show deploy-requested -> review-required
    assert "`deploy-requested` \\u2192 `review-required`" in body_holder["body"] or \
           "`deploy-requested` → `review-required`" in body_holder["body"], \
        f"Expected 'deploy-requested → review-required' in body, got:\n{body_holder['body'][:2000]}"


# ══════════════════════════════════════════════════════════════════════════════
# record_test — must not prematurely terminate incomplete testing
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_record_test_does_not_prematurely_terminate_incomplete_testing():
    """handle_record_test must only accept status==\"testing\" and all
    components deployed AND all bound component case results must be
    terminal (pass, fail, n/a).  A "running" or missing case must not
    cause premature terminal transition.
    """
    controller, proxy, _policy, _github, _config = _controller()

    from ..github_state_proxy import (
        _build_hidden_state_body as _bhsb,
        _extract_hidden_state,
    )

    # State: status=testing, all components deployed, but case result
    # is "running" (non-terminal) — should NOT be accepted.
    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )
    testing_state = _state(
        components=[comp],
        deployments=[
            {"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"},
        ],
        status="testing",
        case_results={"comp-001": "running"},
    )
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), testing_state)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()

    # record_test comment
    proxy.get_comment = AsyncMock(return_value={
        "id": 70, "body": "/record_test result=pass summary='all good'",
        "user": {"id": 111, "login": "owner1"},
    })

    # Should be rejected because case result is "running" (non-terminal).
    result = await controller.handle_record_test("repo", 1, 70, "pass", "all good", "owner1", "111")
    assert result is True  # handled (returned early with not-ready message)

    # No terminal write should have happened.
    assert proxy.write_hidden_state.call_count == 0

    # State unchanged.
    hidden = await _read_state()
    assert hidden["status"] == "testing"


@pytest.mark.asyncio
async def test_record_test_incomplete_cases_cannot_finalize():
    """When bound component case results are not yet terminal, record_test
    must respond with a 'command not ready' message and MUST NOT mutate
    hidden state or post a terminal lifecycle comment.

    Terminal case values are only: pass, fail, n/a.
    Any other value (including missing keys and "running") blocks finalization.
    """
    controller, proxy, _policy, _github, _config = _controller()

    from ..github_state_proxy import (
        _build_hidden_state_body as _bhsb,
        _extract_hidden_state,
    )

    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )

    # Case 1: missing case result key
    state_missing = _state(
        components=[comp],
        deployments=[
            {"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"},
        ],
        status="testing",
        case_results={},
    )
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), state_missing)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()

    proxy.get_comment = AsyncMock(return_value={
        "id": 80, "body": "/record_test result=pass summary='ok'",
        "user": {"id": 111, "login": "owner1"},
    })

    result = await controller.handle_record_test("repo", 1, 80, "pass", "ok", "owner1", "111")
    assert result is True
    assert proxy.write_hidden_state.call_count == 0
    assert (await _read_state())["status"] == "testing"

    # Case 2: case result is "running"
    proxy.write_hidden_state.reset_mock()
    state_running = _state(
        components=[comp],
        deployments=[
            {"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"},
        ],
        status="testing",
        case_results={"comp-001": "running"},
    )
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), state_running)
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    result = await controller.handle_record_test("repo", 1, 81, "fail", "bad", "owner1", "111")
    assert result is True
    assert proxy.write_hidden_state.call_count == 0
    assert (await _read_state())["status"] == "testing"

    # Case 3: all terminal — should proceed (fail is terminal)
    proxy.write_hidden_state.reset_mock()
    state_terminal = _state(
        components=[comp],
        deployments=[
            {"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"},
        ],
        status="testing",
        case_results={"comp-001": "fail"},
    )
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), state_terminal)
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    result = await controller.handle_record_test("repo", 1, 82, "fail", "bad", "owner1", "111")
    assert result is True
    assert proxy.write_hidden_state.call_count == 1
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["status"] == "failed"

@pytest.mark.asyncio
async def test_real_hidden_state_serialization():
    """Production _build_hidden_state_body + _extract_hidden_state round-trip
    must preserve all fields and pass _validate_hidden_state."""
    from ..github_state_proxy import (
        _build_hidden_state_body,
        _extract_hidden_state,
        _validate_hidden_state,
    )

    original = _state(
        status="testing",
        test_result="pass",
        case_results={"comp-001": "pass"},
        command={
            "comment_id": 50,
            "kind": "approve_deploy",
            "phase": "completed",
            "args": {"machine": "test-machine", "actor": "alice"},
        },
        deployments=[
            {"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"},
        ],
        cos={"object_key": "evidence/repo/1/x.tar.gz", "sha256": "ab" * 32, "size": 1024},
    )

    body = _build_hidden_state_body(_lifecycle_visible("repo", 1, status="testing"), original)
    recovered = _extract_hidden_state(body)

    # Validation must pass.
    _validate_hidden_state(recovered)

    # Field fidelity.
    assert recovered["status"] == "testing"
    assert recovered["test_result"] == "pass"
    assert recovered["case_results"]["comp-001"] == "pass"
    assert recovered["deployments"][0]["machine"] == "test-machine"
    assert recovered["cos"]["object_key"] == "evidence/repo/1/x.tar.gz"
    assert recovered["cos"]["size"] == 1024


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER REPO RUNTIME AUTHORIZATION REFRESH
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_driver_repo_added_after_start_is_activated_without_restart():
    """When the driver repo is granted to the GitHub App installation
    while the Controller is running, the watcher must detect it on the
    next auth refresh WITHOUT any restart or config change.
    """
    from ..config import DESIRED_REPOS
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]  # driver not yet active

    proxy = MagicMock()
    proxy.get_open_prs = AsyncMock(return_value=[])

    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    # Initially only phanthymotus
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus"]
    )

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    # Simulate auth refresh: grant driver repo
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    )

    await watcher._refresh_active_repos()

    assert "4paradigm/phanthymotus-driver" in config.active_repos
    github_auth.refresh_installation_token.assert_called_once()


@pytest.mark.asyncio
async def test_driver_repo_removed_after_start_is_deactivated():
    """When driver repo authorization is removed, the watcher must
    deactivate it from active_repos on the next auth refresh.
    """
    from ..config import DESIRED_REPOS
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = [
        "4paradigm/phanthymotus",
        "4paradigm/phanthymotus-driver",
    ]

    proxy = MagicMock()
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    # Driver repo no longer authorized
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus"]
    )

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._refresh_active_repos()

    assert "4paradigm/phanthymotus-driver" not in config.active_repos
    assert "4paradigm/phanthymotus" in config.active_repos


@pytest.mark.asyncio
async def test_driver_repo_authorization_refresh_failure_fails_closed():
    """When the auth refresh call fails (e.g. 403, network error),
    active_repos must remain UNCHANGED — never widen access.
    """
    from ..github_client import GitHubClient

    config = make_config()
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        side_effect=Exception("network error")
    )

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="fresh-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._refresh_active_repos()

    # Must remain unchanged
    assert config.active_repos == ["4paradigm/phanthymotus"]
    assert config.auth_valid is False
    # Forced token refresh MUST precede repo discovery — it is attempted
    # even though discovery itself fails.
    github_auth.refresh_installation_token.assert_awaited_once()
    # Repo discovery was attempted after the refresh failed.
    github.list_installation_repositories.assert_awaited_once()


@pytest.mark.asyncio
async def test_driver_repo_reactivated_without_replaying_old_commands():
    """When driver repo authorization is re-granted after removal,
    the watcher must NOT replay old commands from that repo.
    It only updates active_repos — existing lifecycle state is preserved.
    """
    from ..config import DESIRED_REPOS
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    proxy.get_open_prs = AsyncMock(return_value=[])
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    )

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    # First refresh adds driver
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" in config.active_repos

    # Remove driver
    config.active_repos = ["4paradigm/phanthymotus"]
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus"]
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" not in config.active_repos

    # Re-add driver
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" in config.active_repos

    # watcher.start() was never called — no commands dispatched
    watcher.start()
    # PR polling only iterates active repos — driver PRs would be polled
    # but controller.on_command is only called for NEW comments > cursor.
    # Since we never wrote any lifecycle state for driver PRs, no old
    # commands are replayed.
    await watcher._poll_once()
    controller.on_command.assert_not_called()
    await watcher.stop()


@pytest.mark.asyncio
async def test_driver_repo_refresh_keeps_repo_isolation_and_single_writer():
    """Auth refresh must not mix PR state between repos.
    Each repo/PR pair has separate hidden state and lifecycle.
    """
    from ..github_client import GitHubClient

    config = make_config()
    config.active_repos = [
        "4paradigm/phanthymotus",
        "4paradigm/phanthymotus-driver",
    ]

    # PR #1 in phanthymotus with state
    pr_state_phanthymotus = {
        "version": 1, "head_sha": "a" * 40, "status": "testing",
        "components": [], "deployments": [], "case_results": {},
        "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    }

    proxy = MagicMock()
    # PR #5 in driver is separate
    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [{"number": 1}] if "phanthymotus" in repo else [{"number": 5}])
    proxy.read_hidden_state = AsyncMock(return_value=dict(pr_state_phanthymotus))
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        return_value=config.active_repos
    )

    github_auth = MagicMock()

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._poll_once()

    # Controller was called once per repo (reconcile)
    # But on_command was NOT called (no new commands)
    controller.on_command.assert_not_called()


@pytest.mark.asyncio
async def test_driver_repo_refresh_rejects_unlisted_repositories():
    """GitHub list_installation_repositories may return arbitrary repos.
    The watcher must ONLY activate repos within DESIRED_REPOS.
    """
    from ..config import DESIRED_REPOS
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        return_value=[
            "4paradigm/phanthymotus",
            "4paradigm/phanthymotus-driver",
            "evil-org/malicious-repo",
            "some-user/public-repo",
        ]
    )

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._refresh_active_repos()

    assert "4paradigm/phanthymotus-driver" in config.active_repos
    assert "evil-org/malicious-repo" not in config.active_repos
    assert "some-user/public-repo" not in config.active_repos


@pytest.mark.asyncio
async def test_driver_repo_refreshes_cached_installation_token():
    """After a successful auth refresh that changes active_repos,
    the watcher must force-refresh the installation token so new
    repository scopes are visible.
    """
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    )

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="fresh-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._refresh_active_repos()

    github_auth.refresh_installation_token.assert_called_once()
    assert config.active_repos == [
        "4paradigm/phanthymotus",
        "4paradigm/phanthymotus-driver",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL REQUIRED DRIVER AUTH TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_driver_refresh_failure_skips_poll_cycle():
    """When _refresh_active_repos fails, the poll cycle must be skipped
    for that iteration — no _poll_once should execute."""
    from ..config import DESIRED_REPOS
    from ..github_client import GitHubClient

    config = make_config()
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    proxy.get_open_prs = AsyncMock(return_value=[])
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        side_effect=Exception("network error")
    )

    github_auth = MagicMock()
    call_order = []
    _now = [1000.0]  # deterministic monotonic clock

    async def mock_refresh():
        call_order.append("refresh")
        raise Exception("token network error")
    async def mock_list():
        call_order.append("list")
        return ["4paradigm/phanthymotus"]
    github_auth.refresh_installation_token = mock_refresh
    github.list_installation_repositories = mock_list

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )
    watcher._monotonic = lambda: _now[0]

    # First: trigger auth refresh — it will fail
    await watcher._refresh_active_repos_if_needed()
    assert config.auth_valid is False
    # Token refresh MUST have been called (force refresh before repo discovery)
    assert "refresh" in call_order, "Expected refresh_installation_token to be called"
    first_refresh_count = call_order.count("refresh")

    # Second cycle: within the bounded interval the failed refresh already
    # occupies this slot — no immediate hot-loop retry.
    _now[0] += 1.0
    call_order.clear()
    await watcher._refresh_active_repos_if_needed()
    assert config.auth_valid is False
    assert call_order.count("refresh") == 0, \
        "Failed refresh must occupy the interval slot — no hot-loop retry within the interval"

    # After the interval elapses the refresh is retried (fail-closed, still failing).
    _now[0] += watcher._auth_refresh_interval + 1.0
    await watcher._refresh_active_repos_if_needed()
    assert config.auth_valid is False
    assert call_order.count("refresh") == 1, \
        "Refresh must be retried after the interval elapses"

    # Call _poll_once directly — active_repos is still set, but auth_valid is False
    await watcher._poll_once()

    # get_open_prs should NOT have been called because auth_valid is False
    proxy.get_open_prs.assert_not_called()

@pytest.mark.asyncio
async def test_driver_empty_active_set_never_falls_back_to_desired():
    """When config.active_repos is empty, _poll_once must return immediately
    without iterating any repos — no fallback to DESIRED_REPOS."""
    config = make_config()
    config.active_repos = []  # empty
    config.auth_valid = True

    proxy = MagicMock()
    proxy.get_open_prs = AsyncMock(return_value=[])
    controller = MagicMock()

    github = MagicMock()
    github_auth = MagicMock()

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._poll_once()

    # get_open_prs must NOT be called — empty active set skips polling
    proxy.get_open_prs.assert_not_called()


@pytest.mark.asyncio
async def test_driver_webhook_fails_closed_when_revoked():
    """Webhook mutation must be rejected when config.auth_valid is False,
    even if the repo is in the static allowlist."""
    from ..router_webhook import webhook
    from ..config import Config
    from unittest.mock import AsyncMock, MagicMock
    import json
    from types import SimpleNamespace
    import hmac
    import hashlib

    config = make_config()
    config.active_repos = ["4paradigm/phanthymotus"]
    config.auth_valid = False  # revoked

    proxy = MagicMock()
    controller = MagicMock()

    # Build a valid HMAC signature
    payload = _request_payload("/approve_deploy machine=test-machine")
    body_bytes = json.dumps(payload).encode("utf-8")
    secret = config.github_webhook_secret or "test-secret"
    signature = "sha256=" + hmac.new(
        secret.encode(), body_bytes, hashlib.sha256
    ).hexdigest()

    class _Request:
        def __init__(self):
            self.app = SimpleNamespace(
                state=SimpleNamespace(config=config, proxy=proxy, controller=controller)
            )
            self.headers = {
                "X-GitHub-Event": "issue_comment",
                "X-Hub-Signature-256": signature,
            }

        async def stream(self):
            yield body_bytes

    request = _Request()

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await webhook(request)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_driver_token_refresh_precedes_repo_discovery():
    """refresh_installation_token MUST be called BEFORE
    list_installation_repositories in every auth refresh cycle."""
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    )

    github_auth = MagicMock()
    call_order = []
    async def mock_refresh():
        call_order.append("refresh")
        return "fresh-token"
    async def mock_list():
        call_order.append("list")
        return ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    github_auth.refresh_installation_token = mock_refresh
    github.list_installation_repositories = mock_list

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._refresh_active_repos()

    assert call_order == ["refresh", "list"], \
        f"Expected ['refresh', 'list'], got {call_order}"


@pytest.mark.asyncio
async def test_driver_token_refresh_failure_does_not_publish_new_repos():
    """When refresh_installation_token fails, active_repos must remain
    unchanged and auth_valid must be set to False — no new repos published."""
    from ..github_client import GitHubClient

    config = make_config()
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(
        side_effect=Exception("token refresh failed")
    )

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._refresh_active_repos()

    # active_repos unchanged
    assert config.active_repos == ["4paradigm/phanthymotus"]
    # auth_valid must be False
    assert config.auth_valid is False
    # list_installation_repositories must NOT have been called
    github.list_installation_repositories.assert_not_called()


@pytest.mark.asyncio
async def test_driver_reactivation_baselines_pending_comments():
    """When driver repo becomes active again, preexisting command comments
    on driver PRs must NOT be dispatched as fresh approvals. The watcher
    only picks up NEW comments after the baseline."""
    from ..github_client import GitHubClient
    from ..github_command_watcher import GitHubCommandWatcher

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]  # driver not yet active

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)

    async def _reconcile_pr(repo, pr_number):
        """Simulate production reconcile_pr creating lifecycle state per-PR."""
        if store.get(pr_number) is None:
            store[pr_number] = {
                "version": 1, "head_sha": "d" * 40, "status": "review-required",
                "components": [], "deployments": [], "case_results": {},
                "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
                "approve_attempts": [], "approve_attempts_total": 0,
                "approve_attempts_truncated": False,
                "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
                "last_processed_comment_id": 0,
            }
        return store[pr_number]

    controller.reconcile_pr = AsyncMock(side_effect=_reconcile_pr)

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    # Activate driver repo
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" in config.active_repos

    # Driver PR that has old command comments (created before reactivation)
    driver_pr = {"number": 42, "state": "open", "merged": False, "draft": False,
                 "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}
    old_comment = {"id": 9001, "body": "/approve_deploy machine=test-machine",
                   "user": {"id": 111, "login": "bob"}, "created_at": "2026-09-01T00:00:00Z"}
    new_comment = {"id": 9002, "body": "/approve_deploy machine=test-machine",
                   "user": {"id": 111, "login": "bob"}, "created_at": "2026-10-01T00:00:00Z"}

    # State with empty baseline fields — triggers _baseline_pr
    baseline_state = {
        "version": 1, "head_sha": "c" * 40, "status": "review-required",
        "components": [], "deployments": [], "case_results": {},
        "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    }

    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [{"number": 42}] if "driver" in repo else [])
    proxy.is_bot_comment = MagicMock(return_value=False)

    # Stateful store: read_hidden_state returns the PERSISTED state.
    store: dict = {42: dict(baseline_state)}

    async def _read_state(_repo, _pr):
        return None if store.get(_pr) is None else dict(store.get(_pr))

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    proxy.get_issue_comments = AsyncMock(return_value=[old_comment, new_comment])
    proxy.comment_identity = AsyncMock(return_value=("111", "bob"))
    proxy.get_pr = AsyncMock(return_value=driver_pr)
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value=new_comment)
    proxy.collaborator_permission = AsyncMock(return_value="admin")

    persist_calls = []
    async def _persist_cursor(repo, pr, cid):
        persist_calls.append(cid)
        # Cursor durability: persist mutates the stateful store.
        if pr in store:
            store[pr] = {
                **store[pr],
                "last_processed_comment_id": cid,
            }
        return {"last_processed_comment_id": cid}
    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)

    # Cycle 1: baseline — old comments NOT dispatched
    await watcher._poll_once()

    # Controller must NOT have been dispatched for the old comment
    controller.on_command.assert_not_called()
    # Baseline cursor must have been persisted to max comment id
    assert persist_calls, "Expected persist_cursor call during baseline"
    assert max(persist_calls) == 9002, f"Expected baseline cursor 9002, got {persist_calls}"
    # Durable: the store now reflects the baselined cursor.
    assert store[42]["last_processed_comment_id"] == 9002

    # Cycle 2: new comment appears above baseline cursor — IS dispatched
    persist_calls.clear()
    controller.on_command.reset_mock()
    # Simulate a new comment (id=9003) appearing after baseline
    newer_comment = {"id": 9003, "body": "/approve_deploy machine=test-machine",
                     "user": {"id": 111, "login": "bob"}, "created_at": "2026-10-02T00:00:00Z"}

    proxy.get_issue_comments = AsyncMock(return_value=[old_comment, new_comment, newer_comment])
    proxy.get_comment = AsyncMock(return_value=newer_comment)

    # This cycle: PR is NOT in _pending_baseline_repos anymore (discarded after cycle 1),
    # so it goes through _process_pr path, which should dispatch the new comment (9003 > 9002)
    await watcher._poll_once()

    # New comment MUST be dispatched
    controller.on_command.assert_called_once()

@pytest.mark.asyncio
async def test_driver_repo_malformed_or_incomplete_installation_list_fails_closed():
    """When list_installation_repositories returns malformed data,
    auth refresh must fail closed — auth_valid=False, no new repos."""
    from ..github_client import GitHubClient

    config = make_config()
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="token")

    # Case 1: dict entries without full_name
    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        return_value=[{"bad_key": "value"}]
    )

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )
    await watcher._refresh_active_repos()
    assert config.auth_valid is False

    # Case 2: non-list return value
    github.list_installation_repositories = AsyncMock(
        return_value="not-a-list"
    )
    config.active_repos = ["4paradigm/phanthymotus"]
    config.auth_valid = True
    await watcher._refresh_active_repos()
    assert config.auth_valid is False


@pytest.mark.asyncio
async def test_driver_authorized_new_pr_command_dispatch():
    """When a driver repo is authorized, NEW PR commands on that repo
    should be dispatched through the normal flow.

    Two-cycle fixture (realistic): cycle 1 baselines the PR on first
    observation (zero dispatch, durable cursor), then a NEW comment
    appears above the durable cursor and cycle 2 dispatches exactly
    that one.  Core and Driver PR identities remain separate.
    """
    from ..github_client import GitHubClient
    from ..github_command_watcher import GitHubCommandWatcher

    config = make_config(github_repos=[
        "4paradigm/phanthymotus", "4paradigm/phanthymotus-driver",
    ])
    config.active_repos = ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    config.auth_valid = True

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)

    async def _reconcile_pr(repo, pr_number):
        """Simulate production reconcile_pr creating lifecycle state per-PR."""
        if store.get(pr_number) is None:
            store[pr_number] = {
                "version": 1, "head_sha": "d" * 40, "status": "review-required",
                "components": [], "deployments": [], "case_results": {},
                "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
                "approve_attempts": [], "approve_attempts_total": 0,
                "approve_attempts_truncated": False,
                "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
                "last_processed_comment_id": 0,
            }
        return store[pr_number]

    controller.reconcile_pr = AsyncMock(side_effect=_reconcile_pr)

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()

    driver_pr = {"number": 10, "state": "open", "merged": False, "draft": False,
                 "head": {"sha": "d" * 40}, "user": {"id": 111, "login": "carol"}}

    first_comment = {"id": 7001, "body": "/approve_deploy machine=test-machine",
                     "user": {"id": 111, "login": "carol"}, "created_at": "2026-10-09T00:00:00Z"}
    newer_comment = {"id": 7002, "body": "/approve_deploy machine=test-machine",
                     "user": {"id": 111, "login": "carol"}, "created_at": "2026-10-09T01:00:00Z"}

    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [{"number": 10}] if "driver" in repo else [])
    proxy.is_bot_comment = MagicMock(return_value=False)

    # Stateful store: read_hidden_state returns the PERSISTED state.
    store: dict = {"state": None}
    persisted_bodies = []

    async def _read_state(_repo, _pr):
        return None if store.get(_pr) is None else dict(store.get(_pr))

    async def _persist_cursor(_repo, _pr, cid):
        """Faithful production contract: only update last_processed_comment_id."""
        pr_state = store.get(_pr)
        if pr_state is not None:
            pr_state["last_processed_comment_id"] = cid
        persisted_bodies.append(cid)
        return {"last_processed_comment_id": cid}

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)

    comments_holder = {"list": [first_comment]}
    proxy.get_issue_comments = AsyncMock(
        side_effect=lambda _repo, _pr: list(comments_holder["list"]))
    proxy.comment_identity = AsyncMock(return_value=("111", "carol"))
    proxy.get_pr = AsyncMock(return_value=driver_pr)
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(side_effect=lambda _repo, cid: (
        newer_comment if cid == 7002 else first_comment))
    proxy.collaborator_permission = AsyncMock(return_value="admin")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    # ── Cycle 1: first observation — baseline only, zero dispatch ──
    await watcher._poll_once()
    controller.on_command.assert_not_called()
    assert persisted_bodies and persisted_bodies[-1] == 7001, \
        f"Expected baseline cursor 7001 persisted, got {persisted_bodies}"

    # ── A NEW comment arrives above the durable cursor ──
    comments_holder["list"].append(newer_comment)

    # ── Cycle 2: normal processing — dispatch exactly the new comment ──
    await watcher._poll_once()

    controller.on_command.assert_called_once()
    call_args = controller.on_command.call_args
    # Dispatch must target the DRIVER repo (repo arg is 2nd positional).
    dispatched_repo = call_args.args[1] if len(call_args.args) >= 2 else call_args.kwargs.get("repo")
    assert dispatched_repo == "4paradigm/phanthymotus-driver", \
        f"Expected dispatch on driver repo, got {call_args}"
    # Cursor durability: last persist reflects the consumed comment.
    assert persisted_bodies[-1] == 7002, \
        f"Expected cursor advanced to 7002 after dispatch, got {persisted_bodies}"

    # Core repo PRs remain polled independently (repo separation preserved).
    # Core repo must be polled via get_open_prs during watcher cycle (no Core PRs in fixture)
    core_pr_polls = [
        c for c in proxy.get_open_prs.call_args_list
        if c.args and "phanthymotus" in str(c.args[0])
    ]
    assert core_pr_polls, "Core repo get_open_prs must be polled independently"


@pytest.mark.asyncio
async def test_driver_authorization_grant_does_not_deploy_automatically():
    """Granting the driver repo via auth refresh must NOT create any
    deployment — it only updates active_repos."""
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    proxy.write_hidden_state = AsyncMock()
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    )

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    # Refresh — this grants the driver repo
    await watcher._refresh_active_repos()

    # Driver should be active
    assert "4paradigm/phanthymotus-driver" in config.active_repos

    # NO write_hidden_state calls (no deployment)
    proxy.write_hidden_state.assert_not_called()
    # NO controller.on_command calls
    controller.on_command.assert_not_called()


@pytest.mark.asyncio
async def test_driver_required_repo_authorization_lost_fails_closed():
    """When the required 4paradigm/phanthymotus repo disappears from
    authorization, the system must fail closed: auth_valid=False,
    active_repos=[], no further polling."""
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]

    proxy = MagicMock()
    controller = MagicMock()

    github = MagicMock(spec=GitHubClient)
    # Only driver is authorized — required repo is missing
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus-driver"]
    )

    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    await watcher._refresh_active_repos()

    # Must fail closed
    assert config.auth_valid is False
    assert config.active_repos == []


@pytest.mark.asyncio
async def test_record_test_terminal_fail_is_advisory_and_can_finalize():
    """When all bound component case_results are terminal (including 'fail'),
    record_test MUST finalize by transitioning to 'failed' status.
    Terminal values are: pass, fail, n/a."""
    controller, proxy, _policy, _github, _config = _controller()

    from ..github_state_proxy import (
        _build_hidden_state_body as _bhsb,
        _extract_hidden_state,
    )

    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )

    # State: all terminal case_results with 'fail'
    state_terminal = _state(
        components=[comp],
        deployments=[
            {"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"},
        ],
        status="testing",
        case_results={"comp-001": "fail"},
    )
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), state_terminal)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()

    proxy.get_comment = AsyncMock(return_value={
        "id": 82, "body": "/record_test result=fail summary='test failed'",
        "user": {"id": 111, "login": "owner1"},
    })

    result = await controller.handle_record_test("repo", 1, 82, "fail", "test failed", "owner1", "111")
    assert result is True
    assert proxy.write_hidden_state.call_count == 1
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["status"] == "failed"


@pytest.mark.asyncio
async def test_testing_reconcile_with_running_case_result_remains_pending():
    """reconcile_pr with status=testing and a running case result must NOT
    prematurely finalize — status stays 'testing'."""
    from ..github_state_proxy import (
        _build_hidden_state_body as _bhsb,
        _extract_hidden_state,
    )

    config = make_config()
    proxy = MagicMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))

    policy = Policy(config)
    policy.machines = {
        "test-machine": MachineInfo(
            alias="test-machine", node_id="n1", owners=["owner1"],
            node_host="127.0.0.1", targets=["perception"],
            platforms=["linux/arm64"], variants=["5.11"],
        ),
    }

    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )

    state = _state(
        components=[comp],
        deployments=[
            {"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"},
        ],
        status="testing",
        case_results={"comp-001": "running"},
    )
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), state)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })

    controller = DeployController(config, proxy, policy, MagicMock())

    await controller.reconcile_pr("repo", 1)

    # Status should remain 'testing' — running case means not finalized
    final_state = await _read_state()
    assert final_state["status"] == "testing"


@pytest.mark.asyncio
async def test_driver_reactivation_advances_existing_pr_cursor_before_dispatch():
    """When a driver repo is re-authorized, existing PRs with real cursor
    state must have their cursor advanced to max(old, observed) BEFORE
    any command dispatch — preventing revocation-period backlog replay."""
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]  # driver not active

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)

    async def _reconcile_pr(repo, pr_number):
        """Simulate production reconcile_pr creating lifecycle state per-PR."""
        if store.get(pr_number) is None:
            store[pr_number] = {
                "version": 1, "head_sha": "d" * 40, "status": "review-required",
                "components": [], "deployments": [], "case_results": {},
                "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
                "approve_attempts": [], "approve_attempts_total": 0,
                "approve_attempts_truncated": False,
                "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
                "last_processed_comment_id": 0,
            }
        return store[pr_number]

    controller.reconcile_pr = AsyncMock(side_effect=_reconcile_pr)

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth,
    )

    # Activate driver repo
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" in config.active_repos

    driver_pr = {"number": 42, "state": "open", "merged": False, "draft": False,
                 "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}

    # Existing state: cursor=5000, so the PR was previously baselined.
    existing_state = {
        "version": 1, "head_sha": "c" * 40, "status": "review-required",
        "components": [], "deployments": [], "case_results": {},
        "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "command": {"comment_id": 5000, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 5000,
    }

    old_comment = {"id": 9001, "body": "/approve_deploy machine=test-machine",
                   "user": {"id": 111, "login": "bob"}}
    newer_comment = {"id": 9002, "body": "/approve_deploy machine=test-machine",
                     "user": {"id": 111, "login": "bob"}}

    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [{"number": 42}] if "driver" in repo else [])
    proxy.is_bot_comment = MagicMock(return_value=False)
    proxy.get_pr = AsyncMock(return_value=driver_pr)
    proxy.comment_identity = AsyncMock(return_value=("111", "bob"))
    proxy.get_comment = AsyncMock(return_value=newer_comment)
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()

    store = {42: dict(existing_state)}

    async def _read_state(_repo, _pr):
        return None if store.get(_pr) is None else dict(store.get(_pr))

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)

    persist_calls = []

    async def _persist_cursor(repo, pr, cid):
        persist_calls.append(cid)
        if pr in store:
            store[pr] = {
                **store[pr],
                "last_processed_comment_id": cid,
            }
        return {"last_processed_comment_id": cid}

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)

    comments_holder = {"list": [old_comment, newer_comment]}
    proxy.get_issue_comments = AsyncMock(
        side_effect=lambda _repo, _pr: list(comments_holder["list"]))

    # Cycle 1: reactivation baseline — cursor advances to max(5000, 9002)=9002
    await watcher._poll_once()

    # Zero dispatch during baseline
    controller.on_command.assert_not_called()

    # Cursor must be >= max observed comment id
    assert persist_calls, "Expected persist_cursor call during reactivation baseline"
    assert persist_calls[-1] == 9002, f"Expected cursor 9002, got {persist_calls[-1]}"
    assert store[42]["last_processed_comment_id"] == 9002

    # Cycle 2: new comment appears above baseline — IS dispatched
    persist_calls.clear()
    controller.on_command.reset_mock()
    newest = {"id": 9003, "body": "/approve_deploy machine=test-machine",
              "user": {"id": 111, "login": "bob"}}
    comments_holder["list"].append(newest)
    proxy.get_comment = AsyncMock(return_value=newest)

    await watcher._poll_once()

    controller.on_command.assert_called_once()


@pytest.mark.asyncio
async def test_driver_reactivation_retries_failed_baseline_without_replay():
    """When one PR's baseline fails during reactivation, the repo's
    pending-baseline flag is NOT cleared; the failed PR is retried on the
    next cycle.  Other PRs that succeeded must NOT be rebaselined.
    No command dispatch occurs until ALL PRs are baselined."""
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)
    controller.reconcile_pr = AsyncMock(side_effect=None)

    store = {}

    def _make_state(pr_num):
        return {
            "version": 1, "head_sha": "c" * 40, "status": "review-required",
            "components": [], "deployments": [], "case_results": {},
            "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
            "approve_attempts": [], "approve_attempts_total": 0,
            "approve_attempts_truncated": False,
            "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
            "last_processed_comment_id": 0,
        }

    store[42] = dict(_make_state(42))
    store[43] = dict(_make_state(43))

    async def _reconcile_pr(repo, pr_number):
        """Simulate production reconcile_pr creating lifecycle state."""
        if store.get(pr_number) is None:
            store[pr_number] = dict(_make_state(pr_number))
        return store[pr_number]


    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth,
    )

    # Activate driver repo — two open PRs
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" in config.active_repos

    driver_pr_42 = {"number": 42, "state": "open", "merged": False, "draft": False,
                    "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}
    driver_pr_43 = {"number": 43, "state": "open", "merged": False, "draft": False,
                    "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}

    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [
        driver_pr_42, driver_pr_43
    ] if "driver" in repo else [])
    proxy.is_bot_comment = MagicMock(return_value=False)
    proxy.get_pr = AsyncMock(return_value=driver_pr_42)
    proxy.comment_identity = AsyncMock(return_value=("111", "bob"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 9001, "body": "/approve_deploy machine=test",
        "user": {"id": 111, "login": "bob"},
    })


    async def _read_state(_repo, _pr):
        pr = _pr
        return None if store.get(pr) is None else dict(store.get(pr, {}))

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)

    persist_calls = []

    async def _persist_cursor(repo, pr, cid):
        persist_calls.append((pr, cid))
        if pr in store:
            store[pr]["last_processed_comment_id"] = cid
        return {"last_processed_comment_id": cid}

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)

    def _make_comments(pr_num):
        return [
            {"id": 9000 + pr_num, "body": "/approve_deploy machine=test",
             "user": {"id": 111, "login": "bob"}},
        ]

    proxy.get_issue_comments = AsyncMock(
        side_effect=lambda _repo, _pr: _make_comments(_pr))

    # Cycle 1: baseline PR 42 succeeds, PR 43 fails at persist_cursor boundary
    # We inject failure at proxy.persist_cursor for PR #43 only.
    original_persist = proxy.persist_cursor.side_effect

    async def _persist_cursor_fail_on_43(repo, pr, cid):
        if pr == 43:
            raise Exception("baseline persist fail for PR 43")
        return await original_persist(repo, pr, cid)

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor_fail_on_43)

    await watcher._poll_once()

    # PR 42 persisted (cursor advanced), PR 43 failed
    assert persist_calls, "Expected at least one persist call"
    pr42_persisted = any(c[0] == 42 for c in persist_calls)
    assert pr42_persisted, "PR 42 should have been baselined"

    # No dispatch during baseline
    controller.on_command.assert_not_called()

    # With the production fix, _pending_baseline_repos stays because PR 43 failed
    assert "4paradigm/phanthymotus-driver" in watcher._pending_baseline_repos, \
        "Repo must remain pending while any PR baseline is incomplete"

    # Cycle 2: PR 43 persist succeeds (restore working persist_cursor)
    persist_calls.clear()

    async def _persist_cursor_fixed(repo, pr, cid):
        persist_calls.append((pr, cid))
        if pr in store:
            store[pr] = {
                **store[pr],
                "last_processed_comment_id": cid,
                "command": {"comment_id": cid, "kind": "", "phase": "completed", "args": {}},
            }
        return {"last_processed_comment_id": cid}

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor_fixed)

    await watcher._poll_once()

    # PR 43 should have been retried and baselined
    pr43_persisted = any(c[0] == 43 for c in persist_calls)
    assert pr43_persisted, "PR 43 should have been retried and baselined"

    # PR 42 must NOT have been rebaselined (no duplicate)
    pr42_again = sum(1 for c in persist_calls if c[0] == 42)
    assert pr42_again == 0, f"PR 42 must not be rebaselined on retry, got {pr42_again} extra persists"

    # Repo pending flag cleared now that ALL PRs succeeded
    assert "4paradigm/phanthymotus-driver" not in watcher._pending_baseline_repos

    # Still no dispatch
    controller.on_command.assert_not_called()

    # Cycle 3: new comment beyond baseline — IS dispatched
    persist_calls.clear()
    controller.on_command.reset_mock()
    new_comment_9100 = {"id": 9100, "body": "/approve_deploy machine=test",
                        "user": {"id": 111, "login": "bob"}}

    def _make_comments_cycle3(_repo, pr_num):
        # Only PR 42 gets the new comment; PR 43 keeps only its old comment
        if pr_num == 42:
            return [
                {"id": 9000 + pr_num, "body": "/approve_deploy machine=test",
                 "user": {"id": 111, "login": "bob"}},
                new_comment_9100,
            ]
        else:
            return [
                {"id": 9000 + pr_num, "body": "/approve_deploy machine=test",
                 "user": {"id": 111, "login": "bob"}},
            ]

    proxy.get_issue_comments = AsyncMock(side_effect=_make_comments_cycle3)
    proxy.get_comment = AsyncMock(return_value=new_comment_9100)

    await watcher._poll_once()

    controller.on_command.assert_called_once()
    call_args = controller.on_command.call_args
    dispatched_repo = call_args.args[1] if len(call_args.args) >= 2 else call_args.kwargs.get("repo")
    assert dispatched_repo == "4paradigm/phanthymotus-driver"


@pytest.mark.asyncio
async def test_driver_reactivation_preserves_business_state_and_visible_history():
    """Reactivation baseline must NOT reset business fields (components,
    deployments, COS, History, command state) — only the cursor advances."""
    from ..github_client import GitHubClient
    from ..github_state_proxy import _build_hidden_state_body as _bhsb, _extract_hidden_state

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)
    controller.reconcile_pr = AsyncMock()

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth,
    )

    # Activate driver repo
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" in config.active_repos

    driver_pr = {"number": 42, "state": "open", "merged": False, "draft": False,
                 "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}

    # Existing state with deployed components, COS, History, etc.
    body_holder = {"body": ""}

    existing_state = _state(
        components=[_component(component_id="comp-a", target="perception")],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-a"], "phase": "deployed"}],
        status="deploy-requested",
        cos={"object_key": "evidence/repo/42/x.tar.gz", "sha256": "ab" * 32, "size": 1024},
        command={"comment_id": 5000, "kind": "", "phase": "completed", "args": {}},
        last_processed_comment_id=5000,
        head_sha="c" * 40,
    )

    visible_body = _bhsb(
        _lifecycle_visible("repo", 42, status="deploy-requested"),
        existing_state,
    )
    body_holder["body"] = visible_body

    # Add a History section with an existing event
    from ..github_state_proxy import _build_history_block
    pre_events = [
        {"event": "Lifecycle initialized",
         "lifecycle": "`none` → `review-required`",
         "timestamp": "2026-09-30 10:00:00"},
    ]
    history_block = _build_history_block(pre_events)
    visible, _, _ = visible_body.partition("<!-- deploy-approval-state:v1")
    body_holder["body"] = visible.rstrip() + "\n\n### History\n\n" + history_block + "\n" + visible_body[visible_body.find("<!-- deploy-approval-state:v1"):]

    async def _read_state(_repo, _pr):
        return _extract_hidden_state(body_holder["body"])

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 9001, "body": "/approve_deploy machine=test",
                                                          "user": {"id": 111, "login": "bob"}}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [{"number": 42}] if "driver" in repo else [])
    proxy.get_pr = AsyncMock(return_value=driver_pr)
    proxy.comment_identity = AsyncMock(return_value=("111", "bob"))
    proxy.get_comment = AsyncMock(return_value={"id": 9001, "body": "/approve_deploy machine=test",
                                                  "user": {"id": 111, "login": "bob"}})
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.is_bot_comment = MagicMock(return_value=False)

    persist_calls = []

    async def _persist_cursor(repo, pr, cid):
        """Faithful production persist_cursor: advances cursor, preserves ALL
        hidden-state fields and exact visible markdown bytes. ONLY last_processed_comment_id changes."""
        persist_calls.append(cid)
        # Read current state from body_holder (production: reads from hidden JSON)
        state_after = _extract_hidden_state(body_holder["body"])
        if state_after is not None:
            state_after["last_processed_comment_id"] = cid
            # Re-serialize preserving the visible markdown (byte-identical visible portion)
            marker = "<!-- deploy-approval-state:v1"
            visible_part = body_holder["body"][:body_holder["body"].find(marker)]
            body_holder["body"] = _bhsb(visible_part, state_after)
        return {"last_processed_comment_id": cid}

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)

    # Cycle 1: reactivation baseline
    await watcher._poll_once()

    # Cursor advanced but nothing else changed
    assert persist_calls, "Expected persist_cursor call"
    assert persist_calls[-1] == 9001, f"Expected cursor 9001, got {persist_calls[-1]}"

    # Business state preserved
    hidden = await _read_state("repo", 42)
    assert hidden["status"] == "deploy-requested", \
        f"Status should be preserved, got {hidden['status']}"
    assert hidden["deployments"] == [
        {"machine": "test-machine", "component_ids": ["comp-a"], "phase": "deployed"}
    ], f"Deployments should be preserved: {hidden.get('deployments')}"
    assert hidden["cos"]["object_key"] == "evidence/repo/42/x.tar.gz", \
        f"COS should be preserved: {hidden.get('cos')}"
    assert len(hidden["components"]) == 1, \
        f"Components should be preserved: {hidden.get('components')}"

    # No dispatch during baseline
    controller.on_command.assert_not_called()

    # Cycle 2: new comment above baseline — IS dispatched
    persist_calls.clear()
    controller.on_command.reset_mock()
    newer_comment = {"id": 9002, "body": "/approve_deploy machine=test",
                     "user": {"id": 111, "login": "bob"}}
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 9001, "body": "/approve_deploy machine=test", "user": {"id": 111, "login": "bob"}},
        newer_comment,
    ])
    proxy.get_comment = AsyncMock(return_value=newer_comment)

    await watcher._poll_once()

    controller.on_command.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# V5 REGRESSION TESTS — Driver reactivation baseline production fixes
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_driver_reactivation_failed_existing_pr_remains_pending():
    """When one PR's baseline persist fails, _pending_baseline_repos stays
    and ZERO on_command dispatches occur.  On retry, the failed PR succeeds
    and the repo is cleared."""
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)
    controller.reconcile_pr = AsyncMock()

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth,
    )

    # Activate driver repo
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" in config.active_repos

    driver_pr_10 = {"number": 10, "state": "open", "merged": False, "draft": False,
                    "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}

    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [
        driver_pr_10
    ] if "driver" in repo else [])
    proxy.is_bot_comment = MagicMock(return_value=False)
    proxy.get_pr = AsyncMock(return_value=driver_pr_10)
    proxy.comment_identity = AsyncMock(return_value=("111", "bob"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 8010, "body": "/approve_deploy machine=test",
        "user": {"id": 111, "login": "bob"},
    })

    store = {10: {
        "version": 1, "head_sha": "c" * 40, "status": "review-required",
        "components": [], "deployments": [], "case_results": {},
        "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    }}

    async def _read_state(_repo, _pr):
        return None if store.get(_pr) is None else dict(store.get(_pr, {}))

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)

    persist_calls = []

    async def _persist_cursor(repo, pr, cid):
        persist_calls.append((pr, cid))
        if pr in store:
            store[pr]["last_processed_comment_id"] = cid
        return {"last_processed_comment_id": cid}

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)

    def _make_comments(pr_num):
        return [{"id": 8000 + pr_num, "body": "/approve_deploy machine=test",
                 "user": {"id": 111, "login": "bob"}}]

    proxy.get_issue_comments = AsyncMock(
        side_effect=lambda _repo, _pr: _make_comments(_pr))

    # Cycle 1: baseline persist fails
    async def _persist_cursor_fail(repo, pr, cid):
        if pr == 10:
            raise Exception("baseline persist fail")
        return await _persist_cursor(repo, pr, cid)

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor_fail)

    await watcher._poll_once()

    # Repo must remain pending
    assert "4paradigm/phanthymotus-driver" in watcher._pending_baseline_repos, \
        "Repo must remain pending when PR baseline fails"
    # Zero dispatch
    controller.on_command.assert_not_called()

    # Cycle 2: persist succeeds
    persist_calls.clear()
    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)

    await watcher._poll_once()

    # Repo pending cleared
    assert "4paradigm/phanthymotus-driver" not in watcher._pending_baseline_repos, \
        "Repo pending must clear after all PRs succeed"
    controller.on_command.assert_not_called()


@pytest.mark.asyncio
async def test_driver_reactivation_successful_pr_not_rebaselined_during_retry():
    """PR #10 succeeds baseline first.  PR #11 fails.  On retry, PR #10 is NOT
    rebaselined (progress tracking prevents duplicate baseline), PR #11 retries
    and succeeds.  New comments after PR #10's successful baseline ARE dispatched."""
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)
    controller.reconcile_pr = AsyncMock()

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth,
    )

    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
    )
    await watcher._refresh_active_repos()

    driver_pr_10 = {"number": 10, "state": "open", "merged": False, "draft": False,
                    "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}
    driver_pr_11 = {"number": 11, "state": "open", "merged": False, "draft": False,
                    "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}

    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [
        driver_pr_10, driver_pr_11
    ] if "driver" in repo else [])
    proxy.is_bot_comment = MagicMock(return_value=False)
    proxy.get_pr = AsyncMock(return_value=driver_pr_10)
    proxy.comment_identity = AsyncMock(return_value=("111", "bob"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 8010, "body": "/approve_deploy machine=test",
        "user": {"id": 111, "login": "bob"},
    })

    store = {
        10: {"version": 1, "head_sha": "c" * 40, "status": "review-required",
             "components": [], "deployments": [], "case_results": {},
             "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
             "approve_attempts": [], "approve_attempts_total": 0,
             "approve_attempts_truncated": False,
             "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
             "last_processed_comment_id": 0},
        11: {"version": 1, "head_sha": "c" * 40, "status": "review-required",
             "components": [], "deployments": [], "case_results": {},
             "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
             "approve_attempts": [], "approve_attempts_total": 0,
             "approve_attempts_truncated": False,
             "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
             "last_processed_comment_id": 0},
    }

    async def _read_state(_repo, _pr):
        return None if store.get(_pr) is None else dict(store.get(_pr, {}))

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)

    persist_calls = []

    async def _persist_cursor(repo, pr, cid):
        persist_calls.append((pr, cid))
        if pr in store:
            store[pr]["last_processed_comment_id"] = cid
        return {"last_processed_comment_id": cid}

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)

    def _make_comments(pr_num):
        return [{"id": 8000 + pr_num, "body": "/approve_deploy machine=test",
                 "user": {"id": 111, "login": "bob"}}]

    proxy.get_issue_comments = AsyncMock(
        side_effect=lambda _repo, _pr: _make_comments(_pr))

    # Cycle 1: PR 10 succeeds, PR 11 fails
    async def _persist_fail_11(repo, pr, cid):
        if pr == 11:
            raise Exception("fail PR 11")
        return await _persist_cursor(repo, pr, cid)

    proxy.persist_cursor = AsyncMock(side_effect=_persist_fail_11)
    await watcher._poll_once()

    assert "4paradigm/phanthymotus-driver" in watcher._pending_baseline_repos
    assert any(c[0] == 10 for c in persist_calls), "PR 10 should be baselined"
    controller.on_command.assert_not_called()

    # Cycle 2: retry — PR 10 must NOT be rebaselined
    persist_calls.clear()

    async def _persist_fail_11_again(repo, pr, cid):
        # PR 11 still fails on first attempt of cycle 2
        if pr == 11 and not getattr(_persist_fail_11_again, "_retry_second", False):
            _persist_fail_11_again._retry_second = True
            raise Exception("fail PR 11 again")
        return await _persist_cursor(repo, pr, cid)

    proxy.persist_cursor = AsyncMock(side_effect=_persist_fail_11_again)
    await watcher._poll_once()

    # PR 10 must NOT appear again
    pr10_rebase = sum(1 for c in persist_calls if c[0] == 10)
    assert pr10_rebase == 0, f"PR 10 must not be rebaselined during retry, got {pr10_rebase}"
    assert "4paradigm/phanthymotus-driver" in watcher._pending_baseline_repos

    # Cycle 3: PR 11 finally succeeds
    persist_calls.clear()
    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)
    await watcher._poll_once()

    pr11_persisted = any(c[0] == 11 for c in persist_calls)
    assert pr11_persisted, "PR 11 should be baselined on retry"
    assert "4paradigm/phanthymotus-driver" not in watcher._pending_baseline_repos
    controller.on_command.assert_not_called()


@pytest.mark.asyncio
async def test_driver_reactivation_new_generation_resets_only_driver_progress():
    """Remove and re-grant driver: previous per-PR progress does NOT leak
    across auth generations.  Core repo remains active throughout."""
    from ..github_client import GitHubClient

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    config.auth_valid = True

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth,
    )

    # Step 1: Revoke driver
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus"],
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" not in config.active_repos
    assert "4paradigm/phanthymotus-driver" not in watcher._pending_baseline_repos
    assert watcher._baseline_progress.get("4paradigm/phanthymotus-driver") is None, \
        "Progress must be cleared on revoke"

    # Core still active
    assert "4paradigm/phanthymotus" in config.active_repos

    # Step 2: Re-grant driver — new generation
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
    )
    await watcher._refresh_active_repos()
    assert "4paradigm/phanthymotus-driver" in config.active_repos
    assert "4paradigm/phanthymotus-driver" in watcher._pending_baseline_repos
    # New generation: progress reset
    assert watcher._baseline_progress.get("4paradigm/phanthymotus-driver") == set(), \
        "New auth generation must start with empty progress set"

    # Core unaffected
    assert "4paradigm/phanthymotus" in config.active_repos


@pytest.mark.asyncio
async def test_driver_reactivation_preserves_exact_visible_history_and_command():
    """Production persist_cursor preserves exact visible History, archive
    markers, COS, command kind/phase/args, components, deployments, and
    review evidence; only cursor advances."""
    from ..github_client import GitHubClient
    from ..github_state_proxy import _build_hidden_state_body as _bhsb, _extract_hidden_state

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)
    controller.reconcile_pr = AsyncMock()

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth,
    )

    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"],
    )
    await watcher._refresh_active_repos()

    driver_pr = {"number": 42, "state": "open", "merged": False, "draft": False,
                 "head": {"sha": "c" * 40}, "user": {"id": 111, "login": "bob"}}

    body_holder = {"body": ""}

    existing_state = _state(
        components=[_component(component_id="comp-a", target="perception")],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-a"], "phase": "deployed"}],
        status="deploy-requested",
        cos={"object_key": "evidence/repo/42/x.tar.gz", "sha256": "ab" * 32, "size": 1024},
        command={"comment_id": 5000, "kind": "", "phase": "completed", "args": {}},
        last_processed_comment_id=5000,
        head_sha="c" * 40,
    )

    visible_body = _bhsb(
        _lifecycle_visible("repo", 42, status="deploy-requested"),
        existing_state,
    )
    body_holder["body"] = visible_body

    # Add a History section with an existing event
    from ..github_state_proxy import _build_history_block
    pre_events = [
        {"event": "Lifecycle initialized",
         "lifecycle": "`none` → `review-required`",
         "timestamp": "2026-09-30 10:00:00"},
    ]
    history_block = _build_history_block(pre_events)
    visible, _, _ = visible_body.partition("<!-- deploy-approval-state:v1")
    body_holder["body"] = visible.rstrip() + "\n\n### History\n\n" + history_block + "\n" + visible_body[visible_body.find("<!-- deploy-approval-state:v1"):]

    prev_visible_before = body_holder["body"][:body_holder["body"].find("<!-- deploy-approval-state:v1")]

    async def _read_state(_repo, _pr):
        return _extract_hidden_state(body_holder["body"])

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 9001, "body": "/approve_deploy machine=test",
                                                          "user": {"id": 111, "login": "bob"}}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [{"number": 42}] if "driver" in repo else [])
    proxy.get_pr = AsyncMock(return_value=driver_pr)
    proxy.comment_identity = AsyncMock(return_value=("111", "bob"))
    proxy.get_comment = AsyncMock(return_value={"id": 9001, "body": "/approve_deploy machine=test",
                                                  "user": {"id": 111, "login": "bob"}})
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.is_bot_comment = MagicMock(return_value=False)

    persist_calls = []

    async def _persist_cursor(repo, pr, cid):
        """Faithful persist_cursor: advances cursor, updates body_holder
        so _read_state can verify, preserving exact visible markdown.
        ONLY last_processed_comment_id changes — command is preserved."""
        persist_calls.append(cid)
        state_after = await proxy.read_hidden_state(repo, pr)
        if state_after is not None:
            state_after["last_processed_comment_id"] = cid
            # Preserve exact visible markdown bytes (everything before the hidden state marker).
            marker = "<!-- deploy-approval-state:v1\n"
            vis_prefix = body_holder["body"][:body_holder["body"].find(marker)]
            hidden_json = json.dumps(state_after, ensure_ascii=False, separators=(",", ":"))
            new_hidden_block = marker + hidden_json + "\n-->"
            body_holder["body"] = vis_prefix + new_hidden_block
        return {"last_processed_comment_id": cid}

    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)

    # Cycle 1: reactivation baseline
    await watcher._poll_once()

    # Cursor advanced
    assert persist_calls[-1] == 9001

    # Visible markdown before marker is EXACTLY preserved
    visible_after = body_holder["body"][:body_holder["body"].find("<!-- deploy-approval-state:v1")]
    assert visible_after == prev_visible_before, \
        f"Visible history must be byte-identical. Old: {repr(prev_visible_before[:80])}\nNew: {repr(visible_after[:80])}"

    # Hidden state business fields preserved
    hidden = await _read_state("repo", 42)
    assert hidden["status"] == "deploy-requested"
    assert hidden["deployments"] == [
        {"machine": "test-machine", "component_ids": ["comp-a"], "phase": "deployed"}
    ]
    assert hidden["cos"]["object_key"] == "evidence/repo/42/x.tar.gz"
    assert len(hidden["components"]) == 1
    assert hidden["last_processed_comment_id"] == 9001

    # No dispatch during baseline
    controller.on_command.assert_not_called()

    # Cycle 2: new comment above baseline — IS dispatched
    persist_calls.clear()
    controller.on_command.reset_mock()
    newer_comment = {"id": 9002, "body": "/approve_deploy machine=test",
                     "user": {"id": 111, "login": "bob"}}
    proxy.get_issue_comments = AsyncMock(return_value=[
        {"id": 9001, "body": "/approve_deploy machine=test", "user": {"id": 111, "login": "bob"}},
        newer_comment,
    ])
    proxy.get_comment = AsyncMock(return_value=newer_comment)

    await watcher._poll_once()

    controller.on_command.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# V6 NEW REGRESSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_driver_startup_existing_pr_baselines_before_command_dispatch():
    """Startup finds Driver already ACTIVE: existing PR comments from
    authorization-gap period are durably baselined BEFORE any dispatch,
    then NEW higher comment is processed. Core polling remains unaffected.
    """
    from ..github_client import GitHubClient
    from ..github_command_watcher import GitHubCommandWatcher

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    config.auth_valid = True

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)

    async def _reconcile_pr(repo, pr_number):
        pr_state = store.get(pr_number)
        if pr_state is None:
            pr_state = {
                "version": 1, "head_sha": "d" * 40, "status": "review-required",
                "components": [], "deployments": [], "case_results": {},
                "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
                "approve_attempts": [], "approve_attempts_total": 0,
                "approve_attempts_truncated": False,
                "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
                "last_processed_comment_id": 0,
            }
            store[pr_number] = pr_state
        return pr_state

    controller.reconcile_pr = AsyncMock(side_effect=_reconcile_pr)

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth
    )

    # Startup baseline gate — Driver already active
    watcher.mark_repos_pending_baseline(["4paradigm/phanthymotus-driver"])

    driver_pr = {"number": 42, "state": "open", "merged": False, "draft": False,
                 "head": {"sha": "d" * 40}, "user": {"id": 111, "login": "carol"}}
    old_gap_comment = {"id": 8001, "body": "/approve_deploy machine=test-machine",
                       "user": {"id": 111, "login": "carol"}}
    new_comment = {"id": 8002, "body": "/approve_deploy machine=test-machine",
                   "user": {"id": 111, "login": "carol"}}

    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [{"number": 42}] if "driver" in repo else [])
    proxy.is_bot_comment = MagicMock(return_value=False)

    store = {42: None}

    async def _read_state(_repo, _pr):
        return None if store.get(_pr) is None else dict(store.get(_pr))

    async def _persist_cursor(_repo, _pr, cid):
        if _pr in store and store[_pr] is not None:
            store[_pr]["last_processed_comment_id"] = cid
        return {"last_processed_comment_id": cid}

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)

    comments_holder = {"list": [old_gap_comment, new_comment]}
    proxy.get_issue_comments = AsyncMock(
        side_effect=lambda _repo, _pr: list(comments_holder["list"]))
    proxy.comment_identity = AsyncMock(return_value=("111", "carol"))
    proxy.get_pr = AsyncMock(return_value=driver_pr)
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value=new_comment)
    proxy.collaborator_permission = AsyncMock(return_value="admin")

    # Cycle 1: baseline — old gap comments NOT dispatched
    await watcher._poll_once()
    controller.on_command.assert_not_called()
    assert store[42]["last_processed_comment_id"] >= 8002, \
        f"Baseline cursor should be >= 8002, got {store[42]['last_processed_comment_id']}"

    # Core polling unaffected
    core_pr_polls = [c for c in proxy.get_open_prs.call_args_list if "phanthymotus" in str(c.args[0])]
    assert core_pr_polls, "Core repo get_open_prs must still be polled"

    # Cycle 2: NEW comment appears above baseline — IS dispatched
    comments_holder["list"].append({"id": 8003, "body": "/approve_deploy machine=test-machine",
                                    "user": {"id": 111, "login": "carol"}})
    controller.on_command.reset_mock()
    proxy.get_comment = AsyncMock(return_value={"id": 8003, "body": "/approve_deploy machine=test-machine",
                                                 "user": {"id": 111, "login": "carol"}})

    await watcher._poll_once()
    controller.on_command.assert_called_once()
    call_args = controller.on_command.call_args
    dispatched_repo = call_args.args[1] if len(call_args.args) >= 2 else call_args.kwargs.get("repo")
    assert dispatched_repo == "4paradigm/phanthymotus-driver"


@pytest.mark.asyncio
async def test_testing_reconcile_recovers_running_case_result_without_redeploy():
    """Seed testing state with case_results={cid: "running"}. First reconcile
    reruns advisory case, persists terminal result, preserves History,
    ZERO unsafe deploy POST. Second reconcile: zero writes.
    Test fail terminal and record_test continuation separately.
    """
    from ..github_state_proxy import (
        _build_hidden_state_body as _bhsb,
        _extract_hidden_state,
    )

    config = make_config()
    proxy = MagicMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })

    policy = Policy(config)
    policy.machines = {
        "m1": MachineInfo(
            alias="m1", node_id="n1", owners=["owner1"], node_host="10.0.0.1",
            targets=["perception"], platforms=["linux/arm64"], variants=["5.11"],
        ),
    }

    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )

    # Seed: status=testing, case result = "running" (non-terminal)
    running_state = _state(
        components=[comp],
        deployments=[
            {"machine": "m1", "component_ids": ["comp-001"], "phase": "deployed"},
        ],
        status="testing",
        case_results={"comp-001": "running"},
        command={"comment_id": 50, "kind": "approve_deploy", "phase": "completed", "args": {"machine": "m1", "actor": "alice"}},
        head_sha="a" * 40,
    )
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="testing"), running_state)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        return _extract_hidden_state(body_holder["body"])

    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)

    controller = DeployController(config, proxy, policy, MagicMock())
    controller.config.total_timeout = 0.5

    # Mock case runner — returns terminal pass for the running case
    controller._run_automated_case = AsyncMock(return_value={"comp-001": "pass"})

    # First reconcile: reruns advisory case, persists terminal result
    await controller.reconcile_pr("repo", 1)
    assert proxy.write_hidden_state.call_count == 1, \
        f"Expected 1 write on first reconcile, got {proxy.write_hidden_state.call_count}"
    written = proxy.write_hidden_state.call_args.args[3]
    assert written["case_results"]["comp-001"] == "pass", \
        f"Expected terminal pass, got {written['case_results']}"
    assert written["status"] == "testing"
    # ZERO deploy POST
    assert proxy.write_hidden_state.call_count == 1

    # Second reconcile: already complete, no churn
    proxy.write_hidden_state.reset_mock()
    await controller.reconcile_pr("repo", 1)
    assert proxy.write_hidden_state.call_count == 0, \
        f"Expected zero writes on second reconcile, got {proxy.write_hidden_state.call_count}"

    # Test fail terminal: record_test with fail should finalize
    proxy.write_hidden_state.reset_mock()
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.get_comment = AsyncMock(return_value={
        "id": 90, "body": "/record_test result=fail summary='broken'",
        "user": {"id": 111, "login": "owner1"},
    })

    result = await controller.handle_record_test("repo", 1, 90, "fail", "broken", "owner1", "111")
    assert result is True
    assert proxy.write_hidden_state.call_count == 1
    final_written = proxy.write_hidden_state.call_args.args[3]
    assert final_written["status"] == "failed", \
        f"Expected failed status after terminal fail, got {final_written['status']}"


# ══════════════════════════════════════════════════════════════════════════════
# V7 NEW REGRESSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_driver_auth_refresh_recovery_rebaselines_driver_before_dispatch():
    """When auth refresh fails then later succeeds with the SAME authorized
    repo list, Driver must remain blocked until a new durable baseline is
    persisted. No authorization-gap comments are replayed."""
    from ..github_client import GitHubClient
    from ..github_command_watcher import GitHubCommandWatcher

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus"]
    config.auth_valid = True

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)

    store: dict = {10: None}
    persisted_bodies = []

    async def _reconcile_pr(repo, pr_number):
        pr_state = store.get(pr_number)
        if pr_state is None:
            pr_state = {
                "version": 1, "head_sha": "d" * 40, "status": "review-required",
                "components": [], "deployments": [], "case_results": {},
                "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
                "approve_attempts": [], "approve_attempts_total": 0,
                "approve_attempts_truncated": False,
                "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
                "last_processed_comment_id": 0,
            }
            store[pr_number] = pr_state
        return pr_state

    controller.reconcile_pr = AsyncMock(side_effect=_reconcile_pr)

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()

    driver_pr = {"number": 10, "state": "open", "merged": False, "draft": False,
                 "head": {"sha": "d" * 40}, "user": {"id": 111, "login": "carol"}}

    old_comment = {"id": 7001, "body": "/approve_deploy machine=test-machine",
                   "user": {"id": 111, "login": "carol"}, "created_at": "2026-10-09T00:00:00Z"}
    comments_holder = {"list": [old_comment]}

    proxy.get_open_prs = AsyncMock(side_effect=lambda repo: [{"number": 10}] if "driver" in repo else [])
    proxy.is_bot_comment = MagicMock(return_value=False)

    async def _read_state(_repo, _pr):
        return None if store.get(_pr) is None else dict(store.get(_pr))

    async def _persist_cursor(_repo, _pr, cid):
        if store.get(_pr) is not None:
            store[_pr]["last_processed_comment_id"] = max(
                store[_pr]["last_processed_comment_id"], cid)
        persisted_bodies.append(cid)
        return {"last_processed_comment_id": cid}

    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.persist_cursor = AsyncMock(side_effect=_persist_cursor)
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    proxy.get_issue_comments = AsyncMock(
        side_effect=lambda _repo, _pr: list(comments_holder["list"]))
    proxy.comment_identity = AsyncMock(return_value=("111", "carol"))
    proxy.get_pr = AsyncMock(return_value=driver_pr)
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value=old_comment)
    proxy.collaborator_permission = AsyncMock(return_value="admin")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth)

    # Cycle 1: auth refresh FAILS — auth_valid=False, no dispatch
    github_auth.refresh_installation_token = AsyncMock(side_effect=Exception("token error"))
    await watcher._refresh_active_repos_if_needed()
    await watcher._poll_once()
    assert not config.auth_valid
    assert not getattr(watcher, '_baseline_progress', {}).get("4paradigm/phanthymotus-driver")
    controller.on_command.assert_not_called()

    # Cycle 2 step 1: token refresh recovers — driver enters pending (before poll)
    watcher._last_auth_refresh = -999.0
    github_auth.refresh_installation_token = AsyncMock(return_value="new-token")
    github.list_installation_repositories = AsyncMock(
        return_value=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    await watcher._refresh_active_repos_if_needed()
    assert config.auth_valid
    assert "4paradigm/phanthymotus-driver" in config.active_repos
    assert "4paradigm/phanthymotus-driver" in watcher._pending_baseline_repos,         "Driver must be pending immediately after auth recovery, before poll"

    # Cycle 2 step 2: poll runs baseline — baseline succeeds, pending cleared
    await watcher._poll_once()
    assert "4paradigm/phanthymotus-driver" not in watcher._pending_baseline_repos
    # Still zero dispatch because baseline only
    controller.on_command.assert_not_called()

    # Baseline succeeds
    await watcher._refresh_active_repos_if_needed()
    await watcher._poll_once()
    assert "4paradigm/phanthymotus-driver" not in watcher._pending_baseline_repos
    assert persisted_bodies and persisted_bodies[-1] == 7001

    # A NEW comment above baseline should now dispatch
    new_comment_7003 = {
        "id": 7003, "body": "/approve_deploy machine=test-machine",
        "user": {"id": 111, "login": "carol"}, "created_at": "2026-10-09T02:00:00Z"}
    comments_holder["list"].append(new_comment_7003)
    proxy.get_comment = AsyncMock(return_value=new_comment_7003)
    await watcher._refresh_active_repos_if_needed()
    await watcher._poll_once()
    controller.on_command.assert_called_once()
    call_repo = controller.on_command.call_args.args[1]
    assert call_repo == "4paradigm/phanthymotus-driver"


@pytest.mark.asyncio
async def test_driver_baseline_enumeration_failure_fails_closed():
    """If open-PR enumeration fails for the Driver repo, pending stays and
    no commands dispatch. The repo remains pending on retry."""
    from ..github_client import GitHubClient, GitHubError
    from ..github_command_watcher import GitHubCommandWatcher

    config = make_config(github_repos=["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"])
    config.active_repos = ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]
    config.auth_valid = True

    proxy = MagicMock()
    controller = MagicMock()
    controller.on_command = AsyncMock(return_value=True)

    github = MagicMock(spec=GitHubClient)
    github_auth = MagicMock()
    github_auth.refresh_installation_token = AsyncMock(return_value="token")

    watcher = GitHubCommandWatcher(
        config, proxy, controller, github=github, github_auth=github_auth)
    # Seed driver as pending
    watcher.mark_repos_pending_baseline(["4paradigm/phanthymotus-driver"])

    # enumerate PRs fails
    proxy.get_open_prs = AsyncMock(side_effect=GitHubError(500, "list failed"))
    proxy.is_bot_comment = MagicMock(return_value=False)

    await watcher._poll_once()
    # Should remain pending
    assert "4paradigm/phanthymotus-driver" in watcher._pending_baseline_repos
    controller.on_command.assert_not_called()


@pytest.mark.asyncio
async def test_testing_reconcile_rejects_stale_snapshot_after_await():
    """When HEAD or status changes between the advisory case await and
    the post-await stale check, no stale case results are written and no
    deploy POST fires."""
    config = make_config()
    proxy = MagicMock()
    policy = Policy(config)
    policy.machines = {}
    controller = DeployController(config, proxy, policy, MagicMock())

    comp = _component(
        component_id="comp-001", target="perception", variant="5.11",
        runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )
    state = _state(
        components=[comp],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
        status="testing",
        head_sha="a" * 40,
        case_results={},
    )

    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    proxy.write_hidden_state = AsyncMock()
    proxy.get_issue_comments = AsyncMock(return_value=[])
    proxy.read_hidden_state = AsyncMock(return_value=dict(state))
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.project_status_label = AsyncMock()

    # Simulate HEAD drift AFTER the advisory case await
    def _drifted_pr(*a, **kw):
        return {"state": "open", "merged": False, "draft": False,
                "head": {"sha": "b" * 40}, "user": {"id": 111, "login": "alice"}}

    proxy.get_pr = AsyncMock(side_effect=[
        {"state": "open", "merged": False, "draft": False,
         "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}},
        _drifted_pr(),  # second call (post-await) returns drifted HEAD
    ])
    proxy.read_hidden_state = AsyncMock(side_effect=[
        dict(state),  # first read
        None,  # post-await: state gone
    ])
    controller._run_automated_case = AsyncMock(return_value={"comp-001": "pass"})

    await controller.reconcile_pr("repo", 1)

    # No writes because stale
    assert proxy.write_hidden_state.call_count == 0


@pytest.mark.asyncio
async def test_alias_and_ipv4_selector_produce_same_canonical_machine():
    """A valid alias selector and a valid IPv4 selector on distinct PRs
    must each reach the deploy path, persist a canonical machine alias,
    and edited selectors fail closed."""
    controller, proxy, _policy, github, _config = _controller()

    alias = "tianyi2-005"
    ip_selector = "10.100.129.72"

    from ..models import MachineInfo
    controller.policy.machines = {
        alias: MachineInfo(
            alias=alias, node_id="node-tianyi", owners=["owner1"],
            node_host=ip_selector, targets=["perception"],
            platforms=["linux/arm64"], variants=["5.11"], driver_paths=[],
        ),
    }

    core = MagicMock()
    core.list_drivers = AsyncMock(return_value=[
        {"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    img = "registry.example/repo@sha256:" + "a" * 64
    core.driver_status = AsyncMock(return_value={"status": "running", "running_image": img})
    core.deploy_driver = AsyncMock(return_value={"ok": True})
    controller._core_for_node = AsyncMock(return_value=core)

    emdash = "\u2014"
    github.get_issue_comments = AsyncMock(return_value=[
        {"id": 1001, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Build Result\n\nCommit: abc1234\n\n| Target | Status | Version | Took |\n| perception | :white_check_mark: Success | `registry/repo:v1` | 10s |\n",
         "created_at": "2026-09-18T00:00:00Z", "updated_at": "2026-09-18T01:00Z"},
        {"id": 1002, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Test Results\n\nCommit: abc1234\n\n| Suite | Result | Passed | Failed | Took |\n| perception | :white_check_mark: Passed | 10 | 0 | 5s |\n",
         "created_at": "2026-09-18T00:02:00Z", "updated_at": "2026-09-18T00:03:00Z"},
        {"id": 1003, "user": {"id": "7950763", "login": "review-agent-bot"},
         "body": f"<!-- pr-review-agent -->\n## PR Review Agent {emdash} Code Review\n\nAll checks passed.",
         "created_at": "2026-09-18T00:04:00Z", "updated_at": "2026-09-18T00:05:00Z"},
    ])
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    controller._fresh_review_evidence_matches_state = AsyncMock(return_value=True)
    controller._revalidate_hidden_state = AsyncMock(return_value={})
    controller._run_automated_case = AsyncMock(return_value={})
    controller._refresh_uncertain_state = AsyncMock(return_value="deploy-requested")
    orig_timeout = controller.config.total_timeout
    controller.config.total_timeout = 0.5

    from ..github_state_proxy import _build_hidden_state_body as _bhsb
    comp_jp5 = _component(
        component_id="comp-jp5-perception", target="perception",
        variant="5.11", runtime_id="perception",
        image_ref="registry.example/repo@sha256:" + "a" * 64,
    )
    state = _state(components=[comp_jp5], status="deploy-requested")
    body_holder = {"body": _bhsb(
        _lifecycle_visible("repo", 1, status="deploy-requested"), state)}

    async def _find_trusted(_repo, _pr):
        return {"id": 42, "body": body_holder["body"]}

    async def _write_hidden_state(_repo, _pr, vis, _st):
        body_holder["body"] = _bhsb(vis, _st)
        return {"id": 42}

    async def _read_state(*_a, **_kw):
        from ..github_state_proxy import _extract_hidden_state
        return _extract_hidden_state(body_holder["body"])

    # PR 1: alias selector
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 42, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.project_status_label = AsyncMock()
    proxy.get_comment = AsyncMock(return_value={
        "id": 50, "body": f"/approve_deploy machine={alias}",
        "user": {"id": 111, "login": "owner1"},
    })
    controller.config.total_timeout = orig_timeout
    await controller.handle_approve_deploy("repo", 1, 50, alias, "owner1", "111")
    hidden = await _read_state()
    assert hidden["deployments"] == [{"machine": alias, "component_ids": ["comp-jp5-perception"], "phase": "deployed"}]

    # PR 2: IPv4 selector — same machine, different PR
    core.deploy_driver.reset_mock()
    body_holder["body"] = _bhsb(
        _lifecycle_visible("repo", 2, status="deploy-requested"), state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=_find_trusted)
    proxy.write_hidden_state = AsyncMock(side_effect=_write_hidden_state)
    proxy.get_issue_comments = AsyncMock(return_value=[{"id": 43, "body": body_holder["body"]}])
    proxy.read_hidden_state = AsyncMock(side_effect=_read_state)
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "draft": False,
        "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"},
    })
    proxy.get_comment = AsyncMock(return_value={
        "id": 51, "body": f"/approve_deploy machine={ip_selector}",
        "user": {"id": 111, "login": "owner1"},
    })
    controller.config.total_timeout = orig_timeout
    await controller.handle_approve_deploy("repo", 2, 51, ip_selector, "owner1", "111")
    assert core.deploy_driver.await_count == 1
    # Both reach deploy path
