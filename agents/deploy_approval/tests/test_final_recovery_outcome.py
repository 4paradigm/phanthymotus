"""Final recovery/outcome regressions for Deploy Approval."""

from __future__ import annotations

import inspect
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from .. import agent_core_client as agent_core_client_module
from .. import comments as comments_mod
from ..agent_core_client import (
    AgentCoreClient,
    AgentCoreDeployOutcomeUncertain,
    AgentCoreError,
)
from ..clients_common import SecurityError
from ..config import Config, validate_config
from ..github_state_proxy import GitHubStateProxy, _validate_hidden_state
from ..models import BuildInfo, MachineInfo
from ..policy import Policy
from ..github_command_watcher import GitHubCommandWatcher
from ..router_webhook import webhook
from ..service import DeployController, DeployControllerError, DeployOutcomeUncertain
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
        idx=0,
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
    from ..review_comment_parser import ReviewCommentEvidence, ReviewBuild
    controller, proxy, policy, github, config = _controller()
    evidence = ReviewCommentEvidence(
        head_sha="a" * 40,
        commit_prefix="abcdef1",
        build_comment_id=5001,
        build_comment_updated_at="2026-09-03T10:00:00Z",
        test_comment_id=5002,
        test_comment_updated_at="2026-09-03T10:01:00Z",
        code_review_comment_id=5003,
        code_review_comment_updated_at="2026-09-03T10:02:00Z",
        builds=[ReviewBuild(target="perception", driver_path="", variant="5.11", success=True, version="release.260918.abcdef1", image_tag="ccr.ccs.tencentyun.com/repo:v1")],
        review_author_id="7950763",
    )
    builds = [BuildInfo(0, "perception", "", "5.11", True, "registry.example/repo:v1", True)]
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
async def test_new_approve_from_uncertain_refreshes_before_clean_gate():
    controller, proxy, policy, github, config = _controller()
    state = _state()
    refreshed_state = _state(
        review_evidence={"build_comment_id": 99, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[_component()],
        deployments=[],
        command={"comment_id": 17, "kind": "approve_deploy", "phase": "completed", "args": {"machine": "test-machine"}},
    )
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

    await watcher._process_pr("repo", 1)

    assert state["command"]["phase"] == "uncertain"
    assert controller.on_command.await_count == 0
    assert controller._core_for_node.await_count == 0
    assert controller._deploy_component.await_count == 0
    assert proxy.write_hidden_state.await_count == 0


@pytest.mark.asyncio
async def test_watcher_uncertain_new_approve_refreshes_review_before_clean_gate():
    from ..review_comment_parser import ReviewCommentEvidence, ReviewBuild
    controller, proxy, policy, github, config = _controller()
    state = _state()
    state["command"]["phase"] = "uncertain"
    state["last_processed_comment_id"] = 17

    evidence = ReviewCommentEvidence(
        head_sha="a" * 40,
        commit_prefix="abcdef1",
        build_comment_id=5001,
        build_comment_updated_at="2026-09-03T10:00:00Z",
        test_comment_id=5002,
        test_comment_updated_at="2026-09-03T10:01:00Z",
        code_review_comment_id=5003,
        code_review_comment_updated_at="2026-09-03T10:02:00Z",
        builds=[ReviewBuild(target="perception", driver_path="", variant="5.11", success=True, version="release.260918.abcdef1", image_tag="ccr.ccs.tencentyun.com/repo:v1")],
        review_author_id="7950763",
    )

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
    proxy.persist_cursor = AsyncMock()
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

    await watcher._process_pr("repo", 1)

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
    from ..config import REVIEW_AGENT_GITHUB_LOGIN, REVIEW_AGENT_GITHUB_USER_ID
    from ..tests.conftest import make_config
    return make_config(
        review_comment_author_id=REVIEW_AGENT_GITHUB_USER_ID,
        review_comment_author_login=REVIEW_AGENT_GITHUB_LOGIN,
        agent_core_tokens={"m1": "test-token"},
    )


def _make_policy():
    from ..policy import Policy
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
    policy.check_full_coverage.return_value = None
    policy.get_machine_groups_for_components.return_value = []
    return policy


def _make_controller(fake_proxy, policy, fake_github):
    from ..config import REVIEW_AGENT_GITHUB_LOGIN, REVIEW_AGENT_GITHUB_USER_ID
    from ..service import DeployController
    config = _fake_config()
    config.review_comment_author_id = REVIEW_AGENT_GITHUB_USER_ID
    config.review_comment_author_login = REVIEW_AGENT_GITHUB_LOGIN
    config.agent_core_tokens = {"m1": "test-token"}
    controller = DeployController(config, fake_proxy, policy, fake_github)
    return controller


# ── Legacy digest migration regression tests (no Registry) ──────────


@pytest.mark.asyncio
async def test_legacy_digest_undeployed_component_migrates_to_tag_preserving_component_id(config, monkeypatch):
    """TEST 1: Old component with digest image_ref, no deployment, same semantic key
    -> migrate to tag, keep component_id. ZERO Registry calls."""
    import json
    from unittest.mock import AsyncMock, MagicMock
    from ..github_state_proxy import HIDDEN_STATE_MARKER
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
    import json
    from unittest.mock import AsyncMock, MagicMock
    from ..github_state_proxy import HIDDEN_STATE_MARKER

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
    import json
    from unittest.mock import AsyncMock
    from ..github_state_proxy import HIDDEN_STATE_MARKER

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
    import json
    from unittest.mock import AsyncMock
    from ..github_state_proxy import HIDDEN_STATE_MARKER

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
    import json
    from unittest.mock import AsyncMock
    from ..github_state_proxy import HIDDEN_STATE_MARKER

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
    import json
    from unittest.mock import AsyncMock
    from ..github_state_proxy import HIDDEN_STATE_MARKER

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
    import hashlib
    from unittest.mock import AsyncMock, MagicMock
    from ..review_comment_parser import ReviewBuild

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
    import json
    import hashlib
    from unittest.mock import AsyncMock
    from ..github_state_proxy import HIDDEN_STATE_MARKER

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
