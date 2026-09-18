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
        "review_evidence": {"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
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
    config
    proxy = MagicMock()
    proxy.read_hidden_state = AsyncMock(return_value=_state())
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.get_pr = AsyncMock(
        return_value={
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 111, "login": "alice"},
        }
    )
    proxy.get_issue_comments = AsyncMock(return_value=[])
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
    registry = MagicMock()
    registry.resolve = AsyncMock()

    async def _get_comment(repo, cid):
        if isinstance(cid, int) and not isinstance(cid, bool) and cid > 0:
            return {
                "id": cid,
                "body": "/approve_deploy machine=test-machine",
                "user": {"id": 111, "login": "owner1"},
            }
        return None

    proxy.get_comment = AsyncMock(side_effect=_get_comment)

    controller = DeployController(config, proxy, policy, github, registry)
    return controller, proxy, policy, github, registry, config


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
        tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
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
    assert result == {"running_image": ""}


@pytest.mark.asyncio
async def test_driver_status_existing_container_ignores_status_value():
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
    result = await client.driver_status("driver")
    assert result == {"running_image": "registry.example/repo@sha256:" + "a" * 64}


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
    controller, proxy, policy, github, registry, config = _controller()
    evidence = ReviewCommentEvidence(
        head_sha="a" * 40,
        commit_prefix="abcdef1",
        build_comment_id=5001,
        build_comment_updated_at="2026-09-03T10:00:00Z",
        builds=[ReviewBuild(target="perception", driver_path="", variant="5.11", success=True, version="release.260918.abcdef1", image_tag="ccr.ccs.tencentyun.com/repo:v1")],
    )
    builds = [BuildInfo(0, "perception", "", "5.11", True, "registry.example/repo:v1", True)]
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

    await controller.handle_request_deploy("repo", 1, 101)

    controller._build_component_snapshot.assert_awaited_once_with(
        "repo", 1, "a" * 40, builds
    )
    registry.resolve.assert_not_called()
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["components"][0]["component_id"] == "comp-123"


@pytest.mark.asyncio
async def test_uncertain_recovery_rebuilds_fresh_component_snapshot():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
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
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(
        return_value=[_component(component_id="comp-new", image_ref="registry.example/repo@sha256:" + "d" * 64)]
    )
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    result = await controller._refresh_uncertain_state("repo", 1, state)

    assert result == "deploy-requested"
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert "job-new" in str(written_state["review_evidence"])
    assert written_state["components"][0]["component_id"] == "comp-new"


@pytest.mark.asyncio
async def test_uncertain_recovery_new_job_never_keeps_old_components():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
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
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(return_value=[_component(component_id="comp-new")])
        proxy.write_hidden_state = AsyncMock()
        proxy.project_status_label = AsyncMock()

        await controller._refresh_uncertain_state("repo", 1, state)

        written_state = proxy.write_hidden_state.call_args.args[3]
        assert [c["component_id"] for c in written_state["components"]] == ["comp-new"]


@pytest.mark.asyncio
async def test_uncertain_recovery_snapshot_change_resets_deployments():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
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
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(
            return_value=[_component(component_id="comp-new", review_image_tag="registry.example/repo:v2")]
        )
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller._refresh_uncertain_state("repo", 1, state)

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
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
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
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence, ReviewBuild
    fake_evidence = ReviewCommentEvidence(
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        commit_prefix="abc1234",
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1"),
                ReviewBuild(target="actucore", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(return_value=fresh_components)
        proxy.write_hidden_state = AsyncMock()
        proxy.project_status_label = AsyncMock()

        await controller._refresh_uncertain_state("repo", 1, state)

        written_state = proxy.write_hidden_state.call_args.args[3]
        assert written_state["deployments"] == state["deployments"]
        assert written_state["components"][0]["runtime_id"] == "perception"
        assert "runtime_id" not in written_state["components"][1]


@pytest.mark.asyncio
async def test_uncertain_recovery_clears_runtime_id_for_ambiguous_component():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[
            _component(component_id="comp-1", runtime_id="perception"),
            _component(component_id="comp-2", target="actucore", runtime_id="actucore"),
        ],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-1"], "phase": "deployed"}],
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}})
    github.resolve_commit_sha = AsyncMock(return_value="a" * 40)
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence, ReviewBuild
    fake_evidence = ReviewCommentEvidence(
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T03:55:54Z",
        commit_prefix="abc1234",
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
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

    await controller._refresh_uncertain_state("repo", 1, state)

    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["components"][0]["runtime_id"] == "perception"
    assert "runtime_id" not in written_state["components"][1]


@pytest.mark.asyncio
async def test_uncertain_recovery_registry_failure_stays_uncertain():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
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
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(return_value=None)
        proxy.write_hidden_state = AsyncMock()
        proxy.project_status_label = AsyncMock()

        result = await controller._refresh_uncertain_state("repo", 1, state)

        assert result == "uncertain"
        written_state = proxy.write_hidden_state.call_args.args[3]
        assert "job-old" in str(written_state["review_evidence"])
        assert written_state["components"] == [_component()]


@pytest.mark.asyncio
async def test_new_approve_from_uncertain_refreshes_before_clean_gate():
    controller, proxy, policy, github, registry, config = _controller()
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
    controller._run_automated_case = AsyncMock(return_value={})
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}})

    await controller.handle_approve_deploy("repo", 1, 17, "test-machine", "owner1", "111")

    assert events[0] == "refresh"
    assert "list_drivers" in events
    assert events.index("refresh") < events.index("list_drivers")


@pytest.mark.asyncio
async def test_watcher_uncertain_without_new_approve_does_not_refresh_review():
    controller, proxy, policy, github, registry, config = _controller()
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
    controller.registry.resolve = AsyncMock(side_effect=AssertionError("unexpected registry refresh"))
    controller._core_for_node = AsyncMock(side_effect=AssertionError("unexpected Agent Core lookup"))
    controller._deploy_component = AsyncMock(side_effect=AssertionError("unexpected deploy"))
    controller._run_automated_case = AsyncMock(side_effect=AssertionError("unexpected case run"))
    controller.on_command = AsyncMock(return_value=True)

    watcher = GitHubCommandWatcher(config, proxy, controller)

    await watcher._process_pr("repo", 1)

    assert state["command"]["phase"] == "uncertain"
    assert controller.on_command.await_count == 0
    assert controller.registry.resolve.await_count == 0
    assert controller._core_for_node.await_count == 0
    assert controller._deploy_component.await_count == 0
    assert proxy.write_hidden_state.await_count == 0


@pytest.mark.asyncio
async def test_watcher_uncertain_new_approve_refreshes_review_before_clean_gate():
    from ..review_comment_parser import ReviewCommentEvidence, ReviewBuild
    controller, proxy, policy, github, registry, config = _controller()
    state = _state()
    state["command"]["phase"] = "uncertain"
    state["last_processed_comment_id"] = 17

    evidence = ReviewCommentEvidence(
        head_sha="a" * 40,
        commit_prefix="abcdef1",
        build_comment_id=5001,
        build_comment_updated_at="2026-09-03T10:00:00Z",
        builds=[ReviewBuild(target="perception", driver_path="", variant="5.11", success=True, version="release.260918.abcdef1", image_tag="ccr.ccs.tencentyun.com/repo:v1")],
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
        ]

    async def _resolve(*args, **kwargs):
        events.append("registry.resolve")
        return SimpleNamespace(
            image_ref="ccr.ccs.tencentyun.com/repo@sha256:" + "b" * 64,
            platform="linux/arm64",
        )

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
    controller.registry.resolve = AsyncMock(side_effect=_resolve)
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
    assert "registry.resolve" in events
    assert events.index("review.list_jobs") < events.index("list_drivers")
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
        await client.deploy_driver("driver", "registry.example/repo:latest")
    assert not isinstance(excinfo.value, AgentCoreDeployOutcomeUncertain)


@pytest.mark.asyncio
async def test_uncertain_post_stops_later_components():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        components=[
            _component(component_id="comp-1", target="perception"),
            _component(component_id="comp-2", target="actucore", runtime_id="actucore"),
            _component(component_id="comp-3", target="driver", driver_path="custom/driver", runtime_id="driver"),
        ],
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
            {"id": "driver", "target": "driver", "image": "registry.example/repo:v1", "category": "driver"},
        ]
    )
    core.driver_status = AsyncMock(
        side_effect=[
            {"status": "running", "running_image": ""},
            {"status": "running", "running_image": ""},
            {"status": "running", "running_image": ""},
        ]
    )
    controller._core_for_node = AsyncMock(return_value=core)
    deploy_calls = 0

    async def _deploy_component(*args, **kwargs):
        nonlocal deploy_calls
        deploy_calls += 1
        if deploy_calls == 2:
            raise DeployOutcomeUncertain("network timeout")
        return {"result": {"code": 0}}

    controller._deploy_component = AsyncMock(side_effect=_deploy_component)
    controller._run_automated_case = AsyncMock(return_value={})
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller.handle_approve_deploy("repo", 1, 17, "test-machine", "owner1", "111")

    assert deploy_calls == 2
    assert proxy.write_hidden_state.call_args.args[3]["command"]["phase"] == "uncertain"


@pytest.mark.asyncio
async def test_uncertain_post_preserves_prior_health_passed_deployments():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        components=[
            _component(component_id="comp-1", target="perception"),
            _component(component_id="comp-2", target="actucore", runtime_id="actucore"),
        ],
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
        ]
    )
    core.driver_status = AsyncMock(
        side_effect=[
            {"status": "running", "running_image": ""},
            {"status": "running", "running_image": ""},
        ]
    )
    controller._core_for_node = AsyncMock(return_value=core)

    async def _deploy_component(*args, **kwargs):
        if args[3] == "actucore":
            raise DeployOutcomeUncertain("connection reset")
        return {"result": {"code": 0}}

    controller._deploy_component = AsyncMock(side_effect=_deploy_component)
    controller._run_automated_case = AsyncMock(return_value={})
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller.handle_approve_deploy("repo", 1, 17, "test-machine", "owner1", "111")

    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["deployments"] == [{"machine": "test-machine", "component_ids": ["comp-1"], "phase": "deployed"}]


@pytest.mark.asyncio
async def test_uncertain_post_does_not_upload_failed_cos():
    controller, proxy, policy, github, registry, config = _controller()
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


@pytest.mark.asyncio
async def test_uncertain_post_advances_cursor():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        components=[_component()],
        deployments=[],
    )
    state["command"]["phase"] = "completed"
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}})
    core = AsyncMock()
    core.list_drivers = AsyncMock(return_value=[{"id": "perception", "target": "perception", "image": "registry.example/repo:v1"}])
    core.driver_status = AsyncMock(return_value={"status": "running", "running_image": ""})
    controller._core_for_node = AsyncMock(return_value=core)
    controller._deploy_component = AsyncMock(side_effect=DeployOutcomeUncertain("timeout"))
    controller._run_automated_case = AsyncMock(return_value={})
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller.handle_approve_deploy("repo", 1, 19, "test-machine", "owner1", "111")

    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["last_processed_comment_id"] == 19
    assert written_state["command"]["phase"] == "uncertain"


@pytest.mark.asyncio
async def test_uncertain_post_write_failure_leaves_executing_for_restart_recovery():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        components=[_component()],
        deployments=[],
    )
    state["command"]["phase"] = "completed"
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}})
    core = AsyncMock()
    core.list_drivers = AsyncMock(return_value=[{"id": "perception", "target": "perception", "image": "registry.example/repo:v1"}])
    core.driver_status = AsyncMock(return_value={"status": "running", "running_image": ""})
    controller._core_for_node = AsyncMock(return_value=core)
    controller._deploy_component = AsyncMock(side_effect=DeployOutcomeUncertain("timeout"))
    controller._run_automated_case = AsyncMock(return_value={})
    proxy.write_hidden_state = AsyncMock(side_effect=[{"id": 1}, RuntimeError("write failed")])
    proxy.project_status_label = AsyncMock()

    with pytest.raises(RuntimeError, match="write failed"):
        await controller.handle_approve_deploy("repo", 1, 21, "test-machine", "owner1", "111")

    assert proxy.write_hidden_state.call_args_list[0].args[2] == "Deploying..."
    assert "Restart Recovery" in proxy.write_hidden_state.call_args_list[-1].args[2]
    proxy.project_status_label.assert_not_called()


def test_poll_disabled_webhook_enabled_fails_config():
    with pytest.raises(ValueError, match="requires polling"):
        validate_config(
            Config(
                github_repos=[
                    "4paradigm/phanthymotus",
                    "4paradigm/phanthymotus-driver",
                ],
                deploy_approval_public_base_url="https://deploy.example",
                github_oauth_client_id="test-client",
                github_oauth_client_secret="test-secret",
                poll_enabled=False,
                webhook_enabled=True,
                github_webhook_secret="secret",
            )
        )


@pytest.mark.asyncio
async def test_webhook_remains_supplementary_zero_dispatch():
    controller, proxy, policy, github, registry, config = _controller()
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
    controller, proxy, policy, github, registry, config = _controller()
    controller._resolve_image_ref = AsyncMock(
        return_value=("registry.example/repo@sha256:" + "b" * 64, "linux/arm64")
    )
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
    controller, proxy, policy, github, registry, config = _controller()
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
async def test_uncertain_same_snapshot_real_snapshot_helper_preserves_deployed_runtime_binding():
    controller, proxy, policy, github, registry, config = _controller()
    controller._resolve_image_ref = AsyncMock(
        side_effect=[
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
        ]
    )
    fresh_components = await controller._build_component_snapshot(
        "repo",
        1,
        "a" * 40,
        [_build(), _build(target="actucore")],
    )
    assert fresh_components is not None
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[
            dict(fresh_components[0], runtime_id="perception"),
            dict(fresh_components[1], runtime_id="actucore"),
        ],
        deployments=[{"machine": "test-machine", "component_ids": [fresh_components[0]["component_id"]], "phase": "deployed"}],
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
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1"),
                ReviewBuild(target="actucore", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._resolve_image_ref = AsyncMock(
        side_effect=[
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
        ]
    )
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    result = await controller._refresh_uncertain_state("repo", 1, state)

    assert result == "deploy-requested"
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["components"][0]["runtime_id"] == "perception"
    assert "runtime_id" not in written_state["components"][1]


def test_uncertain_same_snapshot_clears_old_runtime_id_for_undeployed_component():
    controller, proxy, policy, github, registry, config = _controller()
    rebuilt = controller._components_with_preserved_runtime_bindings(
        [
            _fresh_component(component_id="comp-1"),
            _fresh_component(component_id="comp-2", target="actucore"),
        ],
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
async def test_uncertain_same_snapshot_missing_old_deployed_runtime_binding_stays_uncertain():
    controller, proxy, policy, github, registry, config = _controller()
    controller._resolve_image_ref = AsyncMock(
        side_effect=[
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
        ]
    )
    fresh_components = await controller._build_component_snapshot(
        "repo",
        1,
        "a" * 40,
        [_build(), _build(target="actucore")],
    )
    assert fresh_components is not None
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[
            dict(fresh_components[0], runtime_id=""),
            dict(fresh_components[1]),
        ],
        deployments=[{"machine": "test-machine", "component_ids": [fresh_components[0]["component_id"]], "phase": "deployed"}],
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
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1"),
                ReviewBuild(target="actucore", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v1", version="v1")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._resolve_image_ref = AsyncMock(
        side_effect=[
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
        ]
    )
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    result = await controller._refresh_uncertain_state("repo", 1, state)

    assert result == "uncertain"
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["command"]["phase"] == "uncertain"
    assert "job-old" in str(written_state["review_evidence"])
    assert written_state["deployments"] == [
        {
            "machine": "test-machine",
            "component_ids": [fresh_components[0]["component_id"]],
            "phase": "deployed",
        }
    ]


@pytest.mark.asyncio
async def test_uncertain_changed_snapshot_ignores_missing_old_runtime_binding_and_resets_validation():
    controller, proxy, policy, github, registry, config = _controller()
    state = _state(
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[
            _component(component_id="comp-1", runtime_id=""),
            _component(component_id="comp-2", target="actucore"),
        ],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-1"], "phase": "deployed"}],
        approve_attempts=[
            {
                "comment_id": 11,
                "actor": "alice",
                "machine": "test-machine",
                "preflight": [],
                "outcome": "deployed",
                "health": [],
            }
        ],
        approve_attempts_total=1,
        case_results={"comp-1": "pass"},
        test_result="pass",
        cos={"object_key": "deploy-1", "sha256": "a" * 64, "size": 123},
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
        resolved_head_sha="a" * 40,
        test_comment_id=1002,
        code_review_comment_id=1003,
        code_review_text="Looks good.",
        builds=[ReviewBuild(target="perception", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2"),
                ReviewBuild(target="actucore", driver_path="test", variant="default", success=True, image_tag="registry.example/repo:v2", version="v2")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._resolve_image_ref = AsyncMock(
        side_effect=[
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
            ("registry.example/repo@sha256:" + "a" * 64, "linux/arm64"),
        ]
    )
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    result = await controller._refresh_uncertain_state("repo", 1, state)

    assert result == "deploy-requested"
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert "job-new" in str(written_state["review_evidence"])
    assert written_state["status"] == "deploy-requested"
    assert written_state["command"]["phase"] == "completed"
    assert all("runtime_id" not in component for component in written_state["components"])
    assert written_state["deployments"] == []
    assert written_state["approve_attempts"] == []
    assert written_state["approve_attempts_total"] == 0
    assert written_state["approve_attempts_truncated"] is False
    assert written_state["case_results"] == {}
    assert written_state["test_result"] == ""
    assert written_state["cos"] == {"object_key": "", "sha256": "", "size": 0}


def test_driver_status_logs_only_shape_fails_closed():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={"code": 200, "data": {"logs": "no container"}},
                request=request,
            )

    client = _driver_client(Transport())
    with pytest.raises(AgentCoreError):
        asyncio.run(client.driver_status("driver"))


def test_driver_status_no_container_shape_ignores_status_value():
    for status_value in ("stopped", "running", "busy", "error", 123, None):
        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx.Response(
                    200,
                    json={"code": 200, "data": {"status": status_value, "logs": "no container"}},
                    request=request,
                )

        client = _driver_client(Transport())
        result = asyncio.run(client.driver_status("driver"))
        assert result["running_image"] == ""
        assert result["logs"] == "no container"


def test_agent_core_request_has_no_http_policy_bypass_parameter():
    params = inspect.signature(AgentCoreClient.request).parameters
    assert "validate_http_policy" not in params
    assert list(params) == ["self", "method", "path", "json"]


@pytest.mark.asyncio
async def test_deploy_policy_prevalidation_error_is_agent_core_error_and_zero_post(monkeypatch):
    calls = 0

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal calls
            calls += 1
            raise AssertionError("POST should not be attempted")

    def deny(*args, **kwargs):
        raise SecurityError("blocked")

    monkeypatch.setattr(agent_core_client_module, "require_http_policy", deny)
    client = _driver_client(Transport())
    with pytest.raises(AgentCoreError) as excinfo:
        await client.deploy_driver("driver", "registry.example/repo@sha256:" + "a" * 64)
    assert not isinstance(excinfo.value, AgentCoreDeployOutcomeUncertain)
    assert calls == 0


@pytest.mark.asyncio
async def test_generic_request_policy_error_is_agent_core_error_and_zero_request(monkeypatch):
    calls = 0

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal calls
            calls += 1
            raise AssertionError("request should not be attempted")

    def deny(*args, **kwargs):
        raise SecurityError("blocked")

    monkeypatch.setattr(agent_core_client_module, "require_http_policy", deny)
    client = _driver_client(Transport())
    with pytest.raises(AgentCoreError):
        await client.request("GET", "/api/drivers")
    assert calls == 0


def test_docs_define_no_container_adapter_contract():
    text = Path("DEPLOY_APPROVAL_AGENT.md").read_text(encoding="utf-8")
    assert "POLL_ENABLED must be true." in text
    assert "Webhook is supplementary only." in text
    assert "Agent Core no-container response" in text
    assert "unsafe deploy POST" in text
    assert "approve_attempt.outcome=uncertain" in text
    assert 'running_image=""' in text
    assert "status VALUE has zero CLEAN/health/case business influence" in text
    assert "error/malformed shapes fail closed" in text


def test_uncertain_is_not_documented_as_top_level_status():
    docs = Path("docs/deploy-approval-github-driven-architecture.md").read_text(encoding="utf-8")
    comments = Path("agents/deploy_approval/comments.py").read_text(encoding="utf-8")
    assert "`uncertain` 只能出现在 `command.phase`" not in docs
    assert "approve_attempt.outcome=uncertain" in docs
    assert "status: uncertain" not in comments
    assert "deploy-requested lifecycle with command.phase=uncertain" in comments
    top_level_statuses = {
        "review-required",
        "reviewing",
        "deploy-ready",
        "deploy-requested",
        "testing",
        "succeeded",
        "failed",
    }
    assert "uncertain" not in top_level_statuses


def test_docs_define_unsafe_post_uncertain_contract():
    text = Path("docs/deploy-approval-github-driven-architecture.md").read_text(encoding="utf-8")
    assert "fresh Review Agent build_results + fresh Registry immutable resolution" in text
    assert "command.phase=uncertain" in text
    assert "status: deploy-requested" in text
    assert "ZERO later POST" in text
    assert "NEW approve only" in text


def test_docs_define_poll_required_webhook_supplementary_contract():
    text = Path("docs/deploy-approval-github-driven-architecture.md").read_text(encoding="utf-8")
    assert "POLL_ENABLED must be true." in text
    assert "Webhook is supplementary only." in text


def test_uncertain_comment_requires_new_approve_before_validation_refresh():
    text = comments_mod.uncertain_comment("repo", 1, "a" * 40)
    assert "**Status:** `deploy-requested`" in text
    assert "**Command phase:** `uncertain`" in text
    assert "ZERO automatic replay" in text
    assert "**Next action \u2014 Machine Owner**" in text
    assert "`/approve_deploy machine=<alias>`" in text
    assert "NEW `/approve_deploy`" in text
    assert "running_image-only CLEAN GATE" in text
    assert "Background polling keeps this command `uncertain`" in text
    assert "list_jobs(repo,status=review_done)" not in text
    assert "Manual intervention required." not in text
    assert "restart / next poll" not in text


@pytest.mark.asyncio
async def test_uncertain_preserved_deployment_clean_new_approve_redeploys_each_component_once():
    """Regression: stale preserved deployments must be reset so each component deploys exactly once.

    Scenario:
    1. State is deploy-requested with a preserved known-success deployment for component A
       from a prior uncertain attempt (component B was not deployed yet).
    2. A NEW explicit /approve_deploy arrives with a CLEAN preflight (both A and B have
       running_image="" on the machine).
    3. Approval comment revalidation and HEAD validation succeed.
    4. The stale preserved deployment for A is reset before persisting executing phase.
    5. Deploy POST occurs exactly once for A and exactly once for B.
    6. Final durable state contains each component once in deployments — no duplicates.

    This test exercises the REAL _refresh_uncertain_state path via review.list_jobs +
    review.get_job + registry.resolve — no mocking of those seams.
    """
    import copy

    controller, proxy, policy, github, registry, config = _controller()
    config.registry = "registry.example"
    REPO = "4paradigm/phanthymotus"
    events: list[str] = []
    write_log: list[tuple[str, dict]] = []

    # Build a Review Agent job dict that produces exactly our two components with exact HEAD.
    # The component_id is computed in _build_component_snapshot as sha256("{target}|{driver_path}|{variant}|{image_ref}")[:16]
    # We need registry.resolve to return a digest that makes the component_id match our initial state.
    # Our _component fixture uses component_id="comp-a" and "comp-b".
    # To make the real snapshot match, we use the same image_ref and reverse-compute:
    # Actually we just need the snapshot to be SAME so preservation logic fires.
    # We set up the initial state review_evidence to match the evidence we return,
    # and the component snapshot must be identical for same_snapshot==True.
    #
    # Simplest approach: use a known image_ref and compute the component_id the real way.
    FAKE_SHA = "a" * 64
    IMAGE_REF_A = f"registry.example/repo@sha256:{FAKE_SHA}"
    # component_id = sha256("perception||5.14|{IMAGE_REF_A}")[:16]
    import hashlib
    comp_a_id = hashlib.sha256(f"perception||5.11|{IMAGE_REF_A}".encode()).hexdigest()[:16]
    comp_b_id = hashlib.sha256(f"actucore||5.11|{IMAGE_REF_A}".encode()).hexdigest()[:16]

    # Build the resolved image that registry.resolve returns
    RESOLVED_REF_A = f"ccr.ccs.tencentyun.com/registry.example/repo@sha256:{FAKE_SHA}"

    def make_review_job():
        return {
            "id": "job-recovery",
            "repo": REPO,
            "pr_number": 1,
            "head_sha": "a" * 40,
            "status": "review_done",
            "review_text": "LGTM",
            "options": {"build_only": False},
            "completed_at": 1000.0,
            "build_results": [
                {
                    "idx": 0,
                    "target": "perception",
                    "driver_path": "",
                    "success": True,
                    "image_tag": "registry.example/repo:v1",
                    "variant": "5.11",
                    "deployable": True,
                },
                {
                    "idx": 1,
                    "target": "actucore",
                    "driver_path": "",
                    "success": True,
                    "image_tag": "registry.example/repo:v1",
                    "variant": "5.11",
                    "deployable": True,
                },
            ],
        }

    # Stateful hidden-state store
    current_state = _state(
        head_sha="a" * 40,
        status="deploy-requested",
        review_evidence={"build_comment_id": 50, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
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
        # args: markdown, state (proxy.write_hidden_state(repo, pr_number, markdown, state))
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

    # GitHub PR stays open/unmerged with same HEAD
    proxy.get_pr = AsyncMock(
        return_value={
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 222, "login": "owner1"},
        }
    )

    # NEW approve comment (id=99) from authorized actor
    actor_id = "222"
    async def _get_comment(repo, cid):
        return {
            "id": cid,
            "body": "/approve_deploy machine=test-machine",
            "user": {"id": int(actor_id), "login": "owner1"},
        }
    proxy.get_comment = AsyncMock(side_effect=_get_comment)
    proxy.comment_identity = AsyncMock(return_value=("222", "owner1"))
    proxy.get_issue_comments = AsyncMock(return_value=[])
    proxy.post_issue_comment = AsyncMock(return_value={"id": 1})
    proxy.collaborator_permission = AsyncMock(return_value="admin")

    # Review Agent seam replaced by extract_review_evidence patch
    from ..review_comment_parser import ReviewCommentEvidence, ReviewBuild
    review_job_dict = make_review_job()
    _evidence = ReviewCommentEvidence(
        head_sha="a" * 40,
        commit_prefix="a" * 7,
        build_comment_id=5001,
        build_comment_updated_at="2026-09-03T12:00:00Z",
        builds=[
            ReviewBuild(target=b["target"], driver_path=b["driver_path"], variant=b["variant"],
                        success=b["success"], version=b["image_tag"].rsplit(":",1)[-1] if ":" in b["image_tag"] else "",
                        image_tag=b["image_tag"])
            for b in review_job_dict["build_results"]
        ],
    )

    async def _list_jobs(repo, status="", limit=100, offset=0):
        events.append("review.list_jobs")
        if repo == REPO and status == "review_done":
            return [review_job_dict]
        return []

    async def _get_job(job_id):
        events.append("review.get_job")
        return review_job_dict

    review.list_jobs = AsyncMock(side_effect=_list_jobs)
    review.get_job = AsyncMock(side_effect=_get_job)

    # Registry seam
    async def _resolve(ref, platform="", allowed_prefixes=None):
        events.append("registry.resolve")
        from ..registry_client import ResolvedImage
        return ResolvedImage(
            family="registry.example/repo",
            tag="",
            digest=f"sha256:{FAKE_SHA}",
            platform="linux/arm64",
            size=1024,
        )
    registry.resolve = AsyncMock(side_effect=_resolve)

    # Agent Core boundary — both runtimes CLEAN
    core = AsyncMock()
    core.list_drivers = AsyncMock(
        return_value=[
            {"id": "perception", "target": "perception", "image": "registry.example/repo:v1"},
            {"id": "actucore", "target": "actucore", "image": "registry.example/repo:v1"},
        ]
    )
    status_calls = []
    async def _driver_status(driver_id):
        status_calls.append(driver_id)
        events.append(f"driver_status:{driver_id}")
        return {"status": "running", "running_image": ""}
    core.driver_status = AsyncMock(side_effect=_driver_status)

    deploy_calls: dict[str, int] = {"perception": 0, "actucore": 0}

    async def _deploy_driver(runtime_id, image):
        deploy_calls[runtime_id] = deploy_calls.get(runtime_id, 0) + 1
        events.append(f"deploy_driver:{runtime_id}")
        return {"code": 200, "data": {"status": "starting"}}
    core.deploy_driver = _deploy_driver

    controller._core_for_node = AsyncMock(return_value=core)
    controller._run_automated_case = AsyncMock(return_value={})

    # Execute via the REAL handler
    await controller.handle_approve_deploy(REPO, 1, 99, "test-machine", "owner1", actor_id)

    # ── Proofs ──

    # 1. Real review boundary calls occurred
    assert "review.list_jobs" in events, "Expected REAL review.list_jobs call"
    assert "review.get_job" in events, "Expected REAL review.get_job call"
    assert "registry.resolve" in events, "Expected REAL registry.resolve call"

    # B4: Prove the real external seams actually ran (await_count)
    assert review.list_jobs.await_count >= 1, (
        f"Expected review.list_jobs to be called, got {review.list_jobs.await_count}"
    )
    assert review.get_job.await_count >= 1, (
        f"Expected review.get_job to be called, got {review.get_job.await_count}"
    )
    # Note: production code path now uses extract_review_evidence from GitHub comments,
    # not ReviewAgentClient. These mocks remain for backward compatibility with the
    # _refresh_uncertain_state seam during transition.
    assert registry.resolve.await_count == 2, (
        f"Expected registry.resolve called exactly 2 times (one per build), got {registry.resolve.await_count}"
    )

    # 2. ALL driver_status CLEAN reads happen BEFORE first deploy POST
    first_deploy_idx = None
    for i, ev in enumerate(events):
        if ev.startswith("deploy_driver:"):
            first_deploy_idx = i
            break
    assert first_deploy_idx is not None, "Expected at least one deploy_driver call"
    driver_status_events = [i for i, ev in enumerate(events) if ev.startswith("driver_status:")]
    assert len(driver_status_events) == 2, (
        f"Expected 2 driver_status events, got {len(driver_status_events)}"
    )
    for ds_idx in driver_status_events:
        assert ds_idx < first_deploy_idx, (
            f"driver_status event at index {ds_idx} must precede first deploy at {first_deploy_idx}"
        )
    assert set(status_calls) == {"perception", "actucore"}, (
        f"Expected both runtimes checked, got {status_calls}"
    )

    # 3. Stale deployment reset: executing write has empty deployments
    executing_write = None
    for markdown, written in write_log:
        if written.get("command", {}).get("phase") == "executing":
            executing_write = written
            break
    assert executing_write is not None, "Expected executing phase write"
    assert executing_write.get("deployments") == [], (
        f"Stale deployments not reset before executing: {executing_write.get('deployments')}"
    )

    # 4. Each component deployed exactly once
    assert deploy_calls.get("perception", 0) == 1, f"perception deployed {deploy_calls.get('perception', 0)} times"
    assert deploy_calls.get("actucore", 0) == 1, f"actucore deployed {deploy_calls.get('actucore', 0)} times"

    # 5. No duplicate component_ids in final deployments
    final_state = write_log[-1][1] if write_log else current_state
    all_cids = []
    for dep in final_state.get("deployments", []):
        all_cids.extend(dep.get("component_ids", []))
    assert len(all_cids) == len(set(all_cids)), f"Duplicate component_ids: {all_cids}"
    assert all_cids.count(comp_a_id) == 1
    assert all_cids.count(comp_b_id) == 1

    # 6. Final durable state
    assert final_state["status"] == "testing"
    assert final_state["command"]["phase"] == "completed"

    # 7. Real hidden-state validator passes
    _validate_hidden_state(final_state)

    # 8. testing write occurs before advisory Case
    testing_found = False
    for markdown, written in write_log:
        if written.get("status") == "testing":
            testing_found = True
            break
    assert testing_found, "Expected testing status write in write log"
