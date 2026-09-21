"""Stateless GitHub contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

from ..config import Config
from ..github_state_proxy import GitHubStateProxy, MalformedHiddenStateError, _validate_hidden_state
from ..models import MachineInfo
from ..policy import Policy
from ..service import DeployController, DeployControllerError


@pytest.fixture
def config():
    return Config(

        github_repos=["4paradigm/phanthymotus"],
        machine_owners_file="/dev/null",
        poll_interval_seconds=30,
        registry="registry.example",
    )


@pytest.fixture
def mock_github():
    client = MagicMock()
    client.get_comment = AsyncMock()
    client.get_issue_comments = AsyncMock()
    client.post_issue_comment = AsyncMock(return_value={
        "id": 42, "user": {"id": 12345, "login": "test-bot", "type": "Bot"},
        "performed_via_github_app": {"id": 12345},
    })
    client.update_comment = AsyncMock()
    client.get_pr = AsyncMock()
    client.collaborator_permission = AsyncMock()
    client.get_issue_labels = AsyncMock(return_value=[])
    client.set_issue_labels = AsyncMock()
    client.list_open_prs = AsyncMock()
    return client


@pytest.fixture
def proxy(config, mock_github):
    return GitHubStateProxy(config, mock_github, github_app_id="12345")


@pytest.fixture
def policy(config):
    p = Policy(config)
    p.machines = {
        "test-machine": MachineInfo(
            alias="test-machine",
            node_id="node-1",
            owners=["owner1"],
            node_host="127.0.0.1",

            targets=["perception"],
            platforms=["linux/arm64"],
            variants=["5.11"],
        ),
        "driver-machine": MachineInfo(
            alias="driver-machine",
            node_id="node-2",
            owners=["driver-owner"],
            node_host="127.0.0.2",

            targets=["driver"],
            platforms=["linux/arm64"],
            variants=[""],
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
        "status": "deploy-requested",
        "review_evidence": {"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        "components": [_component()],
        "deployments": [],
        "case_results": {},
        "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    }
    state.update(overrides)
    return state




@pytest.mark.asyncio
async def test_request_deploy_re_reads_command_identity_from_github(controller, proxy, mock_github):
    mock_github.get_comment.return_value = {"id": 101, "user": {"id": 111, "login": "alice"}, "body": "/request_deploy"}
    mock_github.get_pr.return_value = {
        "state": "open",
        "merged": False,
        "head": {"sha": "a" * 40},
        "user": {"id": 111, "login": "alice"},
    }
    controller.registry.resolve.return_value = SimpleNamespace(image_ref="registry/repo@sha256:" + "b" * 64, platform="linux/arm64")
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    await controller.handle_request_deploy("4paradigm/phanthymotus", 1, 101)

    mock_github.get_comment.assert_called_once_with("4paradigm/phanthymotus", 101)


@pytest.mark.asyncio
async def test_request_deploy_allows_current_pr_author_id(controller, proxy, mock_github):
    """Narrow permission test: current PR author may execute /request_deploy when all gates valid."""
    from unittest.mock import patch
    from ..review_comment_parser import ReviewCommentEvidence, ReviewBuild

    mock_github.get_comment.return_value = {"id": 101, "user": {"id": 111, "login": "alice"}, "body": "/request_deploy"}
    mock_github.get_pr.return_value = {
        "state": "open",
        "merged": False,
        "head": {"sha": "a" * 40},
        "user": {"id": 111, "login": "alice"},
    }
    proxy.read_hidden_state = AsyncMock(return_value=_state(status="deploy-ready", review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"}, components=[]))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    mock_github.resolve_commit_sha = AsyncMock(return_value="a" * 40)

    # Patch narrow evidence boundary — do NOT test parser/registry/snapshot here
    fake_evidence = ReviewCommentEvidence(
        head_sha="a" * 40,
        commit_prefix="abc1234",
        build_comment_id=1001,
        build_comment_updated_at="2026-09-18T00:01:00Z",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-18T00:03:00Z",
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-18T00:05:00Z",
        builds=[ReviewBuild(target="perception", driver_path="", variant="5.11", success=True, image_tag="registry/repo:v1", version="v1")],
        review_author_id="7950763",
    )
    with patch('agents.deploy_approval.service.extract_review_evidence', return_value=fake_evidence):
        controller._build_component_snapshot = AsyncMock(return_value=[_component()])

    result = await controller.handle_request_deploy("4paradigm/phanthymotus", 1, 101)

    assert result is True
    proxy.write_hidden_state.assert_called_once()
    proxy.project_status_label.assert_called_once_with("4paradigm/phanthymotus", 1, "deploy-requested")


@pytest.mark.asyncio
async def test_request_deploy_rejects_non_pr_author_id(controller, proxy, mock_github):
    mock_github.get_comment.return_value = {"id": 101, "user": {"id": 222, "login": "mallory"}, "body": "/request_deploy"}
    mock_github.get_pr.return_value = {
        "state": "open",
        "merged": False,
        "head": {"sha": "a" * 40},
        "user": {"id": 111, "login": "alice"},
    }
    proxy.write_hidden_state = AsyncMock()
    proxy.post_issue_comment = AsyncMock()

    await controller.handle_request_deploy("4paradigm/phanthymotus", 1, 101)

    controller.registry.resolve.assert_not_called()
    proxy.write_hidden_state.assert_not_called()
    proxy.post_issue_comment.assert_called_once()


@pytest.mark.asyncio
async def test_request_deploy_missing_comment_identity_fails_closed(controller, proxy, mock_github):
    mock_github.get_comment.return_value = {"id": 101, "body": "/request_deploy", "user": {}}
    mock_github.get_pr.return_value = {
        "state": "open",
        "merged": False,
        "head": {"sha": "a" * 40},
        "user": {"id": 111, "login": "alice"},
    }
    proxy.write_hidden_state = AsyncMock()
    proxy.post_issue_comment = AsyncMock()

    await controller.handle_request_deploy("4paradigm/phanthymotus", 1, 101)

    controller.registry.resolve.assert_not_called()
    proxy.write_hidden_state.assert_not_called()
    proxy.post_issue_comment.assert_called_once()


@pytest.mark.asyncio
async def test_request_deploy_unauthorized_has_zero_registry_and_deploy_side_effects(controller, proxy, mock_github):
    mock_github.get_comment.return_value = {"id": 101, "user": {"id": 222, "login": "mallory"}, "body": "/request_deploy"}
    mock_github.get_pr.return_value = {
        "state": "open",
        "merged": False,
        "head": {"sha": "a" * 40},
        "user": {"id": 111, "login": "alice"},
    }
    proxy.write_hidden_state = AsyncMock()
    proxy.post_issue_comment = AsyncMock()

    await controller.handle_request_deploy("4paradigm/phanthymotus", 1, 101)

    controller.registry.resolve.assert_not_called()
    proxy.write_hidden_state.assert_not_called()


@pytest.mark.asyncio
async def test_restart_executing_changes_only_command_phase_to_uncertain(controller, proxy):
    proxy.read_hidden_state = AsyncMock(return_value=_state(command={"comment_id": 1, "kind": "approve_deploy", "phase": "executing", "args": {"machine": "test-machine"}}))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}})

    await controller.reconcile_pr("repo", 1)

    written_state = proxy.write_hidden_state.call_args.args[3]
    markdown = proxy.write_hidden_state.call_args.args[2]
    assert written_state["status"] == "deploy-requested"
    assert written_state["command"]["phase"] == "uncertain"
    assert "**Status:** `deploy-requested`" in markdown
    assert "**Command phase:** `uncertain`" in markdown


@pytest.mark.asyncio
async def test_uncertain_preserves_business_status(controller, proxy):
    proxy.read_hidden_state = AsyncMock(return_value=_state(command={"comment_id": 1, "kind": "approve_deploy", "phase": "executing", "args": {"machine": "test-machine"}}))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}})

    await controller.reconcile_pr("repo", 1)

    assert proxy.write_hidden_state.call_args.args[3]["status"] == "deploy-requested"
    assert proxy.project_status_label.call_args_list[-1].args == ("repo", 1, "deploy-requested")


@pytest.mark.asyncio
async def test_uncertain_visible_status_matches_hidden_status(controller, proxy):
    proxy.read_hidden_state = AsyncMock(return_value=_state(command={"comment_id": 1, "kind": "approve_deploy", "phase": "executing", "args": {"machine": "test-machine"}}))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}})

    await controller.reconcile_pr("repo", 1)

    markdown = proxy.write_hidden_state.call_args.args[2]
    assert "**Status:** `deploy-requested`" in markdown
    assert "**Command phase:** `uncertain`" in markdown


@pytest.mark.asyncio
async def test_uncertain_never_auto_replays_agent_core(controller, proxy):
    proxy.read_hidden_state = AsyncMock(return_value=_state(command={"comment_id": 1, "kind": "approve_deploy", "phase": "executing", "args": {"machine": "test-machine"}}))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.get_pr = AsyncMock(return_value={"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 111, "login": "alice"}})
    controller._core_for_node = AsyncMock()
    controller._deploy_component = AsyncMock()

    await controller.reconcile_pr("repo", 1)

    controller._core_for_node.assert_not_called()
    controller._deploy_component.assert_not_called()


def test_hidden_state_rejects_empty_review_evidence_for_deploy_requested():
    with pytest.raises(MalformedHiddenStateError, match="review_evidence"):
        _validate_hidden_state(_state(status="deploy-requested", review_evidence={}, head_sha="a" * 40))


def test_hidden_state_rejects_empty_platform_for_deploy_requested():
    with pytest.raises(MalformedHiddenStateError, match="resolved_platform"):
        _validate_hidden_state(_state(components=[_component(resolved_platform="")]))


def test_hidden_state_rejects_mutable_image_ref():
    with pytest.raises(MalformedHiddenStateError, match="image_ref"):
        _validate_hidden_state(_state(components=[_component(image_ref="registry/repo:latest")]))

def test_hidden_state_accepts_real_driver_registry_depth():
    ref = (
        "ccr.ccs.tencentyun.com/phanthy-motus/drivers/unitree/g1@sha256:"
        + "a" * 64
    )
    state = _state(components=[_component(
        target="driver",
        driver_path="unitree/g1",
        variant="",
        review_image_tag=(
            "ccr.ccs.tencentyun.com/phanthy-motus/drivers/unitree/g1:release.test"
        ),
        image_ref=ref,
        runtime_id="unitree-g1",
    )])
    validated = _validate_hidden_state(state)
    assert validated["components"][0]["image_ref"] == ref
    assert validated["components"][0]["target"] == "driver"
    assert validated["components"][0]["driver_path"] == "unitree/g1"


def test_hidden_state_rejects_duplicate_component_ids():
    comp = _component()
    with pytest.raises(MalformedHiddenStateError, match="duplicate"):
        _validate_hidden_state(_state(components=[comp, dict(comp)]))


def test_hidden_state_rejects_unknown_deployment_component_id():
    with pytest.raises(MalformedHiddenStateError, match="unknown component"):
        _validate_hidden_state(_state(deployments=[{"machine": "test-machine", "component_ids": ["missing"], "phase": "deployed"}]))


def test_hidden_state_rejects_component_deployed_on_two_machines():
    with pytest.raises(MalformedHiddenStateError, match="deployed more than once"):
        _validate_hidden_state(
            _state(
                deployments=[
                    {"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"},
                    {"machine": "driver-machine", "component_ids": ["comp-001"], "phase": "deployed"},
                ]
            )
        )


def test_hidden_state_rejects_invalid_case_result():
    with pytest.raises(MalformedHiddenStateError, match="case_results"):
        _validate_hidden_state(_state(case_results={"comp-001": "boom"}))


def test_hidden_state_rejects_cos_extra_secret_key():
    with pytest.raises(MalformedHiddenStateError, match="cos keys"):
        _validate_hidden_state(_state(cos={"object_key": "", "sha256": "", "size": 0, "secret": "x"}))


def test_hidden_state_rejects_signed_url():
    with pytest.raises(MalformedHiddenStateError, match="cos keys"):
        _validate_hidden_state(_state(cos={"object_key": "", "sha256": "", "size": 0, "signed_url": "https://example"}))


def test_hidden_state_rejects_invalid_command_phase():
    with pytest.raises(MalformedHiddenStateError, match="command.phase"):
        _validate_hidden_state(_state(command={"comment_id": 1, "kind": "approve_deploy", "phase": "bogus", "args": {"machine": "test-machine"}}))


@pytest.mark.asyncio
async def test_machine_missing_node_host_fails_closed(controller, policy):
    policy.machines["broken"] = MachineInfo(
        alias="broken",
        node_id="node-x",
        owners=["owner1"],
        node_host="",
        targets=["perception"],
        platforms=["linux/arm64"],
    )

    with pytest.raises(DeployControllerError):
        await controller._resolve_core_client("node-x")


def test_machine_missing_targets_fails_closed(controller, policy):
    policy.machines["broken"] = MachineInfo(
        alias="broken",
        node_id="node-x",
        owners=["owner1"],
        node_host="127.0.0.9",
        targets=[],
        platforms=["linux/arm64"],
    )
    assert controller._get_component_ids_for_machine("broken", [_component()]) == []


def test_machine_missing_platforms_fails_closed(controller, policy):
    policy.machines["broken"] = MachineInfo(
        alias="broken",
        node_id="node-x",
        owners=["owner1"],
        node_host="127.0.0.9",
        targets=["perception"],
        platforms=[],
    )
    assert controller._get_component_ids_for_machine("broken", [_component()]) == []


def test_machine_variant_required_fails_closed(controller, policy):
    policy.machines["broken"] = MachineInfo(
        alias="broken",
        node_id="node-x",
        owners=["owner1"],
        node_host="127.0.0.9",
        targets=["perception"],
        platforms=["linux/arm64"],
        variants=["jetson-jp5.11"],
    )
    assert controller._get_component_ids_for_machine("broken", [_component(variant="")]) == []


def test_driver_machine_missing_driver_path_mapping_fails_closed(controller, policy):
    policy.machines["broken"] = MachineInfo(
        alias="broken",
        node_id="node-x",
        owners=["owner1"],
        node_host="127.0.0.9",
        targets=["driver"],
        platforms=["linux/arm64"],
        driver_paths=[],
    )
    assert controller._get_component_ids_for_machine("broken", [_component(target="driver", driver_path="custom/driver")]) == []


def test_unrelated_machine_hidden(controller, policy):
    policy.machines["unrelated"] = MachineInfo(
        alias="unrelated",
        node_id="node-y",
        owners=["owner2"],
        node_host="127.0.0.8",
        targets=["driver"],
        platforms=["linux/arm64"],
    )

    aliases = {g["alias"] for g in controller._get_machine_groups_for_components([_component()])}

    assert "test-machine" in aliases
    assert "unrelated" not in aliases
