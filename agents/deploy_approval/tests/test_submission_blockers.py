"""Focused regressions for the deploy approval blocker round."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..config import Config
from ..commands import ParsedCommand, parse_command
from .. import service as service_mod
from ..github_command_watcher import GitHubCommandWatcher
from ..github_state_proxy import GitHubStateProxy, MalformedHiddenStateError, _validate_hidden_state
from ..models import HiddenState, MachineInfo
from ..policy import Policy
from ..service import DeployController
from .conftest import make_config


class FakeGitHub:
    def __init__(self, pr: dict | None = None):
        self.pr = pr or {
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "user": {"id": 1, "login": "alice"},
        }
        self.comments: dict[int, dict] = {}
        self.next_comment_id = 100
        self.posted: list[dict] = []
        self.updated: list[dict] = []

    async def get_pr(self, repo: str, pr_number: int) -> dict:
        return dict(self.pr)

    async def get_issue_comments(self, repo: str, pr_number: int) -> list[dict]:
        return [dict(c) for c in self.comments.values()]

    async def get_comment(self, repo: str, comment_id: int) -> dict:
        return dict(self.comments.get(comment_id, {}))

    async def post_issue_comment(self, repo: str, pr_number: int, body: str) -> dict:
        comment = {"id": self.next_comment_id, "body": body,
                   "user": {"id": 999, "login": "bot", "type": "Bot"},
                   "performed_via_github_app": {"id": 999}}
        self.comments[self.next_comment_id] = dict(comment)
        self.next_comment_id += 1
        self.posted.append(dict(comment))
        return comment

    async def update_comment(self, repo: str, comment_id: int, body: str) -> dict:
        comment = self.comments[comment_id]
        comment["body"] = body
        self.updated.append({"id": comment_id, "body": body})
        return dict(comment)

    async def collaborator_permission(self, repo: str, actor: str) -> str:
        return "admin"

    async def list_open_prs(self, repo: str) -> list[dict]:
        return [{"number": 1}]

    async def get_issue_labels(self, repo: str, issue_number: int) -> list[str]:
        return []

    async def set_issue_labels(self, repo: str, issue_number: int, labels: list[str]) -> None:
        return None


def _policy(config: Config) -> Policy:
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
            driver_paths=["unitree/go2"],
        )
    }
    return policy


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


def _review_state(**overrides):
    state = {
        "version": 1,
        "head_sha": "a" * 40,
        "status": "review-required",
        "review_evidence": {},
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
    }
    state.update(overrides)
    return state


def _deploy_requested_state(**overrides):
    state = _review_state(
        status="deploy-requested",
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[_component()],
        deployments=[],
    )
    state.update(overrides)
    return state


def _testing_state(**overrides):
    state = _review_state(
        status="testing",
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[_component()],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
    )
    state.update(overrides)
    return state


def _closed_pr(head: str = "a" * 40) -> dict:
    return {"state": "closed", "merged": False, "head": {"sha": head}, "user": {"id": 1, "login": "alice"}}


def _merged_pr(head: str = "a" * 40) -> dict:
    return {"state": "closed", "merged": True, "head": {"sha": head}, "user": {"id": 1, "login": "alice"}}


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def fake_github():
    return FakeGitHub()


@pytest.fixture
def proxy(config, fake_github):
    return GitHubStateProxy(config, fake_github, github_app_id="999")


@pytest.fixture
def policy(config):
    return _policy(config)


@pytest.fixture
def controller(config, proxy, policy, fake_github):
    registry = MagicMock()
    registry.resolve = AsyncMock()
    return DeployController(config, proxy, policy, fake_github, registry)


@pytest.mark.asyncio
async def test_reconcile_no_state_no_review_creates_review_required(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=None)
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    fake_github.pr = {"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 1, "login": "alice"}}
    proxy.get_issue_comments = AsyncMock(return_value=[])

    await controller.reconcile_pr("repo", 1)

    written = proxy.write_hidden_state.call_args.args[3]
    assert written["status"] == "review-required"
    assert written["review_evidence"] == {}
    proxy.project_status_label.assert_called_with("repo", 1, "review-required")


@pytest.mark.asyncio
async def test_request_deploy_requires_deploy_ready(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=_review_state(status="review-required"))
    proxy.project_status_label = AsyncMock()
    proxy.write_hidden_state = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("1", "alice"))
    fake_github.pr = {"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 1, "login": "alice"}}
    controller.registry.resolve = AsyncMock()

    await controller.handle_request_deploy("repo", 1, 11)

    proxy.write_hidden_state.assert_not_called()
    controller.registry.resolve.assert_not_called()


def test_parsed_command_has_no_legacy_build_index_field():
    assert "build_index" not in {f.name for f in fields(ParsedCommand)}
    assert parse_command("/request_deploy build=1").kind == "unknown"


def test_agent_core_client_has_no_registration_contract_residue():
    root = Path(__file__).resolve().parents[3]
    agent = (root / "agents/deploy_approval/agent_core_client.py").read_text(encoding="utf-8")
    deploy = (root / "deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    forbidden = (
        "registered node endpoint",
        "/api/nodes/register",
        "self-registers",
        "registered Agent Cores",
    )
    for phrase in forbidden:
        assert phrase not in agent
        assert phrase not in deploy


def test_deploy_script_has_no_agent_core_self_registration_contract():
    root = Path(__file__).resolve().parents[3]
    deploy = (root / "deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    assert "self-registers via POST /api/nodes/register" not in deploy
    assert "registered Agent Cores" not in deploy


def test_registry_digest_verifier_has_no_rollback_contract_semantics():
    root = Path(__file__).resolve().parents[3]
    registry = (root / "agents/deploy_approval/registry_client.py").read_text(encoding="utf-8")
    forbidden = (
        "historical rollback baseline",
        "rollback baseline",
        "deploy/rollback",
        "previous running_image",
    )
    for phrase in forbidden:
        assert phrase not in registry


def test_hidden_state_rejects_empty_command_phase():
    state = _review_state()
    state["command"]["phase"] = ""

    with pytest.raises(MalformedHiddenStateError):
        _validate_hidden_state(state)


def test_init_hidden_state_defaults_to_completed_phase():
    controller = object.__new__(DeployController)

    state = DeployController._init_hidden_state(
        controller,
        head_sha="a" * 40,
        status="review-required",
    )

    assert state["command"]["phase"] == "completed"


def test_hidden_state_model_defaults_to_completed_phase():
    state = HiddenState()

    assert state.command["phase"] == "completed"



@pytest.mark.asyncio
async def test_consume_cursor_without_state_does_not_bootstrap(controller, proxy):
    proxy.read_hidden_state = AsyncMock(return_value=None)
    proxy.get_pr = AsyncMock()
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()

    watcher = GitHubCommandWatcher(make_config(), proxy, controller)
    await watcher._consume_comment_cursor("repo", 1, 77)

    proxy.get_pr.assert_not_called()
    proxy.write_hidden_state.assert_not_called()
    proxy.project_status_label.assert_not_called()


@pytest.mark.asyncio
async def test_closed_pr_without_state_is_not_bootstrapped(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=None)
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.get_issue_comments = AsyncMock()
    fake_github.pr = _closed_pr()
    watcher = GitHubCommandWatcher(make_config(), proxy, controller)

    await watcher._process_pr("repo", 1)

    proxy.write_hidden_state.assert_not_called()
    proxy.project_status_label.assert_not_called()
    proxy.get_issue_comments.assert_not_called()


@pytest.mark.asyncio
async def test_record_test_closed_pr_has_zero_terminal_write_and_zero_cos(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=_testing_state())
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    controller._upload_evidence = AsyncMock()
    fake_github.pr = _closed_pr()

    await controller.handle_record_test("repo", 1, 99, "pass", "summary", "owner1", "1")

    proxy.write_hidden_state.assert_not_called()
    proxy.project_status_label.assert_not_called()
    controller._upload_evidence.assert_not_called()


@pytest.mark.asyncio
async def test_record_test_merged_pr_has_zero_terminal_write_and_zero_cos(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=_testing_state())
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    controller._upload_evidence = AsyncMock()
    fake_github.pr = _merged_pr()

    await controller.handle_record_test("repo", 1, 99, "pass", "summary", "owner1", "1")

    proxy.write_hidden_state.assert_not_called()
    proxy.project_status_label.assert_not_called()
    controller._upload_evidence.assert_not_called()


def test_no_command_bootstrap_helper_remains():
    assert not hasattr(DeployController, "bootstrap_missing_state")
    assert "bootstrap_missing_state" not in inspect.getsource(service_mod)


def test_deploy_ready_hidden_state_requires_review_evidence():
    with pytest.raises(MalformedHiddenStateError):
        _validate_hidden_state(_review_state(status="deploy-ready", review_evidence={}, components=[]))


def test_deploy_ready_hidden_state_allows_empty_components_with_review_job():
    state = _review_state(status="deploy-ready", review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"}, components=[])

    validated = _validate_hidden_state(state)

    assert validated["status"] == "deploy-ready"
    assert validated["review_evidence"].get("build_comment_id") == 1
    assert validated["components"] == []


@pytest.mark.asyncio
async def test_approve_rechecks_pr_after_clean_gate_before_executing(controller, proxy, fake_github):
    state = _deploy_requested_state()
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.write_hidden_state = AsyncMock(side_effect=lambda *args, **kwargs: events.append("write_hidden_state") or {})
    proxy.project_status_label = AsyncMock()
    events: list[str] = []
    first_pr = {"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 1, "login": "alice"}}
    second_pr = {"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 1, "login": "alice"}}

    get_pr_calls = 0

    async def _get_pr(repo: str, pr_number: int) -> dict:
        nonlocal get_pr_calls
        get_pr_calls += 1
        events.append(f"get_pr_{get_pr_calls}")
        return first_pr if get_pr_calls == 1 else second_pr

    proxy.get_pr = AsyncMock(side_effect=_get_pr)
    proxy.comment_identity = AsyncMock(return_value=("1", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")

    core = MagicMock()
    core.list_drivers = AsyncMock(side_effect=lambda: events.append("list_drivers") or [{"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    core.driver_status = AsyncMock(side_effect=lambda runtime_id: events.append(f"driver_status:{runtime_id}") or {"status": "running", "running_image": ""})
    controller._core_for_node = AsyncMock(return_value=core)
    controller._deploy_component = AsyncMock(side_effect=lambda *args, **kwargs: events.append("deploy_post") or {})
    controller._run_automated_case = AsyncMock(return_value={})
    fake_github.comments[50] = {
        "id": 50,
        "body": "/approve_deploy machine=test-machine",
        "user": {"id": 1, "login": "owner1"},
    }

    await controller.handle_approve_deploy("repo", 1, 50, "test-machine", "owner1", "1")

    assert proxy.get_pr.await_count >= 2
    assert events.index("list_drivers") < events.index("get_pr_2")
    assert events.index("driver_status:perception") < events.index("get_pr_2")
    assert events.index("get_pr_2") < events.index("write_hidden_state") < events.index("deploy_post")
    assert proxy.write_hidden_state.call_count >= 1


@pytest.mark.asyncio
async def test_approve_head_changes_after_clean_gate_zero_deploy_post(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=_deploy_requested_state())
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("1", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.get_pr = AsyncMock(side_effect=[
        {"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 1, "login": "alice"}},
        {"state": "open", "merged": False, "head": {"sha": "b" * 40}, "user": {"id": 1, "login": "alice"}},
    ])
    core = MagicMock()
    core.list_drivers = AsyncMock(return_value=[{"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    core.driver_status = AsyncMock(return_value={"status": "running", "running_image": ""})
    controller._core_for_node = AsyncMock(return_value=core)
    controller._deploy_component = AsyncMock()
    controller._run_automated_case = AsyncMock(return_value={})
    fake_github.comments[51] = {
        "id": 51,
        "body": "/approve_deploy machine=test-machine",
        "user": {"id": 1, "login": "owner1"},
    }

    await controller.handle_approve_deploy("repo", 1, 51, "test-machine", "owner1", "1")

    controller._deploy_component.assert_not_called()
    written = proxy.write_hidden_state.call_args.args[3]
    assert written["status"] == "review-required"


@pytest.mark.asyncio
async def test_approve_pr_closes_after_clean_gate_zero_deploy_post(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=_deploy_requested_state())
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.comment_identity = AsyncMock(return_value=("1", "alice"))
    proxy.collaborator_permission = AsyncMock(return_value="admin")
    proxy.get_pr = AsyncMock(side_effect=[
        {"state": "open", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 1, "login": "alice"}},
        {"state": "closed", "merged": False, "head": {"sha": "a" * 40}, "user": {"id": 1, "login": "alice"}},
    ])
    core = MagicMock()
    core.list_drivers = AsyncMock(return_value=[{"id": "perception", "category": "driver", "image": "registry/repo:latest"}])
    core.driver_status = AsyncMock(return_value={"status": "running", "running_image": ""})
    controller._core_for_node = AsyncMock(return_value=core)
    controller._deploy_component = AsyncMock()
    controller._run_automated_case = AsyncMock(return_value={})
    fake_github.comments[52] = {
        "id": 52,
        "body": "/approve_deploy machine=test-machine",
        "user": {"id": 1, "login": "owner1"},
    }

    await controller.handle_approve_deploy("repo", 1, 52, "test-machine", "owner1", "1")

    controller._deploy_component.assert_not_called()
    written = proxy.write_hidden_state.call_args.args[3]
    assert written["status"] == "review-required"
    proxy.project_status_label.assert_called_with("repo", 1, "review-required")


@pytest.mark.asyncio
async def test_reconcile_succeeded_old_head_becomes_review_required_on_new_head(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=_review_state(status="succeeded", head_sha="a" * 40, review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"}, components=[_component()], deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}], test_result="pass", cos={"object_key": "k", "sha256": "b" * 64, "size": 1}))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    fake_github.pr = {"state": "open", "merged": False, "head": {"sha": "b" * 40}, "user": {"id": 1, "login": "alice"}}

    await controller.reconcile_pr("repo", 1)

    written = proxy.write_hidden_state.call_args.args[3]
    assert written["status"] == "review-required"
    assert written["head_sha"] == "b" * 40
    assert written["components"] == []
    assert written["deployments"] == []
    assert written["cos"] == {"object_key": "", "sha256": "", "size": 0}


@pytest.mark.asyncio
async def test_reconcile_testing_old_head_clears_deployment_snapshot(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=_testing_state(head_sha="a" * 40, components=[_component()], deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}], case_results={"comp-001": "pass"}, cos={"object_key": "k", "sha256": "b" * 64, "size": 1}))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    fake_github.pr = {"state": "open", "merged": False, "head": {"sha": "b" * 40}, "user": {"id": 1, "login": "alice"}}

    await controller.reconcile_pr("repo", 1)

    written = proxy.write_hidden_state.call_args.args[3]
    assert written["status"] == "review-required"
    assert written["components"] == []
    assert written["deployments"] == []
    assert written["case_results"] == {}


@pytest.mark.asyncio
async def test_reconcile_deploy_requested_old_head_clears_validation_snapshot(controller, proxy, fake_github):
    proxy.read_hidden_state = AsyncMock(return_value=_deploy_requested_state(head_sha="a" * 40, components=[_component()], deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}], approve_attempts=[{"comment_id": 1, "actor": "owner1", "machine": "test-machine", "preflight": [], "outcome": "deployed", "health": []}], approve_attempts_total=1, approve_attempts_truncated=False, case_results={"comp-001": "pass"}, test_result="", cos={"object_key": "k", "sha256": "b" * 64, "size": 1}))
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    fake_github.pr = {"state": "open", "merged": False, "head": {"sha": "b" * 40}, "user": {"id": 1, "login": "alice"}}

    await controller.reconcile_pr("repo", 1)

    written = proxy.write_hidden_state.call_args.args[3]
    assert written["status"] == "review-required"
    assert written["components"] == []
    assert written["deployments"] == []
    assert written["approve_attempts"] == []
    assert written["approve_attempts_total"] == 0


@pytest.mark.asyncio
async def test_record_test_cos_rebind_uses_fresh_terminal_state(controller, proxy):
    state = _review_state(
        status="succeeded",
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[_component()],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
        test_result="pass",
        command={"comment_id": 77, "kind": "record_test", "phase": "completed", "args": {}},
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.write_hidden_state = AsyncMock()

    result = await controller._rebind_terminal_cos_if_current(
        "repo",
        1,
        expected_head="a" * 40,
        expected_terminal_status="succeeded",
        expected_comment_id=77,
        expected_command_kind="record_test",
        expected_test_result="pass",
        cos_metadata={"object_key": "obj", "sha256": "c" * 64, "size": 12},
        markdown="terminal",
    )

    assert result is True
    written = proxy.write_hidden_state.call_args.args[3]
    assert written["cos"] == {"object_key": "obj", "sha256": "c" * 64, "size": 12}


@pytest.mark.asyncio
async def test_record_test_cos_rebind_skips_after_head_drift(controller, proxy):
    state = _review_state(
        head_sha="b" * 40,
        status="succeeded",
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[_component()],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
        test_result="pass",
        command={"comment_id": 77, "kind": "record_test", "phase": "completed", "args": {}},
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.write_hidden_state = AsyncMock()

    result = await controller._rebind_terminal_cos_if_current(
        "repo",
        1,
        expected_head="a" * 40,
        expected_terminal_status="succeeded",
        expected_comment_id=77,
        expected_command_kind="record_test",
        expected_test_result="pass",
        cos_metadata={"object_key": "obj", "sha256": "c" * 64, "size": 12},
        markdown="terminal",
    )

    assert result is False
    proxy.write_hidden_state.assert_not_called()


@pytest.mark.asyncio
async def test_record_test_cos_rebind_skips_after_command_changed(controller, proxy):
    state = _review_state(
        status="succeeded",
        review_evidence={"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        components=[_component()],
        deployments=[{"machine": "test-machine", "component_ids": ["comp-001"], "phase": "deployed"}],
        test_result="pass",
        command={"comment_id": 78, "kind": "approve_deploy", "phase": "completed", "args": {}},
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.write_hidden_state = AsyncMock()

    result = await controller._rebind_terminal_cos_if_current(
        "repo",
        1,
        expected_head="a" * 40,
        expected_terminal_status="succeeded",
        expected_comment_id=77,
        expected_command_kind="record_test",
        expected_test_result="pass",
        cos_metadata={"object_key": "obj", "sha256": "c" * 64, "size": 12},
        markdown="terminal",
    )

    assert result is False
    proxy.write_hidden_state.assert_not_called()


@pytest.mark.asyncio
async def test_failed_deploy_cos_rebind_skips_after_lifecycle_changed(controller, proxy):
    state = _review_state(
        status="review-required",
        review_evidence={},
        components=[],
        deployments=[],
        command={"comment_id": 88, "kind": "approve_deploy", "phase": "completed", "args": {}},
    )
    proxy.read_hidden_state = AsyncMock(return_value=state)
    proxy.write_hidden_state = AsyncMock()

    result = await controller._rebind_terminal_cos_if_current(
        "repo",
        1,
        expected_head="a" * 40,
        expected_terminal_status="failed",
        expected_comment_id=88,
        expected_command_kind="approve_deploy",
        cos_metadata={"object_key": "obj", "sha256": "c" * 64, "size": 12},
        markdown="terminal",
    )

    assert result is False
    proxy.write_hidden_state.assert_not_called()


def test_sh_go2_example_uses_unitree_go2():
    repo_root = Path(__file__).resolve().parents[3]
    text = (repo_root / "deploy/deploy-approval/machines.example.yaml").read_text(encoding="utf-8")
    assert "unitree/go2" in text
    assert "unitree/g1" not in text.split("sh-go2:", 1)[1]


def test_node_id_docs_do_not_require_agent_core_registration():
    repo_root = Path(__file__).resolve().parents[3]
    text = (repo_root / "DEPLOY_APPROVAL_AGENT.md").read_text(encoding="utf-8")
    assert "Agent Core's registered node_id" not in text
    assert "Deploy Approval machine-policy internal unique machine identifier" in text


# ── Blocker 2: Repo->Target Source Matrix regression tests ──────────────────

from dataclasses import dataclass


@dataclass
class FakeBuild:
    target: str
    success: bool = True
    image_tag: str = "registry.example/repo:v1"
    driver_path: str = ""


def _is_deployable_build(repo, build):
    """Import the production function and expose it for testing."""
    from ..service import _is_deployable_build as _pdb
    return _pdb(repo, build)


class TestSourceMatrixOfficialCore:
    """Official 4paradigm/phanthymotus: only perception and actucore are deployable."""

    def test_official_phanthymotus_perception_is_deployable(self):
        build = FakeBuild(target="perception")
        assert _is_deployable_build("4paradigm/phanthymotus", build) is True

    def test_official_phanthymotus_actucore_is_deployable(self):
        build = FakeBuild(target="actucore")
        assert _is_deployable_build("4paradigm/phanthymotus", build) is True

    def test_official_phanthymotus_driver_is_not_deployable(self):
        build = FakeBuild(target="driver")
        assert _is_deployable_build("4paradigm/phanthymotus", build) is False

    def test_official_phanthymotus_unknown_target_is_not_deployable(self):
        build = FakeBuild(target="telemetry")
        assert _is_deployable_build("4paradigm/phanthymotus", build) is False


class TestSourceMatrixForkCore:
    """Fork haohao-end/phanthymotus: only perception and actucore are deployable."""

    def test_fork_phanthymotus_perception_is_deployable(self):
        build = FakeBuild(target="perception")
        assert _is_deployable_build("haohao-end/phanthymotus", build) is True

    def test_fork_phanthymotus_actucore_is_deployable(self):
        build = FakeBuild(target="actucore")
        assert _is_deployable_build("haohao-end/phanthymotus", build) is True

    def test_fork_phanthymotus_driver_is_not_deployable(self):
        build = FakeBuild(target="driver")
        assert _is_deployable_build("haohao-end/phanthymotus", build) is False


class TestSourceMatrixDriverRepos:
    """Driver repos: only target=driver with non-empty driver_path is deployable."""

    def test_official_driver_repo_driver_with_path_is_deployable(self):
        build = FakeBuild(target="driver", driver_path="unitree/go2")
        assert _is_deployable_build("4paradigm/phanthymotus-driver", build) is True

    def test_official_driver_repo_driver_without_path_is_not_deployable(self):
        build = FakeBuild(target="driver", driver_path="")
        assert _is_deployable_build("4paradigm/phanthymotus-driver", build) is False

    def test_official_driver_repo_perception_is_not_deployable(self):
        build = FakeBuild(target="perception")
        assert _is_deployable_build("4paradigm/phanthymotus-driver", build) is False

    def test_fork_driver_repo_driver_with_path_is_deployable(self):
        build = FakeBuild(target="driver", driver_path="custom/driver")
        assert _is_deployable_build("haohao-end/phanthymotus-driver", build) is True

    def test_fork_driver_repo_driver_without_path_is_not_deployable(self):
        build = FakeBuild(target="driver", driver_path="  ")
        assert _is_deployable_build("haohao-end/phanthymotus-driver", build) is False


class TestSourceMatrixUnknownRepos:
    """Unknown/third-party repos: ZERO deployable builds."""

    def test_unknown_repo_perception_is_not_deployable(self):
        build = FakeBuild(target="perception")
        assert _is_deployable_build("some-user/phanthymotus", build) is False

    def test_unknown_repo_driver_is_not_deployable(self):
        build = FakeBuild(target="driver", driver_path="unitree/go2")
        assert _is_deployable_build("some-user/phanthymotus-driver", build) is False

    def test_random_repo_is_not_deployable(self):
        build = FakeBuild(target="perception")
        assert _is_deployable_build("org/not-related", build) is False


class TestSourceMatrixFailedBuilds:
    """Failed builds or missing image_tag are never deployable regardless of repo."""

    def test_failed_build_is_not_deployable(self):
        build = FakeBuild(target="perception", success=False)
        assert _is_deployable_build("4paradigm/phanthymotus", build) is False

    def test_empty_image_tag_is_not_deployable(self):
        build = FakeBuild(target="perception", image_tag="")
        assert _is_deployable_build("4paradigm/phanthymotus", build) is False

    def test_driver_repo_failed_build_is_not_deployable(self):
        build = FakeBuild(target="driver", driver_path="go2", success=False)
        assert _is_deployable_build("4paradigm/phanthymotus-driver", build) is False
