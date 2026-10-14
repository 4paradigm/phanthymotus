"""Offline business regressions for PR #314 Cases and final acceptance.

Agent Core, GitHub and COS I/O are mocked. Case selection/execution, final
acceptance and lifecycle writing use the real production business functions.
These tests never constitute real machine JP5.11/JP6.1 acceptance evidence.
"""

from __future__ import annotations

import copy

import pytest
from unittest.mock import AsyncMock, MagicMock

from ..config import Config
from ..case_runner import CaseRunner
from ..policy import Policy
from ..models import MachineInfo
from ..service import DeployController
from .conftest import make_config


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def policy(config):
    p = Policy(config)
    p.machines = {
        "test-machine": MachineInfo(
            alias="test-machine",
            node_id="node-1",
            owners=["owner1"],
            node_host="127.0.0.1",
            targets=["perception", "actucore", "core"],
            platforms=["linux/arm64"],
            variants=["5.11"],
        ),
        "jp61-machine": MachineInfo(
            alias="jp61-machine", node_id="node-2", owners=["owner1"],
            node_host="127.0.0.2", targets=["perception", "actucore", "core"],
            platforms=["linux/arm64"], variants=["6.1"],
        ),
    }
    return p


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
    from ..github_state_proxy import GitHubStateProxy
    return GitHubStateProxy(config, mock_github, github_app_id="12345")


@pytest.fixture
def controller(config, proxy, policy, mock_github):
    return DeployController(config, proxy, policy, mock_github)


# ── Helpers ───────────────────────────────────────────────────────────

def _core_component(**overrides):
    """Build a 'core' target component dict."""
    comp = {
        "component_id": "comp-core",
        "target": "core",
        "driver_path": "",
        "variant": "",
        "review_image_tag": "registry/repo:v1",
        "image_ref": "registry/repo@sha256:" + "c" * 64,
        "resolved_platform": "linux/arm64",
        "runtime_id": "core",
    }
    comp.update(overrides)
    return comp


def _perception_component(**overrides):
    """Build a perception JP5.11 component dict."""
    comp = {
        "component_id": "comp-perc",
        "target": "perception",
        "driver_path": "",
        "variant": "5.11",
        "review_image_tag": "registry/repo:v1",
        "image_ref": "registry/repo@sha256:" + "p" * 64,
        "resolved_platform": "linux/arm64",
        "runtime_id": "perception-driver",
    }
    comp.update(overrides)
    return comp


def _actucore_component(**overrides):
    """Build an actucore JP5.11 component dict."""
    comp = {
        "component_id": "comp-actucore",
        "target": "actucore",
        "driver_path": "",
        "variant": "5.11",
        "review_image_tag": "registry/repo:v1",
        "image_ref": "registry/repo@sha256:" + "u" * 64,
        "resolved_platform": "linux/arm64",
        "runtime_id": "actucore-driver",
    }
    comp.update(overrides)
    return comp


# ── Test 1: Core no-case skips core resolution ────────────────────────

@pytest.mark.asyncio
async def test_pr314_core_no_case_skips_core_resolution(controller):
    """Core has no fixed case → select_case returns None → n/a without calling _core_for_node.

    Even if _core_for_node would raise, Core must return n/a because
    Core has no fixed-case registry entry.
    """
    controller._core_for_node = AsyncMock(side_effect=ConnectionError("core unreachable"))

    core_comp = _core_component()
    deployments = [{"machine": "test-machine", "component_ids": ["comp-core"], "phase": "deployed"}]

    result = await controller._run_automated_case(
        "4paradigm/phanthymotus",
        1,
        "a" * 40,
        [core_comp],
        deployments,
    )

    assert result == {"comp-core": "n/a"}
    controller._core_for_node.assert_not_called()


# ── Test 2: Core missing deployment still fails ──────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [None, "failed", "deploying"])
async def test_pr314_core_missing_deployment_still_fails(controller, phase):
    """Core with no deployment record → fail, not n/a.

    Missing deployment is a genuine failure regardless of case registry.
    """
    core_comp = _core_component()
    # Empty deployments list — no record for comp-core
    deployments = [] if phase is None else [{
        "machine": "test-machine", "component_ids": ["comp-core"], "phase": phase,
    }]

    result = await controller._run_automated_case(
        "4paradigm/phanthymotus",
        1,
        "a" * 40,
        [core_comp],
        deployments,
    )

    assert result == {"comp-core": "fail"}


# ── Test 3: Case selector exception fails closed ─────────────────────

@pytest.mark.asyncio
async def test_pr314_case_selector_exception_fails_closed(controller):
    """If select_case raises, the component must fail — not silently skip."""
    perception_comp = _perception_component()
    deployments = [{"machine": "test-machine", "component_ids": ["comp-perc"], "phase": "deployed"}]

    runner = MagicMock()
    runner.select_case = MagicMock(side_effect=ValueError("registry corrupted"))
    controller._get_case_runner = MagicMock(return_value=runner)

    result = await controller._run_automated_case(
        "4paradigm/phanthymotus",
        1,
        "a" * 40,
        [perception_comp],
        deployments,
    )

    assert result == {"comp-perc": "fail"}


# ── Test 4: Perception JP5.11 health contract pass ───────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["5.11", "6.1"])
async def test_pr314_perception_health_contract_pass(controller, variant):
    """Full offline Perception health chain: running + image + runtime_id pinned +
    driver found + MCP bound + mcp_ping online → pass.
    """
    image_ref = "registry/repo@sha256:" + "p" * 64
    perc_comp = _perception_component(image_ref=image_ref, variant=variant)
    alias = "test-machine" if variant == "5.11" else "jp61-machine"
    deployments = [{"machine": alias, "component_ids": ["comp-perc"], "phase": "deployed"}]

    # Simulate core client responses
    core = AsyncMock()
    core.driver_status = AsyncMock(return_value={
        "status": "running",
        "running_image": image_ref,
    })
    core.list_drivers = AsyncMock(return_value=[
        {
            "id": "perception-driver",
            "target": "perception",
            "mcp_id": "mcp-perception-1",
        }
    ])
    core.list_mcp = AsyncMock(return_value=[
        {
            "id": "mcp-perception-1",
            "url": "http://mcp/perception",
            "status": "online",
        }
    ])
    core.mcp_ping = AsyncMock(return_value={"online": True})
    controller._core_for_node = AsyncMock(return_value=core)

    result = await controller._run_automated_case(
        "4paradigm/phanthymotus",
        1,
        "a" * 40,
        [perc_comp],
        deployments,
    )

    assert result == {"comp-perc": "pass"}
    # Verify every step was called in order
    controller._core_for_node.assert_awaited_once_with("node-1" if variant == "5.11" else "node-2")
    core.driver_status.assert_called_once_with("perception-driver")
    core.list_drivers.assert_called_once()
    core.list_mcp.assert_called_once()
    core.mcp_ping.assert_called_once_with("mcp-perception-1")


# ── Test 5: ActuCore JP5.11 health contract pass ─────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["5.11", "6.1"])
async def test_pr314_actucore_health_contract_pass(controller, variant):
    """Offline ActuCore JP5.11/JP6.1 full health chain passes."""
    image_ref = "registry/repo@sha256:" + "u" * 64
    actucore_comp = _actucore_component(image_ref=image_ref, variant=variant)
    alias = "test-machine" if variant == "5.11" else "jp61-machine"
    deployments = [{"machine": alias, "component_ids": ["comp-actucore"], "phase": "deployed"}]

    core = AsyncMock()
    core.driver_status = AsyncMock(return_value={
        "status": "running",
        "running_image": image_ref,
    })
    core.list_drivers = AsyncMock(return_value=[
        {
            "id": "actucore-driver",
            "target": "actucore",
            "mcp_id": "mcp-actucore-1",
        }
    ])
    core.list_mcp = AsyncMock(return_value=[
        {
            "id": "mcp-actucore-1",
            "url": "http://mcp/actucore",
            "status": "online",
        }
    ])
    core.mcp_ping = AsyncMock(return_value={"online": True})
    controller._core_for_node = AsyncMock(return_value=core)

    result = await controller._run_automated_case(
        "4paradigm/phanthymotus",
        1,
        "a" * 40,
        [actucore_comp],
        deployments,
    )

    assert result == {"comp-actucore": "pass"}
    controller._core_for_node.assert_awaited_once_with("node-1" if variant == "5.11" else "node-2")
    core.driver_status.assert_called_once_with("actucore-driver")
    core.list_drivers.assert_called_once()
    core.list_mcp.assert_called_once()
    core.mcp_ping.assert_called_once_with("mcp-actucore-1")


# ── Test 6: JP5.11 wrong image fails closed ──────────────────────────

@pytest.mark.asyncio
async def test_pr314_jp511_wrong_image_fails_closed(controller):
    """running_image ≠ image_ref → fail. Must not pass even if MCP is online."""
    image_ref = "registry/repo@sha256:" + "p" * 64
    perc_comp = _perception_component(image_ref=image_ref)
    deployments = [{"machine": "test-machine", "component_ids": ["comp-perc"], "phase": "deployed"}]

    core = AsyncMock()
    # Intentionally mismatched running_image
    core.driver_status = AsyncMock(return_value={
        "status": "running",
        "running_image": "registry/other@sha256:" + "w" * 64,
    })
    core.list_drivers = AsyncMock(return_value=[
        {"id": "perception-driver", "target": "perception", "mcp_id": "mcp-1"}
    ])
    core.list_mcp = AsyncMock(return_value=[
        {"id": "mcp-1", "url": "http://mcp/1", "status": "online"}
    ])
    # mcp_ping must NOT be called because image check fails first
    controller._core_for_node = AsyncMock(return_value=core)

    result = await controller._run_automated_case(
        "4paradigm/phanthymotus",
        1,
        "a" * 40,
        [perc_comp],
        deployments,
    )

    assert result == {"comp-perc": "fail"}
    core.mcp_ping.assert_not_called()


# ── Test 7: JP5.11 MCP offline fails closed ──────────────────────────

@pytest.mark.asyncio
async def test_pr314_jp511_mcp_offline_fails_closed(controller):
    """MCP ping returns online=False → fail, even if image matches."""
    image_ref = "registry/repo@sha256:" + "p" * 64
    perc_comp = _perception_component(image_ref=image_ref)
    deployments = [{"machine": "test-machine", "component_ids": ["comp-perc"], "phase": "deployed"}]

    core = AsyncMock()
    core.driver_status = AsyncMock(return_value={
        "status": "running",
        "running_image": image_ref,
    })
    core.list_drivers = AsyncMock(return_value=[
        {"id": "perception-driver", "target": "perception", "mcp_id": "mcp-perception-1"}
    ])
    core.list_mcp = AsyncMock(return_value=[
        {"id": "mcp-perception-1", "url": "http://mcp/perception", "status": "offline"}
    ])
    core.mcp_ping = AsyncMock(return_value={"online": False})
    controller._core_for_node = AsyncMock(return_value=core)

    result = await controller._run_automated_case(
        "4paradigm/phanthymotus",
        1,
        "a" * 40,
        [perc_comp],
        deployments,
    )

    assert result == {"comp-perc": "fail"}
    # Verify the full chain was executed up to MCP ping
    core.driver_status.assert_called_once()
    core.list_drivers.assert_called_once()
    core.list_mcp.assert_called_once()
    core.mcp_ping.assert_called_once_with("mcp-perception-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_fields",
    [
        {"status": "stopped"}, {"status": "failed"}, {"status": "deploying"},
        {"status": "unknown"}, {}, {"status": None}, {"status": True},
        {"status": 1}, {"status": []}, {"status": {}}, {"status": "Running"},
    ],
    ids=["stopped", "failed", "deploying", "unknown", "missing", "null",
         "boolean", "integer", "list", "mapping", "wrong-case"],
)
async def test_stopped_runtime_rejects_automated_case(controller, status_fields):
    component = _perception_component(image_ref="registry.example/perception:jp5.11")
    core = AsyncMock()
    core.driver_status.return_value = {
        "running_image": component["image_ref"], **status_fields,
    }
    core.list_drivers.return_value = [
        {"id": component["runtime_id"], "mcp_id": "bound-mcp"},
    ]
    core.list_mcp.return_value = [{"id": "bound-mcp"}]
    core.mcp_ping.return_value = {"online": True}
    controller._core_for_node = AsyncMock(return_value=core)

    result = await controller._run_automated_case(
        "4paradigm/phanthymotus", 314, "a" * 40, [component],
        [{"machine": "test-machine", "component_ids": [component["component_id"]],
          "phase": "deployed"}],
    )

    assert result == {component["component_id"]: "fail"}
    core.driver_status.assert_awaited_once_with(component["runtime_id"])
    core.list_drivers.assert_not_awaited()
    core.list_mcp.assert_not_awaited()
    core.mcp_ping.assert_not_awaited()


def _final_testing_state():
    components = []
    deployments = []
    for cid, target, variant, alias in [
        ("p511", "perception", "5.11", "test-machine"),
        ("p61", "perception", "6.1", "jp61-machine"),
        ("a511", "actucore", "5.11", "test-machine"),
        ("a61", "actucore", "6.1", "jp61-machine"),
        ("core", "core", "", "test-machine"),
    ]:
        image = f"registry.example/{target}:jp{variant or 'core'}"
        components.append({
            "component_id": cid, "target": target, "variant": variant,
            "driver_path": "", "review_image_tag": image, "image_ref": image,
            "resolved_platform": "linux/arm64", "runtime_id": target,
        })
        deployments.append({
            "machine": alias, "component_ids": [cid], "phase": "deployed",
        })
    return {
        "version": 1, "head_sha": "a" * 40, "status": "testing",
        "review_evidence": {}, "components": components, "deployments": deployments,
        "case_results": {c["component_id"]: "n/a" if c["target"] == "core" else "pass"
                         for c in components},
        "test_result": "", "cos": {"object_key": "", "sha256": "", "size": 0},
        "approve_attempts": [], "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "command": {"comment_id": 100, "kind": "approve_deploy",
                    "phase": "completed", "args": {}},
        "last_processed_comment_id": 100,
    }


@pytest.fixture
def final_acceptance(controller, proxy):
    """Fake external storage only; leave production acceptance/history code intact."""
    store = {"state": _final_testing_state(), "visible": "", "writes": []}

    async def read_state(*args):
        return copy.deepcopy(store["state"])

    async def write_state(repo, pr, visible, state):
        store["state"] = copy.deepcopy(state)
        store["visible"] = visible
        store["writes"].append(copy.deepcopy(state))
        return {"id": 42}

    async def find_comment(*args):
        if not store["visible"]:
            return None
        from ..github_state_proxy import _build_hidden_state_body
        return {"id": 42, "body": _build_hidden_state_body(store["visible"], store["state"])}

    proxy.read_hidden_state = AsyncMock(side_effect=read_state)
    proxy.write_hidden_state = AsyncMock(side_effect=write_state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(side_effect=find_comment)
    proxy.get_issue_comments = AsyncMock(return_value=[])
    proxy.project_status_label = AsyncMock()
    proxy.persist_cursor = AsyncMock()
    proxy.get_pr = AsyncMock(return_value={
        "state": "open", "merged": False, "head": {"sha": "a" * 40},
    })
    controller._snapshot_terminal_runtime_logs = AsyncMock(return_value={})
    controller._upload_evidence = AsyncMock(return_value={
        "object_key": "phanthymotus_pr/phanthymotus/2026-10/2026-10-13/pr-314/evidence-"
                      + "a" * 40 + ".log.gz",
        "sha256": "b" * 64, "size": 100,
    })
    controller.cos.generate_evidence_download_url = MagicMock(
        return_value="https://offline.invalid/evidence",
    )
    controller._core_for_node = AsyncMock(side_effect=AssertionError("unexpected Agent Core access"))
    controller._deploy_component = AsyncMock(side_effect=AssertionError("unexpected deploy POST"))
    controller._deploy_core_component = AsyncMock(side_effect=AssertionError("unexpected Core POST"))
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "component_id, case_result",
    [
        ("p511", "fail"), ("p61", "fail"), ("a511", "fail"), ("a61", "fail"),
        ("driver", "fail"),
        ("p511", "n/a"), ("p61", "n/a"), ("a511", "n/a"), ("a61", "n/a"),
        ("driver", "n/a"), ("p511", None), ("p511", "running"),
        ("p511", "invalid"), ("p511", True), ("p511", {}),
    ],
)
async def test_record_test_pass_rejects_failed_case(
    controller, proxy, final_acceptance, component_id, case_result,
):
    state = final_acceptance["state"]
    if component_id == "driver":
        state["components"].append({
            "component_id": "driver", "target": "driver", "variant": "",
            "driver_path": "unitree/g1", "review_image_tag": "registry.example/driver:v1",
            "image_ref": "registry.example/driver:v1", "resolved_platform": "linux/arm64",
            "runtime_id": "unitree-g1",
        })
        state["deployments"].append({
            "machine": "test-machine", "component_ids": ["driver"], "phase": "deployed",
        })
    if case_result is None:
        state["case_results"].pop(component_id)
    else:
        state["case_results"][component_id] = case_result
    original = copy.deepcopy(state)

    assert await controller.handle_record_test(
        "4paradigm/phanthymotus", 314, 101, "pass", "offline", "owner1", "111",
    ) is True

    assert final_acceptance["state"] == original
    assert final_acceptance["writes"] == []
    controller._upload_evidence.assert_not_awaited()
    proxy.project_status_label.assert_not_awaited()
    controller._deploy_component.assert_not_awaited()
    controller._deploy_core_component.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase, core_case",
    [(None, "n/a"), ("deploying", "n/a"), ("failed", "n/a"),
     ("deployed", "pass"), ("deployed", "fail")],
)
async def test_record_test_pass_requires_core_deployment_and_na(
    controller, final_acceptance, phase, core_case,
):
    state = final_acceptance["state"]
    core_record = state["deployments"][-1]
    if phase is None:
        state["deployments"].remove(core_record)
    else:
        core_record["phase"] = phase
    state["case_results"]["core"] = core_case
    original = copy.deepcopy(state)

    await controller.handle_record_test(
        "4paradigm/phanthymotus", 314, 101, "pass", "offline", "owner1", "111",
    )

    assert final_acceptance["state"] == original
    assert final_acceptance["writes"] == []
    controller._upload_evidence.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_test_pass_accepts_only_deployed_passing_cases_once(
    controller, proxy, final_acceptance,
):
    original = copy.deepcopy(final_acceptance["state"])
    for _ in range(2):
        assert await controller.handle_record_test(
            "4paradigm/phanthymotus", 314, 101, "pass", "offline", "owner1", "111",
        ) is True

    state = final_acceptance["state"]
    assert state["status"] == "succeeded"
    assert state["test_result"] == "pass"
    for key in ("components", "deployments", "case_results"):
        assert state[key] == original[key]
    assert state["case_results"]["core"] == "n/a"
    assert all(value == "pass" for cid, value in state["case_results"].items() if cid != "core")
    assert state["cos"]["object_key"]
    assert "Test recorded" in final_acceptance["visible"]
    assert "\x60testing\x60" in final_acceptance["visible"]
    assert "\x60succeeded\x60" in final_acceptance["visible"]
    controller._upload_evidence.assert_awaited_once()
    controller._snapshot_terminal_runtime_logs.assert_awaited_once()
    controller._core_for_node.assert_not_awaited()
    controller._deploy_component.assert_not_awaited()
    controller._deploy_core_component.assert_not_awaited()
    proxy.persist_cursor.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage, change",
    [("logs", "case-result"), ("upload", "case-result"), ("history", "case-result"),
     ("logs", "runtime-id"), ("logs", "image"), ("logs", "machine"),
     ("upload", "core-deployment")],
)
async def test_record_test_pass_preserves_concurrent_hidden_state_changes(
    controller, proxy, final_acceptance, stage, change,
):
    metadata = copy.deepcopy(controller._upload_evidence.return_value)

    async def concurrent_change(*args, **kwargs):
        state = final_acceptance["state"]
        if change == "case-result":
            state["case_results"]["p511"] = "fail"
        elif change == "runtime-id":
            state["components"][0]["runtime_id"] = "new-runtime"
        elif change == "image":
            state["components"][0]["image_ref"] = "registry.example/perception:new-tag"
        elif change == "machine":
            state["deployments"][0]["machine"] = "jp61-machine"
        else:
            state["deployments"].pop()
        return metadata if stage == "upload" else None if stage == "history" else {}

    if stage == "logs":
        controller._snapshot_terminal_runtime_logs.side_effect = concurrent_change
    elif stage == "upload":
        controller._upload_evidence.side_effect = concurrent_change
    else:
        proxy.find_trusted_lifecycle_comment.side_effect = concurrent_change

    assert await controller.handle_record_test(
        "4paradigm/phanthymotus", 314, 101, "pass", "offline", "owner1", "111",
    ) is False

    state = final_acceptance["state"]
    assert state["status"] == "testing"
    assert state["cos"] == {"object_key": "", "sha256": "", "size": 0}
    assert final_acceptance["writes"] == []
    assert final_acceptance["visible"] == ""
    if change == "case-result":
        assert state["case_results"]["p511"] == "fail"
    elif change == "runtime-id":
        assert state["components"][0]["runtime_id"] == "new-runtime"
    elif change == "image":
        assert state["components"][0]["image_ref"] == "registry.example/perception:new-tag"
    elif change == "machine":
        assert state["deployments"][0]["machine"] == "jp61-machine"
    else:
        assert all("core" not in dep["component_ids"] for dep in state["deployments"])
    if stage == "logs":
        controller._upload_evidence.assert_not_awaited()
    else:
        controller._upload_evidence.assert_awaited_once()
    proxy.project_status_label.assert_not_awaited()
    controller._deploy_component.assert_not_awaited()
    controller._deploy_core_component.assert_not_awaited()
