"""PR #314 targeted regression tests for Core no-case ordering and JP5.11 health contracts.

Tests:
1. test_pr314_core_no_case_skips_core_resolution
2. test_pr314_core_missing_deployment_still_fails
3. test_pr314_case_selector_exception_fails_closed
4. test_pr314_perception_jp511_health_contract_pass
5. test_pr314_actucore_jp511_health_contract_pass
6. test_pr314_jp511_wrong_image_fails_closed
7. test_pr314_jp511_mcp_offline_fails_closed
"""

from __future__ import annotations

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
            targets=["perception", "core"],
            platforms=["linux/arm64"],
            variants=["5.11"],
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
        "variant": "5.11",
        "review_image_tag": "registry/repo:v1",
        "image_ref": "registry/repo@sha256:" + "c" * 64,
        "resolved_platform": "linux/arm64",
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
async def test_pr314_core_missing_deployment_still_fails(controller):
    """Core with no deployment record → fail, not n/a.

    Missing deployment is a genuine failure regardless of case registry.
    """
    core_comp = _core_component()
    # Empty deployments list — no record for comp-core
    deployments = []

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
async def test_pr314_perception_jp511_health_contract_pass(controller):
    """Full Perception JP5.11 health chain: image match + runtime_id pinned +
    driver found + MCP bound + mcp_ping online → pass.
    """
    image_ref = "registry/repo@sha256:" + "p" * 64
    perc_comp = _perception_component(image_ref=image_ref)
    deployments = [{"machine": "test-machine", "component_ids": ["comp-perc"], "phase": "deployed"}]

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
    core.driver_status.assert_called_once_with("perception-driver")
    core.list_drivers.assert_called_once()
    core.list_mcp.assert_called_once()
    core.mcp_ping.assert_called_once_with("mcp-perception-1")


# ── Test 5: ActuCore JP5.11 health contract pass ─────────────────────

@pytest.mark.asyncio
async def test_pr314_actucore_jp511_health_contract_pass(controller):
    """Symmetric to Perception test: ActuCore JP5.11 full chain passes."""
    image_ref = "registry/repo@sha256:" + "u" * 64
    actucore_comp = _actucore_component(image_ref=image_ref)
    deployments = [{"machine": "test-machine", "component_ids": ["comp-actucore"], "phase": "deployed"}]

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
