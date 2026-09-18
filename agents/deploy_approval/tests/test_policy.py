"""Policy tests: machine owners, collaborator authority, self-approval."""
from __future__ import annotations

import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock
import yaml

import agents.deploy_approval.policy as policy_mod
from ..policy import Policy, PolicyError, load_machines, MachineLoadError
from .conftest import TEST_CERT_PEM, make_config


def _machine_yaml(**entry_overrides):
    machine = {
        "node_id": "g1-bj-001",
        "node_host": "10.0.0.1",
        "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": ["alice"],
        "targets": ["perception"],
        "platforms": ["linux/arm64"],
    }
    machine.update(entry_overrides)
    return {"version": 1, "machines": {"g1-bj": machine}}


def _write_yaml(data):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    path = f.name
    f.close()
    return path


def test_machine_owners_load():
    path = _write_yaml({
        "version": 1,
        "machines": {
            "g1-bj": {
                "node_id": "g1-bj-001",
                "node_host": "10.0.0.1",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": ["alice", "bob"],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
            "t800-lab": {
                "node_id": "t800-lab-01",
                "node_host": "10.0.0.2",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": ["charlie"],
                "targets": ["actucore"],
                "platforms": ["linux/arm64"],
            },
        },
    })
    try:
        machines = load_machines(path)
        assert machines["g1-bj"].node_id == "g1-bj-001"
        assert machines["g1-bj"].node_host == "10.0.0.1"
        assert machines["g1-bj"].owners == ["alice", "bob"]
        assert machines["t800-lab"].owners == ["charlie"]
    finally:
        os.unlink(path)


def test_machine_owners_duplicate_node_id():
    path = _write_yaml({
        "version": 1,
        "machines": {
            "g1-bj": {
                "node_id": "node-001",
                "node_host": "10.0.0.1",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": ["alice"],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
            "t800-lab": {
                "node_id": "node-001",
                "node_host": "10.0.0.2",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": ["bob"],
                "targets": ["actucore"],
                "platforms": ["linux/arm64"],
            },
        },
    })
    try:
        with pytest.raises(MachineLoadError, match="duplicate node_id"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_owners_empty_owners():
    path = _write_yaml({
        "version": 1,
        "machines": {
            "g1-bj": {
                "node_id": "g1-bj-001",
                "node_host": "10.0.0.1",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": [],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
        },
    })
    try:
        with pytest.raises(MachineLoadError, match="owners list is empty"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_owners_invalid_version():
    path = _write_yaml({
        "version": 2,
        "machines": {
            "g1-bj": {
                "node_id": "n1",
                "node_host": "10.0.0.1",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": ["alice"],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
        },
    })
    try:
        with pytest.raises(MachineLoadError, match="version must be 1"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_owners_empty_machines():
    path = _write_yaml({"version": 1, "machines": {}})
    try:
        with pytest.raises(MachineLoadError, match="empty"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_owners_missing_node_id():
    path = _write_yaml({
        "version": 1,
        "machines": {
            "g1-bj": {
                "node_host": "10.0.0.1",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": ["alice"],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
        },
    })
    try:
        with pytest.raises(MachineLoadError, match="node_id"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_owners_owner_not_string():
    path = _write_yaml({
        "version": 1,
        "machines": {
            "g1-bj": {
                "node_id": "n1",
                "node_host": "10.0.0.1",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": [123],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
        },
    })
    try:
        with pytest.raises(MachineLoadError, match="non-empty string"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_owners_duplicate_owner_case_insensitive():
    path = _write_yaml({
        "version": 1,
        "machines": {
            "g1-bj": {
                "node_id": "n1",
                "node_host": "10.0.0.1",
                "tls_peer_cert_file": "/run/deploy-approval/certs/test-agent-core.pem",
                "owners": ["Alice", "alice", "ALICE"],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
        },
    })
    try:
        machines = load_machines(path)
        assert machines["g1-bj"].owners == ["alice"]
    finally:
        os.unlink(path)


def test_machine_owners_missing_file():
    with pytest.raises(MachineLoadError, match="not found"):
        load_machines("/nonexistent/path.yaml")


def test_collaborator_write_can_approve():
    assert Policy.collaborator_can_approve("write")
    assert Policy.collaborator_can_approve("maintain")
    assert Policy.collaborator_can_approve("admin")


def test_collaborator_read_triage_none_rejected():
    for perm in ("read", "triage", "pull", "", None, "admin-x", 123):
        assert Policy.collaborator_can_approve(perm) is False


def test_self_approval_allowed_by_id():
    path = _write_yaml(_machine_yaml())
    try:
        p = Policy(make_config(machine_owners_file=path))
        p.load_machines()
        p.can_approve("alice", machine_alias="g1-bj")
    finally:
        os.unlink(path)


def test_approve_not_owner():
    path = _write_yaml(_machine_yaml())
    try:
        p = Policy(make_config(machine_owners_file=path))
        p.load_machines()
        with pytest.raises(PolicyError, match="not an owner"):
            p.can_approve("mallory", machine_alias="g1-bj")
    finally:
        os.unlink(path)


def test_approve_owner_allowed():
    path = _write_yaml(_machine_yaml())
    try:
        p = Policy(make_config(machine_owners_file=path))
        p.load_machines()
        p.can_approve("alice", machine_alias="g1-bj")
    finally:
        os.unlink(path)


def test_policy_fingerprint():
    p = Policy(make_config())
    fp = p._fingerprint(make_config())
    assert fp.startswith("deploy-approval-")
    assert len(fp) > 16


def test_machine_missing_tls_peer_cert_file_fails():
    data = _machine_yaml()
    del data["machines"]["g1-bj"]["tls_peer_cert_file"]
    path = _write_yaml(data)
    try:
        with pytest.raises(MachineLoadError, match="tls_peer_cert_file is required"):
            load_machines(path)
    finally:
        os.unlink(path)


@pytest.mark.parametrize(
    "cert_path,match",
    [
        ("certs/test.pem", "absolute"),
        ("/run/deploy-approval/certs/../test.pem", "must not contain"),
        ("/tmp/test.pem", "under /run/deploy-approval/certs"),
    ],
)
def test_machine_tls_peer_cert_bad_paths_fail(cert_path, match):
    path = _write_yaml(_machine_yaml(tls_peer_cert_file=cert_path))
    try:
        with pytest.raises(MachineLoadError, match=match):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_tls_peer_cert_symlink_fails(tmp_path, monkeypatch):
    target = tmp_path / "target.pem"
    target.write_text(TEST_CERT_PEM, encoding="ascii")
    symlink = tmp_path / "linked.pem"
    symlink.symlink_to(target)
    monkeypatch.setattr(policy_mod, "_cert_filesystem_path", lambda cert_file: symlink)
    path = _write_yaml(_machine_yaml(tls_peer_cert_file="/run/deploy-approval/certs/linked.pem"))
    try:
        with pytest.raises(MachineLoadError, match="symlink"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_tls_peer_cert_missing_file_fails(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pem"
    monkeypatch.setattr(policy_mod, "_cert_filesystem_path", lambda cert_file: missing)
    path = _write_yaml(_machine_yaml(tls_peer_cert_file="/run/deploy-approval/certs/missing.pem"))
    try:
        with pytest.raises(MachineLoadError, match="not found or unreadable"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_tls_peer_cert_invalid_pem_fails(tmp_path, monkeypatch):
    cert = tmp_path / "bad.pem"
    cert.write_text("not a certificate", encoding="ascii")
    monkeypatch.setattr(policy_mod, "_cert_filesystem_path", lambda cert_file: cert)
    path = _write_yaml(_machine_yaml(tls_peer_cert_file="/run/deploy-approval/certs/bad.pem"))
    try:
        with pytest.raises(MachineLoadError, match="exactly one PEM certificate"):
            load_machines(path)
    finally:
        os.unlink(path)


def test_machine_tls_peer_cert_exact_valid_path_passes(tmp_path, monkeypatch):
    cert = tmp_path / "test-agent-core.pem"
    cert.write_text(TEST_CERT_PEM, encoding="ascii")
    monkeypatch.setattr(policy_mod, "_cert_filesystem_path", lambda cert_file: cert)
    path = _write_yaml(_machine_yaml())
    try:
        machines = load_machines(path)
        assert machines["g1-bj"].tls_peer_cert_file == "/run/deploy-approval/certs/test-agent-core.pem"
    finally:
        os.unlink(path)




# ══════════════════════════════════════════════════════════════════════════════
# MIGRATED from test_v10_contract.py
# ══════════════════════════════════════════════════════════════════════════════

# ── Fixtures and helpers for driver runtime fallback tests (migrated) ──


@pytest.fixture
def mock_github():
    from unittest.mock import MagicMock
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
    from ..registry_client import RegistryClient
    registry = MagicMock()
    registry.resolve = AsyncMock()
    from ..service import DeployController
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


# ── Driver runtime fallback tests (migrated) ──

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
