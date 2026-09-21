"""Policy tests: machine owners, collaborator authority, self-approval."""
from __future__ import annotations

import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock
import yaml

import agents.deploy_approval.policy as policy_mod
from ..policy import Policy, PolicyError, load_machines, MachineLoadError
from .conftest import make_config


def _machine_yaml(**entry_overrides):
    machine = {
        "node_id": "g1-bj-001",
        "node_host": "10.0.0.1",
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
                "owners": ["alice", "bob"],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
            "t800-lab": {
                "node_id": "t800-lab-01",
                "node_host": "10.0.0.2",
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
                "owners": ["alice"],
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
            },
            "t800-lab": {
                "node_id": "node-001",
                "node_host": "10.0.0.2",
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
