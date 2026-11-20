"""Deployment must preserve environment bindings and expose the installer."""
from pathlib import Path

import pytest


def test_motion_connection_and_state_survive_the_bundle_deployment():
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1]
    service = yaml.safe_load((root / "deploy/service.yml").read_text())["actucore"]
    environment = dict(item.split("=", 1) for item in service["environment"])
    for name in ("MOTION_SERVICE_URL", "MOTION_API_TOKEN", "MOTION_ENVIRONMENT_URL", "MOTION_ENVIRONMENT_TOKEN"):
        assert environment[name] == "${" + name + ":-}"
    state_path = environment["MOTION_WORKBENCH_STATE_DIR"]
    assert any(mount.endswith(":" + state_path) for mount in service["volumes"])
