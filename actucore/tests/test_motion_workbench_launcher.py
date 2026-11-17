"""The standalone host enforces the same declared plugin boundary over loopback."""

import http.client
import json
import os
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.motion_workbench import Workbench, WorkbenchServer, load_plugins, read_token


@pytest.fixture
def host():
    driver_root = Path(os.environ.get("MOTION_TEST_DRIVER_ROOT") or
                       Path(__file__).resolve().parents[3] / "phanthymotus-driver")
    if not driver_root.is_dir():
        pytest.skip("requires the sibling Driver checkout")
    plugins = load_plugins(driver_root, "http://127.0.0.1:1", "test-secret-never-in-html")
    server = WorkbenchServer(Workbench(plugins), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def request(host, method="GET", path="/", body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", host.server_port, timeout=3)
    values = {"Origin": host.origin, "X-Workbench-Token": host.bridge_token,
              "Content-Type": "application/json"}
    values.update(headers or {})
    conn.request(method, path, body=json.dumps(body) if isinstance(body, dict) else body, headers=values)
    response = conn.getresponse()
    status, data = response.status, response.read().decode()
    conn.close()
    return status, data


def test_real_plugins_load_without_starting_jobs_or_physics(host):
    status, page = request(host)
    assert status == 200
    assert "test-secret-never-in-html" not in page
    assert "new MessageChannel()" in page
    assert "'sandbox','allow-scripts'" in page
    for card in ("motion", "simulation"):
        status, data = request(host, path="/ui/" + card)
        assert status == 200
        assert json.loads(data)["protocol"] == "motus.workbench/1"
        assert "test-secret-never-in-html" not in data
        status, data = request(host, "POST", "/api/" + card, {"method": "config", "args": {}})
        assert status == 200
        assert json.loads(data)["runtime_secret_set"] == {"api_key": True}
        assert "test-secret-never-in-html" not in data
    assert host.workbench.plugins["simulation"]._sessions == {}


@pytest.mark.parametrize("headers", [
    {"Host": "attacker.example"}, {"Origin": "https://attacker.example"},
    {"Origin": "null"}, {"Origin": ""}, {"Sec-Fetch-Site": "cross-site"},
    {"X-Workbench-Token": "wrong"},
])
def test_cross_site_and_unknown_sessions_cannot_call_plugins(host, headers):
    status, _ = request(host, "POST", "/api/motion", {"method": "call", "args": {"action": "info"}}, headers)
    assert status == 403


@pytest.mark.parametrize("body", [
    {"method": "execute", "args": {}},
    {"method": "call", "args": {"action": "shell"}},
    {"method": "call", "args": {"action": "config", "endpoint": "http://evil"}},
    {"method": "call", "args": {"action": "info", "instance_id": "other-card"}},
    {"method": "call", "args": {"action": "train", "endpoint": "http://evil"}},
    {"method": "config", "args": {"values": {"command": "bad"}}},
])
def test_server_checks_action_config_and_routing_permissions(host, body):
    assert request(host, "POST", "/api/motion", body)[0] == 403


def test_body_and_schema_validation_and_explicit_remote_error(host):
    assert request(host, "POST", "/api/motion", "x" * 40001)[0] == 413
    assert request(host, "POST", "/api/motion", "{")[0] == 400
    assert request(host, "POST", "/api/motion", {"method": "call", "args": {"action": "train", "num_envs": True}})[0] == 400
    assert request(host, "POST", "/api/motion", {}, {"Content-Type": "text/plain"})[0] == 415
    status, data = request(host, "POST", "/api/motion", {"method": "call", "args": {"action": "check"}})
    assert status == 502 and json.loads(data)["error"]
    assert "test-secret-never-in-html" not in data


def test_config_updates_runtime_without_sending_credential_back(host):
    status, data = request(host, "POST", "/api/motion", {"method": "config", "args": {"values": {"api_key": "replacement-secret"}}})
    assert status == 200 and json.loads(data) == {"saved": True}
    status, data = request(host, "POST", "/api/motion", {"method": "config", "args": {}})
    assert status == 200 and "replacement-secret" not in data
    assert json.loads(data)["runtime_secret_set"]["api_key"] is True


def test_credentials_require_private_file(tmp_path):
    path = tmp_path / "token"
    path.write_text("private-token\n")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        read_token(path)
    path.chmod(0o600)
    assert read_token(path) == "private-token"
