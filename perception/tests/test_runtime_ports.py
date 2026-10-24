"""Deployment port overrides must isolate containers without changing main.py."""

from __future__ import annotations

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import socket
import stat
import sys

import pytest
import yaml


@pytest.fixture(scope="module")
def launcher():
    path = Path(__file__).resolve().parents[1] / "deploy" / "run_perception.py"
    spec = importlib.util.spec_from_file_location("runtime_ports_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def deployment(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    main_path = app / "main.py"
    main_path.write_text("# No ROS or model imports are needed by these tests.\n")
    config = {
        "mcp_port": 15720,
        "ws_port": 15721,
        "plugins": {"tts": {"enabled": True, "speed": 1.0}},
        "model_dir": "/models/vits2",
        "robot_name": "中文机器人",
        "unrelated": [False, None, 3],
    }
    config_path = app / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return main_path, config_path, config, tmp_path / "runtime"


def read_config(environment):
    return yaml.safe_load(Path(environment["CONFIG_PATH"]).read_text(encoding="utf-8"))


def test_overrides_preserve_source_and_unrelated_configuration(launcher, deployment):
    main_path, source, config, runtime = deployment
    original = source.read_bytes()
    environment = {"MCP_PORT": "15820", "WS_PORT": "15821", "UNCHANGED": "value"}
    result = launcher.prepare_environment(main_path, environment, runtime)

    assert read_config(result) == {**config, "mcp_port": 15820, "ws_port": 15821}
    assert source.read_bytes() == original
    assert "CONFIG_PATH" not in environment
    assert result["UNCHANGED"] == "value"
    assert Path(result["CONFIG_PATH"]).parent == runtime
    assert Path(result["CONFIG_PATH"]) != source


def test_generated_configuration_is_private(launcher, deployment):
    main_path, _, _, runtime = deployment
    result = launcher.prepare_environment(main_path, {"MCP_PORT": "15820"}, runtime)
    assert stat.S_IMODE(Path(result["CONFIG_PATH"]).stat().st_mode) == 0o600


@pytest.mark.parametrize("environment", [{"UNCHANGED": "value"}, {"MCP_PORT": "", "WS_PORT": ""}])
def test_no_overrides_needs_no_configuration_or_runtime_directory(launcher, tmp_path, environment):
    main_path = tmp_path / "missing-app" / "main.py"
    runtime = tmp_path / "runtime"
    assert launcher.prepare_environment(main_path, environment, runtime) == environment
    assert not runtime.exists()
    assert not main_path.parent.exists()


@pytest.mark.parametrize("environment, expected", [
    ({"MCP_PORT": "15820"}, {"mcp_port": 15820, "ws_port": 15721}),
    ({"WS_PORT": "15821"}, {"mcp_port": 15720, "ws_port": 15821}),
    ({"MCP_PORT": "", "WS_PORT": "15821"}, {"mcp_port": 15720, "ws_port": 15821}),
    ({"MCP_PORT": "15820", "WS_PORT": ""}, {"mcp_port": 15820, "ws_port": 15721}),
])
def test_single_override_keeps_other_configured_port(launcher, deployment, environment, expected):
    main_path, _, config, runtime = deployment
    result = launcher.prepare_environment(main_path, environment, runtime)
    assert read_config(result) == {**config, **expected}


def test_explicit_config_path_is_respected(launcher, deployment, tmp_path):
    main_path, default_source, _, runtime = deployment
    source = tmp_path / "custom.yaml"
    custom = {"mcp_port": 17000, "ws_port": 17001, "plugins": ["tts"], "custom": True}
    source.write_text(yaml.safe_dump(custom), encoding="utf-8")
    originals = (source.read_bytes(), default_source.read_bytes())
    environment = {"CONFIG_PATH": str(source), "MCP_PORT": "18000"}

    result = launcher.prepare_environment(main_path, environment, runtime)

    assert read_config(result) == {**custom, "mcp_port": 18000}
    assert (source.read_bytes(), default_source.read_bytes()) == originals
    assert environment["CONFIG_PATH"] == str(source)
    assert result["CONFIG_PATH"] != str(source)


@pytest.mark.parametrize("variable", ["MCP_PORT", "WS_PORT"])
@pytest.mark.parametrize("value", ["0", "-1", "65536", "abc", "1.5"])
def test_invalid_environment_port_is_rejected(launcher, deployment, variable, value):
    main_path, source, _, runtime = deployment
    original = source.read_bytes()
    with pytest.raises(ValueError, match=r"(?i)port"):
        launcher.prepare_environment(main_path, {variable: value}, runtime)
    assert source.read_bytes() == original


def test_valid_port_boundaries(launcher, deployment):
    main_path, _, _, runtime = deployment
    result = launcher.prepare_environment(main_path, {"MCP_PORT": "1", "WS_PORT": "65535"}, runtime)
    assert read_config(result)["mcp_port"] == 1
    assert read_config(result)["ws_port"] == 65535


@pytest.mark.parametrize("environment", [
    {"MCP_PORT": "18000", "WS_PORT": "18000"},
    {"MCP_PORT": "15721"},
    {"WS_PORT": "15720"},
])
def test_colliding_effective_ports_are_rejected(launcher, deployment, environment):
    main_path, source, _, runtime = deployment
    original = source.read_bytes()
    with pytest.raises(ValueError, match=r"(?i)port"):
        launcher.prepare_environment(main_path, environment, runtime)
    assert source.read_bytes() == original


def test_three_instances_keep_separate_configuration_files(launcher, deployment):
    main_path, source, _, runtime = deployment
    original = source.read_bytes()
    results = [
        launcher.prepare_environment(main_path, {
            "MCP_PORT": str(15720 + index * 100),
            "WS_PORT": str(15721 + index * 100),
        }, runtime)
        for index in range(3)
    ]
    assert len({result["CONFIG_PATH"] for result in results}) == 3
    for index, result in enumerate(results):
        config = read_config(result)
        assert (config["mcp_port"], config["ws_port"]) == (15720 + index * 100, 15721 + index * 100)
    assert source.read_bytes() == original


def test_three_instances_can_bind_all_six_configured_ports(launcher, deployment):
    """Use real sockets in one network namespace, like host-network containers."""
    main_path, _, _, runtime = deployment
    with ExitStack() as reservations:
        ports = []
        for _ in range(6):
            sock = reservations.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])

    with ExitStack() as listeners:
        bound_ports = []
        for index in range(3):
            result = launcher.prepare_environment(main_path, {
                "MCP_PORT": str(ports[index * 2]),
                "WS_PORT": str(ports[index * 2 + 1]),
            }, runtime)
            config = read_config(result)
            for key in ("mcp_port", "ws_port"):
                sock = listeners.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                sock.bind(("127.0.0.1", config[key]))
                sock.listen(1)
                bound_ports.append(sock.getsockname()[1])
        assert bound_ports == ports
        assert len(set(bound_ports)) == 6


def test_main_exec_preserves_arguments_and_prepared_environment(launcher, deployment, monkeypatch):
    main_path, _, _, runtime = deployment
    environment = {"MCP_PORT": "15820", "WS_PORT": "15821", "UNCHANGED": "value"}
    prepared = launcher.prepare_environment(main_path, environment, runtime)
    prepare_calls = []
    exec_calls = []

    def prepare(path, environ, runtime_dir):
        prepare_calls.append((path, dict(environ), runtime_dir))
        return prepared

    class ExecReplacedProcess(Exception):
        pass

    def execvpe(executable, arguments, environ):
        exec_calls.append((executable, arguments, environ))
        raise ExecReplacedProcess

    monkeypatch.setattr(launcher, "prepare_environment", prepare)
    monkeypatch.setattr(launcher.os, "environ", environment)
    monkeypatch.setattr(launcher.os, "execvpe", execvpe)
    monkeypatch.setattr(launcher.sys, "argv", ["run_perception.py", str(main_path), "--example", "two words"])
    with pytest.raises(ExecReplacedProcess):
        launcher.main()

    assert len(prepare_calls) == 1
    assert Path(prepare_calls[0][0]) == main_path
    assert prepare_calls[0][1] == environment
    assert exec_calls == [(sys.executable, [sys.executable, str(main_path), "--example", "two words"], prepared)]
