"""Exercise startup port selection without ROS, sockets or background threads."""
from types import SimpleNamespace

import pytest

import vision_stubs  # noqa: F401  (installs ROS stubs before importing main)


@pytest.fixture
def startup(monkeypatch):
    import main

    observed = {}
    config = {}
    monkeypatch.delenv("MCP_PORT", raising=False)
    monkeypatch.delenv("WS_PORT", raising=False)
    monkeypatch.setenv("RCUTILS_LOGGING_SEVERITY_THRESHOLD", "50")
    monkeypatch.setenv("ROS_LOG_LEVEL", "WARN")
    monkeypatch.setattr(main, "_bundle", None)
    monkeypatch.setattr(main, "_load_config", lambda: config)
    monkeypatch.setattr(main.rclpy, "init", lambda: observed.update(ros_initialized=True))
    monkeypatch.setattr(main.rclpy, "shutdown", lambda: None)
    monkeypatch.setattr(main.rclpy.executors, "MultiThreadedExecutor",
                        lambda: SimpleNamespace(shutdown=lambda: None))
    monkeypatch.setattr(main, "PerceptionBundle",
                        lambda *args: SimpleNamespace(shutdown=lambda: None))
    monkeypatch.setattr(main, "signal", SimpleNamespace(
        signal=lambda *args: None, SIGTERM=main.signal.SIGTERM, SIGINT=main.signal.SIGINT))

    def thread(*, target, args=(), **kwargs):
        if target is main._start_ws_thread:
            observed["ws_port"] = args[0]
        return SimpleNamespace(start=lambda: None)

    def server(address, handler):
        observed["mcp_port"] = address[1]
        return SimpleNamespace(serve_forever=lambda: None)

    monkeypatch.setattr(main, "threading", SimpleNamespace(Thread=thread))
    monkeypatch.setattr(main, "ThreadingHTTPServer", server)
    monkeypatch.setattr(main, "_start_registration",
                        lambda port, *args: observed.update(registration_port=port))
    return main, config, observed


@pytest.mark.parametrize("config,env,expected", [
    ({}, {}, (15720, 15721)),
    ({"mcp_port": 17120, "ws_port": 17121}, {}, (17120, 17121)),
    ({"mcp_port": 17120, "ws_port": 17121},
     {"MCP_PORT": "18220", "WS_PORT": "18221"}, (18220, 18221)),
    ({"mcp_port": 17120, "ws_port": 17121},
     {"MCP_PORT": "", "WS_PORT": ""}, (17120, 17121)),
    ({"mcp_port": 17120, "ws_port": 17121}, {"MCP_PORT": "18220"}, (18220, 17121)),
    ({"mcp_port": 17120, "ws_port": 17121}, {"WS_PORT": "18221"}, (17120, 18221)),
    ({"mcp_port": "invalid", "ws_port": "invalid"},
     {"MCP_PORT": "18220", "WS_PORT": "18221"}, (18220, 18221)),
])
def test_startup_ports_follow_environment_then_config_then_defaults(
        startup, monkeypatch, config, env, expected):
    main, saved, observed = startup
    saved.update(config)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    main.main()
    assert observed["mcp_port"] == observed["registration_port"] == expected[0]
    assert observed["ws_port"] == expected[1]


@pytest.mark.parametrize("source,key", [
    ("env", "MCP_PORT"), ("env", "WS_PORT"),
    ("config", "mcp_port"), ("config", "ws_port"),
])
def test_noninteger_port_fails_before_initializing_ros(startup, monkeypatch, source, key):
    main, saved, observed = startup
    if source == "env":
        monkeypatch.setenv(key, "not-a-port")
    else:
        saved[key] = "not-a-port"
    with pytest.raises(ValueError):
        main.main()
    assert observed == {}
