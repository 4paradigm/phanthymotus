"""Obstacle loader and configured-start regressions, without models or a GPU."""

import threading

import pytest

from vision_stubs import _FakeExecutor, _wait_until
from plugins import obstacle as obstacle_module


class _Adapter:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setattr(obstacle_module, "_build_distance_adapter", _Adapter)
    plugin = obstacle_module.ObstacleDistancePlugin({}, _FakeExecutor())
    yield plugin
    plugin.dispatch("obstacle", {"action": "stop"})
    loader = plugin._loader_thread
    if loader is not None:
        loader.join(timeout=3)


def _start(plugin, key="a"):
    return plugin.dispatch("obstacle", {
        "action": "start", "instance_id": key, "input_topic": f"/cam/{key}",
    })


def _config(plugin, key, threshold):
    return plugin.dispatch("obstacle", {
        "action": "config", "instance_id": key,
        "decision_threshold_m": threshold,
    })


def _running(plugin, key="a"):
    def ready():
        with plugin._state_lock:
            node = plugin._nodes.get(key)
            return node is not None and node.state == "running"
    assert _wait_until(ready)


class _PauseAfterCacheCheck:
    """Schedule config/stop after start releases its cache-read lock."""

    def __init__(self, lock):
        self.lock = lock
        self.reached = threading.Event()
        self.resume = threading.Event()
        self.exits = 0

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, *exc):
        self.lock.release()
        if threading.current_thread().name == "mcp-start-test":
            self.exits += 1
            if self.exits == 2:
                self.reached.set()
                assert self.resume.wait(5)


def test_registration_info_and_config_do_not_build(plugin, monkeypatch):
    def unexpected(cfg):
        pytest.fail("model work before start")
    monkeypatch.setattr(obstacle_module, "_build_distance_adapter", unexpected)
    assert plugin.get_tools()
    assert plugin.dispatch("obstacle", {"action": "info"})["state"] == "idle"
    _config(plugin, "a", 1.5)
    assert plugin._loader_thread is None


def test_configured_start_uses_background_loader(plugin, monkeypatch):
    _start(plugin)
    _running(plugin)
    entered, release = threading.Event(), threading.Event()
    threads = []

    def build(cfg):
        threads.append(threading.current_thread().name)
        entered.set()
        assert release.wait(5)
        return _Adapter(cfg)

    monkeypatch.setattr(obstacle_module, "_build_distance_adapter", build)
    _config(plugin, "b", 1.5)
    try:
        assert _start(plugin, "b")["state"] == "loading"
        assert entered.wait(3)
        assert threads == ["obstacle-adapter-loader"]
        assert plugin.dispatch("obstacle", {"action": "info"})["instances"]["b"]["state"] == "loading"
    finally:
        release.set()
    _running(plugin, "b")


@pytest.mark.parametrize("action", ["config", "stop"])
def test_cache_invalidated_during_start_never_builds_on_mcp(plugin, monkeypatch, action):
    _start(plugin)
    _running(plugin)
    _config(plugin, "b", 1.5)
    _start(plugin, "b")
    _running(plugin, "b")
    plugin.dispatch("obstacle", {"action": "stop", "instance_id": "b"})
    old_adapter = plugin._instance_adapters["b"][1]
    paused = _PauseAfterCacheCheck(plugin._state_lock)
    monkeypatch.setattr(plugin, "_state_lock", paused)
    release_build, built, returned = threading.Event(), threading.Event(), threading.Event()
    threads, outcome = [], {}

    def build(cfg):
        threads.append(threading.current_thread().name)
        built.set()
        assert release_build.wait(5)
        return _Adapter(cfg)

    def start():
        try:
            outcome.update(_start(plugin, "b"))
        finally:
            returned.set()

    monkeypatch.setattr(obstacle_module, "_build_distance_adapter", build)
    caller = threading.Thread(target=start, name="mcp-start-test")
    caller.start()
    try:
        assert paused.reached.wait(3)
        if action == "config":
            _config(plugin, "b", 2.0)
        else:
            plugin.dispatch("obstacle", {"action": "stop", "instance_id": "b"})
        paused.resume.set()
        assert returned.wait(1), "MCP start waited for adapter construction"
        assert outcome["state"] == ("loading" if action == "config" else "idle")
        if action == "config":
            assert built.wait(3)
            assert threads == ["obstacle-adapter-loader"]
            assert old_adapter.closed
        else:
            assert threads == [] and "b" not in plugin._nodes
    finally:
        paused.resume.set()
        release_build.set()
        caller.join(timeout=3)
    if action == "config":
        _running(plugin, "b")
        assert plugin._instance_adapters["b"][1].cfg["decision_threshold_m"] == 2.0


def test_config_changed_before_loader_registers_node(plugin, monkeypatch):
    _start(plugin)
    _running(plugin)
    entered, release = threading.Event(), threading.Event()
    original = obstacle_module._ObstacleNode
    created = []

    def create(topic, suffix):
        node = original(topic, suffix)
        created.append(node)
        if len(created) == 1:
            entered.set()
            assert release.wait(5)
        return node

    monkeypatch.setattr(obstacle_module, "_ObstacleNode", create)
    _config(plugin, "b", 1.5)
    try:
        assert _start(plugin, "b")["state"] == "loading"
        assert entered.wait(3)
        _config(plugin, "b", 2.0)
    finally:
        release.set()
    _running(plugin, "b")
    assert created[0].destroyed and len(created) == 2
    assert plugin._instance_adapters["b"][1].cfg["decision_threshold_m"] == 2.0


@pytest.mark.parametrize("action", ["config", "stop"])
def test_config_or_stop_during_loading(plugin, monkeypatch, action):
    entered, release = threading.Event(), threading.Event()
    adapters = []

    def build(cfg):
        adapter = _Adapter(cfg)
        adapters.append(adapter)
        if len(adapters) == 1:
            entered.set()
            assert release.wait(5)
        return adapter

    monkeypatch.setattr(obstacle_module, "_build_distance_adapter", build)
    try:
        assert _start(plugin)["state"] == "loading"
        assert entered.wait(3)
        if action == "config":
            _config(plugin, "", 1.5)
        else:
            plugin.dispatch("obstacle", {"action": "stop"})
    finally:
        release.set()
    assert _wait_until(lambda: plugin._loader_thread is None)
    if action == "config":
        _running(plugin)
        assert len(adapters) == 2 and adapters[0].closed
        assert plugin._adapter.cfg["decision_threshold_m"] == 1.5
    else:
        assert not plugin._nodes and not plugin._pending_starts


def test_loader_failure_retries_and_stop_disposes_node(plugin, monkeypatch):
    attempts = []

    def build(cfg):
        attempts.append(cfg)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return _Adapter(cfg)

    monkeypatch.setattr(obstacle_module, "_build_distance_adapter", build)
    assert _start(plugin)["state"] == "loading"
    assert _wait_until(lambda: plugin._adapter_state == "error")
    assert "model unavailable" in plugin.dispatch("obstacle", {"action": "info"})["error"]
    _start(plugin)
    _running(plugin)
    node = plugin._nodes["a"]
    plugin.dispatch("obstacle", {"action": "stop"})
    assert node.destroyed and not plugin._executor.nodes
    assert len(attempts) == 2
