"""
tests/test_obstacle_plugin.py — obstacle plugin lifecycle/concurrency tests.

ROS stubs come from vision_stubs (installed by conftest before collection).
Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from utils.qos import CAMERA_QOS

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _FakeString,
    _wait_until,
)

# ── Obstacle ─────────────────────────────────────────────────────────────────

import plugins.obstacle as obstacle_plugin  # noqa: E402


class _FakeDistanceAdapter:
    def __init__(self):
        self.closed = False
        self.estimates = 0

    def estimate(self, image_bytes: bytes) -> dict:
        self.estimates += 1
        return {"pred_distance": 3.0, "status": "ok", "fallback": False}

    def close(self):
        self.closed = True


class _ObstacleBuilderProbe:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0
        self.built = []
        self.lock = threading.Lock()

    def __call__(self, cfg):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        adapter = _FakeDistanceAdapter()
        adapter.cfg = dict(cfg)
        self.built.append(adapter)
        return adapter


def _make_obstacle(monkeypatch, builder=None, cfg=None):
    builder = builder or _ObstacleBuilderProbe()
    monkeypatch.setattr(obstacle_plugin, "_build_distance_adapter", builder)
    executor = _FakeExecutor()
    plugin = obstacle_plugin.ObstacleDistancePlugin(
        dict(cfg or {"provider": "local"}), executor
    )
    return plugin, executor, builder


def _obstacle_start_and_wait(plugin, executor, topic, instance_id="", count=1):
    args = {"action": "start", "input_topic": topic}
    if instance_id:
        args["instance_id"] = instance_id
    plugin.dispatch("obstacle", args)
    # Registration now precedes start (README lifecycle rule), so a node can
    # briefly sit in the executor as "idle"; wait for it to actually run.
    assert _wait_until(
        lambda: len(executor.nodes) >= count
        and all(n.state == "running" for n in executor.nodes)
    ), "node never came up"


@pytest.fixture
def obstacle(monkeypatch):
    plugin, executor, builder = _make_obstacle(monkeypatch)
    yield plugin, executor, builder
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_camera_subscription_uses_shared_qos(obstacle):
    plugin, executor, _ = obstacle
    _obstacle_start_and_wait(plugin, executor, "/cam/a")
    assert executor.nodes[0].subscriptions[0].qos is CAMERA_QOS


def test_obstacle_start_returns_loading_without_blocking(monkeypatch):
    plugin, executor, builder = _make_obstacle(monkeypatch, _ObstacleBuilderProbe(delay=1.0))
    started = time.monotonic()
    result = plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a"})
    elapsed = time.monotonic() - started
    assert elapsed < 0.1, f"start blocked for {elapsed:.3f}s"
    assert result == {"state": "loading", "input": "/cam/a", "output": "/cam/a/obstacle"}
    assert _wait_until(lambda: len(executor.nodes) == 1)
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_ten_concurrent_starts_single_flight(monkeypatch):
    plugin, executor, builder = _make_obstacle(monkeypatch, _ObstacleBuilderProbe(delay=0.3))
    results = []

    def call(index):
        results.append(plugin.dispatch("obstacle", {
            "action": "start",
            "input_topic": f"/cam/{index}",
            "instance_id": f"inst{index}",
        }))

    threads = [threading.Thread(target=call, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(res["state"] == "loading" for res in results)
    assert _wait_until(lambda: len(executor.nodes) == 10)
    assert builder.calls == 1, "ten starts must share one engine load"
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_info_does_not_block_while_loading(monkeypatch):
    plugin, executor, builder = _make_obstacle(monkeypatch, _ObstacleBuilderProbe(delay=0.8))
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    started = time.monotonic()
    info = plugin.dispatch("obstacle", {"action": "info"})
    elapsed = time.monotonic() - started
    assert elapsed < 0.1, f"info blocked for {elapsed:.3f}s"
    assert info["state"] == "loading"
    assert info["instances"]["a"]["state"] == "loading"
    assert _wait_until(
        lambda: plugin.dispatch("obstacle", {"action": "info"})["state"] == "running"
    )
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_stop_during_loading_cancels_pending(monkeypatch):
    plugin, executor, builder = _make_obstacle(monkeypatch, _ObstacleBuilderProbe(delay=0.3))
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/b", "instance_id": "b"})
    plugin.dispatch("obstacle", {"action": "stop", "instance_id": "a"})
    assert _wait_until(lambda: len(executor.nodes) == 1)
    time.sleep(0.2)  # loader must not resurrect the stopped instance
    assert len(executor.nodes) == 1
    assert executor.nodes[0].subscriptions[0].topic == "/cam/b"
    info = plugin.dispatch("obstacle", {"action": "info"})
    assert "a" not in info["instances"]
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_config_during_loading_discards_stale_adapter(monkeypatch):
    builder = _ObstacleBuilderProbe(delay=0.3)
    plugin, executor, _ = _make_obstacle(
        monkeypatch, builder, cfg={"provider": "local", "fixed_scene": "indoor"}
    )
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    plugin.dispatch("obstacle", {"action": "config", "fixed_scene": "vehicle"})
    assert _wait_until(lambda: len(executor.nodes) == 1, timeout=4)
    assert builder.calls == 2, "config change during load must trigger a reload"
    assert _wait_until(lambda: len(builder.built) == 2)
    stale = next(a for a in builder.built if a.cfg["fixed_scene"] == "indoor")
    fresh = next(a for a in builder.built if a.cfg["fixed_scene"] == "vehicle")
    assert _wait_until(lambda: stale.closed), "stale adapter must be closed"
    assert plugin._adapter is fresh
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_plain_stop_fully_disposes_node(obstacle):
    plugin, executor, _ = obstacle
    _obstacle_start_and_wait(plugin, executor, "/cam/a", instance_id="a")
    node = executor.nodes[0]
    plugin.dispatch("obstacle", {"action": "stop", "instance_id": "a"})
    assert executor.nodes == []
    assert node.destroyed
    assert plugin._nodes == {}


def test_obstacle_repeated_start_stop_does_not_grow(obstacle):
    plugin, executor, builder = obstacle
    for _ in range(20):
        _obstacle_start_and_wait(plugin, executor, "/cam/a", instance_id="a")
        plugin.dispatch("obstacle", {"action": "stop", "instance_id": "a"})
    assert executor.nodes == []
    assert builder.calls == 1, "engines must be loaded once and cached"
    created = [n for n in _FakeNode.instances if n.subscriptions]
    assert all(node.destroyed for node in created)


def test_obstacle_topic_rebind_disposes_old_node(obstacle):
    plugin, executor, _ = obstacle
    _obstacle_start_and_wait(plugin, executor, "/cam/a", instance_id="x")
    old = executor.nodes[0]
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/b", "instance_id": "x"})
    assert _wait_until(
        lambda: executor.nodes and executor.nodes[-1].subscriptions
        and executor.nodes[-1].subscriptions[0].topic == "/cam/b"
    )
    assert old not in executor.nodes and old.destroyed
    assert len(executor.nodes) == 1


def test_obstacle_per_instance_config_builds_own_adapter(obstacle):
    plugin, executor, builder = obstacle
    _obstacle_start_and_wait(plugin, executor, "/cam/a", instance_id="a")
    plugin.dispatch("obstacle", {
        "action": "config", "instance_id": "b", "fixed_scene": "vehicle",
    })
    _obstacle_start_and_wait(plugin, executor, "/cam/b", instance_id="b", count=2)
    assert builder.calls == 2, "instance with own config gets its own adapter"
    cfgs = sorted(a.cfg.get("fixed_scene", "") for a in builder.built)
    assert cfgs[-1] == "vehicle"
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_stop_during_ready_path_start_leaves_no_orphan(monkeypatch):
    """Same orphan-node class as PR #113: stop racing a fast-path start."""
    plugin, executor, builder = _make_obstacle(monkeypatch)
    _obstacle_start_and_wait(plugin, executor, "/cam/a", instance_id="a")

    real_start = obstacle_plugin._ObstacleNode.start

    def slow_start(self, *a, **k):
        time.sleep(0.2)
        return real_start(self, *a, **k)

    monkeypatch.setattr(obstacle_plugin._ObstacleNode, "start", slow_start)
    thread = threading.Thread(target=plugin.dispatch, args=(
        "obstacle", {"action": "start", "input_topic": "/cam/b", "instance_id": "b"}))
    thread.start()
    time.sleep(0.05)
    plugin.dispatch("obstacle", {"action": "stop", "instance_id": "b"})
    thread.join(timeout=3)
    monkeypatch.setattr(obstacle_plugin._ObstacleNode, "start", real_start)

    assert "b" not in plugin._nodes
    b_nodes = [n for n in _FakeNode.instances
               if n.subscriptions and n.subscriptions[0].topic == "/cam/b"]
    assert all(node.destroyed for node in b_nodes), "orphan node left running"
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_instance_config_start_does_not_block(monkeypatch):
    """After the shared adapter is ready, starting an instance whose
    per-instance adapter is not built yet must still return loading
    immediately instead of building TensorRT engines in the MCP thread."""
    builder = _ObstacleBuilderProbe(delay=0.4)
    plugin, executor, _ = _make_obstacle(monkeypatch, builder)
    _obstacle_start_and_wait(plugin, executor, "/cam/a", instance_id="a")
    assert builder.calls == 1

    plugin.dispatch("obstacle", {
        "action": "config", "instance_id": "b", "fixed_scene": "vehicle",
    })
    started = time.monotonic()
    result = plugin.dispatch("obstacle", {
        "action": "start", "input_topic": "/cam/b", "instance_id": "b",
    })
    elapsed = time.monotonic() - started
    assert elapsed < 0.1, f"per-instance start blocked for {elapsed:.3f}s"
    assert result["state"] == "loading"
    assert _wait_until(lambda: len(executor.nodes) == 2, timeout=4)
    assert builder.calls == 2
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_info_desc_explains_loading_and_error(monkeypatch):
    """desc carries the reason while loading / after failure, like ASR (#113)."""
    builder = _ObstacleBuilderProbe(delay=0.5)
    plugin, executor, _ = _make_obstacle(monkeypatch, builder)
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})

    info = plugin.dispatch("obstacle", {"action": "info"})
    assert info["state"] == "loading" and "Loading" in info["desc"]

    assert _wait_until(
        lambda: plugin.dispatch("obstacle", {"action": "info"})["state"] == "running",
        timeout=4,
    )
    ready = plugin.dispatch("obstacle", {"action": "info"})
    assert ready["state"] == "running" and ready["desc"] == plugin._DESC
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_instance_config_racing_adapter_build(monkeypatch):
    """config(instance_id=...) landing while that instance's per-instance
    adapter is being built must not install/serve the stale adapter, and the
    discarded build must be closed (no engine leak)."""
    builder = _ObstacleBuilderProbe(delay=0.3)
    plugin, executor, _ = _make_obstacle(monkeypatch, builder)
    _obstacle_start_and_wait(plugin, executor, "/cam/a", instance_id="a")
    assert builder.calls == 1  # shared adapter

    plugin.dispatch("obstacle", {
        "action": "config", "instance_id": "b", "fixed_scene": "vehicle",
    })
    plugin.dispatch("obstacle", {
        "action": "start", "input_topic": "/cam/b", "instance_id": "b",
    })
    # per-instance build for b is now sleeping in the loader; change b's
    # config mid-build
    time.sleep(0.05)
    plugin.dispatch("obstacle", {
        "action": "config", "instance_id": "b", "fixed_scene": "indoor",
    })
    plugin.dispatch("obstacle", {
        "action": "start", "input_topic": "/cam/b", "instance_id": "b",
    })
    assert _wait_until(lambda: "b" in plugin._nodes, timeout=5)

    with plugin._state_lock:
        cached_cfg, cached_adapter = plugin._instance_adapters["b"]
    assert cached_cfg.get("fixed_scene") == "indoor", "stale adapter installed"
    # every adapter built with the superseded config must be closed
    stale = [a for a in builder.built
             if getattr(a, "cfg", {}).get("fixed_scene") == "vehicle"]
    assert stale, "expected at least one stale build"
    assert all(a.closed for a in stale), "stale per-instance adapter leaked"
    assert not cached_adapter.closed
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_add_node_failure_leaks_nothing(monkeypatch):
    """If executor.add_node raises during bring-up, the node must be
    destroyed and untracked with no started worker (bot P1: a started but
    unregistered worker would be unreachable forever)."""
    plugin, executor, builder = _make_obstacle(monkeypatch)

    original_add = executor.add_node
    calls = {"n": 0}

    def flaky_add(node):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("executor shutting down")
        return original_add(node)

    monkeypatch.setattr(executor, "add_node", flaky_add)
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert _wait_until(lambda: calls["n"] >= 1, timeout=4)
    time.sleep(0.1)

    assert "a" not in plugin._nodes
    assert "a" not in plugin._pending_starts, "failed instance stuck as loading"
    failed = [n for n in _FakeNode.instances
              if n.subscriptions == [] or (n.subscriptions and n.subscriptions[0].topic == "/cam/a")]
    assert all(n.destroyed for n in failed if not n.subscriptions), "unregistered node leaked"
    # worker was never started on the failed node (register-before-start)
    assert all(not n.subscriptions for n in _FakeNode.instances if n.destroyed)

    # the plugin recovers: a later start succeeds
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert _wait_until(lambda: "a" in plugin._nodes, timeout=4)
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_load_failure_reports_error_and_retries(monkeypatch):
    calls = {"n": 0}

    def flaky_builder(cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("engine download failed")
        return _FakeDistanceAdapter()

    monkeypatch.setattr(obstacle_plugin, "_build_distance_adapter", flaky_builder)
    executor = _FakeExecutor()
    plugin = obstacle_plugin.ObstacleDistancePlugin({"provider": "local"}, executor)
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert _wait_until(
        lambda: plugin.dispatch("obstacle", {"action": "info"})["state"] == "error"
    )
    info = plugin.dispatch("obstacle", {"action": "info"})
    assert "engine download failed" in info["instances"]["a"].get("error", "")
    plugin.dispatch("obstacle", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert _wait_until(
        lambda: plugin.dispatch("obstacle", {"action": "info"})["state"] == "running"
    )
    plugin.dispatch("obstacle", {"action": "stop"})


def test_obstacle_model_dir_confined_to_models_tree(monkeypatch):
    """MCP-supplied model_dir cannot point the root-run downloader outside
    /models (bot P2): the wrapper rejects out-of-tree paths before download."""
    import utils.model_downloader as md
    with pytest.raises(ValueError):
        md.ensure_obstacle_models("/etc/cron.d")
    with pytest.raises(ValueError):
        md.ensure_obstacle_models("/models/../root")


# ── config surface (培育→配置) ───────────────────────────────────────────────

def test_obstacle_config_schema_exposes_only_operator_fields():
    """The config UI renders configSchema verbatim. provider (one valid
    value) and the expert tuning blocks live in config.yaml only; what
    remains are the two genuine operator decisions, and no object-typed
    property may appear (the frontend renders those as [object Object])."""
    from plugins.obstacle import TOOLS

    schema = TOOLS[0]["configSchema"]
    assert set(schema["properties"]) == {"fixed_scene", "decision_threshold_m"}
    for name, spec in schema["properties"].items():
        assert spec["type"] != "object", name
    assert "required" not in schema
    assert schema["properties"]["fixed_scene"]["enum"] == ["indoor", "vehicle"]


def test_obstacle_config_fields_reach_the_adapter(monkeypatch):
    """fixed_scene / decision_threshold_m set via the config action land in
    the adapter cfg on the next start (adapter rebuild path)."""
    plugin, executor, builder = _make_obstacle(monkeypatch)
    plugin.dispatch("obstacle", {
        "action": "config",
        "fixed_scene": "vehicle",
        "decision_threshold_m": 3.5,
    })
    _obstacle_start_and_wait(plugin, executor, "/cam/a")
    cfg = builder.built[-1].cfg
    assert cfg["fixed_scene"] == "vehicle"
    assert cfg["decision_threshold_m"] == 3.5
    plugin.dispatch("obstacle", {"action": "stop"})
