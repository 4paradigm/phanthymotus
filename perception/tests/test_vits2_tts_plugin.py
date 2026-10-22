"""
tests/test_vits2_tts_plugin.py — VITS2 TTS lifecycle tests (host-side).

The point of these: the VITS2 model is a 160 MB download plus three TensorRT
engines plus a warmup pass, while Agent Core gives a processor tools/call 60 s
(agent-core/src/mcp_client.py). So `start` must never wait for the model, and
every path that abandons an utterance must still release its ACP action.

ROS stubs come from vision_stubs (installed by conftest before collection).
Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from vision_stubs import (  # noqa: F401
    _FakeExecutor,
    _FakeNode,
    _wait_until,
)

import plugins.vits2_tts_trt.plugin as vits2  # noqa: E402
from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter  # noqa: E402
# The engine list is owned by plugins/tts.py and imported by this package rather
# than restated, so assert against the source of truth instead of a copy.
import plugins.tts as tts  # noqa: E402


class _FakeAdapter(Vits2TensorRTAdapter):
    """A Vits2TensorRTAdapter that synthesizes without TensorRT.

    Subclassed rather than duck-typed because the plugin asserts the adapter is
    a Vits2TensorRTAdapter before installing it — that isinstance check is the
    guard against silently running some other backend.
    """

    def __init__(self):
        self.speed = 1.0
        self.spoken = []
        self.warmups = 0

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)

    def warmup(self) -> int:
        self.warmups += 1
        return 3200

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        self.spoken.append(text)
        yield b"\x00\x01" * 8


@pytest.fixture(autouse=True)
def _fast_audio_gate(monkeypatch):
    """Drop the DDS subscriber-settle wait; the fake publisher is always matched."""
    monkeypatch.setattr(vits2, "SUBSCRIBER_SETTLE_MS", 0)
    monkeypatch.setattr(vits2, "SUBSCRIBER_WAIT_MS", 100)
    monkeypatch.setattr(vits2, "FRAME_INTERVAL_MS", 0)


@pytest.fixture
def completions(monkeypatch):
    """Capture ACP completion callbacks instead of POSTing to Agent Core."""
    seen = []
    monkeypatch.setattr(
        vits2, "_complete_action",
        lambda action_id, text, frames, interrupted: seen.append(
            (action_id, text, frames, interrupted)
        ),
    )
    return seen


def _installed(plugin):
    """The adapter the plugin actually committed, or None while loading."""
    return plugin._adapter


def _plugin(monkeypatch, *, gated=False, fail=None, cfg=None):
    """Build a plugin whose loader is instrumented instead of touching a GPU.

    `gated=True` blocks every load inside the model install until the test calls
    `state["release"]()`. That is what makes "while the model is loading"
    deterministic — a sleep only makes it likely, and a scheduler hiccup on a
    busy machine then turns the interleaving the test means to exercise into a
    coin flip.

    Each build returns its own adapter, recorded in state["adapters"] in
    completion order. One shared adapter would have let a discarded stale loader
    mutate the object under assertion — a test artifact, not a product bug: the
    plugin never installs a superseded adapter.
    """
    gate = threading.Event()
    state = {
        "loads": 0, "engine_dirs": [], "adapters": [], "callbacks": [], "base_urls": [],
        "release": gate.set, "entered": threading.Event(),
    }

    def fake_ensure(model_dir, family=None, progress_cb=None, stage_cb=None, base_url=None):
        state["loads"] += 1
        state["base_urls"].append(base_url)
        state["callbacks"].append((progress_cb, stage_cb))
        state["entered"].set()
        if gated and not gate.wait(10):
            raise RuntimeError("test never released the load gate")
        if fail is not None and state["loads"] <= fail:
            raise RuntimeError("release download failed")
        return f"{model_dir}/engines/jp61"

    def fake_build(cfg):
        state["engine_dirs"].append(cfg.get("engine_dir"))
        built = _FakeAdapter()
        built.set_speed(float(cfg.get("speed", 1.0)))
        state["adapters"].append(built)
        return built

    import utils.model_downloader as md
    monkeypatch.setattr(md, "ensure_vits2_model", fake_ensure, raising=False)
    monkeypatch.setattr(vits2, "build_adapter", fake_build)

    executor = _FakeExecutor()
    plugin = vits2.TTSPlugin(
        {"model_dir": "/models/vits2", "backend": "trt", "speed": 1.0,
         **(cfg or {})}, executor
    )
    return plugin, executor, state


# ── start never blocks on the model ──────────────────────────────────────────

def test_start_returns_loading_without_waiting_for_the_model(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch, gated=True)

    began = time.monotonic()
    result = plugin.dispatch("tts", {"action": "start", "input_topic": "/say",
                                     "instance_id": "a"})
    elapsed = time.monotonic() - began

    assert result["state"] == "loading"
    assert elapsed < 0.2, f"start blocked for {elapsed:.2f}s"
    assert executor.nodes == [], "a node was created before the model was ready"
    # ...and the instance comes up on its own once the load finishes.
    state["release"]()
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    assert state["loads"] == 1
    assert state["engine_dirs"] == ["/models/vits2/engines/jp61"]


def test_concurrent_starts_load_once_and_yield_one_node_each(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch, gated=True)

    results = []
    threads = [
        threading.Thread(
            target=lambda i=i: results.append(
                plugin.dispatch("tts", {"action": "start",
                                        "input_topic": f"/say{i}",
                                        "instance_id": f"i{i}"})
            )
        )
        for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r["state"] == "loading" for r in results)
    state["release"]()
    assert _wait_until(lambda: len(executor.nodes) == 6)
    assert state["loads"] == 1, f"downloaded {state['loads']} times"


def test_info_reports_loading_for_instance_and_aggregate(monkeypatch):
    plugin, _, state = _plugin(monkeypatch, gated=True)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say",
                            "instance_id": "a"})

    # Regression: the instance branch used to answer "idle" mid-download, so the
    # dashboard showed an idle device that refused to speak.
    per_instance = plugin.dispatch("tts", {"action": "info", "instance_id": "a"})
    aggregate = plugin.dispatch("tts", {"action": "info"})
    assert per_instance["state"] == "loading"
    assert aggregate["state"] == "loading"
    assert "initializing" in per_instance["desc"]

    state["release"]()
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "running"
    )


def test_info_never_triggers_a_load(monkeypatch):
    plugin, _, state = _plugin(monkeypatch)
    for _ in range(3):
        assert plugin.dispatch("tts", {"action": "info"})["state"] == "idle"
    assert state["loads"] == 0


def test_loading_info_reports_download_progress_and_extraction(monkeypatch):
    plugin, _, state = _plugin(monkeypatch, gated=True)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say",
                            "instance_id": "a"})
    try:
        assert state["entered"].wait(3)
        progress, stage = state["callbacks"][0]
        assert callable(progress) and callable(stage)
        progress(25, 40, 160)
        for args in ({"action": "info"}, {"action": "info", "instance_id": "a"}):
            info = plugin.dispatch("tts", args)
            assert info["state"] == "loading"
            assert "25% (40/160 MB)" in info["desc"]
        stage("extract")
        for args in ({"action": "info"}, {"action": "info", "instance_id": "a"}):
            info = plugin.dispatch("tts", args)
            assert "正在解压模型 'VITS2'" in info["desc"]
            assert "%" not in info["desc"]
    finally:
        state["release"]()
    assert _wait_until(lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "running")
    assert plugin.dispatch("tts", {"action": "info"})["desc"] == plugin.DESC


def test_loading_info_advances_to_initialization_and_warmup(monkeypatch):
    # A verified cache hit emits no download callbacks. Slow initialization and
    # warmup still need observable stages, with info responsive from this thread.
    plugin, _, _ = _plugin(monkeypatch)
    build_entered, build_release = threading.Event(), threading.Event()
    warmup_entered, warmup_release = threading.Event(), threading.Event()

    class GatedAdapter(_FakeAdapter):
        def warmup(self):
            warmup_entered.set()
            if not warmup_release.wait(10):
                raise RuntimeError("test never released warmup")
            return super().warmup()

    def build(cfg):
        build_entered.set()
        if not build_release.wait(10):
            raise RuntimeError("test never released initialization")
        return GatedAdapter()

    monkeypatch.setattr(vits2, "build_adapter", build)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say", "instance_id": "a"})
    try:
        assert build_entered.wait(3)
        for args in ({"action": "info"}, {"action": "info", "instance_id": "a"}):
            assert plugin.dispatch("tts", args)["desc"] == "Initializing the VITS2 TensorRT model..."
        build_release.set()
        assert warmup_entered.wait(3)
        for args in ({"action": "info"}, {"action": "info", "instance_id": "a"}):
            assert plugin.dispatch("tts", args)["desc"] == "Warming up the VITS2 TensorRT model..."
    finally:
        build_release.set()
        warmup_release.set()
    assert _wait_until(lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "running")
    assert plugin.dispatch("tts", {"action": "info"})["desc"] == plugin.DESC


def test_stale_download_callbacks_cannot_replace_current_status(monkeypatch):
    plugin, _, state = _plugin(monkeypatch, gated=True)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say", "instance_id": "a"})
    try:
        assert state["entered"].wait(3)
        old_progress, old_stage = state["callbacks"][0]
        old_progress(80, 128, 160)
        plugin.dispatch("tts", {"action": "config", "speed": 0.8})
        assert _wait_until(lambda: len(state["callbacks"]) == 2)
        progress, _ = state["callbacks"][1]
        progress(10, 16, 160)
        old_progress(99, 159, 160)
        old_stage("extract")
        assert "10% (16/160 MB)" in plugin.dispatch("tts", {"action": "info"})["desc"]
    finally:
        state["release"]()
    assert _wait_until(lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "running")
    progress(100, 160, 160)
    old_stage("extract")
    assert plugin.dispatch("tts", {"action": "info"})["desc"] == plugin.DESC


# ── failure and retry ────────────────────────────────────────────────────────

def test_load_failure_surfaces_then_next_start_retries(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch, fail=1)

    first = plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert first["state"] == "loading"
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "error"
    )
    info = plugin.dispatch("tts", {"action": "info"})
    assert "release download failed" in info["error"]
    progress, stage = state["callbacks"][0]
    progress(50, 80, 160)
    stage("extract")
    assert plugin.dispatch("tts", {"action": "info"})["desc"] == "Model load failed: release download failed"

    # A retry is not blocked by the sticky error, and it succeeds.
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    assert state["loads"] == 2


def test_load_failure_releases_queued_speak_actions(monkeypatch, completions):
    plugin, _, state = _plugin(monkeypatch, fail=1, gated=True)

    queued = plugin.dispatch("tts", {"action": "speak", "text": "你好"})
    assert queued["status"] == "queued"
    action_id = queued["action_id"]

    # The ACP barrier waits for this action; a failed load must still end it.
    state["release"]()
    assert _wait_until(lambda: any(c[0] == action_id for c in completions))
    assert [c for c in completions if c[0] == action_id][0][3] is True


# ── stop / interrupt while still loading ─────────────────────────────────────

def test_stop_during_loading_cancels_the_pending_instance(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch, gated=True)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say",
                            "instance_id": "a"})
    assert state["entered"].wait(3), "the loader never started"
    assert plugin.dispatch("tts", {"action": "stop", "instance_id": "a"}) == {
        "state": "idle"
    }

    # Only now may the load finish: it must find nothing left to bring up.
    state["release"]()
    time.sleep(0.3)
    assert executor.nodes == [], "cancelled instance was started anyway"
    assert plugin.dispatch("tts", {"action": "info"})["state"] == "idle"


def test_stop_during_loading_releases_queued_speak(monkeypatch, completions):
    plugin, executor, state = _plugin(monkeypatch, gated=True)

    queued = plugin.dispatch("tts", {"action": "speak", "text": "取消我"})
    assert state["entered"].wait(3), "the loader never started"
    plugin.dispatch("tts", {"action": "stop"})

    assert _wait_until(lambda: any(c[0] == queued["action_id"] for c in completions))
    assert [c for c in completions if c[0] == queued["action_id"]][0][3] is True
    state["release"]()
    time.sleep(0.3)
    assert executor.nodes == []


# ── speak queued before the model is resident ────────────────────────────────

def test_speak_before_ready_plays_once_the_model_lands(monkeypatch, completions):
    plugin, executor, state = _plugin(monkeypatch, gated=True)

    queued = plugin.dispatch("tts", {"action": "speak", "text": "延迟播报"})
    assert queued["action_id"].startswith("speak-")
    assert executor.nodes == [], "the utterance was not queued behind the load"

    state["release"]()
    assert _wait_until(lambda: any(a.spoken == ["延迟播报"] for a in state["adapters"]))
    assert _wait_until(lambda: any(c[0] == queued["action_id"] for c in completions))
    # Completed, not cancelled, and the utterance reached a real node.
    assert [c for c in completions if c[0] == queued["action_id"]][0][3] is False
    assert executor.nodes and executor.nodes[0].state == "running"


def test_speak_after_ready_reuses_the_running_node(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    plugin.dispatch("tts", {"action": "speak", "text": "第一句"})
    plugin.dispatch("tts", {"action": "speak", "text": "第二句"})

    assert _wait_until(
        lambda: _installed(plugin) and _installed(plugin).spoken == ["第一句", "第二句"]
    )
    assert len(executor.nodes) == 1
    assert state["loads"] == 1


# ── interrupt ────────────────────────────────────────────────────────────────

def _running(monkeypatch):
    """A plugin with one node up and its committed adapter."""
    plugin, executor, state = _plugin(monkeypatch)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say",
                            "instance_id": "a"})
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    return plugin, executor, state


def _record(completions, action_id):
    matches = [c for c in completions if c[0] == action_id]
    assert matches, f"no ACP completion for {action_id}"
    return matches[0]


def test_interrupt_while_idle_does_not_swallow_the_next_speak(
    monkeypatch, completions
):
    """Regression, observed on R1: "别说了，说说现在几点了" stopped the robot but
    the answer was never heard.

    Two interrupts land back to back in the real sequence — agent-core's barge-in
    fallback fires one, then the LLM calls tts(action=interrupt) itself a second
    or two later. The second lands with nothing playing, and the interrupt flag
    used to be consumed by the *next* dequeued utterance rather than by the
    utterance being interrupted. So it survived and discarded the reply: the ACP
    action completed within the same second with zero frames, the barrier cleared
    happily, and no synthesis ever ran.
    """
    plugin, _, _ = _running(monkeypatch)
    adapter = _installed(plugin)

    plugin.dispatch("tts", {"action": "interrupt"})
    plugin.dispatch("tts", {"action": "interrupt", "instance_id": "a"})

    queued = plugin.dispatch("tts", {"action": "speak", "text": "现在是下午6点36分。",
                                     "instance_id": "a"})
    action_id = queued["action_id"]

    assert _wait_until(lambda: any(c[0] == action_id for c in completions))
    _aid, _text, frames, interrupted = _record(completions, action_id)
    assert interrupted is False, "the reply after an idle interrupt was cancelled"
    assert frames > 0, "the reply completed without publishing any audio"
    assert "现在是下午6点36分。" in adapter.spoken


def test_interrupt_still_drops_utterances_queued_before_it(monkeypatch, completions):
    """The flip side: an interrupt must cancel what was already queued.

    Guards the generation counter from the trivial "never discard anything" fix.
    """
    plugin, _, _ = _running(monkeypatch)
    adapter = _installed(plugin)

    # Hold the first utterance inside synthesis so the second stays queued and
    # the interrupt lands with one playing and one waiting.
    gate = threading.Event()
    started = threading.Event()
    real_stream = adapter.synthesize_stream

    def gated_stream(text):
        if text == "第一句":
            started.set()
            assert gate.wait(10), "test never released the synthesis gate"
        yield from real_stream(text)

    adapter.synthesize_stream = gated_stream

    first = plugin.dispatch("tts", {"action": "speak", "text": "第一句",
                                    "instance_id": "a"})
    assert started.wait(5), "the first utterance never reached synthesis"
    second = plugin.dispatch("tts", {"action": "speak", "text": "第二句",
                                     "instance_id": "a"})

    plugin.dispatch("tts", {"action": "interrupt", "instance_id": "a"})
    gate.set()

    assert _wait_until(lambda: any(c[0] == second["action_id"] for c in completions))
    assert _record(completions, second["action_id"])[3] is True, \
        "an utterance queued before the interrupt was played anyway"
    assert "第二句" not in adapter.spoken
    assert _wait_until(lambda: any(c[0] == first["action_id"] for c in completions))
    assert _record(completions, first["action_id"])[3] is True

    # ...and the node is still usable afterwards.
    third = plugin.dispatch("tts", {"action": "speak", "text": "第三句",
                                    "instance_id": "a"})
    assert _wait_until(lambda: any(c[0] == third["action_id"] for c in completions))
    assert _record(completions, third["action_id"])[3] is False
    assert "第三句" in adapter.spoken


# ── config ───────────────────────────────────────────────────────────────────

def test_prepare_config_loads_and_warms_without_starting_a_node(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch, cfg={"prepare_on_config": True})

    result = plugin.dispatch("tts", {"action": "config"})

    assert result == {"status": "configured", "model_state": "ready"}
    assert state["loads"] == 1
    assert _installed(plugin).warmups == 1
    assert executor.nodes == []
    started = plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert started["state"] == "running"
    assert state["loads"] == 1


def test_prepare_config_is_bounded_and_concurrent_calls_share_one_loader(monkeypatch):
    monkeypatch.setattr(vits2, "_CONFIG_PREPARE_WAIT_S", 0.03)
    plugin, executor, state = _plugin(
        monkeypatch, gated=True, cfg={"prepare_on_config": True}
    )
    results = []
    began = time.monotonic()
    callers = [threading.Thread(target=lambda: results.append(
        plugin.dispatch("tts", {"action": "config"})
    )) for _ in range(3)]
    try:
        for caller in callers:
            caller.start()
        assert state["entered"].wait(3)
        # info stays available while config waits on the loader.
        assert plugin.dispatch("tts", {"action": "info"})["state"] == "loading"
        for caller in callers:
            caller.join(timeout=1)
        assert all(not caller.is_alive() for caller in callers)
        assert time.monotonic() - began < 1
        assert len(results) == 3
        assert all(r["status"] == "configured" and r["state"] == "loading"
                   for r in results)
        assert state["loads"] == 1
        assert plugin._load_generation == 0
        assert executor.nodes == []
    finally:
        state["release"]()
        for caller in callers:
            caller.join(timeout=1)
    assert _wait_until(lambda: _installed(plugin) is not None)


def test_prepare_config_updates_loading_speed_without_reloading(monkeypatch):
    monkeypatch.setattr(vits2, "_CONFIG_PREPARE_WAIT_S", 0.02)
    plugin, _, state = _plugin(
        monkeypatch, gated=True, cfg={"prepare_on_config": True}
    )
    try:
        assert plugin.dispatch("tts", {"action": "config"})["state"] == "loading"
        result = plugin.dispatch("tts", {"action": "config", "speed": 0.8})
        assert result["state"] == "loading"
        assert state["loads"] == 1
        assert plugin._load_generation == 0
    finally:
        state["release"]()
    assert _wait_until(lambda: _installed(plugin) is not None)
    assert _installed(plugin).speed == pytest.approx(0.8)
    for _ in range(3):
        assert plugin.dispatch("tts", {"action": "config"}) == {
            "status": "configured", "model_state": "ready"
        }
    assert state["loads"] == 1
    assert _installed(plugin).warmups == 1


def test_prepare_config_reports_a_load_error_and_can_retry(monkeypatch):
    plugin, _, state = _plugin(
        monkeypatch, fail=1, cfg={"prepare_on_config": True}
    )
    failed = plugin.dispatch("tts", {"action": "config"})
    assert failed["status"] == "error" and failed["state"] == "error"
    assert "release download failed" in failed["message"]
    assert plugin.dispatch("tts", {"action": "config"}) == {
        "status": "configured", "model_state": "ready"
    }
    assert state["loads"] == 2


def test_prepare_config_does_not_reapply_an_unchanged_resident_speed(monkeypatch):
    plugin, _, state = _plugin(monkeypatch, cfg={"prepare_on_config": True})
    plugin.dispatch("tts", {"action": "config"})
    adapter = _installed(plugin)
    applied = []
    original = adapter.set_speed

    def record(speed):
        applied.append(speed)
        original(speed)

    monkeypatch.setattr(adapter, "set_speed", record)
    plugin.dispatch("tts", {"action": "config"})
    plugin.dispatch("tts", {"action": "config", "speed": 1.0})
    assert applied == []
    plugin.dispatch("tts", {"action": "config", "speed": 0.8})
    assert applied == [0.8], "dispatch preparation must not apply the speed twice"
    assert plugin._cfg["speed"] == adapter.speed == 0.8
    assert state["loads"] == 1


def test_prepare_config_waiting_for_adapter_keeps_lifecycle_responsive(monkeypatch):
    plugin, _, _ = _plugin(monkeypatch, cfg={"prepare_on_config": True})
    plugin.dispatch("tts", {"action": "config"})
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say", "instance_id": "a"})
    adapter = _installed(plugin)
    entered, release, lifecycle_done = threading.Event(), threading.Event(), threading.Event()
    failures, results = [], []
    original = adapter.set_speed

    def blocked_speed(speed):
        entered.set()
        assert release.wait(3), "test did not release the busy adapter"
        original(speed)

    def configure():
        try:
            results.append(plugin.dispatch("tts", {"action": "config", "speed": 0.8}))
        except BaseException as error:
            failures.append(error)

    def lifecycle():
        try:
            assert plugin.dispatch("tts", {"action": "info"})["state"] == "running"
            assert plugin.dispatch("tts", {"action": "interrupt", "instance_id": "a"})["status"] == "interrupted"
            assert plugin.dispatch("tts", {"action": "stop", "instance_id": "a"})["state"] == "idle"
        except BaseException as error:
            failures.append(error)
        finally:
            lifecycle_done.set()

    monkeypatch.setattr(adapter, "set_speed", blocked_speed)
    config_thread = threading.Thread(target=configure)
    lifecycle_thread = threading.Thread(target=lifecycle)
    try:
        config_thread.start()
        assert entered.wait(1)
        assert plugin._cfg["speed"] == 1.0, "save the speed only after the adapter accepts it"
        lifecycle_thread.start()
        assert lifecycle_done.wait(2), "config held the lifecycle lock across adapter work"
        assert config_thread.is_alive(), "the adapter should still be busy"
    finally:
        release.set()
        config_thread.join(timeout=3)
        if lifecycle_thread.ident is not None:
            lifecycle_thread.join(timeout=3)
    assert failures == []
    assert not config_thread.is_alive() and not lifecycle_thread.is_alive()
    assert results == [{"status": "configured", "model_state": "ready"}]
    assert plugin._cfg["speed"] == adapter.speed == 0.8


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("speed", [0, -1, 4.1, float("nan"), float("inf")])
def test_prepare_config_invalid_speed_does_not_mutate_or_load(monkeypatch, resident, speed):
    plugin, _, state = _plugin(monkeypatch, cfg={"prepare_on_config": True})
    if resident:
        plugin.dispatch("tts", {"action": "config"})
    previous = dict(plugin._cfg)
    loads = state["loads"]
    with pytest.raises(ValueError, match="speed"):
        plugin.dispatch("tts", {"action": "config", "speed": speed})
    assert plugin._cfg == previous
    assert state["loads"] == loads
    if resident:
        assert _installed(plugin).speed == 1.0


def test_prepare_config_adapter_rejects_speed_without_saving_it(monkeypatch):
    plugin, _, _ = _plugin(monkeypatch, cfg={"prepare_on_config": True})
    plugin.dispatch("tts", {"action": "config"})
    adapter = _installed(plugin)

    def fail(speed):
        raise RuntimeError("adapter setting failed")

    monkeypatch.setattr(adapter, "set_speed", fail)
    with pytest.raises(RuntimeError, match="adapter setting failed"):
        plugin.dispatch("tts", {"action": "config", "speed": 0.8})
    assert plugin._cfg["speed"] == adapter.speed == 1.0
    assert plugin.dispatch("tts", {"action": "config"})["model_state"] == "ready"


def test_prepare_config_uses_deployment_source_and_zero_wait(monkeypatch):
    plugin, _, state = _plugin(monkeypatch, gated=True, cfg={
        "prepare_on_config": True, "prepare_timeout_s": 0,
        "model_base_url": "http://deployment-assets.example/vits2/",
    })
    try:
        result = plugin.dispatch("tts", {"action": "config"})
        assert result["state"] == "loading"
        assert state["entered"].wait(3)
        assert state["base_urls"] == ["http://deployment-assets.example/vits2/"]
    finally:
        state["release"]()
    assert _wait_until(lambda: _installed(plugin) is not None)


@pytest.mark.parametrize("timeout", [-1, 50.1, float("nan"), float("inf")])
def test_prepare_config_rejects_unbounded_wait(monkeypatch, timeout):
    plugin, _, state = _plugin(monkeypatch, cfg={
        "prepare_on_config": True, "prepare_timeout_s": timeout,
    })
    with pytest.raises(ValueError, match="prepare_timeout_s"):
        plugin.dispatch("tts", {"action": "config"})
    assert state["loads"] == 0


@pytest.mark.parametrize("frame_count", [5, 15])
def test_ten_frame_prebuffer_preserves_short_and_long_speech(monkeypatch, frame_count):
    monkeypatch.setattr(vits2, "PREBUFFER_FRAMES", 10)
    plugin, executor, state = _plugin(monkeypatch)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert _wait_until(lambda: bool(executor.nodes))
    node = executor.nodes[0]
    frames = [bytes([index + 1, 0]) * (vits2.CHUNK_BYTES // 2)
              for index in range(frame_count)]
    tail = b"\x01\x00" * 8
    entered, release = threading.Event(), threading.Event()

    def synthesize(text):
        split = min(frame_count - 1, 9)
        yield b"".join(frames[:split])
        entered.set()
        assert release.wait(3), "test did not release the remaining PCM"
        yield b"".join(frames[split:]) + tail

    monkeypatch.setattr(_installed(plugin), "synthesize_stream", synthesize)
    try:
        node.enqueue("完整播报")
        assert entered.wait(3)
        assert node._pub.messages == [], "audio escaped before the prebuffer filled"
        release.set()
        assert _wait_until(lambda: node._pub.messages
                           and bytes(node._pub.messages[-1]) == vits2.AUDIO_EOF_MAGIC)
        delivered = [bytes(data) for data in node._pub.messages[:-1]]
        assert b"".join(delivered) == b"".join(frames) + tail
        assert len(delivered) == frame_count + 1
    finally:
        release.set()
        plugin.dispatch("tts", {"action": "stop"})


def test_config_speed_updates_a_resident_model_without_a_reload(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    node = executor.nodes[0]

    assert plugin.dispatch("tts", {"action": "config", "speed": 1.4}) == {
        "status": "configured"
    }
    assert _installed(plugin).speed == pytest.approx(1.4)
    # A slider change must not cost a 160 MB reload or drop the live node.
    assert state["loads"] == 1
    assert executor.nodes == [node] and node.state == "running"


def test_config_during_loading_discards_the_stale_adapter(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch, gated=True)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert state["entered"].wait(3), "the loader never started"
    # config now provably lands mid-load, which is the case under test.
    plugin.dispatch("tts", {"action": "config", "speed": 0.8})
    state["release"]()

    # The in-flight loader was building from the old config, so its adapter must
    # be dropped — but the pending start survives and a fresh load serves it.
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running",
                       timeout=5.0)
    # Both loaders ran — one per config — and the one the plugin committed, and
    # that the live node actually uses, is the one built from the new config.
    # Asserted without indexing state["adapters"]: that list is ordered by build
    # completion, and the superseded loader may finish either first or last.
    assert state["loads"] == 2
    assert sorted(a.speed for a in state["adapters"]) == [
        pytest.approx(0.8), pytest.approx(1.0),
    ]
    assert _installed(plugin).speed == pytest.approx(0.8)
    assert executor.nodes[0]._adapter is _installed(plugin)
    assert len(executor.nodes) == 1


def test_speaker_id_other_than_zero_is_refused(monkeypatch):
    _plugin(monkeypatch)  # installs the fakes
    with pytest.raises(ValueError):
        vits2.TTSPlugin({"backend": "trt", "speaker_id": 3}, _FakeExecutor())
    with pytest.raises(ValueError):
        vits2.TTSPlugin({"backend": "onnx"}, _FakeExecutor())


# ── the public tool surface ──────────────────────────────────────────────────

def test_tool_is_the_standard_tts_tool_with_an_engine_selector():
    tools = vits2.TOOLS
    assert [t["name"] for t in tools] == ["tts"]
    config = tools[0]["configSchema"]["properties"]
    # The engine has to be visible in the device panel, or switching it means
    # rebuilding the image (see PR #112 review).
    assert config["tts_engine"]["enum"] == list(tts.TTS_ENGINES)
    assert vits2.TTSPlugin.PREFIX == "tts"


# ── ACP callback and error reporting (device regressions) ────────────────────

def test_acp_callback_tolerates_the_self_signed_agent_core_cert(monkeypatch):
    """Agent Core serves HTTPS with a self-signed cert.

    Without an unverified context every completion POST raised
    CERTIFICATE_VERIFY_FAILED, so no speak action ever completed and the ACP
    barrier waited out its full timeout on each utterance.
    """
    import ssl

    captured = {}

    def fake_urlopen(request, timeout=0, context=None):
        captured["url"] = request.full_url
        captured["context"] = context
        captured["body"] = request.data
        return _NullResponse()

    class _NullResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("AGENT_CORE_URL", "https://localhost:15678")

    vits2._complete_action("speak-1", "你好", 3, interrupted=False)

    assert captured["url"].endswith("/api/acp/complete")
    ctx = captured["context"]
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_NONE and ctx.check_hostname is False
    assert b'"status": "completed"' in captured["body"]


def test_load_error_keeps_the_underlying_cause(monkeypatch):
    """The dashboard must show why TensorRT was unavailable, not just that."""
    plugin, _, _ = _plugin(monkeypatch)

    import utils.model_downloader as md

    def explode(model_dir, family=None, progress_cb=None, stage_cb=None):
        try:
            raise ImportError(
                "libnvdla_compiler.so: cannot open shared object file: "
                "No such file or directory"
            )
        except ImportError as cause:
            raise RuntimeError("TensorRT is not available in this runtime") from cause

    monkeypatch.setattr(md, "ensure_vits2_model", explode, raising=False)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})

    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "error"
    )
    error = plugin.dispatch("tts", {"action": "info"})["error"]
    assert "TensorRT is not available" in error
    assert "libnvdla_compiler.so" in error


def test_info_while_loading_reports_the_pending_output_topic(monkeypatch):
    """The dashboard subscribes to whatever info() reports — it must be real.

    Agent Core's start sequencer calls info() with only instance_id, and while
    the model loads there is no node to ask. Falling back to /perception/tts
    registered the wrong topic on the bus, so the waveform stayed empty while
    audio flowed on <input_topic>/tts. Only reproducible on a cold start: with
    the engine resident, start returns running and info reads the live node.
    """
    plugin, executor, state = _plugin(monkeypatch, gated=True)
    plugin.dispatch("tts", {"action": "start", "instance_id": "card-x",
                            "input_topic": "/remote_control/message"})

    loading = plugin.dispatch("tts", {"action": "info", "instance_id": "card-x"})
    assert loading["state"] == "loading"
    assert [t["topic"] for t in loading["topic_out"]] == [
        "/remote_control/message/tts"
    ]
    assert [t["topic"] for t in loading["topic_in"]] == ["/remote_control/message"]

    # And the same topic once it is actually up.
    state["release"]()
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    running = plugin.dispatch("tts", {"action": "info", "instance_id": "card-x"})
    assert running["state"] == "running"
    assert [t["topic"] for t in running["topic_out"]] == [
        "/remote_control/message/tts"
    ]


# Use a separate copy of the real plugin for pacing tests. The autouse fixture
# above deliberately sets the ordinary module's interval to zero for lifecycle
# tests, which would otherwise hide regressions in live audio delivery.
def _pacing_plugin(monkeypatch):
    name = "plugins.vits2_tts_trt._pacing_test"
    spec = importlib.util.spec_from_file_location(name, Path(vits2.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _VirtualPacingClock:
    def __init__(self):
        self._lock = threading.Lock()
        self._now = 0.0
        self.on_sleep = None

    def monotonic(self):
        with self._lock:
            return self._now

    def advance(self, seconds):
        with self._lock:
            self._now += seconds

    def sleep(self, seconds):
        assert seconds >= 0
        with self._lock:
            self._now += seconds
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        # Give the synth and publisher threads a scheduling opportunity.
        threading.Event().wait(0)


class _PacingAdapter(_FakeAdapter):
    def __init__(self, payloads):
        super().__init__()
        self.payloads = payloads

    def synthesize_stream(self, text):
        self.spoken.append(text)
        yield self.payloads[text]


def _pcm_frame(module, value):
    return bytes((value, 0)) * (module.CHUNK_BYTES // 2)


def _pacing_node(monkeypatch, module, clock, payloads):
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    )
    monkeypatch.setattr(module, "SUBSCRIBER_SETTLE_MS", 0)
    monkeypatch.setattr(module, "FIRST_FRAME_DELAY_MS", 0)
    monkeypatch.setattr(module, "PREBUFFER_FRAMES", 5)
    monkeypatch.setattr(module, "MAX_BURST_FRAMES", 20)
    completions = []
    monkeypatch.setattr(
        module, "_complete_action",
        lambda action_id, text, frames, interrupted: completions.append(
            (action_id, text, frames, interrupted)
        ),
    )
    node = module._Vits2TTSNode(None, _PacingAdapter(payloads))
    monkeypatch.setattr(node, "_wait_for_audio_subscriber", lambda *_: (0.0, 0.0, 1))
    published = []
    original_publish = node._publish

    def record(pcm):
        published.append((clock.monotonic(), bytes(pcm)))
        original_publish(pcm)

    monkeypatch.setattr(node, "_publish", record)
    node.start()
    return node, published, completions


@pytest.mark.parametrize(
    ("allow_fast", "requested", "expected"),
    [
        (None, None, 100),
        (None, "70", 100),
        (None, "0", 100),
        ("false", "70", 100),
        ("1", "70", 70),
        ("1", "0", 0),
        (None, "150", 150),
        ("1", "bad", 100),
        ("1", "-1", 100),
        (None, "1001", 100),
    ],
)
def test_live_frame_interval_requires_explicit_fast_delivery(
    monkeypatch, allow_fast, requested, expected
):
    for name, value in (
        ("MIX_VITS_ALLOW_FAST_DELIVERY", allow_fast),
        ("MIX_VITS_FRAME_INTERVAL_MS", requested),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    module = _pacing_plugin(monkeypatch)
    assert module.PCM_FRAME_MS == pytest.approx(100)
    assert module.FRAME_INTERVAL_MS == pytest.approx(expected)


def test_live_pacing_preserves_pcm_and_eof_over_long_utterance(monkeypatch):
    monkeypatch.delenv("MIX_VITS_ALLOW_FAST_DELIVERY", raising=False)
    monkeypatch.delenv("MIX_VITS_FRAME_INTERVAL_MS", raising=False)
    module = _pacing_plugin(monkeypatch)
    clock = _VirtualPacingClock()
    frames = [_pcm_frame(module, i + 1) for i in range(80)]
    tail = b"\x01\x02" * 7
    payload = b"".join(frames) + tail
    node, published, completions = _pacing_node(
        monkeypatch, module, clock, {"long": payload}
    )
    try:
        node.enqueue("long", action_id="long-1")
        assert _wait_until(lambda: len(completions) == 1, timeout=5)
        assert completions == [("long-1", "long", 81, False)]
        assert [pcm for _, pcm in published[:-1]] == frames + [tail]
        assert published[-1][1] == module.AUDIO_EOF_MAGIC
        times = [at for at, _ in published[:-1]]
        assert times[:5] == pytest.approx([times[0]] * 5)
        assert times[5] - times[0] == pytest.approx(0.1)
        assert times[79] - times[0] == pytest.approx(7.5)
        assert times[80] - times[0] == pytest.approx(7.6)
    finally:
        node.stop()


def test_idle_refills_at_most_the_original_prebuffer(monkeypatch):
    monkeypatch.delenv("MIX_VITS_ALLOW_FAST_DELIVERY", raising=False)
    monkeypatch.delenv("MIX_VITS_FRAME_INTERVAL_MS", raising=False)
    module = _pacing_plugin(monkeypatch)
    clock = _VirtualPacingClock()
    node, published, completions = _pacing_node(
        monkeypatch,
        module,
        clock,
        {
            "first": _pcm_frame(module, 1) * 10,
            "after-idle": _pcm_frame(module, 2) * 12,
        },
    )
    try:
        node.enqueue("first", action_id="first-1")
        assert _wait_until(lambda: len(completions) == 1, timeout=5)
        clock.advance(10.0)
        node.enqueue("after-idle", action_id="second-1")
        assert _wait_until(lambda: len(completions) == 2, timeout=5)
        assert completions[-1] == ("second-1", "after-idle", 12, False)
        assert published[-1][1] == module.AUDIO_EOF_MAGIC
        second_times = [at for at, pcm in published if pcm == _pcm_frame(module, 2)]
        assert len(second_times) == 12
        assert sum(abs(at - second_times[0]) < 1e-9 for at in second_times) <= 5
        assert second_times[5] - second_times[0] == pytest.approx(0.1)
    finally:
        node.stop()


def test_interrupt_while_pacing_allows_next_utterance(monkeypatch):
    monkeypatch.delenv("MIX_VITS_ALLOW_FAST_DELIVERY", raising=False)
    monkeypatch.delenv("MIX_VITS_FRAME_INTERVAL_MS", raising=False)
    module = _pacing_plugin(monkeypatch)
    clock = _VirtualPacingClock()
    first = _pcm_frame(module, 1) * 12
    second = _pcm_frame(module, 2) * 2
    node, published, completions = _pacing_node(
        monkeypatch, module, clock, {"first": first, "second": second}
    )
    interrupted = threading.Event()

    def interrupt_once(seconds):
        if seconds > 0 and not interrupted.is_set():
            interrupted.set()
            node.interrupt()

    clock.on_sleep = interrupt_once
    try:
        node.enqueue("first", action_id="first-1")
        assert _wait_until(lambda: len(completions) == 1, timeout=5)
        assert interrupted.is_set()
        assert completions[0][0] == "first-1"
        assert completions[0][3] is True
        assert [pcm for _, pcm in published].count(module.AUDIO_EOF_MAGIC) == 1
        clock.on_sleep = None
        node.enqueue("second", action_id="second-1")
        assert _wait_until(lambda: len(completions) == 2, timeout=5)
        assert completions[1] == ("second-1", "second", 2, False)
        assert [pcm for _, pcm in published][-3:] == [
            _pcm_frame(module, 2),
            _pcm_frame(module, 2),
            module.AUDIO_EOF_MAGIC,
        ]
    finally:
        node.stop()
