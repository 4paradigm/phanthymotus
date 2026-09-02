"""
tests/test_vits2_tts_plugin.py — VITS2 TTS lifecycle tests (host-side).

The point of these: the VITS2 model is a 60 MB download plus three TensorRT
engines plus a warmup pass, while Agent Core gives a processor tools/call 60 s
(agent-core/src/mcp_client.py). So `start` must never wait for the model, and
every path that abandons an utterance must still release its ACP action.

ROS stubs come from vision_stubs (installed by conftest before collection).
Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import threading
import time

import pytest

from vision_stubs import (  # noqa: F401
    _FakeExecutor,
    _FakeNode,
    _wait_until,
)

import plugins.vits2_tts_trt.plugin as vits2  # noqa: E402
from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter  # noqa: E402


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


def _plugin(monkeypatch, *, gated=False, fail=None):
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
        "loads": 0, "engine_dirs": [], "adapters": [],
        "release": gate.set, "entered": threading.Event(),
    }

    def fake_ensure(model_dir, family=None):
        state["loads"] += 1
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
        {"model_dir": "/models/vits2", "backend": "trt", "speed": 1.0}, executor
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

def test_config_speed_updates_a_resident_model_without_a_reload(monkeypatch):
    plugin, executor, state = _plugin(monkeypatch)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    node = executor.nodes[0]

    assert plugin.dispatch("tts", {"action": "config", "speed": 1.4}) == {
        "status": "configured"
    }
    assert _installed(plugin).speed == pytest.approx(1.4)
    # A slider change must not cost a 60 MB reload or drop the live node.
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
    assert config["tts_engine"]["enum"] == ["vits2_trt", "sherpa_onnx"]
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

    def explode(model_dir, family=None):
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
