"""
tests/test_tts_engine_switch.py — the public tts tool's engine facade.

PR #112 introduced a second TTS implementation but left the engine selectable
only through the image's config.yaml, so the dashboard could neither show nor
change it. These tests pin the behaviour that fixes that: one tool, an engine
field in configSchema, and a switch that disposes the outgoing engine's nodes
before the incoming one publishes on the same topics.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import threading
import time

import pytest

from vision_stubs import _FakeExecutor, _wait_until  # noqa: F401

import plugins.tts as tts  # noqa: E402


class _FakeEngine:
    """Stands in for one engine implementation behind the facade."""

    def __init__(self, name, cfg, executor, record, delay=0.0):
        self.name = name
        self.cfg = dict(cfg)
        self.executor = executor
        self.calls = []
        self.stopped = False
        if delay:
            time.sleep(delay)
        record(self)

    def get_tools(self):
        return tts.TOOLS

    def dispatch(self, name, args):
        action = args.get("action")
        self.calls.append(args)
        if action == "stop":
            self.stopped = True
            return {"state": "idle"}
        if action == "info":
            return {"name": "TTS", "model": self.name, "state": "idle"}
        if action == "config":
            return {"status": "configured", "applied": dict(args)}
        return {"state": "running", "engine_seen": self.name}

    def synthesize_raw(self, text):
        return f"{self.name}:{text}".encode()


class _Engines:
    """Per-test record of what the facade built.

    Deliberately not class-level state on _FakeEngine: a switch left in flight
    by an earlier test would append to a shared list after the next test had
    cleared it, and that pollution looked exactly like a double build.
    """

    def __init__(self):
        self.order = []
        self.impls = {}

    def add(self, impl):
        self.order.append(impl.name)
        self.impls[impl.name] = impl

    def __contains__(self, name):
        return name in self.impls

    def __getitem__(self, name):
        return self.impls[name]


@pytest.fixture(autouse=True)
def _fake_engines(monkeypatch):
    """Replace both real engines with fakes; sherpa's is deliberately slow."""
    engines = _Engines()

    # Patch the two engine constructors, not TTSPlugin._build: the facade's own
    # per-engine model_dir selection and post-construction health check live in
    # _build, and stubbing it out would skip exactly what these tests check.
    # sherpa-onnx really does fetch its Matcha model in __init__, hence the delay.
    monkeypatch.setattr(
        tts, "SherpaOnnxTTSPlugin",
        lambda cfg, executor: _FakeEngine("sherpa_onnx", cfg, executor,
                                          engines.add, delay=0.3),
    )
    monkeypatch.setattr(
        tts.TTSPlugin, "_build_vits2",
        lambda self, cfg: _FakeEngine("vits2_trt", cfg, self._executor, engines.add),
    )
    return engines


def _plugin(**cfg):
    return tts.TTSPlugin({"engine": "vits2_trt", **cfg}, _FakeExecutor())


def test_config_schema_exposes_the_engine_selector():
    props = tts.TOOLS[0]["configSchema"]["properties"]
    assert props["tts_engine"]["enum"] == list(tts.TTS_ENGINES)
    assert props["tts_engine"]["default"] == tts.DEFAULT_TTS_ENGINE
    # One public tool, whatever the engine.
    assert [t["name"] for t in tts.TOOLS] == ["tts"]
    assert tts.TTSPlugin.PREFIX == "tts"


def test_default_engine_is_built_at_startup(_fake_engines):
    plugin = _plugin()
    assert _fake_engines.order == ["vits2_trt"]
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2_trt"


def test_actions_are_forwarded_to_the_active_engine(_fake_engines):
    plugin = _plugin()
    result = plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert result["engine_seen"] == "vits2_trt"
    assert _fake_engines["vits2_trt"].calls[-1]["input_topic"] == "/say"


def test_switch_stops_the_old_engine_and_waits_for_the_new_one(_fake_engines):
    """config waits for the bounded part of a build instead of making the caller
    poll. Answering `loading` for a 5 s session construction is what made the
    dashboard send a start the engine could not honour — see the deferred-start
    tests at the bottom for the case where waiting is not enough."""
    plugin = _plugin()
    outgoing = _fake_engines["vits2_trt"]

    result = plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})

    assert result["status"] == "configured"
    assert "state" not in result, "a completed switch must not report loading"
    assert result["engine"] == "sherpa_onnx"
    # The outgoing engine is stopped before the new one can publish.
    assert outgoing.stopped is True
    # And the engine is live *now*, so the start that follows lands on it.
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "sherpa_onnx"
    assert plugin.dispatch("tts", {"action": "start",
                                   "input_topic": "/say"})["state"] == "running"


def test_config_gives_up_waiting_and_reports_loading(monkeypatch, _fake_engines):
    """A build slower than the bound — a cold model download — still goes async."""
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    result = _plugin().dispatch("tts", {"action": "config",
                                        "tts_engine": "sherpa_onnx"})
    assert result["status"] == "configured"
    assert result["state"] == "loading"


def test_config_reports_a_build_failure_instead_of_loading(monkeypatch):
    class _Boom:
        def __init__(self, cfg, executor):
            raise RuntimeError("Protobuf parsing failed")

    monkeypatch.setattr(tts, "SherpaOnnxTTSPlugin", _Boom)
    monkeypatch.setattr(
        tts.TTSPlugin, "_build_vits2",
        lambda self, cfg: _FakeEngine("vits2_trt", cfg, self._executor, lambda i: None),
    )
    plugin = tts.TTSPlugin({"engine": "vits2_trt"}, _FakeExecutor())
    result = plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    assert result["status"] == "error"
    assert "Protobuf parsing failed" in result["message"]


def test_info_reports_loading_while_the_new_engine_builds(monkeypatch, _fake_engines):
    """Only reachable once config has stopped waiting; until then there is no
    window in which the facade has no engine."""
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})

    info = plugin.dispatch("tts", {"action": "info"})
    assert info["state"] == "loading"
    assert "sherpa_onnx" in info["desc"]
    # Other actions answer loading too, rather than hanging or lying.
    assert plugin.dispatch("tts", {"action": "speak", "text": "hi"})["state"] == "loading"


def test_switching_back_and_forth_keeps_one_engine_live(_fake_engines):
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    assert _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] != "loading"
    )
    sherpa = _fake_engines["sherpa_onnx"]

    plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2_trt"})
    assert sherpa.stopped is True
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] != "loading"
    )
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2_trt"
    assert _fake_engines.order == ["vits2_trt", "sherpa_onnx", "vits2_trt"]


def test_reconfiguring_the_same_engine_does_not_rebuild(_fake_engines):
    plugin = _plugin()
    result = plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2_trt",
                                     "speed": 1.3})
    assert _fake_engines.order == ["vits2_trt"], "same engine was rebuilt"
    # tts_engine is the facade's own field and must not be forwarded as if it
    # were an engine parameter; speed must be.
    applied = result["applied"]
    assert "tts_engine" not in applied and applied["speed"] == 1.3


def test_shared_config_survives_an_engine_switch(_fake_engines):
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "speed": 0.7})
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    assert _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    assert _fake_engines["sherpa_onnx"].cfg["speed"] == 0.7


def test_unknown_engine_is_refused_without_touching_the_live_one(_fake_engines):
    plugin = _plugin()
    with pytest.raises(ValueError):
        plugin.dispatch("tts", {"action": "config", "tts_engine": "espeak"})
    assert _fake_engines["vits2_trt"].stopped is False
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2_trt"


def test_engine_build_failure_is_reported_not_raised(monkeypatch):
    def boom(self, cfg):
        raise RuntimeError("no TensorRT here")

    monkeypatch.setattr(tts.TTSPlugin, "_build_vits2", boom)
    plugin = _plugin()          # must not raise: main.py keeps the tool listed
    info = plugin.dispatch("tts", {"action": "info"})
    assert info["state"] == "error"
    assert "no TensorRT here" in info["error"]
    assert plugin.dispatch("tts", {"action": "start"})["state"] == "error"
    with pytest.raises(RuntimeError):
        plugin.synthesize_raw("hi")


def test_concurrent_switches_leave_exactly_one_engine_live(_fake_engines):
    plugin = _plugin()
    targets = ["sherpa_onnx", "vits2_trt", "sherpa_onnx", "vits2_trt"]
    threads = [
        threading.Thread(
            target=lambda e=e: plugin.dispatch(
                "tts", {"action": "config", "tts_engine": e}
            )
        )
        for e in targets
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"}).get("state") != "loading",
        timeout=5.0,
    )
    info = plugin.dispatch("tts", {"action": "info"})
    assert info["engine"] == info["model"]
    # Every engine built but superseded must have been stopped, so no orphan
    # publisher survives on the shared topic.
    live = info["model"]
    for name, impl in _fake_engines.impls.items():
        if name != live:
            assert impl.stopped is True, f"{name} left running"


# ── per-engine model_dir (device regression) ─────────────────────────────────

def test_each_engine_gets_its_own_model_dir(_fake_engines):
    """config.yaml carries one model_dir, written for the engine it declares.

    Handing /models/vits2 to sherpa-onnx made it download its Matcha model into
    the VITS2 directory and then load the vocoder from there — which is what
    "I picked sherpa_onnx and it downloaded the VITS2 model" looked like.
    """
    plugin = tts.TTSPlugin(
        {"engine": "vits2_trt", "model_dir": "/models/vits2"}, _FakeExecutor()
    )
    assert _fake_engines["vits2_trt"].cfg["model_dir"] == "/models/vits2"

    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    assert _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    assert (_fake_engines["sherpa_onnx"].cfg["model_dir"]
            == tts.ENGINE_MODEL_DIRS["sherpa_onnx"])


def test_configured_model_dir_follows_the_configured_engine(_fake_engines):
    """A sherpa-configured deployment keeps its own path, and VITS2 gets its own."""
    plugin = tts.TTSPlugin(
        {"engine": "sherpa_onnx", "model_dir": "/models/custom/sherpa"},
        _FakeExecutor(),
    )
    assert _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    assert _fake_engines["sherpa_onnx"].cfg["model_dir"] == "/models/custom/sherpa"

    plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2_trt"})
    assert _wait_until(lambda: "vits2_trt" in _fake_engines)
    assert (_fake_engines["vits2_trt"].cfg["model_dir"]
            == tts.ENGINE_MODEL_DIRS["vits2_trt"])


def test_engine_that_reports_error_after_construction_is_not_installed(monkeypatch):
    """sherpa swallows its own model-load failure and reports it through info.

    Installing such an object made the facade claim ready, so start and speak
    "succeeded" against a model that never loaded.
    """
    class _BrokenEngine:
        def dispatch(self, name, args):
            if args.get("action") == "info":
                return {"state": "error", "error": "Protobuf parsing failed"}
            return {"state": "running"}

        def synthesize_raw(self, text):
            raise RuntimeError("no model")

    monkeypatch.setattr(tts, "SherpaOnnxTTSPlugin",
                        lambda cfg, executor: _BrokenEngine())
    plugin = tts.TTSPlugin({"engine": "sherpa_onnx"}, _FakeExecutor())

    info = plugin.dispatch("tts", {"action": "info"})
    assert info["state"] == "error"
    assert "Protobuf parsing failed" in info["error"]
    # And no action may claim success against it.
    assert plugin.dispatch("tts", {"action": "start"})["state"] == "error"
    assert plugin.dispatch("tts", {"action": "speak", "text": "hi"})["state"] == "error"


# ── starts that arrive mid-build ─────────────────────────────────────────────
#
# The dashboard sends config (which triggers the switch) and start back to back,
# so a start during a build is the normal path, not a rare race. Answering
# `state: loading` and dropping it left the engine idle once it finished, and
# Agent Core — which polls `info` after a loading start and reports "启动已取消"
# if it ever sees idle — cancelled the card even though the engine loaded fine.

def test_start_during_a_switch_is_replayed_once_the_engine_is_up(monkeypatch, _fake_engines):
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})

    # The build is still in flight, so the call cannot start anything yet.
    result = plugin.dispatch("tts", {"action": "start",
                                     "instance_id": "card-1",
                                     "input_topic": "/say"})
    assert result["state"] == "loading"

    _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    incoming = _fake_engines["sherpa_onnx"]
    _wait_until(lambda: any(c.get("action") == "start" for c in incoming.calls))

    started = [c for c in incoming.calls if c.get("action") == "start"]
    assert len(started) == 1, "the deferred start must be replayed exactly once"
    assert started[0]["input_topic"] == "/say"
    assert started[0]["instance_id"] == "card-1"


def test_a_stop_during_a_switch_cancels_the_deferred_start(monkeypatch, _fake_engines):
    """Otherwise the node reappears after the operator asked for it to stop."""
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    plugin.dispatch("tts", {"action": "start", "instance_id": "card-1",
                            "input_topic": "/say"})
    assert plugin.dispatch("tts", {"action": "stop",
                                   "instance_id": "card-1"})["state"] == "idle"

    _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    incoming = _fake_engines["sherpa_onnx"]
    time.sleep(0.4)   # past the fake build delay, so a replay would have landed
    assert not [c for c in incoming.calls if c.get("action") == "start"]


def test_deferred_starts_are_per_instance(monkeypatch, _fake_engines):
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    for card, topic in (("card-1", "/say/a"), ("card-2", "/say/b")):
        plugin.dispatch("tts", {"action": "start", "instance_id": card,
                                "input_topic": topic})

    _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    incoming = _fake_engines["sherpa_onnx"]
    _wait_until(lambda: len([c for c in incoming.calls
                             if c.get("action") == "start"]) == 2)
    topics = sorted(c["input_topic"] for c in incoming.calls
                    if c.get("action") == "start")
    assert topics == ["/say/a", "/say/b"]


def test_start_after_the_build_finishes_is_not_replayed_twice(_fake_engines):
    """A start that the live engine already handled must not also be queued."""
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    _wait_until(lambda: plugin.dispatch("tts", {"action": "info"})["state"] != "loading")

    plugin.dispatch("tts", {"action": "start", "instance_id": "card-1",
                            "input_topic": "/say"})
    time.sleep(0.2)
    incoming = _fake_engines["sherpa_onnx"]
    assert len([c for c in incoming.calls if c.get("action") == "start"]) == 1
