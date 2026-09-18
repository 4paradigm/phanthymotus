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

import logging
import threading
import time

import pytest

from vision_stubs import _FakeExecutor, _wait_until  # noqa: F401

import plugins.tts as tts  # noqa: E402

# Captured before any fixture runs. The autouse _fake_engines fixture replaces
# tts.SherpaOnnxTTSPlugin with a stub, so the tests at the bottom that exercise the
# *real* plugin's config path have to hold their own reference to it.
_REAL_SHERPA_PLUGIN = tts.SherpaOnnxTTSPlugin


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
    # sherpa-onnx really does fetch its model in __init__, hence the delay.
    #
    # The fake takes its name from cfg["engine"] rather than a literal: matcha-zh-en
    # and mms-th are two different models behind this one constructor, so a
    # hardcoded name silently filed a Thai engine under "matcha-zh-en".
    monkeypatch.setattr(
        tts, "SherpaOnnxTTSPlugin",
        lambda cfg, executor: _FakeEngine(cfg.get("engine", "matcha-zh-en"), cfg, executor,
                                          engines.add, delay=0.3),
    )
    monkeypatch.setattr(
        tts.TTSPlugin, "_build_vits2",
        lambda self, cfg: _FakeEngine("vits2-zh-en", cfg, self._executor, engines.add),
    )
    return engines


def _plugin(**cfg):
    return tts.TTSPlugin({"engine": "vits2-zh-en", **cfg}, _FakeExecutor())


def test_config_schema_exposes_the_engine_selector():
    props = tts.TOOLS[0]["configSchema"]["properties"]
    assert props["tts_engine"]["enum"] == list(tts.TTS_ENGINES)
    assert props["tts_engine"]["default"] == tts.DEFAULT_TTS_ENGINE
    # One public tool, whatever the engine.
    assert [t["name"] for t in tts.TOOLS] == ["tts"]
    assert tts.TTSPlugin.PREFIX == "tts"


def test_default_engine_is_built_at_startup(_fake_engines):
    plugin = _plugin()
    assert _fake_engines.order == ["vits2-zh-en"]
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2-zh-en"


def test_actions_are_forwarded_to_the_active_engine(_fake_engines):
    plugin = _plugin()
    result = plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert result["engine_seen"] == "vits2-zh-en"
    assert _fake_engines["vits2-zh-en"].calls[-1]["input_topic"] == "/say"


def test_switch_stops_the_old_engine_and_waits_for_the_new_one(_fake_engines):
    """config waits for the bounded part of a build instead of making the caller
    poll. Answering `loading` for a 5 s session construction is what made the
    dashboard send a start the engine could not honour — see the deferred-start
    tests at the bottom for the case where waiting is not enough."""
    plugin = _plugin()
    outgoing = _fake_engines["vits2-zh-en"]

    result = plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})

    assert result["status"] == "configured"
    assert "state" not in result, "a completed switch must not report loading"
    assert result["engine"] == "matcha-zh-en"
    # The outgoing engine is stopped before the new one can publish.
    assert outgoing.stopped is True
    # And the engine is live *now*, so the start that follows lands on it.
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "matcha-zh-en"
    assert plugin.dispatch("tts", {"action": "start",
                                   "input_topic": "/say"})["state"] == "running"


def test_config_gives_up_waiting_and_reports_loading(monkeypatch, _fake_engines):
    """A build slower than the bound — a cold model download — still goes async."""
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    result = _plugin().dispatch("tts", {"action": "config",
                                        "tts_engine": "matcha-zh-en"})
    assert result["status"] == "configured"
    assert result["state"] == "loading"


def test_config_reports_a_build_failure_instead_of_loading(monkeypatch):
    class _Boom:
        def __init__(self, cfg, executor):
            raise RuntimeError("Protobuf parsing failed")

    monkeypatch.setattr(tts, "SherpaOnnxTTSPlugin", _Boom)
    monkeypatch.setattr(
        tts.TTSPlugin, "_build_vits2",
        lambda self, cfg: _FakeEngine("vits2-zh-en", cfg, self._executor, lambda i: None),
    )
    plugin = tts.TTSPlugin({"engine": "vits2-zh-en"}, _FakeExecutor())
    result = plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})
    assert result["status"] == "error"
    assert "Protobuf parsing failed" in result["message"]


def test_info_reports_loading_while_the_new_engine_builds(monkeypatch, _fake_engines):
    """Only reachable once config has stopped waiting; until then there is no
    window in which the facade has no engine."""
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})

    info = plugin.dispatch("tts", {"action": "info"})
    assert info["state"] == "loading"
    assert "matcha-zh-en" in info["desc"]
    # Other actions answer loading too, rather than hanging or lying.
    assert plugin.dispatch("tts", {"action": "speak", "text": "hi"})["state"] == "loading"


def test_switching_back_and_forth_keeps_one_engine_live(_fake_engines):
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})
    assert _wait_until(lambda: "matcha-zh-en" in _fake_engines)
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] != "loading"
    )
    sherpa = _fake_engines["matcha-zh-en"]

    plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2-zh-en"})
    assert sherpa.stopped is True
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] != "loading"
    )
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2-zh-en"
    assert _fake_engines.order == ["vits2-zh-en", "matcha-zh-en", "vits2-zh-en"]


def test_reconfiguring_the_same_engine_does_not_rebuild(_fake_engines):
    plugin = _plugin()
    result = plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2-zh-en",
                                     "speed": 1.3})
    assert _fake_engines.order == ["vits2-zh-en"], "same engine was rebuilt"
    # tts_engine is the facade's own field and must not be forwarded as if it
    # were an engine parameter; speed must be.
    applied = result["applied"]
    assert "tts_engine" not in applied and applied["speed"] == 1.3


def test_shared_config_survives_an_engine_switch(_fake_engines):
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "speed": 0.7})
    plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})
    assert _wait_until(lambda: "matcha-zh-en" in _fake_engines)
    assert _fake_engines["matcha-zh-en"].cfg["speed"] == 0.7


def test_unknown_engine_is_refused_without_touching_the_live_one(_fake_engines):
    plugin = _plugin()
    with pytest.raises(ValueError):
        plugin.dispatch("tts", {"action": "config", "tts_engine": "espeak"})
    assert _fake_engines["vits2-zh-en"].stopped is False
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2-zh-en"


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
    targets = ["matcha-zh-en", "vits2-zh-en", "matcha-zh-en", "vits2-zh-en"]
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
        {"engine": "vits2-zh-en", "model_dir": "/models/vits2"}, _FakeExecutor()
    )
    assert _fake_engines["vits2-zh-en"].cfg["model_dir"] == "/models/vits2"

    plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})
    assert _wait_until(lambda: "matcha-zh-en" in _fake_engines)
    assert (_fake_engines["matcha-zh-en"].cfg["model_dir"]
            == tts.ENGINE_MODEL_DIRS["matcha-zh-en"])

    plugin.dispatch("tts", {"action": "config", "tts_engine": "mms-th"})
    assert _wait_until(lambda: "mms-th" in _fake_engines)
    # Its own directory, not a file alongside Matcha's: the two share no weights.
    assert (_fake_engines["mms-th"].cfg["model_dir"]
            == tts.ENGINE_MODEL_DIRS["mms-th"])
    assert (tts.ENGINE_MODEL_DIRS["mms-th"]
            != tts.ENGINE_MODEL_DIRS["matcha-zh-en"])

    plugin.dispatch("tts", {"action": "config", "tts_engine": "kokoro-multi"})
    assert _wait_until(lambda: "kokoro-multi" in _fake_engines)
    assert (_fake_engines["kokoro-multi"].cfg["model_dir"]
            == tts.ENGINE_MODEL_DIRS["kokoro-multi"])
    # Its own tree too. Kokoro's archives land in `<dir>/gpu` and `<dir>/cpu`
    # beneath it, so sharing a directory with another engine would put two
    # independent ensure_verified_archive installs in one tree.
    assert len(set(tts.ENGINE_MODEL_DIRS.values())) == len(tts.ENGINE_MODEL_DIRS)


# ── legacy engine names (upgrade regression) ─────────────────────────────────

def test_engines_are_named_after_the_model_not_the_runtime():
    """`<model>-<languages>`, the shape asr_model already uses.

    Three engines run on sherpa-onnx, so a runtime name identifies none of them.
    And the dashboard renders the raw enum string, so this is what an operator
    reads in the dropdown next to the ASR one — a second convention would be
    visible.
    """
    assert set(tts.TTS_ENGINES) == {"vits2-zh-en", "matcha-zh-en", "mms-th",
                                    "kokoro-multi"}
    assert tts.DEFAULT_TTS_ENGINE == "vits2-zh-en"
    for engine in tts.TTS_ENGINES:
        assert "_" not in engine, "asr_model uses hyphens; do not mix separators"
        assert "cn" not in engine.split("-"), "zh is the language code; cn is a country"
    # Every engine needs a model dir, or _model_dir_for raises a KeyError on the
    # first switch to it.
    assert set(tts.ENGINE_MODEL_DIRS) == set(tts.TTS_ENGINES)


def test_the_asr_model_field_uses_the_same_shape():
    """Guards the reason for the naming: the two dropdowns sit side by side."""
    import plugins.asr as asr

    asr_models = asr.TOOLS[0]["configSchema"]["properties"]["asr_model"]["enum"]
    assert all("_" not in name for name in asr_models), \
        "asr_model changed separator; tts_engine was matched to it deliberately"


@pytest.mark.parametrize("stored, expected", [
    # What is actually persisted in ConfigDB and config.yaml on deployed robots.
    ("vits2_trt", "vits2-zh-en"),
    ("sherpa_onnx", "matcha-zh-en"),
    # Case and stray whitespace, as a hand-edited YAML value arrives.
    ("VITS2_TRT", "vits2-zh-en"),
    (" sherpa_onnx ", "matcha-zh-en"),
    # The bare forms this branch briefly used before the languages were added.
    ("vits2", "vits2-zh-en"),
    ("matcha", "matcha-zh-en"),
    ("mms_thai", "mms-th"),
    # Underscores fold to hyphens, so someone typing the new name the old way
    # still lands on it rather than getting "Unsupported TTS engine".
    ("vits2_zh_en", "vits2-zh-en"),
    ("mms-th", "mms-th"),
])
def test_the_old_and_hand_typed_names_still_resolve(_fake_engines, stored, expected):
    """Every deployed robot has one of the old names in ConfigDB and config.yaml.

    _select_engine raises on an unknown engine and TTSPlugin.__init__ turns that
    into a build error, so dropping the aliases would not degrade gracefully — it
    would put every existing TTS card into `state: error` on the next restart.
    """
    plugin = tts.TTSPlugin({"engine": stored}, _FakeExecutor())
    assert _wait_until(lambda: expected in _fake_engines)
    # info reports the resolved name, so nothing downstream sees the legacy spelling.
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == expected


def test_every_alias_resolves_to_a_real_engine():
    """A typo in ENGINE_ALIASES would only surface when someone upgraded."""
    for alias, target in tts.ENGINE_ALIASES.items():
        assert target in tts.TTS_ENGINES, f"{alias} points at nothing"
        assert "_" not in alias, "aliases are looked up after _ folds to -"


def test_a_legacy_name_arriving_by_config_also_resolves(_fake_engines):
    plugin = tts.TTSPlugin({"engine": "vits2-zh-en"}, _FakeExecutor())
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    assert _wait_until(lambda: "matcha-zh-en" in _fake_engines)
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "matcha-zh-en"


def test_configured_model_dir_follows_the_configured_engine(_fake_engines):
    """A sherpa-configured deployment keeps its own path, and VITS2 gets its own."""
    plugin = tts.TTSPlugin(
        {"engine": "matcha-zh-en", "model_dir": "/models/custom/sherpa"},
        _FakeExecutor(),
    )
    assert _wait_until(lambda: "matcha-zh-en" in _fake_engines)
    assert _fake_engines["matcha-zh-en"].cfg["model_dir"] == "/models/custom/sherpa"

    plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2-zh-en"})
    assert _wait_until(lambda: "vits2-zh-en" in _fake_engines)
    assert (_fake_engines["vits2-zh-en"].cfg["model_dir"]
            == tts.ENGINE_MODEL_DIRS["vits2-zh-en"])


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
    plugin = tts.TTSPlugin({"engine": "matcha-zh-en"}, _FakeExecutor())

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
    plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})

    # The build is still in flight, so the call cannot start anything yet.
    result = plugin.dispatch("tts", {"action": "start",
                                     "instance_id": "card-1",
                                     "input_topic": "/say"})
    assert result["state"] == "loading"

    _wait_until(lambda: "matcha-zh-en" in _fake_engines)
    incoming = _fake_engines["matcha-zh-en"]
    _wait_until(lambda: any(c.get("action") == "start" for c in incoming.calls))

    started = [c for c in incoming.calls if c.get("action") == "start"]
    assert len(started) == 1, "the deferred start must be replayed exactly once"
    assert started[0]["input_topic"] == "/say"
    assert started[0]["instance_id"] == "card-1"


def test_a_stop_during_a_switch_cancels_the_deferred_start(monkeypatch, _fake_engines):
    """Otherwise the node reappears after the operator asked for it to stop."""
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})
    plugin.dispatch("tts", {"action": "start", "instance_id": "card-1",
                            "input_topic": "/say"})
    assert plugin.dispatch("tts", {"action": "stop",
                                   "instance_id": "card-1"})["state"] == "idle"

    _wait_until(lambda: "matcha-zh-en" in _fake_engines)
    incoming = _fake_engines["matcha-zh-en"]
    time.sleep(0.4)   # past the fake build delay, so a replay would have landed
    assert not [c for c in incoming.calls if c.get("action") == "start"]


def test_deferred_starts_are_per_instance(monkeypatch, _fake_engines):
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})
    for card, topic in (("card-1", "/say/a"), ("card-2", "/say/b")):
        plugin.dispatch("tts", {"action": "start", "instance_id": card,
                                "input_topic": topic})

    _wait_until(lambda: "matcha-zh-en" in _fake_engines)
    incoming = _fake_engines["matcha-zh-en"]
    _wait_until(lambda: len([c for c in incoming.calls
                             if c.get("action") == "start"]) == 2)
    topics = sorted(c["input_topic"] for c in incoming.calls
                    if c.get("action") == "start")
    assert topics == ["/say/a", "/say/b"]


def test_start_after_the_build_finishes_is_not_replayed_twice(_fake_engines):
    """A start that the live engine already handled must not also be queued."""
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})
    _wait_until(lambda: plugin.dispatch("tts", {"action": "info"})["state"] != "loading")

    plugin.dispatch("tts", {"action": "start", "instance_id": "card-1",
                            "input_topic": "/say"})
    time.sleep(0.2)
    incoming = _fake_engines["matcha-zh-en"]
    assert len([c for c in incoming.calls if c.get("action") == "start"]) == 1


def test_a_deferred_start_that_fails_is_reported(monkeypatch, caplog):
    """dispatch REPORTS failure rather than raising it, so `except` never fired.

    On the robot a replayed start hit a failing dry-run and returned
    {"state": "error"}. The replay loop only caught exceptions, so nothing was
    logged and the broken start was indistinguishable from a working one — the sole
    trace was an unrelated warning from the frontend. This is the regression for
    that silence.
    """
    engines = _Engines()

    class _FailingStart(_FakeEngine):
        def dispatch(self, name, args):
            if args.get("action") == "start":
                self.calls.append(args)
                return {"state": "error", "message": "TTS dry-run produced no audio"}
            return super().dispatch(name, args)

    monkeypatch.setattr(
        tts, "SherpaOnnxTTSPlugin",
        lambda cfg, executor: _FailingStart(cfg.get("engine", "matcha-zh-en"), cfg,
                                            executor, engines.add, delay=0.3),
    )
    monkeypatch.setattr(
        tts.TTSPlugin, "_build_vits2",
        lambda self, cfg: _FakeEngine("vits2-zh-en", cfg, self._executor, engines.add),
    )
    monkeypatch.setattr(tts, "ENGINE_SWITCH_WAIT_S", 0.05)

    plugin = tts.TTSPlugin({"engine": "vits2-zh-en"}, _FakeExecutor())
    with caplog.at_level(logging.ERROR, logger="plugins.tts"):
        plugin.dispatch("tts", {"action": "config", "tts_engine": "matcha-zh-en"})
        assert plugin.dispatch("tts", {"action": "start",
                                       "instance_id": "card-1"})["state"] == "loading"
        assert _wait_until(lambda: any("deferred start" in record.getMessage()
                                       for record in caplog.records))

    failure = next(record.getMessage() for record in caplog.records
                   if "deferred start" in record.getMessage())
    assert "card-1" in failure, "the failing instance must be named"
    assert "no audio" in failure, "the engine's own message must be carried through"


# ── config must not rebuild what it just built (device regression) ───────────


class _CountingAdapter:
    """Counts constructions, so a needless rebuild is visible."""

    builds = 0
    dry_run_text = "."

    def __init__(self, cfg):
        type(self).builds += 1
        self.cfg = dict(cfg)
        self.speed = float(cfg.get("speed", 1.0))
        self.speeds = []

    def synthesize(self, text):
        return b"\x00" * 3200

    def synthesize_stream(self, text):
        yield b"\x00" * 3200

    def warmup(self):
        return 3200

    def set_speed(self, speed):
        self.speed = speed
        self.speeds.append(speed)


@pytest.fixture
def _sherpa(monkeypatch):
    """A real SherpaOnnxTTSPlugin over a counting adapter."""
    _CountingAdapter.builds = 0
    holder = {}

    def build(cfg, on_status=None):
        # Mirrors _build_tts_adapter: the plugin hands it a status sink so a
        # rebuild's download progress reaches the card.
        holder["on_status"] = on_status
        holder["adapter"] = _CountingAdapter(cfg)
        return holder["adapter"]

    monkeypatch.setattr(tts, "_build_tts_adapter", build)
    plugin = _REAL_SHERPA_PLUGIN(
        {"engine": "mms-th", "device": "gpu", "speed": 1.0,
         "speaker_id": 0, "thai_phrase_spacing": True}, _FakeExecutor())
    return plugin, holder


def test_an_identical_config_does_not_rebuild_the_session(_sherpa):
    """The "every speak takes 5 s" bug.

    config used to rebuild the adapter unconditionally. On Orin5 that measured
    2.6-2.8 s for the Thai model — a full ONNX session teardown, reload and archive
    re-verification — and it also disposed every node, so the card had to start
    again. The dashboard re-applies a card's config around a speak, so the cost
    landed on every utterance. Same code path serves matcha-zh-en; vits2 has its
    own plugin and never had the bug.
    """
    plugin, _ = _sherpa
    assert _CountingAdapter.builds == 1, "constructed once at init"

    # Exactly what the dashboard sends, including `speed: 1` as an int where the
    # stored value is 1.0 — the normalisation that makes this comparable.
    same = {"action": "config", "device": "gpu", "speed": 1,
            "speaker_id": 0, "thai_phrase_spacing": True}
    for _ in range(3):
        result = plugin.dispatch("tts", same)
        assert result["status"] == "configured"
        assert result["rebuilt"] is False
    assert _CountingAdapter.builds == 1, "an identical config rebuilt the session"


def test_changing_speed_alone_updates_the_resident_model(_sherpa):
    """speed is a per-generate() scale, so it must not reload anything."""
    plugin, holder = _sherpa
    result = plugin.dispatch("tts", {"action": "config", "speed": 1.5})
    assert result["rebuilt"] is False
    assert _CountingAdapter.builds == 1
    assert holder["adapter"].speeds == [1.5], "set_speed was not applied"


@pytest.mark.parametrize("change", [
    {"device": "cpu"},
    {"speaker_id": 1},
    {"thai_phrase_spacing": False},
    {"model_dir": "/models/other"},
])
def test_changing_a_session_key_does_rebuild(_sherpa, change):
    """The other half: a real change must still take effect."""
    plugin, _ = _sherpa
    result = plugin.dispatch("tts", {"action": "config", **change})
    assert result["rebuilt"] is True, f"{change} was ignored"
    assert _CountingAdapter.builds == 2


def test_the_facade_carries_every_shared_field_into_a_new_engine():
    """A hardcoded key list here missed thai_phrase_spacing.

    The engine was then built without it, and the next config saw a change and
    rebuilt the session that had just been built — 2.7 s, once per switch. Deriving
    the list from configSchema is what stops the next field being forgotten.
    """
    shared = {key for key, spec in tts.TOOLS[0]["configSchema"]["properties"].items()
              if spec.get("scope") == "shared"} - {"tts_engine"}
    assert set(tts.SHARED_CONFIG_KEYS) == shared
    assert "thai_phrase_spacing" in tts.SHARED_CONFIG_KEYS


def test_the_adapter_is_warmed_up_at_load(_sherpa):
    """`warmup` in config.yaml did nothing for either sherpa engine.

    Unpaid, the first utterance carried it: 1695 ms of CUDA kernels for the model
    plus 3183 ms for pythainlp's lazy corpus load on the Thai frontend's first
    normalise().
    """
    plugin, holder = _sherpa
    assert holder["adapter"].cfg  # built
    # warmup() is called through the adapter, so a plugin that skipped it would
    # leave the model cold; assert the plugin honours the flag both ways.
    _CountingAdapter.builds = 0
    calls = []
    monkey = _CountingAdapter.warmup
    try:
        _CountingAdapter.warmup = lambda self: calls.append(1) or 3200
        _REAL_SHERPA_PLUGIN({"engine": "mms-th", "warmup": True}, _FakeExecutor())
        assert calls == [1], "warmup: true was ignored"
        _REAL_SHERPA_PLUGIN({"engine": "mms-th", "warmup": False}, _FakeExecutor())
        assert calls == [1], "warmup: false was ignored"
    finally:
        _CountingAdapter.warmup = monkey


def test_info_reports_the_engine_the_facade_actually_has(_fake_engines):
    """An implementation must not name the engine; only the facade knows.

    vits2_tts_trt hardcoded `"engine": "vits2_trt"` in its identity dict, and the
    facade merged it with setdefault — so after the rename the card displayed
    `vits2_trt`, a value not in the configSchema enum, while the sherpa engines
    (which report no engine of their own) showed the right one. It looked like the
    default had never been renamed.
    """
    class _Opinionated(_FakeEngine):
        def dispatch(self, name, args):
            result = super().dispatch(name, args)
            if args.get("action") == "info":
                result["engine"] = "vits2_trt"   # the stale hardcoded name
            return result

    engines = _Engines()
    import plugins.tts as _tts
    _tts.TTSPlugin._build_vits2 = (
        lambda self, cfg: _Opinionated("vits2-zh-en", cfg, self._executor, engines.add))
    plugin = tts.TTSPlugin({"engine": "vits2-zh-en"}, _FakeExecutor())
    reported = plugin.dispatch("tts", {"action": "info"})["engine"]
    assert reported == "vits2-zh-en", f"the impl's stale name won: {reported}"
    assert reported in tts.TTS_ENGINES, "info reported an engine not in the enum"


def test_no_implementation_declares_its_own_engine_name():
    """Guards the rule rather than one instance of breaking it.

    Parses the AST instead of grepping the source: the comment that explains this
    rule quotes the key it forbids, and a substring check matched the comment.
    """
    import ast
    import inspect
    import textwrap

    import plugins.vits2_tts_trt.plugin as vits2_plugin

    tree = ast.parse(textwrap.dedent(inspect.getsource(vits2_plugin.TTSPlugin._identity)))
    dict_keys = [key.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                 for key in node.keys if isinstance(key, ast.Constant)]
    assert "engine" not in dict_keys, (
        "the engine name belongs to plugins/tts.py's facade, not to an engine")


def test_the_default_engine_is_vits2_zh_en():
    """The ZH/EN TensorRT engine is the default everywhere it is declared."""
    import pathlib

    import yaml

    assert tts.DEFAULT_TTS_ENGINE == "vits2-zh-en"
    props = tts.TOOLS[0]["configSchema"]["properties"]
    assert props["tts_engine"]["default"] == "vits2-zh-en"
    config = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    # config.yaml and the schema must agree, or the card shows one default while
    # the process boots another.
    assert config["plugins"]["tts"]["engine"] == "vits2-zh-en"
