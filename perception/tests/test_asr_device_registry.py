"""
Invariants for the ASR (model, device) registry — no hardware or models needed.

Run from the repo root:
    python -m pytest perception/tests -q

These exist because the registry encodes measurements, and a plausible-looking
edit can silently undo them. Two mistakes in particular are cheap to make and
expensive to notice:

- pointing a `gpu` entry at int8 weights, which runs 1.25x-3.3x *slower* than the
  CPU because ONNX Runtime's CUDA provider has no int8 kernels, and
- pointing a `cpu` entry at fp16 weights, which measured 42890 ms against int8's
  3295 ms because ONNX Runtime has no fp16 CPU kernels.

Neither raises; both just make the product slow. Also asserted: the configSchema's
`device` visibility list matches the models that actually have gpu weights, so the
dashboard never offers a choice the plugin would reject.

sherpa_onnx is stubbed out because importing plugins.asr pulls it in transitively.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))

from plugins import asr  # noqa: E402
from utils import model_downloader  # noqa: E402

VALID_DTYPES = {"int8", "fp32", "fp16"}


def _device_specs():
    for model, info in asr.ASR_MODELS.items():
        for device, spec in info["devices"].items():
            yield model, device, spec


def test_every_model_has_a_cpu_entry():
    """cpu is the fallback for an unsupported device request, so it must exist."""
    for model, info in asr.ASR_MODELS.items():
        assert "cpu" in info["devices"], f"{model} has no cpu weights"


def test_device_keys_are_known():
    for model, device, _ in _device_specs():
        assert device in ("cpu", "gpu"), f"{model} declares unknown device {device!r}"


def test_dtypes_are_declared_and_known():
    for model, device, spec in _device_specs():
        assert spec.get("dtype") in VALID_DTYPES, \
            f"{model}/{device} has dtype {spec.get('dtype')!r}"


def test_gpu_entries_are_never_int8():
    """ONNX Runtime's CUDA provider has no int8 kernels: it partitions the graph,
    falls back to CPU node by node, and adds a copy at every boundary. Measured
    1.25x-3.3x slower than the CPU, and it also perturbs the output (3 of 4
    SenseVoice transcripts changed versus the same model on CPU)."""
    for model, device, spec in _device_specs():
        if device == "gpu":
            assert spec["dtype"] != "int8", \
                f"{model}/gpu points at int8 weights, which is slower than cpu"


def test_cpu_entries_are_never_fp16():
    """ONNX Runtime has no fp16 CPU kernels and casts everything: 42890 ms where
    int8 took 3295 ms on the same audio."""
    for model, device, spec in _device_specs():
        if device == "cpu":
            assert spec["dtype"] != "fp16", \
                f"{model}/cpu points at fp16 weights, which is ~10x slower than int8"


def test_download_keys_resolve():
    """A gpu entry must name a pinned bundle; a cpu entry a legacy archive."""
    for model, device, spec in _device_specs():
        key = spec["download"]
        if device == "gpu":
            assert key in model_downloader.SHERPA_GPU_BUNDLES, \
                f"{model}/gpu download key {key!r} is not a SHERPA_GPU_BUNDLES entry"
        else:
            assert key in model_downloader.MODELS, \
                f"{model}/cpu download key {key!r} is not a MODELS entry"


def test_model_dirs_are_unique():
    """Two entries sharing a directory would download over each other's weights."""
    dirs = [spec["dir"] for _, _, spec in _device_specs()]
    assert len(dirs) == len(set(dirs)), "duplicate model_dir in ASR_MODELS"


def test_default_model_exists_and_is_the_schema_default():
    assert asr.DEFAULT_ASR_MODEL in asr.ASR_MODELS
    schema = asr.TOOLS[0]["configSchema"]["properties"]
    assert schema["asr_model"]["default"] == asr.DEFAULT_ASR_MODEL


def test_schema_enum_matches_the_registry():
    schema = asr.TOOLS[0]["configSchema"]["properties"]
    assert sorted(schema["asr_model"]["enum"]) == sorted(asr.ASR_MODELS)


def test_device_field_is_shown_exactly_for_models_with_gpu_weights():
    """Otherwise the dashboard offers gpu on a model whose config action rejects
    it, or hides it on a model that supports it."""
    schema = asr.TOOLS[0]["configSchema"]["properties"]
    shown_for = schema["device"]["x-show-when"]["asr_model"]
    assert sorted(shown_for) == asr.asr_models_supporting("gpu")


def test_device_schema_default_is_cpu():
    schema = asr.TOOLS[0]["configSchema"]["properties"]
    assert schema["device"]["default"] == "cpu"
    assert sorted(schema["device"]["enum"]) == ["cpu", "gpu"]


def test_asr_models_supporting():
    assert "sensevoice-small" in asr.asr_models_supporting("gpu")
    assert "parakeet-en" in asr.asr_models_supporting("gpu")
    # Measured 0.80x on CUDA — deliberately absent.
    assert "x-asr-zh-en" not in asr.asr_models_supporting("gpu")
    assert asr.asr_models_supporting("cpu") == sorted(asr.ASR_MODELS)


@pytest.mark.parametrize("cfg, expected_dir", [
    ({}, None),                                        # registry default
    ({"model_dir": "/custom/place"}, "/custom/place"),  # honoured
])
def test_model_dir_override(cfg, expected_dir):
    spec = asr.ASR_MODELS["sensevoice-small"]["devices"]["cpu"]
    got = asr._model_dir_for(cfg, spec)
    assert got == (expected_dir or spec["dir"])


def test_model_dir_from_another_entry_is_ignored():
    """config.yaml ships a model_dir for one bundle; reusing it for a different
    model or device would download the wrong weights into it."""
    spec = asr.ASR_MODELS["sensevoice-small"]["devices"]["gpu"]
    other = asr.ASR_MODELS["parakeet-en"]["devices"]["cpu"]["dir"]
    assert asr._model_dir_for({"model_dir": other}, spec) == spec["dir"]


# ── warmup ───────────────────────────────────────────────────────────────────
#
# The first CUDA inference cost 1777 ms against a 77 ms steady state (lazy kernel
# loading, cuDNN autotuning, memory pool). Untouched, that lands on the operator's
# first utterance, right after the model finished loading.

def test_silence_wav_is_a_decodable_16k_mono_wav():
    import io
    import wave
    data = asr._silence_wav(0.5)
    with wave.open(io.BytesIO(data)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == asr.SAMPLE_RATE
        assert wf.getnframes() == int(asr.SAMPLE_RATE * 0.5)


def test_warmup_decodes_one_clip():
    calls = []

    class _Adapter:
        def transcribe(self, wav_bytes, language):
            calls.append((len(wav_bytes), language))
            return ""

    asr._warmup_adapter(_Adapter(), "sensevoice-small", "gpu")
    assert len(calls) == 1
    assert calls[0][0] > 0


@pytest.mark.parametrize("cfg, expect_warm", [
    ({}, True),                      # on by default — the gpu first-call cost
    ({"warmup": True}, True),
    ({"warmup": False}, False),      # opt out
])
def test_build_warms_up_unless_disabled(monkeypatch, cfg, expect_warm):
    warmed = []
    monkeypatch.setattr(asr, "_warmup_adapter",
                        lambda adapter, model, device: warmed.append((model, device)))
    monkeypatch.setattr("utils.model_downloader.ensure_model",
                        lambda *a, **k: None)
    monkeypatch.setitem(asr.ASR_MODELS["sensevoice-small"], "adapter",
                        lambda model_dir, device, num_threads: object())

    asr._build_asr_adapter({"asr_model": "sensevoice-small", "device": "cpu", **cfg})
    assert bool(warmed) is expect_warm


def test_warmup_failure_does_not_propagate():
    """A model that cannot decode silence still fails loudly on real audio; it must
    not stop the plugin from coming up."""
    class _Broken:
        def transcribe(self, wav_bytes, language):
            raise RuntimeError("no session")

    asr._warmup_adapter(_Broken(), "sensevoice-small", "gpu")  # must not raise


def _plugin_with(monkeypatch, model, device):
    """An ASRPlugin whose model load is stubbed out, for dispatch-level tests."""
    monkeypatch.setattr(asr, "_build_asr_adapter", lambda *a, **k: object())
    plugin = asr.ASRPlugin({"asr_model": model, "device": device}, executor=None)
    loads = []
    monkeypatch.setattr(plugin, "_load_model_async", lambda name: loads.append(name))
    return plugin, loads


# ── removed models ───────────────────────────────────────────────────────────
#
# paraformer-zh-en, paraformer-offline and zipformer-en were dropped for
# accuracy. A card's `asr_model` lives in agent-core's config DB on each robot,
# so the upgrade cannot rewrite it — without an alias, every deployment that had
# picked one comes up `state: error` after the next restart.


def test_every_removed_model_resolves_to_one_that_exists():
    assert asr.REMOVED_ASR_MODELS, "the alias map must not be emptied"
    for gone, replacement in asr.REMOVED_ASR_MODELS.items():
        assert gone not in asr.ASR_MODELS, f"{gone} is still registered"
        assert replacement in asr.ASR_MODELS, f"{gone} -> missing {replacement}"
        assert asr.resolve_asr_model(gone) == replacement


def test_a_live_model_name_passes_through_untouched():
    for name in asr.ASR_MODELS:
        assert asr.resolve_asr_model(name) == name


def test_the_schema_enum_offers_exactly_the_registered_models():
    enum = asr.TOOLS[0]["configSchema"]["properties"]["asr_model"]["enum"]
    assert sorted(enum) == sorted(asr.ASR_MODELS)
    # A removed name in the dropdown would let an operator re-pick it.
    assert not set(enum) & set(asr.REMOVED_ASR_MODELS)


def test_config_migrates_a_card_still_holding_a_removed_model(monkeypatch):
    """The failure this guards: `state: error` on every upgraded robot."""
    plugin, loads = _plugin_with(monkeypatch, "sensevoice-small", "cpu")

    result = plugin.dispatch("asr", {"action": "config",
                                     "asr_model": "zipformer-en"})

    assert result["status"] != "error", result
    assert plugin._asr_model == "parakeet-en"
    assert loads == ["parakeet-en"]


def test_an_unknown_model_is_still_an_error(monkeypatch):
    """Resolving removed names must not turn every typo into a silent default."""
    plugin, loads = _plugin_with(monkeypatch, "sensevoice-small", "cpu")

    result = plugin.dispatch("asr", {"action": "config",
                                     "asr_model": "whisper-large"})

    assert result["status"] == "error"
    assert loads == []


def test_config_degrades_a_carried_over_device_instead_of_rejecting(monkeypatch):
    """Switching to a cpu-only model must not fail on a stale `device: gpu`.

    The device field is hidden by `x-show-when` for models with no gpu weights,
    but the form still submits the last selected value. Rejecting that left the
    card running the previous model while the operator believed they had
    switched — seen on Orin5, where a parakeet-en request was rejected and the
    transcripts that followed were sensevoice-small's. (parakeet-en has gpu
    weights now, so the cpu-only model under test here is x-asr-zh-en — the
    only one left after the paraformers and zipformer-en were dropped.)
    """
    plugin, loads = _plugin_with(monkeypatch, "sensevoice-small", "gpu")

    result = plugin.dispatch("asr", {"action": "config",
                                     "asr_model": "x-asr-zh-en", "device": "gpu"})

    assert result["status"] != "error", result
    assert result["device"] == "cpu"
    assert plugin._asr_model == "x-asr-zh-en"
    assert loads == ["x-asr-zh-en"]


def test_config_still_rejects_an_explicit_unsupported_device(monkeypatch):
    """Asking for gpu on a cpu-only model is a real error and stays one.

    Degrading this one silently is how a "GPU is not faster" bug report gets
    written against a model that never ran on the GPU at all.
    """
    plugin, loads = _plugin_with(monkeypatch, "x-asr-zh-en", "cpu")

    result = plugin.dispatch("asr", {"action": "config",
                                     "asr_model": "x-asr-zh-en", "device": "gpu"})

    assert result["status"] == "error"
    assert "gpu" in result["message"]
    assert plugin._device == "cpu"
    assert loads == []


def test_a_device_change_mid_load_supersedes_the_load_in_flight(monkeypatch):
    """A switch made while a model is downloading must not be dropped.

    Seen on Orin5: `config {parakeet-en}` started the cpu download, the operator
    then picked gpu, and the second request updated `_device` but never reloaded
    — the in-flight load was simply ignored. The card came up on the int8 cpu
    weights while both the plugin and the dashboard reported gpu.
    """
    import threading

    monkeypatch.setattr(asr, "_build_asr_adapter", lambda *a, **k: object())
    plugin = asr.ASRPlugin({"asr_model": "sensevoice-small", "device": "cpu"},
                           executor=None)

    started, release, built = threading.Event(), threading.Event(), []

    def _fake_build(cfg, on_status=None):
        built.append((cfg["asr_model"], cfg["device"]))
        if len(built) == 1:
            started.set()
            assert release.wait(5), "test deadlock"
        return object()

    monkeypatch.setattr(asr, "_build_asr_adapter", _fake_build)

    first = plugin.dispatch("asr", {"action": "config", "asr_model": "parakeet-en"})
    assert first["status"] == "loading" and first["device"] == "cpu"
    assert started.wait(5)

    second = plugin.dispatch("asr", {"action": "config",
                                     "asr_model": "parakeet-en", "device": "gpu"})
    assert second["status"] == "loading" and second["device"] == "gpu"

    release.set()
    deadline = time.monotonic() + 5
    while plugin._loading and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not plugin._loading
    assert built == [("parakeet-en", "cpu"), ("parakeet-en", "gpu")]
    assert plugin._plugin_cfg["device"] == "gpu"
    assert plugin._load_error is None


def test_config_reports_loading_while_a_load_is_still_running(monkeypatch):
    """A no-op config mid-download must not answer "configured".

    That answer ended the dashboard's progress text and left it on a bare
    spinner for the rest of the download.
    """
    plugin, _ = _plugin_with(monkeypatch, "parakeet-en", "cpu")
    plugin._loading = True
    plugin._load_status = "正在下载模型 'parakeet-en' … 30% (30/100 MB)"

    result = plugin.dispatch("asr", {"action": "config", "asr_model": "parakeet-en"})

    assert result["status"] == "loading"
    assert result["message"] == plugin._load_status


def test_start_returns_loading_instead_of_blocking_for_a_download(monkeypatch):
    """`start` waits only a grace period, then hands back to the info poller.

    Blocking for the whole download pinned an MCP worker thread and kept the
    dashboard from ever polling `info`, which is where the phase text lives.
    """
    plugin, _ = _plugin_with(monkeypatch, "parakeet-en", "cpu")
    monkeypatch.setattr(asr, "MODEL_LOAD_GRACE_S", 0.05)
    plugin._loading = True
    plugin._load_status = "正在下载模型 'parakeet-en' … 30% (30/100 MB)"

    began = time.monotonic()
    result = plugin.dispatch("asr", {"action": "start", "instance_id": "card-1",
                                     "input_topic": "/mic"})

    assert result["state"] == "loading"
    assert result["message"] == plugin._load_status
    assert time.monotonic() - began < 1


def test_a_start_deferred_behind_a_load_runs_when_the_model_is_ready(monkeypatch):
    """Handing `start` back as `loading` must not silently drop it.

    The caller polls `info` after that and treats anything other than `loading`
    as the final answer — so a plugin that came back idle instead of running
    would be read as a cancelled card, and the ASR node would never exist.
    """
    import threading

    monkeypatch.setattr(asr, "_build_asr_adapter", lambda *a, **k: object())
    plugin = asr.ASRPlugin({"asr_model": "sensevoice-small", "device": "cpu"},
                           executor=types.SimpleNamespace(add_node=lambda node: None))
    monkeypatch.setattr(asr, "MODEL_LOAD_GRACE_S", 0.05)

    release = threading.Event()
    monkeypatch.setattr(asr, "_build_asr_adapter",
                        lambda cfg, on_status=None: (release.wait(5), object())[1])

    plugin._load_model_async("parakeet-en")
    result = plugin.dispatch("asr", {"action": "start", "instance_id": "card-1",
                                     "input_topic": "/mic"})
    assert result["state"] == "loading"
    assert "card-1" in plugin._pending_starts
    # The window between "model ready" and "node running" must not read as idle.
    plugin._loading = False
    assert plugin.dispatch("asr", {"action": "info"})["state"] == "loading"
    plugin._loading = True

    starts = []
    monkeypatch.setattr(asr, "_ASRNode", lambda *a, **k: _FakeNode(starts, *a, **k))
    release.set()

    deadline = time.monotonic() + 5
    while not starts and time.monotonic() < deadline:
        time.sleep(0.01)
    assert starts == ["/mic"], "the deferred start never ran"
    assert plugin._pending_starts == {}


def test_stop_drops_a_start_deferred_behind_a_load(monkeypatch):
    """Otherwise it fires minutes after the operator stopped the card."""
    plugin, _ = _plugin_with(monkeypatch, "parakeet-en", "cpu")
    plugin._pending_starts["card-1"] = {"action": "start", "instance_id": "card-1"}

    plugin.dispatch("asr", {"action": "stop", "instance_id": "card-1"})

    assert plugin._pending_starts == {}


class _FakeNode:
    """Stands in for _ASRNode: records the topic it was started on."""

    def __init__(self, log, *args, **kwargs):
        self._log = log
        self.topic = args[0] if args else kwargs.get("input_topic", "")
        self.state = "idle"

    def start(self):
        self.state = "running"
        self._log.append(self.topic)
        return {"state": "running"}


# ── removed trigger modes ────────────────────────────────────────────────────
#
# `kws` ran a second sherpa KeywordSpotter on the raw audio. It was removed in
# favour of `asr_kws`, which gates on the transcript. This one degrades worse
# than a removed model name if left unmapped: an unrecognised trigger_mode
# falls through to `vad`, which is always listening — a wake-word-gated robot
# would silently start answering every utterance in the room.


def test_kws_migrates_to_asr_kws():
    assert "kws" not in asr.TRIGGER_MODES
    assert asr.REMOVED_TRIGGER_MODES["kws"] == "asr_kws"
    assert asr.resolve_trigger_mode("kws", {}) == "asr_kws"


def test_live_trigger_modes_pass_through():
    for mode in asr.TRIGGER_MODES:
        assert asr.resolve_trigger_mode(mode, {}) == mode


def test_the_schema_enum_matches_the_live_trigger_modes():
    enum = asr.TOOLS[0]["configSchema"]["properties"]["trigger_mode"]["enum"]
    assert sorted(enum) == sorted(asr.TRIGGER_MODES)
    assert asr.DEFAULT_TRIGGER_MODE in enum
    assert not set(enum) & set(asr.REMOVED_TRIGGER_MODES)


def test_migrating_carries_the_wake_word_over_from_kws_keywords():
    """Without this the migrated card has no keyword, and asr_kws degrades to
    always-on vad — the exact silent failure this mapping exists to prevent."""
    cfg = {"keywords": ["x iǎo f àn x iǎo f àn @小范小范"]}

    assert asr.resolve_trigger_mode("kws", cfg) == "asr_kws"
    assert cfg["asr_kws_keyword"] == "小范小范"


def test_an_underscored_english_wake_word_becomes_speakable_text():
    cfg = {"keywords": ["▁FA N C Y ▁RO B O T @FANCY_ROBOT"]}

    asr.resolve_trigger_mode("kws", cfg)

    # '_' is a filename convention, not something anyone says.
    assert cfg["asr_kws_keyword"] == "FANCY ROBOT"


def test_an_explicit_asr_kws_keyword_is_not_overwritten():
    cfg = {"asr_kws_keyword": "hello robot",
           "keywords": ["x iǎo f àn x iǎo f àn @小范小范"]}

    assert asr.resolve_trigger_mode("kws", cfg) == "asr_kws"
    assert cfg["asr_kws_keyword"] == "hello robot"


def test_a_token_only_keyword_yields_no_wake_word_rather_than_a_guess():
    """The token side is a spotter lexicon, not text — stripping its spaces
    would produce a wake word nobody can pronounce. Better to have none and
    log, which resolve_trigger_mode does at error level."""
    cfg = {"keywords": ["x iǎo f àn x iǎo f àn"]}

    assert asr.resolve_trigger_mode("kws", cfg) == "asr_kws"
    assert "asr_kws_keyword" not in cfg
