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
    assert "paraformer-zh-en" in asr.asr_models_supporting("gpu")
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
    other = asr.ASR_MODELS["paraformer-zh-en"]["devices"]["cpu"]["dir"]
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
