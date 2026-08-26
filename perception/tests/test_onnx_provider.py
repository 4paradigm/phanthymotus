"""
Host-side unit tests for utils/onnx_provider (no sherpa-onnx install required).

Run from the repo root:
    python -m pytest perception/tests -q

`sherpa_onnx` is replaced by a stub module whose __file__ points into a tmp_path
tree, so the CUDA-wheel marker can be made present or absent at will.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from utils import onnx_provider  # noqa: E402

INT8 = "/models/asr/encoder.int8.onnx"
FP32 = "/models/asr/encoder.onnx"
FP16 = "/models/asr/model.fp16.onnx"


@pytest.fixture(autouse=True)
def _clear_cache():
    """cuda_available is lru_cached for the process lifetime."""
    onnx_provider.cuda_available.cache_clear()
    yield
    onnx_provider.cuda_available.cache_clear()


def _install_fake_sherpa(monkeypatch, tmp_path, *, with_cuda: bool):
    pkg_dir = tmp_path / "sherpa_onnx"
    (pkg_dir / "lib").mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    if with_cuda:
        (pkg_dir / "lib" / "libonnxruntime_providers_cuda.so").write_bytes(b"")
    module = types.ModuleType("sherpa_onnx")
    module.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)


# ── cuda_available ───────────────────────────────────────────────────────────

def test_cuda_available_true_when_marker_present(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.cuda_available() is True


def test_cuda_available_false_on_cpu_wheel(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=False)
    assert onnx_provider.cuda_available() is False


def test_cuda_available_false_when_sherpa_missing(monkeypatch):
    """x86 dev hosts and the unit-test environment have no sherpa-onnx at all."""
    real_import = __import__

    def _fail(name, *args, **kwargs):
        if name == "sherpa_onnx":
            raise ImportError("no sherpa_onnx")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sherpa_onnx", raising=False)
    monkeypatch.setattr("builtins.__import__", _fail)
    assert onnx_provider.cuda_available() is False


# ── dtype detection ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("paths, expected", [
    ((INT8,), True),
    ((FP32,), False),
    ((FP16,), False),
    ((INT8, FP32), True),                       # any int8 file poisons the set
    (("/m/decoder-epoch-99-avg-1.onnx", "/m/joiner-epoch-99-avg-1.int8.onnx"), True),
    ((), False),
    ((None, ""), False),                        # unset paths are not int8 evidence
])
def test_is_quantised(paths, expected):
    assert onnx_provider.is_quantised(paths) is expected


@pytest.mark.parametrize("paths, expected", [
    ((FP16,), True),
    ((FP32,), False),
    ((INT8,), False),
    ((FP32, FP16), True),
    ((), False),
])
def test_is_fp16(paths, expected):
    assert onnx_provider.is_fp16(paths) is expected


# ── normalize_device ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    ("cpu", "cpu"),
    ("gpu", "gpu"),
    ("GPU", "gpu"),
    (" gpu ", "gpu"),
    ("cuda", "gpu"),        # provider name accepted as an alias
    ("nvidia", "gpu"),
    (None, "cpu"),
    ("", "cpu"),
    ("tpu", "cpu"),         # unknown falls back rather than raising
])
def test_normalize_device(value, expected):
    assert onnx_provider.normalize_device(value) == expected


@pytest.mark.parametrize("legacy, expected", [
    ("cuda", "gpu"),
    ("gpu", "gpu"),
    ("cpu", "cpu"),
    ("auto", "cpu"),        # the old `auto` has no device meaning; default applies
])
def test_normalize_device_accepts_legacy_hw_provider(legacy, expected):
    """A deployment mounting its own pre-`device` config.yaml keeps working."""
    assert onnx_provider.normalize_device(None, legacy) == expected


def test_explicit_device_wins_over_legacy_key(monkeypatch):
    assert onnx_provider.normalize_device("cpu", "cuda") == "cpu"


# ── provider_for_device ──────────────────────────────────────────────────────

def test_gpu_with_fp16_weights_uses_cuda(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.provider_for_device("gpu", (FP16,)) == "cuda"


def test_gpu_with_fp32_weights_uses_cuda(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.provider_for_device("gpu", (FP32,)) == "cuda"


def test_gpu_with_int8_weights_falls_back_to_cpu(monkeypatch, tmp_path):
    """A registry bug, and the measured reason to catch it: ONNX Runtime's CUDA EP
    has no int8 kernels, falls back per node, and ran 1.25x-3.3x slower than CPU."""
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.provider_for_device("gpu", (INT8,)) == "cpu"


def test_gpu_on_cpu_only_wheel_falls_back(monkeypatch, tmp_path):
    """jp6.1 and x86 install the PyPI CPU wheel; ASR must still come up."""
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=False)
    assert onnx_provider.provider_for_device("gpu", (FP16,)) == "cpu"


def test_cpu_stays_cpu_for_every_dtype(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    for path in (INT8, FP32, FP16):
        assert onnx_provider.provider_for_device("cpu", (path,)) == "cpu"


def test_cpu_with_fp16_logs_an_error(monkeypatch, tmp_path, caplog):
    """fp16 on CPU measured 42890 ms against int8's 3295 ms — ONNX Runtime has no
    fp16 CPU kernels. It still runs, but the registry is wrong and should say so."""
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    with caplog.at_level("ERROR"):
        assert onnx_provider.provider_for_device("cpu", (FP16,)) == "cpu"
    assert any("fp16" in r.message.lower() or "fp16" in r.getMessage().lower()
               for r in caplog.records)


def test_unknown_device_is_treated_as_cpu(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.provider_for_device("tpu", (FP32,)) == "cpu"


# ── pick_weights ─────────────────────────────────────────────────────────────

def test_pick_weights_returns_first_existing(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"")
    picked = onnx_provider.pick_weights(str(tmp_path), "model.fp16.onnx",
                                        "model.onnx", "model.int8.onnx")
    assert picked == str(tmp_path / "model.onnx")


def test_pick_weights_respects_candidate_order(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "model.int8.onnx").write_bytes(b"")
    assert onnx_provider.pick_weights(str(tmp_path), "model.int8.onnx",
                                      "model.onnx") == str(tmp_path / "model.int8.onnx")
    assert onnx_provider.pick_weights(str(tmp_path), "model.onnx",
                                      "model.int8.onnx") == str(tmp_path / "model.onnx")


def test_pick_weights_falls_back_to_last_candidate(tmp_path):
    """So the caller's own "not found" error names the file it wanted."""
    assert onnx_provider.pick_weights(str(tmp_path), "a.onnx", "b.onnx") == \
        str(tmp_path / "b.onnx")


def test_pick_weights_with_no_candidates(tmp_path):
    assert onnx_provider.pick_weights(str(tmp_path)) == ""
