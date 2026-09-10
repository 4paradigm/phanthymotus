"""
tests/test_kokoro_worker.py — the policy layer for Japanese, now that it owns no process.

`plugins/ort_worker.py` owns the one child that holds every standalone-ORT session in
this process. This module owns the *decisions* about Japanese, and each of them was
forced by a measurement rather than chosen:

  - **which device is safe.** jp5.11's CUDA execution of this graph computes durations
    27.5% short (8 runs: cpu 8.35 s stdev 0.000, one distinct length; cuda 6.05 s
    stdev 0.053, three), which is audibly rushed speech. The graph *is* stochastic — 4
    RandomNormalLike and 7 RandomUniformLike nodes — so comparing waveforms is not a
    valid check, and the duration is;
  - **whether there is room.** Asking for CUDA regardless OOM-killed a jp5.11 rig
    before the duration probe could run, so the headroom check has to come first;
  - **when to give the memory back.** On idle, not on the language switch: the cold
    path is ~15 s end to end and an alternating tour would pay it every time.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import kokoro_worker as kw  # noqa: E402


class _FakeDirect:
    """A KokoroDirect stand-in whose probe length is scripted per provider."""

    lengths: dict = {}
    built: list = []
    closed: list = []

    def __init__(self, model_dir, weights, num_threads=0, provider="cpu",
                 session_key="tts.kokoro", in_process=False):
        self.provider = provider
        self.in_process = in_process
        self.session_key = session_key
        _FakeDirect.built.append((provider, in_process))
        self._session = type("s", (), {"get_providers": lambda self: [provider]})()

    num_speakers, sample_rate = 54, 24000

    def synthesize(self, phonemes, speaker_id=0, speed=1.0):
        return np.zeros(self.lengths.get(self.provider, kw.PROBE_SAMPLES),
                        dtype=np.float32)

    def close(self):
        _FakeDirect.closed.append(self.provider)


@pytest.fixture(autouse=True)
def _reset():
    _FakeDirect.lengths = {}
    _FakeDirect.built = []
    _FakeDirect.closed = []
    yield


def _proxy(monkeypatch, device="gpu", headroom=8192, idle=1e9):
    """A proxy whose sessions are fakes, with the reaper thread neutered."""
    import plugins.kokoro_direct as kd
    monkeypatch.setattr(kd, "KokoroDirect", _FakeDirect)
    monkeypatch.setattr(kw, "_mem_available_mb", lambda: headroom)
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_reap_loop", lambda self: None)
    return kw.KokoroWorkerProxy("/models/kokoro-multi/gpu", "model.onnx",
                                device=device, idle_timeout_s=idle)


# ── which device ──────────────────────────────────────────────────────────────

def test_a_device_that_gets_durations_wrong_is_rejected_for_cpu(monkeypatch):
    """jp5.11's CUDA returns 35-44% of the probe's length. 27% short is rushed speech.

    Gating on the ORT version number would be the wrong fix — it would not catch the
    next line with the same defect — so the gate is the measurement, and it is free
    because the probe already runs to warm the session.
    """
    _FakeDirect.lengths = {"gpu": 21000, "cpu": kw.PROBE_SAMPLES}
    proxy = _proxy(monkeypatch, device="gpu")
    assert proxy.device_used == "cpu"
    assert [p for p, _ in _FakeDirect.built] == ["gpu", "cpu"]
    assert "gpu" in _FakeDirect.closed, "the rejected session must be unloaded"


def test_a_correct_device_is_kept(monkeypatch):
    _FakeDirect.lengths = {"gpu": kw.PROBE_SAMPLES}
    proxy = _proxy(monkeypatch, device="gpu")
    assert proxy.device_used == "gpu"
    assert _FakeDirect.closed == []


def test_a_few_samples_of_deviation_are_tolerated(monkeypatch):
    """A different-but-valid build must not be rejected over rounding."""
    _FakeDirect.lengths = {"gpu": int(kw.PROBE_SAMPLES * 1.03)}
    assert _proxy(monkeypatch, device="gpu").device_used == "gpu"


def test_cpu_failing_the_check_too_is_an_error_not_a_silent_pass(monkeypatch):
    """Then the model or the phoneme table does not match this code — say so."""
    _FakeDirect.lengths = {"gpu": 100, "cpu": 100}
    with pytest.raises(RuntimeError, match="plausible probe duration"):
        _proxy(monkeypatch, device="gpu")


# ── whether there is room ─────────────────────────────────────────────────────

def test_cuda_is_declined_when_the_box_has_no_headroom(monkeypatch):
    """The duration check cannot save a box that OOMs while building the session.

    Measured on jp5.11: 5237 MB available, sherpa's own Kokoro GPU adapter takes 3.2 GB,
    and the CUDA allocation then killed the process — before the probe ran. So this
    guard has to be first; it is the one that can take the process down.
    """
    proxy = _proxy(monkeypatch, device="gpu", headroom=2048)
    assert proxy.device_used == "cpu"
    assert [p for p, _ in _FakeDirect.built] == ["cpu"], (
        "with no headroom it must not even try CUDA")


def test_cuda_is_used_when_there_is_room(monkeypatch):
    assert _proxy(monkeypatch, device="gpu", headroom=4096).device_used == "gpu"


def test_an_unreadable_meminfo_does_not_block_the_gpu(monkeypatch):
    """-1 means "could not tell", which must not be read as "no memory"."""
    assert _proxy(monkeypatch, device="gpu", headroom=-1).device_used == "gpu"


# ── when to give it back ──────────────────────────────────────────────────────

def test_close_unloads_the_session_but_not_the_shared_child(monkeypatch):
    """face's sessions live in the same child and must survive Japanese finishing."""
    proxy = _proxy(monkeypatch)
    proxy.close()
    assert _FakeDirect.closed == ["gpu"]
    proxy.close()                      # idempotent


def test_idle_unloading_releases_the_session(monkeypatch):
    """A card that spoke Japanese once used to hold the session for the adapter's life:
    `set_language` never released it, so switching back to English kept it resident."""
    proxy = _proxy(monkeypatch, idle=0.0)
    proxy._last_used = time.monotonic() - 3600
    direct = None
    with proxy._lock:
        if proxy._direct is not None and \
                (time.monotonic() - proxy._last_used) > proxy._idle_timeout_s:
            direct, proxy._direct = proxy._direct, None
    assert direct is not None, "an idle session must be picked up for unloading"
    direct.close()
    assert _FakeDirect.closed == ["gpu"]


def test_a_synthesize_after_idle_unloading_rebuilds(monkeypatch):
    proxy = _proxy(monkeypatch)
    proxy._direct = None
    out = proxy.synthesize("konnichiwa", speaker_id=0, speed=1.0)
    assert out.size == kw.PROBE_SAMPLES
    assert proxy._direct is not None


# ── failure behaviour ─────────────────────────────────────────────────────────

def test_a_worker_failure_falls_back_in_process_on_cpu(monkeypatch):
    """Japanese losing 8x is acceptable; Japanese breaking is not.

    The fallback must be in-process **and** cpu: a CPU-only session loads no CUDA
    provider, so it never touches the bridge the collision runs through.
    """
    proxy = _proxy(monkeypatch)
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_build",
                        lambda self: (_ for _ in ()).throw(RuntimeError("child gone")))
    proxy._direct = None
    out = proxy.synthesize("konnichiwa")
    assert out.size == kw.PROBE_SAMPLES
    assert ("cpu", True) in _FakeDirect.built, _FakeDirect.built


def test_a_failing_call_falls_back_rather_than_propagating(monkeypatch):
    proxy = _proxy(monkeypatch)
    proxy._direct.synthesize = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("the child died mid-utterance"))
    out = proxy.synthesize("konnichiwa")
    assert out.size == kw.PROBE_SAMPLES
    assert ("cpu", True) in _FakeDirect.built
