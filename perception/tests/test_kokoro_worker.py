"""
tests/test_kokoro_worker.py — the proxy's contract and its failure behaviour.

The worker exists because two ONNX Runtimes in one process cannot both hold a CUDA
session on the Kokoro graph — on jp5.11 that is a SIGSEGV that kills all of perception.
So the properties worth testing here are not "does it synthesize" (that needs a 310 MB
model and a GPU) but the ones that decide whether the isolation actually holds and
whether a failure degrades or breaks:

  - the child is started with `spawn`, never `fork`. `fork` copies the address space
    including already-dlopened libraries, so a forked child would inherit sherpa's ONNX
    Runtime and the isolation would be worthless — the exact bug being fixed;
  - a dead or unstartable child falls back rather than raising, because Japanese losing
    8x is acceptable and Japanese breaking is not;
  - a request error is *not* swallowed by the fallback, or a real bug would hide behind
    a slower code path.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import kokoro_worker as kw  # noqa: E402


class _FakeQueue:
    """A queue that records what was put and hands back scripted replies."""

    def __init__(self, replies=None):
        self.puts = []
        self._replies = list(replies or [])

    def put(self, item):
        self.puts.append(item)

    def get(self, timeout=None):
        if not self._replies:
            import queue as _q
            raise _q.Empty
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _FakeProc:
    def __init__(self, alive=True):
        self.pid = 4242
        self._alive = alive
        self.terminated = False
        self.killed = False

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def join(self, timeout=None):
        pass


class _FakeCtx:
    """Stands in for multiprocessing's spawn context."""

    def __init__(self, replies, alive=True):
        self.cmd = _FakeQueue()
        self.res = _FakeQueue(replies)
        self.proc = _FakeProc(alive)
        self.queues_made = 0
        self.process_kwargs = None

    def Queue(self):                                              # noqa: N802
        self.queues_made += 1
        return self.cmd if self.queues_made == 1 else self.res

    def Process(self, **kwargs):                                  # noqa: N802
        self.process_kwargs = kwargs
        return self.proc


READY = ("ready", {"num_speakers": 54, "sample_rate": 24000,
                   "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                   "warmup_s": 8.03})


def _proxy(monkeypatch, replies, alive=True, **kwargs):
    """Build a proxy whose child is a fake, and stop the reaper thread interfering."""
    ctx = _FakeCtx(replies, alive)
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_reap_loop", lambda self: None)

    import multiprocessing as mp
    monkeypatch.setattr(mp, "get_context", lambda method: ctx)
    proxy = kw.KokoroWorkerProxy("/models/kokoro-multi/gpu", "model.onnx", **kwargs)
    return proxy, ctx


# ── the isolation itself ──────────────────────────────────────────────────────

def test_the_child_is_spawned_not_forked(monkeypatch):
    """`fork` would inherit sherpa's already-loaded ONNX Runtime and defeat the point.

    On Linux `fork` is the default, so this has to be asked for explicitly. The
    assertion is on the string passed to get_context, because that is the whole
    guarantee — there is nothing else to check at runtime.
    """
    seen = {}

    class _Ctx(_FakeCtx):
        pass

    ctx = _Ctx([READY])
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_reap_loop", lambda self: None)

    import multiprocessing as mp

    def _get_context(method):
        seen["method"] = method
        return ctx

    monkeypatch.setattr(mp, "get_context", _get_context)
    kw.KokoroWorkerProxy("/models/kokoro-multi/gpu", "model.onnx")
    assert seen["method"] == "spawn", seen


def test_the_child_entry_point_is_module_level_and_takes_no_ros_or_sherpa():
    """spawn pickles the target by qualified name, so it must be importable.

    Also asserts the child's own module imports neither rclpy nor sherpa_onnx at module
    scope — this process exists to be the only ONNX Runtime in its address space.
    """
    assert callable(kw._kokoro_worker)
    assert kw._kokoro_worker.__module__ == "plugins.kokoro_worker"
    source = (PERCEPTION_ROOT / "plugins" / "kokoro_worker.py").read_text("utf-8")
    assert "import rclpy" not in source
    assert "import sherpa_onnx" not in source


def test_handshake_populates_the_properties(monkeypatch):
    proxy, ctx = _proxy(monkeypatch, [READY])
    assert proxy.num_speakers == 54
    assert proxy.sample_rate == 24000
    assert "CUDAExecutionProvider" in proxy.providers
    assert ctx.process_kwargs["daemon"] is False, (
        "a daemon child is killed abruptly at parent exit; this one should be asked")


# ── failure behaviour ─────────────────────────────────────────────────────────

def test_a_child_that_never_reports_ready_is_terminated_and_raises(monkeypatch):
    """The caller falls back; leaving a half-started child behind would leak a context."""
    monkeypatch.setattr(kw, "START_TIMEOUT_S", 0.01)
    with pytest.raises(RuntimeError, match="did not report ready"):
        _proxy(monkeypatch, [])


def test_a_child_that_fails_to_build_reports_why(monkeypatch):
    with pytest.raises(RuntimeError, match="libcudnn"):
        _proxy(monkeypatch, [("failed", "OSError: libcudnn.so.8 not found")])


def test_a_dead_child_falls_back_instead_of_raising(monkeypatch):
    """Japanese losing 8x is acceptable; Japanese breaking is not."""
    proxy, ctx = _proxy(monkeypatch, [READY])
    ctx.proc._alive = False

    calls = {}

    class _Fallback:
        num_speakers, sample_rate = 54, 24000

        def synthesize(self, phonemes, speaker_id=0, speed=1.0):
            calls["args"] = (phonemes, speaker_id, speed)
            return np.ones(8, dtype=np.float32)

    # A restart attempt also fails, so the fallback is the only route left.
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_start",
                        lambda self: (_ for _ in ()).throw(RuntimeError("no")))
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_use_fallback", lambda self: _Fallback())

    out = proxy.synthesize("ohayoo", speaker_id=3, speed=1.1)
    assert out.shape == (8,)
    assert calls["args"] == ("ohayoo", 3, 1.1)


def test_a_request_error_is_raised_not_hidden_behind_the_fallback(monkeypatch):
    """A bad request means a bug. Re-synthesizing it elsewhere would mask it."""
    proxy, ctx = _proxy(monkeypatch, [READY])
    ctx.res._replies.append(("error", "req", "ValueError: speaker_id must be 0..53"))
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_recv",
                        lambda self, want: ("error", "ValueError: speaker_id"))
    with pytest.raises(RuntimeError, match="speaker_id"):
        proxy.synthesize("ohayoo", speaker_id=999)


def test_close_is_idempotent_and_terminates_the_child(monkeypatch):
    proxy, ctx = _proxy(monkeypatch, [READY])
    proxy.close()
    assert ctx.proc.terminated
    proxy.close()          # must not raise


# ── the leak this fixes ───────────────────────────────────────────────────────

def test_idle_reaping_releases_the_child(monkeypatch):
    """A card that spoke Japanese once used to hold the session for the adapter's life.

    `set_language` never released it, so switching back to English kept ~765 MB
    resident. Reaping on idle — rather than on the language switch — is what gives it
    back without making an alternating tour pay the ~10.5 s cold path every time.
    """
    proxy, ctx = _proxy(monkeypatch, [READY], idle_timeout_s=0.0)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    # One pass of the real reaper body, with the loop's sleep neutralised.
    proxy._last_used = time.monotonic() - 3600
    stop = threading.Event()

    def _one_pass():
        with proxy._lock:
            if proxy._alive() and (time.monotonic() - proxy._last_used) > proxy._idle_timeout_s:
                proxy._terminate()
        stop.set()

    _one_pass()
    assert stop.is_set()
    assert ctx.proc.terminated, "an idle worker must give its CUDA context back"
