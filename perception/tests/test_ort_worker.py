"""
tests/test_ort_worker.py — the process boundary that keeps two ONNX Runtimes apart.

This module exists because the standalone ONNX Runtime and sherpa-onnx's bundled one
share a single provider bridge — an 8 KB library holding one pointer to one runtime's
ProviderHost — so whichever builds a CUDA session second runs against the other's
framework objects. On jp6.1 that is an exception; on jp5.11 it is a SIGSEGV that kills
all of perception.

What is worth asserting here is not "does inference work" (that needs models and a GPU)
but the properties that decide whether the isolation holds at all, and whether a failure
degrades or breaks:

  - the child is started with `spawn`, never `fork`. `fork` copies the address space
    *including already-dlopened libraries*, so a forked child would inherit sherpa's
    runtime and the isolation would be worthless — the exact bug being fixed. This is
    the single most important assertion in the file;
  - the child imports neither rclpy nor sherpa_onnx, or it would not be the only ONNX
    Runtime in its address space;
  - `unload` releases one session without taking the others down, because face and
    Japanese share the child and Japanese unloads on idle;
  - a late reply is dropped rather than mistaken for the current one.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import numpy as np
import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import ort_worker as ow  # noqa: E402


HELLO = ("hello", {"version": "1.18.1",
                   "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]})
LOAD_REPLY = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
              "inputs": [{"name": "input.1", "shape": [1, 3, 640, 640],
                          "type": "tensor(float)"}],
              "outputs": [{"name": f"o{i}", "shape": [16800, 1],
                           "type": "tensor(float)"} for i in range(9)]}


class _Q:
    def __init__(self, replies=None):
        self.puts = []
        self._replies = list(replies or [])

    def put(self, item):
        self.puts.append(item)

    def get(self, timeout=None):
        if not self._replies:
            raise queue.Empty
        return self._replies.pop(0)


class _Proc:
    def __init__(self):
        self.pid = 1234
        self._alive = True
        self.terminated = False

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self._alive = False

    def join(self, timeout=None):
        pass


class _Ctx:
    """Stands in for multiprocessing's spawn context, recording what was asked for."""

    def __init__(self, res_replies):
        self.cmd = _Q()
        self.res = _Q(res_replies)
        self.proc = _Proc()
        self.n_queues = 0
        self.process_kwargs = None

    def Queue(self):                                              # noqa: N802
        self.n_queues += 1
        return self.cmd if self.n_queues == 1 else self.res

    def Process(self, **kwargs):                                  # noqa: N802
        self.process_kwargs = kwargs
        return self.proc


def _worker(monkeypatch, res_replies):
    ctx = _Ctx(res_replies)
    seen = {}

    import multiprocessing as mp

    def _get_context(method):
        seen["method"] = method
        return ctx

    monkeypatch.setattr(mp, "get_context", _get_context)
    return ow.OrtWorker(), ctx, seen


# ── the isolation itself ──────────────────────────────────────────────────────

def test_the_child_is_spawned_not_forked(monkeypatch):
    """`fork` would inherit sherpa's already-loaded runtime and defeat the whole point.

    On Linux `fork` is the default, so this has to be asked for explicitly, and the
    string passed to get_context is the entire guarantee — there is nothing else to
    check at runtime.
    """
    w, ctx, seen = _worker(monkeypatch, [HELLO, ("id", "ok", LOAD_REPLY)])
    w._ensure_started()
    assert seen["method"] == "spawn", seen
    assert ctx.process_kwargs["daemon"] is True, (
        "non-daemon deadlocks at exit: multiprocessing's own atexit hook joins "
        "non-daemon children, and it runs before ours, so it waits for ever on a child "
        "blocked in cmd_q.get(). Measured — see the comment at the Process() call.")


def test_the_child_holds_no_other_runtime():
    """spawn pickles the target by qualified name, so it must be importable — and the
    child must not import the runtime it is being isolated from."""
    assert callable(ow._ort_worker_main)
    assert ow._ort_worker_main.__module__ == "plugins.ort_worker"
    source = (PERCEPTION_ROOT / "plugins" / "ort_worker.py").read_text("utf-8")
    assert "import rclpy" not in source
    assert "import sherpa_onnx" not in source


def test_no_standalone_session_is_created_outside_the_worker():
    """The invariant the whole change rests on: every `ort.InferenceSession` in the
    plugin tree is either inside the worker child or an explicitly-degraded fallback.

    Checked structurally rather than by convention, because a new caller adding one
    would reintroduce the collision silently — and on jp5.11 that is a SIGSEGV.
    """
    import ast

    allowed = {
        "ort_worker.py",          # the child itself, which is the point
        "face_runtime.py",        # `_open_session` fallback, warned about
        "kokoro_direct.py",       # `in_process=True` fallback, cpu-only
    }
    offenders = {}
    for path in (PERCEPTION_ROOT / "plugins").rglob("*.py"):
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "InferenceSession"):
                if path.name not in allowed:
                    offenders.setdefault(path.name, []).append(node.lineno)
    assert not offenders, (
        f"these create a standalone ORT session outside the worker: {offenders}. "
        "See plugins/ort_worker.py for why that collides with sherpa-onnx.")


# ── the protocol ──────────────────────────────────────────────────────────────

def test_load_returns_a_session_shaped_proxy(monkeypatch):
    w, ctx, _ = _worker(monkeypatch, [HELLO, ("x", "ok", LOAD_REPLY)])
    monkeypatch.setattr(ow.OrtWorker, "_request",
                        lambda self, cmd, timeout: LOAD_REPLY)
    sess = w.load("face.det", "/models/det.onnx", ["CUDAExecutionProvider"], {})
    assert sess.get_providers()[0] == "CUDAExecutionProvider"
    assert sess.get_inputs()[0].name == "input.1"
    assert len(sess.get_outputs()) == 9, (
        "the SCRFD decoder hard-asserts nine outputs; the proxy must report them all")


def test_unload_drops_one_session_and_keeps_the_others(monkeypatch):
    """Japanese unloads on idle; face's sessions are in the same child and must live."""
    w, ctx, _ = _worker(monkeypatch, [HELLO])
    sent = []
    monkeypatch.setattr(ow.OrtWorker, "_request",
                        lambda self, cmd, timeout: sent.append(cmd) or LOAD_REPLY)
    w.load("face.det", "/models/det.onnx", ["CPUExecutionProvider"], {})
    w.load("tts.kokoro.ja", "/models/kokoro.onnx", ["CPUExecutionProvider"], {})
    w.unload("tts.kokoro.ja")
    assert "face.det" in w._loaded
    assert "tts.kokoro.ja" not in w._loaded
    assert ("unload", "tts.kokoro.ja") in sent
    assert w.alive, "unloading a session must not kill the child"


def test_a_stale_reply_is_dropped_not_returned(monkeypatch):
    """Requests are serialised, so an out-of-order id means a call that already timed
    out. Returning it would hand one caller another's tensors."""
    w, ctx, _ = _worker(monkeypatch, [HELLO])
    w._ensure_started()
    # A reply for a request nobody is waiting on, then the real one.
    ctx.res._replies.extend([("someone-else", "ok", "stale"), (None, "ok", "fresh")])

    real_id = {}
    original_put = ctx.cmd.put

    def _capture(item):
        real_id["id"] = item[0]
        ctx.res._replies[-1] = (item[0], "ok", "fresh")
        original_put(item)

    ctx.cmd.put = _capture
    assert w._request(("run", "k", None, {}), timeout=5) == "fresh"


def test_a_worker_error_becomes_an_exception_the_caller_can_catch(monkeypatch):
    w, ctx, _ = _worker(monkeypatch, [HELLO])
    w._ensure_started()

    def _put(item):
        ctx.res._replies.append((item[0], "error", "RuntimeError: no such session"))

    ctx.cmd.put = _put
    with pytest.raises(ow.OrtWorkerError, match="no such session"):
        w._request(("run", "k", None, {}), timeout=5)


def test_close_is_idempotent(monkeypatch):
    w, ctx, _ = _worker(monkeypatch, [HELLO])
    w._ensure_started()
    w.close()
    assert ctx.proc.terminated
    w.close()


# ── the escape hatch ──────────────────────────────────────────────────────────

def test_the_worker_is_on_by_default(monkeypatch):
    monkeypatch.delenv("PERCEPTION_ORT_WORKER", raising=False)
    assert ow.worker_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", " off "])
def test_it_can_be_turned_off_for_bisecting(monkeypatch, value):
    """Off restores the collision, so this exists to isolate a problem, not to run in."""
    monkeypatch.setenv("PERCEPTION_ORT_WORKER", value)
    assert ow.worker_enabled() is False
