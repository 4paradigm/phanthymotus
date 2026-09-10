#!/usr/bin/env python3
"""
plugins/ort_worker.py — the standalone ONNX Runtime, in a process of its own.

## Why this exists

Perception is one process holding **two** ONNX Runtimes: sherpa-onnx bundles its own,
and the standalone `onnxruntime` wheel serves face recognition and Kokoro's Japanese
path. They corrupt each other, and the reason is in the dynamic linker rather than in
either library.

`libonnxruntime_providers_shared.so` is an 8 KB library exporting exactly
`Provider_GetHost` and `Provider_SetHost` — a process-global slot holding **one** pointer
to **one** runtime's `ProviderHost`. It carries a SONAME, and `ld.so` deduplicates a
`dlopen` by matching the requested basename against already-loaded objects, so the second
runtime's copy is never mapped: both get the first one. Each runtime writes its own host
into that slot as it loads a provider, **last writer wins**, and the next session built
runs against the other runtime's framework objects:

    Error mapping output names: Could not find OrtValue with name '/Squeeze_2_output_0'

Measured on Orin 6 with `Provider_GetHost()` observed changing value and only ever one
bridge mapped. What it costs, per JetPack line:

| | |
|---|---|
| jp6.1 | an exception — the affected card goes `state: error` |
| jp5.11 | **SIGSEGV, killing the whole perception process** — ASR, VOP, OCR and face with it |

And it is not order-fixable, only order-*avoidable*: with face's CUDA session built
first, `sherpa_onnx.OfflineTts` cannot be constructed at all. Releasing face's session
does not help — measured: the claim on the bridge outlives the session object, so there is
no in-process recovery. Production only escapes it because `main.py` happens to construct
the TTS plugin (`:112-117`) before face (`:140-147`); switching the TTS engine to
`kokoro-multi` while face is on gpu breaks it today.

## What this does

One child process owns **every** standalone-ORT session, so the parent holds only
sherpa's runtime and the collision cannot occur. Not one process per card — one process
per *runtime*, which is the boundary the bug actually has. `plugins/kokoro_worker.py`
proved the shape on the Japanese path; this generalises it and brings face along.

CUDA contexts do not increase: the parent used to hold sherpa's **and** the standalone
one's; now it holds sherpa's and the child holds the standalone one.

## What crosses, and what it costs

The boundary is the two `run()` calls, nothing else. Every bit of pre- and
post-processing — letterboxing, `_distance2bbox`, NMS, Umeyama alignment, the FaceDB
`matrix @ embedding` — is a pure function on arrays and stays in the parent.

Measured on Orin 6, round trip through a `Queue`:

    detection   in 1x3x640x640 f32  2.93 MiB   out 9 arrays  0.96 MiB   ~13 ms
    recognition in 1x3x112x112 f32   147 kB    out (1,512)    2 kB      ~0.4 ms
    Japanese    a phoneme string             out ~0.89 MB              ~2.7 ms

At `detect_fps: 1.0`, the default and the documented design point
(`config.yaml`: "cpu at 1 detection/second is the design point"), that is **~1.3% of a
second**. Shared memory measured 2.4-3x faster (3.27 ms against 9.07 ms for a 2.76 MB
payload) and is the answer if anyone runs `detect_fps: 0` against a fast camera, but it
is not needed at the design point and a `Queue` is far less to get wrong.

The burst paths are the ones to watch, not the steady state: `register_by_stream`
analyses up to 8 frames back to back, `register_by_corpus` up to 200. Those pay ~13 ms
each on top of inference. They are human-initiated and were never latency-critical.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import os
import queue
import threading
import time
import uuid

import numpy as np

log = logging.getLogger(__name__)

# A cold CUDA session costs 2.5 s to build plus ~8 s for the first inference on Orin 6,
# and this may be loading the 310 MB Kokoro graph on a box also serving video.
LOAD_TIMEOUT_S = 90.0
# Detection is ~6 ms warm and a Japanese utterance ~0.6 s. This is a "the child is
# wedged" bound, not a budget.
RUN_TIMEOUT_S = 60.0


class OrtWorkerError(RuntimeError):
    """The child could not serve the request. Callers decide whether to fall back."""


class _SessionMeta:
    """Input/output names and shapes, so a proxy can answer without a round trip.

    `face_runtime.py` reads `get_inputs()[0].name` once at construction and caches it;
    the detection path also needs the nine output names. Both are static, so the child
    reports them at load time and the proxy answers locally afterwards.
    """

    __slots__ = ("name", "shape", "type")

    # `type` shadows the builtin, deliberately: onnxruntime's own NodeArg calls it that
    # and callers read `.type`, so matching the shape it stands in for beats tidiness.
    def __init__(self, name, shape, type):                        # noqa: A002
        self.name, self.shape, self.type = name, shape, type


class OrtWorker:
    """The child process, and the only place a standalone ORT session may be created.

    One instance per perception process, obtained through `get_worker()`. Sessions are
    loaded and unloaded independently, so Japanese can give its 310 MB back without
    disturbing face.
    """

    def __init__(self, num_threads: int = 0):
        self._num_threads = num_threads
        self._lock = threading.RLock()
        self._ctx = None
        self._proc = None
        self._cmd_q = None
        self._res_q = None
        # key -> (model_path, providers, options) so a restarted child can be refilled
        # without every caller having to notice it died.
        self._loaded = {}
        self._meta = {}
        self._closed = False
        atexit.register(self.close)

    # ── process lifecycle ────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        import multiprocessing as mp

        # `spawn`, never the platform default. On Linux that is `fork`, which copies the
        # address space *including already-dlopened libraries* — a forked child would
        # inherit sherpa's ONNX Runtime and the isolation this module exists for would
        # be worthless. This is the single most important line in the file.
        self._ctx = mp.get_context("spawn")
        self._cmd_q = self._ctx.Queue()
        self._res_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_ort_worker_main,
            args=(self._cmd_q, self._res_q, log.getEffectiveLevel()),
            # daemon, and that is not a style choice. `multiprocessing.util
            # ._exit_function` joins every non-daemon child at interpreter exit, and it
            # registers itself when multiprocessing is first imported — which happens
            # *after* this class registers its own atexit hook, so LIFO puts it first.
            # It then joins a child that is blocking on `cmd_q.get()` and waits for
            # ever. Measured: the verification script completed all its checks and then
            # hung at exit with the child still alive.
            #
            # A session host has nothing to flush, so dying with the parent is the
            # right semantics anyway; `close()` is still the normal path.
            daemon=True,
            name="ort_worker",
        )
        self._proc.start()
        try:
            kind, payload = self._res_q.get(timeout=30)
        except queue.Empty:
            self._terminate()
            raise OrtWorkerError("the ORT worker did not come up within 30s")
        if kind != "hello":
            self._terminate()
            raise OrtWorkerError(f"the ORT worker failed to start: {payload}")
        log.info("[ort] worker up: pid=%s onnxruntime=%s providers=%s",
                 self._proc.pid, payload["version"], payload["providers"])

        # Reload whatever was resident before it died, so a caller holding a proxy does
        # not have to know the child was replaced.
        stale, self._loaded = dict(self._loaded), {}
        for key, spec in stale.items():
            try:
                self._load_locked(key, *spec)
                log.info("[ort] reloaded %s after the worker restarted", key)
            except Exception as exc:                              # noqa: BLE001
                log.warning("[ort] could not reload %s: %s", key, exc)

    def _terminate(self) -> None:
        proc, self._proc = self._proc, None
        self._cmd_q = self._res_q = None
        self._meta = {}
        if proc is None:
            return
        try:
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        except Exception:                                         # noqa: BLE001
            pass

    def close(self) -> None:
        """Kill the child, releasing its CUDA context.

        A process exit is the only thing that returns a CUDA context at all — an
        in-process session never does. Safe to call more than once.
        """
        with self._lock:
            if self._closed and self._proc is None:
                return
            self._closed = True
            if self._proc is not None:
                log.info("[ort] worker closing (pid=%s)", self._proc.pid)
            self._terminate()
            self._loaded = {}

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    # ── sessions ─────────────────────────────────────────────────────────────

    def load(self, key: str, model_path: str, providers, options=None,
             postprocess=None):
        """Load a model in the child and return a proxy that behaves like a session.

        `key` is the caller's name for it — reused on a restart to bring it back, and
        used by `unload`. `options` is a plain dict, because `ort.SessionOptions` is not
        picklable; the child rebuilds it.

        `postprocess` is `{"module": ..., "func": ...}`, imported by the child and
        applied to the raw outputs before they are sent back. It exists because a
        detector head is enormous before thresholding and tiny after — SCRFD returns
        16 800 candidates of which a frame keeps nought to eight — so shipping the raw
        arrays across means throwing 99.95% of them away on the far side, measured at
        +27 ms per detection.

        Passing a module path rather than a callable keeps this class ignorant of what
        it is hosting: a closure could not be pickled to a spawned child anyway, and a
        name means the child imports the *same* implementation the in-process path uses
        rather than a copy of it.
        """
        with self._lock:
            self._closed = False
            self._ensure_started()
            return self._load_locked(key, model_path, list(providers),
                                     dict(options or {}),
                                     dict(postprocess) if postprocess else None)

    def _load_locked(self, key, model_path, providers, options, postprocess=None):
        reply = self._request(("load", key, model_path, providers, options,
                               postprocess),
                              timeout=LOAD_TIMEOUT_S)
        self._loaded[key] = (model_path, providers, options)
        self._meta[key] = reply
        return OrtSessionProxy(self, key, reply)

    def unload(self, key: str) -> None:
        """Drop one session, keeping the child and anything else it holds.

        This is why the child is shared rather than one per card: Japanese can give its
        310 MB back on idle without taking face's sessions down with it.
        """
        with self._lock:
            self._loaded.pop(key, None)
            self._meta.pop(key, None)
            if not self.alive:
                return
            try:
                self._request(("unload", key), timeout=30)
            except Exception as exc:                              # noqa: BLE001
                log.warning("[ort] unload %s failed: %s", key, exc)

    def run(self, key: str, output_names, feeds, post_kwargs=None):
        """Run the session. With a postprocessor registered, the reply is its return
        value rather than the raw outputs, and `post_kwargs` are its per-call
        arguments — the letterbox scale, for instance, which changes every frame."""
        with self._lock:
            if self._closed:
                raise OrtWorkerError("the ORT worker is closed")
            if not self.alive:
                log.warning("[ort] worker died; restarting and reloading its sessions")
                self._ensure_started()
            return self._request(("run", key, output_names, feeds, post_kwargs),
                                 timeout=RUN_TIMEOUT_S)

    def _request(self, command, timeout: float):
        """Send one command and wait for its reply. Caller holds `self._lock`."""
        req_id = uuid.uuid4().hex[:8]
        try:
            self._cmd_q.put((req_id,) + command)
        except Exception as exc:                                  # noqa: BLE001
            self._terminate()
            raise OrtWorkerError(f"could not reach the ORT worker: {exc}") from exc

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                raise OrtWorkerError(
                    f"no reply to {command[0]!r} within {timeout:.0f}s")
            try:
                got_id, kind, payload = self._res_q.get(timeout=remaining)
            except queue.Empty:
                continue
            if got_id != req_id:
                # Requests are serialised by `self._lock`, so this is a late reply to a
                # call that already timed out. Dropping it is right; waiting on it would
                # deadlock the next one.
                log.debug("[ort] dropping stale reply %s", got_id)
                continue
            if kind == "error":
                raise OrtWorkerError(payload)
            return payload


class OrtSessionProxy:
    """Enough of `onnxruntime.InferenceSession` for this codebase's two callers.

    `run`, `get_providers`, `get_inputs`, `get_outputs` — that is all
    `plugins/face_runtime.py` and `plugins/kokoro_direct.py` use. Metadata is answered
    locally from the load-time reply, so caching `get_inputs()[0].name` at construction
    costs no round trip.
    """

    def __init__(self, worker: OrtWorker, key: str, meta: dict):
        self._worker = worker
        self._key = key
        self._providers = list(meta["providers"])
        self._inputs = [_SessionMeta(**m) for m in meta["inputs"]]
        self._outputs = [_SessionMeta(**m) for m in meta["outputs"]]

    def run(self, output_names, feeds, run_options=None, post_kwargs=None):
        del run_options                     # never used by either caller
        return self._worker.run(self._key, output_names, feeds, post_kwargs)

    def get_providers(self):
        return list(self._providers)

    def get_inputs(self):
        return list(self._inputs)

    def get_outputs(self):
        return list(self._outputs)

    def unload(self):
        self._worker.unload(self._key)


# ── the single instance ───────────────────────────────────────────────────────

_worker = None
_worker_lock = threading.Lock()


def get_worker() -> OrtWorker:
    """The one ORT worker for this process, created on first use.

    A single child rather than one per plugin: two children would mean two CUDA
    contexts, and the point of moving out of the parent is to keep the count the same
    as before, not to raise it.
    """
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = OrtWorker()
        return _worker


def worker_enabled() -> bool:
    """Whether to route standalone-ORT sessions through the child.

    On by default. `PERCEPTION_ORT_WORKER=0` puts every session back in the perception
    process — which restores the collision, so it exists for bisecting a problem rather
    than as a supported configuration.
    """
    return os.environ.get("PERCEPTION_ORT_WORKER", "1").strip().lower() not in (
        "0", "false", "no", "off")


# ── the child ─────────────────────────────────────────────────────────────────

def _ort_worker_main(cmd_q, res_q, log_level: int) -> None:
    """Child entry point. Holds every standalone-ORT session and nothing else.

    Imports neither `rclpy` nor `sherpa_onnx`: this process exists to be the only ONNX
    Runtime in its address space, and the parent keeps the ROS nodes, the publishers and
    all pre/post-processing.
    """
    # A spawned child gets a fresh interpreter and does not inherit the parent's
    # sys.stdout, so the atomic writer has to be reinstalled — without it a subprocess
    # writing on the parent's fd 1 tears Docker log records.
    try:
        from utils import logsafe
        logsafe.install(check_fd=False)
    except Exception:                                             # noqa: BLE001
        pass

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
        datefmt='%H:%M:%S')
    wlog = logging.getLogger("ort.worker")

    try:
        import onnxruntime as ort
    except Exception as exc:                                      # noqa: BLE001
        res_q.put(("failed", f"{type(exc).__name__}: {exc}"))
        return

    res_q.put(("hello", {"version": ort.__version__,
                         "providers": ort.get_available_providers()}))

    sessions = {}

    def _meta(entries):
        return [{"name": e.name, "shape": list(e.shape), "type": e.type}
                for e in entries]

    while True:
        try:
            command = cmd_q.get()
        except (EOFError, OSError):
            return
        if command is None:
            return
        req_id, verb = command[0], command[1]

        try:
            if verb == "load":
                _, _, key, model_path, providers, options, postprocess = command
                opts = ort.SessionOptions()
                if options.get("intra_op_num_threads") is not None:
                    opts.intra_op_num_threads = int(options["intra_op_num_threads"])
                if options.get("log_severity_level") is not None:
                    opts.log_severity_level = int(options["log_severity_level"])
                level = options.get("graph_optimization_level")
                if level:
                    opts.graph_optimization_level = getattr(
                        ort.GraphOptimizationLevel, level)
                started = time.monotonic()
                sess = ort.InferenceSession(model_path, opts, providers=providers)
                post = None
                if postprocess:
                    module = importlib.import_module(postprocess["module"])
                    post = getattr(module, postprocess["func"])
                    wlog.info("%s post-processes in the worker via %s.%s",
                              key, postprocess["module"], postprocess["func"])
                sessions[key] = (sess, post)
                wlog.info("loaded %s in %.2fs: %s providers=%s",
                          key, time.monotonic() - started,
                          os.path.basename(model_path), sess.get_providers())
                res_q.put((req_id, "ok", {
                    "providers": sess.get_providers(),
                    "inputs": _meta(sess.get_inputs()),
                    "outputs": _meta(sess.get_outputs()),
                }))

            elif verb == "unload":
                _, _, key = command
                dropped = sessions.pop(key, None)
                del dropped
                wlog.info("unloaded %s", key)
                res_q.put((req_id, "ok", None))

            elif verb == "run":
                _, _, key, output_names, feeds, post_kwargs = command
                entry = sessions.get(key)
                if entry is None:
                    res_q.put((req_id, "error",
                               f"session {key!r} is not loaded in the worker"))
                    continue
                sess, post = entry
                outputs = sess.run(output_names, feeds)
                if post is not None:
                    # Threshold here, not in the parent: this is where the 0.96 MiB of
                    # candidates already lives, and the caller wants the survivors.
                    outputs = post(outputs, **(post_kwargs or {}))
                res_q.put((req_id, "ok", outputs))

            else:
                res_q.put((req_id, "error", f"unknown verb {verb!r}"))

        except Exception as exc:                                  # noqa: BLE001
            res_q.put((req_id, "error", f"{type(exc).__name__}: {exc}"))
