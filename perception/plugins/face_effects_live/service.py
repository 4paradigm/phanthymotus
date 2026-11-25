"""Latest-frame processing with bounded history and explicit thread ownership."""
from collections import deque
import threading
import time
import cv2
from .backends import YuNetError
from .engine import Engine


class Service:
    def __init__(self, config, source, on_result=None):
        self.config, self.source, self.on_result = dict(config), source, on_result
        self.condition = threading.Condition()
        self.lifecycle = threading.RLock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.capture_done = threading.Event()
        self.pending = None
        self.jpeg = None
        self.latest = None
        self.last_output_at = 0.
        self.state, self.error = "idle", ""
        self.overwritten = self.processed = self.stale = 0
        self.samples = deque(maxlen=1200)
        self.processor = self.capture = None
        self.requested_effects = None

    def start(self):
        with self.lifecycle:
            if self.processor is not None:
                raise RuntimeError("Service instances are single-use")
            self.state = "starting"
            self.processor = threading.Thread(target=self._process, name="face-effects-process", daemon=True)
            self.processor.start()
            if not self.ready.wait(15):
                self.stop()
                raise RuntimeError("inference initialization timed out")
            if self.state == "error":
                self.stop()
                raise RuntimeError(self.error)
            self.capture = threading.Thread(target=self._capture, name="face-effects-capture", daemon=True)
            self.capture.start()
        return self.status()

    def select(self, names):
        with self.condition:
            self.requested_effects = list(names)

    def _capture(self):
        try:
            while not self.stop_event.is_set():
                frame = self.source.read()
                if frame is None:
                    if getattr(self.source, "eof", False):
                        break
                    continue
                with self.condition:
                    if self.pending is not None:
                        self.overwritten += 1
                    self.pending = frame
                    self.condition.notify_all()
        except Exception as exc:
            self._fail("input", exc)
        finally:
            finish = getattr(self.source, "finish", None)
            if finish:
                finish()
            self.capture_done.set()
            with self.condition:
                self.condition.notify_all()

    def _fail(self, stage, exc):
        # No arbitrary exception strings: capture/SDK errors can contain URLs.
        with self.condition:
            detail = str(exc) if isinstance(exc, YuNetError) else type(exc).__name__
            self.error = f"{stage}: {detail}"
            self.state = "error"
            self.jpeg = self.latest = None
            self.stop_event.set()
            self.condition.notify_all()

    def _process(self):
        engine = None
        try:
            engine = Engine(self.config)
            self.state = "running"
            self.ready.set()
            while not self.stop_event.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.stop_event.is_set()
                                            or self.capture_done.is_set(), timeout=.1)
                    if self.stop_event.is_set():
                        break
                    frame, self.pending = self.pending, None
                    selection, self.requested_effects = self.requested_effects, None
                if selection is not None:
                    engine.effects.select(selection)
                if frame is None:
                    if self.capture_done.is_set():
                        break
                    continue
                result = engine.process(frame)
                if result is None:
                    self.stale += 1
                    continue
                start = time.monotonic()
                ok, encoded = cv2.imencode(".jpg", result.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                jpeg = encoded.tobytes()
                result.metadata["jpeg_ms"] = (time.monotonic()-start)*1000
                result.metadata["receive_to_encode_ms"] = (time.monotonic()-frame.received)*1000
                if time.monotonic()-frame.received > engine.max_age:
                    self.stale += 1
                    continue
                if self.stop_event.is_set():
                    break
                if self.on_result:
                    self.on_result(jpeg, result.metadata)
                with self.condition:
                    self.jpeg, self.latest = jpeg, result.metadata
                    self.last_output_at = time.monotonic()
                    self.processed += 1
                    self.samples.append(result.metadata["receive_to_encode_ms"])
            if self.state != "error":
                self.state = "stopped" if self.stop_event.is_set() else "eof"
        except Exception as exc:
            self._fail("initialization" if engine is None else "processing", exc)
        finally:
            if engine is not None:
                try:
                    engine.close()
                except Exception as exc:
                    self._fail("close", exc)
            self.source.close()
            self.ready.set()

    def status(self):
        import numpy as np
        with self.condition:
            age = None if self.latest is None else self.latest.get("receive_to_encode_ms")
            return {"state": self.state, "error": self.error,
                    "input_error": getattr(self.source, "error", ""),
                    "processed": self.processed, "overwritten": self.overwritten,
                    "stale": self.stale, "source_dropped": getattr(self.source, "dropped", 0),
                    "timestamp_repairs": getattr(self.source, "timestamp_repairs", 0),
                    "last_receive_to_encode_ms": age,
                    "latency_ms": dict(zip(("p50", "p95", "p99"), np.percentile(self.samples, [50, 95, 99]).tolist())) if self.samples else None,
                    "result": self.latest}

    def stop(self):
        with self.lifecycle:
            self.stop_event.set()
            self.source.close()
            with self.condition:
                self.pending = None
                self.condition.notify_all()
            for thread in (self.capture, self.processor):
                if thread and thread is not threading.current_thread():
                    thread.join(3)
            alive = any(t and t.is_alive() for t in (self.capture, self.processor))
            if alive:
                self.state, self.error = "error", "worker did not stop; restart prohibited"
            elif self.state != "error":
                self.state = "stopped"
            if self.capture is None:
                finish = getattr(self.source, "finish", None)
                if finish:
                    finish()
            with self.condition:
                self.jpeg = self.latest = None
            return self.status()
