"""Persistent CUDA Whisper Q5 worker with built-in LID and language LoRA."""
from __future__ import annotations

import atexit
import ctypes.util
import io
import json
import logging
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time
import wave

from plugins.numeric_dictation import normalize_long_numeric_dictation
from plugins.asr_numeric import NumericOutputNormalizer

log = logging.getLogger(__name__)
LANGUAGES = ("en", "zh", "ja", "fr", "de", "es", "ru", "th")


class WhisperTurboAdapter:
    """One Q5 model; automatic language and resident Japanese/Thai/Russian LoRA."""

    def __init__(self, model_dir, device="gpu", num_threads=2):
        if device != "gpu":
            raise ValueError("Whisper Turbo Q5 requires CUDA")
        self._root = Path(model_dir)
        self._model = self._root / "model.bin"
        self._jetpack = os.environ.get("WHISPER_TURBO_JP_VERSION", "")
        self._override = os.environ.get("WHISPER_TURBO_WORKER")
        self._worker = Path(self._override) if self._override else (
            self._root / "runtime" / ("jp" + self._jetpack) / "bin" / "whisper_turbo_worker")
        self._lock = threading.RLock()
        self._process = None
        self._closed = False
        self._stderr = None
        self._numeric = None
        self.last_language = None
        self.last_language_probability = None

    def prepare(self, on_status=None):
        """Download and initialize only after a valid start request."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Whisper Turbo backend is closed")
            if self._process is not None and self._process.poll() is None:
                return
            if not self._override:
                from utils.install_whisper_turbo import ensure_release
                self._worker = ensure_release(self._root, self._jetpack, on_status)
            if on_status:
                on_status("Loading Whisper Turbo on GPU")
            manifest = self._root / "numeric.json"
            if manifest.is_file():
                config = json.loads(manifest.read_text())
                self._numeric = NumericOutputNormalizer(
                    self._root / config["grammar_dir"], config.get("model_names", ()))
            self._start()

    def _start(self):
        if self._closed:
            raise RuntimeError("Whisper Turbo backend is closed")
        if self._process is not None and self._process.poll() is None:
            return
        if not self._worker.is_file() or not self._model.is_file():
            raise FileNotFoundError("Whisper Turbo worker/model missing; call prepare() first")
        env = dict(os.environ)
        library_dir = self._worker.parent.parent / "lib"
        env["LD_LIBRARY_PATH"] = str(library_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
        # Keep CUDA lookup private to this child, including the JP5 launcher.
        preload = env.get("LD_PRELOAD", "").split()
        cuda = ctypes.util.find_library("cuda")
        if cuda and cuda not in preload:
            preload.append(cuda)
        if preload:
            env["LD_PRELOAD"] = " ".join(preload)
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                [str(self._worker), str(self._model)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._stderr, text=True, bufsize=1, env=env)
            if self._closed:
                raise RuntimeError("Whisper Turbo backend is closed")
            atexit.register(self.close)
            ready = self._read(60.0)
            if not ready.get("ready") or not ready.get("gpu"):
                raise RuntimeError("Whisper Turbo failed CUDA initialization")
        except Exception:
            self._close_worker()
            raise
        log.info("[asr] Whisper Turbo ready: load=%.2fs", ready.get("load_s", 0.0))

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Whisper Turbo worker response timeout")
            readable, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not readable:
                raise TimeoutError("Whisper Turbo worker response timeout")
            line = self._process.stdout.readline()
            if not line:
                self._stderr.seek(0, os.SEEK_END)
                self._stderr.seek(max(0, self._stderr.tell() - 2048))
                detail = self._stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError("Whisper Turbo worker exited: " + detail)
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                log.debug("[asr] ignored non-JSON Whisper worker output")
                continue
            if not isinstance(reply, dict):
                raise RuntimeError("Invalid Whisper Turbo worker response")
            return reply

    def _transcribe(self, wav_bytes):
        with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("ASR expects mono 16kHz PCM16 WAV audio")
            pcm = audio.readframes(audio.getnframes())
        if not pcm:
            self.last_language = None
            return "", ""
        with self._lock:
            self._start()
            handle, path = tempfile.mkstemp(prefix="whisper-turbo-", suffix=".wav")
            os.close(handle)
            try:
                with wave.open(path, "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(16000)
                    out.writeframes(pcm)
                self._process.stdin.write(json.dumps({"audio": path}) + "\n")
                self._process.stdin.flush()
                reply = self._read(130.0)
                if reply.get("error"):
                    raise RuntimeError("Whisper Turbo: " + str(reply["error"]))
            except Exception:
                # A timed-out response must never become the next utterance's
                # transcript; restart this worker rather than reuse its pipe.
                self._close_worker()
                raise
            finally:
                os.unlink(path)
            language = str(reply.get("predicted_language") or "").strip().lower()
            if language not in LANGUAGES:
                language = ""
            self.last_language = language or None
            text = str(reply.get("raw_text") or "").strip()
            rendered = (self._numeric.normalize(text, language) if self._numeric is not None
                        else normalize_long_numeric_dictation(text, language))
            return rendered, language

    def transcribe(self, wav_bytes, language="auto"):
        return self._transcribe(wav_bytes)[0]

    def transcribe_rich(self, wav_bytes, language="auto", include_emotion=False):
        text, detected = self._transcribe(wav_bytes)
        return text, ({"lang": detected} if detected else {})

    def transcribe_aligned(self, wav_bytes, language="auto", include_emotion=False):
        text, tags = self.transcribe_rich(wav_bytes, language, include_emotion)
        return text, tags, None

    def close(self):
        self._closed = True
        self._close_worker()

    def _close_worker(self):
        # A blocked recognition owns the request lock. Interrupt its child
        # first so stop cannot wait for the full recognition timeout.
        if not self._lock.acquire(blocking=False):
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            self._lock.acquire()
        try:
            process, self._process = self._process, None
            atexit.unregister(self.close)
            if process is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                process.stdout.close()
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
        finally:
            self._lock.release()
