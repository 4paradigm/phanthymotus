"""Manage one ROS 2 Humble bag recorder process.

This module deliberately does not import ROS: the child inherits the ROS
environment of the hosting service and reports failure if the CLI is absent.
"""

from __future__ import annotations

import math
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path


class Recorder:
    """Own at most one recording; output paths are generated under ``root``."""

    def __init__(self, root, ros2_executable="ros2"):
        self._root = Path(root).expanduser().resolve()
        self._executable = ros2_executable
        self._lock = threading.RLock()
        self._process = None
        self._state = "idle"
        self._message = ""
        self._path = None
        self._topics = []
        self._started = None
        self._duration = 0.0
        self._max_duration = 0.0
        self._min_free = 0
        self._reason = ""
        self._watch_stop = None

    def _snapshot(self):
        duration = (time.monotonic() - self._started
                    if self._started is not None else self._duration)
        return {
            "state": self._state,
            "message": self._message,
            "path": str(self._path) if self._path else None,
            "topics": list(self._topics),
            "duration_sec": round(max(0.0, duration), 2),
            "stop_reason": self._reason or None,
        }

    def _error(self, message):
        result = self._snapshot()
        result.update(state="error", message=message)
        return result

    def start(self, topics, *, max_duration_sec=0, min_free_bytes=0):
        with self._lock:
            self._refresh()
            if self._process is not None:
                return self._error("A recording is already running")
            if (not isinstance(topics, (list, tuple)) or not topics or
                    any(not isinstance(t, str) or
                        not re.fullmatch(r"/[A-Za-z0-9_~/]+", t) for t in topics)):
                return self._error("Provide at least one valid absolute ROS topic name")
            try:
                max_duration = float(max_duration_sec)
                min_free = int(min_free_bytes)
                if (not math.isfinite(max_duration) or max_duration < 0 or
                        min_free < 0 or isinstance(min_free_bytes, bool)):
                    raise ValueError("limits must be non-negative")
                self._root.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(self._root).free < min_free:
                    self._state = "error"
                    self._message = "Insufficient free space for recording"
                    self._path = None
                    self._topics = list(dict.fromkeys(topics))
                    self._duration = 0.0
                    self._reason = ""
                    return self._snapshot()
                # The rosbag CLI requires a path which does not already exist.
                path = self._root / (time.strftime("bag_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12])
                command = [self._executable, "bag", "record", "-o", str(path), *topics]
                # Inherit the service logs: this avoids a blocked unread pipe
                # and preserves ROS/storage errors for diagnosis.
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
            except (OSError, TypeError, ValueError, OverflowError) as exc:
                self._state = "error"
                self._message = f"Could not start ROS bag recorder: {exc}"
                self._path = None
                self._topics = list(dict.fromkeys(topics))
                self._duration = 0.0
                self._reason = ""
                return self._snapshot()
            self._process = process
            self._path = path
            self._topics = list(dict.fromkeys(topics))
            self._started = time.monotonic()
            self._duration = 0.0
            self._max_duration = max_duration
            self._min_free = min_free
            self._state = "recording"
            self._message = "Recorder started; message reception is not yet verified"
            self._reason = ""
            self._watch_stop = threading.Event()
            threading.Thread(target=self._watch, args=(process, self._watch_stop),
                             name="rosbag-recorder-watch", daemon=True).start()
            return self._snapshot()

    def _refresh(self):
        if self._process is not None and self._process.poll() is not None:
            code = self._process.returncode
            self._process = None
            if self._watch_stop:
                self._watch_stop.set()
            if self._started is not None:
                self._duration = time.monotonic() - self._started
                self._started = None
            self._state = "error"
            self._message = f"Recorder exited unexpectedly (code {code}); bag may be incomplete"

    def _watch(self, process, stop_event):
        while not stop_event.wait(1.0):
            with self._lock:
                if self._process is not process:
                    return
                self._refresh()
                if self._process is None:
                    return
                if self._max_duration and time.monotonic() - self._started >= self._max_duration:
                    self._finish("duration_limit")
                    return
                try:
                    if self._min_free and shutil.disk_usage(self._root).free < self._min_free:
                        self._finish("disk_space_limit")
                        return
                except OSError as exc:
                    self._finish("disk_check_failed")
                    self._state = "error"
                    self._message = f"Free-space check failed: {exc}; verify the bag"
                    return

    def _finish(self, reason):
        process = self._process
        if process is None:
            return self._snapshot()
        self._state = "stopping"
        self._reason = reason
        if self._watch_stop:
            self._watch_stop.set()
        try:
            process.send_signal(signal.SIGINT)
            code = process.wait(timeout=15)
            if code != 0 or not self._path.joinpath("metadata.yaml").is_file():
                self._state = "error"
                self._message = f"Recording did not close cleanly (code {code}); bag may be incomplete"
            else:
                self._state = "completed"
                self._message = "Recording stopped and metadata written"
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._state = "error"
            self._message = "Recorder did not stop in time; bag may be incomplete"
        except OSError as exc:
            self._state = "error"
            self._message = f"Could not stop recorder: {exc}; bag may be incomplete"
        if self._started is not None:
            self._duration = time.monotonic() - self._started
            self._started = None
        # Never lose track of a still-running process on a failed stop.
        if process.poll() is not None:
            self._process = None
        return self._snapshot()

    def stop(self):
        with self._lock:
            self._refresh()
            return self._finish("requested")

    def info(self):
        with self._lock:
            self._refresh()
            return self._snapshot()

    def shutdown(self):
        with self._lock:
            self._refresh()
            return self._finish("shutdown")
