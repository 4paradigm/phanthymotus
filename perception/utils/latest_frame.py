"""
utils/latest_frame.py — Single-slot "latest frame wins" buffer.

Real-time vision plugins must not accumulate sensor history: when the
inference worker is slower than the camera, every frame that arrives while
the worker is busy replaces the previous one, and the worker always picks
up the newest frame. Replaces the `queue.Queue(maxsize=N)` + drop-oldest
juggling in the ROS callbacks.

    camera callback → LatestFrame.push(frame)     (overwrites, never blocks)
    worker loop     → LatestFrame.pop(timeout)    (newest frame or None)
    stop()          → LatestFrame.close()          (wakes the worker, pop → None)
"""

from __future__ import annotations

import threading
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class LatestFrame(Generic[T]):
    """Thread-safe single-slot buffer where push() overwrites the pending frame."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[T] = None
        self._has_frame = False
        self._closed = False
        self._dropped = 0

    def push(self, frame: T) -> None:
        """Store `frame`, replacing any frame not yet consumed. Never blocks."""
        with self._cond:
            if self._closed:
                return
            if self._has_frame:
                self._dropped += 1
            self._frame = frame
            self._has_frame = True
            self._cond.notify()

    def pop(self, timeout: Optional[float] = None) -> Optional[T]:
        """Return the newest frame, waiting up to `timeout` seconds.

        Returns None on timeout or once the buffer is closed.
        """
        with self._cond:
            if not self._has_frame and not self._closed:
                self._cond.wait(timeout)
            if not self._has_frame:
                return None
            frame = self._frame
            self._frame = None
            self._has_frame = False
            return frame

    def clear(self) -> None:
        with self._cond:
            self._frame = None
            self._has_frame = False

    def close(self) -> None:
        """Discard the pending frame and wake every waiter; pop() returns None."""
        with self._cond:
            self._closed = True
            self._frame = None
            self._has_frame = False
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def dropped(self) -> int:
        """Number of frames overwritten before being consumed (diagnostics)."""
        return self._dropped


__all__ = ["LatestFrame"]
