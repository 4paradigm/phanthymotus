"""Short-lived, unguessable in-memory JPEG URLs for official VOP URL inference.

VOP does not accept base64. Do not upload files into another service/container
or assume that /tmp is shared. These bytes belong only to this read-only Card.
"""

from contextlib import contextmanager
import secrets
import threading
import time
from .contracts import need

PREFIX = "/piper-goal-frame/"


class FrameExchange:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.lock = threading.Lock()
        self.frames = {}

    @contextmanager
    def expose(self, jpeg):
        need(
            isinstance(jpeg, bytes) and 0 < len(jpeg) <= 8 * 1024 * 1024,
            "IMAGE_TOO_LARGE",
        )
        token = secrets.token_urlsafe(32)
        with self.lock:
            need(len(self.frames) < 8, "IMAGE_EXCHANGE_BUSY")
            self.frames[token] = (jpeg, time.monotonic() + 30)
        url = self.base + PREFIX + token
        try:
            yield url
        finally:
            with self.lock:
                self.frames.pop(token, None)

    def get(self, path):
        if not path.startswith(PREFIX):
            return None
        with self.lock:
            record = self.frames.get(path[len(PREFIX) :])
            return record[0] if record and record[1] >= time.monotonic() else None

    def clear(self):
        with self.lock:
            self.frames.clear()
