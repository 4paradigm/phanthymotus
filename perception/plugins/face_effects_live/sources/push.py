import threading
import time
import math
from ..contracts import Frame


class PushSource:
    """Driver-neutral ingestion: caller supplies BGR pixels; no vendor SDK."""
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.eof = False
        self.error = ""
        self.index = 0
        self.last_timestamp = None
        self.dropped = 0
        self.timestamp_repairs = 0

    def submit(self, image, timestamp=None, source="push", metadata=None):
        received = time.monotonic()
        original = received if timestamp is None else float(timestamp)
        with self.condition:
            if self.closed:
                return False
            timestamp = original
            extra = dict(metadata or {})
            if not math.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
                timestamp = (self.last_timestamp + 1e-6) if self.last_timestamp is not None else received
                self.timestamp_repairs += 1
                extra["timestamp_repaired"] = True
            self.last_timestamp = timestamp
            extra["source_timestamp_valid"] = math.isfinite(original)
            self.index += 1
            # Own the pixels: ROS/camera producers may reuse the incoming buffer.
            frame = Frame(image.copy(), timestamp, self.index, received, source=source, metadata=extra)
            if self.pending is not None:
                self.dropped += 1
            self.pending = frame
            self.condition.notify_all()
            return True

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.closed, timeout=.1)
            frame, self.pending = self.pending, None
            return frame

    def close(self):
        with self.condition:
            self.closed = self.eof = True
            self.pending = None
            self.condition.notify_all()
