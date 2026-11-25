from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import cv2
from ..contracts import Frame


class FileSource:
    def __init__(self, path, paced=True):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("video file missing")
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("cannot decode local video")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not .1 <= self.fps <= 240:
            self.fps = 30.
        self.paced, self.index = bool(paced), 0
        self.stop = threading.Event()
        self.origin = None
        self.eof, self.error = False, ""

    def read(self):
        if self.stop.is_set() or self.eof:
            return None
        if self.origin is None:
            self.origin = time.monotonic()
        if self.paced and self.stop.wait(max(0, self.origin+self.index/self.fps-time.monotonic())):
            return None
        ok, image = self.capture.read()
        received = time.monotonic()
        if not ok:
            self.eof = True
            return None
        frame = Frame(image, self.index/self.fps, self.index, received, source="file")
        self.index += 1
        return frame

    def close(self):
        # Capture release occurs on the reading thread via finish(); no concurrent
        # VideoCapture.read/release on an FFmpeg object.
        self.stop.set()

    def finish(self):
        self.capture.release()


class USBSource:
    def __init__(self, device, width=None, height=None, fps=None):
        if isinstance(device, bool) or not isinstance(device, (str, int)) or device == "":
            raise ValueError("explicit USB device/index required")
        self.device = device
        self.options = [(cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                        (cv2.CAP_PROP_FPS, fps)]
        self.capture = None
        self.stop = threading.Event()
        self.index = self.reconnects = 0
        self.eof, self.error = False, ""

    def _open(self):
        self.capture = cv2.VideoCapture(self.device)
        for prop, value in self.options:
            if value is not None:
                self.capture.set(prop, float(value))

    def read(self):
        if self.stop.is_set():
            return None
        if self.capture is None:
            self._open()
        ok, image = self.capture.read()
        received = time.monotonic()
        if not ok:
            self.error = "video disconnected; retrying"
            self.capture.release()
            self.capture = None
            self.reconnects += 1
            self.stop.wait(.25)
            return None
        self.error = ""
        self.index += 1
        return Frame(image, received, self.index, received, source="usb")

    def close(self):
        self.stop.set()

    def finish(self):
        if self.capture is not None:
            self.capture.release()


class RTSPSource(USBSource):
    def __init__(self, url):
        parts = urlsplit(url)
        if parts.scheme not in ("rtsp", "rtsps") or not parts.hostname:
            raise ValueError("expected RTSP URL")
        super().__init__(url)

    def _open(self):
        # Supported by the FFmpeg backend. Never log credential-bearing URLs.
        self.capture = cv2.VideoCapture(self.device, cv2.CAP_FFMPEG,
                                        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500,
                                         cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000])

    def read(self):
        frame = super().read()
        if frame is None:
            return None
        return Frame(frame.image, frame.timestamp, frame.frame_id, frame.received, source="rtsp")
