"""Robot-independent JPEG QR decoding and bounded content deduplication."""
from __future__ import annotations

class QrDecoder:
    """Decode JPEG frames. Pixel coordinates refer to the original image."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.detector = cv2.QRCodeDetector()

    def decode(self, jpeg: bytes) -> list[dict]:
        import numpy as np
        if not jpeg or len(jpeg) > 5_000_000:
            raise ValueError("empty or oversized JPEG")
        image = self.cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), self.cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid JPEG")
        ok, texts, points, _ = self.detector.detectAndDecodeMulti(image)
        # The single-code path can succeed where multi-code detection fails.
        if not ok or not any(texts):
            text, corners, _ = self.detector.detectAndDecode(image)
            texts, points = ([text], corners) if text else ([], None)
        if points is None:
            return []
        codes = []
        for text, corners in zip(texts, points):
            if not text:
                continue
            codes.append({
                "text": text,
                "corners_px": [[float(x), float(y)] for x, y in corners],
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
            })
        return codes


class QrTracker:
    """Emit once per content until absent for rearm_s. Bounded memory."""

    def __init__(self, rearm_s: float):
        self.rearm_s = rearm_s
        self.last_seen: dict[str, float] = {}
        self.sequence = 0

    def update(self, codes: list[dict], now: float) -> list[dict]:
        self.last_seen = {k: t for k, t in self.last_seen.items()
                          if now - t < self.rearm_s}
        events = []
        for code in codes:
            text = code["text"]
            if text not in self.last_seen:
                self.sequence += 1
                events.append({"sequence": self.sequence, "text": text})
            self.last_seen[text] = now
        # Detection input is physically bounded by image size; also bound history.
        if len(self.last_seen) > 256:
            self.last_seen = dict(sorted(self.last_seen.items(),
                                         key=lambda item: item[1])[-256:])
        return events


def self_check():
    """Catch OpenCV builds with detection but without QR decoding support."""
    import base64
    import numpy as np
    import cv2
    png = base64.b64decode('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')
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    ok, jpeg = cv2.imencode('.jpg', image)
    assert ok, 'OpenCV JPEG encoding unavailable'
    result = QrDecoder().decode(jpeg.tobytes())
    assert [c['text'] for c in result] == ['QR-SELF-CHECK'], 'OpenCV QR decoding unavailable'
