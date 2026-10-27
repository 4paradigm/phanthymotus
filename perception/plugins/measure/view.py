"""Draw measured boxes and labels on the camera frame.

A JPEG in, a JPEG out. The card decides when to publish it; this module only
answers whether the picture shows the same boxes the JSON reports.

**OpenCV, the same drawing stack navi already uses in this image.** Perception
already ships cv2 for the camera and depth cards; the overlay should not drag in
a second image library for four rectangles and some ASCII text.
"""

from __future__ import annotations

import numpy as np


def render(camera_jpeg: bytes, objects: list[dict], *, quality: int = 85) -> bytes:
    """Return a JPEG with every measured box and its W×H label drawn on it."""
    import cv2

    if not camera_jpeg:
        raise ValueError("empty camera frame")
    frame = cv2.imdecode(np.frombuffer(camera_jpeg, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("camera frame is not a decodable JPEG")

    height, width = frame.shape[:2]
    for obj in objects or ():
        box = obj.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite([x1, y1, x2, y2])):
            continue
        x1 = max(0, min(width - 1, int(round(x1 * width))))
        y1 = max(0, min(height - 1, int(round(y1 * height))))
        x2 = max(x1 + 1, min(width - 1, int(round(x2 * width))))
        y2 = max(y1 + 1, min(height - 1, int(round(y2 * height))))

        # Yellow means the box touches a frame edge: the number is a lower
        # bound, not a complete object. Green means a normal visible box.
        colour = (0, 200, 255) if obj.get("clipped") else (0, 220, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 10, 10), 4)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        label = str(obj.get("label") or obj.get("name") or "object")
        # cv2 has no CJK font; keep this ASCII and keep prose in info().
        cv2.putText(frame, label, (x1 + 3, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 4,
                    cv2.LINE_AA)
        cv2.putText(frame, label, (x1 + 3, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                    cv2.LINE_AA)

    quality = max(40, min(95, int(quality)))
    ok, encoded = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return encoded.tobytes()
