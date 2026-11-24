"""Render one hand's 2-D skeleton over the original camera frame.

This is the card-side copy of ``gesture_hand/overlay.py``. ActuCore's image
does not contain the workspace sibling directory, so the card carries its own
copy rather than importing across project roots.

The drawing contract is deliberately narrow, matching the hand-vision card:

* the original BGR frame is copied and kept as the background — no crops, no
  letterbox, no dimming;
* 21 filled green dots mark the joints;
* adjacent joints are joined by thin green lines for the finger bones;
* anti-aliasing keeps the strokes smooth while following a moving hand;
* only one hand is drawn, and only points inside the frame are rendered —
  off-screen joints are not clipped to the edge, because that would invent a
  landmark the model never saw.

Landmarks are normalized image coordinates (x right 0..1, y down 0..1). The
third coordinate is ignored: neither MediaPipe's relative z nor the RTMPose
per-joint confidence has been calibrated to a physical distance, so neither
may influence the drawing.
"""

from __future__ import annotations

import math

# Index pairs into the 21-point hand. Palm edges are included so a fist still
# reads as a hand rather than five loose stalks.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),              # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),              # index
    (5, 9), (9, 10), (10, 11), (11, 12),         # middle
    (9, 13), (13, 14), (14, 15), (15, 16),       # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
)

GREEN = (0, 255, 0)   # BGR — pure green on screen


def image_points(landmarks, width: int, height: int) -> list:
    """Map 21 normalized (x, y) points to integer image pixels.

    None, non-finite and out-of-frame points stay ``None``. Keeping them
    out of the drawing is safer than snapping them to an edge: a fingertip
    genuinely outside the frame must not be reported as a joint on the border.
    """
    if width <= 0 or height <= 0 or len(landmarks) != 21:
        raise ValueError("expected a positive frame size and 21 landmarks")
    points = []
    for point in landmarks:
        if point is None or len(point) < 2:
            points.append(None)
            continue
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < 1 and 0 <= y < 1):
            points.append(None)
            continue
        points.append((round(x * (width - 1)), round(y * (height - 1))))
    return points


def draw_hand(frame, landmarks, *, line_width: int = 1, point_radius: int = 3):
    """Return a copy of the BGR frame with one green, anti-aliased hand."""
    import cv2

    if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError("expected a BGR camera frame")
    if line_width < 1 or point_radius < 1:
        raise ValueError("line width and point radius must be positive")
    height, width = frame.shape[:2]
    points = image_points(landmarks, width, height)
    output = frame.copy()
    for start, end in HAND_CONNECTIONS:
        if points[start] is not None and points[end] is not None:
            cv2.line(output, points[start], points[end], GREEN,
                     line_width, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(output, point, point_radius, GREEN, -1, cv2.LINE_AA)
    return output


def encode_jpeg(frame, quality: int = 90) -> bytes:
    """Encode a BGR frame for the ``image/jpeg`` view topic."""
    import cv2

    if not 1 <= int(quality) <= 100:
        raise ValueError("jpeg quality must be in 1..100")
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),
                                              int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bytes(buffer)
