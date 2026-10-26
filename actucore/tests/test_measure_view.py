"""The overlay renderer: the JPEG that makes the numbers visible.

OpenCV lives in the actucore image, not on every laptop, so these tests skip
when it is absent rather than making the local suite require it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

cv2 = pytest.importorskip("cv2")

from plugins.measure import view  # noqa: E402


def _jpeg(width=320, height=240):
    frame = np.full((height, width, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def _decode(jpeg):
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    return frame


def test_overlay_draws_a_box_and_label_without_changing_the_frame_size():
    source = _jpeg()
    out = view.render(source, [{
        "name": "cup", "box": [0.2, 0.3, 0.5, 0.7], "label": "cup 8.3x7.8cm @1.51m",
        "clipped": False,
    }])
    frame = _decode(out)
    assert frame.shape[:2] == (240, 320)
    # The box is green; a blank grey frame must now contain a green pixel.
    green = frame[:, :, 1].astype(int) - frame[:, :, 0].astype(int)
    assert (green > 40).sum() > 10


def test_overlay_with_no_objects_is_still_a_valid_jpeg():
    out = view.render(_jpeg(), [])
    assert _decode(out).shape[:2] == (240, 320)
