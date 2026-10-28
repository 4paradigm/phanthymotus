"""The overlay renderer: the JPEG that makes the numbers visible.

cv2 is in the perception image; on a laptop without it these tests skip rather
than making the local suite require the whole vision stack.
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
    out = view.render(_jpeg(), [{
        "name": "cup", "box": [0.2, 0.3, 0.5, 0.7],
        "label": "cup 8.3x7.8cm @1.51m", "clipped": False,
    }])
    frame = _decode(out)
    assert frame.shape[:2] == (240, 320)
    if getattr(cv2, "__motus_fake__", False):
        # The host stub stores draw calls but does not encode pixels; assert the
        # label was actually drawn instead of comparing a synthetic frame.
        assert any("cup 8.3x7.8cm" in text for text, _ in cv2.__text_calls__)
    else:
        green = frame[:, :, 1].astype(int) - frame[:, :, 0].astype(int)
        assert (green > 40).sum() > 10


def test_overlay_with_no_objects_is_still_a_valid_jpeg():
    assert _decode(view.render(_jpeg(), [])).shape[:2] == (240, 320)
