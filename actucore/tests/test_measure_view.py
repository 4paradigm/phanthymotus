"""The overlay renderer: the JPEG that makes the numbers visible.

Pillow is in the actucore image and in the local test venv; if it is absent the
test skips rather than making the local suite require it.
"""
from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

Image = pytest.importorskip("PIL.Image")

from plugins.measure import view  # noqa: E402


def _jpeg(width=320, height=240):
    image = Image.new("RGB", (width, height), (180, 180, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _decode(jpeg):
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    return image


def test_overlay_draws_a_box_and_label_without_changing_the_frame_size():
    source = _jpeg()
    out = view.render(source, [{
        "name": "cup", "box": [0.2, 0.3, 0.5, 0.7], "label": "cup 8.3x7.8cm @1.51m",
        "clipped": False,
    }])
    image = _decode(out)
    assert image.size == (320, 240)
    # The box is green; a blank grey frame must now contain a green pixel.
    pixels = list(image.getdata())
    assert any(g > r + 30 and g > b + 30 for r, g, b in pixels)


def test_overlay_with_no_objects_is_still_a_valid_jpeg():
    assert _decode(view.render(_jpeg(), [])).size == (320, 240)
