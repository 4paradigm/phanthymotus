"""vop colour statistics: per-box and per-frame RGB/HSV means, variances, hue names."""
from __future__ import annotations

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (installs ROS stubs, sets sys.path)

cv2 = pytest.importorskip("cv2")

from plugins.vop import color_stats, dominant_hue  # noqa: E402


def _solid(bgr, h=40, w=60):
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def test_frame_stats_have_twelve_numbers_and_a_hue():
    stats = color_stats(_solid((0, 0, 255)))          # pure red in BGR
    assert stats["rgb_mean"] == [255.0, 0.0, 0.0]
    assert stats["rgb_var"] == [0.0, 0.0, 0.0]
    assert len(stats["hsv_mean"]) == 3 and len(stats["hsv_var"]) == 3
    assert stats["dominant_hue"] == "red"


def test_box_stats_crop_only_the_box():
    frame = _solid((0, 0, 0), 100, 100)               # black frame
    frame[20:40, 30:60] = (255, 0, 0)                 # a blue patch
    inside = color_stats(frame, (30, 20, 60, 40))
    assert inside["dominant_hue"] == "blue"
    assert inside["rgb_mean"] == [0.0, 0.0, 255.0]
    whole = color_stats(frame)
    assert whole["dominant_hue"] == "black"
    assert whole["rgb_var"][2] > 0                    # blue channel varies across the frame


def test_box_is_clamped_to_frame_and_empty_box_returns_nothing():
    frame = _solid((0, 255, 0), 50, 50)               # green
    assert color_stats(frame, (-10, -10, 500, 500))["dominant_hue"] == "green"
    assert color_stats(frame, (10, 10, 10, 10)) == {}


def test_brightness_separates_lit_from_dark():
    dark = color_stats(_solid((15, 15, 15)))
    lit = color_stats(_solid((220, 220, 220)))
    assert dark["hsv_mean"][2] < 60 and dark["dominant_hue"] == "black"
    assert lit["hsv_mean"][2] > 200 and lit["dominant_hue"] == "white"


@pytest.mark.parametrize("h,name", [(0, "red"), (15, "orange"), (28, "yellow"),
                                    (60, "green"), (90, "cyan"), (115, "blue"),
                                    (140, "purple"), (160, "pink"), (175, "red")])
def test_hue_names(h, name):
    assert dominant_hue(h, 200, 200) == name


def test_low_saturation_is_achromatic():
    assert dominant_hue(60, 10, 30) == "black"
    assert dominant_hue(60, 10, 120) == "gray"
    assert dominant_hue(60, 10, 230) == "white"
