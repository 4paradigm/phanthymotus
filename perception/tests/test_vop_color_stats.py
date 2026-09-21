"""vop colour statistics: per-box and per-frame RGB/HSV means, variances, labels."""
from __future__ import annotations

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (installs ROS stubs, sets sys.path)

cv2 = pytest.importorskip("cv2")

from plugins.vop import (color_name, color_stats, dominant_brightness,  # noqa: E402
                         dominant_hue, dominant_saturation)


def _solid(bgr, h=40, w=60):
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def test_frame_stats_have_twelve_numbers_and_labels():
    stats = color_stats(_solid((0, 0, 255)))          # pure red in BGR
    assert stats["rgb_mean"] == [255.0, 0.0, 0.0]
    assert stats["rgb_var"] == [0.0, 0.0, 0.0]
    assert len(stats["hsv_mean"]) == 3 and len(stats["hsv_var"]) == 3
    assert stats["dominant_hue"] == "red"
    assert stats["dominant_saturation"] == "vivid"
    assert stats["dominant_brightness"] == "white"
    assert stats["color_name"] == "bright red"


def test_box_stats_crop_only_the_box():
    frame = _solid((0, 0, 0), 100, 100)               # black frame
    frame[20:40, 30:60] = (255, 0, 0)                 # a blue patch
    inside = color_stats(frame, (30, 20, 60, 40))
    assert inside["dominant_hue"] == "blue"
    assert inside["rgb_mean"] == [0.0, 0.0, 255.0]
    whole = color_stats(frame)
    assert whole["dominant_brightness"] == "black"
    assert whole["rgb_var"][2] > 0                    # blue channel varies across the frame


def test_box_is_clamped_to_frame_and_empty_box_returns_nothing():
    frame = _solid((0, 255, 0), 50, 50)               # green
    assert color_stats(frame, (-10, -10, 500, 500))["dominant_hue"] == "green"
    assert color_stats(frame, (10, 10, 10, 10)) == {}


def test_brightness_separates_lit_from_dark():
    dark = color_stats(_solid((15, 15, 15)))
    lit = color_stats(_solid((220, 220, 220)))
    assert dark["hsv_mean"][2] < 60 and dark["color_name"] == "black"
    assert lit["hsv_mean"][2] > 200 and lit["color_name"] == "white"


@pytest.mark.parametrize("h,name", [
    (3, "red"), (12, "orange"), (24, "yellow"), (36, "lime"), (55, "green"),
    (78, "teal"), (90, "cyan"), (104, "azure"), (118, "blue"), (134, "violet"),
    (148, "magenta"), (162, "pink"), (176, "red")])
def test_twelve_hue_bins(h, name):
    assert dominant_hue(h, 200, 200) == name


@pytest.mark.parametrize("v,name", [
    (10, "black"), (50, "dark"), (90, "dim"), (130, "medium"), (180, "bright"), (230, "white")])
def test_six_brightness_levels(v, name):
    assert dominant_brightness(v) == name


@pytest.mark.parametrize("s,name", [(10, "gray"), (50, "muted"), (150, "vivid")])
def test_three_saturation_levels(s, name):
    assert dominant_saturation(s) == name


def test_low_saturation_is_neutral_and_named_by_brightness():
    assert dominant_hue(60, 10, 120) == "neutral"
    assert color_name(60, 10, 20) == "black"
    assert color_name(60, 10, 60) == "dark gray"
    assert color_name(60, 10, 120) == "gray"
    assert color_name(60, 10, 180) == "light gray"
    assert color_name(60, 10, 230) == "white"


def test_color_name_prefixes_brightness_for_hues():
    assert color_name(118, 200, 50) == "dark blue"
    assert color_name(118, 200, 130) == "blue"
    assert color_name(118, 200, 210) == "bright blue"
