"""vop colour statistics: per-box and per-frame RGB/HSV means, variances, labels."""
from __future__ import annotations

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (installs ROS stubs, sets sys.path)

cv2 = pytest.importorskip("cv2")

from plugins.vop import (_BRIGHTNESS_NAMES, _HUE_NAMES,  # noqa: E402
                         _SATURATION_NAMES, color_name, color_stats,
                         color_triple, dominant_brightness, dominant_hue,
                         dominant_saturation)


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


# ── the streamed triple ───────────────────────────────────────────────────────
#
# The stream carries `color_triple`, not `color_name`. These tests exist to pin
# the one fact that decision rests on — the triple loses nothing `color_name`
# keeps, and keeps a great deal `color_name` loses — because it is not obvious
# from reading either function, and getting it backwards costs 73% of the
# colour information for no saving.


def _bucket_probe_values(table, ceiling):
    """One value inside every bucket of a table, plus the boundaries themselves.

    Sweeping a coarse grid would very likely hit every bucket, but "very likely"
    is not what a test should rest on when the tables can be edited. Derived
    from the tables so a new colour name is covered the day it is added.
    """
    values = {0, ceiling - 1}
    for upper, _name in table:
        for probe in (upper - 1, upper, upper + 1):
            if 0 <= probe < ceiling:
                values.add(probe)
    return sorted(values)


_HUES = _bucket_probe_values(_HUE_NAMES, 180)
_SATS = _bucket_probe_values(_SATURATION_NAMES, 256)
_VALS = _bucket_probe_values(_BRIGHTNESS_NAMES, 256)


def _all_triples():
    for h in _HUES:
        for s in _SATS:
            for v in _VALS:
                yield h, s, v


def test_color_name_is_recoverable_from_the_triple():
    """The whole reason the stream may drop `color_name` and keep the triple.

    If this ever fails, the stream has started losing something and
    `publish_color: name` is no longer a lossless trade.
    """
    def rebuild(brightness, _saturation, hue):
        if hue == "neutral":
            return {"black": "black", "dark": "dark gray", "dim": "gray",
                    "medium": "gray", "bright": "light gray",
                    "white": "white"}[brightness]
        if brightness in ("black", "dark"):
            return f"dark {hue}"
        if brightness in ("bright", "white"):
            return f"bright {hue}"
        return hue

    for h, s, v in _all_triples():
        assert rebuild(*color_triple(h, s, v).split(" ")) == color_name(h, s, v)


def test_the_triple_draws_distinctions_color_name_throws_away():
    """The other direction — and the size of the gap.

    `color_name` never consults saturation and flattens six brightness levels
    into three prefixes, so it is strictly coarser. The exact counts are
    asserted as an inequality with a healthy margin rather than as 150 and 41:
    adding a hue should not break this test, but collapsing the triple to
    `color_name`'s resolution should.
    """
    triples = {color_triple(h, s, v) for h, s, v in _all_triples()}
    names = {color_name(h, s, v) for h, s, v in _all_triples()}
    assert len(triples) > 2 * len(names)


def test_the_triple_is_three_space_separated_tokens():
    """Consumers split on spaces, so no label may ever contain one."""
    for h, s, v in _all_triples():
        parts = color_triple(h, s, v).split(" ")
        assert len(parts) == 3
        assert parts == [dominant_brightness(v), dominant_saturation(s),
                         dominant_hue(h, s, v)]


def test_neutral_and_gray_always_come_together():
    """Both are the same `s < 25` test, so the pair is redundant by
    construction — emitted anyway to keep the token count fixed at three.
    Pinned because a future change to either threshold alone would silently
    produce "vivid neutral", which reads like a bug in the detector."""
    for h, s, v in _all_triples():
        _brightness, saturation, hue = color_triple(h, s, v).split(" ")
        assert (hue == "neutral") == (saturation == "gray")
