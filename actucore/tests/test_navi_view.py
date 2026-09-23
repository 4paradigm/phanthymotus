"""The debug overlay: what the card saw and what it decided, on one frame.

Every bug found in this card so far took the same shape — **the numbers all
looked reasonable and the robot did something else.** A field of view that made
the corridor 1.86 m wide; angular bands that were reading the lens barrel rather
than the room; a sidestep amplified ninefold by a deadband. Each was invisible in
`info()` and obvious the moment somebody watched the robot.

So the overlay is tested for the properties that make it worth trusting: that
"no reading" is distinguishable from any depth, that the corridor it draws is the
one the decision used, and — most of all — that it can never take the robot down.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "plugins", ".."))

from plugins.navi import policy as P  # noqa: E402
from plugins.navi import view as V  # noqa: E402

# The geometry below needs no image library and must always run: metres-to-pixels
# through the field of view is the computation that has produced every serious
# bug in this card, and a skipped test of it would be the same silent failure
# one level up. Only the pixel-pushing needs cv2 (present in the actucore image,
# absent on a laptop).
try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

needs_cv2 = pytest.mark.skipif(not HAS_CV2, reason="cv2 不在（笔记本上没有）")


# ── metres to pixels: always tested, no image library involved ──────────────

def test_the_corridor_columns_come_from_the_same_mapping_the_decision_used():
    """A picture drawn from a different mapping than the decision would be worse
    than no picture: it would corroborate whatever the robot did."""
    config = P.Config()
    keep = config.half_width_m + config.clearance_margin_m
    left, right = V.corridor_edges(1.0, config, 640)
    # The edge sits at atan(keep / distance) off the nose, mapped through the
    # half field of view — the same two steps `depth.corridor` takes.
    import math
    expected = math.atan2(keep, 1.0) / config.half_fov_rad
    assert (right - 320) / 320 == pytest.approx(expected, abs=0.01)
    assert (320 - left) == pytest.approx(right - 320, abs=1)


def test_the_corridor_narrows_in_the_image_as_the_distance_grows():
    """Because it is metric. A fixed pair of lines would misrepresent exactly
    the thing this image exists to show."""
    config = P.Config()
    near = V.corridor_edges(0.8, config, 640)
    far = V.corridor_edges(4.0, config, 640)
    assert (far[1] - far[0]) < (near[1] - near[0])


def test_a_wider_lens_puts_the_same_corridor_in_fewer_columns():
    """The bug that started all of this, seen from the drawing side: understate
    the field of view and the same metric width maps to a wider slice."""
    narrow = V.corridor_edges(1.0, P.Config(half_fov_rad=0.55), 640)
    wide = V.corridor_edges(1.0, P.Config(half_fov_rad=0.888), 640)
    assert (wide[1] - wide[0]) < (narrow[1] - narrow[0])


def test_no_usable_reading_draws_no_corridor():
    config = P.Config()
    for bad in (None, 0.0, float("nan"), -1.0):
        assert V.corridor_edges(bad, config, 640) is None


def _decision(vx=0.0, vy=0.0, wz=0.0, status="approaching", distance=None):
    return P.Decision([vx, vy, 0.0, 0.0, 0.0, wz], status, "reason",
                      distance_m=distance, bearing=0.0)


def _depth(value=2.0, holes=None):
    depth = np.full((480, 640), value, dtype=np.float32)
    if holes:
        depth[holes] = np.nan
    return depth


def _render(**over):
    args = dict(depth_m=_depth(), decision=_decision(), track=None, box=None,
                clearance=1.5, coverage=1.0, config=P.Config())
    args.update(over)
    return V.render(**args)


# ── the frame ────────────────────────────────────────────────────────────────

@needs_cv2
def test_everything_drawn_on_top_is_legible_against_a_washed_out_far_field():
    """The defect that came with adopting the dashboard's ramp.

    That ramp washes the far field out to near-white, which is right for depth —
    the eye should pass over distances nobody acts on — and fatal for white text
    and white lines drawn on top of it. The first version was legible only
    because viridis happens to be dark; against the real ramp the distances
    vanished into the background.

    Asserted as "there are casing-dark pixels next to the bright ones", which is
    what survives any background including the flat grey of "no reading".
    """
    frame = _render(depth_m=_depth(5.0),          # far: the washed-out extreme
                    decision=_decision(vx=0.5, status="approaching",
                                       distance=1.2),
                    box=(0.3, 0.3, 0.6, 0.8),
                    track={"state": "confirmed", "range_m": 1.23})
    dark = (frame.astype(int).sum(axis=2) < 120)
    assert dark.sum() > 400, "前景没有深色描边，白底上读不出来"

    plain = _render(depth_m=_depth(5.0), clearance=None)
    assert (plain.astype(int).sum(axis=2) < 120).sum() < dark.sum()


@needs_cv2
def test_it_produces_a_frame_of_the_declared_size():
    frame = _render()
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8


@needs_cv2
def test_no_reading_is_not_a_colour_on_the_depth_ramp():
    """The distinction the whole lens-barrel mask exists for. A confident wrong
    reading and a missing one look identical in any colour ramp, so "no reading"
    has to leave the ramp entirely — flat grey, which is also what the mask
    working looks like from the outside."""
    frame = _render(depth_m=_depth(holes=(slice(0, 40), slice(0, 40))))
    corner = frame[10, 10]
    assert tuple(int(v) for v in corner) == V._NO_READING
    assert not np.array_equal(corner, frame[240, 320]), "和有读数的地方一样了"


def test_near_is_loud_and_far_washes_out():
    """Red belongs to what you are about to hit. The ramp's own rule, and the
    reason it is not a rainbow — inherited with the table rather than invented
    here."""
    lut = V.depth_lut()
    near, far = lut[0].astype(int), lut[-1].astype(int)
    assert near[2] > near[0], "近处应当偏红（BGR）"
    assert far.min() > 150, "远处应当洗白，让眼睛掠过去"


def test_lightness_rises_with_distance_so_it_survives_greyscale():
    """The property that makes it safe for all three common colour-vision
    deficiencies: the reading never depends on telling two hues apart."""
    lut = V.depth_lut().astype(int)
    luma = lut @ [0.114, 0.587, 0.299]          # BGR weights
    drops = np.diff(luma) < -6
    assert not drops.any(), f"亮度曲线有 {drops.sum()} 处明显下陷"


def test_the_table_still_matches_the_dashboards():
    """Two copies, one in Python and one in JavaScript, because a container and
    a browser cannot share a module. Pinned so a change to one shows up here
    rather than as two panels that quietly disagree."""
    import pathlib
    import re

    js = (pathlib.Path(__file__).resolve().parents[2] / "agent-core" / "web"
          / "js" / "renderers" / "camera.js").read_text()
    stops = re.findall(r"\[(\d+\.\d+), 0x([0-9A-F]{2}), 0x([0-9A-F]{2}), "
                       r"0x([0-9A-F]{2})\]", js)
    assert stops, "没能从 camera.js 里解析出色带"
    parsed = [(float(d), int(r, 16), int(g, 16), int(b, 16))
              for d, r, g, b in stops]
    assert parsed == V._DEPTH_STOPS
    assert f"DEPTH_NEAR_SHARE = {V._DEPTH_NEAR_SHARE}" in js
    assert f"DEPTH_NEAR_M = {V._DEPTH_NEAR_M}" in js


@needs_cv2
def test_a_missing_depth_map_still_draws():
    """The overlay's job is to explain a decision, and "blind" is a decision
    worth explaining. Refusing to draw exactly when the input is missing would
    hide the case most in need of a picture."""
    assert _render(depth_m=None).shape == (480, 640, 3)


# ── the corridor drawn is the corridor that decided ─────────────────────────

@needs_cv2
def test_the_corridor_narrows_in_the_image_as_the_clearance_grows():
    """The single fact that took longest to become visible: the corridor is
    metric, so the same half-width covers a different slice of the picture at
    every range. A fixed pair of lines would misrepresent exactly the thing this
    image exists to show."""
    def edges(clearance):
        frame = _render(clearance=clearance)
        lit = np.where((frame[400] == V._CORRIDOR).all(axis=1))[0]
        return lit.min(), lit.max()

    near_l, near_r = edges(0.8)
    far_l, far_r = edges(4.0)
    assert (far_r - far_l) < (near_r - near_l)


@needs_cv2
def test_a_blocked_corridor_is_drawn_in_the_warning_colour():
    config = P.Config()
    frame = _render(clearance=config.obstacle_stop_m - 0.1, config=config)
    assert (frame == np.array(V._WARN)).all(axis=2).any()


@needs_cv2
def test_no_corridor_reading_draws_no_corridor():
    frame = _render(clearance=None)
    assert not (frame == np.array(V._CORRIDOR)).all(axis=2).any()


# ── the target ───────────────────────────────────────────────────────────────

@needs_cv2
def test_the_target_box_lands_where_the_detector_put_it():
    frame = _render(box=(0.25, 0.25, 0.75, 0.75),
                    track={"state": "confirmed", "range_m": 1.23})
    hits = np.argwhere((frame == np.array(V._TARGET)).all(axis=2))
    assert hits.size, "没画出目标框"
    top, left = hits[:, 0].min(), hits[:, 1].min()
    assert 100 < left < 180 and top < 130


@needs_cv2
def test_a_coasting_track_without_a_box_says_it_is_a_prediction():
    """A robot walking towards a prediction looks exactly like one walking
    towards something it can see. That is the whole reason this line exists."""
    frame = _render(box=None, track={"state": "coasting", "range_m": 1.0})
    plain = _render(box=None, track=None)
    assert not np.array_equal(frame, plain)


# ── the command ──────────────────────────────────────────────────────────────

@needs_cv2
def test_each_axis_only_draws_when_it_is_commanding_something():
    still = _render(decision=_decision())
    moving = _render(decision=_decision(vx=0.5))
    assert not np.array_equal(still, moving)


@needs_cv2
def test_the_lateral_arrow_points_left_for_a_positive_command():
    """Body frame y is left-positive. Drawing it the other way would make the
    picture agree with an intuition the code does not share — and this axis has
    already produced one sign bug that looked right in every other signal."""
    left = _render(decision=_decision(vy=0.6))
    right = _render(decision=_decision(vy=-0.6))
    lit = lambda f: np.argwhere((f == np.array(V._ARROW)).all(axis=2))[:, 1]
    assert lit(left).min() < 320 < lit(right).max()


# ── it must never be able to stop the robot ──────────────────────────────────

@needs_cv2
def test_an_encoder_that_produces_bytes_is_all_the_contract_is():
    assert V.encode(_render()).startswith(b"\xff\xd8")


@needs_cv2
def test_a_nonsense_decision_does_not_raise():
    """Rendering runs on the publish tick. Anything it can throw would stall the
    command stream, and a stalled stream is a robot its watchdog stops — a
    picture for a human must not be able to cause a motion fault. The plugin
    wraps this as well; this is the inner layer."""
    for bad in (None, P.Decision(None, "failed", "x"),
                P.Decision([float("nan")] * 6, "avoiding", "x")):
        V.render(depth_m=_depth(), decision=bad, track={"state": "lost"},
                 box=(0.0, 0.0, 1.0, 1.0), clearance=float("nan"),
                 coverage=0.0, config=P.Config())


@needs_cv2
def test_a_tracked_range_that_disagrees_with_the_measurement_is_flagged():
    """The reading that should be impossible, made visible.

    The tracked range is a filtered estimate, ego-motion compensated every tick
    since the last observation — and with no odometry that compensation runs on
    the **commanded** twist. A robot told to walk at 1 m/s that does not quite
    manage it drags its target 0.1 m closer per tick, with nothing but a 4 Hz
    detector pulling back. What that looks like from outside is a tracked range
    *below the nearest thing in the whole picture*, and nothing was reporting it.
    """
    agreeing = _render(box=(0.3, 0.3, 0.6, 0.8), measured=1.20,
                       track={"state": "confirmed", "range_m": 1.23})
    drifted = _render(box=(0.3, 0.3, 0.6, 0.8), measured=2.22,
                      track={"state": "confirmed", "range_m": 0.55})

    warn = lambda f: (f == np.array(V._WARN)).all(axis=2).sum()
    assert warn(drifted) > warn(agreeing), "两个数差一倍也没有任何提示"


@needs_cv2
def test_both_numbers_are_drawn_when_a_measurement_exists():
    plain = _render(box=(0.3, 0.3, 0.6, 0.8),
                    track={"state": "confirmed", "range_m": 1.23})
    both = _render(box=(0.3, 0.3, 0.6, 0.8), measured=1.20,
                   track={"state": "confirmed", "range_m": 1.23})
    assert not np.array_equal(plain, both)
