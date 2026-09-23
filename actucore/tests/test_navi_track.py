"""The tracker, behaviour by behaviour.

Two groups of tests, and they answer different questions.

The **lifecycle** ones are about a number that used to be wrong: navi took one
frame to start chasing something and ten to give up. That asymmetry points the
wrong way — starting is the direction that moves a robot — and it is what made a
single false-positive frame walk a humanoid forward and then choose which way
the subsequent search swept.

The **occlusion** ones are about the thing this module exists for: a person
walks in front of the target for half a second and the robot should not notice.
The one to read first is
`test_a_hidden_target_moves_across_the_frame_as_the_robot_turns` — it is the
whole reason the state is metric and body-frame rather than an image box.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navi import policy as P  # noqa: E402
from plugins.navi import track as T  # noqa: E402

STILL = (0.0, 0.0, 0.0)


def _cfg(**over):
    return P.Config(**over)


def _obj(name="chair", bearing=0.0, confidence=0.9, colour=""):
    """`bearing` here is the **normalised** offset vop publishes, not an angle."""
    out = {"name": name, "position": [bearing, 0.0], "confidence": confidence}
    if colour:
        out["color"] = colour
    return out


def _detections(*objects):
    return {"objects": list(objects)}


def _depth(range_m=3.0):
    return {"map": np.full((480, 640), range_m * 1000.0, dtype=np.uint16)
            .astype(float) / 1000.0,
            "bands": {"left": 5.0, "center": 5.0, "right": 5.0}}


def _feed(tracker, detections, *, config, depth=None, ticks=1, ego=STILL,
          dt=0.1, target="chair", measured=True):
    out = None
    for _ in range(ticks):
        out = tracker.step(detections=detections, depth=depth or _depth(),
                           target=target, config=config, ego=ego, dt=dt,
                           ego_measured=measured)
    return out


# ── lifecycle: the asymmetry that used to point the wrong way ────────────────

def test_one_frame_does_not_move_a_robot():
    """The bug this module was written for.

    A single detection above the confidence bar used to be enough to start
    driving, and the same frame set the direction the later search swept in. So
    one false positive produced "turn left a bit, walk forward a bit, then scan
    counter-clockwise" — which is exactly what it looked like on the robot, and
    none of it was a coincidence.
    """
    config = _cfg()
    tracker = T.Tracker()
    assert _feed(tracker, _detections(_obj()), config=config) is None
    assert tracker.track is not None, "it is remembered..."
    assert tracker.track.state == T.TENTATIVE, "...but not acted on"


def test_three_of_five_confirms():
    config = _cfg()
    tracker = T.Tracker()
    for _ in range(config.confirm_hits - 1):
        assert _feed(tracker, _detections(_obj()), config=config) is None
    assert _feed(tracker, _detections(_obj()), config=config) is not None
    assert tracker.track.state == T.CONFIRMED


def test_a_ghost_that_appears_once_is_dropped():
    """And never drives anything on the way out."""
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj()), config=config)
    for _ in range(config.confirm_window):
        assert _feed(tracker, _detections(), config=config) is None
        if tracker.track is None:
            break
    assert tracker.track is None
    # Read at the moment of the drop: later ticks overwrite it with the ordinary
    # "nothing in view", which says nothing about why the candidate went.
    assert "没能在窗口内确认" in tracker.last_reason


def test_a_flicker_does_not_stop_confirmation():
    """Requiring three *consecutive* frames would never confirm anything: a
    chair that is really there shows up in seven frames out of ten."""
    config = _cfg()
    tracker = T.Tracker()
    for detections in (_detections(_obj()), _detections(),
                       _detections(_obj()), _detections(_obj())):
        drivable = _feed(tracker, detections, config=config)
    assert drivable is not None


# ── occlusion ────────────────────────────────────────────────────────────────

def test_a_confirmed_track_survives_a_short_occlusion():
    """Somebody walks past. The robot should not notice."""
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj()), config=config, ticks=config.confirm_hits)

    for _ in range(5):                      # half a second of nothing
        track = _feed(tracker, _detections(), config=config)
        assert track is not None, "the robot stopped for a passer-by"
        assert track.state == T.COASTING
    assert tracker.track.range_m == pytest.approx(3.0, abs=0.3)


def test_a_hidden_target_moves_across_the_frame_as_the_robot_turns():
    """The reason the state is metric and body-frame rather than an image box.

    The robot turns left at its minimum 1.0 rad/s for half a second with the
    target hidden. The target has not moved, so it must now be half a radian to
    the robot's **right** — most of the way across a 63° frame.

    An image-space tracker coasting on constant velocity would report it still
    dead ahead, because it never saw the rotation. We do not have to see it: it
    is the twist the chassis just reported.
    """
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj(bearing=0.0)), config=config,
          ticks=config.confirm_hits)
    assert tracker.track.bearing_rad == pytest.approx(0.0, abs=0.02)

    _feed(tracker, _detections(), config=config, ticks=5, ego=(0.0, 0.0, 1.0))

    assert tracker.track.bearing_rad == pytest.approx(0.5, abs=0.1)
    assert tracker.track.range_m == pytest.approx(3.0, abs=0.3), \
        "turning in place must not change how far away the target is"


def test_walking_forward_while_hidden_closes_the_distance():
    """The translation half of the same compensation."""
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj()), config=config, ticks=config.confirm_hits)
    _feed(tracker, _detections(), config=config, ticks=5, ego=(0.4, 0.0, 0.0))
    assert tracker.track.range_m == pytest.approx(3.0 - 0.2, abs=0.2)


def test_the_coast_expires_rather_than_running_for_ever():
    """Past a point the extrapolation is a guess. OC-SORT measured how quickly
    that guess rots; the honest answer is to give up and let the card search."""
    config = _cfg(max_coast_s=0.5)
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj()), config=config, ticks=config.confirm_hits)
    for _ in range(10):
        track = _feed(tracker, _detections(), config=config)
        if tracker.track is None:
            break
    assert track is None and tracker.track is None
    assert "外推已不可信" in tracker.last_reason


def test_re_acquisition_rebuilds_velocity_from_the_two_observations():
    """OC-SORT's observation-centric re-update.

    The velocity carried through a coast is the one quantity that has been
    integrating its own error the whole time. On re-acquisition it is rebuilt
    from the last real observation and the new one, which is why a target that
    reappears somewhere else is believed to have walked there rather than to
    have teleported while standing still.
    """
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj(bearing=0.0)), config=config,
          ticks=config.confirm_hits)
    before = tracker.track.mean[2:].copy()

    _feed(tracker, _detections(), config=config, ticks=4)
    _feed(tracker, _detections(_obj(bearing=0.3)), config=config)

    after = tracker.track.mean[2:]
    assert not np.allclose(before, after), "the coast's velocity was kept"
    assert after[1] < 0, "it moved to the right, so its y velocity is negative"
    assert tracker.track.state == T.CONFIRMED


# ── association: the gate is what makes low confidence usable ────────────────

def test_a_dim_detection_sustains_a_track_but_cannot_start_one():
    """ByteTrack's second association. Partial occlusion is exactly what makes a
    detector lose confidence, so discarding that band discards the frames the
    occlusion produced."""
    config = _cfg()
    dim = _detections(_obj(confidence=config.sustain_confidence + 0.05))

    starting = T.Tracker()
    _feed(starting, dim, config=config, ticks=5)
    assert starting.track is None, "a dim detection must not start a chase"

    running = T.Tracker()
    _feed(running, _detections(_obj()), config=config, ticks=config.confirm_hits)
    track = _feed(running, dim, config=config)
    assert track is not None and track.state == T.CONFIRMED


def test_a_dim_detection_outside_the_gate_is_treated_as_background():
    """What separates "the target, seen through something" from noise is not the
    score — it is whether it agrees with where the target was predicted to be."""
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj(bearing=0.0)), config=config,
          ticks=config.confirm_hits)
    elsewhere = _detections(_obj(bearing=0.95,
                                 confidence=config.sustain_confidence + 0.05))
    track = _feed(tracker, elsewhere, config=config)
    assert track.state == T.COASTING, "it counted as a miss, not as the target"
    assert tracker.track.bearing_rad == pytest.approx(0.0, abs=0.05)


def test_a_second_object_of_the_same_name_does_not_steal_the_track():
    """`navigate_to("chair")` with two chairs in frame used to take whichever
    won on confidence that frame, which can change frame to frame."""
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj(bearing=-0.2)), config=config,
          ticks=config.confirm_hits)
    both = _detections(_obj(bearing=-0.2, confidence=0.6),
                       _obj(bearing=0.9, confidence=0.95))
    _feed(tracker, both, config=config, ticks=3)
    assert tracker.track.bearing_rad < 0, "it followed the louder one"


def test_hue_is_a_tie_break_not_a_gate():
    """Colour flickers with the light. Rejecting on it drops a track for a cloud
    passing the window; ignoring it loses the only thing separating two chairs."""
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj(colour="dim muted azure")), config=config,
          ticks=config.confirm_hits)

    # Same place, different hue: still the target.
    track = _feed(tracker, _detections(_obj(colour="bright vivid red")),
                  config=config)
    assert track is not None and track.state == T.CONFIRMED

    # Same hue wins over a different one at equal distance from the prediction.
    tracker.track.hue = "azure"
    cost_same = tracker._gate_cost(
        {"point": tracker.track.position.copy(), "range_m": 3.0, "hue": "azure"},
        config)
    cost_other = tracker._gate_cost(
        {"point": tracker.track.position.copy(), "range_m": 3.0, "hue": "red"},
        config)
    assert cost_same < cost_other


# ── depth it never had ───────────────────────────────────────────────────────

def test_a_target_with_no_depth_reading_is_tracked_but_reports_no_range():
    """The summary-only path has no reading for a band with nothing in it. The
    track still exists — bearing is most of what the control law needs — but the
    placeholder range must never reach the arrival test."""
    config = _cfg()
    tracker = T.Tracker()
    bandless = {"map": None, "bands": {"left": 5.0, "center": 5.0}}
    _feed(tracker, _detections(_obj(bearing=0.5)), config=config, depth=bandless,
          ticks=config.confirm_hits)
    assert tracker.track is not None
    assert tracker.track.range_known is False
    assert tracker.describe()["range_m"] is None


def test_a_real_reading_promotes_a_track_that_started_without_one():
    config = _cfg()
    tracker = T.Tracker()
    bandless = {"map": None, "bands": {"left": 5.0, "center": 5.0}}
    _feed(tracker, _detections(_obj(bearing=0.5)), config=config, depth=bandless)
    _feed(tracker, _detections(_obj(bearing=0.5)), config=config, ticks=3)
    assert tracker.track.range_known is True


# ── believing our own commands less than our own odometry ────────────────────

def test_an_unmeasured_ego_twist_makes_the_filter_lean_on_what_it_sees():
    """`ego` is the command we sent when no odometry is wired, and a command is
    not a measurement — `dry_run`, a refused command, or the deadband all make
    the two disagree. Predicting a rotation that never happened would otherwise
    leave the estimate fighting every observation."""
    config = _cfg()
    measured, guessed = T.Tracker(), T.Tracker()
    for tracker, flag in ((measured, True), (guessed, False)):
        _feed(tracker, _detections(_obj()), config=config,
              ticks=config.confirm_hits, measured=flag)
        _feed(tracker, _detections(), config=config, ticks=3,
              ego=(0.0, 0.0, 1.0), measured=flag)
    assert guessed.track.position_std_m > measured.track.position_std_m


# ── the physical bound that stops a detector glitch becoming motion ──────────

def test_a_detection_that_jumps_is_not_read_as_a_sprinting_target():
    """A box snapping to a different part of the same object is not an object
    that accelerated. Integrating it would make the estimate overshoot past the
    target and command a turn the **wrong way** — and on a chassis whose slowest
    turn is 1 rad/s that is a visible twitch, not a rounding error."""
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj(bearing=0.0)), config=config,
          ticks=config.confirm_hits)
    _feed(tracker, _detections(_obj(bearing=0.9)), config=config, ticks=2)
    speed = math.hypot(*tracker.track.mean[2:])
    assert speed <= config.max_target_speed + 1e-6


# ── conventions, pinned ──────────────────────────────────────────────────────

def test_bearing_is_right_positive_and_y_is_left():
    """Mirror this and the robot chases reflections. It is pinned here because
    three files share the convention and none of them can check the others."""
    point = T.point_of(2.0, 0.5)
    assert point[0] > 0 and point[1] < 0, "a bearing to the right is -y"
    range_m, bearing = T.polar_of(point)
    assert range_m == pytest.approx(2.0) and bearing == pytest.approx(0.5)


# ── one payload is one piece of evidence ─────────────────────────────────────

def _stamped(stamp, *objects):
    return {"timestamp": stamp, "objects": list(objects)}


def test_re_reading_one_payload_does_not_confirm_a_track():
    """Measured on r1_sz: the policy ticks at 10 Hz and vop publishes at 3.6 Hz,
    so the same payload is read on about three consecutive ticks.

    Counting a hit each time turns "three of the last five frames" into "one
    frame, read three times" — which confirms a track off a single false
    positive, the exact failure this lifecycle exists to prevent.
    """
    config = _cfg()
    tracker = T.Tracker()
    payload = _stamped(100.0, _obj())
    for _ in range(6):
        assert _feed(tracker, payload, config=config) is None, \
            "one detection frame confirmed a track by being read repeatedly"
    assert tracker.track.state == T.TENTATIVE
    assert tracker.track.hits == 1


def test_three_distinct_payloads_confirm():
    config = _cfg()
    tracker = T.Tracker()
    for index in range(config.confirm_hits):
        drivable = _feed(tracker, _stamped(100.0 + index, _obj()), config=config)
    assert drivable is not None


def test_a_repeated_payload_still_advances_the_prediction():
    """Only the evidence is skipped. Our own motion continues whether or not a
    new picture arrived, so the estimate must keep moving with it."""
    config = _cfg()
    tracker = T.Tracker()
    for index in range(config.confirm_hits):
        _feed(tracker, _stamped(100.0 + index, _obj(bearing=0.0)), config=config)
    before = tracker.track.bearing_rad

    repeat = _stamped(200.0, _obj(bearing=0.0))
    _feed(tracker, repeat, config=config)              # new stamp: counted
    _feed(tracker, repeat, config=config, ticks=4, ego=(0.0, 0.0, 1.0))
    assert tracker.track.bearing_rad > before + 0.2, \
        "the robot turned while re-reading one frame and the estimate did not move"


def test_a_payload_with_no_stamp_is_counted_every_time():
    """A detector that does not stamp its output cannot be de-duplicated, and
    the safe fallback is to treat each read as evidence — over-counting is the
    behaviour that existed before, not a new hazard."""
    config = _cfg()
    tracker = T.Tracker()
    for _ in range(config.confirm_hits):
        drivable = _feed(tracker, _detections(_obj()), config=config)
    assert drivable is not None
