"""The tracker, behaviour by behaviour.

The lifecycle tests cover immediate association and bounded prediction.

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

def test_first_confident_detection_supplies_a_target():
    config = _cfg()
    tracker = T.Tracker()
    track = _feed(tracker, _detections(_obj()), config=config)
    assert track.observed and track.accepted_detection and track.hits == 1


def test_a_single_detection_still_expires_after_the_coast_budget():
    config = _cfg(max_coast_s=.25)
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj()), config=config)
    coast = _feed(tracker, _detections(), config=config)
    assert coast.state == T.COASTING
    assert _feed(tracker, _detections(), config=config, dt=.2) is None


def test_a_flicker_does_not_replace_the_track():
    """A short miss keeps the target generation."""
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


def test_retained_target_can_reassociate_within_the_observation_motion_budget():
    config = _cfg(max_obs_age_ms=500)
    tracker = T.Tracker()
    original = _feed(tracker, _detections(_obj(name="person")), target="person", config=config)

    def feed(detections, dt):
        return tracker.step(detections=detections, depth=_depth(), target="person",
                            config=config, ego=STILL, dt=dt, retain_identity=True)

    assert feed(_detections(), 1.3) is None
    # A two-metre sideways jump cannot consume the whole occlusion as a
    # movement allowance. A smaller displacement from the same person can.
    assert feed(_detections(_obj(name="person", bearing=1.)), .1) is None
    assert feed(_detections(_obj(name="person", bearing=.5)), .1) is original
    assert original.observed


@pytest.mark.parametrize("side", [-1., 1.])
def test_retained_person_does_not_jump_across_the_camera_after_expiry(side):
    config = _cfg(local_controller="dwa")
    tracker = T.Tracker()

    def feed(bearing, dt):
        detections = _detections() if bearing is None else _detections(_obj(name="person", bearing=bearing))
        return tracker.step(detections=detections, depth=_depth(.85), target="person",
                            config=config, ego=STILL, dt=dt, retain_identity=True)

    original = feed(.9 * side, .1)
    assert original.observed
    assert feed(None, 1.3) is None
    assert feed(-.8 * side, .1) is None
    assert feed(.75 * side, .1) is original
    assert original.observed


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
    # Re-acquisition costs the same evidence as creation — see the test below.
    _feed(tracker, _detections(_obj(bearing=0.3)), config=config,
          ticks=config.confirm_hits)

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


def test_re_reading_one_payload_does_not_update_the_filter_again():
    config = _cfg()
    tracker = T.Tracker()
    payload = _stamped(100.0, _obj())
    assert _feed(tracker, payload, config=config).accepted_detection
    for _ in range(5):
        assert not _feed(tracker, payload, config=config).accepted_detection
    assert tracker.track.hits == 1


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


@pytest.mark.parametrize("coasting", [False, True])
def test_repeated_payloads_do_not_extend_the_coast_budget(coasting):
    config = _cfg(max_coast_s=0.35)
    tracker = T.Tracker()
    for index in range(config.confirm_hits):
        payload = _stamped(index, _obj())
        _feed(tracker, payload, config=config)

    if coasting:
        payload = _stamped(10)
        _feed(tracker, payload, config=config)
        assert tracker.track.state == T.COASTING

    _feed(tracker, payload, config=config, ticks=2)
    assert tracker.track is not None
    _feed(tracker, payload, config=config, ticks=2)
    assert tracker.track is None


def test_reacquisition_after_the_coast_budget_starts_a_new_track():
    config = _cfg(max_coast_s=0.35)
    tracker = T.Tracker()
    for index in range(config.confirm_hits):
        _feed(tracker, _stamped(index, _obj()), config=config)

    generation = tracker.track.generation
    _feed(tracker, _stamped(10), config=config)
    track = _feed(tracker, _stamped(11, _obj()), config=config, dt=config.max_coast_s)
    assert track.observed and track.hits == 1
    assert track.generation != generation


def test_a_payload_is_still_a_duplicate_after_a_blind_tick():
    config = _cfg()
    tracker = T.Tracker()
    payload = _stamped(1, _obj())
    _feed(tracker, payload, config=config)
    _feed(tracker, None, config=config)
    _feed(tracker, payload, config=config)
    assert tracker.track.hits == 1


def test_fresh_target_range_is_the_sample_before_filtering():
    config = _cfg()
    tracker = T.Tracker()
    for index in range(config.confirm_hits):
        _feed(tracker, _stamped(index, _obj()), config=config)

    payload = _stamped(10, _obj())
    track = _feed(tracker, payload, config=config, depth=_depth(2.5))
    assert track.accepted_detection
    assert track.measured_range_m == pytest.approx(2.5)
    assert track.range_m > 2.5

    track = _feed(tracker, payload, config=config, depth=_depth(2.5))
    assert track.observed
    assert not track.accepted_detection
    assert track.measured_range_m is None


@pytest.mark.parametrize("depth", [
    {"map": None, "bands": {"center": 3.0}},
    _depth(0.0),
    {**_depth(), "capture_invalid": True},
])
def test_accepted_bearing_with_no_valid_target_range_has_no_range_evidence(depth):
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj()), config=config, ticks=config.confirm_hits)
    track = _feed(tracker, _detections(_obj()), config=config, depth=depth)
    assert track.accepted_detection
    assert track.measured_range_m is None


@pytest.mark.parametrize("depth_id", ["frame_id", "source_frame_id"])
def test_two_detections_do_not_reuse_the_same_depth_frame_as_evidence(depth_id):
    config = _cfg()
    tracker = T.Tracker()
    for index in range(config.confirm_hits):
        _feed(tracker, _stamped(index, _obj()), config=config)

    def depth_frame(frame, receipt):
        return {**_depth(), "frame_id": receipt, depth_id: frame}

    track = _feed(tracker, _stamped(10, _obj()), config=config,
                  depth=depth_frame(20, 30))
    assert track.measured_range_m == pytest.approx(3.0)
    track = _feed(tracker, _stamped(11, _obj()), config=config,
                  depth=depth_frame(20, 31))
    assert track.accepted_detection
    assert track.measured_range_m is None
    track = _feed(tracker, _stamped(12, _obj()), config=config,
                  depth=depth_frame(21, 32))
    assert track.measured_range_m == pytest.approx(3.0)


def test_reordered_detection_and_depth_payloads_cannot_add_evidence_twice():
    config = _cfg(confirm_hits=2)
    tracker = T.Tracker()
    first = dict(_depth(), frame_id=20)
    second = dict(_depth(), frame_id=21)
    _feed(tracker, _stamped(1, _obj()), config=config, depth=first)
    _feed(tracker, _stamped(2, _obj()), config=config, depth=second)

    track = _feed(tracker, _stamped(1, _obj()), config=config, depth=first)
    assert track.hits == 2 and not track.accepted_detection
    assert track.measured_range_m is None
    track = _feed(tracker, _stamped(3, _obj()), config=config, depth=first)
    assert track.accepted_detection and track.measured_range_m is None

def test_a_payload_with_no_stamp_is_counted_every_time():
    """A detector that does not stamp its output cannot be de-duplicated, and
    the safe fallback is to treat each read as evidence — over-counting is the
    behaviour that existed before, not a new hazard."""
    config = _cfg()
    tracker = T.Tracker()
    for _ in range(config.confirm_hits):
        drivable = _feed(tracker, _detections(_obj()), config=config)
    assert drivable is not None


# ── re-acquisition: reviving a track has the same consequences as starting one ──

def test_first_matching_return_revives_a_coasted_track():
    config = _cfg()
    tracker = T.Tracker()
    original = _feed(tracker, _detections(_obj()), config=config)
    _feed(tracker, _detections(), config=config, ticks=3)
    track = _feed(tracker, _detections(_obj(bearing=.05)), config=config)
    assert track is original and track.observed and track.accepted_detection


@pytest.mark.parametrize("stamped", [True, False])
def test_new_depth_can_associate_a_person_on_the_same_recovery_detection(stamped):
    config = _cfg(local_controller="dwa", max_coast_s=.2)
    tracker = T.Tracker()
    person = _obj("person", bearing=.15)

    def observe(receipt, depth, dt=.1):
        frame = _detections(person)
        if stamped:
            frame["timestamp"] = receipt / 1000
        return tracker.step(detections=frame, depth=depth, target="person", config=config,
                            ego=STILL, dt=dt, receipt_ms=receipt, retain_identity=True)

    original = observe(1000, dict(_depth(), frame_id=1000))
    foreground = dict(_depth(.6), frame_id=1300)
    assert observe(1300, foreground, dt=.3) is None
    assert observe(1300, dict(_depth(), frame_id=1200)) is None
    assert observe(1300, foreground) is None
    recovered = observe(1300, dict(_depth(), frame_id=1400))
    assert recovered is original and recovered.observed and recovered.accepted_detection
    assert observe(1300, dict(_depth(), frame_id=1500)) is recovered
    assert not recovered.accepted_detection


def test_expired_person_associates_against_last_measured_front_after_prediction_drifts():
    config = _cfg(local_controller="dwa", max_coast_s=.2)
    tracker = T.Tracker()
    person = _obj("person")
    original = _feed(tracker, _stamped(1., person), config=config, target="person")
    original.front_reference = np.array([3., 0.])
    original.mean[:2] = [7., 0.]

    def recover(stamp, distance, dt):
        return tracker.step(detections=_stamped(stamp, person), depth=_depth(distance), target="person",
                            config=config, ego=STILL, dt=dt, retain_identity=True)

    assert recover(1.3, 7., .3) is None
    assert recover(1.4, 3., .1) is original
    assert original.measured_range_m == pytest.approx(3.)


def test_reacquiring_a_static_target_after_ego_motion_preserves_zero_velocity():
    config = _cfg()
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj()), config=config, ticks=config.confirm_hits)
    _feed(tracker, _detections(), config=config,
          ego=(0.4, 0.0, 1.0), dt=0.2)

    # The robot translated 8 cm toward the target and rotated 0.2 rad left.
    seen = _detections(_obj(bearing=0.2 / config.half_fov_rad))
    track = _feed(tracker, seen, config=config, depth=_depth(2.92),
                  ticks=config.confirm_hits)
    assert track.state == T.CONFIRMED
    assert track.mean[2:] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_the_gate_cannot_open_wider_than_the_target_could_walk():
    """Covariance growth alone let the gate reach several metres. A target that
    was four metres away is not suddenly at arm's length; it did not walk there,
    so whatever is at arm's length is something else."""
    config = _cfg(max_target_speed=1.0)
    tracker = T.Tracker()
    _feed(tracker, _detections(_obj(bearing=0.0)), config=config,
          depth=_depth(4.0), ticks=config.confirm_hits)
    _feed(tracker, _detections(), config=config, depth=_depth(4.0), ticks=3)

    for _ in range(config.confirm_hits):
        track = _feed(tracker, _detections(_obj(bearing=0.0)), config=config,
                      depth=_depth(0.7))
    assert track.state != T.CONFIRMED, "a 3.3 m jump was accepted as the target"


def test_person_front_surface_uses_latest_valid_depth_frame():
    from plugins.navi.camera import HorizontalIntrinsics

    config = _cfg(confirm_hits=1, depth_intrinsics=HorizontalIntrinsics(80, 60., 40.))
    person = dict(_obj("person"), bbox_norm=(.1, .1, .9, .9))
    tracker = T.Tracker()
    image = np.full((60, 80), 2.5)
    image[16:23, 51:58] = 1.6
    image[15, 21] = .24
    image[39:49, 26:38] = .35
    ground = np.zeros_like(image, dtype=bool)
    ground[39:49, 26:38] = True

    def feed(stamp, source, frame=image):
        return _feed(tracker, {"timestamp": stamp, "objects": [person]}, config=config,
                     target="person", depth={"map": frame, "ground": ground, "frame_id": stamp,
                                             "source_frame_id": source})

    feed(100, 10)
    result = tracker.track
    assert result.range_m == pytest.approx(2.5)
    assert math.hypot(*result.measured_point.standoff_point) == pytest.approx(math.hypot(1.6, 1.6 * 12 / 60), abs=.03)
    assert result.bearing_rad == pytest.approx(0.)
    assert result.measured_range_m == result.range_m

    changed = image.copy()
    changed[16:23, 51:58] = 1.05
    for stamp, source in ((200, 10), (300, 9)):
        result = feed(stamp, source, changed)
        assert result.range_m > 1.5
        assert result.measured_range_m is None
    result = feed(300, 11, changed)  # Still the same valid detection payload.
    assert result.range_m == pytest.approx(2.5)
    assert math.hypot(*result.measured_point.standoff_point) < 1.1
    assert result.measured_range_m == result.range_m

    for stamp in range(400, 1100, 100):
        result = feed(stamp, 11, changed)
    assert result.front_fix is None
    assert not result.navigation_range_known


@pytest.mark.parametrize("foreground,expected", [(1.03, 3.70), (3.40, 3.40)])
def test_person_front_surface_rejects_a_separate_near_layer_at_box_bottom(foreground, expected):
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.depth import front_surface

    frame = np.full((480, 640), 3.70, dtype=np.float32)
    frame[171:187, 303:352] = foreground
    bbox = (284 / 640, 0., 361 / 640, 188 / 480)
    point = front_surface(frame, bbox, HorizontalIntrinsics(640, 492., 318.))

    assert np.linalg.norm(point) == pytest.approx(expected)


@pytest.mark.parametrize(("edge", "expected"), [(True, 2.61), (False, .52)])
def test_person_front_surface_distinguishes_side_occlusion_from_an_interior_near_patch(edge, expected):
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.depth import front_surface

    frame = np.full((480, 640), 2.61, dtype=np.float32)
    left = 162 if edge else 170
    frame[203:308, left:left + 30] = .52
    bbox = (162 / 640, 38 / 480, 215 / 640, 308 / 480)
    point = front_surface(frame, bbox, HorizontalIntrinsics(640, 315., 337.))

    assert point[0] == pytest.approx(expected)
    assert frame[205, left + 4] == pytest.approx(.52)


def test_person_arrival_ignores_bottom_foreground_connected_by_filled_depth():
    from plugins.navi.camera import HorizontalIntrinsics

    cfg = _cfg(stop_distance_m=.60, stop_tolerance_m=.05,
               depth_intrinsics=HorizontalIntrinsics(80, 60., 40.))
    state = P.State(target="person")
    person = dict(_obj("person"), bbox_norm=(.1, .1, .9, .9))
    image = np.full((60, 80), 2.)
    # Mixed depth bridges the foreground and body, so the sorted box depths
    # have no large gap. The near return is confined to the bottom of the box.
    image[45:54, 32:36] = .62
    image[45:54, 36:72] = np.linspace(.62, 2., 36)
    for stamp in (100, 200, 300, 400):
        decision = P.step(detections=_stamped(stamp, person), depth=dict(map=image, frame_id=stamp),
                          odom=dict(vx=0., vy=0., wz=0.), config=cfg, state=state, dt=.1)
    assert not state.arrived
    assert state.tracker.track.range_m == pytest.approx(2., abs=.02)
    assert decision.status != P.ARRIVED
    assert image.min() == pytest.approx(.62)


def test_person_front_surface_ignores_a_new_depth_frame_without_a_return():
    from plugins.navi.camera import HorizontalIntrinsics

    cfg = _cfg(confirm_hits=1, depth_intrinsics=HorizontalIntrinsics(80, 60., 40.))
    tracker = T.Tracker()
    person = dict(_obj("person"), bbox_norm=(.1, .1, .9, .9))
    image = np.full((60, 80), 2.5)
    _feed(tracker, {"timestamp": 100, "objects": [person]}, config=cfg,
          target="person", depth={"map": image, "frame_id": 100})
    distance = tracker.track.measured_range_m
    assert distance == pytest.approx(2.5, abs=.03)
    track = _feed(tracker, {"timestamp": 200, "objects": [person]}, config=cfg,
                  target="person", depth={"map": np.zeros_like(image), "frame_id": 200})
    assert track.measured_range_m is None
    assert track.range_m == pytest.approx(distance)


@pytest.mark.parametrize("new_detection", [False, True])
@pytest.mark.parametrize("detouring", [False, True])
def test_person_foreground_patch_cannot_report_arrival_or_move_detour_goal(new_detection, detouring):
    from plugins.navi.camera import HorizontalIntrinsics

    cfg = _cfg(depth_intrinsics=HorizontalIntrinsics(80, 60., 40.))
    state = P.State(target="person")
    person = dict(_obj("person"), bbox_norm=(.1, .1, .9, .9))
    image = np.full((60, 80), 3.)

    def step(stamp, frame):
        return P.step(detections=_stamped(stamp, person), depth=frame,
                      odom=dict(vx=0., vy=0., wz=0.), config=cfg, state=state, dt=.1)

    for stamp in (100, 200, 300):
        result = step(stamp, dict(map=image, frame_id=stamp))
    track = state.tracker.track
    assert track.observed and result.distance_m == pytest.approx(3.)
    if detouring:
        state.goal_anchor = P.DetourGoal(3., 0., track=track)

    image[22:29, 35:45] = .8
    result = step(400 if new_detection else 300, dict(map=image, frame_id=400))
    assert result.status != P.ARRIVED
    assert result.distance_m == pytest.approx(3.)
    assert track.observed and track.measured_range_m is None
    if detouring:
        assert state.goal_anchor.x == pytest.approx(3.)
        assert state.goal_anchor.verified_range_m is None


@pytest.mark.parametrize("intervals", [(.9, .1, .1), (.1,) * 11])
def test_person_foreground_patch_does_not_become_evidence_after_last_fix_expires(intervals):
    from plugins.navi.camera import HorizontalIntrinsics

    cfg = _cfg(local_planning=True, max_depth_age_ms=800,
               depth_intrinsics=HorizontalIntrinsics(80, 60., 40.))
    tracker = T.Tracker()
    person = dict(_obj("person"), bbox_norm=(.1, .1, .9, .9))
    image = np.full((60, 80), 3.)

    def feed(stamp, dt):
        return _feed(tracker, _stamped(stamp, person), config=cfg, target="person",
                     depth=dict(map=image, frame_id=stamp), dt=dt)

    for stamp in (100, 200, 300):
        track = feed(stamp, .1)
    assert track.observed and track.measured_range_m == pytest.approx(3.)

    image[22:29, 35:45] = 1.6
    for dt in intervals:
        stamp += round(dt * 1000)
        track = feed(stamp, dt)
        assert track.observed and track.measured_range_m is None
    assert track.front_fix is None

    # The first plausible fresh return (feet below the box centre) can be
    # accepted immediately, even if the detector has not published again.
    image[22:29, 35:45] = 3.
    image[44:51, 35:45] = 2.8
    track = _feed(tracker, _stamped(stamp, person), config=cfg, target="person",
                  depth=dict(map=image, frame_id=stamp + 100))
    assert track.measured_range_m == pytest.approx(3.)
    assert math.hypot(*track.measured_point.standoff_point) == pytest.approx(2.8)


def test_local_front_range_retains_only_the_remaining_depth_freshness_window():
    from plugins.navi.camera import HorizontalIntrinsics

    cfg = _cfg(confirm_hits=1, local_planning=True, max_obs_age_ms=500, max_depth_age_ms=1200,
               depth_intrinsics=HorizontalIntrinsics(80, 60., 40.))
    tracker = T.Tracker()
    person = dict(_obj("person"), bbox_norm=(.1, .1, .9, .9))
    depth = dict(map=np.full((60, 80), 2.5), capture_ms=700, frame_id=1000)

    def feed(now, dt):
        return _feed(tracker, {"timestamp": now, "objects": [person]}, config=cfg,
                     target="person", depth=dict(depth, control_ms=now), dt=dt)

    feed(1000, .1)
    seen = feed(1600, .6)
    assert seen.observed and seen.front_fix is not None
    assert seen.measured_range_m is None
    expired = feed(1910, .31)
    assert expired.observed and expired.front_fix is None


def test_readonly_match_does_not_advance_tracker():
    import pickle

    tracker, config = T.Tracker(), _cfg()
    for stamp in (800, 900, 1000):
        _feed(tracker, _stamped(stamp, _obj()), config=config, depth=dict(_depth(), frame_id=stamp))
    before = pickle.dumps(tracker)
    snapshot = tracker.track.snapshot(1000)
    observation = _stamped(1100, _obj(bearing=.01, confidence=.25))
    for _ in range(3):
        match = T.match_observation(snapshot, observation, dict(_depth(), frame_id=1100),
                                    ego_motion=(.01, .02, -.03), elapsed_s=.1, target="chair", config=config)
        assert match.generation == snapshot.generation and match.range_m == pytest.approx(3.)
        assert match.source == 1100
        assert pickle.dumps(tracker) == before
    assert not T.match_observation(snapshot, observation, {"bands": {"center": 3.}},
                                   ego_motion=STILL, elapsed_s=.1, target="chair", config=config)
    old_mean, old_cov = snapshot.mean, snapshot.cov
    tracker.track.mean[:] = -25.
    tracker.track.cov[:] = 10.
    assert old_mean[0] == 3. and old_cov[0][0] < .1
    _feed(tracker, _stamped(2000, _obj()), config=config, dt=config.max_coast_s + .01)
    assert tracker.track.generation > snapshot.generation


@pytest.mark.parametrize("near_patch", [False, True])
def test_readonly_person_match_requires_a_continuous_front_return(near_patch):
    from plugins.navi.camera import HorizontalIntrinsics

    config = _cfg(depth_intrinsics=HorizontalIntrinsics(80, 60., 40.))
    person = dict(_obj("person"), bbox_norm=(.1, .1, .9, .9))
    tracker = T.Tracker()
    image = np.full((60, 80), 3.)
    for stamp in (100, 200, 300):
        _feed(tracker, _stamped(stamp, person), config=config, target="person",
              depth=dict(map=image, frame_id=stamp))
    snapshot = tracker.track.snapshot(300)
    if near_patch:
        image[22:29, 35:45] = .8
    observed = _stamped(400, person)
    depth = dict(map=image, frame_id=400)
    live = _feed(tracker, observed, config=config, target="person", depth=depth)
    match = T.match_observation(snapshot, observed, depth, ego_motion=STILL,
                                elapsed_s=.1, target="person", config=config)
    assert live.accepted_detection
    if near_patch:
        assert live.measured_range_m is None and match is None
        bearing = T.match_observation(snapshot, observed, depth, ego_motion=STILL,
                                      elapsed_s=.1, target="person", config=config,
                                      allow_missing_surface=True)
        assert bearing is not None and bearing.generation == snapshot.generation
        assert bearing.point is None and bearing.range_m is None
    else:
        assert match.range_m == live.measured_range_m
        assert T.match_observation(snapshot, observed, dict(depth, source_frame_id=299),
                                   ego_motion=STILL, elapsed_s=.1, target="person", config=config) is None


@pytest.mark.parametrize("foreground", [True, False])
def test_steering_range_and_physical_front_use_the_same_occluder_classification(foreground):
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.depth import sample_box

    cfg = _cfg(confirm_hits=1, depth_intrinsics=HorizontalIntrinsics(80, 60., 40.))
    tracker = T.Tracker()
    box = (.25, 0., .75, 1.)
    person = dict(_obj("person"), bbox_norm=box)
    image = np.full((60, 80), 3.0 if foreground else .53)
    image[38:, :] = .53
    assert sample_box(image, box) == pytest.approx(.53)
    for source, stamp in ((1, 100), (2, 200), (2, 300)):
        track = _feed(tracker, _stamped(source, person), config=cfg, target="person",
                      depth={"map": image, "frame_id": stamp})
    expected = 3.0 if foreground else .53
    assert track.measured_range_m == pytest.approx(expected)
    assert track.measured_point.standoff_point[0] == pytest.approx(expected)


@pytest.mark.parametrize("case, expected", [("wide", .85), ("narrow", .55), ("foot", .62),
                                           ("ground", .55), ("gap", .55), ("upper_person", .55),
                                           ("left", .85), ("missing_near", .055)])
def test_person_front_behind_a_wide_lower_occluder(case, expected):
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.depth import person_box_measurement

    depth = np.full((72, 160), 1.35)
    # Intermediate depth on the left closes the global histogram gap.
    depth[:, 42:50] = np.linspace(1.35, 2., 8)
    depth[52:, 48:56] = np.repeat(np.linspace(.85, 1.35, 5), 4)[:, None]
    depth[55:, 66:98 if case == "narrow" else 145] = .55
    ground = np.zeros_like(depth, dtype=bool)
    if case == "ground":
        ground[55:, 101:] = True
    if case == "gap":
        depth[55:, 102:105] = np.nan
    if case == "upper_person":
        depth[10:22, 60:65] = .69
    if case == "foot":
        depth[57:70, 57:63] = .62
    box = (.25, 0., .625, 1.)
    if case == "left":
        depth = depth[:, ::-1]
        box = (1. - box[2], box[1], 1. - box[0], box[3])
    if case == "missing_near":
        depth[55:, 66:145] = .055
        depth[55:, 102:105] = 0.
        depth[45:, 48:56] = np.repeat(np.linspace(.35, 1.35, 9), 3)[:, None]
    original = depth.copy()
    _, surface = person_box_measurement(depth, box, HorizontalIntrinsics(160, 150., 80.), ground)
    np.testing.assert_equal(depth, original)
    assert surface[0] == pytest.approx(expected)
