"""
tests/test_gesture_events.py — the per-frame activity channel turned into
sparse, edge-triggered gesture events.

Entirely synthetic and entirely free of ROS, which is the point: every decision
this module makes is a function of (label, track, frame time), so dwell,
release, cooldown and track loss are all testable without a camera, an engine
or a clock.

Times are passed in explicitly rather than taken from a clock, because the two
channels that feed this run at different rates (the body channel at the card's
fps, the hand channel throttled to about 4 Hz) and the thresholds are durations.
A test that slept would be testing the host's scheduler.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import vision_stubs  # noqa: F401  (installs the cv2 / ROS stubs)

from plugins.gesture_events import (  # noqa: E402
    DEFAULT_GESTURES,
    OPTIONAL_GESTURES,
    GestureEventTracker,
)


def person(track_id=1, activity="raising hand", score=0.9, position=(0.1, -0.2),
           point_direction=None):
    """One record in the shape `_PoseNode._describe` builds."""
    verdict = {"posture": "standing", "posture_confidence": 0.8}
    if activity is not None:
        verdict["activity"] = {"name": activity, "name_zh": "x", "score": score}
    if point_direction:
        verdict["point_direction"] = point_direction
    return {"id": track_id, "position": list(position), "verdict": verdict}


def feed(tracker, frames, dt=1 / 12.0, t0=0.0):
    """Run a list of per-frame `persons` lists through the tracker.

    Returns (all_events, final_time) so a test can keep going from where it
    left off without recomputing the clock.
    """
    events, t = [], t0
    for persons in frames:
        events.extend(tracker.update(persons, t))
        t += dt
    return events, t


# ── dwell ────────────────────────────────────────────────────────────────────

def test_nothing_is_emitted_before_the_hold_elapses():
    """An arm passing through the raised position must not announce itself.

    This is the whole reason there is a hold at all: `raising hand` is a
    single-frame geometric read, so a hand on its way to scratch a nose
    produces it for a few frames.
    """
    tracker = GestureEventTracker(hold_s=0.6)
    events, _ = feed(tracker, [[person()]] * 6)      # 6 frames at 12 fps = 0.5 s
    assert events == []


def test_start_fires_once_the_hold_elapses():
    tracker = GestureEventTracker(hold_s=0.6)
    events, _ = feed(tracker, [[person()]] * 10)     # 0.83 s
    assert len(events) == 1
    assert events[0]["event"] == "gesture_start"
    assert events[0]["gesture"] == "raising hand"
    assert events[0]["track"] == 1


def test_start_fires_exactly_once_however_long_it_is_held():
    """The topic is sparse. Holding a gesture for seconds is one message."""
    tracker = GestureEventTracker(hold_s=0.6)
    events, _ = feed(tracker, [[person()]] * 120)    # 10 s
    assert [e["event"] for e in events] == ["gesture_start"]


def test_hold_is_measured_in_seconds_not_frames():
    """The same frame count must behave differently at two sample rates.

    The hand channel is throttled to about 4 Hz and the body channel runs at
    the card's fps. A frame-counting hold would fire at the same frame in both,
    which is 1 s in one and 333 ms in the other — so four frames is enough at
    4 Hz and must not be enough at 12 fps.
    """
    slow = GestureEventTracker(hold_s=0.6)
    slow_events, _ = feed(slow, [[person()]] * 4, dt=0.25)      # spans 0.75 s
    assert [e["event"] for e in slow_events] == ["gesture_start"]

    fast = GestureEventTracker(hold_s=0.6)
    fast_events, _ = feed(fast, [[person()]] * 4, dt=1 / 12.0)  # spans 0.25 s
    assert fast_events == []


def test_a_changing_label_restarts_the_hold():
    tracker = GestureEventTracker(hold_s=0.6)
    frames = ([[person(activity="raising hand")]] * 5
              + [[person(activity="hand waving")]] * 5)
    events, _ = feed(tracker, frames)
    assert events == []


# ── release ──────────────────────────────────────────────────────────────────

def test_one_dropped_frame_does_not_close_the_event():
    """A momentary occlusion, or one frame of the stabiliser holding a label
    back, must not produce an end/start pair."""
    tracker = GestureEventTracker(hold_s=0.3, release_s=0.5)
    frames = [[person()]] * 6 + [[person(activity=None)]] + [[person()]] * 6
    events, _ = feed(tracker, frames)
    assert [e["event"] for e in events] == ["gesture_start"]


def test_end_fires_after_the_release_window():
    tracker = GestureEventTracker(hold_s=0.3, release_s=0.5)
    frames = [[person()]] * 6 + [[person(activity=None)]] * 12   # 1 s absent
    events, _ = feed(tracker, frames)
    kinds = [e["event"] for e in events]
    assert kinds == ["gesture_start", "gesture_end"]
    assert events[1]["reason"] == "released"
    assert events[1]["held_s"] >= 0.0


def test_switching_to_another_gesture_closes_the_first():
    tracker = GestureEventTracker(hold_s=0.3, release_s=0.2, cooldown_s=0.0)
    frames = ([[person(activity="raising hand")]] * 6
              + [[person(activity="hand waving")]] * 12)
    events, _ = feed(tracker, frames)
    kinds = [(e["event"], e["gesture"]) for e in events]
    assert ("gesture_start", "raising hand") in kinds
    assert ("gesture_end", "raising hand") in kinds
    assert ("gesture_start", "hand waving") in kinds


# ── cooldown ─────────────────────────────────────────────────────────────────

def test_cooldown_suppresses_an_immediate_repeat():
    """Somebody waving in bursts is one request, not one per burst."""
    tracker = GestureEventTracker(hold_s=0.3, release_s=0.3, cooldown_s=2.0)
    burst = [[person()]] * 6 + [[person(activity=None)]] * 6
    events, t = feed(tracker, burst * 3)
    assert [e["event"] for e in events] == ["gesture_start", "gesture_end"]


def test_cooldown_lapses_and_the_hold_then_starts_from_scratch():
    """After the cooldown the gesture may fire again — but only after a fresh
    hold, not the instant the cooldown expires."""
    tracker = GestureEventTracker(hold_s=0.5, release_s=0.3, cooldown_s=1.0)
    events, t = feed(tracker, [[person()]] * 10 + [[person(activity=None)]] * 8)
    assert [e["event"] for e in events] == ["gesture_start", "gesture_end"]

    # idle well past the cooldown, then one single frame of the gesture
    more, t = feed(tracker, [[person(activity=None)]] * 24, t0=t)
    one, t = feed(tracker, [[person()]], t0=t)
    assert one == []                                   # hold has not elapsed

    rest, _ = feed(tracker, [[person()]] * 10, t0=t)
    assert [e["event"] for e in rest] == ["gesture_start"]


def test_cooldown_is_per_gesture_not_per_track():
    tracker = GestureEventTracker(hold_s=0.3, release_s=0.2, cooldown_s=5.0)
    frames = ([[person(activity="raising hand")]] * 6
              + [[person(activity=None)]] * 6
              + [[person(activity="hand waving")]] * 8)
    events, _ = feed(tracker, frames)
    gestures = [(e["event"], e["gesture"]) for e in events]
    assert ("gesture_start", "hand waving") in gestures


# ── tracks ───────────────────────────────────────────────────────────────────

def test_two_people_are_independent():
    tracker = GestureEventTracker(hold_s=0.3)
    frames = [[person(1, "raising hand"), person(2, activity=None)]] * 6
    frames += [[person(1, "raising hand"), person(2, "hand waving")]] * 6
    events, _ = feed(tracker, frames)
    starts = {(e["track"], e["gesture"]) for e in events
              if e["event"] == "gesture_start"}
    assert starts == {(1, "raising hand"), (2, "hand waving")}


def test_a_lost_track_closes_its_gesture():
    """Otherwise the stream has an unmatched open: the cooldown never arms and
    a consumer tracking state cannot tell "still holding" from "walked away"."""
    tracker = GestureEventTracker(hold_s=0.3)
    events, t = feed(tracker, [[person()]] * 6)
    assert [e["event"] for e in events] == ["gesture_start"]

    gone, _ = feed(tracker, [[]], t0=t)
    assert len(gone) == 1
    assert gone[0]["event"] == "gesture_end"
    assert gone[0]["reason"] == "track_lost"
    assert gone[0]["track"] == 1


def test_a_lost_track_with_no_active_gesture_is_silent():
    tracker = GestureEventTracker(hold_s=0.6)
    events, t = feed(tracker, [[person()]] * 3)        # still in the hold
    gone, _ = feed(tracker, [[]], t0=t)
    assert events == [] and gone == []


def test_reset_drops_all_state():
    tracker = GestureEventTracker(hold_s=0.3)
    feed(tracker, [[person()]] * 6)
    tracker.reset()
    # No end event for the dropped track, and no cooldown carried over.
    events, _ = feed(tracker, [[person()]] * 6, t0=100.0)
    assert [e["event"] for e in events] == ["gesture_start"]


# ── payload ──────────────────────────────────────────────────────────────────

def test_every_event_carries_priority():
    """collector._PRIORITY_SOURCES has no `dds` entry, so a DDS event scores
    P=0 unless its JSON names a priority. Without this field a wave is
    delivered, logged and ignored."""
    tracker = GestureEventTracker(hold_s=0.3)
    events, t = feed(tracker, [[person()]] * 6)
    gone, _ = feed(tracker, [[]], t0=t)
    for event in events + gone:
        assert event["priority"] == 1


def test_priority_is_configurable():
    tracker = GestureEventTracker(hold_s=0.3, priority=3)
    events, _ = feed(tracker, [[person()]] * 6)
    assert events[0]["priority"] == 3


def test_pointing_carries_the_direction():
    """A pointing event without the direction is not actionable — the
    direction is the entire content of the gesture."""
    tracker = GestureEventTracker(hold_s=0.3)
    frames = [[person(activity="point to something", point_direction="left")]] * 6
    events, _ = feed(tracker, frames)
    assert events[0]["point_direction"] == "left"


def test_position_rides_along():
    tracker = GestureEventTracker(hold_s=0.3)
    events, _ = feed(tracker, [[person(position=(-0.4, 0.1))]] * 6)
    assert events[0]["position"] == [-0.4, 0.1]


# ── whitelist and score ──────────────────────────────────────────────────────

def test_an_unlisted_activity_is_not_a_gesture():
    tracker = GestureEventTracker(hold_s=0.3)
    events, _ = feed(tracker, [[person(activity="walking")]] * 24)
    assert events == []


def test_the_whitelist_is_configurable():
    tracker = GestureEventTracker(gestures=("clapping",), hold_s=0.3)
    events, _ = feed(tracker, [[person(activity="clapping")]] * 6)
    assert [e["gesture"] for e in events] == ["clapping"]
    assert tracker.gestures == ("clapping",)


def test_posture_is_never_a_gesture():
    """Everybody always has a posture, so treating one as a gesture would make
    every standing person permanently signal."""
    tracker = GestureEventTracker(gestures=("standing",), hold_s=0.3)
    events, _ = feed(tracker, [[person(activity=None)]] * 24)
    assert events == []


def test_min_score_rejects_a_weak_label():
    tracker = GestureEventTracker(hold_s=0.3, min_score=0.7)
    events, _ = feed(tracker, [[person(score=0.4)]] * 24)
    assert events == []


def test_min_score_admits_a_strong_label():
    tracker = GestureEventTracker(hold_s=0.3, min_score=0.7)
    events, _ = feed(tracker, [[person(score=0.85)]] * 6)
    assert [e["event"] for e in events] == ["gesture_start"]


def test_a_geometry_label_with_no_score_is_not_rejected_by_min_score():
    """The geometry rules report some activities without a score. Treating a
    missing score as zero would silently drop every one of them the moment a
    min_score was configured."""
    tracker = GestureEventTracker(hold_s=0.3, min_score=0.7)
    frames = [[{"id": 1, "position": [0, 0],
                "verdict": {"activity": {"name": "raising hand",
                                         "name_zh": "举手", "score": None}}}]] * 6
    events, _ = feed(tracker, frames)
    assert [e["event"] for e in events] == ["gesture_start"]


def test_default_and_optional_vocabularies_do_not_overlap():
    assert not set(DEFAULT_GESTURES) & set(OPTIONAL_GESTURES)
