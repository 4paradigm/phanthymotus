#!/usr/bin/env python3
"""
plugins/gesture_events.py — edge-triggered gesture events for the pose card.

The pose card's `data/json` port already says what every person in frame is
doing, once per frame. That is the right shape for a dashboard and the wrong
shape for "somebody is calling me": agent-core copies the **whole message** of a
subscribed topic into the event bus as event text
(`agent-core/src/topic_subscriber.py`), so a person holding an open hand up for
three seconds at 12 fps is thirty-six full payloads of LLM context saying the
same thing. The robot does not need to be told thirty-six times; it needs to be
told once, when the gesture starts.

So this module turns the per-frame activity channel into **transitions**, and
the card publishes those on a topic of their own which stays silent the rest of
the time:

    {input}/poses/gesture     data/json, sparse — one message per transition

Three things about it are not arbitrary.

**The payload carries `priority`.** `collector._PRIORITY_SOURCES` is
`{asr, message, channel, subagent, acp, scheduler}` — `dds` is not in it, so a
DDS-sourced event scores P=0 and lands in the background batch rather than
reaching the main agent. But `collector._extract_priority()` parses the event
text first and honours a `priority` field when it finds one. A gesture event
without that field is delivered, logged, and ignored; with it, it steers. This
is the whole reason the module exists rather than the node just publishing a
state field.

**Dwell is in seconds, not frames.** The body channel runs at the card's fps
(12 by default) and the hand channel is throttled to about 4 Hz, so "hold for
three frames" means 250 ms in one and 750 ms in the other. Every threshold here
is a duration and the caller passes the frame time.

A gesture is nonetheless never announced from a **single** sighting, however
small `hold_s` is: the first frame a label appears on only registers it as a
candidate, and the earliest a start can fire is the frame after. That is the
floor the dwell is there to enforce — one frame of a label is exactly what a
limb passing through a pose produces — so `hold_s=0` means "as soon as it is
confirmed", not "immediately".

**A lost track ends its gesture.** Without that, a person who raises a hand and
walks out of frame leaves an event stream that opened and never closed: the
cooldown never arms, and any consumer tracking state believes the hand is still
up. `prune()` is not an optimisation, it is part of the protocol.
"""

from __future__ import annotations

from typing import Iterable, Optional

#: Activities that count as a gesture aimed at the robot, by default.
#:
#: Deliberately short. These three are the ones a person performs *at* a robot
#: to get its attention, and all three are available at any distance the body
#: is visible at — they are read from the COCO-17 skeleton, so they do not
#: depend on the hand engine or on being close enough for it to work.
#:
#: `hand waving` and `point to something` are produced by both the geometry
#: rules and the skeleton-action model (which calls them by the same names —
#: see pose_stgcn.NTU_TO_POSE_LABEL), `raising hand` by the geometry only.
DEFAULT_GESTURES = ("raising hand", "hand waving", "point to something")

#: Everything else a consumer might reasonably whitelist. Not on by default:
#: each of these comes from the learned backend's NTU-60 vocabulary alone, and
#: that model has no "no gesture" class, so a confident label here is worth
#: less than one the geometry corroborates.
OPTIONAL_GESTURES = (
    "clapping", "cheer up", "salute", "nod head / bow", "shake head",
    "cross hands in front", "put palms together",
)

#: How long the label must hold before a start is emitted. Long enough that an
#: arm passing through the raised position on its way somewhere else does not
#: announce itself; short enough to still feel like a response.
DEFAULT_HOLD_S = 0.6

#: How long the label must be *absent* before the gesture is declared over.
#: A single dropped frame, a momentary occlusion or one frame of the stabiliser
#: holding back a label must not close and reopen the event.
DEFAULT_RELEASE_S = 0.5

#: After an end, how long before the same gesture may start again on the same
#: track. Without this, somebody waving in bursts produces an event per burst
#: and the agent is steered on every one.
DEFAULT_COOLDOWN_S = 2.0


class _TrackState:
    """Per-track gesture state. One active gesture at a time, by construction:
    the activity channel reports one label per person per frame."""

    __slots__ = ("candidate", "candidate_since", "active", "active_since",
                 "last_seen", "cooldown_until")

    def __init__(self) -> None:
        self.candidate: Optional[str] = None
        self.candidate_since: float = 0.0
        self.active: Optional[str] = None
        self.active_since: float = 0.0
        #: When the active gesture was last observed, so release is measured
        #: from the last sighting rather than from the start.
        self.last_seen: float = 0.0
        #: gesture name -> wall time before which it may not start again
        self.cooldown_until: dict = {}


class GestureEventTracker:
    """Per-frame activity labels in, sparse transition events out.

    Pure logic: no ROS, no clock of its own, no I/O. The caller supplies the
    frame time, which is what makes the thresholds mean seconds of video rather
    than seconds of wall clock — the same choice `PoseTracker` makes, and for
    the same reason: a card throttled to 4 Hz and one running at 12 fps must
    behave identically.
    """

    def __init__(self, gestures: Iterable[str] = DEFAULT_GESTURES, *,
                 hold_s: float = DEFAULT_HOLD_S,
                 release_s: float = DEFAULT_RELEASE_S,
                 cooldown_s: float = DEFAULT_COOLDOWN_S,
                 min_score: float = 0.0,
                 priority: int = 1) -> None:
        self._gestures = frozenset(g for g in gestures if g)
        self._hold_s = max(0.0, float(hold_s))
        self._release_s = max(0.0, float(release_s))
        self._cooldown_s = max(0.0, float(cooldown_s))
        self._min_score = max(0.0, float(min_score))
        self._priority = int(priority)
        self._tracks: dict = {}

    @property
    def gestures(self) -> tuple:
        return tuple(sorted(self._gestures))

    def reset(self) -> None:
        self._tracks.clear()

    # ── the one entry point ──────────────────────────────────────────────

    def update(self, persons: list, now: float) -> list:
        """Feed one frame's people, get back the transitions it caused.

        `persons` are the records the node already built — each needs `id`,
        and the verdict's `activity` / `position` / `point_direction`. Returns
        a list of ready-to-publish dicts, usually empty.
        """
        events = []
        seen = set()
        for person in persons:
            track_id = person.get("id")
            if track_id is None:
                continue
            seen.add(track_id)
            events.extend(self._update_one(track_id, person, now))
        events.extend(self._prune(seen, now))
        return events

    def _update_one(self, track_id, person: dict, now: float) -> list:
        state = self._tracks.get(track_id)
        if state is None:
            state = self._tracks[track_id] = _TrackState()

        label, score = self._gesture_of(person)
        events = []

        if state.active is not None:
            if label == state.active:
                state.last_seen = now
                return events
            # Still inside the release window: the gesture is not over, this
            # frame simply did not see it. Returning an end here and a start on
            # the next frame is the flapping this window exists to prevent.
            if (now - state.last_seen) < self._release_s:
                return events
            events.append(self._event("gesture_end", state.active, person, now,
                                      held_s=round(max(0.0, state.last_seen
                                                       - state.active_since), 2),
                                      reason="released"))
            state.cooldown_until[state.active] = now + self._cooldown_s
            state.active = None
            state.candidate = None

        if label is None:
            state.candidate = None
            return events

        if now < state.cooldown_until.get(label, 0.0):
            # In cooldown. The candidate is cleared too, so the hold timer
            # starts from scratch once the cooldown lapses rather than firing
            # the instant it does.
            state.candidate = None
            return events

        if state.candidate != label:
            state.candidate = label
            state.candidate_since = now
            return events

        if (now - state.candidate_since) >= self._hold_s:
            state.active = label
            state.active_since = state.candidate_since
            state.last_seen = now
            state.candidate = None
            events.append(self._event("gesture_start", label, person, now,
                                      score=score))
        return events

    def _prune(self, seen: set, now: float) -> list:
        """Close out tracks that are no longer in frame.

        A track that vanishes mid-gesture must still produce its end, or the
        event stream has an unmatched open and nothing downstream can tell
        "still holding" from "walked away".
        """
        events = []
        for track_id in [k for k in self._tracks if k not in seen]:
            state = self._tracks.pop(track_id)
            if state.active is not None:
                events.append({
                    "priority": self._priority,
                    "event": "gesture_end",
                    "gesture": state.active,
                    "track": track_id,
                    "held_s": round(max(0.0, state.last_seen - state.active_since), 2),
                    "reason": "track_lost",
                    "t": round(now, 3),
                })
        return events

    # ── helpers ──────────────────────────────────────────────────────────

    def _gesture_of(self, person: dict) -> tuple:
        """The whitelisted gesture this person is performing, and its score.

        Reads the `activity` channel only. `posture` is deliberately not
        considered: it is a body shape that everybody always has, so treating
        one as a gesture would mean every standing person is permanently
        signalling.
        """
        verdict = person.get("verdict") or person
        activity = verdict.get("activity")
        if isinstance(activity, dict):
            name, score = activity.get("name"), activity.get("score")
        else:
            name, score = activity, verdict.get("activity_score")
        if not name or name not in self._gestures:
            return None, None
        score = 0.0 if score is None else float(score)
        if score and score < self._min_score:
            return None, None
        return name, score

    def _event(self, kind: str, gesture: str, person: dict, now: float,
               **extra) -> dict:
        verdict = person.get("verdict") or person
        payload = {
            # Read by collector._extract_priority() before any source matching.
            # Without it this is a P=0 background event — see the module
            # docstring.
            "priority": self._priority,
            "event": kind,
            "gesture": gesture,
            "track": person.get("id"),
            "t": round(now, 3),
        }
        position = person.get("position")
        if position is not None:
            payload["position"] = position
        # Where they are pointing is the whole content of a pointing gesture;
        # an event that says "they pointed" without it is not actionable.
        direction = verdict.get("point_direction")
        if direction:
            payload["point_direction"] = direction
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        return payload
