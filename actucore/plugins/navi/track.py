"""Following one thing through flicker and short occlusions.

Detections flicker. A chair that is really there appears in seven of ten
consecutive frames; a person walks past and the target is gone for half a
second; a partly-occluded object comes back at 0.2 confidence instead of 0.8.
Acting on the latest frame alone turns every one of those into a behaviour
change, which on a robot means a lurch, a spurious search, or a chase after
something that never existed.

This module keeps **one** track — the thing `navigate_to` was asked to reach —
and answers "where is it now" every tick, whether or not it was seen this tick.

── what is borrowed, and from where ─────────────────────────────────────────

Nothing here is new. Three ideas, each from a paper that earned it:

* **A track has a lifecycle, and the thresholds are symmetric** (SORT's
  `min_hits` / `max_age`). Before this, navi took **one** frame to start
  chasing something and **ten** to give up — and that asymmetry points the
  wrong way, because starting is the direction that moves a robot. Ultralytics'
  ByteTrack gets this right by default (a new `STrack` is `is_activated=False`
  until it matches again, so two frames), and we ask for three of five.

* **Low-confidence detections are not noise; they are what occlusion looks
  like** (ByteTrack, MOT20: ID switches down 71%). A detection below the
  confidence bar may not *create* a track, but it may *sustain* one that
  already exists, provided it lands inside the gate. The gate is what
  separates a genuinely dim detection from background.

* **Observations have lower variance than propagated estimates** (OC-SORT).
  Even at 1 px of per-frame noise, ten frames of coasting can accumulate an
  error the size of the object. So on re-acquisition the velocity is rebuilt
  from the two real observations either side of the gap rather than trusted
  from the coast.

── what is *not* borrowed, and why we are not using a library ───────────────

Every off-the-shelf tracker — ultralytics, boxmot, supervision, norfair —
carries its state as an image-space bounding box and associates by IoU. Two
things follow, and both are structural rather than a matter of effort:

**Our depth measurement has nowhere to go.** Their Kalman state is
(x, y, aspect, height) in pixels; the only proxy for distance is box height,
and we have a real depth map. Worse, during an occlusion the depth *at the
predicted box* measures the occluder, so range has to be coasted separately no
matter which tracker draws the box — and then two filters, one in pixels and one
in metres, are estimating the same object and can disagree. Two estimators that
can contradict each other is a worse outcome than either alone.

**Their ego-motion compensation cannot run here.** BoT-SORT's GMC and norfair's
`MotionEstimator` both estimate camera motion by optical flow on the raw frame.
actucore has no raw frame — navi subscribes to vop's JSON and a depth map. And
we do not need an estimate: `motus.odom/1` reports the robot's own twist, which
is the exact quantity they are approximating. (The GMC hook could be fed an
affine synthesised from yaw, but that is a workaround around a state vector that
still has no room for depth.)

So the state here is metric and body-frame: **the dominant source of apparent
target motion is our own rotation**, and this robot's deadband means `wz` is
either 0 or ≥1.0 rad/s with nothing between. At 63° horizontal FOV over 640 px
that is ~58 px of image shift appearing or vanishing within a single tick. A
constant-velocity image model cannot see that step coming. We do not have to:
it is the command we sent one tick ago.

**This is scoped to a single target of interest.** If navi ever needs to track
everything in the frame — say, to keep clear of several moving people — the
assignment problem becomes the main event and hand-rolling it would be a
mistake. Reach for boxmot then, and revisit the licence question with it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

TENTATIVE = "tentative"    # seen, not yet trusted enough to move a robot
CONFIRMED = "confirmed"
COASTING = "coasting"      # not seen this tick; position is predicted
REACQUIRING = "reacquiring"  # something matched after a coast, not yet believed
LOST = "lost"

# Body frame throughout: x forward, y **left**, metres. Same convention as
# motus.control/1's twist and motus.odom/1, so `vx`/`vy`/`wz` mean here exactly
# what they mean on the wire.


def _rotation(angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin], [sin, cos]], dtype=float)


def point_of(range_m: float, bearing_rad: float) -> np.ndarray:
    """(range, bearing) → a point in the body frame.

    Positive bearing is to the **right** — that is vop's convention for
    `position[0]` — and y is to the **left**, so the sine is negated. Getting
    this backwards mirrors the whole world and the robot chases reflections.
    """
    return np.array([range_m * math.cos(bearing_rad),
                     -range_m * math.sin(bearing_rad)], dtype=float)


def polar_of(point) -> tuple[float, float]:
    """The inverse: a body-frame point → (range, bearing), bearing right-positive."""
    x, y = float(point[0]), float(point[1])
    return math.hypot(x, y), math.atan2(-y, x)


@dataclass
class Track:
    """One followed object. `mean` is [x, y, vx, vy] in the current body frame."""

    mean: np.ndarray
    cov: np.ndarray
    hue: str = ""
    name: str = ""
    state: str = TENTATIVE
    hits: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    coast_s: float = 0.0
    age_s: float = 0.0
    # The last real observation and how long ago it was, for the re-update on
    # re-acquisition. Without it the coast's own velocity is all that is left,
    # and that is exactly the quantity that has been accumulating error.
    last_obs: np.ndarray | None = None
    since_obs_s: float = 0.0
    # Whether depth has ever actually reported a range for this object. A track
    # can exist without one — the summary-only depth path has no reading for a
    # band with nothing in it — and the policy has to be able to tell, because
    # "assumed 3 m" must not be allowed to decide when the robot has arrived.
    range_known: bool = False

    # Matches accumulated since the track started coasting. Resurrecting a
    # track decides what the robot chases, exactly as creating one does, so it
    # is held to the same evidence — see `_hit`.
    reacquire_hits: int = 0

    @property
    def drivable(self) -> bool:
        """Whether this track may move the robot. Tentative ones may not."""
        return self.state in (CONFIRMED, COASTING, REACQUIRING)

    @property
    def observed(self) -> bool:
        """A live, believed fix — not a prediction and not a fresh guess.

        What "arrived" is allowed to rest on. A coasted position is a good
        enough reason to keep walking and a bad one to declare the journey over.
        """
        return self.state == CONFIRMED and self.coast_s == 0.0

    @property
    def position(self) -> np.ndarray:
        return self.mean[:2]

    @property
    def range_m(self) -> float:
        return polar_of(self.position)[0]

    @property
    def bearing_rad(self) -> float:
        return polar_of(self.position)[1]

    @property
    def position_std_m(self) -> float:
        """One number for "how sure are we where it is", for reporting."""
        return float(math.sqrt(max(0.0, self.cov[0, 0] + self.cov[1, 1])))


def _clamp_speed(track, config) -> None:
    """Refuse to believe the target is moving faster than a target can move.

    A detection that jumps across the frame in one tick is a detection error —
    a box snapping to a different part of the same object, or a different object
    entirely — not an object that accelerated. The filter cannot tell the two
    apart and will happily infer several metres per second, then overshoot past
    the target on the next few ticks and command a turn in the **wrong
    direction**. On a robot whose slowest turn is 1 rad/s that overshoot is not
    a subtle numerical artefact; it is a visible twitch the wrong way.

    A walking person tops out around 2 m/s. Anything above that is the
    detector, so it is clipped rather than integrated.
    """
    speed = float(np.hypot(track.mean[2], track.mean[3]))
    if speed > config.max_target_speed:
        track.mean[2:] *= config.max_target_speed / speed


def _measurement_std(range_m: float, config) -> float:
    """How far off one observation can be, in metres.

    Isotropic on purpose. The honest model is anisotropic — range error is the
    depth sensor's, lateral error is `range × bearing_std` and so grows with
    distance — but at the distances this card operates in the two are the same
    order (at 4 m: 0.20 m lateral against ~0.15 m radial), and the difference
    only shapes the gate. An unvalidated covariance rotation is false precision
    in code that decides whether a robot keeps walking.
    """
    return max(config.range_std_m, range_m * config.bearing_std_rad)


class Tracker:
    """Holds the single track and the whole lifecycle. One instance per target."""

    def __init__(self):
        self.track: Track | None = None
        # The payload stamp the lifecycle last counted. See `step`.
        self._last_stamp = None
        # Why the last tick did what it did — surfaced in info(), because a
        # tracker that quietly refuses to confirm anything looks identical from
        # outside to a camera with nothing in front of it.
        self.last_reason = ""

    # ── the tick ─────────────────────────────────────────────────────────────

    def step(self, *, detections, depth, target: str, config, ego, dt: float,
             ego_measured: bool = True):
        """Predict, associate, update. Returns the track if it may drive, else None.

        `ego` is our own twist as (vx, vy, wz) in the body frame. `ego_measured`
        says whether that came from odometry or from what we last *commanded* —
        see `_predict`, which treats the two differently and must.
        `detections` may be None (no fresh payload), which counts as a miss
        rather than as nothing happening.
        """
        if self.track is not None:
            self._predict(ego, dt, config, ego_measured)
            self.track.age_s += dt
            self.track.since_obs_s += dt

        # **One payload is one piece of evidence, however many times it is read.**
        #
        # The policy ticks at `rate_hz` and the detector publishes at whatever
        # rate it manages; on r1_sz those are 10 Hz and 3.6 Hz, so the card sees
        # the same payload on roughly three consecutive ticks. Counting a hit
        # each time turns "three of the last five frames" into "one frame, read
        # three times" — which confirms a track off a single false positive,
        # the exact failure this lifecycle exists to prevent. It would also fold
        # the same measurement into the filter three times and shrink the
        # covariance as if three cameras had agreed.
        #
        # Prediction still runs above: our own motion continues whether or not
        # a new picture arrived. Only the *evidence* is skipped.
        stamp = (detections or {}).get("timestamp")
        if detections is not None and stamp is not None and stamp == self._last_stamp:
            return self.track if (self.track and self.track.drivable) else None
        self._last_stamp = stamp

        candidates = self._candidates(detections, depth, target, config)
        matched = self._associate(candidates, config)

        if matched is None:
            self._miss(dt, config, candidates)
        else:
            self._hit(matched, config)

        return self.track if (self.track and self.track.drivable) else None

    # ── 1. predict: mostly a statement about our own motion ──────────────────

    def _predict(self, ego, dt: float, config, ego_measured: bool = True):
        """Move the track by the target's own velocity and by ours, inverted.

        This is the ego-motion compensation, and on this robot it is the larger
        of the two terms by far: a target 2 m away, while the chassis turns at
        its minimum 1.0 rad/s, sweeps 0.2 m across the frame in a single 10 Hz
        tick. The target's own walking speed contributes a tenth of that.
        """
        track = self.track
        vx, vy, wz = (float(ego[0] or 0.0), float(ego[1] or 0.0),
                      float(ego[2] or 0.0))
        rotation = _rotation(-wz * dt)
        translation = np.array([vx * dt, vy * dt], dtype=float)

        # p' = R(p + v·dt − d),  v' = R·v
        transition = np.zeros((4, 4))
        transition[:2, :2] = rotation
        transition[:2, 2:] = rotation * dt
        transition[2:, 2:] = rotation
        control = np.concatenate([-rotation @ translation, np.zeros(2)])

        track.mean = transition @ track.mean + control

        # Process noise: what the *target* might do that we have not modelled.
        #
        # With odometry wired, our own motion is known rather than guessed and
        # contributes nothing here. **Without it, the twist above is what we
        # asked for, not what happened** — and the two come apart in ordinary
        # ways: `dry_run`, a command the driver refused because the robot is
        # lying down, or the deadband zeroing a value the policy meant. Predict
        # a rotation that never occurred and the filter spends every tick
        # fighting observations that disagree with it, which shows up as an
        # estimate swinging either side of a target that is standing still.
        #
        # So when the twist is commanded, the process noise is inflated: the
        # filter leans on what it can see rather than on what we believe we did.
        accel = config.target_accel_std
        if not ego_measured:
            accel *= config.commanded_ego_noise_factor
        q_pos = (dt ** 4) / 4.0 * accel ** 2
        q_cross = (dt ** 3) / 2.0 * accel ** 2
        q_vel = (dt ** 2) * accel ** 2
        noise = np.zeros((4, 4))
        noise[:2, :2] = np.eye(2) * q_pos
        noise[:2, 2:] = np.eye(2) * q_cross
        noise[2:, :2] = np.eye(2) * q_cross
        noise[2:, 2:] = np.eye(2) * q_vel
        track.cov = transition @ track.cov @ transition.T + noise

    # ── 2. associate ─────────────────────────────────────────────────────────

    def _candidates(self, detections, depth, target: str, config) -> list:
        """Detections that could be our target, each as a body-frame point.

        The name/key matching is the same as it always was; what is new is that
        a match is a *candidate*, not a decision.
        """
        from . import policy as policy_mod

        objects = (detections or {}).get("objects") or []
        wanted = (target or "").strip().lower()
        if not wanted:
            return []

        out = []
        for obj in objects:
            confidence = float(obj.get("confidence") or 0)
            if confidence < config.sustain_confidence:
                continue
            key = policy_mod.object_key(obj).lower()
            name = str(obj.get("name", "")).lower()
            if not (key == wanted or name == wanted or wanted in name):
                continue

            bearing_rad = policy_mod._bearing(obj) * config.half_fov_rad
            range_m = policy_mod.target_distance(obj, depth, config)
            measured = range_m is not None
            if not measured:
                # No depth at this object — a hole in the map, or the summary
                # fallback with nothing in that band. Place it at the range we
                # already believe, which makes this a bearing-only update in
                # disguise: it can correct where the target is *pointing* and
                # must not be allowed to claim anything about how far.
                range_m = (self.track.range_m if self.track is not None
                           else config.assumed_range_m)
            out.append({
                "obj": obj,
                "point": point_of(range_m, bearing_rad),
                "confidence": confidence,
                "hue": policy_mod.hue_of(obj),
                "name": obj.get("name") or "",
                "range_measured": measured,
                "range_m": range_m,
            })
        return out

    def _associate(self, candidates, config):
        """Pick the candidate this track is looking at, or None.

        ByteTrack's two stages, collapsed to one track: high-confidence
        detections get first refusal; if none of them lands in the gate, the
        low-confidence ones are tried against the same gate. A dim detection
        that agrees with where we predicted the target to be is the target seen
        through something; a dim detection anywhere else is background.
        """
        if not candidates:
            return None

        if self.track is None:
            # Nothing to gate against — only a confident detection may start a
            # track, and it starts tentative. This is the half of the
            # asymmetry that used to move the robot on a single frame.
            confident = [c for c in candidates
                         if c["confidence"] >= config.min_confidence]
            if not confident:
                return None
            return max(confident, key=lambda c: c["confidence"])

        high = [c for c in candidates if c["confidence"] >= config.min_confidence]
        low = [c for c in candidates if c["confidence"] < config.min_confidence]
        for pool in (high, low):
            best, best_cost = None, None
            for candidate in pool:
                cost = self._gate_cost(candidate, config)
                if cost is None:
                    continue
                if best_cost is None or cost < best_cost:
                    best, best_cost = candidate, cost
            if best is not None:
                return best
        return None

    def _gate_cost(self, candidate, config):
        """Mahalanobis distance² to the prediction, or None if outside the gate.

        The gate widens on its own while coasting, because the covariance
        grows — which is the right behaviour and needs no separate rule: the
        longer the target has been hidden, the further from the prediction we
        are willing to believe it reappeared.
        """
        track = self.track
        residual = candidate["point"] - track.position
        std = _measurement_std(candidate["range_m"], config)
        innovation = track.cov[:2, :2] + np.eye(2) * std ** 2
        try:
            cost = float(residual @ np.linalg.solve(innovation, residual))
        except np.linalg.LinAlgError:              # pragma: no cover - degenerate
            return None
        if cost > config.gate_chi2:
            return None

        # **The gate widens while coasting, and that had no ceiling.** Growing
        # it is right — the longer the target has been hidden, the further from
        # the prediction it may legitimately reappear — but after 1.2 s the
        # covariance alone opened it to about 3.7 m, and the detector's class is
        # not stable enough to survive an opening that size. On r1_sz a track on
        # a person four metres away was resurrected onto a traffic cone at 0.7 m
        # that vop had labelled `person` for one frame, and the card then
        # reported arrival.
        #
        # So it is also capped by what the target could physically have done:
        # nothing moves faster than `max_target_speed`, and the ego motion is
        # already in the prediction, so this residual is the target's own travel.
        reach = (config.max_target_speed * max(track.since_obs_s, 1e-3)
                 + 3.0 * std)
        if float(np.hypot(*residual)) > reach:
            return None
        # Colour is a tie-break, never a gate. `object_key` already excludes
        # brightness because it swings with the lighting; hue is steadier but
        # still flickers, and rejecting on it would drop a track for a cloud
        # passing the window.
        if track.hue and candidate["hue"] and candidate["hue"] != track.hue:
            cost += config.hue_mismatch_cost
        return cost

    # ── 3. update ────────────────────────────────────────────────────────────

    def _hit(self, candidate, config):
        point = candidate["point"]
        if self.track is None:
            self._create(candidate, config)
            self.last_reason = (f"看到 {candidate['name']!r}，建立候选轨迹"
                                f"（需 {config.confirm_hits} 帧确认）")
            return

        track = self.track
        was_coasting = track.state in (COASTING, REACQUIRING)

        if was_coasting:
            # **One frame resurrects a track; three create one.** That asymmetry
            # is the same bug that made a single false positive start a chase,
            # moved to the other end of the lifecycle — and it is worse here,
            # because the gate is at its widest exactly when the evidence is at
            # its thinnest. Until the count is met the track keeps coasting on
            # its prediction: still drivable, still not jumped.
            track.reacquire_hits += 1
            track.history.append(True)
            # `coast_s` deliberately keeps its value: until the match is
            # accepted the position being published is still a prediction, and
            # `describe()` should say so. It also bounds this state — at most
            # `confirm_hits` ticks can pass before the track is either accepted
            # or dropped by the coast budget.
            if track.reacquire_hits < config.confirm_hits:
                track.state = REACQUIRING
                self.last_reason = (
                    f"疑似重新看到目标（{track.reacquire_hits}/{config.confirm_hits}）"
                    "，先按预测继续，确认了才接受它的位置")
                return
            track.reacquire_hits = 0

        if was_coasting and track.last_obs is not None and track.since_obs_s > 1e-6:
            # OC-SORT's observation-centric re-update, in the form this state
            # vector allows. The velocity carried through the coast is the one
            # quantity that has been integrating its own error the whole time,
            # so it is rebuilt from the two real observations either side of the
            # gap and the covariance is reset rather than inherited.
            velocity = (point - track.last_obs) / track.since_obs_s
            track.mean = np.concatenate([point, velocity])
            track.cov = self._initial_cov(candidate, config)
        else:
            std = _measurement_std(candidate["range_m"], config)
            if not candidate["range_measured"]:
                # Bearing-only. Inflating the measurement noise is what stops
                # this from asserting a range it never measured.
                std *= config.bearingless_std_factor
            measurement = np.zeros((2, 4))
            measurement[0, 0] = measurement[1, 1] = 1.0
            innovation = (measurement @ track.cov @ measurement.T
                          + np.eye(2) * std ** 2)
            gain = track.cov @ measurement.T @ np.linalg.inv(innovation)
            track.mean = track.mean + gain @ (point - track.position)
            track.cov = (np.eye(4) - gain @ measurement) @ track.cov

        _clamp_speed(track, config)
        track.hue = candidate["hue"] or track.hue
        track.range_known = track.range_known or candidate["range_measured"]
        track.hits += 1
        track.history.append(True)
        track.coast_s = 0.0
        track.last_obs = point.copy()
        track.since_obs_s = 0.0

        if track.state == TENTATIVE:
            if track.hits >= config.confirm_hits:
                track.state = CONFIRMED
                self.last_reason = f"轨迹确认（{track.hits} 帧）"
            else:
                self.last_reason = (f"候选轨迹 {track.hits}/{config.confirm_hits} "
                                    "帧，还不驱动底盘")
        else:
            if was_coasting:
                self.last_reason = "目标重新出现，轨迹恢复"
            else:
                self.last_reason = ""
            track.state = CONFIRMED

    def _create(self, candidate, config):
        self.track = Track(
            mean=np.concatenate([candidate["point"], np.zeros(2)]),
            cov=self._initial_cov(candidate, config),
            hue=candidate["hue"],
            name=candidate["name"],
            state=TENTATIVE,
            hits=1,
            range_known=candidate["range_measured"],
            last_obs=candidate["point"].copy(),
        )
        self.track.history.append(True)

    @staticmethod
    def _initial_cov(candidate, config) -> np.ndarray:
        std = _measurement_std(candidate["range_m"], config)
        if not candidate["range_measured"]:
            std *= config.bearingless_std_factor
        cov = np.eye(4)
        cov[0, 0] = cov[1, 1] = std ** 2
        cov[2, 2] = cov[3, 3] = config.initial_speed_std ** 2
        return cov

    # ── 4. miss ──────────────────────────────────────────────────────────────

    def _miss(self, dt: float, config, candidates):
        track = self.track
        if track is None:
            self.last_reason = ("视野里没有匹配目标的检测" if not candidates
                                else "有疑似检测，但置信度不足以建立轨迹")
            return

        track.history.append(False)

        if track.state == TENTATIVE:
            # A candidate that cannot keep showing up is dropped rather than
            # kept around: it never moved the robot, and holding it would let a
            # ghost win the gate against a real detection later.
            if len(track.history) >= config.confirm_window and \
                    track.hits < config.confirm_hits:
                self.track = None
                self.last_reason = "候选轨迹没能在窗口内确认，丢弃"
            return

        track.state = COASTING
        track.reacquire_hits = 0
        track.coast_s += dt
        if track.coast_s > config.max_coast_s:
            self.track = None
            self.last_reason = (f"目标消失超过 {config.max_coast_s:.1f}s，"
                                "外推已不可信，放弃轨迹")
            return
        self.last_reason = (f"目标暂时看不见（{track.coast_s:.1f}s），"
                            "按自身运动推算位置继续")

    # ── reporting ────────────────────────────────────────────────────────────

    def describe(self) -> dict:
        """What the card puts in `info()`. Coasting must be visible: a robot
        walking towards a prediction looks exactly like one walking towards a
        thing it can see."""
        track = self.track
        if track is None:
            return {"state": LOST, "reason": self.last_reason}
        return {
            "state": track.state,
            "hits": track.hits,
            "coast_s": round(track.coast_s, 2),
            "age_s": round(track.age_s, 1),
            "observed": track.observed,
            "range_m": round(track.range_m, 2) if track.range_known else None,
            "position_std_m": round(track.position_std_m, 2),
            "reason": self.last_reason,
        }
