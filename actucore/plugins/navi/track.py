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

* **A track has a bounded lifetime.** Use the first detection that passes
  confidence and association checks. Keep the target through a short miss
  and expire it when the prediction is no longer reliable.

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
from dataclasses import dataclass

import numpy as np

from .depth import observation_ms
from .odom import PlanarPose

CONFIRMED = "confirmed"
COASTING = "coasting"      # not seen this tick; position is predicted
LOST = "lost"

# VOP's chromatic bins are circular. Adjacent bins can describe the same
# appearance after a small shift in the detector box or lighting.
_HUE_WHEEL = ("red", "orange", "yellow", "lime", "green", "teal", "cyan",
              "azure", "blue", "violet", "magenta", "pink")


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


@dataclass(frozen=True)
class ObservationFrame:
    """Planar map from camera-forward/left coordinates to the body pivot."""

    axes: tuple
    origin: tuple

    @classmethod
    def from_geometry(cls, geometry):
        if geometry is None:
            return None
        mount = geometry.base_from_optical
        axes = ((mount[0, 2], -mount[0, 0]), (mount[1, 2], -mount[1, 0]))
        front_offset = geometry.half_length if geometry.front == 0. else 0.
        return cls(axes, (mount[0, 3] + front_offset, mount[1, 3]))

    def grid_point(self, point, geometry):
        """Project a point in this retained camera frame into the body's grid."""
        forward, left = np.asarray(self.axes) @ point + self.origin
        if geometry.front == 0.:
            forward -= geometry.half_length
        return forward, -left


@dataclass(frozen=True)
class MeasuredPoint:
    """Last real range point, kept in its original optical frame for standoff."""

    point: tuple[float, float]
    frame: ObservationFrame | None
    capture_ms: int | None
    capture_pose: PlanarPose | None
    standoff_point: tuple[float, float] | None = None
    box: tuple | None = None
    received_ms: int | None = None

    @property
    def observation_ms(self):
        return self.received_ms if self.received_ms is not None else self.capture_ms


@dataclass(frozen=True)
class AssociationSnapshot:
    """Numerical evidence for an established track at one prediction time."""

    generation: int
    reference_ms: int
    mean: tuple
    cov: tuple
    name: str
    hue: str
    state: str
    since_obs_s: float
    range_known: bool
    front_required: bool
    front_reference: tuple | None
    front_age_s: float
    front_frame_id: int | None
    has_front_fix: bool
    source: float | None
    latest_source: float | None
    image_region: tuple | None
    frame: ObservationFrame | None


@dataclass(frozen=True)
class TargetMatch:
    generation: int
    source: float | None
    point: tuple[float, float] | None  # camera forward, left, when a surface was measured
    range_m: float | None
    image_region: tuple | None = None
    standoff_point: tuple[float, float] | None = None


def observation_source(detections):
    stamp = (detections or {}).get("timestamp")
    return stamp if type(stamp) in (int, float) and math.isfinite(stamp) else None


def _image_region(obj):
    return (tuple(obj.get("position") or ()), tuple(obj.get("bbox_norm") or ()))


@dataclass
class Track:
    """One followed object. `mean` is [x, y, vx, vy] in the current body frame."""

    mean: np.ndarray
    cov: np.ndarray
    hue: str = ""
    name: str = ""
    state: str = CONFIRMED
    hits: int = 0
    age_s: float = 0.0
    # The last real observation and how long ago it was, for the re-update on
    # re-acquisition. It is transported into the current body frame at each
    # prediction so the difference between observations excludes ego motion.
    last_obs: np.ndarray | None = None
    since_obs_s: float = 0.0
    # Whether depth has ever provided a range for steering, including the
    # summary-only band estimate. A bearing-only track has a placeholder range.
    range_known: bool = False
    # Evidence applied on the current step. The range is the target's depth
    # map sample before filtering, present only on the first use of that frame.
    accepted_detection: bool = False
    measured_range_m: float | None = None
    measured_point: MeasuredPoint | None = None

    # The goal uses the associated box measurement. Keep its nearest coherent
    # surface independently for physical standoff checks.
    front_required: bool = False
    front_fix: np.ndarray | None = None
    front_reference: np.ndarray | None = None  # last trusted fix, even after expiry
    front_box: tuple | None = None
    front_frame_id: int | None = None
    front_age_s: float = 0.

    generation: int = 0
    source: float | None = None
    latest_source: float | None = None  # remains ordered across an unstamped observation
    image_region: tuple | None = None
    frame: ObservationFrame | None = None

    def record_range(self, range_m, point, depth, *, standoff_point=None):
        self.measured_range_m = range_m
        self.measured_point = MeasuredPoint(
            tuple(point), self.frame, depth.get("capture_ms") or depth.get("frame_id") or depth.get("control_ms"),
            depth.get("obstacle_capture_pose"), standoff_point, self.front_box,
            depth.get("frame_id") if depth.get("receipt_timed") else None)

    def snapshot(self, reference_ms):
        return AssociationSnapshot(
            self.generation, reference_ms, tuple(self.mean), tuple(map(tuple, self.cov)),
            self.name, self.hue, self.state, self.since_obs_s, self.range_known, self.front_required,
            tuple(self.front_reference) if self.front_reference is not None else None,
            self.front_age_s, self.front_frame_id, self.front_fix is not None, self.source,
            self.latest_source, self.image_region, self.frame)

    @property
    def drivable(self) -> bool:
        """Whether this track can serve as a navigation target."""
        return self.state in (CONFIRMED, COASTING)

    @property
    def coast_s(self) -> float:
        return self.since_obs_s if self.state == COASTING else 0.0

    @property
    def observed(self) -> bool:
        """A matched detection with no intervening miss."""
        return self.state == CONFIRMED

    @property
    def position(self) -> np.ndarray:
        return self.mean[:2]

    @property
    def navigation_range_known(self) -> bool:
        return self.front_fix is not None if self.front_required else self.range_known

    @property
    def range_m(self) -> float:
        point = self.front_fix
        return polar_of(self.position if point is None else point)[0]

    @property
    def bearing_rad(self) -> float:
        point = self.front_fix
        return polar_of(self.position if point is None else point)[1]

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


def observation_transform(ego_motion, frame, previous):
    """Transport camera coordinates from the previous mount through the body to the current mount."""
    dx, dy, yaw = ego_motion
    rotation = _rotation(-yaw)
    delta = np.array([dx, dy])
    if frame is None:
        return rotation, delta
    previous = previous or frame
    axes = np.asarray(frame.axes)
    old_axes = np.asarray(previous.axes)
    inverse = np.linalg.pinv(axes)
    # p_new = A_new^-1 [R (A_old p_old + origin_old - d) - origin_new].
    # Express the pivot displacement in the original image axes so stationary
    # track points, front references, and the Kalman control share one transform.
    linear = np.eye(2) + inverse @ (rotation @ old_axes - axes)
    old_inverse = inverse if frame.axes == previous.axes else np.linalg.pinv(old_axes)
    translation = old_inverse @ (delta + rotation.T @ frame.origin - previous.origin)
    return linear, translation


def _prediction(mean, cov, rotation, translation, dt, config, ego_measured):
    # p' = R(p + v·dt − d),  v' = R·v
    transition = np.zeros((4, 4))
    transition[:2, :2] = rotation
    transition[:2, 2:] = rotation * dt
    transition[2:, 2:] = rotation
    control = np.concatenate([-rotation @ translation, np.zeros(2)])

    mean = transition @ mean + control
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
    return mean, transition @ cov @ transition.T + noise


def target_objects(detections, target, config):
    """Eligible detector objects in their original order, including sustain evidence."""
    from . import policy as policy_mod

    objects = (detections or {}).get("objects") or []
    wanted = (target or "").strip().lower()
    if not wanted:
        return

    for obj in objects:
        confidence = float(obj.get("confidence") or 0)
        if confidence < config.sustain_confidence:
            continue
        key = policy_mod.object_key(obj).lower()
        name = str(obj.get("name", "")).lower()
        if key == wanted or name == wanted or wanted in name:
            yield obj


def _candidates(detections, depth, target: str, config, position) -> list:
    """Detections that could be our target, each as a body-frame point.

    The name/key matching is the same as it always was; what is new is that
    a match is a *candidate*, not a decision.
    """
    from . import policy as policy_mod

    out = []
    for obj in target_objects(detections, target, config):
        bearing_rad = policy_mod._bearing(obj) * config.half_fov_rad
        range_m, surface = policy_mod._target_measurement(obj, depth, config)
        measured = range_m is not None
        if not measured:
            # No depth at this object — a hole in the map, or the summary
            # fallback with nothing in that band. Place it at the range we
            # already believe, which makes this a bearing-only update in
            # disguise: it can correct where the target is *pointing* and
            # must not be allowed to claim anything about how far.
            range_m = (polar_of(position)[0] if position is not None
                       else config.assumed_range_m)
        out.append({
            "obj": obj,
            "point": point_of(range_m, bearing_rad),
            "confidence": float(obj.get("confidence") or 0),
            "hue": policy_mod.hue_of(obj),
            "name": obj.get("name") or "",
            "range_measured": measured,
            "surface": surface,
            "range_m": range_m,
        })
    return out


def _associate(candidates, config, position, cov=None, since_obs_s=0., hue="", *, person_box=None):
    """Pick the candidate this track is looking at, or None.

    ByteTrack's two stages, collapsed to one track: high-confidence
    detections get first refusal; if none of them lands in the gate, the
    low-confidence ones are tried against the same gate. A dim detection
    that agrees with where we predicted the target to be is the target seen
    through something; a dim detection anywhere else is background.
    """
    if not candidates:
        return None

    if position is None:
        # Without a prior position to gate against, require high confidence.
        confident = [c for c in candidates
                     if c["confidence"] >= config.min_confidence]
        if not confident:
            return None
        return max(confident, key=lambda c: c["confidence"])

    if person_box and config.local_controller == "dwa" and hue in _HUE_WHEEL and len(candidates) > 1:
        index = _HUE_WHEEL.index(hue)
        similar_hues = {_HUE_WHEEL[(index + delta) % len(_HUE_WHEEL)] for delta in (-1, 0, 1)}
        similar = [c for c in candidates if c["confidence"] >= config.min_confidence
                   and c["hue"] in similar_hues and c["obj"].get("bbox_norm")]
        if similar and len(similar) < len(candidates):
            # A delayed depth image can sample the background inside the
            # current person's box. If the image still supports the original
            # identity, let a bad range coast instead of following a neighbour.
            predicted_range, predicted_bearing = polar_of(position)
            # Vertical extent gives continuity even while yaw sweeps both boxes
            # across the frame and the predicted bearing lags the detector.
            visual = min(similar, key=lambda c: (
                abs(c["obj"]["bbox_norm"][1] - person_box[1])
                + abs(c["obj"]["bbox_norm"][3] - person_box[3]),
                abs(polar_of(c["point"])[1] - predicted_bearing),
            ))
            visual_ray = dict(visual, range_m=predicted_range,
                              point=point_of(predicted_range, polar_of(visual["point"])[1]))
            if _gate_cost(position, cov, since_obs_s, hue, visual_ray, config) is not None:
                candidates = [c for c in similar if c["hue"] == visual["hue"]]

    high = [c for c in candidates if c["confidence"] >= config.min_confidence]
    low = [c for c in candidates if c["confidence"] < config.min_confidence]
    for pool in (high, low):
        best, best_cost = None, None
        for candidate in pool:
            cost = _gate_cost(position, cov, since_obs_s, hue, candidate, config)
            if cost is None:
                continue
            if best_cost is None or cost < best_cost:
                best, best_cost = candidate, cost
        if best is not None:
            return best
    return None


def _gate_cost(position, cov, since_obs_s, hue, candidate, config):
    """Mahalanobis distance² to the prediction, or None if outside the gate.

    The gate widens on its own while coasting, because the covariance
    grows — which is the right behaviour and needs no separate rule: the
    longer the target has been hidden, the further from the prediction we
    are willing to believe it reappeared.
    """
    residual = candidate["point"] - position
    std = _measurement_std(candidate["range_m"], config)
    innovation = cov[:2, :2] + np.eye(2) * std ** 2
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
    reach = (config.max_target_speed * max(since_obs_s, 1e-3)
             + 3.0 * std)
    if float(np.hypot(*residual)) > reach:
        return None
    # Colour is a tie-break, never a gate. `object_key` already excludes
    # brightness because it swings with the lighting; hue is steadier but
    # still flickers, and rejecting on it would drop a track for a cloud
    # passing the window.
    if hue and candidate["hue"] and candidate["hue"] != hue:
        cost += config.hue_mismatch_cost
    return cost


def _front_point(point, config, reference, has_fix, range_known, position, dt):
    if point is None:
        return

    range_m = polar_of(point)[0]
    if reference is not None:
        # A coherent foreground patch can lie inside the person's box.
        # Require continuity with the last surface or the associated body
        # position. Failed updates and expiry cannot widen the gate.
        tolerance = 3.0 * _measurement_std(range_m, config)
        reach = tolerance
        if has_fix:
            reach += config.max_target_speed * min(dt, config.max_obs_age_ms / 1000)
        if (math.hypot(*(point - reference)) > reach
                and (not range_known or math.hypot(*(point - position)) > tolerance)):
            return

    return point


def _correct(mean, cov, candidate, config):
    point = candidate["point"]
    std = _measurement_std(candidate["range_m"], config)
    if not candidate["range_measured"]:
        # Bearing-only. Inflating the measurement noise is what stops
        # this from asserting a range it never measured.
        std *= config.bearingless_std_factor
    measurement = np.zeros((2, 4))
    measurement[0, 0] = measurement[1, 1] = 1.0
    innovation = (measurement @ cov @ measurement.T
                  + np.eye(2) * std ** 2)
    gain = cov @ measurement.T @ np.linalg.inv(innovation)
    mean = mean + gain @ (point - mean[:2])
    cov = (np.eye(4) - gain @ measurement) @ cov

    return mean, cov


def match_observation(snapshot, detections, depth, *, ego_motion, elapsed_s, target, config,
                      original_receipt=None, allow_missing_surface=False):
    """Gate a current measurement against a copied track, without consuming evidence.

    ego_motion is the displacement (forward, left, yaw) of the observation
    body pivot since the snapshot, expressed in its original axes.
    """
    if (snapshot.state != CONFIRMED or not 0 <= elapsed_s <= config.max_coast_s - snapshot.since_obs_s
            or not depth or depth.get("map") is None or depth.get("capture_invalid")):
        return None
    frame = ObservationFrame.from_geometry(config.scan_geometry or config.local_geometry)
    rotation, translation = observation_transform(ego_motion, frame, snapshot.frame)
    mean, cov = _prediction(np.asarray(snapshot.mean), np.asarray(snapshot.cov), rotation, translation,
                            elapsed_s, config, True)
    candidates = _candidates(detections, depth, target, config, mean[:2])
    source = observation_source(detections)
    if source is None:
        # A fixed-window caller can identify an unstamped detector receipt
        # only while this generation has never established a producer floor.
        if original_receipt is None or snapshot.latest_source is not None:
            return None
        original_frame = original_receipt
    else:
        if snapshot.latest_source is not None and source < snapshot.latest_source:
            return None
        original_frame = source == snapshot.source
    if original_frame and snapshot.image_region is not None:
        # A reused detector frame still refers to its original image region.
        candidates = [c for c in candidates if _image_region(c["obj"]) == snapshot.image_region]
    person_box = snapshot.image_region[1] if snapshot.front_required and snapshot.image_region else None
    candidate = _associate(candidates, config, mean[:2], cov, snapshot.since_obs_s + elapsed_s, snapshot.hue,
                           person_box=person_box)
    if candidate is None or not candidate["range_measured"]:
        return None
    point = candidate["point"]
    box = candidate["obj"].get("bbox_norm")
    if snapshot.front_required or (candidate["name"].lower() == "person" and box and config.depth_intrinsics):
        frame = depth.get("source_frame_id") or depth.get("frame_id")
        if (not box or config.depth_intrinsics is None or type(frame) is not int or frame <= 0
                or (snapshot.front_frame_id is not None and frame < snapshot.front_frame_id)):
            return None
        reference = (rotation @ (np.asarray(snapshot.front_reference) - translation)
                     if snapshot.front_reference is not None else None)
        corrected, _ = _correct(mean, cov, candidate, config)
        front_limit = config.depth_age_limit_ms if config.local_planning else config.max_obs_age_ms
        has_fix = snapshot.has_front_fix and snapshot.front_age_s + elapsed_s <= front_limit / 1000
        surface = candidate["surface"]
        if candidate["name"].lower() != "person":
            from .depth import front_surface
            surface = front_surface(depth["map"], box, config.depth_intrinsics, depth.get("ground"))
        surface = _front_point(surface, config, reference, has_fix, True, corrected[:2], elapsed_s)
        if surface is None:
            return (TargetMatch(snapshot.generation, source, None, None, _image_region(candidate["obj"]))
                    if allow_missing_surface else None)
        return TargetMatch(snapshot.generation, source, tuple(point), candidate["range_m"],
                           standoff_point=tuple(surface))
    return TargetMatch(snapshot.generation, source, tuple(point), polar_of(point)[0])


class Tracker:
    """Holds the single track and the whole lifecycle. One instance per target."""

    def __init__(self):
        self.track: Track | None = None
        self._generation = 0
        self.matched_detections = None
        self.reentry_visible = False
        # Keep recent stamps across blind ticks and out-of-order delivery.
        self._detection_stamps = deque(maxlen=64)
        self._unmatched_recovery = None
        self._range_frames = deque(maxlen=64)
        # Expose why a target was lost or recovered in info().
        self.last_reason = ""

    # ── the tick ─────────────────────────────────────────────────────────────

    def step(self, *, detections, depth, target: str, config, ego, dt: float,
             ego_measured: bool = True, receipt_ms: int = 0, retain_identity: bool = False):
        """Predict, associate, update. Returns the track if it may drive, else None.

        `ego` is our own twist as (vx, vy, wz) in the body frame. `ego_measured`
        says whether that came from odometry or from what we last *commanded* —
        see `_predict`, which treats the two differently and must.
        `detections` may be None (no fresh payload), which counts as a miss
        rather than as nothing happening.
        """
        self.reentry_visible = False
        expired = False
        if self.track is not None:
            self._predict(ego, dt, config, ego_measured)
            track = self.track
            track.age_s += dt
            track.since_obs_s += dt
            track.accepted_detection = False
            track.measured_range_m = None
            track.front_age_s += dt
            front_limit = config.depth_age_limit_ms if config.local_planning else config.max_obs_age_ms
            if track.front_age_s > front_limit / 1000:
                track.front_fix = None
            # Expiry always ends coasting. A committed detour may still use
            # the original position to associate fresh observations.
            if track.since_obs_s > config.max_coast_s:
                if retain_identity:
                    track.state = LOST
                    track.front_fix = None
                    track.mean[2:] = 0.
                else:
                    self.track = None
                expired = True
                self.last_reason = (f"目标消失超过 {config.max_coast_s:.1f}s，"
                                    "外推已不可信，放弃轨迹")

        # Apply each detector sample to the filter once. Prediction and a
        # new depth measurement can still advance between detector frames.
        stamp = (detections or {}).get("timestamp")
        if stamp is None and detections is not None and receipt_ms:
            stamp = ("receipt", receipt_ms)
        depth_frame = None
        if depth is not None and depth.get("map") is not None and not depth.get("capture_invalid"):
            depth_frame = depth.get("source_frame_id") or depth.get("frame_id")
        if stamp is not None:
            # A late, previously unseen detector frame cannot refresh a DWA
            # motion origin. Its producer clock is used only for ordering.
            older = (config.local_controller == "dwa" and self.track is not None
                     and isinstance(stamp, (int, float)) and self.track.latest_source is not None
                     and stamp < self.track.latest_source)
            repeated = stamp in self._detection_stamps
            last = repeated and stamp == self._detection_stamps[-1]
            pending = self._unmatched_recovery
            # A rejected detector sample can be associated once a newer depth frame arrives.
            retry = (last and expired and self.track is not None and pending is not None
                     and pending[0] == stamp and type(depth_frame) is int
                     and depth_frame > (pending[1] or 0))
            if older or (repeated and not retry):
                if last and self.track is not None:
                    self._update_front(depth, config, dt)
                return self.track if (self.track and self.track.drivable) else None
            if not repeated:
                self._detection_stamps.append(stamp)

        track = self.track
        position = track.position if track is not None else None
        if expired and track is not None and target == "person" and track.front_reference is not None:
            # Keep the measured front, transported by ego motion, as the
            # association reference after the velocity prediction expires.
            position = track.front_reference
        candidates = _candidates(detections, depth, target, config, position)
        all_candidates = candidates
        pending_recovery = bool(expired and track is not None and target == "person" and receipt_ms and candidates)
        association_age = track.since_obs_s if track is not None else 0.
        if expired and track is not None:
            # A fresh metric match can account for one observation window
            # of motion, even if the original target was hidden much longer.
            # Start from the predicted re-entry edge when the target is out
            # of view. Nearby boxes across the image can pass the metric gate.
            half_fov = config.half_fov_rad
            reentry = min(half_fov, max(-half_fov, polar_of(position)[1]))
            candidates = [candidate for candidate in candidates if candidate["range_measured"]
                          and abs(polar_of(candidate["point"])[1] - reentry) <= half_fov]
            association_age = min(association_age, config.max_obs_age_ms / 1000)
        matched = (_associate(candidates, config, position, track.cov, association_age, track.hue,
                              person_box=track.front_box)
                   if track is not None else _associate(candidates, config, None))
        if matched is None and pending_recovery and len(all_candidates) == 1 and not repeated:
            candidate = all_candidates[0]
            if (candidate["confidence"] >= config.min_confidence
                    and candidate["hue"] and candidate["hue"] == track.hue
                    and candidate["obj"].get("bbox_norm") and track.front_box):
                prior_range, _ = polar_of(position)
                beam = dict(candidate, range_m=prior_range,
                            point=point_of(prior_range, polar_of(candidate["point"])[1]))
                self.reentry_visible = _gate_cost(
                    position, track.cov, association_age, track.hue, beam, config) is not None
        if stamp is not None:
            self._unmatched_recovery = (stamp, depth_frame) if matched is None and pending_recovery else None

        self.matched_detections = detections if matched is not None else None
        if matched is not None:
            self._hit(matched, config)
            if self.track.accepted_detection:
                self.track.source = observation_source(detections)
                if self.track.source is not None:
                    self.track.latest_source = self.track.source
                self.track.image_region = _image_region(matched["obj"])
                box = matched["obj"].get("bbox_norm")
                if matched["name"].lower() == "person" and box and config.depth_intrinsics is not None:
                    self.track.front_required = True
                self.track.front_box = tuple(box) if self.track.front_required and box else None
                measurement = (matched["range_m"], matched["surface"]) if matched["name"].lower() == "person" else None
                self._update_front(depth, config, dt, measurement)
            if (self.track.accepted_detection and not self.track.front_required and matched["range_measured"]
                    and depth is not None and depth.get("map") is not None
                    and not depth.get("capture_invalid")):
                depth_id = depth.get("source_frame_id", depth.get("frame_id"))
                if depth_id is None or depth_id not in self._range_frames:
                    self.track.record_range(matched["range_m"], matched["point"], depth)
                    if depth_id is not None:
                        self._range_frames.append(depth_id)
        elif not expired:
            self._miss(candidates)
            if self.track is not None:
                self._update_front(depth, config, dt)

        return self.track if (self.track and self.track.drivable) else None

    def _update_front(self, depth, config, dt, measurement=None):
        """Measure the associated person in the newest usable depth image."""
        track = self.track
        box = track.front_box
        intrinsics = config.depth_intrinsics
        if (not box or intrinsics is None or depth is None or depth.get("map") is None
                or depth.get("capture_invalid")):
            return
        frame = depth.get("source_frame_id") or depth.get("frame_id")
        if type(frame) is not int or frame <= 0 or (
                track.front_frame_id is not None and frame <= track.front_frame_id):
            return
        track.front_frame_id = frame
        stamp = observation_ms(depth)
        now = depth.get("control_ms")
        depth_age_ms = now - stamp if type(now) is int and type(stamp) is int else 0
        coasting = not track.observed
        if coasting and (track.front_fix is None or track.front_reference is None
                         or track.since_obs_s * 1000 >= config.max_obs_age_ms
                         or not 0 <= depth_age_ms < min(config.depth_age_limit_ms, config.max_obs_age_ms)):
            return
        if measurement is None:
            from .depth import person_box_measurement
            measurement = person_box_measurement(depth["map"], box, intrinsics, depth.get("ground"))
        range_m, surface = measurement
        point = _front_point(surface, config, track.front_reference, track.front_fix is not None,
                             track.range_known, track.position, dt)
        if point is None:
            return
        if coasting:
            # The remembered crop can contain a new foreground object. Both
            # the box range and the coherent front must agree with the
            # transported target before the new depth can verify arrival.
            tolerance = config.stop_tolerance_m
            if (abs(range_m - track.range_m) > tolerance
                    or math.dist(point, track.front_reference) > tolerance):
                return
            bearing = track.bearing_rad
        else:
            bearing = (track.image_region[0] or (0.,))[0] * config.half_fov_rad
        track.front_fix = point_of(range_m, bearing)
        track.front_reference = point
        # Frame reception and processing may already have used part of
        # the depth budget. Reading this frame again cannot renew it.
        track.front_age_s = max(0, depth_age_ms) / 1000
        track.record_range(range_m, track.front_fix, depth, standoff_point=tuple(point))

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
        geometry = (config.scan_geometry or config.local_geometry) if config.local_controller == "dwa" else None
        frame = ObservationFrame.from_geometry(geometry)
        rotation, translation = observation_transform((vx * dt, vy * dt, wz * dt), frame, track.frame)
        track.frame = frame

        track.mean, track.cov = _prediction(track.mean, track.cov, rotation, translation,
                                            dt, config, ego_measured)
        if track.last_obs is not None:
            track.last_obs = rotation @ (track.last_obs - translation)
        if track.front_reference is not None:
            track.front_reference = rotation @ (track.front_reference - translation)
            if track.front_fix is not None:
                track.front_fix = rotation @ (track.front_fix - translation)

    # ── 2. associate ─────────────────────────────────────────────────────────

    def _gate_cost(self, candidate, config):
        track = self.track
        return _gate_cost(track.position, track.cov, track.since_obs_s, track.hue, candidate, config)

    # ── 3. update ────────────────────────────────────────────────────────────

    def _hit(self, candidate, config):
        point = candidate["point"]
        if self.track is None:
            self._create(candidate, config)
            self.last_reason = f"已关联目标 {candidate['name']!r}"
            return

        track = self.track
        was_coasting = track.state in (COASTING, LOST)

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
            track.mean, track.cov = _correct(track.mean, track.cov, candidate, config)

        _clamp_speed(track, config)
        track.hue = candidate["hue"] or track.hue
        track.range_known = track.range_known or candidate["range_measured"]
        track.hits += 1
        track.accepted_detection = True
        track.last_obs = point.copy()
        track.since_obs_s = 0.0

        track.state = CONFIRMED
        self.last_reason = "目标重新出现，轨迹恢复" if was_coasting else ""

    def _create(self, candidate, config):
        self._generation += 1
        self.track = Track(
            generation=self._generation,
            frame=(ObservationFrame.from_geometry(config.scan_geometry or config.local_geometry)
                   if config.local_controller == "dwa" else None),
            mean=np.concatenate([candidate["point"], np.zeros(2)]),
            cov=self._initial_cov(candidate, config),
            hue=candidate["hue"],
            name=candidate["name"],
            hits=1,
            range_known=candidate["range_measured"],
            accepted_detection=True,
            last_obs=candidate["point"].copy(),
        )

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

    def _miss(self, candidates):
        track = self.track
        if track is None:
            self.last_reason = ("视野里没有匹配目标的检测" if not candidates
                                else "有疑似检测，但置信度不足以建立轨迹")
            return

        track.state = COASTING
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
            "name": track.name,
            "hits": track.hits,
            "coast_s": round(track.coast_s, 2),
            "age_s": round(track.age_s, 1),
            "observed": track.observed,
            "range_m": round(track.range_m, 2) if track.navigation_range_known else None,
            "position_std_m": round(track.position_std_m, 2),
            "reason": self.last_reason,
        }
