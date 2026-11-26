"""Bounded dynamic-window tracking of local route references.

All positions are robot-relative (forward, right) metres. Twist follows ROS:
(vx forward, vy left, wz counter-clockwise).  A* owns topology and the goal;
this module only selects a short executable command. No ROS or robot IDs.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from struct import pack
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import product
from typing import Literal

import numpy as np

from . import local_path
from .motion import (FootprintSpec, MotionCheck, MotionTrace, Pose, Twist, _DEFAULT_ACCELERATION, _DEFAULT_BRAKING,
                     evaluate_motion, minimum_target_distance, predict_motion, stopping_horizon)


MAX_SEARCH_CANDIDATES = 64
ROUTE_SPEED_SCALES = (1., .75, .5, .25)
PREFERRED_CLEARANCE_M = .30


@dataclass(frozen=True)
class CandidateCheck:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class CandidateScore:
    path: float
    progress: float
    obstacle: float
    smooth: float
    heading: float
    goal: float
    route_change: float = 0.
    route_clearance: float = 0.

    @property
    def route_cost(self):
        return self.route_change + self.route_clearance

    @property
    def total(self):
        return self.path + self.progress + self.obstacle + self.smooth + self.heading + self.goal


@dataclass(frozen=True)
class Choice:
    command: Twist
    trajectory: tuple[Pose, ...]
    score: float
    progress: float
    clearance: float
    trace: MotionTrace | None = None
    check: MotionCheck | None = None

    components: CandidateScore | None = None
    reference_index: int = 0


def _obstacle_key(layer):
    if isinstance(layer, tuple):
        return tuple(map(_obstacle_key, layer))
    if layer is None:
        return None
    if isinstance(layer, local_path.ObstacleSquares):
        return layer.yaw, layer.alignment_error, _array_key(layer.centers)
    return _array_key(layer)


def _array_key(array):
    return array.shape, array.dtype.str, array.tobytes()


def check_measured_standoff(trace, checked, measured_target, stop_distance):
    """Apply the measured target's spacing to the same complete physical trace."""
    if checked.accepted and measured_target is not None:
        point, registration_error = measured_target
        if minimum_target_distance(trace, point) < stop_distance + registration_error - 1e-10:
            return replace(checked, accepted=False, reason="target_standoff")
    return checked


@dataclass(frozen=True)
class MotionContext:
    """The exact inputs to a prediction and its complete-stop check."""

    measured: Twist
    previous: Twist
    horizon: float
    latency: float
    geometry: local_path.Geometry
    footprint: FootprintSpec
    occupied: object
    remembered: object
    supported_region: tuple | None
    target: tuple | None
    stop_distance: float
    acceleration: Twist = _DEFAULT_ACCELERATION
    braking: Twist = _DEFAULT_BRAKING
    rotation_center: tuple | None = None
    terminal_boundary: object = None
    frame: tuple = ()
    measured_target: tuple | None = None

    def key(self):
        g = self.geometry
        mount = _array_key(g.base_from_optical) if g.base_from_optical is not None else None
        geometry = (g.horizontal, g.vertical, g.front, g.rear, g.height, g.free_space_validated,
                    g.min_obstacle_height, mount)
        return (tuple(self.measured), tuple(self.previous), self.horizon, self.latency, geometry,
                self.footprint, _obstacle_key(self.occupied), _obstacle_key(self.remembered),
                tuple(map(tuple, self.supported_region)) if self.supported_region is not None else None,
                tuple(self.target) if self.target is not None else None, self.stop_distance,
                tuple(self.acceleration), tuple(self.braking),
                tuple(self.rotation_center) if self.rotation_center is not None else None,
                self.terminal_boundary, self.frame, self.measured_target)

    @property
    def response_key(self):
        """Kinematics only. The selector always asks for the default step and a complete stop."""
        g = self.geometry
        center = self.rotation_center
        if center is not None and (type(center) is not tuple or len(center) != 2):
            return None
        center = center if center is not None else ()
        values = (*self.measured, *self.previous, self.horizon, self.latency,
                  *self.acceleration, *self.braking, g.front, g.rear, *center)
        # Preserve signed zero and scalar precision. Unusual numeric inputs
        # still use the original predictor and its validation directly.
        if not all(type(x) in (float, np.float64) and math.isfinite(x) for x in values):
            return None
        return (self.footprint.mode, self.rotation_center is None, tuple(map(type, values)),
                pack(f"!{len(values)}d", *values),
                _array_key(g.base_from_optical) if g.base_from_optical is not None else None)

    @cached_property
    def prepared_layers(self):
        return local_path.PreparedObstacleLayers(self.occupied, self.remembered)

    def predict(self, command):
        return predict_motion(
            self.measured, self.previous, command, hold_s=self.horizon, latency_s=self.latency,
            acceleration=self.acceleration, braking=self.braking, geometry=self.geometry,
            mode=self.footprint.mode, rotation_center=self.rotation_center)

    def evaluate(self, prediction):
        checked = evaluate_motion(
            prediction, footprint=self.footprint, occupied=self.occupied, remembered=self.remembered,
            supported_region=self.supported_region, original_goal=self.target,
            stop_distance_m=self.stop_distance, terminal_boundary=self.terminal_boundary,
            prepared_layers=self.prepared_layers)
        return check_measured_standoff(prediction, checked, self.measured_target, self.stop_distance)

    def check(self, command):
        prediction = self.predict(command)
        return prediction, self.evaluate(prediction)


class PredictionCache:
    """Bounded rigid-body responses. Physical and visibility checks stay decision-local."""

    def __init__(self):
        self.traces = OrderedDict()

    def get(self, context, command):
        signature = context.response_key
        if signature is None or not all(type(x) in (float, np.float64) and math.isfinite(x) for x in command):
            return context.predict(command), False
        key = signature, tuple(map(type, command)), pack("!3d", *command)
        cached = self.traces.pop(key, None)
        if cached is not None:
            self.traces[key] = cached
            return cached, True
        trace = context.predict(command)
        self.traces[key] = trace
        if len(self.traces) > MAX_SEARCH_CANDIDATES:
            self.traces.popitem(last=False)
        return trace, False


@dataclass
class CandidateBudget:
    """Decision-local complete-stop outcomes and unused search capacity."""

    remaining: int
    outcomes: dict = field(default_factory=dict)
    predictions: PredictionCache | None = None
    time_left: Callable[[], float] | None = None
    early_return_s: float = math.inf

    def for_context(self, context):
        return self.outcomes.setdefault(context.key(), {})

    def short_window(self):
        return self.time_left is not None and self.time_left() <= self.early_return_s

    def expired(self):
        return self.time_left is not None and self.time_left() <= 0.


def _predict(command, measured, *, horizon, dt, acceleration, braking, stop):
    return predict_motion(measured, command, command, hold_s=horizon, latency_s=0.,
                          dt=dt, acceleration=acceleration, braking=braking,
                          stop=stop, geometry=None, mode="body")


def _rollout(command, measured, *, horizon, dt, acceleration, braking, stop=False):
    return _predict(command, measured, horizon=horizon, dt=dt, acceleration=acceleration,
                    braking=braking, stop=stop).poses


def _collides(poses, obstacles, geometry, half_width, cell_size, margin=0.):
    clearance, hit = local_path.rectangle_clearance(
        poses, local_path.ObstacleSquares(obstacles, 0.), geometry.front, geometry.rear, half_width,
        cell_size=cell_size, padding=margin)
    return hit is not None, clearance


@dataclass(frozen=True)
class _RouteSegment:
    start: tuple[float, float]
    delta: tuple[float, float]
    length: float
    length_squared: float
    walked: float
    tangent: tuple[float, float]


def _route_segments(route):
    segments = []
    walked = 0.
    for (fx, fy), (tx, ty) in zip(route, route[1:]):
        dx, dy = tx - fx, ty - fy
        length = math.hypot(dx, dy)
        if length < 1e-8:
            continue
        tangent = dx / length, dy / length
        if segments and tangent == segments[-1].tangent:
            # A straight run of grid cells is one projection segment. Retain
            # the input waypoints for topology, visibility and publication.
            prior = segments.pop()
            fx, fy = prior.start
            dx, dy = tx - fx, ty - fy
            length = math.hypot(dx, dy)
            walked = prior.walked
        segments.append(_RouteSegment((fx, fy), (dx, dy), length, length * length,
                                      walked, tangent))
        walked += length
    return tuple(segments)


def _project_to_route(point, route, segments=None):
    """Return signed route position, distance to the polyline, and its nearest tangent."""
    at, closest, tangent = 0.0, math.inf, (1., 0.)
    x, y = point
    for segment in _route_segments(route) if segments is None else segments:
        fx, fy = segment.start
        dx, dy = segment.delta
        t = max(0., min(1., ((x-fx)*dx+(y-fy)*dy) / segment.length_squared))
        error = math.hypot(x-fx-t*dx, y-fy-t*dy)
        if error < closest:
            at, closest, tangent = segment.walked + t * segment.length, error, segment.tangent
    if at == 0. and math.isfinite(closest):
        # A front-camera route starts ahead of the chassis pivot. Extend the
        # route coordinate backwards along its first tangent, while measuring
        # cross-track distance to the original finite polyline.
        dx, dy = x - route[0][0], y - route[0][1]
        at = min(0., dx*tangent[0] + dy*tangent[1])
    return at, closest, tangent


def _valid_axis(value, lower, upper, minimum):
    return lower <= value <= upper and (value == 0. or abs(value) >= minimum)


def _axis_values(lower, upper, floor, *references):
    """Sample target commands within signed limits and the downstream deadband."""
    options = {0., floor, -floor, *map(float, references)}
    for factor in (.25, .5, .75, 1.):
        options.update((upper * factor, lower * factor))
    return sorted(v for v in options if _valid_axis(v, lower, upper, floor))


def _legal_command(cmd, limits, floors):
    vx, vy, wz = cmd
    xl, yl, wl = limits
    return (any(cmd) and vx >= 0. and _valid_axis(vx, *xl, floors[0])
            and _valid_axis(vy, *yl, floors[1])
            and _valid_axis(wz, *wl, floors[3] if vx or vy else floors[2]))


def _commands(previous, measured, preferred, limits, floors, extra, budget, *, recovery_speed=0.):
    xl, yl, wl = limits

    def legal(cmd):
        return _legal_command(cmd, limits, floors)

    # Keep a slow entry candidate when the declared upper limit widens the lattice.
    xs = _axis_values(*xl, floors[0], xl[1] / 8, previous[0], measured[0], *(p[0] for p in preferred))
    ys = _axis_values(*yl, floors[1], previous[1], measured[1], *(p[1] for p in preferred))
    yaw_refs = (previous[2], measured[2], *(p[2] for p in preferred))
    standing_yaws = _axis_values(*wl, floors[2], *yaw_refs)
    moving_yaws = _axis_values(*wl, floors[3], *yaw_refs)
    ws = sorted(set(standing_yaws).union(moving_yaws))
    if not xs or not ys or not ws:
        return (), []

    def route_seed(pref):
        vx = min(xs, key=lambda v: abs(v - pref[0]))
        vy = min(ys, key=lambda v: abs(v - pref[1]))
        yaws = moving_yaws if vx or vy else standing_yaws
        return vx, vy, min(yaws, key=lambda v: abs(v - pref[2]), default=0.)

    low = next((x for x in xs if x > 0.), 0.)
    families = [("low_straight", (low, 0., 0.))]
    for name, sign in (("left", 1), ("right", -1)):
        turns = [w for w in moving_yaws if sign * w > 0]
        if low and turns:
            families.extend(((name + "_arc", (low, 0., min(turns, key=abs))),
                             (name + "_turn", (low, 0., max(turns, key=abs)))))
        standing = [w for w in standing_yaws if sign * w > 0]
        if standing:
            families.append((name + "_standing_turn", (0., 0., min(standing, key=abs))))
        lateral = [y for y in ys if sign * y > 0]
        if lateral:
            families.append((name + "_lateral", (0., max(lateral, key=abs), 0.)))
    routes = [route_seed(pref) for pref in preferred]
    seeds = [("previous", previous), *(("route", cmd) for cmd in routes), *families]
    extra = [tuple(cmd) for cmd in extra if legal(cmd)]
    # Keep continuity and supplied motion first. The preferred route pair and
    # motion families get a turn before extra references fill the shortlist.
    ordered = dict.fromkeys(tuple(cmd) for cmd in
                            (previous, *extra[:1], *routes[:2], *(cmd for _, cmd in families),
                             *routes[2:], *extra) if legal(cmd))
    sampled = list(ordered)[:budget]
    slots = budget - len(sampled)
    if slots:
        pool = [cmd for cmd in product(xs, ys, ws) if legal(cmd) and cmd not in ordered]
        if len(pool) > slots:
            pool = [pool[i] for i in np.linspace(0, len(pool) - 1, slots, dtype=int)]
        # Substitute only the tail of the unseeded lattice. Adding a route
        # turn must not reshuffle the entire spread of lateral velocities.
        entry = dict.fromkeys((vx, 0., yaw) for _, _, yaw in routes[:2] if abs(yaw) >= .1
                              for vx in (0., low) if legal((vx, 0., yaw)))
        if 0. < recovery_speed < xl[1] / 8:
            # Use the unseeded tail for a single cell of travel in each legal axis.
            for vy in (0., recovery_speed, -recovery_speed):
                cmd = (recovery_speed, vy, 0.)
                if legal(cmd):
                    entry[cmd] = None
                    if len(pool) < slots and cmd not in ordered and cmd not in pool:
                        pool.append(cmd)
        missing_entry = [cmd for cmd in entry if cmd not in ordered and cmd not in pool]
        if missing_entry:
            for index in reversed(range(len(pool))):
                if pool[index] not in entry:
                    pool[index] = missing_entry.pop(0)
                    if not missing_entry:
                        break
        sampled.extend(pool)
    missed = [name for name, cmd in seeds if legal(cmd) and tuple(cmd) not in sampled]
    return sampled, missed


def _score_route(points, coverage):
    """Give each route the same scoring length using its final known tangent."""
    result = [points[0]]
    remaining = coverage
    tangent = None
    for start, end in zip(points, points[1:]):
        length = math.dist(start, end)
        if length < 1e-8:
            continue
        tangent = tuple((b - a) / length for a, b in zip(start, end))
        if length >= remaining:
            result.append(tuple(a + remaining * direction for a, direction in zip(start, tangent)))
            return tuple(result)
        result.append(end)
        remaining -= length
    if tangent is None:
        return None
    result.append(tuple(a + remaining * direction for a, direction in zip(result[-1], tangent)))
    return tuple(result)


def _preferred_command(points, nominal, horizon, yaw_limit, *, differential=False):
    dx, dy = next(((f, r) for f, r in points[1:] if math.hypot(f, r) >= .45), points[-1])
    if differential and dx > 0.:
        # The circular arc from the body pivot to the lookahead point has
        # curvature 2*y / distance^2. Right in the grid is negative yaw.
        yaw = -2. * nominal * dy / (dx*dx + dy*dy)
        return nominal, 0., min(yaw_limit[1], max(yaw_limit[0], yaw))
    angle = math.atan2(dy, dx)
    turn = min(abs(angle) / max(.4, horizon), abs(yaw_limit[0]) if angle > 0. else yaw_limit[1])
    return max(0., nominal * math.cos(angle)), -nominal * math.sin(angle), -math.copysign(turn, angle)


def remaining_route_length(points, *, segments=None):
    segments = _route_segments(points) if segments is None else segments
    if not segments:
        return 0.
    progress, _, _ = _project_to_route((0., 0.), points, segments)
    return segments[-1].walked + segments[-1].length - max(0., progress)


def stays_on_route(points, positions, half_width):
    segments = _route_segments(points)
    return bool(segments) and all(
        _project_to_route(position, points, segments)[1] <= max(.30, half_width) for position in positions)


def track_route(points, nominal, horizon, yaw_limit, *, body_origin, lateral, lookahead_m=None, yaw_floor=0.):
    """Track the finite visible polyline in the current forward/right grid."""
    segments = _route_segments(points)
    remaining = remaining_route_length(points, segments=segments)
    if remaining <= 1e-8:
        return None
    end = segments[-1].walked + segments[-1].length
    if lookahead_m is None:
        lookahead_m = max(.45, nominal * horizon)
    lookahead = min(end, end - remaining + lookahead_m)
    segment = next((s for s in segments if s.walked + s.length >= lookahead), segments[-1])
    f, r = (start + (lookahead - segment.walked) * axis - pivot
            for start, axis, pivot in zip(segment.start, segment.tangent, body_origin))
    if f <= 0. and (not lateral or abs(r) <= abs(f)):
        return None
    speed = min(nominal, remaining / horizon)
    if not lateral and r:
        # Preserve the requested curvature at a tight bend when yaw saturates.
        yaw_cap = -yaw_limit[0] if r > 0. else yaw_limit[1]
        speed = min(speed, yaw_cap * (f * f + r * r) / (2. * abs(r)))
    vx, vy, wz = _preferred_command(((0., 0.), (f, r)), speed, horizon, yaw_limit, differential=not lateral)
    if 0. < abs(wz) < yaw_floor:
        # A lateral offset resolved by the grid needs steering, even at low speed.
        minimum = math.copysign(yaw_floor, wz)
        turn = abs(r) >= local_path.RESOLUTION / 2 or abs(wz) >= yaw_floor / 2
        wz = minimum if turn and yaw_limit[0] <= minimum <= yaw_limit[1] else 0.
    return (vx, vy, wz), remaining


def _obstacle_score(check):
    # Only rank sweeps that already passed the hard footprint checks.
    return 1.5 * max(0., PREFERRED_CLEARANCE_M - check.min_clearance_m) / PREFERRED_CLEARANCE_M


def pursuit_cost(command, previous, nominal_speed, yaw_limit, check):
    """Rank checked arcs by continuity, progress, and speed near an obstacle."""
    speed_scale = max(.1, nominal_speed)
    speed_fraction = min(1., math.hypot(*command[:2]) / speed_scale)
    change = .6 * math.dist(command[:2], previous[:2]) / speed_scale
    change += .75 * abs(command[2] - previous[2]) / max(.2, *map(abs, yaw_limit))
    return change + .75 * (1. - speed_fraction) + _obstacle_score(check) * (1. + speed_fraction)


def score_candidate(command, trace, check, reference, previous, *, half_width, limits,
                    progress_scale, target, segments=None, initial_position=None, guidance=False):
    """Score the held part of a physically admissible trace on one reference."""
    poses = trace.camera_poses if trace.mode == "camera_approx" else trace.body_poses
    if segments is None:
        segments = _route_segments(reference)
    end_pose = poses[trace.moving_count - 1]
    end_position, error, tangent = _project_to_route(end_pose[:2], reference, segments)
    start_body, end_body = trace.body_poses[0], trace.body_poses[trace.moving_count - 1]
    if trace.mode == "camera_approx":
        end_position = _project_to_route(end_body[:2], reference, segments)[0]
    if initial_position is None:
        initial_position = _project_to_route(start_body[:2], reference, segments)[0]
    progress = end_position - initial_position
    desired_heading = math.atan2(tangent[1], tangent[0])
    if guidance and end_position < segments[-1].walked + segments[-1].length - local_path.RESOLUTION:
        # Aim back onto the route. Its tangent alone can reward driving
        # parallel to a shifted doorway approach with the body off-centre.
        desired_heading = math.atan2(reference[-1][1] - end_pose[1], reference[-1][0] - end_pose[0])
    heading_error = abs(math.remainder(end_pose[2] - desired_heading, math.tau))
    initial_error = abs(math.remainder(poses[0][2] - desired_heading, math.tau))
    turns_toward_route = command[0] == command[1] == 0. and heading_error < initial_error - 1e-10
    # A creeping entry arc may gain heading before gaining half a grid cell.
    # It still has to stay on the route and pass the complete braking sweep.
    entering_turn = (command[0] > 0. and command[2] != 0. and progress > 0.
                     and math.dist(start_body[:2], end_body[:2]) <= local_path.RESOLUTION
                     and initial_error - heading_error >= .05)
    if progress < .025 and not (turns_toward_route or entering_turn):
        return None, progress, "insufficient_progress"
    errors = [_project_to_route(p[:2], reference, segments)[1] for p in poses[:trace.moving_count:4]]
    # A front camera traces an arc while the chassis turns toward the route.
    # Keep that deviation as a score; the complete footprint still checks the turn.
    if not turns_toward_route and (error > max(.24, .75 * half_width)
                                  or max(errors, default=0.) > max(.30, half_width)):
        return None, progress, "off_route"

    path = 3.0 * (sum(errors) / len(errors)) / max(.15, half_width)
    along_route = -2.0 * progress / progress_scale
    obstacle = 0. if check is None else _obstacle_score(check)
    x_limit, y_limit, yaw_limit = limits
    smooth = .6 * (abs(command[0] - previous[0]) / max(.1, x_limit[1])
                   + abs(command[1] - previous[1]) / max(.1, max(abs(y_limit[0]), y_limit[1])))
    smooth += .75 * abs(command[2] - previous[2]) / max(.2, max(abs(yaw_limit[0]), yaw_limit[1]))
    heading = .45 * heading_error / max(.15, half_width) if guidance else .5 * heading_error / math.pi
    goal = 0.
    if target is not None:
        initial_range = math.dist(start_body[:2], target)
        end_range = math.dist(end_body[:2], target)
        goal = .10 * (end_range - initial_range) / max(1., initial_range)
    return CandidateScore(path, along_route, obstacle, smooth, heading, goal), progress, None


def select(*, route, grid, geometry, half_width=None, measured, previous,
           x_limit, y_limit, yaw_limit, floors, target=None, stop_distance=0.,
           horizon=1.2, max_candidates=MAX_SEARCH_CANDIDATES, allow_yaw=True,
           acceleration=_DEFAULT_ACCELERATION, braking=_DEFAULT_BRAKING,
           candidate_commands: tuple[Twist, ...] = (), straight_handoff_limit=None,
           validate_candidate: Callable[[Twist, tuple[Pose, ...], float], CandidateCheck] | None = None,
           footprint_mode: Literal["body", "camera_approx"] = "body",
           rotation_center: tuple[float, float] | None = None, diagnostics: dict | None = None,
           footprint: FootprintSpec | None = None, latency_s=0., remembered=None,
           supported_region=None, terminal_boundary=None, occupied=None, alternate_routes=(),
           candidate_budget: CandidateBudget | None = None, motion_frame=(), measured_target=None,
           guidance=False, follow_published=False, route_error=0., handoff_yaw_step=0., retry_from_rest=False) -> Choice | None:
    """Select a legal target velocity whose complete predicted stop is admissible.

    Routes, occupied cells and targets share the current grid frame. Each
    alternate uses the same physical context and optional full-stop validator.
    The validator receives the mode's reference poses and full stop time.
    Older callers pass a laterally padded half_width. A physical FootprintSpec
    carries its own single margin and takes precedence over that legacy width.
    """
    diag = diagnostics if diagnostics is not None else {}
    rejected = {}
    diag.update(sampled_candidates=0, admissible_candidates=0, physical_evaluations=0, bound_skips=0, response_reuse=0, rejected=rejected,
                reason="invalid_inputs", uncovered_families=[], considered_candidates=0,
                cached_candidates=0, reference_scores=0, scores_by_reference=[])
    if (route is None or len(route) < 2 or grid is None or geometry is None
            or measured is None or max_candidates < 1 or not 0.1 <= horizon <= 3.
            or not 0. <= latency_s <= horizon or len(measured) != 3 or len(previous) != 3
            or not math.isfinite(stop_distance) or stop_distance < 0.
            or footprint_mode not in ("body", "camera_approx")):
        return None
    if not all(math.isfinite(float(v)) for v in (*measured, *previous, *acceleration, *braking,
                                                  *x_limit, *y_limit, *yaw_limit, *floors, *(rotation_center or ()))):
        return None
    if min(*acceleration, *braking) <= 0.:
        return None
    points = tuple((float(f), float(r)) for f, r in route)
    if target is not None and not all(map(math.isfinite, target)):
        return None
    routes = (points, *(tuple((float(f), float(r)) for f, r in alt) for alt in alternate_routes))
    if not all(len(ref) >= 2 and all(math.isfinite(f) and math.isfinite(r) for f, r in ref)
               for ref in routes):
        return None
    legacy_camera = footprint is None and footprint_mode == "camera_approx"
    if footprint is None:
        if half_width is None:
            return None
        if footprint_mode == "camera_approx" and validate_candidate is None:
            diag["reason"] = "missing_candidate_validator"
            return None
        footprint = FootprintSpec(geometry.front, geometry.rear, half_width, 0., footprint_mode)
    else:
        footprint_mode = footprint.mode
        half_width = footprint.half_width + footprint.clearance_margin_m
    if footprint_mode == "body" and target is not None:
        mount = geometry.base_from_optical
        if mount is None or not all(map(math.isfinite, (mount[0, 3], mount[1, 3]))):
            diag["reason"] = "invalid_camera_offset"
            return None
    if not allow_yaw:
        yaw_limit = (0., 0.)
    limits = (x_limit, y_limit, yaw_limit)
    x_speed, y_speed = x_limit[1], max(abs(y_limit[0]), y_limit[1])
    # Include residual/previous motion in the held-distance window. Both
    # references see the same window and score normalization.
    speed = max(math.hypot(x_speed, y_speed), math.hypot(*measured[:2]), math.hypot(*previous[:2]))
    coverage = max(.45, speed * horizon)
    if guidance:
        # Include the body length so bends are visible before the leading corners enter them.
        coverage += geometry.front + geometry.rear
    references = tuple(_score_route(ref, coverage) for ref in routes)
    if any(ref is None for ref in references):
        return None
    translation_speed = math.hypot(x_speed, y_speed)
    progress_scale = max(.2, min(1.0, translation_speed * horizon))
    route_costs = (0.,) if len(routes) == 1 else tuple(
        local_path.route_velocity_change(ref, *limits, previous) for ref in routes)
    route_obstacles = local_path.occupied_squares(grid[1]) if len(routes) > 1 and not follow_published else None
    clearance_costs = tuple(local_path.route_clearance_cost(ref, route_obstacles, geometry, half_width) for ref in routes)
    route_cost_sum = tuple(change + clearance for change, clearance in zip(route_costs, clearance_costs))
    nominal = min(max(x_speed, y_speed), .35)
    preferred = tuple(_preferred_command(ref, nominal, horizon, yaw_limit,
                                         differential=guidance and not y_speed and footprint_mode == "body")
                      for ref in references)

    budget = candidate_budget if candidate_budget is not None else CandidateBudget(max_candidates)
    if budget.expired():
        diag["reason"] = "search_deadline"
        return None
    slots = min(budget.remaining, max_candidates)
    # Only reserve work for an abrupt release of an ongoing turn. Smaller
    # tracking corrections already use the regular route score and Driver ramp.
    transition_slot = bool(slots > 1 and handoff_yaw_step > 0. and any(previous[:2])
                           and abs(previous[2]) > 2 * handoff_yaw_step)
    entry_slot = (slots > 1 and guidance and not follow_published and target is not None
                  and any(yaw_limit) and any(abs(r - ref[0][1]) >= local_path.RESOLUTION / 2
                                             for ref in routes for _, r in ref[1:]))
    slots -= int(transition_slot or entry_slot)
    obstacles = local_path.occupied_squares(grid[1]) if occupied is None else occupied
    context = MotionContext(measured, previous, horizon, latency_s, geometry, footprint, obstacles, remembered,
                            supported_region, target, stop_distance, acceleration, braking, rotation_center,
                            terminal_boundary, motion_frame, measured_target)

    def interpolate(choice, room=math.inf):
        if not (transition_slot or entry_slot) or choice is None or not any(choice.command[:2]) or budget.expired():
            return choice
        vx, vy, wz = choice.command
        delta = wz - previous[2]
        entry_yaw = 0.
        if entry_slot and vx > 0. and vy == 0.:
            # Track the selected reference at the incoming forward speed.
            # The first packet can turn without reopening the route decision.
            body_origin = ((rotation_center or ((geometry.front - geometry.rear) / 2, 0.))
                           if footprint_mode == "camera_approx" else (0., 0.))
            tracked = track_route(routes[choice.reference_index], vx, horizon, yaw_limit,
                                  body_origin=body_origin, lateral=False,
                                  lookahead_m=max(.45, x_speed * horizon), yaw_floor=floors[3])
            route_yaw = tracked[0][2] if tracked is not None else 0.
            if wz * route_yaw >= 0. and abs(wz) < abs(route_yaw):
                entry_yaw = route_yaw
            elif not wz:
                return choice
        if entry_yaw:
            command = vx, 0., entry_yaw
        else:
            if not transition_slot or abs(wz) >= handoff_yaw_step or abs(delta) <= 2 * handoff_yaw_step:
                return choice
            command = vx, vy, previous[2] + math.copysign(handoff_yaw_step, delta)
        if not _legal_command(command, limits, floors):
            return choice
        budget.remaining -= 1
        diag["sampled_candidates"] += 1
        diag["considered_candidates"] += 1
        prediction = context.predict(command)
        if prediction.body_travel_m + prediction.delay_travel_error > room:
            return choice
        reference = points if follow_published else references[choice.reference_index]
        components, progress, _ = score_candidate(
            command, prediction, None, reference, previous, half_width=half_width, limits=limits,
            progress_scale=progress_scale, target=target, guidance=guidance)
        if components is None or progress < choice.progress - local_path.RESOLUTION / 2:
            return choice
        diag["physical_evaluations"] += 1
        check = context.evaluate(prediction)
        if not check.accepted:
            return choice
        poses = prediction.camera_poses if footprint_mode == "camera_approx" else prediction.body_poses
        if validate_candidate is not None and not validate_candidate(command, poses, prediction.duration).accepted:
            return choice
        components = replace(components, obstacle=_obstacle_score(check),
                             route_change=choice.components.route_change,
                             route_clearance=choice.components.route_clearance)
        diag["entry_arc_selected" if entry_yaw else "handoff_yaw_interpolated"] = True
        diag["admissible_candidates"] += 1
        return Choice(command, poses[:prediction.moving_count], components.total, progress,
                      check.min_clearance_m, prediction, check, components, choice.reference_index)

    if follow_published:
        # The existing path has priority. Only these pursuit arcs can renew
        # it, and each still needs a fresh complete sweep through braking.
        body_origin = (rotation_center or ((geometry.front - geometry.rear) / 2, 0.)
                       if footprint_mode == "camera_approx" else (0., 0.))
        lookahead = max(.45, x_speed * horizon)
        segments = _route_segments(points)
        best = None
        best_cost = math.inf
        for scale in ROUTE_SPEED_SCALES[:min(slots, len(ROUTE_SPEED_SCALES) - int(transition_slot))]:
            if budget.expired():
                break
            tracked = track_route(points, x_speed * scale, horizon, yaw_limit,
                                  body_origin=body_origin, lateral=bool(y_speed), lookahead_m=lookahead,
                                  yaw_floor=floors[3])
            if tracked is None:
                break
            command, room = tracked
            if not any(command[:2]) or not _legal_command(command, limits, floors):
                continue
            budget.remaining -= 1
            diag["sampled_candidates"] += 1
            diag["considered_candidates"] += 1
            prediction = context.predict(command)
            if prediction.body_travel_m + prediction.delay_travel_error > room - route_error:
                rejected["insufficient_room"] = rejected.get("insufficient_room", 0) + 1
                continue
            if budget.expired():
                break
            diag["physical_evaluations"] += 1
            check = context.evaluate(prediction)
            if not check.accepted:
                rejected[check.reason] = rejected.get(check.reason, 0) + 1
                continue
            poses = prediction.camera_poses if footprint_mode == "camera_approx" else prediction.body_poses
            if validate_candidate is not None:
                visible = validate_candidate(command, poses, prediction.duration)
                if not visible.accepted:
                    rejected[visible.reason] = rejected.get(visible.reason, 0) + 1
                    continue
            start = _project_to_route(prediction.body_poses[0][:2], points, segments)[0]
            progress = _project_to_route(prediction.body_poses[prediction.moving_count - 1][:2], points, segments)[0] - start
            if progress <= 0.:
                continue
            components = CandidateScore(0., -progress, _obstacle_score(check), 0., 0., 0.)
            diag["admissible_candidates"] += 1
            cost = pursuit_cost(command, previous, max(x_speed, y_speed), yaw_limit, check)
            if cost < best_cost:
                best_cost = cost
                best = Choice(command, poses[:prediction.moving_count], components.total, progress,
                              check.min_clearance_m, prediction, check, components)
                best_room = room - route_error
            if scale == 1. and check.min_clearance_m >= PREFERRED_CLEARANCE_M and command[2] * previous[2] >= 0.:
                break
        if best is not None:
            diag.update(reference_scores=diag["admissible_candidates"],
                        scores_by_reference=[diag["admissible_candidates"]], reason=None)
            return interpolate(best, best_room)
        diag["reason"] = "search_deadline" if budget.expired() else "no_admissible_candidates"
        return None
    outcomes = budget.for_context(context)
    straight = (x_speed, 0., 0.)
    continuing = previous == straight
    straight_reference = (route_cost_sum[0] == min(route_cost_sum) and candidate_budget is not None and target is not None and target[0] > 0.
                          and abs(math.atan2(target[1], target[0])) <= .1
                          and all(f >= 0. and abs(r) <= (max(1e-8, min(local_path.RESOLUTION / 2, .1 * f))
                                                         if continuing else 1e-8)
                                  for f, r in references[0]))

    # Only this one straight command may retain the incoming tracking speed.
    # A close obstacle or a stop beyond the observed reference restores the detour limit.
    handoff = (previous[0], 0., 0.)
    try_handoff = (straight_handoff_limit is not None and x_speed < previous[0] <= straight_handoff_limit
                   and previous[0] >= floors[0] and not previous[1] and straight_reference and slots > 1
                   and all(f >= 0. and abs(r) <= local_path.RESOLUTION / 2 for f, r in route))

    def commands():
        # The legal previous twist is normally the first seed. Build the rest
        # only if its full current-view check or the straight preference fails.
        first = (straight_reference and continuing and slots == MAX_SEARCH_CANDIDATES
                 and _legal_command(previous, limits, floors))
        if try_handoff:
            yield handoff
        if first:
            yield previous
        recovery_speed = local_path.RESOLUTION / horizon if retry_from_rest else 0.
        distinct, missing = (_commands(previous, measured, preferred, limits, floors, candidate_commands,
                                       max_candidates, recovery_speed=recovery_speed) if slots else ((), []))
        diag["uncovered_families"] = missing
        cached = (cmd for cmd in outcomes if _legal_command(cmd, limits, floors))
        for cmd in dict.fromkeys((*distinct, *cached)):
            if not first or cmd != previous:
                yield cmd

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    def admissible(cmd):
        prediction, check = outcomes[cmd]
        if check is None:
            if budget.expired():
                return None
            check = context.evaluate(prediction)
            outcomes[cmd] = prediction, check
            diag["physical_evaluations"] += 1
        if not check.accepted:
            reason = "collision" if legacy_camera and check.reason == "footprint_collision" else check.reason
            reject(reason)
            return None
        poses = prediction.camera_poses if footprint_mode == "camera_approx" else prediction.body_poses
        if validate_candidate is not None:
            checked = validate_candidate(cmd, poses, prediction.duration)
            if not checked.accepted:
                reject(checked.reason or "candidate_guard")
                return None
        return prediction, check, poses

    best = None
    scores_by_reference = [0] * len(references)
    segments_by_reference = tuple(_route_segments(ref) for ref in references)
    initial_positions = None
    bounded = candidate_budget is not None
    entries = []
    first_checked = None
    straight_check = None
    for ordinal, cmd in enumerate(commands()):
        if budget.expired():
            break
        if cmd in outcomes:
            prediction, check = outcomes[cmd]
            diag["cached_candidates"] += 1
        elif slots:
            if budget.predictions is None:
                prediction = context.predict(cmd)
            else:
                prediction, reused = budget.predictions.get(context, cmd)
                diag["response_reuse"] += int(reused)
            check = None
            outcomes[cmd] = prediction, check
            slots -= 1
            budget.remaining -= 1
            diag["sampled_candidates"] += 1
        else:
            continue
        diag["considered_candidates"] += 1
        if initial_positions is None:
            start = prediction.body_poses[0][:2]
            initial_positions = tuple(_project_to_route(start, ref, segments)[0]
                                      for ref, segments in zip(references, segments_by_reference))
        scores, rejection = [], None
        for index, reference in enumerate(references):
            scores_by_reference[index] += 1
            components, progress, reason = score_candidate(
                cmd, prediction, None, reference, previous, half_width=half_width, limits=limits,
                progress_scale=progress_scale, target=target, segments=segments_by_reference[index],
                initial_position=initial_positions[index], guidance=guidance)
            if components is not None:
                components = replace(components, route_change=route_costs[index], route_clearance=clearance_costs[index])
                scores.append((components, progress, index))
            else:
                rejection = reason
        if bounded and not scores:
            reject(rejection)
            continue
        retaining_speed = try_handoff and cmd == handoff
        capped_straight = (ordinal == 0 and straight_reference and cmd == straight
                           and (continuing or (not any(previous) and x_speed == nominal)))
        early_translation = budget.short_window() and any(cmd[:2]) and not retaining_speed
        # Complete one translating candidate before spending the time budget
        # ranking alternatives. Its braking and visibility checks still apply.
        seed = budget.time_left is not None and first_checked is None and any(cmd[:2]) and not retaining_speed
        preferred_check = (retaining_speed or capped_straight) and bool(scores) and scores[0][2] == 0
        if scores and (preferred_check or early_translation or seed):
            checked = admissible(cmd)
            if checked is None:
                continue
            prediction, check, poses = checked
            if capped_straight:
                straight_check = checked
            obstacle = _obstacle_score(check)
            components, progress, index = (scores[0] if preferred_check and not early_translation
                                           else min(scores, key=lambda item: (item[0].route_cost, item[0].total)))
            components = replace(components, obstacle=obstacle)
            choice = Choice(cmd, poses[:prediction.moving_count], components.total, progress,
                            check.min_clearance_m, prediction, check, components, index)
            if early_translation or (preferred_check and obstacle == 0. and (not retaining_speed or all(
                    0. <= f <= route[-1][0] and abs(r) <= local_path.RESOLUTION / 2 for f, r, _ in poses))):
                diag.update(admissible_candidates=1, reference_scores=sum(scores_by_reference),
                            scores_by_reference=scores_by_reference, reason=None)
                return interpolate(choice)
            if seed:
                first_checked = choice
        if retaining_speed:
            continue  # Ordinary detour ranking only admits speeds inside its cap.
        lower = min(((score.route_cost, score.total, index, ordinal) for score, _, index in scores),
                    default=(math.inf, math.inf, 0, ordinal))
        entries.append((lower, ordinal, cmd, scores, rejection))

    if not diag["considered_candidates"]:
        if budget.expired():
            diag["reason"] = "search_deadline"
        else:
            diag["reason"] = "sample_budget_exhausted" if slots == 0 else "no_legal_commands"
        return None

    # Collision cost is nonnegative. Only skip when neither route smoothness
    # nor the tracking score can beat an already physically admitted winner.
    score_band = local_path.RESOLUTION / progress_scale
    def ranking_bound(entry):
        (route_change, score, index, ordinal), _, cmd, _, _ = entry
        if cmd == previous and cmd[0] > 0. and (cmd[1] or cmd[2]):
            score -= score_band
        return guidance and not any(cmd[:2]), route_change, score, index, ordinal

    if bounded:
        entries.sort(key=ranking_bound)
    best_key = None
    retained = None
    for position, entry in enumerate(entries):
        if budget.expired():
            break
        _, ordinal, cmd, scores, rejection = entry
        if bounded and best_key is not None and ranking_bound(entry) >= best_key:
            diag["bound_skips"] += len(entries) - position
            break
        checked = straight_check if ordinal == 0 and straight_check is not None else admissible(cmd)
        if checked is None:
            continue
        prediction, check, poses = checked
        if not scores:
            reject(rejection)
            continue
        diag["admissible_candidates"] += 1
        obstacle = _obstacle_score(check)
        for components, progress, index in scores:
            components = replace(components, obstacle=obstacle)
            # Follow an admissible forward arc on the route before falling back
            # to a stationary heading change.
            key = (guidance and not any(cmd[:2]), components.route_cost, components.total, index, ordinal)
            if (cmd == previous and cmd[0] > 0. and (cmd[1] or cmd[2]) and obstacle == 0.
                    and (retained is None or components.total < retained.score)):
                retained = Choice(cmd, poses[:prediction.moving_count], components.total, progress,
                                  check.min_clearance_m, prediction, check, components, index)
            if best_key is None or key < best_key:
                best_key = key
                best = Choice(cmd, poses[:prediction.moving_count], components.total, progress,
                              check.min_clearance_m, prediction, check, components, index)
    if (retained is not None and best is not None and retained.reference_index == best.reference_index
            and math.dist(retained.trajectory[-1][:2], best.trajectory[-1][:2])
                + half_width * abs(retained.trajectory[-1][2] - best.trajectory[-1][2]) <= local_path.RESOLUTION / 2
            and retained.progress >= best.progress - local_path.RESOLUTION / 2
            and retained.score <= best.score + score_band):
        best = retained
    if best is None and first_checked is not None and budget.expired():
        best = first_checked
        diag["admissible_candidates"] += 1
    diag["reference_scores"] = sum(scores_by_reference)
    diag["scores_by_reference"] = scores_by_reference
    diag["reason"] = None if best is not None else (
        "search_deadline" if budget.expired() else "no_admissible_candidates")
    return interpolate(best)
