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

    def for_context(self, context):
        return self.outcomes.setdefault(context.key(), {})


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
        segments.append(_RouteSegment((fx, fy), (dx, dy), length, length * length,
                                      walked, (dx / length, dy / length)))
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


def _commands(previous, measured, preferred, limits, floors, extra, budget):
    xl, yl, wl = limits

    def legal(cmd):
        return _legal_command(cmd, limits, floors)

    xs = _axis_values(*xl, floors[0], previous[0], measured[0], *(p[0] for p in preferred))
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


def _obstacle_score(check):
    return 1.5 * max(0., .25 - check.min_clearance_m) / .25


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
    if progress < .025 and not turns_toward_route:
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
                   + abs(command[1] - previous[1]) / max(.1, max(abs(y_limit[0]), y_limit[1]))
                   + abs(command[2] - previous[2]) / max(.2, max(abs(yaw_limit[0]), yaw_limit[1])))
    heading = .45 * heading_error / max(.15, half_width) if guidance else .5 * heading_error / math.pi
    goal = 0.
    if target is not None:
        initial_range = math.dist(start_body[:2], target)
        end_range = math.dist(end_body[:2], target)
        goal = .25 * (end_range - initial_range) / max(1., initial_range)
    return CandidateScore(path, along_route, obstacle, smooth, heading, goal), progress, None


def select(*, route, grid, geometry, half_width=None, measured, previous,
           x_limit, y_limit, yaw_limit, floors, target=None, stop_distance=0.,
           horizon=1.2, max_candidates=MAX_SEARCH_CANDIDATES, allow_yaw=True,
           acceleration=_DEFAULT_ACCELERATION, braking=_DEFAULT_BRAKING,
           candidate_commands: tuple[Twist, ...] = (),
           validate_candidate: Callable[[Twist, tuple[Pose, ...], float], CandidateCheck] | None = None,
           footprint_mode: Literal["body", "camera_approx"] = "body",
           rotation_center: tuple[float, float] | None = None, diagnostics: dict | None = None,
           footprint: FootprintSpec | None = None, latency_s=0., remembered=None,
           supported_region=None, terminal_boundary=None, occupied=None, alternate_routes=(),
           candidate_budget: CandidateBudget | None = None, motion_frame=(), measured_target=None, guidance=False) -> Choice | None:
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
    progress_scale = max(.2, min(1.0, math.hypot(x_speed, y_speed) * horizon))
    nominal = min(max(x_speed, y_speed), .35)
    preferred = tuple(_preferred_command(ref, nominal, horizon, yaw_limit,
                                         differential=guidance and not y_speed and footprint_mode == "body")
                      for ref in references)

    budget = candidate_budget if candidate_budget is not None else CandidateBudget(max_candidates)
    slots = min(budget.remaining, max_candidates)
    obstacles = local_path.occupied_squares(grid[1]) if occupied is None else occupied
    context = MotionContext(measured, previous, horizon, latency_s, geometry, footprint, obstacles, remembered,
                            supported_region, target, stop_distance, acceleration, braking, rotation_center,
                            terminal_boundary, motion_frame, measured_target)
    outcomes = budget.for_context(context)
    distinct, missing = (_commands(previous, measured, preferred, limits, floors, candidate_commands,
                                   max_candidates) if slots else ((), []))
    cached = (cmd for cmd in outcomes if _legal_command(cmd, limits, floors))
    commands = dict.fromkeys((*distinct, *cached))
    diag["uncovered_families"] = missing
    if not commands:
        diag["reason"] = "sample_budget_exhausted" if slots == 0 else "no_legal_commands"
        return None

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    def admissible(cmd):
        prediction, check = outcomes[cmd]
        if check is None:
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
    straight_check = None
    for ordinal, cmd in enumerate(commands):
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
                scores.append((components, progress, index))
            else:
                rejection = reason
        if bounded and not scores:
            reject(rejection)
            continue
        # Start or continue a capped straight command on a straight reference when the
        # entire braking sweep has zero proximity cost. Reuse the physical
        # result in ordinary ranking if it is close to an obstacle.
        if (ordinal == 0 and bounded and target is not None and cmd == (x_speed, 0., 0.)
                and (previous == cmd or (not any(previous) and x_speed == nominal))
                and scores and scores[0][2] == 0 and target[0] > 0.
                and abs(math.atan2(target[1], target[0])) <= .1
                and all(f >= 0. and abs(r) < 1e-8 for f, r in references[0])):
            checked = admissible(cmd)
            if checked is None:
                continue
            straight_check = checked
            if _obstacle_score(checked[1]) == 0.:
                prediction, check, poses = checked
                components, progress, index = scores[0]
                diag.update(admissible_candidates=1, reference_scores=sum(scores_by_reference),
                            scores_by_reference=scores_by_reference, reason=None)
                return Choice(cmd, poses[:prediction.moving_count], components.total, progress,
                              check.min_clearance_m, prediction, check, components, index)
        lower = min(((score.total, index, ordinal) for score, _, index in scores), default=(math.inf, 0, ordinal))
        entries.append((lower, ordinal, cmd, scores, rejection))

    # Collision cost is nonnegative. With the original tie order included,
    # no remaining lower bound can beat an already physically admitted winner.
    if bounded:
        entries.sort(key=lambda entry: entry[0])
    best_key = None
    for position, (lower, ordinal, cmd, scores, rejection) in enumerate(entries):
        if bounded and best_key is not None and lower >= best_key:
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
            key = (components.total, index, ordinal)
            if best_key is None or key < best_key:
                best_key = key
                best = Choice(cmd, poses[:prediction.moving_count], key[0], progress,
                              check.min_clearance_m, prediction, check, components, index)
    diag["reference_scores"] = sum(scores_by_reference)
    diag["scores_by_reference"] = scores_by_reference
    diag["reason"] = None if best is not None else "no_admissible_candidates"
    return best
