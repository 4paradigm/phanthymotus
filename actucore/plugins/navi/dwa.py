"""Bounded dynamic-window tracking of an already selected local A* route.

All positions are robot-relative (forward, right) metres. Twist follows ROS:
(vx forward, vy left, wz counter-clockwise).  A* owns topology and the goal;
this module only selects a short executable command. No ROS or robot IDs.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np

from . import local_path


Twist = tuple[float, float, float]
Pose = tuple[float, float, float]
_DEFAULT_ACCELERATION = (.5, .5, 1.)
_DEFAULT_BRAKING = (.45, .45, .8)
_BRAKE_BUDGET_S = 5.


@dataclass(frozen=True)
class CandidateCheck:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class Choice:
    command: Twist
    trajectory: tuple[Pose, ...]
    score: float
    progress: float
    clearance: float


def _response(value, requested, accelerate, brake, dt):
    """Return terminal speed and the signed integral over a linear response."""
    area = 0.
    if value * requested < 0.:
        # Brake to zero before accelerating in the opposite direction.
        to_zero = abs(value) / brake
        if to_zero < dt:
            area = .5 * value * to_zero
            dt -= to_zero
            value = 0.
        else:
            requested = 0.
    rate = brake if abs(requested) < abs(value) else accelerate
    ramp = abs(requested - value) / rate
    if ramp <= dt:
        return requested, area + .5 * (value + requested) * ramp + requested * (dt - ramp)
    end = value + math.copysign(rate * dt, requested - value)
    return end, area + .5 * (value + end) * dt


def stopping_horizon(command, measured, *, horizon,
                     acceleration=_DEFAULT_ACCELERATION, braking=_DEFAULT_BRAKING):
    """Time through the modeled complete stop, or None if its tail exceeds budget.

    ``horizon`` is the moving prediction duration, before braking to zero. The
    response and defaults are shared with selection so refreshed feedback does
    not reuse a shorter braking tail from a previous measured velocity.
    """
    terminal = (_response(value, requested, accelerate, brake, horizon)[0]
                for value, requested, accelerate, brake in zip(measured, command, acceleration, braking))
    required = max(abs(value) / brake for value, brake in zip(terminal, braking))
    return None if required > _BRAKE_BUDGET_S else horizon + required


@dataclass(frozen=True)
class _Prediction:
    poses: tuple[Pose, ...]
    moving_count: int
    duration: float
    stopped: bool
    # Bounds in metres and radians for motion between adjacent samples.
    position_error: float
    half_travel: float
    half_turn: float


def _predict(command, measured, *, horizon, dt, acceleration, braking, stop):
    """Integrate actual response, including rate changes and zero crossings."""
    velocity = tuple(measured)
    f = r = yaw = 0.
    poses = [(f, r, yaw)]
    position_error = half_travel = half_turn = 0.
    duration = horizon

    def integrate(target, seconds):
        nonlocal velocity, f, r, yaw, position_error, half_travel, half_turn
        if not seconds:
            return
        steps = math.ceil(seconds / dt)
        step = seconds / steps
        for _ in range(steps):
            previous = velocity
            vx, forward = _response(previous[0], target[0], acceleration[0], braking[0], step)
            vy, left = _response(previous[1], target[1], acceleration[1], braking[1], step)
            wz, turn = _response(previous[2], target[2], acceleration[2], braking[2], step)
            velocity = (vx, vy, wz)
            middle = yaw - .5 * turn
            c, s = math.cos(middle), math.sin(middle)
            f += forward * c + left * s
            r += forward * s - left * c
            yaw -= turn
            poses.append((f, r, yaw))

            # Component speeds never exceed the endpoint magnitudes during
            # a monotone ramp. Distance to the nearer sample is at most
            # speed * step/2. The heading used to rotate integrated body
            # travel differs from the exact heading by at most omega*step/2.
            vx = max(abs(vx), abs(previous[0]))
            vy = max(abs(vy), abs(previous[1]))
            omega = max(abs(wz), abs(previous[2]))
            travel = math.hypot(vx, vy) * step
            angular_gap = omega * step / 2
            position_error += travel * angular_gap
            half_travel = max(half_travel, travel / 2)
            half_turn = max(half_turn, angular_gap)

    integrate(command, horizon)
    moving_count = len(poses)
    if stop:
        required = max(abs(v) / rate for v, rate in zip(velocity, braking))
        tail = min(required, _BRAKE_BUDGET_S)
        integrate((0., 0., 0.), tail)
        duration += tail
    stopped = max(map(abs, velocity)) <= 1e-10 and (not stop or required <= _BRAKE_BUDGET_S)
    return _Prediction(tuple(poses), moving_count, duration, stopped,
                       position_error, half_travel, half_turn)


def _rollout(command, measured, *, horizon, dt, acceleration, braking, stop=False):
    return _predict(command, measured, horizon=horizon, dt=dt, acceleration=acceleration,
                    braking=braking, stop=stop).poses


def _collides(poses, obstacles, geometry, half_width, cell_size, margin=0.):
    """Sweep the rotated *rectangular* body through finely sampled poses.

    Grid cells remain axis-aligned squares. Project their half-extents into
    the body's axes; testing only centers would miss thin door jambs.
    """
    if obstacles.size == 0 or not len(poses):
        return False, math.inf
    obs_f, obs_r = obstacles.T
    forward, right, yaw = np.asarray(poses).T
    cosine = np.fromiter(map(math.cos, yaw), float, len(poses))
    sine = np.fromiter(map(math.sin, yaw), float, len(poses))
    # Each row is a pose, each column an actual occupied square. Keep the
    # first pose's early exit, then cap each temporary at 32K pose/cell pairs.
    poses_per_batch = max(1, 32768 // len(obstacles))
    start, end = 0, 1
    minimum = math.inf
    while start < len(poses):
        c, s = cosine[start:end, None], sine[start:end, None]
        delta_f = obs_f - forward[start:end, None]
        delta_r = obs_r - right[start:end, None]
        local_f = delta_f * c + delta_r * s
        local_r = -delta_f * s + delta_r * c
        half_cell = 0.5 * cell_size * (abs(c) + abs(s))
        df = np.maximum(np.maximum(-geometry.rear - local_f, local_f - geometry.front) - half_cell, 0.)
        dr = np.maximum(abs(local_r) - half_width - half_cell, 0.)
        near = max(0., float(np.hypot(df, dr).min()) - margin)
        if near <= 1e-9:
            return True, 0.0
        minimum = min(minimum, near)
        start, end = end, min(len(poses), end + poses_per_batch)
    return False, minimum


def _project_to_route(point, route):
    """Return along-route progress, cross-track error, and local tangent."""
    at, closest, tangent, walked = 0.0, math.inf, (1., 0.), 0.0
    x, y = point
    for (fx, fy), (tx, ty) in zip(route, route[1:]):
        dx, dy = tx - fx, ty - fy
        length = math.hypot(dx, dy)
        if length < 1e-8:
            continue
        t = max(0., min(1., ((x-fx)*dx+(y-fy)*dy) / (length*length)))
        error = math.hypot(x-fx-t*dx, y-fy-t*dy)
        if error < closest:
            at, closest, tangent = walked + t * length, error, (dx / length, dy / length)
        walked += length
    return at, closest, tangent


def _valid_axis(value, lower, upper, minimum):
    return lower <= value <= upper and (value == 0. or abs(value) >= minimum)


def _axis_values(lower, upper, floor, prior, measured, preferred):
    """Sample target commands within signed limits and the downstream deadband."""
    options = {0., float(prior), float(measured), float(preferred), floor, -floor}
    for factor in (.25, .5, .75, 1.):
        options.update((upper * factor, lower * factor))
    return sorted(v for v in options if _valid_axis(v, lower, upper, floor))


def _commands(previous, measured, preferred, limits, floors, extra, budget):
    xl, yl, wl = limits

    def legal(cmd):
        vx, vy, wz = cmd
        return (any(cmd) and vx >= 0. and _valid_axis(vx, *xl, floors[0])
                and _valid_axis(vy, *yl, floors[1])
                and _valid_axis(wz, *wl, floors[3] if vx or vy else floors[2]))

    xs = _axis_values(*xl, floors[0], previous[0], measured[0], preferred[0])
    ys = _axis_values(*yl, floors[1], previous[1], measured[1], preferred[1])
    yaw_refs = (previous[2], measured[2], preferred[2])
    standing_yaws = _axis_values(*wl, floors[2], *yaw_refs)
    moving_yaws = _axis_values(*wl, floors[3], *yaw_refs)
    ws = sorted(set(standing_yaws).union(moving_yaws))
    if not xs or not ys or not ws:
        return (), []
    vx = min(xs, key=lambda v: abs(v - preferred[0]))
    vy = min(ys, key=lambda v: abs(v - preferred[1]))
    route_yaws = moving_yaws if vx or vy else standing_yaws
    route_seed = (vx, vy, min(route_yaws, key=lambda v: abs(v - preferred[2]), default=0.))
    low = next((x for x in xs if x > 0.), 0.)
    seeds = [("previous", previous), ("route", route_seed), ("low_straight", (low, 0., 0.))]
    for name, sign in (("left", 1), ("right", -1)):
        turns = [w for w in moving_yaws if sign * w > 0]
        if low and turns:
            turn = min(turns, key=abs)
            seeds.extend(((name + "_arc", (low, 0., turn)),
                          (name + "_turn", (low, 0., max(turns, key=abs)))))
        standing = [w for w in standing_yaws if sign * w > 0]
        if standing:
            seeds.append((name + "_standing_turn", (0., 0., min(standing, key=abs))))
        lateral = [y for y in ys if sign * y > 0]
        if lateral:
            seeds.append((name + "_lateral", (0., max(lateral, key=abs), 0.)))
    seed_commands = [tuple(cmd) for _, cmd in seeds if legal(cmd)]
    extra = [tuple(cmd) for cmd in extra if legal(cmd)]
    # Preserve the first externally supplied motion reference even at small
    # budgets, then reserve a place for each supported family before filling.
    ordered = dict.fromkeys(seed_commands[:1] + extra[:1] + seed_commands + extra)
    sampled = list(ordered)[:budget]
    slots = budget - len(sampled)
    if slots:
        pool = [cmd for cmd in product(xs, ys, ws) if legal(cmd) and cmd not in ordered]
        if len(pool) > slots:
            pool = [pool[i] for i in np.linspace(0, len(pool) - 1, slots, dtype=int)]
        sampled.extend(pool)
    missed = [name for name, cmd in seeds if legal(cmd) and tuple(cmd) not in sampled]
    return sampled, missed


def _offset_poses(poses, offset, origin=(0., 0.)):
    """Apply a rigidly attached point offset in local (forward,right) axes."""
    x, y = offset
    if x == 0. and y == 0. and origin == (0., 0.):
        return poses
    return tuple((f + origin[0] + math.cos(a)*x - math.sin(a)*y,
                  r + origin[1] + math.sin(a)*x + math.cos(a)*y, a) for f, r, a in poses)


def _violates_standoff(camera, measured, command, target, minimum, sample_gap):
    # For single-axis motion without reversal, the continuous camera path
    # is the exact axis-aligned segment. This also permits terminal tangency.
    if measured[2] == command[2] == 0. and (
            measured[1] == command[1] == 0. and measured[0] * command[0] >= 0.
            or measured[0] == command[0] == 0. and measured[1] * command[1] >= 0.):
        closest = [max(min(a, b), min(max(a, b), t))
                   for a, b, t in zip(camera[0][:2], camera[-1][:2], target)]
        return math.dist(closest, target) < minimum - 1e-10
    # Every point lies within sample_gap of at least one checked sample.
    return any(math.hypot(target[0] - f, target[1] - r) < minimum + sample_gap - 1e-10
               for f, r, _ in camera)


def select(*, route, grid, geometry, half_width, measured, previous,
           x_limit, y_limit, yaw_limit, floors, target=None, stop_distance=0.,
           horizon=1.2, max_candidates=64, allow_yaw=True,
           acceleration=_DEFAULT_ACCELERATION, braking=_DEFAULT_BRAKING,
           candidate_commands: tuple[Twist, ...] = (),
           validate_candidate: Callable[[Twist, tuple[Pose, ...], float], CandidateCheck] | None = None,
           footprint_mode: Literal["body", "camera_approx"] = "body",
           rotation_center: tuple[float, float] = (0., 0.), diagnostics: dict | None = None) -> Choice | None:
    """Select a legal target velocity whose complete predicted stop is admissible.

    Routes and targets are in current grid coordinates. In body mode the
    validator receives body poses; in camera_approx it receives camera poses
    and must supply the conservative physical sweep. Its time argument covers
    the requested horizon and the entire modeled brake tail.
    """
    diag = diagnostics if diagnostics is not None else {}
    rejected = {}
    diag.update(sampled_candidates=0, admissible_candidates=0, rejected=rejected,
                reason="invalid_inputs", uncovered_families=[])
    if (route is None or len(route) < 2 or grid is None or geometry is None
            or measured is None or max_candidates < 1 or not 0.1 <= horizon <= 3.
            or len(measured) != 3 or len(previous) != 3
            or not math.isfinite(stop_distance) or stop_distance < 0.
            or footprint_mode not in ("body", "camera_approx")):
        return None
    if not all(math.isfinite(float(v)) for v in (*measured, *previous, *acceleration, *braking,
                                                  *x_limit, *y_limit, *yaw_limit, *floors, *rotation_center)):
        return None
    if min(*acceleration, *braking) <= 0.:
        return None
    points = tuple((float(f), float(r)) for f, r in route)
    if target is not None and not all(map(math.isfinite, target)):
        return None
    if not all(math.isfinite(f) and math.isfinite(r) for f, r in points):
        return None
    if footprint_mode == "camera_approx" and validate_candidate is None:
        diag["reason"] = "missing_candidate_validator"
        return None
    if footprint_mode == "camera_approx":
        offset = (-rotation_center[0], -rotation_center[1])
    elif target is not None:
        mount = geometry.base_from_optical
        if mount is None or not all(map(math.isfinite, (mount[0, 3], mount[1, 3]))):
            diag["reason"] = "invalid_camera_offset"
            return None
        offset = (mount[0, 3], -mount[1, 3])
    _, occupied = grid
    rows, cols = np.nonzero(occupied)
    obstacles = np.column_stack((local_path.FORWARD[rows, cols], local_path.RIGHT[rows, cols]))
    dx, dy = points[-1]
    for f, r in points[1:]:
        if math.hypot(f, r) >= .45:
            dx, dy = f, r
            break
    angle = math.atan2(dy, dx)
    nominal = min(max(x_limit[1], abs(y_limit[0]), y_limit[1]), .35)
    preferred = (max(0., nominal * math.cos(angle)), -nominal * math.sin(angle),
                 -min(abs(angle) / max(.4, horizon), abs(yaw_limit[0]) if angle > 0 else yaw_limit[1]) *
                 (1. if angle >= 0 else -1.))
    if not allow_yaw:
        preferred = (preferred[0], preferred[1], 0.)
        yaw_limit = (0., 0.)
    distinct, missing = _commands(previous, measured, preferred, (x_limit, y_limit, yaw_limit),
                                  floors, candidate_commands, max_candidates)
    diag.update(sampled_candidates=len(distinct), uncovered_families=missing)
    if not distinct:
        diag["reason"] = "no_legal_commands"
        return None

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    best = None
    max_distance = max(.2, min(1.0, x_limit[1]*horizon))
    for cmd in distinct:
        prediction = _predict(cmd, measured, horizon=horizon, dt=.04, acceleration=acceleration,
                              braking=braking, stop=True)
        if not prediction.stopped:
            reject("braking_incomplete")
            continue
        full = prediction.poses
        gap = prediction.position_error + prediction.half_travel
        if footprint_mode == "body":
            corner = math.hypot(max(geometry.front, geometry.rear), half_width)
            collision, clearance = _collides(full, obstacles, geometry, half_width, local_path.RESOLUTION,
                                             gap + corner * prediction.half_turn)
            if collision:
                reject("footprint_collision")
                continue
            camera = _offset_poses(full, offset) if target is not None else ()
        else:
            # The camera turns about the body center c. Its displacement
            # from the current camera origin is c + body_travel - R(yaw)c.
            full = camera = _offset_poses(full, offset, rotation_center)
            clearance = math.inf
        if target is not None and stop_distance > 0.:
            camera_gap = gap + math.hypot(*offset) * prediction.half_turn
            if _violates_standoff(camera, measured, cmd, target, stop_distance, camera_gap):
                reject("target_standoff")
                continue
        if validate_candidate is not None:
            checked = validate_candidate(cmd, full, prediction.duration)
            if not checked.accepted:
                reject(checked.reason or "candidate_guard")
                continue
        moving = full[:prediction.moving_count]
        progress, error, tangent = _project_to_route(moving[-1][:2], points)
        if progress < .025:
            reject("insufficient_progress")
            continue
        errors = [_project_to_route(p[:2], points)[1] for p in moving[::4]]
        if error > max(.24, .75*half_width) or max(errors, default=0) > max(.30, half_width):
            reject("off_route")
            continue
        diag["admissible_candidates"] += 1
        # Explicitly favor motion *along the detour*, even if it temporarily
        # travels away from the original target. The target never changes.
        path_cost = 3.0 * (sum(errors)/max(1,len(errors))) / max(.15, half_width)
        progress_reward = -2.0 * progress / max_distance
        obstacle_cost = 1.5 * max(0., .25-clearance) / .25
        smooth_cost = .6 * (abs(cmd[0]-previous[0])/max(.1,x_limit[1])
                              +abs(cmd[1]-previous[1])/max(.1,max(abs(y_limit[0]),y_limit[1]))
                              +abs(cmd[2]-previous[2])/max(.2,max(abs(yaw_limit[0]),yaw_limit[1])))
        desired_heading = math.atan2(tangent[1], tangent[0])
        heading_cost = .5 * abs(math.remainder(moving[-1][2]-desired_heading, math.tau)) / math.pi
        goal_cost = 0.
        if target is not None:
            initial_range = math.dist(camera[0][:2], target)
            end_range = math.dist(camera[prediction.moving_count-1][:2], target)
            goal_cost = .25 * (end_range - initial_range) / max(1., initial_range)
        score = path_cost+progress_reward+obstacle_cost+smooth_cost+heading_cost+goal_cost
        if best is None or score < best.score:
            best = Choice(cmd, moving, score, progress, clearance)
    diag["reason"] = None if best is not None else "no_admissible_candidates"
    return best
