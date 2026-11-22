"""Bounded dynamic-window tracking of an already selected local A* route.

All positions are robot-relative (forward, right) metres. Twist follows ROS:
(vx forward, vy left, wz counter-clockwise).  A* owns topology and the goal;
this module only selects a short executable command. No ROS or robot IDs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import local_path


@dataclass(frozen=True)
class Choice:
    command: tuple[float, float, float]
    trajectory: tuple[tuple[float, float, float], ...]
    score: float
    progress: float
    clearance: float


def _advance(value: float, requested: float, accelerate: float, brake: float, dt: float) -> float:
    rate = brake if abs(requested) < abs(value) or value * requested < 0 else accelerate
    change = max(-rate * dt, min(rate * dt, requested - value))
    return value + change


def _rollout(command, measured, *, horizon, dt, acceleration, braking, stop=False):
    """Simulate actual chassis response, not instantaneous command velocity."""
    vx, vy, wz = measured
    f = r = yaw = 0.0                 # yaw is clockwise/right-positive
    poses = [(f, r, yaw)]
    steps = max(1, math.ceil(horizon / dt))
    delta = horizon / steps
    for _ in range(steps):
        previous_yaw = yaw
        vx = _advance(vx, command[0], acceleration[0], braking[0], delta)
        vy = _advance(vy, command[1], acceleration[1], braking[1], delta)
        wz = _advance(wz, command[2], acceleration[2], braking[2], delta)
        yaw -= wz * delta
        middle = (previous_yaw + yaw) / 2
        f += (vx * math.cos(middle) + vy * math.sin(middle)) * delta
        r += (vx * math.sin(middle) - vy * math.cos(middle)) * delta
        poses.append((f, r, yaw))
    if stop:
        # A command is only admissible if the robot can also brake on that
        # route, including the downstream response delay already represented
        # by the command horizon. Use the same conservative response model.
        remaining = max(abs(vx) / braking[0], abs(vy) / braking[1], abs(wz) / braking[2])
        remaining = min(remaining + dt, 5.0)
        for _ in range(math.ceil(remaining / dt)):
            old = yaw
            vx = _advance(vx, 0.0, acceleration[0], braking[0], dt)
            vy = _advance(vy, 0.0, acceleration[1], braking[1], dt)
            wz = _advance(wz, 0.0, acceleration[2], braking[2], dt)
            yaw -= wz * dt
            mid = (old + yaw) / 2
            f += (vx * math.cos(mid) + vy * math.sin(mid)) * dt
            r += (vx * math.sin(mid) - vy * math.cos(mid)) * dt
            poses.append((f, r, yaw))
            if max(abs(vx), abs(vy), abs(wz)) < 1e-4:
                break
    return tuple(poses)


def _collides(poses, obstacles, geometry, half_width, cell_size):
    """Sweep the rotated *rectangular* body through finely sampled poses.

    Grid cells remain axis-aligned squares. Project their half-extents into
    the body's axes; testing only centers would miss thin door jambs.
    """
    if obstacles.size == 0:
        return False, math.inf
    obs_f, obs_r = obstacles[:, 0], obstacles[:, 1]
    minimum = math.inf
    for f, r, yaw in poses:
        c, s = math.cos(yaw), math.sin(yaw)
        local_f = (obs_f - f) * c + (obs_r - r) * s
        local_r = -(obs_f - f) * s + (obs_r - r) * c
        half_cell = 0.5 * cell_size * (abs(c) + abs(s))
        df = np.maximum(np.maximum(-geometry.rear - local_f, local_f - geometry.front) - half_cell, 0.)
        dr = np.maximum(abs(local_r) - half_width - half_cell, 0.)
        distance = np.hypot(df, dr)
        near = float(distance.min())
        minimum = min(minimum, near)
        if near <= 1e-9:
            return True, 0.0
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
    return (lower-1e-9 <= value <= upper+1e-9
            and (abs(value) < 1e-9 or abs(value)+1e-9 >= minimum))


def _axis_values(lower, upper, floor, prior, measured, preferred):
    """Sample *commands* respecting the downstream velocity deadband."""
    options = {0., float(prior), float(measured), float(preferred)}
    for factor in (.25, .5, .75, 1.):
        if upper > 0.:
            options.add(upper * factor)
        if lower < 0.:
            options.add(lower * factor)
    if floor > 0.:
        if upper >= floor:
            options.add(floor)
        if lower <= -floor:
            options.add(-floor)
    return sorted((v for v in options if _valid_axis(v, lower, upper, floor)),
                  key=lambda v: abs(v-preferred))


def select(*, route, grid, geometry, half_width, measured, previous,
           x_limit, y_limit, yaw_limit, floors, target=None, stop_distance=0.,
           horizon=1.2, max_candidates=80, allow_yaw=True,
           acceleration=(.5, .5, 1.), braking=(.45, .45, .8)) -> Choice | None:
    """Choose the safest progress-making twist through a local A* route.

    Limits: (min,max) for vx, vy, wz respectively; floors: vx, vy,
    standing_wz, moving_wz. Missing feedback is not a measured zero: callers
    must use their existing non-DWA controller instead.
    """
    if (route is None or len(route) < 2 or grid is None or geometry is None
            or measured is None or max_candidates < 1 or not 0.1 <= horizon <= 3.
            or len(measured) != 3 or len(previous) != 3):
        return None
    if not all(math.isfinite(float(v)) for v in (*measured, *previous, *acceleration, *braking)):
        return None
    if min(*acceleration, *braking) <= 0.:
        return None
    points = tuple((float(f), float(r)) for f, r in route)
    if not all(math.isfinite(f) and math.isfinite(r) for f, r in points):
        return None
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
    xs = _axis_values(max(0., x_limit[0]), x_limit[1], floors[0], previous[0], measured[0], preferred[0])
    ys = _axis_values(y_limit[0], y_limit[1], floors[1], previous[1], measured[1], preferred[1])
    # Moving-yaw floor can differ greatly from standing-yaw floor.
    ws = _axis_values(yaw_limit[0], yaw_limit[1], floors[3], previous[2], measured[2], preferred[2])
    commands = []
    for vx in xs:
        for vy in ys:
            for wz in ws:
                if abs(vx)+abs(vy) < 1e-9 or vx < 0.:
                    continue
                if not _valid_axis(wz, *yaw_limit, floors[3]):
                    continue
                commands.append((vx, vy, wz))
    commands.sort(key=lambda cmd: (abs(cmd[0]-preferred[0]) + abs(cmd[1]-preferred[1])
                                   + .2*abs(cmd[2]-preferred[2]),
                                   abs(cmd[0]-previous[0])+abs(cmd[1]-previous[1])
                                   +.2*abs(cmd[2]-previous[2])))
    # The previously executable speed and the geometric-route seed must not
    # disappear from a bounded sampling budget.
    seeds = [(previous[0], previous[1], previous[2]),
             (preferred[0], preferred[1], preferred[2])]
    distinct = []
    for cmd in [*seeds, *commands]:
        if (len(distinct) >= max_candidates or cmd in distinct or cmd[0] < 0.
                or not _valid_axis(cmd[0], *x_limit, floors[0])
                or not _valid_axis(cmd[1], *y_limit, floors[1])
                or not _valid_axis(cmd[2], *yaw_limit, floors[3] if cmd[0] or cmd[1] else floors[2])):
            continue
        if abs(cmd[0])+abs(cmd[1]) <= 1e-9:
            continue
        distinct.append(cmd)
    if not distinct:
        return None
    best = None
    max_distance = max(.2, min(1.0, x_limit[1]*horizon))
    for cmd in distinct:
        full = _rollout(cmd, measured, horizon=horizon, dt=.04,
                        acceleration=acceleration, braking=braking, stop=True)
        collision, clearance = _collides(full, obstacles, geometry, half_width, local_path.RESOLUTION)
        if collision:
            continue
        # The checked braking sweep must not reach inside the original target's
        # stand-off distance, even if this increases Euclidean target range.
        if target is not None and any(math.hypot(target[0]-f, target[1]-r) < stop_distance - .01
                                      for f, r, _ in full):
            continue
        moving_count = max(1, math.ceil(horizon / .04))
        moving = full[:moving_count+1]
        progress, error, tangent = _project_to_route(moving[-1][:2], points)
        if progress < .025 or error > max(.24, .75*half_width):
            continue
        errors = [_project_to_route(p[:2], points)[1] for p in moving[::4]]
        if max(errors, default=0) > max(.30, half_width):
            continue
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
            goal_cost = .25 * (math.dist(moving[-1][:2],target)-math.hypot(*target))/max(1.,math.hypot(*target))
        score = path_cost+progress_reward+obstacle_cost+smooth_cost+heading_cost+goal_cost
        if best is None or score < best.score:
            best = Choice(cmd, moving, score, progress, clearance)
    return best
