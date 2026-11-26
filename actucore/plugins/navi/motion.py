"""Rate-limited chassis response and physical checks in a local forward/right frame."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Tuple

import numpy as np

from . import local_path

Twist = Tuple[float, float, float]
Pose = Tuple[float, float, float]
_DEFAULT_ACCELERATION = (.5, .5, 1.)
_DEFAULT_BRAKING = (.45, .45, .8)
_BRAKE_BUDGET_S = 5.


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


def _offset_poses(poses, offset):
    """Apply a rigidly attached point offset in local (forward,right) axes."""
    x, y = offset
    if x == 0. and y == 0.:
        return poses
    result = []
    for f, r, yaw in poses:
        c, s = math.cos(yaw), math.sin(yaw)
        result.append((f + c*x - s*y, r + s*x + c*y, yaw))
    return tuple(result)


Mode = Literal["body", "camera_approx"]


@dataclass(frozen=True)
class FootprintSpec:
    """Physical extents about the initial grid reference.

    Body mode anchors these to the body pivot. In camera_approx the extents
    from camera_square (e.g. front=0, rear=2*half_length) follow the camera
    pose, which already accounts for rotation about the displaced body pivot.
    """
    front: float
    rear: float
    half_width: float
    clearance_margin_m: float
    mode: Mode = "body"


@dataclass(frozen=True)
class TravelBudget:
    limit_m: float
    consumed_m: float = 0.


@dataclass(frozen=True)
class StopPlane:
    normal: tuple[float, float]
    offset_m: float


@dataclass(frozen=True)
class MotionTrace:
    """Body pivot and camera poses in the *same grid*. Brake starts at moving_count-1."""
    body_poses: tuple[Pose, ...]
    camera_poses: tuple[Pose, ...]
    times: tuple[float, ...]
    velocities: tuple[Twist, ...]
    moving_count: int
    stopped: bool
    position_error: float
    half_travel: float
    half_turn: float
    delay_position_error: float
    delay_yaw_error: float
    body_travel_m: float
    delay_travel_error: float
    camera_offset: tuple[float, float]
    mode: Mode
    straight_axis: int | None
    interval_errors: tuple[tuple[float, float], ...]

    # Cumulative longitudinal/lateral displacement differences between delivery times.
    delay_axes: tuple[tuple[float, float], ...]
    heading_span: float

    @property
    def duration(self):
        return self.times[-1]

    @property
    def poses(self):
        return self.body_poses


@dataclass(frozen=True)
class MotionCheck:
    accepted: bool
    reason: str | None
    min_clearance_m: float
    first_hit: int | None
    swept_bounds: tuple[tuple[float, float], tuple[float, float]]
    terminal_distance: float | None


def _ramp_arclength(start, end, dt):
    """Integral of the norm of a linearly varying 2D velocity."""
    x, y = start[:2]
    dx, dy = end[0] - x, end[1] - y
    change = math.hypot(dx, dy)
    if change < 1e-12:
        return dt * math.hypot(x, y)
    along = (x * dx + y * dy) / change
    across = abs(x * dy - y * dx) / change
    if across < 1e-12:
        return dt * (abs(along + change)*(along + change) - abs(along)*along) / (2*change)

    def primitive(t):
        return .5 * (t * math.hypot(t, across) + across**2 * math.asinh(t / across))
    return dt * (primitive(along + change) - primitive(along)) / change


def _travel(start, end, integrals, target, acceleration, braking, dt):
    # If the planar response stays on one ray the integrated displacement
    # is already the exact distance, including any changes of speed.
    if start[0] == end[0] == start[1] == end[1] == 0.:
        return 0.
    if (start[1] == end[1] == 0. and start[0]*end[0] >= 0.
            or start[0] == end[0] == 0. and start[1]*end[1] >= 0.):
        return math.hypot(*integrals[:2])
    # Split at a zero crossing or target attainment so each piece is affine.
    cuts = {dt}
    for i in (0, 1):
        v, goal = start[i], target[i]
        reverse = v * goal < 0.
        zero = abs(v) / braking[i] if reverse else 0.
        rate = braking[i] if abs(v) > abs(goal) and not reverse else acceleration[i]
        attain = zero + abs(goal - (0. if reverse else v)) / rate
        cuts.update(t for t in (zero, attain) if 0. < t < dt)
    before, t0, travel = start, 0., 0.
    for t in sorted(cuts):
        after = end if t == dt else tuple(
            _response(start[i], target[i], acceleration[i], braking[i], t)[0] for i in (0, 1))
        travel += _ramp_arclength(before, after, t - t0)
        before, t0 = after, t
    return travel


def predict_motion(measured, previous, command, *, hold_s, latency_s, acceleration, braking,
                   geometry, mode, dt=.04, rotation_center=None, stop=True, initial_bounds=None):
    """Hold includes the uncertain old-command interval. Then brake all axes to zero.

    The nominal switch is at the latest possible arrival. For each scalar
    axis the earliest and latest response bound *every* intermediate response:
    the scalar flow preserves ordering under the same target, and is monotone
    in the target. This also holds across sign reversals and during braking.
    """
    if (mode not in ("body", "camera_approx") or not 0. <= latency_s <= hold_s
            or not all(math.isfinite(v) for v in (*measured, *previous, *command,
                                                   *acceleration, *braking, hold_s, dt))
            or min(*acceleration, *braking, dt) <= 0.):
        raise ValueError("invalid motion response inputs")
    if mode == "camera_approx":
        center = rotation_center if rotation_center is not None else ((geometry.front - geometry.rear)/2, 0.)
        offset = (-center[0], -center[1])
    else:
        center = (0., 0.)
        mount = geometry.base_from_optical if geometry is not None else None
        offset = (mount[0, 3], -mount[1, 3]) if mount is not None else (0., 0.)

    times, velocities, poses = [0.], [tuple(measured)], [(*center, 0.)]
    interval_errors = []
    delay_axes = []
    delay_x = delay_y = heading_span = 0.
    early = tuple(measured)
    lower, upper = initial_bounds if initial_bounds is not None else (None, None)
    uncertain = latency_s > 0. and tuple(previous) != tuple(command)
    position_error = half_travel = half_turn = body_travel = 0.
    delay_position = delay_yaw = delay_travel = 0.

    accel_x, accel_y, accel_yaw = acceleration
    brake_x, brake_y, brake_yaw = braking

    def steady_area(target, step):
        return (_response(target[0], target[0], accel_x, brake_x, step)[1],
                _response(target[1], target[1], accel_y, brake_y, step)[1],
                _response(target[2], target[2], accel_yaw, brake_yaw, step)[1])

    def response(values, target, step, stable_area):
        if values == target:
            return target, stable_area
        end_x, area_x = ((target[0], stable_area[0]) if values[0] == target[0]
                         else _response(values[0], target[0], accel_x, brake_x, step))
        end_y, area_y = ((target[1], stable_area[1]) if values[1] == target[1]
                         else _response(values[1], target[1], accel_y, brake_y, step))
        end_yaw, area_yaw = ((target[2], stable_area[2]) if values[2] == target[2]
                             else _response(values[2], target[2], accel_yaw, brake_yaw, step))
        return (end_x, end_y, end_yaw), (area_x, area_y, area_yaw)

    def integrate(target, early_target, seconds):
        nonlocal delay_x, delay_y, heading_span
        nonlocal early, lower, upper, position_error, half_travel, half_turn, body_travel
        nonlocal delay_position, delay_yaw, delay_travel
        if seconds <= 0.:
            return
        start_time = times[-1]
        steps = math.ceil(seconds / dt)
        step = seconds / steps
        same_target = early_target is target
        target, early_target = tuple(target), tuple(early_target)
        stable_area = steady_area(target, step)
        early_stable_area = stable_area if same_target or not uncertain else steady_area(early_target, step)
        if initial_bounds is not None:
            low_target = tuple(map(min, target, early_target))
            high_target = tuple(map(max, target, early_target))
            low_area = steady_area(low_target, step)
            high_area = steady_area(high_target, step)
        # A converged phase still emits every time and pose. Its response area,
        # travelled distance and interpolation allowances are constant.
        steady = None
        for i in range(steps):
            before = velocities[-1]
            if before == target and steady is not None:
                end, area = target, stable_area
                travel, half_speed_step, angular_gap = steady
            else:
                end, area = response(before, target, step, stable_area)
                travel = _travel(before, end, area, target, acceleration, braking, step)
                speed = math.hypot(max(abs(end[0]), abs(before[0])), max(abs(end[1]), abs(before[1])))
                half_speed_step = speed * step / 2
                angular_gap = max(abs(end[2]), abs(before[2])) * step / 2
                if before == target:
                    steady = travel, half_speed_step, angular_gap
            f, r, yaw = poses[-1]
            middle = yaw - .5 * area[2]
            c, s = math.cos(middle), math.sin(middle)
            poses.append((f + area[0]*c + area[1]*s, r + area[0]*s - area[1]*c, yaw - area[2]))
            velocities.append(end)
            times.append(start_time + (i + 1) * step)
            body_travel += travel
            half_travel = max(half_travel, half_speed_step)
            half_turn = max(half_turn, angular_gap)
            position_error += travel * angular_gap
            if initial_bounds is not None:
                lower, a = response(lower, low_target, step, low_area)
                upper, b = response(upper, high_target, step, high_area)
                dx, dy, da = (max(abs(x - lo), abs(x - hi)) for x, lo, hi in zip(area, a, b))
                delay_x += dx
                delay_y += dy
                delay_travel += dx + dy
                delay_yaw += da
                delay_position += dx + dy + travel * min(2., delay_yaw)
            elif uncertain:
                # Equal states under the same request have identical scalar responses.
                if same_target and early == before:
                    early = end
                    delay_position += travel * min(2., delay_yaw)
                else:
                    early, early_area = response(early, early_target, step, early_stable_area)
                    # Each component difference keeps its sign across the step.
                    # Its absolute integral bounds that of any intermediate delay.
                    dx, dy = abs(early_area[0]-area[0]), abs(early_area[1]-area[1])
                    delay_x += dx
                    delay_y += dy
                    velocity_error = dx + dy
                    delay_travel += velocity_error
                    delay_yaw += abs(early_area[2]-area[2])
                    delay_position += velocity_error + travel * min(2., delay_yaw)
            heading_span += 2 * angular_gap
            delay_axes.append((delay_x, delay_y))
            interval_errors.append((position_error + delay_position + half_speed_step,
                                    delay_yaw + angular_gap))

    integrate(previous, command, latency_s)
    integrate(command, command, hold_s - latency_s)
    moving_count = len(times)
    terminal = velocities[-1]
    if initial_bounds is not None:
        terminal = tuple(max(abs(v), abs(w)) for v, w in zip(lower, upper))
    elif uncertain:
        terminal = tuple(max(abs(v), abs(w)) for v, w in zip(terminal, early))
    required = max(abs(v) / b for v, b in zip(terminal, braking))
    if stop:
        integrate((0., 0., 0.), (0., 0., 0.), min(required, _BRAKE_BUDGET_S))
    stopped = required <= _BRAKE_BUDGET_S and max(map(abs, velocities[-1])) <= 1e-10
    if initial_bounds is not None:
        stopped = stopped and max(map(abs, (*lower, *upper))) <= 1e-10
    elif uncertain:
        stopped = stopped and max(map(abs, early)) <= 1e-10
    body = tuple(poses)
    camera = _offset_poses(body, offset)
    inputs = (measured, command, previous) if latency_s > 0. else (measured, command)
    if initial_bounds is not None:
        inputs += initial_bounds
    # A fixed-heading single-axis path is a segment only without reversals.
    straight = next((axis for axis in (0, 1)
                     if all(v[2] == v[1-axis] == 0. for v in inputs)
                     and (all(v[axis] >= 0. for v in inputs) or all(v[axis] <= 0. for v in inputs))), None)
    return MotionTrace(body, camera, tuple(times), tuple(velocities), moving_count, stopped,
                       position_error, half_travel, half_turn, delay_position, delay_yaw,
                       body_travel, delay_travel, offset, mode, straight, tuple(interval_errors), tuple(delay_axes), heading_span)


def _layers_in_sweep(evidence, bounds):
    (lo_f, lo_r), (hi_f, hi_r) = bounds
    close = []
    for obstacle in evidence if isinstance(evidence, tuple) else (evidence,):
        layer = local_path.occupied_squares(obstacle)
        if layer is None or not len(layer.centers):
            continue
        extent = (local_path.RESOLUTION / 2 * (abs(math.cos(layer.yaw)) + abs(math.sin(layer.yaw)))
                  + layer.alignment_error + 1e-9)
        f, r = layer.centers.T
        mask = ((lo_f - extent <= f) & (f <= hi_f + extent)
                & (lo_r - extent <= r) & (r <= hi_r + extent))
        if mask.any():
            close.append(layer if mask.all() else replace(layer, centers=layer.centers[mask]))
    return tuple(close)


def evaluate_motion(trace, *, footprint, occupied, remembered, supported_region,
                    original_goal, stop_distance_m, terminal_boundary=None, prepared_layers=None,
                    cull_obstacles=False):
    """Check the complete stop. supported_region is a pair of lower/upper (f,r) corners.

    first_hit is the earliest occupied/remembered contact sample. terminal_distance
    is the final camera-to-goal distance. Collision and stop checks include the
    pose integration, inter-sample motion, and all supported delivery delays.
    """
    if footprint.mode != trace.mode:
        raise ValueError("footprint and motion reference frames differ")
    camera_mode = trace.mode == "camera_approx"
    poses = np.asarray(trace.camera_poses if camera_mode else trace.body_poses)
    margin = footprint.clearance_margin_m
    front, rear = footprint.front + margin, footprint.rear + margin
    width = footprint.half_width + margin
    # The pivot stays at the body centre when the initial grid origin is the camera.
    ox, oy = trace.camera_offset if camera_mode else (0., 0.)
    radius = math.hypot(max(abs(ox+front), abs(ox-rear)), abs(oy)+width)
    linear_error = trace.position_error + trace.half_travel + trace.delay_position_error
    angular_error = trace.half_turn + trace.delay_yaw_error
    padding = linear_error + radius * min(2., angular_error)
    # Each endpoint covers its half of both adjacent intervals. Use the
    # end-of-interval error to bound every delivery time within that interval.
    intervals = np.asarray([linear + radius * min(2., angular)
                            for linear, angular in trace.interval_errors])
    sweep_padding = np.maximum(np.r_[0., intervals], np.r_[intervals, 0.])
    f, r, yaw = poses.T
    c, s = np.cos(yaw), np.sin(yaw)
    length, shift = (front+rear)/2, (front-rear)/2
    f, r = f + shift*c, r + shift*s

    def support(nx, ny):
        return (nx*f + ny*r + length*abs(nx*c + ny*s) + width*abs(-nx*s + ny*c)).max() + padding

    bounds = ((-float(support(-1., 0.)), -float(support(0., -1.))),
              (float(support(1., 0.)), float(support(0., 1.))))
    if cull_obstacles:
        occupied = _layers_in_sweep(occupied, bounds)
        remembered = _layers_in_sweep(remembered, bounds)
    if prepared_layers is not None:
        clearance, first_hit, reason = prepared_layers.clearance(
            poses, f, r, yaw, front, rear, width, sweep_padding)
    else:
        clearance, first_hit, reason = math.inf, None, None
        for evidence, collision in ((occupied, "footprint_collision"), (remembered, "remembered_obstacle")):
            # Several registered views can have different grid axes and alignment errors.
            for obstacles in evidence if isinstance(evidence, tuple) else (evidence,):
                distance, hit = local_path.rectangle_clearance(
                    poses, local_path.occupied_squares(obstacles), front, rear, width, padding=sweep_padding)
                clearance = min(clearance, distance)
                if hit is not None and (first_hit is None or hit < first_hit):
                    reason, first_hit = collision, hit
    heading_span = trace.heading_span + trace.delay_yaw_error
    if first_hit is not None and trace.delay_travel_error > 0. and heading_span < math.pi / 2:
        # The delivery-induced body-axis displacement can point in a different
        # direction at each integration step. Nominal rotation plus the delivery yaw error bounds
        # |sin(delta heading)| between any two points on the trace. The
        # independent yaw/pose and between-sample allowances stay isotropic.
        axes = np.asarray(trace.delay_axes)
        mix = math.sin(heading_span)
        # The earlier scalar allowance includes dx + dy. In each grid axis
        # only one of those terms can contribute at full strength.
        interval_f = intervals - (1. - mix) * axes[:, 1]
        interval_r = intervals - (1. - mix) * axes[:, 0]
        extent_f = length * abs(c) + width * abs(s) + np.maximum(np.r_[0., interval_f], np.r_[interval_f, 0.])
        extent_r = length * abs(s) + width * abs(c) + np.maximum(np.r_[0., interval_r], np.r_[interval_r, 0.])
        layers = prepared_layers or local_path.PreparedObstacleLayers(occupied, remembered)
        refined = layers.axis_clearance(f, r, extent_f, extent_r, first_hit)
        if refined > 1e-10:
            clearance, first_hit, reason = refined, None, None
    if not trace.stopped:
        reason = "braking_incomplete"
    if reason is None and supported_region is not None:
        lo, hi = supported_region
        if any(bounds[0][i] < lo[i] or bounds[1][i] > hi[i] for i in (0, 1)):
            reason = "outside_grid"
    if reason is None and isinstance(terminal_boundary, TravelBudget):
        travel = trace.body_travel_m + trace.delay_travel_error + terminal_boundary.consumed_m
        if travel > terminal_boundary.limit_m + 1e-10:
            reason = "insufficient_room"
    elif reason is None and isinstance(terminal_boundary, StopPlane):
        if support(*terminal_boundary.normal) > terminal_boundary.offset_m + 1e-10:
            reason = "stop_plane"

    terminal_distance = None
    if original_goal is not None:
        terminal_distance = math.dist(original_goal, trace.camera_poses[-1][:2])
        if (reason is None and stop_distance_m > 0.
                and minimum_target_distance(trace, original_goal) < stop_distance_m - 1e-10):
            reason = "target_standoff"
    # Culled layers can contain the closest obstacle, but cannot overlap the sweep.
    clearance = min(clearance, 0.) if cull_obstacles else clearance
    return MotionCheck(reason is None, reason, clearance, first_hit, bounds, terminal_distance)


def minimum_target_distance(trace, goal):
    """Lower bound on camera-to-goal spacing through the complete stop."""
    camera = trace.camera_poses
    if trace.straight_axis is not None:
        closest = tuple(max(min(a, b), min(max(a, b), t))
                        for a, b, t in zip(camera[0][:2], camera[-1][:2], goal))
        return math.dist(closest, goal) - trace.delay_position_error
    linear_error = trace.position_error + trace.half_travel + trace.delay_position_error
    angular_error = trace.half_turn + trace.delay_yaw_error
    error = linear_error + math.hypot(*trace.camera_offset) * min(2., angular_error)
    return min(math.dist(p[:2], goal) for p in camera) - error


def minimum_footprint_distance(trace, point, footprint):
    """Lower bound on the distance from a measured point to the moving body."""
    camera_mode = footprint.mode == "camera_approx"
    f, r, yaw = np.asarray(trace.camera_poses if camera_mode else trace.body_poses).T
    dx, dy = point[0] - f, point[1] - r
    c, s = np.cos(yaw), np.sin(yaw)
    forward, right = c * dx + s * dy, -s * dx + c * dy
    gap = np.hypot(np.maximum(0., np.maximum(forward - footprint.front, -forward - footprint.rear)),
                   np.maximum(0., abs(right) - footprint.half_width))
    ox, oy = trace.camera_offset if camera_mode else (0., 0.)
    radius = math.hypot(max(abs(ox + footprint.front), abs(ox - footprint.rear)),
                        abs(oy) + footprint.half_width)
    errors = np.asarray([linear + radius * min(2., angular) for linear, angular in trace.interval_errors])
    return float(np.min(gap - np.maximum(np.r_[0., errors], np.r_[errors, 0.])))
