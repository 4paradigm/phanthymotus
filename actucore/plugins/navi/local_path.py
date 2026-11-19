"""Footprint checks and bounded fixed-heading local navigation."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .camera import HorizontalIntrinsics

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
CARDINAL_STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))
STEPS = CARDINAL_STEPS + ((1, 1), (1, -1), (-1, 1), (-1, -1))
RESOLUTION = .05
START = (10, 28)
SHAPE = (77, 57)
RIGHT, FORWARD = np.meshgrid((np.arange(SHAPE[1]) - START[1]) * RESOLUTION,
                              (np.arange(SHAPE[0]) - START[0]) * RESOLUTION)


def footprint_region(center, half_cells):
    row, col = center
    front, dc, *rear = half_cells
    return np.s_[row-(rear[0] if rear else front):row+front+1, col-dc:col+dc+1]


def allowed_centers(grid, start, half_cells, allow_unknown=False):
    blocked = grid == OCCUPIED if allow_unknown else grid != FREE
    area = footprint_region(start, half_cells)
    blocked[area] = grid[area] == OCCUPIED
    front, dc, *back = half_cells
    rear = back[0] if back else front
    rows, cols = front + rear + 1, 2*dc + 1
    rectangle = np.pad(blocked, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    blocked_count = (rectangle[rows:, cols:] - rectangle[:-rows, cols:]
                     - rectangle[rows:, :-cols] + rectangle[:-rows, :-cols])
    return np.pad(blocked_count == 0, ((rear, front), (dc, dc)), constant_values=False)


def shortest_path(allowed, start, goal, *, later_goal=False, last_goal_col=None, steps=STEPS):
    goal_row, goal_col = goal
    last_row = allowed.shape[0] - 1 if later_goal else goal_row
    last_col = goal_col if last_goal_col is None else last_goal_col
    goals = allowed[goal_row:last_row + 1, goal_col:last_col + 1]
    if not allowed[start] or not goals.any():
        return None

    def distance_to_goals(point):
        row, col = point
        return math.hypot(max(goal_row - row, row - last_row, 0),
                          max(goal_col - col, col - last_col, 0))

    queue = [(distance_to_goals(start), 0.0, start)]
    cost = {start: 0.0}
    parents = {}
    while queue:
        _, walked, current = heapq.heappop(queue)
        row, col = current
        if goal_col <= col <= last_col and goal_row <= row <= last_row:
            route = [current]
            while route[-1] != start:
                route.append(parents[route[-1]])
            return route[::-1]
        if walked > cost[current]:
            continue
        for dr, dc in steps:
            nxt = (row + dr, col + dc)
            if not (0 <= nxt[0] < allowed.shape[0] and 0 <= nxt[1] < allowed.shape[1]):
                continue
            # With a fixed-heading rectangular footprint, check both cardinal
            # translations so a diagonal cannot cut an occupied corner.
            if not (allowed[nxt] and allowed[row + dr, col] and allowed[row, col + dc]):
                continue
            distance = walked + math.hypot(dr, dc)
            if distance < cost.get(nxt, math.inf):
                cost[nxt] = distance
                parents[nxt] = current
                heapq.heappush(queue, (distance + distance_to_goals(nxt), distance, nxt))
    return None


def plan_right(grid, start, right, goal, half_cells, allow_unknown=False):
    allowed = allowed_centers(grid, start, half_cells, allow_unknown)
    # Reach the right of the box before crossing its depth, continue along
    # that side, then return toward the original centerline. All three legs
    # are needed before a right detour is reported.
    first = allowed.copy()
    first[right[0]+1:, :] = False
    passing = allowed.copy()
    passing[:, :right[1]] = False
    passing[:right[0], :] = False
    passing[goal[0]+1:, :] = False
    rejoin = allowed.copy()
    rejoin[:goal[0], :] = False
    right_ahead = (goal[0], right[1])
    pieces = (shortest_path(first, start, right),
              shortest_path(passing, right, right_ahead),
              shortest_path(rejoin, right_ahead, goal))
    if any(piece is None for piece in pieces):
        return None
    return pieces[0] + pieces[1][1:] + pieces[2][1:]


def intrinsics_for(geometry, height, width):
    fx, cx = geometry.horizontal.for_width(width)
    calib_height, fy, cy = geometry.vertical
    scale = height / calib_height
    fy, cy = fy * scale, (cy + 0.5) * scale - 0.5
    return fx, cx, fy, cy


def optical_ground_plane(frame, geometry, ground):
    """Optical plane (normal, offset), where normal · point = offset."""
    if ground is None:
        return None
    rows, cols = np.nonzero(ground[::4, ::4])
    if rows.size < 3:
        return None
    rows, cols = rows * 4, cols * 4
    fx, cx, fy, cy = intrinsics_for(geometry, *frame.shape)
    z = frame[rows, cols]
    x, y = (cols - cx) / fx * z, (rows - cy) / fy * z
    a, b, c = np.linalg.lstsq(np.column_stack((x, z, np.ones_like(z))), y, rcond=None)[0]
    return np.array([-a, 1., -b]), c


def observe(frame, geometry, right, forward, resolution, ground):
    plane = optical_ground_plane(frame, geometry, ground)
    if plane is None:
        return np.zeros(right.shape, dtype=bool), np.zeros(right.shape, dtype=bool)
    height, width = frame.shape
    fx, cx, fy, cy = intrinsics_for(geometry, height, width)
    u = (np.arange(width) - cx) / fx
    v = (np.arange(height) - cy) / fy
    x, y = u * frame, v[:, None] * frame
    optical_normal, c = plane
    transform = geometry.base_from_optical
    rotation, origin = transform[:3, :3], transform[:3, 3]
    normal = rotation @ optical_normal
    if abs(normal[2]) < .5 * np.linalg.norm(normal):
        return np.zeros(right.shape, dtype=bool), np.zeros(right.shape, dtype=bool)
    floor_z = (c + normal @ origin - normal[0]*forward + normal[1]*right) / normal[2]
    # Check several heights below the declared top of the moving envelope.
    free = np.ones(right.shape, dtype=bool)
    top = geometry.height if geometry.height is not None else abs(c / normal[2])
    if not .02 < top <= 2.5:
        return np.zeros(right.shape, dtype=bool), np.zeros(right.shape, dtype=bool)
    for height_m in np.linspace(.02, top, max(4, math.ceil(top/.08))):
        camera_points = (np.stack((forward, -right, floor_z + height_m), axis=-1)
                         - origin) @ rotation
        z = camera_points[..., 2]
        front = z > 0
        denom = np.where(front, z, 1.0)
        cols = np.rint(fx * camera_points[..., 0] / denom + cx).astype(int)
        rows = np.rint(fy * camera_points[..., 1] / denom + cy).astype(int)
        in_image = free & front & (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        distances = frame[rows[in_image], cols[in_image]]
        free[~in_image] = False
        free[in_image] = np.isfinite(distances) & (distances > z[in_image] + 2 * resolution)

    above_floor = (y + optical_normal[0]*x + optical_normal[2]*frame - c) / normal[2]
    hits = np.isfinite(frame) & ~ground & (above_floor >= .02)
    if geometry.height is not None:
        hits &= above_floor <= geometry.height + .05
    body_points = np.column_stack((x[hits], y[hits], frame[hits])) @ rotation.T + origin
    hit_row = np.rint((body_points[:, 0] - forward[0, 0]) / resolution).astype(int)
    hit_col = np.rint((-body_points[:, 1] - right[0, 0]) / resolution).astype(int)
    in_grid = ((hit_row >= 0) & (hit_row < right.shape[0])
               & (hit_col >= 0) & (hit_col < right.shape[1]))
    counts = np.zeros(right.shape, dtype=int)
    np.add.at(counts, (hit_row[in_grid], hit_col[in_grid]), 1)
    return free, counts >= 3


@dataclass(frozen=True)
class Geometry:
    horizontal: HorizontalIntrinsics
    vertical: tuple[int, float, float]
    base_from_optical: np.ndarray
    front: float
    rear: float
    height: float | None


def camera_square(horizontal, vertical, half_width):
    """Plan in the forward camera frame with the declared square footprint."""
    if horizontal is None or vertical is None:
        return None
    # Axis convention only: optical (right, down, forward) to (forward, left, up).
    axes = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                     [0., -1., 0., 0.], [0., 0., 0., 1.]])
    return Geometry(horizontal, vertical, axes, half_width, half_width, None)


def body_square(geometry, vendor):
    """A yaw sweep needs a declared transform about the odometry body origin."""
    if geometry is None or not isinstance(vendor, dict):
        return None
    matrix = vendor.get("body_from_optical")
    invert = matrix is None and vendor.get("cloud_frame_id") in ("body", "base_link")
    if invert:
        matrix = vendor.get("camera_optical_from_cloud")
    try:
        transform = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-6)):
        return None
    rotation = transform[:3, :3]
    if not (np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and math.isclose(np.linalg.det(rotation), 1., abs_tol=1e-5)):
        return None
    if invert:
        transform = np.linalg.inv(transform)
    if transform[0, 2] < .75:  # Unambiguous forward-facing optical axis.
        return None
    return Geometry(geometry.horizontal, geometry.vertical, transform,
                    geometry.front, geometry.rear, geometry.height)


@dataclass(frozen=True)
class Detour:
    command: tuple[float, float] | None
    points: tuple[tuple[float, float], ...]  # metres, (forward, right)
    geometry: Geometry


@dataclass
class PlanCache:
    key: tuple | None = None
    plans: dict[float, Detour | None] = field(default_factory=dict)

    def translation(self, observation, geometry, half_width, ahead, side, max_age_ms,
                    watchdog_ms=300, *, velocity_limits, allow_forward=True):
        frame = observation.get("frame_id")
        ground = observation.get("ground")
        if frame is None or ground is None or geometry is None:
            return None
        key = (frame, round(ahead, 2), max_age_ms, watchdog_ms, observation.get("scan_revision"),
               allow_forward, velocity_limits)
        if self.key != key:
            self.key = key
            self.plans.clear()
        if side not in self.plans:
            scan_grid = observation.get("scan_grid")
            self.plans[side] = _checked_translation(
                observation["map"], ground, geometry, half_width, ahead, side, max_age_ms,
                watchdog_ms, observed_grid=scan_grid, allow_forward=allow_forward,
                velocity_limits=velocity_limits)
        detour = self.plans[side]
        return detour.command if detour is not None else None


def transition_clear(grid, geometry, half_width, horizon_s, velocities):
    """Check the swept rectangle for independent forward and lateral ramps."""
    free, occupied = grid
    forward, right = zip(*((vx * horizon_s, -vy * horizon_s) for vx, vy in velocities))
    rear = -geometry.rear + min(0., *forward)
    front = geometry.front + max(0., *forward)
    left = -half_width + min(0., *right)
    right_edge = half_width + max(0., *right)
    half = RESOLUTION / 2
    if (rear < FORWARD[0, 0] - half or front > FORWARD[-1, 0] + half
            or left < RIGHT[0, 0] - half or right_edge > RIGHT[0, -1] + half):
        return False
    swept = ((FORWARD + half > rear) & (FORWARD - half < front)
             & (RIGHT + half > left) & (RIGHT - half < right_edge))
    added = ((rear < -geometry.rear) & (FORWARD - half < -geometry.rear)
             | (front > geometry.front) & (FORWARD + half > geometry.front)
             | (left < -half_width) & (RIGHT - half < -half_width)
             | (right_edge > half_width) & (RIGHT + half > half_width))
    return not (swept & (occupied | (added & ~free))).any()


def _checked_translation(frame, ground, geometry, half_width, ahead, side, max_age_ms,
                         watchdog_ms=300, *, velocity_limits, observed_grid=None, allow_forward=True):
    (floor_vx, max_vx), (floor_vy, max_vy) = velocity_limits
    if not (max_vx > 0 and max_vy > 0 and floor_vx <= max_vx and floor_vy <= max_vy):
        return None
    resolution = RESOLUTION
    row_zero, col_zero = START
    right, forward = RIGHT, FORWARD
    # A fused scan is expressed at the current pose. Mixing in the original
    # exposure a second time would mark cells free at an outdated pose.
    free, occupied = (observe(frame, geometry, right, forward, resolution, ground)
                      if observed_grid is None else observed_grid)
    if side < 0:
        free, occupied = free[:, ::-1], occupied[:, ::-1]
    grid = np.where(occupied, OCCUPIED, np.where(free, FREE, UNKNOWN))
    plane = ahead * geometry.base_from_optical[0, 2] + geometry.base_from_optical[0, 3]
    obstacle = (occupied & (abs(right) <= half_width) & (abs(forward - plane) <= .10))
    if not obstacle.any():
        return None

    def index(z, x):
        return (round(z / resolution) + row_zero, round(x / resolution) + col_zero)

    edge = float(right[obstacle].max())
    bypass = max(resolution, edge + half_width + resolution)
    start = (row_zero, col_zero)
    via = index(plane, bypass)
    goal = index(plane + geometry.rear + .25, 0)
    if not (row_zero <= via[0] < goal[0] < grid.shape[0] and via[1] < grid.shape[1]):
        return None
    half_cells = tuple(math.floor(half / resolution + .5)
                       for half in (geometry.front, half_width, geometry.rear))
    if half_cells[2] > row_zero or half_cells[1] > col_zero:
        return None
    # Check every footprint against observed obstacles. The search can reach
    # the passing lane at any column outside the obstacle, and rejoin on any
    # clear centerline cell at or beyond the original rejoin distance.
    allowed = allowed_centers(grid, start, half_cells, allow_unknown=True)
    allowed[:row_zero, :] = False
    allowed[:, :col_zero] = False
    if not allow_forward:
        # A near obstacle only permits starting sideways. The diagonal edge
        # also checks this intermediate forward-only grid cell.
        allowed[row_zero + 1, col_zero] = False
    # An additional view can confirm space to bend earlier.
    lane_min = via[1] if observed_grid is None else col_zero + 1
    allowed[via[0]:goal[0], :lane_min] = False
    # A one-cell diagonal needs equal forward and lateral speeds. A longer
    # straight shortcut may still be executable at a different ratio.
    steps = STEPS if max(floor_vx, floor_vy) <= min(max_vx, max_vy) else CARDINAL_STEPS
    path = shortest_path(allowed, start, goal, later_goal=True, steps=steps)
    last_goal_col = None
    if path is None:
        # Keep advancing on the open side if the original line stays blocked.
        goal = (goal[0], col_zero + 1)
        last_goal_col = allowed.shape[1] - 1
        path = shortest_path(allowed, start, goal, later_goal=True,
                             last_goal_col=last_goal_col, steps=steps)
    if path is None or len(path) < 2:
        return None
    # A geometric path may include unobserved space, but the emitted vector
    # must fit inside a measured prefix through staleness, watchdog and sink poll.
    horizon_s = (max_age_ms + watchdog_ms + 50) / 1000
    blocked = grid == OCCUPIED
    unknown = grid == UNKNOWN
    # Cell boundaries, expanded by the moving rectangle, are obstacles in
    # centre coordinates. For unknown cells only the portion outside the
    # original footprint is relevant. Our rays move forward and to one side.
    half_cell = resolution / 2
    f0, f1 = forward - half_cell, forward + half_cell
    r0, r1 = right - half_cell, right + half_cell
    occupied_bounds = (f0[blocked] - geometry.front, f1[blocked] + geometry.rear,
                       r0[blocked] - half_width, r1[blocked] + half_width)
    front = unknown & (f1 > geometry.front)
    flank = unknown & (r1 > half_width)
    front_bounds = (np.maximum(0., f0[front] - geometry.front), f1[front] + geometry.rear,
                    r0[front] - half_width, r1[front] + half_width)
    flank_bounds = (f0[flank] - geometry.front, f1[flank] + geometry.rear,
                    np.maximum(0., r0[flank] - half_width), r1[flank] + half_width)
    def command_to(point):
        row, col = point
        dr, dc = row - row_zero, col - col_zero
        if dr and not allow_forward:
            return None
        norm = math.hypot(dr, dc)
        length = resolution * norm
        df, dx = dr / norm, dc / norm
        # Look ahead along the path only while a straight shortcut is clear
        # of measured obstacles. This preserves the chosen detour's progress.
        if _ray_entry(df, dx, occupied_bounds) <= length:
            return None
        clear = min(length,
                    _ray_entry(df, dx, front_bounds) - 1e-6 if df else math.inf,
                    _ray_entry(df, dx, flank_bounds) - 1e-6 if dx else math.inf)
        speed = min(clear / horizon_s,
                    max_vx / df if df else math.inf, max_vy / dx if dx else math.inf)
        vx, vy = speed * df, speed * dx
        if speed > 0 and (not df or vx >= floor_vx) and (not dx or vy >= floor_vy):
            return vx, -side * vy
        return None

    def detour(path, first, command):
        points = ((0., 0.),) + tuple(((row - row_zero) * resolution, side * (col - col_zero) * resolution)
                                     for row, col in path[first:])
        return Detour(command, points, geometry)

    cardinal = None
    for i in range(len(path) - 1, 0, -1):
        command = command_to(path[i])
        if command is not None:
            if all(command):
                return detour(path, i, command)
            if cardinal is None:
                cardinal = detour(path, i, command)
    if cardinal is not None and cardinal.command[0] == 0.:
        return cardinal

    # Search a separate route that stays on the starting row for at least
    # the travel needed to send a lateral command above the driver's floor.
    first_forward_col = col_zero + max(1, math.ceil(floor_vy * horizon_s / resolution))
    if first_forward_col < allowed.shape[1]:
        allowed[row_zero + 1, :first_forward_col] = False
        side_path = shortest_path(allowed, start, goal, later_goal=True,
                                  last_goal_col=last_goal_col, steps=CARDINAL_STEPS)
        if side_path is not None:
            first_turn = next(i - 1 for i, (row, _) in enumerate(side_path) if row != row_zero)
            command = command_to(side_path[first_turn])
            if command is not None:
                return detour(side_path, first_turn, command)
    return cardinal or detour(path, 1, None)


def _ray_entry(df, dx, bounds):
    """Return distance along the ray to the first intersected rectangle."""
    f0, f1, r0, r1 = bounds
    if not df:
        return r0[(f0 <= 0.) & (f1 >= 0.) & (r1 >= 0.)].clip(min=0.).min(initial=math.inf)
    if not dx:
        return f0[(r0 <= 0.) & (r1 >= 0.) & (f1 >= 0.)].clip(min=0.).min(initial=math.inf)
    enter = np.maximum(0., np.maximum(f0 / df, r0 / dx))
    leave = np.minimum(f1 / df, r1 / dx)
    return enter[enter <= leave].min(initial=math.inf)
