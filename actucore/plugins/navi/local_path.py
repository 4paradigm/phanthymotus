"""Footprint checks and bounded fixed-heading local navigation."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .camera import HorizontalIntrinsics

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


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


def shortest_path(allowed, start, goal):
    if not (allowed[start] and allowed[goal]):
        return None
    queue = [(math.dist(start, goal), 0.0, start)]
    cost = {start: 0.0}
    parents = {}
    while queue:
        _, walked, current = heapq.heappop(queue)
        if current == goal:
            route = [goal]
            while route[-1] != start:
                route.append(parents[route[-1]])
            return route[::-1]
        if walked > cost[current]:
            continue
        row, col = current
        for dr, dc in STEPS:
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
                heapq.heappush(queue, (distance + math.dist(nxt, goal), distance, nxt))
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


def observe(frame, geometry, right, forward, resolution, ground):
    if ground is None or not ground[::4, ::4].any():
        return np.zeros(right.shape, dtype=bool), np.zeros(right.shape, dtype=bool)
    height, width = frame.shape
    fx, cx = geometry.horizontal.for_width(width)
    calib_height, fy, cy = geometry.vertical
    scale = height / calib_height
    fy, cy = fy * scale, (cy + 0.5) * scale - 0.5
    u = (np.arange(width) - cx) / fx
    v = (np.arange(height) - cy) / fy
    x, y = u * frame, v[:, None] * frame
    seed = ground[::4, ::4]
    gs = frame[::4, ::4][seed]
    gx, gy = x[::4, ::4][seed], y[::4, ::4][seed]
    a, b, c = np.linalg.lstsq(np.column_stack((gx, gs, np.ones_like(gs))), gy, rcond=None)[0]
    transform = geometry.base_from_optical
    rotation, origin = transform[:3, :3], transform[:3, 3]
    normal = rotation @ np.array([-a, 1.0, -b])
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

    above_floor = (y - a*x - b*frame - c) / normal[2]
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


@dataclass
class PlanCache:
    key: tuple | None = None
    command: tuple[float, float] | None = None

    def translation(self, observation, geometry, half_width, ahead, side, max_age_ms):
        frame = observation.get("frame_id")
        ground = observation.get("ground")
        if frame is None or ground is None or geometry is None:
            return None
        key = (frame, side, round(ahead, 2))
        if self.key != key:
            command = _checked_translation(
                observation["map"], ground, geometry, half_width, ahead, side, max_age_ms)
            self.key, self.command = key, command
        return self.command


def _checked_translation(frame, ground, geometry, half_width, ahead, side, max_age_ms):
    resolution = .05
    row_zero, col_zero = 10, 28
    right, forward = np.meshgrid((np.arange(57) - col_zero) * resolution,
                                 (np.arange(77) - row_zero) * resolution)
    free, occupied = observe(frame, geometry, right, forward, resolution, ground)
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
    # Live front-only planning uses observed obstacles as constraints.
    # Unmeasured cells are not represented as obstacles.
    path = plan_right(grid, start, via, goal, half_cells, allow_unknown=True)
    if path is None or len(path) < 2:
        return None
    dr, dc = (path[1][axis] - start[axis] for axis in (0, 1))
    if dr < 0 or dc < 0:
        return None
    steps = 1
    for i, cell in enumerate(path[2:], 2):
        if cell != (start[0] + dr * i, start[1] + dc * i):
            break
        steps = i
    # Budget the visible straight prefix through a stale frame and watchdog
    # timeout. Replanning may extend it on the next depth update.
    norm = math.hypot(dr, dc)
    horizon_s = max_age_ms / 1000 + .30
    speed = min(.10, steps * resolution * norm / horizon_s)
    return speed * dr / norm, -side * speed * dc / norm
