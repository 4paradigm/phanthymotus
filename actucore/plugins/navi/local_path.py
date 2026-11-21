"""Footprint checks and bounded fixed-heading local navigation."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from .camera import HorizontalIntrinsics

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
CARDINAL_STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))
STEPS = CARDINAL_STEPS + ((1, 1), (1, -1), (-1, 1), (-1, -1))
RESOLUTION = .05
MIN_OBSTACLE_HEIGHT_M = .02
APPROXIMATE_CAMERA_MIN_OBSTACLE_HEIGHT_M = .05
START = (10, 28)
SHAPE = (77, 57)
RIGHT, FORWARD = np.meshgrid((np.arange(SHAPE[1]) - START[1]) * RESOLUTION,
                              (np.arange(SHAPE[0]) - START[0]) * RESOLUTION)


def footprint_region(center, half_cells):
    row, col = center
    front, dc, *rear = half_cells
    return np.s_[row-(rear[0] if rear else front):row+front+1, col-dc:col+dc+1]


def _blocked_cells(grid, start, half_cells, allow_unknown=False):
    blocked = grid == OCCUPIED if allow_unknown else grid != FREE
    area = footprint_region(start, half_cells)
    # The starting envelope is already occupied by the robot. Preserve
    # measured hits there, and require evidence for cells entered later.
    blocked[area] = grid[area] == OCCUPIED
    return blocked


def allowed_centers(grid, start, half_cells, allow_unknown=False):
    blocked = _blocked_cells(grid, start, half_cells, allow_unknown)
    front, dc, *back = half_cells
    rear = back[0] if back else front
    rows, cols = front + rear + 1, 2*dc + 1
    rectangle = np.pad(blocked, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    blocked_count = (rectangle[rows:, cols:] - rectangle[:-rows, cols:]
                     - rectangle[rows:, :-cols] + rectangle[:-rows, :-cols])
    return np.pad(blocked_count == 0, ((rear, front), (dc, dc)), constant_values=False)


def shortest_path(allowed, start, goal, *, later_goal=False, last_goal_col=None, steps=STEPS,
                  target_cells=None, target_cost=None, cell_cost=None):
    if not allowed[start]:
        return None
    if target_cells is None:
        goal_row, goal_col = goal
        last_row = allowed.shape[0] - 1 if later_goal else goal_row
        last_col = goal_col if last_goal_col is None else last_goal_col
        if not allowed[goal_row:last_row + 1, goal_col:last_col + 1].any():
            return None
    else:
        rows, cols = np.nonzero(allowed & target_cells)
        if not rows.size:
            return None
        goal_row, last_row = int(rows.min()), int(rows.max())
        goal_col, last_col = int(cols.min()), int(cols.max())

    def distance_to_goals(point):
        if target_cost is not None:
            return 0.
        row, col = point
        return math.hypot(max(goal_row - row, row - last_row, 0),
                          max(goal_col - col, col - last_col, 0))

    queue = [(distance_to_goals(start), 0.0, start)]
    cost = {start: 0.0}
    parents = {}
    best, best_score = None, (math.inf, math.inf)

    def path_to(point):
        path = [point]
        while point != start:
            point = parents[point]
            path.append(point)
        return path[::-1]

    # Scalar access in the inner A* loop is cheaper on Python lists.
    traversable = allowed.tolist()
    row_count, col_count = allowed.shape
    costs = cell_cost.tolist() if cell_cost is not None else [[0.] * col_count] * row_count
    moves = [(dr, dc, math.hypot(dr, dc)) for dr, dc in steps]
    while queue:
        _, walked, current = heapq.heappop(queue)
        row, col = current
        if walked > cost[current]:
            continue
        if (goal_col <= col <= last_col and goal_row <= row <= last_row
                and (target_cells is None or target_cells[current])):
            if target_cost is None:
                return path_to(current)
            score = (target_cost[current], walked)
            if score < best_score:
                best, best_score = current, score
        for dr, dc, step_cost in moves:
            next_row, next_col = row + dr, col + dc
            if not (0 <= next_row < row_count and 0 <= next_col < col_count):
                continue
            # With a fixed-heading rectangular footprint, check both cardinal
            # translations so a diagonal cannot cut an occupied corner.
            if not (traversable[next_row][next_col] and traversable[next_row][col]
                    and traversable[row][next_col]):
                continue
            nxt = (next_row, next_col)
            distance = walked + step_cost * (1. + costs[next_row][next_col])
            if distance < cost.get(nxt, math.inf):
                cost[nxt] = distance
                parents[nxt] = current
                heapq.heappush(queue, (distance + distance_to_goals(nxt), distance, nxt))
    return path_to(best) if best is not None else None


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


def _neighbor_count(mask):
    rows, cols = mask.shape
    padded = np.pad(mask, 1)
    return sum(padded[r:r + rows, c:c + cols]
               for r in range(3) for c in range(3) if (r, c) != (1, 1))


def observe(frame, geometry, right, forward, resolution, ground, *, max_hit_depth=None,
            require_valid_floor=False, check_free_space=True):
    def no_floor():
        if not require_valid_floor:
            return np.zeros(right.shape, dtype=bool), np.zeros(right.shape, dtype=bool)

    plane = optical_ground_plane(frame, geometry, ground)
    if plane is None:
        return no_floor()
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
        return no_floor()
    floor_z = (c + normal @ origin - normal[0]*forward + normal[1]*right) / normal[2]
    # Check several heights below the declared top of the moving envelope.
    check_free_space = check_free_space and geometry.free_space_validated
    free = np.full(right.shape, check_free_space, dtype=bool)
    top = geometry.height if geometry.height is not None else abs(c / normal[2])
    if not .02 < top <= 2.5:
        return no_floor()
    heights = np.linspace(.02, top, max(4, math.ceil(top/.08))) if check_free_space else ()
    # On a verified floor return, require the ray to clear the far edge of
    # this cell at the probe height. A fixed two-cell optical gap would reject
    # valid near-floor rays simply because of their grazing angle.
    depth_slope = rotation[:2, 2] - rotation[2, 2] * normal[:2] / normal[2]
    floor_margin = resolution / 2 * max(1., abs(depth_slope).sum())
    for height_m in heights:
        camera_points = (np.stack((forward, -right, floor_z + height_m), axis=-1)
                         - origin) @ rotation
        z = camera_points[..., 2]
        front = z > 0
        denom = np.where(front, z, 1.0)
        cols = np.rint(fx * camera_points[..., 0] / denom + cx).astype(int)
        rows = np.rint(fy * camera_points[..., 1] / denom + cy).astype(int)
        in_image = free & front & (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        pixels = (rows[in_image], cols[in_image])
        distances = frame[pixels]
        margin = np.where(ground[pixels], floor_margin, 2 * resolution)
        free[~in_image] = False
        free[in_image] = np.isfinite(distances) & (distances > z[in_image] + margin)

    above_floor = (y + optical_normal[0]*x + optical_normal[2]*frame - c) / normal[2]
    hits = np.isfinite(frame) & ~ground & (above_floor >= MIN_OBSTACLE_HEIGHT_M)
    if max_hit_depth is not None:
        hits &= frame <= max_hit_depth
    if geometry.height is not None:
        hits &= above_floor <= geometry.height + .05
    body_points = np.column_stack((x[hits], y[hits], frame[hits])) @ rotation.T + origin
    hit_row = np.rint((body_points[:, 0] - forward[0, 0]) / resolution).astype(int)
    hit_col = np.rint((-body_points[:, 1] - right[0, 0]) / resolution).astype(int)
    in_grid = ((hit_row >= 0) & (hit_row < right.shape[0])
               & (hit_col >= 0) & (hit_col < right.shape[1]))
    cells = (hit_row[in_grid], hit_col[in_grid])
    hit_heights = above_floor[hits][in_grid]
    raised = hit_heights >= geometry.min_obstacle_height
    counts = np.zeros(right.shape, dtype=int)
    np.add.at(counts, cells, raised)
    if geometry.min_obstacle_height > MIN_OBSTACLE_HEIGHT_M:
        uncertain = np.zeros(right.shape, dtype=int)
        np.add.at(uncertain, cells, ~raised)
        free &= uncertain < 3
    occupied = counts >= 3
    # Clearing a suspect hit does not provide evidence of free space.
    free &= ~occupied
    neighbors = _neighbor_count(occupied)
    ends = occupied & (neighbors == 1)
    small = occupied & ((neighbors == 0) | (ends & (_neighbor_count(ends) == 1)))
    if small.any():
        sparse_hits = small[cells] & raised
        sparse_cells = cells[0][sparse_hits], cells[1][sparse_hits]
        heights = hit_heights[sparse_hits]
        highest = np.zeros(right.shape)
        lowest = np.full(right.shape, np.inf)
        np.maximum.at(highest, sparse_cells, heights)
        np.minimum.at(lowest, sparse_cells, heights)
        near = max(.40, geometry.front + 2 * resolution)
        weak = ((counts < 8) & (highest - lowest < .05)
                & (forward**2 + right**2 > near**2))
        occupied &= ~(small & ((highest < .05) | weak))
    return free, occupied


@dataclass(frozen=True)
class Geometry:
    horizontal: HorizontalIntrinsics
    vertical: tuple[int, float, float]
    base_from_optical: np.ndarray
    front: float
    rear: float
    height: float | None
    free_space_validated: bool = True
    min_obstacle_height: float = MIN_OBSTACLE_HEIGHT_M


def camera_square(horizontal, vertical, half_width, *, half_length=None):
    """Plan in the camera frame, defaulting to a square footprint."""
    if horizontal is None or vertical is None:
        return None
    # Axis convention only: optical (right, down, forward) to (forward, left, up).
    axes = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                     [0., -1., 0., 0.], [0., 0., 0., 1.]])
    extent = (half_length if type(half_length) in (int, float)
              and math.isfinite(half_length) and half_length > 0 else half_width)
    # The uncalibrated camera approximation can put a few centimetres of
    # floor residual into obstacle cells. Keep those cells unknown.
    return Geometry(horizontal, vertical, axes, extent, extent, None,
                    min_obstacle_height=APPROXIMATE_CAMERA_MIN_OBSTACLE_HEIGHT_M)


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


def target_in_grid(geometry, forward, right):
    f, left = geometry.base_from_optical[:2] @ (right, 0., forward, 1.)
    return float(f), -float(left)


@dataclass
class PlanCache:
    key: tuple | None = None
    plans: dict[float, Detour | None] = field(default_factory=dict)
    grid_frame: int | None = None
    grid_geometry: Geometry | None = None
    grid: tuple | None = None

    def grid_for(self, observation, geometry):
        scanned = observation.get("scan_grid")
        if scanned is not None:
            return scanned
        projected = observation.get("projected_grid")
        if projected is not None:
            return projected
        project_scan_view = observation.get("project_scan_view")
        if project_scan_view is not None:
            return project_scan_view()
        frame = observation["frame_id"]
        if self.grid_frame != frame or self.grid_geometry is not geometry:
            self.grid = observe(observation["map"], geometry, RIGHT, FORWARD,
                                RESOLUTION, observation["ground"])
            self.grid_frame, self.grid_geometry = frame, geometry
        return self.grid

    def translation(self, observation, geometry, half_width, ahead, side, max_age_ms,
                    watchdog_ms=300, *, velocity_limits, allow_forward=True, allow_return=True):
        detour = self.route(observation, geometry, half_width, ahead, side, max_age_ms,
                            watchdog_ms, velocity_limits=velocity_limits, allow_forward=allow_forward,
                            allow_return=allow_return)
        return detour.command if detour is not None else None

    def route(self, observation, geometry, half_width, ahead, side, max_age_ms,
              watchdog_ms=300, *, velocity_limits=None, allow_forward=True, allow_return=True):
        frame = observation.get("frame_id")
        ground = observation.get("ground")
        if frame is None or ground is None or geometry is None:
            return None
        key = (frame, round(ahead, 2), half_width, max_age_ms, watchdog_ms, observation.get("scan_revision"),
               allow_forward, allow_return, velocity_limits, observation.get("stop_target"),
               observation.get("body_half_width"), observation.get("closest"),
               observation.get("preferred_heading"), observation.get("allow_unknown", False))
        if self.key != key:
            self.key = key
            self.plans.clear()
        if side not in self.plans:
            scan_grid = observation.get("scan_grid")
            self.plans[side] = _checked_translation(
                observation["map"], ground, geometry, half_width, ahead, side, max_age_ms,
                watchdog_ms, observed_grid=scan_grid, allow_forward=allow_forward, allow_return=allow_return,
                velocity_limits=velocity_limits, target=observation.get("stop_target"),
                projected_grid=self.grid_for(observation, geometry),
                body_half_width=observation.get("body_half_width"),
                closest=observation.get("closest", False),
                preferred_heading=observation.get("preferred_heading"),
                allow_unknown=observation.get("allow_unknown", False))
        return self.plans[side]


def _route_grid(grid):
    free, occupied = grid
    return np.where(occupied, OCCUPIED, np.where(free, FREE, UNKNOWN))


def _footprint_cells(geometry, half_width):
    return tuple(math.floor(half / RESOLUTION + .5)
                 for half in (geometry.front, half_width, geometry.rear))


def route_obstacles(grid, geometry, half_width, *, allow_unknown=False):
    return _blocked_cells(_route_grid(grid), START, _footprint_cells(geometry, half_width), allow_unknown)


def observed_waypoint(free, forward, right):
    row = START[0] + round(forward / RESOLUTION)
    col = START[1] + round(right / RESOLUTION)
    return 0 <= row < free.shape[0] and 0 <= col < free.shape[1] and bool(free[row, col])


def transition_clear(grid, geometry, half_width, horizon_s, velocities, *, padding=0.):
    """Check measured obstacles in the swept rectangle of the velocity ramps."""
    occupied = grid[1]
    forward, right = zip(*((vx * horizon_s, -vy * horizon_s) for vx, vy in velocities))
    rear = -geometry.rear + min(0., *forward)
    front = geometry.front + max(0., *forward)
    left = -half_width + min(0., *right)
    right_edge = half_width + max(0., *right)
    half = RESOLUTION / 2 + padding
    swept = ((FORWARD + half > rear) & (FORWARD - half < front)
             & (RIGHT + half > left) & (RIGHT - half < right_edge))
    return not (swept & occupied).any()


def motion_sweep_clear(grid, geometry, half_width, horizon, velocities):
    """Bound footprint rotation and the departure from a straight velocity ramp."""
    turn = max(abs(yaw) for _, _, yaw in velocities) * horizon
    speed = max(math.hypot(vx, vy) for vx, vy, _ in velocities)
    radius = math.hypot(max(geometry.front, geometry.rear), half_width)
    padding = radius * min(2., turn) + speed * horizon * min(2., turn / 2)
    return transition_clear(grid, geometry, half_width, horizon,
                            tuple(v[:2] for v in velocities), padding=padding)


def segment_clear(occupied, geometry, half_width, angle, length, *, heading_tolerance=0.):
    """Check a straight segment with the footprint already facing its heading."""
    cos, sin = math.cos(angle), math.sin(angle)
    f, r = FORWARD[occupied], RIGHT[occupied]
    forward, right = f * cos + r * sin, -f * sin + r * cos
    cell = RESOLUTION / math.sqrt(2)
    if heading_tolerance:
        # Rotating a cell through the settling tolerance moves it by at most
        # this chord. A fresh grid can also round it into the next cell.
        cell += 2 * np.hypot(f, r) * math.sin(heading_tolerance / 2) + RESOLUTION
    return not ((forward >= -geometry.rear - cell)
                & (forward <= length + geometry.front + cell)
                & (abs(right) <= half_width + cell)).any()


def opening_route(frame, occupied, geometry, half_width, plane, target):
    """Aim toward a measured opening straddling the forward axis."""
    if (plane <= geometry.front or target is None or target[0] <= plane
            or geometry.horizontal is None or geometry.vertical is None or geometry.base_from_optical[0, 2] <= 0.):
        return None
    band = occupied & (abs(FORWARD - plane) <= .10)
    cols = np.flatnonzero(band.any(axis=0))
    left, right = cols[cols < START[1]], cols[cols > START[1]]
    if (band[:, START[1]].any() or left.size < 2 or right.size < 2
            or left[-2] != left[-1] - 1 or right[1] != right[0] + 1):
        return None
    lo, hi = RIGHT[0, [left[-1], right[0]]]
    if (hi - lo - RESOLUTION < 2 * half_width or min(-lo, hi) > half_width + RESOLUTION
            or not lo < target[1] * plane / target[0] < hi):
        return None

    middle = (hi + lo) / 2
    # Confirm optical returns beyond the jambs across the proposed width.
    # These rays select a heading; they do not clear the driving envelope.
    height, width = frame.shape
    fx, cx, fy, cy = intrinsics_for(geometry, height, width)
    transform = geometry.base_from_optical
    origin = transform[:3, 3]
    f = np.full(math.ceil(2 * half_width / RESOLUTION) + 1, plane)
    r = np.linspace(middle - half_width, middle + half_width, f.size)
    z = origin[2] + (plane - origin[0]) * transform[2, 2] / transform[0, 2]
    optical = (np.column_stack((f, -r, np.full_like(f, z))) - origin) @ transform[:3, :3]
    if np.any(optical[:, 2] <= 0):
        return None
    u = np.rint(cx + fx * optical[:, 0] / optical[:, 2]).astype(int)
    v = np.rint(cy + fy * optical[:, 1] / optical[:, 2]).astype(int)
    if np.any((u < 0) | (u >= width) | (v < 0) | (v >= height)):
        return None
    measured = frame[v, u]
    if not np.all(np.isfinite(measured) & (measured > optical[:, 2] + 2 * RESOLUTION)):
        return None
    # The caller takes the longest safe prefix and checks it again after turning.
    end = min(plane + geometry.rear + .25, FORWARD[-1, 0] - geometry.front)
    points = tuple((forward, forward * middle / plane)
                   for forward in np.arange(0., end + 1e-9, RESOLUTION))
    return Detour(None, points, geometry)


def _checked_translation(frame, ground, geometry, half_width, ahead, side, max_age_ms,
                         watchdog_ms=300, *, velocity_limits, observed_grid=None, allow_forward=True, allow_return=True,
                         target=None, projected_grid=None, body_half_width=None, closest=False,
                         preferred_heading=None, allow_unknown=False):
    if velocity_limits is not None:
        (floor_vx, max_vx), (floor_vy, max_vy) = velocity_limits
        if not (max_vx > 0 and max_vy > 0 and floor_vx <= max_vx and floor_vy <= max_vy):
            return None
    resolution = RESOLUTION
    row_zero, col_zero = START
    right, forward = RIGHT, FORWARD
    # A fused scan is expressed at the current pose. Use only its obstacle
    # positions when supplied, rather than combining them with the old exposure.
    grid_input = observed_grid if observed_grid is not None else projected_grid
    if grid_input is None:
        grid_input = observe(frame, geometry, right, forward, resolution, ground)
    free, occupied = grid_input
    if free is not None:
        free = free & ~occupied
    if allow_unknown and (free is None or not free.any()):
        return None
    if side < 0:
        free = free[:, ::-1] if free is not None else None
        occupied = occupied[:, ::-1]
    grid = (np.where(occupied, OCCUPIED, FREE) if body_half_width is None
            else _route_grid((free, occupied)))
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
    half_cells = _footprint_cells(geometry, half_width)
    if half_cells[2] > row_zero or half_cells[1] > col_zero:
        return None
    # Check every footprint against observed obstacles. The search can reach
    # the passing lane at any column outside the obstacle, and rejoin on any
    # clear centerline cell at or beyond the original rejoin distance.
    footprint_clear = allowed_centers(grid, start, half_cells, allow_unknown)
    # Slightly prefer centres away from the edge of the hard clearance mask.
    # This never makes a disallowed centre traversable.
    cell_cost = None
    if body_half_width is not None and (velocity_limits is not None or closest):
        padded = np.pad(~footprint_clear, 1, constant_values=True)
        cell_cost = .05 * (padded[:-2, 1:-1].astype(float) + padded[2:, 1:-1]
                           + padded[1:-1, :-2] + padded[1:-1, 2:])
    find_path = partial(shortest_path, cell_cost=cell_cost)
    allowed = footprint_clear.copy()
    allowed[:row_zero, :] = False
    allowed[:, :col_zero] = False
    if not allow_forward:
        # A near obstacle only permits starting sideways. The diagonal edge
        # also checks this intermediate forward-only grid cell.
        allowed[row_zero + 1, col_zero] = False
    # A one-cell diagonal needs equal forward and lateral speeds. A longer
    # straight shortcut may still be executable at a different ratio.
    cardinal_steps = CARDINAL_STEPS if allow_return else tuple(step for step in CARDINAL_STEPS if step[1] >= 0)
    steps = cardinal_steps
    if velocity_limits is None or max(floor_vx, floor_vy) <= min(max_vx, max_vy):
        steps = STEPS if allow_return else tuple(step for step in STEPS if step[1] >= 0)
    path, target_cells = None, None
    if target is not None:
        optical_forward, optical_right, stop, tolerance = target
        target_f, target_r = target_in_grid(geometry, optical_forward, optical_right)
        target_r *= side
        ranges = np.hypot(forward - target_f, right - target_r)
        allowed &= ranges >= stop
        if target_f - stop <= forward[goal[0], 0]:
            target_cells = ranges <= stop + max(tolerance, resolution / math.sqrt(2))
            if allow_unknown:
                target_cells &= free
            path = find_path(allowed, start, goal, steps=steps, target_cells=target_cells)
        if path is None and closest:
            target_cells = ranges < ranges[start] - resolution / 2
            if allow_unknown:
                target_cells &= free
            path = find_path(allowed, start, goal, steps=steps, target_cells=target_cells, target_cost=ranges)
            if path is None:
                return None
    if path is None:
        target_cells = None
        # A fused view can confirm space to bend before the passing lane.
        lane_min = via[1] if observed_grid is None else col_zero + 1
        allowed[via[0]:goal[0], :lane_min] = False
        path = find_path(allowed, start, goal, later_goal=True, steps=steps)
    last_goal_col = None
    if path is None:
        # Keep advancing on the open side if the original line stays blocked.
        goal = (goal[0], col_zero + 1)
        last_goal_col = allowed.shape[1] - 1
        path = find_path(allowed, start, goal, later_goal=True, last_goal_col=last_goal_col, steps=steps)
    partial_route = path is None
    if partial_route:
        # Replan from the passing lane when the current grid has no clear exit.
        goal = via
        last_goal_col = allowed.shape[1] - 1
        path = find_path(allowed, start, goal, later_goal=True, last_goal_col=last_goal_col, steps=steps)
    if path is None and body_half_width is not None and not allow_unknown:
        # Stop at an observed foothold in front of an occluded continuation.
        # The next centre would clear observed obstacles but needs more data.
        collision_clear = allowed_centers(np.where(occupied, OCCUPIED, FREE), start, half_cells)
        frontier = np.zeros_like(allowed)
        frontier[:-1] = allowed[:-1] & collision_clear[1:] & ~footprint_clear[1:]
        frontier[:row_zero + 2] = False
        if frontier.any():
            target_cells = frontier
            path = find_path(allowed, start, goal, steps=steps, target_cells=frontier)
    if allow_unknown:
        # A route may cross unknown cells, but its executable part must end
        # at an observed centre. Replan from there with the next camera view.
        if path is None:
            return None
        end = next((i for i in range(len(path) - 1, 0, -1) if free[path[i]]), 0)
        if end:
            path = path[:end + 1]
        else:
            target_cells = np.zeros_like(allowed)
            target_cells[row_zero + 2:, via[1]:] = free[row_zero + 2:, via[1]:]
            path = find_path(allowed, start, via, steps=steps, target_cells=target_cells)
            if path is None:
                return None
        # A different first motion may be needed by the chassis. Keep any
        # alternate search aimed at the same observed endpoint.
        target_cells = np.zeros_like(allowed)
        target_cells[path[-1]] = True
    reposition = path is None
    if reposition:
        # A reachable lateral position with more room lets the next frame replan.
        cols = slice(col_zero, via[1] + 1)
        forward_cells = np.logical_and.accumulate(footprint_clear[row_zero:, cols], axis=0).sum(axis=0)
        lateral_clear = np.logical_and.accumulate(allowed[row_zero, cols])
        openings = np.flatnonzero(lateral_clear & (forward_cells >= forward_cells[0] + 2))
        if not openings.size:
            return None
        path = [(row_zero, col_zero + offset) for offset in range(int(openings[-1]) + 1)]
    if path is None or len(path) < 2:
        return None

    def detour(path, first, command):
        if reposition:
            path = path[:first + 1]
        points = ((0., 0.),) + tuple(((row - row_zero) * resolution, side * (col - col_zero) * resolution)
                                     for row, col in path[first:])
        return Detour(command, points, geometry)

    if velocity_limits is None:
        return detour(path, len(path) - 1 if reposition else 1, None)

    # The straight command must stay within the checked path prefix through
    # the observation window, watchdog and sink poll.
    horizon_s = (max_age_ms + watchdog_ms + 50) / 1000
    # Expand blocked route cells by the footprint in centre coordinates.
    blocked = _blocked_cells(grid, start, half_cells, allow_unknown)
    half_cell = resolution / 2
    f0, f1 = forward - half_cell, forward + half_cell
    r0, r1 = right - half_cell, right + half_cell
    occupied_bounds = (f0[blocked] - geometry.front, f1[blocked] + geometry.rear,
                       r0[blocked] - half_width, r1[blocked] + half_width)
    def command_to(point):
        if allow_unknown and not free[point]:
            return None
        row, col = point
        dr, dc = row - row_zero, col - col_zero
        if dr and not allow_forward:
            return None
        norm = math.hypot(dr, dc)
        length = resolution * norm
        df, dx = dr / norm, dc / norm
        if target is not None:
            closest = min(length, max(0., df * target_f + dx * target_r))
            if math.hypot(target_f - closest * df, target_r - closest * dx) < stop:
                return None
        # Look ahead only while the shortcut is within the traversable route grid.
        if _ray_entry(df, dx, occupied_bounds) <= length:
            return None
        speed = min(length / horizon_s,
                    max_vx / df if df else math.inf, max_vy / dx if dx else math.inf)
        vx, vy = speed * df, speed * dx
        if (partial_route or target_cells is not None) and vx and vy and not transition_clear(
                (free, occupied), geometry, half_width, horizon_s, ((vx, -vy),)):
            return None
        if speed > 0 and (not df or vx >= floor_vx) and (not dx or vy >= floor_vy):
            return vx, -side * vy
        return None

    cardinal, diagonal, continuous = None, None, None
    for i in range(len(path) - 1, 0, -1):
        command = command_to(path[i])
        if command is not None:
            if all(command) and diagonal is None:
                diagonal = detour(path, i, command)
                if preferred_heading is None:
                    return diagonal
            elif not all(command) and cardinal is None:
                cardinal = detour(path, i, command)
            if preferred_heading is not None and continuous is None:
                heading = math.atan2(-command[1], command[0])
                if abs(math.remainder(heading - preferred_heading, math.tau)) <= math.radians(8):
                    continuous = detour(path, i, command)
    candidate = diagonal or cardinal
    if continuous is not None and math.hypot(*continuous.command) >= .8 * math.hypot(*candidate.command):
        return continuous
    if diagonal is not None:
        return diagonal
    # Prefer an initial lateral stretch of three cells when there is room.
    # A single cell can produce a repeatedly selected but ineffective step.
    lead_cells = max(3, math.ceil(floor_vy * horizon_s / resolution))
    lateral_speed = min(max_vy, lead_cells * resolution / horizon_s)
    if cardinal is not None and cardinal.command[0] == 0.:
        if reposition or abs(cardinal.command[1]) >= lateral_speed - 1e-9:
            return cardinal
    if reposition:
        return None

    # Search for a route that completes the lateral stretch before advancing.
    # A shorter checked step is still usable when the longer lane is blocked.
    first_forward_col = col_zero + lead_cells
    if first_forward_col < allowed.shape[1]:
        allowed[row_zero + 1, :first_forward_col] = False
        side_path = find_path(allowed, start, goal, later_goal=True, last_goal_col=last_goal_col,
                              steps=cardinal_steps, target_cells=target_cells)
        if side_path is not None:
            first_turn = next((i - 1 for i, (row, _) in enumerate(side_path) if row != row_zero),
                              len(side_path) - 1)
            command = command_to(side_path[first_turn])
            if command is not None:
                return detour(side_path, first_turn, command)

    if closest and target is not None and cardinal is None:
        # The geometrically closest route can have an initial shortcut below
        # a chassis velocity floor. Try directly executable alternatives in
        # the same bounded command horizon before reporting no progress.
        max_row = row_zero + (math.floor(max_vx * horizon_s / resolution) if allow_forward else 0)
        max_col = col_zero + math.floor(max_vy * horizon_s / resolution)
        improved = allowed[:max_row + 1, :max_col + 1] & (
            ranges[:max_row + 1, :max_col + 1] < ranges[start] - resolution / 2)
        for point in sorted(zip(*np.nonzero(improved)), key=lambda p: ranges[p]):
            command = command_to(point)
            if command is not None:
                return detour((start, point), 1, command)
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
