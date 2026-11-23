"""Footprint checks and bounded fixed-heading local navigation."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from functools import partial
from struct import pack
from threading import Lock

import numpy as np

from .camera import HorizontalIntrinsics

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
CARDINAL_STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))
STEPS = CARDINAL_STEPS + ((1, 1), (1, -1), (-1, 1), (-1, -1))
RESOLUTION = .05
MIN_OBSTACLE_HEIGHT_M = .02
LOW_OBSTACLE_HEIGHT_M = .10
LOW_COMPONENT_AREA_M2 = .05
LOW_COMPONENT_SPAN_M = .50
APPROXIMATE_CAMERA_MIN_OBSTACLE_HEIGHT_M = .05
# Preserve the forward range while covering the body behind a front camera.
START = (20, 28)
SHAPE = (87, 57)
RIGHT, FORWARD = np.meshgrid((np.arange(SHAPE[1]) - START[1]) * RESOLUTION,
                              (np.arange(SHAPE[0]) - START[0]) * RESOLUTION)


SUPPORTED_REGION = ((FORWARD[0, 0] - RESOLUTION / 2, RIGHT[0, 0] - RESOLUTION / 2),
                    (FORWARD[-1, 0] + RESOLUTION / 2, RIGHT[0, -1] + RESOLUTION / 2))


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
    best_target_cost = (target_cost[rows, cols].min()
                        if target_cost is not None and target_cells is not None else None)

    if target_cells is not None:
        top = target_cost[rows, cols] == best_target_cost if best_target_cost is not None else slice(None)
        nearest_row = int(rows[top].min())
        row, col = start
        if (nearest_row > row and (1, 0) in steps and target_cells[nearest_row, col]
                and (best_target_cost is None or target_cost[nearest_row, col] == best_target_cost)):
            lane = np.s_[row+1:nearest_row+1, col]
            # Every eligible best goal requires at least this much forward travel.
            # A traversable axial path with zero added cost attains that lower bound.
            if allowed[lane].all() and (cell_cost is None or not cell_cost[lane].any()):
                return [(r, col) for r in range(row, nearest_row+1)]

    if best_target_cost is not None:
        # Distance to the bounding box of the best-ranked goals is a lower bound.
        rr, cc = np.ogrid[:allowed.shape[0], :allowed.shape[1]]
        dr = np.maximum(0, np.maximum(rows[top].min() - rr, rr - rows[top].max()))
        dc = np.maximum(0, np.maximum(cols[top].min() - cc, cc - cols[top].max()))
        ranked_distance = np.hypot(dr, dc)

    # Give each grid cell a row-major heap key. Scalar distance and parent
    # lookups dominate repeated side and footprint searches on the control tick.
    row_count, col_count = allowed.shape
    if best_target_cost is None:
        row_gap = [max(goal_row - r, r - last_row, 0) for r in range(row_count)]
        col_gap = [max(goal_col - c, c - last_col, 0) for c in range(col_count)]
        ranked_distance = [[math.hypot(dr, dc) for dc in col_gap] for dr in row_gap]
    # Block the border so diagonal moves can test both cardinal neighbours
    # with the same offsets at every cell, including corners.
    moves = [(dr, dc) for dr, dc in steps if abs(dr) < row_count and abs(dc) < col_count]
    pad_row = max((abs(dr) for dr, _ in moves), default=0)
    pad_col = max((abs(dc) for _, dc in moves), default=0)
    padding = ((pad_row, pad_row), (pad_col, pad_col))
    traversable = np.pad(allowed, padding).ravel().tolist()
    stride = col_count + 2 * pad_col
    origin = pad_row * stride + pad_col
    start_id = origin + start[0] * stride + start[1]
    heuristic = np.pad(ranked_distance, padding).ravel().tolist()
    costs = np.pad(cell_cost, padding).ravel().tolist() if cell_cost is not None else [0.] * len(traversable)
    queue = [(heuristic[start_id], 0.0, start_id)]
    cost = [math.inf] * len(traversable)
    cost[start_id] = 0.
    parents = [-1] * len(traversable)
    best, best_score = None, (math.inf, math.inf)

    def path_to(cell):
        path = []
        while cell != -1:
            path.append(divmod(cell - origin, stride))
            cell = parents[cell]
        return path[::-1]

    offsets = [(dr * stride + dc, dr * stride, dc, math.hypot(dr, dc)) for dr, dc in moves]
    while queue:
        _, walked, current = heapq.heappop(queue)
        if walked > cost[current]:
            continue
        row, col = divmod(current - origin, stride)
        if (goal_col <= col <= last_col and goal_row <= row <= last_row
                and (target_cells is None or target_cells[row, col])):
            if target_cost is None or target_cost[row, col] == best_target_cost:
                return path_to(current)
            score = (target_cost[row, col], walked)
            if score < best_score:
                best, best_score = current, score
        for offset, along, across, step_cost in offsets:
            nxt = current + offset
            if not (traversable[nxt] and traversable[current + along] and traversable[current + across]):
                continue
            distance = walked + step_cost * (1. + costs[nxt])
            if distance < cost[nxt]:
                cost[nxt] = distance
                parents[nxt] = current
                heapq.heappush(queue, (distance + heuristic[nxt], distance, nxt))
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


def _isolated_low_cells(occupied, highest, lowest, resolution):
    """Find small, wholly visible patches without a physical connection to a taller return."""
    tall = occupied & (highest >= LOW_OBSTACLE_HEIGHT_M)
    low = occupied & ~tall
    # Entire contiguous runs already meeting the span limit need no component walk.
    columns = np.arange(low.shape[1])
    left = np.maximum.accumulate(np.where(low, -1, columns), axis=1)
    right = np.minimum.accumulate(np.where(low, low.shape[1], columns)[:, ::-1], axis=1)[:, ::-1]
    long_runs = low & ((right - left - 1) * resolution >= LOW_COMPONENT_SPAN_M)
    pending = set(map(tuple, np.argwhere(low & ~long_runs).tolist()))
    removed = np.zeros_like(occupied)
    rows, cols = occupied.shape
    while pending:
        patch = [pending.pop()]
        supported = False
        for r, c in patch:
            if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                supported = True  # The rest of this patch may lie outside the grid.
            for nr in range(max(0, r-1), min(rows, r+2)):
                for nc in range(max(0, c-1), min(cols, c+2)):
                    if (nr, nc) in pending:
                        pending.remove((nr, nc))
                        patch.append((nr, nc))
                    elif long_runs[nr, nc] or (tall[nr, nc] and lowest[nr, nc] <= highest[r, c] + resolution):
                        supported = True
        if supported or len(patch) * resolution**2 >= LOW_COMPONENT_AREA_M2:
            continue
        rr, cc = zip(*patch)
        span = (math.hypot(max(rr)-min(rr), max(cc)-min(cc)) + 1) * resolution
        if span < LOW_COMPONENT_SPAN_M:
            removed[rr, cc] = True
    return removed


def _floorless_hits(frame):
    """Keep vertically supported range returns when a floor plane is unavailable."""
    step = max(1, frame.shape[0] // 50)
    hits = np.zeros(frame.shape, dtype=bool)
    upper, lower = frame[:-step], frame[step:]
    # A floor usually recedes rapidly up the image; vertical faces keep a
    # similar range across several rows. Keep both ends of a supported pair.
    with np.errstate(invalid="ignore"):
        supported = (upper > 0.) & (lower > 0.) & (
            abs(upper - lower) <= .015 + .005 * np.minimum(upper, lower))
    hits[:-step] |= supported
    hits[step:] |= supported
    return hits


def _floor_patch_evidence(frame, heights, pixels, intrinsics, optical_normal):
    """Look for floor on both sides of a low return and a vertical face below it."""
    fx, cx, fy, cy = intrinsics
    row, col = np.divmod(pixels, frame.shape[1])
    depth = frame[row, col]
    floor_above = np.zeros(len(row), dtype=bool)
    floor_below = floor_above.copy()
    for distance in (.06, .08, .10):
        step = np.ceil(distance * fy / depth).astype(int)
        top = row - step
        bottom = row + step
        floor_above |= (top >= 0) & (abs(heights[np.maximum(top, 0), col]) <= .025)
        floor_below |= (bottom < frame.shape[0]) & (
            abs(heights[np.minimum(bottom, frame.shape[0] - 1), col]) <= .025)
    x, y = (col - cx) * depth / fx, (row - cy) * depth / fy
    down = optical_normal / np.linalg.norm(optical_normal)
    face = np.ones(len(row), dtype=bool)
    for distance in (.0225, .045):
        dx, dy, dz = distance * down
        expected = depth + dz
        r = np.rint(cy + (y + dy) * fy / expected).astype(int)
        c = np.rint(cx + (x + dx) * fx / expected).astype(int)
        face &= (r >= 0) & (r < frame.shape[0]) & (c >= 0) & (c < frame.shape[1])
        r, c = np.clip(r, 0, frame.shape[0] - 1), np.clip(c, 0, frame.shape[1] - 1)
        face &= (abs(frame[r, c] - expected) <= .008) & (heights[r, c] >= .003)
    return floor_above & floor_below, face


class DepthGrid(tuple):
    """Planning visibility and occupancy, with stronger evidence for clearing memory."""

    def __new__(cls, free, occupied, confirmed_free, ignored_low_pixels=None):
        grid = super().__new__(cls, (free, occupied))
        grid.confirmed_free = confirmed_free
        grid.ignored_low_pixels = ignored_low_pixels
        return grid


def confirmed_free(grid):
    return getattr(grid, "confirmed_free", grid[0])


def frozen_mask(mask):
    if not mask.flags.writeable and mask.flags.owndata:
        return mask
    result = mask.copy()
    result.setflags(write=False)
    return result


def frozen_grid(grid):
    """Share the same immutable projection with routing, planning and publication."""
    free, occupied = grid
    clear = confirmed_free(grid)
    frozen_free = frozen_mask(free)
    ignored = getattr(grid, "ignored_low_pixels", None)
    return DepthGrid(frozen_free, frozen_mask(occupied), frozen_free if clear is free else frozen_mask(clear),
                     frozen_mask(ignored) if ignored is not None else None)


def observe(frame, geometry, right, forward, resolution, ground, *, max_hit_depth=None,
            require_valid_depth=False, check_free_space=True):
    valid = np.isfinite(frame) & (frame > 0.) if frame is not None else None
    if geometry.horizontal is None or geometry.vertical is None or valid is None or not valid.any():
        if require_valid_depth:
            return None
        empty = np.zeros(right.shape, dtype=bool)
        return empty, empty.copy()
    plane = optical_ground_plane(frame, geometry, ground)
    height, width = frame.shape
    fx, cx, fy, cy = intrinsics_for(geometry, height, width)
    u = (np.arange(width) - cx) / fx
    v = (np.arange(height) - cy) / fy
    above_floor = None
    if plane is not None:
        optical_normal, c = plane
        normal = geometry.base_from_optical[:3, :3] @ optical_normal
        if abs(normal[2]) >= .5 * np.linalg.norm(normal):
            top = geometry.height if geometry.height is not None else abs(c / normal[2])
            if .02 < top <= 2.5:
                x, y = u * frame, v[:, None] * frame
                above_floor = (y + optical_normal[0]*x + optical_normal[2]*frame - c) / normal[2]
    if above_floor is None:
        hits = valid & _floorless_hits(frame)
    else:
        hits = valid & ~ground & (above_floor >= MIN_OBSTACLE_HEIGHT_M)
        if geometry.height is not None:
            hits &= above_floor <= geometry.height + .05
    if max_hit_depth is not None and above_floor is None:
        valid &= frame <= max_hit_depth
        hits &= valid
    samples = valid if check_free_space else hits
    transform = geometry.base_from_optical
    if transform[0, 0] == transform[0, 1] == 0. and transform[0, 2] == 1.:
        # With an optical axis parallel to grid forward, depth alone bounds
        # the supported rows. Keep boundary ties for the normal rasterizer.
        lower = forward[0, 0] - transform[0, 3] - resolution
        upper = forward[-1, 0] - transform[0, 3] + resolution
        samples = samples & (frame >= lower) & (frame <= upper)
    row, col = np.nonzero(samples)
    z = frame[row, col]
    x, y = u[col] * z, v[row] * z
    floor_context = None
    floor_samples = ground[samples] if above_floor is not None and check_free_space else None
    if above_floor is not None and geometry.min_obstacle_height == APPROXIMATE_CAMERA_MIN_OBSTACLE_HEIGHT_M:
        floor_context = (frame, above_floor, samples, (fx, cx, fy, cy), optical_normal)
    return _project_observed(geometry, right, forward, resolution, x, y, z,
                             hits[samples], check_free_space=check_free_space,
                             heights=above_floor[samples] if above_floor is not None else None,
                             floor_context=floor_context, floor_samples=floor_samples,
                             pixel_samples=samples if check_free_space and above_floor is not None else None,
                             max_hit_depth=max_hit_depth if above_floor is not None else None)


def _project_observed(geometry, right, forward, resolution, x, y, z, hits, *,
                      check_free_space, heights, floor_context=None, floor_samples=None, pixel_samples=None,
                      max_hit_depth=None):
    within_depth = z <= max_hit_depth if max_hit_depth is not None else None
    transform = geometry.base_from_optical
    body_f = x * transform[0, 0] + y * transform[0, 1] + z * transform[0, 2] + transform[0, 3]
    hit_row = np.rint((body_f - forward[0, 0]) / resolution).astype(int)
    in_rows = (hit_row >= 0) & (hit_row < right.shape[0])
    all_rows = in_rows.all()
    # Returns outside forward support cannot vote in any grid cell.
    if not all_rows:
        x, y, z = x[in_rows], y[in_rows], z[in_rows]
    body_l = x * transform[1, 0] + y * transform[1, 1] + z * transform[1, 2] + transform[1, 3]
    hit_col = np.rint((-body_l - right[0, 0]) / resolution).astype(int)
    in_cols = (hit_col >= 0) & (hit_col < right.shape[1])
    if all_rows:
        in_grid = in_cols
    else:
        in_grid = in_rows.copy()
        in_grid[in_rows] = in_cols
    cells = (hit_row[in_grid], hit_col[in_cols])
    possible_hits = hits[in_grid]
    raised = possible_hits
    if heights is not None:
        heights = heights[in_grid]
        raised = possible_hits & (heights >= geometry.min_obstacle_height)
    cell_ids = cells[0] * right.shape[1] + cells[1]
    counts = np.bincount(cell_ids[raised], minlength=right.size).reshape(right.shape)
    occupied = counts >= 3
    ignored_low_pixels = None
    removed = None
    raw_removed = None
    if heights is not None and possible_hits.any():
        top = np.zeros(right.shape)
        bottom = np.full(right.shape, np.inf)
        np.maximum.at(top.ravel(), cell_ids[raised], heights[raised])
        np.minimum.at(bottom.ravel(), cell_ids[raised], heights[raised])
        removed = _isolated_low_cells(occupied, top, bottom, resolution)
        raw_removed = removed
        if pixel_samples is not None:
            # The pixel corridor can also see returns below the raster vote
            # threshold. Classify these without joining separate occupied patches.
            weak = possible_hits & ~raised
            np.maximum.at(top.ravel(), cell_ids[weak], heights[weak])
            np.minimum.at(bottom.ravel(), cell_ids[weak], heights[weak])
            present = top > 0.
            if (present & ~occupied).any():
                raw_removed = removed | _isolated_low_cells(present, top, bottom, resolution)
    if within_depth is not None:
        # The range cap belongs to this query. A large platform straddling it
        # must be classified using its complete visible footprint first.
        keep = within_depth[in_grid]
        in_grid[in_grid] = keep
        cells = cells[0][keep], cells[1][keep]
        cell_ids, possible_hits, raised = cell_ids[keep], possible_hits[keep], raised[keep]
        if heights is not None:
            heights = heights[keep]
        counts = np.bincount(cell_ids[raised], minlength=right.size).reshape(right.shape)
        occupied = counts >= 3
    if pixel_samples is not None and raw_removed is not None and raw_removed.any():
        votes = possible_hits & (heights < LOW_OBSTACLE_HEIGHT_M) & raw_removed[cells]
        if votes.any():
            ignored_low_pixels = np.zeros(pixel_samples.shape, dtype=bool)
            ignored_low_pixels.flat[np.flatnonzero(pixel_samples)[in_grid][votes]] = True
    neighbors = _neighbor_count(occupied)
    ends = occupied & (neighbors == 1)
    small = occupied & ((neighbors == 0) | (ends & (_neighbor_count(ends) == 1)))
    if small.any():
        near = max(.40, geometry.front + 2 * resolution)
        far = forward**2 + right**2 > near**2
        weak = (counts < 8) & far
        if heights is None:
            occupied &= ~(small & weak)
        else:
            sparse_hits = small[cells] & raised
            sparse_cells = cells[0][sparse_hits], cells[1][sparse_hits]
            highest = np.zeros(right.shape)
            lowest = np.full(right.shape, np.inf)
            np.maximum.at(highest, sparse_cells, heights[sparse_hits])
            np.minimum.at(lowest, sparse_cells, heights[sparse_hits])
            weak = small & weak & (highest - lowest < .05)
            occupied &= ~(small & (highest < .05))
            if floor_context is None:
                occupied &= ~weak
            else:
                # Only suppress an extra near-floor patch with positive floor evidence.
                # A thin upright face can also rescue a return from the weak-hit rule.
                low = small & far & (highest <= APPROXIMATE_CAMERA_MIN_OBSTACLE_HEIGHT_M + .01)
                candidates = (low | weak)[cells] & raised
                if candidates.any():
                    frame, above_floor, samples, intrinsics, optical_normal = floor_context
                    pixels = np.flatnonzero(samples)[in_grid][candidates]
                    floor, face = _floor_patch_evidence(frame, above_floor, pixels, intrinsics, optical_normal)
                    ids = cell_ids[candidates]
                    floor_votes = np.bincount(ids[floor], minlength=right.size).reshape(right.shape)
                    face_votes = np.bincount(ids[face], minlength=right.size).reshape(right.shape)
                    occupied &= ~((weak | (low & (floor_votes >= .8 * counts))) & (face_votes < 2))
    if removed is not None:
        occupied &= ~removed
    # Only metric returns establish visibility. Unseen gaps, blind regions,
    # and cells behind the last measured surface retain the unknown state.
    free = np.zeros(right.shape, dtype=bool)
    if not check_free_space:
        return free, occupied
    free[cells] = True
    free &= ~occupied

    # Dropping a possible obstacle only changes current occupancy. A fitted
    # floor needs local support, and floorless clearance needs denser returns.
    if floor_samples is None:
        clear_samples, minimum_clear_samples = ~possible_hits, 8
    else:
        clear_samples, minimum_clear_samples = floor_samples[in_grid], 3
    clear_votes = np.bincount(cell_ids[clear_samples], minlength=right.size).reshape(right.shape)
    clear = clear_votes >= minimum_clear_samples
    clear[cells[0][possible_hits], cells[1][possible_hits]] = False
    return DepthGrid(free, occupied, clear, ignored_low_pixels)


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

    @property
    def half_length(self):
        return (self.front + self.rear) / 2


def camera_square(horizontal, vertical, half_width, *, half_length=None, at_front=False):
    """Build a camera-frame approximation of the chassis footprint."""
    if horizontal is None or vertical is None:
        return None
    # Axis convention only: optical (right, down, forward) to (forward, left, up).
    axes = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                     [0., -1., 0., 0.], [0., 0., 0., 1.]])
    extent = (half_length if type(half_length) in (int, float)
              and math.isfinite(half_length) and half_length > 0 else half_width)
    # The uncalibrated camera approximation can put a few centimetres of
    # floor residual into obstacle cells. Keep those cells unknown.
    front, rear = (0., 2 * extent) if at_front else (extent, extent)
    return Geometry(horizontal, vertical, axes, front, rear, None,
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
                    geometry.half_length, geometry.half_length, geometry.height)


@dataclass(frozen=True)
class Detour:
    command: tuple[float, float] | None
    points: tuple[tuple[float, float], ...]  # metres, (forward, right)
    geometry: Geometry
    preview: tuple[tuple[float, float], ...] = ()
    reachable_prefix: bool = False
    topology: tuple[tuple[float, float], ...] = ()  # original ordered nodes, in the observed grid


def target_in_grid(geometry, forward, right):
    f, left = geometry.base_from_optical[:2] @ (right, 0., forward, 1.)
    return float(f), -float(left)


class DepthProjection:
    """Keep raw projections for one received frame, shared by both timers."""

    def __init__(self, frame, ground):
        self.frame, self.ground = frame, ground
        self._lock = Lock()
        self._grids = {}
        self._received = False

    @classmethod
    def received(cls, frame, ground):
        # Only the decoder's bytes-backed frames are safe to share across timers.
        immutable = isinstance(getattr(frame.base, "base", None), bytes) and not frame.flags.writeable
        if immutable and ground is not None:
            ground = np.frombuffer(ground.tobytes(), dtype=ground.dtype).reshape(ground.shape)
        projection = cls(frame, ground)
        projection._received = immutable
        return projection

    @staticmethod
    def for_observation(depth):
        projection = depth.get("grid_projection")
        if (isinstance(projection, DepthProjection) and projection._received
                and projection.frame is depth["map"] and projection.ground is depth.get("ground")):
            return projection
        return DepthProjection(depth["map"], depth.get("ground"))

    def grid_for(self, geometry, *, require_valid_depth, check_free_space):
        geometry_key = (geometry.horizontal, geometry.vertical, geometry.front, geometry.height,
                        geometry.min_obstacle_height, geometry.base_from_optical.tobytes())
        key = (geometry_key, check_free_space)
        with self._lock:
            # The free-space query only changes free cells. Its occupied mask
            # can also serve the command timer's occupancy-only checks.
            full = (geometry_key, True)
            if full in self._grids:
                key = full
            cached = key in self._grids
            grid = self._grids.get(key)
        if not cached:
            # The command timer can check occupancy while planning computes
            # free space for the same image.
            projected = observe(self.frame, geometry, RIGHT, FORWARD, RESOLUTION, self.ground,
                                require_valid_depth=True, check_free_space=check_free_space)
            if projected is not None:
                projected = frozen_grid(projected)
            with self._lock:
                if key not in self._grids and len(self._grids) == 4:
                    del self._grids[next(iter(self._grids))]
                grid = self._grids.setdefault(key, projected)
        if grid is None and not require_valid_depth:
            empty = np.zeros(SHAPE, dtype=bool)
            return empty, empty.copy()
        return grid


def project_observation(observation, geometry, *, require_valid_depth=False, check_free_space=True):
    projection = observation.get("grid_projection")
    if projection is not None:
        return projection.grid_for(geometry, require_valid_depth=require_valid_depth,
                                   check_free_space=check_free_space)
    return observe(observation["map"], geometry, RIGHT, FORWARD, RESOLUTION, observation.get("ground"),
                   require_valid_depth=require_valid_depth, check_free_space=check_free_space)


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
        if observation.get("grid_projection") is not None:
            return project_observation(observation, geometry)
        frame = observation["frame_id"]
        if self.grid_frame != frame or self.grid_geometry is not geometry:
            self.grid = observe(observation["map"], geometry, RIGHT, FORWARD,
                                RESOLUTION, observation.get("ground"))
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
        if frame is None or observation.get("map") is None or geometry is None:
            return None
        key = (frame, round(ahead, 2), half_width, max_age_ms, watchdog_ms, observation.get("scan_revision"),
               allow_forward, allow_return, velocity_limits, observation.get("stop_target"),
               observation.get("body_half_width"), observation.get("closest"),
               observation.get("preferred_heading"), observation.get("allow_unknown", False),
               observation.get("preview_occluded", False), observation.get("finite_turn", False))
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
                closest=observation.get("closest", False), finite_turn=observation.get("finite_turn", False),
                preferred_heading=observation.get("preferred_heading"),
                allow_unknown=observation.get("allow_unknown", False),
                require_observed_endpoint=(ground is not None and geometry.free_space_validated
                                           and not observation.get("preview_occluded")))
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


def occluded_cells(free, occupied, geometry):
    """Mark unknown cells behind the closest measured obstacle on each camera ray."""
    if not occupied.any():
        return np.zeros_like(occupied)
    origin = geometry.base_from_optical[:2, 3]
    forward, right = FORWARD - origin[0], RIGHT + origin[1]
    ranges = np.hypot(forward, right)
    angles = np.arctan2(right, forward)
    ray_count = 360
    rays = (angles + math.pi / 2) * (ray_count / math.pi)
    near = np.full(ray_count, np.inf)
    hit = occupied & (forward > 0.)
    radius = RESOLUTION / 2
    width = np.arctan2(radius, ranges[hit]) * (ray_count / math.pi)
    starts = np.maximum(0, np.floor(rays[hit] - width).astype(int))
    ends = np.minimum(ray_count, np.floor(rays[hit] + width).astype(int) + 1)
    for start, end, distance in zip(starts, ends, ranges[hit]):
        np.minimum(near[start:end], distance, out=near[start:end])
    nearest = near[np.clip(rays.astype(int), 0, ray_count - 1)]
    return (forward > 0.) & ~free & ~occupied & (ranges > nearest + RESOLUTION)


def visible_prefix(points, occluded):
    """Return the count of waypoints reachable before crossing a shadow."""
    if len(points) > 1:
        samples = np.asarray(points)
        delta = np.diff(samples, axis=0)
        # Each sufficiently short segment uses only its two endpoints in
        # the loop below. Check that identical sample set in one array; DWA
        # has many such segments in every complete braking prediction.
        if np.all(np.hypot(delta[:, 0], delta[:, 1]) < RESOLUTION / 2 * (1 - 1e-12)):
            rows = START[0] + np.rint(samples[:, 0] / RESOLUTION).astype(int)
            cols = START[1] + np.rint(samples[:, 1] / RESOLUTION).astype(int)
            valid = (rows >= 0) & (rows < SHAPE[0]) & (cols >= 0) & (cols < SHAPE[1])
            hits = np.flatnonzero(valid)
            hits = hits[occluded[rows[valid], cols[valid]]]
            return max(1, int(hits[0])) if hits.size else len(points)
    for index, (start, end) in enumerate(zip(points, points[1:]), 1):
        count = max(2, math.ceil(2 * math.dist(start, end) / RESOLUTION) + 1)
        f, r = np.linspace(start, end, count).T
        rows = START[0] + np.rint(f / RESOLUTION).astype(int)
        cols = START[1] + np.rint(r / RESOLUTION).astype(int)
        valid = (rows >= 0) & (rows < SHAPE[0]) & (cols >= 0) & (cols < SHAPE[1])
        if occluded[rows[valid], cols[valid]].any():
            return index
    return len(points)


def _transition_hits(occupied, geometry, half_width, horizon_s, velocities, padding):
    forward, right = zip(*((vx * horizon_s, -vy * horizon_s) for vx, vy in velocities))
    rear = -geometry.rear + min(0., *forward)
    front = geometry.front + max(0., *forward)
    left = -half_width + min(0., *right)
    right_edge = half_width + max(0., *right)
    half = RESOLUTION / 2 + padding
    swept = ((FORWARD + half > rear) & (FORWARD - half < front)
             & (RIGHT + half > left) & (RIGHT - half < right_edge))
    return swept & occupied


def transition_clear(grid, geometry, half_width, horizon_s, velocities, *, padding=0.):
    """Check measured obstacles in the swept rectangle of the velocity ramps."""
    return not _transition_hits(grid[1], geometry, half_width, horizon_s, velocities, padding).any()


@dataclass(frozen=True)
class ObstacleSquares:
    """Original map cells expressed in the local (forward, right) frame."""

    centers: np.ndarray
    yaw: float
    alignment_error: float = 0.


@dataclass(frozen=True)
class GridTransform:
    """An earlier grid expressed in the current forward/right coordinates."""

    forward: float = 0.
    right: float = 0.
    yaw: float = 0.
    shift_error: float = 0.
    turn_error: float = 0.

    def points(self, points):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.asarray(points) @ np.array([[c, s], [-s, c]]) + (self.forward, self.right)

    def source_points(self, points):
        """Express current-grid points in this exposure's coordinates."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (np.asarray(points) - (self.forward, self.right)) @ np.array([[c, -s], [s, c]])

    def error(self, radius):
        return self.shift_error + radius * min(2., self.turn_error)

    def squares(self, cells):
        cells = occupied_squares(cells)
        if cells is None or not cells.centers.size:
            return None
        radius = np.linalg.norm(cells.centers, axis=1).max() + RESOLUTION / math.sqrt(2)
        return ObstacleSquares(self.points(cells.centers), cells.yaw + self.yaw,
                               cells.alignment_error + self.error(radius))



def array_revision(array):
    if array is None:
        return None
    # The depth decoder owns immutable bytes. Hash the backing allocation so
    # Python reuses its digest when the two timers inspect the same exposure.
    backing = array.base
    if isinstance(backing, np.ndarray):
        backing = backing.base
    if not (isinstance(backing, bytes) and array.flags.c_contiguous and array.nbytes == len(backing)):
        backing = array.tobytes()
    return array.shape, array.dtype.str, hash(backing)


def clearance_margin(half_width):
    """Physical clearance on each side of the adopted chassis."""
    return .10 * half_width


def occupied_squares(occupied):
    """Keep the original extent and orientation of each measured occupied cell."""
    if isinstance(occupied, ObstacleSquares) or occupied is None:
        return occupied
    rows, cols = np.nonzero(occupied)
    return ObstacleSquares(np.column_stack((FORWARD[rows, cols], RIGHT[rows, cols])), 0.)


def rectangle_clearance(poses, squares, front, rear, width, *, padding=0., cell_size=RESOLUTION):
    """Minimum Euclidean separation and first contact sample for closed rectangles.

    The chassis is expressed about its reference pose. Work in the obstacle
    axes so the same calculation handles raster cells and rotated map memory.
    Padding bounds the continuous motion between the checked samples.
    """
    if not len(poses) or squares is None or not squares.centers.size:
        return math.inf, None
    padding = np.broadcast_to(padding, (len(poses),)) + squares.alignment_error
    f, r, yaw = np.asarray(poses).T
    length, shift = (front + rear) / 2, (front - rear) / 2
    f, r = f + shift * np.cos(yaw), r + shift * np.sin(yaw)
    c, s = math.cos(squares.yaw), math.sin(squares.yaw)
    ox, oy = squares.centers.T
    ox, oy = c * ox + s * oy, -s * ox + c * oy
    x, y = c * f + s * r, -s * f + c * r
    c, s = np.cos(yaw - squares.yaw), np.sin(yaw - squares.yaw)
    half = cell_size / 2
    # Any point in the first body paired with a cell centre gives an upper
    # bound on the minimum distance. Discard only cells whose bounding-box
    # lower bound exceeds both that distance and the contact padding.
    if len(ox) > 32:
        dx, dy = ox - x[0], oy - y[0]
        u = np.maximum(abs(c[0]*dx + s[0]*dy) - length, 0.)
        v = np.maximum(abs(-s[0]*dx + c[0]*dy) - width, 0.)
        cutoff = max(float(np.hypot(u, v).min()) - padding[0], 0.) + padding.max() + 1e-10
        ex, ey = length*abs(c) + width*abs(s) + half, length*abs(s) + width*abs(c) + half
        gap_x = np.maximum(0., np.maximum((x-ex).min() - ox, ox - (x+ex).max()))
        gap_y = np.maximum(0., np.maximum((y-ey).min() - oy, oy - (y+ey).max()))
        near = gap_x*gap_x + gap_y*gap_y <= cutoff*cutoff
        ox, oy = ox[near], oy[near]
    minimum = math.inf
    batch = max(1, 32768 // len(ox))
    start, end = 0, 1
    while start < len(f):
        a, b = c[start:end, None], s[start:end, None]
        dx, dy = ox - x[start:end, None], oy - y[start:end, None]
        if np.all(b == 0.):
            distance = np.hypot(np.maximum(abs(dx) - length - half, 0.),
                                np.maximum(abs(dy) - width - half, 0.))
        else:
            u, v = a * dx + b * dy, -b * dx + a * dy
            cell_extent = half * (abs(a) + abs(b))
            # Four separating axes also detect crossing edges with no interior vertex.
            overlap = ((abs(dx) <= length * abs(a) + width * abs(b) + half)
                       & (abs(dy) <= length * abs(b) + width * abs(a) + half)
                       & (abs(u) <= length + cell_extent) & (abs(v) <= width + cell_extent))
            # For disjoint polygons the closest pair includes a vertex. Check
            # both vertex-to-box directions (including corner-to-corner gaps).
            distance2 = np.full(dx.shape, np.inf)
            for i, j in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                du = np.maximum(abs(u + half * (i * a + j * b)) - length, 0.)
                dv = np.maximum(abs(v + half * (-i * b + j * a)) - width, 0.)
                px = np.maximum(abs(dx - i * length * a + j * width * b) - half, 0.)
                py = np.maximum(abs(dy - i * length * b - j * width * a) - half, 0.)
                distance2 = np.minimum(distance2, np.minimum(du*du + dv*dv, px*px + py*py))
            distance = np.sqrt(np.where(overlap, 0., distance2))
        nearest = distance.min(axis=1) - padding[start:end]
        hits = np.flatnonzero(nearest <= 1e-10)
        if hits.size:
            return 0., start + int(hits[0])
        minimum = min(minimum, float(nearest.min()))
        start, end = end, min(len(f), end + batch)
    return minimum, None


_PREPARED_PAIR_BUDGET = 32768
_CONTACT_EPS = 1e-10


class PreparedObstacleLayers:
    """Packed original squares from one registered DWA scene."""

    def __init__(self, occupied, remembered):
        self.layers = tuple((square, reason)
                            for evidence, reason in ((occupied, "footprint_collision"),
                                                     (remembered, "remembered_obstacle"))
                            for layer in (evidence if isinstance(evidence, tuple) else (evidence,))
                            if (square := occupied_squares(layer)) is not None
                            and len(square.centers))
        self._first_pose = None
        if not self.layers:
            return
        self.yaws = np.asarray([square.yaw for square, _ in self.layers])
        self.cos_yaws = np.asarray([math.cos(yaw) for yaw in self.yaws])
        self.sin_yaws = np.asarray([math.sin(yaw) for yaw in self.yaws])
        self.counts = [len(square.centers) for square, _ in self.layers]
        ids = self.layer_index = np.repeat(np.arange(len(self.counts)), self.counts)
        sx, sy = np.concatenate([square.centers for square, _ in self.layers]).T
        self.grid_f, self.grid_r = sx, sy
        self.ox = self.cos_yaws[ids] * sx + self.sin_yaws[ids] * sy
        self.oy = -self.sin_yaws[ids] * sx + self.cos_yaws[ids] * sy
        self.error = np.asarray([square.alignment_error for square, _ in self.layers])
        self.native_shapes = (np.isfinite(self.error).all() and np.isfinite(sx).all() and np.isfinite(sy).all()
                              and all(square.centers.dtype == np.float64 for square, _ in self.layers))

    def axis_clearance(self, f, r, extent_f, extent_r, first_hit):
        """Prove separation using grid-axis bounds, including each source's cell size and registration."""
        cell = ((abs(self.cos_yaws) + abs(self.sin_yaws)) * (RESOLUTION / 2) + self.error)[self.layer_index]
        df = abs(f[first_hit] - self.grid_f) - extent_f[first_hit] - cell
        dr = abs(r[first_hit] - self.grid_r) - extent_r[first_hit] - cell
        if np.any((df <= _CONTACT_EPS) & (dr <= _CONTACT_EPS)):
            return 0.
        # The initial separation bounds how many cells need a full time series.
        minimum = np.hypot(df.clip(0.), dr.clip(0.)).min()
        df = np.maximum((f-extent_f).min()-self.grid_f, self.grid_f-(f+extent_f).max()) - cell
        dr = np.maximum((r-extent_r).min()-self.grid_r, self.grid_r-(r+extent_r).max()) - cell
        nearby = np.flatnonzero((df <= minimum) & (dr <= minimum))
        batch = max(1, _PREPARED_PAIR_BUDGET // len(f))
        for start in range(0, len(nearby), batch):
            indices = nearby[start:start+batch]
            df = abs(f[:, None] - self.grid_f[indices]) - extent_f[:, None] - cell[indices]
            dr = abs(r[:, None] - self.grid_r[indices]) - extent_r[:, None] - cell[indices]
            minimum = min(minimum, np.hypot(df.clip(0.), dr.clip(0.)).min())
            if minimum <= _CONTACT_EPS:
                break
        return float(minimum)

    def clearance(self, poses, f, r, yaw, front, rear, width, padding):
        if not self.layers:
            return math.inf, None, None
        # The native tail switches formulas as a group for each per-layer block.
        # If it could split any layer into more than one tail block, retain it.
        if not self.native_shapes or max(self.counts) * (len(f) - 1) > _PREPARED_PAIR_BUDGET:
            clearance, first_hit, cause = math.inf, None, None
            for squares, reason in self.layers:
                distance, hit = rectangle_clearance(
                    poses, squares, front, rear, width, padding=padding)
                clearance = min(clearance, distance)
                if hit is not None and (first_hit is None or hit < first_hit):
                    first_hit, cause = hit, reason
            return clearance, first_hit, cause

        x = self.cos_yaws * f[:, None] + self.sin_yaws * r[:, None]
        y = -self.sin_yaws * f[:, None] + self.cos_yaws * r[:, None]
        delta = yaw[:, None] - self.yaws
        c, s = np.cos(delta), np.sin(delta)
        length, half = (front + rear) / 2, RESOLUTION / 2
        ids = self.layer_index
        padding = np.broadcast_to(padding, (len(f),))
        cell_error = self.error[ids]
        cell_pad = padding.max() + cell_error
        # Include the actual broadcast operands so cache reuse preserves
        # even a numerical kernel's last-bit differences across trace shapes.
        key = (pack("!7d", f[0], r[0], yaw[0], front, rear, width, half),
               x[0].tobytes(), y[0].tobytes(), c[0].tobytes(), s[0].tobytes())
        initial = self._first_pose
        if initial is None or initial[0] != key:
            unpadded = self.point_distances(
                self.ox-x[0, ids], self.oy-y[0, ids], c[0, ids], s[0, ids],
                s[0, ids] == 0., length, half, width)
            initial = self._first_pose = (key, unpadded)
        first = initial[1] - (padding[0] + cell_error)
        hits = np.flatnonzero(first <= _CONTACT_EPS)
        if hits.size:
            return 0., 0, self.layers[ids[hits[0]]][1]
        minimum = float(first.min())
        if len(f) == 1:
            return minimum, None, None

        # The expanded rectangle in each source's axes bounds all true separations.
        # The exact first sample gives an upper bound for the whole trace.
        ex = length*abs(c) + width*abs(s) + half
        ey = length*abs(s) + width*abs(c) + half
        gx = np.maximum(0., np.maximum((x-ex).min(axis=0)[ids] - self.ox,
                                       self.ox - (x+ex).max(axis=0)[ids]))
        gy = np.maximum(0., np.maximum((y-ey).min(axis=0)[ids] - self.oy,
                                       self.oy - (y+ey).max(axis=0)[ids]))
        gap = np.hypot(gx, gy)
        keep = np.flatnonzero(gap <= max(minimum, _CONTACT_EPS) + cell_pad + _CONTACT_EPS)
        axis_aligned_tail = np.all(s[1:] == 0., axis=0)
        # Distance from one real cell vertex at the terminal pose to the body
        # bounds the full-trace minimum from above. Use it only for pruning;
        # only the original full evaluator can report a distance or a contact.
        upper_bound = minimum
        if keep.size:
            source = ids[keep]
            dx = self.ox[keep] - x[-1, source]
            dy = self.oy[keep] - y[-1, source]
            a, b = c[-1, source], s[-1, source]
            u, v = a * dx + b * dy, -b * dx + a * dy
            i, j = -np.copysign(1., dx), -np.copysign(1., dy)
            du = np.maximum(abs(u + half * (i*a + j*b)) - length, 0.)
            dv = np.maximum(abs(v + half * (-i*b + j*a)) - width, 0.)
            witness = np.sqrt(du*du + dv*dv)
            aligned = axis_aligned_tail[source]
            if aligned.any():
                witness[aligned] = np.hypot(np.maximum(abs(dx[aligned]) - length - half, 0.),
                                             np.maximum(abs(dy[aligned]) - width - half, 0.))
            upper = witness - (padding[-1] + cell_error[keep])
            upper_bound = min(minimum, float(upper.min()) + _CONTACT_EPS)
            keep = keep[gap[keep] <= max(upper_bound, _CONTACT_EPS) + cell_pad[keep] + _CONTACT_EPS]
        first_hit, hit_layer = len(f), len(self.layers)
        batch = max(1, _PREPARED_PAIR_BUDGET // (len(f) - 1))
        for start in range(0, len(keep), batch):
            cell = keep[start:start+batch]
            layer = ids[cell]
            # A contact at first_hit still needs to check earlier layers for a tie.
            stop = min(len(f), first_hit + 1)
            dx = self.ox[cell] - x[1:stop, layer]
            dy = self.oy[cell] - y[1:stop, layer]
            gap_x = np.maximum(abs(dx) - ex[1:stop, layer], 0.)
            gap_y = np.maximum(abs(dy) - ey[1:stop, layer], 0.)
            limit = max(min(minimum, upper_bound), _CONTACT_EPS) + cell_pad[cell] + _CONTACT_EPS
            row, col = np.nonzero(gap_x*gap_x + gap_y*gap_y <= limit*limit)
            if not row.size:
                continue
            source = layer[col]
            time_index = row + 1
            distance = self.point_distances(dx[row, col], dy[row, col],
                                            c[time_index, source], s[time_index, source],
                                            axis_aligned_tail[source], length, half, width) - (padding[time_index] + cell_error[cell[col]])
            minimum = min(minimum, float(distance.min()))
            hits = np.flatnonzero(distance <= _CONTACT_EPS)
            if hits.size:
                i = hits[0]  # Pose-major and source-major order within each slab.
                if (time_index[i], source[i]) < (first_hit, hit_layer):
                    first_hit, hit_layer = int(time_index[i]), int(source[i])
                minimum = 0.
        return (minimum, None, None) if first_hit == len(f) else (0., first_hit, self.layers[hit_layer][1])

    @staticmethod
    def point_distances(dx, dy, a, b, aligned, length, half, width):
        result = np.empty_like(dx)
        if aligned.any():
            result[aligned] = np.hypot(np.maximum(abs(dx[aligned]) - length - half, 0.),
                                       np.maximum(abs(dy[aligned]) - width - half, 0.))
        if aligned.all():
            return result
        rotate = ~aligned
        dx, dy, a, b = dx[rotate], dy[rotate], a[rotate], b[rotate]
        u, v = a * dx + b * dy, -b * dx + a * dy
        cell_extent = half * (abs(a) + abs(b))
        overlap = ((abs(dx) <= length * abs(a) + width * abs(b) + half)
                   & (abs(dy) <= length * abs(b) + width * abs(a) + half)
                   & (abs(u) <= length + cell_extent) & (abs(v) <= width + cell_extent))
        distance2 = np.full(dx.shape, np.inf)
        for i, j in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            du = np.maximum(abs(u + half * (i * a + j * b)) - length, 0.)
            dv = np.maximum(abs(v + half * (-i * b + j * a)) - width, 0.)
            px = np.maximum(abs(dx - i * length * a + j * width * b) - half, 0.)
            py = np.maximum(abs(dy - i * length * b - j * width * a) - half, 0.)
            distance2 = np.minimum(distance2, np.minimum(du*du + dv*dv, px*px + py*py))
        result[rotate] = np.sqrt(np.where(overlap, 0., distance2))
        return result


def _square_sweep_hits(squares, lower, upper, radius):
    # The pose yaw maps the map axes to forward/right. Both boxes are
    # closed, so tangency counts as contact.
    c, s = math.cos(squares.yaw), math.sin(squares.yaw)
    axes = np.array(((c, s), (s, -c)))
    lo, hi = np.array(lower), np.array(upper)
    half = RESOLUTION / 2
    center, extent = (lo + hi) / 2, (hi - lo) / 2
    square_extent = half * np.abs(axes).sum(axis=1)
    gaps = np.maximum(np.abs(squares.centers - center) - extent - square_extent, 0.)
    candidates = squares.centers[(gaps**2).sum(axis=1) <= radius**2]
    result = np.zeros(SHAPE, dtype=bool)
    if not candidates.size:
        return result

    delta = candidates - center
    # Four separating axes detect overlap even when no vertex is contained.
    overlap = ((np.abs(delta) <= extent + square_extent).all(axis=1)
               & (np.abs(delta @ axes) <= extent @ np.abs(axes) + half).all(axis=1))
    offsets = np.array(((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)))
    vertices = candidates[:, None, :] + (half * offsets) @ axes
    to_rect = np.maximum(0., np.maximum(lo - vertices, vertices - hi))
    corners = center + extent * offsets
    to_square = np.maximum(0., np.abs((corners - candidates[:, None, :]) @ axes) - half)
    distance2 = np.minimum((to_rect**2).sum(axis=2).min(axis=1),
                           (to_square**2).sum(axis=2).min(axis=1))
    # Only project the cells that actually intersect the sweep. The mask is
    # used for diagnostics and choosing a direction, not to decide contact.
    rows, cols = (candidates[overlap | (distance2 <= radius**2)] / RESOLUTION + START).T
    for r in (np.floor(rows), np.ceil(rows)):
        for k in (np.floor(cols), np.ceil(cols)):
            valid = (r >= 0) & (r < SHAPE[0]) & (k >= 0) & (k < SHAPE[1])
            result[r[valid].astype(int), k[valid].astype(int)] = True
    return result


def _sweep_circle(geometry, half_width):
    center = (geometry.front - geometry.rear) / 2
    radius = math.hypot(geometry.half_length, half_width) + clearance_margin(half_width)
    return center, radius


def motion_sweep_bounds(geometry, half_width, horizon, velocities, *, padding=0., directional=False):
    """Center bounds and radius of the conservative chassis motion envelope."""
    center, radius = _sweep_circle(geometry, half_width)
    # Independently ramped components can reach any center in these bounds.
    # Account for bending of the center's path while turning and translating.
    forward, right = zip(*((vx * horizon, -vy * horizon) for vx, vy, _ in velocities))
    turn = max(abs(yaw) for _, _, yaw in velocities) * horizon
    f_max, r_max = max(map(abs, forward)), max(map(abs, right))
    bend_f = bend_r = 0.
    radius += padding
    if directional and turn <= math.pi / 2:
        # A rotation by at most 90 degrees can only shrink the component
        # along each original axis. Bound the cross-axis travel separately.
        bend_f, bend_r = r_max * turn / 2, f_max * turn / 2
    else:
        radius += math.hypot(f_max, r_max) * min(2., turn / 2)
    lower = center + min(0., *forward) - bend_f, min(0., *right) - bend_r
    upper = center + max(0., *forward) + bend_f, max(0., *right) + bend_r
    return lower, upper, radius


def motion_limits_rejection(geometry, half_width, horizon, velocities, *, room=math.inf,
                            bounded=False, room_horizon=None, bounds_horizon=None, padding=0.):
    """Shared travel-room and supported-map limits for a checked motion."""
    if max(math.hypot(v[0], v[1]) for v in velocities) * max(horizon, room_horizon or 0.) > room:
        return "insufficient_room"
    if bounded:
        lower, upper, radius = motion_sweep_bounds(
            geometry, half_width, max(horizon, bounds_horizon or 0.), velocities, padding=padding)
        half_cell = RESOLUTION / 2
        if (lower[0] - radius < FORWARD[0, 0] - half_cell
                or upper[0] + radius > FORWARD[-1, 0] + half_cell
                or lower[1] - radius < RIGHT[0, 0] - half_cell
                or upper[1] + radius > RIGHT[0, -1] + half_cell):
            return "outside_grid"
    return None


def motion_sweep_hits(occupied, geometry, half_width, horizon, velocities, *, padding=0., directional=False):
    """Sweep a circle centered on the chassis through its possible translations."""
    lower, upper, radius = motion_sweep_bounds(
        geometry, half_width, horizon, velocities, padding=padding, directional=directional)
    if isinstance(occupied, ObstacleSquares):
        return _square_sweep_hits(occupied, lower, upper, radius)

    rows, cols = np.nonzero(occupied)
    f, r = FORWARD[rows, cols], RIGHT[rows, cols]
    half_cell = RESOLUTION / 2
    df = np.maximum(0., np.maximum(lower[0] - f, f - upper[0]) - half_cell)
    dr = np.maximum(0., np.maximum(lower[1] - r, r - upper[1]) - half_cell)
    hits = df**2 + dr**2 <= radius**2
    result = np.zeros_like(occupied)
    result[rows[hits], cols[hits]] = True
    return result


def motion_sweep_clear(grid, geometry, half_width, horizon, velocities):
    return not motion_sweep_hits(grid[1], geometry, half_width, horizon, velocities).any()


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


def clearance_route(grid, geometry, half_width, target, stop, approach_origin, previous=()):
    """Find a clearance-weighted reference about the body pivot, on either side of the goal line."""
    free, occupied = grid
    if not free.any() or not occupied.any() or target is None:
        return None
    remaining = np.hypot(FORWARD-target[0], RIGHT-target[1])
    goal = (remaining >= stop) & (remaining <= stop + RESOLUTION)
    if not goal.any():
        return None
    rows, cols = occupied.shape
    columns = np.arange(cols)
    # Squared Euclidean distance to the nearest occupied centre, by two
    # separable one-dimensional minima on this bounded grid.
    low = np.maximum.accumulate(np.where(occupied, columns, -rows-cols), axis=1)
    high = np.minimum.accumulate(np.where(occupied, columns, rows+cols)[:, ::-1], axis=1)[:, ::-1]
    lateral = np.minimum(columns - low, high - columns)
    row_dist = np.subtract.outer(np.arange(rows), np.arange(rows)) ** 2
    clearance = np.sqrt((row_dist[:, :, None] + lateral[None, :, :] ** 2).min(axis=1)) * RESOLUTION
    pivot = -geometry.half_length if geometry.front == 0. else 0.
    start = (START[0] + round(pivot / RESOLUTION), START[1])
    radius = math.ceil(half_width / RESOLUTION)
    if not 0 <= start[0] < rows or 2*radius >= min(rows, cols):
        return None
    # Fusion can spread a measured cell over neighbouring centres. Use the
    # field for route preference. Every command still checks the original
    # occupied squares and the complete oriented footprint to a stop.
    allowed = clearance > half_width - RESOLUTION
    allowed[start] = True
    desired = half_width + 2.5 * RESOLUTION
    cost = 3. * np.square(np.maximum(0., desired - clearance) / (2.5 * RESOLUTION))
    if previous:
        distance = np.full_like(clearance, np.inf)
        for f, r in previous:
            np.minimum(distance, (FORWARD - pivot - f)**2 + (RIGHT - r)**2, out=distance)
        cost += .5 * np.minimum(1., np.sqrt(distance) / max(.1, half_width))
    # Rejoin the original approach line when the obstruction ends. Otherwise
    # any point on the standoff circle, even far to the side, is equally good.
    dx, dy = target[0] - (approach_origin[0] + pivot), target[1] - approach_origin[1]
    length = max(RESOLUTION, math.hypot(dx, dy))
    lateral = abs((FORWARD-target[0])*dy - (RIGHT-target[1])*dx) / length
    cost += .5 * lateral / max(.15, half_width)
    # A single visible centre can lie next to an unseen obstacle face. Prefer
    # a path with measured floor under the whole chassis while remaining able
    # to approach the boundary of an occluded region for the next view.
    size = 2 * radius + 1
    padded = np.pad(~free, ((radius+1, radius), (radius+1, radius)), constant_values=True)
    integral = padded.cumsum(0).cumsum(1)
    unknown = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    cost += 5. * unknown / size**2
    preferred = goal & (lateral <= RESOLUTION) & allowed
    allowed &= remaining >= stop
    if not (goal & allowed).any():
        return None
    path = shortest_path(allowed, start, start, target_cells=preferred, cell_cost=cost) if preferred.any() else None
    if path is None:
        path = shortest_path(allowed, start, start, target_cells=goal, cell_cost=cost)
    if not path or len(path) < 2:
        return None
    # Steer toward the measured lookahead, rather than following a stair-step
    # tangent of the discretized path. The supporting cells remain observable.
    topology = tuple((FORWARD[cell], RIGHT[cell]) for cell in path)
    relative = tuple((f - pivot, r) for f, r in topology)
    look = next((i for i, point in enumerate(relative) if math.hypot(*point) >= .40), len(relative) - 1)
    return Detour(None, ((0., 0.), *relative[look:]), geometry, topology=topology)


def opening_route(frame, occupied, geometry, half_width, plane, target, allow_detour=False):
    """Propose a measured opening; checked detours may leave the direct goal ray."""
    if (plane <= geometry.front or target is None or target[0] <= plane
            or geometry.horizontal is None or geometry.vertical is None or geometry.base_from_optical[0, 2] <= 0.):
        return None
    route = _opening_at_plane(frame, occupied, geometry, half_width, plane, target, allow_detour)
    if route is not None or not allow_detour:
        return route
    # The nearest obstacle determines braking. Its depth can be the edge of a
    # much nearer object, so it need not be a usable plane for checking jambs.
    depths = FORWARD[occupied.any(axis=1), 0]
    nearby = depths[(depths >= plane + RESOLUTION)
                    & (depths < min(target[0], plane + 1.5))]
    for probe in nearby[::2]:
        route = _opening_at_plane(frame, occupied, geometry, half_width, probe, target, True)
        if route is not None:
            return route
    return None


def _opening_at_plane(frame, occupied, geometry, half_width, plane, target, allow_detour):
    band = occupied & (abs(FORWARD - plane) <= .10)
    cols = np.flatnonzero(band.any(axis=0))
    if allow_detour:
        # Each edge is already a measured occupied cell. A checked moving
        # detour can approach an offset gap or a single-cell jamb; requiring
        # the body/goal ray inside it here discards that useful reference.
        gaps = [(RIGHT[0, a], RIGHT[0, b]) for a, b in zip(cols[:-1], cols[1:])
                if RIGHT[0, b] - RIGHT[0, a] - RESOLUTION >= 2 * half_width]
        gaps.sort(key=lambda gap: math.hypot(plane, sum(gap) / 2)
                  + math.hypot(target[0] - plane, target[1] - sum(gap) / 2))
    else:
        left, right = cols[cols < START[1]], cols[cols > START[1]]
        if (band[:, START[1]].any() or left.size < 2 or right.size < 2
                or left[-2] != left[-1] - 1 or right[1] != right[0] + 1):
            return None
        lo, hi = RIGHT[0, [left[-1], right[0]]]
        if (hi - lo - RESOLUTION < 2 * half_width or min(-lo, hi) > half_width + RESOLUTION
                or not lo < target[1] * plane / target[0] < hi):
            return None
        gaps = [(lo, hi)]

    height, width = frame.shape
    fx, cx, fy, cy = intrinsics_for(geometry, height, width)
    transform = geometry.base_from_optical
    origin = transform[:3, 3]
    samples = math.ceil(2 * half_width / RESOLUTION) + 1
    z = origin[2] + (plane - origin[0]) * transform[2, 2] / transform[0, 2]
    for lo, hi in gaps:
        middle = (hi + lo) / 2
        # Far returns only select a heading; they never clear the driving
        # envelope. A detour need not also see the unrelated direct goal ray.
        r = np.linspace(middle - half_width, middle + half_width, samples)
        if not allow_detour:
            r = np.append(r, target[1] * plane / target[0])
        f = np.full(r.size, plane)
        optical = (np.column_stack((f, -r, np.full_like(f, z))) - origin) @ transform[:3, :3]
        if np.any(optical[:, 2] <= 0):
            continue
        u = np.rint(cx + fx * optical[:, 0] / optical[:, 2]).astype(int)
        v = np.rint(cy + fy * optical[:, 1] / optical[:, 2]).astype(int)
        if np.any((u < 0) | (u >= width) | (v < 0) | (v >= height)):
            continue
        measured = frame[v, u]
        valid = np.isfinite(measured) & (measured > 0.)
        opening_valid = valid if allow_detour else valid[:-1]
        if ((not allow_detour and not valid[-1]) or np.count_nonzero(opening_valid) < .75 * samples
                or np.any(measured[valid] <= optical[valid, 2] + 2 * RESOLUTION)):
            continue
        # The caller takes the longest safe prefix and checks it again after turning.
        end = min(plane + geometry.rear + .25, FORWARD[-1, 0] - geometry.front)
        # Centre a tight detour before the front reaches the jambs and
        # hold that line until the rear clears. This is only a reference.
        entry = plane - geometry.front - RESOLUTION
        align = (allow_detour and entry > RESOLUTION
                 and hi - lo - RESOLUTION < 2 * (half_width + geometry.half_length))
        points = tuple((forward, middle * min(1., forward / entry) if align else forward * middle / plane)
                       for forward in np.arange(0., end + 1e-9, RESOLUTION))
        return Detour(None, points, geometry)
    return None


def _turnable_endpoints(cells, occupied, geometry, half_width):
    """Apply the stationary body-circle sweep at each candidate goal cell."""
    rows, cols = np.nonzero(cells)
    if not len(rows) or not occupied.any():
        return cells
    center, radius = _sweep_circle(geometry, half_width)
    half = RESOLUTION / 2
    goals_f, goals_r = FORWARD[rows, cols] + center, RIGHT[rows, cols]
    obstacles_f, obstacles_r = FORWARD[occupied], RIGHT[occupied]
    result = cells.copy()
    for start in range(0, len(rows), 64):
        batch = slice(start, start + 64)
        df = np.maximum(0., np.abs(goals_f[batch, None] - obstacles_f) - half)
        dr = np.maximum(0., np.abs(goals_r[batch, None] - obstacles_r) - half)
        result[rows[batch], cols[batch]] = ~((df**2 + dr**2) <= radius**2).any(axis=1)
    return result


def _checked_translation(frame, ground, geometry, half_width, ahead, side, max_age_ms,
                         watchdog_ms=300, *, velocity_limits, observed_grid=None, allow_forward=True, allow_return=True,
                         target=None, projected_grid=None, body_half_width=None, closest=False,
                         preferred_heading=None, allow_unknown=False, require_observed_endpoint=True, finite_turn=False):
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
    observed_endpoint = allow_unknown and require_observed_endpoint
    if free is not None:
        free = free & ~occupied
    if observed_endpoint and (free is None or not free.any()):
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
    rejoin_in_grid = row_zero <= via[0] < goal[0] < grid.shape[0] and via[1] < grid.shape[1]
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
    reachable_prefix = False
    if target is not None:
        optical_forward, optical_right, stop, tolerance = target
        target_f, target_r = target_in_grid(geometry, optical_forward, optical_right)
        target_r *= side
        ranges = np.hypot(forward - target_f, right - target_r)
        allowed &= ranges >= stop
        target_cells = allowed & (ranges <= stop + max(tolerance, resolution / math.sqrt(2)))
        if observed_endpoint:
            target_cells &= free
        turn_width = half_width if body_half_width is None else body_half_width
        if not finite_turn:
            target_cells = _turnable_endpoints(target_cells, occupied, geometry, turn_width)
        path = find_path(allowed, start, goal, steps=steps, target_cells=target_cells)
        if path is None and closest:
            reachable_prefix = True
            target_cells = ranges < ranges[start] - resolution / 2
            if observed_endpoint:
                target_cells &= free
            path = find_path(allowed, start, goal, steps=steps, target_cells=target_cells, target_cost=ranges)
            if path is None:
                return None
    if path is None and not rejoin_in_grid:
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
    if observed_endpoint:
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
        return Detour(command, points, geometry, reachable_prefix=reachable_prefix)

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
        if observed_endpoint and not free[point]:
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
