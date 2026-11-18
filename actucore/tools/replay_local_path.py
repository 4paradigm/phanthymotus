#!/usr/bin/env python3
"""Replay a fixed-heading right detour from stationary front depth recordings.

Grid coordinates are forward row, right column, in the recording's body frame.
Unknown cells are blocked. The only exception is the footprint already occupied
by the stationary robot, and even there observed obstacle returns are retained.
This tool has no ROS or control publisher.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.navi import camera, depth  # noqa: E402

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


def footprint_region(center, half_cells):
    row, col = center
    dr, dc = half_cells
    return np.s_[row-dr:row+dr+1, col-dc:col+dc+1]


def allowed_centers(grid, start, half_cells, allow_unknown=False):
    blocked = grid == OCCUPIED if allow_unknown else grid != FREE
    area = footprint_region(start, half_cells)
    blocked[area] = grid[area] == OCCUPIED
    dr, dc = half_cells
    rectangle = np.pad(blocked, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    blocked_count = (rectangle[2*dr+1:, 2*dc+1:] - rectangle[:-2*dr-1, 2*dc+1:]
                     - rectangle[2*dr+1:, :-2*dc-1] + rectangle[:-2*dr-1, :-2*dc-1])
    return np.pad(blocked_count == 0, ((dr, dr), (dc, dc)), constant_values=False)


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


def observe(frame, intrinsics, vertical, camera_forward, right, forward, resolution):
    ground = depth.ground_mask(frame, intrinsics, vertical)
    if ground is None:
        return np.zeros(right.shape, dtype=bool), np.zeros(right.shape, dtype=bool)
    height, width = frame.shape
    fx, cx = intrinsics.for_width(width)
    calib_height, fy, cy = vertical
    scale = height / calib_height
    fy, cy = fy * scale, (cy + 0.5) * scale - 0.5
    u = (np.arange(width) - cx) / fx
    v = (np.arange(height) - cy) / fy
    x, y = u * frame, v[:, None] * frame
    seed = ground[::4, ::4]
    gs = frame[::4, ::4][seed]
    gx, gy = x[::4, ::4][seed], y[::4, ::4][seed]
    a, b, c = np.linalg.lstsq(np.column_stack((gx, gs, np.ones_like(gs))), gy, rcond=None)[0]

    z = forward - camera_forward
    front = z > 0
    denom = np.where(front, z, 1.0)
    cols = np.rint(fx * right / denom + cx).astype(int)
    floor_y = a * right + b * z + c
    free = front & (cols >= 0) & (cols < width)
    for height_m in (0.03, 0.08, 0.16, 0.25, 0.40):
        rows = np.rint(fy * (floor_y - height_m) / denom + cy).astype(int)
        in_image = free & (rows >= 0) & (rows < height)
        distances = frame[rows[in_image], cols[in_image]]
        free[~in_image] = False
        free[in_image] = np.isfinite(distances) & (distances > z[in_image] + 2 * resolution)

    # Occupied returns outrank clear rays. Project low to body-height points
    # into the ground grid while preserving unknown cells behind them.
    above_floor = a * x + b * frame + c - y
    hits = np.isfinite(frame) & ~ground & (above_floor >= 0.02) & (above_floor <= 0.48)
    hit_row = np.rint((frame[hits] + camera_forward - forward[0, 0]) / resolution).astype(int)
    hit_col = np.rint((x[hits] - right[0, 0]) / resolution).astype(int)
    in_grid = ((hit_row >= 0) & (hit_row < right.shape[0])
               & (hit_col >= 0) & (hit_col < right.shape[1]))
    counts = np.zeros(right.shape, dtype=int)
    np.add.at(counts, (hit_row[in_grid], hit_col[in_grid]), 1)
    return free, counts >= 3


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("capture", type=Path)
    args.add_argument("--end-ns", type=int, required=True)
    args.add_argument("--seconds", type=float, default=3)
    args.add_argument("--box-roi", type=int, nargs=4, required=True,
                      metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                      help="visible box face in 640x480 depth pixels")
    args.add_argument("--report", type=Path, required=True)
    params = args.parse_args()
    root = params.capture.parent
    cards = {r["tool"]: r["data"] for l in (root / "cards-before.jsonl").open()
             if (r := json.loads(l)).get("data")}
    decl = cards["camera_front_depth"]["camera_info"][0]
    k = camera.horizontal_intrinsics(decl)
    k_y = camera.vertical_intrinsics(decl)
    camera_forward = -decl["vendor"]["camera_optical_from_cloud"][2][3]
    half_width = cards["loco_servo"]["control_interface"]["footprint"]["half_width"] + .10
    half_length = 0.30  # standing body estimate from the S10 model
    span = params.end_ns - int(params.seconds * 1e9)
    def window(key):
        return [r for l in (params.capture / f"{key}.jsonl").open()
                if span <= (r := json.loads(l))["received_ns"] <= params.end_ns
                and not r["first_after_subscription"]]

    odom = window("odom")
    odom_v = np.asarray([[a["data"]["twist"][i] for i in (0, 1, 5)] for a in odom])
    if len(odom_v) < 10 or np.any(np.percentile(abs(odom_v), 95, axis=0) > [.04, .06, .04]):
        raise SystemExit("stationary replay requires a stable odometry window")
    frames = window("depth")
    if len(frames) < 3:
        raise SystemExit("need at least three new depth frames")

    resolution = .025
    r_axis = np.arange(-60, 61) * resolution
    f_axis = np.arange(-22, 141) * resolution
    right, forward = np.meshgrid(r_axis, f_axis)
    free_count = np.zeros(right.shape, dtype=int)
    occupied = np.zeros(right.shape, dtype=bool)
    box_rights, box_depths = [], []
    left, top, right_px, bottom = params.box_roi
    fx, cx = k.for_width(640)
    for frame in frames:
        m = depth.decode((params.capture / frame["data"]["file"]).read_bytes())
        free, hits = observe(m, k, k_y, camera_forward, right, forward, resolution)
        free_count += free
        occupied |= hits
        box = m[top:bottom, left:right_px]
        box_x = box * ((np.arange(left, right_px) - cx) / fx)
        box_rights.append(np.nanpercentile(box_x, 95))
        box_depths.append(np.nanmedian(box))

    grid = np.full(right.shape, UNKNOWN, dtype=np.int8)
    grid[free_count >= max(2, math.ceil(len(frames) * .6))] = FREE
    grid[occupied] = OCCUPIED

    def index(z, x):
        return (round((z - f_axis[0]) / resolution), round((x - r_axis[0]) / resolution))

    start = index(0, 0)
    right_of_box = float(np.median(box_rights)) + half_width + 2 * resolution
    front_face = float(np.median(box_depths)) + camera_forward
    via = index(front_face, right_of_box)
    # A provisional replay waypoint beyond the observed face. Its distance
    # reserves the rear half of the robot and 0.25 m beyond the visible face.
    goal = index(front_face + half_length + .25, 0)
    half_cells = tuple(math.floor(half / resolution + .5) for half in (half_length, half_width))
    path = plan_right(grid, start, via, goal, half_cells)
    geometry_only = plan_right(grid, start, via, goal, half_cells, allow_unknown=True)
    first_move = {}
    new = np.zeros_like(grid, dtype=bool)
    current = np.zeros_like(new)
    current[footprint_region(start, half_cells)] = True
    for name, (dr, dc) in {"right": (0, 1), "forward": (1, 0), "forward_right": (1, 1)}.items():
        new[:] = False
        new[footprint_region((start[0]+dr, start[1]+dc), half_cells)] = True
        cells = grid[new & ~current]
        first_move[name] = {label: int((cells == value).sum()) for label, value in
                            (("unknown", UNKNOWN), ("free", FREE), ("occupied", OCCUPIED))}

    def points(route):
        return None if route is None else [[round(float(r_axis[x]), 3), round(float(f_axis[z]), 3)]
                                           for z, x in route]

    report = dict(depth_frames=len(frames), odom_samples=len(odom),
                  odom_median=np.median(odom_v, axis=0).tolist(), odom_stamp_sources=sorted({
                      a["data"]["vendor"].get("stamp_source", "unknown") for a in odom}),
                  footprint_width_m=2 * half_width, standing_length_estimate_m=2 * half_length,
                  box_visible_face_optical_m=float(np.median(box_depths)),
                  first_move_new_cells=first_move, verified_right_path=points(path),
                  path_ignoring_unknown_for_diagnosis=points(geometry_only),
                  resolution_m=resolution)
    params.report.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(params.report.with_suffix(".npz"), grid=grid, right=r_axis,
                        forward=f_axis, start=start, via=via, goal=goal,
                        path=path or [], geometry_only=geometry_only or [])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
