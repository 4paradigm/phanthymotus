"""Geometry for the `height` card — depth decoding, deprojection, plane fit.

Pure numpy, no ROS, no clock: every function here is testable on a laptop with
a synthetic depth map, which is the only way a measurement card gets reviewed
without a robot. The card (`plugins/height.py`) owns the subscriptions, the
frame window and the error codes; this module owns the arithmetic.

Frames used throughout:

    camera optical (`aligned depth`, the frame the card's K describes)
        x right, y down, z forward. The published depth is the **z-depth** of
        the pixel (distance along the optical axis), in millimetres, 0=invalid.

    gravity-aligned / floor ("mount" mode)
        same axes rotated by the camera's mounting pitch and roll, so y points
        down along gravity. `H` is the camera origin's height above the floor,
        and a point at height h above the floor has y = H - h.

    plane ("floor" mode)
        a plane fitted to the current depth frame. A point's distance to that
        plane is its height above whatever the plane is — the floor, and the
        reason this mode needs no stored mounting calibration at all.

The "top" of a target is taken from the top edge of its bounding box, sampled
slightly inside it (hair is a depth hole) and corrected back up by the pixel
offset. The correction matters: at 3 m one pixel is 5 mm, and a 5 px skip for a
valid depth would otherwise read 2.5 cm short.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

# The `image/depth-zlib` contract is 640x480 uint16 little-endian millimetres,
# 0 meaning "no reading". Mirrored here rather than imported from another
# bundle: the format is a document, not a shared module (same rule as
# `motus.camera/1` and `motus.odom/1`).
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
DEPTH_BYTES = DEPTH_WIDTH * DEPTH_HEIGHT * 2

# Below this many pixels the target is either far away or a depth artefact;
# both read better as "too_far" than as a confident height.
MIN_TARGET_PX = 24
# Keep a margin from the top edge: a box touching it may have its head (or the
# top of the object) outside the picture, and a clipped top silently reads as a
# shorter person.
TOP_MARGIN_PX = 6
# A plane whose inlier fraction or RMS residual is worse than this is not the
# floor; fall back to the calibrated mount instead of reporting a number.
MIN_PLANE_INLIERS = 0.20
MAX_PLANE_RESIDUAL_M = 0.05
# Measurements outside this band are not a person or a floor-standing object;
# refuse rather than return a plausible-looking number.
MIN_HEIGHT_M = 0.15
MAX_HEIGHT_M = 3.0
# Two detections this close in area are treated as "which one?" rather than
# silently picking the larger one.
AMBIGUOUS_AREA_RATIO = 0.85


class GeometryError(ValueError):
    """A measurement that cannot be made, with the reason the card reports.

    `reason` is the stable machine-readable code (the card puts it in `reason`
    of its reply); the message is for a person reading the card log or the
    tool result.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


# ── depth decoding ───────────────────────────────────────────────────────────

def decode_depth(payload, width: int = DEPTH_WIDTH,
                 height: int = DEPTH_HEIGHT) -> np.ndarray:
    """A `image/depth-zlib` payload as a float32 array of metres, NaN invalid.

    Invalid samples become NaN rather than 0.0: every arithmetic step here has
    to propagate "no reading" instead of treating it as a point at the camera.
    A short buffer is refused rather than reshaped — the driver and the card
    would otherwise disagree about the picture and still return numbers.
    """
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise GeometryError("bad_depth", "depth payload is empty")
    try:
        raw = zlib.decompress(bytes(payload))
    except zlib.error as error:
        raise GeometryError("bad_depth", f"depth payload is not zlib: {error}")
    expected = int(width) * int(height) * 2
    if len(raw) != expected:
        raise GeometryError(
            "bad_depth",
            f"depth payload is {len(raw)} bytes, expected {expected} "
            f"({width}x{height} uint16)")
    millimetres = np.frombuffer(raw, dtype="<u2").reshape(int(height), int(width))
    metres = millimetres.astype(np.float32) / 1000.0
    metres[millimetres == 0] = np.nan
    return metres


def encode_depth(depth_m: np.ndarray) -> bytes:
    """Inverse of `decode_depth`, for tests and offline replay."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    height, width = depth_m.shape
    if (width, height) != (DEPTH_WIDTH, DEPTH_HEIGHT):
        raise GeometryError(
            "bad_depth",
            f"depth map is {width}x{height}, the contract is "
            f"{DEPTH_WIDTH}x{DEPTH_HEIGHT}")
    millimetres = np.nan_to_num(depth_m * 1000.0, nan=0.0, posinf=0.0,
                                neginf=0.0)
    millimetres = np.clip(np.rint(millimetres), 0, 65535).astype("<u2")
    return zlib.compress(millimetres.tobytes(), 1)


# ── intrinsics ───────────────────────────────────────────────────────────────

def _positive(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def declaration_for(declarations, topic: str) -> dict:
    """The `motus.camera/1` entry covering `topic`, or {}.

    agent-core passes `{topic: declaration}`; a list is accepted too so the
    same helper works against a card's raw `info()` in tests. Joined on topic,
    because that is the one thing both sides agree on.
    """
    if isinstance(declarations, dict):
        entry = declarations.get(topic)
        return dict(entry) if isinstance(entry, dict) else {}
    for entry in declarations or ():
        if isinstance(entry, dict) and entry.get("topic") == topic:
            return dict(entry)
    return {}


def _config_intrinsics(config: dict):
    fx = float(config.get("fx") or 0.0)
    fy = float(config.get("fy") or 0.0)
    cx = float(config.get("cx") or 0.0)
    cy = float(config.get("cy") or 0.0)
    if _positive(fx) and _positive(fy) and cx > 0 and cy > 0:
        return fx, fy, cx, cy
    return None


def _relative_difference(a: float, b: float) -> float:
    if a == 0.0 and b == 0.0:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def resolve_intrinsics(declaration: dict, config: dict) -> dict:
    """`{fx, fy, cx, cy, width, height, source, notes}` for the depth port.

    Priority is deliberate and was learned the hard way on this project (a
    field of view typed into a config file made a robot refuse doorways):

    1. the `K` the camera itself declared at runtime;
    2. measured intrinsics stored in this card's config (marked `manual`);
    3. a `K` derived from the declared half-FOV angles — only when the
       operator explicitly allows it, because it assumes a centred principal
       point and square-ish pixels;
    4. refuse (`no_camera_info`).

    A declaration whose `K` is missing is the normal cold-start state for the
    Bumi driver: the RealSense subprocess prints its intrinsics after
    `pipeline.start()`, and the declaration falls back to FOV-only until that
    line arrives. That is why step 2 exists at all, and why the card tells the
    operator to restart the project after the camera is up.
    """
    declaration = declaration or {}
    width = int(declaration.get("width") or DEPTH_WIDTH)
    height = int(declaration.get("height") or DEPTH_HEIGHT)
    if (width, height) != (DEPTH_WIDTH, DEPTH_HEIGHT):
        raise GeometryError(
            "intrinsics_size_mismatch",
            f"camera declaration is {width}x{height}; the depth contract is "
            f"{DEPTH_WIDTH}x{DEPTH_HEIGHT}. Refusing to mix geometry from two "
            f"pictures.")
    notes = []
    configured = _config_intrinsics(config)

    runtime = None
    K = declaration.get("K")
    if isinstance(K, (list, tuple)) and len(K) == 9:
        try:
            fx, fy, cx, cy = (float(K[0]), float(K[4]), float(K[2]), float(K[5]))
        except (TypeError, ValueError):
            fx = fy = cx = cy = 0.0
        if _positive(fx) and _positive(fy) and cx > 0 and cy > 0:
            runtime = (fx, fy, cx, cy)

    if runtime is not None:
        fx, fy, cx, cy = runtime
        if configured is not None:
            diffs = [_relative_difference(a, b) for a, b in zip(runtime, configured)]
            if max(diffs) > 0.05:
                notes.append(
                    "config intrinsics disagree with the camera's runtime K by "
                    f"{max(diffs) * 100:.1f}%; using the camera's K")
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "width": width, "height": height,
                "source": "runtime-K", "notes": notes}

    if configured is not None:
        fx, fy, cx, cy = configured
        notes.append("using manually configured intrinsics; no runtime K was "
                     "declared (source=manual)")
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "width": width, "height": height,
                "source": "config-manual", "notes": notes}

    half_h = declaration.get("half_fov_rad")
    half_v = declaration.get("half_fov_v_rad")
    if bool(config.get("allow_fov_intrinsics")) and _positive(half_h) and _positive(half_v):
        try:
            fx = (width / 2.0) / math.tan(float(half_h))
            fy = (height / 2.0) / math.tan(float(half_v))
        except (ValueError, ZeroDivisionError):
            fx = fy = 0.0
        if _positive(fx) and _positive(fy):
            notes.append(
                "intrinsics derived from declared half-FOV angles "
                "(source=derived-from-fov); no principal-point offset and no "
                "distortion are known, so treat the uncertainty as larger")
            return {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0,
                    "width": width, "height": height,
                    "source": "derived-from-fov", "notes": notes}

    raise GeometryError(
        "no_camera_info",
        "no usable intrinsics: the depth card declared no K, this card's "
        "config has no fx/fy/cx/cy, and deriving them from the FOV is not "
        "allowed. Wait until the camera is streaming and restart the project, "
        "or set the intrinsics in the card config (they are then marked "
        "manual).")


# ── targets (pose / vop boxes) ───────────────────────────────────────────────

def scale_bbox(bbox, from_size, to_size):
    """A pixel box in one picture, as a pixel box in another."""
    from_w, from_h = int(from_size[0]), int(from_size[1])
    to_w, to_h = int(to_size[0]), int(to_size[1])
    if from_w <= 0 or from_h <= 0:
        raise GeometryError("bad_detection", "detection resolution is unknown")
    sx, sy = to_w / from_w, to_h / from_h
    return [float(bbox[0]) * sx, float(bbox[1]) * sy,
            float(bbox[2]) * sx, float(bbox[3]) * sy]


def clamp_bbox(bbox, width, height, *, check_clipped=True):
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    if x2 <= x1 or y2 <= y1:
        raise GeometryError("bad_detection", f"box has no area: {bbox!r}")
    if check_clipped and y1 < TOP_MARGIN_PX:
        raise GeometryError(
            "target_clipped",
            f"the target's top edge is at y={y1:.0f} px (margin "
            f"{TOP_MARGIN_PX}); its top is outside the picture, so the height "
            f"would read short. Ask the person to stand further back.")
    if check_clipped and y1 >= height:
        raise GeometryError("bad_detection", "the box is below the picture")
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise GeometryError("bad_detection", f"box is too small: {bbox!r}")
    return [x1, y1, x2, y2]


def _box_height_fraction(bbox, width, height) -> float:
    return (bbox[3] - bbox[1]) / float(height)


def pick_detection(payload, source: str, target: str,
                   detection_size, depth_size) -> dict:
    """The target box to measure, chosen from an upstream detection payload.

    `source` is `pose` (one box per person) or `vop` (one box per object with a
    name). `target` filters: for vop it is a class name; `person`/`object`/
    `auto`/empty mean "the largest". Two detections of similar size are
    reported as `ambiguous_target` instead of picking one — measuring the wrong
    person is worse than asking to stand apart.

    The box is rescaled to the depth picture when the detection resolution
    differs; the depth map is the grid every later step indexes.
    """
    if not isinstance(payload, dict):
        raise GeometryError("bad_detection", "detection payload is not an object")
    depth_w, depth_h = int(depth_size[0]), int(depth_size[1])
    wanted = (target or "").strip().lower()
    candidates = []
    if source == "pose":
        for person in payload.get("persons") or []:
            if not isinstance(person, dict):
                continue
            bbox = person.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            candidates.append({
                "bbox_px": scale_bbox(bbox, detection_size, depth_size),
                "kind": "person",
                "label": "person",
                "track_id": person.get("id"),
            })
    elif source == "vop":
        for item in payload.get("objects") or []:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            name = str(item.get("name") or "object")
            if wanted and wanted not in ("object", "auto") and name.lower() != wanted:
                continue
            candidates.append({
                "bbox_px": scale_bbox(bbox, detection_size, depth_size),
                "kind": "object",
                "label": name,
                "track_id": None,
                "confidence": item.get("confidence"),
            })
    else:
        raise GeometryError("bad_detection", f"unknown detection source {source!r}")

    if not candidates:
        raise GeometryError(
            "target_not_found",
            f"no {source} detection for target {target!r} in the latest "
            f"payload. Check that the detector is running and sees the "
            f"target.")
    for candidate in candidates:
        candidate["bbox_px"] = clamp_bbox(candidate["bbox_px"], depth_w, depth_h)
    candidates.sort(key=lambda c: (c["bbox_px"][2] - c["bbox_px"][0]) *
                                   (c["bbox_px"][3] - c["bbox_px"][1]),
                    reverse=True)
    chosen = candidates[0]
    if len(candidates) > 1:
        biggest = _bbox_area(chosen["bbox_px"])
        runner_up = _bbox_area(candidates[1]["bbox_px"])
        if runner_up >= AMBIGUOUS_AREA_RATIO * biggest:
            raise GeometryError(
                "ambiguous_target",
                f"{len(candidates)} targets are about the same size; cannot "
                f"tell which one to measure. Ask the others to step aside, or "
                f"pass a class name for vop detections.")
    box_h = chosen["bbox_px"][3] - chosen["bbox_px"][1]
    if box_h < MIN_TARGET_PX or _box_height_fraction(
            chosen["bbox_px"], depth_w, depth_h) < 0.12:
        raise GeometryError(
            "too_far",
            "the target is too small in the picture to measure; move closer "
            "(a person is typically 2-3.5 m away).")
    chosen["candidate_count"] = len(candidates)
    return chosen


def _bbox_area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


# ── the target's top point ───────────────────────────────────────────────────

def _patch_median(depth: np.ndarray, u: int, v: int, radius: int = 2):
    height, width = depth.shape
    v0, v1 = max(0, v - radius), min(height, v + radius + 1)
    u0, u1 = max(0, u - radius), min(width, u + radius + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def top_point(depth: np.ndarray, bbox, K: dict, *,
              search_frac: float = 0.03) -> dict:
    """Camera-frame 3D point of the target's top edge.

    The bbox top is the visual top (hair included). Depth on the exact top row
    is frequently a hole — hair absorbs IR — so the first valid sample below it
    is used and the result is corrected back up by the pixel offset it skipped:
    `y_top = y_sample - dv * z / fy`. The vertical coordinate is what carries
    the height; x is kept for the floor-plane distance.

    Returns `{point, y_top, z, valid_ratio, samples}` where `point` is
    `(x, y_top, z)` in the camera frame.
    """
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 4 or box_h < 4:
        raise GeometryError("bad_detection", f"box is too small: {bbox!r}")
    fx, fy, cx, cy = (float(K["fx"]), float(K["fy"]),
                      float(K["cx"]), float(K["cy"]))
    left = x1 + max(1, int(round(0.2 * box_w)))
    right = x2 - max(1, int(round(0.2 * box_w)))
    if right <= left:
        left, right = x1 + 1, max(x1 + 2, x2 - 1)
    columns = max(3, min(9, int(box_w // 8) + 1))
    us = np.linspace(left, right, num=columns).round().astype(int)
    search_px = max(3, int(round(search_frac * box_h)))

    ys, zs, xs = [], [], []
    for u in us:
        for dv in range(0, search_px + 1):
            v = y1 + dv
            if v >= height:
                break
            z = _patch_median(depth, int(u), v)
            if z is None or not np.isfinite(z) or z <= 0:
                continue
            y_sample = (v - cy) * z / fy
            y_top = y_sample - dv * z / fy
            ys.append(float(y_top))
            zs.append(float(z))
            xs.append(float((u - cx) * z / fx))
            break
    if not ys:
        raise GeometryError(
            "no_depth_at_target",
            "no valid depth samples near the target's top edge. Hair, black "
            "clothing, glass and sun can all produce holes; try a step "
            "sideways or ask the person to face the camera.")
    valid_ratio = len(ys) / float(columns)
    z_top = float(np.median(zs))
    y_top = float(np.median(ys))
    x_top = float(np.median(xs))
    return {
        "point": (x_top, y_top, z_top),
        "y_top": y_top,
        "z": z_top,
        "valid_ratio": valid_ratio,
        "samples": len(ys),
    }


# ── floor plane ──────────────────────────────────────────────────────────────

def fit_plane(points) -> dict:
    """Least-squares plane through 3D points, trimmed, oriented up.

    Returns `{normal, offset, height_m, pitch_rad, roll_rad, residual_m,
    inlier_ratio, points}` for `normal · P + offset = 0`, with `normal` a unit
    vector pointing **up** (towards the camera, `normal[1] < 0`) and
    `height_m = offset` the camera origin's distance to the plane.

    Trimming rather than RANSAC: the floor region may contain feet, table legs
    and cable, and three rounds of "fit, drop points far from the plane" is
    enough for that and deterministic in tests. RANSAC's randomness would need
    a seed and a repetition argument every test then has to know about.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 50:
        raise GeometryError(
            "no_ground_reference",
            f"only {len(pts)} usable floor points; the floor must be in view")
    total = len(pts)
    normal = np.array([0.0, -1.0, 0.0])
    mean = pts.mean(axis=0)
    for _ in range(4):
        mean = pts.mean(axis=0)
        centred = pts - mean
        try:
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError as error:
            raise GeometryError("no_ground_reference",
                                f"plane fit failed: {error}")
        candidate = vt[-1]
        norm = float(np.linalg.norm(candidate))
        if norm <= 0:
            raise GeometryError("no_ground_reference", "degenerate plane fit")
        normal = candidate / norm
        distance = np.abs(centred @ normal)
        median = float(np.median(distance))
        threshold = max(0.02, 3.0 * median + 0.01)
        keep = distance <= threshold
        if keep.all() or keep.sum() < 50:
            break
        pts = pts[keep]
    # One final fit on the surviving points, so the returned normal and offset
    # describe the inliers rather than the set before the last trim.
    mean = pts.mean(axis=0)
    centred = pts - mean
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        normal = vt[-1] / float(np.linalg.norm(vt[-1]))
    except np.linalg.LinAlgError as error:
        raise GeometryError("no_ground_reference", f"plane fit failed: {error}")
    distance = np.abs(centred @ normal)
    median = float(np.median(distance))
    threshold = max(0.02, 3.0 * median + 0.01)
    keep = distance <= threshold
    offset = float(-normal @ mean)
    residual = float(np.sqrt(np.mean(distance[keep] ** 2))) if keep.any() else 0.0
    # The floor is below the camera; orient the normal up so `offset` is the
    # camera height and a point above the floor gives a positive distance.
    if normal[1] > 0:
        normal = -normal
        offset = -offset
    height_m = offset if offset >= 0 else -offset
    # See the module docstring: for a level camera the floor normal is
    # (0,-1,0); pitch tips it towards +z, roll towards -x.
    pitch = math.atan2(-float(normal[2]), -float(normal[1]))
    roll = math.atan2(-float(normal[0]),
                      math.hypot(float(normal[1]), float(normal[2])))
    return {
        "normal": [float(v) for v in normal],
        "offset": float(offset),
        "height_m": float(height_m),
        "pitch_rad": float(pitch),
        "roll_rad": float(roll),
        "residual_m": residual,
        "inlier_ratio": float(len(pts) / total),
        "points": int(len(pts)),
    }


def floor_plane_from_depth(depth: np.ndarray, K: dict, *,
                           max_range_m: float = 4.5,
                           min_range_m: float = 0.25,
                           rows_from: float = 0.60,
                           exclude_bbox=None,
                           max_points: int = 12000) -> dict:
    """Fit the floor from the lower part of a depth frame.

    The lower rows are where the floor is when the camera looks forward or
    slightly down; walls and furniture beyond `max_range_m` are dropped before
    the fit, and `exclude_bbox` keeps the target (a person's body) out of it.
    """
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    if rows_from <= 0.0 or rows_from >= 1.0:
        raise GeometryError("no_ground_reference", "rows_from must be in (0,1)")
    v_start = min(height - 2, max(1, int(height * rows_from)))
    fx, fy, cx, cy = (float(K["fx"]), float(K["fy"]),
                      float(K["cx"]), float(K["cy"]))
    step = max(1, int(round(math.sqrt(max(1, (height - v_start) * width /
                                         float(max_points))))))
    rows = np.arange(v_start, height, step)
    cols = np.arange(0, width, step)
    grid = depth[np.ix_(rows, cols)]
    valid = np.isfinite(grid) & (grid >= min_range_m) & (grid <= max_range_m)
    if exclude_bbox is not None:
        x1, y1, x2, y2 = exclude_bbox
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        inside = ((rr >= y1) & (rr <= y2) & (cc >= x1) & (cc <= x2))
        valid &= ~inside
    if valid.sum() < 50:
        raise GeometryError(
            "no_ground_reference",
            "the lower part of the picture has too few valid depth samples "
            "for a floor fit; make sure the floor in front of the robot is "
            "visible and within range")
    z = grid[valid]
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr = rr[valid].astype(np.float32)
    cc = cc[valid].astype(np.float32)
    x = (cc - cx) * z / fx
    y = (rr - cy) * z / fy
    plane = fit_plane(np.stack([x, y, z], axis=1))
    # A wall is also a plane. Only accept a plane whose normal is close to the
    # camera's own down axis — otherwise "height above the plane" would be the
    # distance to a wall, which is a plausible-looking wrong number.
    if abs(float(plane["normal"][1])) < 0.70:
        raise GeometryError(
            "no_ground_reference",
            "the fitted plane is not horizontal enough to be a floor "
            f"(|n_y|={abs(float(plane['normal'][1])):.2f}); point the camera at "
            f"the floor in front of the robot")
    return plane


def height_above_plane(point, plane: dict) -> float:
    """Distance from `point` to the floor plane, positive above it."""
    normal = np.asarray(plane["normal"], dtype=np.float64)
    offset = float(plane["offset"])
    return float(normal @ np.asarray(point, dtype=np.float64) + offset)


# ── mounting (calibrated) mode ───────────────────────────────────────────────

def gravity_align_point(point, pitch_rad: float, roll_rad: float):
    """Camera-frame point in the gravity-aligned frame (y down).

    `pitch_rad` is the camera's mounting pitch, positive when the optical axis
    is below horizontal; `roll_rad` is positive when the camera's right side is
    up (the sign comes out of the floor normal in `fit_plane`).
    """
    x, y, z = (float(v) for v in point)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    # Undo pitch: camera y/z -> gravity y/z, x unchanged.
    y1 = y * cp + z * sp
    z1 = -y * sp + z * cp
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    # Undo roll around the (already level) optical axis.
    x2 = x * cr - y1 * sr
    y2 = x * sr + y1 * cr
    return x2, y2, z1


def height_from_mount(point, camera_height_m: float,
                      pitch_rad: float, roll_rad: float) -> float:
    """Height above the floor from a calibrated camera mount.

    This is the mode that needs no floor in view: the camera's own height and
    tilt are enough. It is only as good as that calibration — at 3 m a 1 degree
    pitch error is 5 cm — which is why the card reports the mode it used and
    the config value it trusted.
    """
    _, y_gravity, _ = gravity_align_point(point, pitch_rad, roll_rad)
    return float(camera_height_m) - float(y_gravity)


# ── plumbing helpers ─────────────────────────────────────────────────────────

def output_topic_for(depth_topic: str) -> str:
    """The card's result topic, derived from the depth topic.

    `/{ns}/camera/depth` -> `/{ns}/height/last`. Deriving rather than
    hardcoding keeps two robots on one DDS domain from overwriting each
    other's results; the fallback is the perception default used by the other
    topic-less cards.
    """
    topic = str(depth_topic or "")
    for suffix in ("/camera/depth", "/visual_depth", "/depth"):
        if topic.endswith(suffix):
            root = topic[: -len(suffix)]
            return f"{root}/height/last" if root else "/perception/height/last"
    return "/perception/height/last"


def median_and_spread(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        raise GeometryError("insufficient_frames", "no valid samples")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    return median, ordered[-1] - ordered[0]


def plausible_height(height_m: float) -> bool:
    return MIN_HEIGHT_M <= float(height_m) <= MAX_HEIGHT_M
